"""`npu ask` / `npu chat` with file access: the model calls the read-only tools in npukit.tools in a loop
until it has what it needs, then answers. Without granted folders it is a plain chat."""

from __future__ import annotations

import json
import socket
import sys
import time

from . import client
from . import config as C
from .tools import SCHEMAS, Files

TRIMMED = "[older tool result removed to save context; call the tool again if you still need it]"


def _note(text: str) -> None:
    if sys.stderr.isatty():
        text = f"\x1b[2m{text}\x1b[0m"
    print(text, file=sys.stderr, flush=True)


def _brief(args: dict) -> str:
    parts = []
    for k in ("path", "pattern", "glob", "offset", "depth"):
        if args.get(k) not in (None, "", 1) or (k == "path" and args.get(k)):
            parts.append(f"{args[k]}" if k in ("path", "pattern", "glob") else f"{k}={args[k]}")
    return " ".join(parts)[:120]


class Session:
    def __init__(self, model: str | None, files: Files, *, think: bool = False, max_tokens: int = 4096,
                 cfg: dict | None = None, quiet: bool = False):
        self.cfg = cfg or C.load()
        self.t = self.cfg["tools"]
        self.files = files
        self.model = model
        self.think = think
        self.max_tokens = max_tokens
        self.quiet = quiet
        self.tools_ok = True
        self.messages: list[dict] = [{"role": "system", "content": self.system_prompt()}]
        self.tokens = 0
        self.last_tps: float | None = None

    def system_prompt(self) -> str:
        host = socket.gethostname()
        date = time.strftime("%Y-%m-%d")
        if not self.files.roots:
            return f"You are a helpful assistant running locally on {host}. Today is {date}."
        roots = ", ".join(self.files.show(r) for r in self.files.roots)
        return (f"You are a helpful assistant running locally on {host}. Today is {date}. You can read files "
                f"(never change them) inside: {roots}. When a question is about files, folders or projects, look "
                "with the tools first and base the answer on what you actually read; mention the files you used. "
                "For a project: list_dir with depth 2, then read the README and the few files that matter. Read "
                "only what you need. Paths can be absolute, start with ~, or be relative to the first folder. "
                "Answer in the language the user writes in.")

    def grant(self, path: str) -> str:
        before = list(self.files.roots)
        self.files = Files([*map(str, before), path], self.cfg)
        if len(self.files.roots) == len(before):
            return f"not a folder (or already granted): {path}"
        self.messages[0] = {"role": "system", "content": self.system_prompt()}
        return "can now read " + ", ".join(self.files.show(r) for r in self.files.roots)

    def _call(self, use_tools: bool) -> dict:
        extra = {"chat_template_kwargs": {"enable_thinking": self.think}}
        if use_tools:
            extra["tools"] = SCHEMAS
        return client.chat(self.messages, self.model, max_tokens=self.max_tokens, timeout=900, **extra)

    def _trim(self) -> None:
        """Keep tool results under [tools] context_chars: blank the oldest first."""
        budget = int(self.t.get("context_chars", 48000))
        tool_msgs = [m for m in self.messages if m["role"] == "tool"]
        total = sum(len(m["content"]) for m in tool_msgs)
        for m in tool_msgs[:-1]:
            if total <= budget:
                break
            if m["content"] != TRIMMED:
                total -= len(m["content"]) - len(TRIMMED)
                m["content"] = TRIMMED

    def ask(self, text: str) -> str:
        self.messages.append({"role": "user", "content": text})
        steps = int(self.t.get("max_steps", 16))
        for step in range(steps + 1):
            use_tools = bool(self.files.roots) and self.tools_ok and step < steps
            try:
                out = self._call(use_tools)
            except client.ApiError as e:
                if use_tools and "tool" in str(e).lower():  # server without tool calls (older FastFlowLM)
                    self.tools_ok = False
                    _note("  (this model server can't call tools: answering without file access)")
                    continue
                raise
            u = out.get("usage") or {}
            self.tokens += int(u.get("completion_tokens") or 0)
            self.last_tps = u.get("decoding_speed_tps") or (out.get("timings") or {}).get("predicted_per_second")
            msg = out["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            entry = {"role": "assistant", "content": msg.get("content") or ""}
            if calls:
                entry["tool_calls"] = calls
            self.messages.append(entry)
            if not calls:
                return (msg.get("content") or "").strip()
            for c in calls:
                fn = c.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                result = self.files.run(fn.get("name", ""), args if isinstance(args, dict) else {})
                if not self.quiet:
                    first = result.splitlines()[0] if result else ""
                    _note(f"  · {fn.get('name')} {_brief(args)}" + (f"  ({first[6:80]})" if first.startswith("error") else ""))
                self.messages.append({"role": "tool", "tool_call_id": c.get("id") or "", "content": result})
            self._trim()
        return ""


def build(args, cfg: dict | None = None) -> Session:
    cfg = cfg or C.load()
    t = cfg["tools"]
    roots = [] if getattr(args, "no_files", False) else [*t.get("read", []), *(getattr(args, "read", None) or [])]
    files = Files(roots, cfg)
    model = args.model or (t.get("model") if files.roots else None)
    return Session(model, files, think=args.think, max_tokens=args.max_tokens, cfg=cfg)


def _footer(s: Session, t0: float) -> None:
    tps = f" · {s.last_tps:.1f} tok/s" if s.last_tps else ""
    _note(f"[{time.monotonic() - t0:.1f}s · {s.tokens} tok{tps}]")


def ask_main(args) -> int:
    q = " ".join(args.prompt) or sys.stdin.read()
    s = build(args)
    t0 = time.monotonic()
    print(s.ask(q))
    _footer(s, t0)
    return 0


HELP = """/read PATH   let the model read another folder for this chat
/model NAME  switch model (fast, quality, vision or a tag)
/clear       forget the conversation
/exit        quit (also Ctrl-D)"""


def chat_main(args) -> int:
    s = build(args)
    where = ", ".join(s.files.show(r) for r in s.files.roots) or "no folders (add [tools] read or --read)"
    _note(f"npu chat · {s.model or C.load()['models']['default']} · can read: {where} · /help")
    first = " ".join(args.prompt).strip()
    interactive = sys.stdin.isatty()
    if interactive:
        try:
            import readline  # noqa: F401  (line editing + history for input())
        except ImportError:
            pass
    while True:
        if first:
            line, first = first, ""
        elif not interactive:
            return 0
        else:
            try:
                line = input("› ").strip()
            except (EOFError, KeyboardInterrupt):
                print(file=sys.stderr)
                return 0
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, rest = line.partition(" ")
            if cmd in ("/exit", "/quit", "/q"):
                return 0
            if cmd == "/clear":
                s.messages = s.messages[:1]
                s.tokens = 0
                _note("conversation cleared")
            elif cmd == "/read" and rest:
                _note(s.grant(rest.strip()))
            elif cmd == "/model" and rest:
                s.model = rest.strip()
                _note(f"model: {s.model}")
            else:
                _note(HELP)
            continue
        t0 = time.monotonic()
        try:
            print(s.ask(line), flush=True)
        except client.ApiError as e:
            _note(f"error: {e}")
            continue
        except KeyboardInterrupt:
            _note("(stopped)")
            continue
        _footer(s, t0)
