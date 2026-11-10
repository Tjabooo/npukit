"""`npu translate`: an eww overlay to translate a message you write.

Pick a language (recent ones first, or type any language), write the message, press Enter: the
translation shows up and lands on the clipboard. Enter again on the same text closes the overlay and
types the translation into the window you were in (wtype), so it works as a "write in my language,
send in theirs" box for chats.

The overlay runs its own eww daemon (config written to $XDG_RUNTIME_DIR/npukit/eww, colours from
[translate.colors]), so it needs no changes to an existing eww setup. Each action the overlay takes
(`--run`, `--draft`, `--lang`, `--chip`) is a short-lived `npu translate` process that updates a small
state file and pushes it to eww."""

from __future__ import annotations

import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from importlib import resources
from pathlib import Path

from . import client
from . import config as C
from . import ocr
from . import ui

WINDOW = "npu-translate"
# text typed into the overlay reaches the shell through a quoted here-document: eww pastes the
# input's value into the command unescaped, so it must never be part of the command line itself
HEREDOC_END = "NPUKIT_TRANSLATE_END_7f3a9c"
MAX_CHIPS = 8
PALETTE = ("bg", "panel", "line", "fg", "dim", "accent", "accent2", "error")


# ── state ────────────────────────────────────────────────────────────────────

def _state_file() -> Path:
    return C.runtime_dir() / "translate.json"


def _draft_file() -> Path:
    return C.runtime_dir() / "translate-draft.txt"


@contextmanager
def _locked():
    with open(C.runtime_dir() / "translate.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def load_state() -> dict:
    try:
        st = json.loads(_state_file().read_text())
        return st if isinstance(st, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(st: dict) -> None:
    f = _state_file()
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False))
    tmp.replace(f)


def chips(recent: list[str], configured: list[str], current: str, limit: int = MAX_CHIPS) -> list[str]:
    """Languages offered as one-click chips: the current one, recent ones, then the configured list."""
    out: list[str] = []
    for lang in [current, *recent, *configured]:
        lang = ocr.language(lang) if lang else ""
        if lang and lang.lower() not in (x.lower() for x in out):
            out.append(lang)
    return out[:limit]


def fresh_state(cfg: dict, old: dict) -> dict:
    t = cfg["translate"]
    lang = old.get("lang") or (ocr.recent_languages() or [""])[0] or ocr.language(
        t.get("to") or cfg["ocr"]["translate_to"])
    return {"open": True, "lang": lang, "chips": chips(ocr.recent_languages(), list(t["languages"]), lang),
            "busy": False, "seq": int(old.get("seq", 0)), "src": "", "src_lang": "", "result": "",
            "status": "", "error": False, "engine": C.engine_label(cfg)}


def view(st: dict) -> dict:
    """What the yuck reads (keeps the variable small and free of internals)."""
    lang = st.get("lang", "")
    return {"open": bool(st.get("open")), "lang": lang, "busy": bool(st.get("busy")),
            "chips": [{"i": i, "name": c, "on": c.lower() == lang.lower()}
                      for i, c in enumerate(st.get("chips", []))],
            "result": st.get("result", ""), "status": st.get("status", ""),
            "error": bool(st.get("error")), "engine": st.get("engine", "")}


def wants_send(st: dict, text: str) -> bool:
    """Enter on text that is already translated (same text, same language) = send it."""
    return bool(st.get("result") and not st.get("busy") and not st.get("error")
                and st.get("src") == text and st.get("src_lang", "").lower() == st.get("lang", "").lower())


# ── eww ──────────────────────────────────────────────────────────────────────

def _eww_dir() -> Path:
    return C.runtime_dir() / "eww"


def eww_cmd(cfg: dict, *args: str) -> list[str]:
    return [cfg["translate"]["eww"], "-c", str(_eww_dir()), *args]


def _eww(cfg: dict, *args: str, timeout: float = 5) -> tuple[int, str]:
    try:
        r = subprocess.run(eww_cmd(cfg, *args), capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, str(e)
    return r.returncode, r.stdout.strip()


def npu_cmd() -> str:
    """How the overlay calls back into npukit (same interpreter and code as this process)."""
    root = Path(__file__).resolve().parent.parent
    return f"PYTHONPATH={shlex.quote(str(root))} {shlex.quote(sys.executable)} -m npukit translate"


def heredoc_action(npu: str, action: str) -> str:
    """Shell for an eww input handler: the input's value ({}) goes in on stdin, never on the command line.
    Backgrounded because eww kills handlers that outlive their :timeout."""
    return f"{npu} {action} <<'{HEREDOC_END}' >/dev/null 2>&1 &\n{{}}\n{HEREDOC_END}"


def draft_action(path: Path) -> str:
    """Every keystroke saves the draft (chips translate it): plain sh, no Python start per key."""
    f = shlex.quote(str(path))
    tmp = shlex.quote(str(path) + ".tmp")
    return f"cat > {tmp} <<'{HEREDOC_END}' && mv -f {tmp} {f}\n{{}}\n{HEREDOC_END}"


def _yuck_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_yuck(cfg: dict) -> str:
    npu = npu_cmd()
    tpl = resources.files("npukit").joinpath("eww/translate.yuck").read_text()
    subs = {
        "@ON_ACCEPT@": _yuck_str(heredoc_action(npu, "--run")),
        "@ON_CHANGE@": _yuck_str(draft_action(_draft_file())),
        "@ON_LANG@": _yuck_str(heredoc_action(npu, "--lang")),
        "@ON_CHIP@": _yuck_str(f"{npu} --chip ${{c.i}} >/dev/null 2>&1 &"),
        "@ON_CLOSE@": _yuck_str(f"{npu} --close >/dev/null 2>&1 &"),
        "@INITIAL@": "'" + json.dumps(view({}), ensure_ascii=False) + "'",
    }
    for k, v in subs.items():
        tpl = tpl.replace(k, v)
    return tpl


def render_scss(cfg: dict) -> str:
    t = cfg["translate"]
    colors = t["colors"]
    head = [f"${k}: {colors[k]};" for k in PALETTE if k in colors]
    head.append(f'$font: "{t["font"]}", monospace;')
    head.append(f"$width: {int(t['width'])}px;")
    return "\n".join(head) + "\n" + resources.files("npukit").joinpath("eww/translate.scss").read_text()


def write_eww_config(cfg: dict) -> bool:
    """Write the overlay's eww config; True when it changed (a running daemon then reloads it)."""
    d = _eww_dir()
    d.mkdir(parents=True, exist_ok=True)
    changed = False
    for name, text in (("eww.yuck", render_yuck(cfg)), ("eww.scss", render_scss(cfg))):
        f = d / name
        if not f.exists() or f.read_text() != text:  # eww reloads on change: don't touch it otherwise
            f.write_text(text)
            changed = True
    return changed


def push(cfg: dict, st: dict) -> None:
    _eww(cfg, "update", "tr=" + json.dumps(view(st), ensure_ascii=False))


def _ensure_daemon(cfg: dict) -> bool:
    if _eww(cfg, "ping", timeout=2)[1] == "pong":
        return True
    subprocess.Popen(eww_cmd(cfg, "daemon"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    for _ in range(50):
        time.sleep(0.1)
        if _eww(cfg, "ping", timeout=2)[1] == "pong":
            return True
    return False


def is_open(cfg: dict) -> bool:
    code, out = _eww(cfg, "active-windows", timeout=2)
    return code == 0 and any(line.split(":")[0].strip() == WINDOW for line in out.splitlines())


def _focused_monitor() -> list[str]:
    try:
        out = subprocess.run(["hyprctl", "monitors", "-j"], capture_output=True, text=True, timeout=2,
                             check=False).stdout
        m = next((m for m in json.loads(out or "[]") if m.get("focused")), None)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return []
    return [x for x in (m.get("name"), m.get("model")) if x] if m else []


def _hook(cfg: dict, key: str) -> None:
    cmd = list(cfg["translate"].get(key) or [])
    if cmd and shutil.which(cmd[0]):
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)


# ── actions ──────────────────────────────────────────────────────────────────

def open_overlay(cfg: dict) -> int:
    if not shutil.which(cfg["translate"]["eww"]):
        ui.notify("Translate", "The translate overlay needs eww (e.g. pacman -S eww)", urgency="critical")
        return 1
    changed = write_eww_config(cfg)
    if not _ensure_daemon(cfg):
        ui.notify("Translate", f"eww didn't start; try: {' '.join(eww_cmd(cfg, 'logs'))}", urgency="critical")
        return 1
    with _locked():
        st = fresh_state(cfg, load_state())
        save_state(st)
    _draft_file().unlink(missing_ok=True)
    if changed:
        time.sleep(0.4)  # a reload resets the variable: let it finish before the state goes in
    push(cfg, st)
    for mon in [*_focused_monitor(), None]:
        if _eww(cfg, "open", WINDOW, *(["--screen", mon] if mon else []))[0] == 0:
            _hook(cfg, "on_open")
            return 0
    ui.notify("Translate", f"eww couldn't open its window; try: {' '.join(eww_cmd(cfg, 'logs'))}",
              urgency="critical")
    return 1


def close_overlay(cfg: dict) -> None:
    with _locked():
        st = load_state()
        st["open"] = False
        st["seq"] = int(st.get("seq", 0)) + 1  # a translation still running is no longer shown
        save_state(st)
    _eww(cfg, "close", WINDOW)
    _hook(cfg, "on_close")


def send(cfg: dict, text: str) -> int:
    """Close, give the focus back to the window underneath, type the translation there."""
    close_overlay(cfg)
    if ui.copy(text) and not cfg["translate"]["type"]:
        return 0
    time.sleep(float(cfg["translate"]["type_delay"]))
    if not ui.type_text(text, list(cfg["translate"]["type"])):
        ui.notify("Translation copied", "Paste it with Ctrl+V" + (
            "" if shutil.which("wtype") else " (install wtype to have it typed)"), timeout_ms=3000)
    return 0


def run(cfg: dict, text: str, lang: str | None = None) -> int:
    """Translate `text` into the chosen language and show it; the same text again = send it."""
    text = text.strip()
    with _locked():
        st = load_state()
        if lang:
            same = (st.get("result") and st.get("src") == text
                    and st.get("src_lang", "").lower() == lang.lower())
            st["lang"] = lang
            have = st.get("chips", [])
            if lang.lower() not in (c.lower() for c in have):  # a typed language joins the chips up front
                st["chips"] = chips([], have, lang)
            if same:  # already showing this translation
                text = ""
        if not text:
            save_state(st)
            push(cfg, st)
            return 0
        if lang is None and wants_send(st, text):
            result = st["result"]
            save_state(st)
            send_it = True
        else:
            send_it = False
            st["seq"] = seq = int(st.get("seq", 0)) + 1
            st.update(busy=True, error=False, status=f"translating into {st['lang']}…", src=text,
                      src_lang=st["lang"], result="")
            save_state(st)
            target = st["lang"]
    if send_it:
        return send(cfg, result)
    push(cfg, st)

    t = cfg["translate"]
    model = t.get("model") or cfg["ocr"]["translate_model"]
    err = None
    try:
        client.call("/npu/activity", {"activity": "translate"}, timeout=5)
    except client.ApiError:
        pass
    try:
        out, took = ocr.translate(text, target, cfg["translate"]["prompt"], model)
    except client.ApiError as e:
        out, took, err = "", 0.0, str(e)
    finally:
        try:
            client.call("/npu/activity", {"activity": None}, timeout=5)
        except client.ApiError:
            pass

    with _locked():
        st = load_state()
        if int(st.get("seq", 0)) != seq:
            return 0  # superseded (language changed, new text, or closed)
        if err or not out:
            st.update(busy=False, error=True, result="", status=err or "the model returned nothing")
        else:
            copied = bool(t["copy"]) and ui.copy(out)
            ocr.remember_language(target)
            st.update(busy=False, error=False, result=out,
                      status=f"{took:.1f}s on {st.get('engine') or C.engine_label(cfg)}"
                             + (" · copied" if copied else "")
                             + (" · Enter again types it" if t["type"] else ""))
        save_state(st)
    push(cfg, st)
    return 0 if not err else 1


def _stdin_text() -> str:
    data = sys.stdin.read() if not sys.stdin.isatty() else ""
    return data[:-1] if data.endswith("\n") else data


def _draft() -> str:
    try:
        return _draft_file().read_text()
    except OSError:
        return ""


def main(args) -> int:
    cfg = C.load()
    if args.close:
        close_overlay(cfg)
        return 0
    if args.draft:
        _draft_file().write_text(_stdin_text())
        return 0
    if args.run:
        text = _stdin_text()
        _draft_file().write_text(text)
        return run(cfg, text)
    if args.lang:
        lang = ocr.language(_stdin_text())
        return run(cfg, _draft(), lang) if lang else 0
    if args.chip is not None:
        names = load_state().get("chips", [])
        if 0 <= args.chip < len(names):
            return run(cfg, _draft(), names[args.chip])
        return 1
    if args.text:  # one-shot, no overlay: npu translate -t sv "hello"
        lang = ocr.language(args.to or cfg["translate"].get("to") or cfg["ocr"]["translate_to"])
        model = cfg["translate"].get("model") or cfg["ocr"]["translate_model"]
        try:
            out, took = ocr.translate(" ".join(args.text), lang, cfg["translate"]["prompt"], model)
        except client.ApiError as e:
            print(f"translate failed: {e}", file=sys.stderr)
            return 1
        print(out)
        print(f"[{took:.1f}s → {lang}]", file=sys.stderr)
        return 0
    if is_open(cfg):  # the bind toggles
        close_overlay(cfg)
        return 0
    return open_overlay(cfg)
