"""Read-only file tools for the chat model: list, find, search and read files, only inside the folders
granted in [tools] read (or with --read). Nothing here writes, deletes or runs anything."""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from . import config as C

MAX_RESULT = 24000          # characters per tool result (~6k tokens)
READ_LINES = 400            # default lines per read_file call
LIST_MAX = 300
MATCH_MAX = 100

SCHEMAS = [
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List a folder: subfolders end with '/', files show their size. Use depth 2-3 to see a "
                       "project's layout in one call. Paths may be absolute, start with ~, or be relative.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "folder to list"},
            "depth": {"type": "integer", "description": "levels to descend, 1-3 (default 1)"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file with line numbers. Long files come in pages: pass offset (first line, "
                       "1-based) and limit (lines) to read further.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "first line to read (default 1)"},
            "limit": {"type": "integer", "description": f"number of lines (default {READ_LINES})"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "search_text",
        "description": "Search file contents for a regular expression (case-insensitive unless it has capitals). "
                       "Returns path:line: text for each match.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "regular expression"},
            "path": {"type": "string", "description": "folder or file to search (default: the first granted folder)"},
            "glob": {"type": "string", "description": "only files matching this, e.g. *.lua"}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "find_files",
        "description": "Find files whose path matches a glob, e.g. *.md or **/server/*.lua.",
        "parameters": {"type": "object", "properties": {
            "glob": {"type": "string"},
            "path": {"type": "string", "description": "folder to look in (default: the first granted folder)"}},
            "required": ["glob"]}}},
]


class Denied(Exception):
    pass


class Files:
    """The granted folders and the rules for what inside them may be read."""

    def __init__(self, roots: list[str], cfg: dict | None = None):
        cfg = cfg or C.load()
        t = cfg["tools"]
        self.roots = []
        for r in roots:
            p = Path(os.path.realpath(C.expand(r)))
            if p.is_dir() and p not in self.roots:
                self.roots.append(p)
        self.deny = [str(g) for g in t.get("deny", [])]
        self.skip_dirs = set(cfg["search"]["exclude_dirs"]) | set(t.get("skip_dirs", []))
        self.home = Path(os.path.realpath(os.path.expanduser("~")))

    # -- paths
    def show(self, p: Path) -> str:
        try:
            return "~/" + str(p.relative_to(self.home)) if p != self.home else "~"
        except ValueError:
            return str(p)

    def resolve(self, path: str | None) -> Path:
        if not self.roots:
            raise Denied("no folders are granted: add them to [tools] read in ~/.config/npukit/config.toml")
        raw = (path or "").strip() or str(self.roots[0])
        p = C.expand(raw)
        if not p.is_absolute():
            p = self.roots[0] / p
        real = Path(os.path.realpath(p))
        if not any(real == r or r in real.parents for r in self.roots):
            raise Denied(f"{raw} is outside the folders you may read ({', '.join(map(self.show, self.roots))})")
        if self.denied(real):
            raise Denied(f"{self.show(real)} is on the deny list ([tools] deny)")
        return real

    def denied(self, p: Path) -> bool:
        parts = p.parts
        for g in self.deny:
            if g.endswith("/"):  # a folder name anywhere in the path
                if g[:-1] in parts:
                    return True
            elif fnmatch.fnmatch(p.name, g):
                return True
        return False

    def _walk(self, top: Path, depth: int | None = None):
        """Files under top (skip_dirs and denied entries left out), sorted, as Paths."""
        base = len(top.parts)
        for d, dirs, files in os.walk(top):
            here = Path(d)
            dirs[:] = sorted(x for x in dirs if x not in self.skip_dirs and not self.denied(here / x))
            if depth is not None and len(here.parts) - base >= depth:
                dirs[:] = []
            for f in sorted(files):
                fp = here / f
                if not self.denied(fp):
                    yield fp

    # -- tools
    def list_dir(self, path: str = "", depth: int = 1) -> str:
        top = self.resolve(path)
        if not top.is_dir():
            raise Denied(f"{self.show(top)} is not a folder")
        depth = max(1, min(int(depth or 1), 3))
        out, n = [self.show(top) + "/"], 0

        def walk(d: Path, level: int):
            nonlocal n
            try:
                entries = sorted(d.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
            except OSError as e:
                out.append("  " * level + f"(unreadable: {e.strerror})")
                return
            for e in entries:
                if n >= LIST_MAX or self.denied(e):
                    continue
                n += 1
                pad = "  " * level
                if e.is_dir():
                    skipped = e.name in self.skip_dirs
                    out.append(f"{pad}{e.name}/" + ("  (not descended)" if skipped and level < depth else ""))
                    if level < depth and not skipped:
                        walk(e, level + 1)
                else:
                    try:
                        size = e.stat().st_size
                    except OSError:
                        size = 0
                    out.append(f"{pad}{e.name}  {_size(size)}")

        walk(top, 1)
        if n >= LIST_MAX:
            out.append(f"… stopped at {LIST_MAX} entries: list a subfolder for more")
        return "\n".join(out)

    def read_file(self, path: str, offset: int = 1, limit: int = READ_LINES) -> str:
        p = self.resolve(path)
        if p.is_dir():
            raise Denied(f"{self.show(p)} is a folder: use list_dir")
        with open(p, "rb") as f:
            head = f.read(8192)
        if b"\0" in head:
            return f"{self.show(p)} is a binary file ({_size(p.stat().st_size)}); not shown"
        lines = p.read_text(errors="replace").splitlines()
        start = max(1, int(offset or 1))
        count = max(1, min(int(limit or READ_LINES), 2000))
        chunk = lines[start - 1:start - 1 + count]
        body, used = [], 0
        for i, ln in enumerate(chunk, start):
            row = f"{i:>5}  {ln[:500]}"
            if used + len(row) > MAX_RESULT:
                break
            body.append(row)
            used += len(row) + 1
        last = start + len(body) - 1
        more = f"\n… lines {last + 1}-{len(lines)} not shown: read_file with offset={last + 1}" \
            if last < len(lines) else ""
        return f"{self.show(p)} · lines {start}-{last} of {len(lines)}\n" + "\n".join(body) + more

    def search_text(self, pattern: str, path: str = "", glob: str = "") -> str:
        top = self.resolve(path)
        try:
            rx = re.compile(pattern, 0 if any(c.isupper() for c in pattern) else re.I)
        except re.error as e:
            raise Denied(f"bad regular expression: {e}")
        hits = self._rg(pattern, top, glob) if shutil.which("rg") else None
        if hits is None:
            hits = []
            files = [top] if top.is_file() else self._walk(top)
            for fp in files:
                if glob and not (fnmatch.fnmatch(fp.name, glob) or fnmatch.fnmatch(str(fp), glob)):
                    continue
                try:
                    if fp.stat().st_size > 2_000_000:
                        continue
                    with open(fp, errors="replace") as f:
                        for i, ln in enumerate(f, 1):
                            if rx.search(ln):
                                hits.append((fp, i, ln.rstrip("\n")))
                                if len(hits) > MATCH_MAX:
                                    break
                except (OSError, UnicodeError):
                    continue
                if len(hits) > MATCH_MAX:
                    break
        hits = [h for h in hits if not self.denied(h[0])]
        if not hits:
            return f"no matches for /{pattern}/ in {self.show(top)}"
        rows = [f"{self.show(fp)}:{i}: {text.strip()[:300]}" for fp, i, text in hits[:MATCH_MAX]]
        if len(hits) > MATCH_MAX:
            rows.append(f"… more than {MATCH_MAX} matches: narrow the pattern, path or glob")
        return _cap("\n".join(rows))

    def _rg(self, pattern: str, top: Path, glob: str):
        args = ["rg", "--json", "-S", "-m", "20", "--max-filesize", "2M"]
        for d in sorted(self.skip_dirs):
            args += ["-g", f"!{d}/"]
        if glob:
            args += ["-g", glob]
        try:
            r = subprocess.run(args + ["-e", pattern, str(top)], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if r.returncode not in (0, 1):
            return None
        hits = []
        for ln in r.stdout.splitlines():
            ev = json.loads(ln)
            if ev.get("type") != "match":
                continue
            d = ev["data"]
            text = d["lines"].get("text") or ""
            hits.append((Path(d["path"].get("text", "")), d["line_number"], text.rstrip("\n")))
            if len(hits) > MATCH_MAX:
                break
        return hits

    def find_files(self, glob: str, path: str = "") -> str:
        top = self.resolve(path)
        pat = glob.strip() or "*"
        found = []
        for fp in self._walk(top):
            rel = str(fp.relative_to(top))
            if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(fp.name, pat) or \
                    (pat.startswith("**/") and fnmatch.fnmatch(rel, pat[3:])):
                found.append(rel)
                if len(found) > LIST_MAX:
                    break
        if not found:
            return f"no files matching {pat} in {self.show(top)}"
        more = f"\n… more than {LIST_MAX}: narrow the glob" if len(found) > LIST_MAX else ""
        return _cap(f"{self.show(top)}/\n" + "\n".join(found[:LIST_MAX]) + more)

    def run(self, name: str, args: dict) -> str:
        fn = {"list_dir": self.list_dir, "read_file": self.read_file,
              "search_text": self.search_text, "find_files": self.find_files}.get(name)
        if not fn:
            return f"error: no tool called {name}"
        try:
            return _cap(fn(**{k: v for k, v in (args or {}).items() if v is not None}))
        except Denied as e:
            return f"error: {e}"
        except TypeError as e:
            return f"error: bad arguments for {name}: {e}"
        except OSError as e:
            return f"error: {e.strerror or e}"


def _size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def _cap(text: str) -> str:
    return text if len(text) <= MAX_RESULT else text[:MAX_RESULT] + "\n… cut here (result too long)"
