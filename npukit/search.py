"""Semantic search over notes and code with NPU embeddings (EmbeddingGemma via FastFlowLM).

`npu index` walks the configured roots (git repos: tracked + untracked-but-not-ignored files;
other dirs: a plain walk with excluded dirs), re-embeds only files whose size or mtime changed,
and keeps everything in one SQLite file. `npu search` embeds the query and ranks chunks by
cosine similarity (numpy when available, plain Python otherwise).
"""

from __future__ import annotations

import array
import fnmatch
import math
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from . import client
from . import config as C
from . import ui

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, size INTEGER, mtime REAL);
CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, path TEXT, start INTEGER, end INTEGER,
                                   text TEXT, vec BLOB);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
"""


def db(cfg) -> sqlite3.Connection:
    p = C.expand(cfg["search"]["db"])
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.executescript(SCHEMA)
    return con


def _git_files(repo: Path) -> list[Path] | None:
    try:
        r = subprocess.run(["git", "-C", str(repo), "ls-files", "-z", "--cached", "--others",
                            "--exclude-standard"], capture_output=True, check=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return [repo / f for f in r.stdout.decode("utf-8", "replace").split("\0") if f]


def candidates(cfg) -> list[Path]:
    s = cfg["search"]
    excl = set(s["exclude_dirs"])
    exts = {e.lower() for e in s["extensions"]}
    names = set(s["names"])
    max_bytes = int(s["max_file_kb"]) * 1024
    globs = [os.path.expanduser(g) for g in s.get("exclude_globs", [])]
    out: list[Path] = []

    def ok(p: Path) -> bool:
        if p.suffix.lower() not in exts and p.name not in names:
            return False
        if any(part in excl for part in p.parts):
            return False
        if any(fnmatch.fnmatch(str(p), g) for g in globs):
            return False
        try:
            st = p.stat()
        except OSError:
            return False
        return 0 < st.st_size <= max_bytes and p.is_file()

    def walk(d: Path):
        try:
            entries = list(os.scandir(d))
        except OSError:
            return
        if any(e.name == ".git" for e in entries):
            files = _git_files(d)
            if files is not None:
                out.extend(p for p in files if ok(p))
                return
        for e in entries:
            if e.is_dir(follow_symlinks=False):
                if e.name not in excl and not e.name.startswith("."):
                    walk(Path(e.path))
            elif e.is_file(follow_symlinks=False) and ok(Path(e.path)):
                out.append(Path(e.path))

    for root in s["roots"]:
        r = C.expand(root)
        if r.is_dir():
            walk(r)
    return sorted(set(out))


def chunk(text: str, n: int, overlap: int) -> list[tuple[int, int, str]]:
    lines = text.splitlines()
    out, i = [], 0
    while i < len(lines):
        j = min(len(lines), i + n)
        body = "\n".join(lines[i:j]).strip()
        if len(body) > 2400:  # long lines: cut by characters (the model sees ~2k tokens max)
            body = body[:2400]
        if body:
            out.append((i + 1, j, body))
        if j == len(lines):
            break
        i = j - overlap
    return out


def _pack(v) -> bytes:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return array.array("f", (x / n for x in v)).tobytes()


def index(args) -> int:
    cfg = C.load()
    s = cfg["search"]
    con = db(cfg)
    t0 = time.monotonic()
    files = candidates(cfg)
    known = {p: (sz, mt) for p, sz, mt in con.execute("SELECT path, size, mtime FROM files")}
    want = set()
    todo = []
    for p in files:
        st = p.stat()
        key = str(p)
        want.add(key)
        if args.full or known.get(key) != (st.st_size, st.st_mtime):
            todo.append((p, st))
    gone = [k for k in known if k not in want]
    for k in gone:
        con.execute("DELETE FROM chunks WHERE path=?", (k,))
        con.execute("DELETE FROM files WHERE path=?", (k,))
    con.commit()
    if not todo:
        print(f"index up to date: {len(files)} files, {len(gone)} removed ({time.monotonic() - t0:.1f}s)")
        return 0
    if not args.quiet:
        print(f"{len(todo)} changed files of {len(files)}")
    batch, pending, n_chunks = int(s["batch"]), [], 0
    done_files: list[tuple[str, os.stat_result]] = []  # all chunks queued; recorded after the next flush
    home = os.path.expanduser("~")

    def flush():
        # batches span files: FastFlowLM has ~1.5 s fixed cost per request, so full batches matter
        nonlocal n_chunks
        if pending:
            texts = [f"title: {path.replace(home, '~')} | text: {body}" for path, _, _, body in pending]
            try:
                rows = list(zip(pending, client.embed(texts)))
            except client.ApiError as e:  # one bad input fails the batch: retry one by one, skip the bad
                if "not reachable" in str(e):
                    raise
                rows = []
                for item, t in zip(pending, texts):
                    try:
                        rows.append((item, client.embed([t])[0]))
                    except client.ApiError as e1:
                        print(f"\nskipped {item[0]}:{item[1]}: {e1}", file=sys.stderr)
            con.executemany("INSERT INTO chunks(path, start, end, text, vec) VALUES (?,?,?,?,?)",
                            [(p, a, b, body, _pack(v)) for (p, a, b, body), v in rows])
            n_chunks += len(rows)
            pending.clear()
        con.executemany("INSERT OR REPLACE INTO files(path, size, mtime) VALUES (?,?,?)",
                        [(k, st.st_size, st.st_mtime) for k, st in done_files])
        done_files.clear()
        con.commit()

    try:
        client.call("/npu/activity", {"activity": "index"}, timeout=5)
        for i, (p, st) in enumerate(todo):
            try:
                raw = p.read_bytes()
            except OSError:
                continue
            text = "" if b"\0" in raw[:4096] else raw.decode("utf-8", "replace")
            key = str(p)
            con.execute("DELETE FROM chunks WHERE path=?", (key,))
            parts = chunk(text, int(s["chunk_lines"]), int(s["overlap_lines"]))
            for a, b, body in parts[: int(s["max_chunks_per_file"])]:
                pending.append((key, a, b, body))
                if len(pending) >= batch:
                    flush()
            done_files.append((key, st))
            if not args.quiet and i % 20 == 0:
                print(f"\r{i + 1}/{len(todo)} files, {n_chunks} chunks", end="", file=sys.stderr)
        flush()
        con.commit()
    finally:
        try:
            client.call("/npu/activity", {"activity": None}, timeout=5)
            client.call("/npu/release", {"component": "embed"}, timeout=5)
        except client.ApiError:
            pass
    dt = time.monotonic() - t0
    print(f"\nindexed {len(todo)} files → {n_chunks} chunks in {dt:.1f}s "
          f"({n_chunks / dt:.1f} chunks/s), {len(gone)} removed")
    return 0


def query(q: str, k: int, cfg) -> list[tuple[float, str, int, int, str]]:
    con = db(cfg)
    rows = con.execute("SELECT path, start, end, text, vec FROM chunks").fetchall()
    if not rows:
        return []
    qv = array.array("f")
    qv.frombytes(_pack(client.embed([f"task: search result | query: {q}"])[0]))
    if np is not None:
        mat = np.frombuffer(b"".join(r[4] for r in rows), dtype=np.float32).reshape(len(rows), -1)
        scores = mat @ np.frombuffer(qv.tobytes(), dtype=np.float32)
        order = np.argsort(-scores)
        ranked = [(float(scores[i]), rows[i]) for i in order[: k * 4]]
    else:
        def dot(b):
            v = array.array("f")
            v.frombytes(b)
            return sum(x * y for x, y in zip(v, qv))
        ranked = sorted(((dot(r[4]), r) for r in rows), key=lambda t: -t[0])[: k * 4]
    out, seen = [], {}
    for score, (path, a, b, text, _) in ranked:  # at most 2 hits per file
        if seen.get(path, 0) >= 2:
            continue
        seen[path] = seen.get(path, 0) + 1
        out.append((score, path, a, b, text))
        if len(out) >= k:
            break
    return out


def _open(cfg, path: str, line: int):
    cmd = [a.replace("{path}", path).replace("{line}", str(line)) for a in cfg["search"]["open"]]
    subprocess.Popen(cmd, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def search(args) -> int:
    cfg = C.load()
    k = args.n or int(cfg["search"]["results"])
    home = os.path.expanduser("~")
    q = " ".join(args.query).strip()
    if args.rofi:
        code, q = ui.rofi([], "search", theme=args.theme, filter_input=True,
                          mesg="semantic search · notes + code · runs on the NPU")
        if code != 0 or not q:
            return 1
    if not q:
        print("usage: npu search QUERY", file=sys.stderr)
        return 2
    t0 = time.monotonic()
    try:
        hits = query(q, k, cfg)
    except client.ApiError as e:
        (ui.notify("Search failed", str(e), urgency="critical") if args.rofi else print(e, file=sys.stderr))
        return 1
    dt = time.monotonic() - t0
    if args.rofi:
        lines = []
        for score, path, a, b, text in hits:
            snippet = " ".join(text.split())[:90].replace("&", "&amp;").replace("<", "&lt;")
            lines.append(f"{path.replace(home, '~')}:{a}  <span alpha='55%'>{score:.2f}  {snippet}</span>")
        r = subprocess.run(["rofi", "-dmenu", "-i", "-markup-rows", "-p", q[:30], "-format", "i",
                            "-mesg", f"{len(hits)} results · {dt:.2f}s"] +
                           (["-theme", args.theme] if args.theme else []),
                           input="\n".join(lines).encode(), capture_output=True, check=False)
        if r.returncode == 0 and r.stdout.strip():
            _, path, a, _, _ = hits[int(r.stdout.strip())]
            _open(cfg, path, a)
        return 0
    for score, path, a, b, text in hits:
        print(f"\033[1m{score:.3f}  {path.replace(home, '~')}:{a}-{b}\033[0m")
        snippet = "\n".join(text.splitlines()[:4])
        print("   " + snippet.replace("\n", "\n   ")[:400])
    print(f"[{len(hits)} hits in {dt:.2f}s]", file=sys.stderr)
    return 0


def stats() -> dict:
    cfg = C.load()
    p = C.expand(cfg["search"]["db"])
    if not p.exists():
        return {"files": 0, "chunks": 0}
    con = db(cfg)
    return {"files": con.execute("SELECT count(*) FROM files").fetchone()[0],
            "chunks": con.execute("SELECT count(*) FROM chunks").fetchone()[0],
            "db_mb": round(p.stat().st_size / 1e6, 1), "updated": p.stat().st_mtime}
