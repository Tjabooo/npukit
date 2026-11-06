"""`npu`: one command for the NPU gateway and the tools built on it."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

from . import __version__
from . import client
from . import config as C
from . import ui


def cmd_status(args) -> int:
    from . import hw, search
    try:
        st = client.call("/npu/status", timeout=5)
    except client.ApiError as e:
        st = {"state": "gateway down", "error": str(e), "npu": hw.npu_info(),
              "power": "ac" if hw.on_ac() else "battery"}
    if args.json:
        print(json.dumps(st, indent=1))
        return 0
    npu = st.get("npu", {})
    cfg = C.load()
    gpu = C.backend(cfg) == "llamacpp"
    print(f"gateway   {C.base_url()}  ·  {st['state']}" + (f" ({st['activity']})" if st.get("activity") else ""))
    if st.get("components"):
        comps = [st["chat_model"] if c == "chat" else {"asr": "whisper", "embed": "embeddings"}[c]
                 for c in st["components"]]
        since = time.time() - (st.get("loaded_since") or time.time())
        pids = ",".join(str(p["pid"]) for p in st.get("procs", [])) or st.get("flm_pid")
        print(f"loaded    {', '.join(comps)}  ·  " + (f"pmode {st['pmode']}  ·  " if st.get("pmode") else "")
              + f"up {since:.0f}s  ·  pid {pids}")
    else:
        print("loaded    nothing")
    if "idle_unload" in st:
        print(f"power     {st['power']}  ·  unload after {st['idle_unload']:.0f}s idle")
    if gpu:
        return _status_llamacpp(cfg, st)
    if npu.get("present"):
        print(f"npu       {npu['pci']}  fw {npu['firmware']}  ·  {npu['runtime']}  ·  "
              f"active {npu['active_ms'] / 1000:.0f}s since boot")
    else:
        print("npu       not found (amdxdna driver loaded?)")
    flm = os.path.expanduser(C.load()["server"]["flm"])
    try:
        out = subprocess.run([flm, "version"], capture_output=True, text=True, timeout=10).stdout
        ver = re.search(r"v?(\d+)\.(\d+)\.(\d+)", out)
        v = tuple(int(x) for x in ver.groups()) if ver else ()
        note = "  ⚠ older than 1.0.12: embeddings are wrong, update FastFlowLM" if v and v < (1, 0, 12) else ""
        print(f"flm       {'.'.join(map(str, v)) or '?'}  ({flm}){note}")
    except (OSError, subprocess.SubprocessError):
        print(f"flm       not found: {flm}")
    if st.get("last_tps"):
        print(f"last      {st['last_tps']:.1f} tok/s")
    if st.get("last_error"):
        print(f"error     {st['last_error']}")
    s = search.stats()
    if s["chunks"]:
        print(f"index     {s['files']} files · {s['chunks']} chunks · {s['db_mb']} MB · "
              f"updated {time.strftime('%Y-%m-%d %H:%M', time.localtime(s['updated']))}")
    return 0


def _status_llamacpp(cfg: dict, st: dict) -> int:
    import shutil
    from . import search
    lc = cfg["llamacpp"]
    for name, key in (("llama", "server"), ("whisper", "whisper")):
        path = shutil.which(os.path.expanduser(lc[key]))
        print(f"{name:<9} {path or 'not found: ' + lc[key]}")
    wm = C.expand(lc["whisper_model"])
    print(f"stt       {wm if wm.exists() else 'missing: ' + str(wm)}")
    print(f"runs on   {C.engine_label(cfg)} (llama.cpp / whisper.cpp)")
    if st.get("last_tps"):
        print(f"last      {st['last_tps']:.1f} tok/s")
    if st.get("last_error"):
        print(f"error     {st['last_error']}")
    s = search.stats()
    if s["chunks"]:
        print(f"index     {s['files']} files · {s['chunks']} chunks · {s['db_mb']} MB")
    return 0


def cmd_models(args) -> int:
    out = client.call("/v1/models", timeout=30)
    cfg = C.load()
    print("aliases:")
    for a, t in cfg["models"]["aliases"].items():
        print(f"  {a:<10} → {C.resolve_model(a, cfg)}")
    if C.backend(cfg) == "llamacpp":
        print(f"models (llama.cpp, {C.engine_label(cfg)}):")
        for m in out["data"]:
            if m.get("owned_by") == "llama.cpp":
                have = {True: f"{m.get('size_gb') or '?':>5} GB", False: "missing", None: "downloads on first use"}
                print(f"  {m['id']:<20} {have[m.get('present')]:<22} {m.get('source', '')}")
        print("add more under [llamacpp.models] in the config")
        return 0
    print("installed (FastFlowLM, NPU):")
    for m in out["data"]:
        if m.get("owned_by") == "FastFlowLM":
            print(f"  {m['id']:<24} {m.get('size_gb') or '':>5} GB  {','.join(m.get('labels') or [])}")
    print("more: flm list   ·   download: flm pull TAG")
    return 0


def cmd_load(args) -> int:
    comp = {"chat": "chat", "stt": "asr", "asr": "asr", "whisper": "asr", "embed": "embed"}.get(args.model)
    body = {"component": comp} if comp and comp != "chat" else {"component": "chat", "model": args.model}
    if args.pmode:
        body["pmode"] = args.pmode
    t0 = time.monotonic()
    st = client.call("/npu/load", body, timeout=600)
    print(f"loaded {st['components']} {st.get('chat_model') or ''} in {time.monotonic() - t0:.1f}s"
          + (f" (pmode {st['pmode']})" if st.get("pmode") else f" on {st.get('runs_on', 'GPU')}"))
    return 0


def cmd_stop(args) -> int:
    try:
        client.call("/npu/stop", {}, timeout=30)
    except client.ApiError as e:
        if "not reachable" in str(e):
            return 0  # nothing running
        print(e, file=sys.stderr)
        return 1
    print(f"{C.engine_label()} unloaded")
    return 0


def cmd_ask(args) -> int:
    q = " ".join(args.prompt) or sys.stdin.read()
    extra = {} if args.think else {"chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.monotonic()
    out = client.chat([{"role": "user", "content": q}], args.model, max_tokens=args.max_tokens, **extra)
    print(out["choices"][0]["message"].get("content", "").strip())
    u = out.get("usage", {})
    tps = u.get("decoding_speed_tps")
    print(f"[{time.monotonic() - t0:.1f}s · {u.get('completion_tokens', '?')} tok"
          + (f" · {tps:.1f} tok/s]" if tps else "]"), file=sys.stderr)
    return 0


def cmd_menu(args) -> int:
    items = [("Dictate (toggle)", ["dictate"]), ("Read text from region (OCR)", ["ocr"]),
             ("Translate region", ["ocr", "--pick"]),
             ("Describe region", ["ocr", "--describe"]), ("Search notes + code", ["search", "--rofi"]),
             ("Load fast model", ["load", "fast"]), ("Load quality model", ["load", "quality"]),
             ("Unload everything", ["stop"]), ("Update search index", ["index"])]
    cfg = C.load()
    theme = args.theme or (str(C.expand(cfg["ui"]["menu_theme"])) if cfg["ui"]["menu_theme"] else None)
    r = subprocess.run(["rofi", "-dmenu", "-i", "-p", "npu", "-format", "i"] +
                       (["-theme", theme] if theme else []),
                       input="\n".join(i for i, _ in items).encode(), capture_output=True, check=False)
    if r.returncode != 0 or not r.stdout.strip():
        return 1
    argv = items[int(r.stdout.strip())][1]
    if argv[0] == "ocr":  # region pick freezes the screen: let the menu vanish first
        ui.wait_menu_gone(float(cfg["ui"].get("menu_close_wait", 0.45)))
    return main(argv)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="npu", description="Everyday local AI on the AMD XDNA2 NPU (FastFlowLM), "
                                "or on a GPU with llama.cpp + whisper.cpp.")
    p.add_argument("--version", action="version", version=f"npukit {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status", help="gateway, loaded models, NPU and index state")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("models", help="aliases and installed NPU models")
    s.set_defaults(fn=cmd_models)

    s = sub.add_parser("load", help="load a model now: fast | quality | TAG | stt | embed")
    s.add_argument("model", nargs="?", default="fast")
    s.add_argument("--pmode", choices=["powersaver", "balanced", "performance", "turbo"])
    s.set_defaults(fn=cmd_load)

    s = sub.add_parser("stop", help="unload everything from the NPU")
    s.set_defaults(fn=cmd_stop)

    s = sub.add_parser("ask", help="one-shot question to a local model")
    s.add_argument("prompt", nargs="*")
    s.add_argument("-m", "--model", default=None)
    s.add_argument("--think", action="store_true", help="let reasoning models think first")
    s.add_argument("--max-tokens", type=int, default=1024)
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("bench", help="tokens/s + package/system power")
    s.add_argument("models", nargs="*", help="NPU models or aliases (default: fast)")
    s.add_argument("--other", nargs="*", help="MODEL[@backend] on another server, e.g. Qwen3.5-4B-GGUF@vulkan")
    s.add_argument("--other-url", default="http://127.0.0.1:13305", help="its base URL (default: Lemonade)")
    s.add_argument("--pmode", help="comma list, e.g. powersaver,balanced,performance")
    s.add_argument("--tokens", type=int, default=256)
    s.add_argument("--runs", type=int, default=2)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=lambda a: __import__("npukit.bench", fromlist=["main"]).main(a))

    s = sub.add_parser("dictate", help="toggle mic recording → Whisper (NPU/GPU) → type + copy")
    s.add_argument("action", nargs="?", default="toggle", choices=["toggle", "start", "stop", "cancel"])
    s.set_defaults(fn=lambda a: __import__("npukit.dictate", fromlist=["main"]).main(a))

    s = sub.add_parser("transcribe", help="transcribe an audio file (16 kHz WAV works best)")
    s.add_argument("file")
    s.add_argument("--language", default="")
    s.set_defaults(fn=lambda a: __import__("npukit.dictate", fromlist=["file_main"]).file_main(a.file, a.language))

    s = sub.add_parser("ocr", help="screen region → text on the clipboard")
    s.add_argument("--describe", action="store_true", help="describe the region instead")
    s.add_argument("--ask", help="custom question about the region")
    s.add_argument("-t", "--translate", nargs="?", const="", metavar="LANG",
                   help="translate the text (default: [ocr] translate_to); LANG = name or code like sv")
    s.add_argument("-p", "--pick", action="store_true",
                   help="translate, choosing the language from a searchable list first")
    s.add_argument("--file", help="read this PNG instead of capturing")
    s.add_argument("--copy", action="store_true", help="copy even with --file")
    s.add_argument("-m", "--model")
    s.set_defaults(fn=lambda a: __import__("npukit.ocr", fromlist=["main"]).main(a))

    s = sub.add_parser("index", help="update the semantic search index")
    s.add_argument("--full", action="store_true", help="re-embed everything")
    s.add_argument("-q", "--quiet", action="store_true")
    s.set_defaults(fn=lambda a: __import__("npukit.search", fromlist=["index"]).index(a))

    s = sub.add_parser("search", help="semantic search over notes + code")
    s.add_argument("query", nargs="*")
    s.add_argument("-n", type=int)
    s.add_argument("--rofi", action="store_true")
    s.add_argument("--theme")
    s.set_defaults(fn=lambda a: __import__("npukit.search", fromlist=["search"]).search(a))

    s = sub.add_parser("menu", help="rofi menu of everything above")
    s.add_argument("--theme")
    s.set_defaults(fn=cmd_menu)

    s = sub.add_parser("serve", help="run the gateway (normally started by npud.socket)")
    s.set_defaults(fn=lambda a: __import__("npukit.gateway", fromlist=["main"]).main() or 0)

    args = p.parse_args(argv)
    try:
        return args.fn(args) or 0
    except client.ApiError as e:
        print(f"npu: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())

