"""`npu dictate`: press once to record, again to transcribe on the NPU and type the text.

The first press starts pw-record and, in parallel, asks the gateway to load Whisper, so the
model is usually ready by the time you stop talking. Works in any Wayland app: the text is
typed with wtype (virtual keyboard) when it is installed and always put on the clipboard.
"""

from __future__ import annotations

import array
import json
import os
import signal
import subprocess
import sys
import time
import wave
from pathlib import Path

from . import client
from . import config as C
from . import ui


def _paths():
    rt = C.runtime_dir()
    return rt / "dictate.json", rt / "dictate.wav"


def _activity(name):
    try:
        client.call("/npu/activity", {"activity": name}, timeout=5)
    except client.ApiError:
        pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start(cfg) -> int:
    meta, wav = _paths()
    d = cfg["dictate"]
    wav.unlink(missing_ok=True)
    cmd = ["timeout", str(int(d["max_seconds"]))] + list(d["record"]) + [str(wav)]
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    meta.write_text(json.dumps({"pid": p.pid, "t": time.time()}))
    try:  # load Whisper while the user talks; show REC in the bar
        client.call("/npu/warm", {"component": "asr", "activity": "rec"}, timeout=180)
    except client.ApiError as e:
        ui.notify("Dictation", f"NPU not ready: {e}", urgency="critical")
    return 0


def rms(path: Path) -> float:
    with wave.open(str(path)) as w:
        if w.getsampwidth() != 2:
            return 1e9
        a = array.array("h", w.readframes(w.getnframes()))
    if not a:
        return 0.0
    step = max(1, len(a) // 64000)
    s = a[::step]
    return (sum(x * x for x in s) / len(s)) ** 0.5


def stop(cfg, info) -> int:
    meta, wav = _paths()
    d = cfg["dictate"]
    meta.unlink(missing_ok=True)
    try:  # signal the whole group: timeout + pw-record finish the WAV header on SIGINT
        os.killpg(info["pid"], signal.SIGINT)
    except OSError:
        pass
    for _ in range(50):
        if not _alive(info["pid"]):
            break
        time.sleep(0.05)
    try:
        os.waitpid(info["pid"], os.WNOHANG)
    except OSError:
        pass
    took = time.time() - info["t"]
    if not wav.exists() or took < 0.4 or wav.stat().st_size < 8000:
        _activity(None)
        ui.notify("Dictation", "Too short, nothing recorded", timeout_ms=2000)
        return 1
    level = rms(wav)
    if level < float(d["min_rms"]):
        _activity(None)
        ui.notify("Dictation", f"Only silence (level {level:.0f})", timeout_ms=2500)
        return 1
    _activity("stt")
    t0 = time.monotonic()
    try:
        text = client.transcribe(wav.read_bytes(), language=d["language"], prompt=d["prompt"])
    except client.ApiError as e:
        _activity(None)
        ui.notify("Dictation failed", str(e), urgency="critical")
        return 1
    finally:
        wav.unlink(missing_ok=True)
    _activity(None)
    if not text:
        ui.notify("Dictation", "No speech recognised", timeout_ms=2500)
        return 1
    if d.get("copy", True):
        ui.copy(text)
    typed = ui.type_text(text, list(d.get("type") or []))
    how = "typed + copied" if typed else "copied (install wtype to type it)"
    ui.notify(f"Dictation · {took:.0f}s audio → {time.monotonic() - t0:.1f}s on NPU",
              f"{text[:200]}\n<i>{how}</i>", timeout_ms=4000)
    print(text)
    return 0


def main(args) -> int:
    cfg = C.load()
    meta, _ = _paths()
    info = None
    if meta.exists():
        try:
            info = json.loads(meta.read_text())  # recorder may have hit max_seconds: still stop()
        except (ValueError, KeyError):
            info = None
    action = args.action
    if action == "cancel":
        if info:
            meta.unlink(missing_ok=True)
            try:
                os.killpg(info["pid"], signal.SIGTERM)
            except OSError:
                pass
            _paths()[1].unlink(missing_ok=True)
        _activity(None)
        return 0
    if action == "start" or (action == "toggle" and not info):
        if info and _alive(info["pid"]):
            return 0
        return start(cfg)
    if not info:
        print("not recording", file=sys.stderr)
        return 1
    return stop(cfg, info)


def file_main(path: str, language: str = "") -> int:
    t0 = time.monotonic()
    text = client.transcribe(Path(path).read_bytes(), language=language)
    print(text)
    print(f"[{time.monotonic() - t0:.1f}s]", file=sys.stderr)
    return 0

