"""`npu ocr`: select a screen region, read it with a vision model on the NPU, copy the text."""

from __future__ import annotations

import base64
import re
import subprocess
import sys
import time
from pathlib import Path

from . import client
from . import config as C
from . import ui

FENCE = re.compile(r"^```[\w+-]*\n(.*?)\n?```$", re.S)


def capture(cmd: list[str]) -> bytes:
    r = subprocess.run(cmd, capture_output=True, check=False)
    return r.stdout if r.returncode == 0 and r.stdout[:8] == b"\x89PNG\r\n\x1a\n" else b""


def read_image(png: bytes, prompt: str, model: str) -> tuple[str, float]:
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt},
                                         {"type": "image_url", "image_url": {"url": url}}]}]
    t0 = time.monotonic()
    out = client.chat(msgs, model, max_tokens=2048, temperature=0,
                      chat_template_kwargs={"enable_thinking": False})
    text = out["choices"][0]["message"].get("content") or ""
    text = text.strip()
    m = FENCE.match(text)
    if m:
        text = m.group(1)
    return text, time.monotonic() - t0


def main(args) -> int:
    cfg = C.load()
    o = cfg["ocr"]
    png = Path(args.file).read_bytes() if args.file else capture(list(o["capture"]))
    if not png:
        return 1  # selection cancelled
    prompt = o["describe_prompt"] if args.describe else o["prompt"]
    if args.ask:
        prompt = args.ask
    try:
        client.call("/npu/activity", {"activity": "ocr"}, timeout=5)
    except client.ApiError:
        pass
    try:
        text, took = read_image(png, prompt, args.model or o["model"])
    except client.ApiError as e:
        ui.notify("OCR failed", str(e), urgency="critical")
        return 1
    finally:
        try:
            client.call("/npu/activity", {"activity": None}, timeout=5)
        except client.ApiError:
            pass
    if not text:
        ui.notify("OCR", "No text found", timeout_ms=2500)
        return 1
    if not args.file or args.copy:
        ui.copy(text)
        ui.notify(f"OCR · {took:.1f}s on NPU · copied", text[:300], timeout_ms=5000)
    print(text)
    print(f"[{took:.1f}s]", file=sys.stderr)
    return 0
