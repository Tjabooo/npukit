"""`npu ocr`: select a screen region, read it with a vision model on the NPU, copy the text.

With `--translate [LANG]` the text is then translated by a chat model (also on the NPU)."""

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

# short codes accepted by --translate; anything else is passed to the model as written
LANGS = {"en": "English", "sv": "Swedish", "de": "German", "fr": "French", "es": "Spanish",
         "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "da": "Danish", "no": "Norwegian",
         "nb": "Norwegian", "fi": "Finnish", "pl": "Polish", "ru": "Russian", "uk": "Ukrainian",
         "tr": "Turkish", "ar": "Arabic", "ja": "Japanese", "ko": "Korean", "zh": "Chinese"}


def capture(cmd: list[str]) -> bytes:
    r = subprocess.run(cmd, capture_output=True, check=False)
    return r.stdout if r.returncode == 0 and r.stdout[:8] == b"\x89PNG\r\n\x1a\n" else b""


def _content(out: dict) -> str:
    text = (out["choices"][0]["message"].get("content") or "").strip()
    m = FENCE.match(text)
    return m.group(1) if m else text


def read_image(png: bytes, prompt: str, model: str) -> tuple[str, float]:
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt},
                                         {"type": "image_url", "image_url": {"url": url}}]}]
    t0 = time.monotonic()
    out = client.chat(msgs, model, max_tokens=2048, temperature=0,
                      chat_template_kwargs={"enable_thinking": False})
    return _content(out), time.monotonic() - t0


def language(name: str) -> str:
    name = name.strip()
    return LANGS.get(name.lower(), name)


def translate_messages(text: str, lang: str, prompt: str) -> list[dict]:
    return [{"role": "system", "content": prompt.format(lang=lang)},
            {"role": "user", "content": text}]


def translate(text: str, lang: str, prompt: str, model: str) -> tuple[str, float]:
    t0 = time.monotonic()
    out = client.chat(translate_messages(text, lang, prompt), model, max_tokens=2048, temperature=0,
                      chat_template_kwargs={"enable_thinking": False})
    return _content(out), time.monotonic() - t0


def _activity(name: str | None) -> None:
    try:
        client.call("/npu/activity", {"activity": name}, timeout=5)
    except client.ApiError:
        pass


def main(args) -> int:
    cfg = C.load()
    o = cfg["ocr"]
    png = Path(args.file).read_bytes() if args.file else capture(list(o["capture"]))
    if not png:
        return 1  # selection cancelled
    prompt = o["describe_prompt"] if args.describe else o["prompt"]
    if args.ask:
        prompt = args.ask
    lang = None
    if getattr(args, "translate", None) is not None:
        lang = language(args.translate or o["translate_to"])
    _activity("ocr")
    try:
        text, took = read_image(png, prompt, args.model or o["model"])
        if text and lang:
            original = text
            text, t2 = translate(original, lang, o["translate_prompt"], o["translate_model"])
            took += t2
    except client.ApiError as e:
        ui.notify("Translate failed" if lang else "OCR failed", str(e), urgency="critical")
        return 1
    finally:
        _activity(None)
    if not text:
        ui.notify("OCR", "No text found", timeout_ms=2500)
        return 1
    if not args.file or args.copy:
        ui.copy(text)
        title = f"→ {lang}" if lang else "OCR"
        ui.notify(f"{title} · {took:.1f}s on NPU · copied", text[:300], timeout_ms=8000 if lang else 5000)
    if lang:
        print(original, file=sys.stderr)
        print("---", file=sys.stderr)
    print(text)
    print(f"[{took:.1f}s]", file=sys.stderr)
    return 0
