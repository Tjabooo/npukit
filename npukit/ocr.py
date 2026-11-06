"""`npu ocr`: select a screen region, read it with a vision model on the NPU, copy the text.

With `--translate [LANG]` the text is then translated by a chat model (also on the NPU)."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import client
from . import config as C
from . import ui

FENCE = re.compile(r"^```[\w+-]*\n(.*?)\n?```$", re.S)

# languages in the --pick list: (code, English name, own name). The codes also work for
# --translate; anything else is passed to the model as written.
LANGUAGES = [
    ("ar", "Arabic", "العربية"), ("bg", "Bulgarian", "български"), ("bn", "Bengali", "বাংলা"),
    ("ca", "Catalan", "català"), ("cs", "Czech", "čeština"), ("da", "Danish", "dansk"),
    ("de", "German", "Deutsch"), ("el", "Greek", "Ελληνικά"), ("en", "English", ""),
    ("es", "Spanish", "español"), ("et", "Estonian", "eesti"), ("fa", "Persian", "فارسی"),
    ("fi", "Finnish", "suomi"), ("fr", "French", "français"), ("he", "Hebrew", "עברית"),
    ("hi", "Hindi", "हिन्दी"), ("hr", "Croatian", "hrvatski"), ("hu", "Hungarian", "magyar"),
    ("id", "Indonesian", "Bahasa Indonesia"), ("is", "Icelandic", "íslenska"), ("it", "Italian", "italiano"),
    ("ja", "Japanese", "日本語"), ("ko", "Korean", "한국어"), ("lt", "Lithuanian", "lietuvių"),
    ("lv", "Latvian", "latviešu"), ("ms", "Malay", "Bahasa Melayu"), ("nl", "Dutch", "Nederlands"),
    ("no", "Norwegian", "norsk"), ("pl", "Polish", "polski"), ("pt", "Portuguese", "português"),
    ("ro", "Romanian", "română"), ("ru", "Russian", "русский"), ("sk", "Slovak", "slovenčina"),
    ("sl", "Slovenian", "slovenščina"), ("sr", "Serbian", "српски"), ("sv", "Swedish", "svenska"),
    ("sw", "Swahili", "Kiswahili"), ("th", "Thai", "ไทย"), ("tl", "Filipino", ""),
    ("tr", "Turkish", "Türkçe"), ("uk", "Ukrainian", "українська"), ("ur", "Urdu", "اردو"),
    ("vi", "Vietnamese", "Tiếng Việt"), ("zh", "Chinese", "中文"),
    ("zh-tw", "Chinese (Traditional)", "繁體中文"),
]
LANGS = {code: name for code, name, _ in LANGUAGES} | {"nb": "Norwegian"}
RECENT_MAX = 6


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


def _recent_file() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return Path(base) / "npukit" / "translate-recent.json"


def recent_languages() -> list[str]:
    try:
        data = json.loads(_recent_file().read_text())
    except (OSError, ValueError):
        return []
    return [x for x in data if isinstance(x, str) and x.strip()][:RECENT_MAX] if isinstance(data, list) else []


def remember_language(lang: str) -> None:
    items = [lang] + [x for x in recent_languages() if x.lower() != lang.lower()]
    f = _recent_file()
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(items[:RECENT_MAX], ensure_ascii=False))
    except OSError:
        pass


def picker_rows(recent: list[str], default: str) -> tuple[list[tuple[str, str]], int]:
    """Rows for the language picker as (shown text, language) and how many lead as recent:
    the last used first (so Enter repeats it), then the default, then A to Z."""
    known = {name.lower(): (code, own) for code, name, own in LANGUAGES}
    order: list[str] = []
    for lang in [*recent, default, *sorted(name for _, name, _ in LANGUAGES)]:
        if lang.lower() not in (x.lower() for x in order):
            order.append(lang)
    rows = []
    for lang in order:
        code, own = known.get(lang.lower(), ("", ""))
        rows.append((f"{lang:<23}{code:<7}{own}".rstrip(), lang))
    return rows, len({x.lower() for x in recent})


def pick_language(cfg: dict) -> str | None:
    """Searchable list of languages (rofi). Typing one that isn't listed and pressing Enter uses it
    as written. None = cancelled."""
    rows, n_recent = picker_rows(recent_languages(), language(cfg["ocr"]["translate_to"]))
    u = cfg["ui"]
    theme = u.get("lang_theme") or u.get("menu_theme")
    args = ["rofi", "-dmenu", "-i", "-p", "into", "-format", "i\ts",
            "-mesg", "Pick a language, then select the screen region. Not listed? Type it and press Enter."]
    if n_recent:
        args += ["-a", ",".join(str(i) for i in range(n_recent))]  # recent ones stand out
    if theme:
        args += ["-theme", str(C.expand(theme))]
    try:
        r = subprocess.run(args, input="\n".join(t for t, _ in rows).encode(), capture_output=True, check=False)
    except OSError:
        ui.notify("Translate", "rofi is needed for the language list", urgency="critical")
        return None
    return parse_pick(r.returncode, r.stdout.decode(), rows)


def parse_pick(code: int, out: str, rows: list[tuple[str, str]]) -> str | None:
    if code != 0 or not out.strip():
        return None
    idx, tab, text = out.rstrip("\n").partition("\t")
    try:
        i = int(idx) if tab else -1
    except ValueError:
        i = -1
    if 0 <= i < len(rows):
        return rows[i][1]
    text = text if tab else idx
    return language(text) if text.strip() else None  # typed, not in the list


def _activity(name: str | None) -> None:
    try:
        client.call("/npu/activity", {"activity": name}, timeout=5)
    except client.ApiError:
        pass


def main(args) -> int:
    cfg = C.load()
    o = cfg["ocr"]
    if getattr(args, "pick", False):  # choose the language first, then the region
        chosen = pick_language(cfg)
        if not chosen:
            return 1
        args.translate = chosen
        if not args.file:  # the region pick freezes the screen: let the list vanish first
            ui.wait_menu_gone(float(cfg["ui"].get("menu_close_wait", 0.45)))
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
            remember_language(lang)
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
