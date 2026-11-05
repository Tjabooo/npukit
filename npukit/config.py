"""Configuration: built-in defaults, overridden by ~/.config/npukit/config.toml."""

from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path

DEFAULTS: dict = {
    "server": {
        "listen": "127.0.0.1:13306",  # OpenAI-compatible endpoint the tools and other apps use
        "flm": "flm",                 # FastFlowLM binary
        "flm_port": 13307,            # private port of the flm child (localhost only)
        "flm_args": [],               # extra `flm serve` arguments
        "max_components": 2,          # chat model + 1 helper fits on an XDNA2 NPU; all three do not
        "exit_after": 60,             # socket-activated gateway exits after this long with nothing loaded
        "start_timeout": 180,
    },
    "models": {
        # aliases usable as "model" in any request; anything else is passed to flm as a tag
        "aliases": {
            "fast": "qwen3.5:4b",
            "quality": "gpt-oss:20b",
            "vision": "qwen3vl-it:4b",
        },
        "default": "fast",
    },
    "power": {
        "ac_online": "",              # path to the AC "online" file; empty = first Mains supply
        "ac_pmode": "balanced",       # flm --pmode on AC: powersaver | balanced | performance | turbo
        "battery_pmode": "powersaver",
        "ac_idle_unload": 600,        # seconds without a request before the model leaves the NPU
        "battery_idle_unload": 120,
    },
    "bar": {
        "file": "~/.local/state/npukit/bar.json",  # waybar custom module reads this
        "signal": 16,                 # RTMIN+N sent to waybar on every change, 0 = off
        "tag": "NPU",
    },
    "dictate": {
        "record": ["pw-record", "--rate", "16000", "--channels", "1", "--format", "s16"],
        "max_seconds": 120,
        "language": "",               # empty = auto-detect
        "prompt": "",                 # spelling hints (names, jargon)
        "type": ["wtype", "-"],       # reads text on stdin; skipped when not installed
        "copy": True,                 # also put the text on the clipboard
        "min_rms": 120,               # quieter recordings are treated as silence (Whisper invents text)
    },
    "ocr": {
        "capture": ["hyprshot", "-z", "-s", "-m", "region", "--raw"],
        "model": "vision",
        "prompt": ("Transcribe all text in this image exactly as written. Keep line breaks, "
                   "indentation and spacing. Output only the text, no commentary, no code fences."),
        "describe_prompt": "Describe this image briefly and precisely.",
        "translate_to": "English",    # `npu ocr --translate` with no language
        "translate_model": "fast",    # chat model that translates the read text
        "translate_prompt": ("Translate the text the user sends into {lang}. Keep line breaks, lists and "
                             "formatting; leave code, names, numbers and URLs as they are. If it is already "
                             "in {lang}, return it unchanged. Output only the translation, no commentary."),
    },
    "ui": {
        "menu_theme": "",             # rofi -theme for `npu menu` (empty = rofi's default)
        "search_theme": "",           # rofi -theme for `npu search --rofi`
    },
    "search": {
        "roots": ["~/workspace", "~/Documents"],
        "db": "~/.local/share/npukit/index.sqlite",
        "exclude_dirs": ["node_modules", ".git", ".venv", "venv", "__pycache__", "target", "build",
                         "dist", ".cache", ".next", "vendor", ".gradle", ".idea", "out"],
        "extensions": [".md", ".txt", ".org", ".rst", ".py", ".rs", ".go", ".js", ".ts", ".tsx",
                       ".jsx", ".mjs", ".c", ".h", ".cpp", ".hpp", ".cs", ".java", ".kt", ".lua",
                       ".sh", ".bash", ".zsh", ".fish", ".toml", ".yaml", ".yml", ".json", ".ini",
                       ".conf", ".css", ".scss", ".html", ".svelte", ".vue", ".sql", ".ps1",
                       ".nix", ".rasi", ".service", ".timer", ".cfg", ".tex", ".dart", ".swift",
                       ".zig", ".hs", ".ex", ".rb", ".php"],
        "names": ["Makefile", "Dockerfile", "README", "LICENSE", "PKGBUILD", "Justfile"],
        # fnmatch patterns on the full path: vendored, generated and minified files
        "exclude_globs": ["*.min.*", "*.map", "*-lock.json", "*.lock", "*/third_party/*",
                          "*/external/*", "*/wp-includes/*", "*/wp-admin/*", "*/site-packages/*"],
        "max_file_kb": 384,
        "max_chunks_per_file": 40,
        "chunk_lines": 60,
        "overlap_lines": 8,
        "batch": 32,
        "results": 12,
        "open": ["xdg-open", "{path}"],  # {path} {line}; e.g. ["code", "-g", "{path}:{line}"]
    },
}

_cache: dict | None = None


def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(os.environ.get("NPUKIT_CONFIG", os.path.join(base, "npukit", "config.toml")))


def _merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict) and k != "aliases":
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load(reload: bool = False) -> dict:
    global _cache
    if _cache is not None and not reload:
        return _cache
    cfg = copy.deepcopy(DEFAULTS)
    p = config_path()
    if p.exists():
        with open(p, "rb") as f:
            _merge(cfg, tomllib.load(f))
    _cache = cfg
    return cfg


def expand(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(p)))


def runtime_dir() -> Path:
    d = Path(os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/npukit-{os.getuid()}") / "npukit"
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    return d


def base_url(cfg: dict | None = None) -> str:
    cfg = cfg or load()
    return "http://" + cfg["server"]["listen"]


def resolve_model(name: str | None, cfg: dict | None = None) -> str:
    """Alias or flm tag -> flm tag. Also accepts Lemonade-style names ("qwen3.5-4b-FLM")."""
    cfg = cfg or load()
    aliases = cfg["models"]["aliases"]
    name = (name or cfg["models"]["default"]).strip()
    for _ in range(4):  # aliases may point at aliases
        if name in aliases:
            name = aliases[name]
    if name.endswith("-FLM"):  # Lemonade naming: qwen3.5-4b-FLM -> qwen3.5:4b
        stem = name[:-4]
        head, _, size = stem.rpartition("-")
        if head:
            name = f"{head}:{size}"
    return name
