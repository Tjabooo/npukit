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
        # flm = FastFlowLM on the XDNA2 NPU; llamacpp = llama.cpp (llama-server) + whisper.cpp
        # (whisper-server) on a GPU via Vulkan/ROCm/CUDA, or the CPU, for machines without an NPU
        "backend": "flm",
    },
    "llamacpp": {
        "server": "llama-server",     # llama.cpp server binary (chat, vision, embeddings)
        "whisper": "whisper-server",  # whisper.cpp server binary (speech to text)
        "label": "GPU",               # where it runs, shown in notifications and `npu status`
        # each component runs as its own server on port flm_port + 0/1/2 (chat/asr/embed); a GPU
        # has room for all of them, so switching the chat model leaves Whisper loaded
        "max_components": 3,
        # headroom for other apps (games): llama.cpp keeps vram_reserve of the card free (--fit-target)
        # and puts what doesn't fit in RAM, MoE experts first; a model whose RAM share would leave less
        # than ram_reserve_gb available is refused; Whisper runs on the CPU when the GPU is at the reserve
        "vram_reserve": 0.2,          # fraction of VRAM to keep free; 0 = llama.cpp's 1 GiB margin
        "ram_reserve_gb": 4,          # 0 = no RAM check
        "threads": 0,                 # CPU threads for model layers in RAM (and CPU Whisper); 0 = all cores
        "overhead_gb": 1.0,           # context + work buffers on top of the model files, for the RAM check
        # every chat server; no -ngl so --fit decides what goes on the GPU
        "args": ["-c", "16384", "-np", "1", "--no-webui"],
        "whisper_model": "~/.local/share/npukit-engines/models/ggml-large-v3-turbo.bin",
        "whisper_args": ["-l", "auto"],
        "embed_model": "embed-gemma:300m",  # a [llamacpp.models] entry
        "embed_args": ["--embeddings", "-c", "2048", "-ub", "2048", "-b", "2048"],
        "download_timeout": 3600,     # s a first start of an hf = "…" model may take (it downloads)
        # used instead of [models.aliases] when the config file sets none
        "aliases": {"fast": "qwen3.5:9b", "quality": "qwen3.6:35b-a3b", "vision": "qwen3.5:9b"},
        # model tag -> GGUF: {path, mmproj} for local files, or {hf = "user/repo[:quant]"} to let
        # llama-server download it (mmproj included) on first use; optional per-model "args"
        "models": {
            "qwen3.5:9b": {"path": "~/.local/share/npukit-engines/models/Qwen3.5-9B-Q4_K_M.gguf",
                           "mmproj": "~/.local/share/npukit-engines/models/Qwen3.5-9B-mmproj-F16.gguf"},
            # MoE, 3B active: 13.2 GB at UD-IQ3_XXS, splits between VRAM and RAM and stays fast
            "qwen3.6:35b-a3b": {"path": "~/.local/share/npukit-engines/models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"},
            "qwen3.5:4b": {"hf": "unsloth/Qwen3.5-4B-GGUF:Q4_K_M"},
            "gpt-oss:20b": {"hf": "ggml-org/gpt-oss-20b-GGUF"},
            "embed-gemma:300m": {"hf": "ggml-org/embeddinggemma-300M-GGUF"},
        },
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
        "lang_theme": "",             # rofi -theme for the `npu ocr --pick` language list (empty = menu_theme)
        "menu_close_wait": 0.45,      # s to let the menu's close animation finish before a region pick
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


def _merge(base: dict, over: dict, path: tuple = ()) -> dict:
    for k, v in over.items():
        # alias tables and single model entries replace the default; everything else merges
        flat = k == "aliases" or path == ("llamacpp", "models")
        if isinstance(v, dict) and isinstance(base.get(k), dict) and not flat:
            _merge(base[k], v, path + (k,))
        else:
            base[k] = v
    return base


def load(reload: bool = False) -> dict:
    global _cache
    if _cache is not None and not reload:
        return _cache
    cfg = copy.deepcopy(DEFAULTS)
    p = config_path()
    user: dict = {}
    if p.exists():
        with open(p, "rb") as f:
            user = tomllib.load(f)
        _merge(cfg, user)
    if backend(cfg) == "llamacpp" and "aliases" not in user.get("models", {}):
        cfg["models"]["aliases"] = dict(cfg["llamacpp"]["aliases"])
    _cache = cfg
    return cfg


def backend(cfg: dict | None = None) -> str:
    cfg = cfg or load()
    return "llamacpp" if str(cfg["server"].get("backend", "flm")).lower() in ("llamacpp", "llama.cpp", "llama") \
        else "flm"


def engine_label(cfg: dict | None = None) -> str:
    """Where the models run, for messages: "NPU", or [llamacpp] label (default "GPU")."""
    cfg = cfg or load()
    return (cfg["llamacpp"].get("label") or "GPU") if backend(cfg) == "llamacpp" else "NPU"


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
