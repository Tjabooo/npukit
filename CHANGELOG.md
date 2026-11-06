# Changelog

All notable changes to this project are listed here ([Keep a Changelog](https://keepachangelog.com), [SemVer](https://semver.org)).

## [Unreleased]

- Translation is faithful by default: the prompt tells the model never to refuse, soften or censor
  profanity, insults or slurs (the 9B model refused vulgar game chat before), and OCR transcribes them as
  written. `[ocr] translate_context` adds notes (source, slang glossary); `[ocr] translate_ocr_model` picks
  the model that reads the region when translating. The GPU quality model gets its vision projector.

- `npu chat` (conversation) and file access for `npu ask` / `npu chat`: with folders in `[tools] read`
  (or `--read PATH`, `/read PATH`) the model calls read-only tools (list_dir, find_files, search_text,
  read_file) before answering. Outside paths, symlink escapes and `[tools] deny` (keys, `.env`, `.ssh/`, …)
  are refused. Falls back to a plain answer on servers without tool calls.
- GPU quality model gets a 32k context for the files it reads.

- `npu translate`: an eww overlay to translate a message you write. Language chips (recent first, any
  language can be typed), Enter translates and copies, Enter again types the translation into the window
  underneath (wtype). Own eww instance, colours from `[translate.colors]`, optional `on_open`/`on_close`
  hooks (a Hyprland submap for Esc). Its own message prompt (`[translate] prompt`): the OCR prompt's
  "return it unchanged if already in {lang}" made small models skip Swedish → English. Also in `npu menu`;
  `npu translate -t LANG "text"` prints without the overlay.
- GPU backend: `[server] backend = "llamacpp"` runs the same tools (dictation, OCR, translate, describe,
  search, the OpenAI endpoint, the waybar module) on llama.cpp + whisper.cpp for machines without an NPU.
  One server per component, so swapping the chat model keeps Whisper loaded; unknown or missing models
  fail before anything is unloaded. `scripts/get-models.sh` downloads the defaults (Qwen3.5-9B + vision
  projector, Whisper large-v3-turbo). `NPUKIT_INDEX=0 ./install.sh` skips the search index timer.
- `npu ocr --describe` / `--ask`: the notification says Describe / Answer instead of OCR.
- GPU backend headroom: `[llamacpp] vram_reserve` (20%) becomes llama.cpp's `--fit-target` (no more
  `-ngl 999`, which forced models that didn't fit into VRAM and spilled), `ram_reserve_gb` (4) refuses
  loads that would starve RAM, `threads` caps CPU threads, Whisper falls back to the CPU at the reserve,
  and npud.service runs with `CPUWeight=50`. `npu status` shows VRAM/RAM per server and overall.
- GPU `quality` is now Qwen3.6-35B-A3B (UD-IQ3_XXS, 13.2 GB) instead of gpt-oss-20b.

- `npu ocr --translate [LANG]`: read a screen region and translate it (default `[ocr] translate_to`,
  English); also in `npu menu`.
- `npu menu`: region actions (OCR, translate, describe) wait for the menu to finish fading out
  before the screen freezes for the selection, so the menu no longer shows up in the capture
  (`[ui] menu_close_wait`, default 0.45 s).
- `npu ocr --pick`: choose the target language in a searchable rofi list before selecting the region
  (English and native names plus codes, the last used on top so Enter repeats it, any unlisted
  language can be typed; `[ui] lang_theme`). "Translate region" in `npu menu` uses it too.

## [0.1.0] - 2026-11-05

- First release.
