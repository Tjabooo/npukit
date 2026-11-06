# Changelog

All notable changes to this project are listed here ([Keep a Changelog](https://keepachangelog.com), [SemVer](https://semver.org)).

## [Unreleased]

- GPU backend: `[server] backend = "llamacpp"` runs the same tools (dictation, OCR, translate, describe,
  search, the OpenAI endpoint, the waybar module) on llama.cpp + whisper.cpp for machines without an NPU.
  One server per component, so swapping the chat model keeps Whisper loaded; unknown or missing models
  fail before anything is unloaded. `scripts/get-models.sh` downloads the defaults (Qwen3.5-9B + vision
  projector, Whisper large-v3-turbo). `NPUKIT_INDEX=0 ./install.sh` skips the search index timer.
- `npu ocr --describe` / `--ask`: the notification says Describe / Answer instead of OCR.

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
