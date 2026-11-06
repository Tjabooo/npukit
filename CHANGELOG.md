# Changelog

All notable changes to this project are listed here ([Keep a Changelog](https://keepachangelog.com), [SemVer](https://semver.org)).

## [Unreleased]

- `npu ocr --translate [LANG]`: read a screen region and translate it (default `[ocr] translate_to`,
  English); also in `npu menu`.
- `npu menu`: region actions (OCR, translate, describe) wait for the menu to finish fading out
  before the screen freezes for the selection, so the menu no longer shows up in the capture
  (`[ui] menu_close_wait`, default 0.45 s).

## [0.1.0] - 2026-11-05

- First release.
