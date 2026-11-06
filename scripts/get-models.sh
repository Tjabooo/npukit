#!/usr/bin/env bash
# Download the default models of the llamacpp (GPU) backend into ~/.local/share/npukit-engines/models:
# Whisper large-v3-turbo (speech to text, 1.6 GB), Qwen3.5-9B Q4_K_M + its vision projector
# (fast/vision: OCR, describe, translate, chat; 6.6 GB) and Qwen3.6-35B-A3B UD-IQ3_XXS (quality,
# 13.2 GB; skip it with --no-quality). Resumes partial downloads; skips files already there.
# NPUKIT_RATE=15M caps the download speed (curl --limit-rate), e.g. while gaming.
# Other [llamacpp.models] entries with hf = "…" are fetched by llama-server on first use instead.
set -euo pipefail
dir=${NPUKIT_MODELS_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/npukit-engines/models}
hf=https://huggingface.co
mkdir -p "$dir"

rate=()
[ -n "${NPUKIT_RATE:-}" ] && rate=(--limit-rate "$NPUKIT_RATE")
get() {  # get <url> <file>
    [ -s "$dir/$2" ] && { echo "have $2"; return; }
    echo "downloading $2"
    curl -fL --retry 5 -C - "${rate[@]}" -o "$dir/$2.part" "$1"
    mv "$dir/$2.part" "$dir/$2"
}

skip=" $* "
[[ $skip == *" --no-whisper "* ]] || get "$hf/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin" ggml-large-v3-turbo.bin
get "$hf/unsloth/Qwen3.5-9B-GGUF/resolve/main/mmproj-F16.gguf" Qwen3.5-9B-mmproj-F16.gguf
get "$hf/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf" Qwen3.5-9B-Q4_K_M.gguf
[[ $skip == *" --no-quality "* ]] || get "$hf/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf" \
    Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
echo "models in $dir"
