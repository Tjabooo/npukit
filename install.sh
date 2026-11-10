#!/usr/bin/env bash
# Install npukit for the current user: the `npu` binary in ~/.local/share/npukit/bin (linked from
# ~/.local/bin), systemd user units (socket-activated gateway, index timer), example config if none exists.
# Without an NPU: set [server] backend = "llamacpp" in the config first (see README).
# Re-run after `git pull` to update. Nothing here needs root.
#
# NPUKIT_BINARY=/path/to/npu ./install.sh   install a binary built elsewhere (same CPU architecture)
# NPUKIT_INDEX=0 ./install.sh               don't enable the 30-minute search index timer (e.g. a gaming PC)
set -euo pipefail
src=$(cd "$(dirname "$0")" && pwd)
dest=${NPUKIT_HOME:-$HOME/.local/share/npukit}
units=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user
conf=${XDG_CONFIG_HOME:-$HOME/.config}/npukit
index=${NPUKIT_INDEX:-1}

backend=flm
if [ -f "$conf/config.toml" ] && grep -Eq '^[[:space:]]*backend[[:space:]]*=[[:space:]]*"llama' "$conf/config.toml"; then
    backend=llamacpp
fi
if [ "$backend" = flm ]; then
    command -v flm >/dev/null || echo "warning: FastFlowLM (flm) not found in PATH; see README → Requirements" >&2
else
    for b in llama-server whisper-server; do
        command -v "$b" >/dev/null || echo "warning: $b not found in PATH; see README → Without an NPU" >&2
    done
fi

if [ -n "${NPUKIT_BINARY:-}" ]; then
    bin=$NPUKIT_BINARY
else
    if ! command -v cargo >/dev/null; then
        echo "error: cargo not found: install Rust (Arch: pacman -S rust) or pass NPUKIT_BINARY=<built npu>" >&2
        exit 1
    fi
    cargo build --release --locked --manifest-path "$src/Cargo.toml"
    bin="$src/target/release/npu"
fi
"$bin" --version >/dev/null

mkdir -p "$dest/bin" "$HOME/.local/bin" "$units" "$conf"
# the Python version's package, venv and launcher (the search index next to them is kept)
rm -rf "$dest/npukit" "$dest/venv"
# replace atomically: a running gateway keeps its old binary until it restarts
install -m 755 "$bin" "$dest/bin/.npu.new"
mv -f "$dest/bin/.npu.new" "$dest/bin/npu"
ln -sf "$dest/bin/npu" "$HOME/.local/bin/npu"
cp "$src"/systemd/*.service "$src"/systemd/*.socket "$src"/systemd/*.timer "$units/"
[ -f "$conf/config.toml" ] || cp "$src/config.example.toml" "$conf/config.toml"

systemctl --user daemon-reload
systemctl --user enable --now npud.socket
if [ "$index" = 0 ]; then
    systemctl --user disable --now npu-index.timer 2>/dev/null || true
else
    systemctl --user enable --now npu-index.timer
fi
# a running gateway keeps old code until it exits; restart it so the update takes effect
systemctl --user try-restart npud.service || true
echo "installed: $("$dest/bin/npu" --version) → $(readlink -f "$HOME/.local/bin/npu") · backend $backend · config $conf/config.toml"
echo "try: npu status   npu ask 'hello'   npu bench"
