#!/usr/bin/env bash
# Install npukit for the current user: code + venv in ~/.local/share/npukit, `npu` in ~/.local/bin,
# systemd user units (socket-activated gateway, index timer), example config if none exists.
# Re-run after `git pull` to update. Nothing here needs root.
set -euo pipefail
src=$(cd "$(dirname "$0")" && pwd)
dest=${NPUKIT_HOME:-$HOME/.local/share/npukit}
units=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user
conf=${XDG_CONFIG_HOME:-$HOME/.config}/npukit

command -v flm >/dev/null || echo "warning: FastFlowLM (flm) not found in PATH; see README → Requirements" >&2

mkdir -p "$dest" "$HOME/.local/bin" "$units" "$conf"
rm -rf "$dest/npukit" "$dest/bin"
cp -r "$src/npukit" "$src/bin" "$dest/"
find "$dest/npukit" -name __pycache__ -prune -exec rm -rf {} +

# own venv; --system-site-packages so a distro numpy (python-numpy) is used when present
if [ ! -x "$dest/venv/bin/python" ]; then
    python3 -m venv --system-site-packages "$dest/venv"
fi
"$dest/venv/bin/python" -c 'import numpy' 2>/dev/null \
    || echo "note: numpy not found; search falls back to pure Python (pacman -S python-numpy, or pip install numpy into $dest/venv)" >&2

ln -sf "$dest/bin/npu" "$HOME/.local/bin/npu"
cp "$src"/systemd/*.service "$src"/systemd/*.socket "$src"/systemd/*.timer "$units/"
[ -f "$conf/config.toml" ] || cp "$src/config.example.toml" "$conf/config.toml"

systemctl --user daemon-reload
systemctl --user enable --now npud.socket npu-index.timer
# a running gateway keeps old code until it exits; restart it so the update takes effect
systemctl --user try-restart npud.service || true
echo "installed: $(readlink -f "$HOME/.local/bin/npu") · config $conf/config.toml"
echo "try: npu status   npu ask 'hello'   npu bench"
