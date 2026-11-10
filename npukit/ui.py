"""Desktop glue: notifications, clipboard, typing, rofi. Each helper degrades quietly."""

from __future__ import annotations

import json
import shutil
import subprocess
import time


def notify(title: str, body: str = "", *, urgency: str = "normal", tag: str = "npukit",
           timeout_ms: int | None = None) -> None:
    if not shutil.which("notify-send"):
        return
    args = ["notify-send", "-a", "npukit", "-u", urgency, "-h", f"string:x-dunst-stack-tag:{tag}"]
    if timeout_ms is not None:
        args += ["-t", str(timeout_ms)]
    subprocess.run(args + [title, body], check=False)


def copy(text: str) -> bool:
    if not shutil.which("wl-copy"):
        return False
    # wl-copy forks a server that owns the selection; don't wait on its pipes
    subprocess.run(["wl-copy", "--", text], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def type_text(text: str, cmd: list[str]) -> bool:
    if not cmd or not shutil.which(cmd[0]):
        return False
    r = subprocess.run(cmd, input=text.encode(), check=False)
    return r.returncode == 0


def rofi(lines: list[str], prompt: str, *, theme: str | None = None, mesg: str | None = None,
         filter_input: bool = False) -> tuple[int, str]:
    """Return (exit code, selected or typed text). -dmenu with -format s."""
    args = ["rofi", "-dmenu", "-i", "-p", prompt, "-format", "i"]
    if filter_input:
        args = ["rofi", "-dmenu", "-p", prompt, "-format", "s"]
    if theme:
        args += ["-theme", theme]
    if mesg:
        args += ["-mesg", mesg]
    r = subprocess.run(args, input="\n".join(lines).encode(), capture_output=True, check=False)
    return r.returncode, r.stdout.decode().strip()


def wait_menu_gone(settle: float, namespace: str = "rofi", timeout: float = 1.5) -> None:
    """Block until the menu is off screen, so a frozen-screen region pick doesn't capture it.

    The compositor fades a closed layer out after rofi exits; wait for the layer to leave
    `hyprctl layers` (when Hyprland is there) and then for the fade itself."""
    if settle <= 0:
        return
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            out = subprocess.run(["hyprctl", "layers", "-j"], capture_output=True, text=True,
                                 timeout=1, check=False).stdout
            mons = json.loads(out or "{}")
        except (OSError, subprocess.TimeoutExpired, ValueError):
            break
        names = [l.get("namespace") for m in mons.values() for lv in m.get("levels", {}).values() for l in lv]
        if namespace not in names:
            break
        time.sleep(0.03)
    time.sleep(settle)
