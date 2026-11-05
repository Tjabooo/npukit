"""Desktop glue: notifications, clipboard, typing, rofi. Each helper degrades quietly."""

from __future__ import annotations

import shutil
import subprocess


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
