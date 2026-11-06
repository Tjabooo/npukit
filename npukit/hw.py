"""Small sysfs readers: power source, package power, battery draw, NPU state, GPU memory."""

from __future__ import annotations

import glob
import os
from pathlib import Path


def _read(path: str | Path, default: str = "") -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def ac_online_path(configured: str = "") -> str:
    if configured:
        return configured
    for d in sorted(glob.glob("/sys/class/power_supply/*")):
        if _read(f"{d}/type") == "Mains" and os.path.exists(f"{d}/online"):
            return f"{d}/online"
    return ""


def on_ac(configured: str = "") -> bool:
    p = ac_online_path(configured)
    return _read(p, "1") == "1" if p else True  # desktops without a battery count as AC


def battery_dir() -> str:
    for d in sorted(glob.glob("/sys/class/power_supply/*")):
        if _read(f"{d}/type") == "Battery" and _read(f"{d}/scope") != "Device":
            return d
    return ""


def battery_watts(bat: str | None = None) -> float | None:
    """System draw from the battery (only meaningful while discharging)."""
    bat = bat if bat is not None else battery_dir()
    if not bat:
        return None
    v = _read(f"{bat}/power_now")
    if v:
        return int(v) / 1e6
    cur, volt = _read(f"{bat}/current_now"), _read(f"{bat}/voltage_now")
    return int(cur) * int(volt) / 1e12 if cur and volt else None


def ppt_path() -> str:
    """amdgpu on an APU reports the whole SoC package power (PPT) without root."""
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        if _read(f"{h}/name") == "amdgpu" and os.path.exists(f"{h}/power1_input"):
            return f"{h}/power1_input"
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        if _read(f"{h}/name") == "amdgpu" and os.path.exists(f"{h}/power1_average"):
            return f"{h}/power1_average"
    return ""


def watts(path: str) -> float | None:
    v = _read(path) if path else ""
    return int(v) / 1e6 if v else None


def npu_device() -> str:
    """sysfs dir of the first amdxdna device (e.g. /sys/bus/pci/devices/0000:c5:00.1)."""
    for a in sorted(glob.glob("/sys/class/accel/accel*")):
        dev = os.path.realpath(f"{a}/device")
        if os.path.basename(os.path.realpath(f"{dev}/driver")) == "amdxdna":
            return dev
    return ""


def npu_info() -> dict:
    dev = npu_device()
    if not dev:
        return {"present": False}
    return {
        "present": True,
        "pci": os.path.basename(dev),
        "firmware": _read(f"{dev}/fw_version"),
        "runtime": _read(f"{dev}/power/runtime_status"),
        "active_ms": int(_read(f"{dev}/power/runtime_active_time", "0") or 0),
        "node": "/dev/accel/" + os.path.basename(glob.glob(f"{dev}/accel/accel*")[0])
        if glob.glob(f"{dev}/accel/accel*") else "",
    }


def gpu_vram() -> dict | None:
    """VRAM of the largest amdgpu card in bytes: {"total", "used"}; None without amdgpu sysfs."""
    best = None
    seen = set()
    for d in glob.glob("/sys/class/drm/card*/device"):
        real = os.path.realpath(d)
        if real in seen:
            continue
        seen.add(real)
        total, used = _read(f"{d}/mem_info_vram_total"), _read(f"{d}/mem_info_vram_used")
        if total.isdigit() and used.isdigit() and (best is None or int(total) > best["total"]):
            best = {"total": int(total), "used": int(used)}
    return best


def proc_vram(pid: int) -> int:
    """Bytes of VRAM a process holds (amdgpu fdinfo, each DRM client counted once); 0 if unknown."""
    total, clients = 0, set()
    try:
        fds = os.listdir(f"/proc/{pid}/fdinfo")
    except OSError:
        return 0
    for fd in fds:
        info = _read(f"/proc/{pid}/fdinfo/{fd}")
        if "drm-driver:\tamdgpu" not in info:
            continue
        fields = dict(ln.split(":", 1) for ln in info.splitlines() if ":" in ln)
        cid = fields.get("drm-client-id", "").strip()
        vram = fields.get("drm-memory-vram", "").split()
        if cid in clients or not vram or not vram[0].isdigit():
            continue
        clients.add(cid)
        total += int(vram[0]) * 1024
    return total


def mem_available() -> int | None:
    """MemAvailable from /proc/meminfo in bytes (what can be used without swapping)."""
    for ln in _read("/proc/meminfo").splitlines():
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) * 1024
    return None


def proc_rss_anon(pid: int) -> int:
    """Anonymous resident memory of a process in bytes (comes back when it exits); 0 if unknown."""
    for ln in _read(f"/proc/{pid}/status").splitlines():
        if ln.startswith("RssAnon:"):
            return int(ln.split()[1]) * 1024
    return 0
