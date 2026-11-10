//! Small sysfs readers: power source, package power, battery draw, NPU state, GPU memory.

use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};

fn read(path: impl AsRef<Path>) -> String {
    fs::read_to_string(path).map(|s| s.trim().to_string()).unwrap_or_default()
}

/// Sorted entries of a directory whose names start with `prefix`.
fn entries(dir: &str, prefix: &str) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = fs::read_dir(dir)
        .map(|it| it.flatten().filter(|e| e.file_name().to_string_lossy().starts_with(prefix)).map(|e| e.path()).collect())
        .unwrap_or_default();
    v.sort();
    v
}

pub fn ac_online_path(configured: &str) -> String {
    if !configured.is_empty() {
        return configured.to_string();
    }
    for d in entries("/sys/class/power_supply", "") {
        if read(d.join("type")) == "Mains" && d.join("online").exists() {
            return d.join("online").to_string_lossy().into_owned();
        }
    }
    String::new()
}

/// On AC power? Desktops without a battery count as AC.
pub fn on_ac(path: &str) -> bool {
    if path.is_empty() {
        return true;
    }
    match fs::read_to_string(path) {
        Ok(s) => s.trim() == "1",
        Err(_) => true,
    }
}

pub fn battery_dir() -> String {
    for d in entries("/sys/class/power_supply", "") {
        if read(d.join("type")) == "Battery" && read(d.join("scope")) != "Device" {
            return d.to_string_lossy().into_owned();
        }
    }
    String::new()
}

/// System draw from the battery (only meaningful while discharging).
pub fn battery_watts(bat: &str) -> Option<f64> {
    if bat.is_empty() {
        return None;
    }
    let v = read(format!("{bat}/power_now"));
    if !v.is_empty() {
        return v.parse::<f64>().ok().map(|x| x / 1e6);
    }
    let (c, u) = (read(format!("{bat}/current_now")), read(format!("{bat}/voltage_now")));
    Some(c.parse::<f64>().ok()? * u.parse::<f64>().ok()? / 1e12)
}

/// amdgpu on an APU reports the whole SoC package power (PPT) without root.
pub fn ppt_path() -> String {
    for file in ["power1_input", "power1_average"] {
        for h in entries("/sys/class/hwmon", "hwmon") {
            if read(h.join("name")) == "amdgpu" && h.join(file).exists() {
                return h.join(file).to_string_lossy().into_owned();
            }
        }
    }
    String::new()
}

pub fn watts(path: &str) -> Option<f64> {
    if path.is_empty() {
        return None;
    }
    read(path).parse::<f64>().ok().map(|x| x / 1e6)
}

/// sysfs dir of the first amdxdna device (e.g. /sys/bus/pci/devices/0000:c5:00.1).
pub fn npu_device() -> Option<PathBuf> {
    for a in entries("/sys/class/accel", "accel") {
        let Ok(dev) = fs::canonicalize(a.join("device")) else { continue };
        let drv = fs::canonicalize(dev.join("driver")).ok();
        if drv.as_deref().and_then(|d| d.file_name()).map(|n| n == "amdxdna").unwrap_or(false) {
            return Some(dev);
        }
    }
    None
}

pub fn npu_info() -> Value {
    let Some(dev) = npu_device() else { return json!({"present": false}) };
    let node = entries(&dev.join("accel").to_string_lossy(), "accel").first()
        .map(|p| format!("/dev/accel/{}", p.file_name().unwrap().to_string_lossy()))
        .unwrap_or_default();
    json!({
        "present": true,
        "pci": dev.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        "firmware": read(dev.join("fw_version")),
        "runtime": read(dev.join("power/runtime_status")),
        "active_ms": read(dev.join("power/runtime_active_time")).parse::<i64>().unwrap_or(0),
        "node": node,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vram {
    pub total: u64,
    pub used: u64,
}

/// VRAM of the largest amdgpu card in bytes; None without amdgpu sysfs.
pub fn gpu_vram() -> Option<Vram> {
    let mut best: Option<Vram> = None;
    let mut seen = std::collections::HashSet::new();
    for c in entries("/sys/class/drm", "card") {
        let d = c.join("device");
        let Ok(real) = fs::canonicalize(&d) else { continue };
        if !seen.insert(real) {
            continue;
        }
        let (Ok(total), Ok(used)) = (read(d.join("mem_info_vram_total")).parse::<u64>(), read(d.join("mem_info_vram_used")).parse::<u64>()) else {
            continue;
        };
        if best.is_none_or(|b| total > b.total) {
            best = Some(Vram { total, used });
        }
    }
    best
}

/// Bytes of VRAM a process holds (amdgpu fdinfo, each DRM client counted once); 0 if unknown.
pub fn proc_vram(pid: u32) -> u64 {
    let Ok(fds) = fs::read_dir(format!("/proc/{pid}/fdinfo")) else { return 0 };
    let mut total = 0;
    let mut clients = std::collections::HashSet::new();
    for fd in fds.flatten() {
        let info = read(fd.path());
        if !info.contains("drm-driver:\tamdgpu") {
            continue;
        }
        let (mut cid, mut vram) = (String::new(), None);
        for ln in info.lines() {
            if let Some((k, v)) = ln.split_once(':') {
                match k {
                    "drm-client-id" => cid = v.trim().to_string(),
                    "drm-memory-vram" => vram = v.split_whitespace().next().and_then(|x| x.parse::<u64>().ok()),
                    _ => {}
                }
            }
        }
        let Some(kib) = vram else { continue };
        if !clients.insert(cid) {
            continue;
        }
        total += kib * 1024;
    }
    total
}

fn kb_field(text: &str, key: &str) -> Option<u64> {
    text.lines().find(|l| l.starts_with(key))?.split_whitespace().nth(1)?.parse::<u64>().ok().map(|k| k * 1024)
}

/// MemAvailable from /proc/meminfo in bytes (what can be used without swapping).
pub fn mem_available() -> Option<u64> {
    kb_field(&read("/proc/meminfo"), "MemAvailable:")
}

/// Anonymous resident memory of a process in bytes (comes back when it exits); 0 if unknown.
pub fn proc_rss_anon(pid: u32) -> u64 {
    kb_field(&read(format!("/proc/{pid}/status")), "RssAnon:").unwrap_or(0)
}
