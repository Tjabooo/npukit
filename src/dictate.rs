//! `npu dictate`: press once to record, again to transcribe (Whisper on the NPU or GPU) and type the text.
//!
//! The first press starts pw-record and, in parallel, asks the gateway to load Whisper, so the model is
//! usually ready by the time you stop talking. Works in any Wayland app: the text is typed with wtype
//! (virtual keyboard) when it is installed and always put on the clipboard.

use crate::client;
use crate::config::{self, strs, Config};
use crate::ui;
use crate::util::{self, pid_alive};
use serde_json::{json, Value};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn paths() -> (PathBuf, PathBuf) {
    let rt = config::runtime_dir();
    (rt.join("dictate.json"), rt.join("dictate.wav"))
}

fn start(cfg: &Config) -> i32 {
    let (meta, wav) = paths();
    let d = &cfg.dictate;
    let _ = std::fs::remove_file(&wav);
    let mut cmd = vec!["timeout".to_string(), (d.max_seconds as i64).to_string()];
    cmd.extend(strs(&d.record));
    cmd.push(wav.display().to_string());
    let child = Command::new(&cmd[0]).args(&cmd[1..]).process_group(0)
        .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn();
    let child = match child {
        Ok(c) => c,
        Err(e) => {
            ui::notify("Dictation", &format!("can't record: {e}"), "critical", None);
            return 1;
        }
    };
    let _ = std::fs::write(&meta, json!({"pid": child.id(), "t": util::now()}).to_string());
    // load Whisper while the user talks; show REC in the bar
    if let Err(e) = client::call("/npu/warm", Some(&json!({"component": "asr", "activity": "rec"})), 180.0) {
        ui::notify("Dictation", &format!("{} not ready: {e}", cfg.engine_label()), "critical", None);
    }
    0
}

/// RMS level of a 16-bit PCM WAV (sampled); huge for other sample widths so they never count as silence.
pub fn rms(path: &Path) -> f64 {
    let Ok(data) = std::fs::read(path) else { return 0.0 };
    if data.len() < 12 || &data[..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return 0.0;
    }
    let (mut i, mut width, mut pcm): (usize, u16, &[u8]) = (12, 0, &[]);
    while i + 8 <= data.len() {
        let id = &data[i..i + 4];
        let size = u32::from_le_bytes([data[i + 4], data[i + 5], data[i + 6], data[i + 7]]) as usize;
        let body = &data[i + 8..data.len().min(i + 8 + size)];
        if id == b"fmt " && body.len() >= 16 {
            width = u16::from_le_bytes([body[14], body[15]]) / 8;
        } else if id == b"data" {
            // a recorder killed before it finished the header leaves the size at 0 or ~4 GB: take what is there
            pcm = if size == 0 { &data[i + 8..] } else { body };
            break;
        }
        i += 8 + size + (size & 1);
    }
    if width != 2 {
        return 1e9;
    }
    let n = pcm.len() / 2;
    if n == 0 {
        return 0.0;
    }
    let step = (n / 64000).max(1);
    let (mut sum, mut cnt) = (0f64, 0usize);
    for k in (0..n).step_by(step) {
        let x = i16::from_le_bytes([pcm[2 * k], pcm[2 * k + 1]]) as f64;
        sum += x * x;
        cnt += 1;
    }
    (sum / cnt as f64).sqrt()
}

fn stop(cfg: &Config, pid: i32, t: f64) -> i32 {
    let (meta, wav) = paths();
    let d = &cfg.dictate;
    let _ = std::fs::remove_file(&meta);
    // signal the whole group: timeout + pw-record finish the WAV header on SIGINT
    unsafe { libc::killpg(pid, libc::SIGINT) };
    for _ in 0..50 {
        if !pid_alive(pid) {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let took = util::now() - t;
    let size = std::fs::metadata(&wav).map(|m| m.len()).unwrap_or(0);
    if took < 0.4 || size < 8000 {
        client::activity(None);
        ui::notify("Dictation", "Too short, nothing recorded", "normal", Some(2000));
        return 1;
    }
    let level = rms(&wav);
    if level < d.min_rms {
        client::activity(None);
        ui::notify("Dictation", &format!("Only silence (level {level:.0})"), "normal", Some(2500));
        return 1;
    }
    client::activity(Some("stt"));
    let t0 = Instant::now();
    let audio = std::fs::read(&wav).unwrap_or_default();
    let _ = std::fs::remove_file(&wav);
    let res = client::transcribe(&audio, &d.language, &d.prompt, 300.0);
    client::activity(None);
    let text = match res {
        Ok(t) => t,
        Err(e) => {
            ui::notify("Dictation failed", &e.0, "critical", None);
            return 1;
        }
    };
    if text.is_empty() {
        ui::notify("Dictation", "No speech recognised", "normal", Some(2500));
        return 1;
    }
    if d.copy {
        ui::copy(&text);
    }
    let typed = ui::type_text(&text, &strs(&d.type_cmd));
    let how = if typed { "typed + copied" } else { "copied (install wtype to type it)" };
    ui::notify(&format!("Dictation · {took:.0}s audio → {:.1}s on {}", t0.elapsed().as_secs_f64(), cfg.engine_label()),
               &format!("{}\n<i>{how}</i>", util::take_chars(&text, 200)), "normal", Some(4000));
    println!("{text}");
    0
}

pub fn main(action: &str) -> i32 {
    let cfg = config::load();
    let (meta, wav) = paths();
    // the recorder may have hit max_seconds: still stop()
    let info: Option<(i32, f64)> = std::fs::read_to_string(&meta).ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| Some((v.get("pid")?.as_i64()? as i32, v.get("t")?.as_f64()?)));
    match action {
        "cancel" => {
            if let Some((pid, _)) = info {
                let _ = std::fs::remove_file(&meta);
                unsafe { libc::killpg(pid, libc::SIGTERM) };
                let _ = std::fs::remove_file(&wav);
            }
            client::activity(None);
            0
        }
        "start" | "toggle" if action == "start" || info.is_none() => {
            if info.is_some_and(|(pid, _)| pid_alive(pid)) {
                return 0;
            }
            start(cfg)
        }
        _ => match info {
            None => {
                eprintln!("not recording");
                1
            }
            Some((pid, t)) => stop(cfg, pid, t),
        },
    }
}

pub fn file_main(path: &str, language: &str) -> Result<i32, client::ApiError> {
    let t0 = Instant::now();
    let audio = std::fs::read(path).map_err(|e| client::ApiError(format!("{path}: {e}")))?;
    println!("{}", client::transcribe(&audio, language, "", 300.0)?);
    eprintln!("[{:.1}s]", t0.elapsed().as_secs_f64());
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(samples: &[i16]) -> Vec<u8> {
        let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut v = b"RIFF".to_vec();
        v.extend((36 + data.len() as u32).to_le_bytes());
        v.extend(b"WAVEfmt ");
        v.extend(16u32.to_le_bytes());
        v.extend(1u16.to_le_bytes()); // PCM
        v.extend(1u16.to_le_bytes()); // mono
        v.extend(16000u32.to_le_bytes());
        v.extend(32000u32.to_le_bytes());
        v.extend(2u16.to_le_bytes());
        v.extend(16u16.to_le_bytes());
        v.extend(b"data");
        v.extend((data.len() as u32).to_le_bytes());
        v.extend(data);
        v
    }

    #[test]
    fn rms_of_pcm() {
        let p = std::env::temp_dir().join(format!("npukit-rms-{}.wav", std::process::id()));
        std::fs::write(&p, wav(&[300, -300, 300, -300])).unwrap();
        assert!((rms(&p) - 300.0).abs() < 1e-9);
        std::fs::write(&p, wav(&[0; 100])).unwrap();
        assert_eq!(rms(&p), 0.0);
        let _ = std::fs::remove_file(&p);
    }
}
