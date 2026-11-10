//! Desktop glue: notifications, clipboard, typing, rofi. Each helper degrades quietly.

use crate::util::which;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub fn notify(title: &str, body: &str, urgency: &str, timeout_ms: Option<u32>) {
    if which("notify-send").is_none() {
        return;
    }
    let mut c = Command::new("notify-send");
    c.args(["-a", "npukit", "-u", urgency, "-h", "string:x-dunst-stack-tag:npukit"]);
    if let Some(t) = timeout_ms {
        c.args(["-t", &t.to_string()]);
    }
    let _ = c.args([title, body]).stdin(Stdio::null()).status();
}

pub fn copy(text: &str) -> bool {
    if which("wl-copy").is_none() {
        return false;
    }
    // wl-copy forks a server that owns the selection; don't wait on its pipes
    let _ = Command::new("wl-copy").args(["--", text])
        .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).status();
    true
}

/// Run `cmd` with `input` on stdin; (exit code, stdout). Err when it can't start.
pub fn pipe(cmd: &[String], input: &[u8]) -> std::io::Result<(i32, Vec<u8>)> {
    let mut child = Command::new(&cmd[0]).args(&cmd[1..])
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn()?;
    let mut stdin = child.stdin.take().unwrap();
    let data = input.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&data);
    });
    let out = child.wait_with_output()?;
    let _ = writer.join();
    Ok((out.status.code().unwrap_or(-1), out.stdout))
}

pub fn type_text(text: &str, cmd: &[String]) -> bool {
    if cmd.is_empty() || which(&cmd[0]).is_none() {
        return false;
    }
    matches!(pipe(cmd, text.as_bytes()), Ok((0, _)))
}

/// rofi -dmenu: (exit code, selected index or typed text).
pub fn rofi(lines: &[String], prompt: &str, theme: Option<&str>, mesg: Option<&str>, filter_input: bool) -> (i32, String) {
    let mut args: Vec<String> = if filter_input {
        vec!["rofi".into(), "-dmenu".into(), "-p".into(), prompt.into(), "-format".into(), "s".into()]
    } else {
        vec!["rofi".into(), "-dmenu".into(), "-i".into(), "-p".into(), prompt.into(), "-format".into(), "i".into()]
    };
    if let Some(t) = theme {
        args.extend(["-theme".into(), t.into()]);
    }
    if let Some(m) = mesg {
        args.extend(["-mesg".into(), m.into()]);
    }
    rofi_args(&args, lines)
}

pub fn rofi_args(args: &[String], lines: &[String]) -> (i32, String) {
    match pipe(args, lines.join("\n").as_bytes()) {
        Ok((code, out)) => (code, String::from_utf8_lossy(&out).trim().to_string()),
        Err(_) => (-1, String::new()),
    }
}

/// Block until the menu is off screen, so a frozen-screen region pick doesn't capture it.
///
/// The compositor fades a closed layer out after rofi exits; wait for the layer to leave
/// `hyprctl layers` (when Hyprland is there) and then for the fade itself.
pub fn wait_menu_gone(settle: f64, namespace: &str) {
    if settle <= 0.0 {
        return;
    }
    let end = Instant::now() + Duration::from_millis(1500);
    while Instant::now() < end {
        let Ok(out) = Command::new("hyprctl").args(["layers", "-j"]).stderr(Stdio::null()).output() else { break };
        let Ok(mons) = serde_json::from_slice::<serde_json::Value>(&out.stdout) else { break };
        let present = mons.as_object().into_iter().flat_map(|m| m.values())
            .filter_map(|m| m.get("levels").and_then(|l| l.as_object()))
            .flat_map(|l| l.values())
            .filter_map(|lv| lv.as_array())
            .flatten()
            .any(|l| l.get("namespace").and_then(|n| n.as_str()) == Some(namespace));
        if !present {
            break;
        }
        std::thread::sleep(Duration::from_millis(30));
    }
    std::thread::sleep(Duration::from_secs_f64(settle));
}
