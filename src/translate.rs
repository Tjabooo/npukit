//! `npu translate`: an eww overlay to translate a message you write.
//!
//! Pick a language (recent ones first, or type any language), write the message, press Enter: the
//! translation shows up and lands on the clipboard. Enter again on the same text closes the overlay and
//! types the translation into the window you were in (wtype), so it works as a "write in my language,
//! send in theirs" box for chats.
//!
//! The overlay runs its own eww daemon (config written to $XDG_RUNTIME_DIR/npukit/eww, colours from
//! [translate.colors]), so it needs no changes to an existing eww setup. Each action the overlay takes
//! (`--run`, `--draft`, `--lang`, `--chip`) is a short-lived `npu translate` process that updates a small
//! state file and pushes it to eww.

use crate::client::{self, ApiError};
use crate::config::{self, strs, Config};
use crate::ocr;
use crate::ui;
use crate::util::{self, sh_quote, which};
use serde_json::{json, Value};
use std::io::{IsTerminal, Read};
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

const WINDOW: &str = "npu-translate";
// text typed into the overlay reaches the shell through a quoted here-document: eww pastes the input's
// value into the command unescaped, so it must never be part of the command line itself
const HEREDOC_END: &str = "NPUKIT_TRANSLATE_END_7f3a9c";
pub const MAX_CHIPS: usize = 8;
const PALETTE: [&str; 8] = ["bg", "panel", "line", "fg", "dim", "accent", "accent2", "error"];
const YUCK: &str = include_str!("../assets/eww/translate.yuck");
const SCSS: &str = include_str!("../assets/eww/translate.scss");

type TranslateFn = Box<dyn Fn(&str, &str, &str, &str) -> Result<(String, f64), ApiError>>;
type EwwFn = Box<dyn Fn(&Config, &[&str]) -> (i32, String)>;
type HookFn = Box<dyn Fn(&Config, &str)>;
type ActivityFn = Box<dyn Fn(Option<&str>)>;
type CopyFn = Box<dyn Fn(&str) -> bool>;

/// What the tests replace: the translator, eww, the open/close commands, the bar and the clipboard.
pub struct Env {
    pub translate: TranslateFn,
    pub eww: EwwFn,
    pub hook: HookFn,
    pub activity: ActivityFn,
    pub copy: CopyFn,
}

impl Default for Env {
    fn default() -> Self {
        Env {
            translate: Box::new(|text, lang, prompt, model| ocr::translate(text, lang, prompt, model, "")),
            eww: Box::new(|cfg, args| eww(cfg, args, 5.0)),
            hook: Box::new(hook),
            activity: Box::new(client::activity),
            copy: Box::new(ui::copy),
        }
    }
}

// ── state ────────────────────────────────────────────────────────────────────

fn state_file() -> PathBuf {
    config::runtime_dir().join("translate.json")
}

fn draft_file() -> PathBuf {
    config::runtime_dir().join("translate-draft.txt")
}

struct Lock(std::fs::File);

fn locked() -> Lock {
    let f = std::fs::File::create(config::runtime_dir().join("translate.lock")).expect("translate lock");
    unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX) };
    Lock(f)
}

impl Drop for Lock {
    fn drop(&mut self) {
        unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN) };
    }
}

pub fn load_state() -> Value {
    std::fs::read_to_string(state_file()).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .filter(Value::is_object).unwrap_or_else(|| json!({}))
}

pub fn save_state(st: &Value) {
    let _ = util::write_atomic(&state_file(), st.to_string().as_bytes());
}

fn s<'a>(st: &'a Value, k: &str) -> &'a str {
    st.get(k).and_then(Value::as_str).unwrap_or("")
}

fn b(st: &Value, k: &str) -> bool {
    st.get(k).and_then(Value::as_bool).unwrap_or(false)
}

fn seq(st: &Value) -> i64 {
    st.get("seq").and_then(Value::as_i64).unwrap_or(0)
}

fn chip_names(st: &Value) -> Vec<String> {
    st.get("chips").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default()
}

/// Languages offered as one-click chips: the current one, recent ones, then the configured list.
pub fn chips(recent: &[String], configured: &[String], current: &str) -> Vec<String> {
    let mut out: Vec<String> = vec![];
    for lang in std::iter::once(current).chain(recent.iter().map(String::as_str)).chain(configured.iter().map(String::as_str)) {
        let lang = if lang.is_empty() { String::new() } else { ocr::language(lang) };
        if !lang.is_empty() && !out.iter().any(|x| x.to_lowercase() == lang.to_lowercase()) {
            out.push(lang);
        }
    }
    out.truncate(MAX_CHIPS);
    out
}

pub fn fresh_state(cfg: &Config, old: &Value) -> Value {
    let t = &cfg.translate;
    let recent = ocr::recent_languages();
    let lang = if !s(old, "lang").is_empty() { s(old, "lang").to_string() }
        else if let Some(r) = recent.first().filter(|r| !r.is_empty()) { r.clone() }
        else { ocr::language(if t.to.is_empty() { &cfg.ocr.translate_to } else { &t.to }) };
    json!({"open": true, "lang": lang, "chips": chips(&recent, &t.languages, &lang),
           "busy": false, "seq": seq(old), "src": "", "src_lang": "", "result": "",
           "status": "", "error": false, "engine": cfg.engine_label()})
}

/// What the yuck reads (keeps the variable small and free of internals).
pub fn view(st: &Value) -> Value {
    let lang = s(st, "lang");
    json!({"open": b(st, "open"), "lang": lang, "busy": b(st, "busy"),
           "chips": chip_names(st).iter().enumerate().map(|(i, c)| json!({"i": i, "name": c, "on": c.to_lowercase() == lang.to_lowercase()})).collect::<Vec<_>>(),
           "result": s(st, "result"), "status": s(st, "status"), "error": b(st, "error"), "engine": s(st, "engine")})
}

/// Enter on text that is already translated (same text, same language) = send it.
pub fn wants_send(st: &Value, text: &str) -> bool {
    !s(st, "result").is_empty() && !b(st, "busy") && !b(st, "error") && s(st, "src") == text
        && s(st, "src_lang").to_lowercase() == s(st, "lang").to_lowercase()
}

// ── eww ──────────────────────────────────────────────────────────────────────

fn eww_dir() -> PathBuf {
    config::runtime_dir().join("eww")
}

fn eww_cmd(cfg: &Config, args: &[&str]) -> Vec<String> {
    let mut v = vec![cfg.translate.eww.clone(), "-c".into(), eww_dir().display().to_string()];
    v.extend(args.iter().map(|a| a.to_string()));
    v
}

fn eww(cfg: &Config, args: &[&str], timeout: f64) -> (i32, String) {
    let cmd = eww_cmd(cfg, args);
    let child = Command::new(&cmd[0]).args(&cmd[1..]).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn();
    let Ok(mut child) = child else { return (1, "eww not found".into()) };
    let end = std::time::Instant::now() + Duration::from_secs_f64(timeout);
    loop {
        match child.try_wait() {
            Ok(Some(st)) => {
                let mut out = String::new();
                if let Some(mut o) = child.stdout.take() {
                    let _ = o.read_to_string(&mut out);
                }
                return (st.code().unwrap_or(1), out.trim().to_string());
            }
            Ok(None) if std::time::Instant::now() < end => std::thread::sleep(Duration::from_millis(5)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return (1, "timed out".into());
            }
        }
    }
}

/// How the overlay calls back into npukit (this same binary).
fn npu_cmd() -> String {
    let exe = std::env::current_exe().map(|p| p.display().to_string()).unwrap_or_else(|_| "npu".into());
    format!("{} translate", sh_quote(&exe))
}

/// Shell for an eww input handler: the input's value ({}) goes in on stdin, never on the command line.
/// Backgrounded because eww kills handlers that outlive their :timeout.
pub fn heredoc_action(npu: &str, action: &str) -> String {
    format!("{npu} {action} <<'{HEREDOC_END}' >/dev/null 2>&1 &\n{{}}\n{HEREDOC_END}")
}

/// Every keystroke saves the draft (chips translate it): plain sh, no process start of ours per key.
pub fn draft_action(path: &std::path::Path) -> String {
    let f = sh_quote(&path.display().to_string());
    let tmp = sh_quote(&format!("{}.tmp", path.display()));
    format!("cat > {tmp} <<'{HEREDOC_END}' && mv -f {tmp} {f}\n{{}}\n{HEREDOC_END}")
}

fn yuck_str(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

pub fn render_yuck() -> String {
    let npu = npu_cmd();
    let subs = [
        ("@ON_ACCEPT@", yuck_str(&heredoc_action(&npu, "--run"))),
        ("@ON_CHANGE@", yuck_str(&draft_action(&draft_file()))),
        ("@ON_LANG@", yuck_str(&heredoc_action(&npu, "--lang"))),
        ("@ON_CHIP@", yuck_str(&format!("{npu} --chip ${{c.i}} >/dev/null 2>&1 &"))),
        ("@ON_CLOSE@", yuck_str(&format!("{npu} --close >/dev/null 2>&1 &"))),
        ("@INITIAL@", format!("'{}'", view(&json!({})))),
    ];
    let mut tpl = YUCK.to_string();
    for (k, v) in subs {
        tpl = tpl.replace(k, &v);
    }
    tpl
}

pub fn render_scss(cfg: &Config) -> String {
    let t = &cfg.translate;
    let mut head: Vec<String> = PALETTE.iter().filter_map(|k| t.colors.get(*k).map(|c| format!("${k}: {c};"))).collect();
    head.push(format!("$font: \"{}\", monospace;", t.font));
    head.push(format!("$width: {}px;", t.width as i64));
    head.join("\n") + "\n" + SCSS
}

/// Write the overlay's eww config; true when it changed (a running daemon then reloads it).
fn write_eww_config(cfg: &Config) -> bool {
    let d = eww_dir();
    let _ = std::fs::create_dir_all(&d);
    let mut changed = false;
    for (name, text) in [("eww.yuck", render_yuck()), ("eww.scss", render_scss(cfg))] {
        let f = d.join(name);
        if std::fs::read_to_string(&f).ok().as_deref() != Some(text.as_str()) {
            // eww reloads on change: don't touch it otherwise
            let _ = std::fs::write(&f, text);
            changed = true;
        }
    }
    changed
}

fn push(env: &Env, cfg: &Config, st: &Value) {
    (env.eww)(cfg, &["update", &format!("tr={}", view(st))]);
}

fn ensure_daemon(cfg: &Config) -> bool {
    if eww(cfg, &["ping"], 2.0).1 == "pong" {
        return true;
    }
    use std::os::unix::process::CommandExt;
    let cmd = eww_cmd(cfg, &["daemon"]);
    let _ = Command::new(&cmd[0]).args(&cmd[1..]).process_group(0)
        .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn();
    for _ in 0..50 {
        std::thread::sleep(Duration::from_millis(100));
        if eww(cfg, &["ping"], 2.0).1 == "pong" {
            return true;
        }
    }
    false
}

fn is_open(cfg: &Config) -> bool {
    let (code, out) = eww(cfg, &["active-windows"], 2.0);
    code == 0 && out.lines().any(|l| l.split(':').next().unwrap_or("").trim() == WINDOW)
}

fn focused_monitor() -> Vec<String> {
    let Ok(out) = Command::new("hyprctl").args(["monitors", "-j"]).stderr(Stdio::null()).output() else { return vec![] };
    let Ok(Value::Array(mons)) = serde_json::from_slice::<Value>(&out.stdout) else { return vec![] };
    mons.iter().find(|m| m.get("focused").and_then(Value::as_bool) == Some(true))
        .map(|m| ["name", "model"].iter().filter_map(|k| m.get(*k).and_then(Value::as_str)).filter(|s| !s.is_empty()).map(str::to_string).collect())
        .unwrap_or_default()
}

fn hook(cfg: &Config, key: &str) {
    let cmd = strs(if key == "on_open" { &cfg.translate.on_open } else { &cfg.translate.on_close });
    if !cmd.is_empty() && which(&cmd[0]).is_some() {
        let _ = Command::new(&cmd[0]).args(&cmd[1..]).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).status();
    }
}

// ── actions ──────────────────────────────────────────────────────────────────

fn open_overlay(env: &Env, cfg: &Config) -> i32 {
    if which(&cfg.translate.eww).is_none() {
        ui::notify("Translate", "The translate overlay needs eww (e.g. pacman -S eww)", "critical", None);
        return 1;
    }
    let changed = write_eww_config(cfg);
    let logs = eww_cmd(cfg, &["logs"]).join(" ");
    if !ensure_daemon(cfg) {
        ui::notify("Translate", &format!("eww didn't start; try: {logs}"), "critical", None);
        return 1;
    }
    let st = {
        let _l = locked();
        let st = fresh_state(cfg, &load_state());
        save_state(&st);
        st
    };
    let _ = std::fs::remove_file(draft_file());
    if changed {
        std::thread::sleep(Duration::from_millis(400)); // a reload resets the variable: let it finish before the state goes in
    }
    push(env, cfg, &st);
    let mons = focused_monitor();
    for mon in mons.iter().map(Some).chain([None]) {
        let mut args = vec!["open", WINDOW];
        if let Some(m) = mon {
            args.extend(["--screen", m.as_str()]);
        }
        if (env.eww)(cfg, &args).0 == 0 {
            (env.hook)(cfg, "on_open");
            return 0;
        }
    }
    ui::notify("Translate", &format!("eww couldn't open its window; try: {logs}"), "critical", None);
    1
}

fn close_overlay(env: &Env, cfg: &Config) {
    {
        let _l = locked();
        let mut st = load_state();
        st["open"] = json!(false);
        st["seq"] = json!(seq(&st) + 1); // a translation still running is no longer shown
        save_state(&st);
    }
    (env.eww)(cfg, &["close", WINDOW]);
    (env.hook)(cfg, "on_close");
}

/// Close, give the focus back to the window underneath, type the translation there.
fn send(env: &Env, cfg: &Config, text: &str) -> i32 {
    close_overlay(env, cfg);
    let typer = strs(&cfg.translate.type_cmd);
    if (env.copy)(text) && typer.is_empty() {
        return 0;
    }
    std::thread::sleep(Duration::from_secs_f64(cfg.translate.type_delay.max(0.0)));
    if !ui::type_text(text, &typer) {
        let hint = if which("wtype").is_some() { "" } else { " (install wtype to have it typed)" };
        ui::notify("Translation copied", &format!("Paste it with Ctrl+V{hint}"), "normal", Some(3000));
    }
    0
}

/// Translate `text` into the chosen language and show it; the same text again = send it.
pub fn run(env: &Env, cfg: &Config, text: &str, lang: Option<&str>) -> i32 {
    let mut text = text.trim().to_string();
    let (st, my_seq, target) = {
        let _l = locked();
        let mut st = load_state();
        if let Some(lang) = lang {
            let same = !s(&st, "result").is_empty() && s(&st, "src") == text && s(&st, "src_lang").to_lowercase() == lang.to_lowercase();
            st["lang"] = json!(lang);
            let have = chip_names(&st);
            if !have.iter().any(|c| c.to_lowercase() == lang.to_lowercase()) {
                st["chips"] = json!(chips(&[], &have, lang)); // a typed language joins the chips up front
            }
            if same {
                text.clear(); // already showing this translation
            }
        }
        if text.is_empty() {
            save_state(&st);
            drop(_l);
            push(env, cfg, &st);
            return 0;
        }
        if lang.is_none() && wants_send(&st, &text) {
            let result = s(&st, "result").to_string();
            save_state(&st);
            drop(_l);
            return send(env, cfg, &result);
        }
        let n = seq(&st) + 1;
        let target = s(&st, "lang").to_string();
        st["seq"] = json!(n);
        st["busy"] = json!(true);
        st["error"] = json!(false);
        st["status"] = json!(format!("translating into {target}…"));
        st["src"] = json!(text);
        st["src_lang"] = json!(target);
        st["result"] = json!("");
        save_state(&st);
        (st, n, target)
    };
    push(env, cfg, &st);

    let t = &cfg.translate;
    let model = if t.model.is_empty() { cfg.ocr.translate_model.clone() } else { t.model.clone() };
    (env.activity)(Some("translate"));
    let res = (env.translate)(&text, &target, &t.prompt, &model);
    (env.activity)(None);

    let st = {
        let _l = locked();
        let mut st = load_state();
        if seq(&st) != my_seq {
            return 0; // superseded (language changed, new text, or closed)
        }
        match &res {
            Ok((out, took)) if !out.is_empty() => {
                let copied = t.copy && (env.copy)(out);
                ocr::remember_language(&target);
                let engine = if s(&st, "engine").is_empty() { cfg.engine_label() } else { s(&st, "engine").to_string() };
                st["busy"] = json!(false);
                st["error"] = json!(false);
                st["result"] = json!(out);
                st["status"] = json!(format!("{took:.1}s on {engine}{}{}", if copied { " · copied" } else { "" },
                                             if t.type_cmd.is_empty() { "" } else { " · Enter again types it" }));
            }
            other => {
                let msg = match other { Err(e) => e.0.clone(), _ => "the model returned nothing".into() };
                st["busy"] = json!(false);
                st["error"] = json!(true);
                st["result"] = json!("");
                st["status"] = json!(msg);
            }
        }
        save_state(&st);
        st
    };
    push(env, cfg, &st);
    if res.is_err() { 1 } else { 0 }
}

fn stdin_text() -> String {
    let mut data = String::new();
    if !std::io::stdin().is_terminal() {
        let _ = std::io::stdin().read_to_string(&mut data);
    }
    if data.ends_with('\n') {
        data.pop();
    }
    data
}

fn draft() -> String {
    std::fs::read_to_string(draft_file()).unwrap_or_default()
}

pub struct TranslateArgs {
    pub text: Vec<String>,
    pub to: Option<String>,
    pub close: bool,
    pub run: bool,
    pub draft: bool,
    pub lang: bool,
    pub chip: Option<usize>,
}

pub fn main(a: &TranslateArgs) -> i32 {
    let cfg = config::load();
    let env = Env::default();
    if a.close {
        close_overlay(&env, cfg);
        return 0;
    }
    if a.draft {
        let _ = std::fs::write(draft_file(), stdin_text());
        return 0;
    }
    if a.run {
        let text = stdin_text();
        let _ = std::fs::write(draft_file(), &text);
        return run(&env, cfg, &text, None);
    }
    if a.lang {
        let lang = ocr::language(&stdin_text());
        return if lang.is_empty() { 0 } else { run(&env, cfg, &draft(), Some(&lang)) };
    }
    if let Some(i) = a.chip {
        return match chip_names(&load_state()).get(i) {
            Some(name) => run(&env, cfg, &draft(), Some(&name.clone())),
            None => 1,
        };
    }
    if !a.text.is_empty() {
        // one-shot, no overlay: npu translate -t sv "hello"
        let t = &cfg.translate;
        let to = a.to.clone().filter(|s| !s.is_empty()).unwrap_or_else(|| if t.to.is_empty() { cfg.ocr.translate_to.clone() } else { t.to.clone() });
        let lang = ocr::language(&to);
        let model = if t.model.is_empty() { &cfg.ocr.translate_model } else { &t.model };
        return match ocr::translate(&a.text.join(" "), &lang, &t.prompt, model, "") {
            Ok((out, took)) => {
                println!("{out}");
                eprintln!("[{took:.1}s → {lang}]");
                0
            }
            Err(e) => {
                eprintln!("translate failed: {e}");
                1
            }
        };
    }
    if is_open(cfg) {
        // the bind toggles
        close_overlay(&env, cfg);
        return 0;
    }
    open_overlay(&env, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn chips_current_then_recent_then_configured() {
        let c = chips(&["sv".into(), "German".into()], &["English".into(), "swedish".into(), "French".into()], "de");
        assert_eq!(c, ["German", "Swedish", "English", "French"]);
        let many: Vec<String> = (0..20).map(|i| format!("L{i}")).collect();
        assert_eq!(chips(&[], &many, "").len(), MAX_CHIPS);
    }

    #[test]
    fn hostile_text_reaches_stdin_verbatim() {
        // eww pastes the input's value into the command unescaped; the here-document keeps it inert
        let dir = std::env::temp_dir().join(format!("npukit-heredoc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("got");
        let nasty = "it's \"$(touch PWNED)\" `touch PWNED` $HOME; rm -rf / & {0} \\n";
        for action in [heredoc_action("tee", &out.display().to_string()), draft_action(&out)] {
            let _ = std::fs::remove_file(&out);
            let st = Command::new("/bin/sh").arg("-c").arg(action.replace("{}", nasty)).current_dir(&dir).status().unwrap();
            assert!(st.success());
            for _ in 0..50 {
                if std::fs::read_to_string(&out).map(|s| !s.is_empty()).unwrap_or(false) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            assert_eq!(std::fs::read_to_string(&out).unwrap(), format!("{nasty}\n"));
            assert!(!dir.join("PWNED").exists());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn render_has_no_placeholders_and_uses_palette() {
        let mut cfg = crate::config::from_str("").unwrap();
        cfg.translate.colors.insert("accent".into(), "#123456".into());
        let y = render_yuck();
        assert!(!regex::Regex::new(r"@[A-Z_]+@").unwrap().is_match(&y));
        assert!(render_scss(&cfg).contains("$accent: #123456;"));
    }

    /// One test owns the process env (XDG dirs) so nothing else races it.
    #[test]
    fn enter_translates_then_enter_again_sends() {
        let tmp = std::env::temp_dir().join(format!("npukit-tr-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::env::set_var("XDG_STATE_HOME", tmp.join("state"));
        std::env::set_var("XDG_RUNTIME_DIR", tmp.join("run"));
        let typed = tmp.join("typed");
        let mut cfg = crate::config::from_str(&format!(
            "[translate]\ncopy = false\neww = \"true\"\ntype_delay = 0\ntype = [\"sh\", \"-c\", \"cat > {}\"]\n", typed.display())).unwrap();
        cfg.translate.languages = vec!["English".into()];
        let calls: Arc<Mutex<Vec<(String, String)>>> = Arc::default();
        let c2 = calls.clone();
        let env = Env {
            translate: Box::new(move |text, lang, _, _| {
                c2.lock().unwrap().push((text.into(), lang.into()));
                Ok((format!("[{lang}] {text}"), 0.1))
            }),
            eww: Box::new(|_, _| (0, String::new())),
            hook: Box::new(|_, _| {}),
            activity: Box::new(|_| {}),
            copy: Box::new(|_| false),
        };
        let mut st = fresh_state(&cfg, &json!({}));
        st["lang"] = json!("German");
        save_state(&st);
        assert_eq!(run(&env, &cfg, "hej", None), 0);
        assert_eq!(s(&load_state(), "result"), "[German] hej");
        run(&env, &cfg, "hej", Some("French")); // chip: same text, new language → translate again
        run(&env, &cfg, "hej", Some("French")); // same chip again: nothing new
        assert_eq!(*calls.lock().unwrap(), [("hej".to_string(), "German".to_string()), ("hej".into(), "French".into())]);
        assert!(chip_names(&load_state()).contains(&"French".to_string()));
        run(&env, &cfg, "hej", None); // Enter again on the same text → type it, close
        assert_eq!(std::fs::read_to_string(&typed).unwrap(), "[French] hej");
        assert!(!b(&load_state(), "open"));
        assert_eq!(calls.lock().unwrap().len(), 2);

        // recent languages: newest first, no duplicates (case-insensitive)
        let _ = std::fs::remove_file(config::state_dir().join("npukit/translate-recent.json"));
        ocr::remember_language("German");
        ocr::remember_language("Klingon");
        ocr::remember_language("german");
        assert_eq!(ocr::recent_languages(), ["german", "Klingon"]);
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
