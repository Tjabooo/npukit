//! `npu ask` / `npu chat` with file access: the model calls the read-only tools in tools.rs in a loop until
//! it has what it needs, then answers. Without granted folders it is a plain chat.

use crate::client::{self, ApiError, ChatOpts};
use crate::config::{self, Config};
use crate::tools::{self, Files};
use crate::util;
use serde_json::{json, Value};
use std::io::IsTerminal;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const TRIMMED: &str = "[older tool result removed to save context; call the tool again if you still need it]";

fn note(text: &str) {
    if std::io::stderr().is_terminal() {
        eprintln!("\x1b[2m{text}\x1b[0m");
    } else {
        eprintln!("{text}");
    }
}

fn brief(args: &Value) -> String {
    let mut parts = vec![];
    for k in ["path", "pattern", "glob", "offset", "depth"] {
        let Some(v) = args.get(k) else { continue };
        let show = match v {
            Value::Null => continue,
            Value::String(s) if s.is_empty() => continue,
            Value::String(s) => s.clone(),
            Value::Number(n) if n.as_i64() == Some(1) && k != "path" => continue,
            v => v.to_string(),
        };
        parts.push(if matches!(k, "path" | "pattern" | "glob") { show } else { format!("{k}={show}") });
    }
    util::take_chars(&parts.join(" "), 120).to_string()
}

pub struct Session<'a> {
    cfg: &'a Config,
    pub files: Files,
    pub model: Option<String>,
    think: bool,
    max_tokens: u32,
    quiet: bool,
    tools_ok: bool,
    pub messages: Vec<Value>,
    pub tokens: u64,
    pub last_tps: Option<f64>,
}

impl<'a> Session<'a> {
    pub fn new(cfg: &'a Config, model: Option<String>, files: Files, think: bool, max_tokens: u32, quiet: bool) -> Session<'a> {
        let mut s = Session { cfg, files, model, think, max_tokens, quiet, tools_ok: true, messages: vec![], tokens: 0, last_tps: None };
        s.messages.push(json!({"role": "system", "content": s.system_prompt()}));
        s
    }

    fn system_prompt(&self) -> String {
        let host = util::hostname();
        let date = util::strftime("%Y-%m-%d", util::now());
        if self.files.roots.is_empty() {
            return format!("You are a helpful assistant running locally on {host}. Today is {date}.");
        }
        let roots: Vec<String> = self.files.roots.iter().map(|r| self.files.show(r)).collect();
        format!("You are a helpful assistant running locally on {host}. Today is {date}. You can read files \
                 (never change them) inside: {}. When a question is about files, folders or projects, look \
                 with the tools first and base the answer on what you actually read; mention the files you used. \
                 For a project: list_dir with depth 2, then read the README and the few files that matter. Read \
                 only what you need. Paths can be absolute, start with ~, or be relative to the first folder. \
                 Answer in the language the user writes in.", roots.join(", "))
    }

    pub fn grant(&mut self, path: &str) -> String {
        let before = self.files.roots.len();
        let mut roots: Vec<String> = self.files.roots.iter().map(|r| r.display().to_string()).collect();
        roots.push(path.to_string());
        self.files = Files::new(&roots, self.cfg);
        if self.files.roots.len() == before {
            return format!("not a folder (or already granted): {path}");
        }
        self.messages[0] = json!({"role": "system", "content": self.system_prompt()});
        let shown: Vec<String> = self.files.roots.iter().map(|r| self.files.show(r)).collect();
        format!("can now read {}", shown.join(", "))
    }

    fn call(&self, use_tools: bool) -> Result<Value, ApiError> {
        let mut extra = json!({"chat_template_kwargs": {"enable_thinking": self.think}});
        if use_tools {
            extra["tools"] = tools::schemas();
        }
        client::chat(&Value::Array(self.messages.clone()), self.model.as_deref(),
                     ChatOpts { max_tokens: self.max_tokens, timeout: 900.0, extra, ..Default::default() })
    }

    /// Keep tool results under [tools] context_chars: blank the oldest first.
    fn trim(&mut self) {
        let budget = self.cfg.tools.context_chars;
        let idx: Vec<usize> = self.messages.iter().enumerate().filter(|(_, m)| m["role"] == "tool").map(|(i, _)| i).collect();
        let len = |m: &Value| m["content"].as_str().map(|s| s.chars().count()).unwrap_or(0);
        let mut total: usize = idx.iter().map(|&i| len(&self.messages[i])).sum();
        for &i in idx.iter().take(idx.len().saturating_sub(1)) {
            if total <= budget {
                break;
            }
            if self.messages[i]["content"] != TRIMMED {
                total -= len(&self.messages[i]) - TRIMMED.chars().count().min(len(&self.messages[i]));
                self.messages[i]["content"] = json!(TRIMMED);
            }
        }
    }

    pub fn ask(&mut self, text: &str) -> Result<String, ApiError> {
        self.messages.push(json!({"role": "user", "content": text}));
        let steps = self.cfg.tools.max_steps;
        for step in 0..=steps {
            let use_tools = !self.files.roots.is_empty() && self.tools_ok && step < steps;
            let out = match self.call(use_tools) {
                Ok(o) => o,
                Err(e) if use_tools && e.0.to_lowercase().contains("tool") => {
                    // server without tool calls (older FastFlowLM)
                    self.tools_ok = false;
                    note("  (this model server can't call tools: answering without file access)");
                    continue;
                }
                Err(e) => return Err(e),
            };
            self.tokens += out.pointer("/usage/completion_tokens").and_then(Value::as_u64).unwrap_or(0);
            self.last_tps = client::tps(&out);
            let msg = out.pointer("/choices/0/message").cloned().unwrap_or(json!({}));
            let calls = msg.get("tool_calls").and_then(Value::as_array).cloned().unwrap_or_default();
            let content = msg.get("content").and_then(Value::as_str).unwrap_or("").to_string();
            let mut entry = json!({"role": "assistant", "content": content});
            if !calls.is_empty() {
                entry["tool_calls"] = json!(calls);
            }
            self.messages.push(entry);
            if calls.is_empty() {
                return Ok(content.trim().to_string());
            }
            for c in &calls {
                let f = c.get("function").cloned().unwrap_or(json!({}));
                let name = f.get("name").and_then(Value::as_str).unwrap_or("");
                let args: Value = match f.get("arguments") {
                    Some(Value::String(s)) => serde_json::from_str(s).unwrap_or(json!({})),
                    Some(v @ Value::Object(_)) => v.clone(),
                    _ => json!({}),
                };
                let args = if args.is_object() { args } else { json!({}) };
                let result = self.files.run(name, &args);
                if !self.quiet {
                    let first = result.lines().next().unwrap_or("");
                    let err = if first.starts_with("error") {
                        format!("  ({})", first.chars().skip(6).take(74).collect::<String>())
                    } else {
                        String::new()
                    };
                    note(&format!("  · {name} {}{err}", brief(&args)));
                }
                self.messages.push(json!({"role": "tool", "tool_call_id": c.get("id").and_then(Value::as_str).unwrap_or(""), "content": result}));
            }
            self.trim();
        }
        Ok(String::new())
    }
}

pub struct AskArgs {
    pub prompt: Vec<String>,
    pub model: Option<String>,
    pub think: bool,
    pub max_tokens: u32,
    pub read: Vec<String>,
    pub no_files: bool,
}

pub fn build<'a>(a: &AskArgs, cfg: &'a Config) -> Session<'a> {
    let roots: Vec<String> = if a.no_files { vec![] } else { cfg.tools.read.iter().chain(a.read.iter()).cloned().collect() };
    let files = Files::new(&roots, cfg);
    let model = a.model.clone().or_else(|| (!files.roots.is_empty()).then(|| cfg.tools.model.clone()));
    Session::new(cfg, model, files, a.think, a.max_tokens, false)
}

fn footer(s: &Session, t0: Instant) {
    let tps = s.last_tps.map(|t| format!(" · {t:.1} tok/s")).unwrap_or_default();
    note(&format!("[{:.1}s · {} tok{tps}]", t0.elapsed().as_secs_f64(), s.tokens));
}

pub fn read_stdin() -> String {
    let mut s = String::new();
    let _ = std::io::Read::read_to_string(&mut std::io::stdin(), &mut s);
    s
}

pub fn ask_main(a: &AskArgs) -> Result<i32, ApiError> {
    let q = if a.prompt.is_empty() { read_stdin() } else { a.prompt.join(" ") };
    let cfg = config::load();
    let mut s = build(a, cfg);
    let t0 = Instant::now();
    println!("{}", s.ask(&q)?);
    footer(&s, t0);
    Ok(0)
}

const HELP: &str = "/read PATH   let the model read another folder for this chat
/model NAME  switch model (fast, quality, vision or a tag)
/clear       forget the conversation
/exit        quit (also Ctrl-D)";

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

extern "C" fn on_sigint(_: libc::c_int) {
    INTERRUPTED.store(true, Ordering::SeqCst);
}

/// Ask on a worker thread so Ctrl-C can cut the request (the model server sees the disconnect).
fn ask_interruptible(s: Session<'static>, line: String) -> (Session<'static>, Option<Result<String, ApiError>>) {
    INTERRUPTED.store(false, Ordering::SeqCst);
    let h = std::thread::spawn(move || {
        let mut s = s;
        let r = s.ask(&line);
        (s, r)
    });
    let mut stopped = false;
    while !h.is_finished() {
        if INTERRUPTED.swap(false, Ordering::SeqCst) {
            stopped = true;
            client::cancel_current();
        }
        std::thread::sleep(Duration::from_millis(30));
    }
    let (s, r) = h.join().expect("chat thread");
    (s, if stopped { None } else { Some(r) })
}

pub fn chat_main(a: &AskArgs) -> Result<i32, ApiError> {
    let cfg: &'static Config = config::load();
    let mut s = build(a, cfg);
    let roots: Vec<String> = s.files.roots.iter().map(|r| s.files.show(r)).collect();
    let wher = if roots.is_empty() { "no folders (add [tools] read or --read)".to_string() } else { roots.join(", ") };
    note(&format!("npu chat · {} · can read: {wher} · /help", s.model.clone().unwrap_or_else(|| cfg.models.default.clone())));
    let mut first = a.prompt.join(" ").trim().to_string();
    let interactive = std::io::stdin().is_terminal();
    unsafe { libc::signal(libc::SIGINT, on_sigint as *const () as libc::sighandler_t) };
    let mut rl = if interactive { rustyline::DefaultEditor::new().ok() } else { None };
    loop {
        let line = if !first.is_empty() {
            std::mem::take(&mut first)
        } else if !interactive {
            return Ok(0);
        } else {
            let read = match rl.as_mut() {
                Some(r) => r.readline("› ").map_err(|_| ()),
                None => {
                    eprint!("› ");
                    let mut l = String::new();
                    match std::io::stdin().read_line(&mut l) {
                        Ok(0) | Err(_) => Err(()),
                        Ok(_) => Ok(l),
                    }
                }
            };
            match read {
                Ok(l) => {
                    if let Some(r) = rl.as_mut() {
                        let _ = r.add_history_entry(l.trim());
                    }
                    l.trim().to_string()
                }
                Err(()) => {
                    eprintln!();
                    return Ok(0);
                }
            }
        };
        if line.is_empty() {
            continue;
        }
        if line.starts_with('/') {
            let (cmd, rest) = line.split_once(' ').unwrap_or((&line, ""));
            let rest = rest.trim();
            match cmd {
                "/exit" | "/quit" | "/q" => return Ok(0),
                "/clear" => {
                    s.messages.truncate(1);
                    s.tokens = 0;
                    note("conversation cleared");
                }
                "/read" if !rest.is_empty() => note(&s.grant(rest)),
                "/model" if !rest.is_empty() => {
                    s.model = Some(rest.to_string());
                    note(&format!("model: {rest}"));
                }
                _ => note(HELP),
            }
            continue;
        }
        let t0 = Instant::now();
        let (back, r) = ask_interruptible(s, line);
        s = back;
        match r {
            None => {
                note("(stopped)");
                continue;
            }
            Some(Err(e)) => {
                note(&format!("error: {e}"));
                continue;
            }
            Some(Ok(answer)) => println!("{answer}"),
        }
        footer(&s, t0);
    }
}
