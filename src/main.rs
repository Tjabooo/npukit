//! `npu`: one command for the gateway and the tools built on it.

mod agent;
mod bench;
mod client;
mod config;
mod dictate;
mod gateway;
mod hw;
mod ocr;
mod search;
mod tools;
mod translate;
mod ui;
mod util;

use clap::{Parser, Subcommand};
use client::ApiError;
use config::Backend;
use serde_json::{json, Value};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "npu", version, about = "Everyday local AI on the AMD XDNA2 NPU (FastFlowLM), or on a GPU with llama.cpp + whisper.cpp.")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// gateway, loaded models, NPU and index state
    Status {
        #[arg(long)]
        json: bool,
    },
    /// aliases and installed models
    Models,
    /// load a model now: fast | quality | TAG | stt | embed
    Load {
        #[arg(default_value = "fast")]
        model: String,
        #[arg(long, value_parser = ["powersaver", "balanced", "performance", "turbo"])]
        pmode: Option<String>,
    },
    /// unload everything
    Stop,
    /// one-shot question to a local model (reads granted folders when asked)
    Ask(AskOpts),
    /// conversation with a local model that can read granted folders
    Chat(AskOpts),
    /// tokens/s + package/system power
    Bench {
        /// models or aliases (default: fast)
        models: Vec<String>,
        /// MODEL[@backend] on another server, e.g. Qwen3.5-4B-GGUF@vulkan
        #[arg(long, num_args = 0..)]
        other: Vec<String>,
        /// its base URL (default: Lemonade)
        #[arg(long, default_value = "http://127.0.0.1:13305")]
        other_url: String,
        /// comma list, e.g. powersaver,balanced,performance
        #[arg(long)]
        pmode: Option<String>,
        #[arg(long, default_value_t = 256)]
        tokens: u32,
        #[arg(long, default_value_t = 2)]
        runs: u32,
        #[arg(long)]
        json: bool,
    },
    /// toggle mic recording → Whisper (NPU/GPU) → type + copy
    Dictate {
        #[arg(default_value = "toggle", value_parser = ["toggle", "start", "stop", "cancel"])]
        action: String,
    },
    /// transcribe an audio file (16 kHz WAV works best)
    Transcribe {
        file: String,
        #[arg(long, default_value = "")]
        language: String,
    },
    /// screen region → text on the clipboard
    Ocr {
        /// describe the region instead
        #[arg(long)]
        describe: bool,
        /// custom question about the region
        #[arg(long)]
        ask: Option<String>,
        /// translate the text (default: [ocr] translate_to); LANG = name or code like sv
        #[arg(short = 't', long, num_args = 0..=1, default_missing_value = "", value_name = "LANG")]
        translate: Option<String>,
        /// translate, choosing the language from a searchable list first
        #[arg(short = 'p', long)]
        pick: bool,
        /// read this PNG instead of capturing
        #[arg(long)]
        file: Option<String>,
        /// copy even with --file
        #[arg(long)]
        copy: bool,
        #[arg(short = 'm', long)]
        model: Option<String>,
    },
    /// eww overlay: pick a language, write a message, Enter translates
    Translate {
        /// translate this and print it instead of opening the overlay
        text: Vec<String>,
        /// language for the one-shot form (names or codes: sv, de, ja, ...)
        #[arg(short = 't', long)]
        to: Option<String>,
        /// close the overlay
        #[arg(long)]
        close: bool,
        #[arg(long, hide = true)]
        run: bool,
        #[arg(long, hide = true)]
        draft: bool,
        #[arg(long, hide = true)]
        lang: bool,
        #[arg(long, hide = true)]
        chip: Option<usize>,
    },
    /// update the semantic search index
    Index {
        /// re-embed everything
        #[arg(long)]
        full: bool,
        #[arg(short, long)]
        quiet: bool,
    },
    /// semantic search over notes + code
    Search {
        query: Vec<String>,
        #[arg(short = 'n')]
        n: Option<usize>,
        #[arg(long)]
        rofi: bool,
        #[arg(long)]
        theme: Option<String>,
    },
    /// rofi menu of everything above
    Menu {
        #[arg(long)]
        theme: Option<String>,
    },
    /// run the gateway (normally started by npud.socket)
    Serve,
}

#[derive(clap::Args)]
struct AskOpts {
    prompt: Vec<String>,
    #[arg(short = 'm', long)]
    model: Option<String>,
    /// let reasoning models think first
    #[arg(long)]
    think: bool,
    #[arg(long)]
    max_tokens: Option<u32>,
    /// also let it read this folder (repeatable); [tools] read in the config is permanent
    #[arg(short = 'r', long = "read", value_name = "PATH")]
    read: Vec<String>,
    /// no file access, even with [tools] read set
    #[arg(long)]
    no_files: bool,
}

impl AskOpts {
    fn args(&self, default_tokens: u32) -> agent::AskArgs {
        agent::AskArgs {
            prompt: self.prompt.clone(), model: self.model.clone(), think: self.think,
            max_tokens: self.max_tokens.unwrap_or(default_tokens), read: self.read.clone(), no_files: self.no_files,
        }
    }
}

fn cmd_status(json_out: bool) -> Result<i32, ApiError> {
    let cfg = config::load();
    let st = client::call("/npu/status", None, 5.0).unwrap_or_else(|e| json!({
        "state": "gateway down", "error": e.0, "npu": hw::npu_info(),
        "power": if hw::on_ac(&hw::ac_online_path(&cfg.power.ac_online)) { "ac" } else { "battery" },
    }));
    if json_out {
        println!("{}", serde_json::to_string_pretty(&st).unwrap());
        return Ok(0);
    }
    let s = |k: &str| st.get(k).and_then(Value::as_str).unwrap_or("").to_string();
    let act = s("activity");
    println!("gateway   {}  ·  {}{}", cfg.base_url(), s("state"), if act.is_empty() { String::new() } else { format!(" ({act})") });
    let comps: Vec<String> = st.get("components").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).map(|c| match c {
        "chat" => s("chat_model"), "asr" => "whisper".into(), _ => "embeddings".into(),
    }).collect()).unwrap_or_default();
    if comps.is_empty() {
        println!("loaded    nothing");
    } else {
        let since = util::now() - st.get("loaded_since").and_then(Value::as_f64).unwrap_or_else(util::now);
        let pids: Vec<String> = st.get("procs").and_then(Value::as_array).map(|a| a.iter()
            .filter_map(|p| p.get("pid").and_then(Value::as_u64)).map(|p| p.to_string()).collect()).unwrap_or_default();
        let pmode = s("pmode");
        println!("loaded    {}  ·  {}up {since:.0}s  ·  pid {}", comps.join(", "),
                 if pmode.is_empty() { String::new() } else { format!("pmode {pmode}  ·  ") }, pids.join(","));
    }
    if let Some(t) = st.get("idle_unload").and_then(Value::as_f64) {
        println!("power     {}  ·  unload after {t:.0}s idle", s("power"));
    }
    let footer = |st: &Value| {
        if let Some(t) = st.get("last_tps").and_then(Value::as_f64) {
            println!("last      {t:.1} tok/s");
        }
        let err = st.get("last_error").and_then(Value::as_str).unwrap_or("");
        if !err.is_empty() {
            println!("error     {err}");
        }
    };
    if cfg.backend() == Backend::LlamaCpp {
        let lc = &cfg.llamacpp;
        for (name, bin) in [("llama", &lc.server), ("whisper", &lc.whisper)] {
            match util::which(&config::expand_str(bin)) {
                Some(p) => println!("{name:<9} {}", p.display()),
                None => println!("{name:<9} not found: {bin}"),
            }
        }
        let wm = config::expand(&lc.whisper_model);
        println!("stt       {}{}", if wm.exists() { "" } else { "missing: " }, wm.display());
        let gb = 1e9;
        for p in st.get("procs").and_then(Value::as_array).cloned().unwrap_or_default() {
            let what = p.get("model").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| {
                match p.pointer("/components/0").and_then(Value::as_str) { Some("asr") => "whisper".into(), Some("embed") => "embeddings".into(), _ => "?".into() }
            });
            let note = p.get("note").and_then(Value::as_str).unwrap_or("");
            let wher = if note.is_empty() { format!("{:.1} GiB VRAM", p.get("vram_mb").and_then(Value::as_f64).unwrap_or(0.0) / 1024.0) } else { format!("on the {note}") };
            println!("  {what:<18} pid {}  ·  {wher}", p.get("pid").map(|x| x.to_string()).unwrap_or_default());
        }
        let frac = lc.vram_reserve;
        if let Some(v) = hw::gpu_vram() {
            println!("vram      {:.1} / {:.1} GB used{}", v.used as f64 / gb, v.total as f64 / gb,
                     if frac > 0.0 { format!("  ·  keeps {:.1} GB ({:.0}%) free", v.total as f64 * frac / gb, frac * 100.0) } else { String::new() });
        }
        if let Some(avail) = hw::mem_available() {
            let keep = lc.ram_reserve_gb;
            println!("ram       {:.1} GB available{}", avail as f64 / gb, if keep > 0.0 { format!("  ·  keeps {keep:.0} GB free") } else { String::new() });
        }
        println!("runs on   {} (llama.cpp / whisper.cpp)", cfg.engine_label());
        footer(&st);
        if let Some(s) = search::stats(cfg).filter(|s| s.chunks > 0) {
            println!("index     {} files · {} chunks · {} MB", s.files, s.chunks, s.db_mb);
        }
        return Ok(0);
    }
    let npu = st.get("npu").cloned().unwrap_or(json!({}));
    if npu.get("present").and_then(Value::as_bool) == Some(true) {
        let g = |k: &str| npu.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        println!("npu       {}  fw {}  ·  {}  ·  active {:.0}s since boot", g("pci"), g("firmware"), g("runtime"),
                 npu.get("active_ms").and_then(Value::as_f64).unwrap_or(0.0) / 1000.0);
    } else {
        println!("npu       not found (amdxdna driver loaded?)");
    }
    let flm = config::expand_str(&cfg.server.flm);
    match std::process::Command::new(&flm).arg("version").stdin(std::process::Stdio::null()).stderr(std::process::Stdio::null()).output() {
        Ok(o) => {
            let out = String::from_utf8_lossy(&o.stdout);
            let ver: Vec<u32> = regex::Regex::new(r"v?(\d+)\.(\d+)\.(\d+)").unwrap().captures(&out)
                .map(|c| (1..4).map(|i| c[i].parse().unwrap_or(0)).collect()).unwrap_or_default();
            let note = if !ver.is_empty() && ver < vec![1, 0, 12] { "  ⚠ older than 1.0.12: embeddings are wrong, update FastFlowLM" } else { "" };
            let v = if ver.is_empty() { "?".into() } else { ver.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(".") };
            println!("flm       {v}  ({flm}){note}");
        }
        Err(_) => println!("flm       not found: {flm}"),
    }
    footer(&st);
    if let Some(s) = search::stats(cfg).filter(|s| s.chunks > 0) {
        println!("index     {} files · {} chunks · {} MB · updated {}", s.files, s.chunks, s.db_mb, util::strftime("%Y-%m-%d %H:%M", s.updated));
    }
    Ok(0)
}

fn cmd_models() -> Result<i32, ApiError> {
    let out = client::call("/v1/models", None, 30.0)?;
    let cfg = config::load();
    println!("aliases:");
    for a in cfg.models.aliases.keys() {
        println!("  {a:<10} → {}", cfg.resolve_model(Some(a)));
    }
    let data = out.get("data").and_then(Value::as_array).cloned().unwrap_or_default();
    let owned = |m: &Value, by: &str| m.get("owned_by").and_then(Value::as_str) == Some(by);
    if cfg.backend() == Backend::LlamaCpp {
        println!("models (llama.cpp, {}):", cfg.engine_label());
        for m in data.iter().filter(|m| owned(m, "llama.cpp")) {
            let have = match m.get("present").and_then(Value::as_bool) {
                Some(true) => format!("{:>5} GB", m.get("size_gb").and_then(Value::as_f64).map(|x| x.to_string()).unwrap_or("?".into())),
                Some(false) => "missing".into(),
                None => "downloads on first use".into(),
            };
            println!("  {:<20} {have:<22} {}", m["id"].as_str().unwrap_or(""), m.get("source").and_then(Value::as_str).unwrap_or(""));
        }
        println!("add more under [llamacpp.models] in the config");
        return Ok(0);
    }
    println!("installed (FastFlowLM, NPU):");
    for m in data.iter().filter(|m| owned(m, "FastFlowLM")) {
        let labels: Vec<&str> = m.get("labels").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
        let size = match m.get("size_gb") { Some(Value::Null) | None => String::new(), Some(v) => v.to_string() };
        println!("  {:<24} {size:>5} GB  {}", m["id"].as_str().unwrap_or(""), labels.join(","));
    }
    println!("more: flm list   ·   download: flm pull TAG");
    Ok(0)
}

fn cmd_load(model: &str, pmode: Option<String>) -> Result<i32, ApiError> {
    let comp = match model { "stt" | "asr" | "whisper" => Some("asr"), "embed" => Some("embed"), _ => None };
    let mut body = match comp { Some(c) => json!({"component": c}), None => json!({"component": "chat", "model": model}) };
    if let Some(p) = pmode {
        body["pmode"] = json!(p);
    }
    let t0 = Instant::now();
    let st = client::call("/npu/load", Some(&body), 600.0)?;
    let comps: Vec<&str> = st.get("components").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
    let tail = match st.get("pmode").and_then(Value::as_str) {
        Some(p) => format!(" (pmode {p})"),
        None => format!(" on {}", st.get("runs_on").and_then(Value::as_str).unwrap_or("GPU")),
    };
    println!("loaded {comps:?} {} in {:.1}s{tail}", st.get("chat_model").and_then(Value::as_str).unwrap_or(""), t0.elapsed().as_secs_f64());
    Ok(0)
}

fn cmd_stop() -> i32 {
    match client::call("/npu/stop", Some(&json!({})), 30.0) {
        Ok(_) => {
            println!("{} unloaded", config::load().engine_label());
            0
        }
        Err(e) if e.0.contains("not reachable") => 0, // nothing running
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}

fn cmd_ask(o: &AskOpts) -> Result<i32, ApiError> {
    let cfg = config::load();
    if !o.no_files && (!cfg.tools.read.is_empty() || !o.read.is_empty()) {
        return agent::ask_main(&o.args(2048)); // file access: the tool-calling loop
    }
    let q = if o.prompt.is_empty() { agent::read_stdin() } else { o.prompt.join(" ") };
    let extra = if o.think { json!({}) } else { client::no_think() };
    let t0 = Instant::now();
    let out = client::chat(&json!([{"role": "user", "content": q}]), o.model.as_deref(),
                           client::ChatOpts { max_tokens: o.max_tokens.unwrap_or(2048), extra, ..Default::default() })?;
    println!("{}", client::content(&out));
    let n = out.pointer("/usage/completion_tokens").map(|v| v.to_string()).unwrap_or("?".into());
    let tps = client::tps(&out).map(|t| format!(" · {t:.1} tok/s]")).unwrap_or("]".into());
    eprintln!("[{:.1}s · {n} tok{tps}", t0.elapsed().as_secs_f64());
    Ok(0)
}

fn cmd_menu(theme: Option<String>) -> Result<i32, ApiError> {
    let items: [(&str, &[&str]); 10] = [
        ("Dictate (toggle)", &["dictate"]), ("Read text from region (OCR)", &["ocr"]),
        ("Translate region", &["ocr", "--pick"]), ("Translate a message", &["translate"]),
        ("Describe region", &["ocr", "--describe"]), ("Search notes + code", &["search", "--rofi"]),
        ("Load fast model", &["load", "fast"]), ("Load quality model", &["load", "quality"]),
        ("Unload everything", &["stop"]), ("Update search index", &["index"]),
    ];
    let cfg = config::load();
    let theme = theme.or_else(|| (!cfg.ui.menu_theme.is_empty()).then(|| config::expand_str(&cfg.ui.menu_theme)));
    let lines: Vec<String> = items.iter().map(|(l, _)| l.to_string()).collect();
    let (code, out) = ui::rofi(&lines, "npu", theme.as_deref(), None, false);
    let Some(i) = out.parse::<usize>().ok().filter(|i| code == 0 && *i < items.len()) else { return Ok(1) };
    let argv = items[i].1;
    if argv[0] == "ocr" {
        // region pick freezes the screen: let the menu vanish first
        ui::wait_menu_gone(cfg.ui.menu_close_wait, "rofi");
    }
    let cli = Cli::try_parse_from(std::iter::once("npu").chain(argv.iter().copied())).expect("menu entry");
    run(cli)
}

fn run(cli: Cli) -> Result<i32, ApiError> {
    match cli.cmd {
        Cmd::Status { json } => cmd_status(json),
        Cmd::Models => cmd_models(),
        Cmd::Load { model, pmode } => cmd_load(&model, pmode),
        Cmd::Stop => Ok(cmd_stop()),
        Cmd::Ask(o) => cmd_ask(&o),
        Cmd::Chat(o) => agent::chat_main(&o.args(4096)),
        Cmd::Bench { models, other, other_url, pmode, tokens, runs, json } => {
            bench::main(&bench::BenchArgs { models, other, other_url, pmode, tokens, runs, json })
        }
        Cmd::Dictate { action } => Ok(dictate::main(&action)),
        Cmd::Transcribe { file, language } => dictate::file_main(&file, &language),
        Cmd::Ocr { describe, ask, translate, pick, file, copy, model } => {
            Ok(ocr::main(&mut ocr::OcrArgs { describe, ask, translate, pick, file, copy, model }))
        }
        Cmd::Translate { text, to, close, run, draft, lang, chip } => {
            Ok(translate::main(&translate::TranslateArgs { text, to, close, run, draft, lang, chip }))
        }
        Cmd::Index { full, quiet } => search::index(&search::IndexArgs { full, quiet }).map_err(ApiError),
        Cmd::Search { query, n, rofi, theme } => Ok(search::search(&search::SearchArgs { query, n, rofi, theme })),
        Cmd::Menu { theme } => cmd_menu(theme),
        Cmd::Serve => Ok(gateway::main(config::load().clone())),
    }
}

fn main() {
    // a closed pipe (npu search | head) ends quietly instead of panicking on the next print
    unsafe { libc::signal(libc::SIGPIPE, libc::SIG_DFL) };
    let cli = Cli::parse();
    let code = match run(cli) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("npu: {e}");
            1
        }
    };
    std::process::exit(code);
}
