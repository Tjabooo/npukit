//! npud: OpenAI-compatible gateway that owns the model servers (FastFlowLM on the NPU, or
//! llama.cpp + whisper.cpp on a GPU).
//!
//! Why a gateway: an XDNA2 NPU only has room for one chat model plus one helper (Whisper or the
//! embedding model), and a second flm process can't get an NPU context while a chat model is loaded.
//! So exactly one `flm serve` runs, started on demand with the components the last few requests
//! needed, stopped after an idle timeout (shorter on battery), and restarted when a request needs
//! something it doesn't have. Idle cost is zero: with nothing loaded the gateway blocks in epoll,
//! and when started by systemd socket activation it exits after a minute without a model.
//!
//! On machines without an NPU (`[server] backend = "llamacpp"`) the same gateway runs one
//! llama-server per chat/embedding model and a whisper-server for speech, each on its own private
//! port, loaded on demand and unloaded on idle the same way, so the tools work unchanged.

use crate::config::{expand, strs, Backend, Config, ModelSpec};
use crate::hw::{self, Vram};
use crate::util;
use regex::bytes::Regex as BytesRegex;
use regex::Regex;
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{watch, Notify};
use tokio::task::AbortHandle;

static ANSI: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\x1b\[[0-9;]*m").unwrap());
// flm | llama.cpp
static TPS: LazyLock<BytesRegex> =
    LazyLock::new(|| BytesRegex::new(r#""(?:decoding_speed_tps|predicted_per_second)"\s*:\s*([0-9.]+)"#).unwrap());
static LLAMA_ERR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(error|E |ERROR)|\b(error|failed to|unable to)\b.*(load|open|model|alloc)").unwrap()
});

pub fn log(msg: impl AsRef<str>) {
    eprintln!("{}", msg.as_ref());
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Comp {
    Chat,
    Asr,
    Embed,
}

impl Comp {
    pub fn name(self) -> &'static str {
        match self {
            Comp::Chat => "chat",
            Comp::Asr => "asr",
            Comp::Embed => "embed",
        }
    }
    pub fn parse(s: &str) -> Option<Comp> {
        match s {
            "chat" => Some(Comp::Chat),
            "asr" => Some(Comp::Asr),
            "embed" => Some(Comp::Embed),
            _ => None,
        }
    }
    fn route(path: &str) -> Option<Comp> {
        match path {
            "/v1/chat/completions" | "/v1/completions" | "/v1/responses" => Some(Comp::Chat),
            "/v1/embeddings" => Some(Comp::Embed),
            "/v1/audio/transcriptions" | "/v1/audio/translations" => Some(Comp::Asr),
            _ => None,
        }
    }
    fn offset(self) -> u16 {
        match self {
            Comp::Chat => 0,
            Comp::Asr => 1,
            Comp::Embed => 2,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Flm,
    Llama,
    Whisper,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Flm => "flm",
            Kind::Llama => "llama",
            Kind::Whisper => "whisper",
        }
    }
}

/// One model server process: the flm child, or one llama-server / whisper-server.
pub struct Proc {
    pub kind: Kind,
    pub components: Vec<Comp>,
    pub chat_model: Option<String>,
    pub port: u16,
    pub argv: Vec<String>,
    pub probe: &'static str,
    pub timeout: Option<f64>, // load timeout; None = [server] start_timeout
    pub note: String,         // e.g. "CPU" when Whisper had to leave the GPU to the reserve
    pid: Mutex<Option<u32>>,
    exited: watch::Sender<bool>,
    lines: Mutex<VecDeque<String>>,
    failed: Mutex<String>,
}

impl Proc {
    pub fn new(kind: Kind, components: Vec<Comp>, chat_model: Option<String>, port: u16, argv: Vec<String>, probe: &'static str) -> Proc {
        Proc {
            kind, components, chat_model, port, argv, probe, timeout: None, note: String::new(),
            pid: Mutex::new(None), exited: watch::channel(false).0,
            lines: Mutex::new(VecDeque::with_capacity(60)), failed: Mutex::new(String::new()),
        }
    }
    fn same(&self, other: &Proc) -> bool {
        self.argv == other.argv && self.port == other.port
    }
    pub fn pid(&self) -> Option<u32> {
        *self.pid.lock().unwrap()
    }
    pub fn alive(&self) -> bool {
        self.pid().is_some() && !*self.exited.borrow()
    }
    fn failed(&self) -> String {
        self.failed.lock().unwrap().clone()
    }
    fn push_line(&self, line: String) {
        let mut l = self.lines.lock().unwrap();
        if l.len() >= 60 {
            l.pop_front();
        }
        l.push_back(line);
    }
    fn signal(&self, sig: i32) {
        if let Some(pid) = self.pid() {
            if self.alive() {
                unsafe { libc::kill(pid as i32, sig) };
            }
        }
    }

    /// Start the process; its exit is tracked in the background. Err = could not start.
    fn start(self: &Arc<Self>) -> Result<(), String> {
        let mut cmd = tokio::process::Command::new(&self.argv[0]);
        cmd.args(&self.argv[1..]).stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped())
            .process_group(0).kill_on_drop(false);
        // if the gateway itself dies (crash, SIGKILL), its model servers follow instead of holding the NPU/VRAM
        unsafe {
            cmd.pre_exec(|| {
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
                Ok(())
            });
        }
        let mut child = cmd.spawn().map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                let hint = match self.kind {
                    Kind::Flm => "install FastFlowLM",
                    Kind::Llama => "install llama.cpp (Arch: llama-cpp ggml-vulkan)",
                    Kind::Whisper => "install whisper.cpp (Arch: whisper-cpp)",
                };
                let base = std::path::Path::new(&self.argv[0]).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                format!("{base} not found: {hint}")
            } else {
                format!("{}: {e}", self.argv[0])
            }
        })?;
        *self.pid.lock().unwrap() = child.id();
        if let Some(out) = child.stdout.take() {
            tokio::spawn(pump(self.clone(), out));
        }
        if let Some(err) = child.stderr.take() {
            tokio::spawn(pump(self.clone(), err));
        }
        let me = self.clone();
        tokio::spawn(async move {
            let _ = child.wait().await;
            me.exited.send_replace(true);
        });
        Ok(())
    }

    async fn wait_exit(&self, limit: Duration) -> bool {
        let mut rx = self.exited.subscribe();
        let ok = tokio::time::timeout(limit, rx.wait_for(|x| *x)).await.is_ok();
        ok
    }
}

/// Read a server's (often chatty) output: keep a short tail, log errors, never block it.
async fn pump(p: Arc<Proc>, mut r: impl tokio::io::AsyncRead + Unpin) {
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = vec![0u8; 65536];
    loop {
        let n = match r.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        buf.extend_from_slice(&chunk[..n]);
        let mut start = 0;
        while let Some(i) = buf[start..].iter().position(|&b| b == b'\n') {
            handle_line(&p, &buf[start..start + i]);
            start += i + 1;
        }
        buf.drain(..start);
        if buf.len() > 65536 {
            buf.clear(); // request bodies with base64 images come as one huge line
        }
    }
}

fn handle_line(p: &Proc, raw: &[u8]) {
    let raw = &raw[..raw.len().min(400)];
    let text = String::from_utf8_lossy(raw);
    let line = ANSI.replace_all(&text, "");
    let line = line.trim();
    if p.kind == Kind::Flm {
        if !["[FLM]", "[ERROR]", "[Error]", "[Warning]"].iter().any(|t| line.starts_with(t)) {
            return; // flm echoes request content even with --quiet: never parse that
        }
        p.push_line(line.to_string());
        if line.starts_with("[ERROR]") || line.starts_with("[Error]") {
            log(format!("flm: {line}"));
            if line.contains("Failed to load") || line.contains("HWCTX") {
                *p.failed.lock().unwrap() = line.to_string();
            }
        }
        return;
    }
    // llama.cpp / whisper.cpp: transcripts start with "[" (timestamps): keep them out of the tail and
    // the journal; errors look like "error: …", "E …" or "…: failed to …"
    if line.is_empty() || line.starts_with('[') {
        return;
    }
    p.push_line(line.to_string());
    if LLAMA_ERR.is_match(line) {
        log(format!("{}: {line}", p.kind.name()));
    }
}

/// Everything that is loaded: one flm process (all components), or a server per component.
#[derive(Clone)]
pub struct ChildSet {
    pub id: u64,
    pub procs: Vec<Arc<Proc>>,
    pub pmode: Option<String>,
}

static CHILD_IDS: AtomicU64 = AtomicU64::new(1);

impl ChildSet {
    pub fn new(procs: Vec<Arc<Proc>>, pmode: Option<String>) -> ChildSet {
        ChildSet { id: CHILD_IDS.fetch_add(1, Ordering::Relaxed), procs, pmode }
    }
    pub fn components(&self) -> Vec<Comp> {
        self.procs.iter().flat_map(|p| p.components.iter().copied()).collect()
    }
    pub fn proc_for(&self, need: Comp) -> Option<&Arc<Proc>> {
        self.procs.iter().find(|p| p.components.contains(&need))
    }
    pub fn chat_model(&self) -> Option<String> {
        self.proc_for(Comp::Chat).and_then(|p| p.chat_model.clone())
    }
    fn port(&self) -> u16 {
        self.proc_for(Comp::Chat).or(self.procs.first()).map(|p| p.port).unwrap_or(0)
    }
    fn has(&self, need: Comp, model: Option<&str>) -> bool {
        match self.proc_for(need) {
            Some(p) if p.alive() => need != Comp::Chat || p.chat_model.as_deref() == model,
            _ => false,
        }
    }
    fn alive(&self) -> bool {
        !self.procs.is_empty() && self.procs.iter().all(|p| p.alive())
    }
}

/// Hardware readings the engine decides with (replaceable in tests).
#[derive(Clone)]
pub struct HwProbe {
    pub gpu_vram: Arc<dyn Fn() -> Option<Vram> + Send + Sync>,
    pub mem_available: Arc<dyn Fn() -> Option<u64> + Send + Sync>,
    pub proc_vram: Arc<dyn Fn(u32) -> u64 + Send + Sync>,
    pub proc_rss_anon: Arc<dyn Fn(u32) -> u64 + Send + Sync>,
}

impl Default for HwProbe {
    fn default() -> Self {
        HwProbe {
            gpu_vram: Arc::new(hw::gpu_vram),
            mem_available: Arc::new(hw::mem_available),
            proc_vram: Arc::new(hw::proc_vram),
            proc_rss_anon: Arc::new(hw::proc_rss_anon),
        }
    }
}

/// FastFlowLM (one process on the NPU serves the chat model and the helpers together), or
/// llama.cpp + whisper.cpp (a server per component, GPU or CPU, so a GPU can hold the chat model,
/// Whisper and the embedding model at once and swapping one leaves the others loaded).
pub struct Engine {
    pub cfg: Config,
    pub backend: Backend,
    pub hw: HwProbe,
}

impl Engine {
    pub fn new(cfg: Config) -> Engine {
        Engine { backend: cfg.backend(), cfg, hw: HwProbe::default() }
    }
    pub fn name(&self) -> &'static str {
        match self.backend {
            Backend::Flm => "flm",
            Backend::LlamaCpp => "llamacpp",
        }
    }
    fn incremental(&self) -> bool {
        self.backend == Backend::LlamaCpp
    }
    pub fn uses_pmode(&self) -> bool {
        self.backend == Backend::Flm
    }
    fn max_components(&self) -> usize {
        match self.backend {
            Backend::Flm => self.cfg.server.max_components,
            Backend::LlamaCpp => self.cfg.llamacpp.max_components,
        }
    }

    /// [llamacpp.models] entry for a tag; a bare .gguf path or a user/repo[:quant] works too.
    pub fn spec(&self, tag: &str) -> Result<ModelSpec, String> {
        let spec = match self.cfg.model_entry(tag) {
            Some(s) => s,
            None if tag.ends_with(".gguf") => ModelSpec { path: Some(tag.into()), ..Default::default() },
            None if tag.contains('/') => ModelSpec { hf: Some(tag.into()), ..Default::default() },
            None => return Err(format!(
                "unknown model '{tag}': add it under [llamacpp.models] (path = \"….gguf\" or hf = \"user/repo:quant\")")),
        };
        if spec.path.as_deref().unwrap_or("").is_empty() && spec.hf.as_deref().unwrap_or("").is_empty() {
            return Err(format!("[llamacpp.models] '{tag}' needs path or hf"));
        }
        for f in [&spec.path, &spec.mmproj].into_iter().flatten() {
            if !f.is_empty() && !expand(f).exists() {
                return Err(format!("model file missing for {tag}: {f}"));
            }
        }
        Ok(spec)
    }

    /// Fail before anything unloads: unknown model, missing file, no room. (May ask Hugging Face for a size.)
    pub fn check(&self, need: Comp, model: Option<&str>, old_chat: Option<u32>) -> Result<(), String> {
        if self.backend == Backend::Flm {
            return Ok(()); // flm reports unknown tags itself
        }
        let lc = &self.cfg.llamacpp;
        match need {
            Comp::Chat => {
                let tag = model.unwrap_or("");
                let spec = self.spec(tag)?;
                self.fits(tag, &spec, old_chat)
            }
            Comp::Embed => self.spec(&lc.embed_model).map(|_| ()),
            Comp::Asr if !expand(&lc.whisper_model).exists() => Err(format!(
                "Whisper model missing: {} (scripts/get-models.sh downloads it)", lc.whisper_model)),
            Comp::Asr => Ok(()),
        }
    }

    /// Bytes of VRAM to leave free, from [llamacpp] vram_reserve (a fraction of the card).
    pub fn vram_reserve(&self, vram: Option<Vram>) -> u64 {
        let frac = self.cfg.llamacpp.vram_reserve;
        match vram {
            Some(v) if frac > 0.0 => (v.total as f64 * frac) as u64,
            _ => 0,
        }
    }

    /// Bytes a chat server needs (model + projector files + [llamacpp] overhead_gb); None if unknown.
    /// hf = "user/repo[:quant]" models are sized from the Hugging Face file list before they download.
    pub fn footprint(&self, spec: &ModelSpec) -> Option<u64> {
        let extra = (self.cfg.llamacpp.overhead_gb * 1e9) as u64;
        if let Some(gb) = spec.size_gb.filter(|g| *g > 0.0) {
            return Some((gb * 1e9) as u64);
        }
        match spec.path.as_deref().filter(|p| !p.is_empty()) {
            None => hf_size(spec.hf.as_deref().unwrap_or("")).map(|s| s + extra),
            Some(p) => {
                let mut total = extra;
                for f in [Some(p), spec.mmproj.as_deref()].into_iter().flatten().filter(|f| !f.is_empty()) {
                    total += std::fs::metadata(expand(f)).map(|m| m.len()).unwrap_or(0);
                }
                Some(total)
            }
        }
    }

    /// Refuse a model whose share outside the GPU (whatever doesn't fit above the VRAM reserve) would
    /// leave less than ram_reserve_gb of RAM available. The chat server it replaces counts as freed.
    pub fn fits(&self, tag: &str, spec: &ModelSpec, old_chat: Option<u32>) -> Result<(), String> {
        let keep = self.cfg.llamacpp.ram_reserve_gb * 1e9;
        if keep <= 0.0 {
            return Ok(());
        }
        let (Some(need), Some(avail)) = (self.footprint(spec), (self.hw.mem_available)()) else { return Ok(()) };
        let vram = (self.hw.gpu_vram)();
        let (mut avail, mut freed_vram) = (avail as f64, 0.0);
        if let Some(pid) = old_chat {
            freed_vram = (self.hw.proc_vram)(pid) as f64;
            avail += (self.hw.proc_rss_anon)(pid) as f64;
        }
        let reserve = self.vram_reserve(vram) as f64;
        let room = vram.map(|v| (v.total as f64 - v.used as f64 + freed_vram - reserve).max(0.0)).unwrap_or(0.0);
        let in_ram = (need as f64 - room).max(0.0);
        if avail - in_ram < keep {
            let gb = 1e9;
            return Err(format!(
                "not enough memory for {tag} right now: ~{:.1} GB would go to RAM (GPU room {:.1} GB after keeping {:.1} GB free), \
                 {:.1} GB available, keeping {:.0} GB free. Close something or use a smaller model",
                in_ram / gb, room / gb, reserve / gb, avail / gb, keep / gb));
        }
        Ok(())
    }

    /// --fit-target = the VRAM reserve: llama.cpp puts what doesn't fit above it in RAM (MoE experts first).
    fn fit_args(&self) -> Vec<String> {
        let reserve = self.vram_reserve((self.hw.gpu_vram)());
        let mut args = if reserve > 0 {
            vec!["--fit".into(), "on".into(), "--fit-target".into(), (reserve / 1048576).to_string()]
        } else {
            vec![]
        };
        if self.cfg.llamacpp.threads > 0 {
            args.extend(["-t".into(), self.cfg.llamacpp.threads.to_string()]);
        }
        args
    }

    fn whisper_on_gpu(&self) -> bool {
        let Some(v) = (self.hw.gpu_vram)() else { return true };
        let size = std::fs::metadata(expand(&self.cfg.llamacpp.whisper_model)).map(|m| m.len()).unwrap_or(0);
        let need = size + 400 * 1048576; // + work buffers
        v.total as i128 - v.used as i128 - need as i128 >= self.vram_reserve(Some(v)) as i128
    }

    fn model_args(spec: &ModelSpec) -> Vec<String> {
        match spec.path.as_deref().filter(|p| !p.is_empty()) {
            Some(p) => {
                let mut a = vec!["-m".into(), expand(p).to_string_lossy().into_owned()];
                if let Some(mm) = spec.mmproj.as_deref().filter(|m| !m.is_empty()) {
                    a.extend(["--mmproj".into(), expand(mm).to_string_lossy().into_owned()]);
                }
                a
            }
            None => vec!["-hf".into(), spec.hf.clone().unwrap_or_default()],
        }
    }

    pub fn plan(&self, wanted: &[Comp], chat_model: Option<&str>, pmode: Option<&str>) -> Result<Vec<Proc>, String> {
        let s = &self.cfg.server;
        if self.backend == Backend::Flm {
            let mut args = vec![crate::config::expand_str(&s.flm), "serve".into()];
            if wanted.contains(&Comp::Chat) {
                args.push(chat_model.unwrap_or("").into());
            }
            if wanted.contains(&Comp::Asr) {
                args.extend(["--asr".into(), "1".into()]);
            }
            if wanted.contains(&Comp::Embed) {
                args.extend(["--embed".into(), "1".into()]);
            }
            // --quiet: flm otherwise logs every request body; load errors are still printed
            args.extend(["--port".into(), s.flm_port.to_string(), "--host".into(), "127.0.0.1".into(),
                         "--pmode".into(), pmode.unwrap_or("balanced").into(), "--quiet".into()]);
            args.extend(strs(&s.flm_args));
            let chat = wanted.contains(&Comp::Chat).then(|| chat_model.unwrap_or("").to_string());
            return Ok(vec![Proc::new(Kind::Flm, wanted.to_vec(), chat, s.flm_port, args, "/v1/models")]);
        }
        let lc = &self.cfg.llamacpp;
        let mut procs = vec![];
        for &comp in wanted {
            let port = s.flm_port + comp.offset();
            if comp == Comp::Asr {
                let mut argv = vec![crate::config::expand_str(&lc.whisper), "-m".into(),
                                    crate::config::expand_str(&lc.whisper_model), "--host".into(), "127.0.0.1".into(),
                                    "--port".into(), port.to_string(), "--inference-path".into(),
                                    "/v1/audio/transcriptions".into()];
                argv.extend(strs(&lc.whisper_args));
                let gpu = self.whisper_on_gpu();
                if !gpu {
                    // the GPU is at its reserve: transcribe on the CPU instead of eating into it
                    argv.push("-ng".into());
                    if lc.threads > 0 {
                        argv.extend(["-t".into(), lc.threads.to_string()]);
                    }
                }
                let mut p = Proc::new(Kind::Whisper, vec![Comp::Asr], None, port, argv, "/health");
                if !gpu {
                    p.note = "CPU".into();
                }
                procs.push(p);
                continue;
            }
            let tag = if comp == Comp::Chat { chat_model.unwrap_or("").to_string() } else { lc.embed_model.clone() };
            let spec = self.spec(&tag)?;
            let extra = if comp == Comp::Chat { &lc.args } else { &lc.embed_args };
            let mut argv = vec![crate::config::expand_str(&lc.server)];
            argv.extend(Self::model_args(&spec));
            argv.extend(["--host".into(), "127.0.0.1".into(), "--port".into(), port.to_string(), "--alias".into(), tag.clone()]);
            argv.extend(self.fit_args());
            argv.extend(strs(extra));
            argv.extend(strs(&spec.args));
            let mut p = Proc::new(Kind::Llama, vec![comp], (comp == Comp::Chat).then_some(tag), port, argv, "/health");
            if spec.hf.is_some() && spec.path.as_deref().unwrap_or("").is_empty() {
                p.timeout = Some(lc.download_timeout); // the first start downloads the model
            }
            procs.push(p);
        }
        Ok(procs)
    }

    async fn installed(&self) -> Result<Vec<Value>, String> {
        if self.backend == Backend::Flm {
            let flm = crate::config::expand_str(&self.cfg.server.flm);
            let fut = tokio::process::Command::new(flm).args(["list", "--json", "--filter", "installed", "--quiet"])
                .stdin(std::process::Stdio::null()).stderr(std::process::Stdio::null()).output();
            let out = tokio::time::timeout(Duration::from_secs(20), fut).await
                .map_err(|_| "flm list timed out".to_string())?.map_err(|e| e.to_string())?;
            let data: Value = serde_json::from_slice(&out.stdout).map_err(|e| e.to_string())?;
            let list = data.get("models").cloned().unwrap_or(data);
            return Ok(list.as_array().cloned().unwrap_or_default().iter().map(|m| json!({
                "id": m.get("name").cloned().unwrap_or(Value::Null), "object": "model", "owned_by": "FastFlowLM",
                "labels": m.get("label").cloned().unwrap_or(json!([])), "size_gb": m.get("footprint").cloned().unwrap_or(Value::Null),
            })).collect());
        }
        Ok(self.cfg.llamacpp.models.iter().map(|(tag, e)| {
            let spec = e.spec();
            let path = spec.path.as_deref().filter(|p| !p.is_empty()).map(expand);
            let size = path.as_ref().and_then(|p| std::fs::metadata(p).ok()).map(|m| (m.len() as f64 / 1e8).round() / 10.0);
            json!({
                "id": tag, "object": "model", "owned_by": "llama.cpp",
                "source": match &path { Some(p) => p.to_string_lossy().into_owned(), None => format!("hf:{}", spec.hf.unwrap_or_default()) },
                "present": path.as_ref().map(|p| p.exists()),
                "size_gb": size,
            })
        }).collect())
    }
}

static HF_SIZES: LazyLock<Mutex<HashMap<String, Option<u64>>>> = LazyLock::new(Default::default);

/// Download size of what llama-server -hf REF fetches (cached; asks the Hugging Face API via curl).
pub fn hf_size(r: &str) -> Option<u64> {
    if let Some(v) = HF_SIZES.lock().unwrap().get(r) {
        return *v;
    }
    let size = hf_size_with(r, &|repo: &str| {
        let url = format!("https://huggingface.co/api/models/{repo}/tree/main?recursive=true");
        let out = std::process::Command::new("curl").args(["-sfL", "--max-time", "8", &url])
            .stdin(std::process::Stdio::null()).stderr(std::process::Stdio::null()).output().ok()?;
        out.status.success().then(|| serde_json::from_slice(&out.stdout).ok()).flatten()
    });
    HF_SIZES.lock().unwrap().insert(r.to_string(), size);
    size
}

/// The GGUF files matching the quant (all parts of a split model; Q4_K_M when no quant is given, like
/// llama.cpp) plus a vision projector if the repo has one. None = unknown (never 0).
pub fn hf_size_with(r: &str, fetch: &dyn Fn(&str) -> Option<Value>) -> Option<u64> {
    let (repo, quant) = r.split_once(':').unwrap_or((r, ""));
    let quant = if quant.is_empty() { "q4_k_m".to_string() } else { quant.to_lowercase() };
    let tree = fetch(repo)?;
    let ggufs: Vec<(String, u64)> = tree.as_array()?.iter().filter_map(|x| {
        let p = x.get("path")?.as_str()?;
        p.ends_with(".gguf").then(|| (p.rsplit('/').next().unwrap_or(p).to_string(), x.get("size").and_then(Value::as_u64).unwrap_or(0)))
    }).collect();
    let models: Vec<&(String, u64)> = ggufs.iter().filter(|(n, _)| !n.to_lowercase().starts_with("mmproj")).collect();
    let mut picked: Vec<u64> = models.iter().filter(|(n, _)| n.to_lowercase().contains(&quant)).map(|(_, s)| *s).collect();
    if picked.is_empty() && !models.is_empty() && !r.contains(':') {
        picked = vec![models.iter().map(|(_, s)| *s).min().unwrap()]; // no Q4_K_M: llama.cpp takes the first; assume the smallest
    }
    let mut proj: Vec<u64> = ggufs.iter().filter(|(n, _)| n.to_lowercase().starts_with("mmproj")).map(|(_, s)| *s).collect();
    proj.sort();
    (!picked.is_empty()).then(|| picked.iter().sum::<u64>() + proj.first().copied().unwrap_or(0))
}

pub struct State {
    pub child: Option<ChildSet>,
    pub state: &'static str, // off | loading | ready | error
    pub activity: Option<String>,
    pub inflight: u32,
    conns: u32,
    last_used: HashMap<Comp, Instant>,
    unload_task: Option<AbortHandle>,
    exit_task: Option<AbortHandle>,
    pub socket_activated: bool,
    pub last_tps: Option<f64>,
    pub last_error: String,
    loaded_at: f64,
    models_cache: Option<(Instant, Vec<Value>)>,
    bar_last: String,
}

pub struct Gateway {
    pub cfg: Config,
    pub engine: Engine,
    pub st: Mutex<State>,
    load_lock: tokio::sync::Mutex<()>,
    idle: Notify,
    ac_path: String,
}

type Resp = (u16, Value);

impl Gateway {
    pub fn new(cfg: Config) -> Arc<Gateway> {
        Self::with_engine(Engine::new(cfg))
    }

    pub fn with_engine(engine: Engine) -> Arc<Gateway> {
        let cfg = engine.cfg.clone();
        Arc::new(Gateway {
            ac_path: hw::ac_online_path(&cfg.power.ac_online),
            cfg,
            engine,
            st: Mutex::new(State {
                child: None, state: "off", activity: None, inflight: 0, conns: 0, last_used: HashMap::new(),
                unload_task: None, exit_task: None, socket_activated: false, last_tps: None,
                last_error: String::new(), loaded_at: 0.0, models_cache: None, bar_last: String::new(),
            }),
            load_lock: tokio::sync::Mutex::new(()),
            idle: Notify::new(),
        })
    }

    // ------------------------------------------------------------------ power policy
    fn on_ac(&self) -> bool {
        hw::on_ac(&self.ac_path)
    }

    /// Seconds without a request before unloading: the loaded chat model's own idle_unload
    /// ([llamacpp.models] entry, e.g. to keep a translation model warm through a game), else by power source.
    pub fn idle_timeout(&self, child: Option<&ChildSet>) -> f64 {
        if self.engine.backend == Backend::LlamaCpp {
            if let Some(chat) = child.and_then(|c| c.chat_model()) {
                if let Some(t) = self.cfg.model_entry(&chat).and_then(|s| s.idle_unload).filter(|t| *t > 0.0) {
                    return t;
                }
            }
        }
        let p = &self.cfg.power;
        if self.on_ac() { p.ac_idle_unload } else { p.battery_idle_unload }
    }

    fn pmode(&self) -> String {
        let p = &self.cfg.power;
        if self.on_ac() { p.ac_pmode.clone() } else { p.battery_pmode.clone() }
    }

    // ------------------------------------------------------------------ model server processes
    async fn spawn(&self, p: Arc<Proc>) -> bool {
        log(format!("start: {}", p.argv.join(" ")));
        let t0 = Instant::now();
        if let Err(e) = p.start() {
            *p.failed.lock().unwrap() = e;
            return false;
        }
        let deadline = t0 + Duration::from_secs_f64(p.timeout.unwrap_or(self.cfg.server.start_timeout));
        while Instant::now() < deadline {
            // polls only while a model is loading
            if !p.alive() || probe(p.port, p.probe).await {
                break;
            }
            tokio::time::sleep(Duration::from_millis(150)).await;
        }
        tokio::time::sleep(Duration::from_millis(50)).await; // let the pump catch a load error printed just before listen
        if !p.alive() || !p.failed().is_empty() || Instant::now() >= deadline {
            let msg = {
                let f = p.failed();
                if !f.is_empty() { f } else {
                    p.lines.lock().unwrap().back().cloned().unwrap_or_else(|| format!("{} did not start", p.kind.name()))
                }
            };
            log(format!("start failed: {msg}"));
            *p.failed.lock().unwrap() = msg;
            kill(&p).await;
            return false;
        }
        log(format!("ready in {:.1}s: {:?} {}", t0.elapsed().as_secs_f64(),
                    p.components.iter().map(|c| c.name()).collect::<Vec<_>>(), p.chat_model.as_deref().unwrap_or("")));
        true
    }

    /// Make `wanted` loaded: flm restarts its one process; llama.cpp starts the servers that are
    /// missing or must change and leaves the rest running. Returns the components that failed.
    async fn load(self: &Arc<Self>, wanted: &[Comp], chat_model: Option<&str>, pmode: Option<String>) -> Vec<Comp> {
        let plan = match self.engine.plan(wanted, chat_model, pmode.as_deref()) {
            Ok(p) => p,
            Err(e) => {
                let mut st = self.st.lock().unwrap();
                st.last_error = e;
                return wanted.to_vec();
            }
        };
        let old: Vec<Arc<Proc>> = self.st.lock().unwrap().child.as_ref().map(|c| c.procs.clone()).unwrap_or_default();
        let keep: Vec<Arc<Proc>> = if self.engine.incremental() {
            old.iter().filter(|p| p.alive() && plan.iter().any(|q| p.same(q))).cloned().collect()
        } else {
            vec![]
        };
        let drop: Vec<Arc<Proc>> = old.iter().filter(|p| !keep.iter().any(|k| Arc::ptr_eq(k, p))).cloned().collect();
        let new: Vec<Arc<Proc>> = plan.into_iter().filter(|q| !keep.iter().any(|p| q.same(p))).map(Arc::new).collect();
        if !drop.is_empty() {
            loop {
                // never pull a model out from under a running request
                let n = self.idle.notified();
                if self.st.lock().unwrap().inflight == 0 {
                    break;
                }
                n.await;
            }
        }
        for p in &drop {
            kill(p).await;
        }
        if !drop.is_empty() && self.engine.backend == Backend::Flm {
            tokio::time::sleep(Duration::from_millis(300)).await; // the driver frees the NPU context right after exit
        }
        {
            let mut st = self.st.lock().unwrap();
            let pm = if self.engine.uses_pmode() { pmode.clone() } else { None };
            st.child = Some(ChildSet::new(keep.iter().chain(new.iter()).cloned().collect(), pm));
            if new.is_empty() {
                return vec![];
            }
            st.state = "loading";
            self.publish(&mut st, false);
        }
        let mut handles = vec![];
        for q in &new {
            let me = self.clone();
            let q = q.clone();
            handles.push(tokio::spawn(async move { me.spawn(q).await }));
        }
        let mut failed: Vec<Arc<Proc>> = vec![];
        for (q, h) in new.iter().zip(handles) {
            if !h.await.unwrap_or(false) {
                failed.push(q.clone());
            }
        }
        let mut st = self.st.lock().unwrap();
        if let Some(c) = st.child.as_mut() {
            c.procs.retain(|p| !failed.iter().any(|f| Arc::ptr_eq(f, p)));
        }
        if let Some(f) = failed.first() {
            st.last_error = f.failed();
        }
        if st.child.as_ref().is_some_and(|c| !c.procs.is_empty()) {
            st.loaded_at = util::now();
            st.state = "ready";
        } else {
            st.child = None;
            st.state = "error";
        }
        self.publish(&mut st, false);
        if !failed.is_empty() && st.child.is_none() {
            self.arm_exit(&mut st);
        }
        failed.iter().flat_map(|q| q.components.iter().copied()).collect()
    }

    async fn stop(self: &Arc<Self>) {
        let child = {
            let mut st = self.st.lock().unwrap();
            if let Some(h) = st.unload_task.take() {
                h.abort();
            }
            st.child.take()
        };
        let Some(child) = child else { return };
        let procs: Vec<_> = child.procs.iter().filter(|p| p.alive()).cloned().collect();
        for p in &procs {
            kill(p).await;
        }
        if !procs.is_empty() && self.engine.backend == Backend::Flm {
            tokio::time::sleep(Duration::from_millis(300)).await; // the driver frees the hardware context right after exit
        }
        let mut st = self.st.lock().unwrap();
        st.state = "off";
        self.publish(&mut st, false);
        self.arm_exit(&mut st);
    }

    /// Make sure something serves `need`, count the caller as in flight, return its port.
    pub async fn acquire(self: &Arc<Self>, need: Comp, model: Option<&str>, pmode: Option<String>) -> Result<u16, String> {
        let pmode = if self.engine.uses_pmode() { pmode } else { None };
        let _guard = self.load_lock.lock().await;
        let (ok, old_chat, keep, cur_chat) = {
            let st = self.st.lock().unwrap();
            let c = st.child.as_ref();
            let ok = c.is_some_and(|c| c.has(need, model) && (pmode.is_none() || c.pmode == pmode));
            let old_chat = c.and_then(|c| c.proc_for(Comp::Chat)).filter(|p| p.alive()).and_then(|p| p.pid());
            let timeout = self.idle_timeout(c);
            let now = Instant::now();
            let mut keep: Vec<Comp> = c.map(|c| c.components()).unwrap_or_default().into_iter().filter(|&x| {
                x != need && st.last_used.get(&x).is_some_and(|t| now.duration_since(*t).as_secs_f64() < timeout)
                    && c.and_then(|c| c.proc_for(x)).is_some_and(|p| p.alive())
            }).collect();
            keep.sort_by_key(|x| std::cmp::Reverse(st.last_used.get(x).copied()));
            (ok, old_chat, keep, c.and_then(|c| c.chat_model()))
        };
        if !ok {
            // unknown model, missing file, no room: fail before unloading (may ask Hugging Face for a size)
            let me = self.clone();
            let m = model.map(str::to_string);
            tokio::task::spawn_blocking(move || me.engine.check(need, m.as_deref(), old_chat)).await
                .map_err(|e| e.to_string())??;
            let mut wanted = vec![need];
            wanted.extend(keep);
            wanted.truncate(self.engine.max_components());
            let chat_model = if need == Comp::Chat { model.map(str::to_string) }
                else if wanted.contains(&Comp::Chat) { cur_chat } else { None };
            let pm = if self.engine.uses_pmode() { Some(pmode.clone().unwrap_or_else(|| self.pmode())) } else { None };
            let mut failed = self.load(&wanted, chat_model.as_deref(), pm.clone()).await;
            if failed.contains(&need) && wanted.len() > 1 && !self.engine.incremental() {
                // flm: the NPU may not fit the combination; try the one component that is needed
                let m = if need == Comp::Chat { model } else { None };
                failed = self.load(&[need], m, pm).await;
            }
            let st = self.st.lock().unwrap();
            if failed.contains(&need) || st.child.as_ref().and_then(|c| c.proc_for(need)).is_none() {
                return Err(if st.last_error.is_empty() { format!("{} failed to start", self.engine.name()) } else { st.last_error.clone() });
            }
        }
        let mut st = self.st.lock().unwrap();
        st.last_used.insert(need, Instant::now());
        st.inflight += 1;
        if let Some(h) = st.unload_task.take() {
            h.abort();
        }
        self.publish(&mut st, false);
        let port = st.child.as_ref().and_then(|c| c.proc_for(need)).map(|p| p.port).unwrap_or(0);
        Ok(port)
    }

    pub fn release(self: &Arc<Self>, need: Option<Comp>) {
        let mut st = self.st.lock().unwrap();
        st.inflight = st.inflight.saturating_sub(1);
        if let Some(n) = need {
            st.last_used.insert(n, Instant::now());
        }
        if st.inflight == 0 {
            self.idle.notify_waiters();
            self.arm_unload(&mut st);
        }
        self.publish(&mut st, false);
    }

    fn arm_unload(self: &Arc<Self>, st: &mut State) {
        if let Some(h) = st.unload_task.take() {
            h.abort();
        }
        let Some(c) = st.child.as_ref() else { return };
        if st.inflight > 0 {
            return;
        }
        let timeout = self.idle_timeout(Some(c));
        let now = Instant::now();
        let delay = match c.components().iter().filter_map(|x| st.last_used.get(x)).max() {
            Some(last) => (timeout - now.duration_since(*last).as_secs_f64()).max(1.0),
            None => 1.0,
        };
        let me = self.clone();
        st.unload_task = Some(tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs_f64(delay)).await;
            let _g = me.load_lock.lock().await;
            let go = {
                let mut st = me.st.lock().unwrap();
                // the stored handle is this task (a newer arm_unload would have aborted it): drop it, or
                // stop() would abort the task it runs in halfway through
                st.unload_task = None;
                st.child.is_some() && st.inflight == 0
            };
            if go {
                log("idle: unloading");
                me.stop().await;
            }
        }).abort_handle());
    }

    fn arm_exit(self: &Arc<Self>, st: &mut State) {
        if let Some(h) = st.exit_task.take() {
            h.abort();
        }
        if !st.socket_activated || st.child.is_some() || st.conns > 0 || st.activity.is_some() {
            return;
        }
        let me = self.clone();
        let after = self.cfg.server.exit_after;
        st.exit_task = Some(tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs_f64(after)).await;
            let mut st = me.st.lock().unwrap();
            if st.child.is_some() || st.conns > 0 || st.activity.is_some() {
                return;
            }
            log("nothing loaded, exiting (systemd starts me again on the next request)");
            me.publish(&mut st, true);
            std::process::exit(0);
        }).abort_handle());
    }

    // ------------------------------------------------------------------ memory guard
    /// Kernel memory-pressure trigger (PSI): costs nothing until memory gets tight. When it fires while
    /// models are loaded and RAM is nearly gone, they are killed at once so the desktop and games live.
    fn start_memory_guard(self: &Arc<Self>) {
        if self.cfg.server.memory_guard_gb <= 0.0 {
            return;
        }
        use std::os::fd::FromRawFd;
        let path = c"/proc/pressure/memory";
        let fd = unsafe { libc::open(path.as_ptr(), libc::O_RDWR | libc::O_NONBLOCK | libc::O_CLOEXEC) };
        // 300 ms of stalls in a 2 s window (2 s steps are what unprivileged triggers may use); the kernel
        // overwrites the last byte with NUL, so send one
        let trig = b"some 300000 2000000\0";
        if fd < 0 || unsafe { libc::write(fd, trig.as_ptr() as *const libc::c_void, trig.len()) } < 0 {
            log(format!("memory guard off: {}", std::io::Error::last_os_error()));
            if fd >= 0 {
                unsafe { libc::close(fd) };
            }
            return;
        }
        let file = unsafe { std::fs::File::from_raw_fd(fd) }; // closed if the thread ends
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<()>();
        std::thread::Builder::new().name("memory-guard".into()).spawn(move || {
            use std::os::fd::AsRawFd;
            loop {
                let mut pfd = libc::pollfd { fd: file.as_raw_fd(), events: libc::POLLPRI, revents: 0 };
                let r = unsafe { libc::poll(&mut pfd, 1, -1) };
                if r < 0 {
                    if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                        continue;
                    }
                    return;
                }
                if pfd.revents & libc::POLLERR != 0 || tx.send(()).is_err() {
                    return;
                }
            }
        }).ok();
        let me = self.clone();
        tokio::spawn(async move {
            while rx.recv().await.is_some() {
                me.on_memory_pressure(None);
            }
        });
    }

    /// Kill the model servers when RAM available drops under [server] memory_guard_gb. True if it did.
    pub fn on_memory_pressure(self: &Arc<Self>, avail: Option<u64>) -> bool {
        let mut st = self.st.lock().unwrap();
        let Some(c) = st.child.clone().filter(|c| !c.procs.is_empty()) else { return false };
        let floor = self.cfg.server.memory_guard_gb * 1e9;
        let Some(avail) = avail.or_else(|| (self.engine.hw.mem_available)()) else { return false };
        if avail as f64 >= floor {
            return false;
        }
        let names: Vec<String> = c.procs.iter().map(|p| p.chat_model.clone().unwrap_or_else(|| p.kind.name().into())).collect();
        st.last_error = format!("unloaded {}: RAM was running out ({:.1} GB left)", names.join(", "), avail as f64 / 1e9);
        log(format!("memory guard: {}", st.last_error));
        for p in &c.procs {
            p.signal(libc::SIGKILL); // frees the memory now; in-flight requests get an error
        }
        st.state = "error";
        self.publish(&mut st, false);
        let me = self.clone();
        tokio::spawn(async move {
            let _g = me.load_lock.lock().await;
            let same = me.st.lock().unwrap().child.as_ref().is_some_and(|x| x.id == c.id);
            if same {
                me.stop().await;
                let mut st = me.st.lock().unwrap();
                st.state = "error";
                me.publish(&mut st, false);
            }
        });
        true
    }

    // ------------------------------------------------------------------ status / waybar
    pub fn status(&self) -> Value {
        let st = self.st.lock().unwrap();
        let c = st.child.as_ref();
        let pm = self.engine.uses_pmode();
        json!({
            "state": if st.inflight > 0 && st.state == "ready" { "busy" } else { st.state },
            "activity": st.activity,
            "components": c.map(|c| c.components().iter().map(|x| x.name()).collect::<Vec<_>>()).unwrap_or_default(),
            "chat_model": c.and_then(|c| c.chat_model()),
            "pmode": if pm { Some(c.and_then(|c| c.pmode.clone()).unwrap_or_else(|| self.pmode())) } else { None },
            "backend": self.engine.name(),
            "runs_on": self.cfg.engine_label(),
            "power": if self.on_ac() { "ac" } else { "battery" },
            "idle_unload": self.idle_timeout(c),
            "inflight": st.inflight,
            "loaded_since": c.map(|_| st.loaded_at),
            "flm_pid": c.and_then(|c| c.procs.first()).and_then(|p| p.pid()),
            "procs": c.map(|c| c.procs.iter().map(|p| json!({
                "kind": p.kind.name(), "components": p.components.iter().map(|x| x.name()).collect::<Vec<_>>(),
                "model": p.chat_model, "port": p.port, "pid": p.pid(), "note": p.note,
                "vram_mb": p.pid().map(|pid| (self.engine.hw.proc_vram)(pid) / 1048576).unwrap_or(0),
            })).collect::<Vec<_>>()).unwrap_or_default(),
            "last_tps": st.last_tps,
            "last_error": st.last_error,
            "npu": hw::npu_info(),
        })
    }

    pub fn publish(&self, st: &mut State, force_off: bool) {
        let b = &self.cfg.bar;
        let col = |k: &str| -> String {
            b.colors.get(k).cloned().unwrap_or_else(|| match k {
                "tag" => "#9b5de5", "dim" => "#6c6c8a", "text" => "#babbf1", "hot" => "#f15bb5",
                "load" => "#00c6ff", _ => "#ff5252",
            }.to_string())
        };
        let s = if force_off { "off" } else if st.inflight > 0 && st.state == "ready" { "busy" } else { st.state };
        let c = if force_off { None } else { st.child.as_ref() };
        let act = if force_off { None } else { st.activity.as_deref() };
        let tagspan = |color: &str| format!("<span size='8pt' weight='800' color='{color}'>{}</span>  ", b.tag);
        let mut parts: Vec<String> = vec![];
        if let Some(m) = c.and_then(|c| c.chat_model()) {
            parts.push(m.replace(':', " "));
        }
        if let Some(c) = c {
            for x in c.components() {
                match x {
                    Comp::Asr => parts.push("STT".into()),
                    Comp::Embed => parts.push("EMB".into()),
                    Comp::Chat => {}
                }
            }
        }
        let what = parts.join(" + ");
        let (cls, text): (String, String) = if act == Some("rec") {
            // solid chip, colored by the bar's CSS (#custom-npu.rec)
            ("rec".into(), format!("<span size='8pt' weight='800'>{}</span>  <span weight='800'>● REC</span>", b.tag))
        } else if let Some(a) = act {
            let label = match a { "stt" => "STT…".into(), "ocr" => "OCR…".into(), "index" => "INDEX…".into(), a => a.to_uppercase() };
            (format!("busy {a}"), format!("{}<span weight='800' color='{}'>{label}</span>", tagspan(&col("hot")), col("hot")))
        } else {
            match s {
                "off" => ("off".into(), format!("<span size='8pt' weight='800' color='{}'>{}</span>", col("dim"), b.tag)),
                "loading" => ("loading".into(), format!("{}<span color='{}'>{}</span>", tagspan(&col("tag")), col("load"),
                                                        if what.is_empty() { "loading" } else { &what })),
                "busy" => ("busy".into(), format!("{}<span weight='800' color='{}'>{what}</span>", tagspan(&col("hot")), col("hot"))),
                "error" => ("error".into(), format!("{}<span color='{}'>error</span>", tagspan(&col("rec")), col("rec"))),
                _ => ("ready".into(), format!("{}<span color='{}'>{what}</span>", tagspan(&col("tag")), col("text"))),
            }
        };
        let ac = self.on_ac();
        let power = if ac { "AC" } else { "battery" };
        let wher = self.cfg.engine_label();
        let pm = self.engine.uses_pmode();
        let mut tip = vec![format!("{wher} · {s}") + &act.map(|a| format!(" · {a}")).unwrap_or_default()];
        if let Some(c) = c {
            let loaded: Vec<String> = c.procs.iter().flat_map(|p| p.components.iter().map(move |x| {
                let n = match x { Comp::Chat => p.chat_model.clone().unwrap_or_default(), Comp::Asr => "whisper (speech to text)".into(),
                                  Comp::Embed => "embeddings".into() };
                if p.note.is_empty() { n } else { format!("{n} on the {}", p.note) }
            })).collect();
            tip.push(format!("loaded: {}", loaded.join(", ")));
            tip.push(if pm { format!("pmode {} · ", c.pmode.clone().unwrap_or_default()) } else { String::new() }
                     + &format!("unloads after {} s idle ({power})", self.idle_timeout(Some(c)) as i64));
        } else {
            tip.push(format!("nothing loaded · {power}") + &if pm { format!(" → pmode {}", self.pmode()) } else { String::new() });
        }
        if let (Some(t), Some(_)) = (st.last_tps, c) {
            tip.push(format!("last reply {t:.1} tok/s"));
        }
        if s == "error" && !st.last_error.is_empty() {
            tip.push(util::take_chars(&st.last_error, 120).to_string());
        }
        tip.push("click: menu · right-click: unload".into());
        let classes: Vec<&str> = cls.split_whitespace().collect();
        let out = json!({"text": text, "class": classes, "tooltip": tip.join("\n"), "alt": classes[0]}).to_string();
        if out == st.bar_last {
            return;
        }
        st.bar_last = out.clone();
        let path = expand(&b.file);
        if let Err(e) = util::write_atomic(&path, (out + "\n").as_bytes()) {
            log(format!("bar write failed: {e}"));
        }
        // machine-readable twin for other tools (e.g. a fleet dashboard reading it over SSH)
        let status = json!({
            "state": s, "activity": act, "components": c.map(|c| c.components().iter().map(|x| x.name()).collect::<Vec<_>>()).unwrap_or_default(),
            "chat_model": c.and_then(|c| c.chat_model()),
            "pmode": if pm { Some(c.and_then(|c| c.pmode.clone()).unwrap_or_else(|| self.pmode())) } else { None },
            "runs_on": wher, "power": if ac { "ac" } else { "battery" },
            "last_tps": if c.is_some() { st.last_tps } else { None }, "updated": util::now(),
        });
        if let Err(e) = util::write_atomic(&path.with_file_name("status.json"), (status.to_string() + "\n").as_bytes()) {
            log(format!("status write failed: {e}"));
        }
        if b.signal > 0 {
            signal_waybar(libc::SIGRTMIN() + b.signal);
        }
    }

    // ------------------------------------------------------------------ HTTP
    async fn handle(self: Arc<Self>, sock: TcpStream) {
        {
            let mut st = self.st.lock().unwrap();
            st.conns += 1;
            if let Some(h) = st.exit_task.take() {
                h.abort();
            }
        }
        let _ = sock.set_nodelay(true);
        let (r, mut w) = sock.into_split();
        let mut r = BufReader::with_capacity(65536, r);
        if let Err(e) = self.handle_inner(&mut r, &mut w).await {
            if e.kind() != std::io::ErrorKind::UnexpectedEof && e.kind() != std::io::ErrorKind::ConnectionReset
                && e.kind() != std::io::ErrorKind::BrokenPipe {
                log(format!("request failed: {e}"));
                let _ = respond(&mut w, 502, &json!({"error": {"message": e.to_string(), "type": "npukit_error"}})).await;
            }
        }
        let _ = w.shutdown().await;
        let mut st = self.st.lock().unwrap();
        st.conns -= 1;
        self.arm_exit(&mut st);
    }

    async fn handle_inner(self: &Arc<Self>, r: &mut BufReader<tokio::net::tcp::OwnedReadHalf>,
                          w: &mut tokio::net::tcp::OwnedWriteHalf) -> std::io::Result<()> {
        let mut first = String::new();
        if r.read_line(&mut first).await? == 0 {
            return Ok(());
        }
        let mut parts = first.trim_end().splitn(3, ' ');
        let (method, target) = (parts.next().unwrap_or("").to_string(), parts.next().unwrap_or("/").to_string());
        let mut headers: Vec<(String, String)> = vec![];
        let mut head_len = first.len();
        loop {
            let mut ln = String::new();
            let n = r.read_line(&mut ln).await?;
            head_len += n;
            if head_len > 1 << 20 {
                return Err(std::io::Error::other("request head too large"));
            }
            if n == 0 || ln == "\r\n" || ln == "\n" {
                break;
            }
            if let Some((k, v)) = ln.split_once(':') {
                headers.push((k.trim().to_string(), v.trim().to_string()));
            }
        }
        let h = |name: &str| headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.clone()).unwrap_or_default();
        if h("expect").eq_ignore_ascii_case("100-continue") {
            w.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").await?;
        }
        let mut body = Vec::new();
        if h("transfer-encoding").to_ascii_lowercase().contains("chunked") {
            loop {
                let mut ln = String::new();
                r.read_line(&mut ln).await?;
                let size = usize::from_str_radix(ln.split(';').next().unwrap_or("").trim(), 16).unwrap_or(0);
                if size == 0 {
                    let mut t = String::new();
                    r.read_line(&mut t).await?;
                    break;
                }
                let s = body.len();
                body.resize(s + size, 0);
                r.read_exact(&mut body[s..]).await?;
                let mut crlf = String::new();
                r.read_line(&mut crlf).await?;
            }
        } else if let Ok(n) = h("content-length").parse::<usize>() {
            body.resize(n, 0);
            r.read_exact(&mut body).await?;
        }
        let path = target.split('?').next().unwrap_or("/").trim_end_matches('/');
        let path = if path.is_empty() { "/" } else { path };

        // -- gateway's own endpoints
        if method == "GET" && matches!(path, "/health" | "/npu/status" | "/api/v1/health") {
            return respond(w, 200, &self.status()).await;
        }
        if method == "GET" && path == "/v1/models" {
            return respond(w, 200, &self.models().await).await;
        }
        if method == "POST" && path.starts_with("/npu/") {
            let req: Value = if body.is_empty() { json!({}) } else {
                match serde_json::from_slice(&body) {
                    Ok(v) => v,
                    Err(_) => return respond(w, 400, &json!({"error": {"message": "invalid JSON"}})).await,
                }
            };
            let (code, v) = self.control(&path[5..], &req).await;
            return respond(w, code, &v).await;
        }

        let need = Comp::route(path);
        let mut model = None;
        if need == Some(Comp::Chat) && !body.is_empty() {
            match serde_json::from_slice::<Value>(&body) {
                Ok(mut req) => {
                    let m = self.cfg.resolve_model(req.get("model").and_then(Value::as_str));
                    req["model"] = json!(m);
                    model = Some(m);
                    body = serde_json::to_vec(&req).unwrap();
                }
                Err(_) => return respond(w, 400, &json!({"error": {"message": "invalid JSON"}})).await,
            }
        }
        let port = match need {
            Some(n) => match self.acquire(n, model.as_deref(), None).await {
                Ok(p) => p,
                Err(e) => return respond(w, 503, &json!({"error": {"message": e, "type": "npu_unavailable"}})).await,
            },
            None => {
                let port = {
                    let mut st = self.st.lock().unwrap();
                    let p = st.child.as_ref().filter(|c| c.alive()).map(|c| c.port());
                    if p.is_some() {
                        st.inflight += 1;
                    }
                    p
                };
                match port {
                    Some(p) => p,
                    None => return respond(w, 404, &json!({"error": {"message": format!("unknown endpoint {path}")}})).await,
                }
            }
        };
        let res = self.forward(port, &method, &target, &headers, &body, w).await;
        self.release(need);
        res
    }

    async fn forward(&self, port: u16, method: &str, target: &str, headers: &[(String, String)], body: &[u8],
                     w: &mut tokio::net::tcp::OwnedWriteHalf) -> std::io::Result<()> {
        let up = TcpStream::connect(("127.0.0.1", port)).await?;
        let _ = up.set_nodelay(true);
        let (ur, mut uw) = up.into_split();
        let skip = ["host", "connection", "content-length", "expect", "transfer-encoding", "keep-alive"];
        let mut out = format!("{method} {target} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n");
        for (k, v) in headers {
            if !skip.contains(&k.to_ascii_lowercase().as_str()) {
                out += &format!("{k}: {v}\r\n");
            }
        }
        if !body.is_empty() || matches!(method, "POST" | "PUT" | "PATCH") {
            out += &format!("Content-Length: {}\r\n", body.len());
        }
        out += "\r\n";
        uw.write_all(out.as_bytes()).await?;
        uw.write_all(body).await?;
        let mut ur = BufReader::with_capacity(65536, ur);
        let mut tail = Tail::default();
        // response head: pass through, minus connection headers
        let mut head = String::new();
        let mut status = String::new();
        ur.read_line(&mut status).await?;
        head += status.trim_end();
        head += "\r\n";
        let (mut len, mut chunked) = (None::<u64>, false);
        loop {
            let mut ln = String::new();
            if ur.read_line(&mut ln).await? == 0 || ln == "\r\n" || ln == "\n" {
                break;
            }
            if let Some((k, v)) = ln.split_once(':') {
                let k = k.trim().to_ascii_lowercase();
                if k == "content-length" {
                    len = v.trim().parse().ok();
                } else if k == "transfer-encoding" && v.to_ascii_lowercase().contains("chunked") {
                    chunked = true;
                }
                if k == "connection" || k == "keep-alive" {
                    continue;
                }
            }
            head += ln.trim_end();
            head += "\r\n";
        }
        head += "Connection: close\r\n\r\n";
        w.write_all(head.as_bytes()).await?;
        let mut buf = vec![0u8; 65536];
        if chunked {
            loop {
                let mut size_line = String::new();
                ur.read_line(&mut size_line).await?;
                w.write_all(size_line.as_bytes()).await?;
                let size = usize::from_str_radix(size_line.split(';').next().unwrap_or("").trim(), 16).unwrap_or(0);
                if size == 0 {
                    loop {
                        let mut t = String::new();
                        let n = ur.read_line(&mut t).await?;
                        w.write_all(t.as_bytes()).await?;
                        if n == 0 || t == "\r\n" || t == "\n" {
                            break;
                        }
                    }
                    break;
                }
                let mut left = size + 2;
                while left > 0 {
                    let n = ur.read(&mut buf[..left.min(65536)]).await?;
                    if n == 0 {
                        return Err(std::io::ErrorKind::UnexpectedEof.into());
                    }
                    w.write_all(&buf[..n]).await?;
                    tail.push(&buf[..n]);
                    left -= n;
                }
            }
        } else if let Some(mut left) = len {
            while left > 0 {
                let n = ur.read(&mut buf[..(left.min(65536) as usize)]).await?;
                if n == 0 {
                    break;
                }
                w.write_all(&buf[..n]).await?;
                tail.push(&buf[..n]);
                left -= n as u64;
            }
        } else {
            loop {
                let n = ur.read(&mut buf).await?;
                if n == 0 {
                    break;
                }
                w.write_all(&buf[..n]).await?;
                tail.push(&buf[..n]);
            }
        }
        w.flush().await?;
        if let Some(t) = TPS.captures(&tail.buf).and_then(|m| std::str::from_utf8(&m[1]).ok()?.parse::<f64>().ok()) {
            self.st.lock().unwrap().last_tps = Some(t);
        }
        Ok(())
    }

    async fn models(&self) -> Value {
        let fresh = self.st.lock().unwrap().models_cache.as_ref().is_some_and(|(t, _)| t.elapsed().as_secs() <= 60);
        if !fresh {
            if let Ok(list) = self.engine.installed().await {
                self.st.lock().unwrap().models_cache = Some((Instant::now(), list));
            }
        }
        let mut data: Vec<Value> = self.cfg.models.aliases.keys().map(|a| json!({
            "id": a, "object": "model", "owned_by": "npukit", "alias_of": self.cfg.resolve_model(Some(a)),
        })).collect();
        if let Some((_, list)) = &self.st.lock().unwrap().models_cache {
            data.extend(list.iter().cloned());
        }
        json!({"object": "list", "data": data})
    }

    async fn control(self: &Arc<Self>, action: &str, req: &Value) -> Resp {
        match action {
            "stop" => {
                let _g = self.load_lock.lock().await;
                if self.st.lock().unwrap().inflight > 0 {
                    return (409, json!({"error": "busy"}));
                }
                self.stop().await;
                (200, self.status())
            }
            "load" | "warm" => {
                let comp = req.get("component").and_then(Value::as_str).and_then(Comp::parse).unwrap_or(Comp::Chat);
                let model = (comp == Comp::Chat).then(|| self.cfg.resolve_model(req.get("model").and_then(Value::as_str)));
                if let Some(a) = req.get("activity") {
                    let mut st = self.st.lock().unwrap();
                    st.activity = a.as_str().filter(|s| !s.is_empty()).map(str::to_string);
                    self.publish(&mut st, false);
                }
                let pmode = req.get("pmode").and_then(Value::as_str).map(str::to_string);
                if let Err(e) = self.acquire(comp, model.as_deref(), pmode).await {
                    return (503, json!({"error": e}));
                }
                self.release(Some(comp));
                (200, self.status())
            }
            "activity" => {
                let mut st = self.st.lock().unwrap();
                st.activity = req.get("activity").and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string);
                self.publish(&mut st, false);
                self.arm_exit(&mut st);
                drop(st);
                (200, self.status())
            }
            "release" => {
                // caller is done with a component: let it age out now
                let mut st = self.st.lock().unwrap();
                if let Some(c) = req.get("component").and_then(Value::as_str).and_then(Comp::parse) {
                    st.last_used.remove(&c); // as if never used: ages out at once
                }
                self.arm_unload(&mut st);
                drop(st);
                (200, self.status())
            }
            _ => (404, json!({"error": format!("unknown action {action}")})),
        }
    }
}

/// The last 32 KiB of a forwarded body (where the tok/s figure is).
#[derive(Default)]
struct Tail {
    buf: Vec<u8>,
}

impl Tail {
    fn push(&mut self, data: &[u8]) {
        const MAX: usize = 32768;
        let data = &data[data.len().saturating_sub(MAX)..];
        if self.buf.len() + data.len() > MAX {
            let cut = self.buf.len() + data.len() - MAX;
            self.buf.drain(..cut.min(self.buf.len()));
        }
        self.buf.extend_from_slice(data);
    }
}

async fn kill(p: &Proc) {
    if !p.alive() {
        return;
    }
    log(format!("stop: {:?} {}", p.components.iter().map(|c| c.name()).collect::<Vec<_>>(), p.chat_model.as_deref().unwrap_or("")));
    p.signal(libc::SIGTERM);
    if !p.wait_exit(Duration::from_secs(8)).await {
        log(format!("{} ignored SIGTERM, killing", p.kind.name()));
        p.signal(libc::SIGKILL);
        p.wait_exit(Duration::from_secs(30)).await;
    }
}

async fn probe(port: u16, path: &str) -> bool {
    let Ok(Ok(mut s)) = tokio::time::timeout(Duration::from_secs(1), TcpStream::connect(("127.0.0.1", port))).await else {
        return false;
    };
    let req = format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
    if s.write_all(req.as_bytes()).await.is_err() {
        return false;
    }
    let mut r = BufReader::new(s);
    let mut line = String::new();
    matches!(tokio::time::timeout(Duration::from_secs(5), r.read_line(&mut line)).await, Ok(Ok(_)))
        && line.contains(" 200")
}

async fn respond(w: &mut tokio::net::tcp::OwnedWriteHalf, code: u16, obj: &Value) -> std::io::Result<()> {
    let body = obj.to_string();
    let reason = match code {
        200 => "OK", 400 => "Bad Request", 404 => "Not Found", 409 => "Conflict",
        502 => "Bad Gateway", 503 => "Service Unavailable", _ => "OK",
    };
    let msg = format!("HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
    w.write_all(msg.as_bytes()).await
}

fn signal_waybar(sig: i32) {
    let Ok(dir) = std::fs::read_dir("/proc") else { return };
    for e in dir.flatten() {
        let name = e.file_name();
        let Some(pid) = name.to_str().and_then(|s| s.parse::<i32>().ok()) else { continue };
        if let Ok(comm) = std::fs::read_to_string(format!("/proc/{pid}/comm")) {
            if matches!(comm.trim(), "waybar" | ".waybar-wrapped") {
                unsafe { libc::kill(pid, sig) };
            }
        }
    }
}

pub fn main(cfg: Config) -> i32 {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("tokio runtime");
    rt.block_on(serve(cfg))
}

async fn serve(cfg: Config) -> i32 {
    use tokio::signal::unix::{signal, SignalKind};
    let gw = Gateway::new(cfg.clone());
    let listen_pid = std::env::var("LISTEN_PID").ok().and_then(|s| s.parse::<u32>().ok());
    let listen_fds = std::env::var("LISTEN_FDS").ok().and_then(|s| s.parse::<i32>().ok()).unwrap_or(0);
    let listener = if listen_pid == Some(std::process::id()) && listen_fds >= 1 {
        use std::os::fd::FromRawFd;
        let std_l = unsafe { std::net::TcpListener::from_raw_fd(3) };
        std_l.set_nonblocking(true).ok();
        unsafe { libc::fcntl(3, libc::F_SETFD, libc::FD_CLOEXEC) }; // model servers must not inherit it
        gw.st.lock().unwrap().socket_activated = true;
        let l = TcpListener::from_std(std_l).expect("socket from systemd");
        log(format!("npud: socket-activated on {}", l.local_addr().map(|a| a.to_string()).unwrap_or_default()));
        l
    } else {
        match TcpListener::bind(&cfg.server.listen).await {
            Ok(l) => {
                log(format!("npud: listening on {}", cfg.server.listen));
                l
            }
            Err(e) => {
                log(format!("npud: can't listen on {}: {e}", cfg.server.listen));
                return 1;
            }
        }
    };
    std::env::remove_var("LISTEN_PID");
    std::env::remove_var("LISTEN_FDS");
    {
        let mut st = gw.st.lock().unwrap();
        gw.publish(&mut st, false);
        gw.arm_exit(&mut st);
    }
    gw.start_memory_guard();
    let mut term = signal(SignalKind::terminate()).expect("SIGTERM handler");
    let mut int = signal(SignalKind::interrupt()).expect("SIGINT handler");
    loop {
        tokio::select! {
            r = listener.accept() => {
                if let Ok((sock, _)) = r {
                    tokio::spawn(gw.clone().handle(sock));
                }
            }
            _ = term.recv() => break,
            _ = int.recv() => break,
        }
    }
    {
        let _g = gw.load_lock.lock().await;
        gw.stop().await;
    }
    let mut st = gw.st.lock().unwrap();
    gw.publish(&mut st, true);
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    struct Fake {
        vram: Arc<Mutex<Vram>>,
        avail: Arc<Mutex<u64>>,
        dir: std::path::PathBuf,
    }

    impl Drop for Fake {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// An engine whose GPU/RAM readings the test sets: 16 GB card, 7 GB used, 14 GB RAM available;
    /// a replaced chat server holds 6 GB VRAM + 1 GB RAM. The "big" model is 13 GB.
    fn engine(name: &str) -> (Engine, Fake) {
        let dir = std::env::temp_dir().join(format!("npukit-gw-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("big.gguf"), b"G").unwrap();
        std::fs::write(dir.join("w.bin"), b"W").unwrap();
        let cfg = crate::config::from_str(&format!(r#"
[server]
backend = "llamacpp"
[llamacpp]
vram_reserve = 0.2
ram_reserve_gb = 4
threads = 6
overhead_gb = 0
whisper_model = "{d}/w.bin"
[llamacpp.models]
big = {{ path = "{d}/big.gguf", size_gb = 13 }}
[bar]
file = "{d}/bar.json"
signal = 0
"#, d = dir.display())).unwrap();
        let vram = Arc::new(Mutex::new(Vram { total: 16 * GB, used: 7 * GB }));
        let avail = Arc::new(Mutex::new(14 * GB));
        let (v, a) = (vram.clone(), avail.clone());
        let mut e = Engine::new(cfg);
        e.hw = HwProbe {
            gpu_vram: Arc::new(move || Some(*v.lock().unwrap())),
            mem_available: Arc::new(move || Some(*a.lock().unwrap())),
            proc_vram: Arc::new(|_| 6 * GB),
            proc_rss_anon: Arc::new(|_| GB),
        };
        (e, Fake { vram, avail, dir })
    }

    #[test]
    fn fit_target_is_the_reserve_and_threads_are_capped() {
        let (e, _f) = engine("fit");
        let argv = &e.plan(&[Comp::Chat], Some("big"), None).unwrap()[0].argv;
        let i = argv.iter().position(|a| a == "--fit-target").unwrap();
        assert_eq!(argv[i + 1].parse::<u64>().unwrap(), (16 * GB / 5) / 1048576);
        assert_eq!(argv[argv.iter().position(|a| a == "-t").unwrap() + 1], "6");
        assert!(!argv.contains(&"-ngl".to_string()));
    }

    #[test]
    fn ram_check() {
        let (e, f) = engine("ram");
        let spec = e.spec("big").unwrap();
        // GPU room 16 - 7 - 3.2 = 5.8 GB, so 7.2 GB goes to RAM: 14 - 7.2 >= 4 fits
        e.fits("big", &spec, None).unwrap();
        *f.avail.lock().unwrap() = 10 * GB; // 10 - 7.2 < 4: refused, with the numbers in the message
        let err = e.fits("big", &spec, None).unwrap_err();
        assert!(err.contains("keeping 4 GB free"), "{err}");
        // the replaced chat server's 6 GB VRAM and 1 GB RAM come back: room 11.8 GB, 1.2 GB to RAM, 11 GB available
        e.fits("big", &spec, Some(1)).unwrap();
    }

    #[test]
    fn hf_models_are_sized_before_download() {
        let tree = json!([{"path": "M-Q4_K_M-00001-of-00002.gguf", "size": 6 * GB},
                          {"path": "M-Q4_K_M-00002-of-00002.gguf", "size": 5 * GB},
                          {"path": "M-Q8_0.gguf", "size": 20 * GB},
                          {"path": "mmproj-F16.gguf", "size": GB}, {"path": "README.md", "size": 1}]);
        let size = |r: &str| hf_size_with(r, &|_| Some(tree.clone()));
        assert_eq!(size("u/a"), Some(12 * GB)); // default Q4_K_M, both parts + projector
        assert_eq!(size("u/b:Q8_0"), Some(21 * GB));
        assert_eq!(size("u/c:IQ1_S"), None); // quant not in the repo: unknown, not 0
        assert_eq!(hf_size_with("u/d", &|_| None), None);
    }

    #[test]
    fn whisper_leaves_the_gpu_at_the_reserve() {
        let (e, f) = engine("whisper");
        let argv = &e.plan(&[Comp::Asr], None, None).unwrap()[0].argv;
        assert!(!argv.contains(&"-ng".to_string()));
        f.vram.lock().unwrap().used = 13 * GB; // 3 GB free < 3.2 GB reserve
        let p = &e.plan(&[Comp::Asr], None, None).unwrap()[0];
        assert!(p.argv.contains(&"-ng".to_string()));
        assert_eq!(p.note, "CPU");
        assert_eq!(p.argv[p.argv.iter().position(|a| a == "-t").unwrap() + 1], "6");
    }

    #[test]
    fn per_model_idle_unload() {
        let (e, _f) = engine("idle");
        let mut cfg = e.cfg.clone();
        cfg.llamacpp.models.insert("small".into(), crate::config::ModelEntry::Spec(ModelSpec { path: Some("/x.gguf".into()), idle_unload: Some(1800.0), ..Default::default() }));
        cfg.power.ac_online = "/nonexistent/online".into(); // reads as AC
        let gw = Gateway::new(cfg);
        let base = gw.idle_timeout(None);
        let mk = |m: &str| ChildSet::new(vec![Arc::new(Proc::new(Kind::Llama, vec![Comp::Chat], Some(m.into()), 1, vec!["x".into()], "/health"))], None);
        assert_eq!(gw.idle_timeout(Some(&mk("small"))), 1800.0); // translation model stays warm
        assert_eq!(gw.idle_timeout(Some(&mk("big"))), base); // others unload as usual
    }

    #[test]
    fn memory_guard_kills_servers_when_ram_runs_out() {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async {
            let (e, f) = engine("guard");
            let gw = Gateway::with_engine(e);
            let p = Arc::new(Proc::new(Kind::Llama, vec![Comp::Chat], Some("big".into()), 1, vec!["sleep".into(), "30".into()], "/health"));
            p.start().unwrap();
            gw.st.lock().unwrap().child = Some(ChildSet::new(vec![p.clone()], None));
            assert!(!gw.on_memory_pressure(Some(3 * GB))); // pressure, but RAM left: keep
            assert!(p.alive());
            assert!(gw.on_memory_pressure(Some(GB)));
            assert!(p.wait_exit(Duration::from_secs(5)).await);
            assert!(gw.st.lock().unwrap().last_error.contains("RAM was running out"));
            tokio::time::sleep(Duration::from_millis(100)).await;
            let st = gw.st.lock().unwrap();
            assert!(st.child.is_none());
            assert_eq!(st.state, "error");
            assert!(std::fs::read_to_string(f.dir.join("bar.json")).unwrap().contains("error"));
        });
    }

    #[test]
    fn tail_keeps_the_end() {
        let mut t = Tail::default();
        t.push(&vec![b'a'; 40000]);
        t.push(br#"{"timings":{"predicted_per_second":42.5}}"#);
        assert_eq!(t.buf.len(), 32768);
        let m = TPS.captures(&t.buf).unwrap();
        assert_eq!(&m[1], b"42.5");
    }
}
