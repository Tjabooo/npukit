//! Configuration: built-in defaults (defaults.toml), overridden by ~/.config/npukit/config.toml.

use indexmap::IndexMap;
use serde::{Deserialize, Deserializer};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const DEFAULTS: &str = include_str!("defaults.toml");

/// A command-line argument from the config: strings, or numbers/bools written without quotes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Arg(pub String);

impl<'de> Deserialize<'de> for Arg {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Any {
            S(String),
            I(i64),
            F(f64),
            B(bool),
        }
        Ok(Arg(match Any::deserialize(d)? {
            Any::S(s) => s,
            Any::I(i) => i.to_string(),
            Any::F(f) => f.to_string(),
            Any::B(b) => if b { "True".into() } else { "False".into() },
        }))
    }
}

pub fn strs(v: &[Arg]) -> Vec<String> {
    v.iter().map(|a| a.0.clone()).collect()
}

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub server: Server,
    pub llamacpp: Llamacpp,
    pub models: Models,
    pub power: Power,
    pub bar: Bar,
    pub dictate: Dictate,
    pub ocr: Ocr,
    pub translate: Translate,
    pub tools: Tools,
    pub ui: Ui,
    pub search: Search,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Server {
    pub listen: String,
    pub flm: String,
    pub flm_port: u16,
    pub flm_args: Vec<Arg>,
    pub max_components: usize,
    pub exit_after: f64,
    pub start_timeout: f64,
    pub backend: String,
    pub memory_guard_gb: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Llamacpp {
    pub server: String,
    pub whisper: String,
    pub label: String,
    pub max_components: usize,
    pub vram_reserve: f64,
    pub ram_reserve_gb: f64,
    pub threads: i64,
    pub overhead_gb: f64,
    pub args: Vec<Arg>,
    pub whisper_model: String,
    pub whisper_args: Vec<Arg>,
    pub embed_model: String,
    pub embed_args: Vec<Arg>,
    pub download_timeout: f64,
    pub aliases: IndexMap<String, String>,
    pub models: IndexMap<String, ModelEntry>,
}

/// A [llamacpp.models] entry: a table, or a bare "….gguf" path / "user/repo[:quant]" string.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum ModelEntry {
    Short(String),
    Spec(ModelSpec),
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct ModelSpec {
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub mmproj: Option<String>,
    #[serde(default)]
    pub hf: Option<String>,
    #[serde(default)]
    pub args: Vec<Arg>,
    #[serde(default)]
    pub idle_unload: Option<f64>,
    #[serde(default)]
    pub size_gb: Option<f64>,
}

impl ModelEntry {
    pub fn spec(&self) -> ModelSpec {
        match self {
            ModelEntry::Spec(s) => s.clone(),
            ModelEntry::Short(s) if s.ends_with(".gguf") => ModelSpec { path: Some(s.clone()), ..Default::default() },
            ModelEntry::Short(s) => ModelSpec { hf: Some(s.clone()), ..Default::default() },
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Models {
    pub aliases: IndexMap<String, String>,
    pub default: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Power {
    pub ac_online: String,
    pub ac_pmode: String,
    pub battery_pmode: String,
    pub ac_idle_unload: f64,
    pub battery_idle_unload: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Bar {
    pub file: String,
    pub signal: i32,
    pub tag: String,
    pub colors: IndexMap<String, String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Dictate {
    pub record: Vec<Arg>,
    pub max_seconds: f64,
    pub language: String,
    pub prompt: String,
    #[serde(rename = "type")]
    pub type_cmd: Vec<Arg>,
    pub copy: bool,
    pub min_rms: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Ocr {
    pub capture: Vec<Arg>,
    pub model: String,
    pub prompt: String,
    pub describe_prompt: String,
    pub translate_to: String,
    pub translate_model: String,
    pub translate_ocr_model: String,
    pub translate_prompt: String,
    pub translate_context: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Translate {
    pub to: String,
    pub model: String,
    pub prompt: String,
    pub languages: Vec<String>,
    pub copy: bool,
    #[serde(rename = "type")]
    pub type_cmd: Vec<Arg>,
    pub type_delay: f64,
    pub eww: String,
    pub on_open: Vec<Arg>,
    pub on_close: Vec<Arg>,
    pub font: String,
    pub width: f64,
    pub colors: IndexMap<String, String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Tools {
    pub read: Vec<String>,
    pub deny: Vec<String>,
    pub skip_dirs: Vec<String>,
    pub model: String,
    pub max_steps: usize,
    pub context_chars: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Ui {
    pub menu_theme: String,
    pub search_theme: String,
    pub lang_theme: String,
    pub menu_close_wait: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Search {
    pub roots: Vec<String>,
    pub db: String,
    pub exclude_dirs: Vec<String>,
    pub extensions: Vec<String>,
    pub names: Vec<String>,
    pub exclude_globs: Vec<String>,
    pub max_file_kb: u64,
    pub max_chunks_per_file: usize,
    pub chunk_lines: usize,
    pub overlap_lines: usize,
    pub batch: usize,
    pub results: usize,
    pub open: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Flm,
    LlamaCpp,
}

pub fn config_path() -> PathBuf {
    if let Some(p) = std::env::var_os("NPUKIT_CONFIG") {
        return PathBuf::from(p);
    }
    let base = std::env::var("XDG_CONFIG_HOME").ok().filter(|s| !s.is_empty())
        .map(PathBuf::from).unwrap_or_else(|| home().join(".config"));
    base.join("npukit").join("config.toml")
}

fn merge(base: &mut toml::Table, over: toml::Table, path: &[&str]) {
    for (k, v) in over {
        // alias tables and single model entries replace the default; everything else merges
        let flat = k == "aliases" || path == ["llamacpp", "models"];
        match (base.get_mut(&k), v) {
            (Some(toml::Value::Table(b)), toml::Value::Table(o)) if !flat => {
                let mut p = path.to_vec();
                p.push(&k);
                merge(b, o, &p);
            }
            (_, v) => {
                base.insert(k, v);
            }
        }
    }
}

/// Defaults merged with a user config (TOML text); the llamacpp alias rule applied.
pub fn from_str(user: &str) -> Result<Config, String> {
    let mut cfg: toml::Table = DEFAULTS.parse().map_err(|e| format!("defaults: {e}"))?;
    let user: toml::Table = user.parse().map_err(|e: toml::de::Error| e.to_string())?;
    let user_aliases = user.get("models").and_then(|m| m.get("aliases")).is_some();
    merge(&mut cfg, user, &[]);
    let mut c: Config = toml::Value::Table(cfg).try_into().map_err(|e: toml::de::Error| e.to_string())?;
    if c.backend() == Backend::LlamaCpp && !user_aliases {
        c.models.aliases = c.llamacpp.aliases.clone();
    }
    Ok(c)
}

pub fn load_from(p: &Path) -> Result<Config, String> {
    let text = match std::fs::read_to_string(p) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(format!("{}: {e}", p.display())),
    };
    from_str(&text).map_err(|e| format!("{}: {e}", p.display()))
}

static CONFIG: OnceLock<Config> = OnceLock::new();

/// The config for this process (read once).
pub fn load() -> &'static Config {
    CONFIG.get_or_init(|| match load_from(&config_path()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("npu: config: {e}");
            std::process::exit(2);
        }
    })
}

impl Config {
    pub fn backend(&self) -> Backend {
        match self.server.backend.to_lowercase().as_str() {
            "llamacpp" | "llama.cpp" | "llama" => Backend::LlamaCpp,
            _ => Backend::Flm,
        }
    }

    /// Where the models run, for messages: "NPU", or [llamacpp] label (default "GPU").
    pub fn engine_label(&self) -> String {
        match self.backend() {
            Backend::LlamaCpp if !self.llamacpp.label.is_empty() => self.llamacpp.label.clone(),
            Backend::LlamaCpp => "GPU".into(),
            Backend::Flm => "NPU".into(),
        }
    }

    pub fn base_url(&self) -> String {
        format!("http://{}", self.server.listen)
    }

    /// Alias or tag -> tag. Also accepts Lemonade-style names ("qwen3.5-4b-FLM").
    pub fn resolve_model(&self, name: Option<&str>) -> String {
        let mut name = name.filter(|s| !s.trim().is_empty()).unwrap_or(&self.models.default).trim().to_string();
        for _ in 0..4 {
            // aliases may point at aliases
            if let Some(t) = self.models.aliases.get(&name) {
                name = t.clone();
            }
        }
        if let Some(stem) = name.strip_suffix("-FLM") {
            if let Some((head, size)) = stem.rsplit_once('-') {
                if !head.is_empty() {
                    name = format!("{head}:{size}");
                }
            }
        }
        name
    }

    /// The [llamacpp.models] entry of a tag, if any.
    pub fn model_entry(&self, tag: &str) -> Option<ModelSpec> {
        self.llamacpp.models.get(tag).map(|e| e.spec())
    }
}

pub fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/"))
}

/// ~ and $VARS expanded, like Python's os.path.expandvars(os.path.expanduser(p)).
pub fn expand(p: &str) -> PathBuf {
    let s = if p == "~" {
        home().to_string_lossy().into_owned()
    } else if let Some(rest) = p.strip_prefix("~/") {
        format!("{}/{}", home().to_string_lossy().trim_end_matches('/'), rest)
    } else {
        p.to_string()
    };
    PathBuf::from(expand_vars(&s))
}

fn expand_vars(s: &str) -> String {
    if !s.contains('$') {
        return s.to_string();
    }
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'$' {
            let (name, end) = if b.get(i + 1) == Some(&b'{') {
                match s[i + 2..].find('}') {
                    Some(j) => (&s[i + 2..i + 2 + j], i + 3 + j),
                    None => ("", i + 1),
                }
            } else {
                let j = s[i + 1..].find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).map_or(s.len(), |j| i + 1 + j);
                (&s[i + 1..j], j)
            };
            if !name.is_empty() {
                if let Ok(v) = std::env::var(name) {
                    out.push_str(&v);
                    i = end;
                    continue;
                }
            }
            out.push('$');
            i += 1;
            continue;
        }
        let ch = s[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

pub fn expand_str(p: &str) -> String {
    expand(p).to_string_lossy().into_owned()
}

/// $XDG_RUNTIME_DIR/npukit (created 0700).
pub fn runtime_dir() -> PathBuf {
    let base = std::env::var("XDG_RUNTIME_DIR").ok().filter(|s| !s.is_empty())
        .map(PathBuf::from).unwrap_or_else(|| PathBuf::from(format!("/tmp/npukit-{}", unsafe { libc::getuid() })));
    let d = base.join("npukit");
    if !d.is_dir() {
        use std::os::unix::fs::DirBuilderExt;
        let _ = std::fs::DirBuilder::new().recursive(true).mode(0o700).create(&d);
    }
    d
}

pub fn state_dir() -> PathBuf {
    std::env::var("XDG_STATE_HOME").ok().filter(|s| !s.is_empty())
        .map(PathBuf::from).unwrap_or_else(|| home().join(".local/state"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_and_lemonade_names() {
        let c = from_str("").unwrap();
        assert_eq!(c.resolve_model(Some("fast")), "qwen3.5:4b");
        assert_eq!(c.resolve_model(None), "qwen3.5:4b");
        assert_eq!(c.resolve_model(Some("qwen3.5-4b-FLM")), "qwen3.5:4b");
        assert_eq!(c.resolve_model(Some("gpt-oss-20b-FLM")), "gpt-oss:20b");
        assert_eq!(c.resolve_model(Some("llama3.2:1b")), "llama3.2:1b");
        assert_eq!(c.backend(), Backend::Flm);
        assert_eq!(c.engine_label(), "NPU");
    }

    #[test]
    fn llamacpp_uses_gpu_aliases_and_merges_model_entries() {
        let c = from_str("[server]\nbackend = \"llamacpp\"\n[llamacpp.models]\n\"qwen3.5:9b\" = { path = \"/x.gguf\", idle_unload = 1800 }\n").unwrap();
        assert_eq!(c.backend(), Backend::LlamaCpp);
        assert_eq!(c.engine_label(), "GPU");
        assert_eq!(c.resolve_model(Some("fast")), "qwen3.5:9b");
        assert!(c.llamacpp.models.contains_key("gpt-oss:20b"));
        let s = c.model_entry("qwen3.5:9b").unwrap();
        assert_eq!(s.hf, None);
        assert_eq!(s.mmproj, None); // the entry replaces the default, it doesn't merge into it
        assert_eq!(s.idle_unload, Some(1800.0));
    }

    #[test]
    fn user_aliases_replace_and_ints_work_as_floats() {
        let c = from_str("[models.aliases]\nfast = \"x:1b\"\n[llamacpp]\nram_reserve_gb = 2\nargs = [\"-c\", 8192]\n").unwrap();
        assert_eq!(c.models.aliases.len(), 1);
        assert_eq!(c.llamacpp.ram_reserve_gb, 2.0);
        assert_eq!(strs(&c.llamacpp.args), vec!["-c", "8192"]);
        assert_eq!(c.llamacpp.whisper_args.len(), 2); // untouched keys keep their defaults
    }

    #[test]
    fn expand_home_and_vars() {
        std::env::set_var("NPUKIT_T_VAR", "v");
        assert_eq!(expand("$NPUKIT_T_VAR/x"), PathBuf::from("v/x"));
        assert_eq!(expand("${NPUKIT_T_VAR}y"), PathBuf::from("vy"));
        assert_eq!(expand("$NOPE_NOT_SET_X/a"), PathBuf::from("$NOPE_NOT_SET_X/a"));
        assert!(expand("~/a").ends_with("a"));
    }
}
