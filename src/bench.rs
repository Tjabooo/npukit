//! `npu bench`: tokens/s and power for models via the gateway and, for comparison, any model on another
//! OpenAI-compatible server (e.g. Lemonade's llama.cpp on the iGPU or CPU).
//!
//! Power comes from sysfs, no root needed: amdgpu's PPT sensor is the whole APU package on Ryzen AI
//! laptops, and the battery's power_now is the full system draw while discharging.

use crate::client::{self, ApiError, ChatOpts};
use crate::config;
use crate::hw;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const PROMPT: &str = "Explain how a hash map works, including hashing, collisions (chaining vs open addressing), \
                      load factor and resizing. Use about 300 words.";

/// (package W, battery W) readings.
type Samples = Arc<Mutex<Vec<(Option<f64>, Option<f64>)>>>;

struct Sampler {
    go: Arc<AtomicBool>,
    samples: Samples,
    h: Option<std::thread::JoinHandle<()>>,
}

#[derive(Default, Clone, Copy)]
struct Power {
    ppt_w: Option<f64>,
    ppt_peak_w: Option<f64>,
    bat_w: Option<f64>,
}

impl Sampler {
    fn start() -> Sampler {
        let (ppt, bat) = (hw::ppt_path(), hw::battery_dir());
        let go = Arc::new(AtomicBool::new(true));
        let samples: Samples = Arc::default();
        let (g, s) = (go.clone(), samples.clone());
        let h = std::thread::spawn(move || {
            while g.load(Ordering::Relaxed) {
                s.lock().unwrap().push((hw::watts(&ppt), hw::battery_watts(&bat)));
                std::thread::sleep(Duration::from_millis(200));
            }
        });
        Sampler { go, samples, h: Some(h) }
    }

    fn stop(mut self) -> Power {
        self.go.store(false, Ordering::Relaxed);
        if let Some(h) = self.h.take() {
            let _ = h.join();
        }
        let s = self.samples.lock().unwrap();
        let ppt: Vec<f64> = s.iter().filter_map(|x| x.0).collect();
        let bat: Vec<f64> = s.iter().filter_map(|x| x.1).filter(|b| *b != 0.0).collect();
        let mean = |v: &[f64]| (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64);
        Power { ppt_w: mean(&ppt), ppt_peak_w: ppt.iter().copied().reduce(f64::max), bat_w: mean(&bat) }
    }
}

struct Row {
    label: String,
    tps: f64,
    ttft_s: Option<f64>,
    load_s: f64,
    wall_s: f64,
    tokens: u64,
    p: Power,
    ac: bool,
}

impl Row {
    fn j_per_tok(&self) -> Option<f64> {
        self.p.ppt_w.filter(|_| self.tps > 0.0).map(|w| w / self.tps)
    }
    fn json(&self) -> Value {
        json!({"label": self.label, "tps": self.tps, "ttft_s": self.ttft_s, "load_s": self.load_s, "wall_s": self.wall_s,
               "tokens": self.tokens, "ppt_w": self.p.ppt_w, "ppt_peak_w": self.p.ppt_peak_w, "bat_w": self.p.bat_w,
               "j_per_tok": self.j_per_tok(), "ac": self.ac})
    }
}

fn run_one(model: &str, base: Option<&str>, tokens: u32) -> Result<(f64, Option<f64>, f64, u64, Power), ApiError> {
    let msgs = json!([{"role": "user", "content": PROMPT}]);
    let s = Sampler::start();
    let t0 = Instant::now();
    let out = client::chat(&msgs, Some(model), ChatOpts { max_tokens: tokens, base, temperature: Some(0.0),
                                                           extra: client::no_think(), ..Default::default() });
    let wall = t0.elapsed().as_secs_f64();
    let pw = s.stop();
    let out = out?;
    let n = out.pointer("/usage/completion_tokens").and_then(Value::as_u64)
        .or_else(|| out.pointer("/timings/predicted_n").and_then(Value::as_u64)).unwrap_or(0);
    let mut tps = client::tps(&out);
    let mut ttft = out.pointer("/usage/prefill_duration_ttft").and_then(Value::as_f64)
        .or_else(|| out.pointer("/timings/prompt_ms").and_then(Value::as_f64).filter(|x| *x > 0.0).map(|ms| ms / 1000.0));
    if let (Some(b), None) = (base, tps) {
        // Lemonade keeps per-request stats for every backend
        if let Ok(st) = client::call_at(b, "/api/v1/stats", None, 10.0) {
            tps = st.get("tokens_per_second").and_then(Value::as_f64).or(tps);
            ttft = st.get("time_to_first_token").and_then(Value::as_f64).or(ttft);
        }
    }
    let tps = tps.unwrap_or(if wall > 0.0 { n as f64 / wall } else { 0.0 });
    Ok((tps, ttft, wall, n, pw))
}

pub struct BenchArgs {
    pub models: Vec<String>,
    pub other: Vec<String>,
    pub other_url: String,
    pub pmode: Option<String>,
    pub tokens: u32,
    pub runs: u32,
    pub json: bool,
}

pub fn main(a: &BenchArgs) -> Result<i32, ApiError> {
    let cfg = config::load();
    let ac = hw::on_ac(&hw::ac_online_path(&cfg.power.ac_online));
    let idle = {
        let s = Sampler::start();
        std::thread::sleep(Duration::from_secs(4));
        s.stop()
    };
    println!("power source: {} · idle package {:.1} W{}", if ac { "AC" } else { "battery" }, idle.ppt_w.unwrap_or(0.0),
             match idle.bat_w { Some(b) if !ac => format!(" · system {b:.1} W"), _ => String::new() });
    let mut targets: Vec<(String, Option<String>)> = if a.models.is_empty() && a.other.is_empty() {
        vec![("fast".into(), None)]
    } else {
        a.models.iter().map(|m| (m.clone(), None)).collect()
    };
    let url = a.other_url.trim_end_matches('/').trim_end_matches("/v1").to_string();
    targets.extend(a.other.iter().map(|m| (m.clone(), Some(url.clone()))));
    let pmodes: Vec<Option<String>> = match &a.pmode {
        Some(p) => p.split(',').map(|x| Some(x.trim().to_string())).collect(),
        None => vec![None],
    };
    let mut rows = vec![];
    for (model, base) in &targets {
        let pms = if base.is_none() { pmodes.clone() } else { vec![None] };
        for pm in pms {
            let t0 = Instant::now();
            let (name, label) = if base.is_none() {
                let name = cfg.resolve_model(Some(model));
                client::call("/npu/load", Some(&json!({"component": "chat", "model": name, "pmode": pm})), 600.0)?;
                let label = name.clone() + &pm.as_ref().map(|p| format!(" [{p}]")).unwrap_or_default();
                (name, label)
            } else {
                // MODEL@backend, e.g. Qwen3.5-4B-GGUF@vulkan (Lemonade's llama.cpp backends)
                let (name, backend) = model.split_once('@').unwrap_or((model, ""));
                let mut body = json!({"model_name": name});
                if !backend.is_empty() {
                    body["llamacpp_backend"] = json!(backend);
                }
                if let Err(e) = client::call_at(base.as_ref().unwrap(), "/api/v1/load", Some(&body), 900.0) {
                    println!("  {model}: load failed: {e}");
                    continue;
                }
                (name.to_string(), format!("{name} [{}]", if backend.is_empty() { "default" } else { backend }))
            };
            let load_s = t0.elapsed().as_secs_f64();
            client::chat(&json!([{"role": "user", "content": "hi"}]), Some(&name),
                         ChatOpts { max_tokens: 8, base: base.as_deref(), extra: client::no_think(), ..Default::default() })?;
            let mut best: Option<Row> = None;
            for _ in 0..a.runs.max(1) {
                let (tps, ttft_s, wall_s, tokens, p) = run_one(&name, base.as_deref(), a.tokens)?;
                if best.as_ref().is_none_or(|b| tps > b.tps) {
                    best = Some(Row { label: label.clone(), tps, ttft_s, load_s, wall_s, tokens, p, ac });
                }
            }
            let r = best.unwrap();
            println!("{}", fmt(&r));
            rows.push(r);
            if let Some(b) = base {
                let _ = client::call_at(b, "/api/v1/unload", Some(&json!({})), 60.0);
            }
        }
    }
    if a.json {
        println!("{}", serde_json::to_string_pretty(&rows.iter().map(Row::json).collect::<Vec<_>>()).unwrap());
    }
    Ok(0)
}

fn fmt(r: &Row) -> String {
    let f = |v: Option<f64>, w: usize, p: usize| v.map(|x| format!("{x:w$.p$}")).unwrap_or_else(|| "-".into());
    format!("  {:<40} {} tok/s  ttft {} s  pkg {} W (peak {})  {}{} J/tok  load {} s",
            r.label, f(Some(r.tps), 6, 1), f(r.ttft_s, 5, 2), f(r.p.ppt_w, 5, 1), f(r.p.ppt_peak_w, 4, 1),
            if r.ac { String::new() } else { format!("sys {} W  ", f(r.p.bat_w, 5, 1)) },
            f(r.j_per_tok(), 5, 2), f(Some(r.load_s), 4, 1))
}
