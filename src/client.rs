//! Tiny blocking HTTP/1.1 client for the gateway (and any OpenAI-compatible server on plain http).

use crate::config;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::Mutex;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct ApiError(pub String);

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ApiError {}

pub type Result<T> = std::result::Result<T, ApiError>;

/// The connection of the request in flight, so another thread can cut it (Ctrl-C in `npu chat`).
pub static CURRENT: Mutex<Option<TcpStream>> = Mutex::new(None);

pub fn cancel_current() {
    if let Some(s) = CURRENT.lock().unwrap().take() {
        let _ = s.shutdown(std::net::Shutdown::Both);
    }
}

fn split_url(url: &str) -> Result<(String, String)> {
    let rest = url.strip_prefix("http://").ok_or_else(|| ApiError(format!("only http:// URLs are supported: {url}")))?;
    let (host, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    Ok((host.to_string(), path.to_string()))
}

/// Send one request; returns (status, body).
pub fn request(url: &str, method: &str, content_type: &str, body: &[u8], timeout: f64) -> Result<(u16, Vec<u8>)> {
    let (host, path) = split_url(url)?;
    let unreachable = |e: &dyn std::fmt::Display| ApiError(format!("gateway not reachable at {url}: {e}"));
    let addr = (if host.contains(':') { host.clone() } else { format!("{host}:80") })
        .to_socket_addrs().map_err(|e| unreachable(&e))?
        .next().ok_or_else(|| unreachable(&"no address"))?;
    let t = Duration::from_secs_f64(timeout.max(0.001));
    let mut s = TcpStream::connect_timeout(&addr, t.min(Duration::from_secs(10))).map_err(|e| unreachable(&e))?;
    let _ = s.set_nodelay(true);
    s.set_read_timeout(Some(t)).ok();
    s.set_write_timeout(Some(t)).ok();
    if let Ok(c) = s.try_clone() {
        *CURRENT.lock().unwrap() = Some(c);
    }
    let mut head = format!("{method} {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\nAccept: application/json\r\n");
    if !body.is_empty() || method == "POST" {
        head += &format!("Content-Type: {content_type}\r\nContent-Length: {}\r\n", body.len());
    }
    head += "\r\n";
    let res = (|| -> std::io::Result<(u16, Vec<u8>)> {
        s.write_all(head.as_bytes())?;
        s.write_all(body)?;
        read_response(&mut BufReader::with_capacity(65536, &s))
    })();
    *CURRENT.lock().unwrap() = None;
    res.map_err(|e| match e.kind() {
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => ApiError(format!("timed out after {timeout:.0}s: {url}")),
        _ => ApiError(format!("connection to {url} failed: {e}")),
    })
}

fn read_response(r: &mut impl BufRead) -> std::io::Result<(u16, Vec<u8>)> {
    let mut line = String::new();
    let (mut len, mut chunked) = (None::<usize>, false);
    let status = loop {
        // skip 1xx responses
        line.clear();
        r.read_line(&mut line)?;
        let status: u16 = line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
        loop {
            line.clear();
            if r.read_line(&mut line)? == 0 || line == "\r\n" || line == "\n" {
                break;
            }
            if let Some((k, v)) = line.split_once(':') {
                let (k, v) = (k.trim().to_ascii_lowercase(), v.trim());
                if k == "content-length" {
                    len = v.parse().ok();
                } else if k == "transfer-encoding" && v.to_ascii_lowercase().contains("chunked") {
                    chunked = true;
                }
            }
        }
        if !(100..200).contains(&status) {
            break status;
        }
    };
    let mut body = Vec::new();
    if chunked {
        loop {
            line.clear();
            r.read_line(&mut line)?;
            let size = usize::from_str_radix(line.split(';').next().unwrap_or("").trim(), 16).unwrap_or(0);
            if size == 0 {
                break;
            }
            let start = body.len();
            body.resize(start + size, 0);
            r.read_exact(&mut body[start..])?;
            line.clear();
            r.read_line(&mut line)?;
        }
    } else if let Some(n) = len {
        body.resize(n, 0);
        r.read_exact(&mut body)?;
    } else {
        r.read_to_end(&mut body)?;
    }
    Ok((status, body))
}

/// Decode a JSON reply; HTTP errors and FastFlowLM's HTTP 200 + {"error": …} become ApiError.
fn decode(status: u16, body: &[u8]) -> Result<Value> {
    if status >= 400 {
        let text = String::from_utf8_lossy(body);
        let err = match serde_json::from_slice::<Value>(body) {
            Ok(v) => match v.get("error") {
                Some(Value::Object(o)) => o.get("message").map(as_text).unwrap_or_else(|| Value::Object(o.clone()).to_string()),
                Some(e) => as_text(e),
                None => text.into_owned(),
            },
            Err(_) => text.into_owned(),
        };
        return Err(ApiError(format!("{status}: {err}")));
    }
    let out: Value = if body.is_empty() { json!({}) } else {
        serde_json::from_slice(body).map_err(|e| ApiError(format!("bad JSON from server: {e}")))?
    };
    if let Some(o) = out.as_object() {
        if let Some(err) = o.get("error") {
            if !["data", "choices", "text"].iter().any(|k| o.contains_key(*k)) {
                return Err(ApiError(match err {
                    Value::Object(e) => e.get("message").map(as_text).unwrap_or_else(|| err.to_string()),
                    e => as_text(e),
                }));
            }
        }
    }
    Ok(out)
}

fn as_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        v => v.to_string(),
    }
}

pub fn call_at(base: &str, path: &str, payload: Option<&Value>, timeout: f64) -> Result<Value> {
    let (method, body) = match payload {
        Some(p) => ("POST", serde_json::to_vec(p).unwrap()),
        None => ("GET", Vec::new()),
    };
    let (status, body) = request(&format!("{base}{path}"), method, "application/json", &body, timeout)?;
    decode(status, &body)
}

/// GET (no payload) or POST JSON to the gateway.
pub fn call(path: &str, payload: Option<&Value>, timeout: f64) -> Result<Value> {
    call_at(&config::load().base_url(), path, payload, timeout)
}

/// Fire-and-forget activity label for the bar.
pub fn activity(name: Option<&str>) {
    let _ = call("/npu/activity", Some(&json!({"activity": name})), 5.0);
}

pub fn transcribe(wav: &[u8], language: &str, prompt: &str, timeout: f64) -> Result<String> {
    let boundary = format!("npukit{:x}{:x}", std::process::id(), crate::util::now().to_bits());
    let mut body = Vec::with_capacity(wav.len() + 1024);
    let mut field = |k: &str, v: &str| {
        body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").as_bytes());
    };
    field("model", "whisper-v3:turbo");
    field("response_format", "json");
    if !language.is_empty() {
        field("language", language);
    }
    if !prompt.is_empty() {
        field("prompt", prompt);
    }
    body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n").as_bytes());
    body.extend_from_slice(wav);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    let url = config::load().base_url() + "/v1/audio/transcriptions";
    let (status, out) = request(&url, "POST", &format!("multipart/form-data; boundary={boundary}"), &body, timeout)?;
    let v = decode(status, &out)?;
    Ok(v.get("text").and_then(Value::as_str).unwrap_or("").trim().to_string())
}

pub struct ChatOpts<'a> {
    pub max_tokens: u32,
    pub base: Option<&'a str>,
    pub temperature: Option<f64>,
    pub timeout: f64,
    pub extra: Value,
}

impl Default for ChatOpts<'_> {
    fn default() -> Self {
        ChatOpts { max_tokens: 1024, base: None, temperature: None, timeout: 600.0, extra: json!({}) }
    }
}

pub fn no_think() -> Value {
    json!({"chat_template_kwargs": {"enable_thinking": false}})
}

pub fn chat(messages: &Value, model: Option<&str>, o: ChatOpts) -> Result<Value> {
    let cfg = config::load();
    let mut body = json!({
        "model": model.unwrap_or(&cfg.models.default),
        "messages": messages,
        "max_tokens": o.max_tokens,
    });
    if let Some(t) = o.temperature {
        body["temperature"] = json!(t);
    }
    if let Value::Object(extra) = o.extra {
        for (k, v) in extra {
            body[k] = v;
        }
    }
    let base = o.base.map(str::to_string).unwrap_or_else(|| cfg.base_url());
    call_at(&base, "/v1/chat/completions", Some(&body), o.timeout)
}

pub fn embed(texts: &[String], timeout: f64) -> Result<Vec<Vec<f32>>> {
    let out = call("/v1/embeddings", Some(&json!({"model": "embed-gemma:300m", "input": texts})), timeout)?;
    let mut data: Vec<&Value> = out.get("data").and_then(Value::as_array).map(|a| a.iter().collect()).unwrap_or_default();
    data.sort_by_key(|d| d.get("index").and_then(Value::as_i64).unwrap_or(0));
    let vecs: Vec<Vec<f32>> = data.iter().map(|d| {
        d.get("embedding").and_then(Value::as_array)
            .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect())
            .unwrap_or_default()
    }).collect();
    if vecs.len() != texts.len() {
        return Err(ApiError(format!("embeddings: asked for {}, got {}", texts.len(), vecs.len())));
    }
    Ok(vecs)
}

/// Text of the first choice of a chat reply.
pub fn content(out: &Value) -> String {
    out.pointer("/choices/0/message/content").and_then(Value::as_str).unwrap_or("").trim().to_string()
}

/// tok/s of a reply: FastFlowLM usage.decoding_speed_tps or llama.cpp timings.predicted_per_second.
pub fn tps(out: &Value) -> Option<f64> {
    out.pointer("/usage/decoding_speed_tps").and_then(Value::as_f64)
        .or_else(|| out.pointer("/timings/predicted_per_second").and_then(Value::as_f64))
        .filter(|x| *x > 0.0)
}
