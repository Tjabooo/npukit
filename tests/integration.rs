//! End to end: `npu serve` against stand-ins for flm and llama-server/whisper-server (tests/fake_*.py),
//! and the CLI tools against that gateway. Runs anywhere: no NPU, no GPU.

use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

const NPU: &str = env!("CARGO_BIN_EXE_npu");

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

fn fake(name: &str) -> PathBuf {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join(name);
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    p
}

/// Gateway under test: a temp dir with its config, the `npu serve` process, its port.
struct Gw {
    dir: PathBuf,
    port: u16,
    proc: Child,
}

impl Gw {
    fn start(name: &str, config: impl Fn(&Path, u16) -> String) -> Gw {
        let dir = std::env::temp_dir().join(format!("npukit-it-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let port = free_port();
        std::fs::write(dir.join("config.toml"), config(&dir, port)).unwrap();
        let proc = Command::new(NPU).arg("serve").envs(env(&dir)).stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap();
        let gw = Gw { dir, port, proc };
        for _ in 0..100 {
            if gw.get("/npu/status").is_some() {
                return gw;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        panic!("gateway did not start");
    }

    fn http(&self, method: &str, path: &str, ctype: &str, body: &[u8]) -> Option<(u16, Value)> {
        let mut s = TcpStream::connect(("127.0.0.1", self.port)).ok()?;
        s.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
        let head = format!("{method} {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\n\r\n", body.len());
        s.write_all(head.as_bytes()).ok()?;
        s.write_all(body).ok()?;
        let mut out = vec![];
        s.read_to_end(&mut out).ok()?;
        let text = String::from_utf8_lossy(&out);
        let (head, body) = text.split_once("\r\n\r\n")?;
        let code = head.split_whitespace().nth(1)?.parse().ok()?;
        Some((code, serde_json::from_str(body).unwrap_or(Value::Null)))
    }

    fn get(&self, path: &str) -> Option<Value> {
        self.http("GET", path, "application/json", b"").map(|r| r.1)
    }

    fn post(&self, path: &str, v: &Value) -> (u16, Value) {
        self.http("POST", path, "application/json", v.to_string().as_bytes()).expect("gateway answers")
    }

    fn status(&self) -> Value {
        self.get("/npu/status").unwrap()
    }

    fn chat(&self, model: &str) -> (u16, Value) {
        self.post("/v1/chat/completions", &json!({"model": model, "messages": [{"role": "user", "content": "hi"}]}))
    }

    fn reply(&self, model: &str) -> String {
        let (code, v) = self.chat(model);
        assert_eq!(code, 200, "{v}");
        v.pointer("/choices/0/message/content").and_then(Value::as_str).unwrap_or("").to_string()
    }

    fn transcribe(&self) -> String {
        let b = "XyZb";
        let body = format!("--{b}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper\r\n--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\nRIFF....fake\r\n--{b}--\r\n");
        let (code, v) = self.http("POST", "/v1/audio/transcriptions", &format!("multipart/form-data; boundary={b}"), body.as_bytes()).unwrap();
        assert_eq!(code, 200, "{v}");
        v["text"].as_str().unwrap_or("").to_string()
    }

    fn components(&self) -> Vec<String> {
        let mut c: Vec<String> = self.status()["components"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
        c.sort();
        c
    }

    fn pids(&self) -> std::collections::HashMap<String, u64> {
        self.status()["procs"].as_array().unwrap().iter()
            .map(|p| (p["components"][0].as_str().unwrap().to_string(), p["pid"].as_u64().unwrap())).collect()
    }

    fn npu(&self, args: &[&str]) -> Output {
        Command::new(NPU).args(args).envs(env(&self.dir)).stdin(Stdio::null()).output().unwrap()
    }

    fn bar_settled(&self) -> Value {
        // the reply reaches us just before the gateway marks the request done
        let mut bar = Value::Null;
        for _ in 0..50 {
            bar = serde_json::from_str(&std::fs::read_to_string(self.dir.join("bar.json")).unwrap()).unwrap();
            if !bar["class"].as_array().unwrap().iter().any(|c| c == "busy") {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        bar
    }

    fn wait_state(&self, want: &str, secs: u64) -> bool {
        let end = Instant::now() + Duration::from_secs(secs);
        while Instant::now() < end {
            if self.status()["state"] == want {
                return true;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        false
    }
}

impl Drop for Gw {
    fn drop(&mut self) {
        unsafe { libc_kill(self.proc.id() as i32, 15) };
        let end = Instant::now() + Duration::from_secs(10);
        while Instant::now() < end && self.proc.try_wait().ok().flatten().is_none() {
            std::thread::sleep(Duration::from_millis(20));
        }
        let _ = self.proc.kill();
        let _ = self.proc.wait();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

extern "C" {
    #[link_name = "kill"]
    fn libc_kill(pid: i32, sig: i32) -> i32;
}

fn env(dir: &Path) -> Vec<(String, String)> {
    vec![
        ("NPUKIT_CONFIG".into(), dir.join("config.toml").display().to_string()),
        ("XDG_STATE_HOME".into(), dir.join("state").display().to_string()),
        ("XDG_RUNTIME_DIR".into(), dir.join("run").display().to_string()),
        ("PATH".into(), "/usr/bin:/bin".into()), // no wl-copy/notify-send/wtype from a desktop session
        ("HOME".into(), dir.display().to_string()),
    ]
}

#[test]
fn flm_gateway() {
    let gw = Gw::start("flm", |d, port| format!(r#"
[server]
listen = "127.0.0.1:{port}"
flm = "{flm}"
flm_port = {fp}
start_timeout = 20
[power]
ac_online = "{d}/ac_online"
ac_pmode = "balanced"
battery_pmode = "powersaver"
ac_idle_unload = 3
battery_idle_unload = 1
[bar]
file = "{d}/bar.json"
signal = 0
[search]
roots = ["{d}/notes"]
db = "{d}/index.sqlite"
"#, d = d.display(), flm = fake("fake_flm.py").display(), fp = free_port()));
    let d = &gw.dir;
    std::fs::write(d.join("ac_online"), "1\n").unwrap();
    std::fs::create_dir_all(d.join("notes")).unwrap();
    std::fs::write(d.join("notes/rust.md"), "# Rust notes\nownership and borrowing in rust\n").unwrap();
    std::fs::write(d.join("notes/cook.txt"), "pancakes: flour, milk, eggs\n").unwrap();
    std::fs::write(d.join("notes/tool"), "#!/bin/sh\necho rust tool\n").unwrap();
    std::fs::write(d.join("notes/blob"), "no shebang, no extension\n").unwrap();

    // 1: the alias is resolved; pmode follows the power source
    let text = gw.reply("fast");
    assert!(text.contains("model=qwen3.5:4b") && text.contains("pmode=balanced"), "{text}");
    let st = gw.status();
    assert_eq!(st["components"], json!(["chat"]));
    assert_eq!(st["last_tps"], json!(12.5));

    // 2: a helper joins the chat model
    gw.reply("fast");
    let (code, v) = gw.post("/v1/embeddings", &json!({"model": "embed-gemma:300m", "input": ["a", "rust"]}));
    assert_eq!(code, 200);
    assert_eq!(v["data"].as_array().unwrap().len(), 2);
    assert_eq!(gw.components(), ["chat", "embed"]);

    // 3: a third component drops the oldest (the NPU holds two)
    gw.reply("fast");
    gw.post("/v1/embeddings", &json!({"input": ["a"]}));
    assert!(gw.transcribe().contains("heard"));
    let c = gw.components();
    assert!(c.contains(&"asr".to_string()) && c.len() == 2, "{c:?}");

    // 4: unknown endpoints pass through to flm, which answers HTTP 200 + {"error": …}
    gw.reply("fast");
    let (_, v) = gw.post("/v1/unknown-endpoint", &json!({"x": 1}));
    assert!(v["error"].as_str().unwrap_or("").contains("unknown"), "{v}");
    let out = gw.npu(&["ask", "hello"]);
    assert!(String::from_utf8_lossy(&out.stdout).contains("model=qwen3.5:4b"));

    // 6: the waybar module and the machine-readable twin
    gw.reply("fast");
    let bar = gw.bar_settled();
    assert!(bar["text"].as_str().unwrap().contains("qwen3.5 4b"), "{bar}");
    assert!(bar["class"].as_array().unwrap().iter().any(|c| c == "ready"));
    let st: Value = serde_json::from_str(&std::fs::read_to_string(d.join("status.json")).unwrap()).unwrap();
    assert_eq!(st["chat_model"], "qwen3.5:4b");
    assert!(st["state"] == "ready" || st["state"] == "busy");
    let out = gw.npu(&["status"]);
    assert!(String::from_utf8_lossy(&out.stdout).contains("loaded    qwen3.5:4b"), "{}", String::from_utf8_lossy(&out.stdout));

    // 7: index + search; extensionless scripts with a shebang are indexed, other extensionless files not
    let out = gw.npu(&["index", "-q"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let out = gw.npu(&["search", "-n", "2", "rust"]);
    let first = String::from_utf8_lossy(&out.stdout).lines().next().unwrap_or("").to_string();
    assert!(first.contains("rust.md"), "{first}");
    let con = Command::new("sh").arg("-c").arg(format!("echo 'select path from files;' | sqlite3 {}/index.sqlite", d.display())).output();
    if let Some(o) = con.ok().filter(|o| o.status.success()) {
        let files = String::from_utf8_lossy(&o.stdout).to_string();
        assert!(files.contains("/notes/tool") && !files.contains("/notes/blob"), "{files}");
    }
    let again = gw.npu(&["index"]);
    assert!(String::from_utf8_lossy(&again.stdout).contains("index up to date"));

    // 8: OCR + translate a file: original on stderr, then ---, the translation on stdout
    std::fs::write(d.join("shot.png"), [b"\x89PNG\r\n\x1a\n".as_slice(), &[b'0'; 16]].concat()).unwrap();
    let out = gw.npu(&["ocr", "--file", &d.join("shot.png").display().to_string(), "-t"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stdout).contains("model="));
    assert!(String::from_utf8_lossy(&out.stderr).contains("---"));

    // 5 (last: it unloads): on battery the pmode is powersaver and the model leaves after 1 s idle
    std::fs::write(d.join("ac_online"), "0\n").unwrap();
    let text = gw.reply("quality");
    assert!(text.contains("pmode=powersaver"), "{text}");
    assert_eq!(gw.status()["idle_unload"], json!(1.0));
    assert!(gw.wait_state("off", 5));
    let out = gw.npu(&["stop"]);
    assert!(out.status.success());
}

#[test]
fn llamacpp_gateway() {
    let gw = Gw::start("llama", |d, port| {
        for f in ["q9.gguf", "mm.gguf", "small.gguf", "whisper.bin"] {
            std::fs::write(d.join(f), b"GGUF").unwrap();
        }
        std::fs::write(d.join("ac_online"), "1\n").unwrap(); // not the host's charger: its state changes the timeouts
        let f = fake("fake_llama.py");
        format!(r#"
[server]
listen = "127.0.0.1:{port}"
backend = "llamacpp"
flm_port = {fp}
start_timeout = 20
[llamacpp]
server = "{f}"
whisper = "{f}"
whisper_model = "{d}/whisper.bin"
vram_reserve = 0     # headroom rules read the host's GPU/RAM: unit-tested in gateway.rs instead
ram_reserve_gb = 0
[llamacpp.models]
"qwen3.5:9b" = {{ path = "{d}/q9.gguf", mmproj = "{d}/mm.gguf" }}
small = {{ path = "{d}/small.gguf" }}
gone = {{ path = "{d}/missing.gguf" }}
[power]
ac_online = "{d}/ac_online"
ac_idle_unload = 3
battery_idle_unload = 3
[bar]
file = "{d}/bar.json"
signal = 0
"#, d = d.display(), f = f.display(), fp = free_port())
    });

    // 1: chat with the vision projector, llama.cpp timings
    let text = gw.reply("fast");
    assert!(text.contains("alias=qwen3.5:9b") && text.contains("vision=True"), "{text}");
    let st = gw.status();
    assert_eq!(st["components"], json!(["chat"]));
    assert_eq!(st["backend"], "llamacpp");
    assert_eq!(st["pmode"], Value::Null);
    assert_eq!(st["last_tps"], json!(88.5));

    // 2: Whisper joins without restarting the chat server
    let before = gw.pids()["chat"];
    assert!(gw.transcribe().contains("whisper"));
    assert_eq!(gw.components(), ["asr", "chat"]);
    assert_eq!(gw.pids()["chat"], before);

    // 3: switching the chat model keeps Whisper
    let pids = gw.pids();
    let text = gw.reply("small");
    assert!(text.contains("alias=small") && text.contains("vision=False"), "{text}");
    let after = gw.pids();
    assert_eq!(after["asr"], pids["asr"]);
    assert_ne!(after["chat"], pids["chat"]);

    // 4: bad models fail before anything unloads
    gw.reply("fast");
    let pids = gw.pids();
    let (code, v) = gw.chat("gone");
    assert_eq!(code, 503);
    assert!(v.to_string().contains("missing"), "{v}");
    let (_, v) = gw.chat("no-such-model");
    assert!(v.to_string().contains("unknown model"), "{v}");
    assert_eq!(gw.pids()["chat"], pids["chat"]);

    // 45: ask with file access: the model calls read_file, the answer quotes the file
    let proj = gw.dir.join("proj");
    std::fs::create_dir_all(&proj).unwrap();
    std::fs::write(proj.join("notes.md"), "helix notes\nline two\n").unwrap();
    let out = gw.npu(&["ask", "-m", "fast", "--read", &proj.display().to_string(), "read notes please"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("helix notes"), "{stdout}\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stderr).contains("read_file notes.md"));

    // models: aliases and configured files
    let out = gw.npu(&["models"]);
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("fast       → qwen3.5:9b") && s.contains("downloads on first use"), "{s}");

    // 5: the bar says GPU; everything unloads after the idle timeout
    gw.reply("fast");
    let bar = gw.bar_settled();
    assert!(bar["text"].as_str().unwrap().contains("qwen3.5 9b"));
    assert!(bar["tooltip"].as_str().unwrap().starts_with("GPU · "));
    assert!(!bar["tooltip"].as_str().unwrap().contains("pmode"));
    let st: Value = serde_json::from_str(&std::fs::read_to_string(gw.dir.join("status.json")).unwrap()).unwrap();
    assert_eq!(st["runs_on"], "GPU");
    assert!(gw.wait_state("off", 8));
    assert_eq!(gw.status()["procs"], json!([]));
}
