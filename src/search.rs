//! Semantic search over notes and code with EmbeddingGemma embeddings (FastFlowLM on the NPU, or llama.cpp).
//!
//! `npu index` walks the configured roots (git repos: tracked + untracked-but-not-ignored files; other dirs:
//! a plain walk with excluded dirs), re-embeds only files whose size or mtime changed, and keeps everything
//! in one SQLite file. `npu search` embeds the query and ranks chunks by cosine similarity: only the vectors
//! are read for ranking, the text of the few hits afterwards.

use crate::client::{self, ApiError};
use crate::config::{self, expand, Config};
use crate::util::{fnmatch, splitlines, take_chars, tilde};
use rusqlite::{params, Connection};
use std::collections::{HashMap, HashSet};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, size INTEGER, mtime REAL);
CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, path TEXT, start INTEGER, end INTEGER,
                                   text TEXT, vec BLOB);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
";

pub fn db(cfg: &Config) -> rusqlite::Result<Connection> {
    let p = expand(&cfg.search.db);
    if let Some(d) = p.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let con = Connection::open(p)?;
    con.execute_batch(SCHEMA)?;
    Ok(con)
}

fn git_files(repo: &Path) -> Option<Vec<PathBuf>> {
    let out = Command::new("git").arg("-C").arg(repo)
        .args(["ls-files", "-z", "--cached", "--others", "--exclude-standard"])
        .stdin(Stdio::null()).stderr(Stdio::null()).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).split('\0').filter(|f| !f.is_empty()).map(|f| repo.join(f)).collect())
}

/// mtime as Python's st_mtime float, so an index written by either version stays valid.
fn mtime(m: &std::fs::Metadata) -> f64 {
    m.mtime() as f64 + m.mtime_nsec() as f64 * 1e-9
}

pub fn candidates(cfg: &Config) -> Vec<PathBuf> {
    let s = &cfg.search;
    let excl: HashSet<&str> = s.exclude_dirs.iter().map(String::as_str).collect();
    let exts: HashSet<String> = s.extensions.iter().map(|e| e.to_lowercase()).collect();
    let names: HashSet<&str> = s.names.iter().map(String::as_str).collect();
    let max_bytes = s.max_file_kb * 1024;
    let globs: Vec<String> = s.exclude_globs.iter().map(|g| config::expand_str(g)).collect();

    // extensionless tools in bin dirs: index when they start with #!
    let is_script = |p: &Path| {
        use std::io::Read;
        let mut b = [0u8; 2];
        std::fs::File::open(p).and_then(|mut f| f.read_exact(&mut b)).is_ok() && &b == b"#!"
    };
    let ok = |p: &Path| -> bool {
        let name = p.file_name().map(|n| n.to_string_lossy()).unwrap_or_default();
        let ext = suffix(&name).to_lowercase();
        if !exts.contains(&ext) && !names.contains(name.as_ref()) && !(ext.is_empty() && is_script(p)) {
            return false;
        }
        if p.components().any(|c| excl.contains(c.as_os_str().to_string_lossy().as_ref())) {
            return false;
        }
        let ps = p.to_string_lossy();
        if globs.iter().any(|g| fnmatch(&ps, g)) {
            return false;
        }
        match std::fs::metadata(p) {
            Ok(m) => m.is_file() && m.len() > 0 && m.len() <= max_bytes,
            Err(_) => false,
        }
    };
    let mut out: Vec<PathBuf> = vec![];
    fn walk(d: &Path, out: &mut Vec<PathBuf>, excl: &HashSet<&str>, ok: &dyn Fn(&Path) -> bool) {
        let Ok(rd) = std::fs::read_dir(d) else { return };
        let entries: Vec<std::fs::DirEntry> = rd.flatten().collect();
        if entries.iter().any(|e| e.file_name() == ".git") {
            if let Some(files) = git_files(d) {
                out.extend(files.into_iter().filter(|p| ok(p)));
                return;
            }
        }
        for e in entries {
            let Ok(ft) = e.file_type() else { continue };
            let name = e.file_name();
            let name = name.to_string_lossy();
            if ft.is_dir() {
                if !excl.contains(name.as_ref()) && !name.starts_with('.') {
                    walk(&e.path(), out, excl, ok);
                }
            } else if ft.is_file() && ok(&e.path()) {
                out.push(e.path());
            }
        }
    }
    for root in &s.roots {
        let r = expand(root);
        if r.is_dir() {
            walk(&r, &mut out, &excl, &ok);
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Path.suffix of Python: ".md" for "a.md", "" for ".bashrc" or "Makefile".
pub fn suffix(name: &str) -> &str {
    match name.rfind('.') {
        Some(i) if i > 0 && i < name.len() - 1 => &name[i..],
        _ => "",
    }
}

/// (first line, last line, text) chunks of `n` lines overlapping by `overlap`.
pub fn chunk(text: &str, n: usize, overlap: usize) -> Vec<(usize, usize, String)> {
    let lines = splitlines(text);
    let (mut out, mut i) = (vec![], 0usize);
    let n = n.max(1);
    while i < lines.len() {
        let j = lines.len().min(i + n);
        let body = lines[i..j].join("\n");
        let body = body.trim();
        // long lines: cut by characters (the model sees ~2k tokens max)
        let body = take_chars(body, 2400);
        if !body.is_empty() {
            out.push((i + 1, j, body.to_string()));
        }
        if j == lines.len() {
            break;
        }
        i = if j > overlap && j - overlap > i { j - overlap } else { j };
    }
    out
}

fn pack(v: &[f32]) -> Vec<u8> {
    let n = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    let n = if n == 0.0 { 1.0 } else { n };
    v.iter().flat_map(|x| ((*x as f64 / n) as f32).to_ne_bytes()).collect()
}

pub struct IndexArgs {
    pub full: bool,
    pub quiet: bool,
}

pub fn index(a: &IndexArgs) -> Result<i32, String> {
    let cfg = config::load();
    index_with(cfg, a)
}

pub fn index_with(cfg: &Config, a: &IndexArgs) -> Result<i32, String> {
    let s = &cfg.search;
    let mut con = db(cfg).map_err(|e| e.to_string())?;
    let t0 = Instant::now();
    let files = candidates(cfg);
    let known: HashMap<String, (i64, f64)> = {
        let mut q = con.prepare("SELECT path, size, mtime FROM files").map_err(|e| e.to_string())?;
        let rows = q.query_map([], |r| Ok((r.get::<_, String>(0)?, (r.get::<_, i64>(1)?, r.get::<_, f64>(2)?))))
            .map_err(|e| e.to_string())?;
        rows.flatten().collect()
    };
    let mut want = HashSet::new();
    let mut todo: Vec<(PathBuf, i64, f64)> = vec![];
    for p in &files {
        let Ok(m) = std::fs::metadata(p) else { continue };
        let key = p.to_string_lossy().into_owned();
        let (size, mt) = (m.len() as i64, mtime(&m));
        let same = known.get(&key).is_some_and(|(ks, km)| *ks == size && (km - mt).abs() < 1e-6);
        want.insert(key);
        if a.full || !same {
            todo.push((p.clone(), size, mt));
        }
    }
    let gone: Vec<&String> = known.keys().filter(|k| !want.contains(*k)).collect();
    {
        let tx = con.transaction().map_err(|e| e.to_string())?;
        for k in &gone {
            tx.execute("DELETE FROM chunks WHERE path=?", [k]).map_err(|e| e.to_string())?;
            tx.execute("DELETE FROM files WHERE path=?", [k]).map_err(|e| e.to_string())?;
        }
        tx.commit().map_err(|e| e.to_string())?;
    }
    if todo.is_empty() {
        println!("index up to date: {} files, {} removed ({:.1}s)", files.len(), gone.len(), t0.elapsed().as_secs_f64());
        return Ok(0);
    }
    if !a.quiet {
        println!("{} changed files of {}", todo.len(), files.len());
    }
    let batch = s.batch.max(1);
    let home = config::home().to_string_lossy().into_owned();
    let mut pending: Vec<(String, usize, usize, String)> = vec![];
    let mut done_files: Vec<(String, i64, f64)> = vec![]; // all chunks queued; recorded after the next flush
    let mut n_chunks = 0usize;

    // batches span files: FastFlowLM has ~1.5 s fixed cost per request, so full batches matter
    let flush = |con: &mut Connection, pending: &mut Vec<(String, usize, usize, String)>,
                 done: &mut Vec<(String, i64, f64)>, n_chunks: &mut usize| -> Result<(), String> {
        let mut rows: Vec<(usize, Vec<f32>)> = vec![];
        if !pending.is_empty() {
            let texts: Vec<String> = pending.iter()
                .map(|(p, _, _, body)| format!("title: {} | text: {body}", if home.len() > 1 { p.replace(&home, "~") } else { p.clone() }))
                .collect();
            match client::embed(&texts, 600.0) {
                Ok(vs) => rows = vs.into_iter().enumerate().collect(),
                Err(e) if e.0.contains("not reachable") => return Err(e.0),
                Err(_) => {
                    // one bad input fails the batch: retry one by one, skip the bad
                    for (i, t) in texts.iter().enumerate() {
                        match client::embed(std::slice::from_ref(t), 600.0) {
                            Ok(mut v) => rows.push((i, v.remove(0))),
                            Err(e1) => eprintln!("\nskipped {}:{}: {e1}", pending[i].0, pending[i].1),
                        }
                    }
                }
            }
        }
        let tx = con.transaction().map_err(|e| e.to_string())?;
        {
            let mut ins = tx.prepare_cached("INSERT INTO chunks(path, start, end, text, vec) VALUES (?,?,?,?,?)").map_err(|e| e.to_string())?;
            for (i, v) in &rows {
                let (p, a, b, body) = &pending[*i];
                ins.execute(params![p, *a as i64, *b as i64, body, pack(v)]).map_err(|e| e.to_string())?;
            }
            let mut f = tx.prepare_cached("INSERT OR REPLACE INTO files(path, size, mtime) VALUES (?,?,?)").map_err(|e| e.to_string())?;
            for (k, size, mt) in done.iter() {
                f.execute(params![k, size, mt]).map_err(|e| e.to_string())?;
            }
        }
        tx.commit().map_err(|e| e.to_string())?;
        *n_chunks += rows.len();
        pending.clear();
        done.clear();
        Ok(())
    };

    client::activity(Some("index"));
    let res = (|| -> Result<(), String> {
        for (i, (p, size, mt)) in todo.iter().enumerate() {
            let Ok(raw) = std::fs::read(p) else { continue };
            let text = if raw[..raw.len().min(4096)].contains(&0) { String::new() } else { String::from_utf8_lossy(&raw).into_owned() };
            let key = p.to_string_lossy().into_owned();
            con.execute("DELETE FROM chunks WHERE path=?", [&key]).map_err(|e| e.to_string())?;
            for (a, b, body) in chunk(&text, s.chunk_lines, s.overlap_lines).into_iter().take(s.max_chunks_per_file) {
                pending.push((key.clone(), a, b, body));
                if pending.len() >= batch {
                    flush(&mut con, &mut pending, &mut done_files, &mut n_chunks)?;
                }
            }
            done_files.push((key, *size, *mt));
            if !a.quiet && i % 20 == 0 {
                eprint!("\r{}/{} files, {} chunks", i + 1, todo.len(), n_chunks);
            }
        }
        flush(&mut con, &mut pending, &mut done_files, &mut n_chunks)
    })();
    client::activity(None);
    let _ = client::call("/npu/release", Some(&serde_json::json!({"component": "embed"})), 5.0);
    res?;
    let dt = t0.elapsed().as_secs_f64();
    println!("\nindexed {} files → {} chunks in {dt:.1}s ({:.1} chunks/s), {} removed",
             todo.len(), n_chunks, n_chunks as f64 / dt, gone.len());
    Ok(0)
}

pub struct Hit {
    pub score: f32,
    pub path: String,
    pub start: i64,
    pub end: i64,
    pub text: String,
}

pub fn query(q: &str, k: usize, cfg: &Config) -> Result<Vec<Hit>, String> {
    let con = db(cfg).map_err(|e| e.to_string())?;
    let qv = client::embed(&[format!("task: search result | query: {q}")], 600.0).map_err(|e: ApiError| e.0)?;
    let qv: Vec<f32> = pack(&qv[0]).chunks_exact(4).map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]])).collect();
    // rank on the vectors alone; read the text of the winners afterwards
    let mut scored: Vec<(f32, i64, String)> = vec![];
    {
        let mut st = con.prepare("SELECT id, path, vec FROM chunks").map_err(|e| e.to_string())?;
        let mut rows = st.query([]).map_err(|e| e.to_string())?;
        while let Some(r) = rows.next().map_err(|e| e.to_string())? {
            let blob = r.get_ref(2).map_err(|e| e.to_string())?.as_blob().unwrap_or(&[]);
            let score = dot(blob, &qv);
            scored.push((score, r.get(0).map_err(|e| e.to_string())?, r.get(1).map_err(|e| e.to_string())?));
        }
    }
    if scored.is_empty() {
        return Ok(vec![]);
    }
    let top = (k * 4).min(scored.len());
    scored.select_nth_unstable_by(top - 1, |a, b| b.0.total_cmp(&a.0));
    scored.truncate(top);
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut out = vec![];
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut get = con.prepare("SELECT start, end, text FROM chunks WHERE id=?").map_err(|e| e.to_string())?;
    for (score, id, path) in scored {
        // at most 2 hits per file
        let n = seen.entry(path.clone()).or_insert(0);
        if *n >= 2 {
            continue;
        }
        *n += 1;
        let (start, end, text) = get.query_row([id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?))).map_err(|e| e.to_string())?;
        out.push(Hit { score, path, start, end, text });
        if out.len() >= k {
            break;
        }
    }
    Ok(out)
}

fn dot(blob: &[u8], q: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let n = (blob.len() / 4).min(q.len());
    let (body, rest) = (n / 8 * 8, n % 8);
    for i in (0..body).step_by(8) {
        for j in 0..8 {
            let o = (i + j) * 4;
            acc[j] += f32::from_ne_bytes([blob[o], blob[o + 1], blob[o + 2], blob[o + 3]]) * q[i + j];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (i, qi) in q.iter().enumerate().skip(body).take(rest) {
        let o = i * 4;
        s += f32::from_ne_bytes([blob[o], blob[o + 1], blob[o + 2], blob[o + 3]]) * qi;
    }
    s
}

fn open(cfg: &Config, path: &str, line: i64) {
    let cmd: Vec<String> = cfg.search.open.iter().map(|a| a.replace("{path}", path).replace("{line}", &line.to_string())).collect();
    if cmd.is_empty() {
        return;
    }
    use std::os::unix::process::CommandExt;
    let _ = Command::new(&cmd[0]).args(&cmd[1..]).process_group(0)
        .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn();
}

pub struct SearchArgs {
    pub query: Vec<String>,
    pub n: Option<usize>,
    pub rofi: bool,
    pub theme: Option<String>,
}

pub fn search(a: &SearchArgs) -> i32 {
    let cfg = config::load();
    let k = a.n.unwrap_or(cfg.search.results);
    let mut q = a.query.join(" ").trim().to_string();
    let theme = a.theme.clone().or_else(|| (!cfg.ui.search_theme.is_empty()).then(|| config::expand_str(&cfg.ui.search_theme)));
    if a.rofi && q.is_empty() {
        let mesg = format!("semantic search · notes + code · runs on the {}", cfg.engine_label());
        let (code, typed) = crate::ui::rofi(&[], "search", theme.as_deref(), Some(&mesg), true);
        if code != 0 || typed.is_empty() {
            return 1;
        }
        q = typed;
    }
    if q.is_empty() {
        eprintln!("usage: npu search QUERY");
        return 2;
    }
    let t0 = Instant::now();
    let hits = match query(&q, k, cfg) {
        Ok(h) => h,
        Err(e) => {
            if a.rofi {
                crate::ui::notify("Search failed", &e, "critical", None);
            } else {
                eprintln!("{e}");
            }
            return 1;
        }
    };
    let dt = t0.elapsed().as_secs_f64();
    if a.rofi {
        let lines: Vec<String> = hits.iter().map(|h| {
            let snippet = h.text.split_whitespace().collect::<Vec<_>>().join(" ");
            let snippet = take_chars(&snippet, 90).replace('&', "&amp;").replace('<', "&lt;");
            format!("{}:{}  <span alpha='55%'>{:.2}  {snippet}</span>", tilde(&h.path), h.start, h.score)
        }).collect();
        let mut args: Vec<String> = ["rofi", "-dmenu", "-i", "-markup-rows", "-p"].map(String::from).to_vec();
        args.push(take_chars(&q, 30).to_string());
        args.extend(["-format".into(), "i".into(), "-mesg".into(), format!("{} results · {dt:.2}s", hits.len())]);
        if let Some(t) = &theme {
            args.extend(["-theme".into(), t.clone()]);
        }
        let (code, out) = crate::ui::rofi_args(&args, &lines);
        if code == 0 {
            if let Some(h) = out.parse::<usize>().ok().and_then(|i| hits.get(i)) {
                open(cfg, &h.path, h.start);
            }
        }
        return 0;
    }
    for h in &hits {
        println!("\x1b[1m{:.3}  {}:{}-{}\x1b[0m", h.score, tilde(&h.path), h.start, h.end);
        let snippet = splitlines(&h.text).into_iter().take(4).collect::<Vec<_>>().join("\n");
        println!("   {}", take_chars(&snippet.replace('\n', "\n   "), 400));
    }
    eprintln!("[{} hits in {dt:.2}s]", hits.len());
    0
}

pub struct Stats {
    pub files: i64,
    pub chunks: i64,
    pub db_mb: f64,
    pub updated: f64,
}

pub fn stats(cfg: &Config) -> Option<Stats> {
    let p = expand(&cfg.search.db);
    let m = std::fs::metadata(&p).ok()?;
    let con = Connection::open_with_flags(&p, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).ok()?;
    let count = |t: &str| con.query_row(&format!("SELECT count(*) FROM {t}"), [], |r| r.get::<_, i64>(0)).unwrap_or(0);
    Some(Stats { files: count("files"), chunks: count("chunks"), db_mb: (m.len() as f64 / 1e5).round() / 10.0, updated: mtime(&m) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_and_line_numbers() {
        let text: String = (1..=100).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n");
        let parts = chunk(&text, 40, 5);
        assert_eq!((parts[0].0, parts[0].1), (1, 40));
        assert_eq!((parts[1].0, parts[1].1), (36, 75));
        assert_eq!(parts.last().unwrap().1, 100);
        assert!(parts[1].2.starts_with("line 36"));
        assert!(chunk("", 40, 5).is_empty());
    }

    #[test]
    fn suffix_like_pathlib() {
        assert_eq!(suffix("a.md"), ".md");
        assert_eq!(suffix(".bashrc"), "");
        assert_eq!(suffix("Makefile"), "");
        assert_eq!(suffix("x.tar.gz"), ".gz");
    }

    #[test]
    fn dot_and_pack() {
        let v = pack(&[3.0, 4.0]);
        let q: Vec<f32> = v.chunks_exact(4).map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]])).collect();
        assert!((q[0] - 0.6).abs() < 1e-6 && (q[1] - 0.8).abs() < 1e-6);
        let long: Vec<f32> = (0..19).map(|i| i as f32).collect();
        let blob: Vec<u8> = long.iter().flat_map(|x| x.to_ne_bytes()).collect();
        let ones = vec![1.0f32; 19];
        assert_eq!(dot(&blob, &ones), (0..19).sum::<i32>() as f32);
    }
}
