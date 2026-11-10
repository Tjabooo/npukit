//! Read-only file tools for the chat model: list, find, search and read files, only inside the folders
//! granted in [tools] read (or with --read). Nothing here writes, deletes or runs anything (except rg).

use crate::config::{expand, Config};
use crate::util::{fnmatch, realpath, splitlines, take_chars, which};
use regex::RegexBuilder;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

const MAX_RESULT: usize = 24000; // characters per tool result (~6k tokens)
const READ_LINES: i64 = 400; // default lines per read_file call
const LIST_MAX: usize = 300;
const MATCH_MAX: usize = 100;

pub fn schemas() -> Value {
    json!([
        {"type": "function", "function": {
            "name": "list_dir",
            "description": "List a folder: subfolders end with '/', files show their size. Use depth 2-3 to see a project's layout in one call. Paths may be absolute, start with ~, or be relative.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "folder to list"},
                "depth": {"type": "integer", "description": "levels to descend, 1-3 (default 1)"}},
                "required": ["path"]}}},
        {"type": "function", "function": {
            "name": "read_file",
            "description": "Read a text file with line numbers. Long files come in pages: pass offset (first line, 1-based) and limit (lines) to read further.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer", "description": "first line to read (default 1)"},
                "limit": {"type": "integer", "description": format!("number of lines (default {READ_LINES})")}},
                "required": ["path"]}}},
        {"type": "function", "function": {
            "name": "search_text",
            "description": "Search file contents for a regular expression (case-insensitive unless it has capitals). Returns path:line: text for each match.",
            "parameters": {"type": "object", "properties": {
                "pattern": {"type": "string", "description": "regular expression"},
                "path": {"type": "string", "description": "folder or file to search (default: the first granted folder)"},
                "glob": {"type": "string", "description": "only files matching this, e.g. *.lua"}},
                "required": ["pattern"]}}},
        {"type": "function", "function": {
            "name": "find_files",
            "description": "Find files whose path matches a glob, e.g. *.md or **/server/*.lua.",
            "parameters": {"type": "object", "properties": {
                "glob": {"type": "string"},
                "path": {"type": "string", "description": "folder to look in (default: the first granted folder)"}},
                "required": ["glob"]}}},
    ])
}

/// The granted folders and the rules for what inside them may be read.
pub struct Files {
    pub roots: Vec<PathBuf>,
    deny: Vec<String>,
    skip_dirs: HashSet<String>,
    home: PathBuf,
}

type R = Result<String, String>;

impl Files {
    pub fn new(roots: &[String], cfg: &Config) -> Files {
        let mut out: Vec<PathBuf> = vec![];
        for r in roots {
            let p = realpath(&expand(r));
            if p.is_dir() && !out.contains(&p) {
                out.push(p);
            }
        }
        let mut skip: HashSet<String> = cfg.search.exclude_dirs.iter().cloned().collect();
        skip.extend(cfg.tools.skip_dirs.iter().cloned());
        Files { roots: out, deny: cfg.tools.deny.clone(), skip_dirs: skip, home: realpath(&crate::config::home()) }
    }

    pub fn show(&self, p: &Path) -> String {
        if p == self.home {
            return "~".into();
        }
        match p.strip_prefix(&self.home) {
            Ok(rel) => format!("~/{}", rel.display()),
            Err(_) => p.display().to_string(),
        }
    }

    fn inside(&self, real: &Path) -> bool {
        self.roots.iter().any(|r| real.starts_with(r))
    }

    fn resolve_path(&self, path: Option<&str>) -> Result<PathBuf, String> {
        let Some(first) = self.roots.first() else {
            return Err("no folders are granted: add them to [tools] read in ~/.config/npukit/config.toml".into());
        };
        let raw = path.map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).unwrap_or_else(|| first.display().to_string());
        let mut p = expand(&raw);
        if !p.is_absolute() {
            p = first.join(p);
        }
        let real = realpath(&p);
        if !self.inside(&real) {
            let roots: Vec<String> = self.roots.iter().map(|r| self.show(r)).collect();
            return Err(format!("{raw} is outside the folders you may read ({})", roots.join(", ")));
        }
        if self.denied(&real) {
            return Err(format!("{} is on the deny list ([tools] deny)", self.show(&real)));
        }
        Ok(real)
    }

    pub fn denied(&self, p: &Path) -> bool {
        let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        for g in &self.deny {
            if let Some(dir) = g.strip_suffix('/') {
                // a folder name anywhere in the path
                if p.components().any(|c| c.as_os_str() == dir) {
                    return true;
                }
            } else if fnmatch(&name, g) {
                return true;
            }
        }
        false
    }

    /// Files under top (skip_dirs, denied entries and symlinks leading outside left out), sorted.
    fn walk(&self, top: &Path, out: &mut Vec<PathBuf>, limit: usize) {
        let Ok(rd) = std::fs::read_dir(top) else { return };
        let mut entries: Vec<std::fs::DirEntry> = rd.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        let (mut dirs, mut files) = (vec![], vec![]);
        for e in entries {
            let p = e.path();
            if self.denied(&p) {
                continue;
            }
            let Ok(ft) = e.file_type() else { continue };
            if ft.is_dir() {
                if !self.skip_dirs.contains(e.file_name().to_string_lossy().as_ref()) {
                    dirs.push(p);
                }
            } else if ft.is_file() || (ft.is_symlink() && p.is_file() && self.inside(&realpath(&p))) {
                files.push(p);
            }
        }
        for f in files {
            if out.len() > limit {
                return;
            }
            out.push(f);
        }
        for d in dirs {
            self.walk(&d, out, limit);
        }
    }

    pub fn list_dir(&self, path: Option<&str>, depth: i64) -> R {
        let top = self.resolve_path(path)?;
        if !top.is_dir() {
            return Err(format!("{} is not a folder", self.show(&top)));
        }
        let depth = depth.clamp(1, 3) as usize;
        let mut out = vec![format!("{}/", self.show(&top))];
        let mut n = 0usize;
        self.list_into(&top, 1, depth, &mut out, &mut n);
        if n >= LIST_MAX {
            out.push(format!("… stopped at {LIST_MAX} entries: list a subfolder for more"));
        }
        Ok(out.join("\n"))
    }

    fn list_into(&self, d: &Path, level: usize, depth: usize, out: &mut Vec<String>, n: &mut usize) {
        let pad = "  ".repeat(level);
        let rd = match std::fs::read_dir(d) {
            Ok(r) => r,
            Err(e) => {
                out.push(format!("{pad}(unreadable: {})", io_msg(&e)));
                return;
            }
        };
        let mut entries: Vec<(bool, String, PathBuf)> = rd.flatten().map(|e| {
            let p = e.path();
            (p.is_dir(), e.file_name().to_string_lossy().into_owned(), p)
        }).collect();
        entries.sort_by_key(|e| (!e.0, e.1.to_lowercase()));
        for (is_dir, name, p) in entries {
            if *n >= LIST_MAX || self.denied(&p) {
                continue;
            }
            *n += 1;
            if is_dir {
                let skipped = self.skip_dirs.contains(&name);
                out.push(format!("{pad}{name}/{}", if skipped && level < depth { "  (not descended)" } else { "" }));
                if level < depth && !skipped {
                    self.list_into(&p, level + 1, depth, out, n);
                }
            } else {
                let size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
                out.push(format!("{pad}{name}  {}", size_str(size)));
            }
        }
    }

    pub fn read_file(&self, path: Option<&str>, offset: i64, limit: i64) -> R {
        let p = self.resolve_path(path)?;
        if p.is_dir() {
            return Err(format!("{} is a folder: use list_dir", self.show(&p)));
        }
        let data = std::fs::read(&p).map_err(|e| io_msg(&e))?;
        if data[..data.len().min(8192)].contains(&0) {
            return Ok(format!("{} is a binary file ({}); not shown", self.show(&p), size_str(data.len() as u64)));
        }
        let text = String::from_utf8_lossy(&data);
        let lines = splitlines(&text);
        let start = offset.max(1) as usize;
        let count = limit.clamp(1, 2000) as usize;
        let mut body = vec![];
        let mut used = 0usize;
        for (i, ln) in lines.iter().enumerate().skip(start - 1).take(count) {
            let row = format!("{:>5}  {}", i + 1, take_chars(ln, 500));
            let len = row.chars().count();
            if used + len > MAX_RESULT {
                break;
            }
            body.push(row);
            used += len + 1;
        }
        let last = start + body.len() - 1;
        let more = if last < lines.len() {
            format!("\n… lines {}-{} not shown: read_file with offset={}", last + 1, lines.len(), last + 1)
        } else {
            String::new()
        };
        Ok(format!("{} · lines {start}-{last} of {}\n{}{more}", self.show(&p), lines.len(), body.join("\n")))
    }

    pub fn search_text(&self, pattern: &str, path: Option<&str>, glob: &str) -> R {
        self.search_impl(pattern, path, glob, which("rg").is_some())
    }

    fn search_impl(&self, pattern: &str, path: Option<&str>, glob: &str, use_rg: bool) -> R {
        let top = self.resolve_path(path)?;
        let insensitive = !pattern.chars().any(|c| c.is_uppercase());
        let rx = RegexBuilder::new(pattern).case_insensitive(insensitive).build()
            .map_err(|e| format!("bad regular expression: {e}"))?;
        let mut hits = if use_rg { self.rg(pattern, &top, glob) } else { None };
        if hits.is_none() {
            let mut found = vec![];
            let files = if top.is_file() { vec![top.clone()] } else {
                let mut v = vec![];
                self.walk(&top, &mut v, usize::MAX);
                v
            };
            'files: for fp in files {
                if !glob.is_empty() && !(fnmatch(&fp.file_name().unwrap_or_default().to_string_lossy(), glob)
                    || fnmatch(&fp.to_string_lossy(), glob)) {
                    continue;
                }
                if std::fs::metadata(&fp).map(|m| m.len() > 2_000_000).unwrap_or(true) {
                    continue;
                }
                let Ok(data) = std::fs::read(&fp) else { continue };
                let text = String::from_utf8_lossy(&data);
                for (i, ln) in text.split('\n').enumerate() {
                    if rx.is_match(ln) {
                        found.push((fp.clone(), i + 1, ln.trim_end_matches('\r').to_string()));
                        if found.len() > MATCH_MAX {
                            break 'files;
                        }
                    }
                }
            }
            hits = Some(found);
        }
        let hits: Vec<_> = hits.unwrap().into_iter().filter(|h| !self.denied(&h.0)).collect();
        if hits.is_empty() {
            return Ok(format!("no matches for /{pattern}/ in {}", self.show(&top)));
        }
        let mut rows: Vec<String> = hits.iter().take(MATCH_MAX)
            .map(|(fp, i, t)| format!("{}:{i}: {}", self.show(fp), take_chars(t.trim(), 300))).collect();
        if hits.len() > MATCH_MAX {
            rows.push(format!("… more than {MATCH_MAX} matches: narrow the pattern, path or glob"));
        }
        Ok(cap(rows.join("\n")))
    }

    fn rg(&self, pattern: &str, top: &Path, glob: &str) -> Option<Vec<(PathBuf, usize, String)>> {
        let mut args: Vec<String> = ["--json", "-S", "-m", "20", "--max-filesize", "2M"].map(String::from).to_vec();
        let mut skip: Vec<&String> = self.skip_dirs.iter().collect();
        skip.sort();
        for d in skip {
            args.extend(["-g".into(), format!("!{d}/")]);
        }
        if !glob.is_empty() {
            args.extend(["-g".into(), glob.into()]);
        }
        args.extend(["-e".into(), pattern.into(), top.display().to_string()]);
        let out = std::process::Command::new("rg").args(&args).stdin(std::process::Stdio::null()).output().ok()?;
        if !matches!(out.status.code(), Some(0) | Some(1)) {
            return None;
        }
        let mut hits = vec![];
        for ln in String::from_utf8_lossy(&out.stdout).lines() {
            let Ok(ev) = serde_json::from_str::<Value>(ln) else { continue };
            if ev.get("type").and_then(Value::as_str) != Some("match") {
                continue;
            }
            let d = &ev["data"];
            let text = d.pointer("/lines/text").and_then(Value::as_str).unwrap_or("");
            let p = d.pointer("/path/text").and_then(Value::as_str).unwrap_or("");
            hits.push((PathBuf::from(p), d["line_number"].as_u64().unwrap_or(0) as usize, text.trim_end_matches('\n').to_string()));
            if hits.len() > MATCH_MAX {
                break;
            }
        }
        Some(hits)
    }

    pub fn find_files(&self, glob: &str, path: Option<&str>) -> R {
        let top = self.resolve_path(path)?;
        let pat = if glob.trim().is_empty() { "*" } else { glob.trim() };
        let mut files = vec![];
        self.walk(&top, &mut files, usize::MAX);
        let mut found = vec![];
        for fp in files {
            let rel = fp.strip_prefix(&top).unwrap_or(&fp).display().to_string();
            let name = fp.file_name().unwrap_or_default().to_string_lossy().into_owned();
            if fnmatch(&rel, pat) || fnmatch(&name, pat) || pat.strip_prefix("**/").is_some_and(|p| fnmatch(&rel, p)) {
                found.push(rel);
                if found.len() > LIST_MAX {
                    break;
                }
            }
        }
        if found.is_empty() {
            return Ok(format!("no files matching {pat} in {}", self.show(&top)));
        }
        let more = if found.len() > LIST_MAX { format!("\n… more than {LIST_MAX}: narrow the glob") } else { String::new() };
        found.truncate(LIST_MAX);
        Ok(cap(format!("{}/\n{}{more}", self.show(&top), found.join("\n"))))
    }

    pub fn run(&self, name: &str, args: &Value) -> String {
        let s = |k: &str| args.get(k).and_then(Value::as_str);
        let i = |k: &str, d: i64| args.get(k).and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64))
            .or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))).filter(|x| *x != 0).unwrap_or(d);
        let r = match name {
            "list_dir" => self.list_dir(s("path"), i("depth", 1)),
            "read_file" => self.read_file(s("path"), i("offset", 1), i("limit", READ_LINES)),
            "search_text" => match s("pattern") {
                Some(p) => self.search_text(p, s("path"), s("glob").unwrap_or("")),
                None => Err("bad arguments for search_text: pattern is required".into()),
            },
            "find_files" => match s("glob") {
                Some(g) => self.find_files(g, s("path")),
                None => Err("bad arguments for find_files: glob is required".into()),
            },
            _ => return format!("error: no tool called {name}"),
        };
        match r {
            Ok(t) => cap(t),
            Err(e) => format!("error: {e}"),
        }
    }
}

fn io_msg(e: &std::io::Error) -> String {
    let s = e.to_string();
    s.split(" (os error").next().unwrap_or(&s).to_string()
}

fn size_str(n: u64) -> String {
    let mut x = n as f64;
    for unit in ["B", "KB", "MB", "GB"] {
        if x < 1024.0 || unit == "GB" {
            return if unit == "B" { format!("{x:.0} {unit}") } else { format!("{x:.1} {unit}") };
        }
        x /= 1024.0;
    }
    format!("{n} B")
}

fn cap(text: String) -> String {
    if text.chars().count() <= MAX_RESULT {
        text
    } else {
        format!("{}\n… cut here (result too long)", take_chars(&text, MAX_RESULT))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Tmp(PathBuf);
    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn setup() -> (Tmp, Files) {
        let t = std::env::temp_dir().join(format!("npukit-tools-{}-{:?}", std::process::id(), std::thread::current().id()));
        let _ = std::fs::remove_dir_all(&t);
        let root = t.join("proj");
        std::fs::create_dir_all(root.join("src/deep")).unwrap();
        std::fs::create_dir_all(root.join("node_modules/x")).unwrap();
        std::fs::write(root.join("README.md"), "# Helix\nA game about hollows.\n").unwrap();
        std::fs::write(root.join("src/main.lua"), "local hollow = 1\nprint(hollow)\n").unwrap();
        std::fs::write(root.join("src/deep/a.txt"), "x\n".repeat(1000)).unwrap();
        std::fs::write(root.join("node_modules/x/hollow.js"), "hollow\n").unwrap();
        std::fs::write(root.join(".env"), "TOKEN=hollow-secret\n").unwrap();
        std::fs::write(root.join("blob.bin"), b"\0\x01\x02hollow").unwrap();
        std::fs::write(t.join("outside.txt"), "not yours hollow\n").unwrap();
        std::os::unix::fs::symlink(t.join("outside.txt"), root.join("escape")).unwrap();
        let cfg = crate::config::from_str("").unwrap();
        let f = Files::new(&[root.display().to_string()], &cfg);
        (Tmp(t), f)
    }

    #[test]
    fn outside_symlink_and_deny_list_are_refused() {
        let (t, f) = setup();
        let outside = t.0.join("outside.txt").display().to_string();
        assert!(f.run("read_file", &json!({"path": outside})).contains("outside the folders"));
        assert!(f.run("read_file", &json!({"path": "../outside.txt"})).contains("outside the folders"));
        assert!(f.run("read_file", &json!({"path": "escape"})).contains("outside the folders"));
        assert!(f.run("read_file", &json!({"path": ".env"})).contains("deny list"));
        assert!(f.run("read_file", &json!({"path": "blob.bin"})).contains("binary"));
        let cfg = crate::config::from_str("").unwrap();
        assert!(Files::new(&[], &cfg).run("list_dir", &json!({"path": "~"})).contains("no folders are granted"));
    }

    #[test]
    fn list_read_search_find() {
        let (_t, f) = setup();
        let listing = f.run("list_dir", &json!({"path": "", "depth": 2}));
        assert!(listing.contains("src/"), "{listing}");
        assert!(listing.contains("main.lua"));
        assert!(listing.contains("node_modules/  (not descended)"));
        assert!(!listing.contains(".env"));
        let page = f.run("read_file", &json!({"path": "src/deep/a.txt", "offset": 10, "limit": 5}));
        assert!(page.contains("lines 10-14 of 1000"), "{page}");
        assert!(page.contains("offset=15"));
        for rg in [true, false] {
            let hits = if rg { f.run("search_text", &json!({"pattern": "hollow"})) } else {
                f.search_impl("hollow", None, "", false).unwrap()
            };
            assert!(hits.contains("README.md:2:"), "{hits}");
            assert!(hits.contains("main.lua:1:"));
            for leak in ["hollow-secret", "node_modules", "outside.txt", "not yours"] {
                assert!(!hits.contains(leak), "{leak} in {hits}");
            }
        }
        assert!(f.run("find_files", &json!({"glob": "*.lua"})).contains("src/main.lua"));
        assert!(f.run("delete_file", &json!({"path": "README.md"})).contains("no tool"));
    }
}
