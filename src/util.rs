//! Small helpers shared by the commands.

use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Shell-style wildcard match like Python's fnmatch.fnmatch on POSIX: `*` and `?` match any
/// character (slashes included), `[abc]`, `[a-z]` and `[!abc]` match sets. Case-sensitive.
pub fn fnmatch(name: &str, pat: &str) -> bool {
    let n: Vec<char> = name.chars().collect();
    let p: Vec<char> = pat.chars().collect();
    let (mut ni, mut pi) = (0usize, 0usize);
    let (mut star_p, mut star_n) = (usize::MAX, 0usize);
    while ni < n.len() {
        if pi < p.len() {
            match p[pi] {
                '*' => {
                    star_p = pi;
                    star_n = ni;
                    pi += 1;
                    continue;
                }
                '?' => {
                    ni += 1;
                    pi += 1;
                    continue;
                }
                '[' => {
                    if let Some((matched, next)) = class(&p, pi, n[ni]) {
                        if matched {
                            ni += 1;
                            pi = next;
                            continue;
                        }
                    } else if n[ni] == '[' {
                        // no closing ]: a literal [
                        ni += 1;
                        pi += 1;
                        continue;
                    }
                }
                c if c == n[ni] => {
                    ni += 1;
                    pi += 1;
                    continue;
                }
                _ => {}
            }
        }
        if star_p != usize::MAX {
            pi = star_p + 1;
            star_n += 1;
            ni = star_n;
            continue;
        }
        return false;
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

/// `[...]` at p[i]: (whether c is in the set, index after the class); None if it never closes.
fn class(p: &[char], i: usize, c: char) -> Option<(bool, usize)> {
    let mut j = i + 1;
    let neg = j < p.len() && p[j] == '!';
    if neg {
        j += 1;
    }
    let start = j;
    let mut hit = false;
    while j < p.len() && (p[j] != ']' || j == start) {
        if j + 2 < p.len() && p[j + 1] == '-' && p[j + 2] != ']' {
            if p[j] <= c && c <= p[j + 2] {
                hit = true;
            }
            j += 3;
        } else {
            if p[j] == c {
                hit = true;
            }
            j += 1;
        }
    }
    if j >= p.len() {
        return None;
    }
    Some((hit != neg, j + 1))
}

/// str.splitlines() of Python: \n, \r\n, \r and the other Unicode line boundaries; no trailing empty line.
pub fn splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut it = s.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        let brk = matches!(c, '\n' | '\r' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e' | '\u{85}' | '\u{2028}' | '\u{2029}');
        if brk {
            out.push(&s[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' {
                if let Some(&(_, '\n')) = it.peek() {
                    it.next();
                    end += 1;
                }
            }
            start = end;
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// The first n characters (not bytes).
pub fn take_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

pub fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

/// os.path.realpath without strict: symlinks resolved as far as the path exists, the rest appended.
pub fn realpath(p: &Path) -> PathBuf {
    let abs = if p.is_absolute() { p.to_path_buf() } else { std::env::current_dir().unwrap_or_default().join(p) };
    if let Ok(r) = abs.canonicalize() {
        return r;
    }
    // resolve the longest existing prefix, normalise the rest lexically
    let mut out = PathBuf::from("/");
    let comps: Vec<Component> = abs.components().collect();
    for (i, c) in comps.iter().enumerate() {
        match c {
            Component::RootDir | Component::Prefix(_) | Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(n) => {
                let next = out.join(n);
                match next.canonicalize() {
                    Ok(r) => out = r,
                    Err(_) => {
                        out = next;
                        for rest in &comps[i + 1..] {
                            match rest {
                                Component::ParentDir => {
                                    out.pop();
                                }
                                Component::Normal(n) => out.push(n),
                                _ => {}
                            }
                        }
                        return out;
                    }
                }
            }
        }
    }
    out
}

/// The full path of a command in $PATH (or the command itself when it contains a slash and exists).
pub fn which(cmd: &str) -> Option<PathBuf> {
    if cmd.is_empty() {
        return None;
    }
    let is_exec = |p: &Path| {
        use std::os::unix::fs::PermissionsExt;
        p.metadata().map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0).unwrap_or(false)
    };
    if cmd.contains('/') {
        let p = PathBuf::from(cmd);
        return is_exec(&p).then_some(p);
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(cmd)).find(|p| is_exec(p))
}

pub fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

extern "C" {
    fn tzset();
}

/// strftime in local time.
pub fn strftime(fmt: &str, t: f64) -> String {
    let secs = t as libc::time_t;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    let mut buf = [0u8; 128];
    let cfmt = std::ffi::CString::new(fmt).unwrap_or_default();
    let n = unsafe {
        tzset();
        libc::localtime_r(&secs, &mut tm);
        libc::strftime(buf.as_mut_ptr() as *mut libc::c_char, buf.len(), cfmt.as_ptr(), &tm)
    };
    String::from_utf8_lossy(&buf[..n]).into_owned()
}

pub fn hostname() -> String {
    let mut buf = [0u8; 256];
    let r = unsafe { libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len()) };
    if r != 0 {
        return "localhost".into();
    }
    let n = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..n]).into_owned()
}

/// Quote for /bin/sh (shlex.quote).
pub fn sh_quote(s: &str) -> String {
    if !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || "@%+=:,./-_".contains(c)) {
        return s.to_string();
    }
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

/// Path shown with ~ for the home directory (str.replace(home, "~") of the Python version).
pub fn tilde(path: &str) -> String {
    let home = crate::config::home().to_string_lossy().into_owned();
    if home.len() > 1 { path.replace(&home, "~") } else { path.to_string() }
}

/// Is a process alive (kill 0)?
pub fn pid_alive(pid: i32) -> bool {
    pid > 0 && unsafe { libc::kill(pid, 0) } == 0
}

/// Write a file atomically (tmp + rename).
pub fn write_atomic(path: &Path, data: &[u8]) -> std::io::Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnmatch_like_python() {
        assert!(fnmatch("a.min.js", "*.min.*"));
        assert!(fnmatch("/x/third_party/y.c", "*/third_party/*"));
        assert!(fnmatch("id_rsa.pub", "id_rsa*"));
        assert!(!fnmatch("main.lua", "*.md"));
        assert!(fnmatch("src/deep/a.txt", "*.txt"));
        assert!(fnmatch("a1", "a[0-9]"));
        assert!(!fnmatch("ab", "a[!b]"));
        assert!(fnmatch(".env.local", ".env.*"));
        assert!(!fnmatch(".env", ".env.*"));
        assert!(fnmatch("", "*"));
        assert!(fnmatch("[x", "[x"));
    }

    #[test]
    fn splitlines_like_python() {
        assert_eq!(splitlines("a\nb\r\nc\rd\n"), vec!["a", "b", "c", "d"]);
        assert_eq!(splitlines(""), Vec::<&str>::new());
        assert_eq!(splitlines("\n\nx"), vec!["", "", "x"]);
    }

    #[test]
    fn base64_matches() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn quoting() {
        assert_eq!(sh_quote("/a/b"), "/a/b");
        assert_eq!(sh_quote("it's"), "'it'\"'\"'s'");
    }
}
