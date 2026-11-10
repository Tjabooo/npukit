//! `npu ocr`: select a screen region, read it with a vision model (NPU or GPU), copy the text.
//!
//! With `--translate [LANG]` the text is then translated by a chat model on the same engine;
//! `--describe` / `--ask` put a description or an answer about the region on the clipboard.

use crate::client::{self, ApiError, ChatOpts};
use crate::config::{self, strs, Config};
use crate::ui;
use crate::util;
use regex::Regex;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::LazyLock;
use std::time::Instant;

static FENCE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)^```[\w+-]*\n(.*?)\n?```$").unwrap());

/// Languages in the --pick list: (code, English name, own name). The codes also work for --translate;
/// anything else is passed to the model as written.
pub const LANGUAGES: &[(&str, &str, &str)] = &[
    ("ar", "Arabic", "العربية"), ("bg", "Bulgarian", "български"), ("bn", "Bengali", "বাংলা"),
    ("ca", "Catalan", "català"), ("cs", "Czech", "čeština"), ("da", "Danish", "dansk"),
    ("de", "German", "Deutsch"), ("el", "Greek", "Ελληνικά"), ("en", "English", ""),
    ("es", "Spanish", "español"), ("et", "Estonian", "eesti"), ("fa", "Persian", "فارسی"),
    ("fi", "Finnish", "suomi"), ("fr", "French", "français"), ("he", "Hebrew", "עברית"),
    ("hi", "Hindi", "हिन्दी"), ("hr", "Croatian", "hrvatski"), ("hu", "Hungarian", "magyar"),
    ("id", "Indonesian", "Bahasa Indonesia"), ("is", "Icelandic", "íslenska"), ("it", "Italian", "italiano"),
    ("ja", "Japanese", "日本語"), ("ko", "Korean", "한국어"), ("lt", "Lithuanian", "lietuvių"),
    ("lv", "Latvian", "latviešu"), ("ms", "Malay", "Bahasa Melayu"), ("nl", "Dutch", "Nederlands"),
    ("no", "Norwegian", "norsk"), ("pl", "Polish", "polski"), ("pt", "Portuguese", "português"),
    ("ro", "Romanian", "română"), ("ru", "Russian", "русский"), ("sk", "Slovak", "slovenčina"),
    ("sl", "Slovenian", "slovenščina"), ("sr", "Serbian", "српски"), ("sv", "Swedish", "svenska"),
    ("sw", "Swahili", "Kiswahili"), ("th", "Thai", "ไทย"), ("tl", "Filipino", ""),
    ("tr", "Turkish", "Türkçe"), ("uk", "Ukrainian", "українська"), ("ur", "Urdu", "اردو"),
    ("vi", "Vietnamese", "Tiếng Việt"), ("zh", "Chinese", "中文"),
    ("zh-tw", "Chinese (Traditional)", "繁體中文"),
];
const RECENT_MAX: usize = 6;

pub fn capture(cmd: &[String]) -> Vec<u8> {
    if cmd.is_empty() {
        return vec![];
    }
    match Command::new(&cmd[0]).args(&cmd[1..]).stdin(Stdio::null()).stderr(Stdio::null()).output() {
        Ok(o) if o.status.success() && o.stdout.starts_with(b"\x89PNG\r\n\x1a\n") => o.stdout,
        _ => vec![],
    }
}

fn unfence(out: &Value) -> String {
    let text = client::content(out);
    match FENCE.captures(&text) {
        Some(m) => m[1].to_string(),
        None => text,
    }
}

fn exact() -> ChatOpts<'static> {
    ChatOpts { max_tokens: 2048, temperature: Some(0.0), extra: client::no_think(), ..Default::default() }
}

pub fn read_image(png: &[u8], prompt: &str, model: &str) -> Result<(String, f64), ApiError> {
    let url = format!("data:image/png;base64,{}", util::base64(png));
    let msgs = json!([{"role": "user", "content": [{"type": "text", "text": prompt},
                                                    {"type": "image_url", "image_url": {"url": url}}]}]);
    let t0 = Instant::now();
    let out = client::chat(&msgs, Some(model), exact())?;
    Ok((unfence(&out), t0.elapsed().as_secs_f64()))
}

/// A code (sv) or name → language name; anything unknown passes through as written.
pub fn language(name: &str) -> String {
    let n = name.trim();
    let low = n.to_lowercase();
    if low == "nb" {
        return "Norwegian".into();
    }
    LANGUAGES.iter().find(|(c, _, _)| *c == low).map(|(_, l, _)| l.to_string()).unwrap_or_else(|| n.to_string())
}

pub fn translate_messages(text: &str, lang: &str, prompt: &str, context: &str) -> Value {
    let mut system = prompt.replace("{lang}", lang);
    if !context.trim().is_empty() {
        system += "\n\n";
        system += context.trim();
    }
    json!([{"role": "system", "content": system}, {"role": "user", "content": text}])
}

pub fn translate(text: &str, lang: &str, prompt: &str, model: &str, context: &str) -> Result<(String, f64), ApiError> {
    let t0 = Instant::now();
    let out = client::chat(&translate_messages(text, lang, prompt, context), Some(model), exact())?;
    Ok((unfence(&out), t0.elapsed().as_secs_f64()))
}

fn recent_file() -> PathBuf {
    config::state_dir().join("npukit/translate-recent.json")
}

pub fn recent_languages() -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(recent_file()) else { return vec![] };
    let Ok(Value::Array(a)) = serde_json::from_str::<Value>(&text) else { return vec![] };
    a.iter().filter_map(Value::as_str).filter(|s| !s.trim().is_empty()).map(str::to_string).take(RECENT_MAX).collect()
}

pub fn remember_language(lang: &str) {
    let mut items = vec![lang.to_string()];
    items.extend(recent_languages().into_iter().filter(|x| x.to_lowercase() != lang.to_lowercase()));
    items.truncate(RECENT_MAX);
    let _ = util::write_atomic(&recent_file(), serde_json::to_string(&items).unwrap().as_bytes());
}

/// Rows for the language picker as (shown text, language) and how many lead as recent:
/// the last used first (so Enter repeats it), then the default, then A to Z.
pub fn picker_rows(recent: &[String], default: &str) -> (Vec<(String, String)>, usize) {
    let mut names: Vec<&str> = LANGUAGES.iter().map(|(_, n, _)| *n).collect();
    names.sort();
    let mut order: Vec<String> = vec![];
    for lang in recent.iter().map(String::as_str).chain([default]).chain(names) {
        if !order.iter().any(|x| x.to_lowercase() == lang.to_lowercase()) {
            order.push(lang.to_string());
        }
    }
    let rows = order.into_iter().map(|lang| {
        let (code, own) = LANGUAGES.iter().find(|(_, n, _)| n.to_lowercase() == lang.to_lowercase())
            .map(|(c, _, o)| (*c, *o)).unwrap_or(("", ""));
        (format!("{lang:<23}{code:<7}{own}").trim_end().to_string(), lang)
    }).collect();
    let mut uniq: Vec<String> = recent.iter().map(|r| r.to_lowercase()).collect();
    uniq.sort();
    uniq.dedup();
    (rows, uniq.len())
}

/// Searchable list of languages (rofi). Typing one that isn't listed and pressing Enter uses it as written.
pub fn pick_language(cfg: &Config) -> Option<String> {
    let (rows, n_recent) = picker_rows(&recent_languages(), &language(&cfg.ocr.translate_to));
    let theme = if cfg.ui.lang_theme.is_empty() { &cfg.ui.menu_theme } else { &cfg.ui.lang_theme };
    let mut args: Vec<String> = ["rofi", "-dmenu", "-i", "-p", "into", "-format", "i\ts", "-mesg",
        "Pick a language, then select the screen region. Not listed? Type it and press Enter."].map(String::from).to_vec();
    if n_recent > 0 {
        // recent ones stand out
        args.extend(["-a".into(), (0..n_recent).map(|i| i.to_string()).collect::<Vec<_>>().join(",")]);
    }
    if !theme.is_empty() {
        args.extend(["-theme".into(), config::expand_str(theme)]);
    }
    let lines: Vec<String> = rows.iter().map(|(t, _)| t.clone()).collect();
    match ui::pipe(&args, lines.join("\n").as_bytes()) {
        Ok((code, out)) => parse_pick(code, &String::from_utf8_lossy(&out), &rows),
        Err(_) => {
            ui::notify("Translate", "rofi is needed for the language list", "critical", None);
            None
        }
    }
}

pub fn parse_pick(code: i32, out: &str, rows: &[(String, String)]) -> Option<String> {
    if code != 0 || out.trim().is_empty() {
        return None;
    }
    let out = out.trim_end_matches('\n');
    let (idx, text, tab) = match out.split_once('\t') {
        Some((i, t)) => (i, t, true),
        None => (out, "", false),
    };
    let i: i64 = if tab { idx.parse().unwrap_or(-1) } else { -1 };
    if i >= 0 && (i as usize) < rows.len() {
        return Some(rows[i as usize].1.clone());
    }
    let text = if tab { text } else { idx };
    (!text.trim().is_empty()).then(|| language(text)) // typed, not in the list
}

/// Start loading `model` in the background; the gateway keeps going if this process exits.
fn warm(model: String) {
    std::thread::spawn(move || {
        let _ = client::call("/npu/warm", Some(&json!({"component": "chat", "model": model})), 300.0);
    });
}

pub struct OcrArgs {
    pub describe: bool,
    pub ask: Option<String>,
    pub translate: Option<String>,
    pub pick: bool,
    pub file: Option<String>,
    pub copy: bool,
    pub model: Option<String>,
}

pub fn main(a: &mut OcrArgs) -> i32 {
    let cfg = config::load();
    let o = &cfg.ocr;
    if a.pick {
        // choose the language first, then the region; the reader model loads while you pick
        let reader = a.model.clone().unwrap_or_else(|| if o.translate_ocr_model.is_empty() { o.model.clone() } else { o.translate_ocr_model.clone() });
        warm(reader);
        let Some(chosen) = pick_language(cfg) else { return 1 };
        a.translate = Some(chosen);
        if a.file.is_none() {
            // the region pick freezes the screen: let the list vanish first
            ui::wait_menu_gone(cfg.ui.menu_close_wait, "rofi");
        }
    }
    let png = match &a.file {
        Some(f) => match std::fs::read(f) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("npu: {f}: {e}");
                return 1;
            }
        },
        None => capture(&strs(&o.capture)),
    };
    if png.is_empty() {
        return 1; // selection cancelled
    }
    let prompt = a.ask.clone().unwrap_or_else(|| if a.describe { o.describe_prompt.clone() } else { o.prompt.clone() });
    let lang = a.translate.as_ref().map(|t| language(if t.is_empty() { &o.translate_to } else { t }));
    client::activity(Some("ocr"));
    let reader = a.model.clone().unwrap_or_else(|| match &lang {
        Some(_) if !o.translate_ocr_model.is_empty() => o.translate_ocr_model.clone(),
        _ => o.model.clone(),
    });
    let res = (|| -> Result<(String, String, f64), ApiError> {
        let (text, mut took) = read_image(&png, &prompt, &reader)?;
        if let (false, Some(l)) = (text.is_empty(), &lang) {
            let (out, t2) = translate(&text, l, &o.translate_prompt, &o.translate_model, &o.translate_context)?;
            took += t2;
            remember_language(l);
            return Ok((text, out, took));
        }
        Ok((String::new(), text, took))
    })();
    client::activity(None);
    let (original, text, took) = match res {
        Ok(r) => r,
        Err(e) => {
            ui::notify(if lang.is_some() { "Translate failed" } else { "OCR failed" }, &e.0, "critical", None);
            return 1;
        }
    };
    if text.is_empty() {
        ui::notify("OCR", "No text found", "normal", Some(2500));
        return 1;
    }
    if a.file.is_none() || a.copy {
        ui::copy(&text);
        let title = match &lang {
            Some(l) => format!("→ {l}"),
            None if a.describe => "Describe".into(),
            None if a.ask.is_some() => "Answer".into(),
            None => "OCR".into(),
        };
        ui::notify(&format!("{title} · {took:.1}s on {} · copied", cfg.engine_label()), util::take_chars(&text, 300), "normal",
                   Some(if lang.is_some() { 8000 } else { 5000 }));
    }
    if lang.is_some() {
        eprintln!("{original}");
        eprintln!("---");
    }
    println!("{text}");
    eprintln!("[{took:.1}s]");
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn languages_and_prompts() {
        let o = crate::config::from_str("").unwrap().ocr;
        assert_eq!(language("sv"), "Swedish");
        assert_eq!(language("Klingon"), "Klingon");
        let msgs = translate_messages("hej", "German", &o.translate_prompt, "");
        assert!(msgs[0]["content"].as_str().unwrap().contains("into German"));
        assert_eq!(msgs[1], json!({"role": "user", "content": "hej"}));
        assert!(msgs[0]["content"].as_str().unwrap().contains("never refuse, soften")); // faithful, uncensored by default
        let ctx = translate_messages("hej", "German", &o.translate_prompt, "Game chat: рак = trash player");
        assert!(ctx[0]["content"].as_str().unwrap().ends_with("Game chat: рак = trash player"));
    }

    #[test]
    fn fences_are_removed() {
        let v = json!({"choices": [{"message": {"content": "```text\nhello\nworld\n```"}}]});
        assert_eq!(unfence(&v), "hello\nworld");
    }

    #[test]
    fn parse_pick_cases() {
        let (rows, _) = picker_rows(&[], "English");
        assert_eq!(parse_pick(0, &format!("0\t{}\n", rows[0].0), &rows).as_deref(), Some("English"));
        assert_eq!(parse_pick(0, "-1\tsv\n", &rows).as_deref(), Some("Swedish")); // typed code
        assert_eq!(parse_pick(0, "-1\tPirate\n", &rows).as_deref(), Some("Pirate")); // typed, not listed
        assert_eq!(parse_pick(1, "", &rows), None); // Esc
        assert_eq!(parse_pick(0, "-1\t \n", &rows), None);
    }

    #[test]
    fn rows_recent_first_then_default_then_a_to_z() {
        let recent = vec!["german".to_string(), "Klingon".to_string()];
        let (rows, n) = picker_rows(&recent, "English");
        assert_eq!(n, 2);
        let langs: Vec<&str> = rows.iter().take(4).map(|(_, l)| l.as_str()).collect();
        assert_eq!(langs, ["german", "Klingon", "English", "Arabic"]);
        assert!(rows.iter().find(|(_, l)| l == "Swedish").unwrap().0.contains("sv"));
        assert_eq!(rows.iter().filter(|(_, l)| l.to_lowercase() == "german").count(), 1);
    }
}
