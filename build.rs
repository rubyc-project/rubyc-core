//! Build-time i18n code generation.
//!
//! Reads `locales/<lang>.json` files (flat string->string maps), validates
//! them against each other, and generates a `lookup(lang, key)` function in
//! `$OUT_DIR/i18n_generated.rs` that the `i18n` module includes.
//!
//! The JSON reader is hand-rolled: the schema is a flat object of string
//! values only, so a full parser would be over-engineering.

use std::collections::BTreeMap;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=locales");
    println!("cargo:rerun-if-changed=build.rs");
    emit_target_os_cfg();

    let root = env::var("CARGO_MANIFEST_DIR").unwrap();
    let locales_dir = Path::new(&root).join("locales");

    let mut langs: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    let mut entries: Vec<PathBuf> = fs::read_dir(&locales_dir)
        .unwrap_or_else(|e| panic!("cannot read locales dir: {e}"))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    entries.sort();

    if entries.is_empty() {
        panic!("no locale files found in locales/");
    }

    for path in &entries {
        let lang = path
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("locale file name must be valid UTF-8")
            .to_owned();
        let source =
            fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let pairs = parse_flat_json(&source)
            .unwrap_or_else(|e| panic!("{}:{}: {e}", path.display(), e.offset));
        for (key, _) in &pairs {
            assert!(
                key.starts_with("rubyc."),
                "{lang}.json: key '{key}' must start with 'rubyc.'"
            );
            assert!(
                key.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_'),
                "{lang}.json: key '{key}' may only contain [A-Za-z0-9_.]"
            );
        }
        let before = langs.len();
        langs.insert(lang, pairs);
        assert_eq!(langs.len(), before + 1, "duplicate language file");
    }

    // Every language must carry exactly the same key set as the first one.
    let reference = langs.values().next().expect("at least one locale");
    for (lang, pairs) in &langs {
        let ref_keys: Vec<&str> = reference.iter().map(|(k, _)| k.as_str()).collect();
        let my_keys: Vec<&str> = pairs.iter().map(|(k, _)| k.as_str()).collect();
        let missing: Vec<_> = ref_keys.iter().filter(|k| !my_keys.contains(k)).collect();
        if let Some(missing) = missing.first() {
            panic!("{lang}.json is missing key '{missing}'");
        }
        let extra: Vec<_> = my_keys.iter().filter(|k| !ref_keys.contains(k)).collect();
        if let Some(extra) = extra.first() {
            panic!("{lang}.json has unknown key '{extra}'");
        }
    }

    // Audit: every t!()/tf!() literal referenced in src must exist in en.json.
    audit_referenced_keys(Path::new(&root).join("src"), reference);

    generate(&langs);
}

/// Scan all crate sources for i18n key literals so a typo'd or renamed key
/// breaks the build instead of leaking `[missing: ...]` at runtime.
fn audit_referenced_keys(src_dir: PathBuf, en: &[(String, String)]) {
    fn visit(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).expect("src readable") {
            let p = entry.expect("entry").path();
            if p.is_dir() {
                visit(&p, out);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(p);
            }
        }
    }
    let mut files = Vec::new();
    visit(&src_dir, &mut files);

    let mut referenced: Vec<String> = Vec::new();
    for file in &files {
        let content = fs::read_to_string(file).unwrap();
        let bytes = content.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            // Find t!( or tf!( followed by an optional whitespace and '"'.
            let t_pos = content[i..].find("t!(").map(|p| i + p);
            let tf_pos = content[i..].find("tf!(").map(|p| i + p);
            let pos = match (t_pos, tf_pos) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                _ => None,
            };
            let Some(pos) = pos else { break };
            // Ensure it's exactly t!/tf! (not part of an identifier like "at!(").
            let ok_char =
                pos == 0 || !(bytes[pos - 1].is_ascii_alphanumeric() || bytes[pos - 1] == b'_');
            if !ok_char {
                i = pos + 3;
                continue;
            }
            let after = pos + 3;
            let rest = &content[after..];
            let rest_trimmed_start = after + rest.len() - rest.trim_start().len();
            if rest.trim_start().starts_with('"') {
                let quote = rest_trimmed_start;
                if let Some(rel_close) = content[quote + 1..].find('"') {
                    let key = &content[quote + 1..quote + 1 + rel_close];
                    referenced.push(key.to_owned());
                }
            }
            i = after;
        }
    }

    for key in &referenced {
        let exists = en.iter().any(|(k, _)| k == key);
        assert!(
            exists,
            "referenced i18n key '{key}' does not exist in locales/en.json"
        );
    }
}

// ---------- minimal flat-JSON parser ----------

pub struct JsonError {
    pub offset: usize,
    pub message: String,
}

impl std::fmt::Display for JsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (at byte {})", self.message, self.offset)
    }
}

struct P<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> P<'a> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn expect(&mut self, b: u8) -> Result<(), JsonError> {
        if self.peek() == Some(b) {
            self.i += 1;
            Ok(())
        } else {
            Err(JsonError {
                offset: self.i,
                message: format!("expected '{}'", b as char),
            })
        }
    }

    fn string(&mut self) -> Result<String, JsonError> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            match self.peek() {
                None => {
                    return Err(JsonError {
                        offset: self.i,
                        message: "unterminated string".into(),
                    });
                }
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    let esc = self.peek().ok_or(JsonError {
                        offset: self.i,
                        message: "unterminated escape".into(),
                    })?;
                    self.i += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            if self.i + 4 > self.s.len() {
                                return Err(JsonError {
                                    offset: self.i,
                                    message: "bad \\u escape".into(),
                                });
                            }
                            let hex =
                                std::str::from_utf8(&self.s[self.i..self.i + 4]).map_err(|_| {
                                    JsonError {
                                        offset: self.i,
                                        message: "bad \\u escape".into(),
                                    }
                                })?;
                            let cp = u32::from_str_radix(hex, 16).map_err(|_| JsonError {
                                offset: self.i,
                                message: "bad \\u escape".into(),
                            })?;
                            self.i += 4;
                            out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                        }
                        other => {
                            return Err(JsonError {
                                offset: self.i - 1,
                                message: format!("unknown escape '\\{}'", other as char),
                            });
                        }
                    }
                }
                Some(_) => {
                    let start = self.i;
                    while self.peek().is_some()
                        && self.peek() != Some(b'"')
                        && self.peek() != Some(b'\\')
                    {
                        self.i += 1;
                    }
                    out.push_str(std::str::from_utf8(&self.s[start..self.i]).map_err(|_| {
                        JsonError {
                            offset: start,
                            message: "invalid UTF-8".into(),
                        }
                    })?);
                }
            }
        }
    }
}

/// Parse `{ "key": "value", ... }` — nothing else is allowed.
pub fn parse_flat_json(text: &str) -> Result<Vec<(String, String)>, JsonError> {
    let mut p = P {
        s: text.as_bytes(),
        i: 0,
    };
    p.ws();
    p.expect(b'{')?;
    let mut out = Vec::new();
    p.ws();
    if p.peek().is_none_or(|b| b == b'}') {
        return Ok(out);
    }
    loop {
        p.ws();
        let key = p.string()?;
        if out.iter().any(|(k, _): &(String, String)| *k == key) {
            return Err(JsonError {
                offset: p.i,
                message: format!("duplicate key '{key}'"),
            });
        }
        p.ws();
        p.expect(b':')?;
        p.ws();
        let value = p.string()?;
        out.push((key, value));
        p.ws();
        match p.peek() {
            Some(b',') => {
                p.i += 1;
            }
            Some(b'}') => {
                p.i += 1;
                break;
            }
            _ => {
                return Err(JsonError {
                    offset: p.i,
                    message: "expected ',' or '}'".into(),
                });
            }
        }
    }
    p.ws();
    if p.i != p.s.len() {
        return Err(JsonError {
            offset: p.i,
            message: "trailing characters after object".into(),
        });
    }
    Ok(out)
}

// ---------- generator ----------

fn generate(langs: &BTreeMap<String, Vec<(String, String)>>) {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest = Path::new(&out_dir).join("i18n_generated.rs");
    let mut g = String::new();

    writeln!(g, "// @generated by build.rs from locales/ - do not edit\n").unwrap();
    writeln!(
        g,
        "pub const LANGS: &[&str] = &[{}];\n",
        langs
            .keys()
            .map(|l| format!("{l:?}"))
            .collect::<Vec<_>>()
            .join(", ")
    )
    .unwrap();

    writeln!(
        g,
        "pub fn lookup(lang: &str, key: &str) -> Option<&'static str> {{"
    )
    .unwrap();
    writeln!(g, "    match lang {{").unwrap();
    for (lang, pairs) in langs {
        writeln!(g, "        {lang:?} => match key {{").unwrap();
        for (key, value) in pairs {
            writeln!(g, "            {key:?} => Some({value:?}),").unwrap();
        }
        writeln!(g, "            _ => None,").unwrap();
        writeln!(g, "        }},").unwrap();
    }
    writeln!(g, "        _ => None,").unwrap();
    writeln!(g, "    }}").unwrap();
    writeln!(g, "}}").unwrap();

    fs::write(dest, g).expect("write generated i18n");
}

/// Compile-time platform selection (Rust's equivalent of C# `#if`).
///
/// Maps the cargo-provided target OS/arch pair onto one canonical
/// `target_os_<name>` cfg flag whose `<name>` is byte-identical to the
/// directories under `src/native/target_os/`. This function is the single
/// place in the project that mentions rustc's `target_os`/`target_arch`
/// words; all source-level gates use only our flags.
fn emit_target_os_cfg() {
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    let names: &[&str] = &[
        "linux_x86_64",
        "linux_arm64",
        "macos_arm64",
        "macos_x64",
        "windows_x64",
        "windows_arm64",
        "android_arm64",
        "ios_arm64",
    ];
    for name in names {
        println!("cargo:rustc-check-cfg=cfg(target_os_{name})");
    }

    let selected = match (os.as_str(), arch.as_str()) {
        ("linux", "x86_64") => Some("linux_x86_64"),
        ("linux", "aarch64") => Some("linux_arm64"),
        ("macos", "aarch64") => Some("macos_arm64"),
        ("macos", "x86_64") => Some("macos_x64"),
        ("windows", "x86_64") => Some("windows_x64"),
        ("windows", "aarch64") => Some("windows_arm64"),
        ("android", _) => Some("android_arm64"),
        ("ios", _) => Some("ios_arm64"),
        _ => None, // unknown host: no flag; linux_x86_64 remains the default fallback
    };
    if let Some(name) = selected {
        println!("cargo:rustc-cfg=target_os_{name}");
    }
}
