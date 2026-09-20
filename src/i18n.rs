//! Internationalization: all user-facing text comes from `locales/*.json`
//! via build-time generation.
//!
//! Keys are crate paths (`rubyc.parser.lexer.Lexer.ident_or_keyword.utf8_expect`)
//! so every message traces back to the exact code location that emits it.

include!(concat!(env!("OUT_DIR"), "/i18n_generated.rs"));

use crate::utilities::format;
use std::sync::OnceLock;

use std::sync::RwLock;

static LANG: RwLock<Option<String>> = RwLock::new(None);

/// The active language: `RUBYC_LANG` env if valid, else `"en"`.
pub fn lang() -> &'static str {
    // Fast path: explicit override.
    if let Some(code) = LANG.read().ok().and_then(|g| g.clone()) {
        return Box::leak(code.into_boxed_str());
    }
    static ENV: OnceLock<String> = OnceLock::new();
    let code = ENV.get_or_init(|| {
        let code = std::env::var("RUBYC_LANG").unwrap_or_else(|_| "en".to_owned());
        if lookup(&code, "rubyc.cli.render.title").is_some() {
            code
        } else {
            "en".to_owned()
        }
    });
    Box::leak(code.clone().into_boxed_str())
}

/// Override the language at runtime. Invalid codes are recorded as-is;
/// lookups fall back to English when a language has no table.
pub fn set_lang(code: &str) {
    if let Ok(mut slot) = LANG.write() {
        *slot = Some(code.to_owned());
    }
}

/// Translate a key in the active language, falling back to English, and
/// finally to a visible marker so gaps are obvious.
pub fn translate(key: &str) -> &'static str {
    let lang = lang();
    lookup(lang, key)
        .or_else(|| (lang != "en").then(|| lookup("en", key)).flatten())
        .unwrap_or("[missing translation]")
}

/// Translate a template key and fill `{0} {1} ...` placeholders.
///
/// Uses the same interpolation engine as printf lowering.
pub fn translate_format(key: &str, args: &[String]) -> String {
    format::interpolate(translate(key), args).unwrap_or_else(|_| format!("[missing: {key}]"))
}

/// Shorthand macro for plain lookups.
#[macro_export]
macro_rules! t {
    ($key:expr) => {
        $crate::i18n::translate($key)
    };
}

/// Shorthand macro for templated lookups with `{N}` arguments.
#[macro_export]
macro_rules! tf {
    ($key:expr $(, $arg:expr)* $(,)?) => {{
        let args: Vec<String> = vec![$($arg.to_string()),*];
        $crate::i18n::translate_format($key, &args)
    }};
}
