//! i18n: generated lookups, fallback chain, templates, language switching.
//!
//! Language selection is process-global state, so every test that asserts
//! locale-specific text takes the shared lock below to avoid racing other
//! suites' expectations.

use crate::i18n::translate;
use crate::t;
use crate::tf;
use std::sync::{Mutex, MutexGuard};

static LANG_LOCK: Mutex<()> = Mutex::new(());

fn lock_lang() -> MutexGuard<'static, ()> {
    match LANG_LOCK.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Reset to English on drop so other suites start from a known state.
struct LangGuard {
    _guard: MutexGuard<'static, ()>,
}

impl Drop for LangGuard {
    fn drop(&mut self) {
        crate::i18n::set_lang("en");
    }
}

fn with_lang() -> LangGuard {
    LangGuard {
        _guard: lock_lang(),
    }
}

#[test]
fn every_en_key_translates() {
    let _g = with_lang();
    // A representative sample from each area; build.rs guarantees the rest.
    for key in [
        "rubyc.cli.render.title",
        "rubyc.cli.COMPILE.about",
        "rubyc.cli.flag.verbose",
        "rubyc.cli.err.unknown_command",
        "rubyc.main.build.stage1",
        "rubyc.models.diagnostic.DecodeError.bad_magic",
        "rubyc.parser.rd.Parser.expect.generic",
        "rubyc.parser.lexer.Lexer.ident_or_keyword.utf8_expect",
        "rubyc.native.ir.LowerError.no_main",
        "rubyc.format.FormatError.unclosed_brace",
        "rubyc.utilities.base64.Base64Error.invalid_length",
        "rubyc.native.target.TargetError.unsupported",
        "rubyc.native.target.TargetError.no_shared_library",
        "rubyc.native.target.TargetError.no_static_archive",
        "rubyc.native.target.TargetError.bad_object",
        "rubyc.lint.W0001",
        "rubyc.lint.W0002",
        "rubyc.models.symbols.Scope.declare.duplicate",
        // Delegate declaration context phrases.
        "rubyc.parser.rd.Parser.ctx.as_delegate_name",
        "rubyc.parser.rd.Parser.ctx.as_generic_parameter",
        "rubyc.parser.rd.Parser.ctx.type_name_segment",
        "rubyc.parser.rd.Parser.ctx.open_delegate_params",
        "rubyc.parser.rd.Parser.ctx.close_delegate_params",
        "rubyc.parser.rd.Parser.ctx.after_delegate",
      ] {
        assert_ne!(translate(key), "[missing translation]", "key {key}");
        assert!(!translate(key).is_empty());
    }
}

#[test]
fn unknown_keys_show_marker_in_en() {
    let _g = with_lang();
    assert_eq!(translate("rubyc.does.not.exist"), "[missing translation]");
}

#[test]
fn unknown_language_falls_back_to_english() {
    let _g = with_lang();
    crate::i18n::set_lang("xx");
    assert!(t!("rubyc.cli.render.title").starts_with("RubyC Compiler v"));
}

#[test]
fn template_fill_with_arguments() {
    let _g = with_lang();
    let out = tf!("rubyc.cli.err.missing_value", "-o");
    assert_eq!(out, "option '-o' requires a value");

    let out = tf!("rubyc.cli.err.invalid_value", "wat", "-O", "a, b");
    assert_eq!(
        out, "invalid value 'wat' for '-O' (allowed: a, b)",
        "positional args fill in order"
    );
}
