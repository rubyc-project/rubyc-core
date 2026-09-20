//! Structured diagnostics: scope context, colored text rendering, and the
//! machine-readable JSON output.

use crate::diagnostics::{line_col, render_json, render_text};
use crate::parser;

const BROKEN: &str = r#"namespace demo {
    class app {
        int main() {
            int x = ;
        }
    }
}
"#;

#[test]
fn parse_errors_carry_scope_context() {
    let errors = parser::parse(BROKEN).expect_err("should fail");
    assert_eq!(errors.len(), 1);
    let e = &errors[0];
    assert_eq!(e.code, "E0001");
    assert_eq!(e.context.namespace.as_deref(), Some("demo"));
    assert_eq!(e.context.class.as_deref(), Some("app"));
    assert_eq!(e.context.function.as_deref(), Some("main"));
}

#[test]
fn text_render_shows_location_caret_and_context() {
    let errors = parser::parse(BROKEN).unwrap_err();
    let out = render_text(&errors, Some(BROKEN), "input.rc", false);
    assert!(out.contains("ERROR"), "{out}");
    assert!(out.contains("🐕🇨-E0001"), "{out}");
    assert!(out.contains("❯"), "{out}");
    assert!(out.contains("input.rc"), "{out}");
    assert!(out.contains("~"), "tilde expected");
    assert!(out.contains("int x = ;"), "{out}");
    assert!(out.contains("│"), "gutter expected");
}

#[test]
fn caret_sits_on_its_own_line_under_the_error() {
    // Regression: the underline used to trail the code on the same line
    // (`... int x = ;   ~`) instead of sitting aligned beneath the error
    // column on the following line.
    let errors = parser::parse(BROKEN).unwrap_err();
    let out = render_text(&errors, Some(BROKEN), "input.rc", false);
    let mut lines = out.lines();
    let code_line = lines
        .find(|l| l.contains("int x = ;"))
        .expect("code line missing");
    assert!(
        !code_line.contains('~'),
        "carets must not trail the code line: {code_line:?}"
    );
    let caret_line = lines.next().expect("caret line missing");
    assert!(caret_line.contains('~'), "caret line missing: {out}");
    let code_col = code_line.find("int x = ;").unwrap();
    let caret_col = caret_line.find('~').unwrap();
    // `;` is the last char of the snippet; the caret must sit under it
    // (gutter widths are identical on both lines).
    assert_eq!(
        caret_col,
        code_col + "int x = ;".len() - 1,
        "caret misaligned:\n{out}"
    );
}

#[test]
fn inline_parser_context_needs_no_locale_key() {
    // Context strings passed inline ("to close ...", "as ...") are already
    // English; they must render verbatim, never as "[missing translation]".
    let errors = parser::parse("class C { Main() { int x = (1; } }\n").unwrap_err();
    assert!(!errors.is_empty());
    let out = render_text(&errors, Some("class C { Main() { int x = (1; } }\n"), "x", false);
    assert!(
        !out.contains("[missing translation]"),
        "inline context leaked to lookup: {out}"
    );
    assert!(out.contains("to close"), "{out}");
}

#[test]
fn colors_appear_only_when_enabled() {
    let errors = parser::parse(BROKEN).unwrap_err();
    let plain = render_text(&errors, Some(BROKEN), "x", false);
    let colored = render_text(&errors, Some(BROKEN), "x", true);
    assert!(!plain.contains('\x1b'));
    assert!(colored.contains("\x1b["));
}

#[test]
fn json_compact_is_one_line_with_all_fields() {
    let errors = parser::parse(BROKEN).unwrap_err();
    let out = render_json(&errors, Some(BROKEN), false);
    assert!(!out.contains('\n'), "compact JSON must be one line");
    for key in [
        "\"code\"",
        "\"severity\"",
        "\"message\"",
        "\"file\"",
        "\"line\"",
        "\"column\"",
        "\"endColumn\"",
        "\"namespace\"",
        "\"class\"",
        "\"function\"",
        "\"sourceLine\"",
    ] {
        assert!(out.contains(key), "missing {key} in {out}");
    }
    assert!(out.contains("\"namespace\":\"demo\""));
    assert!(out.contains("\"function\":\"main\""));
}

#[test]
fn json_pretty_is_multiline() {
    let errors = parser::parse(BROKEN).unwrap_err();
    let out = render_json(&errors, Some(BROKEN), true);
    assert!(out.starts_with('['));
    assert!(out.contains("\n  {\n"));
    assert!(out.ends_with(']'));
}

#[test]
fn json_escapes_special_characters() {
    use crate::diagnostics::json_escape;
    assert_eq!(
        json_escape("quote\" back\\ newline\n"),
        "quote\\\" back\\\\ newline\\n"
    );
}

#[test]
fn line_col_resolution() {
    let src = "abc\ndef\nghi";
    assert_eq!(line_col(src, 0), (1, 1));
    assert_eq!(line_col(src, 4), (2, 1));
    assert_eq!(line_col(src, 6), (2, 3));
    // Out-of-range clamps instead of panicking.
    assert_eq!(line_col(src, 999), (3, 4));
}
