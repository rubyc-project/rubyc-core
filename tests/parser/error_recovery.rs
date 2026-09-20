//! Parser error recovery: invalid programs must produce useful diagnostics
//! instead of panicking.

use crate::parser;

#[test]
fn missing_closing_brace_reports_error() {
    let src = "namespace t { class c { int main() { printf(\"x\"); }";
    let errors = parser::parse(src).expect_err("unbalanced braces must fail");
    assert!(!errors.is_empty());
}

#[test]
fn garbage_input_reports_without_panic() {
    for src in ["@@@@", "class", "int int int", "{", "}"] {
        // Either parses leniently or reports errors — never panics.
        let _ = parser::parse(src);
    }
}

#[test]
fn empty_source_parses_to_an_empty_unit() {
    let unit = parser::parse("").expect("empty source is valid, just empty");
    assert!(unit.namespaces.is_empty());
    assert!(unit.members.is_empty());
    assert!(parser::parse("   \n\t  ").is_ok(), "whitespace-only too");
}

#[test]
fn missing_semicolon_points_at_insertion_spot() {
    let src = "namespace t;\nclass c\n{\n    int main()\n    {\n        printf(\"x\")\n    }\n}";
    let errors = parser::parse(src).expect_err("missing semicolon must fail");
    assert!(
        errors
            .iter()
            .any(|e| e.message.contains("missing ';' after statement")),
        "expected the insertion-point diagnostic, got {:?}",
        errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}
