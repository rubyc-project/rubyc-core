//! Broad coverage net: Display strings, model edge cases, symbol-table
//! behaviors, container parsing, and visitor defaults. If any of these
//! change unexpectedly, something broke.

use crate::models::*;
use crate::native::container::{self, Kind};
use crate::native::target::Target;
use crate::native::ir;
use crate::parser;
use crate::utilities::base64;

// ---------- Display pins ----------

#[test]
fn decode_error_messages_are_stable() {
    let cases: Vec<(DecodeError, &str)> = vec![
        (DecodeError::BadMagic, "not a RubyC file (bad magic)"),
        (
            DecodeError::WrongKind,
            "not a RubyC bytecode file (wrong kind)",
        ),
        (
            DecodeError::UnsupportedVersion(7),
            "bytecode format version 7 is not supported",
        ),
        (DecodeError::UnexpectedEof, "bytecode ended unexpectedly"),
        (DecodeError::UnknownNodeTag(0xAB), "unknown node tag 0xab"),
        (DecodeError::InvalidUtf8, "string data is not valid UTF-8"),
        (
            DecodeError::TrailingBytes,
            "trailing bytes after end of module",
        ),
    ];
    for (err, expected) in cases {
        assert_eq!(err.to_string(), expected);
    }
}

#[test]
fn lower_error_messages() {
    use crate::native::ir::LowerError;
    assert!(
        LowerError::NoMainMethod
            .to_string()
            .starts_with("no 'main' method found")
    );
    assert_eq!(
        LowerError::UnsupportedConstruct("x".into()).to_string(),
        "construct not supported by native backend yet: x"
    );
}

#[test]
fn semantic_lower_error_shows_code_and_help() {
    use crate::native::ir::LowerError;
    use crate::native::semantic::SemanticError;
    let err = LowerError::Semantic(vec![SemanticError {
        code: "E0028",
        message: "class 'C' inherits method 'foo' from both 'B1' and 'B2'".into(),
        help: Some(
            "override the method in the derived class to resolve the ambiguity".into(),
        ),
    }]);
    assert_eq!(
        err.to_string(),
        "error[E0028]: class 'C' inherits method 'foo' from both 'B1' and 'B2'\n  help: override the method in the derived class to resolve the ambiguity"
    );
}

#[test]
fn semantic_lower_error_joins_multiple_diagnostics() {
    use crate::native::ir::LowerError;
    use crate::native::semantic::SemanticError;
    let err = LowerError::Semantic(vec![
        SemanticError {
            code: "E0029",
            message: "class 'C' inherits field 'x' from both 'B1' and 'B2'".into(),
            help: Some("rename the field or the base class".into()),
        },
        SemanticError {
            code: "E0030",
            message: "base class 'B3' not found".into(),
            help: None,
        },
    ]);
    assert_eq!(
        err.to_string(),
        "error[E0029]: class 'C' inherits field 'x' from both 'B1' and 'B2'\n  help: rename the field or the base class\nerror[E0030]: base class 'B3' not found"
    );
}

#[test]
fn format_error_messages() {
    use crate::utilities::format::FormatError;
    assert_eq!(
        FormatError::UnclosedBrace.to_string(),
        "'{' without matching '}'"
    );
    assert_eq!(
        FormatError::InvalidPlaceholder("q".into()).to_string(),
        "invalid placeholder '{q}'"
    );
    assert_eq!(
        FormatError::IndexOutOfRange { index: 4, args: 2 }.to_string(),
        "placeholder {4} but only 2 argument(s) given"
    );
}

#[test]
fn base64_error_messages() {
    assert_eq!(
        base64::Base64Error::InvalidByte(b'!').to_string(),
        "invalid base64 character '!'"
    );
    assert_eq!(
        base64::Base64Error::InvalidLength.to_string(),
        "base64 input has an impossible length"
    );
}

#[test]
fn diagnostic_display_includes_context_path() {
    let d = Diagnostic::new("E0001", Severity::Error, "boom")
        .in_file("a.rc")
        .in_context(ScopeContext {
            namespace: Some("ns".into()),
            class: Some("c".into()),
            function: Some("f".into()),
        });
    assert_eq!(d.to_string(), "error[E0001]: boom in ns::c::f");

    let bare = Diagnostic::new("E0002", Severity::Warning, "plain");
    assert_eq!(bare.to_string(), "warning[E0002]: plain");
}

#[test]
fn target_error_uses_directory_style_name() {
    let err = crate::native::target::TargetError::Unsupported(Target::IosArm64);
    assert_eq!(
        err.to_string(),
        "target os 'ios_arm64' is not implemented yet"
    );
}

#[test]
fn cli_error_messages() {
    use crate::cli::CliError;
    assert_eq!(
        CliError::UnknownCommand("zap".into()).to_string(),
        "unknown command 'zap'"
    );
    assert_eq!(
        CliError::UnknownFlag("--zap".into()).to_string(),
        "unknown option '--zap'"
    );
    assert_eq!(
        CliError::MissingValue("-o".into()).to_string(),
        "option '-o' requires a value"
    );
    assert_eq!(
        CliError::InvalidValue {
            flag: "-O".into(),
            value: "wat".into(),
            allowed: vec!["a".into(), "b".into()]
        }
        .to_string(),
        "invalid value 'wat' for '-O' (allowed: a, b)"
    );
    assert_eq!(
        CliError::DuplicatePositional("x".into()).to_string(),
        "unexpected extra argument 'x'"
    );
}

// ---------- model edges ----------

#[test]
fn span_to_and_slice() {
    let a = Span::new(0, 3);
    let b = Span::new(10, 14);
    let joined = a.to(b);
    assert_eq!((joined.start, joined.end), (0, 14));
    let src = "hello world padding";
    assert_eq!(Span::new(6, 11).slice(src), "world");
    // Overlap takes the union.
    let c = Span::new(2, 20);
    assert_eq!(joined.to(c).end, 20);
}

#[test]
fn keyword_table_is_complete() {
    assert_eq!(Keyword::from_ident("namespace"), Some(Keyword::Namespace));
    assert_eq!(Keyword::from_ident("class"), Some(Keyword::Class));
    assert_eq!(Keyword::from_ident("interface"), Some(Keyword::Interface));
    assert_eq!(Keyword::from_ident("return"), Some(Keyword::Return));
    assert_eq!(Keyword::from_ident("public"), Some(Keyword::Public));
    assert_eq!(Keyword::from_ident("Namespace"), None); // case sensitive
    assert_eq!(Keyword::from_ident(""), None);
}

#[test]
fn kind_byte_round_trip_and_rejection() {
    assert_eq!(Kind::from_byte(0x01), Some(Kind::Bytecode));
    assert_eq!(Kind::from_byte(0x02), Some(Kind::NativeExecutable));
    assert_eq!(Kind::from_byte(0xFF), None);
    assert_eq!(Kind::Bytecode.to_byte(), 0x01);
    assert_eq!(Kind::NativeExecutable.to_byte(), 0x02);
}

#[test]
fn container_rejects_short_input() {
    assert!(container::parse(b"").is_err());
    assert!(container::parse(&[0xEB, 0x08]).is_err());
}

#[test]
fn preamble_length_matches_spec() {
    // jump(2) + magic(8) = 10, plus kind(1)+version(2) = 13 payload start.
    assert_eq!(container::PREAMBLE_LEN, 10);
    assert_eq!(container::build(Kind::Bytecode, 1).len(), 13);
}

// ---------- symbol tables ----------

#[test]
fn scope_stack_lookup_walks_outward() {
    use crate::models::symbols::ScopeStack;
    let mut s = ScopeStack::new();
    s.push();
    s.declare("outer", "int", Access::Default).unwrap();
    s.push();
    assert!(s.lookup("outer").is_some(), "inner scope sees outer names");
    s.pop();
    assert!(s.lookup("outer").is_some());
}

#[test]
fn duplicate_declaration_same_scope_is_error() {
    use crate::models::symbols::ScopeStack;
    let mut s = ScopeStack::new();
    s.push();
    s.declare("x", "int", Access::Default).unwrap();
    // Re-declaring the same name in the same scope is an error (C# CS0136).
    let err = s.declare("x", "long", Access::Default).unwrap_err();
    assert!(
        err.contains("already declared"),
        "expected 'already declared', got: {err}"
    );
    // The original declaration is preserved.
    assert_eq!(s.lookup("x").unwrap().type_name, "int");
    // A nested scope may shadow the outer name (scope isolation).
    s.push();
    s.declare("x", "long", Access::Default).unwrap();
    assert_eq!(s.lookup("x").unwrap().type_name, "long");
    s.pop();
    assert_eq!(s.lookup("x").unwrap().type_name, "int");
}

#[test]
fn temp_slots_never_collide() {
    use crate::models::symbols::ScopeStack;
    let mut s = ScopeStack::new();
    s.push();
    let a = s.temp();
    let b = s.declare("v", "int", Access::Default).unwrap();
    let c = s.temp();
    assert!(a < b && b < c);
    assert_eq!(s.slot_count(), 3);
}

#[test]
fn undeclared_lookup_returns_none() {
    use crate::models::symbols::ScopeStack;
    let mut s = ScopeStack::new();
    s.push();
    assert!(s.lookup("ghost").is_none());
    assert!(s.declare("v", "int", Access::Default).is_ok());
}

// ---------- visitor defaults ----------

#[test]
fn walk_entry_point_visits_unit() {
    struct Probe {
        visited: bool,
    }
    impl crate::models::visitor::Visitor for Probe {
        fn visit_compilation_unit(&mut self, _u: &CompilationUnit) {
            self.visited = true;
        }
    }
    let unit = parser::parse(crate_sample()).unwrap();
    let mut probe = Probe { visited: false };
    crate::models::visitor::walk(&mut probe, &unit);
    assert!(probe.visited);
}

fn crate_sample() -> &'static str {
    crate::testdata::BLOCK_FORM
}

// ---------- lowering diagnostics ----------

#[test]
fn missing_main_reports_cleanly() {
    let unit = parser::parse("namespace t; class c { int start() { printf(\"x\"); } }").unwrap();
    let err = ir::lower(&unit).unwrap_err().to_string();
    assert!(
        err.starts_with("no 'main' method found"),
        "unexpected message: {err}"
    );
    assert!(
        err.contains("help:"),
        "NoMainMethod should carry a fix hint"
    );
}

#[test]
fn unknown_call_reports_function_name() {
    let unit = parser::parse("namespace t; class c { int main() { mystery(1, 2); } }").unwrap();
    let err = ir::lower(&unit).unwrap_err().to_string();
    assert!(err.contains("mystery"), "{err}");
}
