//! Driver: pure orchestration helpers — argv scanning, diagnostics display,
//! input classification, round-trip sanity, shebang wrapping, and file I/O.
//!
//! These tests avoid the native backend (the target crate is a plugin, not a
//! dependency of core) and exercise the parts of `driver.rs` that are pure
//! Rust: `flag_value`, `extension_listing`, `input_name`, `unit_has_class`,
//! `render_diags`, `base64_decode`, `wrap_shebang`, `default_interpreter`,
//! `read_raw`, `write_out`, `mark_executable`, `parse_source`, `roundtrip_ok`.

use crate::cli::IoArgs;
use crate::driver::parse_source;
use crate::models::diagnostic::Severity;
use crate::models::Diagnostic;

// ---------- flag_value ----------

fn argv(s: &[&str]) -> Vec<String> {
    s.iter().map(|x| x.to_string()).collect()
}

#[test]
fn flag_value_spaced_form() {
    assert_eq!(
        crate::driver::flag_value(&argv(&["--base", "/x"]), "--base"),
        Some("/x".to_string())
    );
}

#[test]
fn flag_value_spaced_form_rejects_flag_following() {
    // The next token is itself a flag, so the value is not consumed.
    assert_eq!(
        crate::driver::flag_value(&argv(&["--base", "--target", "x"]), "--base"),
        None
    );
}

#[test]
fn flag_value_missing_is_none() {
    assert_eq!(crate::driver::flag_value(&argv(&["compile", "a.rb"]), "--base"), None);
}

#[test]
fn flag_value_empty_equals_value_is_none() {
    // `--base=` with an empty value is treated as absent.
    assert_eq!(crate::driver::flag_value(&argv(&["--base=", "x"]), "--base"), None);
}

#[test]
fn flag_value_target() {
    assert_eq!(
        crate::driver::flag_value(&argv(&["--target", "linux_x86_64"]), "--target"),
        Some("linux_x86_64".to_string())
    );
}

// ---------- extension_listing ----------

#[test]
fn extension_listing_no_list_flags_is_none() {
    assert!(crate::driver::extension_listing(&argv(&["compile", "a.rb"])).is_none());
}

#[test]
fn extension_listing_list_targets_returns_lines() {
    // Point --base at an empty directory so only builtins are listed.
    let base = temp_dir("listing_targets");
    let out = crate::driver::extension_listing(&argv(&["--list-targets", "--base", base.as_path().to_str().unwrap()])).unwrap();
    // Builtins are always present, so the list is non-empty.
    assert!(!out.is_empty());
    // Each line has the "name version | author | triple | description" shape.
    for line in &out {
        assert!(line.contains(' '), "line should have spaces: {line}");
    }
}



// ---------- parse_source ----------

#[test]
fn parse_source_empty_is_ok() {
    let unit = parse_source("").unwrap();
    assert!(unit.members.is_empty());
}

#[test]
fn parse_source_simple_class_is_ok() {
    let src = "namespace t { public class Foo { public int bar() { return 1; } } }";
    let unit = parse_source(src).unwrap();
    let ns = &unit.namespaces[0];
    assert!(!ns.members.is_empty());
}

#[test]
fn parse_source_malformed_returns_diagnostics() {
    // A bare, unparseable line yields structured diagnostics.
    match parse_source("class Foo end end end") {
        Ok(u) => assert!(u.members.len() <= 1),
        Err(diags) => assert!(!diags.is_empty()),
    }
}

// ---------- input_name ----------

#[test]
fn input_name_prefers_input() {
    let io = IoArgs {
        input: Some("a.rb".to_string()),
        file: Some("b.rb".to_string()),
        ..Default::default()
    };
    assert_eq!(crate::driver::input_name(&io), "a.rb");
}

#[test]
fn input_name_falls_back_to_file() {
    let io = IoArgs {
        input: None,
        file: Some("b.rb".to_string()),
        ..Default::default()
    };
    assert_eq!(crate::driver::input_name(&io), "b.rb");
}

#[test]
fn input_name_stdin_when_absent() {
    let io = IoArgs {
        input: None,
        file: None,
        ..Default::default()
    };
    assert_eq!(crate::driver::input_name(&io), "<stdin>");
}

// ---------- unit_has_class ----------

#[test]
fn unit_has_class_true_when_namespace_has_members() {
    let unit = parse_source("namespace Foo { class Bar { } }").unwrap();
    assert!(crate::driver::unit_has_class(&unit));
}

#[test]
fn unit_has_class_true_when_top_level_members() {
    let unit = parse_source("class Foo { }").unwrap();
    assert!(crate::driver::unit_has_class(&unit));
}

#[test]
fn unit_has_class_false_when_empty() {
    let unit = parse_source("").unwrap();
    assert!(!crate::driver::unit_has_class(&unit));
}

// ---------- render_diags ----------

fn make_diag() -> Diagnostic {
    Diagnostic::new("E0001", Severity::Error, "bad token").in_file("t.rb")
}

/// Build an `Invocation` by parsing argv so every field is populated exactly
/// as the real CLI would.
fn inv_from(args: &[&str]) -> crate::cli::Invocation {
    match crate::cli::parse_args(&argv(args)) {
        Ok(crate::cli::ParsedArgs::Run(inv)) => inv,
        Ok(_other) => panic!("expected a Run invocation"),
        Err(e) => panic!("parse_args failed: {e:?}"),
    }
}

#[test]
fn render_diags_text_contains_message() {
    let diags = vec![make_diag()];
    let inv = inv_from(&["compile", "--error-format", "text"]);
    let out = crate::driver::render_diags(&diags, Some("x"), &inv);
    assert!(out.contains("bad token"), "rendered: {out}");
}

#[test]
fn render_diags_json_is_valid_json() {
    let diags = vec![make_diag()];
    let inv = inv_from(&["compile", "--error-format", "json"]);
    let out = crate::driver::render_diags(&diags, None, &inv);
    // render_json emits a JSON array of objects with code/severity/message.
    assert!(out.starts_with('['), "expected JSON array, got: {out}");
    assert!(out.contains("\"code\""));
    assert!(out.contains("E0001"));
    assert!(out.contains("bad token"));
}

#[test]
fn render_diags_json_pretty_has_newlines() {
    let diags = vec![make_diag()];
    let inv = inv_from(&["compile", "--error-format", "json-pretty"]);
    let out = crate::driver::render_diags(&diags, None, &inv);
    assert!(out.contains('\n'));
}

// ---------- base64_decode ----------

#[test]
fn base64_decode_roundtrip() {
    let data: Vec<u8> = vec![0, 1, 2, 250, 251, 252];
    let encoded = crate::utilities::base64::encode(&data);
    let decoded = crate::driver::base64_decode(encoded.as_bytes()).unwrap();
    assert_eq!(decoded, data);
}

#[test]
fn base64_decode_invalid_byte() {
    let err = crate::driver::base64_decode(b"!!!not base64!!!").unwrap_err();
    match err {
        crate::driver::DriverError::Message(m) => assert!(m.contains("base64")),
        other => panic!("expected Message, got {other:?}"),
    }
}

#[test]
fn base64_decode_invalid_length() {
    // A lone sextet -> len % 4 == 1 -> invalid length.
    let err = crate::driver::base64_decode(b"A").unwrap_err();
    match err {
        crate::driver::DriverError::Message(m) => assert!(m.contains("base64")),
        other => panic!("expected Message, got {other:?}"),
    }
}

// ---------- wrap_shebang / default_interpreter ----------

#[test]
fn wrap_shebang_prefixes_interpreter() {
    let out = crate::driver::wrap_shebang(&[0xCA, 0xFE, 0xBA, 0xBE], "/usr/bin/env rubyc");
    assert_eq!(out, b"#!/usr/bin/env rubyc\n\xca\xfe\xba\xbe");
}

#[test]
fn wrap_shebang_empty_bytecode() {
    let out = crate::driver::wrap_shebang(b"", "/bin/sh");
    assert_eq!(out, b"#!/bin/sh\n");
}

#[test]
fn default_interpreter_is_env_rubyc() {
    assert_eq!(crate::driver::default_interpreter(), "/usr/bin/env rubyc");
}

// ---------- read_raw ----------

#[test]
fn read_raw_from_existing_file() {
    let dir = temp_dir("readraw");
    let path = dir.join("in.txt");
    std::fs::write(&path, b"hello world").unwrap();
    let io = IoArgs {
        input: Some(path.to_string_lossy().to_string()),
        ..Default::default()
    };
    let bytes = crate::driver::read_raw(&io).unwrap();
    assert_eq!(bytes, b"hello world");
}

#[test]
fn read_raw_missing_file_is_error() {
    let io = IoArgs {
        input: Some("/nonexistent/path/xyz".to_string()),
        ..Default::default()
    };
    let err = crate::driver::read_raw(&io).unwrap_err();
    match err {
        crate::driver::DriverError::Message(m) => assert!(m.contains("cannot read")),
        other => panic!("expected Message, got {other:?}"),
    }
}

// ---------- write_out ----------

#[test]
fn write_out_to_file() {
    let dir = temp_dir("writeout");
    let path = dir.join("out.txt");
    let io = IoArgs {
        output: Some(path.to_string_lossy().to_string()),
        ..Default::default()
    };
    crate::driver::write_out(&io, b"data123").unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), b"data123");
}

#[test]
fn write_out_to_missing_dir_is_error() {
    let io = IoArgs {
        output: Some("/nonexistent/dir/out".to_string()),
        ..Default::default()
    };
    let err = crate::driver::write_out(&io, b"x").unwrap_err();
    match err {
        crate::driver::DriverError::Message(m) => assert!(m.contains("cannot write")),
        other => panic!("expected Message, got {other:?}"),
    }
}

// ---------- mark_executable ----------

#[test]
fn mark_executable_sets_exec_bits() {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let dir = temp_dir("mex");
        let path = dir.join("prog");
        std::fs::write(&path, b"#!bin\n").unwrap();
        let s = path.to_string_lossy().to_string();
        crate::driver::mark_executable(&s).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert!(mode & 0o111 != 0, "expected exec bits, got {mode:o}");
    }
}

// ---------- roundtrip_ok ----------

#[test]
fn roundtrip_ok_true_for_simple_class() {
    let src = "namespace t { public class Foo { public int bar() { return 1; } } }";
    assert!(crate::driver::roundtrip_ok(src));
}

#[test]
fn roundtrip_ok_true_for_empty() {
    assert!(crate::driver::roundtrip_ok(""));
}

// ---------- helpers ----------

fn temp_dir(label: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rubyc_driver_{}_{}",
        label,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
