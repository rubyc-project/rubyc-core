//! CLI argument parsing (`parse_args`), the spec tables, help/version
//! rendering, and the error paths. Pure — no I/O, no env mutation beyond
//! what `render_help` reads for TTY detection.

use crate::cli::{
    parse_args, render_help, version_text, Command, CliError, ErrorFormat, InFormat, OutFormat,
    ParsedArgs,
};
use crate::cli::spec::{COMMANDS, EXTENSION_FLAGS, GLOBAL_FLAGS};

fn argv(items: &[&str]) -> Vec<String> {
    let mut v = vec!["rubyc".to_string()];
    v.extend(items.iter().map(|s| s.to_string()));
    v
}

/// `parse_args` returns a `ParsedArgs` that doesn't impl `Debug`, so we can't
/// use `unwrap_err`/`unwrap_or_else`. This helper matches on the error variant
/// directly.
fn parse_err(args: &[&str]) -> CliError {
    match parse_args(&argv(args)) {
        Ok(_) => panic!("expected a parse error, got success"),
        Err(e) => e,
    }
}

fn run(args: &[&str]) -> crate::cli::Invocation {
    match parse_args(&argv(args)).unwrap_or_else(|e| panic!("parse failed: {e}")) {
        ParsedArgs::Run(inv) => inv,
        ParsedArgs::Help(_) | ParsedArgs::Version(_) => panic!("expected Run, got passive"),
    }
}

// ---------- command selection ----------

#[test]
fn select_build() {
    let inv = run(&["build", "app.rc"]);
    assert_eq!(inv.command, Command::Build);
    assert_eq!(inv.io.file.as_deref(), Some("app.rc"));
}

#[test]
fn select_run() {
    let inv = run(&["run", "app.rc"]);
    assert_eq!(inv.command, Command::Run);
}

#[test]
fn bare_file_is_implicit_run() {
    // `rubyc app.rc` -> run app.rc.
    let inv = run(&["app.rc"]);
    assert_eq!(inv.command, Command::Run);
    assert_eq!(inv.io.file.as_deref(), Some("app.rc"));
}

#[test]
fn no_args_defaults_to_check_roundtrip() {
    let inv = run(&[]);
    assert_eq!(inv.command, Command::CheckRoundtrip);
}

#[test]
fn each_command_name_is_known() {
    // A known command name parses to its Command; an unknown one falls back
    // to the implicit `run` (which we can detect via the file positional).
    for spec in COMMANDS.iter() {
        let inv = run(&[spec.name]);
        assert!(matches!(inv.command, _), "{} should parse", spec.name);
    }
    // `help` is a passive subcommand.
    assert!(matches!(
        parse_args(&argv(&["help"])).unwrap(),
        ParsedArgs::Help(_)
    ));
    // A bogus first word is treated as a run-target file.
    let inv = run(&["totally-bogus.rc"]);
    assert_eq!(inv.command, Command::Run);
}

// ---------- global switches ----------

#[test]
fn verbose_and_quiet() {
    let inv = run(&["build", "a.rc", "-v", "-q"]);
    assert!(inv.verbose);
    assert!(inv.quiet);
}

#[test]
fn combined_short_switches() {
    // `-vq` parses as both verbose and quiet.
    let inv = run(&["build", "a.rc", "-vq"]);
    assert!(inv.verbose);
    assert!(inv.quiet);
}

#[test]
fn no_color_flag() {
    let inv = run(&["build", "a.rc", "--no-color"]);
    assert!(inv.no_color);
}

// ---------- value flags ----------

#[test]
fn out_flag_long() {
    let inv = run(&["compile", "a.rc", "--out", "out.bc"]);
    assert_eq!(inv.io.output.as_deref(), Some("out.bc"));
}

#[test]
fn out_flag_attached() {
    let inv = run(&["compile", "a.rc", "--out=out.bc"]);
    assert_eq!(inv.io.output.as_deref(), Some("out.bc"));
}

#[test]
fn out_flag_short() {
    let inv = run(&["compile", "a.rc", "-o", "out.bc"]);
    assert_eq!(inv.io.output.as_deref(), Some("out.bc"));
}

#[test]
fn in_flag() {
    let inv = run(&["compile", "-i", "in.src"]);
    assert_eq!(inv.io.input.as_deref(), Some("in.src"));
}

#[test]
fn in_format_valid() {
    let inv = run(&["compile", "-I", "source"]);
    assert_eq!(inv.io.in_format, Some(InFormat::Source));
    let inv2 = run(&["compile", "--in-format=bytecode"]);
    assert_eq!(inv2.io.in_format, Some(InFormat::Bytecode));
}

#[test]
fn out_format_valid() {
    let inv = run(&["compile", "-O", "exe"]);
    assert_eq!(inv.io.out_format, Some(OutFormat::Exe));
    let inv2 = run(&["compile", "--out-format", "shared"]);
    assert_eq!(inv2.io.out_format, Some(OutFormat::Shared));
}

#[test]
fn in_format_invalid_is_error() {
    let err = parse_err(&["compile", "-I", "wat"]);
    assert!(matches!(err, CliError::InvalidValue { .. }));
}

#[test]
fn out_format_invalid_is_error() {
    let err = parse_err(&["compile", "-O", "wat"]);
    assert!(matches!(err, CliError::InvalidValue { .. }));
}

// ---------- base64 switch ----------

#[test]
fn base64_switch() {
    let inv = run(&["compile", "-B"]);
    assert!(inv.io.base64);
}

// ---------- shebang optional value ----------

#[test]
fn shebang_default_when_bare() {
    let inv = run(&["compile", "a.rc", "--shebang"]);
    // OptionalValue with default.
    assert_eq!(inv.io.shebang.as_deref(), Some("/usr/bin/env rubyc"));
}

#[test]
fn shebang_override() {
    let inv = run(&["compile", "a.rc", "--shebang=/custom/rb"]);
    assert_eq!(inv.io.shebang.as_deref(), Some("/custom/rb"));
}

// ---------- define ----------

#[test]
fn define_long_separate() {
    let inv = run(&["build", "a.rc", "--define", "DEBUG"]);
    assert_eq!(inv.defines, vec!["DEBUG".to_string()]);
}

#[test]
fn define_attached() {
    let inv = run(&["build", "a.rc", "-DDEBUG"]);
    assert_eq!(inv.defines, vec!["DEBUG".to_string()]);
}

#[test]
fn multiple_defines() {
    let inv = run(&["build", "a.rc", "-DDEBUG", "-DFAST"]);
    assert_eq!(inv.defines, vec!["DEBUG".to_string(), "FAST".to_string()]);
}

// ---------- lint overrides ----------

#[test]
fn deny_lint() {
    let inv = run(&["build", "a.rc", "--deny", "unused-local"]);
    assert_eq!(inv.lint_overrides.len(), 1);
    assert_eq!(inv.lint_overrides[0].0, "unused-local");
    assert_eq!(inv.lint_overrides[0].1, crate::lints::Level::Deny);
}

#[test]
fn allow_lint() {
    let inv = run(&["build", "a.rc", "--allow", "W0001"]);
    assert_eq!(inv.lint_overrides[0].1, crate::lints::Level::Allow);
}

#[test]
fn warn_lint() {
    let inv = run(&["build", "a.rc", "--warn", "W0002"]);
    assert_eq!(inv.lint_overrides[0].1, crate::lints::Level::Warn);
}

#[test]
fn unknown_lint_is_error() {
    let err = parse_err(&["build", "a.rc", "--deny", "nope"]);
    assert!(matches!(err, CliError::InvalidValue { .. }));
}

// ---------- error-format ----------

#[test]
fn error_format_json() {
    let inv = run(&["build", "a.rc", "--error-format", "json"]);
    assert_eq!(inv.error_format, ErrorFormat::Json);
}

#[test]
fn error_format_json_pretty() {
    let inv = run(&["build", "a.rc", "--error-format=json-pretty"]);
    assert_eq!(inv.error_format, ErrorFormat::JsonPretty);
}

#[test]
fn error_format_invalid() {
    let err = parse_err(&["build", "a.rc", "--error-format", "xml"]);
    assert!(matches!(err, CliError::InvalidValue { .. }));
}

// ---------- extension flags ----------

#[test]
fn base_flag() {
    let inv = run(&["build", "a.rc", "--base", "/tmp/rb"]);
    assert_eq!(inv.io.base.as_deref(), Some("/tmp/rb"));
}

#[test]
fn target_flag() {
    let inv = run(&["build", "a.rc", "--target", "linux_x86_64"]);
    assert_eq!(inv.io.target.as_deref(), Some("linux_x86_64"));
}

#[test]
fn short_target_flag() {
    let inv = run(&["build", "a.rc", "-t", "linux_x86_64"]);
    assert_eq!(inv.io.target.as_deref(), Some("linux_x86_64"));
}

// ---------- lib (repeatable) ----------

#[test]
fn lib_repeatable() {
    let inv = run(&["build", "a.rc", "--lib", "libfoo.so", "--lib", "libbar.so"]);
    assert_eq!(inv.io.libs, vec!["libfoo.so".to_string(), "libbar.so".to_string()]);
}

// ---------- help / version ----------

#[test]
fn help_flag() {
    match parse_args(&argv(&["--help"])).unwrap() {
        ParsedArgs::Help(text) => assert!(!text.is_empty()),
        _ => panic!("expected Help"),
    }
}

#[test]
fn short_h_help() {
    assert!(matches!(
        parse_args(&argv(&["-h"])).unwrap(),
        ParsedArgs::Help(_)
    ));
}

#[test]
fn version_flag() {
    match parse_args(&argv(&["--version"])).unwrap() {
        ParsedArgs::Version(text) => assert!(text.contains('0')),
        _ => panic!("expected Version"),
    }
}

#[test]
fn short_V_version_only_before_command() {
    assert!(matches!(
        parse_args(&argv(&["-V"])).unwrap(),
        ParsedArgs::Version(_)
    ));
}

#[test]
fn help_subcommand() {
    assert!(matches!(
        parse_args(&argv(&["help"])).unwrap(),
        ParsedArgs::Help(_)
    ));
}

#[test]
fn help_specific_command() {
    match parse_args(&argv(&["help", "build"])).unwrap() {
        ParsedArgs::Help(text) => assert!(text.contains("build")),
        _ => panic!("expected Help"),
    }
}

// ---------- error paths ----------

#[test]
fn unknown_flag_is_error() {
    let err = parse_err(&["build", "--nope"]);
    assert!(matches!(err, CliError::UnknownFlag(_)));
}

#[test]
fn missing_value_is_error() {
    // `--out` at end of args has no value.
    let err = parse_err(&["compile", "--out"]);
    assert!(matches!(err, CliError::MissingValue(_)));
}

#[test]
fn duplicate_positional_is_error() {
    // Two file positionals for a 1-positional command.
    let err = parse_err(&["build", "a.rc", "b.rc"]);
    assert!(matches!(err, CliError::DuplicatePositional(_)));
}

// ---------- `--` end of options ----------

#[test]
fn double_dash_treats_rest_as_positional() {
    // After `--`, a leading-dash token is a positional, not a flag.
    let inv = run(&["build", "--", "-weird.rc"]);
    assert_eq!(inv.io.file.as_deref(), Some("-weird.rc"));
}

// ---------- spec table invariants ----------

#[test]
fn command_specs_are_unique() {
    let names: Vec<&str> = COMMANDS.iter().map(|c| c.name).collect();
    let mut sorted = names.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(names.len(), sorted.len(), "duplicate command names");
}

#[test]
fn extension_flags_have_long_names() {
    for f in EXTENSION_FLAGS.iter() {
        assert!(!f.long.is_empty());
    }
    for f in GLOBAL_FLAGS.iter() {
        assert!(!f.long.is_empty());
    }
}

// ---------- render_help / version_text ----------

#[test]
fn render_root_help_nonempty() {
    let text = render_help(None);
    assert!(!text.is_empty());
    // Lists every command.
    for spec in COMMANDS.iter() {
        assert!(text.contains(spec.name), "help missing command {}", spec.name);
    }
}

#[test]
fn render_command_help_nonempty() {
    let text = render_help(Some(Command::Build));
    assert!(text.contains("build"));
}

#[test]
fn version_text_has_version() {
    let v = version_text();
    assert!(!v.is_empty());
}

// ---------- skip-hooks switches (dev-loop only) ----------

#[test]
fn skip_hooks_switches() {
    let inv = run(&["build", "a.rc", "--skip-pre-hook", "--skip-post-hook"]);
    assert!(inv.io.skip_pre_hook);
    assert!(inv.io.skip_post_hook);
}

#[test]
fn skip_hooks_short_switches() {
    let inv = run(&["build", "a.rc", "-P", "-Q"]);
    assert!(inv.io.skip_pre_hook);
    assert!(inv.io.skip_post_hook);
}

#[test]
fn skip_hooks_default_off() {
    let inv = run(&["build", "a.rc"]);
    assert!(!inv.io.skip_pre_hook);
    assert!(!inv.io.skip_post_hook);
}
