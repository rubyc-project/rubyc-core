//! Preprocessor directive tests: `#define`/`#undef`, conditional blocks
//! (`#If`/`#Else`/`#Elif`/`#EndIf`), `#Error`/`#Warning`, `#Assembly` capture,
//! expression macros + `expand_macros`, and every malformed-input error path.
//!
//! All inputs are in-memory strings — no files, no env, fully deterministic.

use crate::models::{Diagnostic, Severity};
use crate::preprocessor::{expand_macros, preprocess, PreprocOutput};

fn ok(src: &str) -> PreprocOutput {
    preprocess(src, &[], "test.rubyc").expect("preprocess should succeed")
}

fn diags_of(src: &str) -> Vec<Diagnostic> {
    match preprocess(src, &[], "test.rubyc") {
        Ok(_) => Vec::new(),
        Err(d) => d,
    }
}

fn has_error(diags: &[Diagnostic], needle: &str) -> bool {
    diags.iter().any(|d| d.severity == Severity::Error && d.message.contains(needle))
}

// ---------- #define / #undef ----------

#[test]
fn define_adds_symbol() {
    let out = ok("#define FOO\n1");
    assert!(out.symbols.contains(&"FOO".to_string()));
}

#[test]
fn define_respects_cli_symbols() {
    let out = preprocess("#define BAR\n1", &["CLI".to_string()], "t").unwrap();
    assert!(out.symbols.contains(&"CLI".to_string()));
    assert!(out.symbols.contains(&"BAR".to_string()));
}

#[test]
fn define_is_idempotent() {
    let out = ok("#define FOO\n#define FOO\n1");
    let count = out.symbols.iter().filter(|s| *s == "FOO").count();
    assert_eq!(count, 1, "duplicate #define must not double the symbol");
}

#[test]
fn undef_removes_symbol() {
    let out = ok("#define FOO\n#undef FOO\n1");
    assert!(!out.symbols.contains(&"FOO".to_string()));
}

#[test]
fn undef_inside_inactive_branch_is_ignored() {
    // The #undef is in an excluded branch → FOO survives.
    let out = ok("#If NOPE\n#undef FOO\n#EndIf\n#define FOO\n1");
    assert!(out.symbols.contains(&"FOO".to_string()));
}

// ---------- conditionals ----------

#[test]
fn if_true_keeps_body() {
    let out = ok("#define FOO\n#If FOO\nkept\n#EndIf");
    assert!(out.source.contains("kept"));
}

#[test]
fn if_false_blanks_body_but_preserves_line_count() {
    let src = "#If NOPE\nhidden\nstill_hidden\n#EndIf\ntail";
    let out = ok(src);
    assert!(!out.source.contains("hidden"));
    // Line count is preserved so later spans/carets stay exact.
    assert_eq!(out.source.lines().count(), src.lines().count());
    assert!(out.source.contains("tail"));
}

#[test]
fn else_switches_branch_when_if_false() {
    let out = ok("#If NOPE\na\n#Else\nb\n#EndIf");
    // The `a` branch is excluded, `b` (the else branch) is kept.
    let kept: Vec<&str> = out.source.lines().filter(|l| !l.trim().is_empty()).collect();
    assert!(kept.iter().any(|l| l.trim() == "b"));
    assert!(!kept.iter().any(|l| l.trim() == "a"));
}

#[test]
fn else_does_not_run_when_if_true() {
    let out = ok("#define FOO\n#If FOO\na\n#Else\nb\n#EndIf");
    assert!(out.source.contains("a"));
    // The else body 'b' must be blanked (only 'a' kept).
    let kept: Vec<&str> = out.source.lines().filter(|l| !l.trim().is_empty()).collect();
    assert!(kept.iter().any(|l| l.contains("a")));
    assert!(!kept.iter().any(|l| l.trim() == "b"));
}

#[test]
fn elif_runs_when_earlier_branch_failed() {
    let out = ok("#define B\n#If A\nx\n#Elif B\ny\n#EndIf");
    let kept: Vec<&str> = out.source.lines().filter(|l| !l.trim().is_empty()).collect();
    assert!(kept.iter().any(|l| l.contains("y")));
    assert!(!kept.iter().any(|l| l.trim() == "x"));
}

#[test]
fn elif_not_run_when_earlier_branch_active() {
    let out = ok("#define A\n#If A\nx\n#Elif B\ny\n#EndIf");
    let kept: Vec<&str> = out.source.lines().filter(|l| !l.trim().is_empty()).collect();
    assert!(kept.iter().any(|l| l.contains("x")));
    assert!(!kept.iter().any(|l| l.trim() == "y"));
}

#[test]
fn nested_ifs_and() {
    let out = ok("#define A\n#define B\n#If A\n#If B\nboth\n#EndIf\n#EndIf");
    assert!(out.source.contains("both"));
}

#[test]
fn nested_if_inner_false() {
    let out = ok("#define A\n#If A\n#If NOPE\nhidden\n#EndIf\nouter\n#EndIf");
    let kept: Vec<&str> = out.source.lines().filter(|l| !l.trim().is_empty()).collect();
    assert!(kept.iter().any(|l| l.contains("outer")));
    assert!(!kept.iter().any(|l| l.contains("hidden")));
}

#[test]
fn endif_without_if_is_error() {
    let d = diags_of("#EndIf");
    assert!(has_error(&d, "matching #If"));
}

#[test]
fn else_without_if_is_error() {
    let d = diags_of("#Else");
    assert!(has_error(&d, "matching #If"));
}

#[test]
fn elif_without_if_is_error() {
    let d = diags_of("#Elif A");
    assert!(has_error(&d, "matching #if"));
}

#[test]
fn missing_endif_is_error() {
    let d = diags_of("#define A\n#If A\nx");
    assert!(has_error(&d, "missing #EndIf"));
}

// ---------- #Error / #Warning ----------

#[test]
fn error_aborts_with_diagnostic() {
    let d = diags_of("#Error \"boom\"");
    assert!(has_error(&d, "boom"));
}

#[test]
fn error_in_inactive_branch_is_suppressed() {
    // No error → preprocess succeeds.
    let out = ok("#If NOPE\n#Error \"boom\"\n#EndIf\n1");
    assert!(out.diagnostics.is_empty());
}

#[test]
fn warning_does_not_abort() {
    let out = ok("#Warning \"careful\"\n1");
    assert!(out.diagnostics.iter().any(|d| {
        d.severity == Severity::Warning && d.message.contains("careful")
    }));
}

// ---------- unknown / inert directives ----------

#[test]
fn unknown_directive_is_error() {
    let d = diags_of("#Frobnicate x");
    assert!(has_error(&d, "unknown preprocessor directive"));
}

#[test]
fn nullable_region_directives_are_inert() {
    // Must not error and must blank the line.
    let out = ok("#nullable enable\n#region foo\n#endregion\n1");
    assert!(out.diagnostics.is_empty());
}

#[test]
fn error_warning_directives_match_either_case() {
    // C# spells them lowercase (`#error`, `#warning`); both cases must
    // behave identically. Active `#error` is a compile error per .NET
    // (platform files use `#else` / `#error Unknown target` guards);
    // inactive-branch ones are ignored.
    let out = crate::preprocessor::preprocess(
        "#if NEVER_DEFINED\n#error nope\n#endif\n1",
        &[],
        "x",
    )
    .unwrap();
    assert!(out.diagnostics.is_empty());
    // Active `#error` aborts with its message (preprocess returns Err
    // when any diagnostic is an error).
    let err = crate::preprocessor::preprocess("#error boom", &[], "x").unwrap_err();
    assert!(
        err.iter().any(|d| d.message.contains("boom")),
        "active #error must surface its message"
    );
    let warn = crate::preprocessor::preprocess("#warning careful", &[], "x").unwrap();
    assert!(
        warn.diagnostics.iter().any(|d| d.message.contains("careful")),
        "active #warning must surface its message"
    );
}

#[test]
fn pragma_warning_directive_is_inert() {
    // `#pragma warning disable/restore X` only controls reported warnings;
    // it must not error and must blank the line like the other inert
    // directives (pulled dotnet sources use it, e.g. WebClient.cs).
    let out = ok("#pragma warning disable SYSLIB0014 // obsolete use\n1\n#pragma warning restore SYSLIB0014\n2");
    assert!(out.diagnostics.is_empty());
    assert!(out.source.contains('1'));
}

// ---------- #Assembly ----------

#[test]
fn assembly_block_captured_verbatim() {
    let out = ok("#Assembly\nmov rax, 1\nadd rbx, rax\n#EndAssembly\n1");
    assert_eq!(out.assembly_blocks.len(), 1);
    assert!(out.assembly_blocks[0].contains("mov rax, 1"));
    assert!(out.assembly_blocks[0].contains("add rbx, rax"));
    // The source line count is preserved (block becomes one marker line).
    assert!(out.source.lines().count() >= 3);
}

#[test]
fn endassembly_without_start_is_error() {
    let d = diags_of("#EndAssembly");
    assert!(has_error(&d, "matching #Assembly"));
}

#[test]
fn unclosed_assembly_is_error() {
    let d = diags_of("#Assembly\nmov rax, 1");
    assert!(has_error(&d, "unclosed #Assembly"));
}

// ---------- expression macros ----------

#[test]
fn macro_definition_parsed() {
    let out = ok("#macro sq(x) => (x)*(x)\n1");
    assert_eq!(out.macros.len(), 1);
    let (name, params, body) = &out.macros[0];
    assert_eq!(name, "sq");
    assert_eq!(params, &vec!["x".to_string()]);
    assert_eq!(body, "(x)*(x)");
}

#[test]
fn malformed_macro_is_error() {
    let d = diags_of("#macro sq x => (x)");
    assert!(has_error(&d, "malformed #macro"));
}

#[test]
fn expand_macros_substitutes_args() {
    // Each arg is paren-wrapped, and the whole call result is paren-wrapped.
    let macros = vec![("sq".to_string(), vec!["x".to_string()], "(x)*(x)".to_string())];
    let out = expand_macros("sq(3)".to_string(), &macros);
    assert_eq!(out, "(((3))*((3)))");
}

#[test]
fn expand_macros_multiple_params() {
    let macros = vec![("add".to_string(), vec!["a".to_string(), "b".to_string()], "(a)+(b)".to_string())];
    let out = expand_macros("add(1, 2)".to_string(), &macros);
    assert_eq!(out, "(((1))+((2)))");
}

#[test]
fn expand_macros_word_boundary() {
    // `xsq(3)` must NOT match `sq` (preceded by alnum).
    let macros = vec![("sq".to_string(), vec!["x".to_string()], "(x)".to_string())];
    let out = expand_macros("xsq(3)".to_string(), &macros);
    assert_eq!(out, "xsq(3)", "non-boundary name is left alone");
}

#[test]
fn expand_macros_nested_call_args() {
    let macros = vec![("f".to_string(), vec!["x".to_string()], "(x)".to_string())];
    // Balanced-paren scan must handle a nested paren in the arg: the arg is
    // `g(1)`, paren-wrapped → `((g(1)))`, then the whole call paren-wrapped.
    let out = expand_macros("f(g(1))".to_string(), &macros);
    assert_eq!(out, "(((g(1))))");
}

#[test]
fn expand_macros_no_macros_is_noop() {
    let out = expand_macros("sq(3)".to_string(), &[]);
    assert_eq!(out, "sq(3)");
}

// ---------- C# conditional expressions ----------

fn cond_body(src: &str, symbols: &[&str]) -> String {
    let syms: Vec<String> = symbols.iter().map(|s| s.to_string()).collect();
    preprocess(src, &syms, "t")
        .expect("preprocess should succeed")
        .source
}

#[test]
fn if_not_undefined_symbol_is_active() {
    // .NET: undefined symbols are false, so `#if !NATIVEAOT` includes
    // when NATIVEAOT is not defined (broke every `#if !X` region in
    // dotnet/runtime, e.g. MethodBase.InvokerStrategy).
    let out = cond_body("#if !NATIVEAOT\nkept\n#endif\n", &[]);
    assert!(out.contains("kept"), "negated undefined symbol must include");
}

#[test]
fn if_not_defined_symbol_is_inactive() {
    let out = cond_body("#if !NATIVEAOT\nkept\n#endif\n", &["NATIVEAOT"]);
    assert!(!out.contains("kept"), "negated defined symbol must exclude");
}

#[test]
fn if_and_or_parens_evaluate() {
    let src = "#if (MONO && (TARGET_AMD64 || TARGET_ARM64)) || (!MONO && TARGET_X86)\nkept\n#endif\n";
    assert!(cond_body(src, &["MONO", "TARGET_AMD64"]).contains("kept"));
    assert!(cond_body(src, &["TARGET_X86"]).contains("kept"));
    assert!(!cond_body(src, &["MONO", "TARGET_X86"]).contains("kept"));
}

#[test]
fn if_eq_compares_definedness() {
    // `DEBUG == false` holds exactly when DEBUG is undefined.
    let src = "#if DEBUG == false\nkept\n#endif\n";
    assert!(cond_body(src, &[]).contains("kept"));
    assert!(!cond_body(src, &["DEBUG"]).contains("kept"));
}

#[test]
fn elif_uses_full_expression() {
    let src = "#if NOPE\nno\n#elif !ALSO_NOPE\nyes\n#endif\n";
    assert!(cond_body(src, &[]).contains("yes"));
}

#[test]
fn trailing_line_comment_ignored_on_directives() {
    // `#if !MONO // details` — the comment is not part of the
    // expression (silently excluded Array.cs's MethodTable region).
    let src = "#if !MONO // implementation details\nkept\n#endif\n";
    assert!(cond_body(src, &[]).contains("kept"));
    // ... and not part of a defined symbol either.
    let out = ok("#define FOO // the foo\n#if FOO\nkept\n#endif\n");
    assert!(out.symbols.contains(&"FOO".to_string()));
    assert!(out.source.contains("kept"));
}

#[test]
fn negated_undefined_symbol_includes() {
    // `#if !RUNTIME_ASYNC_SUPPORTED` with the symbol undefined
    // includes the region (AsyncProfiler fallback world).
    let src = "#if !RUNTIME_ASYNC_SUPPORTED\nint x = 1;\n#endif\nint y = 2;\n";
    assert!(cond_body(src, &[]).contains("int x = 1;"));
}

#[test]
fn directive_trailing_comment_without_space() {
    // `#endif//TRUST_COMPILE_STATE` (no whitespace before the
    // trailing comment, System.Private.Xml): the keyword is the
    // leading alphabetic run. A whitespace split misses it and the
    // `#if` stack corrupts ("missing #EndIf").
    let src = "#if TRUST_COMPILE_STATE\nint x = 1;\n#endif//TRUST_COMPILE_STATE\nint y = 2;\n";
    assert!(cond_body(src, &["TRUST_COMPILE_STATE"]).contains("int x = 1;"));
    assert!(cond_body(src, &[]).contains("int y = 2;"));
}
