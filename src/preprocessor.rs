//! Preprocessor: line-based directives handled BEFORE tokenization.
//!
//! Supported:
//!   #define SYM          — define a conditional-compilation symbol
//!   #undef SYM           — remove a symbol
//!   --define SYM (CLI)   — pre-seeded symbols
//!   #If EXPR / #Else / #EndIf — C# conditional blocks (nestable):
//!     `!`, `==`, `!=`, `&&`, `||`, parens, `true`/`false`; an
//!     undefined symbol is false (so `#if !NATIVEAOT` includes when
//!     NATIVEAOT is not defined, exactly like .NET).
//!   #Error "msg"         — abort with a compile error
//!   #Warning "msg"       — emit a warning, keep going
//!   #Assembly … #EndAssembly — raw block captured verbatim onto the unit
//!     for the native assembler (encoding milestone).
//!
//! Design rules:
//!   * Directives are recognized only at the START of a line.
//!   * Excluded `#If` branches are dropped while PRESERVING line numbers,
//!     so spans/carets in the remaining text stay exact.
//!   * No textual macro substitution — constants belong to the language.

use crate::models::{Diagnostic, Severity, Span};

#[derive(Debug, Default, Clone)]
pub struct PreprocOutput {
    /// Source with excluded lines blanked (newlines kept) and assembly
    /// blocks replaced by a single marker comment line each.
    pub source: String,
    pub symbols: Vec<String>,
    /// Raw text of every `#Assembly` block, in declaration order.
    pub assembly_blocks: Vec<String>,
    /// Expression macros: name → (params, replacement expression text).
    pub macros: Vec<(String, Vec<String>, String)>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum IfState {
    Active,
    /// A branch already matched; further branches wait for #Else/#EndIf.
    Done,
    Waiting,
}

/// One token of a `#if` / `#elif` conditional expression.
#[derive(Debug, Clone, Copy, PartialEq)]
enum CondToken<'a> {
    Sym(&'a str),
    Not,
    And,
    Or,
    Eq,
    NotEq,
    Open,
    Close,
}

/// Recursive-descent evaluator for C# conditional-compilation
/// expressions (`||` > `&&` > `==`/`!=` > `!` > primary), the way
/// .NET evaluates them: `true`/`false` literals, parentheses, and
/// symbols that are true iff defined (undefined is false).
/// Malformed input evaluates false (inactive branch) rather than
/// aborting the build. Zero heap allocation beyond the token
/// vector, which borrows the input.
struct CondParser<'a> {
    tokens: Vec<CondToken<'a>>,
    pos: usize,
    symbols: &'a [String],
}

impl<'a> CondParser<'a> {
    fn peek(&self) -> Option<CondToken<'a>> {
        self.tokens.get(self.pos).copied()
    }

    fn parse_or(&mut self) -> bool {
        let mut value = self.parse_and();
        while self.peek() == Some(CondToken::Or) {
            self.pos += 1;
            value = self.parse_and() || value;
        }
        value
    }

    fn parse_and(&mut self) -> bool {
        let mut value = self.parse_eq();
        while self.peek() == Some(CondToken::And) {
            self.pos += 1;
            value = self.parse_eq() && value;
        }
        value
    }

    fn parse_eq(&mut self) -> bool {
        let mut value = self.parse_unary();
        loop {
            match self.peek() {
                Some(CondToken::Eq) => {
                    self.pos += 1;
                    value = self.parse_unary() == value;
                }
                Some(CondToken::NotEq) => {
                    self.pos += 1;
                    value = self.parse_unary() != value;
                }
                _ => return value,
            }
        }
    }

    fn parse_unary(&mut self) -> bool {
        if self.peek() == Some(CondToken::Not) {
            self.pos += 1;
            return !self.parse_unary();
        }
        match self.peek() {
            Some(CondToken::Open) => {
                self.pos += 1;
                let value = self.parse_or();
                if self.peek() == Some(CondToken::Close) {
                    self.pos += 1;
                }
                value
            }
            Some(CondToken::Sym(name)) => {
                self.pos += 1;
                name == "true" || self.symbols.iter().any(|s| s == name)
            }
            _ => false,
        }
    }
}

/// Evaluate a `#if` / `#elif` condition the way .NET does; see
/// [`CondParser`]. Empty or malformed input is false.
fn eval_condition(expr: &str, symbols: &[String]) -> bool {
    let bytes = expr.as_bytes();
    let mut tokens: Vec<CondToken> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b' ' | b'\t' => i += 1,
            b'!' => {
                if bytes.get(i + 1) == Some(&b'=') {
                    tokens.push(CondToken::NotEq);
                    i += 2;
                } else {
                    tokens.push(CondToken::Not);
                    i += 1;
                }
            }
            b'&' if bytes.get(i + 1) == Some(&b'&') => {
                tokens.push(CondToken::And);
                i += 2;
            }
            b'|' if bytes.get(i + 1) == Some(&b'|') => {
                tokens.push(CondToken::Or);
                i += 2;
            }
            b'=' if bytes.get(i + 1) == Some(&b'=') => {
                tokens.push(CondToken::Eq);
                i += 2;
            }
            b'(' => {
                tokens.push(CondToken::Open);
                i += 1;
            }
            b')' => {
                tokens.push(CondToken::Close);
                i += 1;
            }
            b'A'..=b'Z' | b'a'..=b'z' | b'_' => {
                let start = i;
                while i < bytes.len()
                    && matches!(
                        bytes[i],
                        b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_'
                    )
                {
                    i += 1;
                }
                tokens.push(CondToken::Sym(&expr[start..i]));
            }
            // `false` lexes as a symbol and evaluates false (undefined);
            // any other byte is malformed: inactive, never abort.
            _ => return false,
        }
    }
    if tokens.is_empty() {
        return false;
    }
    let mut parser = CondParser {
        tokens,
        pos: 0,
        symbols,
    };
    let value = parser.parse_or();
    // Trailing garbage (e.g. legacy `#if A B`) is malformed: false.
    value && parser.pos == parser.tokens.len()
}

pub fn preprocess(
    source: &str,
    cli_symbols: &[String],
    file: &str,
) -> Result<PreprocOutput, Vec<Diagnostic>> {
    let mut out = PreprocOutput {
        source: String::with_capacity(source.len()),
        symbols: cli_symbols.to_vec(),
        macros: Vec::new(),
        assembly_blocks: Vec::new(),
        diagnostics: Vec::new(),
    };

    let mut if_stack: Vec<IfState> = Vec::new();
    let mut in_asm = false;
    let mut asm_buf = String::new();

    'lines: for (idx, raw_line) in source.lines().enumerate() {
        let lineno = idx + 1;
        let trimmed_start = raw_line.trim_start();

        // ---- inside #Assembly capture: everything is raw until #EndAssembly
        if in_asm {
            if trimmed_start == "#EndAssembly" {
                in_asm = false;
                out.assembly_blocks.push(std::mem::take(&mut asm_buf));
            } else {
                asm_buf.push_str(raw_line);
                asm_buf.push('\n');
            }
            // Keep line count stable for later stages.
            out.source.push('\n');
            continue;
        }

        let active_branches = if_stack.iter().all(|s| *s == IfState::Active);
        let directive = trimmed_start.starts_with('#');

        // ---- directives ----
        if directive {
            let body = trimmed_start[1..].trim();
            // Directive keyword = leading alphabetic run: trailing text
            // may abut without whitespace (`#endif//TRUST_COMPILE_STATE`
            // in System.Private.Xml) — a whitespace split misses the
            // keyword and corrupts the `#if` stack ("missing #EndIf").
            let cut = body
                .find(|c: char| !c.is_ascii_alphabetic())
                .unwrap_or(body.len());
            let (kw, rest) = body.split_at(cut);
            let rest = rest.trim();

            /// A `//` comment may trail a directive (`#if !MONO //
            /// details`, `#define X // why`): it is not part of the
            /// expression or symbol. Strip it here — but never for
            /// `#error` / `#warning`, whose message may contain `//`
            /// (URLs). Conditions contain no strings, so a naive
            /// split is exact.
            fn strip_line_comment(rest: &str) -> &str {
                match rest.find("//") {
                    Some(i) => rest[..i].trim_end(),
                    None => rest,
                }
            }

            match kw {
                "define" => {
                    let rest = strip_line_comment(rest);
                    if active_branches && !rest.is_empty() && !out.symbols.contains(&rest.to_owned())
                    {
                        out.symbols.push(rest.to_owned());
                    }
                }
                "undef" => {
                    if active_branches {
                        let rest = strip_line_comment(rest);
                        out.symbols.retain(|s| s != rest);
                    }
                }
                "If" | "if" => {
                    // Full C# conditional expression (see `eval_condition`).
                    let satisfied = active_branches
                        && eval_condition(strip_line_comment(rest), &out.symbols);
                    if_stack.push(if satisfied {
                        IfState::Active
                    } else {
                        IfState::Waiting
                    });
                }
                "Else" | "else" => {
                    let st: Option<&mut IfState> = if_stack.last_mut();
                    match st {
                        Some(st) => {
                            if *st == IfState::Active {
                                *st = IfState::Done; // active → skip rest
                            } else if *st == IfState::Waiting {
                                *st = IfState::Active; // else-branch may run
                            }
                        }
                        _ => {
                            out.diagnostics.push(err_diag(
                                "#Else without matching #If",
                                file,
                                lineno,
                            ));
                        }
                    }
                }
                "Elif" | "elif" => {
                    let parents_active = if_stack[..if_stack.len().saturating_sub(1)]
                        .iter().all(|state| *state == IfState::Active);
                    match if_stack.last_mut() {
                        Some(state) if *state == IfState::Active => *state = IfState::Done,
                        Some(state) if *state == IfState::Waiting => {
                            if parents_active
                                && eval_condition(strip_line_comment(rest), &out.symbols)
                            {
                                *state = IfState::Active;
                            }
                        }
                        Some(_) => {}
                        None => out.diagnostics.push(err_diag(
                            "#elif without matching #if", file, lineno,
                        )),
                    }
                }
                "EndIf" | "endif" => {
                    if if_stack.pop().is_none() {
                        out.diagnostics.push(err_diag(
                            "#EndIf without matching #If",
                            file,
                            lineno,
                        ));
                    }
                }
                // C# directives are lowercase by convention (`#error`,
                // `#warning`); accept both cases like the `#If`/`#EndIf`
                // arms below. An active `#error` is a genuine compile
                // error per .NET (e.g. platform files whose `#else`
                // branch fires `#error Unknown target`).
                "Error" | "error" => {
                    if active_branches {
                        out.diagnostics.push(err_diag(
                            &format!("#Error {rest}"),
                            file,
                            lineno,
                        ));
                    }
                }
                "Warning" | "warning" => {
                    if active_branches {
                        out.diagnostics.push(warn_diag(
                            &format!("#Warning {rest}"),
                            file,
                            lineno,
                        ));
                    }
                }
                // C# source-organization/nullability directives. RubyC does
                // not implement nullable-flow analysis yet, but these do not
                // alter runtime semantics and must not hide the source from
                // subsequent compiler stages.
                "nullable" | "region" | "endregion" => {}
                // `#pragma warning disable/restore X` only controls which
                // warnings the compiler reports; it never changes runtime
                // semantics, so accept-and-blank like the other inert
                // directives. Warning-ID filtering is future work.
                "pragma" => {}
                "macro" => {
                    // Single-line expression macro:
                    //   #macro sq(x) => (x)*(x)
                    if !active_branches {
                        out.source.push('\n');
                        continue 'lines;
                    }
                    if let Some(expanded) = parse_macro_def(rest) {
                        out.macros.push(expanded);
                    } else {
                        out.diagnostics.push(err_diag(
                            "malformed #macro — expected: #macro name(p1,p2) => expression",
                            file,
                            lineno,
                        ));
                    }
                }
                "Assembly" => {
                    if in_asm {
                        out.diagnostics.push(err_diag(
                            "nested #Assembly block",
                            file,
                            lineno,
                        ));
                    }
                    in_asm = true;
                    out.source.push('\n');
                    continue 'lines;
                }
                // `#EndAssembly` closes the block. It used to share this
                // arm with two misspellings (`#EndAssmbly`, `#End`) and
                // then only ever REPORT "without matching #Assembly" —
                // so `#Assembly … #EndAssembly` reported an error and
                // left `in_asm` set, blanking the rest of the file.
                "EndAssembly" => {
                    if !in_asm {
                        out.diagnostics.push(err_diag(
                            "#EndAssembly without matching #Assembly",
                            file,
                            lineno,
                        ));
                    }
                    in_asm = false;
                }
                other => {
                    if active_branches {
                        out.diagnostics.push(err_diag(
                            &format!("unknown preprocessor directive '#{other}'"),
                            file,
                            lineno,
                        ));
                    }
                }
            }
            // Directives themselves become blank lines (line numbers stay).
            out.source.push('\n');
            continue 'lines;
        }

        // ---- normal / excluded content lines ----
        if active_branches {
            out.source.push_str(raw_line);
        } else if raw_line.trim().is_empty() {
            out.source.push_str(raw_line); // keep blank-line rhythm
        }
        // excluded non-blank lines become empty (newline kept below)
        out.source.push('\n');
    }

    if in_asm {
        out.diagnostics.push(err_diag("unclosed #Assembly block", file, 0));
    }
    if !if_stack.is_empty() {
        out.diagnostics.push(err_diag("missing #EndIf", file, 0));
    }

    if out.diagnostics.iter().any(|d| d.severity == Severity::Error) {
        Err(out.diagnostics)
    } else {
        Ok(out)
    }
}

fn err_diag(msg: &str, file: &str, lineno: usize) -> Diagnostic {
    Diagnostic::new("E0004", Severity::Error, format!("preprocessor: {msg}"))
        .at(Span::line(lineno))
        .in_file(file.to_owned())
}

fn warn_diag(msg: &str, file: &str, lineno: usize) -> Diagnostic {
    Diagnostic::new(
        "W0003",
        Severity::Warning,
        format!("preprocessor: {msg}"),
    )
    .at(Span::line(lineno))
    .in_file(file.to_owned())
}

/// `#macro name(a,b) => expression` → (name, params, expr).
fn parse_macro_def(body: &str) -> Option<(String, Vec<String>, String)> {
    let (name, rest) = body.split_once('(')?;
    let name = name.trim().to_owned();
    let (params_txt, expr) = rest.split_once("=>")?;
    let params_txt = params_txt.trim().strip_suffix(')')?.trim();
    let params: Vec<String> = params_txt
        .split(',')
        .map(|p| p.trim().to_owned())
        .filter(|p| !p.is_empty())
        .collect();
    Some((name, params, expr.trim().to_owned()))
}

/// Expand macro call sites in the final source. Balanced-paren scan per
/// line; up to 8 passes so macros may reference other macros.
pub fn expand_macros(
    mut source: String,
    macros: &[(String, Vec<String>, String)],
) -> String {
    if macros.is_empty() {
        return source;
    }
    for _ in 0..8 {
        let before = source.clone();
        for (name, params, expr) in macros {
            source = expand_call_sites(&source, name, params, expr);
        }
        if source == before {
            break;
        }
    }
    source
}

fn expand_call_sites(
    src: &str,
    name: &str,
    params: &[String],
    replacement: &str,
) -> String {
    let pat = format!("{name}(");
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0usize;
    while i < bytes.len() {
        let starts_word = i == 0
            || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
        if src[i..].starts_with(&pat) && starts_word {
            let open = i + pat.len() - 1; // index of '('
            let mut depth = 1usize;
            let mut j = open + 1;
            while j < bytes.len() && depth > 0 {
                match bytes[j] {
                    b'(' => depth += 1,
                    b')' => depth -= 1,
                    _ => {}
                }
                j += 1;
            }
            if depth != 0 {
                out.push_str(&src[i..i + 1]);
                i += 1;
                continue;
            }
            let inner = &src[open + 1..j - 1];
            let args: Vec<&str> = inner.split(',').map(|a| a.trim()).collect();
            let substituted = substitute(params, &args, replacement);
            out.push('(');
            out.push_str(&substituted);
            out.push(')');
            i = j;
            continue;
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

fn substitute(params: &[String], args: &[&str], body: &str) -> String {
    let mut out = body.to_owned();
    for (idx, p) in params.iter().enumerate() {
        let arg = args.get(idx).copied().unwrap_or("0");
        // Word-boundary replace p with (arg).
        let needle = format!("({p})");
        let pat_len = p.len();
        let b = out.as_bytes();
        let mut res = String::new();
        let mut k = 0usize;
        while k < b.len() {
            let word_start =
                k == 0 || !(b[k - 1].is_ascii_alphanumeric() || b[k - 1] == b'_');
            if b[k..].starts_with(p.as_bytes()) && word_start {
                let after = k + pat_len;
                let word_end = after >= b.len()
                    || !(b[after].is_ascii_alphanumeric() || b[after] == b'_');
                if word_end {
                    res.push('(');
                    res.push_str(arg);
                    res.push(')');
                    k = after;
                    continue;
                }
            }
            res.push(b[k] as char);
            k += 1;
        }
        let _ = needle;
        out = res;
    }
    out
}
