//! Frontend pipeline: source text → tokens → [`CompilationUnit`].

pub mod lexer;
pub mod assembly;
pub mod rd;

pub use rd::{parse, parse_lenient};

use crate::models::{CompilationUnit, Diagnostic, ScopeContext, Severity, Span, SyntaxError};
use std::path::Path;

/// Convenience wrapper that reads a source file from disk.
pub fn parse_file(path: &Path) -> Result<CompilationUnit, Vec<SyntaxError>> {
    let src = std::fs::read_to_string(path).map_err(|e| {
        vec![Diagnostic {
            code: "E0101",
            severity: Severity::Error,
            message: format!("cannot read {path:?}: {e}"),
            file: path.to_str().map(|s| s.to_owned()),
            span: Span::default(),
            context: ScopeContext::default(),

            help: None,
        }]
    })?;
    parse(&src)
}
