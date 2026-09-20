//! Structured compiler diagnostics.
//!
//! Every failure carries a stable code, severity, byte-offset span, and the
//! enclosing scope chain (`namespace :: class :: function`) so both the
//! colored text renderer and the JSON output can give IDEs full information.
//! Rendering lives in [`crate::diagnostics`].

use super::span::Span;
use std::fmt;

/// Where in the program structure an error occurred.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ScopeContext {
    pub namespace: Option<String>,
    pub class: Option<String>,
    pub function: Option<String>,
}

impl ScopeContext {
    /// `demo::app::main`, omitting absent levels.
    pub fn path(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(ns) = &self.namespace {
            parts.push(ns.clone());
        }
        if let Some(class) = &self.class {
            parts.push(class.clone());
        }
        if let Some(function) = &self.function {
            parts.push(function.clone());
        }
        parts.join("::")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Error => write!(f, "{}", crate::t!("rubyc.models.diagnostic.Severity.error")),
            Severity::Warning => {
                write!(
                    f,
                    "{}",
                    crate::t!("rubyc.models.diagnostic.Severity.warning")
                )
            }
        }
    }
}

/// A structured compiler diagnostic.
///
/// Kept under the historic alias [`SyntaxError`] so call sites read naturally.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub code: &'static str,
    pub severity: Severity,
    pub message: String,
    /// Source file, set by `parse_file` / the CLI input layer.
    pub file: Option<String>,
    pub span: Span,
    pub context: ScopeContext,
    /// Suggested fix, rendered rustc-style as a `help:` line.
    pub help: Option<String>,
}

/// Historic alias: every parse failure is a Diagnostic today.
pub type SyntaxError = Diagnostic;

impl Diagnostic {
    pub fn new(code: &'static str, severity: Severity, message: impl Into<String>) -> Self {
        Self {
            code,
            severity,
            message: message.into(),
            file: None,
            span: Span::default(),
            context: ScopeContext::default(),
            help: None,
        }
    }

    /// Attach a suggested fix (rendered as a `help:` line).
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    pub fn at(mut self, span: Span) -> Self {
        self.span = span;
        self
    }

    pub fn in_file(mut self, file: impl Into<String>) -> Self {
        self.file = Some(file.into());
        self
    }

    pub fn in_context(mut self, context: ScopeContext) -> Self {
        self.context = context;
        self
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{}]: {}", self.severity, self.code, self.message)?;
        if !self.context.path().is_empty() {
            write!(f, " in {}", self.context.path())?;
        }
        Ok(())
    }
}

/// Failure modes when reading a RubyC bytecode payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    BadMagic,
    WrongKind,
    UnsupportedVersion(u16),
    UnexpectedEof,
    UnknownNodeTag(u8),
    InvalidUtf8,
    TrailingBytes,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::BadMagic => {
                write!(
                    f,
                    "{}",
                    crate::t!("rubyc.models.diagnostic.DecodeError.bad_magic")
                )
            }
            DecodeError::WrongKind => {
                write!(
                    f,
                    "{}",
                    crate::t!("rubyc.models.diagnostic.DecodeError.wrong_kind")
                )
            }
            DecodeError::UnsupportedVersion(v) => {
                write!(
                    f,
                    "{}",
                    crate::tf!("rubyc.models.diagnostic.DecodeError.unsupported_version", v)
                )
            }
            DecodeError::UnexpectedEof => {
                write!(
                    f,
                    "{}",
                    crate::t!("rubyc.models.diagnostic.DecodeError.unexpected_eof")
                )
            }
            DecodeError::UnknownNodeTag(tag) => {
                let tag = format!("0x{tag:02x}");
                write!(
                    f,
                    "{}",
                    crate::tf!("rubyc.models.diagnostic.DecodeError.unknown_node_tag", tag)
                )
            }
            DecodeError::InvalidUtf8 => {
                write!(
                    f,
                    "{}",
                    crate::t!("rubyc.models.diagnostic.DecodeError.invalid_utf8")
                )
            }
            DecodeError::TrailingBytes => {
                write!(
                    f,
                    "{}",
                    crate::t!("rubyc.models.diagnostic.DecodeError.trailing_bytes")
                )
            }
        }
    }
}

impl std::error::Error for DecodeError {}
