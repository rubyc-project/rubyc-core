//! Attribute (annotation) nodes: `[Name("arg", Key = value)]` metadata
//! attached to declarations.
//!
//! Attributes are compiler-facing data — the language has no runtime
//! reflection, so every attribute either drives codegen (like
//! `LibraryImport`) or is validated and ignored by a later pass.

use crate::models::Span;
use crate::models::compare::StructuralEq;

/// A single attribute occurrence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attribute {
    pub name: String,
    pub arguments: Vec<AttributeArg>,
    pub span: Span,
}

/// One argument: positional (`"mylib"`) or named (`EntryPoint = "add"`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttributeArg {
    /// `None` for positional arguments.
    pub name: Option<String>,
    pub value: AttributeValue,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttributeValue {
    Str(String),
    Int(i64),
    /// Float literal bits (`DefaultValue(100d)` in Timer sources).
    /// Stored as bits to keep `Eq` (attributes compare structurally).
    Float(u64),
}

impl StructuralEq for Attribute {
    fn structural_eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.arguments.len() == other.arguments.len()
            && self
                .arguments
                .iter()
                .zip(other.arguments.iter())
                .all(|(a, b)| a.structural_eq(b))
    }
}

impl StructuralEq for AttributeArg {
    fn structural_eq(&self, other: &Self) -> bool {
        self.name == other.name && self.value == other.value
    }
}
