//! Namespace declarations and namespace-level members.

use super::class::ClassDecl;
use super::delegate::DelegateDecl;
use super::interface::InterfaceDecl;
use crate::models::Span;
use crate::models::compare::StructuralEq;

/// A named container of members (`namespace testnc { ... }`).
///
/// Both syntaxes — block form and file-scoped form — normalize to this shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamespaceDecl {
    pub name: String,
    pub members: Vec<Member>,
    pub span: Span,
}

impl StructuralEq for NamespaceDecl {
    fn structural_eq(&self, other: &Self) -> bool {
        self.name == other.name && self.members.structural_eq(&other.members)
    }
}

/// A named enum type with its variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumDecl {
    pub name: String,
    pub underlying_type: String,
    /// (variant name, explicit value or None for auto-assign)
    pub variants: Vec<(String, Option<i64>)>,
    pub span: Span,
}

impl StructuralEq for EnumDecl {
    fn structural_eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.underlying_type == other.underlying_type
            && self.variants == other.variants
    }
}

/// Anything that can appear at namespace or compilation-unit level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Member {
    Class(ClassDecl),
    Interface(InterfaceDecl),
    Enum(EnumDecl),
    Delegate(DelegateDecl),
}

impl Member {
    /// The declared name of this member, regardless of its kind.
    pub fn name(&self) -> &str {
        match self {
            Member::Class(c) => &c.name,
            Member::Interface(i) => &i.name,
            Member::Enum(e) => &e.name,
            Member::Delegate(d) => &d.name,
        }
    }
}

impl StructuralEq for Member {
    fn structural_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Member::Class(a), Member::Class(b)) => a.structural_eq(b),
            (Member::Interface(a), Member::Interface(b)) => a.structural_eq(b),
            (Member::Enum(a), Member::Enum(b)) => a.structural_eq(b),
            (Member::Delegate(a), Member::Delegate(b)) => a.structural_eq(b),
            _ => false,
        }
    }
}
