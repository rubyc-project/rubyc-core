//! A whole parsed source file.

use super::namespace::{Member, NamespaceDecl};
use crate::models::Span;
use crate::models::compare::StructuralEq;

/// A `using Alias = Qualified.Name;` directive. Recorded (not discarded)
/// so later passes resolve the alias: the scan treats the name as a
/// value reference, and the type table resolves it to its target —
/// mirroring .NET, where the alias is another name for the target type.
/// Stored flat on the unit (file boundaries are already merged away for
/// types and members); `using namespace` / `using static` directives
/// carry no name binding and are still discarded.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UsingAlias {
    /// The introduced name (`NativeType` in `using NativeType = ...`).
    pub name: String,
    /// The target as written (`System.Int32`; qualification included).
    pub target: String,
}

/// The root AST node produced by the parser.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CompilationUnit {
    /// Every namespace declared in the file (block or file-scoped form).
    /// A file may declare zero, one, or many namespaces.
    pub namespaces: Vec<NamespaceDecl>,
    /// Top-level members declared outside any namespace.
    pub members: Vec<Member>,
    /// `using Alias = ...` directives in source order (top-level and
    /// namespace-level alike — lookup here is flat, like types).
    pub aliases: Vec<UsingAlias>,
    /// `using static Qualified.Type;` directives (targets as written,
    /// e.g. `System.Reflection.MethodBase`). Their static members are
    /// in scope unqualified — mirroring .NET. Stored flat like
    /// aliases; file boundaries are already merged away.
    pub static_usings: Vec<String>,
    /// Native assembly blocks captured by the source preprocessor.
    pub assembly_blocks: Vec<String>,
    pub span: Span,
}

impl CompilationUnit {
    /// All members across every namespace plus top-level members, in source order.
    pub fn all_members(&self) -> Vec<&Member> {
        let mut out: Vec<&Member> = Vec::new();
        for ns in &self.namespaces {
            out.extend(ns.members.iter());
        }
        out.extend(self.members.iter());
        out
    }

    /// The name of the namespace that directly contains a member with the
    /// given name, if any.
    pub fn namespace_name_of(&self, member_name: &str) -> Option<String> {
        self.namespaces
            .iter()
            .find(|ns| ns.members.iter().any(|m| m.name() == member_name))
            .map(|ns| ns.name.clone())
    }
}

impl StructuralEq for CompilationUnit {
    fn structural_eq(&self, other: &Self) -> bool {
        self.namespaces.len() == other.namespaces.len()
            && self
                .namespaces
                .iter()
                .zip(other.namespaces.iter())
                .all(|(a, b)| a.structural_eq(b))
            && self.members.structural_eq(&other.members)
            && self.aliases == other.aliases
            && self.static_usings == other.static_usings
            && self.assembly_blocks == other.assembly_blocks
    }
}
