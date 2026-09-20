//! Delegate declarations: named callable types.

use super::method::Param;
use crate::models::Span;
use crate::models::compare::StructuralEq;

/// A named callable type (`delegate int Func(int x);`).
///
/// Like methods, the return type is a plain name until a richer type tree
/// lands; generic parameters are kept as names for symmetry with the
/// class/interface generic handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelegateDecl {
    pub access: super::statement::Access,
    pub ret_type: String,
    pub name: String,
    pub generic_params: Vec<String>,
    pub params: Vec<Param>,
    pub span: Span,
}

impl StructuralEq for DelegateDecl {
    fn structural_eq(&self, other: &Self) -> bool {
        self.access == other.access
            && self.ret_type == other.ret_type
            && self.name == other.name
            && self.generic_params == other.generic_params
            && self.params == other.params
    }
}
