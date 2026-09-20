//! Method declarations, parameters and statement blocks.

use super::attribute::Attribute;
use super::expression::Expr;
use super::statement::{Access, Stmt};
use crate::models::Span;
use crate::models::compare::StructuralEq;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MethodDecl {
    pub access: Access,
    /// Compiler-facing annotations (`[LibraryImport(...)]`).
    pub attrs: Vec<Attribute>,
    /// Generic type parameter names (`M<TKey>`); empty for non-generic.
    /// Needed so later passes (scan, lowering) resolve `T` in signatures
    /// and bodies — mirroring `DelegateDecl.generic_params`.
    pub generic_params: Vec<String>,
    /// Name of the return type; a richer type tree once the type system grows.
    pub ret_type: String,
    pub name: String,
    /// Parameter list; empty for methods declared with `()`.
    pub params: Vec<Param>,
    /// `None` for bodyless declarations — import stubs or errors depending
    /// on attributes.
    pub body: Option<Block>,
    pub span: Span,
    pub is_override: bool,
    /// `static` modifier — method is called on the type, not an instance.
    pub is_static: bool,
    /// `abstract` modifier — bodyless methods are valid in abstract classes.
    pub is_abstract: bool,
    /// `partial` modifier — bodyless methods are valid (may be defined in
    /// another partial declaration).
    pub is_partial: bool,
    /// `extern` modifier — bodyless by definition; the implementation is
    /// provided by the runtime (`[MethodImpl(InternalCall)]`, QCall) or
    /// linked native code. Lowering emits a trap stub; never a body.
    pub is_extern: bool,
    /// `async` modifier — the method body may contain `await` expressions.
    /// At runtime, the method runs as a cooperative coroutine managed by the
    /// scheduler; each `await` is a suspension point.
    pub is_async: bool,
    /// Constructor chain: `Some(("base", args))` or `Some(("this", args))`.
    pub base_call: Option<(String, Vec<Expr>)>,
}

impl StructuralEq for MethodDecl {
    fn structural_eq(&self, other: &Self) -> bool {
        self.ret_type == other.ret_type
            && self.name == other.name
            && self.generic_params == other.generic_params
            && self.params.len() == other.params.len()
            && self
                .params
                .iter()
                .zip(other.params.iter())
                .all(|(a, b)| a.structural_eq(b))
            && match (&self.body, &other.body) {
                (Some(a), Some(b)) => a.structural_eq(b),
                (None, None) => true,
                _ => false,
            }
            && self.attrs.len() == other.attrs.len()
            && self
                .attrs
                .iter()
                .zip(other.attrs.iter())
                .all(|(a, b)| a.structural_eq(b))
            && match (&self.base_call, &other.base_call) {
                (None, None) => true,
                (Some((a, aa)), Some((b, bb))) => a == b && aa.len() == bb.len() && aa.iter().zip(bb.iter()).all(|(x, y)| x.structural_eq(y)),
                _ => false,
            }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    pub type_name: String,
    pub name: String,
    pub span: Span,
    pub default_value: Option<Expr>,
    /// Extension-method receiver (`this Span<T> span` on the first
    /// parameter). Recorded by the parser; lowering rewrites
    /// `recv.Ext(args)` to the static `Class.Ext(recv, args)`.
    pub is_this: bool,
    /// `params` array parameter (`params TableIndex[] tables`): the
    /// call supplies zero or more trailing args of the element type
    /// (C# expanded form). Always last by language rules.
    pub is_params: bool,
}

impl StructuralEq for Param {
    fn structural_eq(&self, other: &Self) -> bool {
        self.type_name == other.type_name
            && self.name == other.name
            && self.is_this == other.is_this
            && self.is_params == other.is_params
            && match (&self.default_value, &other.default_value) {
                (Some(a), Some(b)) => a.structural_eq(b),
                (None, None) => true,
                _ => false,
            }
    }
}

/// One `case …:` arm of a switch. Multiple labels share one body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwitchArm {
    /// Constant integer label expressions (patterns arrive in A11).
    pub labels: Vec<Expr>,
    pub body: Block,
}

/// A braced sequence of statements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub span: Span,
}

impl StructuralEq for Block {
    fn structural_eq(&self, other: &Self) -> bool {
        self.stmts.structural_eq(&other.stmts)
    }
}
