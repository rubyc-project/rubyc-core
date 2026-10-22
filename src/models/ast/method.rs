//! Method declarations, parameters and statement blocks.

use super::attribute::Attribute;
use super::expression::Expr;
use super::statement::{Access, Stmt};
use crate::models::Span;
use crate::models::compare::StructuralEq;

/// One `where T : IFoo<T, int>` clause. A type parameter's members
/// resolve through its constraint, so the clauses are recorded rather
/// than discarded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeConstraint {
    /// The constrained type parameter.
    pub param: String,
    /// The constraint types (an interface or class per clause).
    pub types: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MethodDecl {
    pub access: Access,
    /// Compiler-facing annotations (`[LibraryImport(...)]`).
    pub attrs: Vec<Attribute>,
    /// Generic type parameter names (`M<TKey>`); empty for non-generic.
    /// Needed so later passes (scan, lowering) resolve `T` in signatures
    /// and bodies — mirroring `DelegateDecl.generic_params`.
    pub generic_params: Vec<String>,
    /// `where T : ...` clauses. A member or indexer on a type-parameter
    /// receiver binds through them (C#: `x[i]` uses the constraint's
    /// indexer), so they are recorded.
    pub constraints: Vec<TypeConstraint>,
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
    /// `virtual` modifier — the call binds to the RECEIVER's runtime
    /// type, not the static one.
    ///
    /// Recorded, not inferred: a method is dispatch-virtual only when it
    /// says so (or overrides one that did). Treating every call as
    /// virtual makes `new`-hiding indistinguishable from overriding —
    /// `((B)d).M()` runs `D.M` when C# calls `B.M`.
    pub is_virtual: bool,
    /// `sealed` modifier — the method may not be overridden, so its
    /// dispatch target is fixed at the declaring type.
    pub is_sealed: bool,
    /// `new` modifier — this declaration HIDES an inherited member of
    /// the same name instead of overriding it. A call through the base
    /// type must reach the base member.
    pub is_hiding: bool,
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
    /// By-reference parameter (`ref`/`out`/`in`): lowered by address
    /// (callers box the value, callees dereference) so writes flow
    /// back. Without this, `out` results (TryParse, GetBytesFast's
    /// `charsConsumed`, …) are silently lost.
    pub is_byref: bool,
    /// Attributes on the parameter (`[UnsafeAccessorType("...")]`).
    pub attrs: Vec<super::attribute::Attribute>,
}

impl StructuralEq for TypeConstraint {
    fn structural_eq(&self, other: &Self) -> bool {
        self.param == other.param && self.types == other.types
    }
}

impl StructuralEq for Param {
    fn structural_eq(&self, other: &Self) -> bool {
        self.type_name == other.type_name
            && self.name == other.name
            && self.is_this == other.is_this
            && self.is_params == other.is_params
            && self.is_byref == other.is_byref
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
    /// `case X when cond:` — one optional guard per label, parallel to
    /// `labels`. A false guard means the label does NOT select the arm;
    /// dropping it made `case 1 when false:` match anyway.
    pub guards: Vec<Option<Expr>>,
    /// `case RuntimeMethodBuilder mb:` — the designation bound to the
    /// switch SUBJECT when this arm is selected, with its declared type.
    /// Held on the arm (not the label) because the lowering enters one
    /// block per arm and C# requires a section's labels to bind the same
    /// variables.
    pub binding: Option<(String, String)>,
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
