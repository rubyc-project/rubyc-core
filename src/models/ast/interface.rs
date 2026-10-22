//! Interfaces: signatures (optionally with default bodies).

use super::method::Block;
use super::method::Param;
use crate::models::Span;
use crate::models::compare::StructuralEq;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterfaceDecl {
    pub name: String,
    /// Generic type parameter names (`interface Foo<T>`); empty for
    /// non-generic. Needed so later passes resolve `T` in signatures
    /// and default bodies — mirroring `DelegateDecl.generic_params`.
    /// (`arity` below stays in sync: always `generic_params.len()`.)
    pub generic_params: Vec<String>,
    pub methods: Vec<MethodSig>,
    /// Indexers declared by the interface (`TElement this[int i]`).
    /// A member or indexer on a type-parameter receiver binds through
    /// the constraint, so an interface's indexers must be recorded —
    /// they were discarded, leaving `x[i]` untyped.
    pub indexers: Vec<super::class::IndexerDecl>,
    /// Inherited interfaces (`interface A : B, C`): member lookup
    /// walks these transitively (`IAsyncEnumerator<T>` inherits
    /// `DisposeAsync` from `IAsyncDisposable`).
    pub bases: Vec<String>,
    /// Number of generic type parameters (0 for non-generic). Used to
    /// distinguish `IFoo` from `IFoo<T>` (valid C# overloading by arity).
    pub arity: usize,
    /// `partial` modifier — multiple declarations with the same name
    /// are merged into a single interface (mirrors `ClassDecl`).
    pub is_partial: bool,
    pub span: Span,
}

impl StructuralEq for InterfaceDecl {
    fn structural_eq(&self, other: &Self) -> bool {
        // `arity` is excluded: it is derived from the source's generic
        // parameter list and is not round-tripped through the compact
        // bytecode model (decode always sets it to 0). `is_partial`
        // likewise (decode sets false; merge-irrelevant post-parse).
        self.name == other.name
            && self.bases == other.bases
            && self.methods.len() == other.methods.len()
            && self
                .methods
                .iter()
                .zip(&other.methods)
                .all(|(a, b)| a.structural_eq(b))
    }
}

/// A method signature shared by interfaces and classes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MethodSig {
    pub access: super::statement::Access,
    pub ret_type: String,
    pub name: String,
    /// Generic type parameter names (`M<TKey>`);
    /// empty for non-generic.
    pub generic_params: Vec<String>,
    pub params: Vec<Param>,
    /// Default implementation body (interfaces may provide one).
    pub body: Option<Block>,
    /// `static` modifier (C# 8 static interface members dispatch
    /// directly, not virtually).
    pub is_static: bool,
    pub span: Span,
}

impl StructuralEq for MethodSig {
    fn structural_eq(&self, other: &Self) -> bool {
        self.access == other.access
            && self.ret_type == other.ret_type
            && self.name == other.name
            && self.generic_params == other.generic_params
            && self.is_static == other.is_static
            && self.params == other.params
    }
}
