//! Class declarations and their members.

use super::attribute::Attribute;
use super::expression::Expr;
use super::method::{Block, MethodDecl, Param};
use super::namespace::Member;
use super::statement::{Access, Stmt};
use crate::models::Span;
use crate::models::compare::StructuralEq;

/// A declared class field with its initializer and access modifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldDecl {
    pub access: Access,
    pub attrs: Vec<Attribute>,
    pub type_name: String,
    pub name: String,
    pub init: Expr,
    /// Bindings declared while parsing `init` (`out` vars, `is`
    /// patterns): no statement list exists in a field declaration to
    /// hold them, so they ride along here and lower first, in order.
    /// Empty in the common case.
    pub init_pre: Vec<Stmt>,
    /// `const` modifier: the field is a compile-time constant and occupies no
    /// object-layout slot.
    pub is_const: bool,
    /// `static` modifier: the field lives in per-class static storage, not in
    /// the object layout. It is accessed through the type name and is
    /// initialized once at program start.
    pub is_static: bool,
    /// `ref` modifier (`ref T _field` in a ref struct): the slot
    /// holds an address, not a value. Address-taking (`ref field`)
    /// collapses to the target (no extra indirection); value reads
    /// see the address. Without this, byref field addresses resolve
    /// to the slot instead of the target (e.g. `Span._reference`
    /// copies slot bytes instead of buffer data in `Memmove`).
    pub is_byref: bool,
    pub span: Span,
}

/// A class property: an accessor-backed member (`{ get; set; }`, explicit
/// getter/setter bodies, or an expression-bodied `=>` getter).
///
/// `get`/`set` are `None` for auto-properties (only the accessor flags are
/// set). `get_expr` is `Some` only for expression-bodied properties
/// (`int X => expr;`); in that case `get` is `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropertyDecl {
    pub access: Access,
    pub attrs: Vec<Attribute>,
    pub type_name: String,
    pub name: String,
    /// Initializer (`= expr`); `Int(0)` when none is written.
    pub init: Expr,
    /// Explicit getter body; `None` for auto-properties and expression-bodied
    /// properties.
    pub get: Option<Block>,
    /// Explicit setter body; `None` for getter-only and auto set-less props.
    pub set: Option<Block>,
    /// Expression-bodied getter (`int X => expr;`); `None` otherwise.
    pub get_expr: Option<Expr>,
    /// `init` accessor present (`{ get; init; }`).
    pub is_init: bool,
    /// `set` accessor present (`{ get; set; }`); auto-property write flag.
    pub is_settable: bool,
    /// `static` modifier: the property lives in per-class static storage
    /// (accessed through the type name), mirroring FieldDecl.
    pub is_static: bool,
    /// `abstract` modifier: body-less accessors declare no storage —
    /// without this flag an abstract property is indistinguishable
    /// from an auto-property and wrongly gains a backing field
    /// (shifting every later instance offset; e.g. Stream's
    /// `CanRead`/`CanWrite` pushed MemoryStream's `_buffer` out of
    /// reach and faulted `WriteByte`).
    pub is_abstract: bool,
    /// `override` modifier: this declaration overrides a virtual base
    /// member. Only an override takes part in virtual dispatch; a
    /// declaration that merely repeats the name HIDES the base member,
    /// and the base's implementation serves a base-typed receiver.
    pub is_override: bool,
    /// `virtual` modifier: the member may be overridden.
    pub is_virtual: bool,
    /// `new` modifier: the declaration hides a base member.
    pub is_hiding: bool,
    pub span: Span,
}

/// An indexer: `public int this[int index] { get; set; }` or
/// `public int this[int index] => expr;`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexerDecl {
    pub access: Access,
    pub attrs: Vec<Attribute>,
    pub type_name: String,
    /// Indexer parameters (e.g. `int index`).
    pub params: Vec<Param>,
    /// Initializer; `Int(0)` when none is written.
    pub init: Expr,
    /// Explicit getter body; `None` for expression-bodied indexers.
    pub get: Option<Block>,
    /// Explicit setter body; `None` for getter-only indexers.
    pub set: Option<Block>,
    /// Expression-bodied getter (`int this[int i] => expr;`); `None` otherwise.
    pub get_expr: Option<Expr>,
    pub is_settable: bool,
    pub span: Span,
}

/// An event: `public event IntUnary? Changed;`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventDecl {
    pub access: Access,
    pub attrs: Vec<Attribute>,
    pub type_name: String,
    pub name: String,
    pub span: Span,
}

/// A type declaration (`class testc : Base1, IFace { ... }`).
///
/// RubyC allows multiple inheritance of classes AND interfaces — this is a
/// deliberate divergence from C#.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassDecl {
    pub name: String,
    /// Generic type parameter names (`class Foo<T>`); empty for
    /// non-generic. Needed so later passes resolve `T` in member
    /// signatures and bodies — mirroring `DelegateDecl.generic_params`.
    pub generic_params: Vec<String>,
    /// Primary-constructor value parameter names (`class C(int x)`,
    /// `struct S(nint addr, ...)`); empty when absent. In scope for
    /// the whole body, mirroring .NET — same rationale as
    /// `generic_params`, names only.
    pub primary_ctor_params: Vec<String>,
    /// Base classes / interfaces in declaration order.
    pub bases: Vec<String>,
    pub members: Vec<ClassMember>,
    pub span: Span,
    /// `partial` modifier — multiple declarations with the same name are
    /// merged into a single class.
    pub is_partial: bool,
    /// `struct` (vs `class`/`record class`) declaration. Structs keep
    /// their implicit parameterless constructor even with explicit
    /// ctors; classes do not. Parsed uniformly as classes otherwise.
    pub is_struct: bool,
    /// `record` declaration. A record is a class with VALUE semantics:
    /// `==` compares by value, not by reference. The keyword used to be
    /// discarded here, so nothing downstream could tell a record from a
    /// plain class and `new R(1) == new R(1)` was false.
    pub is_record: bool,
    /// Cross-assembly union provenance (set by the driver's dep-member
    /// union, never by the parser): this declaration absorbed members
    /// from another assembly's same-path class. Same-assembly partial
    /// parts never carry it. Semantic uses it to tell a genuine
    /// multi-part partial (merge silently) from a cross-assembly
    /// collision (merge, but preserve each side's constructibility).
    /// Excluded from structural equality and the v1 AST byte contract
    /// like `is_partial`.
    pub union_cross_assembly: bool,
}

impl StructuralEq for ClassDecl {
    fn structural_eq(&self, other: &Self) -> bool {
        // Member-shape equality only (`is_partial`/`is_struct` are
        // declaration kinds, excluded like `is_partial` — and neither
        // travels the v1 AST byte contract).
        self.name == other.name
            && self.generic_params == other.generic_params
            && self.primary_ctor_params == other.primary_ctor_params
            && self.members.structural_eq(&other.members)
    }
}

/// Anything that can appear inside a class body.
///
/// New member kinds (fields, constructors, nested types) are added as
/// variants here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClassMember {
    Method(MethodDecl),
    Field(FieldDecl),
    Property(PropertyDecl),
    Indexer(IndexerDecl),
    Event(EventDecl),
    Nested(Member),
}

impl StructuralEq for ClassMember {
    fn structural_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ClassMember::Method(a), ClassMember::Method(b)) => a.structural_eq(b),
            (ClassMember::Field(a), ClassMember::Field(b)) => {
                a.access == b.access
                    && a.type_name == b.type_name
                    && a.name == b.name
                    && a.is_const == b.is_const
                    && a.is_static == b.is_static
                    && a.init.structural_eq(&b.init)
            }
            (ClassMember::Property(a), ClassMember::Property(b)) => {
                a.access == b.access
                    && a.type_name == b.type_name
                    && a.name == b.name
                    && a.is_init == b.is_init
                    && a.is_settable == b.is_settable
                    && a.init.structural_eq(&b.init)
                    && match (&a.get, &b.get) {
                        (Some(x), Some(y)) => x.structural_eq(y),
                        (None, None) => true,
                        _ => false,
                    }
                    && match (&a.set, &b.set) {
                        (Some(x), Some(y)) => x.structural_eq(y),
                        (None, None) => true,
                        _ => false,
                    }
                    && match (&a.get_expr, &b.get_expr) {
                        (Some(x), Some(y)) => x.structural_eq(y),
                        (None, None) => true,
                        _ => false,
                    }
            }
            (ClassMember::Indexer(a), ClassMember::Indexer(b)) => {
                a.access == b.access
                    && a.type_name == b.type_name
                    && a.params.len() == b.params.len()
                    && a
                        .params
                        .iter()
                        .zip(b.params.iter())
                        .all(|(x, y)| x.type_name == y.type_name && x.name == y.name)
                    && a.is_settable == b.is_settable
                    && a.init.structural_eq(&b.init)
                    && match (&a.get, &b.get) {
                        (Some(x), Some(y)) => x.structural_eq(y),
                        (None, None) => true,
                        _ => false,
                    }
                    && match (&a.set, &b.set) {
                        (Some(x), Some(y)) => x.structural_eq(y),
                        (None, None) => true,
                        _ => false,
                    }
                    && match (&a.get_expr, &b.get_expr) {
                        (Some(x), Some(y)) => x.structural_eq(y),
                        (None, None) => true,
                        _ => false,
                    }
            }
            (ClassMember::Event(a), ClassMember::Event(b)) => {
                a.access == b.access && a.type_name == b.type_name && a.name == b.name
            }
            (ClassMember::Nested(a), ClassMember::Nested(b)) => a.structural_eq(b),
            _ => false,
        }
    }
}
