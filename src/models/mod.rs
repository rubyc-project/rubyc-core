//! All plain data models of the compiler.
//!
//! Models are pure data: no logic beyond small constructors and helpers.
//! Behavior lives in the sibling modules (`parser`, `bytecode`, `traits`).

pub mod ast;
pub mod compare;
pub mod diagnostic;
pub mod span;
pub mod symbols;
pub mod token;
pub mod visitor;

pub use ast::{
    Access, Attribute, AttributeArg, AttributeValue, BinOp, Block, CatchClause, ClassDecl,
    ClassMember, CompilationUnit, DelegateDecl, Expr, FieldDecl, InterfaceDecl, Literal, Member,
    MemberInit, MethodDecl, MethodSig, NamespaceDecl, Param, PropertyDecl, Stmt, EnumDecl, InterpPart,
    SwitchArm, SwitchExprArm, UnOp, EventDecl, IndexerDecl, UsingAlias,
};
pub use diagnostic::{DecodeError, Diagnostic, ScopeContext, Severity, SyntaxError};
pub use span::Span;
pub use token::{Keyword, Token, TokenKind};
