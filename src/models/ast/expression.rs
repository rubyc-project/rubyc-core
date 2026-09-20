//! Expression nodes.

use crate::models::Span;
use crate::models::compare::StructuralEq;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    ShrShr,
    Is,
    /// `??` and `??=` — C# null-coalescing.
    NullCoalesce,
}

impl BinOp {
    /// Token text for diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Lt => "<",
            BinOp::Gt => ">",
            BinOp::Le => "<=",
            BinOp::Ge => ">=",
            BinOp::And => "&&",
            BinOp::Or => "||",
            BinOp::BitAnd => "&",
            BinOp::BitOr => "|",
            BinOp::BitXor => "^",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
            BinOp::ShrShr => ">>>",
            BinOp::Is => "is",
            BinOp::NullCoalesce => "??",
        }
    }

    /// True for operators producing a `bool` result.
    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
    /// `~x` — bitwise complement.
    BitNot,
    /// `*x` — dereference.
    Deref,
    /// `&x` — address-of.
    AddrOf,
    /// `throw expr` — throw expression.
    Throw,
    /// `^x` — C# 8 Index from-end operator.
    IndexFromEnd,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        span: Span,
    },
    Unary {
        op: UnOp,
        inner: Box<Expr>,
        span: Span,
    },
    /// `cond ? then_expr : else_expr`
    Ternary {
        cond: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
        span: Span,
    },
    Call {
        callee: String,
        args: Vec<Expr>,
        span: Span,
    },
    /// `new ClassName(args...)`
    New {
        class_name: String,
        args: Vec<Expr>,
        span: Span,
    },
    /// `object.member` — field read or method call receiver.
    Member {
        object: Box<Expr>,
        name: String,
        span: Span,
    },
    /// `object.method(args)` — pairs a receiver with a method call.
    MethodCall {
        object: Box<Expr>,
        name: String,
        args: Vec<Expr>,
        span: Span,
    },
    Literal(Literal),
    /// `await expr` — suspension point inside an `async` method. The
    /// expression is evaluated, and if it produces a pending task the
    /// coroutine yields until the task completes.
    Await {
        inner: Box<Expr>,
        span: Span,
    },
    /// `new T[n]` — single-dim array allocation.
    NewArray { elem_type: String, size: Box<Expr>, span: Span },
    /// `arr[i]` indexing. `is_range` marks C# 8 range slices
    /// (`a[..n]`, `a[i..j]`): a slice has the CONTAINER's type
    /// (`Span<T>`/`T[]`), not the element type — without the flag the
    /// range parses as a flat index and method chains on the slice
    /// (`span[..n].ToArray()`) mis-resolve against the element type.
    Index { object: Box<Expr>, index: Box<Expr>, is_range: bool, span: Span },
    /// Variable / bare-name reference. Spans enable unknown-name carets.
    Ident {
        name: String,
        span: Span,
    },
    /// `(Type)expr` cast.
    Cast {
        type_name: String,
        expr: Box<Expr>,
        span: Span,
    },
    /// `default(T)` — zero value for type T.
    Default {
        type_name: String,
        span: Span,
    },
    /// `lhs op= rhs` — compound assignment with a complex LHS.
    AssignComplex {
        lhs: Box<Expr>,
        value: Box<Expr>,
        span: Span,
    },
    /// `target = value` — simple assignment expression.
    Assign {
        target: Box<Expr>,
        value: Box<Expr>,
        span: Span,
    },
    /// `target++` — postfix increment: yields the OLD value, stores old+1.
    PostInc { target: Box<Expr>, span: Span },
    /// `target--` — postfix decrement: yields the OLD value, stores old-1.
    PostDec { target: Box<Expr>, span: Span },
    /// `subject switch { arm1, arm2, ..., _ => default }` — C# 8+ switch
    /// expression. Each arm has one or more patterns and a single expression
    /// body separated by `=>`. The discard pattern (`_`) marks the default
    /// arm. Patterns are recorded as opaque strings because the structural
    /// detail of each pattern (relational, type, property, ...) is not yet
    /// represented in the AST.
    SwitchExpr {
        subject: Box<Expr>,
        arms: Vec<SwitchExprArm>,
        span: Span,
    },
    /// `$"..."` / `$@"..."` / `$"""..."""` — an interpolated string. The
    /// body is split into literal text runs and interpolation holes; each
    /// hole is a full expression (so holes may themselves interpolate).
    InterpolatedString {
        parts: Vec<InterpPart>,
        span: Span,
    },
}

/// One part of an interpolated string body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterpPart {
    /// A literal text run (no escapes).
    Text(String),
    /// An interpolation hole `{ ... }`.
    Hole(Box<Expr>),
}

impl StructuralEq for InterpPart {
    fn structural_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (InterpPart::Text(a), InterpPart::Text(b)) => a == b,
            (InterpPart::Hole(a), InterpPart::Hole(b)) => a.structural_eq(b),
            _ => false,
        }
    }
}

/// One arm of a switch expression: one or more pattern labels and a body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwitchExprArm {
    /// Pattern labels (joined by `,`); the discard pattern lives in the same
    /// field as a single `_` entry so the default arm is the last one.
    pub labels: Vec<String>,
    pub body: Box<Expr>,
}

impl StructuralEq for Expr {
    fn structural_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Expr::Binary {
                    op: a,
                    lhs: al,
                    rhs: ar,
                    ..
                },
                Expr::Binary {
                    op: b,
                    lhs: bl,
                    rhs: br,
                    ..
                },
            ) => a == b && al.structural_eq(bl) && ar.structural_eq(br),
            (
                Expr::Unary {
                    op: a, inner: ai, ..
                },
                Expr::Unary {
                    op: b, inner: bi, ..
                },
            ) => a == b && ai.structural_eq(bi),
            (
                Expr::Ternary {
                    cond: ac,
                    then_expr: at,
                    else_expr: ae,
                    ..
                },
                Expr::Ternary {
                    cond: bc,
                    then_expr: bt,
                    else_expr: be,
                    ..
                },
            ) => ac.structural_eq(bc) && at.structural_eq(bt) && ae.structural_eq(be),
            (
                Expr::Call {
                    callee: a, args: x, ..
                },
                Expr::Call {
                    callee: b, args: y, ..
                },
            ) => a == b && x.structural_eq(y),
            (
                Expr::Member {
                    object: ao,
                    name: an,
                    ..
                },
                Expr::Member {
                    object: bo,
                    name: bn,
                    ..
                },
            ) => an == bn && ao.structural_eq(bo),
            (
                Expr::MethodCall {
                    object: ao,
                    name: an,
                    args: xa,
                    ..
                },
                Expr::MethodCall {
                    object: bo,
                    name: bn,
                    args: xb,
                    ..
                },
            ) => an == bn && ao.structural_eq(bo) && xa.structural_eq(xb),
            (
                Expr::New {
                    class_name: a,
                    args: aa,
                    ..
                },
                Expr::New {
                    class_name: b,
                    args: bb,
                    ..
                },
            ) => {
                a == b
                    && aa.len() == bb.len()
                    && aa.iter().zip(bb.iter()).all(|(x, y)| x.structural_eq(y))
            }
            (Expr::Literal(a), Expr::Literal(b)) => a == b,
            (Expr::Ident { name: a, .. }, Expr::Ident { name: b, .. }) => a == b,
            (
                Expr::Cast {
                    type_name: a,
                    expr: ae,
                    ..
                },
                Expr::Cast {
                    type_name: b,
                    expr: be,
                    ..
                },
            ) => a == b && ae.structural_eq(be),
            (
                Expr::Default {
                    type_name: a,
                    ..
                },
                Expr::Default {
                    type_name: b,
                    ..
                },
            ) => a == b,
            (
                Expr::AssignComplex {
                    lhs: al,
                    value: av,
                    ..
                },
                Expr::AssignComplex {
                    lhs: bl,
                    value: bv,
                    ..
                },
            ) => al.structural_eq(bl) && av.structural_eq(bv),
            (
                Expr::Assign {
                    target: at,
                    value: av,
                    ..
                },
                Expr::Assign {
                    target: bt,
                    value: bv,
                    ..
                },
            ) => at.structural_eq(bt) && av.structural_eq(bv),
            (
                Expr::SwitchExpr {
                    subject: as_,
                    arms: aa,
                    ..
                },
                Expr::SwitchExpr {
                    subject: bs,
                    arms: ba,
                    ..
                },
            ) => {
                as_.structural_eq(bs)
                    && aa.len() == ba.len()
                    && aa.iter().zip(ba.iter()).all(|(x, y)| {
                        x.labels == y.labels && x.body.structural_eq(&y.body)
                    })
            }
            (
                Expr::Await {
                    inner: ai,
                    ..
                },
                Expr::Await {
                    inner: bi,
                    ..
                },
            ) => ai.structural_eq(bi),
            (
                Expr::NewArray {
                    elem_type: a,
                    size: as_,
                    ..
                },
                Expr::NewArray {
                    elem_type: b,
                    size: bs,
                    ..
                },
            ) => a == b && as_.structural_eq(bs),
            (
                Expr::Index {
                    object: ao,
                    index: ai,
                    ..
                },
                Expr::Index {
                    object: bo,
                    index: bi,
                    ..
                },
            ) => ao.structural_eq(bo) && ai.structural_eq(bi),
            (
                Expr::InterpolatedString {
                    parts: ap,
                    ..
                },
                Expr::InterpolatedString {
                    parts: bp,
                    ..
                },
            ) => ap.len() == bp.len() && ap.iter().zip(bp.iter()).all(|(x, y)| x.structural_eq(y)),
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Literal {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
}

impl PartialEq for Literal {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Literal::Str(a), Literal::Str(b)) => a == b,
            (Literal::Int(a), Literal::Int(b)) => a == b,
            // Compare by bit pattern: total and deterministic (avoids NaN),
            // and exact for the bytecode round-trip.
            (Literal::Float(a), Literal::Float(b)) => a.to_bits() == b.to_bits(),
            (Literal::Bool(a), Literal::Bool(b)) => a == b,
            (Literal::Null, Literal::Null) => true,
            _ => false,
        }
    }
}
impl Eq for Literal {}
