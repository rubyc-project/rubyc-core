//! Statement nodes.

use super::expression::Expr;
use super::method::SwitchArm;
use crate::models::Span;
use crate::models::compare::StructuralEq;

/// Access modifiers. Parsed and stored now; enforcement arrives with the
/// semantic-analysis pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Access {
    #[default]
    Default,
    Public,
    Private,
    Protected,
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    /// `int x = expr;` — declaration with initializer.
    Let {
        type_name: String,
        name: String,
        init: Expr,
        access: Access,
        span: Span,
    },
    /// `x = expr;`
    Assign {
        name: String,
        value: Expr,
        span: Span,
    },
    /// `obj.field = expr;` — field write through an object reference.
    MemberAssign {
        object: Expr,
        name: String,
        value: Expr,
        span: Span,
    },
    /// An expression evaluated for its effect, followed by `;`.
    Expression(Expr),
    /// A standalone `{ ... }` block: its statements execute in their own
    /// scope (declarations vanish at the closing brace).
    Block {
        body: crate::models::Block,
        span: Span,
    },
    /// `return;` or `return expr;`
    Return { value: Option<Expr>, span: Span },
    /// `if (cond) { … } else { … }` — the else branch is optional.
    If {
        cond: Expr,
        then_block: crate::models::Block,
        else_block: Option<crate::models::Block>,
        span: Span,
    },
    /// `while (cond) { … }`
    While {
        cond: Expr,
        body: crate::models::Block,
        span: Span,
    },
    /// `do { … } while (cond);`
    DoWhile {
        body: crate::models::Block,
        cond: Expr,
        span: Span,
    },
    /// `for (init; cond; step) { … }` — init is a Let/Assign statement,
    /// cond and step are optional.
    For {
        init: Option<Box<Stmt>>,
        cond: Option<Expr>,
        step: Option<Box<Stmt>>,
        body: crate::models::Block,
        span: Span,
    },
    /// `switch (subject) { case N: … break; default: … }`
    /// Labels are constant expressions today (A11 widens to patterns).
    Switch {
        subject: Expr,
        arms: Vec<SwitchArm>,
        default_body: Option<crate::models::Block>,
        span: Span,
    },
    /// `break;`
    Break { span: Span },
    /// `continue;`
    Continue { span: Span },
    /// `goto <label>;`, `goto case <expr>;` or `goto default;`
    Goto {
        /// Bare label, e.g. `GotoStart`. `None` for `goto default;`.
        label: Option<String>,
        /// Present for `goto case <expr>;`.
        case_expr: Option<Expr>,
        span: Span,
    },
    /// `try { … } catch (T e) { … } … finally { … }`.
    /// At least one of `catches` or `finally_block` is present.
    Try {
        body: crate::models::Block,
        catches: Vec<CatchClause>,
        finally_block: Option<crate::models::Block>,
        span: Span,
    },
    /// `throw expr;` or bare `throw;` (rethrow the in-flight exception).
    Throw {
        /// `None` for a bare rethrow.
        expr: Option<Expr>,
        span: Span,
    },
}

/// One `catch` clause of a [`Stmt::Try`]: `catch (T name) { … }` or `catch { … }`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatchClause {
    /// The caught type. `None` = catch-all (`catch { … }`).
    pub ty: Option<String>,
    /// The bound exception variable. `None` if the clause binds no name.
    pub name: Option<String>,
    /// The catch body.
    pub body: crate::models::Block,
    pub span: Span,
}

impl StructuralEq for Stmt {
    fn structural_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Stmt::Let {
                    type_name: a,
                    name: an,
                    init: ai,
                    access: aa,
                    ..
                },
                Stmt::Let {
                    type_name: b,
                    name: bn,
                    init: bi,
                    access: ab,
                    ..
                },
            ) => a == b && an == bn && ai.structural_eq(bi) && aa == ab,
            (
                Stmt::Assign {
                    name: a, value: av, ..
                },
                Stmt::Assign {
                    name: b, value: bv, ..
                },
            ) => a == b && av.structural_eq(bv),
            (
                Stmt::MemberAssign {
                    object: ao,
                    name: an,
                    value: av,
                    ..
                },
                Stmt::MemberAssign {
                    object: bo,
                    name: bn,
                    value: bv,
                    ..
                },
            ) => an == bn && ao.structural_eq(bo) && av.structural_eq(bv),
            (Stmt::Expression(a), Stmt::Expression(b)) => a.structural_eq(b),
            (
                Stmt::If {
                    cond: ac,
                    then_block: at,
                    else_block: ae,
                    ..
                },
                Stmt::If {
                    cond: bc,
                    then_block: bt,
                    else_block: be,
                    ..
                },
            ) => {
                ac.structural_eq(bc)
                    && at.structural_eq(bt)
                    && match (ae, be) {
                        (Some(a), Some(b)) => a.structural_eq(b),
                        (None, None) => true,
                        _ => false,
                    }
            }
            (
                Stmt::While {
                    cond: ac, body: ab, ..
                },
                Stmt::While {
                    cond: bc, body: bb, ..
                },
            ) => ac.structural_eq(bc) && ab.structural_eq(bb),
            (
                Stmt::DoWhile {
                    body: ab, cond: ac, ..
                },
                Stmt::DoWhile {
                    body: bb, cond: bc, ..
                },
            ) => ab.structural_eq(bb) && ac.structural_eq(bc),
            (
                Stmt::For {
                    init: ai,
                    cond: ac,
                    step: as_,
                    body: ab,
                    ..
                },
                Stmt::For {
                    init: bi,
                    cond: bc,
                    step: bs,
                    body: bb,
                    ..
                },
            ) => {
                let inits = match (ai.as_deref(), bi.as_deref()) {
                    (Some(a), Some(b)) => a.structural_eq(b),
                    (None, None) => true,
                    _ => false,
                };
                let conds = match (ac, bc) {
                    (Some(a), Some(b)) => a.structural_eq(b),
                    (None, None) => true,
                    _ => false,
                };
                let steps = match (as_.as_deref(), bs.as_deref()) {
                    (Some(a), Some(b)) => a.structural_eq(b),
                    (None, None) => true,
                    _ => false,
                };
                inits && conds && steps && ab.structural_eq(bb)
            }
            (
                Stmt::Switch {
                    subject: as_,
                    arms: aa,
                    default_body: ad,
                    ..
                },
                Stmt::Switch {
                    subject: bs,
                    arms: ba,
                    default_body: bd,
                    ..
                },
            ) => {
                let subjects = as_.structural_eq(bs);
                let arms_eq = aa.len() == ba.len()
                    && aa.iter().zip(ba.iter()).all(|(x, y)| {
                        x.labels.len() == y.labels.len()
                            && x.labels
                                .iter()
                                .zip(y.labels.iter())
                                .all(|(a, b)| a.structural_eq(b))
                            && x.body.structural_eq(&y.body)
                    });
                let defaults = match (ad, bd) {
                    (Some(a), Some(b)) => a.structural_eq(b),
                    (None, None) => true,
                    _ => false,
                };
                subjects && arms_eq && defaults
            }
            (Stmt::Break { .. }, Stmt::Break { .. })
            | (Stmt::Continue { .. }, Stmt::Continue { .. }) => true,
            (
                Stmt::Goto {
                    label: al,
                    case_expr: ac,
                    ..
                },
                Stmt::Goto {
                    label: bl,
                    case_expr: bc,
                    ..
                },
            ) => {
                al == bl
                    && match (ac, bc) {
                        (Some(a), Some(b)) => a.structural_eq(b),
                        (None, None) => true,
                        _ => false,
                    }
            }
            (Stmt::Return { value: av, .. }, Stmt::Return { value: bv, .. }) => match (av, bv) {
                (Some(a), Some(b)) => a.structural_eq(b),
                (None, None) => true,
                _ => false,
            },
            (
                Stmt::Try {
                    body: ab,
                    catches: ac,
                    finally_block: af,
                    ..
                },
                Stmt::Try {
                    body: bb,
                    catches: bc,
                    finally_block: bf,
                    ..
                },
            ) => {
                ab.structural_eq(bb)
                    && ac.structural_eq(bc)
                    && match (af, bf) {
                        (Some(a), Some(b)) => a.structural_eq(b),
                        (None, None) => true,
                        _ => false,
                    }
            }
            (
                Stmt::Throw { expr: ae, .. },
                Stmt::Throw { expr: be, .. },
            ) => match (ae, be) {
                (Some(a), Some(b)) => a.structural_eq(b),
                (None, None) => true,
                _ => false,
            },
            _ => false,
        }
    }
}

/// Blanket helper so slices of any `StructuralEq` compare element-wise.
impl<T: StructuralEq> StructuralEq for [T] {
    fn structural_eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().zip(other).all(|(a, b)| a.structural_eq(b))
    }
}

impl<T: StructuralEq> StructuralEq for Vec<T> {
    fn structural_eq(&self, other: &Self) -> bool {
        self.as_slice().structural_eq(other.as_slice())
    }
}

impl StructuralEq for CatchClause {
    fn structural_eq(&self, other: &Self) -> bool {
        self.ty == other.ty
            && self.name == other.name
            && self.body.structural_eq(&other.body)
    }
}
