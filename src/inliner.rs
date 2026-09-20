//! `[Inline]` method expansion.
//!
//! A method marked `[Inline]` whose body is a single `return expr;`
//! statement is spliced into every call site as an expression — the
//! classic zero-overhead abstraction. Arguments must be side-effect-free
//! at the call site (literals, identifiers, or other already-expanded
//! expressions); anything else falls back to a normal call.
//!
//! Runs AFTER parsing and BEFORE the unknown-name scan, so expansions are
//! checked exactly like handwritten code.

use crate::models::{ClassMember, CompilationUnit, Expr, Stmt};
use std::collections::BTreeMap;

struct InlineFn {
    params: Vec<String>,
    ret_expr: Expr,
}

pub fn run(unit: &mut CompilationUnit) {
    // ---- collect inline candidates: "Class::name" → (params, ret_expr)
    let mut inlines: BTreeMap<String, InlineFn> = BTreeMap::new();

    for member in members_mut(unit) {
        if let crate::models::Member::Class(class) = member {
            for cm in &mut class.members {
                if let ClassMember::Method(m) = cm {
                    let marked = m.attrs.iter().any(|a| a.name == "Inline");
                    let single_return = matches!(
                        m.body.as_ref().and_then(|b| b.stmts.last()),
                        Some(Stmt::Return { value: Some(_), .. })
                    ) && m.body.as_ref().map(|b| b.stmts.len()).unwrap_or(0) <= 1;
                    if marked && single_return && !m.name.starts_with('~')
                        && let Some(Stmt::Return { value: Some(ret), .. }) =
                            m.body.as_ref().and_then(|b| b.stmts.last())
                    {
                        // Key by the bare method name: call sites reference
                        // the method by name (or `Class::name`, which is
                        // reduced to the bare name during the match), so the
                        // lookup must use the same key.
                        inlines.insert(
                            m.name.clone(),
                            InlineFn {
                                params: m.params.iter().map(|p| p.name.clone()).collect(),
                                ret_expr: ret.clone(),
                            },
                        );
                    }
                }
            }
        }
    }
    if inlines.is_empty() {
        return;
    }

    // ---- expand call sites inside every method body
    for member in members_mut(unit) {
        if let crate::models::Member::Class(class) = member {
            for cm in &mut class.members {
                if let ClassMember::Method(m) = cm
                    && let Some(body) = &mut m.body
                {
                    let self_key = format!("{}::{}", class.name, m.name);
                    for stmt in &mut body.stmts {
                        expand_stmt(stmt, &self_key, &inlines, 0);
                    }
                }
            }
        }
    }
}

fn members_mut(unit: &mut CompilationUnit) -> Vec<&mut crate::models::Member> {
    let mut out: Vec<&mut crate::models::Member> = Vec::new();
    for ns in &mut unit.namespaces {
        out.extend(ns.members.iter_mut());
    }
    out.extend(unit.members.iter_mut());
    out
}

fn expand_stmt(stmt: &mut Stmt, self_key: &str, inlines: &BTreeMap<String, InlineFn>, depth: u8) {
    if depth > 8 {
        return;
    }
    match stmt {
        Stmt::Expression(e) | Stmt::Return { value: Some(e), .. } => {
            expand_expr(e, self_key, inlines, depth);
        }
        Stmt::Let { init, .. } | Stmt::Assign { value: init, .. } => {
            expand_expr(init, self_key, inlines, depth);
        }
        Stmt::MemberAssign { object, value, .. } => {
            expand_expr(object, self_key, inlines, depth);
            expand_expr(value, self_key, inlines, depth);
        }
        Stmt::If {
            cond,
            then_block,
            else_block,
            ..
        } => {
            expand_expr(cond, self_key, inlines, depth);
            for s in &mut then_block.stmts {
                expand_stmt(s, self_key, inlines, depth + 1);
            }
            if let Some(b) = else_block {
                for s in &mut b.stmts {
                    expand_stmt(s, self_key, inlines, depth + 1);
                }
            }
        }
        Stmt::While { cond, body, .. } => {
            expand_expr(cond, self_key, inlines, depth);
            for s in &mut body.stmts {
                expand_stmt(s, self_key, inlines, depth + 1);
            }
        }
        Stmt::DoWhile { body, cond, .. } => {
            for s in &mut body.stmts {
                expand_stmt(s, self_key, inlines, depth + 1);
            }
            expand_expr(cond, self_key, inlines, depth);
        }
        Stmt::For {
            init,
            cond,
            step,
            body,
            ..
        } => {
            if let Some(s) = init.as_mut() {
                expand_stmt(s.as_mut(), self_key, inlines, depth);
            }
            if let Some(c) = cond {
                expand_expr(c, self_key, inlines, depth);
            }
            if let Some(s) = step.as_mut() {
                expand_stmt(s.as_mut(), self_key, inlines, depth);
            }
            for s in &mut body.stmts {
                expand_stmt(s, self_key, inlines, depth + 1);
            }
        }
        Stmt::Switch { subject, arms, default_body, .. } => {
            expand_expr(subject, self_key, inlines, depth);
            for arm in arms {
                for l in &mut arm.labels {
                    expand_expr(l, self_key, inlines, depth);
                }
                for s in &mut arm.body.stmts {
                    expand_stmt(s, self_key, inlines, depth + 1);
                }
            }
            if let Some(b) = default_body {
                for s in &mut b.stmts {
                    expand_stmt(s, self_key, inlines, depth + 1);
                }
            }
        }
        _ => {}
    }
}

#[allow(clippy::only_used_in_recursion)]
fn expand_expr(
    expr: &mut Expr,
    self_key: &str,
    inlines: &BTreeMap<String, InlineFn>,
    depth: u8,
) {
    // Depth-first: expand arguments first.
    match expr {
        Expr::Binary { lhs, rhs, .. } => {
            expand_expr(lhs, self_key, inlines, depth);
            expand_expr(rhs, self_key, inlines, depth);
        }
        Expr::Unary { inner, .. } => expand_expr(inner, self_key, inlines, depth),
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            expand_expr(cond, self_key, inlines, depth);
            expand_expr(then_expr, self_key, inlines, depth);
            expand_expr(else_expr, self_key, inlines, depth);
        }
        Expr::Member { object, .. } => expand_expr(object, self_key, inlines, depth),
        Expr::MethodCall { object, args, .. } => {
            expand_expr(object, self_key, inlines, depth);
            for a in args {
                expand_expr(a, self_key, inlines, depth);
            }
        }
        Expr::New { args, .. } | Expr::Call { args, .. } => {
            for a in args {
                expand_expr(a, self_key, inlines, depth);
            }
        }
        _ => {}
    }

    if let Expr::Call { callee, args, .. } = expr {
        let bare = callee
            .rsplit_once("::")
            .map(|x| x.1.to_owned())
            .unwrap_or_else(|| callee.clone());
        let matches = inlines.contains_key(&bare)
            && args_are_pure(args)
            && inlines
                .get(&bare)
                .map(|f| f.params.len() == args.len())
                .unwrap_or(false);
        if matches && depth <= 8 {
            let f = &inlines[&bare];
            let mut replacement = f.ret_expr.clone();
            let params = f.params.clone();
            substitute_params(&mut replacement, &params, args);
            *expr = replacement;
        }
    }
}


fn args_are_pure(args: &[Expr]) -> bool {
    args.iter().all(|a| {
        matches!(a, Expr::Literal(_) | Expr::Ident { .. })
    })
}

fn substitute_params(expr: &mut Expr, params: &[String], args: &[Expr]) {
    match expr {
        Expr::Ident { name, .. } => {
            if let Some(i) = params.iter().position(|p| p == name)
                && let Some(arg) = args.get(i)
            {
                *expr = arg.clone();
            }
        }
        Expr::Binary { lhs, rhs, .. } => {
            substitute_params(lhs, params, args);
            substitute_params(rhs, params, args);
        }
        Expr::Unary { inner, .. } => substitute_params(inner, params, args),
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            substitute_params(cond, params, args);
            substitute_params(then_expr, params, args);
            substitute_params(else_expr, params, args);
        }
        _ => {}
    }
}
