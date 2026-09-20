//! `[Inline]` method expansion: single-`return expr` bodies are spliced
//! into call sites; non-pure args, wrong arity, and non-inline methods
//! leave the call intact.

use crate::inliner::run;
use crate::models::{ClassMember, CompilationUnit, Expr, Member, Stmt};
use crate::parser;

fn parse(src: &str) -> CompilationUnit {
    parser::parse(src)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}\nsrc:\n{src}"))
}

/// Collect every `Call` callee name in the unit (post-inlining).
fn all_call_callees(unit: &CompilationUnit) -> Vec<String> {
    let mut out = Vec::new();
    let members = unit.all_members();
    for m in members {
        if let Member::Class(c) = m {
            for cm in &c.members {
                if let ClassMember::Method(meth) = cm {
                    if let Some(body) = &meth.body {
                        collect_call_callees_block(&body.stmts, &mut out);
                    }
                }
            }
        }
    }
    out
}

fn collect_call_callees_block(stmts: &[Stmt], out: &mut Vec<String>) {
    for s in stmts {
        match s {
            Stmt::Expression(e) | Stmt::Return { value: Some(e), .. } => collect_call_callees_expr(e, out),
            Stmt::Let { init, .. } | Stmt::Assign { value: init, .. } => collect_call_callees_expr(init, out),
            Stmt::If { cond, then_block, else_block, .. } => {
                collect_call_callees_expr(cond, out);
                collect_call_callees_block(&then_block.stmts, out);
                if let Some(b) = else_block {
                    collect_call_callees_block(&b.stmts, out);
                }
            }
            _ => {}
        }
    }
}

fn collect_call_callees_expr(expr: &Expr, out: &mut Vec<String>) {
    match expr {
        Expr::Call { callee, args, .. } => {
            out.push(callee.clone());
            for a in args {
                collect_call_callees_expr(a, out);
            }
        }
        Expr::Binary { lhs, rhs, .. } => {
            collect_call_callees_expr(lhs, out);
            collect_call_callees_expr(rhs, out);
        }
        Expr::Unary { inner, .. } => collect_call_callees_expr(inner, out),
        Expr::Ternary { cond, then_expr, else_expr, .. } => {
            collect_call_callees_expr(cond, out);
            collect_call_callees_expr(then_expr, out);
            collect_call_callees_expr(else_expr, out);
        }
        _ => {}
    }
}

// ---------- basic inlining ----------

#[test]
fn inline_method_is_spliced() {
    let mut unit = parse(
        "namespace n {
            class c {
                [Inline] public int sq(int x) {
                    return x * x;
                }
                public int f() {
                    return sq(3);
                }
            }
        }",
    );
    run(&mut unit);
    // After inlining, the call to `sq` is gone (spliced into `3 * 3`).
    assert!(!all_call_callees(&unit).contains(&"sq".to_string()));
}

#[test]
fn non_inline_method_not_spliced() {
    let mut unit = parse(
        "namespace n {
            class c {
                public int sq(int x) {
                    return x * x;
                }
                public int f() {
                    return sq(3);
                }
            }
        }",
    );
    run(&mut unit);
    // No `[Inline]` attribute -> call stays.
    assert!(all_call_callees(&unit).contains(&"sq".to_string()));
}

#[test]
fn no_inline_methods_is_noop() {
    let mut unit = parse(
        "namespace n {
            class c {
                public int f() {
                    return 1;
                }
            }
        }",
    );
    run(&mut unit);
    assert!(all_call_callees(&unit).is_empty());
}

// ---------- purity gate ----------

#[test]
fn impure_arg_not_inlined() {
    // `sq(g())` has a non-pure arg (a Call) -> falls back to a normal call.
    let mut unit = parse(
        "namespace n {
            class c {
                [Inline] public int sq(int x) {
                    return x * x;
                }
                public int g() {
                    return 1;
                }
                public int f() {
                    return sq(g());
                }
            }
        }",
    );
    run(&mut unit);
    assert!(all_call_callees(&unit).contains(&"sq".to_string()));
}

#[test]
fn literal_arg_inlined() {
    let mut unit = parse(
        "namespace n {
            class c {
                [Inline] public int dbl(int x) {
                    return x + x;
                }
                public int f() {
                    return dbl(4);
                }
            }
        }",
    );
    run(&mut unit);
    assert!(!all_call_callees(&unit).contains(&"dbl".to_string()));
}

// ---------- arity gate ----------

#[test]
fn wrong_arity_not_inlined() {
    // `sq` takes 1 param, called with 2 -> not inlined.
    let mut unit = parse(
        "namespace n {
            class c {
                [Inline] public int sq(int x) {
                    return x * x;
                }
                public int f() {
                    return sq(1, 2);
                }
            }
        }",
    );
    // This is a semantic error but the parser recovers; the inliner must
    // still leave the call intact (arity mismatch).
    run(&mut unit);
    assert!(all_call_callees(&unit).contains(&"sq".to_string()));
}

// ---------- parameter substitution ----------

#[test]
fn two_param_substitution() {
    let mut unit = parse(
        "namespace n {
            class c {
                [Inline] public int add(int a, int b) {
                    return a + b;
                }
                public int f() {
                    return add(1, 2);
                }
            }
        }",
    );
    run(&mut unit);
    assert!(!all_call_callees(&unit).contains(&"add".to_string()));
}

#[test]
fn param_appearing_twice_both_substituted() {
    // Body `x * x` uses the param twice; both occurrences become the arg.
    let mut unit = parse(
        "namespace n {
            class c {
                [Inline] public int sq(int x) {
                    return x * x;
                }
                public int f() {
                    int r = sq(5);
                    return r;
                }
            }
        }",
    );
    run(&mut unit);
    assert!(!all_call_callees(&unit).contains(&"sq".to_string()));
}

// ---------- control-flow nesting ----------

#[test]
fn inline_inside_if_branch() {
    let mut unit = parse(
        "namespace n {
            class c {
                [Inline] public int dbl(int x) {
                    return x + x;
                }
                public int f() {
                    if (true) {
                        return dbl(2);
                    }
                    return 0;
                }
            }
        }",
    );
    run(&mut unit);
    assert!(!all_call_callees(&unit).contains(&"dbl".to_string()));
}

// ---------- multi-statement body is not inlined ----------

#[test]
fn multi_statement_body_not_inlined() {
    let mut unit = parse(
        "namespace n {
            class c {
                [Inline] public int f2(int x) {
                    int t = x + 1;
                    return t;
                }
                public int g() {
                    return f2(1);
                }
            }
        }",
    );
    // Body has two statements (let + return) -> not a single-return inline.
    run(&mut unit);
    assert!(all_call_callees(&unit).contains(&"f2".to_string()));
}

#[test]
fn empty_unit_noop() {
    let mut unit = parse("");
    run(&mut unit);
    // No panic, no changes.
    assert!(all_call_callees(&unit).is_empty());
}
