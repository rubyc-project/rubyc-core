//! Call targets must keep pointing at the function they name, no matter
//! what else the program contains.
//!
//! A direct call is emitted with a *predicted* function index, because the
//! callee may not be emitted yet. Lifted lambdas are inserted immediately
//! after the method that created them, and the prediction did not account
//! for that: every method declared after the first lambda moved down by the
//! number of lambdas before it. The symptom was not a crash but silently
//! running the WRONG function — `new Both()` (two concrete bases) called
//! `Base.Plain` and `Derived.Get` in place of the two base constructors, so
//! neither base's state was ever initialised.
//!
//! These tests resolve each call's index through `Program::functions` and
//! assert the name it lands on, which is exactly the check the compiler was
//! failing.

use crate::native::ir::{Instr, Program};

fn lower(src: &str) -> Program {
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower_with(&unit, false).unwrap()
}

/// Name of the function each `CallFn` in `method` actually invokes.
fn called_names(p: &Program, method: &str) -> Vec<String> {
    let f = p
        .functions
        .iter()
        .find(|f| f.name.contains(method))
        .unwrap_or_else(|| {
            panic!(
                "no function containing {method:?}; have {:?}",
                p.functions.iter().map(|f| &f.name).collect::<Vec<_>>()
            )
        });
    f.blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .filter_map(|i| match i {
            Instr::CallFn { fidx, .. } => Some(
                p.functions
                    .get(*fidx as usize)
                    .map(|t| t.name.clone())
                    .unwrap_or_else(|| format!("<out of range: {fidx}>")),
            ),
            _ => None,
        })
        .collect()
}

// The bases are NESTED in the class that holds `Main`. Nested classes are
// emitted after their enclosing class's members, so `Main` refers to a
// constructor that does not exist yet — the forward reference that makes
// the predicted index matter at all.
const MI_SRC: &str = r#"namespace t;
public class C {
    public class Left { public int L; public Left() { L = 3; } }
    public class Right { public int R; public Right() { R = 4; } }
    public class Both : Left, Right { }
    public static int Main() {
        Both b = new Both();
        return b.L + b.R;
    }
}
"#;

#[test]
fn multi_base_construction_calls_every_base_constructor() {
    let p = lower(MI_SRC);
    let called = called_names(&p, "Main#0");
    assert!(
        called.iter().any(|n| n.starts_with("Left::Left#")),
        "the first base's constructor must be called; calls were {called:?}"
    );
    assert!(
        called.iter().any(|n| n.starts_with("Right::Right#")),
        "the second base's constructor must be called; calls were {called:?}"
    );
}

/// The same program with an unrelated lambda ahead of it. The lambda adds a
/// function to the vector; the call indices must not move.
#[test]
fn lambda_does_not_shift_later_call_targets() {
    let src = r#"namespace t;
public class C {
    public class Left { public int L; public Left() { L = 3; } }
    public class Right { public int R; public Right() { R = 4; } }
    public class Both : Left, Right { }
    public static int Main() {
        System.Func<int, int> twice = x => x * 2;
        Both b = new Both();
        return b.L + b.R + twice(1);
    }
}
"#;
    let p = lower(src);
    let called = called_names(&p, "Main#0");
    assert!(
        called.iter().any(|n| n.starts_with("Left::Left#")),
        "a lambda elsewhere in the program must not retarget the call; calls were {called:?}"
    );
    assert!(
        called.iter().any(|n| n.starts_with("Right::Right#")),
        "a lambda elsewhere in the program must not retarget the call; calls were {called:?}"
    );
    assert!(
        called.iter().all(|n| !n.starts_with("Base::") && !n.starts_with("Derived::")),
        "the construction must not land on unrelated methods; calls were {called:?}"
    );
}

/// Every direct call in a lambda-bearing program must land on a real
/// function — never past the end of the table, which is how a shifted index
/// would show up in a program whose later methods do not exist.
#[test]
fn no_call_index_falls_off_the_function_table() {
    let src = r#"namespace t;
public class C {
    public class Helper { public int V; public Helper() { V = 7; } }
    public static int Main() {
        System.Func<int, int> twice = x => x * 2;
        Helper h = new Helper();
        return h.V + twice(2);
    }
}
"#;
    let p = lower(src);
    for (i, f) in p.functions.iter().enumerate() {
        for b in &f.blocks {
            for ins in &b.instrs {
                if let Instr::CallFn { fidx, .. } = ins {
                    assert!(
                        (*fidx as usize) < p.functions.len(),
                        "{}.{} calls function {fidx}, but there are only {}",
                        f.name,
                        i,
                        p.functions.len()
                    );
                }
            }
        }
    }
    let called = called_names(&p, "Main#0");
    assert!(
        called.iter().any(|n| n.starts_with("Helper::Helper#")),
        "Helper's constructor must be the callee; calls were {called:?}"
    );
}
