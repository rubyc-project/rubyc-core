//! Async state-machine lowering: verify that `await` inside an async method
//! produces the expected `TaskAlloc`/`Suspend` IR and that the bytecode
//! roundtrip preserves `Expr::Await`.

use crate::native::ir::{Instr, Program};

fn lower(src: &str) -> Program {
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).unwrap()
}

fn find_async_fn(p: &Program) -> Option<usize> {
    p.functions.iter().position(|f| f.is_async)
}

const ASYNC_SRC: &str = r#"namespace t; class c {
    int main() {
        return 0;
    }
    async int doWork() {
        int x = await 42;
        printf("{0}", x);
        return 0;
    }
}
"#;

#[test]
fn async_method_with_await_produces_task_alloc_and_suspend() {
    let p = lower(ASYNC_SRC);
    let idx = find_async_fn(&p).expect("an async function must exist");
    let f = &p.functions[idx];
    assert!(f.is_async, "function must be marked async");

    let has_task_alloc = f.blocks.iter().any(|b| {
        b.instrs
            .iter()
            .any(|i| matches!(i, Instr::TaskAlloc { .. }))
    });
    assert!(has_task_alloc, "TaskAlloc must be present in the state machine");

    let has_suspend = f.blocks.iter().any(|b| {
        b.instrs
            .iter()
            .any(|i| matches!(i, Instr::Suspend { .. }))
    });
    assert!(
        has_suspend,
        "Suspend must be present for the await suspension point"
    );
}

#[test]
fn async_method_with_no_await_is_marked_async() {
    let src = r#"namespace t; class c {
        int main() { return 0; }
        async int noWait() {
            printf("{0}", 1);
            return 0;
        }
    }
    "#;
    let p = lower(src);
    let idx = find_async_fn(&p).expect("an async function must exist");
    assert!(p.functions[idx].is_async);
}

#[test]
fn bytecode_roundtrip_preserves_await_tag() {
    let unit = crate::parser::parse(ASYNC_SRC).unwrap();

    let encoded = crate::bytecode::encode(&unit);
    let decoded = crate::bytecode::decode(&encoded).unwrap();

    let class = decoded
        .namespaces
        .first()
        .and_then(|ns| ns.members.first())
        .expect("namespace member");
    let crate::models::ast::namespace::Member::Class(c) = class else {
        panic!("expected a class member");
    };
    let async_method = c
        .members
        .iter()
        .find_map(|m| match m {
            crate::models::ClassMember::Method(m) if m.is_async => Some(m),
            _ => None,
        })
        .expect("an async method must exist in the decoded class");
    assert!(async_method.is_async, "decoded method must be async");

    assert!(
        encoded.contains(&crate::bytecode::format::tag::AWAIT),
        "encoded bytecode must contain the AWAIT tag (0x{:02X})",
        crate::bytecode::format::tag::AWAIT
    );
}

#[test]
fn branching_async_method_has_no_dangling_edges() {
    // Branching async methods (`BufferedStream.FlushAsyncInternal`:
    // awaits inside ifs) skip the linearizing state-machine transform
    // — it drops non-sequential edges and once emitted a branch past
    // the last block. They lower linearly instead (Suspend is a nop
    // until the async runtime lands).
    let src = r#"namespace t; class c {
        int main() { return 0; }
        async int work(int x) {
            int y = await 1;
            try {
                if (x > 0) {
                    y = await 2;
                    return y;
                }
                while (x > 0) {
                    y = await 3;
                    x = x - 1;
                }
            } finally {
                y = 0;
            }
            return y;
        }
    }
    "#;
    let p = lower(src);
    let idx = find_async_fn(&p).expect("an async function must exist");
    let f = &p.functions[idx];
    assert!(f.is_async, "function must stay marked async");
    for (bi, b) in f.blocks.iter().enumerate() {
        let targets: Vec<usize> = match &b.term {
            crate::native::ir::Terminator::Branch { if_true, if_false, .. } => {
                vec![*if_true as usize, *if_false as usize]
            }
            _ => vec![],
        };
        for t in targets {
            assert!(
                t < f.blocks.len(),
                "block {bi} branches to dangling block {t} ({} blocks)",
                f.blocks.len()
            );
        }
    }
}
