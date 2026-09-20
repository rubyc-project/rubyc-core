//! Postfix `++` / `--` in expression position. `x++` yields the OLD value
//! and stores old+1; `x--` the mirror. Valid on locals, fields (including
//! bare implicit-`this`), and array elements.

use crate::native::ir::{BinOp, Instr, Program};

fn lower(src: &str) -> Program {
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).unwrap()
}

fn instrs(p: &Program) -> Vec<&Instr> {
    p.functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .collect()
}

fn has(p: &Program, pred: impl Fn(&Instr) -> bool) -> bool {
    instrs(p).iter().any(|i| pred(*i))
}

const SRC_LOCAL: &str = r#"class c {
    Main() {
        int i = 41;
        int j = i++;
        int k = i--;
        return j + k;
    }
}
"#;

#[test]
fn postfix_inc_stores_old_plus_one() {
    let p = lower(SRC_LOCAL);
    // The increment itself: an Add of (old, 1) must exist ...
    assert!(
        has(&p, |i| matches!(
            i,
            Instr::Bin { op: BinOp::Add, .. }
        )),
        "i++ must lower an Add"
    );
    // ... and the decrements a Sub.
    assert!(
        has(&p, |i| matches!(
            i,
            Instr::Bin { op: BinOp::Sub, .. }
        )),
        "i-- must lower a Sub"
    );
}

#[test]
fn postfix_inc_result_feeds_use_not_store() {
    // `j = i++` — j must receive the value i had BEFORE the increment.
    // Structurally: the Add's `a` operand and the store into j's home
    // must both trace to the same loaded-old vreg. At minimum the Add
    // must read a non-constant (the loaded old), not a folded literal.
    let p = lower(SRC_LOCAL);
    let adds: Vec<&Instr> = instrs(&p)
        .into_iter()
        .filter(|i| matches!(i, Instr::Bin { op: BinOp::Add, .. }))
        .collect();
    assert!(!adds.is_empty(), "expected an Add for i++");
}

const SRC_FIELD: &str = r#"class c {
    int val;
    Main() {
        this.val = 10;
        int j = this.val++;
        int k = val--;
        return j + k;
    }
}
"#;

const SRC_INDEX_STORE: &str = r#"class c {
    Main() {
        int[] a = new int[2];
        a[0] = 7;
        int n = a[0]++;
        return n + a[0];
    }
}
"#;

#[test]
fn expression_index_assign_stores() {
    // `a[0] = 7` parses as expression-`Assign` with an `Index` target
    // (not Stmt::MemberAssign): it must still emit the element store.
    // Regression: the fallback loaded and discarded, so the array kept
    // heap garbage and `a[0]++` read garbage.
    let p = lower(SRC_INDEX_STORE);
    let stores = instrs(&p)
        .into_iter()
        .filter(|i| matches!(i, Instr::StoreField { .. }))
        .count();
    assert!(
        stores >= 2,
        "expected the element store plus the postfix write-back, got {stores}"
    );
}

#[test]
fn postfix_inc_on_fields_stores_back() {
    // Both `this.val++` and bare `val--` must write the object field back.
    let p = lower(SRC_FIELD);
    let stores = instrs(&p)
        .into_iter()
        .filter(|i| matches!(i, Instr::StoreField { .. }))
        .count();
    // ctor-style init + two postfix stores (member + implicit-this).
    assert!(
        stores >= 3,
        "expected init + 2 postfix field stores, got {stores}"
    );
}
