//! Chained field access through an object-typed field (`this.inner.val`,
//! `o.getinner().val`) must lower to real `LoadField` chains, not the
//! "unknown type -> return 0" placeholder. Regression: `static_type_of`
//! previously had no arm for `Expr::Member`/`Expr::MethodCall`, so any
//! field access whose base was itself a field/call read a bare 0.

use crate::native::ir::{Instr, Program};

fn lower(src: &str) -> Program {
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower_with(&unit, false).unwrap()
}

fn find_fn<'a>(p: &'a Program, needle: &str) -> &'a crate::native::ir::IrFunction {
    p.functions
        .iter()
        .find(|f| f.name.contains(needle))
        .unwrap_or_else(|| panic!("no function containing {needle:?}; have {:?}", p.functions.iter().map(|f| &f.name).collect::<Vec<_>>()))
}

fn count_loadfield(f: &crate::native::ir::IrFunction) -> usize {
    f.blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(i, Instr::LoadField { .. }))
        .count()
}

const SRC: &str = r#"namespace t {
    class Inner {
        public int val;
        public Inner(int v) { this.val = v; }
    }
    class Outer {
        public Inner inner;
        public Outer(int v) { this.inner = new Inner(v); }
        public int GetVal() { return this.inner.val; }
        public Inner GetInner() { return this.inner; }
        public int ViaCall() { return this.GetInner().val; }
    }
}
"#;

#[test]
fn field_through_object_field_is_a_loadfield_chain() {
    let p = lower(SRC);
    let getval = find_fn(&p, "GetVal");
    // `this.inner.val` = LoadField(inner) then LoadField(val).
    assert!(
        count_loadfield(getval) >= 2,
        "GetVal must lower to a LoadField chain (this.inner, then .val); got {}",
        count_loadfield(getval)
    );
}

#[test]
fn field_through_call_result_is_a_loadfield_chain() {
    let p = lower(SRC);
    let via_call = find_fn(&p, "ViaCall");
    // `this.GetInner().val`: the `.val` read is a real LoadField (pre-fix this
    // lowered to a bare Const(0) because `static_type_of` returned "object").
    assert!(
        count_loadfield(via_call) >= 1,
        "ViaCall must lower this.GetInner().val to a LoadField; got {}",
        count_loadfield(via_call)
    );
    // The `.val` read must come from a call result, not a const placeholder.
    assert!(
        via_call.blocks.iter().flat_map(|b| b.instrs.iter())
            .any(|i| matches!(i, Instr::CallVirt { .. })),
        "ViaCall must contain the GetInner virtual call"
    );
}

const SRC_IFACE: &str = r#"namespace t {
    interface IShape {
        int area();
    }
    class Rect : IShape {
        public int w;
        public int h;
        public Rect(int w, int h) { this.w = w; this.h = h; }
        public int area() { return this.w * this.h; }
    }
    class B {
        public int main() {
            IShape s = new Rect(3, 4);
            return s.area();
        }
    }
}
"#;

const SRC_EXPR_MEMBER_ASSIGN: &str = r#"namespace t {
    class Box {
        public int val;
        public Box(int v) { this.val = v; }
        public int bump(int d) { this.val = this.val + d; return this.val; }
    }
}
"#;

#[test]
fn expression_member_assign_emits_storefield() {
    // `this.val = v` parses as an expression-`Assign` with a `Member`
    // target (not Stmt::MemberAssign); it must still emit the StoreField.
    // Regression: the fallback evaluated the target (a load) and dropped
    // the value, so ctors never stored their arguments.
    let p = lower(SRC_EXPR_MEMBER_ASSIGN);
    let ctor = find_fn(&p, "Box#1");
    assert!(
        ctor.blocks.iter().flat_map(|b| b.instrs.iter())
            .any(|i| matches!(i, Instr::StoreField { .. })),
        "ctor this.val = v must lower to a StoreField"
    );
    let bump = find_fn(&p, "bump");
    assert!(
        bump.blocks.iter().flat_map(|b| b.instrs.iter())
            .any(|i| matches!(i, Instr::StoreField { .. })),
        "bump this.val = this.val + d must lower to a StoreField"
    );
}

const SRC_EXPR_FIELD_STORE: &str = r#"namespace t {
    class Box {
        public int val;
        public Box(int v) { this.val = v; }
        public int setit(int d) { return val = d; }
        public int orset(int d) { return val += d; }
    }
}
"#;

#[test]
fn bare_field_expression_assign_stores() {
    // `return val = d` where `val` is a field (not a local): the
    // expression-`Assign` path must resolve the implicit-`this` field and
    // emit the StoreField, mirroring the statement-`Assign` path.
    // Without it the value is returned but the field never updates.
    let p = lower(SRC_EXPR_FIELD_STORE);
    for needle in ["setit", "orset"] {
        let f = find_fn(&p, needle);
        assert!(
            f.blocks.iter().flat_map(|b| b.instrs.iter())
                .any(|i| matches!(i, Instr::StoreField { .. })),
            "{needle} with a bare-field expression store must lower to a StoreField"
        );
    }
}

#[test]
fn method_call_on_interface_typed_receiver_is_virtual() {
    let p = lower(SRC_IFACE);
    let main = find_fn(&p, "main");
    // `s.area()` where `s: IShape` must dispatch through the vtable
    // (CallVirt), not fall to the "unknown type -> return 0" placeholder
    // that the fix eliminated by resolving interface receivers.
    assert!(
        main.blocks.iter().flat_map(|b| b.instrs.iter())
            .any(|i| matches!(i, Instr::CallVirt { .. })),
        "main must contain a CallVirt for s.area(); interface-typed receivers \
         used to lower to a bare Const(0)"
    );
}
