//! Switch-statement section scoping.
//!
//! C# switch sections share ONE scope across all arms: a local declared
//! in `case 0:` is visible (and assignable) in `case 1:`. Per-arm scopes
//! break this (`BinaryReader.FillBuffer` declares `n` in one case and
//! assigns it in the next).

use crate::native::ir::{Instr, Program};

fn lower(src: &str) -> Program {
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).unwrap()
}

fn has(p: &Program, pred: impl Fn(&Instr) -> bool) -> bool {
    p.functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .any(pred)
}

#[test]
fn switch_arm_local_visible_in_later_arm() {
    let p = lower(
        r#"namespace t; class M {
    int main() {
        int s = 1;
        int r = 0;
        switch (s) {
            case 0:
                int n = 5;
                r = n;
                break;
            case 1:
                n = 6;
                r = n;
                break;
        }
        return r;
    }
}"#,
    );
    // Both arms store through the shared local (no undeclared errors,
    // and the method completes with a return).
    assert!(
        has(&p, |i| matches!(i, Instr::StoreField { .. } | Instr::Mov { .. })),
        "expected cross-arm local stores to lower"
    );
}

#[test]
fn sealed_branch_with_heap_local_exits_cleanly() {
    // A scoped block that seals (early `return`) with a heap-typed
    // local alive must not emit past the terminator (debug-assert
    // `instruction after terminator`); the local simply leaks, as at
    // method scope.
    let p = lower(
        r#"namespace t;
class H {
    public H() { }
}
class C {
    int main() {
        int x = 1;
        if (x != 0) {
            H h = new H();
            return 1;
        }
        return 0;
    }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallFn { .. })),
        "expected construction call to lower"
    );
}

#[test]
fn case_type_pattern_binds_arm_variable() {
    // `case TC tc:` declares `tc` for the arm (evaluated as zero —
    // pattern tests are placeholders, but the NAME must resolve).
    let p = lower(
        r#"namespace t;
class TC {
    public int Run() { return 1; }
}
class M {
    int main() {
        object o = new TC();
        int r = 0;
        switch (o) {
            case TC tc:
                r = tc.Run();
                break;
            default:
                break;
        }
        return r;
    }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallVirt { .. })),
        "expected arm call on the pattern variable to lower"
    );
}

#[test]
fn same_pattern_name_across_arms_is_legal() {
    // C# scopes case pattern variables per arm: `case TA tc:` and
    // `case TB tc:` in one switch is legal (sibling to the
    // `RunContinuations` shape). Our shared section scope degrades
    // the second declaration to a re-bind instead of erroring.
    let p = lower(
        r#"namespace t;
class TA { }
class TB { }
class M {
    int main() {
        object o = new TA();
        int r = 0;
        switch (o) {
            case TA tc:
                r = 1;
                break;
            case TB tc:
                r = 2;
                break;
            default:
                break;
        }
        return r;
    }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::Const { imm: 1, .. })),
        "expected switch arms to lower"
    );
}

#[test]
fn index_into_nested_array_prefers_enclosing_scope() {
    // `Bucket[]` inside `HT` means `HT.Bucket`, even when another
    // same-named `Bucket` is declared first (the
    // `Hashtable.Bucket.hash_coll` shape).
    let p = lower(
        r#"namespace t;
class Other {
    class Bucket {
        public int other;
    }
}
class HT {
    struct Bucket {
        public int hash_coll;
    }
    private Bucket[] _buckets;
    public void Insert() {
        int x = _buckets[0].hash_coll;
        if (x != 0) { }
    }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::LoadField { .. })),
        "expected nested element member load"
    );
}

#[test]
fn index_into_non_array_yields_object_placeholder() {
    // Indexing a non-array indexable (`Span<T>` with stripped generics)
    // must not resolve members against the container type
    // (`tmp[0]._exponent` on `Span<DiyFp128>` errored); it degrades
    // to the usual placeholder and compilation succeeds.
    let p = lower(
        r#"namespace t;
class E {
    public int exp;
}
class Sp {
    public E this[int i] { get { return null; } }
}
class C {
    int main() {
        Sp s = new Sp();
        s[0].exp = 5;
        return 0;
    }
}"#,
    );
    assert!(
        has(&p, |_| true),
        "expected indexer member store to lower without error"
    );
}

#[test]
fn switch_block_local_function_visible_in_arms() {
    // A local function declared at switch-block level (after the
    // arms) is visible in every arm (the `Version.TryFormatCore`
    // `ThrowArgumentException` shape).
    let p = lower(
        r#"namespace t;
class C {
    int M(int x) {
        switch (x) {
            case 0:
                Foo();
                break;
            default:
                break;
            static void Foo() { }
        }
        return 0;
    }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::Const { imm: 0, .. })),
        "expected switch-wide local function call to lower"
    );
}

#[test]
fn nested_type_reference_prefers_enclosing_scope() {
    // `Impl` declared in `M3` means `M3.Impl`, not a same-named
    // nested `Impl` declared first elsewhere (the
    // `Matrix3x2.Impl` vs `Matrix4x4.Impl` shape).
    let p = lower(
        r#"namespace t;
class M4 {
    struct Impl {
        public int other;
    }
}
class M3 {
    struct Impl {
        public int mine;
    }
    int main() {
        Impl x = new Impl();
        int y = x.mine;
        if (y != 0) { }
        return 0;
    }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::LoadField { .. })),
        "expected in-scope nested member load"
    );
}

#[test]
fn switch_pattern_var_retypes_per_arm() {
    // Pattern variables are arm-scoped in C# (`case A a:` + `case B a:`
    // is legal): re-binding re-types the shared home so member calls
    // in the later arm resolve against the new type
    // (`Task.RunContinuations`: `case TaskContinuation tc:` after
    // `case RuntimeAsyncTaskContinuation tc:`).
    let src = r#"namespace t;
class A { public int Am() { return 1; } }
class B { public int Bm() { return 2; } }
class User {
    int M(object o) {
        switch (o) {
            case A a: return a.Am();
            case B a: return a.Bm();
        }
        return 0;
    }
    int main() { return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).expect("pattern re-binding must re-type");
}

#[test]
fn switch_tuple_pattern_binds_typed_elements() {
    // `case (VT_I1 or VT_UI1, sbyte s):` binds `s: sbyte` (tuple
    // pattern elements carry their own declarations; ComVariant
    // marshaling in CoreLib). Unbound would be E0025/undeclared.
    let src = r#"namespace t;
class User {
    int M(object o) {
        switch (o) {
            case (1 or 2, sbyte s): return s;
        }
        return 0;
    }
    int main() { return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).expect("tuple element must bind");
}
