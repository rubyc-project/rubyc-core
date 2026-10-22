//! Constant folding: literal-only arithmetic collapses to a single Const
//! before emission. Folding is deliberately restricted to
//! single-assignment temporaries: once loops and branches exist, a
//! variable's value at one program point proves nothing about it later,
//! so reads of locals are runtime values by design.

use crate::native::ir::{Instr, Program};

fn lower(src: &str) -> Program {
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).unwrap()
}

#[test]
fn literal_arithmetic_folds_to_one_const() {
    let p = lower("namespace t; class c { int main() { printf(\"{0}\", 2 * 3 + 1); } }");
    let main = program_main(&p);
    let bins = count_bins(main);
    assert_eq!(bins, 0, "2*3+1 must fold; IR:\n{main:#?}");
}

#[test]
fn local_reads_are_runtime_values_soundly() {
    // `a` is a mutable home: with loops/branches in the language, folding
    // through it was UNSOUND (a back-edge invalidates the constant).
    // Literal sub-expressions still fold — 6 * 7 becomes one Const — but
    // the read of `b` plus the subtraction run at runtime.
    let src = r#"namespace t; class c {
        int main() {
            int a = 6;
            int b = a * 7;
            printf("{0}", b - 1);
        }
    }"#;
    let p = lower(src);
    let main = program_main(&p);
    // Both operations touch locals: the Mul reads `a`, the Sub reads `b`.
    // Neither may fold — folding here was unsound with loops in the
    // language — but the two literals themselves are direct Consts.
    assert_eq!(
        count_arith(main),
        2,
        "exactly Mul(a,7) and Sub(b,1) remain:\n{main:#?}"
    );
}

#[test]
fn division_by_zero_and_min_minus_one_stay_runtime() {
    let p = lower(
        "namespace t; class c { int main() { printf(\"{0} {1}\", 5 / 0, -9223372036854775808 / -1); } }",
    );
    // i64::MIN literal parses via unary neg of 9223372036854775808 (folds),
    // then / -1 must NOT fold — it stays a runtime Bin.
    let main = program_main(&p);
    assert!(
        f_blocks(main).iter().any(|i| matches!(
            i,
            Instr::Bin {
                op: crate::native::ir::BinOp::Div,
                ..
            }
        )),
        "undefined division must remain runtime"
    );
}

#[test]
fn non_constant_operands_are_not_folded() {
    let p = lower(
        r#"namespace t { class c { int Get(int v) { return v; }
        int main() { printf("{0}", Get(2) * 3); } } }"#,
    );
    // A call result is a runtime value: the multiply must SURVIVE.
    assert_eq!(count_arith(&program_main(&p).clone()), 1);
    assert!(f_blocks(program_main(&p)).iter().any(|i| matches!(
        i,
        Instr::Bin {
            op: crate::native::ir::BinOp::Mul,
            ..
        }
    )));
}

// ---- const fields (compile-time constants, no storage) ----

fn const_ints(f: &crate::native::ir::IrFunction) -> Vec<i64> {
    f_blocks(f)
        .iter()
        .filter_map(|i| match i {
            Instr::Const { imm, .. } => Some(*imm),
            _ => None,
        })
        .collect()
}

#[test]
fn const_int_field_folds_to_literal_not_static_read() {
    let p = lower(
        "namespace t; class c { const int X = 42; int main() { printf(\"{0}\", c.X); } }",
    );
    let main = program_main(&p);
    assert!(
        const_ints(main).contains(&42),
        "c.X must fold to Const(42); consts: {:#?}, IR:\n{main:#?}",
        const_ints(main)
    );
    assert!(
        !f_blocks(main).iter().any(|i| matches!(i, Instr::LoadStaticAddr { .. })),
        "a const field must not be read from static storage:\n{main:#?}"
    );
}

#[test]
fn const_field_arithmetic_folds_to_computed_value() {
    let p = lower(
        "namespace t; class c { const int X = 2 * 3 + 4; int main() { printf(\"{0}\", c.X); } }",
    );
    let main = program_main(&p);
    assert!(
        const_ints(main).contains(&10),
        "2*3+4 must fold to Const(10); consts: {:#?}",
        const_ints(main)
    );
}

#[test]
fn const_field_negative_and_bitwise_fold() {
    // i32::MIN and a bitwise expression both fold at analysis time.
    let p = lower(
        "namespace t; class c { const int A = -2147483648; const int B = 0xff & 0x0f; int main() { printf(\"{0} {1}\", c.A, c.B); } }",
    );
    let main = program_main(&p);
    let c = const_ints(main);
    assert!(c.contains(&-2147483648i64), "A must fold to i32::MIN; consts: {c:#?}");
    assert!(c.contains(&15), "0xff & 0x0f must fold to 15; consts: {c:#?}");
}

#[test]
fn const_field_non_constant_init_is_an_error() {
    // A call is not a compile-time constant, so the field cannot be folded.
    let src = "namespace t; class c { static int Get() { return 1; } const int X = Get(); int main() { return 0; } }";
    let unit = crate::parser::parse(src).unwrap();
    let errs = crate::native::semantic::analyze(&unit).unwrap_err();
    assert!(
        errs.iter().any(|e| e.code == "E0030"),
        "expected an E0030 const-initializer error; got {errs:#?}"
    );
}

// ---- helpers ----
fn program_main(p: &Program) -> &crate::native::ir::IrFunction {
    // By name, not position: synthetic helpers (e.g. the placeholder
    // vtable stub) may trail entry functions.
    p.functions
        .iter()
        .find(|f| f.name.contains("main"))
        .expect("main lowers")
}
fn f_blocks(f: &crate::native::ir::IrFunction) -> &[Instr] {
    &f.blocks[0].instrs
}
/// Arithmetic Bins only (Add/Sub/Mul/Div/Rem): `int` results now carry a
/// narrowing `Shl`/`Shr` pair, which these shape tests must not count as
/// a second arithmetic operation.
fn count_arith(main: &crate::native::ir::IrFunction) -> usize {
    f_blocks(main)
        .iter()
        .filter(|i| {
            matches!(
                i,
                Instr::Bin {
                    op: crate::native::ir::BinOp::Add
                        | crate::native::ir::BinOp::Sub
                        | crate::native::ir::BinOp::Mul
                        | crate::native::ir::BinOp::Div
                        | crate::native::ir::BinOp::Rem,
                    ..
                }
            )
        })
        .count()
}

fn count_bins(f: &crate::native::ir::IrFunction) -> usize {
    f_blocks(f)
        .iter()
        .filter(|i| matches!(i, Instr::Bin { .. }))
        .count()
}

#[test]
fn mixed_int_float_const_folds() {
    // `s_DUINT_BASE2 * s_DUINT_BASE` (SQLDecimal powers of two): one
    // side folds from an int literal, the other is a double — C#
    // promotes int → double. Without mixed folding the chain stalls
    // and E0030 fires on `s_DUINT_BASE3`. The `long`→`double`
    // declaration conversion matters too: without it the stored Int
    // 2**32 overflows `checked_mul` instead of computing 2**64.
    // Unwrap asserts lowering (incl. the const fixpoint) succeeds;
    // without the fixes, E0030 fails the build.
    let _ = lower(
        r#"namespace t;
class C {
    private const long B = ((long)1) << 32;
    private const double U = B;
    private const double U2 = U * U;
    private const double U3 = U2 * U;
    double M() { return U3; }
    int Main() { return 0; }
}"#,
    );
}

#[test]
fn unqualified_same_class_const_folds_not_instance_read() {
    // Bare `X` (no `this.`, no `c.`) must fold like the qualified
    // spelling — previously it lowered to a vtable-slot read.
    // No printf here: its argument spill introduces legitimate
    // LoadFields from the temporary box, which the no-storage assertion
    // would misread as a const storage read.
    let p = lower(
        "namespace t; class c { const int X = 42; int main() { int y = X; return y; } }",
    );
    let main = program_main(&p);
    assert!(
        const_ints(main).contains(&42),
        "bare X must fold to Const(42); consts: {:#?}",
        const_ints(main)
    );
    assert!(
        !f_blocks(main).iter().any(|i| matches!(i, Instr::LoadField { .. })),
        "a const field has no storage to read:\n{main:#?}"
    );
}

#[test]
fn nested_class_reads_enclosing_int_const() {
    // A nested class sees enclosing privates (nesting family):
    // previously this failed as an undeclared variable.
    let p = lower(
        r#"namespace t;
class Outer {
    private const int K = 7;
    public class Inner {
        public static int Get() { return K; }
    }
}
class c { int main() { printf("{0}", Outer.Inner.Get()); } }"#,
    );
    let main = program_main(&p);
    assert!(
        const_ints(main).contains(&7) || count_bins(main) == 0,
        "enclosing K must reach the call without an undeclared error:\n{main:#?}"
    );
}

#[test]
fn nested_class_reads_enclosing_string_const() {
    // Mirrors XmlBaseReader.XmlNode reading the outer xmlnsNamespace:
    // the string materializes like a literal, never an instance read.
    let p = lower(
        r#"namespace t;
class Outer {
    private const string Ns = "http://www.w3.org/2000/xmlns/";
    public class Inner {
        public static bool IsNs(string s) { return s == Ns; }
    }
}
class c { int main() { printf("{0}", Outer.Inner.IsNs("x")); } }"#,
    );
    let get = p
        .functions
        .iter()
        .find(|f| f.name.contains("IsNs"))
        .expect("nested IsNs lowers");
    assert!(
        f_blocks(get).iter().any(|i| matches!(i, Instr::LoadStrAddr { .. })),
        "enclosing string const must materialize:\n{get:#?}"
    );
    assert!(
        !f_blocks(get).iter().any(|i| matches!(i, Instr::LoadStaticAddr { .. })),
        "a const has no static storage to read:\n{get:#?}"
    );
}

#[test]
fn qualified_nested_static_call_emits_direct_call() {
    // `Outer.Inner.Seven()`: the dotted receiver denotes a TYPE, so
    // the call must lower to a direct CallFn — previously the receiver
    // typed as `object`, selection missed, and the call folded to zero
    // (nested statics always returned 0 at runtime).
    let src = r#"namespace t;
class Outer {
    public class Inner {
        public static int Seven() { return 7; }
    }
}
class c { int main() { printf("{0}", Outer.Inner.Seven()); } }"#;
    let unit = crate::parser::parse(src).unwrap();
    let p = crate::native::ir::lower(&unit).unwrap();
    let main = p.functions.iter().find(|f| f.name.contains("main")).expect("main lowers");
    // Static-style call lowers direct (CallStatic/CallFn), never a
    // zero placeholder.
    let fidx = main
        .blocks
        .iter()
        .flat_map(|b| &b.instrs)
        .filter_map(|i| match i {
            crate::native::ir::Instr::CallStatic { fidx, .. } => Some(*fidx),
            crate::native::ir::Instr::CallFn { fidx, .. } => Some(*fidx),
            _ => None,
        })
        .next()
        .expect("a direct call is emitted");
    let target = &p.functions[fidx as usize];
    assert!(
        target.name.contains("Seven"),
        "call must target Seven, got {}",
        target.name
    );
}
