//! Floating-point lowering: float literals, arithmetic, comparison, and
//! negation must emit the IEEE 754 `F*` instructions, integer expressions
//! keep the integer `Bin`/`Cmp`/`Neg` path, and a float program survives the
//! bytecode round-trip structurally.

use crate::models::compare::StructuralEq;
use crate::native::ir::{FBinOp, Instr, Program};

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

// Every float value feeds the next, so none is dead; the final float
// comparisons drive the returned integer, keeping the whole chain live.
const SRC: &str = r#"namespace t; class c {
    int main() {
        float a = 1.5;
        float b = 2.25;
        float s = a + b;
        float m = s * b;
        float d = m / b;
        float sub = d - b;
        float neg = -sub;
        float prod = neg * a;
        int gt = prod > a;
        int eq = a == b;
        return gt + eq;
    }
}
"#;

#[test]
fn float_literal_emits_fconst() {
    let p = lower(SRC);
    assert!(
        has(&p, |i| matches!(i, Instr::FConst { bits, .. } if *bits == 1.5f64.to_bits())),
        "FConst(1.5) must be present"
    );
    assert!(
        has(&p, |i| matches!(i, Instr::FConst { bits, .. } if *bits == 2.25f64.to_bits())),
        "FConst(2.25) must be present"
    );
}

#[test]
fn float_arithmetic_emits_fbin_for_all_four_ops() {
    let p = lower(SRC);
    for op in [FBinOp::Add, FBinOp::Sub, FBinOp::Mul, FBinOp::Div] {
        let expected = op;
        assert!(
            has(&p, |i| matches!(i, Instr::FBin { op, .. } if *op == expected)),
            "FBin {expected:?} must be present"
        );
    }
}

#[test]
fn float_comparison_emits_fcmp() {
    let p = lower(SRC);
    assert!(
        has(&p, |i| matches!(i, Instr::FCmp { .. })),
        "FCmp must be present for the float comparisons"
    );
}

#[test]
fn float_negation_emits_fneg() {
    let p = lower(SRC);
    assert!(
        has(&p, |i| matches!(i, Instr::FNeg { .. })),
        "FNeg must be present for `-sub`"
    );
}

#[test]
fn integer_arithmetic_stays_on_the_integer_path() {
    let src = r#"namespace t; class c {
        int main() {
            int x = 3 * 4;
            int y = x - 1;
            int z = x + 1;
            int w = x / 2;
            return y + z + w;
        }
    }
    "#;
    let p = lower(src);
    assert!(
        !has(&p, |i| matches!(i, Instr::FBin { .. })),
        "integer arithmetic must not emit FBin"
    );
    assert!(
        !has(&p, |i| matches!(i, Instr::FCmp { .. })),
        "integer arithmetic must not emit FCmp"
    );
}

#[test]
fn float_program_survives_bytecode_roundtrip() {
    let src = "namespace t; class c {\n    int main() {\n        float a = 1.5;\n        float b = 2.25;\n        float s = a * b - 0.5;\n        int g = s > 1.0;\n        return g;\n    }\n}\n";
    let unit = crate::parser::parse(src).unwrap();
    let encoded = crate::bytecode::encode(&unit);
    let decoded = crate::bytecode::decode(&encoded).unwrap();
    assert!(
        unit.structural_eq(&decoded),
        "float program must round-trip structurally"
    );
}

#[test]
fn decimal_lowes_as_64bit_scalar() {
    // `decimal` is modeled as a 64-bit scalar (true 128-bit decimal
    // semantics are future work): locals declare and arithmetic
    // takes the float path, unblocking decimal-heavy code.
    let p = lower(
        r#"namespace t; class c {
    int main() {
        decimal d = 1.5;
        decimal e = d + 2.5;
        int g = e > 1.0;
        return g;
    }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::FBin { .. })),
        "expected decimal arithmetic on the float path"
    );
}
