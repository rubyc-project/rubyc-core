//! First-class `string` lowering. A string object is laid out as
//! `[0..8)=len, [8..16)=refcount, [16..)=utf8 data`, so:
//! - `s[i]` reads one byte at `obj + 16 + i` -> `LoadByte { off: 0 }` over a
//!   `base + 16 + i` address (byte granularity, not the 8-byte `LoadField`
//!   array path).
//! - `a + b` allocates a runtime-sized object of `len_a + len_b + 16` bytes
//!   (`AllocDyn`), stores the total length at `+0`, refcount `1` at `+8`, and
//!   `MemCpy`s both payloads into `+16`.

use std::collections::{HashMap, HashSet};

use crate::native::ir::{Instr, Program, VReg};

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

/// `dst` register produced by `i`, if any.
fn dst_of(i: &Instr) -> Option<VReg> {
    match i {
        Instr::Const { dst, .. }
        | Instr::FConst { dst, .. }
        | Instr::Neg { dst, .. }
        | Instr::FNeg { dst, .. }
        | Instr::Cmp { dst, .. }
        | Instr::FCmp { dst, .. }
        | Instr::FBin { dst, .. }
        | Instr::Alloc { dst, .. }
        | Instr::AllocDyn { dst, .. }
        | Instr::StackAlloc { dst, .. }
        | Instr::VTableAddr { dst, .. }
        | Instr::LoadStaticAddr { dst, .. }
        | Instr::LoadField { dst, .. }
        | Instr::LoadByte { dst, .. }
        | Instr::LoadStrAddr { dst, .. }
        | Instr::MemCpy { dst, .. }
        | Instr::MemCmp { dst, .. }
        | Instr::ScanInt { dst, .. }
        | Instr::HandlerEnter { dst, .. }
        | Instr::LoadExc { dst, .. } => Some(*dst),
        Instr::Bin { dst, .. } => Some(*dst),
        _ => None,
    }
}

/// Map each `VReg` to the instruction that defines it.
fn defs(p: &Program) -> HashMap<u32, &Instr> {
    let mut m = HashMap::new();
    for i in instrs(p) {
        if let Some(dst) = dst_of(i) {
            m.insert(dst.0, i);
        }
    }
    m
}

/// True if `vreg`'s value is (transitively) built from the `Const { imm }`.
fn reaches_const(defs: &HashMap<u32, &Instr>, vreg: VReg, imm: i64) -> bool {
    let mut seen = HashSet::new();
    let mut stack = vec![vreg];
    while let Some(v) = stack.pop() {
        if !seen.insert(v.0) {
            continue;
        }
        let Some(def) = defs.get(&v.0) else {
            continue;
        };
        match def {
            Instr::Const { imm: n, .. } if *n == imm => return true,
            Instr::Bin { a, b, .. } => {
                stack.push(*a);
                stack.push(*b);
            }
            _ => {}
        }
    }
    false
}

const SRC_INDEX: &str = r#"class c {
    Main() {
        string s = "hello";
        int a = s[0];
        int b = s[4];
        return a + b;
    }
}
"#;

#[test]
fn string_index_emits_loadbyte() {
    let p = lower(SRC_INDEX);
    assert!(
        has(&p, |i| matches!(i, Instr::LoadByte { off, .. } if *off == 0)),
        "`s[i]` on a string must lower to a byte-granularity LoadByte (off = 0)"
    );
}

#[test]
fn string_index_reads_past_the_16_byte_header() {
    let p = lower(SRC_INDEX);
    let defs = defs(&p);
    // The LoadByte's base must be an address computed as `obj + 16 + i`, i.e.
    // it must be built from the `Const { imm: 16 }` data offset.
    let base = p
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .find_map(|i| match i {
            Instr::LoadByte { base, .. } => Some(*base),
            _ => None,
        })
        .expect("a LoadByte base must exist for s[i]");
    assert!(
        reaches_const(&defs, base, 16),
        "the LoadByte base (obj + 16 + i) must incorporate the 16-byte string header offset"
    );
}

#[test]
fn string_index_is_not_an_8_byte_field_load() {
    // The `+16` header offset is a string-specific detail; the byte load must
    // carry the offset in its base, not in a separate `LoadField` of a char.
    let p = lower(SRC_INDEX);
    // A `LoadField` at offset 16 would indicate the old (wrong) 8-byte path.
    assert!(
        !has(&p, |i| matches!(i, Instr::LoadField { off, .. } if *off == 16)),
        "string indexing must not fall back to a 16-byte LoadField"
    );
}

const SRC_CONCAT: &str = r#"class c {
    Main() {
        string a = "he";
        string b = "llo";
        string c = a + b;
        return c.Length;
    }
}
"#;

#[test]
fn string_concat_allocates_a_dynamic_object() {
    let p = lower(SRC_CONCAT);
    assert!(
        has(&p, |i| matches!(i, Instr::AllocDyn { .. })),
        "`a + b` on strings must allocate a runtime-sized object (AllocDyn)"
    );
}

#[test]
fn string_concat_size_is_total_plus_header() {
    let p = lower(SRC_CONCAT);
    let defs = defs(&p);
    // AllocDyn's byte count must be `(len_a + len_b) + 16`, so it must be
    // built from the `Const { imm: 16 }` header size.
    let bytes = p
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .find_map(|i| match i {
            Instr::AllocDyn { bytes, .. } => Some(*bytes),
            _ => None,
        })
        .expect("an AllocDyn byte count must exist for a + b");
    assert!(
        reaches_const(&defs, bytes, 16),
        "the concat allocation size must be (len_a + len_b) + 16"
    );
}

#[test]
fn string_concat_stores_refcount_at_offset_8() {
    let p = lower(SRC_CONCAT);
    assert!(
        has(&p, |i| matches!(i, Instr::StoreField { off, .. } if *off == 8)),
        "concat must store the refcount (1) at object offset 8"
    );
}

#[test]
fn string_concat_copies_both_payloads() {
    let p = lower(SRC_CONCAT);
    let copies = p
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(i, Instr::MemCpy { .. }))
        .count();
    assert!(
        copies >= 2,
        "concat must memcpy both payloads into the new object; got {copies}"
    );
}
