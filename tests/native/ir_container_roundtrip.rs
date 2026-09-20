//! IR container round-trip: `encode_program` → `decode_program` must preserve
//! the full `Program` (the plugin ABI boundary, decision D3), and corrupt
//! input must fail cleanly. Mirrors the bytecode round-trip suite.

use crate::bytecode::traits::ToBytecode;
use crate::models::DecodeError;
use crate::native::container::{self, Kind};
use crate::native::ir::{
    decode_program, encode_program, BinOp, Block, CmpOp, Instr, IrFunction,
    Program, ReceiverKind, Terminator, VReg, IR_FORMAT_VERSION,
};

fn vreg(n: u32) -> VReg {
    VReg(n)
}

/// A synthetic program exercising every `Instr` and `Terminator` variant, all
/// `BinOp`/`CmpOp` arms, plus vtables, imports, exports, a string blob, and
/// a non-zero heap size.
fn exhaustive_program() -> Program {
    let all_bins = |dst: u32, a: u32, b: u32| -> Vec<Instr> {
        let ops = [
            BinOp::Add,
            BinOp::Sub,
            BinOp::Mul,
            BinOp::Div,
            BinOp::Rem,
            BinOp::BitAnd,
            BinOp::BitOr,
            BinOp::And,
            BinOp::Or,
            BinOp::BitXor,
            BinOp::Shl,
            BinOp::Shr,
        ];
        ops.iter()
            .enumerate()
            .map(|(i, op)| Instr::Bin {
                dst: vreg(dst + i as u32),
                op: *op,
                a: vreg(a),
                b: vreg(b),
            })
            .collect()
    };

    let all_cmps = |dst: u32, a: u32, b: u32| -> Vec<Instr> {
        [
            CmpOp::Eq,
            CmpOp::Ne,
            CmpOp::Lt,
            CmpOp::Gt,
            CmpOp::Le,
            CmpOp::Ge,
        ]
        .iter()
        .enumerate()
        .map(|(i, op)| Instr::Cmp {
            dst: vreg(dst + i as u32),
            op: *op,
            a: vreg(a),
            b: vreg(b),
        })
        .collect()
    };

    let main = IrFunction {
        name: "main".into(),
        receiver: ReceiverKind::None,
        arity: 0,
        vregs: 64,
        is_async: false,
        blocks: vec![
            Block {
                instrs: vec![
                    Instr::Const {
                        dst: vreg(0),
                        imm: i64::MIN,
                    },
                    Instr::Mov {
                        dst: vreg(1),
                        a: vreg(0),
                    },
                    Instr::Raw(vec![0x90, 0x48, 0xC3]),
                    Instr::Neg {
                        dst: vreg(2),
                        a: vreg(0),
                    },
                    Instr::CallFn {
                        dst: vreg(3),
                        fidx: 1,
                        args: vec![vreg(0), vreg(1)],
                    },
                    Instr::CallVirt {
                        dst: vreg(4),
                        slot: 2,
                        recv: vreg(0),
                        args: vec![vreg(1)],
                    },
                    Instr::CallImport {
                        dst: vreg(5),
                        import: 0,
                        args: vec![],
                    },
                    Instr::Alloc {
                        dst: vreg(6),
                        bytes: u64::MAX,
                    },
                    Instr::VTableAddr {
                        dst: vreg(7),
                        row: 0,
                    },
                    Instr::LoadField {
                        dst: vreg(8),
                        base: vreg(6),
                        off: 0x10,
                    },
                    Instr::StoreField {
                        base: vreg(6),
                        off: 0x10,
                        src: vreg(8),
                    },
                    Instr::WriteStr { off: 0, len: 3 },
                    Instr::WriteInt { src: vreg(0) },
                    Instr::EWriteInt { src: vreg(1) },
                    Instr::EWriteStr { off: 3, len: 2 },
                    Instr::ScanInt { dst: vreg(9) },
                    Instr::Retain { src: vreg(6) },
                    Instr::Release {
                        src: vreg(6),
                        bytes: 48,
                    },
                    Instr::Exit,
                ],
                term: Terminator::Ret(Some(vreg(0))),
            },
            Block {
                instrs: all_bins(10, 0, 1),
                term: Terminator::Ret(None),
            },
            Block {
                instrs: all_cmps(30, 0, 1),
                term: Terminator::Exit,
            },
            Block {
                instrs: vec![],
                term: Terminator::ExitWith(vreg(1)),
            },
            Block {
                instrs: vec![],
                term: Terminator::Branch {
                    cond: Some(vreg(2)),
                    if_true: 1,
                    if_false: 3,
                },
            },
            Block {
                instrs: vec![],
                term: Terminator::Branch {
                    cond: None,
                    if_true: 0,
                    if_false: 4,
                },
            },
        ],
    };

    let other = IrFunction {
        name: "Other::method".into(),
        receiver: ReceiverKind::This,
        arity: 2,
        vregs: 4,
        is_async: false,
        blocks: vec![Block {
            instrs: vec![Instr::Const {
                dst: vreg(0),
                imm: -42,
            }],
            term: Terminator::Ret(Some(vreg(0))),
        }],
    };

    Program {
        functions: vec![other, main],
        statics: vec![],
        vtables: vec![
            ("Animal".into(), vec![0, 1, usize::MAX]),
            ("Empty".into(), vec![]),
        ],
        imports: vec![("printf".into(), "libc".into()), ("empty".into(), String::new())],
        exports: vec![("main".into(), 1, 0), ("Other::method".into(), 0, 2)],
        strings: b"hello\x00world\x00".to_vec(),
        heap_size: 0xDEADBEEF_CAFEBABE,
        names: vec!["add".to_string(), "sub".to_string()],
        entry: Some(0),
        init: None,
    }
}

#[test]
fn exhaustive_program_round_trips() {
    let p = exhaustive_program();
    let bytes = encode_program(&p);
    let decoded = decode_program(&bytes).expect("exhaustive program must decode");
    assert_eq!(p, decoded, "Program changed across IR round-trip");
}

#[test]
fn reencoding_is_deterministic() {
    let p = exhaustive_program();
    let a = encode_program(&p);
    let b = encode_program(&decode_program(&a).unwrap());
    assert_eq!(a, b, "re-encoding the decoded program differs");
}

#[test]
fn lowered_programs_round_trip() {
    let samples = [
        "namespace t; class c { int main() { printf(\"{0}\", 2 * 3 + 1); } }",
        r#"namespace t { class c { int Get(int v) { return v; }
            int main() { printf("{0}", Get(2) * 3); } } }"#,
        r#"namespace t { class c {
            int main() {
                int a = 6;
                int b = a * 7;
                if (b > 10) { printf("{0}", b); } else { printf("{0}", 1); }
            }
        } }"#,
    ];
    for src in samples {
        let unit = crate::parser::parse(src).unwrap();
        let p = crate::native::ir::lower(&unit).unwrap();
        let bytes = encode_program(&p);
        let decoded = decode_program(&bytes).unwrap();
        assert_eq!(p, decoded, "lowered program changed: {src}");
    }
}

#[test]
fn header_is_jump_dog_kind_ir_version() {
    let bytes = encode_program(&exhaustive_program());
    assert_eq!(&bytes[..2], &container::JUMP_OVER_MAGIC);
    assert_eq!(&bytes[2..10], &container::MAGIC);
    assert_eq!(bytes[10], Kind::Ir.to_byte());
    assert_eq!(&bytes[11..13], &IR_FORMAT_VERSION.to_le_bytes());
}

#[test]
fn bad_magic_is_rejected() {
    let bytes = vec![b'X', 0, 0, 0, 1, 0];
    assert_eq!(decode_program(&bytes), Err(DecodeError::BadMagic));
}

#[test]
fn wrong_kind_is_rejected() {
    // A valid bytecode container is not an IR container.
    let unit = crate::parser::parse("namespace t; class c { int main() {} }").unwrap();
    let bytes = unit.to_bytecode();
    assert_eq!(decode_program(&bytes), Err(DecodeError::WrongKind));
}

#[test]
fn unsupported_version_is_rejected() {
    let mut bytes = container::build(Kind::Ir, 999);
    bytes.push(0x01); // PROGRAM tag so the version check is what fails
    assert_eq!(
        decode_program(&bytes),
        Err(DecodeError::UnsupportedVersion(999))
    );
}

#[test]
fn truncated_input_is_rejected() {
    let bytes = encode_program(&exhaustive_program());
    for cut in 13..bytes.len() {
        assert!(
            decode_program(&bytes[..cut]).is_err(),
            "truncated at {cut} unexpectedly decoded"
        );
    }
}

#[test]
fn trailing_bytes_are_rejected() {
    let mut bytes = encode_program(&exhaustive_program());
    bytes.push(0xFF);
    assert_eq!(decode_program(&bytes), Err(DecodeError::TrailingBytes));
}

#[test]
fn unknown_root_tag_is_rejected() {
    let mut bytes = container::build(Kind::Ir, IR_FORMAT_VERSION);
    bytes.push(0xEE);
    assert_eq!(
        decode_program(&bytes),
        Err(DecodeError::UnknownNodeTag(0xEE))
    );
}

#[test]
fn unknown_instruction_tag_is_rejected() {
    // Valid PROGRAM header, one function, one block, then a bogus instr tag.
    let mut bytes = container::build(Kind::Ir, IR_FORMAT_VERSION);
    bytes.push(0x01); // PROGRAM
    bytes.extend_from_slice(&1u32.to_le_bytes()); // 1 function
    bytes.extend_from_slice(&4u32.to_le_bytes()); // name len
    bytes.extend_from_slice(b"main");
    bytes.push(0); // receiver (None)
    bytes.extend_from_slice(&0u64.to_le_bytes()); // arity
    bytes.extend_from_slice(&1u32.to_le_bytes()); // vregs
    bytes.extend_from_slice(&1u32.to_le_bytes()); // 1 block
    bytes.extend_from_slice(&1u32.to_le_bytes()); // 1 instr
    bytes.push(0xEE); // bogus instr tag
    assert_eq!(
        decode_program(&bytes),
        Err(DecodeError::UnknownNodeTag(0xEE))
    );
}
