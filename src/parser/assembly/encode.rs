//! Two-pass encoder: layout + label resolution, then byte emission.

use super::parse::{AsmItem, AsmOperand, AsmStmt, Reg};

/// Registers encoded as (rex_b, id).
fn rexb(r: Reg) -> bool {
    r.id() >= 8
}

fn modrm(mod_: u8, reg: u8, rm: u8) -> u8 {
    ((mod_ & 3) << 6) | ((reg & 7) << 3) | (rm & 7)
}

fn rex(w: bool, r: bool, x: bool, b: bool) -> u8 {
    let mut v = 0x40u8;
    if w { v |= 0x08 }
    if r { v |= 0x04 }
    if x { v |= 0x02 }
    if b { v |= 0x01 }
    v
}

pub fn encode(items: &[AsmItem]) -> Result<Vec<u8>, String> {
    encode_with_symbols(items).map(|(b, _)| b)
}

type EncodeResult = Result<(Vec<u8>, Vec<(String, usize)>), String>;

pub fn encode_with_symbols(items: &[AsmItem]) -> EncodeResult {
    // Pass 1: sizes. Every instruction has a computable size once we know
    // whether branch displacements are rel8 or rel32 — assume the larger
    // when a label is unresolved, then shrink in pass 2 if it fits.
    //
    // Simpler correct approach used here: single buffer, emit with
    // placeholder rel32 for label branches, then patch — rel32 always
    // encodable regardless of distance.

    let mut code: Vec<u8> = Vec::new();
    let mut labels: Vec<(String, usize)> = Vec::new();
    let mut fixups: Vec<(usize, String, FixKind)> = Vec::new();
    let mut equ: std::collections::BTreeMap<String, i64> = Default::default();

    #[derive(Clone, Copy)]
    enum FixKind {
        Rel32,
        #[allow(dead_code)]
        Abs64,
    }


    for item in items {
        let stmt = match item {
            AsmItem::Text(s) | AsmItem::Data(s) => s,
        };
        match stmt {
            AsmStmt::Label(name) => {
                labels.push((name.clone(), code.len()));
            }
            AsmStmt::Equ { name, value } => {
                equ.insert(name.clone(), *value);
            }
            AsmStmt::SectionData | AsmStmt::SectionText | AsmStmt::Global(_) => {
                // v1: single flat .text blob; data directives inline.
            }

            AsmStmt::Instr { mnem, ops } => {
                let m = mnem.as_str();
                match (m, ops.as_slice()) {
                    ("nop", []) => code.push(0x90),
                    ("ret", []) => code.push(0xC3),
                    ("syscall", []) => code.extend_from_slice(&[0x0F, 0x05]),
                    ("cdq" | "cqo", []) => {
                        // sign-extend eax/rax into edx/rax
                        code.extend_from_slice(&[0x48, 0x99]);
                    }
                    ("push", [AsmOperand::Reg(r)]) => {
                        code.push(0x50 | (r.id() & 7));
                        if rexb(*r) && r.id() < 8 {}
                    }
                    ("pop", [AsmOperand::Reg(r)]) => {
                        code.push(0x58 | (r.id() & 7));
                    }
                    ("inc", [AsmOperand::Reg(r)]) => {
                        code.extend_from_slice(&[0x48, 0xFF]);
                        code.push(modrm(0b11, 0, r.id()));
                    }
                    ("dec", [AsmOperand::Reg(r)]) => {
                        code.extend_from_slice(&[0x48, 0xFF]);
                        code.push(modrm(0b11, 1, r.id()));
                    }
                    ("neg", [AsmOperand::Reg(r)]) => {
                        code.extend_from_slice(&[rex(true, false, false, rexb(*r)), 0xF7]);
                        code.push(modrm(0b11, 3, r.id()));
                    }
                    ("not", [AsmOperand::Reg(r)]) => {
                        code.extend_from_slice(&[rex(true, false, false, rexb(*r)), 0xF7]);
                        code.push(modrm(0b11, 2, r.id()));
                    }
                    ("imul", [AsmOperand::Reg(d), AsmOperand::Reg(a), AsmOperand::Imm(v)]) => {
                        // imul r64, r64, imm32 (sign-extended imm8 form when fits)
                        code.extend_from_slice(&[rex(true, rexb(*d), false, rexb(*a))]);
                        if *v >= -128 && *v <= 127 {
                            code.extend_from_slice(&[0x6B]);
                            code.push(modrm(0b11, d.id(), a.id()));
                            code.push(*v as i8 as u8);
                        } else {
                            code.extend_from_slice(&[0x69]);
                            code.push(modrm(0b11, d.id(), a.id()));
                            code.extend_from_slice(&v.to_le_bytes());
                        }
                    }
                    ("idiv" | "div", [AsmOperand::Reg(r)]) => {
                        code.extend_from_slice(&[rex(true, false, false, rexb(*r)), 0xF7]);
                        code.push(modrm(0b11, if m == "idiv" { 7 } else { 6 }, r.id()));
                    }
                    ("mul", [AsmOperand::Reg(r)]) => {
                        code.extend_from_slice(&[rex(true, false, false, rexb(*r)), 0xF7]);
                        code.push(modrm(0b11, 4, r.id()));
                    }
                    ("shl" | "shr" | "sar", [AsmOperand::Reg(d), AsmOperand::Reg(Reg::Rcx)]) => {
                        // count in cl: D3 /op with REX.W
                        code.extend_from_slice(&[0x48, 0xD3]);
                        let op = match m { "shl" => 4, "shr" => 5, _ => 7 };
                        code.push(modrm(0b11, op, d.id()));
                    }
                    ("shl" | "shr" | "sar", [AsmOperand::Reg(d), AsmOperand::Imm(n)]) => {
                        code.extend_from_slice(&[0x48, 0xC1]);
                        let op = match m { "shl" => 4, "shr" => 5, _ => 7 };
                        code.push(modrm(0b11, op, d.id()));
                        code.push(*n as u8);
                    }
                    ("mov", ops) => mov_enc(&mut code, ops)?,
                    ("lea", [AsmOperand::Reg(d), AsmOperand::Mem { base, index, scale, disp }]) => {
                        emit_mem(&mut code, 0x8D, *d, *base, *index, *scale, *disp);
                    }
                    (
                        "add" | "sub" | "and" | "or" | "xor" | "cmp" | "test",
                        [AsmOperand::Reg(d), AsmOperand::Reg(s)],
                    ) => {
                        // r64 OP r/m64 → opcode base per mnemonic (ADD=01/03 etc.)
                        // We use the "OP r/m, reg" form: 01/29/21/09/31/39/85
                        let opc: u8 = match m {
                            "add" => 0x01,
                            "sub" => 0x29,
                            "and" => 0x21,
                            "or" => 0x09,
                            "xor" => 0x31,
                            "cmp" => 0x39,
                            _ => 0x85, // test
                        };
                        code.push(rex(true, rexb(*d), false, rexb(*s)));
                        code.push(opc);
                        code.push(modrm(0b11, d.id(), s.id()));
                    }
                    (
                        "add" | "sub" | "and" | "or" | "xor" | "cmp",
                        [AsmOperand::Reg(d), AsmOperand::Imm(v)],
                    ) => {
                        // OP r/m64, imm32 (sign-ext) or imm8 compact form
                        let ext: Option<u8> = match m {
                            "add" => Some(0x83),
                            "sub" => Some(0x83),
                            "and" => Some(0x83),
                            "or" => Some(0x83),
                            "xor" => Some(0x83),
                            "cmp" => Some(0x83),
                            _ => None,
                        };
                        let _ = ext;
                        let (opc8, opc32): (u8, u8) = match m {
                            "add" => (0x83, 0x81),
                            "sub" => (0x83, 0x81),
                            "and" => (0x83, 0x81),
                            "or" => (0x83, 0x81),
                            "xor" => (0x83, 0x81),
                            "cmp" => (0x83, 0x81),
                            _ => unreachable!(),
                        };
                        let _ = opc8;
                        let reg_field: u8 = match m {
                            "add" => 0,
                            "sub" => 5,
                            "and" => 4,
                            "or" => 1,
                            "xor" => 6,
                            "cmp" => 7,
                            _ => 0,
                        };
                        code.push(rex(true, false, false, rexb(*d)));
                        if *v >= -128 && *v <= 127 {
                            code.push(opc8);
                            code.push(modrm(0b11, reg_field, d.id()));
                            code.push(*v as i8 as u8);
                        } else {
                            code.push(opc32);
                            code.push(modrm(0b11, reg_field, d.id()));
                            code.extend_from_slice(&(*v as i32).to_le_bytes());
                        }
                    }
                    (
                        "jmp" | "call",
                        [AsmOperand::Label(l)],
                    ) => {
                        let is_call = m == "call";
                        code.push(if is_call { 0xE8 } else { 0xE9 });
                        let site = code.len();
                        code.extend_from_slice(&0i32.to_le_bytes());
                        fixups.push((site, l.clone(), FixKind::Rel32));
                        let _ = is_call;
                    }
                    (cc, [AsmOperand::Label(l)])
                        if matches!(
                            cc,
                            "je" | "jne" | "jl" | "jge" | "jle" | "jg" | "jb" | "jae"
                                | "jbe" | "ja" | "js" | "jns"
                        ) =>
                    {
                        let cc_byte: u8 = match cc {
                            "je" => 0x84,
                            "jne" => 0x85,
                            "jl" => 0x8C,
                            "jge" => 0x8D,
                            "jle" => 0x8E,
                            "jg" => 0x8F,
                            "jb" => 0x82,
                            "jae" => 0x83,
                            "jbe" => 0x86,
                            "ja" => 0x87,
                            "js" => 0x88,
                            _ => 0x89, // jns
                        };
                        code.extend_from_slice(&[0x0F, cc_byte]);
                        let site = code.len();
                        code.extend_from_slice(&0i32.to_le_bytes());
                        fixups.push((site, l.clone(), FixKind::Rel32));
                    }
                    _ => {
                        return Err(format!("unsupported operand forms for '{m}'"));
                    }
                }
            }
        }
    }

    // ---- patch label fixups
    for (site, label, kind) in &fixups {
        let target = labels
            .iter()
            .find(|(l, _)| l == label)
            .map(|(_, off)| *off)
            .or_else(|| equ.get(label).map(|v| *v as usize))
            .ok_or_else(|| format!("undefined symbol '{label}'"))?;
        match kind {
            FixKind::Rel32 => {
                let rel = (target as isize - (site + 4) as isize) as i32;
                code[*site..*site + 4].copy_from_slice(&rel.to_le_bytes());
            }
            FixKind::Abs64 => {
                code[*site..*site + 8].copy_from_slice(&(target as u64).to_le_bytes());
            }
        }
    }

    Ok((code, labels))
}

fn mov_enc(code: &mut Vec<u8>, ops: &[AsmOperand]) -> Result<(), String> {
    match ops {
        // mov r64, imm64 → movabs
        [AsmOperand::Reg(d), AsmOperand::Imm(v)] if *v < i32::MIN as i64 || *v > i32::MAX as i64 => {
            code.extend_from_slice(&[rex(true, false, false, rexb(*d)), 0xB8 | (d.id() & 7)]);
            code.extend_from_slice(&v.to_le_bytes());
            Ok(())
        }
        // mov r64, imm32 (sign-extended)
        [AsmOperand::Reg(d), AsmOperand::Imm(v)] => {
            code.extend_from_slice(&[rex(true, false, false, rexb(*d)), 0xC7]);
            code.push(modrm(0b11, 0, d.id()));
            code.extend_from_slice(&(*v as i32).to_le_bytes());
            Ok(())
        }
        // mov r64, r64
        [AsmOperand::Reg(d), AsmOperand::Reg(s)] => {
            code.push(rex(true, rexb(*d), false, rexb(*s)));
            code.push(0x89);
            code.push(modrm(0b11, s.id(), d.id())); // mov d, s
            Ok(())
        }
        // mov reg, [mem]
        [AsmOperand::Reg(d), AsmOperand::Mem { base, index, scale, disp }] => {
            emit_mem(code, 0x8B, *d, *base, *index, *scale, *disp);
            Ok(())
        }
        // mov [mem], reg
        [AsmOperand::Mem { base, index, scale, disp }, AsmOperand::Reg(s)] => {
            emit_mem(code, 0x89, *s, *base, *index, *scale, *disp);
            Ok(())
        }
        // mov [mem], imm32 — always use ModRM+SIB+disp32 form
        [AsmOperand::Mem { base: Some(b), .. }, AsmOperand::Imm(v)] => {
            code.push(rex(true, false, false, rexb(*b)));
            code.push(0xC7);
            // ModRM mod=00 reg=0 rm=100(SIB), SIB ss=00 idx=100(base=rsp)
            if *b == Reg::Rsp {
                code.push(modrm(0b00, 0, 0b100));
                code.push(0x24);
            } else {
                code.push(modrm(0b00, 0, b.id() & 7));
            }
            code.extend_from_slice(&v.to_le_bytes());
            Ok(())
        }
        [AsmOperand::Mem { .. }, AsmOperand::Imm(_)] => {
            Err("mov [mem],imm requires a base register".into())
        }
        other => Err(format!("bad mov operands {other:?}")),
    }
}

/// Emit ModRM/SIB/displacement for `opcode reg_field` against a memory operand.
fn emit_mem(
    code: &mut Vec<u8>,
    opcode: u8,
    reg: Reg,
    base: Option<Reg>,
    index: Option<Reg>,
    scale: u8,
    disp: i64,
) {
    code.push(rex(true, rexb(reg), index.map(|i| i.needs_rex_b()).unwrap_or(false), base.map(|b| b.needs_rex_b()).unwrap_or(false)));
    code.push(opcode);
    match (base, index) {
        (Some(b), None) => {
            // [base] or [base+disp]. rsp base REQUIRES SIB; rbp requires
            // an explicit displacement byte even when zero.
            if b == Reg::Rsp && disp == 0 {
                code.push(modrm(0b00, reg.id(), 0b100));
                code.push(0x24);
                return;
            }
            let no_disp_ok = disp == 0 && b != Reg::Rbp;
            let (mod_, dsize): (u8, usize) = if no_disp_ok {
                (0b00, 0)
            } else if (-128..=127).contains(&disp) {
                (0b01, 1)
            } else {
                (0b10, 4)
            };
            if b == Reg::Rsp {
                code.push(modrm(mod_, reg.id(), 0b100));
                code.push(0x24);
            } else {
                code.push(modrm(mod_, reg.id(), b.id() & 7));
            }
            match dsize {
                1 => code.push(disp as i8 as u8),
                4 => code.extend_from_slice(&disp.to_le_bytes()),
                _ => {}
            }
        }
        (Some(b), idx) => {
            let idx = idx.or(if b == Reg::Rsp { Some(Reg::Rax) } else { None });
            let sib_scale = match scale { 1=>0,2=>1,4=>2,8=>3,_=>0 };
            code.push(modrm(0b10, reg.id(), 0b100));
            code.push(((sib_scale & 3) << 6) | ((idx.unwrap().id() & 7) << 3) | (b.id() & 7));
            code.extend_from_slice(&disp.to_le_bytes());
        }
        (None, Some(idx)) => {
            code.push(modrm(0b00, reg.id(), 0b100));
            let sib_scale = match scale { 1=>0,2=>1,4=>2,8=>3,_=>0 };
            code.push(((sib_scale & 3) << 6) | ((idx.id() & 7) << 3) | 0b101);
            code.extend_from_slice(&disp.to_le_bytes());
        }
        (None, None) => {
            // absolute addressing: [disp32] via SIB with base=none
            code.push(modrm(0b00, reg.id(), 0b100));
            code.push(0x25); // SIB: no index, base=101
            code.extend_from_slice(&disp.to_le_bytes());
        }
    }
}

