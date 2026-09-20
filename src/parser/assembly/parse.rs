//! Statement parser: turns the token stream into [`AsmItem`]s.

use super::lex::Tok;


#[derive(Debug, Clone, PartialEq)]
pub enum AsmOperand {
    Reg(Reg),
    Imm(i64),
    Mem { base: Option<Reg>, index: Option<Reg>, scale: u8, disp: i64 },
    Label(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AsmStmt {
    Instr { mnem: String, ops: Vec<AsmOperand> },
    Label(String),
    SectionData,
    SectionText,
    Global(String),
    Equ { name: String, value: i64 },
}

#[derive(Debug, Clone, PartialEq)]
pub enum AsmItem {
    Text(AsmStmt),
    Data(AsmStmt),
}

pub fn reg_by_name(n: &str) -> Option<Reg> {
    Some(match n.to_ascii_lowercase().as_str() {
        "rax" | "eax" | "ax" | "al" => Reg::Rax,
        "rcx" | "ecx" | "cx" | "cl" => Reg::Rcx,
        "rdx" | "edx" | "dx" | "dl" => Reg::Rdx,
        "rbx" | "ebx" | "bx" | "bl" => Reg::Rbx,
        "rsp" | "esp" | "sp" => Reg::Rsp,
        "rbp" | "ebp" | "bp" => Reg::Rbp,
        "rsi" | "esi" | "si" => Reg::Rsi,
        "rdi" | "edi" | "di" => Reg::Rdi,
        "r8" => Reg::R8, "r9" => Reg::R9, "r10" => Reg::R10, "r11" => Reg::R11,
        "r12" => Reg::R12, "r13" => Reg::R13, "r14" => Reg::R14, "r15" => Reg::R15,
        _ => return None,
    })
}

// Re-export Reg from the parent module

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Reg {
    Rax, Rcx, Rdx, Rbx, Rsp, Rbp, Rsi, Rdi,
    R8, R9, R10, R11, R12, R13, R14, R15,
}

impl Reg {
    pub fn id(self) -> u8 { self as u8 }
    pub fn needs_rex_b(self) -> bool { self.id() >= 8 }
}

pub fn parse(source: &str) -> Result<Vec<AsmItem>, String> {
    let toks = super::lex::lex(source)?;
    let (items, _) = parse_toks(&toks)?;
    Ok(items)
}

pub fn parse_toks(toks: &[Tok]) -> Result<(Vec<AsmItem>, ()), String> {
    let mut items = Vec::new();
    let mut in_data = false;
    let mut i = 0usize;
    let n = toks.len();

    while i < n {
        match &toks[i] {
            Tok::Newline => { i += 1; }
            Tok::Ident(d) if d == ".data" => { in_data = true; items.push(AsmItem::Text(AsmStmt::SectionData)); i += 1; skip_eol(toks, &mut i); }
            Tok::Ident(d) if d == ".text" => { in_data = false; items.push(AsmItem::Text(AsmStmt::SectionText)); i += 1; skip_eol(toks, &mut i); }
            Tok::Ident(d) if d == "section" || d == ".data" || d == ".text" => {
                // Section directives: track but don't affect encoding (flat blob).
                if d == "section" { i += 1; }  // skip 'section', next token is the name
                i += 1;  // skip the name (.data/.text)
                skip_eol(toks, &mut i);
            }
            Tok::Ident(d) if d == "global" => {
                i += 1;
                if let Some(Tok::Ident(name)) = toks.get(i) {
                    items.push(tag(AsmStmt::Global(name.clone()), in_data));
                    i += 1;
                } else { return Err("expected name after global".into()); }
                skip_eol(toks, &mut i);
            }
            Tok::Ident(name) if toks.get(i + 1) == Some(&Tok::Colon) => {
                items.push(tag(AsmStmt::Label(name.clone()), in_data));
                i += 2;
            }
            Tok::Ident(mnem) => {
                let m = mnem.clone();
                i += 1;
                let mut ops = Vec::new();
                while i < n && toks[i] != Tok::Newline {
                    // Skip NASM size prefixes before memory operands
                    if let Tok::Ident(sz) = &toks[i]
                        && matches!(sz.as_str(), "byte" | "word" | "dword" | "qword")
                        && toks.get(i + 1) == Some(&Tok::LBracket)
                    {
                        i += 1;
                    }
                    ops.push(parse_operand(toks, &mut i)?);
                    if i < n && toks[i] == Tok::Comma { i += 1; continue; }
                    break;
                }
                items.push(tag(AsmStmt::Instr { mnem: m, ops }, in_data));
                skip_eol(toks, &mut i);
            }
            other => return Err(format!("unexpected token {other:?}")),
        }
    }
    Ok((items, ()))
}

fn tag(stmt: AsmStmt, in_data: bool) -> AsmItem {
    if in_data { AsmItem::Data(stmt) } else { AsmItem::Text(stmt) }
}

fn skip_eol(toks: &[Tok], i: &mut usize) {
    while *i < toks.len() && toks[*i] == Tok::Newline { *i += 1; }
}

fn parse_operand(toks: &[Tok], i: &mut usize) -> Result<AsmOperand, String> {
    match toks.get(*i) {
        Some(Tok::LBracket) => {
            *i += 1;
            let mut base = None;
            let mut index: Option<Reg> = None;
            let mut scale = 1u8;
            let mut disp = 0i64;
            loop {
                match toks.get(*i) {
                    Some(Tok::RBracket) => { *i += 1; break; }
                    Some(Tok::Plus) => { *i += 1; }
                    Some(Tok::Star) => { *i += 1;
                        if let Some(Tok::Int(v)) = toks.get(*i) { scale = *v as u8; *i += 1; }
                    }
                    Some(Tok::Int(v)) => { disp += v; *i += 1; }
                    Some(Tok::Minus) => { *i += 1;
                        if let Some(Tok::Int(v)) = toks.get(*i) { disp -= v; *i += 1; }
                    }
                    Some(Tok::Ident(r)) if reg_by_name(r).is_some() && index.is_none() && toks.get(*i+1) != Some(&Tok::Star) && base.is_some() => {
                        index = reg_by_name(r); *i += 1;
                    }
                    Some(Tok::Ident(r)) if reg_by_name(r).is_some() && base.is_none() => {
                        base = reg_by_name(r); *i += 1;
                    }
                    _ => return Err("unexpected token in memory operand".into()),
                }
            }
            Ok(AsmOperand::Mem { base, index, scale, disp })
        }
        Some(Tok::Int(v)) => { let v = *v; *i += 1; Ok(AsmOperand::Imm(v)) }
        Some(Tok::Minus) => { *i += 1;
            match toks.get(*i) {
                Some(Tok::Int(v)) => { let v = *v; *i += 1; Ok(AsmOperand::Imm(-v)) }
                _ => Err("expected number after '-'".into()),
            }
        }
        Some(Tok::Ident(r)) if reg_by_name(r).is_some() => {
            let r = reg_by_name(r).unwrap(); *i += 1; Ok(AsmOperand::Reg(r))
        }
        Some(Tok::Ident(l)) => { let l = l.clone(); *i += 1; Ok(AsmOperand::Label(l)) }
        _ => Err("expected operand".into()),
    }
}
