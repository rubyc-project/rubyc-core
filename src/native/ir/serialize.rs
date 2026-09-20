//! Lowered `Program` ↔ portable IR container bytes.
//!
//! This is the plugin ABI boundary (decision D3): a target backend plugin
//! receives the canonical IR as a versioned 🐕🇨 container, never a Rust IR
//! type and never source text. The layout is the same little-endian,
//! `u32`-length-prefixed style as the AST bytecode container.

use crate::models::DecodeError;
use crate::native::container::{self as container_mod, Kind};
use super::{
    Block, BinOp, CmpOp, FBinOp, FCmpOp, Instr, IrFunction, IrStatic, Program, ReceiverKind,
    Terminator, VReg,
};

/// Current IR container format version written by the encoder.
///
/// v1: full portable IR — functions, blocks, terminators, vtable rows,
/// `Program.statics` table, and the `LoadStaticAddr` instruction
/// (static field storage).
pub const IR_FORMAT_VERSION: u16 = 2;

/// Node tags. One byte per node, children follow recursively.
pub mod tag {
    pub const PROGRAM: u8 = 0x01;

    /// `Option` prefix: absent / present.
    pub const ABSENT: u8 = 0x00;
    pub const PRESENT: u8 = 0x01;

    // Instruction tags.
    pub const CONST: u8 = 0x10;
    pub const MOV: u8 = 0x11;
    pub const RAW: u8 = 0x12;
    pub const BIN: u8 = 0x13;
    pub const CMP: u8 = 0x14;
    pub const NEG: u8 = 0x15;
    pub const CALL_FN: u8 = 0x16;
    pub const CALL_VIRT: u8 = 0x17;
    pub const CALL_IMPORT: u8 = 0x18;
    pub const ALLOC: u8 = 0x19;
    pub const VTABLE_ADDR: u8 = 0x1a;
    pub const LOAD_FIELD: u8 = 0x1b;
    pub const STORE_FIELD: u8 = 0x1c;
    pub const WRITE_STR: u8 = 0x1d;
    pub const WRITE_INT: u8 = 0x1e;
    pub const EWRITE_INT: u8 = 0x1f;
    pub const EWRITE_STR: u8 = 0x20;
    pub const SCAN_INT: u8 = 0x21;
    pub const RETAIN: u8 = 0x22;
    pub const RELEASE: u8 = 0x23;
    pub const EXIT: u8 = 0x24;
    // v2 instruction tags.
    pub const CALL_BY_NAME: u8 = 0x25;
    pub const FCONST: u8 = 0x26;
    pub const FBIN: u8 = 0x27;
    pub const FCMP: u8 = 0x28;
    pub const FNEG: u8 = 0x29;
    pub const STACK_ALLOC: u8 = 0x2a;
    pub const DROP: u8 = 0x2b;
    // Exception-handling instruction tags.
    pub const HANDLER_ENTER: u8 = 0x2c;
    pub const HANDLER_POP: u8 = 0x2d;
    pub const LOAD_EXC: u8 = 0x2e;
    pub const CLEAR_INFLIGHT: u8 = 0x2f;
    // v2 async instruction tags.
    pub const TASK_ALLOC: u8 = 0x30;
    pub const SUSPEND: u8 = 0x31;
    pub const RESUME_TASK: u8 = 0x32;
    // v2 static-call tag: direct call to a ReceiverKind::This function with no
    // receiver (a static method). The backend passes 0 in the receiver register.
    pub const CALL_STATIC: u8 = 0x34;
    // v4 static-field tag: load the absolute address of static slot `idx`
    // into `dst` (the slot lives in the data section).
    pub const LOAD_STATIC_ADDR: u8 = 0x35;
    // String/memory primitives: first-class string support.
    pub const LOAD_STR_ADDR: u8 = 0x36;
    pub const MEM_COPY: u8 = 0x37;
    pub const MEM_CMP: u8 = 0x38;
    pub const ALLOC_DYN: u8 = 0x39;
    pub const WRITE_STRING: u8 = 0x3a;
    pub const LOAD_BYTE: u8 = 0x3b;
    // Dynamic-linking address import: dst = runtime address of
    // `Program::imports[import]` (vtable row / static slot).
    pub const LOAD_IMPORT: u8 = 0x3c;
    // Process argv as `string[]` (startup-saved stack slot; empty
    // array where no slot exists, e.g. JIT).
    pub const BUILD_ARGV: u8 = 0x3d;

    // Terminator tags.
    pub const T_RET: u8 = 0x30;
    pub const T_EXIT: u8 = 0x31;
    pub const T_EXIT_WITH: u8 = 0x32;
    pub const T_BRANCH: u8 = 0x33;
    pub const T_UNWIND: u8 = 0x34;
    pub const T_RETHROW: u8 = 0x35;
}

/// Encode a lowered program into a portable IR container (🐕🇨, `Kind::Ir`).
pub fn encode_program(prog: &Program) -> Vec<u8> {
    let mut out = container_mod::build(Kind::Ir, IR_FORMAT_VERSION);
    let mut w = Writer { out: &mut out };
    w.program(prog);
    out
}

/// Encode a program to raw payload bytes with no RubyC preamble.
///
/// Used by `--no-magic`: the bytes are the bare `PROGRAM` payload (starting
/// with the `TAG_PROGRAM` byte), with no `jump-over-magic` + dog + kind/version
/// header. [`decode_program`] is tolerant and auto-detects the absent preamble.
pub fn encode_program_no_magic(prog: &Program) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let mut w = Writer { out: &mut out };
    w.program(prog);
    out
}

/// Decode a portable IR container back into a lowered program. Every read is
/// bounds-checked: malformed input yields a [`DecodeError`], never a panic.
///
/// Tolerant of a missing RubyC preamble: if the standard header is absent
/// (the `--no-magic` form), the bytes are treated as a raw `PROGRAM` payload
/// and decoded directly.
pub fn decode_program(bytes: &[u8]) -> Result<Program, DecodeError> {
    match container_mod::parse(bytes) {
        Ok(parsed) => {
            if parsed.kind != Kind::Ir {
                return Err(DecodeError::WrongKind);
            }
            if parsed.version != IR_FORMAT_VERSION {
                return Err(DecodeError::UnsupportedVersion(parsed.version));
            }
            let mut r = Reader {
                bytes,
                pos: parsed.payload_start,
            };
            let prog = r.program()?;
            if r.pos != bytes.len() {
                return Err(DecodeError::TrailingBytes);
            }
            Ok(prog)
        }
        // Preamble absent (or unparseable): fall back to the raw `--no-magic`
        // payload, which begins with the `TAG_PROGRAM` byte.
        Err(_) => decode_raw_program(bytes),
    }
}

/// Decode a raw `PROGRAM` payload that carries no RubyC preamble (`--no-magic`).
///
/// The payload begins with the `TAG_PROGRAM` byte. We re-attach a valid
/// preamble in memory and reuse the standard decode path so there is a single
/// payload-decode implementation.
fn decode_raw_program(bytes: &[u8]) -> Result<Program, DecodeError> {
    if bytes.is_empty() || bytes[0] != tag::PROGRAM {
        return Err(DecodeError::BadMagic);
    }
    let mut with_preamble = container_mod::build(Kind::Ir, IR_FORMAT_VERSION);
    with_preamble.extend_from_slice(bytes);
    let payload_start = container_mod::PREAMBLE_LEN + 3;
    let mut r = Reader {
        bytes: &with_preamble,
        pos: payload_start,
    };
    let prog = r.program()?;
    if r.pos != with_preamble.len() {
        return Err(DecodeError::TrailingBytes);
    }
    Ok(prog)
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

struct Writer<'a> {
    out: &'a mut Vec<u8>,
}

impl<'a> Writer<'a> {
    fn byte(&mut self, b: u8) {
        self.out.push(b);
    }
    fn u32(&mut self, v: u32) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }
    fn bytes(&mut self, b: &[u8]) {
        self.out.extend_from_slice(b);
    }
    fn string(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.out.extend_from_slice(s.as_bytes());
    }
    #[allow(dead_code)]
    fn bool8(&mut self, b: bool) {
        self.byte(b as u8);
    }
    fn vreg(&mut self, v: VReg) {
        self.u32(v.0);
    }
    fn opt_vreg(&mut self, v: &Option<VReg>) {
        match v {
            None => self.byte(tag::ABSENT),
            Some(x) => {
                self.byte(tag::PRESENT);
                self.vreg(*x);
            }
        }
    }
    fn opt_u32(&mut self, v: &Option<u32>) {
        match v {
            None => self.byte(tag::ABSENT),
            Some(x) => {
                self.byte(tag::PRESENT);
                self.u32(*x);
            }
        }
    }

    fn program(&mut self, p: &Program) {
        self.byte(tag::PROGRAM);
        self.u32(p.functions.len() as u32);
        for f in &p.functions {
            self.function(f);
        }
        // v4: static field storage table.
        self.u32(p.statics.len() as u32);
        for s in &p.statics {
            self.string(&s.name);
            self.string(&s.class);
            self.string(&s.type_name);
        }
        self.u32(p.vtables.len() as u32);
        for (name, slots) in &p.vtables {
            self.string(name);
            self.u32(slots.len() as u32);
            for s in slots {
                self.u64(*s as u64);
            }
        }
        self.u32(p.imports.len() as u32);
        for (name, lib) in &p.imports {
            self.string(name);
            self.string(lib);
        }
        self.u32(p.exports.len() as u32);
        for (sym, fidx, arity) in &p.exports {
            self.string(sym);
            self.u64(*fidx as u64);
            self.u64(*arity as u64);
        }
        self.u32(p.strings.len() as u32);
        self.bytes(&p.strings);
        self.u64(p.heap_size);
        // v2: names table for CallByName.
        self.u32(p.names.len() as u32);
        for name in &p.names {
            self.string(name);
        }
        // v2: entry point.
        self.opt_u32(&p.entry.map(|e| e as u32));
        // v2: @init function index for shared libraries.
        self.opt_u32(&p.init.map(|e| e as u32));
    }

    fn function(&mut self, f: &IrFunction) {
        self.string(&f.name);
        // v2: ReceiverKind instead of takes_this bool.
        self.byte(receiver_byte(f.receiver));
        self.u64(f.arity as u64);
        self.u32(f.vregs);
        self.u32(f.blocks.len() as u32);
        for b in &f.blocks {
            self.block(b);
        }
        // v2 async flag.
        self.byte(f.is_async as u8);
    }

    fn block(&mut self, b: &Block) {
        self.u32(b.instrs.len() as u32);
        for i in &b.instrs {
            self.instr(i);
        }
        self.terminator(&b.term);
    }

    fn instr(&mut self, i: &Instr) {
        match i {
            Instr::Const { dst, imm } => {
                self.byte(tag::CONST);
                self.vreg(*dst);
                self.i64(*imm);
            }
            Instr::FConst { dst, bits } => {
                self.byte(tag::FCONST);
                self.vreg(*dst);
                self.u64(*bits);
            }
            Instr::Mov { dst, a } => {
                self.byte(tag::MOV);
                self.vreg(*dst);
                self.vreg(*a);
            }
            Instr::Raw(bytes) => {
                self.byte(tag::RAW);
                self.u32(bytes.len() as u32);
                self.bytes(bytes);
            }
            Instr::Bin { dst, op, a, b } => {
                self.byte(tag::BIN);
                self.vreg(*dst);
                self.byte(binop_byte(*op));
                self.vreg(*a);
                self.vreg(*b);
            }
            Instr::FBin { dst, op, a, b } => {
                self.byte(tag::FBIN);
                self.vreg(*dst);
                self.byte(fbinop_byte(*op));
                self.vreg(*a);
                self.vreg(*b);
            }
            Instr::Cmp { dst, op, a, b } => {
                self.byte(tag::CMP);
                self.vreg(*dst);
                self.byte(cmpop_byte(*op));
                self.vreg(*a);
                self.vreg(*b);
            }
            Instr::FCmp { dst, op, a, b } => {
                self.byte(tag::FCMP);
                self.vreg(*dst);
                self.byte(fcmpop_byte(*op));
                self.vreg(*a);
                self.vreg(*b);
            }
            Instr::Neg { dst, a } => {
                self.byte(tag::NEG);
                self.vreg(*dst);
                self.vreg(*a);
            }
            Instr::FNeg { dst, a } => {
                self.byte(tag::FNEG);
                self.vreg(*dst);
                self.vreg(*a);
            }
            Instr::CallFn { dst, fidx, args } => {
                self.byte(tag::CALL_FN);
                self.vreg(*dst);
                self.u32(*fidx);
                self.vreg_list(args);
            }
            Instr::CallStatic { dst, fidx, args } => {
                self.byte(tag::CALL_STATIC);
                self.vreg(*dst);
                self.u32(*fidx);
                self.vreg_list(args);
            }
            Instr::CallVirt { dst, slot, recv, args } => {
                self.byte(tag::CALL_VIRT);
                self.vreg(*dst);
                self.u32(*slot);
                self.vreg(*recv);
                self.vreg_list(args);
            }
            Instr::CallByName { dst, recv, name, args } => {
                self.byte(tag::CALL_BY_NAME);
                self.vreg(*dst);
                self.vreg(*recv);
                self.u32(*name);
                self.vreg_list(args);
            }
            Instr::CallImport { dst, import, args } => {
                self.byte(tag::CALL_IMPORT);
                self.vreg(*dst);
                self.u32(*import);
                self.vreg_list(args);
            }
            Instr::LoadImport { dst, import } => {
                self.byte(tag::LOAD_IMPORT);
                self.vreg(*dst);
                self.u32(*import);
            }
            Instr::Alloc { dst, bytes } => {
                self.byte(tag::ALLOC);
                self.vreg(*dst);
                self.u64(*bytes);
            }
            Instr::StackAlloc { dst, bytes } => {
                self.byte(tag::STACK_ALLOC);
                self.vreg(*dst);
                self.u64(*bytes);
            }
            Instr::VTableAddr { dst, row } => {
                self.byte(tag::VTABLE_ADDR);
                self.vreg(*dst);
                self.u32(*row);
            }
            Instr::LoadStaticAddr { dst, idx } => {
                self.byte(tag::LOAD_STATIC_ADDR);
                self.vreg(*dst);
                self.u32(*idx);
            }
            Instr::LoadField { dst, base, off } => {
                self.byte(tag::LOAD_FIELD);
                self.vreg(*dst);
                self.vreg(*base);
                self.u32(*off);
            }
            Instr::StoreField { base, off, src } => {
                self.byte(tag::STORE_FIELD);
                self.vreg(*base);
                self.u32(*off);
                self.vreg(*src);
            }
            Instr::LoadByte { dst, base, off } => {
                self.byte(tag::LOAD_BYTE);
                self.vreg(*dst);
                self.vreg(*base);
                self.u32(*off);
            }
            Instr::WriteStr { off, len } => {
                self.byte(tag::WRITE_STR);
                self.u32(*off);
                self.u32(*len);
            }
            Instr::LoadStrAddr { dst, off } => {
                self.byte(tag::LOAD_STR_ADDR);
                self.vreg(*dst);
                self.u32(*off);
            }
            Instr::MemCpy { dst, src, len } => {
                self.byte(tag::MEM_COPY);
                self.vreg(*dst);
                self.vreg(*src);
                self.vreg(*len);
            }
            Instr::MemCmp { dst, a, b, len } => {
                self.byte(tag::MEM_CMP);
                self.vreg(*dst);
                self.vreg(*a);
                self.vreg(*b);
                self.vreg(*len);
            }
            Instr::AllocDyn { dst, bytes } => {
                self.byte(tag::ALLOC_DYN);
                self.vreg(*dst);
                self.vreg(*bytes);
            }
            Instr::BuildArgv { dst } => {
                self.byte(tag::BUILD_ARGV);
                self.vreg(*dst);
            }
            Instr::WriteString { src } => {
                self.byte(tag::WRITE_STRING);
                self.vreg(*src);
            }
            Instr::WriteInt { src } => {
                self.byte(tag::WRITE_INT);
                self.vreg(*src);
            }
            Instr::EWriteInt { src } => {
                self.byte(tag::EWRITE_INT);
                self.vreg(*src);
            }
            Instr::EWriteStr { off, len } => {
                self.byte(tag::EWRITE_STR);
                self.u32(*off);
                self.u32(*len);
            }
            Instr::ScanInt { dst } => {
                self.byte(tag::SCAN_INT);
                self.vreg(*dst);
            }
            Instr::Retain { src } => {
                self.byte(tag::RETAIN);
                self.vreg(*src);
            }
            Instr::Release { src, bytes } => {
                self.byte(tag::RELEASE);
                self.vreg(*src);
                self.u64(*bytes);
            }
            Instr::Drop { src } => {
                self.byte(tag::DROP);
                self.vreg(*src);
            }
            Instr::HandlerEnter { dst } => {
                self.byte(tag::HANDLER_ENTER);
                self.vreg(*dst);
            }
            Instr::HandlerPop => {
                self.byte(tag::HANDLER_POP);
            }
            Instr::LoadExc { dst } => {
                self.byte(tag::LOAD_EXC);
                self.vreg(*dst);
            }
            Instr::ClearInflight => {
                self.byte(tag::CLEAR_INFLIGHT);
            }
            Instr::Exit => {
                self.byte(tag::EXIT);
            }
            Instr::TaskAlloc { dst, bytes } => {
                self.byte(tag::TASK_ALLOC);
                self.vreg(*dst);
                self.u64(*bytes);
            }
            Instr::Suspend { state } => {
                self.byte(tag::SUSPEND);
                self.vreg(*state);
            }
            Instr::CoroutineResume { task, dst } => {
                self.byte(tag::RESUME_TASK);
                self.vreg(*task);
                self.vreg(*dst);
            }
        }
    }

    fn vreg_list(&mut self, args: &[VReg]) {
        self.u32(args.len() as u32);
        for a in args {
            self.vreg(*a);
        }
    }

    fn terminator(&mut self, t: &Terminator) {
        match t {
            Terminator::Ret(v) => {
                self.byte(tag::T_RET);
                self.opt_vreg(v);
            }
            Terminator::Exit => {
                self.byte(tag::T_EXIT);
            }
            Terminator::ExitWith(v) => {
                self.byte(tag::T_EXIT_WITH);
                self.vreg(*v);
            }
            Terminator::Branch {
                cond,
                if_true,
                if_false,
            } => {
                self.byte(tag::T_BRANCH);
                self.opt_vreg(cond);
                self.u32(*if_true);
                self.u32(*if_false);
            }
            Terminator::Unwind { src } => {
                self.byte(tag::T_UNWIND);
                self.vreg(*src);
            }
            Terminator::Rethrow => {
                self.byte(tag::T_RETHROW);
            }
        }
    }
}

fn binop_byte(op: BinOp) -> u8 {
    match op {
        BinOp::Add => 0,
        BinOp::Sub => 1,
        BinOp::Mul => 2,
        BinOp::Div => 3,
        BinOp::Rem => 4,
        BinOp::BitAnd => 5,
        BinOp::BitOr => 6,
        BinOp::And => 7,
        BinOp::Or => 8,
        BinOp::BitXor => 9,
        BinOp::Shl => 10,
        BinOp::Shr => 11,
        BinOp::ShrShr => 12,
        BinOp::Is => 13,
    }
}

fn cmpop_byte(op: CmpOp) -> u8 {
    match op {
        CmpOp::Eq => 0,
        CmpOp::Ne => 1,
        CmpOp::Lt => 2,
        CmpOp::Gt => 3,
        CmpOp::Le => 4,
        CmpOp::Ge => 5,
    }
}

fn fbinop_byte(op: FBinOp) -> u8 {
    match op {
        FBinOp::Add => 0,
        FBinOp::Sub => 1,
        FBinOp::Mul => 2,
        FBinOp::Div => 3,
    }
}

fn fcmpop_byte(op: FCmpOp) -> u8 {
    match op {
        FCmpOp::Eq => 0,
        FCmpOp::Ne => 1,
        FCmpOp::Lt => 2,
        FCmpOp::Gt => 3,
        FCmpOp::Le => 4,
        FCmpOp::Ge => 5,
    }
}

fn receiver_byte(r: ReceiverKind) -> u8 {
    match r {
        ReceiverKind::None => 0,
        ReceiverKind::This => 1,
        ReceiverKind::Explicit => 2,
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn byte(&mut self) -> Result<u8, DecodeError> {
        let b = *self.bytes.get(self.pos).ok_or(DecodeError::UnexpectedEof)?;
        self.pos += 1;
        Ok(b)
    }
    fn u32(&mut self) -> Result<u32, DecodeError> {
        let end = self.pos.checked_add(4).ok_or(DecodeError::UnexpectedEof)?;
        if end > self.bytes.len() {
            return Err(DecodeError::UnexpectedEof);
        }
        let v = u32::from_le_bytes(self.bytes[self.pos..end].try_into().unwrap());
        self.pos = end;
        Ok(v)
    }
    fn u64(&mut self) -> Result<u64, DecodeError> {
        let end = self.pos.checked_add(8).ok_or(DecodeError::UnexpectedEof)?;
        if end > self.bytes.len() {
            return Err(DecodeError::UnexpectedEof);
        }
        let v = u64::from_le_bytes(self.bytes[self.pos..end].try_into().unwrap());
        self.pos = end;
        Ok(v)
    }
    fn i64(&mut self) -> Result<i64, DecodeError> {
        Ok(self.u64()? as i64)
    }
    fn bytes(&mut self, len: u32) -> Result<Vec<u8>, DecodeError> {
        let end = self
            .pos
            .checked_add(len as usize)
            .ok_or(DecodeError::UnexpectedEof)?;
        if end > self.bytes.len() {
            return Err(DecodeError::UnexpectedEof);
        }
        let v = self.bytes[self.pos..end].to_vec();
        self.pos = end;
        Ok(v)
    }
    fn string(&mut self) -> Result<String, DecodeError> {
        let len = self.u32()?;
        let raw = self.bytes(len)?;
        let s = String::from_utf8(raw).map_err(|_| DecodeError::InvalidUtf8)?;
        Ok(s)
    }
    #[allow(dead_code)]
    fn bool8(&mut self) -> Result<bool, DecodeError> {
        Ok(self.byte()? != 0)
    }
    fn vreg(&mut self) -> Result<VReg, DecodeError> {
        Ok(VReg(self.u32()?))
    }
    fn opt_vreg(&mut self) -> Result<Option<VReg>, DecodeError> {
        match self.byte()? {
            tag::ABSENT => Ok(None),
            tag::PRESENT => Ok(Some(self.vreg()?)),
            tag => Err(DecodeError::UnknownNodeTag(tag)),
        }
    }
    fn opt_u32(&mut self) -> Result<Option<u32>, DecodeError> {
        match self.byte()? {
            tag::ABSENT => Ok(None),
            tag::PRESENT => Ok(Some(self.u32()?)),
            tag => Err(DecodeError::UnknownNodeTag(tag)),
        }
    }
    fn vreg_list(&mut self) -> Result<Vec<VReg>, DecodeError> {
        let n = self.u32()? as usize;
        if n > self.bytes.len() - self.pos {
            return Err(DecodeError::UnexpectedEof);
        }
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(self.vreg()?);
        }
        Ok(v)
    }

    fn program(&mut self) -> Result<Program, DecodeError> {
        let t = self.byte()?;
        if t != tag::PROGRAM {
            return Err(DecodeError::UnknownNodeTag(t));
        }
        let nf = self.u32()? as usize;
        let mut functions = Vec::with_capacity(nf);
        for _ in 0..nf {
            functions.push(self.function()?);
        }
        // v4: static field storage table.
        let nst = self.u32()? as usize;
        let mut statics = Vec::with_capacity(nst);
        for _ in 0..nst {
            statics.push(IrStatic {
                name: self.string()?,
                class: self.string()?,
                type_name: self.string()?,
            });
        }
        let nvt = self.u32()? as usize;
        let mut vtables = Vec::with_capacity(nvt);
        for _ in 0..nvt {
            let name = self.string()?;
            let n = self.u32()? as usize;
            let mut slots = Vec::with_capacity(n);
            for _ in 0..n {
                slots.push(self.u64()? as usize);
            }
            vtables.push((name, slots));
        }
        let nimp = self.u32()? as usize;
        let mut imports = Vec::with_capacity(nimp);
        for _ in 0..nimp {
            imports.push((self.string()?, self.string()?));
        }
        let nexp = self.u32()? as usize;
        let mut exports = Vec::with_capacity(nexp);
        for _ in 0..nexp {
            exports.push((self.string()?, self.u64()? as usize, self.u64()? as usize));
        }
        let nstr = self.u32()? as usize;
        let strings = self.bytes(nstr as u32)?;
        let heap_size = self.u64()?;
        // v2: names table.
        let nname = self.u32()? as usize;
        let mut names = Vec::with_capacity(nname);
        for _ in 0..nname {
            names.push(self.string()?);
        }
        // v2: entry point.
        let entry = self.opt_u32()?.map(|e| e as usize);
        // v2: @init function index for shared libraries.
        let init = self.opt_u32()?.map(|e| e as usize);
        Ok(Program {
            functions,
            statics,
            vtables,
            imports,
            exports,
            strings,
            heap_size,
            names,
            entry,
            init,
        })
    }

    fn function(&mut self) -> Result<IrFunction, DecodeError> {
        let name = self.string()?;
        let receiver = self.byte()?;
        let receiver = receiver_from_byte(receiver)?;
        let arity = self.u64()? as usize;
        let vregs = self.u32()?;
        let nb = self.u32()? as usize;
        let mut blocks = Vec::with_capacity(nb);
        for _ in 0..nb {
            blocks.push(self.block()?);
        }
        let is_async = self.byte()? != 0;
        Ok(IrFunction {
            name,
            receiver,
            arity,
            blocks,
            vregs,
            is_async,
        })
    }

    fn block(&mut self) -> Result<Block, DecodeError> {
        let ni = self.u32()? as usize;
        let mut instrs = Vec::with_capacity(ni);
        for _ in 0..ni {
            instrs.push(self.instr()?);
        }
        let term = self.terminator()?;
        Ok(Block { instrs, term })
    }

    fn instr(&mut self) -> Result<Instr, DecodeError> {
        match self.byte()? {
            tag::CONST => {
                let dst = self.vreg()?;
                let imm = self.i64()?;
                Ok(Instr::Const { dst, imm })
            }
            tag::FCONST => {
                let dst = self.vreg()?;
                let bits = self.u64()?;
                Ok(Instr::FConst { dst, bits })
            }
            tag::MOV => {
                let dst = self.vreg()?;
                let a = self.vreg()?;
                Ok(Instr::Mov { dst, a })
            }
            tag::RAW => {
                let len = self.u32()?;
                let bytes = self.bytes(len)?;
                Ok(Instr::Raw(bytes))
            }
            tag::BIN => {
                let dst = self.vreg()?;
                let op = self.byte()?;
                let a = self.vreg()?;
                let b = self.vreg()?;
                Ok(Instr::Bin {
                    dst,
                    op: binop_from_byte(op)?,
                    a,
                    b,
                })
            }
            tag::FBIN => {
                let dst = self.vreg()?;
                let op = self.byte()?;
                let a = self.vreg()?;
                let b = self.vreg()?;
                Ok(Instr::FBin {
                    dst,
                    op: fbinop_from_byte(op)?,
                    a,
                    b,
                })
            }
            tag::CMP => {
                let dst = self.vreg()?;
                let op = self.byte()?;
                let a = self.vreg()?;
                let b = self.vreg()?;
                Ok(Instr::Cmp {
                    dst,
                    op: cmpop_from_byte(op)?,
                    a,
                    b,
                })
            }
            tag::FCMP => {
                let dst = self.vreg()?;
                let op = self.byte()?;
                let a = self.vreg()?;
                let b = self.vreg()?;
                Ok(Instr::FCmp {
                    dst,
                    op: fcmpop_from_byte(op)?,
                    a,
                    b,
                })
            }
            tag::NEG => {
                let dst = self.vreg()?;
                let a = self.vreg()?;
                Ok(Instr::Neg { dst, a })
            }
            tag::FNEG => {
                let dst = self.vreg()?;
                let a = self.vreg()?;
                Ok(Instr::FNeg { dst, a })
            }
            tag::CALL_FN => {
                let dst = self.vreg()?;
                let fidx = self.u32()?;
                let args = self.vreg_list()?;
                Ok(Instr::CallFn { dst, fidx, args })
            }
            tag::CALL_STATIC => {
                let dst = self.vreg()?;
                let fidx = self.u32()?;
                let args = self.vreg_list()?;
                Ok(Instr::CallStatic { dst, fidx, args })
            }
            tag::CALL_VIRT => {
                let dst = self.vreg()?;
                let slot = self.u32()?;
                let recv = self.vreg()?;
                let args = self.vreg_list()?;
                Ok(Instr::CallVirt { dst, slot, recv, args })
            }
            tag::CALL_BY_NAME => {
                let dst = self.vreg()?;
                let recv = self.vreg()?;
                let name = self.u32()?;
                let args = self.vreg_list()?;
                Ok(Instr::CallByName { dst, recv, name, args })
            }
            tag::CALL_IMPORT => {
                let dst = self.vreg()?;
                let import = self.u32()?;
                let args = self.vreg_list()?;
                Ok(Instr::CallImport { dst, import, args })
            }
            tag::LOAD_IMPORT => {
                let dst = self.vreg()?;
                let import = self.u32()?;
                Ok(Instr::LoadImport { dst, import })
            }
            tag::ALLOC => {
                let dst = self.vreg()?;
                let bytes = self.u64()?;
                Ok(Instr::Alloc { dst, bytes })
            }
            tag::STACK_ALLOC => {
                let dst = self.vreg()?;
                let bytes = self.u64()?;
                Ok(Instr::StackAlloc { dst, bytes })
            }
            tag::VTABLE_ADDR => {
                let dst = self.vreg()?;
                let row = self.u32()?;
                Ok(Instr::VTableAddr { dst, row })
            }
            tag::LOAD_STATIC_ADDR => {
                let dst = self.vreg()?;
                let idx = self.u32()?;
                Ok(Instr::LoadStaticAddr { dst, idx })
            }
            tag::LOAD_FIELD => {
                let dst = self.vreg()?;
                let base = self.vreg()?;
                let off = self.u32()?;
                Ok(Instr::LoadField { dst, base, off })
            }
            tag::STORE_FIELD => {
                let base = self.vreg()?;
                let off = self.u32()?;
                let src = self.vreg()?;
                Ok(Instr::StoreField { base, off, src })
            }
            tag::LOAD_BYTE => {
                let dst = self.vreg()?;
                let base = self.vreg()?;
                let off = self.u32()?;
                Ok(Instr::LoadByte { dst, base, off })
            }
            tag::WRITE_STR => {
                let off = self.u32()?;
                let len = self.u32()?;
                Ok(Instr::WriteStr { off, len })
            }
            tag::LOAD_STR_ADDR => {
                let dst = self.vreg()?;
                let off = self.u32()?;
                Ok(Instr::LoadStrAddr { dst, off })
            }
            tag::MEM_COPY => {
                let dst = self.vreg()?;
                let src = self.vreg()?;
                let len = self.vreg()?;
                Ok(Instr::MemCpy { dst, src, len })
            }
            tag::MEM_CMP => {
                let dst = self.vreg()?;
                let a = self.vreg()?;
                let b = self.vreg()?;
                let len = self.vreg()?;
                Ok(Instr::MemCmp { dst, a, b, len })
            }
            tag::ALLOC_DYN => {
                let dst = self.vreg()?;
                let bytes = self.vreg()?;
                Ok(Instr::AllocDyn { dst, bytes })
            }
            tag::BUILD_ARGV => {
                let dst = self.vreg()?;
                Ok(Instr::BuildArgv { dst })
            }
            tag::WRITE_STRING => {
                let src = self.vreg()?;
                Ok(Instr::WriteString { src })
            }
            tag::WRITE_INT => {
                let src = self.vreg()?;
                Ok(Instr::WriteInt { src })
            }
            tag::EWRITE_INT => {
                let src = self.vreg()?;
                Ok(Instr::EWriteInt { src })
            }
            tag::EWRITE_STR => {
                let off = self.u32()?;
                let len = self.u32()?;
                Ok(Instr::EWriteStr { off, len })
            }
            tag::SCAN_INT => {
                let dst = self.vreg()?;
                Ok(Instr::ScanInt { dst })
            }
            tag::RETAIN => {
                let src = self.vreg()?;
                Ok(Instr::Retain { src })
            }
            tag::RELEASE => {
                let src = self.vreg()?;
                let bytes = self.u64()?;
                Ok(Instr::Release { src, bytes })
            }
            tag::DROP => {
                let src = self.vreg()?;
                Ok(Instr::Drop { src })
            }
            tag::HANDLER_ENTER => {
                let dst = self.vreg()?;
                Ok(Instr::HandlerEnter { dst })
            }
            tag::HANDLER_POP => Ok(Instr::HandlerPop),
            tag::LOAD_EXC => {
                let dst = self.vreg()?;
                Ok(Instr::LoadExc { dst })
            }
            tag::CLEAR_INFLIGHT => Ok(Instr::ClearInflight),
            tag::EXIT => Ok(Instr::Exit),
            tag::TASK_ALLOC => {
                let dst = self.vreg()?;
                let bytes = self.u64()?;
                Ok(Instr::TaskAlloc { dst, bytes })
            }
            tag::SUSPEND => {
                let state = self.vreg()?;
                Ok(Instr::Suspend { state })
            }
            tag::RESUME_TASK => {
                let task = self.vreg()?;
                let dst = self.vreg()?;
                Ok(Instr::CoroutineResume { task, dst })
            }
            tag => Err(DecodeError::UnknownNodeTag(tag)),
        }
    }

    fn terminator(&mut self) -> Result<Terminator, DecodeError> {
        match self.byte()? {
            tag::T_RET => Ok(Terminator::Ret(self.opt_vreg()?)),
            tag::T_EXIT => Ok(Terminator::Exit),
            tag::T_EXIT_WITH => Ok(Terminator::ExitWith(self.vreg()?)),
            tag::T_BRANCH => {
                let cond = self.opt_vreg()?;
                let if_true = self.u32()?;
                let if_false = self.u32()?;
                Ok(Terminator::Branch {
                    cond,
                    if_true,
                    if_false,
                })
            }
            tag::T_UNWIND => {
                let src = self.vreg()?;
                Ok(Terminator::Unwind { src })
            }
            tag::T_RETHROW => Ok(Terminator::Rethrow),
            tag => Err(DecodeError::UnknownNodeTag(tag)),
        }
    }
}

fn binop_from_byte(b: u8) -> Result<BinOp, DecodeError> {
    match b {
        0 => Ok(BinOp::Add),
        1 => Ok(BinOp::Sub),
        2 => Ok(BinOp::Mul),
        3 => Ok(BinOp::Div),
        4 => Ok(BinOp::Rem),
        5 => Ok(BinOp::BitAnd),
        6 => Ok(BinOp::BitOr),
        7 => Ok(BinOp::And),
        8 => Ok(BinOp::Or),
        9 => Ok(BinOp::BitXor),
        10 => Ok(BinOp::Shl),
        11 => Ok(BinOp::Shr),
        12 => Ok(BinOp::ShrShr),
        13 => Ok(BinOp::Is),
        b => Err(DecodeError::UnknownNodeTag(b)),
    }
}

fn cmpop_from_byte(b: u8) -> Result<CmpOp, DecodeError> {
    match b {
        0 => Ok(CmpOp::Eq),
        1 => Ok(CmpOp::Ne),
        2 => Ok(CmpOp::Lt),
        3 => Ok(CmpOp::Gt),
        4 => Ok(CmpOp::Le),
        5 => Ok(CmpOp::Ge),
        b => Err(DecodeError::UnknownNodeTag(b)),
    }
}

fn fbinop_from_byte(b: u8) -> Result<FBinOp, DecodeError> {
    match b {
        0 => Ok(FBinOp::Add),
        1 => Ok(FBinOp::Sub),
        2 => Ok(FBinOp::Mul),
        3 => Ok(FBinOp::Div),
        b => Err(DecodeError::UnknownNodeTag(b)),
    }
}

fn fcmpop_from_byte(b: u8) -> Result<FCmpOp, DecodeError> {
    match b {
        0 => Ok(FCmpOp::Eq),
        1 => Ok(FCmpOp::Ne),
        2 => Ok(FCmpOp::Lt),
        3 => Ok(FCmpOp::Gt),
        4 => Ok(FCmpOp::Le),
        5 => Ok(FCmpOp::Ge),
        b => Err(DecodeError::UnknownNodeTag(b)),
    }
}

fn receiver_from_byte(b: u8) -> Result<ReceiverKind, DecodeError> {
    match b {
        0 => Ok(ReceiverKind::None),
        1 => Ok(ReceiverKind::This),
        2 => Ok(ReceiverKind::Explicit),
        b => Err(DecodeError::UnknownNodeTag(b)),
    }
}
