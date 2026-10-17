//! AST to IR body lowering.

use super::super::{BlockId, CmpOp, Instr, IrFunction, LowerError, Terminator, VReg};
use crate::native::ir::{ReceiverKind, StringArena};
use crate::models::symbols::ScopeStack;
use crate::models::{Expr, Stmt};
use crate::native::ir::BinOp as IrBinOp;
use crate::native::ir::Ctx;
use super::{calls::*, eval::*, select::*, types::*};
use super::span::coerce_span_return;

/// Lowers one function body.
///
/// Bindings map `ScopeStack` slots to home vregs: a local keeps a single
/// home for its whole lifetime, so assignments are `Mov`s into that home
/// and stay correct across branches. Expression evaluation produces a
/// fresh vreg per result.
///
/// Emission targets the *current* basic block; control-flow statements
/// seal blocks with [`Terminator::Branch`] and switch emission targets.
/// Heap-typed locals record their scope depth so every exit path from a
/// scope (fall-through, `return`, `break`, `continue`) can release them.
pub(crate) struct Body<'a> {
    pub f: &'a mut IrFunction,
    pub strings: &'a mut StringArena,
    pub scopes: ScopeStack,
    /// slot index → home vreg.
    pub bindings: Vec<VReg>,
    /// Vregs whose value is a compile-time constant → that value. Drives
    /// constant folding; every write to a vreg updates or removes its entry.
    pub(crate) const_vals: std::collections::BTreeMap<u32, i64>,
    /// Heap-typed locals: (home vreg, byte size, scope depth at declaration).
    pub(crate) heap_locals: Vec<(VReg, u64, usize)>,
    /// Current nesting depth; the function body itself is depth 1.
    pub(crate) depth: usize,
    /// Innermost-first loop context for break/continue.
    loops: Vec<LoopCtx>,
    /// Inline return capture for inlined property getters (and similar
    /// expression-position bodies): `Some((join, home))` reroutes
    /// `return expr;` to `Mov home, expr; goto join` instead of
    /// function return. `None` for ordinary method bodies.
    pub(crate) inline_ret: Option<(BlockId, VReg)>,
    /// In-progress property inlinings `(class path, property)` for
    /// cycle detection (`A => B`, `B => A` would recurse forever).
    pub(crate) prop_inline_stack: Vec<(String, String)>,
    /// Goto-label targets for the method under lowering (C# labels
    /// are function-scoped): label name → block. `goto` creates the
    /// block eagerly when the label is still ahead; the `Label`
    /// statement switches emission into it. Never cleared by
    /// `scope_exit` (labels outlive blocks).
    labels: std::collections::HashMap<String, BlockId>,
    cur: BlockId,
    cur_sealed: bool,
    /// Active `try` handlers (entered, not yet popped). `return`
    /// inside `try` must pop these before returning, or the handler
    /// count leaks and later throws longjmp into dead frames (the
    /// `throw`-segfault: stale count diverted `HelperExcThrow`).
    /// (`break`/`continue`/`goto` out of `try` leak the same way —
    /// future work; `return` is the observed shape.)
    pub(crate) try_depth: usize,
}

#[derive(Clone, Copy)]
struct LoopCtx {
    break_to: BlockId,
    /// `None` marks a *switch* context: `break` targets it, `continue`
    /// skips past it to the nearest real loop.
    continue_to: Option<BlockId>,
    /// Depth when the loop body was entered; break/continue release heap
    /// locals above this.
    body_depth: usize,
}

pub(crate) struct Checkpoint {
    vregs: u32,    blocks_len: usize,
    cur: BlockId,
    cur_sealed: bool,
    cur_term: Terminator,
    cur_instrs_len: usize,
    bindings_len: usize,
    const_vals: std::collections::BTreeMap<u32, i64>,
    heap_len: usize,
    depth: usize,
    loops_len: usize,
    inline_ret: Option<(BlockId, VReg)>,
    scopes_depth: usize,
    prop_stack_len: usize,
    labels: std::collections::HashMap<String, BlockId>,
    try_depth: usize,
}


/// True when lowering `stmt` can define a goto target even on a
/// sealed fallthrough path: a bare `Label`, or a `Block` whose first
/// statement does (the parser wraps `L: stmt` that way).
fn starts_with_label(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Label { .. } => true,
        Stmt::Block { body, .. } => body.stmts.first().is_some_and(starts_with_label),
        _ => false,
    }
}

impl<'a> Body<'a> {
    pub(crate) fn new(f: &'a mut IrFunction, strings: &'a mut StringArena) -> Self {
        let mut bindings = Vec::new();
        if f.receiver != ReceiverKind::None {
            bindings.push(VReg(0));
        }
        for i in 0..f.arity {
            bindings.push(VReg((i + 1) as u32));
        }
        Body {
            f,
            strings,
            scopes: ScopeStack::new(),
            bindings,
            const_vals: Default::default(),
            heap_locals: Vec::new(),
            depth: 1,
            loops: Vec::new(),
            inline_ret: None,
            prop_inline_stack: Vec::new(),
            labels: std::collections::HashMap::new(),
            cur: 0,
            cur_sealed: false,
            try_depth: 0,
        }
    }

    /// Assemble the finished function. Every reachable block ends in a
    /// terminator installed during lowering; the current block gets
    /// `term_for_open` when left open.
    pub(crate) fn finish_with(self, term_for_open: Terminator) -> IrFunction {
        if !self.cur_sealed {
            self.f.blocks[self.cur as usize].term = term_for_open;
        }
        let blocks = std::mem::take(&mut self.f.blocks);
        let mut f = IrFunction {
            name: self.f.name.clone(),
            receiver: self.f.receiver,
            arity: self.f.arity,
            blocks,
            vregs: self.f.vregs,
            is_async: self.f.is_async,
        };
        // Apply the async state-machine transform if this is an async function.
        if f.is_async {
            f = lower_async_state_machine(f);
        }
        f
    }

    pub(crate) fn instrs_of(&mut self) -> &mut Vec<Instr> {
        &mut self.f.blocks[self.cur as usize].instrs
    }

    /// Append a non-terminating instruction to the current block.
    /// Post-divergence emission evaporates: an expression-`throw`
    /// (`=> throw`, `?? throw`, `?: throw`) seals the block with
    /// `Unwind`, and every later instruction in the same block is
    /// unreachable by construction. Dropping (instead of asserting)
    /// keeps mid-expression divergence sound; fresh blocks resume
    /// emission via `switch_to`, and checkpoints restore the flag.
    pub(crate) fn push_raw(&mut self, i: Instr) {
        if self.cur_sealed {
            return;
        }
        self.instrs_of().push(i);
    }

    /// Seal the current block. First terminator wins (see above):
    /// post-`throw` seals are unreachable, never overrides.
    pub(crate) fn terminate(&mut self, t: Terminator) {
        if self.cur_sealed {
            return;
        }
        self.f.blocks[self.cur as usize].term = t;
        self.cur_sealed = true;
    }

    pub(crate) fn is_sealed(&self) -> bool {
        self.cur_sealed
    }

    /// Create an empty block without leaving the current one.
    pub(crate) fn make_block(&mut self) -> BlockId {
        self.f.block()
    }

    pub(crate) fn switch_to(&mut self, id: BlockId) {
        self.cur = id;
        self.cur_sealed = false;
    }

    /// Unconditional jump to `id`.
    pub(crate) fn goto(&mut self, id: BlockId) {
        self.terminate(Terminator::Branch {
            cond: None,
            if_true: id,
            if_false: id,
        });
    }


    pub(crate) fn checkpoint(&mut self) -> Checkpoint {
        let cur_instrs_len = self.f.blocks[self.cur as usize].instrs.len();
        let cur_term = self.f.blocks[self.cur as usize].term.clone();
        Checkpoint {
            vregs: self.f.vregs,
            blocks_len: self.f.blocks.len(),
            cur: self.cur,
            cur_sealed: self.cur_sealed,
            cur_term,
            cur_instrs_len,
            bindings_len: self.bindings.len(),
            const_vals: self.const_vals.clone(),
            heap_len: self.heap_locals.len(),
            depth: self.depth,
            loops_len: self.loops.len(),
            inline_ret: self.inline_ret,
            prop_stack_len: self.prop_inline_stack.len(),
            scopes_depth: self.scopes.depth(),
            try_depth: self.try_depth,
            labels: self.labels.clone(),
        }
    }

    /// Roll back to a checkpoint, discarding partial IR (failed
    /// inline attempts leave no trace).
    pub(crate) fn restore(&mut self, cp: Checkpoint) {
        self.f.vregs = cp.vregs;
        self.f.blocks.truncate(cp.blocks_len);
        self.cur = cp.cur;
        self.cur_sealed = cp.cur_sealed;
        if let Some(bl) = self.f.blocks.get_mut(self.cur as usize) {
            bl.instrs.truncate(cp.cur_instrs_len);
            bl.term = cp.cur_term;
        }
        self.bindings.truncate(cp.bindings_len);
        self.const_vals = cp.const_vals;
        self.heap_locals.truncate(cp.heap_len);
        self.depth = cp.depth;
        self.try_depth = cp.try_depth;
        self.loops.truncate(cp.loops_len);
        self.inline_ret = cp.inline_ret;
        self.scopes.truncate(cp.scopes_depth);
        self.prop_inline_stack.truncate(cp.prop_stack_len);
        self.labels = cp.labels;
    }

    /// Enter a lexical scope: real symbol scope plus release-tracking depth.
    pub(crate) fn scope_enter(&mut self) {
        self.scopes.push();
        self.depth += 1;
    }

    /// Exit a lexical scope, releasing heap locals declared inside it.
    pub(crate) fn scope_exit(&mut self) {
        if self.is_sealed() {
            // Sealed by return/throw inside: no instruction (including
            // `Release`) may follow the terminator. Drop tracking for
            // this scope's heap locals without emitting (they leak —
            // same trade as `release_scope_locals`).
            while self
                .heap_locals
                .last()
                .is_some_and(|(_, _, d)| *d > self.depth - 1)
            {
                self.heap_locals.pop();
            }
        } else {
            self.release_heap_above(self.depth - 1);
        }
        self.scopes.pop();
        self.depth -= 1;
    }

    /// Release every heap local declared deeper than `target`, deepest first.
    pub(crate) fn release_heap_above(&mut self, target: usize) {
        while let Some(&(home, bytes, d)) = self.heap_locals.last() {
            if d <= target {
                break;
            }
            self.heap_locals.pop();
            self.push_raw(Instr::Release { src: home, bytes });
        }
    }

    /// Register a literal in the shared string arena.
    pub(crate) fn string_literal(&mut self, s: &str) -> (u32, u32) {
        self.strings.literal(s)
    }

    /// Register a user-string literal as UTF-16 (see
    /// `StringArena::utf16_literal`). Returns `(offset, char_len)`.
    pub(crate) fn utf16_literal(&mut self, s: &str) -> (u32, u32) {
        self.strings.utf16_literal(s)
    }

    pub(crate) fn fresh(&mut self) -> VReg {
        self.f.fresh_vreg()
    }

    /// Push a valueless `Instr` (no return vreg) directly into the current body.
    pub(crate) fn raw_vreg(&mut self, i: Instr) {
        self.push_raw(i);
    }

    pub(crate) fn scope_mut(&mut self) -> &mut ScopeStack {
        &mut self.scopes
    }

    pub(crate) fn handle_bytes(&self, type_name: &str, ctx: &Ctx<'_>) -> Option<u64> {
        // Enums lower as raw ints (never heap objects): no tracking,
        // even though they ARE table entries (see `is_heap_ref`).
        let info = ctx.table.class(type_name)?;
        if info.is_enum {
            return None;
        }
        // Pointer-sized integers lower as raw addresses/ints (never
        // heap objects): tracking them retains integer values as
        // objects (`Retain(IntPtr+4)` faulted the casing loop when a
        // nonzero `byteOffset` flowed in). Same class of exclusion
        // as enums above.
        {
            let base = type_name.trim().rsplit('.').next().unwrap_or(type_name);
            if matches!(
                base,
                "IntPtr" | "UIntPtr" | "nint" | "nuint"
            ) {
                return None;
            }
        }
        Some((info.fields.len() + 2) as u64 * 8)
    }

    pub(crate) fn emit_retain(&mut self, src: VReg) {
        self.push_raw(Instr::Retain { src });
    }

    pub(crate) fn emit_release(&mut self, src: VReg, bytes: u64) {
        self.push_raw(Instr::Release { src, bytes });
    }

    pub(crate) fn emit(&mut self, i: Instr) -> VReg {
        let dst = self.f.fresh_vreg();
        self.const_vals.remove(&dst.0); // writes are opaque unless recorded below
        match i {
            Instr::Const { imm, .. } => {
                self.push_raw(Instr::Const { dst, imm });
                self.const_vals.insert(dst.0, imm);
            }
            Instr::Mov { a, .. } => {
                self.push_raw(Instr::Mov { dst, a });
                if let Some(v) = self.const_vals.get(&a.0) {
                    self.const_vals.insert(dst.0, *v);
                }
            }
            Instr::Bin { op, a, b, .. } => self.push_raw(Instr::Bin { dst, op, a, b }),
            Instr::Cmp { op, a, b, .. } => self.push_raw(Instr::Cmp { dst, op, a, b }),
            Instr::Neg { a, .. } => self.push_raw(Instr::Neg { dst, a }),
            Instr::CallFn { fidx, args, .. } => self.push_raw(Instr::CallFn { dst, fidx, args }),
            Instr::CallStatic { fidx, args, .. } => {
                self.push_raw(Instr::CallStatic { dst, fidx, args })
            }
            Instr::CallVirt {
                slot, recv, args, ..
            } => self.push_raw(Instr::CallVirt {
                dst,
                slot,
                recv,
                args,
            }),
            Instr::CallImport { import, args, .. } => {
                self.push_raw(Instr::CallImport { dst, import, args })
            }
            Instr::Alloc { bytes, .. } => self.push_raw(Instr::Alloc { dst, bytes }),
            Instr::VTableAddr { row, .. } => self.push_raw(Instr::VTableAddr { dst, row }),
            Instr::LoadStaticAddr { idx, .. } => {
                self.push_raw(Instr::LoadStaticAddr { dst, idx })
            }
            Instr::LoadImport { import, .. } => {
                self.push_raw(Instr::LoadImport { dst, import })
            }
            Instr::LoadField { base, off, .. } => {
                self.push_raw(Instr::LoadField { dst, base, off })
            }
            other => unreachable!("valueless instruction routed through emit: {other:?}"),
        }
        dst
    }

    pub(crate) fn const_i64(&mut self, imm: i64) -> VReg {
        self.emit(Instr::Const { dst: VReg(0), imm })
    }

    pub(crate) fn alloc(&mut self, dst: VReg, bytes: u64) {
        self.push_raw(Instr::Alloc { dst, bytes });
    }

    /// Allocate a zeroed heap box for a struct entry: vtable row,
    /// refcount 1, and zeroed field slots. Gives `default(T)` and
    /// uninitialized struct positions first-class object behavior
    /// (member access, method calls) instead of null-deref faults.
    /// Callers pass the table entry (fields drive size); the vtable
    /// row counts table-backed classes like `Program` construction.
    pub(crate) fn alloc_struct_box(
        &mut self,
        ctx: &Ctx<'_>,
        entry: &crate::native::semantic::ClassInfo,
    ) -> VReg {
        let bytes = (entry.fields.len() as u64 + 2) * 8;
        let boxed = self.fresh();
        self.alloc(boxed, bytes);
        let vt = self.fresh();
        self.vtable_addr(
            vt,
            super::span::vtable_row_for(ctx, &entry.name, &entry.key.path),
        );
        self.push_raw(Instr::StoreField {
            base: boxed,
            off: 0,
            src: vt,
        });
        let one = self.const_i64(1);
        self.push_raw(Instr::StoreField {
            base: boxed,
            off: 8,
            src: one,
        });
        let z = self.const_i64(0);
        let mut off = 16u32;
        while (off as u64) < bytes {
            self.push_raw(Instr::StoreField {
                base: boxed,
                off,
                src: z,
            });
            off += 8;
        }
        boxed
    }

    pub(crate) fn vtable_addr(&mut self, dst: VReg, row: u32) {
        self.push_raw(Instr::VTableAddr { dst, row });
    }

    /// Load the absolute address of static slot `idx` into `dst`.
    pub(crate) fn load_static_addr(&mut self, dst: VReg, idx: u32) {
        self.push_raw(Instr::LoadStaticAddr { dst, idx });
    }

    /// Store `src` into static slot `idx` (address + offset 0).
    pub(crate) fn store_static(&mut self, idx: u32, src: VReg) {
        let base = self.fresh();
        self.push_raw(Instr::LoadStaticAddr { dst: base, idx });
        self.push_raw(Instr::StoreField { base, off: 0, src });
    }

    pub(crate) fn raw(&mut self, bytes: Vec<u8>) {
        self.push_raw(Instr::Raw(bytes));
    }

    /// Build a first-class string object from the literal at data offset
    /// `off` (length `len`). Layout: [0..8)=length, [8..16)=refcount=1,
    /// [16..16+len)=data. Returns the object pointer vreg.
    pub(crate) fn build_string(&mut self, off: u32, len: u32) -> VReg {
        let total = 16u64 + len as u64;
        let obj = self.fresh();
        self.push_raw(Instr::Alloc { dst: obj, bytes: total });
        let lenv = self.const_i64(len as i64);
        self.push_raw(Instr::StoreField { base: obj, off: 0, src: lenv });
        let one = self.const_i64(1);
        self.push_raw(Instr::StoreField { base: obj, off: 8, src: one });
        let ptr = self.fresh();
        self.push_raw(Instr::LoadStrAddr { dst: ptr, off });
        let s16 = self.const_i64(16);
        let data = self.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::Add,
            a: obj,
            b: s16,
        });
        self.raw_vreg(Instr::MemCpy { dst: data, src: ptr, len: lenv });
        obj
    }

    /// Build a UTF-16 user string from interned `(off, char_len)` (see
    /// `utf16_literal`): length header holds the character count,
    /// data holds `char_len` UTF-16 code units (`char_len*2` bytes).
    pub(crate) fn build_utf16_string(&mut self, off: u32, char_len: u32) -> VReg {
        let total = 16u64 + char_len as u64 * 2;
        let obj = self.fresh();
        self.push_raw(Instr::Alloc { dst: obj, bytes: total });
        let lenv = self.const_i64(char_len as i64);
        self.push_raw(Instr::StoreField { base: obj, off: 0, src: lenv });
        let one = self.const_i64(1);
        self.push_raw(Instr::StoreField { base: obj, off: 8, src: one });
        let ptr = self.fresh();
        self.push_raw(Instr::LoadStrAddr { dst: ptr, off });
        let s16 = self.const_i64(16);
        let data = self.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::Add,
            a: obj,
            b: s16,
        });
        let byte_len = self.const_i64(char_len as i64 * 2);
        self.raw_vreg(Instr::MemCpy { dst: data, src: ptr, len: byte_len });
        obj
    }

    /// Concatenate two heap strings (C# `+` on `string`): allocates a new
    /// object whose data is the byte sequence of `a` followed by `b`.
    pub(crate) fn concat_strings(&mut self, a: VReg, b: VReg) -> VReg {
        // Null concatenates as empty (C# `string.Concat`/`+`/
        // interpolation semantics): framework init routinely
        // concatenates nulls (`typeof(T).Name` on placeholder type
        // boxes, unset optional segments), and faulting on the
        // length load killed static initializers. Each operand
        // resolves to (length, data); null contributes length 0
        // with the new object's data as its (never-read) pointer.
        // Lengths first (both branches), then one allocation.
        let nul = self.const_i64(0);
        let len_a = self.fresh();
        let is_null_a = self.emit(Instr::Cmp {
            dst: VReg(0),
            op: crate::native::ir::CmpOp::Eq,
            a,
            b: nul,
        });
        let null_a_b = self.make_block();
        let ok_a_b = self.make_block();
        let done_a_b = self.make_block();
        self.terminate(Terminator::Branch {
            cond: Some(is_null_a),
            if_true: null_a_b,
            if_false: ok_a_b,
        });
        self.switch_to(null_a_b);
        let z = self.const_i64(0);
        self.push_raw(Instr::Mov { dst: len_a, a: z });
        self.goto(done_a_b);
        self.switch_to(ok_a_b);
        self.push_raw(Instr::LoadField { dst: len_a, base: a, off: 0 });
        self.goto(done_a_b);
        self.switch_to(done_a_b);
        let len_b = self.fresh();
        let is_null_b = self.emit(Instr::Cmp {
            dst: VReg(0),
            op: crate::native::ir::CmpOp::Eq,
            a: b,
            b: nul,
        });
        let null_b_b = self.make_block();
        let ok_b_b = self.make_block();
        let done_b_b = self.make_block();
        self.terminate(Terminator::Branch {
            cond: Some(is_null_b),
            if_true: null_b_b,
            if_false: ok_b_b,
        });
        self.switch_to(null_b_b);
        let z2 = self.const_i64(0);
        self.push_raw(Instr::Mov { dst: len_b, a: z2 });
        self.goto(done_b_b);
        self.switch_to(ok_b_b);
        self.push_raw(Instr::LoadField { dst: len_b, base: b, off: 0 });
        self.goto(done_b_b);
        self.switch_to(done_b_b);
        let total = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: len_a, b: len_b });
        let s16a = self.const_i64(16);
        // Object = 16-byte header (len + refcount) + UTF-16 data:
        // lengths are CHARACTER counts, so byte sizes double them.
        let two = self.const_i64(2);
        let total_bytes = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Mul, a: total, b: two });
        let total_sz = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: total_bytes, b: s16a });
        let newobj = self.fresh();
        self.push_raw(Instr::AllocDyn { dst: newobj, bytes: total_sz });
        self.push_raw(Instr::StoreField { base: newobj, off: 0, src: total });
        let one = self.const_i64(1);
        self.push_raw(Instr::StoreField { base: newobj, off: 8, src: one });
        let s16 = self.const_i64(16);
        let data_new = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: newobj, b: s16 });
        let data_a = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a, b: s16 });
        let len_a_bytes = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Mul, a: len_a, b: two });
        self.raw_vreg(Instr::MemCpy { dst: data_new, src: data_a, len: len_a_bytes });
        let ptr2 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: data_new, b: len_a_bytes });
        let data_b = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: b, b: s16 });
        let len_b_bytes = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Mul, a: len_b, b: two });
        self.raw_vreg(Instr::MemCpy { dst: ptr2, src: data_b, len: len_b_bytes });
        newobj
    }

    pub(crate) fn call_fn(&mut self, fidx: usize, args: Vec<VReg>) -> VReg {
        let dst = self.fresh();
        self.push_raw(Instr::CallFn {
            dst,
            fidx: fidx as u32,
            args,
        });
        dst
    }

    /// Terminate with `exit_group(code)` where the code is a runtime value
    /// (the entry method's return).
    pub(crate) fn exit_with(&mut self, code: VReg) {
        self.terminate(Terminator::ExitWith(code));
    }

    pub(crate) fn mov(&mut self, dst: VReg, a: VReg) {
        self.push_raw(Instr::Mov { dst, a });
        // Writes into shared homes (locals/params) must INVALIDATE their
        // constant status: with loops and branches, a home's value at one
        // program point says nothing about it after a back-edge or a
        // divergent path. Only single-assignment temporaries (tracked in
        // emit()) may carry compile-time constants.
        self.const_vals.remove(&dst.0);
    }

    /// Lower a full body: statements, then an implicit `return 0`.
    /// Bind a placeholder zero for `name` in the current scope.
    /// Primary-constructor params are visible in every member body
    /// (C# scoping), but their captured VALUES need hidden-field
    /// storage (future work — the AST keeps primary names only), so
    /// reads placeholder as zero (documented gap). Names already
    /// declared here (real params/locals) win.
    pub(crate) fn bind_zero_placeholder(&mut self, name: &str) {
        if self.scopes.lookup_here(name).is_some() {
            return;
        }
        let Ok(slot) = self
            .scopes
            .declare(name, "object", crate::models::Access::Default)
        else {
            return;
        };
        while self.bindings.len() <= slot {
            let v = self.fresh();
            self.bindings.push(v);
        }
        let z = self.const_i64(0);
        self.mov(self.bindings[slot], z);
    }

    /// Pre-declare local-function names (`fn`-typed `Let`s) in the
    /// current scope, bound to zero (see `block`). Switch sections
    /// share one scope, so their local functions are visible
    /// switch-wide (`case 0: F(); ... static void F(){}`).
    /// Same-scope duplicates are skipped (re-bound to zero by their
    /// own declaration later); outer names still shadow correctly.
    pub(crate) fn hoist_local_fns<'b>(
        &mut self,
        fns: impl Iterator<Item = (&'b str, crate::models::Access)>,
    ) {
        for (name, access) in fns {
            if self.scopes.lookup_here(name).is_some() {
                continue;
            }
            let Ok(slot) = self.scopes.declare(name, "fn", access) else {
                continue;
            };
            while self.bindings.len() <= slot {
                let v = self.fresh();
                self.bindings.push(v);
            }
            let z = self.const_i64(0);
            self.mov(self.bindings[slot], z);
        }
    }

    pub(crate) fn block(
        &mut self,
        block: &crate::models::Block,
        ctx: &Ctx<'_>,
    ) -> Result<(), LowerError> {
        // Hoist local-function names block-wide (C# scoping: a local
        // function is visible throughout its enclosing block, even for
        // calls that precede its declaration). The parser exposes each
        // local function as an `fn`-typed `Let`; pre-binding it to zero
        // mirrors the semantic scan's pre-registration, so forward calls
        // resolve to the same placeholder the declaration produces.
        for stmt in &block.stmts {
            if let Stmt::Let { type_name, name, access, .. } = stmt
                && type_name == "fn"
                && self.scopes.lookup_here(name).is_none()
            {
                let slot = self
                    .scopes
                    .declare(name, "fn", *access)
                    .map_err(LowerError::UnsupportedConstruct)?;
                while self.bindings.len() <= slot {
                    let v = self.fresh();
                    self.bindings.push(v);
                }
                let z = self.const_i64(0);
                self.mov(self.bindings[slot], z);
            }
        }
        for stmt in &block.stmts {
            // Unreachable code after return/break/continue is skipped —
            // EXCEPT labels: a `Label` is reachable through its gotos
            // even when the fallthrough path is sealed (the `goto
            // FalseExit` idiom jumps forward over the seal), and
            // switching into it unseals emission for what follows.
            // (The parser wraps `L: stmt` in a `Block`, so look
            // through leading blocks.)
            if self.is_sealed() && !starts_with_label(stmt) {
                break; // unreachable code after return/break/continue
            }
            self.statement(stmt, ctx)?;
        }
        Ok(())
    }

    /// Pop all active `try` handlers (for `return` inside `try`,
    /// which otherwise leaks the handler count and diverts later
    /// throws into dead frames). Called before emitting `Ret`.
    pub(crate) fn pop_trys(&mut self) {
        while self.try_depth > 0 {
            self.push_raw(Instr::HandlerPop);
            self.try_depth -= 1;
        }
    }

    /// True when a statement (or a block wrapping one) begins with
    /// a goto `Label`: such statements stay reachable through their
    /// gotos even on a sealed fallthrough path (see `block`).
    pub(crate) fn seal_with_return_zero(&mut self) {
        if !self.is_sealed() {
            let z = self.const_i64(0);
            self.pop_trys();
            self.terminate(Terminator::Ret(Some(z)));
        }
    }

    /// Release every heap-typed local home (reverse declaration order).
    pub(crate) fn release_scope_locals(&mut self) {
        if self.is_sealed() {
            // The block already ends in a terminator (e.g. `return`); we
            // cannot emit `Release` instructions after it. The heap objects
            // are leaked, but this is better than malformed IR.
            self.heap_locals.clear();
            return;
        }
        while let Some((home, bytes, _)) = self.heap_locals.pop() {
            self.push_raw(Instr::Release { src: home, bytes });
        }
    }

    /// By-reference local alias (`ref char x = ref y`): bind the
    /// home to the aliased ADDRESS (not a value copy). `ref`
    /// chains pass through; index/member targets use interior
    /// addresses; anything else boxes once (stable alias, though
    /// writes do not propagate back to a plain source — C# forbids
    /// that shape anyway).
    fn let_ref_alias(
        &mut self,
        name: &str,
        type_name: &str,
        access: &crate::models::Access,
        init: &crate::models::Expr,
        ctx: &Ctx<'_>,
    ) -> Result<(), LowerError> {
        use crate::models::Expr;
        let addr = match init {
            Expr::Ident { name: src, .. } => {
                if let Some(h) = super::eval::byref_home(self, src) {
                    h
                } else {
                    let v = super::eval::eval(init, self, ctx)?;
                    let cell = self.fresh();
                    self.push_raw(Instr::Alloc { dst: cell, bytes: 16 });
                    self.push_raw(Instr::StoreField {
                        base: cell,
                        off: 0,
                        src: v,
                    });
                    cell
                }
            }
            Expr::Index { .. } | Expr::Member { .. } => {
                match super::eval::ref_target_address(self, ctx, init)? {
                    Some(a) => a,
                    None => {
                        let v = super::eval::eval(init, self, ctx)?;
                        let cell = self.fresh();
                        self.push_raw(Instr::Alloc { dst: cell, bytes: 16 });
                        self.push_raw(Instr::StoreField {
                            base: cell,
                            off: 0,
                            src: v,
                        });
                        cell
                    }
                }
            }
            _ => {
                let v = super::eval::eval(init, self, ctx)?;
                let cell = self.fresh();
                self.push_raw(Instr::Alloc { dst: cell, bytes: 16 });
                self.push_raw(Instr::StoreField {
                    base: cell,
                    off: 0,
                    src: v,
                });
                cell
            }
        };
        let slot = self
            .scopes
            .declare_byref(name, type_name, *access, true)
            .map_err(LowerError::UnsupportedConstruct)?;
        while self.bindings.len() <= slot {
            let v = self.fresh();
            self.bindings.push(v);
        }
        self.mov(self.bindings[slot], addr);
        Ok(())
    }

    pub(crate) fn statement(&mut self, stmt: &Stmt, ctx: &Ctx<'_>) -> Result<(), LowerError> {
        match stmt {
            // Local functions lower via the method pre-scan (real
            // mangled functions for static non-capturing ones, legacy
            // zero placeholder otherwise) — never inline here.
            Stmt::LocalFunction { .. } => Ok(()),
            Stmt::Let {
                type_name,
                name,
                init,
                access,
                is_byref,
                span,
                ..
            } => {
                // Hoisted local-function re-declaration (see `block`):
                // the name is already bound to zero in THIS scope, so
                // just re-bind zero instead of tripping the CS0136
                // same-scope duplicate rule on our own hoist.
                if type_name == "fn"
                    && let Some(sym) = self.scopes.lookup_here(name)
                {
                    let home = self.bindings[sym.slot];
                    let z = self.const_i64(0);
                    self.mov(home, z);
                    return Ok(());
                }
                // Same-scope re-declaration degrades to assignment.
                // Valid C# never redeclares in one scope (CS0136), so
                // this only fires for constructs that share it by
                // design — notably switch-statement pattern variables,
                // which are arm-scoped in C# (`case A tc:` + `case B
                // tc:` is legal) but share our section scope. The
                // existing home keeps its value semantics (pattern
                // bindings initialize to zero either way), but the home
                // is RE-TYPED to the new arm's explicit type so member
                // calls resolve correctly (`tc.Run(...)` after
                // `case TaskContinuation tc:`).
                if self.scopes.lookup_here(name).is_some() {
                    let incoming = normalize_type_ref(ctx.table, type_name.as_str());
                    // Class type variables are never alias-resolved
                    // (see `Ctx::is_type_var`).
                    let incoming = if ctx.is_type_var(&incoming) {
                        incoming
                    } else {
                        ctx.table
                            .alias_target(&incoming)
                            .map(|t| t.to_owned())
                            .unwrap_or(incoming)
                    };
                    if incoming != "var" {
                        self.scopes.retype_here(name, &incoming);
                    }
                    return self.statement(
                        &Stmt::Assign {
                            name: name.clone(),
                            value: init.clone(),
                            span: *span,
                        },
                        ctx,
                    );
                }
                // By-reference local (`ref char x = ref y`): the home
                // holds the aliased ADDRESS, not a value copy (dropping
                // `ref` made `CompareStringIgnoreCase` loop forever:
                // `index += 0` on a zeroed copy OOMed init). Resolve
                // the address from the initializer and bind it.
                if *is_byref {
                    return self.let_ref_alias(
                        name,
                        type_name,
                        access,
                        init,
                        ctx,
                    );
                }
                // Normalize the declared type reference: resolve dotted
                // spellings to table key paths when possible
                // (`Matrix3x2.Impl` → `System.Numerics.Matrix3x2.Impl`,
                // alias heads included), else legacy bare-simple form.
                // Preserving qualification keeps scope-stored types
                // precise instead of collapsing to ambiguous simples
                // (which mis-resolve across same-named nesteds).
                // `global::` never appears in keys — strip it first.
                let type_name = normalize_type_ref(ctx.table, type_name.as_str());
                // `using Alias = Target`: resolve to the target's simple
                // name before every check below (the alias IS the
                // target type). Class aliases resolve in `table.class`
                // itself; this covers primitive targets here. Class
                // type variables are exempt (see
                // `Ctx::is_type_var`).
                let type_name = if ctx.is_type_var(&type_name) {
                    type_name
                } else {
                    ctx.table
                        .alias_target(&type_name)
                        .map(|t| t.to_owned())
                        .unwrap_or(type_name)
                };
                // Class-typed locals are object handles at this stage.
                let is_handle = ctx.table.class(&type_name).is_some();
                // Strip array/nullable/pointer suffixes to find the base type.
                let base_name = {
                    let mut s = type_name.clone();
                    loop {
                        if s.ends_with("[]") {
                            s.pop();
                            s.pop();
                        } else if let Some(pos) = s.rfind('[') {
                            s.truncate(pos);
                        } else if s.ends_with('?') || s.ends_with('*') {
                            s.pop();
                        } else {
                            break;
                        }
                    }
                    // Strip namespace prefix (e.g. `global::Foo.Bar` -> `Bar`).
                    if let Some(pos) = s.rfind('.') {
                        s = s[pos + 1..].to_owned();
                    }
                    s
                };
                let is_generic_param = base_name
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_uppercase())
                    .unwrap_or(false)
                    && !ctx.table.class(&base_name).is_some();
                let is_delegate = type_name.starts_with("delegate");
                let is_array = type_name.contains('[');
                let is_handle_like = is_handle
                    || type_name.ends_with('?')
                    || type_name.ends_with('*')
                    || type_name == "string"
                    || type_name == "object"
                    || type_name == "dynamic"
                    || type_name == "tuple"
                    || type_name == "var"
                    || type_name == "fn"
                    || is_array
                    || is_generic_param
                    || is_delegate;
                if !is_handle_like {
                    // Integer and floating-point (`float`/`double`, IEEE 754
                    // double in a 64-bit vreg) scalar types. `decimal` is
                    // modeled as a 64-bit scalar too (NOT true 128-bit
                    // decimal semantics — precision/scale are future work;
                    // this only unblocks compilation of decimal-heavy
                    // code like `Number`).
                    const SUPPORTED: [&str; 15] = [
                        "int", "long", "sbyte", "byte", "short", "ushort", "uint", "ulong", "nint",
                        "nuint", "char", "float", "double", "decimal", "Decimal",
                    ];
                    if !SUPPORTED.contains(&type_name.as_str()) && type_name != "bool" {
                        return Err(LowerError::UnsupportedConstruct(crate::tf!(
                            "rubyc.native.ir.check_supported_type.unsupported",
                            type_name.to_owned()
                        )));
                    }
                }
                // `var` infers the type from the initializer (C#-compatible).
                let mut declared_type = type_name.clone();
                if declared_type == "var" {
                    if let Some(inferred) = static_type_of(init, self, ctx).ok() {
                        // Neither `object` (unknown) nor `_` (discard)
                        // are real types to adopt.
                        if inferred != "object" && inferred != "_" {
                            declared_type = inferred;
                        }
                    }
                }
                // Self-assigning declaration (`MethodInfo? getMethod =
                // getMethod = type.GetMethod(...)`): the initializer
                // assigns to the very name being declared. Bind it
                // (zero) first so the assignment target resolves; the
                // value then flows normally. Reads of the name
                // elsewhere in its own initializer stay errors
                // (C# definite assignment).
                let mut prebound_slot: Option<usize> = None;
                let is_self_assign = matches!(init,
                    crate::models::Expr::Assign { target, .. }
                        if matches!(target.as_ref(),
                            crate::models::Expr::Ident { name: aname, .. } if aname == name));
                if is_self_assign && self.scopes.lookup(name).is_none() {
                    let slot = self
                        .scopes
                        .declare(name, &declared_type, *access)
                        .map_err(LowerError::UnsupportedConstruct)?;
                    while self.bindings.len() <= slot {
                        let v = self.fresh();
                        self.bindings.push(v);
                    }
                    let z = self.const_i64(0);
                    self.mov(self.bindings[slot], z);
                    prebound_slot = Some(slot);
                }
                // Target-typed `new(...)` (C# 9): the parser leaves
                // `class_name` empty; fill it from the declared type so
                // `DelayState state = new(token)` constructs a DelayState.
                let v = match init {
                    crate::models::Expr::New {
                        class_name,
                        args,
                        init,
                        span,
                    } if class_name.is_empty() => {
                        let filled = crate::models::Expr::New {
                            class_name: type_name.clone(),
                            args: args.clone(),
                            init: init.clone(),
                            span: *span,
                        };
                        eval(&filled, self, ctx)?
                    }
                    // Struct local from an array element (`ref Entry e
                    // = ref arr[i]` — `ref`/`!` have no runtime form,
                    // so plain copies share the shape): bind the
                    // slot's own box (materialized + written back
                    // when null — see `index_struct_box`) so later
                    // field stores persist instead of dropping.
                    crate::models::Expr::Index { object, index, .. } => {
                        match super::eval::index_struct_box(self, ctx, object, index)? {
                            Some(boxed) => boxed,
                            None => eval(init, self, ctx)?,
                        }
                    }
                    other => eval(other, self, ctx)?,
                };
                // Implicit conversion to span-typed locals
                // (`ReadOnlySpan<short> rawData = [...]`): the
                // collection evaluates to a bare array; pack it into
                // a span object (mirrors return/argument coercion).
                // Without this the local holds the array and later
                // `rawData[i]` reads `[array+16]` (element 0's value)
                // as the data pointer (the UmAlQuraCalendar fault).
                let v = {
                    let val_ty = static_type_of(init, self, ctx)
                        .unwrap_or_else(|_| "object".to_owned());
                    coerce_span_return(self, ctx, v, &val_ty, &declared_type)?
                };
                let slot = match prebound_slot {
                    // Self-assigning declaration (bound above): reuse
                    // the home instead of re-declaring (which would
                    // trip the same-scope duplicate rule).
                    Some(s) => s,
                    None => self
                        .scopes
                        .declare(name, &declared_type, *access)
                        .map_err(LowerError::UnsupportedConstruct)?,
                };
                if slot < self.bindings.len() {
                    // Shadowing reuses the outer binding index space only
                    // when the slot counter rewinds — it never does, so a
                    // new slot always means a new home.
                }
                while self.bindings.len() <= slot {
                    let v = self.fresh();
                    self.bindings.push(v);
                }
                self.mov(self.bindings[slot], v);
                // Value-type (struct) locals materialize a heap box when
                // the initializer yields null (`S s;`, `default(S)`):
                // a null home segfaults on the first field access
                // (`StoreField` through address zero), while real
                // `new`/call results already arrive as valid handles
                // and bind directly. `Nullable<T>` and fieldless
                // primitives are exempt. Var-to-var struct copies keep
                // aliasing for now (spans, the live case, rebind to
                // fresh handles).
                if let Some(entry) = ctx.table.class(&declared_type).filter(|e| {
                    e.is_struct && !e.fields.is_empty()
                }) && !declared_type.starts_with("Nullable<")
                    && self.const_vals.get(&v.0) == Some(&0)
                {
                    let boxed = self.alloc_struct_box(ctx, entry);
                    self.mov(self.bindings[slot], boxed);
                }
                if let Some(bytes) = self.handle_bytes(&type_name, ctx) {
                    let home = self.bindings[slot];
                    self.heap_locals.push((home, bytes, self.depth));
                }
                Ok(())
            }
            Stmt::Assign { name, value, .. } => {
                // `_` discard: evaluate the value for side effects, then
                // discard the result.
                if name == "_" {
                    let _ = eval(value, self, ctx)?;
                    return Ok(());
                }
                if let Some(sym) = self.scopes.lookup(name) {
                    let home = self.bindings[sym.slot];
                    // By-reference target (`ref`/`out`/`in` param): the
                    // home holds an address — store through it so the
                    // caller observes the write.
                    if sym.is_byref {
                        let v = eval(value, self, ctx)?;
                        self.push_raw(Instr::StoreField {
                            base: home,
                            off: 0,
                            src: v,
                        });
                        return Ok(());
                    }
                    // Heap handles share: retain the incoming value, drop
                    // the old one, then move.
                    let tn = self.scopes.lookup(name).map(|s| s.type_name.clone());
                    let bytes = tn.as_deref().and_then(|t| self.handle_bytes(t, ctx));
                    // Assignment needs the same array/string-to-span view
                    // conversion as a declaration. Otherwise a Span local
                    // holds an array header and reads its refcount as length.
                    let val_ty = static_type_of(value, self, ctx)
                        .unwrap_or_else(|_| "object".to_owned());
                    let v = eval(value, self, ctx)?;
                    let v = coerce_span_return(
                        self, ctx, v, &val_ty, tn.as_deref().unwrap_or("object"),
                    )?;
                    match bytes {
                        Some(bytes) => {
                            self.emit_retain(v);
                            self.emit_release(home, bytes);
                            self.mov(home, v);
                        }
                        None => {
                            self.mov(home, v);
                        }
                    }
                    return Ok(());
                }
                if let Some(info) = this_info(ctx, name)
                    && let Some(field) = info.field(name)
                {
                    // Implicit-`this` write; may still hit an inherited member.
                    ctx.check_access(true, ctx.current_class, name)?;
                    // Static field: write to per-class static storage, independent
                    // of `this` (or to the owner's slot via import when the
                    // field ships from another image).
                    if field.is_static {
                        let Some(base) = static_addr(field, self, ctx) else {
                            return Err(LowerError::UnsupportedConstruct(format!(
                                "static field '{}' has no storage slot",
                                name
                            )));
                        };
                        let v = eval(value, self, ctx)?;
                        self.push_raw(Instr::StoreField { base, off: 0, src: v });
                        return Ok(());
                    }
                    let this = self.this_vreg();
                    let v = eval(value, self, ctx)?;
                    // Span-typed fields assigned from arrays/strings
                    // coerce to views (mirrors the explicit-receiver
                    // path above — implicit-`this` writes like
                    // `ValueStringBuilder._chars = array` take this
                    // branch, and a raw array in the slot reads
                    // refcount as length).
                    let val_ty = static_type_of(value, self, ctx)
                        .unwrap_or_else(|_| "object".to_owned());
                    let v = coerce_span_return(self, ctx, v, &val_ty, &field.type_name)?;
                    // Write barrier (mirrors `MemberAssign`): a stored
                    // reference must be retained, else a later release
                    // of the source (e.g. scope-end for the temp in
                    // `CompareExchange(ref _numInfo, temp, null)`)
                    // frees the object the field points at (the
                    // `_numInfo` use-after-free that faulted
                    // `GetProviderNonNull`).
                    let off = (field.offset_slots * 8) as u32;
                    if super::eval::is_heap_ref(ctx.table, &field.type_name) {
                        let old = self.fresh();
                        self.push_raw(Instr::LoadField {
                            dst: old,
                            base: this,
                            off,
                        });
                        self.emit_retain(v);
                        if let Some(bytes) = self.handle_bytes(&field.type_name, ctx) {
                            self.emit_release(old, bytes);
                        }
                    }
                    self.push_raw(Instr::StoreField { base: this, off, src: v });
                    return Ok(());
                }
                // Accessor-backed property write (`HResult = x` where
                // `HResult` has explicit get/set bodies and no storage
                // slot): dispatch through the setter (virtual deform
                // when overridden, static inline otherwise —
                // `AutoFlush = true` must run StreamWriter's flushing
                // setter). Falls back to evaluate-and-drop when no
                // setter body resolves (legacy behavior).
                // Auto-properties have backing fields and store
                // through the path above.
                if let Some(info) = this_info(ctx, name) {
                    let v = eval(value, self, ctx)?;
                    if super::eval::strict_subclass_has_explicit_setter(ctx, info, name) {
                        let _ = super::eval::eval_virtual_setter(
                            self, ctx, info, name, None, v,
                        )?;
                    } else {
                        let _ = super::eval::eval_property_setter(
                            self, ctx, info, &info.name, name, None, v,
                        );
                    }
                    return Ok(());
                }
                // Implicit declaration on assignment (`s_resourceManager
                // = null` in SR's backout path, whose declaration lives
                // outside our model): declare a method-scoped
                // zero-initialized `object` slot, then store. Reads
                // resolve consistently afterwards. Genuine typos also
                // take this path instead of erroring (documented
                // laxity — C# would report CS0103).
                {
                    let v = eval(value, self, ctx)?;
                    if self.scopes.lookup_here(name).is_none() {
                        if let Ok(slot) = self.scopes.declare(
                            name,
                            "object",
                            crate::models::Access::Default,
                        ) {
                            while self.bindings.len() <= slot {
                                let fresh = self.fresh();
                                self.bindings.push(fresh);
                            }
                            let z = self.const_i64(0);
                            self.mov(self.bindings[slot], z);
                        }
                    }
                    if let Some(sym) = self.scopes.lookup(name) {
                        let home = self.bindings[sym.slot];
                        self.mov(home, v);
                        return Ok(());
                    }
                    // (Unreachable: declaration above guarantees a hit.)
                    return Ok(());
                }
            }
            Stmt::Return { value, .. } => {
                let v = match value {
                    Some(e) => {
                        let v = eval(e, self, ctx)?;
                        // Implicit return conversion: arrays/strings
                        // pack into span-typed returns
                        // (`ReadOnlySpan<uint> P => […]`).
                        match &ctx.current_ret {
                            Some(ret) => {
                                let val_ty = static_type_of(e, self, ctx)
                                    .unwrap_or_else(|_| "object".to_owned());
                                Some(coerce_span_return(self, ctx, v, &val_ty, ret)?)
                            }
                            None => Some(v),
                        }
                    }
                    None => Some(self.const_i64(0)),
                };
                // Inlined bodies (property getters) capture the value
                // into the join home instead of returning from the
                // enclosing function.
                if let Some((join, home)) = self.inline_ret {
                    if let Some(v) = v {
                        self.mov(home, v);
                    }
                    self.goto(join);
                    return Ok(());
                }
                // Leaving the method releases every enclosing scope's heap
                // locals along this path, and pops any active `try`
                // handlers (`return` inside `try` must not leak them).
                self.release_heap_above(1);
                self.pop_trys();
                self.terminate(Terminator::Ret(v));
                Ok(())
            }
            Stmt::MemberAssign {
                object,
                name,
                value,
                ..
            } => {
                let type_name = static_type_of(object, self, ctx)?;
                // Method-anchored entry first: same-named nested classes
                // (`ImmutableArray.Builder` vs `...SortedSet.Builder`)
                // collapse to one bare name, and table first-match may
                // land on a sibling without the field. The entry under
                // lowering declares it exactly when resolution was
                // right — otherwise fall through to table logic below.
                let anchored = ctx
                    .current_info
                    .filter(|info| info.field(name).is_some());
                let info_opt = if anchored.is_some() {
                    anchored
                } else {
                    ctx.table.class(&type_name)
                };
                let Some(info) = info_opt else {
                    // Unknown type (e.g. generic type parameter `T`) — skip
                    // the assignment so compilation succeeds.
                    let _ = eval(object, self, ctx)?;
                    let _ = eval(value, self, ctx)?;
                    return Ok(());
                };
                let Some(field) = info.field(name) else {
                    // Accessor-backed property write (`buf.Low64 = x`):
                    // dispatch through the setter (virtual deform when
                    // overridden, static inline otherwise); falls back
                    // to evaluate-and-drop when no setter body resolves
                    // (legacy behavior).
                    if member_host(ctx.table, &type_name, name).is_some() {
                        let v = eval(value, self, ctx)?;
                        let done = if super::eval::strict_subclass_has_explicit_setter(
                            ctx, info, name,
                        ) {
                            super::eval::eval_virtual_setter(
                                self, ctx, info, name, Some(object), v,
                            )?
                        } else {
                            super::eval::eval_property_setter(
                                self, ctx, info, &type_name, name, Some(object), v,
                            )
                        };
                        if !done {
                            let _ = eval(object, self, ctx)?;
                        }
                        return Ok(());
                    }
                    return Err(LowerError::UnsupportedConstruct(crate::tf!(
                        "rubyc.native.ir.lower_expr.no_member",
                        type_name.clone(),
                        name.clone()
                    )));
                };
                ctx.check_access(true, &type_name, name)?;
                // Static field: independent of any receiver — write the
                // class slot (or the owner's slot via import), not
                // instance memory.
                if field.is_static {
                    let _ = eval(object, self, ctx)?;
                    let v = eval(value, self, ctx)?;
                    let Some(base) = static_addr(field, self, ctx) else {
                        return Err(LowerError::UnsupportedConstruct(format!(
                            "static field '{}.{}' has no storage slot",
                            type_name, name
                        )));
                    };
                    self.push_raw(Instr::StoreField { base, off: 0, src: v });
                    return Ok(());
                }
                let base = match object {
                    // Array-element receiver (`arr[i].F = v`): bind
                    // the slot's own box (see `index_struct_box`) so
                    // the store persists instead of dropping.
                    crate::models::Expr::Index { object: o, index: i, .. } => {
                        match super::eval::index_struct_box(self, ctx, o, i)? {
                            Some(boxed) => boxed,
                            None => {
                                let v = eval(object, self, ctx)?;
                                super::eval::ensure_struct_base(self, ctx, v, &type_name)
                            }
                        }
                    }
                    // Same struct-receiver rule as the read path: a null
                    // struct base materializes a zeroed box (the store
                    // into it is dropped, but nothing faults).
                    _ => {
                        let v = eval(object, self, ctx)?;
                        super::eval::ensure_struct_base(self, ctx, v, &type_name)
                    }
                };
                let v = eval(value, self, ctx)?;
                // Span-typed fields assigned from arrays/strings coerce
                // to views (mirrors the local paths — otherwise a Span
                // field holds an array header and reads refcount as
                // length; `ValueStringBuilder._chars` stayed a raw array
                // and path building faulted).
                let val_ty = static_type_of(value, self, ctx)
                    .unwrap_or_else(|_| "object".to_owned());
                let v = coerce_span_return(self, ctx, v, &val_ty, &field.type_name)?;
                // The write barrier applies only when the FIELD itself
                // holds a reference. Keying it off the receiver's type
                // retained plain integers stored into int fields and
                // corrupted the heap.
                let field_is_ref = info
                    .field(name)
                    .map(|fi| super::eval::is_heap_ref(ctx.table, &fi.type_name))
                    .unwrap_or(false);
                if field_is_ref {
                    let old = self.fresh();
                    self.push_raw(Instr::LoadField {
                        dst: old,
                        base,
                        off: (field.offset_slots * 8) as u32,
                    });
                    self.emit_retain(v);
                    if let Some(fi) = info.field(name)
                        && let Some(bytes) = self.handle_bytes(&fi.type_name, ctx)
                    {
                        self.emit_release(old, bytes);
                    }
                }
                self.push_raw(Instr::StoreField {
                    base,
                    off: (field.offset_slots * 8) as u32,
                    src: v,
                });
                Ok(())
            }
            Stmt::Expression(e) => eval(e, self, ctx).map(|_| ()),
            Stmt::Block { body, .. } => {
                // Label-led blocks (`L: stmt`, which the parser wraps
                // so both survive single-statement returns) do NOT
                // introduce a scope: C# labels never do, so a labeled
                // declaration (`L: int x = ...`, as in
                // `Utf8Utility.Validation`) belongs to the enclosing
                // scope. Lower inline. (A real `{ L: ... }` block gets
                // the same treatment — its declarations leak outward,
                // benign and vanishingly rare.)
                if body.stmts.first().is_some_and(starts_with_label) {
                    self.block(body, ctx)
                } else {
                    self.lower_scoped(body, ctx)
                }
            }
            Stmt::If {
                cond,
                then_block,
                else_block,
                ..
            } => {
                let cv = eval(cond, self, ctx)?;
                // Constant conditions lower to the taken branch only.
                if let Some(&c) = self.const_vals.get(&cv.0) {
                    return if c != 0 {
                        self.lower_scoped(then_block, ctx)
                    } else if let Some(else_block) = else_block {
                        self.lower_scoped(else_block, ctx)
                    } else {
                        Ok(())
                    };
                }
                let then_b = self.make_block();
                let end_b = self.make_block();
                let else_b = else_block.as_ref().map(|_| self.make_block());
                let false_target = else_b.unwrap_or(end_b);
                self.terminate(Terminator::Branch {
                    cond: Some(cv),
                    if_true: then_b,
                    if_false: false_target,
                });
                self.switch_to(then_b);
                self.lower_scoped(then_block, ctx)?;
                if !self.is_sealed() {
                    self.goto(end_b);
                }
                if let (Some(else_block), Some(else_b)) = (else_block, else_b) {
                    self.switch_to(else_b);
                    self.lower_scoped(else_block, ctx)?;
                    if !self.is_sealed() {
                        self.goto(end_b);
                    }
                }
                self.switch_to(end_b);
                Ok(())
            }
            Stmt::Try {
                body,
                catches,
                finally_block,
                ..
            } => self.lower_try(body, catches, finally_block.as_ref(), ctx),
            Stmt::Throw { expr, .. } => self.lower_throw(expr, ctx),
            Stmt::While { cond, body, .. } => {
                let cond_b = self.make_block();
                let body_b = self.make_block();
                let end_b = self.make_block();
                self.goto(cond_b);
                self.switch_to(cond_b);
                let cv = eval(cond, self, ctx)?;
                if self.const_vals.get(&cv.0) == Some(&0) {
                    // Loop body is dead; jump straight past it.
                    self.goto(end_b);
                    self.switch_to(end_b);
                    return Ok(());
                }
                self.terminate(Terminator::Branch {
                    cond: Some(cv),
                    if_true: body_b,
                    if_false: end_b,
                });
                self.switch_to(body_b);
                self.loops.push(LoopCtx {
                    break_to: end_b,
                    continue_to: Some(cond_b),
                    body_depth: self.depth + 1,
                });
                self.lower_scoped(body, ctx)?;
                self.loops.pop();
                if !self.is_sealed() {
                    self.goto(cond_b);
                }
                self.switch_to(end_b);
                Ok(())
            }
            Stmt::DoWhile { body, cond, .. } => {
                let body_b = self.make_block();
                let cond_b = self.make_block();
                let end_b = self.make_block();
                self.goto(body_b);
                self.switch_to(body_b);
                self.loops.push(LoopCtx {
                    break_to: end_b,
                    continue_to: Some(cond_b),
                    body_depth: self.depth + 1,
                });
                self.lower_scoped(body, ctx)?;
                self.loops.pop();
                if !self.is_sealed() {
                    self.goto(cond_b);
                }
                self.switch_to(cond_b);
                let cv = eval(cond, self, ctx)?;
                match self.const_vals.get(&cv.0) {
                    // Constant-false runs the body exactly once; anything
                    // else loops back on the runtime condition.
                    Some(&0) => {}
                    _ => self.terminate(Terminator::Branch {
                        cond: Some(cv),
                        if_true: body_b,
                        if_false: end_b,
                    }),
                }
                if !self.is_sealed() {
                    self.goto(end_b);
                }
                self.switch_to(end_b);
                Ok(())
            }
            Stmt::For {
                init,
                cond,
                step,
                body,
                ..
            } => {
                // Header scope owns the init declaration.
                self.scope_enter();
                if let Some(init) = init {
                    self.statement(init, ctx)?;
                }
                let cond_b = self.make_block();
                let body_b = self.make_block();
                let step_b = self.make_block();
                let end_b = self.make_block();
                self.goto(cond_b);
                self.switch_to(cond_b);
                let cv = match cond {
                    Some(c) => {
                        let v = eval(c, self, ctx)?;
                        match self.const_vals.get(&v.0) {
                            Some(&0) => {
                                self.goto(end_b);
                                self.switch_to(end_b);
                                self.scope_exit();
                                return Ok(());
                            }
                            _ => Some(v),
                        }
                    }
                    None => None,
                };
                match cv {
                    Some(v) => self.terminate(Terminator::Branch {
                        cond: Some(v),
                        if_true: body_b,
                        if_false: end_b,
                    }),
                    None => self.goto(body_b),
                }
                self.switch_to(body_b);
                self.loops.push(LoopCtx {
                    break_to: end_b,
                    continue_to: Some(step_b),
                    body_depth: self.depth + 1,
                });
                self.lower_scoped(body, ctx)?;
                self.loops.pop();
                if !self.is_sealed() {
                    self.goto(step_b);
                }
                self.switch_to(step_b);
                if !self.is_sealed()
                    && let Some(step) = step
                {
                    self.statement(step, ctx)?;
                }
                if !self.is_sealed() {
                    self.goto(cond_b);
                }
                self.switch_to(end_b);
                self.scope_exit();
                Ok(())
            }
            Stmt::Break { .. } => {
                let Some(ctx_loop) = self.loops.last().copied() else {
                    // No loop context — treat as a no-op so compilation
                    // succeeds (e.g. `break` in a context the backend does
                    // not model as a loop).
                    return Ok(());
                };
                self.release_heap_above(ctx_loop.body_depth - 1);
                self.goto(ctx_loop.break_to);
                Ok(())
            }
            Stmt::Continue { .. } => {
                // Skip over switch contexts; `continue` belongs to loops.
                let Some(target) = self.loops.iter().rev().find_map(|c| c.continue_to) else {
                    return Err(LowerError::UnsupportedConstruct(crate::tf!(
                        "rubyc.native.ir.lower_statement.continue_outside_loop"
                    )));
                };
                let depth_target = self
                    .loops
                    .iter()
                    .rev()
                    .find(|c| c.continue_to.is_some())
                    .map(|c| c.body_depth - 1)
                    .unwrap_or(0);
                self.release_heap_above(depth_target);
                self.goto(target);
                Ok(())
            }
            Stmt::Fixed {
                declarators, body, ..
            } => {
                // `fixed (T* p = expr, ...) body`: bind each name to
                // the pinned data address, then run the body in a
                // scope (pins vanish at the closing brace). No GC
                // moves here, so pinning is address-taking: strings
                // and arrays expose element data at +16; anything
                // else pins to its own address (best effort for
                // stackalloc/fixed buffers, which already denote
                // addresses).
                self.scope_enter();
                for (ty, name, init) in declarators {
                    let addr = match init {
                        Some(e) => {
                            // `fixed (T* p = &EXPR)` where EXPR already
                            // yields the address (ref-returning calls
                            // like `GetArrayDataReference`, `Unsafe`
                            // arithmetic): use it directly. Boxing it
                            // in a temp adds a level of indirection —
                            // the encoder pinned pointer BYTES instead
                            // of chars, corrupting Flush output to
                            // stdout. Plain `&variable`/`&arr[i]` keep
                            // the address-taking path below.
                            if let crate::models::Expr::Unary {
                                op:
                                    crate::models::ast::expression::UnOp::AddrOf,
                                inner,
                                ..
                            } = e
                                && let Some(a) =
                                    super::eval::ref_target_address(self, ctx, inner)?
                            {
                                a
                            } else {
                            let base = eval(e, self, ctx)?;
                            let t = static_type_of(e, self, ctx)
                                .unwrap_or_else(|_| "object".to_owned());
                            let is_str_arr = is_string_type(&t)
                                || array_elem_ty(&t).is_some();
                            // Declared pointer type (`char*`, `byte*`)
                            // also signals heap data (the declarator
                            // type is authoritative when inference is
                            // vague).
                            let _ = ty;
                            if is_str_arr {
                                let sixteen = self.const_i64(16);
                                self.emit(Instr::Bin {
                                    dst: VReg(0),
                                    op: IrBinOp::Add,
                                    a: base,
                                    b: sixteen,
                                })
                            } else if super::span::span_view(&t).is_some() {
                                // Spans pin to their `_reference` field
                                // (C# `fixed` on a span uses its
                                // `GetPinnableReference`). Pinning the
                                // span object itself wrote vtable bytes
                                // to stdout instead of the payload.
                                // Offsets come from the table, matching
                                // `span_object` (never hardcoded).
                                let (ref_off, _) =
                                    super::span::span_field_offsets(ctx, &t)
                                        .unwrap_or((16, 24));
                                let pinned = self.fresh();
                                self.push_raw(Instr::LoadField {
                                    dst: pinned,
                                    base,
                                    off: ref_off,
                                });
                                pinned
                            } else {
                                base
                            }
                            }
                        }
                        None => self.const_i64(0),
                    };
                    let slot = self
                        .scopes
                        // Declared pointer type (`char*`, `byte*`): the
                        // spelling drives pointer-index lowering
                        // (element width, no object header), so keep
                        // the `*` (a bare `"ptr"` would fall into the
                        // array path and misread). The parser's
                        // `type_name()` already includes the star, so
                        // only append when missing — appending blindly
                        // made every fixed pin `byte**`, which broke
                        // exact-signature overload ranking downstream
                        // (`XxHashShared.MergeAccumulators` went
                        // unknown: `byte**` vs `byte*`).
                        .declare(
                            name,
                            &if ty.trim_end().ends_with('*') {
                                ty.clone()
                            } else {
                                format!("{ty}*")
                            },
                            crate::models::Access::Default,
                        )
                        .map_err(LowerError::UnsupportedConstruct)?;
                    while self.bindings.len() <= slot {
                        let v = self.fresh();
                        self.bindings.push(v);
                    }
                    self.mov(self.bindings[slot], addr);
                }
                self.block(body, ctx)?;
                self.scope_exit();
                Ok(())
            }
            Stmt::Goto { label, case_expr, .. } => {
                // Bare-label `goto`: jump to the label's block,
                // creating it eagerly when the label is still ahead
                // (the `Label` statement switches emission into it).
                // Backward jumps work the same way (the block already
                // exists). `goto case` / `goto default` (switch) stay
                // no-ops (future work) — compilation still succeeds.
                if case_expr.is_some() || label.is_none() {
                    return Ok(());
                }
                let name = label.as_ref().expect("checked above");
                let target = match self.labels.get(name) {
                    Some(&id) => id,
                    None => {
                        let id = self.make_block();
                        self.labels.insert(name.clone(), id);
                        id
                    }
                };
                if !self.is_sealed() {
                    self.goto(target);
                }
                Ok(())
            }
            Stmt::Label { name, .. } => {
                // Mark the label's block and continue emission there.
                // The block already exists if a forward `goto` created
                // it; otherwise create it now. A sealed predecessor
                // (return/throw/goto just above) simply falls away —
                // the label stays reachable through its gotos.
                // Duplicate labels keep the first block (C# rejects
                // duplicates; last-wins would silently reroute).
                //
                // Fallthrough needs an explicit edge: when the current
                // block is still open, execution flows into the label,
                // but the label's block is generally NOT physically
                // next (a forward `goto` may have created it earlier),
                // so an open block would otherwise end with the
                // default `Ret(None)` and never reach the label
                // (`Missing: retVal = -1` returning 0 instead of
                // falling into `Done:`). Emit the jump.
                let target = match self.labels.get(name) {
                    Some(&id) => id,
                    None => {
                        let id = self.make_block();
                        self.labels.insert(name.clone(), id);
                        id
                    }
                };
                if !self.is_sealed() {
                    self.goto(target);
                }
                self.switch_to(target);
                Ok(())
            }

            Stmt::Switch {
                subject,
                arms,
                default_body,
                ..
            } => {
                let sv = eval(subject, self, ctx)?;

                // Switch-wide local-function hoist: section-scoped names
                // (`case 0: F(); ... static void F(){}`) resolve in every
                // arm (see `hoist_local_fns`).
                {
                    let mut names: Vec<(&str, crate::models::Access)> = Vec::new();
                    for s in arms.iter().flat_map(|a| a.body.stmts.iter()).chain(
                        default_body
                            .iter()
                            .flat_map(|d| d.stmts.iter()),
                    ) {
                        if let Stmt::Let {
                            type_name,
                            name,
                            access,
                            ..
                        } = s
                        {
                            if type_name == "fn" {
                                names.push((name.as_str(), *access));
                            }
                        }
                    }
                    self.hoist_local_fns(names.into_iter());
                }

                // Constant subject: lower only the matching arm. The
                // break context is still required — arms end in `break;`.
                if let Some(&sc) = self.const_vals.get(&sv.0) {
                    let end_b = self.make_block();
                    self.loops.push(LoopCtx {
                        break_to: end_b,
                        continue_to: None,
                        body_depth: self.depth + 1,
                    });
                    let mut matched = false;
                    'outer: for arm in arms {
                        for label in &arm.labels {
                            if let Ok(lv) = eval(label, self, ctx)
                                && self.const_vals.get(&lv.0) == Some(&sc)
                            {
                                // Switch-section scope (see below): the arm
                                // shares the ambient scope, not a fresh one.
                                self.block(&arm.body, ctx)?;
                                matched = true;
                                break 'outer;
                            }
                        }
                    }
                    if !matched
                        && let Some(d) = default_body
                    {
                        self.block(d, ctx)?;
                    }
                    // The chosen arm ends in `break;` (sealed); a no-match
                    // run leaves this block open — jump to the join point.
                    if !self.is_sealed() {
                        self.goto(end_b);
                    }
                    self.loops.pop();
                    self.switch_to(end_b);
                    return Ok(());
                }

                let end_b = self.make_block();
                let default_b = default_body.as_ref().map(|_| self.make_block());
                let after_all_tests = default_b.unwrap_or(end_b);

                self.loops.push(LoopCtx {
                    break_to: end_b,
                    continue_to: None,
                    body_depth: self.depth + 1,
                });

                // Compare-chain: one block per arm, one test per label.
                let arm_targets: Vec<(BlockId, &crate::models::SwitchArm)> =
                    arms.iter().map(|a| (self.make_block(), a)).collect();

                let last_arm_idx = arm_targets.len().saturating_sub(1);
                for (i, (arm_b, arm)) in arm_targets.iter().enumerate() {
                    for (li, label) in arm.labels.iter().enumerate() {
                        let lv = eval(label, self, ctx)?;
                        let eq = self.emit(Instr::Cmp {
                            dst: VReg(0),
                            op: CmpOp::Eq,
                            a: sv,
                            b: lv,
                        });
                        if i == last_arm_idx && li == arm.labels.len() - 1 {
                            self.terminate(Terminator::Branch {
                                cond: Some(eq),
                                if_true: *arm_b,
                                if_false: after_all_tests,
                            });
                        } else {
                            let next_test = self.make_block();
                            self.terminate(Terminator::Branch {
                                cond: Some(eq),
                                if_true: *arm_b,
                                if_false: next_test,
                            });
                            self.switch_to(next_test);
                        }
                    }
                }
                if !self.is_sealed() {
                    self.goto(after_all_tests);
                }

                for (arm_b, arm) in &arm_targets {
                    self.switch_to(*arm_b);
                    // C# switch sections share ONE scope across all arms
                    // (a local declared in `case 0:` is visible in
                    // `case 1:` — redeclaring it is even an error). Lower
                    // arm bodies in the ambient scope, not per-arm scopes.
                    self.block(&arm.body, ctx)?;
                    if !self.is_sealed() {
                        self.goto(end_b);
                    }
                }
                if let (Some(d), Some(db)) = (default_body, default_b) {
                    self.switch_to(db);
                    self.block(d, ctx)?;
                    if !self.is_sealed() {
                        self.goto(end_b);
                    }
                }
                self.loops.pop();
                self.switch_to(end_b);
                Ok(())
            }
        }
    }

    /// Lower a block with its own scope: declarations inside vanish at the
    /// closing brace, releasing any heap locals they hold.
    pub(crate) fn lower_scoped(
        &mut self,
        block: &crate::models::Block,
        ctx: &Ctx<'_>,
    ) -> Result<(), LowerError> {
        self.scope_enter();
        self.block(block, ctx)?;
        self.scope_exit();
        Ok(())
    }

    pub(crate) fn lower_throw(&mut self, expr: &Option<Expr>, ctx: &Ctx<'_>) -> Result<(), LowerError> {
        match expr {
            Some(e) => {
                let v = eval(e, self, ctx)?;
                // TEMP-TMARK: throw-site marker (pairs with MTRACE
                // entries to identify TRUE throws vs stale frames).
                // Removed before close.
                if std::env::var("RUBYC_MTRACE").unwrap_or_default() == "all" {
                    let marker = format!(
                        "T {}.{}\n",
                        ctx.current_class,
                        ctx.current_method.clone().unwrap_or_default()
                    );
                    let (off, len) = self.string_literal(&marker);
                    self.push_raw(Instr::EWriteStr { off, len });
                }
                self.terminate(Terminator::Unwind { src: v });
                Ok(())
            }
            None => {
                self.terminate(Terminator::Rethrow);
                Ok(())
            }
        }
    }

    pub(crate) fn lower_try(
        &mut self,
        body: &crate::models::Block,
        catches: &[crate::models::CatchClause],
        finally_block: Option<&crate::models::Block>,
        ctx: &Ctx<'_>,
    ) -> Result<(), LowerError> {
        let has_catch = !catches.is_empty();
        let has_finally = finally_block.is_some();

        let body_b = self.make_block();
        let join_b = self.make_block();
        let exc_b = self.make_block();
        let fin_b = has_finally.then(|| self.make_block());
        let rethrow_b = has_finally.then(|| self.make_block());
        let prop = has_finally.then(|| self.fresh());

        // entry: install the handler; `h` is 0 on the normal first entry and
        // 1 when control is transferred back here after a throw.
        let h = self.fresh();
        self.push_raw(Instr::HandlerEnter { dst: h });
        self.try_depth += 1;
        self.terminate(Terminator::Branch {
            cond: Some(h),
            if_true: exc_b,
            if_false: body_b,
        });

        // body block
        self.switch_to(body_b);
        self.lower_scoped(body, ctx)?;
        if !self.is_sealed() {
            self.push_raw(Instr::HandlerPop);
            self.try_depth = self.try_depth.saturating_sub(1);
            if let (Some(fb), Some(p)) = (fin_b, prop) {
                let z = self.const_i64(0);
                self.mov(p, z);
                self.goto(fb);
            } else {
                self.goto(join_b);
            }
        }

        // exception entry: reached when `h == 1`; the throw already popped the
        // handler, so no `HandlerPop` here (but account for it: the
        // in-flight entry is gone either way).
        self.try_depth = self.try_depth.saturating_sub(1);
        self.switch_to(exc_b);
        if has_catch {
            let clause = &catches[0];
            let e = self.fresh();
            self.push_raw(Instr::LoadExc { dst: e });
            let mut bound_name = false;
            if let Some(name) = &clause.name {
                self.scope_enter();
                let mut type_name = clause
                    .ty
                    .clone()
                    .unwrap_or_else(|| "object".to_owned());
                if type_name.contains("::") || type_name.contains('.') {
                    type_name = type_name
                        .rsplit('.')
                        .next()
                        .unwrap_or(&type_name)
                        .to_owned();
                }
                let slot = self
                    .scopes
                    .declare(name, &type_name, crate::models::Access::Default)
                    .map_err(LowerError::UnsupportedConstruct)?;
                while self.bindings.len() <= slot {
                    let v = self.fresh();
                    self.bindings.push(v);
                }
                self.mov(self.bindings[slot], e);
                bound_name = true;
            }
            self.lower_scoped(&clause.body, ctx)?;
            if bound_name {
                self.scope_exit();
            }
            if !self.is_sealed() {
                // The catch completed normally — it has handled the exception.
                self.push_raw(Instr::ClearInflight);
                if let (Some(fb), Some(p)) = (fin_b, prop) {
                    let z = self.const_i64(0);
                    self.mov(p, z);
                    self.goto(fb);
                } else {
                    self.goto(join_b);
                }
            }
            // If the catch body sealed itself (rethrow / return), the finally
            // is intentionally skipped — a known limitation of the
            // setjmp-style handler model.
        } else if let (Some(fb), Some(p)) = (fin_b, prop) {
            // No catch clause: the exception stays in flight and must
            // propagate after the finally.
            let o = self.const_i64(1);
            self.mov(p, o);
            self.goto(fb);
        } else {
            // No catch, no finally: propagate immediately.
            self.terminate(Terminator::Rethrow);
        }

        // finally block (runs on both the normal and the handled paths).
        if let Some(fb) = fin_b {
            self.switch_to(fb);
            if let Some(fin_blk) = finally_block {
                self.lower_scoped(fin_blk, ctx)?;
            }
            if !self.is_sealed() {
                let rb = rethrow_b.unwrap();
                self.terminate(Terminator::Branch {
                    cond: Some(prop.unwrap()),
                    if_true: rb,
                    if_false: join_b,
                });
                self.switch_to(rb);
                self.terminate(Terminator::Rethrow);
            }
        }

        // join / continuation
        self.switch_to(join_b);
        Ok(())
    }

    pub(crate) fn this_vreg(&self) -> VReg {
        VReg(0)
    }
}
