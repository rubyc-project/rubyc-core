//! AST to IR body lowering.

use super::super::{BlockId, CmpOp, Instr, IrFunction, LowerError, Terminator, VReg};
use crate::native::ir::{ReceiverKind, StringArena};
use crate::models::symbols::ScopeStack;
use crate::models::{Expr, Stmt};
use crate::native::ir::BinOp as IrBinOp;
use crate::native::ir::Ctx;
use crate::native::semantic::{FIRST_FIELD_SLOT, INSTANCE_HEADER_SLOTS, RUNTIME_TYPE_SLOT};
use super::{calls::*, eval::*, select::*, types::*};
use super::span::coerce_span_return;
use super::typeinfo::*;

/// Refcount word of a string object. The high bit tags the runtime type:
/// class objects carry a plain count (always positive), strings set the
/// sign bit, so `is string` is a single signed test of [obj+8]. Retain
/// increments the low count; Release masks the tag bit before the free
/// check. Boxing will give primitives wrapper rows at [0] instead.
pub(crate) const STRING_REF: i64 = i64::MIN | 1;

/// True when `block` contains a `return` that belongs to this function.
/// Local functions and lambdas are separate functions, so their bodies
/// are not descended into; this is only used to decide whether a
/// try/finally needs the in-flight return slot initialized.
fn block_has_return(block: &crate::models::Block) -> bool {
    block.stmts.iter().any(stmt_has_return)
}

fn stmt_has_return(s: &Stmt) -> bool {
    match s {
        Stmt::Return { .. } => true,
        Stmt::If {
            then_block,
            else_block,
            ..
        } => {
            block_has_return(then_block)
                || else_block.as_ref().is_some_and(block_has_return)
        }
        Stmt::Block { body, .. }
        | Stmt::CheckedBlock { body, .. }
        | Stmt::While { body, .. }
        | Stmt::DoWhile { body, .. }
        | Stmt::For { body, .. }
        | Stmt::Fixed { body, .. } => block_has_return(body),
        Stmt::Switch {
            arms,
            default_body,
            ..
        } => {
            arms.iter().any(|a| block_has_return(&a.body))
                || default_body.as_ref().is_some_and(block_has_return)
        }
        Stmt::Try {
            body,
            catches,
            finally_block,
            ..
        } => {
            block_has_return(body)
                || catches.iter().any(|c| block_has_return(&c.body))
                || finally_block.as_ref().is_some_and(block_has_return)
        }
        Stmt::LocalFunction { .. } => false,
        _ => false,
    }
}

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
    /// Canonical type descriptors and the static slots that cache them.
    pub typeinfo: &'a std::cell::RefCell<TypeRegistry>,
    /// Type parameter name -> hidden type-handle vreg for this function.
    /// The DELEGATE type the next expression is expected to have. A
    /// lambda or a method group takes its runtime type from the
    /// binding it is converted to, and the binding site is what knows
    /// it. `None` when the expression has no expected type.
    pub expected_delegate: Option<String>,
    pub type_handles: std::collections::BTreeMap<String, VReg>,
    /// `where T : ...` clauses of the method under lowering. A member or
    /// indexer on a type-parameter receiver binds through them (C#:
    /// `x[i]` uses the constraint's indexer).
    pub constraints: Vec<crate::models::ast::method::TypeConstraint>,
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
    /// Enclosing `try` statements that carry a `finally`, innermost
    /// last: `(finally block, mode vreg, scope depth at the try)`. A
    /// `return` inside one must run the finally before it returns, so
    /// the return is routed here instead of sealing the block with
    /// `Ret`. The depth bounds which heap locals the return releases on
    /// the way out: everything declared INSIDE the try body dies, but
    /// locals of the enclosing block (the `using` resource, for one)
    /// must stay alive for the finally to use.
    pub(crate) fin_ctx: Vec<(BlockId, VReg, usize)>,
    /// Where a routed return keeps its value, and the flag that marks a
    /// return in flight (1). Shared by every finally in the method: the
    /// innermost finally completes the return or hands it to the next
    /// enclosing one.
    pub(crate) ret_home: Option<VReg>,
    pub(crate) ret_flag: Option<VReg>,
    /// Nesting depth of `checked(...)` in the expression being lowered.
    /// Non-zero means integer arithmetic and narrowing conversions must
    /// raise `System.OverflowException` instead of wrapping.
    pub(crate) checked_depth: usize,
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
    fin_ctx_len: usize,
    ret_home: Option<VReg>,
    ret_flag: Option<VReg>,
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

/// Width/signedness for `checked` overflow bounds.
#[derive(Clone, Copy)]
pub(crate) enum CheckedKind {
    Int32,
    UInt32,
    Int64,
}

impl<'a> Body<'a> {
    pub(crate) fn new(
        f: &'a mut IrFunction,
        strings: &'a mut StringArena,
        typeinfo: &'a std::cell::RefCell<TypeRegistry>,
    ) -> Self {
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
            typeinfo,
            expected_delegate: None,
            type_handles: std::collections::BTreeMap::new(),
            constraints: Vec::new(),
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
            fin_ctx: Vec::new(),
            ret_home: None,
            ret_flag: None,
            checked_depth: 0,
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
            fin_ctx_len: self.fin_ctx.len(),
            ret_home: self.ret_home,
            ret_flag: self.ret_flag,
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
        self.fin_ctx.truncate(cp.fin_ctx_len);
        self.ret_home = cp.ret_home;
        self.ret_flag = cp.ret_flag;
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
        Some((info.fields.len() + INSTANCE_HEADER_SLOTS) as u64 * 8)
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
        type_name: Option<&str>,
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
        // The box's RUNTIME TYPE: a generic struct's box is
        // materialized from the class entry, which does not name the
        // instantiation, so the type comes from the spelling the box
        // was asked for. Without it the box cannot answer `typeof(T)`.
        let runtime = match type_name {
            Some(name) if !entry.generic_params.is_empty() => {
                match self.type_token(name, ctx) {
                    Ok(desc) => desc,
                    // A spelling that still mentions a type parameter
                    // has no concrete runtime type; the slot stays null.
                    Err(_) => self.const_i64(0),
                }
            }
            _ => self.const_i64(0),
        };
        self.push_raw(Instr::StoreField {
            base: boxed,
            off: (RUNTIME_TYPE_SLOT * 8) as u32,
            src: runtime,
        });
        let z = self.const_i64(0);
        let mut off = (FIRST_FIELD_SLOT * 8) as u32;
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

    /// True when a type spelling is `object`, `dynamic`, or an
    /// interface: a primitive stored there must be boxed.
    pub(crate) fn is_object_target(type_name: &str, ctx: &Ctx<'_>) -> bool {
        let b = crate::native::ir::lower::select::base_type_name(type_name);
        if matches!(b.as_str(), "object" | "Object" | "dynamic") {
            return true;
        }
        ctx.table.interfaces.iter().any(|i| {
            i.name == type_name
                || i.name == b
                || i.name.rsplit('.').next() == Some(b.as_str())
        }) || ctx.table.classes.iter().any(|c| {
            c.interfaces.iter().any(|(n, _)| {
                n == type_name
                    || n == &b
                    || n.rsplit('.').next() == Some(b.as_str())
            })
        })
    }

    /// True when an expression's DECLARED type is a reference, so its
    /// value really is a pointer (and a box header may be read). An
    /// expression whose type the compiler cannot establish is NOT a
    /// reference: guessing would read a raw value as a pointer.
    pub(crate) fn reference_source(
        expr: &Expr,
        b: &Body<'_>,
        ctx: &Ctx<'_>,
    ) -> bool {
        let Some(ty) = declared_type_of(expr, b, ctx) else {
            return false;
        };
        let base = crate::native::ir::lower::select::base_type_name(&ty);
        Self::is_object_target(&ty, ctx)
            || matches!(base.as_str(), "string" | "String" | "Object")
            || ty.trim_end().ends_with("[]")
            || ctx
                .table
                .class(&base)
                .is_some_and(|i| !i.is_struct && !i.is_enum)
    }

    /// Coerce a value crossing an object/interface boundary. Boxing is
    /// only for value types; unboxing is the reverse direction and is
    /// SAFE here (pattern bindings are the only implicit source and are
    /// read only after the type test).
    pub(crate) fn coerce_boxing(
        &mut self,
        value: VReg,
        src_ty: &str,
        dst_ty: &str,
        allow_unbox: bool,
        ctx: &Ctx<'_>,
    ) -> Result<VReg, LowerError> {
        let src = crate::native::ir::lower::select::base_type_name(src_ty);
        let dst = crate::native::ir::lower::select::base_type_name(dst_ty);
        if primitive_id(&src).is_some() && Self::is_object_target(dst_ty, ctx) {
            return self.box_value(value, src_ty, ctx);
        }
        if allow_unbox
            && primitive_id(&dst).is_some()
            && Self::is_object_target(src_ty, ctx)
        {
            return self.unbox_value(value, dst_ty, false, ctx);
        }
        Ok(value)
    }

    /// Box a primitive value: wrapper row (or a sentinel when the
    /// primitive has no declared wrapper), refcount tag, value, and the
    /// canonical descriptor. `[0]` stays a valid object/vtable word so
    /// existing reference operations keep working; the refcount tag is
    /// what marks it as a box.
    pub(crate) fn box_value(
        &mut self,
        value: VReg,
        type_name: &str,
        ctx: &Ctx<'_>,
    ) -> Result<VReg, LowerError> {
        let simple = crate::native::ir::lower::select::base_type_name(type_name);
        let tag = if let Some(info) = ctx
            .table
            .class(type_name)
            .or_else(|| ctx.table.class(&simple))
            .filter(|i| i.is_struct || i.is_enum)
        {
            let row = super::span::vtable_row_for(ctx, &info.name, &info.key.path);
            let t = self.fresh();
            self.vtable_addr(t, row);
            t
        } else if let Some(tag) = box_sentinel(&simple) {
            self.const_i64(tag)
        } else {
            return Err(LowerError::UnsupportedConstruct(format!(
                "boxing `{type_name}` into an object is not implemented for this type"
            )));
        };
        let desc = self.type_token(type_name, ctx)?;
        let obj = self.fresh();
        self.push_raw(Instr::Alloc { dst: obj, bytes: BOX_BYTES });
        self.push_raw(Instr::StoreField { base: obj, off: 0, src: tag });
        let refc = self.const_i64(BOX_REF);
        self.push_raw(Instr::StoreField { base: obj, off: 8, src: refc });
        self.push_raw(Instr::StoreField { base: obj, off: 16, src: value });
        self.push_raw(Instr::StoreField { base: obj, off: 24, src: desc });
        Ok(obj)
    }

    /// Unbox `value` as `type_name`. A null or wrong box selects zero
    /// when `strict` is false (pattern bindings are only read after the
    /// test); strict casts raise `InvalidCastException`.
    pub(crate) fn unbox_value(
        &mut self,
        value: VReg,
        type_name: &str,
        strict: bool,
        ctx: &Ctx<'_>,
    ) -> Result<VReg, LowerError> {
        let desc = self.type_token(type_name, ctx)?;
        let zero = self.const_i64(0);
        let is_null = self.emit(Instr::Cmp {
            dst: VReg(0),
            op: CmpOp::Eq,
            a: value,
            b: zero,
        });
        let res = self.fresh();
        let null_b = self.make_block();
        let test_b = self.make_block();
        let join_b = self.make_block();
        self.terminate(Terminator::Branch {
            cond: Some(is_null),
            if_true: null_b,
            if_false: test_b,
        });
        self.switch_to(null_b);
        if strict {
            self.throw_invalid_cast(ctx)?;
        } else {
            self.mov(res, zero);
            self.goto(join_b);
        }
        self.switch_to(test_b);
        let refc = self.fresh();
        self.push_raw(Instr::LoadField { dst: refc, base: value, off: 8 });
        let flag = self.const_i64(BOX_FLAG);
        let flagged = self.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::BitAnd,
            a: refc,
            b: flag,
        });
        let has_flag = self.emit(Instr::Cmp {
            dst: VReg(0),
            op: CmpOp::Ne,
            a: flagged,
            b: zero,
        });
        let objdesc = self.fresh();
        self.push_raw(Instr::LoadField { dst: objdesc, base: value, off: 24 });
        let same = self.emit(Instr::Cmp {
            dst: VReg(0),
            op: CmpOp::Eq,
            a: objdesc,
            b: desc,
        });
        let ok = self.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::BitAnd,
            a: has_flag,
            b: same,
        });
        if strict {
            self.throw_invalid_cast_if(ok, ctx)?;
            let v = self.fresh();
            self.push_raw(Instr::LoadField { dst: v, base: value, off: 16 });
            self.mov(res, v);
        } else {
            let bad_b = self.make_block();
            let good_b = self.make_block();
            self.terminate(Terminator::Branch {
                cond: Some(ok),
                if_true: good_b,
                if_false: bad_b,
            });
            self.switch_to(good_b);
            let v = self.fresh();
            self.push_raw(Instr::LoadField { dst: v, base: value, off: 16 });
            self.mov(res, v);
            self.goto(join_b);
            self.switch_to(bad_b);
            self.mov(res, zero);
            self.goto(join_b);
        }
        self.switch_to(join_b);
        Ok(res)
    }

    fn throw_invalid_cast(&mut self, ctx: &Ctx<'_>) -> Result<(), LowerError> {
        let ex = Expr::New {
            class_name: "System.InvalidCastException".to_owned(),
            args: Vec::new(),
            init: Vec::new(),
            span: Default::default(),
        };
        let obj = eval(&ex, self, ctx)?;
        self.terminate(Terminator::Unwind { src: obj });
        Ok(())
    }

    /// Raise `InvalidCastException` when `ok` is zero, then continue.
    fn throw_invalid_cast_if(
        &mut self,
        ok: VReg,
        ctx: &Ctx<'_>,
    ) -> Result<(), LowerError> {
        let zero = self.const_i64(0);
        let bad = self.emit(Instr::Cmp {
            dst: VReg(0),
            op: CmpOp::Eq,
            a: ok,
            b: zero,
        });
        let throw_b = self.make_block();
        let cont_b = self.make_block();
        self.terminate(Terminator::Branch {
            cond: Some(bad),
            if_true: throw_b,
            if_false: cont_b,
        });
        self.switch_to(throw_b);
        self.throw_invalid_cast(ctx)?;
        self.switch_to(cont_b);
        Ok(())
    }

    /// Build the descriptor for one interned type. All words are
    /// written: the allocator does not promise zeroed memory. The name
    /// strings are made permanent before they are stored, because the
    /// descriptor outlives every local.
    fn build_type_descriptor(
        &mut self,
        req: &TypeRequest,
        ctx: &Ctx<'_>,
    ) -> Result<VReg, LowerError> {
        let obj = self.fresh();
        self.push_raw(Instr::Alloc { dst: obj, bytes: TYPE_BYTES });
        let tag = self.const_i64(TYPE_TAG);
        self.push_raw(Instr::StoreField { base: obj, off: 0, src: tag });
        let refc = self.const_i64(TYPE_REF);
        self.push_raw(Instr::StoreField { base: obj, off: 8, src: refc });
        for (off, lit) in [
            (OFF_NAME, req.name),
            (OFF_FULL_NAME, req.full),
            (OFF_NAMESPACE, req.namespace),
        ] {
            let s = self.build_utf16_string(lit.0, lit.1);
            let immortal = self.const_i64(STRING_IMMORTAL_REF);
            self.push_raw(Instr::StoreField { base: s, off: 8, src: immortal });
            self.push_raw(Instr::StoreField { base: obj, off, src: s });
        }
        let flags = self.const_i64(req.flags as i64);
        self.push_raw(Instr::StoreField { base: obj, off: OFF_FLAGS, src: flags });
        let zero = self.const_i64(0);
        for off in [48, 72] {
            self.push_raw(Instr::StoreField { base: obj, off, src: zero });
        }
        // A generic instantiation carries its arguments: the array at
        // +56 holds one descriptor per argument, and +64 says how
        // many. `typeof(T)` inside a method of the generic class
        // reads the argument it names out of the instance's runtime
        // type, which is this descriptor.
        if !req.type_args.is_empty() {
            let n = req.type_args.len();
            let args = self.fresh();
            self.push_raw(Instr::Alloc {
                dst: args,
                bytes: (n as u64) * 8,
            });
            for (i, arg) in req.type_args.iter().enumerate() {
                let d = self.type_token(arg, ctx)?;
                self.push_raw(Instr::StoreField {
                    base: args,
                    off: (i * 8) as u32,
                    src: d,
                });
            }
            self.push_raw(Instr::StoreField {
                base: obj,
                off: OFF_TYPE_ARGS,
                src: args,
            });
            let count = self.const_i64(n as i64);
            self.push_raw(Instr::StoreField {
                base: obj,
                off: OFF_TYPE_ARG_COUNT,
                src: count,
            });
        } else {
            self.push_raw(Instr::StoreField { base: obj, off: 56, src: zero });
            self.push_raw(Instr::StoreField { base: obj, off: 64, src: zero });
        }
        Ok(obj)
    }

    /// Make a name string permanent before a descriptor stores it.
    fn immortal_string(&mut self, s: VReg) {
        let ir = self.const_i64(STRING_IMMORTAL_REF);
        self.push_raw(Instr::StoreField { base: s, off: 8, src: ir });
    }

    /// Canonical descriptor for `element[]`. Arrays are composed at
    /// run time because the element may be a generic parameter; a
    /// small linked list rooted in a static slot keeps
    /// `typeof(T[]) == typeof(T[])` true across calls. The list is
    /// single-threaded, like the rest of program initialization.
    fn array_token(&mut self, element: VReg) -> VReg {
        let cache_slot = self.typeinfo.borrow().cache_slot();
        let cache = self.fresh();
        self.push_raw(Instr::LoadStaticAddr { dst: cache, idx: cache_slot });
        let head = self.fresh();
        self.push_raw(Instr::LoadField { dst: head, base: cache, off: 0 });
        let cur = self.fresh();
        self.mov(cur, head);
        let res = self.fresh();
        let zero = self.const_i64(0);
        let check_b = self.make_block();
        let test_b = self.make_block();
        let next_b = self.make_block();
        let found_b = self.make_block();
        let build_b = self.make_block();
        let join_b = self.make_block();
        self.goto(check_b);
        self.switch_to(check_b);
        let empty = self.emit(Instr::Cmp {
            dst: VReg(0),
            op: CmpOp::Eq,
            a: cur,
            b: zero,
        });
        self.terminate(Terminator::Branch {
            cond: Some(empty),
            if_true: build_b,
            if_false: test_b,
        });
        self.switch_to(test_b);
        let cur_elem = self.fresh();
        self.push_raw(Instr::LoadField { dst: cur_elem, base: cur, off: 56 });
        let same = self.emit(Instr::Cmp {
            dst: VReg(0),
            op: CmpOp::Eq,
            a: cur_elem,
            b: element,
        });
        self.terminate(Terminator::Branch {
            cond: Some(same),
            if_true: found_b,
            if_false: next_b,
        });
        self.switch_to(next_b);
        let nxt = self.fresh();
        self.push_raw(Instr::LoadField { dst: nxt, base: cur, off: 64 });
        self.mov(cur, nxt);
        self.goto(check_b);
        self.switch_to(found_b);
        self.mov(res, cur);
        self.goto(join_b);
        self.switch_to(build_b);
        let obj = self.fresh();
        self.push_raw(Instr::Alloc { dst: obj, bytes: TYPE_BYTES });
        let tag = self.const_i64(TYPE_TAG);
        self.push_raw(Instr::StoreField { base: obj, off: 0, src: tag });
        let refc = self.const_i64(TYPE_REF);
        self.push_raw(Instr::StoreField { base: obj, off: 8, src: refc });
        let elem_name = self.fresh();
        self.push_raw(Instr::LoadField { dst: elem_name, base: element, off: OFF_NAME });
        let elem_full = self.fresh();
        self.push_raw(Instr::LoadField { dst: elem_full, base: element, off: OFF_FULL_NAME });
        let elem_ns = self.fresh();
        self.push_raw(Instr::LoadField { dst: elem_ns, base: element, off: OFF_NAMESPACE });
        let bracket = self.utf16_literal("[]");
        let bracket = self.build_utf16_string(bracket.0, bracket.1);
        let name = self.concat_strings(elem_name, bracket);
        self.immortal_string(name);
        let full = self.concat_strings(elem_full, bracket);
        self.immortal_string(full);
        self.push_raw(Instr::StoreField { base: obj, off: OFF_NAME, src: name });
        self.push_raw(Instr::StoreField { base: obj, off: OFF_FULL_NAME, src: full });
        self.push_raw(Instr::StoreField { base: obj, off: OFF_NAMESPACE, src: elem_ns });
        let flags = self.const_i64((F_ARRAY | F_SZARRAY | F_HAS_ELEMENT) as i64);
        self.push_raw(Instr::StoreField { base: obj, off: OFF_FLAGS, src: flags });
        let id = self.const_i64(0);
        self.push_raw(Instr::StoreField { base: obj, off: 48, src: id });
        self.push_raw(Instr::StoreField { base: obj, off: 56, src: element });
        self.push_raw(Instr::StoreField { base: obj, off: 64, src: head });
        let z = self.const_i64(0);
        self.push_raw(Instr::StoreField { base: obj, off: 72, src: z });
        self.push_raw(Instr::StoreField { base: cache, off: 0, src: obj });
        self.mov(res, obj);
        self.goto(join_b);
        self.switch_to(join_b);
        res
    }

    /// Split a generic spelling into its base and its arguments.
    fn split_generic_spelling(&self, spelling: &str) -> Option<(String, Vec<String>)> {
        let s = spelling.trim();
        let open = s.find('<')?;
        if !s.ends_with('>') {
            return None;
        }
        let args = explicit_type_args(s);
        if args.is_empty() {
            return None;
        }
        Some((s[..open].trim().to_owned(), args))
    }

    /// The canonical descriptor for a generic instantiation, composed
    /// at run time from the definition's descriptor and the argument
    /// descriptors. The arguments can be type parameters, which are
    /// only known when the method runs, so the composition cannot be
    /// interned at compile time.
    ///
    /// A linked list rooted in a static slot keeps
    /// `typeof(Task<T>) == typeof(Task<T>)` true across calls, the
    /// same way the composed array descriptors do.
    fn generic_token(&mut self, definition: VReg, args: &[VReg]) -> VReg {
        let n = args.len();
        let cache_slot = self.typeinfo.borrow().generic_cache_slot();
        let cache = self.fresh();
        self.push_raw(Instr::LoadStaticAddr { dst: cache, idx: cache_slot });
        let head = self.fresh();
        self.push_raw(Instr::LoadField { dst: head, base: cache, off: 0 });
        let cur = self.fresh();
        self.mov(cur, head);
        let res = self.fresh();
        let zero = self.const_i64(0);
        let check_b = self.make_block();
        let test_b = self.make_block();
        let next_b = self.make_block();
        let found_b = self.make_block();
        let build_b = self.make_block();
        let join_b = self.make_block();
        self.goto(check_b);
        self.switch_to(check_b);
        let empty = self.emit(Instr::Cmp {
            dst: VReg(0),
            op: CmpOp::Eq,
            a: cur,
            b: zero,
        });
        self.terminate(Terminator::Branch {
            cond: Some(empty),
            if_true: build_b,
            if_false: test_b,
        });
        // The definition, the argument count, and every argument must
        // match. The offsets are constants, so the comparison is
        // unrolled - the IR has no computed-offset load.
        self.switch_to(test_b);
        let cur_def = self.fresh();
        self.push_raw(Instr::LoadField { dst: cur_def, base: cur, off: 48 });
        let mut same = self.emit(Instr::Cmp {
            dst: VReg(0),
            op: CmpOp::Eq,
            a: cur_def,
            b: definition,
        });
        let cur_count = self.fresh();
        self.push_raw(Instr::LoadField {
            dst: cur_count,
            base: cur,
            off: 72,
        });
        let want_count = self.const_i64(n as i64);
        let count_ok = self.emit(Instr::Cmp {
            dst: VReg(0),
            op: CmpOp::Eq,
            a: cur_count,
            b: want_count,
        });
        same = self.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::BitAnd,
            a: same,
            b: count_ok,
        });
        let cur_args = self.fresh();
        self.push_raw(Instr::LoadField {
            dst: cur_args,
            base: cur,
            off: 56,
        });
        for (i, arg) in args.iter().enumerate() {
            let have = self.fresh();
            self.push_raw(Instr::LoadField {
                dst: have,
                base: cur_args,
                off: (i * 8) as u32,
            });
            let one = self.emit(Instr::Cmp {
                dst: VReg(0),
                op: CmpOp::Eq,
                a: have,
                b: *arg,
            });
            same = self.emit(Instr::Bin {
                dst: VReg(0),
                op: IrBinOp::BitAnd,
                a: same,
                b: one,
            });
        }
        self.terminate(Terminator::Branch {
            cond: Some(same),
            if_true: found_b,
            if_false: next_b,
        });
        self.switch_to(next_b);
        let nxt = self.fresh();
        self.push_raw(Instr::LoadField { dst: nxt, base: cur, off: 64 });
        self.mov(cur, nxt);
        self.goto(check_b);
        self.switch_to(found_b);
        self.mov(res, cur);
        self.goto(join_b);
        self.switch_to(build_b);
        let obj = self.build_composed_generic(definition, args);
        self.push_raw(Instr::StoreField { base: cache, off: 0, src: obj });
        self.mov(res, obj);
        self.goto(join_b);
        self.switch_to(join_b);
        res
    }

    /// Build the descriptor of a composed generic instantiation: the
    /// definition's name, the full name with the arguments, the
    /// definition, the argument descriptors and their count.
    fn build_composed_generic(&mut self, definition: VReg, args: &[VReg]) -> VReg {
        let obj = self.fresh();
        self.push_raw(Instr::Alloc { dst: obj, bytes: TYPE_BYTES });
        let tag = self.const_i64(TYPE_TAG);
        self.push_raw(Instr::StoreField { base: obj, off: 0, src: tag });
        let refc = self.const_i64(TYPE_REF);
        self.push_raw(Instr::StoreField { base: obj, off: 8, src: refc });
        let def_name = self.fresh();
        self.push_raw(Instr::LoadField {
            dst: def_name,
            base: definition,
            off: OFF_NAME,
        });
        self.immortal_string(def_name);
        self.push_raw(Instr::StoreField {
            base: obj,
            off: OFF_NAME,
            src: def_name,
        });
        let def_full = self.fresh();
        self.push_raw(Instr::LoadField {
            dst: def_full,
            base: definition,
            off: OFF_FULL_NAME,
        });
        let def_ns = self.fresh();
        self.push_raw(Instr::LoadField {
            dst: def_ns,
            base: definition,
            off: OFF_NAMESPACE,
        });
        // The full name: the definition's, then each argument's.
        let mut full = def_full;
        let open = self.utf16_literal("<");
        let open = self.build_utf16_string(open.0, open.1);
        full = self.concat_strings(full, open);
        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                let comma = self.utf16_literal(",");
                let comma = self.build_utf16_string(comma.0, comma.1);
                full = self.concat_strings(full, comma);
            }
            let arg_full = self.fresh();
            self.push_raw(Instr::LoadField {
                dst: arg_full,
                base: *arg,
                off: OFF_FULL_NAME,
            });
            full = self.concat_strings(full, arg_full);
        }
        let close = self.utf16_literal(">");
        let close = self.build_utf16_string(close.0, close.1);
        full = self.concat_strings(full, close);
        self.immortal_string(full);
        self.push_raw(Instr::StoreField {
            base: obj,
            off: OFF_FULL_NAME,
            src: full,
        });
        self.push_raw(Instr::StoreField {
            base: obj,
            off: OFF_NAMESPACE,
            src: def_ns,
        });
        let flags = self.const_i64(F_GENERIC_TYPE as i64);
        self.push_raw(Instr::StoreField {
            base: obj,
            off: OFF_FLAGS,
            src: flags,
        });
        // The definition, the arguments, their count, and the list's
        // next pointer.
        self.push_raw(Instr::StoreField {
            base: obj,
            off: 48,
            src: definition,
        });
        let arr = self.fresh();
        self.push_raw(Instr::Alloc {
            dst: arr,
            bytes: (args.len() as u64) * 8,
        });
        for (i, arg) in args.iter().enumerate() {
            self.push_raw(Instr::StoreField {
                base: arr,
                off: (i * 8) as u32,
                src: *arg,
            });
        }
        self.push_raw(Instr::StoreField {
            base: obj,
            off: 56,
            src: arr,
        });
        let count = self.const_i64(args.len() as i64);
        self.push_raw(Instr::StoreField {
            base: obj,
            off: 72,
            src: count,
        });
        let next = self.const_i64(0);
        self.push_raw(Instr::StoreField {
            base: obj,
            off: 64,
            src: next,
        });
        obj
    }

    /// The canonical descriptor pointer for `spelling`, built once and
    /// cached in a static slot. A bare type parameter reads its hidden
    /// handle; a constructed type that still contains one is reported
    /// until runtime composite descriptors exist.
    pub(crate) fn type_token(
        &mut self,
        spelling: &str,
        ctx: &Ctx<'_>,
    ) -> Result<VReg, LowerError> {
        let mut s = spelling
            .trim()
            .trim_start_matches("global::")
            .trim()
            .to_owned();
        if let Some(sub) = ctx.type_subst
            && let Some(concrete) = subst_type_var(&s, sub)
        {
            s = concrete;
        }
        // Arrays are composed from their element descriptor at run
        // time: the element can be a generic parameter, and the cache
        // keeps identity stable.
        if let Some(elem) = s.strip_suffix("[]") {
            let e = self.type_token(elem, ctx)?;
            return Ok(self.array_token(e));
        }
        if direct_type_var(&s, ctx) {
            if let Some(&v) = self.type_handles.get(&s) {
                return Ok(v);
            }
            // A CLASS type parameter: the runtime type argument, read
            // from the instance's runtime type. The instance knows its
            // instantiation (the construction stored the descriptor),
            // and the descriptor carries one descriptor per argument.
            if let Some(v) = self.class_type_argument(&s, ctx)? {
                return Ok(v);
            }
            // A CLASS type parameter: the runtime type argument, read
            // from the instance's runtime type. The instance knows its
            // instantiation (the construction stored the descriptor),
            // and the descriptor carries one descriptor per argument.
            if let Some(v) = self.class_type_argument(&s, ctx)? {
                return Ok(v);
            }
            return Err(LowerError::UnsupportedConstruct(format!(
                "`typeof({spelling})` needs a runtime type handle for `{s}`, but this method has no hidden type argument for it"
            )));
        }
        // A DELEGATE type is not a class entry, so it has its own
        // descriptor: the table looks delegates up by name, and that
        // name is the identity.
        // A CONSTRUCTED type whose arguments mention a type parameter
        // is composed at run time: the definition's descriptor plus the
        // argument descriptors (a parameter's argument is the handle
        // the method carries).
        // A nullable REFERENCE type is the same runtime type: the
        // annotation is erased, so `Task<T>?` names `Task<T>`.
        let s = s.trim_end_matches('?').to_owned();
        if let Some((base, args)) = self.split_generic_spelling(&s)
            && args
                .iter()
                .any(|a| direct_type_var(a.trim_end_matches('?'), ctx))
        {
            let def = self.type_token(&base, ctx)?;
            let mut descs: Vec<VReg> = Vec::with_capacity(args.len());
            for a in &args {
                // A nullable annotation is erased at run time, so
                // `TResult?` names the runtime type of `TResult`.
                let a = a.trim_end_matches('?');
                if direct_type_var(a, ctx) {
                    if let Some(&v) = self.type_handles.get(a) {
                        descs.push(v);
                        continue;
                    }
                    let Some(v) = self.class_type_argument(a, ctx)? else {
                        return Err(LowerError::UnsupportedConstruct(format!(
                            "`typeof({spelling})` needs a runtime type handle for `{a}`"
                        )));
                    };
                    descs.push(v);
                } else {
                    descs.push(self.type_token(a, ctx)?);
                }
            }
            return Ok(self.generic_token(def, &descs));
        }
        let meta = if ctx.table.is_delegate(&s) {
            delegate_meta(&s, ctx)
        } else {
            type_meta(&s, ctx)
        };
        if meta.has_type_var {
            return Err(LowerError::UnsupportedConstruct(format!(
                "`typeof({spelling})` constructs a type from a generic parameter; runtime composite type descriptors are not implemented yet"
            )));
        }
        let (slot, req) = {
            let mut reg = self.typeinfo.borrow_mut();
            reg.slot_for(&meta.key, &meta, self.strings)
        };
        let addr = self.fresh();
        self.push_raw(Instr::LoadStaticAddr { dst: addr, idx: slot });
        let cached = self.fresh();
        self.push_raw(Instr::LoadField { dst: cached, base: addr, off: 0 });
        let zero = self.const_i64(0);
        let empty = self.emit(Instr::Cmp {
            dst: VReg(0),
            op: CmpOp::Eq,
            a: cached,
            b: zero,
        });
        let res = self.fresh();
        self.mov(res, cached);
        let build_b = self.make_block();
        let join_b = self.make_block();
        self.terminate(Terminator::Branch {
            cond: Some(empty),
            if_true: build_b,
            if_false: join_b,
        });
        self.switch_to(build_b);
        let obj = self.build_type_descriptor(&req, ctx)?;
        self.push_raw(Instr::StoreField { base: addr, off: 0, src: obj });
        self.mov(res, obj);
        self.goto(join_b);
        self.switch_to(join_b);
        Ok(res)
    }

    /// The runtime type argument for a CLASS type parameter (`T` in
    /// `C<T>`), read from the instance's runtime type.
    ///
    /// The instance carries the descriptor of its instantiation, and
    /// that descriptor holds one descriptor per type argument (the
    /// base chain's arguments first, .NET order). `ClassInfo::
    /// type_arg_index` says where this class's own parameters sit in
    /// that list.
    fn class_type_argument(
        &mut self,
        name: &str,
        ctx: &Ctx<'_>,
    ) -> Result<Option<VReg>, LowerError> {
        let Some(info) = ctx.current_info else {
            return Ok(None);
        };
        let Some(pos) = info.generic_params.iter().position(|p| p == name) else {
            return Ok(None);
        };
        if info.is_struct && info.generic_params.is_empty() {
            return Ok(None);
        }
        let Some(&index) = info.type_arg_index.get(pos) else {
            return Ok(None);
        };
        if index != 0 && info.type_arg_index.first().copied().unwrap_or(0) != 0 {
            // A generic base contributes arguments of its own, and the
            // instance's runtime type carries only this class's. Say so
            // rather than read the wrong argument.
            return Err(LowerError::UnsupportedConstruct(format!(
                "`typeof({name})` in a class with a generic base is not supported yet"
            )));
        }
        // The runtime type: the instance's header slot.
        let runtime = self.fresh();
        self.push_raw(Instr::LoadField {
            dst: runtime,
            base: VReg(0),
            off: (RUNTIME_TYPE_SLOT * 8) as u32,
        });
        // An instance constructed with a type VARIABLE has no
        // concrete runtime type, so there is no argument to read.
        let zero = self.const_i64(0);
        let known = self.emit(Instr::Cmp {
            dst: VReg(0),
            op: CmpOp::Ne,
            a: runtime,
            b: zero,
        });
        let has_type = self.make_block();
        let no_type = self.make_block();
        let join_b = self.make_block();
        self.terminate(Terminator::Branch {
            cond: Some(known),
            if_true: has_type,
            if_false: no_type,
        });
        self.switch_to(no_type);
        self.throw_invalid_cast(ctx)?;
        self.switch_to(has_type);
        // Its arguments.
        let args = self.fresh();
        self.push_raw(Instr::LoadField {
            dst: args,
            base: runtime,
            off: OFF_TYPE_ARGS,
        });
        // The one this parameter names.
        let arg = self.fresh();
        self.push_raw(Instr::LoadField {
            dst: arg,
            base: args,
            off: (index * 8) as u32,
        });
        Ok(Some(arg))
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

    /// Build a UTF-16 user string from interned `(off, char_len)` (see
    /// `utf16_literal`). Layout: [0..8)=length in code units,
    /// [8..16)=tagged refcount ([`STRING_REF`]), [16..16+2*char_len)=
    /// UTF-16 data. The tag is what `is string` reads.
    pub(crate) fn build_utf16_string(&mut self, off: u32, char_len: u32) -> VReg {
        let total = 16u64 + char_len as u64 * 2;
        let obj = self.fresh();
        self.push_raw(Instr::Alloc { dst: obj, bytes: total });
        let lenv = self.const_i64(char_len as i64);
        self.push_raw(Instr::StoreField { base: obj, off: 0, src: lenv });
        let sref = self.const_i64(STRING_REF);
        self.push_raw(Instr::StoreField { base: obj, off: 8, src: sref });
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
    /// Encode a runtime string (UTF-16) as UTF-8 in a temporary object
    /// whose length is its BYTE count, so `WriteString` emits valid text.
    ///
    /// IR-only: `LoadByte`/`StoreField` accept a computed base, so the loop
    /// addresses characters and bytes dynamically. Bytes are written left to
    /// right; an 8-byte store zeroes the bytes after it, which later stores
    /// overwrite. A surrogate pair is combined into one four-byte sequence;
    /// an unpaired surrogate becomes U+FFFD.
    pub(crate) fn utf8_string_of(&mut self, s: VReg) -> VReg {
        let zero = self.const_i64(0);
        let one = self.const_i64(1);
        let len = self.fresh();
        self.push_raw(Instr::LoadField { dst: len, base: s, off: 0 });
        let sixteen = self.const_i64(16);
        let data = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: s, b: sixteen });
        let three = self.const_i64(3);
        let cap0 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Mul, a: len, b: three });
        let thirtytwo = self.const_i64(32);
        let cap = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: cap0, b: thirtytwo });
        let out = self.fresh();
        self.push_raw(Instr::AllocDyn { dst: out, bytes: cap });
        let p = self.fresh();
        let p0 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: out, b: sixteen });
        self.mov(p, p0);
        let i = self.fresh();
        self.mov(i, zero);
        let loop_b = self.make_block();
        let body_b = self.make_block();
        let done_b = self.make_block();
        self.goto(loop_b);
        self.switch_to(loop_b);
        let i_ge = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Ge, a: i, b: len });
        self.terminate(Terminator::Branch { cond: Some(i_ge), if_true: done_b, if_false: body_b });
        self.switch_to(body_b);
        let two = self.const_i64(2);
        let two_i = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Mul, a: i, b: two });
        let addr = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: data, b: two_i });
        let lo = self.fresh();
        self.push_raw(Instr::LoadByte { dst: lo, base: addr, off: 0 });
        let hi = self.fresh();
        self.push_raw(Instr::LoadByte { dst: hi, base: addr, off: 1 });
        let eight = self.const_i64(8);
        let hi_s = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Shl, a: hi, b: eight });
        let u = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitOr, a: lo, b: hi_s });
        // Shared constants are defined before the first branch that uses
        // them: a vreg defined inside one arm stays undefined on a sibling
        // path and the backend then reads an uninitialized slot.
        let c80 = self.const_i64(0x80);
        let c800 = self.const_i64(0x800);
        let six = self.const_i64(6);
        let mask = self.const_i64(0x3F);
        let c80b = self.const_i64(0x80);
        let twelve = self.const_i64(12);
        let c0 = self.const_i64(0xC0);
        let e0 = self.const_i64(0xE0);
        let eighteen = self.const_i64(18);
        let f0 = self.const_i64(0xF0);
        let m3ff = self.const_i64(0x3FF);
        let base10000 = self.const_i64(0x10000);
        let ten = self.const_i64(10);
        let four = self.const_i64(4);
        let d800 = self.const_i64(0xD800);
        let dbff = self.const_i64(0xDBFF);
        let dc00 = self.const_i64(0xDC00);
        let dfff = self.const_i64(0xDFFF);
        let repl = self.const_i64(0xFFFD);
        // A high surrogate followed by a low surrogate is ONE code point:
        // combine the pair and emit four UTF-8 bytes, consuming both
        // units. Any other surrogate is unpaired and becomes U+FFFD.
        let hi_ge = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Ge, a: u, b: d800 });
        let hi_le = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Le, a: u, b: dbff });
        let is_high = self.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::BitAnd,
            a: hi_ge,
            b: hi_le,
        });
        let lo_ge = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Ge, a: u, b: dc00 });
        let lo_le = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Le, a: u, b: dfff });
        let is_low = self.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::BitAnd,
            a: lo_ge,
            b: lo_le,
        });
        let is_sur = self.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::BitOr,
            a: is_high,
            b: is_low,
        });
        let uu = self.fresh();
        let bmp_b = self.make_block();
        let sur_b = self.make_block();
        let lone_b = self.make_block();
        let high_b = self.make_block();
        let load_b = self.make_block();
        let pair_b = self.make_block();
        let lone2_b = self.make_block();
        let four_b = self.make_block();
        let next_b = self.make_block();
        let next2_b = self.make_block();
        let sel_join = self.make_block();
        self.terminate(Terminator::Branch {
            cond: Some(is_sur),
            if_true: sur_b,
            if_false: bmp_b,
        });
        self.switch_to(bmp_b);
        self.mov(uu, u);
        self.goto(sel_join);
        self.switch_to(sur_b);
        self.terminate(Terminator::Branch {
            cond: Some(is_low),
            if_true: lone_b,
            if_false: high_b,
        });
        self.switch_to(lone_b);
        self.mov(uu, repl);
        self.goto(sel_join);
        self.switch_to(high_b);
        let i1 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: i, b: one });
        let has2 = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Lt, a: i1, b: len });
        self.terminate(Terminator::Branch {
            cond: Some(has2),
            if_true: load_b,
            if_false: lone2_b,
        });
        self.switch_to(load_b);
        let lo2 = self.fresh();
        self.push_raw(Instr::LoadByte { dst: lo2, base: addr, off: 2 });
        let hi2 = self.fresh();
        self.push_raw(Instr::LoadByte { dst: hi2, base: addr, off: 3 });
        let hi2s = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Shl, a: hi2, b: eight });
        let u2 = self.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::BitOr,
            a: lo2,
            b: hi2s,
        });
        let lo2_ge = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Ge, a: u2, b: dc00 });
        let lo2_le = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Le, a: u2, b: dfff });
        let is_low2 = self.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::BitAnd,
            a: lo2_ge,
            b: lo2_le,
        });
        self.terminate(Terminator::Branch {
            cond: Some(is_low2),
            if_true: pair_b,
            if_false: lone2_b,
        });
        self.switch_to(lone2_b);
        self.mov(uu, repl);
        self.goto(sel_join);
        self.switch_to(pair_b);
        let uo = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitAnd, a: u, b: m3ff });
        let us = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Shl, a: uo, b: ten });
        let u2o = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitAnd, a: u2, b: m3ff });
        let comb = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: us, b: u2o });
        let cp = self.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::Add,
            a: comb,
            b: base10000,
        });
        self.goto(four_b);
        self.switch_to(four_b);
        let h18 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Shr, a: cp, b: eighteen });
        let f0b = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitOr, a: h18, b: f0 });
        let h12 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Shr, a: cp, b: twelve });
        let m12 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitAnd, a: h12, b: mask });
        let f1 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitOr, a: m12, b: c80b });
        let h6f = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Shr, a: cp, b: six });
        let m6 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitAnd, a: h6f, b: mask });
        let f2 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitOr, a: m6, b: c80b });
        let m0 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitAnd, a: cp, b: mask });
        let f3 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitOr, a: m0, b: c80b });
        self.push_raw(Instr::StoreField { base: p, off: 0, src: f0b });
        self.push_raw(Instr::StoreField { base: p, off: 1, src: f1 });
        self.push_raw(Instr::StoreField { base: p, off: 2, src: f2 });
        self.push_raw(Instr::StoreField { base: p, off: 3, src: f3 });
        let p4 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: p, b: four });
        self.mov(p, p4);
        self.goto(next2_b);
        self.switch_to(sel_join);
        let lt80 = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Lt, a: uu, b: c80 });
        let one_b = self.make_block();
        let two_chk = self.make_block();
        self.terminate(Terminator::Branch { cond: Some(lt80), if_true: one_b, if_false: two_chk });
        self.switch_to(two_chk);
        let lt800 = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Lt, a: uu, b: c800 });
        let two_b = self.make_block();
        let three_b = self.make_block();
        self.terminate(Terminator::Branch { cond: Some(lt800), if_true: two_b, if_false: three_b });
        let next_b = self.make_block();
        self.switch_to(one_b);
        self.push_raw(Instr::StoreField { base: p, off: 0, src: uu });
        let p1 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: p, b: one });
        self.mov(p, p1);
        self.goto(next_b);
        self.switch_to(two_b);
        let h6 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Shr, a: uu, b: six });
        let c0 = self.const_i64(0xC0);
        let b0 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitOr, a: h6, b: c0 });
        let low = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitAnd, a: uu, b: mask });
        let b1 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitOr, a: low, b: c80b });
        self.push_raw(Instr::StoreField { base: p, off: 0, src: b0 });
        self.push_raw(Instr::StoreField { base: p, off: 1, src: b1 });
        let p2 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: p, b: two });
        self.mov(p, p2);
        self.goto(next_b);
        self.switch_to(three_b);
        let h12 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Shr, a: uu, b: twelve });
        let t0 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitOr, a: h12, b: e0 });
        let h6b = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Shr, a: uu, b: six });
        let t1 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitAnd, a: h6b, b: mask });
        let t2 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitOr, a: t1, b: c80b });
        let low2 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitAnd, a: uu, b: mask });
        let t3 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitOr, a: low2, b: c80b });
        self.push_raw(Instr::StoreField { base: p, off: 0, src: t0 });
        self.push_raw(Instr::StoreField { base: p, off: 1, src: t2 });
        self.push_raw(Instr::StoreField { base: p, off: 2, src: t3 });
        let p3 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: p, b: three });
        self.mov(p, p3);
        self.goto(next_b);
        self.switch_to(next_b);
        let i1 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: i, b: one });
        self.mov(i, i1);
        self.goto(loop_b);
        self.switch_to(next2_b);
        let i2 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: i, b: two });
        self.mov(i, i2);
        self.goto(loop_b);
        self.switch_to(done_b);
        let pos = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Sub, a: p, b: p0 });
        self.push_raw(Instr::StoreField { base: out, off: 0, src: pos });
        let sref = self.const_i64(STRING_REF);
        self.push_raw(Instr::StoreField { base: out, off: 8, src: sref });
        out
    }

    /// C# value-to-string for the primitive types (`ToString`,
    /// interpolation, `string + value`). `None` means the type has no
    /// conversion yet (float/double/decimal), so the caller reports it.
    pub(crate) fn value_to_string(&mut self, v: VReg, ty: &str) -> Option<VReg> {
        match crate::native::ir::lower::select::base_type_name(ty).as_str() {
            "int" | "Int32" | "long" | "Int64" | "short" | "Int16" | "sbyte" | "SByte"
            | "nint" | "IntPtr" => Some(self.int_to_string(v, true)),
            "uint" | "UInt32" | "ulong" | "UInt64" | "ushort" | "UInt16" | "byte" | "Byte"
            | "nuint" | "UIntPtr" => Some(self.int_to_string(v, false)),
            "bool" | "Boolean" => {
                let t = self.utf16_literal("True");
                let ts = self.build_utf16_string(t.0, t.1);
                let f = self.utf16_literal("False");
                let fs = self.build_utf16_string(f.0, f.1);
                let zero = self.const_i64(0);
                let is_true = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Ne, a: v, b: zero });
                let home = self.fresh();
                let true_b = self.make_block();
                let false_b = self.make_block();
                let join_b = self.make_block();
                self.terminate(Terminator::Branch {
                    cond: Some(is_true),
                    if_true: true_b,
                    if_false: false_b,
                });
                self.switch_to(true_b);
                self.mov(home, ts);
                self.goto(join_b);
                self.switch_to(false_b);
                self.mov(home, fs);
                self.goto(join_b);
                self.switch_to(join_b);
                Some(home)
            }
            "char" | "Char" => {
                let one = self.const_i64(1);
                let s = self.fresh();
                self.push_raw(Instr::Alloc { dst: s, bytes: 24 });
                self.push_raw(Instr::StoreField { base: s, off: 0, src: one });
                let sref = self.const_i64(STRING_REF);
                self.push_raw(Instr::StoreField { base: s, off: 8, src: sref });
                self.push_raw(Instr::StoreField { base: s, off: 16, src: v });
                Some(s)
            }
            _ => None,
        }
    }

    /// Decimal string for an integer value (`ToString`/interpolation).
    ///
    /// The IR only stores at immediate offsets, so the digits are written
    /// by an UNROLLED 19-iteration sequence into a scratch buffer (right
    /// aligned), then copied out with `MemCpy` (whose source pointer may be
    /// computed). `signed` handles the sign and `i64::MIN`; `uint` values
    /// arrive zero-extended. This is a few dozen instructions per call
    /// site — a shared runtime function is a later optimization.
    pub(crate) fn int_to_string(&mut self, v: VReg, signed: bool) -> VReg {
        self.int_to_string_min(v, signed, 1)
    }

    /// Decimal `int_to_string` with at least `min_digits` digits (sign not
    /// counted). `min_digits` is bounded by the 32-unit extraction window.
    pub(crate) fn int_to_string_min(&mut self, v: VReg, signed: bool, min_digits: u32) -> VReg {
        debug_assert!((1..=31).contains(&min_digits));
        // i64::MIN cannot be negated; emit its literal directly.
        let min_lit = {
            // i64::MIN cannot be negated. Precision counts digits after the
            // sign: pad the literal for D20..D31 at build time.
            let pad = min_digits.saturating_sub(19) as usize;
            let text = format!("-{}{}", "0".repeat(pad), "9223372036854775808");
            let t = self.utf16_literal(&text);
            self.build_utf16_string(t.0, t.1)
        };
        let zero = self.const_i64(0);
        let dv = self.fresh();
        let neg = self.fresh();
        let out = self.fresh();
        let end_b = self.make_block();
        let start_b = self.make_block();
        let neg_b = self.make_block();
        let pos_b = self.make_block();
        let join_b = self.make_block();
        if signed {
            let min = self.const_i64(i64::MIN);
            let is_min = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Eq, a: v, b: min });
            let min_b = self.make_block();
            self.terminate(Terminator::Branch { cond: Some(is_min), if_true: min_b, if_false: start_b });
            self.switch_to(min_b);
            // i64::MIN cannot be negated: answer its literal directly.
            self.mov(out, min_lit);
            self.goto(end_b);
            self.switch_to(start_b);
            let is_neg = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Lt, a: v, b: zero });
            self.terminate(Terminator::Branch { cond: Some(is_neg), if_true: neg_b, if_false: pos_b });
            self.switch_to(neg_b);
            let neg_one = self.const_i64(1);
            self.mov(neg, neg_one);
            let a = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Sub, a: zero, b: v });
            self.mov(dv, a);
            self.goto(join_b);
            self.switch_to(pos_b);
            self.mov(neg, zero);
            self.mov(dv, v);
            self.goto(join_b);
            self.switch_to(join_b);
        } else {
            self.mov(neg, zero);
            self.mov(dv, v);
        }
        // k = decimal digit count of dv.
        let k = self.fresh();
        let one = self.const_i64(1);
        self.mov(k, one);
        let t = self.fresh();
        self.mov(t, dv);
        let count_b = self.make_block();
        let count_done = self.make_block();
        let ten = self.const_i64(10);
        let (div_op, rem_op) = if signed {
            (IrBinOp::Div, IrBinOp::Rem)
        } else {
            (IrBinOp::UDiv, IrBinOp::URem)
        };
        self.goto(count_b);
        self.switch_to(count_b);
        let t10 = self.emit(Instr::Bin { dst: VReg(0), op: div_op, a: t, b: ten });
        let at_zero = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Eq, a: t10, b: zero });
        let more_b = self.make_block();
        self.terminate(Terminator::Branch { cond: Some(at_zero), if_true: count_done, if_false: more_b });
        self.switch_to(more_b);
        let k1 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: k, b: one });
        self.mov(k, k1);
        self.mov(t, t10);
        self.goto(count_b);
        // Digits are right aligned in a 32-unit window: 12 zero units then
        // 20 digit units (u64::MAX is 20 digits). Raising k only starts the
        // copy earlier, over the prefilled '0' units.
        self.switch_to(count_done);
        // Raise the digit count to the requested minimum.
        let min_lit = self.const_i64(min_digits as i64);
        let below = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Lt, a: k, b: min_lit });
        let diff = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Sub, a: min_lit, b: k });
        let bump = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Mul, a: below, b: diff });
        let k_raised = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: k, b: bump });
        self.mov(k, k_raised);
        let buf = self.fresh();
        self.push_raw(Instr::Alloc { dst: buf, bytes: 72 });
        // Twelve leading '0' units (three 8-byte stores, four units each).
        let zero_quad = self.const_i64(0x0030_0030_0030_0030);
        for off in [0u32, 8, 16] {
            self.push_raw(Instr::StoreField { base: buf, off, src: zero_quad });
        }
        let dvv = self.fresh();
        self.mov(dvv, dv);
        // Extract least-significant first into registers. Writing them
        // immediately would not work: `StoreField` writes 8 bytes, so a
        // store at a 2-byte offset clobbers the digit written before it.
        let mut digits: Vec<VReg> = Vec::with_capacity(20);
        for _ in 0..20u32 {
            let d = self.emit(Instr::Bin { dst: VReg(0), op: rem_op, a: dvv, b: ten });
            digits.push(d);
            let q = self.emit(Instr::Bin { dst: VReg(0), op: div_op, a: dvv, b: ten });
            self.mov(dvv, q);
        }
        // Write left to right: each 8-byte store only touches bytes to the
        // RIGHT of its char, which later stores overwrite cleanly.
        let forty_eight = self.const_i64(48);
        for p in 0..20u32 {
            let d = digits[(19 - p) as usize];
            let ch = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: d, b: forty_eight });
            self.push_raw(Instr::StoreField { base: buf, off: 24 + 2 * p, src: ch });
        }
        // length = k + sign
        let len = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: k, b: neg });
        let two = self.const_i64(2);
        let bytes = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Mul, a: len, b: two });
        let sixteen = self.const_i64(16);
        let total = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: bytes, b: sixteen });
        let s = self.fresh();
        self.push_raw(Instr::AllocDyn { dst: s, bytes: total });
        self.push_raw(Instr::StoreField { base: s, off: 0, src: len });
        let sref = self.const_i64(STRING_REF);
        self.push_raw(Instr::StoreField { base: s, off: 8, src: sref });
        let data = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: s, b: sixteen });
        // Optional '-' then the digits.
        let pos_b = self.make_block();
        let neg_b = self.make_block();
        let join2 = self.make_block();
        let is_neg2 = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Ne, a: neg, b: zero });
        self.terminate(Terminator::Branch { cond: Some(is_neg2), if_true: neg_b, if_false: pos_b });
        let dest = self.fresh();
        self.switch_to(neg_b);
        let minus = self.const_i64(45);
        self.push_raw(Instr::StoreField { base: data, off: 0, src: minus });
        let d2 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: data, b: two });
        self.mov(dest, d2);
        self.goto(join2);
        self.switch_to(pos_b);
        self.mov(dest, data);
        self.goto(join2);
        self.switch_to(join2);
        // src = buf + 2*(32-k); copy 2*k bytes (the window ends at unit 31).
        let end = self.const_i64(32);
        let back = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Sub, a: end, b: k });
        let back_b = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Mul, a: back, b: two });
        let src = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: buf, b: back_b });
        let nbytes = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Mul, a: k, b: two });
        self.push_raw(Instr::MemCpy { dst: dest, src, len: nbytes });
        self.mov(out, s);
        self.goto(end_b);
        self.switch_to(end_b);
        out
    }

    /// Hexadecimal rendering at the C# type's width. A negative signed
    /// value uses its two's-complement form (short -> 4 digits, int -> 8,
    /// long -> 16); `min_digits` zero-pads like `X4`.
    pub(crate) fn int_to_hex_string(
        &mut self,
        v: VReg,
        bits: u32,
        uppercase: bool,
        min_digits: u32,
    ) -> VReg {
        let zero = self.const_i64(0);
        let one = self.const_i64(1);
        let two = self.const_i64(2);
        let four = self.const_i64(4);
        let nine = self.const_i64(9);
        let fifteen = self.const_i64(15);
        let sixteen = self.const_i64(16);
        let masked = if bits >= 64 {
            v
        } else {
            let mask = self.const_i64((1i64 << bits) - 1);
            self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitAnd, a: v, b: mask })
        };
        // Significant nibbles, at least one.
        let count = self.fresh();
        self.mov(count, one);
        let t = self.fresh();
        self.mov(t, masked);
        let loop_b = self.make_block();
        let body_b = self.make_block();
        let done_b = self.make_block();
        self.goto(loop_b);
        self.switch_to(loop_b);
        let step = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::ShrShr, a: t, b: four });
        let at_zero = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Eq, a: step, b: zero });
        self.terminate(Terminator::Branch {
            cond: Some(at_zero),
            if_true: done_b,
            if_false: body_b,
        });
        self.switch_to(body_b);
        let c1 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: count, b: one });
        self.mov(count, c1);
        self.mov(t, step);
        self.goto(loop_b);
        self.switch_to(done_b);
        // Raise the width to the requested precision (zero padding).
        let min_lit = self.const_i64(min_digits as i64);
        let below = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Lt, a: count, b: min_lit });
        let diff = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Sub, a: min_lit, b: count });
        let bump = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Mul, a: below, b: diff });
        let raised = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: count, b: bump });
        self.mov(count, raised);
        let bytes = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Mul, a: count, b: two });
        let total = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: bytes, b: sixteen });
        // Digits are written with 8-byte stores at a 2-byte stride; the
        // last one overruns the logical end, so leave slack in the object.
        let slack = self.const_i64(8);
        let padded = self.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::Add,
            a: total,
            b: slack,
        });
        let s = self.fresh();
        self.push_raw(Instr::AllocDyn { dst: s, bytes: padded });
        self.push_raw(Instr::StoreField { base: s, off: 0, src: count });
        let sref = self.const_i64(STRING_REF);
        self.push_raw(Instr::StoreField { base: s, off: 8, src: sref });
        let data = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: s, b: sixteen });
        // Digits are written left to right: an 8-byte store zeroes the
        // bytes after it, and later stores overwrite them.
        let rev = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Sub, a: count, b: one });
        let pos = self.fresh();
        self.mov(pos, zero);
        let ploop = self.make_block();
        let pbody = self.make_block();
        let pdone = self.make_block();
        self.goto(ploop);
        self.switch_to(ploop);
        let more = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Lt, a: pos, b: count });
        self.terminate(Terminator::Branch {
            cond: Some(more),
            if_true: pbody,
            if_false: pdone,
        });
        self.switch_to(pbody);
        let shpos = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Sub, a: rev, b: pos });
        let shift = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Mul, a: shpos, b: four });
        let shifted = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::ShrShr, a: masked, b: shift });
        let d = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitAnd, a: shifted, b: fifteen });
        let over9 = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Gt, a: d, b: nine });
        let letter_off = self.const_i64(if uppercase { 7 } else { 39 });
        let delta = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Mul, a: over9, b: letter_off });
        let forty_eight = self.const_i64(48);
        let digit_base = self.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::Add,
            a: d,
            b: forty_eight,
        });
        let ch = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: digit_base, b: delta });
        let off = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Mul, a: pos, b: two });
        let addr = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: data, b: off });
        self.push_raw(Instr::StoreField { base: addr, off: 0, src: ch });
        let p1 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: pos, b: one });
        self.mov(pos, p1);
        self.goto(ploop);
        self.switch_to(pdone);
        s
    }

    /// Pad `s` with spaces to `width` characters. `left` puts the text on
    /// the left (`{0,-w}`); otherwise it is right-aligned. `width` is
    /// compile-time, so the space run is one literal.
    pub(crate) fn pad_string(&mut self, s: VReg, width: u64, left: bool) -> VReg {
        if width == 0 {
            return s;
        }
        let lit = self.utf16_literal(&" ".repeat(width as usize));
        let spaces = self.build_utf16_string(lit.0, lit.1);
        let two = self.const_i64(2);
        let sixteen = self.const_i64(16);
        let w = self.const_i64(width as i64);
        let len = self.fresh();
        self.push_raw(Instr::LoadField { dst: len, base: s, off: 0 });
        let fits = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Ge, a: len, b: w });
        let out = self.fresh();
        let keep_b = self.make_block();
        let pad_b = self.make_block();
        let join_b = self.make_block();
        self.terminate(Terminator::Branch {
            cond: Some(fits),
            if_true: keep_b,
            if_false: pad_b,
        });
        self.switch_to(keep_b);
        self.mov(out, s);
        self.goto(join_b);
        self.switch_to(pad_b);
        let obj = self.fresh();
        self.push_raw(Instr::Alloc { dst: obj, bytes: 16 + 2 * width });
        self.push_raw(Instr::StoreField { base: obj, off: 0, src: w });
        let sref = self.const_i64(STRING_REF);
        self.push_raw(Instr::StoreField { base: obj, off: 8, src: sref });
        let data = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: obj, b: sixteen });
        let sdata = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: s, b: sixteen });
        let spdata = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: spaces, b: sixteen });
        let pad = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Sub, a: w, b: len });
        let padb = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Mul, a: pad, b: two });
        let lenb = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Mul, a: len, b: two });
        if left {
            self.push_raw(Instr::MemCpy { dst: data, src: sdata, len: lenb });
            let d2 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: data, b: lenb });
            self.push_raw(Instr::MemCpy { dst: d2, src: spdata, len: padb });
        } else {
            self.push_raw(Instr::MemCpy { dst: data, src: spdata, len: padb });
            let d2 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: data, b: padb });
            self.push_raw(Instr::MemCpy { dst: d2, src: sdata, len: lenb });
        }
        self.mov(out, obj);
        self.goto(join_b);
        self.switch_to(join_b);
        out
    }

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
        let sref = self.const_i64(STRING_REF);
        self.push_raw(Instr::StoreField { base: newobj, off: 8, src: sref });
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
        // The parser rewrites a using declaration into a Let plus a
        // try/finally; one reaching here (a shape the parser did not
        // wrap) still evaluates its initializer and binds the name.
        if let Stmt::UsingDecl {
            type_name,
            name,
            init,
            span,
        } = stmt
        {
            let as_let = Stmt::Let {
                is_const: false,
                is_byref: false,
                type_name: type_name.clone(),
                name: name.clone(),
                init: init.clone(),
                access: crate::models::Access::Default,
                span: *span,
            };
            return self.statement(&as_let, ctx);
        }
        match stmt {
            // Handled by the guard above; listed so the match is total.
            Stmt::UsingDecl { .. } => Ok(()),
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
                // A BARE array initializer (`int[] a = { 1, 2, 3 };`) is
                // parsed by the expression grammar, which has no declared
                // type: it left `elem_type` empty, so construction used
                // the 8-byte default width while indexed reads used the
                // declared element width (element 1 read the high half of
                // element 0 — zero). Fill it from the declaration here,
                // where both are available, for every declaration form.
                let patched_array;
                let init = if let Expr::NewArray {
                    elem_type,
                    size,
                    init: elems,
                    span: nspan,
                } = init
                {
                    if elem_type.is_empty() && type_name.ends_with("[]") {
                        patched_array = Expr::NewArray {
                            elem_type: type_name[..type_name.len() - 2].trim().to_owned(),
                            size: size.clone(),
                            init: elems.clone(),
                            span: *nspan,
                        };
                        &patched_array
                    } else {
                        init
                    }
                } else {
                    init
                };
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
                let mut type_name = normalize_type_ref(ctx.table, type_name.as_str());
                // `var` takes the INITIALIZER's static type: storing the
                // literal spelling "var" threw away the only information
                // that says how to read the value (`var t = (1, 2)` then
                // `t.Item1` saw a type called "var").
                if type_name == "var" {
                    // `static_type_of` answers `object` for an array
                    // literal, which would lose the tuple spelling a
                    // member read needs; derive it from the expression
                    // shape when the static answer is uninformative.
                    let inferred = match static_type_of(init, self, ctx) {
                        Ok(t) if t != "object" && !t.is_empty() => t,
                        _ => match init {
                            Expr::NewArray { elem_type, .. } if elem_type == "tuple" => {
                                "tuple".to_owned()
                            }
                            Expr::NewArray { elem_type, .. } if !elem_type.is_empty() => {
                                format!("{elem_type}[]")
                            }
                            _ => "object".to_owned(),
                        },
                    };
                    type_name = inferred;
                }
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
                // Explicit generic ARGUMENTS must match a declaration's
                // arity: `Box<int, string>` when `Box` takes one type
                // parameter is CS0305, and it used to be accepted because
                // a handle-typed local only needs "is it a class".
                if let Some(lt) = type_name.find('<') {
                    let base = type_name[..lt].trim();
                    let simple = base.rsplit('.').next().unwrap_or(base);
                    let want = crate::native::semantic::spelling_arity(&type_name);
                    // Only judge types this unit DECLARES. `ReadOnlySpan<char>`
                    // and friends are compiler-known shapes with no
                    // declaration; they must not be reported just for
                    // having no entry.
                    let declared: Vec<&crate::native::semantic::ClassInfo> = ctx
                        .table
                        .classes
                        .iter()
                        .filter(|c| c.name == simple || c.key.path == base)
                        .collect();
                    let has = declared
                        .iter()
                        .any(|c| c.key.arity == want || c.generic_params.len() == want);
                    if want > 0 && !declared.is_empty() && !has {
                        return Err(LowerError::UnsupportedConstruct(crate::tf!(
                            "rubyc.native.ir.check_type.generic_arity",
                            type_name.clone()
                        )));
                    }
                }
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
                    || is_delegate
                    || is_type_receiver(&type_name);
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
                // A delegate-typed local: the value takes its runtime
                // type from the binding it is converted to, and the
                // binding site is what knows it.
                let saved_delegate = self.expected_delegate.take();
                if ctx.table.is_delegate(&type_name) {
                    self.expected_delegate = Some(type_name.clone());
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
                self.expected_delegate = saved_delegate;
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
                    let v = coerce_span_return(self, ctx, v, &val_ty, &declared_type)?;
                    let allow_unbox = Self::reference_source(init, self, ctx);
                    self.coerce_boxing(v, &val_ty, &declared_type, allow_unbox, ctx)?
                };
                // A STRUCT-typed local is a VALUE: `S b = a;` must copy,
                // not alias. Structs are heap objects here, so a copy is a
                // fresh allocation plus a byte copy. `new` is already a
                // fresh object, so it is left alone; every other
                // struct-typed initializer (an identifier, a member, an
                // array element) is a copy source.
                //
                // Before this, `S a = new S(1); S b = a; b.V = 9;` left
                // `a.V == 9` where C# requires 1.
                let v = {
                    let src_is_new = matches!(init, crate::models::Expr::New { .. });
                    let info = ctx
                        .table
                        .class(&declared_type)
                        .or_else(|| ctx.table.class(&base_name));
                    match info {
                        Some(info) if info.is_struct && !src_is_new => {
                            let bytes = 16 + 8 * info.fields.len() as u64;
                            let copy = self.fresh();
                            self.push_raw(Instr::Alloc {
                                dst: copy,
                                bytes,
                            });
                            let len = self.const_i64(bytes as i64);
                            self.push_raw(Instr::MemCpy {
                                dst: copy,
                                src: v,
                                len,
                            });
                            copy
                        }
                        _ => v,
                    }
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
                    let boxed = self.alloc_struct_box(ctx, entry, Some(&declared_type));
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
                // `E += h` / `E -= h` on an EVENT: the parser desugars it
                // to `E = E + h`, which would do pointer arithmetic on
                // delegate handles. Events hold a subscriber LIST, so the
                // two forms become list operations.
                if let Some((op, rhs)) = event_update_shape(name, value)
                    && let Some(field) = event_field(ctx, name)
                {
                    return self.lower_event_update(&field, op, rhs, ctx);
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
                    let v = self.coerce_boxing(
                        v,
                        &val_ty,
                        tn.as_deref().unwrap_or("object"),
                        false,
                        ctx,
                    )?;
                    // STRUCT assignment copies: `b = a;` must not alias,
                    // or a later `b.V = 9` changes `a`. `new` is already a
                    // fresh object, so it is left alone.
                    let v = {
                        let src_is_new =
                            matches!(value, crate::models::Expr::New { .. });
                        match tn.as_deref().and_then(|t| ctx.table.class(t)) {
                            Some(info) if info.is_struct && !src_is_new => {
                                let bytes = 16 + 8 * info.fields.len() as u64;
                                let copy = self.fresh();
                                self.push_raw(Instr::Alloc {
                                    dst: copy,
                                    bytes,
                                });
                                let len = self.const_i64(bytes as i64);
                                self.push_raw(Instr::MemCpy {
                                    dst: copy,
                                    src: v,
                                    len,
                                });
                                copy
                            }
                            _ => v,
                        }
                    };
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
                        let val_ty = static_type_of(value, self, ctx)
                            .unwrap_or_else(|_| "object".to_owned());
                        let v = self.coerce_boxing(v, &val_ty, &field.type_name, false, ctx)?;
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
                    let v = self.coerce_boxing(v, &val_ty, &field.type_name, false, ctx)?;
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
                    // A get-only property cannot be assigned. Without
                    // this the value was evaluated and DROPPED by the
                    // evaluate-and-drop fallback below — C# reports
                    // CS0200. Only a property DECLARED on this class is
                    // checked: an inherited accessor's declaration is
                    // not in this class's member list, and guessing
                    // would reject valid assignments.
                    // The declaration may be INHERITED, so the whole
                    // chain is consulted; `None` (declaration not in the
                    // table) keeps the legacy behaviour.
                    if info.property(name).is_some()
                        && super::eval::property_settable_in_chain(
                            ctx, &info.name, name,
                        ) == Some(false)
                    {
                        return Err(LowerError::UnsupportedConstruct(format!(
                            "property `{name}` has no setter"
                        )));
                    }
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
                                let v = coerce_span_return(self, ctx, v, &val_ty, ret)?;
                                Some(self.coerce_boxing(v, &val_ty, ret, false, ctx)?)
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
                // A `return` inside `try` with a `finally` runs the
                // finally first: stash the value, mark the return in
                // flight, pop this try's handler (the same pop the
                // normal path performs) and branch to the finally. The
                // finally's tail completes the return, or passes it to
                // an enclosing finally. Without this the block sealed
                // with `Ret` and the finally never ran.
                //
                // Only locals declared inside the try body are released
                // here: the enclosing block's locals (a `using`
                // resource, above all) must survive until the finally
                // has used them. The finally's tail releases the rest.
                if let Some(&(fin, prop, try_depth_scope)) = self.fin_ctx.last() {
                    self.release_heap_above(try_depth_scope);
                    let home = match self.ret_home {
                        Some(h) => h,
                        None => {
                            let h = self.fresh();
                            self.ret_home = Some(h);
                            h
                        }
                    };
                    if let Some(v) = v {
                        self.mov(home, v);
                    }
                    let flag = match self.ret_flag {
                        Some(f) => f,
                        None => {
                            let f = self.fresh();
                            self.ret_flag = Some(f);
                            f
                        }
                    };
                    let one = self.const_i64(1);
                    self.mov(flag, one);
                    self.push_raw(Instr::HandlerPop);
                    self.try_depth = self.try_depth.saturating_sub(1);
                    let two = self.const_i64(2);
                    self.mov(prop, two);
                    self.goto(fin);
                    return Ok(());
                }
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
                        // `obj.P = v` on a property that declares no
                        // setter: C# reports CS0200; this path used to
                        // evaluate the value and drop it. Only a
                        // property declared on the resolved class is
                        // checked (inherited accessors stay legacy).
                        if info.property(name).is_some()
                            && super::eval::property_settable_in_chain(
                                ctx, &info.name, name,
                            ) == Some(false)
                        {
                            return Err(LowerError::UnsupportedConstruct(format!(
                                "property `{name}` has no setter"
                            )));
                        }
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
                let v = self.coerce_boxing(v, &val_ty, &field.type_name, false, ctx)?;
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
            Stmt::CheckedBlock { body, .. } => {
                self.checked_depth += 1;
                let r = self.lower_scoped(body, ctx);
                self.checked_depth -= 1;
                r
            }
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
                // A `case T x:` arm binds the SUBJECT, which the folded
                // path does not do — take the general chain when any arm
                // designates.
                let any_binding = arms.iter().any(|a| a.binding.is_some());
                let any_guard = arms
                    .iter()
                    .any(|a| a.guards.iter().any(|g| g.is_some()));
                if let Some(&sc) = self.const_vals.get(&sv.0)
                    && !any_binding
                    && !any_guard
                {
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

                // A STRING subject compares case labels BY VALUE (C#
                // switch on string), not by pointer; the generic `Cmp`
                // below answered false for equal contents.
                let subject_is_string = static_type_of(subject, self, ctx)
                    .map(|t| is_string_type(&t))
                    .unwrap_or(false);
                let last_arm_idx = arm_targets.len().saturating_sub(1);
                for (i, (arm_b, arm)) in arm_targets.iter().enumerate() {
                    for (li, label) in arm.labels.iter().enumerate() {
                        // A bare TYPE NAME as a label is a type pattern
                        // (`case B:`): the arm matches on the subject's
                        // runtime type, not on equality with a value.
                        // Without this the label evaluated to the class
                        // "value" and the arm never matched.
                        let type_label = match label {
                            crate::models::Expr::Ident { name, .. }
                                if self.scopes.lookup(name).is_none()
                                    && ctx.table.class(name).is_some() =>
                            {
                                Some(name.clone())
                            }
                            _ => None,
                        };
                        let eq = match type_label {
                            Some(tn) => {
                                let simple =
                                    tn.rsplit('.').next().unwrap_or(&tn).to_owned();
                                super::eval::emit_type_test(
                                    self, ctx, sv, &simple, &tn,
                                )?
                            }
                            None => {
                                let lv = eval(label, self, ctx)?;
                                let label_is_string = static_type_of(label, self, ctx)
                                    .map(|t| is_string_type(&t))
                                    .unwrap_or(false);
                                if subject_is_string && label_is_string {
                                    string_equal(self, sv, lv)
                                } else {
                                    self.emit(Instr::Cmp {
                                        dst: VReg(0),
                                        op: CmpOp::Eq,
                                        a: sv,
                                        b: lv,
                                    })
                                }
                            }
                        };
                        let is_last =
                            i == last_arm_idx && li == arm.labels.len() - 1;
                        let next_test = if is_last {
                            after_all_tests
                        } else {
                            self.make_block()
                        };
                        // `case X when g:` — the guard decides AFTER the
                        // label matched, so a false guard falls through
                        // to the next test, not to the arm.
                        if let Some(guard) = arm.guards.get(li).and_then(|g| g.as_ref()) {
                            let guard_b = self.make_block();
                            self.terminate(Terminator::Branch {
                                cond: Some(eq),
                                if_true: guard_b,
                                if_false: next_test,
                            });
                            self.switch_to(guard_b);
                            let gv = eval(guard, self, ctx)?;
                            self.terminate(Terminator::Branch {
                                cond: Some(gv),
                                if_true: *arm_b,
                                if_false: next_test,
                            });
                        } else {
                            self.terminate(Terminator::Branch {
                                cond: Some(eq),
                                if_true: *arm_b,
                                if_false: next_test,
                            });
                        }
                        if !is_last {
                            self.switch_to(next_test);
                        }
                    }
                }
                if !self.is_sealed() {
                    self.goto(after_all_tests);
                }

                for (arm_b, arm) in &arm_targets {
                    self.switch_to(*arm_b);
                    // `case T x:` — the arm was selected because the
                    // subject IS a T, so the designation receives the
                    // subject itself. The binding lives only for this
                    // arm's body.
                    let bound = arm.binding.is_some();
                    if let Some((name, ty)) = &arm.binding {
                        self.scope_enter();
                        let ty = if ty.is_empty() { "object" } else { ty.as_str() };
                        let slot = self
                            .scopes
                            .declare(name, ty, crate::models::Access::Default)
                            .map_err(LowerError::UnsupportedConstruct)?;
                        while self.bindings.len() <= slot {
                            let v = self.fresh();
                            self.bindings.push(v);
                        }
                        self.mov(self.bindings[slot], sv);
                    }
                    // C# switch sections share ONE scope across all arms
                    // (a local declared in `case 0:` is visible in
                    // `case 1:` — redeclaring it is even an error). Lower
                    // arm bodies in the ambient scope, not per-arm scopes.
                    self.block(&arm.body, ctx)?;
                    if bound {
                        self.scope_exit();
                    }
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

    /// Raise `System.OverflowException` when `ovf` is non-zero, then
    /// continue in a fresh block. Used by `checked` arithmetic and
    /// conversions.
    pub(crate) fn throw_overflow_if(
        &mut self,
        ovf: VReg,
        ctx: &Ctx<'_>,
    ) -> Result<(), LowerError> {
        let throw_b = self.make_block();
        let cont_b = self.make_block();
        self.terminate(Terminator::Branch {
            cond: Some(ovf),
            if_true: throw_b,
            if_false: cont_b,
        });
        self.switch_to(throw_b);
        // The exception is an ordinary object: `new System.OverflowException()`
        // through the normal construction path, so it carries the type the
        // program declares (and a `catch (OverflowException)` clause can
        // match it). A program that never declares the type still compiles —
        // nothing can observe it there, since nothing can catch it.
        let ex = Expr::New {
            class_name: "System.OverflowException".to_owned(),
            args: Vec::new(),
            init: Vec::new(),
            span: Default::default(),
        };
        let obj = eval(&ex, self, ctx)?;
        self.terminate(Terminator::Unwind { src: obj });
        self.switch_to(cont_b);
        Ok(())
    }

    /// Emit the overflow test for `a <op> b == r` and raise if it fired.
    ///
    /// Two's-complement tests, expressed with the arithmetic the IR already
    /// has — no new instruction, so both backends (interpreter and x86)
    /// agree without changes:
    ///   add: the operands' signs agree and differ from the result's
    ///   sub: the operands' signs differ and the result's differs from `a`
    ///   mul: recovering `a` from `r / b` (with the zero and -1 cases
    ///        handled separately, because division by zero traps and
    ///        `MIN / -1` is itself an overflow)
    pub(crate) fn checked_arith_guard(
        &mut self,
        op: IrBinOp,
        a: VReg,
        bv: VReg,
        r: VReg,
        ctx: &Ctx<'_>,
        kind: CheckedKind,
    ) -> Result<(), LowerError> {
        let zero = self.const_i64(0);
        // 32-bit types overflow at their own bounds; the i64 result of a
        // 32-bit add/sub/mul always fits, so a range test is exact.
        match kind {
            CheckedKind::Int32 => {
                let min = self.const_i64(i32::MIN as i64);
                let max = self.const_i64(i32::MAX as i64);
                let ovf = match op {
                    IrBinOp::Add | IrBinOp::Sub | IrBinOp::Mul => {
                        let lo = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Lt, a: r, b: min });
                        let hi = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Gt, a: r, b: max });
                        self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitOr, a: lo, b: hi })
                    }
                    // INT_MIN / -1 yields 2147483648, which exceeds INT_MAX.
                    IrBinOp::Div => {
                        self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Gt, a: r, b: max })
                    }
                    _ => return Ok(()),
                };
                return self.throw_overflow_if(ovf, ctx);
            }
            CheckedKind::UInt32 => {
                let max = self.const_i64(u32::MAX as i64);
                let ovf = match op {
                    IrBinOp::Add => {
                        self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Gt, a: r, b: max })
                    }
                    IrBinOp::Sub => {
                        self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Lt, a, b: bv })
                    }
                    IrBinOp::Mul => {
                        // `a * b` overflows when `b != 0 && a > MAX / b`;
                        // forming the product could wrap i64.
                        let home = self.fresh();
                        let no = self.const_i64(0);
                        self.mov(home, no);
                        let b_zero = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Eq, a: bv, b: zero });
                        let zero_b = self.make_block();
                        let div_b = self.make_block();
                        let cont_b = self.make_block();
                        self.terminate(Terminator::Branch {
                            cond: Some(b_zero),
                            if_true: cont_b,
                            if_false: div_b,
                        });
                        self.switch_to(div_b);
                        let q = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Div, a: max, b: bv });
                        let over = self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Gt, a, b: q });
                        self.mov(home, over);
                        self.goto(cont_b);
                        self.switch_to(cont_b);
                        let _ = zero_b;
                        home
                    }
                    _ => return Ok(()),
                };
                return self.throw_overflow_if(ovf, ctx);
            }
            CheckedKind::Int64 => {}
        }
        let ovf = match op {
            IrBinOp::Add => {
                let x = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitXor, a, b: r });
                let y = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitXor, a: bv, b: r });
                let t = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitAnd, a: x, b: y });
                self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Lt, a: t, b: zero })
            }
            IrBinOp::Sub => {
                let x = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitXor, a, b: bv });
                let y = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitXor, a, b: r });
                let t = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitAnd, a: x, b: y });
                self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Lt, a: t, b: zero })
            }
            IrBinOp::Mul => {
                let home = self.fresh();
                let no = self.const_i64(0);
                self.mov(home, no);
                let b_is_zero =
                    self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Eq, a: bv, b: zero });
                let neg_one = self.const_i64(-1);
                let min = self.const_i64(i64::MIN);
                let zero_b = self.make_block();
                let neg_b = self.make_block();
                let div_b = self.make_block();
                let cont_b = self.make_block();
                self.terminate(Terminator::Branch {
                    cond: Some(b_is_zero),
                    if_true: cont_b,
                    if_false: zero_b,
                });
                self.switch_to(zero_b);
                let b_is_neg_one =
                    self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Eq, a: bv, b: neg_one });
                self.terminate(Terminator::Branch {
                    cond: Some(b_is_neg_one),
                    if_true: neg_b,
                    if_false: div_b,
                });
                self.switch_to(neg_b);
                let a_is_min =
                    self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Eq, a, b: min });
                self.mov(home, a_is_min);
                self.goto(cont_b);
                self.switch_to(div_b);
                let q = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Div, a: r, b: bv });
                let differs =
                    self.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Ne, a: q, b: a });
                self.mov(home, differs);
                self.goto(cont_b);
                self.switch_to(cont_b);
                home
            }
            _ => return Ok(()),
        };
        self.throw_overflow_if(ovf, ctx)
    }

    pub(crate) fn lower_throw(&mut self, expr: &Option<Expr>, ctx: &Ctx<'_>) -> Result<(), LowerError> {
        match expr {
            Some(e) => {
                let v = eval(e, self, ctx)?;
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
        if let (Some(fb), Some(p)) = (fin_b, prop) {
            self.fin_ctx.push((fb, p, self.depth));
        }

        // A return inside the try (or a catch) stashes its value and sets
        // the in-flight flag before branching to the finally. Those slots
        // are function-wide, and the finally reads the flag on every path
        // that reaches it — including the exception path, where no return
        // ran. Initialize both before installing the handler so the
        // finally never reads an unassigned slot (which made a throw
        // through a `using` inspect garbage and possibly return it).
        if has_finally
            && (block_has_return(body) || catches.iter().any(|c| block_has_return(&c.body)))
        {
            let home = match self.ret_home {
                Some(h) => h,
                None => {
                    let h = self.fresh();
                    self.ret_home = Some(h);
                    h
                }
            };
            let flag = match self.ret_flag {
                Some(f) => f,
                None => {
                    let f = self.fresh();
                    self.ret_flag = Some(f);
                    f
                }
            };
            let z = self.const_i64(0);
            self.mov(home, z);
            self.mov(flag, z);
        }

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
            // EVERY clause must be tested against the thrown object. This
            // used to run `catches[0]` unconditionally, so ANY typed catch
            // matched ANY thrown object: `try { throw new MyErr(); }
            // catch (Other e)` took the Other branch (measured on both
            // backends). C# selects the first clause whose type the object
            // IS, so each clause gets a block and a real type test; falling
            // off the end propagates the exception.
            let e = self.fresh();
            self.push_raw(Instr::LoadExc { dst: e });
            // Two blocks per clause: a decision block (does this clause
            // claim the object?) and a body block. Sharing one block would
            // seal it on the branch and then try to emit the body into a
            // terminated block — which dropped the body and hung.
            let test_blocks: Vec<BlockId> =
                catches.iter().map(|_| self.make_block()).collect();
            let body_blocks: Vec<BlockId> =
                catches.iter().map(|_| self.make_block()).collect();
            let no_match_b = self.make_block();
            self.goto(test_blocks[0]);

            for (i, clause) in catches.iter().enumerate() {
                self.switch_to(test_blocks[i]);
                // Decide whether this clause claims the object.
                let bare = clause
                    .ty
                    .as_deref()
                    .map(|t| t.rsplit('.').next().unwrap_or(t));
                let is_catch_all = matches!(bare, None | Some("object"));
                let next = if i + 1 < catches.len() {
                    test_blocks[i + 1]
                } else {
                    no_match_b
                };
                // `when (filter)`: evaluated AFTER the type matched, and
                // a false filter falls through to the next clause.
                let filter_b = if clause.filter.is_some() {
                    Some(self.make_block())
                } else {
                    None
                };
                let on_match = filter_b.unwrap_or(body_blocks[i]);
                if is_catch_all {
                    self.goto(on_match);
                } else {
                    let spelling = clause.ty.clone().unwrap_or_default();
                    let simple = bare.unwrap_or("").to_owned();
                    let hit = crate::native::ir::lower::eval::emit_type_test(
                        self, ctx, e, &simple, &spelling,
                    )?;
                    self.terminate(Terminator::Branch {
                        cond: Some(hit),
                        if_true: on_match,
                        if_false: next,
                    });
                }
                // The caught exception is bound BEFORE the filter: C#
                // gives the filter the exception variable
                // (`catch (Exception e) when (e is ArgumentException)`),
                // and the filter runs before the handler does.
                let mut bound_name = false;
                let mut catch_slot: Option<usize> = None;
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
                    catch_slot = Some(slot);
                    bound_name = true;
                }

                if let (Some(fb), Some(guard)) = (filter_b, clause.filter.as_ref()) {
                    self.switch_to(fb);
                    // The store goes in the block that reads it: the
                    // filter is a different block from the type test.
                    if let Some(slot) = catch_slot {
                        self.mov(self.bindings[slot], e);
                    }
                    let gv = eval(guard, self, ctx)?;
                    self.terminate(Terminator::Branch {
                        cond: Some(gv),
                        if_true: body_blocks[i],
                        if_false: next,
                    });
                }

                self.switch_to(body_blocks[i]);
                if let Some(slot) = catch_slot {
                    self.mov(self.bindings[slot], e);
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
            }

            // No clause claimed the object: it stays in flight.
            self.switch_to(no_match_b);
            if let (Some(fb), Some(p)) = (fin_b, prop) {
                let o = self.const_i64(1);
                self.mov(p, o);
                self.goto(fb);
            } else {
                self.terminate(Terminator::Rethrow);
            }
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
                // C# forbids leaving a finally clause (CS0157). Routing a
                // return from here would find this same finally as its
                // "enclosing" one and jump to it again — a silent hang.
                if block_has_return(fin_blk) {
                    return Err(LowerError::UnsupportedConstruct(
                        "return inside a finally clause is not allowed (CS0157)"
                            .to_owned(),
                    ));
                }
                self.lower_scoped(fin_blk, ctx)?;
            }
            if !self.is_sealed() {
                // A return in flight leaves through this finally as
                // well: complete it at the outermost finally, or hand
                // it to the next enclosing one (popping that try's
                // handler first, as its own normal path does).
                if let Some(flag) = self.ret_flag {
                    let retb = self.make_block();
                    let after = self.make_block();
                    let one = self.const_i64(1);
                    let pending = self.emit(Instr::Cmp {
                        dst: VReg(0),
                        op: CmpOp::Eq,
                        a: flag,
                        b: one,
                    });
                    self.terminate(Terminator::Branch {
                        cond: Some(pending),
                        if_true: retb,
                        if_false: after,
                    });
                    self.switch_to(retb);
                    let outer = self
                        .fin_ctx
                        .len()
                        .checked_sub(2)
                        .and_then(|i| self.fin_ctx.get(i))
                        .copied();
                    if let Some((ofin, oprop, _)) = outer {
                        self.push_raw(Instr::HandlerPop);
                        self.try_depth = self.try_depth.saturating_sub(1);
                        let two = self.const_i64(2);
                        self.mov(oprop, two);
                        self.goto(ofin);
                    } else {
                        // The return completes here: release the locals
                        // the routed return kept alive (the resource the
                        // finally just disposed).
                        self.release_heap_above(1);
                        self.pop_trys();
                        let home = match self.ret_home {
                            Some(h) => h,
                            None => {
                                let h = self.fresh();
                                self.ret_home = Some(h);
                                h
                            }
                        };
                        self.terminate(Terminator::Ret(Some(home)));
                    }
                    self.switch_to(after);
                }
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

        // The finally's context ends with its block; returns lowered
        // after this try belong to an outer one.
        if has_finally {
            self.fin_ctx.pop();
        }

        // join / continuation
        self.switch_to(join_b);
        Ok(())
    }

    /// `E += h` / `E -= h`: prepend a subscriber node, or unlink the
    /// first node whose delegate matches `rhs`.
    pub(crate) fn lower_event_update(
        &mut self,
        field: &crate::native::semantic::FieldInfo,
        add: bool,
        rhs: &Expr,
        ctx: &Ctx<'_>,
    ) -> Result<(), LowerError> {
        let off = (field.offset_slots * 8) as u32;
        let this = self.this_vreg();
        let head = self.fresh();
        self.push_raw(Instr::LoadField { dst: head, base: this, off });
        let newd = eval(rhs, self, ctx)?;
        if add {
            // APPEND, so handlers run in subscription order (C# invokes
            // them in the order added; prepending ran them backwards).
            let node = self.fresh();
            self.push_raw(Instr::Alloc { dst: node, bytes: 24 });
            let zero = self.const_i64(0);
            self.push_raw(Instr::StoreField { base: node, off: 0, src: zero });
            self.push_raw(Instr::StoreField { base: node, off: 8, src: newd });
            let empty = self.fresh();
            let z = self.const_i64(0);
            self.push_raw(Instr::Cmp { dst: empty, op: CmpOp::Eq, a: head, b: z });
            let set_head = self.make_block();
            let walk = self.make_block();
            let done = self.make_block();
            self.terminate(Terminator::Branch {
                cond: Some(empty),
                if_true: set_head,
                if_false: walk,
            });
            self.switch_to(set_head);
            self.push_raw(Instr::StoreField { base: this, off, src: node });
            self.goto(done);
            self.switch_to(walk);
            let cur = self.fresh();
            self.mov(cur, head);
            let loop_b = self.make_block();
            let body_b = self.make_block();
            self.goto(loop_b);
            self.switch_to(loop_b);
            let nx = self.fresh();
            self.push_raw(Instr::LoadField { dst: nx, base: cur, off: 0 });
            let is_last = self.fresh();
            let z2 = self.const_i64(0);
            self.push_raw(Instr::Cmp { dst: is_last, op: CmpOp::Eq, a: nx, b: z2 });
            self.terminate(Terminator::Branch {
                cond: Some(is_last),
                if_true: body_b,
                if_false: loop_b,
            });
            // advance
            self.switch_to(body_b);
            self.push_raw(Instr::StoreField { base: cur, off: 0, src: node });
            self.goto(done);
            self.switch_to(done);
            return Ok(());
        }
        // remove: walk prev/cur, unlink the first delegate whose
        // fn_addr and kind both match (C# delegate equality is
        // target + method; captures are not part of it).
        let want_fn = self.fresh();
        self.push_raw(Instr::LoadField { dst: want_fn, base: newd, off: 0 });
        let want_kind = self.fresh();
        self.push_raw(Instr::LoadField { dst: want_kind, base: newd, off: 8 });
        let prev = self.fresh();
        let zero = self.const_i64(0);
        self.mov(prev, zero);
        let cur = self.fresh();
        self.mov(cur, head);
        let loop_b = self.make_block();
        let body_b = self.make_block();
        let unlink_b = self.make_block();
        let next_b = self.make_block();
        let done_b = self.make_block();
        self.goto(loop_b);
        self.switch_to(loop_b);
        let none = self.fresh();
        let z2 = self.const_i64(0);
        self.push_raw(Instr::Cmp { dst: none, op: CmpOp::Eq, a: cur, b: z2 });
        self.terminate(Terminator::Branch {
            cond: Some(none),
            if_true: done_b,
            if_false: body_b,
        });
        self.switch_to(body_b);
        let d = self.fresh();
        self.push_raw(Instr::LoadField { dst: d, base: cur, off: 8 });
        let df = self.fresh();
        self.push_raw(Instr::LoadField { dst: df, base: d, off: 0 });
        let dk = self.fresh();
        self.push_raw(Instr::LoadField { dst: dk, base: d, off: 8 });
        let f_eq = self.fresh();
        self.push_raw(Instr::Cmp { dst: f_eq, op: CmpOp::Eq, a: df, b: want_fn });
        let k_eq = self.fresh();
        self.push_raw(Instr::Cmp { dst: k_eq, op: CmpOp::Eq, a: dk, b: want_kind });
        let hit = self.fresh();
        self.push_raw(Instr::Bin { dst: hit, op: IrBinOp::BitAnd, a: f_eq, b: k_eq });
        self.terminate(Terminator::Branch {
            cond: Some(hit),
            if_true: unlink_b,
            if_false: next_b,
        });
        self.switch_to(unlink_b);
        let nn = self.fresh();
        self.push_raw(Instr::LoadField { dst: nn, base: cur, off: 0 });
        let is_head = self.fresh();
        let hz = self.const_i64(0);
        self.push_raw(Instr::Cmp { dst: is_head, op: CmpOp::Eq, a: prev, b: hz });
        let set_head = self.make_block();
        let set_prev = self.make_block();
        self.terminate(Terminator::Branch {
            cond: Some(is_head),
            if_true: set_head,
            if_false: set_prev,
        });
        self.switch_to(set_head);
        self.push_raw(Instr::StoreField { base: this, off, src: nn });
        self.goto(done_b);
        self.switch_to(set_prev);
        self.push_raw(Instr::StoreField { base: prev, off: 0, src: nn });
        self.goto(done_b);
        self.switch_to(next_b);
        self.mov(prev, cur);
        let nx = self.fresh();
        self.push_raw(Instr::LoadField { dst: nx, base: cur, off: 0 });
        self.mov(cur, nx);
        self.goto(loop_b);
        self.switch_to(done_b);
        Ok(())
    }

    pub(crate) fn this_vreg(&self) -> VReg {
        VReg(0)
    }

    /// Save the inline receiver home before an inlined accessor body
    /// overwrites vreg 0 with the accessor's receiver. A static method
    /// has no receiver — vreg 0 is reserved padding — so there is nothing
    /// to preserve, and saving would read an unassigned slot.
    pub(crate) fn save_this(&mut self) -> Option<VReg> {
        if self.f.receiver == ReceiverKind::None {
            return None;
        }
        let saved = self.fresh();
        self.mov(saved, VReg(0));
        Some(saved)
    }

    /// Undo [`Body::save_this`] after the inlined body.
    pub(crate) fn restore_this(&mut self, saved: Option<VReg>) {
        if let Some(saved) = saved {
            self.mov(VReg(0), saved);
        }
    }
}


/// `E = E + h` / `E = E - h` (the parser's desugaring of `E += h` /
/// `E -= h`): returns (is_add, rhs).
fn event_update_shape<'a>(name: &str, value: &'a Expr) -> Option<(bool, &'a Expr)> {
    let Expr::Binary { op, lhs, rhs, .. } = value else {
        return None;
    };
    let Expr::Ident { name: target, .. } = lhs.as_ref() else {
        return None;
    };
    if target != name {
        return None;
    }
    match op {
        crate::models::BinOp::Add => Some((true, rhs)),
        crate::models::BinOp::Sub => Some((false, rhs)),
        _ => None,
    }
}

/// The EVENT `name` (an instance event of the enclosing class), if any.
pub(super) fn event_field(
    ctx: &Ctx<'_>,
    name: &str,
) -> Option<crate::native::semantic::FieldInfo> {
    let mut is_event = false;
    for c in ctx.classes {
        if c.name != ctx.current_class {
            continue;
        }
        is_event = c.members.iter().any(|m| {
            matches!(m, crate::models::ClassMember::Event(e) if e.name == name)
        });
    }
    if !is_event {
        return None;
    }
    let info = ctx.table.class(ctx.current_class)?;
    info.field(name).cloned()
}


/// String equality on two evaluated STRING objects: equal length and
/// equal bytes. Used where the generic `Cmp` would compare pointers
/// (switch on string, case labels).
fn string_equal(b: &mut Body<'_>, a: VReg, c: VReg) -> VReg {
    let len_a = b.fresh();
    b.push_raw(Instr::LoadField { dst: len_a, base: a, off: 0 });
    let len_c = b.fresh();
    b.push_raw(Instr::LoadField { dst: len_c, base: c, off: 0 });
    let len_eq = b.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Eq, a: len_a, b: len_c });
    let s16 = b.const_i64(16);
    let data_a = b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a, b: s16 });
    let data_c = b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: c, b: s16 });
    let same = b.fresh();
    b.push_raw(Instr::MemCmp { dst: same, a: data_a, b: data_c, len: len_a });
    b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitAnd, a: len_eq, b: same })
}
