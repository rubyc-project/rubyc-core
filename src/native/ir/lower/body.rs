//! AST to IR body lowering.

use super::super::{BlockId, CmpOp, Instr, IrFunction, LowerError, Terminator, VReg};
use crate::native::ir::{ReceiverKind, StringArena};
use crate::models::symbols::ScopeStack;
use crate::models::{Expr, Stmt};
use crate::native::ir::BinOp as IrBinOp;
use crate::native::ir::Ctx;
use super::{calls::*, eval::*, select::*, types::*};

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
    cur: BlockId,
    cur_sealed: bool,
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
            cur: 0,
            cur_sealed: false,
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
    pub(crate) fn push_raw(&mut self, i: Instr) {
        debug_assert!(!self.cur_sealed, "instruction after terminator");
        self.instrs_of().push(i);
    }

    /// Seal the current block. Further pushes assert.
    pub(crate) fn terminate(&mut self, t: Terminator) {
        debug_assert!(!self.cur_sealed, "double terminator");
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
        let info = ctx.table.class(type_name)?;
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

    /// Concatenate two heap strings (C# `+` on `string`): allocates a new
    /// object whose data is the byte sequence of `a` followed by `b`.
    pub(crate) fn concat_strings(&mut self, a: VReg, b: VReg) -> VReg {
        let len_a = {
            let d = self.fresh();
            self.push_raw(Instr::LoadField { dst: d, base: a, off: 0 });
            d
        };
        let len_b = {
            let d = self.fresh();
            self.push_raw(Instr::LoadField { dst: d, base: b, off: 0 });
            d
        };
        let total = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: len_a, b: len_b });
        let s16a = self.const_i64(16);
        // Object = 16-byte header (len + refcount) + data.
        let total_sz = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: total, b: s16a });
        let newobj = self.fresh();
        self.push_raw(Instr::AllocDyn { dst: newobj, bytes: total_sz });
        self.push_raw(Instr::StoreField { base: newobj, off: 0, src: total });
        let one = self.const_i64(1);
        self.push_raw(Instr::StoreField { base: newobj, off: 8, src: one });
        let s16 = self.const_i64(16);
        let data_new = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: newobj, b: s16 });
        let data_a = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a, b: s16 });
        self.raw_vreg(Instr::MemCpy { dst: data_new, src: data_a, len: len_a });
        let ptr2 = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: data_new, b: len_a });
        let data_b = self.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: b, b: s16 });
        self.raw_vreg(Instr::MemCpy { dst: ptr2, src: data_b, len: len_b });
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
            if self.is_sealed() {
                break; // unreachable code after return/break/continue
            }
            self.statement(stmt, ctx)?;
        }
        Ok(())
    }

    /// Seal the current block with `return 0` when control can still fall
    /// out of it (used for method bodies and @start).
    pub(crate) fn seal_with_return_zero(&mut self) {
        if !self.is_sealed() {
            let z = self.const_i64(0);
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

    pub(crate) fn statement(&mut self, stmt: &Stmt, ctx: &Ctx<'_>) -> Result<(), LowerError> {
        match stmt {
            Stmt::Let {
                type_name,
                name,
                init,
                access,
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
                        span,
                    } if class_name.is_empty() => {
                        let filled = crate::models::Expr::New {
                            class_name: type_name.clone(),
                            args: args.clone(),
                            span: *span,
                        };
                        eval(&filled, self, ctx)?
                    }
                    other => eval(other, self, ctx)?,
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
                    // Heap handles share: retain the incoming value, drop
                    // the old one, then move.
                    let tn = self.scopes.lookup(name).map(|s| s.type_name.clone());
                    let bytes = tn.as_deref().and_then(|t| self.handle_bytes(t, ctx));
                    match bytes {
                        Some(bytes) => {
                            let v = eval(value, self, ctx)?;
                            self.emit_retain(v);
                            self.emit_release(home, bytes);
                            self.mov(home, v);
                        }
                        None => {
                            let v = eval(value, self, ctx)?;
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
                    self.push_raw(Instr::StoreField { base: this, off: (field.offset_slots * 8) as u32, src: v });
                    return Ok(());
                }
                // Accessor-backed property write (`HResult = x` where
                // `HResult` has explicit get/set bodies and no storage
                // slot): no setter function exists yet (mirrors the
                // read-side zero placeholder) — evaluate for side
                // effects and drop. Auto-properties have backing fields
                // and store through the path above.
                if this_info(ctx, name).is_some()
                {
                    let _ = eval(value, self, ctx)?;
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
                    Some(e) => Some(eval(e, self, ctx)?),
                    None => Some(self.const_i64(0)),
                };
                // Leaving the method releases every enclosing scope's heap
                // locals along this path.
                self.release_heap_above(1);
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
                    // no setter function exists yet — evaluate for side
                    // effects and drop (mirrors the implicit-this path).
                    if member_host(ctx.table, &type_name, name).is_some() {
                        let _ = eval(object, self, ctx)?;
                        let _ = eval(value, self, ctx)?;
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
                let base = eval(object, self, ctx)?;
                let v = eval(value, self, ctx)?;
                // The write barrier applies only when the FIELD itself
                // holds a reference. Keying it off the receiver's type
                // retained plain integers stored into int fields and
                // corrupted the heap.
                let field_is_ref = info
                    .field(name)
                    .map(|fi| ctx.table.class(&fi.type_name).is_some())
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
            Stmt::Block { body, .. } => self.lower_scoped(body, ctx),
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
            Stmt::Goto { .. } => {
                // Label-based jumps are not representable in the native IR.
                // Skip as a no-op so compilation succeeds.
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
            if let (Some(fb), Some(p)) = (fin_b, prop) {
                let z = self.const_i64(0);
                self.mov(p, z);
                self.goto(fb);
            } else {
                self.goto(join_b);
            }
        }

        // exception entry: reached when `h == 1`; the throw already popped the
        // handler, so no `HandlerPop` here.
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
