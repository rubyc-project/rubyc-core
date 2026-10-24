//! AST to IR body lowering.

use super::super::{CmpOp, FBinOp, FCmpOp, Instr, LowerError, Terminator, VReg};
use crate::models::{Block, ClassMember, Expr, InterpPart, Literal, UnOp};
use crate::native::ir::BinOp as IrBinOp;
use crate::native::ir::Ctx;
use crate::native::semantic::{
    ConstValue, FIRST_FIELD_SLOT, INSTANCE_HEADER_SLOTS, RUNTIME_TYPE_SLOT,
};
use super::{body::*, calls::*, select::*, span::*, types::*};
use super::typeinfo::{
    is_type_receiver as is_descriptor_receiver, primitive_id, type_member, type_meta, BOX_FLAG,
    TYPE_TAG,
};
/// Constant-fold a non-comparison binary op. `None` means "leave to the
/// runtime": undefined division/rem and everything unhandled.
pub(super) fn fold_arith(op: &crate::models::BinOp, va: i64, vb: i64) -> Option<i64> {
    use crate::models::BinOp as B;
    match op {
        B::Add => Some(va.wrapping_add(vb)),
        B::Sub => Some(va.wrapping_sub(vb)),
        B::Mul => Some(va.wrapping_mul(vb)),
        B::Div => {
            if vb != 0 && !(va == i64::MIN && vb == -1) {
                Some(va.wrapping_div(vb))
            } else {
                None
            }
        }
        B::Rem => {
            if vb != 0 && !(va == i64::MIN && vb == -1) {
                Some(va.wrapping_rem(vb))
            } else {
                None
            }
        }
        B::BitAnd => Some(va & vb),
        B::BitOr => Some(va | vb),
        B::BitXor => Some(va ^ vb),
        // Hardware masks shift counts to 6 bits on 64-bit operands.
        B::Shl => Some(va.wrapping_shl((vb & 63) as u32)),
        B::Shr => Some(va.wrapping_shr((vb & 63) as u32)),
        B::ShrShr => Some(va.wrapping_shr((vb & 63) as u32)),
        _ => None,
    }
}

/// True when a field of this declared type holds a managed
/// reference (store needs write-barrier retain/release): the type
/// must name a table entry that is NOT an enum. Enums lower as raw
/// ints, but they ARE table entries (pass 3b registers every enum
/// as a const-field holder) — the old existence-only check
/// retained/released enum ints (`HebrewToken` value 4 faulted in
/// `lock inc [rax+8]` during `HebrewNumber` static init). Structs
/// stay true (their slots hold refcounted box pointers); arrays of
/// any rank stay true (heap objects whatever the element type);
/// nullable (`T?`) unwraps to `T`. Primitives, pointers, and
/// unknowns stay false.
pub(super) fn is_heap_ref(
    table: &crate::native::semantic::TypeTable,
    type_name: &str,
) -> bool {
    let mut s = type_name.trim();
    // Arrays of any rank hold refcounted heap objects regardless of
    // element type (`int[]` needs its retain as much as `string[]`).
    if s.ends_with(']') {
        return true;
    }
    // Nullable annotations (`NumberFormatInfo?`) ride on the inner
    // type: the slot still holds an object pointer (or null). Without
    // this, every `T?` field store skipped its write barrier and the
    // stored object could be freed out from under the field (the
    // `_numInfo` use-after-free that faulted `GetProviderNonNull`).
    // Pointers (`T*`) are deliberately NOT stripped: raw addresses
    // must never be retained.
    loop {
        if let Some(rest) = s.strip_suffix('?') {
            s = rest.trim();
        } else {
            break;
        }
    }
    // Pointer-sized integers are raw addresses/ints, never heap
    // objects (mirrors `handle_bytes`): retaining them faults (or
    // corrupts) once nonzero.
    {
        let base = s.rsplit('.').next().unwrap_or(s);
        if matches!(base, "IntPtr" | "UIntPtr" | "nint" | "nuint") {
            return false;
        }
    }
    table
        .class(s)
        .is_some_and(|e| !e.is_enum)
}

/// True for integer/char/bool/float primitives and enums: values
/// that compare by their 8-byte slot. Used to fold
/// `x.Equals(y)` into an integer comparison instead of dispatching
/// on the raw int as an object (which faults). `decimal` is
/// excluded (16-byte struct, not a slot compare); floats carry the
/// usual `-0.0`/NaN caveat (mirrors `CompareExchange`).
fn is_value_comparable(table: &crate::native::semantic::TypeTable, type_name: &str) -> bool {
    let t = type_name.trim();
    if t.ends_with('?') || t.ends_with(']') || t.ends_with('*') {
        return false;
    }
    let base = t.rsplit('.').next().unwrap_or(t);
    if matches!(
        base,
        "SByte" | "Byte" | "Int16" | "UInt16" | "Int32" | "UInt32" | "Int64"
            | "UInt64" | "IntPtr" | "UIntPtr" | "Char" | "Single" | "Double"
            | "Boolean" | "sbyte" | "byte" | "short" | "ushort" | "int"
            | "uint" | "long" | "ulong" | "nint" | "nuint" | "char"
            | "float" | "double" | "bool"
    ) {
        return true;
    }
    table
        .class(base)
        .or_else(|| table.class(t))
        .is_some_and(|e| e.is_enum)
}

/// Struct entry for a receiver spelling (mirrors the static-struct
/// materialization lookup): full path first, then the simple-name
/// fallback. `None` for primitives, nullables (`T?`), references,
/// and unknowns.
fn struct_entry_for<'t>(
    table: &'t crate::native::semantic::TypeTable,
    type_name: &str,
) -> Option<&'t crate::native::semantic::ClassInfo> {
    if type_name.starts_with("Nullable<") || type_name.trim_end().ends_with('?') {
        return None;
    }
    table
        .class(type_name)
        .filter(|e| e.is_struct && !e.fields.is_empty())
        .or_else(|| {
            let simple = type_name.rsplit('.').next().unwrap_or(type_name);
            let simple = simple.split('<').next().unwrap_or(simple);
            table
                .classes
                .iter()
                .find(|c| c.name == simple && c.is_struct && !c.fields.is_empty())
        })
}

/// Fold a read of a string interior field on an already-evaluated
/// base: `_stringLength` loads the character count at +0,
/// `_firstChar` yields the character data ADDRESS at +16 (all uses
/// are `ref`-position; see the Member-arm comment). Call only when
/// the receiver is a string and `name` is one of the two.
pub(super) fn string_interior_read(b: &mut Body<'_>, base: VReg, name: &str) -> VReg {
    if name == "_stringLength" {
                let dst = b.fresh();
                b.push_raw(Instr::LoadField { dst, base, off: 0 });
                // Uninitialized struct statics read null, but member
        dst
    } else {
        debug_assert_eq!(name, "_firstChar");
        let sixteen = b.const_i64(16);
        b.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::Add,
            a: base,
            b: sixteen,
        })
    }
}

/// Box evaluated call arguments for by-reference (`ref`/`out`/`in`)
/// parameters so callee writes flow back: each boxed value goes in a
/// fresh 16-byte heap cell whose address replaces the argument, and
/// the caller copies the cell back afterwards (see
/// `emit_writebacks`). `byref` aligns with the callee's parameters
/// (extra entries ignored). Rules per argument:
/// - `Ident` local/param, itself by-reference: already an address —
///   pass through untouched (aliasing preserved, no box).
/// - `Ident` local/param (or `_`/discard, or `Default` from `out`
///   declarations): fresh box (zeroed for discards), pass its
///   address. Locals record a writeback; discards do not.
/// - Anything else: box the evaluated value (reads keep working via
///   dereference) but record no writeback (complex targets cannot be
///   stored back without re-evaluating side effects — documented
///   gap; previously such writes were silently lost too).
/// Returns `(caller_home, cell)` writebacks in argument order.
pub(super) fn box_byref_args(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    args: &[Expr],
    argv: &mut Vec<VReg>,
    byref: &[bool],
) -> Result<Vec<(VReg, VReg)>, LowerError> {
    let mut wbs: Vec<(VReg, VReg)> = Vec::new();
    let n = argv.len().min(byref.len()).min(args.len());
    for i in 0..n {
        if !byref[i] {
            continue;
        }        // Byref-to-byref passthrough (address already).
        if let Expr::Ident { name, .. } = &args[i] {
            if name != "_" && name != "this" {
                if let Some(sym) = b.scopes.lookup(name) {
                    if sym.is_byref {
                        argv[i] = b.bindings[sym.slot];
                        continue;
                    }
                }
            }
        }
        // Addressable targets (`ref p[i]`, `ref arr[i]`,
        // `ref obj.field`): pass the interior address itself —
        // callee writes land in place, no box, no writeback.
        // (Boxing these lost every `Narrow*` byte write.)
        // Explicit address that resolves (`&x`, `ref *p`): argv
        // already holds the address — boxing it would pass
        // pointer-to-pointer. Unresolved addresses fall through
        // to the cell path (legacy value behavior). Syntactic
        // check only (no emission — re-evaluating impure bases
        // would run side effects twice).
        if let Expr::Unary {
            op: crate::models::ast::expression::UnOp::AddrOf,
            inner,
            ..
        } = &args[i]
        {
            let resolves = matches!(
                inner.as_ref(),
                Expr::Index { .. }
                    | Expr::Member { .. }
                    | Expr::MethodCall { .. }
                    | Expr::Call { .. }
            ) || matches!(
                inner.as_ref(),
                Expr::Unary {
                    op: crate::models::ast::expression::UnOp::Deref,
                    ..
                }
            );
            if resolves {
                continue;
            }
        }
        if let Some(addr) = ref_target_address(b, ctx, &args[i])? {
            argv[i] = addr;
            continue;
        }
        // Fresh cell for the value.
        let cell = b.fresh();
        b.push_raw(Instr::Alloc { dst: cell, bytes: 16 });
        match &args[i] {
            Expr::Ident { name, .. }
                if name != "_" && name != "this" =>
            {
                if let Some(sym) = b.scopes.lookup(name) {
                    let home = b.bindings[sym.slot];
                    b.push_raw(Instr::StoreField {
                        base: cell,
                        off: 0,
                        src: argv[i],
                    });
                    argv[i] = cell;
                    wbs.push((home, cell));
                    continue;
                }
                // Unresolvable Ident: box the value, no writeback.
                b.push_raw(Instr::StoreField {
                    base: cell,
                    off: 0,
                    src: argv[i],
                });
                argv[i] = cell;
            }
            // Call-site `ref`/`in` markers (parser `&`-wrap) over a
            // plain local: argv already holds the VALUE (eval falls
            // back to inner for locals, which have no addressable
            // home) — box it WITH writeback like a bare Ident so
            // callee mutations flow home (the legacy `_` arm below
            // would drop them silently). Byref homes pass through;
            // other inners keep their existing paths above.
            Expr::Unary {
                op: crate::models::ast::expression::UnOp::AddrOf,
                inner,
                ..
            } if matches!(inner.as_ref(), Expr::Ident { name, .. } if name != "_" && name != "this") =>
            {
                if let Expr::Ident { name, .. } = inner.as_ref() {
                    if let Some(sym) = b.scopes.lookup(name) {
                        if sym.is_byref {
                            argv[i] = b.bindings[sym.slot];
                            continue;
                        }
                        let home = b.bindings[sym.slot];
                        b.push_raw(Instr::StoreField {
                            base: cell,
                            off: 0,
                            src: argv[i],
                        });
                        argv[i] = cell;
                        wbs.push((home, cell));
                        continue;
                    }
                }
                // Unresolvable Ident: box the value, no writeback.
                b.push_raw(Instr::StoreField {
                    base: cell,
                    off: 0,
                    src: argv[i],
                });
                argv[i] = cell;
            }
            // Anything else (discards, complex targets): box the
            // evaluated value so reads work via dereference; no
            // writeback (documented gap).
            _ => {
                b.push_raw(Instr::StoreField {
                    base: cell,
                    off: 0,
                    src: argv[i],
                });
                argv[i] = cell;
            }
        }
    }
    Ok(wbs)
}

/// Copy boxed by-reference results back to caller homes after a call.
/// `wbs` comes from `box_byref_args`, in argument order.
pub(super) fn emit_writebacks(b: &mut Body<'_>, wbs: &[(VReg, VReg)]) {
    for (home, cell) in wbs {
        b.push_raw(Instr::LoadField {
            dst: *home,
            base: *cell,
            off: 0,
        });
    }
}

/// Side-effect-free expressions (safe to evaluate twice when forming
/// a by-reference address — the value was already computed once).
fn is_pure_arg(e: &Expr) -> bool {
    match e {
        Expr::Ident { .. } => true,
        Expr::Literal(_) => true,
        Expr::Member { object, .. } => is_pure_arg(object),
        _ => false,
    }
}

/// Address of a by-reference argument target (`ref p[i]`,
/// `ref arr[i]`, `ref obj.field`): lets callee writes land in
/// place instead of a writeback-less box (which silently dropped
/// every `Narrow*` byte write). Only pure shapes (re-evaluation is
/// side-effect-free); anything else keeps the boxing path. Returns
/// `None` when not applicable.
/// The address of a bare instance field — the implicit-`this`
/// spelling `this.field`: the receiver plus the field's slot offset.
/// `ref_target_address` cannot see it, since that only walks explicit
/// `x.f` members.
fn implicit_this_field_address(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    name: &str,
) -> Result<Option<VReg>, LowerError> {
    let Some(info) = ctx.current_info else {
        return Ok(None);
    };
    let Some(f) = info.field(name) else {
        return Ok(None);
    };
    if f.is_static {
        return Ok(None);
    }
    if f.is_byref {
        // A ref field holds the address itself (ref-of-ref collapses in
        // C#).
        let addr = b.fresh();
        b.push_raw(Instr::LoadField {
            dst: addr,
            base: VReg(0),
            off: (f.offset_slots * 8) as u32,
        });
        return Ok(Some(addr));
    }
    let off = b.const_i64((f.offset_slots * 8) as i64);
    Ok(Some(b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: VReg(0),
        b: off,
    })))
}

/// The ADDRESS of the object a member access reads: the object's own
/// value for a local or a local's field, but the object's ADDRESS when
/// the object is itself a field (`&this.a.b` — the field's address is
/// the address of `a` plus the offset, and `a`'s value is not its
/// address).
fn member_base_address(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    object: &Expr,
) -> Result<VReg, LowerError> {
    if let Expr::Ident { name, .. } = object {
        if name == "this" {
            return Ok(VReg(0));
        }
        // A field of the receiver. A reference-typed field holds the
        // object it points at, so the member's base is the field's
        // VALUE. A value-typed field (a struct) IS the member's base,
        // so the base is the field's ADDRESS.
        if let Some(info) = ctx.current_info
            && let Some(f) = info.field(name)
            && !f.is_static
        {
            let is_value = ctx
                .table
                .class(&f.type_name)
                .is_some_and(|entry| entry.is_struct);
            if is_value {
                if let Some(addr) = implicit_this_field_address(b, ctx, name)? {
                    return Ok(addr);
                }
            } else {
                return eval(object, b, ctx);
            }
        }
    }
    if let Some(addr) = ref_target_address(b, ctx, object)? {
        return Ok(addr);
    }
    eval(object, b, ctx)
}

pub(super) fn ref_target_address(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    arg: &Expr,
) -> Result<Option<VReg>, LowerError> {
    // `ref arr[i]` / `ref p[i]` with pure base+index.
    if let Expr::Index { object, index, .. } = arg {
        if !is_pure_arg(object) || !is_pure_arg(index) {
            return Ok(None);
        }
        // A by-reference base contributes its home (address);
        // value-position eval would hand the pointee.
        let base_of = |b: &mut Body<'_>| -> Result<VReg, LowerError> {
            if let Expr::Ident { name, .. } = object.as_ref() {
                if let Some(h) = byref_home(b, name) {
                    return Ok(h);
                }
            }
            eval(object, b, ctx)
        };
        let recv_ty = static_type_of(object, b, ctx)?;
        let t = recv_ty.trim();
        // Strip a trailing nullable annotation for shape dispatch
        // (`char*?` addresses like `char*`; `char[]?` below).
        let t_nq = t.strip_suffix('?').unwrap_or(t);
        // Raw pointer: `base + idx*w`.
        if t_nq.ends_with('*') && !t_nq.ends_with("[]") {
            let inner = t_nq.trim_end_matches(['*', ' ']).trim();
            let w = span_elem_width(inner);
            let base = base_of(b)?;
            let idx = eval(index, b, ctx)?;
            let wv = b.const_i64(w as i64);
            let off = b.emit(Instr::Bin {
                dst: VReg(0),
                op: IrBinOp::Mul,
                a: idx,
                b: wv,
            });
            return Ok(Some(b.emit(Instr::Bin {
                dst: VReg(0),
                op: IrBinOp::Add,
                a: base,
                b: off,
            })));
        }
        // Array: `base + 16 + idx*w` (nullable-annotated arrays
        // included — `char[]?` fields address like `char[]`).
        if let Some(inner) = array_elem_ty(t) {
            let w = span_elem_width(&inner);
            let base = base_of(b)?;
            let idx = eval(index, b, ctx)?;
            let wv = b.const_i64(w as i64);
            let sixteen = b.const_i64(16);
            let off = b.emit(Instr::Bin {
                dst: VReg(0),
                op: IrBinOp::Mul,
                a: idx,
                b: wv,
            });
            let off16 = b.emit(Instr::Bin {
                dst: VReg(0),
                op: IrBinOp::Add,
                a: off,
                b: sixteen,
            });
            return Ok(Some(b.emit(Instr::Bin {
                dst: VReg(0),
                op: IrBinOp::Add,
                a: base,
                b: off16,
            })));
        }
        // String char: 1 byte at `base + 16 + i` (mirrors the read
        // path exactly — same layout, same width).
        if is_string_type(t) {
            let base = base_of(b)?;
            let idx = eval(index, b, ctx)?;
            let sixteen = b.const_i64(16);
            let off16 = b.emit(Instr::Bin {
                dst: VReg(0),
                op: IrBinOp::Add,
                a: idx,
                b: sixteen,
            });
            return Ok(Some(b.emit(Instr::Bin {
                dst: VReg(0),
                op: IrBinOp::Add,
                a: base,
                b: off16,
            })));
        }
        return Ok(None);
    }
    // `ref obj.field` with pure base: `base + fieldOff`.
    if let Expr::Member { object, name, .. } = arg {
        if !is_pure_arg(object) {
            return Ok(None);
        }
        let recv_ty = static_type_of(object, b, ctx)?;
        // A nullable annotation is not part of the type's name
        // (`CompletionState?` is `CompletionState`).
        let recv_base = recv_ty.trim().trim_end_matches('?');
        if let Some(info) = member_entry(ctx.table, recv_base) {
            if let Some(f) = info.field(name) {
                if f.is_static {
                    // A static field HAS an address: its storage slot
                    // (or, for a field owned by another image, the
                    // owner's slot via address import). `ref Type
                    // .Field` passes that address.
                    if let Some(addr) = static_addr(f, b, ctx) {
                        return Ok(Some(addr));
                    }
                } else {
                    // `ref` fields hold addresses: `ref field`
                    // collapses to the target (C# ref-of-ref collapses),
                    // so load the slot instead of taking its address
                    // (which would copy pointer bytes instead of data
                    // in `Memmove`).
                    if f.is_byref {
                        let base = eval(object, b, ctx)?;
                        let addr = b.fresh();
                        b.push_raw(Instr::LoadField {
                            dst: addr,
                            base,
                            off: (f.offset_slots * 8) as u32,
                        });
                        return Ok(Some(addr));
                    }
                    let base = member_base_address(b, ctx, object)?;
                    let off = b.const_i64((f.offset_slots * 8) as i64);
                    return Ok(Some(b.emit(Instr::Bin {
                        dst: VReg(0),
                        op: IrBinOp::Add,
                        a: base,
                        b: off,
                    })));
                }
            }
        }
        return Ok(None);
    }
    // `*p` in address position (`ref *p`, `&*p`): the address IS
    // the pointer itself (`&*p == p`) — evaluate the pointer,
    // never read through it (reading stores pointee byte 0 as a
    // null address into ByReference slots like `Span._reference`,
    // faulting the first use).
    if let Expr::Unary { op, inner, .. } = arg {
        if matches!(
            op,
            crate::models::ast::expression::UnOp::Deref
        ) {
            return Ok(Some(eval(inner, b, ctx)?));
        }
    }
    // `ref MethodCall(...)`: only ref-returning calls are legal in
    // `ref` position (C#), so the evaluated result already IS the
    // address — use it directly, never box it (boxing an address
    // makes a box-holding-address whose later masked reads corrupt
    // to a byte — the casing-table fault). Covers `Unsafe.Add`,
    // `AddByteOffset`, `GetReference`, ref-returning helpers.
    if matches!(
        arg,
        Expr::MethodCall { .. } | Expr::Call { .. }
    ) {
        return Ok(Some(eval(arg, b, ctx)?));
    }
    Ok(None)
}

/// Resolve a `Memmove` address argument to the address itself:
/// byref index/member targets contribute their interior address
/// (`ref arr[i]` passes the element address, but the parser drops
/// `ref`, so plain evaluation yields the element VALUE — copying
/// from address 0/65 instead of the buffer); byref locals
/// contribute their home; already-addresses (pointers, `&x`,
/// intrinsics) pass through untouched via the evaluated value.
fn memmove_addr(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    arg: &Expr,
    evaluated: VReg,
) -> Result<VReg, LowerError> {
    // Call-site `ref` markers unwrap (the home decision below works
    // on the inner location).
    if let Expr::Ident { name, .. } = unmark_ref_arg(arg) {
        if name != "_" && name != "this" {
            if let Some(sym) = b.scopes.lookup(name) {
                if sym.is_byref {
                    return Ok(b.bindings[sym.slot]);
                }
            }
        }
        return Ok(evaluated);
    }
    if let Some(addr) = ref_target_address(b, ctx, arg)? {
        return Ok(addr);
    }
    Ok(evaluated)
}

/// By-reference parameter pattern for a callee identified by
/// declaring class + method name + arity: `Some(flags)` when exactly
/// one method matches (C# forbids overloads differing only in
/// `ref`-vs-`out`, but value-vs-`ref` twins are legal — e.g.
/// `WriteAtOffset` 3/3 — so same-name+arity overloads can still
/// disagree; if several match with different patterns, or none
/// match, return `None` and the call site keeps legacy
/// pass-by-value). This lets call sites box without threading the
/// full overload selection (`selected`) through every emission
/// branch.
pub(super) fn byref_pattern(
    table: &crate::native::semantic::TypeTable,
    declaring_class: &str,
    name: &str,
    arity: usize,
) -> Option<Vec<bool>> {
    let base = name.split('<').next().unwrap_or(name).trim();
    if base.is_empty() {
        return None;
    }
    let mut found: Option<Vec<bool>> = None;
    let mut ambiguous = false;
    for class in &table.classes {
        let is_decl = class.name == declaring_class
            || class.key.path == declaring_class
            || class.name.rsplit('.').next().unwrap_or(&class.name)
                == declaring_class.rsplit('.').next().unwrap_or(declaring_class);
        if !is_decl {
            continue;
        }
        for m in &class.methods {
            if m.name != base || m.param_types.len() != arity {
                continue;
            }
            match &found {
                None => found = Some(m.param_is_byref.clone()),
                Some(f) => {
                    if *f != m.param_is_byref {
                        ambiguous = true;
                    }
                }
            }
        }
    }
    if ambiguous {
        return None;
    }
    found
}

/// Ensure a struct-typed receiver is a usable object: struct fields
/// without initializers (e.g. an embedded `CacheLineSeparated` that
/// is only ever written through) read as null, and member access
/// would fault dereferencing it
/// (`_separated.counts.NumThreadsGoal` in `PortableThreadPool()`).
/// Mirror the static-struct path: null materializes a zeroed box so
/// access proceeds (a store into that box is still dropped — today's
/// property-write fallback semantics — but nothing faults). Non-null
/// receivers pass through untouched, so working struct reads are
/// byte-identical.
pub(super) fn ensure_struct_base(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    base: VReg,
    type_name: &str,
) -> VReg {
    let Some(entry) = struct_entry_for(ctx.table, type_name) else {
        return base;
    };
    let home = b.fresh();
    b.mov(home, base);
    let z = b.const_i64(0);
    let nb = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: crate::native::ir::CmpOp::Eq,
        a: base,
        b: z,
    });
    let box_b = b.make_block();
    let done_b = b.make_block();
    b.terminate(Terminator::Branch {
        cond: Some(nb),
        if_true: box_b,
        if_false: done_b,
    });
    b.switch_to(box_b);
    let boxed = b.alloc_struct_box(ctx, entry, Some(type_name));
    b.mov(home, boxed);
    b.goto(done_b);
    b.switch_to(done_b);
    home
}

/// Load an array element slot as a struct box, materializing the box
/// INTO the slot when it is null: element `i` lives at
/// `base + 16 + i*8` (see `NewArray`); a null slot allocates a
/// zeroed box and stores it back, so the returned handle is always
/// the slot's own box and later field stores persist. Returns `None`
/// when the receiver is not an index into a struct-element array
/// (callers fall back to the ordinary value path). This is what
/// makes `ref Entry e = ref arr[i]` (`ref` has no runtime form —
/// the shape is identical to a plain struct copy) followed by
/// `e.F = v` work: the local binds the slot's box instead of a
/// dropped temp. Plain copies alias the slot too (reads are
/// identical; a written-then-reread copy would observe the
/// array's value instead of a snapshot — accepted: CoreLib only
/// uses this shape for refs).
pub(super) fn index_struct_box(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    object: &Expr,
    index: &Expr,
) -> Result<Option<VReg>, LowerError> {
    let arr_ty = static_type_of(object, b, ctx)?;
    let elem = array_elem_ty(&arr_ty).unwrap_or_default();
    let Some(entry) = struct_entry_for(ctx.table, &elem) else {
        return Ok(None);
    };
    let arrv = eval(object, b, ctx)?;
    let idxv = eval(index, b, ctx)?;
    let eight = b.const_i64(8);
    let sixteen = b.const_i64(16);
    let idx8 = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Mul,
        a: idxv,
        b: eight,
    });
    let off = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: idx8,
        b: sixteen,
    });
    let addr = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: arrv,
        b: off,
    });
    let loaded = b.fresh();
    b.push_raw(Instr::LoadField {
        dst: loaded,
        base: addr,
        off: 0,
    });
    let home = b.fresh();
    b.mov(home, loaded);
    let z = b.const_i64(0);
    let nb = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: crate::native::ir::CmpOp::Eq,
        a: loaded,
        b: z,
    });
    let box_b = b.make_block();
    let done_b = b.make_block();
    b.terminate(Terminator::Branch {
        cond: Some(nb),
        if_true: box_b,
        if_false: done_b,
    });
    b.switch_to(box_b);
    let boxed = b.alloc_struct_box(ctx, entry, Some(&elem));
    b.push_raw(Instr::StoreField {
        base: addr,
        off: 0,
        src: boxed,
    });
    b.mov(home, boxed);
    b.goto(done_b);
    b.switch_to(done_b);
    Ok(Some(home))
}

/// Materialize a `const` field read: constants occupy no storage, so
/// the read folds to a literal (mirrors the qualified-access fold).
/// String constants build exactly like string literals.
pub(super) fn fold_const_field(
    b: &mut Body<'_>,
    f: &crate::native::semantic::FieldInfo,
) -> VReg {
    match f.const_value {
        Some(ConstValue::Int(i)) => b.const_i64(i),
        Some(ConstValue::Float(fv)) => {
            let dst = b.fresh();
            b.push_raw(Instr::FConst {
                dst,
                bits: fv.to_bits(),
            });
            dst
        }
        Some(ConstValue::Str(ref s)) => {
            let (off, len) = b.utf16_literal(s);
            b.build_utf16_string(off, len)
        }
        Some(ConstValue::Null) | None => b.const_i64(0),
    }
}

/// Normalize a value to `bool` (0 or 1): constants fold immediately,
/// anything else becomes `(v != 0)`.
pub(super) fn norm_bool(b: &mut Body<'_>, v: VReg) -> VReg {
    if let Some(&c) = b.const_vals.get(&v.0) {
        return b.emit(Instr::Const {
            dst: VReg(0),
            imm: i64::from(c != 0),
        });
    }
    let zero = b.const_i64(0);
    b.emit(Instr::Cmp {
        dst: VReg(0),
        op: CmpOp::Ne,
        a: v,
        b: zero,
    })
}

/// Whether a (possibly dotted) receiver denotes a TYPE rather than a
/// value: `Outer.Inner` and `System.Foo` are static-style receivers
/// when they resolve to a class entry, while `this.x`, locals, fields
/// and properties stay instance-style even when their TYPE is also a
/// class. Only ever ADDS static classifications the bare-identifier
/// check misses; value-rooted paths are untouched.
fn is_type_receiver(object: &Expr, type_name: &str, b: &Body<'_>, ctx: &Ctx<'_>) -> bool {
    let mut root = object;
    while let Expr::Member { object: o, .. } = root {
        root = o;
    }
    let Expr::Ident { name: root_name, .. } = root else {
        return false;
    };
    if root_name == "this" || root_name == "base" || root_name == "_" {
        return false;
    }
    // A type variable that a generic-inlined body has substituted
    // (`T -> IntNum`) names a TYPE: `T.StaticMember(...)` is static
    // syntax, not a call on a value.
    if ctx.is_type_var(root_name)
        && ctx
            .type_subst
            .is_some_and(|s| s.iter().any(|(v, _)| v == root_name))
    {
        return true;
    }
    if b.scopes.lookup(root_name).is_some() {
        return false;
    }
    if let Some(info) = this_info(ctx, root_name) {
        if info.field(root_name).is_some() || info.property(root_name).is_some() {
            return false;
        }
    }
    if outer_static_field(ctx, root_name).is_some() {
        return false;
    }
    // A dotted receiver is type-style only when the WHOLE path names
    // a type (namespace + class, possibly nested). A path through a
    // field/property (`Holder.SD`, `obj.field`) is a VALUE even when
    // rooted at a type name — the legacy bare class-exists check
    // treated it as static, silently dropping instance calls
    // (fail-open zero: `Holder.SD.GetData()` never emitted the call).
    if let Some(spelled) = dotted_spelling(object) {
        let base = spelled
            .trim_start_matches("global::")
            .split('<')
            .next()
            .unwrap_or(&spelled);
        return ctx.table.class(base).is_some()
            || !ctx.table.alias_targets(base).is_empty();
    }
    // A `using`-alias names no entry of its own, so the bare class-exists
    // check misses it: `NativeType.IsFinite(...)` must still be a
    // static-style receiver, or the call is mistaken for an instance
    // call on a value and fails member resolution.
    ctx.table.class(type_name).is_some()
        || !ctx.table.alias_targets(type_name).is_empty()
}

/// Bind the `field` contextual keyword (C# 14 backing-field) as a
/// zeroed local of the property's type inside an inlined getter.
/// Reads see the default, `field ??= expr` recomputes per read
/// (correct value, no memoization), writes vanish — fail-open
/// semantics matching auto-property placeholders elsewhere.
fn bind_accessor_field(
    b: &mut Body<'_>,
    prop: &crate::native::semantic::PropertyInfo,
) -> Result<(), LowerError> {
    let slot = b
        .scopes
        .declare("field", &prop.type_name, crate::models::Access::Default)
        .map_err(LowerError::UnsupportedConstruct)?;
    while b.bindings.len() <= slot {
        let v = b.fresh();
        b.bindings.push(v);
    }
    let z = b.const_i64(0);
    b.mov(b.bindings[slot], z);
    Ok(())
}

/// Most-derived `Property` declaration for `name` visible from the
/// receiver entry `info`: the receiver's own decl first, then base
/// decls outward (C# override resolution) — NOT file order. The old
/// file-order scan took the BASE's bodyless declaration whenever the
/// base was declared first, hiding the override: `d.P` then read an
/// inherited auto-property backing field (zeros) instead of inlining
/// `=> _w`, and `CanWrite` read backing instead of `=> _writable`.
/// Returns the decl plus its class's primary-ctor params (see the
/// `=>`-body note in `eval_property_getter`).
fn most_derived_prop_decl(
    ctx: &Ctx<'_>,
    info: &crate::native::semantic::ClassInfo,
    name: &str,
) -> Option<(crate::models::ast::PropertyDecl, Vec<String>)> {
    most_derived_prop_accessor(ctx, info, name, false)
        .map(|(decl, params, _)| (decl, params))
}

/// The QUALIFIED path of the class that declares the property — the
/// class whose getter body is inlined, and so the class a member read
/// inside that body belongs to.
fn most_derived_prop_decl_path<'a>(
    ctx: &'a Ctx<'a>,
    info: &crate::native::semantic::ClassInfo,
    name: &str,
) -> Option<&'a str> {
    most_derived_prop_accessor(ctx, info, name, false)
        .and_then(|(_, _, path)| path)
}

/// Most-derived PROPERTY SETTER declaration (mirrors the getter
/// version above, matching `set` block bodies instead of
/// getter bodies). Used by virtual setter dispatch (`AutoFlush =
/// true` must run StreamWriter's flushing setter, not drop).
pub(super) fn most_derived_prop_setter_decl(
    ctx: &Ctx<'_>,
    info: &crate::native::semantic::ClassInfo,
    name: &str,
) -> Option<(crate::models::ast::PropertyDecl, Vec<String>)> {
    most_derived_prop_accessor(ctx, info, name, true)
        .map(|(decl, params, _)| (decl, params))
}

fn most_derived_prop_accessor<'a>(
    ctx: &'a Ctx<'a>,
    info: &crate::native::semantic::ClassInfo,
    name: &str,
    setter: bool,
) -> Option<(crate::models::ast::PropertyDecl, Vec<String>, Option<&'a str>)> {
    let (class_index, member_index, params) = most_derived_prop_accessor_idx(ctx, info, name, setter)?;
    let decl = property_decl_at(ctx, class_index, member_index)?;
    let path = ctx
        .class_paths
        .get(&(ctx.classes[class_index] as *const crate::models::ClassDecl))
        .map(|s| s.as_str());
    Some((decl.clone(), params, path))
}

/// The declaration at a (class, member) index, if it is a property.
fn property_decl_at<'a>(
    ctx: &'a Ctx<'a>,
    class_index: usize,
    member_index: usize,
) -> Option<&'a crate::models::ast::PropertyDecl> {
    match ctx.classes.get(class_index)?.members.get(member_index)? {
        crate::models::ClassMember::Property(decl) => Some(decl),
        _ => None,
    }
}

/// The class and member indices of the declaration that carries a
/// class's property, plus the declaring class's primary-constructor
/// parameters. The walk covers the class's base chain and scans every
/// declaration in the unit at each step, so the answer is cached for
/// the run.
fn most_derived_prop_accessor_idx(
    ctx: &Ctx<'_>,
    info: &crate::native::semantic::ClassInfo,
    name: &str,
    setter: bool,
) -> Option<(usize, usize, Vec<String>)> {
    let key = (info.key.path.clone(), name.to_owned(), setter);
    if let Some(hit) = with_lookup_cache(ctx, |cache| cache.props.get(&key).cloned()) {
        return Some(hit);
    }
    let found = scan_most_derived_prop_accessor(ctx, info, name, setter);
    if let Some((class_index, member_index, params)) = &found {
        with_lookup_cache(ctx, |cache| {
            cache
                .props
                .insert(key, (*class_index, *member_index, params.clone()));
        });
    }
    found
}

/// The walk behind [`most_derived_prop_accessor_idx`].
fn scan_most_derived_prop_accessor(
    ctx: &Ctx<'_>,
    info: &crate::native::semantic::ClassInfo,
    name: &str,
    setter: bool,
) -> Option<(usize, usize, Vec<String>)> {
    let mut queue: Vec<String> = vec![info.name.clone(), info.key.path.clone()];
    let mut seen_curs: Vec<String> = Vec::new();
    // Visited declarations are tracked by position, not by name: two
    // classes can share a simple name across namespaces, and each one
    // carries its own base chain. Keying on the name let the first
    // same-named declaration suppress the rest, so a decoy
    // `System.Runtime.InteropServices.TypeMapping` (empty, no bases)
    // declared before `t.TypeMapping` cut the walk short and hid the
    // inherited `TypeDesc` getter entirely.
    let mut seen_decls: Vec<usize> = Vec::new();
    let mut head: usize = 0;
    while head < queue.len() {
        let cur = queue[head].clone();
        head += 1;
        if seen_curs.iter().any(|s| s == &cur) {
            continue;
        }
        seen_curs.push(cur.clone());
        let mut matches: Vec<(usize, &crate::models::ClassDecl)> = ctx
            .classes
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                c.name == cur
                    || format!("{}.{}", info.key.path, cur) == c.name
                    || c.name.ends_with(&format!(".{cur}"))
            })
            .map(|(i, c)| (i, *c))
            .collect();
        // Prefer the entry's own spelling (same-named siblings in
        // other namespaces never donate the body).
        matches.sort_by_key(|(_, c)| {
            if c.name == info.name || c.name == info.key.path {
                0
            } else {
                1
            }
        });
        for (idx, c) in matches {
            if seen_decls.contains(&idx) {
                continue;
            }
            seen_decls.push(idx);
            for (member_index, m) in c.members.iter().enumerate() {
                if let crate::models::ClassMember::Property(decl) = m
                    && decl.name == name
                    && (!setter || decl.set.is_some())
                {
                    return Some((idx, member_index, c.primary_ctor_params.clone()));
                }
            }
            for b in &c.bases {
                let b = b.split('<').next().unwrap_or(b).trim().to_owned();
                if !b.is_empty() {
                    queue.push(b);
                }
            }
        }
    }
    None
}

/// Whether a type spelling denotes a value type (for
/// `typeof(X).IsValueType` over literal spellings): open type
/// variables count as value types (optimistic fail-open), as do
/// primitives and table-known structs/enums; everything else is
/// not (legacy false).
fn is_value_type_spelling(ctx: &Ctx<'_>, spelling: &str) -> bool {
    if ctx.is_type_var(spelling) {
        return true;
    }
    let base = base_type_name(spelling);
    const PRIMITIVE_VALUE: &[&str] = &[
        "sbyte", "byte", "short", "ushort", "int", "uint", "long", "ulong", "nint", "nuint",
        "char", "float", "double", "decimal", "bool", "SByte", "Byte", "Int16", "UInt16",
        "Int32", "UInt32", "Int64", "UInt64", "IntPtr", "UIntPtr", "Char", "Single", "Double",
        "Decimal", "Boolean", "Void", "void",
    ];
    if PRIMITIVE_VALUE.contains(&base.as_str()) {
        return true;
    }
    // `int?` (Nullable<int>) is a value type; other `?`s are not.
    if base == "Nullable" {
        return true;
    }
    if let Some(entry) = ctx.table.class(&base).or_else(|| ctx.table.class(spelling)) {
        if entry.is_struct || entry.is_enum {
            return true;
        }
    }
    false
}

/// Whether a STRICT subclass of the receiver entry `info` carries
/// an effective explicit getter for `name` (self excluded). When
/// true the read must dispatch virtually — inlining the base body
/// would serve it to derived instances (`CSBase.CanWrite` inlined
/// `=> false` for `CSDerived` instances). Sealed/single hierarchies
/// have none (static inline stays exact).
/// NOTE: `new`-hiding (non-virtual base, shadowing derived) also
/// matches here and misdispatches to the derived body (C# would use
/// the base). Override dwarfs hiding in practice; documented gap.
fn strict_subclass_has_explicit(
    ctx: &Ctx<'_>,
    info: &crate::native::semantic::ClassInfo,
    name: &str,
) -> bool {
    strict_subclass_has_accessor(ctx, info, name, false)
}

/// Whether a STRICT subclass carries an effective explicit SETTER
/// (mirrors the getter version; drives virtual setter dispatch).
pub(super) fn strict_subclass_has_explicit_setter(
    ctx: &Ctx<'_>,
    info: &crate::native::semantic::ClassInfo,
    name: &str,
) -> bool {
    strict_subclass_has_accessor(ctx, info, name, true)
}


/// Whether `sub` OVERRIDES the property `name`, so its implementation
/// serves a base-typed receiver.
///
/// C# distinguishes three shapes: `override` replaces the base member,
/// `new` HIDES it (the base's implementation still serves a base-typed
/// receiver), and a plain redeclaration in a class that has no such
/// member is an implicit interface implementation. Treating every
/// same-named declaration as an override made a hiding property serve
/// the derived body — the base's answer was never used.
fn prop_overrides_base(
    ctx: &Ctx<'_>,
    sub: &crate::native::semantic::ClassInfo,
    base: &crate::native::semantic::ClassInfo,
    name: &str,
) -> bool {
    let Some(decl) = construction_decl_arity(ctx, &sub.key.path, sub.key.arity)
        .or_else(|| construction_decl_arity(ctx, &sub.name, sub.key.arity))
    else {
        return false;
    };
    let Some(prop) = decl.members.iter().find_map(|m| match m {
        crate::models::ClassMember::Property(p) if p.name == name => Some(p),
        _ => None,
    }) else {
        return false;
    };
    if prop.is_hiding {
        return false;
    }
    if prop.is_override {
        return true;
    }
    // An implicit interface implementation: the base member has no
    // implementation of its own, so this body IS the implementation.
    let base_is_interface = ctx.table.interfaces.iter().any(|i| {
        i.name == base.name || i.name == base.key.path
    });
    base_is_interface && (prop.get_expr.is_some() || prop.get.is_some())
}

fn strict_subclass_has_accessor(
    ctx: &Ctx<'_>,
    info: &crate::native::semantic::ClassInfo,
    name: &str,
    setter: bool,
) -> bool {
    for sub_name in virtual_subclasses(ctx, info) {
        let Some(sub) = ctx.table.class(&sub_name) else {
            continue;
        };
        if sub.key.path == info.key.path {
            continue;
        }
        if !prop_overrides_base(ctx, sub, info, name) {
            continue;
        }
        if setter {
            if let Some((decl, _)) = most_derived_prop_setter_decl(ctx, sub, name)
                && decl.set.is_some()
            {
                return true;
            }
        } else if let Some((decl, _)) = most_derived_prop_decl(ctx, sub, name)
            && (decl.get_expr.is_some() || decl.get.is_some())
        {
            return true;
        }
    }
    false
}
/// Declaration `new X()` would construct for spelling `class_name`
/// (exact, simple, or `using`-alias target — mirrors the `new`
/// path): virtual dispatch must resolve the SAME declaration, or
/// rows diverge on duplicate/nested names (`vtable_row_for`
/// name-first-match picks a different decl than `new`'s call-site
/// spelling, and every arm misses at runtime).
fn construction_decl<'a>(
    ctx: &'a Ctx<'a>,
    class_name: &str,
) -> Option<&'a crate::models::ClassDecl> {
    let arity = crate::native::semantic::split_arity(class_name).1;
    construction_decl_arity(ctx, class_name, arity)
}

/// Arity-explicit variant: entry-aware callers (virtual dispatch
/// deforms) pass the receiver entry's own arity, since entry key
/// paths carry no generic arguments (`Task` for a `Task<int>`
/// receiver must resolve the arity-1 definition's row, or every arm
/// misses at runtime and reads fall to zero).
fn construction_decl_arity<'a>(
    ctx: &'a Ctx<'a>,
    class_name: &str,
    arity: usize,
) -> Option<&'a crate::models::ClassDecl> {
    let index = construction_decl_index(ctx, class_name, arity)?;
    ctx.classes.get(index).copied()
}

/// The index of the declaration a class NAME resolves to (the
/// construction and the virtual dispatch both ask). The answer
/// depends only on the unit, so it is cached for the run.
fn construction_decl_index(
    ctx: &Ctx<'_>,
    class_name: &str,
    arity: usize,
) -> Option<usize> {
    let key = (class_name.to_owned(), arity);
    if let Some(index) = with_lookup_cache(ctx, |cache| cache.decls.get(&key).copied()) {
        return Some(index);
    }
    let found = scan_construction_decl(ctx, class_name, arity);
    if let Some(index) = found {
        with_lookup_cache(ctx, |cache| {
            cache.decls.insert(key, index);
        });
    }
    found
}

/// The scan behind [`construction_decl_index`].
fn scan_construction_decl(
    ctx: &Ctx<'_>,
    class_name: &str,
    arity: usize,
) -> Option<usize> {
    let alias_simple: Option<String> = ctx.table.alias_target(class_name).map(|t| {
        let base = t.split('<').next().unwrap_or(t);
        base.rsplit('.').next().unwrap_or(base).to_owned()
    });
    let simple_name = class_name
        .rsplit('.')
        .next()
        .unwrap_or(class_name)
        .split('<')
        .next()
        .unwrap_or(class_name);
    // Generic-instantiation spellings (`Task<VoidTaskResult>`) name
    // no declaration verbatim: resolve to the generic definition
    // (arity-matching decl) so rows agree with `new`'s call-site
    // spelling. Bare spellings keep exact legacy order unless the
    // caller passes an explicit arity (entry-aware dispatch).
    let (def_base, _) =
        crate::native::semantic::split_arity(class_name);
    let def_simple = def_base
        .rsplit('.')
        .next()
        .unwrap_or(def_base.as_str());
    let mut arity_match: Option<usize> = None;
    let mut out: Option<usize> = None;
    for (i, c) in ctx.classes.iter().enumerate() {
        let c: &crate::models::ClassDecl = c;
        if c.name == *class_name
            || c.name == simple_name
            || alias_simple.as_deref() == Some(c.name.as_str())
        {
            return Some(i);
        }
        if arity > 0
            && (c.name == def_simple || c.name == def_base)
            && c.generic_params.len() == arity
            && arity_match.is_none()
        {
            arity_match = Some(i);
        }
        if out.is_none()
            && (c.name == def_simple || c.name == def_base)
        {
            out = Some(i);
        }
    }
    arity_match.or(out)
}

/// The per-run lookup caches. Three questions cost a walk of every
/// declaration in the unit, and the walks dominate the compile time
/// at CoreLib scale (one `Stream` method took 14 seconds):
///
/// - which declaration a class NAME resolves to (the construction and
///   the virtual dispatch both ask),
/// - a declaration's vtable ROW (every `new` asks),
/// - which declaration carries a class's PROPERTY (every property
///   read asks).
///
/// The answers depend only on the unit, so they are cached for the
/// run and dropped when it ends.
struct LookupCache {
    /// The class table the caches were filled for.
    table: usize,
    /// (class name, arity) -> the declaration's index in the unit.
    decls: std::collections::HashMap<(String, usize), usize>,
    /// declaration address -> its construction row.
    rows: std::collections::HashMap<usize, u32>,
    /// (class path, property name, setter) -> (class index, member
    /// index, the declaring class's primary-constructor parameters).
    props: std::collections::HashMap<(String, String, bool), (usize, usize, Vec<String>)>,
}

thread_local! {
    static LOOKUP_CACHE: std::cell::RefCell<Option<LookupCache>> =
        const { std::cell::RefCell::new(None) };
}

/// Drop the lookup caches: called once per lowering entry, so a new
/// unit never reads a previous run's answers.
pub(crate) fn clear_lookup_cache() {
    LOOKUP_CACHE.with(|c| *c.borrow_mut() = None);
}

/// Run `f` with the caches for `ctx`'s class table, filling them as
/// needed.
fn with_lookup_cache<R>(
    ctx: &Ctx<'_>,
    f: impl FnOnce(&mut LookupCache) -> R,
) -> R {
    let table = ctx.table as *const _ as usize;
    LOOKUP_CACHE.with(|cell| {
        let mut slot = cell.borrow_mut();
        let fresh = !matches!(slot.as_ref(), Some(c) if c.table == table);
        if fresh {
            *slot = Some(LookupCache {
                table,
                decls: std::collections::HashMap::new(),
                rows: std::collections::HashMap::new(),
                props: std::collections::HashMap::new(),
            });
        }
        let cache = slot.as_mut().expect("the cache was just installed");
        f(cache)
    })
}

/// Vtable row for a declaration: position among table-backed
/// declarations in emission order (mirrors `Program` row
/// construction). Shared by `new` and virtual dispatch — one
/// definition, no divergence.
fn construction_row(ctx: &Ctx<'_>, decl: &crate::models::ClassDecl) -> u32 {
    let target = (decl as *const crate::models::ClassDecl) as usize;
    let cached = with_lookup_cache(ctx, |cache| cache.rows.get(&target).copied());
    if let Some(row) = cached {
        return row;
    }
    let row = ctx
        .classes
        .iter()
        .take_while(|c| {
            (**c as *const crate::models::ClassDecl) as usize != target
        })
        .filter(|c| ctx.table.class(&c.name).is_some())
        .count() as u32;
    with_lookup_cache(ctx, |cache| {
        cache.rows.insert(target, row);
    });
    row
}

thread_local! {
    /// Count of property reads during lowering (diagnostic).
    pub(crate) static PROP_READ_COUNT: std::cell::RefCell<usize> = const { std::cell::RefCell::new(0) };
}

pub(super) fn virtual_subclasses(ctx: &Ctx<'_>, info: &crate::native::semantic::ClassInfo) -> Vec<String> {
    // The walk is O(classes^2) and the virtual property dispatch asks
    // for it on EVERY property read: at CoreLib scale that was
    // minutes of compile time (one `Stream` method took 24 seconds).
    // The result depends only on the class table, so it is cached per
    // class for the run.
    let key = info.key.path.clone();
    SUBCLASS_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let table = ctx.table as *const _ as usize;
        let fresh = !matches!(cache.as_ref(), Some((t, _)) if *t == table);
        if fresh {
            *cache = Some((table, std::collections::HashMap::new()));
        }
        let Some((_, map)) = cache.as_mut() else {
            return Vec::new();
        };
        if let Some(hit) = map.get(&key) {
            return hit.clone();
        }
        let subs = compute_virtual_subclasses(ctx, info);
        map.insert(key, subs.clone());
        subs
    })
}

/// Drop the cached subclass sets: called once per lowering entry, so
/// a new class table never reads a previous run's answer.
pub(crate) fn clear_subclass_cache() {
    SUBCLASS_CACHE.with(|c| *c.borrow_mut() = None);
}

thread_local! {
    /// Subclass sets per class path, for the run's class table.
    static SUBCLASS_CACHE: std::cell::RefCell<
        Option<(usize, std::collections::HashMap<String, Vec<String>>)>,
    > = std::cell::RefCell::new(None);
}

/// The subclasses of a class: its own declarations plus every
/// declaration that derives from them.
fn compute_virtual_subclasses(
    ctx: &Ctx<'_>,
    info: &crate::native::semantic::ClassInfo,
) -> Vec<String> {
    let matches_info = |n: &str| {
        n == info.name
            || n == info.key.path
            || base_type_name(n) == base_type_name(&info.name)
    };
    let mut subs: Vec<String> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    // Seed: the receiver's own decl(s).
    for c in ctx.classes.iter() {
        if matches_info(&c.name) && !seen.iter().any(|s| s == &c.name) {
            seen.push(c.name.clone());
            subs.push(c.name.clone());
        }
    }
    // Expand: decls whose bases hit the set.
    loop {
        let mut added = false;
        for c in ctx.classes.iter() {
            if seen.iter().any(|s| s == &c.name) {
                continue;
            }
            let under = c.bases.iter().any(|b| {
                let b = b.split('<').next().unwrap_or(b).trim();
                subs.iter().any(|s| b == s || base_type_name(b) == base_type_name(s))
                    || matches_info(b)
            });
            if under {
                seen.push(c.name.clone());
                subs.push(c.name.clone());
                added = true;
            }
        }
        if !added {
            break;
        }
    }
    subs
}

/// Virtual getter dispatch for `recv.Prop` when the static type has
/// no explicit getter (abstract/auto base): the runtime type's
/// override serves, like virtual methods (which have vtable slots;
/// property getters don't — hence this closed-world dispatch).
/// Every subclass with an effective explicit body gets a vtable-row
/// arm inlining that body with the receiver; a single distinct body
/// devirtualizes directly with no chain. `None` when no subclass
/// implements the property (placeholder path keeps legacy zeros).
/// Bodies fail soft per arm (zero) like the static path.
/// NOTE: the receiver re-evaluates here (and once more in the static
/// attempt above) — same as the pre-existing double evaluation at
/// this site; side-effecting receivers at virtual sites repeat.
/// Single evaluation is future work.
fn eval_virtual_getter(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    info: &crate::native::semantic::ClassInfo,
    name: &str,
    object: Option<&Expr>,
) -> Result<Option<VReg>, LowerError> {
    // The dispatch frame is NOT a recursion marker. A getter may read
    // the same property on ANOTHER object (`SyncStream.Position` reads
    // the wrapped stream's position, and the wrapped stream is not a
    // SyncStream). Marking the BASE class as "being inlined" refused
    // that read, so the property had no value at all. Recursion is per
    // CLASS, and `eval_property_getter` guards exactly that when it
    // inlines a getter (the same class and property again is the
    // cycle).
    eval_virtual_getter_inner(b, ctx, info, name, object)
}

/// Body of virtual dispatch (guard handling above).
fn eval_virtual_getter_inner(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    info: &crate::native::semantic::ClassInfo,
    name: &str,
    object: Option<&Expr>,
) -> Result<Option<VReg>, LowerError> {
    // Arms: the subclasses that OVERRIDE the property. A declaration
    // that merely repeats the name HIDES the base member, so it is not
    // an arm.
    let mut arms: Vec<(&crate::native::semantic::ClassInfo, (u32, u32))> = Vec::new();
    for sub_name in virtual_subclasses(ctx, info) {
        let Some(sub) = ctx.table.class(&sub_name) else {
            continue;
        };
        if !prop_overrides_base(ctx, sub, info, name) {
            continue;
        }
        if let Some((decl, _)) = most_derived_prop_decl(ctx, sub, name)
            && (decl.get_expr.is_some() || decl.get.is_some())
        {
            let key = (decl.span.start, decl.span.end);
            arms.push((sub, key));
        }
    }
    let recv = match object {
        Some(o) => eval(o, b, ctx)?,
        None => VReg(0),
    };
    let saved_this = b.save_this();
    b.mov(VReg(0), recv);
    if arms.is_empty() {
        // No override in this unit: the BASE's implementation serves
        // (a hiding declaration does not replace it).
        let val = match eval_property_getter(b, ctx, info, &info.name, name, None)? {
            Some(v) => v,
            None => {
                // No implementation anywhere in this unit: the member
                // is unimplemented, which a well-typed program cannot
                // read. Report it rather than answer a fabricated zero.
                b.restore_this(saved_this);
                return Ok(None);
            }
        };
        b.restore_this(saved_this);
        return Ok(Some(val));
    }
    // Row dispatch: one arm per overriding class, join value via home
    // vreg (same shape as ternary lowering). Receivers carry their
    // vtable ADDRESS at [+0] (not a row index), so arms compare
    // against per-class vtable addresses. Rows come from
    // `construction_decl`/`construction_row` — the exact resolution
    // `new` uses — never `vtable_row_for` (name first-match diverges
    // on duplicate/nested names and every arm misses).
    let rowv = b.fresh();
    b.push_raw(Instr::LoadField {
        dst: rowv,
        base: recv,
        off: 0,
    });
    let home = b.fresh();
    let end_b = b.make_block();
    for (sub, _) in arms {
        // A getter that reads the same property of the SAME class is a
        // cycle; that arm cannot be inlined here.
        if b
            .prop_inline_stack
            .iter()
            .any(|(c, p)| c == &sub.key.path && p == name)
        {
            continue;
        }
        let Some(decl) = construction_decl_arity(ctx, &sub.key.path, sub.key.arity)
            .or_else(|| construction_decl_arity(ctx, &sub.name, sub.key.arity))
        else {
            // No local declaration (cross-image): skip the arm —
            // a wrong row could match another class.
            continue;
        };
        let row = construction_row(ctx, decl);
        let vaddr = b.fresh();
        b.push_raw(Instr::VTableAddr { dst: vaddr, row });
        let cond = b.emit(Instr::Cmp {
            dst: VReg(0),
            op: CmpOp::Eq,
            a: rowv,
            b: vaddr,
        });
        let hit_b = b.make_block();
        let next_b = b.make_block();
        b.terminate(Terminator::Branch {
            cond: Some(cond),
            if_true: hit_b,
            if_false: next_b,
        });
        b.switch_to(hit_b);
        // The arm's own getter guard (in `eval_property_getter`) catches
        // a getter that reads the same property on the same class.
        match eval_property_getter(b, ctx, sub, &sub.name, name, None)? {
            Some(v) => b.mov(home, v),
            None => {
                let z = b.const_i64(0);
                b.mov(home, z);
            }
        }
        if !b.is_sealed() {
            b.goto(end_b);
        }
        b.switch_to(next_b);
    }
    // No override matched: the BASE's implementation serves. Answering
    // zero here invented a value for a property that legitimately holds
    // something else — and a `new`-hiding declaration lands here, where
    // C# uses the base's member.
    match eval_property_getter(b, ctx, info, &info.name, name, None)? {
        Some(v) => b.mov(home, v),
        None => {
            // The base has no implementation and no override matched:
            // the receiver's runtime type does not implement the
            // member, which a well-typed program cannot do. Trap
            // rather than answer a fabricated value.
            b.raw(vec![0xCC]);
            let z = b.const_i64(0);
            b.mov(home, z);
        }
    }
    if !b.is_sealed() {
        b.goto(end_b);
    }
    b.switch_to(end_b);
    b.restore_this(saved_this);
    Ok(Some(home))
}

/// Body of virtual SETTER dispatch (mirrors the getter deform, minus
/// the return value). Shared by explicit-receiver and implicit-`this`
/// writes; `value` was evaluated once by the caller.
pub(super) fn eval_virtual_setter_inner(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    info: &crate::native::semantic::ClassInfo,
    name: &str,
    object: Option<&Expr>,
    value: VReg,
) -> Result<bool, LowerError> {
    // Effective explicit setter per subclass entry.
    let mut arms: Vec<&crate::native::semantic::ClassInfo> = Vec::new();
    for sub_name in virtual_subclasses(ctx, info) {
        let Some(sub) = ctx.table.class(&sub_name) else {
            continue;
        };
        if !prop_overrides_base(ctx, sub, info, name) {
            continue;
        }
        if most_derived_prop_setter_decl(ctx, sub, name)
            .is_some_and(|(decl, _)| decl.set.is_some())
        {
            arms.push(sub);
        }
    }
    // Receiver into VReg(0) (the inline contract); restored after.
    let recv = match object {
        Some(o) => eval(o, b, ctx)?,
        None => VReg(0),
    };
    let saved_this = b.save_this();
    b.mov(VReg(0), recv);
    if arms.is_empty() {
        // No override in this unit: the BASE's setter serves. Dropping
        // the write (the old behavior) lost it silently.
        let done = eval_property_setter(b, ctx, info, &info.name, name, None, value);
        b.restore_this(saved_this);
        return Ok(done);
    }
    // Row dispatch: one arm per overriding class; when none matches,
    // the BASE's setter serves. Rows come from `construction_decl` /
    // `construction_row`, exactly like the getter path.
    let rowv = b.fresh();
    b.push_raw(Instr::LoadField {
        dst: rowv,
        base: recv,
        off: 0,
    });
    let end_b = b.make_block();
    for sub in arms {
        // A setter that writes the same property of the SAME class is a
        // cycle; that arm cannot be inlined here.
        if b
            .prop_inline_stack
            .iter()
            .any(|(c, p)| c == &sub.key.path && p == name)
        {
            continue;
        }
        let Some(decl) = construction_decl_arity(ctx, &sub.key.path, sub.key.arity)
            .or_else(|| construction_decl_arity(ctx, &sub.name, sub.key.arity))
        else {
            continue;
        };
        let row = construction_row(ctx, decl);
        let vaddr = b.fresh();
        b.push_raw(Instr::VTableAddr { dst: vaddr, row });
        let cond = b.emit(Instr::Cmp {
            dst: VReg(0),
            op: CmpOp::Eq,
            a: rowv,
            b: vaddr,
        });
        let hit_b = b.make_block();
        let next_b = b.make_block();
        b.terminate(Terminator::Branch {
            cond: Some(cond),
            if_true: hit_b,
            if_false: next_b,
        });
        b.switch_to(hit_b);
        let _ = eval_property_setter(b, ctx, sub, &sub.name, name, None, value);
        if !b.is_sealed() {
            b.goto(end_b);
        }
        b.switch_to(next_b);
    }
    // No override matched: the BASE's setter serves.
    let _ = eval_property_setter(b, ctx, info, &info.name, name, None, value);
    if !b.is_sealed() {
        b.goto(end_b);
    }
    b.switch_to(end_b);
    b.restore_this(saved_this);
    Ok(true)
}

/// Virtual setter dispatch with outer cycle guard (mirrors
/// `eval_virtual_getter`). Returns true when an arm inlined.
pub(super) fn eval_virtual_setter(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    info: &crate::native::semantic::ClassInfo,
    name: &str,
    object: Option<&Expr>,
    value: VReg,
) -> Result<bool, LowerError> {
    if b.prop_inline_stack
        .iter()
        .any(|(c, p)| c == &info.key.path && p == name)
    {
        return Ok(false);
    }
    b.prop_inline_stack
        .push((info.key.path.clone(), name.to_owned()));
    let result = eval_virtual_setter_inner(b, ctx, info, name, object, value);
    b.prop_inline_stack.pop();
    result
}
/// with the receiver bound as `this`. Returns `Ok(None)` when the
/// property has no inlineable body (auto-properties, statement
/// getters), letting the caller fall back to the placeholder.
#[allow(clippy::too_many_arguments)]
/// Getter body form for inline evaluation: `=> expr` evaluates
/// directly; a block inlines statement-by-statement with return
/// capture.
enum GetterBody {
    Expr(Expr),
    Block(Block),
}

/// `object` is `Some` for `recv.Prop` (evaluated for side effects
/// and used as the receiver) and `None` for bare `Prop` (implicit
/// receiver: static properties need none, instance ones use `VReg(0)`).
fn eval_property_getter(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    info: &crate::native::semantic::ClassInfo,
    type_name: &str,
    name: &str,
    object: Option<&Expr>,
) -> Result<Option<VReg>, LowerError> {
    let prop = match info.property(name) {
        Some(p) => p,
        None => {
            return Ok(None);
        }
    };
    // A property getter that reads itself (`int P => P`) would recurse
    // forever while lowering. A chain that has grown far past any real
    // inheritance depth is the same thing. Both used to break the cycle
    // by answering 0, which reads at run time like a property that
    // legitimately holds zero; report instead.
    if b.prop_inline_stack.len() > 25 {
        let chain: Vec<String> = b
            .prop_inline_stack
            .iter()
            .map(|(c, p)| format!("{c}.{p}"))
            .collect();
        return Err(LowerError::UnsupportedConstruct(crate::tf!(
            "rubyc.native.ir.lower_expr.property_inline_cycle",
            info.key.path.clone(),
            name.to_string(),
            chain.join(" -> ")
        )));
    }
    if b.prop_inline_stack.iter().any(|(c, p)| c == &info.key.path && p == name) {
        // `[Intrinsic]` self-referential property: the JIT replaces
        // the body, so the managed getter never runs. RubyC has no
        // SIMD intrinsics, so such a property (today only
        // `IsHardwareAccelerated`) is always false; answer the target
        // capability instead of inlining (which would recurse
        // forever). Keyed on the `[Intrinsic]` attribute, not on a
        // member name.
        let is_intrinsic = info
            .property(name)
            .map(|pr| pr.is_intrinsic)
            .unwrap_or(false);
        if is_intrinsic {
            let v = b.const_i64(0);
            return Ok(Some(v));
        }
        // Cross-instance recursion: the getter recurses through a
        // DIFFERENT receiver (`CultureInfo.CompareInfo` reads
        // `GetCultureInfo(_name).CompareInfo`; `AggregateException.
        // Message` reads `_innerExceptions[i].Message`), so the
        // instance-insensitive cycle check is a false positive. The
        // body terminates (cache / distinct instances), so allow
        // re-entry; the depth check above is the safety net.
        let cross_instance = (name == "CompareInfo"
            && info.key.path.contains("CultureInfo"))
            || (name == "Message"
                && info.key.path.contains("AggregateException"));
        if !cross_instance {
            let chain: Vec<String> = b
                .prop_inline_stack
                .iter()
                .map(|(c, p)| format!("{c}.{p}"))
                .collect();
            return Err(LowerError::UnsupportedConstruct(crate::tf!(
                "rubyc.native.ir.lower_expr.property_inline_cycle",
                info.key.path.clone(),
                name.to_string(),
                chain.join(" -> ")
            )));
        }
    }
    b.prop_inline_stack
        .push((info.key.path.clone(), name.to_owned()));
    ctx.check_access(true, type_name, name)?;
    // Getter body from the most-derived declaration (see
    // `most_derived_prop_decl`): the override's body wins over the
    // base's, regardless of file order. Primary-constructor params
    // ride along: `=>` bodies may name them
    // (`CustomAttributeType => type`), and member bodies see them as
    // zero placeholders (see the call-site binding in `ir/mod.rs`).
    let mut getter: Option<GetterBody> = None;
    let mut primary_params: Vec<String> = Vec::new();
    if let Some((decl, params)) = most_derived_prop_decl(ctx, info, name) {
        primary_params = params;
        if let Some(e) = decl.get_expr.clone() {
            getter = Some(GetterBody::Expr(e));
        } else if let Some(block) = decl.get.clone() {
            // Block bodies inline whole (return capture below);
            // `LocalFunction` declarations inside ride along for
            // call resolution (emitted separately by the pre-scan).
            getter = Some(GetterBody::Block(block));
        }
    }
    let Some(body) = getter else {
        // No inlineable body (auto/abstract): pop the cycle guard
        // pushed above — leaking it poisons later virtual dispatch
        // for the same property (guard trips on the stale entry).
        b.prop_inline_stack.pop();
        return Ok(None);
    };
    // A getter body that will not lower is reported, not rolled back
    // into a zero. The rollback used to swallow the real failure and
    // hand the caller a placeholder value that ran like a property
    // holding default(T).
    let result = eval_property_body(
        b,
        ctx,
        info,
        prop,
        &primary_params[..],
        type_name,
        name,
        object,
        &body,
    );
    b.prop_inline_stack.pop();
    result.map(Some)
}


/// Bind a generic class's type parameters to the type arguments of the
/// spelling that names it: the getter of `Scalar<int>.AllBitsSet` runs
/// with `T` = `int`. The spelling's arguments may name the reader's
/// own type variables, so the reader's substitution applies to them
/// first — a specialized generic method reads `Scalar<T>`, and its `T`
/// is already bound.
fn class_type_subst(
    ctx: &Ctx<'_>,
    info: &crate::native::semantic::ClassInfo,
    spelling: &str,
) -> Vec<(String, String)> {
    if info.generic_params.is_empty() {
        return Vec::new();
    }
    let args = explicit_type_args(spelling);
    if args.is_empty() {
        return Vec::new();
    }
    let args: Vec<String> = args
        .iter()
        .map(|a| {
            ctx.type_subst
                .and_then(|sub| super::types::subst_type_var(a, sub))
                .unwrap_or_else(|| a.clone())
        })
        .collect();
    info.generic_params.iter().cloned().zip(args).collect()
}

/// The table entry of the class that DECLARES a property. A property
/// declared in the receiver's own class uses the receiver's entry: a
/// same-named non-generic class would otherwise shadow the generic
/// declaration (`Scalar` beside `Scalar<T>`), and the getter would run
/// with no type parameters at all.
fn declaring_entry<'t>(
    ctx: &Ctx<'t>,
    info: &'t crate::native::semantic::ClassInfo,
    prop: &crate::native::semantic::PropertyInfo,
) -> Option<&'t crate::native::semantic::ClassInfo> {
    if info.name == prop.declaring_class {
        return Some(info);
    }
    ctx.table.class(&prop.declaring_class)
}

/// Lower one getter body (expression or block) with receiver,
/// `field`, and primary-param bindings. Separated for fail-soft
/// rollback above.
#[allow(clippy::too_many_arguments)]
fn eval_property_body(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    info: &crate::native::semantic::ClassInfo,
    prop: &crate::native::semantic::PropertyInfo,
    primary_params: &[String],
    type_name: &str,
    name: &str,
    object: Option<&Expr>,
    body: &GetterBody,
) -> Result<VReg, LowerError> {
    // The getter is the declaring class's own code: member lookups
    // inside it resolve against the declarer, like field
    // initializers do. Use the PROPERTY's declaring entry (not the
    // receiver's): an inherited body reading a private base field
    // (`ConsoleStream.CanWrite` reading `_canWrite`, inlined for a
    // `UnixConsoleStream` receiver) is legal base code, but checked
    // as the derived receiver it falsely denies (E0017) and the arm
    // degrades to placeholder zero. Fall back to the receiver entry
    // when the declarer doesn't resolve (same entry in practice).
    // The class SPELLING stays simple (declaration names are
    // simple): qualified paths miss every decl-name lookup, which
    // dropped private helper calls (`EnsureNotClosed`) inside
    // inlined bodies as unknown.
    let declarer = declaring_entry(ctx, info, prop);
    // The getter is the declaring class's own code, so the class's type
    // parameters are bound by the RECEIVER's spelling: reading
    // `Scalar<int>.AllBitsSet` runs the getter with its `T` = `int`.
    // The spelling may itself name the reader's type variables (a
    // specialized generic method reads `Scalar<T>`), so the reader's
    // own substitution applies to the arguments first.
    let class_subst = class_type_subst(ctx, info, type_name);
    let prop_ctx = Ctx {
        table: ctx.table,
        class_paths: ctx.class_paths,
        // The class this context switches to is the property's OWNER,
        // named by its qualified key path: a bare name cannot be
        // matched against the owner's path (the access rule compares
        // two paths), and a nested type's bare name is ambiguous
        // (Utf8 and Utf16 both declare
        // `TryWriteInterpolatedStringHandler`).
        current_class: &prop.declaring_class,
        current_info: declarer.or(Some(info)),
        // The path must name the class that DECLARED the property
        // (the getter's own class, which may be a base of the
        // receiver's): a bare class name cannot be matched against
        // the declaring path (the access rule compares two paths), and
        // a nested type's bare name is ambiguous (Utf8 and Utf16 both
        // declare `TryWriteInterpolatedStringHandler`).
        current_path: most_derived_prop_decl_path(ctx, info, name)
            .or(declarer.map(|i| i.key.path.as_str()))
            .or(ctx.current_path),
        classes: ctx.classes,
        import_index: ctx.import_index,
        dep_imports: ctx.dep_imports,
        host_import_count: ctx.host_import_count,
        method_vars: ctx.method_vars,
                type_subst: (!class_subst.is_empty()).then_some(&class_subst),
        local_fns: ctx.local_fns,
        current_method: Some(name.to_owned()),
        current_ret: Some(prop.type_name.clone()),
        static_index: ctx.static_index,
        defines: ctx.defines,
        lambda_deferred: None,
                    lambda_base: 0,
    };
    if prop.is_static {
        // Static property: no receiver, evaluate for side effects.
        if let Some(o) = object {
            let _ = eval(o, b, ctx)?;
        }
        b.scope_enter();
        bind_accessor_field(b, prop)?;
        for pname in primary_params {
            b.bind_zero_placeholder(pname);
        }
        let val = match &body {
            GetterBody::Expr(e) => {
                let v = eval(e, b, &prop_ctx)?;
                // Implicit return conversion (same as `Stmt::Return`):
                // arrays/strings pack into span-typed properties
                // (`ReadOnlySpan<uint> P => […]`).
                let val_ty = static_type_of(e, b, &prop_ctx)
                    .unwrap_or_else(|_| "object".to_owned());
                coerce_span_return(b, &prop_ctx, v, &val_ty, &prop.type_name)?
            }
            GetterBody::Block(block) => {
                eval_block_inline(b, &prop_ctx, block)?
            }
        };
        b.scope_exit();
        return Ok(val);
    }
    let base = match object {
        Some(o) => eval(o, b, ctx)?,
        // Implicit receiver (`Prop` inside its own class).
        None => VReg(0),
    };
    // Bare field reads and same-class calls hardcode VReg(0) as
    // `this` (see the prologue contract and `calls.rs`), ignoring
    // scope bindings — so point VReg(0) at the receiver for the
    // getter body, restoring afterwards (nesting-safe). The scope
    // binding below covers explicit-`this` paths.
    let saved_this = b.save_this();
    b.mov(VReg(0), base);
    b.scope_enter();
    let slot = b
        .scopes
        .declare("this", type_name, crate::models::Access::Default)
        .map_err(LowerError::UnsupportedConstruct)?;
    while b.bindings.len() <= slot {
        let v = b.fresh();
        b.bindings.push(v);
    }
    b.mov(b.bindings[slot], base);
    bind_accessor_field(b, prop)?;
    for pname in primary_params {
        b.bind_zero_placeholder(pname);
    }
    let val = match &body {
        GetterBody::Expr(e) => {
            let v = eval(e, b, &prop_ctx)?;
            // Implicit return conversion (same as `Stmt::Return`):
            // arrays/strings pack into span-typed properties
            // (`ReadOnlySpan<uint> P => […]`).
            let val_ty = static_type_of(e, b, &prop_ctx)
                .unwrap_or_else(|_| "object".to_owned());
            coerce_span_return(b, &prop_ctx, v, &val_ty, &prop.type_name)?
        }
        GetterBody::Block(block) => eval_block_inline(b, &prop_ctx, block)?,
    };
    b.scope_exit();
    b.restore_this(saved_this);
    Ok(val)
}

/// Lower one setter body inline with receiver and `value` bindings
/// (mirrors the getter path, minus the return value). Returns true
/// on success; fail-soft false (caller falls back to dropping, the
/// legacy behavior) on cycle-guard trip, missing body, or lowering
/// error (with rollback). Guards share the property-inline stack
/// (same key space as getters; cross get/set recursion degrades to
/// placeholder rather than hanging).
#[allow(clippy::too_many_arguments)]
/// Whether the property `name` declared somewhere on `class_name`'s
/// inheritance chain has a setter. `None` when no declaring class is in
/// the table (generic/partial spellings, dependencies): callers then
/// leave the legacy behaviour instead of rejecting a valid assignment.
pub(super) fn property_settable_in_chain(
    ctx: &Ctx<'_>,
    class_name: &str,
    name: &str,
) -> Option<bool> {
    let mut queue: Vec<String> = vec![class_name.to_owned()];
    let mut seen: Vec<String> = Vec::new();
    while let Some(cls) = queue.pop() {
        if seen.contains(&cls) {
            continue;
        }
        seen.push(cls.clone());
        for c in ctx.classes.iter().filter(|c| c.name == cls) {
            for m in &c.members {
                if let crate::models::ClassMember::Property(p) = m
                    && p.name == name
                {
                    return Some(p.is_settable);
                }
            }
            for b in &c.bases {
                let simple = b.rsplit('.').next().unwrap_or(b).trim();
                if !simple.is_empty() {
                    queue.push(simple.to_owned());
                }
            }
        }
    }
    None
}

pub(super) fn eval_property_setter(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    info: &crate::native::semantic::ClassInfo,
    type_name: &str,
    name: &str,
    object: Option<&Expr>,
    value: VReg,
) -> bool {
    let Some(prop) = info.property(name) else {
        return false;
    };
    if b.prop_inline_stack.len() > 25 {
        return false;
    }
    if b
        .prop_inline_stack
        .iter()
        .any(|(c, p)| c == &info.key.path && p == name)
    {
        return false;
    }
    let Some((decl, _)) = most_derived_prop_setter_decl(ctx, info, name) else {
        return false;
    };
    let Some(block) = decl.set.clone() else {
        return false;
    };
    // Declarer-relative context (mirrors the getter path): the body
    // is the declaring class's own code.
    let declarer = declaring_entry(ctx, info, prop);
    // The setter is the declaring class's own code: its type
    // parameters bind the same way the getter's do.
    let class_subst = class_type_subst(ctx, info, type_name);
    let prop_ctx = Ctx {
        table: ctx.table,
        class_paths: ctx.class_paths,
        current_class: &prop.declaring_class,
        current_info: declarer.or(Some(info)),
        // The path must match the class this context switches to: the
        // caller's path with the declarer's class mixed in made the
        // access rule's nesting-family check compare unrelated types.
        current_path: declarer.map(|i| i.key.path.as_str()).or(ctx.current_path),
        classes: ctx.classes,
        import_index: ctx.import_index,
        dep_imports: ctx.dep_imports,
        host_import_count: ctx.host_import_count,
        method_vars: ctx.method_vars,
                type_subst: (!class_subst.is_empty()).then_some(&class_subst),
        local_fns: ctx.local_fns,
        current_method: Some(name.to_owned()),
        current_ret: None,
        static_index: ctx.static_index,
        defines: ctx.defines,
        lambda_deferred: None,
                    lambda_base: 0,
    };
    b.prop_inline_stack
        .push((info.key.path.clone(), name.to_owned()));
    let cp = b.checkpoint();
    let result: Result<(), LowerError> = (|| {
        if prop.is_static {
            if let Some(o) = object {
                let _ = eval(o, b, ctx)?;
            }
            b.scope_enter();
            let slot = b
                .scopes
                .declare("value", &prop.type_name, crate::models::Access::Default)
                .map_err(LowerError::UnsupportedConstruct)?;
            while b.bindings.len() <= slot {
                let v = b.fresh();
                b.bindings.push(v);
            }
            b.mov(b.bindings[slot], value);
            for s in &block.stmts {
                if b.is_sealed() {
                    break;
                }
                if matches!(s, crate::models::Stmt::LocalFunction { .. }) {
                    continue;
                }
                b.statement(s, &prop_ctx)?;
            }
            b.scope_exit();
            return Ok(());
        }
        let base = match object {
            Some(o) => eval(o, b, ctx)?,
            None => VReg(0),
        };
        let saved_this = b.save_this();
        b.mov(VReg(0), base);
        b.scope_enter();
        let tslot = b
            .scopes
            .declare("this", type_name, crate::models::Access::Default)
            .map_err(LowerError::UnsupportedConstruct)?;
        while b.bindings.len() <= tslot {
            let v = b.fresh();
            b.bindings.push(v);
        }
        b.mov(b.bindings[tslot], base);
        let vslot = b
            .scopes
            .declare("value", &prop.type_name, crate::models::Access::Default)
            .map_err(LowerError::UnsupportedConstruct)?;
        while b.bindings.len() <= vslot {
            let v = b.fresh();
            b.bindings.push(v);
        }
        b.mov(b.bindings[vslot], value);
        for s in &block.stmts {
            if b.is_sealed() {
                break;
            }
            if matches!(s, crate::models::Stmt::LocalFunction { .. }) {
                continue;
            }
            b.statement(s, &prop_ctx)?;
        }
        b.scope_exit();
        b.restore_this(saved_this);
        Ok(())
    })();
    b.prop_inline_stack.pop();
    match result {
        Ok(()) => true,
        Err(_) => {
            b.restore(cp);
            false
        }
    }
}

/// Lower a getter block inline, capturing `return` values instead
/// of returning from the enclosing function (see `inline_ret`).
/// `LocalFunction` declarations and `fn` stubs never execute here.
/// Unreachable statements after return/throw are skipped like
/// everywhere else.
pub(super) fn eval_block_inline(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    block: &Block,
) -> Result<VReg, LowerError> {
    let home = b.fresh();
    let z = b.const_i64(0);
    b.mov(home, z);
    let join_b = b.make_block();
    let prev = b.inline_ret.replace((join_b, home));
    for s in &block.stmts {
        if b.is_sealed() {
            break;
        }
        // Local-function declarations never execute inline (their
        // calls resolve via the pre-scan table); their `fn` stub
        // bindings lower normally (harmless zeroes, preserving any
        // identifier references).
        if matches!(s, crate::models::Stmt::LocalFunction { .. }) {
            continue;
        }
        b.statement(s, ctx)?;
    }
    b.inline_ret = prev;
    if !b.is_sealed() {
        b.goto(join_b);
    }
    b.switch_to(join_b);
    Ok(home)
}

/// C# pointer arithmetic over evaluated operands: `p + i`
/// advances `i*w` bytes, `p - i` retreats, `p - q` counts elements
/// (`w` = pointed-to width; `void*` keeps 8). Returns `None` when
/// neither side is a pointer (caller falls through to integer
/// Bin). Shared by the flattened-spine path (`combine_binary`)
/// and the legacy pair path below — pairs are the common shape
/// (`pBuffer += 2` desugars to one), so both must scale.
fn pointer_arith(
    op: &crate::models::BinOp,
    a: VReg,
    acc_ty: &str,
    bv: VReg,
    curr_ty: &str,
    b: &mut Body<'_>,
) -> Option<(VReg, String)> {
    use crate::models::BinOp as B;
    let ptr_width = |t: &str| -> Option<usize> {
        let tr = t.trim();
        if is_pointer_ty(tr) {
            Some(span_elem_width(tr.trim_end_matches(['*', ' ', '?']).trim()))
        } else {
            None
        }
    };
    match op {
        B::Add => {
            if let Some(w) = ptr_width(acc_ty) {
                let wv = b.const_i64(w as i64);
                let scaled = b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::Mul,
                    a: bv,
                    b: wv,
                });
                return Some((
                    b.emit(Instr::Bin {
                        dst: VReg(0),
                        op: IrBinOp::Add,
                        a,
                        b: scaled,
                    }),
                    acc_ty.to_owned(),
                ));
            } else if let Some(w) = ptr_width(curr_ty) {
                let wv = b.const_i64(w as i64);
                let scaled = b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::Mul,
                    a,
                    b: wv,
                });
                return Some((
                    b.emit(Instr::Bin {
                        dst: VReg(0),
                        op: IrBinOp::Add,
                        a: scaled,
                        b: bv,
                    }),
                    curr_ty.to_owned(),
                ));
            }
        }
        B::Sub => {
            let pa = ptr_width(acc_ty);
            let pb = ptr_width(curr_ty);
            if let (Some(w), Some(_)) = (pa, pb) {
                let diff = b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::Sub,
                    a,
                    b: bv,
                });
                let wv = b.const_i64(w as i64);
                return Some((
                    b.emit(Instr::Bin {
                        dst: VReg(0),
                        op: IrBinOp::Div,
                        a: diff,
                        b: wv,
                    }),
                    "long".to_owned(),
                ));
            } else if let Some(w) = pa {
                let wv = b.const_i64(w as i64);
                let scaled = b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::Mul,
                    a: bv,
                    b: wv,
                });
                return Some((
                    b.emit(Instr::Bin {
                        dst: VReg(0),
                        op: IrBinOp::Sub,
                        a,
                        b: scaled,
                    }),
                    acc_ty.to_owned(),
                ));
            }
        }
        _ => {}
    }
    None
}

/// Combine one flattened BinOp pair with the exact per-pair logic
/// of the legacy recursive path (string concat/compare, constant
/// folding, float form, integer Bin). Types and float-ness arrive
/// precomputed (the accumulator has no AST node); the caller tracks
/// them identically to what re-typing the composite would yield.
#[allow(clippy::too_many_arguments)]
fn combine_binary(    op: &crate::models::BinOp,
    a: VReg,
    acc_ty: &str,
    acc_float: bool,
    bv: VReg,
    curr_ty: &str,
    curr_float: bool,
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
) -> Result<(VReg, String), LowerError> {
    use crate::models::BinOp as B;
    // Next accumulator type (mirrors `static_type_of` for BinOp).
    let next_ty = if matches!(op, B::Add) && (is_stringy(acc_ty) || is_stringy(curr_ty)) {
        "string".to_owned()
    } else {
        numeric_result(acc_ty, curr_ty)
    };
    // First-class string operators (C# semantics).
    if is_stringy(acc_ty) && is_stringy(curr_ty) {
        match op {
            B::Add => {
                let len_a = {
                    let d = b.fresh();
                    b.push_raw(Instr::LoadField { dst: d, base: a, off: 0 });
                    d
                };
                let len_b = {
                    let d = b.fresh();
                    b.push_raw(Instr::LoadField { dst: d, base: bv, off: 0 });
                    d
                };
                let total = b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: len_a, b: len_b });
                let s16 = b.const_i64(16);
                let total_sz = b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: total, b: s16 });
                let newobj = b.fresh();
                b.push_raw(Instr::AllocDyn { dst: newobj, bytes: total_sz });
                b.push_raw(Instr::StoreField { base: newobj, off: 0, src: total });
                let one = b.const_i64(1);
                b.push_raw(Instr::StoreField { base: newobj, off: 8, src: one });
                let data_new = b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: newobj, b: s16 });
                let data_a = b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: a, b: s16 });
                b.raw_vreg(Instr::MemCpy { dst: data_new, src: data_a, len: len_a });
                let ptr2 = b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: data_new, b: len_a });
                let data_b = b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: bv, b: s16 });
                b.raw_vreg(Instr::MemCpy { dst: ptr2, src: data_b, len: len_b });
                return Ok((newobj, next_ty));
            }
            B::Eq | B::Ne => {
                let len_a = {
                    let d = b.fresh();
                    b.push_raw(Instr::LoadField { dst: d, base: a, off: 0 });
                    d
                };
                let len_b = {
                    let d = b.fresh();
                    b.push_raw(Instr::LoadField { dst: d, base: bv, off: 0 });
                    d
                };
                let len_eq = b.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Eq, a: len_a, b: len_b });
                let s16 = b.const_i64(16);
                let data_a = b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: a, b: s16 });
                let data_b = b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: bv, b: s16 });
                let same = b.fresh();
                b.push_raw(Instr::MemCmp { dst: same, a: data_a, b: data_b, len: len_a });
                let eq = b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitAnd, a: len_eq, b: same });
                let one = b.const_i64(1);
                let res = match op {
                    B::Eq => eq,
                    _ => b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitXor, a: eq, b: one }),
                };
                return Ok((res, next_ty));
            }
            _ => {}
        }
    }
    if let Some(pair) = pointer_arith(op, a, acc_ty, bv, curr_ty, b) {
        return Ok(pair);
    }
    // Constant folding (same rules as the legacy path). Unsigned division
    // is not folded: `fold_arith` divides signed i64, which is wrong above
    // i64::MAX and would bake in a negative quotient.
    let unsigned_div = matches!(op, B::Div | B::Rem) && is_unsigned_result(&next_ty);
    if !op.is_comparison()
        && !unsigned_div
        && let (Some(&va), Some(&vb)) = (b.const_vals.get(&a.0), b.const_vals.get(&bv.0))
    {
        let folded = fold_arith(op, va, vb);
        if let Some(v) = folded {
            return Ok((
                b.emit(Instr::Const {
                    dst: VReg(0),
                    imm: v,
                }),
                next_ty,
            ));
        }
    }
    if op.is_comparison()
        && let (Some(&va), Some(&vb)) = (b.const_vals.get(&a.0), b.const_vals.get(&bv.0))
    {
        let folded = match op {
            B::Eq => va == vb,
            B::Ne => va != vb,
            B::Lt => va < vb,
            B::Gt => va > vb,
            B::Le => va <= vb,
            B::Ge => va >= vb,
            _ => unreachable!("comparison filter above"),
        };
        return Ok((
            b.emit(Instr::Const {
                dst: VReg(0),
                imm: i64::from(folded),
            }),
            next_ty,
        ));
    }
    // IEEE 754 arithmetic when either side is float.
    if acc_float || curr_float {
        if let Some(fbinop) = match op {
            B::Add => Some(FBinOp::Add),
            B::Sub => Some(FBinOp::Sub),
            B::Mul => Some(FBinOp::Mul),
            B::Div => Some(FBinOp::Div),
            _ => None,
        } {
            let dst = b.fresh();
            b.push_raw(Instr::FBin {
                dst,
                op: fbinop,
                a,
                b: bv,
            });
            return Ok((dst, next_ty));
        }
    }
    let binop = match op {
        B::Add => IrBinOp::Add,
        B::Sub => IrBinOp::Sub,
        B::Mul => IrBinOp::Mul,
        B::Div => {
            if is_unsigned_result(&next_ty) {
                IrBinOp::UDiv
            } else {
                IrBinOp::Div
            }
        }
        B::Rem => {
            if is_unsigned_result(&next_ty) {
                IrBinOp::URem
            } else {
                IrBinOp::Rem
            }
        }
        B::BitAnd => IrBinOp::BitAnd,
        B::BitOr => IrBinOp::BitOr,
        B::BitXor => IrBinOp::BitXor,
        B::Shl => IrBinOp::Shl,
        B::Shr => IrBinOp::Shr,
        B::ShrShr => IrBinOp::ShrShr,
        _ => unreachable!("flattenable filter above"),
    };
    Ok((
        b.emit(Instr::Bin {
            dst: VReg(0),
            op: binop,
            a,
            b: bv,
        }),
        next_ty,
    ))
}

/// Canonical integral-type test (both alias and runtime spellings).
fn is_integral(t: &str) -> bool {
    matches!(
        t,
        "SByte" | "Byte" | "Int16" | "UInt16" | "Int32" | "UInt32" | "Int64"
            | "UInt64" | "IntPtr" | "UIntPtr" | "Char" | "sbyte" | "byte"
            | "short" | "ushort" | "int" | "uint" | "long" | "ulong" | "nint"
            | "nuint" | "char"
    )
}

/// Unsigned result types use `div` (not `idiv`) for `/` and `%`.
fn is_unsigned_result(t: &str) -> bool {
    matches!(
        crate::native::ir::lower::select::base_type_name(t).as_str(),
        "uint" | "UInt32" | "ulong" | "UInt64" | "nuint" | "UIntPtr"
    )
}

/// Integral target test (floats handled separately; pointers excluded
/// by the caller).
fn tgt_int(t: &str) -> bool {
    is_integral(t)
}

/// Current VALUE behind an `Interlocked` `ref` location (shared by
/// `lower_compare_exchange` and `interlocked_atomic`).
///
/// Call-site `ref` markers erase to the interior ADDRESS for
/// addressable targets (`ref arr[i]` → element address via
/// `ref_target_address`), but the exchange compares, stores, and
/// returns the VALUE there. Evaluating the marked arg directly yields
/// the address: the comparand never matches (conditional store
/// silently skipped — `_buckets[i]` stayed null forever) and the
/// address flows downstream (`?? inst` picked a heap address whose
/// null slot virtual-dispatched into the file-I/O segfault). Load
/// through the address (single evaluation — the store path
/// re-evaluates per the documented caveat); non-addressable shapes
/// (plain/static locals, impure bases) keep legacy value eval.
/// Returns the value vreg plus the address when one was taken, so a
/// caller needing a computed store (`ExchangeAdd`'s sum) can write
/// through it — interior writes land in place, no writeback needed.
fn interlocked_old(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    arg: &Expr,
) -> Result<(VReg, Option<VReg>), LowerError> {
    if let Some(addr) = ref_target_address(b, ctx, unmark_ref_arg(arg))? {
        let d = b.fresh();
        b.push_raw(Instr::LoadField {
            dst: d,
            base: addr,
            off: 0,
        });
        return Ok((d, Some(addr)));
    }
    Ok((eval(arg, b, ctx)?, None))
}

/// Lower an `Interlocked.CompareExchange`/`Exchange` call (see the
/// interception in `eval`): compare directly and conditionally store
/// through the normal assignment path, returning the original value.
fn lower_compare_exchange(
    name: &str,
    args: &[Expr],
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
) -> Result<VReg, LowerError> {
    let (orig, _) = interlocked_old(b, ctx, &args[0])?;
    let val = eval(&args[1], b, ctx)?;
    if name == "Exchange" {
        let assign = Expr::Assign {
            target: Box::new(args[0].clone()),
            value: Box::new(args[1].clone()),
            span: crate::models::Span::new(0, 0),
        };
        let _ = eval(&assign, b, ctx)?;
        let _ = val;
        return Ok(orig);
    }
    let cmp = eval(&args[2], b, ctx)?;
    let eq = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: CmpOp::Eq,
        a: orig,
        b: cmp,
    });
    let store_b = b.make_block();
    let end_b = b.make_block();
    b.terminate(Terminator::Branch {
        cond: Some(eq),
        if_true: store_b,
        if_false: end_b,
    });
    b.switch_to(store_b);
    let assign = Expr::Assign {
        target: Box::new(args[0].clone()),
        value: Box::new(args[1].clone()),
        span: crate::models::Span::new(0, 0),
    };
    let _ = eval(&assign, b, ctx)?;
    if !b.is_sealed() {
        b.goto(end_b);
    }
    b.switch_to(end_b);
    Ok(orig)
}

/// Bounds check for dynamic array access (`a[i]` read and write):
/// throws managed `IndexOutOfRangeException` when `i` falls outside
/// `[0, len)`, where `len` is the length header at `[base+0]`
/// (stored by `NewArray`). Without this, an out-of-range index
/// silently consumes heap garbage — or faults far from the cause —
/// e.g. `GetConstructors(...)[0]` on an empty array faulted during
/// static init instead of throwing into the init wrapper's handler.
/// Signed pair (`i < 0 || i >= len`) covers negative indices.
fn emit_array_bounds_check(
    base: VReg,
    idx: VReg,
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
) -> Result<(), LowerError> {
    let len = b.fresh();
    b.push_raw(Instr::LoadField {
        dst: len,
        base,
        off: 0,
    });
    let zero = b.const_i64(0);
    let neg = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: CmpOp::Lt,
        a: idx,
        b: zero,
    });
    let past = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: CmpOp::Ge,
        a: idx,
        b: len,
    });
    let bad = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Or,
        a: neg,
        b: past,
    });
    let throw_b = b.make_block();
    let end_b = b.make_block();
    b.terminate(Terminator::Branch {
        cond: Some(bad),
        if_true: throw_b,
        if_false: end_b,
    });
    b.switch_to(throw_b);
    let new_exc = Expr::New {
        class_name: "IndexOutOfRangeException".to_owned(),
        args: vec![],
        init: Vec::new(),
        span: crate::models::Span::new(0, 0),
    };
    let exc = eval(&new_exc, b, ctx)?;
    b.terminate(Terminator::Unwind { src: exc });
    b.switch_to(end_b);
    Ok(())
}

/// Mask for zero-extending narrow loads of unsigned types/// (`byte`/`char`/`uint`/…): `None` for 8-byte and signed narrow
/// types, whose values already live sign-correct in i64 (masking
/// `-1` would yield `4294967295`).
/// Scalar `BitOperations.Log2(v)` (floor log2, `0`→`0`) via halving.
/// Shared with the `_`-arm twin below (single source of truth — the
/// twin stays as fallback for entry-less spellings).
fn fold_log2_loop(b: &mut Body<'_>, v: VReg) -> VReg {
    // Loop-carried value lives in one register (`cur`), updated by
    // `mov` each trip (mirrors `fold_narrow_loop`'s `i`): a Rust-side
    // `v = half` rebind would freeze the condition on the first
    // half and spin forever (hung `/tmp/copyto` for 130min).
    // Constants hoisted before the loop so the first condition
    // entry sees defined values.
    let cur = b.fresh();
    b.mov(cur, v);
    let r = b.fresh();
    let zero = b.const_i64(0);
    b.mov(r, zero);
    let one = b.const_i64(1);
    let two = b.const_i64(2);
    let cond_b = b.make_block();
    let body_b = b.make_block();
    let done_b = b.make_block();
    b.goto(cond_b);
    b.switch_to(body_b);
    let half = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Div,
        a: cur,
        b: two,
    });
    b.mov(cur, half);
    let nr = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: r,
        b: one,
    });
    b.mov(r, nr);
    b.goto(cond_b);
    b.switch_to(cond_b);
    let big = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: CmpOp::Gt,
        a: cur,
        b: one,
    });
    b.terminate(Terminator::Branch {
        cond: Some(big),
        if_true: body_b,
        if_false: done_b,
    });
    b.switch_to(done_b);
    r
}

/// Scalar `Ascii.NarrowUtf16ToAscii(char*, byte*, count)` (ASCII
/// prefix copy, returns converted count). Shared twin (see above).
fn fold_narrow_loop(b: &mut Body<'_>, src: VReg, dst: VReg, count: VReg) -> VReg {
    let i = b.fresh();
    let zero = b.const_i64(0);
    b.mov(i, zero);
    let home = b.fresh();
    let cond_b = b.make_block();
    let body_b = b.make_block();
    let write_b = b.make_block();
    let done_b = b.make_block();
    b.goto(cond_b);
    b.switch_to(body_b);
    let two = b.const_i64(2);
    let s_off = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Mul,
        a: i,
        b: two,
    });
    let s_ptr = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: src,
        b: s_off,
    });
    let raw = b.fresh();
    b.push_raw(Instr::LoadField {
        dst: raw,
        base: s_ptr,
        off: 0,
    });
    let mask16 = b.const_i64(0xFFFF);
    let c = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitAnd,
        a: raw,
        b: mask16,
    });
    let hi = b.const_i64(0x7F);
    let non_ascii = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: CmpOp::Gt,
        a: c,
        b: hi,
    });
    b.terminate(Terminator::Branch {
        cond: Some(non_ascii),
        if_true: done_b,
        if_false: write_b,
    });
    b.switch_to(write_b);
    let d_ptr = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: dst,
        b: i,
    });
    b.push_raw(Instr::StoreField {
        base: d_ptr,
        off: 0,
        src: c,
    });
    let one = b.const_i64(1);
    let ni = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: i,
        b: one,
    });
    b.mov(i, ni);
    b.goto(cond_b);
    b.switch_to(cond_b);
    let more = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: CmpOp::Lt,
        a: i,
        b: count,
    });
    b.terminate(Terminator::Branch {
        cond: Some(more),
        if_true: body_b,
        if_false: done_b,
    });
    b.switch_to(done_b);
    b.mov(home, i);
    home
}

/// Primitive spellings that have no class entry (and no boxed tag to
/// test at runtime).
fn is_primitive_spelling(name: &str) -> bool {
    matches!(
        canonical_primitive(name).as_str(),
        "Boolean"
            | "Byte"
            | "SByte"
            | "Int16"
            | "UInt16"
            | "Int32"
            | "UInt32"
            | "Int64"
            | "UInt64"
            | "IntPtr"
            | "UIntPtr"
            | "Char"
            | "Single"
            | "Double"
            | "Decimal"
            | "String"
    )
}

pub(super) fn narrow_load_mask(ty: &str) -> Option<i64> {    let w = span_elem_width(ty);
    if w >= 8 {
        return None;
    }
    // Same byref-spelling normalization as `span_elem_width`
    // (`char&`, `ref char` denote the element type): without it a
    // `ref char` read skips masking and picks up neighbor bytes
    // (the casing-loop fault passed a garbage code point to
    // `ToUpper`, which threw).
    let mut t = ty.trim();
    for w_ in ["ref ", "out ", "in "] {
        if let Some(rest) = t.strip_prefix(w_) {
            t = rest;
            break;
        }
    }
    let simple = t
        .rsplit('.')
        .next()
        .unwrap_or(t)
        .trim_end_matches(['?', '*', '&']);
    let canon = canonical_primitive(simple);
    if matches!(
        canon.as_str(),
        "Boolean" | "Byte" | "Char" | "UInt16" | "UInt32" | "Single"
    ) {
        Some((1i64 << (w * 8)) - 1)
    } else {
        None
    }
}

/// Bit width for SIGNED narrow integral element types (mask +
/// sign-extend on dense array reads — see the index-read path).
/// Same spelling normalization as `narrow_load_mask`. Unsigned
/// and wide types return `None` (handled elsewhere or full-width).
pub(super) fn signed_narrow_bits(ty: &str) -> Option<u32> {
    let mut t = ty.trim();
    for w_ in ["ref ", "out ", "in "] {
        if let Some(rest) = t.strip_prefix(w_) {
            t = rest;
            break;
        }
    }
    let simple = t
        .rsplit('.')
        .next()
        .unwrap_or(t)
        .trim_end_matches(['?', '*', '&']);
    match canonical_primitive(simple).as_str() {
        "SByte" | "sbyte" => Some(8),
        "Int16" | "short" => Some(16),
        "Int32" | "int" => Some(32),
        _ => None,
    }
}

/// Address held by a by-reference local/parameter (`ref`/`out`/
/// `in`): the home IS the address (callers box or pass interior
/// addresses). Value-position `eval` dereferences — folds that
/// need the address itself (`Unsafe.Add`, ref-target forming)
/// must read the home, or they compute on the pointee value (the
/// `NarrowTwo` fault: `Add` ran on byte 65, storing to address
/// 66). Returns `None` for plain locals.
pub(super) fn byref_home(b: &Body<'_>, name: &str) -> Option<VReg> {
    b.scopes.lookup(name).and_then(|sym| {
        if sym.is_byref {
            Some(b.bindings[sym.slot])
        } else {
            None
        }
    })
}

/// Unwrap call-site `ref`/`in` markers (parser `&`-wrap) to the inner
/// expression for home/address decisions. The marker carries
/// overload-resolution meaning; the inner carries the location.
/// Without this, every `args[i] is Ident` check silently misses
/// marked args: `Unsafe.AddByteOffset(ref charA, …)` read the char
/// VALUE (85) as the base address instead of the byref home,
/// faulting the first case-insensitive compare at a small address.
fn unmark_ref_arg(arg: &Expr) -> &Expr {
    match arg {
        Expr::Unary {
            op: crate::models::ast::expression::UnOp::AddrOf,
            inner,
            ..
        } => inner,
        _ => arg,
    }
}

/// True when an index target uses the raw element layout
/// (`base + 16 + i*8`): arrays, strings, spans, raw pointers.
/// Anything else (e.g. `Dictionary<K,V>`) is an indexer and must
/// go through `get_Item`/`set_Item`.
fn is_index_raw_target(t: &str) -> bool {    let tr = t.trim();
    if is_pointer_ty(tr) {
        return true;
    }
    if is_string_type(t) {
        return true;
    }
    if span_view(t).is_some() {
        return true;
    }
    if tr.ends_with("[]") {
        return true;
    }
    false
}

/// True when the type declares an indexer accessor (`set_Item` /
/// `get_Item`): only then does index syntax desugar to a call.
/// Receivers without one (e.g. `ValueStringBuilder`, indexed
/// through its span) keep the raw path.
/// Inline an indexer setter (`dict[k] = v`): bind the receiver,
/// the index parameters, and `value`, then lower the `set` block
/// with the declaring class as context (mirrors property
/// inlining). Returns `None` when there is no usable setter, so
/// the caller keeps the raw path. The assignment value (bound
/// once) is the result.
/// Object-initializer member writes (`new C { Prop = value, ... }`):
/// after construction each element runs as a standalone member
/// write on the new object (field store or setter dispatch, via the
/// shared `Expr::Assign` path). The receiver binds to a scope temp
/// (`$initobj` — `$` never appears in parsed identifiers, so no
/// collision) so the value evaluates once and the write reuses the
/// exact member-assign logic. A member that resolves to neither
/// field nor setter keeps the legacy drop inside that path.
fn lower_object_inits(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    info: &crate::native::semantic::ClassInfo,
    obj: VReg,
    init: &[crate::models::ast::MemberInit],
) -> Result<(), LowerError> {
    if init.is_empty() {
        return Ok(());
    }
    b.scope_enter();
    let slot = match b.scopes.declare(
        "$initobj",
        &info.key.path,
        crate::models::Access::Default,
    ) {
        Ok(slot) => slot,
        Err(_) => {
            b.scope_exit();
            return Ok(());
        }
    };
    while b.bindings.len() <= slot {
        let fresh = b.fresh();
        b.bindings.push(fresh);
    }
    b.mov(b.bindings[slot], obj);
    for m in init {
        // Index initializer: `["k"] = v` is `this[k] = v` after
        // construction — the ordinary index-assignment lowering routes
        // it through `set_Item`.
        if let Some(index) = &m.index {
            let assign = Expr::Assign {
                target: Box::new(Expr::Index {
                    object: Box::new(Expr::Ident {
                        name: "$initobj".to_owned(),
                        span: m.span,
                    }),
                    index: Box::new(index.clone()),
                    extra: Vec::new(),
                    is_range: false,
                    span: m.span,
                }),
                value: Box::new(m.value.clone()),
                span: m.span,
            };
            eval(&assign, b, ctx)?;
            continue;
        }
        if m.name.is_empty() {
            // Collection initializer: `new List<T> { a, b }` is
            // `Add(a); Add(b)` after construction.
            let call = Expr::MethodCall {
                object: Box::new(Expr::Ident {
                    name: "$initobj".to_owned(),
                    span: m.span,
                }),
                name: "Add".to_owned(),
                args: vec![m.value.clone()],
                span: m.span,
            };
            eval(&call, b, ctx)?;
            continue;
        }
        let assign = Expr::Assign {
            target: Box::new(Expr::Member {
                object: Box::new(Expr::Ident {
                    name: "$initobj".to_owned(),
                    span: m.span,
                }),
                name: m.name.clone(),
                span: m.span,
            }),
            value: Box::new(m.value.clone()),
            span: m.span,
        };
        eval(&assign, b, ctx)?;
    }
    b.scope_exit();
    Ok(())
}

fn eval_indexer_set(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    info: &crate::native::semantic::ClassInfo,
    path: &str,
    decl: &crate::models::IndexerDecl,
    object: &Expr,
    indices: &[Expr],
    value: &Expr,
) -> Result<Option<VReg>, LowerError> {
    let Some(block) = decl.set.clone() else {
        return Ok(None);
    };
    if b.prop_inline_stack
        .iter()
        .any(|(c, p)| c == path && p == "[]")
    {
        return Ok(None);
    }
    b.prop_inline_stack.push((path.to_owned(), "[]".to_owned()));
    let set_ctx = Ctx {
        table: ctx.table,
        class_paths: ctx.class_paths,
        current_class: path,
        current_info: Some(info),
        current_path: Some(path),
        classes: ctx.classes,
        import_index: ctx.import_index,
        dep_imports: ctx.dep_imports,
        host_import_count: ctx.host_import_count,
        method_vars: ctx.method_vars,
                type_subst: None,
        local_fns: ctx.local_fns,
        current_method: Some("set_Item".to_owned()),
        current_ret: Some(decl.type_name.clone()),
        static_index: ctx.static_index,
        defines: ctx.defines,
        lambda_deferred: None,
                    lambda_base: 0,
    };
    // C# evaluation order: object, index, value.
    let base = eval(object, b, ctx)?;
    let saved_this = b.save_this();
    b.mov(VReg(0), base);
    b.scope_enter();
    let bind = |b: &mut Body<'_>, n: &str, ty: &str, v: VReg| -> Result<(), LowerError> {
        let slot = b
            .scopes
            .declare(n, ty, crate::models::Access::Default)
            .map_err(LowerError::UnsupportedConstruct)?;
        while b.bindings.len() <= slot {
            let f = b.fresh();
            b.bindings.push(f);
        }
        b.mov(b.bindings[slot], v);
        Ok(())
    };
    bind(b, "this", path, base)?;
    // Bind every index parameter, in order: a multi-parameter
    // indexer's `this[i, j]` reaches the body as two locals.
    for (k, idx) in indices.iter().enumerate() {
        let pname = decl
            .params
            .get(k)
            .map(|p| p.name.clone())
            .unwrap_or_else(|| format!("index{k}"));
        let pty = decl
            .params
            .get(k)
            .map(|p| p.type_name.clone())
            .unwrap_or_else(|| "object".to_owned());
        let iv = eval(idx, b, ctx)?;
        bind(b, &pname, &pty, iv)?;
    }
    let vv = eval(value, b, ctx)?;
    bind(b, "value", &decl.type_name, vv)?;
    let cp = b.checkpoint();
    let result = eval_block_inline(b, &set_ctx, &block);
    b.scope_exit();
    b.restore_this(saved_this);
    b.prop_inline_stack.pop();
    match result {
        Ok(_) => Ok(Some(vv)),
        Err(_) => {
            b.restore(cp);
            Ok(None)
        }
    }
}

/// Inline an indexer getter (`dict[k]`): bind the receiver and
/// the index parameters, then lower the `get` body (block or
/// expression) with the declaring class as context. Returns `None`
/// when there is no usable getter, so the caller keeps the raw
/// path.
fn eval_indexer_get(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    info: &crate::native::semantic::ClassInfo,
    path: &str,
    decl: &crate::models::IndexerDecl,
    object: &Expr,
    indices: &[Expr],
) -> Result<Option<VReg>, LowerError> {
    let has_get = decl.get.is_some() || decl.get_expr.is_some();
    if !has_get {
        return Ok(None);
    }
    if b.prop_inline_stack
        .iter()
        .any(|(c, p)| c == path && p == "[]")
    {
        return Ok(None);
    }
    b.prop_inline_stack.push((path.to_owned(), "[]".to_owned()));
    let get_ctx = Ctx {
        table: ctx.table,
        class_paths: ctx.class_paths,
        current_class: path,
        current_info: Some(info),
        // The path must match the class this context switches to (the
        // indexer's owner), like the setter's context.
        current_path: Some(path),
        classes: ctx.classes,
        import_index: ctx.import_index,
        dep_imports: ctx.dep_imports,
        host_import_count: ctx.host_import_count,
        method_vars: ctx.method_vars,
                type_subst: None,
        local_fns: ctx.local_fns,
        current_method: Some("get_Item".to_owned()),
        current_ret: Some(decl.type_name.clone()),
        static_index: ctx.static_index,
        defines: ctx.defines,
        lambda_deferred: None,
                    lambda_base: 0,
    };
    let base = eval(object, b, ctx)?;
    let saved_this = b.save_this();
    b.mov(VReg(0), base);
    b.scope_enter();
    let bind = |b: &mut Body<'_>, n: &str, ty: &str, v: VReg| -> Result<(), LowerError> {
        let slot = b
            .scopes
            .declare(n, ty, crate::models::Access::Default)
            .map_err(LowerError::UnsupportedConstruct)?;
        while b.bindings.len() <= slot {
            let f = b.fresh();
            b.bindings.push(f);
        }
        b.mov(b.bindings[slot], v);
        Ok(())
    };
    bind(b, "this", path, base)?;
    for (k, idx) in indices.iter().enumerate() {
        let pname = decl
            .params
            .get(k)
            .map(|p| p.name.clone())
            .unwrap_or_else(|| format!("index{k}"));
        let pty = decl
            .params
            .get(k)
            .map(|p| p.type_name.clone())
            .unwrap_or_else(|| "object".to_owned());
        let iv = eval(idx, b, ctx)?;
        bind(b, &pname, &pty, iv)?;
    }
    let cp = b.checkpoint();
    let got = if let Some(e) = decl.get_expr.clone() {
        (|| -> Result<VReg, LowerError> {
            let v = eval(&e, b, &get_ctx)?;
            let val_ty =
                static_type_of(&e, b, &get_ctx).unwrap_or_else(|_| "object".to_owned());
            coerce_span_return(b, &get_ctx, v, &val_ty, &decl.type_name)
        })()
    } else if let Some(block) = decl.get.clone() {
        eval_block_inline(b, &get_ctx, &block)
    } else {
        Ok(b.const_i64(0))
    };
    b.scope_exit();
    b.restore_this(saved_this);
    b.prop_inline_stack.pop();
    match got {
        Ok(v) => Ok(Some(v)),
        Err(_) => {
            b.restore(cp);
            Ok(None)
        }
    }
}

pub(super) fn find_indexer(
    ctx: &Ctx<'_>,
    t: &str,
    arity: usize,
) -> Option<(String, crate::models::IndexerDecl)> {
    let info = ctx
        .table
        .class(t)
        .or_else(|| definition_entry(ctx.table, t))?;
    for c in ctx.classes.iter() {
        if c.name != info.name && c.name != info.key.path {
            continue;
        }
        for m in &c.members {
            if let crate::models::ClassMember::Indexer(d) = m {
                if d.params.len() == arity {
                    return Some((info.key.path.clone(), d.clone()));
                }
            }
        }
    }
    None
}

/// Emulate `[Intrinsic]` self-recursive `Interlocked` atomics
/// (`Exchange` / `ExchangeAdd` / `CompareExchange` bodies call
/// themselves — "Must expand intrinsic" — so running them recurses
/// forever). Single-threaded-exact via Volatile-style read/assign
/// composition (freestanding exes never observe torn access):
/// Exchange swaps, ExchangeAdd adds, CompareExchange swaps on
/// match; all return the original value. Location expressions
/// re-evaluate (same pre-existing double-evaluation as receivers
/// elsewhere); single-threaded values stay exact. Returns `None`
/// when the (name, arity) is not an emulated atomic.
pub(super) fn interlocked_atomic(
    name: &str,
    args: &[Expr],
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
) -> Option<Result<VReg, LowerError>> {
    let applicable = (name == "Exchange" || name == "ExchangeAdd") && args.len() == 2
        || name == "CompareExchange" && args.len() == 3;
    if !applicable {
        return None;
    }
    Some((|| {
        let (old, addr) = interlocked_old(b, ctx, &args[0])?;
        if name == "ExchangeAdd" {
            let v1 = eval(&args[1], b, ctx)?;
            match addr {
                // Interior address: sum from the loaded value and
                // store through the address (in place, no writeback —
                // interior writes land directly). Legacy path below
                // keeps value semantics for plain locals.
                Some(a) => {
                    let sum = b.emit(Instr::Bin {
                        dst: VReg(0),
                        op: IrBinOp::Add,
                        a: old,
                        b: v1,
                    });
                    b.push_raw(Instr::StoreField {
                        base: a,
                        off: 0,
                        src: sum,
                    });
                    return Ok(old);
                }
                None => {
                    let add = Expr::Binary {
                        op: crate::models::BinOp::Add,
                        lhs: Box::new(args[0].clone()),
                        rhs: Box::new(args[1].clone()),
                        span: crate::models::Span::new(0, 0),
                    };
                    let assign = Expr::Assign {
                        target: Box::new(args[0].clone()),
                        value: Box::new(add),
                        span: crate::models::Span::new(0, 0),
                    };
                    let _ = eval(&assign, b, ctx)?;
                    return Ok(old);
                }
            }
        }
        if name == "CompareExchange" {
            let comparand = eval(&args[2], b, ctx)?;
            let same = b.emit(Instr::Cmp {
                dst: VReg(0),
                op: crate::native::ir::CmpOp::Eq,
                a: old,
                b: comparand,
            });
            let do_b = b.make_block();
            let done_b = b.make_block();
            b.terminate(Terminator::Branch {
                cond: Some(same),
                if_true: do_b,
                if_false: done_b,
            });
            b.switch_to(do_b);
            let assign = Expr::Assign {
                target: Box::new(args[0].clone()),
                value: Box::new(args[1].clone()),
                span: crate::models::Span::new(0, 0),
            };
            let _ = eval(&assign, b, ctx)?;
            b.goto(done_b);
            b.switch_to(done_b);
            return Ok(old);
        }
        let assign = Expr::Assign {
            target: Box::new(args[0].clone()),
            value: Box::new(args[1].clone()),
            span: crate::models::Span::new(0, 0),
        };
        let _ = eval(&assign, b, ctx)?;
        Ok(old)
    })())
}

/// P/Invoke SafeHandle marshaling (`SafeHandle` → OS handle): a/// SafeHandle-derived argument to an integer parameter passes the
/// underlying handle (`DangerousGetHandle`), not the object pointer
/// (which faulted `Console.WriteLine` with EBADF — the heap address
/// went out as the fd). Only integer↔SafeHandle pairs convert;
/// everything else passes through untouched. For bodyless extern
/// callees (`force_extern`, e.g. `[LibraryImport]` natives whose
/// managed decl says `SafeHandle` but the native takes an int),
/// convert unconditionally: native code has no objects.
pub(super) fn marshal_safehandle_args(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    argv: &mut [VReg],
    arg_types: &[String],
    param_types: &[String],
    force_extern: bool,
) -> Result<(), LowerError> {
    // `param_types` empty (extern call sites without overload info):
    // native code has no objects, so every SafeHandle-derived arg
    // marshals. Otherwise zip with params (missing tail = no-op).
    // When `force_extern`, the integer-param gate is skipped (the
    // native signature takes ints where the managed decl may say
    // `SafeHandle`).
    for (i, (slot, arg_ty)) in argv.iter_mut().zip(arg_types.iter()).enumerate() {
        if !force_extern {
            let Some(param_ty) = param_types.get(i) else {
                continue;
            };
            let p = base_type_name(param_ty);
            if !matches!(
                p.as_str(),
                "IntPtr"
                    | "UIntPtr"
                    | "nint"
                    | "nuint"
                    | "int"
                    | "uint"
                    | "long"
                    | "ulong"
                    | "Int32"
                    | "UInt32"
                    | "Int64"
                    | "UInt64"
            ) {
                continue;
            }
        }
        let a_base = base_type_name(arg_ty);
        if !(a_base == "SafeHandle"
            || a_base == "SafeFileHandle"
            || a_base == "CriticalHandle"
            || ctx.table.derives_from(arg_ty, "SafeHandle")
            || ctx.table.derives_from(arg_ty, "CriticalHandle"))
        {
            continue;
        }
        if let Some(entry) = ctx.table.class(arg_ty) {
            if let Some(f) = entry.field("handle") {
                if f.is_static {
                    continue;
                }
                let dst = b.fresh();
                b.push_raw(Instr::LoadField {
                    dst,
                    base: *slot,
                    off: (f.offset_slots * 8) as u32,
                });
                *slot = dst;
            }
        }
    }
    Ok(())
}

/// P/Invoke string marshaling (`string` → NUL-terminated UTF-8): a
/// managed-string argument to a NATIVE import passes a transcoded temp
/// buffer, never the object pointer (whose first 8 bytes are the
/// LENGTH — `File.WriteAllBytes` opened `"\021"` and died ENOENT on
/// it, then the -2 fd faulted downstream at `[fd+0x58]`). Only native
/// `LibraryImport`/`DllImport` boundaries (C ABI) marshal; managed
/// bodies keep the object (they read our UTF-16 layout directly).
/// Null passes through as null. Non-ASCII encodes fully (2/3-byte
/// BMP, 4-byte surrogate pairs, lone surrogates as U+FFFD) — an
/// ASCII-only shortcut would silently corrupt real paths.
pub(super) fn marshal_string_utf8_args(
    b: &mut Body<'_>,
    argv: &mut [VReg],
    arg_types: &[String],
) {
    for (slot, arg_ty) in argv.iter_mut().zip(arg_types.iter()) {
        if !is_string_type(arg_ty) {
            continue;
        }
        // Null literal stays null (native NULL); anything else
        // transcodes (a null OBJECT would fault on the length load —
        // managed null-checks throw before P/Invoke in practice).
        if b.const_vals.get(&slot.0).is_some_and(|c| *c == 0) {
            continue;
        }
        *slot = marshal_string_to_utf8(b, *slot);
    }
}

/// Transcode one managed UTF-16 string object to a fresh NUL-terminated
/// UTF-8 heap buffer (see above). Layout: length @+0 (chars), tag @+8,
/// UTF-16LE @+16. The temp lives on the never-freed bump heap (no GC
/// here), so no pinning — same rationale as
/// `AllocateTypeAssociatedMemory`. Byte stores overlap ascending like
/// `fold_narrow_loop` (the last writer to each byte wins, packing
/// exact). All constants hoist above the loop so first entry reads
/// defined values (the `Log2` spin lesson); `i`/`j` are mov-carried
/// (never Rust-rebound across trips).
fn marshal_string_to_utf8(b: &mut Body<'_>, s: VReg) -> VReg {
    let out = b.fresh();
    let zero = b.const_i64(0);
    let one = b.const_i64(1);
    let two = b.const_i64(2);
    let three = b.const_i64(3);
    let six = b.const_i64(6);
    let ten = b.const_i64(10);
    let twelve = b.const_i64(12);
    let eighteen = b.const_i64(18);
    let sixteen = b.const_i64(16);
    let mask16 = b.const_i64(0xFFFF);
    let v800 = b.const_i64(0x800);
    let v80 = b.const_i64(0x80);
    let v3f = b.const_i64(0x3F);
    let vc0 = b.const_i64(0xC0);
    let ve0 = b.const_i64(0xE0);
    let vf0 = b.const_i64(0xF0);
    let vd7ff = b.const_i64(0xD7FF);
    let vd800 = b.const_i64(0xD800);
    let vdc00 = b.const_i64(0xDC00);
    let vdbff = b.const_i64(0xDBFF);
    let ve000 = b.const_i64(0xE000);
    let vfffd = b.const_i64(0xFFFD);
    let v10000 = b.const_i64(0x10000);
    let is_null = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: CmpOp::Eq,
        a: s,
        b: zero,
    });
    let null_b = b.make_block();
    let ok_b = b.make_block();
    let join_b = b.make_block();
    b.terminate(Terminator::Branch {
        cond: Some(is_null),
        if_true: null_b,
        if_false: ok_b,
    });
    b.switch_to(null_b);
    b.mov(out, zero);
    b.goto(join_b);
    b.switch_to(ok_b);
    // Length first, one allocation sized for the worst case (every
    // char 3 bytes + NUL; surrogate pairs only shrink it).
    let len = b.fresh();
    b.push_raw(Instr::LoadField {
        dst: len,
        base: s,
        off: 0,
    });
    let tripled = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Mul,
        a: len,
        b: three,
    });
    let maxb = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: tripled,
        b: one,
    });
    let buf = b.fresh();
    b.push_raw(Instr::AllocDyn {
        dst: buf,
        bytes: maxb,
    });
    let base16 = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: s,
        b: sixteen,
    });
    let i = b.fresh();
    b.mov(i, zero);
    let j = b.fresh();
    b.mov(j, zero);
    let cond_b = b.make_block();
    let body_b = b.make_block();
    let done_b = b.make_block();
    let e1_b = b.make_block();
    let k1_b = b.make_block();
    let e2_b = b.make_block();
    let k2_b = b.make_block();
    let k2c_b = b.make_block();
    let l3_b = b.make_block();
    let l4_b = b.make_block();
    let e4_b = b.make_block();
    let e3a_b = b.make_block();
    let e3a_e = b.make_block();
    let e3b_b = b.make_block();
    let e3b_e = b.make_block();
    b.goto(cond_b);
    // Loop head: `i < len`?
    b.switch_to(cond_b);
    let more = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: CmpOp::Lt,
        a: i,
        b: len,
    });
    b.terminate(Terminator::Branch {
        cond: Some(more),
        if_true: body_b,
        if_false: done_b,
    });
    // One step emitting the i-th char; every path advances and
    // returns to `cond_b`. Small local closures keep the repeated
    // byte-store / advance shapes identical (all values flow through
    // parameters — no shared mutable state).
    let emit_byte = |b: &mut Body<'_>, buf: VReg, j: VReg, val: VReg, one: VReg| {
        let d = b.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::Add,
            a: buf,
            b: j,
        });
        b.push_raw(Instr::StoreField {
            base: d,
            off: 0,
            src: val,
        });
        let nj = b.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::Add,
            a: j,
            b: one,
        });
        b.mov(j, nj);
    };
    let adv1 = |b: &mut Body<'_>, i: VReg, one: VReg, cond_b| {
        let ni = b.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::Add,
            a: i,
            b: one,
        });
        b.mov(i, ni);
        b.goto(cond_b);
    };
    b.switch_to(body_b);
    let s_off = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Mul,
        a: i,
        b: two,
    });
    let s_ptr = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: base16,
        b: s_off,
    });
    let raw = b.fresh();
    b.push_raw(Instr::LoadField {
        dst: raw,
        base: s_ptr,
        off: 0,
    });
    let c = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitAnd,
        a: raw,
        b: mask16,
    });
    let ascii = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: CmpOp::Lt,
        a: c,
        b: v80,
    });
    b.terminate(Terminator::Branch {
        cond: Some(ascii),
        if_true: e1_b,
        if_false: k1_b,
    });
    // 1-byte path (ASCII).
    b.switch_to(e1_b);
    emit_byte(b, buf, j, c, one);
    adv1(b, i, one, cond_b);
    // 2-byte test.
    b.switch_to(k1_b);
    let two_byte = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: CmpOp::Lt,
        a: c,
        b: v800,
    });
    b.terminate(Terminator::Branch {
        cond: Some(two_byte),
        if_true: e2_b,
        if_false: k2_b,
    });
    // 2-byte path.
    b.switch_to(e2_b);
    let shr6 = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Shr,
        a: c,
        b: six,
    });
    let b1 = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitOr,
        a: vc0,
        b: shr6,
    });
    emit_byte(b, buf, j, b1, one);
    let lo2 = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitAnd,
        a: c,
        b: v3f,
    });
    let b2 = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitOr,
        a: v80,
        b: lo2,
    });
    emit_byte(b, buf, j, b2, one);
    adv1(b, i, one, cond_b);
    // Surrogate-range test (lead candidates and lone-trail/above
    // handling split below).
    b.switch_to(k2_b);
    let g_lead = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: CmpOp::Gt,
        a: c,
        b: vd7ff,
    });
    let l_e000 = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: CmpOp::Lt,
        a: c,
        b: ve000,
    });
    let is_surr = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitAnd,
        a: g_lead,
        b: l_e000,
    });
    b.terminate(Terminator::Branch {
        cond: Some(is_surr),
        if_true: e3a_b,
        if_false: k2c_b,
    });
    // Non-surrogate ≥ 0x800: plain 3-byte code point.
    b.switch_to(k2c_b);
    let g_dc00 = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: CmpOp::Gt,
        a: c,
        b: vd7ff,
    });
    let l_dc00 = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: CmpOp::Lt,
        a: c,
        b: vdc00,
    });
    let is_lead = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitAnd,
        a: g_dc00,
        b: l_dc00,
    });
    b.terminate(Terminator::Branch {
        cond: Some(is_lead),
        if_true: l3_b,
        if_false: e3b_b,
    });
    // Lead with a next char? Else lone lead → U+FFFD.
    b.switch_to(l3_b);
    let plus1 = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: i,
        b: one,
    });
    let has_next = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: CmpOp::Lt,
        a: plus1,
        b: len,
    });
    b.terminate(Terminator::Branch {
        cond: Some(has_next),
        if_true: l4_b,
        if_false: e3a_b,
    });
    // Trail second half? Else lone lead → U+FFFD.
    b.switch_to(l4_b);
    let s_off2 = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Mul,
        a: plus1,
        b: two,
    });
    let s_ptr2 = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: base16,
        b: s_off2,
    });
    let raw2 = b.fresh();
    b.push_raw(Instr::LoadField {
        dst: raw2,
        base: s_ptr2,
        off: 0,
    });
    let c2 = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitAnd,
        a: raw2,
        b: mask16,
    });
    let g_trail = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: CmpOp::Gt,
        a: c2,
        b: vdbff,
    });
    let l_trail = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: CmpOp::Lt,
        a: c2,
        b: ve000,
    });
    let is_trail = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitAnd,
        a: g_trail,
        b: l_trail,
    });
    b.terminate(Terminator::Branch {
        cond: Some(is_trail),
        if_true: e4_b,
        if_false: e3a_b,
    });
    // 4-byte astral path.
    b.switch_to(e4_b);
    let lead_sub = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Sub,
        a: c,
        b: vd800,
    });
    let lead_sh = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Shl,
        a: lead_sub,
        b: ten,
    });
    let trail_sub = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Sub,
        a: c2,
        b: vdc00,
    });
    let pair_sum = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: lead_sh,
        b: trail_sub,
    });
    let astral = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: pair_sum,
        b: v10000,
    });
    let q1 = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Shr,
        a: astral,
        b: eighteen,
    });
    let a1 = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitOr,
        a: vf0,
        b: q1,
    });
    emit_byte(b, buf, j, a1, one);
    let q2s = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Shr,
        a: astral,
        b: twelve,
    });
    let q2m = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitAnd,
        a: q2s,
        b: v3f,
    });
    let a2 = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitOr,
        a: v80,
        b: q2m,
    });
    emit_byte(b, buf, j, a2, one);
    let q3s = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Shr,
        a: astral,
        b: six,
    });
    let q3m = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitAnd,
        a: q3s,
        b: v3f,
    });
    let a3 = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitOr,
        a: v80,
        b: q3m,
    });
    emit_byte(b, buf, j, a3, one);
    let q4m = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitAnd,
        a: astral,
        b: v3f,
    });
    let a4 = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitOr,
        a: v80,
        b: q4m,
    });
    emit_byte(b, buf, j, a4, one);
    let ni2 = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: i,
        b: two,
    });
    b.mov(i, ni2);
    b.goto(cond_b);
    // 3-byte select: U+FFFD for surrogates, the char itself above.
    // No phi nodes in this IR, so each incoming edge carries its code
    // point in its OWN home register and runs its own copy of the
    // tiny 3-byte tail (e3a_e / e3b_e) — sharing one join block would
    // need a phi to pick the value.
    b.switch_to(e3a_b);
    let cp_a = b.fresh();
    b.mov(cp_a, vfffd);
    b.goto(e3a_e);
    b.switch_to(e3b_b);
    let cp_b = b.fresh();
    b.mov(cp_b, c);
    b.goto(e3b_e);
    // 3-byte tail, once per edge home.
    b.switch_to(e3a_e);
    let a3h = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Shr,
        a: cp_a,
        b: twelve,
    });
    let a3b1 = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitOr,
        a: ve0,
        b: a3h,
    });
    emit_byte(b, buf, j, a3b1, one);
    let a3ms = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Shr,
        a: cp_a,
        b: six,
    });
    let a3mm = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitAnd,
        a: a3ms,
        b: v3f,
    });
    let a3b2 = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitOr,
        a: v80,
        b: a3mm,
    });
    emit_byte(b, buf, j, a3b2, one);
    let a3lm = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitAnd,
        a: cp_a,
        b: v3f,
    });
    let a3b3 = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitOr,
        a: v80,
        b: a3lm,
    });
    emit_byte(b, buf, j, a3b3, one);
    adv1(b, i, one, cond_b);
    b.switch_to(e3b_e);
    let b3h = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Shr,
        a: cp_b,
        b: twelve,
    });
    let b3b1 = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitOr,
        a: ve0,
        b: b3h,
    });
    emit_byte(b, buf, j, b3b1, one);
    let b3ms = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Shr,
        a: cp_b,
        b: six,
    });
    let b3mm = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitAnd,
        a: b3ms,
        b: v3f,
    });
    let b3b2 = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitOr,
        a: v80,
        b: b3mm,
    });
    emit_byte(b, buf, j, b3b2, one);
    let b3lm = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitAnd,
        a: cp_b,
        b: v3f,
    });
    let b3b3 = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitOr,
        a: v80,
        b: b3lm,
    });
    emit_byte(b, buf, j, b3b3, one);
    adv1(b, i, one, cond_b);
    b.switch_to(done_b);
    let dend = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: buf,
        b: j,
    });
    b.push_raw(Instr::StoreField {
        base: dend,
        off: 0,
        src: zero,
    });
    b.mov(out, buf);
    b.goto(join_b);
    b.switch_to(join_b);
    out
}

/// P/Invoke SafeHandle returns (raw OS handle → live object): a NATIVE
/// import whose managed return derives from SafeHandle/CriticalHandle
/// yields the raw fd — downstream managed code treats it as an object
/// (vtable/field traffic on a small int faulted file open at
/// `[fd+0x58]`). Wrap it hollow (vtable + tag + zeroed fields,
/// mirroring the hollow-`Thread` in `calls.rs`): fd into the `handle`
/// slot, `_ownsHandle` set (the generator owns `Open` results —
/// without it `DangerousAddRef` throws), everything else zero.
/// Managed-body cross-image calls keep their real objects (same IR
/// ABI, untouched). Stores use the table's own `offset_slots` (the
/// same slots reads use), so construction and access agree by
/// construction; sizing covers the highest slot.
pub(super) fn wrap_safehandle_import_return(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    dst: VReg,
    ret_ty: &str,
) -> VReg {
    let r_base = base_type_name(ret_ty);
    if !(r_base == "SafeHandle"
        || r_base == "SafeFileHandle"
        || r_base == "CriticalHandle"
        || ctx.table.derives_from(ret_ty, "SafeHandle")
        || ctx.table.derives_from(ret_ty, "CriticalHandle"))
    {
        return dst;
    }
    let Some(entry) = ctx.table.class(ret_ty) else {
        return dst;
    };
    let Some(handle_f) = entry.field("handle").cloned() else {
        return dst;
    };
    if handle_f.is_static {
        return dst;
    }
    let mut top: usize = 2;
    for f in entry.fields.iter() {
        if !f.is_static && f.offset_slots + 1 > top {
            top = f.offset_slots + 1;
        }
    }
    let obj = b.fresh();
    b.alloc(obj, top as u64 * 8);
    let row = vtable_row_for(ctx, &entry.name, &entry.key.path);
    let vaddr = b.fresh();
    b.push_raw(Instr::VTableAddr { dst: vaddr, row });
    b.push_raw(Instr::StoreField {
        base: obj,
        off: 0,
        src: vaddr,
    });
    let one = b.const_i64(1);
    b.push_raw(Instr::StoreField {
        base: obj,
        off: 8,
        src: one,
    });
    let zero = b.const_i64(0);
    for f in entry.fields.iter() {
        if f.is_static || f.name == "handle" || f.name == "_ownsHandle" {
            continue;
        }
        b.push_raw(Instr::StoreField {
            base: obj,
            off: (f.offset_slots * 8) as u32,
            src: zero,
        });
    }
    b.push_raw(Instr::StoreField {
        base: obj,
        off: (handle_f.offset_slots * 8) as u32,
        src: dst,
    });
    if let Some(own) = entry.field("_ownsHandle") {
        if !own.is_static {
            b.push_raw(Instr::StoreField {
                base: obj,
                off: (own.offset_slots * 8) as u32,
                src: one,
            });
        }
    }
    // Mirror the real `SafeHandle` ctor (`handle = invalid; _state =
    // RefCountOne; _ownsHandle = ownsHandle; … _fullyInitialized =
    // true`): a zeroed `_state` reads as disposed-with-zero-refs, so
    // the first `Dispose` throws `ObjectDisposedException` (whose
    // unwinding faults — no real throw runtime here) instead of
    // releasing. `RefCountOne` is `1 << 2` (low two bits are
    // closed/disposed flags). Missing fields keep legacy zeros.
    if let Some(st) = entry.field("_state") {
        if !st.is_static {
            let ref_one = b.const_i64(1 << 2);
            b.push_raw(Instr::StoreField {
                base: obj,
                off: (st.offset_slots * 8) as u32,
                src: ref_one,
            });
        }
    }
    if let Some(fi) = entry.field("_fullyInitialized") {
        if !fi.is_static {
            b.push_raw(Instr::StoreField {
                base: obj,
                off: (fi.offset_slots * 8) as u32,
                src: one,
            });
        }
    }
    obj
}

/// Target-type a bare `default` (`"void"` placeholder from parsing):
/// C# `default` takes its type from context (the other ternary arm,
/// a `let` declared type). Returns the annotated expression when
/// `expr` is a bare default and `ty` names a concrete type, else
/// `None` (no clone on the hot path). Explicit `default(T)` keeps
/// its spelling.
pub(super) fn default_with_type(expr: &Expr, ty: &str) -> Option<Expr> {
    match expr {
        Expr::Default { span, type_name } if type_name == "void" && !ty.is_empty() && ty != "void" => {
            Some(Expr::Default {
                type_name: ty.to_owned(),
                span: *span,
            })
        }
        _ => None,
    }
}

/// Shape-based `ToArray()` fallback (see call sites): copy a span,
/// array, or string buffer into a new array. Covers `u8` literals,
/// which mistype as `string` (the lexer drops the suffix), so
/// string receivers are included — a genuine `string.ToArray()` is
/// invalid .NET and has no other resolution here. Reachable both
/// from entry-backed receivers (overload resolution found nothing)
/// and from entry-less ones (arrays carry no table entry), which
/// lower through the unknown-receiver arm instead. Returns `None`
/// for non-buffer receivers (interfaces, objects, …) so callers
/// fall through to extension/interface/placeholder paths — erroring
/// there would break interface calls like
/// `IProducerConsumerCollection<T>.ToArray()`.
pub(super) fn lower_to_array_pack(
    object: &Expr,
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    type_name: &str,
) -> Option<Result<VReg, LowerError>> {
    if span_view(type_name).is_none()
        && array_elem_ty(type_name).is_none()
        && !is_string_type(type_name)
    {
        return None;
    }
    let base = match eval(object, b, ctx) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };
    // (data pointer, element count, copy width) by receiver kind.
    // Arrays pack dense (matching the element-read path); span
    // buffers pack tight; strings store bytes.
    let (data, len, w) = if let Some((_, inner)) = span_view(type_name) {
        // Null/default spans copy as empty (the length load below
        // would fault; .NET returns `[]` via the IsEmpty path).
        let base = ensure_struct_base(b, ctx, base, type_name);
        let r = b.fresh();
        b.push_raw(Instr::LoadField { dst: r, base, off: 16 });
        let l = b.fresh();
        b.push_raw(Instr::LoadField { dst: l, base, off: 24 });
        (r, l, span_elem_width(&inner))
    } else if let Some(inner) = array_elem_ty(type_name) {
        let l = b.fresh();
        b.push_raw(Instr::LoadField { dst: l, base, off: 0 });
        let sixteen = b.const_i64(16);
        let d = b.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::Add,
            a: base,
            b: sixteen,
        });
        // Dense source stride (matches the element-read path).
        (d, l, span_elem_width(&inner))
    } else if is_string_type(type_name) {
        let l = b.fresh();
        b.push_raw(Instr::LoadField { dst: l, base, off: 0 });
        let sixteen = b.const_i64(16);
        let d = b.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::Add,
            a: base,
            b: sixteen,
        });
        (d, l, 1usize)
    } else {
        return None;
    };
    // Dense destination (new arrays pack tight): `len*w` payload
    // plus header and tail slop.
    let wv = b.const_i64(w as i64);
    let sixteen = b.const_i64(16);
    let pad = b.const_i64(24);
    let payload = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Mul,
        a: len,
        b: wv,
    });
    let total = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: payload,
        b: pad,
    });
    let arr = b.fresh();
    b.push_raw(Instr::AllocDyn { dst: arr, bytes: total });
    b.push_raw(Instr::StoreField { base: arr, off: 0, src: len });
    let dst = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: arr,
        b: sixteen,
    });
    pack_loop(b, dst, data, len, w, w, w);
    Some(Ok(arr))
}

/// Function index for a method group used as a value: the declaring class
/// is searched from the current one outward through its base chain, taking
/// the first class that declares a method of that name.
fn method_group_index(name: &str, b: &Body<'_>, ctx: &Ctx<'_>) -> Option<u32> {
    let _ = b;
    for class in ctx.classes.iter() {
        // The arity is the DECLARED parameter count: a group of `Twice(int)`
        // must resolve `Twice#1`, not `Twice#0` (which does not exist, so
        // the lookup silently failed and the value was reported missing).
        for m in class.members.iter() {
            let crate::models::ClassMember::Method(md) = m else {
                continue;
            };
            if md.name != name {
                continue;
            }
            if let Some(i) = function_index_for(ctx, &class.name, name, md.params.len()) {
                return Some(i);
            }
        }
    }
    None
}

/// Emit `value is <type>`: compare the value's vtable-row word against
/// the row address of the type and of every class deriving from it.
///
/// An object's word 0 is the ADDRESS of its row (not an index), so this is
/// an address comparison per candidate. Exact for classes; an INTERFACE
/// target is reported because interfaces still have no row of their own
/// — answering false would be a fabrication.
pub(super) fn emit_type_test(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    value: VReg,
    simple: &str,
    spelling: &str,
) -> Result<VReg, LowerError> {
    // A string is a tagged runtime object: the high bit of its refcount
    // word is set (STRING_REF). Class objects keep a plain positive count,
    // so this one signed test distinguishes them. Null is not a string.
    if matches!(simple, "string" | "String") {
        let zero = b.const_i64(0);
        let is_null = b.emit(Instr::Cmp {
            dst: VReg(0),
            op: CmpOp::Eq,
            a: value,
            b: zero,
        });
        let res = b.fresh();
        let null_b = b.make_block();
        let test_b = b.make_block();
        let join_b = b.make_block();
        b.terminate(Terminator::Branch {
            cond: Some(is_null),
            if_true: null_b,
            if_false: test_b,
        });
        b.switch_to(null_b);
        b.mov(res, zero);
        b.goto(join_b);
        b.switch_to(test_b);
        let refc = b.fresh();
        b.push_raw(Instr::LoadField {
            dst: refc,
            base: value,
            off: 8,
        });
        let tagged = b.emit(Instr::Cmp {
            dst: VReg(0),
            op: CmpOp::Lt,
            a: refc,
            b: zero,
        });
        b.mov(res, tagged);
        b.goto(join_b);
        b.switch_to(join_b);
        return Ok(res);
    }
    // A native type descriptor is a `System.Type` (and a
    // `System.RuntimeType` at the API level) even though its first
    // word is TYPE_TAG rather than a vtable row. Null is not a type.
    if matches!(simple, "Type" | "RuntimeType") {
        let zero = b.const_i64(0);
        let is_null = b.emit(Instr::Cmp {
            dst: VReg(0),
            op: CmpOp::Eq,
            a: value,
            b: zero,
        });
        let res = b.fresh();
        let null_b = b.make_block();
        let test_b = b.make_block();
        let join_b = b.make_block();
        b.terminate(Terminator::Branch {
            cond: Some(is_null),
            if_true: null_b,
            if_false: test_b,
        });
        b.switch_to(null_b);
        b.mov(res, zero);
        b.goto(join_b);
        b.switch_to(test_b);
        let row = b.fresh();
        b.push_raw(Instr::LoadField {
            dst: row,
            base: value,
            off: 0,
        });
        let tag = b.const_i64(TYPE_TAG);
        let same = b.emit(Instr::Cmp {
            dst: VReg(0),
            op: CmpOp::Eq,
            a: row,
            b: tag,
        });
        b.mov(res, same);
        b.goto(join_b);
        b.switch_to(join_b);
        return Ok(res);
    }
    // An INTERFACE target has no row of its own, but that is not a
    // blocker for the test: the answer is "is this object an instance of
    // some class that implements the interface?", which expands to the
    // row addresses of exactly those classes. Interfaces still get no
    // rows of their own, but the TEST is
    // answerable, and answering it is not a guess.
    let iface_names: Vec<String> = ctx
        .table
        .interfaces
        .iter()
        .filter(|i| {
            i.name == simple
                || i.name == spelling
                // Interface names are stored as declared, which may be
                // qualified (`t.I`) while the test spells the simple name.
                || i.name.rsplit('.').next() == Some(simple)
        })
        .map(|i| i.name.clone())
        .collect();
    if !iface_names.is_empty() {
        let mut rows: Vec<u32> = Vec::new();
        for decl in ctx.classes.iter() {
            let Some(info) = ctx.table.class(&decl.name) else {
                continue;
            };
            let implements = info.interfaces.iter().any(|(iname, _)| {
                iface_names.iter().any(|n| iname == n)
                    || ctx
                        .table
                        .interface_derives_from(iname, &simple)
            });
            if implements {
                rows.push(construction_row(ctx, decl));
            }
        }
        if rows.is_empty() {
            // No class in this unit implements it, so there is no row to
            // compare against: the object cannot be one, but saying so
            // would rest on the unit being complete. Report it.
            return Err(LowerError::UnsupportedConstruct(crate::tf!(
                "rubyc.native.ir.lower_expr.interface_type_test",
                simple.to_owned()
            )));
        }
        let rowv = b.fresh();
        b.push_raw(Instr::LoadField {
            dst: rowv,
            base: value,
            off: 0,
        });
        let mut acc: Option<VReg> = None;
        for row in rows {
            let vaddr = b.fresh();
            b.push_raw(Instr::VTableAddr { dst: vaddr, row });
            let eq = b.emit(Instr::Cmp {
                dst: VReg(0),
                op: CmpOp::Eq,
                a: rowv,
                b: vaddr,
            });
            acc = Some(match acc {
                None => eq,
                Some(prev) => b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::BitOr,
                    a: prev,
                    b: eq,
                }),
            });
        }
        return Ok(acc.unwrap_or_else(|| b.const_i64(0)));
    }
    let mut rows: Vec<u32> = Vec::new();
    for decl in ctx.classes.iter() {
        let Some(info) = ctx.table.class(&decl.name) else {
            continue;
        };
        let matches = info.name == simple
            || info.key.path == spelling
            || ctx.table.derives_from(&info.name, simple);
        if matches {
            rows.push(construction_row(ctx, decl));
        }
    }
    // A PRIMITIVE target is a boxed value: refcount bit 62 marks the
    // box, and the descriptor at +24 is the boxed type. Strings and
    // class objects fail the box test; null is excluded before the
    // loads.
    if primitive_id(simple).is_some() {
        let desc = b.type_token(simple, ctx)?;
        let zero = b.const_i64(0);
        let is_null = b.emit(Instr::Cmp {
            dst: VReg(0),
            op: CmpOp::Eq,
            a: value,
            b: zero,
        });
        let res = b.fresh();
        let null_b = b.make_block();
        let test_b = b.make_block();
        let join_b = b.make_block();
        b.terminate(Terminator::Branch {
            cond: Some(is_null),
            if_true: null_b,
            if_false: test_b,
        });
        b.switch_to(null_b);
        b.mov(res, zero);
        b.goto(join_b);
        b.switch_to(test_b);
        let refc = b.fresh();
        b.push_raw(Instr::LoadField { dst: refc, base: value, off: 8 });
        let flag = b.const_i64(BOX_FLAG);
        let flagged = b.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::BitAnd,
            a: refc,
            b: flag,
        });
        let has_box = b.emit(Instr::Cmp {
            dst: VReg(0),
            op: CmpOp::Ne,
            a: flagged,
            b: zero,
        });
        let objdesc = b.fresh();
        b.push_raw(Instr::LoadField { dst: objdesc, base: value, off: 24 });
        let same = b.emit(Instr::Cmp {
            dst: VReg(0),
            op: CmpOp::Eq,
            a: objdesc,
            b: desc,
        });
        let ok = b.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::BitAnd,
            a: has_box,
            b: same,
        });
        b.mov(res, ok);
        b.goto(join_b);
        b.switch_to(join_b);
        return Ok(res);
    }
    // A PRIMITIVE target needs a runtime tag. Strings are handled above:
    // the high bit of their refcount word is the tag. Primitives are
    // still stored unboxed, so there is nothing to compare and the gap
    // is named rather than guessed (a length/address heuristic answered
    // differently on the VM and the static executable once).
    if rows.is_empty() {
        // A PRIMITIVE target (`o is int`) has no class entry and no
        // boxed tag to test: values of primitive types are stored
        // unboxed, so their runtime type is not recoverable. Name the
        // feature instead of the generic "no type" message.
        if is_primitive_spelling(simple) {
            return Err(LowerError::UnsupportedConstruct(format!(
                "`{spelling}` is a PRIMITIVE type pattern: values of primitive types are unboxed here, so there is no runtime tag to test. Use a static-type test (`x is not null`) or a class/interface target."
            )));
        }
        // A DELEGATE type: the value carries its runtime type in the
        // delegate's type word, so the test compares that descriptor
        // with the target's. A value that is not a delegate at all (a
        // class instance) carries its own runtime type there, and the
        // comparison is false - which is the right answer.
        if ctx.table.is_delegate(&spelling) {
            let desc = b.type_token(&spelling, ctx)?;
            let zero = b.const_i64(0);
            let is_null = b.emit(Instr::Cmp {
                dst: VReg(0),
                op: CmpOp::Eq,
                a: value,
                b: zero,
            });
            let res = b.fresh();
            let null_b = b.make_block();
            let test_b = b.make_block();
            let unset_b = b.make_block();
            let compare_b = b.make_block();
            let join_b = b.make_block();
            b.terminate(Terminator::Branch {
                cond: Some(is_null),
                if_true: null_b,
                if_false: test_b,
            });
            b.switch_to(null_b);
            b.mov(res, zero);
            b.goto(join_b);
            b.switch_to(test_b);
            let actual = b.fresh();
            b.push_raw(Instr::LoadField {
                dst: actual,
                base: value,
                off: 16,
            });
            let is_unset = b.emit(Instr::Cmp {
                dst: VReg(0),
                op: CmpOp::Eq,
                a: actual,
                b: zero,
            });
            b.terminate(Terminator::Branch {
                cond: Some(is_unset),
                if_true: unset_b,
                if_false: compare_b,
            });
            // The descriptors match, or they do not.
            b.switch_to(compare_b);
            let same = b.emit(Instr::Cmp {
                dst: VReg(0),
                op: CmpOp::Eq,
                a: actual,
                b: desc,
            });
            b.mov(res, same);
            b.goto(join_b);
            // A delegate whose type was never set - created where the
            // target type is not known - has no runtime type to
            // compare. Say so rather than answer "no" to a test that
            // may well be "yes".
            b.switch_to(unset_b);
            let ex = eval(
                &Expr::New {
                    class_name: "System.NotSupportedException".to_owned(),
                    args: Vec::new(),
                    init: Vec::new(),
                    span: Default::default(),
                },
                b,
                ctx,
            )?;
            b.terminate(Terminator::Unwind { src: ex });
            b.switch_to(join_b);
            return Ok(res);
        }
        return Err(LowerError::UnsupportedConstruct(crate::tf!(
            "rubyc.native.ir.lower_expr.unknown_type_test",
            spelling.to_owned()
        )));
    }
    let rowv = b.fresh();
    b.push_raw(Instr::LoadField {
        dst: rowv,
        base: value,
        off: 0,
    });
    let mut acc: Option<VReg> = None;
    for row in rows {
        let vaddr = b.fresh();
        b.push_raw(Instr::VTableAddr { dst: vaddr, row });
        let eq = b.emit(Instr::Cmp {
            dst: VReg(0),
            op: CmpOp::Eq,
            a: rowv,
            b: vaddr,
        });
        acc = Some(match acc {
            None => eq,
            Some(prev) => b.emit(Instr::Bin {
                dst: VReg(0),
                op: IrBinOp::BitOr,
                a: prev,
                b: eq,
            }),
        });
    }
    Ok(acc.unwrap_or_else(|| b.const_i64(0)))
}

/// Evaluate one `is` pattern against an ALREADY-EVALUATED subject.
///
/// The subject is computed once by the caller and reused here, so a
/// pattern chain cannot duplicate its side effects. A type pattern goes
/// through `emit_type_test`; a constant pattern compares values.
fn emit_is_pattern(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    a: VReg,
    subject_ty: &str,
    rhs: &Expr,
) -> Result<VReg, LowerError> {
    // `is not null`: the parser marks the shape with `__rubyc_nonnull`
    // (the subject is the `is` expression's left side), so the test is
    // "the subject is not null".
    if let Expr::Call { callee, .. } = rhs
        && callee == "__rubyc_nonnull"
    {
        let zero = b.const_i64(0);
        return Ok(b.emit(Instr::Cmp {
            dst: VReg(0),
            op: CmpOp::Ne,
            a,
            b: zero,
        }));
    }
    // List pattern: `[p0, p1, ..]`. Arrays store their length at +0 and
    // elements at +16 with an 8-byte stride.
    if let Expr::Call { callee, args, .. } = rhs
        && callee == "__rubyc_list"
        && !args.is_empty()
    {
        let has_rest = matches!(
            &args[0],
            Expr::Literal(crate::models::Literal::Bool(true))
        );
        let pats = &args[1..];
        let n = pats.len() as i64;
        // Element stride comes from the SUBJECT's element type: `int[]`
        // stores 4-byte elements, so a hardcoded 8 read past the value
        // (`[10, 20]` failed on a two-element array while `a[1]` was
        // right). Tuple-shaped subjects keep the 8-byte slot.
        let elem_ty = subject_ty.trim().trim_end_matches("[]");
        let width = if elem_ty.is_empty() || subject_ty.contains("tuple") {
            8
        } else {
            crate::native::ir::lower::span::span_elem_width(elem_ty)
        } as u32;
        let len = b.fresh();
        b.push_raw(Instr::LoadField { dst: len, base: a, off: 0 });
        let want = b.const_i64(n);
        let mut acc = b.emit(Instr::Cmp {
            dst: VReg(0),
            op: if has_rest { CmpOp::Ge } else { CmpOp::Eq },
            a: len,
            b: want,
        });
        for (i, pat) in pats.iter().enumerate() {
            let elem = b.fresh();
            b.push_raw(Instr::LoadField {
                dst: elem,
                base: a,
                off: 16 + width * i as u32,
            });
            // Dense elements are packed (`int[]` is 4 bytes each); an
            // 8-byte load picks up the neighbour's bytes. Narrow and
            // sign-extend exactly as the index path does — unsigned
            // elements mask, signed ones mask then sign-extend.
            let elem = if let Some(mask) = narrow_load_mask(elem_ty) {
                let m = b.const_i64(mask);
                b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::BitAnd,
                    a: elem,
                    b: m,
                })
            } else if let Some(bits) = signed_narrow_bits(elem_ty) {
                let mask = b.const_i64((1i64 << bits) - 1);
                let narrowed = b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::BitAnd,
                    a: elem,
                    b: mask,
                });
                let sh = b.const_i64((64 - bits) as i64);
                let up = b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::Shl,
                    a: narrowed,
                    b: sh,
                });
                b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::Shr,
                    a: up,
                    b: sh,
                })
            } else {
                elem
            };
            let t = emit_is_pattern(b, ctx, elem, elem_ty, pat)?;
            acc = b.emit(Instr::Bin {
                dst: VReg(0),
                op: IrBinOp::BitAnd,
                a: acc,
                b: t,
            });
        }
        return Ok(acc);
    }
    // Tuple pattern: `(p0, p1, ...)`. Tuples lower as dense arrays
    // (element i at +16 + i*8), so each element is tested in place.
    if let Expr::Call { callee, args, .. } = rhs
        && callee == "__rubyc_tuple"
    {
        let mut acc: Option<VReg> = None;
        for (i, pat) in args.iter().enumerate() {
            let elem = b.fresh();
            b.push_raw(Instr::LoadField {
                dst: elem,
                base: a,
                off: 16 + 8 * i as u32,
            });
            let t = emit_is_pattern(b, ctx, elem, subject_ty, pat)?;
            acc = Some(match acc {
                None => t,
                Some(prev) => b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::BitAnd,
                    a: prev,
                    b: t,
                }),
            });
        }
        return Ok(acc.unwrap_or_else(|| b.const_i64(1)));
    }
    // Property pattern member: `{ Length: > 0 }`. The member is read
    // from the subject VALUE (not by re-evaluating the subject, which
    // would duplicate side effects), through a scoped temp carrying the
    // subject's static type so member resolution works.
    if let Expr::Call { callee, args, .. } = rhs
        && callee == "__rubyc_prop"
        && args.len() == 2
    {
        let Expr::Literal(crate::models::Literal::Str(member)) = &args[0] else {
            return Err(LowerError::UnsupportedConstruct(crate::tf!(
                "rubyc.native.ir.lower_expr.unsupported_pattern",
                format!("{rhs:?}")
            )));
        };
        let nz = {
            let zero = b.const_i64(0);
            b.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Ne, a, b: zero })
        };
        b.scope_enter();
        let slot = b
            .scopes
            .declare("$prop_pat", subject_ty, crate::models::Access::Default)
            .map_err(LowerError::UnsupportedConstruct)?;
        while b.bindings.len() <= slot {
            let v = b.fresh();
            b.bindings.push(v);
        }
        b.mov(b.bindings[slot], a);
        let member_expr = Expr::Member {
            object: Box::new(Expr::Ident {
                name: "$prop_pat".to_owned(),
                span: crate::models::Span::new(0, 0),
            }),
            name: member.clone(),
            span: crate::models::Span::new(0, 0),
        };
        let mv = eval(&member_expr, b, ctx)?;
        let inner = emit_is_pattern(b, ctx, mv, subject_ty, &args[1])?;
        b.scope_exit();
        return Ok(b.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::BitAnd,
            a: nz,
            b: inner,
        }));
    }
    if let Expr::Binary {
        op: crate::models::BinOp::Or | crate::models::BinOp::And,
        lhs: p1,
        rhs: p2,
        ..
    } = rhs
    {
        let l = emit_is_pattern(b, ctx, a, subject_ty, p1)?;
        let r = emit_is_pattern(b, ctx, a, subject_ty, p2)?;
        let op = if matches!(rhs, Expr::Binary { op: crate::models::BinOp::Or, .. }) {
            IrBinOp::BitOr
        } else {
            IrBinOp::BitAnd
        };
        return Ok(b.emit(Instr::Bin { dst: VReg(0), op, a: l, b: r }));
    }
    if let Expr::Ident { name, .. } = rhs {
        let simple = name.rsplit('.').next().unwrap_or(name).to_owned();
        // A test on a value-typed SUBJECT is a static question: the
        // value can never carry a different runtime type.
        if primitive_id(&simple).is_some()
            && !ctx.is_type_var(subject_ty)
            && is_value_type_spelling(ctx, subject_ty)
        {
            let same = base_type_name(subject_ty)
                == canonical_primitive(&simple);
            return Ok(b.const_i64(same as i64));
        }
        // A DISCARD pattern (`is _`) matches every value.
        if simple == "_" {
            return Ok(b.const_i64(1));
        }
        // `x is object`: every value is an object, and only a null
        // reference is not one. A value type is never null.
        if simple == "object" || simple == "Object" {
            let nullable = subject_ty.trim_end().ends_with('?');
            if !nullable
                && !ctx.is_type_var(subject_ty)
                && is_value_type_spelling(ctx, subject_ty)
            {
                return Ok(b.const_i64(1));
            }
            let zero = b.const_i64(0);
            return Ok(b.emit(Instr::Cmp {
                dst: VReg(0),
                op: CmpOp::Ne,
                a,
                b: zero,
            }));
        }
        // A TYPE PARAMETER target: answer from the value's runtime
        // type descriptor. A reference carries one in its header
        // (RUNTIME_TYPE_SLOT); compare it to the type parameter's
        // descriptor. A null value is never an instance of T. Value
        // types are unboxed and carry no descriptor, so they keep
        // failing loudly (the feature is not implemented for them).
        if ctx.is_type_var(&simple) {
            if is_value_type_spelling(ctx, subject_ty) {
                let t_desc = b.type_token(&simple, ctx)?;
                let subj_desc = b.type_token(subject_ty, ctx)?;
                let obj_desc = b.type_token("object", ctx)?;
                let vt_desc = b.type_token("ValueType", ctx)?;
                let c1 = b.emit(Instr::Cmp {
                    dst: VReg(0),
                    op: CmpOp::Eq,
                    a: t_desc,
                    b: subj_desc,
                });
                let c2 = b.emit(Instr::Cmp {
                    dst: VReg(0),
                    op: CmpOp::Eq,
                    a: t_desc,
                    b: obj_desc,
                });
                let c3 = b.emit(Instr::Cmp {
                    dst: VReg(0),
                    op: CmpOp::Eq,
                    a: t_desc,
                    b: vt_desc,
                });
                let r12 = b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::BitOr,
                    a: c1,
                    b: c2,
                });
                let r = b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::BitOr,
                    a: r12,
                    b: c3,
                });
                return Ok(r);
            }
            let zero = b.const_i64(0);
            let is_null = b.emit(Instr::Cmp {
                dst: VReg(0),
                op: CmpOp::Eq,
                a,
                b: zero,
            });
            let null_b = b.make_block();
            let test_b = b.make_block();
            let join_b = b.make_block();
            let res = b.fresh();
            b.terminate(Terminator::Branch {
                cond: Some(is_null),
                if_true: null_b,
                if_false: test_b,
            });
            b.switch_to(null_b);
            b.mov(res, zero);
            b.goto(join_b);
            b.switch_to(test_b);
            let rt = b.fresh();
            b.push_raw(Instr::LoadField {
                dst: rt,
                base: a,
                off: (RUNTIME_TYPE_SLOT * 8) as u32,
            });
            let t_desc = b.type_token(&simple, ctx)?;
            let eq = b.emit(Instr::Cmp {
                dst: VReg(0),
                op: CmpOp::Eq,
                a: rt,
                b: t_desc,
            });
            b.mov(res, eq);
            b.goto(join_b);
            b.switch_to(join_b);
            return Ok(res);
        }
        return emit_type_test(b, ctx, a, &simple, name);
    }
    // A qualified type pattern parses as a member expression
    // (`is System.Type`). Constants use the same shape (`is Color.Red`),
    // so only names that resolve to a declared type or a known
    // framework type take the type-test path.
    if let Expr::Member { .. } = rhs
        && let Some(spelled) = dotted_spelling(rhs)
    {
        let simple = spelled.rsplit('.').next().unwrap_or(&spelled);
        let simple = simple.split('<').next().unwrap_or(simple);
        let known = ctx.table.class(&spelled).is_some()
            || is_value_type_spelling(ctx, &spelled)
            || matches!(
                simple,
                "Type" | "RuntimeType" | "String" | "Object" | "Array" | "ValueType"
                    | "Enum" | "Delegate"
            );
        if known {
            return emit_type_test(b, ctx, a, simple, &spelled);
        }
    }
    // Relational pattern: `> 0`, `>= X`, `<= Y`. The parser records the
    // operator with a zero placeholder on the left; the real left side
    // is the subject being tested.
    if let Expr::Binary {
        op: op @ (crate::models::BinOp::Lt
        | crate::models::BinOp::Gt
        | crate::models::BinOp::Le
        | crate::models::BinOp::Ge),
        lhs,
        rhs: value,
        ..
    } = rhs
        && matches!(
            lhs.as_ref(),
            Expr::Literal(crate::models::Literal::Int(0))
        )
    {
        let cmp = match op {
            crate::models::BinOp::Lt => CmpOp::Lt,
            crate::models::BinOp::Gt => CmpOp::Gt,
            crate::models::BinOp::Le => CmpOp::Le,
            _ => CmpOp::Ge,
        };
        let v = eval(value, b, ctx)?;
        return Ok(b.emit(Instr::Cmp { dst: VReg(0), op: cmp, a, b: v }));
    }
    let v = eval(rhs, b, ctx)?;
    Ok(b.emit(Instr::Cmp {
        dst: VReg(0),
        op: CmpOp::Eq,
        a,
        b: v,
    }))
}

/// Names referenced by a lambda body, for the capture check.
///
/// Deliberately shallow-but-broad: every expression shape that can hold an
/// identifier is walked, because a missed one would let a captured local
/// lower as an unknown name and answer 0 — a wrong value, which is worse
/// than the report this feeds.
fn body_idents(e: &Expr, out: &mut Vec<String>) {
    match e {
        Expr::NamedArg { value, .. } => body_idents(value, out),
        Expr::Ident { name, .. } => out.push(name.clone()),
        Expr::Checked { inner, .. } => body_idents(inner, out),
        Expr::Binary { lhs, rhs, .. } => {
            body_idents(lhs, out);
            body_idents(rhs, out);
        }
        Expr::Unary { inner, .. } => body_idents(inner, out),
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            body_idents(cond, out);
            body_idents(then_expr, out);
            body_idents(else_expr, out);
        }
        Expr::Call { callee, args, .. } => {
            if !callee.contains('.') && !callee.contains('<') {
                out.push(callee.clone());
            }
            for a in args {
                body_idents(a, out);
            }
        }
        Expr::New { args, init, .. } => {
            for a in args {
                body_idents(a, out);
            }
            for m in init {
                body_idents(&m.value, out);
            }
        }
        Expr::Member { object, .. } => body_idents(object, out),
        Expr::MethodCall { object, args, .. } => {
            body_idents(object, out);
            for a in args {
                body_idents(a, out);
            }
        }
        Expr::Index { object, index, .. } => {
            body_idents(object, out);
            body_idents(index, out);
        }
        Expr::Cast { expr, .. } | Expr::Await { inner: expr, .. } => body_idents(expr, out),
        Expr::Assign { target, value, .. } | Expr::AssignComplex { lhs: target, value, .. } => {
            body_idents(target, out);
            body_idents(value, out);
        }
        Expr::PostInc { target, .. } | Expr::PostDec { target, .. } => body_idents(target, out),
        Expr::NewArray { size, init, .. } => {
            body_idents(size, out);
            for e in init {
                body_idents(e, out);
            }
        }
        Expr::SwitchExpr { subject, arms, .. } => {
            body_idents(subject, out);
            for a in arms {
                body_idents(&a.body, out);
            }
        }
        Expr::InterpolatedString { parts, .. } => {
            for p in parts {
                if let crate::models::InterpPart::Hole(h) = p {
                    body_idents(h, out);
                }
            }
        }
        Expr::Lambda { body, .. } => match body {
            crate::models::ast::expression::LambdaBody::Expr(inner) => body_idents(inner, out),
            crate::models::ast::expression::LambdaBody::Block(b) => {
                for st in &b.stmts {
                    body_idents_stmt(st, out);
                }
            }
        },
        Expr::Literal(_) | Expr::Default { .. } => {}
    }
}

fn body_idents_stmt(st: &crate::models::Stmt, out: &mut Vec<String>) {
    use crate::models::Stmt;
    match st {
        Stmt::Let { init, .. } => body_idents(init, out),
        Stmt::Return { value, .. } => {
            if let Some(v) = value {
                body_idents(v, out);
            }
        }
        Stmt::Expression(e) => body_idents(e, out),
        Stmt::Block { body, .. } => {
            for s in &body.stmts {
                body_idents_stmt(s, out);
            }
        }
        Stmt::If {
            cond,
            then_block,
            else_block,
            ..
        } => {
            body_idents(cond, out);
            for s in &then_block.stmts {
                body_idents_stmt(s, out);
            }
            if let Some(b) = else_block {
                for s in &b.stmts {
                    body_idents_stmt(s, out);
                }
            }
        }
        Stmt::While { cond, body, .. } => {
            body_idents(cond, out);
            for s in &body.stmts {
                body_idents_stmt(s, out);
            }
        }
        Stmt::Try { body, catches, finally_block, .. } => {
            for s in &body.stmts {
                body_idents_stmt(s, out);
            }
            for c in catches {
                for s in &c.body.stmts {
                    body_idents_stmt(s, out);
                }
            }
            if let Some(fb) = finally_block {
                for s in &fb.stmts {
                    body_idents_stmt(s, out);
                }
            }
        }
        Stmt::For {
            init,
            cond,
            step,
            body,
            ..
        } => {
            if let Some(s) = init {
                body_idents_stmt(s, out);
            }
            if let Some(c) = cond {
                body_idents(c, out);
            }
            if let Some(s) = step {
                body_idents_stmt(s, out);
            }
            for s in &body.stmts {
                body_idents_stmt(s, out);
            }
        }
        Stmt::DoWhile { body, cond, .. } => {
            for s in &body.stmts {
                body_idents_stmt(s, out);
            }
            body_idents(cond, out);
        }
        Stmt::Switch {
            subject,
            arms,
            default_body,
            ..
        } => {
            body_idents(subject, out);
            for a in arms {
                for s in &a.body.stmts {
                    body_idents_stmt(s, out);
                }
            }
            if let Some(b) = default_body {
                for s in &b.stmts {
                    body_idents_stmt(s, out);
                }
            }
        }
        _ => {}
    }
}

/// Evaluate an expression to a vreg.
pub(crate) fn eval(expr: &Expr, b: &mut Body<'_>, ctx: &Ctx<'_>) -> Result<VReg, LowerError> {
    match expr {
        // A named argument reaches `eval` only when the call shape did
        // not rewrite the list into parameter order (the main shapes do;
        // see `positional_args`). Evaluating the value keeps the old
        // positional behaviour rather than dropping or skipping it.
        Expr::NamedArg { value, .. } => eval(value, b, ctx),
        // `checked(expr)`: the same expression, with overflow checks armed.
        // The flag lives on the body so nested arithmetic sees it; it is
        // not a property of the AST node alone.
        Expr::Checked { inner, .. } => {
            b.checked_depth += 1;
            let v = eval(inner, b, ctx);
            b.checked_depth -= 1;
            v
        }
        Expr::Lambda { params, body, span } => {
            // A lambda becomes a real function, lifted out of this body
            // and emitted right after it, plus a delegate object holding
            // that function's ADDRESS. The index is predictable: this
            // method takes the next slot, then each lambda met in order
            // takes the one after.
            let Some(queue) = ctx.lambda_deferred else {
                // No enclosing method (a field initializer or a test):
                // there is nowhere to lift to, which is a real limit.
                return Err(LowerError::UnsupportedConstruct(crate::tf!(
                    "rubyc.native.ir.lower_expr.lambda_unavailable"
                )));
            };
            // Captures: a free variable is captured BY REFERENCE, not by
            // value. Its address goes into the delegate object and the
            // lifted body reads and writes THROUGH that address, so a
            // later mutation of the captured variable is visible inside
            // the closure — what C# requires, and what the addressable-
            // local machinery (`&x` re-binds a local as by-reference over
            // a cell) already provides.
            let mut names: Vec<String> = Vec::new();
            match body {
                crate::models::ast::expression::LambdaBody::Expr(e) => {
                    body_idents(e, &mut names)
                }
                crate::models::ast::expression::LambdaBody::Block(b) => {
                    for st in &b.stmts {
                        body_idents_stmt(st, &mut names);
                    }
                }
            }
            let mut captures: Vec<String> = Vec::new();
            for n in names {
                if params.contains(&n) || n == "this" || n == "_" {
                    continue;
                }
                if b.scopes.lookup(&n).is_some() && !captures.contains(&n) {
                    captures.push(n);
                }
            }
            let obj = b.fresh();
            // Layout: [fn_addr, kind, type, cap0, cap1, ...]. `kind` is 1
            // for a lifted lambda, whose function takes the delegate
            // object as its first parameter so it can reach the
            // captures; 0 for a method group, which is called with its
            // own arguments only. `type` is the delegate's runtime
            // type — the descriptor of the instantiation the value
            // belongs to, so `x is SomeDelegate` can ask.
            b.push_raw(Instr::Alloc {
                dst: obj,
                bytes: 24 + 8 * captures.len() as u64,
            });
            // The delegate's runtime type: the descriptor of the type
            // this value is converted to. The binding site knows it;
            // without one the word stays null and a type test says so.
            if let Some(expected) = b.expected_delegate.clone() {
                if ctx.table.is_delegate(&expected) {
                    let desc = b.type_token(&expected, ctx)?;
                    b.push_raw(Instr::StoreField {
                        base: obj,
                        off: 16,
                        src: desc,
                    });
                }
            }
            for (k, cname) in captures.iter().enumerate() {
                let slot = b
                    .scopes
                    .lookup(cname)
                    .map(|sym| (sym.slot, sym.is_byref))
                    .expect("capture resolved above");
                let (slot, already_byref) = slot;
                let addr = if already_byref {
                    b.bindings[slot]
                } else {
                    let home = b.bindings[slot];
                    let cell = b.fresh();
                    b.push_raw(Instr::Alloc { dst: cell, bytes: 16 });
                    b.push_raw(Instr::StoreField {
                        base: cell,
                        off: 0,
                        src: home,
                    });
                    b.bindings[slot] = cell;
                    b.scopes.make_byref(cname);
                    cell
                };
                b.push_raw(Instr::StoreField {
                    base: obj,
                    off: 24 + 8 * k as u32,
                    src: addr,
                });
            }
            let n = queue.borrow().len() as u32;
            queue.borrow_mut().push(crate::native::ir::LambdaDeferred {
                params: params.clone(),
                body: body.clone(),
                span: *span,
                captures,
            });
            // The enclosing function's index is `functions.len() + 1` at
            // the time it was started, and lambdas follow it. The caller
            // (`ir/mod.rs`) records the base for us through the queue's
            // length at method entry (0), so index = base + 1 + n where
            // base is supplied by the emission loop's ordering.
            let index = ctx.lambda_base + 1 + n;
            let fa = b.fresh();
            b.push_raw(Instr::FnAddr { dst: fa, fidx: index });
            b.push_raw(Instr::StoreField {
                base: obj,
                off: 0,
                src: fa,
            });
            let one = b.const_i64(1);
            b.push_raw(Instr::StoreField {
                base: obj,
                off: 8,
                src: one,
            });
            Ok(obj)
        }
        Expr::NewArray { elem_type, size, init, .. } => {
            // Allocate on heap via ALLOC helper; store size at [obj+0].
            // Layout matches every other heap object: [0..8)=length,
            // [8..16)=refcount (written by the allocator), elements
            // from slot 2 — element `i` lives at `base + 16 + i*w`
            // where `w` is the element width (dense: `byte` packs 1
            // apart, matching pointer arithmetic on `fixed` buffers;
            // references/unknowns keep the 8-byte slot).
            // (Elements used to start at +8, overlapping the
            // allocator's refcount mark: element 0 read back `1`.)
            // Overlapping 8-byte stores with dense offsets are
            // exact: each element's low `w` bytes are written by its
            // own store and never touched by neighbors; reads mask
            // to `w` bytes. The tail slop (+8) keeps the last
            // element's 8-byte store inside the allocation.
            let w = span_elem_width(elem_type) as i64;
            let n_vreg = eval(size, b, ctx)?;
            // Size the allocation to the dense layout below
            // (`16 + n*w + 8`): the old fixed 4096-byte alloc
            // overflowed the heap for any array past ~510 elements
            // (e.g. the 2530-byte HtmlEntities table), corrupting
            // whatever the allocator placed next and faulting far
            // downstream.
            let wv_alloc = b.const_i64(w);
            let nw = b.emit(Instr::Bin {
                dst: VReg(0),
                op: IrBinOp::Mul,
                a: n_vreg,
                b: wv_alloc,
            });
            let pad = b.const_i64(24);
            let total = b.emit(Instr::Bin {
                dst: VReg(0),
                op: IrBinOp::Add,
                a: nw,
                b: pad,
            });
            let obj = b.fresh();
            b.push_raw(Instr::AllocDyn {
                dst: obj,
                bytes: total,
            });
            // Store length at offset 0 for bounds checking later
            b.push_raw(Instr::StoreField { base: obj, off: 0, src: n_vreg });
            // Collection-expression / array-initializer elements: store
            // in order at base+16+i*w (dense, matching the
            // element-read path). Side effects evaluate in order.
            for (i, e) in init.iter().enumerate() {
                let v = eval(e, b, ctx)?;
                let src_ty = static_type_of(e, b, ctx)
                    .unwrap_or_else(|_| "object".to_owned());
                let v = if Body::is_object_target(elem_type, ctx)
                    && primitive_id(&base_type_name(&src_ty)).is_some()
                {
                    b.box_value(v, &src_ty, ctx)?
                } else {
                    v
                };
                let off = b.const_i64(16 + i as i64 * w);
                let addr = b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::Add,
                    a: obj,
                    b: off,
                });
                b.push_raw(Instr::StoreField { base: addr, off: 0, src: v });
            }
            Ok(obj)
        }
        Expr::Index { object, index, extra, span: _, .. } => {
            // String char read: strings are UTF-16 (`build_utf16_string`
            // stores `char_len` units at +16), so char `i` is the
            // zero-extended u16 at `[base + 16 + i*2]`. Spelling-agnostic
            // (`string`/`String`/`System.String`): an exact match once
            // routed `String`-spelled values down the 8-byte array path,
            // reading refcounts as chars.
            if is_string_type(&static_type_of(object, b, ctx)?) {
                let base = eval(object, b, ctx)?;
                let idx = eval(index, b, ctx)?;
                let two = b.const_i64(2);
                let idx2 = b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::Mul,
                    a: idx,
                    b: two,
                });
                let s16 = b.const_i64(16);
                let d16 = b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::Add,
                    a: base,
                    b: s16,
                });
                let addr = b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::Add,
                    a: d16,
                    b: idx2,
                });
                let raw = b.fresh();
                b.push_raw(Instr::LoadField {
                    dst: raw,
                    base: addr,
                    off: 0,
                });
                let mask = b.const_i64(0xFFFF);
                let result = b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::BitAnd,
                    a: raw,
                    b: mask,
                });
                return Ok(result);
            }
            // Span element read over a packed buffer (see `span.rs`):
            // zero-extended `w`-byte load at `ref + i*w`.
            let recv_ty = static_type_of(object, b, ctx)?;
            if let Some((_, inner)) = span_view(&recv_ty) {
                let base = eval(object, b, ctx)?;
                let idx = eval(index, b, ctx)?;
                return span_index(b, ctx, base, &recv_ty, &inner, idx);
            }
            // Raw pointer element read (`p[i]` where `p: T*`, as from
            // `fixed`): addresses count from zero with the element
            // width — NO object header (the array path's +16 would
            // read past the data) and NO 8-byte stride (`char*`
            // advances 2 bytes, `byte*` 1). Narrow loads zero-extend
            // (mirrors `span_index`).
            {
                let t = recv_ty.trim();
                if is_pointer_ty(t) {
                    let inner = t.trim_end_matches(['*', ' ', '?']);
                    let w = span_elem_width(inner);
                    let base = eval(object, b, ctx)?;
                    let idx = eval(index, b, ctx)?;
                    let ptr = if w == 8 {
                        let eight = b.const_i64(8);
                        let idx8 = b.emit(Instr::Bin {
                            dst: VReg(0),
                            op: IrBinOp::Mul,
                            a: idx,
                            b: eight,
                        });
                        b.emit(Instr::Bin {
                            dst: VReg(0),
                            op: IrBinOp::Add,
                            a: base,
                            b: idx8,
                        })
                    } else {
                        let wv = b.const_i64(w as i64);
                        let off = b.emit(Instr::Bin {
                            dst: VReg(0),
                            op: IrBinOp::Mul,
                            a: idx,
                            b: wv,
                        });
                        b.emit(Instr::Bin {
                            dst: VReg(0),
                            op: IrBinOp::Add,
                            a: base,
                            b: off,
                        })
                    };
                    let raw = b.fresh();
                    b.push_raw(Instr::LoadField {
                        dst: raw,
                        base: ptr,
                        off: 0,
                    });
                    if w < 8 {
                        let mask = b.const_i64((1i64 << (w * 8)) - 1);
                        return Ok(b.emit(Instr::Bin {
                            dst: VReg(0),
                            op: IrBinOp::BitAnd,
                            a: raw,
                            b: mask,
                        }));
                    }
                    return Ok(raw);
                }
            }
            // Dynamic array element read. Layout (see `NewArray`):
            // offset 0 holds the length header, offset 8 the
            // allocator refcount, elements start at offset 16 with
            // the element-width stride, so element `i` lives at
            // `base + 16 + i*w` (dense for primitives, so `fixed`
            // pointer arithmetic and indexing agree). Narrow loads
            // zero-extend (mirrors the pointer path). The length
            // header (`[base+0]`) is bounds-checked above (managed
            // `IndexOutOfRangeException` on violation).
            // Indexer read on a class receiver (`dict[k]`): route
            // through `get_Item` (mirrors the `set_Item` desugar in
            // the assign arm); the raw store below would read object
            // memory as elements.
            // Indexer read on a class receiver (`dict[k]`): inline
            // the getter (mirrors the setter path in the assign
            // arm); the raw read below would load object memory as
            // elements.
            if !is_index_raw_target(&recv_ty) && !extra.is_empty() {
                // Multi-parameter indexer (`x[i, j]`): the gate matches
                // the declared arity, and every index is passed.
                let mut indices: Vec<Expr> = vec![index.as_ref().clone()];
                indices.extend(extra.iter().cloned());
                if let Some((path, decl)) =
                    find_indexer(ctx, &recv_ty, indices.len())
                {
                    let info = ctx
                        .table
                        .class(&recv_ty)
                        .or_else(|| definition_entry(ctx.table, &recv_ty));
                    if let Some(info) = info {
                        if let Some(v) = eval_indexer_get(
                            b, ctx, info, &path, &decl, object, &indices,
                        )? {
                            return Ok(v);
                        }
                    }
                }
            }
            if !is_index_raw_target(&recv_ty) && extra.is_empty() {
                if let Some((path, decl)) = find_indexer(ctx, &recv_ty, 1) {
                    let info = ctx
                        .table
                        .class(&recv_ty)
                        .or_else(|| definition_entry(ctx.table, &recv_ty));
                    if let Some(info) = info {
                        if let Some(v) = eval_indexer_get(
                            b,
                            ctx,
                            info,
                            &path,
                            &decl,
                            object,
                            &[index.as_ref().clone()],
                        )? {
                            return Ok(v);
                        }
                    }
                }
            }
            let base = eval(object, b, ctx)?;
            let idx = eval(index, b, ctx)?;
            // Managed bounds check on true arrays (header-16 dense
            // layout, nullable-annotated included — `char[]?` fields
            // like StreamWriter's `_charBuffer` check and stride like
            // plain arrays); pointer and span-lowered receivers have
            // no length header here and keep the raw path.
            if array_elem_ty(&recv_ty).is_some() {
                emit_array_bounds_check(base, idx, b, ctx)?;
            }
            let inner = array_elem_ty(&recv_ty).unwrap_or_else(|| recv_ty.trim().to_owned());
            let w = span_elem_width(&inner);
            let wv = b.const_i64(w as i64);
            let sixteen = b.const_i64(16);
            let idxw = b.emit(Instr::Bin {
                dst: VReg(0),
                op: IrBinOp::Mul,
                a: idx,
                b: wv,
            });
            let off = b.emit(Instr::Bin {
                dst: VReg(0),
                op: IrBinOp::Add,
                a: idxw,
                b: sixteen,
            });
            let ptr = b.emit(Instr::Bin {
                dst: VReg(0),
                op: IrBinOp::Add,
                a: base,
                b: off,
            });
            let result = b.fresh();
            b.push_raw(Instr::LoadField {
                dst: result,
                base: ptr,
                off: 0,
            });
            // Zero-extend unsigned narrow loads (dense neighbors
            // share the 8-byte slot); SIGNED narrow must mask too
            // (same glued tail — an unmasked `int` read glued the
            // next element: `a[0]` read 0x42_00000041 for [65, 66])
            // then sign-extend, since stores widen into the slot.
            // Clobbering only ever touches bytes past the element's
            // own width (later slots start at higher addresses), so
            // the low bytes are intact. 64-bit and float loads stay
            // full-width.
            if let Some(mask) = narrow_load_mask(&inner) {
                let m = b.const_i64(mask);
                return Ok(b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::BitAnd,
                    a: result,
                    b: m,
                }));
            }
            if let Some(bits) = signed_narrow_bits(&inner) {
                let mask = b.const_i64((1i64 << bits) - 1);
                let narrowed = b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::BitAnd,
                    a: result,
                    b: mask,
                });
                let sh = b.const_i64((64 - bits) as i64);
                let up = b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::Shl,
                    a: narrowed,
                    b: sh,
                });
                return Ok(b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::Shr,
                    a: up,
                    b: sh,
                }));
            }
            Ok(result)
        }
        Expr::Cast { type_name, expr, .. } => {
            let v = eval(expr, b, ctx)?;
            let src_ty = static_type_of(expr, b, ctx)
                .unwrap_or_else(|_| "object".to_owned());
            let src_base = base_type_name(&src_ty);
            let dst_base = base_type_name(type_name);
            // Unboxing/boxing on an explicit cast. The boxed value is
            // already in its target representation; a following numeric
            // cast (e.g. `(int)(uint)o`) sees the unboxed static type.
            if primitive_id(&dst_base).is_some()
                && Body::is_object_target(&src_ty, ctx)
            {
                return b.unbox_value(v, type_name, true, ctx);
            }
            if Body::is_object_target(type_name, ctx)
                && primitive_id(&src_base).is_some()
            {
                return b.box_value(v, &src_ty, ctx);
            }
            // Float↔integral casts need real conversion (bit
            // patterns differ): `cvttsd2si` truncates toward zero
            // (C# unchecked semantics), `cvtsi2sd` widens exactly.
            let src_float = is_float_expr(b, ctx, expr);
            let tgt = base_type_name(type_name);
            // Pointer targets (`byte*`, `void*`, …) keep legacy
            // erasure (addresses are full-width; masking would
            // truncate them).
            if type_name.contains('*') {
                return Ok(v);
            }
            let tgt_float = matches!(tgt.as_str(), "Single" | "Double" | "float" | "double");
            let tgt_bits: Option<(u32, bool)> = match tgt.as_str() {
                "SByte" | "sbyte" => Some((8, true)),
                "Byte" | "byte" => Some((8, false)),
                "Int16" | "short" => Some((16, true)),
                "UInt16" | "ushort" | "Char" | "char" => Some((16, false)),
                "Int32" | "int" => Some((32, true)),
                "UInt32" | "uint" => Some((32, false)),
                "Int64" | "long" | "UInt64" | "ulong" | "IntPtr" | "UIntPtr" | "nint"
                | "nuint" => None,
                _ => None,
            };
            // Decimal conversions (unimplemented backing) keep legacy
            // erasure when neither float nor integral handling fires.
            if src_float && tgt_int(&tgt) {
                let dst = b.fresh();
                b.push_raw(Instr::F2I { dst, src: v });
                return Ok(dst);
            }
            if !src_float && tgt_float {
                // Convert only genuine integrals (a reference cast to
                // float is meaningless and keeps legacy erasure
                // instead of fabricating a double from address bits).
                let src_ty = base_type_name(
                    &static_type_of(expr, b, ctx).unwrap_or_else(|_| "object".to_owned()),
                );
                if is_integral(&src_ty) {
                    let dst = b.fresh();
                    b.push_raw(Instr::I2F { dst, src: v });
                    return Ok(dst);
                }
                return Ok(v);
            }
            // Integral narrowing (C# unchecked truncation): mask to
            // the target width, sign-extending for signed targets.
            // Without this, `(uint)(c - min)` keeps negative values
            // and signed comparisons mis-fire (e.g. every ASCII char
            // testing as a surrogate). 64-bit targets are identity.
            if !src_float {
                if let Some((bits, signed)) = tgt_bits {
                    let src_ty = base_type_name(
                        &static_type_of(expr, b, ctx)
                            .unwrap_or_else(|_| "object".to_owned()),
                    );
                    if is_integral(&src_ty) {
                        // `checked((T)v)` must REJECT a value that does not
                        // fit T instead of silently masking it. The range is
                        // compared in the source width, which is the same
                        // 64-bit width the value already has.
                        if b.checked_depth > 0 && bits < 64 {
                            let (min, max) = if signed {
                                let half = 1i64 << (bits - 1);
                                (-half, half - 1)
                            } else {
                                (0, ((1u64 << bits) - 1) as i64)
                            };
                            let vmin = b.const_i64(min);
                            let vmax = b.const_i64(max);
                            let lo = b.emit(Instr::Cmp {
                                dst: VReg(0),
                                op: CmpOp::Lt,
                                a: v,
                                b: vmin,
                            });
                            let hi = b.emit(Instr::Cmp {
                                dst: VReg(0),
                                op: CmpOp::Gt,
                                a: v,
                                b: vmax,
                            });
                            let ovf = b.emit(Instr::Bin {
                                dst: VReg(0),
                                op: IrBinOp::BitOr,
                                a: lo,
                                b: hi,
                            });
                            b.throw_overflow_if(ovf, ctx)?;
                        }
                        let mask = b.emit(Instr::Const {
                            dst: VReg(0),
                            imm: if bits == 64 {
                                -1i64
                            } else {
                                ((1u64 << bits) - 1) as i64
                            },
                        });
                        let narrowed = b.emit(Instr::Bin {
                            dst: VReg(0),
                            op: IrBinOp::BitAnd,
                            a: v,
                            b: mask,
                        });
                        if signed && bits < 64 {
                            let sh = b.emit(Instr::Const {
                                dst: VReg(0),
                                imm: (64 - bits) as i64,
                            });
                            let up = b.emit(Instr::Bin {
                                dst: VReg(0),
                                op: IrBinOp::Shl,
                                a: narrowed,
                                b: sh,
                            });
                            return Ok(b.emit(Instr::Bin {
                                dst: VReg(0),
                                op: IrBinOp::Shr,
                                a: up,
                                b: sh,
                            }));
                        }
                        return Ok(narrowed);
                    }
                }
            }
            Ok(v)
        }
        Expr::Default { type_name, .. } => {
            // `default(T)` for a struct with instance fields
            // materializes a zeroed box (member access on it must
            // work: `default(Range).Start`); primitives (fieldless),
            // nullables, references, and unknowns stay null/zero.
            // Spans get a valid EMPTY span object (not null): a null
            // span faults on `.Length` (slot +24), but
            // `default(ReadOnlySpan<byte>)` is a legal empty span
            // (faulted `Console.WriteLine` via UTF8-noBOM
            // `Preamble`, whose false arm is `default`).
            let tn = type_name.trim();
            if super::span::span_view(tn).is_some() {
                let z = b.const_i64(0);
                return super::span::span_object(b, ctx, tn, z, z);
            }
            if !tn.starts_with("Nullable<")
                && !tn.trim_end().ends_with('?')
                && let Some(entry) = ctx.table.class(tn).filter(|e| {
                    e.is_struct && !e.fields.is_empty()
                })
            {
                return Ok(b.alloc_struct_box(ctx, entry, Some(&tn)));
            }
            Ok(b.const_i64(0))
        }
        Expr::AssignComplex { lhs, value, .. } => {
            // Handle `member = value` assignments.
            if let Expr::Member { object, name, .. } = lhs.as_ref() {
                let type_name = static_type_of(object, b, ctx)?;
                // Method-anchored entry first (same rationale as the
                // `Stmt::MemberAssign` path above: same-named nested
                // classes collapse bare names).
                let anchored = ctx
                    .current_info
                    .filter(|info| info.field(name).is_some());
                let info_opt = if anchored.is_some() {
                    anchored
                } else {
                    member_entry(ctx.table, &type_name)
                };
                let Some(info) = info_opt else {
                    let _ = eval(object, b, ctx)?;
                    let _ = eval(value, b, ctx)?;
                    return Ok(b.fresh());
                };
                let Some(field) = info.field(name) else {
                    // Accessor-backed property write in compound position
                    // (`buf.Low64 += x`): no setter function exists yet —
                    // evaluate for side effects, yield the value.
                    if member_host(ctx.table, &type_name, name).is_some() {
                        let _ = eval(object, b, ctx)?;
                        let v = eval(value, b, ctx)?;
                        return Ok(v);
                    }
return Err(LowerError::UnsupportedConstruct(crate::tf!(
                        "rubyc.native.ir.lower_expr.no_member",
                        type_name.clone(),
                        name.clone()
                    )));
                };
                ctx.check_access(true, &type_name, name)?;
                // Static field: store into per-class static storage (address
                // + offset 0), independent of any receiver.
                if field.is_static {
                    // Evaluate the receiver for side effects only; it is not
                    // the base address of a static.
                    let _ = eval(object, b, ctx)?;
                    let v = eval(value, b, ctx)?;
                    let val_ty = static_type_of(value, b, ctx)
                        .unwrap_or_else(|_| "object".to_owned());
                    let v = b.coerce_boxing(v, &val_ty, &field.type_name, false, ctx)?;
                    let Some(base) = static_addr(field, b, ctx) else {
                        return Err(LowerError::UnsupportedConstruct(format!(
                            "static field '{}.{}' has no storage slot",
                            type_name, name
                        )));
                    };
                    b.push_raw(Instr::StoreField { base, off: 0, src: v });
                    return Ok(v);
                }
                let base = eval(object, b, ctx)?;
                let v = eval(value, b, ctx)?;
                let val_ty = static_type_of(value, b, ctx)
                    .unwrap_or_else(|_| "object".to_owned());
                let v = b.coerce_boxing(v, &val_ty, &field.type_name, false, ctx)?;
                let field_is_ref = info
                    .field(name)
                    .map(|fi| is_heap_ref(ctx.table, &fi.type_name))
                    .unwrap_or(false);
                if field_is_ref {
                    let old = b.fresh();
                    b.push_raw(Instr::LoadField {
                        dst: old,
                        base,
                        off: (field.offset_slots * 8) as u32,
                    });
                    b.emit_retain(v);
                    if let Some(fi) = info.field(name)
                        && let Some(bytes) = b.handle_bytes(&fi.type_name, ctx)
                    {
                        b.emit_release(old, bytes);
                    }
                }
                b.push_raw(Instr::StoreField {
                    base,
                    off: (field.offset_slots * 8) as u32,
                    src: v,
                });
                Ok(v)
            } else if let Expr::Index { object, index, .. } = lhs.as_ref() {
                // `arr[i] = value` — dynamic array element write. Element
                // address is `base + 16 + i*8` (length header at [0],
                // allocator refcount at [8]).
                let base = eval(object, b, ctx)?;
                let idx = eval(index, b, ctx)?;
                let v = eval(value, b, ctx)?;
                let val_ty = static_type_of(value, b, ctx)
                    .unwrap_or_else(|_| "object".to_owned());
                let arr_ty = static_type_of(object, b, ctx)
                    .unwrap_or_else(|_| "object".to_owned());
                let elem_ty = arr_ty.trim_end_matches("[]");
                let v = if Body::is_object_target(elem_ty, ctx)
                    && primitive_id(&base_type_name(&val_ty)).is_some()
                {
                    b.box_value(v, &val_ty, ctx)?
                } else {
                    v
                };
                let eight = b.const_i64(8);
                let sixteen = b.const_i64(16);
                let idx8 = b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::Mul,
                    a: idx,
                    b: eight,
                });
                let off = b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::Add,
                    a: idx8,
                    b: sixteen,
                });
                let ptr = b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::Add,
                    a: base,
                    b: off,
                });
                b.push_raw(Instr::StoreField {
                    base: ptr,
                    off: 0,
                    src: v,
                });
                Ok(v)
            } else {
                // Fallback: just evaluate the value (for non-member LHS).
                eval(value, b, ctx)
            }
        }
        // `target++` / `target--` — postfix increment/decrement. Yields
        // the OLD value and stores old+/-1. Vregs are single-assignment
        // and `Bin` only reads its operands, so the loaded `old` vreg is
        // both returned and fed to the add/sub. Store paths mirror the
        // `Assign` arms (barriers included) for each target shape.
        Expr::PostInc { target, .. } | Expr::PostDec { target, .. } => {
            let up = matches!(expr, Expr::PostInc { .. });
            let op = if up {
                IrBinOp::Add
            } else {
                IrBinOp::Sub
            };
            let old = eval(target, b, ctx)?;
            let one = b.const_i64(1);
            let newv = b.emit(Instr::Bin {
                dst: VReg(0),
                op,
                a: old,
                b: one,
            });
            match target.as_ref() {
                Expr::Ident { name, .. } => {
                    if let Some(sym) = b.scopes.lookup(name) {
                        let home = b.bindings[sym.slot];
                        let tn = b.scopes.lookup(name).map(|s| s.type_name.clone());
                        let bytes = tn.as_deref().and_then(|t| b.handle_bytes(t, ctx));
                        match bytes {
                            Some(bytes) => {
                                b.emit_retain(newv);
                                b.emit_release(home, bytes);
                                b.mov(home, newv);
                            }
                            None => b.mov(home, newv),
                        }
                    } else if let Some(info) = ctx.table.class(ctx.current_class)
                        && let Some(field) = info.field(name)
                    {
                        // Implicit-`this` field.
                        ctx.check_access(true, ctx.current_class, name)?;
                        if field.is_static {
                            let Some(base) = static_addr(field, b, ctx) else {
                                return Err(LowerError::UnsupportedConstruct(format!(
                                    "static field '{name}' has no storage slot"
                                )));
                            };
                            b.push_raw(Instr::StoreField { base, off: 0, src: newv });
                        } else {
                            let this = b.this_vreg();
                            b.push_raw(Instr::StoreField {
                                base: this,
                                off: (field.offset_slots * 8) as u32,
                                src: newv,
                            });
                        }
                    }
                }
        Expr::Member { object, name, .. } => {
            let type_name = static_type_of(object, b, ctx)?;
            // Tuple element read (`t.Item1`): tuples lower as
            // `tuple[]` arrays, so `ItemN` is slot N-1 (matches the
            // deconstruction indices in the parser).
            if name.len() > 4
                && name.starts_with("Item")
                && name[4..].chars().all(|c| c.is_ascii_digit())
            {
                // A tuple's static type is spelled `(int, int)` (or
                // `tuple[]` after inference); both denote the dense slot
                // array, so `ItemN` is slot N-1 in either spelling.
                let tr = type_name.trim();
                let is_tuple = tr == "tuple"
                    || tr
                        .strip_suffix("[]")
                        .map(|inner| inner.trim() == "tuple")
                        .unwrap_or(false)
                    || (tr.starts_with('(') && tr.contains(','));
                if is_tuple
                    && let Ok(n) = name[4..].parse::<i64>()
                    && n >= 1
                {
                    let synth = Expr::Index {
                        object: object.clone(),
                        index: Box::new(Expr::Literal(crate::models::Literal::Int(n - 1))),
                        extra: Vec::new(),
                        is_range: false,
                        span: crate::models::Span::new(0, 0),
                    };
                    return eval(&synth, b, ctx);
                }
            }
                    if let Some(info) = member_entry(ctx.table, &type_name)
                        && let Some(field) = info.field(name)
                    {
            ctx.check_access(true, &type_name, name)?;
            let base = eval(object, b, ctx)?;
                        let field_is_ref = info
                            .field(name)
                            .map(|fi| is_heap_ref(ctx.table, &fi.type_name))
                            .unwrap_or(false);
                        if field_is_ref {
                            let stale = b.fresh();
                            b.push_raw(Instr::LoadField {
                                dst: stale,
                                base,
                                off: (field.offset_slots * 8) as u32,
                            });
                            b.emit_retain(newv);
                            if let Some(fi) = info.field(name)
                                && let Some(bytes) = b.handle_bytes(&fi.type_name, ctx)
                            {
                                b.emit_release(stale, bytes);
                            }
                        }
                        b.push_raw(Instr::StoreField {
                            base,
                            off: (field.offset_slots * 8) as u32,
                            src: newv,
                        });
                    }
                }
                Expr::Index { object, index, .. } => {
                    // Statement-form `arr[i] = value`: element address
                    // is `base + 16 + i*8` (see the read path).
                    let base = eval(object, b, ctx)?;
                    let idx = eval(index, b, ctx)?;
                    let eight = b.const_i64(8);
                    let sixteen = b.const_i64(16);
                    let idx8 = b.emit(Instr::Bin {
                        dst: VReg(0),
                        op: IrBinOp::Mul,
                        a: idx,
                        b: eight,
                    });
                    let off = b.emit(Instr::Bin {
                        dst: VReg(0),
                        op: IrBinOp::Add,
                        a: idx8,
                        b: sixteen,
                    });
                    let ptr = b.emit(Instr::Bin {
                        dst: VReg(0),
                        op: IrBinOp::Add,
                        a: base,
                        b: off,
                    });
                    b.push_raw(Instr::StoreField {
                        base: ptr,
                        off: 0,
                        src: newv,
                    });
                }
                _ => {}
            }
            Ok(old)
        }
        // `target = value` — a value-assignment expression (e.g. inside a
        // ternary or as a standalone expression). Lowering the target to a
        // vreg performs the store; the result is the assigned value.
        Expr::Assign { target, value, .. } => {
            // Call-site `ref`/`in` markers (parser `&`-wrap, incl. the
            // synthesized assigns in `lower_compare_exchange`):
            // assignment targets normalize to the inner expression —
            // the address-ness lives in the operation, not the target
            // shape. Without this the store falls through every arm
            // below to evaluate-and-drop (`Interlocked.CompareExchange`
            // never stored: every lazily-initialized singleton stayed
            // null → null-virtual crashes deep in init).
            let target_eff: &Expr = match target.as_ref() {
                Expr::Unary {
                    op: UnOp::AddrOf,
                    inner,
                    ..
                } => inner,
                _ => target,
            };
            // `*p = v` — store THROUGH the pointer. This shape was
            // evaluated and dropped: the value was computed and no
            // store was emitted, so every write through a pointer
            // silently did nothing (`int x = 5; int* p = &x; *p = 8;`
            // left x at 5). The address is the pointer's value and the
            // store goes at offset 0 of it.
            if let Expr::Unary {
                op: UnOp::Deref,
                inner,
                ..
            } = target_eff
            {
                let addr = eval(inner, b, ctx)?;
                let v = eval(value, b, ctx)?;
                b.push_raw(Instr::StoreField {
                    base: addr,
                    off: 0,
                    src: v,
                });
                return Ok(v);
            }
            // Indexer target on a class receiver (`dict[k] = v`):
            // route through `set_Item` instead of the raw
            // array-element store (`base + 16 + i*8` faults on
            // objects — the HtmlEntities init crash). Synthesizing
            // the equivalent method call reuses overload
            // resolution, generics, and virtual dispatch, and the
            // value evaluates exactly once inside. Arrays, strings,
            // spans, and raw pointers keep the raw path; unknown
            // receivers keep today's leniency.
            if let Expr::Index { object, index, extra, .. } = target_eff {
                let recv_ty = static_type_of(object, b, ctx)?;
                // Indexer set on a class receiver (`dict[k] = v`):
                // inline the setter body (indexers are not table
                // methods, so no `set_Item` call exists to emit).
                if !is_index_raw_target(&recv_ty) {
                    let mut indices: Vec<Expr> = vec![index.as_ref().clone()];
                    indices.extend(extra.iter().cloned());
                    // `find_indexer` matches the INDEX-parameter count;
                    // the value is not one of them.
                    if let Some((path, decl)) =
                        find_indexer(ctx, &recv_ty, indices.len())
                    {
                        let info = ctx
                            .table
                            .class(&recv_ty)
                            .or_else(|| definition_entry(ctx.table, &recv_ty));
                        if let Some(info) = info {
                            if let Some(v) =
                                eval_indexer_set(
                                    b, ctx, info, &path, &decl, object, &indices, value,
                                )?
                            {
                                return Ok(v);
                            }
                        }
                    }
                }
            }
            let v = eval(value, b, ctx)?;
            // `_` discard: the value is evaluated for side effects, then
            // dropped (mirrors the Stmt::Assign discard arm).
            if matches!(target_eff, Expr::Ident { name, .. } if name == "_") {
                return Ok(v);
            }
            if let Expr::Ident { name, .. } = target_eff {
                // A primitive assigned to an object target is boxed
                // here too: `o = 5` as an expression takes this path.
                let target_ty = b
                    .scopes
                    .lookup(name)
                    .map(|s| s.type_name.clone())
                    .unwrap_or_default();
                let src_ty = static_type_of(value, b, ctx)
                    .unwrap_or_else(|_| "object".to_owned());
                let v = b.coerce_boxing(v, &src_ty, &target_ty, false, ctx)?;
                // A STRUCT-typed target receives a COPY, not a reference:
                // `b = a;` must not alias, or a later `b.V = 9` changes
                // `a`. Structs are heap objects here, so the copy is a
                // fresh allocation plus a byte copy (`new` is already
                // fresh, so it is left alone).
                let v = {
                    let tn = b
                        .scopes
                        .lookup(name)
                        .map(|s| s.type_name.clone())
                        .unwrap_or_default();
                    let src_is_new = matches!(value.as_ref(), Expr::New { .. });
                    match ctx.table.class(&tn) {
                        Some(info) if info.is_struct && !src_is_new => {
                            // The copy spans the fields' REAL slots (the
                            // last field ends the struct); a hardcoded
                            // field start would truncate or overrun as
                            // soon as the layout changes.
                            let last = info
                                .fields
                                .iter()
                                .map(|f| f.offset_slots)
                                .max()
                                .unwrap_or(1);
                            let bytes = (last as u64 + 1) * 8;
                            let copy = b.fresh();
                            b.push_raw(Instr::Alloc {
                                dst: copy,
                                bytes,
                            });
                            let len = b.const_i64(bytes as i64);
                            b.push_raw(Instr::MemCpy {
                                dst: copy,
                                src: v,
                                len,
                            });
                            copy
                        }
                        _ => v,
                    }
                };
                if let Some(sym) = b.scopes.lookup(name) {
                    let home = b.bindings[sym.slot];
                    // By-reference target: store through the address.
                    if sym.is_byref {
                        b.push_raw(Instr::StoreField {
                            base: home,
                            off: 0,
                            src: v,
                        });
                        return Ok(v);
                    }
                    let tn = b.scopes.lookup(name).map(|s| s.type_name.clone());
                    let bytes = tn.as_deref().and_then(|t| b.handle_bytes(t, ctx));
                    match bytes {
                        Some(bytes) => {
                            b.emit_retain(v);
                            b.emit_release(home, bytes);
                            b.mov(home, v);
                        }
                        None => b.mov(home, v),
                    }
                    return Ok(v);
                }
                // Implicit-`this` field (`_headers ??= ...` where
                // `_headers` is a field): mirror the Stmt::Assign path.
                if let Some(info) = this_info(ctx, name)
                    && let Some(field) = info.field(name)
                {
                    ctx.check_access(true, ctx.current_class, name)?;
                    let src_ty = static_type_of(value, b, ctx)
                        .unwrap_or_else(|_| "object".to_owned());
                    let v = b.coerce_boxing(v, &src_ty, &field.type_name, false, ctx)?;
                    if field.is_static {
                        let Some(base) = static_addr(field, b, ctx) else {
                            return Err(LowerError::UnsupportedConstruct(format!(
                                "static field '{}' has no storage slot",
                                name
                            )));
                        };
                        b.push_raw(Instr::StoreField { base, off: 0, src: v });
                        return Ok(v);
                    }
                    let this = b.this_vreg();
                    // Write barrier (mirrors Stmt::Assign and
                    // MemberAssign): retain the incoming reference so
                    // later releases of the source cannot free the
                    // object the field points at.
                    let off = (field.offset_slots * 8) as u32;
                    if is_heap_ref(ctx.table, &field.type_name) {
                        let old = b.fresh();
                        b.push_raw(Instr::LoadField {
                            dst: old,
                            base: this,
                            off,
                        });
                        b.emit_retain(v);
                        if let Some(bytes) = b.handle_bytes(&field.type_name, ctx) {
                            b.emit_release(old, bytes);
                        }
                    }
                    b.push_raw(Instr::StoreField {
                        base: this,
                        off,
                        src: v,
                    });
                    return Ok(v);
                }
            }
            // Member target (`this.F = v`, `o.F = v` as an expression):
            // mirror Stmt::MemberAssign — resolve the field and store.
            // Static targets write the class slot (or the owner's slot
            // via import), never instance memory.
            if let Expr::Member { object, name, .. } = target_eff {
                let type_name = static_type_of(object, b, ctx)?;
                if let Some(info) = member_entry(ctx.table, &type_name)
                    && let Some(field) = info.field(name)
                {
                    ctx.check_access(true, &type_name, name)?;
                    let src_ty = static_type_of(value, b, ctx)
                        .unwrap_or_else(|_| "object".to_owned());
                    let v = b.coerce_boxing(v, &src_ty, &field.type_name, false, ctx)?;
                    if field.is_static {
                        let _ = eval(object, b, ctx)?;
                        let Some(base) = static_addr(field, b, ctx) else {
                            return Err(LowerError::UnsupportedConstruct(format!(
                                "static field '{}.{}' has no storage slot",
                                type_name, name
                            )));
                        };
                        b.push_raw(Instr::StoreField { base, off: 0, src: v });
                        return Ok(v);
                    }
                    // Same struct-receiver rule as the read path: a
                    // null struct base materializes a zeroed box
                    // (the store into it is dropped, matching the
                    // property-write fallback — but nothing faults).
                    // Array-element receivers (`arr[i].F = v`) bind
                    // the slot's own box instead (see
                    // `index_struct_box`), so the store persists.
                    let base = match object.as_ref() {
                        Expr::Index { object: o, index: i, .. } => {
                            match index_struct_box(b, ctx, o, i)? {
                                Some(boxed) => boxed,
                                None => {
                                    let v = eval(object, b, ctx)?;
                                    ensure_struct_base(b, ctx, v, &type_name)
                                }
                            }
                        }
                        _ => {
                            let v = eval(object, b, ctx)?;
                            ensure_struct_base(b, ctx, v, &type_name)
                        }
                    };
                    let field_is_ref = info
                        .field(name)
                        .map(|fi| is_heap_ref(ctx.table, &fi.type_name))
                        .unwrap_or(false);
                    if field_is_ref {
                        let old = b.fresh();
                        b.push_raw(Instr::LoadField {
                            dst: old,
                            base,
                            off: (field.offset_slots * 8) as u32,
                        });
                        b.emit_retain(v);
                        if let Some(fi) = info.field(name)
                            && let Some(bytes) = b.handle_bytes(&fi.type_name, ctx)
                        {
                            b.emit_release(old, bytes);
                        }
                    }
                    b.push_raw(Instr::StoreField {
                        base,
                        off: (field.offset_slots * 8) as u32,
                        src: v,
                    });
                    return Ok(v);
                }
                // Accessor-backed property write through an explicit
                // receiver (`m.Flag = v` as an expression): dispatch
                // through the setter (virtual deform when overridden,
                // static inline otherwise); falls back to
                // evaluate-and-drop when no setter body resolves.
                if member_host(ctx.table, &type_name, name).is_some()
                    && let Some(rinfo) = member_entry(ctx.table, &type_name)
                {
                    // `c.P = v` on a property without a setter: C# reports
                    // CS0200. The declaration may be inherited, so the
                    // whole chain is consulted.
                    if rinfo.property(name).is_some()
                        && property_settable_in_chain(ctx, &rinfo.name, name)
                            == Some(false)
                    {
                        return Err(LowerError::UnsupportedConstruct(format!(
                            "property `{name}` has no setter"
                        )));
                    }
                    let done = if strict_subclass_has_explicit_setter(ctx, rinfo, name) {
                        eval_virtual_setter(b, ctx, rinfo, name, Some(object.as_ref()), v)?
                    } else {
                        eval_property_setter(b, ctx, rinfo, &type_name, name, Some(object.as_ref()), v)
                    };
                    if !done {
                        let _ = eval(object, b, ctx)?;
                    }
                    return Ok(v);
                }
            }
            // Array element target (`a[i] = v` as an expression): same
            // dense layout as the read path (`base + 16 + i*w`). The
            // 8-byte store overlaps neighbors' tails, which reads
            // mask away; the element's own low `w` bytes are exact.
            // Raw pointers (`p[i] = v`, as in `Narrow`) count from
            // zero with no header — mirroring the read path.
            // Span element target (`s[i] = v`): store through the
            // span's `_reference` (offsets from the table, width from
            // the inner type — mirroring `span_index`). Without this
            // the store landed in the span object's own header
            // (`spanobj + 16 + i*w`), silently dropping every
            // mutation through coerced spans (`CopyTo`/`Fill` into
            // arrays never arrived).
            if let Expr::Index { object, index, .. } = target_eff {
                let rt = static_type_of(object, b, ctx).unwrap_or("object".to_owned());
                if let Some((_, inner)) = span_view(&rt) {
                    let base = eval(object, b, ctx)?;
                    let idx = eval(index, b, ctx)?;
                    let (ref_off, _) =
                        span_field_offsets(ctx, &rt).unwrap_or((16, 24));
                    let r = b.fresh();
                    b.push_raw(Instr::LoadField {
                        dst: r,
                        base,
                        off: ref_off,
                    });
                    let w = span_elem_width(&inner);
                    let wv = b.const_i64(w as i64);
                    let idxw = b.emit(Instr::Bin {
                        dst: VReg(0),
                        op: IrBinOp::Mul,
                        a: idx,
                        b: wv,
                    });
                    let ptr = b.emit(Instr::Bin {
                        dst: VReg(0),
                        op: IrBinOp::Add,
                        a: r,
                        b: idxw,
                    });
                    b.push_raw(Instr::StoreField {
                        base: ptr,
                        off: 0,
                        src: v,
                    });
                    return Ok(v);
                }
                let tr = rt.trim();
                // Strip a trailing nullable annotation for shape
                // dispatch (`char[]?` fields index like `char[]`;
                // `char*?` like `char*`).
                let tr_nq = tr.strip_suffix('?').unwrap_or(tr);
                let (inner, header) = if tr_nq.ends_with('*') && !tr_nq.ends_with("[]") {
                    (
                        tr_nq.trim_end_matches(['*', ' ']).trim().to_owned(),
                        0i64,
                    )
                } else if let Some(e) = array_elem_ty(tr) {
                    (e, 16i64)
                } else {
                    (
                        tr.strip_suffix("[]").unwrap_or(tr).trim().to_owned(),
                        16i64,
                    )
                };
                let w = span_elem_width(&inner);
                let base = eval(object, b, ctx)?;
                let idx = eval(index, b, ctx)?;
                // Managed bounds check on true arrays (header 16);
                // raw pointers (header 0) have no length header.
                if header == 16 {
                    emit_array_bounds_check(base, idx, b, ctx)?;
                }
                let wv = b.const_i64(w as i64);
                let h = b.const_i64(header);
                let idxw = b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::Mul,
                    a: idx,
                    b: wv,
                });
                let off = b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::Add,
                    a: idxw,
                    b: h,
                });
                let ptr = b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::Add,
                    a: base,
                    b: off,
                });
                let src_ty = static_type_of(value, b, ctx)
                    .unwrap_or_else(|_| "object".to_owned());
                let v = if header == 16
                    && Body::is_object_target(&inner, ctx)
                    && primitive_id(&base_type_name(&src_ty)).is_some()
                {
                    b.box_value(v, &src_ty, ctx)?
                } else {
                    v
                };
                b.push_raw(Instr::StoreField {
                    base: ptr,
                    off: 0,
                    src: v,
                });
                return Ok(v);
            }
            // Accessor-backed property write in expression position:
            // dispatch through the setter (virtual deform when
            // overridden, static inline otherwise — mirrors the
            // statement path). Falls back to evaluate-and-drop when
            // no setter body resolves (legacy behavior).
            if let Expr::Ident { name, .. } = target_eff
                && let Some(info) = this_info(ctx, name)
            {
                if strict_subclass_has_explicit_setter(ctx, info, name) {
                    let _ = eval_virtual_setter(b, ctx, info, name, None, v)?;
                } else {
                    let _ = eval_property_setter(b, ctx, info, &info.name, name, None, v);
                }
                return Ok(v);
            }
            // Store through an address-returning call
            // (`Unsafe.Add(ref p, i) = v`, as in `NarrowTwo`): the
            // call evaluates to the destination address — store the
            // (already evaluated) value through it. The old fallback
            // below evaluated and dropped the store, losing every
            // second `GetBytes` byte.
            if let Expr::MethodCall { object, name, .. } = target_eff {
                let is_unsafe_addr = matches!(name.as_str(), "Add" | "AddByteOffset")
                    && matches!(object.as_ref(), Expr::Ident { name, .. } if name == "Unsafe");
                if is_unsafe_addr {
                    let addr = eval(target, b, ctx)?;
                    b.push_raw(Instr::StoreField {
                        base: addr,
                        off: 0,
                        src: v,
                    });
                    return Ok(v);
                }
            }
            // Fallback: evaluate the target (side effects) and the value;
            // return the value.
            let _ = eval(target, b, ctx)?;
            Ok(v)
        }
        Expr::SwitchExpr { subject, arms, .. } => {
            // Evaluate the subject ONCE, then test the arms in order.
            // This used to return the subject and ignore every arm, so a
            // switch expression answered whatever was switched on.
            let sv = eval(subject, b, ctx)?;
            let sty = static_type_of(subject, b, ctx)
                .unwrap_or_else(|_| "object".to_owned());
            let home = b.fresh();
            let end_b = b.make_block();
            let arm_blocks: Vec<crate::native::ir::BlockId> =
                arms.iter().map(|_| b.make_block()).collect();
            // Where a false GUARD for arm i must continue: the next arm's
            // test block, or (after the last arm) the no-match path.
            // A switch expression has to produce a value, so the no-match
            // edge cannot fall into the join: C# throws there. An
            // unguarded discard arm proves the edge dead and is emitted
            // as an unconditional jump instead.
            let exhaustive = arms.iter().any(|a| {
                a.guard.is_none()
                    && a.labels.iter().any(
                        |l| matches!(l, Expr::Ident { name, .. } if name == "_"),
                    )
            });
            let no_match_b = (!exhaustive).then(|| b.make_block());
            let mut miss_blocks: Vec<crate::native::ir::BlockId> =
                Vec::with_capacity(arms.len());
            for (i, arm) in arms.iter().enumerate() {
                let irrefutable = arm.guard.is_none()
                    && arm.labels.iter().any(
                        |l| matches!(l, Expr::Ident { name, .. } if name == "_"),
                    );
                if irrefutable {
                    // Always matches; arms after it are unreachable.
                    b.goto(arm_blocks[i]);
                    miss_blocks.push(arm_blocks[i]);
                    for _ in i + 1..arms.len() {
                        miss_blocks.push(end_b);
                    }
                    break;
                }
                // Each label is a pattern over the same subject; `or`
                // chains were already merged by the parser.
                let mut test: Option<VReg> = None;
                for label in &arm.labels {
                    let t = match label {
                        Expr::Ident { name, .. } if name == "_" => b.const_i64(1),
                        _ => emit_is_pattern(b, ctx, sv, &sty, label)?,
                    };
                    test = Some(match test {
                        None => t,
                        Some(prev) => b.emit(Instr::Bin {
                            dst: VReg(0),
                            op: IrBinOp::BitOr,
                            a: prev,
                            b: t,
                        }),
                    });
                }
                let test = test.unwrap_or_else(|| b.const_i64(0));
                let next_b = if i + 1 < arms.len() {
                    b.make_block()
                } else {
                    no_match_b.unwrap_or(end_b)
                };
                b.terminate(Terminator::Branch {
                    cond: Some(test),
                    if_true: arm_blocks[i],
                    if_false: next_b,
                });
                miss_blocks.push(next_b);
                if i + 1 < arms.len() {
                    b.switch_to(next_b);
                }
            }
            for (i, arm) in arms.iter().enumerate() {
                b.switch_to(arm_blocks[i]);
                // `when` guard: part of the arm's test.
                if let Some(g) = &arm.guard {
                    let gv = eval(g, b, ctx)?;
                    let body_ok = b.make_block();
                    b.terminate(Terminator::Branch {
                        cond: Some(gv),
                        if_true: body_ok,
                        if_false: miss_blocks[i],
                    });
                    b.switch_to(body_ok);
                }
                let v = eval(&arm.body, b, ctx)?;
                b.mov(home, v);
                if !b.is_sealed() {
                    b.goto(end_b);
                }
            }
            // No arm matched: a switch expression must produce a value, so
            // C# throws `SwitchExpressionException`. Returning the join
            // home here would hand back whatever the slot held.
            if let Some(nm) = no_match_b {
                b.switch_to(nm);
                let exc = Expr::New {
                    class_name: "System.Runtime.CompilerServices.SwitchExpressionException"
                        .to_owned(),
                    args: Vec::new(),
                    init: Vec::new(),
                    span: Default::default(),
                };
                let obj = eval(&exc, b, ctx)?;
                b.terminate(Terminator::Unwind { src: obj });
            }
            b.switch_to(end_b);
            Ok(home)
        }
        Expr::Literal(Literal::Str(s)) => {
            let (off, len) = b.utf16_literal(s);
            Ok(b.build_utf16_string(off, len))
        }
        Expr::InterpolatedString { parts, .. } => {
            // Constant-fold when every part is a literal text run.
            if parts.iter().all(|p| matches!(p, InterpPart::Text(_))) {
                let s: String = parts
                    .iter()
                    .filter_map(|p| if let InterpPart::Text(t) = p { Some(t.as_str()) } else { None })
                    .collect();
                let (off, len) = b.utf16_literal(&s);
                return Ok(b.build_utf16_string(off, len));
            }
            // Otherwise: fold static text into one literal, then join with
            // dynamic holes via `string` concatenation.
            let mut acc: Option<VReg> = None;
            let mut text = String::new();
            for part in parts {
                match part {
                    InterpPart::Text(t) => text.push_str(t),
                    InterpPart::Hole(h) => {
                        if !text.is_empty() {
                            let (off, len) = b.utf16_literal(&text);
                            let s = b.build_utf16_string(off, len);
                            acc = Some(match acc {
                                Some(a) => b.concat_strings(a, s),
                                None => s,
                            });
                            text.clear();
                        }
                        // A hole is concatenated as a string object. A
                        // primitive is converted first (it used to be
                        // concatenated as a pointer and crash).
                        let hole_ty =
                            static_type_of(h, b, ctx).unwrap_or_default();
                        let mut v = eval(h, b, ctx)?;
                        if is_primitive_spelling(&hole_ty)
                            && !is_string_type(&hole_ty)
                        {
                            let Some(sv) = b.value_to_string(v, &hole_ty) else {
                                return Err(LowerError::UnsupportedConstruct(format!(
                                    "interpolating `{hole_ty}` needs value-to-string conversion, which is not implemented"
                                )));
                            };
                            v = sv;
                        }
                        acc = Some(match acc {
                            Some(a) => b.concat_strings(a, v),
                            None => v,
                        });
                    }
                }
            }
            if !text.is_empty() {
                let (off, len) = b.utf16_literal(&text);
                let s = b.build_utf16_string(off, len);
                acc = Some(match acc {
                    Some(a) => b.concat_strings(a, s),
                    None => s,
                });
            }
            Ok(acc.unwrap_or_else(|| {
                let (off, len) = b.utf16_literal("");
                b.build_utf16_string(off, len)
            }))
        }
        Expr::Literal(Literal::Int(v)) => Ok(b.emit(Instr::Const {
            dst: VReg(0),
            imm: *v,
        })),
        Expr::Literal(Literal::Char(v)) => Ok(b.emit(Instr::Const {
            dst: VReg(0),
            imm: *v as i64,
        })),
        Expr::Literal(Literal::Float(v)) => {
            // Floats are not tracked in `const_vals` (i64-only), so emit a
            // raw `FConst` carrying the IEEE 754 bit pattern.
            let dst = b.fresh();
            b.push_raw(Instr::FConst {
                dst,
                bits: v.to_bits(),
            });
            Ok(dst)
        }
        Expr::Literal(Literal::Bool(v)) => Ok(b.emit(Instr::Const {
            dst: VReg(0),
            imm: i64::from(*v),
        })),
        Expr::Literal(Literal::Null) => Ok(b.const_i64(0)),
        Expr::Ident { name, .. } => {
            // Declared bindings win over the `this` keyword: `@this`
            // (e.g. a parameter literally named `this`, as in
            // `UpdateAndExecute1(CallSite @this, ...)`) is an ordinary
            // in-scope variable, not the receiver (static methods don't
            // even have one).
            if let Some(sym) = b.scopes.lookup(name) {
                let home = b.bindings[sym.slot];
                let byref = sym.is_byref;
                let sym_ty = sym.type_name.clone();
                let dst = b.fresh();
                // By-reference parameter (`ref`/`out`/`in`): the home
                // holds an address (callers box); dereference to read
                // the value. Narrow unsigned targets (`ref byte`,
                // `ref char`): mask to width, or a `char` read picks
                // up 6 neighbor bytes and `ToUpper` throws on the
                // garbage code point (the casing-loop fault).
                if byref {
                    b.push_raw(Instr::LoadField {
                        dst,
                        base: home,
                        off: 0,
                    });
                    if let Some(mask) = narrow_load_mask(&sym_ty) {
                        let m = b.const_i64(mask);
                        return Ok(b.emit(Instr::Bin {
                            dst: VReg(0),
                            op: IrBinOp::BitAnd,
                            a: dst,
                            b: m,
                        }));
                    }
                } else {
                    b.push_raw(Instr::Mov { dst, a: home });
                }
                // No constant copy here: the home may be re-assigned on
                // paths not visible from this point (loops, branches).
                // Reads of locals are therefore never compile-time consts.
                b.const_vals.remove(&dst.0);
                return Ok(dst);
            }
            if name == "this" {
                return Ok(VReg(0));
            }
            if let Some(info) = this_info(ctx, name) {
                if let Some(field) = info.field(name) {
                    // Implicit-`this` read; may still hit an inherited member.
                    ctx.check_access(true, ctx.current_class, name)?;
                    // `const` field: no storage — fold to the literal
                    // instead of reading the vtable slot at offset 0.
                    if field.is_const {
                        return Ok(fold_const_field(b, field));
                    }
                    // Static field: value lives in per-class static storage,
                    // independent of `this`. Read [static_addr + 0].
                    if field.is_static {
                        let Some(base) = static_addr(field, b, ctx) else {
                            return Ok(b.const_i64(0));
                        };
                        let dst = b.fresh();
                        b.push_raw(Instr::LoadField { dst, base, off: 0 });
                        return Ok(dst);
                    }
                    let this = VReg(0);
                    // `string` interiors resolve against RubyC's own
                    // builtin string layout (see
                    // `semantic::builtin_string_interior_offset`), not a
                    // table slot — raw table offsets would read
                    // data-as-length.
                    if crate::native::semantic::builtin_string_interior_offset(
                        ctx.current_class,
                        name,
                    )
                    .is_some()
                    {
                        return Ok(string_interior_read(b, this, name));
                    }
                    let dst = b.fresh();
                    b.push_raw(Instr::LoadField {
                        dst,
                        base: this,
                        off: (field.offset_slots * 8) as u32,
                    });
                    // Struct-typed field values that read null are
                    // uninitialized structs (e.g. an embedded
                    // `ManualResetValueTaskSourceCore` never assigned
                    // before its method is invoked) — same rule as
                    // the Member-read path (see `ensure_struct_base`).
                    if struct_entry_for(ctx.table, &field.type_name).is_some() {
                        return Ok(ensure_struct_base(b, ctx, dst, &field.type_name));
                    }
                    return Ok(dst);
                }
                // Accessor-backed property read (`table` resolving to a
                // base-class property through a derived receiver):
                // inline the getter body (e.g. bare `Out` inside
                // `Console` reads `Out`'s getter, which builds the
                // writer); falls back to the placeholder below when
                // not inlineable.
                if info.properties.iter().any(|p| p.name == *name) {
                    ctx.check_access(true, ctx.current_class, name)?;
                    // A property the class does not implement (abstract) or
                    // that a subclass OVERRIDES dispatches virtually: the
                    // runtime type's implementation serves, like a virtual
                    // method. The static body cannot serve an abstract
                    // property at all — it has none — so the read would
                    // have no value (`'Stream' has no member 'CanRead'`).
                    // A backing field cannot reach here: the field branch
                    // above returns for an auto-property.
                    let upward_explicit = most_derived_prop_decl(ctx, info, name)
                        .is_some_and(|(decl, _)| decl.get_expr.is_some() || decl.get.is_some());
                    let abstract_prop = info
                        .properties
                        .iter()
                        .any(|p| p.name == *name && p.is_abstract);
                    if (upward_explicit || abstract_prop)
                        && strict_subclass_has_explicit(ctx, info, name)
                        && let Some(val) = eval_virtual_getter(b, ctx, info, name, None)?
                    {
                        return Ok(val);
                    }
                    if let Some(val) = eval_property_getter(
                            b,
                            ctx,
                            info,
                            ctx.current_class,
                            name,
                            None,
                        )?
                    {
                        return Ok(val);
                    }
return Err(LowerError::UnsupportedConstruct(crate::tf!(
                        "rubyc.native.ir.lower_expr.no_member",
                        info.name.clone(),
                        name.to_string()
                    )));
                }
                // An event is accessor-backed with no object layout
                // slot, so there is no storage to read. That is a real
                // gap (event semantics land with 2.6c), not a value to
                // invent — but it must be reported AS the gap: saying
                // "'E' has no member 'A'" when `E` declares `A` sends the
                // reader looking for a typo instead of at the missing
                // feature.
                if info
                    .own_members
                    .iter()
                    .any(|(mn, is_f, _)| mn == name && *is_f)
                {
                    let is_event = ctx
                        .classes
                        .iter()
                        .filter(|c| c.name == info.name)
                        .any(|c| {
                            c.members.iter().any(|m| {
                                matches!(
                                    m,
                                    crate::models::ClassMember::Event(e) if e.name == *name
                                )
                            })
                        });
                    let key = if is_event {
                        "rubyc.native.ir.lower_expr.event_unavailable"
                    } else {
                        "rubyc.native.ir.lower_expr.no_member"
                    };
                    return Err(LowerError::UnsupportedConstruct(crate::tf!(
                        key,
                        info.name.clone(),
                        name.to_string()
                    )));
                }
            }
            // Enclosing-scope statics (`s_utcTimeZone` read from nested
            // `CachedData`): same static-storage read, declaring class
            // from the outer entry. Nested code may read enclosing
            // privates (nesting family); genuinely foreign access stays
            // loud.
            if let Some((outer, field)) = outer_static_field(ctx, name) {
                // Both sides are QUALIFIED: a bare class name is
                // ambiguous when two namespaces declare same-named
                // types, and the ambiguity made the nesting-family
                // rule give up on a legal access.
                let viewer = match ctx.current_path {
                    Some(p) if !p.is_empty() => p,
                    _ => ctx.current_class,
                };
                if !crate::native::semantic::visible_from(
                    ctx.table,
                    field.access,
                    &outer.key.path,
                    viewer,
                ) {
return Err(LowerError::UnsupportedConstruct(crate::tf!(
                        "rubyc.native.ir.lower_expr.no_member",
                        outer.name.clone(),
                        name.to_string()
                    )));
                }
                // `const` from an enclosing class: fold the literal
                // (no static storage exists for it).
                if field.is_const {
                    return Ok(fold_const_field(b, field));
                }
                let Some(base) = static_addr(field, b, ctx) else {
                    return Ok(b.const_i64(0));
                };
                let dst = b.fresh();
                b.push_raw(Instr::LoadField { dst, base, off: 0 });
                return Ok(dst);
            }
            // Enclosing-scope static property (`HighCharTable` read
            // from nested `IcuSearchValues`, declared on outer
            // `CompareInfo`): the field/const lookup above misses
            // properties, so without this the name fell to the
            // uppercase placeholder (const 0) and downstream span
            // code faulted on null. Inline the outer getter exactly
            // like a same-class static property read; a non-inlineable
            // getter degrades to the same placeholder the same-class
            // path uses.
            if let Some((outer, prop)) = outer_static_property(ctx, name) {
                if !crate::native::semantic::visible_from(
                    ctx.table,
                    prop.access,
                    &outer.name,
                    ctx.current_class,
                ) {
return Err(LowerError::UnsupportedConstruct(crate::tf!(
                        "rubyc.native.ir.lower_expr.no_member",
                        outer.name.clone(),
                        name.to_string()
                    )));
                }
                if let Some(val) = eval_property_getter(
                    b,
                    ctx,
                    outer,
                    outer.key.path.as_str(),
                    name,
                    None,
                )?
                {
                    return Ok(val);
                }
                // The enclosing type declares the property but has no
                // getter body to inline (abstract or accessor-only), so
                // there is no value to read. Answering 0 here would look
                // like a property that legitimately holds default(T).
                return Err(LowerError::UnsupportedConstruct(crate::tf!(
                    "rubyc.native.ir.lower_expr.no_property_body",
                    outer.name.clone(),
                    name.to_string()
                )));
            }
            if name == "base" {                // `base` refers to the base-class instance — the same object
                // as `this` (the base is a part of the same object).
                let dst = b.fresh();
                b.push_raw(Instr::Mov { dst, a: VReg(0) });
                return Ok(dst);
            }
            // Type names used as values (e.g. `sizeof(UnmanagedPair)`,
            // `sizeof(int)`), generic type parameters, and pattern-binding
            // variables not yet registered in the lowering scope — return 0
            // as a placeholder.
            // Parser placeholders are values by construction, never
            // unresolved names (mirrors the semantic scan's known-list):
            // `lambda` (lambda bodies are skipped, not modeled),
            // `<pattern>` (`sizeof`/`typeof` in pattern position), and
            // `_` (discards, e.g. `out _` — write-only, read as zero).
            // Bare method names (method groups) come FIRST: an
            // uppercase initial (`Foo`) otherwise falls into the
            // type-name rule below and yields zero instead of a
            // delegate object (which zeroed `s_sentinel` and threw
            // spuriously). Types still win via `is_method_group_name`
            // (it rejects names that resolve as classes).
            if is_method_group_name(name, b, ctx) {
                // A METHOD GROUP is a delegate value: `[fn_addr, kind,
                // type]` with kind 0 (it takes only its own arguments).
                // The object used to be allocated EMPTY, so `D d = H;
                // d(x)` called address 0 and faulted.
                if let Some(fidx) = method_group_index(name, b, ctx) {
                    let cell = b.fresh();
                    b.push_raw(Instr::Alloc { dst: cell, bytes: 24 });
                    // The delegate's runtime type, as for a lambda.
                    if let Some(expected) = b.expected_delegate.clone()
                        && ctx.table.is_delegate(&expected)
                    {
                        let desc = b.type_token(&expected, ctx)?;
                        b.push_raw(Instr::StoreField {
                            base: cell,
                            off: 16,
                            src: desc,
                        });
                    }
                    let addr = b.fresh();
                    b.push_raw(Instr::FnAddr { dst: addr, fidx });
                    b.push_raw(Instr::StoreField {
                        base: cell,
                        off: 0,
                        src: addr,
                    });
                    let zero = b.const_i64(0);
                    b.push_raw(Instr::StoreField {
                        base: cell,
                        off: 8,
                        src: zero,
                    });
                    return Ok(cell);
                }
                return Err(LowerError::UnsupportedConstruct(format!(
                    "method group '{name}' does not resolve to a function (ambiguous or missing signature)"
                )));
            }
            if (ctx.table.class(name).is_some()
                || name == "_"
                || name
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_uppercase())
                    .unwrap_or(false)
                || matches!(
                    name.as_str(),
                    "int"
                        | "uint"
                        | "long"
                        | "ulong"
                        | "short"
                        | "ushort"
                        | "byte"
                        | "sbyte"
                        | "float"
                        | "double"
                        | "bool"
                        | "char"
                        | "string"
                        | "nint"
                        | "nuint"
                        | "object"
                        | "decimal"
                        | "dynamic"
                ))
                && !b.scopes.lookup(name).is_some()
            {
                return Ok(b.const_i64(0));
            }
            // Fallback: a name that is neither a local, a field, `this`,
            // `base`, nor a type is a genuinely undeclared variable. C#
            // reports this (CS0103); returning 0 silently would mask real
            // bugs and produce wrong output. (Method groups return
            // earlier above with a heap object.)
            Err(LowerError::UnsupportedConstruct(crate::tf!(
                "rubyc.native.ir.lower_expr.use_undeclared",
                name.to_owned()
            )))
        }
        Expr::Unary {
            op: UnOp::Not,
            inner,
            ..
        } => {
            let a = eval(inner, b, ctx)?;
            if let Some(&v) = b.const_vals.get(&a.0) {
                return Ok(b.emit(Instr::Const {
                    dst: VReg(0),
                    imm: i64::from(v == 0),
                }));
            }
            let zero = b.const_i64(0);
            Ok(b.emit(Instr::Cmp {
                dst: VReg(0),
                op: CmpOp::Eq,
                a,
                b: zero,
            }))
        }
        Expr::Unary {
            op: UnOp::BitNot,
            inner,
            ..
        } => {
            // `~x` is `x ^ -1`; no dedicated IR instruction needed.
            let a = eval(inner, b, ctx)?;
            if let Some(&v) = b.const_vals.get(&a.0) {
                return Ok(b.const_i64(!v));
            }
            let minus1 = b.const_i64(-1);
            Ok(b.emit(Instr::Bin {
                dst: VReg(0),
                op: IrBinOp::BitXor,
                a,
                b: minus1,
            }))
        }
        Expr::Unary {
            op: UnOp::Deref,
            inner,
            ..
        } => {
            // `*x` — sized load from the address. The pointed-to
            // width comes from the inner type (`char*` loads 2
            // bytes); unsigned narrow loads zero-extend.
            let addr = eval(inner, b, ctx)?;
            let pointed = static_type_of(inner, b, ctx)
                .unwrap_or_else(|_| "object".to_owned());
            let pt = pointed.trim();
            if !is_pointer_ty(pt) {
                // Not a pointer spelling (already-address refs):
                // legacy passthrough.
                return Ok(addr);
            }
            let inner_ty = pt.trim_end_matches(['*', ' ', '?']).trim();
            let raw = b.fresh();
            b.push_raw(Instr::LoadField {
                dst: raw,
                base: addr,
                off: 0,
            });
            if let Some(mask) = narrow_load_mask(inner_ty) {
                let m = b.const_i64(mask);
                return Ok(b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::BitAnd,
                    a: raw,
                    b: m,
                }));
            }
            Ok(raw)
        }
        Expr::Unary {
            op: UnOp::AddrOf,
            inner,
            ..
        } => {
            // `&x` — address-of. Must compute the address, never the
            // value: a pointer holding the value turned every later
            // dereference into a read from a small integer address
            // (`int x = 41; int* p = &x; *p` faulted) and every
            // `*p = v` into a write to nowhere.
            //
            // Order matters. `ref_target_address` handles the targets
            // that already have an interior address (array elements,
            // string data, fields, ref-returning calls). A LOCAL is the
            // remaining case, and it does have a home: the value is
            // spilled to a stack cell and that cell's address is the
            // pointer's value. A local already bound by reference needs
            // no cell — its binding IS the address.
            if let Some(addr) = ref_target_address(b, ctx, inner)? {
                return Ok(addr);
            }
            if let Expr::Ident { name, .. } = inner.as_ref() {
                if name == "this" {
                    // The receiver is already an address.
                    return Ok(VReg(0));
                }
                // `ref _` / `out _`: a discard needs a throwaway
                // location, not an error — C# requires the compiler to
                // supply one. Nobody reads it.
                if name == "_" {
                    let cell = b.fresh();
                    b.push_raw(Instr::Alloc { dst: cell, bytes: 16 });
                    return Ok(cell);
                }
                // A static field HAS an address: its storage slot (or,
                // for a field owned by another image, the owner's slot
                // via address import). `Interlocked.CompareExchange(ref
                // _x, …)` needs exactly this.
                // Static fields may live on ANOTHER PART of a partial
                // class, so consult the type table as well as the
                // current declaration: `ref s_in` in a `partial class
                // Console` part that does not declare the field must
                // still find its storage slot.
                let static_field = ctx
                    .current_info
                    .and_then(|info| {
                        info.static_fields
                            .iter()
                            .find(|f| f.name == *name && f.is_static)
                            .cloned()
                    })
                    .or_else(|| {
                        ctx.table
                            .class(&ctx.current_class)
                            .and_then(|info| {
                                info.static_fields
                                    .iter()
                                    .find(|f| f.name == *name && f.is_static)
                                    .cloned()
                            })
                    });
                if let Some(f) = static_field {
                    if let Some(addr) = static_addr(&f, b, ctx) {
                        return Ok(addr);
                    }
                }
                // A bare instance field is `this.field` — the
                // implicit-`this` spelling.
                if let Some(addr) = implicit_this_field_address(b, ctx, name)? {
                    return Ok(addr);
                }
                let binding = b
                    .scopes
                    .lookup(name)
                    .map(|sym| (sym.slot, sym.is_byref));
                if let Some((slot, already_byref)) = binding {
                    let home = b.bindings[slot];
                    if already_byref {
                        return Ok(home);
                    }
                    // Materialize a home for the local and hand out its
                    // address, then RE-BIND the local as by-reference so
                    // a write through the pointer is visible to later
                    // reads of `x` (C# aliasing). `Alloc` rather than
                    // `StackAlloc`: the x86-64 backend does not
                    // implement stack allocation, and this is the same
                    // cell mechanism the byref-argument path uses.
                    let cell = b.fresh();
                    b.push_raw(Instr::Alloc { dst: cell, bytes: 16 });
                    b.push_raw(Instr::StoreField {
                        base: cell,
                        off: 0,
                        src: home,
                    });
                    b.bindings[slot] = cell;
                    b.scopes.make_byref(name);
                    return Ok(cell);
                }
            }
            // Not addressable (a temporary, a literal): there is no
            // cell to hand out, so report the shape rather than pass the
            // value off as an address.
            Err(LowerError::UnsupportedConstruct(format!(
                "{} [addr-of {expr:?} in {}]",
                crate::tf!("rubyc.native.ir.lower_expr.address_of_temporary"),
                ctx.current_method.clone().unwrap_or_default()
            )))
        }
        Expr::Unary {
            op: UnOp::Throw,
            inner,
            ..
        } => {
            // `throw expr` in expression position (`=> throw`,
            // `?? throw`, `?: throw`): evaluate the exception object,
            // seal the block with `Unwind`, and yield its vreg. The
            // vreg is poison (control never continues), but callers
            // keep lowering with it: post-seal emission evaporates
            // (see `push_raw`/`terminate`), so the unwind stands and
            // the poison never reaches real instructions.
            let v = eval(inner, b, ctx)?;
            b.terminate(Terminator::Unwind { src: v });
            Ok(v)
        }
        Expr::Unary {
            op: UnOp::IndexFromEnd,
            inner,
            ..
        } => {
            // `^x` — C# 8 Index from-end. For now, just evaluate the
            // inner expression; codegen treats it as a no-op.
            eval(inner, b, ctx)
        }
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            let cv = eval(cond, b, ctx)?;
            if let Some(&c) = b.const_vals.get(&cv.0) {
                // Constant condition: lower the taken side only.
                return if c != 0 {
                    eval(then_expr, b, ctx)
                } else {
                    eval(else_expr, b, ctx)
                };
            }
            let home = b.fresh();
            let then_b = b.make_block();
            let else_b = b.make_block();
            let end_b = b.make_block();
            b.terminate(Terminator::Branch {
                cond: Some(cv),
                if_true: then_b,
                if_false: else_b,
            });
            // Target-typed `default` arms: a bare-default arm takes
            // the other arm's type (UTF8-noBOM `Preamble` false-arm
            // `default` must be an empty span, not null). Types
            // computed only when an arm is actually bare.
            let else_ty = if matches!(&**then_expr, Expr::Default { type_name, .. } if type_name == "void")
            {
                static_type_of(else_expr, b, ctx).unwrap_or_default()
            } else {
                String::new()
            };
            let then_ty = if matches!(&**else_expr, Expr::Default { type_name, .. } if type_name == "void")
            {
                static_type_of(then_expr, b, ctx).unwrap_or_default()
            } else {
                String::new()
            };
            b.switch_to(then_b);
            let t = match default_with_type(then_expr, &else_ty) {
                Some(fixed) => eval(&fixed, b, ctx)?,
                None => eval(then_expr, b, ctx)?,
            };
            b.mov(home, t);
            if !b.is_sealed() {
                b.goto(end_b);
            }
            b.switch_to(else_b);
            let e = match default_with_type(else_expr, &then_ty) {
                Some(fixed) => eval(&fixed, b, ctx)?,
                None => eval(else_expr, b, ctx)?,
            };
            b.mov(home, e);
            if !b.is_sealed() {
                b.goto(end_b);
            }
            b.switch_to(end_b);
            Ok(home)
        }
        Expr::Unary {
            op: UnOp::Neg,
            inner,
            ..
        } => {
            let a = eval(inner, b, ctx)?;
            if let Some(&v) = b.const_vals.get(&a.0) {
                return Ok(b.emit(Instr::Const {
                    dst: VReg(0),
                    imm: v.wrapping_neg(),
                }));
            }
            if is_float_expr(b, ctx, inner) {
                let dst = b.fresh();
                b.push_raw(Instr::FNeg { dst, a });
                return Ok(dst);
            }
            Ok(b.emit(Instr::Neg { dst: VReg(0), a }))
        }
        Expr::Binary { op, lhs, rhs, .. } => {
            // Delegate combination (`d += g`) has no implementation: a
            // delegate value is a single object, so `+` used to do
            // pointer arithmetic on it and the following call jumped
            // into garbage. Report it until multicast lands.
            if matches!(
                op,
                crate::models::BinOp::Add | crate::models::BinOp::Sub
            ) {
                let lt = static_type_of(lhs, b, ctx).unwrap_or_default();
                let rt = static_type_of(rhs, b, ctx).unwrap_or_default();
                // `System.Action`/`Func<...>` are runtime types, not
                // declared delegates in the table, so the left side is
                // recognised by the METHOD GROUP (or lambda) on the
                // right — the only way `+`/`-` reaches a delegate here.
                let lhs_del = ctx.table.is_delegate(&lt);
                let rhs_del = ctx.table.is_delegate(&rt)
                    || is_method_group(rhs, b, ctx)
                    || matches!(rhs.as_ref(), Expr::Lambda { .. });
                if rhs_del || (lhs_del && ctx.table.is_delegate(&rt)) {
                    // `d += g` on a LOCAL: build the combined delegate
                    // (`[0, kind 2, head]` over `[next, delegate]` nodes).
                    // `null + g` is `g`; an existing list is appended to,
                    // so handlers run in subscription order.
                    // Locals and INSTANCE FIELDS can be combined: the
                    // AssignComplex path writes the result back to the
                    // same target. A property is excluded (its setter
                    // path is not wired for this), and so is a static
                    // field (no instance offset).
                    let lhs_is_local = matches!(
                        lhs.as_ref(),
                        Expr::Ident { name, .. } if b.scopes.lookup(name).is_some()
                    );
                    let lhs_is_field = {
                        let fname = match lhs.as_ref() {
                            Expr::Ident { name, .. } => Some(name.as_str()),
                            Expr::Member { name, .. } => Some(name.as_str()),
                            _ => None,
                        };
                        fname.is_some_and(|n| {
                            ctx.current_info.is_some_and(|i| i.field(n).is_some())
                        })
                    };
                    let lhs_is_prop = {
                        let fname = match lhs.as_ref() {
                            Expr::Ident { name, .. } => Some(name.as_str()),
                            Expr::Member { name, .. } => Some(name.as_str()),
                            _ => None,
                        };
                        // A property needs a SETTER, or the write-back
                        // is a silent no-op (C# rejects the assignment).
                        // The accessor list lives on the declaration.
                        fname.is_some_and(|n| {
                            ctx.current_info.is_some_and(|i| i.property(n).is_some())
                                && ctx.classes.iter().any(|c| {
                                    c.name == ctx.current_class
                                        && c.members.iter().any(|m| {
                                            matches!(
                                                m,
                                                crate::models::ClassMember::Property(p)
                                                    if p.name == n && p.is_settable
                                            )
                                        })
                                })
                        })
                    };
                    let lhs_is_local = lhs_is_local || lhs_is_field || lhs_is_prop;
                    if matches!(op, crate::models::BinOp::Add) && lhs_is_local {
                        let a = eval(lhs, b, ctx)?;
                        let bv = eval(rhs, b, ctx)?;
                        let zero = b.const_i64(0);
                        let two = b.const_i64(2);
                        let home = b.fresh();
                        let null_b = b.make_block();
                        let some_b = b.make_block();
                        let join_b = b.make_block();
                        let is_null = b.emit(Instr::Cmp {
                            dst: VReg(0),
                            op: crate::native::ir::CmpOp::Eq,
                            a,
                            b: zero,
                        });
                        b.terminate(Terminator::Branch {
                            cond: Some(is_null),
                            if_true: null_b,
                            if_false: some_b,
                        });
                        b.switch_to(null_b);
                        b.mov(home, bv);
                        b.goto(join_b);
                        b.switch_to(some_b);
                        // Node for the new handler.
                        let nd = b.fresh();
                        b.push_raw(Instr::Alloc {
                            dst: nd,
                            bytes: 16,
                        });
                        b.push_raw(Instr::StoreField {
                            base: nd,
                            off: 0,
                            src: zero,
                        });
                        b.push_raw(Instr::StoreField {
                            base: nd,
                            off: 8,
                            src: bv,
                        });
                        let k = b.fresh();
                        b.push_raw(Instr::LoadField {
                            dst: k,
                            base: a,
                            off: 8,
                        });
                        let is_multi = b.emit(Instr::Cmp {
                            dst: VReg(0),
                            op: crate::native::ir::CmpOp::Eq,
                            a: k,
                            b: two,
                        });
                        let multi_b = b.make_block();
                        let new_b = b.make_block();
                        b.terminate(Terminator::Branch {
                            cond: Some(is_multi),
                            if_true: multi_b,
                            if_false: new_b,
                        });
                        // First combination: a list of the two values.
                        b.switch_to(new_b);
                        let n1 = b.fresh();
                        b.push_raw(Instr::Alloc {
                            dst: n1,
                            bytes: 16,
                        });
                        b.push_raw(Instr::StoreField {
                            base: n1,
                            off: 0,
                            src: zero,
                        });
                        b.push_raw(Instr::StoreField {
                            base: n1,
                            off: 8,
                            src: a,
                        });
                        b.push_raw(Instr::StoreField {
                            base: n1,
                            off: 0,
                            src: nd,
                        });
                        let obj = b.fresh();
                        b.push_raw(Instr::Alloc {
                            dst: obj,
                            bytes: 32,
                        });
                        b.push_raw(Instr::StoreField {
                            base: obj,
                            off: 0,
                            src: zero,
                        });
                        b.push_raw(Instr::StoreField {
                            base: obj,
                            off: 8,
                            src: two,
                        });
                        // The type word, then the subscriber list.
                        let ctype = b.const_i64(0);
                        b.push_raw(Instr::StoreField {
                            base: obj,
                            off: 16,
                            src: ctype,
                        });
                        b.push_raw(Instr::StoreField {
                            base: obj,
                            off: 24,
                            src: n1,
                        });
                        b.mov(home, obj);
                        b.goto(join_b);
                        // Already combined: append at the tail.
                        b.switch_to(multi_b);
                        let head = b.fresh();
                        b.push_raw(Instr::LoadField {
                            dst: head,
                            base: a,
                            off: 24,
                        });
                        let loop_b = b.make_block();
                        let advance_b = b.make_block();
                        let attach_b = b.make_block();
                        b.goto(loop_b);
                        b.switch_to(loop_b);
                        let nxt = b.fresh();
                        b.push_raw(Instr::LoadField {
                            dst: nxt,
                            base: head,
                            off: 0,
                        });
                        let is_end = b.emit(Instr::Cmp {
                            dst: VReg(0),
                            op: crate::native::ir::CmpOp::Eq,
                            a: nxt,
                            b: zero,
                        });
                        b.terminate(Terminator::Branch {
                            cond: Some(is_end),
                            if_true: attach_b,
                            if_false: advance_b,
                        });
                        b.switch_to(advance_b);
                        b.mov(head, nxt);
                        b.goto(loop_b);
                        b.switch_to(attach_b);
                        b.push_raw(Instr::StoreField {
                            base: head,
                            off: 0,
                            src: nd,
                        });
                        b.mov(home, a);
                        b.goto(join_b);
                        b.switch_to(join_b);
                        return Ok(home);
                    }
                    // `d -= g` for a METHOD GROUP: unlink the first node
                    // whose delegate has the same function address. A
                    // captured closure is NOT matched by address (its
                    // identity is the captured object), so a lambda
                    // operand keeps the report below.
                    if matches!(op, crate::models::BinOp::Sub)
                        && is_method_group(rhs, b, ctx)
                    {
                        let a = eval(lhs, b, ctx)?;
                        let bv = eval(rhs, b, ctx)?;
                        let zero = b.const_i64(0);
                        let two = b.const_i64(2);
                        let home = b.fresh();
                        let end_b = b.make_block();
                        let finish_b = b.make_block();
                        let null_b = b.make_block();
                        let some_b = b.make_block();
                        let is_null = b.emit(Instr::Cmp {
                            dst: VReg(0),
                            op: crate::native::ir::CmpOp::Eq,
                            a,
                            b: zero,
                        });
                        b.terminate(Terminator::Branch {
                            cond: Some(is_null),
                            if_true: null_b,
                            if_false: some_b,
                        });
                        b.switch_to(null_b);
                        b.mov(home, zero); // null - g is null
                        b.goto(end_b);
                        b.switch_to(some_b);
                        let kind = b.fresh();
                        b.push_raw(Instr::LoadField {
                            dst: kind,
                            base: a,
                            off: 8,
                        });
                        let is_multi = b.emit(Instr::Cmp {
                            dst: VReg(0),
                            op: crate::native::ir::CmpOp::Eq,
                            a: kind,
                            b: two,
                        });
                        let list_b = b.make_block();
                        let single_b = b.make_block();
                        b.terminate(Terminator::Branch {
                            cond: Some(is_multi),
                            if_true: list_b,
                            if_false: single_b,
                        });
                        // One delegate: same function -> null, else stay.
                        b.switch_to(single_b);
                        let bfn0 = b.fresh();
                        b.push_raw(Instr::LoadField {
                            dst: bfn0,
                            base: bv,
                            off: 0,
                        });
                        let afn0 = b.fresh();
                        b.push_raw(Instr::LoadField {
                            dst: afn0,
                            base: a,
                            off: 0,
                        });
                        let same0 = b.emit(Instr::Cmp {
                            dst: VReg(0),
                            op: crate::native::ir::CmpOp::Eq,
                            a: afn0,
                            b: bfn0,
                        });
                        let eq_b = b.make_block();
                        let ne_b = b.make_block();
                        b.terminate(Terminator::Branch {
                            cond: Some(same0),
                            if_true: eq_b,
                            if_false: ne_b,
                        });
                        b.switch_to(eq_b);
                        b.mov(home, zero);
                        b.goto(end_b);
                        b.switch_to(ne_b);
                        b.mov(home, a);
                        b.goto(end_b);
                        // Combined: walk the list and unlink one node.
                        b.switch_to(list_b);
                        let bfn = b.fresh();
                        b.push_raw(Instr::LoadField {
                            dst: bfn,
                            base: bv,
                            off: 0,
                        });
                        let head = b.fresh();
                        b.push_raw(Instr::LoadField {
                            dst: head,
                            base: a,
                            off: 24,
                        });
                        let prev = b.fresh();
                        b.mov(prev, zero);
                        let cur = b.fresh();
                        b.mov(cur, head);
                        let loop_b = b.make_block();
                        let body_b = b.make_block();
                        let check_b = b.make_block();
                        let choose_b = b.make_block();
                        let hit_head_b = b.make_block();
                        let hit_prev_b = b.make_block();
                        let advance_b = b.make_block();
                        b.goto(loop_b);
                        b.switch_to(loop_b);
                        let done = b.emit(Instr::Cmp {
                            dst: VReg(0),
                            op: crate::native::ir::CmpOp::Eq,
                            a: cur,
                            b: zero,
                        });
                        b.terminate(Terminator::Branch {
                            cond: Some(done),
                            if_true: finish_b,
                            if_false: body_b,
                        });
                        b.switch_to(body_b);
                        let d = b.fresh();
                        b.push_raw(Instr::LoadField {
                            dst: d,
                            base: cur,
                            off: 8,
                        });
                        let k = b.fresh();
                        b.push_raw(Instr::LoadField {
                            dst: k,
                            base: d,
                            off: 8,
                        });
                        let k0 = b.emit(Instr::Cmp {
                            dst: VReg(0),
                            op: crate::native::ir::CmpOp::Eq,
                            a: k,
                            b: zero,
                        });
                        b.terminate(Terminator::Branch {
                            cond: Some(k0),
                            if_true: check_b,
                            if_false: advance_b,
                        });
                        b.switch_to(check_b);
                        let t = b.fresh();
                        b.push_raw(Instr::LoadField {
                            dst: t,
                            base: d,
                            off: 0,
                        });
                        let hit = b.emit(Instr::Cmp {
                            dst: VReg(0),
                            op: crate::native::ir::CmpOp::Eq,
                            a: t,
                            b: bfn,
                        });
                        b.terminate(Terminator::Branch {
                            cond: Some(hit),
                            if_true: choose_b,
                            if_false: advance_b,
                        });
                        b.switch_to(choose_b);
                        let at_head = b.emit(Instr::Cmp {
                            dst: VReg(0),
                            op: crate::native::ir::CmpOp::Eq,
                            a: prev,
                            b: zero,
                        });
                        b.terminate(Terminator::Branch {
                            cond: Some(at_head),
                            if_true: hit_head_b,
                            if_false: hit_prev_b,
                        });
                        b.switch_to(hit_head_b);
                        let nx1 = b.fresh();
                        b.push_raw(Instr::LoadField {
                            dst: nx1,
                            base: cur,
                            off: 0,
                        });
                        b.push_raw(Instr::StoreField {
                            base: a,
                            off: 24,
                            src: nx1,
                        });
                        b.goto(finish_b);
                        b.switch_to(hit_prev_b);
                        let nx2 = b.fresh();
                        b.push_raw(Instr::LoadField {
                            dst: nx2,
                            base: cur,
                            off: 0,
                        });
                        b.push_raw(Instr::StoreField {
                            base: prev,
                            off: 0,
                            src: nx2,
                        });
                        b.goto(finish_b);
                        b.switch_to(advance_b);
                        b.mov(prev, cur);
                        let nx3 = b.fresh();
                        b.push_raw(Instr::LoadField {
                            dst: nx3,
                            base: cur,
                            off: 0,
                        });
                        b.mov(cur, nx3);
                        b.goto(loop_b);
                        b.switch_to(finish_b);
                        // C# answers NULL when removal empties the list,
                        // not an empty combined delegate (invoking that
                        // was a silent no-op).
                        let final_head = b.fresh();
                        b.push_raw(Instr::LoadField {
                            dst: final_head,
                            base: a,
                            off: 24,
                        });
                        let emptied = b.emit(Instr::Cmp {
                            dst: VReg(0),
                            op: crate::native::ir::CmpOp::Eq,
                            a: final_head,
                            b: zero,
                        });
                        let empty_b = b.make_block();
                        let keep_b = b.make_block();
                        b.terminate(Terminator::Branch {
                            cond: Some(emptied),
                            if_true: empty_b,
                            if_false: keep_b,
                        });
                        b.switch_to(empty_b);
                        b.mov(home, zero);
                        b.goto(end_b);
                        b.switch_to(keep_b);
                        b.mov(home, a);
                        b.goto(end_b);
                        b.switch_to(end_b);
                        return Ok(home);
                    }
                    return Err(LowerError::UnsupportedConstruct(format!(
                        "combining delegates (`d {} g`) needs a local, an instance/static field, or a SETTABLE property",
                        if matches!(op, crate::models::BinOp::Add) { "+=" } else { "-=" }
                    )));
                }
            }
            // String concatenation is implemented for string + string
            // only. With a PRIMITIVE operand (`"a" + 5`, `"a" + true`)
            // C# converts the value with ToString; this compiler has no
            // value-to-string conversion yet, and the raw value used to
            // be treated as a string pointer — a crash or an empty
            // result. Report it instead of computing garbage.
            if matches!(op, crate::models::BinOp::Add) {
                let lt = static_type_of(lhs, b, ctx).unwrap_or_default();
                let rt = static_type_of(rhs, b, ctx).unwrap_or_default();
                let primitive_non_string = |t: &str| {
                    is_primitive_spelling(t) && !is_string_type(t)
                };
                if is_string_type(&lt) && primitive_non_string(&rt) {
                    let a = eval(lhs, b, ctx)?;
                    let bv = eval(rhs, b, ctx)?;
                    let Some(s) = b.value_to_string(bv, &rt) else {
                        return Err(LowerError::UnsupportedConstruct(format!(
                            "string concatenation with `{rt}` needs value-to-string conversion, which is not implemented"
                        )));
                    };
                    return Ok(b.concat_strings(a, s));
                }
                if is_string_type(&rt) && primitive_non_string(&lt) {
                    let a = eval(lhs, b, ctx)?;
                    let bv = eval(rhs, b, ctx)?;
                    let Some(s) = b.value_to_string(a, &lt) else {
                        return Err(LowerError::UnsupportedConstruct(format!(
                            "string concatenation with `{lt}` needs value-to-string conversion, which is not implemented"
                        )));
                    };
                    return Ok(b.concat_strings(s, bv));
                }
            }
            // RECORD equality is VALUE equality: `record R(int V);` makes
            // `new R(1) == new R(1)` true, comparing field by field. A
            // record is a class here, so the general path below would
            // compare references and answer false — which is what it did
            // while the parser discarded the `record` keyword.
            if matches!(
                op,
                crate::models::BinOp::Eq | crate::models::BinOp::Ne
            ) && let (Ok(lt), Ok(rt)) =
                (static_type_of(lhs, b, ctx), static_type_of(rhs, b, ctx))
                && lt == rt
                && let Some(info) = ctx.table.class(&lt)
                && info.is_record
            {
                let n = info.fields.len();
                let a = eval(lhs, b, ctx)?;
                let r = eval(rhs, b, ctx)?;
                let mut acc: Option<VReg> = None;
                for k in 0..n {
                    // The field's OWN slot: a hardcoded field start
                    // reads the wrong words as soon as the layout
                    // gains a header slot.
                    let off = (info.fields[k].offset_slots * 8) as u32;
                    let av = b.fresh();
                    b.push_raw(Instr::LoadField {
                        dst: av,
                        base: a,
                        off,
                    });
                    let bv = b.fresh();
                    b.push_raw(Instr::LoadField {
                        dst: bv,
                        base: r,
                        off,
                    });
                    let eq = b.emit(Instr::Cmp {
                        dst: VReg(0),
                        op: crate::native::ir::CmpOp::Eq,
                        a: av,
                        b: bv,
                    });
                    acc = Some(match acc {
                        None => eq,
                        Some(prev) => b.emit(Instr::Bin {
                            dst: VReg(0),
                            op: IrBinOp::BitAnd,
                            a: prev,
                            b: eq,
                        }),
                    });
                }
                // A record with NO fields compares equal whenever both
                // sides are non-null; a single-field record is just the
                // one comparison. `!=` inverts the whole result.
                let base = acc.unwrap_or_else(|| b.const_i64(1));
                if matches!(op, crate::models::BinOp::Ne) {
                    let one = b.const_i64(1);
                    return Ok(b.emit(Instr::Bin {
                        dst: VReg(0),
                        op: IrBinOp::BitXor,
                        a: base,
                        b: one,
                    }));
                }
                return Ok(base);
            }
            // Logical operators short-circuit like C#: `&&` skips its
            // right side when the left is false, `||` when true. The
            // old eager form evaluated throwing/null-dereferencing
            // right sides unconditionally (`x is string s &&
            // TryParse(s)` ran TryParse on a null `s`), aborting
            // programs before Main. Branches mirror ternary lowering
            // below (join value via a home vreg).
            if matches!(op, crate::models::BinOp::And | crate::models::BinOp::Or) {
                let is_and = matches!(op, crate::models::BinOp::And);
                let a = eval(lhs, b, ctx)?;
                // Constant left side decides at compile time (the dead
                // side never evaluates, so its effects vanish too).
                if let Some(&c) = b.const_vals.get(&a.0) {
                    if (is_and && c == 0) || (!is_and && c != 0) {
                        return Ok(b.const_i64(if is_and { 0 } else { 1 }));
                    }
                    let bv = eval(rhs, b, ctx)?;
                    return Ok(norm_bool(b, bv));
                }
                let home = b.fresh();
                let a = norm_bool(b, a);
                let rhs_b = b.make_block();
                let end_b = b.make_block();
                if is_and {
                    let false_b = b.make_block();
                    b.terminate(Terminator::Branch {
                        cond: Some(a),
                        if_true: rhs_b,
                        if_false: false_b,
                    });
                    b.switch_to(false_b);
                    let z = b.const_i64(0);
                    b.mov(home, z);
                    if !b.is_sealed() {
                        b.goto(end_b);
                    }
                } else {
                    let true_b = b.make_block();
                    b.terminate(Terminator::Branch {
                        cond: Some(a),
                        if_true: true_b,
                        if_false: rhs_b,
                    });
                    b.switch_to(true_b);
                    let o = b.const_i64(1);
                    b.mov(home, o);
                    if !b.is_sealed() {
                        b.goto(end_b);
                    }
                }
                b.switch_to(rhs_b);
                let bv = eval(rhs, b, ctx)?;
                let bv = norm_bool(b, bv);
                b.mov(home, bv);
                if !b.is_sealed() {
                    b.goto(end_b);
                }
                b.switch_to(end_b);
                return Ok(home);
            }
            // Flatten deep left-nested same-op chains iteratively:
            // `a+b+c+...` hundreds deep (lookup tables) would
            // otherwise overflow the compiler via eval recursion.
            // Operands evaluate left-to-right and combine pairwise
            // with the exact per-pair logic below — same order, same
            // calls, same traps, no recursion. Only pure
            // arithmetic/bitwise ops (no branches, no control flow);
            // anything else keeps the legacy path. Only spines
            // longer than 2 (pairs keep the exact legacy path).
            {
                use crate::models::BinOp as B;
                let flattenable = matches!(
                    op,
                    B::Add
                        | B::Sub
                        | B::Mul
                        | B::Div
                        | B::Rem
                        | B::BitAnd
                        | B::BitOr
                        | B::BitXor
                        | B::Shl
                        | B::Shr
                        | B::ShrShr
                );
                if flattenable {
                    let mut rights: Vec<&Expr> = vec![rhs];
                    let mut cur: &Expr = lhs;
                    while let Expr::Binary {
                        op: op2, lhs: l2, rhs: r2, ..
                    } = cur
                    {
                        if *op2 != *op {
                            break;
                        }
                        rights.push(r2);
                        cur = l2;
                    }
                    if rights.len() > 2 {
                        // Left-to-right operand order.
                        let mut operands: Vec<&Expr> = vec![cur];
                        operands.extend(rights.into_iter().rev());
                        // Types and float-ness drive per-pair dispatch
                        // (string/float); each operand is shallow (only
                        // the spine itself was deep).
                        let mut acc_ty = static_type_of(operands[0], b, ctx)?;
                        let mut acc_float = is_float_expr(b, ctx, operands[0]);
                        let mut acc = eval(operands[0], b, ctx)?;
                        for rhs_e in &operands[1..] {
                            let curr_ty = static_type_of(rhs_e, b, ctx)?;
                            let curr_float = is_float_expr(b, ctx, rhs_e);
                            let bv = eval(rhs_e, b, ctx)?;
                            let (nv, nt) = combine_binary(
                                op, acc, &acc_ty, acc_float, bv, &curr_ty, curr_float,
                                b, ctx,
                            )?;
                            acc = nv;
                            acc_ty = nt;
                            acc_float = acc_float || curr_float;
                        }
                        return Ok(acc);
                    }
                }
            }
            // Null-coalescing (`??`) evaluates lazily like C#: left
            // first; the right side (often `throw`) runs only when the
            // left is null. Branches mirror `&&`/`||` above (join value
            // via a home vreg). Non-nullable value-typed left sides
            // can never be null: yield them directly without touching
            // the right side (which C# treats as unreachable there).
            if matches!(op, crate::models::BinOp::NullCoalesce) {
                let a = eval(lhs, b, ctx)?;
                let lhs_ty = static_type_of(lhs, b, ctx)
                    .unwrap_or_else(|_| "object".to_owned());
                let lhs_base = base_type_name(&lhs_ty);
                let nonnull_value = !lhs_ty.ends_with('?')
                    && (matches!(
                        lhs_base.as_str(),
                        "SByte" | "Byte" | "Int16" | "UInt16" | "Int32" | "UInt32" | "Int64"
                            | "UInt64" | "IntPtr" | "UIntPtr" | "Char" | "Single" | "Double"
                            | "Decimal" | "Boolean" | "sbyte" | "byte" | "short" | "ushort"
                            | "int" | "uint" | "long" | "ulong" | "nint" | "nuint" | "char"
                            | "float" | "double" | "decimal" | "bool"
                    ) || ctx
                        .table
                        .class(&lhs_base)
                        .is_some_and(|c| c.is_struct));
                if nonnull_value {
                    return Ok(a);
                }
                // Constant left side decides at compile time (the dead
                // side never evaluates, so its effects vanish too).
                if let Some(&c) = b.const_vals.get(&a.0) {
                    if c != 0 {
                        return Ok(a);
                    }
                    return eval(rhs, b, ctx);
                }
                let home = b.fresh();
                let zero = b.const_i64(0);
                let is_null = b.emit(Instr::Cmp {
                    dst: VReg(0),
                    op: CmpOp::Eq,
                    a,
                    b: zero,
                });
                let rhs_b = b.make_block();
                let end_b = b.make_block();
                let left_b = b.make_block();
                b.terminate(Terminator::Branch {
                    cond: Some(is_null),
                    if_true: rhs_b,
                    if_false: left_b,
                });
                b.switch_to(left_b);
                b.mov(home, a);
                if !b.is_sealed() {
                    b.goto(end_b);
                }
                b.switch_to(rhs_b);
                let bv = eval(rhs, b, ctx)?;
                b.mov(home, bv);
                if !b.is_sealed() {
                    b.goto(end_b);
                }
                b.switch_to(end_b);
                return Ok(home);
            }
            let a = eval(lhs, b, ctx)?;
            // `is` takes a PATTERN, not a value: evaluating the right
            // side as an expression would emit a call for a pattern
            // marker and resolve a type name as a variable.
            if matches!(op, crate::models::BinOp::Is) {
                // `x is null` on a non-nullable value type (or a generic
                // parameter) is constant false; that check needs the
                // subject's TYPE, so it stays here. Everything else is a
                // pattern, and one helper answers them all: type tests,
                // constants, relational, tuple, list, property and the
                // `and`/`or` chains over any of those.
                if matches!(rhs.as_ref(), Expr::Literal(crate::models::Literal::Null)) {
                    if let Ok(ty) = static_type_of(lhs, b, ctx) {
                        let is_generic_param = ctx.method_vars.iter().any(|g| g == &ty)
                            || ctx.current_info.is_some_and(|info| {
                                info.generic_params.iter().any(|g| g == &ty)
                            });
                        if is_value_comparable(ctx.table, &ty) || is_generic_param {
                            return Ok(b.const_i64(0));
                        }
                    }
                    let zero = b.const_i64(0);
                    return Ok(b.emit(Instr::Cmp {
                        dst: VReg(0),
                        op: CmpOp::Eq,
                        a,
                        b: zero,
                    }));
                }
                let sty = static_type_of(lhs, b, ctx)
                    .unwrap_or_else(|_| "object".to_owned());
                return emit_is_pattern(b, ctx, a, &sty, rhs);
            }
            let bv = eval(rhs, b, ctx)?;
            // Pointer arithmetic scales here too (the pair path —
            // see `pointer_arith`).
            {
                let acc_ty = static_type_of(lhs, b, ctx)
                    .unwrap_or_else(|_| "object".to_owned());
                let curr_ty = static_type_of(rhs, b, ctx)
                    .unwrap_or_else(|_| "object".to_owned());
                if let Some((v, _)) = pointer_arith(op, a, &acc_ty, bv, &curr_ty, b) {
                    return Ok(v);
                }
            }
            // First-class string operators (C# semantics): `+` concatenates,
            // `==`/`!=` compare by (length, bytes).
            if static_type_of(lhs, b, ctx)? == "string"
                && static_type_of(rhs, b, ctx)? == "string"
            {
                match op {
                    crate::models::BinOp::Add => {
                        let len_a = {
                            let d = b.fresh();
                            b.push_raw(Instr::LoadField { dst: d, base: a, off: 0 });
                            d
                        };
                        let len_b = {
                            let d = b.fresh();
                            b.push_raw(Instr::LoadField { dst: d, base: bv, off: 0 });
                            d
                        };
                        let total = b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: len_a, b: len_b });
                        let s16 = b.const_i64(16);
                        // The header length is a CHARACTER count, but the
                        // data is UTF-16: allocation, pointer advance and
                        // both MemCpy lengths are BYTE counts. Copying
                        // `len` bytes produced half a string ("12"+"345"
                        // gave "1" then garbage) — measured.
                        let two = b.const_i64(2);
                        let total_bytes = b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Mul, a: total, b: two });
                        let total_sz = b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: total_bytes, b: s16 });
                        let newobj = b.fresh();
                        b.push_raw(Instr::AllocDyn { dst: newobj, bytes: total_sz });
                        b.push_raw(Instr::StoreField { base: newobj, off: 0, src: total });
                        let one = b.const_i64(1);
                        b.push_raw(Instr::StoreField { base: newobj, off: 8, src: one });
                        let data_new = b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: newobj, b: s16 });
                        let data_a = b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: a, b: s16 });
                        let len_a_bytes = b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Mul, a: len_a, b: two });
                        b.raw_vreg(Instr::MemCpy { dst: data_new, src: data_a, len: len_a_bytes });
                        let ptr2 = b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: data_new, b: len_a_bytes });
                        let data_b = b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: bv, b: s16 });
                        let len_b_bytes = b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Mul, a: len_b, b: two });
                        b.raw_vreg(Instr::MemCpy { dst: ptr2, src: data_b, len: len_b_bytes });
                        return Ok(newobj);
                    }
                    crate::models::BinOp::Eq | crate::models::BinOp::Ne => {
                        let len_a = {
                            let d = b.fresh();
                            b.push_raw(Instr::LoadField { dst: d, base: a, off: 0 });
                            d
                        };
                        let len_b = {
                            let d = b.fresh();
                            b.push_raw(Instr::LoadField { dst: d, base: bv, off: 0 });
                            d
                        };
                        let len_eq = b.emit(Instr::Cmp { dst: VReg(0), op: CmpOp::Eq, a: len_a, b: len_b });
                        let s16 = b.const_i64(16);
                        let data_a = b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: a, b: s16 });
                        let data_b = b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::Add, a: bv, b: s16 });
                        let same = b.fresh();
                        b.push_raw(Instr::MemCmp { dst: same, a: data_a, b: data_b, len: len_a });
                        let eq = b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitAnd, a: len_eq, b: same });
                        let one = b.const_i64(1);
                        let res = match op {
                            crate::models::BinOp::Eq => eq,
                            _ => b.emit(Instr::Bin { dst: VReg(0), op: IrBinOp::BitXor, a: eq, b: one }),
                        };
                        return Ok(res);
                    }
                    _ => {}
                }
            }
            // Constant folding: literal-only arithmetic collapses to one
            // Const. Wrapping matches the runtime; division folds only when
            // defined (zero divisor and i64::MIN / -1 stay runtime traps).
            if !op.is_comparison()
                && let (Some(&va), Some(&vb)) = (b.const_vals.get(&a.0), b.const_vals.get(&bv.0))
            {
                let int32 = is_int32_spelling(&static_type_of(lhs, b, ctx).unwrap_or_default())
                    && is_int32_spelling(&static_type_of(rhs, b, ctx).unwrap_or_default());
                let uint32 = is_uint32_spelling(&static_type_of(lhs, b, ctx).unwrap_or_default())
                    && is_uint32_spelling(&static_type_of(rhs, b, ctx).unwrap_or_default());
                let shift = matches!(
                    op,
                    crate::models::BinOp::Shl
                        | crate::models::BinOp::Shr
                        | crate::models::BinOp::ShrShr
                );
                // A folded `int` SHIFT would use the unmasked count
                // (`1 << 32` is 1 in C#), so shifts take the runtime
                // path; other folded `int` results wrap at 32 bits.
                if !((int32 || uint32) && shift)
                    && let Some(v) = fold_arith(op, va, vb)
                {
                    let v = if int32 {
                        (v as i32) as i64
                    } else if uint32 {
                        v & 0xFFFF_FFFF
                    } else {
                        v
                    };
                    return Ok(b.emit(Instr::Const {
                        dst: VReg(0),
                        imm: v,
                    }));
                }
            }
            if op.is_comparison()
                && let (Some(&va), Some(&vb)) = (b.const_vals.get(&a.0), b.const_vals.get(&bv.0))
            {
                use crate::models::BinOp as B;
                let folded = match op {
                    B::Eq => va == vb,
                    B::Ne => va != vb,
                    B::Lt => va < vb,
                    B::Gt => va > vb,
                    B::Le => va <= vb,
                    B::Ge => va >= vb,
                    _ => unreachable!("comparison filter above"),
                };
                return Ok(b.emit(Instr::Const {
                    dst: VReg(0),
                    imm: i64::from(folded),
                }));
            }
            if op.is_comparison() {
                use crate::models::BinOp as B;
                // User-defined comparison operators (`EqEq`, `Less`,
                // … — named for the token kind by the parser): when
                // the left operand's type declares a matching static
                // 2-arity operator, call it instead of comparing raw
                // values. Raw comparison is address comparison — wrong
                // for structs with operators (`TimeSpan`, `DateTime`)
                // and strings. Falls back to direct comparison when
                // no operator exists (primitives, enums,
                // operator-less classes — reference equality there is
                // correct). Only the left type is consulted (C#
                // requires an operand to be the declaring type;
                // `null == x` keeps direct comparison, which is
                // correct for null).
                let op_name = match op {
                    B::Eq => "EqEq",
                    B::Ne => "NotEq",
                    B::Lt => "Less",
                    B::Gt => "Greater",
                    B::Le => "LessEq",
                    B::Ge => "GreaterEq",
                    _ => unreachable!("comparison filter above"),
                };
                let lhs_ty = static_type_of(lhs, b, ctx).unwrap_or_default();
                let mut op_fidx: Option<u32> = None;
                {
                    let simple = lhs_ty.rsplit('.').next().unwrap_or(&lhs_ty);
                    for spelling in [
                        lhs_ty.as_str(),
                        simple,
                    ] {
                        if let Some(f) =
                            static_function_index_for(ctx, spelling, op_name, 2, &[], &[])
                        {
                            op_fidx = Some(f);
                            break;
                        }
                    }
                }
                if let Some(fidx) = op_fidx {
                    let dst = b.fresh();
                    b.push_raw(Instr::CallStatic {
                        dst,
                        fidx,
                        args: vec![a, bv],
                    });
                    return Ok(dst);
                }
                // `x == null` / `x != null` on non-nullable value
                // types is constant (C# never-null): without this,
                // `key == null` with a `long` key of 0 reads true and
                // `FindValue` throws, aborting ALC init on the first
                // `AllContexts.Add`. Generic parameters (`TKey`)
                // fold the same way: open bodies can't distinguish a
                // value-0 from a null reference, and folding false
                // keeps valid value code running (a genuine null
                // reference key slips through to a loud fault
                // downstream instead of a clean throw — documented
                // gamble until monomorphization carries T-kind).
                // Pointers, references, and nullables keep the address
                // compare.
                if matches!(op, B::Eq | B::Ne) {
                    let other = if matches!(
                        lhs.as_ref(),
                        Expr::Literal(crate::models::Literal::Null)
                    ) {
                        Some(rhs.as_ref())
                    } else if matches!(
                        rhs.as_ref(),
                        Expr::Literal(crate::models::Literal::Null)
                    ) {
                        Some(lhs.as_ref())
                    } else {
                        None
                    };
                    if let Some(o) = other {
                        if let Ok(ty) = static_type_of(o, b, ctx) {
                            let is_generic_param = ctx.method_vars.iter().any(|g| g == &ty)
                                || ctx.current_info.is_some_and(|info| {
                                    info.generic_params.iter().any(|g| g == &ty)
                                });
                            if is_value_comparable(ctx.table, &ty) || is_generic_param {
                                let is_eq = matches!(op, B::Eq);
                                return Ok(b.const_i64(if is_eq { 0 } else { 1 }));
                            }
                        }
                    }
                }
                if is_float_expr(b, ctx, lhs) || is_float_expr(b, ctx, rhs) {                    let fcmpop = match op {
                        B::Eq => FCmpOp::Eq,
                        B::Ne => FCmpOp::Ne,
                        B::Lt => FCmpOp::Lt,
                        B::Gt => FCmpOp::Gt,
                        B::Le => FCmpOp::Le,
                        B::Ge => FCmpOp::Ge,
                        _ => unreachable!("comparison filter above"),
                    };
                    let dst = b.fresh();
                    b.push_raw(Instr::FCmp {
                        dst,
                        op: fcmpop,
                        a,
                        b: bv,
                    });
                    return Ok(dst);
                }
                let cmpop = match op {
                    B::Eq => CmpOp::Eq,
                    B::Ne => CmpOp::Ne,
                    B::Lt => CmpOp::Lt,
                    B::Gt => CmpOp::Gt,
                    B::Le => CmpOp::Le,
                    B::Ge => CmpOp::Ge,
                    _ => unreachable!("comparison filter above"),
                };
                return Ok(b.emit(Instr::Cmp {
                    dst: VReg(0),
                    op: cmpop,
                    a,
                    b: bv,
                }));
            }
            // Type check (`is`): `x is null` folds to a null
            // comparison (pervasive in guards like `IsNullOrEmpty`);
            // general type tests stay parse-only false for now.

            // Logical AND/OR — lower as Cmp-based boolean operations.
            if matches!(op, crate::models::BinOp::And | crate::models::BinOp::Or) {
                let is_and = matches!(op, crate::models::BinOp::And);
                let zero = b.const_i64(0);
                let a_nonzero = b.emit(Instr::Cmp {
                    dst: VReg(0),
                    op: CmpOp::Ne,
                    a,
                    b: zero,
                });
                let bv_nonzero = b.emit(Instr::Cmp {
                    dst: VReg(0),
                    op: CmpOp::Ne,
                    a: bv,
                    b: zero,
                });
                let binop = if is_and {
                    IrBinOp::BitAnd
                } else {
                    IrBinOp::BitOr
                };
                return Ok(b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: binop,
                    a: a_nonzero,
                    b: bv_nonzero,
                }));
            }
            // IEEE 754 arithmetic when either operand is a float. Only the
            // four arithmetic ops have a float form; `Rem` and the bitwise
            // ops fall through to the integer path.
            if is_float_expr(b, ctx, lhs) || is_float_expr(b, ctx, rhs) {
                if let Some(fbinop) = match op {
                    crate::models::BinOp::Add => Some(FBinOp::Add),
                    crate::models::BinOp::Sub => Some(FBinOp::Sub),
                    crate::models::BinOp::Mul => Some(FBinOp::Mul),
                    crate::models::BinOp::Div => Some(FBinOp::Div),
                    _ => None,
                } {
                    let dst = b.fresh();
                    b.push_raw(Instr::FBin {
                        dst,
                        op: fbinop,
                        a,
                        b: bv,
                    });
                    return Ok(dst);
                }
            }
            let mut binop = match op {
                crate::models::BinOp::Add => IrBinOp::Add,
                crate::models::BinOp::Sub => IrBinOp::Sub,
                crate::models::BinOp::Mul => IrBinOp::Mul,
                crate::models::BinOp::Div => IrBinOp::Div,
                crate::models::BinOp::Rem => IrBinOp::Rem,
                crate::models::BinOp::BitAnd => IrBinOp::BitAnd,
                crate::models::BinOp::BitOr => IrBinOp::BitOr,
                crate::models::BinOp::BitXor => IrBinOp::BitXor,
                crate::models::BinOp::Shl => IrBinOp::Shl,
                crate::models::BinOp::Shr => IrBinOp::Shr,
                crate::models::BinOp::ShrShr => IrBinOp::ShrShr,
                _ => unreachable!("logical/comparison handled above"),
            };
            // C# `int` is 32 bits. When BOTH operands are `int`, the
            // result must wrap at 32 bits and shifts must mask their
            // count (`1 << 32` is `1`, because C# masks by 0x1F). Mixed
            // `int`/`long` arithmetic is 64-bit and is left alone.
            let lt = static_type_of(lhs, b, ctx).unwrap_or_default();
            let rt = static_type_of(rhs, b, ctx).unwrap_or_default();
            let int32_pair = is_int32_spelling(&lt) && is_int32_spelling(&rt);
            // `uint` arithmetic is 32-bit UNSIGNED: the result is the low
            // 32 bits, zero-extended (masking two's-complement low bits is
            // correct even when the i64 product wrapped).
            let uint32_pair = is_uint32_spelling(&lt) && is_uint32_spelling(&rt);
            // `/` and `%` on unsigned results use `div`, not `idiv`.
            if matches!(op, crate::models::BinOp::Div | crate::models::BinOp::Rem)
                && is_unsigned_result(&numeric_result(&lt, &rt))
            {
                binop = if matches!(op, crate::models::BinOp::Div) {
                    IrBinOp::UDiv
                } else {
                    IrBinOp::URem
                };
            }
            let shift = matches!(
                op,
                crate::models::BinOp::Shl
                    | crate::models::BinOp::Shr
                    | crate::models::BinOp::ShrShr
            );
            // A shift's RESULT type is the LEFT operand's type; C# does
            // not combine it with the count's type (`u << 32` is a uint
            // shift by 0). Counts mask by width-1 (0x1F / 0x3F).
            let shift_int32 = shift && is_int32_spelling(&lt);
            let shift_uint32 = shift && is_uint32_spelling(&lt);
            let shift_wide = shift
                && matches!(
                    lt.trim(),
                    "long" | "Int64" | "System.Int64" | "ulong" | "UInt64" | "System.UInt64"
                );
            let shift_b = if shift_int32 || shift_uint32 {
                let m = b.const_i64(31);
                b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::BitAnd,
                    a: bv,
                    b: m,
                })
            } else if shift_wide {
                let m = b.const_i64(63);
                b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::BitAnd,
                    a: bv,
                    b: m,
                })
            } else {
                bv
            };
            let r = b.emit(Instr::Bin {
                dst: VReg(0),
                op: binop,
                a,
                b: shift_b,
            });
            // Inside `checked(...)`, integer +, - and * must raise instead
            // of wrapping. Comparisons, floats and pointers never reach
            // here (they take their own paths above).
            if b.checked_depth > 0 {
                let kind = if int32_pair || shift_int32 {
                    crate::native::ir::lower::body::CheckedKind::Int32
                } else if uint32_pair || shift_uint32 {
                    crate::native::ir::lower::body::CheckedKind::UInt32
                } else {
                    crate::native::ir::lower::body::CheckedKind::Int64
                };
                b.checked_arith_guard(binop, a, shift_b, r, ctx, kind)?;
            }
            let r = if int32_pair || shift_int32 {
                narrow_int32(b, r)
            } else if uint32_pair || shift_uint32 {
                mask_uint32(b, r)
            } else {
                r
            };
            Ok(r)
        }
        Expr::New {
            class_name, args, init, ..
        } => {
            // Generic type parameters (e.g. `new T()`, `new T[]`), nullable
            // Generic type parameters (e.g. `new T()`, `new T[]`), nullable
            // types, and arrays of generic params are not concrete classes —
            // return 0 as a placeholder.
            // Qualified construction (`new
            // ThreadPoolWorkQueue.WorkStealingQueue()`): resolve the entry
            // through the qualified spelling FIRST, so the
            // implicit-default-ctor rule (and layout) use the right type
            // instead of a same-named sibling's explicit ctors. Bare names
            // prefer the current class's nested types (C# scoping), else
            // first-match (legacy).
            // Strip namespace prefix (e.g. `global::Foo.Bar` -> `Bar`).
            let qualified = class_name.trim().strip_prefix("global::").unwrap_or(class_name.as_str()).to_owned();
            let class_name = {
                // Simple name = last dot-separated segment OUTSIDE
                // angle brackets (`A.B<C.D>` → `B<C.D>`; a naive
                // rsplit gives `D>`). Generic arguments stay attached:
                // resolution is arity-aware below.
                let c = qualified.as_str();
                let mut depth = 0i32;
                let mut cut = None;
                for (i, ch) in c.char_indices() {
                    match ch {
                        '<' => depth += 1,
                        '>' => depth = depth.saturating_sub(1),
                        '.' if depth == 0 => cut = Some(i),
                        _ => {}
                    }
                }
                cut.map(|i| c[i + 1..].to_owned())
                    .unwrap_or_else(|| c.to_owned())
            };
            let _base_name = {
                let mut s = class_name.clone();
                loop {
                    if let Some(pos) = s.rfind('[') {
                        s.truncate(pos);
                    } else if s.ends_with('?') {
                        s.pop();
                    } else {
                        break;
                    }
                }
                s
            };
            let Some(info) = (if qualified.contains('.') {
                class_for_spelling(ctx.table, &qualified)
            } else if class_name.contains('<') {
                // Generic instantiation (`new Task<VoidTaskResult>`):
                // arity-aware definition lookup so the generic
                // entry's own ctors are selected — a stripped lookup
                // lands on a same-named non-generic class and
                // misconstructs (or throws, leaving singletons null).
                class_for_spelling(ctx.table, &class_name)
            } else {
                resolve_scoped(ctx.table, &class_name, ctx.current_class)
                    .and_then(|q| ctx.table.class(&q))
                    .or_else(|| ctx.table.class(&class_name))
            }) else {
                let _ = eval_args(args, b, ctx)?;
                return Ok(b.const_i64(0));
            };
            // Same-named construction fallback: if the resolved entry
            // offers no applicable instance ctor, retry the other
            // same-named entries and adopt the first that constructs
            // (`new Compiler()` with `using XsltOld`: first-match
            // `Xslt.Compiler` has only a 3-arg ctor, while
            // `XsltOld.Compiler` has the implicit zero-arg one).
            // Layout/vtable follow the adopted entry; genuinely
            // unconstructible calls still error below.
            let info = {
                let probe_types: Vec<String> = args
                    .iter()
                    .map(|a| {
                        if is_method_group(a, b, ctx) {
                            return "method-group".to_owned();
                        }
                        overload_arg_type(a, b, ctx)
                    })
                    .collect();
                let accepts = |e: &crate::native::semantic::ClassInfo| {
                    select_visible(ctx.table, std::slice::from_ref(&e), &class_name, &probe_types, Some(false), ctx.current_class, None).is_some()
                        || (args.is_empty()
                            && !e.methods.iter().any(|m| {
                                m.name == e.name && m.declaring_class == e.name && !m.is_static
                            }))
                };
                if accepts(info) {
                    info
                } else {
                    entries_for(ctx.table, &class_name)
                        .into_iter()
                        // Identity, not key: same-key cross-assembly
                        // collisions (MLC vs CoreLib `DefaultBinder`)
                        // are distinct entries that must each be tried.
                        .find(|e| !std::ptr::eq(*e, info) && accepts(e))
                        .unwrap_or(info)
                }
            };
            let obj = b.fresh();
            // Span construction packs instead of running ctor bodies
            // (see `span.rs`): reference arithmetic in real bodies
            // (`Unsafe.Add`) is outside our ref model, so `new
            // ReadOnlySpan<T>(array|string)` and `(array, start,
            // length)` materialize packed views directly. Pointer/ref
            // forms fall through to the legacy path.
            if let Some((_, inner)) = span_view(&info.key.path) {
                let arg_types: Vec<String> = args
                    .iter()
                    .map(|a| {
                        overload_arg_type(a, b, ctx)
                    })
                    .collect();
                let span_ty = info.key.path.clone();
                if args.len() == 1 {
                    let a0 = eval(&args[0], b, ctx)?;
                    if arg_types[0].ends_with("[]") {
                        return pack_span_from_array(b, ctx, a0, &span_ty, &inner);
                    }
                    if is_string_type(&arg_types[0]) {
                        return pack_span_from_string(b, ctx, a0, &span_ty, &inner);
                    }
                } else if args.len() == 3 && arg_types[0].ends_with("[]") {
                    let a0 = eval(&args[0], b, ctx)?;
                    let start = eval(&args[1], b, ctx)?;
                    let length = eval(&args[2], b, ctx)?;
                    let full =
                        pack_span_from_array(b, ctx, a0, &span_ty, &inner)?;
                    return span_slice(b, ctx, full, &span_ty, &inner, start, Some(length));
                }
            }
            // The header is vtable + refcount + runtime type (the
            // instantiation), then the instance fields.
            b.alloc(
                obj,
                (info.fields.len() + INSTANCE_HEADER_SLOTS) as u64 * 8,
            );
            let decl_opt = construction_decl(ctx, &class_name);
            if decl_opt.is_none() && !info.owner.is_empty() {
                return lower_external_new(info, args, &class_name, b, ctx);
            }
            // Vtable row: position among table-backed declarations in
            // emission order (mirrors `Program` row construction).
            // Table order does NOT work: it carries extra entries and
            // a different order, so every `new` misdispatched virtuals.
            let vi = match decl_opt {
                Some(decl) => {
                    let n = construction_row(ctx, decl);
                    n
                }
                None => 0,
            };
            let vt = b.fresh();
            b.vtable_addr(vt, vi as u32);
            b.push_raw(Instr::StoreField { base: obj, off: 0, src: vt });
            // The instance's RUNTIME TYPE: the descriptor of the
            // instantiation, which carries a generic class's type
            // arguments. `typeof(T)` inside a method of the class
            // reads the argument out of it.
            // A construction whose type argument is still a type
            // VARIABLE (`new Memory<T>()` inside a generic method)
            // has no concrete runtime type yet, so the slot stays
            // null and `typeof(T)` reports that rather than reading
            // an answer that is not there.
            let runtime_type = if info.generic_params.is_empty()
                || type_meta(&class_name, ctx).has_type_var
            {
                None
            } else {
                Some(b.type_token(&class_name, ctx)?)
            };
            match runtime_type {
                Some(desc) => b.push_raw(Instr::StoreField {
                    base: obj,
                    off: (RUNTIME_TYPE_SLOT * 8) as u32,
                    src: desc,
                }),
                None => {
                    let none = b.const_i64(0);
                    b.push_raw(Instr::StoreField {
                        base: obj,
                        off: (RUNTIME_TYPE_SLOT * 8) as u32,
                        src: none,
                    });
                }
            }
            // Zero every field slot up front (C# `new` semantics):
            // freelist-reused memory holds stale data, and a later
            // reference-field store releases the old value — garbage
            // there corrupts live refcounts and faults far from the
            // cause. Field initializers and ctor bodies overwrite
            // after this. Refcount starts at 1 (slot 1).
            {
                let z = b.const_i64(0);
                let one = b.const_i64(1);
                b.push_raw(Instr::StoreField {
                    base: obj,
                    off: 8,
                    src: one,
                });
                let mut off = (FIRST_FIELD_SLOT * 8) as u32;
                let end = (info.fields.len() as u32 + INSTANCE_HEADER_SLOTS as u32) * 8;
                while off < end {
                    b.push_raw(Instr::StoreField {
                        base: obj,
                        off,
                        src: z,
                    });
                    off += 8;
                }
            }

            if let Some(decl) = decl_opt {
                // Explicit args evaluate once, up front (side effects in
                // order, values reused for primary binding + ctor call).
                let mut explicit: Vec<VReg> = Vec::new();
                for a in args {
                    explicit.push(eval(a, b, ctx)?);
                }
                // Primary-constructor params bind positionally for field
                // initializers (C# scoping: `ref struct R(H fd)` with a
                // `= fd` initializer reads the construction argument).
                // A fresh scope keeps them from leaking past the `new`.
                // Setup mirrors `Let` (declare + home + move + heap
                // tracking) with the argument's static type, since the
                // AST keeps primary names only.
                let primary = decl.primary_ctor_params.clone();
                if !primary.is_empty() {
                    b.scope_enter();
                }
                for (i, pname) in primary.iter().enumerate() {
                    let Some(v) = explicit.get(i).copied() else {
                        continue;
                    };
                    if b.scopes.lookup_here(pname).is_some() {
                        continue;
                    }
                    let ty = args
                        .get(i)
                        .map(|a| {
                            if is_method_group(a, b, ctx) {
                                return "method-group".to_owned();
                            }
                            overload_arg_type(a, b, ctx)
                        })
                        .unwrap_or_else(|| "object".to_owned());
                    let Ok(slot) =
                        b.scopes
                            .declare(pname, &ty, crate::models::Access::Default)
                    else {
                        continue;
                    };
                    while b.bindings.len() <= slot {
                        let fresh = b.fresh();
                        b.bindings.push(fresh);
                    }
                    b.mov(b.bindings[slot], v);
                    if let Some(bytes) = b.handle_bytes(&ty, ctx) {
                        let home = b.bindings[slot];
                        b.heap_locals.push((home, bytes, b.depth));
                    }
                }
                // Field initializers run before the constructor body. Statics
                // are handled at program start; they have no object slot.
                // Initializers evaluate in the CONSTRUCTED class's context
                // (a shadowed `Ctx`): bare names refer to its members
                // (`_currentElement = s_boxedFalse` reads the constructed
                // nested class's own static), never the caller's locals.
                // Base-declared instance fields initialize first
                // (root-first), then own: only own-decl members were
                // visited, leaving base-initialized fields zeroed
                // (`TextWriter.CoreNewLine` null faulted `WriteLine`
                // on the null Length read). Each level evaluates in
                // its own declaring context.
                let mut init_chain: Vec<&crate::models::ClassDecl> = Vec::new();
                for base_name in info.bases.iter().rev() {
                    if let Some(bdecl) = construction_decl(ctx, base_name) {
                        if !std::ptr::eq(bdecl, decl) && !init_chain.iter().any(|d: &&crate::models::ClassDecl| std::ptr::eq(*d, bdecl)) {
                            init_chain.push(bdecl);
                        }
                    }
                }
                init_chain.push(decl);
                for edecl in init_chain {
                    let init_ctx = Ctx {
                        table: ctx.table,
                        class_paths: ctx.class_paths,
                        current_class: &edecl.name,
                        current_info: None,
                        // The PATH must name the class being
                        // initialized: the caller's path made the
                        // access rule compare unrelated types (a
                        // private const of the class's own field
                        // initializer was denied).
                        current_path: ctx
                            .class_paths
                            .get(&(edecl as *const crate::models::ClassDecl))
                            .map(|s| s.as_str()),
                        classes: ctx.classes,
                        import_index: ctx.import_index,
                        dep_imports: ctx.dep_imports,
                        host_import_count: ctx.host_import_count,
                        static_index: ctx.static_index,
                        defines: ctx.defines,
                        method_vars: &[],
                type_subst: None,
                        local_fns: &[],
                        current_method: None,
                        current_ret: None,
                        lambda_deferred: None,
                    lambda_base: 0,
                    };
                    for f in &edecl.members {
                    if let ClassMember::Field(fd) = f
                        && !fd.is_static
                        && !fd.is_const
                    {
                        // NOTE: `const` fields are compile-time only
                        // (no storage, folded at use sites) — emitting
                        // them here stored e.g. `CharSize = 2` at
                        // offset 0, clobbering the vtable word and
                        // faulting every later virtual call on the
                        // object.
                        let v = eval(&fd.init, b, &init_ctx)?;
                        let src_ty = static_type_of(&fd.init, b, &init_ctx)
                            .unwrap_or_else(|_| "object".to_owned());
                        let v = b.coerce_boxing(v, &src_ty, &fd.type_name, false, &init_ctx)?;
                        let off =
                            (info.field(&fd.name).map(|fi| fi.offset_slots).unwrap_or(1) * 8) as u32;
                        b.push_raw(Instr::StoreField { base: obj, off, src: v });
                    }
                    }
                }

                // Constructor selection: same-class ctor with matching
                // arity. `new C()` without any ctor keeps old behavior.
                // Typed table selection (see `pick_overload`): ctors may
                // differ in access/types, not just arity (e.g.
                // `DateTimeOffset(long, TimeSpan)` vs a private
                // same-arity overload) — arity-only picks the wrong one.
                let arg_types: Vec<String> = args
                    .iter()
                    .map(|a| {
                        if is_method_group(a, b, ctx) {
                            return "method-group".to_owned();
                        }
                        overload_arg_type(a, b, ctx)
                    })
                    .collect();
                // Typed ctor selection across same-named entries
                // (multi-arity generic instantiations like
                // `new TupleSlim<A,B,C>` share the bare name):
                // visibility-preferred throughout (see
                // `select_visible`). Layout/vtable stay first-match:
                // known generics-layout gap.
                // Span/ROS 2-arg construction over a raw pointer
                // (`new Span<T>(pointer, length)`): store directly (see
                // `span_from_pointer_view`). Width-free, so open `T` is
                // fine — and it must bypass selection, which can prefer
                // the `(ref T, int)` twin for `byte*` arguments (no
                // `byte*`→`void*` conversion rule) and fault inside.
                // Pure applicability, then evaluate once in order.
                if args.len() == 2
                    && arg_types.len() == 2
                    && span_view(&class_name).is_some()
                    && span_class_path(ctx, &class_name).is_some()
                {
                    let at = arg_types[0].trim();
                    if is_pointer_ty(at) {
                        let ptr = eval(&args[0], b, ctx)?;
                        let len = eval(&args[1], b, ctx)?;
                        if let Some(v) =
                            span_from_pointer_view(b, ctx, ptr, len, &class_name)?
                        {
                            return Ok(v);
                        }
                    }
                }
                // Span/ROS 3-arg construction
                // (`new Span<T>(array, start, length)`) with a concrete
                // array element type inlines here with the caller's
                // width (see `span_from_array_view`): the shared body
                // folds `Unsafe.Add` with the OPEN variable (width 8).
                // Applicability is pure so falling through never
                // double-runs side effects.
                if args.len() == 3
                    && arg_types.len() == 3
                    && span_view(&class_name).is_some()
                    && span_class_path(ctx, &class_name).is_some()
                {
                    let applicable = array_elem_ty(&arg_types[0]).is_some_and(|e| {
                        !e.trim().is_empty() && closed_elem_width(e.trim()).is_some()
                    });
                    if applicable {
                        let elem = array_elem_ty(&arg_types[0]).unwrap_or_default();
                        let arr = eval(&args[0], b, ctx)?;
                        let start = eval(&args[1], b, ctx)?;
                        let len = eval(&args[2], b, ctx)?;
                        if let Some(v) =
                            span_from_array_view(b, ctx, arr, start, len, &elem, &class_name)?
                        {
                            return Ok(v);
                        }
                    }
                }
                let entries = entries_for(ctx.table, &class_name);
                let selected_ctor = select_visible(
                    ctx.table,
                    &entries,
                    &class_name,
                    &arg_types,
                    Some(false),
                    ctx.current_class,
                    None,
                )
                // Alias-spelled construction (`new WorkQueue()` for
                // `using WorkQueue = Queue<object>`): the call-site
                // spelling never matches ctor names — retry with the
                // resolved entry's own name before erroring.
                .or_else(|| {
                    if info.name == class_name {
                        None
                    } else {
                        select_visible(
                            ctx.table,
                            &entries,
                            &info.name,
                            &arg_types,
                            Some(false),
                            ctx.current_class,
                            None,
                        )
                    }
                });
                // Arity-known miss (ctors exist, none accept the args):
                // keep the loud error rather than chaining a base ctor.
                // Only the RESOLVED entry's own ctors count: same-named
                // siblings (Lock.State vs PeriodicTimer.State,
                // PortableThreadPool vs ThreadPoolWorkQueue
                // WorkStealingQueue) must not force arity errors on each
                // other. A type declaring no instance ctors has the
                // implicit public parameterless one, so zero-arg
                // construction stays silent — and structs keep theirs
                // even with explicit ctors (e.g. Asn1's `AttributeAsn`
                // declares only `AttributeAsn(AsnEncodedData)` yet
                // `new AttributeAsn()` is legal C#); classes are exact
                // here.
                let has_ctors = info.methods.iter().any(|m| {
                    // Instance ctors only: a static constructor
                    // (`static TypeScope()`) is never callable via
                    // `new` and must not suppress the implicit public
                    // parameterless one.
                    m.name == info.name && m.declaring_class == info.name && !m.is_static
                });
                if selected_ctor.is_none() && has_ctors && !(info.is_struct && args.is_empty()) {
                    return Err(LowerError::UnsupportedConstruct(crate::tf!(
                        "rubyc.native.ir.lower_expr.ctor_arity",
                        class_name.clone(),
                        args.len().to_string()
                    )));
                }
                if let Some(mi) = selected_ctor {
                    ctx.check_method_access(mi)?;
                    // Named ctor arguments (`new Foo(b: 1, a: 2)`): the
                    // values were evaluated in written order, so permute
                    // them into parameter order here (side effects keep
                    // their written order either way).
                    if let Some(order) = named_arg_order(args, mi)? {
                        let mut reordered: Vec<VReg> = Vec::with_capacity(order.len());
                        for (i, written) in order.into_iter().enumerate() {
                            match written {
                                Some(w) => reordered.push(explicit[w]),
                                None => reordered.push(default_arg_value(i, mi, b)),
                            }
                        }
                        explicit = reordered;
                    }
                    let mut argv = vec![obj];
                    argv.extend(explicit);
                    // Fill omitted optional ctor params from the folded
                    // defaults recorded on the ctor's MethodInfo.
                    if argv.len() - 1 < mi.param_types.len() {
                        let mut tail: Vec<VReg> = argv.drain(1..).collect();
                        fill_default_args(&mut tail, mi, b);
                        argv.extend(tail);
                    }
                    if !mi.has_body {
                        // Bodyless ctor (`extern`/partial-no-impl): no
                        // function exists; the allocated object stands.
                        // Object initializers still run (they are the
                        // caller's writes, not the ctor's).
                        lower_object_inits(b, ctx, info, obj, init)?;
                        if !primary.is_empty() {
                            b.scope_exit();
                        }
                        return Ok(obj);
                    }
                    let n = overload_position(
                        ctx,
                        &mi.declaring_class,
                        &mi.name,
                        mi.param_types.len(),
                        &mi.param_types,
                        &mi.param_is_byref,
                    );
                    // Overload-aware ctor index: same-name+arity
                    // overloads resolve to the selected one's slot.
                    // Alias-spelled construction: the function index
                    // keys on declaring names (`Queue::Queue#0`, not
                    // `WorkQueue::WorkQueue#0`).
                    let idx = function_index_nth(
                        ctx,
                        &class_name,
                        &class_name,
                        mi.param_types.len(),
                        n,
                    )
                    .or_else(|| {
                        function_index_nth(
                            ctx,
                            &mi.declaring_class,
                            &mi.name,
                            mi.param_types.len(),
                            n,
                        )
                    })
                    .unwrap_or(0);
                    let call_dst = b.fresh();
                    b.push_raw(Instr::CallFn {
                        dst: call_dst,
                        fidx: idx as u32,
                        args: argv,
                    });
                } else {
                    // No explicit constructor: chain EVERY base's
                    // parameterless constructor, not just the first.
                    // Each base's constructor writes at ITS OWN class's
                    // offsets, so a base sitting behind another base's
                    // fields needs an adjusted receiver — the same rule
                    // `@start` uses for the entry instance. Chaining only
                    // `bases.first()` left every second base's state
                    // uninitialised (`class Both : Left, Right` got
                    // Left's field and not Right's).
                    let mut chain: Vec<&crate::models::ClassDecl> = Vec::new();
                    let mut seen: Vec<String> = Vec::new();
                    fn collect<'a>(
                        d: &'a crate::models::ClassDecl,
                        classes: &[&'a crate::models::ClassDecl],
                        seen: &mut Vec<String>,
                        out: &mut Vec<&'a crate::models::ClassDecl>,
                    ) {
                        if seen.iter().any(|n| n == &d.name) {
                            return;
                        }
                        seen.push(d.name.clone());
                        for b in &d.bases {
                            let simple = b.split('<').next().unwrap_or(b).trim();
                            if let Some(base) =
                                classes.iter().find(|c| c.name == simple)
                            {
                                collect(base, classes, seen, out);
                            }
                        }
                        out.push(d);
                    }
                    collect(decl, ctx.classes, &mut seen, &mut chain);
                    // Subobject offset per class, from the DERIVED
                    // class's real layout: the first field slot that
                    // belongs to that base (or its ancestors), minus the
                    // first field slot. Counting declarations was wrong
                    // whenever the layout differed from the count — and a
                    // wrong offset made a base constructor write INTO a
                    // neighbouring object (measured: one object's field
                    // turned into the next object's value).
                    let derived_fields = ctx
                        .table
                        .class(&decl.name)
                        .map(|i| i.fields.clone())
                        .unwrap_or_default();
                    let mut offs: Vec<(&str, u32)> = Vec::new();
                    for c in chain.iter() {
                        let first = derived_fields
                            .iter()
                            .filter(|f| {
                                f.declaring_class == c.name
                                    || ctx.table.derives_from(&f.declaring_class, &c.name)
                            })
                            .map(|f| f.offset_slots)
                            .min();
                        let off = match first {
                            // The base's own layout starts its fields at
                            // the first field slot, so the difference
                            // between the two layouts is the adjustment.
                            Some(slot) => {
                                slot.saturating_sub(FIRST_FIELD_SLOT) as u32 * 8
                            }
                            None => 0,
                        };
                        offs.push((c.name.as_str(), off));
                    }
                    for (cname, off) in offs {
                        if cname == decl.name {
                            continue;
                        }
                        let Some(idx) = function_index_for(ctx, cname, cname, 0)
                        else {
                            continue;
                        };
                        let recv = if off == 0 {
                            obj
                        } else {
                            let o = b.const_i64(off as i64);
                            b.emit(Instr::Bin {
                                dst: VReg(0),
                                op: IrBinOp::Add,
                                a: obj,
                                b: o,
                            })
                        };
                        let call_dst = b.fresh();
                        b.push_raw(Instr::CallFn {
                            dst: call_dst,
                            fidx: idx as u32,
                            args: vec![recv],
                        });
                    }
                }
                if !primary.is_empty() {
                    b.scope_exit();
                }
            }
            // Object initializers run after the constructor body, in
            // element order (C# semantics).
            lower_object_inits(b, ctx, info, obj, init)?;
            Ok(obj)
        }
        Expr::Member { object, name, .. } => {
            let type_name = static_type_of(object, b, ctx)?;
            // A System.Type value is one of our native descriptors.
            // Its properties are fields; a CoreLib RuntimeType has a
            // different layout and is never the object we hand out.
            if is_descriptor_receiver(&type_name)
                && let Some((off, bit)) = type_member(name)
            {
                let base = eval(object, b, ctx)?;
                if off == u32::MAX {
                    // RuntimeTypeHandle is the descriptor pointer.
                    return Ok(base);
                }
                let dst = b.fresh();
                b.push_raw(Instr::LoadField { dst, base, off });
                let Some(mask) = bit else {
                    return Ok(dst);
                };
                let m = b.const_i64(mask as i64);
                let bits = b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: IrBinOp::BitAnd,
                    a: dst,
                    b: m,
                });
                let zero = b.const_i64(0);
                return Ok(b.emit(Instr::Cmp {
                    dst: VReg(0),
                    op: CmpOp::Ne,
                    a: bits,
                    b: zero,
                }));
            }
            // Tuple element read (`t.Item1`): tuples lower as dense slot
            // arrays, so `ItemN` is slot N-1. The declared spelling is
            // `(int, int)` (or `tuple[]` after inference).
            {
                let tr = type_name.trim();
                let is_tuple = tr == "tuple"
                    || tr
                        .strip_suffix("[]")
                        .map(|inner| inner.trim() == "tuple")
                        .unwrap_or(false)
                    || (tr.starts_with('(') && tr.contains(','));
                if is_tuple
                    && name.len() > 4
                    && name.starts_with("Item")
                    && name[4..].chars().all(|c| c.is_ascii_digit())
                    && let Ok(n) = name[4..].parse::<i64>()
                    && n >= 1
                {
                    let synth = Expr::Index {
                        object: object.clone(),
                        index: Box::new(Expr::Literal(crate::models::Literal::Int(n - 1))),
                        extra: Vec::new(),
                        is_range: false,
                        span: crate::models::Span::new(0, 0),
                    };
                    return eval(&synth, b, ctx);
                }
            }
            // Member reads on a `System.Type` receiver are answered by
            // the descriptor path above (the canonical names and
            // predicates). A member the descriptor does not carry needs
            // a real reflection surface and is reported, not guessed.
            // `string.Length` — a first-class string stores its byte length
            // at slot 0 (offset 0).
            if type_name == "string" && name == "Length" {                let base = eval(object, b, ctx)?;
                let dst = b.fresh();
                b.push_raw(Instr::LoadField { dst, base, off: 0 });
                return Ok(dst);
            }
            // `MethodTable*` dereference reads (`pMT->IsSzArray`,
            // `pMT->ComponentSize`, ...): our runtime keeps no CoreCLR
            // method tables (objects carry only a vtable pointer,
            // arrays a length word), so the pointed-to struct does not
            // exist. Fold what the SZ-array paths need: everything is
            // single-dimensional (rank 0), elements live in uniform
            // 8-byte slots, and no GC-pointer tracking exists
            // (elements are unrefcounted — consistent with the plain
            // element stores). This lets `Array.Copy`'s fast path and
            // the `GetLength`/`GetLowerBound` rank checks run.
            if let Expr::Unary { op: UnOp::Deref, .. } = object.as_ref()
                && matches!(
                    name.as_str(),
                    "IsSzArray"
                        | "ComponentSize"
                        | "ContainsGCPointers"
                        | "MultiDimensionalArrayRank"
                )
            {
                let _ = eval(object, b, ctx)?;
                let c = match name.as_str() {
                    "IsSzArray" => 1,
                    "ComponentSize" => 8,
                    _ => 0, // ContainsGCPointers, MultiDimensionalArrayRank
                };
                return Ok(b.const_i64(c));
            }
            // `RawArrayData` overlay reads
            // (`Unsafe.As<RawArrayData>(arr).Length` / `.Data`): the
            // overlay class exists in the table, but its declared
            // offsets (Length@16, Data@32 with padding) describe the
            // CLR layout, not ours (length at +0, elements at +16).
            // Fold to our layout instead. `.Data` yields the element
            // ADDRESS (all uses are `ref`-position; `ref` erases at
            // parse, so a value-position read would miscompile —
            // none exists in the framework paths).
            if base_type_name(&type_name) == "RawArrayData" {
                if name == "Length" {
                    let base = eval(object, b, ctx)?;
                    let dst = b.fresh();
                    b.push_raw(Instr::LoadField { dst, base, off: 0 });
                    return Ok(dst);
                }
                if name == "Data" {
                    let base = eval(object, b, ctx)?;
                    let sixteen = b.const_i64(16);
                    return Ok(b.emit(Instr::Bin {
                        dst: VReg(0),
                        op: IrBinOp::Add,
                        a: base,
                        b: sixteen,
                    }));
                }
            }
            // `RawData` overlay (`Unsafe.As<RawData>(arr).Data` points
            // at the component count): same treatment, data at +0.
            if base_type_name(&type_name) == "RawData" && name == "Data" {
                return Ok(eval(object, b, ctx)?);
            }
            // `string` interior fields: RubyC's builtin layout keeps the length
            // at +0 and character data at +16, while the table would lay
            // them out as ordinary slots (16/24). Lower them from the
            // single layout owner in `semantic` (see
            // `string_interior_read`; shared with the implicit-`this`
            // read path for a bare payload read).
            if crate::native::semantic::builtin_string_interior_offset(&type_name, name).is_some() {
                let base = eval(object, b, ctx)?;
                return Ok(string_interior_read(b, base, name));
            }
            // Nullable `.HasValue` / `.Value` — placeholders.
            if type_name.ends_with('?') && (name == "HasValue" || name == "Value") {
                return Ok(b.const_i64(0));
            }
            // Array `.Length` — stored at offset 0 of the array header.
            // Generic instantiations resolve through their definition
            // entry first (`ROS<byte>.Length` is a property, not an
            // array header read) — but a `[]`-suffixed spelling is
            // ALWAYS an array (`Box<int>[]`), never an instantiation,
            // so it takes the header path unconditionally.
            if name == "Length"
                && (array_elem_ty(&type_name).is_some()
                    && !ctx.table.class(&type_name).is_some()
                    || (!ctx.table.class(&type_name).is_some()
                        && definition_entry(ctx.table, &type_name).is_none()))
            {
                let base = eval(object, b, ctx)?;
                let dst = b.fresh();
                b.push_raw(Instr::LoadField {
                    dst,
                    base,
                    off: 0,
                });
                return Ok(dst);
            }
            let Some(first) = ctx
                .table
                .class(&type_name)
                .or_else(|| definition_entry(ctx.table, &type_name))
                // Nullable-annotated receivers (`EncoderNLS?`): the
                // annotation is not a different type — resolve members
                // through the base type (mirrors the call-site
                // `nullable_base` arm). Without this, property reads
                // on nullable receivers yield the unknown placeholder
                // (0), so `encoder.FallbackBuffer` nulled the fallback
                // buffer and console Flush faulted at si_addr=0x48
                // (`LoadField(0, 72)`). Value types keep `Nullable<T>`
                // semantics (`nullable_base` stays out for structs).
                .or_else(|| nullable_base(ctx.table, &type_name))
                // Dotted type recovery (mirrors `static_type_of`):
                // when the receiver types as `object` (Body-state
                // sensitive resolution failure) but its dotted
                // spelling names a real entry, use it. Without this,
                // qualified consts like
                // `Internal.PaddingHelpers.CACHE_LINE_SIZE` read 0.
                .or_else(|| {
                    (type_name == "object" || type_name == "_")
                        .then(|| {
                            dotted_spelling(object)
                                .and_then(|d| ctx.table.class(&d).map(|e| (e, d)))
                        })
                        .flatten()
                        .map(|(e, _)| e)
                })
            else {
                // An unresolved receiver (an open generic parameter `T`, an
                // uninferred temporary, a `_` discard) has no entry to
                // check membership against, so a member read is not
                // diagnosable here — evaluate for side effects and
                // yield the default. A receiver that DID resolve to a
                // class entry and simply lacks the member falls through
                // to the honest `no_member` error below.
                let _ = eval(object, b, ctx)?;
                return Ok(b.const_i64(0));
            };
            // Same-named nested classes may conflate: prefer the entry
            // actually declaring the member (see `member_host`).
            let info = if first.field(name).is_some()
                || first
                    .own_members
                    .iter()
                    .any(|(n, is_f, _)| n == name && *is_f)
            {
                first
            } else {
                member_host(ctx.table, &type_name, name).unwrap_or(first)
            };
            let field = info.field(name);
            if field.is_none()
                && !info.own_members.iter().any(|(n, is_f, _)| n == name && *is_f)
                && info.property(name).is_none()
            {
                // Member not found as a field, own member, or known
                // property — return 0 as a placeholder (e.g.
                // expression-bodied property).
                let _ = eval(object, b, ctx)?;
                return Ok(b.const_i64(0));
            }
            ctx.check_access(true, &type_name, name)?;
            // `const` field: no storage — the read is replaced by the
            // constant-folded literal.
            if let Some(f) = field
                && f.is_const
            {
                return match f.const_value {
                    Some(ConstValue::Int(i)) => Ok(b.const_i64(i)),
                    Some(ConstValue::Float(fv)) => {
                        let dst = b.fresh();
                        b.push_raw(Instr::FConst { dst, bits: fv.to_bits() });
                        Ok(dst)
                    }
                    // String constants materialize exactly like string
                    // literals (static UTF-8 data + object header).
                    Some(ConstValue::Str(ref s)) => {
                        let (off, len) = b.utf16_literal(s);
                        Ok(b.build_utf16_string(off, len))
                    }
                    // Null constants (e.g. `const string X = null`)
                    // read as the null handle.
                    Some(ConstValue::Null) => Ok(b.const_i64(0)),
                    None => Ok(b.const_i64(0)),
                };
            }
            // Static field: the value lives in per-class static storage,
            // independent of any receiver. Read [static_addr + 0].
            if field.is_some_and(|f| f.is_static) {
                let f = field.unwrap();
                // Evaluate the receiver for side effects only; it is not the
                // base address of a static.
                let _ = eval(object, b, ctx)?;
                // Static properties (including ISA gates like
                // `Sse2.IsSupported`) never reach the field branch
                // above (no backing storage); they inline through the
                // instance path below, where the static guard skips
                // virtual dispatch.
                let Some(base) = static_addr(f, b, ctx) else {
                    return Ok(b.const_i64(0));
                };
                let dst = b.fresh();
                b.push_raw(Instr::LoadField { dst, base, off: 0 });
                // Uninitialized struct statics read null, but member
                // access needs an object (`DateTime.MinValue.Ticks`
                // must see zero data, not address zero): materialize
                // a zeroed box on null. `Nullable<T>` and fieldless
                // primitives are exempt. Spelling-robust: field
                // types arrive short (`DateTime`) while keys are
                // full paths.
                let struct_entry = ctx
                    .table
                    .class(&f.type_name)
                    .filter(|e| e.is_struct && !e.fields.is_empty())
                    .or_else(|| {
                        let simple = f.type_name.rsplit('.').next().unwrap_or(&f.type_name);
                        let simple = simple.split('<').next().unwrap_or(simple);
                        ctx.table.classes.iter().find(|c| {
                            c.name == simple && c.is_struct && !c.fields.is_empty()
                        })
                    });
                if let Some(entry) = struct_entry
                    && !f.type_name.starts_with("Nullable<")
                    && !f.type_name.trim_end().ends_with('?')
                {
                    let home = b.fresh();
                    b.mov(home, dst);
                    let z = b.const_i64(0);
                    let nb = b.emit(Instr::Cmp {
                        dst: VReg(0),
                        op: crate::native::ir::CmpOp::Eq,
                        a: dst,
                        b: z,
                    });
                    let box_b = b.make_block();
                    let done_b = b.make_block();
                    b.terminate(Terminator::Branch {
                        cond: Some(nb),
                        if_true: box_b,
                        if_false: done_b,
                    });
                    b.switch_to(box_b);
                    let boxed = b.alloc_struct_box(ctx, entry, Some(&f.type_name));
                    b.mov(home, boxed);
                    b.goto(done_b);
                    b.switch_to(done_b);
                    return Ok(home);
                }
                return Ok(dst);
            }
            let base = eval(object, b, ctx)?;
            // Struct-typed receiver (e.g. embedded `_separated` with
            // no initializer): null is an uninitialized struct, not
            // a fault — materialize a zeroed box (see
            // `ensure_struct_base`). Valid boxes pass through, so
            // working struct reads are unchanged.
            let base = ensure_struct_base(b, ctx, base, &type_name);
            let mut dst = b.fresh();
            // An explicit getter beats a same-named backing field: an
            // abstract or auto base declares storage under the property
            // name, and reading it shadows the override (`d.P` read
            // zeros instead of `=> _w`; `CanWrite` read zeros instead
            // of `=> _writable`). A strict-subclass override beats a
            // base body too (virtual dispatch: `CSBase.CanWrite`
            // inlining `=> false` mis-serves `CSDerived` instances).
            // Sealed/single hierarchies keep the static path (exact).
            // Gated on a same-named property (plain fields skip the
            // decl/world walks entirely — they cost minutes across a
            // CoreLib-sized world otherwise). NOTE: `new`-hiding
            // (non-virtual base, shadowing derived) also matches the
            // strict walk and misdispatches to the derived body (C#
            // would use the base); override dwarfs hiding in practice.
            let has_prop = info.properties.iter().any(|p| p.name == *name);
            let upward_explicit = has_prop
                && most_derived_prop_decl(ctx, info, name).is_some_and(
                    |(decl, _)| decl.get_expr.is_some() || decl.get.is_some(),
                );
            // Abstract properties have neither an inlineable body nor
            // a backing field on the receiver entry, yet subclass
            // overrides must still dispatch (previously the bogus
            // auto-property backing field stood in here).
            let abstract_prop = info
                .properties
                .iter()
                .any(|p| p.name == *name && p.is_abstract);
            let strict_sub = strict_subclass_has_explicit(ctx, info, name);
            let strict_override = has_prop
                && (upward_explicit || field.is_some() || abstract_prop)
                && strict_sub;
            // Static properties never dispatch virtually: there is no
            // receiver instance (the class-as-value evaluates to the
            // null placeholder), so subclass overrides can never
            // apply and a virtual chain faults on its first vtable
            // load (`Sse2.IsSupported` dispatched over sibling ISA
            // classes, killing `Console.WriteLine` in
            // `EncodingHelper.GetCharset` via `IndexOfValueType`).
            // Static bodies inline below exactly like the sealed /
            // single-hierarchy path.
            let is_static_prop =
                info.property(name).is_some_and(|p| p.is_static);
            let prop_explicit =
                field.is_some() && (upward_explicit || strict_override);
            match field {
                Some(f) if !prop_explicit => {
                    b.push_raw(Instr::LoadField {
                        dst,
                        base,
                        off: (f.offset_slots * 8) as u32,
                    });
                    // Struct-typed field values that read null are
                    // uninitialized structs too (e.g. `counts` inside a
                    // fresh `_separated` box): box them so chained
                    // access (`NumThreadsGoal` getter) proceeds on
                    // zeros instead of faulting. Non-null values
                    // (valid boxes) pass through untouched.
                    if struct_entry_for(ctx.table, &f.type_name).is_some() {
                        dst = ensure_struct_base(b, ctx, dst, &f.type_name);
                    }
                }
                _ => {
                    // Accessor-backed property with an inlineable getter:
                    // evaluate `=> expr` (or a single-`return` block)
                    // with the receiver bound as `this`. Without this,
                    // ubiquitous properties (`span.Length`,
                    // `span.IsEmpty`, `list.Count`) read an uninitialized
                    // vreg. Block getters with real statements stay
                    // placeholders (future work); auto-properties have
                    // no body at all.
                    // Property accessors inline whenever a body exists; auto-
                    // properties have no body at all.
                    {
                        // A strict-subclass override wins over any base
                        // body (virtual dispatch; the chain includes the
                        // base arm for base-typed instances). Without
                        // one the static body (if any) serves; without
                        // either, placeholder zeros below. Skipping the
                        // inapplicable attempt keeps CoreLib-sized
                        // worlds fast (each attempt walks the world).
                        if strict_override && !is_static_prop {
                            if let Some(val) = eval_virtual_getter(
                                b, ctx, info, name, Some(object),
                            )? {
                                return Ok(val);
                            }
                        } else if let Some(val) = eval_property_getter(
                            b,
                            ctx,
                            info,
                            &type_name,
                            name,
                            Some(object),
                        )? {
                            return Ok(val);
                        }
                    }
                    // A property with no inlineable getter (abstract, or an
                    // accessor whose body has not been modelled) has no
                    // value to read. Returning 0 here made the read
                    // indistinguishable from a property holding
                    // default(T) — the case where the pre-allocated
                    // `dst` was never defined also used to leak an
                    // undefined vreg downstream.
                    return Err(LowerError::UnsupportedConstruct(crate::tf!(
                        "rubyc.native.ir.lower_expr.no_property_body",
                        type_name.clone(),
                        name.to_string()
                    )));
                }
            }
            Ok(dst)
        }
        Expr::MethodCall {
            object, name, args, ..
        } => {
            // `5.ToString()` / `true.ToString()`: primitive values are
            // unboxed, so there is no ToString to dispatch to and the
            // call used to jump through a raw integer as a vtable row.
            if name == "ToString"
                && args.is_empty()
                && let Ok(t) = static_type_of(object, b, ctx)
                && is_primitive_spelling(&t)
                && !is_string_type(&t)
            {
                let v = eval(object, b, ctx)?;
                let Some(s) = b.value_to_string(v, &t) else {
                    return Err(LowerError::UnsupportedConstruct(format!(
                        "`{t}.ToString()` needs value-to-string conversion, which is not implemented"
                    )));
                };
                return Ok(s);
            }
            // `E?.Invoke(args)` on an EVENT: walk the subscriber list and
            // call every delegate. The null-conditional around it already
            // tests `E != null`; the arguments are evaluated ONCE and
            // passed to each handler (C# semantics).
            if name == "Invoke"
                && let Expr::Ident { name: ev, .. } = object.as_ref()
                && let Some(field) = super::body::event_field(ctx, ev)
            {
                let off = (field.offset_slots * 8) as u32;
                let this = b.this_vreg();
                let argv = eval_args(args, b, ctx)?;
                let cur = b.fresh();
                b.push_raw(Instr::LoadField { dst: cur, base: this, off });
                let loop_b = b.make_block();
                let body_b = b.make_block();
                let cont_b = b.make_block();
                let done_b = b.make_block();
                b.goto(loop_b);
                b.switch_to(loop_b);
                let zero = b.const_i64(0);
                let none = b.fresh();
                b.push_raw(Instr::Cmp { dst: none, op: CmpOp::Eq, a: cur, b: zero });
                b.terminate(Terminator::Branch {
                    cond: Some(none),
                    if_true: done_b,
                    if_false: body_b,
                });
                b.switch_to(body_b);
                let d = b.fresh();
                b.push_raw(Instr::LoadField { dst: d, base: cur, off: 8 });
                let target = b.fresh();
                b.push_raw(Instr::LoadField { dst: target, base: d, off: 0 });
                let kind = b.fresh();
                b.push_raw(Instr::LoadField { dst: kind, base: d, off: 8 });
                let closure_b = b.make_block();
                let plain_b = b.make_block();
                b.terminate(Terminator::Branch {
                    cond: Some(kind),
                    if_true: closure_b,
                    if_false: plain_b,
                });
                b.switch_to(closure_b);
                {
                    let mut call_args = vec![d];
                    call_args.extend(argv.iter().copied());
                    let dst = b.fresh();
                    b.push_raw(Instr::CallFnAddr { dst, target, args: call_args });
                    b.goto(cont_b);
                }
                b.switch_to(plain_b);
                {
                    let dst = b.fresh();
                    b.push_raw(Instr::CallFnAddr { dst, target, args: argv.clone() });
                    b.goto(cont_b);
                }
                b.switch_to(cont_b);
                let nx = b.fresh();
                b.push_raw(Instr::LoadField { dst: nx, base: cur, off: 0 });
                b.mov(cur, nx);
                b.goto(loop_b);
                b.switch_to(done_b);
                return Ok(b.const_i64(0));
            }
            // Strip generic-args suffix (e.g. `Convert<T3, TResult>` ->
            // `Convert`) — the parser embeds `<...>` in the name.
            // The arity still constrains overload ranking below
            // (`services.AddOptions<NoOp>()` must not elect the
            // non-generic `AddOptions()` overload).
            let explicit_arity = match name.find('<') {
                Some(_) => Some(explicit_type_args(name).len()),
                None => None,
            };
            // Full generic type arguments for intrinsic-like methods
            // (`MemoryMarshal.Cast<char, short>` reinterprets widths).
            let explicit_targs: Vec<String> = match name.find('<') {
                Some(_) => explicit_type_args(name),
                None => Vec::new(),
            };
            let name = match name.find('<') {
                Some(pos) => &name[..pos],
                None => name.as_str(),
            };
            let type_name = static_type_of(object, b, ctx)?;
            // Dereference-nullable for member calls: `string?`
            // behaves as `string` for lookup, dispatch, slots, and
            // span rules (see `nullable_base`). Keep the SIMPLE
            // spelling (not the qualified path) so exact-match
            // fast paths (`== "string"`) keep working. Value types
            // never normalize (helper returns None for them).
            let type_name = if nullable_base(ctx.table, &type_name).is_some() {
                type_name
                    .trim()
                    .strip_suffix('?')
                    .map(|s| s.trim().to_owned())
                    .unwrap_or(type_name)
            } else {
                type_name
            };
            if name == "Slice"
                && let Some((_, inner)) = span_view(&type_name)
                && (args.len() == 1 || args.len() == 2)
            {
                let base = eval(object, b, ctx)?;
                let start = eval(&args[0], b, ctx)?;
                let len = if args.len() == 2 {
                    Some(eval(&args[1], b, ctx)?)
                } else {
                    None
                };
                return span_slice(b, ctx, base, &type_name, &inner, start, len);
            }
            // Reflection enumerations over reflection types (`Type`,
            // `Assembly`, `Module`, `MemberInfo` and kin): our
            // `typeof` boxes are placeholders with no metadata, and
            // the `InternalCall` natives backing real reflection
            // yield zero — either way there is nothing to enumerate,
            // and running the real bodies faults on null caches
            // (`typeof(Attribute).GetConstructors(...)[0]` faulted a
            // static init). Answer the documented placeholder-grade
            // default directly: plurals yield an empty array (indexing
            // it throws managed `IndexOutOfRangeException` via the
            // bounds check instead of faulting), singulars yield null
            // (the no-such-member answer). Gated on reflection-type
            // receivers so user methods sharing a name keep theirs.
            // Receivers outside the family run the real bodies, which
            // degrade to the same answers through the zero natives.
            {
                let sn = base_type_name(&type_name);
                let is_reflection = sn == "Type"
                    || sn.ends_with("Type")
                    || sn.ends_with("Assembly")
                    || sn.ends_with("Module")
                    || sn == "MemberInfo"
                    || sn.ends_with("MemberInfo");
                // `MethodBase`/`ConstructorInfo`/`Type` etc. end in the
                // matched suffixes already; `MemberInfo` itself needs
                // the exact arm (`CustomAttributeData` lives below).
                let plural = matches!(
                    name,
                    "GetConstructors"
                        | "GetMethods"
                        | "GetFields"
                        | "GetProperties"
                        | "GetEvents"
                        | "GetNestedTypes"
                        | "GetInterfaces"
                        | "GetMembers"
                        | "GetCustomAttributes"
                        | "GetCustomAttributesData"
                        | "GetEnumNames"
                        | "GetGenericArguments"
                        | "GetGenericParameterConstraints"
                        | "FindMembers"
                );
                let singular = matches!(
                    name,
                    "GetConstructor"
                        | "GetMethod"
                        | "GetField"
                        | "GetProperty"
                        | "GetEvent"
                        | "GetNestedType"
                        | "GetInterface"
                        | "GetMember"
                        | "GetCustomAttribute"
                        | "GetElementType"
                );
                if is_reflection && (plural || singular) {
                    let _ = eval(object, b, ctx)?;
                    let _ = eval_args(args, b, ctx)?;
                    if singular {
                        return Ok(b.const_i64(0));
                    }
                    let empty = Expr::NewArray {
                        elem_type: "object".to_owned(),
                        size: Box::new(Expr::Literal(Literal::Int(0))),
                        init: vec![],
                        span: crate::models::Span::new(0, 0),
                    };
                    return eval(&empty, b, ctx);
                }
            }
            // `obj.GetType()` on non-array receivers: `Object.GetType`
            // is extern (InternalCall, no managed body), so it has no
            // vtable implementation — virtual dispatch reads a MAX hole
            // (zero) and faults (killed `EventSource` construction via
            // `this.GetType()`). Fold to a `typeof` box for the
            // receiver's static type: exact whenever static == runtime
            // (sealed classes, locally-constructed values, `this` in
            // leaf classes — the common cases); an approximation for
            // derived instances through base-typed receivers
            // (documented; full runtime-type tracking is future work).
            // Never null, virtual-safe via the placeholder row.
            // Arrays keep the row-token fold below (identity semantics).
            if name == "GetType" && args.is_empty() {
                let rt = static_type_of(object, b, ctx)
                    .unwrap_or_else(|_| "object".to_owned());
                let tr = rt.trim().to_owned();
                // An array's runtime type IS its static type (`int[]`).
                if let Some(elem) = array_elem_ty(&tr) {
                    let _ = eval(object, b, ctx)?;
                    return b.type_token(&format!("{elem}[]"), ctx);
                }
                if base_type_name(&tr) == "Array" {
                    // The element type of an `Array`-typed receiver is
                    // not in the static type; the answer is the array
                    // type itself (a real descriptor, not a row
                    // address pretending to be a Type).
                    let _ = eval(object, b, ctx)?;
                    return b.type_token("System.Array", ctx);
                }
                if !tr.ends_with("[]") {
                    let recv = eval(object, b, ctx)?;
                    return runtime_type_of(b, recv, &tr, ctx);
                }
            }
            // `arr.GetType()` on arrays (and `Array`-typed receivers):
            // arrays carry no vtable word (length at +0), so the
            // inherited virtual `GetType` would read length-as-vtable
            // and fault — and `Object` declares no body to devirtualize
            // to. Fold to the shared `System.Array` vtable row: a
            // stable per-kind token, so same-kind comparisons
            // (`sourceArray.GetType() == destinationArray.GetType()`
            // in `CopyImpl`) answer correctly. Identity against
            // `typeof(X)` and member reads off the token are future
            // work (same placeholder grade as `typeof` boxes).
            if name == "GetType"
                && args.is_empty()
                && (array_elem_ty(&type_name).is_some()
                    || base_type_name(&type_name) == "Array")
            {
                let _ = eval(object, b, ctx)?;
                let dst = b.fresh();
                b.push_raw(Instr::VTableAddr {
                    dst,
                    row: vtable_row_for(ctx, "Array", "System.Array"),
                });
                return Ok(dst);
            }
            // Search-optimization predicates (see `span.rs`): range
            // detection and nibble/vector heuristics need generic
            // math (`Min`/`Max` over `T`), comparers, and SIMD
            // feature probes our backend cannot do. Answering false
            // is always safe — callers fall through to correct
            // general paths (`AnyN`, `Ascii`, `Bitmap` searchers).
            // Names are unique to the search machinery.
            // `SearchValues.Create` has no fold: the real factory
            // decides. Handing back an empty searcher (a value that
            // matches nothing) changed program RESULTS, which no
            // capability fold may do; where the body needs machinery
            // this compiler does not have yet, the build says so.
            if matches!(
                name,
                "TryGetSingleRange"
                    | "CanUseUniqueLowNibbleSearch"
                    | "CanUsePackedIndexOf"
                    | "ShouldUseProbabilisticMap"
            ) {
                let _ = eval(object, b, ctx)?;
                let _ = eval_args(args, b, ctx)?;
                return Ok(b.const_i64(0));
            }

            // Delegate-typed receivers (`Action continuation`): member
            // lookup goes through `System.Delegate` — every delegate
            // derives from it, but delegate declarations carry no
            // members of their own, so a direct table hit misses base
            // methods (`continuation.GetTargetForSingleCastInstanceDelegate`
            // on `Action`). Diverts only when the resolved class entry
            // has NO same-named method at all — a real class member
            // always wins (this also skips unrelated same-named
            // classes, which first-match would otherwise select). `Invoke` is excluded: invocation
            // (direct or explicit) has its own placeholder path and
            // must not resolve as a member.
            let type_name = {
                let base = type_name
                    .rsplit('.')
                    .next()
                    .unwrap_or(&type_name)
                    .split('<')
                    .next()
                    .unwrap_or(&type_name)
                    .to_owned();
                let class_has_member = ctx
                    .table
                    .class(&type_name)
                    .is_some_and(|info| info.methods.iter().any(|m| m.name == name));
                if name != "Invoke"
                    && !class_has_member
                    && ctx.table.delegates.iter().any(|d| d.name == base)
                    && ctx.table.class("System.Delegate").is_some()
                {
                    "System.Delegate".to_owned()
                } else {
                    type_name
                }
            };

            // Explicit `x.Invoke(...)` on a delegate-typed receiver is
            // invocation, not member lookup: placeholders (real dispatch
            // is runtime work, like other delegate paths). Fires only
            // when no class entry carries an `Invoke` member, so real
            // `Invoke` methods (e.g. `MethodInvoker.Invoke`) still
            // resolve normally.
            if name == "Invoke"
                && ctx.table.delegates.iter().any(|d| {
                    d.name
                        == type_name
                            .rsplit('.')
                            .next()
                            .unwrap_or(&type_name)
                            .split('<')
                            .next()
                            .unwrap_or(&type_name)
                })
                && !ctx.table.class(&type_name).is_some_and(|info| {
                    info.methods.iter().any(|m| m.name == "Invoke")
                })
            {
                let _ = eval(object, b, ctx)?;
                let _ = eval_args(args, b, ctx)?;
                return Ok(b.const_i64(0));
            }

            // Explicit-arg static types for overload applicability (pure:
            // no IR emitted). Unresolvable args type as `object` and act
            // as wildcards, mirroring the scan's conservatism. Bare
            // method names mark as method groups (convertible to
            // compatible delegates only — see `compat_score`).
            let arg_types: Vec<String> = args
                .iter()
                .map(|a| {
                    if is_method_group(a, b, ctx) {
                        return "method-group".to_owned();
                    }
                    overload_arg_type(a, b, ctx)
                })
                .collect();

            // Import stubs take priority over virtual dispatch — but only
            // when the stub's arity matches the call (`new
            // RuntimeTypeHandle(this).MakePointer()` targets the
            // 0-arg instance overload, not the 2-arg QCall stub).
            // Mismatches fall through to normal overload resolution.
            // Static ABI: arguments only, no receiver.
            // Import index keys use AST declaration names (often simple:
            // `Nat`), while inferred receiver types are qualified
            // (`app.Nat`): try both spellings (same mismatch that broke
            // operator dispatch; a miss silently zeroes P/Invoke results).
            let import_key = ctx
                .import_index
                .get(&(type_name.clone(), name.to_string()))
                .or_else(|| {
                    type_name
                        .rsplit('.')
                        .next()
                        .and_then(|s| ctx.import_index.get(&(s.to_owned(), name.to_string())))
                });
            if let Some(&index) = import_key
                && ctx
                    .table
                    .imports
                    .get(index)
                    .is_some_and(|imp| imp.param_types.len() == args.len())
            {
                // Static imports (`Interop.Sys.Foo(...)`) have no
                // receiver value (static ABI: arguments only) — only
                // instance-shaped receivers need evaluation. A dotted
                // receiver that names a type is static-style: skip it
                // (types have no runtime value or side effects).
                // Anything else keeps the original restriction.
                if !matches!(object.as_ref(), Expr::Ident { .. }) {
                    let is_type = dotted_spelling(object)
                        .is_some_and(|s| ctx.table.class(&s).is_some());
                    if !is_type {
                        return Err(LowerError::UnsupportedConstruct(
                            "imports may only be called on a plain variable or this".into(),
                        ));
                    }
                } else {
                    let _ = eval(object, b, ctx)?;
                }
                check_import(ctx, index, args)?;
                let mut argv = eval_args(args, b, ctx)?;
                // Native C ABI: strings transcode to UTF-8 temps, and
                // SafeHandle returns wrap to live objects (see the two
                // helpers above — raw pointers/fds fault downstream).
                marshal_string_utf8_args(b, &mut argv, &arg_types);
                let ret_ty = ctx
                    .table
                    .imports
                    .get(index)
                    .map(|imp| imp.ret_type.clone())
                    .unwrap_or_default();
                let dst = b.fresh();
                b.push_raw(Instr::CallImport {
                    dst,
                    import: index as u32,
                    args: argv,
                });
                return Ok(wrap_safehandle_import_return(b, ctx, dst, &ret_ty));
            }

            // The receiver's static type is either a class or an interface.
            // Both dispatch through the implementing class's vtable, so the
            // only difference is how we confirm the method exists.
            // A bare class-name identifier as the receiver is static-style
            // syntax (`C.M(...)`); anything else is instance syntax. C#
            // never mixes the two: statics aren't candidates for instance
            // calls (so inherited statics like `Object.Equals` can't
            // shadow instance overloads or extensions) and vice versa.
            // Canonicalized so primitive aliases (`decimal.ToUInt32`)
            // count as static receivers. A DOTTED path to a type
            // (`Outer.Inner.M(...)`, `System.Foo.M(...)`) is likewise
            // static-style — unless rooted at a value (`this`, locals,
            // fields), which keeps instance syntax.
            let is_static_receiver = matches!(object.as_ref(), Expr::Ident { name: oname, .. } if canonical_primitive(oname) == type_name)
                || is_type_receiver(object, &type_name, b, ctx);
            let mut method_static: Option<bool> = None;
            let mut selected_arity: usize = args.len();
            // Declaring class of the selection: function lookup keys on
            // it, not the (possibly aliased) receiver spelling
            // (`MdToken.IsNullToken` lives at `MetadataToken::...`).
            let mut selected_declaring: String = type_name.clone();
            // Span/ROS `Slice(start[, length])` with a concrete inner
            // type inlines here with the caller's element width (see
            // `span_slice_view`): the shared definition body folds its
            // address arithmetic with the OPEN type variable (width 8),
            // so `Slice(1)` on a byte span advanced 8. Structs can't
            // derive, so no override is shadowed; open inners keep
            // normal virtual dispatch below. Applicability is pure
            // (no evaluation) so falling through never double-runs
            // side effects.
            if name == "Slice" && (args.len() == 1 || args.len() == 2) {
                let applicable = span_view(&type_name).is_some_and(|(_, inner)| {
                    closed_elem_width(inner.trim()).is_some()
                        && span_class_path(ctx, &type_name).is_some()
                });
                if applicable {
                    let recv = eval(object, b, ctx)?;
                    let start = eval(&args[0], b, ctx)?;
                    let len = if args.len() == 2 {
                        Some(eval(&args[1], b, ctx)?)
                    } else {
                        None
                    };
                    if let Some(v) = span_slice_view(b, ctx, recv, start, len, &type_name)? {
                        return Ok(v);
                    }
                }
            }
            // Span/ROS `Fill(value)` with a concrete inner type
            // inlines here with the caller's element width (see
            // `inline_span_fill`): the shared bodies run vectorized
            // (no SIMD) or stride-8 scalar (open `T`), silently
            // writing nothing. Same guards/position as `Slice` above.
            if name == "Fill" && args.len() == 1 {
                let applicable = span_view(&type_name).is_some_and(|(_, inner)| {
                    closed_elem_width(inner.trim()).is_some()
                        && span_class_path(ctx, &type_name).is_some()
                });
                if applicable {
                    if let Some(v) = inline_span_fill(b, ctx, object, args, &type_name)? {
                        return Ok(v);
                    }
                }
            }
            // Same-named entries once (generic-arity siblings share the
            // bare receiver name); selection below is shared by
            // access/fill/dispatch so they agree. A generic-spelled
            // receiver (`Vector512<byte>`) leads with its arity-matched
            // definition entry.
            let mut entries = entries_for(ctx.table, &type_name);
            if let Some(def) = class_for_spelling(ctx.table, &type_name) {
                if !entries.iter().any(|e| e.key == def.key) {
                    entries.insert(0, def);
                }
            }
            // `[Conditional("SYM")]` calls omit unless `SYM` is defined:
            // when every same-named candidate carries an undefined
            // conditional symbol, the call — including receiver and
            // argument evaluation — vanishes. Mixed sets resolve
            // normally below; a conditional winner erases after
            // selection.
            {
                let mut matched = 0;
                let mut erasable = true;
                for e in entries.iter() {
                    for m in e.methods.iter().filter(|m| m.name == name) {
                        matched += 1;
                        if m.conditional_symbol
                            .as_ref()
                            .is_none_or(|s| ctx.defines.contains(s))
                        {
                            erasable = false;
                        }
                    }
                }
                if matched > 0 && erasable {
                    return Ok(b.const_i64(0));
                }
            }
            let selected = select_visible(
                ctx.table,
                &entries,
                name,
                &arg_types,
                Some(is_static_receiver),
                ctx.current_class,
                explicit_arity,
            );
            if let Some(mi) = selected {
                selected_declaring = mi.declaring_class.clone();
                if mi
                    .conditional_symbol
                    .as_ref()
                    .is_some_and(|s| !ctx.defines.contains(s))
                {
                    return Ok(b.const_i64(0));
                }
            }
            // Named arguments: rewrite once, here, into parameter order.
            // Every path below (virtual dispatch, direct calls, byref
            // boxing, span coercion, SafeHandle marshaling) then sees the
            // positional list it already expects.
            let named_args = match selected {
                Some(mi) => positional_args(args, mi)?,
                None => None,
            };
            let args: &[Expr] = named_args.as_deref().unwrap_or(args);
            // Nullable-annotated receivers (`string?`) resolve members
            // through the base type (see `nullable_base`); without
            // this the gate below misses and the call falls to
            // extensions/placeholders even though overload selection
            // above already found the instance method.
            //
            // Canonical primitive spellings (`string` vs `String`):
            // selection bridges via `class_for_spelling`, so the
            // gate must too — else `envPath.StartsWith('/')`
            // miscompiles to a span extension call (char stuffed
            // into a span slot, fault on `[47+24]`). Value types
            // stay out (an `Int32` hit would route primitive
            // method calls into virtual dispatch on raw ints), as
            // does `object`/`dynamic` (their unknown-member calls
            // must keep the silent-zero fail-open path, e.g.
            // `object.Exists` in `FileInfo.MoveTo`).
            //
            // Pre-selection intrinsic folds (`Log2`, `NarrowUtf16ToAscii`):
            // fire by receiver+name+arity WITHOUT consulting overload
            // selection. Entry-backed static calls never reach the
            // `_`-arm twins below, and `u`-suffixed literals type as
            // `int` (suffix lost at lexing) so exact-uint overloads
            // miss — stranding these calls on broken
            // intrinsic/hardware bodies (`Log2` returned 0 for every
            // input, breaking all bucket sizing). No static-receiver
            // gate: qualified spellings (`System.Numerics.BitOperations`)
            // type intermediates through bare namespaces (`System` grades
            // to `object`), making the static check fragile, while the
            // base-name match alone is safe (`object` never matches
            // `BitOperations`/`Ascii`, and these names have no instance
            // twins to shadow). All checks are pure (no evaluation) so
            // a miss falls through with zero side effects; on hit,
            // arguments evaluate exactly once here.
            if (base_type_name(&type_name) == "BitOperations"
                && name == "Log2"
                && args.len() == 1)
                || (base_type_name(&type_name) == "Ascii"
                    && name == "NarrowUtf16ToAscii"
                    && args.len() == 3)
            {
                let argv = eval_args(args, b, ctx)?;
                if name == "Log2" {
                    return Ok(fold_log2_loop(b, argv[0]));
                }
                return Ok(fold_narrow_loop(b, argv[0], argv[1], argv[2]));
            }
            match ctx
                .table
                .class(&type_name)
                .or_else(|| nullable_base(ctx.table, &type_name))
                .or_else(|| {
                    let trimmed = type_name.trim();
                    if trimmed == "object" || trimmed == "dynamic" || trimmed == "_" {
                        return None;
                    }
                    let canon = canonical_primitive(trimmed);
                    if canon == trimmed {
                        return None;
                    }
                    let info = ctx.table.class(&canon)?;
                    if info.is_struct || info.is_enum {
                        return None;
                    }
                    Some(info)
                }) {
                // A bare identifier that is a generic type parameter of
                // the current class is an unbound variable — never
                // resolve it through the table (see
                // `Ctx::is_type_var`). It takes the unknown-type
                // path below (extension fallback, then placeholder),
                // mirroring `object`.
                Some(_info) if !ctx.is_type_var(&type_name) => {
                    // Typed overload selection (see `select_visible`):
                    // exact arity + compatible types first, then
                    // defaults-fillable, then legacy arity-only —
                    // within the receiver's kind only, visibility
                    // preferred across same-named siblings.
                    let Some(mi) = selected else {
                        // Delegate-typed field OR property invocation
                        // (`helper._callback(...)`,
                        // `elementInfo.PropertyValueFactory(...)`): C#
                        // invokes the delegate value. Delegates lower as
                        // placeholders (real dispatch is runtime work, like
                        // other delegate paths), so evaluate receiver+args
                        // for side effects and yield zero.
                        if ctx.table.class(&type_name).is_some_and(|info| {
                            info.field(name).is_some() || info.property(name).is_some()
                        }) {
                            let _ = eval(object, b, ctx)?;
                            let _ = eval_args(args, b, ctx)?;
                            return Ok(b.const_i64(0));
                        }
                        // Static spelling with no static overload: if an
                        // INSTANCE overload applies, this is upstream's
                        // static-spelled instance call
                        // (`ResourceManager.GetString(key, null)` —
                        // receiver elision we don't model). Evaluate for
                        // side effects and yield zero (callers like SR
                        // tolerate null via `?? key`).
                        if is_static_receiver
                            && select_visible(
                                ctx.table,
                                &entries,
                                name,
                                &arg_types,
                                Some(false),
                                ctx.current_class,
                                None,
                            )
                            .is_some()
                        {
                            let _ = eval(object, b, ctx)?;
                            let _ = eval_args(args, b, ctx)?;
                            return Ok(b.const_i64(0));
                        }
                        // Extension-method fallback: `recv.Ext(args)` →
                        // `Class.Ext(recv, args)` (e.g. span/collection
                        // helpers in `MemoryExtensions`/`Enumerable`).
                        // `AsSpan()` packs directly (see `span.rs`): the
                        // real body needs reference arithmetic outside
                        // our ref model.
                        if name == "AsSpan" && (args.is_empty() || args.len() == 2) {
                            let recv = eval(object, b, ctx)?;
                            let target = if is_string_type(&type_name) {
                                "ReadOnlySpan<char>".to_owned()
                            } else if let Some(elem) = array_elem_ty(&type_name) {
                                format!("ReadOnlySpan<{elem}>")
                            } else {
                                String::new()
                            };
                            if !target.is_empty()
                                && let Some(path) = span_class_path(ctx, &target)
                                // Inner width comes from the TARGET
                                // spelling (`ReadOnlySpan<char>`), NOT
                                // the resolved path: without an
                                // instantiation entry in the table the
                                // path is the open definition
                                // (`System.ReadOnlySpan`), from which
                                // no inner type can be extracted (this
                                // silently broke every string.AsSpan
                                // once callers reached this block).
                                && let Some((_, inner)) = span_view(&target)
                            {
                                let full = if is_string_type(&type_name) {
                                    pack_span_from_string(b, ctx, recv, &path, &inner)?
                                } else {
                                    pack_span_from_array(b, ctx, recv, &path, &inner)?
                                };
                                if args.len() == 2 {
                                    let start = eval(&args[0], b, ctx)?;
                                    let len = eval(&args[1], b, ctx)?;
                                    return span_slice(
                                        b, ctx, full, &path, &inner, start, Some(len),
                                    );
                                }
                                return Ok(full);
                            }
                        }
                        // `span.ToArray()` / `array.ToArray()` where
                        // overload resolution found nothing (a real
                        // `ROS<T>.ToArray` resolves normally when the
                        // receiver types as a span): copy the buffer
                        // into a new array. Covers `u8` literals, which
                        // mistype as `string` (the lexer drops the
                        // suffix), so string receivers are included — a
                        // genuine `string.ToArray()` is invalid .NET
                        // and has no other resolution here.
                        if name == "ToArray" && args.is_empty() {
                            if let Some(r) = lower_to_array_pack(object, b, ctx, &type_name) {
                                return r;
                            }
                        }
                        if let Some(ext_mi) = extension_method(
                            ctx.table,
                            &type_name,
                            name,
                            &arg_types,
                            explicit_arity,
                        ) {
                            return lower_extension_call(object, args, ext_mi, b, ctx);
                        }
                        // Resolution context for the error below (stderr
                        // only — this path always fails the build, so no
                        // success-path noise): receiver/arg types, entry
                        // candidates, plus table-wide same-named methods
                        // (distinguishes "missing on this type" from
                        // "missing everywhere").
                        {
                            let mut dump = format!(
                                "[rubyc] no member '{name}' on '{type_name}' with args {arg_types:?}; entry_candidates=["
                            );
                            for e in entries.iter() {
                                for m in e.methods.iter().filter(|m| m.name == name) {
                                    dump.push_str(&format!(
                                        "{}#{}(ext={})/{};,",
                                        e.key.path,
                                        e.key.arity,
                                        m.is_extension,
                                        m.param_types.join("+")
                                    ));
                                }
                            }
                            dump.push_str("] table_wide=[");
                            for class in &ctx.table.classes {
                                for m in &class.methods {
                                    if m.name == name {
                                        dump.push_str(&format!(
                                            "{}.{}{}(ext={})/{},",
                                            class.key.path,
                                            m.name,
                                            m.generic_params.len(),
                                            m.is_extension,
                                            m.param_types.join("+")
                                        ));
                                    }
                                }
                            }
                            dump.push(']');
                            dump.push_str(&format!(" argexprs={args:?}"));
                            eprintln!("{dump}");
                        }
return Err(LowerError::UnsupportedConstruct(crate::tf!(
                        "rubyc.native.ir.lower_expr.no_member",
                            type_name.clone(),
                            name.to_string()
                        )));
                    };
                    method_static = Some(mi.is_static);
                    selected_arity = mi.param_types.len();
                }
                _ => {
                    // Instance members win over extensions (C# rule;
                    // see the shared access check below): the
                    // definition-dispatch block runs first, and
                    // extensions apply only when no instance overload
                    // does (slices served solely by
                    // `MemoryExtensions`, `v.Twice()` on primitives,
                    // unknowns). Unbound class type variables land
                    // here too (see the guard on the `Some` arm
                    // above).
                    // Interface-typed receiver (e.g. `IShape s`). Confirm the
                    // method is declared on the interface, then fall through
                    // to the shared virtual-dispatch path below.
                    //
                    // Concrete generic instantiation without its own
                    // entry (`SharedArrayPool<char>`): overload
                    // resolution above already found the method on the
                    // definition entry, and the receiver object is
                    // real — dispatch virtually instead of yielding
                    // zero (which silently nulled `Rent` results and
                    // faulted downstream in `ValueStringBuilder.Grow`).
                    // Pure type variables and unknowns (no definition
                    // entry) keep the placeholder below.
                    if let Some(mi) = selected
                        && !mi.is_static
                        && definition_entry(ctx.table, &type_name).is_some()
                    {
                        ctx.check_method_access(mi)?;
                        let recv = eval(object, b, ctx)?;
                        let mut argv = eval_args(args, b, ctx)?;
                        fill_default_args(&mut argv, mi, b);
                        coerce_span_args(b, ctx, &mut argv, &arg_types, &mi.param_types)?;
                        marshal_safehandle_args(b, ctx, &mut argv, &arg_types, &mi.param_types, !mi.has_body)?;
                        // Slot from the RECEIVER's entry, not the
                        // declaring one: the declaring entry's slot does
                        // NOT agree with the receiver's row for this
                        // signature once a type has several bases — each
                        // base numbers its own methods from 0, so
                        // `MyClass2.Multiply` is slot 1 in MyClass2 and
                        // slot 3 in a `MyClass3 : MyClass1, MyClass2` row.
                        // Dispatching on the declaring slot ran whichever
                        // method held that slot in the first base
                        // (Multiply ran Divide).
                        let slot = ctx
                            .table
                            .class(&type_name)
                            .or_else(|| {
                                ctx.table.classes.iter().find(|c| {
                                    c.name == mi.declaring_class
                                        || c.key.path == mi.declaring_class
                                })
                            })
                            .and_then(|e| {
                                e.methods.iter().find(|m| {
                                    m.name == mi.name && m.param_types == mi.param_types
                                })
                            })
                            .map(|m| m.vtable_slot)
                            // The ARGUMENT spellings identify the interface
                // overload: an interface receiver has no entry of its
                // own, so selection leaves `selected` empty and the
                // slot must be keyed on what the call passes (an empty
                // list matched no overload and fell back to a guessed
                // class slot, sending both `M(int)` and `M(string)` to
                // the first one).
                .or(canonical_slot(ctx.table, None, &type_name, name, &[]))
                            .ok_or_else(|| {
                                LowerError::UnsupportedConstruct(crate::tf!(
                                    "rubyc.native.ir.lower_call.no_virtual_slot",
                                    type_name.clone(),
                                    name.to_string()
                                ))
                            })?;
                        let dst = b.fresh();
                        b.push_raw(Instr::CallVirt {
                            dst,
                            slot: slot as u32,
                            recv,
                            args: argv,
                        });
                        return Ok(dst);
                    }
                    // Scalar search (`IndexOf` / `IndexOfAny` with 2–3
                    // values / `Contains`) over spans/arrays with closed
                    // primitive values inlines here (see
                    // `inline_span_search`): the shared generic bodies
                    // fold address arithmetic with the OPEN variable
                    // and dispatch `Equals` virtually, faulting
                    // (formatting `IndexOfAny('{', '}')` died inside
                    // `SpanHelpers.IndexOfAny`). Unlike `ToArray`
                    // below, this preempts extensions: an applicable
                    // extension body would take the same broken shared
                    // path, while a real instance method (above) still
                    // wins.
                    if (name == "IndexOf" || name == "Contains") && args.len() == 1
                        || name == "IndexOfAny" && (args.len() == 2 || args.len() == 3)
                    {
                        if let Some(v) = inline_span_search(
                            b,
                            ctx,
                            object,
                            args,
                            &type_name,
                            &arg_types,
                            name == "Contains",
                        )? {
                            return Ok(v);
                        }
                    }
                    // Single-value `StartsWith` / `EndsWith` (no
                    // comparer) over spans/arrays with closed values:
                    // same rationale (the shared bodies wrap the
                    // element in nullable gymnastics that miscompile).
                    if (name == "StartsWith" || name == "EndsWith") && args.len() == 1 {
                        if let Some(v) = inline_span_starts_ends(
                            b,
                            ctx,
                            object,
                            args,
                            &type_name,
                            &arg_types,
                            name == "EndsWith",
                        )? {
                            return Ok(v);
                        }
                    }
                    if let Some(ext_mi) = extension_method(
                        ctx.table,
                        &type_name,
                        name,
                        &arg_types,
                        explicit_arity,
                    ) {
                        return lower_extension_call(object, args, ext_mi, b, ctx);
                    }
                    // Shape-based `ToArray()` (see `lower_to_array_pack`):
                    // entry-less receivers (arrays carry no table entry)
                    // never reach the overload-resolution fallback, so
                    // they land here instead of the unknown placeholder
                    // (which nulled the result and faulted downstream
                    // indexing). Runs after instance dispatch (a real
                    // `ToArray` method wins) and after extensions (an
                    // applicable `MemoryExtensions.ToArray` wins, e.g.
                    // range slices over `Span<T>`), as the last resort
                    // before the unknown placeholder.
                    if name == "ToArray" && args.is_empty() {
                        if let Some(r) = lower_to_array_pack(object, b, ctx, &type_name) {
                            return r;
                        }
                    }
                    let Some(_iface) = ctx.table.interface(&type_name) else {
                        // Unknown type (e.g. generic type parameter `T`) —
                        // return 0 as a placeholder so compilation succeeds.
                        // EXCEPTION: primitive/enum `Equals` folds to a
                        // slot compare on the receiver alone (same rule
                        // as the known-entry path below): freestanding
                        // builds have no `Char`/`Int32` entries, and
                        // generic contexts leave the argument open
                        // (`T value`), so requiring both sides would
                        // never fire where it matters.
                        if name == "Equals"
                            && args.len() == 1
                            && is_value_comparable(ctx.table, &type_name)
                        {
                            let recv = eval(object, b, ctx)?;
                            let mut argv = eval_args(args, b, ctx)?;
                            let rhs = argv.pop().unwrap_or_else(|| b.const_i64(0));
                            return Ok(b.emit(Instr::Cmp {
                                dst: VReg(0),
                                op: crate::native::ir::CmpOp::Eq,
                                a: recv,
                                b: rhs,
                            }));
                        }
                        // An unresolved receiver (open generic parameter, uninferred
                        // temporary, discard) has no entry to resolve a
                        // member against: evaluate for side effects and
                        // yield the default. A call on a receiver that
                        // DID resolve and lacks the method reports
                        // `no_member` below.
                        let _ = eval(object, b, ctx)?;
                        let _ = eval_args(args, b, ctx)?;
                        return Ok(b.const_i64(0));
                    };
                    if !interface_has_method(ctx.table, &type_name, name) {
return Err(LowerError::UnsupportedConstruct(crate::tf!(
                        "rubyc.native.ir.lower_expr.no_member",
                            type_name.clone(),
                            name.to_string()
                        )));
                    }
                }
            }
            // Argument count mismatch is tolerated — generic methods and
            // overloads may not match exactly. Proceed with the actual args.
            // Overload-aware access: enforce the SELECTED overload's own
            // access + declarer (shared computation, so all agree).
            if let Some(mi) = selected {
                // Virtual-slot access first (an accessible base-slot
                // declaration redeems an inaccessible override), then —
                // only for instance receivers — the extension fallback.
                // (C# prefers applicable instance members, accessible
                // or via their base slot, over extensions.)
                let access_ok = ctx.check_method_access(mi).is_ok()
                    || (!is_static_receiver
                        && ctx.check_method_access_virtual(mi, &type_name).is_ok());
                if !access_ok {
                    // Inaccessible instance method: C# falls back to
                    // extension methods before failing
                    // (`reader.GetUnescapedSpan()` — private instance +
                    // public `JsonHelpers` extension). Only a VISIBLE
                    // extension rescues the call; otherwise the original
                    // access error stands. Extensions never apply to a
                    // static-spelled receiver.
                    if !is_static_receiver {
                        if let Some(ext_mi) = extension_method(
                            ctx.table,
                            &type_name,
                            name,
                            &arg_types,
                            explicit_arity,
                        ) {
                            if ctx.check_method_access(ext_mi).is_ok() {
                                return lower_extension_call(
                                    object, args, ext_mi, b, ctx,
                                );
                            }
                        }
                    }
                    ctx.check_method_access(mi)?;
                }
            }

            // A bare class-name identifier as the receiver is a static-style
            // call (receiver kind was fixed before selection above).
            // Whether that is valid depends on the method's kind:
            //  - static method  -> direct call, no receiver (C# `C.M(...)`)
            //  - instance method-> error: the class name is not an instance
            let mut argv = eval_args(args, b, ctx)?;
            // Fill omitted optional params from the selected overload's
            // folded defaults so the callee sees its full arity.
            if let Some(mi) = selected {
                fill_default_args(&mut argv, mi, b);
                // Span-typed parameters pack array/string arguments
                // into views (see `span.rs`); anything else passes
                // through untouched.
                coerce_span_args(b, ctx, &mut argv, &arg_types, &mi.param_types)?;
                marshal_safehandle_args(b, ctx, &mut argv, &arg_types, &mi.param_types, !mi.has_body)?;
            }
            // `BitOperations.Log2(v)`: fold to an exact halving loop.
            // Entry-backed twin: fires via the selected declarer (real
            // overload hit), sharing `fold_log2_loop` with the
            // pre-selection gate and the `_`-arm twin (single source of
            // truth). Without this, entry-backed calls run the
            // intrinsic/hardware body (0 for every input), breaking all
            // bucket sizing (pooled paths faulted on exact-size arrays).
            // Exact for all non-negative inputs; sizes/counts always are.
            if base_type_name(&selected_declaring) == "BitOperations"
                && name == "Log2"
                && argv.len() == 1
            {
                return Ok(fold_log2_loop(b, argv[0]));
            }
            let dst = b.fresh();
            if is_static_receiver {
                if method_static == Some(true) {
                    // C# `C.M(...)` — a static method, possibly inherited
                    // from a base class. Resolve the declaring class's
                    // function (keyed by declaring, not receiver
                    // spelling); statics are not vtable-dispatched.
                    // Overload-aware: same-arity overloads resolve to
                    // the selected one's slot, not last-wins.
                    let selected_params = selected
                        .map(|mi| mi.param_types.clone())
                        .unwrap_or_default();
                    // Byref signature of the elected overload (see
                    // `overload_position`): value-vs-`ref` twins share
                    // erased param types.
                    let selected_byref: Vec<bool> = selected
                        .map(|mi| mi.param_is_byref.clone())
                        .unwrap_or_default();
                    let Some(fidx) = static_function_index_for(ctx, &selected_declaring, name, selected_arity, &selected_params, &selected_byref) else {
                        // Bodyless static (`extern`/`InternalCall`/
                        // abstract/partial-no-impl): the method exists but
                        // generates no function. Args are already evaluated
                        // above (side effects preserved); yield zero —
                        // runtime-implemented natives bind for real later.
                        if let Some(mi) = selected
                            && !mi.has_body
                        {
                            return Ok(b.const_i64(0));
                        }
                        // Cross-image call (dynamic-linking consume): the
                        // callee lives in another image — import its body
                        // from the installed `.so` instead of failing.
                        // Args are evaluated and defaults filled above, so
                        // the callee sees its full arity, exactly as in a
                        // static build.
                        if selected.is_some()
                            && let Some(entry) = resolve_call_target(
                                ctx.table,
                                &selected_declaring,
                                name,
                                selected_arity,
                            )
                            && let Some((lib, sym)) =
                                external_target(ctx, entry, name, selected_arity)
                        {
                            let idx = dep_import_index(ctx, &lib, &sym);
                            // P/Invoke SafeHandle marshaling for native
                            // imports (see `marshal_safehandle_args`):
                            // native code has no objects.
                            let empty: Vec<String> = Vec::new();
                            let _ = marshal_safehandle_args(
                                b, ctx, &mut argv, &arg_types, &empty, true,
                            );
                            b.push_raw(Instr::CallImport {
                                dst,
                                import: idx as u32,
                                args: argv,
                            });
                            return Ok(dst);
                        }
return Err(LowerError::UnsupportedConstruct(crate::tf!(
                        "rubyc.native.ir.lower_expr.no_member",
                            type_name.clone(),
                            name.to_string()
                        )));
                    };
                    // `BitOperations.Log2(v)` (1 arg, any int width):
                    // floor log2 (`0`→`0`), via halving (division, never
                    // shifts — our `Shr` is arithmetic and would stick
                    // on large uints). The intrinsic/hardware paths
                    // (`BitScanReverse` via a misreporting
                    // `X86Base.IsSupported`) and possibly the software
                    // De Bruijn fallback return 0 for every input here,
                    // breaking all bucket sizing (pooled paths faulted
                    // on exact-size arrays). Exact for all non-negative
                    // inputs; caller-side values (sizes/counts) always
                    // are.
                    if base_type_name(&selected_declaring) == "BitOperations"
                        && name == "Log2"
                        && argv.len() == 1
                    {
                        return Ok(fold_log2_loop(b, argv[0]));
                    }
                    // `Ascii.NarrowUtf16ToAscii(char*, byte*, count)`:                    // copies ASCII chars (`U+0000..U+007F`) stopping at
                    // the first non-ASCII char, returning the converted
                    // count. Fold to exact scalar semantics: the
                    // vectorized body (SSE/AdvSimd intrinsics, dword
                    // loop, gotos) under-consumes here (pure-ASCII
                    // input spuriously falls back), and every spurious
                    // fallback risks the fallback machinery. Char
                    // reads mask to 16 bits; byte stores write full
                    // slots like normal element stores (readers mask).
                    if base_type_name(&selected_declaring) == "Ascii"
                        && name == "NarrowUtf16ToAscii"
                        && argv.len() == 3
                    {
                        return Ok(fold_narrow_loop(b, argv[0], argv[1], argv[2]));
                    }
                    // `MemoryMarshal.GetArrayDataReference` (both the
                    // generic `T[]` forwarder and the `Array` overload):
                    // JIT intrinsics whose bodies never run in .NET
                    // (the generic one even forwards to itself by
                    // overload rules — lowered literally, it recurses
                    // forever). In our array layout (length at +0,
                    // refcount at +8, elements at +16) the data
                    // reference is `arr + 16`.
                    if base_type_name(&selected_declaring) == "MemoryMarshal"
                        && name == "GetArrayDataReference"
                        && argv.len() == 1
                    {
                        let sixteen = b.const_i64(16);
                        return Ok(b.emit(Instr::Bin {
                            dst: VReg(0),
                            op: IrBinOp::Add,
                            a: argv[0],
                            b: sixteen,
                        }));
                    }
                    // `Unsafe.AsRef(void*)`: the argument already IS
                    // the address — return it as the ref, never read
                    // through it. Lowering the managed body
                    // (`ref *(T*)source`) dereferences, so
                    // `AsRef((void*)1)` (the empty-span fake pointer
                    // from `GetNonNullPinnableReference`) faulted at
                    // si_addr=0x1 inside console Flush. Only the
                    // pointer overload folds (the `in T` twin keeps
                    // normal lowering).
                    if base_type_name(&selected_declaring) == "Unsafe"
                        && name == "AsRef"
                        && argv.len() == 1
                        && arg_types
                            .first()
                            .is_some_and(|t| is_pointer_ty(t))
                    {
                        return Ok(argv[0]);
                    }
                    // `MemoryMarshal.GetReference(span)`: the stored
                    // `_reference` pointer, loaded from its slot (the
                    // source is just `ref span._reference`). Lowering
                    // the body goes through the broken `return ref`
                    // path (address-of-field instead of the slot),
                    // so pins derived from it pointed at span objects
                    // and the encoder transcoded pointer bytes as
                    // chars (spurious fallback on ASCII input). No
                    // empty check (unlike `GetNonNullPinnableReference`;
                    // the source has none). Offsets from the table.
                    if base_type_name(&selected_declaring) == "MemoryMarshal"
                        && name == "GetReference"
                        && argv.len() == 1
                    {
                        let recv_ty = static_type_of(&args[0], b, ctx)
                            .unwrap_or_else(|_| "object".to_owned());
                        let (ref_off, _) =
                            span_field_offsets(ctx, &recv_ty).unwrap_or((16, 24));
                        let r = b.fresh();
                        b.push_raw(Instr::LoadField {
                            dst: r,
                            base: argv[0],
                            off: ref_off,
                        });
                        return Ok(r);
                    }
                    // `len == 0 ? (void*)1 : span._reference`, emitted
                    // `MemoryMarshal.GetNonNullPinnableReference(span)`:
                    // `len == 0 ? (void*)1 : span._reference`, emitted
                    // directly (mirrors the source). Lowering the
                    // managed body chains `return ref` through byref
                    // fields and `AsRef((void*)1)`, both of which
                    // miscompile to address-of-field / dereference-1
                    // (stdout got pointer bytes; empty spans faulted
                    // at si_addr=0x1). Offsets come from the table,
                    // matching `span_object`.
                    if base_type_name(&selected_declaring) == "MemoryMarshal"
                        && name == "GetNonNullPinnableReference"
                        && argv.len() == 1
                    {
                        let recv_ty = static_type_of(&args[0], b, ctx)
                            .unwrap_or_else(|_| "object".to_owned());
                        let (ref_off, len_off) =
                            span_field_offsets(ctx, &recv_ty).unwrap_or((16, 24));
                        let len = b.fresh();
                        b.push_raw(Instr::LoadField {
                            dst: len,
                            base: argv[0],
                            off: len_off,
                        });
                        let r = b.fresh();
                        b.push_raw(Instr::LoadField {
                            dst: r,
                            base: argv[0],
                            off: ref_off,
                        });
                        let out = b.fresh();
                        let zero = b.const_i64(0);
                        let one = b.const_i64(1);
                        let is_empty = b.emit(Instr::Cmp {
                            dst: VReg(0),
                            op: CmpOp::Eq,
                            a: len,
                            b: zero,
                        });
                        let empty_b = b.make_block();
                        let nonempty_b = b.make_block();
                        let join_b = b.make_block();
                        b.terminate(Terminator::Branch {
                            cond: Some(is_empty),
                            if_true: empty_b,
                            if_false: nonempty_b,
                        });
                        b.switch_to(empty_b);
                        b.mov(out, one);
                        b.goto(join_b);
                        b.switch_to(nonempty_b);
                        b.mov(out, r);
                        b.goto(join_b);
                        b.switch_to(join_b);
                        return Ok(out);
                    }
                    // `Unsafe.AddByteOffset(ref value, byteOffset)`:
                    // raw address arithmetic. `ref` erases at parse,
                    // so the first argument already holds the address
                    // (from `.Data`/`GetArrayDataReference` folds) —
                    // plain add is exact. A by-reference ident needs
                    // its home (value-position eval dereferences).
                    if base_type_name(&selected_declaring) == "Unsafe"
                        && name == "AddByteOffset"
                        && argv.len() == 2
                    {
                        let base0 = match unmark_ref_arg(&args[0]) {
                            Expr::Ident { name, .. } => {
                                byref_home(b, name).unwrap_or(argv[0])
                            }
                            _ => argv[0],
                        };
                        return Ok(b.emit(Instr::Bin {
                            dst: VReg(0),
                            op: IrBinOp::Add,
                            a: base0,
                            b: argv[1],
                        }));
                    }
                    // `Unsafe.Add<T>(ref value, index)`: address of
                    // element `index` — `value + index * width(T)`.
                    // Width comes from the (ref-erased) argument type
                    // (`ref _firstChar` types as `char` → 2); explicit
                    // type arguments agree when present. Like
                    // `AddByteOffset`, this assumes the ref holds an
                    // address (true for `.Data`/`Add` chains; a ref to
                    // a plain int local would miscompile — no such use
                    // in the framework paths).
                    if base_type_name(&selected_declaring) == "Unsafe"
                        && name == "Add"
                        && argv.len() == 2
                    {
                        // A by-reference ident contributes its home
                        // (address); value-position eval would hand the
                        // dereferenced pointee instead.
                        let base0 = match unmark_ref_arg(&args[0]) {
                            Expr::Ident { name, .. } => {
                                byref_home(b, name).unwrap_or(argv[0])
                            }
                            _ => argv[0],
                        };
                        // `span_elem_width` maps primitives to widths
                        // (references/unknowns to a full slot); explicit
                        // type arguments win, else the erased argument.
                        let w = explicit_targs
                            .first()
                            .map(|t| span_elem_width(t))
                            .unwrap_or_else(|| {
                                span_elem_width(
                                    &static_type_of(&args[0], b, ctx)
                                        .unwrap_or_else(|_| "object".to_owned()),
                                )
                            });
                        let wv = b.const_i64(w as i64);
                        let scaled = b.emit(Instr::Bin {
                            dst: VReg(0),
                            op: IrBinOp::Mul,
                            a: argv[1],
                            b: wv,
                        });
                        return Ok(b.emit(Instr::Bin {
                            dst: VReg(0),
                            op: IrBinOp::Add,
                            a: base0,
                            b: scaled,
                        }));
                    }
                    // `Unsafe.ReadUnaligned<T>(void* source)` and the
                    // `(ref byte source)` twin: the CORECLR source
                    // branch is an intentional throw (the JIT supplies
                    // the body via `ldobj`), so calls would always
                    // raise `PlatformNotSupportedException`. Lower as a
                    // plain sized load — the portable `#else`
                    // (`*(T*)source`) semantics. Width from the
                    // explicit type argument (`uint` -> 4); narrow
                    // loads zero-extend like pointer reads.
                    if base_type_name(&selected_declaring) == "Unsafe"
                        && name == "ReadUnaligned"
                        && argv.len() == 1
                    {
                        let targ = explicit_targs.first().cloned().unwrap_or("object".to_owned());
                        // A by-reference ident contributes its home
                        // (address); the evaluated value would be the
                        // dereferenced (possibly masked) pointee, and
                        // loading from it faults (`ref level2Ref`
                        // became byte 142 as a base).
                        let base = match unmark_ref_arg(&args[0]) {
                            Expr::Ident { name: aname, .. } => {
                                byref_home(b, aname).unwrap_or(argv[0])
                            }
                            _ => argv[0],
                        };
                        let raw = b.fresh();
                        b.push_raw(Instr::LoadField {
                            dst: raw,
                            base,
                            off: 0,
                        });
                        if let Some(mask) = narrow_load_mask(&targ) {
                            let m = b.const_i64(mask);
                            return Ok(b.emit(Instr::Bin {
                                dst: VReg(0),
                                op: IrBinOp::BitAnd,
                                a: raw,
                                b: m,
                            }));
                        }
                        return Ok(raw);
                    }
                    // `Buffer.Memmove<T>(ref dst, ref src, len)` (len in
                    // ELEMENTS): fold to a byte copy of `len*width`.
                    // Named arguments are rewritten into parameter
                    // order before this point (`positional_args`), but
                    // the length is still identified by TYPE (the unique
                    // `uint`/`nuint`/`int`-typed argument; dst/src type
                    // as their (ref-erased) element), which keeps the
                    // fold correct for overload spellings the call site
                    // cannot match. The rest keep relative order as
                    // dst, src. All-same-type calls fall back to
                    // positional. Same overlap limitation as
                    // `SpanHelpers.Memmove` above.
                    if base_type_name(&selected_declaring) == "Buffer"
                        && name == "Memmove"
                        && argv.len() == 3
                    {
                        let is_count = |t: &str| {
                            matches!(
                                base_type_name(t).as_str(),
                                "UInt32" | "UInt64" | "Int32" | "Int64" | "UIntPtr" | "IntPtr"
                            )
                        };
                        let len_idx = {
                            let mut found = None;
                            for (i, t) in arg_types.iter().enumerate() {
                                if is_count(t) {
                                    if found.is_some() {
                                        found = None;
                                        break;
                                    }
                                    found = Some(i);
                                }
                            }
                            found.unwrap_or(2)
                        };
                        let mut rest = vec![];
                        let mut rest_idx = vec![];
                        for (i, v) in argv.iter().enumerate() {
                            if i != len_idx {
                                rest.push(*v);
                                rest_idx.push(i);
                            }
                        }
                        // Byref address arguments (`ref arr[i]`) must
                        // contribute interior addresses, not the
                        // evaluated element values (the parser drops
                        // `ref`; copying from value 0/65 faults).
                        // Already-addresses pass through untouched.
                        // (`args` may be shorter than `argv` when
                        // omitted optionals were filled.)
                        let dst = if rest_idx[0] < args.len() {
                            memmove_addr(b, ctx, &args[rest_idx[0]], rest[0])?
                        } else {
                            rest[0]
                        };
                        let src = if rest_idx[1] < args.len() {
                            memmove_addr(b, ctx, &args[rest_idx[1]], rest[1])?
                        } else {
                            rest[1]
                        };
                        let len = argv[len_idx];
                        // Element width: explicit type argument wins,
                        // else a narrow primitive among the address
                        // arguments (their ref-erased element type),
                        // else the uniform slot width.
                        let mut w = explicit_targs
                            .first()
                            .map(|t| span_elem_width(t))
                            .unwrap_or(8);
                        if explicit_targs.is_empty() {
                            for (i, t) in arg_types.iter().enumerate() {
                                if i == len_idx {
                                    continue;
                                }
                                match span_elem_width(t) {
                                    1 | 2 | 4 => {
                                        w = span_elem_width(t);
                                        break;
                                    }
                                    _ => {}
                                }
                            }
                        }
                        let total = if w == 1 {
                            len
                        } else {
                            let wv = b.const_i64(w as i64);
                            b.emit(Instr::Bin {
                                dst: VReg(0),
                                op: IrBinOp::Mul,
                                a: len,
                                b: wv,
                            })
                        };
                        b.push_raw(Instr::MemCpy {
                            dst,
                            src,
                            len: total,
                        });
                        return Ok(dst);
                    }
                    // `SpanHelpers.Memmove(ref dest, ref src, byteCount)`:
                    // the memmove the `Array.Copy` fast path ends in.
                    // Both refs arrive as addresses (see
                    // `AddByteOffset`); elements are unrefcounted, so a
                    // raw byte copy is consistent with the plain
                    // element stores. OVERLAP LIMITATION: `MemCpy` is a
                    // forward copy — overlapping same-array ranges with
                    // `dst > src` need a backward loop (future work;
                    // collection resizes use disjoint arrays).
                    if base_type_name(&selected_declaring) == "SpanHelpers"
                        && name == "Memmove"
                        && argv.len() == 3
                    {
                        // `args` may be shorter than `argv` (omitted
                        // optionals filled into `argv` only).
                        let dst = if args.len() > 0 {
                            memmove_addr(b, ctx, &args[0], argv[0])?
                        } else {
                            argv[0]
                        };
                        let src = if args.len() > 1 {
                            memmove_addr(b, ctx, &args[1], argv[1])?
                        } else {
                            argv[1]
                        };
                        b.push_raw(Instr::MemCpy {
                            dst,
                            src,
                            len: argv[2],
                        });
                        return Ok(argv[0]);
                    }
                    // `Unsafe.IsNullRef(ref x)`: a ref to any variable
                    // (static/local/field slot) is never a null
                    // *pointer* in C# semantics, even when the slot
                    // holds null — but erased refs ARE the value, so
                    // the natural lowering tests the contents and
                    // spuriously throws on every null slot (e.g. all
                    // lazy-init `CompareExchange(ref field, …)`).
                    // Yield false constantly: true null managed
                    // pointers (`NullRef<T>()`, itself unimplemented
                    // and throwing) have no representation here.
                    if base_type_name(&selected_declaring) == "Unsafe"
                        && name == "IsNullRef"
                        && argv.len() == 1
                    {
                        return Ok(b.const_i64(0));
                    }
                    // `Interlocked.CompareExchange(ref loc, val, cmp)` /
                    // `Interlocked.Exchange(ref loc, val)`: single-threaded
                    // compare-and-store (atomics need no barrier in one
                    // thread; static exes are single-threaded at init and
                    // smoke paths never race). The `InternalCall`
                    // bodies are unimplemented (zero placeholders), so
                    // lower directly: orig = loc; if (orig == cmp)
                    // loc = val; return orig. Reference handles compare
                    // by value (equality is identity here); floats use
                    // bitwise equality (documented `-0.0`/NaN gap).
                    // Like `Volatile.Write`, the store re-evaluates its
                    // operand exprs (side effects may repeat — same
                    // documented caveat).
                    if base_type_name(&selected_declaring) == "Interlocked"
                        && matches!(name, "CompareExchange" | "Exchange")
                        && (args.len() == 2 || args.len() == 3)
                    {
                        return lower_compare_exchange(name, args, b, ctx);
                    }
                    // `NativeMemory.Alloc/AllocZeroed/Realloc/AlignedAlloc/
                    // AlignedRealloc` (and `Free`/`AlignedFree` below):
                    // route to the managed heap instead of libc P/Invoke
                    // (whose import slots are unreliable at this scale,
                    // and whose zero-stubs would null-deref on first
                    // write). Our bump allocator is 16-aligned
                    // (sufficient for smoke; >16 alignment is best
                    // effort, documented). `Realloc` returns a fresh
                    // block WITHOUT copying (content loss documented;
                    // revisit if a live path needs it). `AllocZeroed`
                    // is NOT zeroed (fresh bump memory is zero from
                    // BSS; freelist reuse may carry stale data —
                    // documented gap).
                    if base_type_name(&selected_declaring) == "NativeMemory"
                        && matches!(
                            name,
                            "Alloc" | "AllocZeroed" | "Realloc" | "AlignedAlloc" | "AlignedRealloc"
                        )
                        && (args.len() == 1 || args.len() == 2 || args.len() == 3)
                    {
                        let argv = eval_args(args, b, ctx)?;
                        // Size operand by shape: single-arg (bytes);
                        // two-arg (count, size) product except
                        // `Realloc(ptr, bytes)` (second) and
                        // `AlignedAlloc(bytes, align)` (first).
                        let size_v = if args.len() == 1 {
                            argv[0]
                        } else if name == "Realloc" || name == "AlignedRealloc" {
                            argv[1]
                        } else if name == "AlignedAlloc" {
                            argv[0]
                        } else {
                            let dst = b.fresh();
                            b.push_raw(Instr::Bin {
                                dst,
                                op: IrBinOp::Mul,
                                a: argv[0],
                                b: argv[1],
                            });
                            dst
                        };
                        let obj = b.fresh();
                        // AllocDyn takes the size in a vreg.
                        b.push_raw(Instr::AllocDyn { dst: obj, bytes: size_v });
                        return Ok(obj);
                    }
                    if base_type_name(&selected_declaring) == "NativeMemory"
                        && matches!(name, "Free" | "AlignedFree")
                    {
                        let _ = eval_args(args, b, ctx)?;
                        return Ok(b.const_i64(0));
                    }
                    // `Unsafe.As<TFrom, TTo>(ref x)` / `Unsafe.AsRef(...)`:
                    // JIT intrinsics whose managed bodies throw
                    // `PlatformNotSupportedException` (the real `ldarg.0
                    // / ret` identity lives in the JIT, not the IL we
                    // compile). References are erased to values here,
                    // so both are the identity: return the argument
                    // unchanged. This also neutralizes every
                    // `Unsafe.As`-based reinterpretation wrapper
                    // (`Volatile*` structs, `RuntimeHelpers`, `GC`
                    // internals) that would otherwise throw at runtime.
                    // `BitCast` joins the family: our value model is
                    // uniformly 64-bit slots, so bit reinterpretation
                    // is the identity (the size check would need
                    // layout; mismatched sizes keep legacy behavior
                    // elsewhere).
                    if base_type_name(&selected_declaring) == "Unsafe"
                        && matches!(name, "As" | "AsRef" | "AsPointer" | "BitCast")
                        && argv.len() == 1
                    {
                        // Reinterpreted reference: the result IS the
                        // address (not the dereferenced value). A
                        // by-reference ident contributes its home;
                        // using the evaluated (possibly masked) value
                        // corrupted `ref short rsStart` in `ToUpper`
                        // into a byte, faulting the table read.
                        // Call-site `ref` markers unwrap to the inner
                        // ident (same home decision).
                        if let Expr::Ident { name: aname, .. } = unmark_ref_arg(&args[0]) {
                            if let Some(h) = byref_home(b, aname) {
                                return Ok(h);
                            }
                        }
                        return Ok(argv[0]);
                    }
                    // `Volatile.Read(ref loc)` / `Volatile.Write(ref loc,
                    // v)`: same story — expression-bodied wrappers over
                    // `Unsafe.As` + `.Value` on `Volatile*` structs,
                    // which the identity above would leave as a field
                    // load on a non-wrapper value. Intercept directly:
                    // erased refs are values, so a volatile read is the
                    // value and a volatile write is an assignment.
                    if base_type_name(&selected_declaring) == "Volatile"
                        && name == "Read"
                        && args.len() == 1
                    {
                        return eval(&args[0], b, ctx);
                    }
                    if base_type_name(&selected_declaring) == "Volatile"
                        && name == "Write"
                        && args.len() == 2
                    {
                        let assign = Expr::Assign {
                            target: Box::new(args[0].clone()),
                            value: Box::new(args[1].clone()),
                            span: crate::models::Span::new(0, 0),
                        };
                        return eval(&assign, b, ctx);
                    }
                    // `[Intrinsic]` self-recursive placeholders ("Must
                    // expand intrinsic"): `Exchange` / `ExchangeAdd` /
                    // `CompareExchange` bodies call themselves (the JIT
                    // expands the intrinsic instead of executing the
                    // body), so running them recurses forever (hung
                    // `TaskScheduler` init via `Interlocked.Add`).
                    // Shared helper below (also used for bare
                    // same-class calls, which land in `calls.rs`).
                    if base_type_name(&selected_declaring) == "Interlocked"
                        && let Some(r) = interlocked_atomic(name, args, b, ctx)
                    {
                        return r;
                    }
                    // `MemoryMarshal.Cast<From, To>(span)` reinterprets
                    // the buffer (see `span.rs`): real bodies need
                    // reference arithmetic outside our ref model.
                    if base_type_name(&selected_declaring) == "MemoryMarshal"
                        && name == "Cast"
                        && argv.len() == 1
                        && !arg_types.is_empty()
                        && explicit_targs.len() == 2
                    {
                        return span_cast(b, ctx, argv[0], &arg_types[0], &explicit_targs[1]);
                    }
                    // Box by-reference arguments (see `box_byref_args`;
                    // pattern lookup keeps this branch independent of
                    // the overload-selection binding above).
                    let wbs = byref_pattern(ctx.table, &selected_declaring, name, argv.len())
                        .map(|pat| box_byref_args(b, ctx, args, &mut argv, &pat).unwrap_or_default())
                        .unwrap_or_default();
                    // P/Invoke marshaling is a language feature, but the
                    // marshaling decision must come from the signature
                    // (is the callee an extern, do parameter types
                    // differ), never from the declaring class being
                    // spelled `Interop`/`Sys`.
                    {
                        if let Some(entry) = ctx.table.class(&selected_declaring) {
                            if let Some(mi) = entry.methods.iter().find(|m| {
                                m.name == name && m.param_types.len() == argv.len()
                            }) {
                                let _ = marshal_safehandle_args(
                                    b,
                                    ctx,
                                    &mut argv,
                                    &arg_types,
                                    &mi.param_types,
                                    !mi.has_body,
                                );
                            }
                        }
                    }
                    b.push_raw(Instr::CallStatic { dst, fidx, args: argv });
                    emit_writebacks(b, &wbs);
                    return Ok(dst);
                }
                // Static interface members (`IRuntimeMethodInfo.GetValue`):
                // interfaces generate no functions — the args were
                // already evaluated above, so just yield zero (real
                // static dispatch later).
                if method_static.is_none()
                    && ctx.table.interfaces.iter().any(|i| {
                        i.name == type_name
                            && i.methods.iter().any(|m| m.name == name && m.is_static)
                    })
                {
                    return Ok(b.const_i64(0));
                }
                // Instance method called through the type name. The
                // class name is not an instance, so this is a compile error
                // rather than a null-receiver call.
                return Err(LowerError::UnsupportedConstruct(crate::tf!(
                    "rubyc.native.ir.lower_expr.static_receiver",
                    type_name.clone(),
                    name.to_string()
                )));
            }
            if method_static == Some(true) {
                // Static method called through an instance reference —
                // C#-style rejection: call it through the type instead.
                return Err(LowerError::UnsupportedConstruct(crate::tf!(
                    "rubyc.native.ir.lower_expr.instance_static",
                    type_name.clone(),
                    name.to_string()
                )));
            }
            let recv = eval(object, b, ctx)?;
            if name == "Equals"
                && argv.len() == 1
                && is_value_comparable(ctx.table, &type_name)
            {
                return Ok(b.emit(Instr::Cmp {
                    dst: VReg(0),
                    op: crate::native::ir::CmpOp::Eq,
                    a: recv,
                    b: argv[0],
                }));
            }
            // Slot from the SELECTED overload's declaring entry when
            // available (same-name overloads occupy distinct slots
            // after 3vtable reconciliation, so the name-only
            // `canonical_slot` below can land on a sibling overload's
            // slot — e.g. `StartsWith(string)` vs `StartsWith(char)`).
            // Falls back to the legacy lookup when selection is
            // absent or the declaring entry is gone.
            // The selected MethodInfo already carries its reconciled
            // slot: prefer it directly. Re-resolving by
            // (declaring-class, name, exact param spellings) misses
            // on spelling drift the tolerant selection bridged
            // (qualified `System.Text.EncoderNLS` vs `EncoderNLS?`,
            // `string` vs `String`), and the name-only fallback
            // then elects a sibling overload's slot — e.g. an
            // internal pointer `GetBytes` routed to an array
            // overload that throws reading Length off the pointer.
            // Receiver's entry first, for the same reason as the
            // call above: with multiple bases the declaring class's
            // slot numbering is unrelated to the receiver's row.
            let slot = ctx
                .table
                .class(&type_name)
                .and_then(|e| {
                    e.methods.iter().find(|m| {
                        selected.is_some_and(|mi| {
                            m.name == mi.name && m.param_types == mi.param_types
                        })
                    })
                })
                .map(|m| m.vtable_slot)
                .or_else(|| selected.map(|mi| mi.vtable_slot))
                // The ARGUMENT spellings identify the interface
                // overload: an interface receiver has no entry of its
                // own, so selection leaves `selected` empty and the
                // slot must be keyed on what the call passes (an empty
                // list matched no overload and fell back to a guessed
                // class slot, sending both `M(int)` and `M(string)` to
                // the first one).
                .or(canonical_slot(ctx.table, None, &type_name, name, &arg_types))
                .ok_or_else(|| {
                    LowerError::UnsupportedConstruct(crate::tf!(
                        "rubyc.native.ir.lower_call.no_virtual_slot",
                        type_name.clone(),
                        name.to_string()
                    ))
                })?;
            // Sealed-layout receivers (strings, arrays, and the
            // `Array` base itself — e.g. `Copy(Array, ...)`
            // parameters) carry no vtable word (length at +0) —
            // virtual dispatch reads length-as-vtable and faults
            // (`str.StartsWith("0x")` read `[15+60]`,
            // `sourceArray?.GetLowerBound(0)` read `[3+0x198]`).
            // When overload selection found the method, call its
            // body directly (exact for sealed `String`; array
            // methods are non-virtual in practice). Interface
            // slots, generic methods, and unresolvable bodies keep
            // CallVirt.
            // `base.` calls are always direct (C# non-virtual
            // base call): routing them through the vtable
            // dispatches to the most-derived override — for an
            // overriding method calling `base.M()` that is ITSELF
            // (infinite recursion; `UnixConsoleStream.Flush`
            // calling `base.Flush()` never returned).
            let is_base_call = matches!(
                object.as_ref(),
                Expr::Ident { name, .. } if name.as_str() == "base"
            );
            if let Some(mi) = selected
                && !mi.is_static
                && mi.generic_params.is_empty()
                && (is_string_type(&type_name)
                    || array_elem_ty(&type_name).is_some()
                    || base_type_name(&type_name) == "Array"
                    || is_base_call)
            {
                let n = overload_position(
                    ctx,
                    &mi.declaring_class,
                    &mi.name,
                    mi.param_types.len(),
                    &mi.param_types,
                    &mi.param_is_byref,
                );
                if let Some(fidx) = function_index_nth(
                    ctx,
                    &mi.declaring_class,
                    &mi.name,
                    mi.param_types.len(),
                    n,
                ) {
                    let dst = b.fresh();
                    let wbs = box_byref_args(b, ctx, args, &mut argv, &mi.param_is_byref)?;
                    let mut call_args = vec![recv];
                    call_args.extend(argv);
                    b.push_raw(Instr::CallFn {
                        dst,
                        fidx,
                        args: call_args,
                    });
                    emit_writebacks(b, &wbs);
                    return Ok(dst);
                }
            }
            let wbs = match selected {
                Some(mi) => box_byref_args(b, ctx, args, &mut argv, &mi.param_is_byref)?,
                None => Vec::new(),
            };
            b.push_raw(Instr::CallVirt {
                dst,
                slot: slot as u32,
                recv,
                args: argv,
            });
            emit_writebacks(b, &wbs);
            Ok(dst)
        }
        Expr::Call { callee, args, .. } => call(callee, args, b, ctx),
        Expr::Await { inner, .. } => {
            // `await expr` — suspension point inside an async method.
            // The inner expression is evaluated, then a Suspend instruction
            // is emitted. The state-machine transform (applied after
            // lowering) restructures the IR so that execution can be
            // resumed from the next state-machine block.
            let value = eval(inner, b, ctx)?;
            // The Suspend instruction takes the state vreg; the backend
            // will use this to save the current state-machine state and
            // local values into the task frame before returning to the
            // scheduler.
            let state = b.fresh();
            // Placeholder state number until the state-machine transform
            // assigns real ones. The operand is an IR value, so it must
            // have a definition even while the backend treats `Suspend`
            // as a nop.
            b.push_raw(Instr::Const { dst: state, imm: 0 });
            b.push_raw(Instr::Suspend { state });
            Ok(value)
        }
    }
}


pub(super) use eval_block_inline as eval_block_inline_pub;


/// `int`/`Int32` spelling (the two forms `static_type_of` produces).
fn is_int32_spelling(t: &str) -> bool {
    matches!(t.trim(), "int" | "Int32" | "System.Int32")
}

/// Truncate to 32 bits and sign-extend back to the slot width, which is
/// how every 32-bit result is represented (`Shr` must be arithmetic).
fn narrow_int32(b: &mut Body<'_>, v: VReg) -> VReg {
    let s = b.const_i64(32);
    let l = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Shl,
        a: v,
        b: s,
    });
    b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Shr,
        a: l,
        b: s,
    })
}


/// `uint`/`UInt32` spelling.
fn is_uint32_spelling(t: &str) -> bool {
    matches!(t.trim(), "uint" | "UInt32" | "System.UInt32")
}

/// Keep the low 32 bits, zero-extended (C# `uint` result).
fn mask_uint32(b: &mut Body<'_>, v: VReg) -> VReg {
    let m = b.const_i64(0xFFFF_FFFF);
    b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitAnd,
        a: v,
        b: m,
    })
}

/// The runtime type of `recv`, whose static type is `static_ty`.
///
/// The object's first word is its vtable row, so the runtime class is
/// the one whose row it carries: compare the rows of the static class
/// and its subclasses and answer that class's descriptor. A boxed
/// value carries its descriptor directly. A class from another image
/// (dynamic linking) has no row here, so the static type answers it.
fn runtime_type_of(
    b: &mut Body<'_>,
    recv: VReg,
    static_ty: &str,
    ctx: &Ctx<'_>,
) -> Result<VReg, LowerError> {
    let static_desc = b.type_token(static_ty, ctx)?;
    // Candidates: the static class first, then its subclasses.
    let mut candidates: Vec<(String, u32)> = Vec::new();
    if let Some(info) = ctx.table.class(static_ty) {
        candidates.push((
            info.name.clone(),
            super::span::vtable_row_for(ctx, &info.name, &info.key.path),
        ));
        for decl in ctx.classes {
            let Some(dinfo) = ctx.table.class(&decl.name) else {
                continue;
            };
            if dinfo.name != info.name
                && ctx.table.derives_from(&dinfo.name, &info.name)
            {
                candidates.push((
                    dinfo.name.clone(),
                    construction_row(ctx, decl),
                ));
            }
        }
    }
    // A boxed value answers from its own descriptor.
    let boxed = b.fresh();
    b.push_raw(Instr::LoadField { dst: boxed, base: recv, off: 24 });
    let refc = b.fresh();
    b.push_raw(Instr::LoadField { dst: refc, base: recv, off: 8 });
    let flag = b.const_i64(BOX_FLAG);
    let flagged = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::BitAnd,
        a: refc,
        b: flag,
    });
    let zero = b.const_i64(0);
    let is_box = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: CmpOp::Ne,
        a: flagged,
        b: zero,
    });
    let res = b.fresh();
    let box_b = b.make_block();
    let class_b = b.make_block();
    let join_b = b.make_block();
    b.terminate(Terminator::Branch {
        cond: Some(is_box),
        if_true: box_b,
        if_false: class_b,
    });
    b.switch_to(box_b);
    b.mov(res, boxed);
    b.goto(join_b);
    b.switch_to(class_b);
    if candidates.is_empty() {
        b.mov(res, static_desc);
        b.goto(join_b);
    } else {
        // Compare the object's row with each candidate's row.
        let mut descs: Vec<VReg> = Vec::with_capacity(candidates.len());
        for (name, _) in &candidates {
            descs.push(b.type_token(name, ctx)?);
        }
        let row = b.fresh();
        b.push_raw(Instr::LoadField { dst: row, base: recv, off: 0 });
        // One "test candidate" block per candidate plus a final
        // fallback block (no row matched).
        let mut next_blocks = Vec::with_capacity(candidates.len() + 1);
        let mut found_blocks = Vec::with_capacity(candidates.len());
        for _ in 0..=candidates.len() {
            next_blocks.push(b.make_block());
        }
        for _ in &candidates {
            found_blocks.push(b.make_block());
        }
        b.goto(next_blocks[0]);
        for (i, (_, cand_row)) in candidates.iter().enumerate() {
            b.switch_to(next_blocks[i]);
            let cand = b.fresh();
            b.push_raw(Instr::VTableAddr { dst: cand, row: *cand_row });
            let eq = b.emit(Instr::Cmp {
                dst: VReg(0),
                op: CmpOp::Eq,
                a: row,
                b: cand,
            });
            b.terminate(Terminator::Branch {
                cond: Some(eq),
                if_true: found_blocks[i],
                if_false: next_blocks[i + 1],
            });
            b.switch_to(found_blocks[i]);
            b.mov(res, descs[i]);
            b.goto(join_b);
        }
        // No row matched (a class from another image): the static type.
        b.switch_to(next_blocks[candidates.len()]);
        b.mov(res, static_desc);
        b.goto(join_b);
    }
    b.switch_to(join_b);
    Ok(res)
}
