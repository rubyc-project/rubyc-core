//! AST to IR body lowering.

use super::super::{CmpOp, FBinOp, FCmpOp, Instr, LowerError, Terminator, VReg};
use crate::models::{ClassMember, Expr, InterpPart, Literal, UnOp};
use crate::native::ir::BinOp as IrBinOp;
use crate::native::ir::Ctx;
use crate::native::semantic::{ConstValue};
use super::{body::*, calls::*, select::*, types::*};
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
            let (off, len) = b.string_literal(s);
            b.build_string(off, len)
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
    ctx.table.class(type_name).is_some()
}

/// Evaluate an expression to a vreg.
pub(crate) fn eval(expr: &Expr, b: &mut Body<'_>, ctx: &Ctx<'_>) -> Result<VReg, LowerError> {
    match expr {
        Expr::NewArray { size, .. } => {
            // Allocate on heap via ALLOC helper; store size at [obj+0].
            // Elements start at slot 1 (after vtable+size header).
            let n_vreg = eval(size, b, ctx)?;
            // Multiply by 8 bytes per element + 8 header = (n+1)*8
            // For now allocate fixed 4096 bytes per array (safe upper bound).
            let alloc_size = b.const_i64(4096);
            let obj = b.fresh();
            b.alloc(obj, 4096);
            // Store length at offset 0 for bounds checking later
            b.push_raw(Instr::StoreField { base: obj, off: 0, src: n_vreg });
            let _ = alloc_size;
            Ok(obj)
        }
        Expr::Index { object, index, span: _, .. } => {
            // String char read: chars are 1 byte at `[base + 16 + i]` (layout:
            // [0..8)=len, [8..16)=refcount, [16..)=data). Zero-extend the byte.
            // Spelling-agnostic (`string`/`String`/`System.String`): an
            // exact match once routed `String`-spelled values down the
            // 8-byte array path, reading refcounts as chars.
            if is_string_type(&static_type_of(object, b, ctx)?) {
                let base = eval(object, b, ctx)?;
                let idx = eval(index, b, ctx)?;
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
                    b: idx,
                });
                let result = b.fresh();
                b.push_raw(Instr::LoadByte {
                    dst: result,
                    base: addr,
                    off: 0,
                });
                return Ok(result);
            }
            // Dynamic array element read. Layout (see `NewArray`): offset 0
            // holds the length header, elements start at offset 8 with an
            // 8-byte stride, so element `i` lives at `base + 8 + i*8`.
            // Bounds checking arrives with exceptions (A9).
            let base = eval(object, b, ctx)?;
            let idx = eval(index, b, ctx)?;
            let eight = b.const_i64(8);
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
                b: eight,
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
            Ok(result)
        }
        Expr::Cast { expr, .. } => eval(expr, b, ctx),
        Expr::Default { .. } => Ok(b.const_i64(0)),
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
                    ctx.table.class(&type_name)
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
                let field_is_ref = info
                    .field(name)
                    .map(|fi| ctx.table.class(&fi.type_name).is_some())
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
                // address is `base + 8 + i*8` (length header at [0]).
                let base = eval(object, b, ctx)?;
                let idx = eval(index, b, ctx)?;
                let v = eval(value, b, ctx)?;
                let eight = b.const_i64(8);
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
                    b: eight,
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
                    if let Some(info) = ctx.table.class(&type_name)
                        && let Some(field) = info.field(name)
                    {
                        ctx.check_access(true, &type_name, name)?;
                        let base = eval(object, b, ctx)?;
                        let field_is_ref = info
                            .field(name)
                            .map(|fi| ctx.table.class(&fi.type_name).is_some())
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
                    let base = eval(object, b, ctx)?;
                    let idx = eval(index, b, ctx)?;
                    let eight = b.const_i64(8);
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
                        b: eight,
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
            let v = eval(value, b, ctx)?;
            // `_` discard: the value is evaluated for side effects, then
            // dropped (mirrors the Stmt::Assign discard arm).
            if matches!(target.as_ref(), Expr::Ident { name, .. } if name == "_") {
                return Ok(v);
            }
            if let Expr::Ident { name, .. } = target.as_ref() {
                if let Some(sym) = b.scopes.lookup(name) {
                    let home = b.bindings[sym.slot];
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
                    b.push_raw(Instr::StoreField {
                        base: this,
                        off: (field.offset_slots * 8) as u32,
                        src: v,
                    });
                    return Ok(v);
                }
            }
            // Member target (`this.F = v`, `o.F = v` as an expression):
            // mirror Stmt::MemberAssign — resolve the field and store.
            // Static targets write the class slot (or the owner's slot
            // via import), never instance memory.
            if let Expr::Member { object, name, .. } = target.as_ref() {
                let type_name = static_type_of(object, b, ctx)?;
                if let Some(info) = ctx.table.class(&type_name)
                    && let Some(field) = info.field(name)
                {
                    ctx.check_access(true, &type_name, name)?;
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
                    let base = eval(object, b, ctx)?;
                    let field_is_ref = info
                        .field(name)
                        .map(|fi| ctx.table.class(&fi.type_name).is_some())
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
            }
            // Array element target (`a[i] = v` as an expression): same
            // layout as the read path (`base + 8 + i*8`).
            if let Expr::Index { object, index, .. } = target.as_ref() {
                let base = eval(object, b, ctx)?;
                let idx = eval(index, b, ctx)?;
                let eight = b.const_i64(8);
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
                    b: eight,
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
                return Ok(v);
            }
            // Accessor-backed property write in expression position
            // (see the statement arm): no storage, no setter function
            // yet — the value (already evaluated) stands as the result.
            if let Expr::Ident { name, .. } = target.as_ref()
                && this_info(ctx, name).is_some()
            {
                return Ok(v);
            }
            // Fallback: evaluate the target (side effects) and the value;
            // return the value.
            let _ = eval(target, b, ctx)?;
            Ok(v)
        }
        Expr::SwitchExpr { subject, .. } => eval(subject, b, ctx),
        Expr::Literal(Literal::Str(s)) => {
            let (off, len) = b.string_literal(s);
            Ok(b.build_string(off, len))
        }
        Expr::InterpolatedString { parts, .. } => {
            // Constant-fold when every part is a literal text run.
            if parts.iter().all(|p| matches!(p, InterpPart::Text(_))) {
                let s: String = parts
                    .iter()
                    .filter_map(|p| if let InterpPart::Text(t) = p { Some(t.as_str()) } else { None })
                    .collect();
                let (off, len) = b.string_literal(&s);
                return Ok(b.build_string(off, len));
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
                            let (off, len) = b.string_literal(&text);
                            let s = b.build_string(off, len);
                            acc = Some(match acc {
                                Some(a) => b.concat_strings(a, s),
                                None => s,
                            });
                            text.clear();
                        }
                        let v = eval(h, b, ctx)?;
                        acc = Some(match acc {
                            Some(a) => b.concat_strings(a, v),
                            None => v,
                        });
                    }
                }
            }
            if !text.is_empty() {
                let (off, len) = b.string_literal(&text);
                let s = b.build_string(off, len);
                acc = Some(match acc {
                    Some(a) => b.concat_strings(a, s),
                    None => s,
                });
            }
            Ok(acc.unwrap_or_else(|| {
                let (off, len) = b.string_literal("");
                b.build_string(off, len)
            }))
        }
        Expr::Literal(Literal::Int(v)) => Ok(b.emit(Instr::Const {
            dst: VReg(0),
            imm: *v,
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
                let dst = b.fresh();
                b.push_raw(Instr::Mov { dst, a: home });
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
                    let dst = b.fresh();
                    b.push_raw(Instr::LoadField {
                        dst,
                        base: this,
                        off: (field.offset_slots * 8) as u32,
                    });
                    return Ok(dst);
                }
                // Accessor-backed property read (`table` resolving to a
                // base-class property through a derived receiver):
                // properties occupy no storage, so reads lower to a
                // placeholder — but the read must SUCCEED (not error as
                // undeclared). Evaluate nothing (no receiver side
                // effects beyond `this`); access still checked.
                if info.properties.iter().any(|p| p.name == *name) {
                    ctx.check_access(true, ctx.current_class, name)?;
                    return Ok(b.const_i64(0));
                }
                // Event fields are accessor-backed with no object layout slot;
                // return 0 as a placeholder.
                if info
                    .own_members
                    .iter()
                    .any(|(mn, is_f, _)| mn == name && *is_f)
                {
                    return Ok(b.const_i64(0));
                }
            }
            // Enclosing-scope statics (`s_utcTimeZone` read from nested
            // `CachedData`): same static-storage read, declaring class
            // from the outer entry. Nested code may read enclosing
            // privates (nesting family); genuinely foreign access stays
            // loud.
            if let Some((outer, field)) = outer_static_field(ctx, name) {
                if !crate::native::semantic::visible_from(
                    ctx.table,
                    field.access,
                    &outer.name,
                    ctx.current_class,
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
            if (ctx.table.class(name).is_some()
                || name == "lambda"
                || name == "<pattern>"
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
            // bugs and produce wrong output. EXCEPTION: a bare method
            // name (method group) in value position — passed as a
            // delegate argument, stored, compared — has no value to
            // evaluate; yield zero like other delegate placeholders
            // (`: base(x, _HandleErrorCode)`, `M(cb)`).
            if is_method_group_name(name, b, ctx) {
                return Ok(b.const_i64(0));
            }
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
            // `*x` — dereference. For now, just evaluate the inner expression.
            eval(inner, b, ctx)
        }
        Expr::Unary {
            op: UnOp::AddrOf,
            inner,
            ..
        } => {
            // `&x` — address-of. For now, just evaluate the inner expression.
            eval(inner, b, ctx)
        }
        Expr::Unary {
            op: UnOp::Throw,
            inner,
            ..
        } => {
            // `throw expr` — for now, just evaluate the inner expression.
            eval(inner, b, ctx)
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
            b.switch_to(then_b);
            let t = eval(then_expr, b, ctx)?;
            b.mov(home, t);
            if !b.is_sealed() {
                b.goto(end_b);
            }
            b.switch_to(else_b);
            let e = eval(else_expr, b, ctx)?;
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
            // Logical operators normalize both sides to bool before the
            // bitwise combine; the eager form is branch-free and matches
            // C# results whenever the right side has no side effects.
            if matches!(op, crate::models::BinOp::And | crate::models::BinOp::Or) {
                let logical = if matches!(op, crate::models::BinOp::And) {
                    IrBinOp::And
                } else {
                    IrBinOp::Or
                };
                let a = eval(lhs, b, ctx)?;
                let bv = eval(rhs, b, ctx)?;
                let a = norm_bool(b, a);
                let bv = norm_bool(b, bv);
                return Ok(b.emit(Instr::Bin {
                    dst: VReg(0),
                    op: logical,
                    a,
                    b: bv,
                }));
            }
            let a = eval(lhs, b, ctx)?;
            let bv = eval(rhs, b, ctx)?;
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
                        // Object = 16-byte header (len + refcount) + data.
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
                let folded = fold_arith(op, va, vb);
                if let Some(v) = folded {
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
                if is_float_expr(b, ctx, lhs) || is_float_expr(b, ctx, rhs) {
                    let fcmpop = match op {
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
            // Type check (`is`) — parse-only; always false for now.
            if matches!(op, crate::models::BinOp::Is) {
                return Ok(b.emit(Instr::Const { dst: VReg(0), imm: 0 }));
            }
            // Null-coalescing (`??`) — return the right-hand side as a
            // placeholder (assume the left is non-null).
            if matches!(op, crate::models::BinOp::NullCoalesce) {
                return Ok(bv);
            }
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
            let binop = match op {
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
            Ok(b.emit(Instr::Bin {
                dst: VReg(0),
                op: binop,
                a,
                b: bv,
            }))
        }
        Expr::New {
            class_name, args, ..
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
                let c = qualified.as_str();
                if c.contains('.') {
                    c.rsplit('.').next().unwrap_or(c).to_owned()
                } else {
                    c.to_owned()
                }
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
                        static_type_of(a, b, ctx).unwrap_or_else(|_| "object".to_owned())
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
                        .find(|e| e.key != info.key && accepts(e))
                        .unwrap_or(info)
                }
            };
            let obj = b.fresh();
            b.alloc(obj, (info.fields.len() + 2) as u64 * 8);
            // `using`-alias spelling (`new WorkQueue()` for
            // `using WorkQueue = Queue<object>`): the alias itself
            // declares nothing — the AST declaration (field
            // initializers, primary params, bases) lives under the
            // target entry's simple name. Resolve once for the
            // declaration lookups below; anything else keeps the
            // call-site spelling.
            let alias_simple: Option<String> =
                ctx.table.alias_target(&class_name).map(|t| {
                    let base = t.split('<').next().unwrap_or(t);
                    base.rsplit('.').next().unwrap_or(base).to_owned()
                });
            let has_decl = |c: &&crate::models::ClassDecl| {
                c.name == *class_name || alias_simple.as_deref() == Some(c.name.as_str())
            };
            // Cross-image construction (dynamic-linking consume): no AST
            // declaration exists for this entry, and it ships from another
            // image — allocate against the owner's vtable row and run the
            // owner's ctor instead of the hollow local row below.
            if ctx.classes.iter().find(|c| has_decl(c)).is_none() && !info.owner.is_empty() {
                return lower_external_new(info, args, &class_name, b, ctx);
            }
            let vi = ctx
                .table
                .classes
                .iter()
                .position(|c| c.key == info.key)
                .unwrap_or(0);
            let vt = b.fresh();
            b.vtable_addr(vt, vi as u32);
            b.push_raw(Instr::StoreField { base: obj, off: 0, src: vt });

            if let Some(decl) = ctx.classes.iter().find(|c| has_decl(c)) {
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
                            static_type_of(a, b, ctx).unwrap_or_else(|_| "object".to_owned())
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
                let init_ctx = Ctx {
                    table: ctx.table,
                    current_class: &class_name,
                    current_info: None,
                    current_path: ctx.current_path,
                    classes: ctx.classes,
                    import_index: ctx.import_index,
                    dep_imports: ctx.dep_imports,
                    host_import_count: ctx.host_import_count,
                    static_index: ctx.static_index,
                    defines: ctx.defines,
                    method_vars: &[],
                };
                for f in &decl.members {
                    if let ClassMember::Field(fd) = f
                        && !fd.is_static
                    {
                        let v = eval(&fd.init, b, &init_ctx)?;
                        let off =
                            (info.field(&fd.name).map(|fi| fi.offset_slots).unwrap_or(1) * 8) as u32;
                        b.push_raw(Instr::StoreField { base: obj, off, src: v });
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
                        static_type_of(a, b, ctx).unwrap_or_else(|_| "object".to_owned())
                    })
                    .collect();
                // Typed ctor selection across same-named entries
                // (multi-arity generic instantiations like
                // `new TupleSlim<A,B,C>` share the bare name):
                // visibility-preferred throughout (see
                // `select_visible`). Layout/vtable stay first-match:
                // known generics-layout gap.
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
                    // Reuse the up-front evaluated args (no double side
                    // effects).
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
                        if !primary.is_empty() {
                            b.scope_exit();
                        }
                        return Ok(obj);
                    }
                    let fn_index = functions_index(ctx);
                    let idx = fn_index
                        .get(&format!("{class_name}::{class_name}#{}", mi.param_types.len()))
                        // Alias-spelled construction: the function index
                        // keys on declaring names (`Queue::Queue#0`, not
                        // `WorkQueue::WorkQueue#0`).
                        .or_else(|| {
                            fn_index.get(&format!(
                                "{}::{}#{}",
                                mi.declaring_class,
                                mi.name,
                                mi.param_types.len()
                            ))
                        })
                        .copied()
                        .unwrap_or(0);
                    let call_dst = b.fresh();
                    b.push_raw(Instr::CallFn {
                        dst: call_dst,
                        fidx: idx as u32,
                        args: argv,
                    });
                } else if let Some(base) = decl.bases.first() {
                    // C# implicit chaining: a class with no explicit ctor
                    // calls the first base's zero-arg ctor, if one exists.
                    if let Some(idx) =
                        functions_index(ctx).get(&format!("{base}::{base}#0")).copied()
                    {
                        let call_dst = b.fresh();
                        b.push_raw(Instr::CallFn {
                            dst: call_dst,
                            fidx: idx as u32,
                            args: vec![obj],
                        });
                    }
                }
                if !primary.is_empty() {
                    b.scope_exit();
                }
            }
            Ok(obj)
        }
        Expr::Member { object, name, .. } => {
            let type_name = static_type_of(object, b, ctx)?;
            // `string.Length` — a first-class string stores its byte length
            // at slot 0 (offset 0).
            if type_name == "string" && name == "Length" {
                let base = eval(object, b, ctx)?;
                let dst = b.fresh();
                b.push_raw(Instr::LoadField { dst, base, off: 0 });
                return Ok(dst);
            }
            // Nullable `.HasValue` / `.Value` — placeholders.
            if type_name.ends_with('?') && (name == "HasValue" || name == "Value") {
                return Ok(b.const_i64(0));
            }
            // Array `.Length` — stored at offset 0 of the array header.
            if name == "Length" && !ctx.table.class(&type_name).is_some() {
                // Non-class type with `.Length` — treat as array.
                let base = eval(object, b, ctx)?;
                let dst = b.fresh();
                b.push_raw(Instr::LoadField {
                    dst,
                    base,
                    off: 0,
                });
                return Ok(dst);
            }
            let Some(first) = ctx.table.class(&type_name) else {
                // Unknown type (e.g. generic type parameter `T`) — return 0
                // as a placeholder so compilation succeeds.
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
            {
                // Member not found as a field or own member — return 0 as a
                // placeholder (e.g. expression-bodied property).
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
                        let (off, len) = b.string_literal(s);
                        Ok(b.build_string(off, len))
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
                let Some(base) = static_addr(f, b, ctx) else {
                    return Ok(b.const_i64(0));
                };
                let dst = b.fresh();
                b.push_raw(Instr::LoadField { dst, base, off: 0 });
                return Ok(dst);
            }
            let base = eval(object, b, ctx)?;
            let dst = b.fresh();
            match field {
                Some(f) => {
                    b.push_raw(Instr::LoadField {
                        dst,
                        base,
                        off: (f.offset_slots * 8) as u32,
                    });
                }
                None => {
                    // Expression-bodied property or other accessor-backed
                    // member — return 0 as a placeholder.
                    let _ = b.const_i64(0);
                }
            }
            Ok(dst)
        }
        Expr::MethodCall {
            object, name, args, ..
        } => {
            // Strip generic-args suffix (e.g. `Convert<T3, TResult>` ->
            // `Convert`) — the parser embeds `<...>` in the name.
            // The arity still constrains overload ranking below
            // (`services.AddOptions<NoOp>()` must not elect the
            // non-generic `AddOptions()` overload).
            let explicit_arity = match name.find('<') {
                Some(_) => Some(explicit_type_args(name).len()),
                None => None,
            };
            let name = match name.find('<') {
                Some(pos) => &name[..pos],
                None => name.as_str(),
            };
            let type_name = static_type_of(object, b, ctx)?;

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
                    static_type_of(a, b, ctx).unwrap_or_else(|_| "object".to_owned())
                })
                .collect();

            // Import stubs take priority over virtual dispatch — but only
            // when the stub's arity matches the call (`new
            // RuntimeTypeHandle(this).MakePointer()` targets the
            // 0-arg instance overload, not the 2-arg QCall stub).
            // Mismatches fall through to normal overload resolution.
            // Static ABI: arguments only, no receiver.
            if let Some(&index) = ctx.import_index.get(&(type_name.clone(), name.to_string()))
                && ctx
                    .table
                    .imports
                    .get(index)
                    .is_some_and(|imp| imp.param_types.len() == args.len())
            {
                if !matches!(object.as_ref(), Expr::Ident { .. }) {
                    return Err(LowerError::UnsupportedConstruct(
                        "imports may only be called on a plain variable or this".into(),
                    ));
                }
                check_import(ctx, index, args)?;
                let argv = eval_args(args, b, ctx)?;
                let dst = b.fresh();
                b.push_raw(Instr::CallImport {
                    dst,
                    import: index as u32,
                    args: argv,
                });
                return Ok(dst);
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
                // Conditional winner in a mixed set: the resolved
                // overload itself is omitted when its symbol is
                // undefined (C# checks the winner, not the set).
                if mi
                    .conditional_symbol
                    .as_ref()
                    .is_some_and(|s| !ctx.defines.contains(s))
                {
                    return Ok(b.const_i64(0));
                }
            }
            match ctx.table.class(&type_name) {
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
                    // Extension methods also resolve on receivers without
                    // a class entry (primitives like `int`, generic
                    // instantiations): `v.Twice()` → `Ext.Twice(v)`.
                    // Unbound class type variables land here too (see
                    // the guard on the `Some` arm above).
                    if let Some(ext_mi) = extension_method(
                        ctx.table,
                        &type_name,
                        name,
                        &arg_types,
                        explicit_arity,
                    ) {
                        return lower_extension_call(object, args, ext_mi, b, ctx);
                    }
                    // Interface-typed receiver (e.g. `IShape s`). Confirm the
                    // method is declared on the interface, then fall through
                    // to the shared virtual-dispatch path below.
                    let Some(_iface) = ctx.table.interface(&type_name) else {
                        // Unknown type (e.g. generic type parameter `T`) —
                        // return 0 as a placeholder so compilation succeeds.
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
            }
            let dst = b.fresh();
            if is_static_receiver {
                if method_static == Some(true) {
                    // C# `C.M(...)` — a static method, possibly inherited
                    // from a base class. Resolve the declaring class's
                    // function (keyed by declaring, not receiver
                    // spelling); statics are not vtable-dispatched.
                    let Some(fidx) = static_function_index_for(ctx, &selected_declaring, name, selected_arity) else {
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
                    b.push_raw(Instr::CallStatic { dst, fidx, args: argv });
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
            b.push_raw(Instr::CallVirt {
                dst,
                slot: canonical_slot(ctx.table, None, &type_name, name) as u32,
                recv,
                args: argv,
            });
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
            b.push_raw(Instr::Suspend { state });
            Ok(value)
        }
    }
}
