//! Packed span views (`System.Span<T>` / `System.ReadOnlySpan<T>`) over
//! arrays and strings.
//!
//! Representation: an ordinary heap object matching the `ReadOnlySpan<T>`
//! struct layout (`_reference` then `_length` fields, read from the type
//! table, never hardcoded). Array-to-span argument coercion builds a
//! VIEW over the array payload (length from the header, data at +16 —
//! the same addresses the element path uses), so mutations through
//! the span reach the array (a copying pack silently lost every
//! `CopyTo`/`Fill` write). Strings still snapshot into a packed
//! buffer (immutable source — unobservable). The explicit `ToArray`
//! fallback still copies (see `lower_to_array_pack` in `eval.rs`).
//! Generic-instantiation calls whose width is only known at the call
//! site (`new Span<T>(array, start, length)`, `Slice`) inline here
//! with the caller's closed element type: the shared definition
//! body folds its address arithmetic with the OPEN type variable
//! (width 8), misplacing every narrow start/slice.
//!
//! Only the operations the runtime needs are intercepted here (see
//! the call sites in `eval.rs`); everything else flows through the
//! ordinary struct/member machinery on top of real storage.

use super::super::{CmpOp, Instr, LowerError, Terminator, VReg};
use super::{body::*, select::*};
use crate::models::Expr;
use crate::native::ir::BinOp as IrBinOp;
use crate::native::ir::Ctx;
use crate::native::semantic::ClassInfo;

/// `(is_readonly, inner)` for `Span<T>` / `ReadOnlySpan<T>` spellings
/// in any qualification (`System.ReadOnlySpan<byte>`,
/// `global::System.Span<char>`, ...). Anything else yields `None`.
pub(super) fn span_view(ty: &str) -> Option<(bool, String)> {
    let s = ty.trim().trim_start_matches("global::");
    let lt = s.find('<')?;
    let head = s[..lt].rsplit('.').next().unwrap_or(&s[..lt]);
    let readonly = match head {
        "ReadOnlySpan" => true,
        "Span" => false,
        _ => return None,
    };
    // Balanced inner between the outer `<...>` (nested generics stay
    // whole, like the span extraction in `types.rs`).
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    let mut end = bytes.len();
    for (i, &c) in bytes.iter().enumerate().skip(lt) {
        match c {
            b'<' => depth += 1,
            b'>' => {
                depth -= 1;
                if depth == 0 {
                    end = i;
                    break;
                }
            }
            _ => {}
        }
    }
    let inner = s[lt + 1..end].trim().to_owned();
    if inner.is_empty() {
        return None;
    }
    Some((readonly, inner))
}

/// Packed element width for a span inner type: true primitive widths
/// (`byte` → 1, `char` → 2, `int` → 4, ...); reference, unknown, and
/// generic-parameter inners occupy a full slot (8).
pub(super) fn span_elem_width(inner: &str) -> usize {
    // By-reference spellings (`byte&`, `ref byte`) denote the
    // element type: strip them (a `ref byte` address scales by 1,
    // not 8 — the `NarrowTwo` mis-stride that zeroed every
    // `GetBytes` output lived here).
    let mut t = inner.trim();
    for w in ["ref ", "out ", "in "] {
        if let Some(rest) = t.strip_prefix(w) {
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
    match canon.as_str() {
        "Boolean" | "Byte" | "SByte" => 1,
        "Char" | "Int16" | "UInt16" => 2,
        "Int32" | "UInt32" | "Single" => 4,
        "Int64" | "UInt64" | "Double" | "IntPtr" | "UIntPtr" => 8,
        _ => 8,
    }
}

/// Closed element width for caller-side span inlining (`new
/// Span<T>(array, start, length)`, `Slice`): `Some(w)` only when the
/// inner spelling names a primitive with a known width. Open type
/// variables (`T`, even ones invisible to the caller's own generic
/// context — e.g. leaking from an extension's declared return),
/// references, and unknown structs yield `None` so callers keep
/// normal dispatch. `span_elem_width`'s default-8 bucket cannot
/// distinguish these, and inlining with a defaulted width
/// misplaces narrow spans exactly like the shared body does.
pub(super) fn closed_elem_width(inner: &str) -> Option<usize> {
    let mut t = inner.trim();
    for w in ["ref ", "out ", "in "] {
        if let Some(rest) = t.strip_prefix(w) {
            t = rest;
            break;
        }
    }
    let simple = t
        .rsplit('.')
        .next()
        .unwrap_or(t)
        .trim_end_matches(['?', '*', '&']);
    // Same arms as `span_elem_width`, minus the default bucket: only
    // positively-known primitive widths inline (notably NOT Boolean —
    // `span_elem_width` folds it to a full slot, and inlining with a
    // different stride would corrupt bool spans either way).
    match canonical_primitive(simple).as_str() {
        "Byte" | "SByte" | "byte" | "sbyte" => Some(1),
        "Char" | "Int16" | "UInt16" | "char" | "short" | "ushort" => Some(2),
        "Int32" | "UInt32" | "Single" | "int" | "uint" | "float" => Some(4),
        "Int64" | "UInt64" | "Double" | "IntPtr" | "UIntPtr" | "long" | "ulong" | "double" => Some(8),
        _ => None,
    }
}

/// Vtable row for a class entry: counts table-backed classes before
/// it, mirroring `Program` vtable construction in `ir/mod.rs`.
pub(super) fn vtable_row_for(
    ctx: &Ctx<'_>,
    entry_name: &str,
    entry_path: &str,
) -> u32 {
    let mut vi: u32 = 0;
    for c in ctx.classes.iter() {
        if c.name == entry_name || c.name == entry_path {
            break;
        }
        if ctx.table.class(&c.name).is_some() {
            vi += 1;
        }
    }
    vi
}

/// Resolve the class entry backing a span spelling (exact hit,
/// else same-base entry carrying the inner type). Span spellings
/// arrive short (`ReadOnlySpan<byte>`) while keys are full paths.
fn span_entry<'t>(
    ctx: &'t Ctx<'_>,
    span_ty: &str,
) -> Option<&'t ClassInfo> {
    if let Some(e) = ctx.table.class(span_ty) {
        return Some(e);
    }
    let (ro, inner) = span_view(span_ty)?;
    let want = if ro { "ReadOnlySpan" } else { "Span" };
    // Instantiation carrying this inner type, else the generic
    // definition (members and layout shared by all inners).
    ctx.table.classes.iter().find(|c| {
        span_view(&c.key.path).is_some_and(|(r, i)| {
            r == ro && base_type_name(&i) == base_type_name(&inner)
        }) || {
            let base = c
                .key
                .path
                .rsplit('.')
                .next()
                .unwrap_or(&c.key.path)
                .split('<')
                .next()
                .unwrap_or(&c.key.path);
            base == want
                && (c.key.path.contains(inner.as_str())
                    || c.key.arity > 0
                    || !c.generic_params.is_empty())
        }
    })
}

/// `(_reference, _length)` field offsets for a span class,
/// located through the table (layout follows declaration order),
/// falling back to the canonical slots when the entry is thin
/// (generic instantiations may not repeat fields).
pub(super) fn span_field_offsets(
    ctx: &Ctx<'_>,
    span_ty: &str,
) -> Option<(u32, u32)> {
    let entry = span_entry(ctx, span_ty)?;
    let ref_off = entry
        .field("_reference")
        .map(|f| (f.offset_slots * 8) as u32)
        .unwrap_or(16);
    let len_off = entry
        .field("_length")
        .map(|f| (f.offset_slots * 8) as u32)
        .unwrap_or(24);
    Some((ref_off, len_off))
}
/// Allocate a span object for an already-packed buffer: header +
/// `_reference`/`_length` fields located through the table (never
/// hardcoded), vtable row set so methods dispatch.
pub(super) fn span_object(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    span_ty: &str,
    data: VReg,
    len: VReg,
) -> Result<VReg, LowerError> {
    // Best-effort entry for vtable naming; layout fallbacks below
    // keep unregistered spellings working.
    let entry = span_entry(ctx, span_ty);
    let (ref_off, len_off) = span_field_offsets(ctx, span_ty).unwrap_or((16, 24));
    let (vname, vpath) = match &entry {
        Some(e) => (e.name.clone(), e.key.path.clone()),
        None => (span_ty.to_owned(), span_ty.to_owned()),
    };
    let obj = b.fresh();
    b.alloc(obj, 32);
    let vt = b.fresh();
    b.vtable_addr(vt, vtable_row_for(ctx, &vname, &vpath));
    b.push_raw(Instr::StoreField {
        base: obj,
        off: 0,
        src: vt,
    });
    let one = b.const_i64(1);
    b.push_raw(Instr::StoreField {
        base: obj,
        off: 8,
        src: one,
    });
    b.push_raw(Instr::StoreField {
        base: obj,
        off: ref_off,
        src: data,
    });
    b.push_raw(Instr::StoreField {
        base: obj,
        off: len_off,
        src: len,
    });
    Ok(obj)
}

/// `MemoryMarshal.Cast` reinterpretation: same buffer, length
/// rescaled by the element-width ratio (`len * from_w / to_w`).
/// Array/string sources pack first (element type from the source
/// spelling); span sources reinterpret in place. Falls back to
/// length-preserving when widths are unknown.
pub(super) fn span_cast(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    src: VReg,
    from_ty: &str,
    to_inner: &str,
) -> Result<VReg, LowerError> {
    // Normalize the source to a span: arrays and strings pack
    // (their element type is the spelling's inner), spans pass
    // through for reinterpretation below.
    let (span, from_inner) = match span_view(from_ty) {
        Some((_, inner)) => (src, inner),
        None => {
            let elem = if is_string_type(from_ty) {
                "char".to_owned()
            } else if let Some(e) = from_ty.trim_end().strip_suffix("[]") {
                e.to_owned()
            } else {
                return Err(LowerError::UnsupportedConstruct(format!(
                    "span cast source: {from_ty}"
                )));
            };
            let tmp_ty = span_class_path(ctx, &format!("ReadOnlySpan<{elem}>"))
                .unwrap_or_else(|| "ReadOnlySpan<byte>".to_owned());
            let packed = if is_string_type(from_ty) {
                pack_span_from_string(b, ctx, src, &tmp_ty, &elem)?
            } else {
                pack_span_from_array(b, ctx, src, &tmp_ty, &elem)?
            };
            (packed, elem)
        }
    };
    let from_ty_eff = span_class_path(ctx, &format!("ReadOnlySpan<{from_inner}>"))
        .unwrap_or_else(|| from_ty.to_owned());
    let from_w = span_elem_width(&from_inner);
    let to_w = span_elem_width(to_inner);
    let (ref_off, len_off) = span_field_offsets(ctx, &from_ty_eff).unwrap_or((16, 24));
    let r = b.fresh();
    b.push_raw(Instr::LoadField {
        dst: r,
        base: span,
        off: ref_off,
    });
    let l = b.fresh();
    b.push_raw(Instr::LoadField {
        dst: l,
        base: span,
        off: len_off,
    });
    let fw = b.const_i64(from_w as i64);
    let tw = b.const_i64(to_w.max(1) as i64);
    let scaled = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Mul,
        a: l,
        b: fw,
    });
    // Integer division; guard against a zero divisor structurally
    // (`to_w.max(1)` above) since widths are always positive.
    let nlen = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Div,
        a: scaled,
        b: tw,
    });
    // Target entry: same span shape with the new inner type when
    // the table has it, else the source spelling (layout-identical).
    let target = span_entry(ctx, &from_ty_eff)
        .map(|e| e.key.path.clone())
        .unwrap_or_else(|| from_ty_eff.clone());
    span_object(b, ctx, &target, r, nlen)
}

/// Span slice view (no copy): `{ref + start*w, len}` with the
/// receiver's element width. `len` is `None` for the
/// single-argument form (rest of the span).
pub(super) fn span_slice(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    span: VReg,
    span_ty: &str,
    inner: &str,
    start: VReg,
    len: Option<VReg>,
) -> Result<VReg, LowerError> {
    let w = span_elem_width(inner);
    let (ref_off, len_off) = span_field_offsets(ctx, span_ty).unwrap_or((16, 24));
    let r = b.fresh();
    b.push_raw(Instr::LoadField {
        dst: r,
        base: span,
        off: ref_off,
    });
    let l0 = b.fresh();
    b.push_raw(Instr::LoadField {
        dst: l0,
        base: span,
        off: len_off,
    });
    let wv = b.const_i64(w as i64);
    let off = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Mul,
        a: start,
        b: wv,
    });
    let nref = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: r,
        b: off,
    });
    let nlen = match len {
        Some(l) => l,
        None => {
            let d = b.emit(Instr::Bin {
                dst: VReg(0),
                op: IrBinOp::Sub,
                a: l0,
                b: start,
            });
            d
        }
    };
    span_object(b, ctx, span_ty, nref, nlen)
}

/// Span element read: zero-extended `w`-byte load at
/// `ref + i*w`, with `w` from the static inner type.
pub(super) fn span_index(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    span: VReg,
    span_ty: &str,
    inner: &str,
    idx: VReg,
) -> Result<VReg, LowerError> {
    let w = span_elem_width(inner);
    let (ref_off, _) = span_field_offsets(ctx, span_ty).unwrap_or((16, 24));
    let r = b.fresh();
    b.push_raw(Instr::LoadField {
        dst: r,
        base: span,
        off: ref_off,
    });
    let wv = b.const_i64(w as i64);
    let off = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Mul,
        a: idx,
        b: wv,
    });
    let addr = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: r,
        b: off,
    });
    let dst = b.fresh();
    if w == 1 {
        b.push_raw(Instr::LoadByte {
            dst,
            base: addr,
            off: 0,
        });
        return Ok(dst);
    }
    b.push_raw(Instr::LoadField {
        dst,
        base: addr,
        off: 0,
    });
    if w < 8 {
        let mask = b.const_i64((1i64 << (w * 8)) - 1);
        let clean = b.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::BitAnd,
            a: dst,
            b: mask,
        });
        return Ok(clean);
    }
    Ok(dst)
}
/// Copy loop: `n` elements from `src` (stride `src_stride`,
/// copying `copy_w` bytes each) to `dst` (stride `dst_stride`).
/// A full zero slot lands first so narrow copies never inherit
/// freelist garbage in the padding bytes (safe for all widths:
/// the zero store at `i*dst_stride` only touches elements
/// `i..`, never already-copied ones before `i`).
/// Copy `n` elements `src_stride` apart into slots `dst_stride`
/// apart, zeroing each destination slot first (see body).
pub(super) fn pack_loop(
    b: &mut Body<'_>,
    dst: VReg,
    src: VReg,
    n: VReg,
    dst_stride: usize,
    copy_w: usize,
    src_stride: usize,
) {
    let dw = b.const_i64(dst_stride as i64);
    let cw = b.const_i64(copy_w as i64);
    let sv = b.const_i64(src_stride as i64);
    let zero = b.const_i64(0);
    let one = b.const_i64(1);
    let i = b.fresh();
    b.mov(i, zero);
    let cond_b = b.make_block();
    let body_b = b.make_block();
    let end_b = b.make_block();
    b.goto(cond_b);
    b.switch_to(body_b);
    // Zero a full slot first so narrow copies never inherit
    // freelist garbage in the padding bytes (safe for all w:
    // the zero store at `i*w` only touches elements `i..i+7`,
    // never already-copied ones before `i`).
    let doff = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Mul,
        a: i,
        b: dw,
    });
    let dptr = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: dst,
        b: doff,
    });
    b.push_raw(Instr::StoreField {
        base: dptr,
        off: 0,
        src: zero,
    });
    let soff = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Mul,
        a: i,
        b: sv,
    });
    let sptr = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: src,
        b: soff,
    });
    b.push_raw(Instr::MemCpy {
        dst: dptr,
        src: sptr,
        len: cw,
    });
    let ni = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: i,
        b: one,
    });
    b.mov(i, ni);
    b.goto(cond_b);
    b.switch_to(cond_b);
    let c = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: CmpOp::Lt,
        a: i,
        b: n,
    });
    b.terminate(Terminator::Branch {
        cond: Some(c),
        if_true: body_b,
        if_false: end_b,
    });
    b.switch_to(end_b);
}

/// Resolve the class path backing a span parameter spelling
/// (exact hit, else same-base entry carrying the inner type).
/// Resolve the class path backing a span parameter spelling
/// (exact hit, else same-base entry carrying the inner type, else
/// the generic definition which serves all inners).
pub(super) fn span_class_path(ctx: &Ctx<'_>, param_ty: &str) -> Option<String> {
    if ctx.table.class(param_ty).is_some() {
        return Some(param_ty.to_owned());
    }
    let (ro, inner) = span_view(param_ty)?;
    let want = if ro { "ReadOnlySpan" } else { "Span" };
    // Instantiation entry carrying this inner type first.
    if let Some(p) = ctx
        .table
        .classes
        .iter()
        .map(|c| c.key.path.clone())
        .find(|p| {
            span_view(p).is_some_and(|(r, i)| r == ro && base_type_name(&i) == base_type_name(&inner))
                || {
                    let base = p
                        .rsplit('.')
                        .next()
                        .unwrap_or(p)
                        .split('<')
                        .next()
                        .unwrap_or(p);
                    base == want && p.contains(inner.as_str())
                }
        })
    {
        return Some(p);
    }
    // Generic definition (layout and members shared by all inners).
    ctx.table
        .classes
        .iter()
        .map(|c| c.key.path.clone())
        .find(|p| {
            let base = p
                .rsplit('.')
                .next()
                .unwrap_or(p)
                .split('<')
                .next()
                .unwrap_or(p);
            base == want
                && ctx
                    .table
                    .class(p)
                    .is_some_and(|e| e.key.arity > 0 || !e.generic_params.is_empty())
        })
}

/// Coerce one call argument into a span parameter: arrays and
/// strings pack into views; already-span values pass through.
/// Returns the replacement vreg, or `None` when no coercion
/// applies (leaving the original argument untouched).
pub(super) fn coerce_span_arg(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    argv: VReg,
    arg_ty: &str,
    param_ty: &str,
) -> Result<Option<VReg>, LowerError> {
    let Some((_, inner)) = span_view(param_ty) else {
        return Ok(None);
    };
    if span_view(arg_ty).is_some() {
        return Ok(None);
    }
    let Some(path) = span_class_path(ctx, param_ty) else {
        return Ok(None);
    };
    if let Some(_elem) = array_elem_ty(arg_ty) {
        // VIEW, not snapshot: build the span directly over the
        // array's payload (length from the header, data at +16 —
        // the same addresses the element path uses). A copying
        // pack silently lost every mutation through the span
        // (`CopyTo`/`Fill` into coerced spans wrote a scratch
        // buffer; the array never changed) and read stale data.
        let l = b.fresh();
        b.push_raw(Instr::LoadField { dst: l, base: argv, off: 0 });
        let sixteen = b.const_i64(16);
        let data = b.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::Add,
            a: argv,
            b: sixteen,
        });
        return Ok(Some(span_object(b, ctx, &path, data, l)?));
    }
    if is_string_type(arg_ty) {
        return Ok(Some(pack_span_from_string(b, ctx, argv, &path, &inner)?));
    }
    Ok(None)
}

/// Apply span coercion across a call's argument list (declared
/// parameter types decide; actual argument types select). Missing
/// types coerce nothing — legacy behavior unchanged.
pub(super) fn coerce_span_args(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    argv: &mut [VReg],
    arg_types: &[String],
    param_types: &[String],
) -> Result<(), LowerError> {
    for ((slot, param), arg_ty) in argv.iter_mut().zip(param_types.iter()).zip(arg_types.iter())
    {
        if let Some(v) = coerce_span_arg(b, ctx, *slot, arg_ty, param)? {
            *slot = v;
        }
    }
    Ok(())
}

/// Coerce a `return` value into the declared return type: arrays
/// and strings pack into span-typed returns (implicit C#
/// conversion, e.g. `ReadOnlySpan<uint> P => [0, 31, …]`).
/// Returns the replacement vreg, or the original when no
/// coercion applies.
pub(super) fn coerce_span_return(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    v: VReg,
    val_ty: &str,
    ret_ty: &str,
) -> Result<VReg, LowerError> {
    if let Some(nv) = coerce_span_arg(b, ctx, v, val_ty, ret_ty)? {
        return Ok(nv);
    }
    Ok(v)
}

/// Inline `span.StartsWith(value)` / `span.EndsWith(value)` for a
/// single closed-primitive value (no comparer): length check plus a
/// direct first/last element compare. The shared generic bodies wrap
/// the element in nullable gymnastics (`span[0]?.Equals(value) ??
/// (object?)value is null`) that miscompile (faulted path checks
/// during file open). Structs can't derive, so no override is
/// shadowed; anything else keeps normal dispatch (`None`).
/// Evaluates receiver then value left-to-right, exactly once.
pub(super) fn inline_span_starts_ends(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    object: &Expr,
    args: &[Expr],
    recv_ty: &str,
    arg_types: &[String],
    from_end: bool,
) -> Result<Option<VReg>, LowerError> {
    if args.len() != 1 || arg_types.len() != 1 {
        return Ok(None);
    }
    let elem = if let Some((_, inner)) = span_view(recv_ty) {
        inner
    } else if let Some(e) = array_elem_ty(recv_ty) {
        e
    } else {
        return Ok(None);
    };
    let Some(w) = closed_elem_width(&elem) else {
        return Ok(None);
    };
    if !span_search_cmp_ty(&elem) || !span_search_cmp_ty(&arg_types[0]) {
        return Ok(None);
    }
    let recv = super::eval::eval(object, b, ctx)?;
    let val = super::eval::eval(&args[0], b, ctx)?;
    let (base, len) = if span_view(recv_ty).is_some() {
        let (ref_off, len_off) = span_field_offsets(ctx, recv_ty).unwrap_or((16, 24));
        let r = b.fresh();
        b.push_raw(Instr::LoadField {
            dst: r,
            base: recv,
            off: ref_off,
        });
        let l = b.fresh();
        b.push_raw(Instr::LoadField {
            dst: l,
            base: recv,
            off: len_off,
        });
        (r, l)
    } else {
        let sixteen = b.const_i64(16);
        let data = b.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::Add,
            a: recv,
            b: sixteen,
        });
        let l = b.fresh();
        b.push_raw(Instr::LoadField {
            dst: l,
            base: recv,
            off: 0,
        });
        (data, l)
    };
    let result = b.fresh();
    let zero = b.const_i64(0);
    b.mov(result, zero);
    let nonempty_b = b.make_block();
    let done_b = b.make_block();
    let has = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: CmpOp::Gt,
        a: len,
        b: zero,
    });
    b.terminate(Terminator::Branch {
        cond: Some(has),
        if_true: nonempty_b,
        if_false: done_b,
    });
    b.switch_to(nonempty_b);
    let idx = if from_end {
        let one = b.const_i64(1);
        b.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::Sub,
            a: len,
            b: one,
        })
    } else {
        zero
    };
    let wv = b.const_i64(w as i64);
    let off = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Mul,
        a: idx,
        b: wv,
    });
    let addr = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: base,
        b: off,
    });
    let raw = b.fresh();
    b.push_raw(Instr::LoadField {
        dst: raw,
        base: addr,
        off: 0,
    });
    let mut e = raw;
    if let Some(mask) = super::eval::narrow_load_mask(&elem) {
        let m = b.const_i64(mask);
        e = b.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::BitAnd,
            a: raw,
            b: m,
        });
    } else if let Some(bits) = super::eval::signed_narrow_bits(&elem) {
        let mask = b.const_i64((1i64 << bits) - 1);
        let narrowed = b.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::BitAnd,
            a: raw,
            b: mask,
        });
        let sh = b.const_i64((64 - bits) as i64);
        let up = b.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::Shl,
            a: narrowed,
            b: sh,
        });
        e = b.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::Shr,
            a: up,
            b: sh,
        });
    }
    let eq = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: CmpOp::Eq,
        a: e,
        b: val,
    });
    let set_b = b.make_block();
    b.terminate(Terminator::Branch {
        cond: Some(eq),
        if_true: set_b,
        if_false: done_b,
    });
    b.switch_to(set_b);
    let one = b.const_i64(1);
    b.mov(result, one);
    b.goto(done_b);
    b.switch_to(done_b);
    Ok(Some(result))
}

/// Closed integer-like primitive for span search/slice value
/// comparison (`bool`, `char`, integrals — never floats, whose bitwise
/// `==` differs from C# `==`, and never comparers/objects).
fn span_search_cmp_ty(t: &str) -> bool {
    let mut s = t.trim().to_owned();
    loop {
        if s.ends_with("[]") {
            s.pop();
            s.pop();
        } else if s.ends_with('?') || s.ends_with('*') {
            s.pop();
        } else {
            break;
        }
    }
    let simple = s.rsplit('.').next().unwrap_or(&s);
    matches!(
        canonical_primitive(simple).as_str(),
        "Boolean" | "Byte" | "SByte" | "Char" | "Int16" | "UInt16" | "Int32"
            | "UInt32" | "Int64" | "UInt64"
    )
}

/// Scalar `IndexOf` / `IndexOfAny` (2–3 values) / `Contains` over a
/// span or array with closed primitive elements and values, inlined
/// here with exact widths and integer comparison. The shared generic
/// bodies fold address arithmetic with the OPEN type variable
/// (width 8) and dispatch `Equals` virtually, faulting (formatting
/// `IndexOfAny('{', '}')` died inside `SpanHelpers.IndexOfAny`).
/// Only integer-like primitives (`bool`, `char`, integrals — never
/// floats, whose bitwise `==` differs from C# `==`, and never
/// comparers) and exact arities (extra args mean a comparer
/// overload); everything else keeps normal dispatch (`None`).
/// Returns the first index, `-1`, or a bool for `Contains`.
/// Evaluates receiver then values left-to-right, exactly once —
/// callers must not pre-evaluate (double side effects).
pub(super) fn inline_span_search(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    object: &Expr,
    args: &[Expr],
    recv_ty: &str,
    arg_types: &[String],
    as_bool: bool,
) -> Result<Option<VReg>, LowerError> {
    // Receiver element type (span inner or array element,
    // nullable-annotated arrays included).
    let elem = if let Some((_, inner)) = span_view(recv_ty) {
        inner
    } else if let Some(e) = array_elem_ty(recv_ty) {
        e
    } else {
        return Ok(None);
    };
    let Some(w) = closed_elem_width(&elem) else {
        return Ok(None);
    };
    if !span_search_cmp_ty(&elem) {
        return Ok(None);
    }
    if arg_types.len() != args.len() {
        return Ok(None);
    }
    for t in arg_types {
        if !span_search_cmp_ty(t) {
            return Ok(None);
        }
    }
    let recv = super::eval::eval(object, b, ctx)?;
    let mut vals = Vec::with_capacity(args.len());
    for a in args {
        vals.push(super::eval::eval(a, b, ctx)?);
    }
    // Receiver base address + element count.
    let (base, len) = if span_view(recv_ty).is_some() {
        let (ref_off, len_off) = span_field_offsets(ctx, recv_ty).unwrap_or((16, 24));
        let r = b.fresh();
        b.push_raw(Instr::LoadField {
            dst: r,
            base: recv,
            off: ref_off,
        });
        let l = b.fresh();
        b.push_raw(Instr::LoadField {
            dst: l,
            base: recv,
            off: len_off,
        });
        (r, l)
    } else {
        let sixteen = b.const_i64(16);
        let data = b.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::Add,
            a: recv,
            b: sixteen,
        });
        let l = b.fresh();
        b.push_raw(Instr::LoadField {
            dst: l,
            base: recv,
            off: 0,
        });
        (data, l)
    };
    let result = b.fresh();
    if as_bool {
        let z = b.const_i64(0);
        b.mov(result, z);
    } else {
        let neg = b.const_i64(-1);
        b.mov(result, neg);
    }
    let i = b.fresh();
    let zero = b.const_i64(0);
    b.mov(i, zero);
    let cond_b = b.make_block();
    let body_b = b.make_block();
    let done_b = b.make_block();
    b.goto(cond_b);
    b.switch_to(body_b);
    let wv = b.const_i64(w as i64);
    let off = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Mul,
        a: i,
        b: wv,
    });
    let addr = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: base,
        b: off,
    });
    let raw = b.fresh();
    b.push_raw(Instr::LoadField {
        dst: raw,
        base: addr,
        off: 0,
    });
    // Mask narrow loads (dense neighbors share slots); sign-extend
    // signed ones so negative values compare equal to their
    // literals (mirrors the array read path).
    let mut e = raw;
    if let Some(mask) = super::eval::narrow_load_mask(&elem) {
        let m = b.const_i64(mask);
        e = b.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::BitAnd,
            a: raw,
            b: m,
        });
    } else if let Some(bits) = super::eval::signed_narrow_bits(&elem) {
        let mask = b.const_i64((1i64 << bits) - 1);
        let narrowed = b.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::BitAnd,
            a: raw,
            b: mask,
        });
        let sh = b.const_i64((64 - bits) as i64);
        let up = b.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::Shl,
            a: narrowed,
            b: sh,
        });
        e = b.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::Shr,
            a: up,
            b: sh,
        });
    }
    let mut matched = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: CmpOp::Eq,
        a: e,
        b: vals[0],
    });
    for v in vals.iter().skip(1) {
        let eq = b.emit(Instr::Cmp {
            dst: VReg(0),
            op: CmpOp::Eq,
            a: e,
            b: *v,
        });
        matched = b.emit(Instr::Bin {
            dst: VReg(0),
            op: IrBinOp::BitOr,
            a: matched,
            b: eq,
        });
    }
    let hit_b = b.make_block();
    let next_b = b.make_block();
    b.terminate(Terminator::Branch {
        cond: Some(matched),
        if_true: hit_b,
        if_false: next_b,
    });
    b.switch_to(hit_b);
    if as_bool {
        let one = b.const_i64(1);
        b.mov(result, one);
    } else {
        b.mov(result, i);
    }
    b.goto(done_b);
    b.switch_to(next_b);
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
        b: len,
    });
    b.terminate(Terminator::Branch {
        cond: Some(more),
        if_true: body_b,
        if_false: done_b,
    });
    b.switch_to(done_b);
    Ok(Some(result))
}
/// `new ReadOnlySpan<T>(pointer, length)` for pointer first
/// arguments: store the pointer and length directly. Width-free
/// (no arithmetic), so open `T` is fine. The shared body is
/// equally trivial, but overload selection can prefer the
/// `(ref T, int)` twin for `byte*` arguments (no `byte*`→`void*`
/// conversion rule), whose byref-field assignment then faults —
/// this killed console Flush inside the encoding fallback stub.
/// Returns `None` (normal construction) unless the shape matches
/// and the definition path resolves.
pub(super) fn span_from_pointer_view(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    ptr: VReg,
    len: VReg,
    span_ty: &str,
) -> Result<Option<VReg>, LowerError> {
    if span_view(span_ty).is_none() {
        return Ok(None);
    }
    let Some(path) = span_class_path(ctx, span_ty) else {
        return Ok(None);
    };
    Ok(Some(span_object(b, ctx, &path, ptr, len)?))
}

/// Inline `span.Fill(value)` with the CALLER's closed element
/// width: scalar store loop over `_reference`. The shared bodies
/// (`Span<T>.Fill` → `SpanHelpers.Fill`) run the vectorized path
/// (no SIMD here) or stride-8 scalar fallback (open `T`), silently
/// writing nothing visible. Full-slot stores match normal element
/// stores (readers mask). Returns a dummy zero (fills are
/// statements; C# `void`). `None` (normal dispatch) unless the
/// receiver is a `Span`/`ReadOnlySpan` with a concrete inner type.
/// Evaluates receiver then value left-to-right, exactly once.
pub(super) fn inline_span_fill(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    object: &Expr,
    args: &[Expr],
    recv_ty: &str,
) -> Result<Option<VReg>, LowerError> {
    if args.len() != 1 {
        return Ok(None);
    }
    let Some((_, inner)) = span_view(recv_ty) else {
        return Ok(None);
    };
    let Some(w) = closed_elem_width(&inner) else {
        return Ok(None);
    };
    let (ref_off, len_off) = span_field_offsets(ctx, recv_ty).unwrap_or((16, 24));
    let recv = super::eval::eval(object, b, ctx)?;
    let val = super::eval::eval(&args[0], b, ctx)?;
    let r = b.fresh();
    b.push_raw(Instr::LoadField {
        dst: r,
        base: recv,
        off: ref_off,
    });
    let len = b.fresh();
    b.push_raw(Instr::LoadField {
        dst: len,
        base: recv,
        off: len_off,
    });
    let i = b.fresh();
    let zero = b.const_i64(0);
    b.mov(i, zero);
    let cond_b = b.make_block();
    let body_b = b.make_block();
    let done_b = b.make_block();
    b.goto(cond_b);
    b.switch_to(body_b);
    let wv = b.const_i64(w as i64);
    let off = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Mul,
        a: i,
        b: wv,
    });
    let addr = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: r,
        b: off,
    });
    b.push_raw(Instr::StoreField {
        base: addr,
        off: 0,
        src: val,
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
        b: len,
    });
    b.terminate(Terminator::Branch {
        cond: Some(more),
        if_true: body_b,
        if_false: done_b,
    });
    b.switch_to(done_b);
    Ok(Some(b.const_i64(0)))
}

/// Inline `new Span<T>(array, start, length)` /
/// `new ReadOnlySpan<T>(array, start, length)` with the CALLER's
/// closed element width: `_reference = array + 16 + start*w`,
/// `_length = length`. The shared definition body folds its
/// `Unsafe.Add` with the OPEN type variable (width 8), misplacing
/// every narrow start (a span meant for `arr+19` pointed at
/// `arr+40` and read zeros). No range checks (matches the pack
/// precedent). Returns `None` — keep normal construction — unless
/// the array element type is concrete and the definition path
/// resolves.
pub(super) fn span_from_array_view(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    arr: VReg,
    start: VReg,
    len: VReg,
    elem_ty: &str,
    span_ty: &str,
) -> Result<Option<VReg>, LowerError> {
    // Open element variables keep the shared body: their width is
    // genuinely unknowable at the call site (the definition body
    // handles wide/open cases as before). `closed_elem_width`
    // (not `span_elem_width`) decides: the default-8 bucket cannot
    // distinguish an open `T` from a real 8-byte type, and inlining
    // with a defaulted width misplaces narrow spans exactly like
    // the shared body does.
    let Some(w) = closed_elem_width(elem_ty) else {
        return Ok(None);
    };
    let Some(path) = span_class_path(ctx, span_ty) else {
        return Ok(None);
    };
    let sixteen = b.const_i64(16);
    let base = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: arr,
        b: sixteen,
    });
    let wv = b.const_i64(w as i64);
    let off = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Mul,
        a: start,
        b: wv,
    });
    let data = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: base,
        b: off,
    });
    Ok(Some(span_object(b, ctx, &path, data, len)?))
}

/// Inline `span.Slice(start[, length])` with the CALLER's closed
/// element width (same open-`T` stride-8 root cause as
/// `span_from_array_view`: `Slice(1)` on a byte span advanced 8).
/// `None` (normal virtual dispatch) unless the receiver is a
/// `Span`/`ReadOnlySpan` with a concrete inner type and a
/// resolvable definition path. No range checks.
pub(super) fn span_slice_view(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    recv: VReg,
    start: VReg,
    len: Option<VReg>,
    recv_ty: &str,
) -> Result<Option<VReg>, LowerError> {
    let Some((_, inner)) = span_view(recv_ty) else {
        return Ok(None);
    };
    // Same closed-width rule as `span_from_array_view` (open `T`
    // keeps virtual dispatch).
    let Some(w) = closed_elem_width(&inner) else {
        return Ok(None);
    };
    let Some(path) = span_class_path(ctx, recv_ty) else {
        return Ok(None);
    };
    let (ref_off, len_off) = span_field_offsets(ctx, recv_ty).unwrap_or((16, 24));
    let r = b.fresh();
    b.push_raw(Instr::LoadField {
        dst: r,
        base: recv,
        off: ref_off,
    });
    let new_len = match len {
        Some(l) => l,
        None => {
            let cur = b.fresh();
            b.push_raw(Instr::LoadField {
                dst: cur,
                base: recv,
                off: len_off,
            });
            b.emit(Instr::Bin {
                dst: VReg(0),
                op: IrBinOp::Sub,
                a: cur,
                b: start,
            })
        }
    };
    let wv = b.const_i64(w as i64);
    let off = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Mul,
        a: start,
        b: wv,
    });
    let data = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: r,
        b: off,
    });
    Ok(Some(span_object(b, ctx, &path, data, new_len)?))
}

/// Pack an array (8-byte stride, data at +16) into a contiguous
/// span buffer, then wrap it as a span object of type `span_ty`.
pub(super) fn pack_span_from_array(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    arr: VReg,
    span_ty: &str,
    inner: &str,
) -> Result<VReg, LowerError> {
    let w = span_elem_width(inner);
    let len = b.fresh();
    b.push_raw(Instr::LoadField {
        dst: len,
        base: arr,
        off: 0,
    });
    let wv = b.const_i64(w as i64);
    let total = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Mul,
        a: len,
        b: wv,
    });
    // Tail slop (see `pack_span_from_string`).
    let slop = b.const_i64(8);
    let total = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: total,
        b: slop,
    });
    let data = b.fresh();
    b.push_raw(Instr::AllocDyn {
        dst: data,
        bytes: total,
    });
    let sixteen = b.const_i64(16);
    let src = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: arr,
        b: sixteen,
    });
    pack_loop(b, data, src, len, w, w, w);
    span_object(b, ctx, span_ty, data, len)
}

/// Pack a string (packed bytes at +16) into a span buffer of width
/// `w` (bytes zero-extend: the zero-slot store above runs first).
pub(super) fn pack_span_from_string(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    s: VReg,
    span_ty: &str,
    inner: &str,
) -> Result<VReg, LowerError> {
    let w = span_elem_width(inner);
    let len = b.fresh();
    b.push_raw(Instr::LoadField {
        dst: len,
        base: s,
        off: 0,
    });
    let wv = b.const_i64(w as i64);
    let total = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Mul,
        a: len,
        b: wv,
    });
    // Tail slop: the pack loop's 8-byte zero stores overrun the
    // last element by up to 7 bytes.
    let slop = b.const_i64(8);
    let total = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: total,
        b: slop,
    });
    let data = b.fresh();
    b.push_raw(Instr::AllocDyn {
        dst: data,
        bytes: total,
    });
    let sixteen = b.const_i64(16);
    let src = b.emit(Instr::Bin {
        dst: VReg(0),
        op: IrBinOp::Add,
        a: s,
        b: sixteen,
    });
    // Source units are UTF-16 chars (2 bytes each — all strings are
    // built by `build_utf16_string`); copy one code unit per element
    // into `w`-byte slots.
    pack_loop(b, data, src, len, w, 2, 2);
    span_object(b, ctx, span_ty, data, len)
}
