//! AST to IR body lowering.

use super::super::{Block, Instr, IrFunction, LowerError, Terminator, VReg};
use crate::models::{ClassMember, Expr, Literal};
use crate::native::ir::Ctx;
use crate::native::semantic::{ClassInfo, FieldInfo, MethodInfo, PropertyInfo, TypeTable};
use super::{body::*, select::*, types::*};
use super::eval::eval;
use super::span::{placeholder_row, vtable_row_for};

// Direct-call function index, cached per lowering run: building it
// scans every class and method, so rebuilding per call site is
// O(calls x methods) — fine for small units, hours for a full
// DataContractSerialization. The cache keys on the table address
// (stable across one run) and is cleared at each lowering entry, so
// sequential builds and parallel test threads never share entries.
thread_local! {
    static FN_INDEX_CACHE: std::cell::RefCell<
        Option<(
            usize,
            std::rc::Rc<std::collections::BTreeMap<String, Vec<usize>>>,
        )>,
    > = std::cell::RefCell::new(None);
}

/// Drop the cached index: called once per lowering entry, before any
/// call site resolves.
pub(crate) fn clear_fn_index_cache() {
    FN_INDEX_CACHE.with(|c| *c.borrow_mut() = None);
}/// Resolve a bare call against static local functions visible in
/// `ctx`'s scope: `(enclosing class, enclosing method-or-property,
/// name, arity)` with class matched by simple name (spellings
/// arrive short and qualified alike). Returns the function index,
/// or `None` (method lookup / legacy placeholder path).
fn resolve_local_fn(ctx: &Ctx<'_>, callee: &str, arity: usize) -> Option<usize> {
    // TEMP-LOCALDBG.
    if std::env::var("RUBYC_LOCALDBG").is_ok() {
        eprintln!(
            "[rubyc] TEMP-LOCALDBG call={callee} class={} method={:?} table={}",
            ctx.current_class,
            ctx.current_method,
            ctx.local_fns.len()
        );
    }
    let current_method = ctx.current_method.as_deref()?;
    let simple_class = ctx.current_class.rsplit('.').next().unwrap_or(ctx.current_class);
    ctx.local_fns.iter().find_map(|e| {
        let class_ok = e.0 == ctx.current_class || e.0 == simple_class;
        if class_ok && e.1 == current_method && e.2 == callee && e.3 == arity && e.4 != usize::MAX {
            Some(e.4)
        } else {
            None
        }
    })
}

/// Return type for a resolved local call (`None` = fall through).
pub(super) fn resolve_local_ret(ctx: &Ctx<'_>, callee: &str, arity: usize) -> Option<String> {
    let current_method = ctx.current_method.as_deref()?;
    let simple_class = ctx.current_class.rsplit('.').next().unwrap_or(ctx.current_class);
    ctx.local_fns.iter().find_map(|e| {
        let class_ok = e.0 == ctx.current_class || e.0 == simple_class;
        if class_ok && e.1 == current_method && e.2 == callee && e.3 == arity && e.4 != usize::MAX {
            Some(e.5.clone())
        } else {
            None
        }
    })
}

/// Whether a STRICT subclass of the receiver entry overrides this
/// exact method signature (`is_override`, not a `new` hide): a bare
/// `this` call inside a base body must then dispatch virtually
/// (Template Method), not run the base body directly (which threw
/// `NotImplementedException` from `Encoding.GetBytesFast` instead
/// of reaching `UTF8Encoding`'s override). `new`-hides never
/// trigger (only `is_override` counts), so non-virtual hides keep
/// C# direct-call semantics. Transitive (grandchild overrides
/// count); self excluded.
fn subclass_overrides(
    ctx: &Ctx<'_>,
    info: &ClassInfo,
    name: &str,
    param_types: &[String],
) -> bool {
    fn norm_ty(s: &str) -> String {
        let mut t = s.to_lowercase();
        for w in ["out ", "ref ", "in "] {
            if let Some(rest) = t.strip_prefix(w) {
                t = rest.to_string();
                break;
            }
        }
        t.retain(|c| !c.is_whitespace());
        t.trim_end_matches('?').trim_end_matches('&').to_string()
    }
    let want: Vec<String> = param_types.iter().map(|p| norm_ty(p)).collect();
    for sub in super::eval::virtual_subclasses(ctx, info) {
        if sub == info.name || sub == info.key.path {
            continue;
        }
        for class in ctx.classes.iter().filter(|c| c.name == sub) {
            for m in &class.members {
                if let ClassMember::Method(md) = m {
                    if md.name == name
                        && md.body.is_some()
                        && md.is_override
                        && md.params.len() == want.len()
                        && md
                            .params
                            .iter()
                            .zip(want.iter())
                            .all(|(p, t)| norm_ty(&p.type_name) == *t)
                    {
                        return true;
                    }
                }
            }
        }
    }
    false
}

pub(super) fn call(callee: &str, args: &[Expr], b: &mut Body<'_>, ctx: &Ctx<'_>) -> Result<VReg, LowerError> {
    if callee == "printf" || callee == "print" || callee == "put_str" {
        return printf(args, Fd::Stdout, b, ctx);
    }
    // Static local functions of the enclosing scope resolve first
    // (C# scoping prefers locals). Scope is (class, method); arity
    // must match exactly; anything else falls through to method
    // lookup (legacy). `usize::MAX` marks locals whose lowering
    // failed — also legacy.
    if let Some(fidx) = resolve_local_fn(ctx, callee, args.len()) {
        let argv = eval_args(args, b, ctx)?;
        let dst = b.fresh();
        b.push_raw(Instr::CallFn {
            dst,
            fidx: fidx as u32,
            args: argv,
        });
        return Ok(dst);
    }
    if callee == "eprintf" {
        return printf(args, Fd::Stderr, b, ctx);
    }
    if callee == "scan_int" {
        if !args.is_empty() {
            return Err(LowerError::UnsupportedConstruct(
                "scan_int takes no arguments".into(),
            ));
        }
        let dst = b.fresh();
        b.push_raw(Instr::ScanInt { dst });
        return Ok(dst);
    }
    // Bare same-class calls reaching `&local`+native random seeds
    // (`GenerateSeed()`, `GenerateGlobalSeed()`): the MethodCall-arm
    // intercepts only fire on qualified spellings, but initializers
    // call them bare (`DefaultSeed = GenerateSeed()`), which lands
    // here and would run the real body (`&seed` erases to null →
    // native write to address zero). Same fixed seeds, same
    // determinism rationale; see the MethodCall-arm notes.
    // Bare `InitializeCurrentThread()` inside `Thread` (called when
    // `t_currentThread` is null): the real body calls the
    // `ThreadNative` QCalls (`GetCurrentThread`, `Initialize`), which
    // smoke shims cannot provide (no managed thread objects exist yet
    // to hand out). Emit a hollow `Thread` instead (vtable + live tag
    // + zeroed fields, mirroring construction without running the ctor
    // body) and publish it to `t_currentThread`: enough for
    // `ExecutionContext`/`AsyncLocal`/culture paths that only need a
    // non-null thread whose fields read null/zero (e.g. file open via
    // `AppContext` switches). Bail (normal lowering) if anything is
    // missing — never half-build.
    if args.is_empty()
        && callee == "InitializeCurrentThread"
        && (ctx.current_class == "Thread" || ctx.current_class.ends_with(".Thread"))
    {
        if let Some(entry) = ctx.table.class("Thread") {
            if let Some(tfield) = entry.field("t_currentThread").cloned() {
                if tfield.is_static {
                    if let Some(sbase) = static_addr(&tfield, b, ctx) {
                        let row = vtable_row_for(ctx, &entry.name, &entry.key.path);
                        let obj = b.fresh();
                        b.alloc(obj, (entry.fields.len() + 2) as u64 * 8);
                        let vaddr = b.fresh();
                        b.push_raw(Instr::VTableAddr { dst: vaddr, row });
                        b.push_raw(Instr::StoreField { base: obj, off: 0, src: vaddr });
                        let one = b.const_i64(1);
                        b.push_raw(Instr::StoreField { base: obj, off: 8, src: one });
                        let zero = b.const_i64(0);
                        for i in 2..(entry.fields.len() + 2) {
                            b.push_raw(Instr::StoreField {
                                base: obj,
                                off: (i * 8) as u32,
                                src: zero,
                            });
                        }
                        b.push_raw(Instr::StoreField { base: sbase, off: 0, src: obj });
                        return Ok(obj);
                    }
                }
            }
        }
    }
    if args.is_empty() && (callee == "GenerateSeed" || callee == "GenerateGlobalSeed") {
        let cur = base_type_name(ctx.current_class);
        if (callee == "GenerateSeed" && cur == "Marvin")
            || (callee == "GenerateGlobalSeed" && cur == "HashCode")
        {
            return Ok(b.const_i64(if callee == "GenerateSeed" {
                0x9E3779B97F4A7C15u64 as i64
            } else {
                0x9E37_79B9
            }));
        }
    }
    // Bare same-class `[Intrinsic]` atomics (`ExchangeAdd(...)` inside
    // `Interlocked.Add`, and the self-recursive placeholder bodies
    // themselves): the MethodCall-arm emulation only fires on
    // qualified spellings, but these land here and would run the real
    // self-recursive body forever. Same emulation, same rationale;
    // see the MethodCall-arm notes.
    if base_type_name(ctx.current_class) == "Interlocked"
        && let Some(r) = super::eval::interlocked_atomic(callee, args, b, ctx)
    {
        return r;
    }
    // Bare same-class thin-lock primitives (mirrors the `Interlocked`
    // twin above): bodies that call back into `ObjectHeader` land here
    // instead of the qualified path. Family-gated the same way.
    if base_type_name(ctx.current_class).contains("ObjectHeader")
        && let Some(r) = super::eval::monitor_atomic(callee, args, b, ctx)
    {
        return r;
    }
    if callee == "sizeof" {        // C# unmanaged sizes (the argument is always a TYPE, never
        // a value — no evaluation). Correct sizes matter for
        // byte↔element conversions (`Latin1Utility` divides byte
        // counts by `sizeof(char)` = 2; the old flat 8 truncated
        // every 3-char scan to 0). Unknowns keep the word placeholder.
        if let Some(spelled) = args.first().and_then(dotted_spelling) {
            let simple = spelled.rsplit('.').next().unwrap_or(&spelled);
            let canon = canonical_primitive(simple.trim_end_matches(['?', '*']));
            let size: Option<i64> = match canon.as_str() {
                "Boolean" | "Byte" | "SByte" => Some(1),
                "Char" | "Int16" | "UInt16" => Some(2),
                "Int32" | "UInt32" | "Single" => Some(4),
                "Int64" | "UInt64" | "Double" | "IntPtr" | "UIntPtr" => Some(8),
                "Decimal" => Some(16),
                _ if simple.starts_with("Vector128<") => Some(16),
                _ if simple.starts_with("Vector256<") => Some(32),
                _ if simple.starts_with("Vector512<") => Some(64),
                _ if simple.starts_with("Vector<") => {
                    // Generic vector width is unknown without
                    // hardware acceleration details; callers guard
                    // on `IsHardwareAccelerated` (false here) before
                    // using it, so any consistent value works.
                    Some(16)
                }
                _ => None,
            };
            if let Some(s) = size {
                return Ok(b.const_i64(s));
            }
        }
        // Unknown/generic: word placeholder (a type name like
        // `int[]` is undeclared as a variable — no evaluation).
        return Ok(b.const_i64(8));
    }
    if callee == "typeof" {
        // `typeof` is a C# operator (type argument, no evaluation).
        // Return a vtable'd placeholder box for the named type — NOT
        // null: framework code assumes `typeof(X)` is non-null
        // (attribute tables, `VerifyPseudoCustomAttribute`, …) and
        // dereferences it. The box carries the synthetic placeholder
        // row (all zero stubs), never the named type's own: virtual
        // calls on placeholders have no implementation, and
        // dispatching X's row executes foreign slots or faults past
        // short rows (`CreateDefaultEqualityComparer` faulted calling
        // Type virtuals on a `typeof(object)` box). Every such call
        // now answers the C# default (zero/false/null) crash-free.
        // Storage still sizes by the named type's field count, so
        // field reads stay in bounds; identity comparisons between
        // two `typeof(X)` boxes stay false; reflection over it yields
        // defaults. Unresolvable names fall back to a small
        // `System.Object` box (same guarantee).
        if let Some(spelled) = args.first().and_then(dotted_spelling) {
            let simple = spelled.rsplit('.').next().unwrap_or(&spelled);
            let row = placeholder_row(ctx);
            // Storage: the type's field count (+vtable+refcount), so
            // field reads stay in bounds; default when unknown.
            let nfields = ctx
                .table
                .class(&spelled)
                .or_else(|| ctx.table.class(simple))
                .map(|e| e.fields.len())
                .unwrap_or(2);
            let obj = b.fresh();
            b.alloc(obj, (nfields + 2) as u64 * 8);
            let rv = b.fresh();
            b.push_raw(Instr::VTableAddr { dst: rv, row });
            b.push_raw(Instr::StoreField { base: obj, off: 0, src: rv });
            let z = b.const_i64(0);
            let one = b.const_i64(1);
            b.push_raw(Instr::StoreField { base: obj, off: 8, src: one });
            for i in 0..nfields {
                b.push_raw(Instr::StoreField {
                    base: obj,
                    off: (16 + i * 8) as u32,
                    src: z,
                });
            }
            return Ok(obj);
        }
        return Ok(b.const_i64(0));
    }
    if callee == "__reftype" || callee == "__makeref" || callee == "__refvalue" {
        // Undocumented TypedReference keywords (`__reftype(tr)` takes a
        // typed reference): evaluate args for side effects, yield zero
        // (mirrors the scan's known-list; real typing is future work).
        let _ = eval_args(args, b, ctx)?;
        return Ok(b.const_i64(0));
    }
    // Bare same-class ref-model intrinsics (mirror the qualified
    // interceptions in `eval.rs`): field initializers and method
    // bodies call `GenerateGlobalSeed()`, `As`/`AsRef`,
    // `Read`/`Write` unqualified, so they land here instead of the
    // `Unsafe.As(...)` path. Gated on the enclosing class's base
    // name, so unrelated same-named user methods are untouched.
    // (Generic suffixes stripped: `As<TFrom, TTo>` selects `As`.)
    {
        let bare = match callee.find('<') {
            Some(pos) => &callee[..pos],
            None => callee,
        };
        let current_base = ctx
            .current_class
            .rsplit('.')
            .next()
            .unwrap_or(ctx.current_class)
            .split('<')
            .next()
            .unwrap_or(ctx.current_class);
        // Deterministic hash seed (see `eval.rs`): address-of-local
        // erases to a value, so the native random fill would corrupt
        // arbitrary memory.
        if current_base == "HashCode" && bare == "GenerateGlobalSeed" && args.is_empty() {
            return Ok(b.const_i64(0x9E37_79B9));
        }
        // JIT-identity intrinsics (managed bodies throw
        // `PlatformNotSupportedException`; real `ldarg.0 / ret`
        // lives in the JIT): erased refs are values, so both are
        // the identity.
        if current_base == "Unsafe" && matches!(bare, "As" | "AsRef" | "AsPointer" | "BitCast") && args.len() == 1 {
            let argv = eval_args(args, b, ctx)?;
            return Ok(argv[0]);
        }
        // `RuntimeHelpers.GetMethodTable(obj)` (see `eval.rs`): the
        // intrinsic's own body calls itself bare, so the qualified
        // interception never fires for it — catch the bare form here
        // too (our vtable pointer at object+0). Arrays carry no
        // vtable word (length at +0): hand them the shared
        // `System.Array` row token (mirrors `eval.rs`), or
        // downstream `MethodTable*` dereferences load `[length]`
        // and fault.
        if current_base == "RuntimeHelpers" && bare == "GetMethodTable" && args.len() == 1 {
            let t0 = static_type_of(&args[0], b, ctx).unwrap_or_else(|_| "object".to_owned());
            let tr = t0.trim();
            if super::select::array_elem_ty(tr).is_some() || base_type_name(tr) == "Array" {
                let dst = b.fresh();
                b.push_raw(Instr::VTableAddr {
                    dst,
                    row: vtable_row_for(ctx, "Array", "System.Array"),
                });
                return Ok(dst);
            }
            let argv = eval_args(args, b, ctx)?;
            let dst = b.fresh();
            b.push_raw(Instr::LoadField {
                dst,
                base: argv[0],
                off: 0,
            });
            return Ok(dst);
        }
        // `RuntimeHelpers.IsReferenceOrContainsReferences<T>()` bare
        // (see `eval.rs`): the intrinsic's own body calls itself
        // bare, so the qualified interception never fires for it —
        // fold the same way from the explicit type argument (open
        // variables fold false — see `eval.rs`).
        if current_base == "RuntimeHelpers" && bare == "IsReferenceOrContainsReferences" && args.is_empty() {
            let t = explicit_type_args(callee).first().cloned().unwrap_or_else(|| "object".to_owned());
            let r = !ctx.is_type_var(&t) && type_contains_refs(ctx.table, &t);
            return Ok(b.const_i64(r as i64));
        }
        // `MemoryMarshal.GetArrayDataReference` bare (see `eval.rs`):
        // the generic forwarder calls the overload bare, which
        // overload resolution maps back to itself (exact `T[]`
        // match beats the `Array` conversion) — infinite recursion.
        // Array data starts at +16 in our layout.
        if current_base == "MemoryMarshal" && bare == "GetArrayDataReference" && args.len() == 1 {
            let argv = eval_args(args, b, ctx)?;
            let sixteen = b.const_i64(16);
            return Ok(b.emit(Instr::Bin {
                dst: VReg(0),
                op: crate::native::ir::BinOp::Add,
                a: argv[0],
                b: sixteen,
            }));
        }
        // `Thread.FastPollGC()` bare (see `eval.rs`): the
        // intrinsic's own body calls itself bare — erase to no-op.
        if current_base == "Thread" && bare == "FastPollGC" && args.is_empty() {
            return Ok(b.const_i64(0));
        }
        // `Interlocked.MemoryBarrier()` bare (see `eval.rs`): the
        // intrinsic's own body calls itself bare — erase to no-op.
        if current_base == "Interlocked" && bare == "MemoryBarrier" && args.is_empty() {
            return Ok(b.const_i64(0));
        }
        // `Unsafe.IsNullRef(ref x)` (see `eval.rs`): a ref to any
        // variable is never a null pointer — yield false constantly.
        if current_base == "Unsafe" && bare == "IsNullRef" && args.len() == 1 {
            let _ = eval_args(args, b, ctx)?;
            return Ok(b.const_i64(0));
        }
        // `GenericCache.CreateCacheTable` (see `eval.rs`): instance
        // bare call from the cache ctor — same bypass (raw memory
        // reinterpretation unimplementable, cache unusable anyway).
        if current_base == "GenericCache" && bare == "CreateCacheTable" {
            let _ = eval_args(args, b, ctx)?;
            return Ok(b.const_i64(0));
        }
        // Volatile wrappers over `Unsafe.As` (see `eval.rs`): a
        // volatile read is the value, a volatile write is an
        // assignment.
        if current_base == "Volatile" && bare == "Read" && args.len() == 1 {
            let argv = eval_args(args, b, ctx)?;
            return Ok(argv[0]);
        }
        if current_base == "Volatile" && bare == "Write" && args.len() == 2 {
            let assign = Expr::Assign {
                target: Box::new(args[0].clone()),
                value: Box::new(args[1].clone()),
                span: crate::models::Span::new(0, 0),
            };
            return eval(&assign, b, ctx);
        }
    }

    // Strip explicit generic type arguments for name-based lookups
    // below (`M<T>(...)` selects overload `M`; the args still drive
    // generic inference in ranking, and `static_type_of` substitutes
    // them into result types). The full spelling stays in `callee`
    // for the error message. Without this, generic bare calls never
    // match (`call to 'TryParse...(....)'`).
    let name = match callee.find('<') {
        Some(pos) => &callee[..pos],
        None => callee,
    };
    // Explicit generic arity constrains overload ranking below
    // (`services.AddOptions<NoOp>()` must not elect the non-generic
    // overload); plain names leave selection untouched.
    let explicit_arity = match callee.find('<') {
        Some(_) => Some(explicit_type_args(callee).len()),
        None => None,
    };

    // Import stubs win over same-class methods (a stub has no body, so it
    // can never shadow a real implementation of the same name) — but
    // only at matching arity (see the member-call path).
    if let Some(index) = ctx.import_of(name)
        && ctx
            .table
            .imports
            .get(index)
            .is_some_and(|imp| imp.param_types.len() == args.len())
    {
        check_import(ctx, index, args)?;
        let mut argv = eval_args(args, b, ctx)?;
        // Native C ABI (see `marshal_string_utf8_args` /
        // `wrap_safehandle_import_return`): strings transcode,
        // SafeHandle returns wrap. Bare-name args typing mirrors the
        // member-call path (`object` fallback degrades to untouched).
        let arg_types: Vec<String> = args
            .iter()
            .map(|a| static_type_of(a, b, ctx).unwrap_or_else(|_| "object".to_owned()))
            .collect();
        super::eval::marshal_string_utf8_args(b, &mut argv, &arg_types);
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
        return Ok(super::eval::wrap_safehandle_import_return(
            b, ctx, dst, &ret_ty,
        ));
    }

    // Search-optimization predicates (see `span.rs`): safe-false
    // bypass BEFORE selection, so generic/inferred calls (which may
    // not resolve to a `selected` overload) are still neutralized.
    // Bare same-class calls (`TryGetSingleRange` inside `Create`)
    // arrive here; qualified ones are handled in `eval.rs`.
    // Generic suffixes stripped (`TryGetSingleRange<T>` matches).
    let bare_name = match name.find('<') {
        Some(pos) => &name[..pos],
        None => name,
    };
    if matches!(
        bare_name,
        "TryGetSingleRange"
            | "CanUseUniqueLowNibbleSearch"
            | "CanUsePackedIndexOf"
            | "ShouldUseProbabilisticMap"
    ) {
        let _ = eval_args(args, b, ctx)?;
        return Ok(b.const_i64(0));
    }
    // Typed overload selection (see `pick_overload`): arg types are
    // pure to compute (no IR emitted), so selection precedes eval.
    let arg_types: Vec<String> = args
        .iter()
        .map(|a| {
            if is_method_group(a, b, ctx) {
                return "method-group".to_owned();
            }
            static_type_of(a, b, ctx).unwrap_or_else(|_| "object".to_owned())
        })
        .collect();
    // Typed overload selection across the method-anchored entry,
    // first-match, and same-named siblings (`WhenAllPromise<T>`
    // lowers under the simple name `WhenAllPromise`): visibility
    // preferred throughout (see `select_visible`). Methods only
    // fail-open here — a genuine unknown still errors below.
    let mut entries: Vec<&ClassInfo> = Vec::new();
    if let Some(cur) = ctx.current_info {
        entries.push(cur);
    }
    entries.extend(entries_for(ctx.table, ctx.current_class));
    let selected = select_visible(
        ctx.table,
        &entries,
        name,
        &arg_types,
        None,
        ctx.current_class,
        explicit_arity,
    );
    if let Some(mi) = selected {
        // TEMP-CALLDIAG: audit bare-call selection.
        // Removed before close.
        if std::env::var("RUBYC_CALLDIAG").is_ok()
            && (name == "InitializeUserDefaultUICulture" || name == "InitializeUserDefaultCulture")
        {
            eprintln!(
                "[rubyc] TEMP-CALLDIAG {name} static={} declaring={} in {}.{}",
                mi.is_static,
                mi.declaring_class,
                ctx.current_class,
                ctx.current_method.clone().unwrap_or_default()
            );
        }
        // TEMP-CALLDIAG2: audit GetBytesWithFallback bare-call choice.
        // Removed before close.
        if std::env::var("RUBYC_SELDIAG").is_ok() && name == "GetBytesWithFallback" {
            eprintln!(
                "[rubyc] TEMP-CALLDIAG2 args={arg_types:?} selected={}.{}({:?}) in {}.{}",
                mi.declaring_class,
                mi.name,
                mi.param_types,
                ctx.current_class,
                ctx.current_method.clone().unwrap_or_default()
            );
        }
        // TEMP-GBDIAG: audit GetBytes overload choice (encoder path
        // misroutes to an array overload, reading Length off a
        // pointer). Removed before close.
        if std::env::var("RUBYC_GBDIAG").is_ok() && name == "GetBytes" {
            eprintln!(
                "[rubyc] TEMP-GBDIAG args={arg_types:?} selected={}.{}({:?}) in {}.{}",
                mi.declaring_class,
                mi.name,
                mi.param_types,
                ctx.current_class,
                ctx.current_method.clone().unwrap_or_default()
            );
        }
        // `[Conditional("SYM")]` (e.g. `Debug.Assert`,
        // `VerifyPseudoCustomAttribute`): the call — including
        // argument evaluation — vanishes unless SYM is defined.
        // Mirrors the qualified path (which checks both the set and
        // the winner); bare same-class calls are the common shape
        // for these helpers, so omitting it here executes
        // debug-only code in release (e.g. reflection over
        // `typeof(...)` placeholders that are null at runtime).
        if mi
            .conditional_symbol
            .as_ref()
            .is_some_and(|s| !ctx.defines.contains(s))
        {
            return Ok(b.const_i64(0));
        }
        // Argument count mismatch is tolerated — generic methods and
        // overloads may not match exactly; omitted optional params are
        // filled from the selected overload's folded defaults.
        // Plain call inside the class: may still resolve to an inherited
        // member whose visibility must be checked against this class —
        // enforced on the SELECTED overload (see above).
        ctx.check_method_access(mi)?;

        let this = VReg(0);
        // Fill omitted optionals on the EXPLICIT args (`this` is not a
        // declared param), then prepend the receiver — but only for
        // instance callees. A static callee takes no receiver (its
        // vreg 0 is the first param, not a phantom `this`): prepending
        // one shifts every arg into garbage (the first param reads
        // zero, so guards like `x == null` spuriously fire and cold
        // throws go off). Static-to-static same-class calls hit this
        // path (`Get(n)` inside `Try`), so the receiver must be
        // conditional here, not unconditional.
        let mut explicit = eval_args(args, b, ctx)?;
        let full_arity = mi.param_types.len();
        fill_default_args(&mut explicit, mi, b);
        // Box by-reference arguments BEFORE prepending the receiver
        // (flags align with explicit params). See `box_byref_args`.
        let wbs = super::eval::box_byref_args(b, ctx, args, &mut explicit, &mi.param_is_byref)?;
        // Span-typed parameters pack array/string arguments into
        // views (see `span.rs`); anything else passes through
        // untouched. Missing here faulted
        // `StreamWriter.WriteLine(string?)`: the bare-call path
        // passed the string object as a span, whose chars then read
        // as a data pointer in `Buffer.MemoryCopy`.
        super::span::coerce_span_args(b, ctx, &mut explicit, &arg_types, &mi.param_types)?;
        super::eval::marshal_safehandle_args(b, ctx, &mut explicit, &arg_types, &mi.param_types, !mi.has_body)?;
        // `CallFn` bodies take `this` as their first parameter, so
        // instance calls prepend it; `CallVirt` carries the receiver
        // separately and takes explicit args only (prepending there
        // shifts every parameter into garbage — e.g. a bool landing
        // in an object slot and faulting as a null receiver).
        let mut argv = if mi.is_static {
            explicit.clone()
        } else {
            let mut v = vec![this];
            v.extend(explicit.iter().copied());
            v
        };

        let own = ctx
            .classes
            .iter()
            .find(|c| {
                // Qualification-tolerant: method contexts carry the
                // AST decl name while inlined property contexts carry
                // the qualified key path — both denote the same class.
                // A strict compare missed own methods from inlined
                // getters (falling through to virtual dispatch below).
                c.name == ctx.current_class
                    || c.name
                        == ctx.current_class.rsplit('.').next().unwrap_or(ctx.current_class)
            })
            .map(|c| {
                c.members
                    .iter()
                    .any(|m| matches!(m, ClassMember::Method(m) if m.name == name))
            })
            .unwrap_or(false);

        let dst = b.fresh();
        // Bodyless own method (`abstract`, `extern`, partial-no-impl
        // declared here): no body exists to call directly. INSTANCE
        // abstract methods dispatch virtually so overrides serve
        // (e.g. abstract base `GetByteCount(char[],int,int)` must
        // reach Latin1's override; yielding zero broke every base
        // count path). Static bodyless methods (`extern`, QCalls)
        // are never virtual — they fall through to the static path
        // below (direct, else import, else zero).
        // The slot is signature-aware: name-only `canonical_slot`
        // lands on the wrong overload's slot.
        if own && !mi.has_body && !mi.is_static {
            let slot = ctx
                .table
                .classes
                .iter()
                .find(|c| {
                    c.name == mi.declaring_class || c.key.path == mi.declaring_class
                })
                .and_then(|e| {
                    e.methods.iter().find(|m| {
                        m.name == mi.name && m.param_types == mi.param_types
                    })
                })
                .map(|m| m.vtable_slot)
                .unwrap_or_else(|| {
                    canonical_slot(ctx.table, ctx.current_info, ctx.current_class, name)
                });
            b.push_raw(Instr::CallVirt {
                dst,
                slot: slot as u32,
                recv: this,
                args: explicit,
            });
            super::eval::emit_writebacks(b, &wbs);
            return Ok(dst);
        }
        if own && mi.has_body {
            // Template-method dispatch: a bare `this` call inside a
            // base body must reach subclass overrides (C# virtual
            // dispatch). When a strict subclass overrides this exact
            // signature, dispatch virtually through the
            // signature-aware slot (same lookup as the abstract-own
            // path); otherwise the direct call below stays exact.
            // `new`-hides never trigger (only `is_override` counts).
            if !mi.is_static
                && ctx.current_info.is_some_and(|info| {
                    subclass_overrides(ctx, info, name, &mi.param_types)
                })
            {
                let vslot = ctx
                    .table
                    .classes
                    .iter()
                    .find(|c| {
                        c.name == mi.declaring_class || c.key.path == mi.declaring_class
                    })
                    .and_then(|e| {
                        e.methods.iter().find(|m| {
                            m.name == mi.name && m.param_types == mi.param_types
                        })
                    })
                    .map(|m| m.vtable_slot)
                    .unwrap_or_else(|| {
                        canonical_slot(ctx.table, ctx.current_info, ctx.current_class, name)
                    });
                b.push_raw(Instr::CallVirt {
                    dst,
                    slot: vslot as u32,
                    recv: this,
                    args: explicit,
                });
                super::eval::emit_writebacks(b, &wbs);
                return Ok(dst);
            }
            // Function indices mirror declaration order with @start at 0.
            // The selected overload's full arity keys the lookup (omitted
            // optionals were filled above); its position among
            // same-name+arity overloads picks the slot, so overloads
            // stop collapsing onto one body. A miss falls back to the
            // first declared overload.
            // Key on the SELECTED overload's declaring class, not the
            // current class: inlined contexts carry the qualified key
            // path while function keys use the AST decl name — keying
            // on the current class missed and the range fallback below
            // elected an alphabetically-adjacent WRONG function (e.g.
            // `InitializeUserDefaultUICulture` calling
            // `SystemException#0`).
            let prefix = format!("{}::{name}#", mi.declaring_class);
            // (Key folded into the Nth lookup below.)
            let n = overload_position(
                ctx,
                &mi.declaring_class,
                name,
                full_arity,
                &mi.param_types,
                &mi.param_is_byref,
            );
            let idx = function_index_nth(ctx, &mi.declaring_class, name, full_arity, n)
                .or_else(|| {
                    functions_index(ctx)
                        .range(prefix..)
                        .next()
                        .and_then(|(_, v)| v.last())
                        .copied()
                        .map(|i| i as u32)
                })
                .unwrap_or(0);
            b.push_raw(Instr::CallFn {
                dst,
                fidx: idx as u32,
                args: argv,
            });
        } else {
            // TEMP-CALLDIAG2: audit bare-to-virtual fallback.
            // Removed before close.
            if std::env::var("RUBYC_CALLDIAG").is_ok() {
                eprintln!(
                    "[rubyc] TEMP-CALLDIAG2 bare-virt {name} declaring={} static={} params={:?} in {}.{}",
                    mi.declaring_class,
                    mi.is_static,
                    mi.param_types,
                    ctx.current_class,
                    ctx.current_method.clone().unwrap_or_default()
                );
            }
            // Static callees are never virtual, even when the
            // `own` check below fails (it compares the current
            // class by bare name, but inlined contexts carry the
            // qualified path — so own-method statics like
            // `InitializeUserDefaultUICulture` fell through to a
            // virtual call on the garbage `VReg(0)` receiver).
            // Resolve direct via the declaring class instead.
            if mi.is_static {
                match static_function_index_for(
                    ctx,
                    &mi.declaring_class,
                    name,
                    full_arity,
                    &mi.param_types,
                    &mi.param_is_byref,
                ) {
                    Some(fidx) => {
                        b.push_raw(Instr::CallStatic {
                            dst,
                            fidx,
                            args: argv,
                        });
                        super::eval::emit_writebacks(b, &wbs);
                        return Ok(dst);
                    }
                    None => {
                        return Ok(b.const_i64(0));
                    }
                }
            }
            // Instance: virtual dispatch with a signature-aware slot
            // (same lookup as the abstract-own path above — the
            // name-only fallback lands on the wrong overload).
            let vslot = ctx
                .table
                .classes
                .iter()
                .find(|c| {
                    c.name == mi.declaring_class || c.key.path == mi.declaring_class
                })
                .and_then(|e| {
                    e.methods.iter().find(|m| {
                        m.name == mi.name && m.param_types == mi.param_types
                    })
                })
                .map(|m| m.vtable_slot)
                .unwrap_or_else(|| {
                    canonical_slot(ctx.table, ctx.current_info, ctx.current_class, name)
                });
            b.push_raw(Instr::CallVirt {
                dst,
                slot: vslot as u32,
                recv: this,
                args: explicit,
            });
        }
        super::eval::emit_writebacks(b, &wbs);
        return Ok(dst);
    }

    // If the callee is a local variable (e.g. a delegate parameter), emit a
    // placeholder call that returns 0. Delegates are not fully wired into the
    // runtime yet, but this lets compilation succeed.
    if b.scopes.lookup(name).is_some() {
        return Ok(b.const_i64(0));
    }
    // Field/property-held delegate invocation (`m_action()` where
    // `m_action` is an `Action` field, or `InternalSetNameMethod()`
    // where that's a delegate-typed property): same placeholder —
    // invocation runs nothing.
    if this_info(ctx, name)
        .is_some_and(|info| info.field(name).is_some() || info.property(name).is_some())
    {
        return Ok(b.const_i64(0));
    }
    // Enclosing-scope statics: unqualified calls from nested code
    // resolve outward (`AlternateLookup` ctor calling outer
    // `IsCompatibleKey(...)`). Instance members need an instance and
    // are never candidates; the resolved static lowers direct
    // (`CallStatic`, no receiver).
    if let Some(mi) = outer_static(ctx, name, &arg_types) {
        ctx.check_method_access(mi)?;
        let mut argv = eval_args(args, b, ctx)?;
        fill_default_args(&mut argv, mi, b);
        if !mi.has_body {
            return Ok(b.const_i64(0));
        }
        let full = mi.param_types.len();
        let dst = b.fresh();
        match static_function_index_for(ctx, &mi.declaring_class, name, full, &mi.param_types, &mi.param_is_byref) {
            Some(fidx) => {
                b.push_raw(Instr::CallStatic {
                    dst,
                    fidx,
                    args: argv,
                });
                return Ok(dst);
            }
            None => {
                // Cross-image call (dynamic-linking consume): the
                // callee lives in another image — import its body
                // instead of yielding zero.
                if let Some(entry) =
                    resolve_call_target(ctx.table, &mi.declaring_class, name, full)
                    && let Some((lib, sym)) = external_target(ctx, entry, name, full)
                {
                    let idx = dep_import_index(ctx, &lib, &sym);
                    b.push_raw(Instr::CallImport {
                        dst,
                        import: idx as u32,
                        args: argv,
                    });
                    return Ok(dst);
                }
                return Ok(b.const_i64(0));
            }
        }
    }
    // Global static fallback (`using static` imports): last resort
    // before erroring, same direct lowering as above.
    if let Some(mi) = global_static(ctx, name, &arg_types) {
        ctx.check_method_access(mi)?;
        let mut argv = eval_args(args, b, ctx)?;
        fill_default_args(&mut argv, mi, b);
        if !mi.has_body {
            return Ok(b.const_i64(0));
        }
        let full = mi.param_types.len();
        let dst = b.fresh();
        match static_function_index_for(ctx, &mi.declaring_class, name, full, &mi.param_types, &mi.param_is_byref) {
            Some(fidx) => {
                b.push_raw(Instr::CallStatic {
                    dst,
                    fidx,
                    args: argv,
                });
                return Ok(dst);
            }
            None => {
                // Cross-image call (dynamic-linking consume): the
                // callee lives in another image — import its body
                // instead of yielding zero.
                if let Some(entry) =
                    resolve_call_target(ctx.table, &mi.declaring_class, name, full)
                    && let Some((lib, sym)) = external_target(ctx, entry, name, full)
                {
                    let idx = dep_import_index(ctx, &lib, &sym);
                    b.push_raw(Instr::CallImport {
                        dst,
                        import: idx as u32,
                        args: argv,
                    });
                    return Ok(dst);
                }
                return Ok(b.const_i64(0));
            }
        }
    }
    Err({
        // Resolution context (stderr only — always fails the build):
        // callee plus arg types, so the next wall diagnoses from one
        // log without probe cycles.
        eprintln!("[rubyc] unknown call '{callee}' with args {arg_types:?}");
        LowerError::UnsupportedConstruct(crate::tf!(
            "rubyc.native.ir.lower_call.unknown_call",
            callee.to_owned()
        ))
    })
}

/// Global static fallback for unqualified calls (`using static`
/// imports, e.g. `RotateLeft(...)` via `using static
/// System.Numerics.BitOperations`): any static method unit-wide with
/// a compatible overload. Last resort by design — C# would reject
/// genuinely ambiguous same-named statics, so valid input never
/// depends on which one wins; declaration order breaks ties.
pub(super) fn global_static<'a>(
    ctx: &Ctx<'a>,
    callee: &str,
    arg_types: &[String],
) -> Option<&'a MethodInfo> {
    let mut best: Option<(&MethodInfo, (u32, usize))> = None;
    for c in &ctx.table.classes {
        // Ranked applicable first (exact, then defaults).
        let cands: Vec<&MethodInfo> = c
            .methods
            .iter()
            .filter(|m| m.name == callee && m.is_static)
            .collect();
        // (Reuse rank_candidates directly: exact-then-defaults applied
        // inside via score tiers.)
        if let Some(m) = rank_candidates(&cands, arg_types, 0, ctx.table, &c.generic_params) {
            // Prefer better (lower-score, fewer-params) across classes.
            let key = (
                arg_types
                    .iter()
                    .enumerate()
                    .map(|(i, a)| {
                        compat_score(a, &m.param_types[i], ctx.table, &m.generic_params)
                            .unwrap_or(u32::MAX)
                    })
                    .sum::<u32>(),
                m.param_types.len(),
            );
            let better = match &best {
                None => true,
                Some((_, bk)) => key < *bk,
            };
            if better {
                best = Some((m, key));
            }
        }
    }
    best.map(|(m, _)| m)
}

/// Enclosing-scope static field lookup for unqualified reads from
/// nested code (`s_utcTimeZone` inside `CachedData`, declared on the
/// outer `TimeZoneInfo`). Instance fields need an instance and are
/// never candidates; `const` fields need no storage and fold at the
/// use site. Mirrors `outer_static` (methods).
pub(super) fn outer_static_field<'t>(
    ctx: &'t Ctx<'_>,
    name: &str,
) -> Option<(&'t ClassInfo, &'t FieldInfo)> {
    // Enclosing paths, innermost first, deduped (same walk as
    // `outer_static`).
    let mut outers: Vec<String> = Vec::new();
    for c in ctx
        .table
        .classes
        .iter()
        .filter(|c| c.name == ctx.current_class)
    {
        let mut prefix = c.key.path.clone();
        while let Some(i) = prefix.rfind('.') {
            prefix.truncate(i);
            if !outers.iter().any(|o| *o == prefix) {
                outers.push(prefix.clone());
            }
        }
    }
    for outer in &outers {
        for c in ctx
            .table
            .classes
            .iter()
            .filter(|c| c.key.path == *outer)
        {
            if let Some(f) = c.fields.iter().find(|f| f.name == name && f.is_static) {
                return Some((c, f));
            }
            // `const` fields occupy no storage but are visible to nested
            // code like statics (a nested class reads enclosing privates);
            // they live in `const_fields`, not `fields`. The use site
            // folds them to literals.
            if let Some(f) = c.const_fields.iter().find(|f| f.name == name) {
                return Some((c, f));
            }
            if let Some(f) = c
                .static_fields
                .iter()
                .find(|f| f.name == name)
            {
                return Some((c, f));
            }
        }
    }
    None
}

/// Enclosing-scope static property lookup for unqualified reads
/// from nested code (`HighCharTable` inside `IcuSearchValues`,
/// declared on the outer `CompareInfo`). The missing case: fields,
/// consts (above), and methods (`outer_static` below) each have an
/// outward walk, but properties did not — so the read fell past to
/// the uppercase placeholder (const 0) and downstream span code
/// faulted on null. Instance properties need an instance and are
/// never candidates, mirroring fields.
pub(super) fn outer_static_property<'t>(
    ctx: &'t Ctx<'_>,
    name: &str,
) -> Option<(&'t ClassInfo, &'t PropertyInfo)> {
    // Enclosing paths, innermost first, deduped (same walk as
    // `outer_static_field`, plus a qualified-`current_class` match:
    // method contexts sometimes carry the full path).
    let mut outers: Vec<String> = Vec::new();
    for c in ctx
        .table
        .classes
        .iter()
        .filter(|c| c.name == ctx.current_class || c.key.path == ctx.current_class)
    {
        let mut prefix = c.key.path.clone();
        while let Some(i) = prefix.rfind('.') {
            prefix.truncate(i);
            if !outers.iter().any(|o| *o == prefix) {
                outers.push(prefix.clone());
            }
        }
    }
    for outer in &outers {
        for c in ctx
            .table
            .classes
            .iter()
            .filter(|c| c.key.path == *outer)
        {
            if let Some(p) = c
                .properties
                .iter()
                .find(|p| p.name == name && p.is_static)
            {
                return Some((c, p));
            }
        }
    }
    None
}
/// code: every table class whose qualified path is a proper prefix of
/// a same-named current-class entry's path is an enclosing scope
/// (innermost first); its STATIC methods participate (instance
/// members need an instance). Ranked overload pick per scope; first
/// scope with a hit wins (mirrors C# outward lookup). Namespace
/// prefixes never match a class entry, so they self-exclude.
pub(super) fn outer_static<'a>(
    ctx: &Ctx<'a>,
    callee: &str,
    arg_types: &[String],
) -> Option<&'a MethodInfo> {
    // Enclosing paths, innermost first, deduped.
    let mut outers: Vec<String> = Vec::new();
    for c in ctx
        .table
        .classes
        .iter()
        .filter(|c| c.name == ctx.current_class)
    {
        let mut prefix = c.key.path.clone();
        while let Some(i) = prefix.rfind('.') {
            prefix.truncate(i);
            if !outers.iter().any(|o| *o == prefix) {
                outers.push(prefix.clone());
            }
        }
    }
    for outer in &outers {
        for c in ctx
            .table
            .classes
            .iter()
            .filter(|c| c.key.path == *outer)
        {
            // Explicit generic arity applies here too (same rule as
            // everywhere: `M<X>` only considers 1-param generics).
            let arity = match callee.find('<') {
                Some(_) => Some(explicit_type_args(callee).len()),
                None => None,
            };
            if let Some(mi) =
                pick_overload(c, callee, arg_types, ctx.table, Some(true), Some(ctx.current_class), arity)
            {
                return Some(mi);
            }
        }
    }
    None
}
pub(super) fn functions_index(
    ctx: &Ctx<'_>,
) -> std::rc::Rc<std::collections::BTreeMap<String, Vec<usize>>> {
    // Index MUST mirror the emission loop below (same classes, same
    // skips, same order, same +1): direct-call lookups resolve to
    // positions in the final functions vector. Same-name+arity
    // overloads append in declaration order (NOT last-wins):
    // overload-aware call sites pick their selected overload's slot
    // instead of every same-arity call collapsing onto one body.
    let key = ctx.table as *const TypeTable as usize;
    if let Some((cached_key, map)) = FN_INDEX_CACHE.with(|c| c.borrow().clone()) {
        if cached_key == key {
            return map;
        }
    }
    let mut built: std::collections::BTreeMap<String, Vec<usize>> =
        std::collections::BTreeMap::new();
    // Static local functions emit first (indices 1..=L); methods
    // follow. Offset by the same shared count the emitter uses, so
    // positions agree (see `collect_static_locals`).
    let mut next =
        1usize + crate::native::ir::collect_static_locals(ctx.classes).len();
    for class in ctx.classes {
        if ctx.table.class(&class.name).is_none() {
            continue;
        }
        for m in &class.members {
            if let ClassMember::Method(m) = m {
                // Mirror the lowering skip rules (bodyless, destructors
                // never become functions, and neither do >64-param
                // methods beyond the frame budget).
                if m.body.is_none() || m.name.starts_with('~') || m.params.len() > 64 {
                    continue;
                }
                let fname = format!("{}::{}#{}", class.name, m.name, m.params.len());
                built.entry(fname).or_default().push(next);
                next += 1;
            }
        }
    }
    let map = std::rc::Rc::new(built);
    FN_INDEX_CACHE.with(|c| *c.borrow_mut() = Some((key, map.clone())));
    map
}

/// Position of the selected overload among its declaring entry's
/// same-name+arity methods (declaration order), for indexing into
/// the per-key function list. Falls back to the last slot (legacy
/// last-wins) when the overload is not found there (inherited or
/// synthesized entries).
pub(super) fn overload_position(
    ctx: &Ctx<'_>,
    declaring_class: &str,
    name: &str,
    arity: usize,
    param_types: &[String],
    byref: &[bool],
) -> usize {
    // Declaration order (ClassDecl members), NOT table order: this
    // must mirror `functions_index` (same classes, same skips, same
    // order), or same-arity overloads misroute (Unicode
    // `GetByteCount#3`: table lists the `char*` overload first, but
    // lowering emits the `char[]` one first — the old table-order
    // position elected the wrong body for every direct call).
    // Declaration spellings drift from table spellings (`String`
    // vs `string`), so compare canonicalized; fall back to the
    // legacy last-wins when nothing matches.
    fn canon_sig_ty(s: &str) -> String {
        s.to_lowercase().split_whitespace().collect::<String>()
    }
    let query: Vec<String> = param_types.iter().map(|t| canon_sig_ty(t)).collect();
    let short = declaring_class
        .rsplit('.')
        .next()
        .unwrap_or(declaring_class);
    let mut positions: usize = 0;
    let mut matched: Option<usize> = None;
    let mut found_class = false;
    for class in ctx.classes {
        if class.name != declaring_class && class.name != short {
            continue;
        }
        // Same skip as `functions_index` (entry-less classes emit
        // no functions).
        if ctx.table.class(&class.name).is_none() {
            continue;
        }
        found_class = true;
        for m in &class.members {
            let crate::models::ClassMember::Method(md) = m else {
                continue;
            };
            if md.name != name || md.params.len() != arity {
                continue;
            }
            // Same skips as `functions_index` (bodyless, dtors,
            // >64 params never become functions).
            if md.body.is_none() || md.name.starts_with('~') || md.params.len() > 64 {
                continue;
            }
            if matched.is_none() {
                let decl_sig: Vec<String> = md
                    .params
                    .iter()
                    .map(|p| canon_sig_ty(&p.type_name))
                    .collect();
                // Table spellings may carry `?`/`[]`/qualifiers the
                // decl spells loosely — compare both raw-normalized
                // and with pointer/array markers intact (they
                // distinguish `char*` from `char[]`). Byref-ness joins
                // the signature: value-vs-`ref` twins (e.g.
                // `WriteAtOffset` 3/3) otherwise collapse to the
                // first-declared body and the forwarder recurses into
                // itself. Missing query flags (synthetic call sites)
                // behave legacy (first exact match).
                let decl_byref: Vec<bool> = md.params.iter().map(|p| p.is_byref).collect();
                let byref_ok = decl_sig.len() == query.len()
                    && decl_byref
                        .iter()
                        .enumerate()
                        .all(|(i, b)| *b == byref.get(i).copied().unwrap_or(false));
                // (Table spellings may carry `?`/`[]`/qualifiers the
                // decl spells loosely; pointer/array markers stay
                // intact — they distinguish `char*` from `char[]`.)
                if byref_ok && decl_sig == query {
                    matched = Some(positions);
                }
            }
            positions += 1;
        }
    }
    if found_class {
        return matched.unwrap_or_else(|| positions.saturating_sub(1));
    }
    // No ClassDecl (synthesized/inherited entry): legacy
    // table-order behavior, plus the byref signature (value-vs-`ref`
    // twins share erased param types — see above).
    let mut tpositions: Vec<usize> = Vec::new();
    let mut tmatched: Option<usize> = None;
    if let Some(entry) = ctx.table.class(declaring_class) {
        for m in &entry.methods {
            if m.name != name || m.param_types.len() != arity {
                continue;
            }
            let byref_ok = m.param_types.len() == param_types.len()
                && m
                    .param_is_byref
                    .iter()
                    .enumerate()
                    .all(|(i, b)| *b == byref.get(i).copied().unwrap_or(false));
            if tmatched.is_none() && byref_ok && m.param_types == param_types {
                tmatched = Some(tpositions.len());
            }
            tpositions.push(0);
        }
    }
    match tmatched {
        Some(n) => n,
        None => tpositions.len().saturating_sub(1),
    }
}

/// Index of the direct function for `class.name#arity`, if it
/// exists: the LAST same-arity overload (legacy last-wins, kept
/// for arity-only callers like base-ctor chains).
/// Used for static-style calls (`C.M(...)`) which are lowered as a direct
/// `CallFn` (no vtable, no receiver).
pub(super) fn function_index_for(
    ctx: &Ctx<'_>,
    class_name: &str,
    method: &str,
    arity: usize,
) -> Option<u32> {
    let fname = format!("{class_name}::{method}#{arity}");
    functions_index(ctx)
        .get(&fname)
        .and_then(|v| v.last())
        .copied()
        .map(|i| i as u32)
}

/// Index of the Nth same-name+arity overload's function (declaration
/// order), falling back to the last slot. Call sites that ran
/// overload selection pass their winner's position here instead of
/// collapsing onto one body.
pub(super) fn function_index_nth(
    ctx: &Ctx<'_>,
    class_name: &str,
    method: &str,
    arity: usize,
    n: usize,
) -> Option<u32> {
    let fname = format!("{class_name}::{method}#{arity}");
    functions_index(ctx).get(&fname).map(|v| {
        v.get(n).copied().unwrap_or_else(|| {
            v.last().copied().unwrap_or(0)
        }) as u32
    })
}

/// Like `function_index_for`, but also walks the base-class chain: static
/// methods are inherited, so `Derived.M(...)` resolves to the function
/// declared on the base that defines it. Overload-aware (Nth slot).
pub(super) fn static_function_index_for(
    ctx: &Ctx<'_>,
    class_name: &str,
    method: &str,
    arity: usize,
    param_types: &[String],
    byref: &[bool],
) -> Option<u32> {
    let mut seen: std::collections::BTreeSet<String> = Default::default();
    let mut stack = vec![class_name.to_string()];
    while let Some(name) = stack.pop() {
        if !seen.insert(name.clone()) {
            continue;
        }
        let n = overload_position(ctx, &name, method, arity, param_types, byref);
        if let Some(fidx) = function_index_nth(ctx, &name, method, arity, n) {
            return Some(fidx);
        }
        if let Some(info) = ctx.table.class(&name) {
            for base in &info.bases {
                if ctx.table.class(base).is_some() {
                    stack.push(base.clone());
                }
            }
        }
    }
    None
}

/// Canonical virtual-dispatch slot: the method's recorded position in
/// the receiver entry's method list. Method lists are base-chain order
/// first (single inheritance makes bases a stable prefix), so derived
/// rows share their bases' numbering — stable across builds from the
/// same sources, unlike global first-seen (which shifts with table
/// content/order and can never agree cross-image). `recv` is the exact
/// entry when known (`this` → `current_info`); otherwise same-named
/// entries are searched for the first declaring the method (mirrors
/// `member_host`'s declarer preference). Interface receivers and total
/// misses fall back to the first implementer in table order, else 0
/// (today's leniency for interface dispatch, documented).
pub(super) fn canonical_slot(
    table: &TypeTable,
    recv: Option<&ClassInfo>,
    recv_type: &str,
    method: &str,
) -> usize {
    // TEMP-SLOTDIAG: audit slot selection for a named method.
    // Removed before close.
    let diag = std::env::var("RUBYC_SLOTDIAG").unwrap_or_default();
    let want = !diag.is_empty() && method == diag;
    if let Some(info) = recv {
        if let Some(m) = info.methods.iter().find(|m| m.name == method) {
            if want {
                eprintln!("[rubyc] TEMP-SLOTDIAG recv-entry {} slot={}", info.key.path, m.vtable_slot);
            }
            return m.vtable_slot;
        }
    }
    let r = table
        .classes
        .iter()
        .filter(|c| c.name == recv_type || c.key.path == recv_type)
        .filter_map(|c| c.methods.iter().find(|m| m.name == method))
        .map(|m| m.vtable_slot)
        .next()
        .or_else(|| {
            table
                .classes
                .iter()
                .filter_map(|c| c.methods.iter().find(|m| m.name == method))
                .map(|m| m.vtable_slot)
                .next()
        })
        .unwrap_or(0);
    if want {
        eprintln!("[rubyc] TEMP-SLOTDIAG recv_type={recv_type} slot={r}");
    }
    r
}

#[derive(Clone, Copy)]
enum Fd {
    Stdout,
    Stderr,
}

pub(super) fn printf(args: &[Expr], fd: Fd, b: &mut Body<'_>, ctx: &Ctx<'_>) -> Result<VReg, LowerError> {
    use crate::utilities::format::Segment;
    let Some((Expr::Literal(Literal::Str(format)), rest)) = args.split_first() else {
        return Err(LowerError::UnsupportedConstruct(
            crate::t!("rubyc.native.ir.lower_call.not_literal").to_owned(),
        ));
    };
    let segments = crate::utilities::format::split_format(format)
        .map_err(|e| LowerError::UnsupportedConstruct(format!("printf format string: {e}")))?;
    crate::utilities::format::validate_args(&segments, rest.len())
        .map_err(|e| LowerError::UnsupportedConstruct(format!("printf: {e}")))?;

    // Arguments evaluate left-to-right into their own vregs; liveness keeps
    // each alive until its write segment consumes it. Whether each arg is a
    // first-class string is known up front (static type) so the render loop
    // can pick the right write instruction. Spelling-agnostic like the
    // element paths (an exact match once sent `String`-spelled values
    // down WriteInt).
    let vals = eval_args(rest, b, ctx)?;
    let is_string: Vec<bool> = rest
        .iter()
        .map(|e| static_type_of(e, b, ctx).map(|t| is_string_type(&t)).unwrap_or(false))
        .collect();
    for segment in segments {
        match (segment, fd) {
            (Segment::Text(text), Fd::Stdout) => {
                let (off, len) = b.string_literal(&text);
                b.push_raw(Instr::WriteStr { off, len });
            }
            (Segment::Text(text), Fd::Stderr) => {
                let (off, len) = b.string_literal(&text);
                b.push_raw(Instr::EWriteStr { off, len });
            }
            (Segment::Arg(i), Fd::Stdout) => {
                if is_string[i] {
                    b.raw_vreg(Instr::WriteString { src: vals[i] });
                } else {
                    b.push_raw(Instr::WriteInt { src: vals[i] });
                }
            }
            (Segment::Arg(i), Fd::Stderr) => {
                b.push_raw(Instr::EWriteInt { src: vals[i] });
            }
        }
    }
    let zero = b.const_i64(0);
    Ok(zero)
}

// ---------------------------------------------------------------------------
// Async state-machine lowering
// ---------------------------------------------------------------------------

/// Transform an async function into a state-machine coroutine.
///
/// The transform:
/// 1. Identifies all `Suspend` instructions (suspension points).
/// 2. Splits the function body into N+1 state segments, where N is the
///    number of suspension points.
/// 3. Each segment is one "state" of the coroutine.
/// 4. The first block (state 0) starts with a `TaskAlloc` to allocate the
///    task frame.
/// 5. Each block that ends with a `Suspend` saves state and returns to the
///    scheduler (the backend handles the actual suspension).
/// 6. A dispatch block at the entry selects the starting state based on
///    whether this is first entry or a resume.
///
/// The state vreg passed to `Suspend` is the state number (0, 1, 2, ...).
/// When the coroutine is resumed, the backend reads the saved state from the
/// task frame and jumps to the corresponding block.
pub(super) fn lower_async_state_machine(mut f: IrFunction) -> IrFunction {
    // Count suspension points and record their positions.
    let mut suspend_positions: Vec<(usize, usize)> = Vec::new(); // (block_idx, instr_idx)
    for (bi, block) in f.blocks.iter().enumerate() {
        for (ii, instr) in block.instrs.iter().enumerate() {
            if matches!(instr, Instr::Suspend { .. }) {
                suspend_positions.push((bi, ii));
            }
        }
    }

    if suspend_positions.is_empty() {
        // No suspension points — not actually async, leave as-is.
        f.is_async = true;
        return f;
    }

    // Safety gate: the transform below linearizes the body (it keeps
    // instructions but rebuilds terminators), so it is only valid for
    // branch-free flows. Any conditional branch, throw, or non-final
    // return would have its edge dropped or rewritten to a dangling
    // target (`BufferedStream.FlushAsyncInternal` branched into a
    // 4-block hole). Leave such functions linear: the backend treats
    // `Suspend` as a nop, so they compile (synchronously) and real
    // suspension arrives with the async runtime.
    let linear = f.blocks.iter().enumerate().all(|(bi, b)| {
        let last = bi + 1 == f.blocks.len();
        match &b.term {
            Terminator::Branch { cond: None, if_true, if_false } => {
                !last && *if_true as usize == bi + 1 && *if_false as usize == bi + 1
            }
            Terminator::Ret(_) | Terminator::Exit | Terminator::ExitWith(_) => last,
            Terminator::Branch { .. } | Terminator::Unwind { .. } | Terminator::Rethrow => false,
        }
    });
    if !linear {
        f.is_async = true;
        return f;
    }

    let n_states = suspend_positions.len() + 1;
    let task_bytes = 24 + (f.vregs * 8) as u64;

    // Allocate vregs for the transform.
    let task = f.fresh_vreg(); // task frame pointer
    let state_vreg = f.fresh_vreg(); // current state (for dispatch)
    let _ = state_vreg; // used by Suspend instructions (replaced below)

    // We need to restructure the blocks. The strategy:
    // - The original blocks are linearized in order.
    // - Each Suspend splits the sequence: instructions before Suspend form
    //   the tail of state S, and instructions after Suspend (in the same or
    //   subsequent blocks) form the head of state S+1.
    //
    // For the initial implementation, we handle the common case where
    // Suspend instructions appear in the linear flow (no branches across
    // suspend points). Each Suspend is the last instruction of its block,
    // and the next block starts the next state.
    //
    // Build new blocks: one per state, each containing the instructions
    // between consecutive suspend points.

    // Flatten all instructions in order, tracking which state each belongs to.
    let mut state_segments: Vec<Vec<Instr>> = vec![Vec::new(); n_states];
    let mut current_state = 0usize;
    for block in f.blocks.iter() {
        for instr in block.instrs.iter() {
            if matches!(instr, Instr::Suspend { .. }) {
                // This Suspend ends the current state segment.
                // The next instruction starts state+1.
                current_state += 1;
                continue;
            }
            if current_state < n_states {
                state_segments[current_state].push(instr.clone());
            }
        }
    }

    // Collect terminators from the original blocks (we need the final Ret).
    let last_term = f
        .blocks
        .last()
        .map(|b| b.term.clone())
        .unwrap_or(Terminator::Ret(None));

    // Build the new block layout:
    // Block 0: dispatch (load state, jump to appropriate state block)
    // Block 1..N: state 0..N-1 (body + Suspend or fall-through)
    // Block N+1: final state (body + Ret)
    //
    // For simplicity, we use a sequential layout:
    // Block 0: TaskAlloc + dispatch (branch on state)
    // Block 1: state 0 body, ends with Suspend(0) or jump to next
    // Block 2: state 1 body, ends with Suspend(1) or jump to next
    // ...
    // Block N: state N-1 body, ends with Suspend(N-1) or jump to final
    // Block N+1: state N body + Ret

    let mut new_blocks: Vec<Block> = Vec::new();

    // Block 0: prologue + dispatch.
    // TaskAlloc, then load the state from the task frame (offset 16),
    // and branch to the appropriate state block.
    let _dispatch_block_id = 0u32;
    let mut dispatch_block = Block::empty();
    dispatch_block.instrs.push(Instr::TaskAlloc {
        dst: task,
        bytes: task_bytes,
    });
    // Load state from task frame: [task + 16]
    dispatch_block.instrs.push(Instr::LoadField {
        dst: state_vreg,
        base: task,
        off: 16,
    });
    // For the dispatch, we chain conditional branches:
    // if state == 0 goto state_0
    // if state == 1 goto state_1
    // ...
    // The branches are built after all state blocks exist.
    dispatch_block.term = Terminator::Branch {
        cond: None,
        if_true: 1, // default: state 0
        if_false: 1,
    };
    new_blocks.push(dispatch_block);

    // State blocks: one per state.
    for (s, segment) in state_segments.iter().enumerate() {
        let mut state_block = Block::empty();
        state_block.instrs = segment.clone();

        if s < n_states - 1 {
            // This state ends with a Suspend. The Suspend was stripped from
            // the segment; we add it back with the correct state value.
            let suspend_state = f.fresh_vreg();
            state_block.instrs.push(Instr::Const {
                dst: suspend_state,
                imm: s as i64,
            });
            state_block.instrs.push(Instr::Suspend { state: suspend_state });
            // After Suspend, the backend returns to the scheduler.
            // When resumed, it re-enters at the dispatch block.
            // For the IR, we mark this as a "suspend" terminator by
            // using an Exit (the backend intercepts Suspend before
            // reaching the terminator).
            state_block.term = Terminator::Exit;
        } else {
            // Final state: ends with the original terminator.
            state_block.term = last_term.clone();
        }
        new_blocks.push(state_block);
    }

    // Build the dispatch branches: instead of a simple fall-through,
    // we need a chain of comparisons. For now, use a simplified dispatch:
    // the backend reads the state and jumps directly (the IR just needs
    // the state blocks to be laid out sequentially). The dispatch block
    // falls through to state 0; the backend handles re-entry at the
    // correct state via the task frame.
    //
    // This is sufficient for the backend to implement the state machine:
    // it reads the state from the task frame and jumps to block (state+1).

    f.blocks = new_blocks;
    f.is_async = true;

    f
}