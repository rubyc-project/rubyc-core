//! AST to IR body lowering.

use super::super::{Block, Instr, IrFunction, LowerError, Terminator, VReg};
use crate::models::{ClassMember, Expr, Literal};
use crate::native::ir::Ctx;
use crate::native::semantic::{ClassInfo, FieldInfo, MethodInfo, PropertyInfo, TypeTable};
use super::{body::*, select::*, types::*};
use super::eval::eval;


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

/// Split a generic type spelling into its base and top-level
/// arguments. `None` for a non-generic spelling.
fn split_type_generic(s: &str) -> Option<(&str, Vec<String>)> {
    let open = s.find('<')?;
    if !s.ends_with('>') {
        return None;
    }
    let args = explicit_type_args(s);
    if args.is_empty() {
        return None;
    }
    Some((s[..open].trim(), args))
}

/// Whether two type spellings name the same type. `Task` and
/// `System.Threading.Tasks.Task` are one type, so inferring `T` from
/// both is agreement, not a conflict.
fn same_type(ctx: &Ctx<'_>, a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    match (ctx.table.class(a), ctx.table.class(b)) {
        (Some(x), Some(y)) => x.key.path == y.key.path,
        _ => false,
    }
}

/// Bind one type parameter by matching a declared parameter type
/// against an argument's static type. C# inference is broader than
/// this (interfaces, conversions, return types); the shapes here cover
/// the generic calls in CoreLib's numeric and collection code, and an
/// unbound parameter is reported instead of guessed.
fn unify_type_arg(
    ctx: &Ctx<'_>,
    param: &str,
    arg: &str,
    vars: &[String],
    map: &mut std::collections::BTreeMap<String, Vec<String>>,
) -> Result<(), LowerError> {
    let p = param.trim();
    // A by-reference argument types as `T&` (the parser represents
    // both `ref x` and `&x` with AddrOf). The parameter's declared
    // type is `T`; strip the address marker before matching shapes.
    let a = arg.trim().trim_end_matches('&').trim();
    if p.is_empty()
        || a.is_empty()
        || a == "_"
        || a == "object"
        || a == "Object"
        || a == "null"
        || a == "method-group"
    {
        return Ok(());
    }
    if vars.iter().any(|v| v == p) {
        // The candidates ACCUMULATE: C# fixes them afterwards (the
        // best common type), so a conflict is not an error here.
        let entry = map.entry(p.to_owned()).or_default();
        if !entry.iter().any(|e| same_type(ctx, e, a)) {
            entry.push(a.to_owned());
        }
        Ok(())
    } else if let Some(inner) = p.strip_suffix("[]") {
        if let Some(arg_inner) = a.strip_suffix("[]") {
            unify_type_arg(ctx, inner, arg_inner, vars, map)?;
        }
        Ok(())
    } else if p.ends_with('*') {
        // Pointer-typed generic parameters (`LoadAligned<T>(T* source)`):
        // bind the element type from a pointer argument.
        let inner = p.trim_end_matches('*').trim();
        let arg_inner = if a.ends_with('*') {
            a.trim_end_matches('*').trim()
        } else {
            a
        };
        if !inner.is_empty() && !arg_inner.is_empty() {
            unify_type_arg(ctx, inner, arg_inner, vars, map)?;
        }
        Ok(())
    } else if let Some(inner) = p.strip_suffix('?') {
        let arg_inner = a.strip_suffix('?').unwrap_or(a);
        unify_type_arg(ctx, inner, arg_inner, vars, map)
    } else if p.starts_with("System.Nullable<") || p.starts_with("Nullable<") {
        let inner = p
            .split('<')
            .nth(1)
            .and_then(|s| s.strip_suffix('>'))
            .unwrap_or("");
        let arg_inner = a.strip_suffix('?').unwrap_or(a);
        unify_type_arg(ctx, inner, arg_inner, vars, map)
    } else if let Some((pb, pargs)) = split_type_generic(p) {
        if let Some((ab, aargs)) = split_type_generic(a) {
            if base_type_name(pb) == base_type_name(ab) && pargs.len() == aargs.len() {
                for (pp, aa) in pargs.iter().zip(aargs.iter()) {
                    unify_type_arg(ctx, pp, aa, vars, map)?;
                }
            }
        } else if base_type_name(pb) == "ReadOnlySpan"
            || base_type_name(pb) == "Span"
        {
            // A span argument is a packed array or a string; both
            // bind the element type.
            if is_string_type(a) && pargs.len() == 1 {
                unify_type_arg(ctx, &pargs[0], "char", vars, map)?;
            } else if let Some(elem) = a.strip_suffix("[]") {
                if pargs.len() == 1 {
                    unify_type_arg(ctx, &pargs[0], elem, vars, map)?;
                }
            }
        }
        Ok(())
    } else {
        Ok(())
    }
}

/// Concrete/explicit type arguments for a generic call. A caller's own
/// type parameter stays a spelling; `Body::type_token` resolves it to
/// the hidden handle.
fn method_type_args(
    ctx: &Ctx<'_>,
    mi: &crate::native::semantic::MethodInfo,
    callee: &str,
    arg_types: &[String],
) -> Result<Vec<String>, LowerError> {
    if mi.generic_params.is_empty() {
        return Ok(Vec::new());
    }
    let explicit = explicit_type_args(callee);
    if !explicit.is_empty() {
        if explicit.len() != mi.generic_params.len() {
            return Err(LowerError::UnsupportedConstruct(format!(
                "`{callee}` supplies {} type argument(s) but declares {}",
                explicit.len(),
                mi.generic_params.len()
            )));
        }
        return Ok(explicit);
    }
    let mut map: std::collections::BTreeMap<String, Vec<String>> =
        std::collections::BTreeMap::new();
    for (i, p) in mi.param_types.iter().enumerate() {
        let Some(a) = arg_types.get(i) else {
            break;
        };
        unify_type_arg(ctx, p, a, &mi.generic_params, &mut map)?;
    }
    let mut out = Vec::with_capacity(mi.generic_params.len());
    for v in &mi.generic_params {
        match map.get(v) {
            Some(cands) if !cands.is_empty() => {
                out.push(fix_type_argument(ctx, callee, v, cands)?)
            }
            _ => {
                return Err(LowerError::UnsupportedConstruct(format!(
                    "cannot infer generic type argument `{v}` for `{callee}`; pass it explicitly"
                )));
            }
        }
    }
    Ok(out)
}

/// Fix one type parameter's candidates (C# 12.6.3.4, "Fixing type
/// arguments"). The candidates are the types the arguments offered.
/// A type-parameter candidate loses to a concrete one — the concrete
/// type is the exact match — and among concrete candidates the best
/// common type wins: the one every other candidate converts to (`int`
/// and `long` infer `long`, `int` and `double` infer `double`).
fn fix_type_argument(
    ctx: &Ctx<'_>,
    callee: &str,
    var: &str,
    candidates: &[String],
) -> Result<String, LowerError> {
    let mut unique: Vec<&String> = Vec::new();
    for c in candidates {
        if !unique.iter().any(|u| same_type(ctx, u, c)) {
            unique.push(c);
        }
    }
    if unique.len() == 1 {
        return Ok(unique[0].clone());
    }
    let concrete: Vec<&String> = unique
        .iter()
        .copied()
        .filter(|c| !ctx.is_type_var(c))
        .collect();
    let pool: Vec<&String> = if concrete.is_empty() {
        unique
    } else {
        concrete
    };
    if pool.len() == 1 {
        return Ok(pool[0].clone());
    }
    for c in &pool {
        if pool
            .iter()
            .all(|o| same_type(ctx, o, c) || converts_to(ctx, o, c))
        {
            return Ok((*c).clone());
        }
    }
    Err(LowerError::UnsupportedConstruct(format!(
        "generic type argument `{var}` is inferred as both `{}` and `{}`",
        pool[0], pool[1]
    )))
}

/// Evaluate the arguments of a call, telling a DELEGATE argument the
/// delegate type it is converted to: the parameter names it, and the
/// value takes its runtime type from the conversion.
fn eval_args_with_delegate_types(
    args: &[Expr],
    mi: &MethodInfo,
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
) -> Result<Vec<VReg>, LowerError> {
    let mut out = Vec::with_capacity(args.len());
    for (i, a) in args.iter().enumerate() {
        let saved = b.expected_delegate.take();
        if let Some(pt) = mi.param_types.get(i)
            && ctx.table.is_delegate(pt)
        {
            b.expected_delegate = Some(pt.clone());
        }
        out.push(eval(a, b, ctx)?);
        b.expected_delegate = saved;
    }
    Ok(out)
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
        let mut argv = eval_args(args, b, ctx)?;
        // Local functions inherit the enclosing method's type
        // parameters and were emitted with their handles at the end
        // of the argument list.
        for v in ctx.method_vars {
            argv.push(b.type_token(v, ctx)?);
        }
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
        // The operand is a TYPE, not an expression: evaluate it as a
        // name. A bare type parameter reads its hidden handle; a
        // concrete type gets its canonical descriptor. Identity is the
        // descriptor pointer, which is what `typeof(A) == typeof(B)`
        // and a generic `typeof(T)` comparison need.
        let Some(Expr::Ident { name: ty, .. }) = args.first() else {
            return Err(LowerError::UnsupportedConstruct(
                "`typeof` needs a type name; this operand shape is not supported"
                    .to_owned(),
            ));
        };
        if args.len() != 1 {
            return Err(LowerError::UnsupportedConstruct(
                "`typeof` takes exactly one type operand".to_owned(),
            ));
        }
        return b.type_token(ty, ctx);
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
        // JIT-identity intrinsics (managed bodies throw
        // `PlatformNotSupportedException`; real `ldarg.0 / ret`
        // lives in the JIT): erased refs are values, so both are
        // the identity.
        if current_base == "Unsafe" && matches!(bare, "As" | "AsRef" | "AsPointer" | "BitCast") && args.len() == 1 {
            let argv = eval_args(args, b, ctx)?;
            return Ok(argv[0]);
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
        // `Unsafe.IsNullRef(ref x)` (see `eval.rs`): a ref to any
        // variable is never a null pointer — yield false constantly.
        if current_base == "Unsafe" && bare == "IsNullRef" && args.len() == 1 {
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

    // Generic method with explicit CONCRETE type arguments and a body:
    // inline a SPECIALIZATION of it. A generic body is lowered once with
    // its type parameters erased, so `T.StaticAbstractMember` has no
    // runtime meaning; substituting the concrete type at the call site
    // lets those members resolve like any other static member. This is
    // how C# generic math reaches `TValue.One` / `TValue.IsZero(x)`.
    if !explicit_type_args(callee).is_empty() {
        let sub_seed: Vec<(String, String)> = Vec::new();
        let targs: Vec<String> = explicit_type_args(callee)
            .iter()
            .map(|t| {
                super::types::subst_type_var(
                    t,
                    ctx.type_subst.unwrap_or(&sub_seed),
                )
                .unwrap_or_else(|| t.clone())
            })
            .collect();
        if !targs.is_empty()
            && targs.iter().all(|t| !ctx.is_type_var(t))
            && let Some((ginfo, gdecl)) =
                find_generic_method(ctx, name, targs.len(), args.len())
            && let Some(gbody) = gdecl.body.clone()
            // A lambda in the body must be lifted by the function that
            // owns it. The inline path lowers into the CALLER's body,
            // which may be a local function with no lambda queue;
            // calling the real generic function keeps the lambda (and
            // its type parameters) in its own frame.
            && crate::native::ir::count_lambdas(&gbody) == 0
        {
            let argv = eval_args(args, b, ctx)?;
            let subst: Vec<(String, String)> = gdecl
                .generic_params
                .iter()
                .cloned()
                .zip(targs.iter().cloned())
                .collect();
            let ret_spelling = super::types::subst_type_var(&gdecl.ret_type, &subst)
                .unwrap_or_else(|| gdecl.ret_type.clone());
            let icontext = Ctx {
                // The class the inlined body belongs to, by its SIMPLE
                // name: the access check compares the viewer against
                // the declaring class's simple name, and the qualified
                // path lives in `current_path` for scope resolution.
                current_class: &ginfo.name,
                current_info: Some(ginfo),
                current_path: Some(&ginfo.key.path),
                // Keep the type parameters visible: they are still
                // type VARIABLES for shape checks, and the substitution
                // resolves them to concrete types where it matters.
                method_vars: &gdecl.generic_params,
                type_subst: Some(&subst),
                current_method: Some(name.to_owned()),
                current_ret: Some(ret_spelling),
                lambda_deferred: None,
                lambda_base: 0,
                ..ctx.clone()
            };
            b.scope_enter();
            for (p, v) in gdecl.params.iter().zip(argv.into_iter()) {
                let ty = super::types::subst_type_var(&p.type_name, &subst)
                    .unwrap_or_else(|| p.type_name.clone());
                let slot = b
                    .scopes
                    .declare(&p.name, &ty, crate::models::Access::Default)
                    .map_err(LowerError::UnsupportedConstruct)?;
                while b.bindings.len() <= slot {
                    let fresh = b.fresh();
                    b.bindings.push(fresh);
                }
                b.mov(b.bindings[slot], v);
            }
            let res = super::eval::eval_block_inline_pub(b, &icontext, &gbody);
            b.scope_exit();
            return res;
        }
    }

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
            .map(|a| overload_arg_type(a, b, ctx))
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
    //
    // An ERASED generic body calls `M<T>(...)` with its own type
    // parameters; a static member on such a parameter types as
    // `object`, which would disqualify a value-typed parameter and lose
    // the callee entirely. The type arguments are explicit and generic,
    // so they cannot discriminate overloads here: let those unknown
    // arguments wildcard instead (the erased body is never the executed
    // path when a concrete instantiation exists, but it must still
    // resolve so the program builds).
    let erased_generic_call = !explicit_type_args(callee).is_empty()
        && explicit_type_args(callee).iter().any(|t| {
            ctx.is_type_var(t)
                || ctx.type_subst.is_some_and(|s| {
                    s.iter().any(|(v, _)| v == t)
                        && !ctx
                            .type_subst
                            .is_some_and(|s| s.iter().any(|(v, c)| v == t && !ctx.is_type_var(c)))
                })
        });
    let arg_types: Vec<String> = args
        .iter()
        .map(|a| {
            if is_method_group(a, b, ctx) {
                return "method-group".to_owned();
            }
            let t = overload_arg_type(a, b, ctx);
            if erased_generic_call && (t == "object" || ctx.is_type_var(&t)) {
                "_".to_owned()
            } else {
                t
            }
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
    let all_entries = entries_for(ctx.table, ctx.current_class);
    let is_static = enclosing_is_static(ctx, b);
    if is_static {
        if let Some(first) = all_entries.first() {
            if !entries.iter().any(|e| e.key == first.key) {
                entries.push(first);
            }
        }
    } else {
        entries.extend(all_entries);
    }
    // In a static context, prefer static overloads: an inherited
    // instance `Equals`/`ToString` must not shadow the intended
    // static one. Try statics first; fall back to both so an instance
    // member reached from a static context still surfaces as CS0120
    // (reported by the check below) instead of "unknown method".
    let staticness = if enclosing_is_static(ctx, b) {
        Some(true)
    } else {
        None
    };
    let mut selected = select_visible(
        ctx.table,
        &entries,
        name,
        &arg_types,
        staticness,
        ctx.current_class,
        explicit_arity,
    );
    if selected.is_none() && staticness.is_some() {
        selected = select_visible(
            ctx.table,
            &entries,
            name,
            &arg_types,
            None,
            ctx.current_class,
            explicit_arity,
        );
    }
    if let Some(mi) = selected {
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
        // CS0120: an INSTANCE member cannot be reached from a static
        // context. The receiver would be `VReg(0)`, which in a static
        // method is not a `this` at all — the call compiled and read
        // garbage. Reported with the C# rule rather than lowered.
        if !mi.is_static && enclosing_is_static(ctx, b) {
            return Err(LowerError::UnsupportedConstruct(crate::tf!(
                "rubyc.native.ir.lower_call.instance_from_static",
                name.to_string(),
                ctx.current_class.to_string()
            )));
        }

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
        // Named arguments (`M(b: 1, a: 2)`) are rewritten into
        // parameter order before evaluation, so byref boxing, `params`
        // packing and defaults all see a positional list.
        let named = positional_args(args, mi)?;
        let args: &[Expr] = named.as_deref().unwrap_or(args);
        let mut explicit = eval_args_with_delegate_types(args, mi, b, ctx)?;
        let full_arity = mi.param_types.len();
        // `params` packs the trailing arguments into one array BEFORE
        // defaults (the array itself is the last parameter).
        let params_slot = mi.param_types.len().saturating_sub(1);
        let last_is_array = mi.has_params
            && args.len() == mi.param_types.len()
            && args.last().is_some_and(|a| {
                static_type_of(a, b, ctx)
                    .map(|t| base_type_name(&t) == base_type_name(&mi.param_types[params_slot]))
                    .unwrap_or(false)
            });
        pack_params(&mut explicit, 0, mi, b, last_is_array);
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
        // Hidden type handles: every generic method receives one token
        // per type parameter, in declaration order, after the explicit
        // parameters. The callee binds them to `typeof(T)`; a concrete
        // argument gets its canonical descriptor, a caller type
        // parameter gets the caller's own handle.
        let positional_types: Vec<String> = args
            .iter()
            .map(|a| overload_arg_type(a, b, ctx))
            .collect();
        // A primitive passed to an object/interface parameter is boxed
        // here, at the last possible moment (after defaults and
        // by-reference boxing).
        for i in 0..explicit.len().min(mi.param_types.len()) {
            let Some(at) = positional_types.get(i) else {
                break;
            };
            if mi.param_is_byref.get(i).copied().unwrap_or(false) {
                continue;
            }
            let ab = super::select::base_type_name(at);
            if super::typeinfo::primitive_id(&ab).is_some()
                && Body::is_object_target(&mi.param_types[i], ctx)
            {
                explicit[i] = b.box_value(explicit[i], at, ctx)?;
            }
        }
        if !mi.generic_params.is_empty() {
            let targs = method_type_args(ctx, mi, callee, &positional_types)?;
            for t in &targs {
                explicit.push(b.type_token(t, ctx)?);
            }
        }
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
                .or(canonical_slot(
                    ctx.table,
                    ctx.current_info,
                    ctx.current_class,
                    name,
                    &mi.param_types,
                ))
                .ok_or_else(|| {
                    LowerError::UnsupportedConstruct(crate::tf!(
                        "rubyc.native.ir.lower_call.no_virtual_slot",
                        ctx.current_class,
                        name.to_string()
                    ))
                })?;
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
                    .or(canonical_slot(
                        ctx.table,
                        ctx.current_info,
                        ctx.current_class,
                        name,
                        &mi.param_types,
                    ))
                    .ok_or_else(|| {
                        LowerError::UnsupportedConstruct(crate::tf!(
                            "rubyc.native.ir.lower_call.no_virtual_slot",
                            ctx.current_class,
                            name.to_string()
                        ))
                    })?;
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
                .or(canonical_slot(
                    ctx.table,
                    ctx.current_info,
                    ctx.current_class,
                    name,
                    &mi.param_types,
                ))
                .ok_or_else(|| {
                    LowerError::UnsupportedConstruct(crate::tf!(
                        "rubyc.native.ir.lower_call.no_virtual_slot",
                        ctx.current_class,
                        name.to_string()
                    ))
                })?;
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

    // A delegate held in a local, invoked as `f(args)`. The object's
    // word 0 is the ADDRESS to call; word 8 says which convention the
    // callee uses — 1 for a lifted lambda (it takes the delegate object
    // first, so it can reach its captures), 0 for a method group (it
    // takes only its own arguments). The two call arms are selected at
    // run time because the same local can hold either.
    if let Some(sym) = b.scopes.lookup(name) {
        let (slot, is_byref) = (sym.slot, sym.is_byref);
        let obj = if is_byref {
            let home = b.bindings[slot];
            let o = b.fresh();
            b.push_raw(Instr::LoadField {
                dst: o,
                base: home,
                off: 0,
            });
            o
        } else {
            b.bindings[slot]
        };
        return invoke_delegate(b, ctx, obj, args);
    }
    // Field/property-held delegate invocation (`m_action()` where
    // `m_action` is an `Action` field, or `InternalSetNameMethod()`
    // where that's a delegate-typed property): same gap.
    if let Some(field) = this_info(ctx, name).and_then(|info| info.field(name)) {
        let obj = if field.is_static {
            // A static field lives in the class's storage; the address
            // is the slot that HOLDS the delegate handle.
            let Some(addr) = static_addr(field, b, ctx) else {
                return Err(LowerError::UnsupportedConstruct(format!(
                    "static field '{name}' has no storage slot for a delegate"
                )));
            };
            let v = b.fresh();
            b.push_raw(Instr::LoadField {
                dst: v,
                base: addr,
                off: 0,
            });
            v
        } else {
            let off = (field.offset_slots * 8) as u32;
            let this = b.this_vreg();
            let v = b.fresh();
            b.push_raw(Instr::LoadField {
                dst: v,
                base: this,
                off,
            });
            v
        };
        return invoke_delegate(b, ctx, obj, args);
    }
    if this_info(ctx, name).is_some_and(|info| info.property(name).is_some()) {
        // A delegate-typed property: read it through the normal member
        // path (the getter may be inlined), then invoke the value.
        let span = crate::models::Span::new(0, 0);
        let member = Expr::Member {
            object: Box::new(Expr::Ident {
                name: String::from("this"),
                span,
            }),
            name: name.to_string(),
            span,
        };
        let obj = eval(&member, b, ctx)?;
        return invoke_delegate(b, ctx, obj, args);
    }
    // Enclosing-scope statics: unqualified calls from nested code
    // resolve outward (`AlternateLookup` ctor calling outer
    // `IsCompatibleKey(...)`). Instance members need an instance and
    // are never candidates; the resolved static lowers direct
    // (`CallStatic`, no receiver).
    if let Some(mi) = outer_static(ctx, name, &arg_types) {
        ctx.check_method_access(mi)?;
        let mut argv = eval_args_with_delegate_types(args, mi, b, ctx)?;
        let params_slot = mi.param_types.len().saturating_sub(1);
        let last_is_array = mi.has_params
            && args.len() == mi.param_types.len()
            && args.last().is_some_and(|a| {
                static_type_of(a, b, ctx)
                    .map(|t| base_type_name(&t) == base_type_name(&mi.param_types[params_slot]))
                    .unwrap_or(false)
            });
        pack_params(&mut argv, 0, mi, b, last_is_array);
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
        let mut argv = eval_args_with_delegate_types(args, mi, b, ctx)?;
        let params_slot = mi.param_types.len().saturating_sub(1);
        let last_is_array = mi.has_params
            && args.len() == mi.param_types.len()
            && args.last().is_some_and(|a| {
                static_type_of(a, b, ctx)
                    .map(|t| base_type_name(&t) == base_type_name(&mi.param_types[params_slot]))
                    .unwrap_or(false)
            });
        pack_params(&mut argv, 0, mi, b, last_is_array);
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
/// The class under lowering, by its qualified path when the context
/// knows it: a nested type's simple name is ambiguous (`AlternateLookup`
/// exists in three collections), and the enclosing-scope walk must
/// start from the right one.
fn current_class_entry<'t>(
    ctx: &'t Ctx<'t>,
) -> Option<&'t crate::native::semantic::ClassInfo> {
    if let Some(path) = ctx.current_path
        && !path.is_empty()
        && let Some(info) = ctx.table.classes.iter().find(|c| c.key.path == path)
    {
        return Some(info);
    }
    ctx.table.classes.iter().find(|c| c.name == ctx.current_class)
}

pub(super) fn outer_static_field<'t>(
    ctx: &'t Ctx<'_>,
    name: &str,
) -> Option<(&'t ClassInfo, &'t FieldInfo)> {
    // Enclosing paths, innermost first, deduped (same walk as
    // `outer_static`).
    let mut outers: Vec<String> = Vec::new();
    for c in current_class_entry(ctx).into_iter() {
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
    for c in current_class_entry(ctx).into_iter() {
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
/// Whether the method (or accessor) being lowered is `static`.
///
/// The enclosing context is a name plus an entry, so this looks the
/// declaration up rather than carrying a flag — accessors and inlined
/// bodies rebuild the context and would have to repeat it anyway.
fn enclosing_is_static(ctx: &Ctx<'_>, b: &Body<'_>) -> bool {
    let Some(info) = ctx.current_info else {
        return false;
    };
    let Some(name) = ctx.current_method.as_deref() else {
        return false;
    };
    let same_name = |m: &&crate::native::semantic::MethodInfo| {
        m.name == name && m.declaring_class == info.name
    };
    // The overload under lowering is named, not identified: a class can
    // declare a static and an instance method with the same name
    // (`RuntimeTypeHandle.GetConstraints`), and the first same-named
    // entry answered for the other one. The declared return type and
    // the arity tell them apart; the name alone is the fallback.
    let arity = b.f.arity;
    info.methods
        .iter()
        .find(|m| {
            same_name(m)
                && Some(&m.ret_type) == ctx.current_ret.as_ref()
                && m.param_types.len() == arity
        })
        .or_else(|| {
            info.methods.iter().find(|m| {
                same_name(m) && Some(&m.ret_type) == ctx.current_ret.as_ref()
            })
        })
        .or_else(|| info.methods.iter().find(same_name))
        .map(|m| m.is_static)
        .unwrap_or(false)
}

pub(super) fn outer_static<'a>(
    ctx: &Ctx<'a>,
    callee: &str,
    arg_types: &[String],
) -> Option<&'a MethodInfo> {
    // Enclosing paths, innermost first, deduped.
    let mut outers: Vec<String> = Vec::new();
    for c in current_class_entry(ctx).into_iter() {
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
pub(crate) fn functions_index(
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
        1usize + crate::native::ir::collect_static_locals(ctx.classes, ctx.class_paths).len();
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
                // The emitter appends this body's lifted lambdas right
                // after it, so the NEXT method starts past them. Without
                // this term every method after the first lambda resolved
                // to the wrong function — a program containing a lambda
                // called its constructors through unrelated methods.
                next += 1
                    + m.body
                        .as_ref()
                        .map(crate::native::ir::count_lambdas)
                        .unwrap_or(0);
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
    // Function names are built from the AST class name (`IntNum`),
    // while `declaring_class` may carry the namespace (`t.IntNum`).
    // Probe both spellings (and their bases).
    let simple = class_name.rsplit('.').next().unwrap_or(class_name);
    let mut stack = vec![class_name.to_string(), simple.to_string()];
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
/// `member_host`'s declarer preference). A total miss returns `None`:
/// dispatching to slot 0 would jump into an unrelated function body,
/// so callers report it.
pub(super) fn canonical_slot(
    table: &TypeTable,
    recv: Option<&ClassInfo>,
    recv_type: &str,
    method: &str,
    param_types: &[String],
) -> Option<usize> {
    // A call through an INTERFACE uses the shared interface range: its
    // slot is the same in every implementer's row. Deriving the slot
    // from one implementing class (the old fallback) picked a class
    // slot, which differs between implementers and dispatched to an
    // unrelated body (or off the end of a shorter row).
    {
        let simple = recv_type
            .rsplit('.')
            .next()
            .unwrap_or(recv_type)
            .split('<')
            .next()
            .unwrap_or(recv_type)
            .trim();
        if table.interfaces.iter().any(|i| i.name == simple) {
            let sig = param_types
                .iter()
                .map(|t| crate::native::semantic::normalize_param_type(t))
                .collect::<Vec<_>>()
                .join(",");
            if let Some(slot) =
                crate::native::semantic::iface_slot_for(table, simple, method, &sig)
            {
                return Some(slot);
            }
            // Generic declarations and substituted call sites spell
            // parameters differently; with a single member of that
            // name the signature is still unambiguous.
            if let Some(slot) =
                crate::native::semantic::iface_slot_by_name(table, simple, method)
            {
                return Some(slot);
            }
        }
    }
    // The exact receiver entry, when the caller knows it, is the most
    // precise answer available — it names one class rather than a
    // spelling shared with same-named types.
    recv.and_then(|c| c.methods.iter().find(|m| m.name == method))
        .map(|m| m.vtable_slot)
        .or_else(|| {
            table
                .classes
                .iter()
                .filter(|c| c.name == recv_type || c.key.path == recv_type)
                .filter_map(|c| c.methods.iter().find(|m| m.name == method))
                .map(|m| m.vtable_slot)
                .next()
        })
        // An interface receiver declares its own members, but interface
        // declarations carry no vtable row (they have no body), so the
        // slot has to come from a class. Prefer one that actually
        // implements this interface.
        .or_else(|| {
            let simple = recv_type.rsplit('.').next().unwrap_or(recv_type);
            table
                .classes
                .iter()
                .find(|c| {
                    c.interfaces.iter().any(|(iname, _)| {
                        iname == recv_type || iname.rsplit('.').next() == Some(simple)
                    })
                })
                .and_then(|c| {
                    // Explicit interface implementation first: the
                    // method is registered under its qualified
                    // spelling (`int I.M()` -> `I.M`). C# gives it
                    // precedence over a same-named plain method when
                    // the call comes through the interface.
                    c.methods
                        .iter()
                        .find(|m| {
                            m.name.rsplit_once('.').is_some_and(|(qual, name)| {
                                name == method
                                    && (qual == recv_type
                                        || qual.rsplit('.').next() == Some(simple)
                                        || recv_type.rsplit('.').next() == Some(qual))
                            })
                        })
                        .or_else(|| c.methods.iter().find(|m| m.name == method))
                })
                .map(|m| m.vtable_slot)
        })
        .or_else(|| {
            // Interfaces still generate no row of their own, so a call
            // through an interface-typed receiver whose implementer is
            // not in this unit has no dispatch target to find. Fall
            // back to the same-named slot table-wide rather than
            // dropping the call: interfaces are C# and their members
            // must still be callable. Giving interfaces their own rows
            // until then this remains the one
            // place a slot can come from the wrong class, and it only
            // fires for interface receivers.
            let simple = recv_type.rsplit('.').next().unwrap_or(recv_type);
            let receiver_is_interface = table
                .interfaces
                .iter()
                .any(|i| i.name == recv_type || i.name.rsplit('.').next() == Some(simple))
                || table.classes.iter().any(|c| {
                    c.interfaces.iter().any(|(iname, _)| {
                        iname == recv_type || iname.rsplit('.').next() == Some(simple)
                    })
                });
            if !receiver_is_interface {
                return None;
            }
            table
                .classes
                .iter()
                .filter_map(|c| c.methods.iter().find(|m| m.name == method))
                .map(|m| m.vtable_slot)
                .next()
        })
}

#[derive(Clone, Copy)]
enum Fd {
    Stdout,
    Stderr,
}

#[derive(Clone, Copy)]
enum IntKind {
    Signed(u32),
    Unsigned(u32),
}

fn integer_kind(ty: &str) -> Option<IntKind> {
    Some(match crate::native::ir::lower::select::base_type_name(ty).as_str() {
        "int" | "Int32" => IntKind::Signed(32),
        "uint" | "UInt32" => IntKind::Unsigned(32),
        "long" | "Int64" | "nint" | "IntPtr" => IntKind::Signed(64),
        "ulong" | "UInt64" | "nuint" | "UIntPtr" => IntKind::Unsigned(64),
        "short" | "Int16" => IntKind::Signed(16),
        "ushort" | "UInt16" => IntKind::Unsigned(16),
        "byte" | "Byte" => IntKind::Unsigned(8),
        "sbyte" | "SByte" => IntKind::Signed(8),
        _ => return None,
    })
}

/// `D5` / `X4` / `F2` / `G`: one format character plus optional digits.
fn parse_numeric_spec(spec: &str) -> Option<(char, Option<u32>)> {
    let mut chars = spec.chars();
    let kind = chars.next()?;
    let rest = chars.as_str();
    let digits = if rest.is_empty() {
        None
    } else if rest.chars().all(|c| c.is_ascii_digit()) {
        Some(rest.parse().ok()?)
    } else {
        return None;
    };
    match kind {
        'D' | 'd' | 'X' | 'x' | 'F' | 'f' | 'G' | 'g' => Some((kind, digits)),
        _ => None,
    }
}

/// Render one printf argument to a UTF-16 string, applying a numeric
/// format. Unsupported specifiers and types report instead of printing a
/// different value.
fn render_arg_string(
    b: &mut Body<'_>,
    v: VReg,
    ty: &str,
    item: &crate::utilities::format::FormatItem,
) -> Result<VReg, LowerError> {
    let base = crate::native::ir::lower::select::base_type_name(ty);
    let unsupported = |what: &str| {
        LowerError::UnsupportedConstruct(format!(
            "printf: {what} is not implemented for '{ty}' (value would be printed differently)"
        ))
    };
    if is_string_type(ty) {
        if item.format.is_some() {
            return Err(LowerError::UnsupportedConstruct(
                "printf: a format specifier cannot be applied to a string".to_owned(),
            ));
        }
        return Ok(v);
    }
    if let Some(kind) = integer_kind(ty) {
        let (signed, bits) = match kind {
            IntKind::Signed(b) => (true, b),
            IntKind::Unsigned(b) => (false, b),
        };
        let Some(spec) = item.format.as_deref() else {
            return Ok(b.int_to_string(v, signed));
        };
        let Some((kind, digits)) = parse_numeric_spec(spec) else {
            return Err(LowerError::UnsupportedConstruct(format!(
                "printf: unsupported integer format ':{spec}'"
            )));
        };
        match kind {
            'D' | 'd' => {
                let n = digits.unwrap_or(1);
                if n > 31 {
                    return Err(LowerError::UnsupportedConstruct(format!(
                        "printf: D precision {n} is above the 31-digit formatter"
                    )));
                }
                Ok(b.int_to_string_min(v, signed, n.max(1)))
            }
            'X' | 'x' => {
                let n = digits.unwrap_or(1);
                if n > 64 {
                    return Err(LowerError::UnsupportedConstruct(format!(
                        "printf: X precision {n} is above the 64-digit formatter"
                    )));
                }
                Ok(b.int_to_hex_string(v, bits, kind == 'X', n.max(1)))
            }
            'F' | 'f' => {
                let n = digits.unwrap_or(2);
                if n > 4096 {
                    return Err(LowerError::UnsupportedConstruct(format!(
                        "printf: F precision {n} is unreasonably large"
                    )));
                }
                let s = b.int_to_string(v, signed);
                if n == 0 {
                    Ok(s)
                } else {
                    let lit = b.utf16_literal(&format!(".{}", "0".repeat(n as usize)));
                    let zeros = b.build_utf16_string(lit.0, lit.1);
                    Ok(b.concat_strings(s, zeros))
                }
            }
            'G' | 'g' => {
                if digits.is_some() {
                    return Err(LowerError::UnsupportedConstruct(format!(
                        "printf: G precision is not implemented (':{spec}')"
                    )));
                }
                Ok(b.int_to_string(v, signed))
            }
            _ => Err(LowerError::UnsupportedConstruct(format!(
                "printf: unsupported integer format ':{spec}'"
            ))),
        }
    } else if matches!(base.as_str(), "bool" | "Boolean" | "char" | "Char") {
        if item.format.is_some() {
            return Err(LowerError::UnsupportedConstruct(format!(
                "printf: a format specifier cannot be applied to '{ty}'"
            )));
        }
        b.value_to_string(v, &base).ok_or_else(|| unsupported("this format"))
    } else if matches!(
        base.as_str(),
        "float" | "Single" | "double" | "Double" | "decimal" | "Decimal"
    ) {
        Err(LowerError::UnsupportedConstruct(format!(
            "printf: '{ty}' formatting needs a shortest-round-trip formatter, which is not implemented"
        )))
    } else {
        Err(unsupported("this type"))
    }
}

pub(super) fn printf(args: &[Expr], fd: Fd, b: &mut Body<'_>, ctx: &Ctx<'_>) -> Result<VReg, LowerError> {
    use crate::utilities::format::{Align, Segment};
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
    // each alive until its write segment consumes it. Each segment reads its
    // argument's static type to choose decimal, hex, padding or string
    // rendering; the type is spelling-agnostic like the element paths.
    let vals = eval_args(rest, b, ctx)?;
    // Spill the evaluated arguments into a temporary array: keeping every
    // argument live across the per-segment string encoding overflows the
    // allocator (a UTF-8 encode is ~100 instructions), and a value was
    // corrupted once five arguments were live. Loading one at a time keeps
    // liveness flat; evaluation order is unchanged.
    let nargs = vals.len();
    let boxed = b.fresh();
    b.push_raw(Instr::Alloc { dst: boxed, bytes: 16 + 8 * nargs as u64 });
    for (i, v) in vals.iter().enumerate() {
        b.push_raw(Instr::StoreField { base: boxed, off: 16 + 8 * i as u32, src: *v });
    }
    for segment in segments {
        match segment {
            Segment::Text(text) => match fd {
                Fd::Stdout => {
                    let (off, len) = b.string_literal(&text);
                    b.push_raw(Instr::WriteStr { off, len });
                }
                Fd::Stderr => {
                    let (off, len) = b.string_literal(&text);
                    b.push_raw(Instr::EWriteStr { off, len });
                }
            },
            Segment::Arg(item) => {
                let v = b.fresh();
                b.push_raw(Instr::LoadField {
                    dst: v,
                    base: boxed,
                    off: 16 + 8 * item.index as u32,
                });
                let ty = static_type_of(&rest[item.index], b, ctx).unwrap_or_default();
                let known = !ty.is_empty()
                    && !matches!(
                        crate::native::ir::lower::select::base_type_name(&ty).as_str(),
                        "object" | "Object" | "unknown" | "_"
                    );
                if !known {
                    // Inference failed (qualified static calls are the
                    // common case). The pre-existing behavior is an integer
                    // write; keep it, but never pretend to apply a format.
                    if item.format.is_none() && item.align.is_none() {
                        match fd {
                            Fd::Stdout => b.push_raw(Instr::WriteInt { src: v }),
                            Fd::Stderr => b.push_raw(Instr::EWriteInt { src: v }),
                        }
                        continue;
                    }
                    return Err(LowerError::UnsupportedConstruct(
                        "printf: a format needs a known static argument type".to_owned(),
                    ));
                }
                // Fast path: a plain non-string argument keeps printf's
                // established integer write (bools and chars included, as
                // 1/0 and their code unit). A formatted argument opts into
                // C#-style rendering below.
                if item.format.is_none() && item.align.is_none() && !is_string_type(&ty) {
                    let decimal_ok = match integer_kind(&ty) {
                        Some(IntKind::Signed(_)) => true,
                        Some(IntKind::Unsigned(bits)) => bits != 64,
                        None => matches!(
                            crate::native::ir::lower::select::base_type_name(&ty).as_str(),
                            "bool" | "Boolean"
                        ),
                    };
                    if decimal_ok {
                        match fd {
                            Fd::Stdout => b.push_raw(Instr::WriteInt { src: v }),
                            Fd::Stderr => b.push_raw(Instr::EWriteInt { src: v }),
                        }
                        continue;
                    }
                }
                if matches!(fd, Fd::Stderr) {
                    return Err(LowerError::UnsupportedConstruct(
                        "eprintf: formatted or non-integer arguments are not implemented"
                            .to_owned(),
                    ));
                }
                let s = render_arg_string(b, v, &ty, &item)?;
                let s = match item.align {
                    Some(Align::Right(width)) | Some(Align::Left(width)) => {
                        if width > 4096 {
                            return Err(LowerError::UnsupportedConstruct(format!(
                                "printf: alignment width {width} is unreasonably large"
                            )));
                        }
                        let left = matches!(item.align, Some(Align::Left(_)));
                        b.pad_string(s, width as u64, left)
                    }
                    None => s,
                };
                let enc = b.utf8_string_of(s);
                b.raw_vreg(Instr::WriteString { src: enc });
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

    // Restructure the blocks into one per state. The original blocks
    // are linearized in order; each Suspend splits the sequence, so
    // the instructions before a Suspend are the tail of state S and
    // the ones after it (in the same or a later block) are the head of
    // state S+1.
    //
    // Only the common shape is handled: Suspend instructions in the
    // linear flow, with no branch across a suspend point. Each Suspend
    // is the last instruction of its block, and the next block starts
    // the next state.

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

    // The final terminator comes from the last original block.
    let last_term = f
        .blocks
        .last()
        .map(|b| b.term.clone())
        .unwrap_or(Terminator::Ret(None));

    // The new layout is sequential: block 0 is the prologue and the
    // dispatch, blocks 1..N are the states, and the last block ends
    // the function.

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
    // The dispatch branches are built after all the state blocks exist.
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
            // the segment; it goes back with the correct state value.
            let suspend_state = f.fresh_vreg();
            state_block.instrs.push(Instr::Const {
                dst: suspend_state,
                imm: s as i64,
            });
            state_block.instrs.push(Instr::Suspend { state: suspend_state });
            // After Suspend, the backend returns to the scheduler.
            // When resumed, it re-enters at the dispatch block. The IR
            // marks the suspend with an Exit (the backend intercepts
            // Suspend before reaching the terminator).
            state_block.term = Terminator::Exit;
        } else {
            // Final state: ends with the original terminator.
            state_block.term = last_term.clone();
        }
        new_blocks.push(state_block);
    }

    // The dispatch is a simplified one: the backend reads the state and
    // jumps directly, so the IR only needs the state blocks laid out
    // sequentially. The dispatch block falls through to state 0, and
    // re-entry at the right state goes through the task frame.
    //
    // That is enough for the backend to implement the state machine: it
    // reads the state from the task frame and jumps to block (state+1).

    f.blocks = new_blocks;
    f.is_async = true;

    f
}

/// Find a generic method with `arity` type parameters and `argc`
/// parameters that has a body. Prefers the enclosing class.
fn find_generic_method<'a>(
    ctx: &Ctx<'a>,
    name: &str,
    arity: usize,
    argc: usize,
) -> Option<(&'a crate::native::semantic::ClassInfo, &'a crate::models::ast::method::MethodDecl)> {
    let mut candidates: Vec<&crate::models::ClassDecl> = Vec::new();
    for c in ctx.classes {
        if c.name == ctx.current_class {
            candidates.push(c);
        }
    }
    for c in ctx.classes {
        if !candidates.iter().any(|x| std::ptr::eq(*x, *c)) {
            candidates.push(c);
        }
    }
    for c in candidates {
        let Some(info) = ctx.table.class(&c.name) else {
            continue;
        };
        for m in &c.members {
            if let crate::models::ClassMember::Method(m) = m {
                if m.name == name
                    && m.generic_params.len() == arity
                    && m.params.len() == argc
                    && m.body.is_some()
                {
                    return Some((info, m));
                }
            }
        }
    }
    None
}


/// Pack the trailing arguments of a `params` call into one array.
///
/// `void Sum(params int[] xs)` called as `Sum(1, 2, 3)` used to pass
/// three vregs to a one-parameter callee, so the callee read argument 1
/// as the array and faulted. The receiver (when present) sits at index
/// 0 of `argv` and is not part of the explicit list.
pub(super) fn pack_params(
    argv: &mut Vec<VReg>,
    recv: usize,
    mi: &crate::native::semantic::MethodInfo,
    b: &mut Body<'_>,
    // The caller passed ONE argument for the params slot and it is
    // already an array of the parameter's type (`Sum(new int[]{…})`):
    // leave it alone. Without this the array handle would be packed as
    // an ELEMENT of a new array.
    last_is_array: bool,
) {
    if !mi.has_params || mi.param_types.is_empty() {
        return;
    }
    let fixed = mi.param_types.len() - 1;
    if argv.len().saturating_sub(recv) == fixed {
        // No values for the params slot: C# supplies an EMPTY array.
        let nv = b.const_i64(0);
        let total = b.const_i64(24);
        let obj = b.fresh();
        b.push_raw(Instr::AllocDyn { dst: obj, bytes: total });
        b.push_raw(Instr::StoreField { base: obj, off: 0, src: nv });
        argv.push(obj);
        return;
    }
    if last_is_array {
        return;
    }
    let elem_ty = super::select::base_type_name(&mi.param_types[fixed]);
    let w = crate::native::ir::lower::span::span_elem_width(&elem_ty) as i64;
    let extra: Vec<VReg> = argv[recv + fixed..].to_vec();
    let n = extra.len() as i64;
    let nv = b.const_i64(n);
    let total = b.const_i64(24 + n * w);
    let obj = b.fresh();
    b.push_raw(Instr::AllocDyn { dst: obj, bytes: total });
    b.push_raw(Instr::StoreField { base: obj, off: 0, src: nv });
    for (i, v) in extra.iter().enumerate() {
        b.push_raw(Instr::StoreField {
            base: obj,
            off: (16 + i as i64 * w) as u32,
            src: *v,
        });
    }
    argv.truncate(recv + fixed);
    argv.push(obj);
}

/// Invoke a delegate VALUE (single or combined) with `args`.
///
/// Shared by the local and field paths: the kind dispatch and the
/// multicast walk exist once, so a field delegate behaves exactly like
/// a local one.
pub(super) fn invoke_delegate(
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
    obj: VReg,
    args: &[Expr],
) -> Result<VReg, LowerError> {
    let target = b.fresh();
    b.push_raw(Instr::LoadField {
        dst: target,
        base: obj,
        off: 0,
    });
    let kind = b.fresh();
    b.push_raw(Instr::LoadField {
        dst: kind,
        base: obj,
        off: 8,
    });
    let zero = b.const_i64(0);
    // Arguments are evaluated ONCE here: a combined delegate runs
    // every subscriber with the same values, and evaluating inside
    // each branch would multiply their side effects.
    let arg_v = eval_args(args, b, ctx)?;
    let two = b.const_i64(2);
    let is_multi = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: crate::native::ir::CmpOp::Eq,
        a: kind,
        b: two,
    });
    let multi_b = b.make_block();
    let single_b = b.make_block();
    let join_b = b.make_block();
    let home = b.fresh();
    b.terminate(Terminator::Branch {
        cond: Some(is_multi),
        if_true: multi_b,
        if_false: single_b,
    });
    // kind == 2: the value is `[0, 2, head]` and head is the first
    // node of a `[next, delegate]` list. Run every subscriber in
    // subscription order; the last result wins (C# returns the last
    // handler's value).
    b.switch_to(multi_b);
    {
        let cur = b.fresh();
        b.push_raw(Instr::LoadField {
            dst: cur,
            base: obj,
            off: 24,
        });
        let body_b = b.make_block();
        let after_call = b.make_block();
        // Do-while: a combined delegate always has at least one node
        // (remove-all answers null, never an empty list), so the first
        // node is read unconditionally. Checking for an empty list first
        // also left `home` unassigned on the exit edge.
        b.goto(body_b);
        b.switch_to(body_b);
        let d = b.fresh();
        b.push_raw(Instr::LoadField {
            dst: d,
            base: cur,
            off: 8,
        });
        let t2 = b.fresh();
        b.push_raw(Instr::LoadField {
            dst: t2,
            base: d,
            off: 0,
        });
        let k2 = b.fresh();
        b.push_raw(Instr::LoadField {
            dst: k2,
            base: d,
            off: 8,
        });
        let closure2 = b.emit(Instr::Cmp {
            dst: VReg(0),
            op: crate::native::ir::CmpOp::Ne,
            a: k2,
            b: zero,
        });
        let c2 = b.make_block();
        let p2 = b.make_block();
        b.terminate(Terminator::Branch {
            cond: Some(closure2),
            if_true: c2,
            if_false: p2,
        });
        b.switch_to(c2);
        {
            let mut argv = vec![d];
            argv.extend(arg_v.iter().copied());
            let dst = b.fresh();
            b.push_raw(Instr::CallFnAddr { dst, target: t2, args: argv });
            b.mov(home, dst);
            b.goto(after_call);
        }
        b.switch_to(p2);
        {
            let dst = b.fresh();
            b.push_raw(Instr::CallFnAddr {
                dst,
                target: t2,
                args: arg_v.clone(),
            });
            b.mov(home, dst);
            b.goto(after_call);
        }
        b.switch_to(after_call);
        let nx = b.fresh();
        b.push_raw(Instr::LoadField {
            dst: nx,
            base: cur,
            off: 0,
        });
        b.mov(cur, nx);
        let more = b.emit(Instr::Cmp {
            dst: VReg(0),
            op: crate::native::ir::CmpOp::Ne,
            a: cur,
            b: zero,
        });
        b.terminate(Terminator::Branch {
            cond: Some(more),
            if_true: body_b,
            if_false: join_b,
        });
    }
    // kind 0/1: a single delegate, as before.
    b.switch_to(single_b);
    let is_closure = b.emit(Instr::Cmp {
        dst: VReg(0),
        op: crate::native::ir::CmpOp::Ne,
        a: kind,
        b: zero,
    });
    let closure_b = b.make_block();
    let plain_b = b.make_block();
    b.terminate(Terminator::Branch {
        cond: Some(is_closure),
        if_true: closure_b,
        if_false: plain_b,
    });
    b.switch_to(closure_b);
    {
        let mut argv = vec![obj];
        argv.extend(arg_v.iter().copied());
        let dst = b.fresh();
        b.push_raw(Instr::CallFnAddr { dst, target, args: argv });
        b.mov(home, dst);
        b.goto(join_b);
    }
    b.switch_to(plain_b);
    {
        let argv = arg_v;
        let dst = b.fresh();
        b.push_raw(Instr::CallFnAddr { dst, target, args: argv });
        b.mov(home, dst);
        b.goto(join_b);
    }
    b.switch_to(join_b);
    return Ok(home);
}
