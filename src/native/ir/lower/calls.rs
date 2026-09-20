//! AST to IR body lowering.

use super::super::{Block, Instr, IrFunction, LowerError, Terminator, VReg};
use crate::models::{ClassMember, Expr, Literal};
use crate::native::ir::Ctx;
use crate::native::semantic::{ClassInfo, FieldInfo, MethodInfo, TypeTable};
use super::{body::*, select::*, types::*};

// Direct-call function index, cached per lowering run: building it
// scans every class and method, so rebuilding per call site is
// O(calls x methods) — fine for small units, hours for a full
// DataContractSerialization. The cache keys on the table address
// (stable across one run) and is cleared at each lowering entry, so
// sequential builds and parallel test threads never share entries.
thread_local! {
    static FN_INDEX_CACHE: std::cell::RefCell<
        Option<(usize, std::rc::Rc<std::collections::BTreeMap<String, usize>>)>,
    > = std::cell::RefCell::new(None);
}

/// Drop the cached index: called once per lowering entry, before any
/// call site resolves.
pub(crate) fn clear_fn_index_cache() {
    FN_INDEX_CACHE.with(|c| *c.borrow_mut() = None);
}pub(super) fn call(callee: &str, args: &[Expr], b: &mut Body<'_>, ctx: &Ctx<'_>) -> Result<VReg, LowerError> {
    if callee == "printf" || callee == "print" || callee == "put_str" {
        return printf(args, Fd::Stdout, b, ctx);
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
    if callee == "sizeof" {
        // Return a placeholder size (8 bytes for a pointer/word). The
        // argument is always a TYPE, never a value — no evaluation
        // (a type name like `int[]` is undeclared as a variable).
        return Ok(b.const_i64(8));
    }
    if callee == "typeof" {
        // `typeof` is a C# operator; return 0 as a placeholder (same:
        // type argument, no evaluation).
        return Ok(b.const_i64(0));
    }
    if callee == "__reftype" || callee == "__makeref" || callee == "__refvalue" {
        // Undocumented TypedReference keywords (`__reftype(tr)` takes a
        // typed reference): evaluate args for side effects, yield zero
        // (mirrors the scan's known-list; real typing is future work).
        let _ = eval_args(args, b, ctx)?;
        return Ok(b.const_i64(0));
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
        let argv = eval_args(args, b, ctx)?;
        let dst = b.fresh();
        b.push_raw(Instr::CallImport {
            dst,
            import: index as u32,
            args: argv,
        });
        return Ok(dst);
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
        // Argument count mismatch is tolerated — generic methods and
        // overloads may not match exactly; omitted optional params are
        // filled from the selected overload's folded defaults.
        // Plain call inside the class: may still resolve to an inherited
        // member whose visibility must be checked against this class —
        // enforced on the SELECTED overload (see above).
        ctx.check_method_access(mi)?;

        let this = VReg(0);
        // Fill omitted optionals on the EXPLICIT args (`this` is not a
        // declared param), then prepend the receiver.
        let mut explicit = eval_args(args, b, ctx)?;
        let full_arity = mi.param_types.len();
        fill_default_args(&mut explicit, mi, b);
        let mut argv = vec![this];
        argv.extend(explicit);

        let own = ctx
            .classes
            .iter()
            .find(|c| c.name == ctx.current_class)
            .map(|c| {
                c.members
                    .iter()
                    .any(|m| matches!(m, ClassMember::Method(m) if m.name == name))
            })
            .unwrap_or(false);

        let dst = b.fresh();
        let map = functions_index(ctx);
        // Bodyless own method (`extern`/partial-no-impl declared here):
        // no function exists — yield zero rather than mis-calling the
        // first overload or `@start` via the fallbacks below.
        if own && !mi.has_body {
            return Ok(b.const_i64(0));
        }
        if own {
            // Function indices mirror declaration order with @start at 0.
            // The selected overload's full arity keys the lookup (omitted
            // optionals were filled above); a miss falls back to the
            // first declared overload.
            let prefix = format!("{}::{name}#", ctx.current_class);
            let idx = map
                .get(&format!("{prefix}{full_arity}"))
                .copied()
                .or_else(|| map.range(prefix..).next().map(|(_, &i)| i))
                .unwrap_or(0);
            b.push_raw(Instr::CallFn {
                dst,
                fidx: idx as u32,
                args: argv,
            });
        } else {
            b.push_raw(Instr::CallVirt {
                dst,
                slot: canonical_slot(
                    ctx.table,
                    ctx.current_info,
                    ctx.current_class,
                    name,
                ) as u32,
                recv: this,
                args: argv,
            });
        }
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
        match static_function_index_for(ctx, &mi.declaring_class, name, full) {
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
        match static_function_index_for(ctx, &mi.declaring_class, name, full) {
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
) -> std::rc::Rc<std::collections::BTreeMap<String, usize>> {
    // Index MUST mirror the emission loop below (same classes, same
    // skips, same order, same +1): direct-call lookups resolve to
    // positions in the final functions vector. In particular, entries
    // overwrite (last-wins) so same-arity overloads (`Task#2` vs
    // `bool#2`) don't desync the map from the vector — `or_insert`
    // (first-wins on distinct count) silently misroutes every call
    // after the first collision.
    let key = ctx.table as *const TypeTable as usize;
    if let Some((cached_key, map)) = FN_INDEX_CACHE.with(|c| c.borrow().clone()) {
        if cached_key == key {
            return map;
        }
    }
    let mut built = std::collections::BTreeMap::new();
    let mut next = 1usize; // +1 skips @start, like the emission loop.
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
                built.insert(fname, next);
                next += 1;
            }
        }
    }
    let map = std::rc::Rc::new(built);
    FN_INDEX_CACHE.with(|c| *c.borrow_mut() = Some((key, map.clone())));
    map
}

/// Index of the direct function for `class.name#arity`, if it exists.
/// Used for static-style calls (`C.M(...)`) which are lowered as a direct
/// `CallFn` (no vtable, no receiver).
pub(super) fn function_index_for(
    ctx: &Ctx<'_>,
    class_name: &str,
    method: &str,
    arity: usize,
) -> Option<u32> {
    let fname = format!("{class_name}::{method}#{arity}");
    functions_index(ctx).get(&fname).copied().map(|i| i as u32)
}

/// Like `function_index_for`, but also walks the base-class chain: static
/// methods are inherited, so `Derived.M(...)` resolves to the function
/// declared on the base that defines it.
pub(super) fn static_function_index_for(
    ctx: &Ctx<'_>,
    class_name: &str,
    method: &str,
    arity: usize,
) -> Option<u32> {
    let mut seen: std::collections::BTreeSet<String> = Default::default();
    let mut stack = vec![class_name.to_string()];
    while let Some(name) = stack.pop() {
        if !seen.insert(name.clone()) {
            continue;
        }
        if let Some(fidx) = function_index_for(ctx, &name, method, arity) {
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
    if let Some(info) = recv {
        if let Some(m) = info.methods.iter().find(|m| m.name == method) {
            return m.vtable_slot;
        }
    }
    table
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
        .unwrap_or(0)
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