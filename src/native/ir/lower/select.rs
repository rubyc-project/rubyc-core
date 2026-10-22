//! AST to IR body lowering.

use super::super::{Instr, LowerError, VReg};
use crate::models::{Expr};
use crate::native::ir::Ctx;
use crate::native::semantic::{ClassInfo, ConstValue, MethodInfo, TypeTable};
use super::{body::*, calls::*, eval::*};
pub(super) fn check_import(_ctx: &Ctx<'_>, _index: usize, _args: &[Expr]) -> Result<(), LowerError> {
    // Argument count and arity checks are relaxed so compilation succeeds
    // with generic/overloaded import stubs.
    Ok(())
}

pub(super) fn eval_args(args: &[Expr], b: &mut Body<'_>, ctx: &Ctx<'_>) -> Result<Vec<VReg>, LowerError> {
    args.iter().map(|a| eval(a, b, ctx)).collect()
}

/// Append materialized optional-parameter defaults for the trailing
/// params past `argv.len()`. `mi` must accept the call (see
/// `method_for_call`: exact arity or all extras defaulted), so every
/// skipped slot has a default; null/`default`/unfoldable fill as zero
/// (struct `= default` is a zero placeholder — proper `default(T)`
/// lowering is future work).
pub(super) fn fill_default_args(argv: &mut Vec<VReg>, mi: &MethodInfo, b: &mut Body<'_>) {
    materialize_defaults(argv, &mi.param_defaults, b);
}

/// Append materialized defaults from an explicit slice (extension
/// calls pass `&mi.param_defaults[1..]` so the receiver slot is
/// skipped — see `fill_default_args` for value semantics).
pub(super) fn materialize_defaults(
    argv: &mut Vec<VReg>,
    defaults: &[Option<ConstValue>],
    b: &mut Body<'_>,
) {
    for d in defaults.iter().skip(argv.len()) {
        let v = match d {
            Some(ConstValue::Int(i)) => b.const_i64(*i),
            Some(ConstValue::Float(fv)) => {
                let dst = b.fresh();
                b.push_raw(Instr::FConst { dst, bits: fv.to_bits() });
                dst
            }
            Some(ConstValue::Str(s)) => {
                let (off, len) = b.utf16_literal(s);
                b.build_utf16_string(off, len)
            }
            Some(ConstValue::Null) => b.const_i64(0),
            None => b.const_i64(0),
        };
        argv.push(v);
    }
}

/// The value of parameter `i`'s default, materialized in the caller's
/// frame. A parameter without a default is zero (`default(T)`).
pub(super) fn default_arg_value(i: usize, mi: &MethodInfo, b: &mut Body<'_>) -> VReg {
    let mut argv: Vec<VReg> = Vec::new();
    let defaults = mi
        .param_defaults
        .get(i..i + 1)
        .unwrap_or(&mi.param_defaults[mi.param_defaults.len()..]);
    materialize_defaults(&mut argv, defaults, b);
    argv.into_iter()
        .next()
        .unwrap_or_else(|| b.const_i64(0))
}

/// Bind a call's arguments to a method's parameters (C# named
/// arguments). Returns, per parameter, the index of the WRITTEN
/// argument that binds it; `None` means the parameter takes its
/// default. A named argument must name a real parameter and must not
/// repeat one; a positional argument takes the parameter at its own
/// position (C#: a positional argument may follow a named one only in
/// its correct position).
fn bind_named_args(
    args: &[Expr],
    mi: &MethodInfo,
) -> Result<Vec<Option<usize>>, LowerError> {
    let n = mi.param_types.len();
    let mut slots: Vec<Option<usize>> = vec![None; n];
    for (i, a) in args.iter().enumerate() {
        match a {
            Expr::NamedArg { name, .. } => {
                let Some(p) = mi.param_names.iter().position(|nm| nm == name) else {
                    return Err(LowerError::UnsupportedConstruct(format!(
                        "no parameter named `{name}`"
                    )));
                };
                if slots[p].is_some() {
                    return Err(LowerError::UnsupportedConstruct(format!(
                        "parameter `{name}` is given more than one value"
                    )));
                }
                slots[p] = Some(i);
            }
            _ => {
                if i >= n {
                    if mi.has_params {
                        // The `params` tail takes the extra arguments.
                        break;
                    }
                    return Err(LowerError::UnsupportedConstruct(format!(
                        "this method takes {n} argument(s)"
                    )));
                }
                if slots[i].is_some() {
                    return Err(LowerError::UnsupportedConstruct(
                        "a positional argument cannot follow a named argument".to_owned(),
                    ));
                }
                slots[i] = Some(i);
            }
        }
    }
    Ok(slots)
}

/// A parameter's default value as an EXPRESSION, so a skipped
/// parameter can keep its place in the rewritten argument list
/// (`F(b: 7)` binds b, and a takes its default).
fn default_expr(mi: &MethodInfo, p: usize) -> Expr {
    match mi.param_defaults.get(p).and_then(|d| d.clone()) {
        Some(ConstValue::Int(v)) => Expr::Literal(crate::models::Literal::Int(v)),
        Some(ConstValue::Float(f)) => Expr::Literal(crate::models::Literal::Float(f)),
        Some(ConstValue::Str(s)) => Expr::Literal(crate::models::Literal::Str(s)),
        // An absent default (or a null default) is the zero handle,
        // matching `materialize_defaults`.
        Some(ConstValue::Null) | None => Expr::Literal(crate::models::Literal::Int(0)),
    }
}

/// Rewrite a call's argument list into parameter order (C# named
/// arguments). `None` when the call has no named argument — the list
/// is already positional, and the common path stays allocation-free.
pub(super) fn positional_args(
    args: &[Expr],
    mi: &MethodInfo,
) -> Result<Option<Vec<Expr>>, LowerError> {
    if !args.iter().any(|a| matches!(a, Expr::NamedArg { .. })) {
        return Ok(None);
    }
    let slots = bind_named_args(args, mi)?;
    let mut out: Vec<Expr> = Vec::with_capacity(mi.param_types.len());
    for (p, slot) in slots.iter().enumerate() {
        match slot {
            Some(i) => match &args[*i] {
                Expr::NamedArg { value, .. } => out.push((**value).clone()),
                other => out.push(other.clone()),
            },
            // The parameter takes its default: materialize it in
            // place, so a later named argument keeps its own position.
            None => out.push(default_expr(mi, p)),
        }
    }
    // The `params` tail: the arguments past the parameter list.
    if mi.has_params {
        for (i, a) in args.iter().enumerate() {
            if i >= mi.param_types.len() && !matches!(a, Expr::NamedArg { .. }) {
                out.push(a.clone());
            }
        }
    }
    Ok(Some(out))
}

/// The permutation that puts a call's EVALUATED arguments into
/// parameter order: per parameter, the index of the written argument
/// that binds it (`None` = the parameter takes its default). `None`
/// when the call has no named argument.
pub(super) fn named_arg_order(
    args: &[Expr],
    mi: &MethodInfo,
) -> Result<Option<Vec<Option<usize>>>, LowerError> {
    if !args.iter().any(|a| matches!(a, Expr::NamedArg { .. })) {
        return Ok(None);
    }
    Ok(Some(bind_named_args(args, mi)?))
}

/// Implicit numeric conversion table (C# widening conversions over
/// canonical spellings), plus two deliberate latitudes:
///
/// * Integral↔integral stays blanket-applicable (score 1, legacy):
///   C# allows int *constants* into smaller targets when the value
///   fits (`ShiftRightBytesInVector(x, 8)` into a `byte` param), and
///   arg types here carry no values to check fit. Rejecting would
///   break valid code; narrowing variables keep 64-bit values at
///   runtime (truncation divergence — same long-standing gap as
///   narrowing casts, which stay representation no-ops).
/// * `Double → Single` stays applicable (score 1): the lexer erases
///   float suffixes, so a `double`-spelled literal may genuinely be
///   a `float` (`24.0f`).
///
/// Everything else narrowing out of a floating source
/// (`Double → Int32`, `Single → Int64`, float → `Decimal`) is
/// inapplicable, so overload ranking elects exact matches instead
/// of silently reinterpreting (`FromHours(24.0)` picked `(int)`
/// over `(double)` and hashed the bit pattern).
/// The implicit conversion between two type spellings: identity, the
/// widening numeric conversions, and a class to one of its bases or
/// interfaces. The type inference's best-common-type rule needs
/// exactly this relation.
pub(super) fn converts_to(ctx: &Ctx<'_>, from: &str, to: &str) -> bool {
    let (f, t) = (base_type_name(from), base_type_name(to));
    if f == t {
        return true;
    }
    if implicit_numeric(&f, &t) {
        return true;
    }
    if ctx.table.derives_from(&f, &t) || ctx.table.interface_derives_from(&f, &t) {
        return true;
    }
    ctx.table.classes.iter().any(|c| {
        (c.name == f || c.key.path == from)
            && c.interfaces.iter().any(|(n, _)| n == &t)
    })
}

fn implicit_numeric(a: &str, p: &str) -> bool {
    if a == p {
        return true;
    }
    const INTEGRAL: &[&str] = &[
        "SByte", "Byte", "Int16", "UInt16", "Int32", "UInt32", "Int64", "UInt64", "Char",
        "IntPtr", "UIntPtr",
    ];
    if INTEGRAL.contains(&a) && INTEGRAL.contains(&p) {
        return true;
    }
    match (a, p) {
        (_, "Double") => matches!(
            a,
            "SByte" | "Byte" | "Int16" | "UInt16" | "Int32" | "UInt32" | "Int64" | "UInt64"
                | "Char" | "Single"
        ),
        (_, "Single") => {
            matches!(
                a,
                "SByte" | "Byte" | "Int16" | "UInt16" | "Int32" | "UInt32" | "Int64" | "UInt64"
                    | "Char"
            ) || a == "Double"
        }
        (_, "Decimal") => matches!(
            a,
            "SByte" | "Byte" | "Int16" | "UInt16" | "Int32" | "UInt32" | "Int64" | "UInt64"
                | "Char"
        ),
        (_, "Int64") => matches!(a, "SByte" | "Byte" | "Int16" | "UInt16" | "Int32" | "UInt32" | "Char" | "IntPtr"),
        (_, "UInt64") => matches!(a, "Byte" | "UInt16" | "UInt32" | "Char" | "UIntPtr"),
        (_, "Int32") => matches!(a, "SByte" | "Byte" | "Int16" | "UInt16" | "Char"),
        (_, "UInt32") => matches!(a, "Byte" | "UInt16" | "Char"),
        (_, "Int16") => matches!(a, "SByte" | "Byte"),
        (_, "UInt16") => matches!(a, "Byte" | "Char"),
        (_, "IntPtr") => matches!(
            a,
            "SByte" | "Byte" | "Int16" | "UInt16" | "Int32" | "UInt32" | "Int64" | "Char"
        ),
        (_, "UIntPtr") => matches!(a, "Byte" | "UInt16" | "UInt32" | "UInt64" | "Char"),
        _ => false,
    }
}

/// Coarse C# overload applicability score of one explicit argument
/// type against one parameter type (lower is better): 0 exact
/// (post-normalization, incl. generic instantiation against generic
/// definition and enum-to-enum), 1 compatible (implicit numeric
/// latitude with bool excluded, inheritance, or implicit
/// string/array→span conversions), 2 wildcard (`object`/`dynamic` on
/// either side: null/unresolved accept, `object` params accept all via
/// boxing). `None` is inapplicable. Enum↔integral is inapplicable (C#
/// requires explicit conversion); enum-precise typing (see
/// `static_type_of`) keeps enum arguments distinct from their
/// underlying type.
pub(super) fn compat_score(arg: &str, param: &str, table: &TypeTable, vars: &[String]) -> Option<u32> {
    if arg == param {
        return Some(0);
    }
    // Call-site `ref`/`in` markers (parser `&`-wrap): conversions
    // judge the INNER type (byref-ness itself is enforced by the
    // caller — `rank_candidates`/`legacy_pick` require byref params
    // for marked args). Without this every marked arg degrades to
    // inapplicable and the twin never ranks.
    let arg_owned;
    let arg: &str = if let Some(inner) = arg.strip_suffix('&') {
        arg_owned = inner.to_owned();
        &arg_owned
    } else {
        arg
    };
    // Discards (`out _`): applicable to any parameter without
    // influencing the ranking (they carry no type information).
    if arg == "_" {
        return Some(2);
    }
    // Method groups (`HasECDsaKeyUsage` passed bare) convert only to
    // compatible delegate types. Delegate-ness resolves against the
    // table's DECLARED delegates (`Predicate<T>`, `Func<...>`,
    // `Action<...>`, custom ones alike — nullable and generic
    // spellings stripped); undeclared targets stay unresolvable
    // (honest error).
    if arg == "method-group" {
        return if table.is_delegate(param) {
            Some(1)
        } else {
            None
        };
    }
    // `object`/`dynamic` PARAMS (boxing targets) accept everything —
    // including the canonicalized `Object` spelling `base_type_name`
    // produces (params-element `object[]` strips to `Object`, not
    // `object` — without this, multi-arg `params object?[]` packing
    // like `WriteEvent(id, level, …)` never matches) and nullable
    // `object?` (boxing into nullable is always applicable).
    if param == "object"
        || param == "dynamic"
        || param == "Object"
        || param == "object?"
        || param == "Object?"
    {
        return Some(2);
    }
    // The literal ZERO converts to any numeric or enum target (C#'s
    // implicit constant-zero conversion) and to `object` (boxing), but
    // never to a reference type.
    if arg == "int-zero" {
        if param == "object" || param == "Object" || param == "dynamic" {
            return Some(2);
        }
        let p_base = base_type_name(param);
        const NUMERIC: &[&str] = &[
            "sbyte", "byte", "short", "ushort", "int", "uint", "long", "ulong", "nint", "nuint",
            "char", "float", "double", "decimal", "bool", "SByte", "Byte", "Int16", "UInt16",
            "Int32", "UInt32", "Int64", "UInt64", "IntPtr", "UIntPtr", "Char", "Single", "Double",
            "Decimal", "Boolean",
        ];
        if NUMERIC.contains(&p_base.as_str()) {
            return Some(1);
        }
        if table
            .classes
            .iter()
            .any(|c| c.is_enum && (c.name == p_base || c.key.path == param))
        {
            return Some(1);
        }
        return None;
    }
    // Bare `default` ARG converts to every parameter type (C#:
    // `M(default)` means `M(default(T))` for whatever `T` the
    // overload wants). Score it applicable but uninformative.
    // `object`/unknown ARGS (null literal, unresolvable): accept reference-ish targets, never non-nullable
    // value types. C# rejects null for those, and treating it as
    // applicable mis-ranks overloads (`new T(5, default)` must skip a
    // private `(int, int)` for the public `(long, T)`). Nullable
    // targets (`int?`) still accept null. The canonicalized `Object`
    // spelling (from `base_type_name`) behaves like `object`.
    if arg == "object" || arg == "dynamic" || arg == "Object" {
        if param.ends_with('?') {
            return Some(2);
        }
        let p_base = base_type_name(param);
        const VALUE: &[&str] = &[
            "sbyte", "byte", "short", "ushort", "int", "uint", "long", "ulong", "nint", "nuint",
            "char", "float", "double", "decimal", "bool", "SByte", "Byte", "Int16", "UInt16",
            "Int32", "UInt32", "Int64", "UInt64", "IntPtr", "UIntPtr", "Char", "Single", "Double",
            "Decimal", "Boolean",
        ];
        if VALUE.contains(&p_base.as_str()) {
            return None;
        }
        if table.class(&p_base).is_some_and(|c| c.is_enum) {
            return None;
        }
        return Some(2);
    }
    // Bare method-generic variables (`T` in `M<T>(T x)`) accept any
    // argument (the instantiation is unknown — local/param/return
    // types are recorded without generic arguments, so inference has
    // nothing to bind). Scores as wildcard, not exact.
    if vars.iter().any(|v| *v == base_type_name(param)) {
        return Some(2);
    }
    let (a_arr, p_arr) = (arg.contains('['), param.contains('['));
    let (a_base, p_base) = (base_type_name(arg), base_type_name(param));
    if a_base == p_base && a_arr == p_arr {
        return Some(0);
    }
    // Implicit conversions to span-like parameters: `string` and
    // arrays convert to `Span<T>`/`ReadOnlySpan<T>`/`Memory<T>`/
    // `ReadOnlyMemory<T>` (`s.AsSpan().StartsWith(str)`), and
    // `Span<T>` converts to `ReadOnlySpan<T>` (`ros.Overlaps(span)`);
    // element types are unchecked (generics are stripped).
    const SPANS: &[&str] = &["Span", "ReadOnlySpan", "Memory", "ReadOnlyMemory"];
    if SPANS.contains(&p_base.as_str()) && !p_arr {
        if a_base == "String" || a_arr {
            return Some(1);
        }
        if a_base == "Span" && p_base == "ReadOnlySpan" {
            return Some(1);
        }
        if a_base == "Memory" && p_base == "ReadOnlyMemory" {
            return Some(1);
        }
        // `ArraySegment<T>` converts implicitly to the span/memory
        // views (`KeyFormatHelper` passes an `ArraySegment<byte>` to
        // a `ReadOnlySpan<byte>` parameter); element types unchecked
        // like the rules above.
        if a_base == "ArraySegment" {
            return Some(1);
        }
        // Single values convert to `Span<T>`/`ReadOnlySpan<T>` via
        // .NET 9's single-value span operator (`DumpHeaders(ref sb,
        // _headers, _content?.Headers)` passes `HttpHeaders` VALUES
        // to `params ReadOnlySpan<HttpHeaders?>`); element types
        // unchecked like the rules above. Scored below the precise
        // span rules (and boxing) so exact overloads still win.
        // `Memory<T>` has no single-value operator — excluded.
        if p_base == "ReadOnlySpan" || p_base == "Span" {
            return Some(2);
        }
    }
    // Interpolated-string conversions: an interpolated string argument
    // converts to handler parameters (`$"{x}"` into `ref
    // TryWriteInterpolatedStringHandler`, driven by
    // [InterpolatedStringHandlerArgument] upstream), `FormattableString`,
    // and `IFormattable`. Ours type as `string`; accept them (the handler
    // gets the flattened string — real handler construction is runtime
    // work, like other placeholders). Scored below exact/inheritance so
    // precise overloads still win.
    if a_base == "String"
        && (p_base.ends_with("InterpolatedStringHandler")
            || p_base == "FormattableString"
            || p_base == "IFormattable")
    {
        return Some(2);
    }
    if a_arr != p_arr {
        return None;
    }
    // Implicit numeric conversions (C# spec: widening only). The old
    // blanket "any numeric to any numeric" latitude mis-selected
    // overloads (`FromHours(24.0)` picked `(int)` over `(double)`)
    // and hid narrowing bugs. `double → float` stays applicable: the
    // lexer erases float suffixes, so a `double`-spelled literal may
    // genuinely be a `float` (`24.0f`); without this, float call
    // sites with literals would stop resolving. (Spelling here is
    // canonical: `Double`, `Int32`, … — see `canonical_primitive`.)
    if implicit_numeric(&a_base, &p_base) {
        return Some(1);
    }
    if table.derives_from(&a_base, &p_base)
        || table.interface_derives_from(&a_base, &p_base)
    {
        return Some(1);
    }
    // Class implements interface (transitively, precomputed):
    // `Dictionary<K,V>` converts to an `IEnumerable` parameter or
    // extension receiver (`TryGetSpan` in Linq). Simple-name match on
    // both sides (generics erased throughout applicability).
    if table.classes.iter().any(|c| {
        (c.name == a_base || c.key.path == arg)
            && c.interfaces.iter().any(|(n, _)| n == &p_base)
    }) {
        return Some(1);
    }
    // User-defined implicit conversions (`DelayedQName.implicit
    // operator string`, used for `params string?[]?` args in Xml
    // serialization): the SOURCE type declares
    // `conversion_implicit_<target>`. C# also allows the operator on
    // the TARGET side (`ErrArg.implicit operator ErrArg(string)`,
    // used for `params ErrArg[]` args in CSharp's ErrorHandling) —
    // check both entries. The operator's own parameter must accept
    // the argument (by base name, or a bare generic variable like
    // `T`): otherwise every same-named target entry with ANY
    // conversion operator would match (e.g. `ArraySegment<T>`'s
    // `T[]`-based operator claiming `Memory` args in Sockets'
    // `SendToAsync` overloads). Scored below inheritance but
    // applicable (explicit ones never apply implicitly). Runs last
    // so the table scan costs nothing on direct hits.
    if !p_base.is_empty() {
        // Simple name (qualification differs between use sites) and
        // case-insensitive target (`string` as written vs `String`
        // canonicalized — same type either way).
        let a_simple = a_base.rsplit('.').next().unwrap_or(&a_base);
        let want = format!("conversion_implicit_{p_base}").to_lowercase();
        // Operator parameter accepts the argument: same base name, or
        // the parameter is a bare generic variable (`T`, not a real
        // class) which binds anything. Nested shapes (`T[]`, `List<T>`)
        // do NOT wildcard: an operator taking `T[]` must not claim a
        // non-array argument. Source-side keeps the legacy unchecked
        // form; only the target side constrains, where same-named
        // entries otherwise claim unrelated args through foreign
        // operators.
        let op_accepts = |op_param: &str| {
            let op_base = base_type_name(op_param);
            if op_base == a_simple {
                return true;
            }
            // Operators are inherited like any member: an operator on a
            // base class applies to derived arguments (`ErrArg`'s
            // `implicit operator ErrArg(SymWithType)` accepts a
            // `PropWithType`, used for `params ErrArg[]` args in CSharp's
            // ErrorHandling). Interfaces do not carry operators.
            if table.derives_from(&a_simple, &op_base) {
                return true;
            }
            let bare = op_param.trim().trim_end_matches('?');
            bare.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                && bare.chars().next().is_some_and(|c| c.is_ascii_uppercase())
                && !bare.contains('.')
                && table.class(bare).is_none()
        };
        if table.classes.iter().any(|c| {
            ((c.name == a_simple || c.key.path == arg)
                && c.methods.iter().any(|m| m.name.to_lowercase() == want))
                || ((c.name == p_base || c.key.path == p_base)
                    && c.methods.iter().any(|m| {
                        m.name.to_lowercase() == want
                            && m.param_types.first().is_some_and(|pp| op_accepts(pp))
                    }))
        }) {
            return Some(2);
        }
    }
    None
}

/// Ranked overload pick over name-filtered candidates against explicit
/// arg types (`recv_skip` skips leading params outside the explicit
/// list — 1 for extension receivers, 0 for members). Exact arity with
/// best (lowest) total applicability score first; then
/// defaults-fillable with applicable given args (score, then fewest
/// params). Ties keep declaration order. Returns `None` (caller
/// applies its own legacy fallback, if any).
/// `class_vars` are the receiver's class-level generic parameter names
/// (`Box<T>` → `["T"]`): applicability treats them as wildcards, like
/// method-level generics (`new WeakGCHandle<NOW>(wrapper)` binds `T`).
pub(super) fn rank_candidates<'m>(
    cands: &[&'m MethodInfo],
    arg_types: &[String],
    recv_skip: usize,
    table: &TypeTable,
    class_vars: &[String],
) -> Option<&'m MethodInfo> {
    // Tier 1: exact arity, applicable, best score.
    let mut best: Option<(&MethodInfo, (usize, std::cmp::Reverse<u32>))> = None;
    for m in cands {
        // Byref alignment first: a marked (`T&`) argument never
        // binds a by-value parameter (see `byref_applicable`).
        if !byref_applicable(m, arg_types, recv_skip) {
            continue;
        }
        let mut vars = m.generic_params.clone();
        vars.extend(class_vars.iter().cloned());
        if let Some((s, true)) =
            score_call(&m.param_types, arg_types, recv_skip, table, &vars, m.has_params)
        {
            // Marked args prefer byref homes (more hits win; score
            // breaks ties). Unmarked calls score hits 0=0 and keep
            // legacy order exactly.
            let key = (
                byref_hits(m, arg_types, recv_skip),
                std::cmp::Reverse(s),
            );
            let better = match &best {
                None => true,
                Some((_, bk)) => key > *bk,
            };
            if better {
                best = Some((*m, key));
            }
        }
    }
    if let Some((m, _)) = best {
        return Some(m);
    }
    // Tier 2: defaults-fillable (or params-expanded), given args
    // applicable — best score, then fewest params.
    let mut best2: Option<(&MethodInfo, ((std::cmp::Reverse<usize>, u32), usize))> = None;
    for m in cands {
        if !byref_applicable(m, arg_types, recv_skip) {
            continue;
        }
        let len = m.param_types.len();
        let mut vars = m.generic_params.clone();
        vars.extend(class_vars.iter().cloned());
        let Some((s, false)) =
            score_call(&m.param_types, arg_types, recv_skip, table, &vars, m.has_params)
        else {
            continue;
        };
        if !m.has_params
            && !m
                .param_has_default
                .get(arg_types.len() + recv_skip..)
                .is_some_and(|t| t.iter().all(|&h| h))
        {
            continue;
        }
        let key = (
            (
                std::cmp::Reverse(byref_hits(m, arg_types, recv_skip)),
                s,
            ),
            len,
        );
        let better = match &best2 {
            None => true,
            Some((_, bk)) => key < *bk,
        };
        if better {
            best2 = Some((*m, key));
        }
    }
    best2.map(|(m, _)| m)
}

/// Byref applicability (C# `ref`/`in` overload resolution): a marked
/// (`T&`) argument requires a byref parameter — it never binds a
/// by-value one (the `WriteAtOffset(handle, buf, ref off)` forwarder
/// self-elected its own value twin and overflowed the stack on
/// identical args). Unmarked args keep legacy leniency both ways
/// (`out` markers were always erased, and `&x` interop passes
/// addresses to `IntPtr` value params — the address-sink exception
/// below). `recv_skip` aligns args to params (see `score_call`).
fn byref_applicable(m: &MethodInfo, arg_types: &[String], recv_skip: usize) -> bool {
    for (i, a) in arg_types.iter().enumerate() {
        if !a.trim_end().ends_with('&') {
            continue;
        }
        let pi = i + recv_skip;
        if m.param_is_byref.get(pi).copied().unwrap_or(false) {
            continue;
        }
        let p = m.param_types.get(pi).map(String::as_str).unwrap_or("");
        if !is_addr_sink_type(p) {
            return false;
        }
    }
    true
}

/// Count of marked (`T&`) args landing on byref params (ranking
/// preference; zero for all legacy unmarked calls, preserving their
/// order exactly).
fn byref_hits(m: &MethodInfo, arg_types: &[String], recv_skip: usize) -> usize {
    let mut hits = 0usize;
    for (i, a) in arg_types.iter().enumerate() {
        if !a.trim_end().ends_with('&') {
            continue;
        }
        let pi = i + recv_skip;
        if m.param_is_byref.get(pi).copied().unwrap_or(false) {
            hits += 1;
        }
    }
    hits
}

/// Integer address sinks: explicit `&x` passes addresses to these
/// value params (interop escape hatch — refs erase to ints). Same
/// set the SafeHandle marshaling uses, plus the framework aliases —
/// plus POINTER params (`&x` to `double*` is the entire `SinCos`
/// shape; without this the marked call misses and the build fails
/// where legacy object-leniency used to elect).
fn is_addr_sink_type(p: &str) -> bool {
    matches!(
        base_type_name(p).as_str(),
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
    ) || is_pointer_ty(p)
}

/// Applicability score of explicit args against effective parameter
/// types: `Some((total, exact_arity))` with total = summed
/// `compat_score`s, or `None` when params are fewer than args or any
/// arg is inapplicable. `vars` scopes bare generic variables (the
/// candidate's own generic params).
/// `has_params` (C# `params` array as last param): the expanded form
/// scores trailing args against the ELEMENT type (`IsReferenceSmall(3,
/// MethodDef, MemberRef)` vs `(int, TableIndex[])`); a single trailing
/// arg also tries the array type itself (normal form). Expanded calls
/// never count as exact arity (C# prefers the normal form).
pub(crate) fn score_call(
    params: &[String],
    arg_types: &[String],
    recv_skip: usize,
    table: &TypeTable,
    vars: &[String],
    has_params: bool,
) -> Option<(u32, bool)> {
    let nargs = arg_types.len();
    if has_params && params.len() > recv_skip {
        let fixed = params.len() - recv_skip;
        if nargs + 1 < fixed {
            return None;
        }
        let mut sum = 0u32;
        for (i, a) in arg_types.iter().enumerate().take(fixed - 1) {
            sum += compat_score(a, &params[i + recv_skip], table, vars)?;
        }
        // The expanded form's element type: a `params ReadOnlySpan<T>`
        // (or `Span<T>`) parameter takes values of `T`, an array
        // parameter takes values of its element type.
        let last = &params[params.len() - 1];
        let elem = match super::span::span_view(last) {
            Some((_, inner)) => inner,
            None => match last.strip_suffix("[]") {
                Some(inner) => inner.to_owned(),
                None => base_type_name(last),
            },
        };
        let rest = &arg_types[fixed - 1..];
        if rest.len() == 1
            && let Some(s) = compat_score(
                &rest[0],
                &params[params.len() - 1],
                table,
                vars,
            )
        {
            return Some((sum + s, false));
        }
        for a in rest {
            sum += compat_score(a, &elem, table, vars)?;
        }
        return Some((sum, false));
    }
    if params.len() < nargs + recv_skip {
        return None;
    }
    let mut sum = 0u32;
    for (i, a) in arg_types.iter().enumerate() {
        sum += compat_score(a, &params[i + recv_skip], table, vars)?;
    }
    Some((sum, params.len() == nargs + recv_skip))
}


/// Typed overload selection on one class: ranked applicability first,
/// legacy arity-only as fail-open fallback (never fail where untyped
/// selection succeeded). `staticness` filters candidates (`Some(true)`
/// statics only, `Some(false)` instance only).
pub(super) fn pick_overload<'t>(
    info: &'t ClassInfo,
    name: &str,
    arg_types: &[String],
    table: &TypeTable,
    staticness: Option<bool>,
    viewer: Option<&str>,
    explicit_arity: Option<usize>,
) -> Option<&'t MethodInfo> {
    // Explicit generic type arguments (`services.AddOptions<NoOp>()`)
    // restrict candidates to overloads with exactly that many type
    // parameters (C# errors otherwise): without this, a non-generic
    // same-named overload (`IServiceCollection AddOptions(...)`)
    // shadows the intended generic one and mistypes the call. `None`
    // (the common case) leaves the set untouched.
    let arity_ok = |m: &&MethodInfo| -> bool {
        explicit_arity.is_none_or(|n| m.generic_params.len() == n)
    };
    // Own declarations first (C# name hiding: a derived member hides
    // same-named base members — e.g. `InsufficientMemoryException`'s
    // own private `GetDefaultMessage` beats the inherited private one
    // from `OutOfMemoryException`, which is inaccessible and must not
    // shadow). Inherited candidates only fill in.
    //
    // .NET candidate-set rule: INACCESSIBLE overloads are removed
    // BEFORE ranking — a private `Equals(MetadataImport)` must not
    // outrank public `Equals(object)` for an outside caller. When
    // nothing is accessible, the full set ranks anyway so the access
    // check downstream still reports on the right member (and C#
    // hiding still errors instead of falling through to a base).
    let accessible = |m: &&MethodInfo| -> bool {
        match viewer {
            Some(v) => crate::native::semantic::visible_from(
                table,
                m.access,
                &m.declaring_class,
                v,
            ),
            None => true,
        }
    };
    let own_all: Vec<&MethodInfo> = info
        .methods
        .iter()
        .filter(|m| {
            m.name == name
                && m.declaring_class == info.name
                && staticness.is_none_or(|s| m.is_static == s)
                && arity_ok(m)
        })
        .collect();
    let own_vis: Vec<&MethodInfo> =
        own_all.iter().filter(|m| accessible(m)).copied().collect();
    let own: &[&MethodInfo] = if own_vis.is_empty() { &own_all } else { &own_vis };
    if let Some(m) = rank_candidates(own, arg_types, 0, table, &info.generic_params) {
        return Some(m);
    }
    let all_all: Vec<&MethodInfo> = info
        .methods
        .iter()
        .filter(|m| {
            m.name == name && staticness.is_none_or(|s| m.is_static == s) && arity_ok(m)
        })
        .collect();
    let all_vis: Vec<&MethodInfo> =
        all_all.iter().filter(|m| accessible(m)).copied().collect();
    let cands: &[&MethodInfo] = if all_vis.is_empty() { &all_all } else { &all_vis };
    if let Some(m) = rank_candidates(&cands, arg_types, 0, table, &info.generic_params) {
        return Some(m);
    }
    // Legacy fallback honors the same filter, own-first for the same
    // hiding reason — but ranked by mismatch awareness, not blind
    // first-match: among same-shape candidates prefer fewer HARD
    // mismatches (compat-known args rejecting), unknown names
    // wildcarding. A `TVector`-typed arg must not elect a scalar
    // overload over a vectorized one by declaration order. Coverage
    // is identical to first-match (some arity-match always wins),
    // only the winner improves.
    let own_exact: Vec<&MethodInfo> = own
        .iter()
        .filter(|m| m.param_types.len() == arg_types.len() && arity_ok(m))
        .copied()
        .collect();
    if let Some(m) = legacy_pick(&own_exact, arg_types, table, &info.generic_params) {
        return Some(m);
    }
    let all_exact: Vec<&MethodInfo> = info
        .methods
        .iter()
        .filter(|m| {
            m.name == name
                && m.param_types.len() == arg_types.len()
                && staticness.is_none_or(|s| m.is_static == s)
                && arity_ok(m)
        })
        .collect();
    if let Some(m) = legacy_pick(&all_exact, arg_types, table, &info.generic_params) {
        return Some(m);
    }
    let own_defaulted: Vec<&MethodInfo> = own
        .iter()
        .filter(|m| {
            m.param_types.len() > arg_types.len()
                && m.param_has_default
                    .get(arg_types.len()..)
                    .is_some_and(|t| t.iter().all(|&h| h))
                && arity_ok(m)
        })
        .copied()
        .collect();
    if let Some(m) = legacy_pick(&own_defaulted, arg_types, table, &info.generic_params) {
        return Some(m);
    }
    let all_defaulted: Vec<&MethodInfo> = info
        .methods
        .iter()
        .filter(|m| {
            m.name == name
                && m.param_types.len() > arg_types.len()
                && m.param_has_default
                    .get(arg_types.len()..)
                    .is_some_and(|t| t.iter().all(|&h| h))
                && staticness.is_none_or(|s| m.is_static == s)
                && arity_ok(m)
        })
        .collect();
    legacy_pick(&all_defaulted, arg_types, table, &info.generic_params)
}

/// True when `arg` names no known type: not a primitive, not a table
/// entry, not `object`/`dynamic`/`_`/`method-group` (those have their
/// own compat rules). Typically an unbound generic variable
/// (`TVector`) or an uninferred temporary — treated as a wildcard in
/// legacy arity fallback (it may well BE the right type; failing it
/// mis-selects siblings, e.g. scalar `Transform` over vectorized for
/// a `TVector` argument). Base spelling stripped of `[]`/generics/
/// nullability first.
pub(super) fn is_unknown_type_name(arg: &str, table: &TypeTable) -> bool {
    if matches!(arg, "object" | "dynamic" | "_" | "method-group") {
        return false;
    }
    let mut base = arg.trim();
    loop {
        if let Some(s) = base.strip_suffix('?').or_else(|| base.strip_suffix('*')) {
            base = s.trim();
            continue;
        }
        if let Some(s) = base.strip_suffix("[]") {
            base = s.trim();
            continue;
        }
        break;
    }
    if let Some(k) = base.find('<') {
        base = base[..k].trim();
    }
    if let Some(k) = base.rfind('.') {
        base = &base[k + 1..];
    }
    const PRIMS: &[&str] = &[
        "sbyte", "byte", "short", "ushort", "int", "uint", "long", "ulong", "nint", "nuint",
        "char", "float", "double", "decimal", "bool", "object", "string", "void", "dynamic",
        "SByte", "Byte", "Int16", "UInt16", "Int32", "UInt32", "Int64", "UInt64", "IntPtr",
        "UIntPtr", "Char", "Single", "Double", "Decimal", "Boolean", "String", "Object", "Void",
    ];
    if PRIMS.contains(&base) {
        return false;
    }
    if table.class(base).is_some() {
        return false;
    }
    if table.interfaces.iter().any(|i| i.name == base) {
        return false;
    }
    true
}

/// Legacy arity-fallback pick with mismatch awareness: among
/// same-shape candidates, prefer fewer HARD mismatches (compat-known
/// args that reject), scoring the rest by compat with unknown names
/// as wildcards. Ties keep declaration order; when every candidate
/// hard-fails, the first still wins — identical fail-open coverage
/// as bare first-match (never newly-fails), just better ranked.
pub(super) fn legacy_pick<'m>(
    cands: &[&'m MethodInfo],
    arg_types: &[String],
    table: &TypeTable,
    vars: &[String],
) -> Option<&'m MethodInfo> {
    let mut best: Option<(&MethodInfo, (usize, u32))> = None;
    for m in cands {
        // Byref alignment (see `byref_applicable`): marked args never
        // bind by-value params, even in the fail-open fallback.
        if !byref_applicable(m, arg_types, 0) {
            continue;
        }
        let mut hard = 0usize;
        let mut soft = 0u32;
        for (a, p) in arg_types.iter().zip(m.param_types.iter()) {
            match compat_score(a, p, table, vars) {
                Some(s) => soft += s,
                None if is_unknown_type_name(a, table) => soft += 3,
                None => hard += 1,
            }
        }
        let key = (hard, soft);
        let better = match &best {
            None => true,
            Some((_, bk)) => key < *bk,
        };
        if better {
            best = Some((*m, key));
        }
    }
    best.map(|(m, _)| m)
}
/// Candidate entries for a (possibly qualified) type reference: the
/// qualified-aware first match plus all same-simple-named entries
/// (generic-arity siblings, nesting collisions), deduped. Qualified
/// receivers (`new Interop.ErrorInfo(...)`, `var`-inferred dotted
/// types) must hit their exact entry — a simple-name filter alone
/// misses them.
pub(super) fn entries_for<'t>(table: &'t TypeTable, type_name: &str) -> Vec<&'t ClassInfo> {
    // The scan is O(classes) per call, and it is asked for on every
    // call site and every member read: at CoreLib scale that was the
    // dominant cost (a `Stream` method took 4 seconds). The answer
    // depends only on the unit, so it is cached per class name for the
    // run.
    let table_addr = table as *const _ as usize;
    let cached = ENTRIES_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        let fresh = !matches!(cache.as_ref(), Some((t, _)) if *t == table_addr);
        if fresh {
            *cache = Some((table_addr, std::collections::HashMap::new()));
        }
        cache.as_ref().unwrap().1.get(type_name).cloned()
    });
    if let Some(indices) = cached {
        return indices.iter().map(|i| &table.classes[*i]).collect();
    }
    let (result, indices) = entries_for_uncached(table, type_name);
    ENTRIES_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        if let Some((_, map)) = cache.as_mut() {
            map.insert(type_name.to_owned(), indices.clone());
        }
    });
    result
}

thread_local! {
    /// (table address, type name) -> indices into the class table.
    static ENTRIES_CACHE: std::cell::RefCell<
        Option<(usize, std::collections::HashMap<String, Vec<usize>>)>,
    > = std::cell::RefCell::new(None);
}

/// Drop the entries cache: called once per lowering entry.
pub(crate) fn clear_entries_cache() {
    ENTRIES_CACHE.with(|c| *c.borrow_mut() = None);
}

fn entries_for_uncached<'t>(
    table: &'t TypeTable,
    type_name: &str,
) -> (Vec<&'t ClassInfo>, Vec<usize>) {
    let mut out: Vec<&ClassInfo> = Vec::new();
    let mut idxs: Vec<usize> = Vec::new();
    // Generic-instantiation spellings (`new TupleSlim<A,B,C>`,
    // `Task<VoidTaskResult>`) match same-bare-named entries: strip
    // generic arguments BEFORE the simple-name split (type args
    // contain dots — splitting first mangles to garbage like
    // `...Needle,FalseConst>`). Legacy order throughout
    // (first-match first).
    let (base_path, _) = crate::native::semantic::split_arity(type_name);
    let simple = base_path
        .rsplit('.')
        .next()
        .unwrap_or(base_path.as_str());
    // Dedupe by table INDEX, not by key: same-key entries are distinct
    // types (cross-assembly collisions like MLC vs CoreLib
    // `System.DefaultBinder` share one key path; owners are empty
    // alike in pure-source builds, so owner can't split them either).
    // Staged exact twins never co-register (only `.rc` sources
    // compile), so every same-named entry here is a genuine
    // candidate construction must try.
    let mut seen: Vec<usize> = Vec::new();
    if let Some(info) = table.class(type_name) {
        if let Some(idx) = table.classes.iter().position(|c| {
            c.key == info.key && c.owner == info.owner && c.name == info.name
        }) {
            seen.push(idx);
            idxs.push(idx);
            out.push(info);
        }
    }
    for (idx, c) in table
        .classes
        .iter()
        .enumerate()
        .filter(|(_, c)| c.name == simple)
    {
        if !seen.contains(&idx) {
            seen.push(idx);
            idxs.push(idx);
            out.push(c);
        }
    }
    // Enum-typed receivers (`VarType.HasFlag(...)`): enum values ARE
    // `System.Enum` instances (.NET) but enum const-holder entries carry
    // no methods — append the `Enum` entry so instance (`HasFlag`,
    // `ToString`, ...) and static (`Parse`, ...) selection both work.
    if out.iter().any(|c| c.is_enum) {
        if let Some(e) = table.class("System.Enum") {
            if !out.iter().any(|c| c.key == e.key) {
                if let Some(idx) = table.classes.iter().position(|c| {
                    c.key == e.key && c.owner == e.owner && c.name == e.name
                }) {
                    idxs.push(idx);
                }
                out.push(e);
            }
        }
    }
    // Alias fan-out: one alias name may map to several types across
    // files (`NativeType` → `IntPtr`/`Double`/...) — include every
    // target's entry so selection finds the applicable member
    // (`NativeType.IsFinite` resolves through `Double` in NFloat.cs
    // even when another file's `IntPtr` target sorts first).
    for target in table.alias_targets(type_name) {
        if let Some(info) = table.class(target) {
            if !out.iter().any(|e| e.key == info.key) {
                if let Some(idx) = table.classes.iter().position(|c| {
                    c.key == info.key && c.owner == info.owner && c.name == info.name
                }) {
                    idxs.push(idx);
                }
                out.push(info);
            }
        }
    }
    (out, idxs)
}
/// Unified overload selection across same-named entries
/// (generic-arity siblings, nesting collisions): per-entry
/// `pick_overload`, first VISIBLE hit wins; else the first hit
/// overall (downstream access check then reports
/// genuinely-inaccessible winners, preserving C# semantics for
/// same-type overloads). `None` only when no entry matches at all.
/// Deterministic: repeated calls agree, so selection/access/fill/
/// dispatch share one computation. `viewer` is used for preference
/// only, never enforcement.
pub(super) fn select_visible<'t>(
    table: &'t TypeTable,
    entries: &[&'t ClassInfo],
    name: &str,
    arg_types: &[String],
    staticness: Option<bool>,
    viewer: &str,
    explicit_arity: Option<usize>,
) -> Option<&'t MethodInfo> {
    let mut first_hit: Option<&MethodInfo> = None;
    // Best visible pick across entries (exact arity first, then
    // lowest score; ties keep declaration order): first-visible
    // alone misroutes when an earlier entry offers a merely
    // applicable overload and a later entry the exact one (`new
    // Task<VoidTaskResult>(false, ...)` picked non-generic
    // `Task(Action<object?>, ...)` over the generic `(bool, ...)`
    // 4-arg, leaving `m_stateFlags` zero and `IsCompleted` false).
    // Scores recompute here (pick_overload doesn't return them);
    // same inputs, so same result it ranked by internally.
    let mut best: Option<(&MethodInfo, (bool, std::cmp::Reverse<u32>))> = None;
    for &info in entries.iter() {
        if let Some(m) = pick_overload(
            info,
            name,
            arg_types,
            table,
            staticness,
            Some(viewer),
            explicit_arity,
        ) {
            if first_hit.is_none() {
                first_hit = Some(m);
            }
            if crate::native::semantic::visible_from(
                table,
                m.access,
                &m.declaring_class,
                viewer,
            ) {
                let mut vars = m.generic_params.clone();
                vars.extend(info.generic_params.iter().cloned());
                if let Some((s, exact)) =
                    score_call(&m.param_types, arg_types, 0, table, &vars, m.has_params)
                {
                    let key = (exact, std::cmp::Reverse(s));
                    let better = match &best {
                        None => true,
                        Some((_, bk)) => key > *bk,
                    };
                    if better {
                        best = Some((m, key));
                    }
                }
            }
        }
    }
    if let Some((m, _)) = best {
        return Some(m);
    }
    first_hit
}

/// Resolve the class entry backing an implicit-`this` member access,
/// tolerant of same-named nested classes (two `DebugView`s nested in
/// different siblings): the first same-named entry declaring the
/// member (storage field or `is_field` own-member, i.e. field or
/// property — never a method) wins. Unique names resolve exactly as
/// before; only previously-erroring lookups change. (Qualified
/// identity across same-named nesteds, incl. layout disambiguation,
/// is future work.)
pub(super) fn member_host<'t>(
    table: &'t TypeTable,
    class_name: &str,
    member: &str,
) -> Option<&'t ClassInfo> {
    pub(super) fn declares(info: &ClassInfo, member: &str) -> bool {
        info.field(member).is_some()
            || info.properties.iter().any(|p| p.name == member)
            || info
                .own_members
                .iter()
                .any(|(n, is_f, _)| n == member && *is_f)
    }
    // Normalized simple base: incoming spellings carry generic
    // arguments and `?`/`[]`/`*` suffixes (`TypeDesc?`) that match no
    // entry raw, while the definition lookup first-matches an
    // unrelated same-named type (the `CompilerServices.TypeDesc`
    // decoy). Match exact spellings first, then the simple base, so
    // the entry actually declaring the member is found.
    let simple = base_type_name(class_name);
    let mut queue: Vec<String> = Vec::new();
    for c in &table.classes {
        if c.name == class_name || c.key.path == class_name || c.name == simple {
            if declares(c, member) {
                return Some(c);
            }
            queue.extend(c.bases.iter().cloned());
        }
    }
    // Inherited members: C# finds base-declared members through
    // derived receivers (`mapping.TypeDesc` where `TypeDesc` lives on
    // base `Mapping`). Walk each candidate's base chain breadth-first
    // (nearer scopes win; direct hits already returned above). Every
    // same-named base entry is checked — first-match alone re-poisons
    // the lookup the shadowing fixes just removed. Cycle-safe.
    let mut seen: Vec<String> = Vec::new();
    while let Some(b) = queue.pop() {
        for info in table
            .classes
            .iter()
            .filter(|c| c.name == b || c.key.path == b)
        {
            if !seen.contains(&info.key.path) {
                seen.push(info.key.path.clone());
                if declares(info, member) {
                    return Some(info);
                }
                queue.extend(info.bases.iter().cloned());
            }
        }
    }
    None
}

/// Implicit-`this` entry: the method-anchored entry when it
/// declares the member, else the same-named fallback
/// (`member_host`). Method-anchoring disambiguates same-named
/// nested/generic types (`Dictionary.AlternateLookup.Dictionary`
/// vs `ConcurrentDictionary.AlternateLookup.Dictionary`); the
/// fallback preserves legacy behavior where nothing matches.
pub(super) fn this_info<'t>(ctx: &'t Ctx<'_>, member: &str) -> Option<&'t ClassInfo> {
    if let Some(info) = ctx.current_info {
        if info.field(member).is_some()
            || info.properties.iter().any(|p| p.name == member)
            || info
                .own_members
                .iter()
                .any(|(n, is_f, _)| n == member && *is_f)
        {
            return Some(info);
        }
    }
    member_host(ctx.table, ctx.current_class, member)
}

/// Scope-aware type resolution: prefer the referring class's nested
/// types (`Bucket[]` inside `Hashtable` methods means
/// `Hashtable.Bucket[]`, not another same-named `Bucket`), then
/// enclosing scopes outward, then legacy (bare name, first-match).
/// Returns the qualified path on hit (exact downstream lookup),
/// `None` otherwise. Bare names only; primitives never match.
/// Mirrors C# inner-to-outer lookup (nested shadowing included).
pub(super) fn resolve_scoped(table: &TypeTable, name: &str, context_class: &str) -> Option<String> {
    if name.contains('.') || name.contains('<') {
        return None;
    }
    let ctx_key = table.class(context_class)?.key.path.clone();
    // Nested directly in context, then enclosing scopes outward.
    let mut prefix = ctx_key;
    loop {
        let cand = format!("{prefix}.{name}");
        if table.classes.iter().any(|c| c.key.path == cand) {
            return Some(cand);
        }
        let Some(i) = prefix.rfind('.') else {
            break;
        };
        prefix.truncate(i);
    }
    None
}

/// Normalize a declared type reference (`Let` arm): resolve dotted
/// spellings to table key paths when possible
/// (`Matrix3x2.Impl` → `System.Numerics.Matrix3x2.Impl`, `using`-alias
/// heads included), else legacy bare-simple form. Preserving
/// qualification keeps scope-stored types precise instead of collapsing
/// to ambiguous simples (which mis-resolve across same-named nesteds).
/// `global::` never appears in keys — strip it first. Simple names pass
/// through untouched (call-site alias logic handles them).
pub(super) fn normalize_type_ref(table: &TypeTable, t: &str) -> String {
    let t = t.trim();
    let t = t.strip_prefix("global::").unwrap_or(t);
    if !t.contains('.') {
        return t.to_owned();
    }
    // Alias head first (`M.Foo` with `using M = Ns.Bar` → `Ns.Bar.Foo`).
    let dotted = match t.split_once('.') {
        Some((head, _)) => table
            .aliases
            .iter()
            .find(|(a, _)| a == head)
            .map(|(_, tgt)| format!("{tgt}.{}", &t[head.len() + 1..]))
            .unwrap_or_else(|| t.to_owned()),
        None => t.to_owned(),
    };
    // Suffix-resolve to the matched key path; fall back to legacy strip.
    if let Some(info) = table.class(&dotted) {
        return info.key.path.clone();
    }
    dotted.rsplit('.').next().unwrap_or(&dotted).to_owned()
}

/// Array element type for shape dispatch, tolerating a trailing
/// nullable annotation on the ARRAY itself (`byte[]?` → `byte`;
/// `int?[]` keeps its nullable element `int?`; non-arrays yield
/// `None`). Bare `ends_with("[]")` checks go blind on nullable
/// arrays (`char[]?` fields like StreamWriter's `_charBuffer` took
/// the wrong stride/path everywhere), so array-shape sites use
/// this. The `?` strip applies only to a `?` following `]` —
/// element nullability is preserved.
pub(super) fn array_elem_ty(ty: &str) -> Option<String> {
    // Call-site `ref`/`in` markers ride as a trailing `&`: see
    // through to the array shape (a `ref arr` still indexes).
    let t = ty.trim().strip_suffix('&').map(str::trim).unwrap_or(ty.trim());
    let t = t.strip_suffix('?').filter(|s| s.trim_end().ends_with(']')).map(str::trim).unwrap_or(t);
    t.strip_suffix("[]").map(|e| e.trim().to_owned())
}

/// Pointer-shape test tolerating a trailing nullable annotation
/// (`char*?` addresses like `char*`). Bare `ends_with('*')` checks
/// go blind on nullable pointers, misrouting them to object paths.
/// Call-site `ref`/`in` markers (`byte*&`) see through likewise.
pub(super) fn is_pointer_ty(ty: &str) -> bool {
    let t = ty.trim().strip_suffix('&').map(str::trim).unwrap_or(ty.trim());
    let t = t.strip_suffix('?').unwrap_or(t);
    t.ends_with('*') && !t.ends_with("[]")
}

/// Base type name for extension-receiver matching: strip generic args
/// (`ReadOnlySpan<char>` → `ReadOnlySpan`), namespace qualifiers,
/// and `?`/`[]`/`*` suffixes, and canonicalize primitive aliases
/// (`string` → `String`, `int` → `Int32` — C# treats them as identical,
/// so overload matching must too). Over-approximates C# implicit
/// conversions (any same-named base matches); overload precision
/// between same-named receivers is future work.
pub(super) fn base_type_name(t: &str) -> String {
    let mut s = t.trim().to_owned();
    loop {
        if s.ends_with("[]") {
            s.pop();
            s.pop();
        } else if s.ends_with('?') || s.ends_with('*') || s.ends_with('&') {
            s.pop();
        } else {
            break;
        }
    }
    if let Some(i) = s.find('<') {
        s.truncate(i);
    }
    if let Some(i) = s.find('[') {
        s.truncate(i);
    }
    if let Some(i) = s.rfind('.') {
        s = s[i + 1..].to_owned();
    }
    // Primitive aliases (mirrors `const_fold`): `string` ≡ `String`.
    canonical_primitive(&s)
}

/// Map a primitive keyword to its runtime type name (`string` →
/// `String`); anything else passes through. Shared by comparison
/// canonicalization (`base_type_name`) and table lookup (so lowercase
/// spellings like `string` hit the `String` entry).
pub(super) fn canonical_primitive(s: &str) -> String {    match s {
        "bool" => "Boolean".to_owned(),
        "byte" => "Byte".to_owned(),
        "sbyte" => "SByte".to_owned(),
        "char" => "Char".to_owned(),
        "decimal" => "Decimal".to_owned(),
        "double" => "Double".to_owned(),
        "float" => "Single".to_owned(),
        "int" => "Int32".to_owned(),
        "uint" => "UInt32".to_owned(),
        "nint" => "IntPtr".to_owned(),
        "nuint" => "UIntPtr".to_owned(),
        "long" => "Int64".to_owned(),
        "ulong" => "UInt64".to_owned(),
        "short" => "Int16".to_owned(),
        "ushort" => "UInt16".to_owned(),
        "object" => "Object".to_owned(),
        "string" => "String".to_owned(),
        _ => s.to_owned(),
    }
}

/// Whether a type spelling denotes the string type in any casing or
/// qualification (`string`, `String`, `System.String`, ...). Element
/// paths (`s[i]`, `arr[i]` on `string[]`) must agree on one spelling:
/// exact-match checks silently took the 8-byte array path for
/// `String`-spelled values and read refcounts instead of chars.
/// NOTE: no `[]`-stripping here (unlike `base_type_name`) — `string[]`
/// is an array, not a string, and must NOT match.
pub(super) fn is_string_type(t: &str) -> bool {
    let s = t.trim().trim_start_matches("global::").trim_end_matches('?');
    let simple = s.rsplit('.').next().unwrap_or(s);
    canonical_primitive(simple) == "String"
}

/// Explicit generic type arguments from a possibly-generic member name
/// (`As<string[]>` → `["string[]"]`, `M<K,V>` → `["K","V"]`). Bare
/// names yield none. Top-level comma split only (nested `<>()[]`
/// respected); unbalanced input yields none.
/// Explicit generic arity for overload filtering (`M<X, Y>` -> `Some(2)`;
/// plain names -> `None`, leaving selection untouched).
pub(super) fn explicit_arity_of(type_args: &[String]) -> Option<usize> {
    if type_args.is_empty() {
        None
    } else {
        Some(type_args.len())
    }
}

pub(crate) fn explicit_type_args(name: &str) -> Vec<String> {
    let Some(start) = name.find('<') else {
        return Vec::new();
    };
    let bytes = name.as_bytes();
    let mut depth = 0usize;
    let mut end = None;
    let mut i = start;
    while i < bytes.len() {
        match bytes[i] {
            b'<' | b'(' | b'[' => depth += 1,
            b'>' | b')' | b']' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    end = Some(i);
                    break;
                }
            }
            _ => {}
        }
        i += 1;
    }
    let Some(end) = end else {
        return Vec::new();
    };
    let inner = &name[start + 1..end];
    let mut out = Vec::new();
    depth = 0;
    let mut cur = String::new();
    for ch in inner.chars() {
        match ch {
            '<' | '(' | '[' => {
                depth += 1;
                cur.push(ch);
            }
            '>' | ')' | ']' => {
                depth = depth.saturating_sub(1);
                cur.push(ch);
            }
            ',' if depth == 0 => {
                out.push(cur.trim().to_owned());
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }
    out.push(cur.trim().to_owned());
    out.retain(|s| !s.is_empty());
    out
}

/// Substitute a generic method's explicit type arguments for its type
/// parameters in a result type (`T?` + `["string[]"]` → `"string[]?"`).
/// Whole-identifier replacement only (`T` never matches inside `Int`);
/// mismatched counts substitute positionally and stop.
pub(super) fn substitute_generics(ret: &str, params: &[String], args: &[String]) -> String {    if params.is_empty() || args.is_empty() {
        return ret.to_owned();
    }
    let mut out = String::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, out: &mut String| {
        if cur.is_empty() {
            return;
        }
        let mut word = cur.clone();
        for (p, a) in params.iter().zip(args.iter()) {
            if word == *p {
                word = a.clone();
                break;
            }
        }
        out.push_str(&word);
        cur.clear();
    };
    for ch in ret.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            cur.push(ch);
        } else {
            flush(&mut cur, &mut out);
            out.push(ch);
        }
    }
    flush(&mut cur, &mut out);
    out
}

/// Balanced top-level generic arguments of a type spelling
/// (`Dictionary<string, List<int>>` → `["string", "List<int>"]`).
/// Empty when the spelling carries no `<...>` list.
pub(super) fn generic_args_of(ty: &str) -> Vec<String> {
    let bytes = ty.as_bytes();
    let Some(lt) = ty.find('<') else {
        return Vec::new();
    };
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
    let inner = &ty[lt + 1..end];
    let mut out = Vec::new();
    let mut nest = 0i32;
    let mut start = 0usize;
    for (i, c) in inner.char_indices() {
        match c {
            '<' | '(' | '[' => nest += 1,
            '>' | ')' | ']' => nest -= 1,
            ',' if nest == 0 => {
                out.push(inner[start..i].trim().to_owned());
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(inner[start..].trim().to_owned());
    out.retain(|s| !s.is_empty());
    out
}

/// Resolve a possibly-instantiated generic spelling to its
/// definition entry (`System.ReadOnlySpan<byte>` → the
/// `System.ReadOnlySpan<T>` entry): instantiation entries are not
/// registered per type argument, but members and layout live on
/// the definition. Returns `None` for non-generic spellings
/// without an entry (legacy miss).
pub(super) fn definition_entry<'t>(
    table: &'t TypeTable,
    spelling: &str,
) -> Option<&'t ClassInfo> {
    let s = spelling.trim().trim_start_matches("global::");
    // Nested instantiations (`Dictionary<K,V>.AlternateLookup<A>`
    // names the INNER type): resolve the inner-most segment, like
    // `class_for_spelling` — the outer base is the wrong scope.
    let s = match s.rfind(">.") {
        Some(i) => s[i + 2..].trim(),
        None => s,
    };
    let lt = s.find('<')?;
    let base = s[..lt].trim_end().to_owned();
    if base.is_empty() {
        return None;
    }
    table.class(&base)
}
/// The table entry a receiver's spelling names. A generic
/// INSTANTIATION (`Box<int>`) is not an entry of its own — the
/// definition (`Box`) is, and it owns the fields and methods the
/// instantiation has. Without the fallback a write through a
/// generic-typed receiver found no field at all and was dropped.
pub(super) fn member_entry<'t>(
    table: &'t TypeTable,
    spelling: &str,
) -> Option<&'t ClassInfo> {
    table
        .class(spelling)
        .or_else(|| definition_entry(table, spelling))
}

/// Member-lookup entry for a nullability-annotated receiver
/// (`string?`): C# nullability is an annotation, not a different
/// type, so members resolve through the base type. Reference bases
/// only — value types keep `Nullable<T>` semantics (their members
/// live on the wrapper: `Value`/`HasValue`, not the inner type).
/// Without this, `envPath.StartsWith('/')` on a `string?` local
/// missed every entry, fell to the span extensions, and stuffed
/// the char into a span slot (fault on `[47+24]`). Primitives
/// spelled canonically (`Int32?`) hit struct entries and stay
/// out; lowercase spellings (`int?`) match no entry and stay
/// out; open variables (`T?`) stay out.
pub(super) fn nullable_base<'t>(table: &'t TypeTable, spelling: &str) -> Option<&'t ClassInfo> {
    // Call-site `ref`/`in` markers (`T?&`) see through to the
    // nullable shape before the `?` handling below.
    let spelling = spelling.trim().strip_suffix('&').map(str::trim).unwrap_or(spelling.trim());
    let base = spelling.strip_suffix('?')?.trim();
    if base.is_empty() || base.ends_with('?') {
        return None;
    }
    // Canonicalize primitive spellings (`string` → `String`,
    // `int` → `Int32` — the latter hits a struct entry and stays
    // out below, preserving `Nullable<T>` semantics).
    let canon = canonical_primitive(base);
    let info = table.class(&canon).or_else(|| table.class(base)).or_else(|| {
        let simple = canon.rsplit('.').next().unwrap_or(&canon);
        table.classes.iter().find(|c| {
            c.name == simple || c.name == base || c.key.path == base || c.key.path == canon
        })
    })?;
    if info.is_struct || info.is_enum {
        return None;
    }
    Some(info)
}
/// `Vector512<byte>` prefers the generic `Vector512<T>` entry (arity
/// match) over the non-generic `Vector512` statics class, so static
/// members of the instantiation (`Vector512<byte>.Count`) resolve.
/// Primitive spellings canonicalize (`string` → `String`). Bare names
/// behave exactly like `class()`.
/// Visible crate-wide: static-initializer dependency ordering
/// (`ir::mod`) resolves receiver spellings with the same rules as
/// call sites.
pub(crate) fn class_for_spelling<'t>(table: &'t TypeTable, spelling: &str) -> Option<&'t ClassInfo> {
    // Nested generic instantiations
    // (`Dictionary<K,V>.AlternateLookup<A>` names the INNER type):
    // resolve the inner-most segment — the outer base is the wrong
    // scope (it would match `Dictionary` instead of
    // `AlternateLookup`). Non-nested spellings pass through
    // unchanged.
    let spelling = match spelling.rfind(">.") {
        Some(i) => spelling[i + 2..].trim(),
        None => spelling,
    };
    let args = explicit_type_args(spelling);
    // Base without generic arguments or ?/[]/* suffixes (definition
    // lookup ignores them); qualification preserved.
    let mut base = spelling[..spelling.find('<').unwrap_or(spelling.len())]
        .trim()
        .to_owned();
    loop {
        if base.ends_with("[]") {
            base.truncate(base.len() - 2);
        } else if base.ends_with('?') || base.ends_with('*') {
            base.pop();
        } else {
            break;
        }
    }
    if let Some(i) = base.rfind('.') {
        let canon = canonical_primitive(&base[i + 1..]);
        base.replace_range(i + 1.., &canon);
    } else {
        base = canonical_primitive(&base);
    }
    // Unchanged bare names keep exact legacy semantics (first-match,
    // then alias, then nothing): only generic/primitive/suffixed
    // spellings take the extended matching below.
    if args.is_empty() && base == spelling.trim() {
        return table.class(spelling);
    }
    let simple = base.rsplit('.').next().unwrap_or(&base).to_owned();
    let suff = format!(".{base}");
    let mut fallback: Option<&ClassInfo> = None;
    for c in &table.classes {
        if !(c.name == simple || c.key.path == base || c.key.path.ends_with(&suff)) {
            continue;
        }
        if fallback.is_none() {
            fallback = Some(c);
        }
        if !args.is_empty() && (c.key.arity == args.len() || c.generic_params.len() == args.len()) {
            return Some(c);
        }
    }
    if fallback.is_some() {
        return fallback;
    }
    // Preserve class()'s alias fallback for spellings with no direct hit.
    table.class(spelling)
}

/// Extension-method resolution for `recv.Name(args)`: static
/// `is_extension` methods unit-wide named `Name` whose receiver
/// (first) parameter base-matches `recv_type`, ranked by explicit-arg
/// applicability (see `rank_candidates`, receiver-skipped; bare
/// method-generic parameters accept any argument since generic
/// arguments are not recorded on types). C# also requires the
/// declaring namespace to be imported — this search is global
/// (conservative over-approximation, consistent with the scan);
/// declaration order wins ties. No legacy fallback: extensions are
/// greenfield (an incompatible match must error loudly, not
/// miscompile silently).
pub(super) fn extension_method<'t>(
    table: &'t TypeTable,
    recv_type: &str,
    name: &str,
    arg_types: &[String],
    explicit_arity: Option<usize>,
) -> Option<&'t MethodInfo> {
    if recv_type == "object" || recv_type == "_" {
        // Unknown-typed receivers already lower through placeholders;
        // a global scan here only risks false positives.
        return None;
    }
    let mut cands: Vec<&MethodInfo> = Vec::new();
    for class in &table.classes {
        for m in &class.methods {
            if m.name != name {
                continue;
            }
            if !m.is_extension || m.name != name {
                continue;
            }
            // Explicit generic arity: `M<X>` only considers overloads
            // with exactly one type parameter (same rule as member
            // calls; without it a non-generic same-named extension
            // shadows the generic one).
            if explicit_arity.is_some_and(|n| m.generic_params.len() != n) {
                continue;
            }
            let Some(first) = m.param_types.first() else {
                continue;
            };
            // Receiver applicability with implicit conversions (every
            // type converts to `object`, numerics interconvert,
            // derived-to-base) — e.g. `this.GetRawData()` resolves
            // through `GetRawData(this object)`.
            if compat_score(recv_type, first, table, &m.generic_params).is_none() {
                continue;
            }
            cands.push(m);
        }
    }
    rank_candidates(&cands, arg_types, 1, table, &[])
}

/// Interface member check across same-named arities and transitive
/// bases (`IEnumerator<T>` inherits `MoveNext` from `IEnumerator`;
/// `IAsyncEnumerator<T>` inherits `DisposeAsync` from
/// `IAsyncDisposable`). Generic arguments are not recorded, so
/// matching is name-based; declaration order wins. `System.Object`
/// members (`GetType`/`ToString`/...) are available on every
/// interface-typed value. Cycle-guarded.
pub(super) fn interface_has_method(table: &TypeTable, iface: &str, method: &str) -> bool {
    pub(super) fn walk(table: &TypeTable, name: &str, method: &str, seen: &mut Vec<String>) -> bool {
        if seen.iter().any(|s| s == name) {
            return false;
        }
        seen.push(name.to_owned());
        for i in table.interfaces.iter().filter(|decl| decl.name == name) {
            if i.methods.iter().any(|m| m.name == method) {
                return true;
            }
            for b in &i.bases {
                // Strip generic args and namespaces (`Ns.IEnumerable<T>`
                // matches by simple name, arity-agnostic like the rest).
                let mut base = b.as_str();
                if let Some(k) = base.find('<') {
                    base = &base[..k];
                }
                if let Some(k) = base.rfind('.') {
                    base = &base[k + 1..];
                }
                if walk(table, base.trim(), method, seen) {
                    return true;
                }
            }
        }
        false
    }
    if walk(table, iface, method, &mut Vec::new()) {
        return true;
    }
    // Every interface value is an object: `System.Object` members
    // resolve too (non-virtual dispatch nuances are the backend's).
    table
        .classes
        .iter()
        .filter(|c| c.name == "Object")
        .flat_map(|c| c.methods.iter())
        .any(|m| m.name == method)
}

/// Lower an extension-method call `recv.Name(args)` as the direct
/// static call `Declaring.Name(recv, args, ...defaults)`.
pub(super) fn lower_extension_call(
    object: &Expr,
    args: &[Expr],
    ext_mi: &MethodInfo,
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
) -> Result<VReg, LowerError> {
    ctx.check_method_access(ext_mi)?;
    let recv = eval(object, b, ctx)?;
    let mut explicit = eval_args(args, b, ctx)?;
    // Fill omitted optionals past the receiver slot.
    if ext_mi.param_defaults.len() > 1 {
        materialize_defaults(&mut explicit, &ext_mi.param_defaults[1..], b);
    }
    if !ext_mi.has_body {
        return Ok(b.const_i64(0));
    }
    let full = ext_mi.param_types.len();
    let mut argv = vec![recv];
    argv.extend(explicit);
    let dst = b.fresh();
    match static_function_index_for(
        ctx,
        &ext_mi.declaring_class,
        &ext_mi.name,
        full,
        &ext_mi.param_types,
        &ext_mi.param_is_byref,
    ) {
        Some(fidx) => {
            b.push_raw(Instr::CallStatic { dst, fidx, args: argv });
            Ok(dst)
        }
        // Declared but unlowered: placeholder, never a mis-call.
        None => Ok(b.const_i64(0)),
    }
}

/// A bare identifier naming a method (not a local/field/property/type)
/// is a C# method group, implicitly convertible to a compatible
/// delegate type (`certificate.GetPublicKey<ECDsa>(HasECDsaKeyUsage)`
/// passes the group to `Predicate<X509Certificate2>`). Values win over
/// methods in expression context, mirroring `static_type_of` order, and
/// only current-class methods count (cross-class groups via
/// `using static` stay unresolvable — honest error, future work).
pub(crate) fn is_method_group(expr: &Expr, b: &Body<'_>, ctx: &Ctx<'_>) -> bool {
    let Expr::Ident { name, .. } = expr else {
        return false;
    };
    is_method_group_name(name, b, ctx)
}

/// Name-based method-group check (see `is_method_group`): true when the
/// bare name denotes a current-class method and nothing else (no local,
/// field, property, outer static, or type). Used where only the name
/// is at hand (e.g. the undeclared-variable fallback in `eval`).
pub(crate) fn is_method_group_name(name: &str, b: &Body<'_>, ctx: &Ctx<'_>) -> bool {
    if b.scopes.lookup(name).is_some() {
        return false;
    }
    if let Some(info) = this_info(ctx, name) {
        if info.field(name).is_some() || info.property(name).is_some() {
            return false;
        }
    }
    if outer_static_field(ctx, name).is_some() {
        return false;
    }
    if ctx.table.class(name).is_some() {
        return false;
    }
    ctx.table
        .class(ctx.current_class)
        .is_some_and(|info| info.method(name).is_some())
}

/// Register (or reuse) a cross-image import, returning its
/// `Program.imports` index (host count + dep slot). Shared across all
/// method bodies of one build via `Ctx::dep_imports` (first-use order
/// = deterministic AST order).
pub(crate) fn dep_import_index(ctx: &Ctx, lib_hint: &str, symbol: &str) -> usize {
    let mut map = ctx.dep_imports.borrow_mut();
    let n = map.len();
    let slot = *map
        .entry((lib_hint.to_owned(), symbol.to_owned()))
        .or_insert(n);
    ctx.host_import_count + slot
}

/// Qualified export symbol for an (entry, method): computable
/// identically by importer (from `.rci`) and exporter (from its table).
/// Mirrors `qualified_exports_from` (arity-only: overloads sharing
/// name+arity share the symbol, like bare names today).
pub(crate) fn qualified_symbol(entry: &ClassInfo, method: &str, arity: usize) -> String {
    format!("{}::{}#{}", entry.key.path, method, arity)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table_of(src: &str) -> TypeTable {
        let unit = crate::parser::rd::parse(src).expect("repro parses");
        crate::native::semantic::analyze(&unit).expect("repro analyzes")
    }

    /// User-defined implicit operators are inherited: an operator on a
    /// base class applies to derived arguments (CSharp's
    /// `ErrorHandling.Error(ErrorCode, params ErrArg[])` calls pass a
    /// `PropWithType` where `ErrArg` offers `implicit operator
    /// ErrArg(SymWithType)`).
    #[test]
    fn inherited_implicit_operator_applies_to_derived_arg() {
        let table = table_of(
            r#"namespace t {
                class SymWithType { }
                class PropWithType : SymWithType { }
                class ErrArg {
                    public static implicit operator ErrArg(SymWithType swt) { return new ErrArg(); }
                }
            }"#,
        );
        assert_eq!(
            compat_score("t.PropWithType", "ErrArg", &table, &[]),
            Some(2),
            "base-class operator accepts derived arg"
        );
        assert_eq!(
            compat_score("t.SymWithType", "ErrArg", &table, &[]),
            Some(2),
            "direct operator still accepts"
        );
        assert_eq!(
            compat_score("int", "string", &table, &[]),
            None,
            "unrelated types still rejected (no over-widening)"
        );
    }

    /// Same-name lookup never lists one entry twice: a duplicated
    /// entry poisons the adopted-entry fallback (pointer-identity
    /// skips both copies) and turns resolvable constructions into
    /// loud `ctor_arity` failures.
    #[test]
    fn entries_for_single_entry_yields_one() {
        use crate::native::semantic::TypeKey;
        let table = TypeTable {
            classes: vec![ClassInfo {
                is_record: false,
                type_arg_index: Vec::new(),
                name: "Dup".to_owned(),
                key: TypeKey::new("Sys.Dup", 0),
                bases: Vec::new(),
                fields: Vec::new(),
                static_fields: Vec::new(),
                const_fields: Vec::new(),
                methods: Vec::new(),
                own_members: Vec::new(),
                interfaces: Vec::new(),
                properties: Vec::new(),
                is_enum: false,
                generic_params: Vec::new(),
                is_struct: false,
                owner: String::new(),
            }],
            interfaces: Vec::new(),
            delegates: Vec::new(),
            imports: Vec::new(),
            aliases: Vec::new(),
        };
        assert_eq!(entries_for(&table, "Dup").len(), 1);
    }
}
