//! AST to IR body lowering.

use super::super::{Instr, LowerError, VReg};
use crate::models::{BinOp as AstBinOp, Expr, Literal};
use crate::native::ir::Ctx;
use crate::native::semantic::{ClassInfo, FieldInfo, TypeTable};
use super::{body::*, calls::*, select::*};
/// Import symbol for a class's vtable row: the row lives in the owning
/// image; consumers point the object's slot 0 at it via `LoadImport`.
/// Format shared with the backend (see `target::vtable_symbol`).
pub(super) fn vtable_symbol(entry: &ClassInfo) -> String {
    crate::native::target::vtable_symbol(&entry.key.path)
}

/// Address of a static field's storage: the local slot, or — when the
/// declaring entry ships from another image — the owner's slot via
/// address import. `None` means no storage anywhere (callers keep their
/// existing miss behavior: loud error on writes, zero on reads).
pub(super) fn static_addr(field: &FieldInfo, b: &mut Body<'_>, ctx: &Ctx<'_>) -> Option<VReg> {
    // External statics first: the local slot (if any) is hollow zeroed
    // storage — the dep's initializers ran in its home image, not here.
    if let Some(entry) = member_host(ctx.table, &field.declaring_class, &field.name)
        && !entry.owner.is_empty()
    {
        let shadowed = ctx.classes.iter().any(|c| {
            c.name == entry.name
                && c.members.iter().any(|m| match m {
                    crate::models::ClassMember::Field(fd) => {
                        fd.name == field.name && fd.is_static
                    }
                    _ => false,
                })
        });
        if !shadowed {
            let lib = format!("{}.so", entry.owner);
            let sym = crate::native::target::static_symbol(&entry.key.path, &field.name);
            let idx = dep_import_index(ctx, &lib, &sym);
            let base = b.fresh();
            b.push_raw(Instr::LoadImport {
                dst: base,
                import: idx as u32,
            });
            return Some(base);
        }
    }
    ctx.static_index
        .get(&(field.declaring_class.clone(), field.name.clone()))
        .map(|&idx| {
            let base = b.fresh();
            b.load_static_addr(base, idx);
            base
        })
}

/// Cross-image construction (`new` on a class owned by another image):
/// allocate locally (layout comes from the fragment), point slot 0 at
/// the owner's vtable row via address import, and run the owner's ctor
/// via `CallImport`. Mirrors the decl path's selection/defaults/
/// bodyless rules; arity-known misses stay loud (`ctor_arity`).
pub(super) fn lower_external_new(
    info: &ClassInfo,
    args: &[crate::models::Expr],
    class_name: &str,
    b: &mut Body<'_>,
    ctx: &Ctx<'_>,
) -> Result<VReg, LowerError> {
    let obj = b.fresh();
    b.alloc(obj, (info.fields.len() + 2) as u64 * 8);
    let lib = format!("{}.so", info.owner);
    let vt = b.fresh();
    let vidx = dep_import_index(ctx, &lib, &vtable_symbol(info));
    b.push_raw(Instr::LoadImport {
        dst: vt,
        import: vidx as u32,
    });
    b.push_raw(Instr::StoreField {
        base: obj,
        off: 0,
        src: vt,
    });
    let arg_types: Vec<String> = args
        .iter()
        .map(|a| {
            if is_method_group(a, b, ctx) {
                return "method-group".to_owned();
            }
            static_type_of(a, b, ctx).unwrap_or_else(|_| "object".to_owned())
        })
        .collect();
    let selected = select_visible(
        ctx.table,
        std::slice::from_ref(&info),
        class_name,
        &arg_types,
        Some(false),
        ctx.current_class,
        None,
    )
    // Alias-spelled construction (see the decl path above): retry with
    // the resolved entry's own name.
    .or_else(|| {
        if info.name == class_name {
            None
        } else {
            select_visible(
                ctx.table,
                std::slice::from_ref(&info),
                &info.name,
                &arg_types,
                Some(false),
                ctx.current_class,
                None,
            )
        }
    });
    let has_ctors = info.methods.iter().any(|m| {
        m.name == info.name && m.declaring_class == info.name && !m.is_static
    });
    if selected.is_none() && has_ctors && !(info.is_struct && args.is_empty()) {
        return Err(LowerError::UnsupportedConstruct(crate::tf!(
            "rubyc.native.ir.lower_expr.ctor_arity",
            class_name.to_owned(),
            args.len().to_string()
        )));
    }
    if let Some(mi) = selected {
        ctx.check_method_access(mi)?;
        let mut argv = vec![obj];
        argv.extend(eval_args(args, b, ctx)?);
        if argv.len() - 1 < mi.param_types.len() {
            let mut tail: Vec<VReg> = argv.drain(1..).collect();
            fill_default_args(&mut tail, mi, b);
            argv.extend(tail);
        }
        if mi.has_body
            && let Some((lib, sym)) = external_target(ctx, info, &mi.name, mi.param_types.len())
        {
            let idx = dep_import_index(ctx, &lib, &sym);
            let call_dst = b.fresh();
            b.push_raw(Instr::CallImport {
                dst: call_dst,
                import: idx as u32,
                args: argv,
            });
        }
    }
    Ok(obj)
}

/// Resolve the declaring entry for a selected overload, for
/// cross-image import routing: same-named entries holding the method
/// (name+arity) first, else first-match by name (status quo ante for
/// type resolution).
pub(super) fn resolve_call_target<'t>(
    table: &'t TypeTable,
    declaring_class: &str,
    method: &str,
    arity: usize,
) -> Option<&'t ClassInfo> {
    table
        .classes
        .iter()
        .filter(|c| c.name == declaring_class || c.key.path == declaring_class)
        .find(|c| {
            c.methods
                .iter()
                .any(|m| m.name == method && m.param_types.len() == arity)
        })
        .or_else(|| {
            table
                .classes
                .iter()
                .find(|c| c.name == declaring_class || c.key.path == declaring_class)
        })
}

/// Dep-method import target, if the callee lives in another image:
/// owner non-empty (fragment-attributed) AND no AST declaration
/// compiled locally (shared-file edge: AST exists → local body wins,
/// matching `fn_index` behavior). Returns (lib hint `<owner>.so`,
/// qualified symbol). Bootstrap/test units (empty owners) never hit
/// this — zero behavior change there.
pub(crate) fn external_target(
    ctx: &Ctx,
    entry: &ClassInfo,
    method: &str,
    arity: usize,
) -> Option<(String, String)> {
    if entry.owner.is_empty() {
        return None;
    }
    let local = ctx.classes.iter().any(|c| {
        c.name == entry.name
            && c.members.iter().any(|m| match m {
                crate::models::ClassMember::Method(md) => {
                    md.name == method && md.params.len() == arity
                }
                _ => false,
            })
    });
    if local {
        return None;
    }
    Some((
        format!("{}.so", entry.owner),
        qualified_symbol(entry, method, arity),
    ))
}

pub(super) fn static_type_of(expr: &Expr, b: &Body<'_>, ctx: &Ctx<'_>) -> Result<String, LowerError> {    match expr {
        Expr::Ident { name, .. } => {
            // Declared bindings win over the `this` keyword (mirrors
            // eval): an `@this` parameter types as its declared type,
            // not the enclosing class.
            if let Some(sym) = b.scopes.lookup(name) {
                // Prefer nesting-qualified types (`Impl` declared in
                // `Matrix3x2` means `Matrix3x2.Impl`, not a same-named
                // nested elsewhere like `Matrix4x4.Impl` whose members
                // differ). Falls back to the recorded spelling.
                if let Some(q) = resolve_scoped(ctx.table, &sym.type_name, ctx.current_class) {
                    return Ok(q);
                }
                return Ok(sym.type_name.clone());
            }
            if name == "this" {
                return Ok(ctx.current_class.to_owned());
            }
            if name == "base" {
                // `base` refers to the base-class instance; its static type is
                // the current class's base. Prefer the exact entry under
                // lowering (`current_info` is path+signature
                // disambiguated): a short-name table lookup first-matches
                // unrelated same-named types (Xsl's baseless
                // `UnionIterator` STRUCT shadowing Linq's class → base
                // types as `Object` → "'Object' has no member ...").
                // Mirrors `this_info`'s current-first order.
                if let Some(base_name) = ctx
                    .current_info
                    .and_then(|info| info.bases.first())
                    .cloned()
                    .or_else(|| {
                        ctx.table
                            .class(ctx.current_class)
                            .and_then(|info| info.bases.first().cloned())
                    })
                {
                    return Ok(base_name);
                }
                return Ok(ctx.current_class.to_owned());
            }
            // The discard (`out _`, `_ = ...`): undeclared by design.
            // Keep it distinct from `object`/`null` so overload ranking
            // treats it as applicable-but-uninformative (a discard
            // accepts any parameter type) instead of rejecting
            // non-nullable value types.
            if name == "_" {
                return Ok("_".to_owned());
            }
            // Implicit-`this` field read: the type is the FIELD's
            // declared type (`_buffer.IndexOf(...)` types as `string`,
            // not the current class — typing it as the class broke
            // every instance call on a field, e.g. `StringParser`
            // has no member `IndexOf` for `_buffer.IndexOf(c, i)`).
            // Properties read the same way (`TypeHandle.Verify...`
            // where `TypeHandle` is an instance property, not the
            // same-named struct): members beat same-named types in
            // expression context, mirroring C# lookup order.
            if let Some(info) = this_info(ctx, name) {
                if let Some(f) = info.field(name) {
                    return Ok(f.type_name.clone());
                }
                if let Some(p) = info.property(name) {
                    return Ok(p.type_name.clone());
                }
            }
            // Enclosing-scope statics use their declared type (mirrors
            // the eval path above).
            if let Some((_, field)) = outer_static_field(ctx, name) {
                return Ok(field.type_name.clone());
            }
            // Primitive aliases as type references
            // (`decimal.ToUInt32(...)`, `int.Parse(...)`): resolve to
            // the framework name so entry lookups hit (case-sensitive
            // `table.class` misses the keyword spelling, degrading
            // return-type inference to `object` and breaking overload
            // ranking downstream).
            {
                let canon = canonical_primitive(name);
                if canon != *name && ctx.table.class(&canon).is_some() {
                    return Ok(canon);
                }
            }
            // Class name used as a qualified static caller (e.g.
            // `GenericAlgorithms.Convert(...)`). Return the class name so the
            // MethodCall handler can resolve the method.
            if ctx.table.class(name).is_some() {
                return Ok(name.clone());
            }
            // `using static` imports (e.g. `Decimal_Ctor_Int32` via
            // `using static ...CachedReflectionInfo`): file-scoped in
            // C#, approximated unit-wide here — resolve a bare name
            // against STATIC fields and properties (bare methods are
            // values only as groups, handled by the marker above).
            // Unique match only; ambiguity or nothing falls through
            // to legacy behavior below (a typo must stay loud
            // downstream, not silently mistype).
            {
                let mut found: Option<String> = None;
                let mut count = 0u32;
                for class in &ctx.table.classes {
                    for f in class.fields.iter().chain(class.static_fields.iter()) {
                        if f.is_static && f.name == *name {
                            found = Some(f.type_name.clone());
                            count += 1;
                        }
                    }
                    // `const` fields live in their own vec (always
                    // static): same treatment for hashing calls via
                    // `using static`-style bare references.
                    for f in &class.const_fields {
                        if f.name == *name {
                            found = Some(f.type_name.clone());
                            count += 1;
                        }
                    }
                    // Only STATIC properties participate in `using
                    // static` imports (mirrors the fields loop above):
                    // an unrelated instance property with the same
                    // bare name (`ZipArchive.Changed`) must not
                    // uniquely resolve an event invocation
                    // (`Changed?.Invoke(...)` in Composition) to its
                    // type.
                    for p in &class.properties {
                        if p.is_static && p.name == *name {
                            found = Some(p.type_name.clone());
                            count += 1;
                        }
                    }
                }
                if count == 1 {
                    return Ok(found.unwrap_or_else(|| "object".to_owned()));
                }
            }
            // Uppercase-starting identifier not in scope and not a class —
            // likely a generic type parameter. Return the name itself.
            if name
                .chars()
                .next()
                .map(|c| c.is_ascii_uppercase())
                .unwrap_or(false)
            {
                return Ok(name.clone());
            }
            // Fallback: unknown identifier (e.g. pattern-binding var in a
            // switch arm not yet registered). Treat as `object`.
            Ok("object".to_owned())
        }
        Expr::New { class_name, .. } => Ok(class_name.clone()),
        Expr::Member { object, name, .. } => {
            // Generic instantiation spelling (`Vector512<byte>` parses
            // as `Member(Vector512, "<byte>")`): when the object names
            // a TYPE, this is the instantiated type reference, not a
            // member access.
            if name.starts_with('<') {
                let base = static_type_of(object, b, ctx)?;
                if ctx.table.class(&base).is_some() {
                    return Ok(format!("{base}{name}"));
                }
                return Ok("object".to_owned());
            }
            let base_type = static_type_of(object, b, ctx)?;
            // Arrays expose `Length`/`Rank` (int) and `LongLength`
            // (long) without a class entry (`byte[].Length`).
            if base_type.contains('[') {
                if name == "Length" || name == "Rank" {
                    return Ok("int".to_owned());
                }
                if name == "LongLength" {
                    return Ok("long".to_owned());
                }
            }
            // Arity-aware entry resolution: a generic-spelled receiver
            // (`Vector512<byte>`) prefers the generic definition over
            // a same-named non-generic class.
            let Some(first) = class_for_spelling(ctx.table, &base_type) else {
                return Ok("object".to_owned());
            };
            // Same-named nesting fallback (see `member_host`): prefer
            // the entry actually declaring the member.
            let info = if first.field(name).is_some() || first.property(name).is_some() {
                first
            } else if member_host(ctx.table, &base_type, name).is_some() {
                // Base-declared member (`table` resolving to a
                // base-class property through a derived receiver):
                // the declaration lives up the chain. `first` (the
                // derived entry) has no such member, so read the
                // declared type off the actual declarer instead of
                // degrading to `object` (which rejects overloads —
                // `new ConstNode(table!, ...)` typed its 1st arg
                // `object` instead of `DataTable`).
                let found = match member_host(ctx.table, &base_type, name) {
                    Some(info) => info.property(name).map(|p| p.type_name.clone()),
                    None => None,
                };
                return Ok(found.unwrap_or_else(|| "object".to_owned()));
            } else {
                first
            };
            if let Some(f) = info.field(name) {
                // Enum member access (`FileMode.Open`) types as the
                // ENUM, not its underlying integral type — overload
                // applicability compares enum-to-enum (C# has no
                // implicit enum↔int conversion).
                if f.is_const && info.is_enum {
                    return Ok(info.name.clone());
                }
                return Ok(f.type_name.clone());
            }
            // Accessor-backed property (`dateTime.Ticks`): no
            // storage, but the DECLARED type keeps overload
            // applicability precise (reads still lower to a
            // placeholder until accessors become functions).
            if let Some(p) = info.property(name) {
                return Ok(p.type_name.clone());
            }
            // Nested-type access (`Outer.Inner` as a type reference):
            // the member is a type nested in the base entry, not a
            // field or property. C# finds inherited nesteds too, so
            // walk the base chain. Uppercase-gated (types are
            // PascalCase by convention; a lowercase miss degrades to
            // `object` exactly as before, so this only ever ADDS
            // resolutions, never changes existing ones).
            if name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
                let mut scope: Option<&ClassInfo> = Some(first);
                while let Some(entry) = scope {
                    let nested_path = format!("{}.{}", entry.key.path, name);
                    if let Some(nested) = ctx.table.class(&nested_path) {
                        return Ok(nested.key.path.clone());
                    }
                    scope = entry
                        .bases
                        .first()
                        .and_then(|b| ctx.table.class(b));
                }
            }
            Ok("object".to_owned())
        }
        Expr::MethodCall {
            object, name, args, ..
        } => {
            // Explicit generic type arguments (`Unsafe.As<string[]>`)
            // for return-type substitution below.
            let type_args = explicit_type_args(name);
            let name = match name.find('<') {
                Some(pos) => name[..pos].to_string(),
                None => name.clone(),
            };
            let base_type = static_type_of(object, b, ctx)?;
            let arg_types: Vec<String> = args
                .iter()
                .map(|a| {
                    if is_method_group(a, b, ctx) {
                        return "method-group".to_owned();
                    }
                    static_type_of(a, b, ctx).unwrap_or_else(|_| "object".to_owned())
                })
                .collect();
            // Overload-aware result type (a first-match `method()` may
            // return a sibling overload's type, e.g. `Task` for a
            // `ValueTask`-returning `WriteAsync` overload).
            // Canonicalized comparison so primitive aliases
            // (`decimal.ToUInt32`) count as static receivers.
            let static_recv = matches!(object.as_ref(), Expr::Ident { name: oname, .. } if canonical_primitive(oname) == base_type);
            let mut entries = entries_for(ctx.table, &base_type);
            if let Some(def) = class_for_spelling(ctx.table, &base_type) {
                if !entries.iter().any(|e| e.key == def.key) {
                    entries.insert(0, def);
                }
            }
            if let Some(m) = select_visible(
                ctx.table,
                &entries,
                &name,
                &arg_types,
                Some(static_recv),
                ctx.current_class,
                explicit_arity_of(&type_args),
            ) {
                // Substitute generic type arguments into the result:
                // explicit ones first (`Unsafe.As<string[]>(bucket)`),
                // then inferred from the actual args (`b.AsMemory(0,1)`
                // types as `Memory<byte>`, not bare `Memory` — without
                // this, downstream overloads miss and legacy first-match
                // elects the wrong overload).
                return Ok(substitute_generics(&m.ret_type, &m.generic_params, &type_args));
            }
            // Extension fallback so chained extension calls keep their
            // declared result type (`s.AsSpan().StartsWith(...)`),
            // with explicit generic args substituted as above.
            if let Some(ext) = extension_method(
                ctx.table,
                &base_type,
                &name,
                &arg_types,
                explicit_arity_of(&type_args),
            ) {
                return Ok(substitute_generics(&ext.ret_type, &ext.generic_params, &type_args));
            }
            Ok("object".to_owned())
        }
        // `arr[i]` — the element type is the array's element type.
        // Only true `T[]` arrays resolve precisely (recursively, so
        // `a[i][j]` on `int[][]` works). Non-array indexables
        // (`Span<T>`, `List<T>`, dictionaries — generic arguments are
        // not recorded) yield `object`: returning the CONTAINER type
        // mis-resolves member access (`tmp[0]._exponent` on a
        // `Span<DiyFp128>`), while `object` degrades to the usual
        // placeholder.
        Expr::Index {
            object, is_range, ..
        } => {
            let arr_type = static_type_of(object, b, ctx)?;
            // Range slices (`span[..n]`, `arr[i..j]`) have the
            // CONTAINER's type, not the element's — method chains on
            // the slice (`span[..n].ToArray()`) resolve against it.
            // (Codegen still emits the legacy element-read shape;
            // real slicing is future work.)
            if *is_range {
                return Ok(arr_type);
            }
            // `s[i]` on a string yields a character (byte value), not a
            // string: typing it "string" would route `printf("{0}", s[i])`
            // through WriteString and dereference the byte as an object.
            // Spelling-agnostic like the eval site below.
            if is_string_type(&arr_type) {
                return Ok("char".to_owned());
            }
            // True arrays resolve their element scope-aware (a `Bucket[]`
            // field declared in `Hashtable` holds `Hashtable.Bucket`s).
            if let Some(stripped) = arr_type.strip_suffix("[]") {
                if let Some(q) = resolve_scoped(ctx.table, stripped, ctx.current_class) {
                    return Ok(q);
                }
                return Ok(stripped.to_owned());
            }
            // Span/memory views index to their element (`Span<int>`
            // `bits[0]` is int, not object). Balanced extraction keeps
            // nested arguments whole; multi-arg lookalikes fall
            // through to object below.
            if let Some(k) = arr_type.find('<') {
                let base = base_type_name(&arr_type[..k]);
                if ["Span", "ReadOnlySpan", "Memory", "ReadOnlyMemory"]
                    .contains(&base.as_str())
                {
                    let bytes = arr_type.as_bytes();
                    let mut depth = 0i32;
                    let mut end = bytes.len();
                    for (i, &c) in bytes.iter().enumerate().skip(k) {
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
                    let inner = arr_type[k + 1..end].trim();
                    // Exactly one top-level argument (commas only
                    // inside nesting, e.g. tuple elements).
                    let mut nest = 0i32;
                    let mut parts = 1usize;
                    for c in inner.chars() {
                        match c {
                            '<' | '(' | '[' => nest += 1,
                            '>' | ')' | ']' => nest -= 1,
                            ',' if nest == 0 => parts += 1,
                            _ => {}
                        }
                    }
                    if !inner.is_empty() && parts == 1 {
                        return Ok(inner.to_owned());
                    }
                }
            }
            Ok("object".to_owned())
        }
        Expr::Literal(l) => {
            let ty = match l {
                Literal::Float(_) => "float",
                Literal::Int(_) => "int",
                Literal::Bool(_) => "bool",
                Literal::Str(_) => "string",
                Literal::Null => "object",
            };
            Ok(ty.to_owned())
        }
        Expr::InterpolatedString { .. } => Ok("string".to_owned()),
        // `default(T)` is a `T`. Bare `default` parses with the
        // placeholder type `void` (never a real type — `default(void)`
        // is illegal) and types as the `default` sentinel: C# converts
        // it to every parameter type (`M(default)` means
        // `M(default(T))`), unlike the null literal which stays
        // `object` and never matches non-nullable value types.
        Expr::Default { type_name, .. } => {
            if type_name.is_empty() || type_name == "void" {
                Ok("object".to_owned())
            } else {
                Ok(type_name.clone())
            }
        }
        // Explicit casts (`(nuint)offset`) type as their target — an
        // unknown result would poison overload applicability downstream
        // (`StoreUnsafe(ref buf, (nuint)offset)` vs `nuint`).
        Expr::Cast { type_name, .. } => Ok(type_name.clone()),
        // Binary operators (coarse C# typing — keeps overload
        // applicability precise instead of degrading everything to
        // `object`): comparisons/logicals yield `bool`; `??` yields
        // the left type when known; string `+` yields `string`;
        // numeric arithmetic promotes coarsely (double > float >
        // decimal > ulong > long > uint > int); anything else stays
        // unknown (custom `operator+` etc. resolve downstream).
        Expr::Binary { op, lhs, rhs, .. } => {
            if op.is_comparison()
                || matches!(op, &AstBinOp::And | &AstBinOp::Or | &AstBinOp::Is)
            {
                return Ok("bool".to_owned());
            }
            if matches!(op, &AstBinOp::NullCoalesce) {
                let l = static_type_of(lhs, b, ctx)?;
                if l != "object" {
                    return Ok(l);
                }
                return static_type_of(rhs, b, ctx);
            }
            let (l, r) = (
                static_type_of(lhs, b, ctx)?,
                static_type_of(rhs, b, ctx)?,
            );
            if matches!(op, &AstBinOp::Add) && (is_stringy(&l) || is_stringy(&r)) {
                return Ok("string".to_owned());
            }
            Ok(numeric_result(&l, &r))
        }
        // Ternaries: same-typed branches keep their type, else unknown.
        Expr::Ternary {
            then_expr,
            else_expr,
            ..
        } => {
            // Target-typed `default`: `cond ? X : default` types as X
            // (C# target typing — bare `default`, recorded with the
            // `"void"` placeholder type, converts to the other branch's
            // type). Without this the ternary degrades to `object` and
            // overload ranking collapses to legacy first-match, electing
            // the wrong overload (e.g. the `Task` overload over
            // `ValueTask` for `SendAsync(..., endOfMessage ?
            // Flags.EndOfMessage : default, ...)` → "'Task' has no
            // member 'AsTask'").
            if matches!(
                then_expr.as_ref(),
                Expr::Default { type_name, .. } if type_name.is_empty() || type_name == "void"
            ) {
                return static_type_of(else_expr, b, ctx);
            }
            if matches!(
                else_expr.as_ref(),
                Expr::Default { type_name, .. } if type_name.is_empty() || type_name == "void"
            ) {
                return static_type_of(then_expr, b, ctx);
            }
            let (t, e) = (
                static_type_of(then_expr, b, ctx)?,
                static_type_of(else_expr, b, ctx)?,
            );
            if t == e {
                Ok(t)
            } else {
                Ok("object".to_owned())
            }
        }
        // Bare calls (`GetIdx(i)`): overload-aware result type from
        // the current class entry (mirrors the lowering path's
        // current-info selection). Unresolvable stays `object`
        // (existing fallback behavior for unknown callees).
        Expr::Call { callee, args, .. } => {
            let type_args = explicit_type_args(callee);
            let name = match callee.find('<') {
                Some(pos) => callee[..pos].to_string(),
                None => callee.clone(),
            };
            let arg_types: Vec<String> = args
                .iter()
                .map(|a| {
                    if is_method_group(a, b, ctx) {
                        return "method-group".to_owned();
                    }
                    static_type_of(a, b, ctx).unwrap_or_else(|_| "object".to_owned())
                })
                .collect();
            let mut entries: Vec<&ClassInfo> = Vec::new();
            if let Some(cur) = ctx.current_info {
                entries.push(cur);
            }
            entries.extend(entries_for(ctx.table, ctx.current_class));
            if let Some(m) = select_visible(
                ctx.table,
                &entries,
                &name,
                &arg_types,
                None,
                ctx.current_class,
                explicit_arity_of(&type_args),
            ) {
                return Ok(substitute_generics(&m.ret_type, &m.generic_params, &type_args));
            }
            Ok("object".to_owned())
        }
        // Logical negation (`!x`) is always `bool` (the operand must
        // be boolean; applicability is overload resolution's job, not
        // typing's). Without this, `!call()` fell to the catch-all
        // `object` and bool-param overloads mis-resolved (`CheckError(!_,
        // ...)` in Xml serialization: "unknown call 'CheckError'").
        Expr::Unary { op, .. } if matches!(op, crate::models::UnOp::Not) => {
            Ok("bool".to_owned())
        }
        // Fallback: any other expression type — treat as `object`.
        _ => Ok("object".to_owned()),
    }
}

/// True for the string type in either spelling (`string`/`String`).
pub(super) fn is_stringy(t: &str) -> bool {
    t == "string" || t == "String"
}

/// Coarse C# numeric promotion for already-typed operands: both sides
/// must be numeric (else `object` — custom operators, enums, unknowns
/// keep legacy behavior); result follows `decimal > double > float >
/// ulong > long > uint > int`, lowercase spellings matching common
/// declarations (overload matching canonicalizes regardless). An
/// unknown side (`object` from stripped generics, e.g. span indexing)
/// defers to a numeric other side (`sepList[i] - currIndex` types as
/// `int`, not `object`).
pub(super) fn numeric_result(l: &str, r: &str) -> String {
    const NUM: &[&str] = &[
        "sbyte", "byte", "short", "ushort", "int", "uint", "long", "ulong", "char", "float",
        "double", "decimal", "SByte", "Byte", "Int16", "UInt16", "Int32", "UInt32", "Int64",
        "UInt64", "Char", "Single", "Double", "Decimal",
        // Pointer-sized integers (canonicalized from `nint`/`nuint`):
        // without these, `nuint * int` degrades to `object` and
        // downstream overloads miss.
        "nint", "nuint", "IntPtr", "UIntPtr",
    ];
    // Canonicalize first: keyword spellings (`ulong`, `uint`) must hit
    // the same priority branches as framework names (`UInt64`);
    // otherwise `ulong + uint` degrades to `int` instead of `ulong`.
    let (lb, rb) = (
        canonical_primitive(&base_type_name(l)),
        canonical_primitive(&base_type_name(r)),
    );
    if (lb == "Object" || lb == "dynamic") && NUM.contains(&rb.as_str()) {
        return r.to_owned();
    }
    if (rb == "Object" || rb == "dynamic") && NUM.contains(&lb.as_str()) {
        return l.to_owned();
    }
    if !NUM.contains(&lb.as_str()) || !NUM.contains(&rb.as_str()) {
        return "object".to_owned();
    }
    for t in [
        "Decimal", "Double", "Single", "UInt64", "Int64", "UInt32", "UIntPtr", "IntPtr",
    ] {
        if lb == t || rb == t {
            return match t {
                "Decimal" => "decimal".to_owned(),
                "Double" => "double".to_owned(),
                "Single" => "float".to_owned(),
                "UInt64" => "ulong".to_owned(),
                "Int64" => "long".to_owned(),
                "UIntPtr" => "nuint".to_owned(),
                "IntPtr" => "nint".to_owned(),
                _ => "uint".to_owned(),
            };
        }
    }
    "int".to_owned()
}

/// True when `expr` produces a 64-bit IEEE 754 value. Drives the choice
/// between integer (`Bin`/`Cmp`/`Neg`) and float (`FBin`/`FCmp`/`FNeg`)
/// instructions. A comparison's result is always an integer bool, so it is
/// never float.
pub(super) fn is_float_expr(b: &Body<'_>, ctx: &Ctx<'_>, expr: &Expr) -> bool {
    match expr {
        Expr::Literal(Literal::Float(_)) => true,
        Expr::Ident { name, .. } => {
            if name == "float" || name == "double" || name == "decimal" {
                return true;
            }
            match b.scopes.lookup(name) {
                Some(s)
                    if s.type_name == "float"
                        || s.type_name == "double"
                        || s.type_name == "decimal"
                        || s.type_name == "Decimal" =>
                {
                    true
                }
                _ => false,
            }
        }
        // Comparisons yield an integer bool.
        Expr::Binary { op, .. } if op.is_comparison() => false,
        // Arithmetic / bitwise: float when either operand is float.
        Expr::Binary { lhs, rhs, .. } => {
            is_float_expr(b, ctx, lhs) || is_float_expr(b, ctx, rhs)
        }
        Expr::Unary { inner, .. } => is_float_expr(b, ctx, inner),
        Expr::Ternary {
            then_expr,
            else_expr,
            ..
        } => is_float_expr(b, ctx, then_expr) || is_float_expr(b, ctx, else_expr),
        _ => false,
    }
}
