//! AST to IR body lowering.

use super::super::{Instr, LowerError, VReg};
use crate::models::{BinOp as AstBinOp, Expr, Literal};
use crate::native::ir::Ctx;
use crate::native::semantic::{ClassInfo, FieldInfo, INSTANCE_HEADER_SLOTS, TypeTable};
use super::{body::*, calls::*, select::*};
use super::typeinfo::{is_type_receiver as is_descriptor_receiver, type_member};
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
    // The header is vtable + refcount + runtime type, then the fields.
    b.alloc(
        obj,
        (info.fields.len() + INSTANCE_HEADER_SLOTS) as u64 * 8,
    );
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
            overload_arg_type(a, b, ctx)
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

/// Leftmost identifier of a dotted path (`Lib` in `Lib.C.x`).
fn dotted_root(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Ident { name, .. } => Some(name),
        Expr::Member { object, name, .. } => {
            if name.starts_with('<') {
                return None;
            }
            dotted_root(object)
        }
        _ => None,
    }
}

/// Whether a dotted path's root denotes a VALUE (member access) rather
/// than a namespace/type (type reference): locals/params, `this`,
/// `base`, current-class fields/properties, or method groups.
fn dotted_root_is_value(expr: &Expr, b: &Body<'_>, ctx: &Ctx<'_>) -> bool {
    let Some(root) = dotted_root(expr) else {
        return true;
    };
    if root == "this" || root == "base" || root == "_" {
        return true;
    }
    if b.scopes.lookup(root).is_some() {
        return true;
    }
    if let Some(info) = this_info(ctx, root) {
        if info.field(root).is_some() || info.property(root).is_some() {
            return true;
        }
    }
    is_method_group_name(root, b, ctx)
}

/// Flatten a dotted path expression (`Lib.C`, `System.Text.Json`) to
/// its source spelling. Returns `None` for anything else (calls,
/// indexers, generic spellings like `Vector512<byte>` which parse as
/// `Member(Vector512, "<byte>")`).
pub(super) fn dotted_spelling(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Ident { name, .. } => Some(name.clone()),
        Expr::Member { object, name, .. } => {
            if name.starts_with('<') {
                return None;
            }
            dotted_spelling(object).map(|base| format!("{base}.{name}"))
        }
        _ => None,
    }
}

pub(crate) fn static_type_of(expr: &Expr, b: &Body<'_>, ctx: &Ctx<'_>) -> Result<String, LowerError> {
    match expr {
        // A named argument has its value's type: overload selection and
        // coercion run before the argument list is rewritten to
        // parameter order.
        Expr::NamedArg { value, .. } => static_type_of(value, b, ctx),
        Expr::Ident { name, .. } => {
            // A generic-inlined body knows its type arguments: `T` in
            // `Test<IntNum>` IS `IntNum`, so `T.StaticMember` resolves
            // against the concrete type.
            if let Some(sub) = ctx.type_subst
                && let Some(concrete) = subst_type_var(name, sub)
            {
                return Ok(concrete);
            }
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
                    // A `using`-alias can leave a field's declared type
                    // as the alias name (`NativeType _value` where the
                    // alias stands for `IntPtr` or `Double`). The
                    // spelling resolves to no single entry, so carry it
                    // through as-is and let overload ranking treat it
                    // as a wildcard (see `is_using_alias`).
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
        Expr::SwitchExpr { arms, .. } => {
            // The arm bodies decide the result type: all-int arms make
            // the expression an int, not an unknown reference.
            let mut common: Option<String> = None;
            for arm in arms {
                let t = static_type_of(&arm.body, b, ctx)
                    .unwrap_or_else(|_| "object".to_owned());
                match &common {
                    None => common = Some(t),
                    Some(c) if *c == t => {}
                    Some(_) => {
                        common = Some("object".to_owned());
                        break;
                    }
                }
            }
            Ok(common.unwrap_or_else(|| "object".to_owned()))
        }
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
            let mut base_type = static_type_of(object, b, ctx)?;
            // Descriptor-backed System.Type properties: names are
            // strings, the Is*/Has* predicates are bools. Without this
            // the member types as `object` and comparisons/printf treat
            // a string pointer as an integer.
            if is_descriptor_receiver(&base_type)
                && let Some((off, bit)) = type_member(name)
            {
                if off == u32::MAX {
                    return Ok("System.RuntimeTypeHandle".to_owned());
                }
                return Ok(if bit.is_some() { "bool" } else { "string" }.to_owned());
            }
            // Namespace-qualified or nested type reference (`Lib.C`
            // used as a receiver): if the receiver itself spells a
            // known class path and its root is not a value in scope,
            // it is a type, not a member access — resolve it directly
            // instead of degrading through the bare namespace ident.
            // (Without this, the recursion below types `Lib` as
            // `object` and every qualified access silently becomes a
            // zero placeholder.) Value-rooted paths keep existing
            // behavior: only non-value roots take this shortcut.
            if !dotted_root_is_value(object, b, ctx) {
                if let Some(dotted) = dotted_spelling(object) {
                    if ctx.table.class(&dotted).is_some() {
                        base_type = dotted;
                    }
                }
            }
            // Arrays expose `Length`/`Rank` (int) and `LongLength`
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
            // Namespace-qualified or nested type reference (`Lib.C`):
            // the recursion above degrades bare namespace idents
            // (`Lib`) to `object`, losing the path — every qualified
            // access then silently becomes a zero placeholder. If the
            // whole dotted spelling names a class, this IS a type
            // reference; else if the receiver spelling alone does,
            // recover it for member lookup below. Value receivers keep
            // priority: usable bases never reach this fallback, and an
            // unresolvable receiver spelling falls through to the
            // legacy `object` behavior unchanged.
            // Degraded-`object` recovery: in a full world the degraded
            // `object` RESOLVES (System.Object), so the `Some` arm
            // below keeps it and the qualified path is lost forever
            // (`System.Numerics.BitOperations` typed as `object` — the
            // Log2 fold missed and the probe printed `a0` — while a
            // bare unit without Object recovered fine). When the base
            // is just the degraded unknown AND the full dotted spelling
            // names a real class rooted at a non-value, prefer the
            // dotted hit. Value-rooted paths keep legacy behavior
            // (`Holder.SD`-style fail-open zeros survive: a local whose
            // dotted path coincidentally spells a class must not
            // reroute).
            let dotted_hit = if matches!(base_type.trim(), "object" | "Object")
                && !dotted_root_is_value(object, b, ctx)
            {
                dotted_spelling(expr).filter(|d| ctx.table.class(d).is_some())
            } else {
                None
            };
            let base_type = match dotted_hit {
                Some(hit) => hit,
                None => match class_for_spelling(ctx.table, &base_type) {
                    Some(_) => base_type,
                    None => {
                        dotted_spelling(expr)
                            .filter(|d| ctx.table.class(d).is_some())
                            .or_else(|| {
                                dotted_spelling(object)
                                    .filter(|d| ctx.table.class(d).is_some())
                            })
                            .unwrap_or(base_type.clone())
                    }
                },
            };
            // Self-denoting type reference: the whole dotted spelling
            // already IS the resolved base (`Lib.C` typing member `C`
            // against base `Lib.C`). The expression names the type
            // itself, not a member of it — return it directly instead
            // of searching `C` among `Lib.C`'s members (which degrades
            // to `object` and collapses every qualified access above
            // it). Only exact equality triggers, so genuine member
            // paths (`Lib.C.counter`) and value receivers are
            // unaffected.
            if let Some(spelled) = dotted_spelling(expr) {
                if spelled == base_type {
                    return Ok(base_type);
                }
            }
            let Some(first) = class_for_spelling(ctx.table, &base_type) else {
                return Ok("object".to_owned());
            };
            // Same-named sibling fallback: `class_for_spelling`
            // first-matches (`State` may hit `XxHash32.State`, which
            // lacks `TotalLength`), while the member lives on a
            // same-named sibling (`XxHashShared.State`, imported via
            // `using static`). Prefer any same-named entry that
            // actually declares the member — otherwise genuine
            // members degrade to `object` and poison overload ranking
            // downstream (`XxHashShared.MergeAccumulators` went
            // unknown on an `object` length product).
            let simple_base = base_type
                .rsplit('.')
                .next()
                .unwrap_or(base_type.as_str())
                .split('<')
                .next()
                .unwrap_or(base_type.as_str());
            let first = {
                let mut hit = first;
                if hit.field(name).is_none() && hit.property(name).is_none() {
                    for cand in &ctx.table.classes {
                        let cand_simple =
                            cand.name.rsplit('.').next().unwrap_or(&cand.name);
                        if cand_simple != simple_base {
                            continue;
                        }
                        if cand.field(name).is_some() || cand.property(name).is_some() {
                            hit = cand;
                            break;
                        }
                    }
                }
                hit
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
            // `x.GetType()` yields a `System.Type`: every object has
            // the member, and the compiler answers the call with the
            // receiver's runtime type.
            if name == "GetType" && args.is_empty() {
                return Ok("System.Type".to_owned());
            }
            // A primitive value's `ToString()` is a string: without this
            // the call typed as `object`/`int`, so `==` compared the
            // string HANDLE and printf printed it as a number.
            if name == "ToString"
                && args.is_empty()
                && matches!(
                    base_type_name(&base_type).as_str(),
                    "int"
                        | "Int32"
                        | "uint"
                        | "UInt32"
                        | "long"
                        | "Int64"
                        | "ulong"
                        | "UInt64"
                        | "short"
                        | "Int16"
                        | "ushort"
                        | "UInt16"
                        | "byte"
                        | "Byte"
                        | "sbyte"
                        | "SByte"
                        | "nint"
                        | "IntPtr"
                        | "nuint"
                        | "UIntPtr"
                        | "bool"
                        | "Boolean"
                        | "char"
                        | "Char"
                )
            {
                return Ok("string".to_owned());
            }
            let arg_types: Vec<String> = args
                .iter()
                .map(|a| {
                    if is_method_group(a, b, ctx) {
                        return "method-group".to_owned();
                    }
                    overload_arg_type(a, b, ctx)
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
                let mut ret = substitute_generics(&m.ret_type, &m.generic_params, &type_args);
                // Class type arguments: a call on `ROS<byte>` returning
                // `ROS<T>` (the declaring class's parameter) must type
                // as `ROS<byte>`, not open `ROS<T>` — otherwise member
                // lookups miss the table and degrade to array-shape
                // reads at wrong offsets. Same-entry methods only;
                // inherited generic variance stays legacy.
                if let Some(recv) = class_for_spelling(ctx.table, &base_type)
                    .or_else(|| definition_entry(ctx.table, &base_type))
                {
                    if let Some(decl) = ctx.table.class(&m.declaring_class) {
                        if decl.key == recv.key && !decl.generic_params.is_empty() {
                            ret = substitute_generics(
                                &ret,
                                &decl.generic_params,
                                &generic_args_of(&base_type),
                            );
                        }
                    }
                }
                return Ok(ret);
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
            // A type-parameter receiver binds through its CONSTRAINT
            // (C#: `x[i]` uses the constraint's indexer, so
            // `ISimdVector<TVectorDouble, double>` makes the element a
            // double).
            if ctx.is_type_var(&arr_type) {
                for clause in &b.constraints {
                    if clause.param != arr_type {
                        continue;
                    }
                    for ct in &clause.types {
                        let (cbase, cargs) = match ct.find('<') {
                            Some(k) => (
                                ct[..k].trim().to_owned(),
                                explicit_type_args(ct),
                            ),
                            None => (ct.clone(), Vec::new()),
                        };
                        // The constraint may name an INTERFACE, which
                        // the table keeps beside the classes, so both
                        // sources are consulted: a class's own
                        // declaration and the table's interfaces.
                        let cparams: Vec<String> = ctx
                            .classes
                            .iter()
                            .find(|c| c.name == cbase)
                            .map(|c| c.generic_params.clone())
                            .or_else(|| {
                                ctx.table
                                    .interfaces
                                    .iter()
                                    .find(|i| i.name == cbase)
                                    .map(|i| i.generic_params.clone())
                            })
                            .unwrap_or_default();
                        let decl = ctx
                            .classes
                            .iter()
                            .find(|c| c.name == cbase)
                            .and_then(|c| {
                                c.members.iter().find_map(|m| match m {
                                    crate::models::ClassMember::Indexer(d)
                                        if d.params.len() == 1 =>
                                    {
                                        Some(d.clone())
                                    }
                                    _ => None,
                                })
                            })
                            .or_else(|| {
                                ctx.table
                                    .interfaces
                                    .iter()
                                    .find(|i| i.name == cbase)
                                    .and_then(|i| {
                                        i.indexers
                                            .iter()
                                            .find(|d| d.params.len() == 1)
                                            .cloned()
                                    })
                            });
                        let Some(decl) = decl else {
                            continue;
                        };
                        let ret = decl.type_name.clone();
                        let ret = if cargs.is_empty() {
                            ret
                        } else {
                            substitute_generics(&ret, &cparams, &cargs)
                        };
                        if !ret.is_empty() && !ctx.is_type_var(&ret) {
                            return Ok(ret);
                        }
                    }
                }
            }
            // A class INDEXER: the element type is the indexer's
            // declared type, with the RECEIVER's type arguments
            // substituted (`Vector128<double>[i]` is a double, not
            // `object`). Without the substitution the element typed as
            // `object`, and an overload taking the element type
            // (`double.Cos(x[i])`) did not resolve.
            {
                let resolved = ctx
                    .table
                    .alias_target(&arr_type)
                    .map(|t| t.to_owned())
                    .unwrap_or_else(|| {
                        resolve_scoped(ctx.table, &arr_type, ctx.current_class)
                            .unwrap_or_else(|| arr_type.clone())
                    });
                let (base, targs) = match resolved.find('<') {
                    Some(k) => (
                        resolved[..k].trim().to_owned(),
                        explicit_type_args(&resolved),
                    ),
                    None => (resolved.clone(), Vec::new()),
                };
                if let Some(info) = ctx.table.class(&base)
                    && let Some((_, decl)) = super::eval::find_indexer(ctx, &base, 1)
                {
                    let ret = decl.type_name.clone();
                    let ret = if targs.is_empty() {
                        ret
                    } else {
                        substitute_generics(&ret, &info.generic_params, &targs)
                    };
                    if !ret.is_empty() && !ctx.is_type_var(&ret) {
                        return Ok(ret);
                    }
                }
            }
            Ok("object".to_owned())
        }
        Expr::Literal(l) => {
            let ty = match l {
                // C# types unsuffixed/`d`-suffixed float literals as
                // `double` (`24.0`, `24d`); `f`-suffixed (`24.0f`) is
                // `float`, but the lexer erases suffixes, so every
                // float literal spells `double` (the common case) and
                // overload latitude keeps `double → float` applicable
                // for genuine `float` literals (see `compat_score`).
                Literal::Float(_) => "double",
                Literal::Int(_) => "int",
                Literal::Char(_) => "char",
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
                // Bare `default` (target type unknown here): report as
                // discard-like wildcard, NOT `object`. `object` means
                // null/unknown and is correctly REJECTED for value-type
                // and enum parameters — but `default` converts to every
                // parameter type (`M(default)` means `M(default(T))`
                // for whatever `T` the overload wants), so it must rank
                // applicable-but-uninformative like `out _` (see
                // `compat_score`). Without this, overloads taking
                // structs/enums never match bare-default args (the
                // `Task` base-chain ctor silently skipped, leaving
                // `m_stateFlags` zero and `IsCompleted` false).
                Ok("_".to_owned())
            } else {
                Ok(type_name.clone())
            }
        }
        // Explicit casts (`(nuint)offset`) type as their target — an
        // unknown result would poison overload applicability downstream
        // (`StoreUnsafe(ref buf, (nuint)offset)` vs `nuint`).
        Expr::Cast { type_name, .. } => Ok(type_name.clone()),
        // Array allocation: `new T[n]` types as `T[]`; collection
        // expressions (`[e1, …]`, `{ e1, … }`) carry no element type
        // — infer from the first initializer (all elements share
        // the 8-byte representation, so the coarse element type
        // only drives `[]`-detection and span packing, never
        // layout). Tuples (`elem_type == "tuple"`) are NOT arrays
        // and stay unknown here.
        Expr::NewArray { elem_type, init, .. } => {
            if elem_type == "tuple" {
                return Ok("object".to_owned());
            }
            if !elem_type.is_empty() {
                return Ok(format!("{elem_type}[]"));
            }
            if let Some(first) = init.first() {
                let t = static_type_of(first, b, ctx)?;
                if t != "object" {
                    return Ok(format!("{t}[]"));
                }
            }
            Ok("object".to_owned())
        }
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
            // Deep left-nested same-op spines (`a+b+c+...` hundreds
            // deep in lookup tables) overflow via recursion below, so
            // fold them iteratively here (mirrors eval's flattening):
            // collect leaf types down the spine (each leaf is shallow
            // — only the spine itself is deep), then join pairwise.
            // Only pure arithmetic/bitwise ops (no branches, no
            // control flow); anything else keeps the legacy path.
            // Only spines longer than 2 (pairs keep the exact legacy
            // path, including the string/float special cases).
            {
                let flattenable = matches!(
                    op,
                    AstBinOp::Add
                        | AstBinOp::Sub
                        | AstBinOp::Mul
                        | AstBinOp::Div
                        | AstBinOp::Rem
                        | AstBinOp::BitAnd
                        | AstBinOp::BitOr
                        | AstBinOp::BitXor
                        | AstBinOp::Shl
                        | AstBinOp::Shr
                        | AstBinOp::ShrShr
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
                        let mut acc = static_type_of(cur, b, ctx)?;
                        for r in rights.iter().rev() {
                            let t = static_type_of(r, b, ctx)?;
                            let is_ptr = |x: &str| {
                                let xr = x.trim();
                                xr.ends_with('*') && !xr.ends_with("[]")
                            };
                            acc = if matches!(op, &AstBinOp::Add)
                                && (is_stringy(&acc) || is_stringy(&t))
                            {
                                "string".to_owned()
                            } else if matches!(op, &AstBinOp::Add) && is_ptr(&acc) {
                                acc
                            } else if matches!(op, &AstBinOp::Add) && is_ptr(&t) {
                                t
                            } else {
                                numeric_result(&acc, &t)
                            };
                        }
                        return Ok(acc);
                    }
                }
            }
            let (l, r) = (
                static_type_of(lhs, b, ctx)?,
                static_type_of(rhs, b, ctx)?,
            );
            if matches!(op, &AstBinOp::Add) && (is_stringy(&l) || is_stringy(&r)) {
                return Ok("string".to_owned());
            }
            // Deep left-nested same-op spines (`a+b+c+...` hundreds deep
            // in lookup tables) would overflow via the recursion above.
            // Detect them here and fold iteratively instead: collect
            // leaf types down the spine (each leaf is shallow), then
            // join. Only for spines longer than 2 (common pairs keep
            // the exact legacy path).
            // NOTE: implemented in eval flattening; typing stays
            // recursive (frames are small here — measured safe past
            // 2000 depth; revisit if a deeper shape appears).
            // Pointer arithmetic types as the pointer (mirrors
            // `pointer_arith` in eval: `p + i` stays `T*`, `p - q`
            // counts elements). Without this, ranking sees `byte* +
            // int` as `int` while lowering scales it as a pointer —
            // the mismatch misroutes every address-taking call
            // (`XxHashShared.MergeAccumulators` went unknown).
            if matches!(op, &AstBinOp::Add | &AstBinOp::Sub) {
                let is_ptr = |t: &str| {
                    let tr = t.trim();
                    tr.ends_with('*') && !tr.ends_with("[]")
                };
                if matches!(op, &AstBinOp::Sub) && is_ptr(&l) && is_ptr(&r) {
                    return Ok("long".to_owned());
                }
                for t in [&l, &r] {
                    if is_ptr(t) {
                        return Ok(t.clone());
                    }
                }
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
            // Both branches carry the same type: that is the expression's
            // type. Degrading to `object` made a value-type result travel
            // as if it were a reference (an int landing in an object slot
            // and read back as one).
            {
                let tt = static_type_of(then_expr, b, ctx)
                    .unwrap_or_else(|_| "object".to_owned());
                let et = static_type_of(else_expr, b, ctx)
                    .unwrap_or_else(|_| "object".to_owned());
                if tt == et && tt != "object" {
                    return Ok(tt);
                }
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
            } else if t.trim().strip_suffix('?').unwrap_or(t.trim())
                == e.trim().strip_suffix('?').unwrap_or(e.trim())
            {
                // Nullable-annotation-only difference (`byte[]?` vs
                // `byte[]`): unify to the nullable spelling (preserves
                // nullability; shape dispatch like array checks strips
                // it as needed). Without this the ternary degrades to
                // `object` and downstream coercion bails (a nullable
                // array stayed a raw array in a span-typed local —
                // zeroed-length spans faulted console Flush).
                Ok(if t.trim().ends_with('?') {
                    t
                } else {
                    e
                })
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
            // `sizeof(T)` types as `int` (C# spec; the value fold in
            // calls.rs computes the size). Without this it degrades
            // to `object`, poisoning enclosing arithmetic
            // (`AccumulatorCount * sizeof(ulong)` typed
            // `int*object`, collapsing a `MergeAccumulators` overload
            // to unknown).
            if name == "sizeof" {
                return Ok("int".to_owned());
            }
            // `typeof(T)` types as `System.Type` (C# spec; the value
            // is a placeholder box in calls.rs). Without this it
            // degrades to `object`, hiding `Type`-family receivers
            // from reflection interceptions (and any `Type`-gated
            // logic): `typeof(X).GetConstructors(...)` ran the real
            // body and faulted instead of answering empty.
            if name == "typeof" {
                return Ok("System.Type".to_owned());
            }
            // Compiler intrinsics with a known result type: `scan_int`
            // reads an int, and the print intrinsics return 0. Typing
            // them as `object` (the unknown fallback) let a value type
            // travel as a reference and be unboxed from a raw int.
            if matches!(
                name.as_str(),
                "scan_int" | "printf" | "print" | "put_str" | "eprintf"
            ) {
                return Ok("int".to_owned());
            }
            // `x.GetType()` yields a `System.Type` (every object has
            // it). The compiler answers the call from the receiver's
            // runtime type, so the result is a real descriptor.
            if name == "GetType" {
                return Ok("System.Type".to_owned());
            }
            // Static local functions type by declaration (checked
            // before method lookup, mirroring call lowering).
            if let Some(ret) = resolve_local_ret(ctx, &name, args.len()) {
                return Ok(ret);
            }
            let arg_types: Vec<String> = args
                .iter()
                .map(|a| {
                    if is_method_group(a, b, ctx) {
                        return "method-group".to_owned();
                    }
                    overload_arg_type(a, b, ctx)
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
        // Bitwise complement and numeric negation preserve the
        // operand type (`~TotalLength` stays `ulong`; without this
        // they degraded to `object` and rejected overloads — the
        // `high64` half of `MergeAccumulators` went unknown on it).
        Expr::Unary { op, inner, .. }
            if matches!(
                op,
                crate::models::UnOp::BitNot | crate::models::UnOp::Neg
            ) =>
        {
            static_type_of(inner, b, ctx)
        }
        // Assignment expression (`x = v`, incl. nested assigns in
        // ternaries like `(_byteBuffer = new byte[...])`): types as
        // the assigned value (C# types it as the target, which the
        // value converts to). Without this it degrades to `object`
        // and poisons enclosing ternaries (`stackalloc-array` vs
        // `object` → `object`), which then skips span coercion —
        // StreamWriter's `byteBuffer` stayed a raw array and Flush
        // encoded into a zero-length destination.
        Expr::Assign { value, .. } => static_type_of(value, b, ctx),
        // A dereference reads the POINTED-TO type: `*(byte*)p` is a
        // byte. Without this the type degraded to `object`, which
        // poisons the generic inference (`Memmove(ref *(byte*)dst,
        // ...)` inferred nothing) and every overload that takes the
        // pointed-to type.
        Expr::Unary {
            op: crate::models::UnOp::Deref,
            inner,
            ..
        } => {
            let pointed = static_type_of(inner, b, ctx)?;
            let t = pointed.trim();
            if t.ends_with('*') {
                Ok(t.trim_end_matches('*').trim().to_owned())
            } else {
                Ok(pointed)
            }
        }
        // Call-site `ref`/`in` markers (parser) and explicit
        // address-of: the inner type, &-suffixed so overload ranking
        // can require byref parameters (unmarked value behavior
        // self-elected same-name value twins — `WriteAtOffset`
        // recursed forever on identical args). Consumers strip `&`
        // for conversions (see `base_type_name`, `compat_score`,
        // `span_elem_width`); explicit `&x` interop to integer
        // params stays applicable via the address-sink rule.
        Expr::Unary {
            op: crate::models::UnOp::AddrOf,
            inner,
            ..
        } => Ok(format!("{}&", static_type_of(inner, b, ctx)?)),
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
    // Same-typed operands keep their type: `Vector<T> & Vector<T>` is
    // `Vector<T>`, not `object`. Without this, a bitwise/logical op on
    // a non-numeric type degraded to `object`, which poisoned the
    // generic inference (`IsZero(vector & Vector<T>.One)` inferred
    // nothing for `T`).
    if l == r && !l.trim().is_empty() {
        return l.to_owned();
    }
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
        // Iterative worklist (never recurse): hundred-deep `+` chains
        // in lookup tables would otherwise overflow the compiler.
        Expr::Binary { .. } => {
            let mut work = vec![expr];
            while let Some(e) = work.pop() {
                match e {
                    Expr::Literal(Literal::Float(_)) => return true,
                    Expr::Binary { lhs, rhs, .. } => {
                        work.push(lhs);
                        work.push(rhs);
                    }
                    Expr::Unary { inner, .. } => work.push(inner),
                    Expr::Ternary {
                        then_expr,
                        else_expr,
                        ..
                    } => {
                        work.push(then_expr);
                        work.push(else_expr);
                    }
                    // Calls/members/indexers: check the callee shape
                    // without resolving (resolution recurses separately
                    // and shallowly).
                    _ => {}
                }
            }
            false
        }
        _ => false,
    }
}



/// Replace a type variable with its concrete binding. The variable may
/// be the whole spelling (`T`), a decorated one (`T[]`, `T?`), or an
/// argument of a constructed type (`List<T>`, `Dictionary<K, V>`):
/// specializing a generic method replaces the parameter everywhere it
/// appears, so the specialized body names concrete types.
pub(super) fn subst_type_var(
    spelling: &str,
    sub: &[(String, String)],
) -> Option<String> {
    let t = spelling.trim();
    let (base, suffix) = split_type_suffix(t);
    let base = subst_type_base(base, sub)?;
    Some(format!("{base}{suffix}"))
}

/// The trailing decoration of a type spelling: an array rank (`[]`,
/// `[,]`), a pointer marker (`*`), or a nullable annotation (`?`).
fn split_type_suffix(t: &str) -> (&str, &str) {
    if t.ends_with(']')
        && let Some(open) = t.rfind('[')
    {
        return (&t[..open], &t[open..]);
    }
    if let Some(base) = t.strip_suffix('*') {
        return (base, "*");
    }
    if let Some(base) = t.strip_suffix('?') {
        return (base, "?");
    }
    (t, "")
}

/// Substitute inside a type spelling that carries no trailing
/// decoration. A constructed type keeps its shape; only the arguments
/// that name a type variable change.
fn subst_type_base(t: &str, sub: &[(String, String)]) -> Option<String> {
    if t.ends_with('>')
        && let Some(open) = t.find('<')
    {
        let args = explicit_type_args(t);
        if !args.is_empty() {
            let mut changed = false;
            let mut out = Vec::with_capacity(args.len());
            for a in &args {
                match subst_type_var(a, sub) {
                    Some(c) => {
                        changed = true;
                        out.push(c);
                    }
                    None => out.push(a.clone()),
                }
            }
            if changed {
                return Some(format!("{}<{}>", &t[..open], out.join(",")));
            }
        }
    }
    for (var, concrete) in sub {
        if t == var {
            return Some(concrete.clone());
        }
    }
    None
}

/// The DECLARED type of an expression, or `None` when the compiler
/// cannot establish one. Unlike [`static_type_of`], an unknown shape
/// is not reported as `object`: the difference matters where the
/// compiler must know that a value really is a reference (implicit
/// unboxing of a pattern binding reads the box header, which is only
/// safe for a reference).
pub(crate) fn declared_type_of(
    expr: &Expr,
    b: &Body<'_>,
    ctx: &Ctx<'_>,
) -> Option<String> {
    match expr {
        Expr::Ident { name, .. } => b.scopes.lookup(name).map(|s| s.type_name.clone()),
        Expr::Member { object, name, .. } => {
            let recv = declared_type_of(object, b, ctx)?;
            let info = ctx.table.class(&recv).or_else(|| definition_entry(ctx.table, &recv))?;
            if let Some(f) = info.field(name) {
                return Some(f.type_name.clone());
            }
            info.property(name).map(|p| p.type_name.clone())
        }
        Expr::Index { object, .. } => {
            let recv = declared_type_of(object, b, ctx)?;
            if recv.trim_end().ends_with("[]") {
                Some(recv.trim_end_matches("[]").to_owned())
            } else {
                None
            }
        }
        Expr::Cast { type_name, .. } => Some(type_name.clone()),
        Expr::Literal(crate::models::Literal::Null) => Some("object".to_owned()),
        Expr::Call { .. } => {
            // A call types by its declared return type; an unresolved
            // callee (a compiler intrinsic the table does not know) has
            // no declared type.
            let t = static_type_of(expr, b, ctx).ok()?;
            if t == "object" { None } else { Some(t) }
        }
        _ => None,
    }
}

/// Argument type for overload selection. The literal zero keeps its
/// own spelling: C# converts the constant 0 implicitly to any enum
/// type, and no other integer literal may convert to one.
pub(crate) fn overload_arg_type(
    expr: &Expr,
    b: &Body<'_>,
    ctx: &Ctx<'_>,
) -> String {
    if matches!(expr, Expr::Literal(crate::models::Literal::Int(0))) {
        return "int-zero".to_owned();
    }
    static_type_of(expr, b, ctx).unwrap_or_else(|_| "object".to_owned())
}
