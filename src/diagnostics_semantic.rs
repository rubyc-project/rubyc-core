//! Best-effort name resolution over a *recovered* AST.
//!
//! Runs after syntax errors are reported so that unknown names and unknown
//! functions still surface instead of being swallowed by the aborted
//! pipeline. Deliberately conservative: anything ambiguous is skipped.
//! Locals declared with `let` become visible AFTER their initializer
//! (C# definite-assignment style).

use crate::models::{
    ClassMember, CompilationUnit, Diagnostic, Expr, Member, ScopeContext, Severity, Span, Stmt,
};
use crate::utilities::suggest::did_you_mean;

/// Names callable without a receiver, everywhere.
const BUILTINS: &[&str] = &["printf", "print", "put_str", "eprintf", "scan_int", "null", "sizeof", "typeof",
    // Parser markers for pattern shapes the `is` lowering expands; they
    // are never real calls.
    "__rubyc_nonnull",
    "__rubyc_prop",
    "__rubyc_tuple",
    "__rubyc_list",
    // Undocumented TypedReference keywords (`__reftype(tr)` takes a
    // value, like `typeof` takes a type): always resolvable.
    "__reftype", "__makeref", "__refvalue",
    // The `field` contextual keyword (backing field in accessors):
    // always resolvable, mirroring `this`.
    "field",
    "Keyword(Checked)", "Keyword(Unchecked)", "base", "this",
];

/// Primitive type names usable as value references (`int.MaxValue`,
/// `typeof(int)`, `sizeof(double)`, etc.). These are identifiers, not
/// keywords, so the conservative scan must not flag them as unknown names.
const PRIMITIVES: &[&str] = &[
    "int", "uint", "long", "ulong", "short", "ushort", "byte", "sbyte", "nint", "nuint", "bool",
    "char", "float", "double", "decimal", "string", "object", "var", "void",
];

/// `System.Object` methods, instance and static (`ReferenceEquals`).
/// Every type inherits them, so they are implicitly in scope in every
/// method body (e.g. unqualified `GetType()`) even though no unit
/// declares them.
const OBJECT_METHODS: &[&str] = &[
    "GetType",
    "ToString",
    "Equals",
    "ReferenceEquals",
    "GetHashCode",
];

/// True for names the scan must never flag: builtins, primitives,
/// `System.Object` methods, and synthesized (`~`-prefixed) names.
fn is_known_name(name: &str) -> bool {
    BUILTINS.contains(&name)
        || PRIMITIVES.contains(&name)
        || OBJECT_METHODS.contains(&name)
        || name.starts_with('~')
}

/// True for callees the scan must never flag as unknown functions.
fn is_known_callee(name: &str) -> bool {
    BUILTINS.contains(&name) || OBJECT_METHODS.contains(&name)
}

#[derive(Clone)]
struct ClassInfo {
    fields: Vec<String>,
    methods: Vec<String>,
}

/// Collect nested type names (`Outer.Inner`, including `ref struct`
/// declarations such as `Number.BigInteger`) so the conservative scan
/// treats them as valid value references in enclosing bodies.
fn collect_nested_type_names(member: &Member, out: &mut Vec<String>) {
    if let Member::Class(class) = member {
        for nested_member in &class.members {
            if let ClassMember::Nested(nested) = nested_member {
                out.push(nested.name().to_string());
                collect_nested_type_names(nested, out);
            }
        }
    }
}

/// Conservative same-name scope scan (see `scan_with_fragments` for
/// the fragment-aware form used by dynamic-linking consume).
pub fn scan(unit: &CompilationUnit) -> Vec<Diagnostic> {
    scan_with_fragments(unit, &[])
}

/// Like [`scan`], plus dependency fragments from dynamic-linking
/// consume (see `native::rci`): dep-declared names seed the membership
/// sets so skipped dep sources don't read as unknown (`E0025`).
/// Seeding only ever REMOVES false errors (membership tests); genuine
/// typos still fail precisely downstream with the full table present.
/// Empty fragments behave exactly as before.
pub fn scan_with_fragments(
    unit: &CompilationUnit,
    fragments: &[crate::native::rci::Fragment],
) -> Vec<Diagnostic> {
    let mut diags = Vec::new();
    // Every field/method declared ANYWHERE: names in these sets are never
    // flagged by this conservative pass — inheritance/access give them a
    // precise verdict downstream (e.g. E0017).
    let mut any_member: Vec<String> = Vec::new();

    // Classes live under any namespace (block or file-scoped form) or at top level.
    let members = unit.all_members();

    let mut class_bases: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    let classes: Vec<(String, ClassInfo)> = members
        .iter()
        .filter_map(|m| match m {
            Member::Class(class) => {
                let mut info = ClassInfo {
                    fields: Vec::new(),
                    methods: Vec::new(),
                };
                for cm in &class.members {
                    match cm {
                        // Const fields are accessible as values
                        // (`ConstantInterpolated`), so include them too.
                        ClassMember::Field(f) => info.fields.push(f.name.clone()),
                        ClassMember::Property(p) => info.fields.push(p.name.clone()),
                        // Events are accessible as values (`handler = Changed`).
                        // Indexers are unnamed (accessed via `obj[i]`), so no
                        // name to add.
                        ClassMember::Event(e) => info.fields.push(e.name.clone()),
                        ClassMember::Indexer(_) | ClassMember::Nested(_) => {}
                        ClassMember::Method(m) => {
                            if m.name != class.name && !m.name.starts_with('~') {
                                info.methods.push(m.name.clone());
                            }
                        }
                    }
                }
                // Bases UNION across partial declarations (a part may
                // add `, IFace`); last-wins would drop the others.
                // Generic args stripped (`Lazy<T>` → `Lazy`): this
                // conservative scan is arity-blind by design (it only
                // needs member names, never layouts).
                class_bases.entry(class.name.clone()).or_default().extend(
                    class
                        .bases
                        .iter()
                        .map(|b| match b.find('<') {
                            Some(i) => b[..i].trim_end().to_owned(),
                            None => b.clone(),
                        }),
                );
                Some((class.name.clone(), info))
            }
            _ => None,
        })
        .collect();
    // Merge `partial` declarations (same class name, several files):
    // every part sees the union of fields/methods, mirroring .NET
    // (e.g. `RuntimeType.ObjectType` declared in the CoreCLR part,
    // used in the libraries part). Duplicates are harmless: the
    // scan only tests membership.
    let mut merged: Vec<(String, ClassInfo)> = Vec::new();
    for (name, info) in classes {
        match merged.iter_mut().find(|(n, _)| n == &name) {
            Some((_, existing)) => {
                existing.fields.extend(info.fields);
                existing.methods.extend(info.methods);
            }
            None => merged.push((name, info)),
        }
    }
    let mut classes = merged;
    // Fragment seeding (dynamic-linking consume): dep-declared classes
    // (with their fields/methods/bases), interfaces, delegates, and
    // enums join the membership sets above. Duplicates merge the same
    // way (harmless — membership only). Nested dep types seed by short
    // name (over-approx, safe direction: fewer false errors; precise
    // verdicts happen downstream with the full table). Seeding here —
    // before `any_member`/`type_names`/nested derivation — keeps every
    // downstream set consistent.
    for frag in fragments {
        for c in &frag.classes {
            let mut fields: Vec<String> = c
                .fields
                .iter()
                .map(|f| f.name.clone())
                .collect();
            fields.extend(c.properties.iter().map(|p| p.name.clone()));
            fields.extend(c.const_fields.iter().map(|f| f.name.clone()));
            let methods: Vec<String> = c.methods.iter().map(|m| m.name.clone()).collect();
            match classes.iter_mut().find(|(n, _)| n == &c.name) {
                Some((_, existing)) => {
                    existing.fields.extend(fields);
                    existing.methods.extend(methods);
                }
                None => classes.push((c.name.clone(), ClassInfo { fields, methods })),
            }
        }
    }
    // Primary-constructor params unioned across `partial` parts (the
    // ctor lives on one declaration, users on another) — same
    // rationale as the member merge above.
    let mut ctor_params: std::collections::BTreeMap<String, Vec<String>> =
        Default::default();
    for m in &members {
        if let Member::Class(class) = m {
            ctor_params
                .entry(class.name.clone())
                .or_default()
                .extend(class.primary_ctor_params.iter().cloned());
        }
    }
    for (_, info) in &classes {
        any_member.extend(info.fields.iter().cloned());
        any_member.extend(info.methods.iter().cloned());
    }

    // A bare name two types declare is resolved by scope, then by
    // declaration order — a guess the user cannot see. C# rejects such
    // a reference outright (CS0104). The choice is still made (so
    // qualified spellings and single-name code keep building), but each
    // ambiguous name is now reported with the entries it could mean.
    {
        let mut by_name: std::collections::BTreeMap<String, Vec<String>> = Default::default();
        {
            let mut collect = |members: &[Member], ns: &str| {
                for m in members {
                    let Member::Class(c) = m else { continue };
                    let path = if ns.is_empty() {
                        c.name.clone()
                    } else {
                        format!("{ns}.{}", c.name)
                    };
                    by_name.entry(c.name.clone()).or_default().push(path.clone());
                    for nested in &c.members {
                        if let ClassMember::Nested(n) = nested {
                            by_name
                                .entry(n.name().to_string())
                                .or_default()
                                .push(format!("{path}.{}", n.name()));
                        }
                    }
                }
            };
            // Declaration sites, NOT `all_members()`: that flattens
            // namespaced types to the top level, so each one would be
            // counted twice and every class would look ambiguous.
            collect(&unit.members, "");
            for ns in &unit.namespaces {
                collect(&ns.members, &ns.name);
            }
        }
        for (name, mut paths) in by_name {
            paths.sort();
            paths.dedup();
            if paths.len() < 2 {
                continue;
            }
            diags.push(
                Diagnostic::new(
                    "W0005",
                    Severity::Warning,
                    format!(
                        "'{name}' is declared more than once ({}) — an unqualified reference to it is resolved by scope, then by declaration order",
                        paths.join(", ")
                    ),
                )
                .in_context(ScopeContext::default()),
            );
        }
    }

    // Every named type declared in the unit (class/enum/interface/delegate)
    // is a valid value reference (static access, enum members, delegates).
    let mut type_names: Vec<&str> = classes.iter().map(|(n, _)| n.as_str()).collect();
    for m in &members {
        match m {
            Member::Enum(e) => type_names.push(e.name.as_str()),
            Member::Interface(i) => type_names.push(i.name.as_str()),
            Member::Delegate(d) => type_names.push(d.name.as_str()),
            Member::Class(_) => {}
        }
    }
    // Nested types live inside classes (e.g. `Number.BigInteger`) and are
    // valid value references in enclosing bodies.
    let mut nested_type_names: Vec<String> = Vec::new();
    for m in &members {
        collect_nested_type_names(m, &mut nested_type_names);
    }
    // `using Alias = ...` names are valid value references too
    // (`nint_t`, `NativeType`); precise target resolution is downstream.
    let alias_names: Vec<&str> = unit.aliases.iter().map(|a| a.name.as_str()).collect();
    // `using static Ns.Type;` imports the target type's members
    // (constants like `MaxStackAllocArgCount`, helpers) into scope —
    // mirroring .NET. Targets resolve by simple name against
    // top-level unit classes (all CoreLib static-usings do).
    let mut imported: Vec<String> = Vec::new();
    for target in &unit.static_usings {
        let simple = target.rsplit('.').next().unwrap_or(target);
        if let Some((_, info)) = classes.iter().find(|(n, _)| n == simple) {
            imported.extend(info.fields.iter().cloned());
            imported.extend(info.methods.iter().cloned());
        }
    }
    // Fragment bases/names (companion to the class seeding above,
    // which ran before `any_member` derivation): dep base spellings
    // resolve, and dep type names read as valid value references.
    for frag in fragments {
        for c in &frag.classes {
            for b in &c.bases {
                class_bases.entry(c.name.clone()).or_default().push(b.clone());
            }
            type_names.push(c.name.as_str());
            nested_type_names.push(c.name.clone());
        }
        for i in &frag.interfaces {
            type_names.push(i.name.as_str());
            nested_type_names.push(i.name.clone());
        }
        for d in &frag.delegates {
            type_names.push(d.name.as_str());
        }
    }

    for member in &members {
        if let Member::Class(class) = member {
            // Conservative member set: own fields/methods plus `this`.
            // Inheritance-aware resolution needs the type table, which is
            // unreliable on a recovered AST.
            let empty = ClassInfo {
                fields: vec![],
                methods: vec![],
            };
            let own = classes
                .iter()
                .find(|(n, _)| n == &class.name)
                .map(|(_, i)| i.clone())
                .unwrap_or_else(|| empty.clone());
            let mut scope = Scope {
                context: String::new(),
                names: {
                    let mut v = vec!["this".to_owned()];
                    v.extend(own.fields.iter().cloned());
                    v
                },
            };
            scope.names.extend(own.methods.iter().cloned());

            // Suggestion candidates for this class's bodies.
            // Effective members include inherited ones.
            let eff_fields = effective(&class.name, &classes, &class_bases, 0);
            let eff_methods = effective(&class.name, &classes, &class_bases, 1);
            let mut scope = Scope {
                context: String::new(),
                names: {
                    let mut v = vec!["this".to_owned()];
                    v.extend(eff_fields.iter().cloned());
                    v
                },
            };
            scope.names.extend(eff_methods.iter().cloned());
            // Type names are valid value references (e.g. static class access
            // `GenericReturns.M<T>(...)`, enum members `TestEnum.A`). The
            // conservative scan must not flag a name that is a known type in
            // the unit; precise resolution happens downstream.
            scope.names.extend(type_names.iter().map(|s| s.to_string()));
            scope.names.extend(nested_type_names.iter().cloned());
            scope.names.extend(alias_names.iter().map(|s| s.to_string()));
            scope.names.extend(imported.iter().cloned());
            // Primitive type names are valid value references too
            // (`int.MaxValue`, `typeof(int)`, `sizeof(double)`).
            scope.names.extend(PRIMITIVES.iter().map(|s| s.to_string()));
            // The class's own generic type parameters (`class Foo<T>`):
            // valid value refs throughout its member signatures/bodies.
            scope.names.extend(class.generic_params.iter().cloned());
            // Primary-constructor value parameters (`class C(int x)`),
            // unioned across `partial` parts: in scope for every body.
            if let Some(params) = ctor_params.get(&class.name) {
                scope.names.extend(params.iter().cloned());
            }
            // Generic type params (`T.Kind`, `U value`) are valid value refs.
            // Single-letter / T1-T10 names stay allowlisted: local generic
            // functions (`M<T>()` lowers to unparameterized `Let`) have no
            // AST node to carry them.
            for tp in ["T", "U", "V", "K", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"] {
                scope.names.push(tp.to_string());
            }
            // `_` is the discard variable, always valid in C#.
            scope.names.push("_".to_string());
            let mut candidate_strings: Vec<String> = eff_fields
                .iter()
                .chain(eff_methods.iter())
                .cloned()
                .collect();
            candidate_strings.extend(BUILTINS.iter().map(|s| s.to_string()));
            candidate_strings.extend(PRIMITIVES.iter().map(|s| s.to_string()));
            candidate_strings.extend(type_names.iter().map(|s| s.to_string()));
            candidate_strings.extend(nested_type_names.iter().cloned());
            candidate_strings.extend(alias_names.iter().map(|s| s.to_string()));
            candidate_strings.extend(imported.iter().cloned());
            let candidates: Vec<&str> = candidate_strings.iter().map(|s| s.as_str()).collect();
            for cm in &class.members {
                if let ClassMember::Method(m) = cm
                    && let Some(body) = &m.body
                {
                    let mut method_scope = scope.clone();
                    method_scope.context = format!("{}.{}", class.name, m.name);
                    // The method's own generic type parameters
                    // (`TResult M<TDecimal>(...)`).
                    for tp in &m.generic_params {
                        method_scope.declare(tp);
                    }
                    for p in &m.params {
                        method_scope.declare(&p.name);
                    }
                    let member_guard = MemberGuard {
                        any: &any_member,
                        own: &eff_methods,
                    };
                    walk_block(body, &method_scope, &candidates, &mut diags, &member_guard);
                }
            }
        }
    }
    diags
}

#[derive(Clone)]
struct Scope {
    names: Vec<String>,
    /// Enclosing `Class.Method`, for diagnostics only.
    context: String,
}

impl Scope {
    fn has(&self, n: &str) -> bool {
        self.names.iter().any(|x| x == n)
    }
    fn declare(&mut self, n: &str) {
        self.names.push(n.to_owned());
    }
}

fn unknown_name(name: &str, span: Span) -> Diagnostic {
    Diagnostic::new("E0025", Severity::Error, format!("unknown name '{name}'")).at(span)
}

/// Bare callee for scope/member lookups: explicit generic type
/// arguments are not part of the name (`M<T>(...)` looks up `M`;
/// substitution happens downstream). The full spelling stays in
/// messages.
fn bare_callee(callee: &str) -> &str {
    match callee.find('<') {
        Some(pos) => &callee[..pos],
        None => callee,
    }
}

/// Walk base chains collecting effective field/method names.
fn effective(
    name: &str,
    classes: &[(String, ClassInfo)],
    bases: &std::collections::BTreeMap<String, Vec<String>>,
    want: usize,
) -> Vec<String> {
    // Base spellings may be qualified (`System.Collections.CollectionBase`)
    // while entries key by simple name: compare simple throughout (same
    // simple-name philosophy as the rest of this scan).
    fn simple(n: &str) -> &str {
        n.rsplit('.').next().unwrap_or(n)
    }
    let mut out = Vec::new();
    let mut stack = vec![name.to_owned()];
    let mut seen = std::collections::BTreeSet::new();
    while let Some(cur) = stack.pop() {
        if !seen.insert(cur.clone()) {
            continue;
        }
        if let Some(b) = bases.get(simple(&cur)) {
            stack.extend(b.iter().map(|s| simple(s).to_owned()));
        }
        if let Some((_, info)) = classes.iter().find(|(n, _)| n == simple(&cur)) {
            let src = if want == 0 {
                &info.fields
            } else {
                &info.methods
            };
            out.extend(src.iter().cloned());
        }
    }
    out
}

struct MemberGuard<'a> {
    /// Every field/method declared anywhere in the unit.
    any: &'a [String],
    /// Effective methods of the class whose bodies are being walked
    /// (own + inherited). `this.M()` is checked against this set.
    own: &'a [String],
}

impl MemberGuard<'_> {
    fn known(&self, n: &str) -> bool {
        self.any.iter().any(|x| x == n)
    }

    /// Whether `name` is a method of the enclosing class. Bare
    /// `M(...)` inside a body is handled by the scope check (the
    /// enclosing class's members are in scope); only the `this.`
    /// form needs the member set.
    fn own_has_method(&self, n: &str) -> bool {
        self.own.iter().any(|x| x == n)
    }
}

#[allow(clippy::only_used_in_recursion)]
fn walk_expr(
    expr: &Expr,
    scope: &Scope,
    candidates: &[&str],
    out: &mut Vec<Diagnostic>,
    guard: &MemberGuard,
) {
    match expr {
        Expr::NamedArg { value, .. } => walk_expr(value, scope, candidates, out, guard),
        // `checked(x)` runs `x` in the same scope with the same names.
        Expr::Checked { inner, .. } => walk_expr(inner, scope, candidates, out, guard),
        Expr::Lambda { params, body, .. } => {
            // A lambda body is code that runs later, but it IS code: walk
            // it so `arg => OnX(arg)` resolves `arg` and reports a real
            // unknown name instead of hiding one behind the closure. The
            // parameters are declared first — without them the body's own
            // `x` reads as an unknown name.
            let mut inner = scope.clone();
            for p in params {
                inner.declare(p);
            }
            match body {
                crate::models::ast::expression::LambdaBody::Expr(e) => {
                    walk_expr(e, &inner, candidates, out, guard)
                }
                crate::models::ast::expression::LambdaBody::Block(b) => {
                    for st in &b.stmts {
                        walk_stmt(st, &mut inner, candidates, out, guard);
                    }
                }
            }
        }
        Expr::NewArray { size, .. } => walk_expr(size, scope, candidates, out, guard),
        Expr::Index { object, index, .. } => {
            walk_expr(object, scope, candidates, out, guard);
            walk_expr(index, scope, candidates, out, guard);
        }
        Expr::Ident { name, span } => {
            if !scope.has(name) && !is_known_name(name) {
                let mut d = unknown_name(name, *span);
                if !scope.context.is_empty() {
                    d.message = format!("{} [in {}]", d.message, scope.context);
                }
                if let Some(hint) = did_you_mean(name, candidates) {
                    d.help = Some(hint);
                }
                out.push(d);
            }
        }
        Expr::Call { callee, args, span } => {
            // `typeof(T)` / `sizeof(T)` take TYPES, not values — never
            // walk the arguments (`typeof(T[])` flagged 'T[]').
            // (`nameof(x)` DOES take a value; its args walk normally.)
            if callee == "typeof" || callee == "sizeof" {
                return;
            }
            // Unqualified calls may target a method declared by another
            // class/source in the same compilation unit. Precise overload
            // and access checks belong to lowering; this conservative scan
            // should only reject names absent from the entire unit.
            // (Look up the bare name: `M<T>` is `M` for scope purposes.)
            let bare = bare_callee(callee);
            if !scope.has(bare) && !guard.known(bare) && !is_known_callee(bare) {
                let mut d = Diagnostic::new(
                    "E0026",
                    Severity::Error,
                    format!(
                        "unknown function '{callee}'{}",
                        if scope.context.is_empty() {
                            String::new()
                        } else {
                            format!(" [in {}]", scope.context)
                        }
                    ),
                )
                .at(*span);
                if let Some(hint) = did_you_mean(callee, candidates) {
                    d.help = Some(hint);
                }
                out.push(d);
            }
            for a in args {
                walk_expr(a, scope, candidates, out, guard);
            }
        }
        Expr::Binary { op, lhs, rhs, .. } => {
            walk_expr(lhs, scope, candidates, out, guard);
            // The right side of `is` is a PATTERN, not a name: a type
            // (`is Ns.Type v`) or a constant (`is State.None`) is
            // resolved by the lowering, so the value-name scan must not
            // report it as an unknown variable.
            if !matches!(op, crate::models::BinOp::Is) {
                walk_expr(rhs, scope, candidates, out, guard);
            }
        }
        Expr::Unary { inner, .. } => walk_expr(inner, scope, candidates, out, guard),
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            walk_expr(cond, scope, candidates, out, guard);
            walk_expr(then_expr, scope, candidates, out, guard);
            walk_expr(else_expr, scope, candidates, out, guard);
        }
        Expr::New { args, .. } => {
            for a in args {
                walk_expr(a, scope, candidates, out, guard);
            }
        }
        Expr::MethodCall {
            object,
            name,
            args,
            span,
        } => {
            walk_expr(object, scope, candidates, out, guard);
            // A `this.M()` call resolves against the enclosing class:
            // its effective method set (own + inherited) is the
            // contract, so anything outside it is a real typo.
            // Interface members count too (an unimplemented member is
            // the implementation's error, not the caller's).
            //
            // Unit-wide names count too: `this.M()` may target an
            // EXTENSION method declared on another type (legal C#), and
            // a base the scan has no ClassInfo for (`object`) still
            // contributes its methods. Same philosophy as bare calls:
            // only names absent from the entire unit are typos.
            if matches!(object.as_ref(), Expr::Ident { name: n, .. } if n == "this")
                && !is_known_callee(bare_callee(name))
                && !guard.own_has_method(bare_callee(name))
                && !guard.known(bare_callee(name))
            {
                out.push(
                    Diagnostic::new(
                        "E0026",
                        Severity::Error,
                        format!("unknown function '{name}'"),
                    )
                    .at(*span),
                );
            }
            for a in args {
                walk_expr(a, scope, candidates, out, guard);
            }
        }
        Expr::Member { object, name, .. } => {
            // A dotted path through a name that resolves HERE is
            // qualification, not a value: `System.Array.Copy` (via
            // `Array`), `this.field`, `arg.Member`. The node's OWN
            // name counts too (the helper only sees the object below
            // it). Flag the root only when NO segment resolves;
            // precise receiver checking belongs to lowering's type
            // table.
            if !(scope.has(name) || is_known_name(name) || member_path_resolves(object, scope)) {
                walk_expr(object, scope, candidates, out, guard);
            }
        }
        Expr::Cast { expr, .. } => {
            walk_expr(expr, scope, candidates, out, guard);
        }
        Expr::Default { .. } => {}
        Expr::Assign { target, value, .. } => {
            walk_expr(target, scope, candidates, out, guard);
            walk_expr(value, scope, candidates, out, guard);
        }
        Expr::AssignComplex { lhs, value, .. } => {
            walk_expr(lhs, scope, candidates, out, guard);
            walk_expr(value, scope, candidates, out, guard);
        }
        Expr::PostInc { target, .. } | Expr::PostDec { target, .. } => {
            walk_expr(target, scope, candidates, out, guard);
        }
        Expr::SwitchExpr { subject, arms, .. } => {
            walk_expr(subject, scope, candidates, out, guard);
            for arm in arms {
                walk_expr(&arm.body, scope, candidates, out, guard);
            }
        }
        Expr::Await { inner, .. } => walk_expr(inner, scope, candidates, out, guard),
        Expr::InterpolatedString { parts, .. } => {
            for part in parts {
                if let crate::models::InterpPart::Hole(h) = part {
                    walk_expr(h, scope, candidates, out, guard);
                }
            }
        }
        Expr::Literal(_) => {}
    }
}



/// True when any segment of a member-access chain resolves in scope:
/// `System.Array.Copy` (via `Array`), `this.x` (via `this`),
/// `param.Member` (via the parameter). Every level's name counts —
/// the chain object of `A.B.C` is `A.B`, whose own name (`B`) must
/// be checked before descending to the root. Call results are
/// values, so chains rooted in a call never resolve here (they walk
/// normally).
fn member_path_resolves(expr: &Expr, scope: &Scope) -> bool {
    match expr {
        Expr::Ident { name, .. } => scope.has(name) || is_known_name(name),
        Expr::Member { object, name, .. } => {
            scope.has(name) || is_known_name(name) || member_path_resolves(object, scope)
        }
        Expr::MethodCall { object, .. } => member_path_resolves(object, scope),
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn walk_stmt(
    stmt: &Stmt,
    scope: &mut Scope,
    candidates: &[&str],
    out: &mut Vec<Diagnostic>,
    guard: &MemberGuard,
) {
    match stmt {
        Stmt::Let { name, init, .. } => {
            walk_expr(init, scope, candidates, out, guard);
            scope.declare(name); // visible after the declaration line
        }
        Stmt::Assign { name, value, span } => {
            // Assignment to a possible inherited member gets its precise
            // verdict (E0017) downstream; never preempt with unknown-name.
            // Builtins (`field` backing-field keyword, `this`, primitives)
            // resolve too.
            if !scope.has(name) && !guard.known(name) && !is_known_name(name) {
                let mut d = unknown_name(name, *span);
                if let Some(hint) = did_you_mean(name, candidates) {
                    d.help = Some(hint);
                }
                out.push(d);
            }
            walk_expr(value, scope, candidates, out, guard);
        }
        Stmt::MemberAssign { object, value, .. } => {
            walk_expr(object, scope, candidates, out, guard);
            walk_expr(value, scope, candidates, out, guard);
        }
        Stmt::Expression(e) => walk_expr(e, scope, candidates, out, guard),
        Stmt::Block { body, .. } => {
            // Label-led blocks (`L: stmt`, which the parser wraps so
            // both survive single-statement returns) share the
            // enclosing scope: C# labels introduce none, so a labeled
            // declaration (`L: int x = ...`) is visible after the
            // block. Anything else gets its own scope as usual.
            let label_led = body.stmts.first().is_some_and(|s| {
                matches!(s, crate::models::Stmt::Label { .. })
            });
            if label_led {
                for stmt in &body.stmts {
                    if let crate::models::Stmt::Let { name, .. } = stmt {
                        scope.declare(name);
                    }
                }
                for stmt in &body.stmts {
                    walk_stmt(stmt, scope, candidates, out, guard);
                }
            } else {
                walk_block(body, scope, candidates, out, guard)
            }
        }
        Stmt::Return { value: Some(e), .. } => walk_expr(e, scope, candidates, out, guard),
        Stmt::If {
            cond,
            then_block,
            else_block,
            ..
        } => {
            walk_expr(cond, scope, candidates, out, guard);
            walk_block(then_block, scope, candidates, out, guard);
            if let Some(b) = else_block {
                walk_block(b, scope, candidates, out, guard);
            }
        }
        Stmt::While { cond, body, .. } => {
            walk_expr(cond, scope, candidates, out, guard);
            walk_block(body, scope, candidates, out, guard);
        }
        Stmt::DoWhile { body, cond, .. } => {
            walk_block(body, scope, candidates, out, guard);
            walk_expr(cond, scope, candidates, out, guard);
        }
        Stmt::For {
            init,
            cond,
            step,
            body,
            ..
        } => {
            // C#-style: header scope lives around the whole loop.
            let mut loop_scope = scope.clone();
            if let Some(s) = init {
                walk_stmt(s, &mut loop_scope, candidates, out, guard);
            }
            if let Some(c) = cond {
                walk_expr(c, &loop_scope, candidates, out, guard);
            }
            if let Some(s) = step {
                walk_stmt(s, &mut loop_scope, candidates, out, guard);
            }
            walk_block(body, &loop_scope, candidates, out, guard);
        }
        Stmt::Break { .. } | Stmt::Continue { .. } => {}
        Stmt::Fixed { declarators, body, .. } => {
            // Pinned names scope over the body (like a block).
            let mut inner = scope.clone();
            for (_, name, init) in declarators {
                if let Some(e) = init {
                    walk_expr(e, &mut inner, candidates, out, guard);
                }
                inner.declare(name);
            }
            walk_block(body, &inner, candidates, out, guard);
        }
        _ => {}
    }
}

fn walk_block(
    block: &crate::models::Block,
    parent: &Scope,
    candidates: &[&str],
    out: &mut Vec<Diagnostic>,
    guard: &MemberGuard,
) {
    let mut scope = parent.clone();
    // Pre-register all `Let` names in the block so that references earlier
    // in the block (e.g. `is`-pattern bindings, lambda params, `out` args
    // that the parser queues after the using statement) resolve. This is a
    // conservative over-approximation: C# locals are in scope from their
    // declaration to the end of the block, and the scanner does not model
    // use-before-declaration precisely.
    for stmt in &block.stmts {
        if let crate::models::Stmt::Let { name, .. } = stmt {
            scope.names.push(name.clone());
        }
    }
    for stmt in &block.stmts {
        walk_stmt(stmt, &mut scope, candidates, out, guard);
    }
}
