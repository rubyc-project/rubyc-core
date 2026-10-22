//! Semantic analysis: class/interface tables, inheritance chains,
//! visibility enforcement, and interface conformance.
//!
//! Runs after parsing, before lowering. Produces [`TypeTable`] records the
//! codegen uses for object layout and vtables, plus diagnostics for every
//! rule violation (access modifiers, missing interface methods, unknown
//! bases, cyclic hierarchies).

mod table;

pub use table::{
    ClassInfo, ConstValue, FieldInfo, FIRST_FIELD_SLOT, ImportFn, INSTANCE_HEADER_SLOTS,
    MethodInfo, PropertyInfo, RUNTIME_TYPE_SLOT, TypeKey, TypeTable, split_arity,
};

use crate::models::diagnostic::{Diagnostic, ScopeContext, Severity};
use crate::models::ast::expression::{BinOp, Expr, Literal, UnOp};
use crate::models::{Access, ClassDecl, CompilationUnit, EnumDecl, InterfaceDecl, MethodDecl};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticError {
    pub code: &'static str,
    pub message: String,
    /// Suggested fix, rendered as a `help:` line.
    pub help: Option<String>,
}

fn err(code: &'static str, message: impl Into<String>) -> SemanticError {
    SemanticError {
        code,
        message: message.into(),
        help: None,
    }
}

fn err_help(
    code: &'static str,
    message: impl Into<String>,
    help: impl Into<String>,
) -> SemanticError {
    SemanticError {
        code,
        message: message.into(),
        help: Some(help.into()),
    }
}

/// Constant-fold a compile-time `const` initializer to a [`ConstValue`].
///
/// Supports integer/float literals, unary negation, and the arithmetic and
/// bitwise binary operators over constants — the full set of expressions a
/// `const` field may legitimately use in C#. Returns `None` when the
/// expression is not a compile-time constant (identifier, call, etc.).
/// Context for constant folding: enum values, resolved class consts
/// (fixpoint), and the referring class (key + base chain) for scope.
/// Convert a folded constant to a field's declared type (C# implicit
/// constant conversion): integer values assigned to `float`/`double`
/// (`Single`/`Double`) become Float. Without this, `double X = <long>`
/// folds as Int and downstream arithmetic overflows (`checked_mul`
/// on 2**32 operands) or mistypes overload selection.
fn convert_const(value: ConstValue, type_name: &str) -> ConstValue {
    let base = type_name
        .trim_end_matches(['?', '*'])
        .rsplit('.')
        .next()
        .unwrap_or(type_name);
    // Strip a single generic argument list (`Nullable<double>` — the
    // `[`/`(` forms never appear on const declarations).
    let base = base.split('<').next().unwrap_or(base);
    match (&value, base) {
        (ConstValue::Int(i), "float" | "Single" | "double" | "Double") => {
            ConstValue::Float(*i as f64)
        }
        _ => value,
    }
}

struct ConstCx<'a> {
    enums: &'a std::collections::BTreeMap<(String, String), i64>,
    /// ((class key, const name) → value), populated fixpoint-style.
    consts: &'a std::collections::BTreeMap<(TypeKey, String), ConstValue>,
    /// Class chains for base-scoped lookup (key → base keys).
    chains: &'a std::collections::BTreeMap<TypeKey, Vec<TypeKey>>,
    /// All class keys (for resolving `Type.Const` member objects).
    keys: &'a Vec<TypeKey>,
    /// Referring class key (for bare-name scope).
    current: TypeKey,
}

/// Fold a constant expression. `enums` maps (enum simple name,
/// variant) → value for `Enum.Member` references (`BindingFlags.X`);
/// `cx` resolves const-to-const references (same class, bases, or
/// qualified `Type.Const`). Callers pass precomputed tables.
fn const_fold(expr: &Expr, cx: &ConstCx) -> Option<ConstValue> {
    let enums = cx.enums;
    match expr {
        Expr::Literal(l) => match l {
            Literal::Int(v) => Some(ConstValue::Int(*v)),
            Literal::Char(v) => Some(ConstValue::Int(*v as i64)),
            Literal::Float(v) => Some(ConstValue::Float(*v)),
            Literal::Bool(v) => Some(ConstValue::Int(i64::from(*v))),
            Literal::Str(s) => Some(ConstValue::Str(s.clone())),
            // `const string X = null` is legal C# (Interop consts
            // like `LOCALE_NAME_USER_DEFAULT`); the null reference
            // is a valid constant value, lowered as handle 0.
            Literal::Null => Some(ConstValue::Null),
        },
        // Explicit numeric conversions (`(int)0x...`, `(byte)x`):
        // fold the inner value (narrowing assumed intended, mirroring
        // `unchecked` semantics — `checked` is desugared by the parser).
        Expr::Cast { expr: inner, .. } => const_fold(inner, cx),
        // Enum member reference (`BindingFlags.Instance`,
        // `Outer.NestedEnum.Value`): look up the precomputed value by
        // the enum's simple name (last segment before the variant).
        // Bare names are NOT resolved (too ambiguous).
        // Class const reference (`Math.PI`, `DirectorySeparatorCharAsString`
        // bare in scope): look up the fixpoint-resolved const table.
        Expr::Member { object, name, .. } => {
            // Flatten `A.B.C` → ([A, B], C); `X.Y` → ([X], Y).
            fn flatten(mut e: &Expr, out: &mut Vec<String>) {
                match e {
                    Expr::Member { object, name, .. } => {
                        flatten(object, out);
                        out.push(name.clone());
                    }
                    Expr::Ident { name, .. } => {
                        out.push(name.clone());
                    }
                    _ => {}
                }
            }
            let mut segs: Vec<String> = Vec::new();
            flatten(object.as_ref(), &mut segs);
            if segs.is_empty() {
                return None;
            }
            // Enum case: second-to-last segment is the enum simple name
            // (`A.B` + `C` → enum `B`, variant `C`; `X` + `Y` → `X`, `Y`).
            if let Some(v) = enums
                .get(&(
                    segs.last().cloned().unwrap_or_default(),
                    name.clone(),
                ))
                .copied()
                .map(ConstValue::Int)
            {
                return Some(v);
            }
            // Class-associated const (`Type.Const`): resolve the object
            // (last segment) to a key (first simple-name match).
            // Primitive aliases (`short.MinValue` = `Int16.MinValue`)
            // map to their runtime type names first.
            let simple = segs.last().cloned().unwrap_or_default();
            let simple = match simple.as_str() {
                "bool" => "Boolean",
                "byte" => "Byte",
                "sbyte" => "SByte",
                "char" => "Char",
                "decimal" => "Decimal",
                "double" => "Double",
                "float" => "Single",
                "int" => "Int32",
                "uint" => "UInt32",
                "nint" => "IntPtr",
                "nuint" => "UIntPtr",
                "long" => "Int64",
                "ulong" => "UInt64",
                "short" => "Int16",
                "ushort" => "UInt16",
                "object" => "Object",
                "string" => "String",
                _ => simple.as_str(),
            };
            // Prefer the referrer's own scope (same namespace/outer):
            // five different `Keywords` classes exist; first-match
            // grabs the wrong one and its consts miss.
            let mut cands: Vec<&TypeKey> = cx
                .keys
                .iter()
                .filter(|k| k.simple_name() == simple)
                .collect();
            cands.sort_by_key(|k| {
                if k.scope_prefix() == cx.current.scope_prefix() {
                    0
                } else {
                    1
                }
            });
            cands.into_iter().find_map(|k| {
                cx.consts.get(&(k.clone(), name.clone())).cloned()
            })
        }
        // Bare const reference (`MaxTicks`, `DirectorySeparatorCharAsString`):
        // same class first, then base chain in order (C# scoping).
        Expr::Ident { name, .. } => cx
            .consts
            .get(&(cx.current.clone(), name.clone()))
            .cloned()
            .or_else(|| {
                cx.chains.get(&cx.current).and_then(|chain| {
                    chain.iter().find_map(|base| {
                        cx.consts.get(&(base.clone(), name.clone())).cloned()
                    })
                })
            }),
        Expr::Unary { op, inner, .. } => match op {
            UnOp::Neg => match const_fold(inner, cx) {
                Some(ConstValue::Int(v)) => Some(ConstValue::Int(-v)),
                Some(ConstValue::Float(v)) => Some(ConstValue::Float(-v)),
                Some(ConstValue::Str(_) | ConstValue::Null) | None => None,
            },
            UnOp::Not => match const_fold(inner, cx) {
                Some(ConstValue::Int(v)) => Some(ConstValue::Int(if v == 0 { 1 } else { 0 })),
                _ => None,
            },
            UnOp::BitNot => match const_fold(inner, cx) {
                Some(ConstValue::Int(v)) => Some(ConstValue::Int(!v)),
                _ => None,
            },
            _ => None,
        },
        // `sizeof(T)` for primitives (64-bit target): byte size.
        // (`sizeof` of composite types needs layout — not constant
        // here; those stay unresolvable.)
        //
        // `checked(x)` / `unchecked(x)` wrappers (e.g. framework consts
        // like `Int64.MinValue = unchecked((long)0x8000...L)`): fold
        // the single argument (wrapping assumed intended, mirroring
        // the cast arm above).
        Expr::Call { callee, args, .. }
            if (callee == "checked" || callee == "unchecked") && args.len() == 1 =>
        {
            const_fold(&args[0], cx)
        }
        Expr::Call { callee, args, .. } if callee == "sizeof" => {
            let [arg] = args.as_slice() else {
                return None;
            };
            let Expr::Ident { name, .. } = arg else {
                return None;
            };
            match name.as_str() {
                "bool" | "byte" | "sbyte" => Some(ConstValue::Int(1)),
                "short" | "ushort" | "char" => Some(ConstValue::Int(2)),
                "int" | "uint" | "float" => Some(ConstValue::Int(4)),
                "long" | "ulong" | "double" | "nint" | "nuint" | "IntPtr"
                | "UIntPtr" => Some(ConstValue::Int(8)),
                "decimal" => Some(ConstValue::Int(16)),
                _ => None,
            }
        }
        Expr::Binary { op, lhs, rhs, .. } => {
            let l = const_fold(lhs, cx)?;
            let r = const_fold(rhs, cx)?;
            // String concatenation (`"a" + "b"`) folds; mixing strings
            // with anything else is not constant.
            if matches!(op, BinOp::Add) {
                if let (ConstValue::Str(a), ConstValue::Str(b)) = (&l, &r) {
                    return Some(ConstValue::Str(format!("{a}{b}")));
                }
            }
            // Comparisons fold to 1/0 (`unchecked((int)(5023) <= 0 ?
            // ... : ...)` HRESULT macros in msquic interop).
            match (&l, &r) {
                (ConstValue::Int(a), ConstValue::Int(b)) => {
                    let cmp = match op {
                        BinOp::Eq => Some((*a == *b) as i64),
                        BinOp::Ne => Some((*a != *b) as i64),
                        BinOp::Lt => Some((*a < *b) as i64),
                        BinOp::Le => Some((*a <= *b) as i64),
                        BinOp::Gt => Some((*a > *b) as i64),
                        BinOp::Ge => Some((*a >= *b) as i64),
                        _ => None,
                    };
                    if let Some(v) = cmp {
                        return Some(ConstValue::Int(v));
                    }
                }
                (ConstValue::Float(a), ConstValue::Float(b)) => {
                    let cmp = match op {
                        BinOp::Eq => Some((*a == *b) as i64),
                        BinOp::Ne => Some((*a != *b) as i64),
                        BinOp::Lt => Some((*a < *b) as i64),
                        BinOp::Le => Some((*a <= *b) as i64),
                        BinOp::Gt => Some((*a > *b) as i64),
                        BinOp::Ge => Some((*a >= *b) as i64),
                        _ => None,
                    };
                    if let Some(v) = cmp {
                        return Some(ConstValue::Int(v));
                    }
                }
                _ => {}
            }
            match (l, r) {
                (ConstValue::Int(a), ConstValue::Int(b)) => {
                    let v = match op {
                        BinOp::Add => a.checked_add(b)?,
                        BinOp::Sub => a.checked_sub(b)?,
                        BinOp::Mul => a.checked_mul(b)?,
                        BinOp::Div => a.checked_div(b)?,
                        BinOp::Rem => a.checked_rem(b)?,
                        BinOp::BitAnd => a & b,
                        BinOp::BitOr => a | b,
                        BinOp::BitXor => a ^ b,
                        BinOp::Shl => a.checked_shl(b.try_into().ok()?)?,
                        BinOp::Shr => a.checked_shr(b.try_into().ok()?)?,
                        _ => return None,
                    };
                    Some(ConstValue::Int(v))
                }
                (ConstValue::Float(a), ConstValue::Float(b)) => {
                    let v = match op {
                        BinOp::Add => a + b,
                        BinOp::Sub => a - b,
                        BinOp::Mul => a * b,
                        BinOp::Div => a / b,
                        _ => return None,
                    };
                    Some(ConstValue::Float(v))
                }
                // Mixed int/float arithmetic (`s_DUINT_BASE2 *
                // s_DUINT_BASE` where one side folds from an int
                // literal): promote the int exactly like C# (int →
                // double is lossless below 2**53; SQLDecimal's powers
                // of two are exact).
                (ConstValue::Int(a), ConstValue::Float(b)) => {
                    let v = match op {
                        BinOp::Add => a as f64 + b,
                        BinOp::Sub => a as f64 - b,
                        BinOp::Mul => a as f64 * b,
                        BinOp::Div => a as f64 / b,
                        _ => return None,
                    };
                    Some(ConstValue::Float(v))
                }
                (ConstValue::Float(a), ConstValue::Int(b)) => {
                    let v = match op {
                        BinOp::Add => a + b as f64,
                        BinOp::Sub => a - b as f64,
                        BinOp::Mul => a * b as f64,
                        BinOp::Div => a / b as f64,
                        _ => return None,
                    };
                    Some(ConstValue::Float(v))
                }
                _ => None,
            }
        }
        // Ternary over constants (`unchecked(A <= B ? X : Y)` HRESULT
        // macros): fold the condition, take the live branch. Nonzero
        // integers and `true` select `then`; zero/`false`/empty/null
        // select `else`.
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            let live = match const_fold(cond, cx)? {
                ConstValue::Int(v) => v != 0,
                ConstValue::Float(v) => v != 0.0,
                ConstValue::Str(s) => !s.is_empty(),
                ConstValue::Null => false,
            };
            if live {
                const_fold(then_expr, cx)
            } else {
                const_fold(else_expr, cx)
            }
        }
        // Const interpolated string (C# 10+): `$"..."` is a valid const        // initializer when every hole is itself constant
        // (`XmlKeyHelper.IEnumerableOfXElementTypeName`). Holes render
        // invariantly: strings verbatim, bools `True`/`False`, null
        // empty, numerics decimal (floats approximate C#'s G format).
        Expr::InterpolatedString { parts, .. } => {
            use crate::models::ast::expression::InterpPart;
            let mut out = String::new();
            for part in parts {
                match part {
                    InterpPart::Text(t) => out.push_str(t),
                    InterpPart::Hole(h) => {
                        if let Expr::Literal(Literal::Bool(b)) = h.as_ref() {
                            out.push_str(if *b { "True" } else { "False" });
                            continue;
                        }
                        match const_fold(h, cx)? {
                            ConstValue::Str(s) => out.push_str(&s),
                            ConstValue::Int(v) => out.push_str(&v.to_string()),
                            ConstValue::Float(v) => out.push_str(&v.to_string()),
                            ConstValue::Null => {}
                        }
                    }
                }
            }
            Some(ConstValue::Str(out))
        }
        _ => None,
    }
}

/// A class declaration with its identity: namespace, outer-class path
/// (for nested types), and the declaration itself. Identity is
/// (namespace, path, name, arity) — same-named types in different
/// scopes or arities (`Task` vs `Task<TResult>`, `A.HashHelpers` vs
/// `B.HashHelpers`, `Outer.Inner` vs top-level `Inner`) are distinct
/// types. The flat bare-name model collides them
/// (false E0013 duplicates, false E0014 cycles).
struct ScopedClass {
    /// Enclosing namespace (`""` for global).
    ns: String,
    /// Outer class path from the top level (`[]` for top-level).
    outer: Vec<String>,
    decl: ClassDecl,
}

/// The index of each of a class's OWN type parameters inside the
/// instantiation's type-argument list. The base chain's arguments come
/// first (.NET order: a base class's `T` is argument 0), so a derived
/// class's own parameters follow the base chain's count.
fn own_type_arg_index(
    chain: &[crate::native::semantic::table::TypeKey],
    raw_classes: &[ScopedClass],
    decl: &ClassDecl,
) -> Vec<u32> {
    let base_args: u32 = chain
        .iter()
        .filter_map(|k| raw_classes.iter().find(|c| &c.key() == k))
        .map(|c| c.decl.generic_params.len() as u32)
        .sum();
    (0..decl.generic_params.len() as u32)
        .map(|i| base_args + i)
        .collect()
}

impl ScopedClass {
    fn key(&self) -> crate::native::semantic::table::TypeKey {
        use crate::native::semantic::table::TypeKey;
        let mut path = String::new();
        if !self.ns.is_empty() {
            path.push_str(&self.ns);
        }
        for o in &self.outer {
            if !path.is_empty() {
                path.push('.');
            }
            path.push_str(o);
        }
        if !path.is_empty() {
            path.push('.');
        }
        path.push_str(&self.decl.name);
        TypeKey::new(path, self.decl.generic_params.len())
    }
}

/// Analyze all classes/interfaces of a unit.
pub fn analyze(unit: &CompilationUnit) -> Result<TypeTable, Vec<SemanticError>> {
    analyze_with_owners(unit, &std::collections::HashMap::new())
}

/// Like [`analyze`], plus per-entry library ownership for
/// dynamic-linking: `owners` maps top-level member keys
/// (`TypeKey`, as the driver records per loaded file) to the owning
/// library name. Entries attribute exactly on key hit, else inherit
/// their nearest enclosing class entry's owner (nesting shares the
/// file's lib); unknown stays `""` (treated as local). The driver
/// records main-project members first so merged main+dep entries
/// attribute to the main library, mirroring content main-wins.
pub fn analyze_with_owners(
    unit: &CompilationUnit,
    owners: &std::collections::HashMap<
        crate::native::semantic::table::TypeKey,
        String,
    >,
) -> Result<TypeTable, Vec<SemanticError>> {
    analyze_with_preloaded(unit, owners, &Preloaded::empty())
}

/// Pre-resolved table entries loaded from dependency `.rci` fragments
/// (see `native::rci`) instead of compiling dep sources. Entries are
/// post-resolution (flattened members, folded consts, assigned slots)
/// and merge verbatim — never recomputed. The driver filters fragments
/// (freshness hash, main-key wins, first-wins across fragments) before
/// handing them here.
pub struct Preloaded {
    pub classes: Vec<crate::native::semantic::table::ClassInfo>,
    pub interfaces: Vec<crate::models::InterfaceDecl>,
    pub delegates: Vec<crate::models::DelegateDecl>,
}

impl Preloaded {
    pub fn empty() -> Self {
        Preloaded {
            classes: Vec::new(),
            interfaces: Vec::new(),
            delegates: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.classes.is_empty() && self.interfaces.is_empty() && self.delegates.is_empty()
    }
}

/// Like [`analyze`], plus per-entry library ownership for
/// dynamic-linking: `owners` maps top-level member keys
/// (`TypeKey`, as the driver records per loaded file) to the owning
/// library name. Entries attribute exactly on key hit, else inherit
/// their nearest enclosing class entry's owner (nesting shares the
/// file's lib); unknown stays `""` (treated as local). The driver
/// records main-project members first so merged main+dep entries
/// attribute to the main library, mirroring content main-wins.
pub fn analyze_with_preloaded(
    unit: &CompilationUnit,
    owners: &std::collections::HashMap<
        crate::native::semantic::table::TypeKey,
        String,
    >,
    preloaded: &Preloaded,
) -> Result<TypeTable, Vec<SemanticError>> {
    // Collect classes WITH namespace context (top-level members are
    // global-ns; namespace members carry their namespace). Flat
    // `all_members()` loses this, colliding same-named types.
    let mut scoped: Vec<ScopedClass> = Vec::new();
    for m in &unit.members {
        if let crate::models::Member::Class(c) = m {
            scoped.push(ScopedClass {
                ns: String::new(),
                outer: Vec::new(),
                decl: c.clone(),
            });
        }
    }
    for ns in &unit.namespaces {
        for m in &ns.members {
            if let crate::models::Member::Class(c) = m {
                scoped.push(ScopedClass {
                    ns: ns.name.clone(),
                    outer: Vec::new(),
                    decl: c.clone(),
                });
            }
        }
    }

    let mut interfaces: Vec<InterfaceDecl> = Vec::new();
    let mut raw_enums: Vec<(String, EnumDecl)> = Vec::new();
    // Nested enums too (`PayloadLengthFieldSize` inside
    // `EventManifestEntry`): const initializers name them qualified
    // (`A.B.C`), and top-level-only collection would miss them.
    // Paths are namespace-qualified (`System.Threading.StackCrawlMark`,
    // not bare `StackCrawlMark`) so qualified lookups hit their entries
    // like every other type.
    fn collect_enums(
        members: &[&crate::models::Member],
        prefix: &str,
        out: &mut Vec<(String, EnumDecl)>,
    ) {
        for &m in members {
            match m {
                crate::models::Member::Enum(e) => {
                    let path = if prefix.is_empty() {
                        e.name.clone()
                    } else {
                        format!("{prefix}.{}", e.name)
                    };
                    out.push((path, e.clone()));
                }
                crate::models::Member::Class(c) => {
                    let path = if prefix.is_empty() {
                        c.name.clone()
                    } else {
                        format!("{prefix}.{}", c.name)
                    };
                    let nested: Vec<&crate::models::Member> = c
                        .members
                        .iter()
                        .filter_map(|cm| match cm {
                            crate::models::ClassMember::Nested(n) => Some(n),
                            _ => None,
                        })
                        .collect();
                    collect_enums(&nested, &path, out);
                }
                _ => {}
            }
        }
    }
    for ns in &unit.namespaces {
        let members: Vec<&crate::models::Member> = ns.members.iter().collect();
        collect_enums(&members, &ns.name, &mut raw_enums);
    }
    {
        let members: Vec<&crate::models::Member> = unit.members.iter().collect();
        collect_enums(&members, "", &mut raw_enums);
    }
    for member in unit.all_members() {
        match member {
            crate::models::Member::Enum(_) => {}
            crate::models::Member::Delegate(_) => {
                // Delegates are callable types with no object layout; they are
                // not part of the class/interface tables — but they ARE
                // indexed (see delegates below) for method-group
                // convertibility.
            }
            crate::models::Member::Interface(i) => {
                // Duplicate interface names (same name AND same generic arity).
                // C# allows `IFoo` and `IFoo<T>` as distinct types.
                // `partial` parts merge (members + bases), mirroring
                // classes (`IDtdParserAdapter` across sync/async files);
                // anything else sharing the key is a genuine duplicate.
                if let Some(existing) = interfaces
                    .iter_mut()
                    .find(|e| e.name == i.name && e.arity == i.arity)
                {
                    if existing.is_partial && i.is_partial {
                        existing.methods.extend(i.methods.clone());
                        existing.indexers.extend(i.indexers.clone());
                        existing.bases.extend(i.bases.clone());
                    } else {
                        return Err(vec![err(
                            "E0012",
                            format!("interface '{}' is declared more than once", i.name),
                        )]);
                    }
                } else {
                    interfaces.push(i.clone());
                }
            }
            crate::models::Member::Class(_) => {}
        }
    }
    // Preloaded dependency interfaces (main-wins on (name, arity)
    // collision; cross-lib partials don't exist in practice —
    // partials are same-assembly by language rule).
    for i in &preloaded.interfaces {
        if !interfaces
            .iter()
            .any(|e| e.name == i.name && e.arity == i.arity)
        {
            interfaces.push(i.clone());
        }
    }

    let mut errors: Vec<SemanticError> = Vec::new();

    // Merge partial class declarations: same (namespace, path, name,
    // arity), both partial → combine members and bases. Anything else
    // sharing a key is a genuine duplicate (E0013); different keys
    // (arity/namespace/nesting) are distinct types, never merged.
    let mut merged: Vec<ScopedClass> = Vec::new();
    for c in scoped {
        let key = c.key();
        if let Some(existing) = merged
            .iter_mut()
            .find(|e| e.key() == key)
        {
            if existing.decl.is_partial && c.decl.is_partial {
                // Cross-assembly provenance survives the merge: a part
                // that absorbed another assembly's same-path class
                // (driver dep-union) merging here means the entry spans
                // assemblies, and each side keeps its constructibility
                // below.
                let cross = existing.decl.union_cross_assembly || c.decl.union_cross_assembly;
                let has_instance_ctor = |ms: &[crate::models::ClassMember], n: &str| {
                    ms.iter().any(|m| match m {
                        crate::models::ClassMember::Method(x) => {
                            x.name == n && !x.is_static
                        }
                        _ => false,
                    })
                };
                let e_had = has_instance_ctor(&existing.decl.members, &existing.decl.name);
                let i_had = has_instance_ctor(&c.decl.members, &c.decl.name);
                existing.decl.members.extend(c.decl.members);
                existing.decl.bases.extend(c.decl.bases);
                existing.decl.union_cross_assembly =
                    existing.decl.union_cross_assembly || c.decl.union_cross_assembly;
                if cross
                    && e_had != i_had
                    && !existing.decl.members.iter().any(|m| match m {
                        crate::models::ClassMember::Method(x) => {
                            x.name == existing.decl.name
                                && x.params.is_empty()
                                && !x.is_static
                        }
                        _ => false,
                    })
                {
                    // Exactly one side declared instance ctors, so the
                    // other relied on the implicit parameterless one —
                    // which exists only as absence and would be lost in
                    // the union, breaking that side's `new X()` with a
                    // loud `ctor_arity` (MLC vs CoreLib
                    // `System.DefaultBinder`). Preserve it as an
                    // explicit empty-body ctor (indistinguishable from
                    // a source-written `X() { }`). Pure same-assembly
                    // partials (no provenance flag) never take this
                    // path: splitting ctors across parts is normal
                    // there, and `new X()` stays an error as C#
                    // requires.
                    let name = existing.decl.name.clone();
                    existing.decl.members.push(crate::models::ClassMember::Method(
                        crate::models::MethodDecl {
                            access: crate::models::Access::Internal,
                            attrs: Vec::new(),
                            generic_params: Vec::new(),
                            constraints: Vec::new(),
                            ret_type: String::new(),
                            name,
                            params: Vec::new(),
                            body: Some(crate::models::Block {
                                stmts: Vec::new(),
                                span: crate::models::Span::default(),
                            }),
                            span: crate::models::Span::default(),
                            is_override: false,
                            is_virtual: false,
                            is_sealed: false,
                            is_hiding: false,
                            is_static: false,
                            is_abstract: false,
                            is_partial: false,
                            is_extern: false,
                            is_async: false,
                            base_call: None,
                        },
                    ));
                }
            } else {
                errors.push(err(
                    "E0013",
                    format!("class '{}' is declared more than once", c.decl.name),
                ));
            }
        } else {
            merged.push(c);
        }
    }
    // Flatten nested classes with qualified paths (`Outer.Inner`),
    // recursing (nested-in-nested). Queue-based to handle depth.
    // Keys distinguish arity/namespace/nesting (`Task` vs `Task<T>`,
    // `A.HashHelpers` vs `B.HashHelpers`), ending false E0013/E0014.
    use crate::native::semantic::table::TypeKey;
    let mut raw_classes: Vec<ScopedClass> = merged;
    let mut seen_keys: BTreeSet<TypeKey> = BTreeSet::new();
    for c in &raw_classes {
        seen_keys.insert(c.key());
    }
    let mut queue: Vec<usize> = (0..raw_classes.len()).collect();
    while let Some(idx) = queue.pop() {
        // Clone the parent context (ns + outer path) for children.
        let (parent_ns, parent_outer, parent_name) = {
            let p = &raw_classes[idx];
            (p.ns.clone(), p.outer.clone(), p.decl.name.clone())
        };
        let members = raw_classes[idx].decl.members.clone();
        for m in members {
            if let crate::models::ClassMember::Nested(member) = m {
                match member {
                    crate::models::Member::Class(nc) => {
                        let mut outer = parent_outer.clone();
                        outer.push(parent_name.clone());
                        let child = ScopedClass {
                            ns: parent_ns.clone(),
                            outer,
                            decl: nc,
                        };
                        if seen_keys.insert(child.key()) {
                            queue.push(raw_classes.len());
                            raw_classes.push(child);
                        } else if child.decl.is_partial {
                            // Same-key nested partial part (e.g.
                            // `AsyncProfiler.AsyncCallstack` split across
                            // the libraries and CoreCLR parts): merge
                            // members/bases into the winning entry like
                            // top-level partials, or one part's nesteds
                            // (and their bases) vanish. Re-queue the
                            // winner: it may already have been walked,
                            // and the merged-in members need discovery
                            // (idempotent via `seen_keys`/interface
                            // dedup; each part merges once, so this
                            // terminates).
                            let key = child.key();
                            if let Some(existing) =
                                raw_classes.iter_mut().find(|e| e.key() == key)
                            {
                                if existing.decl.is_partial {
                                    existing.decl.members.extend(child.decl.members);
                                    existing.decl.bases.extend(child.decl.bases);
                                    let pos = raw_classes
                                        .iter()
                                        .position(|e| e.key() == key)
                                        .expect("nested partial winner present");
                                    queue.push(pos);
                                }
                            }
                        }
                    }
                    crate::models::Member::Interface(ni) => {
                        // Same name+arity dedup as top-level interfaces.
                        if !interfaces
                            .iter()
                            .any(|e| e.name == ni.name && e.arity == ni.arity)
                        {
                            interfaces.push(ni);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // ---- pass 1: register every class key (bases may forward-reference).
    // Duplicates (same key, non-partial—partials already merged) are E0013.
    // Distinct keys (arity/namespace/nesting) coexist without error.
    // Filtered preload set: preloaded entries whose key collides with
    // a raw (compiled) key are dropped — raw wins (main declarations
    // and unconsumed-dep sources take precedence over fragment
    // copies; the driver additionally filters top-level main keys
    // from fragments up front). Within fragments, first-wins by key
    // (driver concatenates in dep order; shared files carry identical
    // content). Everything below consumes `pre`, never raw
    // `preloaded`, so behavior is uniform.
    let raw_keys: std::collections::BTreeSet<TypeKey> =
        raw_classes.iter().map(|c| c.key()).collect();
    let mut pre_seen: std::collections::BTreeSet<TypeKey> = std::collections::BTreeSet::new();
    let pre: Vec<&crate::native::semantic::table::ClassInfo> = preloaded
        .classes
        .iter()
        .filter(|c| !raw_keys.contains(&c.key) && pre_seen.insert(c.key.clone()))
        .collect();
    let pre_keys: std::collections::BTreeSet<TypeKey> =
        pre.iter().map(|c| c.key.clone()).collect();
    let mut owned: BTreeSet<TypeKey> = BTreeSet::new();
    for c in &raw_classes {
        if !owned.insert(c.key()) {
            errors.push(err(
                "E0013",
                format!("class '{}' is declared more than once", c.decl.name),
            ));
        }
    }
    // Register interface names too, so that bases which are interfaces
    // (e.g. `class Foo : IBar`) resolve instead of triggering E0011.
    // Interfaces have no base chain of their own in this model.
    // Nested-class keys (for prefer_visible in base resolution): a
    // nested helper must not shadow a top-level type by sort order.
    let nested: std::collections::BTreeSet<TypeKey> = raw_classes
        .iter()
        .filter(|c| !c.outer.is_empty())
        .map(|c| c.key())
        .collect();
    for i in &interfaces {
        owned.insert(TypeKey::new(i.name.clone(), i.arity));
    }
    // Preloaded dependency keys resolve like compiled ones (dep bases
    // of main classes, `Type.Const` member objects, interface bases).
    // Nested preloaded keys stay out of the `nested` visibility set
    // below (lenient: treated as visible — the demotion heuristic
    // needs raw nesting info we don't synthesize for fragments).
    for k in &pre_keys {
        owned.insert(k.clone());
    }
    for i in &preloaded.interfaces {
        owned.insert(TypeKey::new(i.name.clone(), i.arity));
    }
    // `object` is the implicit root of every class and `ValueType` the
    // implicit root of every struct — both are language rules, so the
    // compiler knows them without a library (C# spec §8.9.1, §15.2.4).
    // Every OTHER framework type must come from a real library: an
    // `Exception`/`IDisposable`/`Enum`/`Delegate`/`Attribute`/`String`
    // base that no declaration and no loaded fragment provides is
    // simply unresolved, and the compiler reports it.
    for (ns_path, arity) in [("System.Object", 0), ("System.ValueType", 0)] {
        owned.insert(TypeKey::new(ns_path, arity));
    }

    let mut table = TypeTable {
        // Preloaded dependency entries first: lookups hit them for
        // dep bases/members, and pass 3 appends compiled entries after
        // (first-match then prefers fragments — correct for shared
        // content; the driver filters top-level main keys from
        // fragments up front so main declarations never shadow this
        // way). Raw-key collisions are belt-and-suspenders skipped.
        classes: {
            let raw_keys: std::collections::BTreeSet<TypeKey> =
                raw_classes.iter().map(|c| c.key()).collect();
            preloaded
                .classes
                .iter()
                .filter(|c| !raw_keys.contains(&c.key))
                .cloned()
                .collect()
        },
        interfaces,
        delegates: {
            let mut ds: Vec<crate::models::DelegateDecl> = unit
                .all_members()
                .into_iter()
                .filter_map(|m| match m {
                    crate::models::Member::Delegate(d) => Some(d.clone()),
                    _ => None,
                })
                .collect();
            for d in &preloaded.delegates {
                if !ds
                    .iter()
                    .any(|e| e.name == d.name && e.generic_params.len() == d.generic_params.len())
                {
                    ds.push(d.clone());
                }
            }
            ds
        },
        imports: Vec::new(),
        aliases: {
            // Generic type parameter names declared anywhere in the
            // unit (classes incl. nested, methods, interfaces,
            // delegates). A `using` alias colliding with one (`using T
            // = XmlQueryTypeFactory`, file-scoped by C# rules) must
            // not shadow it globally — the alias is dropped (its own
            // file's uses degrade to placeholders; everywhere else
            // `T` stays an open variable instead of resolving to the
            // alias target and failing member lookup).
            fn collect_vars(
                members: &[&crate::models::Member],
                out: &mut std::collections::BTreeSet<String>,
            ) {
                for m in members {
                    match *m {
                        crate::models::Member::Class(c) => {
                            out.extend(c.generic_params.iter().cloned());
                            for cm in &c.members {
                                match cm {
                                    crate::models::ClassMember::Method(md) => {
                                        out.extend(md.generic_params.iter().cloned())
                                    }
                                    crate::models::ClassMember::Nested(n) => {
                                        collect_vars(&[n], out)
                                    }
                                    _ => {}
                                }
                            }
                        }
                        crate::models::Member::Interface(i) => {
                            out.extend(i.generic_params.iter().cloned())
                        }
                        crate::models::Member::Delegate(d) => {
                            out.extend(d.generic_params.iter().cloned())
                        }
                        _ => {}
                    }
                }
            }
            let mut type_vars = std::collections::BTreeSet::new();
            collect_vars(&unit.all_members(), &mut type_vars);
            unit.aliases
                .iter()
                .filter(|a| !type_vars.contains(&a.name))
                .map(|a| (a.name.clone(), a.target.clone()))
                .collect()
        },
    };

    // ---- pass 0: [LibraryImport] stubs.
    for class in &raw_classes {
        for member in &class.decl.members {
            let crate::models::ClassMember::Method(method) = member else {
                continue;
            };
            let attr = method.attrs.iter().find(|a| {
                // Attribute names may be qualified
                // (`System.Runtime.InteropServices.LibraryImport`);
                // match the simple name.
                a.name.rsplit('.').next().unwrap_or(&a.name) == "LibraryImport"
            });
            let Some(attr) = attr else {
                if method.body.is_none()
                    && !method.is_abstract
                    && !method.is_partial
                    && !method.is_extern
                {
                    errors.push(err_help(
                        "E0019",
                        format!(
                            "method '{}.{}' has no body; only [LibraryImport] stubs may be bodyless",
                            class.decl.name, method.name
                        ),
                        "mark it [LibraryImport(\"lib\", EntryPoint = \"sym\")] if it wraps an external function",
                    ));
                }
                continue;
            };
            if method.body.is_some() {
                errors.push(err_help(
                    "E0020",
                    format!("[LibraryImport] '{}' must not have a body", method.name),
                    "remove the body — the linker binds the external symbol",
                ));
                continue;
            }
            // Positional arg 0 (or `library`) names the library; positional
            // 1 / `entryPoint` overrides the entry point.
            let mut library = None;
            let mut entry_point = None;
            for arg in &attr.arguments {
                match (&arg.name, &arg.value) {
                    (None, crate::models::AttributeValue::Str(s)) if library.is_none() => {
                        library = Some(s.clone());
                    }
                    (None, crate::models::AttributeValue::Str(s)) if entry_point.is_none() => {
                        entry_point = Some(s.clone());
                    }
                    (Some(k), crate::models::AttributeValue::Str(s))
                        if k.eq_ignore_ascii_case("library") =>
                    {
                        library = Some(s.clone());
                    }
                    (Some(k), crate::models::AttributeValue::Str(s))
                        if k.eq_ignore_ascii_case("entrypoint") =>
                    {
                        entry_point = Some(s.clone());
                    }
                    _ => {}
                }
            }
            let Some(library) = library else {
                    errors.push(err_help(
                        "E0021",
                        format!(
                            "[LibraryImport] on '{}.{}' requires a library name",
                            class.decl.name, method.name
                        ),
                    "add a positional string: [LibraryImport(\"mylib\")]",
                ));
                continue;
            };
            table.imports.push(ImportFn {
                class: class.decl.name.clone(),
                method: method.name.clone(),
                library,
                entry_point: entry_point.unwrap_or_else(|| method.name.clone()),
                ret_type: method.ret_type.clone(),
                param_types: method.params.iter().map(|p| p.type_name.clone()).collect(),
            });
        }
    }

    // ---- pass 2: resolve base chains with cycle detection.
    // Chains hold qualified keys; a repeat of the SAME key is a true
    // cycle, while same-named distinct types (`Task` vs `Task<T>`)
    // resolve separately and never falsely cycle.
    struct Resolved {
        chain: Vec<TypeKey>, // direct bases then transitive, order-stable
    }
    /// Order ambiguous same-key candidates so VISIBLE ones come first:
    /// top-level types always are; a nested type is visible only from
    /// inside its outer (`Other.Holder` or deeper). A nested helper
    /// (`DelegateHelpers.TypeInfo`) must never beat the top-level type
    /// a bare base name means (`System.Reflection.TypeInfo`) merely by
    /// sorting first — yet it stays selectable as a last resort, so
    /// genuinely nested-only references keep today's lenient global
    /// fallback instead of erroring.
    fn prefer_visible<'a>(
        mut cands: Vec<&'a TypeKey>,
        context: &TypeKey,
        nested: &std::collections::BTreeSet<TypeKey>,
    ) -> Vec<&'a TypeKey> {
        cands.sort_by_key(|k| {
            let invisible = nested.contains(*k)
                && *k != context
                && {
                    let outer = k.scope_prefix();
                    context.path != outer
                        && !context.path.starts_with(&format!("{outer}."))
                };
            invisible as u8
        });
        cands
    }
    /// Resolve a base-class string (`Ns.Foo<T>`, `Bar`, `A.B`) to a
    /// declared key, preferring `context` (the derived class: same
    /// namespace/outer scope first, then global). `None` when unknown.
    fn resolve_base(
        base: &str,
        context: &TypeKey,
        owned: &BTreeSet<TypeKey>,
        nested: &std::collections::BTreeSet<TypeKey>,
    ) -> Option<TypeKey> {
        use crate::native::semantic::table::split_arity;
        let (path, arity) = split_arity(base);
        // Qualified reference: exact path + arity, else PATH-SUFFIX
        // match (`X86Base.X64` written inside `System...Sse` resolves
        // to `...X86Base.X64`, not to a sibling `...Sse.X64`).
        if path.contains('.') {
            let key = TypeKey::new(path.clone(), arity);
            if owned.contains(&key) {
                return Some(key);
            }
            let suffix = format!(".{path}");
            let mut hits: Vec<&TypeKey> = owned
                .iter()
                .filter(|k| {
                    k.arity == arity
                        && (k.path == path || k.path.ends_with(suffix.as_str()))
                })
                .collect();
            if hits.len() == 1 {
                return Some(hits.swap_remove(0).clone());
            }
            if hits.len() > 1 {
                // Prefer the referrer's own scope among suffix hits.
                let hits = prefer_visible(hits, context, nested);
                if let Some(k) = hits
                    .iter()
                    .find(|k| {
                        k.scope_prefix() == context.scope_prefix()
                            && *k != &context
                    })
                {
                    return Some((*k).clone());
                }
                // Otherwise fall through to simple-name search below
                // (which applies the same scope preference globally).
            }
            // Fall through to simple-name search.
        }
        // Simple name (+arity): candidates across scopes.
        let simple = path.rsplit('.').next().unwrap_or(&path);
        let mut candidates: Vec<&TypeKey> = owned
            .iter()
            .filter(|k| k.simple_name() == simple && k.arity == arity)
            .collect();
        if candidates.is_empty() {
            return None;
        }
        if candidates.len() == 1 {
            return Some(candidates.swap_remove(0).clone());
        }
        // Ambiguous: prefer the derived class's own scope, then global.
        // Invisible nesteds sort last first (see prefer_visible).
        let candidates = prefer_visible(candidates, context, nested);
        if let Some(k) = candidates
            .iter()
            .find(|k| k.scope_prefix() == context.scope_prefix())
        {
            return Some((*k).clone());
        }
        if let Some(k) = candidates.iter().find(|k| k.scope_prefix().is_empty()) {
            return Some((*k).clone());
        }
        Some(candidates.into_iter().next().unwrap().clone())
    }
    fn resolve_chain(
        key: &TypeKey,
        owned: &BTreeSet<TypeKey>,
        raw: &BTreeMap<TypeKey, Vec<String>>,
        seen: &mut Vec<TypeKey>,
        errors: &mut Vec<SemanticError>,
        nested: &BTreeSet<TypeKey>,
    ) -> Vec<TypeKey> {
        if seen.iter().any(|s| s == key) {
            errors.push(err(
                "E0014",
                format!("inheritance cycle at '{}'", key.path),
            ));
            return Vec::new();
        }
        seen.push(key.clone());
        let mut chain = Vec::new();
        if let Some(direct) = raw.get(key) {
            for base in direct {
                let Some(resolved) = resolve_base(base, key, owned, &nested) else {
                    errors.push(err(
                        "E0011",
                        format!("unknown base '{base}' for class '{}'", key.path),
                    ));
                    continue;
                };
                // Skip interface bases here: interfaces have no base
                // chain of their own in this model (handled separately
                // when building ClassInfo.interfaces).
                // (Checked via table below; chains only track classes.)
                chain.push(resolved.clone());
                for inherited in resolve_chain(&resolved, owned, raw, seen, errors, &nested) {
                    if !chain.contains(&inherited) {
                        chain.push(inherited);
                    }
                }
            }
        }
        // C# implicit inheritance: a class/struct with no explicit class
        // base derives from `object` (its members — `GetType`,
        // `ToString`, `Equals`, ... — resolve through the chain). This
        // lives INSIDE the recursion so transitive bases get it too
        // (`Stream : Mid` with baseless `Mid` chains to `object`).
        // `object` itself is the root and stays baseless; units without
        // an `object` declaration (tiny tests) are unaffected.
        if chain.is_empty() && key.simple_name() != "Object" {
            let mut cands: Vec<&TypeKey> = owned
                .iter()
                .filter(|k| k.simple_name() == "Object" && k.arity == 0)
                .collect();
            cands.sort_by_key(|k| {
                if k.scope_prefix() == key.scope_prefix() {
                    0
                } else {
                    1
                }
            });
            if let Some(obj) = cands.into_iter().next() {
                chain.push(obj.clone());
            }
        }
        seen.pop();
        chain.retain(|b| b != key);
        chain
    }

    let mut raw_map: BTreeMap<TypeKey, Vec<String>> = BTreeMap::new();
    for c in &raw_classes {
        raw_map
            .entry(c.key())
            .or_default()
            .extend(c.decl.bases.iter().cloned());
    }
    // Preloaded dependency bases: transitive chain walks stop at the
    // dep boundary with the fragment's own (already-flattened) members,
    // so only the dep's DIRECT base strings are needed here (its
    // transitive content rides along verbatim). Skip keys raw already
    // covers (main-wins, mirroring the table insert above).
    for c in pre.iter().copied() {
        raw_map
            .entry(c.key.clone())
            .or_insert_with(|| c.bases.clone());
    }
    let mut resolved: Vec<Resolved> = Vec::new();
    for c in &raw_classes {
        let mut seen = Vec::new();
        let key = c.key();
        let chain = resolve_chain(&key, &owned, &raw_map, &mut seen, &mut errors, &nested);
        resolved.push(Resolved { chain });
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    // ---- pass 3a: multiple-inheritance conflict detection.
    // When a class has more than one direct base, check for ambiguous
    // method overrides and field redefinitions. Bases resolve to keys
    // (same-namespace/arity aware); unresolvable bases were already
    // reported as E0011 and are skipped here.
    for c in &raw_classes {
        if c.decl.bases.len() <= 1 {
            continue;
        }
        // Collect method names declared by each direct base.
        let mut method_sources: Vec<(String, &ClassDecl)> = Vec::new();
        let context = c.key();
        for base_name in &c.decl.bases {
            let Some(key) = resolve_base(base_name, &context, &owned, &nested) else {
                continue;
            };
            if let Some(bc) = raw_classes
                .iter()
                .find(|cc| cc.key() == key)
                .map(|cc| &cc.decl)
            {
                method_sources.push((base_name.clone(), bc));
            }
        }
        // Ambiguous method: two bases declare the same method name.
        let mut seen_methods: Vec<(String, String)> = Vec::new(); // (method_name, first_base)
        for (bname, bc) in &method_sources {
            for m in &bc.members {
                if let crate::models::ClassMember::Method(mm) = m {
                    if mm.ret_type.is_empty() {
                        continue; // constructors
                    }
                    if let Some((_, first_base)) =
                        seen_methods.iter().find(|(n, _)| n == &mm.name)
                    {
                        if first_base != bname {
                            errors.push(err_help(
                                "E0028",
                                format!(
                                    "class '{}' inherits method '{}' from both '{}' and '{}'",
                                    c.decl.name, mm.name, first_base, bname
                                ),
                                "override the method in the derived class to resolve the ambiguity",
                            ));
                        }
                    } else {
                        seen_methods.push((mm.name.clone(), bname.clone()));
                    }
                }
            }
        }
        // Ambiguous field: two bases declare the same field name.
        let mut seen_fields: Vec<(String, String)> = Vec::new(); // (field_name, first_base)
        for (bname, bc) in &method_sources {
            for m in &bc.members {
                if let crate::models::ClassMember::Field(f) = m {
                    if let Some((_, first_base)) =
                        seen_fields.iter().find(|(n, _)| n == &f.name)
                    {
                        if first_base != bname {
                            errors.push(err_help(
                                "E0029",
                                format!(
                                    "class '{}' inherits field '{}' from both '{}' and '{}'",
                                    c.decl.name, f.name, first_base, bname
                                ),
                                "hide the field with a same-named field in the derived class, or rename one base's field",
                            ));
                        }
                    } else {
                        seen_fields.push((f.name.clone(), bname.clone()));
                    }
                }
            }
        }
    }

    // Precompute enum variant values for const-folding (`BindingFlags.X`
    // in const initializers). Same auto-numbering as pass 3b below (which
    // reuses this map); explicit values reset the counter, matching C#.
    let mut enum_values: std::collections::BTreeMap<(String, String), i64> =
        Default::default();
    for (_, e) in &raw_enums {
        let mut next: i64 = 0;
        for (name, explicit) in &e.variants {
            let value = match explicit {
                Some(v) => {
                    next = v + 1;
                    *v
                }
                None => {
                    let v = next;
                    next = v + 1;
                    v
                }
            };
            enum_values.insert((e.name.clone(), name.clone()), value);
        }
    }
    // Preloaded dependency enums seed the value map (their ClassInfo
    // const_fields carry folded values verbatim). First-wins: compiled
    // enums keep precedence on (enum, variant) collision, mirroring
    // main-wins elsewhere. Only Int consts map (the table is i64 —
    // same pre-existing limitation as compiled enums).
    for c in pre.iter().copied() {
        if !c.is_enum {
            continue;
        }
        for f in &c.const_fields {
            if let Some(ConstValue::Int(v)) = f.const_value {
                enum_values.entry((c.name.clone(), f.name.clone())).or_insert(v);
            }
        }
    }

    // ---- pass 2b: resolve class constants (fixpoint).
    // `const` initializers may reference enums (`BindingFlags.X`),
    // other consts (same class, bases, or `Type.Const`), strings,
    // and casts — in any declaration order. Iterate to fixpoint
    // (each round resolves those whose dependencies are known);
    // leftovers are genuinely unresolvable (E0030 in pass 3).
    // Chains map + key list for scoped const lookup.
    let mut chains_map: std::collections::BTreeMap<TypeKey, Vec<TypeKey>> =
        Default::default();
    for (sc, res) in raw_classes.iter().zip(&resolved) {
        chains_map.insert(sc.key(), res.chain.clone());
    }
    let all_keys: Vec<TypeKey> = raw_classes
        .iter()
        .map(|sc| sc.key())
        // Preloaded dependency keys participate in `Type.Const`
        // member-object resolution exactly like compiled ones
        // (filtered set — belt-dropped keys resolve via raw).
        .chain(pre.iter().map(|c| c.key.clone()))
        .collect();
    // (class key, const name, declared type, init expr) for every const
    // field. The declared type converts folded values at insert (C#
    // implicit constant conversion — `double X = <long>` stores Float,
    // so downstream folds see the converted value, not the raw Int).
    let mut pending: Vec<(
        TypeKey,
        String,
        String,
        &crate::models::ast::expression::Expr,
    )> = Vec::new();
    for sc in &raw_classes {
        for m in &sc.decl.members {
            if let crate::models::ClassMember::Field(f) = m {
                if f.is_const {
                    pending.push((sc.key(), f.name.clone(), f.type_name.clone(), &f.init));
                }
            }
        }
    }
    let mut const_values: std::collections::BTreeMap<(TypeKey, String), ConstValue> =
        Default::default();
    // Preloaded dependency consts seed the fixpoint map (already
    // folded + declared-type-converted at their own build). First-wins:
    // compiled entries overwrite below only via their own keys (no
    // key sharing — driver filters main keys from fragments).
    // Seeding (not just map insertion at use sites) matters: main
    // const initializers reference dep consts mid-fixpoint.
    for c in pre.iter().copied() {
        for f in c
            .const_fields
            .iter()
            .chain(c.fields.iter().filter(|f| f.is_const))
        {
            if let Some(v) = &f.const_value {
                const_values
                    .entry((c.key.clone(), f.name.clone()))
                    .or_insert_with(|| v.clone());
            }
        }
    }
    for _round in 0..=pending.len() {
        let mut progressed = false;
        let mut still_pending = Vec::new();
        for (key, name, type_name, init) in pending {
            let cx = ConstCx {
                enums: &enum_values,
                consts: &const_values,
                chains: &chains_map,
                keys: &all_keys,
                current: key.clone(),
            };
            match const_fold(init, &cx) {
                Some(v) => {
                    const_values.insert((key, name), convert_const(v, &type_name));
                    progressed = true;
                }
                None => still_pending.push((key, name, type_name, init)),
            }
        }
        pending = still_pending;
        if !progressed {
            break;
        }
    }
    // (Unresolved leftovers simply stay absent; pass 3 reports E0030.)

    // ---- pass 3: build ClassInfo (layout + methods + access data).
    // Chains hold qualified keys; sources resolve by key (distinct
    // types never conflate, ending layout corruption from collisions).
    for (decl, res) in raw_classes.iter().zip(&resolved) {
        let mut fields: Vec<FieldInfo> = Vec::new();
        let mut static_fields: Vec<FieldInfo> = Vec::new();
        let mut const_fields: Vec<FieldInfo> = Vec::new();
        let mut methods: Vec<MethodInfo> = Vec::new();
        let mut properties: Vec<PropertyInfo> = Vec::new();

        // Layout walks the full base chain first (declaration order), then
        // own fields. Duplicate names keep the FIRST occurrence.
        let mut field_sources: Vec<(&ScopedClass,)> = Vec::new();
        for base_key in &res.chain {
            if let Some(bc) = raw_classes.iter().find(|c| &c.key() == base_key) {
                field_sources.push((bc,));
            }
        }
        field_sources.push((decl,));

        // Preloaded dependency members (chain order, verbatim copies):
        // the raw loop below then appends/updates own + raw-base
        // members on top with its unchanged rules. Verbatim offsets,
        // slots, and folded values keep cross-image agreement (the
        // consumer never recomputes dep layout). Static fields are
        // NOT copied (per-image storage; cross-image static access
        // errors loudly at lowering). own_members stay own-only
        // (built from `decl` below, mirroring the raw path).
        // First-wins dedup + hidden-update-in-place mirror the raw
        // rules exactly; only the value source differs (recorded vs
        // computed).
        if !preloaded.classes.is_empty() {
            for base_key in &res.chain {
                let Some(pre) =
                    pre.iter().copied().find(|c| c.key == *base_key)
                else {
                    continue;
                };
                for f in &pre.fields {
                    if let Some(existing) = fields.iter_mut().find(|fi| fi.name == f.name) {
                        existing.type_name = f.type_name.clone();
                        existing.access = f.access;
                        existing.declaring_class = f.declaring_class.clone();
                        continue;
                    }
                    fields.push(f.clone());
                }
                for f in &pre.const_fields {
                    if let Some(existing) = const_fields
                        .iter_mut()
                        .find(|fi| fi.name == f.name)
                    {
                        existing.type_name = f.type_name.clone();
                        existing.access = f.access;
                        existing.const_value = f.const_value.clone();
                        existing.declaring_class = f.declaring_class.clone();
                        continue;
                    }
                    const_fields.push(f.clone());
                }
                for m in &pre.methods {
                    // Byref-ness joins the key (see the raw loop below):
                    // value-vs-`ref` twins share erased param types.
                    if methods.iter().any(|mi| {
                        mi.name == m.name
                            && mi.param_types == m.param_types
                            && mi.param_is_byref == m.param_is_byref
                            && mi.declaring_class == m.declaring_class
                    }) {
                        continue;
                    }
                    methods.push(m.clone());
                }
                for p in &pre.properties {
                    if let Some(existing) =
                        properties.iter_mut().find(|pi| pi.name == p.name)
                    {
                        existing.type_name = p.type_name.clone();
                        existing.access = p.access;
                        // declaring_class keeps the first (most-derived)
                        // declarer; see the raw loop above.
                        continue;
                    }
                    properties.push(p.clone());
                }
            }
        }
        // Canonical instance-field slot order (root-first): slot
        // numbers must agree across every entry in a hierarchy
        // (a base ctor stores via the base entry while reads use
        // the derived entry). Chain order is direct-first, so
        // positional numbering (`fields.len() + 2` below) puts the
        // same field at different slots per entry (faulted
        // `ConsoleStream._canWrite`: stored at slot 9 via the base
        // entry, read at slot 3 via the derived entry). Pre-assign
        // root-first here; the loop below uses these instead.
        // Seeded with already-pushed preloaded (verbatim) fields so
        // recomputed slots continue after them. Method, property,
        // static, and const handling keep their existing
        // direct-first order (override/hiding resolution depends on
        // it); only instance slot numbers change. First occurrence
        // wins, mirroring the loop's dedup.
        let mut field_slot_order: Vec<String> = Vec::new();
        for fi in &fields {
            if !field_slot_order.iter().any(|n| n == &fi.name) {
                field_slot_order.push(fi.name.clone());
            }
        }
        {
            // Ancestors first, then this class. `res.chain` is ordered
            // "a base, then ITS ancestors" per declared base, so for
            // `C : A, B` it reads [A, Object, B]: reversing it put B
            // BEFORE A and the second base's fields landed ahead of the
            // first base's. Sorting by inheritance depth (stable, so
            // declaration order breaks ties) gives the order a layout
            // needs — Object, A, B, C — for any number of bases.
            let depth = |k: &crate::native::semantic::table::TypeKey| -> usize {
                let mut d = 0usize;
                let mut cur = Some(k.clone());
                while let Some(c) = cur {
                    let Some(parent) = raw_classes
                        .iter()
                        .find(|sc| &sc.key() == &c)
                        .and_then(|sc| sc.decl.bases.first())
                    else {
                        break;
                    };
                    let simple = parent.split('<').next().unwrap_or(parent).trim();
                    cur = raw_classes
                        .iter()
                        .find(|sc| sc.decl.name == simple)
                        .map(|sc| sc.key());
                    d += 1;
                    if d > 64 {
                        break;
                    }
                }
                d
            };
            let mut chain: Vec<&ScopedClass> = res
                .chain
                .iter()
                .filter_map(|base_key| {
                    raw_classes.iter().find(|c| &c.key() == base_key)
                })
                .collect();
            chain.sort_by_key(|sc| depth(&sc.key()));
            let mut root_first: Vec<&ScopedClass> = chain;
            root_first.push(decl);
            for src in root_first {
                for m in &src.decl.members {
                    match m {
                        crate::models::ClassMember::Field(f)
                            if !f.is_static && !f.is_const =>
                        {
                            if !field_slot_order.iter().any(|n| n == &f.name) {
                                field_slot_order.push(f.name.clone());
                            }
                        }
                        crate::models::ClassMember::Property(p)
                            if p.get.is_none()
                                && p.set.is_none()
                                && p.get_expr.is_none()
                                && !p.is_static
                                && !p.is_abstract =>
                        {
                            if !field_slot_order.iter().any(|n| n == &p.name) {
                                field_slot_order.push(p.name.clone());
                            }
                        }
                        // An event has storage too (the subscriber-list
                        // head): it takes a slot like an auto-property.
                        crate::models::ClassMember::Event(e) => {
                            if !field_slot_order.iter().any(|n| n == &e.name) {
                                field_slot_order.push(e.name.clone());
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        for (src,) in &field_sources {
            for member in &src.decl.members {
                match member {
                    crate::models::ClassMember::Field(f) => {
                        // `const` fields are compile-time constants: no object
                        // layout slot. Their initializer is constant-folded now
                        // and every read is replaced by the folded literal.
                        if f.is_const {
                            if let Some(existing) = const_fields
                                .iter_mut()
                                .find(|fi| fi.name == f.name)
                            {
                                // Hidden base member: the derived
                                // declaration wins in place (see below).
                                existing.type_name = f.type_name.clone();
                                existing.access = f.access;
                                existing.const_value = const_values
                                    .get(&(src.key(), f.name.clone()))
                                    .cloned()
                                    .map(|v| convert_const(v, &f.type_name));
                                existing.declaring_class = src.decl.name.clone();
                                continue;
                            }
                            // Resolved by the fixpoint pass above (same
                            // class, bases, enums, strings, casts); absent
                            // means genuinely unresolvable (cycle/unknown).
                            // Convert to the declared type (C# implicit
                            // constant conversion): `double X = <long>`
                            // folds as Int and must become Float, else
                            // downstream arithmetic overflows or
                            // mistypes (`s_DUINT_BASE2 * s_DUINT_BASE`
                            // with Int 2**32 operands overflows
                            // checked_mul instead of computing 2**64).
                            let const_value = const_values
                                .get(&(src.key(), f.name.clone()))
                                .cloned()
                                .map(|v| convert_const(v, &f.type_name));
                            if const_value.is_none() {
                                errors.push(err_help(
                                    "E0030",
                                    format!(
                                        "const field '{}' in class '{}' must be initialized with a compile-time constant expression",
                                        f.name, decl.decl.name
                                    ),
                                    "use a literal or an expression of literals (e.g. `1 + 2`, `-\u{2212}5`, `3.14`)",
                                ));
                            }
                            const_fields.push(FieldInfo {
                                name: f.name.clone(),
                                type_name: f.type_name.clone(),
                                access: f.access,
                                offset_slots: 0,
                                is_static: false,
                                is_const: true,
                                const_value,
                                is_byref: false,
                                declaring_class: src.decl.name.clone(),
                            });
                            continue;
                        }
                        // `static` fields live in per-class static storage, not
                        // the object layout. First occurrence (base chain, then
                        // own) wins, mirroring the instance-field rule.
                        if f.is_static {
                            if let Some(existing) = static_fields
                                .iter_mut()
                                .find(|fi| fi.name == f.name)
                            {
                                // Hidden base member: derived wins in place.
                                existing.type_name = f.type_name.clone();
                                existing.access = f.access;
                                existing.declaring_class = src.decl.name.clone();
                                continue;
                            }
                            static_fields.push(FieldInfo {
                                name: f.name.clone(),
                                type_name: f.type_name.clone(),
                                access: f.access,
                                offset_slots: 0,
                                is_static: true,
                                is_const: false,
                                const_value: None,
                                is_byref: false,
                                declaring_class: src.decl.name.clone(),
                            });
                            continue;
                        }
                        if let Some(existing) =
                            fields.iter_mut().find(|fi| fi.name == f.name)
                        {
                            // Hidden base member: the derived declaration
                            // wins in place (keeps base-first layout order
                            // for inheritance compatibility, but resolves
                            // and access-checks as the derived member —
                            // e.g. `InvokeGenerator._fac` over
                            // `QilCloneVisitor._fac`).
                            existing.type_name = f.type_name.clone();
                            existing.access = f.access;
                            existing.declaring_class = src.decl.name.clone();
                            continue;
                        }
                        // No field-count cap: the x86-64 backend emits
                        // disp32 for offsets beyond disp8 range
                        // (`field_load`/`field_store` select ModRM 10 +
                        // imm32), so arbitrarily many fields address
                        // correctly. (An old disp8-only cap lived here;
                        // removed once disp32 fallback was verified.)
                        //
                        // Layout override: `String._stringLength` lives
                        // at offset 0, not in a field slot. Our string
                        // objects store the character count at +0
                        // (vtable-less, refcount at +8, data at +16),
                        // so every table-driven read of `_stringLength`
                        // (direct field reads, inlined `Length`
                        // getters) must use slot 0. Without this they
                        // read slot 2 (+16, first data word) as the
                        // length — e.g. `ToCharArray` on `"\n"` saw
                        // length 10 (0x0A) instead of 1 and overran.
                        let is_string_len =
                            builtin_string_interior_offset(&src.decl.name, &f.name) == Some(0);
                        let offset_slots = if is_string_len {
                            0
                        } else {
                            // Canonical root-first slot (see
                            // field_slot_order above); legacy
                            // positional numbering on miss (unreachable
                            // — same declarations feed both).
                            field_slot_order
                                .iter()
                                .position(|n| n == &f.name)
                                .map(|i| i + FIRST_FIELD_SLOT)
                                .unwrap_or(fields.len() + FIRST_FIELD_SLOT)
                        };
                        fields.push(FieldInfo {
                            name: f.name.clone(),
                            type_name: f.type_name.clone(),
                            access: f.access,
                            offset_slots,
                            is_static: false,
                            is_const: false,
                            const_value: None,
                            is_byref: f.is_byref,
                            declaring_class: src.decl.name.clone(),
                        });
                    }
                    // Properties are accessor-backed: explicit-body properties
                    // keep their own storage fields (e.g. `_value`), so no
                    // layout slot is added here. An AUTO-property (no explicit
                    // accessor bodies) gets a compiler-generated backing field
                    // under the property name, matching C# auto-property
                    // semantics, so `obj.X` / `obj.X = v` read/write it.
                    crate::models::ClassMember::Property(p) => {
                        // Record every property's declared type (own +
                        // inherited) for member-access typing
                        // (`dateTime.Ticks` is `long`). Hidden base
                        // members resolve as the derived declaration
                        // (see fields above). Storage (auto backing
                        // fields) is handled just below; explicit
                        // properties occupy no slot.
                        if let Some(existing) =
                            properties.iter_mut().find(|pi| pi.name == p.name)
                        {
                            existing.type_name = p.type_name.clone();
                            existing.access = p.access;
                            // declaring_class follows the nearest EXPLICIT
                            // provider (bodies inline from there): sources
                            // run base-first with own last, so
                            // unconditional overwrite lands on base-most
                            // (bodyless abstract), breaking
                            // declarer-relative resolution inside inlined
                            // bodies (private base fields read as
                            // undeclared, faulting `Stream.CanWrite`
                            // arms to zero). Only an explicit incoming
                            // body re-points the declarer.
                            if p.get.is_some()
                                || p.set.is_some()
                                || p.get_expr.is_some()
                            {
                                existing.declaring_class = src.decl.name.clone();
                            }
                            // Abstractness clears on concrete override
                            // (derived entries must not read abstract):
                            // only a bodyless-abstract incoming keeps it.
                            existing.is_abstract =
                                existing.is_abstract && p.is_abstract;
                            // Intrinsic is sticky: if any declaration
                            // carries it, the merged property is intrinsic.
                            existing.is_intrinsic =
                                existing.is_intrinsic
                                    || p.attrs.iter().any(|a| a.name == "Intrinsic");
                        } else {
                            properties.push(PropertyInfo {
                                name: p.name.clone(),
                                type_name: p.type_name.clone(),
                                access: p.access,
                                declaring_class: src.decl.name.clone(),
                                is_static: p.is_static,
                                is_abstract: p.is_abstract,
                                is_intrinsic: p
                                    .attrs
                                    .iter()
                                    .any(|a| a.name == "Intrinsic"),
                            });
                        }
                        if p.get.is_some() || p.set.is_some() || p.get_expr.is_some() {
                            continue;
                        }
                        // Abstract properties declare no storage (their
                        // body-less accessors are not auto-properties):
                        // synthesizing a backing field shifts every
                        // later instance offset.
                        if p.is_abstract {
                            continue;
                        }
                        if fields.iter().any(|fi| fi.name == p.name) {
                            continue;
                        }
                        // Static auto-properties (`static T X { get; …
                        // }`) back into STATIC storage (real slots,
                        // zero-initialized — initializers are ignored,
                        // matching instance auto-props whose inits never
                        // run either). Instance backing fields must NOT
                        // serve statics (reads fault off a null
                        // receiver); statics must NOT occupy object
                        // layout (they'd shift instance offsets).
                        if p.is_static {
                            if static_fields.iter().any(|fi| fi.name == p.name) {
                                continue;
                            }
                            static_fields.push(FieldInfo {
                                name: p.name.clone(),
                                type_name: p.type_name.clone(),
                                access: p.access,
                                offset_slots: 0,
                                is_static: true,
                                is_const: false,
                                const_value: None,
                                is_byref: false,
                                declaring_class: src.decl.name.clone(),
                            });
                            continue;
                        }
                        // No field-count cap (see instance-field case above:
                        // disp32 fallback covers large offsets).
                        // (Static auto-properties return above into
                        // static storage; only instance props reach here.)
                        // Canonical root-first slot (see
                        // field_slot_order above).
                        let offset_slots = field_slot_order
                            .iter()
                            .position(|n| n == &p.name)
                            .map(|i| i + FIRST_FIELD_SLOT)
                            .unwrap_or(fields.len() + FIRST_FIELD_SLOT);
                        fields.push(FieldInfo {
                            name: p.name.clone(),
                            type_name: p.type_name.clone(),
                            access: p.access,
                            offset_slots,
                            is_static: false,
                            is_const: false,
                            const_value: None,
                            // Auto-properties are never `ref` (ref
                            // properties carry bodies and return above).
                            is_byref: false,
                            declaring_class: src.decl.name.clone(),
                        });
                    }
                    // An EVENT needs storage: it holds the head of the
                    // subscriber list (`add` prepends, `remove` unlinks,
                    // invocation walks it). Indexers and nested types
                    // remain slot-less.
                    crate::models::ClassMember::Event(e) => {
                        let offset_slots = field_slot_order
                            .iter()
                            .position(|n| n == &e.name)
                            .map(|i| i + FIRST_FIELD_SLOT)
                            .unwrap_or(fields.len() + FIRST_FIELD_SLOT);
                        fields.push(crate::native::semantic::FieldInfo {
                            name: e.name.clone(),
                            type_name: "object".to_owned(),
                            access: e.access,
                            is_static: false,
                            is_const: false,
                            const_value: None,
                            is_byref: false,
                            offset_slots,
                            declaring_class: src.decl.name.clone(),
                        });
                    }
                    crate::models::ClassMember::Indexer(_)
                    | crate::models::ClassMember::Nested(_) => {}
                    crate::models::ClassMember::Method(m) => {
                        // Overloads (`Dispose()` vs `Dispose(bool)`) are
                        // distinct methods: dedup by (name, param types),
                        // not name alone (which dropped all but the first
                        // overload, breaking interface conformance).
                        // Dedup is scoped to the DECLARING class: identical
                        // signatures across base/derived are hides and
                        // overrides, not duplicates (dropping the derived
                        // copy breaks access and direct dispatch, e.g. a
                        // derived private `GetDefaultMessage` shadowed by
                        // the base's). Generic arity joins the key:
                        // `AddOptions()` vs `AddOptions<TOptions>(...)`
                        // share erased params but are distinct overloads
                        // (dropping the generic one breaks explicit-args
                        // calls like `services.AddOptions<NoOp>()`).
                        let param_types: Vec<String> = m
                            .params
                            .iter()
                            .map(|p| p.type_name.clone())
                            .collect();
                        // Byref-ness joins the key: value-vs-`ref` twins
                        // (`WriteAtOffset(h, buf, off)` vs `(…, ref off)`)
                        // share erased param types — dropping the second
                        // stranded every `ref`-marked call on the value
                        // twin (the forwarder electing itself, recursing
                        // forever on identical args).
                        let param_byref: Vec<bool> =
                            m.params.iter().map(|p| p.is_byref).collect();
                        if methods.iter().any(|mi| {
                            mi.name == m.name
                                && mi.param_types == param_types
                                && mi.param_is_byref == param_byref
                                && mi.declaring_class == src.decl.name
                                && mi.generic_params.len() == m.generic_params.len()
                        }) {
                            continue;
                        }
                        let slot = methods.len();
                        let dcx = ConstCx {
                            enums: &enum_values,
                            consts: &const_values,
                            chains: &chains_map,
                            keys: &all_keys,
                            current: src.key(),
                        };
                        let mut param_has_default = Vec::with_capacity(m.params.len());
                        let mut param_defaults = Vec::with_capacity(m.params.len());
                        for p in &m.params {
                            match &p.default_value {
                                None => {
                                    param_has_default.push(false);
                                    param_defaults.push(None);
                                }
                                Some(d) => {
                                    param_has_default.push(true);
                                    param_defaults.push(const_fold(d, &dcx));
                                }
                            }
                        }
                        methods.push(MethodInfo {
                            name: m.name.clone(),
                            ret_type: m.ret_type.clone(),
                            param_types: m.params.iter().map(|p| p.type_name.clone()).collect(),
                            param_names: m.params.iter().map(|p| p.name.clone()).collect(),
                            param_has_default,
                            // `params` is always last by language rules;
                            // any flagged param marks the method.
                            has_params: m.params.iter().any(|p| p.is_params),
                            param_is_byref: m.params.iter().map(|p| p.is_byref).collect(),
                            param_defaults,
                            access: m.access,
                            is_static: m.is_static,
                            has_body: m.body.is_some(),
                            declaring_class: src.decl.name.clone(),
                            is_extension: m.is_static
                                && m.params.first().is_some_and(|p| p.is_this),
                            generic_params: m.generic_params.clone(),
                            vtable_slot: slot,
                            // `[Conditional("SYM")]`: calls omit unless
                            // `SYM` is defined (C# release semantics).
                            conditional_symbol: m.attrs.iter().find_map(|a| {
                                // Qualified form
                                // (`System.Diagnostics.Conditional`)
                                // matches like the simple name.
                                if a.name.rsplit('.').next().unwrap_or(&a.name)
                                    != "Conditional"
                                {
                                    return None;
                                }
                                a.arguments.first().and_then(|arg| match &arg.value {
                                    crate::models::ast::attribute::AttributeValue::Str(
                                        s,
                                    ) => Some(s.clone()),
                                    _ => None,
                                })
                            }),
                        });
                    }
                }
            }
        }

        // Interfaces implemented directly or via any base, matched by
        // (simple name, arity): bare `ICollection` is arity 0 and never
        // generic `ICollection<T>`. Name-only matching falsely attributes
        // generic members to non-generic implementers (and vice versa).
        fn iface_key(name: &str) -> (String, usize) {
            let simple = name.rsplit('.').next().unwrap_or(name);
            match simple.find('<') {
                None => (simple.to_owned(), 0),
                Some(i) => {
                    let args = &simple[i + 1..];
                    let arity = if args.trim().is_empty() {
                        0
                    } else {
                        args.split(',').count()
                    };
                    (simple[..i].trim_end().to_owned(), arity)
                }
            }
        }
        // (name, arity) pairs: arity distinguishes `ICollection`
        // from `ICollection<T>` (name-only lists conflate them).
        let mut implemented: Vec<(String, usize)> = Vec::new();
        // Own key + chain keys (qualified, arity-aware) drive matching.
        let own_key = decl.key();
        // Map each relevant key to its raw base strings (for textual
        // `IFoo<T>` forms written in bases lists).
        for key in std::iter::once(&own_key).chain(res.chain.iter()) {
            // Direct: this key IS an interface (class implementing an
            // interface with the same simple name is nonsense, but the
            // (name, arity) pair may still match an interface entry —
            // e.g. colliding declarations; check anyway, harmless).
            // Primary: raw base strings of the corresponding declaration.
            let bases: Vec<String> = raw_classes
                .iter()
                .find(|cc| &cc.key() == key)
                .map(|cc| cc.decl.bases.clone())
                .unwrap_or_default();
            // The key itself, if it names an interface.
            let (kn, ka) = (key.simple_name().to_owned(), key.arity);
            for iface in &table.interfaces {
                if iface.name == kn
                    && iface.arity == ka
                    && !implemented.contains(&(iface.name.clone(), iface.arity))
                {
                    implemented.push((iface.name.clone(), iface.arity));
                }
                for b in &bases {
                    let (bn, ba) = iface_key(b);
                    if bn == iface.name
                        && ba == iface.arity
                        && !implemented.contains(&(iface.name.clone(), iface.arity))
                    {
                        implemented.push((iface.name.clone(), iface.arity));
                    }
                }
            }
        }
        // Transitive closure over interface bases (`IList` → `ICollection`
        // → `IEnumerable`): `resolve_chain` tracks classes only, so
        // interface-to-interface hops never enter `res.chain`. Without
        // this, `implemented` holds only direct interfaces and
        // class-to-interface conversions (`Dictionary` → `IEnumerable`
        // extension receivers) miss. Cycle-safe: only new pairs enqueue.
        {
            let mut queue: Vec<(String, usize)> = implemented.clone();
            while let Some((n, a)) = queue.pop() {
                if let Some(decl) = table
                    .interfaces
                    .iter()
                    .find(|i| i.name == n && i.arity == a)
                {
                    for b in &decl.bases {
                        let (bn, ba) = iface_key(b);
                        if !implemented.contains(&(bn.clone(), ba))
                            && table
                                .interfaces
                                .iter()
                                .any(|i| i.name == bn && i.arity == ba)
                        {
                            implemented.push((bn.clone(), ba));
                            queue.push((bn, ba));
                        }
                    }
                }
            }
        }

        let own_members: Vec<(String, bool, Access)> = decl
            .decl
            .members
            .iter()
            .map(|m| match m {
                crate::models::ClassMember::Field(f) => (f.name.clone(), true, f.access),
                // Const fields still count as own members but are not storage.
                crate::models::ClassMember::Property(p) => (p.name.clone(), true, p.access),
                // Indexers and events are accessor-backed; listed as own members
                // but not storage.
                crate::models::ClassMember::Indexer(idx) => (idx.params.first().map(|p| p.name.clone()).unwrap_or_default(), true, idx.access),
                crate::models::ClassMember::Event(evt) => (evt.name.clone(), true, evt.access),
                crate::models::ClassMember::Nested(_) => (String::new(), false, crate::models::Access::Default),
                crate::models::ClassMember::Method(m) => (m.name.clone(), false, m.access),
            })
            .collect();

        table.classes.push(ClassInfo {
            // Simple name stays for display/compat; `key` carries the
            // qualified identity. `bases` stores simple names (resolved
            // chains collapsed) so downstream string lookups behave as
            // before, now computed from correct chains.
            name: decl.decl.name.clone(),
            key: decl.key(),
            bases: res
                .chain
                .iter()
                .map(|k| k.simple_name().to_owned())
                .collect(),
            fields,
            static_fields,
            const_fields,
            methods: {
                methods
            },
            own_members,
            interfaces: implemented,
            properties,
            is_enum: false,
            generic_params: decl.decl.generic_params.clone(),
            type_arg_index: own_type_arg_index(&res.chain, &raw_classes, &decl.decl),
            is_struct: decl.decl.is_struct,
            is_record: decl.decl.is_record,
            // Ownership is attributed by the post-pass below (needs the
            // complete table for parent-chain inheritance).
            owner: String::new(),
        });
    }

    // ---- pass 3vtable: vtable slot reconciliation.
    // Per-entry positional slots misalign across inheritance: a base
    // method sits at different positions in base vs derived entries
    // (e.g. `Encoding.SetDefaultFallbacks` at 35 vs
    // `Latin1Encoding`'s own method at 35), so cross-class virtual
    // dispatch lands in unrelated bodies. Reassign slots base-first:
    // each (name, signature) keeps its base entry's slot (overrides
    // share it — our dispatch is always virtual, so hides behave as
    // overrides consistently); new methods take fresh slots after
    // the inherited max. `methods[]` ORDER is untouched (only slot
    // VALUES change), so overload resolution and selection are
    // unaffected. Processed base-first via memoized recursion
    // (cycles keep positional slots as fallback).
    {
        fn sig_of(m: &MethodInfo) -> (String, String, usize) {
            (
                m.name.clone(),
                m.param_types.join(","),
                m.generic_params.len(),
            )
        }
        /// EVERY class base, in declaration order — not just the first.
        ///
        /// C# has single inheritance, so "the base" was unambiguous; with
        /// concrete multiple inheritance a type has several, and each one
        /// contributes a member set and a slot range. Resolving only the
        /// first left the other bases' methods with no slots at all, so a
        /// call to them landed on whichever method held the same slot in
        /// the first base (`MyClass3.Multiply(10,100)` ran `Divide` and
        /// answered 0).
        ///
        /// Same-named entries are preferred by namespace, as before; a
        /// base whose name resolves to nothing (an interface) is skipped.
        fn base_indices(table: &TypeTable, cls: &ClassInfo) -> Vec<usize> {
            let ns = cls
                .key
                .path
                .rsplit_once('.')
                .map(|(n, _)| n)
                .unwrap_or("");
            let mut out: Vec<usize> = Vec::new();
            for b in &cls.bases {
                let mut pick: Option<usize> = None;
                let mut fallback: Option<usize> = None;
                for (i, c) in table.classes.iter().enumerate() {
                    if c.key.simple_name() != *b && c.name != *b {
                        continue;
                    }
                    if c.key.path.rsplit_once('.').map(|(n, _)| n).unwrap_or("") == ns {
                        pick = Some(i);
                        break;
                    }
                    fallback.get_or_insert(i);
                }
                if let Some(i) = pick.or(fallback) {
                    if !out.contains(&i) {
                        out.push(i);
                    }
                }
            }
            out
        }

        fn assign(
            table: &mut TypeTable,
            idx: usize,
            stack: &mut Vec<String>,
        ) {
            let key = table.classes[idx].key.path.clone();
            if stack.contains(&key) {
                return; // cycle: keep positional slots
            }
            stack.push(key.clone());
            // Every base is finalized first, then their slot maps are
            // UNIONED in declaration order: a signature two bases share
            // keeps the first base's slot (decision B — declaration
            // order decides), and distinct signatures from the second
            // base take their own slots after the first base's range.
            let bases = base_indices(table, &table.classes[idx].clone());
            for bi in &bases {
                assign(table, *bi, stack);
            }
            let mut inherited: std::collections::BTreeMap<
                (String, String, usize),
                usize,
            > = Default::default();
            for bi in &bases {
                for m in table.classes[*bi].methods.iter() {
                    inherited.entry(sig_of(m)).or_insert(m.vtable_slot);
                }
            }
            let mut next =
                inherited.values().copied().max().map(|m| m + 1).unwrap_or(0);
            stack.pop();
        }
        // NOTE: `table.classes` grows below (enums); reconcile only
        // the classes built so far (enums clone Object's list verbatim
        // and stay consistent automatically).
        let n = table.classes.len();
        for idx in 0..n {
            assign(&mut table, idx, &mut Vec::new());
        }
    }

    // ---- pass 3b: register enums as const-field types.
    // Each variant becomes a compile-time constant field, so `Color.Green`
    // resolves to the folded literal (auto-assigned or explicit value),
    // matching C# enum semantics. Values come from the precomputed map
    // shared with const-folding above (single numbering implementation).
    // Enums additionally inherit `System.Object` members (`ToString`,
    // `GetHashCode`, ...) and base link: every enum value is an object
    // (`ParseFailureKind.X.ToString()`), and overload applicability
    // treats enums as convertible to `object`.
    let object_methods: Vec<MethodInfo> = table
        .classes
        .iter()
        .find(|c| c.name == "Object")
        .map(|c| c.methods.clone())
        .unwrap_or_default();
    for (path, e) in &raw_enums {
        let mut const_fields: Vec<FieldInfo> = Vec::new();
        for (name, _) in &e.variants {
            let value = enum_values
                .get(&(e.name.clone(), name.clone()))
                .copied()
                .unwrap_or_default();
            const_fields.push(FieldInfo {
                name: name.clone(),
                type_name: e.underlying_type.clone(),
                access: Access::Public,
                offset_slots: 0,
                is_static: false,
                is_const: true,
                const_value: Some(ConstValue::Int(value)),
                is_byref: false,
                declaring_class: e.name.clone(),
            });
        }
        table.classes.push(ClassInfo {
            name: e.name.clone(),
            // Namespace-qualified keys like every other type, so
            // qualified spellings (`System.Threading.StackCrawlMark`)
            // hit their entries instead of falling to object.
            key: crate::native::semantic::table::TypeKey::new(path.clone(), 0),
            bases: if object_methods.is_empty() {
                Vec::new()
            } else {
                vec!["Object".to_owned()]
            },
            fields: Vec::new(),
            static_fields: Vec::new(),
            const_fields,
            methods: object_methods.clone(),
            own_members: Vec::new(),
            interfaces: Vec::new(),
            properties: Vec::new(),
            is_enum: true,
            generic_params: Vec::new(),
            type_arg_index: Vec::new(),
            // Enums lower as const-field holders; constructor rules
            // never apply to them.
            is_struct: false, is_record: false,
            // Ownership is attributed by the post-pass below (needs the
            // complete table for parent-chain inheritance).
            owner: String::new(),
        });
    }

    // ---- pass 4: interface conformance.
    // Look up by qualified key (not bare name) so colliding types
    // check against their OWN table entry.
    for decl in &raw_classes {
        let key = decl.key();
        let Some(info) = table.class_key(&key) else {
            continue;
        };
        for (iface_name, iface_arity) in &info.interfaces {
            let Some(iface) = table
                .interfaces
                .iter()
                .find(|i| &i.name == iface_name && i.arity == *iface_arity)
            else {
                continue;
            };
            // Generic interfaces use type parameters (e.g. `T`) in their method
            // signatures that are substituted with concrete types in the
            // implementation. The backend does not perform generic type
            // substitution, so param-type comparison is only meaningful for
            // non-generic interfaces.
            let check_params = iface.arity == 0;
            for sig in &iface.methods {
                // Explicit implementations (`void IFoo.Bar()`) satisfy
                // `Bar`: match bare names and `Iface.Method` forms
                // (qualified or simple interface prefix). Overloads
                // (`ToString()` vs `ToString(IFormatProvider?)`) match
                // by full normalized signature — arity alone picks the
                // wrong overload (`Enum.ToUInt64(object)` for sig
                // `ToUInt64(IFormatProvider?)`).
                // Nullable annotations (`IFormatProvider?`) and
                // qualification (`RuntimeType.X` vs `X`) have no runtime
                // representation — normalize before comparing
                // (mirroring `ref`/`out` erasure in parameter parsing).
                fn norm_ty(t: &str) -> String {
                    let t = t.trim_end_matches('?').trim_end();
                    let t = t.strip_prefix("global::").unwrap_or(t);
                    match t.rfind('.') {
                        Some(i) => t[i + 1..].to_owned(),
                        None => t.to_owned(),
                    }
                }
                let sig_norm: Vec<String> = sig
                    .params
                    .iter()
                    .map(|p| norm_ty(&p.type_name))
                    .collect();
                let mut names = vec![sig.name.clone()];
                let qualified = format!("{}.{}", iface.name, sig.name);
                let simple_qualified = format!(
                    "{}.{}",
                    iface.name.rsplit('.').next().unwrap_or(&iface.name),
                    sig.name
                );
                if simple_qualified != qualified {
                    names.push(simple_qualified);
                }
                names.push(qualified);
                // Strip generic args for comparison (`ICollection<T>.Remove`
                // explicit impls match `ICollection.Remove`; arity already
                // matched above so `<T>` carries no signal here).
                fn strip_generics(s: &str) -> String {
                    let mut out = String::with_capacity(s.len());
                    let mut depth = 0i32;
                    for ch in s.chars() {
                        match ch {
                            '<' => depth += 1,
                            '>' => depth -= 1,
                            _ => {
                                if depth == 0 {
                                    out.push(ch);
                                }
                            }
                        }
                    }
                    out
                }
                // A bare name that resolves to no known type is an open
                // generic variable from an enclosing scope (see above):
                // strip array/nullable/pointer suffixes, then require
                // absence from classes, interfaces, and primitives.
                // Using-aliases do NOT count (`using T = ...` is
                // file-scoped; letting it make `T` concrete globally
                // breaks overload conformance elsewhere). Direct table
                // scan, bypassing the alias-resolving
                // `class()`/`interface()` lookups.
                fn is_open_type_param(table: &TypeTable, t: &str) -> bool {
                    let mut b = t.trim();
                    loop {
                        if let Some(s) = b.strip_suffix("[]") {
                            b = s;
                        } else if let Some(s) = b.strip_suffix(['?', '*']) {
                            b = s;
                        } else {
                            break;
                        }
                    }
                    if b.is_empty() {
                        return false;
                    }
                    const PRIMS: &[&str] = &[
                        "bool", "byte", "sbyte", "char", "decimal", "double",
                        "float", "int", "uint", "nint", "nuint", "long",
                        "ulong", "short", "ushort", "object", "string",
                        "void", "dynamic", "Boolean", "Byte", "SByte", "Char",
                        "Decimal", "Double", "Single", "Int32", "UInt32",
                        "IntPtr", "UIntPtr", "Int64", "UInt64", "Int16",
                        "UInt16", "Object", "String",
                    ];
                    if PRIMS.contains(&b) {
                        return false;
                    }
                    table.classes.iter().all(|c| c.name != b)
                        && table.interfaces.iter().all(|i| i.name != b)
                }
                // Exact (name, normalized types) match first. A parameter
                // that names no known type is a generic variable from an
                // enclosing scope (`T` in `FindItemIndex(T item)` on an
                // interface nested in `FrozenSetInternalBase<T, ...>`,
                // bound to `string` by the implementation): it matches
                // any concrete type. Genuine mismatches (both sides
                // known, differing) still fall through to E0016 below.
                // Explicit implementations may qualify through an outer
                // type (`Outer.IComImportAdapter.GetRuntimeCallableWrapper`):
                // match those by dot-boundary suffix too.
                let provided = names.iter().find_map(|n| {
                    let suffixed = format!(".{n}");
                    info.methods.iter().find(|m| {
                        (m.name == *n
                            || strip_generics(&m.name) == *n
                            || m.name.ends_with(&suffixed))
                            && m.param_types.len() == sig_norm.len()
                            && m.param_types
                                .iter()
                                .map(|p| norm_ty(p))
                                .zip(sig_norm.iter())
                                .all(|(cn, sn)| {
                                    cn == *sn || is_open_type_param(&table, sn)
                                })
                    })
                });
                // Fallback: same name+arity (for E0016 reporting when
                // types genuinely differ).
                let arity_match = names.iter().find_map(|n| {
                    let suffixed = format!(".{n}");
                    info.methods.iter().find(|m| {
                        (m.name == *n
                            || strip_generics(&m.name) == *n
                            || m.name.ends_with(&suffixed))
                            && m.param_types.len() == sig_norm.len()
                    })
                });
                match (provided, arity_match) {
                    (Some(_), _) => {}
                    (None, Some(_)) => {
                        if check_params {
                            let mut ctx = decl.outer.clone();
                            ctx.push(decl.decl.name.clone());
                            errors.push(err(
                                "E0016",
                                format!(
                                    "'{}' signature mismatch for interface '{}.{}' [in {}]",
                                    sig.name,
                                    iface_name,
                                    sig.name,
                                    ctx.join(".")
                                ),
                            ));
                        }
                    }
                    (None, None) => {
                        // Default interface implementations (DIMs)
                        // inherited through the hierarchy satisfy too: if
                        // any transitively-implemented interface that
                        // extends (or is) the requiring one provides a
                        // body for this member, no error
                        // (`IArrayElementMarshaler` defaults
                        // `IArrayMarshaler.AllocateSpaceForManaged` for
                        // all its implementers). Mirrors the `sig.body`
                        // exemption for directly-defaulted members.
                        let satisfied_by_dim = info.interfaces.iter().any(|(pname, parity)| {
                            table
                                .interfaces
                                .iter()
                                .filter(|i| &i.name == pname && i.arity == *parity)
                                .any(|idecl| {
                                    table.interface_derives_from(&idecl.name, iface_name)
                                        && idecl.methods.iter().any(|m| {
                                            (m.name == sig.name
                                                || strip_generics(&m.name) == sig.name
                                                || m.name.ends_with(&format!(".{}", sig.name)))
                                                && m.params.len() == sig.params.len()
                                                && m.body.is_some()
                                        })
                                })
                        });
                        if sig.body.is_none() && !satisfied_by_dim {
                            errors.push(err(
                                "E0015",
                                format!(
                                    "class '{}' does not implement '{}.{}' required by interface",
                                    decl.decl.name, iface_name, sig.name
                                ),
                            ));
                        }
                    }
                }
            }
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    // ---- ownership post-pass (dynamic-linking groundwork; no-op when
    // `owners` is empty). Needs the complete table: nested entries
    // inherit through their parent chain, and parents are only all
    // present now. Paths strictly shorten upward, so no cycles.
    if !owners.is_empty() {
        fn owner_of(
            classes: &[crate::native::semantic::table::ClassInfo],
            owners: &std::collections::HashMap<
                crate::native::semantic::table::TypeKey,
                String,
            >,
            key: &crate::native::semantic::table::TypeKey,
            memo: &mut std::collections::HashMap<
                crate::native::semantic::table::TypeKey,
                String,
            >,
        ) -> String {
            if let Some(o) = memo.get(key) {
                return o.clone();
            }
            if let Some(o) = owners.get(key) {
                memo.insert(key.clone(), o.clone());
                return o.clone();
            }
            let o = key
                .path
                .rfind('.')
                .and_then(|i| {
                    let parent = &key.path[..i];
                    classes
                        .iter()
                        .find(|c| c.key.path == *parent)
                        .map(|p| owner_of(classes, owners, &p.key, memo))
                })
                .unwrap_or_default();
            memo.insert(key.clone(), o.clone());
            o
        }
        let keys: Vec<crate::native::semantic::table::TypeKey> =
            table.classes.iter().map(|c| c.key.clone()).collect();
        let mut memo = std::collections::HashMap::new();
        let resolved: Vec<String> =
            keys.iter().map(|k| owner_of(&table.classes, owners, k, &mut memo)).collect();
        for (c, o) in table.classes.iter_mut().zip(resolved) {
            // Preloaded entries keep their recorded owner (from `.rci`);
            // only fill entries the post-pass newly attributes.
            if c.owner.is_empty() {
                c.owner = o;
            }
        }
    }

    Ok(table)
}

use std::collections::{BTreeMap, BTreeSet};

// Re-exported so callers can build scope context without knowing models.
pub fn context_of(unit: &CompilationUnit, class_name: &str, function: &str) -> ScopeContext {
    let ns = unit.namespace_name_of(class_name);
    ScopeContext {
        namespace: ns,
        class: Some(class_name.to_owned()),
        function: Some(function.to_owned()),
    }
}

/// Offset of a `string` interior member in RubyC's builtin string
/// representation, or `None` when the member is an ordinary field.
///
/// RubyC owns the layout of its builtin `string`:
/// `+0` character count · `+8` reference count · `+16` UTF-16 payload.
/// A `string` implementation must be built to that representation, so
/// the compiler — not any particular library — decides these offsets.
///
/// The member spellings below are the ones `System.String` uses
/// (`_stringLength`, `_firstChar`). That is a compatibility shim, not a
/// language rule: it is why this helper exists in exactly one place
/// instead of being open-coded per stage. The fix is a language-level
/// contract (lowering `Length` and the indexer off the builtin type) so
/// no library's private field names are known here.
pub fn builtin_string_interior_offset(declaring_type: &str, member: &str) -> Option<u32> {
    if declaring_type
        .rsplit('.')
        .next()
        .unwrap_or(declaring_type)
        != "String"
    {
        return None;
    }
    match member {
        "_stringLength" => Some(0),
        "_firstChar" => Some(16),
        _ => None,
    }
}

/// Convert a semantic error into a diagnostic stamped with class context.
pub fn to_diagnostic(e: &SemanticError, ctx: &ScopeContext) -> Diagnostic {
    let d = Diagnostic::new(e.code, Severity::Error, e.message.clone()).in_context(ctx.clone());
    match &e.help {
        Some(h) => d.with_help(h.clone()),
        None => d,
    }
}

// ---- access checking -------------------------------------------------------

/// C# visibility rules. `Default` (no modifier on a class member) behaves
/// like `private`, matching C#'s default member accessibility.
pub fn visible_from(table: &TypeTable, access: Access, declaring: &str, viewer: &str) -> bool {
    match access {
        Access::Public => true,
        // `internal` is assembly-visible: one library build is one
        // assembly, so everything in the unit sees it (cross-assembly
        // `internal` enforcement arrives with cross-library references).
        Access::Internal => true,
        // Private (and default): same class, or the same nesting
        // family — C# nested types access the enclosing type's
        // privates and vice versa (`SyncStream`, nested in `Stream`,
        // calls `Stream`'s private helpers).
        Access::Private | Access::Default => {
            same_type_or_nesting(table, declaring, viewer)
        }
        // Protected: same class, derived classes, or the same nesting
        // family (nested types access enclosing protecteds, e.g.
        // `CompatDerivedImpl` calling `Random.Sample` — upstream
        // compiles it, so the language allows it). Also nested-in-
        // derived: a nested class shares its enclosing types'
        // protected access (`Http2Stream` nested in `Http2Connection
        // : HttpConnectionBase` reads `_pool`).
        Access::Protected => {
            same_type_or_nesting(table, declaring, viewer)
                || table.derives_from(viewer, declaring)
                || nested_in_derived(table, declaring, viewer)
        }
    }
}

/// True when `viewer` is nested (transitively) inside a class that
/// derives from `declaring` (or is it): nested classes share their
/// enclosing types' protected rights. Walks each viewer entry's
/// qualified key outward; any enclosing class entry satisfying the
/// check grants access.
fn nested_in_derived(table: &TypeTable, declaring: &str, viewer: &str) -> bool {
    let mut outers: Vec<String> = Vec::new();
    for c in table
        .classes
        .iter()
        .filter(|c| c.name == viewer || c.key.path == viewer)
    {
        let mut prefix = c.key.path.clone();
        while let Some(i) = prefix.rfind('.') {
            prefix.truncate(i);
            if !outers.contains(&prefix) {
                outers.push(prefix.clone());
            }
        }
    }
    for outer in &outers {
        let Some(info) = table.classes.iter().find(|o| o.key.path == *outer) else {
            continue;
        };
        if declaring == info.name
            || declaring == info.key.path
            || table.derives_from(&info.name, declaring)
        {
            return true;
        }
    }
    false
}

/// True when `a` and `b` name types in one nesting family: the same
/// entry (different spellings — qualified `System.IO.Stream.ReadWriteTask`
/// vs simple `ReadWriteTask`), or one qualified path extending the
/// other (`t.Stream` vs `t.Stream.SyncStream`). Checks ALL same-named
/// entries: simple names collide across families (`HashSet`/`Dictionary`/
/// `ConcurrentDictionary` each nest an `AlternateLookup`), and
/// first-match alone picks the wrong family. Any in-family pair
/// allows (a truly unrelated viewer with the same simple name is
/// C#-illegal input anyway; qualified identity is future work).
fn same_nesting_family(table: &TypeTable, a: &str, b: &str) -> bool {
    let (pa, pb) = (class_paths(table, a), class_paths(table, b));
    pa.iter().any(|x| {
        pb.iter().any(|y| {
            x == y || x.starts_with(&format!("{y}.")) || y.starts_with(&format!("{x}."))
        })
    })
}

/// The table entries a class name may denote, as their qualified
/// paths. A BARE name that matches several types identifies none of
/// them: keeping every candidate let `Inner` claim the nesting family
/// of `A.Outer.Inner` because one candidate happened to be that path —
/// so an unrelated `Inner` inherited access it never had. Ambiguity is
/// not evidence; a dotted spelling is what disambiguates.
fn class_paths(table: &TypeTable, n: &str) -> Vec<String> {
    let hits: Vec<String> = table
        .classes
        .iter()
        .filter(|c| {
            c.name == n
                || c.key.path == n
                // Namespace-relative spellings
                // (`WaitSubsystem.ThreadWaitInfo` as-written inside
                // `System.Threading`): match by dot-boundary suffix.
                // That is C# name lookup, which searches enclosing
                // namespaces outward.
                || c.key.path.ends_with(&format!(".{n}"))
        })
        .map(|c| c.key.path.clone())
        .collect();
    if !n.contains('.') && hits.len() > 1 {
        return Vec::new();
    }
    hits
}

/// The last segment of a qualified path (`Ns.Foo` → `Foo`).
fn simple_name(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or(name)
}

/// The declaring class and the viewer denote the SAME type, or one
/// nests the other: C# lets a nested type read the enclosing type's
/// private members (`Dictionary.AlternateLookup` reading
/// `Dictionary.GetBucket`). The viewer's qualified path identifies it
/// exactly; when a name is too ambiguous to identify a type, the
/// simple names are compared — the answer the old name comparison
/// gave.
fn same_type_or_nesting(table: &TypeTable, declaring: &str, viewer: &str) -> bool {
    if declaring == viewer {
        return true;
    }
    let (d, v) = (class_paths(table, declaring), class_paths(table, viewer));
    if d.is_empty() || v.is_empty() {
        return simple_name(declaring) == simple_name(viewer);
    }
    d.iter().any(|x| {
        v.iter().any(|y| {
            x == y || x.starts_with(&format!("{y}.")) || y.starts_with(&format!("{x}."))
        })
    })
}

/// Arity of a type spelling: number of top-level generic arguments
/// (`WeakReference<T>` → 1, `Dictionary<K,V>` → 2, `WeakReference` → 0).
/// Used to disambiguate same-named entries by arity (see `class_arity`).
pub(crate) fn spelling_arity(spelling: &str) -> usize {
    let Some(start) = spelling.find('<') else {
        return 0;
    };
    let bytes = spelling.as_bytes();
    let mut depth = 0usize;
    let mut end = None;
    let mut i = start;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => depth += 1,
            b'>' => {
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
        return 0;
    };
    let inner = &spelling[start + 1..end];
    if inner.trim().is_empty() {
        return 0;
    }
    let mut count = 1usize;
    depth = 0;
    for ch in inner.chars() {
        match ch {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => count += 1,
            _ => {}
        }
    }
    count
}

impl TypeTable {
    fn declaring_class_of(
        &self,
        start: &str,
        name: &str,
        is_field: bool,
    ) -> Option<(String, Access)> {
        // Arity-aware entry selection: `WeakReference` (arity 0, public
        // `Target`) vs `WeakReference<T>` (arity 1, private `Target`) —
        // first-match by simple name elects the wrong entry and falsely
        // denies access (E0017 in TypeDescriptor.CreateAssociation).
        // A spelling with N top-level type args prefers arity N;
        // arg-less spellings prefer arity 0.
        let wanted = spelling_arity(start);
        let mut queue: Vec<String> = vec![start.to_owned()];
        let mut seen: Vec<String> = Vec::new();
        while let Some(class_name) = queue.pop() {
            if seen.contains(&class_name) {
                continue;
            }
            seen.push(class_name.clone());
            if let Some(info) = self.class_arity(&class_name, wanted) {
                if let Some((_, _, access)) = info
                    .own_members
                    .iter()
                    .find(|(n, f, _)| n == name && *f == is_field)
                {
                    return Some((class_name, *access));
                }
                queue.extend(info.bases.iter().cloned());
            }
        }
        None
    }

    /// Walk `start`'s base chain (start first) and report the first class
    /// that itself declares the field: its name plus the field's access.
    pub fn declaring_class_of_field(&self, start: &str, name: &str) -> Option<(String, Access)> {
        self.declaring_class_of(start, name, true)
    }

    /// Same walk for methods: the first class whose OWN declaration list
    /// contains the name (its `Function` lives there too).
    pub fn declaring_class_of_method(&self, start: &str, name: &str) -> Option<(String, Access)> {
        self.declaring_class_of(start, name, false)
    }

    /// The declaring class of `name`, walking a PRECISE owner's base
    /// chain. Starting from a bare type spelling resolves the owner by
    /// name first-match, so a same-named sibling could answer for it
    /// (25 `Enumerator`s share a name); starting from the entry itself
    /// cannot.
    pub fn declaring_class_of_in(
        &self,
        owner: &ClassInfo,
        name: &str,
        is_field: bool,
    ) -> Option<(String, Access)> {
        let mut queue = vec![owner.key.path.clone()];
        let mut seen: Vec<String> = Vec::new();
        while let Some(cur) = queue.pop() {
            if seen.contains(&cur) {
                continue;
            }
            seen.push(cur.clone());
            let Some(info) = self.class(&cur) else {
                continue;
            };
            if let Some((_, _, access)) = info
                .own_members
                .iter()
                .find(|(n, f, _)| n == name && *f == is_field)
            {
                return Some((info.key.path.clone(), *access));
            }
            queue.extend(info.bases.iter().cloned());
        }
        None
    }
}

// Keep MethodDecl import used (signatures reference it through models).
#[allow(unused)]
fn _witness(_: &MethodDecl) {}

// ---- interface dispatch slots -------------------------------------------
//
// A virtual call carries one row index and the runtime reads that index
// from the RECEIVER's row. A class's own slots are numbered base-first
// for that class, so the same method sits at different indices in two
// unrelated implementers; an interface call cannot use a class slot.
// Interface members therefore get their own range ABOVE every class
// slot, and every row is padded to that range. Both the row builder
// (`native::ir`) and call lowering (`canonical_slot`) derive the range
// from this one function, so they cannot drift.

/// Normalized parameter spelling for slot keys: case, qualifiers,
/// nullable/ref markers, and whitespace do not change the signature.
pub fn normalize_param_type(s: &str) -> String {
    let mut t = s.to_lowercase();
    for w in ["out ", "ref ", "in ", "params "] {
        if let Some(rest) = t.strip_prefix(w) {
            t = rest.to_string();
            break;
        }
    }
    t.retain(|c| !c.is_whitespace());
    t.trim_end_matches('?').trim_end_matches('&').to_string()
}

/// First slot of the interface range: one past the highest class slot.
pub fn iface_slot_base(table: &TypeTable) -> usize {
    table
        .classes
        .iter()
        .flat_map(|c| c.methods.iter().map(|m| m.vtable_slot))
        .max()
        .map(|m| m + 1)
        .unwrap_or(0)
}

/// Every interface member with its stable slot, sorted by key so the
/// numbering depends only on the declarations, not on table order.
pub fn iface_slot_entries(table: &TypeTable) -> Vec<((String, String, String), usize)> {
    let mut keys: Vec<(String, String, String)> = Vec::new();
    for i in &table.interfaces {
        for m in &i.methods {
            let sig = m
                .params
                .iter()
                .map(|p| normalize_param_type(&p.type_name))
                .collect::<Vec<_>>()
                .join(",");
            let key = (i.name.clone(), m.name.clone(), sig);
            if !keys.contains(&key) {
                keys.push(key);
            }
        }
    }
    keys.sort();
    let base = iface_slot_base(table);
    keys.into_iter()
        .enumerate()
        .map(|(i, k)| (k, base + i))
        .collect()
}

/// Interface names reachable from `iface`, itself first, with inherited
/// bases after it (generic arguments and qualification stripped, like
/// member lookup elsewhere).
fn iface_and_bases(table: &TypeTable, iface: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut queue: Vec<String> = vec![iface.to_owned()];
    while let Some(name) = queue.pop() {
        if out.contains(&name) {
            continue;
        }
        out.push(name.clone());
        for d in table.interfaces.iter().filter(|d| d.name == name) {
            for b in &d.bases {
                let mut base = b.as_str();
                if let Some(k) = base.find('<') {
                    base = &base[..k];
                }
                let simple = base.rsplit('.').next().unwrap_or(base).trim();
                // A base declared only in a dependency is not in the
                // table; its members are not compiled here either.
                queue.push(simple.to_owned());
            }
        }
    }
    out
}

/// Slot for one interface member, declared on the interface or on one
/// of its bases (`IAsyncEnumerator.DisposeAsync`-style inheritance).
pub fn iface_slot_for(
    table: &TypeTable,
    iface: &str,
    method: &str,
    param_sig: &str,
) -> Option<usize> {
    let entries = iface_slot_entries(table);
    let names = iface_and_bases(table, iface);
    for name in &names {
        if let Some((_, slot)) = entries
            .iter()
            .find(|((i, m, s), _)| i == name && m == method && s == param_sig)
        {
            return Some(*slot);
        }
    }
    None
}

/// Slot for the only member named `method` in `iface` or its bases
/// (used when the call-site signature cannot match a generic spelling).
pub fn iface_slot_by_name(table: &TypeTable, iface: &str, method: &str) -> Option<usize> {
    let entries = iface_slot_entries(table);
    for name in iface_and_bases(table, iface) {
        let hits: Vec<usize> = entries
            .iter()
            .filter(|((i, m, _), _)| i == &name && m == method)
            .map(|(_, slot)| *slot)
            .collect();
        match hits.as_slice() {
            [one] => return Some(*one),
            _ => {}
        }
    }
    None
}
