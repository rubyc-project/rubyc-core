//! The type tables produced by semantic analysis: everything codegen and
//! lowering need to know about classes, fields, methods, imports, and
//! inheritance — plus the query helpers over them.

use crate::models::{Access, DelegateDecl, InterfaceDecl};

/// Identity of a type declaration: qualified dotted path (namespace +
/// outer classes + simple name, e.g. `System.Threading.Tasks.Task` or
/// `Ns.Outer.Inner`) plus generic arity (`Task` vs `Task<TResult>`).
/// C# allows same-named types to coexist when they differ in namespace,
/// nesting, or arity; the flat simple-name model collides them (false
/// E0013 duplicates, false E0014 cycles). Keys keep them distinct;
/// bare references resolve with same-scope preference (see `resolve`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TypeKey {
    /// Dotted path: `Ns.Outer.Inner` or bare `Name` for globals.
    pub path: String,
    /// Generic arity (`generic_params.len()`); 0 for non-generic.
    pub arity: usize,
}

impl TypeKey {
    pub fn new(path: impl Into<String>, arity: usize) -> Self {
        TypeKey {
            path: path.into(),
            arity,
        }
    }

    /// Simple (last-segment) name: `Ns.Foo` → `Foo`.
    pub fn simple_name(&self) -> &str {
        self.path.rsplit('.').next().unwrap_or(&self.path)
    }

    /// Enclosing namespace-ish prefix (`Ns.Outer` for `Ns.Outer.Inner`,
    /// `""` for top-level): used to prefer same-scope resolutions.
    pub fn scope_prefix(&self) -> &str {
        match self.path.rfind('.') {
            Some(i) => &self.path[..i],
            None => "",
        }
    }
}

/// Parse a type reference string into (path, arity): `Ns.Foo<K,V>`
/// → (`Ns.Foo`, 2); `Bar` → (`Bar`, 0); `A.B<T>.C` → (`A.B.C`, 1)
/// (arity counts the LAST segment's args — sufficient for lookup).
pub fn split_arity(name: &str) -> (String, usize) {
    let name = name.trim();
    match name.find('<') {
        None => (name.to_owned(), 0),
        Some(i) => {
            let (path, rest) = name.split_at(i);
            // Count top-level comma-separated args. Brackets of any
            // kind nest: tuple element commas (`(A a, B b)`), array
            // rank commas (`[,]`), and nested generics never separate
            // (`IReadOnlyCollection<(TElement Element, TPriority
            // Priority)>` has arity 1, not 2).
            let mut depth = 0i32;
            let mut nesting = 0i32;
            let mut count = 1usize;
            let mut any_token = false;
            for ch in rest.chars() {
                match ch {
                    '<' => depth += 1,
                    '>' => depth -= 1,
                    '(' | '[' => nesting += 1,
                    ')' | ']' => nesting -= 1,
                    ',' if depth == 1 && nesting == 0 => count += 1,
                    ' ' | '\t' => {}
                    _ => {
                        if depth >= 1 {
                            any_token = true;
                        }
                    }
                }
            }
            if !any_token {
                return (path.trim().to_owned(), 0);
            }
            (path.trim().to_owned(), count)
        }
    }
}

/// One compiled method within a class.
#[derive(Debug, Clone)]
pub struct MethodInfo {
    pub name: String,
    pub ret_type: String,
    pub param_types: Vec<String>,
    /// Declared parameter names, in parameter order. Named arguments
    /// (`M(b: 1, a: 2)`) are rewritten into this order before anything
    /// evaluates them.
    pub param_names: Vec<String>,
    /// Per-param: whether the parameter declares a default value
    /// (`void M(int x = 5)`). Calls omitting trailing defaulted params
    /// are filled from `param_defaults` at lowering (C# optional args).
    pub param_has_default: Vec<bool>,
    /// The last parameter is a `params` array (`void M(params int[]
    /// xs)`): calls pass zero or more trailing args of the element
    /// type, forwarded as extra argv (expanded form). Always last by
    /// C# rules, so one flag suffices.
    pub has_params: bool,
    /// Folded default values (same positions as `param_types`); `None`
    /// with `param_has_default[i] == true` means null/`default`/unfoldable
    /// and materializes as zero at the call site.
    pub param_defaults: Vec<Option<ConstValue>>,
    /// Per-param: by-reference (`ref`/`out`/`in`). Callers box the
    /// value and pass the box address; callees dereference on access
    /// so writes flow back (required for `out` results).
    pub param_is_byref: Vec<bool>,
    /// Declared visibility; drives C#-style access enforcement.
    pub access: Access,
    /// `static` modifier — called on the type, not an instance.
    pub is_static: bool,
    /// Whether the declaration carries a body. Bodyless methods
    /// (`extern`/`InternalCall`/abstract/partial-no-impl) generate no
    /// function: static calls to them lower to a zero placeholder
    /// (runtime-implemented natives bind for real later).
    pub has_body: bool,
    /// The class that declares this method. Overloads may declare
    /// different access (`internal` 2-arg + `private` 3-arg): access
    /// enforcement resolves against the SELECTED overload's entry,
    /// never a name-only first match.
    pub declaring_class: String,
    /// Extension method (`static` with a `this` first parameter):
    /// callable as `recv.M(args)`, lowered as `Class.M(recv, args)`.
    pub is_extension: bool,
    /// Method-level generic parameters (`M<T>(...)`): drive generic
    /// argument inference for extension calls (`ReadOnlySpan<T>` ←
    /// `ReadOnlySpan<char>` binds `T`).
    pub generic_params: Vec<String>,
    /// `[Conditional("SYM")]` symbol: calls to this method are omitted
    /// unless `SYM` is defined (C# release semantics — e.g.
    /// `Debug.Assert` without `DEBUG`). `None` for ordinary methods.
    pub conditional_symbol: Option<String>,
    /// Slot in the class vtable (assigned per unique method name across the
    /// hierarchy).
    pub vtable_slot: usize,
}

/// The compile-time value of a `const` field, constant-folded at semantic
/// analysis time. `const` fields occupy no runtime storage: every read is
/// replaced by this literal.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstValue {
    Int(i64),
    Float(f64),
    /// Compile-time string constants (`const string X = "..."`,
    /// concatenations thereof). Lowering materializes these (they
    /// occupy no object storage).
    Str(String),
    /// `const string X = null` (legal C#: the null reference is a
    /// valid string-constant value). Lowers to the null handle (0),
    /// exactly like an unresolved/defaulted reference.
    Null,
}

/// A property with its declared type (own + inherited). Explicit
/// (accessor-backed) properties occupy no storage — reads lower to a
/// placeholder — but their TYPE drives overload applicability
/// (`dateTime.Ticks` is `long`, not `object`).
#[derive(Debug, Clone)]
pub struct PropertyInfo {
    pub name: String,
    pub type_name: String,
    pub access: Access,
    /// The class that declares this property.
    pub declaring_class: String,
    /// `static` modifier: only static properties participate in
    /// `using static` bare-name imports (mirrors FieldInfo).
    pub is_static: bool,
    /// `abstract` modifier: body-less accessors declare no storage
    /// (never auto-property backing fields).
    pub is_abstract: bool,
}

/// A field with its object-layout offset (in slots, after the vtable word).
#[derive(Debug, Clone)]
pub struct FieldInfo {
    pub name: String,
    pub type_name: String,
    pub access: Access,
    /// Offset from the start of the object (in 8-byte slots). The vtable
    /// pointer occupies slot 0, so the first field is slot 1.
    pub offset_slots: usize,
    /// `static` modifier: the field lives in per-class static storage, not in
    /// the object layout (`offset_slots` is meaningless for it).
    pub is_static: bool,
    /// `const` modifier: a compile-time constant occupying no storage. Reads
    /// are constant-folded to `const_value`.
    pub is_const: bool,
    /// The folded value of a `const` field; `None` for non-const fields.
    pub const_value: Option<ConstValue>,
    /// `ref` modifier: the slot holds an address (see `FieldDecl`).
    pub is_byref: bool,
    /// The class that declares this field. For inherited statics accessed
    /// through a derived type, this is the base class that owns the storage.
    pub declaring_class: String,
}

/// Slots at the head of a class instance, before its fields.
///
/// Slot 0 holds the vtable address, slot 1 the reference count, and
/// slot 2 the instance's RUNTIME TYPE — the descriptor of the
/// instantiation, which is what carries a generic class's type
/// arguments (a `ReadOnlyMemory<byte>` instance knows its `byte`).
/// The runtime type is a pointer, so it is one slot wide like the
/// other two.
///
/// Every layout site reads the size from here: the field slots start
/// at `INSTANCE_HEADER_SLOTS`, and the instance is that many slots
/// plus the fields. A base constructor writes at its own class's
/// offsets and the receiver it gets is adjusted by the difference
/// between the derived layout's first field slot and this one, so the
/// adjustment stays right when the header grows.
pub const INSTANCE_HEADER_SLOTS: usize = 3;

/// The slot the instance's runtime type lives in.
pub const RUNTIME_TYPE_SLOT: usize = 2;

/// The first slot a class instance's fields occupy.
pub const FIRST_FIELD_SLOT: usize = INSTANCE_HEADER_SLOTS;

/// Everything known about one class after semantic analysis.
#[derive(Debug, Clone)]
pub struct ClassInfo {
    pub name: String,
    /// Qualified identity (namespace + outer path + arity). Distinct
    /// types sharing a simple name (`Task` vs `Task<TResult>`,
    /// `A.HashHelpers` vs `B.HashHelpers`) have distinct keys; `name`
    /// stays simple for display and ABI-adjacent uses.
    pub key: TypeKey,
    /// Resolved base chain (classes + interfaces), declaration order,
    /// duplicates removed. Empty when the class has no bases.
    pub bases: Vec<String>,
    /// Instance fields in layout order (base fields first). Static fields are
    /// NOT here — they live in `static_fields` and occupy no object slot.
    pub fields: Vec<FieldInfo>,
    /// Static fields (base chain first, then own). They live in per-class
    /// static storage, not the object layout.
    pub static_fields: Vec<FieldInfo>,
    /// `const` fields (base chain first, then own). They occupy no storage —
    /// reads are constant-folded to their `const_value`.
    pub const_fields: Vec<FieldInfo>,
    /// Methods visible on this class (own + inherited), keyed by name.
    pub methods: Vec<MethodInfo>,
    /// Names + access of members DECLARED directly on this class (not
    /// inherited). Access enforcement resolves against these.
    pub own_members: Vec<(String, bool /*is_field*/, Access)>,
    /// Interfaces this class implements directly or transitively, as
    /// (simple name, arity). Arity distinguishes `ICollection` (0) from
    /// `ICollection<T>` (1) — name-only lists falsely attribute generic
    /// members to non-generic implementers.
    pub interfaces: Vec<(String, usize)>,
    /// Properties visible on this class (own + inherited), keyed by
    /// name. Explicit properties occupy no storage but their TYPE
    /// drives overload applicability.
    pub properties: Vec<PropertyInfo>,
    /// Enum marker: enums lower as const-field holders (no methods).
    /// Drives enum-precise typing (a `FileMode.Open` access types as
    /// `FileMode`, not its `int` underlying type) for overload
    /// applicability.
    pub is_enum: bool,
    /// Generic type parameter names (`Box<T>` → `["T"]`). Lets overload
    /// ranking treat class-level type variables as applicable wildcards
    /// (`new WeakGCHandle<NativeObjectWrapper>(wrapper)` binds `T`),
    /// mirroring how method-level `generic_params` already work.
    pub generic_params: Vec<String>,
    /// Index of each of this class's OWN type parameters inside the
    /// instantiation's type-argument list (the base chain's arguments
    /// come first, .NET order). `typeof(T)` inside a method of a
    /// generic class reads the argument at this index from the
    /// instance's runtime type.
    pub type_arg_index: Vec<u32>,
    /// `struct` declaration: keeps its implicit parameterless
    /// constructor even with explicit ctors (classes do not).
    pub is_struct: bool,
    /// `record` declaration: `==` compares by VALUE, field by field.
    pub is_record: bool,
    /// Declaring library (project name) for dynamic-linking ownership:
    /// calls into methods whose declaring entry has a different owner
    /// than the main project become runtime imports instead of
    /// statically compiled copies. Nested types inherit their file's
    /// lib via the ownership post-pass in `analyze_with_owners`.
    /// Empty when unknown (bootstrap paths, unit tests, legacy callers
    /// of `analyze`) — treated as local.
    pub owner: String,
}

impl ClassInfo {
    pub fn field(&self, name: &str) -> Option<&FieldInfo> {
        // Derived-first: a derived field hides same-named base fields
        // (C# hiding, `new` or implicit). The vec is layout order
        // (base fields first), so search in reverse — otherwise
        // `InvokeGenerator._fac` (XsltQilFactory) resolves to the base
        // `QilCloneVisitor._fac` (QilFactory) and member calls on it
        // mis-resolve.
        self.fields
            .iter()
            .rev()
            .find(|f| f.name == name)
            .or_else(|| self.static_fields.iter().rev().find(|f| f.name == name))
            .or_else(|| self.const_fields.iter().rev().find(|f| f.name == name))
    }

    pub fn method(&self, name: &str) -> Option<&MethodInfo> {
        self.methods.iter().find(|m| m.name == name)
    }

    pub fn property(&self, name: &str) -> Option<&PropertyInfo> {
        // Derived-first, like `field` above (hiding).
        self.properties.iter().rev().find(|p| p.name == name)
    }

    /// Overload selection for a call with `nargs` explicit arguments:
    /// exact arity first, else the smallest overload whose trailing
    /// params (past `nargs`) all have defaults (C# optional arguments).
    /// `None` when no overload can accept `nargs` arguments.
    pub fn method_for_call(&self, name: &str, nargs: usize) -> Option<&MethodInfo> {
        if let Some(m) = self
            .methods
            .iter()
            .find(|m| m.name == name && m.param_types.len() == nargs)
        {
            return Some(m);
        }
        self.methods
            .iter()
            .filter(|m| {
                m.name == name
                    && m.param_types.len() > nargs
                    && m.param_has_default
                        .get(nargs..)
                        .is_some_and(|tail| tail.iter().all(|&h| h))
            })
            .min_by_key(|m| m.param_types.len())
    }
}

/// One declared external import ([LibraryImport] stub).
#[derive(Debug, Clone)]
pub struct ImportFn {
    /// Declaring class (scoping only — ns/class never reach the ABI).
    pub class: String,
    pub method: String,
    /// Library hint from the attribute (e.g. `"native_add"`).
    pub library: String,
    /// Actual exported symbol to bind against (`EntryPoint`, else method).
    pub entry_point: String,
    pub ret_type: String,
    pub param_types: Vec<String>,
}

/// Full semantic picture of a compilation unit's types.
#[derive(Debug, Default)]
pub struct TypeTable {
    pub classes: Vec<ClassInfo>,
    pub interfaces: Vec<InterfaceDecl>,
    /// `delegate` declarations (`Predicate<T>`, `Func<...>`, custom
    /// ones alike): method-group convertibility resolves against
    /// DECLARED delegates, never a hardcoded name list — the runtime
    /// owns its delegate types, the compiler only indexes them.
    pub delegates: Vec<DelegateDecl>,
    /// `[LibraryImport]` declarations in source order.
    pub imports: Vec<ImportFn>,
    /// `using Alias = Target` bindings (alias → target as written).
    /// Lookups resolve through them: the alias is another name for
    /// the target type.
    pub aliases: Vec<(String, String)>,
}

impl TypeTable {
    /// True when `name` (nullable suffix, generic arguments, and
    /// qualification stripped) names a DECLARED `delegate` type.
    /// Method-group convertibility consults this — never a hardcoded
    /// name list.
    pub fn is_delegate(&self, name: &str) -> bool {
        let mut p = name.trim_end_matches('?');
        if let Some(k) = p.find('<') {
            p = &p[..k];
        }
        let simple = p.rsplit('.').next().unwrap_or(p);
        self.delegates.iter().any(|d| {
            d.name == simple
                || d.name.rsplit('.').next().unwrap_or(&d.name) == simple
        })
    }
    /// Resolve a `using`-alias to its target as written
    /// (`System.Int32`, or `System.Collections.Generic.Queue<object>`
    /// with generic arguments); `None` when `name` is not an alias.
    /// Callers that need a bare name strip generics/qualification
    /// themselves (arity-aware matching lives in `class`).
    pub fn alias_target(&self, name: &str) -> Option<&str> {
        self.aliases
            .iter()
            .find(|(alias, _)| alias == name)
            .map(|(_, target)| target.as_str())
    }

    /// ALL targets of a `using`-alias name, as written. Aliases are
    /// file-scoped in C# but the unit merges files, so one name may map
    /// to several types (`NativeType` → `IntPtr` in CLong.cs, `Double`
    /// in NFloat.cs). Selection tries every target's entry (see
    /// `entries_for`); first-match alone picks the wrong type.
    pub fn alias_targets(&self, name: &str) -> Vec<&str> {
        let mut out: Vec<&str> = Vec::new();
        for (alias, target) in &self.aliases {
            if alias == name && !out.contains(&target.as_str()) {
                out.push(target);
            }
        }
        out
    }

    pub fn class(&self, name: &str) -> Option<&ClassInfo> {
        // Backward-compatible first match by simple name (ignores
        // namespace/arity). Correct for unambiguous names (the common
        // case); ambiguous ones need `resolve` + `class_key` below.
        // A qualified query (`Ns.Foo`) matches its exact path too, plus
        // partially-qualified spellings (`Matrix3x2.Impl` for
        // `System.Numerics.Matrix3x2.Impl`) by suffix — first suffix hit
        // wins (documented ambiguity limitation).
        // A BARE spelling that has a non-generic declaration must bind to
        // it, even when a generic same-named type was declared first:
        // `Box<int>` and `Box` are different types with different layouts,
        // and first-match put both spellings on one entry, so
        // `Box<int> x; x.A = 5;` and `Box y; y.B = 3;` wrote into the same
        // slots and read zero back.
        //
        // Strictly additive on purpose — it changes an answer ONLY when a
        // non-generic declaration exists. Code that writes `Span` for an
        // argument-erased `Span<T>` keeps resolving as before, which a
        // broader "prefer arity 0" rule broke.
        // A spelling WITH arguments names that arity — but ONLY when the
        // name is genuinely ambiguous, i.e. a same-named declaration of
        // another arity also exists. Applying it unconditionally changed
        // the answer for spellings like `Span<T>` whose only declaration
        // is generic, which broke calls that rely on the old resolution.
        if name.contains('<') {
            let (bare, arity) = split_arity(name);
            let same_name = |c: &&ClassInfo| c.name == bare || c.key.path == bare;
            if self.classes.iter().any(|c| same_name(&c) && c.key.arity != arity) {
                if let Some(c) = self.classes.iter().find(|c| {
                    same_name(&c)
                        && (c.key.arity == arity || c.generic_params.len() == arity)
                }) {
                    return Some(c);
                }
            }
        }
        if !name.contains('<')
            && let Some(zero) = self.classes.iter().find(|c| {
                (c.name == name || c.key.path == name) && c.generic_params.is_empty()
            })
        {
            return Some(zero);
        }
        self.classes
            .iter()
            .find(|c| c.name == name || c.key.path == name)
            .or_else(|| {
                // Alias targets carry generic arguments
                // (`WorkQueue` → `...Queue<object>`): match by base
                // path/name AND arity, so the generic definition wins
                // over a same-named non-generic class (legacy `Queue`).
                let target = self.alias_target(name)?;
                let (base_path, arity) = split_arity(target);
                let simple = base_path.rsplit('.').next().unwrap_or(&base_path);
                let suff = format!(".{base_path}");
                self.classes.iter().find(|c| {
                    (c.key.path == base_path || c.key.path.ends_with(&suff) || c.name == simple)
                        && (c.key.arity == arity || c.generic_params.len() == arity)
                })
            })
            .or_else(|| {
                if !name.contains('.') {
                    return None;
                }
                let suff = format!(".{name}");
                self.classes.iter().find(|c| c.key.path.ends_with(&suff))
            })
    }

    /// Precise lookup by qualified key.
    pub fn class_key(&self, key: &TypeKey) -> Option<&ClassInfo> {
        self.classes.iter().find(|c| &c.key == key)
    }

    /// Like [`TypeTable::class`], but same-named entries with arity ==
    /// `wanted` win over first-match order (`WeakReference` arity 0 vs
    /// `WeakReference<T>` arity 1 — access checks must resolve against
    /// the entry the spelling actually names). Falls back to plain
    /// first-match when no entry carries the wanted arity.
    pub fn class_arity(&self, name: &str, wanted: usize) -> Option<&ClassInfo> {
        let first = self.class(name)?;
        if first.key.arity == wanted || first.generic_params.len() == wanted {
            return Some(first);
        }
        self.classes
            .iter()
            .filter(|c| c.name == name || c.key.path == name)
            .find(|c| c.key.arity == wanted || c.generic_params.len() == wanted)
            .or(Some(first))
    }

    /// Resolve a type reference string to a key, preferring `context`
    /// (the referring scope: same namespace/outer first, then global).
    /// Returns `None` when nothing matches.
    pub fn resolve(&self, name: &str, context: Option<&TypeKey>) -> Option<TypeKey> {
        let name = name.trim();
        // Qualified reference (`Ns.Foo`, `A.B<T>`): match path + arity.
        let (path, arity) = split_arity(name);
        if path.contains('.') {
            return self
                .classes
                .iter()
                .map(|c| c.key.clone())
                .find(|k| k.path == path && k.arity == arity);
        }
        // Bare reference: candidates with matching simple name + arity.
        let mut candidates: Vec<TypeKey> = self
            .classes
            .iter()
            .map(|c| c.key.clone())
            .filter(|k| k.simple_name() == path && k.arity == arity)
            .collect();
        if candidates.is_empty() {
            return None;
        }
        if candidates.len() == 1 {
            return Some(candidates.swap_remove(0));
        }
        // Ambiguous: prefer the referrer's own scope, then global.
        if let Some(ctx) = context {
            if let Some(k) = candidates
                .iter()
                .find(|k| k.scope_prefix() == ctx.scope_prefix())
            {
                return Some(k.clone());
            }
            if let Some(k) = candidates.iter().find(|k| k.scope_prefix().is_empty()) {
                return Some(k.clone());
            }
        }
        candidates.into_iter().next()
    }

    /// The entries a bare simple name could mean, when more than one
    /// declares it with the same arity and the referrer's scope cannot
    /// decide (no context). C# reports CS0104 here; the compiler used
    /// to take the first and say nothing, so a reader had no way to know
    /// the choice was arbitrary.
    pub fn interface(&self, name: &str) -> Option<&InterfaceDecl> {
        self.interfaces.iter().find(|i| i.name == name)
    }

    /// Does `class_name` implement/transitively derive from `ancestor`?
    pub fn derives_from(&self, class_name: &str, ancestor: &str) -> bool {
        self.derives_from_seen(class_name, ancestor, &mut Vec::new())
    }

    /// Cycle-guarded `derives_from`: base graphs with spelling-variant
    /// cycles (blocked from `resolve_chain` but reachable here through
    /// per-call string spellings) must terminate — a cycle contributes
    /// nothing (returns false for that branch), never hangs the build.
    fn derives_from_seen(
        &self,
        class_name: &str,
        ancestor: &str,
        seen: &mut Vec<String>,
    ) -> bool {
        if class_name == ancestor {
            return true;
        }
        // Same entry under different spellings (qualified
        // `System.IO.Stream.ReadWriteTask` vs simple `ReadWriteTask`).
        if let (Some(a), Some(b)) = (self.class(class_name), self.class(ancestor)) {
            if a.key == b.key {
                return true;
            }
        }
        if seen.contains(&class_name.to_owned()) {
            return false;
        }
        seen.push(class_name.to_owned());
        // An exact qualified path decides on its own — no ambiguity is
        // possible there.
        if let Some(info) = self.classes.iter().find(|c| c.key.path == class_name) {
            return info
                .bases
                .iter()
                .any(|b| self.derives_from_seen(b, ancestor, seen));
        }
        // A BARE name is the unresolved case, and it is inherent to the
        // data model rather than a choice made here: `MethodInfo` /
        // `FieldInfo` record `declaring_class` as a NAME, so by the time
        // an access question is asked, `ArrayConverter` could be either
        // of two real types. Walking every same-named entry and asking
        // which of them actually derives from the declaring type answers
        // the question "is there a type called `ArrayConverter` that
        // derives from it?" — which is the most that can be known here.
        //
        // It can grant access that C# would refuse, when a same-named
        // type in another family happens to derive from the ancestor.
        // Making it exact means carrying the member's TypeKey instead of
        // its name; ambiguous bare names are
        // reported by W0005 in the meantime.
        self.classes
            .iter()
            .filter(|c| c.name == class_name)
            .any(|info| {
                info.bases
                    .iter()
                    .any(|b| self.derives_from_seen(b, ancestor, seen))
            })
    }

    /// Does interface `iface` extend `ancestor` (transitively)? Names
    /// compare by simple name (generics erased, mirroring the rest of
    /// overload applicability): `IList` → `ICollection` →
    /// `IEnumerable`, so extension methods on `IEnumerable<T>`
    /// (`TryGetSpan`) accept `IList<T>` receivers. Cycle-guarded.
    pub fn interface_derives_from(&self, iface: &str, ancestor: &str) -> bool {
        fn simple(t: &str) -> &str {
            let mut s = t.trim();
            if let Some(i) = s.find('<') {
                s = &s[..i];
            }
            s.rsplit('.').next().unwrap_or(s)
        }
        if simple(iface) == simple(ancestor) {
            return true;
        }
        let mut seen: Vec<String> = Vec::new();
        let mut stack = vec![simple(iface).to_owned()];
        let want = simple(ancestor).to_owned();
        while let Some(cur) = stack.pop() {
            if seen.contains(&cur) {
                continue;
            }
            seen.push(cur.clone());
            for decl in self.interfaces.iter().filter(|i| i.name == cur) {
                for b in &decl.bases {
                    let sb = simple(b).to_owned();
                    if sb == want {
                        return true;
                    }
                    stack.push(sb);
                }
            }
        }
        false
    }
}

// ---- access checking -------------------------------------------------------

/// C# visibility rules. `Default` (no modifier on a class member) behaves
/// like `private`, matching C#'s default member accessibility.
#[allow(dead_code)] // used by the IR lowering access pass
pub fn visible_from(table: &TypeTable, access: Access, declaring: &str, viewer: &str) -> bool {
    match access {
        Access::Public => true,
        // `internal` is visible throughout the compilation unit: every
        // `.rcp` build is a single assembly today. Assembly-boundary
        // enforcement arrives with cross-library references.
        Access::Internal => true,
        Access::Private | Access::Default => declaring == viewer,
        Access::Protected => declaring == viewer || table.derives_from(viewer, declaring),
    }
}

impl ClassInfo {
    /// Does THIS class itself declare the member (not inherit it)?
    pub fn declares_field(&self, name: &str) -> Option<&FieldInfo> {
        self.fields.iter().find(|f| f.name == name)
    }

    pub fn declares_method(&self, name: &str) -> Option<&MethodInfo> {
        self.methods.iter().find(|m| m.name == name)
    }
}
