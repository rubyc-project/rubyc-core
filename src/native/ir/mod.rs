//! Virtual-register IR and AST lowering.
//!
//! [`instructions`] defines the three-address form the backend consumes;
//! [`lower`] translates typed AST into it. See [`instructions`] for the
//! representation contract.

mod instructions;
mod lower;
mod serialize;

pub use instructions::{
    BinOp, Block, BlockId, CmpOp, FBinOp, FCmpOp, Instr, IrFunction, IrStatic, Program,
    ReceiverKind, Terminator, VReg,
};
pub use serialize::{decode_program, encode_program, encode_program_no_magic, IR_FORMAT_VERSION};

/// Shared literal-string arena. Offsets assigned here are the `Sym::String`
/// values the encoder relocates against; the bytes land at the start of the
/// image's data section.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct StringArena {
    pub bytes: Vec<u8>,
    index: std::collections::BTreeMap<String, (u32, u32)>,
}

impl StringArena {
    pub fn new() -> Self {
        Self::default()
    }

    /// `(offset, len)` of the literal's bytes, deduplicated.
    pub fn literal(&mut self, s: &str) -> (u32, u32) {
        if let Some(hit) = self.index.get(s) {
            return *hit;
        }
        let off = self.bytes.len() as u32;
        self.bytes.extend_from_slice(s.as_bytes());
        let hit = (off, s.len() as u32);
        self.index.insert(s.to_owned(), hit);
        hit
    }
}

use crate::models::{Access, ClassDecl, ClassMember, CompilationUnit, Member, MethodDecl};
use crate::native::semantic::{self, ClassInfo, MethodInfo, SemanticError, TypeTable};

/// Why lowering failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LowerError {
    NoMainMethod,
    UnsupportedConstruct(String),
    /// Semantic analysis (name resolution, MI conflicts, base calls) rejected
    /// the unit; carries the structured errors with their fix hints.
    Semantic(Vec<SemanticError>),
    /// C# visibility rule violated; `code` is the stable diagnostic id.
    Access {
        code: &'static str,
        message: String,
        help: String,
    },
    /// Cross-image mutable static access in a dynamic-linking build:
    /// the static lives in another image (copy-linking would snapshot
    /// it — silently wrong for mutable state). Fails the build loudly
    /// instead; the library stays static (remove `[build] dynamic`)
    /// until GOT-style sharing lands.
    ExternalStatic {
        class: String,
        field: String,
    },
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LowerError::NoMainMethod => write!(
                f,
                "{}\n  help: add a method named 'main' to one class — it becomes the entry point",
                crate::t!("rubyc.native.ir.LowerError.no_main")
            ),
            LowerError::UnsupportedConstruct(s) => {
                let s = s.clone();
                write!(
                    f,
                    "{}",
                    crate::tf!("rubyc.native.ir.LowerError.unsupported_prefix", s)
                )
            }
            LowerError::Semantic(errors) => {
                let mut out = String::new();
                for (i, e) in errors.iter().enumerate() {
                    if i > 0 {
                        out.push('\n');
                    }
                    out.push_str(&format!("error[{}]: {}", e.code, e.message));
                    if let Some(help) = &e.help {
                        out.push_str(&format!("\n  help: {help}"));
                    }
                }
                write!(f, "{out}")
            }
            LowerError::Access {
                code,
                message,
                help,
            } => write!(
                f,
                "{}\n  help: {help}",
                crate::tf!(
                    "rubyc.native.ir.LowerError.access",
                    code.to_string(),
                    message.clone()
                )
            ),
            LowerError::ExternalStatic { class, field } => write!(
                f,
                "error[E0040]: cross-image mutable static '{class}.{field}' in a dynamic-linking build\n  help: shared mutable statics need GOT-style sharing (not implemented); build this library static (remove `[build] dynamic`)"
            ),
        }
    }
}

/// Per-function lowering context shared by [`lower`] passes.
#[derive(Clone, Copy)]
pub(crate) struct Ctx<'a> {
    pub table: &'a TypeTable,
    pub current_class: &'a str,
    /// Table entry matching the method under lowering, disambiguating
    /// same-named nested/generic types (`HashSet`/`Dictionary`/
    /// `ConcurrentDictionary` each nest an `AlternateLookup`): matched
    /// by method name + arity + param types, falling back to
    /// first-match. `None` outside method bodies. Implicit-`this`
    /// resolutions prefer this over name lookup.
    pub current_info: Option<&'a ClassInfo>,
    /// Qualified AST path of the class being lowered
    /// (`Ns.OrderedDictionary.KeyCollection.Enumerator`), when known:
    /// anchors `current_info` to the right same-named entry (25
    /// `Enumerator`s share names/signatures). `None` outside method
    /// bodies or when the path is unavailable (legacy first-match).
    pub current_path: Option<&'a str>,
    pub classes: &'a [&'a ClassDecl],
    /// `(class, method)` → index into `Program::imports`.
    pub import_index: &'a std::collections::BTreeMap<(String, String), usize>,
    /// Cross-image (dependency) imports registered during lowering:
    /// `(lib hint, symbol)` → dep slot, appended after host imports in
    /// `Program::imports`. Shared across all method bodies of one
    /// build (first-use order = deterministic AST order). Empty in
    /// static builds (no external callees) — then `Program::imports`
    /// is exactly today's host list.
    pub dep_imports:
        &'a std::cell::RefCell<std::collections::BTreeMap<(String, String), usize>>,
    /// Number of host imports (`table.imports.len()`): dep call
    /// indices are offset by this (`host + dep_slot`).
    pub host_import_count: usize,
    /// Generic type parameters of the method under lowering (`M<T>`).
    /// Empty outside method bodies (field initializers, tests).
    pub method_vars: &'a [String],
    /// `(declaring class, field name)` → index into `Program::statics`.
    pub static_index: &'a std::collections::BTreeMap<(String, String), u32>,
    /// Active preprocessor symbols (CLI `-D` plus `[build] defines`):
    /// `[Conditional("SYM")]` calls omit unless listed here.
    pub defines: &'a [String],
}

impl Ctx<'_> {
    /// True when `name` is a generic type parameter in scope — the
    /// class's (`MemberInfoCache<T>`) or the method's (`ThrowIfZero<T>`).
    /// Such names are unbound variables: file-scoped `using` aliases
    /// (e.g. `using T = ...`) and table lookups must never resolve
    /// them to concrete types. Used at every alias-resolution site
    /// (local declarations, member lookup).
    pub(crate) fn is_type_var(&self, name: &str) -> bool {
        if name.contains('.') || name.contains('<') {
            return false;
        }
        // Strip nullable/pointer/array suffixes (`T?`, `T*`, `T[]`):
        // the variable is still unbound.
        let mut base = name.trim();
        loop {
            if let Some(s) = base.strip_suffix("[]") {
                base = s;
            } else if let Some(s) = base.strip_suffix(['?', '*']) {
                base = s;
            } else {
                break;
            }
        }
        if self.method_vars.iter().any(|p| p == base) {
            return true;
        }
        self.current_info
            .is_some_and(|info| info.generic_params.iter().any(|p| p == base))
    }

    /// Enforce C# member visibility for `member` accessed through static
    /// type `owner_type`. The declaring class is located along the owner's
    /// base chain; `Default` behaves like `private`.
    pub(crate) fn check_access(
        &self,
        is_field: bool,
        owner_type: &str,
        member: &str,
    ) -> Result<(), LowerError> {
        let found = if is_field {
            self.table.declaring_class_of_field(owner_type, member)
        } else {
            self.table.declaring_class_of_method(owner_type, member)
        };
        let Some((declaring, access)) = found else {
            return Ok(());
        };
        if semantic::visible_from(self.table, access, &declaring, self.current_class) {
            return Ok(());
        }
        Err(access_denied(access, is_field, &declaring, member))
    }

    /// Overload-aware method visibility: enforce the SELECTED overload's
    /// own access + declarer. Name-only lookup conflates overloads with
    /// different access (`FileSystem.MoveDirectory`: `internal` 2-arg +
    /// `private` 3-arg) and falsely denies legal calls.
    pub(crate) fn check_method_access(&self, mi: &MethodInfo) -> Result<(), LowerError> {
        if semantic::visible_from(self.table, mi.access, &mi.declaring_class, self.current_class)
        {
            return Ok(());
        }
        Err(access_denied(mi.access, false, &mi.declaring_class, &mi.name))
    }

    /// Virtual-slot method visibility: like `check_method_access`, but an
    /// inaccessible `protected` override is redeemed by an accessible
    /// same-signature declaration on a base of the RECEIVER
    /// (`JsonNode.ReplaceWith` calling `jsonArray.SetItem` — a `private
    /// protected` override of the `private protected virtual` declared
    /// on `JsonNode` itself, which C# binds and access-checks as the
    /// base slot). Dispatch stays virtual, so the override still runs;
    /// only the access verdict uses the base slot. Signature equality
    /// (name + exact param types) keeps unrelated same-named overloads
    /// from rescuing each other.
    pub(crate) fn check_method_access_virtual(
        &self,
        mi: &MethodInfo,
        recv_type: &str,
    ) -> Result<(), LowerError> {
        if semantic::visible_from(self.table, mi.access, &mi.declaring_class, self.current_class)
        {
            return Ok(());
        }
        if matches!(mi.access, Access::Protected) {
            if let Some(info) = self.table.class(recv_type) {
                for m in &info.methods {
                    if m.name == mi.name
                        && m.param_types == mi.param_types
                        && semantic::visible_from(
                            self.table,
                            m.access,
                            &m.declaring_class,
                            self.current_class,
                        )
                    {
                        return Ok(());
                    }
                }
            }
        }
        Err(access_denied(mi.access, false, &mi.declaring_class, &mi.name))
    }

    /// Import index for a bare-name stub of the current class, if any.
    pub(crate) fn import_of(&self, callee: &str) -> Option<usize> {
        self.import_index
            .get(&(self.current_class.to_owned(), callee.to_owned()))
            .copied()
    }
}

/// Build the access-denied error for `member` declared by `declaring`
/// (shared by the name-based and overload-aware checks).
fn access_denied(access: Access, is_field: bool, declaring: &str, member: &str) -> LowerError {
    let (code, message, help) = match (access, is_field) {
        (Access::Protected, false) => (
            "E0018",
            crate::tf!(
                "rubyc.native.ir.access.protected_method",
                declaring.to_owned(),
                member.to_owned()
            ),
            format!(
                "declare '{member}' public, or call it from inside '{declaring}' or a class derived from it"
            ),
        ),
        (Access::Protected, true) => (
            "E0018",
            crate::tf!(
                "rubyc.native.ir.access.protected_field",
                declaring.to_owned(),
                member.to_owned()
            ),
            format!(
                "declare '{member}' public, or access it from inside '{declaring}' or a class derived from it"
            ),
        ),
        (_, false) => (
            "E0017",
            crate::tf!(
                "rubyc.native.ir.access.private_method",
                declaring.to_owned(),
                member.to_owned()
            ),
            format!(
                "declare '{member}' public, or add a method inside '{declaring}' that reads it for you"
            ),
        ),
        (_, true) => (
            "E0017",
            crate::tf!(
                "rubyc.native.ir.access.private_field",
                declaring.to_owned(),
                member.to_owned()
            ),
            format!(
                "declare '{member}' public, or add a method inside '{declaring}' that wraps it"
            ),
        ),
    };
    LowerError::Access {
        code,
        message,
        help,
    }
}

/// Resolve the table entry for the method under lowering among
/// same-named classes: exact (method name, arity, param types) first,
/// then (name, arity), then first-match legacy. Disambiguates
/// same-named nested/generic types by what they contain, so
/// implicit-`this` resolutions hit the right entry.
fn current_entry<'t>(
    table: &'t TypeTable,
    class_name: &str,
    method: &MethodDecl,
    path_hint: Option<&str>,
) -> Option<&'t ClassInfo> {
    let mtypes: Vec<String> = method.params.iter().map(|p| p.type_name.clone()).collect();
    let mut fallback: Option<&ClassInfo> = None;
    let mut arity_fallback: Option<&ClassInfo> = None;
    // Same-named candidates with the hinted entry first: a bare
    // name+signature match is ambiguous across same-named nested
    // classes (`Enumerator.MoveNext()` exists 25 times), so the AST
    // path of the class under lowering wins over table order.
    let mut cands: Vec<&ClassInfo> = table
        .classes
        .iter()
        .filter(|c| c.name == class_name)
        .collect();
    if let Some(hint) = path_hint {
        cands.sort_by_key(|c| if c.key.path == hint { 0 } else { 1 });
    }
    for c in cands {
        if fallback.is_none() {
            fallback = Some(c);
        }
        let mut arity_hit = false;
        for mi in c.methods.iter().filter(|m| m.name == method.name) {
            if mi.param_types.len() != mtypes.len() {
                continue;
            }
            arity_hit = true;
            if mi.param_types == mtypes {
                return Some(c);
            }
        }
        if arity_hit && arity_fallback.is_none() {
            arity_fallback = Some(c);
        }
    }
    arity_fallback.or(fallback)
}

/// Analyze + lower a unit into a multi-function program.
///
/// Every class method becomes an [`IrFunction`] named `Class::method`; a
/// synthetic `@start` constructs the entry class, runs field initializers,
/// and calls `main` with the instance as receiver.
pub fn lower(unit: &CompilationUnit) -> Result<Program, LowerError> {
    lower_with(unit, true)
}

/// Like [`lower`], but `require_main == false` accepts units without a
/// `main` (shared libraries): no synthetic `@start` is emitted and function
/// indices are used as-is.
pub fn lower_with(unit: &CompilationUnit, require_main: bool) -> Result<Program, LowerError> {
    lower_with_defines(unit, require_main, &[])
}

/// Like [`lower_with`], plus the active preprocessor symbols: calls to
/// `[Conditional("SYM")]` methods are omitted unless `SYM` is defined
/// (C# release semantics — e.g. `Debug.Assert` without `DEBUG`).
pub fn lower_with_defines(
    unit: &CompilationUnit,
    require_main: bool,
    defines: &[String],
) -> Result<Program, LowerError> {
    lower_with_owners(
        unit,
        require_main,
        defines,
        &std::collections::HashMap::new(),
        &semantic::Preloaded::empty(),
    )
}

/// Like [`lower_with_defines`], plus per-entry library ownership for
/// dynamic-linking (see `analyze_with_owners`): entries whose owner
/// differs from the main project become runtime imports downstream.
/// Empty owners behave exactly as before.
pub fn lower_with_owners(
    unit: &CompilationUnit,
    require_main: bool,
    defines: &[String],
    owners: &std::collections::HashMap<
        crate::native::semantic::TypeKey,
        String,
    >,
    preloaded: &semantic::Preloaded,
) -> Result<Program, LowerError> {
    // Fresh run, fresh call-index cache (keyed by table address).
    lower::clear_fn_index_cache();
    let table = semantic::analyze_with_preloaded(unit, owners, preloaded)
        .map_err(LowerError::Semantic)?;

    let members = unit.all_members();
    let mut classes: Vec<&ClassDecl> = Vec::new();
    for member in &members {
        if let Member::Class(class) = member {
            classes.push(class);
        }
    }

    // Flatten nested classes depth-first for FUNCTION GENERATION and
    // the function index (`Ctx.classes`): nested methods (e.g.
    // `Task.ContingentProperties.SetEvent`) lower to functions keyed
    // by simple name, matching call-site resolution. Vtable rows,
    // slot assignment, and ctor pre-index stay top-level-only
    // (nested virtual dispatch is a known backend gap, unchanged).
    fn flatten<'a>(decl: &'a ClassDecl, out: &mut Vec<&'a ClassDecl>) {
        out.push(decl);
        for m in &decl.members {
            if let ClassMember::Nested(Member::Class(nc)) = m {
                flatten(nc, out);
            }
        }
    }
    let mut all_classes: Vec<&ClassDecl> = Vec::new();
    for c in &classes {
        flatten(c, &mut all_classes);
    }

    // Qualified AST path per class declaration (pointer-keyed): method
    // anchoring (`current_entry`) must resolve same-named nested classes
    // to the entry under lowering, not table first-match (25
    // `Enumerator`s share names and signatures; the wrong anchor types
    // `_enumerator` as `IAsyncEnumerator` instead of the sibling
    // nested struct).
    fn record_class<'a>(
        decl: &'a ClassDecl,
        prefix: &str,
        map: &mut std::collections::HashMap<*const ClassDecl, String>,
    ) {
        let path = if prefix.is_empty() {
            decl.name.clone()
        } else {
            format!("{prefix}.{}", decl.name)
        };
        map.insert(decl as *const ClassDecl, path.clone());
        for m in &decl.members {
            if let ClassMember::Nested(Member::Class(nc)) = m {
                record_class(nc, &path, map);
            }
        }
    }
    let mut class_paths: std::collections::HashMap<*const ClassDecl, String> =
        std::collections::HashMap::new();
    for ns in &unit.namespaces {
        for m in &ns.members {
            if let Member::Class(c) = m {
                record_class(c, &ns.name, &mut class_paths);
            }
        }
    }
    for m in &unit.members {
        if let Member::Class(c) = m {
            record_class(c, "", &mut class_paths);
        }
    }

    // ---- vtable slots are per-class positional (recorded on each
    // `MethodInfo` at semantic build: base-chain order first, so
    // derived rows share their bases' prefix). No global numbering is
    // computed here anymore — global first-seen shifts with table
    // content/order and can never agree cross-image. Row construction
    // below and `canonical_slot` at call sites both read the recorded
    // positions.

    let mut functions: Vec<IrFunction> = Vec::new();
    let mut fn_index: std::collections::BTreeMap<String, usize> = Default::default();
    let mut vtables: Vec<(String, Vec<usize>)> = Vec::new();
    let mut strings = StringArena::new();
    // Cross-image (dependency) imports registered during lowering:
    // `(lib hint, symbol)` → dep slot, appended after host imports in
    // `Program::imports` at construction below. Shared across all
    // method bodies (first-use order = deterministic AST order); empty
    // in static builds.
    let dep_imports: std::cell::RefCell<
        std::collections::BTreeMap<(String, String), usize>,
    > = Default::default();
    let import_index: std::collections::BTreeMap<(String, String), usize> = table
        .imports
        .iter()
        .enumerate()
        .map(|(i, imp)| ((imp.class.clone(), imp.method.clone()), i))
        .collect();

    // Nearest implementation of `method` visible from `class_name`.
    //
    // Host imports first (stable indices), then cross-image imports in
    // first-use order (BTreeMap iteration = sorted = deterministic
    // across builds from the same sources).
    fn dep_extended_imports(
        table: &TypeTable,
        dep_imports: &std::cell::RefCell<
            std::collections::BTreeMap<(String, String), usize>,
        >,
    ) -> Vec<(String, String)> {
        let mut all: Vec<(String, String)> = table
            .imports
            .iter()
            .map(|i| (i.library.clone(), i.entry_point.clone()))
            .collect();
        let dep = dep_imports.borrow();
        let mut ordered: Vec<((String, String), usize)> =
            dep.iter().map(|(k, v)| (k.clone(), *v)).collect();
        ordered.sort_by_key(|(_, v)| *v);
        all.extend(ordered.into_iter().map(|(k, _)| k));
        all
    }

    // Only actual functions count: `ClassInfo.methods` lists inherited
    // names too, so the search keeps walking until it reaches the class
    // that truly declares the body. Overloads of the same name dispatch
    // to the FIRST declared body, matching the semantic layer's
    // one-`MethodInfo`-per-name rule.
    fn impl_index(
        classes: &[&ClassDecl],
        table: &TypeTable,
        class_name: &str,
        method: &str,
        fn_index: &std::collections::BTreeMap<String, usize>,
    ) -> Option<usize> {
        impl_index_seen(classes, table, class_name, method, fn_index, &mut Vec::new())
    }
    // Cycle-guarded base walk: spelling-variant base cycles (blocked
    // from `resolve_chain` but reachable here through per-call string
    // spellings) terminate as holes instead of overflowing the stack
    // mid-build.
    fn impl_index_seen(
        classes: &[&ClassDecl],
        table: &TypeTable,
        class_name: &str,
        method: &str,
        fn_index: &std::collections::BTreeMap<String, usize>,
        seen: &mut Vec<String>,
    ) -> Option<usize> {
        if class_name == "@start" {
            return None;
        }
        if seen.contains(&class_name.to_owned()) {
            return None;
        }
        seen.push(class_name.to_owned());
        if let Some(class) = classes.iter().find(|c| c.name == class_name) {
            if let Some(decl) = class.members.iter().find_map(|m| match m {
                ClassMember::Method(md) if md.name == method && md.body.is_some() => Some(md),
                _ => None,
            })
            && let Some(&idx) =
                fn_index.get(&format!("{}::{}#{}", class_name, method, decl.params.len()))
            {
                return Some(idx);
            }
        }
        let info = table.class(class_name)?;
        info.bases.iter().find_map(|b| {
            impl_index_seen(classes, table, b, method, fn_index, seen)
        })
    }

    // Precomputed constructor function indices (declaration-order numbering
    // with @start at 0) so `: base(args)` chains resolve even when a derived
    // class is declared before its base. Must mirror the lowering skip rules
    // below (no body, destructors, >64 params).
    let mut ctor_index: std::collections::BTreeMap<(String, usize), usize> = Default::default();
    {
        let mut n = 0usize;
        for class in &all_classes {
            if table.class(&class.name).is_none() {
                continue;
            }
            for m in &class.members {
                if let ClassMember::Method(method) = m {
                    if method.body.is_none() || method.name.starts_with('~') {
                        continue;
                    }
                    // Mirror the lowering skip rules (no body,
                    // destructors, >64 params).
                    if method.params.len() > 64 {
                        continue;
                    }
                    n += 1;
                    if method.ret_type.is_empty() && method.name == class.name {
                        ctor_index.insert((class.name.clone(), method.params.len()), n);
                    }
                }
            }
        }
    }

    // ---- static field storage.
    // One slot per (declaring class, field name), in class declaration order.
    // Inherited statics (declared on a base) are deduped by declaring class,
    // so `Derived.N` shares the base's single storage slot.
    // `IrStatic.class` is the qualified class path (not the bare declaring
    // name): the backend exports each slot as a `<path>::<field>` OBJECT
    // symbol, and importers must resolve the same string. Slot identity
    // (`static_index` keys) stays bare — unchanged behavior for locals.
    let mut statics: Vec<IrStatic> = Vec::new();
    let mut static_index: std::collections::BTreeMap<(String, String), u32> =
        std::collections::BTreeMap::new();
    for info in &table.classes {
        for fi in &info.static_fields {
            let key = (fi.declaring_class.clone(), fi.name.clone());
            if !static_index.contains_key(&key) {
                static_index.insert(key.clone(), statics.len() as u32);
                statics.push(IrStatic {
                    name: fi.name.clone(),
                    class: info.key.path.clone(),
                    type_name: fi.type_name.clone(),
                });
            }
        }
    }

    // ---- methods

/// Whether an entry method takes exactly one `string[]` parameter
/// (`Main(string[] args)` in any namespace spelling): @start builds
/// and passes the argv array only for those.
fn takes_argv(method: &crate::models::MethodDecl) -> bool {
    method.params.len() == 1
        && matches!(
            method.params[0].type_name.trim_start_matches("System.").trim_start_matches("global::"),
            "string[]" | "String[]"
        )
}
    // Entry candidates carry whether they take `string[]`: @start
    // builds the argv array only for those (`Main(string[] args)`),
    //paramless entry points keep the old receiver-only call.
    let mut main_upper: Option<(String, usize, bool)> = None;
    let mut main_lower: Option<(String, usize, bool)> = None;

    for class in &all_classes {
        let Some(_info) = table.class(&class.name) else {
            continue;
        };
        for m in &class.members {
            let ClassMember::Method(method) = m else {
                continue;
            };
            // Frame budget: call cleanup spills to imm32 past 120
            // bytes, so methods up to 64 params (64-element vector
            // factories) generate normally. Beyond that, calls fail
            // loudly at function-index lookup instead of miscompiling.
            if method.params.len() > 64 {
                continue;
            }
            // Bodyless declarations are import stubs — never functions.
            let Some(body) = &method.body else {
                continue;
            };
            // Destructors (~Class) are not wired into the runtime yet; skip
            // them (they are not called explicitly in the fixture).
            if method.name.starts_with('~') {
                continue;
            }
            // Function identity includes arity: overloaded methods (and
            // constructors) of the same name must map to distinct functions.
            let fname = format!("{}::{}#{}", class.name, method.name, method.params.len());

            let mut f = IrFunction::new(
                &fname,
                // Static methods take no receiver: the caller passes
                // only declared args, so vreg 0 must be the first
                // param, not a phantom `this`. Declaring `This` here
                // shifted every static arg into garbage (params read
                // zero).
                if method.is_static {
                    ReceiverKind::None
                } else {
                    ReceiverKind::This
                },
                method.params.len(),
            );
            f.is_async = method.is_async;
            let mut lb = lower::Body::new(&mut f, &mut strings);
            let ctx = Ctx {
                table: &table,
                current_class: &class.name,
                current_info: current_entry(
                    &table,
                    &class.name,
                    method,
                    class_paths
                        .get(&(*class as *const ClassDecl))
                        .map(|s| s.as_str()),
                ),
                current_path: class_paths
                    .get(&(*class as *const ClassDecl))
                    .map(|s| s.as_str()),
                classes: &all_classes,
                import_index: &import_index,
                dep_imports: &dep_imports,
                host_import_count: table.imports.len(),
                static_index: &static_index,
                defines,
                method_vars: &method.generic_params,
            };

            lb.scope_mut().push();
            // Bindings were seeded: vreg 0 = this, 1..=n = params.
            // `this` binds only in instance methods WITHOUT an `@this`
            // parameter: `@this` declares an ordinary parameter which
            // shadows the receiver keyword per C# scoping (static
            // methods, the common case, never have a receiver). Bare
            // `this` still resolves via the keyword fallbacks in
            // eval/static_type_of when unbound.
            let has_this_param = method.params.iter().any(|p| p.name == "this");
            if !method.is_static && !has_this_param {
                lb.scope_mut()
                    .declare("this", &class.name, Access::Default)
                    .map_err(LowerError::UnsupportedConstruct)?;
            }
            for p in &method.params {
                lb.scope_mut()
                    .declare(&p.name, &p.type_name, Access::Default)
                    .map_err(LowerError::UnsupportedConstruct)?;
            }
            // Primary-constructor params are visible in every member
            // body; their values bind at `new` (field initializers) and
            // placeholder here (see `bind_zero_placeholder`).
            for pname in &class.primary_ctor_params {
                lb.bind_zero_placeholder(pname);
            }

            // Constructor chain: `: base(args)` or `: this(args)`. A
            // constructor without an explicit chain implicitly calls the
            // first base's zero-arg constructor, if one exists (C#
            // semantics).
            // Out-vars declared in the chain args (`: base(M(p, out var
            // u, ...))`) are spliced as leading Lets in the body — but
            // the base call evaluates FIRST. Pre-declare exactly those
            // names (matched against the leading Lets, with their
            // declared types) so base-arg evaluation finds them; genuine
            // typos elsewhere still error. Later, the body Lets degrade
            // to same-valued assignments.
            // NOTE: only explicit `: base(...)`/`: this(...)` chains
            // carry args here (the implicit zero-arg synthesis below has
            // none), so this runs solely for declared chains.
            if let Some((_, chain_args)) = method.base_call.as_ref() {
                if let Some(body) = method.body.as_ref() {
                    let mut leading: Vec<(String, String)> = Vec::new();
                    for stmt in &body.stmts {
                        match stmt {
                            crate::models::Stmt::Let {
                                name, type_name, ..
                            } => leading.push((name.clone(), type_name.clone())),
                            _ => break,
                        }
                    }
                    // NOTE: base args are evaluated below (`eval` per arg);
                    // this pre-pass only declares, never evaluates.
                    for a in chain_args.iter() {
                        if let crate::models::Expr::Ident { name, .. } = a {
                            if lb.scopes.lookup(name).is_none() {
                                if let Some((_, ty)) = leading.iter().find(|(n, _)| n == name)
                                {
                                    if let Ok(slot) = lb.scopes.declare(
                                        name,
                                        ty,
                                        crate::models::Access::Default,
                                    ) {
                                        while lb.bindings.len() <= slot {
                                            let v = lb.fresh();
                                            lb.bindings.push(v);
                                        }
                                        let z = lb.const_i64(0);
                                        lb.push_raw(Instr::Mov {
                                            dst: lb.bindings[slot],
                                            a: z,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let chain = method.base_call.clone().or_else(|| {
                if method.name == class.name
                    && method.ret_type.is_empty()
                    && let Some(base) = class.bases.first()
                    && classes.iter().any(|c| {
                        c.name == *base
                            && c.members.iter().any(|m| {
                                matches!(
                                    m,
                                    ClassMember::Method(md)
                                        if md.name == c.name
                                            && md.ret_type.is_empty()
                                            && md.params.is_empty()
                                )
                            })
                    })
                {
                    Some(("base".to_string(), Vec::new()))
                } else {
                    None
                }
            });
            if let Some((target, args)) = chain {
                if target == "base" {
                    // Find the base class and call its constructor.
                    let base_name = class.bases.first().map(|s| s.as_str()).unwrap_or("");
                    if !base_name.is_empty() {
                        let base_ctors: Vec<_> = classes
                            .iter()
                            .find(|c| c.name == base_name)
                            .map(|bc| {
                                bc.members
                                    .iter()
                                    .filter_map(|m| match m {
                                        ClassMember::Method(m)
                                            if m.name == bc.name && m.ret_type.is_empty() =>
                                        {
                                            Some(m)
                                        }
                                        _ => None,
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                        if !base_ctors.is_empty() {
                            let matching = base_ctors
                                .iter()
                                .find(|m| m.params.len() == args.len())
                                .copied();
                            if let Some(ctor) = matching {
                                let mut argv = vec![VReg(0)]; // this
                                for a in &args {
                                    // Bare method names (method groups)
                                    // eval to zero via the
                                    // undeclared-variable fallback in
                                    // `eval` (delegate placeholder).
                                    argv.push(lower::eval(a, &mut lb, &ctx)?);
                                }
                                let idx = ctor_index
                                    .get(&(base_name.to_string(), ctor.params.len()))
                                    .copied()
                                    .unwrap_or(0);
                                lb.call_fn(idx, argv);
                            }
                        }
                    }
                }
                // `this(args)` is not supported yet.
            }

            lb.block(body, &ctx).map_err(|e| match e {
                // Locating lowering failures across 1800-file units:
                // every construct/access error names its method.
                LowerError::UnsupportedConstruct(msg) => {
                    LowerError::UnsupportedConstruct(format!(
                        "{msg} [in {}.{}]",
                        class.name, method.name
                    ))
                }
                LowerError::Access {
                    code,
                    message,
                    help,
                } => LowerError::Access {
                    code,
                    message: format!(
                        "{message} [in {}.{}]",
                        class.name, method.name
                    ),
                    help,
                },
                other => other,
            })?;

            // Implicit `return 0;` after releasing heap locals.
            lb.release_scope_locals();
            lb.seal_with_return_zero();

            // +1 reserves slot 0 for the synthetic @start that is
            // prepended below; every index recorded here must match the
            // FINAL Program::functions layout.
            let idx = functions.len() + 1;
            functions.push(lb.finish_with(Terminator::Ret(None)));
            fn_index.insert(fname, idx);

            // Entry-point candidates: C#-style `Main` preferred over the
            // legacy RubyC-flavored lowercase `main`.
            match method.name.as_str() {
                "Main" => {
                    if main_upper.is_some() {
                        return Err(LowerError::UnsupportedConstruct(crate::tf!(
                            "rubyc.native.ir.entry.ambiguous"
                        )));
                    }
                    main_upper = Some((class.name.clone(), idx, takes_argv(method)));
                }
                "main" => {
                    main_lower.get_or_insert((class.name.clone(), idx, takes_argv(method)));
                }
                _ => {}
            }
        }

        // Vtable row for this class: nearest implementing function index
        // per slot; `usize::MAX` marks abstract holes. Slots are the
        // entry's own recorded positions (per-class positional
        // numbering: base-chain order first, so derived rows share
        // their bases' prefix — stable across builds from the same
        // sources, unlike global first-seen). Each distinct position
        // resolves its nearest implementation independently, so
        // overloads/hides sharing a name each land correctly.
        let entry_slots: Vec<(usize, String)> = table
            .class(&class.name)
            .map(|e| {
                e.methods
                    .iter()
                    .map(|m| (m.vtable_slot, m.name.clone()))
                    .collect()
            })
            .unwrap_or_default();
        let max_slot = entry_slots.iter().map(|(s, _)| *s).max().unwrap_or(0);
        let mut row = vec![usize::MAX; max_slot + 1];
        for (slot, mname) in entry_slots {
            // Preserve abstract holes: only fill when an implementation
            // actually resolves (a slot with no body anywhere stays MAX
            // instead of collapsing to an unrelated index 0 function).
            if let Some(idx) = impl_index(&all_classes, &table, &class.name, &mname, &fn_index) {
                row[slot] = idx;
            }
        }
        vtables.push((
            class_paths
                .get(&(*class as *const ClassDecl))
                .cloned()
                .unwrap_or_else(|| class.name.clone()),
            row,
        ));
    }

    // ---- @start entry (executables only)
    // `Main` (C#-style) wins over legacy lowercase `main`.
    let main_fn = main_upper.clone().or(main_lower.clone());
    let Some((entry_class, main_idx, main_argv)) = main_fn else {
        if require_main {
            return Err(LowerError::NoMainMethod);
        }
        // No @start in shared libraries: strip the reserved slot from
        // every index (vtable rows and exports alike).
        let vtables = vtables
            .into_iter()
            .map(|(name, row)| {
                (
                    name,
                    row.into_iter()
                        .map(|s| if s == usize::MAX { s } else { s - 1 })
                        .collect(),
                )
            })
            .collect();
        // Shared libraries run static field initializers from a
        // synthetic `@init` (same loop as `@start` below, returning
        // instead of exiting): the image writer registers it in
        // `.init_array` so `dlopen` runs it before any symbol is
        // used. Without this, cross-image static reads see zeroed
        // storage. Emitted whenever statics exist.
        let init = if statics.is_empty() {
            None
        } else {
            let init_ctx = Ctx {
                table: &table,
                current_class: "",
                current_info: None,
                current_path: None,
                classes: &all_classes,
                import_index: &import_index,
                dep_imports: &dep_imports,
                host_import_count: table.imports.len(),
                static_index: &static_index,
                defines,
                method_vars: &[],
            };
            let mut inf = IrFunction::new("@init", ReceiverKind::None, 0);
            let mut ib = lower::Body::new(&mut inf, &mut strings);
            ib.scope_mut().push();
            for class in &all_classes {
                for m in &class.members {
                    let ClassMember::Field(fd) = m else {
                        continue;
                    };
                    if !fd.is_static {
                        continue;
                    }
                    let Some(&idx) =
                        static_index.get(&(class.name.clone(), fd.name.clone()))
                    else {
                        continue;
                    };
                    // Same declaring-class viewer rule as `@start`.
                    let field_ctx = Ctx {
                        current_class: &class.name,
                        ..init_ctx
                    };
                    // Bindings declared while parsing the initializer
                    // (`out` vars) lower first so the init sees them.
                    for pre in &fd.init_pre {
                        ib.statement(pre, &field_ctx)?;
                    }
                    let v = lower::eval(&fd.init, &mut ib, &field_ctx)?;
                    ib.store_static(idx, v);
                }
            }
            let init_idx = functions.len();
            functions.push(ib.finish_with(Terminator::Ret(None)));
            Some(init_idx)
        };
        return Ok(Program {
            functions,
            statics: statics.clone(),
            vtables,
            imports: dep_extended_imports(&table, &dep_imports),
            // Shared libraries have no @start: fn_index is exe-numbered,
            // so exported indices shift down by one.
            exports: {
                let mut e = exports_from(&table, &classes, &fn_index, -1)?;
                e.extend(qualified_exports_from(&table, &fn_index, -1));
                e
            },
            strings: strings.bytes,
            heap_size: 64 * 1024,
            names: Vec::new(),
            entry: None,
            init,
        });
    };
    let Some(entry_decl) = classes.iter().find(|c| c.name == entry_class) else {
        return Err(LowerError::NoMainMethod);
    };
    let Some(info) = table.class(&entry_class) else {
        return Err(LowerError::NoMainMethod);
    };

    let object_bytes = (info.fields.len() + 2) as u64 * 8;
    let start_ctx = Ctx {
        table: &table,
        current_class: &entry_class,
        current_info: None,
        current_path: None,
        classes: &classes,
        import_index: &import_index,
        dep_imports: &dep_imports,
        host_import_count: table.imports.len(),
        static_index: &static_index,
        defines,
        method_vars: &[],
    };

    let mut sf = IrFunction::new("@start", ReceiverKind::None, 0);
    let mut sl = lower::Body::new(&mut sf, &mut strings);
    sl.scope_mut().push();

    // Source-level assembly blocks execute in declaration order before the
    // managed entry object is initialized. They share the same assembler as
    // standalone `.asm` inputs and flow through Instr::Raw to every backend.
    for source in &unit.assembly_blocks {
        let bytes = crate::parser::assembly::assemble(source)
            .map_err(|error| LowerError::UnsupportedConstruct(format!("assembly block: {error}")))?;
        sl.raw(bytes);
    }

    // Static field initializers run once, before the entry object exists.
    // Each static is initialized at its declaring class (a static appears in
    // exactly one class's own member list), so no initializer runs twice.
    for class in &all_classes {
        for m in &class.members {
            let ClassMember::Field(fd) = m else {
                continue;
            };
            if !fd.is_static {
                continue;
            }
            let Some(&idx) = static_index.get(&(class.name.clone(), fd.name.clone())) else {
                continue;
            };
            // The initializer is the declaring class's own code (a
            // private member call inside it is legal): evaluate with
            // that class as viewer, not the entry class.
            let field_ctx = Ctx {
                current_class: &class.name,
                ..start_ctx
            };
            // Bindings declared while parsing the initializer
            // (`out` vars) lower first so the init sees them.
            for pre in &fd.init_pre {
                sl.statement(pre, &field_ctx)?;
            }
            let v = lower::eval(&fd.init, &mut sl, &field_ctx)?;
            sl.store_static(idx, v);
        }
    }

    let obj = sl.fresh();
    sl.alloc(obj, object_bytes);
    let vi = vtables
        .iter()
        .position(|(n, _)| n == &entry_class || n.ends_with(&format!(".{entry_class}")))
        .unwrap_or(0);
    let vt = sl.fresh();
    sl.vtable_addr(vt, vi as u32);
    sl.push_raw(Instr::StoreField { base: obj, off: 0, src: vt });

    // Field initializers run against the fresh object. Statics are handled
    // above (they have no object slot).
    for fd in entry_decl.members.iter().filter_map(|m| match m {
        ClassMember::Field(fd) if !fd.is_static => Some(fd),
        _ => None,
    }) {
        let v = lower::eval(&fd.init, &mut sl, &start_ctx)?;
        let off = (info.field(&fd.name).map(|fi| fi.offset_slots).unwrap_or(1) * 8) as u32;
        sl.push_raw(Instr::StoreField { base: obj, off, src: v });
    }

    let main_ret = if main_argv {
        // `Main(string[] args)`: materialize process argv (kernel stack
        // via the backend's startup slot) and pass it after the receiver,
        // like any other second parameter.
        let argv = sl.fresh();
        sl.push_raw(Instr::BuildArgv { dst: argv });
        sl.call_fn(main_idx, vec![obj, argv])
    } else {
        sl.call_fn(main_idx, vec![obj])
    };
    // @start has no caller: terminate with exit_group, never a ret.
    // An int-returning entry point becomes the process exit code.
    sl.exit_with(main_ret);

    functions.insert(0, sl.finish_with(Terminator::Exit));

    Ok(Program {
        functions,
        statics,
        vtables,
        imports: dep_extended_imports(&table, &dep_imports),
        exports: {
                let mut e = exports_from(&table, &classes, &fn_index, 0)?;
                e.extend(qualified_exports_from(&table, &fn_index, 0));
                e
            },
        strings: strings.bytes,
        heap_size: 64 * 1024,
        names: Vec::new(),
        entry: Some(0), // @start is always function 0 for executables.
        init: None, // Executables run initializers from @start instead.
    })
}

/// Collect `(symbol, function index, arity)` for public body-bearing
/// methods. Namespaces and classes are compiler-side organization; the
/// exposed ABI surface is bare symbols.
fn exports_from(
    table: &TypeTable,
    classes: &[&ClassDecl],
    fn_index: &std::collections::BTreeMap<String, usize>,
    shift: isize,
) -> std::result::Result<Vec<(String, usize, usize)>, LowerError> {
    let mut out: Vec<(String, usize, usize)> = Vec::new();
    for class in classes {
        let Some(own) = table
            .classes
            .iter()
            .find(|c| c.name == class.name)
            .map(|c| c.own_members.clone())
        else {
            continue;
        };
        for (name, is_field, access) in own {
            if is_field
                || access != Access::Public
                || name == "main"
                || name == "Main"
                || name.starts_with('~')
                // Overrides share the base symbol; never re-export.
                || method_is_override(class, &name)
            {
                continue;
            }
            if let Some(arity) = method_arity(class, &name)
                && method_has_body(class, &name)
                && let Some(&idx) = fn_index.get(&format!("{}::{}#{}", class.name, name, arity))
            {
                // Duplicate symbols (overloads) are tolerated — keep the
                // first export and skip subsequent ones.
                if out.iter().any(|(existing, _, _)| *existing == name) {
                    continue;
                }
                // Executables prepend @start, shifting every index.
                out.push((name, (idx as isize + shift) as usize, arity));
            }
        }
    }
    Ok(out)
}

/// Qualified exports for dynamic linking: `{key.path}::{method}#{arity}`
/// per table entry with a lowered body. Unlike the bare names above,
/// these are unambiguous across libraries (same-named types in different
/// namespaces/libs get distinct symbols) and computable identically on
/// both sides from the declaring entry alone (no first-match): the
/// importer derives them from `.rci` fragments, the exporter from its
/// own table built from the same sources. Additive — bare exports are
/// untouched for C hosts. Entries without a lowered body in this image
/// (future dynamic builds skip dep bodies) are skipped automatically
/// via the `fn_index` miss, so the same code serves static and dynamic
/// builds. Same-name+arity overloads share one symbol (first wins),
/// exactly like the bare names today — cross-lib overload-heavy cases
/// stay static until verified.
fn qualified_exports_from(
    table: &TypeTable,
    fn_index: &std::collections::BTreeMap<String, usize>,
    shift: isize,
) -> Vec<(String, usize, usize)> {
    let mut out: Vec<(String, usize, usize)> = Vec::new();
    for entry in &table.classes {
        for m in &entry.methods {
            if m.access != Access::Public
                || !m.has_body
                || m.name == "main"
                || m.name == "Main"
                || m.name.starts_with('~')
            {
                continue;
            }
            let arity = m.param_types.len();
            let sym = format!("{}::{}#{}", entry.key.path, m.name, arity);
            if out.iter().any(|(existing, _, _)| *existing == sym) {
                continue;
            }
            if let Some(&idx) =
                fn_index.get(&format!("{}::{}#{}", entry.name, m.name, arity))
            {
                out.push((sym, (idx as isize + shift) as usize, arity));
            }
        }
    }
    out
}

fn method_is_override(class: &ClassDecl, method: &str) -> bool {
    class.members.iter().any(|m| match m {
        ClassMember::Method(md) => md.name == method && md.is_override,
        _ => false,
    })
}

fn method_has_body(class: &ClassDecl, method: &str) -> bool {
    class.members.iter().any(|m| match m {
        ClassMember::Method(md) => md.name == method && md.body.is_some(),
        _ => false,
    })
}

fn method_arity(class: &ClassDecl, method: &str) -> Option<usize> {
    class.members.iter().find_map(|m| match m {
        ClassMember::Method(md) if md.name == method && md.body.is_some() => Some(md.params.len()),
        _ => None,
    })
}
