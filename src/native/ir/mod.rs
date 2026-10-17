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
    utf16_index: std::collections::BTreeMap<String, (u32, u32)>,
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

    /// `(offset, char_len)` of the literal encoded as UTF-16LE,
    /// deduplicated separately (user strings are UTF-16; the
    /// `literal` pool above stays UTF-8 for diagnostics). Rust's
    /// `encode_utf16` handles multi-byte sequences and surrogates,
    /// so `char_len` matches .NET `String.Length` semantics.
    pub fn utf16_literal(&mut self, s: &str) -> (u32, u32) {
        if let Some(hit) = self.utf16_index.get(s) {
            return *hit;
        }
        let off = self.bytes.len() as u32;
        let units: Vec<u16> = s.encode_utf16().collect();
        for u in &units {
            self.bytes.extend_from_slice(&u.to_le_bytes());
        }
        let hit = (off, units.len() as u32);
        self.utf16_index.insert(s.to_owned(), hit);
        hit
    }
}

use crate::models::{Access, ClassDecl, ClassMember, CompilationUnit, Expr, Member, MethodDecl};
use crate::models::{Block as AstBlock, Param as AstParam, Stmt as AstStmt};
use crate::models::visitor::Visitor;
use crate::native::semantic::{self, ClassInfo, MethodInfo, SemanticError, TypeTable};

/// A static local function collected for real emission
/// (`EnsureInitialized` inside a property getter lowers to a
/// mangled private function instead of the legacy zero
/// placeholder). Shared by the pre-scan (emission + table) and the
/// direct-call index (offset), so both agree on count and order.
#[derive(Debug, Clone)]
pub(crate) struct LocalFnDecl {
    pub class_name: String,
    pub method_name: String,
    pub local_name: String,
    pub params: Vec<AstParam>,
    pub ret_type: String,
    pub body: AstBlock,
    /// Enclosing method's generic params (locals inherit them).
    pub method_vars: Vec<String>,
}

/// Collect static local functions in deterministic order (classes
/// → methods → source order, descending into nested blocks).
/// Instance, async, generic, bodiless, or fancy-param locals stay
/// on the legacy placeholder path (not collected).
pub(crate) fn collect_static_locals(classes: &[&ClassDecl]) -> Vec<LocalFnDecl> {
    fn walk_block(
        out: &mut Vec<LocalFnDecl>,
        class_name: &str,
        method_name: &str,
        method_vars: &[String],
        block: &AstBlock,
    ) {
        for stmt in &block.stmts {
            match stmt {
                AstStmt::LocalFunction {
                    name,
                    params,
                    ret_type,
                    body,
                    is_static,
                    ..
                } => {
                    if *is_static {
                        out.push(LocalFnDecl {
                            class_name: class_name.to_owned(),
                            method_name: method_name.to_owned(),
                            local_name: name.clone(),
                            params: params.clone(),
                            ret_type: ret_type.clone(),
                            body: body.clone(),
                            method_vars: method_vars.to_owned(),
                        });
                    }
                    walk_block(out, class_name, method_name, method_vars, body);
                }
                AstStmt::Block { body, .. } => {
                    walk_block(out, class_name, method_name, method_vars, body)
                }
                AstStmt::If {
                    then_block,
                    else_block,
                    ..
                } => {
                    walk_block(out, class_name, method_name, method_vars, then_block);
                    if let Some(b) = else_block {
                        walk_block(out, class_name, method_name, method_vars, b);
                    }
                }
                AstStmt::While { body, .. } | AstStmt::DoWhile { body, .. } => {
                    walk_block(out, class_name, method_name, method_vars, body)
                }
                AstStmt::For { body, .. } => {
                    walk_block(out, class_name, method_name, method_vars, body)
                }
                AstStmt::Switch {
                    arms, default_body, ..
                } => {
                    for arm in arms {
                        walk_block(out, class_name, method_name, method_vars, &arm.body);
                    }
                    if let Some(b) = default_body {
                        walk_block(out, class_name, method_name, method_vars, b);
                    }
                }
                AstStmt::Try {
                    body,
                    catches,
                    finally_block,
                    ..
                } => {
                    walk_block(out, class_name, method_name, method_vars, body);
                    for c in catches {
                        walk_block(out, class_name, method_name, method_vars, &c.body);
                    }
                    if let Some(b) = finally_block {
                        walk_block(out, class_name, method_name, method_vars, b);
                    }
                }
                _ => {}
            }
        }
    }

    let mut out = Vec::new();
    for class in classes {
        for m in &class.members {
            if let ClassMember::Method(method) = m {
                if let Some(body) = &method.body {
                    walk_block(
                        &mut out,
                        &class.name,
                        &method.name,
                        &method.generic_params,
                        body,
                    );
                }
            }
            // Property/indexer accessors hold bodies too (`Out`'s
            // getter hides `EnsureInitialized`); locals there must
            // also emit or their calls stay placeholders.
            if let ClassMember::Property(prop) = m {
                if let Some(body) = &prop.get {
                    walk_block(&mut out, &class.name, &prop.name, &[], body);
                }
                if let Some(body) = &prop.set {
                    walk_block(&mut out, &class.name, &prop.name, &[], body);
                }
            }
        }
    }
    out
}

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
#[derive(Clone)]
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
    /// Static local functions visible in the method under lowering:
    /// `(enclosing class, enclosing method-or-property, name, arity,
    /// function index, return type)`, collected by the pre-scan.
    /// Bare calls check this before method lookup. Empty outside
    /// method bodies (field initializers, tests).
    pub local_fns: &'a [(String, String, String, usize, usize, String)],
    /// Enclosing method (or property, for inlined accessors) name for
    /// local-function scoping above. `None` outside method bodies.
    /// Owned: property-inline contexts rebuild it per read.
    pub current_method: Option<String>,
    /// Declared return type of the method/property under lowering
    /// (`None` outside bodies): `return` values coerce to it
    /// (arrays/strings pack into span-typed returns). Owned like
    /// `current_method`; property-inline contexts set the
    /// property's type.
    pub current_ret: Option<String>,
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
        // Base-slot redemption for `protected` (mirrors
        // `check_method_access_virtual`): an inaccessible override is
        // redeemed by an accessible same-kind declaration on a base of
        // the receiver (`task.Prop` in `Task` binds C# through the
        // static type `Task`, whose own `protected` declaration the
        // viewer accesses legally — denying on the derived override's
        // account rejects shipped runtime code like
        // `Task.AnyTaskRequiresNotifyDebuggerOfWaitCompletion`).
        // Only `protected` redeems; `private` stays denied. Dispatch
        // is unaffected (still virtual).
        if matches!(access, Access::Protected) {
            let mut queue: Vec<String> = vec![owner_type.to_owned()];
            let mut seen: Vec<String> = Vec::new();
            while let Some(cur) = queue.pop() {
                if seen.contains(&cur) {
                    continue;
                }
                seen.push(cur.clone());
                let next = if is_field {
                    self.table.declaring_class_of_field(&cur, member)
                } else {
                    self.table.declaring_class_of_method(&cur, member)
                };
                if let Some((decl2, access2)) = next {
                    if decl2 != declaring
                        && semantic::visible_from(
                            self.table,
                            access2,
                            &decl2,
                            self.current_class,
                        )
                    {
                        return Ok(());
                    }
                    if let Some(info) = self.table.class(&decl2) {
                        queue.extend(info.bases.iter().cloned());
                    }
                } else if let Some(info) = self.table.class(&cur) {
                    queue.extend(info.bases.iter().cloned());
                }
            }
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

/// Collect every class referenced as a static receiver inside one
/// static field initializer (`CultureInfo.InvariantCulture`,
/// `TimeSpan.FromHours(..)`, `HResults.COR_E_X`). Conservative:
/// name collisions with locals add harmless extra edges (ordering
/// only); a missed edge merely keeps today's declaration order for
/// that pair (fail-safe direction — never wrong code).
///
/// `new X(...)` in an initializer runs X's instance constructor at
/// init time, and that body may read another class's static
/// (`ThreadPoolInstance = new PortableThreadPool()` running the
/// ctor's `NativeRuntimeEventSource.Log.IsEnabled()` before
/// `Log` initialized — null receiver, segfault). The `New` arm
/// therefore scans X's instance ctor bodies transitively,
/// including `base(...)` chains, so those reads constrain order
/// too.
///
/// One level more: a ctor may call a same-class helper whose body
/// does the static read (`ThreadPoolWorkQueue()` → `this.
/// RefreshLoggingEnabled()` → `FrameworkEventSource.Log`). `Call`
/// (bare `RefreshLoggingEnabled()`), `this.`/`base.`-qualified
/// calls, and same-class property getters are therefore followed
/// into the owner's declaration as well (`owner` tracks whose body
/// is under scan, since transitive ctor scans leave `current`,
/// the initializer's class).
///
/// And the level that actually bites in framework init: an
/// initializer calls a STATIC method/property of ANOTHER class
/// whose body reads yet more statics (`HillClimbing.IsDisabled =
/// GetBooleanComPlusOrDotNetConfig(...)` → `TryParse` →
/// `NumberFormatInfo.InvariantInfo` getter →
/// `CultureInfo.InvariantCulture`, uninitialized at that point —
/// fault reading its fields). Ordering only the callee class is
/// not enough; the callee's transitive reads must constrain order
/// too. So `MethodCall`/`Member` arms also follow statically
/// resolvable cross-class receivers into those bodies. Instance
/// calls are followed the same way (over-approximation: the
/// receiver's runtime type is unknown, but any static read in the
/// body is a real ordering constraint if that body runs at init).
///
/// Termination and cost: a per-collector cycle `stack`, a depth
/// cap (64 — deeper bails out as a fail-safe miss), and a SHARED
/// memo (`InitScanMemo`, one entry per (class, member) for the
/// whole `init_class_order` call, so each body is walked once no
/// matter how many initializers reach it). Cached ref sets are
/// complete except the triggering class itself (see
/// `receiver_class`), which `edge_to` skips anyway.
struct InitScanMemo {
    refs: std::collections::HashMap<(String, String), Vec<String>>,
}

struct InitRefCollector<'a> {
    table: &'a TypeTable,
    classes: &'a [&'a ClassDecl],
    current: &'a str,
    owner: String,
    refs: Vec<String>,
    stack: Vec<(String, String)>,
    depth: usize,
    memo: &'a std::cell::RefCell<InitScanMemo>,
    /// Lexical value scopes for receiver resolution (method bodies
    /// only): parameter name → declared type. A bare `Ident`
    /// receiver that names a parameter (`cultureData.LongDates()`)
    /// resolves to the parameter's class — without this, every
    /// chain through a local/parameter is invisible and the edge
    /// is lost (`DateTimeFormat` sorting before
    /// `CalendarData.Invariant` was initialized → null read).
    /// Innermost scope last; pushed per scanned body.
    scopes: Vec<std::collections::HashMap<String, String>>,
}

impl<'a> InitRefCollector<'a> {
    /// Resolve a Member/MethodCall receiver chain (`A.B.C`, `A`) to
    /// its declaring class name, longest-prefix first (leading
    /// segments may be namespaces). `this`/`base`/locals resolve to
    /// nothing.
    fn receiver_class(&self, object: &Expr) -> Option<String> {
        fn flatten(e: &Expr, out: &mut Vec<String>) -> bool {
            match e {
                Expr::Member { object, name, .. } => {
                    if !flatten(object, out) {
                        return false;
                    }
                    out.push(name.clone());
                    true
                }
                Expr::Ident { name, .. } => {
                    out.push(name.clone());
                    true
                }
                _ => false,
            }
        }
        let table = self.table;
        let mut segs: Vec<String> = Vec::new();
        if !flatten(object, &mut segs) {
            return None;
        }
        // Single-identifier receivers: parameters/locals first (a
        // parameter shadows any same-named class), then owner fields
        // (`_cultureData` reads the field's declared type), then the
        // legacy class-name match below. Unresolvable spellings fall
        // through instead of returning `None` so e.g. a parameter
        // named like a namespace still resolves as a class.
        if segs.len() == 1 {
            for scope in self.scopes.iter().rev() {
                if let Some(ty) = scope.get(&segs[0])
                    && let Some(info) = lower::class_for_spelling(table, ty)
                {
                    return Some(info.name.clone());
                }
            }
            if let Some(owner_info) = table.class(&self.owner)
                && let Some(f) = owner_info.field(&segs[0])
                && let Some(info) = lower::class_for_spelling(table, &f.type_name)
            {
                return Some(info.name.clone());
            }
        }
        let mut path = segs.join(".");
        loop {
            if let Some(info) = lower::class_for_spelling(table, &path) {
                // No `current` exclusion here (it used to skip
                // self-references): memo replay across top-level
                // classes needs complete ref sets, and the
                // self-edge is dropped anyway by `edge_to`.
                return Some(info.name.clone());
            }
            match path.find('.') {
                Some(i) => path = path[i + 1..].to_owned(),
                None => return None,
            }
        }
    }

    fn record(&mut self, object: &Expr) {
        if let Some(name) = self.receiver_class(object) {
            if !self.refs.contains(&name) {
                self.refs.push(name);
            }
        }
    }

    /// Every class in the unit (transitively) derived from `owner`,
    /// by simple-name base matching (mirrors the base edges in
    /// `init_class_order`). Used to follow virtual dispatch into
    /// overrides (see `follow_owner_call`).
    fn derived_classes(&self, owner: &str) -> Vec<String> {
        let mut found: Vec<String> = Vec::new();
        let mut queue: Vec<String> = vec![owner.to_owned()];
        let mut seen: Vec<String> = vec![owner.to_owned()];
        while let Some(cur) = queue.pop() {
            let cur_simple = cur.rsplit('.').next().unwrap_or(&cur).to_owned();
            for c in self.classes.iter() {
                if seen.iter().any(|s| s == &c.name) {
                    continue;
                }
                let is_child = c.bases.iter().any(|b| {
                    let b = b.split('<').next().unwrap_or(b).trim();
                    b == cur || b.rsplit('.').next().unwrap_or(b) == cur_simple
                });
                if is_child {
                    seen.push(c.name.clone());
                    queue.push(c.name.clone());
                    found.push(c.name.clone());
                }
            }
        }
        found
    }

    /// Shared follow protocol for ctor/method/property bodies:
    /// cycle-guarded (`stack`), depth-capped, memoized across the
    /// whole ordering pass. Returns `Some(refs_len)` when the caller
    /// should walk bodies now (recording into `refs`), `None` when
    /// the refs are already present (replayed from memo) or the
    /// follow is cut off. Callers bracket with `follow_end`.
    fn follow_begin(&mut self, key: &(String, String)) -> Option<usize> {
        if self.depth > 64 || self.stack.contains(key) {
            return None;
        }
        let cached: Option<Vec<String>> =
            self.memo.borrow().refs.get(key).cloned();
        if let Some(cached) = cached {
            // TEMP-ORDERDIAG3: trace memo replays. Removed before close.
            if std::env::var("RUBYC_ORDERDIAG").is_ok()
                && (key.0.contains("AppContextConfigHelper")
                    || key.0.contains("NumberFormatInfo")
                    || key.0.contains("HillClimbing"))
            {
                eprintln!(
                    "[rubyc] TEMP-ORDERDIAG3 replay={}:{:?} n={}",
                    key.0,
                    key.1,
                    cached.len()
                );
            }
            for r in cached {
                if !self.refs.contains(&r) {
                    self.refs.push(r);
                }
            }
            return None;
        }
        self.stack.push(key.clone());
        self.depth += 1;
        Some(self.refs.len())
    }

    fn follow_end(&mut self, key: (String, String), start: usize) {
        self.depth = self.depth.saturating_sub(1);
        self.stack.pop();
        let found: Vec<String> = self.refs[start..].to_vec();
        self.memo.borrow_mut().refs.insert(key, found);
    }

    /// Scan every instance constructor body of an already-resolved
    /// entry class for static reads (called when an initializer
    /// constructs it via `new`). Recursion through nested `new`
    /// (the `New` arm calls back here) is bounded by the shared
    /// follow protocol (cycle stack, depth cap, memo).
    /// Self-construction (`new X()` inside X's own static init,
    /// e.g. `ThreadPoolInstance`) is scanned too — the edge to self
    /// is meaningless but the ctor's reads of OTHER classes'
    /// statics still constrain order. Base-class ctors (run by
    /// `base(...)` before the derived body) are scanned too —
    /// their static reads constrain order equally.
    fn scan_ctor_bodies(&mut self, class_name: &str) {
        // TEMP-ORDERDIAG: trace the transitive ctor scan. Removed before close.
        let diag = std::env::var("RUBYC_ORDERDIAG").is_ok();
        let key = (class_name.to_owned(), "\0ctor".to_owned());
        let Some(start) = self.follow_begin(&key) else {
            return;
        };
        let simple = class_name.rsplit('.').next().unwrap_or(class_name);
        let mut base_spellings: Vec<String> = Vec::new();
        let mut bodies: Vec<(
            *const Vec<crate::models::Stmt>,
            Vec<(String, String)>,
        )> = Vec::new();
        let mut found_decl = false;
        let mut ctor_count = 0usize;
        if let Some(decl) = self.classes.iter().find(|c| c.name == class_name) {
            found_decl = true;
            for m in &decl.members {
                if let ClassMember::Method(md) = m {
                    if md.is_static || md.name != simple {
                        continue;
                    }
                    if let Some((kind, args)) = &md.base_call {
                        for a in args {
                            self.visit_expr(a);
                        }
                        if kind == "base" {
                            base_spellings.extend(decl.bases.iter().cloned());
                        }
                        // `this(...)` chains stay in this class: every
                        // sibling ctor body is scanned below anyway.
                    }
                    if let Some(body) = &md.body {
                        let params = md
                            .params
                            .iter()
                            .map(|p| (p.name.clone(), p.type_name.clone()))
                            .collect();
                        bodies.push((&body.stmts as *const Vec<crate::models::Stmt>, params));
                        ctor_count += 1;
                    }
                }
            }
        }
        // Raw pointers avoid holding `self.classes` borrowed across
        // the `visit_stmt` recursion (`visit_expr` reborrows `self`
        // mutably via the `New` arm). The pointers derive from
        // `self.classes` entries, which outlive the collector.
        // TEMP-ORDERDIAG: trace the transitive ctor scan. Removed before close.
        if diag {
            eprintln!(
                "[rubyc] TEMP-ORDERDIAG scan={class_name} decl={found_decl} ctors={ctor_count} refs={:?}",
                self.refs
            );
        }
        // Bodies belong to the constructed class: nested same-class
        // calls (`this.RefreshLoggingEnabled()`) resolve there, not
        // in the initializer's class.
        let prev_owner = std::mem::replace(&mut self.owner, class_name.to_owned());
        for (ptr, params) in bodies {
            // Ctor parameters scope the body the same way method
            // parameters do (see `scopes`).
            self.scopes.push(params.into_iter().collect());
            // SAFETY: `ptr` points into `self.classes` (see above);
            // the AST is never mutated during collection.
            for s in unsafe { &*ptr } {
                self.visit_stmt(s);
            }
            self.scopes.pop();
        }
        for b in &base_spellings {
            let base = b
                .split('<')
                .next()
                .unwrap_or(b)
                .trim()
                .trim_end_matches('?')
                .trim_end_matches("[]");
            let mut path = base.to_owned();
            loop {
                if let Some(info) = lower::class_for_spelling(self.table, &path) {
                    self.scan_ctor_bodies(&info.name.clone());
                    break;
                }
                match path.find('.') {
                    Some(i) => path = path[i + 1..].to_owned(),
                    None => break,
                }
            }
        }
        self.owner = prev_owner;
        self.follow_end(key, start);
    }

    /// Scan one method/getter body of an already-resolved entry
    /// class (shared follow protocol bounds recursion). By default
    /// every same-named body is scanned (overloads share the name) —
    /// conservative and ordering-only — but when the call-site arity
    /// is known only arity-matching overloads are scanned (a 2-arg
    /// `TryParse` call doesn't drag in the 4-arg overload's
    /// statics). No match falls back to all same-named
    /// (optional/default params make arity flexible). `owner` swaps
    /// to the declaring class for the scan so nested calls resolve
    /// in the right scope, then swaps back.
    fn scan_method_body(&mut self, owner: &str, name: &str, arity: Option<usize>) {
        let base = name.split('<').next().unwrap_or(name).trim();
        if base.is_empty() {
            return;
        }
        // Arity joins the memo key: different call shapes scan
        // different overload subsets.
        let member_key = match arity {
            Some(a) => format!("{base}/{a}"),
            None => base.to_owned(),
        };
        let key = (owner.to_owned(), member_key);
        let Some(start) = self.follow_begin(&key) else {
            return;
        };
        // Copy the slice out so the immutable borrow of
        // `self.classes` ends before the `visit_*` recursion
        // reborrows `self` mutably.
        let classes: &[&ClassDecl] = self.classes;
        let Some(decl) = classes.iter().find(|c| c.name == owner) else {
            // No declaration (e.g. a table-only entry): memoize the
            // empty set so the guard below still balances.
            self.follow_end(key, start);
            return;
        };
        let mut bodies: Vec<(
            *const Vec<crate::models::Stmt>,
            Vec<(String, String)>,
        )> = Vec::new();
        let mut get_exprs: Vec<*const Expr> = Vec::new();
        // Arity filter: exact matches first; empty result falls back
        // to every same-named body (optional/default params make
        // arity flexible; properties ignore arity).
        let mut matched_any = false;
        for m in &decl.members {
            match m {
                ClassMember::Method(md) if md.name == base => {
                    if arity.is_some_and(|a| md.params.len() != a) {
                        continue;
                    }
                    matched_any = true;
                    if let Some(body) = &md.body {
                        // Clone the parameter list (name → type) so
                        // the body is scanned with its own scope (see
                        // `scopes`): receivers naming parameters
                        // resolve to the parameter's class.
                        let params = md
                            .params
                            .iter()
                            .map(|p| (p.name.clone(), p.type_name.clone()))
                            .collect();
                        bodies.push((&body.stmts as *const Vec<crate::models::Stmt>, params));
                    }
                }
                ClassMember::Property(pd) if pd.name == base => {
                    matched_any = true;
                    if let Some(get) = &pd.get {
                        // Property getters take no parameters.
                        bodies.push((
                            &get.stmts as *const Vec<crate::models::Stmt>,
                            Vec::new(),
                        ));
                    }
                    if let Some(ge) = &pd.get_expr {
                        get_exprs.push(ge as *const Expr);
                    }
                }
                _ => {}
            }
        }
        if !matched_any {
            for m in &decl.members {
                match m {
                    ClassMember::Method(md) if md.name == base => {
                        if let Some(body) = &md.body {
                            let params = md
                                .params
                                .iter()
                                .map(|p| (p.name.clone(), p.type_name.clone()))
                                .collect();
                            bodies.push((
                                &body.stmts as *const Vec<crate::models::Stmt>,
                                params,
                            ));
                        }
                    }
                    ClassMember::Property(pd) if pd.name == base => {
                        if let Some(get) = &pd.get {
                            // Property getters take no parameters.
                            bodies.push((
                                &get.stmts as *const Vec<crate::models::Stmt>,
                                Vec::new(),
                            ));
                        }
                        if let Some(ge) = &pd.get_expr {
                            get_exprs.push(ge as *const Expr);
                        }
                    }
                    _ => {}
                }
            }
        }
        // Inherited members live on base classes, not the owner:
        // a call resolving to an inherited method (`SetResult` on a
        // sentinel box inheriting `StateMachineBox.SetResult`) must
        // scan the base body, or the transitive static reads below
        // it are invisible and the edge is lost. Walk bases
        // (memo-keyed per base, so hierarchies terminate).
        if bodies.is_empty() && get_exprs.is_empty() {
            let base_names: Vec<String> = decl
                .bases
                .iter()
                .map(|b| b.split('<').next().unwrap_or(b).trim().to_owned())
                .filter(|b| !b.is_empty())
                .collect();
            for b in base_names {
                self.scan_method_body(&b, name, arity);
            }
            self.follow_end(key, start);
            return;
        }
        let prev = std::mem::replace(&mut self.owner, owner.to_owned());
        // TEMP-ORDERDIAG2: trace method-body scans. Removed before close.
        if std::env::var("RUBYC_ORDERDIAG").is_ok()
            && (owner.contains("AppContextConfigHelper")
                || owner.contains("NumberFormatInfo")
                || name.contains("TryParse"))
        {
            eprintln!("[rubyc] TEMP-ORDERDIAG2 methodscan={owner}::{base}");
        }
        // SAFETY: pointers derive from `self.classes` entries, which
        // outlive the collector; the AST is never mutated mid-scan.
        for (ptr, params) in bodies {
            // Each overload scans under its own parameter scope (see
            // `scopes`); property getters carry no parameters.
            self.scopes.push(params.into_iter().collect());
            for s in unsafe { &*ptr } {
                self.visit_stmt(s);
            }
            self.scopes.pop();
        }
        for ptr in get_exprs {
            self.visit_expr(unsafe { &*ptr });
        }
        self.owner = prev;
        self.follow_end(key, start);
    }

    /// Follow a `this.`-qualified call or member access into the
    /// owner's declaration when it names a same-class
    /// method/property; otherwise a no-op (locals, globals, other
    /// classes' members are handled by the existing arms).
    fn follow_owner_call(&mut self, object: &Expr, name: &str, arity: Option<usize>) {
        let owner_simple = self
            .owner
            .rsplit('.')
            .next()
            .unwrap_or(self.owner.as_str())
            .to_owned();
        let same_owner = match object {
            Expr::Ident { name: recv, .. } => {
                recv == "this" || recv == &owner_simple || recv == &self.owner
            }
            _ => false,
        };
        if same_owner {
            let owner = self.owner.clone();
            self.scan_method_body(&owner, name, arity);
            // Virtual dispatch runs the most-derived override, not
            // the owner's body: a base ctor calling `this.Foo()`
            // executes `Derived.Foo()`, whose static reads constrain
            // order too (base `Encoding` ctor → override
            // `Latin1Encoding.SetDefaultFallbacks` → static
            // `SingletonInstance`; missing it faulted the encoding
            // fallback on a null object). Scan overrides in every
            // (transitively) derived class. Over-approximation, but
            // ordering-only and memo-bounded.
            for d in self.derived_classes(&owner) {
                self.scan_method_body(&d, name, arity);
            }
            return;
        }
        // Statically resolvable receiver (`B.Foo()`, `B.Prop`,
        // including bare `Ident` class names): the callee body runs
        // at init time when the caller does, so its transitive
        // static reads constrain order too. Instance calls are
        // followed the same way (over-approximation: the runtime
        // type is unknown, but any static read in the body is a
        // real constraint if it runs). Field reads and locals hit
        // no method/property below and no-op (a local sharing a
        // class name adds a harmless extra edge at most).
        // NOTE: this must run for `Ident` receivers too — an early
        // `return` here once skipped exactly the
        // `AppContextConfigHelper.GetBoolean...` chain (CultureInfo
        // never ordered, fault reading its fields).
        if let Some(cls) = self.receiver_class(object) {
            self.scan_method_body(&cls, name, arity);
            // Same virtual-dispatch over-approximation as above: the
            // runtime type may derive from the static receiver, so
            // overrides' static reads constrain order too.
            for d in self.derived_classes(&cls) {
                self.scan_method_body(&d, name, arity);
            }
        }
        if let Expr::Ident { name: recv, .. } = object {
            if recv == "base" {
                // `base.Foo()` runs the base-class method at init
                // time: scan every base declaration's same-named body.
                let classes: &[&ClassDecl] = self.classes;
                let owner = self.owner.clone();
                let mut bases: Vec<String> = Vec::new();
                if let Some(decl) = classes.iter().find(|c| c.name == owner) {
                    bases.extend(decl.bases.iter().cloned());
                }
                for b in &bases {
                    let base = b
                        .split('<')
                        .next()
                        .unwrap_or(b)
                        .trim()
                        .trim_end_matches('?')
                        .trim_end_matches("[]");
                    let mut path = base.to_owned();
                    loop {
                        if let Some(info) = lower::class_for_spelling(self.table, &path) {
                            let entry = info.name.clone();
                            self.scan_method_body(&entry, name, None);
                            break;
                        }
                        match path.find('.') {
                            Some(i) => path = path[i + 1..].to_owned(),
                            None => break,
                        }
                    }
                }
            }
        }
    }
}

impl Visitor for InitRefCollector<'_> {
    fn visit_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Binary { lhs, rhs, .. } => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
            }
            Expr::Unary { inner, .. } => self.visit_expr(inner),
            Expr::Ternary {
                cond,
                then_expr,
                else_expr,
                ..
            } => {
                self.visit_expr(cond);
                self.visit_expr(then_expr);
                self.visit_expr(else_expr);
            }
            Expr::Call { callee, args, .. } => {
                for a in args {
                    self.visit_expr(a);
                }
                // Bare same-class call (`RefreshLoggingEnabled()` in a
                // ctor): the callee body runs at init time, so its
                // static reads constrain order. Non-methods (locals,
                // globals, static imports) simply match nothing.
                // Arity narrows overloads (see `scan_method_body`).
                let owner = self.owner.clone();
                self.scan_method_body(&owner, callee, Some(args.len()));
            }
            Expr::New { class_name, args, .. } => {
                // `new X(...)` in an initializer runs X's constructor,
                // which may read X's statics (or its bases'): order X
                // first (e.g. `ThreadPoolInstance = new
                // PortableThreadPool()` needing sibling statics).
                // Cycles fall back to declaration order (documented).
                let base = class_name
                    .split('<')
                    .next()
                    .unwrap_or(class_name)
                    .trim();
                // Self-construction (`new X()` inside X's own static
                // init) adds no edge, but the ctor body still runs at
                // init time — its static reads constrain order, so the
                // resolution/scan below runs regardless.
                if !base.is_empty() {
                    // Resolve like a receiver spelling (qualified or
                    // simple) to the entry name for edge keying.
                    let mut path = base.to_owned();
                    loop {
                        if let Some(info) = lower::class_for_spelling(self.table, &path) {
                            if info.name != self.current
                                && !self.refs.contains(&info.name)
                            {
                                self.refs.push(info.name.clone());
                            }
                            // The ctor body runs at init time: its own
                            // static reads (e.g. `NativeRuntimeEventSource.Log`
                            // inside `PortableThreadPool()`) constrain order too.
                            let entry = info.name.clone();
                            self.scan_ctor_bodies(&entry);
                            break;
                        }
                        match path.find('.') {
                            Some(i) => path = path[i + 1..].to_owned(),
                            None => break,
                        }
                    }
                }
                for a in args {
                    self.visit_expr(a);
                }
            }
            Expr::Member { object, name, .. } => {
                self.record(object);
                self.visit_expr(object);
                // Same-class property read (`this.Foo`, `C.Foo`): an
                // explicit getter body runs at init time.
                self.follow_owner_call(object, name, None);
            }
            Expr::MethodCall { object, name, args, .. } => {
                self.record(object);
                self.visit_expr(object);
                for a in args {
                    self.visit_expr(a);
                }
                // Same-class helper (`this.RefreshLoggingEnabled()`):
                // the callee body runs at init time.
                self.follow_owner_call(object, name, Some(args.len()));
            }
            Expr::Literal(_) => {}
            Expr::Await { inner, .. } => self.visit_expr(inner),
            Expr::NewArray { size, .. } => self.visit_expr(size),
            Expr::Index { object, index, .. } => {
                self.visit_expr(object);
                self.visit_expr(index);
            }
            Expr::Ident { .. } => {}
            Expr::Cast { expr, .. } => self.visit_expr(expr),
            Expr::Default { .. } => {}
            Expr::AssignComplex { lhs, value, .. } => {
                self.visit_expr(lhs);
                self.visit_expr(value);
            }
            Expr::Assign { target, value, .. } => {
                self.visit_expr(target);
                self.visit_expr(value);
            }
            Expr::PostInc { target, .. } | Expr::PostDec { target, .. } => {
                self.visit_expr(target);
            }
            Expr::SwitchExpr { subject, arms, .. } => {
                self.visit_expr(subject);
                for arm in arms {
                    self.visit_expr(&arm.body);
                }
            }
            Expr::InterpolatedString { parts, .. } => {
                for p in parts {
                    if let crate::models::InterpPart::Hole(e) = p {
                        self.visit_expr(e);
                    }
                }
            }
        }
    }
}

/// Declaration-order-preserving topological order of classes for
/// static-initializer emission: a class sorts after every other
/// class its static initializers reference (member receivers via
/// [`InitRefCollector`], plus direct base classes, whose statics a
/// derived initializer may read through inheritance). References to
/// classes outside the unit are ignored. Cycles fall back to
/// declaration order for the cyclic remainder (documented
/// divergence; .NET observes partially-initialized state there).
fn init_class_order<'c>(
    all_classes: &[&'c ClassDecl],
    table: &TypeTable,
) -> Vec<&'c ClassDecl> {
    let n = all_classes.len();
    let pos: std::collections::BTreeMap<&str, usize> = all_classes
        .iter()
        .enumerate()
        .map(|(i, c)| (c.name.as_str(), i))
        .collect();
    // deps[i] = indices that must initialize before i.
    let mut deps: Vec<Vec<usize>> = vec![Vec::new(); n];
    // Shared method-scan memo for the whole ordering pass (see
    // `InitScanMemo`): each (class, member) body is walked once no
    // matter how many initializers reach it.
    let memo = std::cell::RefCell::new(InitScanMemo {
        refs: std::collections::HashMap::new(),
    });
    for (i, class) in all_classes.iter().enumerate() {
        let mut edge_to = |name: &str, deps: &mut Vec<Vec<usize>>| {
            if let Some(&j) = pos.get(name) {
                if j != i && !deps[i].contains(&j) {
                    deps[i].push(j);
                }
            }
        };
        for b in &class.bases {
            let base_simple = b
                .split('<')
                .next()
                .unwrap_or(b)
                .rsplit('.')
                .next()
                .unwrap_or(b);
            // Match by simple or full name against unit classes.
            for other in all_classes.iter() {
                let other_simple =
                    other.name.rsplit('.').next().unwrap_or(&other.name);
                if other.name == *b || other_simple == base_simple {
                    let nm = other.name.clone();
                    edge_to(&nm, &mut deps);
                    break;
                }
            }
        }
        let mut collector = InitRefCollector {
            table,
            classes: all_classes,
            current: class.name.as_str(),
            owner: class.name.clone(),
            refs: Vec::new(),
            stack: Vec::new(),
            depth: 0,
            memo: &memo,
            scopes: Vec::new(),
        };
        for m in &class.members {
            // Static field AND static auto-property initializers run
            // in the prologue; both constrain its order. Explicit-
            // accessor properties are inert (no storage, nothing runs).
            if let ClassMember::Field(fd) = m {
                if !fd.is_static {
                    continue;
                }
                for pre in &fd.init_pre {
                    collector.visit_stmt(pre);
                }
                collector.visit_expr(&fd.init);
            } else if let ClassMember::Property(pd) = m {
                if !pd.is_static
                    || pd.get.is_some()
                    || pd.set.is_some()
                    || pd.get_expr.is_some()
                {
                    continue;
                }
                collector.visit_expr(&pd.init);
            }
        }
        for r in std::mem::take(&mut collector.refs) {
            // TEMP-ORDERDIAG: show final refs per class. Removed before close.
            // TEMP-SENTDIAG (2026-10): sentinel crash diagnosis. Removed before close.
            if std::env::var("RUBYC_ORDERDIAG").is_ok()
                && (class.name.contains("PortableThreadPool")
                    || class.name.contains("HillClimbing")
                    || class.name.contains("PoolingAsyncValueTaskMethodBuilder")
                    || class.name.contains("ManualResetValueTaskSourceCore")
                    || class.name.contains("StateMachineBox")
                    || class.name.contains("SyncSuccessSentinel")
                    || r.contains("NativeRuntimeEventSource")
                    || r.contains("ManualResetValueTaskSourceCoreShared")
                    || r.contains("CultureInfo"))
            {
                eprintln!("[rubyc] TEMP-ORDERDIAG class={} ref={r}", class.name);
            }
            edge_to(&r, &mut deps);
        }
    }
    // Stable Kahn: earliest-declared ready class first; on stall
    // (cycle) emit the class that unblocks the most other stalled
    // classes (most dependents among the un-emitted set), then
    // fewest unmet deps, then earliest index for determinism. A
    // stall forces some class to observe partial state; emitting a
    // dependency first (e.g. CultureInfo before the HillClimbing
    // that waits on it) resolves the stall for its waiters, while
    // emitting the waiter first (what earliest-index did) faults
    // on the unread dependency. Fewest-unmet alone picks the
    // waiter (it has just 1 unmet: the dependency itself).
    let mut emitted = vec![false; n];
    let mut order: Vec<&'c ClassDecl> = Vec::with_capacity(n);
    // TEMP-ORDERDIAG4: dump positions of key classes + cycle stalls.
    // Removed before close.
    let order_diag = std::env::var("RUBYC_ORDERDIAG").is_ok();
    for _ in 0..n {
        let mut pick: Option<usize> = None;
        for i in 0..n {
            if emitted[i] {
                continue;
            }
            if deps[i].iter().all(|&d| emitted[d]) {
                pick = Some(i);
                break;
            }
        }
        let stalled = pick.is_none();
        let i = pick.unwrap_or_else(|| {
            // Cycle: emit the un-emitted class waited on by the most
            // other un-emitted classes (it unblocks the most), then
            // fewest unmet deps, then earliest index.
            let unemitted: Vec<usize> = (0..n).filter(|&i| !emitted[i]).collect();
            unemitted
                .iter()
                .min_by_key(|&&i| {
                    let waiters = unemitted
                        .iter()
                        .filter(|&&y| y != i && deps[y].contains(&i))
                        .count();
                    let unmet = deps[i].iter().filter(|&&d| !emitted[d]).count();
                    (
                        std::cmp::Reverse(waiters),
                        unmet,
                        i,
                    )
                })
                .copied()
                .expect("order loop bounded by n")
        });
        emitted[i] = true;
        order.push(all_classes[i]);
        if order_diag && stalled {
            eprintln!(
                "[rubyc] TEMP-ORDERDIAG4 stall@{}, emitting {} with {} unmet deps",
                order.len(),
                all_classes[i].name,
                deps[i].iter().filter(|&&d| !emitted[d]).count()
            );
            // TEMP-ORDERDIAG7: name the unmet deps (cycle diagnosis).
            // Removed before close.
            if all_classes[i].name == "HillClimbing"
                || all_classes[i].name == "CultureInfo"
                || all_classes[i].name == "NumberFormatInfo"
            {
                let unmet: Vec<&str> = deps[i]
                    .iter()
                    .filter(|&&d| !emitted[d])
                    .map(|&d| all_classes[d].name.as_str())
                    .collect();
                eprintln!(
                    "[rubyc] TEMP-ORDERDIAG7 {} unmet={:?}",
                    all_classes[i].name, unmet
                );
            }
            // TEMP-ORDERDIAG8: trace one cycle by DFS through unmet
            // deps, starting from HillClimbing's stall (the crash
            // relevant one). Removed before close.
            if order_diag && stalled && all_classes[i].name == "HillClimbing" {
                static FIRST: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !FIRST.swap(true, std::sync::atomic::Ordering::SeqCst) {
                    let mut path: Vec<usize> = vec![i];
                    let mut seen: std::collections::HashSet<usize> =
                        std::collections::HashSet::new();
                    seen.insert(i);
                    loop {
                        let cur = *path.last().unwrap();
                        let nxt = deps[cur]
                            .iter()
                            .find(|&&d| !emitted[d] && !seen.contains(&d));
                        match nxt {
                            Some(&n) => {
                                seen.insert(n);
                                path.push(n);
                            }
                            None => {
                                // Dead end or back-edge: find a member of
                                // path that `cur` waits on (the cycle).
                                if let Some(&back) = deps[cur]
                                    .iter()
                                    .find(|&&d| path.contains(&d))
                                {
                                    let start = path.iter().position(|&x| x == back).unwrap();
                                    let names: Vec<&str> = path[start..]
                                        .iter()
                                        .map(|&x| all_classes[x].name.as_str())
                                        .collect();
                                    eprintln!(
                                        "[rubyc] TEMP-ORDERDIAG8 cycle={:?}",
                                        names
                                    );
                                }
                                break;
                            }
                        }
                        if path.len() > 40 {
                            break;
                        }
                    }
                }
            }
        }
    }
    if order_diag {
        // TEMP-ORDERDIAG6: verify Kahn invariant (every dep emitted
        // before its dependent). Removed before close.
        let position: std::collections::HashMap<usize, usize> = order
            .iter()
            .enumerate()
            .map(|(p, c)| {
                let idx = all_classes.iter().position(|a| std::ptr::eq(*a, *c)).unwrap_or(usize::MAX);
                (idx, p)
            })
            .collect();
        let mut violations = 0usize;
        for (i, _) in all_classes.iter().enumerate() {
            for &d in &deps[i] {
                let (pi, pd) = (position[&i], position[&d]);
                if pd > pi {
                    violations += 1;
                    if violations <= 5 {
                        eprintln!(
                            "[rubyc] TEMP-ORDERDIAG6 VIOLATION {}@{} after {}@{}",
                            all_classes[d].name, pd, all_classes[i].name, pi
                        );
                    }
                }
            }
        }
        if violations > 0 {
            eprintln!("[rubyc] TEMP-ORDERDIAG6 violations={violations}");
        }
        for (pos, c) in order.iter().enumerate() {
            if c.name.contains("CultureInfo") && !c.name.contains("Data")
                || c.name == "HillClimbing"
                || c.name == "PortableThreadPool"
                // TEMP-SENTDIAG positions. Removed before close.
                || c.name.contains("PoolingAsyncValueTaskMethodBuilder")
                || c.name.contains("ManualResetValueTaskSourceCore")
                || c.name.contains("StateMachineBox")
                || c.name.contains("SyncSuccessSentinel")
            {
                let idx = all_classes
                    .iter()
                    .position(|a| std::ptr::eq(*a, *c))
                    .unwrap_or(usize::MAX);
                eprintln!("[rubyc] TEMP-ORDERDIAG4 order@{pos} = {} idx={idx}", c.name);
            }
        }
        // TEMP-ORDERDIAG5: dump deps of key classes. Removed before close.
        if let (Some(&hi), Some(&ci)) = (pos.get("HillClimbing"), pos.get("CultureInfo")) {
            eprintln!(
                "[rubyc] TEMP-ORDERDIAG5 hi={hi} ci={ci} contains={} ndeps={}",
                deps[hi].contains(&ci),
                deps[hi].len()
            );
            let cnames: Vec<&str> = deps[ci].iter().map(|&d| all_classes[d].name.as_str()).collect();
            eprintln!("[rubyc] TEMP-ORDERDIAG5 CultureInfo ndeps={} deps={:?}", cnames.len(), cnames);
        }
    }
    order
}
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
    // TEMP-VTALIGN: verify vtable row basis (emission order) matches
    // table order used by `new`. Removed before close.
    if std::env::var("RUBYC_VTALIGN").is_ok() {
        let filtered: Vec<&str> = all_classes
            .iter()
            .filter(|c| table.class(&c.name).is_some())
            .map(|c| c.name.as_str())
            .collect();
        let mut mism = 0usize;
        let mut first_mismatch = String::new();
        for (i, f) in filtered.iter().enumerate() {
            let tp = table.classes.get(i).map(|c| c.key.path.as_str()).unwrap_or("<none>");
            // Compare by simple name to tolerate qualification diffs.
            let fs = f.rsplit('.').next().unwrap_or(f);
            let ts = tp.rsplit('.').next().unwrap_or(tp).split('<').next().unwrap_or(tp);
            if fs != ts {
                if mism == 0 {
                    first_mismatch = format!("all[{i}]={f} table[{i}]={tp}");
                }
                mism += 1;
            }
        }
        eprintln!(
            "[rubyc] TEMP-VTALIGN filtered={} table={} mismatches={} first={}",
            filtered.len(),
            table.classes.len(),
            mism,
            first_mismatch
        );
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
    // Same-name+arity overloads append in emission order (NOT
    // last-wins): positional lookups (`impl_index`) pick their own
    // slot; legacy arity-only callers take the last below.
    let mut fn_index: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
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
        sig: &[String],
        fn_index: &std::collections::BTreeMap<String, Vec<usize>>,
    ) -> Option<usize> {
        // Exact signature first across the whole chain, arity
        // fallback second: the single-pass fallback elects a
        // derived arity-match (e.g. an array `GetBytes`) without
        // ever reaching the base's exact implementation (an
        // internal pointer `GetBytes`), routing virtual calls to a
        // body that throws reading Length off the pointer.
        impl_index_seen(classes, table, class_name, method, sig, fn_index, &mut Vec::new(), true).or_else(
            || {
                impl_index_seen(
                    classes,
                    table,
                    class_name,
                    method,
                    sig,
                    fn_index,
                    &mut Vec::new(),
                    false,
                )
            },
        )
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
        sig: &[String],
        fn_index: &std::collections::BTreeMap<String, Vec<usize>>,
        seen: &mut Vec<String>,
        exact_only: bool,
    ) -> Option<usize> {
        // TEMP-WALKDBG2: print every walk entry (does the base walk
        // recurse?). Removed before close.
        if std::env::var("RUBYC_WALKDBG").is_ok()
            && method == "GetBytes"
            && class_name.contains("Encoding")
        {
            eprintln!(
                "[rubyc] TEMP-WALKDBG2 enter class={} siglen={} seen={:?}",
                class_name,
                sig.len(),
                seen,
            );
        }
        if class_name == "@start" {
            return None;
        }
        if seen.contains(&class_name.to_owned()) {
            return None;
        }
        seen.push(class_name.to_owned());
        // Same-named declarations are usually partial parts of one
        // class (`Encoding` plus its `Encoding.Internal` part): a
        // first-match-only search stops at a part lacking the method
        // and never sees the body in a later part (lost
        // `SetDefaultFallbacks` impl, faulting `ConsoleEncoding`
        // construction on a null vtable slot). Try every same-named
        // declaration before giving up and walking bases. First hit
        // still wins, so single-declaration behavior is unchanged.
        let mut found: Option<&crate::models::ast::method::MethodDecl> = None;
        for class in classes.iter().filter(|c| c.name == class_name) {
            // Signature-aware match: same-name overloads sharing an
            // arity but not param types (Latin1 GetByteCount#1 span
            // vs string; GetBytes#5 char[] vs string) must resolve
            // to their own bodies, not the first same-arity body.
            // Type spellings drift between table and decl
            // (`String`/`string`, `out ` prefixes), so compare
            // normalized; on no exact hit fall back to the first
            // same-arity body (yesterday's behavior, never worse).
            fn norm_ty(s: &str) -> String {
                let mut t = s.to_lowercase();
                for w in ["out ", "ref ", "in "] {
                    if let Some(rest) = t.strip_prefix(w) {
                        t = rest.to_string();
                        break;
                    }
                }
                t.retain(|c| !c.is_whitespace());
                // Nullable annotations (`EncoderNLS?`) don't change
                // the runtime signature: without this, a nullable
                // spelling never matches its declaration and the
                // same-arity fallback elects a sibling overload's
                // body (an internal pointer `GetBytes` routed to an
                // array overload that throws reading Length off the
                // pointer).
                t.trim_end_matches('?').trim_end_matches('&').to_string()
            }
            let by_sig = class.members.iter().find_map(|m| match m {
                ClassMember::Method(md)
                    if md.name == method
                        && md.params.len() == sig.len()
                        && md
                            .params
                            .iter()
                            .zip(sig.iter())
                            .all(|(p, t)| norm_ty(&p.type_name) == norm_ty(t))
                        && md.body.is_some() =>
                {
                    Some(md)
                }
                _ => None,
            });
            // TEMP-SIGDBG: show both sides when a GetBytes sig
            // match fails at an Encoding level (spelling drift?).
            // Removed before close.
            if std::env::var("RUBYC_SIGDBG").is_ok()
                && method == "GetBytes"
                && sig.len() == 5
                && (class_name.contains("Encoding"))
            {
                for m in class.members.iter().filter_map(|m| match m {
                    ClassMember::Method(md)
                        if md.name == method && md.params.len() == sig.len() =>
                    {
                        Some(md)
                    }
                    _ => None,
                }) {
                    eprintln!(
                        "[rubyc] TEMP-SIGDBG {}.{} decl=[{:?}] vs sig=[{:?}] body={}",
                        class_name,
                        method,
                        m.params.iter().map(|p| p.type_name.clone()).collect::<Vec<_>>(),
                        sig,
                        m.body.is_some()
                    );
                }
            }
            // Remember the last attempt for the audit below; a hit
            // with a resolving function index returns immediately.
            // In exact-only mode there is no arity fallback (first
            // pass): a same-arity mismatch must not preempt the base
            // walk.
            found = if exact_only {
                by_sig
            } else {
                by_sig.or_else(|| {
                    class.members.iter().find_map(|m| match m {
                        ClassMember::Method(md)
                            if md.name == method
                                && md.params.len() == sig.len()
                                && md.body.is_some() =>
                        {
                            Some(md)
                        }
                        _ => None,
                    })
                })
            };
            // Positional resolution among same-name+arity overloads:
            // `fn_index` keys (`Class::name#arity`) conflate overloads
            // sharing an arity (Unicode `GetByteCount#3` char[] vs
            // char*: last-wins sent every #3 slot to one body). Count
            // the found declaration's position in declaration order
            // (same skips as emission: bodyless, dtors, >64 params)
            // and take that slot — mirroring `functions_index` and
            // the fixed `overload_position`.
            if let Some(decl) = found {
                let mut pos = 0usize;
                let mut here: Option<usize> = None;
                for m in &class.members {
                    if let crate::models::ClassMember::Method(md) = m {
                        if md.name != method || md.params.len() != sig.len() {
                            continue;
                        }
                        if md.body.is_none()
                            || md.name.starts_with('~')
                            || md.params.len() > 64
                        {
                            continue;
                        }
                        if std::ptr::eq(md, decl) {
                            here = Some(pos);
                            break;
                        }
                        pos += 1;
                    }
                }
                let key = format!("{}::{}#{}", class_name, method, decl.params.len());
                if let Some(vec) = fn_index.get(&key) {
                    if let Some(n) = here {
                        if let Some(&idx) = vec.get(n) {
                            return Some(idx);
                        }
                    }
                    if let Some(&idx) = vec.last() {
                        return Some(idx);
                    }
                }
            }
            // No hit in this same-named declaration (partial part
            // without the method): try the next one.
        }
            // TEMP-IMPLDBG2: outcome audit. Removed before close.
            // TEMP-SENTDIAG6: include ConsoleEncoding/SetDefaultFallbacks
            // (bisecting the slot-35 fault). Removed before close.
            if std::env::var("RUBYC_IMPLDBG2").is_ok()
                && (class_name.contains("Latin1")
                    || class_name.contains("ConsoleEncoding")
                    || class_name.contains("Encoding"))
                && (method == "GetByteCount"
                    || method == "GetBytes"
                    || method == "SetDefaultFallbacks")
            {
                let key = format!(
                    "{}::{}#{}",
                    class_name,
                    method,
                    found.map(|d| d.params.len()).unwrap_or(999)
                );
                eprintln!(
                    "[rubyc] TEMP-IMPLDBG2 class={} method={} tablesig={:?} found={} keyhit={} bases={:?}",
                    class_name,
                    method,
                    sig,
                    found
                        .map(|d| format!(
                            "{}#{}({})",
                            d.name,
                            d.params.len(),
                            d.params
                                .iter()
                                .map(|p| p.type_name.clone())
                                .collect::<Vec<_>>()
                                .join(",")
                        ))
                        .unwrap_or("NONE".to_string()),
                    fn_index
                        .get(&key)
                        .and_then(|v| v.last())
                        .map(|i| i.to_string())
                        .unwrap_or("MISS".to_string()),
                    table
                        .class(class_name)
                        .map(|e| e.bases.clone())
                );
            }
        // TEMP-WALKDBG: trace impl_index_seen walk decisions.
        // Removed before close.
        if std::env::var("RUBYC_WALKDBG").is_ok()
            && method == "GetBytes"
            && class_name.contains("Encoding")
        {
            let n_same_named =
                classes.iter().filter(|c| c.name == class_name).count();
            let table_hit = table.class(class_name).is_some();
            let nbases = table.class(class_name).map(|e| e.bases.len()).unwrap_or(999);
            eprintln!(
                "[rubyc] TEMP-WALKDBG class={} siglen={} same_named_decls={} table_hit={} nbases={} seen_len={}",
                class_name,
                sig.len(),
                n_same_named,
                table_hit,
                nbases,
                seen.len(),
            );
        }
        let info = table.class(class_name)?;
        info.bases.iter().find_map(|b| {
            impl_index_seen(classes, table, b, method, sig, fn_index, seen, exact_only)
        })
    }

    // `ctor_index` is built after `local_decls` (below): local
    // functions push first, so constructor numbering must start past
    // them. Kept adjacent to its use-site comment there.

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
                // TEMP-STATICDIAG: dump static indices for crash correlation.
                // Removed before close.
                if std::env::var("RUBYC_STATICDIAG").is_ok()
                    && (fi.name.contains("UICulture")
                        || fi.name.contains("Invariant")
                        || fi.name.contains("uiCulture")
                        // TEMP-SENTDIAG2. Removed before close.
                        || fi.name.contains("s_sentinel")
                        // TEMP-SWDIAG3: newline static (bisecting the
                        // CoreNewLine-null fault). Removed before close.
                        || fi.name.contains("s_coreNewLine")
                        || fi.name.contains("NewLineConst"))
                {
                    eprintln!(
                        "[rubyc] TEMP-STATICDIAG idx={} {}.{} : {}",
                        statics.len(),
                        fi.declaring_class,
                        fi.name,
                        fi.type_name
                    );
                }
                // TEMP-STATICDIAG2: dump specific indices for crash correlation.
                // Removed before close.
                if std::env::var("RUBYC_STATICDIAG2").is_ok() {
                    eprintln!(
                        "[rubyc] TEMP-STATICDIAG2 idx={} {}.{} : {}",
                        statics.len(),
                        fi.declaring_class,
                        fi.name,
                        fi.type_name
                    );
                }
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

/// Pick the best same-arity ctor overload for a `: base(...)` /
/// `: this(...)` chain by scoring chain-arg static types (C#
/// overload resolution via `score_call`, lower wins, ties keep
/// declaration order). Declaration order alone misroutes:
/// `HashSet(int)` chained `: this(capacity, null)` into the
/// first-declared arity-2 overload (`IEnumerable<T>, ...`) and
/// enumerated the int capacity as a collection (fault reading
/// address 11). `None` when no overload applies (caller skips the
/// chain like a bodyless target).
fn pick_chain_ctor(
    cands: &[(Vec<String>, usize)],
    arg_tys: &[String],
    table: &TypeTable,
    class_vars: &[String],
) -> Option<usize> {
    let mut best: Option<(u32, usize)> = None;
    for (sig, idx) in cands {
        if let Some((total, _)) = lower::score_call(sig, arg_tys, 0, table, class_vars, false) {
            let better = match &best {
                None => true,
                Some((b, _)) => total < *b,
            };
            if better {
                best = Some((total, *idx));
            }
        }
    }
    best.map(|(_, i)| i)
}
    // Entry candidates carry whether they take `string[]`: @start
    // builds the argv array only for those (`Main(string[] args)`),
    //paramless entry points keep the old receiver-only call.
    let mut main_upper: Option<(String, usize, bool, bool)> = None;
    let mut main_lower: Option<(String, usize, bool, bool)> = None;

    // Static local functions emit FIRST (indices 1..=L), methods
    // follow: every index in the image stays positional and both
    // index structures (`fn_index` inline, `functions_index` with
    // the same count offset) agree. Table entries are
    // (name, arity, fidx, return type); failed lowerings keep their
    // slot (dummy body) but get a sentinel fidx so calls fall back
    // to the legacy zero placeholder instead of misrouting.
    let local_decls = collect_static_locals(&all_classes);
    // TEMP-LOCALDBG.
    if std::env::var("RUBYC_LOCALDBG").is_ok() {
        for ld in &local_decls {
            eprintln!(
                "[rubyc] TEMP-LOCALDBG collected {}.{}.{}#{}",
                ld.class_name,
                ld.method_name,
                ld.local_name,
                ld.params.len()
            );
        }
    }
    let mut local_table: Vec<(String, String, String, usize, usize, String)> = local_decls
        .iter()
        .enumerate()
        .map(|(j, ld)| {
            (
                ld.class_name.clone(),
                ld.method_name.clone(),
                ld.local_name.clone(),
                ld.params.len(),
                j + 1,
                ld.ret_type.clone(),
            )
        })
        .collect();
    // Precomputed constructor function indices (declaration-order
    // numbering with @start at 0) so `: base(args)` chains resolve
    // even when a derived class is declared before its base. Must
    // mirror the lowering skip rules below (no body, destructors,
    // >64 params, TEMP-SKIPMETHOD): any divergence silently retargets
    // every chain call (see below). Numbering starts past the local
    // functions, which push first (each `local_decls` entry emits
    // exactly one function, including fail-open placeholders).
    //
    // Keyed by (class, arity) to a declaration-ordered list of
    // (param-type signature, fidx): same-arity overloads MUST NOT
    // collapse — `Exception(string?, Exception?)` vs
    // `Exception(SerializationInfo, StreamingContext)` shared one
    // slot under last-wins, retargeting every `: base(msg, inner)`
    // chain in the framework into the serialization ctor (infinite
    // `ThrowIfNull` recursion at runtime). The use site selects by
    // exact signature, falling back to the first (declaration
    // order) entry.
    let mut ctor_index: std::collections::BTreeMap<(String, usize), Vec<(Vec<String>, usize)>> =
        Default::default();
    {
        let mut n = local_decls.len();
        for class in &all_classes {
            if table.class(&class.name).is_none() {
                continue;
            }
            for m in &class.members {
                if let ClassMember::Method(method) = m {
                    if method.body.is_none() || method.name.starts_with('~') {
                        continue;
                    }
                    if method.params.len() > 64 {
                        continue;
                    }
                    // TEMP-SKIPMETHOD mirror (diagnostic): a skipped
                    // method occupies no slot. Removed before close.
                    if std::env::var("RUBYC_SKIPMETHOD").is_ok()
                        && method.name == "LCIDToLocaleName"
                    {
                        continue;
                    }
                    n += 1;
                    if method.ret_type.is_empty() && method.name == class.name {
                        let sig: Vec<String> =
                            method.params.iter().map(|p| p.type_name.clone()).collect();
                        ctor_index
                            .entry((class.name.clone(), method.params.len()))
                            .or_default()
                            .push((sig, n));
                    }
                }
            }
        }
    }
    for (j, ld) in local_decls.iter().enumerate() {
        let fname = format!(
            "{}::{}${}#{}",
            ld.class_name, ld.method_name, ld.local_name,
            ld.params.len()
        );
        let mut f = IrFunction::new(&fname, ReceiverKind::None, ld.params.len());
        let mut lb = lower::Body::new(&mut f, &mut strings);
        let lctx = Ctx {
            table: &table,
            current_class: &ld.class_name,
            current_info: None,
            current_path: None,
            classes: &all_classes,
            import_index: &import_index,
            dep_imports: &dep_imports,
            host_import_count: table.imports.len(),
            static_index: &static_index,
            defines,
            method_vars: &ld.method_vars,
            local_fns: &local_table,
            current_method: Some(ld.method_name.clone()),
            current_ret: Some(ld.ret_type.clone()),
        };
        let lowered: Result<(), LowerError> = (|| {
            lb.scope_mut().push();
            // TEMP-MTRACE for locals (mirrors method markers).
            if std::env::var("RUBYC_MTRACE").is_ok() {
                let marker = format!("M {}::{} (local)\n", ld.class_name, ld.local_name);
                let (off, len) = lb.string_literal(&marker);
                lb.push_raw(Instr::EWriteStr { off, len });
            }
            for p in &ld.params {
                lb.scope_mut()
                    .declare(&p.name, &p.type_name, Access::Default)
                    .map_err(LowerError::UnsupportedConstruct)?;
            }
            for s in &ld.body.stmts {
                if lb.is_sealed() {
                    break; // unreachable code after return/throw
                }
                lb.statement(s, &lctx)?;
            }
            lb.release_scope_locals();
            lb.seal_with_return_zero();
            Ok(())
        })();
        match lowered {
            Ok(()) => {
                functions.push(lb.finish_with(Terminator::Ret(None)));
            }
            Err(_) => {
                let mut db = lower::Body::new(&mut f, &mut strings);
                db.scope_mut().push();
                db.seal_with_return_zero();
                functions.push(db.finish_with(Terminator::Ret(None)));
                local_table[j].4 = usize::MAX;
            }
        }
    }

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
            // TEMP-SKIPMETHOD (diagnostic): skip runaway methods.
            // Removed before close.
            if std::env::var("RUBYC_SKIPMETHOD").is_ok()
                && method.name == "LCIDToLocaleName"
            {
                continue;
            }
            // TEMP-BIGFN: method start trace (last line before an
            // overflow names the culprit).
            if std::env::var("RUBYC_BIGFN").is_ok() {
                eprintln!("[rubyc] TEMP-METHOD {}.{}", class.name, method.name);
            }
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
                // Static local functions of this method (emitted
                // above); bare calls resolve against them first.
                local_fns: &local_table,
                current_method: Some(method.name.clone()),
                current_ret: Some(method.ret_type.clone()),
            };

            lb.scope_mut().push();
            // TEMP-IRDUMP: print lowered IR for target methods.
            // `RUBYC_IRDUMP` lists `Method` or `Class.Method` names
            // (comma-separated; default preserves the legacy
            // `SetDefaultFallbacks` target). Removed before close.
            let dump_spec = std::env::var("RUBYC_IRDUMP").unwrap_or_default();
            let dump_ir = !dump_spec.is_empty()
                && {
                    let full = format!("{}.{}", class.name, method.name);
                    dump_spec.split(',').any(|w| {
                        w.trim() == method.name || w.trim() == full || w.trim().is_empty() && method.name == "SetDefaultFallbacks"
                    })
                };
            // TEMP-MTRACE: method-entry markers for the UTF8NoBOM trap
            // hunt. RUBYC_MTRACE=all traces everything; otherwise only
            // Encoding/Fallback classes. Removed before close.
            // The `@f{idx}` suffix is the emission function index
            // (matches TEMP-FIDXDIAG): overloads share `Class.method`
            // spellings, so the suffix is the only way to tell
            // `IsEnabled#0` from `IsEnabled#3` in a crash trace.
            if std::env::var("RUBYC_MTRACE").is_ok()
                && (std::env::var("RUBYC_MTRACE").unwrap_or_default() == "all"
                    || class.name.contains("Encoding")
                    || class.name.contains("Fallback"))
            {
                let marker = format!(
                    "M {}::{}@f{}\n",
                    class.name,
                    method.name,
                    functions.len() + 1
                );
                let (off, len) = lb.string_literal(&marker);
                lb.push_raw(Instr::EWriteStr { off, len });
            }
            // TEMP-MTRACEVALS2: raw argument values (vregs 1..=4,
            // comma-separated) for count/address verification.
            // Removed before close.
            if std::env::var("RUBYC_MTRACEVALS").is_ok()
                && (method.name == "TranscodeToUtf8"
                    || method.name == "GetBytesFast"
                    || method.name == "Rent"
                    || method.name == "Span"
                    || method.name == "ReadOnlySpan")
            {
                let n = method.params.len().min(4) as u32;
                for i in 1..=n {
                    lb.push_raw(Instr::EWriteInt { src: VReg(i) });
                    let (coff, clen) = lb.string_literal(",");
                    lb.push_raw(Instr::EWriteStr { off: coff, len: clen });
                }
            }
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
                    .declare_byref(&p.name, &p.type_name, Access::Default, p.is_byref)
                    .map_err(LowerError::UnsupportedConstruct)?;
            }
            // TEMP-ARGTRACE: print integer params at entry of a named
            // method (bisecting arg corruption). Removed before close.
            // `RUBYC_ARGTRACE=1` keeps the legacy `FromUnits` target;
            // any other value names a method substring to trace.
            if (std::env::var("RUBYC_ARGTRACE").is_ok() && method.name == "FromUnits")
                || (std::env::var("RUBYC_ARGTRACE").is_ok()
                    && method.name.contains(
                        &std::env::var("RUBYC_ARGTRACE").unwrap_or_default(),
                    )
                    && std::env::var("RUBYC_ARGTRACE").unwrap_or_default() != "1")
            {
                let (off, len) = lb.string_literal(&format!("ARGTRACE {} ", method.name));
                lb.push_raw(Instr::EWriteStr { off, len });
                for (i, _p) in method.params.iter().enumerate() {
                    // Param homes are VReg(i+1) by convention (VReg0 is
                    // `this`/padding) — read directly, bypassing
                    // scope/binding lookup (which may itself be under
                    // test).
                    let v = crate::native::ir::VReg((i + 1) as u32);
                    lb.push_raw(Instr::EWriteInt { src: v });
                    let (off, len) = lb.string_literal(" ");
                    lb.push_raw(Instr::EWriteStr { off, len });
                }
                let (off, len) = lb.string_literal("\n");
                lb.push_raw(Instr::EWriteStr { off, len });
            }
            // TEMP-ARGTRACE2: dump ReadOnlySpan ctor params (pointer,
            // length) to bisect the [3] fault. Removed before close.
            if std::env::var("RUBYC_ARGTRACE2").is_ok()
                && class.name.contains("ReadOnlySpan")
                && method.params.len() == 2
            {
                let tag = format!(
                    "ROS-ctor({},{}) ",
                    method
                        .params
                        .first()
                        .map(|p| p.type_name.as_str())
                        .unwrap_or("?"),
                    method
                        .params
                        .get(1)
                        .map(|p| p.type_name.as_str())
                        .unwrap_or("?")
                );
                let (off, len) = lb.string_literal(&tag);
                lb.push_raw(Instr::EWriteStr { off, len });
                for p in &method.params {
                    if let Some(sym) = lb.scope_mut().lookup(&p.name) {
                        let slot = sym.slot;
                        if slot < lb.bindings.len() {
                            let v = lb.bindings[slot];
                            lb.push_raw(Instr::EWriteInt { src: v });
                            let (off, len) = lb.string_literal("\n");
                            lb.push_raw(Instr::EWriteStr { off, len });
                        }
                    }
                }
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
                    // A miss means the base ctor has no function
                    // (bodyless stub or unknown base): skip the chain
                    // call rather than calling function 0 (@start).
                    // Emitting a wrong-target call is the worst failure
                    // mode (silent jump into an unrelated body); a
                    // skipped bodyless base matches stub handling
                    // elsewhere. Same-arity overloads rank by chain-arg
                    // types (see `pick_chain_ctor`): arity-only
                    // collapsing once routed every
                    // `: base(msg, inner)` into
                    // `Exception(SerializationInfo, StreamingContext)`.
                    let base_name = class.bases.first().map(|s| s.as_str()).unwrap_or("");
                    if !base_name.is_empty() {
                        // TEMP-CHAINDBG2: audit base-chain resolution.
                        // Removed before close.
                        let chain_dbg = std::env::var("RUBYC_CHAINDBG2").is_ok()
                            && (class.name.contains("Task") || base_name.contains("Task"));
                        if let Some(cands) =
                            ctor_index.get(&(base_name.to_string(), args.len()))
                        {
                            let arg_tys: Vec<String> = args
                                .iter()
                                .map(|a| {
                                    lower::static_type_of(a, &lb, &ctx)
                                        .unwrap_or_else(|_| "object".to_owned())
                                })
                                .collect();
                            let base_vars: Vec<String> = classes
                                .iter()
                                .find(|c| c.name == base_name)
                                .map(|c| c.generic_params.clone())
                                .unwrap_or_default();
                            if let Some(idx) =
                                pick_chain_ctor(cands, &arg_tys, &table, &base_vars)
                            {
                                if chain_dbg {
                                    eprintln!(
                                        "[rubyc] TEMP-CHAINDBG2 base-chain {}.{}[generics={}] -> {}#{} fidx={} argtys={:?}",
                                        class.name,
                                        method.name,
                                        class.generic_params.len(),
                                        base_name,
                                        args.len(),
                                        idx,
                                        arg_tys,
                                    );
                                }
                                let mut argv = vec![VReg(0)]; // this
                                for a in &args {
                                    // Bare method names (method groups)
                                    // eval to zero via the
                                    // undeclared-variable fallback in
                                    // `eval` (delegate placeholder).
                                    argv.push(lower::eval(a, &mut lb, &ctx)?);
                                }
                                lb.call_fn(idx, argv);
                            } else if chain_dbg {
                                eprintln!(
                                    "[rubyc] TEMP-CHAINDBG2 base-chain MISS(no-pick) {}.{} -> {}#{} candidates={} argtys={:?}",
                                    class.name,
                                    method.name,
                                    base_name,
                                    args.len(),
                                    cands.len(),
                                    arg_tys,
                                );
                            }
                        } else if chain_dbg {
                            eprintln!(
                                "[rubyc] TEMP-CHAINDBG2 base-chain MISS(no-key) {}.{} -> {}#{}",
                                class.name, method.name, base_name, args.len(),
                            );
                        }
                    }
                }
                // `: this(args)` chains to another constructor of the
                // SAME class (e.g. `MemoryStream() : this(0)`): without
                // this the delegating ctor runs an empty body and none
                // of the target's field stores happen (`CanWrite`
                // stayed false, breaking every `StreamWriter` over a
                // default-constructed stream). Resolve the overload by
                // arity/signature like the base path and invoke it on
                // the same `this` (VReg 0) before the body. Field
                // initializers bind at `new`, not in ctor bodies, so no
                // double-init. A miss (bodyless/unknown target) skips
                // like the base path rather than calling function 0.
                else if target == "this" {
                    if let Some(cands) = ctor_index.get(&(class.name.clone(), args.len())) {
                        // Rank by chain-arg types (see `pick_chain_ctor`;
                        // first-declared alone misroutes). A chain that
                        // can only resolve to this very constructor
                        // (same arity+signature, no alternative) keeps
                        // self: degenerate but C#-legal recursion.
                        let here_sig: Vec<String> = method
                            .params
                            .iter()
                            .map(|p| p.type_name.clone())
                            .collect();
                        let pool: Vec<(Vec<String>, usize)> = {
                            let others: Vec<(Vec<String>, usize)> = cands
                                .iter()
                                .filter(|(s, _)| *s != here_sig)
                                .cloned()
                                .collect();
                            if others.is_empty() {
                                cands.clone()
                            } else {
                                others
                            }
                        };
                        let arg_tys: Vec<String> = args
                            .iter()
                            .map(|a| {
                                lower::static_type_of(a, &lb, &ctx)
                                    .unwrap_or_else(|_| "object".to_owned())
                            })
                            .collect();
                        if let Some(idx) =
                            pick_chain_ctor(&pool, &arg_tys, &table, &class.generic_params)
                        {
                            let mut argv = vec![VReg(0)]; // this
                            for a in &args {
                                argv.push(lower::eval(a, &mut lb, &ctx)?);
                            }
                            lb.call_fn(idx, argv);
                        }
                    }
                }
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
            let mut finished = lb.finish_with(Terminator::Ret(None));
            // TEMP-MEXIT: return markers pair with the MTRACE entry
            // above, so a crash dump reconstructs the managed call
            // stack (unmatched trailing M entries = live frames).
            // Removed before close. `@f{idx}` matches the entry
            // marker (emission function index; see TEMP-FIDXDIAG).
            if std::env::var("RUBYC_MTRACE").unwrap_or_default() == "all" {
                let marker = format!("R {}.{}@f{}\n", class.name, method.name, idx);
                let (off, len) = strings.literal(&marker);
                for bl in finished.blocks.iter_mut() {
                    if matches!(bl.term, Terminator::Ret(_)) {
                        bl.instrs.push(Instr::EWriteStr { off, len });
                    }
                }
            }
            // TEMP-BIGFN: flag giant functions (inlining bloat).
            // ALSO prints every method start (last line before an
            // overflow names the culprit).
            if std::env::var("RUBYC_BIGFN").is_ok() && finished.vregs > 5000 {
                eprintln!(
                    "[rubyc] TEMP-BIGFN {}.{} vregs={} blocks={}",
                    class.name, method.name, finished.vregs, finished.blocks.len()
                );
            }
            if dump_ir {
                eprintln!("[rubyc] TEMP-IRDUMP {}.{}({}) vregs={} blocks={}",
                    class.name, method.name,
                    method.params.iter().map(|p| format!("{}:{}", p.name, p.type_name)).collect::<Vec<_>>().join(", "),
                    finished.vregs, finished.blocks.len());
                for (bi, bl) in finished.blocks.iter().enumerate() {
                    eprintln!("  [block {bi}] term={:?}", bl.term);
                    for ins in &bl.instrs {
                        eprintln!("    {ins:?}");
                    }
                }
            }
            functions.push(finished);
            // TEMP-NULLBASE-IR: scan the finished function for
            // LoadField/StoreField off a vreg that is provably Const 0
            // (single-block dataflow is enough for the prologue
            // patterns). Removed before close.
            if std::env::var("RUBYC_NULLBASE").is_ok() {
                let last_fn = functions.last().expect("just pushed");
                for bl in &last_fn.blocks {
                    let mut consts: std::collections::BTreeMap<u32, i64> =
                        Default::default();
                    for ins in &bl.instrs {
                        match ins {
                            Instr::Const { dst, imm } => {
                                consts.insert(dst.0, *imm);
                            }
                            Instr::Mov { dst, a } => {
                                if let Some(v) = consts.get(&a.0).copied() {
                                    consts.insert(dst.0, v);
                                } else {
                                    consts.remove(&dst.0);
                                }
                            }
                            Instr::LoadField { dst, base, off } => {
                                if consts.get(&base.0) == Some(&0) {
                                    eprintln!(
                                        "[rubyc] TEMP-NULLBASE-IR load off={off} in {}.{}",
                                        class.name, method.name
                                    );
                                }
                                consts.remove(&dst.0);
                            }
                            _ => {}
                        }
                    }
                }
            }
            fn_index.entry(fname).or_default().push(idx);

            // Entry-point candidates: C#-style `Main` preferred over the
            // legacy RubyC-flavored lowercase `main`.
            match method.name.as_str() {
                "Main" => {
                    if main_upper.is_some() {
                        return Err(LowerError::UnsupportedConstruct(crate::tf!(
                            "rubyc.native.ir.entry.ambiguous"
                        )));
                    }
                    main_upper = Some((class.name.clone(), idx, takes_argv(method), method.is_static));
                }
                "main" => {
                    main_lower.get_or_insert((class.name.clone(), idx, takes_argv(method), method.is_static));
                }
                _ => {}
            }
        }
    }

    // Vtable rows resolve against the COMPLETE function index: a
    // derived row's slots fill from base implementations that may
    // sort later in declaration order (a System.Console class
    // deriving a CoreLib base), so building each row inline right
    // after its own methods silently drops every late-sorting base
    // implementation (null slots — faulted `ConsoleEncoding`
    // construction on `SetDefaultFallbacks`). Row order is unchanged
    // (same class list, same order), so indices are stable.
    for class in &all_classes {
        let Some(_info) = table.class(&class.name) else {
            continue;
        };

        // Vtable row for this class: nearest implementing function index
        // per slot; `usize::MAX` marks abstract holes. Slots are the
        // entry's own recorded positions (per-class positional
        // numbering: base-chain order first, so derived rows share
        // their bases' prefix — stable across builds from the same
        // sources, unlike global first-seen). Each distinct position
        // resolves its nearest implementation independently, so
        // overloads/hides sharing a name each land correctly.
        let entry_slots: Vec<(usize, String, Vec<String>)> = table
            .class(&class.name)
            .map(|e| {
                e.methods
                    .iter()
                    .map(|m| (m.vtable_slot, m.name.clone(), m.param_types.clone()))
                    .collect()
            })
            .unwrap_or_default();
        let max_slot = entry_slots.iter().map(|(s, _, _)| *s).max().unwrap_or(0);
        let mut row = vec![usize::MAX; max_slot + 1];
        for (slot, mname, psig) in &entry_slots {
            // Preserve abstract holes: only fill when an implementation
            // actually resolves (a slot with no body anywhere stays MAX
            // instead of collapsing to an unrelated index 0 function).
            if let Some(idx) = impl_index(&all_classes, &table, &class.name, mname, psig, &fn_index) {
                row[*slot] = idx;
            }
        }
        let row_len = row.len();
        vtables.push((
            class_paths
                .get(&(*class as *const ClassDecl))
                .cloned()
                .unwrap_or_else(|| class.name.clone()),
            row,
        ));
        // TEMP-VIDBG: row index audit.
        if std::env::var("RUBYC_VIDBG").is_ok()
            && (class.name == "UTF8Encoding" || class.name.ends_with(".UTF8Encoding"))
        {
            eprintln!(
                "[rubyc] TEMP-VIDBG row {} slots={}",
                vtables.len() - 1,
                row_len
            );
        }
    }

    // Synthetic placeholder-type row: `typeof(X)` boxes carry this row
    // instead of X's own, so virtual calls on placeholders (which have
    // no real implementation) land on a zero stub instead of executing
    // a foreign slot or faulting past a short row (e.g.
    // `CreateDefaultEqualityComparer` calling Type virtuals on a
    // `typeof(object)` box faulted reading past Object's 4-entry row).
    // Every slot answers the C# default (zero/false/null) — the
    // documented placeholder grade, crash-free. Appended last, so no
    // existing function index shifts (direct-call site indices stay
    // valid); the zero function is vtable-referenced, hence a DCE
    // root (kept). `typeof` sites compute this index as the
    // table-backed declaration count (see `placeholder_row` in
    // lowering). NOTE: `functions.last()` is no longer main — read
    // entry points by name.
    let zero_idx = functions.len() + 1;
    {
        let mut zf = IrFunction::new("__placeholder_zero", ReceiverKind::None, 0);
        let dst = zf.fresh_vreg();
        zf.blocks[0].instrs.push(Instr::Const { dst, imm: 0 });
        zf.blocks[0].term = Terminator::Ret(Some(dst));
        functions.push(zf);
    }
    {
        let maxlen = vtables
            .iter()
            .map(|(_, r)| r.len())
            .max()
            .unwrap_or(0)
            .max(1);
        vtables.push(("__placeholder_type".to_owned(), vec![zero_idx; maxlen]));
    }

    // ---- @start entry (executables only)
    // `Main` (C#-style) wins over legacy lowercase `main`.
    let main_fn = main_upper.clone().or(main_lower.clone());
    let Some((entry_class, main_idx, main_argv, main_static)) = main_fn else {
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
                local_fns: &local_table,
                current_method: None,
                current_ret: None,
            };
            let mut inf = IrFunction::new("@init", ReceiverKind::None, 0);
            let mut ib = lower::Body::new(&mut inf, &mut strings);
            ib.scope_mut().push();

    for class in &all_classes {
                for m in &class.members {
                    // Static fields and static auto-properties alike
                    // (textual order; see the `@start` loop).
                    let (mname, minit, mpre, is_prop): (&String, &crate::models::Expr, Option<&Vec<crate::models::Stmt>>, bool) = match m {
                        ClassMember::Field(fd) if fd.is_static => {
                            (&fd.name, &fd.init, Some(&fd.init_pre), false)
                        }
                        ClassMember::Property(pd)
                            if pd.is_static
                                && pd.get.is_none()
                                && pd.set.is_none()
                                && pd.get_expr.is_none() =>
                        {
                            (&pd.name, &pd.init, None, true)
                        }
                        _ => continue,
                    };
                    let Some(&idx) =
                        static_index.get(&(class.name.clone(), mname.clone()))
                    else {
                        continue;
                    };
                    // Same declaring-class viewer rule as `@start`.
                    let field_ctx = Ctx {
                        current_class: &class.name,
                        current_method: init_ctx.current_method.clone(),
                        current_ret: None,
                        ..init_ctx
                    };
                    // Bindings declared while parsing the initializer
                    // (`out` vars; fields only) lower first so the
                    // init sees them.
                    if let Some(pres) = mpre {
                        for pre in pres {
                            ib.statement(pre, &field_ctx)?;
                        }
                    }
                    let cp = ib.checkpoint();
                    let v = match lower::eval(minit, &mut ib, &field_ctx) {
                        Ok(v) => v,
                        Err(e) if is_prop => {
                            ib.restore(cp);
                            eprintln!(
                                "[rubyc] warning: static initializer for {}.{} failed to lower ({}); leaving zeroed storage",
                                class.name, mname, e
                            );
                            ib.const_i64(0)
                        }
                        Err(e) => return Err(e),
                    };
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
        // `all_classes` (flattened, nested included), NOT top-level
        // `classes`: `vtable_row_for` counts this list to index the
        // vtable vec (built over `all_classes`), and top-level-only
        // counting undercounts past nested classes — static-init
        // struct placeholders got Object's row, misdispatching
        // every virtual call on them (e.g. `DateTime.MinValue`
        // placeholder dispatching `AddMilliseconds` to
        // `Object.ToString`).
        classes: &all_classes,
        import_index: &import_index,
        dep_imports: &dep_imports,
        host_import_count: table.imports.len(),
        static_index: &static_index,
        defines,
        method_vars: &[],
        local_fns: &local_table,
        current_method: None,
        current_ret: None,
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
    //
    // Dependency order (see `init_class_order`): eager linear
    // initialization must respect initializer dependencies — a static
    // whose initializer reads another class's static observes zeroed
    // storage when that class sorts later in declaration order
    // (`DateTimeFormat.InvariantFormatInfo` reading
    // `CultureInfo.InvariantCulture`). C# runs type initializers
    // lazily on first access; the closest eager equivalent is
    // dependency order with declaration-order fallback for cycles
    // (which matches today's behavior there, and .NET's
    // partially-initialized semantics closely enough for smoke).
    for class in init_class_order(&all_classes, &table) {
        for m in &class.members {
            // Static fields AND static auto-properties initialize here,
            // in textual order (C# runs both in declaration order).
            // Instance members never do (field initializers run at
            // `new`); explicit-accessor properties never do (no
            // storage, nothing to run). Without the property arm,
            // static auto-properties (e.g. COM strategy singletons)
            // stayed null forever and faulted first use.
            // `is_prop` marks the property arm: its initializers may
            // use `out`-var declarations the parser only drains for
            // fields (proper support is future work) — on lowering
            // failure they fail soft to zeroed storage with a loud
            // warning (same divergence grade as swallowed runtime
            // throws), never a build break. Fields keep hard errors.
            let (mname, minit, mpre, is_prop): (
                &String,
                &crate::models::Expr,
                Option<&Vec<crate::models::Stmt>>,
                bool,
            ) = match m {
                ClassMember::Field(fd) if fd.is_static => {
                    (&fd.name, &fd.init, Some(&fd.init_pre), false)
                }
                ClassMember::Property(pd)
                    if pd.is_static
                        && pd.get.is_none()
                        && pd.set.is_none()
                        && pd.get_expr.is_none() =>
                {
                    (&pd.name, &pd.init, None, true)
                }
                _ => continue,
            };
            // TEMP-STATICDIAG3: audit static-index misses (a miss
            // silently skips the initializer — zeroed storage
            // forever). Removed before close.
            if std::env::var("RUBYC_STATICDIAG").is_ok()
                && (class.name.contains("CultureInfo")
                    && !class.name.contains("Data")
                    // TEMP-SENTDIAG3. Removed before close.
                    || class.name.contains("ManualResetValueTaskSourceCoreShared")
                    // TEMP-SWDIAG4. Removed before close.
                    || class.name.contains("TextWriter")
                    || class.name.contains("Environment"))
            {
                eprintln!(
                    "[rubyc] TEMP-STATICDIAG3 class={} field={} has_idx={}",
                    class.name,
                    mname,
                    static_index.contains_key(&(class.name.clone(), mname.clone()))
                );
            }
            let Some(&idx) = static_index.get(&(class.name.clone(), mname.clone())) else {
                continue;
            };
            // TEMP-INITTRACE: mark each EXECUTED static initializer on
            // stderr so a throw/crash during @start identifies the
            // culprit by its last line. RUBYC_SKIP_INIT (;-separated
            // substrings of "Class::field") skips matching initializers
            // (leaving zeroed storage) to walk past throwers. Gated on
            // RUBYC_TRACE_INIT; all removed before close.
            if let Ok(skip) = std::env::var("RUBYC_SKIP_INIT") {
                let key = format!("{}::{}", class.name, mname);
                if !skip.is_empty() && skip.split(';').any(|s| !s.is_empty() && key.contains(s)) {
                    continue;
                }
            }
            // TEMP-INITTRACE: zero instead of evaluating (keep the
            // store) — splits eval-corruption from store-corruption.
            // Removed before close.
            let zero_init = std::env::var("RUBYC_ZERO_INIT").map(|z| {
                let key = format!("{}::{}", class.name, mname);
                !z.is_empty() && z.split(';').any(|s| !s.is_empty() && key.contains(s))
            }).unwrap_or(false);
            // TEMP-INITTRACE: allowlist — only initialize matching
            // classes (semicolon substrings). Removed before close.
            if let Ok(only) = std::env::var("RUBYC_ONLY_INIT") {
                let key = format!("{}::{}", class.name, mname);
                if !only.is_empty() && !only.split(';').any(|s| !s.is_empty() && key.contains(s)) {
                    continue;
                }
            }
            if std::env::var("RUBYC_TRACE_INIT").is_ok() {
                let marker = format!("INIT {}::{}\n", class.name, mname);
                let (off, len) = sl.string_literal(&marker);
                sl.push_raw(Instr::EWriteStr { off, len });
            }
            // The initializer is the declaring class's own code (a
            // private member call inside it is legal): evaluate with
            // that class as viewer, not the entry class.
              let field_ctx = Ctx {
                current_class: &class.name,
                current_method: start_ctx.current_method.clone(),
                current_ret: None,
                ..start_ctx
            };
            // Throwing initializers must not kill the executable:
            // real runtimes run type initializers lazily on first
            // access, but @start runs everything eagerly up front, so
            // a single throwing static (a resource lookup, a host
            // switch, a not-yet-implemented path) would abort the
            // program before Main. Wrap each init so a throw leaves
            // zeroed storage instead (the default(T) the field would
            // hold had its initializer never run). Swallowing is a
            // documented divergence, limited to static-init; ordinary
            // method bodies keep exact throw semantics.
            // TEMP-INITTRACE: RUBYC_NO_WRAP disables the swallowing
            // wrapper (plain eval+store). Removed before close.
            let no_wrap = std::env::var("RUBYC_NO_WRAP").is_ok();
            let (body_b, join_b, exc_b) = if no_wrap {
                (None, None, None)
            } else {
                let bb = sl.make_block();
                let jb = sl.make_block();
                let eb = sl.make_block();
                let h = sl.fresh();
                sl.push_raw(Instr::HandlerEnter { dst: h });
                sl.terminate(Terminator::Branch {
                    cond: Some(h),
                    if_true: eb,
                    if_false: bb,
                });
                sl.switch_to(bb);
                (Some(bb), Some(jb), Some(eb))
            };
            // Bindings declared while parsing the initializer
            // (`out` vars; fields only — property initializers carry
            // none) lower first so the init sees them.
            if let Some(pres) = mpre {
                for pre in pres {
                    sl.statement(pre, &field_ctx)?;
                }
            }
            let v = if zero_init {
                sl.const_i64(0)
            } else {
                let cp = sl.checkpoint();
                match lower::eval(minit, &mut sl, &field_ctx) {
                    Ok(v) => v,
                    Err(e) if is_prop => {
                        // Loud fail-soft (see the arm comment): the
                        // executable builds with zeroed storage for
                        // this static instead of aborting the build.
                        sl.restore(cp);
                        eprintln!(
                            "[rubyc] warning: static initializer for {}.{} failed to lower ({}); leaving zeroed storage",
                            class.name, mname, e
                        );
                        sl.const_i64(0)
                    }
                    Err(e) => return Err(e),
                }
            };
            sl.store_static(idx, v);
            if !no_wrap {
                sl.push_raw(Instr::HandlerPop);
                sl.goto(join_b.unwrap());
                sl.switch_to(exc_b.unwrap());
                let e = sl.fresh();
                sl.push_raw(Instr::LoadExc { dst: e });
                sl.goto(join_b.unwrap());
                sl.switch_to(join_b.unwrap());
            }
            let _ = (body_b, join_b, exc_b);
        }
    }
    let obj = sl.fresh();
    // TEMP-INITTRACE: mark @start completion (inits done, Main next).
    if std::env::var("RUBYC_TRACE_INIT").is_ok() {
        let (off, len) = sl.string_literal("MAIN\n");
        sl.push_raw(Instr::EWriteStr { off, len });
    }
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
        // NOTE: `const` fields are compile-time only (no storage):
        // emitting them here would store e.g. `CharSize = 2` at
        // offset 0, clobbering the entry object's vtable word (see
        // the same fix in the `new` path in `lower/eval.rs`).
        ClassMember::Field(fd) if !fd.is_static && !fd.is_const => Some(fd),
        _ => None,
    }) {
        let v = lower::eval(&fd.init, &mut sl, &start_ctx)?;
        let off = (info.field(&fd.name).map(|fi| fi.offset_slots).unwrap_or(1) * 8) as u32;
        sl.push_raw(Instr::StoreField { base: obj, off, src: v });
    }

    let main_ret = if main_argv {
        // `Main(string[] args)`: materialize process argv (kernel stack
        // via the backend's startup slot) and pass it after the receiver,
        // like any other second parameter. Static entries take no
        // receiver (static ABI packs params from rdi), so they receive
        // [argv] alone — passing `obj` first shifts every param by one
        // (`args` aliased the receiver: Length read its vtable word).
        let argv = sl.fresh();
        sl.push_raw(Instr::BuildArgv { dst: argv });
        if main_static {
            sl.call_fn(main_idx, vec![argv])
        } else {
            sl.call_fn(main_idx, vec![obj, argv])
        }
    } else {
        sl.call_fn(main_idx, vec![obj])
    };
    // @start has no caller: terminate with exit_group, never a ret.
    // An int-returning entry point becomes the process exit code.
    sl.exit_with(main_ret);

    // TEMP-STARTDUMP: print the @start function IR (static-init
    // prologue) for init-evaluation debugging. Removed before close.
    if std::env::var("RUBYC_IRDUMP").unwrap_or_default().split(',').any(|w| w.trim() == "@start") {
        eprintln!("[rubyc] TEMP-STARTDUMP @start vregs={} blocks={}", sl.f.vregs, sl.f.blocks.len());
        for (bi, bl) in sl.f.blocks.iter().enumerate() {
            eprintln!("  [block {bi}] term={:?}", bl.term);
            for ins in &bl.instrs {
                eprintln!("    {ins:?}");
            }
        }
    }

    functions.insert(0, sl.finish_with(Terminator::Exit));

    // TEMP-FNMAP: index to name for DCE drop correlation.
    // Removed before close.
    if std::env::var("RUBYC_DCEAUDIT").unwrap_or_default().contains("list") {
        for (i, f) in functions.iter().enumerate() {
            if f.name.contains("UTF8Encoding")
                || f.name.contains("SetDefaultFallbacks")
                || f.name.contains("ThrowIfNegative")
                || f.name.contains("IsNullOrEmpty")
                || f.name.contains("EncoderReplacementFallback")
            {
                eprintln!("[rubyc] TEMP-FNMAP idx={i} {}", f.name);
            }
        }
    }

    // TEMP-FIDXDIAG: resolve call-site fidx to function names
    // (bisecting wrong-target static calls). `RUBYC_FIDXDIAG=1,2,3`.
    // Removed before close.
    if let Ok(spec) = std::env::var("RUBYC_FIDXDIAG") {
        for w in spec.split(',') {
            if let Ok(i) = w.trim().parse::<usize>() {
                eprintln!(
                    "[rubyc] TEMP-FIDXDIAG idx={i} -> {}",
                    functions.get(i).map(|f| f.name.as_str()).unwrap_or("<OOB>")
                );
            }
        }
    }
    if std::env::var("RUBYC_VTROWDBG").is_ok() {
        // Reverse index for slot inspection.
        let mut names_by_idx: std::collections::BTreeMap<usize, &str> = Default::default();
        for class in &all_classes {
            for m in &class.members {
                if let ClassMember::Method(mm) = m {
                    let key = format!("{}::{}#{}", class.name, mm.name, mm.params.len());
                    if let Some(fn_vec) = fn_index.get(&key) {
                        for fidx in fn_vec {
                            names_by_idx.insert(*fidx, &mm.name);
                        }
                    }
                }
            }
        }
        for (i, (n, row)) in vtables.iter().enumerate() {
            // TEMP-GBSLOT-OUTER: let the GetBytes audit run standalone.
            // Removed before close.
            if n.contains("UTF8Encoding") || n.contains("EncoderFallback") || std::env::var("RUBYC_GBSLOT").is_ok() || std::env::var("RUBYC_VTROWDBG").is_ok_and(|v| v.contains("DateTime")) && n.contains("DateTime") || std::env::var("RUBYC_VTROWDBG").is_ok_and(|v| v == "ALL") || std::env::var("RUBYC_VTROWDBG2").is_ok_and(|v| v.contains("Encoding")) && (n == "System.Text.Encoding" || n == "System.Text.Latin1Encoding" || n == "System.Text.UnicodeEncoding" || n == "System.Text.ConsoleEncoding" || n == "Encoding" || n == "Latin1Encoding" || n == "UnicodeEncoding" || n == "ConsoleEncoding") {
                // TEMP-VTROWDBG2: entry-slot audit (slot
                // disagreement between table and emitted rows).
                // Removed before close.
                // TEMP-SENTDIAG5: include ConsoleEncoding (bisecting the
                // SetDefaultFallbacks slot-35 fault). Removed before close.
                // TEMP-GBSLOT: dump every GetBytes slot+params for the
                // encoder misroute (slot 17 vs true base-5 slot).
                // Removed before close.
                if std::env::var("RUBYC_GBSLOT").is_ok()
                    && (n == "System.Text.Encoding" || n == "Encoding")
                {
                    for e in table.classes.iter().filter(|c| c.key.path == *n) {
                        for m in e.methods.iter().filter(|m| m.name == "GetBytes") {
                            eprintln!(
                                "[rubyc] TEMP-GBSLOT {}.GetBytes({:?}) slot={} static={} body={}",
                                e.key.path,
                                m.param_types,
                                m.vtable_slot,
                                m.is_static,
                                m.has_body
                            );
                        }
                    }
                }
                // TEMP-GBSLOT2: same for the sealed rows (is base-5
                // inherited into the entry?).
                // Removed before close.
                if std::env::var("RUBYC_GBSLOT").is_ok()
                    && n.contains("Sealed")
                {
                    for e in table.classes.iter().filter(|c| c.key.path == *n) {
                        for m in e.methods.iter().filter(|m| m.name == "GetBytes") {
                            eprintln!(
                                "[rubyc] TEMP-GBSLOT2 {}.GetBytes({:?}) slot={} body={}",
                                e.key.path,
                                m.param_types,
                                m.vtable_slot,
                                m.has_body
                            );
                        }
                    }
                }
                if std::env::var("RUBYC_VTROWDBG2").is_ok_and(|v| v.contains("Encoding")) && (n == "System.Text.Encoding" || n == "System.Text.Latin1Encoding" || n == "System.Text.UnicodeEncoding" || n == "System.Text.ConsoleEncoding" || n == "Encoding" || n == "Latin1Encoding" || n == "UnicodeEncoding" || n == "ConsoleEncoding")
                {
                    let matches: Vec<String> = table.classes.iter().filter(|c| c.key.path == *n).map(|c| format!("{}[{}]", c.name, c.methods.len())).collect();
                    eprintln!("[rubyc] TEMP-VTROWDBG2 rowname={n} entries={:?}", matches);
                    for e in table.classes.iter().filter(|c| c.key.path == *n) {
                        let mut parts: Vec<String> = Vec::new();
                        for m in e.methods.iter().take(6) {
                            parts.push(format!("{}#{}:{}", m.name, m.param_types.len(), m.vtable_slot));
                        }
                        let slot = |nm: &str| e.methods.iter().find(|m| m.name == nm).map(|m| m.vtable_slot.to_string()).unwrap_or("MISSING".to_owned());
                        let has_gettype = e.methods.iter().any(|m| m.name == "GetType");
                        let has_sdf = e.methods.iter().any(|m| m.name == "SetDefaultFallbacks");
                        eprintln!("[rubyc] TEMP-VTROWDBG2 entry={} methods={} slot_GetType={} slot_SDF={} slot_GetEnc={} first6=[{}]", e.key.path, e.methods.len(), slot("GetType"), slot("SetDefaultFallbacks"), slot("GetEncodings"), parts.join(" "));
                    }
                }
                let slots: Vec<String> = row
                    .iter()
                    .enumerate()
                    .map(|(s, f)| {
                        format!(
                            "{s}={}:{}",
                            f,
                            names_by_idx.get(f).copied().unwrap_or("?")
                        )
                    })
                    .collect();
                eprintln!("[rubyc] TEMP-VTROWDBG row={i} {n} [{}]", slots.join(" "));
            }
        }
    }

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
        // Executable heap: eager `@start` initialization allocates
        // every string literal and constructed static in the
        // linked closure (CoreLib alone needs megabytes) before
        // `Main` runs. 64 KiB starved mid-init and faulted on
        // unmapped heap extension; 64 MiB did the same once
        // indirect-call routing started reaching bigger init
        // closures (casing/compare tables during static init bumped
        // exactly to the 64 MiB line). 256 MiB is BSS (no file
        // cost, kernel-overcommitted) with headroom for real
        // programs; revisit if a leak (not just init weight) shows.
        heap_size: 256 * 1024 * 1024,
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
    fn_index: &std::collections::BTreeMap<String, Vec<usize>>,
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
                && let Some(&idx) = fn_index
                    .get(&format!("{}::{}#{}", class.name, name, arity))
                    .and_then(|v| v.last())
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
    fn_index: &std::collections::BTreeMap<String, Vec<usize>>,
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
            if let Some(&idx) = fn_index
                .get(&format!("{}::{}#{}", entry.name, m.name, arity))
                .and_then(|v| v.last())
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
