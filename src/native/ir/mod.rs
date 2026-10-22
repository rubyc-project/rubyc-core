//! Virtual-register IR and AST lowering.
//!
//! [`instructions`] defines the three-address form the backend consumes;
//! [`lower`] translates typed AST into it. See [`instructions`] for the
//! representation contract.

mod instructions;
mod lower;
mod serialize;
mod verify;

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
use crate::native::semantic::{
    self, ClassInfo, INSTANCE_HEADER_SLOTS, MethodInfo, SemanticError, TypeTable,
};

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
    /// Local constants visible at the declaration. A static local
    /// function may use them (a constant is not a captured variable);
    /// they are materialized in its own frame.
    pub consts: Vec<(String, String, Expr)>,
    /// `extern` local function: no body of its own. With an
    /// `[UnsafeAccessor]` attribute it names a target method.
    pub is_extern: bool,
    /// Attributes on the declaration (`[UnsafeAccessor(...)]`).
    pub attrs: Vec<crate::models::Attribute>,
    /// The ENCLOSING method's `where` clauses. A local function
    /// inherits the enclosing method's type parameters, so it inherits
    /// their constraints too — a member on one of those parameters
    /// binds through them.
    pub constraints: Vec<crate::models::ast::method::TypeConstraint>,
    /// The enclosing class's QUALIFIED path. A local function's body
    /// belongs to that class, and the access rule compares paths (a
    /// bare class name is ambiguous for a nested type).
    pub class_path: Option<String>,
}

/// How many lambda functions a method body will add to `Program::functions`.
///
/// A lifted lambda is emitted immediately after the method that created it,
/// so every method after the first lambda shifts by one. The function-index
/// map is a *prediction* built before lowering starts (call sites need the
/// index of a method that has not been emitted yet), so it has to know this
/// count up front — without it, `new Both()` resolved its base constructors
/// to whichever functions happened to sit at the predicted slots, and a
/// program containing a lambda ran the wrong code.
///
/// Local functions are deliberately not descended into: their bodies are
/// lowered with `lambda_deferred: None`, so a lambda there is reported
/// rather than lifted and cannot shift anything.
pub(crate) fn count_lambdas(block: &crate::models::Block) -> usize {
    fn walk_block(b: &crate::models::Block) -> usize {
        b.stmts.iter().map(walk_stmt).sum()
    }
    fn walk_stmt(s: &crate::models::Stmt) -> usize {
        use crate::models::Stmt as S;
        match s {
            S::Let { init, .. } | S::UsingDecl { init, .. } => walk_expr(init),
            S::Assign { value, .. } => walk_expr(value),
            S::MemberAssign { object, value, .. } => walk_expr(object) + walk_expr(value),
            S::Expression(e) => walk_expr(e),
            S::CheckedBlock { body, .. } => walk_block(body),
            S::Block { body, .. } => walk_block(body),
            S::Return { value, .. } => value.as_ref().map(walk_expr).unwrap_or(0),
            S::If {
                cond,
                then_block,
                else_block,
                ..
            } => {
                walk_expr(cond)
                    + walk_block(then_block)
                    + else_block.as_ref().map(walk_block).unwrap_or(0)
            }
            S::While { cond, body, .. } => walk_expr(cond) + walk_block(body),
            S::DoWhile { body, cond, .. } => walk_block(body) + walk_expr(cond),
            S::For {
                init,
                cond,
                step,
                body,
                ..
            } => {
                init.as_deref().map(walk_stmt).unwrap_or(0)
                    + cond.as_ref().map(walk_expr).unwrap_or(0)
                    + step.as_deref().map(walk_stmt).unwrap_or(0)
                    + walk_block(body)
            }
            S::Switch {
                subject,
                arms,
                default_body,
                ..
            } => {
                walk_expr(subject)
                    + arms
                        .iter()
                        .map(|a| {
                            a.labels.iter().map(walk_expr).sum::<usize>()
                                + walk_block(&a.body)
                        })
                        .sum::<usize>()
                    + default_body.as_ref().map(walk_block).unwrap_or(0)
            }
            S::Goto { case_expr, .. } => case_expr.as_ref().map(walk_expr).unwrap_or(0),
            S::Fixed {
                declarators, body, ..
            } => {
                declarators
                    .iter()
                    .map(|(_, _, init)| init.as_ref().map(walk_expr).unwrap_or(0))
                    .sum::<usize>()
                    + walk_block(body)
            }
            S::Try {
                body,
                catches,
                finally_block,
                ..
            } => {
                walk_block(body)
                    + catches.iter().map(|c| walk_block(&c.body)).sum::<usize>()
                    + finally_block.as_ref().map(walk_block).unwrap_or(0)
            }
            S::Throw { expr, .. } => expr.as_ref().map(walk_expr).unwrap_or(0),
            S::LocalFunction { .. } => 0,
            S::Break { .. } | S::Continue { .. } | S::Label { .. } => 0,
        }
    }
    fn walk_expr(e: &crate::models::Expr) -> usize {
        use crate::models::Expr as E;
        match e {
            E::NamedArg { value, .. } => walk_expr(value),
            // The lambda itself is one function; its body adds more.
            E::Lambda { body, .. } => {
                1 + match body {
                    crate::models::ast::expression::LambdaBody::Expr(inner) => walk_expr(inner),
                    crate::models::ast::expression::LambdaBody::Block(b) => walk_block(b),
                }
            }
            E::Checked { inner, .. } => walk_expr(inner),
            E::Binary { lhs, rhs, .. } => walk_expr(lhs) + walk_expr(rhs),
            E::Unary { inner, .. } => walk_expr(inner),
            E::Ternary {
                cond,
                then_expr,
                else_expr,
                ..
            } => walk_expr(cond) + walk_expr(then_expr) + walk_expr(else_expr),
            E::Call { args, .. } => args.iter().map(walk_expr).sum(),
            E::New { args, init, .. } => {
                args.iter().map(walk_expr).sum::<usize>()
                    + init.iter().map(|m| walk_expr(&m.value)).sum::<usize>()
            }
            E::Member { object, .. } => walk_expr(object),
            E::MethodCall { object, args, .. } => {
                walk_expr(object) + args.iter().map(walk_expr).sum::<usize>()
            }
            E::Await { inner, .. } => walk_expr(inner),
            E::NewArray { size, init, .. } => {
                walk_expr(size) + init.iter().map(walk_expr).sum::<usize>()
            }
            E::Index { object, index, .. } => walk_expr(object) + walk_expr(index),
            E::Cast { expr, .. } => walk_expr(expr),
            E::AssignComplex { lhs, value, .. } => walk_expr(lhs) + walk_expr(value),
            E::Assign { target, value, .. } => walk_expr(target) + walk_expr(value),
            E::PostInc { target, .. } | E::PostDec { target, .. } => walk_expr(target),
            E::SwitchExpr { subject, arms, .. } => {
                walk_expr(subject) + arms.iter().map(|a| walk_expr(&a.body)).sum::<usize>()
            }
            E::InterpolatedString { parts, .. } => parts
                .iter()
                .map(|p| match p {
                    crate::models::InterpPart::Hole(h) => walk_expr(h),
                    crate::models::InterpPart::Text(_) => 0,
                })
                .sum(),
            E::Ident { .. } | E::Literal(_) | E::Default { .. } => 0,
        }
    }
    walk_block(block)
}

/// Collect static local functions in deterministic order (classes
/// → methods → source order, descending into nested blocks).
/// Instance, async, generic, bodiless, or fancy-param locals stay
/// on the legacy placeholder path (not collected).
pub(crate) fn collect_static_locals(
    classes: &[&ClassDecl],
    class_paths: &std::collections::HashMap<*const ClassDecl, String>,
) -> Vec<LocalFnDecl> {
    fn walk_block(
        out: &mut Vec<LocalFnDecl>,
        classes: &[&ClassDecl],
        class_paths: &std::collections::HashMap<*const ClassDecl, String>,
        owner: &ClassDecl,
        class_name: &str,
        method_name: &str,
        method_vars: &[String],
        consts: &mut Vec<(String, String, Expr)>,
        block: &AstBlock,
    ) {
        let mark = consts.len();
        for stmt in &block.stmts {
            match stmt {
                AstStmt::Let {
                    is_const: true,
                    type_name,
                    name,
                    init,
                    ..
                } => {
                    consts.push((name.clone(), type_name.clone(), init.clone()));
                }
                AstStmt::LocalFunction {
                    name,
                    params,
                    ret_type,
                    body,
                    is_static,
                    is_extern,
                    attrs,
                    ..
                } => {
                    if *is_static {
                        // The enclosing method's constraints: a local
                        // function inherits its type parameters, so it
                        // inherits their constraints.
                        let constraints = classes
                            .iter()
                            .find(|c| c.name == class_name)
                            .and_then(|c| {
                                c.members.iter().find_map(|m| match m {
                                    crate::models::ClassMember::Method(md)
                                        if md.name == *method_name =>
                                    {
                                        Some(md.constraints.clone())
                                    }
                                    _ => None,
                                })
                            })
                            .unwrap_or_default();
                        out.push(LocalFnDecl {
                            class_name: class_name.to_owned(),
                            method_name: method_name.to_owned(),
                            local_name: name.clone(),
                            params: params.clone(),
                            ret_type: ret_type.clone(),
                            body: body.clone(),
                            method_vars: method_vars.to_owned(),
                            consts: consts.clone(),
                            is_extern: *is_extern,
                            attrs: attrs.clone(),
                            constraints,
                            class_path: class_paths
                                .get(&(owner as *const ClassDecl))
                                .cloned(),
                        });
                    }
                    let mut inner = consts.clone();
                    walk_block(out, classes, class_paths, owner, class_name, method_name, method_vars, &mut inner, body);
                }
                AstStmt::Block { body, .. } => {
                    walk_block(out, classes, class_paths, owner, class_name, method_name, method_vars, consts, body)
                }
                AstStmt::If {
                    then_block,
                    else_block,
                    ..
                } => {
                    walk_block(out, classes, class_paths, owner, class_name, method_name, method_vars, consts, then_block);
                    if let Some(b) = else_block {
                        walk_block(out, classes, class_paths, owner, class_name, method_name, method_vars, consts, b);
                    }
                }
                AstStmt::While { body, .. } | AstStmt::DoWhile { body, .. } => {
                    walk_block(out, classes, class_paths, owner, class_name, method_name, method_vars, consts, body)
                }
                AstStmt::For { body, .. } => {
                    walk_block(out, classes, class_paths, owner, class_name, method_name, method_vars, consts, body)
                }
                AstStmt::Switch {
                    arms, default_body, ..
                } => {
                    for arm in arms {
                        walk_block(out, classes, class_paths, owner, class_name, method_name, method_vars, consts, &arm.body);
                    }
                    if let Some(b) = default_body {
                        walk_block(out, classes, class_paths, owner, class_name, method_name, method_vars, consts, b);
                    }
                }
                AstStmt::Try {
                    body,
                    catches,
                    finally_block,
                    ..
                } => {
                    walk_block(out, classes, class_paths, owner, class_name, method_name, method_vars, consts, body);
                    for c in catches {
                        walk_block(out, classes, class_paths, owner, class_name, method_name, method_vars, consts, &c.body);
                    }
                    if let Some(b) = finally_block {
                        walk_block(out, classes, class_paths, owner, class_name, method_name, method_vars, consts, b);
                    }
                }
                _ => {}
            }
        }
        consts.truncate(mark);
    }

    let mut out = Vec::new();
    for class in classes {
        for m in &class.members {
            if let ClassMember::Method(method) = m {
                if let Some(body) = &method.body {
                    let mut consts = Vec::new();
                    walk_block(
                        &mut out,
                        classes,
                        class_paths,
                        class,
                        &class.name,
                        &method.name,
                        &method.generic_params,
                        &mut consts,
                        body,
                    );
                }
            }
            // Property/indexer accessors hold bodies too (`Out`'s
            // getter hides `EnsureInitialized`); locals there must
            // also emit or their calls stay placeholders.
            if let ClassMember::Property(prop) = m {
                if let Some(body) = &prop.get {
                    let mut consts = Vec::new();
                    walk_block(
                        &mut out,
                        classes,
                        class_paths,
                        class,
                        &class.name,
                        &prop.name,
                        &[],
                        &mut consts,
                        body,
                    );
                }
                if let Some(body) = &prop.set {
                    let mut consts = Vec::new();
                    walk_block(
                        &mut out,
                        classes,
                        class_paths,
                        class,
                        &class.name,
                        &prop.name,
                        &[],
                        &mut consts,
                        body,
                    );
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
    /// Lowering produced IR that fails its own consistency check (a value
    /// read on a path that never assigns it). This is a compiler bug, not
    /// a source error; failing here prevents emitting a silent miscompile.
    IrInvalid(String),
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
            LowerError::IrInvalid(detail) => write!(
                f,
                "internal compiler error: invalid IR: {detail}"
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
    /// AST declaration -> qualified path map, so a context that
    /// switches class can name it by PATH. A nested type's bare name is
    /// ambiguous (Utf8 and Utf16 both declare
    /// `TryWriteInterpolatedStringHandler`) and the table stores nested
    /// classes under their qualified name, so a bare name does not
    /// resolve at all.
    pub class_paths: &'a std::collections::HashMap<*const crate::models::ClassDecl, String>,
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
    /// Concrete substitutions for a GENERIC-INLINED call body
    /// (`Test<IntNum>` inlines `Test<T>` with `T -> IntNum`). `None`
    /// everywhere else: a method is lowered once with its type
    /// parameters erased, so `T.Member` has no runtime meaning unless a
    /// specialization supplies the concrete type.
    pub type_subst: Option<&'a [(String, String)]>,
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
    /// Lambdas encountered while lowering the current method, in the
    /// order they were met. `eval` cannot append to
    /// `Program::functions`, so it records the lambda here and returns a
    /// delegate object referring to a predicted index; the method's
    /// emission loop drains this list afterwards and emits one function
    /// per entry, which is why the prediction holds.
    ///
    /// `None` outside method bodies (field initializers, tests), where a
    /// lambda has nowhere to be lifted to and is reported.
    pub lambda_deferred:
        Option<&'a std::cell::RefCell<Vec<LambdaDeferred>>>,
    /// This method's own final function index, so a lambda can predict
    /// where its lifted function will land (`lambda_base + 1 + position`).
    pub lambda_base: u32,
}

/// One lambda waiting to be lifted into a function, recorded by `eval`
/// and emitted after the enclosing method.
///
/// OWNED rather than borrowed: `eval` sees the expression through a
/// shorter borrow than the context's `'a`, so a reference could not be
/// stored. A lambda body is small, and each is copied exactly once.
#[derive(Clone)]
pub struct LambdaDeferred {
    pub params: Vec<String>,
    pub body: crate::models::ast::expression::LambdaBody,
    pub span: crate::models::Span,
    /// Free variables, in delegate-object word order (word 8+k*8 holds
    /// the ADDRESS of the k-th capture).
    pub captures: Vec<String>,
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

    /// The class the current code is in, as the access rules see it:
    /// the QUALIFIED key path where it is known. A bare class name is
    /// ambiguous when two namespaces declare same-named nested types,
    /// and the ambiguity made the nesting-family rule give up.
    fn viewer_path(&self) -> &str {
        if let Some(p) = self.current_path
            && !p.is_empty()
        {
            return p;
        }
        if let Some(info) = self.current_info {
            return info.key.path.as_str();
        }
        &self.current_class
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
        // The owner is the class under lowering when the member is
        // read on `this`: a bare type spelling is ambiguous (25
        // same-named `Enumerator`s), so the anchored entry is the
        // precise one.
        let anchored = self
            .current_info
            .filter(|i| i.name == owner_type || i.key.path == owner_type);
        let found = match anchored {
            Some(info) => self.table.declaring_class_of_in(info, member, is_field),
            None => {
                if is_field {
                    self.table.declaring_class_of_field(owner_type, member)
                } else {
                    self.table.declaring_class_of_method(owner_type, member)
                }
            }
        };
        let Some((declaring, access)) = found else {
            return Ok(());
        };
        if semantic::visible_from(self.table, access, &declaring, self.viewer_path()) {
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
                            &self.current_class,
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
        // The viewer is named by its qualified path: a nested type's
        // simple name is ambiguous (`AlternateLookup` exists in three
        // collections), and the enclosing type's private members are
        // reachable from it (C#).
        if semantic::visible_from(self.table, mi.access, &mi.declaring_class, self.viewer_path())
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
        if semantic::visible_from(self.table, mi.access, &mi.declaring_class, &self.current_class)
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
                            &self.current_class,
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
        if let Some(decl) = self.classes.iter().find(|c| c.name == class_name) {
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
                    }
                }
            }
        }
        // Raw pointers avoid holding `self.classes` borrowed across
        // the `visit_stmt` recursion (`visit_expr` reborrows `self`
        // mutably via the `New` arm). The pointers derive from
        // `self.classes` entries, which outlive the collector.
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
            Expr::NamedArg { value, .. } => self.visit_expr(value),
            Expr::Checked { inner, .. } => self.visit_expr(inner),
            // A static initializer may contain a lambda; its body's
            // static reads still happen (lazily, but before any use), so
            // they constrain init order and must be collected.
            Expr::Lambda { body, .. } => match body {
                crate::models::ast::expression::LambdaBody::Expr(e) => {
                    self.visit_expr(e)
                }
                crate::models::ast::expression::LambdaBody::Block(b) => {
                    for st in &b.stmts {
                        self.visit_stmt(st);
                    }
                }
            },
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
    lower::clear_subclass_cache();
    lower::clear_lookup_cache();
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
                statics.push(IrStatic {
                    name: fi.name.clone(),
                    class: info.key.path.clone(),
                    type_name: fi.type_name.clone(),
                });
            }
        }
    }

    // Canonical type descriptors live in static slots after the real
    // statics. The registry only records requests during lowering; the
    // slots themselves are appended once the final count is known (a
    // `typeof` in @start can add more).
    let type_registry =
        std::cell::RefCell::new(lower::TypeRegistry::new(statics.len() as u32));

    // ---- methods



/// The `[UnsafeAccessor]` target of an extern local function: the
/// method name from the attribute, and the target type from the first
/// parameter's `[UnsafeAccessorType("Type, Assembly")]`.
fn accessor_target(
    attrs: &[crate::models::Attribute],
    params: &[crate::models::Param],
) -> Option<(String, String)> {
    let ua = attrs.iter().find(|a| {
        a.name.rsplit('.').next() == Some("UnsafeAccessor")
    })?;
    let method = ua.arguments.iter().find_map(|a| match (&a.name, &a.value) {
        (Some(n), crate::models::AttributeValue::Str(s)) if n == "Name" => Some(s.clone()),
        _ => None,
    })?;
    let type_attr = params.first()?.attrs.iter().find(|a| {
        a.name.rsplit('.').next() == Some("UnsafeAccessorType")
    })?;
    let type_str = type_attr.arguments.first().and_then(|a| match &a.value {
        crate::models::AttributeValue::Str(s) => Some(s.clone()),
        _ => None,
    })?;
    let type_name = type_str
        .split(',')
        .next()
        .unwrap_or(&type_str)
        .trim()
        .to_owned();
    Some((type_name, method))
}

/// The target's function index and whether it is static. `None` when
/// the target is not in this unit (the call then throws, as .NET does
/// for a missing accessor target).
fn resolve_accessor_target(
    ctx: &Ctx<'_>,
    type_name: &str,
    method_name: &str,
    local_arity: usize,
) -> Option<(u32, bool)> {
    let info = ctx.table.class(type_name)?;
    let mi = info.methods.iter().find(|m| m.name == method_name)?;
    // A static target drops the local's dummy receiver parameter.
    let want = if mi.is_static {
        local_arity.saturating_sub(1)
    } else {
        local_arity
    };
    if mi.param_types.len() != want {
        return None;
    }
    let key = format!("{}::{}#{}", mi.declaring_class, mi.name, mi.param_types.len());
    let idx = lower::functions_index(ctx).get(&key)?.last().copied()?;
    Some((idx as u32, mi.is_static))
}

/// Append one zeroed static slot per canonical type descriptor. The
/// slots are appended after every body is lowered, because @start and
/// @init can still request new types.
fn append_type_slots(
    registry: &std::cell::RefCell<lower::TypeRegistry>,
    statics: &mut Vec<IrStatic>,
) {
    let n = registry.borrow().len();
    for i in 0..n {
        statics.push(IrStatic {
            name: format!("__type{i}"),
            class: "@types".to_owned(),
            type_name: "object".to_owned(),
        });
    }
}

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
/// The class name and the type arguments of a generic spelling:
/// `System.Threading.Tasks.Task<int>` gives `("Task", ["int"])`.
fn split_generic_spelling(spelling: &str) -> (String, Vec<String>) {
    let s = spelling.trim();
    let Some(open) = s.find('<') else {
        return (s.to_owned(), Vec::new());
    };
    let base = &s[..open];
    let simple = base.rsplit('.').next().unwrap_or(base);
    (simple.to_owned(), lower::explicit_type_args(s))
}

/// Substitute type variables inside a spelling: `T` becomes `int`,
/// and `Func<T>?` becomes `Func<int>?`. The variables are whole
/// identifiers, so the substitution never touches a name that merely
/// starts with the same letters.
fn substitute_type_vars(spelling: &str, sub: &[(String, String)]) -> String {
    if sub.is_empty() {
        return spelling.to_owned();
    }
    let mut out = String::with_capacity(spelling.len());
    let mut rest = spelling;
    while !rest.is_empty() {
        let end = rest
            .find(|c: char| !(c.is_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        let (word, tail) = rest.split_at(end);
        match sub.iter().find(|(v, _)| v == word) {
            Some((_, concrete)) => out.push_str(concrete),
            None => out.push_str(word),
        }
        rest = tail;
    }
    out
}

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
    let local_decls = collect_static_locals(&all_classes, &class_paths);
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
                    // Same shift as `functions_index`: the ctor chain
                    // table numbers functions, so a preceding method's
                    // lifted lambdas move every later constructor.
                    n += 1 + method.body.as_ref().map(count_lambdas).unwrap_or(0);
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
        let mut f = IrFunction::new(
            &fname,
            ReceiverKind::None,
            ld.params.len() + ld.method_vars.len(),
        );
        let mut lb = lower::Body::new(&mut f, &mut strings, &type_registry);
        let lctx = Ctx {
            table: &table,
            class_paths: &class_paths,
            current_class: &ld.class_name,
            // The local function's body belongs to the enclosing class,
            // named by its QUALIFIED path: the access rule compares
            // paths, and a nested type's bare name is ambiguous.
            current_info: ld
                .class_path
                .as_deref()
                .and_then(|p| table.class(p)),
            current_path: ld.class_path.as_deref(),
            classes: &all_classes,
            import_index: &import_index,
            dep_imports: &dep_imports,
            host_import_count: table.imports.len(),
            static_index: &static_index,
            defines,
            method_vars: &ld.method_vars,
                type_subst: None,
            local_fns: &local_table,
            current_method: Some(ld.method_name.clone()),
            current_ret: Some(ld.ret_type.clone()),
            lambda_deferred: None,
                    lambda_base: 0,
        };
        let lowered: Result<(), LowerError> = (|| {
            lb.scope_mut().push();
            for p in &ld.params {
                lb.scope_mut()
                    .declare(&p.name, &p.type_name, Access::Default)
                    .map_err(LowerError::UnsupportedConstruct)?;
            }
            // A static local function inherits the enclosing method's
            // type parameters; it receives their handles after its own
            // parameters and reads them through the same map.
            for g in &ld.method_vars {
                let slot = lb
                    .scope_mut()
                    .declare(
                        &format!("__type${g}"),
                        "System.Type",
                        Access::Default,
                    )
                    .map_err(LowerError::UnsupportedConstruct)?;
                let v = lb.bindings[slot];
                lb.type_handles.insert(g.clone(), v);
            }
            // The enclosing method's constraints: a member on one of
            // the inherited type parameters binds through them.
            lb.constraints = ld.constraints.clone();
            // Constants visible at the declaration are part of the
            // local function's compile-time world; materialize them in
            // its own frame (they are not captures).
            for (cname, cty, cinit) in &ld.consts {
                let v = lower::eval(cinit, &mut lb, &lctx)?;
                let slot = lb
                    .scope_mut()
                    .declare(cname, cty, Access::Default)
                    .map_err(LowerError::UnsupportedConstruct)?;
                while lb.bindings.len() <= slot {
                    let fresh = lb.fresh();
                    lb.bindings.push(fresh);
                }
                let home = lb.bindings[slot];
                lb.mov(home, v);
            }
            // An `extern` local function (an `[UnsafeAccessor]`
            // declaration) has no body: the attribute names the target
            // method — the attribute carries the method name, and the
            // first parameter's `[UnsafeAccessorType]` carries the
            // target type. The call is emitted here, so the target's
            // function index is known.
            if ld.is_extern {
                let target = accessor_target(&ld.attrs, &ld.params);
                let resolved = target.as_ref().and_then(|(ty, m)| {
                    resolve_accessor_target(&lctx, ty, m, ld.params.len())
                });
                match resolved {
                    Some((fidx, is_static)) => {
                        let mut argv: Vec<crate::native::ir::VReg> = Vec::new();
                        for i in 0..ld.params.len() {
                            // A static target takes no receiver: the
                            // local's first parameter is the dummy .NET
                            // declares for it.
                            if is_static && i == 0 {
                                continue;
                            }
                            argv.push(lb.bindings[i]);
                        }
                        let dst = lb.fresh();
                        if is_static {
                            lb.push_raw(crate::native::ir::Instr::CallStatic {
                                dst,
                                fidx,
                                args: argv,
                            });
                        } else {
                            lb.push_raw(crate::native::ir::Instr::CallFn {
                                dst,
                                fidx,
                                args: argv,
                            });
                        }
                        lb.terminate(crate::native::ir::Terminator::Ret(Some(dst)));
                        return Ok(());
                    }
                    None => {
                        // The target is not in this unit: .NET throws
                        // for a missing accessor target, and the
                        // source's `catch` handles it. The exception
                        // type must exist for that to mean anything:
                        // a null exception object would crash the
                        // handler instead of reporting.
                        if lctx.table.class("System.MissingMethodException").is_none() {
                            return Err(LowerError::UnsupportedConstruct(format!(
                                "`[UnsafeAccessor]` target '{}' is not in this unit, and System.MissingMethodException is not available to report it",
                                target
                                    .map(|(t, m)| format!("{t}.{m}"))
                                    .unwrap_or_else(|| "an unnamed target".to_owned())
                            )));
                        }
                        let ex = crate::models::Expr::New {
                            class_name: "System.MissingMethodException".to_owned(),
                            args: Vec::new(),
                            init: Vec::new(),
                            span: Default::default(),
                        };
                        let obj = lower::eval(&ex, &mut lb, &lctx)?;
                        lb.terminate(crate::native::ir::Terminator::Unwind { src: obj });
                        return Ok(());
                    }
                }
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
            // A method that cannot be lowered gets no body at all. It
            // used to be emitted as a stub returning 0, so calling it
            // silently produced 0; now the missing implementation is a
            // hard build error.
            Err(e) => {
                return Err(e);
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
                method.params.len() + method.generic_params.len(),
            );
            f.is_async = method.is_async;
            let mut lb = lower::Body::new(&mut f, &mut strings, &type_registry);
            // Lambdas met while lowering THIS body. Their functions are
            // emitted immediately after it, so the index `eval` predicts
            // (`base + 1 + position`) is exactly where each one lands.
            let lambda_deferred: std::cell::RefCell<Vec<LambdaDeferred>> =
                std::cell::RefCell::new(Vec::new());
            // This method's own final index: nothing pushes functions
            // while a body is being lowered, so it is known here.
            let lambda_base = (functions.len() + 1) as u32;
            let ctx = Ctx {
                table: &table,
                class_paths: &class_paths,
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
                type_subst: None,
                lambda_deferred: Some(&lambda_deferred),
                lambda_base,
                // Static local functions of this method (emitted
                // above); bare calls resolve against them first.
                local_fns: &local_table,
                current_method: Some(method.name.clone()),
                current_ret: Some(method.ret_type.clone()),
            };

            lb.scope_mut().push();

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
            // The method's constraints: a member or indexer on a
            // type-parameter receiver binds through them.
            lb.constraints = method.constraints.clone();
            // Hidden type handles follow the declared parameters. They
            // are not source names; the scope slot still maps to the
            // argument register the caller filled.
            for g in &method.generic_params {
                let slot = lb
                    .scope_mut()
                    .declare(
                        &format!("__type${g}"),
                        "System.Type",
                        Access::Default,
                    )
                    .map_err(LowerError::UnsupportedConstruct)?;
                let v = lb.bindings[slot];
                lb.type_handles.insert(g.clone(), v);
            }
            {
                let declared_fields: Vec<String> = table
                    .class(&class.name)
                    .map(|i| i.fields.iter().map(|f| f.name.clone()).collect())
                    .unwrap_or_default();
                for pname in &class.primary_ctor_params {
                    if !declared_fields.iter().any(|f| f == pname) {
                        lb.bind_zero_placeholder(pname);
                    }
                }
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
                    let base_decl = class.bases.first().map(|s| s.as_str()).unwrap_or("");
                    // A generic base is declared with its arguments
                    // (`Task<int>`), but its constructors are the
                    // class's own (`Task`), and their signatures spell
                    // the base's type parameters (`TResult`). So the
                    // lookup uses the class name, and the arguments
                    // stand for the parameters when the candidates
                    // are ranked.
                    let (base_name, base_args) = split_generic_spelling(base_decl);
                    if !base_name.is_empty() {
                        // A base constructor that does not resolve is reported, not
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
                            // The base's type parameters stand for the
                            // derived class's arguments: `Task<int>`'s
                            // constructor takes a `TResult`, which is
                            // `int` here.
                            let subst: Vec<(String, String)> = base_vars
                                .iter()
                                .cloned()
                                .zip(base_args.iter().cloned())
                                .collect();
                            let cands: Vec<(Vec<String>, usize)> = cands
                                .iter()
                                .map(|(sig, idx)| {
                                    (
                                        sig.iter()
                                            .map(|t| substitute_type_vars(t, &subst))
                                            .collect(),
                                        *idx,
                                    )
                                })
                                .collect();
                            if let Some(idx) = pick_chain_ctor(&cands, &arg_tys, &table, &base_vars)
                            {
                                let mut argv = vec![VReg(0)]; // this
                                for a in &args {
                                    // Bare method names (method groups)
                                    // eval to zero via the
                                    // undeclared-variable fallback in
                                    // `eval` (delegate placeholder).
                                    argv.push(lower::eval(a, &mut lb, &ctx)?);
                                }
                                lb.call_fn(idx, argv);
                            } else {
                                // `: base(args)` with no applicable
                                // constructor: skipping it leaves the
                                // base half-constructed, which reads as
                                // a successful construction. Report it.
                                return Err(LowerError::UnsupportedConstruct(
                                    format!(
                                        "no constructor of '{base_name}' accepts {} argument(s) for '{}.{}(: base(...))'",
                                        args.len(),
                                        class.name,
                                        method.name
                                    ),
                                ));
                            }
                        } else {
                            // Arity-matched constructor candidates
                            // exist under no name at all — the base
                            // type is missing from the unit entirely.
                            return Err(LowerError::UnsupportedConstruct(format!(
                                "no constructor candidates for base '{base_name}' in '{}.{}(: base(...))'",
                                class.name, method.name
                            )));
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
            functions.push(finished);

            // Emit one function per lambda this body created. The indices
            // `eval` predicted (`base + 1 + position`) are exactly where
            // these land, because the bodies are lowered in the order they
            // were met and each may append more (a lambda inside a lambda),
            // which the index-driven loop handles.
            {
                let mut i = 0usize;
                loop {
                    let next: Option<LambdaDeferred> = {
                        let q = lambda_deferred.borrow();
                        q.get(i).cloned()
                    };
                    let Some(lam) = next else { break };
                    i += 1;
                    // Arity is the delegate object FIRST, then the
                    // lambda's declared parameters: the call site passes
                    // the object so the body can reach its captures.
                    let mut inf = IrFunction::new(
                        format!("{}::<lambda{}>", method.name, i),
                        ReceiverKind::None,
                        lam.params.len() + 1,
                    );
                    let mut fb = lower::Body::new(&mut inf, &mut strings, &type_registry);
                    let lctx = Ctx {
                        current_method: Some(format!(
                            "{}::<lambda{}>",
                            method.name, i
                        )),
                        current_ret: None,
                        lambda_deferred: Some(&lambda_deferred),
                        lambda_base,
                        ..ctx
                    };
                    fb.scope_mut().push();
                    let mut lowered: Result<(), LowerError> = Ok(());
                    // Slot 0 is the delegate object (VReg(1) in this ABI:
                    // VReg(0) is the reserved receiver slot).
                    let delegate_vreg = VReg(1);
                    if let Err(e) =
                        fb.scope_mut().declare("__delegate", "object", crate::models::Access::Default)
                    {
                        return Err(LowerError::UnsupportedConstruct(e));
                    }
                    for p in &lam.params {
                        // Parameter types come from the delegate at the
                        // call site, which lowering does not see; `object`
                        // keeps them addressable and 8 bytes wide, which is
                        // what the ABI passes.
                        if let Err(e) = fb.scope_mut().declare(
                            p,
                            "object",
                            crate::models::Access::Default,
                        ) {
                            lowered = Err(LowerError::UnsupportedConstruct(e));
                            break;
                        }
                    }
                    for (k, cname) in lam.captures.iter().enumerate() {
                        // Captured names are BY-REFERENCE locals whose
                        // binding is the address stored in the delegate
                        // object, so the body's reads dereference and its
                        // writes store through — C# capture semantics.
                        let slot = match fb.scope_mut().declare(
                            cname,
                            "object",
                            crate::models::Access::Default,
                        ) {
                            Ok(s) => s,
                            Err(e) => return Err(LowerError::UnsupportedConstruct(e)),
                        };
                        let addr = fb.fresh();
                        fb.push_raw(Instr::LoadField {
                            dst: addr,
                            base: delegate_vreg,
                            off: 24 + 8 * k as u32,
                        });
                        while fb.bindings.len() <= slot {
                            let v = fb.fresh();
                            fb.bindings.push(v);
                        }
                        fb.bindings[slot] = addr;
                        fb.scopes.make_byref(cname);
                    }
                    if lowered.is_ok() {
                        lowered = match &lam.body {
                            crate::models::ast::expression::LambdaBody::Expr(e) => {
                                // The body's value IS the return value:
                                // terminate with it rather than moving it
                                // into a vreg (`VReg(0)` is the receiver
                                // slot, not a return register, so writing
                                // there discarded the result).
                                lower::eval(e, &mut fb, &lctx).map(|v| {
                                    fb.terminate(Terminator::Ret(Some(v)));
                                })
                            }
                            crate::models::ast::expression::LambdaBody::Block(b) => {
                                let mut r = Ok(());
                                for st in &b.stmts {
                                    if fb.is_sealed() {
                                        break;
                                    }
                                    if let Err(e) = fb.statement(st, &lctx) {
                                        r = Err(e);
                                        break;
                                    }
                                }
                                r
                            }
                        };
                    }
                    fb.scope_mut().pop();
                    if let Err(e) = lowered {
                        return Err(e);
                    }
                    fb.seal_with_return_zero();
                    functions.push(inf);
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
                    // C# requires an explicit return type on the entry
                    // point: `static Main()` (no type) is CS0017-adjacent
                    // and `static void Main()` is the only void form.
                    // A declaration with no return type at all is a
                    // different thing from `void`, and accepting it
                    // silently made a malformed entry point look valid.
                    if method.ret_type.is_empty() {
                        return Err(LowerError::UnsupportedConstruct(crate::tf!(
                            "rubyc.native.ir.entry.main_needs_return_type"
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
        // Interface members live ABOVE every class slot: their index is
        // the same in every row, which is what a call through an
        // interface-typed receiver needs. Class slots keep their own
        // numbering below the range.
        let iface_entries = crate::native::semantic::iface_slot_entries(&table);
        let iface_end = iface_entries
            .last()
            .map(|(_, s)| *s + 1)
            .unwrap_or(0);
        let max_slot = entry_slots
            .iter()
            .map(|(s, _, _)| *s)
            .max()
            .unwrap_or(0)
            .max(iface_end.saturating_sub(1));
        let mut row = vec![usize::MAX; max_slot + 1];
        for (slot, mname, psig) in &entry_slots {
            // Preserve abstract holes: only fill when an implementation
            // actually resolves (a slot with no body anywhere stays MAX
            // instead of collapsing to an unrelated index 0 function).
            if let Some(idx) = impl_index(&all_classes, &table, &class.name, mname, psig, &fn_index) {
                row[*slot] = idx;
            }
        }
        if let Some(ci) = table.class(&class.name) {
            for (iname, _arity) in &ci.interfaces {
                for ((ikey, mname, msig), slot) in &iface_entries {
                    if ikey != iname {
                        continue;
                    }
                    let sig: Vec<String> = if msig.is_empty() {
                        Vec::new()
                    } else {
                        msig.split(',').map(|t| t.to_owned()).collect()
                    };
                    // An explicit implementation (`int I.M()`, stored
                    // as `I.M`) wins over a plain same-named method,
                    // matching C# binding for calls through the
                    // interface; inherited bodies resolve through
                    // `impl_index`'s base walk.
                    let explicit = format!("{iname}.{mname}");
                    let idx = impl_index(
                        &all_classes,
                        &table,
                        &class.name,
                        &explicit,
                        &sig,
                        &fn_index,
                    )
                    .or_else(|| {
                        impl_index(&all_classes, &table, &class.name, mname, &sig, &fn_index)
                    });
                    if let Some(i) = idx {
                        row[*slot] = i;
                    }
                }
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
                class_paths: &class_paths,
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
                type_subst: None,
                local_fns: &local_table,
                current_method: None,
                current_ret: None,
                lambda_deferred: None,
                    lambda_base: 0,
            };
            let mut inf = IrFunction::new("@init", ReceiverKind::None, 0);
            let mut ib = lower::Body::new(&mut inf, &mut strings, &type_registry);
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
                        current_path: class_paths
                            .get(&(*class as *const ClassDecl))
                            .map(|s| s.as_str()),
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
                    let v = lower::eval(minit, &mut ib, &field_ctx)?;
                    ib.store_static(idx, v);
                }
            }
            let init_idx = functions.len();
            functions.push(ib.finish_with(Terminator::Ret(None)));
            Some(init_idx)
        };
        append_type_slots(&type_registry, &mut statics);
        let program = Program {
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
        };
        verify::verify(&program).map_err(LowerError::IrInvalid)?;
        return Ok(program);
    };
    let Some(entry_decl) = classes.iter().find(|c| c.name == entry_class) else {
        return Err(LowerError::NoMainMethod);
    };
    let Some(info) = table.class(&entry_class) else {
        return Err(LowerError::NoMainMethod);
    };

    let object_bytes = (info.fields.len() + INSTANCE_HEADER_SLOTS) as u64 * 8;
    let start_ctx = Ctx {
        table: &table,
        class_paths: &class_paths,
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
                type_subst: None,
        local_fns: &local_table,
        current_method: None,
        current_ret: None,
        lambda_deferred: None,
                    lambda_base: 0,
    };

    let mut sf = IrFunction::new("@start", ReceiverKind::None, 0);
    let mut sl = lower::Body::new(&mut sf, &mut strings, &type_registry);
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
            // A static initializer that cannot be lowered is a build error.
            // Properties used to fail soft to zeroed storage here (their
            // `out`-var pre-statements are only drained for fields), but
            // a warning plus a silent 0 is indistinguishable from a
            // successful initialization at runtime.
            let (mname, minit, mpre): (
                &String,
                &crate::models::Expr,
                Option<&Vec<crate::models::Stmt>>,
            ) = match m {
                ClassMember::Field(fd) if fd.is_static => {
                    (&fd.name, &fd.init, Some(&fd.init_pre))
                }
                ClassMember::Property(pd)
                    if pd.is_static
                        && pd.get.is_none()
                        && pd.set.is_none()
                        && pd.get_expr.is_none() =>
                {
                    (&pd.name, &pd.init, None)
                }
                _ => continue,
            };
            let Some(&idx) = static_index.get(&(class.name.clone(), mname.clone())) else {
                continue;
            };
              let field_ctx = Ctx {
                current_class: &class.name,
                // The PATH must match the class this context switches
                // to: inheriting the caller's path made the access
                // rule compare unrelated types (a private const of
                // the field's own class was denied).
                current_path: start_ctx
                    .table
                    .class(&class.name)
                    .map(|i| i.key.path.as_str()),
                current_method: start_ctx.current_method.clone(),
                current_ret: None,
                ..start_ctx
            };
            // A throwing static initializer propagates, exactly as C#
            // requires. @start runs every type initializer eagerly,
            // so a throw here now aborts the program instead of being
            // swallowed into zeroed storage: a field left at its
            // default after a failed initializer reads like a
            // successful initialization that happened to produce 0.
            // Bindings declared while parsing the initializer
            // (`out` vars; fields only — property initializers carry
            // none) lower first so the init sees them.
            if let Some(pres) = mpre {
                for pre in pres {
                    sl.statement(pre, &field_ctx)?;
                }
            }
            // A static initializer that cannot be lowered is a build error, for
            // properties as much as fields: a warning plus zeroed
            // storage is indistinguishable at runtime from a
            // successfully-initialized default.
            let v = lower::eval(minit, &mut sl, &field_ctx)?;
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

    // The entry instance must be CONSTRUCTED, not just allocated. This
    // used to allocate the object, run the entry class's own field
    // initializers, and hand it straight to `Main` — which meant an
    // instance `Main` observed a receiver whose constructor had never
    // run (`class C { int W; public C() { W = 3; } int Main() { return W; } }`
    // answered 0) and, when the entry class inherited, whose BASE's
    // initializers had not run either.
    //
    // A constructor function is self-contained: it runs the class's field
    // initializers and then its body, and chains to the base. So when the
    // entry class has a parameterless one, calling it is the whole job.
    // Only when there is no constructor does this fall back to running
    // the initializers inline — and then it must walk the base chain
    // (base first), because nothing else will.
    let entry_ctor = fn_index
        .get(&format!("{entry_class}::{entry_class}#0"))
        .and_then(|v| v.last())
        .copied();
    if let Some(ctor_idx) = entry_ctor {
        let call_dst = sl.fresh();
        sl.push_raw(Instr::CallFn {
            dst: call_dst,
            fidx: ctor_idx as u32,
            args: vec![obj],
        });
    } else {
        // EVERY base, not just the first: a type with several bases has
        // a constructor (or initializers) in each, and following only
        // `bases.first()` left the others uninitialized
        // (`class C : A, B` ran A's constructor and silently skipped
        // B's). Depth-first, dependencies first, so each class is
        // visited after everything it derives from; the `seen` set
        // terminates on the diamond case and on cycles.
        let mut chain: Vec<&ClassDecl> = Vec::new();
        let mut seen: Vec<String> = Vec::new();
        fn collect<'a>(
            decl: &'a ClassDecl,
            classes: &[&'a ClassDecl],
            seen: &mut Vec<String>,
            out: &mut Vec<&'a ClassDecl>,
        ) {
            if seen.iter().any(|n| n == &decl.name) {
                return;
            }
            seen.push(decl.name.clone());
            for b in &decl.bases {
                let simple = b.split('<').next().unwrap_or(b).trim();
                if let Some(base) = classes.iter().find(|d| d.name == simple) {
                    collect(base, classes, seen, out);
                }
            }
            out.push(decl);
        }
        collect(entry_decl, &classes, &mut seen, &mut chain);
        // Base first, so a derived initializer sees a filled base. Each
        // class in the chain contributes exactly one of the two: its
        // constructor if it declares a parameterless one (a constructor
        // runs its own field initializers and chains to ITS base), or
        // its initializers inline. Without this, a base whose
        // initialisation lives in a constructor body — the common case
        // — was skipped entirely.
        // Where each class's subobject begins inside the entry object.
        // `chain` is ancestors-first, and the layout puts each class's
        // fields after its ancestors', so a base's subobject starts at
        // 16 + 8 * (field slots before it). A constructor is emitted once
        // and writes at ITS class's offsets, so the receiver handed to a
        // base constructor must be adjusted by this amount — otherwise
        // every base writes at the same offsets and they collide (which
        // is exactly what `class C : A, B` did).
        let mut base_off: std::collections::BTreeMap<String, u32> = Default::default();
        {
            let mut slots: u32 = 0; // field slots, not object slots
            for decl_in_chain in chain.iter() {
                // RELATIVE to the first field slot: a class compiled on
                // its own puts its first field at object slot 2, so a
                // base whose subobject also starts at slot 2 needs no
                // adjustment, and only bases sitting behind another
                // base's fields are shifted.
                base_off.insert(decl_in_chain.name.clone(), slots * 8);
                let n = decl_in_chain
                    .members
                    .iter()
                    .filter(|m| {
                        matches!(
                            m,
                            ClassMember::Field(fd) if !fd.is_static && !fd.is_const
                        )
                    })
                    .count() as u32;
                slots += n;
            }
        }
        for decl_in_chain in chain.iter() {
            let ctor = fn_index
                .get(&format!(
                    "{name}::{name}#0",
                    name = decl_in_chain.name
                ))
                .and_then(|v| v.last())
                .copied();
            if let Some(cidx) = ctor {
                if decl_in_chain.name != entry_class {
                    let off = base_off
                        .get(&decl_in_chain.name)
                        .copied()
                        .unwrap_or(0);
                    let recv = if off == 0 {
                        obj
                    } else {
                        let o = sl.const_i64(off as i64);
                        sl.emit(Instr::Bin {
                            dst: VReg(0),
                            op: crate::native::ir::BinOp::Add,
                            a: obj,
                            b: o,
                        })
                    };
                    let call_dst = sl.fresh();
                    sl.push_raw(Instr::CallFn {
                        dst: call_dst,
                        fidx: cidx as u32,
                        args: vec![recv],
                    });
                }
                continue;
            }
            for fd in decl_in_chain.members.iter().filter_map(|m| match m {
                // NOTE: `const` fields are compile-time only (no
                // storage): emitting them here would store e.g.
                // `CharSize = 2` at offset 0, clobbering the entry
                // object's vtable word (see the same fix in the `new`
                // path in `lower/eval.rs`).
                ClassMember::Field(fd) if !fd.is_static && !fd.is_const => Some(fd),
                _ => None,
            }) {
                let v = lower::eval(&fd.init, &mut sl, &start_ctx)?;
                let off = (info
                    .field(&fd.name)
                    .map(|fi| fi.offset_slots)
                    .unwrap_or(1)
                    * 8) as u32;
                sl.push_raw(Instr::StoreField { base: obj, off, src: v });
            }
        }
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


    functions.insert(0, sl.finish_with(Terminator::Exit));


    append_type_slots(&type_registry, &mut statics);
    let program = Program {
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
    };
    verify::verify(&program).map_err(LowerError::IrInvalid)?;
    Ok(program)
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
                // Generic methods take hidden type handles. The image
                // export table and the consumer's import path do not
                // yet carry them, so exporting one would publish an
                // ABI that the other side cannot call correctly.
                && !table.class(&class.name).is_some_and(|c| {
                    c.methods.iter().any(|m| {
                        m.name == name
                            && m.param_types.len() == arity
                            && !m.generic_params.is_empty()
                    })
                })
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
                || !m.generic_params.is_empty()
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
