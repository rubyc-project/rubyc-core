//! Driver: the orchestration layer between the CLI and the compiler
//! internals, also usable as an embedding API.
//!
//! Host applications call [`compile_source`], [`run_source`] or the
//! lower-level functions directly; `main.rs` is a three-line shim over
//! [`cli_main`].

use crate::bytecode::traits::{FromBytecode, ToBytecode};
use crate::cli::{self, Command, InFormat, Invocation, IoArgs, OutFormat, ParsedArgs};
use crate::diagnostics::{render_json, render_text};
use crate::models::compare::StructuralEq;
use crate::models::diagnostic::Severity;
use crate::models::{CompilationUnit, Diagnostic};
use crate::native::semantic::TypeKey;
use crate::native::target::{InputKind, NativePipeline};
use crate::parser;
use crate::utilities::base64;
use std::io::{IsTerminal, Read, Write};
use std::path::PathBuf;

/// Failure modes surfaced to embedders and the CLI alike.
#[derive(Debug, Clone)]
pub enum DriverError {
    /// Plain human-readable failure.
    Message(String),
    /// Structured parse/semantic diagnostics plus the source they refer to.
    Diagnostics {
        diags: Vec<Diagnostic>,
        source: String,
    },
    /// The program ran and exited with this status.
    ExitStatus(i32),
}

impl std::fmt::Display for DriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DriverError::Message(m) => write!(f, "{m}"),
            DriverError::Diagnostics { diags, .. } => {
                let joined = diags
                    .iter()
                    .map(std::string::ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; ");
                write!(f, "{joined}")
            }
            DriverError::ExitStatus(code) => write!(f, "exit status {code}"),
        }
    }
}

type Result<T> = std::result::Result<T, DriverError>;

// --------------------------------------------------------------- public API

/// Entry point mirroring the command-line tool: parse argv, dispatch,
/// render, and report the process exit code.
pub fn cli_main(argv: Vec<String>, pipeline: &NativePipeline) -> i32 {
    // Discovery flags short-circuit normal dispatch.
    if let Some(lines) = extension_listing(&argv) {
        for line in lines {
            println!("{line}");
        }
        return 0;
    }
    match cli::parse_args(&argv) {
        Ok(ParsedArgs::Help(text)) => {
            print!("{text}");
            0
        }
        Ok(ParsedArgs::Version(text)) => {
            println!("{text}");
            0
        }
        Ok(ParsedArgs::Run(invocation)) => match dispatch(&invocation, pipeline) {
            Ok(()) => 0,
            Err(DriverError::ExitStatus(code)) => code,
            Err(e) => {
                eprintln!("{}", crate::tf!("rubyc.cli.error_prefix", e.to_string()));
                1
            }
        },
        Err(e) => {
            eprintln!("{}", crate::tf!("rubyc.cli.error_prefix", e));
            eprintln!("{}", crate::t!("rubyc.cli.try_help"));
            2
        }
    }
}

/// Render discovery output without performing terminal I/O. This is the
/// embedding/test surface used by `cli_main` for `--list-*` flags.
pub fn extension_listing(argv: &[String]) -> Option<Vec<String>> {
    let registry = discovery_registry(argv);
    if argv.iter().any(|a| a == "--list-targets") {
        return Some(
            registry
                .targets()
                .iter()
                .map(|target| {
                    format!(
                        "{} {} | {} | {} | {}",
                        target.metadata.name,
                        target.metadata.version,
                        target.metadata.author,
                        target.triple,
                        target.metadata.description
                    )
                })
                .collect(),
        );
    }
    None
}

fn discovery_registry(argv: &[String]) -> crate::extensions::ExtensionRegistry {
    crate::extensions::ExtensionRegistry::discover_with(
        flag_value(argv, "--base").as_deref().map(|s| s.as_ref()),
        flag_value(argv, "--targets-dir").as_deref().map(|s| s.as_ref()),
    )
}

/// Read a `--flag <value>` or `--flag=<value>` pair from raw argv.
pub(crate) fn flag_value(argv: &[String], flag: &str) -> Option<String> {
    let prefix = format!("{flag}=");
    for (i, arg) in argv.iter().enumerate() {
        if let Some(value) = arg.strip_prefix(&prefix) {
            if !value.is_empty() {
                return Some(value.to_owned());
            }
        } else if arg == flag {
            if let Some(next) = argv.get(i + 1) {
                if !next.starts_with('-') {
                    return Some(next.clone());
                }
            }
        }
    }
    None
}

/// Parse RubyC source into a compilation unit, returning structured
/// diagnostics on failure.
pub fn parse_source(source: &str) -> std::result::Result<CompilationUnit, Vec<Diagnostic>> {
    parser::parse(source)
}

/// Convenience: compile source text to a native executable image.
pub fn compile_source(
    source: &str,
    libs: &[PathBuf],
    pipeline: &NativePipeline,
) -> std::result::Result<Vec<u8>, String> {
    let unit = parse_source(source).map_err(|errs| {
        errs.iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; ")
    })?;
    crate::native::compile_to_bytes_with_libs(&unit, pipeline.backend, pipeline.image_writer, libs, &[])
        .map_err(|e| e.to_string())
}

/// Convenience: JIT-execute source text, returning its exit status.
pub fn run_source(source: &str, pipeline: &NativePipeline) -> std::result::Result<i32, String> {
    let unit = parse_source(source).map_err(|errs| {
        errs.iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; ")
    })?;
    let dyn_dir = crate::extensions::install_root().join("libraries");
    crate::native::jit::jit_run_with_dyn_libs(&unit, pipeline.backend, pipeline.memory, &[], &dyn_dir)
        .map_err(|e| e.to_string())
}

// ------------------------------------------------------------------ dispatch

fn dispatch(invocation: &Invocation, pipeline: &NativePipeline) -> Result<()> {
    if matches!(
        invocation.command,
        Command::Install
            | Command::Remove
            | Command::Update
            | Command::ListRemote
            | Command::Download
            | Command::New
            | Command::Create
            | Command::Add
    ) {
        return match invocation.command {
            Command::Install => cmd_install(invocation, pipeline),
            Command::Remove => cmd_remove(invocation),
            Command::Update => cmd_update(invocation),
            Command::Download => cmd_download(invocation),
            Command::New => cmd_new(invocation),
            Command::Create => cmd_new(invocation),
            Command::Add => cmd_add(invocation),
            _ => cmd_list_remote(),
        };
    }

    if matches!(invocation.command, Command::List) {
        return cmd_list(invocation);
    }

    // Load the selected target plugin. It is owned here so the borrowed
    // pipeline below can safely reference it for the command's duration.
    // Languages are in-process (not plugins), so only the target is loaded.
    let target = resolve_selection(invocation)?;
    let pipeline = match &target {
        Some(t) => pipeline.with_target(t),
        _ => *pipeline,
    };
    // Plugin execution routing is future work: the pipeline above always
    // compiles with the statically-linked backend, so a selected plugin
    // naming any other target must fail loudly here — never silently
    // emit for the wrong platform.
    if let Some(t) = &target {
        if t.name != pipeline.target.target_os_name() {
            return Err(DriverError::Message(crate::tf!(
                "rubyc.cli.err.target_mismatch",
                t.name.clone(),
                pipeline.target.target_os_name().to_owned()
            )));
        }
    }

    let outcome = match invocation.command {
        Command::Compile => cmd_compile(invocation, &pipeline),
        Command::Build => cmd_build(invocation, &pipeline),
        Command::Run => cmd_run(invocation, &pipeline),
        Command::Lint => cmd_lint(invocation, &pipeline),
        Command::Dump => cmd_dump(&invocation.io, &invocation, &pipeline),
        Command::CheckRoundtrip => cmd_check_roundtrip(invocation.io.file.as_deref()),
        _ => Ok(()),
    };
    match outcome {
        Ok(()) => Ok(()),
        // Parse/semantic diagnostics honor --error-format and render once,
        // with the configured visual style — never through the generic
        // "error: …" prefix path.
        Err(DriverError::Diagnostics { diags, source }) => {
            eprint!("{}", render_diags(&diags, Some(&source), invocation));
            Err(DriverError::ExitStatus(1))
        }
        Err(e) => Err(e),
    }
}

/// Load the selected target plugin from the install root.
///
/// When `--target` names an installed plugin, it is loaded and returned. When
/// nothing is selected, the default host target plugin is loaded if present.
/// When no plugin is installed at all, a clear "no target installed or
/// selected" message is returned.
///
/// Languages are not plugins: the in-process parser/IR lowering in
/// `rubyc-core` is always used, so there is no language selection to resolve.
fn resolve_selection(
    invocation: &Invocation,
) -> Result<Option<crate::plugin::TargetPlugin>> {
    let io = &invocation.io;
    let base = resolve_base(io);

    let targets = crate::plugin::discover_targets(&base);

    // Pick the target: explicit --target, else the default host target.
    let target: Option<crate::plugin::TargetPlugin> = if let Some(name) = &io.target {
        let available: Vec<String> = targets.iter().map(|t| t.name.clone()).collect();
        match targets.into_iter().find(|t| &t.name == name) {
            Some(t) => Some(t),
            None => {
                return Err(DriverError::Message(crate::tf!(
                    "rubyc.cli.err.unknown_target",
                    name.clone(),
                    available.join(", ")
                )));
            }
        }
    } else {
        targets.into_iter().find(|t| t.name == "linux_x86_64")
    };

    // If the user explicitly requested a target that wasn't found, report
    // clearly. Otherwise fall back to the statically-linked backend.
    if target.is_none() {
        if io.target.is_some() {
            let msg = crate::t!("rubyc.cli.err.no_target_installed_or_selected");
            return Err(DriverError::Message(msg.to_string()));
        }
        // No explicit selection: use the statically-linked backend.
        return Ok(None);
    }

    Ok(target)
}

/// Resolve the install root from CLI flags + environment.
fn resolve_base(io: &IoArgs) -> std::path::PathBuf {
    if let Some(base) = &io.base {
        return std::path::PathBuf::from(base);
    }
    crate::extensions::install_root()
}

fn lint_config(inv: &Invocation) -> crate::lints::LintConfig {
    let mut config = crate::lints::LintConfig::default();
    for (key, level) in &inv.lint_overrides {
        config.set(key, *level);
    }
    config
}

/// Render lint diagnostics to stderr and report whether any were denied.
fn report_lints(diags: &[Diagnostic], inv: &Invocation) -> bool {
    if diags.is_empty() {
        return false;
    }
    let fatal = diags.iter().any(|d| d.severity == Severity::Error);
    if inv.quiet && !fatal {
        return fatal;
    }
    let rendered = render_diags(diags, None, inv);
    eprint!("{rendered}");
    fatal
}

/// Lint a parsed unit; produce a fatal error when a `--deny` rule fired.
fn check_lints(unit: &CompilationUnit, inv: &Invocation) -> Result<()> {
    let diags = crate::lints::analyze(unit, &lint_config(inv));
    if report_lints(&diags, inv) {
        Err(DriverError::ExitStatus(1))
    } else {
        Ok(())
    }
}

/// What a command received after sniffing/decoding.
#[derive(Clone)]
enum Input {
    Source(String),
    Unit(CompilationUnit),
}

// ----------------------------------------------------------------- commands

/// `compile` is a backward-compatible alias for [`cmd_build`].
fn cmd_compile(invocation: &Invocation, pipeline: &NativePipeline) -> Result<()> {
    cmd_build(invocation, pipeline)
}

/// Produce AST bytes from the given input using the in-process parser
/// (`into_unit` → `to_bytecode`).
///
/// `--no-magic` emits raw payload bytes with no RubyC preamble.
fn ast_bytes_from_input(
    input: Input,
    invocation: &Invocation,
) -> Result<Vec<u8>> {
    let unit = into_unit(input, invocation)?;
    if invocation.io.no_magic {
        Ok(unit.to_bytecode_no_magic())
    } else {
        Ok(unit.to_bytecode())
    }
}

/// `.asm` files: assemble → minimal MachineCode → standard ELF writer.
/// Entry point is `_start` (or byte 0). 🐕🇨 preamble kept via `entry_preamble`.
fn build_asm_file(
    path: &str,
    out: &crate::cli::IoArgs,
    _invocation: &Invocation,
    pipeline: &NativePipeline,
) -> Result<()> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| DriverError::from_msg(format!("cannot read {path}: {e}")))?;

    use crate::native::target::MachineCode;
    let (code, labels) =
        crate::parser::assembly::assemble_with_symbols(&source).map_err(DriverError::from_msg)?;

    let entry_off = labels
        .iter()
        .find(|(l, _)| l == "_start")
        .map(|(_, off)| *off)
        .unwrap_or(0);

    let mc = MachineCode {
        code,
        data: vec![],
        fn_bases: vec![entry_off],
        vtable_offsets: vec![],
        vtable_names: vec![],
        static_offsets: vec![],
        static_names: vec![],
        helper_itoa: None,
        helper_itoa_err: None,
        helper_scan_int: None,
        helper_retain: None,
        helper_release: None,
        helper_alloc: None,
        helper_exc_enter: None,
        helper_exc_pop: None,
        helper_exc_throw: None,
        helper_exc_rethrow: None,
        helper_exc_load: None,
        helper_exc_clear: None,
        exc_state_offset: None,
        helper_argv: None,
        import_names: vec![],
        import_bases: vec![],
        import_abs: vec![],
        exports: vec![],
        export_takes_receiver: vec![],
        init_fn: None,
        relocs: vec![],
        heap_size: 4096,
        strip_symbols: false,
    };

    let elf = pipeline
        .image_writer
        .write_image(&mc, !out.no_magic)
        .map_err(|e| DriverError::from_msg(e.to_string()))?;
    write_out(out, &elf)?;
    // Mark output executable
    if let Some(path) = &out.output {
        let _ = std::fs::set_permissions(path, std::os::unix::fs::PermissionsExt::from_mode(0o755));
    }
    Ok(())
}

/// A same-key interface collision whose bases differ: methods union
/// immediately, but the winning bases are decided post-load, when
/// every interface in the unit is present for exact transitive
/// requirement counts.
pub(crate) struct DeferredIface {
    ns: String,
    name: String,
    arity: usize,
    own_existing: usize,
    own_incoming: usize,
    bases_existing: Vec<String>,
    bases_incoming: Vec<String>,
    existing_is_main: bool,
}

/// Total requirements of one interface view: own methods plus every
/// transitively inherited base-interface method. Simple-name matching
/// throughout (same philosophy as the semantic scan's `effective`).
fn iface_total_reqs(
    own: usize,
    bases: &[String],
    ifaces: &[(String, usize, Vec<String>)],
    seen: &mut Vec<String>,
) -> usize {
    fn simple(n: &str) -> &str {
        n.split('<').next().unwrap_or(n).rsplit('.').next().unwrap_or(n)
    }
    let mut total = own;
    for b in bases {
        let s = simple(b.trim()).to_owned();
        if seen.iter().any(|x| x == &s) {
            continue;
        }
        seen.push(s.clone());
        if let Some((_, count, sub)) = ifaces.iter().find(|(n, _, _)| n == &s) {
            let sub = sub.clone();
            total += iface_total_reqs(*count, &sub, ifaces, seen);
        }
    }
    total
}

/// Resolve deferred interface-base collisions after every source is
/// loaded (transitive data is complete only then). A main-project entry
/// keeps its bases absolutely (C# CS0436: source wins over reference);
/// a dep/dep entry adopts the weaker view (fewer total requirements):
/// polyfill stubs ("define necessary members only") must not gain the
/// real view's bases, which their structs could never satisfy, while
/// full implementers satisfy the narrower view trivially. Ties keep the
/// existing entry (deterministic: dependency files load sorted).
pub(crate) fn resolve_deferred_ifaces(
    unit: &mut crate::models::CompilationUnit,
    deferred: &[DeferredIface],
) {
    use crate::models::Member;
    // Flattened (namespaces and top level alike): base lookup matches
    // by simple name, mirroring the semantic scan.
    let mut ifaces: Vec<(String, usize, Vec<String>)> = Vec::new();
    for m in unit.all_members() {
        if let Member::Interface(i) = m {
            ifaces.push((i.name.clone(), i.methods.len(), i.bases.clone()));
        }
    }
    for d in deferred {
        if d.existing_is_main {
            continue;
        }
        let entries: &mut Vec<Member> = if d.ns.is_empty() {
            &mut unit.members
        } else if let Some(ns) = unit.namespaces.iter_mut().find(|n| n.name == d.ns) {
            &mut ns.members
        } else {
            continue;
        };
        let Some(entry) = entries.iter_mut().find(|e| match e {
            Member::Interface(i) => i.name == d.name && i.generic_params.len() == d.arity,
            _ => false,
        }) else {
            continue;
        };
        let ce = iface_total_reqs(d.own_existing, &d.bases_existing, &ifaces, &mut Vec::new());
        let ci = iface_total_reqs(d.own_incoming, &d.bases_incoming, &ifaces, &mut Vec::new());
        if ci < ce {
            if let Member::Interface(i) = entry {
                i.bases = d.bases_incoming.clone();
            }
        }
    }
}
/// Identity key of a top-level type declaration: (namespace, kind,
/// name, arity). Matches declarations across main/dependency files for
/// cross-boundary union (below).
pub(crate) fn top_level_type_key(ns: &str, m: &crate::models::Member) -> Option<(String, char, String, usize)> {
    use crate::models::Member;
    match m {
        Member::Class(c) => Some((
            ns.to_owned(),
            'C',
            c.name.clone(),
            c.generic_params.len(),
        )),
        Member::Interface(i) => Some((
            ns.to_owned(),
            'I',
            i.name.clone(),
            i.generic_params.len(),
        )),
        Member::Enum(e) => Some((ns.to_owned(), 'E', e.name.clone(), 0)),
        Member::Delegate(d) => Some((
            ns.to_owned(),
            'D',
            d.name.clone(),
            d.generic_params.len(),
        )),
    }
}

/// Union a dependency top-level declaration into already-combined
/// members: when an entry with the same key exists, merge into it
/// (per-member main-wins) and return `None`; otherwise return the
/// member back for normal appending.
///
/// Same-key interfaces with differing bases only union their methods
/// here; the winning bases are deferred (`deferred`, decided post-load
/// by `resolve_deferred_ifaces`, when transitive data is complete).
/// `main_keys` snapshots the main project's keys so a main entry keeps
/// its bases absolutely (C# CS0436).
pub(crate) fn union_into(
    combined_members: &mut Vec<crate::models::Member>,
    ns: &str,
    mut incoming: crate::models::Member,
    main_keys: &std::collections::HashSet<(String, char, String, usize)>,
    deferred: &mut Vec<DeferredIface>,
) -> Option<crate::models::Member> {
    // NOTE: never use `?` here — a find-miss must return the member
    // for appending, not drop it (a `?` conflation once silently
    // discarded thousands of dependency members).
    let key = match top_level_type_key(ns, &incoming) {
        Some(k) => k,
        None => return Some(incoming),
    };
    let Some(pos) = combined_members.iter().position(|e| {
        top_level_type_key(ns, e).is_some_and(|k| {
            k.0 == key.0 && k.1 == key.1 && k.2 == key.2 && k.3 == key.3
        })
    }) else {
        return Some(incoming);
    };
    if let (crate::models::Member::Interface(e), crate::models::Member::Interface(i)) =
        (&combined_members[pos], &incoming)
    {
        if e.bases != i.bases {
            deferred.push(DeferredIface {
                ns: ns.to_owned(),
                name: key.2.clone(),
                arity: key.3,
                own_existing: e.methods.len(),
                own_incoming: i.methods.len(),
                bases_existing: e.bases.clone(),
                bases_incoming: i.bases.clone(),
                existing_is_main: main_keys.contains(&key),
            });
            if let crate::models::Member::Interface(i) = &mut incoming {
                i.bases.clear();
            }
        }
    }
    let existing = &mut combined_members[pos];
    merge_dep_member(existing, incoming);
    None
}

/// Preserve a lost implicit parameterless ctor after merging two
/// same-path class declarations from different assemblies (at least
/// one side is a cross-assembly union — see `union_cross_assembly`).
/// When exactly one side declares instance ctors, the other side
/// relied on the implicit parameterless one, which exists only as
/// absence and would otherwise be lost, breaking that side's `new
/// X()` with a loud `ctor_arity` (MLC vs CoreLib
/// `System.DefaultBinder`). Synthesizes an explicit empty-body ctor
/// (indistinguishable from a source-written `X() { }`). No-op when
/// neither side declares ctors (natural implicit path), when both
/// do, or when a 0-arg ctor already survived the merge. Callers pass
/// the pre-merge ctor states; the entry is mutated in place.
fn preserve_union_implicit_ctor(
    existing_had_ctors: bool,
    incoming_had_ctors: bool,
    e: &mut crate::models::ClassDecl,
) {
    if existing_had_ctors == incoming_had_ctors {
        return;
    }
    let has_zero_arg = e.members.iter().any(|m| match m {
        crate::models::ClassMember::Method(x) => {
            x.name == e.name && x.params.is_empty() && !x.is_static
        }
        _ => false,
    });
    if has_zero_arg {
        return;
    }
    e.members.push(crate::models::ClassMember::Method(
        crate::models::MethodDecl {
            access: crate::models::ast::statement::Access::Internal,
            attrs: Vec::new(),
            generic_params: Vec::new(),
            ret_type: String::new(),
            name: e.name.clone(),
            params: Vec::new(),
            body: Some(crate::models::Block {
                stmts: Vec::new(),
                span: crate::models::Span::new(0, 0),
            }),
            span: crate::models::Span::new(0, 0),
            is_override: false,
            is_static: false,
            is_abstract: false,
            is_partial: false,
            is_extern: false,
            is_async: false,
            base_call: None,
        },
    ));
}

/// Merge a dependency top-level declaration into the main project's
/// same-key entry (cross-assembly union): per-member main-wins, so
/// shared sources (compiled into several assemblies) and
/// per-assembly helpers (whose members differ per library) complete
/// each other instead of conflicting: each assembly compiles its own
/// full copy. Within one project, genuine duplicates still error
/// downstream (E0012/E0013); only the main/dep boundary unions.
pub(crate) fn merge_dep_member(existing: &mut crate::models::Member, incoming: crate::models::Member) {
    use crate::models::Member;
    match (existing, incoming) {
        (Member::Class(e), Member::Class(i)) => {
            // Cross-assembly provenance: this entry absorbed another
            // assembly's same-path class. Same-assembly partial parts
            // never pass through here (main files append directly), so
            // the flag precisely marks collision unions for semantic's
            // partial-merge rule below.
            e.union_cross_assembly = true;
            for b in i.bases {
                if !e.bases.contains(&b) {
                    e.bases.push(b);
                }
            }
            // Same strictness as `merge_project_types` below: partial
            // parts agree on struct/class upstream.
            e.is_struct = e.is_struct && i.is_struct;
            let has_instance_ctor = |ms: &[crate::models::ClassMember], n: &str| {
                ms.iter().any(|m| match m {
                    crate::models::ClassMember::Method(x) => {
                        x.name == n && !x.is_static
                    }
                    _ => false,
                })
            };
            let e_had_ctors = has_instance_ctor(&e.members, &e.name);
            let i_had_ctors = has_instance_ctor(&i.members, &e.name);
            for m in i.members {
                if !class_member_is_dup(&e.members, &m) {
                    e.members.push(m);
                }
            }
            // Cross-assembly same-path siblings: preserve a lost
            // implicit ctor (see `preserve_union_implicit_ctor`).
            // Ctors arriving later via semantic partial-merge or final
            // coalescing are covered by the sibling rules there (keyed
            // off `union_cross_assembly`, set above).
            preserve_union_implicit_ctor(e_had_ctors, i_had_ctors, e);
        }
        (Member::Interface(e), Member::Interface(i)) => {
            for b in i.bases {
                if !e.bases.contains(&b) {
                    e.bases.push(b);
                }
            }
            for m in i.methods {
                if !e.methods.iter().any(|x| {
                    x.name == m.name
                        && x.params.len() == m.params.len()
                        && x
                            .params
                            .iter()
                            .zip(m.params.iter())
                            .all(|(a, b)| a.type_name == b.type_name)
                }) {
                    e.methods.push(m);
                }
            }
        }
        (Member::Enum(e), Member::Enum(i)) => {
            for (vn, vv) in i.variants {
                if !e.variants.iter().any(|(en, _)| en == &vn) {
                    e.variants.push((vn, vv));
                }
            }
        }
        // Delegates (and kind mismatches): keep the main project's
        // declaration; the dependency copy is dropped.
        _ => {}
    }
}

/// Exact-duplicate class member (same kind+name+arity+types+generic
/// arity+byref): the main project's copy wins, the dependency's is
/// dropped. Generic arity joins the key — `M<T>(IServiceCollection)` vs
/// `M<T,U>(IServiceCollection)` share erased params but are distinct
/// overloads (dropping one breaks explicit-args calls); mirrors the
/// semantic table's dedup key. Byref-ness joins it too — value-vs-`ref`
/// twins (e.g. `WriteAtOffset` 3/3) share erased params; dropping the
/// second stranded every `ref`-marked call (self-election → infinite
/// recursion on identical args).
pub(crate) fn class_member_is_dup(existing: &[crate::models::ClassMember], incoming: &crate::models::ClassMember) -> bool {
    use crate::models::ClassMember;
    match incoming {
        ClassMember::Method(m) => existing.iter().any(|e| match e {
            ClassMember::Method(x) => {
                x.name == m.name
                    && x.params.len() == m.params.len()
                    && x
                        .params
                        .iter()
                        .zip(m.params.iter())
                        .all(|(a, b)| a.type_name == b.type_name && a.is_byref == b.is_byref)
                    && x.generic_params.len() == m.generic_params.len()
            }
            _ => false,
        }),
        ClassMember::Field(f) => existing.iter().any(|e| match e {
            ClassMember::Field(x) => x.name == f.name,
            _ => false,
        }),
        ClassMember::Property(p) => existing.iter().any(|e| match e {
            ClassMember::Property(x) => x.name == p.name,
            _ => false,
        }),
        ClassMember::Indexer(i) => existing.iter().any(|e| match e {
            ClassMember::Indexer(x) => {
                x.params.len() == i.params.len()
                    && x
                        .params
                        .iter()
                        .zip(i.params.iter())
                        .all(|(a, b)| a.type_name == b.type_name)
            }
            _ => false,
        }),
        ClassMember::Event(e) => existing.iter().any(|x| match x {
            ClassMember::Event(xx) => xx.name == e.name,
            _ => false,
        }),
        // Nested types: keep both (downstream partial-merge/drop
        // handles them with main-first order preserved).
        ClassMember::Nested(_) => false,
    }
}

/// Load every source in a project and merge it before semantic analysis.
/// Also merges the `ruby_c` dependency closure: each entry names a sibling
/// `.rcp` whose pre-build hook is run (materializing its transient `.rc`
/// sources) and whose sources join the unit, so cross-assembly type
/// references resolve. Returns the merged unit plus the dependency projects
/// (pre-hooks already run) so the caller can run their post-build hooks.
/// Visible for closure integration tests (driver/closure_defines).
pub(crate) fn load_project_unit_with_deps(
    path: &std::path::Path,
    invocation: &Invocation,
    consume: bool,
) -> Result<(
    CompilationUnit,
    Vec<crate::project::Project>,
    std::collections::HashMap<TypeKey, String>,
    Vec<crate::native::rci::Fragment>,
)> {    use crate::project::Project;
    let project = Project::load(path).map_err(DriverError::from_msg)?;
    let base = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    // BFS over the `ruby_c` closure; canonicalized paths break cycles and
    // diamonds (A -> B -> A, or two parents sharing one dep). Paths are
    // canonicalized up front (not just for the seen-set) so errors name
    // clean locations instead of `A/../B/../C` chains.
    let mut deps: Vec<Project> = Vec::new();
    let mut seen: std::collections::BTreeSet<std::path::PathBuf> = Default::default();
    seen.insert(canonical_or(path));
    let mut queue: Vec<std::path::PathBuf> = project
        .ruby_c_libs
        .iter()
        .map(|e| {
            resolve_dep(base, e)
                .map(|p| canonical_or(&p))
                .map_err(|m| DriverError::from_msg(m))
        })
        .collect::<Result<Vec<_>>>()?;
    while let Some(dep_path) = queue.pop() {
        let canon = canonical_or(&dep_path);
        if !seen.insert(canon) {
            continue;
        }
        // A missing dep `.rcp` means the pull/regen step must re-run
        // first (post-build hooks keep `.rcp` files since the
        // stable-reference change, so absence is a genuinely stale
        // tree) — say so instead of leaking a bare OS error.
        let dep = Project::load(&dep_path).map_err(|e| {
            DriverError::from_msg(format!(
                "cannot read dependency {}: {e} (re-run the pull/regen step to restore .rcp files)",
                dep_path.display()
            ))
        })?;
        if invocation.io.skip_pre_hook {
            eprintln!("[hook:pre-build] skipped (dev)");
        } else {
            dep.run_hook("pre-build", invocation.io.target.as_deref()).map_err(DriverError::from_msg)?;
        }
        let dep_base = dep_path.parent().unwrap_or_else(|| std::path::Path::new("."));
        for e in &dep.ruby_c_libs {
            let p = resolve_dep(dep_base, e).map_err(|m| DriverError::from_msg(m))?;
            queue.push(canonical_or(&p));
        }
        deps.push(dep);
    }
    // Every failure path below must leave the vendored tree `.cs`-only:
    // run the dependency post-hooks (which delete the materialized `.rc`
    // files) whenever loading fails — at source collection, per-file
    // parse, or the final scan alike. (On success the caller owns cleanup.)
    // Note: dep pre-hooks have already run above, so `.rc` files may
    // exist from here on; all `?` returns must go through this closure.
    let result = (|| -> Result<(
        CompilationUnit,
        std::collections::HashMap<TypeKey, String>,
        Vec<crate::native::rci::Fragment>,
    )> {
    // Dynamic-linking consume: per-dep installed fragments replace
    // dep sources (opt-in via `[build] dynamic = true`). Each dep is
    // decided independently (mixed mode: fresh fragments load, the
    // rest compile from sources). A consumed dep's FILES are
    // skipped below, but its `ruby_c` deps still recurse (their
    // entries are needed — fragments carry only their own lib).
    // Freshness failures fall back to sources silently (a note names
    // the dep); the build stays correct, just slower.
    let mut consumed: std::collections::BTreeSet<usize> = Default::default();
    let mut fragments: Vec<crate::native::rci::Fragment> = Vec::new();
    if consume && project.dynamic {
        let dyn_dir = crate::extensions::install_root().join("libraries");
        for (di, dep) in deps.iter().enumerate() {
            if let Some(frag) = dep_fragment(dep, &dyn_dir) {
                if !invocation.quiet {
                    eprintln!("[rubyc] consuming {}.rci instead of sources", dep.name);
                }
                consumed.insert(di);
                fragments.push(frag);
            } else if !invocation.quiet {
                eprintln!(
                    "[rubyc] compiling {} from sources (no fresh .rci)",
                    dep.name
                );
            }
        }
        // `using`-driven discovery: a `using Ns;` with no owner among
        // the main project or its explicit deps auto-links the single
        // installed library owning that namespace (fragment consume —
        // no sources needed). Zero or one unknown owner stays silent
        // (existing unknown-name errors apply); two or more is a loud
        // ambiguity error. Only direct main-project usings participate:
        // discovered libraries declare their own `ruby_c` closure, and
        // installed fragments are trusted as-written (the post-build
        // hook installs `.rci` next to its image; without local sources
        // there is no hash to check against).
        if let Ok(main_files) = project.source_files() {
            let known: std::collections::BTreeSet<&str> = std::iter::once(project.name.as_str())
                .chain(deps.iter().map(|d| d.name.as_str()))
                .collect();
            let mut added: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
            match crate::native::index::ensure(&dyn_dir) {
                Ok(index) => {
                    for ns in usings_in_files(&main_files) {
                        let owners = index.lookup(&ns);
                        if owners.iter().any(|o| known.contains(o.as_str())) {
                            continue;
                        }
                        let unknown: Vec<&str> = owners
                            .iter()
                            .filter(|o| !known.contains(o.as_str()))
                            .map(|o| o.as_str())
                            .collect();
                        match unknown.as_slice() {
                            [] => {}
                            [lib] => {
                                if added.insert(lib.to_string())
                                    && let Some(frag) = installed_fragment(lib, &dyn_dir)
                                {
                                    if !invocation.quiet {
                                        eprintln!(
                                            "[rubyc] auto-linking {lib}.rci for `using {ns};`"
                                        );
                                    }
                                    fragments.push(frag);
                                }
                            }
                            many => {
                                return Err(DriverError::from_msg(format!(
                                    "`using {ns};` is ambiguous: owned by {} — add an explicit [libs] ruby_c entry for the intended library",
                                    many.join(", ")
                                )));
                            }
                        }
                    }
                }
                Err(e) => {
                    if !invocation.quiet {
                        eprintln!("[rubyc] warning: skipping using-discovery ({e})");
                    }
                }
            }
        }
    }
    // Main-project files first, dependencies after: a main-project type
    // declaration shadows same-key dependency declarations (C# CS0436:
    // source wins over reference instead of erroring E0012 duplicates).
    let mut main_files = project.source_files().map_err(DriverError::from_msg)?;
    // Dependency files each preprocess under their OWNING project's
    // symbols (every assembly compiles separately: the same shared
    // file can produce different views per owner, and each view must
    // survive).
    // Same-content copies with differing defines are DIFFERENT inputs:
    // the (content, defines) dedup below loads each distinct view once
    // and same-key entries member-union (identical members dedup by
    // name+signature, complementary `#if` branches add). Same content
    // + same defines still dedups to one load. BTreeMap keeps
    // path-sorted load order, so union tie-breaks are unchanged.
    let mut dep_owners: std::collections::BTreeMap<std::path::PathBuf, Vec<usize>> =
        Default::default();
    for (di, dep) in deps.iter().enumerate() {
        match dep.source_files() {
            Ok(files) => {
                for f in files {
                    let owners = dep_owners.entry(f).or_default();
                    if !owners.contains(&di) {
                        owners.push(di);
                    }
                }
            }
            Err(e) => {
                // A dependency with no materialized sources contributes
                // no types. That is expected for platform-only libraries
                // (e.g. a Windows-only lib whose every source the
                // pre-build platform filter dropped): skip it with a
                // warning rather than failing the whole build. If the
                // main project truly needs its types, the semantic scan
                // reports the unknown names honestly below.
                if !invocation.quiet {
                    eprintln!(
                        "[rubyc] skipping dependency with no sources on this platform: {} ({e})",
                        dep.path.display()
                    );
                }
            }
        }
    }
    main_files.sort();
    main_files.dedup();
    // Path-sorted load order; each file carries its
    // owning projects for per-owner defines below. Files owned ONLY
    // by consumed deps are skipped (their types arrive via fragments);
    // files shared with an uncompiled owner still load (that owner
    // needs them compiled).
    let dep_files: Vec<(std::path::PathBuf, Vec<usize>)> = dep_owners
        .into_iter()
        .filter(|(_, owners)| owners.iter().any(|oi| !consumed.contains(oi)))
        .collect();
    // Preprocessor symbols are PER-PROJECT: CLI `-D` plus that file's
    // own project `[build] defines`. A global merge leaks
    // assembly-scoped symbols across the unit (one library's private
    // symbol can preprocess out another library's public members).
    // Shared files get the union of their content-group owners'
    // defines at load below (content-hash dedup keeps one copy).
    let mut main_defines: Vec<String> = invocation.defines.clone();
    for extra in &project.defines {
        if !main_defines.contains(extra) {
            main_defines.push(extra.clone());
        }
    }
    let mut combined = CompilationUnit::default();
    // Content-hash dedup: shared sources (fetched into several
    // libraries of one closure) are byte-identical under different
    // paths; including the same view twice would define every type
    // twice. Copies whose whole content group shares one view dedup
    // to a single load; copies with differing defines load separately
    // and member-union below, so no owner loses its view.
    let mut seen_content: std::collections::HashSet<(u64, Vec<String>)> =
        std::collections::HashSet::new();
    // Deps whose prebuild was already re-run for a vanished file: one
    // re-stage per dep per build. Verdicts are deterministic (same
    // csproj + env), so a second vanish in the same dep is prune churn
    // from the other build, not a deleted file needing restoration —
    // re-running prebuild every time feeds a cross-build restage loop.
    let mut prebuild_retried: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    // Dependency top-level declarations union into the combined unit's
    // same-key entries (per-member main-wins via `merge_dep_member`);
    // genuinely new types append normally below.
    let load_file = |source_path: &std::path::PathBuf,
                         is_main: bool,
                         combined: &mut CompilationUnit,
                         seen_content: &mut std::collections::HashSet<(u64, Vec<String>)>,
                         main_keys: &std::collections::HashSet<(
                            String,
                            char,
                            String,
                            usize,
                        )>,
                         deferred: &mut Vec<DeferredIface>,
                         file_defines: &[String],
                         owner_name: &str,
                         owner_dep: Option<&crate::project::Project>,
                         target: Option<&str>,
                         type_owners: &mut std::collections::HashMap<TypeKey, String>,
                         prebuild_retried: &mut std::collections::HashSet<String>|
     -> Result<()> {
        let source = match std::fs::read_to_string(source_path) {
            Ok(source) => source,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Concurrent-build race: another process re-staged then
                // pruned (or vice versa) between our glob and this read.
                // At most ONE prebuild re-run per dep per build: verdicts
                // are deterministic (same csproj + env), so the first
                // re-run restores anything the other build's cleanup
                // deleted, and any later vanish in the same dep is the
                // other build's prune churn — re-running prebuild for
                // those feeds a cross-build restage loop. A file still
                // absent was deliberately excluded — solo runs never
                // list it either — so skip it with a note rather than
                // failing the whole build.
                // other build's prune churn — re-running prebuild for
                // those feeds a cross-build restage loop. A file still
                // absent after the re-run gets a short backoff window:
                // the other build may be mid-restage and about to bring
                // it back (its own prebuild re-copies unconditionally).
                // Past the window it was deliberately excluded — solo
                // runs never list it either — so skip it with a note
                // rather than failing the whole build.
                if let Some(dep) = owner_dep {
                    if prebuild_retried.insert(dep.name.clone()) {
                        let _ = dep.run_hook("pre-build", target);
                    }
                }
                let mut source = std::fs::read_to_string(source_path).ok();
                for _ in 0..10 {
                    if source.is_some() {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    source = std::fs::read_to_string(source_path).ok();
                }
                match source {
                    Some(source) => source,
                    None => {
                        eprintln!(
                            "[rubyc] note: {} vanished mid-load (concurrent prune); skipping",
                            source_path.display()
                        );
                        return Ok(());
                    }
                }
            }
            Err(e) => {
                return Err(DriverError::from_msg(format!(
                    "cannot read {}: {e}",
                    source_path.display()
                )));
            }
        };
        {
            // Dedup key = (content, defines): identical bytes under
            // different symbols are DIFFERENT inputs (the main project
            // loads first, so a content-only key would keep a narrow
            // copy and drop the full one). Same content + same defines
            // skips; differing defines loads and member-unions below.
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            source.hash(&mut hasher);
            let content = hasher.finish();
            let mut syms = file_defines.to_vec();
            syms.sort();
            if !seen_content.insert((content, syms)) {
                return Ok(());
            }
        }
        let filename = source_path.display().to_string();
        let pre = crate::preprocessor::preprocess(&source, file_defines, &filename)
            .map_err(|diagnostics| DriverError::Diagnostics {
                diags: diagnostics
                    .into_iter()
                    .map(|diagnostic| diagnostic.in_file(filename.clone()))
                    .collect(),
                source: source.clone(),
            })?;
        let expanded = crate::preprocessor::expand_macros(pre.source, &pre.macros);
        let mut unit =
            parser::parse(&expanded).map_err(|diagnostics| DriverError::Diagnostics {
                diags: diagnostics
                    .into_iter()
                    .map(|diagnostic| diagnostic.in_file(filename.clone()))
                    .collect(),
                source: expanded.clone(),
            })?;

        // Ownership for dynamic-linking (see `analyze_with_owners`):
        // every top-level member key attributes to this file's library.
        // First recording wins: main files load first, so merged
        // main+dep entries attribute to main (mirroring content
        // main-wins); shared multi-owner files attribute to their
        // first loader. Nested members inherit through the parent
        // chain downstream.
        {
            use crate::models::Member;
            let mut record = |ns: &str, m: &Member| {
                let (name, arity) = match m {
                    Member::Class(c) => (c.name.clone(), c.generic_params.len()),
                    Member::Interface(i) => (i.name.clone(), i.generic_params.len()),
                    Member::Enum(e) => (e.name.clone(), 0),
                    Member::Delegate(d) => (d.name.clone(), d.generic_params.len()),
                };
                let path = if ns.is_empty() {
                    name
                } else {
                    format!("{ns}.{name}")
                };
                type_owners
                    .entry(TypeKey::new(path, arity))
                    .or_insert_with(|| owner_name.to_owned());
            };
            for m in &unit.members {
                record("", m);
            }
            for ns in &unit.namespaces {
                for m in &ns.members {
                    record(&ns.name, m);
                }
            }
        }

        if !is_main {
            // Cross-boundary union (see `merge_dep_member`).
            let mut kept_top: Vec<crate::models::Member> = Vec::new();
            for m in unit.members.drain(..) {
                if let Some(m) = union_into(&mut combined.members, "", m, main_keys, deferred) {
                    kept_top.push(m);
                }
            }
            unit.members = kept_top;
            for ns in &mut unit.namespaces {
                if let Some(existing) = combined
                    .namespaces
                    .iter_mut()
                    .find(|n| n.name == ns.name)
                {
                    let ns_name = ns.name.clone();
                    let mut kept_ns: Vec<crate::models::Member> = Vec::new();
                    for m in ns.members.drain(..) {
                        if let Some(m) =
                            union_into(&mut existing.members, &ns_name, m, main_keys, deferred)
                        {
                            kept_ns.push(m);
                        }
                    }
                    ns.members = kept_ns;
                }
                // Namespaces the main project lacks append wholesale below.
            }
        }

        combined.members.append(&mut unit.members);
        combined.aliases.append(&mut unit.aliases);
        combined.static_usings.append(&mut unit.static_usings);
        combined.assembly_blocks.extend(pre.assembly_blocks);
        // Merge namespaces by name: same-named namespaces across files have
        // their members combined (partial-namespace semantics); distinct
        // namespaces coexist in the unit.
        for mut namespace in unit.namespaces.drain(..) {
            if let Some(existing) = combined
                .namespaces
                .iter_mut()
                .find(|n| n.name == namespace.name)
            {
                existing.members.append(&mut namespace.members);
                existing.span = existing.span.to(namespace.span);
            } else {
                combined.namespaces.push(namespace);
            }
        }
        Ok(())
    };

    let main_keys_empty: std::collections::HashSet<(String, char, String, usize)> =
        Default::default();
    let mut deferred: Vec<DeferredIface> = Vec::new();
    // Per-entry library ownership for dynamic-linking (see
    // `analyze_with_owners`); returned alongside the unit below.
    let mut type_owners: std::collections::HashMap<TypeKey, String> = Default::default();
    for source_path in &main_files {
        load_file(
            source_path,
            true,
            &mut combined,
            &mut seen_content,
            &main_keys_empty,
            &mut deferred,
            &main_defines,
            &project.name,
            Some(&project),
            invocation.io.target.as_deref(),
            &mut type_owners,
            &mut prebuild_retried,
        )?;
    }
    // Snapshot the main project's keys: a same-key main entry keeps its
    // interface bases absolutely (C# CS0436); only dep/dep collisions
    // resolve weaker-wins below.
    let mut main_keys: std::collections::HashSet<(String, char, String, usize)> =
        Default::default();
    for m in &combined.members {
        if let Some(k) = top_level_type_key("", m) {
            main_keys.insert(k);
        }
    }
    for ns in &combined.namespaces {
        for m in &ns.members {
            if let Some(k) = top_level_type_key(&ns.name, m) {
                main_keys.insert(k);
            }
        }
    }
    for (source_path, owners) in &dep_files {
        // The file's own owners' defines (usually exactly one owner;
        // several only when dep closures overlap paths).
        let mut file_defines: Vec<String> = invocation.defines.clone();
        for oi in owners {
            for extra in &deps[*oi].defines {
                if !file_defines.contains(extra) {
                    file_defines.push(extra.clone());
                }
            }
        }
        // Ownership attributes to the first owning dependency
        // (deterministic BFS order); main files already claimed
        // shared keys above. Prefer an UNCONSUMED owner: a shared file
        // compiled here belongs (for future import routing) to the lib
        // actually emitting its code, not to a consumed dep whose
        // fragment merely shadows it.
        let owner_name = owners
            .iter()
            .find(|oi| !consumed.contains(oi))
            .or_else(|| owners.first())
            .and_then(|oi| deps.get(*oi))
            .map(|d| d.name.as_str())
            .unwrap_or("");
        let owner_dep = owners
            .iter()
            .find(|oi| !consumed.contains(oi))
            .or_else(|| owners.first())
            .and_then(|oi| deps.get(*oi));
        load_file(
            source_path,
            false,
            &mut combined,
            &mut seen_content,
            &main_keys,
            &mut deferred,
            &file_defines,
            owner_name,
            owner_dep,
            invocation.io.target.as_deref(),
            &mut type_owners,
            &mut prebuild_retried,
        )?;
    }
    resolve_deferred_ifaces(&mut combined, &deferred);

    merge_project_types(&mut combined);

    crate::inliner::run(&mut combined);
    let diagnostics = crate::diagnostics_semantic::scan_with_fragments(&combined, &fragments);
    if diagnostics.is_empty() {
        // Cross-fragment duplicate keys (shared files claimed by two
        // consumed deps): first fragment in dep order wins, mirroring
        // first-match resolution. (Within one fragment, keys are
        // unique by construction.)
        {
            let mut seen: std::collections::BTreeSet<
                crate::native::semantic::TypeKey,
            > = Default::default();
            for frag in &mut fragments {
                frag.classes.retain(|c| seen.insert(c.key.clone()));
            }
        }
        Ok((combined, type_owners, fragments))
    } else {
        Err(DriverError::Diagnostics {
            diags: diagnostics,
            source: String::new(),
        })
    }
    })();
    if result.is_err() {
        for dep in &deps {
            if !invocation.io.skip_post_hook {
                let _ = dep.run_hook("post-build", invocation.io.target.as_deref());
            }
        }
    }
    result.map(|(unit, type_owners, fragments)| (unit, deps, type_owners, fragments))
}

/// Installed fragment for a dependency, when fresh: the image (`.so`)
/// must exist under the install dir, the fragment's lib name must
/// match, and its source hash must equal a fresh hash of the dep's
/// current own sources. ANY mismatch (or unreadable file) falls back
/// to compiling from sources — stale fragments are never trusted.
/// Returns `None` for "compile from sources". Fragment bytes come
/// from the image's post-magic table first, then its legacy
/// `.rubyc_types` section, then the `.rci` sidecar file.
fn dep_fragment(
    dep: &crate::project::Project,
    dyn_dir: &std::path::Path,
) -> Option<crate::native::rci::Fragment> {
    let so_path = dyn_dir.join(format!("{}.so", dep.name));
    let rci_path = dyn_dir.join(format!("{}.rci", dep.name));
    if !so_path.is_file() {
        return None;
    }
    if let Ok(image) = std::fs::read(&so_path) {
        let embedded = crate::native::embedded::table_bytes(&image)
            .or_else(|| crate::native::embedded::section_bytes(&image));
        if let Some(bytes) = embedded
            && let Ok(frag) = crate::native::rci::decode(bytes)
            && frag.lib_name == dep.name
            && frag.source_hash == rci_source_hash(dep)
        {
            return Some(frag);
        }
    }
    let bytes = std::fs::read(&rci_path).ok()?;
    let frag = crate::native::rci::decode(&bytes).ok()?;
    if frag.lib_name != dep.name {
        return None;
    }
    if frag.source_hash != rci_source_hash(dep) {
        return None;
    }
    Some(frag)
}

/// Fragment for a `using`-discovered library: same shape as
/// [`dep_fragment`], but there is no local project (no sources to hash
/// against), so installed fragments are trusted as-written — the
/// post-build hook installs them next to their image, and a missing
/// image (`.so` or `.a`) vetoes the fragment. Post-magic table wins
/// over the legacy section over the sidecar file; name mismatch vetoes
/// either way.
fn installed_fragment(lib: &str, lib_dir: &std::path::Path) -> Option<crate::native::rci::Fragment> {
    let check = |frag: crate::native::rci::Fragment| (frag.lib_name == lib).then_some(frag);
    let mut image_seen = false;
    for ext in ["so", "a"] {
        let image_path = lib_dir.join(format!("{lib}.{ext}"));
        if !image_path.is_file() {
            continue;
        }
        image_seen = true;
        let Ok(image) = std::fs::read(&image_path) else {
            continue;
        };
        let embedded = if ext == "a" {
            crate::native::arlib::read(&image)
                .ok()
                .and_then(|members| {
                    crate::native::embedded::archive_member(&members).map(|b| b.to_vec())
                })
        } else {
            crate::native::embedded::table_bytes(&image)
                .or_else(|| crate::native::embedded::section_bytes(&image))
                .map(|b| b.to_vec())
        };
        if let Some(bytes) = embedded
            && let Ok(frag) = crate::native::rci::decode(&bytes)
            && let Some(frag) = check(frag)
        {
            return Some(frag);
        }
    }
    if !image_seen {
        return None;
    }
    let bytes = std::fs::read(lib_dir.join(format!("{lib}.rci"))).ok()?;
    let frag = crate::native::rci::decode(&bytes).ok()?;
    check(frag)
}

/// Plain `using Ns;` namespaces across source files (lexical scan —
/// resolution drops the directives at parse, so discovery reads them
/// from text). Sorted and deduplicated.
fn usings_in_files(files: &[std::path::PathBuf]) -> Vec<String> {
    let mut out = Vec::new();
    for path in files {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        out.extend(crate::native::index::usings_in_source(&text));
    }
    out.sort();
    out.dedup();
    out
}

/// Content hash of a project's own sources for `.rci` freshness:
/// sorted main-file bytes plus project defines and the compiler
/// identity. Deliberately EXCLUDES the dep set: a dep-list change
/// cannot alter the lib's OWN entries (owner-scoped emit), and
/// cross-lib partials (the only counterexample) are C#-illegal.
/// Any input change that could alter the lib's own table entries
/// (sources, symbols, toolchain) changes the hash; a consume path
/// with a mismatching hash falls back to sources (safe direction —
/// stale fragments are never trusted). Already-emitted `.rci` files
/// with the older dep-names hash simply mismatch and recompile.
pub(crate) fn rci_source_hash(project: &crate::project::Project) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    env!("CARGO_PKG_VERSION").hash(&mut hasher);
    // The compiler binary itself: versions are pinned, so the version
    // string alone can never distinguish builds. A same-version logic
    // change (e.g. a dedup-rule fix) otherwise leaves stale fragments
    // trusted. Best-effort; unreadable binary falls back to the
    // version string above. Cached: one read per process no matter
    // how many deps hash.
    static EXE_HASH: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    let exe_hash = EXE_HASH.get_or_init(|| {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        std::env::current_exe()
            .ok()
            .and_then(|exe| std::fs::read(&exe).ok())
            .hash(&mut h);
        Some(h.finish())
    });
    exe_hash.hash(&mut hasher);
    let mut defines = project.defines.clone();
    defines.sort();
    defines.hash(&mut hasher);
    let mut files = project.source_files().unwrap_or_default();
    files.sort();
    for f in &files {
        std::fs::read(f).unwrap_or_default().hash(&mut hasher);
    }
    hasher.finish()
}

/// Resolve one `ruby_c` entry against the referencing project's directory.
/// - Path form (contains a separator or ends in `.rcp`): joined onto the
///   base exactly as before (`../Foo/Foo.rcp`, absolute paths, `./x.rcp`).
/// - Bare name (`System.Collections`): searched as `<dir>/<Name>/<Name>.rcp`
///   (plus the flat `<dir>/<Name>.rcp`) across `RUBYC_LIBRARIES`
///   (colon-separated library roots) then the referencing project's own
///   directory (the dotnet-tree sibling layout). This lets user projects
///   outside the tree reference installed/pulled libraries by name instead
///   of brittle absolute paths — and keeps working when transient `.rcp`
///   files are regenerated, since the NAME is stable while paths churn.
fn resolve_dep(base: &std::path::Path, entry: &str) -> std::result::Result<std::path::PathBuf, String> {
    if entry.contains('/') || entry.contains('\\') || entry.ends_with(".rcp") {
        return Ok(base.join(entry));
    }
    let mut dirs: Vec<std::path::PathBuf> = Vec::new();
    if let Some(env) = std::env::var_os("RUBYC_LIBRARIES") {
        dirs.extend(std::env::split_paths(&env));
    }
    dirs.push(base.to_path_buf());
    for d in &dirs {
        for cand in [
            d.join(entry).join(format!("{entry}.rcp")),
            d.join(format!("{entry}.rcp")),
        ] {
            if cand.is_file() {
                return Ok(cand);
            }
        }
    }
    Err(format!(
        "cannot resolve library '{entry}': searched {} (set RUBYC_LIBRARIES to add roots)",
        dirs.iter()
            .map(|d| d.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// Canonicalize for cycle detection; fall back to the raw path when the
/// file does not (yet) exist.
fn canonical_or(p: &std::path::Path) -> std::path::PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

/// Load every source in a project and merge it before semantic analysis.
/// Single-project callers with no interest in the dependency closure.
fn load_project_unit(
    path: &std::path::Path,
    invocation: &Invocation,
) -> Result<CompilationUnit> {
    load_project_unit_with_deps(path, invocation, false).map(|(unit, _, _, _)| unit)
}

/// Coalesce declarations of the same class across project files. The parser
/// currently discards the `partial` marker, so duplicate-member validation is
/// left to the semantic/type-table pass while the source fragments become the
/// single class shape expected by lowering.
fn merge_project_types(unit: &mut CompilationUnit) {
    fn merge(members: &mut Vec<crate::models::Member>) {
        use crate::models::Member;
        // Key by (name, arity): same-named types with different generic
        // arity (`Task` vs `Task<TResult>`) are DISTINCT types and must
        // never merge (merging unions bases, fabricating self-loops like
        // `Task : Task` from `Task<TResult> : Task`). Same-key partials
        // still merge below; true duplicates are E0013 downstream.
        let mut index = std::collections::BTreeMap::<(String, usize), usize>::new();
        let mut merged = Vec::with_capacity(members.len());
        for member in std::mem::take(members) {
            match member {
                Member::Class(mut class) => {
                    let key = (class.name.clone(), class.generic_params.len());
                    if let Some(&position) = index.get(&key)
                        && let Member::Class(existing) = &mut merged[position]
                    {
                        for base in class.bases.drain(..) {
                            if !existing.bases.contains(&base) {
                                existing.bases.push(base);
                            }
                        }
                        // Partial parts agree on struct/class upstream;
                        // strict AND keeps any mismatch loud downstream.
                        existing.is_struct = existing.is_struct && class.is_struct;
                        // Cross-assembly coalescing (either side absorbed
                        // another assembly's same-path class via the dep
                        // union): preserve a lost implicit ctor exactly
                        // like the dep union does. Pure same-assembly
                        // partials (both unflagged) merge silently, so
                        // splitting ctors across parts stays an error as
                        // C# requires.
                        let cross = existing.union_cross_assembly
                            || class.union_cross_assembly;
                        let has_instance_ctor =
                            |ms: &[crate::models::ClassMember], n: &str| {
                                ms.iter().any(|m| match m {
                                    crate::models::ClassMember::Method(x) => {
                                        x.name == n && !x.is_static
                                    }
                                    _ => false,
                                })
                            };
                        let e_had =
                            has_instance_ctor(&existing.members, &existing.name);
                        let i_had =
                            has_instance_ctor(&class.members, &existing.name);
                        existing.members.append(&mut class.members);
                        existing.span = existing.span.to(class.span);
                        existing.union_cross_assembly = cross;
                        if cross {
                            preserve_union_implicit_ctor(e_had, i_had, existing);
                        }
                    } else {
                        index.insert(key, merged.len());
                        merged.push(Member::Class(class));
                    }
                }
                other => merged.push(other),
            }
        }
        *members = merged;
    }

    // Nested partials (`private sealed partial class HillClimbing`
    // split across `PortableThreadPool.HillClimbing{,.Complex}.cs`):
    // the top-level merge never sees them, so lowering found two
    // `HillClimbing` decls and the field-init `find` hit the
    // fieldless part first (null `_randomIntervalGenerator`,
    // segfault in the ctor). Recurse the same merge into every
    // class's nested classes; cross-assembly provenance and the
    // implicit-ctor rule ride along unchanged.
    fn merge_nested(class: &mut crate::models::ClassDecl) {
        use crate::models::{ClassMember, Member};
        let taken = std::mem::take(&mut class.members);
        let mut rest: Vec<ClassMember> = Vec::with_capacity(taken.len());
        let mut nested: Vec<Member> = Vec::new();
        let mut anchor: Option<usize> = None;
        for m in taken {
            if let ClassMember::Nested(Member::Class(nc)) = m {
                if anchor.is_none() {
                    anchor = Some(rest.len() + nested.len());
                }
                nested.push(Member::Class(nc));
            } else {
                rest.push(m);
            }
        }
        if nested.is_empty() {
            class.members = rest;
            return;
        }
        merge(&mut nested);
        let mut out: Vec<ClassMember> = Vec::with_capacity(rest.len() + nested.len());
        let anchor = anchor.unwrap_or(rest.len());
        let mut nested_iter = nested.into_iter();
        let mut inserted = false;
        for (i, m) in rest.into_iter().enumerate() {
            if !inserted && i == anchor {
                for n in nested_iter.by_ref() {
                    if let Member::Class(mut nc) = n {
                        merge_nested(&mut nc);
                        out.push(ClassMember::Nested(Member::Class(nc)));
                    } else {
                        out.push(ClassMember::Nested(n));
                    }
                }
                inserted = true;
            }
            out.push(m);
        }
        if !inserted {
            for n in nested_iter {
                if let Member::Class(mut nc) = n {
                    merge_nested(&mut nc);
                    out.push(ClassMember::Nested(Member::Class(nc)));
                } else {
                    out.push(ClassMember::Nested(n));
                }
            }
        }
        class.members = out;
    }

    merge(&mut unit.members);
    for namespace in &mut unit.namespaces {
        merge(&mut namespace.members);
    }
    // Recurse into top-level and namespace classes (their nested
    // partials merge above); deeper levels recurse themselves.
    for member in unit.members.iter_mut() {
        if let crate::models::Member::Class(c) = member {
            merge_nested(c);
        }
    }
    for namespace in unit.namespaces.iter_mut() {
        for member in namespace.members.iter_mut() {
            if let crate::models::Member::Class(c) = member {
                merge_nested(c);
            }
        }
    }
}



/// Dependency fragments for consume paths, concatenated in dep order
/// (analysis dedups first-wins). Shared by library builds and the `.rci`
/// emit below so the sidecar mirrors the built table.
fn build_preloaded(
    fragments: &[crate::native::rci::Fragment],
) -> crate::native::semantic::Preloaded {
    let mut classes = Vec::new();
    let mut interfaces = Vec::new();
    let mut delegates = Vec::new();
    for frag in fragments {
        classes.extend(frag.classes.iter().cloned());
        interfaces.extend(frag.interfaces.iter().cloned());
        delegates.extend(frag.delegates.iter().cloned());
    }
    crate::native::semantic::Preloaded {
        classes,
        interfaces,
        delegates,
    }
}

/// `.rci` sidecar (this lib's own post-resolution table entries for
/// downstream builds), shared by the `.so` and `.a` outputs: the same
/// sidecar serves dynamic and static consumers. Re-analysis is pure and
/// cheap next to lowering; it uses the SAME preloaded fragments as the
/// build, so the sidecar mirrors the built table (inherited dep members
/// included — a bare re-analysis would omit them). A failed re-analysis
/// never fails the build (lowering already succeeded) — it only skips
/// the sidecar.
/// This lib's own post-resolution table entries for downstream builds:
/// `(source_hash, classes, owned interfaces, owned delegates)`. Errors
/// carry the re-analysis diagnostics; callers warn (sidecar) or fall
/// back (embed) instead of failing the build, since lowering already
/// succeeded.
#[allow(clippy::type_complexity)]
fn fragment_payload(
    project: &crate::project::Project,
    unit: &CompilationUnit,
    type_owners: &std::collections::HashMap<TypeKey, String>,
    preloaded: &crate::native::semantic::Preloaded,
) -> std::result::Result<
    (
        u64,
        Vec<crate::native::semantic::ClassInfo>,
        Vec<crate::models::InterfaceDecl>,
        Vec<crate::models::DelegateDecl>,
    ),
    Vec<crate::native::semantic::SemanticError>,
> {
    let table =
        crate::native::semantic::analyze_with_preloaded(unit, type_owners, preloaded)?;
    let hash = rci_source_hash(project);
    // Interfaces/delegates carry no owner field: attribute by
    // owners-map match on (simple name, arity).
    // Over-approximation (same name+arity in two libs) resolves
    // first-wins on consume, mirroring the table's own conflation.
    let owned = |name: &str, arity: usize| {
        type_owners.iter().any(|(k, v)| {
            v == &project.name
                && k.arity == arity
                && k.path.rsplit('.').next().unwrap_or(&k.path) == name
        })
    };
    let ifaces: Vec<_> = table
        .interfaces
        .iter()
        .filter(|i| owned(&i.name, i.arity))
        .cloned()
        .collect();
    let delegates: Vec<_> = table
        .delegates
        .iter()
        .filter(|d| owned(&d.name, d.generic_params.len()))
        .cloned()
        .collect();
    Ok((hash, table.classes, ifaces, delegates))
}

fn emit_rci_sidecar(
    out_dir: &std::path::Path,
    project: &crate::project::Project,
    unit: &CompilationUnit,
    type_owners: &std::collections::HashMap<TypeKey, String>,
    preloaded: &crate::native::semantic::Preloaded,
) {
    let t = std::time::Instant::now();
    match fragment_payload(project, unit, type_owners, preloaded) {
        Ok((hash, classes, ifaces, delegates)) => {
            let bytes =
                crate::native::rci::encode(&project.name, hash, &classes, &ifaces, &delegates);
            let rci_path = out_dir.join(format!("{}.rci", project.name));
            match std::fs::write(&rci_path, &bytes) {
                Ok(()) => eprintln!(
                    "[rubyc] wrote {} ({:.1}s)",
                    rci_path.display(),
                    t.elapsed().as_secs_f64()
                ),
                Err(e) => eprintln!("[rubyc] warning: cannot write {}: {e}", rci_path.display()),
            }
        }
        Err(errs) => eprintln!(
            "[rubyc] warning: skipping .rci sidecar (re-analysis failed: {})",
            errs.iter()
                .map(|e| format!("{} {}", e.code, e.message))
                .collect::<Vec<_>>()
                .join("; ")
        ),
    }
}

/// Build a RubyC project from a `.rcp` file.
fn build_project(path: &str, invocation: &Invocation, pipeline: &NativePipeline) -> Result<()> {    use crate::project::Project;
    let project_path = std::path::PathBuf::from(path);
    let project = Project::load(&project_path).map_err(DriverError::from_msg)?;

    eprintln!("[rubyc] building {} v{}", project.name, project.version);

    if invocation.io.skip_pre_hook {
        eprintln!("[hook:pre-build] skipped (dev)");
    } else {
        project
            .run_hook("pre-build", invocation.io.target.as_deref())
            .map_err(DriverError::from_msg)?;
    }

    // The build is wrapped so the post-build hook always runs: on failure it
    // still cleans up the transient `.rc` sources (the install step is a no-op
    // when no artifact was produced).
    let build = (|| -> Result<()> {
        let t = std::time::Instant::now();
        let (unit, dep_projects, type_owners, fragments) =
            load_project_unit_with_deps(&project_path, invocation, project.dynamic)?;
        eprintln!(
            "[rubyc] timing: load+merge (hooks, preprocess, parse) {:.1}s",
            t.elapsed().as_secs_f64()
        );
        // Dependency `.rc` materializations are cleaned by their own
        // post-build hooks (install is a no-op: no per-dep artifact),
        // on success and on compile failure alike.
        let result = (|| -> Result<()> {
        // A solution delegates to projects, but output naming remains relative
        // to each `.rcp`. Preserve an explicit `-o` for a single-project build.
        let mut project_io = invocation.io.clone();
        project_io.file = Some(project_path.display().to_string());
        project_io.input = None;
        // Emission options: explicit CLI flags join the project's own
        // `[build]` keys — either source enables the pass (a CLI flag
        // never disables a TOML `true`; full CLI-overrides-TOML
        // precedence for valued flags arrives with TOML parity).
        project_io.optimize |= project.optimize;
        project_io.strip |= project.strip;
        // Valued TOML keys apply only when the CLI left them unset:
        // explicit flags always win over the file.
        if project_io.target.is_none() {
            project_io.target.clone_from(&project.target);
        }
        if !project_io.rci_only {
            project_io.rci_only = project.rci_only;
        }
        let project_invocation = Invocation {
            command: invocation.command,
            io: project_io,
            verbose: invocation.verbose,
            quiet: invocation.quiet,
            error_format: invocation.error_format,
            no_color: invocation.no_color,
            lint_overrides: invocation.lint_overrides.clone(),
            defines: invocation.defines.clone(),
        };
        // Honor the project's [target] output (default: exe), unless
        // `-O` overrides it (e.g. `install libs --format` rebuilds one
        // tree in another format without editing 181 project files).
        let out_format = project_invocation
            .io
            .out_format
            .unwrap_or(OutFormat::from_str(&project.output_format).unwrap_or(OutFormat::Exe));
        if out_format == OutFormat::Shared {
            // Dependency fragments for dynamic-linking consume (also
            // reused for the `.rci` emit below so the sidecar mirrors
            // the built table).
            let preloaded = build_preloaded(&fragments);
            // Emit `out/<name>.so` so the post-build hook can install it
            // (the `.rci` below lands next to it, unless embedded).
            let base = project_path.parent().unwrap_or_else(|| std::path::Path::new("."));
            let out_dir = base.join("out");
            std::fs::create_dir_all(&out_dir).map_err(|e| {
                DriverError::from_msg(format!("cannot create {}: {e}", out_dir.display()))
            })?;
            // Embedded by default: fragment bytes ride the post-magic table
            // instead of a sidecar file (no image under `--rci-only`,
            // so nothing to embed into there). Analysis failure falls
            // back to a plain image plus the file.
            let embed = !project_invocation.io.no_embed_types && !project_invocation.io.rci_only;
            let embed_bytes = if embed {
                match fragment_payload(&project, &unit, &type_owners, &preloaded) {
                    Ok((hash, classes, ifaces, delegates)) => Some(crate::native::rci::encode(
                        &project.name,
                        hash,
                        &classes,
                        &ifaces,
                        &delegates,
                    )),
                    Err(errs) => {
                        eprintln!(
                            "[rubyc] warning: embedding skipped (re-analysis failed: {})",
                            errs.iter()
                                .map(|e| format!("{} {}", e.code, e.message))
                                .collect::<Vec<_>>()
                                .join("; ")
                        );
                        None
                    }
                }
            } else {
                None
            };
            // `--rci-only` (fragment backfill): skip codegen and the
            // image write entirely; the sidecar below is the product.
            if !project_invocation.io.rci_only {
                let image = crate::native::compile_shared_library_with_owners(
                    &unit,
                    pipeline.backend,
                    pipeline.image_writer,
                    &project_invocation
                        .io
                        .libs
                        .iter()
                        .map(PathBuf::from)
                        .collect::<Vec<_>>(),
                    // CLI `-D` symbols drive `[Conditional]` omission
                    // (project `[build] defines` are platform symbols,
                    // never conditional ones).
                    &invocation.defines,
                    &type_owners,
                    &preloaded,
                    embed_bytes.as_deref(),
                    crate::native::EmitOptions {
                        optimize: project_invocation.io.optimize,
                        strip: project_invocation.io.strip,
                    },
                )
                .map_err(|e| DriverError::from_msg(e.to_string()))?;
                let out_path = out_dir.join(format!("{}.so", project.name));
                std::fs::write(&out_path, &image).map_err(|e| {
                    DriverError::from_msg(format!("cannot write {}: {e}", out_path.display()))
                })?;
                eprintln!("[rubyc] wrote {}", out_path.display());
            }
            // Same `.rci` sidecar serves dynamic and static consumers —
            // unless the fragment already rides inside the image.
            if !(embed && embed_bytes.is_some()) {
                emit_rci_sidecar(&out_dir, &project, &unit, &type_owners, &preloaded);
            }
        } else if out_format == OutFormat::Static {
            // Static archives merge dependency sources (the `dynamic`
            // fragment path stays shared-only: a dynamic member has no
            // home image to copy-link from in a `.a`).
            if project.dynamic {
                return Err(DriverError::from_msg(format!(
                    "{}: [build] dynamic = true needs output = \"shared\" (static archives link dependency sources, not installed images)",
                    project.name
                )));
            }
            // Embedded by default: second `<name>.rci` archive member instead
            // of a sidecar file (no archive under `--rci-only`, so
            // nothing to embed into there). Analysis failure falls back
            // to a plain archive plus the file.
            let embed = !project_invocation.io.no_embed_types && !project_invocation.io.rci_only;
            let embed_bytes = if embed {
                match fragment_payload(
                    &project,
                    &unit,
                    &type_owners,
                    &build_preloaded(&fragments),
                ) {
                    Ok((hash, classes, ifaces, delegates)) => Some(crate::native::rci::encode(
                        &project.name,
                        hash,
                        &classes,
                        &ifaces,
                        &delegates,
                    )),
                    Err(errs) => {
                        eprintln!(
                            "[rubyc] warning: embedding skipped (re-analysis failed: {})",
                            errs.iter()
                                .map(|e| format!("{} {}", e.code, e.message))
                                .collect::<Vec<_>>()
                                .join("; ")
                        );
                        None
                    }
                }
            } else {
                None
            };
            let archive = crate::native::compile_static_archive(
                &unit,
                pipeline.backend,
                &project.name,
                &invocation.defines,
                embed_bytes.as_deref(),
            )
            .map_err(|e| DriverError::from_msg(e.to_string()))?;
            // Emit `out/<name>.a` so the post-build hook can install it
            // (the `.rci` below lands next to it).
            let base = project_path.parent().unwrap_or_else(|| std::path::Path::new("."));
            let out_dir = base.join("out");
            std::fs::create_dir_all(&out_dir).map_err(|e| {
                DriverError::from_msg(format!("cannot create {}: {e}", out_dir.display()))
            })?;
            // `--rci-only` (fragment backfill): skip codegen and the
            // archive write entirely; the sidecar below is the product.
            if !project_invocation.io.rci_only {
                let out_path = out_dir.join(format!("{}.a", project.name));
                std::fs::write(&out_path, &archive).map_err(|e| {
                    DriverError::from_msg(format!("cannot write {}: {e}", out_path.display()))
                })?;
                eprintln!("[rubyc] wrote {}", out_path.display());
            }
            // Same `.rci` sidecar serves dynamic and static consumers —
            // unless the fragment already rides inside the archive.
            if !(embed && embed_bytes.is_some()) {
                emit_rci_sidecar(
                    &out_dir,
                    &project,
                    &unit,
                    &type_owners,
                    &build_preloaded(&fragments),
                );
            }
        } else {
            emit_exe(
                &project_invocation,
                &Input::Unit(unit),
                &project_invocation.io,
                pipeline,
            )?;
        }
        Ok(())
        })();
        for dep in &dep_projects {
            if !invocation.io.skip_post_hook {
                let _ = dep.run_hook("post-build", invocation.io.target.as_deref());
            }
        }
        result
    })();

    if !invocation.io.skip_post_hook {
        project
            .run_hook("post-build", invocation.io.target.as_deref())
            .map_err(DriverError::from_msg)?;
    }

    match build {
        Ok(()) => {
            eprintln!("[rubyc] {} built successfully", project.name);
            Ok(())
        }
        Err(e) => Err(e),
    }
}

fn build_solution(path: &str, invocation: &Invocation, pipeline: &NativePipeline) -> Result<()> {
    let solution_path = std::path::PathBuf::from(path);
    let solution = crate::project::Solution::load(&solution_path).map_err(DriverError::from_msg)?;
    let projects = solution.project_files().map_err(DriverError::from_msg)?;
    if projects.len() > 1 && invocation.io.output.is_some() {
        return Err(DriverError::from_msg(
            "--output is ambiguous for a solution containing multiple projects",
        ));
    }
    // Parallelism: explicit `--jobs` wins; otherwise the max TOML
    // `[build] jobs` across member projects; unset everywhere = serial.
    let mut jobs = invocation.io.jobs;
    if jobs == 0 {
        for pf in &projects {
            if let Ok(p) = crate::project::Project::load(pf) {
                jobs = jobs.max(p.jobs);
            }
        }
    }
    let jobs = jobs.max(1).min(projects.len().max(1));
    // Solution profile (`--profile NAME`): resolve once, then merge its
    // defines/optimize/strip into the invocation every project builds
    // with. Unknown names fail loudly (a typo must not silently build
    // the wrong configuration).
    let mut invocation = invocation.clone();
    if let Some(name) = invocation.io.profile.clone() {
        match solution.profiles.get(&name) {
            Some(profile) => {
                for d in &profile.defines {
                    if !invocation.defines.contains(d) {
                        invocation.defines.push(d.clone());
                    }
                }
                invocation.io.optimize |= profile.optimize;
                invocation.io.strip |= profile.strip;
            }
            None => {
                let mut known: Vec<&String> = solution.profiles.keys().collect();
                known.sort();
                return Err(DriverError::from_msg(format!(
                    "unknown solution profile '{name}' (known: {})",
                    known
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )));
            }
        }
    }
    if jobs <= 1 {
        for project in &projects {
            build_project(&project.display().to_string(), &invocation, pipeline)?;
        }
        return Ok(());
    }
    build_projects_parallel(&projects, &invocation, pipeline, jobs)
}

/// Build independent projects, each in its own thread pulled from an
/// atomic counter (tail work balances itself — no chunking). Projects
/// are independent builds (distinct sources, distinct `out/`
/// artifacts, per-command hook dirs); the shared pipeline ports are
/// stateless. Every project runs to completion; the first error in
/// project order wins after the join.
fn build_projects_parallel(
    projects: &[std::path::PathBuf],
    invocation: &Invocation,
    pipeline: &NativePipeline,
    jobs: usize,
) -> Result<()> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: Vec<std::sync::Mutex<Option<Result<()>>>> =
        (0..projects.len()).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|s| {
        for _ in 0..jobs {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= projects.len() {
                    break;
                }
                let r = build_project(
                    &projects[i].display().to_string(),
                    invocation,
                    pipeline,
                );
                *results[i].lock().unwrap() = Some(r);
            });
        }
    });
    for cell in &results {
        if let Some(Err(e)) = cell.lock().unwrap().take() {
            return Err(e);
        }
    }
    Ok(())
}

fn cmd_build(invocation: &Invocation, pipeline: &NativePipeline) -> Result<()> {
    let io = &invocation.io;
    if invocation.verbose {
        eprintln!("{}", crate::t!("rubyc.main.build.stage1"));
    }
    // Pulled-tree builds fetch first: a requested `.rcp`/`.rcs` under
    // `dotnet-runtime/` that is not on disk means the tree was never
    // pulled (ordinary typos elsewhere never trigger this).
    if let Some(f) = io.file.as_deref() {
        if f.contains("dotnet-runtime") && !std::path::Path::new(f).is_file() {
            eprintln!("[rubyc] build: {f} is missing from the pulled tree; pulling first ...");
            run_pull_script()?;
        }
    }
    // Project/solution files take the project system path.
    if let Some(f) = io.file.as_deref() {
        let lower = f.to_lowercase();
        if lower.ends_with(".rcp") {
            return build_project(f, invocation, pipeline);
        }
        if lower.ends_with(".rcs") {
            return build_solution(f, invocation, pipeline);
        }
    }
    // .asm files take the assembler path directly.
    if io
        .file
        .as_deref()
        .map(|f| f.ends_with(".asm"))
        .unwrap_or(false)
    {
        return build_asm_file(io.file.as_deref().unwrap(), io, invocation, pipeline);
    }
    // Auto-detect nearest project file when no explicit file given.
    if io.file.is_none() && io.input.is_none() {
        let cwd = std::env::current_dir().unwrap_or_default();
        if let Some(project_path) = crate::project::Project::find_nearest(&cwd) {
            let path = project_path.to_str().unwrap_or("project.rcp");
            return if path.to_ascii_lowercase().ends_with(".rcs") {
                build_solution(path, invocation, pipeline)
            } else {
                build_project(path, invocation, pipeline)
            };
        }
    }
    let input = read_input(io, pipeline)?;
    if invocation.verbose {
        eprintln!("{}", crate::t!("rubyc.main.build.stage2"));
    }
    // Bytecode variants lint the parsed unit before encoding.
    if matches!(
        io.out_format,
        Some(OutFormat::Bytecode)
            | Some(OutFormat::BytecodeShebang)
            | Some(OutFormat::Base64Bytecode)
    ) {
        let unit = into_unit(input.clone(), invocation)?;
        check_lints(&unit, invocation)?;
    }
    // `build` emits a native executable by default; `--out-format` (or
    // `--shebang`) selects the other image kinds.
    let out_format = match io.out_format {
        Some(f) => f,
        None if io.shebang.is_some() => OutFormat::BytecodeShebang,
        None => OutFormat::Exe,
    };
    match out_format {
        OutFormat::Exe => return emit_exe(invocation, &input, io, pipeline),
        OutFormat::Shared => {
            let unit = into_unit(input, invocation)?;
            check_lints(&unit, invocation)?;
            // Direct compiles have no project: fragment attribution is
            // empty-owners (everything local), hash 0. Analysis failure
            // falls back to a plain image, never fails the build.
            let embed_bytes = if !io.no_embed_types {
                match crate::native::analyze_fragment_bytes(
                    &unit,
                    &std::collections::HashMap::new(),
                    "lib",
                    0,
                ) {
                    Ok(bytes) => Some(bytes),
                    Err(_) => {
                        eprintln!("[rubyc] warning: embedding skipped (analysis failed)");
                        None
                    }
                }
            } else {
                None
            };
            let image = crate::native::compile_shared_library(
                &unit,
                pipeline.backend,
                pipeline.image_writer,
                &io.libs.iter().map(PathBuf::from).collect::<Vec<_>>(),
                &invocation.defines,
                embed_bytes.as_deref(),
                crate::native::EmitOptions {
                    optimize: io.optimize,
                    strip: io.strip,
                },
            )
            .map_err(|e| DriverError::from_msg(e.to_string()))?;
            write_out(io, &image)?;
        }
        OutFormat::Static => {
            let unit = into_unit(input, invocation)?;
            check_lints(&unit, invocation)?;
            // Member name defaults to the output file stem when `-o`
            // names the archive (`foo.a` -> member `foo`), else `lib`.
            let member = io
                .output
                .as_deref()
                .and_then(|o| {
                    std::path::Path::new(o)
                        .file_stem()
                        .and_then(|s| s.to_str())
                })
                .unwrap_or("lib")
                .to_owned();
            // Direct compiles have no project: fragment attribution is
            // empty-owners (everything local), hash 0. Analysis failure
            // falls back to a plain archive, never fails the build.
            let embed_bytes = if !io.no_embed_types {
                match crate::native::analyze_fragment_bytes(
                    &unit,
                    &std::collections::HashMap::new(),
                    &member,
                    0,
                ) {
                    Ok(bytes) => Some(bytes),
                    Err(_) => {
                        eprintln!("[rubyc] warning: embedding skipped (analysis failed)");
                        None
                    }
                }
            } else {
                None
            };
            let archive = crate::native::compile_static_archive(
                &unit,
                pipeline.backend,
                &member,
                &invocation.defines,
                embed_bytes.as_deref(),
            )
            .map_err(|e| DriverError::from_msg(e.to_string()))?;
            write_out(io, &archive)?;
        }
        OutFormat::Bytecode => {
            let bytes = ast_bytes_from_input(input, invocation)?;
            if io.base64 {
                write_out(io, base64_encode(&bytes).as_bytes())?;
            } else {
                write_out(io, &bytes)?;
            }
        }
        OutFormat::BytecodeShebang => {
            let bytes = ast_bytes_from_input(input, invocation)?;
            let interpreter = io.shebang.clone().unwrap_or_else(default_interpreter);
            write_script(io, &wrap_shebang(&bytes, &interpreter))?;
        }
        OutFormat::Base64Bytecode => {
            let bytes = ast_bytes_from_input(input, invocation)?;
            write_out(io, base64_encode(&bytes).as_bytes())?;
        }
    }
    if !invocation.quiet {
        report(io, invocation.verbose, "rubyc.main.report.compiled");
    }
    Ok(())
}

/// Run a `dynamic = true` project against installed dependency images:
/// load the dep closure with fragment consume, lower with owners, and
/// resolve imports at load time before execution. Returns the process
/// exit status; non-zero is reported by the caller.
fn run_project_with_dyn(
    project_path: &std::path::Path,
    invocation: &Invocation,
    pipeline: &NativePipeline,
) -> Result<()> {
    let t = std::time::Instant::now();
    let (unit, _deps, type_owners, fragments) =
        load_project_unit_with_deps(project_path, invocation, true)?;
    eprintln!(
        "[rubyc] timing: load+merge (hooks, preprocess, parse) {:.1}s",
        t.elapsed().as_secs_f64()
    );
    let preloaded = {
        let mut classes = Vec::new();
        let mut interfaces = Vec::new();
        let mut delegates = Vec::new();
        for frag in &fragments {
            classes.extend(frag.classes.iter().cloned());
            interfaces.extend(frag.interfaces.iter().cloned());
            delegates.extend(frag.delegates.iter().cloned());
        }
        crate::native::semantic::Preloaded {
            classes,
            interfaces,
            delegates,
        }
    };
    let dyn_dir = crate::extensions::install_root().join("libraries");
    let status = crate::native::jit::jit_run_with_owners(
        &unit,
        pipeline.backend,
        pipeline.memory,
        &[],
        &dyn_dir,
        &type_owners,
        &preloaded,
    )
    .map_err(|e| DriverError::from_msg(e.to_string()))?;
    if status != 0 {
        eprintln!("{}", crate::tf!("rubyc.main.run.failed_status", status));
        return Err(DriverError::ExitStatus(status));
    }
    Ok(())
}

fn cmd_run(invocation: &Invocation, pipeline: &NativePipeline) -> Result<()> {    let io = &invocation.io;
    if invocation.verbose {
        eprintln!("{}", crate::t!("rubyc.main.run.stage1"));
    }
    // Solutions can't be JIT-run directly; build them instead.
    if io
        .file
        .as_deref()
        .map(|f| f.to_lowercase().ends_with(".rcs"))
        .unwrap_or(false)
    {
        return build_solution(io.file.as_deref().unwrap(), invocation, pipeline);
    }
    let input = if let Some(path) = io.file.as_deref().filter(|path| path.ends_with(".rcp")) {
        // Dynamic projects run against installed dependency images:
        // load the dep closure with fragment consume, lower with
        // owners, and resolve imports at load time. Static projects
        // take the plain path below (unchanged behavior).
        let project_path = std::path::Path::new(path);
        let dynamic = crate::project::Project::load(project_path)
            .map(|p| p.dynamic)
            .unwrap_or(false);
        if dynamic {
            return run_project_with_dyn(project_path, invocation, pipeline);
        }
        Input::Unit(load_project_unit(project_path, invocation)?)
    } else if io.file.is_none() && io.input.is_none() {
        // Auto-detect the nearest project file when none is given.
        let cwd = std::env::current_dir().unwrap_or_default();
        match crate::project::Project::find_nearest(&cwd) {
            Some(p) => {
                let path = p.to_str().unwrap_or("project.rcp").to_string();
                if path.to_ascii_lowercase().ends_with(".rcs") {
                    return build_solution(&path, invocation, pipeline);
                }
                Input::Unit(load_project_unit(std::path::Path::new(&path), invocation)?)
            }
            None => read_input(io, pipeline)?,
        }
    } else {
        read_input(io, pipeline)?
    };
    if invocation.verbose {
        eprintln!("{}", crate::t!("rubyc.main.run.stage2"));
    }
    let unit = into_unit(input, invocation)?;
    check_lints(&unit, invocation)?;
    let dyn_dir = crate::extensions::install_root().join("libraries");
    let status = crate::native::jit::jit_run_with_dyn_libs(
        &unit,
        pipeline.backend,
        pipeline.memory,
        &[],
        &dyn_dir,
    )
    .map_err(|e| DriverError::from_msg(e.to_string()))?;
    if status != 0 {
        eprintln!("{}", crate::tf!("rubyc.main.run.failed_status", status));
        return Err(DriverError::ExitStatus(status));
    }
    Ok(())
}

fn cmd_dump(io: &IoArgs, invocation: &Invocation, pipeline: &NativePipeline) -> Result<()> {
    let raw = read_raw(io)?;
    match classify(&raw, pipeline)? {
        Input::Source(text) => {
            // `dump` shows the AST only, never the source text.
            // Preprocess exactly like every other source path so `#If`
            // regions, macros, and `#Assembly` blocks behave identically.
            let filename = invocation
                .io
                .file
                .clone()
                .unwrap_or_else(|| "<input>".to_owned());
            let pre = crate::preprocessor::preprocess(&text, &invocation.defines, &filename)
                .map_err(|diagnostics| {
                    DriverError::Message(
                        diagnostics
                            .iter()
                            .map(|e| e.to_string())
                            .collect::<Vec<_>>()
                            .join("; "),
                    )
                })?;
        let expanded = crate::preprocessor::expand_macros(pre.source, &pre.macros);
            let unit = parser::parse(&expanded).map_err(|errors| {
                DriverError::Message(
                    errors
                        .iter()
                        .map(|e| e.to_string())
                        .collect::<Vec<_>>()
                        .join("; "),
                )
            })?;
            println!(
                "{}\n{unit:#?}",
                crate::t!("rubyc.main.dump.ast_from_source")
            );
            if invocation.verbose {
                dump_ir_and_mc(&unit, pipeline)?;
            }
        }
        Input::Unit(unit) => {
            println!(
                "{}\n{unit:#?}",
                crate::t!("rubyc.main.dump.ast_from_bytecode")
            );
            if invocation.verbose {
                dump_ir_and_mc(&unit, pipeline)?;
            }
        }
    }
    Ok(())
}

/// Lower a unit to IR and (if a target is available) to MachineCode,
/// printing their Debug representations to stdout.
fn dump_ir_and_mc(unit: &CompilationUnit, pipeline: &NativePipeline) -> Result<()> {
    let program = crate::native::ir::lower(unit).map_err(|e| {
        DriverError::Message(crate::tf!("rubyc.main.dump.pipeline_failed", e.to_string()))
    })?;
    println!(
        "{}\n{program:#?}",
        crate::t!("rubyc.main.dump.ir_header")
    );
    match pipeline.backend.lower(&program) {
        Ok(mc) => {
            println!(
                "{}\n{mc:#?}",
                crate::t!("rubyc.main.dump.mc_header")
            );
        }
        Err(e) => {
            println!(
                "{}\n  {}",
                crate::t!("rubyc.main.dump.mc_header"),
                crate::tf!("rubyc.main.dump.pipeline_failed", e.to_string())
            );
        }
    }
    Ok(())
}

fn cmd_check_roundtrip(file: Option<&str>) -> Result<()> {
    let mut failures = 0;
    for sample in crate::testdata::SAMPLES {
        if roundtrip_ok(sample.source) {
            println!(
                "{}",
                crate::tf!("rubyc.main.checkroundtrip.ok", sample.name)
            );
        } else {
            println!(
                "{}",
                crate::tf!("rubyc.main.checkroundtrip.fail", sample.name)
            );
            failures += 1;
        }
    }
    if let Some(file) = file {
        let source = std::fs::read_to_string(file)
            .map_err(|e| DriverError::from_msg(format!("cannot read {file}: {e}")))?;
        if roundtrip_ok(&source) {
            println!("{}", crate::tf!("rubyc.main.checkroundtrip.ok", file));
        } else {
            println!("{}", crate::tf!("rubyc.main.checkroundtrip.fail", file));
            failures += 1;
        }
    }
    if failures > 0 {
        Err(DriverError::Message(crate::tf!(
            "rubyc.main.checkroundtrip.failures",
            failures
        )))
    } else {
        Ok(())
    }
}

// -------------------------------------------------------- extension management

fn cmd_install(invocation: &Invocation, pipeline: &NativePipeline) -> Result<()> {
    let name = invocation
        .io
        .file
        .as_deref()
        .ok_or_else(|| DriverError::Message("install: missing package name".into()))?;
    if name == "libs" {
        return cmd_install_libs(invocation, pipeline);
    }
    let kind = invocation
        .io
        .kind
        .as_deref()
        .ok_or_else(|| DriverError::Message("install: --kind target is required".into()))?;
    if kind != "target" {
        return Err(DriverError::Message(format!(
            "install: invalid --kind '{kind}' (expected 'target')"
        )));
    }
    let source = invocation
        .io
        .url
        .as_deref()
        .ok_or_else(|| DriverError::Message("install: --url (path to .so) is required".into()))?;

    let base = resolve_base(&invocation.io);
    let subdir = "targets";
    let plugin_dir = base.join(subdir).join(name);
    std::fs::create_dir_all(&plugin_dir).map_err(|e| {
        DriverError::Message(format!("install: cannot create {}: {e}", plugin_dir.display()))
    })?;

    // Symlink the .so into place.
    let source_path = std::path::PathBuf::from(source);
    if !source_path.exists() {
        return Err(DriverError::Message(format!(
            "install: source not found: {source}"
        )));
    }
    let lib_name = format!("lib{}.so", name.replace('-', "_"));
    let link_path = plugin_dir.join(&lib_name);
    // Remove existing link/file if present.
    let _ = std::fs::remove_file(&link_path);
    #[cfg(unix)]
    std::os::unix::fs::symlink(&source_path, &link_path).map_err(|e| {
        DriverError::Message(format!("install: symlink failed: {e}"))
    })?;
    #[cfg(not(unix))]
    std::fs::copy(&source_path, &link_path).map_err(|e| {
        DriverError::Message(format!("install: copy failed: {e}"))
    })?;

    let manifest_name = format!("{name}.toml");
    let manifest_path = plugin_dir.join(&manifest_name);
    let triple = if kind == "target" {
        "x86_64-unknown-linux-gnu".to_string()
    } else {
        String::new()
    };
    let manifest = format!(
        "name = \"{name}\"\nversion = \"1.0.0\"\nkind = \"{kind}\"\nlibrary = \"{lib_name}\"\n{triple_line}\n",
        triple_line = if triple.is_empty() { String::new() } else { format!("triple = \"{triple}\"\n") }
    );
    std::fs::write(&manifest_path, manifest).map_err(|e| {
        DriverError::Message(format!("install: cannot write manifest: {e}"))
    })?;

    println!(
        "{}",
        crate::tf!("rubyc.cli.installed", name.to_string())
    );
    Ok(())
}

/// Walk upward from the current directory for `rel` (a `/`-separated
/// path like `dotnet-runtime/src/libraries/dotnet-runtime.rcs`).
/// Returns the first hit, or `None` past the filesystem root.
fn find_upward(rel: &str) -> Option<std::path::PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let candidate = dir.join(rel);
        if candidate.is_file() {
            return Some(candidate);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Ensure the pulled runtime tree, returning its solution path.
/// Missing tree (or missing solution): run `tools/pull-dotnet.sh`
/// from the enclosing checkout (found upward, like the tree) with
/// inherited stdio so pull progress is visible; a missing script or
/// a still-missing tree afterwards is a loud error (offline with
/// nothing cached cannot proceed). Present tree: best-effort
/// update check — behind upstream runs the pull script, network
/// failure warns and proceeds with the local tree.
fn ensure_pulled_tree() -> Result<std::path::PathBuf> {
    let rel = "dotnet-runtime/src/libraries/dotnet-runtime.rcs";
    if let Some(rcs) = find_upward(rel) {
        if rcs.is_file() {
            if tree_is_behind(&rcs) {
                run_pull_script()?;
            }
            return Ok(rcs);
        }
    }
    run_pull_script()?;
    find_upward(rel).filter(|p| p.is_file()).ok_or_else(|| {
        DriverError::Message(
            "install libs: pull script ran but produced no dotnet-runtime/src/libraries/dotnet-runtime.rcs".into(),
        )
    })
}

/// Run `tools/pull-dotnet.sh` from the enclosing checkout (found
/// upward, like the tree) with inherited stdio so pull progress is
/// visible. Missing script or failing run is a loud error: offline
/// with nothing cached cannot proceed.
fn run_pull_script() -> Result<()> {
    let script = find_upward("tools/pull-dotnet.sh").ok_or_else(|| {
        DriverError::Message(
            "install libs: no pulled runtime tree found (no dotnet-runtime/src/libraries/dotnet-runtime.rcs upward) and no tools/pull-dotnet.sh to fetch it; clone the rubyc repo or run from inside it".into(),
        )
    })?;
    eprintln!("[rubyc] install libs: pulling dotnet/runtime ...");
    let status = std::process::Command::new("bash")
        .arg(&script)
        .status()
        .map_err(|e| {
            DriverError::Message(format!("install libs: cannot run pull script: {e}"))
        })?;
    if !status.success() {
        return Err(DriverError::Message(
            "install libs: pull script failed (offline?); nothing cached to fall back on".into(),
        ));
    }
    Ok(())
}

/// Whether a present tree is behind upstream (local HEAD vs
/// `ls-remote`). Non-git trees and network failures report false
/// with a warning and proceed on the local tree.
fn tree_is_behind(rcs: &std::path::Path) -> bool {
    let root = rcs
        .ancestors()
        .find(|a| a.join(".git").is_dir() && a.ends_with("dotnet-runtime"));
    let Some(root) = root else { return false };
    let local = std::process::Command::new("git")
        .args(["-C"])
        .arg(root)
        .args(["rev-parse", "HEAD"])
        .output();
    let remote = std::process::Command::new("git")
        .args(["ls-remote", "origin", "HEAD"])
        .current_dir(root)
        .output();
    match (local, remote) {
        (Ok(l), Ok(r)) if l.status.success() && r.status.success() => {
            let l = String::from_utf8_lossy(&l.stdout);
            let r = String::from_utf8_lossy(&r.stdout);
            !l.trim().is_empty() && !r.is_empty() && !r.starts_with(l.trim())
        }
        _ => {
            eprintln!("[rubyc] install libs: update check skipped (offline or unreachable); using the local tree");
            false
        }
    }
}

/// Whether an installed image is fresh for a project: the image
/// (`.so`, or `static/<name>.a`) exists and its embedded-or-sidecar
/// fragment decodes with a matching lib name and a source hash equal
/// to a fresh hash of the project's current own sources. Anything
/// else (missing, undecodable, stale) means "build it".
fn lib_is_fresh(
    project_path: &std::path::Path,
    want_static: bool,
    dyn_dir: &std::path::Path,
) -> bool {
    let Ok(project) = crate::project::Project::load(project_path) else {
        return false;
    };
    let image = if want_static {
        dyn_dir.join("static").join(format!("{}.a", project.name))
    } else {
        dyn_dir.join(format!("{}.so", project.name))
    };
    let Ok(bytes) = std::fs::read(&image) else {
        return false;
    };
    let frag_bytes: Option<Vec<u8>> = if want_static {
        crate::native::arlib::read(&bytes)
            .ok()
            .and_then(|members| {
                crate::native::embedded::archive_member(&members).map(|b| b.to_vec())
            })
            .or_else(|| {
                std::fs::read(dyn_dir.join(format!("{}.rci", project.name))).ok()
            })
    } else {
        crate::native::embedded::table_bytes(&bytes)
            .or_else(|| crate::native::embedded::section_bytes(&bytes))
            .map(|b| b.to_vec())
            .or_else(|| std::fs::read(dyn_dir.join(format!("{}.rci", project.name))).ok())
    };
    let Some(frag_bytes) = frag_bytes else {
        return false;
    };
    let Ok(frag) = crate::native::rci::decode(&frag_bytes) else {
        return false;
    };
    frag.lib_name == project.name && frag.source_hash == rci_source_hash(&project)
}

/// `install libs [--format shared|static|both]`: ensure the pulled
/// runtime tree, rebuild stale libraries, install via the post-build
/// hooks. Each stage skips fresh work: pull runs only when the tree is
/// missing or behind upstream, and only libraries whose installed
/// image predates their inputs rebuild. `--format both` runs a shared
/// pass then a static pass (dynamic projects skip the static pass —
/// their members have no home image to copy-link from).
fn cmd_install_libs(invocation: &Invocation, pipeline: &NativePipeline) -> Result<()> {
    let formats: Vec<bool> = match invocation.io.install_format.as_deref().unwrap_or("shared") {
        "shared" => vec![false],
        "static" => vec![true],
        "both" => vec![false, true],
        other => {
            return Err(DriverError::Message(format!(
                "install libs: invalid --format '{other}' (expected 'shared', 'static', or 'both')"
            )));
        }
    };
    let rcs = ensure_pulled_tree()?;
    let solution = crate::project::Solution::load(&rcs).map_err(DriverError::from_msg)?;
    let projects = solution.project_files().map_err(DriverError::from_msg)?;
    let dyn_dir = crate::extensions::install_root().join("libraries");
    for want_static in formats {
        let mut stale = Vec::new();
        let mut skipped_dynamic = 0usize;
        for project in &projects {
            // Dynamic members cannot static-link (no home image to
            // copy-link from): skip them in the static pass with a note,
            // erroring only on explicit single-project requests (see
            // `build_project`). Unreadable projects stay stale so the
            // build reports them instead of silently skipping.
            let dynamic = crate::project::Project::load(project)
                .map(|p| p.dynamic)
                .unwrap_or(false);
            if want_static && dynamic {
                skipped_dynamic += 1;
                continue;
            }
            if !lib_is_fresh(project, want_static, &dyn_dir) {
                stale.push(project.clone());
            }
        }
        let kind = if want_static { "static" } else { "shared" };
        if stale.is_empty() {
            eprintln!("[rubyc] install libs: all {} libraries up to date ({kind})", projects.len());
            continue;
        }
        eprintln!(
            "[rubyc] install libs: building {} of {} libraries ({kind})",
            stale.len(),
            projects.len()
        );
        if skipped_dynamic > 0 {
            eprintln!(
                "[rubyc] install libs: skipping {skipped_dynamic} dynamic projects for the static pass"
            );
        }
        let mut pass = invocation.clone();
        pass.io.out_format = Some(if want_static {
            OutFormat::Static
        } else {
            OutFormat::Shared
        });
        let jobs = pass.io.jobs.max(1).min(stale.len().max(1));
        if jobs <= 1 {
            for project in &stale {
                build_project(&project.display().to_string(), &pass, &pipeline)?;
            }
        } else {
            build_projects_parallel(&stale, &pass, &pipeline, jobs)?;
        }
    }
    Ok(())
}

fn cmd_remove(invocation: &Invocation) -> Result<()> {
    let name = invocation
        .io
        .file
        .as_deref()
        .ok_or_else(|| DriverError::Message("remove: missing package name or project".into()))?;

    // No `--kind`: remove a project from the nearest solution (`.rcs`) instead
    // of removing an installed plugin.
    let Some(kind) = invocation.io.kind.as_deref() else {
        return remove_solution_project(name);
    };

    if kind != "target" {
        return Err(DriverError::Message(format!(
            "remove: invalid --kind '{kind}' (expected 'target')"
        )));
    }
    let base = resolve_base(&invocation.io);
    let subdir = "targets";
    let plugin_dir = base.join(subdir).join(name);
    if !plugin_dir.exists() {
        return Err(DriverError::Message(format!(
            "remove: {name} not installed"
        )));
    }
    std::fs::remove_dir_all(&plugin_dir).map_err(|e| {
        DriverError::Message(format!("remove: cannot remove {}: {e}", plugin_dir.display()))
    })?;
    println!(
        "{}",
        crate::tf!("rubyc.cli.removed", name.to_string())
    );
    Ok(())
}

// ---------- project / solution scaffolding ----------

/// Find the nearest `.rcs` solution walking up from `start_dir`.
fn find_nearest_solution(start_dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let mut dir = Some(start_dir.to_path_buf());
    while let Some(d) = dir {
        if let Ok(entries) = std::fs::read_dir(&d) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("rcs") {
                    return Some(path);
                }
            }
        }
        dir = d.parent().map(|p| p.to_path_buf());
    }
    None
}

/// Relative path from `base` to `target` (both should share a prefix).
fn relative_path(target: &std::path::Path, base: &std::path::Path) -> std::path::PathBuf {
    if let Ok(r) = target.strip_prefix(base) {
        return r.to_path_buf();
    }
    let mut up = base.to_path_buf();
    let mut prefix = std::path::PathBuf::new();
    loop {
        if let Ok(r) = target.strip_prefix(&up) {
            return prefix.join(r);
        }
        match up.parent() {
            Some(p) => {
                up = p.to_path_buf();
                prefix.push("..");
            }
            None => return target.to_path_buf(),
        }
    }
}

fn cmd_new(invocation: &Invocation) -> Result<()> {
    let name = invocation
        .io
        .file
        .as_deref()
        .ok_or_else(|| DriverError::Message("new: missing project or solution name".into()))?;
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(DriverError::Message(format!(
            "new: invalid name '{name}' (use letters, digits, '-', '_')"
        )));
    }
    let cwd = std::env::current_dir()
        .map_err(|e| DriverError::Message(format!("new: cannot resolve cwd: {e}")))?;

    if invocation.io.solution {
        let path = cwd.join(format!("{name}.rcs"));
        if path.exists() {
            return Err(DriverError::Message(format!(
                "new: {} already exists",
                path.display()
            )));
        }
        std::fs::write(&path, solution_template(name)).map_err(|e| {
            DriverError::Message(format!("new: cannot write {}: {e}", path.display()))
        })?;
        println!("created solution {}", path.display());
        return Ok(());
    }

    let dir = cwd.join(name);
    if dir.exists() {
        return Err(DriverError::Message(format!(
            "new: {} already exists",
            dir.display()
        )));
    }
    std::fs::create_dir_all(&dir).map_err(|e| {
        DriverError::Message(format!("new: cannot create {}: {e}", dir.display()))
    })?;
    let rcp = dir.join("project.rcp");
    std::fs::write(&rcp, project_rcp_template(name)).map_err(|e| {
        DriverError::Message(format!("new: cannot write {}: {e}", rcp.display()))
    })?;
    let main_rc = dir.join("main.rc");
    std::fs::write(&main_rc, main_rc_template()).map_err(|e| {
        DriverError::Message(format!("new: cannot write {}: {e}", main_rc.display()))
    })?;
    println!(
        "created project {}\n  {}\n  {}",
        dir.display(),
        rcp.display(),
        main_rc.display()
    );
    Ok(())
}

fn cmd_add(invocation: &Invocation) -> Result<()> {
    let project_arg = invocation
        .io
        .file
        .as_deref()
        .ok_or_else(|| DriverError::Message("add: missing project path (.rcp)".into()))?;
    let cwd = std::env::current_dir()
        .map_err(|e| DriverError::Message(format!("add: cannot resolve cwd: {e}")))?;
    let project = if std::path::Path::new(project_arg).is_absolute() {
        std::path::PathBuf::from(project_arg)
    } else {
        cwd.join(project_arg)
    };
    if !project.is_file() {
        return Err(DriverError::Message(format!(
            "add: project file not found: {}",
            project.display()
        )));
    }
    if project.extension().and_then(|e| e.to_str()) != Some("rcp") {
        return Err(DriverError::Message(format!(
            "add: not a .rcp project: {}",
            project.display()
        )));
    }
    let solution = find_nearest_solution(&cwd).ok_or_else(|| {
        DriverError::Message(format!(
            "add: no .rcs solution found in or above {}",
            cwd.display()
        ))
    })?;

    let sol_dir = solution
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let rel = relative_path(&project, sol_dir);
    let entry = format!("path = \"{}\"", rel.display());

    let text = std::fs::read_to_string(&solution).map_err(|e| {
        DriverError::Message(format!("add: cannot read {}: {e}", solution.display()))
    })?;
    if text.lines().any(|l| l.trim() == entry) {
        println!("{} is already in {}", rel.display(), solution.display());
        return Ok(());
    }

    let mut new_text = text;
    if !new_text.ends_with('\n') {
        new_text.push('\n');
    }
    new_text.push_str("\n[[project]]\n");
    new_text.push_str(&entry);
    new_text.push('\n');
    std::fs::write(&solution, new_text).map_err(|e| {
        DriverError::Message(format!("add: cannot write {}: {e}", solution.display()))
    })?;
    println!("added {} to {}", rel.display(), solution.display());
    Ok(())
}

fn remove_solution_project(name: &str) -> Result<()> {
    let cwd = std::env::current_dir()
        .map_err(|e| DriverError::Message(format!("remove: cannot resolve cwd: {e}")))?;
    let solution = find_nearest_solution(&cwd).ok_or_else(|| {
        DriverError::Message(format!(
            "remove: no .rcs solution found in or above {}",
            cwd.display()
        ))
    })?;
    let text = std::fs::read_to_string(&solution).map_err(|e| {
        DriverError::Message(format!("remove: cannot read {}: {e}", solution.display()))
    })?;
    let project_base = std::path::Path::new(name)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(name);

    let lines: Vec<&str> = text.lines().collect();
    let mut out = String::new();
    let mut removed = false;
    let mut i = 0usize;
    while i < lines.len() {
        let line = lines[i];
        if line.trim() == "[[project]]" {
            // The block's `path = "…"` value is the next one before the next section.
            let mut path_value: Option<String> = None;
            let mut j = i + 1;
            while j < lines.len()
                && lines[j].trim() != "[[project]]"
                && !lines[j].trim().starts_with('[')
            {
                if let Some(r) = lines[j].trim().strip_prefix("path") {
                    if let Some(v) = r.trim().strip_prefix('=') {
                        path_value = Some(v.trim().trim_matches('"').to_owned());
                    }
                }
                j += 1;
            }
            let is_match = path_value
                .as_deref()
                .map(|p| {
                    p == name
                        || std::path::Path::new(p)
                            .file_name()
                            .and_then(|n| n.to_str())
                            == Some(project_base)
                })
                .unwrap_or(false);
            if is_match {
                removed = true;
                i = j;
                continue;
            }
        }
        out.push_str(line);
        out.push('\n');
        i += 1;
    }
    if !removed {
        return Err(DriverError::Message(format!(
            "remove: '{name}' is not a project in {}",
            solution.display()
        )));
    }
    std::fs::write(&solution, out).map_err(|e| {
        DriverError::Message(format!("remove: cannot write {}: {e}", solution.display()))
    })?;
    println!("removed {name} from {}", solution.display());
    Ok(())
}

fn project_rcp_template(name: &str) -> String {
    format!(
        "[meta]\n\
         name = \"{name}\"\n\
         version = \"0.1.0\"\n\
         author = \"\"\n\
         description = \"{name} project\"\n\
         \n\
          [target]\n\
          platform = \"linux_x86_64\"\n\
          output = \"exe\"\n\
         \n\
         [build]\n\
         sources = [\"*.rc\"]\n\
         entry-point = \"main\"\n"
    )
}

fn main_rc_template() -> String {
    "namespace app {\n\
     \n\
     class Program {\n\
     \n\
     int main() {\n\
     \n\
     int answer = 42;\n\
     \n\
     printf(\"Hello from RubyC! {0}\", answer);\n\
     \n\
     return 0;\n\
     }\n\
     }\n\
     }"
        .to_owned()
}

fn solution_template(name: &str) -> String {
    format!("solution \"{name}\"\nversion = \"0.1.0\"\n")
}

fn cmd_update(invocation: &Invocation) -> Result<()> {
    let name = invocation
        .io
        .file
        .as_deref()
        .ok_or_else(|| DriverError::Message("update: missing package name".into()))?;
    let kind = invocation
        .io
        .kind
        .as_deref()
        .ok_or_else(|| DriverError::Message("update: --kind target is required".into()))?;
    if kind != "target" {
        return Err(DriverError::Message(format!(
            "update: invalid --kind '{kind}' (expected 'target')"
        )));
    }
    let source = invocation
        .io
        .url
        .as_deref()
        .ok_or_else(|| DriverError::Message("update: --url (path to .so) is required".into()))?;

    let base = resolve_base(&invocation.io);
    let subdir = "targets";
    let plugin_dir = base.join(subdir).join(name);
    if !plugin_dir.exists() {
        return Err(DriverError::Message(format!(
            "update: {name} not installed; run install first"
        )));
    }

    let source_path = std::path::PathBuf::from(source);
    if !source_path.exists() {
        return Err(DriverError::Message(format!(
            "update: source not found: {source}"
        )));
    }
    let lib_name = format!("lib{}.so", name.replace('-', "_"));
    let link_path = plugin_dir.join(&lib_name);
    let _ = std::fs::remove_file(&link_path);
    #[cfg(unix)]
    std::os::unix::fs::symlink(&source_path, &link_path).map_err(|e| {
        DriverError::Message(format!("update: symlink failed: {e}"))
    })?;
    #[cfg(not(unix))]
    std::fs::copy(&source_path, &link_path).map_err(|e| {
        DriverError::Message(format!("update: copy failed: {e}"))
    })?;

    println!(
        "{}",
        crate::tf!("rubyc.cli.updated", name.to_string())
    );
    Ok(())
}

fn cmd_list_remote() -> Result<()> {
    Err(DriverError::Message(
        crate::tf!(
            "rubyc.cli.err.not_implemented",
            "list-remote".to_string()
        ),
    ))
}

fn cmd_download(invocation: &Invocation) -> Result<()> {
    let name = invocation
        .io
        .file
        .as_deref()
        .ok_or_else(|| DriverError::Message("download: missing package name".into()))?;
    Err(DriverError::Message(
        crate::tf!("rubyc.cli.err.not_implemented", format!("download {name}")),
    ))
}

fn cmd_list(invocation: &Invocation) -> Result<()> {
    let _base = resolve_base(&invocation.io);
    let registry = crate::extensions::ExtensionRegistry::discover_with(
        invocation.io.base.as_deref().map(|s| s.as_ref()),
        invocation.io.targets_dir.as_deref().map(|s| s.as_ref()),
    );
    for target in registry.targets() {
        println!(
            "{:<12} {} {} | {} | {} | {}",
            "target",
            target.metadata.name,
            target.metadata.version,
            target.metadata.author,
            target.triple,
            target.metadata.description
        );
    }
    Ok(())
}

fn cmd_lint(invocation: &Invocation, pipeline: &NativePipeline) -> Result<()> {
    let io = &invocation.io;
    if invocation.verbose {
        eprintln!("{}", crate::t!("rubyc.main.build.stage1"));
    }
    let input = read_input(io, pipeline)?;
    let unit = into_unit(input, invocation)?;
    check_lints(&unit, invocation)?;
    Ok(())
}

// -------------------------------------------------------- shared pipelines

/// Read bytes from -i/positional/stdin, then classify.
///
/// With `--base64`, input is decoded from base64 — except in Auto mode,
/// where undecodable input falls back to being treated as plain source
/// (so `cat x.rc | rubyc run` keeps working even with `--base64` set).
fn read_input(io: &IoArgs, pipeline: &NativePipeline) -> Result<Input> {
    let raw = read_raw(io)?;
    match io.in_format.unwrap_or(InFormat::Auto) {
        InFormat::Source => {
            let raw = if io.base64 { base64_decode(&raw)? } else { raw };
            let text = String::from_utf8(raw)
                .map_err(|_| DriverError::from_msg("input is not valid UTF-8"))?;
            Ok(Input::Source(text))
        }
        InFormat::Bytecode => {
            let decoded = base64_decode(&raw)?;
            Ok(Input::Unit(load_bytecode_strict(&decoded)?))
        }
        InFormat::Auto => {
            let decoded: Vec<u8> = if io.base64 {
                match base64_decode(&raw) {
                    Ok(d) => d,
                    Err(_) => raw.clone(),
                }
            } else {
                raw
            };
            classify(&decoded, pipeline)
        }
    }
}

pub(crate) fn base64_decode(raw: &[u8]) -> std::result::Result<Vec<u8>, DriverError> {
    let text = String::from_utf8(raw.to_vec())
        .map_err(|_| DriverError::from_msg("base64 input is not text"))?;
    base64::decode(&text).map_err(|e| DriverError::from_msg(format!("bad base64: {e}")))
}

/// Platform-policy classification (shebang handling lives in the loader).
fn classify(bytes: &[u8], pipeline: &NativePipeline) -> Result<Input> {
    match pipeline.loader.sniff(bytes) {
        InputKind::Source => {
            let text = String::from_utf8(bytes.to_vec()).map_err(|_| {
                DriverError::from_msg("input is not valid UTF-8 and is not RubyC bytecode")
            })?;
            Ok(Input::Source(text))
        }
        InputKind::Bytecode { payload_start } => {
            crate::bytecode::decode_payload(&bytes[payload_start..])
                .map(Input::Unit)
                .map_err(|e| DriverError::from_msg(format!("bytecode rejected: {e}")))
        }
    }
}

/// Strict bytecode load: requires the standard preamble at offset 0.
fn load_bytecode_strict(bytes: &[u8]) -> std::result::Result<CompilationUnit, DriverError> {
    CompilationUnit::from_bytecode(bytes)
        .map_err(|e| DriverError::from_msg(format!("bytecode rejected: {e}")))
}

pub(crate) fn input_name(io: &IoArgs) -> String {
    io.input
        .clone()
        .or_else(|| io.file.clone())
        .unwrap_or_else(|| "<stdin>".to_owned())
}

fn into_unit(input: Input, inv: &Invocation) -> Result<CompilationUnit> {
    let name = input_name(&inv.io);
    let (source, assembly_blocks) = match &input {
        Input::Unit(unit) => return Ok(unit.clone()),
        Input::Source(text) => {
            // Preprocessor pass: directives + conditional blocks.
            let pre = match crate::preprocessor::preprocess(text, &inv.defines, &name) {
                Ok(o) => o,
                Err(mut ds) => {
                    for d in ds.iter_mut() {
                        d.file = Some(name.clone());
                    }
                    return Err(DriverError::Diagnostics {
                        diags: ds,
                        source: text.clone(),
                    });
                }
            };
            if !pre.diagnostics.is_empty() {
                eprint!("{}", render_diags(&pre.diagnostics, Some(text), inv));
            }
            let source = crate::preprocessor::expand_macros(pre.source, &pre.macros);
            (source, pre.assembly_blocks)
        }
    };

    match parser::parse(&source) {
        Ok(mut unit) => {
            unit.assembly_blocks = assembly_blocks;
            // [Inline] expansion first so downstream checks see final code.
            crate::inliner::run(&mut unit);
            // Unknown-name / unknown-function gate runs on clean parses
            // too, before codegen ever sees them.
            let found = crate::diagnostics_semantic::scan(&unit);
            if found.is_empty() {
                Ok(unit)
            } else {
                Err(DriverError::Diagnostics {
                    diags: found.into_iter().map(|d| d.in_file(name.clone())).collect(),
                    source,
                })
            }
        }
        Err(errors) => {
            let mut stamped: Vec<Diagnostic> = errors
                .into_iter()
                .map(|e| e.in_file(name.clone()))
                .collect();

            // Best-effort semantic scan on the recovered AST.
            let (lenient_unit, _) = parser::parse_lenient(&source);
            if !stamped.is_empty() && unit_has_class(&lenient_unit) {
                for d in crate::diagnostics_semantic::scan(&lenient_unit) {
                    stamped.push(d.in_file(name.clone()));
                }
            }

            Err(DriverError::Diagnostics {
                diags: stamped,
                source,
            })
        }
    }
}

/// Recovery sanity gate: only run downstream scans when at least one class
/// body survived parsing.
pub(crate) fn unit_has_class(unit: &CompilationUnit) -> bool {
    unit.namespaces.iter().any(|ns| !ns.members.is_empty()) || !unit.members.is_empty()
}

/// Render diagnostics in the configured format (used for parse errors).
pub(crate) fn render_diags(diags: &[Diagnostic], source: Option<&str>, inv: &Invocation) -> String {
    let color =
        !inv.no_color && std::env::var_os("NO_COLOR").is_none() && std::io::stderr().is_terminal();
    match inv.error_format {
        crate::cli::ErrorFormat::Text => render_text(diags, source, "<input>", color),
        crate::cli::ErrorFormat::Json => render_json(diags, source, false),
        crate::cli::ErrorFormat::JsonPretty => render_json(diags, source, true),
    }
}

pub(crate) fn read_raw(io: &IoArgs) -> std::result::Result<Vec<u8>, DriverError> {
    let path = io.input.clone().or_else(|| io.file.clone());
    match path.as_deref() {
        None | Some("-") => {
            let mut buf = Vec::new();
            std::io::stdin()
                .read_to_end(&mut buf)
                .map_err(|e| DriverError::from_msg(format!("cannot read stdin: {e}")))?;
            Ok(buf)
        }
        Some(path) => std::fs::read(path)
            .map_err(|e| DriverError::from_msg(format!("cannot read {path}: {e}"))),
    }
}

pub(crate) fn write_out(io: &IoArgs, bytes: &[u8]) -> Result<()> {
    match io.output.as_deref() {
        None | Some("-") => {
            std::io::stdout()
                .write_all(bytes)
                .map_err(|e| DriverError::from_msg(format!("cannot write stdout: {e}")))?;
            Ok(())
        }
        Some(path) => std::fs::write(path, bytes)
            .map_err(|e| DriverError::from_msg(format!("cannot write {path}: {e}"))),
    }
}

fn emit_exe(
    invocation: &Invocation,
    input: &Input,
    io: &IoArgs,
    pipeline: &NativePipeline,
) -> Result<()> {
    let unit = into_unit(input.clone(), invocation)?;
    if invocation.verbose {
        eprintln!("{}", crate::t!("rubyc.main.build.stage2"));
    }
    let image = crate::native::compile_to_bytes_with_libs_preamble_opts(
        &unit,
        pipeline.backend,
        pipeline.image_writer,
        &io.libs.iter().map(PathBuf::from).collect::<Vec<_>>(),
        !io.no_magic,
        &invocation.defines,
        crate::native::EmitOptions {
            optimize: io.optimize,
            strip: io.strip,
        },
    )
    .map_err(|e| DriverError::from_msg(e.to_string()))?;

    match io.output.as_deref() {
        None => {
            // No explicit output: write <stem> in the current directory.
            let stem = io
                .input
                .clone()
                .or_else(|| io.file.clone())
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("a.out"));
            let out = stem.with_extension("");
            std::fs::write(&out, &image).map_err(|e| {
                DriverError::from_msg(format!("cannot write {}: {e}", out.display()))
            })?;
            mark_executable(&out.to_string_lossy())?;
            println!(
                "{}",
                crate::tf!("rubyc.main.cmd_compile.built", out.display().to_string())
            );
            Ok(())
        }
        Some("-") => {
            let bytes = if io.base64 {
                base64_encode(&image).into_bytes()
            } else {
                image
            };
            std::io::stdout()
                .write_all(&bytes)
                .map_err(|e| DriverError::from_msg(format!("cannot write stdout: {e}")))?;
            Ok(())
        }
        Some(path) => {
            std::fs::write(path, &image)
                .map_err(|e| DriverError::from_msg(format!("cannot write {path}: {e}")))?;
            mark_executable(path)
        }
    }
}

/// Built executables should be runnable out of the box.
pub(crate) fn mark_executable(path: &str) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(path)
            .map_err(|e| DriverError::from_msg(format!("cannot stat {path}: {e}")))?
            .permissions();
        perms.set_mode(perms.mode() | 0o111);
        std::fs::set_permissions(path, perms)
            .map_err(|e| DriverError::from_msg(format!("cannot chmod {path}: {e}")))?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

/// Write a shebang script and mark it executable so `./script` just works.
fn write_script(io: &IoArgs, bytes: &[u8]) -> Result<()> {
    write_out(io, bytes)?;
    if let Some(path) = io.output.as_deref().filter(|p| *p != "-") {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
            .map_err(|e| DriverError::from_msg(format!("cannot chmod {path}: {e}")))?;
    }
    Ok(())
}

pub(crate) fn wrap_shebang(bytecode: &[u8], interpreter: &str) -> Vec<u8> {
    let mut v = Vec::with_capacity(interpreter.len() + bytecode.len() + 2);
    v.extend_from_slice(b"#!");
    v.extend_from_slice(interpreter.as_bytes());
    v.push(b'\n');
    v.extend_from_slice(bytecode);
    v
}

fn base64_encode(bytes: &[u8]) -> String {
    base64::encode(bytes)
}

pub(crate) fn default_interpreter() -> String {
    "/usr/bin/env rubyc".to_string()
}

fn report(io: &IoArgs, verbose: bool, verb_key: &str) {
    if verbose {
        let verb = crate::t!(verb_key);
        match &io.output {
            Some(out) => eprintln!("[{verb}] -> {out}"),
            None => eprintln!("[{verb}] -> stdout"),
        }
    }
}

pub(crate) fn roundtrip_ok(source: &str) -> bool {
    match parser::parse(source) {
        Ok(unit) => CompilationUnit::from_bytecode(&unit.to_bytecode())
            .map(|decoded| decoded.structural_eq(&unit))
            .unwrap_or(false),
        Err(_) => false,
    }
}

impl DriverError {
    pub(crate) fn from_msg(m: impl Into<String>) -> Self {
        DriverError::Message(m.into())
    }
}


