//! Native executable compilation and JIT execution.
//!
//! Pipeline: AST → IR (`ir`) → machine code (injected per-target backend) →
//! executable image or in-memory JIT invocation.

pub mod arlib;
pub mod container;
pub mod embedded;
pub mod index;
pub mod ir;
pub mod jit;
pub mod linker;
pub mod rci;
pub mod semantic;
pub mod target;

use crate::models::CompilationUnit;
use crate::native::target::{CodeGenBackend, ImageWriter};
use std::path::Path;

/// Encode a type fragment for embedding (`--embed-types`): the same
/// bytes a `.rci` sidecar would carry. Empty owners means a standalone
/// unit, so every interface/delegate counts as owned by `lib_name`;
/// otherwise only entries attributed to `lib_name` are included.
/// `source_hash` is 0 for direct compiles (no project to hash —
/// link-time users accept those fragments as-is, like `--lib` bodies).
#[allow(clippy::type_complexity)]
pub fn analyze_fragment_bytes(
    unit: &CompilationUnit,
    owners: &std::collections::HashMap<crate::native::semantic::TypeKey, String>,
    lib_name: &str,
    source_hash: u64,
) -> Result<
    Vec<u8>,
    Vec<crate::native::semantic::SemanticError>,
> {
    let table = crate::native::semantic::analyze_with_owners(unit, owners)?;
    let owned = |name: &str, arity: usize| {
        owners.is_empty()
            || owners.iter().any(|(k, v)| {
                v == lib_name
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
    Ok(crate::native::rci::encode(
        lib_name,
        source_hash,
        &table.classes,
        &ifaces,
        &delegates,
    ))
}

/// Everything that can go wrong while producing a native executable.
#[derive(Debug)]
pub enum Error {
    /// AST cannot be lowered to the native IR yet.
    Lower(ir::LowerError),
    /// The requested target has no implementation.
    Target(target::TargetError),
    /// Writing the output file failed.
    Io(std::io::Error),
    /// Linking `[LibraryImport]` stubs against provided libraries failed.
    Link(String),
    /// A plugin call (parse/lower/write_image) returned an error.
    Plugin(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Lower(e) => write!(f, "{e}"),
            Error::Target(e) => write!(f, "{e}"),
            Error::Io(e) => write!(f, "{e}"),
            Error::Link(e) => write!(f, "{e}"),
            Error::Plugin(e) => write!(f, "plugin: {e}"),
        }
    }
}

impl std::error::Error for Error {}

/// Compile a unit into an executable image file via the injected backend and
/// image writer.
///
/// For Linux x86-64 this is a fully static ELF: entry point begins with the
/// jump-over-magic block, followed by raw syscall code (no libc).
pub fn compile_to_native<B: CodeGenBackend + ?Sized, W: ImageWriter + ?Sized>(
    unit: &CompilationUnit,
    backend: &B,
    writer: &W,
    output: &Path,
) -> Result<(), Error> {
    let bytes = compile_to_bytes(unit, backend, writer)?;
    std::fs::write(output, bytes).map_err(Error::Io)?;
    Ok(())
}

/// Compile a unit into the raw executable image bytes via the injected ports.
pub fn compile_to_bytes<B: CodeGenBackend + ?Sized, W: ImageWriter + ?Sized>(
    unit: &CompilationUnit,
    backend: &B,
    writer: &W,
) -> Result<Vec<u8>, Error> {
    compile_to_bytes_with_libs(unit, backend, writer, &[], &[])
}

/// Like [`compile_to_bytes`], resolving any `[LibraryImport]` stubs against
/// the given shared-object files via link-time symbol extraction.
pub fn compile_to_bytes_with_libs<B: CodeGenBackend + ?Sized, W: ImageWriter + ?Sized>(
    unit: &CompilationUnit,
    backend: &B,
    writer: &W,
    libs: &[std::path::PathBuf],
    defines: &[String],
) -> Result<Vec<u8>, Error> {
    compile_to_bytes_with_libs_preamble(unit, backend, writer, libs, true, defines)
}

/// Like [`compile_to_bytes_with_libs`], but `entry_preamble` controls whether
/// the `🐕🇨` jump-over-magic block is placed at the entry point. `false` is the
/// `--no-magic` form: the image carries no RubyC preamble and is a plain ELF.
pub fn compile_to_bytes_with_libs_preamble<B: CodeGenBackend + ?Sized, W: ImageWriter + ?Sized>(
    unit: &CompilationUnit,
    backend: &B,
    writer: &W,
    libs: &[std::path::PathBuf],
    entry_preamble: bool,
    defines: &[String],
) -> Result<Vec<u8>, Error> {
    compile_to_bytes_with_libs_preamble_opts(
        unit,
        backend,
        writer,
        libs,
        entry_preamble,
        defines,
        EmitOptions::default(),
    )
}

/// Image emission options: size optimizations applied after linking.
/// Both default off: output is byte-identical to the unoptimized image.
#[derive(Debug, Clone, Copy, Default)]
pub struct EmitOptions {
    /// Function-level dead-code elimination (`--optimize`): drop
    /// functions unreachable from the entry point, exports, `@init`,
    /// and address-taken references.
    pub optimize: bool,
    /// Drop `.symtab`/`.strtab` from executable images (`--strip`).
    /// Shared libraries always keep `.dynsym` (required for linking).
    pub strip: bool,
}

/// Like [`compile_to_bytes_with_libs_preamble`], plus emission options.
pub fn compile_to_bytes_with_libs_preamble_opts<
    B: CodeGenBackend + ?Sized,
    W: ImageWriter + ?Sized,
>(
    unit: &CompilationUnit,
    backend: &B,
    writer: &W,
    libs: &[std::path::PathBuf],
    entry_preamble: bool,
    defines: &[String],
    opts: EmitOptions,
) -> Result<Vec<u8>, Error> {
    let t = std::time::Instant::now();
    let program = ir::lower_with_defines(unit, true, defines).map_err(Error::Lower)?;
    eprintln!("[rubyc] timing: semantic+lower {:.1}s", t.elapsed().as_secs_f64());

    let t = std::time::Instant::now();
    let mut mc = backend.lower(&program).map_err(Error::Target)?;
    eprintln!("[rubyc] timing: backend {:.1}s", t.elapsed().as_secs_f64());

    // Static archives (`.a`) merge first; only ELF objects flow
    // through body extraction (the ELF parser rejects archives).
    // Static goes first so already-resolved slots survive: extraction
    // only touches slots still null (see `resolve_imports`).
    let (elf_libs, archives): (Vec<&std::path::PathBuf>, Vec<&std::path::PathBuf>) = libs
        .iter()
        .partition(|p| !p.extension().is_some_and(|e| e == "a"));
    let elf_libs: Vec<std::path::PathBuf> = elf_libs.into_iter().cloned().collect();
    let archives: Vec<&Path> = archives.into_iter().map(|p| p.as_path()).collect();
    // Static archives (`.a`): decode every member and merge it into the
    // image, resolving imports (including `.so`-hinted dependency
    // imports, which have no loader in a standalone executable) against
    // member exports.
    if !archives.is_empty() {
        let mut members = Vec::new();
        for path in &archives {
            let bytes = std::fs::read(path).map_err(Error::Io)?;
            let archive = arlib::read(&bytes).map_err(Error::Link)?;
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_owned();
            for m in archive {
                // Type-fragment members (`.rci`) ride along for readers;
                // only machine-code members link.
                if m.name.ends_with(".rci") {
                    continue;
                }
                let decoded = backend.decode_object(&m.bytes).map_err(Error::Target)?;
                members.push((stem.clone(), decoded));
            }
        }
        linker::link_static_archives(&mut mc, members).map_err(Error::Link)?;
    }
    // Dead import slots (null, referenced by nothing) leave before
    // resolution: unreachable natives must never demand `--lib` bodies
    // or fail the link, optimized or not. Referenced slots still
    // resolve (or fail loudly) below.
    linker::drop_unreferenced_null_slots(&mut mc);
    if opts.optimize {
        // DCE first: slots only dead code references leave with it, so
        // unreachable natives never demand `--lib` bodies below.
        let t = std::time::Instant::now();
        linker::dead_code_eliminate(&mut mc);
        eprintln!("[rubyc] timing: dce {:.1}s", t.elapsed().as_secs_f64());
    }
    if !mc.import_names.is_empty() {
        linker::resolve_imports(&mut mc, &elf_libs).map_err(Error::Link)?;
    }
    mc.strip_symbols = opts.strip;

    // Executables are self-contained static images (no dynamic
    // linker involved): dependency-image imports (`.so` hints, left
    // null by the linker for load-time resolution) cannot resolve in
    // a standalone executable — fail loudly instead of the writer's
    // unlinked-import panic. Run with `rubyc run` (the host process
    // resolves dep imports at load time) or build as a shared
    // library instead. Shared libraries keep null slots by design
    // (resolved when the final image loads).
    let unlinked_dep: Vec<String> = mc
        .import_names
        .iter()
        .enumerate()
        .filter(|(i, (lib, _))| {
            lib.ends_with(".so") && mc.import_bases.get(*i).copied() == Some(usize::MAX)
        })
        .map(|(_, (_, entry))| entry.clone())
        .collect();
    if !unlinked_dep.is_empty() {
        return Err(Error::Link(format!(
            "executable cannot consume dependency images (standalone images carry no loader): {} — run with `rubyc run` or build as a shared library / from sources",
            unlinked_dep.join(", ")
        )));
    }

    // The ELF writer chooses final vaddrs; resolve relocations against them
    // by asking the writer for the layout first.
    let t = std::time::Instant::now();
    let image = writer
        .write_image(&mc, entry_preamble)
        .map_err(Error::Target)?;
    eprintln!("[rubyc] timing: image {:.1}s", t.elapsed().as_secs_f64());

    // With the standard preamble, keep the invariant that it is embedded.
    // `--no-magic` (entry_preamble = false) skips this by design.
    if entry_preamble {
        let want = container::build(container::Kind::NativeExecutable, 0);
        assert!(
            image
                .windows(container::PREAMBLE_LEN)
                .any(|w| w == &want[..container::PREAMBLE_LEN]),
            "emitted image must embed the RubyC jump-over-magic preamble"
        );
    }
    Ok(image)
}

/// Compile a unit into a shared object (`-O shared`): an ET_DYN image whose
/// public methods are exported as plain symbols. Absolute addresses are
/// resolved by the dynamic loader via `R_X86_64_RELATIVE` records, so the
/// artifact loads at any base a C host's `dlopen` chooses.
///
/// Note: libraries that themselves declare `[LibraryImport]` stubs are not
/// supported yet — imports resolve at executable link time only.
pub fn compile_shared_library<B: CodeGenBackend + ?Sized, W: ImageWriter + ?Sized>(
    unit: &CompilationUnit,
    backend: &B,
    writer: &W,
    libs: &[std::path::PathBuf],
    defines: &[String],
    embed_types: Option<&[u8]>,
    opts: EmitOptions,
) -> Result<Vec<u8>, Error> {
    compile_shared_library_with_owners(
        unit,
        backend,
        writer,
        libs,
        defines,
        &std::collections::HashMap::new(),
        &crate::native::semantic::Preloaded::empty(),
        embed_types,
        opts,
    )
}

/// Like [`compile_shared_library`], plus per-entry library ownership
/// for dynamic-linking (see `ir::lower_with_owners`). Empty owners
/// behave exactly as before. `embed_types` carries pre-encoded `.rci`
/// bytes into a `.rubyc_types` section instead of a sidecar file.
pub fn compile_shared_library_with_owners<
    B: CodeGenBackend + ?Sized,
    W: ImageWriter + ?Sized,
>(
    unit: &CompilationUnit,
    backend: &B,
    writer: &W,
    libs: &[std::path::PathBuf],
    defines: &[String],
    owners: &std::collections::HashMap<crate::native::semantic::TypeKey, String>,
    preloaded: &crate::native::semantic::Preloaded,
    embed_types: Option<&[u8]>,
    opts: EmitOptions,
) -> Result<Vec<u8>, Error> {
    // Libraries need no `main`; no @start stub is generated.
    let t = std::time::Instant::now();
    let program =
        ir::lower_with_owners(unit, false, defines, owners, preloaded).map_err(Error::Lower)?;
    eprintln!("[rubyc] timing: semantic+lower {:.1}s", t.elapsed().as_secs_f64());

    let t = std::time::Instant::now();
    let mut mc = backend.lower(&program).map_err(Error::Target)?;
    eprintln!("[rubyc] timing: backend {:.1}s", t.elapsed().as_secs_f64());
    if !mc.import_names.is_empty() {
        if libs.is_empty() {
            // Shared libraries routinely carry undefined symbols (resolved
            // by the host at load — e.g. CoreLib QCalls satisfied by the
            // runtime). Like `ld -shared`, warn and emit with null import
            // slots instead of refusing the artifact.
            eprintln!(
                "[rubyc] warning: library declares {} [LibraryImport] stub(s) but no --lib files were given; imports will be null until linked at load",
                mc.import_names.len()
            );
        } else {
            linker::resolve_imports(&mut mc, libs).map_err(Error::Link)?;
        }
    }
    if opts.optimize {
        linker::dead_code_eliminate(&mut mc);
    }
    mc.strip_symbols = opts.strip;

    let t = std::time::Instant::now();
    let image = match embed_types {
        Some(types) => writer
            .write_shared_library_embedded(&mc, types)
            .map_err(Error::Target)?,
        None => writer.write_shared_library(&mc).map_err(Error::Target)?,
    };
    eprintln!("[rubyc] timing: image {:.1}s", t.elapsed().as_secs_f64());

    // Same preamble invariant as executables, at offset 64 instead of the
    // entry point: ELF owns offset 0 in shared objects too.
    let want = container::build(container::Kind::SharedLibrary, 0);
    assert!(
        image.len() >= 13 + 3 && image[64..64 + 2] == want[..2],
        "shared library must embed the RubyC preamble after the ELF header"
    );
    assert!(
        image[66..74] == container::MAGIC,
        "shared library must embed the dog magic"
    );
    Ok(image)
}

/// Compile a unit into a static archive (`-O static`): one `.a` member
/// carrying the library's unlinked `MachineCode` (libraries need no
/// `main`; no @start stub is generated, exactly like shared builds).
/// The same `.rci` sidecar serves both dynamic and static consumers.
/// Static state duplicates per linked image (see `linker`).
/// `embed_types` adds a second `<member>.rci` archive member with
/// pre-encoded fragment bytes instead of a sidecar file.
pub fn compile_static_archive<B: CodeGenBackend + ?Sized>(
    unit: &CompilationUnit,
    backend: &B,
    member_name: &str,
    defines: &[String],
    embed_types: Option<&[u8]>,
) -> Result<Vec<u8>, Error> {
    let t = std::time::Instant::now();
    let program =
        ir::lower_with_owners(unit, false, defines, &std::collections::HashMap::new(), &crate::native::semantic::Preloaded::empty())
            .map_err(Error::Lower)?;
    eprintln!("[rubyc] timing: semantic+lower {:.1}s", t.elapsed().as_secs_f64());

    let t = std::time::Instant::now();
    let mc = backend.lower(&program).map_err(Error::Target)?;
    eprintln!("[rubyc] timing: backend {:.1}s", t.elapsed().as_secs_f64());

    let object = backend.encode_object(&mc).map_err(Error::Target)?;
    let mut members = vec![arlib::Member {
        name: member_name.to_owned(),
        bytes: object,
    }];
    if let Some(types) = embed_types {
        members.push(arlib::Member {
            name: format!("{member_name}.rci"),
            bytes: types.to_vec(),
        });
    }
    Ok(arlib::write(&members))
}


