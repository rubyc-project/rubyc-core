//! JIT compilation: lower IR to machine code in-process and execute it.
//!
//! Safety model (PoC scope):
//! - Memory is mapped writable, filled, then re-protected read+execute.
//!   Writable-and-executable memory never exists at the same time (W^X).
//! - Every byte of generated code comes from our own backend; no user
//!   input is ever copied into the executable region unchecked. The only
//!   syscalls the code can make are the ones we emit (`write`, `exit_group`).
//! - Code size is capped before mapping to bound resource use.
//!
//! Residual risk, stated honestly: executing hostile bytecode needs OS
//! sandboxing (seccomp) on top; that is future work.

use crate::models::CompilationUnit;
use crate::native::ir;
use crate::native::target::{CodeGenBackend, MemoryManager};

/// Upper bound on emitted machine code accepted by the JIT (DoS guard).
const MAX_CODE_BYTES: usize = 1 << 20;

/// Errors raised while JIT-compiling and running a program.
#[derive(Debug)]
pub enum JitError {
    /// AST cannot be lowered yet (missing main, unsupported construct).
    Lower(ir::LowerError),
    /// The current target has no backend implementation.
    Target(crate::native::target::TargetError),
    /// Program exceeds the JIT size cap.
    TooLarge(usize),
    /// OS memory management failed.
    Mmap(std::io::Error),
    /// Import resolution against `--lib` files failed.
    Link(String),
}

impl std::fmt::Display for JitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JitError::Lower(e) => write!(f, "{e}"),
            JitError::Target(e) => write!(f, "{e}"),
            JitError::TooLarge(n) => {
                let n = n.to_string();
                write!(
                    f,
                    "{}",
                    crate::tf!("rubyc.native.jit.JitError.too_large", n)
                )
            }
            JitError::Mmap(e) => {
                let e = e.to_string();
                write!(f, "{}", crate::tf!("rubyc.native.jit.JitError.mmap", e))
            }
            JitError::Link(e) => {
                write!(
                    f,
                    "{}",
                    crate::tf!("rubyc.native.jit.JitError.link", e.to_string())
                )
            }
        }
    }
}

impl std::error::Error for JitError {}

/// Compile `unit` to native code in-memory and execute it.
///
/// Memory layout (one contiguous RW mapping, then code range made RX):
///
/// ```text
/// [code + helpers][strings + vtables][object heap]
/// ```
pub fn jit_run<B: CodeGenBackend + ?Sized, M: MemoryManager + ?Sized>(
    unit: &CompilationUnit,
    backend: &B,
    memory: &M,
) -> Result<i32, JitError> {
    jit_run_with_libs(unit, backend, memory, &[])
}

/// Like [`jit_run`], resolving any `[LibraryImport]` stubs against the
/// given shared objects before execution.
pub fn jit_run_with_libs<B: CodeGenBackend + ?Sized, M: MemoryManager + ?Sized>(
    unit: &CompilationUnit,
    backend: &B,
    memory: &M,
    libs: &[std::path::PathBuf],
) -> Result<i32, JitError> {
    let program = ir::lower(unit).map_err(JitError::Lower)?;
    let mut mc = backend.lower(&program).map_err(JitError::Target)?;

    // Link imports into the code blob before relocation; Sym::Import
    // resolves through import_bases during mc.relocate below.
    if !mc.import_names.is_empty() {
        crate::native::linker::resolve_imports(&mut mc, libs, true).map_err(JitError::Link)?;
    }
    jit_execute(mc, memory, Vec::new())
}

/// Like [`jit_run_with_libs`], additionally resolving dependency-image
/// (`.so`-hinted) imports at load time: each dep `.so` under `dyn_dir`
/// is `dlopen`ed and its symbols recorded as absolute import addresses
/// before relocation. The handles stay alive across execution.
/// Entries without a `.so` hint, or already statically extracted, are
/// untouched. This is the `run` path for `dynamic = true` builds.
pub fn jit_run_with_dyn_libs<B: CodeGenBackend + ?Sized, M: MemoryManager + ?Sized>(
    unit: &CompilationUnit,
    backend: &B,
    memory: &M,
    libs: &[std::path::PathBuf],
    dyn_dir: &std::path::Path,
) -> Result<i32, JitError> {
    let program = ir::lower(unit).map_err(JitError::Lower)?;
    let mut mc = backend.lower(&program).map_err(JitError::Target)?;

    if !mc.import_names.is_empty() {
        crate::native::linker::resolve_imports(&mut mc, libs, true).map_err(JitError::Link)?;
        let _handles = crate::native::linker::resolve_dep_imports_into(&mut mc, dyn_dir)
            .map_err(JitError::Link)?;
        return jit_execute(mc, memory, _handles);
    }
    jit_execute(mc, memory, Vec::new())
}

/// Like [`jit_run_with_dyn_libs`], lowering with dependency owners and
/// preloaded fragments (dynamic-consume builds): dep members resolve
/// cross-image and lower to imports, resolved at load time below.
/// This is the `run` path for `dynamic = true` projects.
pub fn jit_run_with_owners<B: CodeGenBackend + ?Sized, M: MemoryManager + ?Sized>(
    unit: &CompilationUnit,
    backend: &B,
    memory: &M,
    libs: &[std::path::PathBuf],
    dyn_dir: &std::path::Path,
    owners: &std::collections::HashMap<
        crate::native::semantic::TypeKey,
        String,
    >,
    preloaded: &crate::native::semantic::Preloaded,
) -> Result<i32, JitError> {
    let program =
        ir::lower_with_owners(unit, true, &[], owners, preloaded).map_err(JitError::Lower)?;
    let mut mc = backend.lower(&program).map_err(JitError::Target)?;

    if !mc.import_names.is_empty() {
        crate::native::linker::resolve_imports(&mut mc, libs, true).map_err(JitError::Link)?;
        let _handles = crate::native::linker::resolve_dep_imports_into(&mut mc, dyn_dir)
            .map_err(JitError::Link)?;
        return jit_execute(mc, memory, _handles);
    }
    jit_execute(mc, memory, Vec::new())
}
fn jit_execute<M: MemoryManager + ?Sized>(
    mut mc: crate::native::target::MachineCode,
    memory: &M,
    _handles: Vec<crate::libloader::Handle>,
) -> Result<i32, JitError> {

    if mc.code.len() > MAX_CODE_BYTES {
        return Err(JitError::TooLarge(mc.code.len()));
    }

    // Page-aligned regions in one contiguous RW mapping.
    const PAGE: usize = 4096;
    let code_r = mc.code.len().div_ceil(PAGE) * PAGE;
    let data_r = mc.data.len().div_ceil(PAGE) * PAGE;
    let heap_r = (mc.heap_size as usize).div_ceil(PAGE) * PAGE;
    let total = code_r + data_r + heap_r;

    let page = memory.map_rw(total).map_err(JitError::Mmap)?;

    let code_base = page;
    let data_base = page + code_r as u64;
    let heap_base = page + (code_r + data_r) as u64;

    // Relocate against final addresses while still in owned memory...
    mc.relocate(code_base, data_base, heap_base);

    if let Some(path) = std::env::var_os("RUBYC_DUMP_CODE") {
        if let Ok(mut f) = std::fs::File::create(path) {
            use std::io::Write;
            let _ = f.write_all(&mc.code);
        }
    }
    if let Some(path) = std::env::var_os("RUBYC_DUMP_DATA") {
        if let Ok(mut f) = std::fs::File::create(path) {
            use std::io::Write;
            let _ = f.write_all(&mc.data);
        }
    }

    // ...then copy into the mapping. The heap bump pointer starts at
    // heap_base + 8 (first 8 bytes of the heap hold it).
    #[allow(unsafe_code)]
    unsafe {
        std::ptr::copy_nonoverlapping(mc.code.as_ptr(), page as *mut u8, mc.code.len());
        std::ptr::copy_nonoverlapping(mc.data.as_ptr(), data_base as *mut u8, mc.data.len());
        if heap_r > 0 {
            // New heap header: [+0] freelist head (NULL = empty),
            // [+8] bump pointer starting past a 4 KiB reserve.
            (heap_base as *mut u64).write_unaligned(0);
            ((heap_base + 8) as *mut u64).write_unaligned(heap_base + 4096);
            // The JIT is not a process entry, so there is no kernel argv
            // stack above rsp. Pre-seed the startup slot with a block whose
            // first word (argc) is 0 — the heap header qualifies — so the
            // `@start` stub keeps it instead of saving the host rsp.
            if let Some(off) = mc.helper_argv {
                ((data_base + off as u64) as *mut u64).write_unaligned(heap_base);
            }
        }
    }
    memory.protect_exec(page, code_r).map_err(JitError::Mmap)?;

    let status = invoke(page);
    memory.unmap(page, total);
    Ok(status)
}

#[allow(non_snake_case)]
struct _Unused;

/// Call the JIT entry point. The calling convention for a zero-argument
/// `() -> i32` function is identical across every target we support, so
/// this stays in the shared module; only syscalls are OS-specific.
fn invoke(addr: u64) -> i32 {
    // SAFETY: addr is RX memory holding backend-emitted code for this ABI.
    #[allow(unsafe_code)]
    unsafe {
        let f: extern "C" fn() -> i32 = std::mem::transmute(addr);
        f()
    }
}
