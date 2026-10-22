//! Compilation targets and the backend contracts every target implements.
//!
//! [`Target`] is the target *space*. The concrete machine-code backends live
//! in per-target crates (e.g. `rubyc-target-linux-x86_64`) and are injected
//! at run time through [`NativePipeline`] — core never references a concrete
//! target crate (D1); it only consumes the four port traits defined here.
//!
//! Compile-time host detection uses custom cfg flags emitted by build.rs —
//! one per canonical name (`target_os_<name>`) — which are the Rust
//! equivalent of C#'s `#if LINUX_X86_64`.

use crate::native::ir::Program;

/// Every target OS RubyC intends to support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    LinuxX86_64,
    LinuxArm64,
    MacosArm64,
    MacosX64,
    WindowsX64,
    WindowsArm64,
    AndroidArm64,
    IosArm64,
}

impl Target {
    /// Canonical names, byte-for-byte equal to the per-target crate suffixes
    /// (`rubyc-target-<name>`) and the discovery registry keys.
    pub const ALL_TARGET_OS: &[&str] = &[
        "linux_x86_64",
        "linux_arm64",
        "macos_arm64",
        "macos_x64",
        "windows_x64",
        "windows_arm64",
        "android_arm64",
        "ios_arm64",
    ];

    /// All targets, in canonical order.
    pub const ALL: &[Target] = &[
        Target::LinuxX86_64,
        Target::LinuxArm64,
        Target::MacosArm64,
        Target::MacosX64,
        Target::WindowsX64,
        Target::WindowsArm64,
        Target::AndroidArm64,
        Target::IosArm64,
    ];

    /// The canonical target-os name for this target (its directory name).
    pub fn target_os_name(self) -> &'static str {
        match self {
            Target::LinuxX86_64 => "linux_x86_64",
            Target::LinuxArm64 => "linux_arm64",
            Target::MacosArm64 => "macos_arm64",
            Target::MacosX64 => "macos_x64",
            Target::WindowsX64 => "windows_x64",
            Target::WindowsArm64 => "windows_arm64",
            Target::AndroidArm64 => "android_arm64",
            Target::IosArm64 => "ios_arm64",
        }
    }

    /// Reverse mapping of [`Target::target_os_name`].
    pub fn from_target_os_name(name: &str) -> Option<Target> {
        Self::ALL_TARGET_OS
            .iter()
            .position(|n| *n == name)
            .map(|i| Self::ALL[i])
    }

    /// The target OS this build runs on. Selection keys off custom cfg
    /// flags emitted by build.rs; linux_x86_64 is the default fallback.
    pub fn current() -> Option<Target> {
        if cfg!(target_os_linux_x86_64) {
            return Some(Target::LinuxX86_64);
        }
        None
    }

}

/// A target plus the four backend ports a host injects for it.
///
/// This is the unit a composition root (the `rubyc-cli` binary, the C
/// embedding host) hands to `driver::cli_main` and the native entry points.
/// Core never constructs one; it only consumes the injected `&dyn` ports,
/// which keeps core free of any target-crate dependency (D1).
#[derive(Clone, Copy)]
pub struct NativePipeline<'a> {
    pub target: Target,
    pub backend: &'a dyn CodeGenBackend,
    pub image_writer: &'a dyn ImageWriter,
    pub loader: &'a dyn BytecodeLoader,
    pub memory: &'a dyn MemoryManager,
    /// When set, compilation goes through the plugin ABI (byte-through-byte)
    /// instead of the in-process backend/writer ports.
    pub plugin_target: Option<&'a crate::plugin::TargetPlugin>,
}

impl<'a> NativePipeline<'a> {
    pub fn new(
        target: Target,
        backend: &'a dyn CodeGenBackend,
        image_writer: &'a dyn ImageWriter,
        loader: &'a dyn BytecodeLoader,
        memory: &'a dyn MemoryManager,
    ) -> Self {
        Self { target, backend, image_writer, loader, memory, plugin_target: None }
    }

    /// Set the target plugin to route machine-code generation and image
    /// writing through the FFI boundary. Languages are always in-process.
    pub fn with_target(self, target: &'a crate::plugin::TargetPlugin) -> Self {
        Self { plugin_target: Some(target), ..self }
    }
}

/// Why a compilation against a specific target failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetError {
    /// The target exists but has no implementation yet.
    Unsupported(Target),
    /// Shared-library emission is not implemented for this target.
    NoSharedLibrary,
    /// Static-archive members are not implemented for this target.
    NoStaticArchive,
    /// An archive member failed to decode.
    BadObject(String),
    /// Type embedding is not implemented for this target.
    NoTypeEmbed,
}

impl std::fmt::Display for TargetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TargetError::Unsupported(t) => {
                write!(
                    f,
                    "{}",
                    crate::tf!(
                        "rubyc.native.target.TargetError.unsupported",
                        t.target_os_name()
                    )
                )
            }
            TargetError::NoSharedLibrary => {
                write!(
                    f,
                    "{}",
                    crate::t!("rubyc.native.target.TargetError.no_shared_library")
                )
            }
            TargetError::NoStaticArchive => {
                write!(
                    f,
                    "{}",
                    crate::t!("rubyc.native.target.TargetError.no_static_archive")
                )
            }
            TargetError::BadObject(e) => {
                write!(
                    f,
                    "{}",
                    crate::tf!(
                        "rubyc.native.target.TargetError.bad_object",
                        e
                    )
                )
            }
            TargetError::NoTypeEmbed => {
                write!(
                    f,
                    "{}",
                    crate::t!("rubyc.native.target.TargetError.no_type_embed")
                )
            }
        }
    }
}

impl std::error::Error for TargetError {}

/// Lowers IR into machine code plus everything needed to place it.
///
/// Stateless ports must stay callable from any thread: solution builds
/// fan projects out to a thread pool sharing one pipeline.
pub trait CodeGenBackend: Sync {
    fn lower(&self, program: &Program) -> Result<MachineCode, TargetError>;

    /// Serialize one library's `MachineCode` into an archive member
    /// (static `.a` builds). The default says no; targets with a
    /// relocatable object form override it.
    fn encode_object(&self, _mc: &MachineCode) -> Result<Vec<u8>, TargetError> {
        Err(TargetError::NoStaticArchive)
    }

    /// The inverse: decode an archive member back for static linking.
    fn decode_object(&self, _bytes: &[u8]) -> Result<MachineCode, TargetError> {
        Err(TargetError::NoStaticArchive)
    }
}

/// Produces a loadable executable image for one target.
pub trait ImageWriter: Sync {
    fn write_image(&self, mc: &MachineCode, entry_preamble: bool) -> Result<Vec<u8>, TargetError>;

    /// Emit a shared object (`-O shared`): an ET_DYN image exporting public
    /// methods so C hosts can `dlopen`/`dlsym` them. The 🐕🇨 container
    /// header sits at offset 64, right after the ELF header.
    fn write_shared_library(&self, _mc: &MachineCode) -> Result<Vec<u8>, TargetError> {
        Err(TargetError::NoSharedLibrary)
    }

    /// Emit a shared object with the library's type fragment embedded
    /// (`--embed-types`): same image plus a non-loaded `.rubyc_types`
    /// section carrying the `.rci` bytes, so consumers read types from
    /// the binary instead of a sidecar file. Defaults to a loud error;
    /// targets without an ELF section table override it. (The loaded
    /// `.so`-plugin ABI has no slot for the fragment — embedding works
    /// through the in-process backend only.)
    fn write_shared_library_embedded(
        &self,
        _mc: &MachineCode,
        _types: &[u8],
    ) -> Result<Vec<u8>, TargetError> {
        Err(TargetError::NoTypeEmbed)
    }
}

/// Classifies piped-in / loaded bytes as bytecode or source.
pub trait BytecodeLoader: Sync {
    fn sniff(&self, bytes: &[u8]) -> InputKind;
}

/// Result of inspecting raw input bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    /// RubyC bytecode; `payload_start` is the offset of the kind byte.
    Bytecode { payload_start: usize },
    /// Anything else: treat the bytes as RubyC source text.
    Source,
}

/// OS memory management for executable pages (JIT).
pub trait MemoryManager: Sync {
    fn map_rw(&self, len: usize) -> Result<u64, std::io::Error>;
    fn protect_exec(&self, addr: u64, len: usize) -> Result<(), std::io::Error>;
    fn unmap(&self, addr: u64, len: usize);
}

/// Symbol a relocation resolves against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sym {
    /// Function body i, resolved against the final code base address.
    Fn(usize),
    /// Byte `offset` inside the read-only data blob.
    String(usize),
    /// Address of a class's vtable row: `data_base + byte_offset`.
    VtableOff(usize),
    /// Address of static field slot `idx`: `data_base + static_offsets[idx]`.
    Static(usize),
    /// Start of the object heap.
    HeapBase,
    /// Runtime helpers appended to the code blob.
    HelperItoa,
    /// Atomic increment (refcount).
    HelperRetain,
    /// Atomic decrement + freelist push on zero.
    HelperRelease,
    /// stderr twin of ITOA.
    HelperItoaErr,
    /// scan_int() helper.
    HelperScanInt,
    HelperAlloc,
    /// Exception runtime: register jmp_buf, save regs, return 0/1.
    HelperExcEnter,
    /// Exception runtime: decrement the handler depth.
    HelperExcPop,
    /// Exception runtime: set inflight, longjmp (or exit).
    HelperExcThrow,
    /// Exception runtime: longjmp to the inflight thrower (or exit).
    HelperExcRethrow,
    /// Exception runtime: return the inflight exception pointer.
    HelperExcLoad,
    /// Exception runtime: clear the inflight exception pointer.
    HelperExcClear,
    /// External import i (into `MachineCode::import_names`), resolved
    /// against `import_bases` after link-time extraction.
    Import(usize),
    /// Byte `offset` inside the read-only data blob marking the start of the
    /// exception runtime state block (inflight / depth / jmp_bufs).
    ExcState(usize),
    /// Address of the 8-byte startup argv slot. In a process image
    /// `@start` writes the kernel rsp there before its prologue; a host
    /// without a kernel stack (the JIT) pre-seeds it with its own argv
    /// block. Its data offset is recorded in `MachineCode::helper_argv`;
    /// `BuildArgv` reads argc through it (empty array where zero).
    ArgvRsp,
    /// Name i in `Program::names`, resolved to a runtime lookup table entry
    /// for `CallByName` dynamic dispatch.
    Name(usize),
}

/// Which blob a relocation's `site` indexes into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Section {
    Code,
    Data,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocKind {
    /// Patch an 8-byte absolute address (movabs immediate).
    Abs64,
    /// Patch a 32-bit relative displacement (E8/E9 call/jmp).
    Rel32Call,
}

/// One pending address fixup recorded during emission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reloc {
    pub kind: RelocKind,
    pub section: Section,
    /// Byte offset of the field to patch within its section's buffer.
    pub site: usize,
    pub sym: Sym,
}

/// Import/export symbol for a class's vtable row: the row lives in the
/// owning image; consumers point slot 0 at it via `LoadImport`. Emitted
/// as an OBJECT symbol in the shared library's `.dynsym`.
pub fn vtable_symbol(class_path: &str) -> String {
    format!("{class_path}::__vtable")
}

/// Import/export symbol for a static field slot: the storage lives in
/// the owning image; consumers address it via `LoadImport`. Emitted as
/// an OBJECT symbol in the shared library's `.dynsym`. No arity suffix
/// (unlike method symbols), so field and method namespaces never collide.
pub fn static_symbol(class_path: &str, field: &str) -> String {
    format!("{class_path}::{field}")
}

/// Fully emitted machine code, pending relocation.
///
/// Layout contract: `data` begins with literal strings, followed by one
/// 8-byte vtable word per slot per class (see `vtable_offsets`).
#[derive(Debug, Clone)]
pub struct MachineCode {
    /// Executable bytes: function bodies then runtime helpers.
    pub code: Vec<u8>,
    /// Read-only data: literal strings then vtable address words.
    pub data: Vec<u8>,
    /// Byte offset of each function body within `code`.
    pub fn_bases: Vec<usize>,
    /// Byte offset of each class's vtable row within `data`.
    pub vtable_offsets: Vec<usize>,
    /// Class path per vtable row (parallel to `vtable_offsets`): the
    /// shared-library writer exports each row as a `<path>::__vtable`
    /// OBJECT symbol so downstream images can point slot 0 at the
    /// owner's row instead of a hollow local copy.
    pub vtable_names: Vec<String>,
    /// Byte offset of each static field slot within `data` (8 bytes each,
    /// zero-initialized). Index matches `Program::statics`.
    pub static_offsets: Vec<usize>,
    /// Full import/export symbol per static slot (parallel to
    /// `static_offsets`, `<path>::<field>`): the shared-library writer
    /// exports each slot as an OBJECT symbol so downstream images can
    /// address the owner's storage via `LoadImport` instead of a hollow
    /// local copy.
    pub static_names: Vec<String>,
    /// Offset within `code` of the itoa helper (None = not referenced).
    pub helper_itoa: Option<usize>,
    /// stderr itoa twin (None = not referenced).
    pub helper_itoa_err: Option<usize>,
    /// scan_int() helper (None = not referenced).
    pub helper_scan_int: Option<usize>,
    pub helper_retain: Option<usize>,
    pub helper_release: Option<usize>,
    /// Offset within `code` of the alloc helper (None = not referenced).
    pub helper_alloc: Option<usize>,
    /// Offsets within `code` of the six exception-runtime helpers
    /// (enter, pop, throw, rethrow, load_exc, clear_inflight); None when
    /// exceptions are unused.
    pub helper_exc_enter: Option<usize>,
    pub helper_exc_pop: Option<usize>,
    pub helper_exc_throw: Option<usize>,
    pub helper_exc_rethrow: Option<usize>,
    pub helper_exc_load: Option<usize>,
    pub helper_exc_clear: Option<usize>,
    /// Byte offset within `data` of the exception state block (None when
    /// exceptions are unused).
    pub exc_state_offset: Option<usize>,
    /// Byte offset within `data` of the startup argv slot (None when no
    /// `BuildArgv` is lowered, i.e. paramless entry points). The `@start`
    /// prologue writes the kernel rsp there and `BuildArgv` reads it
    /// back. Not a code offset: the builder is inlined at its use site.
    pub helper_argv: Option<usize>,
    /// `(library hint, entry point)` per declared import, in call-index
    /// order.
    /// `(library hint, entry point)` per declared import, in call-index
    /// order.
    pub import_names: Vec<(String, String)>,
    /// Code offsets of each extracted import body; filled by link-time
    /// symbol extraction (`native::linker`). Empty means unresolved.
    pub import_bases: Vec<usize>,
    /// Absolute runtime addresses per import, parallel to
    /// `import_names`; filled by load-time dynamic linking
    /// (`native::linker::resolve_dep_imports_into`). `u64::MAX` means
    /// unresolved. Preferred over `import_bases` during relocation
    /// when set: the callee lives in another loaded image.
    pub import_abs: Vec<u64>,
    /// `(exported symbol name, function index, arity)` for shared libraries.
    pub exports: Vec<(String, usize, usize)>,
    /// Whether each export takes a receiver, parallel to `exports`:
    /// the shared-library writer emits a register-shifting trampoline
    /// for receiver-taking entries and points receiverless (static)
    /// entries straight at their bodies (SysV and body conventions
    /// already agree there).
    pub export_takes_receiver: Vec<bool>,
    /// Function index of `@init` (shared libraries only): the writer
    /// registers it in `.init_array` so the loader runs static field
    /// initializers at load time. `None` when the program has no
    /// `@init` (executables, or libraries without statics).
    pub init_fn: Option<usize>,
    pub relocs: Vec<Reloc>,
    /// Object-heap bytes required at runtime.
    pub heap_size: u64,
    /// Drop `.symtab`/`.strtab` at image-write time (`--strip`).
    /// Data-only hint for writers; never affects relocations, the
    /// dynamic symbol table (required for linking), or behavior.
    /// Serialized as a trailing codec flag (absent = false).
    pub strip_symbols: bool,
}

impl MachineCode {
    /// Contiguous page-aligned region sizes for (code, data, heap).
    pub fn regions(&self, page: u64) -> (u64, u64, u64) {
        let code = (self.code.len() as u64).div_ceil(page) * page;
        let data = (self.data.len() as u64).div_ceil(page) * page;
        let heap = self.heap_size.div_ceil(page) * page;
        (code, data, heap)
    }

    fn resolve(&self, sym: &Sym, code_base: u64, data_base: u64, heap_base: u64) -> u64 {
        match sym {
            Sym::Fn(i) => match self.fn_bases.get(*i) {
                Some(b) => code_base + *b as u64,
                None => code_base,
            },
            Sym::String(off) => data_base + *off as u64,
            Sym::VtableOff(off) => data_base + *off as u64,
            Sym::Static(idx) => {
                data_base
                    + self.static_offsets.get(*idx).copied().unwrap_or(0) as u64
            }
            Sym::HeapBase => heap_base,
            Sym::HelperItoa => code_base + self.helper_itoa.unwrap_or(0) as u64,
            Sym::HelperItoaErr => code_base + self.helper_itoa_err.unwrap_or(0) as u64,
            Sym::HelperScanInt => code_base + self.helper_scan_int.unwrap_or(0) as u64,
            Sym::HelperRetain => code_base + self.helper_retain.unwrap_or(0) as u64,
            Sym::HelperRelease => code_base + self.helper_release.unwrap_or(0) as u64,
            Sym::HelperAlloc => code_base + self.helper_alloc.unwrap_or(0) as u64,
            Sym::HelperExcEnter => code_base + self.helper_exc_enter.unwrap_or(0) as u64,
            Sym::HelperExcPop => code_base + self.helper_exc_pop.unwrap_or(0) as u64,
            Sym::HelperExcThrow => code_base + self.helper_exc_throw.unwrap_or(0) as u64,
            Sym::HelperExcRethrow => code_base + self.helper_exc_rethrow.unwrap_or(0) as u64,
            Sym::HelperExcLoad => code_base + self.helper_exc_load.unwrap_or(0) as u64,
            Sym::HelperExcClear => code_base + self.helper_exc_clear.unwrap_or(0) as u64,
            Sym::ExcState(off) => data_base + *off as u64,
            Sym::ArgvRsp => {
                // Recorded when the slot was appended. Do not derive this
                // from `data.len()`: `relocate` takes the data vector
                // before resolving, so a live length read is zero and the
                // slot resolved to the data base (the JIT then read the
                // first literal bytes as a saved stack pointer).
                data_base + self.helper_argv.unwrap_or(0) as u64
            }
            Sym::Import(i) => {
                // Load-time address wins when the loader resolved it;
                // otherwise fall back to the extracted body offset.
                if let Some(&addr) = self.import_abs.get(*i)
                    && addr != u64::MAX
                {
                    return addr;
                }
                code_base + self.import_bases.get(*i).copied().unwrap_or(usize::MAX) as u64
            }
            // Name symbols are resolved at link time by the backend;
            // return a sentinel until a runtime name table is wired.
            Sym::Name(_) => usize::MAX as u64,
        }
    }

    /// Apply every relocation against final runtime addresses, patching the
    /// code and data blobs in place. Call once the blobs have been copied
    /// to their permanent locations and BEFORE making code executable.
    pub fn relocate(&mut self, code_base: u64, data_base: u64, heap_base: u64) {
        let mut code = std::mem::take(&mut self.code);
        let mut data = std::mem::take(&mut self.data);
        for r in &self.relocs {
            let target = self.resolve(&r.sym, code_base, data_base, heap_base);
            match (r.section, r.kind) {
                (Section::Code, RelocKind::Abs64) => {
                    code[r.site..r.site + 8].copy_from_slice(&target.to_le_bytes());
                }
                (Section::Data, RelocKind::Abs64) => {
                    // Vtable slot words live in the data blob; patch them
                    // with absolute function addresses.
                    data[r.site..r.site + 8].copy_from_slice(&target.to_le_bytes());
                }
                (Section::Code, RelocKind::Rel32Call) => {
                    // rel32 is relative to the end of the instruction
                    let after_addr = code_base + r.site as u64 + 4;
                    let rel = target as i64 - after_addr as i64;
                    code[r.site..r.site + 4].copy_from_slice(&(rel as i32).to_le_bytes());
                }
                // Call relocations only ever target the code section.
                (Section::Data, RelocKind::Rel32Call) => {}
            }
        }
        self.code = code;
        self.data = data;
    }
}
