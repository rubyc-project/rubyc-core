# `rubyc-core`

The RubyC compiler core: every compilation stage, the plugin ABI and
cross-OS loader, the project system, diagnostics, and i18n. This is a
**library crate** (`lib`) — the `rubyc` CLI and every plugin link against
it. It ships no binary of its own.

```
rubyc-core/
├── src/          compiler stages + plugin infrastructure
├── locales/      i18n diagnostics (en.json + generated code)
├── build.rs      platform target_os_* cfgs + locale codegen
└── plugins/
    └── targets/
```

> The plugin crates under `plugins/` are **their own git repositories**
> (nested submodules of this one). `rubyc-core` only *depends on* them; it does
> not contain their source.

## Pipeline stages

| Module | Path | What it does |
|---|---|---|
| Lexer / Parser | `src/parser/` | hand-written lexer; recursive-descent parser → span-tagged AST |
| Assembler | `src/parser/assembly/` | two-pass x86-64 assembler |
| Bytecode | `src/bytecode/` | 🐕🇨 container, tag-tree encode/decode (round-trip verified) |
| AST | `src/models/` | `CompilationUnit`, declarations, statements, expressions |
| Semantic | `src/native/semantic/` | type tables, inheritance, C# access rules (E0017/E0018) |
| IR | `src/native/ir/` | virtual-register three-address lowering; `encode_program`/`decode_program` |
| Linker | `src/native/linker.rs` | resolves IR symbols into machine code |
| Type fragments | `src/native/rci.rs` | the `.rci` sidecar: a library's own semantic entries |
| Target trait | `src/native/target.rs` | what a backend must provide (lower, write image, run) |
| JIT | `src/native/jit.rs` | W^X: mmap W → copy → mprotect RX |
| Lints | `src/lints.rs` | rule engine (`--warn/--deny/--allow`), stable ids |
| Diagnostics | `src/diagnostics.rs`, `src/diagnostics_semantic.rs` | codes, spans, scope context, carets |
| i18n | `src/i18n.rs` | locale lookup; English fallback |
| Project | `src/project.rs` | `.rcp` / `.rcs` project + solution system |
| Preprocessor | `src/preprocessor.rs` | `#define/#If/#Error/#Assembly`, `#macro` |
| Inliner | `src/inliner.rs` | `[Inline]` method expansion |
| Plugins | `src/plugin/` | ABI (`PluginBytes`), loader, function pointers |
| Lib loader | `src/libloader/` | cross-OS `.so`/`.dylib`/`.dll` loading (libloading + goblin) |
| CLI | `src/cli/` | argument parsing, command specs, driver glue |
| Extensions | `src/extensions.rs` | `install_root()`, `ExtensionMetadata`, target traits |

The x86-64 backend is a separate crate, not a module here:
`plugins/targets/rubyc-target-linux-x86_64/src/x86/` holds the
encoder (`encode_x64.rs`), the register allocator (`regalloc.rs`), the
image writers (`elf.rs`, `shlib.rs`), the allocator and the runtime
shims (`runtime.rs`).

## Public API

### Prelude

The `pub mod prelude` at `src/lib.rs` re-exports the common consumer surface:

```rust
use rubyc::prelude::{parse, parse_file, CompilationUnit, ToBytecode, FromBytecode, encode, decode, NativeTarget};
```

| Item | Signature | Description |
|---|---|---|
| `parse` | `fn parse(src: &str) -> Result<CompilationUnit, Vec<Diagnostic>>` | Parse source text into a `CompilationUnit` AST. |
| `parse_file` | `fn parse_file(path: &Path) -> Result<CompilationUnit, Vec<Diagnostic>>` | Read and parse a source file. |
| `encode` / `decode` | `ToBytecode` / `FromBytecode` | Serialize/deserialize an AST to/from 🐕🇨 bytecode bytes. |
| `CompilationUnit` | struct | The root AST node (namespaces → classes → members). |
| `NativeTarget` | `pub use native::target::Target` | The native target enum the core drives a backend through. |

### Driver

`src/driver.rs` is the pipeline: parse → preprocess → lower → emit, and
the entry point the CLI drives.

```rust
// Compile source text to a native executable image.
pub fn compile_source(
    source: &str,
    libs: &[PathBuf],
    pipeline: &NativePipeline,
) -> Result<Vec<u8>, String>;

// JIT-execute source text, returning its exit status.
pub fn run_source(source: &str, pipeline: &NativePipeline) -> Result<i32, String>;

// Parse only; the error carries the diagnostics.
pub fn parse_source(source: &str) -> Result<CompilationUnit, Vec<Diagnostic>>;
```

`cli_main` mirrors the CLI and is the entry point the `rubyc` binary
calls.

### IR

`src/native/ir/` is the seam between languages and targets:

```rust
// A target can build ANY language; a language targets ANY target.
pub fn lower(unit: &CompilationUnit) -> Result<Program, LowerError>;
pub fn lower_with(unit: &CompilationUnit, require_main: bool) -> Result<Program, LowerError>;
pub fn encode_program(program: &Program) -> Vec<u8>;     // IR → bytes
pub fn decode_program(bytes: &[u8]) -> Result<Program, _>; // bytes → IR
```

### Plugin ABI

`src/plugin/` defines the byte contract plugins pass across FFI:

```rust
/// A result blob: either success bytes or an error string.
pub struct PluginBytes { /* success: Vec<u8> | error: String */ }
impl PluginBytes {
    pub fn success(bytes: Vec<u8>) -> Self;
    pub fn error(msg: String) -> Self;
    pub fn take_result(&self) -> ...;
}
```

Target plugins export `rbxt_name` / `rbxt_triple` / `rbxt_base_addr` /
`rbxt_lower` / `rbxt_write_image` / `rbxt_write_shared` / `rbxt_run`. The
`libloading`-backed loader resolves these symbols from a `.so`.

### Cross-OS library loader

`src/libloader/` wraps `libloading` + `goblin` behind a single `Load` trait,
with per-OS modules (`elf/`, `macho/`, `pe/`). Two entry points:

```rust
/// Load a real on-disk cdylib via dlopen / LoadLibrary.
pub fn load_path(path: &Path) -> Result<Handle, LibError>;
/// Load hand-assembled bytes via the mmap loader (dlopen would crash).
pub fn load_bytes(bytes: &[u8]) -> Result<Handle, LibError>;
```

### Extensions & discovery

`src/extensions.rs` owns the install-root resolution and the polymorphic
metadata traits:

```rust
/// ~/.local/share/rubyc (XDG on Linux; the right location on macOS/Windows).
/// Precedence: $RUBYC_BASE > $RUBYC_HOME > platform default.
pub fn install_root() -> PathBuf;

pub struct ExtensionMetadata { name, display_name, author, version, description, license, homepage, repository }

pub trait TargetExtension: Send + Sync { fn metadata(&self) -> &ExtensionMetadata; fn triple(&self) -> &str; }
```

### Project system

`src/project.rs` reads and resolves `.rcp` (project) and `.rcs` (solution) files:

```rust
pub fn find_nearest(start_dir: &Path) -> Option<PathBuf>;  // walk up for .rcp/.rcs
pub fn load(path: &Path) -> Result<Self, String>;          // a project or a solution
```

### Lints

```rust
pub fn analyze(unit: &CompilationUnit, config: &LintConfig) -> Vec<Diagnostic>;
// config: --warn / --deny / --allow; warnings never fail builds unless denied
```

## Plugin directory layout

After building, the CLI installs plugins to `install_root()`:

```
$install_root/
  targets/<name>/{<name>.toml, lib<name>.so}
```

## Building

`rubyc-core` is a workspace member (see the [root](../) `Cargo.toml`). From the
workspace root:

```bash
cargo build -p rubyc-core
cargo test -p rubyc-core
cargo clippy -p rubyc-core --all-targets
```

Its `build.rs` emits the `target_os_<name>` check-cfgs and generates the i18n
table from `locales/en.json` (build-time codegen).
