//! Central CLI specification and hand-rolled argument parser.
//!
//! Everything the command line accepts lives in these tables — they drive
//! parsing AND help rendering, so documentation cannot drift from behavior.
//! Zero third-party dependencies.

mod args;
mod help;
pub mod spec;

pub use args::parse_args;
pub use help::{render_help, version_text};
pub use spec::{ArgKind, COMMANDS, CommandSpec, FlagSpec, GLOBAL_FLAGS};

// ---------- output formats ----------

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ErrorFormat {
    Text,
    Json,
    JsonPretty,
}

// ---------- parsed results ----------

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum InFormat {
    Auto,
    Source,
    Bytecode,
}

impl InFormat {
    pub(super) fn from_str(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(Self::Auto),
            "source" => Some(Self::Source),
            "bytecode" => Some(Self::Bytecode),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum OutFormat {
    Bytecode,
    BytecodeShebang,
    Base64Bytecode,
    Exe,
    /// Shared library (ET_DYN .so) exporting public methods.
    Shared,
    /// Static archive (.a) of the library's unlinked machine code.
    Static,
}

impl OutFormat {
    pub(super) fn from_str(s: &str) -> Option<Self> {
        match s {
            "bytecode" => Some(Self::Bytecode),
            "bytecode-shebang" => Some(Self::BytecodeShebang),
            "base64-bytecode" => Some(Self::Base64Bytecode),
            "exe" => Some(Self::Exe),
            "shared" => Some(Self::Shared),
            "static" => Some(Self::Static),
            _ => None,
        }
    }
}

/// Shared I/O options consumed by every command handler.
#[derive(Default, Clone, Debug)]
pub struct IoArgs {
    pub file: Option<String>,
    pub input: Option<String>,
    pub output: Option<String>,
    pub in_format: Option<InFormat>,
    pub out_format: Option<OutFormat>,
    pub base64: bool,
    pub shebang: Option<String>,
    /// Shared objects for `[LibraryImport]` linking (`--lib`, repeatable).
    pub libs: Vec<String>,
    /// Extension installation root (`--base`): `targets/` inside.
    pub base: Option<String>,
    /// Explicit target-manifest directory (`--targets-dir`); beats `--base`.
    pub targets_dir: Option<String>,
    /// Selected target name (`--target`); validated against discovery.
    pub target: Option<String>,
    /// Plugin kind for install/remove/update (`--kind`): "target".
    pub kind: Option<String>,
    /// URL or local path for download/install (`--url`).
    pub url: Option<String>,
    /// Emit debug symbols (.symtab) in ELF output (`--debug`).
    pub debug: bool,
    /// `new --solution`: scaffold a `.rcs` solution instead of a `.rcp` project.
    pub solution: bool,
    /// Suppress the `🐕🇨` RubyC preamble in all emitted artifacts
    /// (AST bytecode, IR, and executable image) (`--no-magic`).
    pub no_magic: bool,
    /// Skip pre-build hooks: assumes transient sources were already
    /// materialized (`--skip-pre-hook`). Dev-loop only — normal
    /// builds must run hooks so `.rc` staging happens.
    pub skip_pre_hook: bool,
    /// Skip post-build hooks: leaves transient sources in place
    /// (`--skip-post-hook`). Dev-loop only.
    pub skip_post_hook: bool,
    /// Emit only the `.rci` sidecar, skipping codegen and image writes
    /// (`--rci-only`). Backfill mode: fragments for downstream
    /// dynamic builds without paying for machine code.
    pub rci_only: bool,
    /// Solution build parallelism (`--jobs N`, default 1): how many
    /// projects build concurrently. Single-project builds ignore it.
    pub jobs: usize,
    /// Embed the library's type fragment in the built image
    /// (default on for library outputs; `--no-embed-types` disables).
    /// Consumers read types from the binary itself instead of an
    /// `.rci` sidecar file. Plain C/C++ hosts take the opt-out for a
    /// byte-identical pure-ABI image.
    pub no_embed_types: bool,
    /// Library set format for `install libs` (`--format
    /// shared|static|both`, default `shared`): which image kind each
    /// pulled library builds. `both` runs a shared pass then a static
    /// pass (dynamic projects skip the static pass — see `build_project`).
    pub install_format: Option<String>,
    /// Function-level dead-code elimination (`--optimize`):
    /// drop functions unreachable from the entry point, exports,
    /// `@init`, and address-taken references. Off by default: output
    /// is byte-identical to the unoptimized image.
    pub optimize: bool,
    /// Drop `.symtab`/`.strtab` from executable images (`--strip`).
    /// Shared libraries always keep `.dynsym` (required for linking).
    pub strip: bool,
    /// Solution profile to apply (`--profile NAME`): merges the named
    /// `[profile]` section's defines/optimize/strip into every project
    /// of a solution build. Unset = no profile.
    pub profile: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Command {
    Compile,
    Build,
    Run,
    Lint,
    Dump,
    CheckRoundtrip,
    List,
    Install,
    Download,
    Remove,
    Update,
    ListRemote,
    New,
    Create,
    Add,
}

/// A fully parsed command line.
#[derive(Debug, Clone)]
pub struct Invocation {
    pub command: Command,
    pub io: IoArgs,
    pub verbose: bool,
    pub quiet: bool,
    pub error_format: ErrorFormat,
    pub no_color: bool,
    /// Lint level overrides from `--warn/--deny/--allow`, in CLI order.
    pub lint_overrides: Vec<(String, crate::lints::Level)>,
    /// Symbols seeded for `#If` (from `--define`), e.g. DEBUG.
    pub defines: Vec<String>,
}

/// Argument-parsing failures, rendered by [`render_error`].
#[derive(Debug, Clone, PartialEq)]
pub enum CliError {
    UnknownCommand(String),
    UnknownFlag(String),
    MissingValue(String),
    InvalidValue {
        flag: String,
        value: String,
        allowed: Vec<String>,
    },
    DuplicatePositional(String),
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CliError::UnknownCommand(name) => {
                write!(
                    f,
                    "{}",
                    crate::tf!("rubyc.cli.err.unknown_command", name.clone())
                )
            }
            CliError::UnknownFlag(flag) => {
                write!(
                    f,
                    "{}",
                    crate::tf!("rubyc.cli.err.unknown_flag", flag.clone())
                )
            }
            CliError::MissingValue(flag) => {
                write!(
                    f,
                    "{}",
                    crate::tf!("rubyc.cli.err.missing_value", flag.clone())
                )
            }
            CliError::InvalidValue {
                flag,
                value,
                allowed,
            } => {
                write!(
                    f,
                    "invalid value '{value}' for '{flag}' (allowed: {})",
                    allowed.join(", ")
                )
            }
            CliError::DuplicatePositional(word) => {
                write!(
                    f,
                    "{}",
                    crate::tf!("rubyc.cli.err.extra_positional", word.clone())
                )
            }
        }
    }
}

/// Outcome when the user asked for passive output instead of compilation.
pub enum ParsedArgs {
    Run(Invocation),
    /// `-h` / `--help` / `help`: text to print, colored per flag.
    Help(String),
    /// `-V` / `--version`.
    Version(String),
}
