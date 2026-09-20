//! Cross-OS dynamic library loader.
//!
//! Wraps `libloading` (the OS loader FFI) + `goblin` (binary format validation)
//! behind a single `Load` trait, so the plugin layer is platform-agnostic.
//!
//! - Linux → ELF (`elf/`)
//! - macOS → Mach-O (`macho/`)
//! - Windows → PE (`pe/`)
//!
//! The host resolves ABI symbols (`rbxt_*`) through
//! [`Load::sym`], which returns the process address of the exported function.

pub mod error;
pub use error::LibError;

/// A loaded native library.
pub trait Load {
    /// Look up an exported symbol by name, returning its process address.
    fn sym(&self, name: &str) -> Option<u64>;
    /// Resolve a set of known symbol names into a name → address map.
    fn symbols(&self, names: &[&str]) -> std::collections::BTreeMap<String, u64>;
    /// The base address of the loaded object (0 if unavailable).
    fn base(&self) -> u64;
}

#[cfg(target_os = "linux")]
mod elf;
#[cfg(target_os = "linux")]
pub use elf::*;

#[cfg(target_os = "macos")]
mod macho;
#[cfg(target_os = "macos")]
pub use macho::*;

#[cfg(target_os = "windows")]
mod pe;
#[cfg(target_os = "windows")]
pub use pe::*;


