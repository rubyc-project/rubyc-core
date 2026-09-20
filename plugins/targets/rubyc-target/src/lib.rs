//! Polymorphic compilation-target trait and registry.
//!
//! Each target backend lives in its own crate and implements
//! [`TargetBackend`]. The registry holds the compile-time built-in backends;
//! `.so` target plugins are discovered and loaded by the core.

pub use rubyc::extensions::{ExtensionMetadata, TargetExtension};

/// What every compilation target must provide.
pub trait TargetBackend: Send + Sync {
    /// Canonical name (matches directory under `~/.local/share/rubyc/targets/`).
    fn name(&self) -> &'static str;
    /// Rust-style triple, e.g. `"x86_64-unknown-linux-gnu"`.
    fn triple(&self) -> &'static str;
    /// Lower IR to machine code for this target.
    fn lower(&self, program: &[u8]) -> Result<Vec<u8>, String>;
    /// Write the final image (ELF/PE/Mach-O/flat binary).
    fn write_image(
        &self,
        mc: &[u8],
        entry_preamble: bool,
    ) -> Result<Vec<u8>, String>;
}

/// Registry that knows about all available targets.
pub(crate) mod discovery;

pub struct TargetRegistry {
    backends: Vec<Box<dyn TargetBackend>>,
}

impl TargetRegistry {
    /// Create a registry with only built-in targets.
    pub fn builtin() -> Self {
        let mut reg = Self {
            backends: Vec::new(),
        };
        // Built-in linux_x86_64 is compiled in.
        if let Some(b) = crate::discovery::load_builtin("linux_x86_64") {
            reg.backends.push(b);
        }
        reg
    }

    pub fn list_targets(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.backends.iter().map(|b| b.name()).collect();
        names.sort();
        names
    }

    pub fn find_target(&self, name: &str) -> Option<&dyn TargetBackend> {
        self.backends.iter().find(|b| b.name() == name).map(|b| b.as_ref())
    }

    pub fn backend_count(&self) -> usize {
        self.backends.len()
    }
}

#[cfg(test)]
#[path = "../tests/mod.rs"]
mod tests;
