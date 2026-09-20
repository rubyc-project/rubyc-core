//! Plugin loader: turns a manifest + `.so` into a callable plugin.
//!
//! A plugin lives in a directory containing two files:
//! - `plugin.toml` (or `<name>.toml`) — manifest (name, version, triple,
//!   library, entry, kind, dependencies)
//! - `lib<name>.so` — the cdylib (symlinked from the build tree at install
//!   time)
//!
//! The loader reads the manifest, loads the ELF via the hand-rolled loader
//! ([`crate::plugin::Plugin::load_path`]), and resolves the ABI entry
//! symbols. The host then calls the plugin through the function pointers in
//! [`TargetPlugin`], passing and receiving raw bytes.

use std::os::raw::c_char;
use std::path::Path;
use std::sync::Arc;

use crate::extensions::{ExtensionMetadata, ExtensionRegistry, TargetInfo};

use super::abi::PluginBytes;
use super::Plugin;

/// A loaded target plugin: function pointers into the `.so`.
///
/// Every call site below uses inline `asm!` rather than a Rust fn-pointer call.
/// This is a workaround for a rustc codegen bug: casting a `dlsym` pointer to a
/// `*const (unsafe extern "C" fn())` and calling it emits a memory-indirect
/// `call *0x0(%rbp)` instead of a register-indirect `call *%rbp`.
///
/// **Do not** factor these into a shared helper: extracting the `asm!` into a
/// separate function triggers a *second* codegen bug where the caller fails to
/// set up `rsi`/`rdx` (the SysV arg registers) before the call, so the callee
/// reads garbage for `len`. The `asm!` must stay inline in each caller so the
/// `in("rdi")`/`in("rsi")` constraints are visible to the compiler at the call
/// site. The register constraints are fixed by the SysV ABI and must not be
/// re-derived.
#[derive(Clone)]
pub struct TargetPlugin {
    pub name: String,
    pub triple: String,
    pub metadata: ExtensionMetadata,
    pub plugin: Arc<Plugin>,
    pub lower: *const (),
    pub write_image: *const (),
    pub write_shared: *const (),
    pub run: *const (),
}

// Read-only dispatch table: every method takes `&self` and calls out
// through the function pointers without mutating them, so sharing one
// plugin across solution-build threads is sound (the call itself is
// already `unsafe` at each site).
unsafe impl Sync for TargetPlugin {}

impl TargetPlugin {
    /// Lower IR bytes to machine code bytes.
    pub unsafe fn lower(&self, ir: &[u8]) -> Result<Vec<u8>, String> {
        let f = self.lower as u64;
        let out: u64;
        unsafe {
            std::arch::asm!(
                "call {}",
                in(reg) f,
                in("rdi") ir.as_ptr() as *const c_char,
                in("rsi") ir.len() as u64,
                lateout("rax") out,
                options(nostack),
            );
        }
        unsafe { super::abi::take_result((*(out as *const PluginBytes)).clone_or_zero()) }
    }

    /// Write the machine-code bytes to an executable image.
    pub unsafe fn write_image(&self, mc: &[u8], entry_preamble: bool) -> Result<Vec<u8>, String> {
        let f = self.write_image as u64;
        let out: u64;
        unsafe {
            std::arch::asm!(
                "call {}",
                in(reg) f,
                in("rdi") mc.as_ptr() as *const c_char,
                in("rsi") mc.len() as u64,
                in("rdx") entry_preamble as u64,
                lateout("rax") out,
                options(nostack),
            );
        }
        unsafe { super::abi::take_result((*(out as *const PluginBytes)).clone_or_zero()) }
    }

    /// Write the machine-code bytes to a shared object (ET_DYN).
    pub unsafe fn write_shared(&self, mc: &[u8]) -> Result<Vec<u8>, String> {
        let f = self.write_shared as u64;
        let out: u64;
        unsafe {
            std::arch::asm!(
                "call {}",
                in(reg) f,
                in("rdi") mc.as_ptr() as *const c_char,
                in("rsi") mc.len() as u64,
                lateout("rax") out,
                options(nostack),
            );
        }
        unsafe { super::abi::take_result((*(out as *const PluginBytes)).clone_or_zero()) }
    }

    /// Execute machine-code bytes in-memory (JIT). Returns the exit status.
    pub unsafe fn run(&self, mc: &[u8]) -> i32 {
        let f = self.run as u64;
        let status: i32;
        unsafe {
            std::arch::asm!(
                "call {}",
                in(reg) f,
                in("rdi") mc.as_ptr() as *const c_char,
                in("rsi") mc.len() as u64,
                lateout("eax") status,
                options(nostack),
            );
        }
        status
    }
}

impl PluginBytes {
    /// Copy the buffer into a new owned `PluginBytes` (for `take_result`).
    pub fn clone_or_zero(&self) -> PluginBytes {
        if self.data.is_null() || self.len == 0 {
            return PluginBytes::empty();
        }
        let mut v = Vec::with_capacity(self.len as usize);
        unsafe {
            std::ptr::copy_nonoverlapping(self.data as *const u8, v.as_mut_ptr(), self.len as usize);
            v.set_len(self.len as usize);
        }
        if self.ok != 0 {
            PluginBytes::success(v)
        } else {
            let s = String::from_utf8_lossy(&v).to_string();
            PluginBytes::error(s)
        }
    }
}

struct Manifest {
    name: String,
    version: String,
    triple: String,
    library: String,
    metadata: ExtensionMetadata,
}

/// Parse the manifest in `dir`.
fn load_manifest(dir: &Path) -> Result<Manifest, String> {
    let candidate = dir.join("plugin.toml");
    let toml_path = if candidate.exists() {
        candidate
    } else {
        dir.file_name()
            .map(|n| dir.join(format!("{}.toml", n.to_string_lossy())))
            .filter(|p| p.exists())
            .unwrap_or_else(|| dir.join("plugin.toml"))
    };
    let raw = std::fs::read_to_string(&toml_path)
        .map_err(|e| format!("failed to read manifest {}: {e}", toml_path.display()))?;

    let name = get_str(&raw, "name")
        .unwrap_or_else(|| dir.file_name().map(|n| n.to_string_lossy().into()).unwrap_or_default());
    let version = get_str(&raw, "version").unwrap_or_else(|| "0.0.0".into());
    let triple = get_str(&raw, "triple").unwrap_or_default();
    let library = get_str(&raw, "library")
        .unwrap_or_else(|| format!("lib{}.so", name.replace('-', "_")));
    let display_name = get_str(&raw, "display-name")
        .or_else(|| get_str(&raw, "display_name"))
        .unwrap_or_else(|| name.clone());
    let author = get_str(&raw, "author").unwrap_or_default();
    let description = get_str(&raw, "description").unwrap_or_default();
    let license = get_str(&raw, "license").unwrap_or_else(|| "MIT".into());
    let homepage = get_str(&raw, "homepage").unwrap_or_default();
    let repository = get_str(&raw, "repository").unwrap_or_default();

    Ok(Manifest {
        name: name.clone(),
        version: version.clone(),
        triple,
        library,
        metadata: ExtensionMetadata {
            name,
            display_name,
            author,
            version,
            description,
            license,
            homepage,
            repository,
        },
    })
}

/// Extract a top-level string value from a simple TOML document.
fn get_str(raw: &str, key: &str) -> Option<String> {
    for line in raw.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.starts_with('[') {
            continue;
        }
        if let Some(idx) = line.find('=') {
            let k = line[..idx].trim();
            if k == key {
                let v = line[idx + 1..].trim();
                let v = v.trim_matches('"').trim_matches('\'');
                return Some(v.to_string());
            }
        }
    }
    None
}

/// Load a target plugin from a directory.
pub fn load_target_plugin(dir: &Path) -> Result<TargetPlugin, String> {
    let manifest = load_manifest(dir)?;
    let lib_path = dir.join(&manifest.library);
    let plugin = Plugin::load_path(&lib_path)
        .map_err(|e| format!("failed to load plugin {}: {e}", lib_path.display()))?;
    let plugin = Arc::new(plugin);

    let lower = plugin
        .require_sym("rbxt_lower")
        .map_err(|e| format!("target {}: {e}", manifest.name))? as *const ();
    let write_image = plugin
        .require_sym("rbxt_write_image")
        .map_err(|e| format!("target {}: {e}", manifest.name))? as *const ();
    let run = plugin
        .require_sym("rbxt_run")
        .map_err(|e| format!("target {}: {e}", manifest.name))? as *const ();
    let write_shared = plugin
        .require_sym("rbxt_write_shared")
        .map_err(|e| format!("target {}: {e}", manifest.name))? as *const ();

    Ok(TargetPlugin {
        name: manifest.name,
        triple: manifest.triple,
        metadata: manifest.metadata,
        plugin,
        lower,
        write_image,
        write_shared,
        run,
    })
}

/// Discover all target plugins under `base/targets/`.
pub fn discover_targets(base: &Path) -> Vec<TargetPlugin> {
    let dir = base.join("targets");
    if !dir.is_dir() {
        return Vec::new();
    }
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for e in entries.flatten() {
            if e.path().is_dir() {
                if let Ok(p) = load_target_plugin(&e.path()) {
                    out.push(p);
                }
            }
        }
    }
    out
}

/// Build an `ExtensionRegistry` from discovered target plugins, for
/// `--list-targets`.
#[allow(dead_code)]
pub fn registry_from_plugins(targets: &[TargetPlugin]) -> ExtensionRegistry {
    let mut reg = ExtensionRegistry::empty();
    for t in targets {
        reg.push_target(TargetInfo {
            metadata: t.metadata.clone(),
            triple: t.triple.clone(),
            manifest: None,
        });
    }
    reg
}
