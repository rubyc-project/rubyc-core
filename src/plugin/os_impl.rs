//! Platform dispatch: bridges the cross-OS `libloader` to the plugin layer.
//!
//! On each OS, the matching backend (`libloader::elf` / `libloader::macho` /
//! `libloader::pe`) is used to load the shared object and resolve the ABI
//! symbol names. The plugin layer then calls the resolved function pointers
//! directly (absolute addresses from `dlsym`/`GetProcAddress`).

use std::collections::BTreeMap;
use std::path::Path;

use super::plugin::PluginError;
use crate::libloader::{LibError, Load};

/// ABI symbol names the plugin layer requires, resolved up-front.
const ABI_SYMBOLS: &[&str] = &[
    "rbxt_name",
    "rbxt_triple",
    "rbxt_base_addr",
    "rbxt_lower",
    "rbxt_write_image",
    "rbxt_write_shared",
    "rbxt_run",
];

/// Keeps the loaded library alive for the lifetime of the `Plugin`.
pub struct OsHandle {
    _handle: Box<dyn Load>,
}

impl Drop for OsHandle {
    fn drop(&mut self) {
        // The `Box<dyn Load>` is dropped here, which runs the backend's `Drop`
        // (e.g. `dlclose`/`FreeLibrary`).
    }
}

pub fn load_path(path: &Path) -> Result<(u64, BTreeMap<String, *const u8>, OsHandle), PluginError> {
    // Dispatch to the OS-specific backend.
    let handle: Box<dyn Load> = load_path_inner(path)?;
    let base = handle.base();
    let syms = handle.symbols(ABI_SYMBOLS);
    let symbols: BTreeMap<String, *const u8> = syms
        .iter()
        .map(|(name, addr)| (name.clone(), *addr as *const u8))
        .collect();
    Ok((base, symbols, OsHandle { _handle: handle }))
}

pub fn load_bytes(
    bytes: &[u8],
) -> Result<(u64, BTreeMap<String, *const u8>, OsHandle), PluginError> {
    let handle: Box<dyn Load> = load_bytes_inner(bytes)?;
    let base = handle.base();
    let syms = handle.symbols(ABI_SYMBOLS);
    let symbols: BTreeMap<String, *const u8> = syms
        .iter()
        .map(|(name, addr)| (name.clone(), *addr as *const u8))
        .collect();
    Ok((base, symbols, OsHandle { _handle: handle }))
}

fn map_err(e: LibError) -> PluginError {
    PluginError::Load(e.to_string())
}

// The three supported hosts share one implementation (each OS's `libloader`
// backend already dispatches to its own `load_path`/`load_bytes`).
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
fn load_path_inner(path: &Path) -> Result<Box<dyn Load>, PluginError> {
    Ok(Box::new(crate::libloader::load_path(path).map_err(map_err)?))
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
fn load_bytes_inner(bytes: &[u8]) -> Result<Box<dyn Load>, PluginError> {
    Ok(Box::new(crate::libloader::load_bytes(bytes).map_err(map_err)?))
}
