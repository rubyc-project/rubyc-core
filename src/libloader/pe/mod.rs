//! Windows PE loader.
//!
//! Uses `libloading`'s safe `Library` API (a `LoadLibraryExW`/`GetProcAddress`/
//! `FreeLibrary` wrapper) to load the DLL, and `goblin` to validate the PE
//! magic and confirm the export directory is present.
//!
//! Reference: `libloading` (github.com/nagisa/rust_libloading), `os/windows/mod.rs`,
//! and `goblin` (github.com/m4b/goblin) `pe/mod.rs`.

use std::collections::BTreeMap;
use std::path::Path;

use libloading::Library;

use super::error::LibError;
use super::Load;

/// Handle to a loaded DLL.
pub struct Handle {
    lib: Library,
    base: u64,
}

unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

impl Load for Handle {
    fn sym(&self, name: &str) -> Option<u64> {
        let sym: libloading::Symbol<u64> = unsafe { self.lib.get(name.as_bytes()) }.ok()?;
        Some(*sym)
    }

    fn symbols(&self, names: &[&str]) -> BTreeMap<String, u64> {
        let mut out = BTreeMap::new();
        for name in names {
            if let Some(addr) = self.sym(name) {
                out.insert(name.to_string(), addr);
            }
        }
        out
    }

    fn base(&self) -> u64 {
        self.base
    }
}

/// Validate that `bytes` is a well-formed PE image with an export table.
fn validate_pe(bytes: &[u8]) -> Result<u64, LibError> {
    let pe = goblin::pe::PE::parse(bytes)
        .map_err(|e| LibError::BadImage(format!("PE parse failed: {e}")))?;
    if pe.exports.is_empty() {
        return Err(LibError::BadImage("PE has no exported symbols".to_string()));
    }
    Ok(pe.image_base)
}

/// Load a DLL from `path`.
pub fn load_path(path: &Path) -> Result<Handle, LibError> {
    let lib = unsafe { Library::new(path) }
        .map_err(|e| LibError::LoadFailed(format!("{e}")))?;
    Ok(Handle { lib, base: 0 })
}

/// Load a DLL from in-memory bytes (spills to a temp file, then loads).
///
/// `libloading` requires a file path; for in-memory images we validate with
/// `goblin`, spill to a temp file, load, then clean up.
pub fn load_bytes(bytes: &[u8]) -> Result<Handle, LibError> {
    let base = validate_pe(bytes)?;
    let mut tmp = std::env::temp_dir();
    tmp.push(format!("rubyc-lib-{:x}.dll", std::process::id()));
    std::fs::write(&tmp, bytes)?;
    let result = load_path(&tmp);
    let _ = std::fs::remove_file(&tmp);
    let mut handle = result?;
    handle.base = base;
    Ok(handle)
}
