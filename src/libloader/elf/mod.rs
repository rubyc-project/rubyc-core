//! Linux ELF loader.
//!
//! Primary path: raw `dlopen`/`dlsym`/`dlclose`/`munmap` FFI for real cdylib
//! plugins. Fallback: the hand-rolled `mmap`-based ELF loader
//! (`crate::plugin::linux::load_elf`) for hand-assembled ELF images (e.g. the
//! roundtrip test) that `dlopen` rejects (corrupt section headers, no dynamic
//! relocations, etc.).
//!
//! This module does not use `libloading` (that wrapper is only used by the
//! Windows PE loader); it binds the `libdl`/`libc` entry points directly.

use std::collections::BTreeMap;
use std::os::raw::c_void;
use std::path::Path;

use super::error::LibError;
use super::Load;

unsafe extern "C" {
    fn dlopen(filename: *const std::os::raw::c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const std::os::raw::c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> i32;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
}

/// `dlopen` flags: resolve all symbol references immediately (`RTLD_NOW`)
/// rather than lazily on first use, so a missing dependency fails at load
/// time. `RTLD_LOCAL` keeps the plugin's symbols out of the global namespace
/// (the plugin resolves its own `libgcc_s`/`libc` references independently).
const RTLD_NOW: i32 = 0x2;
const RTLD_LOCAL: i32 = 0x0;

/// Handle to a loaded shared object. Either a `dlopen` handle (real cdylib)
/// or a `mmap`-ed region (hand-assembled ELF).
pub enum Handle {
    Dl {
        handle: *mut c_void,
        base: u64,
    },
    Mmap {
        region: *mut u8,
        region_len: usize,
        base: u64,
        symbols: BTreeMap<String, u64>,
    },
}

unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

impl Load for Handle {
    fn sym(&self, name: &str) -> Option<u64> {
        match self {
            Handle::Dl { handle, .. } => {
                let c = std::ffi::CString::new(name).ok()?;
                let ptr = unsafe { dlsym(*handle, c.as_ptr()) };
                (!ptr.is_null()).then_some(ptr as u64)
            }
            Handle::Mmap { symbols, .. } => symbols.get(name).copied(),
        }
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
        match self {
            Handle::Dl { base, .. } => *base,
            Handle::Mmap { base, .. } => *base,
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        match self {
            Handle::Dl { handle, .. } => {
                if !handle.is_null() {
                    unsafe { dlclose(*handle) };
                }
            }
            Handle::Mmap { region, region_len, .. } => {
                unsafe {
                    munmap(*region as *mut c_void, *region_len);
                }
            }
        }
    }
}

/// Load a shared object from `path`. Tries `dlopen` first; falls back to the
/// `mmap` loader if the OS loader rejects the image.
///
/// Uses `RTLD_NOW | RTLD_LOCAL`: `RTLD_NOW` resolves the plugin's symbol
/// references (it links against `libgcc_s`, `libc`, and its own static data)
/// immediately so a missing dependency fails at load time; `RTLD_LOCAL` keeps
/// the plugin's symbols out of the global namespace.
pub fn load_path(path: &Path) -> Result<Handle, LibError> {
    let cpath = std::ffi::CString::new(path.to_string_lossy().as_bytes())
        .map_err(|e| LibError::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, e)))?;
    let handle = unsafe { dlopen(cpath.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
    if handle.is_null() {
        let bytes = std::fs::read(path)?;
        return load_mmap(&bytes);
    }
    Ok(Handle::Dl { handle, base: 0 })
}

/// Load a shared object from in-memory bytes.
///
/// Goes straight to the `mmap` loader (no `dlopen`): in-memory images are
/// typically hand-assembled ELFs (e.g. the roundtrip test) that `dlopen`
/// either rejects or crashes on. Real cdylib plugins are loaded via
/// [`load_path`], which uses `dlopen`.
pub fn load_bytes(bytes: &[u8]) -> Result<Handle, LibError> {
    if bytes.len() < 64 || &bytes[0..4] != &[0x7f, b'E', b'L', b'F'] {
        return Err(LibError::BadImage("not an ELF image".to_string()));
    }
    // Validate with goblin so a corrupt image fails with a clear error rather
    // than a silent `mmap` of garbage.
    goblin::elf::Elf::parse(bytes)
        .map_err(|e| LibError::BadImage(format!("ELF parse failed: {e}")))?;
    load_mmap(bytes)
}

/// The hand-rolled `mmap`-based ELF loader (no constructors, no relocations).
fn load_mmap(bytes: &[u8]) -> Result<Handle, LibError> {
    let (base, symbols, region, region_len) = crate::plugin::linux::load_elf(bytes)
        .map_err(|e| LibError::BadImage(e))?;
    // Convert *const u8 -> u64 for the symbol map.
    let symbols: BTreeMap<String, u64> = symbols
        .into_iter()
        .map(|(name, ptr)| (name, ptr as u64))
        .collect();
    // Re-encode region as *mut u8.
    let region = region as *mut u8;
    Ok(Handle::Mmap {
        region,
        region_len,
        base,
        symbols,
    })
}
