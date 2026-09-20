//! macOS Mach-O loader.
//!
//! Uses `libloading`'s safe `Library` API (the same `dlopen`/`dlsym`/`dlclose`
//! wrapper as Linux) to load the dylib, and `goblin` to:
//! - validate the magic (thin `MH_MAGIC`/`MH_MAGIC_64` or fat `FAT_MAGIC`),
//! - for fat (universal) binaries, select the slice matching the host arch and
//!   spill just that slice to a temp file before loading.
//!
//! Reference: `libloading` (unix shim) and `goblin` (github.com/m4b/goblin)
//! `mach/fat.rs`, `mach/header.rs`.

use std::collections::BTreeMap;
use std::path::Path;

use libloading::Library;

use super::error::LibError;
use super::Load;

/// Handle to a loaded dylib.
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

const MH_MAGIC: u32 = 0xFEED_FACE;
const MH_MAGIC_64: u32 = 0xFEED_FACF;
const FAT_MAGIC: u32 = 0xCAFE_BABE;
const FAT_MAGIC_64: u32 = 0xCAFE_BABF;

/// Host arch as a Mach-O `cpu_type` (for fat-slice selection).
fn host_cpu_type() -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        0x0100_0007 // CPU_TYPE_ARM64
    }
    #[cfg(target_arch = "x86_64")]
    {
        0x0100_0006 // CPU_TYPE_X86_64
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        0
    }
}

/// Validate the magic and, for fat binaries, return the offset of the host-arch
/// slice (0 for thin binaries).
fn host_slice_offset(bytes: &[u8]) -> Result<usize, LibError> {
    if bytes.len() < 8 {
        return Err(LibError::BadImage("truncated Mach-O".to_string()));
    }
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    match magic {
        MH_MAGIC | MH_MAGIC_64 => Ok(0),
        FAT_MAGIC | FAT_MAGIC_64 => {
            if bytes.len() < 12 {
                return Err(LibError::BadImage("truncated fat header".to_string()));
            }
            let nfat = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
            let host = host_cpu_type();
            for i in 0..nfat {
                let off = 8 + i * 20;
                if off + 20 > bytes.len() {
                    return Err(LibError::BadImage("truncated fat arch entry".to_string()));
                }
                let cpu = u32::from_le_bytes([
                    bytes[off],
                    bytes[off + 1],
                    bytes[off + 2],
                    bytes[off + 3],
                ]);
                if cpu == host {
                    let offset = u32::from_le_bytes([
                        bytes[off + 8],
                        bytes[off + 9],
                        bytes[off + 10],
                        bytes[off + 11],
                    ]) as usize;
                    return Ok(offset);
                }
            }
            Err(LibError::BadImage(format!(
                "no fat slice for host arch (cpu_type={host:#x})"
            )))
        }
        _ => Err(LibError::BadImage(format!("bad Mach-O magic {magic:#x}"))),
    }
}

/// Load a dylib from `path`.
pub fn load_path(path: &Path) -> Result<Handle, LibError> {
    let lib = unsafe { Library::new(path) }
        .map_err(|e| LibError::LoadFailed(format!("{e}")))?;
    Ok(Handle { lib, base: 0 })
}

/// Load a dylib from in-memory bytes.
///
/// Thin images are spilled whole; fat images are reduced to the host-arch slice
/// so the OS loader gets a clean thin image.
pub fn load_bytes(bytes: &[u8]) -> Result<Handle, LibError> {
    let off = host_slice_offset(bytes)?;
    let image = &bytes[off..];
    // Validate the thin slice with goblin (parses the Mach-O header + load
    // commands) so a corrupt image fails with a clear error.
    goblin::mach::MachO::parse(image, 0)
        .map_err(|e| LibError::BadImage(format!("Mach-O parse failed: {e}")))?;
    let mut tmp = std::env::temp_dir();
    tmp.push(format!("rubyc-lib-{:x}.dylib", std::process::id()));
    std::fs::write(&tmp, image)?;
    let result = load_path(&tmp);
    let _ = std::fs::remove_file(&tmp);
    result
}
