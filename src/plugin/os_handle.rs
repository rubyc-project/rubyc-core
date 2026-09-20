use std::collections::BTreeMap;
use std::path::Path;

use super::plugin::PluginError;

pub struct OsHandle {
    region: *mut u8,
    region_len: usize,
}

impl Drop for OsHandle {
    fn drop(&mut self) {
        unsafe {
            crate::plugin::os_handle::munmap(self.region, self.region_len);
        }
    }
}

pub fn load_path(path: &Path) -> Result<(u64, BTreeMap<String, *const u8>, OsHandle), PluginError> {
    let bytes = std::fs::read(path).map_err(PluginError::Io)?;
    load_bytes(&bytes)
}

pub fn load_bytes(bytes: &[u8]) -> Result<(u64, BTreeMap<String, *const u8>, OsHandle), PluginError> {
    let (base, symbols, region, region_len) = crate::plugin::elf_load(bytes).map_err(PluginError::Load)?;
    Ok((
        base,
        symbols,
        OsHandle { region, region_len },
    ))
}

extern "C" {
    fn munmap(addr: *mut u8, len: usize) -> i32;
}
