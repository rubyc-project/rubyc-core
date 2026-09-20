use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug)]
pub enum PluginError {
    UnsupportedPlatform,
    Io(std::io::Error),
    Load(String),
    MissingSymbol(String),
}

impl std::fmt::Display for PluginError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PluginError::UnsupportedPlatform => write!(f, "plugin loading is not supported on this platform"),
            PluginError::Io(err) => write!(f, "io error: {err}"),
            PluginError::Load(msg) => write!(f, "plugin load failed: {msg}"),
            PluginError::MissingSymbol(name) => write!(f, "missing exported symbol: {name}"),
        }
    }
}

impl std::error::Error for PluginError {}

impl From<std::io::Error> for PluginError {
    fn from(err: std::io::Error) -> Self {
        PluginError::Io(err)
    }
}

pub struct Plugin {
    base: u64,
    symbols: BTreeMap<String, *const u8>,
    _handle: super::os_impl::OsHandle,
}

impl Plugin {
    pub fn load_path(path: &Path) -> Result<Self, PluginError> {
        let (base, symbols, handle) = super::os_impl::load_path(path)?;
        Ok(Self { base, symbols, _handle: handle })
    }

    pub fn load_bytes(bytes: &[u8]) -> Result<Self, PluginError> {
        let (base, symbols, handle) = super::os_impl::load_bytes(bytes)?;
        Ok(Self { base, symbols, _handle: handle })
    }

    pub fn base(&self) -> u64 {
        self.base
    }

    pub fn sym(&self, name: &str) -> Option<*const u8> {
        self.symbols.get(name).copied()
    }

    pub fn require_sym(&self, name: &str) -> Result<*const u8, PluginError> {
        self.symbols
            .get(name)
            .copied()
            .ok_or_else(|| PluginError::MissingSymbol(name.to_string()))
    }

    pub fn symbol_names(&self) -> impl Iterator<Item = &str> {
        self.symbols.keys().map(|s| s.as_str())
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::plugin::roundtrip::build_roundtrip_plugin;


    type FnPtr = unsafe fn() -> usize;

    #[test]
    fn roundtrip_plugin_exports_roundtrip() {
        let bytes = build_roundtrip_plugin();
        let plugin = Plugin::load_bytes(&bytes).unwrap();
        assert_ne!(plugin.base(), 0);

        let name_sym = plugin.require_sym("rbxt_name").unwrap();
        let triple_sym = plugin.require_sym("rbxt_triple").unwrap();
        let base_sym = plugin.require_sym("rbxt_base_addr").unwrap();

        assert_ne!(name_sym as u64, 0);
        assert_ne!(triple_sym as u64, 0);
        assert_ne!(base_sym as u64, 0);

        assert_eq!(plugin.base(), name_sym as u64 - 0x180);
        assert_eq!(plugin.base(), triple_sym as u64 - 0x188);
        assert_eq!(plugin.base(), base_sym as u64 - 0x190);
    }
}
