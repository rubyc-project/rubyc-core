mod abi;
mod loader;
mod plugin;
pub use abi::PluginBytes;
pub use loader::{discover_targets, load_target_plugin, TargetPlugin};
pub use plugin::{Plugin, PluginError};

#[cfg(target_os = "linux")]
pub(crate) mod linux;
mod os_impl;

#[cfg(test)]
pub mod roundtrip;
