//! Built-in target discovery.
//!
#![allow(dead_code)]

//! Target backends are either compile-time built-ins (loaded via
//! [`load_builtin`]) or `.so` plugins discovered by the core. This module
//! provides the default install directory and the built-in backend loader.

/// Default discovery directory per platform.
pub fn default_targets_dir() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("RUBYC_TARGETS_DIR") {
        return dir.into();
    }
    home_local().join("rubyc").join("targets")
}

fn home_local() -> std::path::PathBuf {
    #[cfg(target_os = "macos")]
    {
        if let Ok(home) = std::env::var("HOME") {
            return std::path::PathBuf::from(home)
                .join("Library/Application Support");
        }
        std::path::PathBuf::from(".")
    }
    #[cfg(target_os = "windows")]
    {
        if let Ok(dir) = std::env::var("LOCALAPPDATA") {
            return dir.into();
        }
        std::path::PathBuf::from(".")
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        if let Ok(home) = std::env::var("HOME") {
            return std::path::PathBuf::from(home).join(".local");
        }
        std::path::PathBuf::from(".")
    }
}

/// Load a built-in backend by name (compile-time linked).
pub fn load_builtin(name: &str) -> Option<Box<dyn crate::TargetBackend>> {
    // linux_x86_64 is the only compile-time-linked backend for now.
    // Additional targets are discovered via manifest files.
    match name {
        "linux_x86_64" => Some(Box::new(BuiltinLinuxX86_64)),
        _ => None,
    }
}

struct BuiltinLinuxX86_64;

impl crate::TargetBackend for BuiltinLinuxX86_64 {
    fn name(&self) -> &'static str {
        "linux_x86_64"
    }
    fn triple(&self) -> &'static str {
        "x86_64-unknown-linux-gnu"
    }
    fn lower(&self, _program: &[u8]) -> Result<Vec<u8>, String> {
        Err("linux_x86_64 lowering handled by core pipeline".into())
    }
    fn write_image(&self, _mc: &[u8], _entry_preamble: bool) -> Result<Vec<u8>, String> {
        Err("linux_x86_64 image writing handled by core pipeline".into())
    }
}


