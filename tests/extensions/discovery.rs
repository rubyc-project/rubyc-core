use crate::extensions::{default_targets_dir, install_root, ExtensionRegistry};

fn temp_dir(label: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("rubyc_{label}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn builtin_metadata_is_complete() {
    let registry = ExtensionRegistry::builtin();
    let target = &registry.targets()[0];
    assert_eq!(target.metadata.name, "linux_x86_64");
    assert!(!target.metadata.author.is_empty());
    assert!(!target.metadata.version.is_empty());
    assert!(!target.triple.is_empty());
}

#[test]
fn explicit_directories_discover_manifests() {
    let targets = temp_dir("target_manifests");
    std::fs::write(targets.join("wasm.toml"), r#"
[target]
name = "wasm32"
display-name = "WebAssembly"
author = "Target Author"
version = "2.3.4"
description = "test target"
license = "Apache-2.0"
triple = "wasm32-unknown-unknown"
"#).unwrap();
    let registry = ExtensionRegistry::discover(Some(&targets));
    let target = registry.targets().iter().find(|v| v.metadata.name == "wasm32").unwrap();
    assert_eq!(target.metadata.author, "Target Author");
    assert_eq!(target.triple, "wasm32-unknown-unknown");
    let _ = std::fs::remove_dir_all(targets);
}

#[test]
fn wrong_kind_manifest_is_ignored_by_target_scan() {
    let dir = temp_dir("wrong_kind");
    std::fs::write(dir.join("language.toml"), "[language]\nname = \"not-a-target\"").unwrap();
    let registry = ExtensionRegistry::discover(Some(&dir));
    assert_eq!(registry.targets().len(), 1);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn platform_defaults_end_in_extension_directories() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvGuard::new();
    assert!(default_targets_dir().ends_with("rubyc/targets"));
}

// ---------- base directory override (D7: --base / RUBYC_BASE) ----------

const BASE_TARGET: &str = "[target]\nname = \"wasm32\"\ndisplay-name = \"WebAssembly\"\nauthor = \"Base Author\"\nversion = \"1.0\"\ndescription = \"base dir target\"\ntriple = \"wasm32-unknown-unknown\"\n";

#[test]
fn base_directory_maps_to_kind_subdirs() {
    let base = temp_dir("base_map");
    std::fs::create_dir_all(base.join("targets")).unwrap();
    std::fs::write(base.join("targets/wasm.toml"), BASE_TARGET).unwrap();

    let registry = ExtensionRegistry::discover_with(Some(&base), None);
    assert!(
        registry.targets().iter().any(|t| t.metadata.name == "wasm32"),
        "base/targets must be scanned"
    );
    // builtins must still be present
    assert!(registry.targets().iter().any(|t| t.metadata.name == "linux_x86_64"));
    let _ = std::fs::remove_dir_all(base);
}

#[test]
fn explicit_kind_dir_wins_over_base() {
    let base = temp_dir("base_lose");
    let explicit = temp_dir("explicit_win");
    std::fs::create_dir_all(base.join("targets")).unwrap();
    std::fs::write(base.join("targets/from_base.toml"), "[target]\nname = \"from_base\"\nversion = \"1\"\n").unwrap();
    std::fs::write(explicit.join("from_explicit.toml"), "[target]\nname = \"from_explicit\"\nversion = \"1\"\n").unwrap();

    let registry = ExtensionRegistry::discover_with(Some(&base), Some(&explicit));
    assert!(
        registry.targets().iter().any(|t| t.metadata.name == "from_explicit"),
        "explicit per-kind dir must be scanned"
    );
    assert!(
        !registry.targets().iter().any(|t| t.metadata.name == "from_base"),
        "base/targets must be ignored when an explicit per-kind dir is given"
    );
    let _ = std::fs::remove_dir_all(base);
    let _ = std::fs::remove_dir_all(explicit);
}

#[test]
fn missing_base_subdirs_keep_builtins_only() {
    let base = temp_dir("base_missing");
    std::fs::create_dir_all(base.join("unrelated")).unwrap();

    let registry = ExtensionRegistry::discover_with(Some(&base), None);
    assert!(registry.targets().iter().any(|t| t.metadata.name == "linux_x86_64"));
    assert_eq!(registry.targets().len(), 1, "only the builtin target expected");
    let _ = std::fs::remove_dir_all(base);
}

// ---------- install_root() platform default + overrides ----------
//
// `install_root` reads process-global environment variables, so these tests
// run under a process-wide lock to keep env mutation deterministic while the
// rest of the suite runs in parallel.

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Snapshot the env vars that `install_root` consults, then restore them on drop.
struct EnvGuard {
    ruby_base: Option<std::ffi::OsString>,
    ruby_home: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
    xdg_data_home: Option<std::ffi::OsString>,
}

impl EnvGuard {
    fn new() -> Self {
        let g = Self {
            ruby_base: std::env::var_os("RUBYC_BASE"),
            ruby_home: std::env::var_os("RUBYC_HOME"),
            home: std::env::var_os("HOME"),
            xdg_data_home: std::env::var_os("XDG_DATA_HOME"),
        };
        g.clear();
        g
    }
    fn clear(&self) {
        unsafe {
            std::env::remove_var("RUBYC_BASE");
            std::env::remove_var("RUBYC_HOME");
            std::env::remove_var("HOME");
            std::env::remove_var("XDG_DATA_HOME");
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(v) = &self.ruby_base { std::env::set_var("RUBYC_BASE", v); }
            if let Some(v) = &self.ruby_home { std::env::set_var("RUBYC_HOME", v); }
            if let Some(v) = &self.home { std::env::set_var("HOME", v); }
            if let Some(v) = &self.xdg_data_home { std::env::set_var("XDG_DATA_HOME", v); }
        }
    }
}

#[test]
fn install_root_uses_rubyc_base_overrides() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvGuard::new();
    unsafe { std::env::set_var("RUBYC_BASE", "/opt/custom-base"); }
    assert_eq!(install_root(), std::path::PathBuf::from("/opt/custom-base"));

    unsafe {
        std::env::remove_var("RUBYC_BASE");
        std::env::set_var("RUBYC_HOME", "/opt/custom-home");
    }
    assert_eq!(install_root(), std::path::PathBuf::from("/opt/custom-home"));
}

#[test]
fn install_root_uses_xdg_data_home_when_set() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvGuard::new();
    #[cfg(target_os = "linux")]
    {
        unsafe {
            std::env::set_var("HOME", "/home/tester");
            std::env::set_var("XDG_DATA_HOME", "/home/tester/.local/share");
        }
        assert_eq!(
            install_root(),
            std::path::PathBuf::from("/home/tester/.local/share/rubyc")
        );

        // A relative XDG_DATA_HOME resolves against $HOME.
        unsafe { std::env::set_var("XDG_DATA_HOME", "data"); }
        assert_eq!(
            install_root(),
            std::path::PathBuf::from("/home/tester/data/rubyc")
        );
    }
}

#[test]
fn install_root_linux_defaults_to_local_share_rubyc() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvGuard::new();
    #[cfg(target_os = "linux")]
    {
        unsafe { std::env::set_var("HOME", "/home/tester"); }
        assert_eq!(
            install_root(),
            std::path::PathBuf::from("/home/tester/.local/share/rubyc")
        );
    }
    #[cfg(target_os = "macos")]
    {
        unsafe { std::env::set_var("HOME", "/Users/tester"); }
        assert_eq!(
            install_root(),
            std::path::PathBuf::from("/Users/tester/Library/Application Support/rubyc")
        );
    }
    #[cfg(target_os = "windows")]
    {
        unsafe { std::env::set_var("LOCALAPPDATA", r"C:\Users\tester\AppData\Local"); }
        assert_eq!(
            install_root(),
            std::path::PathBuf::from(r"C:\Users\tester\AppData\Local\rubyc")
        );
    }
}
