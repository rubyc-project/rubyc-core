//! Plugin loader: `PluginBytes` FFI result plumbing, symbol resolution on the
//! hand-assembled-ELF roundtrip target, and discovery filtering.
//!
//! `load_target_plugin`/`discover_*` take a directory
//! and call `Plugin::load_path` (which tries `dlopen` first). `dlopen` on the
//! hand-assembled roundtrip ELF crashes (corrupt section headers), so the
//! directory-based paths are only exercised for the *empty / no-manifest*
//! cases (which never reach `dlopen`). The FFI `asm!` call sites in
//! `TargetPlugin::lower`/`write_image` are not exercised here because invoking
//! them on the roundtrip ELF segfaults; symbol *resolution* is the part under
//! test.

use crate::plugin::{
    discover_targets, load_target_plugin, Plugin, PluginBytes, PluginError, TargetPlugin,
};
use crate::plugin::roundtrip::build_roundtrip_plugin;

// ---------- PluginBytes ----------

#[test]
fn pluginbytes_empty() {
    let b = PluginBytes::empty();
    assert_eq!(b.ok, 1);
    assert_eq!(b.len, 0);
    assert_eq!(b.cap, 0);
    assert!(b.data.is_null());
}

#[test]
fn pluginbytes_success() {
    let data = vec![1u8, 2, 3, 4];
    let b = PluginBytes::success(data);
    assert_eq!(b.ok, 1);
    assert_eq!(b.len, 4);
    assert_eq!(b.cap, 4);
    assert!(!b.data.is_null());
}

#[test]
fn pluginbytes_error_carries_nul_message() {
    let b = PluginBytes::error("boom".to_string());
    assert_eq!(b.ok, 0);
    // len includes the trailing NUL.
    assert_eq!(b.len, 5);
}

#[test]
fn pluginbytes_release_safe_on_null() {
    let b = PluginBytes::empty();
    unsafe { b.release() };
}

#[test]
fn pluginbytes_clone_or_zero_null() {
    let b = PluginBytes::empty();
    let c = b.clone_or_zero();
    assert_eq!(c.len, 0);
    assert!(c.data.is_null());
}

#[test]
fn pluginbytes_clone_or_zero_copies() {
    let b = PluginBytes::success(vec![9u8, 8, 7]);
    let c = b.clone_or_zero();
    assert_eq!(c.len, 3);
    assert_eq!(c.ok, 1);
}

// ---------- Plugin (roundtrip ELF via load_bytes) ----------

#[test]
fn roundtrip_elf_loads_and_resolves_symbols() {
    let bytes = build_roundtrip_plugin();
    let plugin = Plugin::load_bytes(&bytes).unwrap();
    assert_ne!(plugin.base(), 0);
    // The roundtrip ELF exports these (per roundtrip.rs):
    for sym in [
        "rbxt_name",
        "rbxt_triple",
        "rbxt_base_addr",
        "rbxt_lower",
        "rbxt_write_image",
    ] {
        assert!(plugin.sym(sym).is_some(), "missing symbol {sym}");
    }
    assert!(plugin.sym("nope").is_none());
}

#[test]
fn require_sym_missing_is_error() {
    let bytes = build_roundtrip_plugin();
    let plugin = Plugin::load_bytes(&bytes).unwrap();
    let err = match plugin.require_sym("definitely_not_there") {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    };
    match err {
        PluginError::MissingSymbol(name) => assert_eq!(name, "definitely_not_there"),
        other => panic!("expected MissingSymbol, got {other:?}"),
    }
}

#[test]
fn symbol_names_iterates() {
    let bytes = build_roundtrip_plugin();
    let plugin = Plugin::load_bytes(&bytes).unwrap();
    let names: Vec<&str> = plugin.symbol_names().collect();
    assert!(names.contains(&"rbxt_name"));
    assert!(names.contains(&"rbxt_lower"));
}

#[test]
fn symbol_address_is_within_loaded_region() {
    let bytes = build_roundtrip_plugin();
    let plugin = Plugin::load_bytes(&bytes).unwrap();
    let base = plugin.base();
    let sym = plugin.require_sym("rbxt_lower").unwrap() as u64;
    // The symbol address is base + offset into the first load segment.
    assert!(sym >= base, "symbol {sym:#x} < base {base:#x}");
    // The roundtrip ELF is one 0x1000-byte page; the symbol must be within it.
    assert!(
        sym - base < 0x1000,
        "symbol offset {:#x} outside page",
        sym - base
    );
}

// ---------- helpers ----------

fn temp_dir(label: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rubyc_loader_{}_{}",
        label,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Build a well-formed `TargetPlugin` from the roundtrip ELF (mmap-loaded).
/// We do NOT call its FFI methods here (they segfault on the hand-assembled
/// ELF); we only verify the struct is populated with non-null function
/// pointers and correct metadata.
fn make_target_plugin() -> TargetPlugin {
    let bytes = build_roundtrip_plugin();
    let plugin = Plugin::load_bytes(&bytes).unwrap();
    let plugin = std::sync::Arc::new(plugin);
    let lower = plugin.require_sym("rbxt_lower").unwrap() as *const ();
    let write_image = plugin.require_sym("rbxt_write_image").unwrap() as *const ();
    TargetPlugin {
        name: "roundtrip".into(),
        triple: "roundtrip-bare".into(),
        metadata: crate::extensions::ExtensionMetadata {
            name: "roundtrip".into(),
            display_name: "roundtrip".into(),
            author: String::new(),
            version: "1.0.0".into(),
            description: String::new(),
            license: "MIT".into(),
            homepage: String::new(),
            repository: String::new(),
        },
        plugin,
        lower,
        write_image,
        write_shared: write_image,
        run: lower,
    }
}

// ---------- TargetPlugin struct shape ----------

#[test]
fn target_plugin_shape() {
    let plugin = make_target_plugin();
    assert_eq!(plugin.name, "roundtrip");
    assert_eq!(plugin.triple, "roundtrip-bare");
    assert_eq!(plugin.metadata.name, "roundtrip");
    assert_eq!(plugin.metadata.version, "1.0.0");
    // All four ABI function pointers are non-null.
    assert!(!plugin.lower.is_null());
    assert!(!plugin.write_image.is_null());
    assert!(!plugin.write_shared.is_null());
    assert!(!plugin.run.is_null());
}

// ---------- load_target_plugin (directory path; safe cases only) ----------

#[test]
fn load_target_plugin_missing_dir_is_error() {
    let dir = temp_dir("nope");
    // No manifest, no .so -> manifest read fails (never reaches dlopen).
    let err = match load_target_plugin(&dir) {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    };
    assert!(!err.is_empty());
}

// ---------- discover_targets (safe empty cases) ----------

#[test]
fn discover_targets_empty_when_no_dir() {
    let base = temp_dir("notargets");
    let targets = discover_targets(&base);
    assert!(targets.is_empty());
}

#[test]
fn discover_targets_skips_directories_without_manifest() {
    let base = temp_dir("skip");
    let tdir = base.join("targets").join("empty");
    std::fs::create_dir_all(&tdir).unwrap();
    // No manifest -> load_target_plugin errors -> filtered out. No dlopen.
    let targets = discover_targets(&base);
    assert!(targets.is_empty());
}
