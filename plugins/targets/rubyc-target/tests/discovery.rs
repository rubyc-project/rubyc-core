use crate::discovery;

#[test]
fn load_builtin_linux_x86_64() {
    let b = discovery::load_builtin("linux_x86_64");
    assert!(b.is_some());
    let b = b.unwrap();
    assert_eq!(b.name(), "linux_x86_64");
    assert_eq!(b.triple(), "x86_64-unknown-linux-gnu");
}

#[test]
fn load_builtin_unknown_returns_none() {
    assert!(discovery::load_builtin("unknown_target_xyz").is_none());
}

#[test]
fn default_targets_dir_returns_path() {
    let dir = discovery::default_targets_dir();
    assert!(!dir.as_os_str().is_empty());
}

#[test]
#[allow(unsafe_code)]
fn env_var_overrides_targets_dir() {
    let original = std::env::var("RUBYC_TARGETS_DIR").ok();
    // SAFETY: this test restores the variable before returning.
    unsafe { std::env::set_var("RUBYC_TARGETS_DIR", "/tmp/rubyc_test_targets") };
    let dir = discovery::default_targets_dir();
    assert_eq!(dir, std::path::PathBuf::from("/tmp/rubyc_test_targets"));
    match original {
        Some(v) => unsafe { std::env::set_var("RUBYC_TARGETS_DIR", v) },
        None => unsafe { std::env::remove_var("RUBYC_TARGETS_DIR") },
    }
}
