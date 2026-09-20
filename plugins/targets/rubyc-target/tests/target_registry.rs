#[test]
fn builtin_includes_linux_x86_64() {
    let reg = crate::TargetRegistry::builtin();
    let names = reg.list_targets();
    assert!(names.contains(&"linux_x86_64"), "builtin registry must include linux_x86_64, got: {:?}", names);
}

#[test]
fn find_target_resolves_known_backend() {
    let reg = crate::TargetRegistry::builtin();
    let t = reg.find_target("linux_x86_64");
    assert!(t.is_some());
    assert_eq!(t.unwrap().name(), "linux_x86_64");
}

#[test]
fn find_target_returns_none_for_unknown() {
    let reg = crate::TargetRegistry::builtin();
    assert!(reg.find_target("nonexistent_target_xyz").is_none());
}

#[test]
fn backend_count_matches_list_targets_len() {
    let reg = crate::TargetRegistry::builtin();
    assert_eq!(reg.backend_count(), reg.list_targets().len());
}

#[test]
fn list_targets_is_sorted() {
    let reg = crate::TargetRegistry::builtin();
    let names = reg.list_targets();
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(names, sorted, "list_targets must return sorted names");
}

