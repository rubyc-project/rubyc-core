//! Project source globbing: recursive `**`, single-level globs, literal paths.

use crate::project::Project;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn setup() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("rubyc_glob_{}_{}", std::process::id(), n));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn make_project(root: &Path, sources: &[&str]) -> Project {
    Project {
        path: root.join("proj.rcp"),
        sources: sources.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    }
}

fn write(root: &Path, rel: &str) {
    let p = root.join(rel);
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(p, "class X { int y() { return 0; } }\n").unwrap();
}

fn names(files: Vec<PathBuf>) -> Vec<String> {
    let mut v: Vec<String> = files
        .iter()
        .map(|f| f.to_string_lossy().to_string())
        .collect();
    v.sort();
    v
}

#[test]
fn recursive_glob_matches_nested_sources() {
    let root = setup();
    write(&root, "src/a.rc");
    write(&root, "src/deep/b/c.rc");
    write(&root, "src/deep/d.rc");
    let p = make_project(&root, &["src/**/*.rc"]);
    let got = names(p.source_files().unwrap());
    assert!(got.iter().any(|f| f.ends_with("src/a.rc")), "got {got:?}");
    assert!(got.iter().any(|f| f.ends_with("src/deep/b/c.rc")), "got {got:?}");
    assert!(got.iter().any(|f| f.ends_with("src/deep/d.rc")), "got {got:?}");
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn single_level_glob_does_not_recurse() {
    let root = setup();
    write(&root, "src/a.rc");
    write(&root, "src/deep/b.rc");
    let p = make_project(&root, &["src/*.rc"]);
    let got = names(p.source_files().unwrap());
    assert!(got.iter().any(|f| f.ends_with("src/a.rc")), "got {got:?}");
    assert!(!got.iter().any(|f| f.ends_with("src/deep/b.rc")), "got {got:?}");
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn literal_file_path_is_accepted() {
    let root = setup();
    write(&root, "src/main.rc");
    let p = make_project(&root, &["src/main.rc"]);
    assert_eq!(p.source_files().unwrap().len(), 1);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn build_defines_parse() {
    // `[build] defines = [...]` seeds preprocessor symbols (like
    // csproj `DefineConstants`, e.g. `TARGET_LINUX` emitted per
    // platform by the pull script).
    let root = setup();
    let rcp = root.join("main.rcp");
    fs::write(
        &rcp,
        "[meta]\nname = \"Main\"\n[build]\nsources = [\"src/**/*.rc\"]\ndefines = [\"TARGET_LINUX\", \"TARGET_64BIT\"]\n",
    )
    .unwrap();
    let p = Project::load(&rcp).unwrap();
    assert_eq!(p.defines, vec!["TARGET_LINUX".to_string(), "TARGET_64BIT".to_string()]);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn libs_ruby_c_entries_parse() {
    // `[libs] ruby_c = [...]` names sibling `.rcp` projects whose sources
    // join the referencing project's compilation unit (cross-assembly
    // type resolution). The entries must survive the minimal TOML parse.
    let root = setup();
    let rcp = root.join("main.rcp");
    fs::write(
        &rcp,
        "[meta]\nname = \"Main\"\n[build]\nsources = [\"src/**/*.rc\"]\n[libs]\nruby_c = [\"../Dep/Dep.rcp\"]\n",
    )
    .unwrap();
    let p = Project::load(&rcp).unwrap();
    assert_eq!(p.ruby_c_libs, vec!["../Dep/Dep.rcp".to_string()]);
    assert!(p.external_libs.is_empty());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn glob_matching_nothing_is_an_error() {
    let root = setup();
    fs::create_dir_all(root.join("src")).unwrap();
    let p = make_project(&root, &["src/**/*.rc"]);
    assert!(p.source_files().is_err());
    let _ = fs::remove_dir_all(&root);
}
