//! Driver closure: shared sources under divergent per-project defines.
//!
//! A byte-identical shared file staged by two dependencies with
//! different `[build] defines` contributes BOTH views: each copy
//! preprocesses under its own owner's symbols (negated gates like
//! `#if !SYSTEM_PRIVATE_CORELIB` need their owner's exact view —
//! cf. Metadata's public `AssemblyNameInfo` ctor vs CoreLib's linked
//! copy of the same file), the (content, defines) dedup loads each
//! distinct view once, and same-key entries member-union (cf.
//! `HexConverter.AsciiToHexVector128`, visible only under CoreLib's
//! `SYSTEM_PRIVATE_CORELIB`).

use crate::cli::{Command, ErrorFormat, IoArgs};
use crate::driver::load_project_unit_with_deps;
use crate::models::Member;

/// Serializes the tests that point the install root at a temp dir
/// (`RUBYC_BASE` is process-global; parallel tests would clobber it).
static INSTALL_ROOT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn invocation() -> crate::cli::Invocation {
    crate::cli::Invocation {
        command: Command::Build,
        io: IoArgs {
            skip_pre_hook: true,
            skip_post_hook: true,
            ..Default::default()
        },
        verbose: false,
        quiet: true,
        error_format: ErrorFormat::Text,
        no_color: true,
        lint_overrides: Vec::new(),
        defines: Vec::new(),
    }
}

fn write(path: &std::path::Path, text: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, text).unwrap();
}

fn rcp(dir: &std::path::Path, name: &str, defines: &[&str], deps: &[&str]) {
    let defs = defines
        .iter()
        .map(|d| format!("\"{d}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let libs = deps
        .iter()
        .map(|d| format!("\"../{d}/{d}.rcp\""))
        .collect::<Vec<_>>()
        .join(", ");
    write(
        &dir.join(format!("{name}.rcp")),
        &format!(
            "[meta]\nname = \"{name}\"\nversion = \"1\"\ndescription = \"probe\"\n\n\
             [target]\nplatform = \"linux_x86_64\"\noutput = \"shared\"\n\n\
             [build]\nsources = [\"src/**/*.rc\"]\nentry-point = \"\"\ndefines = [{defs}]\n\n\
             [libs]\nruby_c = [{libs}]\n"
        ),
    );
}

#[test]
fn shared_file_divergent_defines_union() {
    // dep_full stages Shared.rc with FULL defined (extra member
    // compiled in); dep_narrow stages the IDENTICAL bytes without it.
    // The main project references both: the merged unit must carry
    // the gated member AND the common one, exactly once each.
    let root = std::env::temp_dir().join(format!(
        "rubyc-closure-defines-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let shared = "namespace Sys {\n    public static class Help {\n        public static int Common() { return 0; }\n#if FULL\n        public static int Gated() { return 1; }\n#endif\n    }\n}\n";
    for dep in ["dep_full", "dep_narrow"] {
        write(&root.join(dep).join("src/Shared.rc"), shared);
    }
    rcp(&root.join("dep_full"), "dep_full", &["FULL"], &[]);
    rcp(&root.join("dep_narrow"), "dep_narrow", &[], &[]);
    write(
        &root.join("main/src/main.rc"),
        "namespace t {\nclass P {\nint main() { return Sys.Help.Gated() + Sys.Help.Common(); }\n}\n}\n",
    );
    rcp(&root.join("main"), "main", &[], &["dep_full", "dep_narrow"]);

    let inv = invocation();
    let (unit, _, _, _) = load_project_unit_with_deps(&root.join("main/main.rcp"), &inv, false)
        .expect("closure must load");
    let ns = unit
        .namespaces
        .iter()
        .find(|n| n.name == "Sys")
        .expect("Sys namespace present");
    let entry = ns
        .members
        .iter()
        .find(|m| m.name() == "Help")
        .expect("single Help entry");
    let Member::Class(c) = entry else {
        panic!("Help must stay a class");
    };
    let mut names: Vec<&str> = c
        .members
        .iter()
        .filter_map(|m| match m {
            crate::models::ClassMember::Method(md) => Some(md.name.as_str()),
            _ => None,
        })
        .collect();
    names.sort();
    assert_eq!(
        names,
        vec!["Common", "Gated"],
        "union must carry both views exactly once: {names:?}"
    );
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn bare_name_dep_resolves_via_libraries_env() {
    // `ruby_c = ["dep_bare"]` (no path) resolves through
    // RUBYC_LIBRARIES: `<root>/<Name>/<Name>.rcp`. The main project
    // lives OUTSIDE the library tree (like a user project), so the
    // sibling fallback cannot save it — only the env search path.
    let root = std::env::temp_dir().join(format!(
        "rubyc-closure-barename-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let libroot = root.join("libs");
    write(
        &libroot.join("dep_bare/src/Bare.rc"),
        "namespace Sys {\n    public static class Bare {\n        public static int Value() { return 7; }\n    }\n}\n",
    );
    rcp(&libroot.join("dep_bare"), "dep_bare", &[], &[]);
    let maindir = root.join("userproj");
    write(
        &maindir.join("src/main.rc"),
        "namespace t {\nclass P {\nint main() { return Sys.Bare.Value(); }\n}\n}\n",
    );
    // Bare name on purpose: no `../` path.
    write(
        &maindir.join("main.rcp"),
        "[meta]\nname = \"main\"\nversion = \"1\"\ndescription = \"probe\"\n\n\
         [target]\nplatform = \"linux_x86_64\"\noutput = \"exe\"\n\n\
         [build]\nsources = [\"src/**/*.rc\"]\nentry-point = \"\"\ndefines = []\n\n\
         [libs]\nruby_c = [\"dep_bare\"]\n",
    );
    let prev = std::env::var_os("RUBYC_LIBRARIES");
    // SAFETY: single-threaded test process; no other thread reads env.
    unsafe {
        std::env::set_var("RUBYC_LIBRARIES", &libroot);
    }
    let result = load_project_unit_with_deps(&maindir.join("main.rcp"), &invocation(), false);
    unsafe {
        match prev {
            Some(v) => std::env::set_var("RUBYC_LIBRARIES", v),
            None => std::env::remove_var("RUBYC_LIBRARIES"),
        }
    }
    let (unit, _, _, _) = result.expect("bare-name dep must resolve via RUBYC_LIBRARIES");
    assert!(
        unit.namespaces.iter().any(|n| n.name == "Sys"),
        "dep namespace Sys must merge into the unit"
    );
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn type_owners_attribute_per_file_lib() {
    // Dynamic-linking ownership: every top-level member key in the
    // returned map attributes to the library whose file declared it
    // (main project first, so merged main+dep entries attribute to
    // main). Nested members are absent from the map (they inherit
    // through the parent chain in `analyze_with_owners`).
    let root = std::env::temp_dir().join(format!(
        "rubyc-owners-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    write(
        &root.join("dep_a/src/a.rc"),
        "namespace da {\nclass A {\nclass Inner { }\n}\n}\n",
    );
    rcp(&root.join("dep_a"), "dep_a", &[], &[]);
    write(
        &root.join("dep_b/src/b.rc"),
        "namespace db {\nclass B { }\n}\n",
    );
    rcp(&root.join("dep_b"), "dep_b", &[], &[]);
    write(
        &root.join("main/src/main.rc"),
        "namespace t {\nclass P {\nint main() { return 0; }\n}\n}\n",
    );
    rcp(&root.join("main"), "main", &[], &["dep_a", "dep_b"]);

    let inv = invocation();
    let (unit, _, owners, _) = load_project_unit_with_deps(&root.join("main/main.rcp"), &inv, false)
        .expect("closure must load");
    let _ = unit;
    use crate::native::semantic::TypeKey;
    assert_eq!(
        owners.get(&TypeKey::new("t.P", 0)).map(String::as_str),
        Some("main"),
        "main type attributes to main"
    );
    assert_eq!(
        owners.get(&TypeKey::new("da.A", 0)).map(String::as_str),
        Some("dep_a"),
        "dep type attributes to its lib"
    );
    assert_eq!(
        owners.get(&TypeKey::new("db.B", 0)).map(String::as_str),
        Some("dep_b"),
        "second dep attributes to its lib"
    );
    assert!(
        !owners.keys().any(|k| k.path == "da.A.Inner"),
        "nested members inherit downstream, not in the map"
    );
}

#[test]
fn consume_loads_fragment_and_skips_dep_sources() {
    let _guard = INSTALL_ROOT_LOCK.lock().unwrap();
    // Dynamic-linking consume (`[build] dynamic = true`): a dep with a
    // fresh installed `.rci` (+ `.so`) loads as a fragment — its
    // SOURCES never enter the unit — while its entries resolve
    // through preload exactly as if compiled.
    let root = std::env::temp_dir().join(format!(
        "rubyc-consume-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    write(
        &root.join("dep_a/src/a.rc"),
        "namespace da {\nclass A {\npublic int Base = 3;\npublic virtual int M() { return 1; }\n}\n}\n",
    );
    rcp(&root.join("dep_a"), "dep_a", &[], &[]);
    write(
        &root.join("main/src/main.rc"),
        "namespace t {\nclass P : da.A {\npublic int G() { return M() + Base; }\n}\n}\n",
    );
    // main project with dynamic opt-in (rcp() has no flag; append it).
    rcp(&root.join("main"), "main", &[], &["dep_a"]);
    {
        use std::fmt::Write as _;
        let mut text =
            std::fs::read_to_string(root.join("main/main.rcp")).unwrap();
        write!(text, "\n[build]\ndynamic = true\n").unwrap();
        // NOTE: duplicate [build] section — parser takes ... verify below.
        std::fs::write(root.join("main/main.rcp"), text).unwrap();
    }

    // Build a REAL fragment the way a dep build would: load dep
    // sources (no consume), analyze, encode with the true hash.
    let inv = invocation();
    let dep_proj =
        crate::project::Project::load(&root.join("dep_a/dep_a.rcp")).unwrap();
    let (dep_unit, _, dep_owners, _) =
        load_project_unit_with_deps(&root.join("dep_a/dep_a.rcp"), &inv, false)
            .expect("dep must load");
    let dep_table =
        crate::native::semantic::analyze_with_owners(&dep_unit, &dep_owners).unwrap();
    let hash = crate::driver::rci_source_hash(&dep_proj);
    let bytes = crate::native::rci::encode("dep_a", hash, &dep_table.classes, &[], &[]);
    let dyn_dir = root.join("dyn");
    std::fs::create_dir_all(&dyn_dir).unwrap();
    std::fs::write(dyn_dir.join("dep_a.rci"), &bytes).unwrap();
    std::fs::write(dyn_dir.join("dep_a.so"), b"fake-so").unwrap();

    // Point the install root at the temp dir (single consume test;
    // no other test reads install_root concurrently).
    unsafe { std::env::set_var("RUBYC_BASE", &dyn_dir.parent().unwrap().join("root")) };
    // NOTE: install layout is <root>/libraries — stage there.
    let staged = dyn_dir.parent().unwrap().join("root/libraries");
    std::fs::create_dir_all(&staged).unwrap();
    std::fs::rename(dyn_dir.join("dep_a.rci"), staged.join("dep_a.rci")).unwrap();
    std::fs::rename(dyn_dir.join("dep_a.so"), staged.join("dep_a.so")).unwrap();
    let (unit, _, owners, fragments) =
        load_project_unit_with_deps(&root.join("main/main.rcp"), &inv, true)
            .expect("main must load");
    unsafe { std::env::remove_var("RUBYC_BASE") };

    assert_eq!(fragments.len(), 1, "dep fragment must load");
    assert_eq!(fragments[0].lib_name, "dep_a");
    // Dep sources skipped: no AST class A in the unit ...
    let has_a = unit.all_members().iter().any(|m| match m {
        crate::models::Member::Class(c) => c.name == "A",
        _ => false,
    });
    assert!(!has_a, "consumed dep sources must not enter the unit");
    // ... and no dep keys in the owners map (fragments carry owners).
    assert!(
        !owners.keys().any(|k| k.path.starts_with("da.")),
        "consumed dep must not claim owners"
    );
    // Full circle: preload reproduces the dep entry with its owner,
    // and the derived class inherits through the boundary.
    let pre = crate::native::semantic::Preloaded {
        classes: fragments[0].classes.clone(),
        interfaces: fragments[0].interfaces.clone(),
        delegates: fragments[0].delegates.clone(),
    };
    let table =
        crate::native::semantic::analyze_with_preloaded(&unit, &owners, &pre)
            .expect("preloaded analyze must succeed");
    let a = table
        .classes
        .iter()
        .find(|c| c.key.path == "da.A")
        .expect("fragment entry present");
    assert_eq!(a.owner, "dep_a");
    let p = table
        .classes
        .iter()
        .find(|c| c.key.path == "t.P")
        .expect("main entry present");
    assert!(
        p.methods.iter().any(|m| m.name == "M"),
        "inherited dep method must be visible"
    );
    assert!(
        p.fields.iter().any(|f| f.name == "Base"),
        "inherited dep field must be visible"
    );
}

#[test]
fn using_discovery_auto_links_and_rejects_ambiguity() {
    let _guard = INSTALL_ROOT_LOCK.lock().unwrap();
    // `using`-driven discovery (dynamic consume): a `using` namespace
    // owned by exactly one installed library auto-links its fragment
    // with no `ruby_c` entry; two owners fail loudly naming both.
    let root = std::env::temp_dir().join(format!(
        "rubyc-disco-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let stage = |lib: &str, ns: &str, class: &str| {
        write(
            &root.join(format!("{lib}/src/x.rc")),
            &format!("namespace {ns} {{\nclass {class} {{\npublic int M() {{ return 1; }}\n}}\n}}\n"),
        );
        rcp(&root.join(lib), lib, &[], &[]);
        let inv = invocation();
        let proj = crate::project::Project::load(&root.join(format!("{lib}/{lib}.rcp"))).unwrap();
        let (unit, _, owners, _) =
            load_project_unit_with_deps(&root.join(format!("{lib}/{lib}.rcp")), &inv, false)
                .expect("dep must load");
        let table =
            crate::native::semantic::analyze_with_owners(&unit, &owners).unwrap();
        let hash = crate::driver::rci_source_hash(&proj);
        let bytes = crate::native::rci::encode(lib, hash, &table.classes, &[], &[]);
        let staged = root.join("root/libraries");
        std::fs::create_dir_all(&staged).unwrap();
        std::fs::write(staged.join(format!("{lib}.rci")), &bytes).unwrap();
        std::fs::write(staged.join(format!("{lib}.so")), b"fake-so").unwrap();
    };
    stage("disco", "dd", "D");
    // Main project: dynamic opt-in, NO ruby_c deps, `using dd;`.
    write(
        &root.join("main/src/main.rc"),
        "using dd;\nnamespace t {\nclass P {\npublic int G() { return 1; }\n}\n}\n",
    );
    rcp(&root.join("main"), "main", &[], &[]);
    {
        use std::fmt::Write as _;
        let mut text = std::fs::read_to_string(root.join("main/main.rcp")).unwrap();
        write!(text, "\n[build]\ndynamic = true\n").unwrap();
        std::fs::write(root.join("main/main.rcp"), text).unwrap();
    }
    unsafe { std::env::set_var("RUBYC_BASE", &root.join("root")) };
    let inv = invocation();
    let (_, _, _, fragments) =
        load_project_unit_with_deps(&root.join("main/main.rcp"), &inv, true)
            .expect("single owner auto-links");
    assert_eq!(fragments.len(), 1, "discovered fragment must load");
    assert_eq!(fragments[0].lib_name, "disco");
    // A second owner of the same namespace turns the using ambiguous.
    stage("disco2", "dd", "E");
    let err = match load_project_unit_with_deps(&root.join("main/main.rcp"), &inv, true) {
        Ok(_) => panic!("ambiguous using must fail"),
        Err(e) => format!("{e:?}"),
    };
    unsafe { std::env::remove_var("RUBYC_BASE") };
    assert!(err.contains("disco"), "names first owner: {err}");
    assert!(err.contains("disco2"), "names second owner: {err}");
    assert!(err.contains("dd"), "names the namespace: {err}");
    std::fs::remove_dir_all(&root).ok();
}
