//! Guards the band-aid ledger against regression.
//!
//! `documentation/compiler/band-aids.md` is the live inventory of places
//! the compiler branches on a **library type or member name** while
//! lowering. It used to claim it was machine-checked by a test that did
//! not exist, so nothing actually stopped a removed fold from coming
//! back.
//!
//! This is that check, and it is deliberately narrow. A broad scan for
//! "any uppercase string in a comparison" is useless — it flags `Length`
//! and `Add`, which are the compiler's own vocabulary as much as the
//! libraries'. What matters is the specific set of folds the campaign
//! removed: those are the ones that would silently restore a fabricated
//! value if someone reintroduced them. Each entry below names a fold and
//! the name it was keyed on.
//!
//! Two directions are checked:
//!
//! 1. A removed fold must not reappear in the lowering sources.
//! 2. The ledger must still name every removed fold, so the record and
//!    the guard cannot drift apart.

use std::path::PathBuf;

/// Folds removed by the campaign, as (keyed name, what it fabricated).
/// The keyed name is matched against comparisons and `matches!` arms in
/// the lowering sources.
const REMOVED: &[(&str, &str)] = &[
    ("ThrowIf", "ArgumentOutOfRangeException validator family"),
    ("IsSupported", "blanket hardware/feature support answered true"),
    ("IsHardwareAccelerated", "blanket hardware acceleration answered true"),
    ("GetMethodTable", "object header method-table read"),
    ("IsBitwiseEquatable", "closed-primitive bitwise-equality check"),
    ("CreateCacheTable", "GenericCache table fabricated as 0"),
    ("FastPollGC", "Thread.FastPollGC answered 0"),
    ("MemoryBarrier", "Interlocked.MemoryBarrier answered 0"),
    ("AllocateTypeAssociatedMemory", "RuntimeHelpers allocation interception"),
    ("Invariant", "GlobalizationMode.Invariant answered true"),
    ("ExceptionFallback", "encoding-fallback reads nulled out"),
    ("SetDefaultFallbacks", "encoding fallback setup neutralized"),
    (
        "IsReferenceOrContainsReferences",
        "GC layout query answered with a constant",
    ),
    (
        "SearchValues",
        "search-vector factory answered with a searcher that matches nothing",
    ),
];

/// Sources the guard covers: the shared lowering layer, which is where
/// the ledger says library-keyed folds do not belong.
const SCANNED: &[&str] = &[
    "src/native/ir/lower/eval.rs",
    "src/native/ir/lower/calls.rs",
    "src/native/ir/lower/select.rs",
    "src/native/ir/lower/types.rs",
    "src/native/ir/lower/span.rs",
    "src/native/ir/mod.rs",
    "src/native/semantic/mod.rs",
];

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn ledger_path() -> PathBuf {
    crate_root().join("../documentation/compiler/band-aids.md")
}

/// Lines that compare against a string literal `name` — the shape a
/// name-keyed fold takes (`if m.name == "ThrowIfNull"`,
/// `matches!(n, "IsSupported")`).
fn keyed_lines(src: &str, name: &str) -> Vec<usize> {
    let needle = format!("\"{name}");
    src.lines()
        .enumerate()
        .filter(|(_, l)| {
            let t = l.trim_start();
            !t.starts_with("//")
                && l.contains(&needle)
                && (l.contains("==")
                    || l.contains("!=")
                    || l.contains("matches!(")
                    || l.contains("starts_with(")
                    || l.contains("ends_with(")
                    || l.contains("contains("))
        })
        .map(|(i, _)| i + 1)
        .collect()
}

#[test]
fn removed_folds_have_not_come_back() {
    let root = crate_root();
    let mut hits: Vec<String> = Vec::new();
    let mut scanned = 0usize;
    for rel in SCANNED {
        let path = root.join(rel);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("guard scans {}: {e}", path.display()));
        scanned += 1;
        for (name, what) in REMOVED {
            for line in keyed_lines(&src, name) {
                hits.push(format!("{}:{line}: `{name}` ({what})", path.display()));
            }
        }
    }
    assert_eq!(
        scanned,
        SCANNED.len(),
        "the guard must scan every listed source"
    );
    assert!(
        hits.is_empty(),
        "a removed library-keyed fold is back in the lowering layer:\n  {}\n\
         These fabricate values for names the compiler is not supposed to know. \
         If one is genuinely needed, it belongs in the ledger with a disposition \
         and in `REMOVED`/documented intrinsically — not silently restored.",
        hits.join("\n  ")
    );
}

#[test]
fn ledger_still_names_every_removed_fold() {
    // The record and the guard must not drift: a fold missing from the
    // ledger is how the ledger stops being read.
    let path = ledger_path();
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("the ledger must exist at {}: {e}", path.display()));
    let missing: Vec<&str> = REMOVED
        .iter()
        .map(|(n, _)| *n)
        .filter(|n| !src.contains(n))
        .collect();
    assert!(
        missing.is_empty(),
        "the ledger no longer records these removed folds: {missing:?}\n\
         Add a row to {} so the inventory stays complete.",
        path.display()
    );
    // And it must not claim a machine check it does not have.
    assert!(
        !src.contains("tests/band_aid_guard.rs"),
        "the ledger points at a path that does not exist; the guard lives at \
         rubyc-core/tests/hygiene/band_aid_guard.rs"
    );
}
