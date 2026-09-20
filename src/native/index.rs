//! `using`-driven dependency discovery over installed libraries.
//!
//! Explicit `ruby_c` entries name dep projects by path, but a program
//! that only says `using System.Text;` names no project at all. The
//! index maps every installed library's fragment entries (type paths
//! from its `.rci`) back to the owning lib, so the driver can auto-link
//! the single library owning a `using` namespace and fail loudly when
//! two libraries claim it.
//!
//! Layout: `libraries/index.tsv`, regenerated from `libraries/*.rci`
//! whenever any `.rci` is newer than it (mtime-guarded, so steady-state
//! builds pay one stat per fragment, not a full re-decode). Rows are
//! `type_path\tkind\tlib`, sorted for stable diffs; `kind` is one of
//! `class`, `interface`, `delegate`. A lookup matches a namespace
//! exactly or as a parent (`System.Text` owns `System.Text.Decoder`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Regenerated index file name inside the libraries directory.
pub const INDEX_FILE: &str = "index.tsv";

/// One decoded row: a fragment type and the lib that owns it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub path: String,
    pub kind: &'static str,
    pub lib: String,
}

/// Namespace → owning libraries (every ancestor namespace of every row
/// maps to its lib, so `using A.B;` finds lib owning `A.B.C.D`).
#[derive(Debug, Default)]
pub struct Index {
    owners: BTreeMap<String, BTreeSet<String>>,
}

impl Index {
    /// Build from decoded rows.
    pub fn build(rows: &[Row]) -> Self {
        let mut owners: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for row in rows {
            // The full path and every ancestor namespace resolve here.
            let mut rest = row.path.as_str();
            loop {
                owners
                    .entry(rest.to_owned())
                    .or_default()
                    .insert(row.lib.clone());
                match rest.rfind('.') {
                    Some(i) => rest = &rest[..i],
                    None => break,
                }
            }
        }
        Index { owners }
    }

    /// Libraries owning `namespace` (exact or ancestor match), sorted.
    pub fn lookup(&self, namespace: &str) -> Vec<String> {
        self.owners
            .get(namespace)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }
}

/// Load or regenerate the index for `lib_dir` (`.../libraries`).
/// Missing `.rci` files decode-skipped (the consume path reports those
/// per-dependency); an empty fragment set yields an empty index without
/// writing a file.
pub fn ensure(lib_dir: &Path) -> Result<Index, String> {
    let tsv = lib_dir.join(INDEX_FILE);
    let mut rcis: Vec<PathBuf> = Vec::new();
    let entries = std::fs::read_dir(lib_dir).map_err(|e| format!("cannot list {}: {e}", lib_dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("cannot list {}: {e}", lib_dir.display()))?;
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "rci") {
            rcis.push(path);
        }
    }
    rcis.sort();
    if rcis.is_empty() {
        return Ok(Index::default());
    }
    let tsv_fresh = tsv.is_file()
        && mtime(&tsv)
            .zip(rcis.iter().map(|p| mtime(p)).collect::<Option<Vec<_>>>())
            .is_some_and(|(t, ms)| ms.iter().all(|m| *m <= t));
    if tsv_fresh {
        let text =
            std::fs::read_to_string(&tsv).map_err(|e| format!("cannot read {}: {e}", tsv.display()))?;
        return Ok(Index::build(&parse_rows(&text)));
    }
    let mut rows = Vec::new();
    for path in &rcis {
        let Ok(bytes) = std::fs::read(path) else { continue };
        let Ok(frag) = crate::native::rci::decode(&bytes) else {
            continue;
        };
        for c in &frag.classes {
            rows.push(Row {
                path: c.key.path.clone(),
                kind: "class",
                lib: frag.lib_name.clone(),
            });
        }
        for i in &frag.interfaces {
            rows.push(Row {
                path: i.name.clone(),
                kind: "interface",
                lib: frag.lib_name.clone(),
            });
        }
        for d in &frag.delegates {
            rows.push(Row {
                path: d.name.clone(),
                kind: "delegate",
                lib: frag.lib_name.clone(),
            });
        }
    }
    rows.sort_by(|a, b| (&a.path, a.kind, &a.lib).cmp(&(&b.path, b.kind, &b.lib)));
    rows.dedup();
    let mut text = String::new();
    for row in &rows {
        text.push_str(&row.path);
        text.push('\t');
        text.push_str(row.kind);
        text.push('\t');
        text.push_str(&row.lib);
        text.push('\n');
    }
    // Atomic write (temp + rename): parallel backfill builds share one
    // index file, and a torn write would poison every later lookup. A
    // failed write never fails the build (the in-memory index is
    // already built) — it only means the next build re-decodes.
    let tmp = lib_dir.join(format!("{}.tmp-{}", INDEX_FILE, std::process::id()));
    if std::fs::write(&tmp, &text).is_err() || std::fs::rename(&tmp, &tsv).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return Ok(Index::build(&rows));
    }
    Ok(Index::build(&parse_rows(&text)))
}

fn mtime(path: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

fn parse_rows(text: &str) -> Vec<Row> {
    let mut rows = Vec::new();
    for line in text.lines() {
        if line.is_empty() {
            continue;
        }
        let mut cols = line.split('\t');
        let (Some(path), Some(kind), Some(lib)) = (cols.next(), cols.next(), cols.next()) else {
            continue;
        };
        let kind = match kind {
            "class" => "class",
            "interface" => "interface",
            "delegate" => "delegate",
            _ => continue,
        };
        rows.push(Row {
            path: path.to_owned(),
            kind,
            lib: lib.to_owned(),
        });
    }
    rows
}

/// Lexically collect `using X;` namespaces from source text: plain
/// namespace imports only (`using Alias =` and `using static` carry
/// no namespace to discover and are skipped).
pub fn usings_in_source(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("using ") else {
            continue;
        };
        let rest = rest.trim();
        if rest.starts_with("static ") || rest.contains('=') {
            continue;
        }
        let Some(semi) = rest.find(';') else { continue };
        let ns = rest[..semi].trim();
        if !ns.is_empty() && ns.chars().all(|c| c.is_alphanumeric() || c == '.' || c == '_') {
            out.push(ns.to_owned());
        }
    }
    out.sort();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(path: &str, lib: &str) -> Row {
        Row {
            path: path.to_owned(),
            kind: "class",
            lib: lib.to_owned(),
        }
    }

    #[test]
    fn lookup_matches_exact_and_ancestors() {
        let index = Index::build(&[
            row("System.Text.Decoder", "CoreLib"),
            row("System.Text.Encoder", "CoreLib"),
            row("System.IO.File", "IO"),
        ]);
        assert_eq!(index.lookup("System.Text.Decoder"), vec!["CoreLib"]);
        assert_eq!(index.lookup("System.Text"), vec!["CoreLib"]);
        assert_eq!(index.lookup("System"), vec!["CoreLib", "IO"]);
        assert!(index.lookup("System.Net").is_empty());
    }

    #[test]
    fn lookup_multi_owner_sorted() {
        let index = Index::build(&[row("X.Y.Z", "B"), row("X.Y.W", "A")]);
        assert_eq!(index.lookup("X.Y"), vec!["A", "B"]);
    }

    #[test]
    fn usings_scan_skips_aliases_and_statics() {
        let src = "using System.Text;\nusing IO = System.IO;\nusing static Math;\nnamespace A { }\n";
        assert_eq!(usings_in_source(src), vec!["System.Text"]);
    }

    #[test]
    fn ensure_roundtrips_through_tsv() {
        let dir = std::env::temp_dir().join(format!(
            "rubyc-index-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // No .rci files: empty index, no file written.
        assert!(ensure(&dir).expect("empty dir").lookup("A.B").is_empty());
        assert!(!dir.join(INDEX_FILE).is_file());
        // A decodable (empty) fragment regenerates the index file.
        let bytes = crate::native::rci::encode("LibA", 1, &[], &[], &[]);
        std::fs::write(dir.join("LibA.rci"), &bytes).unwrap();
        let index = ensure(&dir).expect("regen");
        assert!(index.lookup("A.B").is_empty(), "no rows, no owners");
        assert!(dir.join(INDEX_FILE).is_file(), "tsv written");
        // Second call loads the file instead of re-decoding.
        let index = ensure(&dir).expect("tsv load");
        assert!(index.lookup("A.B").is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
