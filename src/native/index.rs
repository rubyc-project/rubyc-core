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
/// Fragment bytes come from each lib's image first (embedded
/// `.rubyc_types` section in `.so`, `<member>.rci` member in `.a`) and
/// fall back to its `.rci` sidecar file; libs with neither are skipped
/// (the consume path reports those per-dependency). The table regenerates
/// whenever any artifact is newer than it (mtime-guarded, so
/// steady-state builds pay stats, not full re-decodes). An empty
/// fragment set yields an empty index without writing a file.
pub fn ensure(lib_dir: &Path) -> Result<Index, String> {
    let tsv = lib_dir.join(INDEX_FILE);
    let entries = std::fs::read_dir(lib_dir).map_err(|e| format!("cannot list {}: {e}", lib_dir.display()))?;
    let mut libs: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    let mut artifacts: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("cannot list {}: {e}", lib_dir.display()))?;
        let path = entry.path();
        let is_artifact = path.extension().is_some_and(|e| {
            e == "rci" || e == "so" || e == "a"
        });
        if !is_artifact {
            continue;
        }
        artifacts.push(path.clone());
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            libs.entry(stem.to_owned()).or_default().push(path);
        }
    }
    libs.values_mut().for_each(|v| v.sort());
    if libs.is_empty() {
        return Ok(Index::default());
    }
    let tsv_fresh = tsv.is_file()
        && mtime(&tsv)
            .zip(artifacts.iter().map(|p| mtime(p)).collect::<Option<Vec<_>>>())
            .is_some_and(|(t, ms)| ms.iter().all(|m| *m <= t));
    if tsv_fresh {
        let text =
            std::fs::read_to_string(&tsv).map_err(|e| format!("cannot read {}: {e}", tsv.display()))?;
        return Ok(Index::build(&parse_rows(&text)));
    }
    let mut rows = Vec::new();
    for (lib, paths) in &libs {
        let mut frag_bytes: Option<Vec<u8>> = None;
        for path in paths {
            let Ok(image) = std::fs::read(path) else {
                continue;
            };
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            let embedded = match ext {
                "so" => super::embedded::table_bytes(&image)
                    .or_else(|| super::embedded::section_bytes(&image))
                    .map(|b| b.to_vec()),
                "a" => super::arlib::read(&image)
                    .ok()
                    .and_then(|members| {
                        super::embedded::archive_member(&members).map(|b| b.to_vec())
                    }),
                _ => None,
            };
            if embedded.is_some() {
                frag_bytes = embedded;
                break;
            }
            if ext == "rci" {
                frag_bytes = Some(image);
            }
        }
        let Some(frag_bytes) = frag_bytes else {
            continue;
        };
        let Ok(frag) = crate::native::rci::decode(&frag_bytes) else {
            continue;
        };
        if frag.lib_name != *lib {
            continue;
        }
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

/// The library artifact a `using` namespace names DIRECTLY (decision
/// A): `using X.Y;` means the artifact `X.Y` in the libraries
/// directory. The index is the fallback for assemblies installed under
/// a different name (CoreLib), not the authority for a direct hit.
pub fn direct_library(ns: &str, lib_dir: &Path) -> Option<String> {
    for candidate in [
        lib_dir.join(format!("{ns}.so")),
        lib_dir.join(format!("{ns}.a")),
        lib_dir.join("static").join(format!("{ns}.a")),
    ] {
        if candidate.is_file() {
            return Some(ns.to_owned());
        }
    }
    None
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

    /// Fragment embedded in an image with no sidecar file at all:
    /// the index must still cover the lib.
    #[test]
    fn ensure_indexes_embedded_without_sidecar() {
        use crate::native::semantic::{ClassInfo, TypeKey};
        let dir = std::env::temp_dir().join(format!(
            "rubyc-index-emb-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let class = ClassInfo {
            is_record: false,
            type_arg_index: Vec::new(),
            name: "Widget".into(),
            key: TypeKey::new("A.B.Widget", 0),
            bases: vec![],
            fields: vec![],
            static_fields: vec![],
            const_fields: vec![],
            methods: vec![],
            own_members: vec![],
            interfaces: vec![],
            properties: vec![],
            is_enum: false,
            generic_params: vec![],
            is_struct: false,
            owner: "LibE".into(),
        };
        let frag = crate::native::rci::encode("LibE", 7, &[class], &[], &[]);
        // Minimal ELF carrying the fragment as `.rubyc_types`.
        let strings = b"\0.rubyc_types\0.shstrtab\0";
        let shoff = 64usize;
        let types_off = shoff + 3 * 64;
        let str_off = types_off + frag.len();
        let mut image = vec![0u8; str_off + strings.len()];
        image[0..4].copy_from_slice(b"\x7fELF");
        image[40..48].copy_from_slice(&(shoff as u64).to_le_bytes());
        image[58..60].copy_from_slice(&64u16.to_le_bytes());
        image[60..62].copy_from_slice(&3u16.to_le_bytes());
        image[62..64].copy_from_slice(&2u16.to_le_bytes());
        let mut sh = |i: usize, name: u32, typ: u32, off: usize, size: usize| {
            let at = shoff + i * 64;
            image[at..at + 4].copy_from_slice(&name.to_le_bytes());
            image[at + 4..at + 8].copy_from_slice(&typ.to_le_bytes());
            image[at + 24..at + 32].copy_from_slice(&(off as u64).to_le_bytes());
            image[at + 32..at + 40].copy_from_slice(&(size as u64).to_le_bytes());
        };
        sh(1, 1, 1, types_off, frag.len());
        sh(2, 14, 3, str_off, strings.len());
        image[types_off..types_off + frag.len()].copy_from_slice(&frag);
        image[str_off..str_off + strings.len()].copy_from_slice(strings);
        std::fs::write(dir.join("LibE.so"), &image).unwrap();
        let index = ensure(&dir).expect("regen from embedded");
        assert_eq!(index.lookup("A.B"), vec!["LibE"]);
        assert!(dir.join(INDEX_FILE).is_file(), "tsv written");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
