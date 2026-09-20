//! Compiler extension metadata and installed target discovery.
//!
//! Extensions are described by small TOML manifests. Discovery is deliberately
//! data-only: loading executable third-party backend code requires a stable ABI
//! and is a separate concern from making installed extensions visible to tools.

use std::path::{Path, PathBuf};

/// Standard package metadata shared by target extensions.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ExtensionMetadata {
    pub name: String,
    pub display_name: String,
    pub author: String,
    pub version: String,
    pub description: String,
    pub license: String,
    pub homepage: String,
    pub repository: String,
}

impl ExtensionMetadata {
    fn normalize(mut self) -> Self {
        if self.display_name.is_empty() {
            self.display_name = self.name.clone();
        }
        self
    }
}

/// Polymorphic metadata surface implemented by compilation targets.
pub trait TargetExtension: Send + Sync {
    fn metadata(&self) -> &ExtensionMetadata;
    fn triple(&self) -> &str;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetInfo {
    pub metadata: ExtensionMetadata,
    pub triple: String,
    pub manifest: Option<PathBuf>,
}

#[derive(Debug, Clone, Default)]
pub struct ExtensionRegistry {
    targets: Vec<TargetInfo>,
}

impl ExtensionRegistry {
    pub fn builtin() -> Self {
        let version = env!("CARGO_PKG_VERSION").to_owned();
        Self {
            targets: vec![TargetInfo {
                metadata: ExtensionMetadata {
                    name: "linux_x86_64".into(),
                    display_name: "Linux x86-64".into(),
                    author: "RubyC contributors".into(),
                    version,
                    description: "Static ELF and in-process JIT backend".into(),
                    license: "MIT".into(),
                    ..Default::default()
                },
                triple: "x86_64-unknown-linux-gnu".into(),
                manifest: None,
            }],
        }
    }

    /// Discover targets from an explicit directory, or the platform default
    /// when no override is supplied.
    pub fn discover(targets_dir: Option<&Path>) -> Self {
        Self::discover_with(None, targets_dir)
    }

    /// Discovery with a base directory: `base/targets` is scanned unless a
    /// per-kind directory is supplied explicitly.
    /// Precedence: explicit per-kind dir > base dir > platform default
    /// (`RUBYC_TARGETS_DIR` / `RUBYC_BASE` / `RUBYC_HOME`).
    pub fn discover_with(base: Option<&Path>, targets_dir: Option<&Path>) -> Self {
        let mut registry = Self::builtin();
        let targets = targets_dir
            .map(Path::to_path_buf)
            .or_else(|| base.map(|b| b.join("targets")))
            .unwrap_or_else(default_targets_dir);
        registry.scan_targets(targets);
        registry
    }

    pub fn targets(&self) -> &[TargetInfo] { &self.targets }

    /// An empty registry, used by the plugin loader before discovery fills it.
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn push_target(&mut self, info: TargetInfo) {
        upsert(&mut self.targets, info, |v| &v.metadata.name);
    }

    fn scan_targets(&mut self, dir: PathBuf) {
        for path in manifests(&dir) {
            if let Ok(text) = std::fs::read_to_string(&path) {
                let values = parse_manifest(&text);
                if section(&values) != Some("target") { continue; }
                let metadata = metadata(&values);
                if metadata.name.is_empty() { continue; }
                let info = TargetInfo {
                    triple: value(&values, "triple").to_owned(),
                    metadata,
                    manifest: Some(path),
                };
                upsert(&mut self.targets, info, |v| &v.metadata.name);
            }
        }
        self.targets.sort_by(|a, b| a.metadata.name.cmp(&b.metadata.name));
    }
}

fn upsert<T>(items: &mut Vec<T>, item: T, name: impl Fn(&T) -> &str) {
    let item_name = name(&item).to_owned();
    if let Some(index) = items.iter().position(|v| name(v) == item_name) {
        items[index] = item;
    } else {
        items.push(item);
    }
}

pub fn default_targets_dir() -> PathBuf {
    std::env::var_os("RUBYC_TARGETS_DIR").map(PathBuf::from).unwrap_or_else(|| install_root().join("targets"))
}

pub fn install_root() -> PathBuf {
    if let Some(path) = std::env::var_os("RUBYC_BASE") { return path.into(); }
    if let Some(path) = std::env::var_os("RUBYC_HOME") { return path.into(); }
    #[cfg(target_os = "windows")]
    if let Some(path) = std::env::var_os("LOCALAPPDATA") {
        return PathBuf::from(path).join("rubyc");
    }
    #[cfg(target_os = "macos")]
    if let Some(path) = std::env::var_os("HOME") {
        return PathBuf::from(path).join("Library/Application Support/rubyc");
    }
    #[cfg(target_os = "linux")]
    {
        // XDG data dir: $XDG_DATA_HOME (if set, else ~/.local/share)/rubyc.
        if let Some(xdg) = std::env::var_os("XDG_DATA_HOME") {
            let xdg = PathBuf::from(xdg);
            return if xdg.is_absolute() {
                xdg.join("rubyc")
            } else if let Some(home) = home_dir() {
                home.join(xdg).join("rubyc")
            } else {
                PathBuf::from(xdg).join("rubyc")
            };
        }
        if let Some(path) = std::env::var_os("HOME") {
            return PathBuf::from(path).join(".local/share/rubyc");
        }
    }
    // Fallback (no HOME, non-Linux, or XDG_DATA_HOME relative with no HOME).
    PathBuf::from(".local/share/rubyc")
}

/// `$HOME` as a `PathBuf`, or `None` if unset.
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

fn manifests(dir: &Path) -> Vec<PathBuf> {
    // Installed layout nests manifests (`targets/<name>/<name>.toml`),
    // so descend one level; flat files directly under `dir` still count.
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .into_iter().flatten().flatten()
        .flat_map(|entry| {
            let path = entry.path();
            if path.is_dir() {
                std::fs::read_dir(&path)
                    .into_iter().flatten().flatten()
                    .map(|e| e.path())
                    .collect::<Vec<_>>()
            } else {
                vec![path]
            }
        })
        .filter(|path| path.extension().is_some_and(|ext| ext == "toml"))
        .collect();
    paths.sort();
    paths
}

type Values = Vec<(String, String)>;
fn parse_manifest(text: &str) -> Values {
    text.lines().filter_map(|line| {
        let line = line.split('#').next()?.trim();
        if line.is_empty() { return None; }
        if line.starts_with('[') && line.ends_with(']') {
            return Some(("$section".into(), line[1..line.len()-1].trim().into()));
        }
        let (key, raw) = line.split_once('=')?;
        Some((key.trim().into(), raw.trim().into()))
    }).collect()
}
fn value<'a>(values: &'a Values, key: &str) -> &'a str {
    values.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v.trim_matches('"')).unwrap_or("")
}
fn section(values: &Values) -> Option<&str> { values.iter().find(|(k, _)| k == "$section").map(|(_, v)| v.as_str()) }
fn metadata(values: &Values) -> ExtensionMetadata {
    ExtensionMetadata {
        name: value(values, "name").into(), display_name: value(values, "display-name").into(),
        author: value(values, "author").into(), version: value(values, "version").into(),
        description: value(values, "description").into(), license: value(values, "license").into(),
        homepage: value(values, "homepage").into(), repository: value(values, "repository").into(),
    }.normalize()
}
