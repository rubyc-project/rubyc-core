//! Project system: `.rcs` (solution) and `.rcp` (project) TOML files.
//!
//! Minimal TOML subset parser (no external dependencies):
//!   [section] headers
//!   key = value (string, integer, boolean)
//!
//! Solution files group projects; project files describe how to build.

use std::path::{Path, PathBuf};

// ---------- .rcp (project) ----------

#[derive(Debug, Clone, Default)]
pub struct Project {
    /// From `[meta] name = "…"`.
    pub name: String,
    pub version: String,
    pub author: String,
    pub description: String,
    /// From `[target] platform = "…"`, `output = "exe"|"shared"|"static"|"bytecode"`.
    /// `static` emits `out/<name>.a` (one archive member with the lib's
    /// unlinked machine code) for the static link step; the same `.rci`
    /// sidecar serves both shared and static consumers. Static state
    /// duplicates per linked image (see `native::linker`).
    pub platform: String,
    pub language: String,
    pub output_format: String,
    /// From `[build] sources = ["*.rc"]`, `entry-point = "Main"`,
    /// `defines = ["TARGET_LINUX"]` (like csproj `DefineConstants`:
    /// preprocessor symbols for this project, seeded before compilation).
    pub sources: Vec<String>,
    pub entry_point: String,
    pub defines: Vec<String>,
    /// From `[build] dynamic = true`: link dependencies dynamically
    /// (consume installed `.rci` fragments + copy-link method bodies
    /// from installed `.so` files) instead of recompiling every dep's
    /// sources. Default false (today's static-closure build). Opt-in
    /// per library after verification — unverified dynamic builds
    /// risk silent misresolution (see `native::rci`).
    pub dynamic: bool,
    /// From `[build] optimize = true`: function-level dead-code
    /// elimination (`--optimize`). CLI `--optimize` also enables it;
    /// either source suffices (explicit CLI never disables a TOML `true`).
    pub optimize: bool,
    /// From `[build] strip = true`: drop `.symtab`/`.strtab` from
    /// executable images (`--strip`). Same OR rule as `optimize`.
    pub strip: bool,
    /// From `[build] jobs = N`: solution build parallelism for this
    /// project (`--jobs N`). 0 = unset (CLI `--jobs` or 1 wins).
    pub jobs: usize,
    /// From `[build] target = "…"`: compilation target name
    /// (`--target`). `None` = unset (CLI `--target` wins).
    pub target: Option<String>,
    /// From `[build] rci-only = true`: emit only the `.rci` sidecar
    /// (`--rci-only`). Same OR rule as `optimize`.
    pub rci_only: bool,
    /// From `[libs] external = [...]`, `ruby_c = [...]`.
    pub external_libs: Vec<String>,
    pub ruby_c_libs: Vec<String>,
    /// From `[hooks] pre-build / post-build`.
    pub hook_pre_build: Option<String>,
    pub hook_post_build: Option<String>,
    /// Path to the .rcp file itself.
    pub path: PathBuf,
}

impl Project {
    pub fn parse(text: &str, path: &Path) -> Result<Self, String> {
        let mut p = Project {
            path: path.to_path_buf(),
            ..Default::default()
        };
        let mut section = String::new();
        let mut dead: Vec<String> = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            if line.starts_with('[') && line.ends_with(']') {
                section = line[1..line.len() - 1].to_owned();
                if !matches!(section.as_str(), "meta" | "target" | "build" | "libs" | "hooks") {
                    dead.push(format!("[{section}]"));
                }
                continue;
            }
            let Some(eq) = line.find('=') else {
                dead.push(format!("line '{line}'"));
                continue;
            };
            let key = line[..eq].trim();
            let val = parse_value(line[eq + 1..].trim());
            let known = if section.is_empty() {
                false
            } else {
                matches!(
                    (section.as_str(), key),
                    ("meta", "name" | "version" | "author" | "description")
                        | ("target", "platform" | "language" | "output")
                        | (
                            "build",
                            "sources"
                                | "entry-point"
                                | "defines"
                                | "dynamic"
                                | "optimize"
                                | "strip"
                                | "jobs"
                                | "target"
                                | "rci-only"
                        )
                        | ("libs", "external" | "ruby_c")
                        | ("hooks", "pre-build" | "post-build")
                )
            };
            if !known
                && !dead.iter().any(|d| d == &format!("[{section}]"))
            {
                dead.push(if section.is_empty() {
                    key.to_owned()
                } else {
                    format!("{section}.{key}")
                });
            }

            match (section.as_str(), key) {
                ("meta", "name") => p.name = as_string(&val),
                ("meta", "version") => p.version = as_string(&val),
                ("meta", "author") => p.author = as_string(&val),
                ("meta", "description") => p.description = as_string(&val),
                ("target", "platform") => p.platform = as_string(&val),
                ("target", "language") => p.language = as_string(&val),
                ("target", "output") => p.output_format = as_string(&val),
                ("hooks", "pre-build") => p.hook_pre_build = Some(as_string(&val)),
                ("hooks", "post-build") => p.hook_post_build = Some(as_string(&val)),
                _ => {}
            }
            if section == "build" && key == "sources" {
                p.sources = parse_string_array(&val);
            }
            if section == "build" && key == "entry-point" {
                p.entry_point = as_string(&val);
            }
            if section == "build" && key == "defines" {
                p.defines = parse_string_array(&val);
            }
            if section == "build" && key == "dynamic" {
                p.dynamic = val.trim() == "true";
            }
            if section == "build" && key == "optimize" {
                p.optimize = val.trim() == "true";
            }
            if section == "build" && key == "strip" {
                p.strip = val.trim() == "true";
            }
            if section == "build" && key == "jobs" {
                if let Ok(n) = val.trim().parse::<usize>() {
                    p.jobs = n;
                }
            }
            if section == "build" && key == "target" {
                let s = as_string(&val);
                if !s.is_empty() {
                    p.target = Some(s);
                }
            }
            if section == "build" && key == "rci-only" {
                p.rci_only = val.trim() == "true";
            }
            if section == "libs" && key == "external" {
                p.external_libs = parse_string_array(&val);
            }
            if section == "libs" && key == "ruby_c" {
                p.ruby_c_libs = parse_string_array(&val);
            }
        }
        if !dead.is_empty() {
            dead.sort();
            dead.dedup();
            return Err(format!(
                "unknown project key(s) in {}: {}",
                path.display(),
                dead.join(", ")
            ));
        }
        Ok(p)
    }

    /// Load from file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("cannot read {path:?}: {e}"))?;
        Self::parse(&text, path)
    }

    /// Resolve and expand every source entry relative to the project file.
    /// Results are canonicalized, deduplicated, and sorted so source order on
    /// disk cannot affect compilation.
    pub fn source_files(&self) -> Result<Vec<PathBuf>, String> {
        // A bare filename (`rubyc build foo.rcp`) has an EMPTY parent —
        // treat it as the current dir, otherwise every glob silently
        // matches nothing ("source pattern matched no files").
        let base: &Path = match self.path.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => Path::new("."),
        };
        let mut files = Vec::new();
        for entry in &self.sources {
            let pattern = base.join(entry);
            if !entry.contains(['*', '?']) {
                if !pattern.is_file() {
                    return Err(format!("source file not found: {}", pattern.display()));
                }
                files.push(pattern);
                continue;
            }

            let segments: Vec<String> = entry
                .split('/')
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect();
            let matched = glob_expand(base, &segments);
            if matched.is_empty() {
                return Err(format!(
                    "source pattern matched no files: {}",
                    pattern.display()
                ));
            }
            files.extend(matched);
        }
        files.sort();
        files.dedup();
        if files.is_empty() {
            return Err(format!(
                "project contains no source files: {}",
                self.path.display()
            ));
        }
        Ok(files)
    }

    /// Find nearest `.rcp` or `.rcs` walking up from `start_dir`.
    pub fn find_nearest(start_dir: &Path) -> Option<PathBuf> {
        let mut dir = Some(start_dir.to_path_buf());
        while let Some(d) = dir {
            for name in ["project.rcp", "MyProject.rcp"] {
                let candidate = d.join(name);
                if candidate.exists() {
                    return Some(candidate);
                }
            }
                if let Ok(entries) = std::fs::read_dir(&d) {
                for entry in entries.flatten() {
                    let ext = entry
                        .path()
                        .extension()
                        .map(|e| e.to_string_lossy().to_lowercase());
                    if matches!(ext.as_deref(), Some("rcp") | Some("rcs")) {
                        return Some(entry.path());
                    }
                }
            }
            dir = d.parent().map(|p| p.to_path_buf());
        }
        None
    }

    /// Run a build hook via shell.
    /// Run a pre/post-build hook with the selected target triple visible
    /// as `RUBYC_TARGET` (e.g. `linux_x86_64`; empty when none was
    /// selected). Hooks pick target-OS-specific sources off it (the
    /// dotnet way: same hook, different files per target OS).
    pub fn run_hook(&self, hook: &str, target: Option<&str>) -> Result<(), String> {
        let cmd = match hook {
            "pre-build" => &self.hook_pre_build,
            "post-build" => &self.hook_post_build,
            _ => return Ok(()),
        };
        if let Some(cmd) = cmd {
            eprintln!("[hook:{hook}] {cmd}");
            // Hooks run relative to the project directory so generated
            // `pre-build`/`post-build` scripts can use relative paths.
            // A bare filename (`rubyc build foo.rcp` from its own dir)
            // has an empty parent — fall back to "." instead of failing
            // the spawn with ENOENT.
            let dir = self
                .path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."));
            let status = std::process::Command::new("sh")
                .arg("-c")
                .arg(cmd)
                .current_dir(dir)
                .env("RUBYC_TARGET", target.unwrap_or(""))
                .status()
                .map_err(|e| format!("hook failed to start: {e}"))?;
            if !status.success() {
                return Err(format!(
                    "hook '{hook}' exited with {}",
                    status.code().unwrap_or(-1)
                ));
            }
        }
        Ok(())
    }
}

// ---------- .rcs (solution) ----------

#[derive(Debug, Clone, Default)]
pub struct Solution {
    pub name: String,
    pub version: String,
    pub projects: Vec<PathBuf>,
    pub profiles: std::collections::BTreeMap<String, ProfileConfig>,
    pub hook_pre_solution: Option<String>,
    pub path: PathBuf,
}

#[derive(Debug, Clone, Default)]
pub struct ProfileConfig {
    pub defines: Vec<String>,
    pub optimize: bool,
    pub strip: bool,
}

impl Solution {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("cannot read {path:?}: {e}"))?;
        let mut sol = Solution {
            path: path.to_path_buf(),
            ..Default::default()
        };
        let mut section = String::new();
        let mut dead: Vec<String> = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            if line.starts_with("[[project]]") {
                // Next path = ... is a project reference
                section = "__project__".into();
                continue;
            }
            if let Some(rest) = line.strip_prefix("solution ") {
                sol.name = as_string(rest);
                continue;
            }
            if line.starts_with('[') && line.ends_with(']') {
                section = line[1..line.len() - 1].trim_start_matches('[').to_owned();
                let known_section = matches!(section.as_str(), "solution" | "hooks")
                    || section
                        .strip_prefix("profile.")
                        .is_some_and(|name| !name.is_empty());
                if !known_section {
                    dead.push(format!("[{section}]"));
                }
                continue;
            }
            let Some(eq) = line.find('=') else {
                dead.push(format!("line '{line}'"));
                continue;
            };
            let key = line[..eq].trim();
            let val = parse_value(line[eq + 1..].trim());
            let known = match section.as_str() {
                "" => matches!(key, "name" | "version"),
                "solution" => key == "version",
                "__project__" => key == "path",
                "hooks" => key == "pre-solution",
                s if s.starts_with("profile.") => {
                    matches!(key, "defines" | "optimize" | "strip")
                }
                _ => false,
            };
            if !known && !dead.iter().any(|d| d == &format!("[{section}]")) {
                dead.push(if section.is_empty() {
                    key.to_owned()
                } else {
                    format!("{section}.{key}")
                });
            }

            match (section.as_str(), key) {
                ("meta", "name") | (_, "name") if section.is_empty() => sol.name = as_string(&val),
                (_, "version") if section.is_empty() || section == "solution" => {
                    sol.version = as_string(&val)
                }
                ("__project__", "path") => {
                    sol.projects.push(PathBuf::from(as_string(&val)));
                }
                ("hooks", "pre-solution") => sol.hook_pre_solution = Some(as_string(&val)),
                _ => {
                    // `[profile.NAME]` sections: named build profiles
                    // selected with `--profile NAME` (defines append,
                    // optimize/strip OR into every project).
                    if let Some(name) = section.strip_prefix("profile.") {
                        if !name.is_empty() {
                            let entry = sol.profiles.entry(name.to_owned()).or_default();
                            match key {
                                "defines" => entry.defines = parse_string_array(&val),
                                "optimize" => entry.optimize = val.trim() == "true",
                                "strip" => entry.strip = val.trim() == "true",
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
        if !dead.is_empty() {
            dead.sort();
            dead.dedup();
            return Err(format!(
                "unknown solution key(s) in {}: {}",
                path.display(),
                dead.join(", ")
            ));
        }
        Ok(sol)
    }

    /// Project paths resolved relative to the solution file.
    pub fn project_files(&self) -> Result<Vec<PathBuf>, String> {
        // Same empty-parent rule as `source_files` above.
        let base: &Path = match self.path.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => Path::new("."),
        };
        let mut projects = Vec::new();
        for entry in &self.projects {
            let path = base.join(entry);
            if !path.is_file() {
                return Err(format!("project file not found: {}", path.display()));
            }
            if path.extension().and_then(|ext| ext.to_str()) != Some("rcp") {
                return Err(format!(
                    "solution entry is not an .rcp project: {}",
                    path.display()
                ));
            }
            if projects.contains(&path) {
                return Err(format!("duplicate project in solution: {}", path.display()));
            }
            projects.push(path);
        }
        if projects.is_empty() {
            return Err(format!(
                "solution contains no projects: {}",
                self.path.display()
            ));
        }
        Ok(projects)
    }
}

// ---------- TOML helpers ----------

fn parse_value(raw: &str) -> String {
    raw.trim().to_owned()
}

fn as_string(val: &str) -> String {
    // Strip TOML string delimiters only.
    let v = val.trim();
    if v.len() >= 2
        && ((v.starts_with('"') && v.ends_with('"')) || (v.starts_with('\'') && v.ends_with('\'')))
    {
        v[1..v.len() - 1].to_owned()
    } else {
        v.to_owned()
    }
}

fn parse_string_array(val: &str) -> Vec<String> {
    let inner = val.trim_start_matches('[').trim_end_matches(']');
    inner
        .split(',')
        .map(|s| s.trim().trim_matches('"').to_owned())
        .filter(|s| !s.is_empty())
        .collect()
}

fn wildcard_matches(pattern: &str, value: &str) -> bool {
    let pattern = pattern.as_bytes();
    let value = value.as_bytes();
    let (mut p, mut v, mut star, mut retry) = (0usize, 0usize, None, 0usize);
    while v < value.len() {
        if p < pattern.len() && (pattern[p] == b'?' || pattern[p] == value[v]) {
            p += 1;
            v += 1;
        } else if p < pattern.len() && pattern[p] == b'*' {
            star = Some(p);
            p += 1;
            retry = v;
        } else if let Some(star_pos) = star {
            p = star_pos + 1;
            retry += 1;
            v = retry;
        } else {
            return false;
        }
    }
    while p < pattern.len() && pattern[p] == b'*' {
        p += 1;
    }
    p == pattern.len()
}

/// Expand a glob pattern (relative to `root`) into matching file paths.
/// A `**` segment matches zero or more directory levels; every other segment
/// is matched against a single path component via `wildcard_matches`.
fn glob_expand(root: &Path, segments: &[String]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    glob_walk(root, segments, &mut out);
    out
}

fn glob_walk(dir: &Path, segments: &[String], out: &mut Vec<PathBuf>) {
    if segments.is_empty() {
        if dir.is_file() {
            out.push(dir.to_path_buf());
        }
        return;
    }
    if segments[0] == "**" {
        // Zero levels: try the remainder of the pattern here.
        glob_walk(dir, &segments[1..], out);
        // One or more levels: descend into each subdirectory.
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    glob_walk(&path, segments, out);
                }
            }
        }
        return;
    }
    let last = segments.len() == 1;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if !wildcard_matches(&segments[0], &name) {
                continue;
            }
            if last {
                if path.is_file() {
                    out.push(path);
                }
            } else if path.is_dir() {
                glob_walk(&path, &segments[1..], out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn parse(text: &str) -> Project {
        Project::parse(text, Path::new("t.rcp")).expect("test project parses")
    }

    #[test]
    fn build_optimize_strip_keys_parse() {
        let p = parse(
            "[meta]\nname = \"x\"\n[target]\noutput = \"exe\"\n[build]\n\
             sources = []\noptimize = true\nstrip = true\n",
        );
        assert!(p.optimize, "optimize = true parses");
        assert!(p.strip, "strip = true parses");
    }

    #[test]
    fn build_optimize_strip_default_off() {
        let p = parse("[meta]\nname = \"x\"\n[build]\nsources = []\n");
        assert!(!p.optimize && !p.strip, "absent keys default to false");
    }

    #[test]
    fn build_jobs_target_rcionly_keys_parse() {
        let p = parse(
            "[meta]\nname = \"x\"\n[build]\nsources = []\njobs = 6\n\
             target = \"linux_x86_64\"\nrci-only = true\n",
        );
        assert_eq!(p.jobs, 6, "jobs parses");
        assert_eq!(p.target.as_deref(), Some("linux_x86_64"), "target parses");
        assert!(p.rci_only, "rci-only parses");
    }

    #[test]
    fn build_jobs_target_rcionly_default_unset() {
        let p = parse("[meta]\nname = \"x\"\n[build]\nsources = []\n");
        assert_eq!(p.jobs, 0, "jobs defaults to unset (0)");
        assert_eq!(p.target, None, "target defaults to unset");
        assert!(!p.rci_only, "rci-only defaults to false");
    }

    #[test]
    fn solution_profile_sections_parse() {
        let dir = std::env::temp_dir();
        let path = dir.join("rubyc_profile_test.rcs");
        std::fs::write(
            &path,
            "solution mini\nversion = \"1\"\n[[project]]\npath = \"a.rcp\"\n\
             [profile.release]\ndefines = [\"DEBUG\"]\noptimize = true\nstrip = true\n",
        )
        .unwrap();
        let sol = Solution::load(&path).expect("solution parses");
        let _ = std::fs::remove_file(&path);
        let prof = sol.profiles.get("release").expect("profile parsed");
        assert_eq!(prof.defines, vec!["DEBUG".to_string()]);
        assert!(prof.optimize && prof.strip);
    }
}
