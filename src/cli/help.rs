//! Help and version text rendering.

use super::Command;
use super::spec::{ArgKind, COMMANDS, CommandSpec, EXTENSION_FLAGS, FlagSpec, GLOBAL_FLAGS, spec_for};

// ---------- help rendering ----------

/// ANSI styling helpers; no-ops when color is disabled.
struct Palette {
    bold: &'static str,
    cyan: &'static str,
    yellow: &'static str,
    reset: &'static str,
}

impl Palette {
    fn enabled() -> bool {
        std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal()
    }

    fn plain() -> Self {
        Self {
            bold: "",
            cyan: "",
            yellow: "",
            reset: "",
        }
    }

    fn colored() -> Self {
        Self {
            bold: "\x1b[1m",
            cyan: "\x1b[36m",
            yellow: "\x1b[33m",
            reset: "\x1b[0m",
        }
    }

    fn bold(&self, s: &str) -> String {
        format!("{}{s}{}", self.bold, self.reset)
    }

    fn cyan(&self, s: &str) -> String {
        format!("{}{s}{}", self.cyan, self.reset)
    }

    fn yellow(&self, s: &str) -> String {
        format!("{}{s}{}", self.yellow, self.reset)
    }
}

pub fn version_text() -> String {
    crate::tf!("rubyc.cli.version", env!("CARGO_PKG_VERSION"))
}

use std::io::IsTerminal;

/// Render root help (`rubyc --help`) or per-command help.
pub fn render_help(command: Option<Command>) -> String {
    let palette = if Palette::enabled() {
        Palette::colored()
    } else {
        Palette::plain()
    };
    let mut out = String::new();

    match command.map(|c| spec_for(Some(c))) {
        Some(Some(spec)) => render_command_help(spec, &palette, &mut out),
        _ => render_root_help(&palette, &mut out),
    }
    out
}

fn render_root_help(p: &Palette, out: &mut String) {
    out.push_str(&format!(
        "{}\n\n",
        crate::tf!(
            "rubyc.cli.render.title",
            env!("CARGO_PKG_VERSION").to_owned()
        )
    ));
    out.push_str(&format!(
        "{} {} {} {}\n\n",
        p.cyan(
            crate::t!("rubyc.cli.render.usage")
                .split_whitespace()
                .next()
                .unwrap_or("Usage:"),
        ),
        p.bold("rubyc"),
        p.yellow("<COMMAND>"),
        "[OPTIONS]"
    ));
    out.push_str(&p.cyan(&*crate::t!("rubyc.cli.render.commands")));
    out.push('\n');
    let width = COMMANDS.iter().map(|c| c.name.len()).max().unwrap_or(0);
    for c in COMMANDS {
        out.push_str(&format!(
            "  {:width$}  {}\n",
            c.name,
            crate::t!(c.about),
            width = width
        ));
    }
    out.push('\n');
    out.push_str(&p.cyan(&*crate::t!("rubyc.cli.render.global_options")));
    out.push('\n');
    render_flag_rows(
        &[
            GLOBAL_FLAGS,
            EXTENSION_FLAGS,
            &[
                FlagSpec {
                    short: None,
                    long: "no-color",
                    kind: ArgKind::Switch,
                    choices: &[],
                    placeholder: "",
                    help: "rubyc.cli.flag.no_color",
                },
                FlagSpec {
                    short: None,
                    long: "error-format",
                    kind: ArgKind::Required,
                    choices: &["text", "json", "json-pretty"],
                    placeholder: "FMT",
                    help: "rubyc.cli.flag.error_format",
                },
                FlagSpec {
                    short: None,
                    long: "warn",
                    kind: ArgKind::Required,
                    choices: &[],
                    placeholder: "LINT",
                    help: "rubyc.cli.flag.warn",
                },
                FlagSpec {
                    short: None,
                    long: "deny",
                    kind: ArgKind::Required,
                    choices: &[],
                    placeholder: "LINT",
                    help: "rubyc.cli.flag.deny",
                },
                FlagSpec {
                    short: None,
                    long: "allow",
                    kind: ArgKind::Required,
                    choices: &[],
                    placeholder: "LINT",
                    help: "rubyc.cli.flag.allow",
                },
            ],
        ],
        p,
        out,
    );
    out.push_str(&format!(
        "  -h, --help       {}\n  -V, --version    {}\n\n",
        crate::t!("rubyc.cli.render.print_help"),
        crate::t!("rubyc.cli.render.print_version")
    ));
    out.push_str(&p.cyan(&*crate::t!("rubyc.cli.render.quickstart")));
    out.push('\n');
    for key in [
        "rubyc.cli.render.qs1",
        "rubyc.cli.render.qs2",
        "rubyc.cli.render.qs3",
        "rubyc.cli.render.qs4",
    ] {
        out.push_str(&*crate::t!(key));
        out.push('\n');
    }
    out.push('\n');
    out.push_str(crate::t!("rubyc.cli.render.exit_codes"));
    out.push_str("\n\n");
    out.push_str("  --list-targets       enumerate installed compilation targets\n");
    out.push_str(crate::t!("rubyc.cli.render.help_more"));
    out.push('\n');
}

fn render_command_help(spec: &CommandSpec, p: &Palette, out: &mut String) {
    out.push_str(&format!("{}\n\n", crate::t!(spec.about)));
    out.push_str(&format!(
        "{} {} {} [OPTIONS] [FILE]\n\n",
        p.cyan("Usage:"),
        p.bold("rubyc"),
        p.bold(spec.name)
    ));
    if !spec.flags.is_empty() {
        out.push_str(&p.cyan(&*crate::t!("rubyc.cli.render.options")));
        out.push('\n');
        render_flag_rows(&[spec.flags], p, out);
        out.push('\n');
    }
    out.push_str(&p.cyan(&*crate::t!("rubyc.cli.render.global_options")));
    render_flag_rows(&[GLOBAL_FLAGS, EXTENSION_FLAGS], p, out);
    out.push('\n');
    for line in crate::t!(spec.examples).lines() {
        out.push_str(line.trim_end());
        out.push('\n');
    }
}

fn render_flag_rows(groups: &[&[FlagSpec]], p: &Palette, out: &mut String) {
    // Build rows first so columns align across groups.
    let mut rows: Vec<(String, String)> = Vec::new();
    for group in groups {
        for f in *group {
            let left = match (f.short, f.kind) {
                (Some(s), ArgKind::Switch) => format!("-{s}, --{}", f.long),
                (Some(s), _) => format!("-{s}, --{} <{}>", f.long, f.placeholder),
                (None, ArgKind::Switch) => format!("    --{}", f.long),
                (None, _) => format!("    --{} <{}>", f.long, f.placeholder),
            };
            rows.push((left, crate::t!(f.help).to_owned()));
        }
    }
    let width = rows.iter().map(|(l, _)| l.len()).max().unwrap_or(0);
    for (left, help) in rows {
        let padded = p.yellow(&format!("{left:<width$}"));
        out.push_str(&format!("  {padded}  {help}\n"));
    }
}
