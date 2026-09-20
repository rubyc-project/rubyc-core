//! Diagnostic rendering: colored structured text and machine-readable JSON.
//!
//! The text style is RubyC's own — compact, colored, scope-aware. The JSON
//! form is stable and complete (file, line, columns, scope chain, source
//! line) so editors can consume it directly.

use crate::models::{Diagnostic, Severity};

/// Resolve a byte offset into a 1-based line and column.
pub fn line_col(source: &str, offset: u32) -> (usize, usize) {
    let offset = (offset as usize).min(source.len());
    let before = &source[..offset];
    let line = before.matches('\n').count() + 1;
    let column = offset - before.rfind('\n').map(|i| i + 1).unwrap_or(0) + 1;
    (line, column)
}

/// Extract the full source line containing `offset`.
fn source_line(source: &str, offset: u32) -> &str {
    let offset = (offset as usize).min(source.len());
    let start = source[..offset].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let end = source[offset..]
        .find('\n')
        .map(|i| offset + i)
        .unwrap_or(source.len());
    source[start..end].trim_end_matches('\r')
}

struct Palette {
    red: &'static str,
    light_cyan: &'static str,
    yellow: &'static str,
    white: &'static str,
    reset: &'static str,
}

impl Palette {
    fn plain() -> Self {
        Self {
            red: "",
            light_cyan: "",
            yellow: "",
            white: "",
            reset: "",
        }
    }

    fn colored() -> Self {
        Self {
            red: "\x1b[31m",
            light_cyan: "\x1b[96m",
            yellow: "\x1b[33m",
            white: "\x1b[97m",
            reset: "\x1b[0m",
        }
    }

    fn red(&self, s: &str) -> String {
        format!("{}{s}{}", self.red, self.reset)
    }

    fn light_cyan(&self, s: &str) -> String {
        format!("{}{s}{}", self.light_cyan, self.reset)
    }

    fn yellow(&self, s: &str) -> String {
        format!("{}{s}{}", self.yellow, self.reset)
    }

    fn white(&self, s: &str) -> String {
        format!("{}{s}{}", self.white, self.reset)
    }
}

/// Render diagnostics in the RubyC style:
///
/// ```text
/// ERROR [🐕🇨-100001] ❯ Undefined variable
/// file: main.rc:5
///   4 │ printf("{0}", name);
///   5 │ printf("{0}", nope);
///   6 │
///     │       ~~~~
/// ```
pub fn render_text(
    diags: &[Diagnostic],
    source: Option<&str>,
    file_fallback: &str,
    color: bool,
) -> String {
    let p = if color {
        Palette::colored()
    } else {
        Palette::plain()
    };
    let mut out = String::new();
    for d in diags {
        let severity = match d.severity {
            Severity::Error => p.red("ERROR"),
            Severity::Warning => p.yellow("WARN "),
        };
        let code = p.red(&format!("🐕🇨-{}", d.code));
        let arrow = p.light_cyan("❯");
        let title = p.white(&d.message);
        out.push_str(&format!("{severity} [{code}] {arrow} {title}\n"));

        let path = d.file.clone().unwrap_or_else(|| file_fallback.to_owned());

        // A project build merges many files but only carries one source
        // string (sometimes empty); a span that does not fit must degrade
        // to file+offset instead of panicking the whole build.
        if let Some(src) = source.filter(|s| {
            let end = d.span.end as usize;
            end <= s.len() && s.is_char_boundary(d.span.start as usize) && s.is_char_boundary(end)
        }) {
            let (line, col) = line_col(src, d.span.start);
            out.push_str(&format!("{}:{}\n", p.white(&path), p.yellow(&line.to_string())));

            let lines: Vec<&str> = src.lines().collect();
            let line_idx = line - 1;
            let gw = line.to_string().len();

            if line_idx > 0 {
                let prev = lines.get(line_idx - 1).unwrap_or(&"");
                out.push_str(&format!(
                    " {:>w$} {} {}\n",
                    line_idx - 1,
                    p.light_cyan("│"),
                    p.white(prev),
                    w = gw
                ));
            }

            let raw_line = source_line(src, d.span.start);
            const TAB_WIDTH: usize = 4;
            let text_line = raw_line.replace('\t', &" ".repeat(TAB_WIDTH));

            let byte_col = col.saturating_sub(1);
            let mut vis_col = 0usize;
            for ch in raw_line.chars().take(byte_col) {
                vis_col += if ch == '\t' { TAB_WIDTH } else { 1 };
            }
            let pad = " ".repeat(vis_col);
            let expanded_len = text_line.chars().count();
            let max_width = expanded_len.saturating_sub(vis_col).max(1);
            
            // Calculate the underline width, clamping to the current line.
            let line_start = d.span.start;
            let line_end_offset = src[line_start as usize..]
                .find('\n')
                .map(|i| line_start as usize + i)
                .unwrap_or(src.len()) as u32;
            let effective_end = d.span.end.min(line_end_offset);
            let width = effective_end.saturating_sub(d.span.start).max(1) as usize;
            let carets = p.red(&"~".repeat(width.min(max_width)));

            out.push_str(&format!(
                " {:>w$} {} {}\n",
                line_idx,
                p.light_cyan("│"),
                p.white(&text_line),
                w = gw
            ));
            // The underline goes on its own line, aligned under the code
            // text — never trailing the code on the same line.
            out.push_str(&format!(
                " {:>w$} {} {pad}{carets}\n",
                "",
                p.light_cyan("│"),
                w = gw
            ));

            if line_idx + 1 < lines.len() {
                let next = lines.get(line_idx + 1).unwrap_or(&"");
                out.push_str(&format!(
                    " {:>w$} {} {}\n",
                    line_idx + 1,
                    p.light_cyan("│"),
                    p.white(next),
                    w = gw
                ));
            }
        } else {
            // No usable source text (e.g. a merged multi-file unit):
            // still show where the diagnostic points.
            out.push_str(&p.white(&format!(
                "{path}:{}-{}",
                d.span.start, d.span.end
            )));
            out.push('\n');
        }

        if let Some(help) = &d.help {
            out.push_str(&format!("{}: {}\n", p.light_cyan("help"), p.white(help)));
        }
        out.push('\n');
    }
    out
}

// ---------- JSON ----------

/// Escape a string per the JSON spec (control chars, quotes, backslashes).
pub fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn json_string(s: &str) -> String {
    format!("\"{}\"", json_escape(s))
}

/// Render diagnostics as JSON. `pretty` expands with newlines/indentation;
/// compact output is a single line, suitable for piping between tools.
pub fn render_json(diags: &[Diagnostic], source: Option<&str>, pretty: bool) -> String {
    let mut out = String::from(if pretty { "[\n" } else { "[" });
    for (idx, d) in diags.iter().enumerate() {
        let (line, col) = source
            .map(|src| line_col(src, d.span.start))
            .unwrap_or((0, 0));
        let end_col = col + d.span.end.saturating_sub(d.span.start).max(1) as usize;
        let severity = match d.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
        };

        let fields: Vec<(String, String)> = vec![
            ("code".into(), format!("\"{}\"", d.code)),
            ("severity".into(), format!("\"{severity}\"")),
            ("message".into(), json_string(&d.message)),
            (
                "help".into(),
                match &d.help {
                    Some(h) => json_string(h),
                    None => "null".into(),
                },
            ),
            (
                "file".into(),
                match &d.file {
                    Some(f) => json_string(f),
                    None => "null".into(),
                },
            ),
            ("line".into(), line.to_string()),
            ("column".into(), col.to_string()),
            ("endColumn".into(), end_col.to_string()),
            (
                "namespace".into(),
                match &d.context.namespace {
                    Some(v) => json_string(v),
                    None => "null".into(),
                },
            ),
            (
                "class".into(),
                match &d.context.class {
                    Some(v) => json_string(v),
                    None => "null".into(),
                },
            ),
            (
                "function".into(),
                match &d.context.function {
                    Some(v) => json_string(v),
                    None => "null".into(),
                },
            ),
            (
                "sourceLine".into(),
                match source {
                    Some(src) => json_string(source_line(src, d.span.start)),
                    None => "null".into(),
                },
            ),
        ];

        if pretty {
            out.push_str("  {\n");
            for (i, (k, v)) in fields.iter().enumerate() {
                let comma = if i + 1 == fields.len() { "" } else { "," };
                out.push_str(&format!("    \"{k}\": {v}{comma}\n"));
            }
            out.push_str("  }");
        } else {
            out.push('{');
            let joined: Vec<String> = fields.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect();
            out.push_str(&joined.join(","));
            out.push('}');
        }
        if idx + 1 != diags.len() {
            out.push(',');
        }
        if pretty {
            out.push('\n');
        }
    }
    out.push(']');
    out
}
