//! Shared string interpolation used by `printf` lowering today and by any
//! other format-driven feature later.

/// Alignment of one placeholder, from `{index,width}` / `{index,-width}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Align {
    /// Positive width: pad on the left (right-aligned).
    Right(usize),
    /// Negative width: pad on the right (left-aligned).
    Left(usize),
}

/// One `{index[,alignment][:format]}` item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatItem {
    pub index: usize,
    pub align: Option<Align>,
    /// Text after `:`; `None` when no colon was present or it was empty.
    pub format: Option<String>,
}

/// One piece of a parsed format string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment {
    /// Literal text (escapes already resolved).
    Text(String),
    /// A placeholder (index, optional alignment, optional format spec).
    Arg(FormatItem),
}

/// Errors from parsing a format string or applying it to arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatError {
    /// `{` never closed.
    UnclosedBrace,
    /// Braces not forming a valid index (`{}`, `{x}`, `{1x}`).
    InvalidPlaceholder(String),
    /// Placeholder refers past the last argument.
    IndexOutOfRange { index: usize, args: usize },
    /// Alignment is not `[-]digits`.
    InvalidAlignment(String),
    /// A format specifier reached a renderer that cannot apply it.
    UnsupportedSpecifier(String),
}

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FormatError::UnclosedBrace => write!(f, "'{{' without matching '}}'"),
            FormatError::InvalidPlaceholder(text) => write!(
                f,
                "{}",
                crate::tf!("rubyc.format.FormatError.invalid_placeholder", text.clone())
            ),
            FormatError::IndexOutOfRange { index, args } => {
                write!(
                    f,
                    "placeholder {{{index}}} but only {args} argument(s) given"
                )
            }
            FormatError::InvalidAlignment(text) => {
                write!(f, "invalid alignment '{text}' (expected [-]digits)")
            }
            FormatError::UnsupportedSpecifier(spec) => {
                write!(f, "format specifier ':{spec}' cannot be applied here")
            }
        }
    }
}

impl std::error::Error for FormatError {}

fn parse_item(content: &str) -> Result<FormatItem, FormatError> {
    // Index, then optional `,alignment`, then optional `:format`. A colon
    // inside the format text belongs to the format string (custom numeric
    // formats use it), so only the FIRST colon splits.
    let (head, format) = match content.split_once(':') {
        Some((h, f)) => (h, (!f.is_empty()).then(|| f.to_owned())),
        None => (content, None),
    };
    let (index_text, align_text) = match head.split_once(',') {
        Some((i, a)) => (i, Some(a)),
        None => (head, None),
    };
    if index_text.is_empty() || !index_text.chars().all(|c| c.is_ascii_digit()) {
        return Err(FormatError::InvalidPlaceholder(content.to_owned()));
    }
    let index = index_text
        .parse()
        .map_err(|_| FormatError::InvalidPlaceholder(content.to_owned()))?;
    let align = match align_text {
        None => None,
        Some(text) => {
            let (left, digits) = match text.strip_prefix('-') {
                Some(rest) => (true, rest),
                None => (false, text),
            };
            if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
                return Err(FormatError::InvalidAlignment(text.to_owned()));
            }
            let width: usize = digits
                .parse()
                .map_err(|_| FormatError::InvalidAlignment(text.to_owned()))?;
            Some(if left { Align::Left(width) } else { Align::Right(width) })
        }
    };
    Ok(FormatItem {
        index,
        align,
        format,
    })
}

/// Parse a format string into literal/placeholder segments.
///
/// `{{` and `}}` are escapes for literal braces. Placeholders are
/// `{index}`, `{index,width}`, `{index,-width}` or
/// `{index[,width][:format]}`.
pub fn split_format(format: &str) -> Result<Vec<Segment>, FormatError> {
    let mut segments = Vec::new();
    let mut text = String::new();
    let mut chars = format.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '{' => match chars.peek() {
                Some('{') => {
                    chars.next();
                    text.push('{');
                }
                _ => {
                    if !text.is_empty() {
                        segments.push(Segment::Text(std::mem::take(&mut text)));
                    }
                    let mut content = String::new();
                    let mut closed = false;
                    for ic in chars.by_ref() {
                        match ic {
                            '}' => {
                                closed = true;
                                break;
                            }
                            _ => content.push(ic),
                        }
                    }
                    if !closed {
                        return Err(FormatError::UnclosedBrace);
                    }
                    segments.push(Segment::Arg(parse_item(&content)?));
                }
            },
            '}' => {
                if chars.peek() == Some(&'}') {
                    chars.next();
                    text.push('}');
                } else {
                    return Err(FormatError::InvalidPlaceholder("}".into()));
                }
            }
            _ => text.push(c),
        }
    }
    if !text.is_empty() {
        segments.push(Segment::Text(text));
    }
    Ok(segments)
}

/// Validate placeholder indexes against the argument count.
pub fn validate_args(segments: &[Segment], args: usize) -> Result<(), FormatError> {
    for segment in segments {
        if let Segment::Arg(item) = segment
            && item.index >= args
        {
            return Err(FormatError::IndexOutOfRange {
                index: item.index,
                args,
            });
        }
    }
    Ok(())
}

/// Apply a parsed alignment to an already-rendered argument.
pub fn align_text(text: &str, align: Option<Align>) -> String {
    match align {
        None => text.to_owned(),
        Some(Align::Right(width)) => {
            let len = text.chars().count();
            if len >= width {
                text.to_owned()
            } else {
                format!("{}{text}", " ".repeat(width - len))
            }
        }
        Some(Align::Left(width)) => {
            let len = text.chars().count();
            if len >= width {
                text.to_owned()
            } else {
                format!("{text}{}", " ".repeat(width - len))
            }
        }
    }
}

/// Convenience: render a format string with pre-rendered arguments.
///
/// Alignment is applied. A non-empty format specifier is an error: the
/// arguments are already strings, so there is no numeric type to format.
pub fn interpolate(format: &str, args: &[String]) -> Result<String, FormatError> {
    let segments = split_format(format)?;
    validate_args(&segments, args.len())?;
    let mut out = String::with_capacity(format.len());
    for segment in segments {
        match segment {
            Segment::Text(t) => out.push_str(&t),
            Segment::Arg(item) => {
                if let Some(spec) = &item.format {
                    return Err(FormatError::UnsupportedSpecifier(spec.clone()));
                }
                out.push_str(&align_text(&args[item.index], item.align));
            }
        }
    }
    Ok(out)
}
