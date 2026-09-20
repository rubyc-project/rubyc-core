//! Shared string interpolation used by `printf` lowering today and by any
//! other format-driven feature later.

/// One piece of a parsed format string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment {
    /// Literal text (escapes already resolved).
    Text(String),
    /// `{N}` — take argument N.
    Arg(usize),
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
        }
    }
}

impl std::error::Error for FormatError {}

/// Parse a format string into literal/placeholder segments.
///
/// `{{` and `}}` are escapes for literal braces. Placeholders are zero-based
/// argument indexes: `{0}`, `{1}`, ...
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
                    let mut index = String::new();
                    let mut closed = false;
                    for ic in chars.by_ref() {
                        match ic {
                            '}' => {
                                closed = true;
                                break;
                            }
                            _ => index.push(ic),
                        }
                    }
                    if !closed {
                        return Err(FormatError::UnclosedBrace);
                    }
                    if !index.chars().all(|d| d.is_ascii_digit()) || index.is_empty() {
                        return Err(FormatError::InvalidPlaceholder(index));
                    }
                    segments.push(Segment::Arg(index.parse().unwrap()));
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
        if let Segment::Arg(i) = segment
            && *i >= args
        {
            return Err(FormatError::IndexOutOfRange { index: *i, args });
        }
    }
    Ok(())
}

/// Convenience: render a format string with pre-rendered arguments.
pub fn interpolate(format: &str, args: &[String]) -> Result<String, FormatError> {
    let segments = split_format(format)?;
    validate_args(&segments, args.len())?;
    let mut out = String::with_capacity(format.len());
    for segment in segments {
        match segment {
            Segment::Text(t) => out.push_str(&t),
            Segment::Arg(i) => out.push_str(&args[i]),
        }
    }
    Ok(out)
}
