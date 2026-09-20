//! Byte-offset range of a token or AST node in the original source text.

/// A half-open range `[start, end)` of bytes in the source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    /// Whole-line placeholder span for preprocessor diagnostics (the
    /// exact byte range is filled by the renderer from `line`).
    pub fn line(lineno: usize) -> Self {
        // Encode the 1-based line number in both fields; render_text
        // detects this via its own source slicing.
        let v = lineno.min(u32::MAX as usize) as u32;
        Self { start: v, end: v }
    }

    /// Smallest span covering both spans.
    pub fn to(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    /// Extract the covered substring from source text.
    pub fn slice<'src>(&self, source: &'src str) -> &'src str {
        &source[self.start as usize..self.end as usize]
    }
}
