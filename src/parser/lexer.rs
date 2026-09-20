//! Converts RubyC source text into a flat token stream.
//!
//! Trivia such as whitespace and comments is discarded here, so the parser
//! works uniformly regardless of brace placement or line breaks.

use crate::models::{Diagnostic, Keyword, ScopeContext, Severity, Span, Token, TokenKind};

pub struct Lexer<'src> {
    src: &'src [u8],
    pos: usize,
    /// Offset where the token currently being scanned began.
    token_start: usize,
    /// Lexical diagnostics (e.g. unterminated block comments).
    diags: Vec<Diagnostic>,
}

/// Tokens plus any lexical diagnostics from scanning.
pub struct LexOutput {
    pub tokens: Vec<Token>,
    pub diags: Vec<Diagnostic>,
}

impl<'src> Lexer<'src> {
    pub fn new(src: &'src str) -> Self {
        Self {
            src: src.as_bytes(),
            pos: 0,
            token_start: 0,
            diags: Vec::new(),
        }
    }

    /// Lex the whole input. The returned vector always ends with an `Eof`
    /// token so the parser never has to bounds-check.
    pub fn tokenize(self) -> Vec<Token> {
        self.tokenize_with_diags().tokens
    }

    /// Tokens plus lexical diagnostics (unterminated comments, ...).
    pub fn tokenize_with_diags(mut self) -> LexOutput {
        let mut tokens = Vec::new();
        loop {
            let token = self.next_token();
            let at_eof = token.kind == TokenKind::Eof;
            tokens.push(token);
            if at_eof {
                return LexOutput {
                    tokens,
                    diags: std::mem::take(&mut self.diags),
                };
            }
        }
    }

    fn next_token(&mut self) -> Token {
        self.skip_trivia();
        self.token_start = self.pos;
        let Some(&b) = self.peek() else {
            return self.make(TokenKind::Eof);
        };
        let kind = match b {
            b'{' => self.single(TokenKind::LeftBrace),
            b'}' => self.single(TokenKind::RightBrace),
            b'(' => self.single(TokenKind::LeftParen),
            b')' => self.single(TokenKind::RightParen),
            b'[' => self.single(TokenKind::LeftBracket),
            b']' => self.single(TokenKind::RightBracket),
            b';' => self.single(TokenKind::Semicolon),
            b',' => self.single(TokenKind::Comma),
            b'.' => match self.src.get(self.pos + 1) {
                Some(b'.') => self.double2(TokenKind::DotDot),
                // Leading-dot fractional literal: `.90` is a C# double.
                // (Rust's float parser accepts the leading-dot form.)
                Some(b'0'..=b'9') => {
                    self.pos += 1; // '.'
                    while matches!(self.peek(), Some(b'0'..=b'9')) {
                        self.pos += 1;
                    }
                    if matches!(self.peek(), Some(b'e' | b'E')) {
                        self.pos += 1;
                        if matches!(self.peek(), Some(b'+' | b'-')) {
                            self.pos += 1;
                        }
                        while matches!(self.peek(), Some(b'0'..=b'9')) {
                            self.pos += 1;
                        }
                    }
                    if matches!(
                        self.peek(),
                        Some(b'f' | b'F' | b'd' | b'D' | b'm' | b'M')
                    ) {
                        self.pos += 1;
                    }
                    let text =
                        std::str::from_utf8(&self.src[self.token_start..self.pos])
                            .unwrap_or("0");
                    let cleaned: String =
                        text.chars().filter(|c| *c != '_').collect();
                    let value = cleaned
                        .trim_end_matches(|c| {
                            matches!(c, 'f' | 'F' | 'd' | 'D' | 'm' | 'M')
                        })
                        .parse::<f64>()
                        .unwrap_or(0.0);
                    TokenKind::FloatLiteral(value.to_bits())
                }
                _ => self.single(TokenKind::Dot),
            },
            b':' => self.single(TokenKind::Colon),
            b'?' => match self.src.get(self.pos + 1) {
                Some(b'?') if self.src.get(self.pos + 2) == Some(&b'=') => {
                    self.pos += 3;
                    TokenKind::QuestionQuestionAssign
                }
                Some(b'?') => self.double2(TokenKind::QuestionQuestion),
                Some(b'.') => self.double2(TokenKind::QuestionDot),
                _ => self.single(TokenKind::Question),
            },
            b'~' => self.single(TokenKind::Tilde),
            // Two-byte operators first (longest match wins), then singles.
            b'+' => match self.src.get(self.pos + 1) {
                Some(b'+') => self.double2(TokenKind::PlusPlus),
                Some(b'=') => self.double2(TokenKind::PlusAssign),
                _ => self.single(TokenKind::Plus),
            },
            b'-' => match self.src.get(self.pos + 1) {
                Some(b'-') => self.double2(TokenKind::MinusMinus),
                Some(b'=') => self.double2(TokenKind::MinusAssign),
                Some(b'>') => self.double2(TokenKind::Arrow),
                _ => self.single(TokenKind::Minus),
            },
            b'*' => match self.src.get(self.pos + 1) {
                Some(b'=') => self.double2(TokenKind::StarAssign),
                _ => self.single(TokenKind::Star),
            },
            b'/' if self.src.get(self.pos + 1) != Some(&b'/') => {
                if self.src.get(self.pos + 1) == Some(&b'=') {
                    self.pos += 2;
                    TokenKind::SlashAssign
                } else {
                    self.single(TokenKind::Slash)
                }
            }
            b'%' => match self.src.get(self.pos + 1) {
                Some(b'=') => self.double2(TokenKind::PercentAssign),
                _ => self.single(TokenKind::Percent),
            },
            b'^' => match self.src.get(self.pos + 1) {
                Some(b'=') => self.double2(TokenKind::CaretAssign),
                _ => self.single(TokenKind::Hat),
            },
            b'&' => match self.src.get(self.pos + 1) {
                Some(b'&') => self.double2(TokenKind::AmpAmp),
                Some(b'=') => self.double2(TokenKind::AmpAssign),
                _ => self.single(TokenKind::Amp),
            },
            b'|' => match self.src.get(self.pos + 1) {
                Some(b'|') => self.double2(TokenKind::PipePipe),
                Some(b'=') => self.double2(TokenKind::PipeAssign),
                _ => self.single(TokenKind::Pipe),
            },
            b'=' => {
                if self.src.get(self.pos + 1) == Some(&b'=') {
                    self.pos += 2;
                    TokenKind::EqEq
                } else if self.src.get(self.pos + 1) == Some(&b'>') {
                    self.pos += 2;
                    TokenKind::FatArrow
                } else {
                    self.single(TokenKind::Assign)
                }
            }
            b'!' => {
                if self.src.get(self.pos + 1) == Some(&b'=') {
                    self.pos += 2;
                    TokenKind::NotEq
                } else {
                    self.single(TokenKind::Bang)
                }
            }
            b'<' => match self.src.get(self.pos + 1) {
                Some(b'<') => {
                    if self.src.get(self.pos + 2) == Some(&b'=') {
                        self.pos += 3;
                        TokenKind::ShlAssign
                    } else {
                        self.double2(TokenKind::Shl)
                    }
                }
                Some(b'=') => self.double2(TokenKind::LessEq),
                _ => self.single(TokenKind::Less),
            },
            b'>' => match (self.src.get(self.pos + 1), self.src.get(self.pos + 2)) {
                (Some(b'>'), Some(b'>')) => {
                    if self.src.get(self.pos + 3) == Some(&b'=') {
                        self.pos += 4;
                        TokenKind::ShrShrAssign
                    } else {
                        self.pos += 3;
                        TokenKind::ShrShr
                    }
                }
                (Some(b'>'), _) => {
                    if self.src.get(self.pos + 2) == Some(&b'=') {
                        self.pos += 3;
                        TokenKind::ShrAssign
                    } else {
                        self.double2(TokenKind::Shr)
                    }
                }
                (Some(b'='), _) => self.double2(TokenKind::GreaterEq),
                _ => self.single(TokenKind::Greater),
            },
            b'"' if self.src.get(self.pos + 1) == Some(&b'"')
                && self.src.get(self.pos + 2) == Some(&b'"') =>
            {
                // C# 11 raw string literal: `"""..."""`. The number of
                // quotes at the start determines the closing sequence.
                // The first line is the indent spec; we record the
                // body verbatim and drop the indent.
                self.pos += 3;
                let mut value = Vec::new();
                // Read until we see three or more consecutive `"` characters.
                while self.pos < self.src.len() {
                    if self.peek() == Some(&b'"')
                        && self.src.get(self.pos + 1) == Some(&b'"')
                        && self.src.get(self.pos + 2) == Some(&b'"')
                    {
                        self.pos += 3;
                        // Allow additional quotes (3+ on each side).
                        while self.peek() == Some(&b'"') {
                            value.push(b'"');
                            self.pos += 1;
                        }
                        break;
                    }
                    value.push(self.src[self.pos]);
                    self.pos += 1;
                }
                TokenKind::StringLiteral(String::from_utf8_lossy(&value).trim().to_string())
            }
            b'"' => self.string_literal(),
            b'$' if self.src.get(self.pos + 1) == Some(&b'"')
                && self.src.get(self.pos + 2) == Some(&b'"')
                && self.src.get(self.pos + 3) == Some(&b'"') =>
            {
                // Interpolated raw string: `$"""..."""`.
                self.interpolated_raw_string()
            }
            b'$' if self.src.get(self.pos + 1) == Some(&b'@')
                && self.src.get(self.pos + 2) == Some(&b'"') =>
            {
                // Interpolated verbatim string: `$@"..."`.
                self.pos += 2;
                self.interpolated_verbatim_string()
            }
            b'$' if self.src.get(self.pos + 1) == Some(&b'"') => {
                // Interpolated string: `$"..."`.
                self.pos += 1; // Skip '$'
                self.interpolated_string()
            }
            b'@' if self.src.get(self.pos + 1) == Some(&b'"') => {
                // Verbatim string: `@"..."`. Only `"` ends it; backslashes
                // are literal. `""` is an escape for a single `"`.
                self.pos += 1; // Skip '@'
                self.verbatim_string_literal()
            }
            b'\'' => self.char_literal(),
            b'0'..=b'9' => self.int_literal(),
            b'A'..=b'Z' | b'a'..=b'z' | b'_' => self.ident_or_keyword(),
            b'@' if matches!(self.src.get(self.pos + 1), Some(b'A'..=b'Z' | b'a'..=b'z' | b'_')) => {
                // Verbatim identifier: `@class`, `@this`, `@new`. The `@`
                // is dropped and the name is ALWAYS an identifier token —
                // never keyword-mapped (C#: `@x` ≡ `x` as a name, even
                // for reserved words like `@this` in SZArrayHelper).
                self.pos += 1; // consume '@'
                self.token_start = self.pos;
                while matches!(
                    self.peek(),
                    Some(b'A'..=b'Z' | b'a'..=b'z' | b'_' | b'0'..=b'9')
                ) {
                    self.pos += 1;
                }
                let text = &self.src[self.token_start..self.pos];
                let text = std::str::from_utf8(text).expect(crate::t!(
                    "rubyc.parser.lexer.Lexer.ident_or_keyword.utf8_expect"
                ));
                TokenKind::Identifier(text.to_owned())
            }
            _ => {
                // Unknown byte: consume it so lexing always makes progress.
                self.pos += 1;
                return self.make(TokenKind::Eof);
            }
        };
        self.make(kind)
    }

    fn skip_trivia(&mut self) {
        while let Some(&b) = self.peek() {
            if b == b' ' || b == b'\t' || b == b'\r' || b == b'\n' {
                self.pos += 1;
            } else if b >= 0x80 {
                // Unicode whitespace (C# spec; real dotnet sources
                // contain NBSP U+00A0, e.g. Task.cs) and the UTF-8 BOM
                // (U+FEFF). Rust's `is_whitespace` covers the Zs
                // category plus TAB/VT/FF/LF/CR/NEL. Anything else ends
                // trivia (the unknown-byte arm reports it downstream).
                let rest = &self.src[self.pos..];
                let s = std::str::from_utf8(rest).unwrap_or("");
                match s.chars().next() {
                    Some(ch) if ch == '\u{FEFF}' || ch.is_whitespace() => {
                        self.pos += ch.len_utf8();
                    }
                    _ => break,
                }
            } else if b == b'/' && self.src.get(self.pos + 1) == Some(&b'/') {
                // Line comment: run to end of line.
                while let Some(&c) = self.peek()
                    && c != b'\n'
                {
                    self.pos += 1;
                }
            } else if b == b'/' && self.src.get(self.pos + 1) == Some(&b'*') {
                // Block comment: non-nesting, first '*/' closes it.
                self.pos += 2;
                let mut closed = false;
                while let Some(&c) = self.peek() {
                    if c == b'*' && self.src.get(self.pos + 1) == Some(&b'/') {
                        self.pos += 2;
                        closed = true;
                        break;
                    }
                    self.pos += 1;
                }
                if !closed {
                    self.diags.push(
                        Diagnostic::new(
                            "E0005",
                            Severity::Error,
                            crate::t!(
                                "rubyc.parser.lexer.Lexer.skip_trivia.unterminated_block_comment"
                            ),
                        )
                        .at(Span::new(self.token_start as u32, self.pos as u32))
                        .in_context(ScopeContext::default()),
                    );
                }
            } else {
                break;
            }
        }
    }

    fn single(&mut self, kind: TokenKind) -> TokenKind {
        self.pos += 1;
        kind
    }

    /// Consume the already-verified second byte and return `kind`.
    fn double2(&mut self, kind: TokenKind) -> TokenKind {
        self.pos += 2;
        kind
    }

    fn ident_or_keyword(&mut self) -> TokenKind {
        while matches!(
            self.peek(),
            Some(b'A'..=b'Z' | b'a'..=b'z' | b'_' | b'0'..=b'9')
        ) {
            self.pos += 1;
        }
        let text = &self.src[self.token_start..self.pos];
        let text = std::str::from_utf8(text).expect(crate::t!(
            "rubyc.parser.lexer.Lexer.ident_or_keyword.utf8_expect"
        ));
        match Keyword::from_ident(text) {
            Some(kw) => TokenKind::Keyword(kw),
            None => TokenKind::Identifier(text.to_owned()),
        }
    }

    fn int_literal(&mut self) -> TokenKind {
        // C# 11 numeric literal forms:
        //   - decimal: `0`, `1_000`
        //   - hex: `0xFF`
        //   - binary: `0b0111`
        //   - floats: `0.0`, `1.5e10`, `3.25f`, `9.75m`, `2.5d`, `1e10`
        // A literal is a float when it has a fractional part, an exponent, or
        // a float suffix (`f`/`d`/`m`); otherwise it is an integer.
        let mut is_float = false;
        if self.peek() == Some(&b'0')
            && matches!(self.src.get(self.pos + 1), Some(b'x' | b'X' | b'b' | b'B'))
        {
            self.pos += 2;
            while matches!(
                self.peek(),
                Some(b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F' | b'_')
            ) {
                self.pos += 1;
            }
        } else {
            while matches!(self.peek(), Some(b'0'..=b'9' | b'_')) {
                self.pos += 1;
            }
            // Float: digits.digits with optional exponent and suffix.
            if self.peek() == Some(&b'.')
                && matches!(self.src.get(self.pos + 1), Some(b'0'..=b'9'))
            {
                is_float = true;
                self.pos += 1; // '.'
                while matches!(self.peek(), Some(b'0'..=b'9' | b'_')) {
                    self.pos += 1;
                }
                if matches!(self.peek(), Some(b'e' | b'E')) {
                    self.pos += 1;
                    if matches!(self.peek(), Some(b'+' | b'-')) {
                        self.pos += 1;
                    }
                    while matches!(self.peek(), Some(b'0'..=b'9')) {
                        self.pos += 1;
                    }
                }
                // Suffix: `f`, `F`, `d`, `D`, `m`, `M`.
                if matches!(self.peek(), Some(b'f' | b'F' | b'd' | b'D' | b'm' | b'M')) {
                    self.pos += 1;
                }
            } else if matches!(self.peek(), Some(b'e' | b'E')) {
                // `1e10` etc. with no decimal point.
                is_float = true;
                self.pos += 1;
                if matches!(self.peek(), Some(b'+' | b'-')) {
                    self.pos += 1;
                }
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
                if matches!(self.peek(), Some(b'f' | b'F' | b'd' | b'D' | b'm' | b'M')) {
                    self.pos += 1;
                }
            } else if matches!(self.peek(), Some(b'f' | b'F' | b'd' | b'D' | b'm' | b'M')) {
                // Integer with float suffix (`0f`, `0d`, `0m`).
                is_float = true;
                self.pos += 1;
            }
        }
        // Integral suffix: `L`/`l`/`U`/`u`/`UL`/`ul`/`Lu`/etc. (not a float).
        if !is_float && matches!(self.peek(), Some(b'u' | b'U' | b'l' | b'L')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'u' | b'U' | b'l' | b'L')) {
                self.pos += 1;
            }
        }
        let text = std::str::from_utf8(&self.src[self.token_start..self.pos]).unwrap_or("0");
        // Strip digit separators (`_`) for the parse.
        let cleaned: String = text.chars().filter(|c| *c != '_').collect();
        if is_float {
            // Drop a trailing float suffix before parsing the mantissa.
            let value = cleaned
                .trim_end_matches(|c| matches!(c, 'f' | 'F' | 'd' | 'D' | 'm' | 'M'))
                .parse::<f64>()
                .unwrap_or(0.0);
            TokenKind::FloatLiteral(value.to_bits())
        } else {
            TokenKind::IntLiteral(Self::parse_int_literal(&cleaned))
        }
    }

/// Parse a non-float numeric literal: decimal, hex (`0x…`), or binary (`0b…`).
/// Returns the unsigned magnitude (the token is `IntLiteral(u64)`); a leading
/// unary minus is applied later at the AST level. A trailing integral suffix
/// (`u`/`U`/`l`/`L`) is not part of the value and is ignored, mirroring C#'s
/// integral literal suffixes.
fn parse_int_literal(cleaned: &str) -> u64 {
    let lower = cleaned.to_ascii_lowercase();
    let (radix, digits) = if let Some(hex) = lower.strip_prefix("0x") {
        (16u32, hex)
    } else if let Some(bin) = lower.strip_prefix("0b") {
        (2u32, bin)
    } else {
        (10u32, lower.as_str())
    };
    let valid: String = digits.chars().take_while(|c| c.is_digit(radix)).collect();
    if valid.is_empty() {
        0
    } else {
        u64::from_str_radix(&valid, radix).unwrap_or(0)
    }
}

    /// Consume an optional suffix on a string literal: `u8`, etc. The
    /// suffix is dropped from the token value because the compact AST
    /// does not distinguish UTF-8 strings.
    fn eat_string_suffix(&mut self) {
        if self.peek() == Some(&b'u')
            && self.src.get(self.pos + 1) == Some(&b'8')
        {
            self.pos += 2;
        }
    }

    fn string_literal(&mut self) -> TokenKind {
        self.pos += 1;
        let mut value = Vec::new();
        while let Some(&b) = self.peek() {
            if b == b'"' {
                self.pos += 1;
                break;
            }
            // Escape set: \\ \" \n \t \r \0 \a \b \f \v \u0041 ...
            if b == b'\\'
                && let Some(&esc) = self.src.get(self.pos + 1)
            {
                let decoded = match esc {
                    b'n' => Some(b'\n'),
                    b't' => Some(b'\t'),
                    b'r' => Some(b'\r'),
                    b'0' => Some(0),
                    b'a' => Some(0x07),
                    b'b' => Some(0x08),
                    b'f' => Some(0x0c),
                    b'v' => Some(0x0b),
                    b'\\' => Some(b'\\'),
                    b'"' => Some(b'"'),
                    b'\'' => Some(b'\''),
                    _ => None,
                };
                if let Some(byte) = decoded {
                    value.push(byte);
                    self.pos += 2;
                    continue;
                }
                // Unicode escape: `\uXXXX` (1-4 hex digits).
                if esc == b'u'
                    && let Some(d) = self.parse_unicode_escape()
                {
                    value.push(d);
                    continue;
                }
                // Unknown escape: keep the backslash verbatim.
            }
            value.push(b);
            self.pos += 1;
        }
        self.eat_string_suffix();
        TokenKind::StringLiteral(String::from_utf8_lossy(&value).into_owned())
    }

    /// Parse `\uXXXX` where XXXX is 1-4 hex digits. Caller has already
    /// consumed `\` and `u`.
    fn parse_unicode_escape(&mut self) -> Option<u8> {
        self.pos += 2; // skip `\u`
        let mut value: u32 = 0;
        for _ in 0..4 {
            let d = match self.peek() {
                Some(b'0'..=b'9') => self.peek().unwrap() - b'0',
                Some(b'a'..=b'f') => self.peek().unwrap() - b'a' + 10,
                Some(b'A'..=b'F') => self.peek().unwrap() - b'A' + 10,
                _ => break,
            };
            value = value * 16 + d as u32;
            self.pos += 1;
        }
        Some(value as u8)
    }

    /// Interpolated string `$"..."`. The body is captured RAW (no escape
    /// processing, no `{` handling) so the parser can split it into text
    /// runs and interpolation holes. Nested string literals inside holes are
    /// handled by tracking brace depth and string state.
    fn interpolated_string(&mut self) -> TokenKind {
        self.pos += 1; // skip opening `"`
        let mut value = Vec::new();
        let mut brace_depth = 0i32;
        let mut in_string = false;
        let mut string_escape = false;
        while let Some(&b) = self.peek() {
            if in_string {
                value.push(b);
                if string_escape {
                    string_escape = false;
                } else if b == b'\\' {
                    string_escape = true;
                } else if b == b'"' {
                    in_string = false;
                }
                self.pos += 1;
                continue;
            }
            if brace_depth > 0 {
                // Inside an interpolation hole
                match b {
                    b'{' => {
                        brace_depth += 1;
                        value.push(b);
                        self.pos += 1;
                        continue;
                    }
                    b'}' => {
                        brace_depth -= 1;
                        value.push(b);
                        self.pos += 1;
                        continue;
                    }
                    b'"' => {
                        // Start of nested string literal
                        in_string = true;
                        value.push(b);
                        self.pos += 1;
                        continue;
                    }
                    _ => {
                        value.push(b);
                        self.pos += 1;
                        continue;
                    }
                }
            }
            // Outside any hole
            if b == b'"' {
                self.pos += 1;
                break;
            }
            // `{{` outside a hole is an escaped literal brace, not a
            // hole (`$"namespace {n} {{"`): without this the lexer
            // opens a phantom hole, the following `"` looks nested,
            // and the string swallows the rest of the file.
            if b == b'{' && self.src.get(self.pos + 1) == Some(&b'{') {
                value.push(b'{');
                value.push(b'{');
                self.pos += 2;
                continue;
            }
            if b == b'{' {
                brace_depth = 1;
                value.push(b);
                self.pos += 1;
                continue;
            }
            if b == b'\\' {
                // Escape sequence: copy both bytes
                value.push(b);
                self.pos += 1;
                if let Some(&next) = self.peek() {
                    value.push(next);
                    self.pos += 1;
                }
                continue;
            }
            value.push(b);
            self.pos += 1;
        }
        self.eat_string_suffix();
        TokenKind::InterpolatedString {
            raw: String::from_utf8_lossy(&value).into_owned(),
            verbatim: false,
        }
    }

    /// Interpolated verbatim string `$@"..."`. `""` is a literal `"` and
    /// backslashes are kept verbatim; the body is captured RAW.
    fn interpolated_verbatim_string(&mut self) -> TokenKind {
        self.pos += 1; // skip opening `"`
        let mut value = Vec::new();
        let mut brace_depth = 0i32;
        let mut in_string = false;
        while let Some(&b) = self.peek() {
            if in_string {
                value.push(b);
                if b == b'"' {
                    if self.src.get(self.pos + 1) == Some(&b'"') {
                        // `""` inside string = escaped quote
                        value.push(b'"');
                        self.pos += 1;
                    } else {
                        in_string = false;
                    }
                }
                self.pos += 1;
                continue;
            }
            if brace_depth > 0 {
                // Inside an interpolation hole
                match b {
                    b'{' => {
                        brace_depth += 1;
                        value.push(b);
                        self.pos += 1;
                        continue;
                    }
                    b'}' => {
                        brace_depth -= 1;
                        value.push(b);
                        self.pos += 1;
                        continue;
                    }
                    b'"' => {
                        if self.src.get(self.pos + 1) == Some(&b'"') {
                            // `""` = literal quote
                            value.push(b'"');
                            self.pos += 1;
                        } else {
                            // Start of nested string literal
                            in_string = true;
                            value.push(b);
                            self.pos += 1;
                            continue;
                        }
                        value.push(b);
                        self.pos += 1;
                        continue;
                    }
                    _ => {
                        value.push(b);
                        self.pos += 1;
                        continue;
                    }
                }
            }
            // Outside any hole
            if b == b'"' {
                if self.src.get(self.pos + 1) == Some(&b'"') {
                    value.push(b'"');
                    self.pos += 1;
                } else {
                    self.pos += 1;
                    break;
                }
            }
            // `{{` outside a hole is an escaped literal brace, not a
            // hole (same as the non-verbatim form above).
            if b == b'{' && self.src.get(self.pos + 1) == Some(&b'{') {
                value.push(b'{');
                value.push(b'{');
                self.pos += 2;
                continue;
            }
            if b == b'{' {
                brace_depth = 1;
                value.push(b);
                self.pos += 1;
                continue;
            }
            value.push(b);
            self.pos += 1;
        }
        self.eat_string_suffix();
        TokenKind::InterpolatedString {
            raw: String::from_utf8_lossy(&value).into_owned(),
            verbatim: true,
        }
    }

    /// Interpolated raw string `$"""..."""`. The body is captured RAW and the
    /// leading indent is dropped (matching the non-interpolated raw form).
    fn interpolated_raw_string(&mut self) -> TokenKind {
        self.pos += 4; // skip `$"""`
        let mut value = Vec::new();
        while self.pos < self.src.len() {
            if self.peek() == Some(&b'"')
                && self.src.get(self.pos + 1) == Some(&b'"')
                && self.src.get(self.pos + 2) == Some(&b'"')
            {
                self.pos += 3;
                while self.peek() == Some(&b'"') {
                    value.push(b'"');
                    self.pos += 1;
                }
                break;
            }
            value.push(self.src[self.pos]);
            self.pos += 1;
        }
        self.eat_string_suffix();
        TokenKind::InterpolatedString {
            raw: String::from_utf8_lossy(&value).trim().to_string(),
            verbatim: true,
        }
    }

    /// Verbatim string literal: only `"` ends it, and `""` is a literal `"`.
    /// Backslashes are kept verbatim. Used for both `@"..."` and `$@"..."`.
    fn verbatim_string_literal(&mut self) -> TokenKind {
        self.pos += 1;
        let mut value = Vec::new();
        while let Some(&b) = self.peek() {
            if b == b'"' {
                if self.src.get(self.pos + 1) == Some(&b'"') {
                    value.push(b'"');
                    self.pos += 2;
                    continue;
                }
                self.pos += 1;
                break;
            }
            // Newlines (including `\r\n`) are allowed in verbatim strings.
            value.push(b);
            self.pos += 1;
        }
        self.eat_string_suffix();
        TokenKind::StringLiteral(String::from_utf8_lossy(&value).into_owned())
    }

    fn char_literal(&mut self) -> TokenKind {
        self.pos += 1; // Skip opening quote
        // C# `char` is a UTF-16 code unit (up to 0xFFFF); values are
        // carried in the u64 literal payload.
        let value: u64 = if let Some(&b) = self.peek() {
            let v = if b == b'\\' {
                self.pos += 1;
                match self.peek() {
                    Some(b'n') => {
                        self.pos += 1;
                        b'\n' as u64
                    }
                    Some(b't') => {
                        self.pos += 1;
                        b'\t' as u64
                    }
                    Some(b'r') => {
                        self.pos += 1;
                        b'\r' as u64
                    }
                    Some(b'0') => {
                        self.pos += 1;
                        0
                    }
                    Some(b'\\') => {
                        self.pos += 1;
                        b'\\' as u64
                    }
                    Some(b'\'') => {
                        self.pos += 1;
                        b'\'' as u64
                    }
                    // `\uXXXX` — exactly 4 hex digits.
                    Some(b'u') => {
                        self.pos += 1;
                        Self::lex_hex_escape(&self.src, &mut self.pos, 4)
                    }
                    // `\UXXXXXXXX` — exactly 8 hex digits.
                    Some(b'U') => {
                        self.pos += 1;
                        Self::lex_hex_escape(&self.src, &mut self.pos, 8)
                    }
                    // `\xH..HHHH` — 1 to 4 hex digits.
                    Some(b'x') => {
                        self.pos += 1;
                        let mut val = 0u64;
                        let mut digits = 0u8;
                        while digits < 4 {
                            match self.src.get(self.pos) {
                                Some(c) if c.is_ascii_hexdigit() => {
                                    val = val * 16
                                        + (*c as char).to_digit(16).unwrap_or(0) as u64;
                                    self.pos += 1;
                                    digits += 1;
                                }
                                _ => break,
                            }
                        }
                        val
                    }
                    _ => {
                        // Unknown escape: mirror the previous `?`
                        // degradation, then consume the char.
                        self.pos += 1;
                        b'?' as u64
                    }
                }
            } else if (b as char).is_ascii() {
                self.pos += 1;
                b as u64
            } else {
                // Multi-byte UTF-8 literal: decode the scalar value.
                let rest = &self.src[self.pos..];
                let s = std::str::from_utf8(rest).unwrap_or("");
                match s.chars().next() {
                    Some(ch) => {
                        self.pos += ch.len_utf8();
                        ch as u64
                    }
                    None => {
                        self.pos += 1;
                        b'?' as u64
                    }
                }
            };
            v
        } else {
            0
        };
        // Skip closing quote
        if self.peek() == Some(&b'\'') {
            self.pos += 1;
        }
        TokenKind::IntLiteral(value)
    }

    /// Read exactly `n` hex digits at `pos`, advancing past them (fewer
    /// when input runs out). Used by `\u` / `\U` char escapes.
    fn lex_hex_escape(src: &[u8], pos: &mut usize, n: u8) -> u64 {
        let mut val = 0u64;
        for _ in 0..n {
            match src.get(*pos) {
                Some(c) if c.is_ascii_hexdigit() => {
                    val = val * 16 + (*c as char).to_digit(16).unwrap_or(0) as u64;
                    *pos += 1;
                }
                _ => break,
            }
        }
        val
    }

    fn make(&self, kind: TokenKind) -> Token {
        Token {
            kind,
            span: Span::new(self.token_start as u32, self.pos as u32),
        }
    }

    fn peek(&self) -> Option<&u8> {
        self.src.get(self.pos)
    }
}
