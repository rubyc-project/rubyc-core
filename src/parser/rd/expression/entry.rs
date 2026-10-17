//! Expressions: precedence climbing, unary, postfix chains, primaries.

use crate::models::{Access, BinOp, Expr, InterpPart, Keyword, Literal, Span, Stmt, SwitchExprArm, TokenKind, UnOp};

use super::super::{Parser, binop_precedence, expr_span, expr_span_last};
use super::types::{parse_interp_hole, read_unicode_escape};
impl Parser {
    /// Precedence-climbing entry point: `cond ? a : b` wraps binary.
    /// Also handles assignment expressions (`=`, `+=`, `-=`, etc.) which
    /// have lower precedence than ternary.
    pub(in crate::parser::rd) fn expression(&mut self) -> Option<Expr> {
        let start = self.peek().span;
        // Detect lambda expressions: `static (params) =>`, `(params) =>`,
        // `name =>`. Must be checked before `binary_expr` which doesn't
        // know about lambda prefixes.
        if let Some(lambda) = self.try_lambda() {
            return Some(lambda);
        }
        let cond = self.binary_expr(0)?;
        if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Switch)) {
            let sw = self.switch_expression(cond, start)?;
            return self.binary_tail(sw, 0);
        }
        if matches!(self.peek().kind, TokenKind::Keyword(Keyword::With)) {
            let wh = self.with_expression(cond, start)?;
            return self.binary_tail(wh, 0);
        }
        if self.peek().kind == TokenKind::Question {
            self.bump();
            let then_expr = self.expression()?;
            self.expect(&TokenKind::Colon, "to separate ternary branches")?;
            let else_expr = self.expression()?;
            let cond = Expr::Ternary {
                cond: Box::new(cond),
                then_expr: Box::new(then_expr),
                else_expr: Box::new(else_expr),
                span: start.to(self.peek_at(0).span),
            };
            // After the ternary, check for assignment.
            if self.peek().kind == TokenKind::Assign {
                self.bump();
                let rhs = self.expression()?;
                return Some(Expr::Assign {
                    target: Box::new(cond),
                    value: Box::new(rhs),
                    span: start.to(expr_span_last(&self.tokens[..self.pos])),
                });
            }
            return Some(cond);
        }
        // Assignment expression: `target = value`.
        if self.peek().kind == TokenKind::Assign {
            self.bump();
            let rhs = self.expression()?;
            return Some(Expr::Assign {
                target: Box::new(cond),
                value: Box::new(rhs),
                span: start.to(expr_span_last(&self.tokens[..self.pos])),
            });
        }
        // Compound assignment expression: `target op= value` (e.g.
        // `return _headers ??= new ...;`). Desugar exactly like statement
        // position (`x op= rhs` ≡ `x = x op rhs`): bare names become
        // `Assign` (lowering stores locals and implicit-`this` fields);
        // complex targets become `AssignComplex` (member/index stores).
        let compound = match self.peek().kind {
            TokenKind::PlusAssign => Some(BinOp::Add),
            TokenKind::MinusAssign => Some(BinOp::Sub),
            TokenKind::StarAssign => Some(BinOp::Mul),
            TokenKind::SlashAssign => Some(BinOp::Div),
            TokenKind::PercentAssign => Some(BinOp::Rem),
            TokenKind::AmpAssign => Some(BinOp::BitAnd),
            TokenKind::PipeAssign => Some(BinOp::BitOr),
            TokenKind::CaretAssign => Some(BinOp::BitXor),
            TokenKind::ShlAssign => Some(BinOp::Shl),
            TokenKind::ShrAssign => Some(BinOp::Shr),
            TokenKind::ShrShrAssign => Some(BinOp::ShrShr),
            TokenKind::QuestionQuestionAssign => Some(BinOp::NullCoalesce),
            _ => None,
        };
        if let Some(op) = compound {
            self.bump();
            let rhs = self.expression()?;
            let end = expr_span_last(&self.tokens[..self.pos]);
            return Some(Self::compound_to_assign(cond, op, rhs, start, end));
        }
        Some(cond)
    }

    /// Desugar `target op rhs` exactly like statement position
    /// (`x op rhs` ≡ `x = x op rhs`): bare names become `Assign`
    /// (lowering stores locals and implicit-`this` fields); complex
    /// targets become `AssignComplex` (member/index stores).
    fn compound_to_assign(
        target: Expr,
        op: BinOp,
        rhs: Expr,
        start: Span,
        end: Span,
    ) -> Expr {
        let value = Expr::Binary {
            op,
            lhs: Box::new(target.clone()),
            rhs: Box::new(rhs),
            span: start.to(end),
        };
        if matches!(target, Expr::Ident { .. }) {
            Expr::Assign {
                target: Box::new(target),
                value: Box::new(value),
                span: start.to(end),
            }
        } else {
            Expr::AssignComplex {
                lhs: Box::new(target),
                value: Box::new(value),
                span: start.to(end),
            }
        }
    }

    /// Parse an interpolated-string body (captured raw by the lexer) into
    /// `InterpPart`s. `verbatim` (true for `@`/`$@`/`$"""` forms) means
    /// backslashes are literal (only `\uXXXX` is an escape); otherwise
    /// (`decode` true) the usual escape set is decoded. Holes are `{ ... }`
    /// regions where `{{`/`}}` are escaped literals. Each hole is parsed as a
    /// full expression, so holes may themselves be interpolated strings
    /// (recursion).
    pub(in crate::parser::rd) fn build_interp_parts(&mut self, raw: &str, verbatim: bool) -> Option<Vec<InterpPart>> {
        let decode = !verbatim;
        let bytes = raw.as_bytes();
        let mut parts: Vec<InterpPart> = Vec::new();
        let mut text = String::new();
        let mut i = 0usize;

        let flush_text = |parts: &mut Vec<InterpPart>, text: &mut String| {
            if !text.is_empty() {
                parts.push(InterpPart::Text(std::mem::take(text)));
            }
        };

        while i < bytes.len() {
            let b = bytes[i];
            // Escaped braces: `{{` -> `{`, `}}` -> `}`.
            if b == b'{' && bytes.get(i + 1) == Some(&b'{') {
                text.push('{');
                i += 2;
                continue;
            }
            if b == b'}' && bytes.get(i + 1) == Some(&b'}') {
                text.push('}');
                i += 2;
                continue;
            }
            if b == b'{' {
                // A hole: read its balanced brace body.
                let mut depth = 1usize;
                let mut j = i + 1;
                while j < bytes.len() && depth > 0 {
                    match bytes[j] {
                        b'{' => depth += 1,
                        b'}' => depth -= 1,
                        _ => {}
                    }
                    j += 1;
                }
                flush_text(&mut parts, &mut text);
                let start = i + 1;
                let end = if j > i + 1 { j - 1 } else { start };
                let body = std::str::from_utf8(&bytes[start..end]).unwrap_or("");
                let hole = parse_interp_hole(body)?;
                parts.push(InterpPart::Hole(Box::new(hole)));
                i = j;
                continue;
            }
            if b == b'\\' {
                if decode {
                    // Standard escape set (non-verbatim interpolated strings).
                    if let Some(&esc) = bytes.get(i + 1) {
                        if let Some(ch) = match esc {
                            b'n' => Some('\n'),
                            b't' => Some('\t'),
                            b'r' => Some('\r'),
                            b'0' => Some('\0'),
                            b'a' => Some('\x07'),
                            b'b' => Some('\x08'),
                            b'f' => Some('\x0c'),
                            b'v' => Some('\x0b'),
                            b'\\' => Some('\\'),
                            b'"' => Some('"'),
                            b'\'' => Some('\''),
                            b'u' => read_unicode_escape(bytes, i + 2).map(|(c, consumed)| {
                                i += 2 + consumed; // advance past the escape
                                c
                            }),
                            _ => None,
                        } {
                            text.push(ch);
                            if esc != b'u' {
                                i += 2;
                            }
                            continue;
                        }
                    }
                } else {
                    // Verbatim/raw forms: only `\uXXXX` is an escape.
                    if bytes.get(i + 1) == Some(&b'u') {
                        match read_unicode_escape(bytes, i + 2) {
                            Some((ch, consumed)) => {
                                text.push(ch);
                                i += 2 + consumed;
                                continue;
                            }
                            None => {
                                text.push('\\');
                                i += 1;
                                continue;
                            }
                        }
                    }
                }
                // Unknown escape: keep the backslash verbatim.
                text.push('\\');
                i += 1;
                continue;
            }
            text.push(char::from(b));
            i += 1;
        }
        flush_text(&mut parts, &mut text);
        Some(parts)
    }

    /// `subject with { Key = value, ... }` — C# 9 non-destructive
    /// mutation expression. The result is the subject itself; the body
    /// is consumed but the per-field updates are dropped because the
    /// compact AST does not model records.
    fn with_expression(&mut self, _subject: Expr, start: Span) -> Option<Expr> {
        self.bump(); // 'with'
        if self.peek().kind == TokenKind::LeftBrace {
            self.bump();
            while self.peek().kind != TokenKind::RightBrace
                && self.peek().kind != TokenKind::Eof
            {
                if self.eat(&TokenKind::Comma).is_some() {
                    continue;
                }
                let _ = self.expect_identifier("as with-property name");
                if self.eat(&TokenKind::Assign).is_some() {
                    let mark = self.pos;
                    if self.expression().is_none() && self.pos == mark {
                        self.bump();
                    }
                }
            }
            self.expect(&TokenKind::RightBrace, "to close with body")?;
        }
        Some(Expr::New {
            class_name: "with".to_owned(),
            args: Vec::new(),
            init: Vec::new(),
            span: start.to(self.peek_at(0).span),
        })
    }

    /// `subject switch { pattern1, pattern2, ... => body, ..., _ => default }`.
    /// The subject has already been parsed and is passed in.
    fn switch_expression(&mut self, subject: Expr, start: Span) -> Option<Expr> {
        self.bump(); // 'switch'
        self.expect(&TokenKind::LeftBrace, "to open switch expression body")?;
        let mut arms = Vec::new();
        while self.peek().kind != TokenKind::RightBrace && self.peek().kind != TokenKind::Eof {
            // Parse one or more `or`-separated pattern labels.
            let mut labels = Vec::new();
            loop {
                let label = self.parse_switch_expr_pattern_label();
                labels.push(label);
                // `or` chains another pattern; `,` chains another arm's
                // labels (handled by the outer arm loop).
                if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Or_)) {
                    self.bump();
                } else {
                    break;
                }
            }
            // Optional `when` clause: `pattern when expr`.
            // A bare `name =>` guard would misparse as a single-param
            // lambda, consuming the arm's own `=>` — take the name
            // alone. Every other shape parses as a full expression
            // (parenthesized lambdas inside still work; only a top-level
            // bare name directly before `=>` is ambiguous, and there C#
            // takes the arm arrow).
            if matches!(self.peek().kind, TokenKind::Keyword(Keyword::When_)) {
                self.bump();
                if self.peek_is_name(0)
                    && self.peek_at(1).kind == TokenKind::FatArrow
                {
                    self.bump(); // guard name; the `=>` stays the arm's
                } else {
                    let _ = self.expression();
                }
            }
            // `=> body`
            self.expect(&TokenKind::FatArrow, "in switch expression arm")?;
            let body = self.expression()?;
            // Optional trailing comma between arms.
            self.eat(&TokenKind::Comma);
            arms.push(SwitchExprArm {
                labels,
                body: Box::new(body),
            });
        }
        self.expect(&TokenKind::RightBrace, "to close switch expression body")?;
        Some(Expr::SwitchExpr {
            subject: Box::new(subject),
            arms,
            span: start.to(self.peek_at(0).span),
        })
    }

    /// Parse one pattern label inside a switch expression. Returns a string
    /// representation since the AST does not model the full pattern grammar.
    fn parse_switch_expr_pattern_label(&mut self) -> String {
        let start = self.peek().span;
        // `nameof(X)` — constant pattern. (`sizeof`/`typeof` lex as
        // identifiers and already route through the type branch.)
        if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Nameof_)) {
            self.bump(); // 'nameof'
            if self.peek().kind == TokenKind::LeftParen {
                self.skip_balanced_bracket();
            }
            return "nameof(..)".to_owned();
        }
        // `default` / `default(T)` — default literal, optionally typed
        // (mirrors `primary`; e.g. `default(char) => ...`).
        if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Default)) {
            self.bump(); // 'default'
            if self.peek().kind == TokenKind::LeftParen {
                self.bump(); // '('
                let mut type_name = String::new();
                // Reuse the full type grammar when it looks like one;
                // otherwise skip to `)` — either way the value is the
                // type's default and the label stays inert.
                let mark = self.pos;
                let err_mark = self.errors.len();
                if let Some(t) = self.type_name() {
                    type_name = t;
                } else {
                    self.pos = mark;
                    self.errors.truncate(err_mark);
                    while !matches!(
                        self.peek().kind,
                        TokenKind::RightParen | TokenKind::Eof
                    ) {
                        self.bump();
                    }
                }
                let _ = self.expect(&TokenKind::RightParen, "to close default type");
                if type_name.is_empty() {
                    return "default".to_owned();
                }
                return format!("default({type_name})");
            }
            return "default".to_owned();
        }
        // Negated pattern: `not <pattern>` (e.g. `not null`). Recurse
        // for the inner pattern so `not not x` and `not (A, B)` work;
        // mirrors the statement-level `case_label`.
        if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Not_)) {
            self.bump(); // 'not'
            let inner = self.parse_switch_expr_pattern_label();
            return format!("not {inner}");
        }
        // Discard pattern.
        if let TokenKind::Identifier(name) = &self.peek().kind
            && name == "_"
        {
            self.bump();
            return "_".to_owned();
        }
        // Relational pattern: `<rel> expr`.
        let rel_op = match self.peek().kind {
            TokenKind::GreaterEq => Some(">="),
            TokenKind::LessEq => Some("<="),
            TokenKind::Greater => Some(">"),
            TokenKind::Less => Some("<"),
            _ => None,
        };
        if let Some(op) = rel_op {
            self.bump();
            // Consume the rest of the pattern as a string.
            let mut label = format!("{op} ");
            let mut depth = 0i32;
            while self.peek().kind != TokenKind::Comma
                && self.peek().kind != TokenKind::FatArrow
                && self.peek().kind != TokenKind::RightBrace
                && self.peek().kind != TokenKind::Eof
            {
                if matches!(self.peek().kind, TokenKind::LeftParen | TokenKind::LeftBracket | TokenKind::LeftBrace) {
                    depth += 1;
                } else if matches!(self.peek().kind, TokenKind::RightParen | TokenKind::RightBracket | TokenKind::RightBrace) {
                    depth -= 1;
                }
                if depth <= 0
                    && matches!(self.peek().kind, TokenKind::Keyword(Keyword::And_) | TokenKind::Keyword(Keyword::Or_))
                {
                    break;
                }
                let s = format!("{:?}", self.peek().kind);
                let s = s.trim_matches('\'').to_string();
                label.push_str(&s);
                label.push(' ');
                self.bump();
            }
            // Eat the rest of the `and X` / `or X` chain.
            while matches!(self.peek().kind, TokenKind::Keyword(Keyword::And_) | TokenKind::Keyword(Keyword::Or_)) {
                let op = if matches!(self.peek().kind, TokenKind::Keyword(Keyword::And_)) { "and" } else { "or" };
                self.bump();
                label.push_str(&format!("{op} "));
                while self.peek().kind != TokenKind::Comma
                    && self.peek().kind != TokenKind::FatArrow
                    && self.peek().kind != TokenKind::RightBrace
                    && self.peek().kind != TokenKind::Eof
                {
                    let s = format!("{:?}", self.peek().kind);
                    let s = s.trim_matches('\'').to_string();
                    label.push_str(&s);
                    label.push(' ');
                    self.bump();
                }
            }
            let _ = start;
            return label.trim().to_owned();
        }
        // Cast in arm position: `(int)X.Y` — parens holding a single
        // type (no top-level comma) followed by an operand are a cast,
        // not a positional pattern. Same disambiguation as `case`
        // labels and `is`-patterns; without it the cast is skipped as
        // a pattern and the member access breaks the arm.
        if self.peek().kind == TokenKind::LeftParen && !self.paren_holds_comma() {
            let mark = self.pos;
            let err_mark = self.errors.len();
            self.bump(); // '('
            // Only a plain identifier-ish type qualifies; anything else
            // rewinds to the pattern handling below.
            let mut tn: Option<String> = None;
            if matches!(self.peek().kind, TokenKind::Identifier(_)) {
                let first = self.bump();
                let TokenKind::Identifier(t) = first.kind else {
                    unreachable!()
                };
                let mut full = t;
                let mut ok = true;
                while self.peek().kind == TokenKind::Dot {
                    self.bump();
                    match self.peek().kind.clone() {
                        TokenKind::Identifier(seg) => {
                            self.bump();
                            full.push('.');
                            full.push_str(&seg);
                        }
                        _ => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok && self.peek().kind == TokenKind::RightParen {
                    self.bump(); // ')'
                    tn = Some(full);
                }
            }
            if let Some(type_name) = tn {
                // A cast label: consume the operand up to the arm
                // terminator (labels are inert strings; the operand may
                // be a member path, literal, or call).
                let mut label = format!("cast {type_name}");
                let mut depth = 0i32;
                loop {
                    let terminal = match &self.peek().kind {
                        TokenKind::FatArrow
                        | TokenKind::Comma
                        | TokenKind::Eof => true,
                        TokenKind::RightBrace => depth == 0,
                        TokenKind::Keyword(Keyword::And_)
                        | TokenKind::Keyword(Keyword::Or_)
                        | TokenKind::Keyword(Keyword::When_) => depth == 0,
                        TokenKind::LeftParen
                        | TokenKind::LeftBracket
                        | TokenKind::LeftBrace => {
                            depth += 1;
                            false
                        }
                        TokenKind::RightParen | TokenKind::RightBracket => {
                            depth = depth.saturating_sub(1);
                            false
                        }
                        _ => false,
                    };
                    if terminal {
                        break;
                    }
                    let s = format!("{:?}", self.peek().kind);
                    label.push_str(s.trim_matches('\''));
                    label.push(' ');
                    self.bump();
                }
                return label.trim().to_owned();
            } else {
                self.pos = mark;
                self.errors.truncate(err_mark);
            }
        }
        // Property / positional / list / discard-as-bracket patterns.
        if self.peek().kind == TokenKind::LeftBrace
            || self.peek().kind == TokenKind::LeftBracket
            || self.peek().kind == TokenKind::LeftParen
        {
            let open = self.pos;
            self.skip_balanced_bracket();
            // Inner designations (`{ P: T x }`) bind in the enclosing
            // scope, like the trailing one below.
            self.queue_inner_designations(open);
            // Optional variable binding (`{ ... } x`): a value in the
            // enclosing scope, like type-pattern bindings below. The
            // name may be a contextual keyword (`{ ... } record`).
            if let Some((binding, span)) = self.take_pattern_binding() {
                self.pre_pending.push(Stmt::Let {
                    is_byref: false,
                    type_name: String::from("var"),
                    name: binding,
                    init: Expr::Literal(Literal::Int(0)),
                    access: Access::Default,
                    span: start.to(span),
                });
            }
            return "<pattern>".to_owned();
        }
        // Type pattern, type-property pattern, or constant.
        if matches!(self.peek().kind, TokenKind::Identifier(_)) {
            // Consume the type segment (identifier + dots + generics + array).
            let mut label = String::new();
            let s = format!("{:?}", self.peek().kind);
            let s = s.trim_matches('\'').to_string();
            label.push_str(&s);
            label.push(' ');
            self.bump();
            // Generic arguments: `KeyValuePair<K, V>`, `Dictionary<K,V>`.
            // Gated (balanced + type-ish follower) so `a < b` still
            // parses as a comparison for the binary fold below.
            // (`end` is a lookahead offset: bump that many times.)
            if let Some(end) = self.generic_args_end_at(0) {
                if self.generic_follow_ok(end) {
                    for _ in 0..end {
                        self.bump();
                    }
                    label.push_str("<...>");
                    label.push(' ');
                }
            }
            // Dotted segments (each with its own optional generics).
            while self.peek().kind == TokenKind::Dot {
                self.bump();
                let s = format!("{:?}", self.peek().kind);
                let s = s.trim_matches('\'').to_string();
                label.push('.');
                label.push_str(&s);
                label.push(' ');
                self.bump();
                if let Some(end) = self.generic_args_end_at(0) {
                    if self.generic_follow_ok(end) {
                        for _ in 0..end {
                            self.bump();
                        }
                        label.push_str("<...>");
                        label.push(' ');
                    }
                }
            }
            // Optional positional pattern: `Type(...)`.
            if self.peek().kind == TokenKind::LeftParen {
                let open = self.pos;
                self.skip_balanced_bracket();
                self.queue_inner_designations(open);
            }
            // Property pattern: `Type { ... }` (after the optional `Type(...)`).
            if self.peek().kind == TokenKind::LeftBrace {
                let open = self.pos;
                self.skip_balanced_bracket();
                self.queue_inner_designations(open);
            }
            // Array rank suffixes: `string[] values`.
            while self.peek().kind == TokenKind::LeftBracket {
                self.skip_balanced_bracket();
            }
            // Nullable suffix: `T? x`.
            if self.peek().kind == TokenKind::Question {
                self.bump();
            }
            // Optional variable binding (`Type x`): a value in the
            // enclosing scope — expose it like `is`-pattern bindings.
            // (`_` discards need no declaration; anything else does.)
            // The name may be a contextual keyword (`case T record:`).
            if let Some((binding, span)) = self.take_pattern_binding() {
                label.push_str(&binding);
                label.push(' ');
                if binding != "_" {
                    self.pre_pending.push(Stmt::Let {
                        is_byref: false,
                        type_name: String::from("var"),
                        name: binding,
                        init: Expr::Literal(Literal::Int(0)),
                        access: Access::Default,
                        span: start.to(span),
                    });
                }
            }
            // Constant-expression continuation (`A >> B`, `X + Y`):
            // fold `op operand` pairs into the label. Arm terminators
            // (`=>`, `,`, `}`, `or`, `and`, `when`) always stop the
            // fold, so nothing a later stage needs is ever consumed.
            self.fold_pattern_binary_tail(&mut label);
            return label.trim().to_owned();
        }
        // Constant pattern (literal, etc.). A leading `-`/`+` on a
        // numeric literal is part of the constant (`-1 => ...`), not a
        // binary operator — consume both as one pattern.
        let mut sign = String::new();
        if matches!(
            self.peek().kind,
            TokenKind::Minus | TokenKind::Plus
        ) && matches!(self.peek_at(1).kind, TokenKind::IntLiteral(_)) {
            let s = format!("{:?}", self.peek().kind);
            sign.push_str(s.trim_matches('\''));
            sign.push(' ');
            self.bump(); // sign
        }
        let s = format!("{:?}", self.peek().kind);
        let s = s.trim_matches('\'').to_string();
        self.bump();
        let s = if sign.is_empty() {
            s
        } else {
            format!("{sign}{s}")
        };
        // Consume any literal suffix (`L`, `U`, `UL`, `f`, `m`, `d`).
        if matches!(&self.peek().kind, TokenKind::Identifier(suf)
            if matches!(suf.as_str(), "L" | "l" | "U" | "u" | "UL" | "Ul" | "uL" | "ul" | "f" | "F" | "m" | "M" | "d" | "D"))
        {
            self.bump();
        }
        s
    }

    /// Fold `op operand` pairs after a switch-expression arm pattern
    /// into its label string (constant-expression patterns like
    /// `MethodDef >> RowIdBitCount`). Stops at arm terminators (`=>`,
    /// `,`, `}`, `or`, `and`, `when`) and EOF, leaving them for the
    /// arm loop — so only tokens that would otherwise be an immediate
    /// parse error are ever consumed.
    fn fold_pattern_binary_tail(&mut self, label: &mut String) {
        /// True for tokens that continue a constant expression. Excludes
        /// everything the arm loop or guard handling needs: `=>`, `,`,
        /// braces, `or`/`and`/`when`, and `=`/`!`/`:`-family tokens that
        /// never continue an expression.
        fn is_binop(kind: &TokenKind) -> bool {
            matches!(
                kind,
                TokenKind::Plus
                    | TokenKind::Minus
                    | TokenKind::Star
                    | TokenKind::Slash
                    | TokenKind::Percent
                    | TokenKind::Caret
                    | TokenKind::Amp
                    | TokenKind::Pipe
                    | TokenKind::Shl
                    | TokenKind::Shr
                    | TokenKind::ShrShr
                    | TokenKind::Less
                    | TokenKind::Greater
                    | TokenKind::LessEq
                    | TokenKind::GreaterEq
                    | TokenKind::EqEq
                    | TokenKind::NotEq
                    | TokenKind::AmpAmp
                    | TokenKind::PipePipe
                    | TokenKind::QuestionQuestion
            )
        }
        loop {
            if !is_binop(&self.peek().kind) {
                break;
            }
            let s = format!("{:?}", self.peek().kind);
            label.push_str(s.trim_matches('\''));
            label.push(' ');
            self.bump(); // operator
            // One operand: tokens until a terminator or the next
            // top-level binary op. Bracket depth is tracked so
            // `Foo(a, b)`-style operands survive intact.
            let mut depth = 0i32;
            let mut any = false;
            loop {
                let stop = match &self.peek().kind {
                    TokenKind::FatArrow | TokenKind::Comma | TokenKind::Eof => true,
                    TokenKind::Keyword(Keyword::And_)
                    | TokenKind::Keyword(Keyword::Or_)
                    | TokenKind::Keyword(Keyword::When_) => depth == 0,
                    TokenKind::LeftParen
                    | TokenKind::LeftBracket
                    | TokenKind::LeftBrace => {
                        depth += 1;
                        false
                    }
                    TokenKind::RightParen
                    | TokenKind::RightBracket
                    | TokenKind::RightBrace => {
                        if depth == 0 {
                            true
                        } else {
                            depth -= 1;
                            false
                        }
                    }
                    k => is_binop(k) && depth == 0,
                };
                if stop {
                    break;
                }
                let s = format!("{:?}", self.peek().kind);
                label.push_str(s.trim_matches('\''));
                label.push(' ');
                self.bump();
                any = true;
            }
            if !any {
                break;
            }
        }
    }

    pub(in crate::parser::rd) fn binary_expr(&mut self, min_precedence: u8) -> Option<Expr> {
        let lhs = self.unary()?;
        self.binary_tail(lhs, min_precedence)
    }

    /// Binary-operator continuation for an already-parsed left side:
    /// `switch {...} ?? x`, `with {...} + y` — the switch/with forms
    /// return before the operator loop, so feed their result back
    /// through it instead of stranding trailing operators.
    pub(in crate::parser::rd) fn binary_tail(&mut self, mut lhs: Expr, min_precedence: u8) -> Option<Expr> {
        loop {
            // `is` type-check / pattern operator (low precedence, non-associative).
            // Supports `value is TypeName`, `value is TypeName varName`
            // (type pattern with binding), `value is > 100` (relational
            // pattern), and `value is null` / `value is not null`.
            if matches!(
                self.peek().kind,
                TokenKind::Keyword(Keyword::Is_)
            ) && min_precedence <= 1
            {
                let is_span = self.peek().span;
                self.bump();
                // Optional `not` for `is not null` style patterns.
                let negated = matches!(self.peek().kind, TokenKind::Keyword(Keyword::Not_));
                if negated {
                    self.bump();
                }
                // Build the pattern rhs by parsing one primary pattern, then
                // chaining any `and`/`or` pattern conjunctions/disjunctions.
                let mut pattern = self.parse_is_pattern_primary()?;
                loop {
                    match self.peek().kind {
                        TokenKind::Keyword(Keyword::And_) => {
                            self.bump();
                            let rhs = self.parse_is_pattern_primary()?;
                            let span =
                                expr_span(&pattern).to(expr_span_last(&self.tokens[..self.pos]));
                            pattern = Expr::Binary {
                                op: BinOp::And,
                                lhs: Box::new(pattern),
                                rhs: Box::new(rhs),
                                span,
                            };
                        }
                        TokenKind::Keyword(Keyword::Or_) => {
                            self.bump();
                            let rhs = self.parse_is_pattern_primary()?;
                            let span =
                                expr_span(&pattern).to(expr_span_last(&self.tokens[..self.pos]));
                            pattern = Expr::Binary {
                                op: BinOp::Or,
                                lhs: Box::new(pattern),
                                rhs: Box::new(rhs),
                                span,
                            };
                        }
                        _ => break,
                    }
                }
                let span = is_span.to(expr_span_last(&self.tokens[..self.pos]));
                // Negation wraps the whole test (`x is not null` is
                // `!(x is null)` — the subject belongs inside, not
                // dropped (the old form yielded `!(pattern)`, always
                // true for `is not null`).
                let test = Expr::Binary {
                    op: BinOp::Is,
                    lhs: Box::new(lhs),
                    rhs: Box::new(pattern),
                    span,
                };
                lhs = if negated {
                    Expr::Unary {
                        op: UnOp::Not,
                        inner: Box::new(test),
                        span,
                    }
                } else {
                    test
                };
                continue;
            }
            // `as` cast operator (low precedence, non-associative).
            if matches!(
                self.peek().kind,
                TokenKind::Keyword(Keyword::As_)
            ) && min_precedence <= 1
            {
                let as_span = self.peek().span;
                self.bump();
                // Full type: `as string[]`, `as Ns.T?`, etc.
                let type_name = self.type_name()?;
                let span = as_span.to(expr_span_last(&self.tokens[..self.pos]));
                lhs = Expr::Cast {
                    type_name,
                    expr: Box::new(lhs),
                    span,
                };
                continue;
            }
            if let Some((precedence, op)) = binop_precedence(&self.peek().kind) {
                if precedence < min_precedence {
                    break;
                }
                self.bump();
                let rhs = self.binary_expr(precedence + 1)?; // left-associative
                let span = expr_span(&lhs).to(expr_span(&rhs));
                lhs = Expr::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                    span,
                };
                continue;
            }
            // Stop at case/colon/semicolon to avoid consuming switch labels.
            if matches!(
                self.peek().kind,
                TokenKind::Keyword(Keyword::Case)
                    | TokenKind::Colon
                    | TokenKind::Semicolon
                    | TokenKind::Keyword(Keyword::When_)
            ) {
                break;
            }
            break;
        }
        Some(lhs)
    }

    /// Lookahead: balanced `[...]` at the cursor followed by a lambda
    /// shape (`async`/`static` optional, then `(params) =>` or `name =>`).
    /// Pure lookahead — consumes nothing on any outcome.
    fn lambda_shape_after_attrs(&self) -> bool {
        let mut depth = 0i32;
        let mut k = 0usize;
        loop {
            match self.peek_at(k).kind {
                TokenKind::LeftBracket => depth += 1,
                TokenKind::RightBracket => {
                    depth -= 1;
                    if depth == 0 {
                        k += 1;
                        break;
                    }
                }
                TokenKind::Eof => return false,
                _ => {}
            }
            k += 1;
        }
        // Optional `async` / `static` prefixes (either order, both optional).
        for _ in 0..2 {
            if matches!(
                self.peek_at(k).kind,
                TokenKind::Keyword(Keyword::Async) | TokenKind::Keyword(Keyword::Static)
            ) {
                k += 1;
            }
        }
        if self.peek_at(k).kind == TokenKind::LeftParen {
            // Balanced parens then `=>`?
            let mut d = 1i32;
            let mut j = k + 1;
            while d > 0 && self.peek_at(j).kind != TokenKind::Eof {
                match self.peek_at(j).kind {
                    TokenKind::LeftParen => d += 1,
                    TokenKind::RightParen => d -= 1,
                    _ => {}
                }
                j += 1;
            }
            self.peek_at(j).kind == TokenKind::FatArrow
        } else {
            self.peek_is_name(k) && self.peek_at(k + 1).kind == TokenKind::FatArrow
        }
    }

    /// Parse a lambda expression: `(params) => body`, `name => body`, or
    /// `static (params) => body`, `static name => body`. Returns `Some(expr)`
    /// if the current position looks like a lambda, otherwise `None`.
    pub(in crate::parser::rd) fn try_lambda(&mut self) -> Option<Expr> {
        // Attributes on lambdas (C# 10+): `[Attr(...)] (params) => body`.
        // Consume the list only when a lambda shape follows the closing
        // bracket (a bare `[` otherwise starts a collection/indexer).
        // Attributes carry no lowering meaning and are dropped.
        if self.peek().kind == TokenKind::LeftBracket && self.lambda_shape_after_attrs() {
            let _ = self.attribute_lists();
        }
        // Async modifier: `async (params) =>`, `async name =>`,
        // `async static ...` (either order). `async` alone as a param
        // name (`async =>`) is handled as a name-lambda below.
        if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Async)) {
            // `async =>`: param literally named `async`.
            if self.peek_at(1).kind == TokenKind::FatArrow {
                let start = self.peek().span;
                self.bump(); // 'async' as name
                self.expect(&TokenKind::FatArrow, "in lambda")?;
                self.consume_lambda_body();
                return Some(Expr::Ident {
                    name: "lambda".to_owned(),
                    span: start.to(self.peek_at(0).span),
                });
            }
            // Look past `async [static]` for a lambda shape; only
            // consume when it matches (bare `async` may be an identifier).
            let mut o = 1usize;
            if matches!(
                self.peek_at(o).kind,
                TokenKind::Keyword(Keyword::Static)
            ) {
                o += 1;
            }
            let shape = if self.peek_at(o).kind == TokenKind::LeftParen {
                // Balanced parens then `=>`?
                let mut d = 1i32;
                let mut k = o + 1;
                while d > 0 && self.peek_at(k).kind != TokenKind::Eof {
                    match self.peek_at(k).kind {
                        TokenKind::LeftParen => d += 1,
                        TokenKind::RightParen => d -= 1,
                        _ => {}
                    }
                    k += 1;
                }
                self.peek_at(k).kind == TokenKind::FatArrow
            } else {
                matches!(self.peek_at(o).kind, TokenKind::Identifier(_))
                    && self.peek_at(o + 1).kind == TokenKind::FatArrow
            };
            if shape {
                self.bump(); // 'async'
                // Fall through: remaining `static`/params/name handled below.
            } else {
                return None;
            }
        }
        // Static lambda: `static (params) =>` or `static name =>`
        // (possibly with `async` after `static`).
        if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Static)) {
            if let Some(lambda) = self.try_static_lambda() {
                return Some(lambda);
            }
            return None;
        }

        // Detect `(params) =>` or `name =>`.
        // `(params) =>`: current token is `(`, scan past balanced parens
        // to find `=>`.
        if self.peek().kind == TokenKind::LeftParen {
            let mut offset = 1;
            let mut depth = 1i32;
            while depth > 0 && self.peek_at(offset).kind != TokenKind::Eof {
                match self.peek_at(offset).kind {
                    TokenKind::LeftParen => depth += 1,
                    TokenKind::RightParen => depth -= 1,
                    _ => {}
                }
                if depth > 0 {
                    offset += 1;
                } else {
                    offset += 1;
                    break;
                }
            }
            if self.peek_at(offset).kind == TokenKind::FatArrow {
                // Confirmed: consume `(params) =>` and the body.
                let start = self.peek().span;
                // Consume the parenthesized parameter list.
                let mut depth = 1i32;
                self.bump(); // '('
                loop {
                    match self.peek().kind {
                        TokenKind::LeftParen => {
                            depth += 1;
                            self.bump();
                        }
                        TokenKind::RightParen => {
                            depth -= 1;
                            self.bump();
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {
                            self.bump();
                        }
                    }
                }
                self.expect(&TokenKind::FatArrow, "in lambda")?;
                self.consume_lambda_body();
                return Some(Expr::Ident {
                    name: "lambda".to_owned(),
                    span: start.to(self.peek_at(0).span),
                });
            }
            return None;
        }

        // `name =>`: current token is an identifier (or a contextual
        // keyword usable as one, e.g. `file =>`), next is `=>`.
        if self.peek_is_name(0)
            && self.peek_at(1).kind == TokenKind::FatArrow
        {
            let start = self.peek().span;
            self.bump(); // name
            self.expect(&TokenKind::FatArrow, "in lambda")?;
            // Body.
            self.consume_lambda_body();
            return Some(Expr::Ident {
                name: "lambda".to_owned(),
                span: start.to(self.peek_at(0).span),
            });
        }

        None
    }

    /// Consume the body of a lambda expression. If the body is a block,
    /// consume until the matching `}`. Otherwise, try to parse an expression;
    /// if the next token is `=`, it's an assignment-expression body.
    fn consume_lambda_body(&mut self) {
        if self.peek().kind == TokenKind::LeftBrace {
            let mut depth = 0i32;
            while self.peek().kind != TokenKind::Eof {
                if self.peek().kind == TokenKind::LeftBrace {
                    depth += 1;
                } else if self.peek().kind == TokenKind::RightBrace {
                    depth -= 1;
                    if depth == 0 {
                        self.bump();
                        break;
                    }
                }
                self.bump();
            }
        } else {
            let mark = self.pos;
            if self.expression().is_none() && self.pos == mark {
                // Expression parsing failed; consume tokens until a `,` or
                // `)` at depth 0, a `;`, or Eof.
                self.consume_until_comma_or_close();
            } else if self.peek().kind == TokenKind::Assign {
                // Assignment-expression body: `x => x = 1`.
                self.bump(); // '='
                if self.expression().is_none() {
                    self.consume_until_comma_or_close();
                }
            }
        }
    }

    /// Consume tokens until a `,` or `)` at depth 0, a `;`, or Eof.
    fn consume_until_comma_or_close(&mut self) {
        let mut depth = 0i32;
        while self.peek().kind != TokenKind::Eof {
            match self.peek().kind {
                TokenKind::LeftParen | TokenKind::LeftBrace | TokenKind::LeftBracket => {
                    depth += 1;
                    self.bump();
                }
                TokenKind::RightParen | TokenKind::RightBrace | TokenKind::RightBracket => {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                    self.bump();
                }
                TokenKind::Comma if depth == 0 => break,
                TokenKind::Semicolon => {
                    self.bump();
                    break;
                }
                _ => {
                    self.bump();
                }
            }
        }
    }

    /// Parse a static-lambda prefix: `static name =>` or
    /// `static (params) =>`. Returns `Some(expr)` if the current
    /// position looks like a static lambda, otherwise `None`.
    pub(in crate::parser::rd) fn try_static_lambda(&mut self) -> Option<Expr> {
        if !matches!(self.peek().kind, TokenKind::Keyword(Keyword::Static)) {
            return None;
        }
        // Optional `async` after `static` (`static async (...) =>`).
        let mut offset = 1;
        if matches!(
            self.peek_at(offset).kind,
            TokenKind::Keyword(Keyword::Async)
        ) {
            offset += 1;
        }
        if self.peek_at(offset).kind == TokenKind::LeftParen {
            // `(params) => body`
            offset += 1;
            // Scan past the parenthesized parameter list.
            let mut depth = 1i32;
            while depth > 0 && self.peek_at(offset).kind != TokenKind::Eof {
                match self.peek_at(offset).kind {
                    TokenKind::LeftParen => depth += 1,
                    TokenKind::RightParen => depth -= 1,
                    _ => {}
                }
                if depth > 0 {
                    offset += 1;
                } else {
                    offset += 1;
                    break;
                }
            }
        } else if matches!(self.peek_at(offset).kind, TokenKind::Identifier(_)) {
            // `name =>`
            offset += 1;
        } else {
            return None;
        }
        if self.peek_at(offset).kind != TokenKind::FatArrow {
            return None;
        }
        // Confirmed: consume `static [async]`, the parameter(s), the
        // `=>`, and the body, then return a placeholder expression.
        self.bump(); // 'static'
        if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Async)) {
            self.bump(); // 'async'
        }
        if self.peek().kind == TokenKind::LeftParen {
            let mut depth = 1i32;
            self.bump();
            loop {
                if self.peek().kind == TokenKind::LeftParen {
                    depth += 1;
                } else if self.peek().kind == TokenKind::RightParen {
                    depth -= 1;
                }
                self.bump();
                if depth == 0 || self.peek().kind == TokenKind::Eof {
                    break;
                }
            }
        } else {
            self.bump(); // single identifier
        }
            self.expect(&TokenKind::FatArrow, "in static lambda")?;
            self.consume_lambda_body();
        Some(Expr::Ident {
            name: "lambda".to_owned(),
            span: self.peek_at(0).span,
        })
    }

    pub(in crate::parser::rd) fn unary(&mut self) -> Option<Expr> {
        // `await expr` — suspension point inside an async method.
        if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Await)) {
            let op_token = self.bump();
            let inner = self.expression()?;
            return Some(Expr::Await {
                inner: Box::new(inner),
                span: op_token.span.to(expr_span_last(&self.tokens[..self.pos])),
            });
        }
        // Prefix increment/decrement: `++x` / `--x` ≡ `x = x +/- 1`
        // (yielding the NEW value, unlike the postfix forms).
        if matches!(
            self.peek().kind,
            TokenKind::PlusPlus | TokenKind::MinusMinus
        ) {
            let up = self.peek().kind == TokenKind::PlusPlus;
            let start = self.peek().span;
            self.bump();
            let target = self.unary()?;
            let end = expr_span_last(&self.tokens[..self.pos]);
            let one = Expr::Literal(Literal::Int(1));
            return Some(Self::compound_to_assign(
                target,
                if up { BinOp::Add } else { BinOp::Sub },
                one,
                start,
                end,
            ));
        }
        // Unary plus: `+x` ≡ `x` (numeric identity, per C#; the
        // backend models no numeric promotions, so this desugars to
        // the operand itself with no new AST or backend surface).
        if self.peek().kind == TokenKind::Plus {
            self.bump(); // '+'
            return self.unary();
        }
        let op = match self.peek().kind {
            TokenKind::Minus => Some(UnOp::Neg),
            TokenKind::Bang => Some(UnOp::Not),
            TokenKind::Tilde => Some(UnOp::BitNot),
            TokenKind::Star => Some(UnOp::Deref),
            TokenKind::Amp => Some(UnOp::AddrOf),
            TokenKind::Hat => Some(UnOp::IndexFromEnd),
            TokenKind::Keyword(Keyword::Not_) => Some(UnOp::Not),
            _ => None,
        };
        if let Some(op) = op {
            let op_token = self.bump();
            // Address-of a generic method group (`&M<T>`, `&A.B<T>` — a
            // function-pointer value): the operand is a bare name, not a
            // call. The normal postfix path only accepts `<...>` before
            // `(`, so `&M<T>;` would strand the brackets. Parse the
            // dotted name with one balanced list per segment here;
            // anything else (plain `&x`, `&(e)`, `&M<T>(...)`) keeps the
            // legacy path below untouched.
            if op == UnOp::AddrOf
                && matches!(self.peek().kind, TokenKind::Identifier(_))
            {
                let mark = self.pos;
                let err_mark = self.errors.len();
                if let Some(operand) = self.method_group_operand() {
                    if self.peek().kind != TokenKind::LeftParen {
                        return Some(Expr::Unary {
                            op,
                            inner: Box::new(operand),
                            span: op_token
                                .span
                                .to(expr_span_last(&self.tokens[..self.pos])),
                        });
                    }
                }
                self.pos = mark;
                self.errors.truncate(err_mark);
            }
            let inner = self.unary()?;
            return Some(Expr::Unary {
                op,
                inner: Box::new(inner),
                span: op_token.span.to(expr_span_last(&self.tokens[..self.pos])),
            });
        }
        self.postfix()
    }
}