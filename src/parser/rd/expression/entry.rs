//! Expressions: precedence climbing, unary, postfix chains, primaries.

use crate::models::{
    BinOp, Expr, InterpPart, Keyword, Literal, Span, SwitchExprArm, TokenKind, UnOp,
};

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
            // Parse one or more `or`-separated pattern labels as REAL
            // patterns (the same parser `is` uses), so they lower through
            // the same tests instead of being inert text.
            let mut labels = Vec::new();
            loop {
                let label = self
                    .parse_is_pattern_chain(Some(&subject))
                    .unwrap_or(Expr::Ident {
                        name: "_".to_owned(),
                        span: self.peek().span,
                    });
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
            let mut guard: Option<Expr> = None;
            if matches!(self.peek().kind, TokenKind::Keyword(Keyword::When_)) {
                self.bump();
                if self.peek_is_name(0)
                    && self.peek_at(1).kind == TokenKind::FatArrow
                {
                    // A bare name directly before `=>`: the arrow belongs
                    // to the arm, so the guard is the name itself.
                    let span = self.peek().span;
                    let name = match self.peek().kind.clone() {
                        TokenKind::Identifier(n) => n,
                        _ => String::new(),
                    };
                    self.bump();
                    guard = Some(Expr::Ident { name, span });
                } else {
                    guard = self.expression();
                }
            }
            // `=> body`
            self.expect(&TokenKind::FatArrow, "in switch expression arm")?;
            let body = self.expression()?;
            // Optional trailing comma between arms.
            self.eat(&TokenKind::Comma);
            arms.push(SwitchExprArm {
                labels,
                guard,
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
                let mut pattern = self.parse_is_pattern_primary(Some(&lhs))?;
                loop {
                    match self.peek().kind {
                        TokenKind::Keyword(Keyword::And_) => {
                            self.bump();
                            let rhs = self.parse_is_pattern_primary(Some(&lhs))?;
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
                            let rhs = self.parse_is_pattern_primary(Some(&lhs))?;
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
                return Some(self.lambda_expr(vec!["async".to_owned()], start));
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
                // Confirmed: `(params) =>`. The parameter NAMES are kept:
                // without them the body cannot resolve its own
                // parameters.
                let start = self.peek().span;
                let params = self.consume_lambda_params();
                self.expect(&TokenKind::FatArrow, "in lambda")?;
                return Some(self.lambda_expr(params, start));
            }
            return None;
        }

        // `name =>`: current token is an identifier (or a contextual
        // keyword usable as one, e.g. `file =>`), next is `=>`.
        if self.peek_is_name(0)
            && self.peek_at(1).kind == TokenKind::FatArrow
        {
            let start = self.peek().span;
            let span = self.peek().span;
            let name = String::from_utf8_lossy(
                self.source
                    .get(span.start as usize..span.end as usize)
                    .unwrap_or(&[]),
            )
            .into_owned();
            self.bump(); // name
            self.expect(&TokenKind::FatArrow, "in lambda")?;
            return Some(self.lambda_expr(vec![name], start));
        }

        None
    }

    /// Consume the body of a lambda expression. If the body is a block,
    /// consume until the matching `}`. Otherwise, try to parse an expression;
    /// if the next token is `=`, it's an assignment-expression body.
    /// Parse a lambda body: `{ stmts }` or `expr`.
    ///
    /// This used to consume the body and throw it away (the whole lambda
    /// became a magic placeholder), so a lambda's reads and side effects
    /// vanished from every downstream pass. It now returns the body.
    fn consume_lambda_body(&mut self) -> crate::models::ast::expression::LambdaBody {
        use crate::models::ast::expression::LambdaBody;
        if self.peek().kind == TokenKind::LeftBrace {
            let block = self.block();
            return LambdaBody::Block(block);
        }
        match self.expression() {
            Some(e) => LambdaBody::Expr(Box::new(e)),
            // Malformed body: keep the shape so the caller can still
            // report its own diagnostics about the enclosing construct.
            None => LambdaBody::Expr(Box::new(Expr::Literal(Literal::Int(0)))),
        }
    }

    /// Build a lambda node from the parameters already consumed and the
    /// body just parsed.
    fn lambda_expr(&mut self, params: Vec<String>, start: Span) -> Expr {
        let body = self.consume_lambda_body();
        Expr::Lambda {
            params,
            body,
            span: start.to(self.peek_at(0).span),
        }
    }

    /// Parameter names from a `( ... )` lambda parameter list, already
    /// positioned at `(`. Types and modifiers are skipped: a group's LAST
    /// identifier is its name (`int x` -> `x`, `x` -> `x`, `out int y` ->
    /// `y`), which is the same rule the declaration parser uses to find a
    /// declared name.
    pub(super) fn consume_lambda_params(&mut self) -> Vec<String> {
        let mut names: Vec<String> = Vec::new();
        if self.peek().kind != TokenKind::LeftParen {
            return names;
        }
        self.bump(); // '('
        let mut current: Vec<String> = Vec::new();
        let mut depth = 1i32;
        while depth > 0 && self.peek().kind != TokenKind::Eof {
            match self.peek().kind.clone() {
                TokenKind::LeftParen => {
                    depth += 1;
                    current.clear();
                    self.bump();
                }
                TokenKind::RightParen => {
                    depth -= 1;
                    if !current.is_empty()
                        && let Some(last) = current.last()
                    {
                        names.push(last.clone());
                    }
                    current.clear();
                    self.bump();
                }
                TokenKind::Comma if depth == 1 => {
                    if let Some(last) = current.last() {
                        names.push(last.clone());
                    }
                    current.clear();
                    self.bump();
                }
                TokenKind::Identifier(n) => {
                    current.push(n);
                    self.bump();
                }
                TokenKind::Keyword(_) => {
                    // Modifiers (`ref`, `out`, `in`, `params`, `scoped`)
                    // are not the name; a contextual keyword used AS a
                    // name is, and it reads from the source.
                    let span = self.peek().span;
                    let txt = String::from_utf8_lossy(
                        &self.source[span.start as usize..span.end as usize],
                    )
                    .into_owned();
                    current.push(txt);
                    self.bump();
                }
                _ => {
                    self.bump();
                }
            }
        }
        // A list without a trailing `)` (malformed) keeps what it found.
        if !current.is_empty()
            && let Some(last) = current.last()
        {
            names.push(last.clone());
        }
        names
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
        // Confirmed: consume `static [async]`, the parameter NAMES, the
        // `=>`, and the body. The names are kept: a `static` lambda is an
        // ordinary lambda with no capture, and dropping its parameters
        // made every use of one read as an unknown name. (This used to
        // hand over an empty list, from when no lambda could be lowered
        // at all.)
        self.bump(); // 'static'
        if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Async)) {
            self.bump(); // 'async'
        }
        let params = if self.peek().kind == TokenKind::LeftParen {
            self.consume_lambda_params()
        } else {
            let name = self.expect_identifier("as lambda parameter")?;
            vec![name]
        };
        let start = self.peek().span;
        self.expect(&TokenKind::FatArrow, "in static lambda")?;
        Some(self.lambda_expr(params, start))
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