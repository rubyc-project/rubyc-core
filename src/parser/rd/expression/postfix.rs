//! Expressions: precedence climbing, unary, postfix chains, primaries.

use crate::models::{Expr, Literal, Span, TokenKind, UnOp};

use super::super::{Parser, expr_span};
impl Parser {
    fn dot_name_tail(
        &mut self,
        expr: Expr,
        start: Span,
        name: String,
    ) -> Option<Expr> {
        // Check for generic type arguments: `name<T1, T2>(...)`
        // Only if the next token after '<' is an identifier,
        // keyword (type name), or `(` (tuple type argument),
        // AND the matching '>' is followed by '(' (method call
        // context).
        let mut generic_args = String::new();
        if self.peek().kind == TokenKind::Less
            && matches!(
                self.peek_at(1).kind,
                TokenKind::Identifier(_)
                    | TokenKind::Keyword(_)
                    | TokenKind::LeftParen
            )
            && (self.is_generic_type_args() || self.is_bare_generic_args())
        {
            self.bump(); // '<'
            let mut depth: i32 = 1;
            let mut args_str = String::new();
            loop {
                match self.peek().kind {
                    TokenKind::Less => {
                        depth += 1;
                        args_str.push('<');
                    }
                    TokenKind::Greater => {
                        depth -= 1;
                        if depth > 0 {
                            args_str.push('>');
                        }
                    }
                    // `>>` / `>>>` are lexed as Shr / ShrShr; in a
                    // generic argument list they are nested closers.
                    // Emit every closer except the final one (the
                    // `format!` below appends it): dropping any other
                    // strands open brackets (`M<X<T>>` recorded as
                    // `M<X<T>`), breaking explicit-arity filtering and
                    // overload ranking downstream.
                    TokenKind::Shr => {
                        // Two closers: emit the inner ones, break past
                        // the final. depth 2->0 emits one; depth 1->-1
                        // emits none (second `>` is outer junk, consumed
                        // for invalid input).
                        if depth == 2 {
                            args_str.push('>');
                        } else if depth > 2 {
                            args_str.push_str(">>");
                        }
                        depth -= 2;
                        if depth < 0 {
                            depth = 0;
                        }
                    }
                    TokenKind::ShrShr => {
                        // Three closers, same rule.
                        if depth == 3 {
                            args_str.push_str(">>");
                        } else if depth > 3 {
                            args_str.push_str(">>>");
                        } else if depth == 2 {
                            args_str.push('>');
                        }
                        depth -= 3;
                        if depth < 0 {
                            depth = 0;
                        }
                    }
                    _ => {
                        // Source text, not Debug format: generic names
                        // must read `As<string[]>` (overload ranking and
                        // return-type substitution parse them), never
                        // `As<Identifier("string")LeftBracket...>`.
                        let sp = self.peek().span;
                        args_str.push_str(
                            std::str::from_utf8(
                                &self.source[sp.start as usize
                                    ..(sp.end as usize).min(self.source.len())],
                            )
                            .unwrap_or(""),
                        );
                    }
                }
                self.bump();
                if depth == 0 || self.peek().kind == TokenKind::Eof {
                    break;
                }
            }
            generic_args = args_str;
        }
        if self.peek().kind == TokenKind::LeftParen {
            self.bump();
            let args = self.argument_list();
            let close = self.expect(
                &TokenKind::RightParen,
                "rubyc.parser.rd.Parser.ctx.close_arg_list",
            );
            let full_name = if generic_args.is_empty() {
                name
            } else {
                format!("{}<{}>", name, generic_args)
            };
            Some(Expr::MethodCall {
                object: Box::new(expr),
                name: full_name,
                args,
                span: start.to(close.map_or(self.peek().span, |t| t.span)),
            })
        } else {
            Some(Expr::Member {
                object: Box::new(expr),
                name,
                span: start.to(self.peek_at(0).span),
            })
        }
    }

    /// Dotted path text for a member chain (`a.b.C<T>`), or `None`
    /// for non-nameable receivers. Used to name generic calls.
    fn member_path_text(expr: &Expr) -> Option<String> {
        match expr {
            Expr::Ident { name, .. } => Some(name.clone()),
            Expr::Member { object, name, .. } => {
                Some(format!("{}.{}", Self::member_path_text(object)?, name))
            }
            _ => None,
        }
    }

    pub(in crate::parser::rd) fn postfix(&mut self) -> Option<Expr> {
        let mut expr = self.primary()?;
        loop {
            if self.peek().kind == TokenKind::Less
                && matches!(
                    self.peek_at(1).kind,
                    TokenKind::Identifier(_)
                        | TokenKind::Keyword(_)
                        | TokenKind::LeftParen
                )
                && (self.is_generic_type_args() || self.is_bare_generic_args())
            {
                self.bump(); // '<'
                let mut depth: i32 = 1;
                let mut args_str = String::new();
                loop {
                    match self.peek().kind {
                        TokenKind::Less => {
                            depth += 1;
                            args_str.push('<');
                        }
                        TokenKind::Greater => {
                            depth -= 1;
                            if depth > 0 {
                                args_str.push('>');
                            }
                        }
                    TokenKind::Shr => {
                        // Two closers: emit the inner ones (the format
                        // wrapper below appends the final one); same
                        // rule as `dot_name_tail`.
                        if depth == 2 {
                            args_str.push('>');
                        } else if depth > 2 {
                            args_str.push_str(">>");
                        }
                        depth -= 2;
                        if depth < 0 {
                            depth = 0;
                        }
                    }
                    TokenKind::ShrShr => {
                        // Three closers, same rule.
                        if depth == 3 {
                            args_str.push_str(">>");
                        } else if depth > 3 {
                            args_str.push_str(">>>");
                        } else if depth == 2 {
                            args_str.push('>');
                        }
                        depth -= 3;
                        if depth < 0 {
                            depth = 0;
                        }
                    }
                        _ => {
                            // Source text, not Debug format (see
                            // `dot_name_tail`'s identical arm).
                            let sp = self.peek().span;
                            args_str.push_str(
                                std::str::from_utf8(
                                    &self.source[sp.start as usize
                                        ..(sp.end as usize).min(self.source.len())],
                                )
                                .unwrap_or(""),
                            );
                        }
                    }
                    self.bump();
                    if depth == 0 || self.peek().kind == TokenKind::Eof {
                        break;
                    }
                }
                let s = expr_span(&expr);
                expr = Expr::Member {
                    object: Box::new(expr),
                    name: format!("<{}>", args_str),
                    span: s.to(self.peek_at(0).span),
                };
            } else if self.peek().kind == TokenKind::Dot {
                self.bump();
                let name = self.expect_identifier("after '.'")?;
                let start = expr_span(&expr);
                expr = self.dot_name_tail(expr, start, name)?;
            } else if self.peek().kind == TokenKind::QuestionDot
                && self.peek().span.end - self.peek().span.start == 2
            {
                // Null-conditional member access: `expr?.Member` — the
                // token must be exactly `?.` (a spaced `? .` starts a
                // ternary). Shares the name+generics+call tail with `.`
                // above (so `?.M<T>(...)` works); the AST collapses it
                // into a regular member access since the runtime does
                // not yet distinguish null-conditional semantics.
                self.bump(); // '?.'
                let name = self.expect_identifier("after '?.'")?;
                let start = expr_span(&expr);
                expr = self.dot_name_tail(expr, start, name)?;
            } else if self.peek().kind == TokenKind::Question
                && self.peek_at(1).kind == TokenKind::Dot
                && matches!(self.peek_at(2).kind, TokenKind::Identifier(_))
            {
                // Split null-conditional: `expr?` newline `.Member`
                // (fluent chains like `.Value?\n.ToString()`). A ternary
                // branch can never start with `.Identifier` (only a
                // numeric literal like `.5` can, and that is not an
                // identifier), so this is unambiguous. Same collapse
                // as adjacent `?.` above.
                self.bump(); // '?'
                self.bump(); // '.'
                let name = self.expect_identifier("after '? .'")?;
                let start = expr_span(&expr);
                expr = self.dot_name_tail(expr, start, name)?;
            } else if self.peek().kind == TokenKind::Arrow {
                // Pointer member access: `p->Field` ≡ `(*p).Field`.
                self.bump(); // '->'
                let name = self.expect_identifier("after '->'")?;
                let s = expr_span(&expr);
                let deref = Expr::Unary {
                    op: UnOp::Deref,
                    inner: Box::new(expr),
                    span: s.clone(),
                };
                expr = Expr::Member {
                    object: Box::new(deref),
                    name,
                    span: s.to(self.peek_at(0).span),
                };
            } else if self.peek().kind == TokenKind::LeftBracket {
                let obj_span = expr_span(&expr);
                self.bump();
                // Index argument: a single expression, possibly followed by
                // a range `..` and another expression, or comma-separated
                // for multi-dim. C# 8 ranges (`[..end]`, `[start..end]`)
                // set `is_range`: a slice has the CONTAINER's type, so
                // method chains on it (`span[..n].ToArray()`) must not
                // resolve against the element type.
                let mut index = None;
                let mut is_range = false;
                if self.peek().kind == TokenKind::RightBracket {
                    // Empty `[]` is invalid C#; consume to avoid leaving
                    // an unbalanced bracket.
                } else if self.peek().kind == TokenKind::DotDot {
                    // Start-omitted range: `[..end]`. The start defaults
                    // to 0; record the end as the index shape.
                    is_range = true;
                    self.bump();
                    if self.peek().kind != TokenKind::RightBracket {
                        index = Some(self.expression()?);
                    } else {
                        index = Some(Expr::Literal(Literal::Int(0)));
                    }
                } else {
                    index = Some(self.expression()?);
                    if self.peek().kind == TokenKind::DotDot {
                        is_range = true;
                        self.bump();
                        if self.peek().kind != TokenKind::RightBracket {
                            let _ = self.expression();
                        }
                    }
                }
                while self.eat(&TokenKind::Comma).is_some() {
                    let _ = self.expression();
                }
                self.expect(&TokenKind::RightBracket, "to close index")?;
                expr = Expr::Index {
                    object: Box::new(expr),
                    index: Box::new(index.unwrap_or(Expr::Literal(Literal::Int(0)))),
                    is_range,
                    span: obj_span.to(self.peek_at(0).span),
                };
            } else if self.peek().kind == TokenKind::DotDot {
                // Range continuation: `start..end` (either side may be
                // absent, though a missing start here means the indexer
                // above already handled `[..end]`). Slicing has no
                // runtime model yet, so like the indexer this keeps the
                // start (already in `expr`) and drops the end.
                self.bump(); // '..'
                match self.peek().kind {
                    TokenKind::Comma
                    | TokenKind::RightParen
                    | TokenKind::RightBracket
                    | TokenKind::Semicolon
                    | TokenKind::RightBrace
                    | TokenKind::Colon
                    | TokenKind::FatArrow
                    | TokenKind::Eof => {}
                    _ => {
                        let _ = self.expression();
                    }
                }
            } else if self.peek().kind == TokenKind::QuestionDot
                && self.peek().span.end - self.peek().span.start == 2
            {
                // Null-conditional member access: `expr?.Member` — the
                // token must be exactly `?.` (a spaced `? .` starts a
                // ternary, as for `? [` below). Bump the `?.` and the
                // following identifier; the AST collapses it into a
                // regular member access since the runtime does not yet
                // distinguish null-conditional semantics.
                self.bump(); // '?.'
                let name = self.expect_identifier("after '?.'")?;
                let s = expr_span(&expr);
                expr = Expr::Member {
                    object: Box::new(expr),
                    name,
                    span: s.to(self.peek_at(0).span),
                };
            } else if self.peek().kind == TokenKind::Question
                && self.peek_at(1).kind == TokenKind::Dot
                && matches!(self.peek_at(2).kind, TokenKind::Identifier(_))
            {
                // Split null-conditional: `expr?` newline `.Member`
                // (same rule as the other postfix loop — a ternary
                // branch can never start with `.Identifier`). Collapses
                // like adjacent `?.` above.
                self.bump(); // '?'
                self.bump(); // '.'
                let name = self.expect_identifier("after '? .'")?;
                let s = expr_span(&expr);
                expr = Expr::Member {
                    object: Box::new(expr),
                    name,
                    span: s.to(self.peek_at(0).span),
                };
            } else if self.peek().kind == TokenKind::Question
                && self.peek_at(1).kind == TokenKind::LeftBracket
                && self.peek().span.end == self.peek_at(1).span.start
            {
                // Null-conditional indexer: `expr?[idx]` — the `?` and
                // `[` must be ADJACENT. A spaced `? [` starts a ternary
                // whose branch is a collection (`a ? [y] : x`), which
                // the `?` handling in `expression()` owns.
                self.bump(); // '?'
                self.bump(); // '['
                let mut index = None;
                if self.peek().kind != TokenKind::RightBracket {
                    index = Some(self.expression()?);
                }
                while self.eat(&TokenKind::Comma).is_some() {
                    let _ = self.expression();
                }
                let close = self
                    .expect(&TokenKind::RightBracket, "to close null-conditional index");
                let s = expr_span(&expr);
                expr = Expr::Index {
                    object: Box::new(expr),
                    index: Box::new(index.unwrap_or(Expr::Literal(Literal::Int(0)))),
                    is_range: false,
                    span: s.to(close.map_or(self.peek().span, |t| t.span)),
                };
            } else if self.peek().kind == TokenKind::Bang {
                // Null-forgiving operator: `expr!`
                self.bump();
                // For now, treat as identity (the `!` is a compile-time assertion).
            } else if matches!(
                self.peek().kind,
                TokenKind::PlusPlus | TokenKind::MinusMinus
            ) {
                // Postfix increment/decrement: `i++`, `a[i]--`, `o.F++`.
                // Yields the OLD value (lowering stores old+/-1); valid on
                // any assignable target, including indexers and fields.
                let up = self.peek().kind == TokenKind::PlusPlus;
                let op_span = self.bump().span;
                let s = expr_span(&expr);
                expr = if up {
                    Expr::PostInc {
                        target: Box::new(expr),
                        span: s.to(op_span),
                    }
                } else {
                    Expr::PostDec {
                        target: Box::new(expr),
                        span: s.to(op_span),
                    }
                };
            } else if self.peek().kind == TokenKind::LeftParen
                && matches!(expr, Expr::Ident { .. })
            {
                // Call on a bare name after a transparent postfix:
                // `Callback!(args)`. Plain `Ident(` never reaches here
                // (primary consumes it as a Call), so a `(` here means
                // something like `!` intervened.
                let s = expr_span(&expr);
                let callee = match expr {
                    Expr::Ident { name, .. } => name,
                    _ => unreachable!(),
                };
                self.bump(); // '('
                let args = self.argument_list();
                let close = self.expect(
                    &TokenKind::RightParen,
                    "rubyc.parser.rd.Parser.ctx.close_arg_list",
                );
                expr = Expr::Call {
                    callee,
                    args,
                    span: s.to(close.map_or(self.peek().span, |t| t.span)),
                };
            } else if self.peek().kind == TokenKind::LeftParen
                && matches!(expr, Expr::Member { .. })
            {
                // A `(` after a member is either a generic call
                // (`M<T>(...)` — the name carries `<...>` from the
                // generic type access arm above) or a plain call
                // (`?.M(...)`, `p->F(...)` — those arms make a bare
                // Member and leave `(` unconsumed; plain `.M(...)`
                // never reaches here, the Dot arm owns it).
                let is_generic = matches!(&expr, Expr::Member { name, .. } if name.contains('<'));
                if is_generic {
                    // Model as a Call with the full dotted path as
                    // callee (mirroring qualified method names like
                    // `IValueProvider<int>.GetValue`). A non-nameable
                    // receiver (e.g. a call result) is not a generic
                    // call at all — break out and let it error
                    // downstream instead of spinning here.
                    // The `<...>` suffix MUST be kept: dropping it
                    // loses explicit type arguments, so generic
                    // return-type substitution sees none and the
                    // result degrades (`GetOrCreateCacheEntry<BoxCache>()`
                    // typed as `T`, failing `.Box` downstream).
                    let suffix = match &expr {
                        Expr::Member { name, .. } => name.clone(),
                        _ => String::new(),
                    };
                    let path = match &expr {
                        Expr::Member { object, .. } => Self::member_path_text(object),
                        _ => None,
                    };
                    if let Some(callee) = path.map(|p| format!("{p}{suffix}")) {
                        self.bump();
                        let args = self.argument_list();
                        let close = self.expect(
                            &TokenKind::RightParen,
                            "rubyc.parser.rd.Parser.ctx.close_arg_list",
                        );
                        let s = expr_span(&expr);
                        expr = Expr::Call {
                            callee,
                            args,
                            span: s.to(close.map_or(self.peek().span, |t| t.span)),
                        };
                    } else {
                        break;
                    }
                } else {
                    // Plain call on a member value.
                    let s = expr_span(&expr);
                    let (object, name) = match expr {
                        Expr::Member { object, name, .. } => (object, name),
                        _ => unreachable!(),
                    };
                    self.bump(); // '('
                    let args = self.argument_list();
                    let close = self.expect(
                        &TokenKind::RightParen,
                        "rubyc.parser.rd.Parser.ctx.close_arg_list",
                    );
                    expr = Expr::MethodCall {
                        object,
                        name,
                        args,
                        span: s.to(close.map_or(self.peek().span, |t| t.span)),
                    };
                }
            } else if self.peek().kind == TokenKind::LeftParen
                && !matches!(expr, Expr::Ident { .. } | Expr::Member { .. })
            {
                // Invoking a non-name value: `(T)x(args)` (delegate call),
                // `(a)(b)`, etc. In .NET this is a delegate `Invoke`;
                // model it as such — unknown receivers degrade to a
                // placeholder until delegates land.
                self.bump();
                let args = self.argument_list();
                let close = self.expect(
                    &TokenKind::RightParen,
                    "rubyc.parser.rd.Parser.ctx.close_arg_list",
                );
                let s = expr_span(&expr);
                expr = Expr::MethodCall {
                    object: Box::new(expr),
                    name: "Invoke".to_owned(),
                    args,
                    span: s.to(close.map_or(self.peek().span, |t| t.span)),
                };
            } else {
                break;
            }
        }
        Some(expr)
    }

}