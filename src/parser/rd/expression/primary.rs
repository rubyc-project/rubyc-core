//! Expressions: precedence climbing, unary, postfix chains, primaries.

use crate::models::{Expr, Keyword, Literal, TokenKind, UnOp};

use super::super::{Parser, describe, expr_span_last};
impl Parser {
    pub(in crate::parser::rd) fn primary(&mut self) -> Option<Expr> {
        let token = self.peek().clone();
        match &token.kind {
            TokenKind::LeftBracket => {
                // Collection expression: `[e1, e2, ..spread, ...]` (C# 12).
                // Modeled exactly like the `{ ... }` implicit array
                // literal below: a `NewArray` sized by element count.
                // Spread elements (`..x`) contribute one slot each.
                self.bump(); // '['
                let mut elements = Vec::new();
                while self.peek().kind != TokenKind::RightBracket
                    && self.peek().kind != TokenKind::Eof
                {
                    if self.eat(&TokenKind::Comma).is_some() {
                        continue;
                    }
                    // Spread prefix: `..expr`.
                    if self.peek().kind == TokenKind::DotDot {
                        self.bump();
                    }
                    let mark = self.pos;
                    if self.expression().is_none() {
                        if self.pos == mark {
                            self.bump();
                        }
                    } else {
                        elements.push(Expr::Literal(Literal::Int(0)));
                    }
                }
                self.expect(&TokenKind::RightBracket, "to close collection expression")?;
                return Some(Expr::NewArray {
                    elem_type: String::new(),
                    size: Box::new(Expr::Literal(Literal::Int(elements.len() as i64))),
                    span: token.span.to(self.peek_at(0).span),
                });
            }
            TokenKind::LeftBrace => {
                // Implicit array literal: `{ e1, e2, ... }` inside a
                // multi-dimensional array initializer. The compact AST
                // models it as a `NewArray` with an empty element type.
                self.bump();
                let mut elements = Vec::new();
                while self.peek().kind != TokenKind::RightBrace
                    && self.peek().kind != TokenKind::Eof
                {
                    if self.eat(&TokenKind::Comma).is_some() {
                        continue;
                    }
                    let mark = self.pos;
                    if self.expression().is_none() {
                        if self.pos == mark {
                            self.bump();
                        }
                    } else {
                        elements.push(Expr::Literal(Literal::Int(0)));
                    }
                }
                self.expect(&TokenKind::RightBrace, "to close array literal")?;
                return Some(Expr::NewArray {
                    elem_type: String::new(),
                    size: Box::new(Expr::Literal(Literal::Int(elements.len() as i64))),
                    span: token.span.to(self.peek_at(0).span),
                });
            }
            TokenKind::DotDot => {
                // Start-omitted range: `..end` (or lone `..`) in
                // expression position, e.g. `AsSpan(..^1)`. Slicing has
                // no runtime model yet: record the end (or 0 when
                // absent), mirroring the indexer above. (The `[..end]`
                // and spread forms are claimed by their own loops
                // before primary is consulted, so this only fires for
                // genuine range expressions.)
                self.bump(); // '..'
                let end = match self.peek().kind {
                    TokenKind::Comma
                    | TokenKind::RightParen
                    | TokenKind::RightBracket
                    | TokenKind::Semicolon
                    | TokenKind::RightBrace
                    | TokenKind::Colon
                    | TokenKind::FatArrow
                    | TokenKind::Eof => None,
                    _ => self.expression(),
                };
                return Some(end.unwrap_or(Expr::Literal(Literal::Int(0))));
            }
            TokenKind::LeftParen => {
                // Cast: `(Type)expr` — the token after '(' is a type name
                // (identifier) followed by ')'. Also accept `(Type?)` for
                // nullable casts and `(Type[])` / `(Type[,])` for array
                // casts, which are common in the stress fixture.
                let second = &self.peek_at(2).kind;
                // A `(T)` / `(T?)` / `(T[])` cast. For the nullable form
                // `(T?)`, require the `)` to follow the `?` so we do not
                // misread a ternary `(b ? a : b)` as a cast of type `b?`.
                // Also accept generic types: `(Dictionary<K,V>)expr`.
                // Function-pointer cast: `(delegate* ...)expr`. The
                // `delegate` keyword plus `*` can only start a type.
                let is_delegate_cast = matches!(
                    self.peek_at(1).kind,
                    TokenKind::Keyword(Keyword::Delegate_)
                ) && self.peek_at(2).kind == TokenKind::Star;
                // Transparent `global::` qualifier:
                // `(global::System.Object)o` (generated serializers).
                // `global ::` can only start a type, so look ahead past
                // the qualifier over dots, generic lists, and suffixes.
                let is_global_cast = matches!(
                    &self.peek_at(1).kind,
                    TokenKind::Identifier(n) if n == "global"
                ) && self.peek_at(2).kind == TokenKind::Colon
                    && self.peek_at(3).kind == TokenKind::Colon
                    && matches!(self.peek_at(4).kind, TokenKind::Identifier(_))
                    && {
                        let mut i = 5usize;
                        loop {
                            while self.peek_at(i).kind == TokenKind::Dot
                                && matches!(
                                    self.peek_at(i + 1).kind,
                                    TokenKind::Identifier(_)
                                )
                            {
                                i += 2;
                            }
                            if !matches!(
                                self.peek_at(i).kind,
                                TokenKind::Less | TokenKind::Shl
                            ) {
                                break;
                            }
                            let mut depth = 0usize;
                            loop {
                                match self.peek_at(i).kind {
                                    TokenKind::Eof => break,
                                    TokenKind::Less => depth += 1,
                                    TokenKind::Shl => depth += 2,
                                    TokenKind::Greater => {
                                        depth = depth.saturating_sub(1);
                                        if depth == 0 {
                                            break;
                                        }
                                    }
                                    TokenKind::Shr => {
                                        depth = depth.saturating_sub(2);
                                        if depth == 0 {
                                            break;
                                        }
                                    }
                                    TokenKind::ShrShr => {
                                        depth = depth.saturating_sub(3);
                                        if depth == 0 {
                                            break;
                                        }
                                    }
                                    _ => {}
                                }
                                i += 1;
                            }
                            i += 1; // past the closer
                        }
                        let j = self.cast_gate_suffixes(i);
                        matches!(self.peek_at(j).kind, TokenKind::RightParen)
                    };
                let is_cast = (matches!(self.peek_at(1).kind, TokenKind::Identifier(_))
                    && match second {
                        TokenKind::RightParen => true,
                        // Array-type cast `(T[])x` / `(T[,])x`: the brackets
                        // must be EMPTY rank specifiers. A non-empty bracket
                        // (`(a[i] ...)`) is an indexer, not a type.
                        TokenKind::LeftBracket => matches!(
                            self.peek_at(3).kind,
                            TokenKind::RightBracket | TokenKind::Comma
                        ),
                        // Nullable cast `(T?)expr`, with `*`/`[...]`
                        // suffixes in any order and combination
                        // (`(T?[])x`, `(object?[]?)x`). A bare `?` not
                        // followed by `)` (or suffixes leading to one)
                        // is a ternary.
                        TokenKind::Question => {
                            let mut i = 3usize;
                            loop {
                                let mut progressed = false;
                                if self.peek_at(i).kind == TokenKind::Question {
                                    i += 1;
                                    progressed = true;
                                }
                                while self.peek_at(i).kind == TokenKind::Star {
                                    i += 1;
                                    progressed = true;
                                }
                                while self.peek_at(i).kind == TokenKind::LeftBracket {
                                    let n = Self::skip_balanced_bracket_at(self, i);
                                    if n == i {
                                        break;
                                    }
                                    i = n;
                                    progressed = true;
                                }
                                if !progressed {
                                    break;
                                }
                            }
                            matches!(self.peek_at(i).kind, TokenKind::RightParen)
                        }
                        // Pointer cast: `(T*)expr`, `(T**)expr`, with
                        // optional `[...]`/`?` after the stars (same
                        // interleaved loop).
                        TokenKind::Star => {
                            let mut i = 2usize;
                            loop {
                                let mut progressed = false;
                                if self.peek_at(i).kind == TokenKind::Question {
                                    i += 1;
                                    progressed = true;
                                }
                                while self.peek_at(i).kind == TokenKind::Star {
                                    i += 1;
                                    progressed = true;
                                }
                                while self.peek_at(i).kind == TokenKind::LeftBracket {
                                    let n = Self::skip_balanced_bracket_at(self, i);
                                    if n == i {
                                        break;
                                    }
                                    i = n;
                                    progressed = true;
                                }
                                if !progressed {
                                    break;
                                }
                            }
                            matches!(self.peek_at(i).kind, TokenKind::RightParen)
                        }
                        // Generic type: `(T<...>)expr` — find the matching `)`.
                        // `>>`/`>>>` lex as Shr/ShrShr (2/3 closers).
                        // Suffixes may follow the list (`(T<U>?)x`,
                        // mirroring the Dot arm below).
                        TokenKind::Less => {
                            let mut depth = 0usize;
                            let mut i = 2usize; // start at the `<`
                            loop {
                                match self.peek_at(i).kind {
                                    TokenKind::Eof => break,
                                    TokenKind::Less => depth += 1,
                                    TokenKind::Shl => depth += 2,
                                    TokenKind::Greater => {
                                        depth = depth.saturating_sub(1);
                                        if depth == 0 {
                                            break;
                                        }
                                    }
                                    TokenKind::Shr => {
                                        depth = depth.saturating_sub(2);
                                        if depth == 0 {
                                            break;
                                        }
                                    }
                                    TokenKind::ShrShr => {
                                        depth = depth.saturating_sub(3);
                                        if depth == 0 {
                                            break;
                                        }
                                    }
                                    _ => {}
                                }
                                i += 1;
                            }
                            // `i` sits on the closer; interleaved suffixes
                            // and dotted nested-type segments follow
                            // (`(Dictionary<K,V>.Enumerator)x`).
                            let j = self.cast_gate_suffixes(i + 1);
                            matches!(self.peek_at(j).kind, TokenKind::RightParen)
                        }
                        // Dot-qualified type: `(A.B)expr` or `(A.B<...>)expr`.
                        TokenKind::Dot => {
                            let mut i = 2usize;
                            // Consume dot-qualified segments.
                            while self.peek_at(i).kind == TokenKind::Dot
                                && matches!(
                                    self.peek_at(i + 1).kind,
                                    TokenKind::Identifier(_)
                                )
                            {
                                i += 2;
                            }
                            // Optional generic list (`>>`/`>>>` lex as
                            // Shr/ShrShr).
                            if matches!(
                                self.peek_at(i).kind,
                                TokenKind::Less | TokenKind::Shl
                            ) {
                                let mut depth = 0usize;
                                loop {
                                    match self.peek_at(i).kind {
                                        TokenKind::Eof => break,
                                        TokenKind::Less => depth += 1,
                                        TokenKind::Shl => depth += 2,
                                        TokenKind::Greater => {
                                            depth = depth.saturating_sub(1);
                                            if depth == 0 {
                                                i += 1;
                                                break;
                                            }
                                        }
                                        TokenKind::Shr => {
                                            depth = depth.saturating_sub(2);
                                            if depth == 0 {
                                                i += 1;
                                                break;
                                            }
                                        }
                                        TokenKind::ShrShr => {
                                            depth = depth.saturating_sub(3);
                                            if depth == 0 {
                                                i += 1;
                                                break;
                                            }
                                        }
                                        _ => {}
                                    }
                                    i += 1;
                                }
                            }
                            // Optional nullable/pointer/array suffixes in
                            // any order and combination (`[]?`, `?[]?`),
                            // plus dotted nested-type segments after a
                            // generic list (`(Ns.D<K,V>.Enumerator)x`).
                            let i = self.cast_gate_suffixes(i);
                            matches!(self.peek_at(i).kind, TokenKind::RightParen)
                        }
                        _ => false,
                    })
                    || is_delegate_cast
                    || is_global_cast;
                if is_cast {
                    // Speculative cast parse: `(x)` (grouping) and `(T)x`
                    // (cast) are syntactically identical without type info.
                    // Attempt the cast; if no operand follows (or the type
                    // parse wanders), rewind — dropping speculative errors —
                    // and fall through to tuple/grouping handling below.
                    let mark = self.pos;
                    let err_mark = self.errors.len();
                    self.bump(); // '('
                    // Parse the cast type: identifier + optional dots,
                    // generic list, nullable/array suffixes.
                    let type_name = self.qualified_type_name();
                    self.bump(); // ')'
                    if let Some(inner) = self.unary() {
                        return Some(Expr::Cast {
                            type_name,
                            expr: Box::new(inner),
                            span: token.span.to(expr_span_last(&self.tokens[..self.pos])),
                        });
                    }
                    self.pos = mark;
                    self.errors.truncate(err_mark);
                }
                // Cast to a tuple type: `(int, Token[])x`, `((int, T))y`.
                // The comma excludes these from the cast gate above
                // (tuple-literal protection), but a tuple *type* whose
                // `)` is followed by an operand can only be a cast — a
                // literal is never applied to anything (`(a,b).M` keeps
                // the literal path: `.` is no operand starter).
                // The outer parens of `((T,...))x` are consumed as the
                // cast's own pair; the inner pair is the type.
                if self.tuple_cast_operand_at().is_some() {
                    let mark = self.pos;
                    let err_mark = self.errors.len();
                    self.bump(); // '(' (depth 1: the cast's own pair)
                    // Consume the tuple type through its `)` (validated
                    // shape guarantees balance).
                    let mut depth = 1i32;
                    while self.peek().kind != TokenKind::Eof {
                        match self.peek().kind {
                            TokenKind::LeftParen => depth += 1,
                            TokenKind::RightParen => {
                                depth -= 1;
                                self.bump();
                                if depth == 0 {
                                    break;
                                }
                                continue;
                            }
                            _ => {}
                        }
                        self.bump();
                    }
                    if let Some(inner) = self.unary() {
                        return Some(Expr::Cast {
                            type_name: "tuple".to_owned(),
                            expr: Box::new(inner),
                            span: token.span.to(expr_span_last(&self.tokens[..self.pos])),
                        });
                    }
                    self.pos = mark;
                    self.errors.truncate(err_mark);
                }
                // Tuple literal: `(e1, e2, ...)` with optional element
                // names (`(thisRef: this, x)` — names are metadata; only
                // values matter). We require at least one comma so `(x)`
                // is unambiguously a parenthesized expr.
                if self.contains_tuple_literal_ahead() {
                    self.bump(); // '('
                    let mut elements = Vec::new();
                    loop {
                        if self.peek().kind == TokenKind::RightParen {
                            break;
                        }
                        // Named element: `name: value`. A `:` here can
                        // only be an element name (ternary colons are
                        // consumed inside `expression()`); `::` is a
                        // namespace alias, never an element name.
                        if self.peek_is_name(0)
                            && self.peek_at(1).kind == TokenKind::Colon
                            && self.peek_at(2).kind != TokenKind::Colon
                        {
                            self.bump(); // name
                            self.bump(); // ':'
                        }
                        let mark = self.pos;
                        if self.expression().is_none() {
                            if self.pos == mark {
                                self.bump();
                            }
                        } else {
                            elements.push(Expr::Literal(Literal::Int(0)));
                        }
                        if self.eat(&TokenKind::Comma).is_none() {
                            break;
                        }
                    }
                    self.expect(&TokenKind::RightParen, "to close tuple literal")?;
                    return Some(Expr::NewArray {
                        elem_type: "tuple".to_owned(),
                        size: Box::new(Expr::Literal(Literal::Int(elements.len() as i64))),
                        span: token.span.to(self.peek_at(0).span),
                    });
                }
                self.bump();
                let inner = self.expression()?;
                self.expect(&TokenKind::RightParen, "to close parenthesized expression")?;
                Some(inner)
            }
            TokenKind::Keyword(Keyword::Default) => {
                self.bump(); // 'default'
                // `default` (no type) or `default(T)` (with type)
                if self.peek().kind == TokenKind::LeftParen {
                    self.bump();
                    let type_name = self.type_name()?;
                    self.expect(&TokenKind::RightParen, "to close default type")?;
                    let end = expr_span_last(&self.tokens[..self.pos]);
                    Some(Expr::Default {
                        type_name,
                        span: token.span.to(end),
                    })
                } else {
                    // `default` without a type — use "void" as placeholder.
                    let end = token.span;
                    Some(Expr::Default {
                        type_name: "void".to_string(),
                        span: end,
                    })
                }
            }
            TokenKind::Keyword(Keyword::New) => {
                self.bump(); // 'new'
                // Tuple-typed array creation:
                // `new (T1 n1, T2 n2)[] { ... }`, `new (T1, T2)[size]`.
                // A `(...)` holding `Type Name` pairs — or all unnamed
                // types with a primitive/complex element — is a tuple
                // type, never an argument list (`new (a, b)` stays
                // target-typed below).
                if self.peek().kind == TokenKind::LeftParen
                    && (self.tuple_type_at(0).is_some()
                        || self.unnamed_tuple_type_at(0).is_some())
                {
                    self.bump(); // '('
                    let mut pdepth = 1i32;
                    while pdepth > 0 && self.peek().kind != TokenKind::Eof {
                        match self.peek().kind {
                            TokenKind::LeftParen => pdepth += 1,
                            TokenKind::RightParen => pdepth -= 1,
                            _ => {}
                        }
                        self.bump();
                    }
                    // Rank suffixes `[]`.
                    while self.peek().kind == TokenKind::LeftBracket
                        && matches!(
                            self.peek_at(1).kind,
                            TokenKind::RightBracket | TokenKind::Comma
                        )
                    {
                        self.bump(); // '['
                        while self.peek().kind != TokenKind::RightBracket
                            && self.peek().kind != TokenKind::Eof
                        {
                            self.bump();
                        }
                        self.bump(); // ']'
                    }
                    // `?`/`*` suffixes.
                    while matches!(
                        self.peek().kind,
                        TokenKind::Question | TokenKind::Star
                    ) {
                        self.bump();
                    }
                    // Sized `[e0, ...]` (only the first is stored),
                    // optionally followed by an initializer
                    // `new (A, B)[n] { ... }`.
                    if self.peek().kind == TokenKind::LeftBracket {
                        self.bump(); // '['
                        let size = self.expression()?;
                        while self.eat(&TokenKind::Comma).is_some() {
                            let _ = self.expression()?;
                        }
                        self.expect(&TokenKind::RightBracket, "to close array size")?;
                        // Optional trailing initializer: consume balanced
                        // braces so the field `;` is found.
                        if self.peek().kind == TokenKind::LeftBrace {
                            self.bump(); // '{'
                            let mut depth = 1i32;
                            while depth > 0 && self.peek().kind != TokenKind::Eof {
                                match self.peek().kind {
                                    TokenKind::LeftBrace => depth += 1,
                                    TokenKind::RightBrace => depth -= 1,
                                    _ => {}
                                }
                                self.bump();
                            }
                        }
                        let end = expr_span_last(&self.tokens[..self.pos]);
                        return Some(Expr::NewArray {
                            elem_type: "tuple".to_owned(),
                            size: Box::new(size),
                            span: token.span.to(end),
                        });
                    }
                    // Initializer `{ ... }`: count top-level elements.
                    let mut count = 0i64;
                    if self.eat(&TokenKind::LeftBrace).is_some() {
                        while self.peek().kind != TokenKind::RightBrace
                            && self.peek().kind != TokenKind::Eof
                        {
                            if self.eat(&TokenKind::Comma).is_some() {
                                continue;
                            }
                            let mark = self.pos;
                            if self.expression().is_none() {
                                if self.pos == mark {
                                    self.bump();
                                }
                            } else {
                                count += 1;
                            }
                        }
                        self.expect(&TokenKind::RightBrace, "to close array initializer")?;
                    }
                    let end = expr_span_last(&self.tokens[..self.pos]);
                    return Some(Expr::NewArray {
                        elem_type: "tuple".to_owned(),
                        size: Box::new(Expr::Literal(Literal::Int(count))),
                        span: token.span.to(end),
                    });
                }
                // Target-typed `new(...)` — C# 9. The type is inferred
                // from the assignment target. The disambiguation: a
                // target-typed `new` is followed directly by `(` with
                // no type identifier in between. Any `new (` starts
                // one — a type name can never begin with `(`, and
                // `new(variable)` is the common case.
                if self.peek().kind == TokenKind::LeftParen {
                    self.bump();
                    let args = self.argument_list();
                    self.expect(
                        &TokenKind::RightParen,
                        "rubyc.parser.rd.Parser.ctx.close_arg_list",
                    )?;
                    let _end = expr_span_last(&self.tokens[..self.pos]);
                    if self.peek().kind == TokenKind::LeftBrace {
                        self.bump();
                        while self.peek().kind != TokenKind::RightBrace
                            && self.peek().kind != TokenKind::Eof
                        {
                            if self.eat(&TokenKind::Comma).is_some() {
                                continue;
                            }
                            if matches!(self.peek().kind, TokenKind::Identifier(_))
                                && self.peek_at(1).kind == TokenKind::Assign
                            {
                                self.bump();
                                self.bump();
                                let _ = self.expression();
                            } else {
                                let _ = self.expression();
                            }
                        }
                        self.expect(&TokenKind::RightBrace, "to close initializer")?;
                    }
                    return Some(Expr::New {
                        class_name: String::new(),
                        args,
                        span: token.span.to(self.peek_at(0).span),
                    });
                }
                // Anonymous type: `new { Name = expr, ... }`. The `new`
                // is followed directly by `{` (no type name).
                if self.peek().kind == TokenKind::LeftBrace {
                    self.bump();
                    while self.peek().kind != TokenKind::RightBrace
                        && self.peek().kind != TokenKind::Eof
                    {
                        if self.eat(&TokenKind::Comma).is_some() {
                            continue;
                        }
                        let _ = self.expect_identifier("as anonymous member name");
                        if self.eat(&TokenKind::Assign).is_some() {
                            let mark = self.pos;
                            if self.expression().is_none() && self.pos == mark {
                                self.bump();
                            }
                        }
                    }
                    self.expect(&TokenKind::RightBrace, "to close anonymous type")?;
                    return Some(Expr::New {
                        class_name: "anonymous".to_owned(),
                        args: Vec::new(),
                        span: token.span.to(self.peek_at(0).span),
                    });
                }
                // Implicit array: `new[] { e1, e2, ... }` — element
                // type inferred from the initializer. Modeled as a
                // NewArray with unknown element type and the count.
                if self.peek().kind == TokenKind::LeftBracket
                    && self.peek_at(1).kind == TokenKind::RightBracket
                {
                    self.bump(); // '['
                    self.bump(); // ']'
                    let mut count = 0i64;
                    if self.eat(&TokenKind::LeftBrace).is_some() {
                        while self.peek().kind != TokenKind::RightBrace
                            && self.peek().kind != TokenKind::Eof
                        {
                            if self.eat(&TokenKind::Comma).is_some() {
                                continue;
                            }
                            let mark = self.pos;
                            if self.expression().is_none() {
                                if self.pos == mark {
                                    self.bump();
                                }
                            } else {
                                count += 1;
                            }
                        }
                        self.expect(&TokenKind::RightBrace, "to close array initializer")?;
                    }
                    let end = expr_span_last(&self.tokens[..self.pos]);
                    return Some(Expr::NewArray {
                        elem_type: String::new(),
                        size: Box::new(Expr::Literal(Literal::Int(count))),
                        span: token.span.to(end),
                    });
                }
                // `new T[size1, size2, ...]` followed optionally by
                // `[]` suffixes for the element type. The first `[]` after
                // the type holds the sizes; subsequent `[]` belong to the
                // element type. We distinguish from `T[,]` (array type
                // with empty dims) by requiring the first thing inside
                // `[` to NOT be a comma or `]`. The element type may be
                // dotted (`new Ns.Inner[n]`, `new Interop.Sys.IPAddress[c]`);
                // a single-identifier fast path would misread those as a
                // type name with a rank suffix, swallowing the size.
                let mut name_len = 0usize;
                if matches!(self.peek().kind, TokenKind::Identifier(_)) {
                    name_len = 1;
                    while self.peek_at(name_len).kind == TokenKind::Dot
                        && matches!(
                            self.peek_at(name_len + 1).kind,
                            TokenKind::Identifier(_)
                        )
                    {
                        name_len += 2;
                    }
                }
                if name_len > 0
                    && self.peek_at(name_len).kind == TokenKind::LeftBracket
                    && self.peek_at(name_len + 1).kind != TokenKind::RightBracket
                    && self.peek_at(name_len + 1).kind != TokenKind::Comma
                {
                    let mut elem_type = self.expect_identifier("as element type")?;
                    while self.peek().kind == TokenKind::Dot
                        && matches!(self.peek_at(1).kind, TokenKind::Identifier(_))
                    {
                        self.bump(); // '.'
                        elem_type.push('.');
                        elem_type.push_str(&self.expect_identifier("as element type")?);
                    }
                    self.bump(); // '['
                    // One or more comma-separated sizes. The first size is
                    // the element count; multi-dimensional sizes are parsed
                    // but only the first is stored (jagged arrays are
                    // modeled as 1-D for now).
                    let size = self.expression()?;
                    while self.eat(&TokenKind::Comma).is_some() {
                        let _ = self.expression()?;
                    }
                    self.expect(&TokenKind::RightBracket, "to close array size")?;
                    // Consume any `[]` suffixes (jagged array element type).
                    while self.peek().kind == TokenKind::LeftBracket
                        && self.peek_at(1).kind == TokenKind::RightBracket
                    {
                        self.bump();
                        self.bump();
                    }
                    // Optional trailing initializer: `new byte[1] { 0 }`
                    // is legal C# (size + initializer). Consume balanced
                    // braces so the statement `;` is found.
                    if self.peek().kind == TokenKind::LeftBrace {
                        self.bump(); // '{'
                        let mut depth = 1i32;
                        while depth > 0 && self.peek().kind != TokenKind::Eof {
                            match self.peek().kind {
                                TokenKind::LeftBrace => depth += 1,
                                TokenKind::RightBrace => depth -= 1,
                                _ => {}
                            }
                            self.bump();
                        }
                    }
                    let end = expr_span_last(&self.tokens[..self.pos]);
                    return Some(Expr::NewArray {
                        elem_type,
                        size: Box::new(size),
                        span: token.span.to(end),
                    });
                }
                // Use the full type-name parser so `new` accepts qualified
                // (`System.Collections.Hashtable`) and global-namespace
                // (`global::Ns`) class names, and any combination of
                // array / nullable suffixes (`T[]`, `T[][]`, `T[,]`, `T?[]`).
                let class_name = self.type_name()?;
                let full_name = class_name.clone();
                // Detect whether the type ended in `[]` (array type) — in
                // that case, a following `{` is an array initializer, not
                // an object initializer.
                let is_array_type = full_name.ends_with("[]")
                    || full_name.contains('[');
                let mut args = Vec::new();
                if self.eat(&TokenKind::LeftParen).is_some() {
                    args = self.argument_list();
                    self.expect(
                        &TokenKind::RightParen,
                        "rubyc.parser.rd.Parser.ctx.close_arg_list",
                    )?;
                }
                if is_array_type {
                    // Array initializer: `{ e1, e2, ... }`.
                    if self.eat(&TokenKind::LeftBrace).is_some() {
                        while self.peek().kind != TokenKind::RightBrace
                            && self.peek().kind != TokenKind::Eof
                        {
                            if self.eat(&TokenKind::Comma).is_some() {
                                continue;
                            }
                            let mark = self.pos;
                            if self.expression().is_none() {
                                // Ensure forward progress: bump at least one
                                // token on parse failure so the loop can
                                // converge.
                                if self.pos == mark {
                                    self.bump();
                                }
                            }
                        }
                        self.expect(&TokenKind::RightBrace, "to close array initializer")?;
                    }
                } else if self.peek().kind == TokenKind::LeftBrace {
                    // Object or collection initializer: `{ e1, e2, ... }` or
                    // `{ Key = Value, ... }` or `{ [idx] = Value, ... }`.
                    // The compact AST collapses all of these into a single
                    // shape; the structural round-trip is stable because
                    // both encode/decode sides synthesize the same shape.
                    self.bump();
                    while self.peek().kind != TokenKind::RightBrace
                        && self.peek().kind != TokenKind::Eof
                    {
                        if self.eat(&TokenKind::Comma).is_some() {
                            continue;
                        }
                        // Index-initializer: `[expr] = value`.
                        if self.peek().kind == TokenKind::LeftBracket {
                            self.bump(); // '['
                            let mark = self.pos;
                            if self.expression().is_none() && self.pos == mark {
                                self.bump();
                            }
                            self.expect(&TokenKind::RightBracket, "to close index in initializer")?;
                            if self.eat(&TokenKind::Assign).is_some() {
                                let mark = self.pos;
                                if self.expression().is_none() && self.pos == mark {
                                    self.bump();
                                }
                            }
                        }
                        // `Key = Value` (object initializer).
                        else if matches!(self.peek().kind, TokenKind::Identifier(_))
                            && self.peek_at(1).kind == TokenKind::Assign
                        {
                            self.bump();
                            self.bump(); // '='
                            let mark = self.pos;
                            if self.expression().is_none() && self.pos == mark {
                                self.bump();
                            }
                        }
                        // Collection initializer: just a value.
                        else {
                            let mark = self.pos;
                            if self.expression().is_none() {
                                if self.pos == mark {
                                    self.bump();
                                }
                            }
                        }
                    }
                    self.expect(&TokenKind::RightBrace, "to close initializer")?;
                }
                let end = expr_span_last(&self.tokens[..self.pos]);
                Some(Expr::New {
                    class_name: full_name,
                    args,
                    span: token.span.to(end),
                })
            }
            TokenKind::Keyword(Keyword::This) => {
                self.bump();
                // `this` binds like a variable: slot 0 of every method frame.
                Some(Expr::Ident {
                    name: "this".to_owned(),
                    span: self.peek().span,
                })
            }
            TokenKind::Keyword(Keyword::Base) => {
                self.bump();
                Some(Expr::Ident {
                    name: "base".to_owned(),
                    span: self.peek().span,
                })
            }
            TokenKind::Keyword(Keyword::Ref_) => {
                // `ref expr` — take a reference (parse-only, semantics later).
                self.bump();
                self.expression()
            }
            TokenKind::Keyword(Keyword::True) => {
                self.bump();
                Some(Expr::Literal(Literal::Bool(true)))
            }
            TokenKind::Keyword(Keyword::False) => {
                self.bump();
                Some(Expr::Literal(Literal::Bool(false)))
            }
            TokenKind::Keyword(Keyword::Null) => {
                self.bump();
                Some(Expr::Literal(Literal::Null))
            }
            TokenKind::Keyword(Keyword::Nameof_) if self.peek_at(1).kind == TokenKind::LeftParen => {
                // `nameof(expr)` — returns the name of the expression as a string.
                // Consumes tokens until the matching ')', handling nested
                // parens/brackets: `nameof(x)`, `nameof(A.B.C)`, `nameof(m())`,
                // `nameof(arr[0])`, etc.
                self.bump(); // 'nameof'
                self.bump(); // '('
                let mut depth = 1usize;
                while depth > 0 {
                    match self.peek().kind {
                        TokenKind::LeftParen
                        | TokenKind::LeftBracket
                        | TokenKind::LeftBrace => depth += 1,
                        TokenKind::RightParen
                        | TokenKind::RightBracket
                        | TokenKind::RightBrace => depth -= 1,
                        TokenKind::Eof => break,
                        _ => {}
                    }
                    if depth > 0 {
                        self.bump();
                    }
                }
                // depth == 0 → current token is the closing ')'
                self.bump();
                // For now return a placeholder string; the actual name extraction
                // is a compile-time constant fold (TODO: proper nameof lowering).
                Some(Expr::Literal(Literal::Str("name".to_owned())))
            }
            TokenKind::Keyword(Keyword::Throw) => {
                // `throw expr` — throw expression (C# 7+)
                self.bump();
                let inner = self.unary()?;
                let span = token.span.to(expr_span_last(&self.tokens[..self.pos]));
                Some(Expr::Unary {
                    op: UnOp::Throw,
                    inner: Box::new(inner),
                    span,
                })
            }
            TokenKind::Keyword(Keyword::Delegate_)
                if self.peek_at(1).kind == TokenKind::LeftParen
                    || self.peek_at(1).kind == TokenKind::LeftBrace =>
            {
                // Anonymous method: `delegate(params) { body }` or
                // parameterless `delegate { body }` (signature inferred
                // from the target delegate type, per C#). We consume the
                // whole construct as a single expression (treated as a
                // NewArray of zero elements). The compact AST does not
                // model closures.
                self.bump(); // 'delegate'
                if self.peek().kind == TokenKind::LeftParen {
                    self.expect(&TokenKind::LeftParen, "to open anonymous method params")?;
                    while self.peek().kind != TokenKind::RightParen
                        && self.peek().kind != TokenKind::Eof
                    {
                        self.bump();
                    }
                    self.expect(&TokenKind::RightParen, "to close anonymous method params")?;
                }
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
                }
                Some(Expr::NewArray {
                    elem_type: "anonymous_method".to_owned(),
                    size: Box::new(Expr::Literal(Literal::Int(0))),
                    span: token.span.to(self.peek_at(0).span),
                })
            }
            TokenKind::Keyword(Keyword::From) if matches!(self.peek_at(1).kind, TokenKind::Identifier(_)) => {
                // LINQ query expression. Swallow tokens up to the
                // terminating `;` so the surrounding expression is
                // balanced. The compact AST treats the whole query
                // as a single string literal.
                let mut text = String::from("query:");
                let mut depth = 0i32;
                // A closer at depth 0 ends the query without being
                // consumed: it belongs to an enclosing construct — the
                // parens wrapping `(from ... select ...)`, or the arg
                // list in `Foo(from ... select ...)`. Swallowing it
                // leaves the outer expression unbalanced (`expected ')'`,
                // as in TypedParts' `(... select new {...}).ToArray()`).
                while depth > 0
                    || !matches!(
                        self.peek().kind,
                        TokenKind::Semicolon
                            | TokenKind::Eof
                            | TokenKind::Comma
                            | TokenKind::RightParen
                            | TokenKind::RightBrace
                            | TokenKind::RightBracket
                    )
                {
                    if matches!(self.peek().kind, TokenKind::LeftParen | TokenKind::LeftBrace | TokenKind::LeftBracket) {
                        depth += 1;
                    } else if matches!(self.peek().kind, TokenKind::RightParen | TokenKind::RightBrace | TokenKind::RightBracket) {
                        depth -= 1;
                    }
                    let s = format!("{:?}", self.peek().kind);
                    let s = s.trim_matches('\'').to_string();
                    text.push(' ');
                    text.push_str(&s);
                    self.bump();
                }
                Some(Expr::Literal(Literal::Str(text)))
            }
            TokenKind::Keyword(Keyword::Stackalloc) => {
                // `stackalloc T[n]` or `stackalloc T[] { ... }` — C# 11
                // stack-allocated array. `T` is dotted/generic
                // (`Interop.Sys.IOVector`) or a tuple (`(int, int)`);
                // `qualified_name` is used instead of `type_name`
                // because the latter folds `[N]` into the type, hiding
                // the size from us. Modeled as a NewArray.
                self.bump();
                let mark = self.pos;
                let err_mark = self.errors.len();
                // Tuple element type first: `(int, int)[N]` (shared
                // helper validates the shape and consumes through `)`;
                // the brackets that follow are the size, not a suffix).
                let mut tuple_type: Option<String> = None;
                if self.peek().kind == TokenKind::LeftParen {
                    tuple_type = self.consume_tuple_type();
                }
                let elem_opt = tuple_type.or_else(|| {
                    self.qualified_name().map(|mut t| {
                        // Pointer suffixes (`Vector128<float>*[3]`):
                        // fold into the recorded type, like `type_name`.
                        while self.peek().kind == TokenKind::Star {
                            self.bump();
                            t.push('*');
                        }
                        t
                    })
                });
                if let Some(elem_type) = elem_opt {
                    // Sized: `T[n]` (only the first size is stored), or
                    // `T[] { ... }` initializer (size = element count).
                    if self.peek().kind == TokenKind::LeftBracket {
                        self.bump(); // '['
                        if self.peek().kind == TokenKind::RightBracket {
                            self.bump(); // ']'
                            let mut count = 0i64;
                            if self.eat(&TokenKind::LeftBrace).is_some() {
                                while self.peek().kind != TokenKind::RightBrace
                                    && self.peek().kind != TokenKind::Eof
                                {
                                    if self.eat(&TokenKind::Comma).is_some() {
                                        continue;
                                    }
                                    let emark = self.pos;
                                    if self.expression().is_none() {
                                        if self.pos == emark {
                                            self.bump();
                                        }
                                    } else {
                                        count += 1;
                                    }
                                }
                                self.expect(
                                    &TokenKind::RightBrace,
                                    "to close stackalloc initializer",
                                )?;
                            }
                            return Some(Expr::NewArray {
                                elem_type,
                                size: Box::new(Expr::Literal(Literal::Int(count))),
                                span: token.span.to(self.peek_at(0).span),
                            });
                        }
                        let size = self.expression()?;
                        self.expect(&TokenKind::RightBracket, "to close stackalloc size")?;
                        // Optional trailing initializer: `stackalloc T[N]
                        // { ... }` lists the elements (the size stays).
                        if self.peek().kind == TokenKind::LeftBrace {
                            self.bump(); // '{'
                            let mut depth = 1i32;
                            while depth > 0 && self.peek().kind != TokenKind::Eof {
                                match self.peek().kind {
                                    TokenKind::LeftBrace => depth += 1,
                                    TokenKind::RightBrace => depth -= 1,
                                    _ => {}
                                }
                                self.bump();
                            }
                        }
                        return Some(Expr::NewArray {
                            elem_type,
                            size: Box::new(size),
                            span: token.span.to(self.peek_at(0).span),
                        });
                    }
                }
                // Not a type-led stackalloc: rewind and report (a
                // partial parse may have errored speculatively).
                self.pos = mark;
                self.errors.truncate(err_mark);
                self.unexpected(
                    "rubyc.parser.rd.Parser.unexpected_expression",
                    &describe(self.peek()),
                    self.peek().span,
                );
                None
            }
            TokenKind::Identifier(name) if self.peek_at(1).kind == TokenKind::LeftParen => {
                let callee = name.clone();
                self.bump();
                self.bump();
                // `typeof(T)` / `sizeof(T)` take a TYPE, not an
                // expression — `typeof(AccessRule<T>)` would otherwise
                // misparse the generic brackets as comparisons. Record
                // the type name as the single argument.
                if callee == "typeof" || callee == "sizeof" {
                    // Tuple type: `typeof((TCollection, TKey, TValue))`
                    // (System.Text.Json cache keys). Opaque — the inner
                    // shape never affects lowering, like other compact
                    // type names.
                    let type_name = if self.peek().kind == TokenKind::LeftParen {
                        self.skip_balanced_bracket();
                        String::from("tuple")
                    } else {
                        self.type_name()?
                    };
                    let close = self.expect(
                        &TokenKind::RightParen,
                        "rubyc.parser.rd.Parser.ctx.close_arg_list",
                    );
                    return Some(Expr::Call {
                        callee,
                        args: vec![Expr::Ident {
                            name: type_name,
                            span: token.span,
                        }],
                        span: token.span.to(close.map_or(self.peek().span, |t| t.span)),
                    });
                }
                let args = self.argument_list();
                let close = self.expect(
                    &TokenKind::RightParen,
                    "rubyc.parser.rd.Parser.ctx.close_arg_list",
                );
                Some(Expr::Call {
                    callee,
                    args,
                    span: token.span.to(close.map_or(self.peek().span, |t| t.span)),
                })
            }
            TokenKind::Keyword(Keyword::Checked)
            | TokenKind::Keyword(Keyword::Unchecked)
                if self.peek_at(1).kind == TokenKind::LeftParen =>
            {
                // `checked(expr)` / `unchecked(expr)` are C# overflow-control
                // operators. The backend does not model overflow checking, so
                // we desugar to the inner expression (a pass-through).
                self.bump(); // 'checked' / 'unchecked'
                self.bump(); // '('
                let inner = self.expression()?;
                self.expect(
                    &TokenKind::RightParen,
                    "rubyc.parser.rd.Parser.ctx.close_arg_list",
                );
                Some(inner)
            }
            TokenKind::Identifier(name) => {
                let name_span = self.peek().span;
                self.bump();
                Some(Expr::Ident {
                    name: name.clone(),
                    span: name_span,
                })
            }
            // C# contextual keywords are legal value names
            // (`file.Length`, `required`, ...). Read the source text so
            // the name matches its usage.
            TokenKind::Keyword(
                Keyword::File
                | Keyword::Required
                | Keyword::Scoped
                | Keyword::Record_
                | Keyword::Async
                | Keyword::Await
                | Keyword::Yield
                | Keyword::With
                | Keyword::From
                | Keyword::And_
                | Keyword::Or_
                | Keyword::Not_
                | Keyword::When_,
            ) => {
                let name_span = self.peek().span;
                let bytes =
                    &self.source[name_span.start as usize..name_span.end as usize];
                let name = std::str::from_utf8(bytes).unwrap_or("").to_owned();
                self.bump();
                Some(Expr::Ident { name, span: name_span })
            }
            TokenKind::StringLiteral(value) => {
                self.bump();
                Some(Expr::Literal(Literal::Str(value.clone())))
            }
            TokenKind::InterpolatedString { raw, verbatim } => {
                let span = self.peek().span;
                self.bump();
                let parts = self.build_interp_parts(raw, *verbatim)?;
                Some(Expr::InterpolatedString {
                    parts,
                    span,
                })
            }
            TokenKind::IntLiteral(value) => {
                self.bump();
                Some(Expr::Literal(Literal::Int(*value as i64)))
            }
            TokenKind::FloatLiteral(bits) => {
                self.bump();
                Some(Expr::Literal(Literal::Float(f64::from_bits(*bits))))
            }
            other => {
                self.unexpected(
                    "rubyc.parser.rd.Parser.unexpected_expression",
                    &format!("{other:?}"),
                    token.span,
                );
                None
            }
        }
    }

}