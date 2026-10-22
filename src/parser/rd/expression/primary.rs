//! Expressions: precedence climbing, unary, postfix chains, primaries.

use crate::models::{Diagnostic, Expr, Keyword, Literal, Severity, TokenKind, UnOp};

use super::super::{Parser, describe, expr_span_last};

/// The string a `nameof(...)` argument evaluates to: the last simple
/// name, exactly as C# defines it.
///
/// Accepted shapes (everything else has no name, so the caller reports
/// it):
///   `x`  `this`  `int`  `default`      — a simple name
///   `A.B`  `a.b.c`  `List<int>.Node`  — member access
///   `m()`  `a.m()`                    — an invocation names its method
///   `x.y`  where `x` is `this`/`base`
///
/// The argument text is NOT source text: generic arguments are skipped
/// wholesale, so `nameof(List<int>.Node)` is "Node", not "List<int>".
fn nameof_name(parts: &[TokenKind]) -> Option<String> {
    let mut i = 0usize;
    // First segment: a name (identifier or keyword like `int`).
    let mut last = match parts.first()? {
        TokenKind::Identifier(n) => n.clone(),
        TokenKind::Keyword(kw) => format!("{kw:?}").to_lowercase(),
        _ => return None,
    };
    i += 1;
    loop {
        // A generic argument list belongs to the previous segment.
        if matches!(parts.get(i), Some(TokenKind::Less)) {
            let mut depth = 0usize;
            while let Some(t) = parts.get(i) {
                match t {
                    TokenKind::Less => depth += 1,
                    TokenKind::Greater => depth -= 1,
                    // `>>` / `>=` close two / compare: a `nameof`
                    // argument is never a comparison, so treat the
                    // shift operators as two closers.
                    TokenKind::Shr => depth = depth.saturating_sub(2),
                    TokenKind::ShrShr => depth = depth.saturating_sub(3),
                    _ => {}
                }
                i += 1;
                if depth == 0 {
                    break;
                }
            }
        }
        match parts.get(i) {
            Some(TokenKind::Dot) => {
                i += 1;
                last = match parts.get(i)? {
                    TokenKind::Identifier(n) => n.clone(),
                    TokenKind::Keyword(kw) => format!("{kw:?}").to_lowercase(),
                    _ => return None,
                };
                i += 1;
            }
            // A call names the callee: `nameof(m())` is "m". Arguments
            // inside the call do not participate.
            Some(TokenKind::LeftParen) => {
                let mut depth = 0usize;
                while let Some(t) = parts.get(i) {
                    match t {
                        TokenKind::LeftParen => depth += 1,
                        TokenKind::RightParen => {
                            depth -= 1;
                            i += 1;
                            if depth == 0 {
                                break;
                            }
                            continue;
                        }
                        _ => {}
                    }
                    i += 1;
                }
                return Some(last);
            }
            None => return Some(last),
            _ => return None,
        }
    }
}
impl Parser {
    /// Render the next `n` tokens as a type spelling (`Box<int>`,
    /// `Ns.Outer<T>.Inner`, `int[]?`). Reconstructing from tokens keeps
    /// the element type identical to every other type in the compiler:
    /// a source slice cannot see past a comment, an attribute, or a
    /// non-ASCII identifier.
    pub(in crate::parser::rd) fn render_type_tokens(&self, n: usize) -> String {
        let mut out = String::new();
        for k in 0..n {
            match self.peek_at(k).kind {
                TokenKind::Identifier(ref name) => out.push_str(name),
                TokenKind::Dot => out.push('.'),
                TokenKind::Less | TokenKind::Shl => out.push('<'),
                TokenKind::Greater => out.push('>'),
                TokenKind::Shr => out.push_str(">>"),
                TokenKind::ShrShr => out.push_str(">>>"),
                TokenKind::Comma => out.push(','),
                TokenKind::Question => out.push('?'),
                TokenKind::Star => out.push('*'),
                TokenKind::LeftBracket => out.push('['),
                TokenKind::RightBracket => out.push(']'),
                TokenKind::Keyword(kw) => {
                    out.push_str(&format!("{kw:?}").to_lowercase());
                }
                // Nothing else belongs in a type spelling; drop it
                // rather than inventing text for it.
                _ => {}
            }
        }
        out
    }

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
                    match self.expression() {
                        None => {
                            if self.pos == mark {
                                self.bump();
                            }
                        }
                        Some(e) => elements.push(e),
                    }
                }
                self.expect(&TokenKind::RightBracket, "to close collection expression")?;
                return Some(Expr::NewArray {
                    elem_type: String::new(),
                    size: Box::new(Expr::Literal(Literal::Int(elements.len() as i64))),
                    init: elements,
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
                    match self.expression() {
                        None => {
                            if self.pos == mark {
                                self.bump();
                            }
                        }
                        Some(e) => elements.push(e),
                    }
                }
                self.expect(&TokenKind::RightBrace, "to close array literal")?;
                return Some(Expr::NewArray {
                    elem_type: String::new(),
                    size: Box::new(Expr::Literal(Literal::Int(elements.len() as i64))),
                    init: elements,
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
                        if let Some(e) = self.expression() {
                            elements.push(e);
                        } else {
                            if self.pos == mark {
                                self.bump();
                            }
                            elements.push(Expr::Literal(Literal::Int(0)));
                        }
                        if self.eat(&TokenKind::Comma).is_none() {
                            break;
                        }
                    }
                    self.expect(&TokenKind::RightParen, "to close tuple literal")?;
                    // Tuple literal keeps its elements: it lowers as a
                    // dense `tuple[]` array (8-byte slots, matching the
                    // element-read path and deconstruction below). The
                    // old code discarded values into an empty array, so
                    // every deconstructed name read back zero (the
                    // `CompareStringIgnoreCase` infinite loop that OOMed
                    // init: `index += 0` forever).
                    return Some(Expr::NewArray {
                        elem_type: "tuple".to_owned(),
                        size: Box::new(Expr::Literal(Literal::Int(elements.len() as i64))),
                        span: token.span.to(self.peek_at(0).span),
                        init: elements,
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
                    let type_name = self.type_name_full()?;
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
                            init: Vec::new(),
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
                        init: Vec::new(),
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
                        init: Vec::new(),
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
                        init: Vec::new(),
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
                    // The elements ARE the value: dropping them produced an
                    // array of the right length with every slot zero, which
                    // compiles, runs, and is silently wrong.
                    let mut elements: Vec<Expr> = Vec::new();
                    if self.eat(&TokenKind::LeftBrace).is_some() {
                        while self.peek().kind != TokenKind::RightBrace
                            && self.peek().kind != TokenKind::Eof
                        {
                            if self.eat(&TokenKind::Comma).is_some() {
                                continue;
                            }
                            let mark = self.pos;
                            match self.expression() {
                                Some(e) => elements.push(e),
                                None => {
                                    if self.pos == mark {
                                        self.bump();
                                    }
                                }
                            }
                        }
                        self.expect(&TokenKind::RightBrace, "to close array initializer")?;
                    }
                    let end = expr_span_last(&self.tokens[..self.pos]);
                    return Some(Expr::NewArray {
                        elem_type: String::new(),
                        size: Box::new(Expr::Literal(Literal::Int(elements.len() as i64))),
                        span: token.span.to(end),
                        init: elements,
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
                // Rank (`[]`), nullable (`?`), and pointer (`*`)
                // suffixes may sit between the element name and the
                // SIZED bracket (`new ushort[]?[n]`, `new int?[n]`):
                // without this scan the size is swallowed as an empty
                // rank suffix and the whole thing degrades to an
                // unresolvable `New` (null array, faults downstream).
                // Pure lookahead — nothing is consumed here.
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
                let mut suffix_end = name_len;
                let mut suffix = String::new();
                loop {
                    if self.peek_at(suffix_end).kind == TokenKind::LeftBracket
                        && self.peek_at(suffix_end + 1).kind == TokenKind::RightBracket
                    {
                        suffix.push_str("[]");
                        suffix_end += 2;
                    } else if self.peek_at(suffix_end).kind == TokenKind::Question {
                        suffix.push('?');
                        suffix_end += 1;
                    } else if self.peek_at(suffix_end).kind == TokenKind::Star {
                        suffix.push('*');
                        suffix_end += 1;
                    } else {
                        break;
                    }
                }
                if name_len > 0
                    && self.peek_at(suffix_end).kind == TokenKind::LeftBracket
                    && self.peek_at(suffix_end + 1).kind != TokenKind::RightBracket
                    && self.peek_at(suffix_end + 1).kind != TokenKind::Comma
                {
                    let mut elem_type = self.expect_identifier("as element type")?;
                    while self.peek().kind == TokenKind::Dot
                        && matches!(self.peek_at(1).kind, TokenKind::Identifier(_))
                    {
                        self.bump(); // '.'
                        elem_type.push('.');
                        elem_type.push_str(&self.expect_identifier("as element type")?);
                    }
                    // Consume the scanned suffix run in the exact
                    // lookahead order (`suffix` already holds the text —
                    // just bump past the tokens here).
                    loop {
                        if self.peek().kind == TokenKind::LeftBracket
                            && self.peek_at(1).kind == TokenKind::RightBracket
                        {
                            self.bump(); // '['
                            self.bump(); // ']'
                        } else if matches!(
                            self.peek().kind,
                            TokenKind::Question | TokenKind::Star
                        ) {
                            self.bump();
                        } else {
                            break;
                        }
                    }
                    elem_type.push_str(&suffix);
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
                        // `new T[n] { e1, e2 }` — legal when the count
                        // matches. The ELEMENTS are the value (the size
                        // must agree), so the array is built from them.
                        // Reporting instead would leave the written
                        // values out of a value the program already has.
                        self.bump(); // '{'
                        let mut elements: Vec<Expr> = Vec::new();
                        while self.peek().kind != TokenKind::RightBrace
                            && self.peek().kind != TokenKind::Eof
                        {
                            if self.eat(&TokenKind::Comma).is_some() {
                                continue;
                            }
                            let mark = self.pos;
                            match self.expression() {
                                Some(e) => elements.push(e),
                                None => {
                                    if self.pos == mark {
                                        self.bump();
                                    }
                                }
                            }
                        }
                        self.expect(&TokenKind::RightBrace, "to close array initializer")?;
                        let end = expr_span_last(&self.tokens[..self.pos]);
                        return Some(Expr::NewArray {
                            elem_type,
                            size: Box::new(Expr::Literal(Literal::Int(
                                elements.len() as i64,
                            ))),
                            span: token.span.to(end),
                            init: elements,
                        });
                    }
                    let end = expr_span_last(&self.tokens[..self.pos]);
                    return Some(Expr::NewArray {
                        elem_type,
                        size: Box::new(size),
                        span: token.span.to(end),
                        init: Vec::new(),
                    });
                }
                // Generic array creation (`new Box<int>[n]`): the
                // non-generic fast path above only fires for bare
                // `Identifier[`, and `type_name()` below swallows the
                // sized brackets as an empty `[]` rank suffix (losing
                // the size). Detect `Ident (<...>)? [` here via
                // lookahead and parse manually, slicing the element
                // spelling from source.
                if matches!(self.peek().kind, TokenKind::Identifier(_)) {
                    let mut j = 1usize;
                    while self.peek_at(j).kind == TokenKind::Dot
                        && matches!(
                            self.peek_at(j + 1).kind,
                            TokenKind::Identifier(_)
                        )
                    {
                        j += 2;
                    }
                    if matches!(
                        self.peek_at(j).kind,
                        TokenKind::Less | TokenKind::Shl
                    ) {
                                let mut depth = 0i32;
                        loop {
                            match self.peek_at(j).kind {
                                TokenKind::Less | TokenKind::Shl => depth += 1,
                                TokenKind::Greater | TokenKind::Shr | TokenKind::ShrShr => {
                                    depth -= 1;
                                }
                                TokenKind::Eof => break,
                                _ => {}
                            }
                            j += 1;
                            if depth == 0 {
                                break;
                            }
                        }
                    }
                    if self.peek_at(j).kind == TokenKind::LeftBracket
                        && self.peek_at(j + 1).kind != TokenKind::RightBracket
                        && self.peek_at(j + 1).kind != TokenKind::Comma
                    {
                        // Element spelling from TOKENS, `Ident` through
                        // the end of the type (before `[`). A source slice
                        // begins at `new`, whose own `n` is the first ASCII
                        // letter, so `new Box<int>[2]` used to spell its
                        // element `new Box<int>` — a name that matches no
                        // entry and silently fell back to a default layout.
                        let elem_type = self.render_type_tokens(j);
                        // Consume through `[size]`.
                        while self.pos < self.tokens.len()
                            && self.peek().kind != TokenKind::LeftBracket
                        {
                            self.bump();
                        }
                        self.bump(); // '['
                        let size = self.expression()?;
                        while self.eat(&TokenKind::Comma).is_some() {
                            let _ = self.expression()?;
                        }
                        self.expect(&TokenKind::RightBracket, "to close array size")?;
                        let end = expr_span_last(&self.tokens[..self.pos]);
                        return Some(Expr::NewArray {
                            elem_type,
                            size: Box::new(size),
                            span: token.span.to(end),
                            init: Vec::new(),
                        });
                    }
                }
                // Use the full type-name parser so `new` accepts qualified
                // (`System.Collections.Hashtable`) and global-namespace
                // (`global::Ns`) class names, generic instantiations
                // (`new Task<VoidTaskResult>(...)` must resolve the
                // generic entry's own ctors — stripping the arguments
                // misconstructs, e.g. a fresh Task whose
                // `IsCompleted` is false trips `CheckAsyncTaskInProgress`
                // or a throwing ctor leaves singletons null), and any
                // combination of array / nullable suffixes
                // (`T[]`, `T[][]`, `T[,]`, `T?[]`).
                let class_name = self.type_name_full()?;
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
                    // `new int[] { e1, e2, ... }` IS an array value: the
                    // elements are the array, and dropping them (as this
                    // used to) produced a null/empty array that still
                    // compiled and ran.
                    let mut elements: Vec<Expr> = Vec::new();
                    if self.eat(&TokenKind::LeftBrace).is_some() {
                        while self.peek().kind != TokenKind::RightBrace
                            && self.peek().kind != TokenKind::Eof
                        {
                            if self.eat(&TokenKind::Comma).is_some() {
                                continue;
                            }
                            let mark = self.pos;
                            match self.expression() {
                                Some(e) => elements.push(e),
                                None => {
                                    // Forward progress on a parse failure.
                                    if self.pos == mark {
                                        self.bump();
                                    }
                                }
                            }
                        }
                        self.expect(&TokenKind::RightBrace, "to close array initializer")?;
                    }
                    let end = expr_span_last(&self.tokens[..self.pos]);
                    // `int[]` -> element type `int`; the rank suffixes are
                    // part of the ARRAY, not of its element.
                    let elem_type = full_name.trim_end_matches("[]").to_owned();
                    return Some(Expr::NewArray {
                        elem_type,
                        size: Box::new(Expr::Literal(Literal::Int(elements.len() as i64))),
                        span: token.span.to(end),
                        init: elements,
                    });
                } else if self.peek().kind == TokenKind::LeftBrace {
                    // Object or collection initializer: `{ e1, e2, ... }` or
                    // `{ Key = Value, ... }` or `{ [idx] = Value, ... }`.
                    // `Key = Value` elements are captured as member
                    // writes that run after construction (see
                    // `Expr::New.init`); index- and
                    // collection-initializer elements have no member
                    // name to write through and are still parsed only
                    // (documented gap).
                    let mut init = Vec::new();
                    self.bump();
                    while self.peek().kind != TokenKind::RightBrace
                        && self.peek().kind != TokenKind::Eof
                    {
                        if self.eat(&TokenKind::Comma).is_some() {
                            continue;
                        }
                        // Index-initializer: `[expr] = value`.
                        if self.peek().kind == TokenKind::LeftBracket {
                            // `["key"] = value` — an INDEX initializer:
                            // `set_Item(key, value)` after construction.
                            // It used to vanish (the collection came back
                            // empty) and then to be reported.
                            let idx_span = self.peek().span;
                            self.bump(); // '['
                            let mut index: Option<Expr> = None;
                            let mark = self.pos;
                            match self.expression() {
                                Some(e) => index = Some(e),
                                None => {
                                    if self.pos == mark {
                                        self.bump();
                                    }
                                }
                            }
                            self.expect(&TokenKind::RightBracket, "to close index in initializer")?;
                            if self.eat(&TokenKind::Assign).is_some() {
                                let vmark = self.pos;
                                match self.expression() {
                                    Some(value) => init.push(crate::models::ast::MemberInit {
                                        name: String::new(),
                                        value,
                                        index,
                                        span: idx_span,
                                    }),
                                    None if self.pos == vmark => {
                                        self.bump();
                                    }
                                    None => {}
                                }
                            }
                        }
                        // `Key = Value` (object initializer).
                        else if matches!(self.peek().kind, TokenKind::Identifier(_))
                            && self.peek_at(1).kind == TokenKind::Assign
                        {
                            let key_span = self.peek().span;
                            let Some(key) = self.expect_identifier("as initializer member name") else {
                                self.bump(); // '='
                                let mark = self.pos;
                                if self.expression().is_none() && self.pos == mark {
                                    self.bump();
                                }
                                continue;
                            };
                            self.bump(); // '='
                            let mark = self.pos;
                            match self.expression() {
                                Some(value) => init.push(crate::models::ast::MemberInit {
                                    name: key,
                                    value,
                                    index: None,
                                    span: key_span,
                                }),
                                None if self.pos == mark => {
                                    self.bump();
                                }
                                None => {}
                            }
                        }
                        // Collection initializer: a bare value is an
                        // `Add(value)` call after construction. It used
                        // to be parsed and dropped, so the collection
                        // came back empty; it is recorded with an empty
                        // member name and lowered as `Add`.
                        else {
                            let mark = self.pos;
                            let span = self.peek().span;
                            match self.expression() {
                                Some(value) => init.push(crate::models::ast::MemberInit {
                                    name: String::new(),
                                    value,
                                    index: None,
                                    span,
                                }),
                                None => {
                                    if self.pos == mark {
                                        self.bump();
                                    }
                                }
                            }
                        }
                    }
                    self.expect(&TokenKind::RightBrace, "to close initializer")?;
                    let end = expr_span_last(&self.tokens[..self.pos]);
                    return Some(Expr::New {
                        class_name: full_name,
                        args,
                        init,
                        span: token.span.to(end),
                    });
                }
                let end = expr_span_last(&self.tokens[..self.pos]);
                Some(Expr::New {
                    class_name: full_name,
                    args,
                    init: Vec::new(),
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
                // Exception: `ref *p` (deref) keeps the address
                // (`&*p == p`): dropping `ref` reads the pointee value,
                // faulting ByReference stores (`Span._reference = ref
                // *(T*)pointer` stored byte 0 as null) and mis-passing
                // interop byref args (`Memmove(ref *dst, ...)`).
                // Every other `ref X` keeps legacy value behavior.
                self.bump();
                let start = self.peek().span;
                match self.expression() {
                    Some(deref @ Expr::Unary {
                        op: crate::models::ast::expression::UnOp::Deref,
                        ..
                    }) => {
                        let end = self.peek().span;
                        Some(Expr::Unary {
                            op: crate::models::ast::expression::UnOp::AddrOf,
                            inner: Box::new(deref),
                            span: start.to(end),
                        })
                    }
                    other => other,
                }
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
                // `nameof(expr)` — a compile-time constant naming the
                // LAST simple name of the argument. It used to return
                // the fixed string "name" whatever the argument was, so
                // reflection-style code and property-name-based lookups
                // silently read the wrong key.
                let open_span = self.peek().span;
                self.bump(); // 'nameof'
                self.bump(); // '('
                // Collect the argument's tokens up to the matching ')'.
                let mut depth = 1usize;
                let mut parts: Vec<TokenKind> = Vec::new();
                let mut close_span = open_span;
                while depth > 0 {
                    match self.peek().kind {
                        TokenKind::LeftParen
                        | TokenKind::LeftBracket
                        | TokenKind::LeftBrace => depth += 1,
                        TokenKind::RightParen
                        | TokenKind::RightBracket
                        | TokenKind::RightBrace => {
                            depth -= 1;
                            if depth == 0 {
                                close_span = self.peek().span;
                                break;
                            }
                        }
                        TokenKind::Eof => break,
                        _ => {}
                    }
                    parts.push(self.peek().kind.clone());
                    self.bump();
                }
                // depth == 0 → current token is the closing ')'
                self.bump();
                match nameof_name(&parts) {
                    Some(name) => Some(Expr::Literal(Literal::Str(name))),
                    None => {
                        // C# has no name to give here (`nameof(a + b)`,
                        // `nameof(arr[0])`, a bare literal). Reporting it
                        // beats answering a constant that is wrong.
                        let mut diag = Diagnostic::new(
                            "E0032",
                            Severity::Error,
                            crate::t!("rubyc.parser.rd.expression.primary.nameof_argument"),
                        )
                        .at(open_span.to(close_span))
                        .in_context(self.context.clone());
                        diag.help = Some(
                            "nameof needs a name: a variable, field, property, type, or a call to one"
                                .to_owned(),
                        );
                        self.errors.push(diag);
                        Some(Expr::Literal(Literal::Str(String::new())))
                    }
                }
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
                // parameterless `delegate { body }`. It is a lambda: C#
                // gives it the same semantics (the target delegate supplies
                // the parameter types, the body captures like any closure),
                // so it becomes the same `Expr::Lambda` node. Parameter
                // TYPES are skipped for that reason — `delegate (Node n)`
                // and `(n) =>` are the same value.
                let start = token.span;
                self.bump(); // 'delegate'
                let params = self.consume_lambda_params();
                if self.peek().kind != TokenKind::LeftBrace {
                    self.expect(&TokenKind::LeftBrace, "to open an anonymous method body")?;
                }
                let body = self.block();
                Some(Expr::Lambda {
                    params,
                    body: crate::models::ast::expression::LambdaBody::Block(body),
                    span: start.to(self.peek_at(0).span),
                })
            }
            TokenKind::Keyword(Keyword::From) if matches!(self.peek_at(1).kind, TokenKind::Identifier(_)) => {
                // Query syntax (`from x in xs ... select ...`). It is
                // real C#, but the compiler cannot evaluate it: there
                // is no query lowering, so the whole expression would
                // have to be a fabricated value. It used to become one
                // string literal, so `var q = from ... select x;`
                // compiled and then handed callers that string.
                // Consume the query — so the rest of the file
                // is still parsed and the enclosing expression stays
                // balanced — and report it instead.
                let start = self.peek().span;
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
                    self.bump();
                }
                let mut diag = Diagnostic::new(
                    "E0031",
                    Severity::Error,
                    crate::t!("rubyc.parser.rd.expression.primary.query_expression"),
                )
                .at(start.to(self.peek().span))
                .in_context(self.context.clone());
                diag.help = Some(
                    "rewrite the query with method syntax (Select / Where / OrderBy)".to_owned(),
                );
                self.errors.push(diag);
                Some(Expr::Literal(Literal::Int(0)))
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
                                init: Vec::new(),
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
                            init: Vec::new(),
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
                    // Tuple type: `typeof((TCollection, TKey, TValue))`.
                    // The inner shape does not survive the compact type
                    // model, so the argument becomes the opaque name
                    // "tuple".
                    let type_name = if self.peek().kind == TokenKind::LeftParen {
                        let open = self.peek().span;
                        self.skip_balanced_bracket();
                        // `sizeof` only accepts unmanaged types and a
                        // tuple is a managed one, so C# rejects this
                        // outright. It used to size the opaque "tuple"
                        // name instead, printing a number (8) for a
                        // construct that cannot be compiled.
                        // `typeof` is left alone: it already reports the
                        // real gap downstream (no reflection runtime).
                        if callee == "sizeof" {
                            self.errors.push(
                                Diagnostic::new(
                                    "E0039",
                                    Severity::Error,
                                    crate::t!(
                                        "rubyc.parser.rd.expression.primary.sizeof_tuple"
                                    ),
                                )
                                .at(token.span.to(self.peek_at(0).span))
                                .in_context(self.context.clone()),
                            );
                        }
                        let _ = open;
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
                // `checked(expr)` / `unchecked(expr)` — C# overflow-control
                // operators. Arithmetic in this compiler always wraps, which
                // IS `unchecked`, so that form desugars exactly. `checked`
                // promises an `OverflowException` that never arrives, so it
                // is reported rather than quietly discarded.
                let checked_kw = matches!(self.peek().kind, TokenKind::Keyword(Keyword::Checked));
                let start = self.peek().span;
                self.bump(); // 'checked' / 'unchecked'
                self.bump(); // '('
                let inner = self.expression()?;
                let close = self.expect(
                    &TokenKind::RightParen,
                    "rubyc.parser.rd.Parser.ctx.close_arg_list",
                );
                // `checked` is a real node: the lowering arms overflow
                // checks for the arithmetic and narrowing inside it.
                // `unchecked` is the default model, so it stays the plain
                // expression it always was.
                let span = start.to(close.map_or(self.peek().span, |t| t.span));
                if checked_kw {
                    Some(Expr::Checked {
                        inner: Box::new(inner),
                        span,
                    })
                } else {
                    Some(inner)
                }
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
            TokenKind::CharLiteral(value) => {
                self.bump();
                Some(Expr::Literal(Literal::Char(*value)))
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