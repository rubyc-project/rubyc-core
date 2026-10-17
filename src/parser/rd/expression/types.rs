//! Expressions: precedence climbing, unary, postfix chains, primaries.

use crate::models::{Access, BinOp, Expr, Keyword, Literal, Span, Stmt, TokenKind};

use super::super::{Parser, expr_span_last};
impl Parser {
    pub(in crate::parser::rd) fn argument_list(&mut self) -> Vec<Expr> {
        let mut args = Vec::new();
        if self.peek().kind == TokenKind::RightParen {
            return args;
        }
        loop {
            // Named argument: `name: value`. The `:` separates the
            // argument name from the value; the value is what we
            // record in the AST. The label may be an identifier or a
            // contextual keyword (e.g. `required:`).
            if matches!(
                self.peek().kind,
                TokenKind::Identifier(_) | TokenKind::Keyword(_)
            ) && self.peek_at(1).kind == TokenKind::Colon
            {
                self.bump(); // name
                self.bump(); // ':'
            }
            // Modifier-prefixed parameter: `out int x`, `ref int x`,
            // `in int x`. Consume the modifier and the type identifier;
            // the following identifier is the (newly declared, for `out`)
            // variable name. Expose it to the semantic scan as a zero-init
            // `Let` so references to it resolve.
            // Declared `out` args also rewrite the slot (below).
            let mut out_default: Option<(String, String, Span)> = None;
            // Call-site `ref`/`in` marker (see below): the slot wraps
            // in `&` (address-of) so overload ranking can require a
            // byref parameter — erased markers self-elected a
            // same-name value twin (`WriteAtOffset(handle, buf, ref
            // off)` recursed into itself forever). `out` keeps legacy
            // Ident slots (its declaration/writeback machinery owns
            // those paths).
            let mut ref_mark: Option<Span> = None;
            if matches!(
                self.peek().kind,
                TokenKind::Keyword(Keyword::Out_)
                    | TokenKind::Keyword(Keyword::Ref_)
                    | TokenKind::Keyword(Keyword::In_)
            ) {
                let is_out = matches!(self.peek().kind, TokenKind::Keyword(Keyword::Out_));
                let mod_span = self.peek().span;
                self.bump();
                if !is_out {
                    ref_mark = Some(mod_span);
                }
                // Optionally skip a full type (`out T x`, `out T[]? x`,
                // `out Ns.T? x`, `out (long, long) f`) before the value
                // identifier. A member or index expression
                // (`out helper._f`, `out a[i]`) is a value, not a type
                // — detect by trial: consume a type and keep it only if
                // a name follows, rewinding (with speculative errors
                // dropped) otherwise. A bare `out x` (or `out _`) names
                // an EXISTING variable — declare nothing.
                let mut declared = false;
                // The explicitly-written type (`out bool x` declares a
                // bool — NOT var-inferred from the zero init, which
                // would mistype it as int and break overload ranking).
                // Tuples fold to the synthetic identifier, like params.
                let mut declared_ty: Option<String> = None;
                if matches!(self.peek().kind, TokenKind::Identifier(_))
                    || self.peek().kind == TokenKind::LeftParen
                {
                    let mark = self.pos;
                    let err_mark = self.errors.len();
                    if self.peek().kind == TokenKind::LeftParen {
                        let _ = self.consume_tuple_type();
                        // Suffixes the tuple type may carry belong to
                        // the type, not the value (`out (uint, uint)[]?
                        // ranges` — System.Text.RegularExpressions).
                        // Without these the `[` fails the name check
                        // below and the `?` misparses as a ternary.
                        loop {
                            if self.peek().kind == TokenKind::LeftBracket {
                                self.skip_balanced_bracket();
                            } else if self.peek().kind == TokenKind::Star {
                                self.bump();
                            } else if self.peek().kind == TokenKind::Question
                                && self.question_is_nullable()
                            {
                                self.bump();
                            } else {
                                break;
                            }
                        }
                    } else {
                        declared_ty = self.type_name_full();
                    }
                    if !self.peek_is_name(0) {
                        self.pos = mark;
                        self.errors.truncate(err_mark);
                        declared_ty = None;
                    } else {
                        declared = true;
                    }
                }
                // Only `out` arguments introduce a new variable; `ref` and
                // `in` refer to existing variables. The name may be a
                // contextual keyword (`out T record`) — read source text.
                // Queued pre-statement: the variable is in scope for the
                // call itself and everything after it. The argument slot
                // carries the declared name (rewritten below): the slot
                // is write-only (the callee overwrites), and the `Let`
                // types the name identically for ranking.
                if is_out && declared && self.peek_is_name(0) {
                    let name_span = self.peek().span;
                    let bytes =
                        &self.source[name_span.start as usize..name_span.end as usize];
                    let var_name =
                        std::str::from_utf8(bytes).unwrap_or("").to_owned();
                    let ty = declared_ty.unwrap_or_else(|| String::from("var"));
                    self.pre_pending.push(Stmt::Let {
                        is_byref: false,
                        type_name: ty.clone(),
                        name: var_name.clone(),
                        init: Expr::Literal(Literal::Int(0)),
                        access: Access::Default,
                        span: mod_span.to(name_span),
                    });
                    out_default = Some((ty, var_name, mod_span.to(name_span)));
                }
            }
            match out_default {
                Some((_ty, var_name, span)) => {
                    self.bump(); // consume the out-var name
                    // Slot carries the declared name (not `default(T)`):
                    // `box_byref_args` boxes `Ident` locals WITH a
                    // writeback, so the callee's value flows back into
                    // the variable (inline `out` previously boxed a
                    // nameless default and the write was lost — every
                    // `Try*`/`Fast` out-param read back zero). The
                    // `Let` types the name identically for ranking.
                    // Base-init position keeps the old nameless default
                    // (see `in_base_init`): the name is genuinely
                    // undeclared at the call and an `Ident` would fail
                    // to lower there.
                    if self.in_base_init {
                        args.push(Expr::Default {
                            type_name: _ty,
                            span,
                        });
                    } else {
                        args.push(Expr::Ident {
                            name: var_name,
                            span,
                        });
                    }
                }
                None => match self.expression() {
                    Some(arg) => {
                        if let Some(mspan) = ref_mark {
                            let end = self.peek().span;
                            args.push(Expr::Unary {
                                op: crate::models::ast::expression::UnOp::AddrOf,
                                inner: Box::new(arg),
                                span: mspan.to(end),
                            });
                        } else {
                            args.push(arg);
                        }
                    }
                    None => return args,
                },
            }
            if self.eat(&TokenKind::Comma).is_none() {
                return args;
            }
        }
    }

    /// Look ahead to determine if `(...)` is a tuple literal — a
    /// parenthesized list of expressions separated by commas, as in
    /// `(7, "seven")`. A parenthesized expression like `(x)` has no comma
    /// and is not a tuple.
    /// Generic argument list suffix (`<int, string>`, with `Shl`/`Shr`
    /// lexing for nested `<<`/`>>`): appends the balanced bracket
    /// sequence to `name`. Shared by the leading type and every
    /// dot-qualified segment, so nested generics (`A<X>.B<Y>`,
    /// `ConcurrentDictionary<string, CacheEntry>.AlternateLookup<…>`)
    /// parse as one type instead of stranding `<` as less-than.
    fn generic_args_suffix(&mut self, name: &mut String) {
        // Generic argument list: `A<int, string>`. Note: the lexer may
        // produce `Shl`/`Shr` for `<<`/`>>` (shift operators), which
        // represent TWO `<` or TWO `>`.
        if matches!(self.peek().kind, TokenKind::Less | TokenKind::Shl) {
            let opening_depth = match self.peek().kind {
                TokenKind::Less => 1,
                TokenKind::Shl => 2,
                _ => unreachable!(),
            };
            name.push('<');
            if opening_depth == 2 {
                name.push('<');
            }
            self.bump();
            let mut depth = opening_depth;
            while depth > 0 {
                match self.peek().kind {
                    TokenKind::Eof => break,
                    TokenKind::Less => {
                        depth += 1;
                        name.push('<');
                        self.bump();
                    }
                    TokenKind::Shl => {
                        depth += 2;
                        name.push_str("<<");
                        self.bump();
                    }
                    TokenKind::Greater => {
                        depth -= 1;
                        if depth == 0 {
                            name.push('>');
                            self.bump();
                            break;
                        }
                        name.push('>');
                        self.bump();
                    }
                    TokenKind::Shr => {
                        // Both `>` belong to this list at depth>=2;
                        // shallower depths tolerate junk.
                        // (Dropping one here stranded names like
                        // `A<X<T>>` as `A<X<T>`.)
                        name.push_str(">>");
                        depth -= 2;
                        self.bump();
                        if depth <= 0 {
                            break;
                        }
                    }
                    TokenKind::ShrShr => {
                        // All three, same rule.
                        name.push_str(">>>");
                        depth -= 3;
                        self.bump();
                        if depth <= 0 {
                            break;
                        }
                    }
                    TokenKind::Comma => {
                        name.push(',');
                        self.bump();
                    }
                    TokenKind::Identifier(ref s) => {
                        name.push_str(s);
                        self.bump();
                    }
                    _ => {
                        // Consume any other token (keywords, literals, etc.)
                        // that may appear in generic arguments.
                        self.bump();
                    }
                }
            }
        }
    }

    /// Parses a type name that may be dot-qualified and generic:
    /// `A`, `A.B`, `A<B>`, `A<B>.C`, `A[]`, `A?`, etc.
    pub(in crate::parser::rd) fn qualified_type_name(&mut self) -> String {
        // Function-pointer type: `delegate* [unmanaged[cc]] <...>`.
        // Records as opaque `delegate*` (see declaration `type_name`).
        if matches!(
            self.peek().kind,
            TokenKind::Keyword(Keyword::Delegate_)
        ) && self.peek_at(1).kind == TokenKind::Star
        {
            self.bump(); // 'delegate'
            self.bump(); // '*'
            if matches!(self.peek().kind, TokenKind::Identifier(_)) {
                self.bump(); // e.g. 'unmanaged'
                if self.peek().kind == TokenKind::LeftBracket {
                    let mut depth = 0usize;
                    while self.peek().kind != TokenKind::Eof {
                        match self.peek().kind {
                            TokenKind::LeftBracket => depth += 1,
                            TokenKind::RightBracket => {
                                depth = depth.saturating_sub(1);
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
                }
            }
            if matches!(self.peek().kind, TokenKind::Less | TokenKind::Shl) {
                let mut depth = 0usize;
                while self.peek().kind != TokenKind::Eof {
                    match self.peek().kind {
                        TokenKind::Eof => break,
                        TokenKind::Less => depth += 1,
                        TokenKind::Shl => depth += 2,
                        TokenKind::Greater => {
                            depth = depth.saturating_sub(1);
                            if depth == 0 {
                                self.bump();
                                break;
                            }
                        }
                        TokenKind::Shr => {
                            depth = depth.saturating_sub(2);
                            if depth == 0 {
                                self.bump();
                                break;
                            }
                        }
                        TokenKind::ShrShr => {
                            depth = depth.saturating_sub(3);
                            if depth == 0 {
                                self.bump();
                                break;
                            }
                        }
                        _ => {}
                    }
                    self.bump();
                }
            }
            // Trailing pointer depth: `delegate* ... **` (pointer to
            // function pointer, e.g. vtable slots). Recorded, like the
            // leading star.
            let mut stars = "delegate*".to_owned();
            while self.peek().kind == TokenKind::Star {
                self.bump();
                stars.push('*');
            }
            return stars;
        }
        let mut name = match self.peek().kind.clone() {
            TokenKind::Identifier(n) => {
                self.bump();
                n
            }
            _ => String::new(),
        };
        // Transparent `global::` qualifier
        // (`(global::System.Object)o` in generated serializers): same
        // type as the unqualified spelling.
        if name == "global"
            && self.peek().kind == TokenKind::Colon
            && self.peek_at(1).kind == TokenKind::Colon
        {
            self.bump();
            self.bump();
            if let TokenKind::Identifier(n) = self.peek().kind.clone() {
                self.bump();
                name = n;
            }
        }
        // Dot-qualified segments: `A.B.C`.
        while self.peek().kind == TokenKind::Dot {
            self.bump();
            if let TokenKind::Identifier(seg) = self.peek().kind.clone() {
                self.bump();
                name.push('.');
                name.push_str(&seg);
            } else {
                break;
            }
        }
        // Generic argument list: `A<int, string>`. Note: the lexer may
        // produce `Shl`/`Shr` for `<<`/`>>` (shift operators), which
        // represent TWO `<` or TWO `>`.
        self.generic_args_suffix(&mut name);
        // More dot-qualified segments after generics: `List<int>.Enumerator`.
        // Each segment may itself carry generic arguments
        // (`A<X>.B<Y>`): consume them here, or the `<` strands as
        // less-than downstream.
        while self.peek().kind == TokenKind::Dot {
            self.bump();
            if let TokenKind::Identifier(seg) = self.peek().kind.clone() {
                self.bump();
                name.push('.');
                name.push_str(&seg);
                self.generic_args_suffix(&mut name);
            } else {
                break;
            }
        }
        // Suffixes interleave in any order (`T?`, `T[]?`, `T*[]`,
        // …): loop until none remains.
        loop {
            let mut progressed = false;
            if self.peek().kind == TokenKind::Question {
                name.push('?');
                self.bump();
                progressed = true;
            }
            // Pointer suffixes: `T*`, `T**`.
            while self.peek().kind == TokenKind::Star {
                name.push('*');
                self.bump();
                progressed = true;
            }
            // Array suffixes: `[]`.
            while self.peek().kind == TokenKind::LeftBracket {
                self.bump();
                if self.peek().kind == TokenKind::RightBracket {
                    name.push_str("[]");
                    self.bump();
                } else {
                    // Multi-dimensional: `[, ...]`.
                    name.push('[');
                    while self.peek().kind != TokenKind::RightBracket
                        && self.peek().kind != TokenKind::Eof
                    {
                        name.push(',');
                        self.bump();
                    }
                    name.push(']');
                    self.bump();
                }
                progressed = true;
            }
            if !progressed {
                break;
            }
        }
        name
    }

    /// Lookahead helper for cast detection: given the index of a `[`,
    /// return the index just past its matching `]` (nesting-aware).
    /// Returns the input index unchanged when unbalanced.
    pub(in crate::parser::rd) fn skip_balanced_bracket_at(&self, mut i: usize) -> usize {
        let mut depth = 0usize;
        loop {
            match self.peek_at(i).kind {
                TokenKind::Eof => return i,
                TokenKind::LeftBracket => depth += 1,
                TokenKind::RightBracket => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        return i + 1;
                    }
                }
                _ => {}
            }
            i += 1;
        }
    }

    /// Advance a cast-gate lookahead index over interleaved type
    /// suffixes (`?`/`*`/`[...]`) and dotted nested-type segments
    /// (`.Member`, each with its own generic list and suffixes):
    /// `(A<T>.B<U>[])x`, `(Ns.Dictionary<K,V>.Enumerator)x`. Pure
    /// lookahead — consumes nothing. Shared by the cast gate's
    /// generic and dot-qualified arms (both previously demanded `)`
    /// right after one type-argument list).
    pub(in crate::parser::rd) fn cast_gate_suffixes(&self, mut i: usize) -> usize {
        loop {
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
                // Only EMPTY brackets (`[]`, `[,]`) are type rank
                // specifiers: a bracket holding an expression (`[o]`)
                // is an indexer value (`(t.C[o].M)` is a grouped
                // member chain, never a cast). `skip_balanced_bracket`
                // consumes any bracket, so gate emptiness explicitly
                // (mirrors `skip_suffixes_at`, narrowed the same way).
                while self.peek_at(i).kind == TokenKind::LeftBracket
                    && matches!(
                        self.peek_at(i + 1).kind,
                        TokenKind::RightBracket | TokenKind::Comma
                    )
                {
                    let n = self.skip_balanced_bracket_at(i);
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
            // Another `.Member` segment?
            if self.peek_at(i).kind != TokenKind::Dot
                || !matches!(self.peek_at(i + 1).kind, TokenKind::Identifier(_))
            {
                break;
            }
            i += 2;
            // Per-segment generic list (`>>`/`>>>` lex as Shr/ShrShr).
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
        }
        i
    }

    pub(in crate::parser::rd) fn contains_tuple_literal_ahead(&self) -> bool {
        if self.peek().kind != TokenKind::LeftParen {
            return false;
        }
        let mut offset = 1usize;
        let mut depth = 1i32;
        while depth > 0 {
            let kind = &self.peek_at(offset).kind;
            match kind {
                TokenKind::LeftParen | TokenKind::LeftBracket | TokenKind::LeftBrace => {
                    depth += 1;
                }
                TokenKind::RightParen | TokenKind::RightBracket | TokenKind::RightBrace => {
                    depth -= 1;
                    if depth == 0 {
                        return false;
                    }
                }
                TokenKind::Comma if depth == 1 => return true,
                TokenKind::Eof => return false,
                _ => {}
            }
            offset += 1;
        }
        false
    }

    /// Look ahead to check if the current `<` token starts a valid generic type
    /// argument list (type names separated by commas, ending with `>`).
    pub(in crate::parser::rd) fn is_generic_type_args(&self) -> bool {
        let mut offset = 0;
        if self.peek_at(offset).kind != TokenKind::Less {
            return false;
        }
        offset += 1;
        let mut depth: i32 = 1;
        let mut seen_type_token = false;
        while depth > 0 {
            let kind = &self.peek_at(offset).kind;
            if matches!(kind, TokenKind::Identifier(_) | TokenKind::Keyword(_)
                | TokenKind::Dot | TokenKind::Comma
                | TokenKind::LeftBracket | TokenKind::RightBracket
                | TokenKind::LeftParen | TokenKind::RightParen)
            {
                seen_type_token = true;
            }
            if matches!(kind, TokenKind::Question | TokenKind::QuestionDot
                | TokenKind::QuestionQuestion)
                && !seen_type_token
            {
                return false;
            }
            // `>>` and `>>>` are lexed as Shr / ShrShr. In a generic
            // argument list these are always nested closers (there is no
            // shift-right inside a type argument), so treat them as 2 and
            // 3 consecutive `>` tokens respectively.
            match kind {
                TokenKind::Less => depth += 1,
                TokenKind::Shr => {
                    if depth < 2 {
                        return false;
                    }
                    depth -= 2;
                }
                TokenKind::ShrShr => {
                    if depth < 3 {
                        return false;
                    }
                    depth -= 3;
                }
                TokenKind::Greater => depth -= 1,
                TokenKind::Eof => return false,
                _ => {
                    if !matches!(kind, TokenKind::Identifier(_) | TokenKind::Keyword(_)
                        | TokenKind::Dot | TokenKind::Comma
                        | TokenKind::LeftBracket | TokenKind::RightBracket
                        | TokenKind::LeftParen | TokenKind::RightParen
                        | TokenKind::Question | TokenKind::QuestionDot
                        | TokenKind::QuestionQuestion)
                    {
                        return false;
                    }
                }
            }
            if depth == 0 {
                offset += 1;
                return matches!(
                    self.peek_at(offset).kind,
                    TokenKind::LeftParen | TokenKind::Dot
                );
            }
            offset += 1;
        }
        depth == 0
    }

    /// Offset (from the cursor) just past the balanced `>` closing the
    /// `<...>` list at the cursor, or `None` when there is no balanced
    /// list. `>>` / `>>>` count as nested closers, mirroring
    /// `is_generic_type_args` — but unlike it, this imposes NOTHING on
    /// what follows the list, so method groups (`&M<T>;`) validate while
    /// comparisons (`&x < y;`, no closer) do not. Bounded, always
    /// terminates.
    pub(in crate::parser::rd) fn generic_close_at(&self) -> Option<usize> {
        if self.peek_at(0).kind != TokenKind::Less {
            return None;
        }
        if !matches!(
            self.peek_at(1).kind,
            TokenKind::Identifier(_) | TokenKind::Keyword(_) | TokenKind::LeftParen
        ) {
            return None;
        }
        let mut offset = 1usize;
        let mut depth = 1i32;
        loop {
            if offset > 10000 {
                return None;
            }
            match &self.peek_at(offset).kind {
                TokenKind::Less => depth += 1,
                TokenKind::Greater => depth -= 1,
                TokenKind::Shr => {
                    if depth < 2 {
                        return None;
                    }
                    depth -= 2;
                }
                TokenKind::ShrShr => {
                    if depth < 3 {
                        return None;
                    }
                    depth -= 3;
                }
                TokenKind::Eof => return None,
                TokenKind::Identifier(_)
                | TokenKind::Keyword(_)
                | TokenKind::Dot
                | TokenKind::Comma
                | TokenKind::LeftBracket
                | TokenKind::RightBracket
                | TokenKind::LeftParen
                | TokenKind::RightParen
                | TokenKind::Question
                | TokenKind::QuestionDot
                | TokenKind::QuestionQuestion => {}
                _ => return None,
            }
            offset += 1;
            if depth == 0 {
                return Some(offset);
            }
        }
    }

    /// Balanced `<...>` NOT followed by `(` or `.` — a bare generic
    /// name (method group `Array.Empty<object>`, type in `is`/`as`) that
    /// can only be generic: an unclosed `<` comparison is syntactically
    /// invalid before `)`, `]`, `}`, `,`, or `:`. (Deliberately excludes
    /// `;` and operators, where `a < b;`-style comparisons stay valid.)
    /// Bounded via `generic_close_at`; always terminates.
    pub(in crate::parser::rd) fn is_bare_generic_args(&self) -> bool {
        let Some(end) = self.generic_close_at() else {
            return false;
        };
        matches!(
            self.peek_at(end).kind,
            TokenKind::RightParen
                | TokenKind::RightBracket
                | TokenKind::RightBrace
                | TokenKind::Comma
                | TokenKind::Colon
                // Statement end: a generic method group in value
                // position (`NoDependencies =
                // Enumerable.Empty<T>;`). A comparison can't end a
                // statement here without a closer first.
                | TokenKind::Semicolon
        )
    }

    /// Dotted method-group operand after `&` (`M`, `M<T>`, `A.B<T>`):
    /// bare segment names joined as a Member chain, one balanced
    /// `<...>` list per segment allowed. Returns `None` (caller
    /// rewinds) unless at least one list was present — plain `&x`
    /// keeps the legacy path.
    pub(in crate::parser::rd) fn method_group_operand(&mut self) -> Option<Expr> {
        let TokenKind::Identifier(first) = self.peek().kind.clone() else {
            return None;
        };
        let start = self.peek().span;
        self.bump();
        let mut expr = Expr::Ident {
            name: first,
            span: start,
        };
        let mut saw_generics = false;
        loop {
            if self.peek().kind == TokenKind::Less && self.generic_close_at().is_some() {
                saw_generics = true;
                self.skip_generic_parameter_list();
                continue;
            }
            if self.peek().kind == TokenKind::Dot
                && matches!(self.peek_at(1).kind, TokenKind::Identifier(_))
            {
                self.bump(); // '.'
                let TokenKind::Identifier(seg) = self.peek().kind.clone() else {
                    return None;
                };
                let end = self.peek().span;
                self.bump();
                expr = Expr::Member {
                    object: Box::new(expr),
                    name: seg,
                    span: start.to(end),
                };
                continue;
            }
            break;
        }
        if saw_generics { Some(expr) } else { None }
    }

    /// Parse one primary C# pattern for the `is` operator's rhs. Mirrors
    /// [`super::statement::Parser::parse_pattern_primary`] but only the
    /// shape needed inside an expression (no `and`/`or` chaining here).
    pub(in crate::parser::rd) fn parse_is_pattern_primary(&mut self) -> Option<Expr> {
        let start = self.peek().span;
        // `sizeof(T)` / `typeof(T)` / `nameof(X)` in pattern position
        // (`is sizeof(char) or sizeof(byte)`): consume the name plus
        // balanced parens as one constant-ish pattern. Without this
        // the `(` falls into positional-pattern handling and the `)`
        // breaks the enclosing `or` chain.
        let is_size_name = match &self.peek().kind {
            TokenKind::Identifier(n) => n == "sizeof" || n == "typeof",
            TokenKind::Keyword(Keyword::Nameof_) => true,
            _ => false,
        };
        if is_size_name && self.peek_at(1).kind == TokenKind::LeftParen {
            self.bump(); // 'sizeof' / 'typeof' / 'nameof'
            self.skip_balanced_bracket();
            return Some(Expr::Ident {
                name: "<pattern>".to_owned(),
                span: start.to(self.peek_at(0).span),
            });
        }
        // Discard / var: `_`, `var x`, `var (x)`, `var _`.
        if let TokenKind::Identifier(name) = &self.peek().kind
            && (name == "_" || name == "var")
        {
            let name = name.clone();
            self.bump();
            // `var x` (or `var record` — contextual keywords are valid
            // designations): declare the binding in the enclosing scope
            // like type-pattern bindings do, so later uses resolve
            // (`while (A is var b && ...)` in FileProviders). The `is`
            // test itself is always true; yield a benign literal (the
            // `is` lowering ignores operands). `_` discards need no
            // declaration.
            if name == "var" {
                if let Some((binding, span)) = self.take_pattern_binding() {
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
                    return Some(Expr::Literal(Literal::Int(0)));
                }
            }
            return Some(Expr::Ident {
                name,
                span: start.to(self.peek_at(0).span),
            });
        }
        // Relational pattern: `<rel> expr`.
        let rel_op = match self.peek().kind {
            TokenKind::GreaterEq => Some(BinOp::Ge),
            TokenKind::LessEq => Some(BinOp::Le),
            TokenKind::Greater => Some(BinOp::Gt),
            TokenKind::Less => Some(BinOp::Lt),
            _ => None,
        };
        if let Some(op) = rel_op {
            self.bump();
            // Full shift/additive/multiplicative operand
            // (`> -X - 1`): precedence 6 admits shifts (6) and up,
            // excluding comparisons, booleans, and `is` itself.
            // Without an operator following, this parses exactly
            // what `unary()` did — so working code is unaffected.
            let rhs = self.binary_expr(6)?;
            let end = expr_span_last(&self.tokens[..self.pos]);
            return Some(Expr::Binary {
                op,
                lhs: Box::new(Expr::Literal(Literal::Int(0))),
                rhs: Box::new(rhs),
                span: start.to(end),
            });
        }
        // Cast in pattern position: `(byte)'l'` — parens holding a
        // single type (no top-level comma) followed by an operand are a
        // cast, not a positional pattern. Mirrors the statement-level
        // `case`-label handling; without it `is not (byte)'\t'` skips
        // `(byte)` as a pattern and chokes on the char literal.
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
                if let Some(inner) = self.unary() {
                    let end = expr_span_last(&self.tokens[..self.pos]);
                    return Some(Expr::Cast {
                        type_name,
                        expr: Box::new(inner),
                        span: start.to(end),
                    });
                }
            }
            self.pos = mark;
            self.errors.truncate(err_mark);
        }
        // Property / positional / list patterns (`{ ... } x`,
        // `(...) x`, `[...] x`). A trailing binding identifier
        // declares a value in the enclosing scope (`is { } state`) —
        // expose it like type-pattern bindings below.
        if self.peek().kind == TokenKind::LeftBrace
            || self.peek().kind == TokenKind::LeftParen
            || self.peek().kind == TokenKind::LeftBracket
        {
            let open = self.pos;
            self.skip_balanced_bracket();
            // Designations nested in the pattern (`{ P: Type x }`,
            // `(int x, string y)`) bind in the enclosing scope too —
            // without them the uses fail downstream (`E0025`).
            self.queue_inner_designations(open);
            // A trailing binding identifier declares a value in the
            // enclosing scope — it may be a contextual keyword
            // (`is (...) record`).
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
            return Some(Expr::Literal(Literal::Int(0)));
        }
        // Type pattern: `Type`, `Type var` — an identifier followed by an
        // optional binding identifier. Also handles `Type { ... }` (property
        // pattern with type prefix) and `Type { ... } var` (with binding),
        // plus array-suffixed types (`T[] x`, `T[,] x`), plus dotted
        // constant paths (`State.None`, `A.B.C` in `or` chains).
        if matches!(self.peek().kind, TokenKind::Identifier(_)) {
            let type_name = self.expect_identifier("as type")?.to_string();
            // Generic arguments: `IList<string?>`, `Dictionary<K,V>`.
            // Consumed (not recorded) to match the compact type model;
            // without this the `<` misparses as less-than.
            if matches!(self.peek().kind, TokenKind::Less | TokenKind::Shl) {
                self.skip_generic_parameter_list();
            }
            // Nullable suffix on the pattern type itself: `string? x`.
            // Must not eat a ternary `?` (`is T ? a : b`): only consume
            // when what follows cannot start a ternary branch, or is an
            // identifier that is itself not followed by `:` (binding, not
            // `? branch :`).
            if self.peek().kind == TokenKind::Question && self.question_is_nullable() {
                self.bump();
            }
            // Dotted constant path: fold `A.B.C` into a member chain so
            // `is State.None or State.Ready` parses. The value folds
            // downstream like any member access. A trailing identifier
            // (`is Ns.T x`) is the binding, typed by the full path.
            let mut path: Option<Expr> = None;
            let mut full = type_name.clone();
            while self.peek().kind == TokenKind::Dot
                && matches!(self.peek_at(1).kind, TokenKind::Identifier(_))
            {
                self.bump(); // '.'
                let seg = self.expect_identifier("as pattern path segment")?;
                full.push('.');
                full.push_str(&seg);
                // Generics per segment: `Ns.Generic<T>.Member`.
                if matches!(self.peek().kind, TokenKind::Less | TokenKind::Shl) {
                    self.skip_generic_parameter_list();
                }
                let base = path.unwrap_or(Expr::Ident {
                    name: type_name.clone(),
                    span: start,
                });
                let end = expr_span_last(&self.tokens[..self.pos]);
                path = Some(Expr::Member {
                    object: Box::new(base),
                    name: seg,
                    span: start.to(end),
                });
            }
            if let Some(p) = path {
                // Nullable suffix on a dotted path (`is Ns.T? x`).
                if self.peek().kind == TokenKind::Question && self.question_is_nullable() {
                    self.bump();
                }
                if matches!(self.peek().kind, TokenKind::Identifier(_)) || matches!(self.peek().kind, TokenKind::Keyword(_)) {
                    if let Some((binding, span)) = self.take_pattern_binding() {
                        self.pre_pending.push(Stmt::Let {
                            is_byref: false,
                            type_name: full,
                            name: binding,
                            init: Expr::Literal(Literal::Int(0)),
                            access: Access::Default,
                            span: start.to(span),
                        });
                    }
                }
                return Some(p);
            }
            // Suffixes interleave in any order (`T[]?[]`, `T?[]?`, `T*`):
            // loop until none remains. Each `?` is validated against a
            // ternary misread (`is T ? a : b`).
            loop {
                let mut progressed = false;
                if self.peek().kind == TokenKind::LeftBracket {
                    self.skip_balanced_bracket();
                    progressed = true;
                }
                if self.peek().kind == TokenKind::Question && self.question_is_nullable() {
                    self.bump();
                    progressed = true;
                }
                if self.peek().kind == TokenKind::Star {
                    self.bump();
                    progressed = true;
                }
                if !progressed {
                    break;
                }
            }
            if self.peek().kind == TokenKind::LeftBrace {
                // Property pattern with a type prefix (`T { P: U x }`):
                // inner designations bind like the trailing one below.
                let open = self.pos;
                self.skip_balanced_bracket();
                self.queue_inner_designations(open);
            }
            // A trailing binding identifier declares the pattern value in
            // the enclosing scope; expose it to the semantic scan as a
            // zero-init `Let` (drained by `block`). The type is the
            // pattern's type name so that member access on the binding
            // var resolves. The binding may be a contextual keyword
            // (`is T record`).
            if let Some((binding, span)) = self.take_pattern_binding() {
                self.pre_pending.push(Stmt::Let {
                    is_byref: false,
                    type_name: type_name.clone(),
                    name: binding,
                    init: Expr::Literal(Literal::Int(0)),
                    access: Access::Default,
                    span: start.to(span),
                });
            }
            return Some(Expr::Literal(Literal::Int(0)));
        }
        // Fallback: a constant pattern (`null`, `5`, `"x"`,
        // `SomeEnum.Value`, `-1`). Parsed at unary level — NEVER a
        // full expression: boolean/assignment/closure operators
        // (`||`, `&&`, `??`, `=>`, …) terminate the pattern in C#.
        // The old full-expression fallback swallowed trailing
        // operators (`v is null || v.Length == 0` became
        // `v is (null || …)`), silently inverting guards.
        self.unary()
    }

    /// True if the `?` at the cursor is a nullable-type suffix (not a
    /// ternary branch). Caller guarantees `peek() == Question`.
    /// `is string? x` / `is T? { ... }` consume; `is T ? a.B : c`
    /// does not. Scans ahead: a top-level single `:` means ternary;
    /// a pattern terminator first means nullable. Bracket depth is
    /// tracked so `:` inside `[...]`, `(...)`, `{...}` is ignored.
    fn question_is_nullable(&self) -> bool {
        let mut i = 1usize;
        let mut depth = 0usize;
        loop {
            match &self.peek_at(i).kind {
                TokenKind::Eof => return true,
                // Openers: `{`/`[` continue a pattern (nullable);
                // `(` is ambiguous (positional pattern vs parenthesized
                // ternary branch) — ternary is far more common, so a
                // bare `(` means ternary. But once inside a branch,
                // all openers just nest.
                TokenKind::LeftBrace | TokenKind::LeftBracket => {
                    if depth == 0 && i == 1 {
                        return true;
                    }
                    depth += 1;
                }
                TokenKind::LeftParen => {
                    if depth == 0 && i == 1 {
                        return false;
                    }
                    depth += 1;
                }
                TokenKind::RightParen | TokenKind::RightBrace | TokenKind::RightBracket => {
                    if depth == 0 {
                        return true;
                    }
                    depth = depth.saturating_sub(1);
                }
                // A second `?` right away (`T? ?? y`, `T? ?.M`) belongs
                // to the nullable type, not a ternary.
                TokenKind::Question | TokenKind::QuestionQuestion | TokenKind::QuestionDot => {
                    if depth == 0 && i == 1 {
                        return true;
                    }
                }
                // Pattern/pattern-list endings at depth 0: nullable.
                TokenKind::Comma
                | TokenKind::Semicolon
                | TokenKind::FatArrow => {
                    if depth == 0 {
                        return true;
                    }
                }
                TokenKind::Keyword(
                    Keyword::And_ | Keyword::Or_ | Keyword::When_,
                )
                | TokenKind::AmpAmp
                | TokenKind::PipePipe => {
                    if depth == 0 {
                        return true;
                    }
                }
                // Single `:` at depth 0 is the ternary separator.
                // `::` is a namespace separator — skip both.
                TokenKind::Colon => {
                    if depth == 0 {
                        if self.peek_at(i + 1).kind == TokenKind::Colon {
                            i += 2;
                            continue;
                        }
                        return false;
                    }
                }
                _ => {}
            }
            i += 1;
        }
    }

    /// Consume a balanced bracket pair (`()`, `[]`, or `{}`) including the
    /// opening and closing tokens. Used to skip pattern internals.
    pub(in crate::parser::rd) fn skip_balanced_bracket(&mut self) {
        let open = match self.peek().kind {
            TokenKind::LeftBrace => TokenKind::LeftBrace,
            TokenKind::LeftParen => TokenKind::LeftParen,
            TokenKind::LeftBracket => TokenKind::LeftBracket,
            _ => return,
        };
        let close = match open {
            TokenKind::LeftBrace => TokenKind::RightBrace,
            TokenKind::LeftParen => TokenKind::RightParen,
            TokenKind::LeftBracket => TokenKind::RightBracket,
            _ => return,
        };
        self.bump(); // opening
        let mut depth = 1usize;
        loop {
            if self.peek().kind == open {
                depth += 1;
            } else if self.peek().kind == close {
                depth -= 1;
            }
            self.bump();
            if depth == 0 || self.peek().kind == TokenKind::Eof {
                break;
            }
        }
    }
}

/// Parse a single interpolation hole (the text between its outermost
/// braces). A trailing `,width` or `:format` specifier is dropped (a known
/// simplification). Nested interpolation inside the hole is handled by the
/// normal expression parser, since the hole body is parsed as a full
/// expression.
pub(in crate::parser::rd) fn parse_interp_hole(body: &str) -> Option<Expr> {
    let s = body.trim();
    if s.is_empty() {
        return None;
    }
    let expr_src = split_interp_spec(s).0;
    let lexed = crate::parser::lexer::Lexer::new(expr_src).tokenize();
    let mut p = crate::parser::rd::Parser::new(lexed);
    let expr = p.expression()?;
    Some(expr)
}

/// Split an interpolation hole body into `(expression, specifier)`. The
/// specifier is the text after the first top-level `,` or `:` (not inside
/// braces/parens). Returns the trimmed expression source and the specifier.
pub(in crate::parser::rd) fn split_interp_spec(s: &str) -> (&str, &str) {
    let mut depth = 0i32;
    let mut spec_start = None;
    for (idx, c) in s.char_indices() {
        match c {
            '{' | '(' => depth += 1,
            '}' | ')' => depth -= 1,
            ',' | ':' if depth == 0 => {
                spec_start = Some(idx);
                break;
            }
            _ => {}
        }
    }
    match spec_start {
        Some(idx) => {
            let spec = s[idx..].trim_start_matches(|c| c == ',' || c == ':').trim();
            (s[..idx].trim(), spec)
        }
        None => (s, ""),
    }
}

/// Read a `\uXXXX` code point (1-4 hex digits) at `bytes[from]`. Returns
/// `(char, hex_digits_consumed)` or `None` if no hex digits follow.
pub(in crate::parser::rd) fn read_unicode_escape(bytes: &[u8], from: usize) -> Option<(char, usize)> {
    let mut value: u32 = 0;
    let mut k = 0usize;
    while k < 4 && from + k < bytes.len() {
        let d = match bytes[from + k] {
            b'0'..=b'9' => bytes[from + k] - b'0',
            b'a'..=b'f' => bytes[from + k] - b'a' + 10,
            b'A'..=b'F' => bytes[from + k] - b'A' + 10,
            _ => break,
        };
        value = value * 16 + d as u32;
        k += 1;
    }
    if k == 0 {
        return None;
    }
    Some((char::from_u32(value)?, k))
}
