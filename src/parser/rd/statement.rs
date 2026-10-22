//! Statements and blocks.

use crate::models::{
    Access, BinOp, Block, CatchClause, Diagnostic, Expr, Keyword, Literal, Severity, Span, Stmt,
    SwitchArm, TokenKind, UnOp,
};

use super::{Parser, describe, expr_span, expr_span_last};

impl Parser {
    pub(super) fn block(&mut self) -> Block {
        // Bindings queued by the ENCLOSING statement (e.g. `out` vars or
        // `is`-pattern bindings from an `if`/`while` condition) belong to
        // the outer block under C# scoping — not to this nested body.
        // Stash them so statements parsed here cannot steal them; restore
        // on exit so the outer `block()` drains them after the statement.
        let stashed_pending = std::mem::take(&mut self.pending);
        let stashed_pre = std::mem::take(&mut self.pre_pending);
        let open = self.expect(&TokenKind::LeftBrace, "to open method body");
        let start = open.map_or(self.peek().span, |t| t.span);
        let mut stmts = Vec::new();
        let mut safety = 0u32;
        while self.peek().kind != TokenKind::RightBrace && self.peek().kind != TokenKind::Eof {
            safety += 1;
            if safety > 100000 {
                break;
            }
            let before = self.pos;
            let stmt_start = stmts.len();
            match self.statement() {
                Some(stmt) => {
                    // Bindings declared during this statement's parsing (e.g.
                    // an `is`-pattern binding var) must appear BEFORE the
                    // statement that uses them, so prepend them.
                    if !self.pre_pending.is_empty() {
                        let pre = std::mem::take(&mut self.pre_pending);
                        stmts.splice(
                            stmt_start..stmt_start,
                            pre,
                        );
                    }
                    stmts.push(stmt);
                    // Trailing declarators from a multi-declarator statement
                    // are queued as their own `Let`s; emit them in order.
                    stmts.append(&mut self.pending);
                }
                None => {
                    if !self.pre_pending.is_empty() {
                        stmts.append(&mut self.pre_pending);
                    }
                    if self.pos == before {
                        self.bump();
                    } else {
                        self.synchronize();
                    }
                }
            }
        }
        let close = self.expect(&TokenKind::RightBrace, "to close method body");
        // Restore the enclosing statement's queued bindings ahead of any
        // leftover from a failed trailing statement (order is irrelevant
        // to the scan, which pre-registers block Lets as a set).
        let mut rest_pending = std::mem::take(&mut self.pending);
        self.pending = stashed_pending;
        self.pending.append(&mut rest_pending);
        let mut rest_pre = std::mem::take(&mut self.pre_pending);
        self.pre_pending = stashed_pre;
        self.pre_pending.append(&mut rest_pre);
        let span = start.to(close.map_or(self.peek().span, |t| t.span));
        let stmts = wrap_using_decls(stmts, span);
        Block { stmts, span }
    }

    /// Weak tuple-declaration shape: `(...)` validated by
    /// `member_tuple_at`, plus a declarator visibly following
    /// (`name =` or `name;`). An argument list can never be followed
    /// by those, so weak-unnamed tuples (`(A, B)[] x = ...`) are
    /// unambiguous here. Pure lookahead, no consumption.
    fn weak_tuple_decl_at(&self) -> bool {
        self.peek().kind == TokenKind::LeftParen
            && self.member_tuple_at(0).is_some_and(|past| {
                self.peek_is_name(past)
                    && matches!(
                        self.peek_at(past + 1).kind,
                        TokenKind::Assign | TokenKind::Semicolon
                    )
            })
    }

    /// True if a tuple declaration (or destructuring) starts here:
    /// optional `ref`/`scoped`/`readonly` prefix, then `(...)` in
    /// strict (named-pair) or weak (follower-validated) shape.
    /// Bump-and-rewind: never consumes, drops speculative errors.
    fn tuple_decl_here(&mut self) -> bool {
        let mark = self.pos;
        let err_mark = self.errors.len();
        while matches!(
            self.peek().kind,
            TokenKind::Keyword(Keyword::Ref_)
                | TokenKind::Keyword(Keyword::Scoped)
                | TokenKind::Keyword(Keyword::Readonly)
        ) {
            self.bump();
        }
        let ok = self.peek().kind == TokenKind::LeftParen
            && (self.tuple_decl_shape_at() || self.weak_tuple_decl_at());
        self.pos = mark;
        self.errors.truncate(err_mark);
        ok
    }

    /// Gate for `var (bindings) = init;`: `var`, then a balanced
    /// paren group, then `=`. Bounded lookahead, no consumption.
    /// (`var` alone is a legal identifier/call — `var = 5;`,
    /// `var(x)` — so the parens-plus-`=` shape must match.)
    fn var_tuple_decl_here(&mut self) -> bool {
        if !matches!(&self.peek().kind, TokenKind::Identifier(n) if n == "var") {
            return false;
        }
        if self.peek_at(1).kind != TokenKind::LeftParen {
            return false;
        }
        let mut depth = 0i32;
        let mut i = 1usize;
        loop {
            if i > 500 {
                return false;
            }
            match self.peek_at(i).kind {
                TokenKind::LeftParen => depth += 1,
                TokenKind::RightParen => {
                    depth -= 1;
                    if depth == 0 {
                        return self.peek_at(i + 1).kind == TokenKind::Assign;
                    }
                }
                TokenKind::Eof | TokenKind::Semicolon | TokenKind::LeftBrace | TokenKind::RightBrace => {
                    return false;
                }
                _ => {}
            }
            i += 1;
        }
    }

    /// Collect binding names from a `var (...)` pattern (cursor just
    /// past the opening paren): bare names (`a`), discards (`_`),
    /// typed elements (`T name`, type skipped), nested groups
    /// (recursed). Stops at the closing paren or EOF.
    fn collect_tuple_bindings(&mut self, out: &mut Vec<String>) {
        loop {
            match self.peek().kind {
                TokenKind::RightParen | TokenKind::Eof => break,
                TokenKind::Comma => {
                    self.bump();
                }
                TokenKind::LeftParen => {
                    self.bump(); // '('
                    self.collect_tuple_bindings(out);
                    if self.peek().kind == TokenKind::RightParen {
                        self.bump();
                    }
                }
                TokenKind::Identifier(_) => {
                    // Bare name (`a,`) when followed by `,`/`)`;
                    // otherwise a type (`T name`) — skip it, take name.
                    if matches!(
                        self.peek_at(1).kind,
                        TokenKind::Comma | TokenKind::RightParen
                    ) {
                        if let TokenKind::Identifier(name) = self.peek().kind.clone() {
                            out.push(name);
                        }
                        self.bump();
                    } else {
                        let _ = self.type_name();
                        if let Some(name) = self.expect_identifier_or_keyword(
                            "rubyc.parser.rd.Parser.ctx.as_variable_name",
                        ) {
                            out.push(name);
                        }
                    }
                }
                _ => {
                    // Anything else (literals, stray keywords): consume
                    // one token so collection always terminates.
                    self.bump();
                }
            }
        }
    }

    /// Tuple-declaration shape `(T1 n1, T2 n2, ...)`: each top-level
    /// element must be a type (`Ident(.Ident)*` + generics + suffixes,
    /// builtin keyword types included), and at least one element must
    /// carry a trailing name. Returns false for argument lists
    /// (`(a, b)`), groupings (`(x)`), and anything else. Bounded
    /// lookahead, no consumption.
    fn tuple_decl_shape_at(&self) -> bool {
        if !matches!(self.peek_at(0).kind, TokenKind::LeftParen) {
            return false;
        }
        let mut i = 1usize;
        let mut named = 0u32;
        let mut parts = 0u32;
        let mut guard = 0u32;
        // A parenthesized single element (`((T)x).M(...)` — a grouped
        // cast, not a tuple) must not satisfy the shape alone: C# has
        // no 1-element tuple types.
        let mut saw_paren_element = false;
        loop {
            if guard > 200 {
                return false;
            }
            guard += 1;
            // Element type: Identifier or keyword builtin (`int`,
            // `string`, `var`, ...) plus dotted segments — or a nested
            // tuple type (`(char Char, ...) Literal` element). An
            // INDEXER (`a[i]`) is a value, never an element type: a
            // trailing `.M` after one (`(t.C[o].M)`) is a grouped
            // member chain, not a tuple declaration. The balanced
            // scan below only rejects when the bracket is NOT also
            // closable as an EMPTY rank specifier paired with the
            // element scan (`skip_tuple_element_type_at`, which stops
            // at non-empty brackets); `t.C[]` stays a valid element.
            if self.peek_at(i).kind == TokenKind::LeftParen {
                saw_paren_element = true;
            }
            // Balanced-bracket scan for a `[` holding an expression:
            // `t.C[o]` — indexers end the element scan here (a type
            // rank specifier is always EMPTY: `[]`, `[,]`).
            {
                let mut k = i;
                let mut depth = 0i32;
                let mut bracket_expr = false;
                while k < i + 60 {
                    match self.peek_at(k).kind {
                        TokenKind::Eof => break,
                        TokenKind::LeftParen => depth += 1,
                        TokenKind::RightParen => {
                            depth -= 1;
                            if depth < 0 {
                                break;
                            }
                        }
                        TokenKind::LeftBracket => {
                            // Empty rank `[]`/`[,]`: keep scanning.
                            let mut m = k + 1;
                            let mut empty = true;
                            let mut d2 = 0usize;
                            loop {
                                match self.peek_at(m).kind {
                                    TokenKind::Eof => break,
                                    TokenKind::LeftBracket => {
                                        d2 += 1;
                                        empty = false;
                                    }
                                    TokenKind::RightBracket => {
                                        if d2 == 0 {
                                            m += 1;
                                            break;
                                        }
                                        d2 = d2.saturating_sub(1);
                                    }
                                    TokenKind::Comma if d2 == 0 => {}
                                    _ => {
                                        empty = false;
                                    }
                                }
                                m += 1;
                            }
                            // Only a NON-empty bracket is an indexer
                            // value. An empty rank (`t.C[]`) belongs to
                            // the type: skip past it and keep scanning
                            // (without this, `(byte[])x`-style element
                            // shapes misfire as indexers).
                            if !empty {
                                bracket_expr = true;
                                break;
                            }
                            k = m.wrapping_sub(1);
                        }
                        TokenKind::Comma if depth == 0 => break,
                        _ => {}
                    }
                    k += 1;
                }
                if bracket_expr {
                    return false;
                }
            }
            let Some(next) = self.skip_tuple_element_type_at(i) else {
                return false;
            };
            i = next;
            if self.peek_is_name(i)
                && matches!(
                    self.peek_at(i + 1).kind,
                    TokenKind::Comma | TokenKind::RightParen
                )
            {
                named += 1;
                i += 1;
            }
            parts += 1;
            if matches!(self.peek_at(i).kind, TokenKind::Comma) {
                i += 1;
                // Trailing comma is not a tuple type.
                if matches!(self.peek_at(i).kind, TokenKind::RightParen) {
                    return false;
                }
                continue;
            } else if matches!(self.peek_at(i).kind, TokenKind::RightParen) {
                break;
            } else {
                return false;
            }
        }
        parts >= 1 && named >= 1 && (parts >= 2 || !saw_paren_element)
    }

    /// Detect a local function declaration: `Type Name(params) { body }`.
    /// The name is followed directly by `(` (optionally preceded by a
    /// generic parameter list `<...>`) with no `=` or `;` in between.
    /// An optional `static` modifier may precede the return type.
    fn looks_like_local_function(&self) -> bool {
        let mut offset = 0;
        // Optional `static` / `async` / `unsafe` modifiers, plus `ref` /
        // `readonly` / `scoped` return modifiers (`ref object? Local()`).
        // A `ref` local declaration (`ref int x = ...`) still fails the
        // gate below (no `(` after the name), so this is safe.
        while matches!(
            self.peek_at(offset).kind,
            TokenKind::Keyword(Keyword::Static)
                | TokenKind::Keyword(Keyword::Async)
                | TokenKind::Keyword(Keyword::Unsafe)
                | TokenKind::Keyword(Keyword::Ref_)
                | TokenKind::Keyword(Keyword::Readonly)
                | TokenKind::Keyword(Keyword::Scoped)
                | TokenKind::Keyword(Keyword::Extern)
        ) {
            offset += 1;
        }
        // Tuple return type: `(T1 n1, T2 n2) Name(` or `(T1, T2) Name(`.
        // Scan balanced parens (nesting-aware) plus `[]`/`?`/`*`
        // suffixes, then require `Name(`.
        if self.peek_at(offset).kind == TokenKind::LeftParen {
            let mut pdepth = 0i32;
            let mut i = offset;
            loop {
                match self.peek_at(i).kind {
                    TokenKind::Eof => return false,
                    TokenKind::LeftParen => pdepth += 1,
                    TokenKind::RightParen => {
                        pdepth -= 1;
                        if pdepth == 0 {
                            i += 1;
                            break;
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            i = self.skip_suffixes_at(i);
            if self.peek_is_name(i)
                && self.peek_at(i + 1).kind == TokenKind::LeftParen
            {
                return true;
            }
            return false;
        }
        if !matches!(self.peek_at(offset).kind, TokenKind::Identifier(_)) {
            return false;
        }
        offset += 1;
        // Dotted segments on the type, each of which may carry a generic list.
        while self.peek_at(offset).kind == TokenKind::Dot {
            offset += 1;
            if !matches!(self.peek_at(offset).kind, TokenKind::Identifier(_)) {
                return false;
            }
            offset += 1;
            if matches!(
                self.peek_at(offset).kind,
                TokenKind::Less | TokenKind::Shl
            ) {
                let mut depth = 0i32;
                while self.peek_at(offset).kind != TokenKind::Eof {
                    match self.peek_at(offset).kind {
                        TokenKind::Less => depth += 1,
                        TokenKind::Shl => depth += 2,
                        TokenKind::Greater => {
                            depth -= 1;
                            if depth == 0 {
                                offset += 1;
                                break;
                            }
                        }
                        TokenKind::Shr => {
                            depth -= 2;
                            if depth == 0 {
                                offset += 1;
                                break;
                            }
                        }
                        TokenKind::ShrShr => {
                            depth -= 3;
                            if depth == 0 {
                                offset += 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                    offset += 1;
                }
            }
        }
        // Generic arguments on a plain (non-dotted) return type:
        // `ValueTask<int> Core(...)`. `>>`/`>>>` lex as Shr/ShrShr.
        if matches!(
            self.peek_at(offset).kind,
            TokenKind::Less | TokenKind::Shl
        ) {
            let mut depth = 0i32;
            while self.peek_at(offset).kind != TokenKind::Eof {
                match self.peek_at(offset).kind {
                    TokenKind::Less => depth += 1,
                    TokenKind::Shl => depth += 2,
                    TokenKind::Greater => {
                        depth -= 1;
                        if depth == 0 {
                            offset += 1;
                            break;
                        }
                    }
                    TokenKind::Shr => {
                        depth -= 2;
                        if depth == 0 {
                            offset += 1;
                            break;
                        }
                    }
                    TokenKind::ShrShr => {
                        depth -= 3;
                        if depth == 0 {
                            offset += 1;
                            break;
                        }
                    }
                    _ => {}
                }
                offset += 1;
            }
        }
        // Suffixes on the return type: `HeaderEntry[] Local()`,
        // `string? Local()`, `int* Local()`. Loop until none remains.
        loop {
            match self.peek_at(offset).kind {
                TokenKind::Question | TokenKind::Star => offset += 1,
                TokenKind::LeftBracket => {
                    // `[]` or `[,]` rank specifiers.
                    let mut depth = 0usize;
                    loop {
                        match self.peek_at(offset).kind {
                            TokenKind::Eof => break,
                            TokenKind::LeftBracket => depth += 1,
                            TokenKind::RightBracket => {
                                depth = depth.saturating_sub(1);
                                if depth == 0 {
                                    offset += 1;
                                    break;
                                }
                            }
                            _ => {}
                        }
                        offset += 1;
                    }
                }
                _ => break,
            }
        }
        // The next token must be `(` for a function definition.
        if !matches!(self.peek_at(offset).kind, TokenKind::Identifier(_)) {
            return false;
        }
        offset += 1;
        if self.peek_at(offset).kind == TokenKind::Less {
            let mut depth = 0i32;
            while self.peek_at(offset).kind != TokenKind::Eof {
                match self.peek_at(offset).kind {
                    TokenKind::Less => depth += 1,
                    TokenKind::Greater => {
                        depth -= 1;
                        if depth == 0 {
                            offset += 1;
                            break;
                        }
                    }
                    _ => {}
                }
                offset += 1;
            }
        }
        // The next token must be `(` for a function definition.
        self.peek_at(offset).kind == TokenKind::LeftParen
    }

    /// Parse a local function declaration, preserving it as
    /// `Stmt::LocalFunction` when static, non-generic, and bodied
    /// (lowered to a real function); anything else keeps the legacy
    /// zero placeholder (`Let fn = 0` + `Break`).
    fn parse_local_function(
        &mut self,
        start: Span,
        attrs: Vec<crate::models::ast::attribute::Attribute>,
    ) -> Option<Stmt> {
        // Optional `static` / `async` / `unsafe` modifiers, plus `ref` /
        // `readonly` / `scoped` return modifiers (by-ref-ness has no
        // runtime representation here), plus `extern` (extern locals
        // are bodyless `[UnsafeAccessor]` declarations).
        let mut is_static = false;
        let mut is_async = false;
        let mut is_extern = false;
        while matches!(
            self.peek().kind,
            TokenKind::Keyword(Keyword::Static)
                | TokenKind::Keyword(Keyword::Async)
                | TokenKind::Keyword(Keyword::Unsafe)
                | TokenKind::Keyword(Keyword::Ref_)
                | TokenKind::Keyword(Keyword::Readonly)
                | TokenKind::Keyword(Keyword::Scoped)
                | TokenKind::Keyword(Keyword::Extern)
        ) {
            if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Static)) {
                is_static = true;
            }
            if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Async)) {
                is_async = true;
            }
            if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Extern)) {
                is_extern = true;
            }
            self.bump();
        }
        // Tuple return type: `(T1 n1, T2 n2) Name(` or `(T1, T2) Name(`.
        // Consumed as a unit; the element names are not bindings
        // (unlike tuple *declarations*, which never appear here).
        // Tuple-returning locals keep the legacy placeholder.
        let mut tuple_ret = false;
        let mut ret_type: Option<String> = None;
        if self.peek().kind == TokenKind::LeftParen {
            tuple_ret = true;
            let mut depth = 0i32;
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
            loop {
                match self.peek().kind {
                    TokenKind::Question | TokenKind::Star => {
                        self.bump();
                    }
                    TokenKind::LeftBracket => {
                        let mut bdepth = 0usize;
                        while self.peek().kind != TokenKind::Eof {
                            match self.peek().kind {
                                TokenKind::LeftBracket => bdepth += 1,
                                TokenKind::RightBracket => {
                                    bdepth = bdepth.saturating_sub(1);
                                    self.bump();
                                    if bdepth == 0 {
                                        break;
                                    }
                                    continue;
                                }
                                _ => {}
                            }
                            self.bump();
                        }
                    }
                    _ => break,
                }
            }
        } else {
            ret_type = self.type_name();
        }
        let lf_name = self.expect_identifier("as function name")?;
        // Generic parameter list (`>>`/`>>>` lex as Shr/ShrShr):
        // generic locals keep the legacy placeholder.
        let mut is_generic = false;
        if matches!(self.peek().kind, TokenKind::Less | TokenKind::Shl) {
            is_generic = true;
            let mut depth = 0i32;
            while self.peek().kind != TokenKind::Eof {
                match self.peek().kind {
                    TokenKind::Less => depth += 1,
                    TokenKind::Shl => depth += 2,
                    TokenKind::Greater => {
                        depth -= 1;
                        if depth == 0 {
                            self.bump();
                            break;
                        }
                    }
                    TokenKind::Shr => {
                        depth -= 2;
                        if depth == 0 {
                            self.bump();
                            break;
                        }
                    }
                    TokenKind::ShrShr => {
                        depth -= 3;
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
        // Parameter list, parsed for real (malformed lists fall
        // back to the legacy skip-and-placeholder below).
        self.expect(&TokenKind::LeftParen, "to open local function params")?;
        let params = self.param_list();
        let params_ok = matches!(self.peek().kind, TokenKind::RightParen)
            || matches!(self.peek().kind, TokenKind::Eof);
        if matches!(self.peek().kind, TokenKind::RightParen) {
            self.bump();
        } else {
            let mut depth = 0i32;
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
        }
        // Type constraints: `where T : ...` (consumed, not recorded —
        // same as all other declarations).
        self.skip_type_constraints();
        // Body: `=> expr;`, a block `{ ... }`, or a bare `;` for
        // bodyless `extern` declarations (`[UnsafeAccessor]` locals).
        // Only block and expression bodies become real functions.
        enum LocalBody {
            Block(Block),
            Missing,
        }
        let mut body = LocalBody::Missing;
        if self.peek().kind == TokenKind::FatArrow {
            self.bump();
            if let Some(e) = self.expression() {
                let espan = start.to(self.peek_at(0).span);
                body = LocalBody::Block(Block {
                    stmts: vec![Stmt::Return {
                        value: Some(e),
                        span: espan,
                    }],
                    span: espan,
                });
            }
            let _ = self.expect(&TokenKind::Semicolon, "after expression-bodied local function");
        } else if self.peek().kind == TokenKind::Semicolon {
            self.bump(); // bodyless extern: no body to consume
        } else {
            body = LocalBody::Block(self.block());
        }
        let end = start.to(self.peek_at(0).span);
        // The local function name is a callable value in its ENCLOSING scope.
        // Keep exposing it to the semantic scan as a zero-initialized
        // `Let` (harmless when real lowering succeeds; the fallback
        // value when it does not). Push it AFTER the body so the
        // enclosing `block()` (not the body's own block) drains it.
        self.pending.push(Stmt::Let {
            is_const: false,
            is_byref: false,
            type_name: String::from("fn"),
            name: lf_name.clone(),
            init: Expr::Literal(Literal::Int(0)),
            access: Access::Default,
            span: end,
        });
        // Real node only for the supported shape; everything else
        // keeps the legacy no-op exactly as before.
        let supported = is_static
            && !is_async
            && !tuple_ret
            && !is_generic
            && params_ok
            && ret_type.is_some()
            && !params.iter().any(|p| p.is_this || p.is_params);
        match (supported, body) {
            (true, LocalBody::Block(b)) => Some(Stmt::LocalFunction {
                name: lf_name,
                params,
                ret_type: ret_type.unwrap_or_else(|| "void".to_owned()),
                body: b,
                is_static: true,
                is_extern: false,
                attrs,
                span: end,
            }),
            // A bodiless `extern` local function (an `[UnsafeAccessor]`
            // declaration) is a real declaration: the attribute names
            // the target method. It keeps the legacy placeholder name
            // binding above, and the body is empty.
            (true, LocalBody::Missing) if is_extern => Some(Stmt::LocalFunction {
                name: lf_name,
                params,
                ret_type: ret_type.unwrap_or_else(|| "void".to_owned()),
                body: Block {
                    stmts: Vec::new(),
                    span: end,
                },
                is_static: true,
                is_extern: true,
                attrs,
                span: end,
            }),
            _ => Some(Stmt::Break { span: end }),
        }
    }

    /// Detect a typed lambda: `Type (params) => body;` — an identifier
    /// (the return type), optional suffixes, then a parenthesized list,
    /// then `=>`. Also handles an attribute prefix:
    /// `[Attr] (params) => body;` and `[Attr] Type (params) => body;`.
    fn looks_like_typed_lambda(&self) -> bool {
        let mut offset = 0;
        while self.peek_at(offset).kind == TokenKind::LeftBracket {
            offset += 1;
            let mut depth = 1i32;
            while depth > 0 && self.peek_at(offset).kind != TokenKind::Eof {
                match self.peek_at(offset).kind {
                    TokenKind::LeftBracket | TokenKind::LeftParen | TokenKind::LeftBrace => {
                        depth += 1;
                    }
                    TokenKind::RightBracket | TokenKind::RightParen | TokenKind::RightBrace => {
                        depth -= 1;
                    }
                    _ => {}
                }
                if depth == 0 {
                    offset += 1;
                    break;
                }
                offset += 1;
            }
        }
        // The next thing is either an optional return type (`Type`) or
        // directly the parameter list `(`.
        if matches!(self.peek_at(offset).kind, TokenKind::Identifier(_)) {
            offset += 1;
            if self.peek_at(offset).kind == TokenKind::Question {
                offset += 1;
            }
            while self.peek_at(offset).kind == TokenKind::LeftBracket
                && self.peek_at(offset + 1).kind == TokenKind::RightBracket
            {
                offset += 2;
            }
            while self.peek_at(offset).kind == TokenKind::Dot {
                offset += 1;
                if !matches!(self.peek_at(offset).kind, TokenKind::Identifier(_)) {
                    return false;
                }
                offset += 1;
            }
        }
        if self.peek_at(offset).kind != TokenKind::LeftParen {
            return false;
        }
        let mut depth = 1i32;
        offset += 1;
        while depth > 0 && self.peek_at(offset).kind != TokenKind::Eof {
            match self.peek_at(offset).kind {
                TokenKind::LeftParen => depth += 1,
                TokenKind::RightParen => depth -= 1,
                _ => {}
            }
            if depth == 0 {
                return self.peek_at(offset + 1).kind == TokenKind::FatArrow;
            }
            offset += 1;
        }
        false
    }

    /// Detect a lambda expression starting at the current position:
    /// either a single parameter `name =>` or a parenthesized
    /// parameter list `(params) =>`, optionally preceded by the
    /// `static` modifier (`static name =>` or `static (params) =>`).
    /// Also detects an explicit-return-type lambda: `Type (params) =>`.
    fn looks_like_lambda(&self) -> bool {
        let mut offset = 0;
        if matches!(self.peek_at(offset).kind, TokenKind::Keyword(Keyword::Static)) {
            offset += 1;
        }
        if self.peek_at(offset).kind == TokenKind::LeftParen {
            // `(params) => body` or `Type (params) => body`. We need to
            // distinguish from a parenthesized expression. Look for the
            // matching `)` followed by `=>`.
            let mut depth = 1i32;
            let mut scan = offset + 1;
            while depth > 0 && self.peek_at(scan).kind != TokenKind::Eof {
                match self.peek_at(scan).kind {
                    TokenKind::LeftParen => depth += 1,
                    TokenKind::RightParen => depth -= 1,
                    _ => {}
                }
                if depth == 0 {
                    return self.peek_at(scan + 1).kind == TokenKind::FatArrow;
                }
                scan += 1;
            }
            return false;
        }
        matches!(self.peek_at(offset).kind, TokenKind::Identifier(_))
            && self.peek_at(offset + 1).kind == TokenKind::FatArrow
    }

    /// Consume a lambda body: the `=>` token, then either a block
    /// `{ ... }` or a single expression, then the trailing `;`.
    /// Returns the start span of the body.
    fn parse_lambda_body(&mut self) -> Option<Span> {
        if self.eat(&TokenKind::FatArrow).is_none() {
            return None;
        }
        let body_start = self.peek_at(0).span;
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
            let _ = self.expression();
        }
        let _ = self.expect(&TokenKind::Semicolon, "after lambda body");
        Some(body_start)
    }

    /// Parse a single-parameter or parenthesized lambda statement. The
    /// body is consumed up to the matching `;` or closing brace.
    fn parse_lambda_statement(&mut self, start: Span) -> Option<Stmt> {
        // A lambda is an EXPRESSION, and C# allows only a narrow set of
        // expressions as statements (call, assignment, increment,
        // `await`, `new`). A bare lambda statement is CS0201. The shape is
        // still consumed so the rest of the block parses, and the rule is
        // reported rather than the lambda being silently dropped.
        let mut params: Vec<String> = Vec::new();
        if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Static)) {
            self.bump();
        }
        if self.peek().kind == TokenKind::LeftParen {
            let mut depth = 1i32;
            let mut current: Vec<String> = Vec::new();
            self.bump();
            while depth > 0 && self.peek().kind != TokenKind::Eof {
                match self.peek().kind.clone() {
                    TokenKind::LeftParen => {
                        depth += 1;
                        current.clear();
                        self.bump();
                    }
                    TokenKind::RightParen => {
                        depth -= 1;
                        if let Some(last) = current.last() {
                            params.push(last.clone());
                        }
                        current.clear();
                        self.bump();
                    }
                    TokenKind::Comma if depth == 1 => {
                        if let Some(last) = current.last() {
                            params.push(last.clone());
                        }
                        current.clear();
                        self.bump();
                    }
                    TokenKind::Identifier(n) => {
                        current.push(n);
                        self.bump();
                    }
                    _ => {
                        self.bump();
                    }
                }
            }
        } else {
            let span = self.peek().span;
            let name = String::from_utf8_lossy(
                &self.source[span.start as usize..span.end as usize],
            )
            .into_owned();
            params.push(name);
            self.bump();
        }
        if self.eat(&TokenKind::FatArrow).is_none() {
            self.unexpected(
                "rubyc.parser.rd.Parser.ctx.after_expression",
                &describe(self.peek()),
                self.peek().span,
            );
            return None;
        }
        if self.peek().kind == TokenKind::LeftBrace {
            let _ = self.block();
        } else {
            let _ = self.expression();
        }
        self.expect(&TokenKind::Semicolon, "after lambda body")?;
        let mut diag = Diagnostic::new(
            "E0040",
            Severity::Error,
            crate::t!("rubyc.parser.rd.statement.lambda_not_a_statement"),
        )
        .at(start.to(self.peek_at(0).span))
        .in_context(self.context.clone());
        diag.help = Some(
            "assign the lambda to a delegate-typed variable first, then call that".to_owned(),
        );
        self.errors.push(diag);
        let _ = params;
        None
    }

    /// Parse a control-flow body: a braced block, or a single statement
    /// (no braces). The single-statement form is used for compact
    /// constructs like `if (cond) stmt;` and `while (cond) stmt;`.
    pub(super) fn control_body(&mut self) -> Block {
        let start = self.peek().span;
        if self.peek().kind == TokenKind::LeftBrace {
            return self.block();
        }
        // Empty statement body: `while (c) ;`, `if (c) ;`. A lone `;`
        // is a no-op — return an empty block.
        if self.peek().kind == TokenKind::Semicolon {
            self.bump();
            let end = self.peek_at(0).span;
            return Block {
                stmts: Vec::new(),
                span: start.to(end),
            };
        }
        // Like `block()`: bindings queued by the enclosing statement's
        // header (condition `out` vars, `is`-pattern bindings) belong to
        // the outer block, not to this synthetic single-statement body.
        let stashed_pending = std::mem::take(&mut self.pending);
        let stashed_pre = std::mem::take(&mut self.pre_pending);
        let stmt = self.statement();
        let stmts = match stmt {
            Some(s) => {
                // Bindings declared during this statement's parsing (e.g. an
                // `is`-pattern binding var) must appear BEFORE the statement.
                let pre = std::mem::take(&mut self.pre_pending);
                let mut v = pre;
                v.push(s);
                v.append(&mut self.pending);
                v
            }
            None => {
                // Bail forward to avoid an infinite loop on a parse error.
                self.bump();
                std::mem::take(&mut self.pre_pending)
            }
        };
        let mut rest_pending = std::mem::take(&mut self.pending);
        self.pending = stashed_pending;
        self.pending.append(&mut rest_pending);
        let mut rest_pre = std::mem::take(&mut self.pre_pending);
        self.pre_pending = stashed_pre;
        self.pre_pending.append(&mut rest_pre);
        Block {
            stmts,
            span: start.to(self.peek_at(0).span),
        }
    }

    pub(super) fn statement(&mut self) -> Option<Stmt> {
        let start = self.peek().span;

        // Empty statement: a lone `;` (e.g. stray `foo(); ;`). Legal C#,
        // a no-op — same shape as `using`/`lock` lowering.
        if self.peek().kind == TokenKind::Semicolon {
            self.bump();
            return Some(Stmt::Break {
                span: start.to(self.peek_at(0).span),
            });
        }

        // `checked { stmts }` / `unchecked { stmts }`. Wrapping is the
        // default, so `unchecked` stays a plain block; `checked` becomes a
        // region the lowering arms overflow checks in.
        if matches!(
            self.peek().kind,
            TokenKind::Keyword(Keyword::Checked) | TokenKind::Keyword(Keyword::Unchecked)
        ) && self.peek_at(1).kind == TokenKind::LeftBrace
        {
            let is_checked =
                matches!(self.peek().kind, TokenKind::Keyword(Keyword::Checked));
            let start_kw = self.peek().span;
            self.bump(); // 'checked' / 'unchecked'
            if is_checked {
                let inner = self.statement()?;
                let (body, end) = match inner {
                    Stmt::Block { body, span } => (body, span),
                    other => (
                        crate::models::Block {
                            stmts: vec![other],
                            span: start_kw,
                        },
                        start_kw,
                    ),
                };
                return Some(Stmt::CheckedBlock {
                    body,
                    span: start_kw.to(end),
                });
            }
        }

        // `unsafe { stmts }` — an unsafe region. The backend does not
        // track safety (pointer ops lower uniformly via Deref/AddrOf),
        // so this is a plain block, like `checked` above.
        if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Unsafe))
            && self.peek_at(1).kind == TokenKind::LeftBrace
        {
            self.bump(); // 'unsafe'
        }

        // Standalone block: `{ stmts }`. The statements execute in
        // their own scope — keep them as a real Block node (dropping
        // them silences code).
        if self.peek().kind == TokenKind::LeftBrace {
            let body = self.block();
            let span = body.span;
            return Some(Stmt::Block { body, span });
        }

        // Goto label: `Name:` prefixing a statement. Preserved as
        // `Stmt::Label` (native lowering turns it into a block
        // target for `goto`). The colon must not be the first half
        // of `::`: `global::Ns` (and alias-qualified names) are
        // types, never labels.
        if matches!(self.peek().kind, TokenKind::Identifier(_))
            && self.peek_at(1).kind == TokenKind::Colon
            && self.peek_at(2).kind != TokenKind::Colon
        {
            let TokenKind::Identifier(name) = self.peek().kind.clone() else {
                unreachable!("peeked Identifier above");
            };
            let name_span = self.peek().span;
            self.bump(); // label name
            self.bump(); // ':'
            let label = Stmt::Label {
                name,
                span: name_span,
            };
            // A label at the end of a block has no following statement.
            if matches!(
                self.peek().kind,
                TokenKind::RightBrace | TokenKind::Eof | TokenKind::Keyword(Keyword::Case) | TokenKind::Keyword(Keyword::Default)
            ) {
                return Some(label);
            }
            // Wrap so both survive the single-statement return.
            return self.statement().map(|stmt| {
                Stmt::Block {
                    body: Block {
                        stmts: vec![label, stmt],
                        span: name_span,
                    },
                    span: name_span,
                }
            });
        }

        // Lambda expression at statement level: `name => expr;` or
        // `(params) => expr;` — a bodyless lambda. We consume the
        // expression-bodied form as a no-op for the round-trip.
        if self.looks_like_lambda() {
            return self.parse_lambda_statement(start);
        }

        // Local function: `Type Name(params) { body }` declared inside a
        // method body. C# 7+. Heuristic: the type's identifier, name,
        // and a `(` (with no `=` or `;` in between) means a function
        // definition. Attributes may precede it
        // (`[MethodImpl(...)] static int F() => ...`): consume them
        // only when a local function actually follows, rewinding
        // otherwise (a leading `[` can also start an expression).
        if self.peek().kind == TokenKind::LeftBracket {
            let mark = self.pos;
            let err_mark = self.errors.len();
            let attrs = self.attribute_lists();
            if self.looks_like_local_function() {
                return self.parse_local_function(start, attrs);
            }
            self.pos = mark;
            self.errors.truncate(err_mark);
        }
        if self.looks_like_local_function() {
            return self.parse_local_function(start, Vec::new());
        }

        // C# 7+ tuple type declaration: `(T1 Name1, T2 Name2, ...) name = ...;`.
        let start = self.peek().span;

        // `var` tuple deconstruction: `var (a, b) = expr;`, `var (T a, b)
        // = expr;`, nested `var (a, (b, c)) = expr;`. Each bound name is
        // exposed to the semantic scan as a zero-init `Let` (drained by
        // `block`); the initializer evaluates as the statement itself —
        // mirroring the destructuring-assignment form below.
        if self.var_tuple_decl_here() {
            self.bump(); // 'var'
            self.bump(); // '('
            let mut bindings = Vec::new();
            self.collect_tuple_bindings(&mut bindings);
            self.expect(&TokenKind::RightParen, "to close var tuple pattern")?;
            self.expect(&TokenKind::Assign, "after var tuple pattern")?;
            let value = self.expression()?;
            for name in bindings {
                if name != "_" {
                    self.pending.push(Stmt::Let {
                        is_const: false,
                        is_byref: false,
                        type_name: String::from("var"),
                        name,
                        init: Expr::Literal(Literal::Int(0)),
                        access: Access::Default,
                        span: start.to(self.peek_at(0).span),
                    });
                }
            }
            self.expect(
                &TokenKind::Semicolon,
                "rubyc.parser.rd.Parser.ctx.after_declaration",
            )?;
            return Some(Stmt::Expression(value));
        }

        // C# 7+ tuple type declaration: `(T1 Name1, T2 Name2, ...) name = ...;`
        // or destructuring `(T1 n1, T2 n2) = expr;`. Types may carry
        // dots/generics/suffixes (`Http3FrameType? frameType`,
        // `List<int> xs`); at least one element must be a `Type Name`
        // pair so `(a, b)` argument lists never match — EXCEPT when a
        // declarator visibly follows (`(A, B)[] x = ...`): an argument
        // list can never be followed by `name =`, so weak-unnamed
        // tuples are unambiguous there too. An optional `ref`/`scoped`
        // prefix (`ref (A, B) x = ref ...`) is consumed as inert.
        if self.tuple_decl_here() {
            // Re-skip the optional prefix for real (the gate rewound).
            while matches!(
                self.peek().kind,
                TokenKind::Keyword(Keyword::Ref_)
                    | TokenKind::Keyword(Keyword::Scoped)
                    | TokenKind::Keyword(Keyword::Readonly)
            ) {
                self.bump();
            }
            // Parse the tuple type `(T1 Name1, T2 Name2, ...)` element by
            // element, recovering the deconstruction target names. Each
            // element is `Type Name` (two identifiers) or `_` (discard) or
            // `Type` (unnamed). Nested parens (tuple-of-tuples) are skipped.
            let mut depth = 1i32;
            self.bump(); // '('
            // Deconstruction targets as (declared type, name): the
            // types flow into the synthesized `Let`s below so later
            // uses resolve precisely (`(A a, B b) = e; a.M()` types
            // `a` as `A`, not `var`/`int`). Elements without a
            // declared type (discards, contextual names) fall back
            // to `var`, as before.
            let mut decon_names: Vec<(String, String)> = Vec::new();
            loop {
                if self.peek().kind == TokenKind::LeftParen {
                    // Nested tuple element: skip to its matching ')'.
                    depth += 1;
                    self.bump();
                    while depth > 1 && self.peek().kind != TokenKind::Eof {
                        match self.peek().kind {
                            TokenKind::LeftParen => depth += 1,
                            TokenKind::RightParen => depth -= 1,
                            _ => {}
                        }
                        self.bump();
                    }
                    // A name may follow the nested group
                    // (`(char Char, ...) literal = ...`): it is a
                    // deconstruction target too — without it the use
                    // fails downstream (undeclared 'literal').
                    if let Some((nm, _)) = self.take_pattern_binding() {
                        decon_names.push((String::from("var"), nm));
                    }
                    continue;
                }
                if self.peek().kind == TokenKind::RightParen {
                    depth -= 1;
                    self.bump();
                    if depth == 0 {
                        break;
                    }
                    continue;
                }
                if self.peek().kind == TokenKind::Comma {
                    self.bump();
                    continue;
                }
                // An element: `Type Name` or `_` or `Type`, where the
                // type may carry dots/generics/suffixes
                // (`Http3FrameType? frameType`, `List<int> xs`).
                if matches!(self.peek().kind, TokenKind::Identifier(_)) {
                    let ty_start = self.peek().span.start as usize;
                    self.bump(); // type base
                    while self.peek().kind == TokenKind::Dot
                        && self.peek_is_name(1)
                    {
                        self.bump(); // '.'
                        self.bump(); // segment
                    }
                    if matches!(
                        self.peek().kind,
                        TokenKind::Less | TokenKind::Shl
                    ) {
                        self.skip_generic_parameter_list();
                    }
                    loop {
                        match self.peek().kind {
                            TokenKind::Question | TokenKind::Star => {
                                self.bump();
                            }
                            TokenKind::LeftBracket => {
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
                            _ => break,
                        }
                    }
                    if self.peek_is_name(0) {
                        // Slice the declared type text (base + dots +
                        // generics + suffixes) from the source: precise
                        // types flow into the synthesized `Let`s.
                        // Whitespace-free (spellings compare equal
                        // site-independently, like `type_name_full`).
                        let ty_end = self.peek().span.start as usize;
                        let ty_text: String = std::str::from_utf8(
                            &self.source[ty_start.min(self.source.len())
                                ..ty_end.min(self.source.len())],
                        )
                        .unwrap_or("var")
                        .chars()
                        .filter(|c| !c.is_whitespace())
                        .collect();
                        let ty_text =
                            if ty_text.is_empty() { String::from("var") } else { ty_text };
                        if let TokenKind::Identifier(name) = self.peek().kind.clone() {
                            decon_names.push((ty_text, name));
                        } else if matches!(self.peek().kind, TokenKind::Keyword(_)) {
                            // Contextual-keyword name (`from`, ...):
                            // record source text.
                            let span = self.peek().span;
                            let bytes = &self.source
                                [span.start as usize..span.end as usize];
                            decon_names.push((
                                ty_text,
                                std::str::from_utf8(bytes).unwrap_or("").to_owned(),
                            ));
                        }
                        self.bump(); // name
                    }
                } else if matches!(self.peek().kind, TokenKind::Keyword(_)) {
                    // `_` discard or a keyword type — skip.
                    self.bump();
                } else {
                    self.bump();
                }
            }
            // Decide between two forms:
            //   (a) `(T1 n1, T2 n2) name = ...;` — declaration with
            //       initializer. The tuple type may carry `[]`/`?`
            //       suffixes (`(T1 n1, T2 n2)[] arr = ...`).
            //   (b) `(T1 n1, T2 n2) = expr;` — destructuring assignment to
            //       already-declared names.
            // Array suffixes between `)` and the name (form (a) only).
            let mut tuple_suffix = String::new();
            while self.peek().kind == TokenKind::LeftBracket {
                self.bump(); // '['
                if self.peek().kind == TokenKind::RightBracket {
                    self.bump();
                    tuple_suffix.push_str("[]");
                } else {
                    while self.peek().kind != TokenKind::RightBracket
                        && self.peek().kind != TokenKind::Eof
                    {
                        self.bump();
                    }
                    if self.peek().kind == TokenKind::RightBracket {
                        self.bump();
                        tuple_suffix.push_str("[]");
                    }
                }
            }
            if self.peek().kind == TokenKind::Question {
                self.bump();
                tuple_suffix.push('?');
            }
            if matches!(self.peek().kind, TokenKind::Identifier(_)) {
                // Form (a).
                let name = self.expect_identifier("after tuple type")?;
                let init = if self.eat(&TokenKind::Assign).is_some() {
                    self.expression()?
                } else {
                    Expr::Literal(Literal::Int(0))
                };
                self.expect(
                    &TokenKind::Semicolon,
                    "rubyc.parser.rd.Parser.ctx.after_declaration",
                )?;
                return Some(Stmt::Let {
                    is_const: false,
                    is_byref: false,
                    type_name: format!("tuple{tuple_suffix}"),
                    name,
                    init,
                    access: Access::Default,
                    span: start.to(self.peek_at(0).span),
                });
            }
            if self.eat(&TokenKind::Assign).is_some() {
                // Form (b): destructuring assignment `(T1 n1, T2 n2) = expr;`.
                // Tuples lower as `tuple[]` arrays (see the literal rule),
                // so destructure by indexing a temp: each name reads its
                // slot once, in order, with a single evaluation of the
                // right-hand side. (The old code evaluated the rhs and
                // dropped it, leaving every name zero — `index += 0`
                // looped forever and OOMed init in
                // `CompareStringIgnoreCase`.)
                let value = self.expression()?;
                let end = start.to(self.peek_at(0).span);
                let tmp = "__tuple_tmp".to_owned();
                self.pending.push(Stmt::Let {
                    is_const: false,
                    is_byref: false,
                    type_name: "tuple[]".to_owned(),
                    name: tmp.clone(),
                    init: value,
                    access: Access::Default,
                    span: end,
                });
                for (ty, dn) in &decon_names {
                    self.pending.push(Stmt::Let {
                        is_const: false,
                        is_byref: false,
                        type_name: ty.clone(),
                        name: dn.clone(),
                        init: Expr::Literal(Literal::Int(0)),
                        access: Access::Default,
                        span: end,
                    });
                }
                for (i, (_, dn)) in decon_names.iter().enumerate() {
                    if dn == "_" {
                        continue;
                    }
                    self.pending.push(Stmt::Expression(Expr::Assign {
                        target: Box::new(Expr::Ident {
                            name: dn.clone(),
                            span: end,
                        }),
                        value: Box::new(Expr::Index {
                            object: Box::new(Expr::Ident {
                                name: tmp.clone(),
                                span: end,
                            }),
                            index: Box::new(Expr::Literal(Literal::Int(i as i64))),
                            extra: Vec::new(),
                            is_range: false,
                            span: end,
                        }),
                        span: end,
                    }));
                }
                self.expect(
                    &TokenKind::Semicolon,
                    "rubyc.parser.rd.Parser.ctx.after_declaration",
                )?;
                return Some(Stmt::Expression(Expr::Literal(Literal::Int(0))));
            }
            // Could not recover cleanly; bail.
            self.unexpected(
                "rubyc.parser.rd.Parser.ctx.after_declaration",
                &describe(self.peek()),
                self.peek().span,
            );
            return None;
        }

        // `return [expr];`
        if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Return)) {
            self.bump();
            let value = if self.peek().kind == TokenKind::Semicolon {
                None
            } else {
                Some(self.expression()?)
            };
            self.expect(
                &TokenKind::Semicolon,
                "rubyc.parser.rd.Parser.ctx.after_statement",
            )?;
            return Some(Stmt::Return {
                value,
                span: start.to(self.peek_at(0).span),
            });
        }

        match self.peek().kind.clone() {
            TokenKind::Keyword(Keyword::If) => return self.if_statement(),
            TokenKind::Keyword(Keyword::Lock) => {
                // `lock (obj) { ... }` — mutual exclusion. The backend
                // has no threading model, so the guarded expression is
                // still evaluated (it may carry side effects) and the
                // body runs as a plain block.
                self.bump(); // 'lock'
                self.expect(&TokenKind::LeftParen, "to open lock target")?;
                let _ = self.expression();
                self.expect(&TokenKind::RightParen, "to close lock target")?;
                let body = self.control_body();
                let span = body.span;
                return Some(Stmt::Block { body, span });
            }
            TokenKind::Keyword(Keyword::While) => return self.while_statement(),
            TokenKind::Keyword(Keyword::Do) => return self.do_while_statement(),
            TokenKind::Keyword(Keyword::For) => return self.for_statement(),
            TokenKind::Keyword(Keyword::Switch) => return self.switch_statement(),
            TokenKind::Keyword(Keyword::Break) => {
                self.bump();
                self.expect(&TokenKind::Semicolon, "after 'break'")?;
                return Some(Stmt::Break {
                    span: start.to(self.peek_at(0).span),
                });
            }
            TokenKind::Keyword(Keyword::Continue) => {
                self.bump();
                self.expect(&TokenKind::Semicolon, "after 'continue'")?;
                return Some(Stmt::Continue {
                    span: start.to(self.peek_at(0).span),
                });
            }
            TokenKind::Keyword(Keyword::Goto) => {
                self.bump();
                // `goto case <expr>;` or `goto default;`
                if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Case)) {
                    self.bump();
                    let case_expr = self.expression()?;
                    self.expect(&TokenKind::Semicolon, "after 'goto case'")?;
                    return Some(Stmt::Goto {
                        label: None,
                        case_expr: Some(case_expr),
                        span: start.to(self.peek_at(0).span),
                    });
                }
                if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Default)) {
                    self.bump();
                    self.expect(&TokenKind::Semicolon, "after 'goto default'")?;
                    return Some(Stmt::Goto {
                        label: None,
                        case_expr: None,
                        span: start.to(self.peek_at(0).span),
                    });
                }
                // `goto <label>;`
                let TokenKind::Identifier(label) = self.peek().kind.clone() else {
                    self.error(
                        "E0001",
                        "expected a label, 'case', or 'default' after 'goto'".to_owned(),
                        self.peek().span,
                    );
                    return None;
                };
                self.bump();
                self.expect(&TokenKind::Semicolon, "after 'goto'")?;
                return Some(Stmt::Goto {
                    label: Some(label),
                    case_expr: None,
                    span: start.to(self.peek_at(0).span),
                });
            }
            TokenKind::Keyword(Keyword::Await)
                if matches!(
                    self.peek_at(1).kind,
                    TokenKind::Keyword(Keyword::Using_)
                        | TokenKind::Keyword(Keyword::Foreach)
                ) =>
            {
                // `await using (...)` / `await foreach (... in ...)`
                // (System.Text.Json's `Write.Pipe`). The async part is
                // inert here; the handling below applies unchanged.
                self.bump(); // 'await'
                if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Using_)) {
                    return self.using_statement(start);
                }
                // `await foreach`: re-dispatch with the `await` consumed
                // (the next arm handles `foreach` normally).
                return self.statement();
            }
            TokenKind::Keyword(Keyword::Using_) => {
                return self.using_statement(start);
            }
            TokenKind::Keyword(Keyword::Foreach) => {
                // `foreach (Type var in expr) { ... }`. The compact AST
                // collapses the loop into a no-op for the bytecode
                // round-trip; the body is parsed normally.
                self.bump(); // 'foreach'
                self.expect(&TokenKind::LeftParen, "to open foreach header")?;
                // Deconstruction: `foreach ((T1 a, T2 b) in expr)` or
                // `foreach (var (a, b) in expr)` (C# 7; no outer
                // variable — the tuple IS the iteration variable).
                // Elements are `Type name`, `Type? name`, or discards
                // (`Type _` / bare `_`); the whole header still lowers
                // to the existing no-op shape, like the plain form.
                let is_var_tuple = matches!(&self.peek().kind, TokenKind::Identifier(n) if n == "var")
                    && self.peek_at(1).kind == TokenKind::LeftParen;
                if self.peek().kind == TokenKind::LeftParen || is_var_tuple {
                    if is_var_tuple {
                        self.bump(); // 'var'
                    }
                    self.bump(); // '('
                    loop {
                        if self.peek().kind == TokenKind::RightParen {
                            break;
                        }
                        // Element: `[ref] Type[?/[]] name` or discard.
                        while matches!(
                            self.peek().kind,
                            TokenKind::Keyword(Keyword::Ref_)
                                | TokenKind::Keyword(Keyword::Out_)
                                | TokenKind::Keyword(Keyword::In_)
                        ) {
                            self.bump();
                        }
                        let is_bare = matches!(self.peek().kind, TokenKind::Identifier(_))
                            && matches!(
                                self.peek_at(1).kind,
                                TokenKind::Comma | TokenKind::RightParen
                            );
                        if is_bare {
                            // Bare name or discard: `x` / `_`.
                            self.bump();
                        } else {
                            let _ = self.type_name();
                            // The name may be absent only for a bare
                            // discard already handled above; anything
                            // else (including `_`) consumes one name.
                            let _ = self.expect_identifier_or_keyword(
                                "rubyc.parser.rd.Parser.ctx.as_variable_name",
                            );
                        }
                        if self.eat(&TokenKind::Comma).is_none() {
                            break;
                        }
                    }
                    self.expect(&TokenKind::RightParen, "to close foreach pattern")?;
                    // Optional enclosing variable: `((int a, int b) item
                    // in ...)` names the tuple itself alongside the
                    // deconstruction. (`in` is reserved, never a name,
                    // so a name here is unambiguously the variable.)
                    if self.peek_is_name(0) {
                        self.bump(); // outer variable
                    }
                    if matches!(self.peek().kind, TokenKind::Keyword(Keyword::In_)) {
                        self.bump();
                    }
                    let _ = self.expression();
                    self.expect(&TokenKind::RightParen, "to close foreach header")?;
                    let _ = self.control_body();
                    return Some(Stmt::Break {
                        span: start.to(self.peek_at(0).span),
                    });
                }
                // Iteration-variable modifiers: `ref`, `ref readonly`,
                // `ref struct` element types are handled by type_name.
                // Only the by-ref/readonly markers need skipping here.
                if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Ref_)) {
                    self.bump();
                    if matches!(
                        self.peek().kind,
                        TokenKind::Keyword(Keyword::Readonly)
                    ) {
                        self.bump();
                    }
                }
                let var_type = self
                    .type_name()
                    .unwrap_or_else(|| "var".to_owned());
                let var_name = self.expect_identifier("as foreach variable")?;
                if matches!(self.peek().kind, TokenKind::Keyword(Keyword::In_)) {
                    self.bump();
                }
                let coll = self.expression()?;
                self.expect(&TokenKind::RightParen, "to close foreach header")?;
                let body = self.control_body();
                // Desugar to an index loop over `Length` (arrays,
                // strings, and spans lower both correctly, so those
                // iterate for real). Anything without `Length`
                // reads zero and skips the body — the legacy no-op.
                // Types WITH `Length` but no integer indexer (invalid
                // C# — no enumerator exists either) may iterate
                // garbage instead of skipping; no valid program does
                // this.
                let n = self.foreach_seq;
                self.foreach_seq += 1;
                let coll_name = format!("__foreach_coll_{n}");
                let idx_name = format!("__foreach_idx_{n}");
                let sp = start.to(self.peek_at(0).span);
                let ident = |name: &str| Expr::Ident {
                    name: name.to_owned(),
                    span: sp,
                };
                let mut stmts = vec![Stmt::Let {
                    is_const: false,
                    is_byref: false,
                    type_name: var_type,
                    name: var_name,
                    init: Expr::Index {
                        object: Box::new(ident(&coll_name)),
                        index: Box::new(ident(&idx_name)),
                        extra: Vec::new(),
                        is_range: false,
                        span: sp,
                    },
                    access: Access::Default,
                    span: sp,
                }];
                stmts.extend(body.stmts);
                return Some(Stmt::Block {
                    body: Block {
                        stmts: vec![
                            Stmt::Let {
                                is_const: false,
                                is_byref: false,
                                type_name: "var".to_owned(),
                                name: coll_name.clone(),
                                init: coll,
                                access: Access::Default,
                                span: sp,
                            },
                            Stmt::For {
                                init: Some(Box::new(Stmt::Let {
                                    is_const: false,
                                    is_byref: false,
                                    type_name: "int".to_owned(),
                                    name: idx_name.clone(),
                                    init: Expr::Literal(Literal::Int(0)),
                                    access: Access::Default,
                                    span: sp,
                                })),
                                cond: Some(Expr::Binary {
                                    op: BinOp::Lt,
                                    lhs: Box::new(ident(&idx_name)),
                                    rhs: Box::new(Expr::Member {
                                        object: Box::new(ident(&coll_name)),
                                        name: "Length".to_owned(),
                                        span: sp,
                                    }),
                                    span: sp,
                                }),
                                step: Some(Box::new(Stmt::Assign {
                                    name: idx_name.clone(),
                                    value: Expr::Binary {
                                        op: BinOp::Add,
                                        lhs: Box::new(ident(&idx_name)),
                                        rhs: Box::new(Expr::Literal(Literal::Int(1))),
                                        span: sp,
                                    },
                                    span: sp,
                                })),
                                body: Block { stmts, span: sp },
                                span: sp,
                            },
                        ],
                        span: sp,
                    },
                    span: sp,
                });
            }
            TokenKind::Keyword(Keyword::From) if matches!(self.peek_at(1).kind, TokenKind::Identifier(_)) => {
                // LINQ query expression: `from x in xs [where/select/...]`.
                // We swallow the whole expression as a no-op so the
                // statement boundary (`;`) is recognized.
                while self.peek().kind != TokenKind::Semicolon
                    && self.peek().kind != TokenKind::Eof
                {
                    if self.peek().kind == TokenKind::LeftBrace {
                        // Stop at a brace so block-scoped queries still
                        // match correctly.
                        break;
                    }
                    self.bump();
                }
                self.expect(&TokenKind::Semicolon, "to end query expression")?;
                return Some(Stmt::Break {
                    span: start.to(self.peek_at(0).span),
                });
            }
            TokenKind::Keyword(Keyword::Fixed) => {
                // `fixed (type* varName = expr, ...) { ... }` — multiple
                // declarators share the one type (C#). Preserved as
                // `Stmt::Fixed` (names, inits, and body all survive);
                // lowering pins each name to its data address and runs
                // the body. Previously this dropped the inits (binding
                // zero) and the body (returning no-op), so all pinned
                // code silently vanished.
                self.bump(); // 'fixed'
                self.expect(&TokenKind::LeftParen, "to open fixed statement")?;
                // Parse the declaration: `type* varName = expr`.
                let type_name = self.type_name()?;
                // Optional `*` for pointer types.
                if self.eat(&TokenKind::Star).is_some() {
                    // The `*` is consumed; type_name already holds the base type.
                }
                let mut declarators: Vec<(String, String, Option<Expr>)> = Vec::new();
                if let Some(name) = self.expect_identifier("as fixed variable name") {
                    let init = if self.eat(&TokenKind::Assign).is_some() {
                        self.expression()
                    } else {
                        None
                    };
                    declarators.push((type_name.clone(), name, init));
                }
                // Further declarators (`, name [= expr]*) share the type.
                while self.eat(&TokenKind::Comma).is_some() {
                    // A `,` here always continues the declarator list
                    // (fixed headers contain nothing else); a missing
                    // name still errors below, preserving diagnostics.
                    if let Some(extra) = self.expect_identifier("as fixed variable name") {
                        let extra_init = if self.eat(&TokenKind::Assign).is_some() {
                            self.expression()
                        } else {
                            None
                        };
                        declarators.push((type_name.clone(), extra, extra_init));
                    }
                }
                self.expect(&TokenKind::RightParen, "to close fixed statement")?;
                // Body: either a block or a single nested statement
                // (`fixed (a = ..) fixed (b = ..) { ... }` nests).
                let body = if self.peek().kind == TokenKind::LeftBrace {
                    self.block()
                } else if let Some(stmt) = self.statement() {
                    Block {
                        stmts: vec![stmt],
                        span: start.to(self.peek_at(0).span),
                    }
                } else {
                    self.synchronize();
                    Block {
                        stmts: Vec::new(),
                        span: start.to(self.peek_at(0).span),
                    }
                };
                return Some(Stmt::Fixed {
                    declarators,
                    body,
                    span: start.to(self.peek_at(0).span),
                });
            }
            TokenKind::Keyword(Keyword::Yield) => {
                // `yield return expr;` or `yield break;`
                self.bump(); // 'yield'
                if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Return)) {
                    self.bump(); // 'return'
                    let _value = self.expression()?;
                    self.expect(&TokenKind::Semicolon, "after 'yield return'")?;
                } else if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Break)) {
                    self.bump(); // 'break'
                    self.expect(&TokenKind::Semicolon, "after 'yield break'")?;
                }
                return Some(Stmt::Break {
                    span: start.to(self.peek_at(0).span),
                });
            }
            TokenKind::Keyword(Keyword::Try) => {
                // `try { ... } catch (T e) { ... } ... finally { ... }`
                self.bump(); // 'try'
                let body = self.block();
                let mut catches = Vec::new();
                while matches!(self.peek().kind, TokenKind::Keyword(Keyword::Catch)) {
                    let cs = self.peek().span;
                    self.bump(); // 'catch'
                    let mut ty = None;
                    let mut name = None;
                    if self.eat(&TokenKind::LeftParen).is_some() {
                        ty = Some(self.type_name()?);
                        if matches!(self.peek().kind, TokenKind::Identifier(_)) {
                            name = Some(self.expect_identifier("catch variable name")?);
                        }
                        self.expect(&TokenKind::RightParen, "to close catch parameter")?;
                    }
                    // `when (filter)` — part of the clause's test: it is
                    // evaluated after the type matches, and a false
                    // filter falls through to the next clause. It used to
                    // be parsed and DISCARDED, so `catch (E) when (false)`
                    // caught anyway.
                    let mut filter: Option<Expr> = None;
                    if matches!(self.peek().kind, TokenKind::Keyword(Keyword::When_)) {
                        self.bump(); // 'when'
                        self.expect(&TokenKind::LeftParen, "to open when clause")?;
                        filter = self.expression();
                        self.expect(&TokenKind::RightParen, "to close when clause")?;
                    }
                    let cbody = self.block();
                    catches.push(CatchClause {
                        ty,
                        name,
                        filter,
                        body: cbody,
                        span: cs.to(self.peek_at(0).span),
                    });
                }
                let mut finally_block = None;
                if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Finally)) {
                    self.bump(); // 'finally'
                    finally_block = Some(self.block());
                }
                return Some(Stmt::Try {
                    body,
                    catches,
                    finally_block,
                    span: start.to(self.peek_at(0).span),
                });
            }
            TokenKind::Keyword(Keyword::Throw) => {
                // `throw expr;` or bare `throw;` (rethrow).
                self.bump(); // 'throw'
                let expr = if self.peek().kind != TokenKind::Semicolon {
                    Some(self.expression()?)
                } else {
                    None
                };
                self.expect(&TokenKind::Semicolon, "after 'throw'")?;
                return Some(Stmt::Throw {
                    expr,
                    span: start.to(self.peek_at(0).span),
                });
            }
            _ => {}
        }

        // Array type declaration: `T[] name = ...`
        if matches!(self.peek().kind, TokenKind::Identifier(_))
            && self.peek_at(1).kind == TokenKind::LeftBracket
            && self.peek_at(2).kind == TokenKind::RightBracket
            && matches!(self.peek_at(3).kind, TokenKind::Identifier(_))
        {
            let elem_type = self.expect_identifier("as element type")?;
            self.bump(); // '['
            self.bump(); // ']'
            let name = self.expect_identifier("as variable name")?;
            let init = if self.eat(&TokenKind::Assign).is_some() {
                self.expression()?
            } else {
                Expr::Literal(Literal::Int(0))
            };
            // Trailing declarators share the element type
            // (`XPathNode[] pageTemp, pageNew, pageCopy;`): queue each
            // as its own `Let` like the general declaration path.
            while self.eat(&TokenKind::Comma).is_some() {
                let next = self.expect_identifier("as variable name")?;
                let next_init = if self.eat(&TokenKind::Assign).is_some() {
                    self.expression()?
                } else {
                    Expr::Literal(Literal::Int(0))
                };
                self.pending.push(Stmt::Let {
                    is_const: false,
                    is_byref: false,
                    type_name: format!("{elem_type}[]"),
                    name: next,
                    init: next_init,
                    access: Access::Default,
                    span: start.to(self.peek_at(0).span),
                });
            }
            self.expect(&TokenKind::Semicolon, "after array declaration")?;
            return Some(Stmt::Let {
                is_const: false,
                is_byref: false,
                type_name: format!("{elem_type}[]"),
                name,
                init,
                access: Access::Default,
                span: start.to(self.peek_at(0).span),
            });
        }

        // Function pointer type: `delegate*[unmanaged[cc]]<RetType, ...>
        // name = expr;`
        if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Delegate_))
            && self.peek_at(1).kind == TokenKind::Star
        {
            self.bump(); // 'delegate'
            self.bump(); // '*'
            // Optional calling convention: `unmanaged`, `unmanaged[Cdecl]`.
            if matches!(self.peek().kind, TokenKind::Identifier(_)) {
                self.bump();
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
            self.expect(&TokenKind::Less, "to open function pointer type")?;
            // Parse the type argument list: `RetType, Param1, ...`.
            // Parameters may carry `ref` / `in` / `out` modifiers
            // (`delegate*<ref byte, void>`), recorded as a prefix —
            // mirroring .NET, where they are part of the signature.
            let mut type_args = Vec::new();
            if self.peek().kind != TokenKind::Greater {
                loop {
                    let prefix = match self.peek().kind {
                        TokenKind::Keyword(Keyword::Ref_) => "ref ",
                        TokenKind::Keyword(Keyword::In_) => "in ",
                        TokenKind::Keyword(Keyword::Out_) => "out ",
                        _ => "",
                    };
                    if !prefix.is_empty() {
                        self.bump();
                    }
                    let ty = self.type_name()?;
                    type_args.push(format!("{prefix}{ty}"));
                    if self.eat(&TokenKind::Comma).is_none() {
                        break;
                    }
                }
            }
            self.expect(&TokenKind::Greater, "to close function pointer type")?;
            let name = self.expect_identifier("as function pointer name")?;
            let init = if self.eat(&TokenKind::Assign).is_some() {
                self.expression()?
            } else {
                Expr::Literal(Literal::Int(0))
            };
            self.expect(&TokenKind::Semicolon, "after function pointer declaration")?;
            let type_str = format!("delegate*<{}>", type_args.join(", "));
            return Some(Stmt::Let {
                is_const: false,
                is_byref: false,
                type_name: type_str,
                name,
                init,
                access: Access::Default,
                span: start.to(self.peek_at(0).span),
            });
        }

        // Local constant: `const Type name = expr;`. Const-ness is a
        // compile-time assignment rule, not a runtime representation —
        // lower exactly like a regular local declaration.
        let mut saw_const = false;
        if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Const)) {
            self.bump();
            saw_const = true;
        }

        // Declaration: `Type name = expr;` / `Type name;`
        //
        // Types are ordinary identifiers, so `boom int` must NOT read as
        // "declare int of type boom". Enter this arm only for a builtin
        // type name, or the unmistakable `T x =` shape; anything else
        // falls through to expression-statement, whose diagnostic lands
        // on the actual typo.
        if saw_const || self.likely_type_decl() {
            // `ref` / `out` / `scoped` modifiers on the local:
            // `ref int x = ref y;`, `scoped Span<char> s = ..;`.
            // `ref`/`out` locals alias their initializer's address
            // (see `is_byref` on `Stmt::Let`); the rest discard.
            let mut is_byref_local = false;
            for _ in 0..4 {
                if matches!(
                    self.peek().kind,
                    TokenKind::Keyword(Keyword::Ref_)
                        | TokenKind::Keyword(Keyword::Out_)
                        | TokenKind::Keyword(Keyword::Scoped)
                        | TokenKind::Keyword(Keyword::Readonly)
                ) {
                    if matches!(
                        self.peek().kind,
                        TokenKind::Keyword(Keyword::Ref_)
                            | TokenKind::Keyword(Keyword::Out_)
                    ) {
                        is_byref_local = true;
                    }
                    self.bump();
                } else {
                    break;
                }
            }
            let type_name = self.type_name_full()?;
            let name = self.expect_identifier("rubyc.parser.rd.Parser.ctx.as_variable_name")?;
            let init = if self.eat(&TokenKind::Assign).is_some() {
                // A plain lambda RHS (`var f = (x) => x;`) is handled by
                // `expression()` below, which builds a real Lambda node.
                // The explicit-return-type form is `Type (params) => body`
                // and is genuinely different, so this arm is only taken
                // when there IS a return type: with the cursor already at
                // `(`, the parenthesised list is the PARAMETERS, not a
                // return type. Getting this wrong used to swallow the
                // body: `var f = (x) => x;` produced `init: 0` with the
                // lambda discarded, and the leftover `(x) => x;` was then
                // re-parsed as a statement and rejected.
                if self.peek().kind != TokenKind::LeftParen
                    && self.looks_like_typed_lambda()
                {
                        while self.peek().kind == TokenKind::LeftBracket {
                        self.bump();
                        let mut depth = 1i32;
                        loop {
                            if self.peek().kind == TokenKind::LeftBracket {
                                depth += 1;
                            } else if self.peek().kind == TokenKind::RightBracket {
                                depth -= 1;
                            }
                            self.bump();
                            if depth == 0 || self.peek().kind == TokenKind::Eof {
                                break;
                            }
                        }
                    }
                    // Skip the optional return-type identifier and any
                    // suffixes (`[]`, `?`, generics). Distinguish from
                    // a parameter list: if the next token is `(`, there
                    // is no return type to skip.
                    if matches!(self.peek().kind, TokenKind::Identifier(_)) {
                        self.bump();
                        while self.peek().kind == TokenKind::Question
                            || (self.peek().kind == TokenKind::LeftBracket
                                && self.peek_at(1).kind == TokenKind::RightBracket)
                        {
                            self.bump();
                            if self.peek().kind == TokenKind::RightBracket {
                                self.bump();
                            }
                        }
                        while self.peek().kind == TokenKind::Dot {
                            self.bump();
                            self.bump(); // dotted segment identifier
                        }
                    }
                    let _ = self.parse_lambda_body();
                    return Some(Stmt::Let {
                        is_const: false,
                        is_byref: false,
                        type_name,
                        name,
                        init: Expr::Literal(Literal::Int(0)),
                        access: Access::Default,
                        span: start.to(self.peek_at(0).span),
                    });
                }
                let expr = self.expression();
                // Lambda at RHS: `Type name = params => body;` (statement
                // position lambda). Also handles `static params =>`.
                if matches!(self.peek().kind, TokenKind::FatArrow)
                    || matches!(self.peek().kind, TokenKind::Keyword(Keyword::Static))
                {
                    if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Static)) {
                        self.bump();
                    }
                    let _ = self.parse_lambda_body();
                    // The lambda body is skipped (not parsed into the AST),
                    // so the param is never referenced in the AST. Use a
                    // neutral init so the scanner does not try to resolve
                    // the param name.
                    return Some(Stmt::Let {
                        is_const: false,
                        is_byref: false,
                        type_name,
                        name,
                        init: Expr::Literal(Literal::Int(0)),
                        access: Access::Default,
                        span: start.to(self.peek_at(0).span),
                    });
                }
                expr?
            } else {
                // No '=' after the declared name. `Type name;` (or
                // `Type a, b;` — the multi-declarator loop below eats the
                // comma) is a legitimate C# uninitialized declaration
                // (definite assignment is enforced by flow, not syntax):
                // synthesize zero with no diagnostic. Only any other
                // continuation is a genuine missing-'=' error.
                if matches!(
                    self.peek().kind,
                    TokenKind::Semicolon | TokenKind::Comma
                ) {
                    Expr::Literal(Literal::Int(0))
                } else {
                    // Missing '=' between the declared name and its value.
                    // Point at the token where '=' belongs; keep parsing
                    // the initializer so later errors still surface.
                    let span = self.peek().span;
                    let mut diag = Diagnostic::new(
                        "E0002",
                        Severity::Error,
                        format!("syntax error: missing '=' in declaration of '{name}'"),
                    )
                    .at(span)
                    .in_context(self.context.clone());
                    diag.help = Some(format!("write `{type_name} {name} = <value>;`"));
                    self.errors.push(diag);
                    match self.expression() {
                        Some(e) => e,
                        None => Expr::Literal(Literal::Int(0)),
                    }
                }
            };
            // Multi-declarator: `int a = 1, b = 2, c;` or `nint a = (nint)b, c;`.
            // Each trailing declarator becomes its own `Stmt::Let`, queued on
            // `pending` and drained by `block()`. Previously the trailing
            // names were dropped, so downstream passes never saw them.
            while self.eat(&TokenKind::Comma).is_some() {
                // Optional `T` type prefix when the next thing starts with
                // a cast or another type, e.g. `, T name = ...` or
                // `, (T)name = ...`.
                if matches!(self.peek().kind, TokenKind::Identifier(_))
                    && matches!(self.peek_at(1).kind, TokenKind::Identifier(_))
                {
                    self.bump();
                }
                if matches!(self.peek().kind, TokenKind::LeftParen) {
                    // Cast initializer — skip the parenthesized type.
                    let mut depth = 1i32;
                    self.bump(); // '('
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
                }
                let tstart = self.peek().span;
                let tname = self.expect_identifier("as variable name")?;
                let init = if self.eat(&TokenKind::Assign).is_some() {
                    self.expression()
                        .unwrap_or_else(|| Expr::Literal(Literal::Int(0)))
                } else {
                    Expr::Literal(Literal::Int(0))
                };
                self.pending.push(Stmt::Let {
                    is_const: saw_const,
                    is_byref: false,
                    type_name: type_name.clone(),
                    name: tname,
                    init,
                    access: Access::Default,
                    span: tstart.to(self.peek_at(0).span),
                });
            }
            self.expect(
                &TokenKind::Semicolon,
                "rubyc.parser.rd.Parser.ctx.after_declaration",
            )?;
            return Some(Stmt::Let {
                is_const: saw_const,
                is_byref: is_byref_local,
                type_name,
                name,
                init,
                access: Access::Default,
                span: start.to(self.peek_at(0).span),
            });
        }

        // Assignment: `name = expr;` and compound forms `name op= expr;`
        if matches!(self.peek().kind, TokenKind::Identifier(_))
            && let Some((op, width)) = self.compound_assign_op_at(1)
        {
            let name = self.expect_identifier("as assignment target")?;
            self.pos += width;
            let rhs = self.expression()?;
            self.expect(
                &TokenKind::Semicolon,
                "rubyc.parser.rd.Parser.ctx.after_assignment",
            )?;
            // Desugar `x op= rhs` to `x = x op rhs`. Both sides of a plain
            // name read have no side effects, so this is order-safe.
            let lhs = Expr::Ident {
                name: name.clone(),
                span: start,
            };
            let value = match op {
                Some(binop) => Expr::Binary {
                    op: binop,
                    lhs: Box::new(lhs.clone()),
                    rhs: Box::new(rhs),
                    span: start.to(self.peek_at(0).span),
                },
                None => rhs,
            };
            return Some(Stmt::Assign {
                name,
                value,
                span: start.to(self.peek_at(0).span),
            });
        }

        // Compound assignment with complex LHS: `*p += expr;`, `arr[i] += expr;`, etc.
        // Look ahead to find a top-level compound assignment operator
        // (nesting-aware: operators inside parens/brackets/braces do
        // not trigger).
        if self.find_compound_assign_ahead().is_some() {
            // Parse as an expression. The expression-level `op=`
            // handling consumes the assignment itself for every target
            // shape (names, members, indexers, derefs); then only the
            // `;` remains.
            let mark = self.pos;
            let err_mark = self.errors.len();
            let pending_mark = self.pending.len();
            let pre_mark = self.pre_pending.len();
            let lhs = self.expression()?;
            if matches!(
                lhs,
                Expr::Assign { .. } | Expr::AssignComplex { .. }
            ) {
                self.expect(
                    &TokenKind::Semicolon,
                    "rubyc.parser.rd.Parser.ctx.after_assignment",
                )?;
                return Some(Stmt::Expression(lhs));
            }
            // Otherwise the trigger fired on something the expression
            // parser shaped differently (e.g. a compound inside a
            // ternary: `c ? a += 1 : b;`) — rewind and let the generic
            // paths below handle it. Never blindly bump: the cursor
            // sits after a complete expression, so a bump would eat
            // the `;` and corrupt the following statement. Rewind
            // queued bindings too (see above).
            self.pos = mark;
            self.errors.truncate(err_mark);
            self.pending.truncate(pending_mark);
            self.pre_pending.truncate(pre_mark);
        }

        // Increment / decrement as statement position sugar:
        // `x++;` ≡ `x = x + 1;` (value form `y = x++` arrives later).
        if matches!(self.peek().kind, TokenKind::Identifier(_))
            && matches!(
                self.peek_at(1).kind,
                TokenKind::PlusPlus | TokenKind::MinusMinus
            )
        {
            let name = self.expect_identifier("as increment target")?;
            // The operator sits directly ahead now that the name is consumed.
            let up = self.peek().kind == TokenKind::PlusPlus;
            self.bump();
            self.expect(&TokenKind::Semicolon, "after increment/decrement")?;
            return Some(Stmt::Assign {
                name: name.clone(),
                value: Expr::Binary {
                    op: if up { BinOp::Add } else { BinOp::Sub },
                    lhs: Box::new(Expr::Ident { name, span: start }),
                    rhs: Box::new(Expr::Literal(Literal::Int(1))),
                    span: start,
                },
                span: start,
            });
        }

        // Complex assignment targets: `expr.ident = expr;` or
        // `expr[index] = expr;`. Position save/restore keeps the
        // plain-expression fall-through intact when no `=` follows.
        // Queue save/restore too: the speculative parse may queue
        // bindings (`out` vars, pattern Lets) that the re-parse would
        // otherwise duplicate.
        let mark = self.pos;
        let pending_mark = self.pending.len();
        let pre_mark = self.pre_pending.len();
        if let Some(lhs) = self.expression()
            && self.peek().kind == TokenKind::Assign
        {
            self.bump(); // '='
            let value = self.expression()?;
            self.expect(
                &TokenKind::Semicolon,
                "rubyc.parser.rd.Parser.ctx.after_assignment",
            )?;
            // For the compact AST, all non-trivial LHS shapes
            // (member, indexer, etc.) collapse into a single
            // AssignComplex node. The structural round-trip is
            // stable because both ends synthesize the same shape.
            return Some(Stmt::Expression(Expr::AssignComplex {
                lhs: Box::new(lhs),
                value: Box::new(value),
                span: start.to(self.peek_at(0).span),
            }));
        }
        self.pos = mark;
        self.pending.truncate(pending_mark);
        self.pre_pending.truncate(pre_mark);

        let expr = self.expression()?;
        self.expect(
            &TokenKind::Semicolon,
            "rubyc.parser.rd.Parser.ctx.after_statement",
        )?;
        Some(Stmt::Expression(expr))
    }

    /// If tokens at `offset` form a (compound) assignment operator, return
    /// its binary equivalent and token width. Plain `=` yields `(None, 1)`.
    fn compound_assign_op_at(&self, offset: usize) -> Option<(Option<BinOp>, usize)> {
        use TokenKind::*;
        let kind = &self.peek_at(offset).kind;
        let (op, width) = match kind {
            Assign => (None, 1),
            PlusAssign => (Some(BinOp::Add), 1),
            MinusAssign => (Some(BinOp::Sub), 1),
            StarAssign => (Some(BinOp::Mul), 1),
            SlashAssign => (Some(BinOp::Div), 1),
            PercentAssign => (Some(BinOp::Rem), 1),
            AmpAssign => (Some(BinOp::BitAnd), 1),
            PipeAssign => (Some(BinOp::BitOr), 1),
            CaretAssign => (Some(BinOp::BitXor), 1),
            ShlAssign => (Some(BinOp::Shl), 1),
            ShrAssign => (Some(BinOp::Shr), 1),
            ShrShrAssign => (Some(BinOp::ShrShr), 1),
            QuestionQuestionAssign => (Some(BinOp::NullCoalesce), 1),
            _ => return None,
        };
        Some((op, width))
    }

    /// Look ahead in the token stream to find a compound assignment operator
    /// (e.g. `+=`, `-=`, `*=`, etc.) that is not part of a type or expression.
    fn find_compound_assign_ahead(&self) -> Option<BinOp> {
        use TokenKind::*;
        // Scan ahead up to 40 tokens for a TOP-LEVEL compound assignment
        // operator. Nesting (`()`, `[]`, `{}`) is tracked: an operator
        // inside parens — e.g. the `??=` in `(x ??= y).Add(z);` — belongs
        // to the inner expression, and firing on it misparses the whole
        // statement. Stop at statement end, `=>` (lambda whose body may
        // hold assignments), or EOF.
        let mut depth = 0i32;
        for i in 0..40 {
            let kind = &self.peek_at(i).kind;
            match kind {
                LeftParen | LeftBracket | LeftBrace => depth += 1,
                RightParen | RightBracket | RightBrace => {
                    if depth == 0 {
                        return None;
                    }
                    depth -= 1;
                }
                _ if depth > 0 => {}
                PlusAssign => return Some(BinOp::Add),
                MinusAssign => return Some(BinOp::Sub),
                StarAssign => return Some(BinOp::Mul),
                SlashAssign => return Some(BinOp::Div),
                PercentAssign => return Some(BinOp::Rem),
                AmpAssign => return Some(BinOp::BitAnd),
                PipeAssign => return Some(BinOp::BitOr),
                CaretAssign => return Some(BinOp::BitXor),
                ShlAssign => return Some(BinOp::Shl),
                ShrAssign => return Some(BinOp::Shr),
                ShrShrAssign => return Some(BinOp::ShrShr),
                QuestionQuestionAssign => return Some(BinOp::NullCoalesce),
                Semicolon | RightBrace | Eof | FatArrow => return None,
                _ => {}
            }
        }
        None
    }

    /// `if (cond) { … } [else { … } | else if …]`
    fn if_statement(&mut self) -> Option<Stmt> {
        let start = self.peek().span;
        self.bump(); // 'if'
        self.expect(&TokenKind::LeftParen, "to open if condition")?;
        let cond = self.expression()?;
        self.expect(&TokenKind::RightParen, "to close if condition")?;
        let then_block = self.control_body();
        let else_block = if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Else)) {
            self.bump();
            // `else if` chains as a nested block holding one statement.
            if matches!(self.peek().kind, TokenKind::Keyword(Keyword::If)) {
                let inner_start = self.peek().span;
                let inner = self.if_statement()?;
                Some(Block {
                    stmts: vec![inner],
                    span: inner_start,
                })
            } else {
                Some(self.control_body())
            }
        } else {
            None
        };
        Some(Stmt::If {
            cond,
            then_block,
            else_block,
            span: start.to(self.peek_at(0).span),
        })
    }

    /// `while (cond) { … }` (or a single-statement body without braces)
    fn while_statement(&mut self) -> Option<Stmt> {
        let start = self.peek().span;
        self.bump(); // 'while'
        self.expect(&TokenKind::LeftParen, "to open while condition")?;
        let cond = self.expression()?;
        self.expect(&TokenKind::RightParen, "to close while condition")?;
        let body = self.control_body();
        Some(Stmt::While {
            cond,
            body,
            span: start.to(self.peek_at(0).span),
        })
    }

    /// `do { … } while (cond);` — body must be a block.
    /// `do <single-statement> while (cond);` (no braces — as in
    /// `ExpressionParser.ScanDate`'s `do _pos++; while (...)`) shares
    /// `control_body` with the other loops (same scoping/empty-body
    /// rules).
    fn do_while_statement(&mut self) -> Option<Stmt> {
        let start = self.peek().span;
        self.bump(); // 'do'
        let body = self.control_body();
        if !matches!(self.peek().kind, TokenKind::Keyword(Keyword::While)) {
            return None; // caller's synchronize() reports the position
        }
        self.bump();
        self.expect(&TokenKind::LeftParen, "to open do-while condition")?;
        let cond = self.expression()?;
        self.expect(&TokenKind::RightParen, "to close do-while condition")?;
        self.expect(&TokenKind::Semicolon, "after do-while")?;
        Some(Stmt::DoWhile {
            body,
            cond,
            span: start.to(self.peek_at(0).span),
        })
    }

    /// `for ([init]; [cond]; [step]) { … }` — init is a declaration or
    /// assignment; step is an assignment or increment.
    /// One or more comma-separated for-header expression statements
    /// (`i = 0, j = 0`): a single statement stays bare, several wrap
    /// in a `Block` (same shape as the multi-step clause below).
    /// Returns `None` only when the first slot fails to parse.
    fn for_header_exprs(&mut self) -> Option<Stmt> {
        let start = self.peek().span;
        let first = self.for_clause_stmt()?;
        if self.peek().kind != TokenKind::Comma {
            return Some(first);
        }
        let mut stmts = vec![first];
        while self.eat(&TokenKind::Comma).is_some() {
            if let Some(s) = self.for_clause_stmt() {
                stmts.push(s);
            } else {
                break;
            }
        }
        Some(Stmt::Block {
            body: crate::models::Block {
                stmts,
                span: start.to(self.peek_at(0).span),
            },
            span: start,
        })
    }

    fn for_statement(&mut self) -> Option<Stmt> {
        let start = self.peek().span;
        self.bump(); // 'for'
        self.expect(&TokenKind::LeftParen, "to open for header")?;

        // init clause: a declaration takes a full type (`T`, `T?`,
        // `T[]`, `Ns.Generic<K>.Inner`, ...) plus `name = value`.
        // `likely_type_decl` covers the `Type Name` shape; the parse
        // below is speculative — anything but `name =` rewinds to the
        // expression-statement path (C# for-init declarations always
        // initialize, so a missing `=` means "not a declaration").
        // Further declarators (`for (int i = 0, n = len; ...)`) collect
        // into `extra_inits`; the tail hoists them with the first into
        // a wrapping block around a bare-init `for`.
        let mut extra_inits: Vec<Stmt> = Vec::new();
        let init: Option<Box<Stmt>> = if self.peek().kind == TokenKind::Semicolon {
            self.bump();
            None
        } else if self.likely_type_decl() {
            let mark = self.pos;
            let err_mark = self.errors.len();
            let type_name = self.type_name()?;
            // Name without erroring (rewind on failure instead):
            // identifiers and contextual keywords, by source text.
            let name_opt = if self.peek_is_name(0) {
                let span = self.peek().span;
                let bytes =
                    &self.source[span.start as usize..span.end as usize];
                let n = std::str::from_utf8(bytes).unwrap_or("").to_owned();
                self.bump();
                Some(n)
            } else {
                None
            };
            match (name_opt, self.peek().kind.clone()) {
                (Some(name), TokenKind::Assign) => {
                    self.bump(); // '='
                    let value = self.expression()?;
                    let first = Stmt::Let {
                        is_const: false,
                        is_byref: false,
                        type_name: type_name.clone(),
                        name,
                        init: value,
                        access: Access::Default,
                        span: start,
                    };
                    while self.eat(&TokenKind::Comma).is_some() {
                        let extra = self.expect_identifier("as variable name")?;
                        let extra_init = if self.eat(&TokenKind::Assign).is_some() {
                            self.expression()?
                        } else {
                            Expr::Literal(Literal::Int(0))
                        };
                        extra_inits.push(Stmt::Let {
                            is_const: false,
                            is_byref: false,
                            type_name: type_name.clone(),
                            name: extra,
                            init: extra_init,
                            access: Access::Default,
                            span: start,
                        });
                    }
                    self.expect(&TokenKind::Semicolon, "after for-init")?;
                    Some(Box::new(first))
                }
                _ => {
                    self.pos = mark;
                    self.errors.truncate(err_mark);
                    let stmt = self.for_header_exprs();
                    self.expect(&TokenKind::Semicolon, "after for-init")?;
                    stmt.map(Box::new)
                }
            }
        } else {
            let stmt = self.for_header_exprs();
            self.expect(&TokenKind::Semicolon, "after for-init")?;
            stmt.map(Box::new)
        };

        // condition clause
        let cond = if self.peek().kind == TokenKind::Semicolon {
            None
        } else {
            Some(self.expression()?)
        };
        self.expect(&TokenKind::Semicolon, "after for-condition")?;

        // step clause: one or more comma-separated statements
        // (`for (...; ...; i++, j++)`), shared with multi-expression
        // inits above.
        let step: Option<Box<Stmt>> = if self.peek().kind == TokenKind::RightParen {
            None
        } else {
            self.for_header_exprs().map(Box::new)
        };
        self.expect(&TokenKind::RightParen, "to close for header")?;

        let body = self.control_body();
        let span = start.to(self.peek_at(0).span);
        if extra_inits.is_empty() {
            return Some(Stmt::For {
                init,
                cond,
                step,
                body,
                span,
            });
        }
        // Multi-declarator init (`for (int i = 0, n = len; ...)`): the
        // header scope owns only a single init, and the body must see
        // every name — hoist all declarators into a wrapping block
        // around a bare-init `for`.
        let mut stmts = Vec::with_capacity(extra_inits.len() + 2);
        if let Some(first) = init {
            stmts.push(*first);
        }
        stmts.append(&mut extra_inits);
        stmts.push(Stmt::For {
            init: None,
            cond,
            step,
            body,
            span,
        });
        Some(Stmt::Block {
            body: crate::models::Block { stmts, span },
            span,
        })
    }

    /// One statement inside a for-header slot: plain assignment, compound
    /// assignment, or increment/decrement — no trailing semicolon.
    fn for_clause_stmt(&mut self) -> Option<Stmt> {
        let start = self.peek().span;
        let mark = self.pos;
        let err_mark = self.errors.len();
        let pending_mark = self.pending.len();
        let pre_mark = self.pre_pending.len();
        // Prefix increment/decrement: `++i` / `--i` ≡ `i = i +/- 1`.
        if matches!(
            self.peek().kind,
            TokenKind::PlusPlus | TokenKind::MinusMinus
        ) {
            let up = self.peek().kind == TokenKind::PlusPlus;
            self.bump();
            let name = self.expect_identifier("as increment target")?;
            return Some(Stmt::Assign {
                name: name.clone(),
                value: Expr::Binary {
                    op: if up { BinOp::Add } else { BinOp::Sub },
                    lhs: Box::new(Expr::Ident { name, span: start }),
                    rhs: Box::new(Expr::Literal(Literal::Int(1))),
                    span: start,
                },
                span: start,
            });
        }
        let name = self.expect_identifier("as assignment target")?;
        if let Some((op, width)) = self.compound_assign_op_at(0) {
            self.pos += width;
            let rhs = self.expression()?;
            let lhs = Expr::Ident {
                name: name.clone(),
                span: start,
            };
            let value = match op {
                Some(binop) => Expr::Binary {
                    op: binop,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                    span: start,
                },
                None => rhs,
            };
            return Some(Stmt::Assign {
                name,
                value,
                span: start,
            });
        }
        if matches!(
            self.peek().kind,
            TokenKind::PlusPlus | TokenKind::MinusMinus
        ) {
            let up = self.peek().kind == TokenKind::PlusPlus;
            self.bump();
            return Some(Stmt::Assign {
                name: name.clone(),
                value: Expr::Binary {
                    op: if up { BinOp::Add } else { BinOp::Sub },
                    lhs: Box::new(Expr::Ident { name, span: start }),
                    rhs: Box::new(Expr::Literal(Literal::Int(1))),
                    span: start,
                },
                span: start,
            });
        }
        // General expression statements (`_bufferIndex.Value++`,
        // `M()`, `a[i] = x`): the bare-name shapes above miss member /
        // indexer / call targets. Rewind (dropping speculative errors
        // and queued bindings) and parse a full expression, accepted
        // only at a header delimiter (`,`/`;`/`)`) so garbage still
        // errors at the caller's expected-token checks.
        self.pos = mark;
        self.errors.truncate(err_mark);
        self.pending.truncate(pending_mark);
        self.pre_pending.truncate(pre_mark);
        if let Some(e) = self.expression()
            && matches!(
                self.peek().kind,
                TokenKind::Comma | TokenKind::Semicolon | TokenKind::RightParen
            )
        {
            return Some(Stmt::Expression(e));
        }
        self.pos = mark;
        self.errors.truncate(err_mark);
        self.pending.truncate(pending_mark);
        self.pre_pending.truncate(pre_mark);
        None
    }

    /// Whether a statement can never fall through to what follows it:
    /// jumps (`break`/`continue`/`return`/`goto`/`throw`), a block
    /// ending in one, an `if`/`else` whose branches both qualify, or an
    /// unconditional loop (`while (true)`, `for (;;)`) with no `break`
    /// that exits it. Conservative everywhere else.
    fn stmt_never_falls_through(stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Break { .. }
            | Stmt::Continue { .. }
            | Stmt::Return { .. }
            | Stmt::Goto { .. }
            | Stmt::Throw { .. } => true,
            // A label is just a jump target: execution falls through
            // to whatever follows it.
            Stmt::Label { .. } => false,
            Stmt::Switch {
                arms,
                default_body,
                ..
            } => {
                // A switch statement never falls through when every arm
                // (including `default`, which must exist — otherwise an
                // unmatched subject falls through) never falls through
                // (`case S.Clear: switch (t) { ... returns ... }` as an
                // outer arm body, System.Private.Xml).
                match default_body {
                    Some(default_b) => {
                        Self::arm_never_falls_through(default_b)
                            && arms
                                .iter()
                                .all(|a| Self::arm_never_falls_through(&a.body))
                    }
                    None => false,
                }
            }
            Stmt::Block { body, .. } => {
                Self::last_effective_stmt(body).is_some_and(Self::stmt_never_falls_through)
            }
            Stmt::Fixed { body, .. } => {
                Self::last_effective_stmt(body).is_some_and(Self::stmt_never_falls_through)
            }
            Stmt::If {
                then_block,
                else_block,
                ..
            } => match else_block {
                Some(else_b) => {
                    Self::last_effective_stmt(then_block)
                        .is_some_and(Self::stmt_never_falls_through)
                        && Self::last_effective_stmt(else_b)
                            .is_some_and(Self::stmt_never_falls_through)
                }
                None => false,
            },
            Stmt::While { cond, body, .. } => {
                Self::expr_is_const_true(cond) && !Self::block_has_exiting_break(body)
            }
            Stmt::For { cond, body, .. } => {
                cond.as_ref().is_none_or(Self::expr_is_const_true)
                    && !Self::block_has_exiting_break(body)
            }
            Stmt::Try {
                body,
                catches,
                finally_block,
                ..
            } => {
                // End-point reachability (C#): a try statement never
                // falls through when the try end and every catch end
                // are unreachable, or when its finally block itself
                // never falls through (e.g. `try{return;}finally{M();}`
                // always returns — the falling finally runs during
                // unwinding, not after).
                let abrupt = |b: &crate::models::Block| {
                    b.stmts
                        .last()
                        .is_some_and(Self::stmt_never_falls_through)
                };
                (abrupt(body) && catches.iter().all(|c| abrupt(&c.body)))
                    || finally_block.as_ref().is_some_and(|f| abrupt(f))
            }
            _ => false,
        }
    }

    /// Last statement that executes at a block's end point: trailing
    /// `Let`s never do. They are compile-time bindings only — `out`
    /// pre-declarations (`return F(a, out b);` leaves `[Return, Let]`)
    /// and local-function placeholders (`...; static Task F(){...}`).
    /// A block of nothing but Lets still falls through.
    fn last_effective_stmt(body: &crate::models::Block) -> Option<&Stmt> {
        body.stmts.iter().rev().find(|s| {
            !matches!(
                s,
                Stmt::Let { .. } | Stmt::LocalFunction { .. }
            )
        })
    }

    /// Whether a switch arm body can never fall through: its last
    /// *executable* statement qualifies per `stmt_never_falls_through`.
    fn arm_never_falls_through(body: &crate::models::Block) -> bool {
        Self::last_effective_stmt(body).is_some_and(Self::stmt_never_falls_through)
    }

    /// Constant-`true` condition: literal `true` (C# `while (true)`).
    fn expr_is_const_true(expr: &Expr) -> bool {
        matches!(expr, Expr::Literal(Literal::Bool(true)))
    }

    /// Whether a `break;` inside the block exits the ENCLOSING loop
    /// (not a nested loop or `switch`, whose `break` stays inside).
    fn block_has_exiting_break(block: &crate::models::Block) -> bool {
        block.stmts.iter().any(Self::stmt_has_exiting_break)
    }

    /// See `block_has_exiting_break`.
    fn stmt_has_exiting_break(stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Break { .. } => true,
            Stmt::Block { body, .. } => Self::block_has_exiting_break(body),
            Stmt::If {
                then_block,
                else_block,
                ..
            } => {
                Self::block_has_exiting_break(then_block)
                    || else_block
                        .as_ref()
                        .is_some_and(Self::block_has_exiting_break)
            }
            Stmt::Try {
                body,
                catches,
                finally_block,
                ..
            } => {
                Self::block_has_exiting_break(body)
                    || catches
                        .iter()
                        .any(|c| Self::block_has_exiting_break(&c.body))
                    || finally_block
                        .as_ref()
                        .is_some_and(Self::block_has_exiting_break)
            }
            // A `break` inside a nested loop or `switch` exits THAT
            // construct, never the enclosing loop.
            Stmt::While { .. }
            | Stmt::DoWhile { .. }
            | Stmt::For { .. }
            | Stmt::Switch { .. } => false,
            _ => false,
        }
    }

    /// Shared `using` handling for `using (...)` / `using Type ...` and
    /// `await using (...)`: consumes the leading `using` itself.
    /// (`await` is consumed by the caller.)
    fn using_statement(&mut self, start: Span) -> Option<Stmt> {
        self.bump(); // 'using'
        if self.peek().kind == TokenKind::LeftParen {
            self.bump(); // '('
            // `using (resource) { body }`: execute the body with the
            // resource bound (desugared to `Let`s + the body block).
            // Disposal itself is not modeled — same posture as the
            // `using Type name = expr;` declaration form below (fds
            // stay open; the kernel reclaims on exit. Flushing
            // disposables on Dispose is a known gap in both forms).
            // Previously the whole body was SKIPPED (returned as a
            // no-op `Break`), so `File.ReadAllBytes` — whose entire
            // implementation sits inside `using (...)` — lowered to
            // an empty body returning null. On any parse trouble,
            // rewind and take the legacy skip (never fail where the
            // old shape succeeded).
            let mark = self.pos;
            let err_mark = self.errors.len();
            let mut preamble: Vec<Stmt> = Vec::new();
            // Resources disposed when the body finishes, in the order
            // they were declared (disposal runs in reverse).
            let mut disposables: Vec<String> = Vec::new();
            let mut ok = true;
            if self.peek().kind == TokenKind::RightParen {
                self.bump();
            } else {
                loop {
                    // Trial declaration: `Type name =` (mirrors the
                    // `out`-var trial in argument lists — a member or
                    // index expression is a value, not a type).
                    let tmark = self.pos;
                    let terr_mark = self.errors.len();
                    let mut decl_ty: Option<String> = None;
                    if self.peek().kind == TokenKind::LeftParen {
                        // Tuple-typed resource (exotic): keep the
                        // trial simple — a bare expression fallback
                        // covers failures below.
                        if self.consume_tuple_type().is_some() {
                            decl_ty = Some(String::from("tuple"));
                        }
                    } else {
                        decl_ty = self.type_name_full();
                    }
                    let is_decl = decl_ty.is_some() && self.peek_is_name(0);
                    if !is_decl {
                        self.pos = tmark;
                        self.errors.truncate(terr_mark);
                    }
                    if is_decl {
                        let ty = decl_ty.unwrap_or_else(|| String::from("var"));
                        let Some(rname) = self.expect_identifier("as using resource name") else {
                            ok = false;
                            break;
                        };
                        // C# requires an initializer; without `=`
                        // take the legacy skip below.
                        if self.eat(&TokenKind::Assign).is_none() {
                            ok = false;
                            break;
                        }
                        let Some(init) = self.expression() else {
                            ok = false;
                            break;
                        };
                        if !self.pre_pending.is_empty() {
                            let pre = std::mem::take(&mut self.pre_pending);
                            preamble.extend(pre);
                        }
                        preamble.push(Stmt::Let {
                            is_const: false,
                            is_byref: false,
                            type_name: ty,
                            name: rname.clone(),
                            init,
                            access: Access::Default,
                            span: start.to(self.peek().span),
                        });
                        disposables.push(rname);
                        preamble.append(&mut self.pending);
                    } else {
                        let Some(e) = self.expression() else {
                            ok = false;
                            break;
                        };
                        if !self.pre_pending.is_empty() {
                            let pre = std::mem::take(&mut self.pre_pending);
                            preamble.extend(pre);
                        }
                        // `using (expr)`: the resource is disposed, so
                        // it needs a name. C# treats it as an anonymous
                        // local; the span offset keeps it unique.
                        let temp = format!("__using{}", start.start);
                        preamble.push(Stmt::Let {
                            is_const: false,
                            is_byref: false,
                            type_name: String::from("var"),
                            name: temp.clone(),
                            init: e,
                            access: Access::Default,
                            span: start.to(self.peek().span),
                        });
                        disposables.push(temp);
                        preamble.append(&mut self.pending);
                    }
                    if self.eat(&TokenKind::Comma).is_none() {
                        break;
                    }
                }
                if ok {
                    if self.expect(&TokenKind::RightParen, "to close using resources").is_none() {
                        ok = false;
                    }
                }
            }
            if !ok {
                // Legacy skip: resource text + body, no-op Break.
                self.pos = mark;
                self.errors.truncate(err_mark);
                let mut depth = 0i32;
                loop {
                    if self.peek().kind == TokenKind::LeftParen {
                        depth += 1;
                    } else if self.peek().kind == TokenKind::RightParen {
                        depth -= 1;
                    }
                    self.bump();
                    if depth <= 0 || self.peek().kind == TokenKind::Eof {
                        break;
                    }
                }
                let _ = self.control_body();
                return Some(Stmt::Break {
                    span: start.to(self.peek_at(0).span),
                });
            }
            let body = self.control_body();
            let end = self.peek().span;
            let span = start.to(end);
            let mut body_stmts = body.stmts;
            if !self.pre_pending.is_empty() {
                body_stmts.extend(std::mem::take(&mut self.pre_pending));
            }
            body_stmts.append(&mut self.pending);
            // `using (R r = e) { body }` is `{ let r = e; try { body }
            // } finally { r.Dispose(); } }`, with the resources disposed
            // in reverse order. The finally also runs on `return` and on
            // a throw (the try lowering routes both through it).
            let mut wrapped = Stmt::Block {
                body: Block {
                    stmts: body_stmts,
                    span,
                },
                span,
            };
            for name in disposables.into_iter().rev() {
                let dispose = Expr::MethodCall {
                    object: Box::new(Expr::Ident {
                        name: name.clone(),
                        span,
                    }),
                    name: String::from("Dispose"),
                    args: Vec::new(),
                    span,
                };
                wrapped = Stmt::Try {
                    body: Block {
                        stmts: vec![wrapped],
                        span,
                    },
                    catches: Vec::new(),
                    finally_block: Some(Block {
                        stmts: vec![Stmt::Expression(dispose)],
                        span,
                    }),
                    span,
                };
            }
            let mut stmts = preamble;
            stmts.push(wrapped);
            return Some(Stmt::Block {
                body: Block { stmts, span },
                span,
            });
        }
        // `using Type name = expr;` (declaration). `block()` rewrites
        // this into a `Let` plus a try/finally that disposes at the end
        // of the enclosing block; only a block may contain it (C# does
        // not allow a using declaration as an embedded statement).
        let type_name = self.type_name().unwrap_or_else(|| "var".to_owned());
        let name = self.expect_identifier("as using-decl name")?;
        let init = if self.eat(&TokenKind::Assign).is_some() {
            self.expression()?
        } else {
            Expr::Literal(Literal::Int(0))
        };
        let end = self
            .expect(&TokenKind::Semicolon, "after using declaration")
            .map_or(self.peek().span, |token| token.span);
        Some(Stmt::UsingDecl {
            type_name,
            name,
            init,
            span: start.to(end),
        })
    }

    /// `switch (subject) { case N: … break; default: … }`
    ///
    /// C#-style: every arm body must end in `break;` (no fallthrough).
    /// Labels are constant integer expressions for now; the AST is shaped so
    /// A11 can widen labels into full patterns.
    fn switch_statement(&mut self) -> Option<Stmt> {
        let start = self.peek().span;
        self.bump(); // 'switch'
        self.expect(&TokenKind::LeftParen, "to open switch subject")?;
        let subject = self.expression()?;
        self.expect(&TokenKind::RightParen, "to close switch subject")?;
        // Bindings queued by the SUBJECT (e.g. `out` vars, `is`
        // patterns) belong to the enclosing block, not to the first
        // arm body (whose inline drain would otherwise steal them).
        // Stash them across arm parsing; restore on exit so the
        // outer `block()` drains them after the statement.
        let stashed_pending = std::mem::take(&mut self.pending);
        let stashed_pre = std::mem::take(&mut self.pre_pending);
        self.expect(&TokenKind::LeftBrace, "to open switch body")?;

        let mut arms: Vec<SwitchArm> = Vec::new();
        let mut default_body: Option<Block> = None;

        while self.peek().kind != TokenKind::RightBrace && self.peek().kind != TokenKind::Eof {
            // Collect one arm's labels: `case A: case B:` … or `default:`.
            let mut labels: Vec<Expr> = Vec::new();
            let mut guards: Vec<Option<Expr>> = Vec::new();
            let mut arm_binding: Option<(String, String)> = None;
            let mut is_default_arm = false;
            loop {
                match self.peek().kind {
                    TokenKind::Keyword(Keyword::Case) => {
                        self.bump();
                        // Parse the case label. Handle relational patterns like `>= 10`,
                        // `and`/`or` patterns, and `when` clauses.
                        let label = self.case_label()?;
                        // `case X when cond:` — the guard is part of the
                        // arm's test, evaluated only after the label
                        // matched.
                        let guard = if matches!(
                            self.peek().kind,
                            TokenKind::Keyword(Keyword::When_)
                        ) {
                            self.bump();
                            self.expression()
                        } else {
                            None
                        };
                        self.expect(&TokenKind::Colon, "after case label")?;
                        if let Some(b) = self.case_binding.take() {
                            // One binding per section (C# requires every
                            // label in a section to bind the same names).
                            arm_binding = Some(b);
                        }
                        labels.push(label);
                        guards.push(guard);
                    }
                    TokenKind::Keyword(Keyword::Default) => {
                        self.bump();
                        self.expect(&TokenKind::Colon, "after 'default'")?;
                        if default_body.is_some() {
                            let span = self.peek().span;
                            self.error("E0001", "duplicate 'default' arm".to_owned(), span);
                        }
                        is_default_arm = true;
                        // No break: `default:` stacks with further labels
                        // (`default: case X:` shares one body, per C#).
                        // A non-label token below ends collection.
                    }
                    _ => break, // body starts here
                }
            }

            // Arm body runs until the next case/default/closing brace.
            let body_start = self.peek().span;
            let mut stmts: Vec<Stmt> = Vec::new();
            while !matches!(
                self.peek().kind,
                TokenKind::Keyword(Keyword::Case)
                    | TokenKind::Keyword(Keyword::Default)
                    | TokenKind::RightBrace
                    | TokenKind::Eof
            ) {
                let stmt_start = stmts.len();
                let Some(stmt) = self.statement() else {
                    self.synchronize();
                    continue;
                };
                if !self.pre_pending.is_empty() {
                    let pre = std::mem::take(&mut self.pre_pending);
                    stmts.splice(stmt_start..stmt_start, pre);
                }
                stmts.push(stmt);
                stmts.append(&mut self.pending);
            }
            let body_end = self.peek_at(0).span;
            let body = Block {
                stmts,
                span: body_start.to(body_end),
            };

            // No-fallthrough rule (C#): the arm's end point must be
            // unreachable — a trailing jump (`break;`, `continue;`,
            // `return …;`, `goto …;`, `throw …;`), a block ending in
            // one, or an if/else whose branches both qualify.
            // Trailing local-function declarations are ignored (they
            // never execute at the arm's end point).
            let ends_with_break = Self::arm_never_falls_through(&body);
            if !ends_with_break {
                let span = body_end;
                self.error(
                    "E0001",
                    "switch arm must end with 'break;' (fallthrough is not allowed)".to_owned(),
                    span,
                );
            }

            if is_default_arm {
                if labels.is_empty() {
                    default_body = Some(body);
                }
            } else {
                arms.push(SwitchArm { labels, guards, binding: arm_binding, body });
            }
        }

        self.expect(&TokenKind::RightBrace, "to close switch body")?;
        // Restore the subject's queued bindings ahead of any leftover
        // from a failed trailing arm (order is irrelevant to the scan,
        // which pre-registers block Lets as a set).
        let mut rest_pending = std::mem::take(&mut self.pending);
        self.pending = stashed_pending;
        self.pending.append(&mut rest_pending);
        let mut rest_pre = std::mem::take(&mut self.pre_pending);
        self.pre_pending = stashed_pre;
        self.pre_pending.append(&mut rest_pre);
        Some(Stmt::Switch {
            subject,
            arms,
            default_body,
            span: start.to(self.peek_at(0).span),
        })
    }

    /// Parse a case label. Handles the full C# 11 pattern grammar
    /// permissively: relational, type, var, property, positional, list, and
    /// constant patterns; `and`/`or` conjunctions/disjunctions; and `when`
    /// clauses. The structural detail of each pattern is dropped — the AST
    /// just needs to be valid and round-trip through bytecode.
    fn case_label(&mut self) -> Option<Expr> {
        let start = self.peek().span;
        // Optional `not` for negated patterns.
        let negated = matches!(self.peek().kind, TokenKind::Keyword(Keyword::Not_));
        if negated {
            self.bump();
        }
        let primary = self.parse_pattern_primary()?;
        // Handle `and`/`or` conjunctions/disjunctions.
        let mut label = if negated {
            Expr::Unary {
                op: UnOp::Not,
                inner: Box::new(primary),
                span: start.to(self.peek_at(0).span),
            }
        } else {
            primary
        };
        loop {
            match self.peek().kind {
                TokenKind::Keyword(Keyword::And_) => {
                    self.bump();
                    let rhs = self.parse_pattern_primary()?;
                    let span = expr_span(&label).to(expr_span(&rhs));
                    label = Expr::Binary {
                        op: BinOp::And,
                        lhs: Box::new(label),
                        rhs: Box::new(rhs),
                        span,
                    };
                }
                TokenKind::Keyword(Keyword::Or_) => {
                    self.bump();
                    let rhs = self.parse_pattern_primary()?;
                    let span = expr_span(&label).to(expr_span(&rhs));
                    label = Expr::Binary {
                        op: BinOp::Or,
                        lhs: Box::new(label),
                        rhs: Box::new(rhs),
                        span,
                    };
                }
                _ => break,
            }
        }
        // A `when` guard is NOT consumed here: it belongs to the arm
        // (evaluated only after the label matched), and the switch loop
        // records it per label. This used to swallow the expression, so
        // `case 1 when false:` matched anyway.
        Some(label)
    }

    /// Parse a single primary pattern: relational, type, var, property,
    /// positional, list, or constant. Any unrecognized shape is skipped as a
    /// constant placeholder so the surrounding `case` parsing keeps going.
    /// True if the parenthesized group ahead holds a top-level comma
    /// (`(a, b)` is a positional pattern; `(T)` may be a cast).
    /// Shared with the expression-level `is`-pattern parser, which
    /// needs the identical cast-vs-pattern disambiguation.
    pub(super) fn paren_holds_comma(&self) -> bool {
        let mut depth = 0usize;
        let mut i = 0usize;
        loop {
            match self.peek_at(i).kind {
                TokenKind::Eof => return false,
                TokenKind::LeftParen => depth += 1,
                TokenKind::RightParen => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        return false;
                    }
                }
                TokenKind::Comma if depth == 1 => return true,
                _ => {}
            }
            i += 1;
        }
    }

    /// Try a `Type... name` binding at the cursor (tuple-pattern
    /// element): dotted qualification, generic arguments, array ranks,
    /// and nullability are skipped over; the trailing identifier (unless
    /// `_`) is queued as a `Let` with the head type (mirroring
    /// top-level type patterns, where qualified/generic heads degrade
    /// through the normal unknown-type paths). Consumes nothing unless
    /// the full shape matches.
    fn try_tuple_element_binding(&mut self, start: crate::models::Span) -> bool {
        let head = match &self.peek().kind {
            TokenKind::Identifier(t) => t.clone(),
            _ => return false,
        };
        let mut offset = 1usize;
        // Dotted segments (`Ns.Type`).
        while self.peek_at(offset).kind == TokenKind::Dot {
            offset += 1;
            if !matches!(self.peek_at(offset).kind, TokenKind::Identifier(_)) {
                return false;
            }
            offset += 1;
        }
        // Generic arguments (`D<T>`, nested).
        if self.peek_at(offset).kind == TokenKind::Less {
            let mut depth = 0i32;
            loop {
                match &self.peek_at(offset).kind {
                    TokenKind::Less => depth += 1,
                    TokenKind::Greater => depth -= 1,
                    TokenKind::Shr => depth -= 2,
                    TokenKind::ShrShr => depth -= 3,
                    TokenKind::Eof => return false,
                    _ => {}
                }
                offset += 1;
                if depth <= 0 {
                    break;
                }
            }
        }
        // Array ranks (`T[]`, `T[,]`).
        while self.peek_at(offset).kind == TokenKind::LeftBracket {
            offset += 1;
            while matches!(
                self.peek_at(offset).kind,
                TokenKind::Comma | TokenKind::RightBracket
            ) {
                // `[]` or `[,,]`: commas and the closer belong here.
                if self.peek_at(offset).kind == TokenKind::RightBracket {
                    offset += 1;
                    break;
                }
                offset += 1;
            }
        }
        // Nullability (`T?`).
        if self.peek_at(offset).kind == TokenKind::Question {
            offset += 1;
        }
        let (binding, name_span) = match &self.peek_at(offset).kind {
            TokenKind::Identifier(b) => (b.clone(), self.peek_at(offset).span),
            _ => return false,
        };
        for _ in 0..=offset {
            self.bump();
        }
        if binding != "_" {
            self.pre_pending.push(Stmt::Let {
                is_const: false,
                is_byref: false,
                type_name: head,
                name: binding,
                init: Expr::Literal(Literal::Int(0)),
                access: Access::Default,
                span: start.to(name_span),
            });
        }
        true
    }

    fn parse_pattern_primary(&mut self) -> Option<Expr> {
        let start = self.peek().span;
        // Discard / var pattern: `_`, `var x`, `var (x)`, `var _`.
        if matches!(self.peek().kind, TokenKind::Identifier(_))
            && let TokenKind::Identifier(name) = &self.peek().kind
            && (name == "_" || name == "var")
        {
            let name = name.clone();
            self.bump();
            // `var x` or `var _` — consume the optional binding identifier.
            // The binding lives in the enclosing scope (like `is`-pattern
            // bindings); `_` discards need no declaration. The name may
            // be a contextual keyword (`case var record:`).
            if name == "var" {
                // Non-names (`and`, `(`, ...) stay for the caller.
                if let Some((binding, span)) = self.take_pattern_binding() {
                    if binding != "_" {
                        self.pre_pending.push(Stmt::Let {
                            is_const: false,
                            is_byref: false,
                            type_name: String::from("var"),
                            name: binding,
                            init: Expr::Literal(Literal::Int(0)),
                            access: Access::Default,
                            span: start.to(span),
                        });
                    }
                }
            }
            return Some(Expr::Ident {
                name,
                span: start.to(self.peek_at(0).span),
            });
        }
        // Relational pattern: `<rel> expr` — same code as before.
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
            // (`case > -X - 1:`), mirroring the expression-level
            // `is`-pattern handling. Precedence 6 admits shifts and
            // up, excluding comparisons, booleans, and `is` itself.
            let rhs = self.binary_expr(6)?;
            let span = start.to(expr_span_last(&self.tokens[..self.pos]));
            return Some(Expr::Binary {
                op,
                lhs: Box::new(Expr::Literal(Literal::Int(0))),
                rhs: Box::new(rhs),
                span,
            });
        }
        // Cast in label position: `(byte)'l'` — parens holding a single
        // type (no top-level comma) followed by an operand are a cast,
        // not a positional pattern.
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
        // Property / positional / list patterns share bracket handling.
        if self.peek().kind == TokenKind::LeftBrace
            || self.peek().kind == TokenKind::LeftParen
            || self.peek().kind == TokenKind::LeftBracket
        {
            // Tuple patterns `(p1, p2, ...)` bind `Type name` elements
            // (`case (VT, sbyte s):` binds `s: sbyte`); other bracket
            // shapes keep the legacy skip (positional/list patterns
            // bind after the bracket, if at all).
            if self.peek().kind == TokenKind::LeftParen {
                self.bump(); // '('
                loop {
                    if self.peek().kind == TokenKind::RightParen {
                        self.bump();
                        break;
                    }
                    if self.peek().kind == TokenKind::Eof {
                        break;
                    }
                    if self.peek().kind == TokenKind::Comma {
                        self.bump();
                        continue;
                    }
                    // Nested section: skip whole (bindings inside
                    // `or`-operands or sub-patterns are future work).
                    if matches!(
                        self.peek().kind,
                        TokenKind::LeftBrace
                            | TokenKind::LeftParen
                            | TokenKind::LeftBracket
                    ) {
                        self.skip_balanced_bracket();
                        continue;
                    }
                    if self.try_tuple_element_binding(start) {
                        continue;
                    }
                    // Anything else (constants, member access, `or`/`and`,
                    // discards): single token, no binding.
                    self.bump();
                }
            } else {
                self.skip_balanced_bracket();
            }
            // Optional variable binding after the closing bracket.
            if matches!(self.peek().kind, TokenKind::Identifier(_)) {
                self.bump();
            }
            return Some(Expr::Literal(Literal::Int(0)));
        }
        // Type pattern: `Type name` or `Type? name` (nullable type), or just
        // `Type` (with a leading optional `?`). Heuristic: an identifier
        // followed by another identifier, a `?`, a `<` (generic), a `[`
        // (array), a `.` (qualified), or a `(` (positional) is a type.
        if matches!(self.peek().kind, TokenKind::Identifier(_)) {
            let mut offset = 1;
            let mut saw_generic = false;
            while self.peek_at(offset).kind == TokenKind::Dot {
                offset += 1;
                if !matches!(self.peek_at(offset).kind, TokenKind::Identifier(_)) {
                    break;
                }
                offset += 1;
            }
            if self.peek_at(offset).kind == TokenKind::Less {
                saw_generic = true;
                let mut depth = 0usize;
                loop {
                    match &self.peek_at(offset).kind {
                        TokenKind::Less => depth += 1,
                        TokenKind::Greater => {
                            depth -= 1;
                            if depth == 0 {
                                offset += 1;
                                break;
                            }
                        }
                        TokenKind::Eof => break,
                        _ => {}
                    }
                    offset += 1;
                }
            }
            while self.peek_at(offset).kind == TokenKind::LeftBracket
                && self.peek_at(offset + 1).kind == TokenKind::RightBracket
            {
                offset += 2;
            }
            if self.peek_at(offset).kind == TokenKind::Question {
                offset += 1;
            }
            // Positional pattern: `Type(...)` — consume params and the
            // optional binding identifier (declared in the enclosing
            // scope; `_` needs nothing).
            if self.peek_at(offset).kind == TokenKind::LeftParen {
                for _ in 0..offset {
                    self.bump();
                }
                let open = self.pos;
                self.skip_balanced_bracket();
                // Inner designations (`(int x, ...)`) bind too.
                self.queue_inner_designations(open);
                // Optional trailing identifier (var binding).
                // The binding may be a contextual keyword (`case T record:`).
                if let Some((binding, span)) = self.take_pattern_binding() {
                    if binding != "_" {
                        self.pre_pending.push(Stmt::Let {
                            is_const: false,
                            is_byref: false,
                            type_name: String::from("var"),
                            name: binding,
                            init: Expr::Literal(Literal::Int(0)),
                            access: Access::Default,
                            span: start.to(span),
                        });
                    }
                }
                return Some(Expr::Literal(Literal::Int(0)));
            }
            // Property pattern: `Type { ... }` or `Type { ... } var`.
            if self.peek_at(offset).kind == TokenKind::LeftBrace {
                for _ in 0..offset {
                    self.bump();
                }
                let open = self.pos;
                self.skip_balanced_bracket();
                // Inner designations (`{ P: T x }`) bind too.
                self.queue_inner_designations(open);
                // The binding may be a contextual keyword (`case T record:`).
                if let Some((binding, span)) = self.take_pattern_binding() {
                    if binding != "_" {
                        self.pre_pending.push(Stmt::Let {
                            is_const: false,
                            is_byref: false,
                            type_name: String::from("var"),
                            name: binding,
                            init: Expr::Literal(Literal::Int(0)),
                            access: Access::Default,
                            span: start.to(span),
                        });
                    }
                }
                return Some(Expr::Literal(Literal::Int(0)));
            }
            // If the next token is a binding identifier, it's a type pattern.
            // The binding is declared with the pattern's head type
            // (`case TC tc` binds `tc: TC`, not `var`) so member calls
            // on it resolve; qualified/generic heads degrade through
            // the normal unknown-type paths. The binding may be a
            // contextual keyword (`case T record:`).
            if self.peek_is_pattern_binding(offset) {
                let head_ty = match self.peek().kind.clone() {
                    TokenKind::Identifier(t) => t,
                    _ => String::from("var"),
                };
                for _ in 0..offset {
                    self.bump();
                }
                // A type pattern selects the arm by RUNTIME TYPE. With a
                // BINDING (`case B b:`) the matched value must also be
                // assigned to the arm variable, which the pattern parser
                // cannot do (it does not see the switch subject), so that
                // form is still reported. WITHOUT a binding (`case B:`)
                // the arm test is exactly `subject is B`, which the
                // lowering implements, so it parses as the type test.
                let has_binding = self.peek_is_pattern_binding(0);
                if has_binding {
                    // `case T x:` — record the designation for the switch
                    // lowering, which has the subject and binds it there.
                    if let Some((binding, span)) = self.take_pattern_binding() {
                        if binding != "_" {
                            self.case_binding = Some((binding.clone(), head_ty.clone()));
                        }
                        let _ = span;
                    }
                }
                return Some(Expr::Ident {
                    name: head_ty,
                    span: start.to(self.peek_at(0).span),
                });
            }
            // Bare generic type pattern: `case GenericEqualityComparer<T>:`.
            // A trailing `:` after a generic list can only close a type
            // pattern — a constant or comparison never looks like this
            // (comparisons need a right operand; dotted constants like
            // `X.Y:` have no generics and keep the constant path below).
            if saw_generic && matches!(self.peek_at(offset).kind, TokenKind::Colon) {
                // `case GenericEqualityComparer<string>:` — a GENERIC
                // type pattern without a binding. The spelling (with its
                // arguments) is what the type test resolves.
                let spelling = self.render_type_tokens(offset);
                for _ in 0..offset {
                    self.bump();
                }
                return Some(Expr::Ident {
                    name: spelling,
                    span: start.to(self.peek_at(0).span),
                });
            }
        }
        // Fallback: a constant pattern (literal, identifier, string, etc.).
        self.expression()
    }
}

impl Parser {
    /// Checks if the current position starts a type declaration:
    /// `Type name = ...` / `Type name;` where Type may have array,
    /// nullable, or generic suffixes.
    fn likely_type_decl(&self) -> bool {
        // Optional `ref` / `out` / `scoped` modifiers on the local
        // declaration (`scoped` combines: `scoped ref`, `ref scoped`;
        // `ref readonly` / `out readonly` are C# 12 ref-readonly locals).
        let mut off = 0;
        for _ in 0..4 {
            if matches!(
                self.peek_at(off).kind,
                TokenKind::Keyword(Keyword::Ref_)
                    | TokenKind::Keyword(Keyword::Out_)
                    | TokenKind::Keyword(Keyword::Scoped)
                    | TokenKind::Keyword(Keyword::Readonly)
            ) {
                off += 1;
            } else {
                break;
            }
        }
        if !matches!(self.peek_at(off).kind, TokenKind::Identifier(_)) {
            return false;
        }
        off += 1;
        // C# global/alias namespace: `global::Ns`.
        if self.peek_at(off).kind == TokenKind::Colon
            && self.peek_at(off + 1).kind == TokenKind::Colon
        {
            off += 2;
            if !matches!(self.peek_at(off).kind, TokenKind::Identifier(_)) {
                return false;
            }
            off += 1;
        }
        // Dotted segments, each of which may carry its own generic list:
        // `Outer<int>.Inner<long>.Deep<...>`. The first segment may also
        // carry a generic list before the first dot.
        if matches!(
            self.peek_at(off).kind,
            TokenKind::Less | TokenKind::Shl
        ) {
            let mut depth = 0i32;
            loop {
                match self.peek_at(off).kind {
                    TokenKind::Less => depth += 1,
                    TokenKind::Shl => depth += 2,
                    TokenKind::Greater => {
                        depth -= 1;
                        if depth == 0 {
                            off += 1;
                            break;
                        }
                    }
                    // `>>`/`>>>` lex as Shr/ShrShr (nested closers).
                    TokenKind::Shr => {
                        depth = depth.saturating_sub(2);
                        if depth == 0 {
                            off += 1;
                            break;
                        }
                    }
                    TokenKind::ShrShr => {
                        depth = depth.saturating_sub(3);
                        if depth == 0 {
                            off += 1;
                            break;
                        }
                    }
                    TokenKind::Eof => return false,
                    _ => {}
                }
                off += 1;
            }
        }
        while self.peek_at(off).kind == TokenKind::Dot {
            off += 1;
            if !matches!(self.peek_at(off).kind, TokenKind::Identifier(_)) {
                return false;
            }
            off += 1;
            if matches!(
                self.peek_at(off).kind,
                TokenKind::Less | TokenKind::Shl
            ) {
                let mut depth = 0i32;
                loop {
                    match self.peek_at(off).kind {
                        TokenKind::Less => depth += 1,
                        TokenKind::Shl => depth += 2,
                        TokenKind::Greater => {
                            depth -= 1;
                            if depth == 0 {
                                off += 1;
                                break;
                            }
                        }
                        // `>>`/`>>>` lex as Shr/ShrShr (nested closers).
                        TokenKind::Shr => {
                            depth = depth.saturating_sub(2);
                            if depth == 0 {
                                off += 1;
                                break;
                            }
                        }
                        TokenKind::ShrShr => {
                            depth = depth.saturating_sub(3);
                            if depth == 0 {
                                off += 1;
                                break;
                            }
                        }
                        TokenKind::Eof => return false,
                        _ => {}
                    }
                    off += 1;
                }
            }
        }
        // Array: `T[]`, `T[][]`, `T[,]`, etc. Nullable and array suffixes
        // can interleave: `T?[]`, `T[]?`, `T?[]?`.
        loop {
            let mut progressed = false;
            if self.peek_at(off).kind == TokenKind::Question {
                off += 1;
                progressed = true;
            }
            // Pointer suffix: `int*`, `int**`, etc.
            while self.peek_at(off).kind == TokenKind::Star {
                off += 1;
                progressed = true;
            }
            while self.peek_at(off).kind == TokenKind::LeftBracket {
                off += 1; // '['
                if self.peek_at(off).kind == TokenKind::Comma {
                    // Multi-dimensional: `[,]`
                    loop {
                        off += 1; // ','
                        if self.peek_at(off).kind == TokenKind::RightBracket {
                            break;
                        }
                    }
                    off += 1; // ']'
                } else if self.peek_at(off).kind == TokenKind::RightBracket {
                    off += 1; // ']'
                } else {
                    // `[n]` — an INDEXER, not a rank specifier: the
                    // "type" is really a value (`(t.C[o].X)` grouped
                    // member chain), so this is not a declaration.
                    return false;
                }
                progressed = true;
            }
            if !progressed {
                break;
            }
        }
        // The next token must be a name (the variable name; contextual
        // keywords qualify).
        self.peek_is_name(off)
    }
}

/// Rewrite `using Type name = expr;` declarations into the try/finally C#
/// compiles them to.
///
/// A declaration disposes at the END OF THE ENCLOSING BLOCK, so the tail
/// after it moves inside `try { … } finally { name.Dispose(); }`, with the
/// `Let` outside it (the finally must be able to read the name). Several
/// declarations nest from the last to the first, which disposes them in
/// reverse order — the same rule the parenthesized form follows.
fn wrap_using_decls(mut stmts: Vec<Stmt>, span: crate::models::Span) -> Vec<Stmt> {
    loop {
        let Some(idx) = stmts
            .iter()
            .rposition(|s| matches!(s, Stmt::UsingDecl { .. }))
        else {
            return stmts;
        };
        let Stmt::UsingDecl {
            type_name,
            name,
            init,
            span: decl_span,
        } = stmts[idx].clone()
        else {
            unreachable!()
        };
        let tail: Vec<Stmt> = stmts.split_off(idx + 1);
        stmts.pop(); // the declaration itself
        let dispose = Expr::MethodCall {
            object: Box::new(Expr::Ident {
                name: name.clone(),
                span: decl_span,
            }),
            name: String::from("Dispose"),
            args: Vec::new(),
            span: decl_span,
        };
        let try_stmt = Stmt::Try {
            body: Block {
                stmts: tail,
                span,
            },
            catches: Vec::new(),
            finally_block: Some(Block {
                stmts: vec![Stmt::Expression(dispose)],
                span,
            }),
            span,
        };
        let let_stmt = Stmt::Let {
            is_const: false,
            is_byref: false,
            type_name,
            name,
            init,
            access: Access::Default,
            span: decl_span,
        };
        stmts.push(Stmt::Block {
            body: Block {
                stmts: vec![let_stmt, try_stmt],
                span,
            },
            span,
        });
    }
}
