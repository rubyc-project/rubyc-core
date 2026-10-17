//! Recursive-descent + precedence-climbing parser: tokens → CompilationUnit.
//!
//! Split across sibling files by grammar area; the [`Parser`] state, token
//! cursor primitives, and shared helpers live here.

use crate::models::{
    BinOp, CompilationUnit, Diagnostic, Expr, Keyword, ScopeContext, Severity, Span,
    SyntaxError, Token, TokenKind,
};
use crate::parser::lexer::Lexer;
pub(super) mod declaration;
pub(super) mod expression;
pub(super) mod statement;
pub(super) mod unit;

pub fn parse(source: &str) -> Result<CompilationUnit, Vec<SyntaxError>> {
    let (unit, errors) = parse_lenient(source);
    if errors.is_empty() {
        Ok(unit)
    } else {
        Err(errors)
    }
}

/// Parse returning the (recovery-patched) unit EVEN WHEN syntax errors
/// exist. Callers that want best-effort downstream diagnostics use this;
/// [`parse`] keeps its all-or-nothing contract.
pub fn parse_lenient(source: &str) -> (CompilationUnit, Vec<SyntaxError>) {
    let lexed = Lexer::new(source).tokenize_with_diags();
    let mut parser = Parser {
        tokens: lexed.tokens,
        pos: 0,
        errors: lexed.diags,
        context: ScopeContext::default(),
        source: source.as_bytes().to_vec(),
        pending: Vec::new(),
        pending_members: Vec::new(),
        pre_pending: Vec::new(),
        in_base_init: false,
        enum_values: Vec::new(),
        aliases: Vec::new(),
        static_usings: Vec::new(),
        foreach_seq: 0,
    };
    let unit = parser.compilation_unit();
    (unit, std::mem::take(&mut parser.errors))
}

pub(super) struct Parser {
    pub(super) tokens: Vec<Token>,
    pub(super) pos: usize,
    pub(super) errors: Vec<SyntaxError>,
    /// Raw source bytes: lets diagnostics point at insertion positions
    /// (e.g. a missing ';' at the end of the PREVIOUS line).
    pub(super) source: Vec<u8>,
    /// Enclosing namespace :: class :: function for error context.
    pub(super) context: ScopeContext,
    /// Monotonic counter for hygienic `foreach` desugar temporaries
    /// (`__foreach_coll_N`, `__foreach_idx_N`).
    pub(super) foreach_seq: u32,
    /// Trailing declarators from a multi-declarator statement
    /// (`int a = 1, b = 2;`) that the compact `Let` AST cannot carry in a
    /// single node. Drained by `block()` so each becomes its own `Stmt::Let`.
    pub(super) pending: Vec<crate::models::Stmt>,
    /// Inside `: base(...)` / `: this(...)` argument lists: `out`
    /// variables declared there lower after the chained call (their
    /// `Let`s splice at the body front), so their slots must stay
    /// the old nameless default (box without writeback) — an `Ident`
    /// would fail as undeclared at the call.
    pub(super) in_base_init: bool,
    /// Trailing declarators from a multi-declarator field
    /// (`string? a, b, c;`) that one `FieldDecl` cannot carry.
    /// Drained by the class body loop so each becomes its own member.
    pub(super) pending_members: Vec<crate::models::ClassMember>,
    /// Bindings that must be declared BEFORE the statement that uses them
    /// (e.g. an `is`-pattern binding var used in the same expression).
    /// Drained before each statement in `block()`.
    pub(super) pre_pending: Vec<crate::models::Stmt>,
    /// Resolved `(enum, variant, value)` triples from enums already parsed
    /// in this file. Lets later enum initializers reference them
    /// (`B = OtherEnum.X`) exactly; references outside the file still
    /// degrade to auto-numbering.
    pub(super) enum_values: Vec<(String, String, i64)>,
    /// `using Alias = Qualified.Name;` directives in source order.
    /// Moved into the unit by `compilation_unit()`; later passes
    /// resolve the names (scan scope, type table).
    pub(super) aliases: Vec<crate::models::UsingAlias>,
    /// `using static Qualified.Type;` targets in source order. Moved
    /// into the unit by `compilation_unit()`; the scan imports the
    /// target types' members into scope.
    pub(super) static_usings: Vec<String>,
}

/// Binding powers for binary operators; higher binds tighter.
pub(super) fn binop_precedence(kind: &TokenKind) -> Option<(u8, BinOp)> {
    match kind {
        TokenKind::PipePipe => Some((0, BinOp::Or)),
        TokenKind::AmpAmp => Some((1, BinOp::And)),
        TokenKind::Pipe => Some((2, BinOp::BitOr)),
        TokenKind::Hat => Some((3, BinOp::BitXor)),
        TokenKind::Amp => Some((4, BinOp::BitAnd)),
        TokenKind::EqEq => Some((5, BinOp::Eq)),
        TokenKind::NotEq => Some((5, BinOp::Ne)),
        TokenKind::Less => Some((5, BinOp::Lt)),
        TokenKind::Greater => Some((5, BinOp::Gt)),
        TokenKind::LessEq => Some((5, BinOp::Le)),
        TokenKind::GreaterEq => Some((5, BinOp::Ge)),
        TokenKind::Shl => Some((6, BinOp::Shl)),
        TokenKind::Shr => Some((6, BinOp::Shr)),
        TokenKind::ShrShr => Some((6, BinOp::ShrShr)),
        TokenKind::QuestionQuestion => Some((0, BinOp::NullCoalesce)),
        TokenKind::Plus => Some((7, BinOp::Add)),
        TokenKind::Minus => Some((7, BinOp::Sub)),
        TokenKind::Star => Some((8, BinOp::Mul)),
        TokenKind::Slash => Some((8, BinOp::Div)),
        TokenKind::Percent => Some((8, BinOp::Rem)),
        _ => None,
    }
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            pos: 0,
            errors: Vec::new(),
            context: ScopeContext::default(),
            source: Vec::new(),
            pending: Vec::new(),
            pending_members: Vec::new(),
            pre_pending: Vec::new(),
        in_base_init: false,
            enum_values: Vec::new(),
            aliases: Vec::new(),
            static_usings: Vec::new(),
        foreach_seq: 0,
        }
    }

    pub(super) fn peek(&self) -> &Token {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    /// Look at the token `offset` positions ahead of the cursor.
    pub(super) fn peek_at(&self, offset: usize) -> &Token {
        self.tokens
            .get(self.pos + offset)
            .unwrap_or_else(|| self.tokens.last().expect("stream ends with Eof"))
    }

    /// True if the current token can start a type name (identifier).
    pub(super) fn is_type_start(&self) -> bool {
        matches!(self.peek().kind, TokenKind::Identifier(_))
    }

    /// True if the token at `offset` can serve as a declared NAME:
    /// an identifier or a C# contextual keyword (`from`, `file`,
    /// `required`, `scoped`, `record`, `async`, `await`, `yield`,
    /// `with`, `and`, `or`, `not`, `when`). Lookahead positions must
    /// use this everywhere `expect_identifier` (which accepts the same
    /// set) will consume.
    pub(super) fn peek_is_name(&self, offset: usize) -> bool {
        match &self.peek_at(offset).kind {
            TokenKind::Identifier(_) => true,
            TokenKind::Keyword(
                kw @ (Keyword::File
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
                | Keyword::When_),
            ) => {
                let _ = kw;
                true
            }
            _ => false,
        }
    }

    /// Consume a pattern-binding name: an identifier or a contextual
    /// keyword usable as a name (`is T record`, `case T record:` —
    /// `record` lexes as `Keyword::Record_`, not an identifier).
    /// Pattern combinators (`and`/`or`/`not`) and guards (`when`) are
    /// NOT names — they continue the pattern (`is T and ...`).
    pub(super) fn take_pattern_binding(&mut self) -> Option<(String, Span)> {
        match &self.peek().kind {
            TokenKind::Identifier(name) => {
                let name = name.clone();
                let span = self.peek().span;
                self.bump();
                Some((name, span))
            }
            TokenKind::Keyword(
                Keyword::File
                | Keyword::Required
                | Keyword::Scoped
                | Keyword::Record_
                | Keyword::Async
                | Keyword::Await
                | Keyword::Yield
                | Keyword::With
                | Keyword::From,
            ) => {
                let span = self.peek().span;
                let bytes = &self.source[span.start as usize..span.end as usize];
                let src = std::str::from_utf8(bytes).unwrap_or("").to_owned();
                self.bump();
                Some((src, span))
            }
            _ => None,
        }
    }

    /// Queue `Let`s for designations nested inside an already-skipped
    /// pattern range: `{ Nullable: true, Element: JsonTypeInfo e }`,
    /// `{ Value: { } t }`, `(int x, string y)`, `[int x, ..]`. Call
    /// immediately after the balanced skip (scans tokens `[open,
    /// self.pos)` without consuming). A designation is a `Type name`
    /// pair right after `:`, `(`, `[`, `,` — or a CLOSER (`{ } t`
    /// binds `t` to a value matching the empty pattern) — whose
    /// follower closes the element (`,`, `}`, `)`, `]`).
    /// Without this the uses fail downstream (`E0025 unknown name`),
    /// e.g. System.Text.Json's `{ ..., ElementTypeInfo: JsonTypeInfo
    /// elementTypeInfo }` + `typeInfo = elementTypeInfo`. The opaque
    /// skip stays (the AST keeps no pattern detail); only the bindings
    /// are exposed, like trailing `is { } x` bindings.
    pub(super) fn queue_inner_designations(&mut self, open: usize) {
        // NOTE: `peek_at`/`skip_generic_at`/etc. are cursor-RELATIVE,
        // but this range lies BEHIND the cursor — index `tokens`
        // absolutely here.
        let end = self.pos;
        let kind_at = |s: &Parser, a: usize| -> TokenKind {
            s.tokens
                .get(a)
                .map(|t| t.kind.clone())
                .unwrap_or(TokenKind::Eof)
        };
        let span_at = |s: &Parser, a: usize| -> crate::models::Span {
            s.tokens
                .get(a)
                .map(|t| t.span)
                .unwrap_or(crate::models::Span::default())
        };
        let mut i = open;
        while i < end {
            // After a CLOSER the designation follows directly — the
            // group itself is the "type" (`{ } t`, `(...) t`,
            // `[...] t` bind `t`). Recorded as `var` (opaque group).
            if matches!(
                kind_at(self, i),
                TokenKind::RightBrace | TokenKind::RightParen | TokenKind::RightBracket
            ) {
                let j = i + 1;
                let is_binding = match kind_at(self, j) {
                    TokenKind::Identifier(_) => true,
                    TokenKind::Keyword(kw) => matches!(
                        kw,
                        Keyword::File
                            | Keyword::Required
                            | Keyword::Scoped
                            | Keyword::Record_
                            | Keyword::Async
                            | Keyword::Await
                            | Keyword::Yield
                            | Keyword::With
                            | Keyword::From
                    ),
                    _ => false,
                };
                if is_binding
                    && matches!(
                        kind_at(self, j + 1),
                        TokenKind::Comma
                            | TokenKind::RightBrace
                            | TokenKind::RightParen
                            | TokenKind::RightBracket
                    )
                {
                    let nspan = span_at(self, j);
                    let ns = nspan.start as usize;
                    let ne = nspan.end as usize;
                    let name = if ne <= self.source.len() && ns <= ne {
                        std::str::from_utf8(&self.source[ns..ne])
                            .unwrap_or("")
                            .to_owned()
                    } else {
                        String::new()
                    };
                    if !name.is_empty() && name != "_" {
                        self.pre_pending.push(crate::models::Stmt::Let {
                            type_name: String::from("var"),
                            name,
                            init: crate::models::Expr::Literal(
                                crate::models::Literal::Int(0),
                            ),
                            is_byref: false,
                            access: crate::models::Access::Default,
                            span: span_at(self, open).to(nspan),
                        });
                    }
                    i = j + 1;
                    continue;
                }
                i += 1;
                continue;
            }
            if !matches!(
                kind_at(self, i),
                TokenKind::Colon
                    | TokenKind::LeftParen
                    | TokenKind::LeftBracket
                    | TokenKind::Comma
            ) {
                i += 1;
                continue;
            }
            let mut j = i + 1;
            // Type start: identifier, contextual name, or builtin
            // keyword — but not a nested group, literal, constant, or
            // combinator (`(`, `not`, `true`, ...).
            let type_ok = match kind_at(self, j) {
                TokenKind::Identifier(_) => true,
                TokenKind::Keyword(kw) => matches!(
                    kw,
                    Keyword::File
                        | Keyword::Required
                        | Keyword::Scoped
                        | Keyword::Record_
                        | Keyword::Async
                        | Keyword::Await
                        | Keyword::Yield
                        | Keyword::With
                        | Keyword::From
                ),
                _ => false,
            };
            if !type_ok {
                i += 1;
                continue;
            }
            let tstart = span_at(self, j).start as usize;
            j += 1;
            while kind_at(self, j) == TokenKind::Dot
                && matches!(kind_at(self, j + 1), TokenKind::Identifier(_))
            {
                j += 2;
            }
            // Generic list (absolute depth scan; `>>`/`>>>` lex as
            // Shr/ShrShr).
            if matches!(kind_at(self, j), TokenKind::Less | TokenKind::Shl) {
                let mut depth = 0usize;
                loop {
                    match kind_at(self, j) {
                        TokenKind::Eof => break,
                        TokenKind::Less => depth += 1,
                        TokenKind::Shl => depth += 2,
                        TokenKind::Greater => {
                            depth = depth.saturating_sub(1);
                            if depth == 0 {
                                j += 1;
                                break;
                            }
                        }
                        TokenKind::Shr => {
                            depth = depth.saturating_sub(2);
                            if depth == 0 {
                                j += 1;
                                break;
                            }
                        }
                        TokenKind::ShrShr => {
                            depth = depth.saturating_sub(3);
                            if depth == 0 {
                                j += 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                    j += 1;
                    if j >= end + 200 {
                        break;
                    }
                }
            }
            // Suffixes (`?`, `*`, `[...]`).
            loop {
                let mut progressed = false;
                if kind_at(self, j) == TokenKind::Question {
                    j += 1;
                    progressed = true;
                }
                while kind_at(self, j) == TokenKind::Star {
                    j += 1;
                    progressed = true;
                }
                while kind_at(self, j) == TokenKind::LeftBracket {
                    let mut depth = 0usize;
                    let mut k = j;
                    let found = loop {
                        match kind_at(self, k) {
                            TokenKind::Eof => break false,
                            TokenKind::LeftBracket => depth += 1,
                            TokenKind::RightBracket => {
                                depth = depth.saturating_sub(1);
                                if depth == 0 {
                                    k += 1;
                                    break true;
                                }
                            }
                            _ => {}
                        }
                        k += 1;
                        if k >= end + 200 {
                            break false;
                        }
                    };
                    if !found {
                        break;
                    }
                    j = k;
                    progressed = true;
                }
                if !progressed {
                    break;
                }
            }
            // Designation (never a combinator/guard/constant).
            let is_binding = match kind_at(self, j) {
                TokenKind::Identifier(_) => true,
                TokenKind::Keyword(kw) => matches!(
                    kw,
                    Keyword::File
                        | Keyword::Required
                        | Keyword::Scoped
                        | Keyword::Record_
                        | Keyword::Async
                        | Keyword::Await
                        | Keyword::Yield
                        | Keyword::With
                        | Keyword::From
                ),
                _ => false,
            };
            if !is_binding {
                i = j;
                continue;
            }
            // Follower must close the element.
            if !matches!(
                kind_at(self, j + 1),
                TokenKind::Comma
                    | TokenKind::RightBrace
                    | TokenKind::RightParen
                    | TokenKind::RightBracket
            ) {
                i = j;
                continue;
            }
            let nspan = span_at(self, j);
            let ns = nspan.start as usize;
            let ne = nspan.end as usize;
            let name = if ne <= self.source.len() && ns <= ne {
                std::str::from_utf8(&self.source[ns..ne])
                    .unwrap_or("")
                    .to_owned()
            } else {
                String::new()
            };
            if !name.is_empty() && name != "_" {
                let ty = if tstart <= ns && ns <= self.source.len() {
                    std::str::from_utf8(&self.source[tstart..ns])
                        .unwrap_or("var")
                        .trim()
                        .to_owned()
                } else {
                    String::from("var")
                };
                self.pre_pending.push(crate::models::Stmt::Let {
                    type_name: if ty.is_empty() {
                        String::from("var")
                    } else {
                        ty
                    },
                    name,
                    init: crate::models::Expr::Literal(crate::models::Literal::Int(0)),
                    is_byref: false,
                    access: crate::models::Access::Default,
                    span: span_at(self, open).to(nspan),
                });
            }
            i = j + 1;
        }
    }

    /// True if the token at `offset` can be a pattern-binding name:
    /// an identifier or a contextual keyword (`case T record:`) —
    /// but not a combinator/guard/constant (`and`/`or`/`not`/`when`,
    /// `true`/`false`/`null`), which continue or close the pattern.
    pub(super) fn peek_is_pattern_binding(&self, offset: usize) -> bool {
        match &self.peek_at(offset).kind {
            TokenKind::Identifier(_) => true,
            TokenKind::Keyword(
                Keyword::File
                | Keyword::Required
                | Keyword::Scoped
                | Keyword::Record_
                | Keyword::Async
                | Keyword::Await
                | Keyword::Yield
                | Keyword::With
                | Keyword::From,
            ) => true,
            _ => false,
        }
    }

    pub(super) fn bump(&mut self) -> Token {
        let token = self.peek().clone();
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        token
    }

    /// Lookahead: index just past a balanced `<...>` generic list
    /// starting at `start` (which must be `Less` or `Shl`). `>>`/`>>>`
    /// lex as Shr/ShrShr. Returns the `Eof` index when unbalanced.
    pub(super) fn skip_generic_at(&self, start: usize) -> usize {
        let mut depth = 0i32;
        let mut i = start;
        loop {
            match self.peek_at(i).kind {
                TokenKind::Eof => break,
                TokenKind::Less => depth += 1,
                TokenKind::Shl => depth += 2,
                TokenKind::Greater => {
                    depth -= 1;
                    if depth == 0 {
                        i += 1;
                        break;
                    }
                }
                TokenKind::Shr => {
                    depth -= 2;
                    if depth == 0 {
                        i += 1;
                        break;
                    }
                }
                TokenKind::ShrShr => {
                    depth -= 3;
                    if depth == 0 {
                        i += 1;
                        break;
                    }
                }
                _ => {}
            }
            i += 1;
        }
        i
    }

    /// If `start` opens a balanced generic list, return the index just
    /// past its closer; otherwise `None`. Lets pattern/type gates
    /// distinguish `A<B>` (type) from `a < b` (comparison) without
    /// consuming on failure.
    pub(super) fn generic_args_end_at(&self, start: usize) -> Option<usize> {
        if !matches!(
            self.peek_at(start).kind,
            TokenKind::Less | TokenKind::Shl
        ) {
            return None;
        }
        let end = self.skip_generic_at(start);
        if matches!(self.peek_at(end).kind, TokenKind::Eof) {
            return None;
        }
        Some(end)
    }

    /// True if the token at `end` (just past a balanced generic list)
    /// can continue a type in pattern position: `.`/`(`/`{`/`[`/`?`,
    /// a bare `=>`, or a binding name followed by `=>`/`or`/`and`/`when`.
    /// Anything else (e.g. the `b` in `a < b =>`, whose scan only
    /// "balances" by accident) means comparison, not generics.
    pub(super) fn generic_follow_ok(&self, end: usize) -> bool {
        match &self.peek_at(end).kind {
            TokenKind::Dot
            | TokenKind::LeftParen
            | TokenKind::LeftBrace
            | TokenKind::LeftBracket
            | TokenKind::Question
            | TokenKind::FatArrow => true,
            _ if self.peek_is_name(end) => {
                matches!(
                    &self.peek_at(end + 1).kind,
                    TokenKind::FatArrow
                        | TokenKind::Keyword(Keyword::Or_)
                        | TokenKind::Keyword(Keyword::And_)
                        | TokenKind::Keyword(Keyword::When_)
                )
            }
            _ => false,
        }
    }

    /// Lookahead: index past any `?`/`*`/`[...]` type suffixes starting
    /// at `start`.
    pub(super) fn skip_suffixes_at(&self, mut i: usize) -> usize {
        loop {
            match self.peek_at(i).kind {
                TokenKind::Question | TokenKind::Star => i += 1,
                TokenKind::LeftBracket => {
                    // Index SUFFIX (`a[i].M`, `a[i, j].M`) is a value,
                    // not a rank specifier: only EMPTY brackets (`[]`,
                    // `[,]`) continue a type spelling. A bracket holding
                    // an expression belongs to the value being parsed —
                    // consuming it here lets tuple-cast lookahead eat
                    // `(t.C[o].M)` as a "type", stranding the operand.
                    // (This helper is shared with declaration paths where
                    // only rank specifiers are legal anyway, so narrowing
                    // it is safe everywhere.)
                    let mut j = i + 1;
                    let mut depth = 0usize;
                    let mut empty = true;
                    loop {
                        match self.peek_at(j).kind {
                            TokenKind::Eof => break,
                            TokenKind::LeftBracket => {
                                depth += 1;
                                empty = false;
                            }
                            TokenKind::RightBracket => {
                                if depth == 0 {
                                    j += 1;
                                    break;
                                }
                                depth = depth.saturating_sub(1);
                            }
                            TokenKind::Comma if depth == 0 => {}
                            _ => {
                                empty = false;
                            }
                        }
                        j += 1;
                    }
                    if !empty {
                        break;
                    }
                    i = j;
                }
                _ => break,
            }
        }
        i
    }

    /// If the cursor starts a tuple TYPE `(T1 n1, T2 n2, ...)` with
    /// optional `[]`/`?`/`*` suffixes, return the index just past the
    /// suffixes. Detection requires a `Type Name` pair (name followed
    /// by `,`/`)`) so argument/expression lists (`(a, b)`, `(x)`)
    /// never match.
    pub(super) fn tuple_type_at(&self, base: usize) -> Option<usize> {
        if !matches!(self.peek_at(base).kind, TokenKind::LeftParen) {
            return None;
        }
        let mut i = base + 1;
        let mut pdepth = 1i32;
        let mut adepth = 0i32;
        let mut found_pair = false;
        loop {
            match &self.peek_at(i).kind {
                TokenKind::Eof => return None,
                TokenKind::LeftParen => {
                    if adepth == 0 {
                        pdepth += 1;
                    }
                }
                TokenKind::RightParen => {
                    if adepth == 0 {
                        pdepth -= 1;
                        if pdepth == 0 {
                            i += 1;
                            break;
                        }
                    }
                }
                TokenKind::Less => adepth += 1,
                TokenKind::Shl => adepth += 2,
                TokenKind::Greater => adepth = (adepth - 1).max(0),
                TokenKind::Shr => adepth = (adepth - 2).max(0),
                TokenKind::ShrShr => adepth = (adepth - 3).max(0),
                _ => {}
            }
            // Pair check at the top level: `Type ... Name` with the name
            // followed by `,`/`)`.
            if pdepth == 1
                && adepth == 0
                && matches!(self.peek_at(i).kind, TokenKind::Identifier(_))
            {
                let mut j = i + 1;
                while self.peek_at(j).kind == TokenKind::Dot
                    && self.peek_is_name(j + 1)
                {
                    j += 2;
                }
                if matches!(
                    self.peek_at(j).kind,
                    TokenKind::Less | TokenKind::Shl
                ) {
                    j = self.skip_generic_at(j);
                }
                j = self.skip_suffixes_at(j);
                if self.peek_is_name(j)
                    && matches!(
                        self.peek_at(j + 1).kind,
                        TokenKind::Comma | TokenKind::RightParen
                    )
                {
                    found_pair = true;
                }
            }
            i += 1;
        }
        if !found_pair {
            return None;
        }
        Some(self.skip_suffixes_at(i))
    }

    /// If the cursor starts a tuple TYPE as above AND a member name
    /// follows the suffixes, return that name's token index. Used for
    /// tuple-typed members (`(T n)[] Prop { get; }`). Also accepts
    /// unnamed tuples (see below) when a name follows.
    pub(super) fn tuple_type_shape_at(&self) -> Option<usize> {
        if let Some(past) = self.tuple_type_at(0) {
            if self.peek_is_name(past) {
                return Some(past);
            }
        }
        if let Some(past) = self.unnamed_tuple_type_at(0) {
            if self.peek_is_name(past) {
                return Some(past);
            }
        }
        // Member-position fallback: `(A, B) Name` with plain custom
        // types satisfies neither rule above (no element names, no
        // primitive/complex element), yet at member position an
        // argument list is impossible — so any 2+ comma-separated
        // type-like elements plus a trailing name is a tuple type.
        if let Some(past) = self.member_tuple_at(0) {
            if self.peek_is_name(past) {
                return Some(past);
            }
        }
        None
    }

    /// If the cursor opens a tuple type `(...)` (named, unnamed, or
    /// member-position shape), consume exactly through the matching
    /// `)` and return `"tuple"`. Otherwise consume nothing and return
    /// `None` (speculative errors dropped). Suffixes (`[]`/`?`) are
    /// left for the caller — for `stackalloc`/`out` the brackets that
    /// follow are sizes, not suffixes.
    pub(super) fn consume_tuple_type(&mut self) -> Option<String> {
        if self.peek().kind != TokenKind::LeftParen {
            return None;
        }
        if !(self.tuple_type_at(0).is_some()
            || self.unnamed_tuple_type_at(0).is_some()
            || self.member_tuple_at(0).is_some())
        {
            return None;
        }
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
        Some("tuple".to_owned())
    }

    /// True if the token at `idx` can start a cast operand: a name,
    /// a literal, `true`/`false`/`null`, a nested `(`, or `new`.
    /// Anything else (`;`, `,`, `)`, `.`, operators, ...) means the
    /// preceding parens were a literal/grouping, not a cast type.
    fn is_cast_operand_start(&self, idx: usize) -> bool {
        self.peek_is_name(idx)
            || matches!(
                &self.peek_at(idx).kind,
                TokenKind::IntLiteral(_)
                    | TokenKind::StringLiteral(_)
                    | TokenKind::FloatLiteral(_)
                    | TokenKind::Keyword(Keyword::True)
                    | TokenKind::Keyword(Keyword::False)
                    | TokenKind::Keyword(Keyword::Null)
                    | TokenKind::LeftParen
                    | TokenKind::Keyword(Keyword::New)
            )
    }

    /// If the cursor opens a tuple TYPE directly applied to an operand
    /// (`(int, T)x`), return the index past the tuple's `)`+suffixes.
    /// A tuple literal is never applied to anything (`(a, b)`,
    /// `(a, b).M` keep the literal path: `;`/`,`/`)`/`.` never start
    /// an operand), so this disambiguates cast-to-tuple from literals.
    /// Also accepts the doubled shape (`((int, T))x`: outer pair is
    /// the cast's own, inner pair the type). Pure lookahead.
    pub(super) fn tuple_cast_operand_at(&self) -> Option<usize> {
        // Doubled: `((...)` — validate the inner tuple at base 1, then
        // require the outer `)` and an operand starter.
        if matches!(self.peek_at(0).kind, TokenKind::LeftParen)
            && matches!(self.peek_at(1).kind, TokenKind::LeftParen)
        {
            let inner_end = self
                .tuple_type_at(1)
                .or_else(|| self.unnamed_tuple_type_at(1))
                .or_else(|| self.member_tuple_at(1))?;
            if !matches!(self.peek_at(inner_end).kind, TokenKind::RightParen) {
                return None;
            }
            let after = inner_end + 1;
            if self.is_cast_operand_start(after) {
                return Some(after);
            }
            return None;
        }
        // Single: `(...)x`. A NON-tuple parenthesized expression
        // (`(a.b[i].C)`) is never a tuple cast — only consult the
        // shape matchers when a top-level comma is present (tuple
        // types/literals always contain one; grouped expressions may
        // not). Without this gate, `(_currentDataTable.Columns[ordinal]
        // .ColumnName)` matches `member_tuple_at` via
        // `skip_tuple_element_type_at` (dotted names + index suffixes
        // look type-like) and the grouping is consumed as a cast's
        // type, stranding the operand.
        if !self.paren_holds_comma() {
            return None;
        }
        let end = self
            .tuple_type_at(0)
            .or_else(|| self.unnamed_tuple_type_at(0))
            .or_else(|| self.member_tuple_at(0))?;
        if self.is_cast_operand_start(end) {
            return Some(end);
        }
        None
    }

    /// Member-position tuple type `(T1, T2, ...) Name`: 2+ top-level
    /// comma-separated type-like elements (names optional per
    /// element), then suffixes. Returns the index past the suffixes.
    /// ONLY valid where an argument list is impossible (member
    /// declarations) — never use in expression position.
    pub(super) fn member_tuple_at(&self, base: usize) -> Option<usize> {
        if !matches!(self.peek_at(base).kind, TokenKind::LeftParen) {
            return None;
        }
        let mut i = base + 1;
        let mut parts = 0u32;
        let mut guard = 0u32;
        loop {
            if guard > 200 {
                return None;
            }
            guard += 1;
            // Element type (nested tuples included).
            let Some(next) = self.skip_tuple_element_type_at(i) else {
                return None;
            };
            i = next;
            // Optional element name (`(int a, B)` mixes named/unnamed).
            if self.peek_is_name(i)
                && matches!(
                    self.peek_at(i + 1).kind,
                    TokenKind::Comma | TokenKind::RightParen
                )
            {
                i += 1;
            }
            parts += 1;
            if matches!(self.peek_at(i).kind, TokenKind::Comma) {
                i += 1;
                if matches!(self.peek_at(i).kind, TokenKind::RightParen) {
                    return None; // trailing comma
                }
                continue;
            } else if matches!(self.peek_at(i).kind, TokenKind::RightParen) {
                i += 1;
                break;
            } else {
                return None;
            }
        }
        if parts < 2 {
            return None;
        }
        Some(self.skip_suffixes_at(i))
    }

    /// Skip one tuple-element TYPE at token index `i` and return the
    /// index past it (suffixes included), or `None` when no element
    /// type starts here. Handles nested tuple types
    /// (`(RegexNode LoopNode, (char Char, ...) Literal)? x` — the
    /// inner `(...)` is an element type), dotted names, generic lists,
    /// and `?`/`*`/`[...]` suffixes. Shared by the member-tuple and
    /// declaration-tuple shape scanners (both previously rejected a
    /// `(`-starting element, so nested named tuples fell through to
    /// expression parsing as tuple literals).
    pub(super) fn skip_tuple_element_type_at(&self, i: usize) -> Option<usize> {
        if self.peek_at(i).kind == TokenKind::LeftParen {
            // Nested tuple type: skip the balanced group, then any
            // suffixes (`(...)?`, `(...)?[]`).
            let mut depth = 0i32;
            let mut j = i;
            loop {
                if j - i > 500 {
                    return None;
                }
                match self.peek_at(j).kind {
                    TokenKind::Eof => return None,
                    TokenKind::LeftParen => depth += 1,
                    TokenKind::RightParen => {
                        depth -= 1;
                        if depth == 0 {
                            j += 1;
                            break;
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            return Some(self.skip_suffixes_at(j));
        }
        // Element type: identifier, contextual name, or keyword
        // builtin (`int`, `string`, ...), plus dotted segments.
        // Index SUFFIXES are values, not rank specifiers: `a[i]` ends
        // the element here (a trailing `.M` then fails the
        // Comma/RightParen check and the whole shape misses, instead
        // of `member_tuple_at` eating `(t.C[o].M)` as a "type").
        // `skip_suffixes_at` already stops at non-empty brackets; the
        // `?`/`*` scalar suffixes below stay valid.
        if !(matches!(self.peek_at(i).kind, TokenKind::Identifier(_))
            || self.peek_is_name(i)
            || matches!(self.peek_at(i).kind, TokenKind::Keyword(_)))
        {
            return None;
        }
        let mut j = i + 1;
        while self.peek_at(j).kind == TokenKind::Dot && self.peek_is_name(j + 1) {
            j += 2;
        }
        if matches!(self.peek_at(j).kind, TokenKind::Less | TokenKind::Shl) {
            let next = self.skip_generic_at(j);
            if next == j {
                return None;
            }
            j = next;
        }
        // `?`/`*` scalar suffixes and EMPTY-bracket rank specifiers
        // (`[]`, `[,]`). A bracket holding an expression (`a[i]`) is
        // an indexer value, not part of a type: stop there so the
        // trailing Comma/RightParen check fails the shape instead of
        // `member_tuple_at` eating `(t.C[o].M)` as a "type".
        // (Mirrors `skip_suffixes_at`, which narrowed the same way.)
        loop {
            match self.peek_at(j).kind {
                TokenKind::Question | TokenKind::Star => j += 1,
                TokenKind::LeftBracket => {
                    let mut k = j + 1;
                    let mut depth = 0usize;
                    let mut empty = true;
                    loop {
                        match self.peek_at(k).kind {
                            TokenKind::Eof => break,
                            TokenKind::LeftBracket => {
                                depth += 1;
                                empty = false;
                            }
                            TokenKind::RightBracket => {
                                if depth == 0 {
                                    k += 1;
                                    break;
                                }
                                depth = depth.saturating_sub(1);
                            }
                            TokenKind::Comma if depth == 0 => {}
                            _ => {
                                empty = false;
                            }
                        }
                        k += 1;
                    }
                    if !empty {
                        break;
                    }
                    j = k;
                }
                _ => break,
            }
        }
        Some(j)
    }

    /// Unnamed tuple TYPE `(T1, T2, ...)` (no element names): every
    /// top-level element must be type-like (name + dots/generics/
    /// suffixes), with at least two elements, and at least one element
    /// primitive-named (`int`, `string`, ...) or structurally complex
    /// (dotted/generic/suffixed) — otherwise `(a, b)` (an argument
    /// list) would match. Returns the index past `)`+suffixes.
    pub(super) fn unnamed_tuple_type_at(&self, base: usize) -> Option<usize> {
        if !matches!(self.peek_at(base).kind, TokenKind::LeftParen) {
            return None;
        }
        const PRIMS: &[&str] = &[
            "int", "uint", "long", "ulong", "short", "ushort", "byte",
            "sbyte", "nint", "nuint", "bool", "char", "float", "double",
            "decimal", "string", "object", "var", "void", "dynamic",
        ];
        let mut i = base + 1;
        let mut pdepth = 1i32;
        let mut adepth = 0i32;
        let mut bdepth = 0usize;
        let mut parts = 0u32;
        let mut strong = false;
        // Set after a `.`: the next identifier is a qualified segment,
        // not a second bare name.
        let mut expect_segment = false;
        // Per-part state (only meaningful at all depths zero).
        let mut has_name = false;
        let mut complex = false;
        macro_rules! finish_part {
            () => {
                if !has_name || expect_segment {
                    return None;
                }
                parts += 1;
                if complex {
                    strong = true;
                }
                has_name = false;
                complex = false;
            };
        }
        loop {
            let top = pdepth == 1 && adepth == 0 && bdepth == 0;
            match &self.peek_at(i).kind {
                TokenKind::Eof => return None,
                TokenKind::LeftParen => return None, // nested parens: not a plain tuple type
                TokenKind::LeftBrace => return None,
                TokenKind::RightParen => {
                    pdepth -= 1;
                    if pdepth == 0 {
                        finish_part!();
                        i += 1;
                        break;
                    }
                    return None; // unbalanced
                }
                TokenKind::Less => adepth += 1,
                TokenKind::Shl => adepth += 2,
                TokenKind::Greater => adepth = (adepth - 1).max(0),
                TokenKind::Shr => adepth = (adepth - 2).max(0),
                TokenKind::ShrShr => adepth = (adepth - 3).max(0),
                TokenKind::LeftBracket => {
                    // A bracket holding an expression (`a[i]`) is an
                    // indexer VALUE, not a type suffix: `(t.C[o].M)`
                    // must not match as an unnamed tuple type (only
                    // `(t, u)`-style lists do). Only EMPTY brackets
                    // (`[]`, `[,]`) continue the shape; anything else
                    // rejects. (Mirrors `skip_suffixes_at`, narrowed
                    // the same way.)
                    if top {
                        let mut k = i + 1;
                        let mut depth = 0usize;
                        let mut empty = true;
                        loop {
                            match self.peek_at(k).kind {
                                TokenKind::Eof => break,
                                TokenKind::LeftBracket => {
                                    depth += 1;
                                    empty = false;
                                }
                                TokenKind::RightBracket => {
                                    if depth == 0 {
                                        k += 1;
                                        break;
                                    }
                                    depth = depth.saturating_sub(1);
                                }
                                TokenKind::Comma if depth == 0 => {}
                                _ => {
                                    empty = false;
                                }
                            }
                            k += 1;
                        }
                        if !empty {
                            return None;
                        }
                        complex = true;
                    }
                    bdepth += 1;
                }
                TokenKind::RightBracket => {
                    bdepth = bdepth.saturating_sub(1);
                }
                TokenKind::Comma => {
                    if top {
                        finish_part!();
                        // Reset for the next part (macros expanded above).
                        has_name = false;
                        complex = false;
                    }
                }
                TokenKind::Identifier(name) => {
                    if top {
                        if expect_segment {
                            expect_segment = false;
                        } else if !has_name {
                            has_name = true;
                            if PRIMS.contains(&name.as_str()) {
                                complex = true;
                            }
                        } else {
                            // Second bare name — not a type.
                            return None;
                        }
                    }
                }
                TokenKind::Dot => {
                    if top {
                        if !has_name || expect_segment {
                            return None;
                        }
                        complex = true;
                        expect_segment = true;
                    }
                }
                TokenKind::Question | TokenKind::Star => {
                    if top {
                        if !has_name {
                            return None;
                        }
                        complex = true;
                    }
                }
                _ => {
                    // Keywords, literals, operators, `this`, etc. Only
                    // acceptable inside generics/brackets.
                    if top {
                        return None;
                    }
                }
            }
            // After a dot, the next token must be a name segment.
            if top
                && matches!(self.peek_at(i).kind, TokenKind::Dot)
                && !matches!(self.peek_at(i + 1).kind, TokenKind::Identifier(_))
            {
                return None;
            }
            i += 1;
        }
        if parts < 2 || !strong {
            return None;
        }
        Some(self.skip_suffixes_at(i))
    }

    pub(super) fn error(&mut self, code: &'static str, message: impl Into<String>, span: Span) {
        // Every parser diagnostic is explicitly a *syntax* error — the
        // label is part of the message so text and JSON output both say it.
        self.errors.push(
            Diagnostic::new(
                code,
                Severity::Error,
                format!("syntax error: {}", message.into()),
            )
            .at(span)
            .in_context(self.context.clone()),
        );
    }

    /// Unexpected-token flavor (E0001).
    pub(super) fn unexpected(&mut self, expected_key: &str, found: &str, span: Span) {
        let expected = crate::t!(expected_key);
        let msg = crate::tf!(
            "rubyc.parser.rd.Parser.expect.generic",
            expected,
            found.to_owned()
        );
        self.error("E0001", msg, span);
    }

    /// Consume the current token if it matches; otherwise record an error.
    pub(super) fn expect(&mut self, kind: &TokenKind, context: &str) -> Option<Token> {
        if self.peek().kind == *kind {
            return Some(self.bump());
        }
        // Recovery: a `:` at this point is almost always a named-argument
        // separator that the argument list didn't consume. Bump it and the
        // value token so the surrounding parse can continue.
        if matches!(self.peek().kind, TokenKind::Colon) {
            self.bump();
            if self.peek().kind != TokenKind::Eof {
                self.bump();
            }
            return None;
        }
        let found = describe(self.peek());
        // Full `rubyc.` keys go through locale lookup; inline English
        // context ("to close index", ...) IS the message — routing it
        // through lookup can only produce "[missing translation]".
        let where_ = if context.starts_with("rubyc.") {
            crate::t!(context).to_owned()
        } else {
            context.to_owned()
        };

        // A missing ';' at end of line points at the INSERTION spot —
        // the end of the previous line — not at the next-line token.
        if *kind == TokenKind::Semicolon
            && let Some(prev) = self.pos.checked_sub(1).and_then(|i| self.tokens.get(i))
            && self
                .source
                .get(prev.span.end as usize..self.peek().span.start as usize)
                .is_some_and(|between| between.contains(&b'\n'))
        {
            let mut diag = Diagnostic::new(
                "E0001",
                Severity::Error,
                "syntax error: missing ';' after statement".to_owned(),
            )
            .at(Span::new(prev.span.end, prev.span.end))
            .in_context(self.context.clone());
            diag.help = Some("insert ';' here".to_owned());
            self.errors.push(diag);
            return None;
        }

        let span = self.peek().span;
        self.error(
            "E0001",
            format!("expected {} {where_}, found {found}", human_kind(kind)),
            span,
        );
        None
    }

    pub(super) fn eat(&mut self, kind: &TokenKind) -> Option<Token> {
        (self.peek().kind == *kind).then(|| self.bump())
    }

    /// Skip forward to just past the next `;` or before a `}`, so parsing of
    /// enclosing constructs can resume after a malformed statement.
    pub(super) fn synchronize(&mut self) {
        loop {
            match self.peek().kind {
                TokenKind::Semicolon => {
                    self.bump();
                    return;
                }
                TokenKind::RightBrace | TokenKind::Eof => return,
                _ => {
                    self.bump();
                }
            }
        }
    }

    // ---- grammar productions live in the sibling files ----

    pub(super) fn expect_identifier(&mut self, context: &str) -> Option<String> {
        match self.peek().kind.clone() {
            TokenKind::Identifier(name) => {
                self.bump();
                Some(name)
            }
            // C# contextual keywords (`file`, `required`, `scoped`,
            // `record`, `async`, `await`, `yield`, `with`, `from`, `and`,
            // `or`, `not`, `when`) are legal identifiers. Accept them by
            // source text so real names like `var file` parse.
            TokenKind::Keyword(
                kw @ (Keyword::File
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
                | Keyword::When_),
            ) => {
                let _ = kw;
                let span = self.peek().span;
                let bytes = &self.source[span.start as usize..span.end as usize];
                let src = std::str::from_utf8(bytes).unwrap_or("").to_owned();
                self.bump();
                Some(src)
            }
            other => {
                let span = self.peek().span;
                // Full `rubyc.` keys go through locale lookup; inline
                // English context ("as type", ...) IS the message.
                let phrase = if context.starts_with("rubyc.") {
                    crate::t!(context).to_owned()
                } else {
                    context.to_owned()
                };
                let msg = crate::tf!(
                    "rubyc.parser.rd.Parser.expected_identifier",
                    phrase,
                    describe_kind(&other)
                );
                self.error("E0001", msg, span);
                None
            }
        }
    }

    pub(super) fn expect_identifier_or_keyword(&mut self, context: &str) -> Option<String> {
        match self.peek().kind.clone() {
            TokenKind::Identifier(name) => {
                self.bump();
                Some(name)
            }
            TokenKind::Keyword(_) => {
                // Use the source text of the keyword (e.g. `required`, not
                // the Debug form `Required`) so the name matches its usage.
                let span = self.peek().span;
                let bytes = &self.source[span.start as usize..span.end as usize];
                let src = std::str::from_utf8(bytes).unwrap_or("").to_owned();
                self.bump();
                Some(src)
            }
            other => {
                let span = self.peek().span;
                // Full `rubyc.` keys go through locale lookup; inline
                // English context IS the message.
                let phrase = if context.starts_with("rubyc.") {
                    crate::t!(context).to_owned()
                } else {
                    context.to_owned()
                };
                let msg = crate::tf!(
                    "rubyc.parser.rd.Parser.expected_identifier",
                    phrase,
                    describe_kind(&other)
                );
                self.error("E0001", msg, span);
                None
            }
        }
    }
}

pub(super) fn expr_span(expr: &Expr) -> Span {
    match expr {
        Expr::Binary { span, .. } | Expr::Unary { span, .. } | Expr::Call { span, .. } => *span,
        _ => Span::default(),
    }
}

/// End span of the most recently consumed token.
pub(super) fn expr_span_last(tokens: &[Token]) -> Span {
    tokens.last().map(|t| t.span).unwrap_or_default()
}

pub(super) fn describe(token: &Token) -> String {
    describe_kind(&token.kind)
}

pub(super) fn describe_kind(kind: &TokenKind) -> String {
    match kind {
        TokenKind::Identifier(n) => format!("identifier '{n}'"),
        TokenKind::StringLiteral(s) => format!("string \"{s}\""),
        TokenKind::IntLiteral(v) => format!("integer {v}"),
        TokenKind::FloatLiteral(bits) => format!("float {}", f64::from_bits(*bits)),
        TokenKind::Keyword(kw) => format!("'{kw:?}'"),
        TokenKind::Eof => "end of file".into(),
        other => human_kind(other),
    }
}

/// Human name for a token kind we *expected* — punctuation reads as
/// punctuation, not as a Debug enum variant.
pub(super) fn human_kind(kind: &TokenKind) -> String {
    match kind {
        TokenKind::Semicolon => "';'".into(),
        TokenKind::Colon => "':'".into(),
        TokenKind::Comma => "','".into(),
        TokenKind::Assign => "'='".into(),
        TokenKind::EqEq => "'=='".into(),
        TokenKind::FatArrow => "'=>'".into(),
        TokenKind::Arrow => "'->'".into(),
        TokenKind::LeftParen => "'('".into(),
        TokenKind::RightParen => "')'".into(),
        TokenKind::LeftBrace => "'{'".into(),
        TokenKind::RightBrace => "'}'".into(),
        TokenKind::ShlAssign => "'<<='".into(),
        TokenKind::ShrAssign => "'>>='".into(),
        TokenKind::ShrShrAssign => "'>>>='".into(),
        // No delegation back to `describe_kind`: every unmatched kind must
        // terminate here or error paths recurse to stack overflow.
        other => format!("'{other:?}'"),
    }
}
