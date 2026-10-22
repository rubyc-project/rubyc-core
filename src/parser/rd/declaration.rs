//! Declarations: interfaces, classes, members, attributes, methods, params.

use crate::models::{
    Access, Attribute, AttributeArg, AttributeValue, BinOp, Block, ClassDecl, ClassMember,
    DelegateDecl, Diagnostic, EventDecl, Expr, FieldDecl, IndexerDecl, InterfaceDecl, Keyword,
    Literal, MethodDecl, MethodSig, Param, PropertyDecl, Severity, Span, TokenKind, UnOp,
};

/// Fold an enum initializer to a constant: integer literals, references
/// to earlier variants of the same enum, dotted references to variants of
/// enums already parsed in this file (`OtherEnum.X`), unary (`-`, `!`,
/// `~`) and binary (`+ - * / % & | ^ << >>`) combinations, and casts.
/// Returns `None` when the expression is not a compile-time constant
/// (e.g. a cross-file reference, which degrades to auto-numbering).
fn fold_enum_expr(
    expr: &Expr,
    variants: &[(String, Option<i64>)],
    extern_enums: &[(String, String, i64)],
) -> Option<i64> {
    match expr {
        Expr::Literal(Literal::Int(v)) => Some(*v as i64),
        Expr::Ident { name, .. } => variants
            .iter()
            .find(|(n, _)| n == name)
            .and_then(|(_, v)| *v),
        Expr::Member { object, name, .. } => {
            // Dotted path: match the trailing `Enum.Variant` segments
            // against enums completed earlier in this file.
            let mut segs = vec![name.clone()];
            let mut cur = object.as_ref();
            loop {
                match cur {
                    Expr::Ident { name: n, .. } => {
                        segs.push(n.clone());
                        break;
                    }
                    Expr::Member { object: o, name: n, .. } => {
                        segs.push(n.clone());
                        cur = o.as_ref();
                    }
                    _ => return None,
                }
            }
            if segs.len() < 2 {
                return None;
            }
            let variant = &segs[0];
            let en = &segs[1];
            extern_enums
                .iter()
                .find(|(e, v, _)| e == en && v == variant)
                .map(|(_, _, val)| *val)
        }
        Expr::Unary { op, inner, .. } => {
            let v = fold_enum_expr(inner, variants, extern_enums)?;
            match op {
                UnOp::Neg => Some(v.wrapping_neg()),
                UnOp::Not => Some(if v == 0 { 1 } else { 0 }),
                UnOp::BitNot => Some(!v),
                _ => None,
            }
        }
        Expr::Binary { op, lhs, rhs, .. } => {
            let a = fold_enum_expr(lhs, variants, extern_enums)?;
            let b = fold_enum_expr(rhs, variants, extern_enums)?;
            match op {
                BinOp::Add => Some(a.wrapping_add(b)),
                BinOp::Sub => Some(a.wrapping_sub(b)),
                BinOp::Mul => Some(a.wrapping_mul(b)),
                BinOp::Div => {
                    if b == 0 {
                        None
                    } else {
                        Some(a.wrapping_div(b))
                    }
                }
                BinOp::Rem => {
                    if b == 0 {
                        None
                    } else {
                        Some(a.wrapping_rem(b))
                    }
                }
                BinOp::BitAnd => Some(a & b),
                BinOp::BitOr => Some(a | b),
                BinOp::BitXor => Some(a ^ b),
                BinOp::Shl => Some(a.wrapping_shl(b as u32)),
                BinOp::Shr => Some(a.wrapping_shr(b as u32)),
                _ => None,
            }
        }
        Expr::Cast { expr, .. } => fold_enum_expr(expr, variants, extern_enums),
        _ => None,
    }
}

/// True for expression shapes that may denote a compile-time constant
/// (possibly from another file): names, member paths, literals, unary
/// and binary combinations, casts, and parenthesizations thereof.
/// Calls, allocations, and lambdas are never constant-shaped.
fn is_const_shaped(expr: &Expr) -> bool {
    match expr {
        Expr::Literal(_) | Expr::Ident { .. } => true,
        Expr::Member { object, .. } => is_const_shaped(object),
        Expr::Unary { inner, .. } => is_const_shaped(inner),
        Expr::Binary { lhs, rhs, .. } => is_const_shaped(lhs) && is_const_shaped(rhs),
        Expr::Cast { expr, .. } => is_const_shaped(expr),
        _ => false,
    }
}

use super::{Parser, describe};

impl Parser {
    /// Wrap a single expression-bodied-member statement (`return expr;`
    //  for value members, the bare expression for effect members) with
    //  the bindings queued while parsing its expression: `is`-pattern
    //  bindings (`pre_pending`) before it, `out` vars (`pending`) after
    //  — mirroring `block()`. Without this, `=> (obj is T x) && ...`
    //  leaks its bindings instead of declaring them in the body.
    fn expression_body(&mut self, stmt: crate::models::Stmt, span: Span) -> Block {
        let mut stmts = std::mem::take(&mut self.pre_pending);
        stmts.push(stmt);
        stmts.append(&mut self.pending);
        Block { stmts, span }
    }

    /// `[modifiers] interface Name { signatures }`
    pub(super) fn interface_decl(&mut self, is_partial: bool) -> Option<InterfaceDecl> {
        let kw = self.bump();
        let name = self.expect_identifier("after 'interface'")?;
        // Record generic type parameters (`interface Foo<T>`) for later
        // passes; `arity` stays the count, as before.
        let generic_params = self.generic_parameter_list();
        let arity = generic_params.len();
        let saved_class = self.context.class.clone();
        self.context.class = Some(name.clone());
        let start = kw.span;

        // Inherited interfaces: `interface A : B, C`. Recorded for
        // transitive member lookup (`IAsyncEnumerator<T>` inherits
        // `DisposeAsync` from `IAsyncDisposable`); class→interface
        // conformance resolves separately as before.
        let mut bases = Vec::new();
        if self.peek().kind == TokenKind::Colon {
            self.bump(); // ':'
            loop {
                // Preserve generic arguments as written
                // (`IEnumerable<T>`), mirroring class bases so
                // arity-aware resolution traverses interface chains.
                // A bare `type_name()` would drop `<T>` to bare
                // `IEnumerable` (arity 0), which never matches the
                // arity-1 entry and truncates transitive implemented
                // lists (`IList` losing `IEnumerable`).
                let mut base: Option<String> = None;
                if matches!(self.peek().kind, TokenKind::Identifier(_)) {
                    let mut b = self.expect_identifier("as base interface")?;
                    while self.eat(&TokenKind::Dot).is_some() {
                        b.push('.');
                        b.push_str(&self.expect_identifier("after '.' in base type")?);
                    }
                    if matches!(
                        self.peek().kind,
                        TokenKind::Less | TokenKind::Shl
                    ) {
                        let start_off = self.peek().span.start as usize;
                        self.skip_generic_parameter_list();
                        let end_off = self.tokens[self
                            .pos
                            .saturating_sub(1)]
                        .span
                        .end as usize;
                        if end_off > start_off && end_off <= self.source.len() {
                            b.push_str(
                                std::str::from_utf8(&self.source[start_off..end_off])
                                    .unwrap_or(""),
                            );
                        }
                    }
                    base = Some(b);
                } else if let Some(b) = self.type_name() {
                    base = Some(b);
                }
                if let Some(b) = base {
                    bases.push(b);
                }
                if self.eat(&TokenKind::Comma).is_none() {
                    break;
                }
            }
        }

        self.skip_type_constraints();

        let mut methods = Vec::new();
        let mut indexers = Vec::new();
        if self
            .expect(&TokenKind::LeftBrace, "to open interface body")
            .is_some()
        {
            while self.peek().kind != TokenKind::RightBrace && self.peek().kind != TokenKind::Eof {
                // Leading attributes (`[Obsolete(...)]`): contracts only,
                // inert for signatures — consume and discard. (MethodSig
                // carries no attrs, like Param.)
                if self.peek().kind == TokenKind::LeftBracket {
                    let _ = self.attribute_lists();
                    continue;
                }
                // An indexer (`TElement this[int i] { get; }`): the
                // interface AST kept only method signatures, so the
                // indexer was discarded and a type-parameter receiver
                // constrained to this interface had nothing to bind.
                if self.interface_indexer_ahead() {
                    match self.class_member() {
                        Some(ClassMember::Indexer(d)) => indexers.push(d),
                        _ => self.synchronize(),
                    }
                    continue;
                }
                if self.skip_non_method_interface_member() {
                    continue;
                }
                match self.interface_method_sig() {
                    Some(sig) => methods.push(sig),
                    None => self.synchronize(),
                }
            }
            self.expect(&TokenKind::RightBrace, "to close interface body");
            self.eat(&TokenKind::Semicolon);
        }
        self.context.class = saved_class;
        Some(InterfaceDecl {
            name,
            generic_params,
            methods,
            indexers,
            bases,
            arity,
            is_partial,
            span: start.to(self.peek().span),
        })
    }

    /// Whether the next interface member is an indexer (`RetType
    /// this[P]`): a `this` token followed by `[` before the member's
    /// body. The scan stops at `{`/`;` so a `this` inside a default
    /// value is not mistaken for one.
    fn interface_indexer_ahead(&self) -> bool {
        let mut i = self.pos;
        while i < self.tokens.len() {
            match &self.tokens[i].kind {
                TokenKind::Keyword(Keyword::This) => {
                    return matches!(
                        self.tokens.get(i + 1).map(|t| &t.kind),
                        Some(TokenKind::LeftBracket)
                    );
                }
                TokenKind::LeftBrace | TokenKind::Semicolon | TokenKind::Eof => {
                    return false;
                }
                _ => i += 1,
            }
        }
        false
    }

    /// Consume interface properties and operators that have no AST node yet.
    /// Recognizing their complete declaration boundary prevents one valid
    /// member from cascading into top-level parser errors.
    fn skip_non_method_interface_member(&mut self) -> bool {
        let start = self.pos;
        while matches!(
            self.peek().kind,
            TokenKind::Keyword(Keyword::Static)
                | TokenKind::Keyword(Keyword::Abstract)
                | TokenKind::Keyword(Keyword::Virtual)
        ) {
            self.bump();
        }

        // Scan the declaration header. A method has `name(`; a property has
        // `{`/`=>`, and an operator contains the `operator` identifier.
        let mut cursor = self.pos;
        let mut property_or_operator = false;
        while cursor < self.tokens.len() {
            match &self.tokens[cursor].kind {
                TokenKind::LeftBrace
                | TokenKind::Assign
                | TokenKind::FatArrow
                | TokenKind::Semicolon => {
                    property_or_operator = true;
                    break;
                }
                TokenKind::Keyword(Keyword::Operator_) => {
                    property_or_operator = true;
                    break;
                }
                TokenKind::LeftParen | TokenKind::RightBrace | TokenKind::Eof => break,
                _ => cursor += 1,
            }
        }
        if !property_or_operator {
            self.pos = start;
            return false;
        }

        let mut brace_depth = 0usize;
        loop {
            match self.peek().kind {
                TokenKind::LeftBrace => {
                    brace_depth += 1;
                    self.bump();
                }
                TokenKind::RightBrace if brace_depth > 0 => {
                    brace_depth -= 1;
                    self.bump();
                    if brace_depth == 0 {
                        return true;
                    }
                }
                TokenKind::Semicolon if brace_depth == 0 => {
                    self.bump();
                    return true;
                }
                TokenKind::RightBrace | TokenKind::Eof => return true,
                _ => {
                    self.bump();
                }
            }
        }
    }

    /// `delegate RetType Name<GenericParams>(params) ;`
    ///
    /// A delegate is a named callable type. The generic parameter list is
    /// recorded as names (variance qualifiers are consumed); invocation
    /// semantics arrive with the lambda/closure work.
    pub(super) fn delegate_decl(&mut self, access: Access) -> Option<DelegateDecl> {
        let start = self.bump().span;
        // By-ref return: `delegate ref T D(...)`, `delegate ref readonly
        // T D(...)`. By-ref-ness has no runtime representation here
        // (same as `ref readonly` parameters) — consume and discard.
        if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Ref_)) {
            self.bump();
            if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Readonly)) {
                self.bump();
            }
        }
        let return_type = self.type_name()?;
        let name = self.expect_identifier("rubyc.parser.rd.Parser.ctx.as_delegate_name")?;
        let generic_params = self.generic_parameter_list();
        self.expect(&TokenKind::LeftParen, "rubyc.parser.rd.Parser.ctx.open_delegate_params")?;
        let params = self.param_list();
        self.expect(&TokenKind::RightParen, "rubyc.parser.rd.Parser.ctx.close_delegate_params")?;
        // Type constraints: `delegate void M<T>(...) where T : struct;`.
        self.skip_type_constraints();
        self.expect(&TokenKind::Semicolon, "rubyc.parser.rd.Parser.ctx.after_delegate")?;
        Some(DelegateDecl {
            access,
            ret_type: return_type,
            name,
            generic_params,
            params,
            span: start.to(self.peek_at(0).span),
        })
    }

    /// Interface member: `Access RetType Name(params) { default body? }`.
    /// The return type may be omitted (defaults to `int`).
    pub(super) fn interface_method_sig(&mut self) -> Option<MethodSig> {
        let access = self.access_modifier();
        let start = self.peek().span;
        // Member modifiers legal on interface members: `new` (hiding),
        // `static` / `sealed` / `abstract` / `virtual` / `override` /
        // `extern` (static abstract/virtual members, DIM), `unsafe`.
        // `static` is recorded (static interface members dispatch
        // directly); the rest carry no codegen meaning for signatures
        // — consume and discard.
        let mut is_static = false;
        while matches!(
            self.peek().kind,
            TokenKind::Keyword(Keyword::New)
                | TokenKind::Keyword(Keyword::Static)
                | TokenKind::Keyword(Keyword::Sealed)
                | TokenKind::Keyword(Keyword::Abstract)
                | TokenKind::Keyword(Keyword::Virtual)
                | TokenKind::Keyword(Keyword::Override)
                | TokenKind::Keyword(Keyword::Extern)
                | TokenKind::Keyword(Keyword::Unsafe)
        ) {
            is_static = is_static
                || matches!(self.peek().kind, TokenKind::Keyword(Keyword::Static));
            self.bump();
        }
        // By-ref return: `ref T M(...)`, `ref readonly T M(...)`
        // (e.g. `IGenericCacheEntry<T>.GetStorageRef`). By-ref-ness has
        // no runtime representation here — consume and discard, like
        // `ref readonly` parameters and delegate returns.
        if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Ref_)) {
            self.bump();
            if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Readonly)) {
                self.bump();
            }
        }
        // Optional return type: `Ident Ident(` vs `Ident(`. The type may
        // be dotted (`Ns.Type Name(`), carry a generic list per segment
        // (`A<T>.B Name(`, `IEnumerator<T> Name(`), and take `?`/`[]`/`*`
        // suffixes: scan past exactly those before looking for the name.
        let mut after_type = 1usize;
        if matches!(self.peek().kind, TokenKind::Identifier(_)) {
            let mut i = 1usize;
            loop {
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
                            _ => {}
                        }
                        i += 1;
                    }
                }
                if self.peek_at(i).kind == TokenKind::Dot && self.peek_is_name(i + 1) {
                    i += 2;
                    continue;
                }
                break;
            }
            loop {
                match self.peek_at(i).kind {
                    TokenKind::Question | TokenKind::Star => i += 1,
                    TokenKind::LeftBracket => {
                        // `[]` or `[...]`: skip to the matching `]`.
                        let mut depth = 0usize;
                        loop {
                            match self.peek_at(i).kind {
                                TokenKind::Eof => break,
                                TokenKind::LeftBracket => depth += 1,
                                TokenKind::RightBracket => {
                                    depth = depth.saturating_sub(1);
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
                _ => break,
                }
            }
            after_type = i;
        }
        // Tuple return type: `(TSelf, TSelf) Name(...)` — fold into a
        // synthetic identifier first (same splice as tuple-typed
        // members), so the `Ident Ident(` gate below proceeds.
        if matches!(self.peek().kind, TokenKind::LeftParen) {
            if let Some(name_idx) = self.tuple_type_shape_at() {
                let type_start = self.peek().span.start as usize;
                let type_end =
                    self.tokens[self.pos + name_idx - 1].span.end as usize;
                let text = std::str::from_utf8(
                    &self.source[type_start..type_end.min(self.source.len())],
                )
                .unwrap_or("tuple")
                .to_owned();
                let span = self.peek().span.to(
                    self.tokens[self.pos + name_idx - 1].span,
                );
                self.tokens.splice(
                    self.pos..self.pos + name_idx,
                    [crate::models::Token {
                        kind: TokenKind::Identifier(text),
                        span,
                    }],
                );
            }
        }
        let ret_type = if matches!(self.peek_at(after_type).kind, TokenKind::Identifier(_)) {
            self.type_name()?
        } else {
            "int".to_owned()
        };
        let mut name = self.expect_identifier("as method name")?;
        // Generic methods: `Name<TKey>(...)` — the list directly before
        // `(` declares this signature's type parameters (recorded);
        // qualifier lists (`IFoo<T>.`) are dropped (mirrors `method_decl`).
        let mut sig_generics = Vec::new();
        loop {
            if self.peek().kind == TokenKind::Less {
                let gp = self.generic_parameter_list();
                if self.peek().kind != TokenKind::Dot {
                    sig_generics = gp;
                    break;
                }
            }
            if self.eat(&TokenKind::Dot).is_some() {
                match self.expect_identifier("method name segment") {
                    Some(seg) => {
                        name.push('.');
                        name.push_str(&seg);
                    }
                    None => break,
                }
            } else {
                break;
            }
        }
        self.expect(&TokenKind::LeftParen, "to open parameter list")?;
        let params = self.param_list();
        self.expect(&TokenKind::RightParen, "to close parameter list")?;
        // Type constraints: `where T : ...` — recorded, because a
        // member on a type-parameter receiver binds through them.
        let constraints = self.parse_type_constraints();

        let body = if self.eat(&TokenKind::LeftBrace).is_some() {
            let mut stmts = Vec::new();
            while self.peek().kind != TokenKind::RightBrace && self.peek().kind != TokenKind::Eof {
                match self.statement() {
                    Some(stmt) => {
                        stmts.push(stmt);
                        stmts.append(&mut self.pending);
                    }
                    None => self.synchronize(),
                }
            }
            let close = self.expect(&TokenKind::RightBrace, "to close default body");
            Some(Block {
                stmts,
                span: start.to(close.map_or(self.peek().span, |t| t.span)),
            })
        } else if self.eat(&TokenKind::FatArrow).is_some() {
            // Expression-bodied interface member
            // (`static virtual bool M(...) => expr;`): runs for value.
            let expression = self.expression()?;
            let end = self
                .expect(&TokenKind::Semicolon, "after expression-bodied member")
                .map_or(self.peek().span, |token| token.span);
            Some(self.expression_body(
                crate::models::Stmt::Return {
                    value: Some(expression),
                    span: end,
                },
                start.to(end),
            ))
        } else {
            self.expect(&TokenKind::Semicolon, "after interface method signature");
            None
        };

        Some(MethodSig {
            access,
            ret_type,
            name,
            generic_params: sig_generics,
            params,
            body,
            is_static,
            span: start,
        })
    }

    /// `[modifiers] class Name[(bases...)] { class-members }`
    #[allow(dead_code)]
    pub(super) fn class_decl(&mut self, access: Access) -> Option<ClassDecl> {
        self.class_decl_impl(access, false, false, false, false)
    }

    pub(super) fn class_decl_impl(
        &mut self,
        access: Access,
        keyword_consumed: bool,
        is_partial: bool,
        is_struct: bool,
        is_record: bool,
    ) -> Option<ClassDecl> {
        let _ = access; // enforcement deferred to semantic analysis
        let kw = if keyword_consumed {
            self.peek().span
        } else {
            self.bump().span
        };
        let name = self.expect_identifier("after 'class'")?;
        // Record generic type parameters (`class Foo<T>`) for later
        // passes (scan scope, lowering) instead of skipping them.
        let generic_params = self.generic_parameter_list();
        // Save/restore the enclosing class: nested types must not clobber
        // the outer context (later members like `~Outer()` need it).
        let saved_class = self.context.class.clone();
        self.context.class = Some(name.clone());
        let start = kw;

        // Optional constructor-style parens: `class Class2() { ... }` or
        // `record Rec(int Id, string Name) { ... }`. Value parameter
        // names are recorded (attributes/defaults/modifiers handled by
        // `param_list`); they are in scope for the whole body (.NET).
        let mut primary_ctor_params = Vec::new();
        // The full parameter list is kept for RECORDS: a positional record
        // parameter becomes a public field with a constructor storing it
        // (`record R(int V)` ⇒ field `V` + `R(int V)`). Only the names are
        // needed for the primary-constructor scoping used elsewhere.
        let mut primary_ctor_specs: Vec<Param> = Vec::new();
        if self.peek().kind == TokenKind::LeftParen {
            self.bump();
            if self.peek().kind != TokenKind::RightParen {
                primary_ctor_specs = self.param_list();
                primary_ctor_params = primary_ctor_specs
                    .iter()
                    .map(|p| p.name.clone())
                    .collect();
            }
            self.expect(&TokenKind::RightParen, "to close class name parens");
        }

        // Inheritance list: `: Base1, Base2, IFace`. A base may carry
        // generic arguments (`: Base<T>`) — recorded as written
        // (`Base<T>`) so arity-aware resolution distinguishes
        // `Task` from `Task<T>` (mirroring .NET). Primary-constructor
        // arguments (`: Base(a, b)`) are consumed, not recorded.
        let mut bases = Vec::new();
        if self.eat(&TokenKind::Colon).is_some() {
            loop {
                let mut base = self.expect_identifier("as base class or interface")?;
                while self.eat(&TokenKind::Dot).is_some() {
                    base.push('.');
                    base.push_str(&self.expect_identifier("after '.' in base type")?);
                }
                // Generic arguments on the base (`: Base<T>`, `: A.B<K,V>`):
                // splice their exact source text so arity-aware resolution
                // distinguishes `Task` from `Task<T>`. In base position a
                // `<` always opens generics (never less-than), so no
                // validation gate is needed.
                if matches!(
                    self.peek().kind,
                    TokenKind::Less | TokenKind::Shl
                ) {
                    let start_off = self.peek().span.start as usize;
                    self.skip_generic_parameter_list();
                    let end_off = self.tokens[self
                        .pos
                        .saturating_sub(1)]
                    .span
                    .end as usize;
                    if end_off > start_off && end_off <= self.source.len() {
                        base.push_str(
                            std::str::from_utf8(&self.source[start_off..end_off])
                                .unwrap_or(""),
                        );
                    }
                }
                bases.push(base);
                if self.peek().kind == TokenKind::LeftParen {
                    self.bump(); // '('
                    let mut depth = 1i32;
                    while depth > 0 && self.peek().kind != TokenKind::Eof {
                        match self.peek().kind {
                            TokenKind::LeftParen => depth += 1,
                            TokenKind::RightParen => depth -= 1,
                            _ => {}
                        }
                        self.bump();
                    }
                }
                if self.eat(&TokenKind::Comma).is_none() {
                    break;
                }
            }
        }

        self.skip_type_constraints();

        let mut members = Vec::new();
        // Struct-style declaration: `struct Name(int Id);` — no body.
        if self.eat(&TokenKind::Semicolon).is_some() {
            // `record R(int V);` takes this path, so the positional
            // members are synthesised here too (the body path below does
            // the same for `record R(int V) { ... }`).
            if is_record || !primary_ctor_specs.is_empty() {
                members.extend(synthesize_record_members(
                    &name,
                    &primary_ctor_specs,
                    if is_record { Access::Public } else { Access::Private },
                    start,
                ));
            }
            return Some(ClassDecl {
                is_record,
                name,
                generic_params: generic_params.clone(),
                primary_ctor_params: primary_ctor_params.clone(),
                bases,
                members,
                span: start,
                is_partial,
                is_struct,
                union_cross_assembly: false,
            });
        }
        if self
            .expect(
                &TokenKind::LeftBrace,
                "rubyc.parser.rd.Parser.ctx.open_class_body",
            )
            .is_some()
        {
            while self.peek().kind != TokenKind::RightBrace && self.peek().kind != TokenKind::Eof {
                match self.class_member() {
                    Some(member) => {
                        members.push(member);
                        members.append(&mut self.pending_members);
                    }
                    None => self.synchronize(),
                }
            }
        }
        let end = self.expect(
            &TokenKind::RightBrace,
            "rubyc.parser.rd.Parser.ctx.close_class_body",
        );
        self.eat(&TokenKind::Semicolon);
        self.context.class = saved_class;
        // A positional RECORD (or a class with primary-constructor
        // parameters, which C# captures into the instance) synthesises
        // its members here too.
        if is_record || !primary_ctor_specs.is_empty() {
            members.extend(synthesize_record_members(
                &name,
                &primary_ctor_specs,
                if is_record { Access::Public } else { Access::Private },
                start,
            ));
        }
        Some(ClassDecl {
            is_record,
            name,
            generic_params,
            primary_ctor_params,
            bases,
            members,
            span: start.to(end.map_or(self.peek().span, |t| t.span)),
            is_partial,
            is_struct,
            union_cross_assembly: false,
        })
    }

    /// Lookahead: explicit-interface indexer `Ret IFoo<T>.this[`.
    /// True when `.this[` appears ahead with only type-ish tokens
    /// before it (names, dots, generics, `?`/`*`/`[]`/`,`). Anything
    /// else (`(`, `{`, `;`, keywords like `return`) ends the scan as
    /// false. Bounded so it always terminates.
    fn explicit_this_indexer_at(&self) -> bool {
        let mut i = 0usize;
        loop {
            if i > 64 {
                return false;
            }
            match &self.peek_at(i).kind {
                TokenKind::Eof => return false,
                TokenKind::Keyword(Keyword::This) => {
                    return i > 0
                        && self.peek_at(i - 1).kind == TokenKind::Dot
                        && self.peek_at(i + 1).kind == TokenKind::LeftBracket;
                }
                TokenKind::Identifier(_)
                | TokenKind::Dot
                | TokenKind::Less
                | TokenKind::Shl
                | TokenKind::Greater
                | TokenKind::Shr
                | TokenKind::ShrShr
                | TokenKind::Question
                | TokenKind::Star
                | TokenKind::LeftBracket
                | TokenKind::RightBracket
                | TokenKind::Comma => {}
                _ => {
                    if !self.peek_is_name(i) {
                        return false;
                    }
                }
            }
            i += 1;
        }
    }

    /// Shared `[params]` + body tail for indexers. Caller has consumed
    /// everything through `this`; the return type, access, attributes,
    /// and start span arrive as arguments.
    fn indexer_tail(
        &mut self,
        type_name: String,
        access: Access,
        attrs: Vec<Attribute>,
        start: Span,
    ) -> Option<ClassMember> {
        self.expect(
            &TokenKind::LeftBracket,
            "rubyc.parser.rd.Parser.ctx.open_indexer_params",
        )?;
        let mut params = Vec::new();
        if self.peek().kind != TokenKind::RightBracket {
            loop {
                let param = self.param()?;
                params.push(param);
                if self.eat(&TokenKind::Comma).is_none() {
                    break;
                }
            }
        }
        self.expect(
            &TokenKind::RightBracket,
            "rubyc.parser.rd.Parser.ctx.close_indexer_params",
        )?;
        let (init, get, set, get_expr, is_settable) =
            if self.peek().kind == TokenKind::LeftBrace {
                let (init, get, set, get_expr, _is_init, is_settable) =
                    self.property_accessors()?;
                (init, get, set, get_expr, is_settable)
            } else {
                self.expect(
                    &TokenKind::FatArrow,
                    "rubyc.parser.rd.Parser.ctx.open_indexer_expr",
                )?;
                let expression = self.expression()?;
                self.expect(
                    &TokenKind::Semicolon,
                    "rubyc.parser.rd.Parser.ctx.after_expression",
                )?;
                (
                    Expr::Literal(Literal::Int(0)),
                    None,
                    None,
                    Some(expression),
                    false,
                )
            };
        Some(ClassMember::Indexer(IndexerDecl {
            access,
            attrs,
            type_name,
            params,
            init,
            get,
            set,
            get_expr,
            is_settable,
            span: start.to(self.peek_at(0).span),
        }))
    }

/// Positional-record members: one public field per parameter plus a
/// constructor that stores each one (`record R(int V)` ⇒ field `V` and
/// `R(int V) { this.V = V; }`).
///
/// Without this a positional record had a parameter and NO field, so
/// `new R(7).V` read 0 and value equality compared zero fields and
/// answered true for any two instances.

    /// Split an extension receiver parameter (`MethodInfo method`,    /// `Dictionary<string, int> d`, `scoped ReadOnlySpan<byte> span`)
    /// into (type, name) on the last top-level whitespace
    /// (bracket-aware so generic commas don't split). Returns `None`
    /// for a bare type (`(MethodInfo)`) or garbage.
    fn split_receiver_param(text: &str) -> Option<(String, String)> {
        let text = text.trim();
        let bytes = text.as_bytes();
        let mut depth = 0usize;
        let mut split = None;
        let mut i = bytes.len();
        while i > 0 {
            i -= 1;
            match bytes[i] {
                b'>' | b']' | b')' => depth += 1,
                b'<' | b'[' | b'(' if depth > 0 => depth -= 1,
                b' ' | b'\t' | b'\n' | b'\r' if depth == 0 => {
                    split = Some(i);
                    break;
                }
                _ => {}
            }
        }
        let at = split?;
        let (ty, name) = (text[..at].trim(), text[at + 1..].trim());
        if name.is_empty()
            || !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return None;
        }
        // By-value parameter modifiers are not part of the type.
        let mut ty = ty;
        for prefix in ["scoped ", "ref ", "in ", "out "] {
            if let Some(rest) = ty.strip_prefix(prefix) {
                ty = rest.trim_start();
            }
        }
        if ty.is_empty() {
            return None;
        }
        Some((ty.to_owned(), name.to_owned()))
    }

    /// Extension declaration (C# 14): `extension Receiver { members }`
    /// or `extension<TParams>(ReceiverType) where ... { members }`.
    /// Members parse with the normal member parser into a synthetic
    /// nested class (`Extension_<Receiver>`) so their types and
    /// functions register in the semantic table; lowering ignores
    /// nested classes it never calls into, like all unreferenced
    /// members. Attributes on the declaration itself are dropped
    /// (the synthetic node carries none, like `Param` and `MethodSig`).
    /// Returns `None` — consuming nothing and recording no errors —
    /// when the shape does not match, so a member genuinely named
    /// `extension` (or `extension()` with an empty receiver, i.e. a
    /// default-return method) falls through to the normal gates.
    fn extension_decl(&mut self, start: Span) -> Option<ClassDecl> {
        let mark = self.pos;
        let err_mark = self.errors.len();
        match self.peek().kind.clone() {
            TokenKind::Identifier(n) if n == "extension" => {
                self.bump();
            }
            _ => {
                self.pos = mark;
                self.errors.truncate(err_mark);
                return None;
            }
        }
        // Optional type parameters `<T, ...>` (recorded: members use
        // them, e.g. `extension<T>(Vector<T>)`).
        let mut ext_generics = Vec::new();
        if matches!(
            self.peek().kind,
            TokenKind::Less | TokenKind::Shl
        ) {
            ext_generics = self.generic_parameter_list();
        }
        // Receiver `(Type)` — required, non-empty, and free of
        // top-level `=`/`;`/`,` (those mean "not a receiver").
        if self.peek().kind != TokenKind::LeftParen {
            self.pos = mark;
            self.errors.truncate(err_mark);
            return None;
        }
        let recv_text_start = self.peek().span.end as usize;
        let mut depth = 0i32;
        let mut angles = 0i32;
        let mut inner_tokens = 0u32;
        let mut ok = true;
        while self.peek().kind != TokenKind::Eof {
            match self.peek().kind {
                TokenKind::LeftParen => depth += 1,
                TokenKind::RightParen => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                // Generic commas (`Dictionary<TKey, TValue>`) are not
                // separators — only a top-level comma disqualifies.
                // `<<`/`>>` lex as Shl/Shr; comparisons inside a
                // receiver type are vanishingly rare, and a missed
                // close just fails recognition (never misparses).
                TokenKind::Less => angles += 1,
                TokenKind::Greater => angles -= 1,
                TokenKind::Shl => angles += 2,
                TokenKind::Shr => angles -= 2,
                TokenKind::Assign | TokenKind::Semicolon | TokenKind::Comma
                    if depth == 1 && angles == 0 =>
                {
                    ok = false;
                    break;
                }
                _ => {}
            }
            if depth >= 1
                && !matches!(
                    self.peek().kind,
                    TokenKind::LeftParen | TokenKind::RightParen
                )
            {
                inner_tokens += 1;
            }
            self.bump();
        }
        if !ok || inner_tokens == 0 || self.peek().kind != TokenKind::RightParen {
            self.pos = mark;
            self.errors.truncate(err_mark);
            return None;
        }
        let recv_text_end = self.peek().span.start as usize;
        self.bump(); // ')'
        // Optional `where` constraints.
        self.skip_type_constraints();
        // A body `{` is required — without it this is not an
        // extension (a method named `extension` takes it from here).
        if self.peek().kind != TokenKind::LeftBrace {
            self.pos = mark;
            self.errors.truncate(err_mark);
            return None;
        }
        self.bump(); // '{'
        let mut members = Vec::new();
        while self.peek().kind != TokenKind::RightBrace && self.peek().kind != TokenKind::Eof {
            match self.class_member() {
                Some(member) => {
                    members.push(member);
                    members.append(&mut self.pending_members);
                }
                None => self.synchronize(),
            }
        }
        if self.peek().kind != TokenKind::RightBrace {
            self.pos = mark;
            self.errors.truncate(err_mark);
            return None;
        }
        self.bump(); // '}'
        self.eat(&TokenKind::Semicolon);
        // Deterministic unique-ish name from the receiver text; the
        // semantic table registers nested classes by simple name.
        let recv_text = std::str::from_utf8(
            &self.source[recv_text_start.min(self.source.len())
                ..recv_text_end.min(self.source.len())],
        )
        .unwrap_or("");
        let recv_clean: String = recv_text
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect();
        let name = if recv_clean.is_empty() {
            "Extension".to_owned()
        } else {
            format!("Extension_{recv_clean}")
        };
        // The receiver value parameter (`extension(MethodInfo method)`)
        // is in scope for every member body — expose it as a synthetic
        // instance field so name resolution (scan and lowering) finds
        // it. Without this, uses fail (`undeclared variable 'method'`).
        // Split on the last top-level whitespace (bracket-aware, so
        // `Dictionary<string, int> d` keeps its generic comma).
        if let Some((recv_ty, recv_name)) = Self::split_receiver_param(recv_text) {
            if recv_name != "_" {
                members.insert(
                    0,
                    ClassMember::Field(FieldDecl {
                        access: Access::Default,
                        attrs: Vec::new(),
                        type_name: recv_ty,
                        name: recv_name,
                        init: Expr::Literal(Literal::Int(0)),
                        init_pre: Vec::new(),
                        is_const: false,
                        is_static: false,
                        is_byref: false,
                        span: start.to(self.peek_at(0).span),
                    }),
                );
            }
        }
        Some(ClassDecl {
            name,
            generic_params: ext_generics,
            primary_ctor_params: Vec::new(),
            bases: Vec::new(),
            members,
            span: start.to(self.peek_at(0).span),
            is_partial: false,
            // Synthetic extension holder: reference semantics.
            is_struct: false, is_record: false,
            union_cross_assembly: false,
        })
    }

    /// Class members: attributes, then fields or methods.
    pub(super) fn class_member(&mut self) -> Option<ClassMember> {        // Modifiers and attribute lists may appear in either order:
        // `public [Inline] int f()` or `[Inline] public int f()`.
        let mut access = Access::Default;
        let mut attrs: Vec<Attribute> = Vec::new();
        loop {
            let before = self.pos;
            let a = self.access_modifier();
            if a != Access::Default {
                access = a;
            }
            attrs.extend(self.attribute_lists());
            if self.pos == before {
                break;
            }
        }
        let start = self.peek().span;

        // Declaration modifiers apply to both fields and methods. Consume
        // them before deciding which member shape follows.
        let mut is_override = false;
        let mut is_virtual = false;
        let mut is_sealed = false;
        let mut is_hiding = false;
        let mut is_const = false;
        let mut is_static = false;
        let mut is_abstract = false;
        let mut is_partial = false;
        let mut is_async = false;
        let mut is_extern = false;
        // `ref` modifier (`ref T x` — ref fields, ref returns): the
        // modifiers loop below consumes it with the ignored
        // modifiers; capture it here so field declarations can
        // record address-holding slots.
        let mut is_byref = false;
        loop {
            match self.peek().kind {
                TokenKind::Keyword(Keyword::Override) => {
                    is_override = true;
                    self.bump();
                }
                TokenKind::Keyword(Keyword::Static) => {
                    is_static = true;
                    self.bump();
                }
                TokenKind::Keyword(Keyword::Const) => {
                    is_const = true;
                    self.bump();
                }
                TokenKind::Keyword(Keyword::Abstract) => {
                    is_abstract = true;
                    self.bump();
                }
                TokenKind::Keyword(Keyword::Partial) => {
                    is_partial = true;
                    self.bump();
                }
                TokenKind::Keyword(Keyword::Async) => {
                    is_async = true;
                    self.bump();
                }
                TokenKind::Keyword(Keyword::Internal) => {
                    // `internal` is assembly-visible; the backend has no
                    // assembly concept, so treat it as `public`.
                    access = Access::Public;
                    self.bump();
                }
                // `virtual` / `sealed` / `new` decide how a call
                // binds: whether it follows the receiver's runtime type,
                // whether it is fixed to the declaring type, and whether
                // it hides rather than overrides. Recorded for the
                // dispatch decision; the remaining modifiers
                // here carry no runtime meaning for a method.
                TokenKind::Keyword(Keyword::Virtual) => {
                    is_virtual = true;
                    self.bump();
                }
                TokenKind::Keyword(Keyword::Sealed) => {
                    is_sealed = true;
                    self.bump();
                }
                TokenKind::Keyword(Keyword::New) => {
                    is_hiding = true;
                    self.bump();
                }
                TokenKind::Keyword(Keyword::Readonly)
                | TokenKind::Keyword(Keyword::Out_)
                | TokenKind::Keyword(Keyword::In_)
                | TokenKind::Keyword(Keyword::Required)
                | TokenKind::Keyword(Keyword::Unsafe)
                | TokenKind::Keyword(Keyword::File)
                | TokenKind::Keyword(Keyword::Fixed)
                | TokenKind::Keyword(Keyword::Volatile) => {
                    self.bump();
                }
                // `ref` modifier is meaningful (ref fields hold
                // addresses; ref returns are addresses) — record it
                // instead of discarding with the ignored modifiers.
                TokenKind::Keyword(Keyword::Ref_) => {
                    is_byref = true;
                    self.bump();
                }
                TokenKind::Keyword(Keyword::Extern) => {
                    // Recorded (not discarded): bodyless `extern`
                    // methods are runtime-provided (InternalCall/QCall),
                    // valid without `[LibraryImport]`.
                    is_extern = true;
                    self.bump();
                }
                // `safe` (safe-context declarations). It is a
                // compile-time escape-analysis annotation with no
                // runtime meaning, so consuming it changes nothing
                // about the program — the same treatment `scoped` gets.
                // Deliberately NOT a lexer keyword: a variable may
                // legally be named `safe`, so the identifier is only
                // taken as a modifier when a type or `partial`
                // follows it.
                TokenKind::Identifier(ref n)
                    if n == "safe"
                        && matches!(
                            self.peek_at(1).kind,
                            TokenKind::Identifier(_)
                                | TokenKind::Keyword(Keyword::Partial)
                        ) =>
                {
                    self.bump();
                }
                // `closed` (C# 14 closed hierarchies). The spelling
                // gate is the same idea as `safe` above. Unlike `safe`
                // this one DOES change what the language accepts —
                // nobody may derive from a closed type — and nothing
                // here enforces it. Silently dropping it means a
                // program C# rejects compiles here, so say so instead.
                TokenKind::Identifier(ref n)
                    if n == "closed"
                        && matches!(
                            self.peek_at(1).kind,
                            TokenKind::Keyword(Keyword::Class)
                                | TokenKind::Keyword(Keyword::Struct_)
                                | TokenKind::Keyword(Keyword::Interface)
                                | TokenKind::Keyword(Keyword::Record_)
                        ) =>
                {
                    let span = self.peek().span;
                    self.bump();
                    self.errors.push(
                        Diagnostic::new(
                            "W0004",
                            Severity::Warning,
                            crate::t!("rubyc.parser.rd.declaration.closed_not_enforced"),
                        )
                        .at(span)
                        .in_context(self.context.clone()),
                    );
                }
                _ => break,
            }
        }

        // Extension declaration (C# 14): `extension Receiver { ... }`
        // or `extension<TParams>(ReceiverType) where ... { ... }`.
        // `extension` is not a lexer keyword, so match its identifier
        // text; anything else shaped (a member genuinely named
        // `extension`, `extension()` with an empty receiver, ...) falls
        // through to the gates below untouched.
        if matches!(self.peek().kind, TokenKind::Identifier(ref n) if n == "extension") {
            if let Some(decl) = self.extension_decl(start) {
                return Some(ClassMember::Nested(crate::models::Member::Class(decl)));
            }
        }

        // Nested type declaration: `class`, `struct`, `interface`, `enum`,
        // `record`, `delegate`. A `delegate` followed by `*` is a
        // function-pointer TYPE (`delegate* ...`), not a declaration.
        if (matches!(
            self.peek().kind,
            TokenKind::Keyword(Keyword::Class)
                | TokenKind::Keyword(Keyword::Struct_)
                | TokenKind::Keyword(Keyword::Interface)
                | TokenKind::Keyword(Keyword::Enum_)
                | TokenKind::Keyword(Keyword::Record_)
        ) || (matches!(self.peek().kind, TokenKind::Keyword(Keyword::Delegate_))
            && self.peek_at(1).kind != TokenKind::Star))
        {
            if let Some(mut member) = self.member() {
                // `partial` was consumed by this member's modifier loop
                // above, so `member()` never sees it: restore it onto
                // nested class/struct/interface decls (parsed as
                // classes / interfaces). Without this, same-key nested
                // `partial` parts never merge and one part's members
                // vanish downstream (`AsyncProfiler.AsyncCallstack`
                // across the libraries and CoreCLR parts;
                // `IDtdParserAdapter` across its sync/async files).
                if is_partial {
                    if let crate::models::Member::Class(decl) = &mut member {
                        decl.is_partial = true;
                    }
                    if let crate::models::Member::Interface(decl) = &mut member {
                        decl.is_partial = true;
                    }
                }
                return Some(ClassMember::Nested(member));
            }
        }

        // Tuple-typed member: `(T1 n1, T2 n2)[] Name ...`. Fold the
        // tuple type (plus suffixes) into a single synthetic identifier
        // token so the field/property/method gates below — which only
        // accept identifier-start types — proceed unchanged. The text
        // is the exact source slice.
        if matches!(self.peek().kind, TokenKind::LeftParen) {
            if let Some(name_idx) = self.tuple_type_shape_at() {
                let type_start = self.peek().span.start as usize;
                let type_end =
                    self.tokens[self.pos + name_idx - 1].span.end as usize;
                let text = std::str::from_utf8(
                    &self.source[type_start..type_end.min(self.source.len())],
                )
                .unwrap_or("tuple")
                .to_owned();
                let span = self.peek().span.to(
                    self.tokens[self.pos + name_idx - 1].span,
                );
                self.tokens.splice(
                    self.pos..self.pos + name_idx,
                    [crate::models::Token {
                        kind: TokenKind::Identifier(text),
                        span,
                    }],
                );
            }
        }

        // Operator: `Type operator [checked] <op> (params) { … }` or
        // `Type operator [checked] <op> (params) => expr;`.
        // Conversion operator: `[implicit|explicit] operator Target(params)`.
        // The return type is parsed speculatively (it may carry
        // generics: `Complex<T> operator - (...)`); anything but
        // `operator` after it rewinds to the other member gates.
        // Explicit interface implementations qualify the operator
        // (`IAdditionOperators<T,T,T>.operator +(...)`): interface
        // segments (each with optional generics) are consumed and
        // dropped, like other qualifications.
        let mut op_ret: Option<String> = None;
        if self.is_type_start() {
            let mark = self.pos;
            let err_mark = self.errors.len();
            if let Some(t) = self.type_name() {
                // Explicit implementation qualification:
                // `Iface<T>.Sub...` segments between the return type
                // and `operator` (`IAdditionOperators<T,T,T>.operator
                // +(...)`). Each segment is a name with optional
                // generics; the scan stops at the first non-`Name.`
                // step, and anything but `operator` after it rewinds.
                loop {
                    if !self.peek_is_name(0) {
                        break;
                    }
                    self.bump(); // interface segment
                    if matches!(
                        self.peek().kind,
                        TokenKind::Less | TokenKind::Shl
                    ) {
                        self.skip_generic_parameter_list();
                    }
                    if self.peek().kind == TokenKind::Dot {
                        self.bump(); // '.'
                        continue;
                    }
                    break;
                }
                if matches!(
                    self.peek().kind,
                    TokenKind::Keyword(Keyword::Operator_)
                ) {
                    op_ret = Some(t);
                } else {
                    self.pos = mark;
                    self.errors.truncate(err_mark);
                }
            } else {
                self.pos = mark;
                self.errors.truncate(err_mark);
            }
        }
        if let Some(ret_type) = op_ret {
            self.bump(); // 'operator'
            // Optional `checked` prefix.
            if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Checked)) {
                self.bump();
            }
            // Conversion target (`implicit operator Target(params)`):
            // a type, not an operator symbol. Try it first: a conversion
            // target is always followed by the parameter list `(`.
            // Otherwise rewind for the operator-symbol path below.
            let op_name;
            {
                let mark = self.pos;
                let err_mark = self.errors.len();
                if let Some(target) = self.type_name() {
                    if self.peek().kind == TokenKind::LeftParen {
                        // The "return type" is really the conversion
                        // modifier (`implicit`/`explicit operator
                        // Target`): record it in the name so overload
                        // resolution can find implicit conversions
                        // (`DelayedQName.implicit operator string`).
                        // Explicit ones are recorded distinctly (casts
                        // only; inapplicable to implicit calls).
                        op_name = if ret_type == "implicit" {
                            format!("conversion_implicit_{target}")
                        } else if ret_type == "explicit" {
                            format!("conversion_explicit_{target}")
                        } else {
                            format!("conversion_{target}")
                        };
                    } else {
                        self.pos = mark;
                        self.errors.truncate(err_mark);
                        let op_token = self.peek().clone();
                        op_name = format!("{:?}", op_token.kind);
                        self.bump();
                    }
                } else {
                    self.pos = mark;
                    self.errors.truncate(err_mark);
                    let op_token = self.peek().clone();
                    op_name = format!("{:?}", op_token.kind);
                    self.bump();
                }
            }
            // Parse params and body manually (method_decl expects to parse
            // the return type and name itself).
            self.expect(&TokenKind::LeftParen, "to open operator parameter list")?;
            let params = self.param_list();
            self.expect(&TokenKind::RightParen, "to close operator parameter list")?;
            let body = if self.eat(&TokenKind::Semicolon).is_some() {
                None
            } else if self.eat(&TokenKind::FatArrow).is_some() {
                let expression = self.expression()?;
                let end = self
                    .expect(&TokenKind::Semicolon, "after expression-bodied member")
                    .map_or(self.peek().span, |token| token.span);
                Some(self.expression_body(
                    crate::models::Stmt::Return {
                        value: Some(expression),
                        span: end,
                    },
                    start.to(end),
                ))
            } else {
                Some(self.block())
            };
            let end = body
                .as_ref()
                .map(|b| b.span)
                .unwrap_or_else(|| self.peek_at(0).span);
            return Some(ClassMember::Method(MethodDecl {
                access,
                attrs,
                // Operators cannot declare type parameters.
                generic_params: Vec::new(),
                constraints: Vec::new(),
                ret_type,
                name: op_name,
                params,
                body,
                is_override,
                is_virtual: true,
                is_sealed: false,
                is_hiding: false,
                is_static,
                is_abstract: false,
                is_partial: false,
                is_async: false,
                // Operators are never `extern` (checked by the E0019
                // gate if one ever appears bodyless).
                is_extern: false,
                base_call: None,
                span: start.to(end),
            }));
        }

        // Event: `public event IntUnary? Changed;`, explicit
        // interface form `event T IFoo.Bar { add => ...; ... }`
        // (qualification recorded, mirroring method names).
        if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Event_)) {
            self.bump(); // 'event'
            let type_name = self.type_name_full()?;
            let mut name = self.expect_identifier("rubyc.parser.rd.Parser.ctx.as_event_name")?;
            while self.eat(&TokenKind::Dot).is_some() {
                match self.expect_identifier("rubyc.parser.rd.Parser.ctx.as_event_name") {
                    Some(seg) => {
                        name.push('.');
                        name.push_str(&seg);
                    }
                    None => break,
                }
            }
            // Custom accessors: `event T N { add { ... } remove { ... } }`.
            // The accessor bodies wire add/remove semantics the backend
            // does not model; events lower as fields, so skip the block.
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
            } else {
                self.expect(
                    &TokenKind::Semicolon,
                    "rubyc.parser.rd.Parser.ctx.after_statement",
                )?;
            }
            return Some(ClassMember::Event(EventDecl {
                access,
                attrs,
                type_name,
                name,
                span: start.to(self.peek_at(0).span),
            }));
        }

        // Indexer: `public int this[int index] { get; set; }` or
        // `public int this[int index] => expr;`. `this` follows the
        // return type, which may be dotted (`Ns.Type this[...]`),
        // generic (`IEnumerable<T>`), `global::`-qualified, and carry
        // `?`/`*` suffixes (`string? this[string key]`).
        let mut tpos = 1usize;
        if matches!(self.peek().kind, TokenKind::Identifier(_)) {
            loop {
                if matches!(
                    self.peek_at(tpos).kind,
                    TokenKind::Less | TokenKind::Shl
                ) {
                    tpos = self.skip_generic_at(tpos);
                }
                if self.peek_at(tpos).kind == TokenKind::Dot
                    && (matches!(self.peek_at(tpos + 1).kind, TokenKind::Identifier(_))
                        || self.peek_is_name(tpos + 1))
                {
                    tpos += 2;
                    continue;
                }
                if self.peek_at(tpos).kind == TokenKind::Colon
                    && self.peek_at(tpos + 1).kind == TokenKind::Colon
                    && matches!(self.peek_at(tpos + 2).kind, TokenKind::Identifier(_))
                {
                    tpos += 3;
                    continue;
                }
                break;
            }
            tpos = self.skip_suffixes_at(tpos);
        }
        let next_is_this = matches!(self.peek().kind, TokenKind::Identifier(_))
            && matches!(
                self.peek_at(tpos).kind,
                TokenKind::Keyword(Keyword::This)
            );
        if next_is_this {
            let type_name = self.type_name()?;
            self.bump(); // 'this'
            return self.indexer_tail(type_name, access, attrs, start);
        }

        // Explicit interface indexer: `RetType IFoo<T>.this[P] ...`.
        // The qualification is consumed (not recorded — the compact
        // model names indexers by shape); the `[params]` + body tail is
        // identical to the plain form above.
        if self.explicit_this_indexer_at() {
            let mark = self.pos;
            let err_mark = self.errors.len();
            if let Some(type_name) = self.type_name() {
                // Consume `IFoo<T>.` segments; stop at `.this`.
                // (`peek_is_name` covers identifiers + contextual
                // keywords.) The gate pre-validated the shape; `ok`
                // only guards against desync.
                let mut ok = true;
                while ok {
                    if self.peek_is_name(0) {
                        self.bump();
                    } else {
                        ok = false;
                        break;
                    }
                    if matches!(
                        self.peek().kind,
                        TokenKind::Less | TokenKind::Shl
                    ) {
                        self.skip_generic_parameter_list();
                    }
                    while matches!(
                        self.peek().kind,
                        TokenKind::Question | TokenKind::Star
                    ) {
                        self.bump();
                    }
                    if self.eat(&TokenKind::Dot).is_some() {
                        if matches!(
                            self.peek().kind,
                            TokenKind::Keyword(Keyword::This)
                        ) {
                            break;
                        }
                        continue;
                    }
                    ok = false;
                    break;
                }
                if ok
                    && matches!(
                        self.peek().kind,
                        TokenKind::Keyword(Keyword::This)
                    )
                {
                    self.bump(); // 'this'
                    return self.indexer_tail(type_name, access, attrs, start);
                }
            }
            // Desync guard: the gate pre-validated, but never corrupt
            // the stream — rewind and let the other gates try.
            self.pos = mark;
            self.errors.truncate(err_mark);
        }

        // Fixed-size array field: `Type Name[size];`.
        if matches!(self.peek().kind, TokenKind::Identifier(_))
            && self.likely_fixed_array_field()
        {
            let type_name = self.type_name_full()?;
            let name = self.expect_identifier("rubyc.parser.rd.Parser.ctx.as_variable_name")?;
            self.bump(); // '['
            let _size = self.expression()?;
            self.expect(&TokenKind::RightBracket, "to close fixed array size")?;
            self.expect(
                &TokenKind::Semicolon,
                "rubyc.parser.rd.Parser.ctx.after_statement",
            )?;
            return Some(ClassMember::Field(FieldDecl {
                access,
                attrs,
                type_name,
                name,
                init: Expr::Literal(Literal::Int(0)),
                init_pre: Vec::new(),
                is_const,
                is_static,
                is_byref: false,
                span: start.to(self.peek_at(0).span),
            }));
        }

        // Field: `Type name [= expr] ;`. A `=>` at position 2 is an
        // expression-bodied property, not a field initializer, so it must not
        // match here (handled by the property gate below).
        // The type may have array (`[]`) or nullable (`?`) suffixes.
        let name_offset = self.likely_type_then_name();
        if let Some(noff) = name_offset {
            let after_name = self.peek_at(noff + 1).kind.clone();
            let is_field = match after_name {
                TokenKind::Semicolon => true,
                // Multi-declarator fields: `string? a, b, c;`.
                TokenKind::Comma => true,
                TokenKind::Assign => !matches!(self.peek_at(noff + 2).kind, TokenKind::Greater),
                _ => false,
            };
            if is_field {
                let type_name = self.type_name_full()?;
                let name = self.expect_identifier("rubyc.parser.rd.Parser.ctx.as_variable_name")?;
                let init = if self.eat(&TokenKind::Assign).is_some() {
                    self.expression()?
                } else {
                    Expr::Literal(Literal::Int(0))
                };
                // Bindings declared while parsing the initializer
                // (`out` vars, `is` patterns) have no statement list
                // to drain into here — they ride on the field and
                // lower first.
                let init_pre = std::mem::take(&mut self.pre_pending);
                // Trailing declarators (`, b [= e], ...`) share the
                // type, access, attributes, and flags; each becomes
                // its own member, drained by the class body loop.
                while self.eat(&TokenKind::Comma).is_some() {
                    let extra = self.expect_identifier(
                        "rubyc.parser.rd.Parser.ctx.as_variable_name",
                    )?;
                    let extra_init = if self.eat(&TokenKind::Assign).is_some() {
                        self.expression()?
                    } else {
                        Expr::Literal(Literal::Int(0))
                    };
                    let extra_pre = std::mem::take(&mut self.pre_pending);
                    self.pending_members.push(ClassMember::Field(FieldDecl {
                        access,
                        attrs: attrs.clone(),
                        type_name: type_name.clone(),
                        name: extra,
                        init: extra_init,
                        init_pre: extra_pre,
                        is_const,
                        is_static,
                        is_byref,
                        span: start.to(self.peek_at(0).span),
                    }));
                }
                self.expect(
                    &TokenKind::Semicolon,
                    "rubyc.parser.rd.Parser.ctx.after_statement",
                )?;
                return Some(ClassMember::Field(FieldDecl {
                    access,
                    attrs,
                    type_name,
                    name,
                    init,
                    init_pre,
                    is_const,
                    is_static,
                    is_byref,
                    span: start.to(self.peek_at(0).span),
                }));
            }
        }

        // Property: `Type name { accessors }` or `Type name => expr;`.
        // The field gate above only matches `=` not followed by `>`, so `=>`
        // falls through here. Accessor blocks begin with `get`/`set`/`init`.
        // The type may have array (`[]`) or nullable (`?`) suffixes.
        let prop_name_offset = self.likely_type_then_name();
        if let Some(noff) = prop_name_offset {
            let after_name = self.peek_at(noff + 1).kind.clone();
            let is_property = matches!(
                after_name,
                TokenKind::LeftBrace | TokenKind::Assign | TokenKind::FatArrow
            );
            if is_property {
                let type_name = self.type_name_full()?;
                let name = self.qualified_name()?;
            let (init, get, set, get_expr, is_init, is_settable) =
                if self.peek().kind == TokenKind::LeftBrace {
                    self.property_accessors()?
                } else {
                    // Expression-bodied: `Type name => expr;`
                    // Parse as an expression: it handles plain values,
                    // ternaries (`HasRange ? a : b`), and compound
                    // assignments (`x ??= rhs`). Parsing as a statement
                    // misreads a leading `Ident ?` as a nullable-type
                    // declaration.
                    self.expect(&TokenKind::FatArrow, "to begin accessor list")?;
                    let expression = self.expression()?;
                    self.expect(
                        &TokenKind::Semicolon,
                        "after expression-bodied member",
                    );
                    (
                        Expr::Literal(Literal::Int(0)),
                        None,
                        None,
                        Some(expression),
                        false,
                        false,
                    )
                };
            let end = self.peek_at(0).span;
            return Some(ClassMember::Property(PropertyDecl {
                access,
                attrs,
                type_name,
                name,
                init,
                get,
                set,
                get_expr,
                is_init,
                is_settable,
                is_static,
                is_abstract,
                is_override,
                is_virtual,
                is_hiding,
                span: start.to(end),
            }));
            }
        }

        // `virtual` / `override`: documentation for dispatch (name-based),
        // but `override` suppresses a second export of the same symbol.
        // Destructor: `~Name() { body }` — name must match the class.
        if matches!(self.peek().kind, TokenKind::Tilde) {
            self.bump(); // '~'
            let name = self.expect_identifier("as destructor name")?;
            let expected = self.context.class.clone().unwrap_or_default();
            if name != expected {
                let span = self.peek().span;
                self.error(
                    "E0024",
                    format!(
                        "syntax error: destructor '~{name}()' does not match class '{expected}'"
                    ),
                    span,
                );
            }
            self.expect(&TokenKind::LeftParen, "to open parameter list")?;
            self.expect(&TokenKind::RightParen, "to close parameter list")?;
            // Destructors never return a value; ret_type "" marks that.
            // Expression-bodied form (`~C() => stmt;`) runs for effect.
            let body = if self.eat(&TokenKind::FatArrow).is_some() {
                let expr = self.expression()?;
                let end = self
                    .expect(&TokenKind::Semicolon, "after expression-bodied destructor")
                    .map_or(self.peek().span, |token| token.span);
                Some(self.expression_body(
                    crate::models::Stmt::Expression(expr),
                    start.to(end),
                ))
            } else {
                Some(self.block())
            };
            let end = body.as_ref().map(|b| b.span).unwrap_or(start);
            return Some(ClassMember::Method(MethodDecl {
                access,
                attrs,
                // Destructors cannot declare type parameters.
                generic_params: Vec::new(),
                constraints: Vec::new(),
                ret_type: String::new(),
                name: format!("~{name}"),
                params: Vec::new(),
                body,
                is_override: false,
                is_virtual: false,
                is_sealed: false,
                is_hiding: false,
                is_static: false,
                is_abstract: false,
                is_partial: false,
                is_async: false,
                // Destructors always have bodies; never `extern`.
                is_extern: false,
                base_call: None,
                span: end,
            }));
        }

        // Method: `[RetType] Name(params) { body }` — omitted type defaults to int.
        self.method_decl(
            access,
            attrs,
            start,
            is_override,
            is_virtual,
            is_sealed,
            is_hiding,
            is_static,
            is_abstract,
            is_partial,
            is_async,
            is_extern,
        )
        .map(ClassMember::Method)
    }

    /// Zero or more `[Name]` / `[Name(arg, Key = value)]` attribute lists.
    pub(super) fn attribute_lists(&mut self) -> Vec<Attribute> {
        let mut attrs = Vec::new();
        while self.peek().kind == TokenKind::LeftBracket {
            let open = self.bump();
            // Attribute target specifier: `[return: Attr]`, `[param: ...]`,
            // `[field: ...]`, etc. The target only selects where the
            // attribute applies; consume and discard it. Single colon
            // only — `global::X` (double colon) is a qualifier, not a
            // target (`[global::System.ComponentModel.EditorBrowsable...]`
            // in Composition sources).
            if matches!(
                self.peek().kind,
                TokenKind::Identifier(_) | TokenKind::Keyword(_)
            ) && self.peek_at(1).kind == TokenKind::Colon
                && self.peek_at(2).kind != TokenKind::Colon
            {
                self.bump();
                self.bump(); // ':'
            }
            // Transparent `global::` qualifier on the attribute name.
            if matches!(&self.peek().kind, TokenKind::Identifier(n) if n == "global")
                && self.peek_at(1).kind == TokenKind::Colon
                && self.peek_at(2).kind == TokenKind::Colon
            {
                self.bump();
                self.bump();
                self.bump(); // 'global', ':', ':'
            }
            // Comma-separated attributes share one bracket in C#:
            // `[A(...), B(...)]`.
            loop {
                let Some(mut name) = (match self.peek().kind.clone() {
                    TokenKind::Identifier(n) => {
                        self.bump();
                        Some(n)
                    }
                    TokenKind::Keyword(kw) => {
                        self.bump();
                        Some(format!("{kw:?}").to_lowercase())
                    }
                    _ => None,
                }) else {
                    let found = describe(self.peek());
                    self.unexpected(
                        "rubyc.parser.rd.Parser.ctx.attribute_name",
                        &found,
                        self.peek().span,
                    );
                    return attrs;
                };
                // Dotted segments: `System.Runtime.CompilerServices`
                while self.eat(&TokenKind::Dot).is_some() {
                    name.push('.');
                    match self.expect_identifier("attribute name segment") {
                        Some(seg) => name.push_str(&seg),
                        None => return attrs,
                    }
                }
                self.skip_generic_parameter_list();
                let mut arguments = Vec::new();
                if self.eat(&TokenKind::LeftParen).is_some() {
                    loop {
                        if self.peek().kind == TokenKind::RightParen {
                            break;
                        }
                        // Named argument? `Ident = value` or `Ident: value`
                        // (both forms are legal C#, e.g.
                        // `[NotNullWhen(returnValue: true)]`). A double
                        // colon is a `global::` qualifier, not a named
                        // argument (`global::System...` enum values).
                        let arg_name = match self.peek().kind.clone() {
                            TokenKind::Identifier(n)
                                if matches!(
                                    self.peek_at(1).kind,
                                    TokenKind::Assign | TokenKind::Colon
                                ) && !(self.peek_at(1).kind == TokenKind::Colon
                                    && self.peek_at(2).kind == TokenKind::Colon) =>
                            {
                                self.bump();
                                self.bump(); // '=' or ':'
                                Some(n)
                            }
                            _ => None,
                        };
                        match self.attribute_value() {
                            Some((value, span)) => arguments.push(AttributeArg {
                                name: arg_name,
                                value,
                                span,
                            }),
                            None => break,
                        }
                        if self.eat(&TokenKind::Comma).is_none() {
                            break;
                        }
                    }
                    self.expect(&TokenKind::RightParen, "to close attribute arguments");
                }
                attrs.push(Attribute {
                    name,
                    arguments,
                    span: open.span.to(self.peek().span),
                });
                // Another attribute in the same bracket (`[A, B]`)?
                // A trailing comma (`[A(x),]`, as in Odbc sources) is
                // legal C# — close instead of demanding another name.
                if self.eat(&TokenKind::Comma).is_some() {
                    if self.peek().kind == TokenKind::RightBracket {
                        self.bump();
                        break;
                    }
                    continue;
                }
                if self
                    .expect(&TokenKind::RightBracket, "to close attribute")
                    .is_none()
                {
                    return attrs;
                }
                break;
            }
        }
        attrs
    }

    pub(super) fn attribute_value(&mut self) -> Option<(AttributeValue, Span)> {
        let token = self.peek().clone();
        // Parenthesized group: a cast (`(int)X`) or a grouping
        // (`(a + b)`). Consume balanced parens, then an operand if one
        // follows (casts always have one; tuples stand alone). Produces
        // the value directly, then joins the shared tail loop below.
        let first = if token.kind == TokenKind::LeftParen {
            self.bump(); // '('
            let mut depth = 1i32;
            while depth > 0 && self.peek().kind != TokenKind::Eof {
                match self.peek().kind {
                    TokenKind::LeftParen => depth += 1,
                    TokenKind::RightParen => depth -= 1,
                    _ => {}
                }
                self.bump();
            }
            match self.peek().kind {
                TokenKind::Comma
                | TokenKind::RightParen
                | TokenKind::RightBracket
                | TokenKind::Semicolon
                | TokenKind::Eof => {}
                _ => {
                    let _ = self.attribute_value();
                }
            }
            Some((AttributeValue::Str("".to_owned()), token.span))
        } else {
            match &token.kind {
            TokenKind::StringLiteral(s) => {
                self.bump();
                Some((AttributeValue::Str(s.clone()), token.span))
            }
            // Interpolated strings (`$"..."`) are single lexer tokens;
            // accept like plain strings (e.g. `Justification =
            // $"... ({nameof(X)}) ..."`).
            TokenKind::InterpolatedString { raw, .. } => {
                self.bump();
                Some((AttributeValue::Str(raw.clone()), token.span))
            }
            TokenKind::LeftBracket => {
                // Collection value: `[typeof(X), ...]` (e.g.
                // `CallConvs = [typeof(CallConvCdecl)]`). Values are inert
                // metadata — balance the brackets, discard the contents.
                self.bump(); // '['
                let mut depth = 1usize;
                while depth > 0 && self.peek().kind != TokenKind::Eof {
                    match self.peek().kind {
                        TokenKind::LeftBracket => depth += 1,
                        TokenKind::RightBracket => depth -= 1,
                        _ => {}
                    }
                    self.bump();
                }
                Some((AttributeValue::Str("".to_owned()), token.span))
            }
            TokenKind::IntLiteral(v) => {
                self.bump();
                Some((AttributeValue::Int(*v as i64), token.span))
            }
            // A char attribute argument (`[DefaultValue(' ')]`): the code
            // unit is the value, and the parameter's declared type is char.
            TokenKind::CharLiteral(v) => {
                self.bump();
                Some((AttributeValue::Int(*v as i64), token.span))
            }
            // Float literal (`DefaultValue(100d)` in Timer sources).
            TokenKind::FloatLiteral(bits) => {
                self.bump();
                Some((AttributeValue::Float(*bits), token.span))
            }
            // Signed constant: `[OverloadResolutionPriority(-1)]`.
            // C# attribute arguments allow unary `-`/`+` on constants.
            TokenKind::Minus | TokenKind::Plus
                if matches!(
                    self.peek_at(1).kind,
                    TokenKind::IntLiteral(_) | TokenKind::FloatLiteral(_)
                ) =>
            {
                let negative = self.peek().kind == TokenKind::Minus;
                self.bump(); // sign
                let value = match self.bump().kind {
                    TokenKind::IntLiteral(v) => {
                        AttributeValue::Int(if negative { -(v as i64) } else { v as i64 })
                    }
                    TokenKind::FloatLiteral(bits) => AttributeValue::Float(
                        if negative {
                            (-f64::from_bits(bits)).to_bits()
                        } else {
                            bits
                        },
                    ),
                    _ => unreachable!(),
                };
                Some((value, token.span))
            }
            TokenKind::Keyword(Keyword::New) => {
                // Object/array creation (`new[] {...}`, `new int[] {...}`,
                // `new Foo(...)`): attribute values are inert metadata, so
                // consume an optional dotted type with generics, then every
                // balanced bracket group that follows. Without this, `new`
                // falls into the name arm below and the trailing `[`
                // breaks argument parsing.
                self.bump(); // 'new'
                // Optional type: a leading identifier (`new int[]`, `new
                // Foo`) plus dotted continuations (`new A.B.C(...)`).
                // `new[]` / `new(...)` skip straight to the groups below.
                if matches!(self.peek().kind, TokenKind::Identifier(_)) {
                    self.bump();
                    while self.peek().kind == TokenKind::Dot {
                        self.bump();
                        if !matches!(self.peek().kind, TokenKind::Identifier(_)) {
                            break;
                        }
                        self.bump();
                    }
                }
                if self.peek().kind == TokenKind::Less {
                    self.skip_generic_parameter_list();
                }
                loop {
                    let (open, close) = match self.peek().kind {
                        TokenKind::LeftBrace => (TokenKind::LeftBrace, TokenKind::RightBrace),
                        TokenKind::LeftParen => (TokenKind::LeftParen, TokenKind::RightParen),
                        TokenKind::LeftBracket => (TokenKind::LeftBracket, TokenKind::RightBracket),
                        _ => break,
                    };
                    self.bump();
                    let mut depth = 1usize;
                    while depth > 0 && self.peek().kind != TokenKind::Eof {
                        if self.peek().kind == open {
                            depth += 1;
                        } else if self.peek().kind == close {
                            depth -= 1;
                        }
                        self.bump();
                    }
                }
                Some((AttributeValue::Str("".to_owned()), token.span))
            }
            TokenKind::Identifier(_) | TokenKind::Keyword(_) => {
                // Consume a (possibly dot-qualified) name, optional generic
                // list, and optional call args: `A.B.C`, `A.B<int>`,
                // `A.B.C(x)`, `nameof(x)`, etc. A leading `global::`
                // qualifier is transparent (generated code passes
                // `global::System...` enum values as attribute args).
                if matches!(&token.kind, TokenKind::Identifier(n) if n == "global")
                    && self.peek_at(1).kind == TokenKind::Colon
                    && self.peek_at(2).kind == TokenKind::Colon
                {
                    self.bump();
                    self.bump();
                    self.bump(); // 'global', ':', ':'
                }
                self.bump();
                // Dot-qualified segments: `A.B.C`.
                while self.peek().kind == TokenKind::Dot {
                    self.bump();
                    if !matches!(self.peek().kind, TokenKind::Identifier(_)) {
                        break;
                    }
                    self.bump();
                }
                // Generic argument list: `A<int, string>`.
                if self.peek().kind == TokenKind::Less {
                    self.skip_generic_parameter_list();
                }
                if self.eat(&TokenKind::LeftParen).is_some() {
                    let mut depth = 1usize;
                    while depth > 0 && self.peek().kind != TokenKind::Eof {
                        match self.peek().kind {
                            TokenKind::LeftParen => depth += 1,
                            TokenKind::RightParen => depth -= 1,
                            _ => {}
                        }
                        self.bump();
                    }
                }
                Some((AttributeValue::Str("".to_owned()), token.span))
            }
            _ => None,
            }
        };
        // Binary tails for EVERY value shape (`A | B` enum flags,
        // `1 * N` constant arithmetic, `-1 + X`, ...): attribute values
        // are inert metadata, so consume `op value` tails with the
        // discard semantics. Stops at `,`/`)`/`]` (argument ends).
        if first.is_some() {
            while matches!(
                self.peek().kind,
                TokenKind::Pipe
                    | TokenKind::Amp
                    | TokenKind::Plus
                    | TokenKind::Minus
                    | TokenKind::Star
                    | TokenKind::Slash
                    | TokenKind::Percent
                    | TokenKind::Caret
                    | TokenKind::Shl
                    | TokenKind::Shr
                    | TokenKind::ShrShr
            ) {
                self.bump();
                // The operand is another value of the same shape;
                // recursion keeps nesting (`A | B | C`) working.
                if self.attribute_value().is_none() {
                    break;
                }
            }
        }
        first
    }

    /// Consume a generic argument/parameter list while preserving the compact
    /// AST's base type name. Semantic generic modeling is deliberately kept
    /// separate from recognizing otherwise valid declarations.
    pub(super) fn skip_generic_parameter_list(&mut self) {
        let _ = self.generic_parameter_count();
    }

    /// Consume an optional generic argument list (`<T, U, ...>`) and
    /// return its NORMALIZED source text (`<T,U>` — interior whitespace
    /// stripped so spellings compare equal site-independently, e.g.
    /// `Predicate<X509Certificate2>?` keeps its argument for overload
    /// ranking instead of degrading to `Predicate?`). Returns `None`
    /// when no list follows. Balances nesting and accepts
    /// lexer-merged tokens (`<<`/`>>`/`>>>`), mirroring
    /// `generic_parameter_count`.
    pub(super) fn capture_generic_list(&mut self) -> Option<String> {
        let start = match self.peek().kind {
            TokenKind::Less | TokenKind::Shl => self.peek().span.start as usize,
            _ => return None,
        };
        let mut depth: i32 = match self.peek().kind {
            TokenKind::Less => 1,
            TokenKind::Shl => 2,
            _ => return None,
        };
        self.bump();
        let mut end = start;
        while depth != 0 && self.peek().kind != TokenKind::Eof {
            let t = self.bump();
            end = t.span.end as usize;
            match t.kind {
                TokenKind::Less => depth += 1,
                TokenKind::Shl => depth += 2,
                TokenKind::Greater => depth -= 1,
                TokenKind::Shr => depth -= 2,
                TokenKind::ShrShr => depth = depth.saturating_sub(3),
                _ => {}
            }
        }
        let raw = std::str::from_utf8(
            &self.source[start.min(self.source.len())..end.min(self.source.len())],
        )
        .unwrap_or("");
        Some(raw.chars().filter(|c| !c.is_whitespace()).collect())
    }

    /// Consume an optional generic parameter list (`<T, U, ...>`) and return
    /// the number of top-level type parameters (0 if no list). Nested generic
    /// arguments (e.g. `<Dictionary<string, int>>`) are counted as one.
    fn generic_parameter_count(&mut self) -> usize {
        // Accept both `<` and `<<` (lexed as Shl) as the opening token.
        // `Shl` represents TWO `<`, so start depth at 2.
        let opening_depth: i32 = match self.peek().kind {
            TokenKind::Less => 1,
            TokenKind::Shl => 2,
            _ => return 0,
        };
        self.bump();
        let mut depth = opening_depth;
        let mut count = 1usize; // at least one param if the list is non-empty
        // A new top-level param starts after each top-level comma.
        while depth != 0 && self.peek().kind != TokenKind::Eof {
            match self.bump().kind {
                TokenKind::Less => depth += 1,
                TokenKind::Shl => depth += 2, // `<<` = two `<`
                TokenKind::Greater => depth -= 1,
                TokenKind::Shr => depth -= 2, // `>>` = two `>`
                TokenKind::ShrShr => depth = depth.saturating_sub(3), // `>>>`
                TokenKind::Comma if depth == 1 => count += 1,
                _ => {}
            }
        }
        count
    }

    /// A type name that may be qualified (`System.Threading.Tasks.Task`),
    /// generic (`Task<int>`), and array (`int[]`). Returns the base name with
    /// the generic argument list consumed but dropped, matching the compact
    /// type model. Array suffixes (`[]`) are appended to the name.
    pub(super) fn type_name(&mut self) -> Option<String> {
        self.type_name_impl(false)
    }

    /// Full-spelling variant: generic arguments are captured into the
    /// name (`Predicate<X509Certificate2>?`), whitespace-free so
    /// spellings compare equal site-independently. Used everywhere a
    /// type flows into resolution (params, fields, properties,
    /// returns, locals); the compact form survives only for
    /// lookahead probing.
    pub(super) fn type_name_full(&mut self) -> Option<String> {
        self.type_name_impl(true)
    }

    fn type_name_impl(&mut self, keep_args: bool) -> Option<String> {
        // Function-pointer type: `delegate* <unmanaged[cc]> <T...>`
        // (C# 9). The backend has no function-pointer model; the
        // signature is consumed and the type records as `delegate*`
        // (an opaque, unknown type downstream, like other externals).
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
            // Signature type list: `<T1, TRet>`.
            if matches!(self.peek().kind, TokenKind::Less | TokenKind::Shl) {
                self.skip_generic_parameter_list();
            }
            // Trailing pointer depth: `delegate* ... **` (mirrors the
            // expression-level delegate handling).
            let mut stars = "delegate*".to_owned();
            while self.peek().kind == TokenKind::Star {
                self.bump();
                stars.push('*');
            }
            return Some(stars);
        }
        let mut name = self.expect_identifier("rubyc.parser.rd.Parser.ctx.as_type")?;
        loop {
            // `Outer<int>.Inner<long>.Deep<...>`: after the first identifier
            // there may be a generic list before the first dot. Note: the
            // lexer may produce `Shl` for `<<`, so we accept both.
            // With `keep_args` the arguments join the spelling (see
            // `type_name_full`); otherwise they are consumed and dropped.
            if matches!(self.peek().kind, TokenKind::Less | TokenKind::Shl) {
                if keep_args {
                    if let Some(args) = self.capture_generic_list() {
                        name.push_str(&args);
                    }
                } else {
                    self.skip_generic_parameter_list();
                }
            }
            if self.eat(&TokenKind::Dot).is_some() {
                name.push('.');
                name.push_str(
                    &self.expect_identifier("rubyc.parser.rd.Parser.ctx.type_name_segment")?,
                );
            } else if self.peek().kind == TokenKind::Colon
                && self.peek_at(1).kind == TokenKind::Colon
            {
                // C# global/alias namespace: `global::Ns` or `alias::Ns`.
                self.bump();
                self.bump();
                if name == "global" {
                    // Transparent qualifier: `global::Ns` names the same
                    // type as `Ns` (generated serializers qualify
                    // everything, e.g. `(global::System.Object)o`).
                    // Restart the name at the next segment so recorded
                    // spellings resolve like their unqualified twins.
                    name = self.expect_identifier(
                        "rubyc.parser.rd.Parser.ctx.type_name_segment",
                    )?;
                } else {
                    name.push_str("::");
                    name.push_str(
                        &self.expect_identifier("rubyc.parser.rd.Parser.ctx.type_name_segment")?,
                    );
                }
            } else {
                break;
            }
        }
        // Trailing generic list after a dotted name (`A.B<T>` when the
        // loop above broke early): same keep/drop contract as above.
        if matches!(self.peek().kind, TokenKind::Less | TokenKind::Shl) {
            if keep_args {
                if let Some(args) = self.capture_generic_list() {
                    name.push_str(&args);
                }
            } else {
                self.skip_generic_parameter_list();
            }
        }
        // Array / nullable / pointer suffixes: `int[]`, `T?[]`, `T[]?`,
        // `void*`, etc. They can interleave in any order; the AST only
        // stores the concatenated suffix.
        loop {
            let mut progressed = false;
            if self.peek().kind == TokenKind::Question {
                self.bump();
                name.push('?');
                progressed = true;
            }
            while self.peek().kind == TokenKind::Star {
                self.bump();
                name.push('*');
                progressed = true;
            }
            while self.peek().kind == TokenKind::LeftBracket {
                self.bump(); // '['
                if self.peek().kind == TokenKind::Comma {
                    name.push('[');
                    loop {
                        self.bump(); // ','
                        name.push(',');
                        if self.peek().kind == TokenKind::RightBracket {
                            break;
                        }
                    }
                    self.bump(); // ']'
                    name.push(']');
                } else if self.peek().kind == TokenKind::RightBracket {
                    self.bump(); // ']'
                    name.push_str("[]");
                } else {
                    while self.peek().kind != TokenKind::RightBracket
                        && self.peek().kind != TokenKind::Eof
                    {
                        self.bump();
                    }
                    if self.peek().kind == TokenKind::RightBracket {
                        self.bump();
                        name.push_str("[]");
                    }
                }
                progressed = true;
            }
            if !progressed {
                break;
            }
        }
        // Pointer suffixes: `int*`, `int**`, `int*[]`, `int*?`, etc. Pointer
        // stars may appear after the base name and interleave with array /
        // nullable suffixes; the AST stores the concatenated suffix.
        loop {
            if self.peek().kind == TokenKind::Star {
                self.bump();
                name.push('*');
            } else if self.peek().kind == TokenKind::Question {
                self.bump();
                name.push('?');
            } else {
                break;
            }
        }
        Some(name)
    }

    /// Parses a (possibly dot-qualified) name: `Name` or `A.B.C`.
    /// Used for explicit interface implementation names.
    pub(super) fn qualified_name(&mut self) -> Option<String> {
        let mut name = self.expect_identifier("rubyc.parser.rd.Parser.ctx.as_variable_name")?;
        loop {
            // Generic list on this segment:
            // `IReadOnlyDictionary<string, T>.Keys`. Consumed and
            // dropped to match the compact type model. `>>`/`>>>`
            // lex as Shr/ShrShr.
            if matches!(
                self.peek().kind,
                TokenKind::Less | TokenKind::Shl
            ) {
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
            if self.eat(&TokenKind::Dot).is_some() {
                name.push('.');
                name.push_str(&self.expect_identifier("name segment")?);
            } else {
                break;
            }
        }
        Some(name)
    }

    /// Peeks ahead (without consuming) to determine whether the current
    /// position starts a type name (identifier + optional dots, generic
    /// params, array suffixes, nullable suffix) followed by an identifier
    /// (the member name). Returns `Some(name_offset)` where `name_offset` is
    /// the index of the member-name token, or `None` if the pattern does not
    /// match.
    fn likely_type_then_name(&self) -> Option<usize> {
        // Start: must be an identifier (the type's base name) — or a
        // `delegate*` function-pointer type.
        if matches!(
            self.peek().kind,
            TokenKind::Keyword(Keyword::Delegate_)
        ) && self.peek_at(1).kind == TokenKind::Star
        {
            // Skip `delegate * [unmanaged[cc]] <...>`; the name follows.
            let mut offset = 2usize;
            if matches!(self.peek_at(offset).kind, TokenKind::Identifier(_)) {
                offset += 1;
                if self.peek_at(offset).kind == TokenKind::LeftBracket {
                    let mut depth = 0usize;
                    while self.peek_at(offset).kind != TokenKind::Eof {
                        match self.peek_at(offset).kind {
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
            }
            if matches!(self.peek_at(offset).kind, TokenKind::Less | TokenKind::Shl) {
                let mut depth = 0usize;
                loop {
                    match &self.peek_at(offset).kind {
                        TokenKind::Eof => return None,
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
                            depth = depth.saturating_sub(2);
                            if depth == 0 {
                                offset += 1;
                                break;
                            }
                        }
                        TokenKind::ShrShr => {
                            depth = depth.saturating_sub(3);
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
            // Trailing pointer depth (`delegate* ... **`): skip stars
            // so the name is still found.
            while self.peek_at(offset).kind == TokenKind::Star {
                offset += 1;
            }
            if matches!(self.peek_at(offset).kind, TokenKind::Identifier(_)) {
                return Some(offset);
            }
            return None;
        }
        if !matches!(self.peek().kind, TokenKind::Identifier(_)) {
            return None;
        }
        let mut offset = 1usize;
        // Dotted segments: `.Identifier`
        while self.peek_at(offset).kind == TokenKind::Dot {
            offset += 1;
            if !matches!(self.peek_at(offset).kind, TokenKind::Identifier(_)) {
                return None;
            }
            offset += 1;
        }
        // C# global/alias namespace: `global::Ns` or `alias::Ns`.
        if self.peek_at(offset).kind == TokenKind::Colon
            && self.peek_at(offset + 1).kind == TokenKind::Colon
        {
            offset += 2;
            if !matches!(self.peek_at(offset).kind, TokenKind::Identifier(_)) {
                return None;
            }
            offset += 1;
            // Continue with any further `.Identifier` segments.
            while self.peek_at(offset).kind == TokenKind::Dot {
                offset += 1;
                if !matches!(self.peek_at(offset).kind, TokenKind::Identifier(_)) {
                    return None;
                }
                offset += 1;
            }
        }
        // Generic parameter list(s) interleaved with dotted segments:
        // `List<int>`, `List<int>.Enumerator`, `A<X>.B<Y>` (nested
        // generics like
        // `ConcurrentDictionary<…>.AlternateLookup<…>`). Each segment
        // may carry its own list; loop until neither follows. Note:
        // the lexer may produce `Shl`/`Shr` for `<<`/`>>` (shift
        // operators), which represent TWO `<` or TWO `>`. We treat
        // them as two of the respective bracket.
        loop {
            if matches!(
                self.peek_at(offset).kind,
                TokenKind::Less | TokenKind::Shl
            ) {
                match self.generic_args_end_at(offset) {
                    Some(end) => offset = end,
                    None => return None,
                }
            }
            // Dotted segments after a generic list: each may carry its
            // own list, handled by the next loop iteration.
            if self.peek_at(offset).kind != TokenKind::Dot {
                break;
            }
            offset += 1;
            if !matches!(self.peek_at(offset).kind, TokenKind::Identifier(_)) {
                return None;
            }
            offset += 1;
        }
        // The suffixes can appear in any combination and order: `T[]?[]`,
        // `T?[]?`, `T*`, etc. Loop until no suffix is present.
        loop {
            let mut progressed = false;
            // Nullable suffix: `?`
            if self.peek_at(offset).kind == TokenKind::Question {
                offset += 1;
                progressed = true;
            }
            // Pointer suffixes: `*`, `**`.
            while self.peek_at(offset).kind == TokenKind::Star {
                offset += 1;
                progressed = true;
            }
            // Array suffixes: `[]` pairs, `[,]` multi-dim
            while self.peek_at(offset).kind == TokenKind::LeftBracket {
                offset += 1; // '['
                if self.peek_at(offset).kind == TokenKind::Comma {
                    loop {
                        offset += 1; // ','
                        if self.peek_at(offset).kind == TokenKind::RightBracket {
                            break;
                        }
                    }
                    offset += 1; // ']'
                } else if self.peek_at(offset).kind == TokenKind::RightBracket {
                    offset += 1; // ']'
                } else {
                    while self.peek_at(offset).kind != TokenKind::RightBracket
                        && self.peek_at(offset).kind != TokenKind::Eof
                    {
                        offset += 1;
                    }
                    if self.peek_at(offset).kind == TokenKind::RightBracket {
                        offset += 1;
                    }
                }
                progressed = true;
            }
            if !progressed {
                break;
            }
        }
        // The next token must be a name (the member name; contextual
        // keywords qualify). For explicit interface implementations, the
        // name may be dot-qualified: `Type INames.MemberName`. In that
        // case, return the offset of the LAST identifier in the qualified
        // name so the caller can check what follows the full name
        // (e.g. `=>` for a property).
        if self.peek_is_name(offset) {
            let mut name_end = offset;
            loop {
                // Generic list on this segment:
                // `IReadOnlyDictionary<string, T>.Keys`. Consumed for
                // lookahead only. `>>`/`>>>` lex as Shr/ShrShr.
                if matches!(
                    self.peek_at(name_end + 1).kind,
                    TokenKind::Less | TokenKind::Shl
                ) {
                    let mut depth = 0i32;
                    let mut i = name_end + 1;
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
                    name_end = i - 1; // `i` is past `>`; end at `>` itself
                }
                // Consume dot-qualified segments: `.Identifier`.
                if self.peek_at(name_end + 1).kind == TokenKind::Dot
                    && self.peek_is_name(name_end + 2)
                {
                    name_end += 2; // skip '.' and the identifier
                } else {
                    break;
                }
            }
            Some(name_end)
        } else {
            None
        }
    }

    /// Checks if the current position starts a fixed-size array field:
    /// `Type Name[size];`.
    fn likely_fixed_array_field(&self) -> bool {
        // Start: must be an identifier (the type's base name).
        if !matches!(self.peek().kind, TokenKind::Identifier(_)) {
            return false;
        }
        let mut offset = 1usize;
        // Dotted segments: `.Identifier`
        while self.peek_at(offset).kind == TokenKind::Dot {
            offset += 1;
            if !matches!(self.peek_at(offset).kind, TokenKind::Identifier(_)) {
                return false;
            }
            offset += 1;
        }
        // Generic parameter list: `<...>`
        if self.peek_at(offset).kind == TokenKind::Less {
            let mut depth = 0usize;
            loop {
                match &self.peek_at(offset).kind {
                    TokenKind::Eof => return false,
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
        // Array suffixes: `[]` pairs
        while self.peek_at(offset).kind == TokenKind::LeftBracket
            && self.peek_at(offset + 1).kind == TokenKind::RightBracket
        {
            offset += 2;
        }
        // Nullable suffix: `?`
        if self.peek_at(offset).kind == TokenKind::Question {
            offset += 1;
        }
        // The next token must be an identifier (the member name).
        if !matches!(self.peek_at(offset).kind, TokenKind::Identifier(_)) {
            return false;
        }
        // The token after the name must be `[` followed by a size
        // expression: integer literals (`[16]`) or const references
        // (`[MAX_IP_ADDRESS_BYTES]`, `[N * 2]` in Interop structs).
        // Anything else (e.g. `(`) means this is not a fixed field.
        self.peek_at(offset + 1).kind == TokenKind::LeftBracket
            && matches!(
                self.peek_at(offset + 2).kind,
                TokenKind::IntLiteral(_) | TokenKind::Identifier(_)
            )
    }

    /// `<in T, out T, ...>` → names only; variance qualifiers are consumed.
    fn generic_parameter_list(&mut self) -> Vec<String> {
        if self.eat(&TokenKind::Less).is_none() {
            return Vec::new();
        }
        let mut params = Vec::new();
        loop {
            // Attributes on type parameters
            // (`<[Attr] T>`): contracts only, inert — consume and drop.
            // Progress-guarded: a malformed `[` must not spin here.
            while self.peek().kind == TokenKind::LeftBracket {
                let mark = self.pos;
                let _ = self.attribute_lists();
                if self.pos == mark
                    || self.peek().kind != TokenKind::LeftBracket
                {
                    break;
                }
            }
            while matches!(
                self.peek().kind,
                TokenKind::Keyword(Keyword::In_) | TokenKind::Keyword(Keyword::Out_)
            ) {
                self.bump();
            }
            if self.peek().kind == TokenKind::Greater {
                break;
            }
            let Some(p) =
                self.expect_identifier("rubyc.parser.rd.Parser.ctx.as_generic_parameter") else {
                    break;
                };
            params.push(p);
            if self.eat(&TokenKind::Comma).is_none() {
                break;
            }
        }
        self.expect(&TokenKind::Greater, "to close generic parameter list");
        params
    }

    /// Consume one or more C# `where T : ...` clauses. Constraints do not yet
    /// affect lowering, but must not prevent the enclosing declaration parsing.
    /// Parse `where T : IFoo<T, int>, U : class` into the recorded
    /// constraints. The clauses are what a type parameter's members
    /// resolve against (C# binds `x[i]` through the constraint), so
    /// they are kept instead of discarded.
    pub(super) fn parse_type_constraints(
        &mut self,
    ) -> Vec<crate::models::ast::method::TypeConstraint> {
        let mut out = Vec::new();
        while self.peek().kind == TokenKind::Keyword(Keyword::Where_) {
            // A clause shape the parser does not recognize (the C# 13
            // `allows ref struct` form, for one) must not break the
            // parse: rewind and skip the constraints the old way.
            let mark = self.pos;
            let err_mark = self.errors.len();
            self.bump();
            loop {
                if !matches!(self.peek().kind, TokenKind::Identifier(_))
                    || !matches!(self.peek_at(1).kind, TokenKind::Colon)
                {
                    self.pos = mark;
                    self.errors.truncate(err_mark);
                    self.skip_type_constraints();
                    return out;
                }
                let Some(param) = self.expect_identifier("type parameter name") else {
                    break;
                };
                if self.expect(&TokenKind::Colon, "in a type constraint").is_none() {
                    break;
                }
                let mut types = Vec::new();
                loop {
                    // The `ref struct` / `allows ref struct` constraint
                    // modifiers (C# 13/14) are not type names.
                    let is_allows = matches!(&self.peek().kind, TokenKind::Identifier(n) if n == "allows");
                    if (is_allows
                        && matches!(self.peek_at(1).kind, TokenKind::Keyword(Keyword::Ref_))
                        && matches!(self.peek_at(2).kind, TokenKind::Keyword(Keyword::Struct_)))
                        || (matches!(self.peek().kind, TokenKind::Keyword(Keyword::Ref_))
                            && matches!(self.peek_at(1).kind, TokenKind::Keyword(Keyword::Struct_)))
                    {
                        if is_allows {
                            self.bump();
                        }
                        self.bump();
                        self.bump();
                        types.push("ref struct".to_owned());
                        if !matches!(self.peek().kind, TokenKind::Comma) {
                            break;
                        }
                        self.bump();
                        continue;
                    }
                    // The special constraints (`class`, `struct`) are
                    // keywords, not type names.
                    if let TokenKind::Keyword(k) = self.peek().kind {
                        let special = match k {
                            Keyword::Class => Some("class"),
                            Keyword::Struct_ => Some("struct"),
                            _ => None,
                        };
                        if let Some(sp) = special {
                            self.bump();
                            // `class?` / `struct?`: the nullable
                            // annotation belongs to the constraint.
                            if matches!(self.peek().kind, TokenKind::Question) {
                                self.bump();
                            }
                            types.push(sp.to_owned());
                            if !matches!(self.peek().kind, TokenKind::Comma) {
                                break;
                            }
                            if matches!(self.peek_at(1).kind, TokenKind::Identifier(_))
                                && matches!(self.peek_at(2).kind, TokenKind::Colon)
                            {
                                break;
                            }
                            self.bump();
                            continue;
                        }
                    }
                    let Some(t) = self.type_name_full() else {
                        self.pos = mark;
                        self.errors.truncate(err_mark);
                        self.skip_type_constraints();
                        return out;
                    };
                    types.push(t);
                    // A comma may separate the next PARAMETER clause
                    // (`where T : IFoo, U : IBar`): an identifier
                    // followed by `:` starts a new clause, and that
                    // comma belongs to the clause, not to the type list.
                    if !matches!(self.peek().kind, TokenKind::Comma) {
                        break;
                    }
                    if matches!(self.peek_at(1).kind, TokenKind::Identifier(_))
                        && matches!(self.peek_at(2).kind, TokenKind::Colon)
                    {
                        break;
                    }
                    self.bump();
                }
                out.push(crate::models::ast::method::TypeConstraint {
                    param,
                    types,
                });
                if self.eat(&TokenKind::Comma).is_none() {
                    break;
                }
            }
        }
        out
    }

    pub(super) fn skip_type_constraints(&mut self) {
        while self.peek().kind == TokenKind::Keyword(Keyword::Where_) {
            self.bump();
            while !matches!(
                self.peek().kind,
                TokenKind::LeftBrace
                    | TokenKind::Eof
                    | TokenKind::Keyword(Keyword::Where_)
                    // Expression-bodied member: `where T : C => expr;`.
                    // The `=>` ends the constraint list; consuming it
                    // (and the body) unbalances the whole file.
                    | TokenKind::FatArrow
                    | TokenKind::Semicolon
            ) {
                self.bump();
            }
        }
    }

    /// `[Access] [attrs] [RetType] Name(params) { body } | Name(params);`
    /// RetType defaults to `int`; a trailing `;` declares a bodyless method.
    pub(super) fn method_decl(
        &mut self,
        access: Access,
        attrs: Vec<Attribute>,
        _start: Span,
        is_override: bool,
        is_virtual: bool,
        is_sealed: bool,
        is_hiding: bool,
        is_static: bool,
        is_abstract: bool,
        is_partial: bool,
        is_async: bool,
        is_extern: bool,
    ) -> Option<MethodDecl> {
        // Optional return type: a type name followed by an identifier and `(`
        // means `RetType Name(params)`; a single identifier followed by `(`
        // means `Name(params)` with the default return type.
        // The return type may have array (`[]`) or nullable (`?`) suffixes.
        let ret_type = if let Some(noff) = self.likely_type_then_name() {
            // `RetType Name(params)` or `RetType Name<T>(params)` or
            // `RetType Namespace<T>.Name(params)` — the token after the
            // method name is `(`, `<`, or `.`.
            if matches!(
                self.peek_at(noff + 1).kind,
                TokenKind::LeftParen | TokenKind::Less | TokenKind::Dot
            ) {
                self.type_name_full()?
            } else {
                // `Type Name` without `(` is not a valid method; fall through
                // to treat the first identifier as the name with a default
                // return type.
                let is_ctor = matches!(&self.peek().kind,
                    TokenKind::Identifier(n)
                    if Some(n.as_str()) == self.context.class.as_deref());
                if is_ctor {
                    String::new()
                } else {
                    "int".to_owned()
                }
            }
        } else if matches!(self.peek_at(1).kind, TokenKind::LeftParen) {
            // `Name(...)` with no return type: a constructor when Name is
            // the enclosing class, otherwise the legacy int default.
            let is_ctor = matches!(&self.peek().kind,
                TokenKind::Identifier(n)
                if Some(n.as_str()) == self.context.class.as_deref());
            if is_ctor {
                String::new()
            } else {
                "int".to_owned()
            }
        } else {
            self.expect_identifier("rubyc.parser.rd.Parser.ctx.as_return_type")?
        };
        let mut name = self.expect_identifier("as method name")?;
        // Namespace-qualified method name with optional generics:
        // `IValueProvider<int>.GetValue` or `Name<T, U>(...)`. Only a
        // generic list directly before `(` declares this method's type
        // parameters (recorded); qualifier lists — including nested
        // ones like `ICollection<KeyValuePair<K, V>>` — are skipped
        // blindly. Validation first so a `<` that is not generics
        // (or unbalanced input) never consumes.
        let mut method_generics = Vec::new();
        loop {
            if self.peek().kind == TokenKind::Less {
                let Some(end) = self.generic_args_end_at(0) else {
                    break;
                };
                if !self.generic_follow_ok(end) {
                    break;
                }
                if matches!(self.peek_at(end).kind, TokenKind::LeftParen) {
                    // Method's own list (`M<T>(`): record names.
                    method_generics = self.generic_parameter_list();
                    break;
                }
                // Qualifier's list (`IFoo<T>.`): skip silently.
                self.skip_generic_parameter_list();
            }
            if self.eat(&TokenKind::Dot).is_some() {
                match self.expect_identifier("method name segment") {
                    Some(seg) => {
                        name.push('.');
                        name.push_str(&seg);
                    }
                    None => break,
                }
            } else {
                break;
            }
        }
        self.context.function = Some(name.clone());

        self.expect(&TokenKind::LeftParen, "to open parameter list")?;
        let params = self.param_list();
        self.expect(&TokenKind::RightParen, "to close parameter list")?;

        // Generic constraints: `where T : Constraint, U : Constraint`.
        // Type constraints: `where T : ...` — recorded, because a
        // member on a type-parameter receiver binds through them.
        let constraints = self.parse_type_constraints();

        // Constructor chain: `: this(args)` or `: base(args)`.
        let mut base_call = None;
        if self.peek().kind == TokenKind::Colon
            && matches!(
                self.peek_at(1).kind,
                TokenKind::Keyword(Keyword::This) | TokenKind::Keyword(Keyword::Base)
            )
        {
            self.bump(); // ':'
            let is_this = matches!(self.peek().kind, TokenKind::Keyword(Keyword::This));
            self.bump(); // 'this' or 'base'
            self.expect(&TokenKind::LeftParen, "to open chained ctor args")?;
            self.in_base_init = true;
            let args = self.argument_list();
            self.in_base_init = false;
            self.expect(&TokenKind::RightParen, "to close chained ctor args")?;
            base_call = Some((if is_this { "this".to_owned() } else { "base".to_owned() }, args));
        }
        // `out` variables declared in the `: base(...)`/`: this(...)`
        // argument list (`: base(AlgorithmFromHandle(p, out var u, ...))`)
        // are in scope for the constructor BODY (C# definite-assignment
        // puts them there) — but `block()` below stashes pre-body
        // bindings for the outer scope, stranding them at member level
        // where nothing drains the queue. Capture them here and splice
        // them at the front of the body below.
        let mut base_bindings = std::mem::take(&mut self.pre_pending);
        base_bindings.append(&mut self.pending);

        // Bodyless declaration: import stub or (later) a semantic error.
        let mut body = if self.eat(&TokenKind::Semicolon).is_some() {
            None
        } else if self.eat(&TokenKind::FatArrow).is_some() {
            let expression = self.expression()?;
            let end = self
                .expect(&TokenKind::Semicolon, "after expression-bodied member")
                .map_or(self.peek().span, |token| token.span);
            Some(self.expression_body(
                crate::models::Stmt::Return {
                    value: Some(expression),
                    span: end,
                },
                _start.to(end),
            ))
        } else {
            Some(self.block())
        };
        // Splice captured `: base(...)` out-var bindings at the front
        // (they run before the body statements, mirroring evaluation
        // order: base call first, then body).
        if !base_bindings.is_empty() {
            if let Some(b) = body.as_mut() {
                let mut stmts = std::mem::take(&mut base_bindings);
                stmts.append(&mut b.stmts);
                b.stmts = stmts;
            }
        }
        let end = body
            .as_ref()
            .map(|b| b.span)
            .unwrap_or_else(|| self.peek_at(0).span);

        Some(MethodDecl {
            access,
            attrs,
            generic_params: method_generics,
            constraints,
            ret_type,
            name,
            params,
            body,
            is_override,
            is_virtual,
            is_sealed,
            is_hiding,
            is_static,
            is_abstract,
            is_partial,
            is_async,
            is_extern,
            base_call,
            span: end,
        })
    }

    /// Comma-separated `(Type name)` pairs. The empty list is valid.
    pub(super) fn param_list(&mut self) -> Vec<Param> {
        let mut params = Vec::new();
        if self.peek().kind == TokenKind::RightParen {
            return params;
        }
        loop {
            let Some(param) = self.param() else {
                return params;
            };
            params.push(param);
            if self.eat(&TokenKind::Comma).is_none() {
                return params;
            }
        }
    }

    pub(super) fn param(&mut self) -> Option<Param> {
        // Attributes may precede the parameter (`[Foo] string name`).
        let attrs = self.attribute_lists();
        let mut saw_ref_like = false;
        // Extension-method receiver (`this Span<T> span`): recorded on
        // the first parameter (lowering rewrites `recv.Ext(args)` to
        // `Class.Ext(recv, args)`). `params` likewise recorded (call
        // supplies trailing element args; see `compat_score`).
        let mut is_this = false;
        let mut is_params = false;
        while matches!(
            self.peek().kind,
            TokenKind::Keyword(Keyword::Ref_)
                | TokenKind::Keyword(Keyword::Out_)
                | TokenKind::Keyword(Keyword::In_)
                | TokenKind::Keyword(Keyword::This)
                | TokenKind::Keyword(Keyword::Params_)
                | TokenKind::Keyword(Keyword::Scoped)
        ) {
            saw_ref_like = saw_ref_like
                || matches!(
                    self.peek().kind,
                    TokenKind::Keyword(Keyword::Ref_)
                        | TokenKind::Keyword(Keyword::Out_)
                        | TokenKind::Keyword(Keyword::In_)
                );
            is_this = is_this
                || matches!(self.peek().kind, TokenKind::Keyword(Keyword::This));
            is_params = is_params
                || matches!(self.peek().kind, TokenKind::Keyword(Keyword::Params_));
            self.bump();
        }
        // `ref readonly T x` — ref-readonly parameters (by-ref-ness has
        // no runtime representation here). Only after ref/out/in;
        // a bare `readonly` is not a valid parameter modifier.
        if saw_ref_like
            && matches!(self.peek().kind, TokenKind::Keyword(Keyword::Readonly))
        {
            self.bump();
        }
        // Tuple element type: `(int, int) x`, `(T n, U m)[] y`.
        // Fold it into a synthetic identifier (as for tuple-typed
        // members) so the normal type path below proceeds unchanged.
        // Safe here: `(` at parameter start can only open a tuple
        // type (default values come after `=`, never here).
        if matches!(self.peek().kind, TokenKind::LeftParen) {
            if let Some(end) = self
                .tuple_type_at(0)
                .or_else(|| self.unnamed_tuple_type_at(0))
                .or_else(|| self.member_tuple_at(0))
            {
                let type_start = self.peek().span.start as usize;
                let type_end =
                    self.tokens[self.pos + end - 1].span.end as usize;
                let text = std::str::from_utf8(
                    &self.source[type_start..type_end.min(self.source.len())],
                )
                .unwrap_or("tuple")
                .to_owned();
                let span = self
                    .peek()
                    .span
                    .to(self.tokens[self.pos + end - 1].span);
                self.tokens.splice(
                    self.pos..self.pos + end,
                    [crate::models::Token {
                        kind: TokenKind::Identifier(text),
                        span,
                    }],
                );
            }
        }
        let type_name = self.type_name_full()?;
        let name = self.expect_identifier_or_keyword("rubyc.parser.rd.Parser.ctx.as_parameter_name")?;
        let default_value = if self.eat(&TokenKind::Assign).is_some() {
            self.expression()
        } else {
            None
        };
        Some(Param {
            type_name,
            name,
            span: Span::default(),
            default_value,
            is_this,
            is_params,
            is_byref: saw_ref_like,
            attrs,
        })
    }
    /// `{ get; set; }` / `{ get { … } set { … } }` accessor list.
    ///
    /// Returns `(init, get, set, get_expr, is_init, is_settable)`. An optional
    /// `= expr` trailing the closing brace sets the initializer (auto-prop with
    /// a backing value).
    pub(super) fn property_accessors(
        &mut self,
    ) -> Option<(Expr, Option<Block>, Option<Block>, Option<Expr>, bool, bool)> {
        self.expect(&TokenKind::LeftBrace, "to open accessor list")?;
        let mut get: Option<Block> = None;
        let mut set: Option<Block> = None;
        let mut get_expr: Option<Expr> = None;
        let mut is_init = false;
        let mut is_settable = false;

        while self.peek().kind != TokenKind::RightBrace && self.peek().kind != TokenKind::Eof {
            // Attributes on individual accessors (e.g. `[MemberNotNull(..)]
            // set`): contracts only, inert for codegen — consume and drop.
            if self.peek().kind == TokenKind::LeftBracket {
                let _ = self.attribute_lists();
                continue;
            }
            // Consume optional access modifier before accessor keyword
            // (e.g. `private set`, `public get`), plus the `readonly`
            // modifier on accessors of readonly structs/members
            // (`readonly get => ...`).
            let _ = self.access_modifier();
            if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Readonly)) {
                self.bump();
            }
            match self.peek().kind {
                TokenKind::Identifier(ref name) if name == "get" => {
                    self.bump();
                    if self.peek().kind == TokenKind::Semicolon {
                        self.bump();
                        continue;
                    }
                    if self.peek().kind == TokenKind::FatArrow {
                        self.bump(); // '=>'
                        let expression = self.expression()?;
                        self.expect(
                            &TokenKind::Semicolon,
                            "rubyc.parser.rd.Parser.ctx.after_expression",
                        )?;
                        get_expr = Some(expression);
                        continue;
                    }
                    get = Some(self.block());
                }
                TokenKind::Identifier(ref name) if name == "set" => {
                    self.bump();
                    is_settable = true;
                    if self.peek().kind == TokenKind::Semicolon {
                        self.bump();
                        continue;
                    }
                    if self.peek().kind == TokenKind::FatArrow {
                        // `set => expr;` — expression-bodied setter.
                        self.bump(); // '=>'
                        let _expression = self.expression()?;
                        self.expect(
                            &TokenKind::Semicolon,
                            "rubyc.parser.rd.Parser.ctx.after_expression",
                        )?;
                        continue;
                    }
                    set = Some(self.block());
                }
                TokenKind::Identifier(ref name) if name == "init" => {
                    self.bump();
                    is_init = true;
                    if self.peek().kind == TokenKind::Semicolon {
                        self.bump();
                    } else {
                        // Defensive: an `init { … }` body is not expressible in
                        // the current model; consume it so recovery stays clean.
                        let _ = self.block();
                    }
                }
                _ => {
                    let found = describe(self.peek());
                    self.unexpected(
                        "rubyc.parser.rd.Parser.ctx.accessor",
                        &found,
                        self.peek().span,
                    );
                    break;
                }
            }
        }

        self.expect(&TokenKind::RightBrace, "to close accessor list")?;
        let init = if self.eat(&TokenKind::Assign).is_some() {
            let init = self.expression()?;
            // An initializer always terminates the property with ';'.
            self.expect(
                &TokenKind::Semicolon,
                "rubyc.parser.rd.Parser.ctx.after_expression",
            )?;
            init
        } else {
            // No initializer: the property ends at the closing brace. A stray
            // ';' is tolerated for symmetry with the `= expr;` form.
            if self.peek().kind == TokenKind::Semicolon {
                self.bump();
            }
            Expr::Literal(Literal::Int(0))
        };
        Some((init, get, set, get_expr, is_init, is_settable))
    }

    /// `enum Name : UnderlyingType { A = 0, B = 1, … }`
    pub(super) fn enum_decl(&mut self) -> Option<crate::models::EnumDecl> {
        use crate::models::{EnumDecl, TokenKind};

        let start = self.peek().span;
        self.bump(); // 'enum'
        let name = self.expect_identifier("as enum name")?;

        // Optional underlying type: `: int`, `: byte`, etc.
        let mut underlying_type = "int".to_owned();
        if self.peek().kind == TokenKind::Colon {
            self.bump();
            underlying_type = self.expect_identifier("as underlying type")?;
        }

        self.expect(&TokenKind::LeftBrace, "to open enum body")?;

        let mut variants = Vec::new();
        let mut auto_value = 0i64;

        while self.peek().kind != TokenKind::RightBrace && self.peek().kind != TokenKind::Eof {
            if self.eat(&TokenKind::Comma).is_some() {
                continue;
            }
            // Attributes on variants (`[Obsolete(...)] AllowNone`):
            // contracts only, inert for codegen — consume and drop.
            // (Enum variants carry no attrs, like params and sigs.)
            if self.peek().kind == TokenKind::LeftBracket {
                let _ = self.attribute_lists();
                continue;
            }
            let vname = self.expect_identifier("as variant name")?;
            let mut value = Some(auto_value);
            if self.eat(&TokenKind::Assign).is_some() {
                // Fast path: a (possibly negative) integer literal that
                // STANDS ALONE (followed by `,` or `}`, modulo a `-` prefix
                // and an L/U suffix). A literal LED combination
                // (`0xF003F | X`) belongs to the expression path below.
                let mut end = 0usize;
                if self.peek_at(end).kind == TokenKind::Minus {
                    end += 1;
                }
                let is_lit = matches!(self.peek_at(end).kind, TokenKind::IntLiteral(_))
                    && {
                        end += 1;
                        if matches!(&self.peek_at(end).kind, TokenKind::Identifier(s)
                        if matches!(s.as_str(), "L" | "l" | "U" | "u" | "UL" | "Ul" | "uL" | "ul"))
                        {
                            end += 1;
                        }
                        matches!(
                            self.peek_at(end).kind,
                            TokenKind::Comma | TokenKind::RightBrace
                        )
                    };
                if is_lit {
                    let negative = self.eat(&TokenKind::Minus).is_some();
                    match self.peek().kind {
                        TokenKind::IntLiteral(v) => {
                            let iv = if negative { -(v as i64) } else { v as i64 };
                            value = Some(iv);
                            auto_value = iv;
                            self.bump();
                            // C# integral suffixes are lexed as an adjacent identifier.
                            if matches!(&self.peek().kind, TokenKind::Identifier(s)
                            if matches!(s.as_str(), "L" | "l" | "U" | "u" | "UL" | "Ul" | "uL" | "ul"))
                            {
                                self.bump();
                            }
                        }
                        _ => {
                            let span = self.peek().span;
                            self.error("E0001", "expected integer for enum value".to_owned(), span);
                            return None;
                        }
                    }
                } else {
                    // Non-literal initializer: a reference to an earlier
                    // variant (`B = A`), a dotted cross-enum reference
                    // (`B = Other.X`), optionally combined (`A + 1`,
                    // `1 << 4`, `(int)X`, `-A`). Parse a full expression
                    // (which handles its own unary minus) and fold it
                    // against the variants collected so far — C# enum
                    // initializers must be compile-time constant.
                    // Unresolvable references (other files) degrade to
                    // auto-numbering rather than erroring.
                    // Clone the cross-enum table first: `expression()`
                    // needs `&mut self`.
                    let extern_table = self.enum_values.clone();
                    match self.expression() {
                        Some(e) => match fold_enum_expr(&e, &variants, &extern_table) {
                            Some(iv) => {
                                value = Some(iv);
                                auto_value = iv;
                            }
                            None if is_const_shaped(&e) => {
                                // A constant-shaped reference that does not
                                // resolve in this file (typically another
                                // enum's variant): degrade to auto-numbering
                                // rather than blocking compilation.
                            }
                            _ => {
                                let span = self.peek().span;
                                self.error(
                                    "E0001",
                                    "expected integer for enum value".to_owned(),
                                    span,
                                );
                                return None;
                            }
                        },
                        None => {
                            let span = self.peek().span;
                            self.error("E0001", "expected integer for enum value".to_owned(), span);
                            return None;
                        }
                    }
                }
            }
            variants.push((vname, value));
            auto_value += 1;
            if self.peek().kind == TokenKind::Comma {
                self.bump();
            }
        }

        self.expect(&TokenKind::RightBrace, "to close enum body")?;
        // Permitted trailing `;`, as for classes.
        self.eat(&TokenKind::Semicolon);
        // Publish resolved variants for later enums in this file to
        // reference (`B = OtherEnum.X`).
        for (vname, vval) in &variants {
            if let Some(v) = vval {
                self.enum_values
                    .push((name.clone(), vname.clone(), *v));
            }
        }
        Some(EnumDecl {
            name,
            underlying_type,
            variants,
            span: start.to(self.peek_at(0).span),
        })
    }
}

fn synthesize_record_members(
    name: &str,
    specs: &[Param],
    field_access: Access,
    span: crate::models::Span,
) -> Vec<ClassMember> {
    let mut out: Vec<ClassMember> = Vec::new();
    if specs.is_empty() {
        return out;
    }
    for spec in specs {
        out.push(ClassMember::Field(FieldDecl {
            access: field_access,
            attrs: Vec::new(),
            type_name: spec.type_name.clone(),
            name: spec.name.clone(),
            init: Expr::Literal(Literal::Int(0)),
            init_pre: Vec::new(),
            is_const: false,
            is_static: false,
            is_byref: false,
            span,
        }));
    }
    let mut body: Vec<crate::models::Stmt> = Vec::new();
    for spec in specs {
        body.push(crate::models::Stmt::Expression(Expr::Assign {
            target: Box::new(Expr::Member {
                object: Box::new(Expr::Ident {
                    name: "this".to_owned(),
                    span,
                }),
                name: spec.name.clone(),
                span,
            }),
            value: Box::new(Expr::Ident {
                name: spec.name.clone(),
                span,
            }),
            span,
        }));
    }
    out.push(ClassMember::Method(MethodDecl {
        access: Access::Public,
        attrs: Vec::new(),
        generic_params: Vec::new(),
        constraints: Vec::new(),
        ret_type: String::new(),
        name: name.to_owned(),
        params: specs.to_vec(),
        body: Some(Block { stmts: body, span }),
        span,
        is_override: false,
        is_virtual: false,
        is_sealed: false,
        is_hiding: false,
        is_static: false,
        is_abstract: false,
        is_partial: false,
        is_async: false,
        is_extern: false,
        base_call: None,
    }));
    out
}
