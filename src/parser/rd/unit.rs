//! Entry points: compilation units, namespaces, member dispatch.

use crate::models::{
    Access, CompilationUnit, Keyword, Member, NamespaceDecl, TokenKind, UsingAlias,
};

use super::Parser;

impl Parser {
    pub(super) fn compilation_unit(&mut self) -> CompilationUnit {
        let start = self.peek().span;
        let mut namespaces = Vec::new();
        let mut members = Vec::new();
        while self.peek().kind != TokenKind::Eof {
            if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Namespace)) {
                // A `namespace` at unit level starts a (block or file-scoped)
                // namespace. The file-scoped form consumes the rest of the
                // file, so the loop terminates naturally.
                match self.namespace_decl() {
                    Some(ns) => namespaces.push(ns),
                    None => {
                        let before = self.pos;
                        self.synchronize();
                        if self.pos == before {
                            self.bump();
                        }
                    }
                }
            } else if self.peek().kind == TokenKind::LeftBracket {
                // File-level attribute: `[assembly: …]`, `[module: …]`, etc.
                // Consumed and discarded — RubyC has no attribute metadata.
                self.skip_balanced_bracket();
            } else {
                self.parse_member_or_using(&mut members);
            }
        }
        CompilationUnit {
            namespaces,
            members,
            aliases: std::mem::take(&mut self.aliases),
            static_usings: std::mem::take(&mut self.static_usings),
            assembly_blocks: Vec::new(),
            span: start.to(self.peek().span),
        }
    }

    /// Parse either namespace form:
    ///   `namespace name { members }`  (block form)
    ///   `namespace name;`             (file-scoped form)
    pub(super) fn namespace_decl(&mut self) -> Option<NamespaceDecl> {
        let kw = self.eat(&TokenKind::Keyword(Keyword::Namespace))?;
        let name = self.parse_dot_qualified_name().unwrap_or_default();

        self.context.namespace = Some(name.clone());
        if self.eat(&TokenKind::Semicolon).is_some() {
            // File-scoped: following top-level declarations live in this namespace.
            let mut members = Vec::new();
            while self.peek().kind != TokenKind::Eof {
                self.parse_member_or_using(&mut members);
            }
            return Some(NamespaceDecl {
                name,
                members,
                span: kw.span.to(self.peek().span),
            });
        }

        let mut members = Vec::new();
        if self
            .expect(
                &TokenKind::LeftBrace,
                "rubyc.parser.rd.Parser.ctx.open_namespace_body",
            )
            .is_some()
        {
            while self.peek().kind != TokenKind::RightBrace && self.peek().kind != TokenKind::Eof {
                self.parse_member_or_using(&mut members);
            }
            self.expect(
                &TokenKind::RightBrace,
                "rubyc.parser.rd.Parser.ctx.close_namespace_body",
            );
        }
        Some(NamespaceDecl {
            name,
            members,
            span: kw.span.to(self.peek().span),
        })
    }

    /// Optional access modifier before a member; recorded, enforced later.
    pub(super) fn access_modifier(&mut self) -> Access {
        let access = match self.peek().kind {
            TokenKind::Keyword(Keyword::Public) => Some(Access::Public),
            TokenKind::Keyword(Keyword::Private) => Some(Access::Private),
            TokenKind::Keyword(Keyword::Protected) => Some(Access::Protected),
            TokenKind::Keyword(Keyword::Internal) => Some(Access::Internal),
            _ => None,
        };
        // Only consume when followed by something that can start a member
        // (identifier / another modifier keyword), not by unrelated tokens.
        let next_starts_member = matches!(
            self.peek_at(1).kind,
            TokenKind::Identifier(_)
                | TokenKind::Keyword(Keyword::Public)
                | TokenKind::Keyword(Keyword::Private)
                | TokenKind::Keyword(Keyword::Protected)
                // `public ~Class()` — destructor with explicit access.
                | TokenKind::Tilde
                | TokenKind::Keyword(Keyword::Virtual)
                | TokenKind::Keyword(Keyword::Override)
                | TokenKind::Keyword(Keyword::Static)
                | TokenKind::Keyword(Keyword::Sealed)
                | TokenKind::Keyword(Keyword::Abstract)
                | TokenKind::Keyword(Keyword::Internal)
                | TokenKind::Keyword(Keyword::Readonly)
                | TokenKind::Keyword(Keyword::Const)
                | TokenKind::Keyword(Keyword::Class)
                | TokenKind::Keyword(Keyword::Interface)
                | TokenKind::Keyword(Keyword::Struct_)
                | TokenKind::Keyword(Keyword::Enum_)
                | TokenKind::Keyword(Keyword::Record_)
                | TokenKind::Keyword(Keyword::Delegate_)
                // `public [Attr] …`
                | TokenKind::LeftBracket
                // `public (T1 n1, ...) ...` — tuple-typed member.
                | TokenKind::LeftParen
                // `public event …` / `public Type this[ … ]`
                | TokenKind::Keyword(Keyword::Event_)
                | TokenKind::Keyword(Keyword::This)
                // `public ref struct …` / `public partial class …`
                | TokenKind::Keyword(Keyword::Ref_)
                | TokenKind::Keyword(Keyword::Partial)
                // `public new …` — method hiding.
                | TokenKind::Keyword(Keyword::New)
                // `public required …` — required member.
                | TokenKind::Keyword(Keyword::Required)
                // `public unsafe …` — unsafe member.
                | TokenKind::Keyword(Keyword::Unsafe)
                // `public fixed …` — fixed-size array field.
                | TokenKind::Keyword(Keyword::Fixed)
                // `private async …` — async method.
                | TokenKind::Keyword(Keyword::Async)
                // `private volatile …` — volatile field.
                | TokenKind::Keyword(Keyword::Volatile)
                // `private extern …` — extern method.
                | TokenKind::Keyword(Keyword::Extern)
        );
        if let Some(access) = access.filter(|_| next_starts_member) {
            self.bump();
            // Combined accessibility: `private protected` /
            // `protected private` (accessible to derived classes in the
            // same assembly — modeled as `protected`: within one library
            // build every class shares the assembly, so derived access
            // resolves identically; cross-assembly narrowing arrives
            // with cross-library references), and `protected internal` /
            // `internal protected` (treated as public, consistent with
            // bare `internal` above).
            let combined = match (access, &self.peek().kind) {
                (Access::Private, TokenKind::Keyword(Keyword::Protected))
                | (Access::Protected, TokenKind::Keyword(Keyword::Private)) => {
                    Some(Access::Protected)
                }
                (Access::Protected, TokenKind::Keyword(Keyword::Internal))
                | (Access::Internal, TokenKind::Keyword(Keyword::Protected)) => {
                    Some(Access::Public)
                }
                _ => None,
            };
            if let Some(merged) = combined {
                self.bump();
                return merged;
            }
            return access;
        }
        Access::Default
    }

    /// Parse a dot-qualified name: `A`, `A.B`, `A.B.C`, …
    /// Returns `None` if the current token is not an identifier.
    fn parse_dot_qualified_name(&mut self) -> Option<String> {
        let first = self.expect_identifier("after 'namespace'")?;
        let mut name = first;
        while self.peek().kind == TokenKind::Dot {
            self.bump();
            let next = self.expect_identifier("in namespace name")?;
            name.push('.');
            name.push_str(&next);
        }
        Some(name)
    }

    /// Parse either a type member (class/interface/enum/…), a `using`
    /// directive, or a file/namespace-level attribute list. Namespace
    /// usings and attributes are consumed and discarded; `using Alias`
    /// bindings are recorded on the parser (moved into the unit);
    /// members are pushed onto `members`. On parse failure, recovers
    /// and makes progress.
    fn parse_member_or_using(&mut self, members: &mut Vec<Member>) {
        if self.is_using_directive() {
            self.parse_using_directive();
            return;
        }
        if self.peek().kind == TokenKind::LeftBracket {
            // File/namespace-level attribute: `[assembly: …]`, `[module: …]`,
            // etc. Consumed and discarded.
            self.skip_balanced_bracket();
            return;
        }
        match self.member() {
            Some(member) => members.push(member),
            None => {
                let before = self.pos;
                self.synchronize();
                if self.pos == before {
                    self.bump();
                }
            }
        }
    }

    /// True when the current position starts a `using` directive
    /// (possibly prefixed with `global` as an identifier).
    pub(super) fn is_using_directive(&self) -> bool {
        // `global using …` — 'global' is an identifier, 'using' is the keyword.
        if matches!(
            self.peek().kind,
            TokenKind::Identifier(ref id) if id == "global"
        ) && matches!(self.peek_at(1).kind, TokenKind::Keyword(Keyword::Using_)) {
            return true;
        }
        matches!(self.peek().kind, TokenKind::Keyword(Keyword::Using_))
    }

    /// Consume a `using` directive: `using [static] Name(.Name)* ;`,
    /// `global using ... ;`, or the alias form
    /// `using Alias = Qualified.Name ;`.
    /// Namespace/type usings are parsed and discarded — RubyC has no
    /// namespace resolution; all types are looked up in the merged
    /// compilation unit. Alias bindings ARE recorded (`self.aliases`)
    /// so later passes resolve them, exactly like .NET.
    pub(super) fn parse_using_directive(&mut self) {
        // Optional `global` prefix (lexed as an identifier).
        if matches!(
            self.peek().kind,
            TokenKind::Identifier(ref id) if id == "global"
        ) {
            self.bump();
        }
        // `using` keyword.
        self.bump();
        // `using static Qualified.Type;` — record the target; its
        // static members import into scope (resolved downstream).
        if matches!(self.peek().kind, TokenKind::Keyword(Keyword::Static)) {
            self.bump();
            if let Some(target) = self.parse_dot_qualified_name() {
                if !target.is_empty() {
                    self.static_usings.push(target);
                }
            }
            self.skip_to_semicolon();
            return;
        }
        // Alias form: `Alias = Qualified.Name ;`.
        if let TokenKind::Identifier(alias) = self.peek().kind.clone()
            && self.peek_at(1).kind == TokenKind::Assign
        {
            let alias = alias.clone();
            self.bump(); // alias name
            self.bump(); // '='
            // Dotted target (`System.Int32`); generic arguments on the
            // target (`Queue<object>`) are RECORDED as written — arity
            // distinguishes the generic definition from a same-named
            // non-generic class at lookup (`WorkQueue` → `Queue<T>`,
            // not legacy `Queue`).
            let mut target = self.parse_dot_qualified_name().unwrap_or_default();
            // Balanced `<...>` here always opens generic arguments (a
            // `using`-alias target is a type, never a comparison), so
            // gate on closeness alone — NOT on `is_generic_type_args`,
            // which additionally demands `(`/`.` after the list (for
            // generic calls) and would drop `<object>` before `;`.
            if self.peek().kind == TokenKind::Less && self.generic_close_at().is_some() {
                let start_off = self.peek().span.start as usize;
                self.skip_generic_parameter_list();
                let end_off = self.tokens[self.pos.saturating_sub(1)].span.end as usize;
                if end_off > start_off && end_off <= self.source.len() {
                    target.push_str(
                        std::str::from_utf8(&self.source[start_off..end_off]).unwrap_or(""),
                    );
                }
            }
            // A trailing `?` (`using X = string?;`) is not part of the
            // target type for resolution purposes.
            if self.peek().kind == TokenKind::Question {
                self.bump();
            }
            self.skip_to_semicolon();
            if !alias.is_empty() && !target.is_empty() {
                self.aliases.push(UsingAlias { name: alias, target });
            }
            return;
        }
        // The namespace/type name: `A.B.C` or (legacy spelling)
        // `A.B.C = alias` — consumed and discarded.
        self.skip_qualified_name_and_optional_alias();
    }

    /// Skip `Ident(.Ident)*` followed by an optional `= Ident` and a `;`.
    fn skip_qualified_name_and_optional_alias(&mut self) {
        // First identifier (may be a keyword like `System` — no,
        // namespace names are identifiers).
        if !matches!(self.peek().kind, TokenKind::Identifier(_)) {
            // Not a valid using directive; recover by skipping to ';'.
            self.skip_to_semicolon();
            return;
        }
        self.bump();
        // `.Name` chain.
        while self.peek().kind == TokenKind::Dot {
            self.bump();
            if matches!(self.peek().kind, TokenKind::Identifier(_)) {
                self.bump();
            } else {
                break;
            }
        }
        // Optional `= alias`.
        if self.peek().kind == TokenKind::Assign {
            self.bump();
            if matches!(self.peek().kind, TokenKind::Identifier(_)) {
                self.bump();
            }
        }
        // Trailing `;`.
        self.skip_to_semicolon();
    }

    /// Bump tokens until (and including) a `;` or `}` or `Eof`.
    fn skip_to_semicolon(&mut self) {
        while self.peek().kind != TokenKind::Eof
            && self.peek().kind != TokenKind::Semicolon
            && self.peek().kind != TokenKind::RightBrace
        {
            self.bump();
        }
        if self.peek().kind == TokenKind::Semicolon {
            self.bump();
        }
    }

    pub(super) fn member(&mut self) -> Option<Member> {
        let _attributes = self.attribute_lists();
        let access = self.access_modifier();
        let mut is_partial = false;
        while matches!(
            self.peek().kind,
            TokenKind::Keyword(Keyword::Sealed)
                | TokenKind::Keyword(Keyword::Abstract)
                | TokenKind::Keyword(Keyword::Static)
                | TokenKind::Keyword(Keyword::Readonly)
                | TokenKind::Keyword(Keyword::Partial)
                | TokenKind::Keyword(Keyword::Ref_)
                | TokenKind::Keyword(Keyword::Unsafe)
                | TokenKind::Keyword(Keyword::File)
                | TokenKind::Keyword(Keyword::Fixed)
        ) {
            if self.peek().kind == TokenKind::Keyword(Keyword::Partial) {
                is_partial = true;
            }
            self.bump();
        }
        // `closed` modifier (C# 14, contextual like `safe` above — see
        // class_member's identical arm): consumed and ignored here too
        // so top-level `closed class` parses.
        if matches!(self.peek().kind, TokenKind::Identifier(ref n) if n == "closed")
            && matches!(
                self.peek_at(1).kind,
                TokenKind::Keyword(Keyword::Class)
                    | TokenKind::Keyword(Keyword::Struct_)
                    | TokenKind::Keyword(Keyword::Interface)
                    | TokenKind::Keyword(Keyword::Record_)
            )
        {
            self.bump();
        }
        match &self.peek().kind {
            TokenKind::Keyword(Keyword::Interface) => self.interface_decl(is_partial).map(Member::Interface),
            TokenKind::Keyword(Keyword::Class) => {
                self.class_decl_impl(access, false, is_partial, false).map(Member::Class)
            }
            TokenKind::Keyword(Keyword::Enum_) => self.enum_decl().map(Member::Enum),
            TokenKind::Keyword(Keyword::Struct_) => {
                // Structs parse as classes, flagged: value semantics
                // arrive with A6, but constructor rules already differ
                // (a struct keeps its implicit parameterless ctor even
                // with explicit ctors).
                self.class_decl_impl(access, false, is_partial, true).map(Member::Class)
            }
            TokenKind::Keyword(Keyword::Delegate_) => {
                self.delegate_decl(access).map(Member::Delegate)
            }
            TokenKind::Keyword(Keyword::Record_) => {
                // `record struct Name(...)` / `record class Name(...)` —
                // bump past 'record' and 'struct'/'class', then parse the
                // class body.
                if matches!(
                    self.peek_at(1).kind,
                    TokenKind::Keyword(Keyword::Struct_)
                        | TokenKind::Keyword(Keyword::Class)
                ) {
                    let record_is_struct = matches!(
                        self.peek_at(1).kind,
                        TokenKind::Keyword(Keyword::Struct_)
                    );
                    self.bump(); // 'record'
                    self.bump(); // 'struct' or 'class'
                    self.class_decl_impl(access, true, is_partial, record_is_struct)
                        .map(Member::Class)
                } else {
                    // `record Name(...)` — parsed as a class.
                    self.class_decl_impl(access, false, is_partial, false).map(Member::Class)
                }
            }
            other => {
                let span = self.peek().span;
                let msg = crate::tf!(
                    "rubyc.parser.rd.Parser.expect.generic",
                    crate::t!("rubyc.parser.rd.Parser.ctx.class_expected"),
                    format!("{other:?}")
                );
                self.error("E0001", msg, span);
                None
            }
        }
    }
}
