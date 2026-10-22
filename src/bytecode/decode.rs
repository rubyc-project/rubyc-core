//! Bytecode bytes → AST.
//!
//! Every read is bounds-checked: malformed input produces a
//! [`DecodeError`], never a panic or out-of-bounds access.

use crate::bytecode::format::{self, FORMAT_VERSION};
use crate::models::{
    Access, BinOp, Block, CatchClause, ClassDecl, ClassMember, CompilationUnit, DecodeError,
    DelegateDecl, EnumDecl, EventDecl, Expr, FieldDecl, IndexerDecl, InterfaceDecl, InterpPart,
    Literal, Member, MethodDecl, MethodSig, NamespaceDecl, Param, PropertyDecl, Span, Stmt,
    SwitchExprArm, UnOp,
};

/// Decode bytecode bytes back into a compilation unit. Spans are zeroed; the
/// format does not carry source locations.
///
/// Tolerant of a missing RubyC preamble: if the standard `jump-over-magic` +
/// dog + kind/version header is absent (the `--no-magic` form), the bytes are
/// treated as a raw unit payload and decoded directly.
pub fn decode(bytes: &[u8]) -> Result<CompilationUnit, DecodeError> {
    match crate::native::container::parse(bytes) {
        Ok(parsed) => {
            if parsed.kind != crate::native::container::Kind::Bytecode {
                return Err(DecodeError::WrongKind);
            }
            if parsed.version != FORMAT_VERSION {
                return Err(DecodeError::UnsupportedVersion(parsed.version));
            }
            let mut r = Reader {
                bytes,
                pos: parsed.payload_start,
            };
            let unit = r.unit()?;
            if r.pos != bytes.len() {
                return Err(DecodeError::TrailingBytes);
            }
            Ok(unit)
        }
        // Preamble absent (or unparseable): fall back to the raw `--no-magic`
        // payload, which begins at the `KIND` byte.
        Err(_) => decode_raw_payload(bytes),
    }
}

/// Decode a raw unit payload that carries no RubyC preamble (`--no-magic`).
///
/// The payload is the bare unit encoding (kind byte first). We re-attach a
/// valid preamble in memory and reuse the standard payload Reader (positioned
/// past the preamble) so there is a single payload-decode path.
fn decode_raw_payload(bytes: &[u8]) -> Result<CompilationUnit, DecodeError> {
    if bytes.is_empty() {
        return Err(DecodeError::BadMagic);
    }
    // A raw payload's first byte must be a valid Kind; otherwise it is not a
    // bytecode payload at all.
    if crate::native::container::Kind::from_byte(bytes[0]).is_none() {
        return Err(DecodeError::BadMagic);
    }
    let mut with_preamble =
        crate::native::container::build(crate::native::container::Kind::Bytecode, FORMAT_VERSION);
    with_preamble.extend_from_slice(bytes);
    let mut r = Reader {
        bytes: &with_preamble,
        pos: crate::native::container::PREAMBLE_LEN + 3,
    };
    let unit = r.unit()?;
    if r.pos != with_preamble.len() {
        return Err(DecodeError::TrailingBytes);
    }
    Ok(unit)
}

/// Decode bytecode whose preamble was consumed by a platform loader
/// (shebang-stripped scripts etc.): `bytes` starts at the kind byte.
pub fn decode_payload(bytes: &[u8]) -> Result<CompilationUnit, DecodeError> {
    if bytes.len() < 3 {
        return Err(DecodeError::UnexpectedEof);
    }
    let kind = crate::native::container::Kind::from_byte(bytes[0]).ok_or(DecodeError::WrongKind)?;
    if kind != crate::native::container::Kind::Bytecode {
        return Err(DecodeError::WrongKind);
    }
    let version = u16::from_le_bytes([bytes[1], bytes[2]]);
    if version != FORMAT_VERSION {
        return Err(DecodeError::UnsupportedVersion(version));
    }

    let mut r = Reader { bytes, pos: 3 };
    let unit = r.unit()?;
    if r.pos != bytes.len() {
        return Err(DecodeError::TrailingBytes);
    }
    Ok(unit)
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn byte(&mut self) -> Result<u8, DecodeError> {
        let b = *self.bytes.get(self.pos).ok_or(DecodeError::UnexpectedEof)?;
        self.pos += 1;
        Ok(b)
    }

    fn u32(&mut self) -> Result<usize, DecodeError> {
        let end = self.pos.checked_add(4).ok_or(DecodeError::UnexpectedEof)?;
        if end > self.bytes.len() {
            return Err(DecodeError::UnexpectedEof);
        }
        let v = u32::from_le_bytes(self.bytes[self.pos..end].try_into().unwrap());
        self.pos = end;
        Ok(v as usize)
    }

    fn u64(&mut self) -> Result<u64, DecodeError> {
        let end = self.pos.checked_add(8).ok_or(DecodeError::UnexpectedEof)?;
        if end > self.bytes.len() {
            return Err(DecodeError::UnexpectedEof);
        }
        let v = u64::from_le_bytes(self.bytes[self.pos..end].try_into().unwrap());
        self.pos = end;
        Ok(v)
    }

    fn string(&mut self) -> Result<String, DecodeError> {
        let len = self.u32()?;
        let end = self
            .pos
            .checked_add(len)
            .ok_or(DecodeError::UnexpectedEof)?;
        if end > self.bytes.len() {
            return Err(DecodeError::UnexpectedEof);
        }
        let s = std::str::from_utf8(&self.bytes[self.pos..end])
            .map_err(|_| DecodeError::InvalidUtf8)?;
        self.pos = end;
        Ok(s.to_owned())
    }

    /// Guard against absurd counts in corrupt files allocating huge vectors:
    /// each element costs at least one byte of payload.
    fn count(&self, n: usize) -> Result<usize, DecodeError> {
        if n > self.bytes.len() - self.pos {
            return Err(DecodeError::UnexpectedEof);
        }
        Ok(n)
    }

    fn unit(&mut self) -> Result<CompilationUnit, DecodeError> {
        let tag = self.byte()?;
        if tag != format::tag::UNIT {
            return Err(DecodeError::UnknownNodeTag(tag));
        }
        let mut namespaces = {
            let n = self.u32()?;
            self.count(n)?;
            Vec::with_capacity(n)
        };
        for _ in 0..namespaces.capacity() {
            namespaces.push(self.namespace()?);
        }
        let mut members = {
            let n = self.u32()?;
            self.count(n)?;
            Vec::with_capacity(n)
        };
        for _ in 0..members.capacity() {
            members.push(self.member()?);
        }
        let mut aliases = {
            let n = self.u32()?;
            self.count(n)?;
            Vec::with_capacity(n)
        };
        for _ in 0..aliases.capacity() {
            let name = self.string()?;
            let target = self.string()?;
            aliases.push(crate::models::UsingAlias { name, target });
        }
        let mut static_usings = {
            let n = self.u32()?;
            self.count(n)?;
            Vec::with_capacity(n)
        };
        for _ in 0..static_usings.capacity() {
            static_usings.push(self.string()?);
        }
        let mut assembly_blocks = {
            let n = self.u32()?;
            self.count(n)?;
            Vec::with_capacity(n)
        };
        for _ in 0..assembly_blocks.capacity() {
            assembly_blocks.push(self.string()?);
        }
        Ok(CompilationUnit {
            namespaces,
            members,
            aliases,
            static_usings,
            assembly_blocks,
            span: Span::default(),
        })
    }

    fn namespace(&mut self) -> Result<NamespaceDecl, DecodeError> {
        if self.byte()? != format::tag::NAMESPACE {
            return Err(DecodeError::UnknownNodeTag(format::tag::NAMESPACE));
        }
        let name = self.string()?;
        let mut members = {
            let n = self.u32()?;
            self.count(n)?;
            Vec::with_capacity(n)
        };
        for _ in 0..members.capacity() {
            members.push(self.member()?);
        }
        Ok(NamespaceDecl {
            name,
            members,
            span: Span::default(),
        })
    }

    fn member(&mut self) -> Result<Member, DecodeError> {
        let tag = self.byte()?;
        match tag {
            format::tag::CLASS => Ok(Member::Class(self.class()?)),
            format::tag::DELEGATE => Ok(Member::Delegate(self.delegate()?)),
            format::tag::INTERFACE => Ok(Member::Interface(self.interface()?)),
            format::tag::ENUM => {
                let name = self.string()?;
                let underlying_type = self.string()?;
                let variant_count = self.u32()? as usize;
                let mut variants = Vec::with_capacity(variant_count);
                for _ in 0..variant_count {
                    let variant_name = self.string()?;
                    let value = if self.byte()? == format::tag::PRESENT {
                        Some(self.u64()? as i64)
                    } else {
                        None
                    };
                    variants.push((variant_name, value));
                }
                Ok(Member::Enum(EnumDecl {
                    name,
                    underlying_type,
                    variants,
                    span: Span::default(),
                }))
            }
            tag => Err(DecodeError::UnknownNodeTag(tag)),
        }
    }

    fn delegate(&mut self) -> Result<DelegateDecl, DecodeError> {
        let access = match self.byte()? {
            1 => Access::Public,
            2 => Access::Private,
            3 => Access::Protected,
            4 => Access::Internal,
            _ => Access::Default,
        };
        let ret_type = self.string()?;
        let name = self.string()?;
        let mut generic_params = {
            let n = self.u32()?;
            self.count(n)?;
            Vec::with_capacity(n)
        };
        for _ in 0..generic_params.capacity() {
            generic_params.push(self.string()?);
        }
        let mut params = {
            let n = self.u32()?;
            self.count(n)?;
            Vec::with_capacity(n)
        };
        for _ in 0..params.capacity() {
            params.push(self.param()?);
        }
        Ok(DelegateDecl {
            access,
            ret_type,
            name,
            generic_params,
            params,
            span: Span::default(),
        })
    }

    /// One indexer declaration (shared by class members and an
    /// interface's own indexers).
    fn indexer_decl(&mut self) -> Result<IndexerDecl, DecodeError> {
        let access = match self.byte()? {
            1 => Access::Public,
            2 => Access::Private,
            3 => Access::Protected,
            4 => Access::Internal,
            _ => Access::Default,
        };
        let is_settable = self.byte()? != 0;
        let attrs = self.attrs()?;
        let type_name = self.string()?;
        let param_count = self.u32()?;
        let _reserved = self.count(param_count)?;
        let mut params = Vec::with_capacity(param_count as usize);
        for _ in 0..params.capacity() {
            params.push(self.param()?);
        }
        let init = self.expr()?;
        let get = if self.byte()? == format::tag::PRESENT {
            Some(self.block()?)
        } else {
            None
        };
        let set = if self.byte()? == format::tag::PRESENT {
            Some(self.block()?)
        } else {
            None
        };
        let get_expr = if self.byte()? == format::tag::PRESENT {
            Some(self.expr()?)
        } else {
            None
        };
        Ok(IndexerDecl {
            access,
            attrs,
            type_name,
            params,
            init,
            get,
            set,
            get_expr,
            is_settable,
            span: Span::default(),
        })
    }

    fn interface(&mut self) -> Result<InterfaceDecl, DecodeError> {
        let name = self.string()?;
        let mut generic_params = {
            let n = self.u32()?;
            self.count(n)?;
            Vec::with_capacity(n)
        };
        for _ in 0..generic_params.capacity() {
            generic_params.push(self.string()?);
        }
        let mut bases = {
            let n = self.u32()?;
            self.count(n)?;
            Vec::with_capacity(n)
        };
        for _ in 0..bases.capacity() {
            bases.push(self.string()?);
        }
        let mut indexers = {
            let n = self.u32()?;
            self.count(n)?;
            Vec::with_capacity(n)
        };
        for _ in 0..indexers.capacity() {
            indexers.push(self.indexer_decl()?);
        }
        let mut methods = {
            let n = self.u32()?;
            self.count(n)?;
            Vec::with_capacity(n)
        };
        for _ in 0..methods.capacity() {
            let access = match self.byte()? {
                1 => Access::Public,
                2 => Access::Private,
                3 => Access::Protected,
                4 => Access::Internal,
                _ => Access::Default,
            };
            let is_static = match self.byte()? {
                format::tag::PRESENT => true,
                format::tag::ABSENT => false,
                tag => return Err(DecodeError::UnknownNodeTag(tag)),
            };
            let ret_type = self.string()?;
            let m_name = self.string()?;
            let mut generic_params = {
                let n = self.u32()?;
                self.count(n)?;
                Vec::with_capacity(n)
            };
            for _ in 0..generic_params.capacity() {
                generic_params.push(self.string()?);
            }
            let mut params = {
                let n = self.u32()?;
                self.count(n)?;
                Vec::with_capacity(n)
            };
            for _ in 0..params.capacity() {
                params.push(self.param()?);
            }
            let body = if self.byte()? == format::tag::PRESENT {
                Some(self.block()?)
            } else {
                None
            };
            methods.push(MethodSig {
                access,
                ret_type,
                name: m_name,
                generic_params,
                params,
                body,
                is_static,
                span: Span::default(),
            });
        }
        Ok(InterfaceDecl {
            name,
            generic_params,
            methods,
            indexers,
            bases,
            arity: 0,
            is_partial: false,
            span: Span::default(),
        })
    }

    fn class(&mut self) -> Result<ClassDecl, DecodeError> {
        let name = self.string()?;
        let mut generic_params = {
            let n = self.u32()?;
            self.count(n)?;
            Vec::with_capacity(n)
        };
        for _ in 0..generic_params.capacity() {
            generic_params.push(self.string()?);
        }
        let mut primary_ctor_params = {
            let n = self.u32()?;
            self.count(n)?;
            Vec::with_capacity(n)
        };
        for _ in 0..primary_ctor_params.capacity() {
            primary_ctor_params.push(self.string()?);
        }
        let mut bases = {
            let n = self.u32()?;
            self.count(n)?;
            Vec::with_capacity(n)
        };
        for _ in 0..bases.capacity() {
            bases.push(self.string()?);
        }
        let mut members = {
            let n = self.u32()?;
            self.count(n)?;
            Vec::with_capacity(n)
        };
        for _ in 0..members.capacity() {
            members.push(self.class_member()?);
        }
        Ok(ClassDecl {
            name,
            generic_params,
            primary_ctor_params,
            bases,
            members,
            span: Span::default(),
            is_partial: false,
            // Not carried by the v1 AST contract (same as `is_partial`).
            is_struct: false, is_record: false,
            // Cross-assembly unions never cross a bytecode boundary
            // (linking merges sources, not images).
            union_cross_assembly: false,
        })
    }

    fn class_member(&mut self) -> Result<ClassMember, DecodeError> {
        match self.peek_byte() {
            format::tag::FIELD => {
                self.byte()?;
                let access = match self.byte()? {
                    1 => Access::Public,
                    2 => Access::Private,
                    3 => Access::Protected,
                    4 => Access::Internal,
                    _ => Access::Default,
                };
                let is_const = self.byte()? != 0;
                let is_static = self.byte()? != 0;
                let is_byref = self.byte()? != 0;
                let attrs = self.attrs()?;
                let type_name = self.string()?;
                let name = self.string()?;
                let init = self.expr()?;
                // v6: bindings declared while parsing the initializer.
                let n = self.u32()?;
                self.count(n)?;
                let mut init_pre = Vec::with_capacity(n);
                for _ in 0..n {
                    init_pre.push(self.stmt()?);
                }
                Ok(ClassMember::Field(FieldDecl {
                    access,
                    attrs,
                    type_name,
                    name,
                    init,
                    init_pre,
                    is_const,
                    is_static,
                    is_byref,
                    span: Span::default(),
                }))
            }
            format::tag::PROPERTY => {
                self.byte()?;
                let access = match self.byte()? {
                    1 => Access::Public,
                    2 => Access::Private,
                    3 => Access::Protected,
                    4 => Access::Internal,
                    _ => Access::Default,
                };
                let flags = self.byte()?;
                let is_settable = flags & 0b01 != 0;
                let is_init = flags & 0b10 != 0;
                let is_static = flags & 0b100 != 0;
                let is_abstract = flags & 0b1000 != 0;
                let is_override = flags & 0b1_0000 != 0;
                let is_virtual = flags & 0b10_0000 != 0;
                let is_hiding = flags & 0b100_0000 != 0;
                let attrs = self.attrs()?;
                let type_name = self.string()?;
                let name = self.string()?;
                let init = self.expr()?;
                let get = if self.byte()? == format::tag::PRESENT {
                    Some(self.block()?)
                } else {
                    None
                };
                let set = if self.byte()? == format::tag::PRESENT {
                    Some(self.block()?)
                } else {
                    None
                };
                let get_expr = if self.byte()? == format::tag::PRESENT {
                    Some(self.expr()?)
                } else {
                    None
                };
                Ok(ClassMember::Property(PropertyDecl {
                    is_override,
                    is_virtual,
                    is_hiding,
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
                    span: Span::default(),
                }))
            }
            format::tag::INDEXER => {
                self.byte()?;
                let idx = self.indexer_decl()?;
                Ok(ClassMember::Indexer(idx))
            }
            format::tag::EVENT => {
                self.byte()?;
                let access = match self.byte()? {
                    1 => Access::Public,
                    2 => Access::Private,
                    3 => Access::Protected,
                    4 => Access::Internal,
                    _ => Access::Default,
                };
                let attrs = self.attrs()?;
                let type_name = self.string()?;
                let name = self.string()?;
                Ok(ClassMember::Event(EventDecl {
                    access,
                    attrs,
                    type_name,
                    name,
                    span: Span::default(),
                }))
            }
            format::tag::METHOD => {
                self.byte()?;
                Ok(ClassMember::Method(self.method()?))
            }
            format::tag::NESTED => {
                self.byte()?;
                Ok(ClassMember::Nested(self.member()?))
            }
            tag => Err(DecodeError::UnknownNodeTag(tag)),
        }
    }

    /// Attribute list: count then each `[name, arg_count, args...]`.
    fn attrs(&mut self) -> Result<Vec<crate::models::Attribute>, DecodeError> {
        let mut attrs = {
            let n = self.u32()?;
            self.count(n)?;
            Vec::with_capacity(n)
        };
        for _ in 0..attrs.capacity() {
            let name = self.string()?;
            let mut arguments = {
                let n = self.u32()?;
                self.count(n)?;
                Vec::with_capacity(n)
            };
            for _ in 0..arguments.capacity() {
                let arg_name = match self.byte()? {
                    format::tag::PRESENT => Some(self.string()?),
                    _ => None,
                };
                let value = match self.byte()? {
                    format::tag::attr_value::STR => {
                        crate::models::AttributeValue::Str(self.string()?)
                    }
                    format::tag::attr_value::FLOAT => {
                        crate::models::AttributeValue::Float(self.u64()?)
                    }
                    _ => crate::models::AttributeValue::Int(self.u64()? as i64),
                };
                arguments.push(crate::models::AttributeArg {
                    name: arg_name,
                    value,
                    span: Span::default(),
                });
            }
            attrs.push(crate::models::Attribute {
                name,
                arguments,
                span: Span::default(),
            });
        }
        Ok(attrs)
    }

    fn method(&mut self) -> Result<MethodDecl, DecodeError> {
        // METHOD tag already consumed; access byte precedes fields.
        let access = match self.byte()? {
            1 => Access::Public,
            2 => Access::Private,
            3 => Access::Protected,
            4 => Access::Internal,
            _ => Access::Default,
        };
        let attrs = self.attrs()?;
        let ret_type = self.string()?;
        let name = self.string()?;
        let mut generic_params = {
            let n = self.u32()?;
            self.count(n)?;
            Vec::with_capacity(n)
        };
        for _ in 0..generic_params.capacity() {
            generic_params.push(self.string()?);
        }
        let mut params = {
            let n = self.u32()?;
            self.count(n)?;
            Vec::with_capacity(n)
        };
        for _ in 0..params.capacity() {
            params.push(self.param()?);
        }
        let body = match self.byte()? {
            format::tag::PRESENT => Some(self.block()?),
            _ => None,
        };
        let is_override = self.byte()? != 0;
        let is_static = self.byte()? != 0;
        let is_async = self.byte()? != 0;
        let is_extern = self.byte()? != 0;
        // Constructor chain `: base(args)` / `: this(args)`.
        let base_call = match self.byte()? {
            format::tag::PRESENT => {
                let target = self.string()?;
                let n = self.u32()?;
                self.count(n)?;
                let mut args = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    args.push(self.expr()?);
                }
                Some((target, args))
            }
            _ => None,
        };
        Ok(MethodDecl {
            access,
            attrs,
            generic_params,
            constraints: Vec::new(),
            ret_type,
            name,
            params,
            body,
            is_override,
            is_virtual: is_override,
            is_sealed: false,
            is_hiding: false,
            is_static,
            is_abstract: false,
            is_partial: false,
            is_async,
            is_extern,
            base_call,
            span: Span::default(),
        })
    }

    fn param(&mut self) -> Result<Param, DecodeError> {
        let type_name = self.string()?;
        let name = self.string()?;
        let default_value = match self.byte()? {
            format::tag::PRESENT => Some(self.expr()?),
            format::tag::ABSENT => None,
            tag => return Err(DecodeError::UnknownNodeTag(tag)),
        };
        // Extension-method receiver marker.
        let is_this = match self.byte()? {
            format::tag::PRESENT => true,
            format::tag::ABSENT => false,
            tag => return Err(DecodeError::UnknownNodeTag(tag)),
        };
        // Parameter attributes (see encode).
        let attrs = self.attrs()?;
        // By-reference marker (see encode).
        let is_byref = match self.byte()? {
            format::tag::PRESENT => true,
            format::tag::ABSENT => false,
            tag => return Err(DecodeError::UnknownNodeTag(tag)),
        };
        Ok(Param {
            type_name,
            name,
            span: Span::default(),
            default_value,
            is_this,
            // Not carried by the v1 AST contract (same as `is_struct`).
            is_params: false,
            is_byref,
            attrs,
        })
    }

    fn block(&mut self) -> Result<Block, DecodeError> {
        let mut stmts = {
            let n = self.u32()?;
            self.count(n)?;
            Vec::with_capacity(n)
        };
        for _ in 0..stmts.capacity() {
            stmts.push(self.stmt()?);
        }
        Ok(Block {
            stmts,
            span: Span::default(),
        })
    }

    fn stmt(&mut self) -> Result<Stmt, DecodeError> {
        if self.peek_byte() == format::tag::RETURN {
            self.byte()?;
            let has_value = self.byte()? == format::tag::PRESENT;
            let value = if has_value { Some(self.expr()?) } else { None };
            return Ok(Stmt::Return {
                value,
                span: Span::default(),
            });
        }
        match self.peek_byte() {
            format::tag::LET => {
                self.byte()?;
                let type_name = self.string()?;
                let name = self.string()?;
                let access = match self.byte()? {
                    1 => Access::Public,
                    2 => Access::Private,
                    3 => Access::Protected,
                    4 => Access::Internal,
                    _ => Access::Default,
                };
                let is_byref = match self.byte()? {
                    format::tag::PRESENT => true,
                    format::tag::ABSENT => false,
                    tag => return Err(DecodeError::UnknownNodeTag(tag)),
                };
                let is_const = match self.byte()? {
                    format::tag::PRESENT => true,
                    format::tag::ABSENT => false,
                    tag => return Err(DecodeError::UnknownNodeTag(tag)),
                };
                let init = self.expr()?;
                Ok(Stmt::Let {
                    is_const,
                    type_name,
                    name,
                    init,
                    access,
                    is_byref,
                    span: Span::default(),
                })
            }
            format::tag::MEMBER_ASSIGN => {
                self.byte()?;
                let object = self.expr()?;
                let name = self.string()?;
                let value = self.expr()?;
                Ok(Stmt::MemberAssign {
                    object,
                    name,
                    value,
                    span: Span::default(),
                })
            }
            format::tag::ASSIGN => {
                self.byte()?;
                let name = self.string()?;
                let value = self.expr()?;
                Ok(Stmt::Assign {
                    name,
                    value,
                    span: Span::default(),
                })
            }
            format::tag::stmt::IF => {
                self.byte()?;
                let cond = self.expr()?;
                let then_block = self.block()?;
                let else_block = if self.byte()? == format::tag::PRESENT {
                    Some(self.block()?)
                } else {
                    None
                };
                Ok(Stmt::If {
                    cond,
                    then_block,
                    else_block,
                    span: Span::default(),
                })
            }
            format::tag::stmt::WHILE => {
                self.byte()?;
                let cond = self.expr()?;
                let body = self.block()?;
                Ok(Stmt::While {
                    cond,
                    body,
                    span: Span::default(),
                })
            }
            format::tag::stmt::CHECKED_BLOCK => {
                self.byte()?;
                let body = self.block()?;
                Ok(Stmt::CheckedBlock {
                    body,
                    span: Span::default(),
                })
            }
            format::tag::stmt::BLOCK => {
                self.byte()?;
                let body = self.block()?;
                Ok(Stmt::Block {
                    body,
                    span: Span::default(),
                })
            }
            format::tag::stmt::DO_WHILE => {
                self.byte()?;
                let body = self.block()?;
                let cond = self.expr()?;
                Ok(Stmt::DoWhile {
                    body,
                    cond,
                    span: Span::default(),
                })
            }
            format::tag::stmt::FOR => {
                self.byte()?;
                let init = if self.byte()? == format::tag::PRESENT {
                    Some(Box::new(self.stmt()?))
                } else {
                    None
                };
                let cond = if self.byte()? == format::tag::PRESENT {
                    Some(self.expr()?)
                } else {
                    None
                };
                let step = if self.byte()? == format::tag::PRESENT {
                    Some(Box::new(self.stmt()?))
                } else {
                    None
                };
                let body = self.block()?;
                Ok(Stmt::For {
                    init,
                    cond,
                    step,
                    body,
                    span: Span::default(),
                })
            }
            format::tag::stmt::SWITCH => {
                self.byte()?;
                let subject = self.expr()?;
                let arm_count = self.u64()? as usize;
                let mut arms = Vec::with_capacity(arm_count);
                for _ in 0..arm_count {
                    let label_count = self.u64()? as usize;
                    let mut labels = Vec::with_capacity(label_count);
                    for _ in 0..label_count {
                        labels.push(self.expr()?);
                    }
                    let body = self.block()?;
                    let guards = vec![None; labels.len()];
                    arms.push(crate::models::SwitchArm { labels, guards, binding: None, body });
                }
                let default_body = if self.byte()? == format::tag::PRESENT {
                    Some(self.block()?)
                } else {
                    None
                };
                Ok(Stmt::Switch {
                    subject,
                    arms,
                    default_body,
                    span: Span::default(),
                })
            }
            format::tag::stmt::BREAK => {
                self.byte()?;
                Ok(Stmt::Break {
                    span: Span::default(),
                })
            }
            format::tag::stmt::CONTINUE => {
                self.byte()?;
                Ok(Stmt::Continue {
                    span: Span::default(),
                })
            }
            format::tag::stmt::LABEL => {
                self.byte()?;
                let name = self.string()?;
                Ok(Stmt::Label {
                    name,
                    span: Span::default(),
                })
            }
            format::tag::stmt::FIXED => {
                self.byte()?;
                let n = self.u32()? as usize;
                self.count(n)?;
                let mut declarators = Vec::with_capacity(n);
                for _ in 0..n {
                    let ty = self.string()?;
                    let name = self.string()?;
                    let init = if self.byte()? == format::tag::PRESENT {
                        Some(self.expr()?)
                    } else {
                        None
                    };
                    declarators.push((ty, name, init));
                }
                let body = self.block()?;
                Ok(Stmt::Fixed {
                    declarators,
                    body,
                    span: Span::default(),
                })
            }
            format::tag::stmt::GOTO => {
                self.byte()?;
                let label = if self.byte()? == format::tag::PRESENT {
                    Some(self.string()?)
                } else {
                    None
                };
                let case_expr = if self.byte()? == format::tag::PRESENT {
                    Some(self.expr()?)
                } else {
                    None
                };
                Ok(Stmt::Goto {
                    label,
                    case_expr,
                    span: Span::default(),
                })
            }
            format::tag::stmt::TRY => {
                self.byte()?;
                let body = self.block()?;
                let catch_count = self.u64()? as usize;
                let mut catches = Vec::with_capacity(catch_count);
                for _ in 0..catch_count {
                    let ty = if self.byte()? == format::tag::PRESENT {
                        Some(self.string()?)
                    } else {
                        None
                    };
                    let name = if self.byte()? == format::tag::PRESENT {
                        Some(self.string()?)
                    } else {
                        None
                    };
                    let cbody = self.block()?;
                    catches.push(CatchClause {
                        filter: None,
                        ty,
                        name,
                        body: cbody,
                        span: Span::default(),
                    });
                }
                let finally_block = if self.byte()? == format::tag::PRESENT {
                    Some(self.block()?)
                } else {
                    None
                };
                Ok(Stmt::Try {
                    body,
                    catches,
                    finally_block,
                    span: Span::default(),
                })
            }
            format::tag::stmt::THROW => {
                self.byte()?;
                let expr = if self.byte()? == format::tag::PRESENT {
                    Some(self.expr()?)
                } else {
                    None
                };
                Ok(Stmt::Throw {
                    expr,
                    span: Span::default(),
                })
            }
            format::tag::stmt::LOCAL_FUNCTION => {
                self.byte()?;
                let name = self.string()?;
                let mut params = Vec::new();
                let count = self.u32()?;
                for _ in 0..count {
                    params.push(self.param()?);
                }
                let ret_type = self.string()?;
                let body = self.block()?;
                let is_static = self.byte()? != 0;
                let is_extern = self.byte()?;
                let attrs = self.attrs()?;
                Ok(Stmt::LocalFunction {
                    is_extern: is_extern != 0,
                    attrs,
                    name,
                    params,
                    ret_type,
                    body,
                    is_static,
                    span: Span::default(),
                })
            }
            _ => Ok(Stmt::Expression(self.expr()?)),
        }
    }

    /// Look at the next byte without consuming it.
    fn peek_byte(&self) -> u8 {
        self.bytes.get(self.pos).copied().unwrap_or(0)
    }

    fn binop(&mut self) -> Result<BinOp, DecodeError> {
        Ok(match self.byte()? {
            format::tag::binop::SUB => BinOp::Sub,
            format::tag::binop::MUL => BinOp::Mul,
            format::tag::binop::DIV => BinOp::Div,
            format::tag::binop::EQ => BinOp::Eq,
            format::tag::binop::NE => BinOp::Ne,
            format::tag::binop::LT => BinOp::Lt,
            format::tag::binop::GT => BinOp::Gt,
            format::tag::binop::LE => BinOp::Le,
            format::tag::binop::GE => BinOp::Ge,
            format::tag::binop::AND => BinOp::And,
            format::tag::binop::OR => BinOp::Or,
            format::tag::binop::REM => BinOp::Rem,
            format::tag::binop::BIT_AND => BinOp::BitAnd,
            format::tag::binop::BIT_OR => BinOp::BitOr,
            format::tag::binop::BIT_XOR => BinOp::BitXor,
            format::tag::binop::SHL => BinOp::Shl,
            format::tag::binop::SHR => BinOp::Shr,
            format::tag::binop::SHR_SHR => BinOp::ShrShr,
            format::tag::binop::NULL_COALESCE => BinOp::NullCoalesce,
            format::tag::binop::IS => BinOp::Is,
            _ => BinOp::Add,
        })
    }

    fn expr(&mut self) -> Result<Expr, DecodeError> {
        let tag = self.peek_byte();
        if tag == format::tag::BIN || tag == format::tag::UNARY {
            self.byte()?;
            return match tag {
                format::tag::BIN => {
                    let op = self.binop()?;
                    let lhs = self.expr()?;
                    let rhs = self.expr()?;
                    Ok(Expr::Binary {
                        op,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                        span: Span::default(),
                    })
                }
                _ => {
                    let op = match self.byte()? {
                        format::tag::unop::NOT => UnOp::Not,
                        format::tag::unop::BIT_NOT => UnOp::BitNot,
                        format::tag::unop::DEREF => UnOp::Deref,
                        format::tag::unop::ADDR_OF => UnOp::AddrOf,
                        format::tag::unop::THROW => UnOp::Throw,
                        format::tag::unop::INDEX_FROM_END => UnOp::IndexFromEnd,
                        _ => UnOp::Neg,
                    };
                    let inner = self.expr()?;
                    Ok(Expr::Unary {
                        op,
                        inner: Box::new(inner),
                        span: Span::default(),
                    })
                }
            };
        }
        match self.byte()? {
            format::tag::CALL => {
                let callee = self.string()?;
                let mut args = {
                    let n = self.u32()?;
                    self.count(n)?;
                    Vec::with_capacity(n)
                };
                for _ in 0..args.capacity() {
                    args.push(self.expr()?);
                }
                Ok(Expr::Call {
                    callee,
                    args,
                    span: Span::default(),
                })
            }
            format::tag::STR => Ok(Expr::Literal(Literal::Str(self.string()?))),
            format::tag::INT => Ok(Expr::Literal(Literal::Int(self.u64()? as i64))),
            format::tag::CHAR => Ok(Expr::Literal(Literal::Char(self.u32()? as u32))),
            format::tag::FLOAT => Ok(Expr::Literal(Literal::Float(
                f64::from_bits(self.u64()?),
            ))),
            format::tag::BOOL => Ok(Expr::Literal(Literal::Bool(self.byte()? != 0))),
            format::tag::NULL => Ok(Expr::Literal(Literal::Null)),
            format::tag::NEW => {
                let class_name = self.string()?;
                let n = self.u32()?;
                let mut args = Vec::with_capacity(n);
                for _ in 0..n {
                    args.push(self.expr()?);
                }
                let m = self.u32()?;
                let mut init = Vec::with_capacity(m);
                for _ in 0..m {
                    let name = self.string()?;
                    let value = self.expr()?;
                    init.push(crate::models::ast::MemberInit {
                        index: None,
                        name,
                        value,
                        span: Span::default(),
                    });
                }
                Ok(Expr::New {
                    class_name,
                    args,
                    init,
                    span: Span::default(),
                })
            }
            format::tag::MEMBER => {
                let object = Box::new(self.expr()?);
                let name = self.string()?;
                Ok(Expr::Member {
                    object,
                    name,
                    span: Span::default(),
                })
            }
            format::tag::NEW_ARRAY => {
                let elem_type = self.string()?;
                let size = Box::new(self.expr()?);
                // v3: collection-expression initializers follow the
                // size (absent in v2 payloads — but the version gate
                // rejects those before decode, so unguarded read is
                // exact).
                let n = self.u32()?;
                let mut init = Vec::with_capacity(n);
                for _ in 0..n {
                    init.push(self.expr()?);
                }
                Ok(Expr::NewArray { elem_type, size, init, span: Span::default() })
            }
            format::tag::INDEX => {
                let object = Box::new(self.expr()?);
                let index = Box::new(self.expr()?);
                let is_range = self.byte()? != 0;
                Ok(Expr::Index { object, index, extra: Vec::new(), is_range, span: Span::default() })
            }
            format::tag::TERNARY => {
                let cond = Box::new(self.expr()?);
                let then_expr = Box::new(self.expr()?);
                let else_expr = Box::new(self.expr()?);
                Ok(Expr::Ternary {
                    cond,
                    then_expr,
                    else_expr,
                    span: Span::default(),
                })
            }
            format::tag::IDENT => Ok(Expr::Ident {
                name: self.string()?,
                span: Span::default(),
            }),
            format::tag::CAST => {
                let type_name = self.string()?;
                let expr = Box::new(self.expr()?);
                Ok(Expr::Cast { type_name, expr, span: Span::default() })
            }
            format::tag::SWITCH_EXPR => {
                let subject = Box::new(self.expr()?);
                let n = self.u32()?;
                self.count(n)?;
                let mut arms = Vec::with_capacity(n);
                for _ in 0..n {
                    let label_count = self.u32()?;
                    self.count(label_count)?;
                    let mut labels = Vec::with_capacity(label_count);
                    for _ in 0..label_count {
                        labels.push(self.expr()?);
                    }
                    let guard = if self.byte()? == format::tag::PRESENT {
                        Some(self.expr()?)
                    } else {
                        None
                    };
                    let body = Box::new(self.expr()?);
                    arms.push(SwitchExprArm { labels, guard, body });
                }
                Ok(Expr::SwitchExpr {
                    subject,
                    arms,
                    span: Span::default(),
                })
            }
            format::tag::METHOD_CALL => {
                let object = Box::new(self.expr()?);
                let name = self.string()?;
                let mut args = {
                    let n = self.u32()?;
                    self.count(n)?;
                    Vec::with_capacity(n)
                };
                for _ in 0..args.capacity() {
                    args.push(self.expr()?);
                }
                Ok(Expr::MethodCall {
                    object,
                    name,
                    args,
                    span: Span::default(),
                })
            }
            format::tag::ASSIGN_COMPLEX => {
                let lhs = Box::new(self.expr()?);
                let value = Box::new(self.expr()?);
                Ok(Expr::AssignComplex {
                    lhs,
                    value,
                    span: Span::default(),
                })
            }
            format::tag::EXPR_ASSIGN => {
                let target = Box::new(self.expr()?);
                let value = Box::new(self.expr()?);
                Ok(Expr::Assign {
                    target,
                    value,
                    span: Span::default(),
                })
            }
            format::tag::POST_INC => {
                let target = Box::new(self.expr()?);
                Ok(Expr::PostInc {
                    target,
                    span: Span::default(),
                })
            }
            format::tag::POST_DEC => {
                let target = Box::new(self.expr()?);
                Ok(Expr::PostDec {
                    target,
                    span: Span::default(),
                })
            }
            format::tag::DEFAULT => {
                let type_name = self.string()?;
                Ok(Expr::Default {
                    type_name,
                    span: Span::default(),
                })
            }
            format::tag::LAMBDA => {
                // Written by the encoder purely so the round-trip fails
                // loudly. Decoding cannot reconstruct a closure, and
                // guessing one would be worse than refusing.
                return Err(DecodeError::UnknownNodeTag(format::tag::LAMBDA));
            }
            format::tag::CHECKED => {
                let inner = self.expr()?;
                Ok(Expr::Checked {
                    inner: Box::new(inner),
                    span: Span::default(),
                })
            }
            format::tag::AWAIT => {
                let inner = Box::new(self.expr()?);
                Ok(Expr::Await {
                    inner,
                    span: Span::default(),
                })
            }
            format::tag::INTERP_STRING => {
                let n = self.u32()?;
                self.count(n)?;
                let mut parts = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    let kind = self.byte()?;
                    match kind {
                        0 => parts.push(InterpPart::Text(self.string()?)),
                        1 => parts.push(InterpPart::Hole(Box::new(self.expr()?))),
                        _ => return Err(DecodeError::UnknownNodeTag(kind)),
                    }
                }
                Ok(Expr::InterpolatedString {
                    parts,
                    span: Span::default(),
                })
            }
            tag => Err(DecodeError::UnknownNodeTag(tag)),
        }
    }
}
