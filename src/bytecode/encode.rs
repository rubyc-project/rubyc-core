//! AST → bytecode container bytes.

use crate::bytecode::format::{self, FORMAT_VERSION};
use crate::models::{
    Access, BinOp, Block, ClassDecl, ClassMember, CompilationUnit, DelegateDecl, Expr, Literal,
    Member, MethodDecl, NamespaceDecl, Param, Stmt, UnOp,
};
use crate::native::container::{self as container_mod, Kind};

/// Encode a compilation unit into portable bytecode bytes.
///
/// All integers are little-endian and fixed-width; strings are
/// `u32`-length-prefixed UTF-8. Spans are not part of the format: they are
/// parse artifacts, not program meaning.
pub fn encode(unit: &CompilationUnit) -> Vec<u8> {
    let out = container_mod::build(Kind::Bytecode, FORMAT_VERSION);

    let mut w = Writer { out };
    w.unit(unit);
    w.out
}

/// Encode a compilation unit to raw payload bytes with no RubyC preamble.
///
/// Used by `--no-magic`: the bytes are the bare unit payload (starting with
/// the `KIND` byte), with no `jump-over-magic` + dog + kind/version header.
/// Decoding is tolerant — [`crate::bytecode::decode`] auto-detects the absent
/// preamble and parses the raw payload.
pub fn encode_no_magic(unit: &CompilationUnit) -> Vec<u8> {
    let mut w = Writer { out: Vec::new() };
    w.unit(unit);
    w.out
}

struct Writer {
    out: Vec<u8>,
}

impl Writer {
    fn byte(&mut self, b: u8) {
        self.out.push(b);
    }

    fn u32(&mut self, v: usize) {
        self.out.extend_from_slice(&(v as u32).to_le_bytes());
    }

    fn u64(&mut self, v: u64) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    fn string(&mut self, s: &str) {
        self.u32(s.len());
        self.out.extend_from_slice(s.as_bytes());
    }

    fn unit(&mut self, unit: &CompilationUnit) {
        self.byte(format::tag::UNIT);
        self.u32(unit.namespaces.len());
        for ns in &unit.namespaces {
            self.namespace(ns);
        }
        self.u32(unit.members.len());
        for member in &unit.members {
            self.member(member);
        }
        self.u32(unit.aliases.len());
        for alias in &unit.aliases {
            self.string(&alias.name);
            self.string(&alias.target);
        }
        self.u32(unit.static_usings.len());
        for target in &unit.static_usings {
            self.string(target);
        }
        self.u32(unit.assembly_blocks.len());
        for block in &unit.assembly_blocks {
            self.string(block);
        }
    }

    fn namespace(&mut self, ns: &NamespaceDecl) {
        self.byte(format::tag::NAMESPACE);
        self.string(&ns.name);
        self.u32(ns.members.len());
        for member in &ns.members {
            self.member(member);
        }
    }

    fn member(&mut self, member: &Member) {
        match member {
            Member::Enum(enum_decl) => {
                self.byte(format::tag::ENUM);
                self.string(&enum_decl.name);
                self.string(&enum_decl.underlying_type);
                self.u32(enum_decl.variants.len());
                for (name, value) in &enum_decl.variants {
                    self.string(name);
                    match value {
                        Some(v) => {
                            self.byte(format::tag::PRESENT);
                            self.u64(*v as u64);
                        }
                        None => self.byte(format::tag::ABSENT),
                    }
                }
            }
            Member::Class(class) => {
                self.byte(format::tag::CLASS);
                self.class(class);
            }
            Member::Delegate(delegate) => {
                self.byte(format::tag::DELEGATE);
                self.delegate(delegate);
            }
            Member::Interface(iface) => {
                self.byte(format::tag::INTERFACE);
                self.string(&iface.name);
                self.u32(iface.generic_params.len());
                for gp in &iface.generic_params {
                    self.string(gp);
                }
                self.u32(iface.bases.len());
                for b in &iface.bases {
                    self.string(b);
                }
                self.u32(iface.indexers.len());
                for idx in &iface.indexers {
                    self.indexer_decl(idx);
                }
                self.u32(iface.methods.len());
                for sig in &iface.methods {
                    let access_byte = match sig.access {
                        crate::models::Access::Public => 1,
                        crate::models::Access::Private => 2,
                        crate::models::Access::Protected => 3,
                        crate::models::Access::Internal => 4,
                        _ => 0,
                    };
                    self.byte(access_byte);
                    self.byte(if sig.is_static {
                        format::tag::PRESENT
                    } else {
                        format::tag::ABSENT
                    });
                    self.string(&sig.ret_type);
                    self.string(&sig.name);
                    self.u32(sig.generic_params.len());
                    for gp in &sig.generic_params {
                        self.string(gp);
                    }
                    self.u32(sig.params.len());
                    for param in &sig.params {
                        self.param(param);
                    }
                    match &sig.body {
                        None => self.byte(format::tag::ABSENT),
                        Some(block) => {
                            self.byte(format::tag::PRESENT);
                            self.block(block);
                        }
                    }
                }
            }
        }
    }

    fn class(&mut self, class: &ClassDecl) {
        self.string(&class.name);
        self.u32(class.generic_params.len());
        for gp in &class.generic_params {
            self.string(gp);
        }
        self.u32(class.primary_ctor_params.len());
        for pp in &class.primary_ctor_params {
            self.string(pp);
        }
        self.u32(class.bases.len());
        for base in &class.bases {
            self.string(base);
        }
        self.u32(class.members.len());
        for member in &class.members {
            match member {
                ClassMember::Method(method) => {
                    self.byte(format::tag::METHOD);
                    let access_byte = match method.access {
                        crate::models::Access::Public => 1,
                        crate::models::Access::Private => 2,
                        crate::models::Access::Protected => 3,
                        crate::models::Access::Internal => 4,
                        _ => 0,
                    };
                    self.byte(access_byte);
                    self.attrs(&method.attrs);
                    self.method(method);
                }
                ClassMember::Field(field) => {
                    self.byte(format::tag::FIELD);
                    let access_byte = match field.access {
                        crate::models::Access::Public => 1,
                        crate::models::Access::Private => 2,
                        crate::models::Access::Protected => 3,
                        crate::models::Access::Internal => 4,
                        _ => 0,
                    };
                    self.byte(access_byte);
                    self.byte(field.is_const as u8);
                    self.byte(field.is_static as u8);
                    self.byte(field.is_byref as u8);
                    self.attrs(&field.attrs);
                    self.string(&field.type_name);
                    self.string(&field.name);
                    self.expr(&field.init);
                    // v6: bindings declared while parsing the
                    // initializer (`out` vars, `is` patterns).
                    self.u32(field.init_pre.len());
                    for stmt in &field.init_pre {
                        self.stmt(stmt);
                    }
                }
                ClassMember::Property(prop) => {
                    self.byte(format::tag::PROPERTY);
                    let access_byte = match prop.access {
                        crate::models::Access::Public => 1,
                        crate::models::Access::Private => 2,
                        crate::models::Access::Protected => 3,
                        crate::models::Access::Internal => 4,
                        _ => 0,
                    };
                    self.byte(access_byte);
                    let mut flags = 0u8;
                    if prop.is_settable {
                        flags |= 0b01;
                    }
                    if prop.is_init {
                        flags |= 0b10;
                    }
                    if prop.is_static {
                        flags |= 0b100;
                    }
                    if prop.is_abstract {
                        flags |= 0b1000;
                    }
                    if prop.is_override {
                        flags |= 0b1_0000;
                    }
                    if prop.is_virtual {
                        flags |= 0b10_0000;
                    }
                    if prop.is_hiding {
                        flags |= 0b100_0000;
                    }
                    self.byte(flags);
                    self.attrs(&prop.attrs);
                    self.string(&prop.type_name);
                    self.string(&prop.name);
                    self.expr(&prop.init);
                    match &prop.get {
                        Some(block) => {
                            self.byte(format::tag::PRESENT);
                            self.block(block);
                        }
                        None => self.byte(format::tag::ABSENT),
                    }
                    match &prop.set {
                        Some(block) => {
                            self.byte(format::tag::PRESENT);
                            self.block(block);
                        }
                        None => self.byte(format::tag::ABSENT),
                    }
                    match &prop.get_expr {
                        Some(expr) => {
                            self.byte(format::tag::PRESENT);
                            self.expr(expr);
                        }
                        None => self.byte(format::tag::ABSENT),
                    }
                }
                ClassMember::Indexer(idx) => {
                    self.byte(format::tag::INDEXER);
                    self.indexer_decl(idx);
                }
                ClassMember::Event(evt) => {
                    self.byte(format::tag::EVENT);
                    let access_byte = match evt.access {
                        crate::models::Access::Public => 1,
                        crate::models::Access::Private => 2,
                        crate::models::Access::Protected => 3,
                        crate::models::Access::Internal => 4,
                        _ => 0,
                    };
                    self.byte(access_byte);
                    self.attrs(&evt.attrs);
                    self.string(&evt.type_name);
                    self.string(&evt.name);
                }
                ClassMember::Nested(nested) => {
                    self.byte(format::tag::NESTED);
                    self.member(nested);
                }
            }
        }
    }

    fn delegate(&mut self, delegate: &DelegateDecl) {
        let access_byte = match delegate.access {
            Access::Public => 1,
            Access::Private => 2,
            Access::Protected => 3,
            Access::Internal => 4,
            _ => 0,
        };
        self.byte(access_byte);
        self.string(&delegate.ret_type);
        self.string(&delegate.name);
        self.u32(delegate.generic_params.len());
        for gp in &delegate.generic_params {
            self.string(gp);
        }
        self.u32(delegate.params.len());
        for param in &delegate.params {
            self.param(param);
        }
    }

    /// Attribute list: count then each `[name, arg_count, args...]`.
    fn attrs(&mut self, attrs: &[crate::models::Attribute]) {
        self.u32(attrs.len());
        for attr in attrs {
            self.string(&attr.name);
            self.u32(attr.arguments.len());
            for arg in &attr.arguments {
                match &arg.name {
                    Some(n) => {
                        self.byte(format::tag::PRESENT);
                        self.string(n);
                    }
                    None => self.byte(format::tag::ABSENT),
                }
                match &arg.value {
                    crate::models::AttributeValue::Str(s) => {
                        self.byte(format::tag::attr_value::STR);
                        self.string(s);
                    }
                    crate::models::AttributeValue::Int(v) => {
                        self.byte(format::tag::attr_value::INT);
                        self.u64(*v as u64);
                    }
                    crate::models::AttributeValue::Float(bits) => {
                        self.byte(format::tag::attr_value::FLOAT);
                        self.u64(*bits);
                    }
                }
            }
        }
    }

    fn method(&mut self, method: &MethodDecl) {
        self.string(&method.ret_type);
        self.string(&method.name);
        self.u32(method.generic_params.len());
        for gp in &method.generic_params {
            self.string(gp);
        }
        self.u32(method.params.len());
        for param in &method.params {
            self.param(param);
        }
        // Bodyless declarations (import stubs) encode ABSENT.
        match &method.body {
            Some(block) => {
                self.byte(format::tag::PRESENT);
                self.block(block);
            }
            None => self.byte(format::tag::ABSENT),
        }
        self.byte(method.is_override as u8);
        self.byte(method.is_static as u8);
        self.byte(method.is_async as u8);
        self.byte(method.is_extern as u8);
        // Constructor chain `: base(args)` / `: this(args)`.
        match &method.base_call {
            Some((target, args)) => {
                self.byte(format::tag::PRESENT);
                self.string(target);
                self.u32(args.len());
                for a in args {
                    self.expr(a);
                }
            }
            None => self.byte(format::tag::ABSENT),
        }
    }

    fn param(&mut self, param: &Param) {
        self.string(&param.type_name);
        self.string(&param.name);
        match &param.default_value {
            Some(expr) => {
                self.byte(format::tag::PRESENT);
                self.expr(expr);
            }
            None => self.byte(format::tag::ABSENT),
        }
        // Extension-method receiver marker.
        self.byte(if param.is_this {
            format::tag::PRESENT
        } else {
            format::tag::ABSENT
        });
        // Parameter attributes (`[UnsafeAccessorType("...")]`).
        self.attrs(&param.attrs);
        // By-reference marker (`ref`/`out`/`in`): load-bearing for
        // lowering (callers box, callees dereference), unlike
        // `is_params` which the v1 contract drops.
        self.byte(if param.is_byref {
            format::tag::PRESENT
        } else {
            format::tag::ABSENT
        });
    }

    fn block(&mut self, block: &Block) {
        self.u32(block.stmts.len());
        for stmt in &block.stmts {
            self.stmt(stmt);
        }
    }

    fn stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Expression(expr) => self.expr(expr),
            Stmt::Return { value, .. } => {
                self.byte(format::tag::RETURN);
                match value {
                    Some(expr) => {
                        self.byte(format::tag::PRESENT);
                        self.expr(expr);
                    }
                    None => self.byte(format::tag::ABSENT),
                }
            }
            Stmt::MemberAssign {
                object,
                name,
                value,
                ..
            } => {
                self.byte(format::tag::MEMBER_ASSIGN);
                self.expr(object);
                self.string(name);
                self.expr(value);
            }
            Stmt::Assign { name, value, .. } => {
                self.byte(format::tag::ASSIGN);
                self.string(name);
                self.expr(value);
            }
            // The parser rewrites using declarations into a Let plus a
            // try/finally before any backend sees them; encoding the
            // declaration shape as its Let keeps the encoder total.
            Stmt::UsingDecl {
                type_name,
                name,
                init,
                ..
            } => {
                self.byte(format::tag::LET);
                self.string(type_name);
                self.string(name);
                self.byte(crate::models::Access::Default as u8);
                self.byte(format::tag::ABSENT);
                self.expr(init);
            }
            Stmt::Let {
                type_name,
                name,
                init,
                access,
                is_byref,
                is_const,
                ..
            } => {
                self.byte(format::tag::LET);
                self.string(type_name);
                self.string(name);
                self.byte(*access as u8);
                self.byte(if *is_byref {
                    format::tag::PRESENT
                } else {
                    format::tag::ABSENT
                });
                self.byte(if *is_const {
                    format::tag::PRESENT
                } else {
                    format::tag::ABSENT
                });
                self.expr(init);
            }
            Stmt::If {
                cond,
                then_block,
                else_block,
                ..
            } => {
                self.byte(format::tag::stmt::IF);
                self.expr(cond);
                self.block(then_block);
                match else_block {
                    Some(b) => {
                        self.byte(format::tag::PRESENT);
                        self.block(b);
                    }
                    None => self.byte(format::tag::ABSENT),
                }
            }
            Stmt::While { cond, body, .. } => {
                self.byte(format::tag::stmt::WHILE);
                self.expr(cond);
                self.block(body);
            }
            Stmt::CheckedBlock { body, .. } => {
                self.byte(format::tag::stmt::CHECKED_BLOCK);
                self.block(body);
            }
            Stmt::Block { body, .. } => {
                self.byte(format::tag::stmt::BLOCK);
                self.block(body);
            }
            Stmt::DoWhile { body, cond, .. } => {
                self.byte(format::tag::stmt::DO_WHILE);
                self.block(body);
                self.expr(cond);
            }
            Stmt::For {
                init,
                cond,
                step,
                body,
                ..
            } => {
                self.byte(format::tag::stmt::FOR);
                match init {
                    Some(s) => {
                        self.byte(format::tag::PRESENT);
                        self.stmt(s);
                    }
                    None => self.byte(format::tag::ABSENT),
                }
                match cond {
                    Some(e) => {
                        self.byte(format::tag::PRESENT);
                        self.expr(e);
                    }
                    None => self.byte(format::tag::ABSENT),
                }
                match step {
                    Some(s) => {
                        self.byte(format::tag::PRESENT);
                        self.stmt(s);
                    }
                    None => self.byte(format::tag::ABSENT),
                }
                self.block(body);
            }
            Stmt::Switch {
                subject,
                arms,
                default_body,
                ..
            } => {
                self.byte(format::tag::stmt::SWITCH);
                self.expr(subject);
                self.u64(arms.len() as u64);
                for arm in arms {
                    self.u64(arm.labels.len() as u64);
                    for label in &arm.labels {
                        self.expr(label);
                    }
                    self.block(&arm.body);
                }
                match default_body {
                    Some(b) => {
                        self.byte(format::tag::PRESENT);
                        self.block(b);
                    }
                    None => self.byte(format::tag::ABSENT),
                }
            }
            Stmt::Break { .. } => self.byte(format::tag::stmt::BREAK),
            Stmt::Continue { .. } => self.byte(format::tag::stmt::CONTINUE),
            Stmt::Label { name, .. } => {
                self.byte(format::tag::stmt::LABEL);
                self.string(name);
            }
            Stmt::Fixed {
                declarators, body, ..
            } => {
                self.byte(format::tag::stmt::FIXED);
                self.u32(declarators.len());
                for (ty, name, init) in declarators {
                    self.string(ty);
                    self.string(name);
                    match init {
                        Some(e) => {
                            self.byte(format::tag::PRESENT);
                            self.expr(e);
                        }
                        None => self.byte(format::tag::ABSENT),
                    }
                }
                self.block(body);
            }
            Stmt::Goto {
                label,
                case_expr,
                ..
            } => {
                self.byte(format::tag::stmt::GOTO);
                match label {
                    Some(l) => {
                        self.byte(format::tag::PRESENT);
                        self.string(l);
                    }
                    None => self.byte(format::tag::ABSENT),
                }
                match case_expr {
                    Some(e) => {
                        self.byte(format::tag::PRESENT);
                        self.expr(e);
                    }
                    None => self.byte(format::tag::ABSENT),
                }
            }
            Stmt::Try {
                body,
                catches,
                finally_block,
                ..
            } => {
                self.byte(format::tag::stmt::TRY);
                self.block(body);
                self.u64(catches.len() as u64);
                for c in catches {
                    match &c.ty {
                        Some(t) => {
                            self.byte(format::tag::PRESENT);
                            self.string(t);
                        }
                        None => self.byte(format::tag::ABSENT),
                    }
                    match &c.name {
                        Some(n) => {
                            self.byte(format::tag::PRESENT);
                            self.string(n);
                        }
                        None => self.byte(format::tag::ABSENT),
                    }
                    self.block(&c.body);
                }
                match finally_block {
                    Some(b) => {
                        self.byte(format::tag::PRESENT);
                        self.block(b);
                    }
                    None => self.byte(format::tag::ABSENT),
                }
            }
            Stmt::Throw { expr, .. } => {
                self.byte(format::tag::stmt::THROW);
                match expr {
                    Some(e) => {
                        self.byte(format::tag::PRESENT);
                        self.expr(e);
                    }
                    None => self.byte(format::tag::ABSENT),
                }
            }
            Stmt::LocalFunction {
                name,
                params,
                ret_type,
                body,
                is_static,
                is_extern,
                attrs,
                ..
            } => {
                self.byte(format::tag::stmt::LOCAL_FUNCTION);
                self.string(name);
                self.u32(params.len());
                for p in params {
                    self.param(p);
                }
                self.string(ret_type);
                self.block(body);
                self.byte(if *is_static { 1 } else { 0 });
                self.byte(if *is_extern { 1 } else { 0 });
                self.attrs(attrs);
            }
        }
    }

    /// One indexer declaration (shared by class members and an
    /// interface's own indexers).
    fn indexer_decl(&mut self, idx: &crate::models::IndexerDecl) {
        let access_byte = match idx.access {
            crate::models::Access::Public => 1,
            crate::models::Access::Private => 2,
            crate::models::Access::Protected => 3,
            crate::models::Access::Internal => 4,
            _ => 0,
        };
        self.byte(access_byte);
        self.byte(if idx.is_settable { 1 } else { 0 });
        self.attrs(&idx.attrs);
        self.string(&idx.type_name);
        self.u32(idx.params.len());
        for param in &idx.params {
            self.param(param);
        }
        self.expr(&idx.init);
        match &idx.get {
            Some(block) => {
                self.byte(format::tag::PRESENT);
                self.block(block);
            }
            None => self.byte(format::tag::ABSENT),
        }
        match &idx.set {
            Some(block) => {
                self.byte(format::tag::PRESENT);
                self.block(block);
            }
            None => self.byte(format::tag::ABSENT),
        }
        match &idx.get_expr {
            Some(expr) => {
                self.byte(format::tag::PRESENT);
                self.expr(expr);
            }
            None => self.byte(format::tag::ABSENT),
        }
    }

    fn binop(&mut self, op: BinOp) {
        self.byte(match op {
            BinOp::Add => format::tag::binop::ADD,
            BinOp::Sub => format::tag::binop::SUB,
            BinOp::Mul => format::tag::binop::MUL,
            BinOp::Div => format::tag::binop::DIV,
            BinOp::Eq => format::tag::binop::EQ,
            BinOp::Ne => format::tag::binop::NE,
            BinOp::Lt => format::tag::binop::LT,
            BinOp::Gt => format::tag::binop::GT,
            BinOp::Le => format::tag::binop::LE,
            BinOp::Ge => format::tag::binop::GE,
            BinOp::And => format::tag::binop::AND,
            BinOp::Or => format::tag::binop::OR,
            BinOp::Rem => format::tag::binop::REM,
            BinOp::BitAnd => format::tag::binop::BIT_AND,
            BinOp::BitOr => format::tag::binop::BIT_OR,
            BinOp::BitXor => format::tag::binop::BIT_XOR,
            BinOp::Shl => format::tag::binop::SHL,
            BinOp::Shr => format::tag::binop::SHR,
            BinOp::ShrShr => format::tag::binop::SHR_SHR,
            BinOp::Is => format::tag::binop::IS,
            BinOp::NullCoalesce => format::tag::binop::NULL_COALESCE,
        });
    }

    fn expr(&mut self, expr: &Expr) {
        match expr {
            // The parser's named-argument marker: the value is what the
            // encoder sees once the argument list is positional.
            Expr::NamedArg { value, .. } => self.expr(value),
            // A lambda cannot be represented in this model. Writing a
            // tag the decoder REJECTS is the honest option: the
            // alternative is picking some shape that decodes into a
            // different expression, which would make a round-trip silently
            // change the program.
            Expr::Lambda { .. } => {
                self.byte(format::tag::LAMBDA);
            }
            // Overflow control is a real part of the expression: dropping
            // the wrapper would make a round-trip silently turn checked
            // arithmetic into wrapping arithmetic.
            Expr::Checked { inner, .. } => {
                self.byte(format::tag::CHECKED);
                self.expr(inner);
            }
            Expr::Binary { op, lhs, rhs, .. } => {
                self.byte(format::tag::BIN);
                self.binop(*op);
                self.expr(lhs);
                self.expr(rhs);
            }
            Expr::Unary { op, inner, .. } => {
                self.byte(format::tag::UNARY);
                self.byte(match op {
                    UnOp::Neg => format::tag::unop::NEG,
                    UnOp::Not => format::tag::unop::NOT,
                    UnOp::BitNot => format::tag::unop::BIT_NOT,
                    UnOp::Deref => format::tag::unop::DEREF,
                    UnOp::AddrOf => format::tag::unop::ADDR_OF,
                    UnOp::Throw => format::tag::unop::THROW,
                    UnOp::IndexFromEnd => format::tag::unop::INDEX_FROM_END,
                });
                self.expr(inner);
            }
            Expr::Call { callee, args, .. } => {
                self.byte(format::tag::CALL);
                self.string(callee);
                self.u32(args.len());
                for arg in args {
                    self.expr(arg);
                }
            }
            Expr::Literal(Literal::Str(value)) => {
                self.byte(format::tag::STR);
                self.string(value);
            }
            Expr::Literal(Literal::Int(value)) => {
                self.byte(format::tag::INT);
                self.u64(*value as u64);
            }
            Expr::Literal(Literal::Char(value)) => {
                self.byte(format::tag::CHAR);
                self.u32(*value as usize);
            }
            Expr::Literal(Literal::Float(value)) => {
                self.byte(format::tag::FLOAT);
                self.u64(value.to_bits());
            }
            Expr::Literal(Literal::Bool(value)) => {
                self.byte(format::tag::BOOL);
                self.byte(*value as u8);
            }
            Expr::Literal(Literal::Null) => {
                self.byte(format::tag::NULL);
            }
            Expr::Ternary {
                cond,
                then_expr,
                else_expr,
                ..
            } => {
                self.byte(format::tag::TERNARY);
                self.expr(cond);
                self.expr(then_expr);
                self.expr(else_expr);
            }
            Expr::New {
                class_name, args, init, ..
            } => {
                self.byte(format::tag::NEW);
                self.string(class_name);
                self.u32(args.len());
                for arg in args {
                    self.expr(arg);
                }
                self.u32(init.len());
                for m in init {
                    self.string(&m.name);
                    self.expr(&m.value);
                }
            }
            Expr::Member { object, name, .. } => {
                self.byte(format::tag::MEMBER);
                self.expr(object);
                self.string(name);
            }
            Expr::MethodCall {
                object, name, args, ..
            } => {
                self.byte(format::tag::METHOD_CALL);
                self.expr(object);
                self.string(name);
                self.u32(args.len());
                for arg in args {
                    self.expr(arg);
                }
            }
            Expr::NewArray { elem_type, size, init, .. } => {
                self.byte(format::tag::NEW_ARRAY);
                self.string(elem_type);
                self.expr(size);
                self.u32(init.len());
                for e in init {
                    self.expr(e);
                }
            }
            Expr::Index { object, index, is_range, .. } => {
                self.byte(format::tag::INDEX);
                self.expr(object);
                self.expr(index);
                self.byte(u8::from(*is_range));
            }
            Expr::Ident { name, .. } => {
                self.byte(format::tag::IDENT);
                self.string(name);
            }
            Expr::Cast { type_name, expr, .. } => {
                self.byte(format::tag::CAST);
                self.string(type_name);
                self.expr(expr);
            }
            Expr::Default { type_name, .. } => {
                self.byte(format::tag::DEFAULT);
                self.string(type_name);
            }
            Expr::Await { inner, .. } => {
                self.byte(format::tag::AWAIT);
                self.expr(inner);
            }
            Expr::AssignComplex { lhs, value, .. } => {
                self.byte(format::tag::ASSIGN_COMPLEX);
                self.expr(lhs);
                self.expr(value);
            }
            Expr::Assign { target, value, .. } => {
                self.byte(format::tag::EXPR_ASSIGN);
                self.expr(target);
                self.expr(value);
            }
            Expr::PostInc { target, .. } => {
                self.byte(format::tag::POST_INC);
                self.expr(target);
            }
            Expr::PostDec { target, .. } => {
                self.byte(format::tag::POST_DEC);
                self.expr(target);
            }
            Expr::SwitchExpr { subject, arms, .. } => {
                self.byte(format::tag::SWITCH_EXPR);
                self.expr(subject);
                self.u32(arms.len());
                for arm in arms {
                    self.u32(arm.labels.len());
                    for label in &arm.labels {
                        self.expr(label);
                    }
                    match &arm.guard {
                        Some(g) => {
                            self.byte(format::tag::PRESENT);
                            self.expr(g);
                        }
                        None => self.byte(format::tag::ABSENT),
                    }
                    self.expr(&arm.body);
                }
            }
            Expr::InterpolatedString { parts, .. } => {
                self.byte(format::tag::INTERP_STRING);
                self.u32(parts.len());
                for part in parts {
                    match part {
                        crate::models::InterpPart::Text(t) => {
                            self.byte(0);
                            self.string(t);
                        }
                        crate::models::InterpPart::Hole(h) => {
                            self.byte(1);
                            self.expr(h);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::decode;
    use crate::models::compare::StructuralEq;
    use crate::parser::rd;

    fn roundtrip(src: &str) -> bool {
        let unit = match rd::parse(src) {
            Ok(u) => u,
            Err(_) => return false,
        };
        let bytes = super::encode(&unit);
        match decode(&bytes) {
            Ok(d) => unit.structural_eq(&d),
            Err(_) => false,
        }
    }

    #[test]
    fn interp_string_constant_folds() {
        // $"" is an empty interpolation — should roundtrip as InterpolatedString.
        assert!(roundtrip("class P { public static void M() { string s = $\"\"; } }"));
    }

    #[test]
    fn interp_string_with_hole() {
        assert!(roundtrip("class P { public static void M() { string s = $\"A{7}C\"; } }"));
    }

    #[test]
    fn interp_string_nested() {
        assert!(roundtrip("class P { public static void M() { string s = $\"x{\"$n{3}\"}\"; } }"));
    }

    #[test]
    fn interp_string_escaped_braces() {
        assert!(roundtrip("class P { public static void M() { string s = $\"{{lit}}{5}\"; } }"));
    }

    #[test]
    fn interp_string_no_holes() {
        assert!(roundtrip("class P { public static void M() { string s = $\"no holes\"; } }"));
    }
}
