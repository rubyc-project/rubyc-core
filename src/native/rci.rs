//! `.rci` sidecars: per-library semantic fragments for dynamic linking.
//!
//! A build that compiles `lib`'s sources against its dependencies'
//! sources carries every dep's full table. The `.rci` records the
//! lib's OWN post-resolution entries so a later build can load dep
//! *signatures* instead of re-parsing (and re-lowering) dep *sources*
//! — the semantic half of dynamic linking (the code half is
//! cross-image calls resolved from installed `.so` files).
//!
//! Only main-owned entries are stored (classes by `owner`, interfaces
//! and delegates by owners-map match — see `encode`). Dep aliases are
//! deliberately ABSENT: C# file-scoped `using X = ...` aliases are
//! invisible outside their file, so main code can never need a dep's
//! aliases (v1 stored all aliases unit-wide; v2 drops them).
//!
//! Format (all integers little-endian; strings are u32 length + bytes):
//! ```text
//! magic "RCI1", format version u32 (=3)
//! lib name, source-hash u64 (freshness vs the dep's current sources)
//! classes: u32 count, each in TABLE order (declaration order —
//!   consumers recompute nothing order-sensitive from it):
//!   name, key.path, key.arity u32, owner, bases[u32 n, strings]
//!   fields/static_fields/const_fields: u32 n, each:
//!     name, type_name, access byte, offset_slots u64,
//!     is_static/is_const/is_byref bytes, const_value (tag byte +
//!     payload: 0 none, 1 Int i64, 2 Float u64 bits, 3 Str, 4 Null),
//!     declaring_class
//!   methods: u32 n, each:
//!     name, ret_type, param_types[], param_has_default bytes[],
//!     has_params byte, param_is_byref bytes[], param_defaults[]
//!     (const_value each), access byte,
//!     is_static/has_body/is_extension bytes,
//!     declaring_class, generic_params[], conditional_symbol
//!     (present byte + string), vtable_slot u32
//!   own_members: u32 n, each (name, is_field byte, access byte)
//!   interfaces: u32 n, each (name, arity u32)
//!   properties: u32 n, each
//!   (name, type_name, access byte, declaring_class, is_static byte,
//!   is_abstract byte)
//!   is_enum/is_struct bytes, generic_params[]
//! interfaces: u32 n, each:
//!   name, generic_params[], methods[] (access, ret, name, generics,
//!   params[] (type, name, is_this, is_params, is_byref), is_static),
//!   bases[], arity u32
//! delegates: u32 n, each:
//!   access byte, ret_type, name, generic_params[],
//!   params[] (type, name, is_this, is_params, is_byref)
//! ```
//!
//! Decode lives here too (used by the consume path); both must stay in
//! lockstep — bump `VERSION` on any change.

use crate::models::Access;
use crate::models::{DelegateDecl, InterfaceDecl, MethodSig, Param};
use crate::native::semantic::{
    ClassInfo, ConstValue, FieldInfo, MethodInfo, PropertyInfo,
};

pub const VERSION: u32 = 6;
const MAGIC: &[u8; 4] = b"RCI1";

fn access_byte(a: Access) -> u8 {
    match a {
        Access::Public => 1,
        Access::Private => 2,
        Access::Protected => 3,
        Access::Internal => 4,
        _ => 0,
    }
}

fn access_of(b: u8) -> Access {
    match b {
        1 => Access::Public,
        2 => Access::Private,
        3 => Access::Protected,
        4 => Access::Internal,
        _ => Access::Default,
    }
}

struct Writer {
    out: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Writer { out: Vec::new() }
    }

    fn u8(&mut self, x: u8) {
        self.out.push(x);
    }

    fn u32(&mut self, x: u32) {
        self.out.extend_from_slice(&x.to_le_bytes());
    }

    fn u64(&mut self, x: u64) {
        self.out.extend_from_slice(&x.to_le_bytes());
    }

    fn i64(&mut self, x: i64) {
        self.out.extend_from_slice(&x.to_le_bytes());
    }

    fn string(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.out.extend_from_slice(s.as_bytes());
    }

    fn strings(&mut self, xs: &[String]) {
        self.u32(xs.len() as u32);
        for x in xs {
            self.string(x);
        }
    }

    fn access(&mut self, a: Access) {
        self.u8(access_byte(a));
    }

    fn const_value(&mut self, v: &Option<ConstValue>) {
        match v {
            None => self.u8(0),
            Some(ConstValue::Int(i)) => {
                self.u8(1);
                self.i64(*i);
            }
            Some(ConstValue::Float(f)) => {
                self.u8(2);
                self.u64(f.to_bits());
            }
            Some(ConstValue::Str(s)) => {
                self.u8(3);
                self.string(s);
            }
            Some(ConstValue::Null) => self.u8(4),
        }
    }

    fn field(&mut self, f: &FieldInfo) {
        self.string(&f.name);
        self.string(&f.type_name);
        self.access(f.access);
        self.u64(f.offset_slots as u64);
        self.u8(u8::from(f.is_static));
        self.u8(u8::from(f.is_const));
        self.u8(u8::from(f.is_byref));
        self.const_value(&f.const_value);
        self.string(&f.declaring_class);
    }

    fn method(&mut self, m: &MethodInfo) {
        self.string(&m.name);
        self.string(&m.ret_type);
        self.strings(&m.param_types);
        // Parameter names travel with the fragment: a call from the main
        // project with a named argument is rewritten here, so the callee's
        // names are part of its ABI description.
        self.strings(&m.param_names);
        self.u32(m.param_has_default.len() as u32);
        for b in &m.param_has_default {
            self.u8(u8::from(*b));
        }
        self.u8(u8::from(m.has_params));
        self.u32(m.param_is_byref.len() as u32);
        for b in &m.param_is_byref {
            self.u8(u8::from(*b));
        }
        self.u32(m.param_defaults.len() as u32);
        for d in &m.param_defaults {
            self.const_value(d);
        }
        self.access(m.access);
        self.u8(u8::from(m.is_static));
        self.u8(u8::from(m.has_body));
        self.u8(u8::from(m.is_extension));
        self.string(&m.declaring_class);
        self.strings(&m.generic_params);
        match &m.conditional_symbol {
            Some(s) => {
                self.u8(1);
                self.string(s);
            }
            None => self.u8(0),
        }
        self.u32(m.vtable_slot as u32);
    }

    fn property(&mut self, p: &PropertyInfo) {
        self.string(&p.name);
        self.string(&p.type_name);
        self.access(p.access);
        self.string(&p.declaring_class);
        self.u8(u8::from(p.is_static));
        self.u8(u8::from(p.is_abstract));
        self.u8(u8::from(p.is_intrinsic));
    }

    fn param(&mut self, p: &Param) {
        self.string(&p.type_name);
        self.string(&p.name);
        self.u8(u8::from(p.is_this));
        self.u8(u8::from(p.is_params));
        self.u8(u8::from(p.is_byref));
    }

    fn method_sig(&mut self, m: &MethodSig) {
        self.access(m.access);
        self.string(&m.ret_type);
        self.string(&m.name);
        self.strings(&m.generic_params);
        self.u32(m.params.len() as u32);
        for p in &m.params {
            self.param(p);
        }
        self.u8(u8::from(m.is_static));
    }

    fn interface(&mut self, i: &InterfaceDecl) {
        self.string(&i.name);
        self.strings(&i.generic_params);
        self.u32(i.methods.len() as u32);
        for m in &i.methods {
            self.method_sig(m);
        }
        self.strings(&i.bases);
        self.u32(i.arity as u32);
    }

    fn delegate(&mut self, d: &DelegateDecl) {
        self.access(d.access);
        self.string(&d.ret_type);
        self.string(&d.name);
        self.strings(&d.generic_params);
        self.u32(d.params.len() as u32);
        for p in &d.params {
            self.param(p);
        }
    }

    fn class(&mut self, c: &ClassInfo) {
        self.string(&c.name);
        self.string(&c.key.path);
        self.u32(c.key.arity as u32);
        self.string(&c.owner);
        self.strings(&c.bases);
        self.u32(c.fields.len() as u32);
        for f in &c.fields {
            self.field(f);
        }
        self.u32(c.static_fields.len() as u32);
        for f in &c.static_fields {
            self.field(f);
        }
        self.u32(c.const_fields.len() as u32);
        for f in &c.const_fields {
            self.field(f);
        }
        self.u32(c.methods.len() as u32);
        for m in &c.methods {
            self.method(m);
        }
        self.u32(c.own_members.len() as u32);
        for (n, is_f, a) in &c.own_members {
            self.string(n);
            self.u8(u8::from(*is_f));
            self.access(*a);
        }
        self.u32(c.interfaces.len() as u32);
        for (n, ar) in &c.interfaces {
            self.string(n);
            self.u32(*ar as u32);
        }
        self.u32(c.properties.len() as u32);
        for p in &c.properties {
            self.property(p);
        }
        self.u8(u8::from(c.is_enum));
        self.strings(&c.generic_params);
        self.u8(u8::from(c.is_struct));
    }
}

/// Encode one library's own entries in TABLE order (declaration order —
/// consumers recompute nothing order-sensitive from it), plus its
/// interfaces and delegates. `source_hash` is the content hash of the
/// lib's own sources for freshness checks on consume.
pub fn encode(
    lib_name: &str,
    source_hash: u64,
    classes: &[ClassInfo],
    interfaces: &[InterfaceDecl],
    delegates: &[DelegateDecl],
) -> Vec<u8> {
    let mut w = Writer::new();
    w.out.extend_from_slice(MAGIC);
    w.u32(VERSION);
    w.string(lib_name);
    w.u64(source_hash);
    let own: Vec<&ClassInfo> = classes.iter().filter(|c| c.owner == lib_name).collect();
    w.u32(own.len() as u32);
    for c in own {
        w.class(c);
    }
    w.u32(interfaces.len() as u32);
    for i in interfaces {
        w.interface(i);
    }
    w.u32(delegates.len() as u32);
    for d in delegates {
        w.delegate(d);
    }
    w.out
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    err: Option<String>,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Reader {
            data,
            pos: 0,
            err: None,
        }
    }

    fn bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.err.is_some() || self.pos + n > self.data.len() {
            self.err = Some("truncated .rci".to_owned());
            return None;
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Some(s)
    }

    fn u8(&mut self) -> u8 {
        self.bytes(1).map(|b| b[0]).unwrap_or(0)
    }

    fn u32(&mut self) -> u32 {
        self.bytes(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .unwrap_or(0)
    }

    fn u64(&mut self) -> u64 {
        self.bytes(8)
            .map(|b| {
                u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
            })
            .unwrap_or(0)
    }

    fn i64(&mut self) -> i64 {
        self.u64() as i64
    }

    fn string(&mut self) -> String {
        let n = self.u32() as usize;
        match self.bytes(n) {
            Some(b) => String::from_utf8_lossy(b).into_owned(),
            None => String::new(),
        }
    }

    fn strings(&mut self) -> Vec<String> {
        let n = self.u32() as usize;
        (0..n).map(|_| self.string()).collect()
    }

    fn access(&mut self) -> Access {
        access_of(self.u8())
    }

    fn const_value(&mut self) -> Option<ConstValue> {
        match self.u8() {
            1 => Some(ConstValue::Int(self.i64())),
            2 => Some(ConstValue::Float(f64::from_bits(self.u64()))),
            3 => Some(ConstValue::Str(self.string())),
            4 => Some(ConstValue::Null),
            _ => None,
        }
    }

    fn field(&mut self) -> FieldInfo {
        let name = self.string();
        let type_name = self.string();
        let access = self.access();
        let offset_slots = self.u64() as usize;
        let is_static = self.u8() != 0;
        let is_const = self.u8() != 0;
        let is_byref = self.u8() != 0;
        let const_value = self.const_value();
        let declaring_class = self.string();
        FieldInfo {
            name,
            type_name,
            access,
            offset_slots,
            is_static,
            is_const,
            is_byref,
            const_value,
            declaring_class,
        }
    }

    fn method(&mut self) -> MethodInfo {
        let name = self.string();
        let ret_type = self.string();
        let param_types = self.strings();
        let param_names = self.strings();
        let n = self.u32() as usize;
        let param_has_default = (0..n).map(|_| self.u8() != 0).collect();
        let has_params = self.u8() != 0;
        let n = self.u32() as usize;
        let param_is_byref = (0..n).map(|_| self.u8() != 0).collect();
        let n = self.u32() as usize;
        let param_defaults = (0..n).map(|_| self.const_value()).collect();
        let access = self.access();
        let is_static = self.u8() != 0;
        let has_body = self.u8() != 0;
        let is_extension = self.u8() != 0;
        let declaring_class = self.string();
        let generic_params = self.strings();
        let conditional_symbol = if self.u8() != 0 {
            Some(self.string())
        } else {
            None
        };
        let vtable_slot = self.u32() as usize;
        MethodInfo {
            name,
            ret_type,
            param_types,
            param_names,
            param_has_default,
            has_params,
            param_is_byref,
            param_defaults,
            access,
            is_static,
            has_body,
            declaring_class,
            is_extension,
            generic_params,
            conditional_symbol,
            vtable_slot,
        }
    }

    fn property(&mut self) -> PropertyInfo {
        let name = self.string();
        let type_name = self.string();
        let access = self.access();
        let declaring_class = self.string();
        let is_static = self.u8() != 0;
        let is_abstract = self.u8() != 0;
        let is_intrinsic = self.u8() != 0;
        PropertyInfo {
            name,
            type_name,
            access,
            declaring_class,
            is_static,
            is_abstract,
            is_intrinsic,
        }
    }

    fn param(&mut self) -> Param {
        let type_name = self.string();
        let name = self.string();
        let is_this = self.u8() != 0;
        let is_params = self.u8() != 0;
        let is_byref = self.u8() != 0;
        Param {
            type_name,
            name,
            span: crate::models::Span::default(),
            default_value: None,
            is_this,
            is_params,
            is_byref,
            attrs: Vec::new(),
        }
    }

    fn method_sig(&mut self) -> MethodSig {
        let access = self.access();
        let ret_type = self.string();
        let name = self.string();
        let generic_params = self.strings();
        let n = self.u32() as usize;
        let params = (0..n).map(|_| self.param()).collect();
        let is_static = self.u8() != 0;
        MethodSig {
            access,
            ret_type,
            name,
            generic_params,
            params,
            body: None,
            is_static,
            span: crate::models::Span::default(),
        }
    }

    fn interface(&mut self) -> InterfaceDecl {
        let name = self.string();
        let generic_params = self.strings();
        let n = self.u32() as usize;
        let methods = (0..n).map(|_| self.method_sig()).collect();
        let bases = self.strings();
        let arity = self.u32() as usize;
        InterfaceDecl {
            name,
            indexers: Vec::new(),
            generic_params,
            methods,
            bases,
            arity,
            is_partial: false,
            span: crate::models::Span::default(),
        }
    }

    fn delegate(&mut self) -> DelegateDecl {
        let access = self.access();
        let ret_type = self.string();
        let name = self.string();
        let generic_params = self.strings();
        let n = self.u32() as usize;
        let params = (0..n).map(|_| self.param()).collect();
        DelegateDecl {
            access,
            ret_type,
            name,
            generic_params,
            params,
            span: crate::models::Span::default(),
        }
    }

    fn class(&mut self) -> ClassInfo {
        use crate::native::semantic::TypeKey;
        let name = self.string();
        let path = self.string();
        let arity = self.u32() as usize;
        let owner = self.string();
        let bases = self.strings();
        let n = self.u32() as usize;
        let fields = (0..n).map(|_| self.field()).collect();
        let n = self.u32() as usize;
        let static_fields = (0..n).map(|_| self.field()).collect();
        let n = self.u32() as usize;
        let const_fields = (0..n).map(|_| self.field()).collect();
        let n = self.u32() as usize;
        let methods = (0..n).map(|_| self.method()).collect();
        let n = self.u32() as usize;
        let own_members = (0..n)
            .map(|_| (self.string(), self.u8() != 0, self.access()))
            .collect();
        let n = self.u32() as usize;
        let interfaces = (0..n).map(|_| (self.string(), self.u32() as usize)).collect();
        let n = self.u32() as usize;
        let properties = (0..n).map(|_| self.property()).collect();
        let is_enum = self.u8() != 0;
        let generic_params = self.strings();
        let is_struct = self.u8() != 0;
        ClassInfo {
            type_arg_index: Vec::new(),
            name,
            key: TypeKey::new(path, arity),
            bases,
            fields,
            static_fields,
            const_fields,
            methods,
            own_members,
            interfaces,
            properties,
            is_enum,
            generic_params,
            is_struct,
            // The fragment format does not carry record-ness yet; a
            // record arriving from a library is treated as a plain class
            // (value equality is lost across the boundary, tracked with
            // the fragment format rather than guessed here).
            is_record: false,
            owner,
        }
    }
}

/// Decoded fragment: owning lib, source hash, entries, interfaces,
/// delegates.
pub struct Fragment {
    pub lib_name: String,
    pub source_hash: u64,
    pub classes: Vec<ClassInfo>,
    pub interfaces: Vec<InterfaceDecl>,
    pub delegates: Vec<DelegateDecl>,
}

/// Decode one `.rci` blob. Errors on bad magic, version mismatch, or
/// truncation (all reported as strings; the consume path falls back
/// to sources on any error).
pub fn decode(data: &[u8]) -> Result<Fragment, String> {
    let mut r = Reader::new(data);
    let magic = r.bytes(4).unwrap_or_default();
    if magic != MAGIC {
        return Err("bad .rci magic".to_owned());
    }
    let version = r.u32();
    if version != VERSION {
        return Err(format!("unsupported .rci version {version}"));
    }
    let lib_name = r.string();
    let source_hash = r.u64();
    let n = r.u32() as usize;
    let classes = (0..n).map(|_| r.class()).collect();
    let n = r.u32() as usize;
    let interfaces = (0..n).map(|_| r.interface()).collect();
    let n = r.u32() as usize;
    let delegates = (0..n).map(|_| r.delegate()).collect();
    match r.err {
        Some(e) => Err(e),
        None => Ok(Fragment {
            lib_name,
            source_hash,
            classes,
            interfaces,
            delegates,
        }),
    }
}
