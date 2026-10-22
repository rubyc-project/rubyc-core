//! Native type descriptors.
//!
//! C# exposes a type as a first-class value (`typeof(X)`), and generic
//! code must be able to ask questions about a type argument that is only
//! known when the method runs. Each referenced type therefore gets one
//! descriptor in the generated program:
//!
//! ```text
//! +0   TYPE_TAG
//! +8   refcount (immortal: the descriptor is canonical)
//! +16  name string
//! +24  full-name string
//! +32  namespace string
//! +40  flags
//! +48  reserved (runtime type id)
//! +56  element descriptor (an array) or the type arguments (a
//!      generic instantiation)
//! +64  the argument count (a generic instantiation) or the array
//!      cache head (an array)
//! ```
//!
//! The descriptor is built lazily on first use and stored in a static
//! slot, so `typeof(X) == typeof(X)` compares one pointer. There is no
//! type registry service at run time: the slot and the descriptor are
//! ordinary generated data. Generic methods receive these pointers as
//! hidden arguments, which is how `typeof(T)` learns its concrete type.

use std::collections::BTreeMap;

use super::select::{canonical_primitive, explicit_type_args};
use super::types::subst_type_var;
use crate::native::ir::{Ctx, StringArena};

/// First word of a descriptor. The `0xFFFF` prefix cannot be a vtable
/// row, a string length, or a heap pointer, so class and string tests
/// reject descriptors without a special case.
pub(crate) const TYPE_TAG: i64 = 0xFFFF_0000_0000_0001u64 as i64;
/// Boxed value refcount tag (bit 62). A box's first word may be a
/// wrapper vtable row, so the refcount is what lets a runtime type check
/// tell a box from an ordinary object. Strings use bit 63; descriptors
/// and classes have neither.
pub(crate) const BOX_FLAG: i64 = 0x4000_0000_0000_0000u64 as i64;
pub(crate) const BOX_REF: i64 = BOX_FLAG | 1;
pub(crate) const BOX_BYTES: u64 = 32;

/// Descriptor refcount. Release decrements it but never reaches zero,
/// because the canonical descriptor lives in a static slot forever. The
/// two high tag bits stay clear so a descriptor is never taken for a box.
pub(crate) const TYPE_REF: i64 = 0x3FFF_FFFF_FFFF_FFFF;
pub(crate) const TYPE_BYTES: u64 = 80;

/// Refcount of a string that belongs to the program's permanent data
/// (type names). It keeps the string tag bit and a count high enough
/// that no program can release it to zero.
pub(crate) const STRING_IMMORTAL_REF: i64 = i64::MIN | 0x0000_0100_0000_0000;

/// First word of a box whose primitive has no declared wrapper class
/// (standalone programs). CoreLib builds use the real Int32/Boolean/...
/// vtable row instead; the refcount box tag plus the descriptor at +24
/// make the two forms interchangeable to type tests.
pub(crate) const BOX_TAG_BASE: i64 = 0xFFFF_0000_0000_0100u64 as i64;

/// Stable id per boxable primitive (keyword and framework spellings).
pub(crate) fn primitive_id(simple: &str) -> Option<i64> {
    Some(match canonical_primitive(simple).as_str() {
        "Boolean" => 1,
        "Char" => 2,
        "SByte" => 3,
        "Byte" => 4,
        "Int16" => 5,
        "UInt16" => 6,
        "Int32" => 7,
        "UInt32" => 8,
        "Int64" => 9,
        "UInt64" => 10,
        "IntPtr" => 11,
        "UIntPtr" => 12,
        "Single" => 13,
        "Double" => 14,
        "Decimal" => 15,
        _ => return None,
    })
}

pub(crate) fn box_sentinel(simple: &str) -> Option<i64> {
    primitive_id(simple).map(|id| BOX_TAG_BASE + id)
}

pub(crate) const F_VALUE_TYPE: u64 = 1 << 0;
pub(crate) const F_PRIMITIVE: u64 = 1 << 1;
pub(crate) const F_ENUM: u64 = 1 << 2;
pub(crate) const F_ARRAY: u64 = 1 << 3;
pub(crate) const F_SZARRAY: u64 = 1 << 4;
pub(crate) const F_POINTER: u64 = 1 << 5;
pub(crate) const F_BYREFLIKE: u64 = 1 << 6;
pub(crate) const F_GENERIC_TYPE: u64 = 1 << 7;
pub(crate) const F_CLASS: u64 = 1 << 8;
pub(crate) const F_INTERFACE: u64 = 1 << 9;
pub(crate) const F_HAS_ELEMENT: u64 = 1 << 10;

pub(crate) const OFF_NAME: u32 = 16;
/// The type arguments of a generic instantiation: a pointer to an
/// array of argument descriptors. An array descriptor uses this slot
/// for its element type, a generic one for its arguments — the two
/// descriptor kinds never overlap.
pub(crate) const OFF_TYPE_ARGS: u32 = 56;
/// How many type arguments the array at `OFF_TYPE_ARGS` holds.
pub(crate) const OFF_TYPE_ARG_COUNT: u32 = 64;
pub(crate) const OFF_FULL_NAME: u32 = 24;
pub(crate) const OFF_NAMESPACE: u32 = 32;
pub(crate) const OFF_FLAGS: u32 = 40;

/// One interned name plus the type's identity key.
#[derive(Clone)]
pub(crate) struct TypeRequest {
    pub name: (u32, u32),
    pub full: (u32, u32),
    pub namespace: (u32, u32),
    pub flags: u64,
    /// The type arguments of a generic instantiation, as spellings.
    /// Empty for a non-generic type.
    pub type_args: Vec<String>,
}

/// Type spellings interned by this compilation. `base` is the first
/// static slot reserved for descriptors; slot `base + i` caches the
/// descriptor for `slots[i]`. Slots are appended to `Program::statics`
/// after lowering, so every `LoadStaticAddr` here is valid.
pub(crate) struct TypeRegistry {
    pub base: u32,
    slots: Vec<TypeRequest>,
    index: BTreeMap<String, u32>,
}

impl TypeRegistry {
    pub fn new(base: u32) -> Self {
        Self {
            base,
            slots: Vec::new(),
            index: BTreeMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        // Two reserved slots cache the list heads: the composed arrays
        // and the composed generic instantiations.
        self.slots.len() + 2
    }

    pub fn cache_slot(&self) -> u32 {
        self.base
    }

    /// The static slot holding the head of the list of composed
    /// generic instantiations (the array list has its own).
    pub fn generic_cache_slot(&self) -> u32 {
        self.base + 1
    }

    /// Intern a type and return its absolute static slot plus the
    /// interned request. Same canonical key -> same slot.
    pub fn slot_for(
        &mut self,
        key: &str,
        meta: &TypeMeta,
        strings: &mut StringArena,
    ) -> (u32, TypeRequest) {
        if let Some(&i) = self.index.get(key) {
            return (self.base + 1 + i, self.slots[i as usize].clone());
        }
        let req = TypeRequest {
            name: strings.utf16_literal(&meta.name),
            full: strings.utf16_literal(&meta.full),
            namespace: strings.utf16_literal(&meta.namespace),
            flags: meta.flags,
            type_args: meta.type_args.clone(),
        };
        let i = self.slots.len() as u32;
        self.slots.push(req.clone());
        self.index.insert(key.to_owned(), i);
        (self.base + 1 + i, req)
    }
}

/// Canonical identity, display names and flags for one spelling.
pub(crate) struct TypeMeta {
    pub key: String,
    pub name: String,
    pub full: String,
    pub namespace: String,
    pub flags: u64,
    pub has_type_var: bool,
    /// The type arguments of a generic instantiation, as spellings.
    /// Empty for a non-generic type.
    pub type_args: Vec<String>,
}

struct Primitive {
    full: &'static str,
    simple: &'static str,
    value_type: bool,
    primitive: bool,
}

/// Primitive and pseudo-primitive spellings, keyword and framework
/// forms alike. `String` and `Object` are classes, not primitives;
/// `Decimal` is a value type but not a primitive.
fn primitive(base: &str) -> Option<Primitive> {
    let p = |full, simple, value_type, primitive| Primitive {
        full,
        simple,
        value_type,
        primitive,
    };
    Some(match base {
        "bool" | "Boolean" => p("System.Boolean", "Boolean", true, true),
        "char" | "Char" => p("System.Char", "Char", true, true),
        "sbyte" | "SByte" => p("System.SByte", "SByte", true, true),
        "byte" | "Byte" => p("System.Byte", "Byte", true, true),
        "short" | "Int16" => p("System.Int16", "Int16", true, true),
        "ushort" | "UInt16" => p("System.UInt16", "UInt16", true, true),
        "int" | "Int32" => p("System.Int32", "Int32", true, true),
        "uint" | "UInt32" => p("System.UInt32", "UInt32", true, true),
        "long" | "Int64" => p("System.Int64", "Int64", true, true),
        "ulong" | "UInt64" => p("System.UInt64", "UInt64", true, true),
        "nint" | "IntPtr" => p("System.IntPtr", "IntPtr", true, true),
        "nuint" | "UIntPtr" => p("System.UIntPtr", "UIntPtr", true, true),
        "float" | "Single" => p("System.Single", "Single", true, true),
        "double" | "Double" => p("System.Double", "Double", true, true),
        "decimal" | "Decimal" => p("System.Decimal", "Decimal", true, false),
        "void" | "Void" => p("System.Void", "Void", true, true),
        "string" | "String" => p("System.String", "String", false, false),
        "object" | "Object" => p("System.Object", "Object", false, false),
        _ => return None,
    })
}

fn split_generic(s: &str) -> Option<(&str, Vec<String>)> {
    let open = s.find('<')?;
    if !s.ends_with('>') {
        return None;
    }
    let base = s[..open].trim();
    let args = explicit_type_args(s);
    if args.is_empty() {
        return None;
    }
    Some((base, args))
}

fn namespace_of(full: &str) -> String {
    // Ignore dots inside generic arguments: `Ns.List<Other.X>` has the
    // namespace `Ns`.
    let mut depth = 0i32;
    let mut last = None;
    for (i, c) in full.char_indices() {
        match c {
            '<' | '[' | '(' => depth += 1,
            '>' | ']' | ')' => depth -= 1,
            '.' if depth == 0 => last = Some(i),
            _ => {}
        }
    }
    last.map(|i| full[..i].to_owned()).unwrap_or_default()
}

fn simple_of(full: &str) -> String {
    let ns = namespace_of(full);
    let rest = if ns.is_empty() {
        full
    } else {
        &full[ns.len() + 1..]
    };
    rest.split('<').next().unwrap_or(rest).to_owned()
}

/// True when `spelling` is exactly a type parameter (not `T[]`, `T?`,
/// a constructed type, or a qualified name). Only a bare parameter has
/// a hidden handle of its own.
pub(crate) fn direct_type_var(spelling: &str, ctx: &Ctx<'_>) -> bool {
    let s = spelling
        .trim()
        .trim_start_matches("global::")
        .trim();
    !s.is_empty()
        && !s.contains(['[', ']', '*', '?', '<'])
        && ctx.is_type_var(s)
}

fn class_meta(full: String, info: &crate::native::semantic::ClassInfo, ctx: &Ctx<'_>) -> TypeMeta {
    let simple = simple_of(&full);
    let mut flags = 0u64;
    if info.is_struct {
        flags |= F_VALUE_TYPE;
    }
    if info.is_enum {
        flags |= F_VALUE_TYPE | F_ENUM;
    }
    let interface = ctx.table.interfaces.iter().any(|i| {
        i.name == full
            || i.name == info.name
            || i.name.rsplit('.').next() == Some(simple.as_str())
    }) || ctx.table.classes.iter().any(|c| {
        c.interfaces.iter().any(|(name, _)| {
            name == &full || name == &info.name
        })
    });
    if interface {
        flags |= F_INTERFACE;
    } else if !info.is_struct && !info.is_enum {
        flags |= F_CLASS;
    }
    TypeMeta {
        key: full.clone(),
        name: simple,
        full: full.clone(),
        namespace: namespace_of(&full),
        flags,
        has_type_var: false,
        type_args: Vec::new(),
    }
}

fn base_meta(spelling: &str, ctx: &Ctx<'_>) -> TypeMeta {
    let s = spelling.trim().trim_start_matches("global::").trim();
    let base = canonical_primitive(s.rsplit('.').next().unwrap_or(s));

    // A declared entry wins: `N.Int32` and a local `Int32` are not
    // System.Int32. Primitive names are only a fallback when no class
    // carries the spelling.
    if let Some(info) = ctx.table.class(s).or_else(|| ctx.table.class(&base)) {
        let full = info.key.path.clone();
        let simple = simple_of(&full);
        // CoreLib declares the primitive types. Mark them from the
        // canonical full name, not from the declaration shape.
        if let Some(p) = primitive(&simple) {
            let mut meta = class_meta(full, info, ctx);
            meta.name = p.simple.to_owned();
            meta.full = p.full.to_owned();
            meta.namespace = namespace_of(p.full);
            if p.value_type {
                meta.flags |= F_VALUE_TYPE;
            }
            if p.primitive {
                meta.flags |= F_PRIMITIVE;
            } else {
                meta.flags &= !F_PRIMITIVE;
            }
            if p.full == "System.String" || p.full == "System.Object" {
                meta.flags = (meta.flags & !F_VALUE_TYPE) | F_CLASS;
            }
            return meta;
        }
        return class_meta(full, info, ctx);
    }

    if let Some(p) = primitive(&base) {
        let mut flags = 0u64;
        if p.value_type {
            flags |= F_VALUE_TYPE;
        }
        if p.primitive {
            flags |= F_PRIMITIVE;
        }
        if !p.value_type {
            flags |= F_CLASS;
        }
        return TypeMeta {
            key: p.full.to_owned(),
            name: p.simple.to_owned(),
            full: p.full.to_owned(),
            namespace: namespace_of(p.full),
            flags,
            has_type_var: false,
            type_args: Vec::new(),
        };
    }

    // An undeclared type is still a usable identity (the library may
    // supply it later); metadata is limited to what the spelling says.
    TypeMeta {
        key: s.to_owned(),
        name: simple_of(s),
        full: s.to_owned(),
        namespace: namespace_of(s),
        flags: if s.ends_with("[]") { 0 } else { F_CLASS },
        has_type_var: false,
        type_args: Vec::new(),
    }
}

/// The canonical descriptor of a DELEGATE type. A delegate is not a
/// class entry, so it has no table row; the table looks delegates up
/// by name, and that name is the identity the rest of the compiler
/// uses. The arguments are canonical spellings, so `Action<object?>` and
/// `Action<System.Object>` are the same runtime type.
pub(crate) fn delegate_meta(spelling: &str, ctx: &Ctx<'_>) -> TypeMeta {
    let s = spelling.trim();
    let (base, args) = split_generic(s).unwrap_or((s, Vec::new()));
    let simple = base.rsplit('.').next().unwrap_or(base);
    let name = ctx
        .table
        .delegates
        .iter()
        .find(|d| d.name == simple)
        .map(|d| d.name.clone())
        .unwrap_or_else(|| simple.to_owned());
    let metas: Vec<TypeMeta> = args.iter().map(|a| type_meta(a, ctx)).collect();
    let inner: Vec<String> = metas.iter().map(|m| m.full.clone()).collect();
    let key = if inner.is_empty() {
        format!("delegate:{name}")
    } else {
        format!("delegate:{name}<{}>", inner.join(","))
    };
    let (display, full) = if inner.is_empty() {
        (name.clone(), name.clone())
    } else {
        (
            format!("{name}`{}", inner.len()),
            format!("{name}<{}>", inner.join(",")),
        )
    };
    TypeMeta {
        key,
        name: display,
        full,
        // The delegate declarations carry no namespace of their own,
        // so the descriptor's is empty.
        namespace: String::new(),
        flags: F_CLASS,
        has_type_var: false,
        type_args: inner,
    }
}

pub(crate) fn type_meta(spelling: &str, ctx: &Ctx<'_>) -> TypeMeta {
    let mut s = spelling
        .trim()
        .trim_start_matches("global::")
        .trim()
        .to_owned();
    if let Some(sub) = ctx.type_subst
        && let Some(concrete) = subst_type_var(&s, sub)
    {
        s = concrete;
    }
    if s == "dynamic" {
        s = "object".to_owned();
    }

    if let Some(elem) = s.strip_suffix("[]") {
        let e = type_meta(elem, ctx);
        return TypeMeta {
            key: format!("{}[]", e.key),
            name: format!("{}[]", e.name),
            full: format!("{}[]", e.full),
            namespace: e.namespace,
            flags: F_ARRAY | F_SZARRAY | F_HAS_ELEMENT,
            has_type_var: e.has_type_var,
            type_args: Vec::new(),
        };
    }
    if let Some(elem) = s.strip_suffix('*') {
        let e = type_meta(elem, ctx);
        return TypeMeta {
            key: format!("{}*", e.key),
            name: format!("{}*", e.name),
            full: format!("{}*", e.full),
            namespace: e.namespace,
            flags: F_POINTER | F_HAS_ELEMENT,
            has_type_var: e.has_type_var,
            type_args: Vec::new(),
        };
    }
    if let Some(elem) = s.strip_suffix('?') {
        let e = type_meta(elem, ctx);
        // `string?` is an annotation, not a separate runtime type.
        if e.flags & F_VALUE_TYPE == 0 {
            return e;
        }
        let inner = e.key.clone();
        return TypeMeta {
            key: format!("System.Nullable<{inner}>"),
            name: "Nullable`1".to_owned(),
            full: format!("System.Nullable<{}>", e.full),
            namespace: "System".to_owned(),
            flags: F_VALUE_TYPE | F_GENERIC_TYPE,
            has_type_var: e.has_type_var,
            type_args: Vec::new(),
        };
    }

    if let Some((base, args)) = split_generic(&s) {
        let b = base_meta(base, ctx);
        let metas: Vec<TypeMeta> = args.iter().map(|a| type_meta(a, ctx)).collect();
        let inner: Vec<String> = metas.iter().map(|m| m.full.clone()).collect();
        let key = format!("{}<{}>", b.key, metas.iter().map(|m| m.key.clone()).collect::<Vec<_>>().join(","));
        let simple = b.name.split('`').next().unwrap_or(&b.name);
        let mut flags = F_GENERIC_TYPE | (b.flags & (F_VALUE_TYPE | F_CLASS | F_INTERFACE));
        if simple == "Nullable" {
            flags |= F_VALUE_TYPE;
            flags &= !F_CLASS;
        }
        if simple == "Span" || simple == "ReadOnlySpan" {
            flags |= F_BYREFLIKE;
        }
        return TypeMeta {
            key,
            name: format!("{simple}`{}", metas.len()),
            full: format!("{}<{}>", b.full, inner.join(",")),
            namespace: b.namespace,
            flags,
            has_type_var: b.has_type_var || metas.iter().any(|m| m.has_type_var),
            // The arguments are kept as spellings: the descriptor
            // builds their descriptors when it is first created, and
            // `typeof(T)` reads one back out of it.
            type_args: metas.iter().map(|m| m.full.clone()).collect(),
        };
    }

    let mut meta = base_meta(&s, ctx);
    // `T` itself has no compile-time identity; the hidden handle
    // supplies it. The flag tells `type_token` not to intern it.
    meta.has_type_var = direct_type_var(&s, ctx);
    meta
}

/// Field offset and bit for a `System.Type` member that a descriptor
/// answers. `None` means the member is not descriptor-backed (a method,
/// or reflection that still has no implementation).
pub(crate) fn type_member(name: &str) -> Option<(u32, Option<u64>)> {
    Some(match name {
        "Name" => (OFF_NAME, None),
        "FullName" => (OFF_FULL_NAME, None),
        "Namespace" => (OFF_NAMESPACE, None),
        "IsValueType" => (OFF_FLAGS, Some(F_VALUE_TYPE)),
        "IsPrimitive" => (OFF_FLAGS, Some(F_PRIMITIVE)),
        "IsEnum" => (OFF_FLAGS, Some(F_ENUM)),
        "IsArray" => (OFF_FLAGS, Some(F_ARRAY)),
        "IsSZArray" => (OFF_FLAGS, Some(F_SZARRAY)),
        "IsPointer" => (OFF_FLAGS, Some(F_POINTER)),
        "IsByRefLike" => (OFF_FLAGS, Some(F_BYREFLIKE)),
        "IsGenericType" => (OFF_FLAGS, Some(F_GENERIC_TYPE)),
        "IsClass" => (OFF_FLAGS, Some(F_CLASS)),
        "IsInterface" => (OFF_FLAGS, Some(F_INTERFACE)),
        "HasElementType" => (OFF_FLAGS, Some(F_HAS_ELEMENT)),
        // A runtime type handle is the descriptor pointer itself; the
        // lowering returns the receiver instead of loading a field.
        "TypeHandle" => (u32::MAX, None),
        _ => return None,
    })
}

/// True for the receiver type of a descriptor-valued expression
/// (`typeof(X)` types as `System.Type`; a `Type` variable too).
pub(crate) fn is_type_receiver(type_name: &str) -> bool {
    let b = super::select::base_type_name(type_name);
    matches!(b.as_str(), "Type" | "RuntimeType")
}
