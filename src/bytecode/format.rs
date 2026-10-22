//! Constants and node tags of the bytecode payload format.

/// Current format version written by the encoder.
///
/// v5: v4 plus an `attr_value::FLOAT` tag for float attribute args
/// (`DefaultValue(100d)` — previously unrepresentable).
/// v4: v3 plus a dedicated `EXPR_ASSIGN` tag for `Expr::Assign`
/// (previously mis-encoded as `ASSIGN_COMPLEX`, decoding to the
/// semantically distinct `AssignComplex` — broke round-trips of
/// index/element assignment like `labels[5] = 0xFFFF`).
/// v3: v2 plus `Access::Internal` as access code 4 (was collapsed to
/// `Default` on decode, breaking round-trips of `internal` members).
/// v2: v1 plus the range-slice flag on index nodes (`Expr::Index`
/// carries `is_range`, one byte).
/// v1: full bytecode payload — classes, methods and fields (each carrying the
/// `static` modifier), delegates, properties, indexers, events, enums,
/// attributes, and source assembly blocks.
/// Pre-release: mismatched payloads fail closed rather than misreading.
/// v9: `NEW` carries the object-initializer member list.
/// v10: `PROPERTY` flags carry `is_abstract` (no storage for abstract
/// accessors — previously indistinguishable from auto-properties).
/// v11: `FIELD` carries `is_byref` (`ref T` slots hold addresses).
/// v12: `CHAR` expression tag for character literals (previously they
///      decoded as integers, so `var c = 'A'` lost the `char` type).
pub const FORMAT_VERSION: u16 = 15;

/// Node tags. One byte per node, children follow recursively.
///
/// Layout per tag:
///- `UNIT` opt-namespace(u8), member_count(u32), members,
///  assembly_block_count(u32), assembly block strings
///- `NAMESPACE` name(string), member_count(u32), members
///- `CLASS` name(string), base_count(u32), bases, member_count(u32), members
///- `METHOD` ret_type(string), name(string), param_count(u32),
///  params, stmt_count(u32), statements
///- `DELEGATE` access(u8), ret_type(string), name(string),
///  generic_param_count(u32), generic params, param_count(u32), params
///- `FIELD` access(u8), const_bit(u8), static_bit(u8), attrs,
///  type_name(string), name(string), init(expr)
///- `PROPERTY` access(u8), flags(u8: bit0 settable, bit1 init-only),
///  attrs, type_name(string), name(string), init(expr),
///  get(opt block), set(opt block), get_expr(opt expr)
///- `INDEXER` access(u8), flags(u8: bit0 settable), attrs, type_name(string),
///  param_count(u32), params, init(expr),
///  get(opt block), set(opt block), get_expr(opt expr)
///- `EVENT` access(u8), attrs, type_name(string), name(string)
///- `PARAM` type_name(string), name(string)
///- `CALL` callee(string), arg_count(u32), arguments
///- `BIN` op(u8), lhs(expr), rhs(expr)
///- `UNARY` op(u8), inner(expr)
///- `LET` type(string), name(string), access(u8), init(expr)
///- `ASSIGN` name(string), value(expr)
///- `STR` value(string)
///- `INT` value(u64)
///- `IDENT` name(string)
pub mod tag {
    pub const UNIT: u8 = 0x01;
    pub const NAMESPACE: u8 = 0x02;
    pub const CLASS: u8 = 0x03;
    pub const INTERFACE: u8 = 0x06;
    pub const NEW: u8 = 0x1F;
    pub const MEMBER: u8 = 0x20;
    pub const RETURN: u8 = 0x1A;
    pub const FIELD: u8 = 0x1B;
    pub const METHOD_CALL: u8 = 0x1C;
    /// Attribute list prefix (attr_count follows).
    pub const ATTRS: u8 = 0x1D;
    pub const MEMBER_ASSIGN: u8 = 0x1E;
    pub const METHOD: u8 = 0x04;
    pub const PARAM: u8 = 0x05;
    pub const CALL: u8 = 0x10;
    pub const BIN: u8 = 0x14;
    pub const UNARY: u8 = 0x15;
    pub const LET: u8 = 0x16;
    pub const ASSIGN: u8 = 0x17;
    pub const STR: u8 = 0x11;
    pub const INT: u8 = 0x12;
    pub const IDENT: u8 = 0x13;
    pub const BOOL: u8 = 0x18;
    pub const TERNARY: u8 = 0x19;
    pub const NULL: u8 = 0x1A;
    pub const NEW_ARRAY: u8 = 0x21;
    pub const INDEX: u8 = 0x22;
    pub const DELEGATE: u8 = 0x23;
    pub const PROPERTY: u8 = 0x24;
    pub const INDEXER: u8 = 0x25;
    pub const EVENT: u8 = 0x26;
    pub const CAST: u8 = 0x27;
    /// Switch expression node: subject(expr), arm_count(u32),
    /// arms(arm_count * (label_count(u32) + label strings + body expr)).
    pub const SWITCH_EXPR: u8 = 0x28;
    /// Compound assignment: lhs(expr), value(expr).
    pub const ASSIGN_COMPLEX: u8 = 0x29;
    /// Enum type: name(string), underlying_type(string),
    /// variant_count(u32), variants(variant_count * (name(string), value_present(byte) + value(u64))).
    pub const ENUM: u8 = 0x2A;
    /// Nested type inside a class: followed by a member (class/interface/enum/delegate).
    pub const NESTED: u8 = 0x2B;
    /// `default(T)` — type_name(string).
    pub const DEFAULT: u8 = 0x2C;
    /// `await expr` — inner(expr).
    pub const AWAIT: u8 = 0x2D;
    /// `float`/`double` literal — bits(u64) of the `f64` value.
    pub const FLOAT: u8 = 0x2E;
    /// `char` literal — the UTF-16 code unit (u32).
    pub const CHAR: u8 = 0x40;
    /// `$"..."` interpolated string — count(u32) of parts; each part is a
    /// kind byte (0 = text `str`, 1 = hole `expr`).
    pub const INTERP_STRING: u8 = 0x2F;
    /// `target++` — target(expr). Yields the old value.
    pub const POST_INC: u8 = 0x30;
    /// `target--` — target(expr). Yields the old value.
    pub const POST_DEC: u8 = 0x31;
    /// A lambda. The bytecode model does not carry closures, so this tag
    /// exists to make the gap EXPLICIT: encoding writes it and decoding
    /// REJECTS it, rather than the encoder picking a shape that decodes to
    /// a different expression (which is how a lambda would silently become
    /// something else on a round-trip).
    pub const LAMBDA: u8 = 0x3E;
    /// `checked(expr)` overflow-control wrapper.
    pub const CHECKED: u8 = 0x3F;
    /// Simple assignment expression `target = value`:
    /// target(expr), value(expr). Distinct from `ASSIGN_COMPLEX`
    /// (`lhs op= rhs` shape with complex-LHS lowering).
    pub const EXPR_ASSIGN: u8 = 0x3B;

    /// Option prefix: absent / present, used for optional child nodes.
    pub const ABSENT: u8 = 0x00;
    pub const PRESENT: u8 = 0x01;

    /// Attribute value kinds.
    pub mod attr_value {
        pub const STR: u8 = 0;
        pub const INT: u8 = 1;
        pub const FLOAT: u8 = 2;
    }

    /// Expression operator tags.
    pub mod binop {
        pub const ADD: u8 = 0;
        pub const SUB: u8 = 1;
        pub const MUL: u8 = 2;
        pub const DIV: u8 = 3;
        pub const EQ: u8 = 4;
        pub const NE: u8 = 5;
        pub const LT: u8 = 6;
        pub const GT: u8 = 7;
        pub const LE: u8 = 8;
        pub const GE: u8 = 9;
        pub const AND: u8 = 10;
        pub const OR: u8 = 11;
        pub const REM: u8 = 12;
        pub const BIT_AND: u8 = 13;
        pub const BIT_OR: u8 = 14;
        pub const BIT_XOR: u8 = 15;
        pub const SHL: u8 = 16;
        pub const SHR: u8 = 17;
        pub const SHR_SHR: u8 = 18;
        pub const IS: u8 = 19;
        pub const NULL_COALESCE: u8 = 20;
    }
    pub mod unop {
        pub const NEG: u8 = 0;
        pub const NOT: u8 = 1;
        pub const BIT_NOT: u8 = 2;
        pub const DEREF: u8 = 3;
        pub const ADDR_OF: u8 = 4;
        pub const THROW: u8 = 5;
        pub const INDEX_FROM_END: u8 = 6;
    }

    /// Statement tags.
    pub mod stmt {
        pub const IF: u8 = 0x30;
        pub const WHILE: u8 = 0x31;
        pub const DO_WHILE: u8 = 0x32;
        pub const FOR: u8 = 0x33;
        pub const BREAK: u8 = 0x34;
        pub const CONTINUE: u8 = 0x35;
        pub const SWITCH: u8 = 0x36;
        pub const GOTO: u8 = 0x37;
        pub const TRY: u8 = 0x38;
        pub const THROW: u8 = 0x39;
        /// Standalone `{ ... }` block — block(block).
        pub const BLOCK: u8 = 0x3A;
        /// Local function declaration — string(name), u32(params) +
        /// param() each, string(ret_type), block(body), byte(is_static).
        pub const LOCAL_FUNCTION: u8 = 0x3B;
        /// Goto label definition — string(name).
        pub const LABEL: u8 = 0x3C;
        /// `fixed` statement — u32(n) + (string(type), string(name),
        /// byte(has-init) + expr?) each, then block(body).
        pub const FIXED: u8 = 0x3D;
        /// `checked { ... }` — block(body).
        pub const CHECKED_BLOCK: u8 = 0x3E;
    }
}
