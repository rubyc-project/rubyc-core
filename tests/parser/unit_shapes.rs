//! Parser tests: valid programs must produce the expected tree.

use crate::models::compare::StructuralEq;
use crate::models::{ClassDecl, ClassMember, Expr, Literal, Member, Stmt};
use crate::testdata::{BLOCK_FORM, FILE_SCOPED_FORM, KN_R_STYLE, ONE_LINE};
use crate::{parser, prelude::*};

/// All sources must parse to the same structure (spans ignored).
fn assert_same_ast(sources: &[&str]) {
    let units: Vec<CompilationUnit> = sources
        .iter()
        .map(|source| {
            parser::parse(source)
                .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{source}"))
        })
        .collect();
    for unit in &units {
        assert!(
            unit.structural_eq(&units[0]),
            "ASTs differ between source variants"
        );
    }
}

#[test]
fn hello_world_variants_normalize_to_one_ast() {
    assert_same_ast(&[BLOCK_FORM, FILE_SCOPED_FORM, KN_R_STYLE, ONE_LINE]);
}

#[test]
fn parses_expected_structure() {
    let unit = parser::parse(BLOCK_FORM).unwrap();
    let ns = unit.namespaces.first().expect("namespace should be present");
    assert_eq!(ns.name, "testnc");

    assert!(unit.members.is_empty());
    assert_eq!(ns.members.len(), 1);

    let Member::Class(class) = &ns.members[0] else {
        panic!("expected class member")
    };
    assert_eq!(class.name, "testc");
    assert_eq!(class.members.len(), 1);

    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    assert_eq!(method.ret_type, "int");
    assert_eq!(method.name, "main");
    assert_eq!(method.params.len(), 0);
    assert_eq!(method.body.as_ref().unwrap().stmts.len(), 1);

    match &method.body.as_ref().unwrap().stmts[0] {
        Stmt::Expression(Expr::Call { callee, args, .. }) => {
            assert_eq!(callee, "printf");
            assert_eq!(args.len(), 1);
            assert_eq!(
                args[0],
                Expr::Literal(Literal::Str("Hello World!".to_owned()))
            );
        }
        other => panic!("expected a call statement, got {other:?}"),
    }
}

#[test]
fn spans_point_at_source_text() {
    let unit = parser::parse(ONE_LINE).unwrap();
    // The namespace span must cover the keyword itself.
    assert_eq!(
        &ONE_LINE[unit.namespaces[0].span.start as usize..][..9],
        "namespace"
    );
}

// Silence unused-import warning for re-exported helper types.
#[allow(unused)]
fn _type_witnesses(_: ClassDecl) {}

#[test]
fn ref_struct_and_ref_field_parse() {
    let src = r#"namespace test {
    public ref struct R {
        public ref int Value;
        public R(ref int value) {
            Value = ref value;
        }
    }
}"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let ns = &unit.namespaces[0];
    assert_eq!(ns.members.len(), 1);
}

#[test]
fn object_initializer_parse() {
    let src = r#"namespace test {
    public class Foo {
        public int X;
        public int Y;
        public static Foo Create() {
            return new Foo { X = 1, Y = 2 };
        }
    }
}"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let ns = &unit.namespaces[0];
    assert_eq!(ns.members.len(), 1);
}

#[test]
fn object_initializer_members_captured() {
    // `new Foo { X = 1, Y = 2 }` must capture the member writes on
    // `Expr::New.init` (they run after construction — field stores
    // or setter dispatch). Index- and collection-initializer
    // elements stay dropped (documented gap).
    let src = r#"namespace test {
    public class Foo {
        public int X;
        public int Y;
        public static Foo Create() {
            return new Foo { X = 1, Y = 2 };
        }
    }
}"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(decl) = &unit.namespaces[0].members[0] else {
        panic!("expected class");
    };
    let m = decl
        .members
        .iter()
        .find_map(|m| match m {
            ClassMember::Method(m) if m.name == "Create" => Some(m),
            _ => None,
        })
        .expect("Create method");
    let body = m.body.as_ref().expect("body");
    let Stmt::Return { value: Some(Expr::New { init, .. }), .. } = &body.stmts[0] else {
        panic!("expected return-new, got {:?}", body.stmts[0]);
    };
    assert_eq!(init.len(), 2, "both member writes captured");
    assert_eq!(init[0].name, "X");
    assert_eq!(init[1].name, "Y");
    assert!(matches!(init[0].value, Expr::Literal(Literal::Int(1))));
}

#[test]
fn ref_in_assignment_parse() {
    let src = r#"namespace test {
    public ref struct RefFieldHolder {
        public ref int Value;
        public RefFieldHolder(ref int value) {
            Value = ref value;
        }
    }
}"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let ns = &unit.namespaces[0];
    assert_eq!(ns.members.len(), 1);
}

#[test]
fn array_size_literal_is_captured() {
    // Regression: `new int[4]` used to parse as size 0 (the size expression
    // was consumed and discarded).
    let src = r#"namespace test {
    public class C {
        public int main() {
            int[] a = new int[4];
            return 0;
        }
    }
}"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let ns = &unit.namespaces[0];
    let Member::Class(class) = &ns.members[0] else {
        panic!("expected class");
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method");
    };
    let stmts = method.body.as_ref().unwrap().stmts.as_slice();
    match &stmts[0] {
        Stmt::Let { init, .. } => match &init {
            Expr::NewArray { elem_type, size, .. } => {
                assert_eq!(elem_type, "int");
                assert_eq!(size.as_ref(), &Expr::Literal(Literal::Int(4)));
            }
            other => panic!("expected NewArray, got {other:?}"),
        },
        other => panic!("expected Let, got {other:?}"),
    }
}

#[test]
fn array_index_literal_is_captured() {
    // Regression: `a[1]` used to parse as index 0 (the index expression was
    // consumed and discarded).
    let src = r#"namespace test {
    public class C {
        public int main() {
            int[] a = new int[4];
            int x = a[1];
            return 0;
        }
    }
}"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let ns = &unit.namespaces[0];
    let Member::Class(class) = &ns.members[0] else {
        panic!("expected class");
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method");
    };
    let stmts = method.body.as_ref().unwrap().stmts.as_slice();
    match &stmts[1] {
        Stmt::Let { init, .. } => match &init {
            Expr::Index { object, index, .. } => {
                assert!(matches!(
                    object.as_ref(),
                    Expr::Ident { name, .. } if name == "a"
                ));
                assert_eq!(index.as_ref(), &Expr::Literal(Literal::Int(1)));
            }
            other => panic!("expected Index, got {other:?}"),
        },
        other => panic!("expected Let, got {other:?}"),
    }
}

#[test]
fn stackalloc_size_literal_is_captured() {
    // Regression: `stackalloc T[n]` used to parse as size 0.
    let src = r#"namespace test {
    public class C {
        public int main() {
            int* p = stackalloc int[8];
            return 0;
        }
    }
}"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let ns = &unit.namespaces[0];
    let Member::Class(class) = &ns.members[0] else {
        panic!("expected class");
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method");
    };
    let stmts = method.body.as_ref().unwrap().stmts.as_slice();
    match &stmts[0] {
        Stmt::Let { init, .. } => match &init {
            Expr::NewArray { elem_type, size, .. } => {
                assert_eq!(elem_type, "int");
                assert_eq!(size.as_ref(), &Expr::Literal(Literal::Int(8)));
            }
            other => panic!("expected NewArray, got {other:?}"),
        },
        other => panic!("expected Let, got {other:?}"),
    }
}

#[test]
fn array_index_variable_is_captured() {
    // The index must be the variable reference, not a synthetic literal.
    let src = r#"namespace test {
    public class C {
        public int main() {
            int[] a = new int[4];
            int i = 2;
            int x = a[i];
            return 0;
        }
    }
}"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let ns = &unit.namespaces[0];
    let Member::Class(class) = &ns.members[0] else {
        panic!("expected class");
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method");
    };
    let stmts = method.body.as_ref().unwrap().stmts.as_slice();
    match &stmts[2] {
        Stmt::Let { init, .. } => match &init {
            Expr::Index { index, .. } => {
                assert!(matches!(
                    index.as_ref(),
                    Expr::Ident { name, .. } if name == "i"
                ));
            }
            other => panic!("expected Index, got {other:?}"),
        },
        other => panic!("expected Let, got {other:?}"),
    }
}

#[test]
fn multiple_block_namespaces_in_one_file() {
    let src = r#"namespace A { class C { public static int Get() { return 41; } } }
namespace B { class Program { public static int Main() { return 0; } } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    assert_eq!(unit.namespaces.len(), 2);
    assert_eq!(unit.namespaces[0].name, "A");
    assert_eq!(unit.namespaces[1].name, "B");
    assert!(unit.members.is_empty());
    // Each namespace keeps its own members; they do not leak across.
    assert_eq!(unit.namespaces[0].members.len(), 1);
    assert_eq!(unit.namespaces[1].members.len(), 1);
    // `all_members` flattens in source order (A's class then B's class).
    assert_eq!(unit.all_members().len(), 2);
}

#[test]
fn namespace_and_top_level_member_coexist() {
    let src = r#"namespace A { class C { public int x() { return 1; } } }
class Top { public int y() { return 2; } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    assert_eq!(unit.namespaces.len(), 1);
    assert_eq!(unit.namespaces[0].name, "A");
    assert_eq!(unit.members.len(), 1);
    // The top-level class is discoverable by name for context resolution.
    assert_eq!(unit.namespace_name_of("Top"), None);
    assert_eq!(unit.namespace_name_of("C"), Some("A".to_string()));
}

#[test]
fn target_typed_new_with_identifier_arg_parses() {
    // `new(cancellationToken)` — C# 9 target-typed new where the first
    // argument is an identifier. Regression: the parser's `new (` guard
    // rejected an identifier in first position, so this failed with
    // "expected identifier as type".
    let src = r#"class C { Main() { DelayState state = new(cancellationToken); } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    let Some(block) = &method.body else {
        panic!("expected method body")
    };
    let Stmt::Let { init, .. } = &block.stmts[0] else {
        panic!("expected let statement")
    };
    assert!(
        matches!(init, Expr::New { class_name, .. } if class_name.is_empty()),
        "expected target-typed New with empty class_name, got {init:?}"
    );
}

#[test]
fn attribute_interpolated_string_arg_parses() {
    // `Justification = $"... ({nameof(X)}) ..."` — interpolated
    // strings lex as one token; accept them like plain strings.
    let src = r#"class C { [Attr("A", Justification = $"x ({nameof(Y)}) y")] void M() { } }"#;
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn attribute_cast_and_group_args_parse() {
    // `[Event((int)E.X, Level = ...)]` — casts and groupings in
    // attribute arguments (balanced parens + optional operand).
    let src = r#"class C { [Event((int)E.X, Level = L.Info)] void M() { } }"#;
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn attribute_arithmetic_tail_parses() {
    // `[FieldOffset(1 * N)]` — constant arithmetic in attribute
    // arguments. The tail loop must run after EVERY value shape
    // (regression: it only ran after names, so `1 * X` failed).
    let src = r#"struct S { [FieldOffset(1 * Internal.PaddingHelpers.CACHE_LINE_SIZE)] public int Head; }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    assert!(
        class.members.iter().any(|m| matches!(
            m,
            ClassMember::Field(f) if f.name == "Head"
        )),
        "attributed field must survive member parsing"
    );
}

#[test]
fn attribute_flag_or_tail_parses() {
    // `[Attr(A | B)]` — enum flags combined with `|` in an attribute
    // argument (pervasive in pulled dotnet sources). Attribute values are
    // inert metadata; the parse must simply survive the tail.
    let src = r#"class C { [Attr(A.B | A.C)] Main() { } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    assert_eq!(method.attrs.len(), 1);
    assert_eq!(method.attrs[0].name, "Attr");
}

#[test]
fn colon_named_attribute_argument_parses() {
    // `[NotNullWhen(returnValue: true)]` — C# named attribute
    // arguments accept `:` as well as `=`.
    let src = r#"class C { [Attr(returnValue: true)] void M() { } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    assert_eq!(method.attrs.len(), 1);
    assert_eq!(method.attrs[0].arguments.len(), 1);
    assert_eq!(
        method.attrs[0].arguments[0].name.as_deref(),
        Some("returnValue")
    );
}

#[test]
fn signed_attribute_argument_parses() {
    // `[OverloadResolutionPriority(-1)]` — C# attribute arguments
    // allow unary `-`/`+` on constants.
    let src = r#"class C { [Attr(-1)] [Attr2(+2)] Main() { } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    assert_eq!(method.attrs.len(), 2);
}

#[test]
fn comma_separated_attributes_share_one_bracket() {
    // `[A("x"), B]` — C# allows several attributes in one bracket.
    let src = r#"class C { [A("x"), B] Main() { } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    assert_eq!(method.attrs.len(), 2);
    assert_eq!(method.attrs[0].name, "A");
    assert_eq!(method.attrs[1].name, "B");
}

#[test]
fn local_const_parses_as_let() {
    // `const int mask = 3;` — const-ness is a compile-time rule, not a
    // runtime representation; it lowers exactly like a local.
    let src = r#"class C { Main() { const int mask = 3; return mask; } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    let Some(block) = &method.body else {
        panic!("expected method body")
    };
    let Stmt::Let {
        type_name, name, init, ..
    } = &block.stmts[0]
    else {
        panic!("expected let statement")
    };
    assert_eq!(type_name, "int");
    assert_eq!(name, "mask");
    assert!(matches!(init, Expr::Literal(Literal::Int(3))));
}

#[test]
fn readonly_accessor_modifier_parses() {
    // `readonly get => ...` — the readonly modifier on accessors of
    // readonly structs/members.
    let src = r#"struct S { int _l; int Length { readonly get => _l; set => _l = value; } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    assert!(
        class.members.iter().any(|m| matches!(
            m,
            ClassMember::Property(p) if p.name == "Length"
        )),
        "readonly-getter property must survive member parsing"
    );
}

#[test]
fn interface_qualified_method_name_parses() {
    // `IUtf8SpanFormattable.TryFormat(...)` — explicit implementation
    // qualification on interface members (mirrors `method_decl`).
    let src = r#"interface I { unsafe bool IUtf8SpanFormattable.TryFormat(Span<byte> dest, out int written); }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Interface(iface) = &unit.members[0] else {
        panic!("expected interface member")
    };
    assert!(
        iface.methods.iter().any(|m| m.name == "IUtf8SpanFormattable.TryFormat"),
        "qualified method must survive interface parsing"
    );
}

#[test]
fn interface_unsafe_modifier_parses() {
    // `unsafe void M(byte* p);` — `unsafe` is legal on interface
    // members (same discard list as new/static/extern/...).
    let src = r#"interface I { unsafe void Encode(byte* oneByte, int* dest); }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Interface(iface) = &unit.members[0] else {
        panic!("expected interface member")
    };
    assert!(
        iface.methods.iter().any(|m| m.name == "Encode"),
        "unsafe interface method must survive parsing"
    );
}

#[test]
fn interface_expression_bodied_member_parses() {
    // `static virtual bool M(...) => expr;` — expression-bodied
    // interface members (static virtuals) run for value.
    let src = r#"interface I {
        static virtual bool M(ReadOnlySpan<char> input, out int result) => Parser.Try(input, out result);
        bool N(int x);
    }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Interface(iface) = &unit.members[0] else {
        panic!("expected interface member")
    };
    assert!(
        iface.methods.iter().any(|m| m.name == "M" && m.body.is_some()),
        "expression-bodied member must survive with a body"
    );
    assert!(
        iface.methods.iter().any(|m| m.name == "N" && m.body.is_none()),
        "bodiless member must survive without a body"
    );
}

#[test]
fn interface_tuple_return_method_parses() {
    // `static virtual (TSelf, TSelf) DivRem(...)` — tuple returns on
    // interface methods fold like tuple-typed members.
    let src = r#"interface I {
        static virtual (TSelf Quotient, TSelf Remainder) DivRem(TSelf left, TSelf right) {
            return null;
        }
    }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Interface(iface) = &unit.members[0] else {
        panic!("expected interface member")
    };
    assert!(
        iface.methods.iter().any(|m| m.name == "DivRem"),
        "tuple-return method must survive interface parsing"
    );
}

#[test]
fn interface_generic_method_where_constraints_parse() {
    // `M<T>(...) where T : C;` — constraints on generic interface
    // methods (the member-method path already had them).
    let src = r#"interface I {
        OperationStatus M<TDecoder>(TDecoder decoder, bool last) where TDecoder : IDecoder<T>;
    }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Interface(iface) = &unit.members[0] else {
        panic!("expected interface member")
    };
    assert!(
        iface.methods.iter().any(|m| m.name == "M"),
        "constrained generic method must survive interface parsing"
    );
}

#[test]
fn interface_generic_method_parses() {
    // `Ret Name<T>(params);` — generic methods in interfaces (mirrors
    // `method_decl`). Regression: "expected '(' ... found 'Less'".
    let src = r#"interface I {
        IOrderedEnumerable<TElement> CreateOrderedEnumerable<TKey>(Func<TElement, TKey> keySelector, bool descending);
    }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Interface(iface) = &unit.members[0] else {
        panic!("expected interface member")
    };
    assert!(
        iface.methods.iter().any(|m| m.name == "CreateOrderedEnumerable"),
        "generic method must survive interface parsing"
    );
}

#[test]
fn interface_member_attributes_are_skipped() {
    // `[Obsolete(...)]` on interface members: contracts only, inert
    // for signatures. Regression: the interface body loop never
    // consumed them ("expected identifier as type, found '['").
    let src = r#"interface I {
        [Obsolete("Use Y instead.")]
        ISymbolReader? GetReader(int importer);
        [Attr] int P { get; }
    }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Interface(iface) = &unit.members[0] else {
        panic!("expected interface member")
    };
    assert!(
        iface.methods.iter().any(|m| m.name == "GetReader"),
        "attributed method must survive interface parsing"
    );
}

#[test]
fn attribute_return_target_specifier_parses() {
    // `[return: NotNullIfNotNull(...)]` — the target selects where the
    // attribute applies; it is consumed and discarded.
    let src = r#"class C { [return: Attr(nameof(x))] string M(string x) { return x; } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    assert_eq!(method.attrs.len(), 1);
    assert_eq!(method.attrs[0].name, "Attr");
}

#[test]
fn event_custom_accessors_parse() {
    // `event T N { add { } remove { } }` — accessor bodies are inert
    // here; the event still declares as a field-like member.
    let src = r#"class C { public event Handler H { add { } remove { } } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Event(event) = &class.members[0] else {
        panic!("expected event member")
    };
    assert_eq!(event.name, "H");
}

#[test]
fn explicit_interface_event_parses() {
    // `event T IFoo.Bar { add => ...; remove => ...; }` — explicit
    // interface qualification records the full path; expression
    // accessors are skipped like block ones.
    let src = r#"class C { event Handler? IFoo.Bar { add => Changed += value; remove => Changed -= value; } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Event(event) = &class.members[0] else {
        panic!("expected event member")
    };
    assert_eq!(event.name, "IFoo.Bar");
}

#[test]
fn enum_variant_attributes_are_skipped() {
    // `[Obsolete(...)] AllowNone` — attributes on enum variants are
    // contracts only, inert for codegen.
    let src = "enum E {\n    [System.ObsoleteAttribute(\"Use Y.\")]\n    AllowNone,\n    Normal = 1,\n}";
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Enum(decl) = &unit.members[0] else {
        panic!("expected enum member")
    };
    assert!(decl.variants.iter().any(|(n, _)| n == "AllowNone"));
    assert!(decl.variants.iter().any(|(n, _)| n == "Normal"));
}

#[test]
fn uninitialized_declaration_is_zero_init_without_error() {
    // `Type name;` is legitimate C# — zero-init, no diagnostic.
    let src = r#"class C { Main() { int n; n = 1; return n; } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    let Some(block) = &method.body else {
        panic!("expected method body")
    };
    let Stmt::Let { name, init, .. } = &block.stmts[0] else {
        panic!("expected let statement")
    };
    assert_eq!(name, "n");
    assert!(matches!(init, Expr::Literal(Literal::Int(0))));
}

#[test]
fn prefix_increment_in_for_step_parses() {
    // `for (...; ...; ++i)` — prefix form desugars like postfix.
    let src = r#"class C { Main() { for (int i = 0; i < 3; ++i) { } } }"#;
    parser::parse(src).unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn for_multi_expression_init_and_step() {
    // `for (i = 0, j = 0; ...; i += 2, j++)` — comma-separated
    // expression inits and steps (not just declarations/steps).
    let src = r#"class C { Main() { for (i = 0, j = 0; i < n; i += 2, j++) { } } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    let Some(block) = &method.body else {
        panic!("expected method body")
    };
    assert!(
        matches!(&block.stmts[0], Stmt::For { .. }),
        "multi-expression for must parse as a For"
    );
}

#[test]
fn for_init_accepts_full_types() {
    // `for (ImmutableStack<T> f = this; ...)` — for-init declarations
    // take full types (generics, `?`, `[]`, dotted). The non-declaration
    // forms (`for (i = 0; ...)`, `for (;;)`) must keep working.
    for src in [
        r#"class C { Main() { for (int i = 0; i < 3; i++) { } } }"#,
        r#"class C { Main() { for (ImmutableStack<T> f = this; !f.IsEmpty; f = f.Pop()) { } } }"#,
        r#"class C { Main() { for (Ns.Outer<int>.Inner? x = y; x != null; x = x.Next()) { } } }"#,
        r#"class C { Main() { for (i = 0; i < 3; i++) { } } }"#,
        r#"class C { Main() { for (int i = 0, n = 5; i < n; i++) { } } }"#,
        r#"class C { Main() { for (;;) { break; } } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn parenthesized_assign_then_member_call_parses() {
    // `(x ??= y).Add(z);` — the compound-assign lookahead must ignore
    // operators nested in parens (it used to fire on the inner `??=`,
    // then eat the `;` and corrupt the next statement). Same for all
    // compound forms.
    for src in [
        r#"class C { Main() { (x ??= y).Add(z); return 0; } }"#,
        r#"class C { Main() { (x += y).Add(z); return 0; } }"#,
        r#"class C { Main() { (a.b ??= y).Add(z); return 0; } }"#,
        r#"class C { Main() { o.F += v; return 0; } }"#,
        r#"class C { Main() { arr[i] += v; return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn member_compound_assign_statement_stores_once() {
    // `o.F += v;` — the statement path must not double-consume now that
    // expressions handle `op=` themselves (regression: it ate the `;`
    // and failed on the following token).
    let src = r#"class C { Main() { o.F += v; } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    let Some(block) = &method.body else {
        panic!("expected method body")
    };
    assert!(
        matches!(
            &block.stmts[0],
            Stmt::Expression(Expr::AssignComplex { .. })
        ),
        "expected AssignComplex statement, got {:?}",
        block.stmts[0]
    );
}

#[test]
fn postfix_member_call_terminates() {
    // `X.Y()?.M()`, `p->F(1)`, `a.b<T>(1)` — a `(` after a member
    // made by `?.`/`->` (or any non-generic path) must parse as a
    // plain call. Regression: the generic-call arm fired on every
    // `Member + (`, and with a non-nameable receiver it consumed
    // nothing, spinning the postfix loop forever.
    for src in [
        r#"class C { Main() { X.Y()?.M(); return 0; } }"#,
        r#"class C { Main() { var s = p->F(1); return 0; } }"#,
        r#"class C { Main() { var s = a.b<T>(1); return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn null_forgiving_call_parses() {
    // `Callback!(args)` / `o.M!(x)` — null-forgiving on an invocation
    // target is transparent; the call still parses. Regression: the
    // postfix loop had no `Ident + (` arm (primary owns plain calls),
    // so `(` after `!` errored.
    let src = r#"class C { Main() { Callback!(1, 2); o.M!(3); return 0; } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    let Some(block) = &method.body else {
        panic!("expected method body")
    };
    assert!(
        matches!(&block.stmts[0], Stmt::Expression(Expr::Call { callee, .. }) if callee == "Callback"),
        "expected Callback call, got {:?}",
        block.stmts[0]
    );
}

#[test]
fn postfix_inc_in_argument_position() {
    // `f(i++)` — postfix increment is an expression yielding the old
    // value; it must survive argument-list parsing as a PostInc node.
    let src = r#"class C { Main() { f(i++); g(a[j]--); } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    let Some(block) = &method.body else {
        panic!("expected method body")
    };
    let Stmt::Expression(Expr::Call { args, .. }) = &block.stmts[0] else {
        panic!("expected call statement")
    };
    assert!(
        matches!(args[0], Expr::PostInc { .. }),
        "expected PostInc arg, got {:?}",
        args[0]
    );
    let Stmt::Expression(Expr::Call { args, .. }) = &block.stmts[1] else {
        panic!("expected call statement")
    };
    assert!(
        matches!(&args[0], Expr::PostDec { target, .. } if matches!(target.as_ref(), Expr::Index { .. })),
        "expected PostDec-of-index arg, got {:?}",
        args[0]
    );
}

#[test]
fn pointer_member_access_desugars_to_deref() {
    // `p->Field` ≡ `(*p).Field` — longest-match lexes `->` as one token.
    let src = r#"class C { Main() { int x = p->Field; return x; } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    let Some(block) = &method.body else {
        panic!("expected method body")
    };
    let Stmt::Let { init, .. } = &block.stmts[0] else {
        panic!("expected let statement")
    };
    assert!(
        matches!(
            init,
            Expr::Member { object, .. }
            if matches!(object.as_ref(), Expr::Unary { op: crate::models::UnOp::Deref, .. })
        ),
        "expected Member(Deref) shape, got {init:?}"
    );
}

#[test]
fn fixed_multi_declarator_parses() {
    // `fixed (char* v0 = ..., v1 = ...)` — declarators after the first
    // share the one type (regression: "to close fixed statement,
    // found ','").
    let src = r#"class C { Main() { fixed (char* v0 = (string?)a[0], v1 = (string?)a[1]) { x = 1; } return 0; } }"#;
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn standalone_block_keeps_statements() {
    // `{ ... }` as a statement must keep its statements in a real Block
    // node with its own scope — never drop them (regression: they were
    // replaced by a span-only placeholder, silently deleting code).
    let src = r#"class C { Main() { int x = 1; { x = 2; } return x; } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    let Some(block) = &method.body else {
        panic!("expected method body")
    };
    assert_eq!(block.stmts.len(), 3);
    let Stmt::Block { body, .. } = &block.stmts[1] else {
        panic!("expected block statement, got {:?}", block.stmts[1])
    };
    assert_eq!(body.stmts.len(), 1);
}

#[test]
fn class_primary_ctor_with_base_args_parses() {
    // `class C(A a) : Base(a)` — C# 12 primary-constructor class with
    // base-constructor arguments. Both lists are consumed, not recorded.
    let src = r#"class B { } class C { private sealed class D(int x) : B { } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[1] else {
        panic!("expected class member")
    };
    assert!(
        class.members.iter().any(|m| matches!(
            m,
            ClassMember::Nested(nested) if nested.name() == "D"
        )),
        "nested primary-ctor class must survive member parsing"
    );
}

#[test]
fn switch_arm_return_then_local_function_passes() {
    // `default: return X; static async Task F() { ... }` — a trailing
    // local-function declaration never executes at the arm's end point,
    // so the `return` satisfies the no-fallthrough rule. Regression:
    // only the literal last statement was checked.
    let src = r#"class C { Main() {
        switch (t) {
            case 1: return 1;
            default:
                return DisposeAsync(x, y);
                static async Task DisposeAsync(int a, int b) { return 0; }
        }
    } }"#;
    parser::parse(src).unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn switch_default_stacks_with_case_labels() {
    // `default: case X:` — default stacks with further labels sharing
    // one body, per C#. Previously collection stopped at `default:`,
    // misparsing the `case` as a body statement (fallthrough error).
    let src = r#"class C { Main() {
        switch (t) {
            default:
            case 1: return 1;
        }
    } }"#;
    parser::parse(src).unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn switch_arm_ending_in_block_or_if_else_passes() {
    // C# end-point rule: an arm is fine when control cannot fall out —
    // a block ending in a jump, or an if/else with jumps on both sides.
    // Regression: only a top-level trailing jump was accepted, so
    // `case X: { ...; return y; }` falsely errored.
    let src = r#"class C { Main() {
        switch (t) {
            case 1: { return 1; }
            case 2: if (c) { return 2; } else { return 3; }
            default: return d;
        }
    } }"#;
    parser::parse(src).unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn switch_arm_trailing_out_lets_ignored() {
    // `case X: return F(a, out b);` — the `out` pre-declaration lands
    // as a trailing `Let` after the `Return`; end-point analysis must
    // skip trailing Lets (they never execute there).
    let src = r#"class C { Main() {
        switch (t) {
            case 'O': return F(value, out written);
            default: return null;
        }
    } }"#;
    parser::parse(src).unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn is_pattern_sizeof_operands_parse() {
    // `is sizeof(char) or sizeof(byte)` — `sizeof`/`typeof`/`nameof`
    // in `is`-patterns consume name + balanced parens as one unit.
    for src in [
        r#"class C { Main() { Debug.Assert(sizeof(TChar) is sizeof(char) or sizeof(byte)); return 0; } }"#,
        r#"class C { Main() { if (x is typeof(int)) { return 1; } return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn relational_pattern_full_operand_parses() {
    // `is > -X - 1 and < Y` / `case > -5 - 1:` — relational pattern
    // operands are full shift/additive expressions, not just unary.
    // Regression: the `- 1` derailed `and`-chaining ("found 'And_'").
    for src in [
        r#"class C { Main() { if (value is > -X - 1 and < Y) { return 1; } return 0; } }"#,
        r#"class C { Main() { switch (t) { case > -5 - 1: return 2; default: return 0; } } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn bare_generic_case_and_combined_access() {
    // `case GenericEqualityComparer<string>:` — a bare generic type
    // is a type pattern (constants/comparisons never carry `<>`).
    // `private protected` — combined accessibility modifiers.
    let src = r#"class C {
        Main() {
            switch (t) {
                case GenericEqualityComparer<string>:
                    return 1;
                default:
                    return 0;
            }
        }
        private protected virtual bool M() { return true; }
    }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    assert!(
        class.members.iter().any(|m| matches!(
            m,
            ClassMember::Method(method) if method.name == "M"
        )),
        "combined-access method must survive member parsing"
    );
}

#[test]
fn switch_arm_try_catch_throw_passes() {
    // `case X: try { return ..; } catch (E e) { throw ..; }` — the
    // arm never falls through per C# end-point reachability, as does
    // `try { return ..; } finally { ...; }` (the falling finally runs
    // during unwinding). A falling try still errors.
    for src in [
        r#"class C { Main() { switch (t) { case 1: try { return 1; } catch (E e) { throw e; } default: return 0; } } }"#,
        r#"class C { Main() { switch (t) { case 1: try { return 1; } finally { M(); } default: return 0; } } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
    let bad = r#"class C { Main() { switch (t) { case 1: try { M(); } catch (E e) { N(); } default: return 0; } } }"#;
    let errors = parser::parse(bad).expect_err("falling try should fail");
    assert!(
        errors.iter().any(|e| e.message.contains("fallthrough")),
        "expected fallthrough error, got {:?}",
        errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

#[test]
fn switch_arm_plain_fallthrough_still_errors() {
    // A bare value statement at an arm's end is still a real error.
    let src = r#"class C { Main() { switch (t) { case 1: x = 1; default: return d; } } }"#;
    let errors = parser::parse(src).expect_err("fallthrough should fail");
    assert!(
        errors.iter().any(|e| e.message.contains("fallthrough")),
        "expected fallthrough error, got {:?}",
        errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}

#[test]
fn nested_generic_qualified_method_name_parses() {
    // `ICollection<KeyValuePair<K, V>>.CopyTo(...)` — qualifier lists
    // nest generics (`>>` closers); only the pre-`(` list records.
    let src = r#"class C : ICollection<KeyValuePair<TKey, TValue>> {
        void ICollection<KeyValuePair<TKey, TValue>>.CopyTo(KeyValuePair<TKey, TValue>[] array, int index) {
        }
    }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    assert!(
        class.members.iter().any(|m| matches!(
            m,
            ClassMember::Method(method) if method.name.ends_with("CopyTo")
        )),
        "qualified generic method must survive member parsing"
    );
}

#[test]
fn using_alias_directive_recorded() {
    // `using NativeType = System.Int32;` records (name, target);
    // plain and `static` usings are still discarded.
    let src = "using System;\nusing static System.Math;\nusing NativeType = System.Int32;\nclass C {\n}\n";
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    assert_eq!(unit.aliases.len(), 1, "only the alias binds a name");
    assert_eq!(unit.aliases[0].name, "NativeType");
    assert_eq!(unit.aliases[0].target, "System.Int32");
}

#[test]
fn address_of_generic_method_group_parses() {
    // `&AwaiterOnCompletedFromContinuation<TAwaiter>` — address of a
    // generic method group (function-pointer value): parses as
    // `AddrOf` over the bare method name, with no errors. Plain
    // `&x` keeps working (regression guard in the same unit).
    let src = r#"class C {
        public static unsafe void M() {
            P = &N<int>;
            Q = &x;
        }
    }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = class
        .members
        .iter()
        .find(|m| matches!(m, ClassMember::Method(md) if md.name == "M"))
        .expect("method M")
    else {
        panic!("expected method member")
    };
    let body = method.body.as_ref().expect("method body");
    assert_eq!(body.stmts.len(), 2, "both assignments must survive");
    let Stmt::Assign { value, .. } = &body.stmts[0] else {
        panic!("expected assignment, got {:?}", body.stmts[0])
    };
    assert!(
        matches!(value, Expr::Unary { op, inner, .. }
            if *op == crate::models::UnOp::AddrOf
                && matches!(inner.as_ref(), Expr::Ident { name, .. } if name == "N")),
        "method group must be AddrOf(bare name), got {value:?}"
    );
}

#[test]
fn function_pointer_local_with_ref_param_parses() {
    // `delegate*<ref byte, void> p = ...;` — `ref`/`in`/`out`
    // parameter modifiers are part of function-pointer signatures
    // (broke Array.CoreCLR's `constructorFtn` local).
    let src = r#"class C {
        public static unsafe void M() {
            delegate*<ref byte, void> p = GetFtn();
        }
    }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = class
        .members
        .iter()
        .find(|m| matches!(m, ClassMember::Method(md) if md.name == "M"))
        .expect("method M")
    else {
        panic!("expected method member")
    };
    let body = method.body.as_ref().expect("method body");
    let Stmt::Let { type_name, name, .. } = &body.stmts[0] else {
        panic!("expected local, got {:?}", body.stmts[0])
    };
    assert_eq!(type_name, "delegate*<ref byte, void>");
    assert_eq!(name, "p");
}

#[test]
fn using_static_directive_recorded() {
    // `using static System.Reflection.MethodBase;` records its target;
    // the members import into scope downstream (scan).
    let src = "using System;\nusing static System.Reflection.MethodBase;\nclass C {\n}\n";
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    assert_eq!(unit.static_usings, vec!["System.Reflection.MethodBase".to_string()]);
    assert!(unit.aliases.is_empty());
}

#[test]
fn var_tuple_deconstruction_parses() {
    // `var (sinPi, cosPi) = float.SinCosPi(x);` — each binding is
    // exposed (drained `Let`s); the initializer evaluates.
    // (`var` alone stays an identifier: `var = 5;` below.)
    let src = r#"class C {
        public void M() {
            var (sinPi, cosPi) = F(1);
            var _ = 0;
            int v = sinPi + cosPi;
        }
    }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = class
        .members
        .iter()
        .find(|m| matches!(m, ClassMember::Method(md) if md.name == "M"))
        .expect("method M")
    else {
        panic!("expected method member")
    };
    let body = method.body.as_ref().expect("method body");
    let lets: Vec<&str> = body
        .stmts
        .iter()
        .filter_map(|s| match s {
            Stmt::Let { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        lets.contains(&"sinPi") && lets.contains(&"cosPi"),
        "tuple bindings must be declared: {lets:?}"
    );
}

#[test]
fn ref_returning_delegate_parses() {
    // `delegate ref readonly TValue D(...)` — by-ref returns on
    // delegates (FrozenDictionary's AlternateLookupDelegate).
    let src = r#"class C {
        internal delegate ref readonly int Lookup(int key);
    }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let Member::Delegate(delegate) = class
        .members
        .iter()
        .find_map(|m| match m {
            ClassMember::Nested(n) => Some(n),
            _ => None,
        })
        .expect("delegate member")
    else {
        panic!("expected nested delegate")
    };
    assert_eq!(delegate.name, "Lookup");
}

#[test]
fn dotted_pointer_return_in_interface_parses() {
    // `ComWrappers.ComInterfaceEntry* M(out int c);` — dotted return
    // types with pointer suffixes in interfaces (static abstract DIM).
    let src = r#"namespace N {
        public unsafe interface IComExposedClass {
            public static abstract ComWrappers.ComInterfaceEntry* GetComInterfaceEntries(out int count);
        }
    }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Interface(iface) = &unit.namespaces[0].members[0] else {
        panic!("expected interface member")
    };
    assert!(
        iface.methods.iter().any(|m| m.name == "GetComInterfaceEntries"),
        "dotted-pointer DIM must survive"
    );
}

#[test]
fn ref_returning_interface_method_parses() {
    // `ref TCache? GetStorageRef(...)` — by-ref returns on
    // interface signatures (static abstract DIM members).
    let src = r#"namespace N {
        public interface ICache<T> {
            public static abstract ref T? GetStorageRef(int kind);
        }
    }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Interface(iface) = &unit.namespaces[0].members[0] else {
        panic!("expected interface member")
    };
    assert!(
        iface.methods.iter().any(|m| m.name == "GetStorageRef"),
        "ref-returning DIM must survive"
    );
}

#[test]
fn primary_ctor_params_recorded() {
    // `ref struct Holder(nint addr, nuint length)` — primary
    // constructor value params are recorded (not skipped).
    let src = r#"class C {
        internal unsafe ref struct Holder(nint addr, nuint length) {
            public void Free() {
            }
        }
    }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Nested(Member::Class(holder)) = &class.members[0] else {
        panic!("expected nested struct")
    };
    assert_eq!(holder.primary_ctor_params, vec!["addr".to_string(), "length".to_string()]);
}

#[test]
fn extern_modifier_recorded() {
    // `internal static extern string FastAllocateString(int length);`
    // — `extern` is recorded (not discarded); bodyless is expected.
    let src = r#"class C {
        internal static extern string FastAllocateString(int length);
        public static void Fine() {
        }
    }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = class
        .members
        .iter()
        .find(|m| matches!(m, ClassMember::Method(md) if md.name == "FastAllocateString"))
        .expect("method FastAllocateString")
    else {
        panic!("expected method member")
    };
    assert!(method.is_extern, "extern must be recorded");
    assert!(method.body.is_none(), "extern is bodyless");
}

#[test]
fn generic_return_operator_parses() {
    // `Complex<T> operator -(Complex<T> value)` — operators accept
    // generic return types (the gate used to require `Type operator`
    // with a bare identifier type). Explicit implementations qualify
    // the operator (`Iface<T>.operator +`).
    let src = r#"struct Complex<T> {
        public static Complex<T> operator -(Complex<T> value) { return value; }
        public static byte IAdditionOperators<byte, byte, byte>.operator +(byte left, byte right) { return left; }
    }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ops: Vec<&str> = class
        .members
        .iter()
        .filter_map(|m| match m {
            ClassMember::Method(method) => Some(method.name.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        ops.contains(&"Minus"),
        "generic-return operator must survive member parsing: {ops:?}"
    );
    assert!(
        ops.contains(&"Plus"),
        "explicit-interface operator must survive member parsing: {ops:?}"
    );
}

#[test]
fn delegate_where_constraints_parse() {
    // `delegate void M<T>(...) where T : struct;` — delegates accept
    // type constraints like every other declaration.
    let src = r#"public delegate void MeasurementCallback<T>(Instrument instrument, T measurement) where T : struct;"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Delegate(decl) = &unit.members[0] else {
        panic!("expected delegate member")
    };
    assert_eq!(decl.name, "MeasurementCallback");
}

#[test]
fn function_pointer_deref_invoke_parses() {
    // `(*(delegate* unmanaged[MemberFunction]<...>** )pThis)[3](...)`
    // — cast to pointer-to-function-pointer, deref, vtable index,
    // invoke. The trailing `**` must not break the cast gate (it used
    // to eat the first `*` as `)` and rewind the whole cast).
    let src = r#"class C { Main() {
        return (*(delegate* unmanaged[MemberFunction]<IntPtr, Guid*, IntPtr*, int>**)pThis)[3](pThis, guid);
    } }"#;
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn function_pointer_types_parse() {
    // `delegate* unmanaged[Cdecl]<...>` — function-pointer types in
    // field and parameter positions. Must not route to a nested
    // `delegate` declaration; records as an opaque `delegate*` type.
    let src = r#"class C {
        private static readonly delegate* unmanaged[Cdecl]<uint, void*, int> cb;
        int M(delegate*<int, int> f) { return 0; }
    }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Field(field) = &class.members[0] else {
        panic!("expected field member, got {:?}", class.members[0])
    };
    assert_eq!(field.type_name, "delegate*");
    let ClassMember::Method(method) = &class.members[1] else {
        panic!("expected method member")
    };
    assert_eq!(method.params[0].type_name, "delegate*");
}

#[test]
fn nullable_generic_param_keeps_type_argument() {
    // `Predicate<X509Certificate2>?` must keep its argument for
    // overload ranking — degrading to `Predicate?` loses the arity
    // and the argument type (whitespace inside `<...>` normalizes
    // away so spellings compare equal site-independently).
    let src = r#"class C {
        void M(Predicate<C> a, Predicate<C>? b, System.Func<C, int> c) { }
    }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member, got {:?}", class.members[0])
    };
    let types: Vec<&str> = method.params.iter().map(|p| p.type_name.as_str()).collect();
    assert_eq!(
        types,
        vec!["Predicate<C>", "Predicate<C>?", "System.Func<C,int>"],
        "generic arguments must survive, whitespace-free: {types:?}"
    );
}

#[test]
fn attributed_type_parameters_parse() {    // `M<[Attr] T>()` — attributes on type parameter declarations
    // are contracts only; the names still record.
    let src = r#"class C { public static T CreateInstance<[Attr] T>() where T : struct { return null; } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    assert_eq!(method.generic_params, vec!["T".to_string()]);
}

#[test]
fn tuple_return_local_function_parses() {
    // `static (A, B) Name(...)` — local functions accept tuple return
    // types (named or unnamed elements). The gate must not misread
    // them as parenthesized expressions.
    for src in [
        r#"class C { Main() { static (int, string) M(int x) { return null; } return 0; } }"#,
        r#"class C { Main() { static (SslProtocols a, SslProtocols b) M(SslProtocols p) { return null; } return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn array_return_local_function_parses() {
    // `HeaderEntry[] Local() { ... }` — local functions accept array,
    // nullable, and pointer return-type suffixes (previously only the
    // bare/generic forms were recognized, so this fell into the array
    // declaration path and errored on the `(`).
    let src = r#"class C { Main() { HeaderEntry[] Local() { return null; } string? S() { return null; } return 0; } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    let Some(block) = &method.body else {
        panic!("expected method body")
    };
    assert!(
        block.stmts.iter().any(|s| matches!(
            s,
            Stmt::Let { name, .. } if name == "Local"
        )),
        "array-return local function name must be exposed to the enclosing scope"
    );
}

#[test]
fn global_qualified_name_is_not_a_label() {
    // `global::Ns.T x = ...` — the statement-level goto-label check
    // (`Ident :`) must exclude `::`, or it eats `global` + the first
    // colon and the declaration misparses ("found Colon").
    for src in [
        r#"class C { Main() { global::System.Foo x = 1; return 0; } }"#,
        r#"class C { Main() { global::System.Runtime.InteropServices.Marshalling.SafeHandleMarshaller<global::Microsoft.Win32.SafeHandles.SafeX509ChainHandle>.ManagedToUnmanagedOut m = new(); return 0; } }"#,
        r#"class C { Main() { done: x = 1; return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn generic_return_indexer_parses() {
    // `IEnumerable<T> this[K key] => ...` — indexers accept generic
    // return types. Regression: the gate only saw `Type this`,
    // so `Type<T> this` fell to method parsing ("found 'Less'").
    let src = r#"class C {
        public IEnumerable<TElement> this[TKey key] => GetGrouping(key);
        public int this[int i] { get; }
    }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let indexers = class
        .members
        .iter()
        .filter(|m| matches!(m, ClassMember::Indexer(_)))
        .count();
    assert_eq!(indexers, 2);
}

#[test]
fn stray_semicolon_is_empty_statement() {
    // `foo(); ;` — a lone `;` is a legal C# empty statement (no-op),
    // never an error.
    let src = r#"class C { Main() { Foo(); ; return 0; } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    let Some(block) = &method.body else {
        panic!("expected method body")
    };
    assert_eq!(block.stmts.len(), 3);
}

#[test]
fn access_modifier_before_tuple_type_parses() {
    // `private (int A, bool B) M(...)` — the access modifier gate must
    // accept `(` (tuple-typed member) as a member start; otherwise the
    // modifier is left unconsumed and method parsing fails on it.
    let src = r#"class C { private (int A, bool B) M(int x) { return null; } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    assert!(
        class.members.iter().any(|m| matches!(
            m,
            ClassMember::Method(method) if method.name == "M"
        )),
        "tuple-return method must survive member parsing"
    );
}

#[test]
fn foreach_deconstruction_no_outer_variable() {
    // `foreach ((T a, U b) in expr)` and `foreach (var (a, b) in expr)`
    // — C# deconstruction has NO outer variable. Regression: the
    // parser required one (`(...) x in`), failing real code with
    // "expected identifier as foreach variable, found 'in'".
    for src in [
        r#"class C { Main() { foreach ((int a, string b) in items) { } } }"#,
        r#"class C { Main() { foreach ((Assembly _, MetadataReaderProvider? provider) in cache) { } } }"#,
        r#"class C { Main() { foreach ((bool _, T Priority) item in items) { } } }"#,
        r#"class C { Main() { foreach (var (a, b) in items) { } } }"#,
        r#"class C { Main() { foreach (int x in items) { } } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn switch_arm_nameof_pattern_parses() {
    // `nameof(X) => ...` — `nameof` lexes as a keyword, so it needs
    // its own arm-label path (constant pattern).
    let src = r#"class C { Main() { int r = x switch { nameof(IsX) => 1, _ => 2, }; return r; } }"#;
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn switch_arm_default_literal_parses() {
    // `default(char) => ...` — typed default literal in
    // switch-expression arms (mirrors `primary`).
    let src = r#"class C { Main() { int r = x switch { 'O' => 1, default(char) => 2, _ => 3, }; return r; } }"#;
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn switch_arm_negated_pattern_parses() {
    // `not null => ...` — negated patterns in switch-expression arms
    // (mirrors statement-level `case` labels, which already handle it).
    for src in [
        r#"class C { Main() { int r = x switch { null => 0, not null => 1, }; return r; } }"#,
        r#"class C { Main() { int r = x switch { not 1 => 0, _ => 1, }; return r; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn switch_arm_negative_constant_parses() {
    // `-1 => ...` — the sign is part of the constant pattern, not a
    // binary operator.
    let src = r#"class C { Main() { int r = x switch { -1 => 10, 0 => 20, _ => 30, }; return r; } }"#;
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn switch_arm_generic_type_pattern_parses() {
    // `KeyValuePair<K, V> t => ...` — generic type patterns in
    // switch-expression arms. The `<...>` must be claimed as generics
    // (balanced + type-ish follower), never as less-than for the
    // binary fold; `a < b =>` stays a comparison.
    for src in [
        r#"class C { Main() { int r = x switch { KeyValuePair<IPAddress, AddressFamily> t => 1, _ => 2, }; return r; } }"#,
        r#"class C { Main() { int r = x switch { A.B<C.D> y => 1, _ => 2, }; return r; } }"#,
        r#"class C { Main() { int r = x switch { a < b => 1, _ => 2, }; return r; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn switch_arm_cast_pattern_parses() {
    // `(int)X.Y => ...` — a cast in arm position is not a positional
    // pattern. Mirrors the `case`-label and `is`-pattern handling.
    for src in [
        r#"class C { Main() { int r = x switch { (int)E.Code => 1, _ => 2, }; return r; } }"#,
        r#"class C { Main() { int r = x switch { (int)E.Code when c => 1, _ => 2, }; return r; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn switch_arm_binary_constant_pattern_parses() {
    // `MethodDef >> RowIdBitCount => ...` — constant-expression
    // patterns with binary operators. The fold stops at arm
    // terminators, so `or`/`and`/`when`/`,`/`=>` keep working.
    for src in [
        r#"class C { Main() { int r = x switch { A >> B => 1, _ => 2, }; return r; } }"#,
        r#"class C { Main() { int r = x switch { A.B >> C.D + E => 1, _ => 2, }; return r; } }"#,
        r#"class C { Main() { int r = x switch { A >> B or C << D => 1, _ => 2, }; return r; } }"#,
        r#"class C { Main() { int r = x switch { A >> B when c => 1, _ => 2, }; return r; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn switch_arm_bare_name_when_guard_parses() {
    // `CONST when guard => body` — a bare `name =>` guard must not
    // parse as a single-param lambda eating the arm's `=>`.
    for src in [
        r#"class C { Main() { int r = x switch { A.B when c => 1, _ => 2, }; return r; } }"#,
        r#"class C { Main() { int r = x switch { A.B when c && d => 1, _ => 2, }; return r; } }"#,
        r#"class C { Main() { int r = x switch { A.B when Foo(v => v > 0) => 1, _ => 2, }; return r; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn ref_readonly_parameters_parse() {
    // `ref readonly BigInteger left` — ref-readonly parameters (the
    // by-ref-ness is inert here, as for all parameter modifiers).
    let src = r#"class C { static BigInteger BitwiseAnd(ref readonly BigInteger left, ref readonly BigInteger right) { return null; } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    assert_eq!(method.params.len(), 2);
}

#[test]
fn range_expressions_parse() {
    // `..^1`, `0..^1`, `2..` in expression position (arguments,
    // initializers). Slicing has no runtime model: the informative
    // side is recorded, the rest dropped — but it must parse.
    for src in [
        r#"class C { Main() { var a = AsSpan(..^1); return 0; } }"#,
        r#"class C { Main() { var b = x[0..^1]; return 0; } }"#,
        r#"class C { Main() { var c = y[2..]; return 0; } }"#,
        r#"class C { Main() { var d = z[..]; return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn generic_call_tuple_type_arguments_parse() {
    // `Array.Empty<(TElement, TPriority)>()` — tuple types inside
    // generic call argument lists. The `is_generic_type_args` gate
    // and both call-site gates must accept `(` after `<`.
    let src = r#"class C { Main() { _nodes = Array.Empty<(TElement, TPriority)>(); return 0; } }"#;
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn spaced_ternary_vs_adjacent_null_conditional() {
    // `a ? [y] : x` is a ternary with a collection branch; `a?[y]`
    // (adjacent) is a null-conditional index. Same for `?.` vs `? .`.
    // Regression: the postfix grab took both, dropping the statement
    // (member bodies errored, statements went silent).
    for src in [
        r#"class C { Main() { var v = a ? [y] : x; return 0; } }"#,
        r#"class C { int[] P => a ? [y] : x; }"#,
        r#"class C { Main() { var w = a?[y]; return 0; } }"#,
        r#"class C { Main() { var u = a?.M; return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
    // Shape check: spaced form is a ternary, adjacent an index.
    let unit = parser::parse(r#"class C { Main() { var v = a ? [y] : x; var w = a?[y]; return 0; } }"#).unwrap();
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    let Some(block) = &method.body else {
        panic!("expected method body")
    };
    let Stmt::Let { init: tern, .. } = &block.stmts[0] else {
        panic!("expected let statement")
    };
    assert!(
        matches!(tern, Expr::Ternary { .. }),
        "spaced form must be a ternary, got {tern:?}"
    );
    let Stmt::Let { init: idx, .. } = &block.stmts[1] else {
        panic!("expected let statement")
    };
    assert!(
        matches!(idx, Expr::Ternary { .. }),
        "adjacent form must be a null-guarded index (ternary), got {idx:?}"
    );
}

#[test]
fn unary_plus_desugars_to_operand() {
    // `? +1 : 0` — C# unary plus is numeric identity. Desugars to
    // the operand with no new AST surface.
    let src = r#"class C { Main() { int x = b ? +1 : 0; return x; } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    let Some(block) = &method.body else {
        panic!("expected method body")
    };
    let Stmt::Let { init, .. } = &block.stmts[0] else {
        panic!("expected let statement")
    };
    let Expr::Ternary { then_expr, .. } = init else {
        panic!("expected ternary initializer, got {init:?}")
    };
    assert!(
        matches!(then_expr.as_ref(), Expr::Literal(Literal::Int(1))),
        "unary plus must desugar to its operand, got {then_expr:?}"
    );
}

#[test]
fn stackalloc_generic_pointer_element_parses() {
    // `stackalloc Vector128<float>*[3] { ... }` — generic element
    // types with pointer suffixes.
    let src = r#"class C { Main() { var v = stackalloc Vector128<float>*[3] { null, }; return 0; } }"#;
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn stackalloc_tuple_element_parses() {
    // `stackalloc (int, int)[N]` — the size brackets belong to the
    // allocation, not the type. Regression: shape validation ran
    // past suffixes and consumed `[N]`, leaving nothing to size.
    let src = r#"class C { Main() { var s = stackalloc (int, int)[16]; return 0; } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    let Some(block) = &method.body else {
        panic!("expected method body")
    };
    assert!(
        matches!(&block.stmts[0], Stmt::Let { init: Expr::NewArray { .. }, .. }),
        "tuple stackalloc must lower to NewArray"
    );
}

#[test]
fn stackalloc_sized_with_initializer_parses() {
    // `stackalloc T[N] { ... }` — a size together with an element
    // list (regression: the `{` misparsed as a block start,
    // "missing ';' after statement").
    let src = r#"class C { Main() { Interop.Sys.T* p = stackalloc Interop.Sys.T[4] { Interop.Sys.T.A, }; return 0; } }"#;
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn stackalloc_dotted_and_initializer_forms_parse() {
    // `stackalloc Interop.Sys.IOVector[n]` (dotted element type) and
    // `stackalloc int[] { ... }` (inferred-size initializer).
    // Regression: only bare `T[n]` parsed ("found identifier
    // 'Interop'").
    for src in [
        r#"class C { Main() { var a = flag ? stackalloc GCHandle[4] : new GCHandle[4]; return 0; } }"#,
        r#"class C { Main() { var b = flag ? stackalloc Interop.Sys.IOVector[4] : new Interop.Sys.IOVector[4]; return 0; } }"#,
        r#"class C { Main() { int[] c = stackalloc int[] { 1, 2 }; return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn sized_array_with_initializer_parses() {
    // `new byte[1] { 0 }` — C# allows a size together with an
    // initializer. The braces must be consumed, not misparsed as a
    // block (regression: "expected ';' after statement, found '{'").
    let src = r#"class C { static byte[] E = new byte[1] { 0 }; Main() { int[] a = new int[2] { 1, 2 }; return 0; } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    assert!(
        class.members.iter().any(|m| matches!(
            m,
            ClassMember::Field(f) if f.name == "E"
        )),
        "sized+initialized field must survive member parsing"
    );
}

#[test]
fn named_tuple_literal_elements_parse() {
    // `(thisRef: this, x, isAck: 1)` — tuple element names are metadata;
    // only values matter. Previously the `:` broke tuple parsing and the
    // enclosing call misparsed (fallthrough cascade).
    let src = r#"class C { Main() { var t = (thisRef: this, x, isAck: 1); return 0; } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    let Some(block) = &method.body else {
        panic!("expected method body")
    };
    assert!(
        matches!(&block.stmts[0], Stmt::Let { name, .. } if name == "t"),
        "tuple literal must parse as an initializer"
    );
}

#[test]
fn weak_tuple_declaration_and_ref_prefix() {
    // `(A, B)[] x = ...` (weak-unnamed, follower-validated) and
    // `ref (A, B) x = ref ...` (ref prefix, inert). An argument list
    // can never be followed by `name =`, so both are unambiguous.
    for src in [
        r#"class C { Main() { (X509Certificate2, int)[] elements = Foo(); return 0; } }"#,
        r#"class C { Main() { ref (X509Certificate2, int) currentValue = ref elements[i]; return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn unnamed_tuple_creation_and_arg_lists() {
    // `new (HeaderDescriptor, byte[])[n]` — unnamed tuple type (one
    // primitive/complex element required). Plain `(a, b)` stays an
    // argument list, never a tuple.
    let src = r#"class C { Main() {
        var t = new (HeaderDescriptor, byte[])[3];
        var u = Foo(a, b);
        return 0;
    } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    let Some(block) = &method.body else {
        panic!("expected method body")
    };
    let Stmt::Let { init, .. } = &block.stmts[0] else {
        panic!("expected let statement")
    };
    assert!(
        matches!(init, Expr::NewArray { .. }),
        "unnamed tuple creation must lower to NewArray, got {init:?}"
    );
    let Stmt::Let { init: init2, .. } = &block.stmts[1] else {
        panic!("expected let statement")
    };
    assert!(
        !matches!(init2, Expr::NewArray { .. }),
        "plain (a, b) must stay a call argument list, got {init2:?}"
    );
}

#[test]
fn tuple_typed_parameters_parse() {
    // `Compare((int, int) x, (int, int) y)` — tuple types in
    // parameter position fold like tuple-typed members.
    let src = r#"class C { public int Compare((int, int) x, (int, int) y) { return 0; } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    assert_eq!(method.params.len(), 2);
}

#[test]
fn safe_modifier_not_a_keyword() {
    // `public safe object? Object;` — `safe` is a contextual member
    // modifier (safe-critical annotations), deliberately not a lexer
    // keyword: a variable may still be named `safe`.
    let src = r#"struct P { public safe object? Object; } class C { Main() { bool safe = true; return 0; } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(s) = &unit.members[0] else {
        panic!("expected struct member")
    };
    assert!(
        s.members.iter().any(|m| matches!(
            m,
            ClassMember::Field(f) if f.name == "Object"
        )),
        "safe-modified field must survive member parsing"
    );
}

#[test]
fn extension_block_parses() {
    // `extension<T>(Vector<T>) where ... { members }` (C# 14) parses
    // members into a synthetic nested class; a member genuinely named
    // `extension` (including `extension()` with an empty receiver)
    // still takes the method path.
    let src = r#"class C {
        extension<T>(Vector<T>) where T : IFloatingPoint<T> {
            public static Vector<T> Pi {
                get;
            }
        }
        void extension() {
        }
    }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    assert!(
        class.members.iter().any(|m| matches!(
            m,
            ClassMember::Nested(n) if n.name() == "Extension_VectorT"
        )),
        "extension block must become a synthetic nested class"
    );
    assert!(
        class.members.iter().any(|m| matches!(
            m,
            ClassMember::Method(method) if method.name == "extension"
        )),
        "method named extension must keep the method path"
    );
}

#[test]
fn plain_custom_tuple_member_parses() {
    // `(SocketOptionName, SocketOptionLevel) ToSocketOptions(...)` —
    // plain custom types need neither element names nor
    // primitive/complex elements at member position (an argument
    // list is impossible there). The strict rules still guard
    // expression positions (`new (a, b)` stays target-typed).
    let src = r#"class C {
        private static (SocketOptionName, SocketOptionLevel) ToSocketOptions(TrackableSocketOptions options) => options switch { _ => null, };
    }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    assert!(
        class.members.iter().any(|m| matches!(
            m,
            ClassMember::Method(method) if method.name == "ToSocketOptions"
        )),
        "plain-tuple-return method must survive member parsing"
    );
}

#[test]
fn tuple_typed_member_and_creation_parse() {
    // `(T1 n1, T2 n2)[] Prop { get; }` — tuple-typed members fold the
    // tuple into a single type token; `new (T n)[] { ... }` creates an
    // implicit tuple array. Neither is an argument list.
    let src = r#"class C {
        (int a, string b)[] Prop { get; }
        Main() { var x = new (int a, string b)[] { (1, "x") }; return 0; }
    }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    assert!(
        class.members.iter().any(|m| matches!(
            m,
            ClassMember::Property(p) if p.name == "Prop"
        )),
        "tuple-typed property must survive member parsing"
    );
}

#[test]
fn extern_local_function_parses() {
    // `[Attr] [return: Attr] static extern object M(...);` — extern
    // locals are bodyless `[UnsafeAccessor]` declarations (no body,
    // just `;`). The `extern` modifier must gate and consume like the
    // other local-function modifiers.
    let src = r#"class C { Main() {
        [UnsafeAccessor(UnsafeAccessorKind.Method, Name = "get_Settings")]
        [return: UnsafeAccessorType(T)]
        static extern object GetSettings(SocketsHttpHandler handler);
        return 0;
    } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    let Some(block) = &method.body else {
        panic!("expected method body")
    };
    assert!(
        block.stmts.iter().any(|s| matches!(
            s,
            Stmt::Let { name, .. } if name == "GetSettings"
        )),
        "extern local function name must be exposed to the enclosing scope"
    );
}

#[test]
fn attributed_local_function_parses() {
    // `[MethodImpl(...)] static int F() => ...` — attributes precede
    // the local function; they are consumed, not parsed as an index
    // expression statement.
    let src = r#"class C { Main() { [Attr] static int F(int x) { return x; } return F(1); } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    let Some(block) = &method.body else {
        panic!("expected method body")
    };
    assert!(
        block.stmts.iter().any(|s| matches!(
            s,
            Stmt::Let { name, .. } if name == "F"
        )),
        "attributed local function name must be exposed to the enclosing scope"
    );
}

#[test]
fn ref_return_local_function_parses() {
    // `ref object? Local(...)` — local functions accept `ref` /
    // `readonly` return modifiers (by-ref-ness is inert here).
    let src = r#"class C { Main() { ref object? Local(int i) { return null; } return 0; } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    let Some(block) = &method.body else {
        panic!("expected method body")
    };
    assert!(
        block.stmts.iter().any(|s| matches!(
            s,
            Stmt::Let { name, .. } if name == "Local"
        )),
        "ref-return local function name must be exposed to the enclosing scope"
    );
}

#[test]
fn local_function_where_constraints_parse() {
    // `... M<T>(...) where T : C { ... }` — local functions accept
    // type constraints like every other declaration.
    let src = r#"class C { Main() { static R<T>? M<T>(T s, T e) where T : INumber<T> { return null; } return 0; } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    let Some(block) = &method.body else {
        panic!("expected method body")
    };
    assert!(
        block.stmts.iter().any(|s| matches!(
            s,
            Stmt::Let { name, .. } if name == "M"
        )),
        "constrained local function name must be exposed to the enclosing scope"
    );
}

#[test]
fn async_generic_local_function_parses() {
    // `async ValueTask<int> Core(...) { ... }` — local functions accept
    // `async` (like `static`) and generic return types.
    let src = r#"class C { Main() { async ValueTask<int> Core(int x) { return x; } return Core(1); } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    let Some(block) = &method.body else {
        panic!("expected method body")
    };
    assert!(
        block.stmts.iter().any(|s| matches!(
            s,
            Stmt::Let { name, .. } if name == "Core"
        )),
        "local function name must be exposed to the enclosing scope"
    );
}

#[test]
fn out_arg_tuple_type_parses() {
    // `out (long, long) fileId` — tuple types in out/ref/in
    // arguments (shared `consume_tuple_type` helper with the
    // stackalloc arm). Groupings (`out (x)`) still rewind.
    for src in [
        r#"class C { Main() { if (!F(file, out (long, long) fileId)) { return 1; } return 0; } }"#,
        r#"class C { Main() { F(out (x)); return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn out_arg_contextual_keyword_name_parses() {
    // `out DnsRecord record` — the declared name may be a contextual
    // keyword. The trial type-skip and the pre-declaration must both
    // accept it (regression: "found 'Record_'" from both checks).
    for src in [
        r#"class C { Main() { if (!r.Try(out DnsRecord record)) { return 1; } return 0; } }"#,
        r#"class C { Main() { if (!r.Try(out int x)) { return 1; } return 0; } }"#,
        r#"class C { Main() { r.Get(out helper._f); return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn expression_coalesce_assign_desugars() {
    // `return h ??= v;` — compound assignment in expression position
    // desugars exactly like statement position (`h = h ?? v`).
    let src = r#"class C { Main() { x = y ??= z; } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    let Some(block) = &method.body else {
        panic!("expected method body")
    };
    let Stmt::Assign { name, value, .. } = &block.stmts[0] else {
        panic!("expected assign statement")
    };
    assert_eq!(name, "x");
    // Bare-name targets desugar to plain `Assign` (lowering stores locals
    // and implicit-`this` fields through that path); only complex targets
    // become `AssignComplex`.
    let Expr::Assign { target: inner_t, value: inner, .. } = value else {
        panic!("expected Assign rhs, got {value:?}")
    };
    assert!(matches!(inner_t.as_ref(), Expr::Ident { name, .. } if name == "y"));
    assert!(matches!(
        inner.as_ref(),
        Expr::Binary { op: crate::models::BinOp::NullCoalesce, .. }
    ));
}

#[test]
fn tuple_deconstruction_with_suffixed_types_parses() {
    // `(Http3FrameType? frameType, long payloadLength) = expr;` —
    // tuple deconstruction where element types carry `?`/generics.
    // Regression: the gate only accepted bare `Type Name` pairs, so
    // the `?` fell through to expression parsing ("expected ':'").
    for src in [
        r#"class C { Main() { (int a, long b) = Foo(); return 0; } }"#,
        r#"class C { Main() { (Http3FrameType? frameType, long payloadLength) = Foo(); return 0; } }"#,
        r#"class C { Main() { (List<int> xs, string s) = Foo(); return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
    // Negative controls: argument lists and groupings must NOT match.
    for src in [
        r#"class C { Main() { Foo(a, b); return 0; } }"#,
        r#"class C { Main() { var x = (a + b); return x; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn multi_declarator_fields_parse() {
    // `string? a, b, c;` — multi-declarator fields share one type;
    // each declarator becomes its own member. Regression: the field
    // gate only accepted `;`/`=` after the name, so the `,` fell
    // through to method parsing ("expected '('").
    let src = r#"class C {
        private readonly string? _a, _b, _c;
        private int x = 1, y = 2;
    }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let Member::Class(class) = &unit.members[0] else {
        panic!("expected class member")
    };
    let fields: Vec<&str> = class
        .members
        .iter()
        .filter_map(|m| match m {
            ClassMember::Field(f) => Some(f.name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(fields, vec!["_a", "_b", "_c", "x", "y"]);
}

#[test]
fn tuple_cast_with_operand_parses() {
    // `(int, Token[])x` and `((int, Token[]))y!` — a tuple type whose
    // `)` is followed by an operand is a cast, never a literal
    // (`(a, b)`, `(a, b).M` keep the literal path).
    for src in [
        r#"class C { Main() { var v = (int, Token[])x; return 0; } }"#,
        r#"class C { Main() { var v = ((int, Token[]))state!; return 0; } }"#,
        r#"class C { Main() { var t = (a, b); var u = (a, b).Item1; return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn mixed_suffix_casts_parse() {
    // `(object?[]?)x`, `(A.B[]?)x` — nullable/pointer/array suffixes
    // combine in any order on every cast gate arm (plain, pointer,
    // generic, dotted). Each arm runs its shapes in a loop now.
    for src in [
        r#"class C { Main() { var a = (object?[]?)Foo(); return 0; } }"#,
        r#"class C { Main() { var b = (A.B[]?)Foo(); return 0; } }"#,
        r#"class C { Main() { var c = (T*[]?)Foo(); return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn generic_array_nullable_cast_parses() {
    // `(KeyValuePair<K, V>[]?)expr` — suffixes after a generic list
    // combine in any order. The gate ran each shape once, so `[]?`
    // fell through to grouping.
    let src = r#"class C { Main() { var a = (KeyValuePair<TKey, TValue>[]?)Foo(); return 0; } }"#;
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn generic_nullable_cast_parses() {
    // `(IEnumerable<Claim>?)null`, `(string?)null`, `(IIdentity?)null`
    // — casts whose generic type carries a nullable suffix (common in
    // `: this(...)` chained-ctor argument lists).
    for src in [
        r#"class C { C() : this((IIdentity?)null) { } }"#,
        r#"class C { C() : this((IEnumerable<Claim>?)null, (string?)null) { } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn is_pattern_cast_operand_parses() {
    // `is not (byte)'\t' and not (byte)' '` — a parenthesized single
    // type in an `is`-pattern is a cast, not a positional pattern.
    // Regression: `(byte)` was skipped as a pattern and the char
    // literal broke the enclosing `if` ("found integer 9").
    for src in [
        r#"class C { Main() { if (x is (byte)'a') { return 1; } return 0; } }"#,
        r#"class C { Main() { if (x is not (byte)'\t' and not (byte)' ') { return 1; } return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn async_lambda_forms_parse() {
    // `async x =>`, `async (a, b) =>`, `static async (x) =>` — the
    // async modifier must not break lambda detection (regression:
    // `ContinueWith(async closeTask =>` failed with "expected ')'").
    for src in [
        r#"class C { Main() { var t = Foo(async x => 1); return 0; } }"#,
        r#"class C { Main() { var t = Foo(async (a, b) => 1); return 0; } }"#,
        r#"class C { Main() { var t = Foo(static async (x) => 1); return 0; } }"#,
        r#"class C { Main() { var t = Foo(async static (x) => 1); return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn fixed_buffer_const_size_parses() {
    // `internal fixed byte Address[MAX_IP_ADDRESS_BYTES];` — fixed-size
    // buffers with const-expression sizes (Interop structs), not just
    // integer literals. Regression: the gate demanded IntLiteral, so
    // this fell through to method parsing ("expected '(' ... found
    // identifier 'Address'").
    let src = r#"class C { internal fixed byte Address[MAX_IP_ADDRESS_BYTES]; int Main() { return 0; } }"#;
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn closed_modifier_parses() {
    // `private closed class TraditionalKem` (C# 14 closed hierarchies):
    // `closed` is contextual — consumed as a modifier only before a
    // type keyword, while a variable named `closed` still parses.
    let src = r#"namespace t;
closed class Top {
}
class Outer {
    private closed class Inner {
    }
}
class User {
    int M() { int closed = 1; return closed; }
    int main() { return 0; }
}"#;
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn is_pattern_contextual_keyword_binding_parses() {
    // `record` lexes as Keyword::Record_, not an identifier — but it is
    // a valid designation, parameter, and local name
    // (System.Formats.Nrbf: `value is SerializationRecord record`,
    // `Add(SerializationRecord record)`, `out SerializationRecord?
    // record`). Regression: the pattern-binding checks demanded
    // Identifier, so the `if` lost its `)` ("expected ')' to close if
    // condition, found 'Record_'") and the rest of the file cascaded.
    for src in [
        r#"class C { int Main() { if (value is SerializationRecord record) { return 1; } return 0; } }"#,
        r#"class C { void Add(SerializationRecord record) { return; } int Main() { return 0; } }"#,
        r#"class C { Main() { switch (x) { case SerializationRecord record: return 1; default: return 0; } } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn cast_generic_nested_type_parses() {
    // `(Dictionary<TKey, TValue>.Enumerator)state.Current` — a cast
    // whose type is generic AND dot-qualified (nested type), from
    // System.Text.Json's DictionaryOfTKeyTValueConverter. Regression:
    // the cast gate's generic arm demanded `)` right after the type
    // argument list, so the parens misparsed as a grouping
    // ("expected ';' after assignment, found identifier 'state'").
    for src in [
        r#"class C { int Main() { e = (Dictionary<string, int>.Enumerator)x.Current; return 0; } }"#,
        r#"class C { int Main() { e = (System.Collections.Generic.Dictionary<string, int>.Enumerator)x; return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn await_foreach_parses() {
    // `await foreach (TValue item in ...)` — async enumeration, from
    // System.Text.Json's JsonSerializer.Write.Pipe. Regression: the
    // statement dispatch only knew `await using`, so `await foreach`
    // fell into expression parsing ("expected an expression, found
    // Keyword(Foreach)") and the enclosing try/catch cascaded.
    let src = r#"class C { async void M() { await foreach (TValue item in value.WithCancellation(cancellationToken)) { Write(item); } } int Main() { return 0; } }"#;
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn typeof_tuple_type_parses() {
    // `typeof((TCollection, TKey, TValue))` — a tuple *type* as a
    // typeof argument (System.Text.Json immutable-dictionary cache
    // keys). Regression: the typeof branch demanded a plain type
    // name ("expected identifier as type, found '('") and the call
    // args cascaded.
    for src in [
        r#"class C { int Main() { var t = typeof((TCollection, TKey, TValue)); return 0; } }"#,
        r#"class C { int Main() { var t = typeof((TCollection, TElement)); return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn nested_named_tuple_decl_parses() {
    // `(RegexNode LoopNode, (char Char, string? String, ...) Literal)?
    // x = ...` — a nullable named-tuple local whose element is itself
    // a named tuple (System.Text.RegularExpressions
    // RegexFindOptimizations). Regression: the tuple shape scanners
    // rejected a `(`-starting element, so the declaration fell through
    // to expression parsing ("expected ')' to close tuple literal,
    // found identifier 'LoopNode'").
    for src in [
        r#"class C { int Main() { (RegexNode LoopNode, (char Char, string? String) Literal)? x = F(root); return 0; } }"#,
        r#"class C { int Main() { (RegexNode loopNode, (char Char, string? String) literal) = G(); return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn grouped_cast_member_forgiving_call_parses() {
    // `((JsonTypeInfo<T>)jsonTypeInfo).SerializeHandler!(writer, value);`
    // — a grouped cast receiver with a null-forgiving member call
    // (System.Text.Json JsonMetadataServicesConverter). Regression:
    // the nested-tuple shape work made `((T)x)` match the 1-element
    // tuple-decl shape, so the statement misparsed as a declaration
    // ("expected in declaration, found 'Dot'").
    let src = r#"class C { int Main() { ((JsonTypeInfo<T>)jsonTypeInfo).SerializeHandler!(writer, value); return 0; } }"#;
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn out_suffixed_tuple_type_parses() {
    // `out (uint, uint)[]? ranges` — an inline out-var whose type is a
    // suffixed tuple (System.Text.RegularExpressions BDDRangeConverter).
    // Regression: the out-var path left tuple suffixes unconsumed, so
    // the `[` failed the name check and the `?` misparsed as a ternary
    // ("expected ':' to separate ternary branches, found ')'").
    let src = r#"class C { int Main() { if (!_rangeCache.TryGetValue(set, out (uint, uint)[]? ranges)) { return 1; } return 0; } }"#;
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn array_multi_declarator_parses() {
    // `XPathNode[] pageTemp, pageNew, pageCopy;` — trailing
    // declarators share the array element type
    // (System.Private.Xml XPathDocumentBuilder). Regression: the
    // array fast path demanded `;` after the first declarator
    // ("expected ';' after array declaration, found ','").
    // The nullable shape (`T[]? a, b;`) rides the general path.
    for src in [
        r#"class C { int Main() { XPathNode[] pageTemp, pageNew, pageCopy; return 0; } }"#,
        r#"class C { int Main() { XPathNode[] a = null, b; return 0; } }"#,
        r#"class C { int Main() { XPathNode[]? pageOverride, pageOrig; return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn nested_switch_arm_never_falls_through() {
    // An outer arm whose body is a switch that never falls through
    // (every arm jumps, `default` present) satisfies the no-fallthrough
    // rule (System.Private.Xml XmlSubtreeReader: `case
    // S.ClearNsAttributes:` bodies are nested switches). Regression:
    // `Stmt::Switch` had no case in the terminator check, so the arm
    // was reported at the NEXT case label ("switch arm must end with
    // 'break;'").
    for src in [
        r#"enum S { A, B } enum T { X, Y } class C { int M(S s, T t) { switch (s) { case S.A: switch (t) { case T.X: throw new InvalidOperationException(); case T.Y: return 1; default: return 0; } case S.B: return 2; default: return 0; } } int Main() { return 0; } }"#,
        r#"enum S { A, B } enum T { X, Y } class C { int M(S s, T t) { switch (s) { case S.A: switch (t) { case T.X: return 1; default: return 0; } case S.B: _x = 1; goto case S.A; default: return 0; } } int _x; int Main() { return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn interp_escaped_braces_in_call_parses() {
    // `WriteLine($"namespace {n} {{")` — escaped closing braces
    // inside an interpolated call argument (System.Private.Xml
    // Compilation). The lexer opened a phantom hole and swallowed
    // the file ("expected ')' ... found end of file").
    let src = r#"class C { void M(object w, string n) { w.WriteLine($"namespace {n} {{"); } int Main() { return 0; } }"#;
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn dotted_indexer_return_type_parses() {
    // `public Ns.Type? this[int index] { ... }` — an indexer whose
    // return type is dotted/nullable (reference-assembly stubs,
    // e.g. `AuthorizationRule? this[int index]`). Regression: the
    // indexer gate skipped generics/suffixes but not dots, so the
    // member fell into method parsing ("expected identifier as
    // method name, found 'Dot'") and the class cascaded.
    for src in [
        r#"class C { public Ns.Type? this[int index] { get { throw null; } } int Main() { return 0; } }"#,
        r#"class C { public System.Security.AccessControl.AuthorizationRule? this[int index] { get { throw null; } set { } } int Main() { return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn global_qualifier_cast_parses() {
    // `((global::System.Xml.XmlQualifiedName)o)` — a cast with a
    // transparent `global::` qualifier (generated serializers,
    // System.Private.Xml). Regression: the cast gate saw `(` ident
    // `:` and fell into grouping ("expected ')' ... found identifier
    // 'System'").
    for src in [
        r#"class C { int Main() { object x = ((global::System.Xml.XmlQualifiedName)o); return 0; } }"#,
        r#"class C { void M(global::System.Object o) { global::System.Object p = o; } int Main() { return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn for_header_accepts_member_expression_init_and_step() {
    // `for (_bufferIndex.Value++; _bufferIndex.Value < _buffer.Count; _bufferIndex.Value++)`
    // (Linq `TakeOrSkipQueryOperator`): for-init/step accept full
    // expression statements — member/indexer/call targets, not just
    // bare identifiers. Regression: "expected ';' after for-init,
    // found 'Dot'".
    for src in [
        r#"class C { bool M(Buf b, int n) { for (b.Value++; b.Value < n; b.Value++) { } return true; } int Main() { return 0; } }"#,
        r#"class C { void M() { for (Init(); Check(); Step()) { } } int Main() { return 0; } }"#,
        r#"class C { void M(int[] a) { for (a[0] = 1; a[0] < 9; a[0]++) { } } int Main() { return 0; } }"#,
        r#"class C { int F; void M() { for (F = 0; F < 9; F += 2) { } } int Main() { return 0; } }"#,
    ] {
        let unit = parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
        let Member::Class(class) = &unit.members[0] else {
            panic!("expected class member")
        };
        let Some(method) = class.members.iter().find_map(|m| match m {
            ClassMember::Method(method) => Some(method),
            _ => None,
        }) else {
            panic!("expected method member")
        };
        let Some(block) = &method.body else {
            panic!("expected method body")
        };
        assert!(
            matches!(&block.stmts[0], Stmt::For { .. }),
            "member-expression for must parse as a For"
        );
    }
}

#[test]
fn global_qualified_attribute_name_parses() {
    // `[global::System.ComponentModel.EditorBrowsableAttribute(global::System.ComponentModel.EditorBrowsableState.Never)]`
    // (Composition sources): `global::` is a transparent qualifier on
    // the attribute name (and inside its arguments), not a target
    // specifier. Regression: "expected as attribute name, found ':'".
    let src = r#"class C { [global::System.ComponentModel.EditorBrowsableAttribute(global::System.ComponentModel.EditorBrowsableState.Never)] public virtual int Parts { get { return 0; } } int Main() { return 0; } }"#;
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
}

#[test]
fn attribute_list_trailing_comma_parses() {
    // `[\nDefaultValue(...),\n]` (Odbc sources): a trailing comma in
    // an attribute list is legal C#. Regression: "expected as
    // attribute name, found 'RightBracket'".
    let src = r#"class C { [System.ComponentModel.DefaultValue(1),] public int X { get { return 0; } } int Main() { return 0; } }"#;
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
}

#[test]
fn interface_arrow_property_skips_cleanly() {
    // `internal static bool IsSupported => ...` in an interface
    // (TypeConverter's IDesignerHost): the interface-member scanner
    // must recognize `=>` as a property boundary, not a method
    // header. Regression: "expected '(' to open parameter list,
    // found '=>'".
    let src = "namespace D {\npublic interface I : J {\n[Attr(\"x\")]\ninternal static bool P => M(\"s\", out bool b) ? b : true;\nbool Loading { get; }\n}\n}";
    let pre = crate::preprocessor::preprocess(src, &[], "test.rc")
        .unwrap_or_else(|errors| panic!("preprocess failed: {errors:?}"));
    parser::parse(&pre.source)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{}", pre.source));
}

#[test]
fn float_attribute_arg_parses() {
    // `[TimersDescription(...), DefaultValue(100d)]` (Timer sources):
    // float literals (with `f`/`d`/`m` suffixes) are legal attribute
    // arguments. Regression: "expected ')' to close attribute
    // arguments, found float 100".
    let src = r#"class C { [A(B.C), DefaultValue(100d)] public double X { get { return 0; } } [A(-1.5f)] int Y() { return 0; } int Main() { return 0; } }"#;
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
}

#[test]
fn grouped_indexer_member_parses() {
    // `return (_currentDataTable.Columns[ordinal].ColumnName);`
    // (DataTableReader): a parenthesized member/indexer chain is a
    // grouping, never a tuple cast. Regression: the tuple-cast
    // lookahead consumed the group as a type, stranding `)` with
    // "expected ';' after statement".
    let src = r#"class T { string[] C; } class C { string M(T t, int o) { return (t.C[o]); } int Main() { return 0; } }"#;
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
}

#[test]
fn grouped_indexer_member_chain_parses() {
    // `return (_currentDataTable.Columns[ordinal].ColumnName);`
    // (DataTableReader): a parenthesized indexer+member chain is a
    // grouping — the cast gate must not read `[ordinal]` as a rank
    // specifier. Regression: "expected ';' after statement, found
    // ')'" (the chain parsed as a cast type, stranding the operand).
    let src = r#"class T { string M(int o) { return (t.C[o].X); } int Main() { return 0; } }"#;
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
}

#[test]
fn do_single_statement_body_parses() {
    // `do _pos++; while (...)` (Data ExpressionParser.ScanDate): C#
    // allows a braceless single-statement `do` body. Regression:
    // "expected '{' to open method body, found identifier '_pos'".
    let src = r#"class C { void M(int[] t) { int p = 0; do p++; while (p < t.Length); } int Main() { return 0; } }"#;
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
}

#[test]
fn dotted_array_creation_parses_to_new_array() {
    // Regression (System.Net.NameResolution): `new Interop.Sys.IPAddress[c]`
    // used to misparse as `New("Interop.Sys.IPAddress[]")` with the size
    // swallowed as a rank suffix (the sized branch only accepted a single
    // identifier) — a silent hollow alloc in static builds, and a loud
    // `ctor_arity` failure once owners route it cross-image.
    let src = r#"namespace Ns { class Inner { } } namespace t { class K { Ns.Inner[] M(int n) { return new Ns.Inner[n]; } int Main() { return 0; } } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let ns = &unit.namespaces[1];
    let Member::Class(class) = &ns.members[0] else {
        panic!("expected class");
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method");
    };
    let stmts = method.body.as_ref().unwrap().stmts.as_slice();
    match &stmts[0] {
        Stmt::Return { value: Some(expr), .. } => match expr {
            Expr::NewArray { elem_type, size, .. } => {
                assert_eq!(elem_type, "Ns.Inner");
                assert!(matches!(size.as_ref(), Expr::Ident { name, .. } if name == "n"));
            }
            other => panic!("expected NewArray, got {other:?}"),
        },
        other => panic!("expected Return, got {other:?}"),
    }
}

#[test]
fn nested_generic_field_type_parses() {
    // Regression (MemoryCache): a field typed `A<X>.B<Y>` (nested class
    // with its own type arguments,
    // `ConcurrentDictionary<…>.AlternateLookup<…>`) used to fail with
    // "expected identifier as method name, found 'Less'" — the
    // declaration lookahead accepted only one generic list before
    // dots, so the second `<` stranded the parse.
    let src = r#"namespace t { class Outer<T> { class Inner<U> { } } class C { Outer<string>.Inner<int> f; int Main() { return 0; } } }"#;
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let ns = &unit.namespaces[0];
    let Member::Class(class) = &ns.members[1] else {
        panic!("expected class C");
    };
    let ClassMember::Field(field) = &class.members[0] else {
        panic!("expected field, got {:?}", class.members[0]);
    };
    // Compact spelling (generic arguments dropped outside parameter
    // types); the point is it parses as one field, not an error.
    assert_eq!(field.type_name, "Outer<string>.Inner<int>");
}

#[test]
fn split_null_conditional_member_parses() {
    // Regression (ConfigurationBinder): a null-conditional split across
    // lines — `.Value?` newline `.ToString()` — used to fail with
    // "expected an expression, found Dot". A ternary branch can never
    // start with `.Identifier` (only a numeric like `.5` can), so
    // `?` + `.Ident` is unambiguously null-conditional.
    let src = "namespace t { class C { object v; string M() { string s = v.Value? .ToString(); return s; } int Main() { return 0; } } }";
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
}

#[test]
fn switch_expression_binary_continuation_parses() {
    // Regression (LoggerFactoryScopeProvider): `switch {…} ?? x` used to
    // fail with "expected ';' after statement, found 'QuestionQuestion'"
    // — the switch form returned before the operator loop, stranding
    // trailing binary operators. The result feeds back through it now.
    let src = "namespace t { class C { string M(int x) { string s = x switch { 1 => null, _ => null, } ?? string.Empty; return s; } int Main() { return 0; } } }";
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let ns = &unit.namespaces[0];
    let Member::Class(class) = &ns.members[0] else {
        panic!("expected class");
    };
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method");
    };
    let stmts = method.body.as_ref().unwrap().stmts.as_slice();
    match &stmts[0] {
        Stmt::Let { init, .. } => match init {
            Expr::Binary { op, .. } => {
                assert!(matches!(op, crate::models::BinOp::NullCoalesce));
            }
            other => panic!("expected ?? binary, got {other:?}"),
        },
        other => panic!("expected Let, got {other:?}"),
    }
}

#[test]
fn nested_generic_call_name_keeps_closers() {
    // Regression (ServiceProviderKeyedServiceExtensions): a call with
    // nested generic arguments
    // (`GetRequiredKeyedService<IEnumerable<T>>(…)`) recorded its name
    // as `…IEnumerable<T>` (missing the final `>`) — the `>>` (Shr)
    // closer dropped one bracket. The truncated name yields no
    // explicit type args, so overload ranking mis-elects a
    // non-generic overload.
    let src = "namespace t { class C { void M(IServiceProvider p, object k) { p.GetRequiredKeyedService<System.Collections.Generic.IEnumerable<T>>(k); } int Main() { return 0; } } interface IServiceProvider { } }";
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let ns = &unit.namespaces[0];
    let class = ns.members.iter().find_map(|m| match m {
        Member::Class(c) if c.name == "C" => Some(c),
        _ => None,
    }).expect("expected class C");
    let ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method");
    };
    let dbg = format!("{method:#?}");
    assert!(
        dbg.contains("GetRequiredKeyedService<System.Collections.Generic.IEnumerable<T>>"),
        "call name must keep both closers: {dbg}"
    );
}

#[test]
fn nested_generic_type_name_keeps_closers() {
    // Same `>>` closer rule for type positions: parameter types keep
    // full spellings (`A<B<int>>`), so overload ranking sees them.
    let src = "namespace t { class A<X> { } class B<Y> { } class C { void M(A<B<int>> x) { } int Main() { return 0; } } }";
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let dbg = format!("{unit:#?}");
    assert!(
        dbg.contains("A<B<int>>"),
        "type name must keep both closers: {dbg}"
    );
}

#[test]
fn postfix_nested_generic_call_keeps_closers() {
    // Regression (LengthBucketsFrozenDictionary): a postfix generic
    // call with nested arguments
    // (`GetAlternateEqualityComparer<ReadOnlySpan<char>>(…)`) recorded
    // its name as `…<ReadOnlySpan<char>` (missing `>`) — the `>>`
    // (Shr) closer dropped a bracket here too (same bug as the
    // call-name and type-name readers).
    let src = "namespace t { class C { void M(object o) { o.GetAlternateEqualityComparer<System.ReadOnlySpan<char>>(); } int Main() { return 0; } } }";
    let unit = parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
    let dbg = format!("{unit:#?}");
    assert!(
        dbg.contains("GetAlternateEqualityComparer<System.ReadOnlySpan<char>>"),
        "call name must keep both closers: {dbg}"
    );
}

#[test]
fn contextual_keyword_lambda_param_parses() {
    // `file` lexes as a keyword but C# allows it as an identifier
    // almost everywhere, including lambda parameters
    // (`files.Select(file => file.Path)`). The name-lambda arm only
    // accepted plain identifiers.
    let src = "namespace t { class F { string Path; } class C { void M(System.Collections.Generic.List<F> files) { files.Select(file => file.Path); } int Main() { return 0; } } } namespace System.Collections.Generic { class List<T> { } }";
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn generic_method_group_reference_parses() {
    // A generic method name without invocation (`NoDependencies =
    // Enumerable.Empty<T>;`) is a method group in value position,
    // not a call. The generic gate only accepted followers that
    // continue the expression (`(`, `.`, `)`, ...) — never a
    // statement end.
    let src = "namespace t { class C { object o = System.Linq.Enumerable.Empty<int>; int Main() { return 0; } } } namespace System.Linq { static class Enumerable { public static System.Collections.Generic.IEnumerable<T> Empty<T>() { return new System.Collections.Generic.List<T>(); } } } namespace System.Collections.Generic { interface IEnumerable<T> { } class List<T> : IEnumerable<T> { } }";
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn parenthesized_query_with_anonymous_type_parses() {
    // TypedParts shape: `(from ... let ... where ... select new {...})`
    // followed by `.ToArray()`. The query swallower must stop at the
    // enclosing `)` instead of eating it (`expected ')'`).
    let src = "namespace t { class C { object M(object xs) { var imports = (from pi in xs where pi != null let site = new Site(pi) select new { Site = site }).ToArray(); return imports; } int Main() { return 0; } } }";
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn query_as_call_argument_parses() {
    // The arg-list `)` likewise ends the query without being consumed.
    let src = "namespace t { class C { object M(object xs) { var r = Take(from pi in xs where pi != null select pi); return r; } object Take(object q) { return q; } int Main() { return 0; } } }";
    parser::parse(src)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn attributed_lambda_forms_parse() {
    // Attributes on lambdas (C# 10+): `[Attr(...)] (params) => body`.
    // The attribute list is consumed only when a lambda shape follows
    // the closing bracket (a bare `[` otherwise starts a collection);
    // attributes carry no lowering meaning and are dropped. Regression:
    // event subscription `E += [Attr] (x) => {...}` failed with
    // "expected ';' after assignment".
    for src in [
        r#"class C { Main() { E += [System.Obsolete("x")] (x) => { }; return 0; } }"#,
        r#"class C { Main() { var f = [A] (a, b) => a; return 0; } }"#,
        r#"class C { Main() { var f = [A] x => x; return 0; } }"#,
        r#"class C { Main() { var f = [A] static (x) => x; return 0; } }"#,
        r#"class C { Main() { var f = [A] async (x) => x; return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
    // Collection expressions still parse as collections, not lambdas.
    parser::parse(r#"class C { Main() { var xs = new int[3]; xs[0] = 1; return 0; } }"#)
        .unwrap_or_else(|errors| panic!("parse failed: {errors:?}"));
}

#[test]
fn attribute_new_array_arguments_parse() {
    // `new[]` / `new T[]` / `new T(...)` as attribute argument values
    // (e.g. `[UnmanagedCallConv(CallConvs = new[] { typeof(X) })]`):
    // values are inert metadata, consumed balanced. Regression: Common
    // interop sources failed with "expected ')' to close attribute
    // arguments, found 'LeftBracket'".
    for src in [
        r#"class C { [A(CallConvs = new[] { typeof(X) })] int M() { return 0; } }"#,
        r#"class C { [A(Vals = new int[] { 1, 2 })] int M() { return 0; } }"#,
        r#"class C { [A(V = new Foo(1, "x"))] int M() { return 0; } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}

#[test]
fn extension_block_generic_receiver_parses() {
    // Extension blocks whose receiver type carries generic commas
    // (`extension<TKey, TValue>(ConcurrentDictionary<TKey, TValue> d)`):
    // the receiver scanner must not mistake the generic comma for a
    // separator. Regression: Common polyfills failed with "expected
    // identifier as method name, found 'Less'".
    for src in [
        r#"class P { extension<TKey, TValue>(ConcurrentDictionary<TKey, TValue> dictionary) where TKey : notnull { public int M() { return 0; } } }"#,
        r#"class P { extension<T>(List<T> items) { public int M() { return 0; } } }"#,
    ] {
        parser::parse(src)
            .unwrap_or_else(|errors| panic!("parse failed: {errors:?} for source:\n{src}"));
    }
}
