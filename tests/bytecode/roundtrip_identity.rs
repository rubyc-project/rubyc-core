//! Bytecode round-trip tests: encode → decode must preserve structure for
//! every shared sample, and corrupt input must fail cleanly.

use crate::bytecode::traits::ToBytecode;
use crate::bytecode::{decode, format};
use crate::models::compare::StructuralEq;
use crate::models::{DelegateDecl, Member};
use crate::native::container::{self, Kind};
use crate::parser;
use crate::testdata::SAMPLES;

#[test]
fn every_sample_round_trips() {
    for sample in SAMPLES {
        let unit = parser::parse(sample.source)
            .unwrap_or_else(|e| panic!("sample {} failed to parse: {e:?}", sample.name));
        let bytes = unit.to_bytecode();
        let decoded = decode(&bytes)
            .unwrap_or_else(|e| panic!("sample {} failed to decode: {e}", sample.name));
        assert!(
            decoded.structural_eq(&unit),
            "sample {} changed across round-trip",
            sample.name
        );
    }
}

#[test]
fn encoding_is_deterministic() {
    for sample in SAMPLES {
        let unit = parser::parse(sample.source).unwrap();
        let a = unit.to_bytecode();
        let b = decode(&a).unwrap().to_bytecode();
        assert_eq!(a, b, "re-encoding differs for {}", sample.name);
    }
}

#[test]
fn using_aliases_survive_bytecode_round_trip() {
    let unit = crate::parser::parse("using NativeType = System.Int32;\nusing static System.Math;\nclass C {\n}\n").unwrap();
    assert_eq!(unit.aliases.len(), 1);
    assert_eq!(unit.static_usings.len(), 1);
    let bytes = unit.to_bytecode();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.aliases, unit.aliases);
    assert_eq!(decoded.static_usings, unit.static_usings);
    assert!(unit.structural_eq(&decoded));
    // Re-encoding is byte-identical (deterministic).
    assert_eq!(decoded.to_bytecode(), bytes);
}

#[test]
fn assembly_blocks_survive_bytecode_round_trip() {
    let mut unit = crate::parser::parse(crate::testdata::BLOCK_FORM).unwrap();
    unit.assembly_blocks = vec!["nop\n".into(), "xor rax, rax\n".into()];
    let bytes = unit.to_bytecode();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded.assembly_blocks, unit.assembly_blocks);
    assert!(unit.structural_eq(&decoded));
}

#[test]
fn header_is_jump_dog_kind_version() {
    let unit = parser::parse(SAMPLES[0].source).unwrap();
    let bytes = unit.to_bytecode();
    // [EB 08][🐕🇨][kind=bytecode][version]
    assert_eq!(&bytes[..2], &container::JUMP_OVER_MAGIC);
    assert_eq!(&bytes[2..10], &container::MAGIC);
    assert_eq!(bytes[10], Kind::Bytecode.to_byte());
    assert_eq!(&bytes[11..13], &format::FORMAT_VERSION.to_le_bytes());
}

#[test]
fn delegate_round_trips() {
    let src = "namespace ns {\n\
        public delegate int Callback(out int id, in string name);\n\
        public delegate System.Threading.Tasks.Task<int> Fetcher<T>(string url);\n\
    }\n";
    let unit = parser::parse(src).expect("delegate source parses");
    let bytes = unit.to_bytecode();
    let decoded = decode(&bytes).expect("delegate bytecode decodes");
    assert!(decoded.structural_eq(&unit), "delegate changed across round-trip");

    // Both delegates survive with their generic params and variance qualifiers.
    let members = decoded
        .namespaces
        .first()
        .expect("namespace")
        .members
        .iter()
        .map(|m| match m {
            Member::Delegate(d) => d.clone(),
            _ => panic!("expected only delegates, got {m:?}"),
        })
        .collect::<Vec<DelegateDecl>>();
    assert_eq!(members.len(), 2);
    assert_eq!(members[0].name, "Callback");
    assert_eq!(members[0].ret_type, "int");
    assert_eq!(members[0].params.len(), 2);
    assert_eq!(members[1].name, "Fetcher");
    assert_eq!(members[1].generic_params, vec!["T".to_string()]);
    // Generic *arguments* on the return type are consumed and dropped; the
    // compact model keeps the base name.
    assert_eq!(members[1].ret_type, "System.Threading.Tasks.Task");
}

#[test]
fn array_and_nullable_type_round_trips() {
    let src = "namespace ns {\n\
        public class Box {\n\
            private int[] _values;\n\
            private int? _maybe;\n\
            private string[] _names;\n\
            public int[] Values => _values;\n\
            public int? Maybe => _maybe;\n\
        }\n\
    }\n";
    let unit = parser::parse(src).expect("array/nullable source parses");
    let bytes = unit.to_bytecode();
    let decoded = decode(&bytes).expect("array/nullable bytecode decodes");
    assert!(decoded.structural_eq(&unit), "array/nullable changed across round-trip");

    let class = decoded
        .namespaces
        .first()
        .expect("namespace")
        .members
        .iter()
        .find_map(|m| match m {
            Member::Class(c) => Some(c),
            _ => None,
        })
        .expect("Box class present");

    let members = class.members.clone();
    assert_eq!(members.len(), 5, "expected 5 members, got {}", members.len());

    // int[] field.
    match &members[0] {
        crate::models::ClassMember::Field(f) => {
            assert_eq!(f.type_name, "int[]");
            assert_eq!(f.name, "_values");
        }
        other => panic!("expected int[] field, got {other:?}"),
    }
    // int? field.
    match &members[1] {
        crate::models::ClassMember::Field(f) => {
            assert_eq!(f.type_name, "int?");
            assert_eq!(f.name, "_maybe");
        }
        other => panic!("expected int? field, got {other:?}"),
    }
    // int[] property (expression-bodied).
    match &members[3] {
        crate::models::ClassMember::Property(p) => {
            assert_eq!(p.type_name, "int[]");
            assert_eq!(p.name, "Values");
            assert!(p.get_expr.is_some());
        }
        other => panic!("expected int[] property, got {other:?}"),
    }
    // int? property (expression-bodied).
    match &members[4] {
        crate::models::ClassMember::Property(p) => {
            assert_eq!(p.type_name, "int?");
            assert_eq!(p.name, "Maybe");
            assert!(p.get_expr.is_some());
        }
        other => panic!("expected int? property, got {other:?}"),
    }
}

#[test]
fn const_and_property_round_trips() {
    let src = "namespace ns {\n\
        public class Widget {\n\
            public const int Answer = 42;\n\
            public const string Greeting = \"hi\";\n\
            private int _value;\n\
            public int Value { get { return _value; } set { _value = value; } }\n\
            public int Auto { get; set; } = 11;\n\
            public int GetterOnly { get; } = 22;\n\
            public string InitOnly { get; init; } = \"INIT\";\n\
            public int Expression => _value * 2;\n\
            public int this[int i] { get { return _value; } set { _value = value; } }\n\
            public event System.Action Changed;\n\
        }\n\
    }\n";
    let unit = parser::parse(src).expect("const/property source parses");
    let bytes = unit.to_bytecode();
    let decoded = decode(&bytes).expect("const/property bytecode decodes");
    assert!(decoded.structural_eq(&unit), "const/property changed across round-trip");

    let class = decoded
        .namespaces
        .first()
        .expect("namespace")
        .members
        .iter()
        .find_map(|m| match m {
            Member::Class(c) => Some(c),
            _ => None,
        })
        .expect("Widget class present");

    let members = class.members.clone();
    assert_eq!(members.len(), 10, "expected 10 members, got {}", members.len());

    // Two const fields, no layout slot, init preserved.
    match &members[0] {
        crate::models::ClassMember::Field(f) => {
            assert!(f.is_const);
            assert_eq!(f.name, "Answer");
            assert!(matches!(&f.init, crate::models::Expr::Literal(crate::models::Literal::Int(42))));
        }
        other => panic!("expected const field, got {other:?}"),
    }
    match &members[1] {
        crate::models::ClassMember::Field(f) => {
            assert!(f.is_const);
            assert_eq!(f.name, "Greeting");
        }
        other => panic!("expected const field, got {other:?}"),
    }
    // `_value` is a plain (non-const) backing field.
    match &members[2] {
        crate::models::ClassMember::Field(f) => {
            assert!(!f.is_const);
            assert_eq!(f.name, "_value");
        }
        other => panic!("expected field, got {other:?}"),
    }

    // Explicit accessor property.
    match &members[3] {
        crate::models::ClassMember::Property(p) => {
            assert_eq!(p.name, "Value");
            assert!(p.get.is_some());
            assert!(p.set.is_some());
            assert!(p.get_expr.is_none());
            assert!(p.is_settable);
            assert!(!p.is_init);
        }
        other => panic!("expected property, got {other:?}"),
    }

    // Auto settable property with initializer.
    match &members[4] {
        crate::models::ClassMember::Property(p) => {
            assert_eq!(p.name, "Auto");
            assert!(p.is_settable);
            assert!(!p.is_init);
            assert!(p.get.is_none());
            assert!(p.set.is_none());
            assert!(matches!(&p.init, crate::models::Expr::Literal(crate::models::Literal::Int(11))));
        }
        other => panic!("expected property, got {other:?}"),
    }

    // Getter-only.
    match &members[5] {
        crate::models::ClassMember::Property(p) => {
            assert_eq!(p.name, "GetterOnly");
            assert!(!p.is_settable);
            assert!(!p.is_init);
        }
        other => panic!("expected property, got {other:?}"),
    }

    // Init-only.
    match &members[6] {
        crate::models::ClassMember::Property(p) => {
            assert_eq!(p.name, "InitOnly");
            assert!(p.is_init);
            assert!(!p.is_settable);
        }
        other => panic!("expected property, got {other:?}"),
    }

    // Expression-bodied property.
    match &members[7] {
        crate::models::ClassMember::Property(p) => {
            assert_eq!(p.name, "Expression");
            assert!(p.get_expr.is_some());
            assert!(p.get.is_none());
            assert!(p.set.is_none());
            assert!(!p.is_settable);
            assert!(!p.is_init);
        }
        other => panic!("expected property, got {other:?}"),
    }

    // Indexer with explicit accessors.
    match &members[8] {
        crate::models::ClassMember::Indexer(idx) => {
            assert_eq!(idx.type_name, "int");
            assert_eq!(idx.params.len(), 1);
            assert_eq!(idx.params[0].name, "i");
            assert!(idx.get.is_some());
            assert!(idx.set.is_some());
            assert!(idx.get_expr.is_none());
            assert!(idx.is_settable);
        }
        other => panic!("expected indexer, got {other:?}"),
    }

    // Event.
    match &members[9] {
        crate::models::ClassMember::Event(evt) => {
            assert_eq!(evt.name, "Changed");
            assert_eq!(evt.type_name, "System.Action");
        }
        other => panic!("expected event, got {other:?}"),
    }
}

#[test]
fn bad_magic_is_rejected() {
    let bytes = vec![b'X', 0, 0, 0, 1, 0];
    assert_eq!(decode(&bytes), Err(crate::models::DecodeError::BadMagic));
}

#[test]
fn unsupported_version_is_rejected() {
    let mut bytes = container::build(Kind::Bytecode, 999);
    bytes.push(0x01); // UNIT tag so the version check is what fails
    assert_eq!(
        decode(&bytes),
        Err(crate::models::DecodeError::UnsupportedVersion(999))
    );
}

#[test]
fn wrong_kind_is_rejected() {
    let mut bytes = container::build(Kind::NativeExecutable, format::FORMAT_VERSION);
    bytes.push(0x01);
    assert_eq!(decode(&bytes), Err(crate::models::DecodeError::WrongKind));
}

#[test]
fn truncated_input_is_rejected() {
    let unit = parser::parse(SAMPLES[0].source).unwrap();
    let bytes = unit.to_bytecode();
    for cut in 13..bytes.len() {
        let result = decode(&bytes[..cut]);
        assert!(result.is_err(), "truncated at {cut} unexpectedly decoded");
    }
}

#[test]
fn trailing_bytes_are_rejected() {
    let unit = parser::parse(SAMPLES[0].source).unwrap();
    let mut bytes = unit.to_bytecode();
    bytes.push(0xFF);
    assert_eq!(
        decode(&bytes),
        Err(crate::models::DecodeError::TrailingBytes)
    );
}

#[test]
fn unknown_node_tag_is_rejected() {
    // Valid header followed by an invalid root tag.
    let mut bytes = container::build(Kind::Bytecode, format::FORMAT_VERSION);
    bytes.push(0xEE);
    assert_eq!(
        decode(&bytes),
        Err(crate::models::DecodeError::UnknownNodeTag(0xEE))
    );
}

#[test]
fn generic_params_round_trip() {
    // Generic type parameter names must survive encode/decode on
    // classes, methods, interfaces, and delegates.
    let src = r#"namespace n {
        class Box<T> {
            public T GetOrDefault<TDefault>(TDefault d) { return d; }
        }
        interface ISeq<T> {
            T First();
        }
        delegate U Convert<T, U>(T value);
    }"#;
    let unit = parser::parse(src).unwrap();
    let bytes = unit.to_bytecode();
    let decoded = decode(&bytes).unwrap();
    assert!(
        decoded.structural_eq(&unit),
        "generic units must round-trip structurally"
    );
    let Some(ns) = decoded.namespaces.iter().find(|n| n.name == "n") else {
        panic!("expected namespace n")
    };
    let Member::Class(class) = &ns.members[0] else {
        panic!("expected class member")
    };
    assert_eq!(class.generic_params, vec!["T".to_string()]);
    let crate::models::ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    assert_eq!(method.generic_params, vec!["TDefault".to_string()]);
}

#[test]
fn extension_receiver_round_trip() {
    // The `this` receiver marker on extension-method parameters must
    // survive encode/decode (lowering keys off it).
    let src = r#"namespace n {
        class Ext {
            public static int Twice(this int x) { return x; }
        }
    }"#;
    let unit = parser::parse(src).unwrap();
    let bytes = unit.to_bytecode();
    let decoded = decode(&bytes).unwrap();
    assert!(
        decoded.structural_eq(&unit),
        "extension units must round-trip structurally"
    );
    let Some(ns) = decoded.namespaces.iter().find(|n| n.name == "n") else {
        panic!("expected namespace n")
    };
    let Member::Class(class) = &ns.members[0] else {
        panic!("expected class member")
    };
    let crate::models::ClassMember::Method(method) = &class.members[0] else {
        panic!("expected method member")
    };
    assert!(
        method.params.first().is_some_and(|p| p.is_this),
        "expected first param to keep is_this"
    );
}

#[test]
fn interface_bases_round_trip() {
    // Inherited-interface lists must survive encode/decode (member
    // lookup walks them transitively).
    let src = r#"namespace n {
        interface IDisposable {
            void Dispose();
        }
        interface IAsyncEnum : IDisposable {
            int Next();
        }
    }"#;
    let unit = parser::parse(src).unwrap();
    let bytes = unit.to_bytecode();
    let decoded = decode(&bytes).unwrap();
    assert!(
        decoded.structural_eq(&unit),
        "interface units must round-trip structurally"
    );
    let Some(ns) = decoded.namespaces.iter().find(|n| n.name == "n") else {
        panic!("expected namespace n")
    };
    let faces: Vec<&crate::models::InterfaceDecl> = ns
        .members
        .iter()
        .filter_map(|m| match m {
            Member::Interface(i) => Some(i),
            _ => None,
        })
        .collect();
    let child = faces
        .iter()
        .find(|i| i.name == "IAsyncEnum")
        .expect("expected IAsyncEnum");
    assert_eq!(child.bases, vec!["IDisposable".to_string()]);
}

#[test]
fn internal_access_survives_round_trip() {
    // `Access::Internal` used to collapse to `Default` on decode
    // (the codec only knew Public/Private/Protected), breaking
    // round-trips of any `internal` member (found via
    // `internal int` in System.Net.Primitives).
    let unit = parser::parse(
        "class C { internal int Y = 1; internal const uint M = 0xFF000000; internal void M() { } internal int P { get { return 0; } } int Main() { return 0; } }",
    )
    .unwrap();
    let bytes = unit.to_bytecode();
    let decoded = decode(&bytes).unwrap();
    assert!(decoded.structural_eq(&unit));
    assert_eq!(decoded.to_bytecode(), bytes);
}

#[test]
fn index_assign_keeps_simple_assign_shape() {
    // `labels[5] = 0xFFFF` (element store) parses to `Expr::Assign`
    // with an index target — NOT the semantically distinct
    // `AssignComplex`. The codec once reused the `ASSIGN_COMPLEX`
    // tag for it (decoding to the wrong variant), then the dedicated
    // `EXPR_ASSIGN` tag collided with `stmt::DO_WHILE` (both 0x32):
    // decode misrouted and died with `UnexpectedEof`.
    let unit = parser::parse(
        "class C { static void M(ushort[] n) { n[0] = 65535; } int Main() { return 0; } }",
    )
    .unwrap();
    let bytes = unit.to_bytecode();
    let decoded = decode(&bytes).unwrap();
    assert!(decoded.structural_eq(&unit));
    assert_eq!(decoded.to_bytecode(), bytes);
}

#[test]
fn char_literals_survive_bytecode_round_trip() {
    // Without a CHAR tag a char literal decoded as an integer and
    // `var c = 'A'` lost its type across `compile -O bytecode`.
    let unit = parser::parse("class C { int Main() { var c = 'A'; return c; } }").unwrap();
    let bytes = unit.to_bytecode();
    let decoded = decode(&bytes).expect("char literal decodes");
    assert!(decoded.structural_eq(&unit), "char literal changed type");
}
