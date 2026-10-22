//! `.rci` fragment roundtrip: encode then decode must preserve every
//! field (the consume path trusts decoded entries verbatim).

use crate::models::Access;
use crate::native::rci;
use crate::native::semantic::{
    ClassInfo, ConstValue, FieldInfo, MethodInfo, PropertyInfo, TypeKey,
};

fn sample_class() -> ClassInfo {
    ClassInfo {
        is_record: false,
        type_arg_index: Vec::new(),
        name: "Widget".into(),
        key: TypeKey::new("t.Widget", 1),
        bases: vec!["Object".into()],
        fields: vec![FieldInfo {
            name: "count".into(),
            type_name: "int".into(),
            access: Access::Private,
            offset_slots: 3,
            is_static: false,
            is_const: false,
            is_byref: false,
            const_value: None,
            declaring_class: "Widget".into(),
        }],
        static_fields: vec![FieldInfo {
            name: "cache".into(),
            type_name: "object".into(),
            access: Access::Public,
            offset_slots: 0,
            is_static: true,
            is_const: false,
            is_byref: false,
            const_value: None,
            declaring_class: "Widget".into(),
        }],
        const_fields: vec![FieldInfo {
            name: "Max".into(),
            type_name: "int".into(),
            access: Access::Public,
            offset_slots: 0,
            is_static: false,
            is_const: true,
            is_byref: false,
            const_value: Some(ConstValue::Int(9)),
            declaring_class: "Widget".into(),
        }],
        methods: vec![MethodInfo {
            param_names: vec![],
            name: "Get".into(),
            ret_type: "int".into(),
            param_types: vec!["string".into()],
            param_has_default: vec![true],
            param_is_byref: vec![false],
            has_params: false,
            param_defaults: vec![Some(ConstValue::Null)],
            access: Access::Protected,
            is_static: false,
            has_body: true,
            declaring_class: "Widget".into(),
            is_extension: false,
            generic_params: vec!["T".into()],
            conditional_symbol: Some("DEBUG".into()),
            vtable_slot: 7,
        }],
        own_members: vec![("count".into(), true, Access::Private)],
        interfaces: vec![("IDrop".into(), 0)],
        properties: vec![PropertyInfo {
            name: "Size".into(),
            type_name: "int".into(),
            access: Access::Internal,
            declaring_class: "Widget".into(),
            is_static: true,
            is_abstract: false,
        }],
        is_enum: false,
        generic_params: vec!["T".into()],
        is_struct: false,
        owner: "LibA".into(),
    }
}

#[test]
fn rci_roundtrip_preserves_entries() {
    let other = ClassInfo {
        is_record: false,
        type_arg_index: Vec::new(),
        name: "Other".into(),
        key: TypeKey::new("t.Other", 0),
        bases: vec![],
        fields: vec![],
        static_fields: vec![],
        const_fields: vec![],
        methods: vec![],
        own_members: vec![],
        interfaces: vec![],
        properties: vec![],
        is_enum: true,
        generic_params: vec![],
        is_struct: false,
        owner: "LibB".into(),
    };
    let bytes = rci::encode(
        "LibA",
        0x1234,
        &[sample_class(), other],
        &[sample_iface()],
        &[sample_delegate()],
    );
    let frag = rci::decode(&bytes).expect("roundtrip must decode");
    assert_eq!(frag.lib_name, "LibA");
    assert_eq!(frag.source_hash, 0x1234);
    // Only LibA-owned entries are stored.
    assert_eq!(frag.classes.len(), 1);
    let c = &frag.classes[0];
    assert_eq!(c.name, "Widget");
    assert_eq!(c.key, TypeKey::new("t.Widget", 1));
    assert_eq!(c.owner, "LibA");
    assert_eq!(c.bases, vec!["Object".to_owned()]);
    assert_eq!(c.fields.len(), 1);
    assert_eq!(c.fields[0].offset_slots, 3);
    assert_eq!(c.static_fields.len(), 1);
    assert!(c.static_fields[0].is_static);
    assert_eq!(
        c.const_fields[0].const_value,
        Some(ConstValue::Int(9))
    );
    assert_eq!(c.methods.len(), 1);
    let m = &c.methods[0];
    assert_eq!(m.param_has_default, vec![true]);
    assert_eq!(m.param_defaults, vec![Some(ConstValue::Null)]);
    assert_eq!(m.conditional_symbol.as_deref(), Some("DEBUG"));
    assert_eq!(m.vtable_slot, 7);
    assert!(c.properties[0].is_static);
    assert_eq!(c.interfaces, vec![("IDrop".to_owned(), 0)]);
    assert_eq!(frag.interfaces.len(), 1);
    assert_eq!(frag.interfaces[0].name, "IDrop");
    assert_eq!(frag.interfaces[0].methods.len(), 1);
    assert_eq!(frag.interfaces[0].methods[0].name, "Area");
    assert!(frag.interfaces[0].methods[0].params[0].is_this);
    assert_eq!(frag.delegates.len(), 1);
    assert_eq!(frag.delegates[0].name, "Handler");
    assert!(frag.delegates[0].params[0].is_params);
}

fn sample_iface() -> crate::models::InterfaceDecl {
    crate::models::InterfaceDecl {
        indexers: vec![],
        name: "IDrop".into(),
        generic_params: vec![],
        methods: vec![crate::models::MethodSig {
            access: crate::models::Access::Public,
            ret_type: "int".into(),
            name: "Area".into(),
            generic_params: vec![],
            params: vec![crate::models::Param {
                attrs: vec![],
                type_name: "IDrop".into(),
                name: "self".into(),
                span: Default::default(),
                default_value: None,
                is_this: true,
                is_params: false,
                is_byref: false,
            }],
            body: None,
            is_static: false,
            span: Default::default(),
        }],
        bases: vec![],
        arity: 0,
        is_partial: false,
        span: Default::default(),
    }
}

fn sample_delegate() -> crate::models::DelegateDecl {
    crate::models::DelegateDecl {
        access: crate::models::Access::Public,
        ret_type: "void".into(),
        name: "Handler".into(),
        generic_params: vec!["T".into()],
        params: vec![crate::models::Param {
            attrs: vec![],
            type_name: "T[]".into(),
            name: "args".into(),
            span: Default::default(),
            default_value: None,
            is_this: false,
            is_params: true,
            is_byref: false,
        }],
        span: Default::default(),
    }
}

#[test]
fn rci_decode_rejects_bad_magic_and_version() {
    assert!(rci::decode(b"NOPE").is_err());
    let mut bytes = rci::encode("L", 0, &[], &[], &[]);
    // Corrupt the version word (offset 4).
    bytes[4] = 0xFF;
    assert!(rci::decode(&bytes).is_err());
    // Truncate.
    assert!(rci::decode(&bytes[..10]).is_err());
}
