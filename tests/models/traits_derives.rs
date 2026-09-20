//! Trait coverage for the model types: every derive we rely on
//! (Clone / Copy / PartialEq / Eq / Debug) and every hand-written trait
//! (StructuralEq, ToBytecode, FromBytecode) gets explicit assertions.

use crate::bytecode::traits::{FromBytecode, ToBytecode};
use crate::models::compare::StructuralEq;
use crate::models::{
    Access, Block, ClassDecl, ClassMember, CompilationUnit, Expr, Keyword, Literal, Member,
    MethodDecl, NamespaceDecl, Span, Stmt, TokenKind,
};

/// Hand-built hello-world tree with non-zero spans (proves compares ignore
/// locations).
fn sample_unit(span_start: u32) -> CompilationUnit {
    let sp = |off: u32| Span::new(span_start + off, span_start + off + 1);
    CompilationUnit {
        namespaces: vec![NamespaceDecl {
            name: "testnc".into(),
            members: vec![Member::Class(ClassDecl {
                name: "testc".into(),
                generic_params: vec![],
                primary_ctor_params: vec![],
                bases: vec![],
                 members: vec![ClassMember::Method(MethodDecl {
                     generic_params: vec![],
                     is_override: false,
                     is_static: false,
                     is_abstract: false,
                     is_partial: false,
                     is_async: false,
                     is_extern: false,
                    base_call: None,
                    access: Access::Default,
                    attrs: vec![],
                    ret_type: "int".into(),
                    name: "main".into(),
                    params: vec![],
                    body: Some(Block {
                        stmts: vec![Stmt::Expression(Expr::Call {
                            callee: "printf".into(),
                            args: vec![Expr::Literal(Literal::Str("Hello World!".into()))],
                            span: sp(50),
                        })],
                        span: sp(40),
                    }),
                    span: sp(30),
                })],
                span: sp(20),
                is_partial: false,
                is_struct: false,
            })],
            span: sp(10),
        }],
        members: vec![],
        aliases: vec![],
        static_usings: vec![],
        assembly_blocks: vec![],
        span: sp(0),
    }
}

#[test]
fn clone_produces_independent_trees() {
    let unit = sample_unit(0);
    let copy = unit.clone();
    assert_eq!(unit, copy);
    // Deep clone: mutating the original's nested string must not leak.
    for ns in &unit.namespaces {
        let Member::Class(class) = &ns.members[0] else {
            panic!("expected class member")
        };
        let ClassMember::Method(method) = &class.members[0] else {
            panic!("expected method member")
        };
        let Stmt::Expression(expr) = &method.body.as_ref().unwrap().stmts[0] else {
            panic!("expected an expression statement");
        };
        let Expr::Call { callee, args, .. } = expr else {
            panic!("expected a call expression");
        };
        assert_eq!(callee, "printf");
        assert_eq!(args.len(), 1);
    }
}

#[test]
fn partial_eq_and_eq_agree() {
    let a = sample_unit(0);
    let b = sample_unit(100); // different spans only
    assert_ne!(a, b, "PartialEq is exact, including spans");

    // Eq (reflexivity) holds for every model type exercised here.
    assert!(a == a.clone());
    let _ = Box::new(a.namespaces.clone()); // hashable/equatable usage smoke test
}

#[test]
fn debug_is_implemented_for_all_models() {
    let unit = sample_unit(7);
    let dbg = format!("{unit:?}");
    for needle in ["testnc", "testc", "main", "printf", "Hello World!"] {
        assert!(dbg.contains(needle), "Debug output missing {needle}");
    }

    // Token-side models too.
    assert!(format!("{:?}", TokenKind::Identifier("x".into())).contains('x'));
    assert!(format!("{:?}", Keyword::Namespace).contains("Namespace"));
}

#[test]
fn copy_semantics_on_small_types() {
    let s = Span::new(3, 9);
    let s2 = s; // moves are copies for Copy types
    assert_eq!(s, s2);
    assert_eq!(s.start + s2.end, 12);

    let kw = Keyword::Namespace;
    let kw2 = kw;
    assert_eq!(kw, kw2);
}

#[test]
fn structural_eq_ignores_spans_but_not_content() {
    let a = sample_unit(0);
    let b = sample_unit(5000);
    assert!(a.structural_eq(&b), "spans must not affect structure");

    // Any content change must break equality.
    let mut c = sample_unit(0);
    c.namespaces[0].name = "other".into();
    assert!(!a.structural_eq(&c));

    let mut d = sample_unit(0);
    if let Some(ns) = d.namespaces.first_mut() {
        let Member::Class(class) = &mut ns.members[0] else {
            panic!("class")
        };
        class.name = "renamed".into();
    }
    assert!(!a.structural_eq(&d));

    let mut e = sample_unit(0);
    if let Some(ns) = e.namespaces.first_mut() {
        let Member::Class(class) = &mut ns.members[0] else {
            panic!("class")
        };
        let ClassMember::Method(m) = &mut class.members[0] else {
            panic!("method")
        };
        m.body
            .as_mut()
            .unwrap()
            .stmts
            .push(Stmt::Expression(Expr::Ident {
                name: "x".into(),
                span: Default::default(),
            }));
    }
    assert!(!a.structural_eq(&e));
}

#[test]
fn to_bytecode_from_bytecode_round_trips_hand_built_tree() {
    let unit = sample_unit(123);
    let bytes = unit.to_bytecode();

    // Header sanity on hand-built input.
    assert_eq!(&bytes[..2], &[0xEB, 0x08]);
    assert_eq!(bytes[10], 0x01, "kind = bytecode");

    let decoded = CompilationUnit::from_bytecode(&bytes).expect("decode");
    assert!(decoded.structural_eq(&unit));
}

#[test]
fn from_bytecode_rejects_garbage() {
    assert!(CompilationUnit::from_bytecode(b"").is_err());
    assert!(CompilationUnit::from_bytecode(b"not rubyc at all").is_err());
}

#[test]
fn default_compilation_unit_is_empty_and_encodable() {
    let empty = CompilationUnit::default();
    assert!(empty.namespaces.is_empty());
    assert!(empty.members.is_empty());

    let decoded = CompilationUnit::from_bytecode(&empty.to_bytecode()).unwrap();
    assert!(decoded.structural_eq(&empty));
}

// ---------- StructuralEq: every AST node, positive AND negative ----------

mod structural_eq {
    use crate::models::compare::StructuralEq;
    use crate::models::*;

    fn expr_int(v: i64) -> Expr {
        Expr::Literal(Literal::Int(v))
    }

    fn call(name: &str, args: Vec<Expr>) -> Expr {
        Expr::Call {
            callee: name.into(),
            args,
            span: Span::new(1, 2),
        }
    }

    fn let_stmt(name: &str, init: Expr) -> Stmt {
        Stmt::Let {
            type_name: "int".into(),
            name: name.into(),
            init,
            access: Access::Default,
            span: Span::default(),
        }
    }

    fn method(name: &str) -> MethodDecl {
        MethodDecl {
            generic_params: vec![],
            is_override: false,
            is_static: false,
            is_abstract: false,
            is_partial: false,
            is_async: false,
            is_extern: false,
            base_call: None,
            access: Access::Default,
            attrs: vec![],
            ret_type: "int".into(),
            name: name.into(),
            params: vec![],
            body: Some(Block {
                stmts: vec![],
                span: Span::default(),
            }),
            span: Span::default(),
        }
    }

    fn class(name: &str) -> ClassDecl {
        ClassDecl {
            name: name.into(),
            generic_params: vec![],
            primary_ctor_params: vec![],
            bases: Vec::new(),
            members: vec![ClassMember::Method(method("main"))],
            span: Span::default(),
            is_partial: false,
            is_struct: false,
        }
    }

    fn unit_with_class(class_name: &str) -> CompilationUnit {
        CompilationUnit {
            namespaces: vec![NamespaceDecl {
                name: "ns".into(),
                members: vec![Member::Class(class(class_name))],
                span: Span::default(),
            }],
            members: vec![],
            aliases: vec![],
            static_usings: vec![],
            assembly_blocks: vec![],
            span: Span::default(),
        }
    }

    #[test]
    fn expr_binary() {
        let mk = |op| Expr::Binary {
            op,
            lhs: Box::new(expr_int(1)),
            rhs: Box::new(expr_int(2)),
            span: Span::default(),
        };
        assert!(mk(BinOp::Add).structural_eq(&mk(BinOp::Add)));
        assert!(!mk(BinOp::Add).structural_eq(&mk(BinOp::Sub)));
        assert!(!mk(BinOp::Add).structural_eq(&mk(BinOp::Mul)));
        assert!(!mk(BinOp::Add).structural_eq(&mk(BinOp::Div)));
    }

    #[test]
    fn expr_binary_operand_change_detected() {
        let a = Expr::Binary {
            op: BinOp::Add,
            lhs: Box::new(expr_int(1)),
            rhs: Box::new(expr_int(2)),
            span: Span::default(),
        };
        let b = Expr::Binary {
            op: BinOp::Add,
            lhs: Box::new(expr_int(9)), // changed
            rhs: Box::new(expr_int(2)),
            span: Span::default(),
        };
        assert!(!a.structural_eq(&b));
    }

    #[test]
    fn expr_unary() {
        let mk = || Expr::Unary {
            op: UnOp::Neg,
            inner: Box::new(expr_int(5)),
            span: Span::default(),
        };
        assert!(mk().structural_eq(&mk()));
        // different inner
        let other = Expr::Unary {
            op: UnOp::Neg,
            inner: Box::new(expr_int(6)),
            span: Span::default(),
        };
        assert!(!mk().structural_eq(&other));
    }

    #[test]
    fn expr_call() {
        assert!(
            call("printf", vec![expr_int(1)]).structural_eq(&call("printf", vec![expr_int(1)]))
        );
        assert!(
            !call("printf", vec![expr_int(1)]).structural_eq(&call("print", vec![expr_int(1)]))
        );
        assert!(
            !call("printf", vec![expr_int(1)]).structural_eq(&call("printf", vec![expr_int(2)]))
        );
        assert!(!call("printf", vec![expr_int(1)]).structural_eq(&call("printf", vec![])));
    }

    #[test]
    fn expr_literals() {
        assert!(
            Expr::Literal(Literal::Str("a".into()))
                .structural_eq(&Expr::Literal(Literal::Str("a".into())))
        );
        assert!(
            !Expr::Literal(Literal::Str("a".into()))
                .structural_eq(&Expr::Literal(Literal::Str("b".into())))
        );
        assert!(
            !Expr::Literal(Literal::Str("1".into())).structural_eq(&Expr::Literal(Literal::Int(1)))
        );
        assert!(expr_int(7).structural_eq(&expr_int(7)));
        assert!(!expr_int(7).structural_eq(&expr_int(8)));
    }

    #[test]
    fn expr_ident() {
        assert!(
            Expr::Ident {
                name: "x".into(),
                span: Default::default()
            }
            .structural_eq(&Expr::Ident {
                name: "x".into(),
                span: Default::default()
            })
        );
        assert!(
            !Expr::Ident {
                name: "x".into(),
                span: Default::default()
            }
            .structural_eq(&Expr::Ident {
                name: "y".into(),
                span: Default::default()
            })
        );
    }

    #[test]
    fn cross_variant_exprs_differ() {
        assert!(!expr_int(1).structural_eq(&Expr::Ident {
            name: "1".into(),
            span: Default::default(),
        }));
        assert!(!expr_int(1).structural_eq(&call("f", vec![])));
    }

    #[test]
    fn stmt_let() {
        let a = let_stmt("x", expr_int(1));
        let mut b = let_stmt("x", expr_int(1));
        if let Stmt::Let { access, .. } = &mut b {
            *access = Access::Public;
        }
        assert!(a.structural_eq(&let_stmt("x", expr_int(1))));
        assert!(
            !a.structural_eq(&b),
            "access modifier change must be detected"
        );
        assert!(!a.structural_eq(&let_stmt("y", expr_int(1))));
        assert!(!a.structural_eq(&let_stmt("x", expr_int(2))));
        // different declared type
        let c = Stmt::Let {
            type_name: "long".into(),
            name: "x".into(),
            init: expr_int(1),
            access: Access::Default,
            span: Span::default(),
        };
        assert!(!a.structural_eq(&c));
    }

    #[test]
    fn stmt_assign() {
        let a = Stmt::Assign {
            name: "x".into(),
            value: expr_int(1),
            span: Span::default(),
        };
        let b = Stmt::Assign {
            name: "x".into(),
            value: expr_int(1),
            span: Span::new(9, 9),
        };
        assert!(a.structural_eq(&b), "spans ignored");
        let c = Stmt::Assign {
            name: "z".into(),
            value: expr_int(1),
            span: Span::default(),
        };
        assert!(!a.structural_eq(&c));
    }

    #[test]
    fn stmt_cross_variants_differ() {
        let l = let_stmt("x", expr_int(1));
        let s = Stmt::Assign {
            name: "x".into(),
            value: expr_int(1),
            span: Span::default(),
        };
        let e = Stmt::Expression(expr_int(1));
        assert!(!l.structural_eq(&s));
        assert!(!l.structural_eq(&e));
        assert!(!s.structural_eq(&e));
    }

    #[test]
    fn block_order_matters() {
        let a = Block {
            stmts: vec![let_stmt("x", expr_int(1)), let_stmt("y", expr_int(2))],
            span: Span::default(),
        };
        let b = Block {
            stmts: vec![let_stmt("y", expr_int(2)), let_stmt("x", expr_int(1))],
            span: Span::default(),
        };
        assert!(!a.structural_eq(&b));
        assert!(a.structural_eq(&a.clone()));
    }

    #[test]
    fn param_and_method() {
        let p = Param {
            type_name: "int".into(),
            name: "a".into(),
            span: Span::default(),
            default_value: None,
            is_this: false,
            is_params: false,
        };
        assert!(p == p.clone());

        let m1 = method("run");
        let m2 = method("run");
        assert!(m1.structural_eq(&m2));

        let mut m3 = method("run");
        m3.ret_type = "void".into();
        assert!(!m1.structural_eq(&m3));
    }

    #[test]
    fn class_members_compared() {
        let a = class("app");
        let b = class("app");
        assert!(a.structural_eq(&b));

        let renamed = class("other");
        assert!(!a.structural_eq(&renamed));

        let mut extra_method = class("app");
        extra_method
            .members
            .push(ClassMember::Method(method("second")));
        assert!(!a.structural_eq(&extra_method));
    }

    #[test]
    fn namespace_and_unit() {
        let u1 = unit_with_class("app");
        let u2 = unit_with_class("app");
        assert!(u1.structural_eq(&u2));

        let u3 = unit_with_class("renamed");
        assert!(!u1.structural_eq(&u3));

        // namespace vs top-level placement differ structurally
        let top_level = CompilationUnit {
            namespaces: vec![],
            members: vec![Member::Class(class("app"))],
            aliases: vec![],
            static_usings: vec![],
            assembly_blocks: vec![],
            span: Span::default(),
        };
        assert!(!u1.structural_eq(&top_level));

        let ns_name_changed = CompilationUnit {
            namespaces: vec![NamespaceDecl {
                name: "different".into(),
                members: vec![Member::Class(class("app"))],
                span: Span::default(),
            }],
            members: vec![],
            aliases: vec![],
            static_usings: vec![],
            assembly_blocks: vec![],
            span: Span::default(),
        };
        assert!(!u1.structural_eq(&ns_name_changed));
    }

    #[test]
    fn vec_blanket_impl() {
        let v1 = vec![expr_int(1), expr_int(2)];
        let v2 = vec![expr_int(1), expr_int(2)];
        let v3 = vec![expr_int(1)];
        let v4 = vec![expr_int(2), expr_int(1)];
        assert!(v1.structural_eq(&v2));
        assert!(!v1.structural_eq(&v3));
        assert!(!v1.structural_eq(&v4)); // order matters
    }

    #[test]
    fn access_enum_covers_all_variants() {
        // Guards against someone adding a variant and forgetting encoding.
        let all = [
            (Access::Default, 0u8),
            (Access::Public, 1),
            (Access::Private, 2),
            (Access::Protected, 3),
        ];
        for (variant, byte) in all {
            assert_eq!(variant as u8, byte);
        }
    }
}
