//! Visitor traversal: `walk` reaches every node kind in a compilation
//! unit, and overriding a visit method takes over its recursion.

use crate::models::visitor::{walk, Visitor};
use crate::models::{ClassDecl, CompilationUnit, Expr, Literal, NamespaceDecl};
use crate::parser;
use crate::testdata::BLOCK_FORM;

struct CountingVisitor {
    units: usize,
    namespaces: usize,
    classes: usize,
    methods: usize,
    calls: usize,
}

impl Visitor for CountingVisitor {
    fn visit_compilation_unit(&mut self, unit: &CompilationUnit) {
        self.units += 1;
        // Default recursion still runs.
        for ns in &unit.namespaces {
            self.visit_namespace(ns);
        }
        for member in &unit.members {
            self.visit_member(member);
        }
    }

    fn visit_namespace(&mut self, ns: &NamespaceDecl) {
        self.namespaces += 1;
        for member in &ns.members {
            self.visit_member(member);
        }
    }

    fn visit_class(&mut self, class: &ClassDecl) {
        self.classes += 1;
        for member in &class.members {
            self.visit_class_member(member);
        }
    }

    fn visit_method(&mut self, method: &crate::models::MethodDecl) {
        self.methods += 1;
        if let Some(body) = &method.body {
            self.visit_block(body);
        }
    }

    fn visit_expr(&mut self, expr: &Expr) {
        if let Expr::Call { args, .. } = expr {
            self.calls += 1;
            for arg in args {
                self.visit_expr(arg);
            }
        }
    }
}

#[test]
fn visitor_walks_the_whole_tree() {
    let unit = parser::parse(BLOCK_FORM).unwrap();

    let mut v = CountingVisitor {
        units: 0,
        namespaces: 0,
        classes: 0,
        methods: 0,
        calls: 0,
    };
    walk(&mut v, &unit);

    assert_eq!(
        (v.units, v.namespaces, v.classes, v.methods, v.calls),
        (1, 1, 1, 1, 1)
    );
}

#[test]
fn overriding_a_visit_method_replaces_its_recursion() {
    let unit = parser::parse(BLOCK_FORM).unwrap();

    // Literal nodes are reachable through default visit_expr.
    struct LiteralFinder {
        found: bool,
    }
    impl Visitor for LiteralFinder {
        fn visit_expr(&mut self, expr: &Expr) {
            if matches!(expr, Expr::Literal(Literal::Str(s)) if s == "Hello World!") {
                self.found = true;
            }
            // Overriding a visit method replaces its default recursion,
            // so keep walking call arguments manually.
            if let Expr::Call { args, .. } = expr {
                for arg in args {
                    self.visit_expr(arg);
                }
            }
        }
    }
    let mut f = LiteralFinder { found: false };
    f.visit_compilation_unit(&unit);
    assert!(f.found);
}
