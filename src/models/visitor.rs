//! Generic AST traversal.
//!
//! A [`Visitor`] receives callbacks for each node; the default
//! implementations recurse so implementors only override what they care
//! about. The bytecode encoder and future checkers are implemented as
//! visitors.

use crate::models::{
    Block, ClassDecl, ClassMember, CompilationUnit, Expr, Member, MethodDecl, NamespaceDecl, Param,
    Stmt,
};

pub trait Visitor {
    fn visit_compilation_unit(&mut self, unit: &CompilationUnit) {
        for ns in &unit.namespaces {
            self.visit_namespace(ns);
        }
        for member in &unit.members {
            self.visit_member(member);
        }
    }

    fn visit_namespace(&mut self, ns: &NamespaceDecl) {
        for member in &ns.members {
            self.visit_member(member);
        }
    }

    fn visit_member(&mut self, member: &Member) {
        match member {
            Member::Class(class) => self.visit_class(class),
            Member::Interface(iface) => self.visit_interface(iface),
            Member::Enum(_) => {}
            Member::Delegate(_) => {}
        }
    }

    fn visit_interface(&mut self, _iface: &crate::models::InterfaceDecl) {}

    fn visit_class(&mut self, class: &ClassDecl) {
        for member in &class.members {
            self.visit_class_member(member);
        }
    }

    fn visit_class_member(&mut self, member: &ClassMember) {
        match member {
            ClassMember::Method(method) => self.visit_method(method),
            // Field initializers are visited once object construction
            // lowering exists; walking them now would double-count.
            ClassMember::Field(_) => {}
            // Property accessor bodies are visited once accessor lowering
            // exists; walking them now would double-count.
            ClassMember::Property(_) => {}
            // Indexer and event bodies are visited once accessor lowering
            // exists; walking them now would double-count.
            ClassMember::Indexer(_) | ClassMember::Event(_) | ClassMember::Nested(_) => {}
        }
    }

    fn visit_method(&mut self, method: &MethodDecl) {
        for param in &method.params {
            self.visit_param(param);
        }
        if let Some(body) = &method.body {
            self.visit_block(body);
        }
    }

    fn visit_param(&mut self, _param: &Param) {}

    fn visit_block(&mut self, block: &Block) {
        for stmt in &block.stmts {
            self.visit_stmt(stmt);
        }
    }

    fn visit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Expression(expr) => self.visit_expr(expr),
            Stmt::Block { body, .. } => self.visit_block(body),
            Stmt::Fixed { declarators, body, .. } => {
                for (_, _, init) in declarators {
                    if let Some(e) = init {
                        self.visit_expr(e);
                    }
                }
                self.visit_block(body);
            }
            Stmt::Assign { value, .. } => self.visit_expr(value),
            Stmt::MemberAssign { object, value, .. } => {
                self.visit_expr(object);
                self.visit_expr(value);
            }
            Stmt::Let { init, .. } => self.visit_expr(init),
            Stmt::Return { value, .. } => {
                if let Some(expr) = value {
                    self.visit_expr(expr);
                }
            }
            Stmt::If {
                cond,
                then_block,
                else_block,
                ..
            } => {
                self.visit_expr(cond);
                self.visit_block(then_block);
                if let Some(b) = else_block {
                    self.visit_block(b);
                }
            }
            Stmt::While { cond, body, .. } => {
                self.visit_expr(cond);
                self.visit_block(body);
            }
            Stmt::DoWhile { body, cond, .. } => {
                self.visit_block(body);
                self.visit_expr(cond);
            }
            Stmt::For {
                init,
                cond,
                step,
                body,
                ..
            } => {
                if let Some(s) = init {
                    self.visit_stmt(s);
                }
                if let Some(c) = cond {
                    self.visit_expr(c);
                }
                if let Some(s) = step {
                    self.visit_stmt(s);
                }
                self.visit_block(body);
            }
            Stmt::Switch {
                subject,
                arms,
                default_body,
                ..
            } => {
                self.visit_expr(subject);
                for arm in arms {
                    for label in &arm.labels {
                        self.visit_expr(label);
                    }
                    self.visit_block(&arm.body);
                }
                if let Some(b) = default_body {
                    self.visit_block(b);
                }
            }
            Stmt::Try {
                body,
                catches,
                finally_block,
                ..
            } => {
                self.visit_block(body);
                for c in catches {
                    self.visit_block(&c.body);
                }
                if let Some(b) = finally_block {
                    self.visit_block(b);
                }
            }
            Stmt::Throw { expr, .. } => {
                if let Some(e) = expr {
                    self.visit_expr(e);
                }
            }
            Stmt::LocalFunction { body, .. } => {
                self.visit_block(body);
            }
            Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::Goto { .. } | Stmt::Label { .. } => {}
        }
    }

    fn visit_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Call { args, .. } => {
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            Expr::Ternary {
                cond,
                then_expr,
                else_expr,
                ..
            } => {
                self.visit_expr(cond);
                self.visit_expr(then_expr);
                self.visit_expr(else_expr);
            }
            _ => {}
        }
    }
}

/// Walk a whole compilation unit with the given visitor.
pub fn walk<V: Visitor>(visitor: &mut V, unit: &CompilationUnit) {
    visitor.visit_compilation_unit(unit);
}
