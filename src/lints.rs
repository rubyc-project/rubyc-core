//! Lint infrastructure: named rules with configurable levels, in the spirit
//! of rustc's `-W`/`-D`/`-A` flags and Cargo's `[lints]` tables.
//!
//! Rules carry stable ids (`W00xx`) so users can target them on the CLI:
//!
//! ```text
//! rubyc build --deny unused-local app.rc
//! rubyc run --allow W0002 script.rc
//! ```
//!
//! Warnings render through the shared diagnostic pipeline and never fail a
//! build on their own — only rules upgraded to `Deny` do.

use crate::models::diagnostic::{Diagnostic, ScopeContext, Severity};
use crate::models::span::Span;
use crate::models::{Access, ClassDecl, ClassMember, CompilationUnit, Expr, Stmt};

/// Severity policy for one rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// Rule is off.
    Allow,
    /// Report as a warning; the build succeeds.
    Warn,
    /// Report as an error; the build fails.
    Deny,
}

/// One lint rule.
#[derive(Debug, Clone, Copy)]
pub struct LintRule {
    /// Stable id usable on the CLI (`W0001`).
    pub id: &'static str,
    /// Human-readable name (`unused-local`).
    pub name: &'static str,
    /// Level applied when no CLI override matches.
    pub default: Level,
}

/// The rule registry. New lints join here; ids are never reused.
pub const RULES: &[LintRule] = &[
    LintRule {
        id: "W0001",
        name: "unused-local",
        default: Level::Warn,
    },
    LintRule {
        id: "W0002",
        name: "unused-private-member",
        default: Level::Warn,
    },
];

impl LintRule {
    pub fn by_key(key: &str) -> Option<&'static LintRule> {
        RULES.iter().find(|r| r.id == key || r.name == key)
    }
}

/// Resolved per-rule configuration.
#[derive(Debug, Clone, Default)]
pub struct LintConfig {
    /// `(rule id or name, level)` from CLI flags; later flags win.
    pub overrides: Vec<(String, Level)>,
}

impl LintConfig {
    pub fn set(&mut self, key: &str, level: Level) {
        self.overrides.push((key.to_owned(), level));
    }

    pub fn level_for(&self, id: &str) -> Level {
        let Some(rule) = RULES.iter().find(|r| r.id == id) else {
            return Level::Allow;
        };
        let mut level = rule.default;
        for (key, lv) in &self.overrides {
            if key == rule.id || key == rule.name {
                level = *lv;
            }
        }
        level
    }
}

fn diagnostic_for(
    config: &LintConfig,
    id: &'static str,
    message: String,
    span: Span,
    context: ScopeContext,
) -> Option<Diagnostic> {
    let severity = match config.level_for(id) {
        Level::Allow => return None,
        Level::Warn => Severity::Warning,
        Level::Deny => Severity::Error,
    };
    Some(
        Diagnostic::new(id, severity, message)
            .at(span)
            .in_context(context),
    )
}

/// Run every enabled rule over `unit`, producing diagnostics whose severity
/// reflects the configured level (`Allow` produces nothing).
pub fn analyze(unit: &CompilationUnit, config: &LintConfig) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    out.extend(unused_locals(unit, config));
    out.extend(unused_private_members(unit, config));
    out
}

fn ns_members(unit: &CompilationUnit) -> Vec<&crate::models::Member> {
    unit.all_members()
}

fn class_context(unit: &CompilationUnit, class: Option<&String>) -> ScopeContext {
    let namespace = match class {
        Some(c) => unit.namespace_name_of(c),
        None => None,
    };
    ScopeContext {
        namespace,
        class: class.cloned(),
        function: None,
    }
}

// ---------- W0001: unused local ----------

/// Locals declared but never read. Assignment targets alone do not count as
/// uses (`x = 5` with no subsequent read is still dead). Parameters and
/// `this` are not tracked; name matching inside one method body keeps this
/// conservative rather than scope-exact.
fn unused_locals(unit: &CompilationUnit, config: &LintConfig) -> Vec<Diagnostic> {
    let mut diags = Vec::new();
    for member in ns_members(unit) {
        let crate::models::Member::Class(class) = member else {
            continue;
        };
        for cm in &class.members {
            let ClassMember::Method(method) = cm else {
                continue;
            };
            let Some(body) = &method.body else {
                continue;
            };
            let mut declared: Vec<(String, Span)> = Vec::new();
            collect_lets(body, &mut declared);
            if declared.is_empty() {
                continue;
            }
            let mut reads: Vec<String> = Vec::new();
            collect_reads_block(body, &mut reads);
            for (name, span) in declared {
                if reads.contains(&name) {
                    continue;
                }
                let ctx = ScopeContext {
                    namespace: unit.namespace_name_of(&class.name),
                    class: Some(class.name.clone()),
                    function: Some(method.name.clone()),
                };
                if let Some(d) = diagnostic_for(
                    config,
                    "W0001",
                    crate::tf!("rubyc.lint.W0001", name.clone()),
                    span,
                    ctx,
                ) {
                    diags.push(d);
                }
            }
        }
    }
    diags
}

fn collect_lets(block: &crate::models::Block, out: &mut Vec<(String, Span)>) {
    for stmt in &block.stmts {
        if let Stmt::Let { name, span, .. } = stmt {
            out.push((name.clone(), *span))
        }
    }
}

/// Identifier reads inside a block. Assignment targets are excluded (a pure
/// write is not an observation). Control-flow statements are walked so
/// reads inside nested bodies count.
fn collect_reads_block(block: &crate::models::Block, out: &mut Vec<String>) {
    for stmt in &block.stmts {
        match stmt {
            Stmt::Let { init, .. } | Stmt::UsingDecl { init, .. } => {
                collect_reads_expr(init, out)
            }
            Stmt::Assign { value, .. } => collect_reads_expr(value, out),
            Stmt::MemberAssign { object, value, .. } => {
                collect_reads_expr(object, out);
                collect_reads_expr(value, out);
            }
            Stmt::Expression(e) => collect_reads_expr(e, out),
            Stmt::CheckedBlock { body, .. } => collect_reads_block(body, out),
            Stmt::Block { body, .. } => collect_reads_block(body, out),
            Stmt::Fixed { declarators, body, .. } => {
                for (_, _, init) in declarators {
                    if let Some(e) = init {
                        collect_reads_expr(e, out);
                    }
                }
                collect_reads_block(body, out);
            }
            Stmt::Return { value: Some(e), .. } => collect_reads_expr(e, out),
            Stmt::Return { value: None, .. } => {}
            Stmt::If {
                cond,
                then_block,
                else_block,
                ..
            } => {
                collect_reads_expr(cond, out);
                collect_reads_block(then_block, out);
                if let Some(b) = else_block {
                    collect_reads_block(b, out);
                }
            }
            Stmt::While { cond, body, .. } => {
                collect_reads_expr(cond, out);
                collect_reads_block(body, out);
            }
            Stmt::DoWhile { body, cond, .. } => {
                collect_reads_block(body, out);
                collect_reads_expr(cond, out);
            }
            Stmt::For {
                init,
                cond,
                step,
                body,
                ..
            } => {
                if let Some(s) = init {
                    collect_reads_stmt(s, out);
                }
                if let Some(c) = cond {
                    collect_reads_expr(c, out);
                }
                if let Some(s) = step {
                    collect_reads_stmt(s, out);
                }
                collect_reads_block(body, out);
            }
            Stmt::Switch {
                subject,
                arms,
                default_body,
                ..
            } => {
                collect_reads_expr(subject, out);
                for arm in arms {
                    collect_reads_block(&arm.body, out);
                }
                if let Some(b) = default_body {
                    collect_reads_block(b, out);
                }
            }
            Stmt::Try {
                body,
                catches,
                finally_block,
                ..
            } => {
                collect_reads_block(body, out);
                for c in catches {
                    collect_reads_block(&c.body, out);
                }
                if let Some(b) = finally_block {
                    collect_reads_block(b, out);
                }
            }
            Stmt::Throw { expr, .. } => {
                if let Some(e) = expr {
                    collect_reads_expr(e, out);
                }
            }
            Stmt::LocalFunction { params, body, .. } => {
                for p in params {
                    if let Some(d) = &p.default_value {
                        collect_reads_expr(d, out);
                    }
                }
                collect_reads_block(body, out);
            }
            Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::Label { .. } => {}
            Stmt::Goto { case_expr, .. } => {
                if let Some(e) = case_expr {
                    collect_reads_expr(e, out);
                }
            }
        }
    }
}

fn collect_reads_stmt(stmt: &Stmt, out: &mut Vec<String>) {
    // Reuse the block walker for single statements.
    let shim = crate::models::Block {
        stmts: vec![stmt.clone()],
        span: Default::default(),
    };
    collect_reads_block(&shim, out);
}

fn collect_reads_expr(expr: &Expr, out: &mut Vec<String>) {
    match expr {
        Expr::NamedArg { value, .. } => collect_reads_expr(value, out),
        Expr::Checked { inner, .. } => collect_reads_expr(inner, out),
            // A lambda body is code that runs later, but it is still
            // code: its reads belong to the enclosing method's set (the
            // closure reads them), so walk it rather than skipping it.
            Expr::Lambda { body, .. } => match body {
                crate::models::ast::expression::LambdaBody::Expr(e) => {
                    collect_reads_expr(e, out)
                }
                crate::models::ast::expression::LambdaBody::Block(b) => {
                    for st in &b.stmts {
                        collect_reads_stmt(st, out);
                    }
                }
            },
        Expr::Binary { lhs, rhs, .. } => {
            collect_reads_expr(lhs, out);
            collect_reads_expr(rhs, out);
        }
        Expr::Unary { inner, .. } => collect_reads_expr(inner, out),
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            collect_reads_expr(cond, out);
            collect_reads_expr(then_expr, out);
            collect_reads_expr(else_expr, out);
        }
        Expr::NewArray { size, .. } => collect_reads_expr(size, out),
        Expr::Index { object, index, .. } => {
            collect_reads_expr(object, out);
            collect_reads_expr(index, out);
        }
        Expr::Call { callee, args, .. } => {
            // A local invoked as `f(...)` is READ: without the callee a
            // delegate held in a local was reported as "declared but
            // never used" while every call through it worked.
            //
            // An explicit generic suffix is part of the CALL, not the
            // name: `One<string>("x")` names `One`, and recording
            // `One<string>` reported the method unused.
            out.push(
                callee
                    .split('<')
                    .next()
                    .unwrap_or(callee)
                    .rsplit('.')
                    .next()
                    .unwrap_or(callee)
                    .to_owned(),
            );
            for a in args {
                collect_reads_expr(a, out);
            }
        }
        Expr::New { args, .. } => {
            for a in args {
                collect_reads_expr(a, out);
            }
        }
        Expr::Member { object, .. } => collect_reads_expr(object, out),
        Expr::MethodCall { object, args, .. } => {
            collect_reads_expr(object, out);
            for a in args {
                collect_reads_expr(a, out);
            }
        }
        Expr::Cast { expr, .. } => collect_reads_expr(expr, out),
        Expr::Default { .. } => {}
        Expr::Assign { target, value, .. } => {
            collect_reads_expr(target, out);
            collect_reads_expr(value, out);
        }
        Expr::AssignComplex { lhs, value, .. } => {
            collect_reads_expr(lhs, out);
            collect_reads_expr(value, out);
        }
        Expr::PostInc { target, .. } | Expr::PostDec { target, .. } => {
            collect_reads_expr(target, out);
        }
        Expr::SwitchExpr { subject, arms, .. } => {
            collect_reads_expr(subject, out);
            for arm in arms {
                collect_reads_expr(&arm.body, out);
            }
        }
        Expr::Await { inner, .. } => collect_reads_expr(inner, out),
        Expr::InterpolatedString { parts, .. } => {
            for part in parts {
                if let crate::models::InterpPart::Hole(h) = part {
                    collect_reads_expr(h, out);
                }
            }
        }
        Expr::Ident { name: n, .. } => out.push(n.clone()),
        Expr::Literal(_) => {}
    }
}

// ---------- W0002: unused private member ----------

/// Private/default members never referenced anywhere in the unit. The check
/// is deliberately conservative — any name match anywhere counts as a use,
/// so shadowing can hide a genuinely dead member but a live one is never
/// flagged.
fn unused_private_members(unit: &CompilationUnit, config: &LintConfig) -> Vec<Diagnostic> {
    let mut diags = Vec::new();

    // Every member name referenced anywhere in any body (plus locals).
    let mut referenced: Vec<String> = Vec::new();
    for member in ns_members(unit) {
        let crate::models::Member::Class(class) = member else {
            continue;
        };
        for cm in &class.members {
            if let ClassMember::Method(method) = cm
                && let Some(body) = &method.body
            {
                collect_name_refs_block(body, &mut referenced);
            }
        }
    }

    for member in ns_members(unit) {
        let crate::models::Member::Class(class) = member else {
            continue;
        };
        for cm in &class.members {
            let (name, access, is_field) = match cm {
                ClassMember::Field(f) => (f.name.clone(), f.access, true),
                ClassMember::Property(p) => (p.name.clone(), p.access, true),
                ClassMember::Indexer(_) | ClassMember::Event(_) | ClassMember::Nested(_) => continue,
                ClassMember::Method(m) => (m.name.clone(), m.access, false),
            };
            let is_private = matches!(access, Access::Private | Access::Default);
            // An explicit interface implementation (`int I.M()`) is
            // registered under its qualified spelling and is only
            // reachable through interface dispatch, so no call site
            // ever names it. C# does not consider it dead; nor do we.
            let is_explicit_impl = name.contains('.');
            // Entry points are roots: never flagged as unused.
            if !is_private
                || is_explicit_impl
                || referenced.contains(&name)
                || name == "main"
                || name == "Main"
            {
                continue;
            }
            let kind = if is_field { "field" } else { "method" };
            if let Some(d) = diagnostic_for(
                config,
                "W0002",
                crate::tf!(
                    "rubyc.lint.W0002",
                    kind.to_owned(),
                    class.name.clone(),
                    name
                ),
                Span::default(),
                class_context(unit, Some(&class.name)),
            ) {
                diags.push(d);
            }
        }
    }
    diags
}

fn collect_name_refs_block(block: &crate::models::Block, out: &mut Vec<String>) {
    for stmt in &block.stmts {
        match stmt {
            Stmt::Let { name, init, .. } | Stmt::UsingDecl { name, init, .. } => {
                out.push(name.clone());
                collect_name_refs_expr(init, out);
            }
            Stmt::Assign { name, value, .. } => {
                out.push(name.clone());
                collect_name_refs_expr(value, out);
            }
            Stmt::MemberAssign {
                object,
                name,
                value,
                ..
            } => {
                collect_name_refs_expr(object, out);
                out.push(name.clone());
                collect_name_refs_expr(value, out);
            }
            Stmt::Expression(e) => collect_name_refs_expr(e, out),
            Stmt::CheckedBlock { body, .. } => collect_name_refs_block(body, out),
            Stmt::Block { body, .. } => collect_name_refs_block(body, out),
            Stmt::Fixed { declarators, body, .. } => {
                for (_, _, init) in declarators {
                    if let Some(e) = init {
                        collect_name_refs_expr(e, out);
                    }
                }
                collect_name_refs_block(body, out);
            }
            Stmt::Return { value: Some(e), .. } => collect_name_refs_expr(e, out),
            Stmt::Return { value: None, .. } => {}
            Stmt::If {
                cond,
                then_block,
                else_block,
                ..
            } => {
                collect_name_refs_expr(cond, out);
                collect_name_refs_block(then_block, out);
                if let Some(b) = else_block {
                    collect_name_refs_block(b, out);
                }
            }
            Stmt::While { cond, body, .. } => {
                collect_name_refs_expr(cond, out);
                collect_name_refs_block(body, out);
            }
            Stmt::DoWhile { body, cond, .. } => {
                collect_name_refs_block(body, out);
                collect_name_refs_expr(cond, out);
            }
            Stmt::For {
                init,
                cond,
                step,
                body,
                ..
            } => {
                if let Some(s) = init {
                    collect_name_refs_stmt(s, out);
                }
                if let Some(c) = cond {
                    collect_name_refs_expr(c, out);
                }
                if let Some(s) = step {
                    collect_name_refs_stmt(s, out);
                }
                collect_name_refs_block(body, out);
            }
            Stmt::Switch {
                subject,
                arms,
                default_body,
                ..
            } => {
                collect_name_refs_expr(subject, out);
                for arm in arms {
                    collect_name_refs_block(&arm.body, out);
                }
                if let Some(b) = default_body {
                    collect_name_refs_block(b, out);
                }
            }
            Stmt::Try {
                body,
                catches,
                finally_block,
                ..
            } => {
                collect_name_refs_block(body, out);
                for c in catches {
                    if let Some(name) = &c.name {
                        out.push(name.clone());
                    }
                    collect_name_refs_block(&c.body, out);
                }
                if let Some(b) = finally_block {
                    collect_name_refs_block(b, out);
                }
            }
            Stmt::Throw { expr, .. } => {
                if let Some(e) = expr {
                    collect_name_refs_expr(e, out);
                }
            }
            Stmt::LocalFunction { name, params, body, .. } => {
                out.push(name.clone());
                for p in params {
                    if let Some(d) = &p.default_value {
                        collect_name_refs_expr(d, out);
                    }
                }
                collect_name_refs_block(body, out);
            }
            Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::Label { .. } => {}
            Stmt::Goto { case_expr, .. } => {
                if let Some(e) = case_expr {
                    collect_name_refs_expr(e, out);
                }
            }
        }
    }
}

fn collect_name_refs_stmt(stmt: &Stmt, out: &mut Vec<String>) {
    let shim = crate::models::Block {
        stmts: vec![stmt.clone()],
        span: Default::default(),
    };
    collect_name_refs_block(&shim, out);
}

fn collect_name_refs_expr(expr: &Expr, out: &mut Vec<String>) {
    match expr {
        Expr::NamedArg { value, .. } => collect_name_refs_expr(value, out),
        Expr::Checked { inner, .. } => collect_name_refs_expr(inner, out),
        // A lambda body is code that runs later, but its name references
        // are real references for the enclosing method (a closure reads
        // them), so they are collected like any other expression's.
        Expr::Lambda { body, .. } => match body {
            crate::models::ast::expression::LambdaBody::Expr(e) => {
                collect_name_refs_expr(e, out)
            }
            crate::models::ast::expression::LambdaBody::Block(b) => {
                for st in &b.stmts {
                    collect_name_refs_stmt(st, out);
                }
            }
        },
        Expr::NewArray { size, .. } => collect_name_refs_expr(size, out),
        Expr::Index { object, index, .. } => {
            collect_name_refs_expr(object, out);
            collect_name_refs_expr(index, out);
        }
        Expr::Binary { lhs, rhs, .. } => {
            collect_name_refs_expr(lhs, out);
            collect_name_refs_expr(rhs, out);
        }
        Expr::Unary { inner, .. } => collect_name_refs_expr(inner, out),
        Expr::Ternary {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            collect_name_refs_expr(cond, out);
            collect_name_refs_expr(then_expr, out);
            collect_name_refs_expr(else_expr, out);
        }
        Expr::Call { callee, args, .. } => {
            out.push(
                callee
                    .split('<')
                    .next()
                    .unwrap_or(callee)
                    .rsplit('.')
                    .next()
                    .unwrap_or(callee)
                    .to_owned(),
            );
            for a in args {
                collect_name_refs_expr(a, out);
            }
        }
        Expr::New { args, .. } => {
            for a in args {
                collect_name_refs_expr(a, out);
            }
        }
        Expr::Member { object, name, .. } => {
            collect_name_refs_expr(object, out);
            out.push(name.clone());
        }
        Expr::MethodCall {
            object, name, args, ..
        } => {
            collect_name_refs_expr(object, out);
            // `M<T>()`: the suffix names the instantiation, not the
            // member (same false "never used" as the Call case).
            out.push(
                name
                    .split('<')
                    .next()
                    .unwrap_or(name)
                    .rsplit('.')
                    .next()
                    .unwrap_or(name)
                    .to_owned(),
            );
            for a in args {
                collect_name_refs_expr(a, out);
            }
        }
        Expr::Cast { expr, .. } => collect_name_refs_expr(expr, out),
        Expr::Default { .. } => {}
        Expr::Assign { target, value, .. } => {
            collect_name_refs_expr(target, out);
            collect_name_refs_expr(value, out);
        }
        Expr::AssignComplex { lhs, value, .. } => {
            collect_name_refs_expr(lhs, out);
            collect_name_refs_expr(value, out);
        }
        Expr::PostInc { target, .. } | Expr::PostDec { target, .. } => {
            collect_name_refs_expr(target, out);
        }
        Expr::SwitchExpr { subject, arms, .. } => {
            collect_name_refs_expr(subject, out);
            for arm in arms {
                collect_name_refs_expr(&arm.body, out);
            }
        }
        Expr::Await { inner, .. } => collect_name_refs_expr(inner, out),
        Expr::InterpolatedString { parts, .. } => {
            for part in parts {
                if let crate::models::InterpPart::Hole(h) = part {
                    collect_name_refs_expr(h, out);
                }
            }
        }
        Expr::Ident { name: n, .. } => out.push(n.clone()),
        Expr::Literal(_) => {}
    }
}

/// Expose class declarations for future rules that need layout info.
pub fn classes_of(unit: &CompilationUnit) -> Vec<&ClassDecl> {
    ns_members(unit)
        .iter()
        .filter_map(|m| match m {
            crate::models::Member::Class(c) => Some(c),
            _ => None,
        })
        .collect()
}
