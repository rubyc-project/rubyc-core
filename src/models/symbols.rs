//! Scope and symbol models.
//!
//! Shaped for the future: class members will live in parent scopes keyed by
//! their namespace/class path, and every entry carries an [`Access`]
//! modifier ready for the semantic-enforcement pass.

use super::ast::statement::Access;

/// One declared name (variable today; fields/methods later).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol {
    pub name: String,
    /// Declared type name (`int`, `long`, ...). Semantic checks widen this
    /// into a real type tree later.
    pub type_name: String,
    /// Frame slot assigned during lowering.
    pub slot: usize,
    pub access: Access,
    /// By-reference parameter (`ref`/`out`/`in`): the home holds an
    /// address (callers box), and accesses dereference through it.
    pub is_byref: bool,
}

/// A single lexical scope.
#[derive(Debug, Default, Clone)]
pub struct Scope {
    symbols: Vec<Symbol>,
}

impl Scope {
    pub fn declare(&mut self, symbol: Symbol) -> Result<(), String> {
        // Redeclaring a name in the same scope is an error (C#-compatible,
        // CS0136). Nested scopes may still shadow an outer name — that is
        // handled by scope isolation, not by overwriting here.
        if self.symbols.iter().any(|s| s.name == symbol.name) {
            return Err(crate::tf!(
                "rubyc.models.symbols.Scope.declare.duplicate",
                symbol.name.clone()
            ));
        }
        self.symbols.push(symbol);
        Ok(())
    }

    pub fn lookup(&self, name: &str) -> Option<&Symbol> {
        self.symbols.iter().find(|s| s.name == name)
    }

    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }
}

/// Stack of scopes; lookup walks outward.
///
/// Future: namespace/class tables become outermost scopes keyed by path
/// (`demo::app`), so member resolution flows through the same mechanism.
#[derive(Debug, Default)]
pub struct ScopeStack {
    scopes: Vec<Scope>,
    /// Next frame slot to hand out (never reused across a method body).
    next_slot: usize,
}

impl ScopeStack {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self) {
        self.scopes.push(Scope::default());
    }

    pub fn pop(&mut self) {
        self.scopes.pop();
    }

    /// Depth for checkpoint/restore (see `Body` fail-soft inlining).
    pub fn depth(&self) -> usize {
        self.scopes.len()
    }

    /// Truncate to a shallower depth (drops pushed scopes).
    /// Slot numbers stay monotonic (never reused).
    pub fn truncate(&mut self, depth: usize) {
        self.scopes.truncate(depth);
    }

    pub fn declare(
        &mut self,
        name: &str,
        type_name: &str,
        access: Access,
    ) -> Result<usize, String> {
        self.declare_byref(name, type_name, access, false)
    }

    /// Declare with an explicit by-reference flag (for `ref`/`out`/`in`
    /// parameters, whose homes hold addresses).
    pub fn declare_byref(
        &mut self,
        name: &str,
        type_name: &str,
        access: Access,
        is_byref: bool,
    ) -> Result<usize, String> {
        let slot = self.next_slot;
        self.scopes
            .last_mut()
            .ok_or("no open scope")?
            .declare(Symbol {
                name: name.to_owned(),
                type_name: type_name.to_owned(),
                slot,
                access,
                is_byref,
            })?;
        self.next_slot += 1;
        Ok(slot)
    }

    /// Allocate an anonymous temp slot (compiler-generated).
    pub fn temp(&mut self) -> usize {
        let slot = self.next_slot;
        self.next_slot += 1;
        slot
    }

    pub fn lookup(&self, name: &str) -> Option<&Symbol> {
        self.scopes.iter().rev().find_map(|s| s.lookup(name))
    }

    /// Lookup in the innermost scope only. Drives hoist-aware
    /// re-declaration (a hoisted local-function name is skipped, not
    /// re-declared) without disturbing outer shadowing.
    pub fn lookup_here(&self, name: &str) -> Option<&Symbol> {
        self.scopes.last().and_then(|s| s.lookup(name))
    }

    /// Re-type an innermost-scope symbol in place. Switch-statement
    /// pattern variables (`case A tc:` + `case B tc:`) are arm-scoped
    /// in C# but share our section scope: re-binding re-types the home
    /// so later member calls resolve against the new arm's type.
    /// Returns false when no such symbol exists.
    pub fn retype_here(&mut self, name: &str, type_name: &str) -> bool {
        if let Some(sym) = self
            .scopes
            .last_mut()
            .and_then(|s| s.symbols.iter_mut().find(|x| x.name == name))
        {
            sym.type_name = type_name.to_owned();
            true
        } else {
            false
        }
    }

    /// Total slots used — drives the native frame size.
    pub fn slot_count(&self) -> usize {
        self.next_slot
    }
}
