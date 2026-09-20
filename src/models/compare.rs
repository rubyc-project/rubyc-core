//! Span-insensitive comparison of syntax trees.
//!
/// Two trees are structurally equal when every node kind, name and child
/// matches, ignoring source locations. Used by round-trip tests to prove a
/// decode(encode(tree)) preserves program meaning.
pub trait StructuralEq {
    fn structural_eq(&self, other: &Self) -> bool;
}
