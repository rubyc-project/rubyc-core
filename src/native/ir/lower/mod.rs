//! AST to IR body lowering.
//!
//! Split by responsibility: `body` owns emission state, `eval`
//! lowers expressions and statements, `select` resolves overloads,
//! `types` computes static types, `calls` lowers call sites and
//! tracks function indices.

mod body;
mod calls;
mod eval;
mod select;
mod types;

pub(crate) use body::Body;
pub(crate) use calls::clear_fn_index_cache;
pub(crate) use eval::eval;
