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
mod span;
mod typeinfo;
mod types;

pub(crate) use body::Body;
pub(crate) use typeinfo::TypeRegistry;
pub(crate) use calls::clear_fn_index_cache;
pub(crate) use eval::clear_subclass_cache;
pub(crate) use eval::clear_lookup_cache;
pub(crate) use eval::PROP_READ_COUNT;
pub(crate) use calls::functions_index;
pub(crate) use eval::eval;
pub(crate) use select::class_for_spelling;
pub(crate) use select::explicit_type_args;
pub(crate) use select::score_call;
pub(crate) use types::static_type_of;
