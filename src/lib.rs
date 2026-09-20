//! RubyC compiler library.
//!
//! Pipeline stages:
//! - `parser`: source text → tokens → [`models::CompilationUnit`]
//! - `bytecode`: AST ↔ portable bytecode bytes
//! - `native`:   AST → native executable or in-process JIT execution

pub mod bytecode;
pub mod cli;
pub mod diagnostics;
pub mod diagnostics_semantic;
pub mod driver;
pub mod extensions;
pub mod i18n;
pub mod inliner;
pub mod libloader;
pub mod lints;
pub mod models;
pub mod native;
pub mod parser;
pub mod plugin;
pub mod preprocessor;
pub mod project;

pub mod testdata;
pub mod utilities;

/// Unit tests, grouped by module area under `tests/`.
#[cfg(test)]
#[path = "../tests/mod.rs"]
mod tests;

/// Common imports for consumers of the compiler API.
pub mod prelude {
    pub use crate::bytecode::traits::{FromBytecode, ToBytecode};
    pub use crate::bytecode::{decode, encode};
    pub use crate::models::CompilationUnit;
    pub use crate::models::compare::StructuralEq;
    pub use crate::native::target::Target as NativeTarget;
    pub use crate::parser::{parse, parse_file};
}
