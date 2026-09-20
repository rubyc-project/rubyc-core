//! RubyC's built-in x86-64 assembler.
//!
//! NASM-flavored Intel syntax subset, two-pass encoding.

mod encode;
mod lex;
pub mod parse;

pub use parse::{AsmItem, AsmOperand, AsmStmt, Reg};

/// Assemble source into machine bytes.
pub fn assemble(source: &str) -> Result<Vec<u8>, String> {
    let items = parse::parse(source)?;
    encode::encode(&items)
}

/// Assemble and return `(code, symbols)`.
type AssembleWithSymbols = Result<(Vec<u8>, Vec<(String, usize)>), String>;

pub fn assemble_with_symbols(source: &str) -> AssembleWithSymbols {
    let items = parse::parse(source)?;
    encode::encode_with_symbols(&items)
}
