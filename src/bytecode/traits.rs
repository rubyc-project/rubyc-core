//! Serialization contracts for compiler models.
//!
//! The concrete encoding lives in `crate::bytecode`; these traits expose it
//! through the model types so callers never import the backend directly.

use crate::models::{CompilationUnit, DecodeError};

/// Types that can be written to the portable bytecode representation.
pub trait ToBytecode {
    fn to_bytecode(&self) -> Vec<u8>;
    /// Encode without the RubyC preamble (`--no-magic`). Default is the
    /// standard (preamble) encoding; backends override to emit raw payload.
    fn to_bytecode_no_magic(&self) -> Vec<u8> {
        self.to_bytecode()
    }
}

/// Types that can be reconstructed from bytecode bytes.
pub trait FromBytecode: Sized {
    fn from_bytecode(bytes: &[u8]) -> Result<Self, DecodeError>;
}

impl ToBytecode for CompilationUnit {
    fn to_bytecode(&self) -> Vec<u8> {
        crate::bytecode::encode(self)
    }
    fn to_bytecode_no_magic(&self) -> Vec<u8> {
        crate::bytecode::encode_no_magic(self)
    }
}

impl FromBytecode for CompilationUnit {
    fn from_bytecode(bytes: &[u8]) -> Result<Self, DecodeError> {
        crate::bytecode::decode(bytes)
    }
}
