//! The portable bytecode container: encode and decode ASTs.

pub mod decode;
pub mod encode;
pub mod format;

pub mod traits;
pub use decode::{decode, decode_payload};
pub use encode::{encode, encode_no_magic};

use crate::models::{CompilationUnit, DecodeError};
use crate::native::target::{BytecodeLoader, InputKind};

/// Load a compilation unit from raw bytes that may carry platform wrapping
/// (shebang lines). Classification is delegated to the injected loader.
pub fn load_bytes<L: BytecodeLoader + ?Sized>(
    bytes: &[u8],
    loader: &L,
) -> Result<CompilationUnit, DecodeError> {
    match loader.sniff(bytes) {
        InputKind::Source => Err(DecodeError::BadMagic),
        InputKind::Bytecode { payload_start } => decode_payload(&bytes[payload_start..]),
    }
}

/// Save a compilation unit to a file.
pub fn save(unit: &CompilationUnit, path: &std::path::Path) -> std::io::Result<()> {
    std::fs::write(path, encode(unit))
}

/// Load a compilation unit from a file.
pub fn load(path: &std::path::Path) -> Result<CompilationUnit, LoadError> {
    let bytes = std::fs::read(path)?;
    decode(&bytes).map_err(LoadError::Decode)
}

/// Failure to load a bytecode file from disk or memory.
#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Decode(crate::models::DecodeError),
}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        LoadError::Io(e)
    }
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "{e}"),
            LoadError::Decode(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for LoadError {}
