//! Unified error type for the cross-OS library loader.

/// Errors that can occur when loading a native library or resolving a symbol.
#[derive(Debug)]
pub enum LibError {
    /// The file could not be read.
    Io(std::io::Error),
    /// The bytes did not match the expected binary format (bad magic, truncated, etc.).
    BadImage(String),
    /// The library could not be loaded by the OS loader.
    LoadFailed(String),
    /// A required exported symbol was not found.
    MissingSymbol(String),
    /// The host platform is not supported by this loader.
    UnsupportedPlatform,
}

impl std::fmt::Display for LibError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LibError::Io(err) => write!(f, "io error: {err}"),
            LibError::BadImage(msg) => write!(f, "bad image: {msg}"),
            LibError::LoadFailed(msg) => write!(f, "load failed: {msg}"),
            LibError::MissingSymbol(name) => write!(f, "missing exported symbol: {name}"),
            LibError::UnsupportedPlatform => {
                write!(f, "this platform is not supported by the library loader")
            }
        }
    }
}

impl std::error::Error for LibError {}

impl From<std::io::Error> for LibError {
    fn from(err: std::io::Error) -> Self {
        LibError::Io(err)
    }
}
