//! Shared file preamble for every RubyC artifact.
//!
//! Every bytecode file and every RubyC native executable begins with
//! a short jump over the project magic — the dog emoji (Ruby is the dog)
//! followed by the letter C — so that accidental execution of a data file
//! skips the magic bytes instead of decoding them as instructions:
//!
//! ```text
//! offset 0: EB 08                     jmp rel8 +8  (lands after the magic)
//! offset 2: F0 9F 90 95 F0 9F 87 A8   🐕🇨
//! ```
//!
//! Executable formats whose loaders require their own magic at offset 0
//! (e.g. ELF's `\x7fELF`) place this preamble at the entry point instead of
//! the start of the file; [`parse`] accepts both placements.

/// `jmp rel8 +8`: skip the 8 magic bytes when executed.
pub const JUMP_OVER_MAGIC: [u8; 2] = [0xEB, 0x08];

/// 🐕🇨 — Ruby the dog, and C.
pub const MAGIC: [u8; 8] = [0xF0, 0x9F, 0x90, 0x95, 0xF0, 0x9F, 0x87, 0xA8];

/// Total size of jump + magic.
pub const PREAMBLE_LEN: usize = JUMP_OVER_MAGIC.len() + MAGIC.len();

/// What kind of payload follows the preamble.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// AST bytecode consumed by `rubyc run` / the JIT.
    Bytecode,
    /// A native executable image (ELF on Linux).
    NativeExecutable,
    /// A shared library (ELF ET_DYN on Linux); preamble sits at offset 64,
    /// right after the ELF header, since ELF owns offset 0.
    SharedLibrary,
    /// A shared library carrying its type fragment inline: after the
    /// 13-byte preamble comes a u64 LE table length, then the fragment
    /// bytes, then the payload. Old readers reject the kind byte
    /// instead of misparsing, so no format-version bump was needed.
    SharedLibraryWithTypes,
    /// Lowered IR (`Program`) — the plugin ABI boundary (decision D3).
    Ir,
}

impl Kind {
    pub fn to_byte(self) -> u8 {
        match self {
            Kind::Bytecode => 0x01,
            Kind::NativeExecutable => 0x02,
            Kind::SharedLibrary => 0x03,
            Kind::SharedLibraryWithTypes => 0x05,
            Kind::Ir => 0x04,
        }
    }

    pub fn from_byte(b: u8) -> Option<Kind> {
        match b {
            0x01 => Some(Kind::Bytecode),
            0x02 => Some(Kind::NativeExecutable),
            0x03 => Some(Kind::SharedLibrary),
            0x05 => Some(Kind::SharedLibraryWithTypes),
            0x04 => Some(Kind::Ir),
            _ => None,
        }
    }
}

/// Serialize the 13-byte header: jump, magic, kind byte, LE version.
pub fn build(kind: Kind, version: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(PREAMBLE_LEN + 3);
    out.extend_from_slice(&JUMP_OVER_MAGIC);
    out.extend_from_slice(&MAGIC);
    out.push(kind.to_byte());
    out.extend_from_slice(&version.to_le_bytes());
    out
}

/// Result of parsing a preamble.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Parsed {
    pub kind: Kind,
    pub version: u16,
    /// Offset of the first payload byte: 13, or 13 + 8 + table length
    /// for [`Kind::SharedLibraryWithTypes`].
    pub payload_start: usize,
}

/// Parse a standard (file-start) preamble.
///
/// Returns an error string describing the first problem found; callers map
/// that onto their own error type.
pub fn parse(bytes: &[u8]) -> Result<Parsed, String> {
    if bytes.len() < PREAMBLE_LEN + 3 {
        return Err("file too short for RubyC header".into());
    }
    if bytes[..2] != JUMP_OVER_MAGIC {
        return Err("missing RubyC jump preamble".into());
    }
    if bytes[2..10] != MAGIC {
        return Err("missing RubyC dog magic".into());
    }
    let kind = Kind::from_byte(bytes[10]).ok_or("unknown payload kind")?;
    let version = u16::from_le_bytes([bytes[11], bytes[12]]);
    if kind == Kind::SharedLibraryWithTypes {
        if bytes.len() < PREAMBLE_LEN + 3 + 8 {
            return Err("file too short for embedded type table length".into());
        }
        let len = u64::from_le_bytes(
            bytes[PREAMBLE_LEN + 3..PREAMBLE_LEN + 3 + 8]
                .try_into()
                .map_err(|_| "file too short for embedded type table length")?,
        ) as usize;
        let payload_start = PREAMBLE_LEN + 3 + 8 + len;
        if payload_start > bytes.len() {
            return Err("embedded type table overruns the file".into());
        }
        return Ok(Parsed {
            kind,
            version,
            payload_start,
        });
    }
    Ok(Parsed {
        kind,
        version,
        payload_start: PREAMBLE_LEN + 3,
    })
}

/// Locate the preamble in an executable image where a foreign loader owns
/// the first bytes (e.g. `\x7fELF`): scan for jump+magic at any offset.
pub fn parse_embedded(bytes: &[u8]) -> Option<Parsed> {
    (0..bytes.len().saturating_sub(PREAMBLE_LEN + 3)).find_map(|i| parse(&bytes[i..]).ok())
}
