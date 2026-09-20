//! RFC 4648 standard Base64 (`A-Z a-z 0-9 + /`, `=` padding).
//!
//! Encoding always pads; decoding accepts missing padding and ignores
//! ASCII whitespace so wrapped text survives round-trips.

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Base64Error {
    /// A character outside the alphabet (after whitespace removal).
    InvalidByte(u8),
    /// Input length (post-whitespace) cannot form whole groups.
    InvalidLength,
}

/// Encode bytes into padded Base64 text.
pub fn encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;

        out.push(ALPHABET[(triple >> 18 & 0x3F) as usize] as char);
        out.push(ALPHABET[(triple >> 12 & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[(triple >> 6 & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// Decode Base64 text; whitespace is skipped, padding optional.
pub fn decode(text: &str) -> Result<Vec<u8>, Base64Error> {
    let mut sextets = Vec::with_capacity(text.len());
    for &b in text.as_bytes() {
        match b {
            b'A'..=b'Z' => sextets.push((b - b'A') as u32),
            b'a'..=b'z' => sextets.push((b - b'a' + 26) as u32),
            b'0'..=b'9' => sextets.push((b - b'0' + 52) as u32),
            b'+' => sextets.push(62),
            b'/' => sextets.push(63),
            b'=' | b'\n' | b'\r' | b' ' | b'\t' => {}
            _ => return Err(Base64Error::InvalidByte(b)),
        }
    }
    // Padding '=' was skipped, so trim trailing incomplete sextet bits by
    // dropping the sextets that only carried padding: length % 4 in {1} is
    // invalid, {2,3} carry 1..2 bytes.
    if sextets.len() % 4 == 1 {
        return Err(Base64Error::InvalidLength);
    }

    let mut out = Vec::with_capacity(sextets.len() * 3 / 4);
    for group in sextets.chunks(4) {
        let mut triple = 0u32;
        for (i, s) in group.iter().enumerate() {
            triple |= s << (18 - i * 6);
        }
        out.push((triple >> 16) as u8);
        if group.len() > 2 {
            out.push((triple >> 8) as u8);
        }
        if group.len() > 3 {
            out.push(triple as u8);
        }
    }
    Ok(out)
}

impl std::fmt::Display for Base64Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Base64Error::InvalidByte(b) => {
                let b = (*b as char).to_string();
                write!(
                    f,
                    "{}",
                    crate::tf!("rubyc.utilities.base64.Base64Error.invalid_byte", b)
                )
            }
            Base64Error::InvalidLength => {
                write!(
                    f,
                    "{}",
                    crate::t!("rubyc.utilities.base64.Base64Error.invalid_length")
                )
            }
        }
    }
}

impl std::error::Error for Base64Error {}
