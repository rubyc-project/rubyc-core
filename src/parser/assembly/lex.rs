//! Assembly tokenizer.

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    /// Identifier: mnemonic, register, label reference, or directive.
    Ident(String),
    /// Integer literal (already parsed; dec/0x hex/'c' char).
    Int(i64),
    Colon,
    Comma,
    LBracket,
    RBracket,
    Plus,
    Minus,
    Star,
    Newline,
}

pub fn lex(src: &str) -> Result<Vec<Tok>, String> {
    let b = src.as_bytes();
    let mut toks = Vec::new();
    let mut i = 0usize;
    while i < b.len() {
        match b[i] {
            b'\n' => {
                toks.push(Tok::Newline);
                i += 1;
            }
            b'\r' | b' ' | b'\t' => i += 1,
            b';' => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            b',' => {
                toks.push(Tok::Comma);
                i += 1;
            }
            b':' => {
                toks.push(Tok::Colon);
                i += 1;
            }
            b'[' => {
                toks.push(Tok::LBracket);
                i += 1;
            }
            b']' => {
                toks.push(Tok::RBracket);
                i += 1;
            }
            b'+' => {
                toks.push(Tok::Plus);
                i += 1;
            }
            b'*' => {
                toks.push(Tok::Star);
                i += 1;
            }
            b'-' => {
                // Negative immediate: fold into the next int.
                i += 1;
                let (v, ni) = int_at(src, i)?;
                toks.push(Tok::Int(-v));
                i = ni;
            }
            b'\'' => {
                // char literal 'x' or '\n'
                i += 1;
                let c = if b[i] == b'\\' {
                    i += 1;
                    match b[i] {
                        b'n' => b'\n',
                        b't' => b'\t',
                        b'r' => b'\r',
                        b'0' => 0,
                        b'\\' => b'\\',
                        b'\'' => b'\'',
                        other => other,
                    }
                } else {
                    b[i]
                };
                i += 1; // the char
                if i < b.len() && b[i] == b'\'' {
                    i += 1;
                }
                toks.push(Tok::Int(c as i64));
            }
            b'0'..=b'9' => {
                let (v, ni) = int_at(src, i)?;
                toks.push(Tok::Int(v));
                i = ni;
            }
            _ => {
                let start = i;
                while i < b.len()
                    && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'.')
                {
                    i += 1;
                }
                if start == i {
                    return Err(format!("unexpected byte {:#04x} at {}", b[i], i));
                }
                toks.push(Tok::Ident(src[start..i].to_owned()));
            }
        }
    }
    Ok(toks)
}

fn int_at(src: &str, mut i: usize) -> Result<(i64, usize), String> {
    let b = src.as_bytes();
    let neg_ok = false;
    let _ = neg_ok;
    if b[i] == b'0' && i + 1 < b.len() && (b[i + 1] | 0x20) == b'x' {
        i += 2;
        let start = i;
        while i < b.len() && b[i].is_ascii_hexdigit() {
            i += 1;
        }
        let v = i64::from_str_radix(&src[start..i], 16)
            .map_err(|_| "bad hex literal".to_owned())?;
        return Ok((v, i));
    }
    let start = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    src[start..i]
        .parse::<i64>()
        .map(|v| (v, i))
        .map_err(|_| "bad integer literal".to_owned())
}
