//! GNU archive (`.a`) reader/writer for static libraries.
//!
//! Each member holds one library's serialized `MachineCode`; the static
//! link step merges members into the final image. No SysV symbol index:
//! our linker resolves members itself, and plain archives keep GNU `ar`
//! (`t`/`x`) interoperable.
//!
//! Layout: `!<arch>\n`, then per member a 60-byte header
//! (name[16] mtime[12] uid[6] gid[6] mode[8] size[10] `` `\n ``) plus
//! even-padded bytes. Names over 15 bytes go through the `//` string
//! table as `/OFFSET`.

/// Archive magic.
pub const MAGIC: &[u8; 8] = b"!<arch>\n";
const HEADER_LEN: usize = 60;

/// One archive member: file name plus raw bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Member {
    pub name: String,
    pub bytes: Vec<u8>,
}

fn pad_field(out: &mut Vec<u8>, text: &str, width: usize) {
    let bytes = text.as_bytes();
    let n = bytes.len().min(width);
    out.extend_from_slice(&bytes[..n]);
    out.resize(out.len() + (width - n), b' ');
}

/// Encode members into an archive. Long names (over 15 bytes including the
/// trailing `/`) are stored in a leading `//` string table.
pub fn write(members: &[Member]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    // Collect long names first so member headers can reference offsets.
    let mut table = Vec::<u8>::new();
    let mut offsets: Vec<Option<u64>> = Vec::with_capacity(members.len());
    for m in members {
        let short = format!("{}/", m.name);
        if short.len() <= 16 {
            offsets.push(None);
        } else {
            offsets.push(Some(table.len() as u64));
            table.extend_from_slice(m.name.as_bytes());
            table.push(b'/');
            table.push(b'\n');
        }
    }
    if !table.is_empty() {
        if table.len() % 2 == 1 {
            table.push(b'\n');
        }
        push_member(&mut out, "//", &table, 0);
    }
    for (m, off) in members.iter().zip(offsets.iter()) {
        let header_name = match off {
            Some(o) => format!("/{o}"),
            None => format!("{}/", m.name),
        };
        push_member(&mut out, &header_name, &m.bytes, 0o100644);
    }
    out
}

fn push_member(out: &mut Vec<u8>, name: &str, bytes: &[u8], mode: u32) {
    pad_field(out, name, 16);
    pad_field(out, "0", 12);
    pad_field(out, "0", 6);
    pad_field(out, "0", 6);
    pad_field(out, &format!("{mode:o}"), 8);
    pad_field(out, &bytes.len().to_string(), 10);
    out.extend_from_slice(b"\x60\x0a");
    out.extend_from_slice(bytes);
    if bytes.len() % 2 == 1 {
        out.push(b'\n');
    }
}

/// Decode an archive into its members. The `//` string table and any `/`
/// symbol index are consumed internally, never returned as members.
pub fn read(bytes: &[u8]) -> Result<Vec<Member>, String> {
    if bytes.len() < MAGIC.len() || &bytes[..MAGIC.len()] != MAGIC {
        return Err("not a GNU archive (bad magic)".into());
    }
    let mut members = Vec::new();
    let mut strings: &[u8] = &[];
    let mut at = MAGIC.len();
    while at < bytes.len() {
        if at + HEADER_LEN > bytes.len() {
            return Err("truncated member header".into());
        }
        let header = &bytes[at..at + HEADER_LEN];
        if &header[58..60] != b"\x60\x0a" {
            return Err(format!("bad member header terminator at offset {at}"));
        }
        let raw_name = std::str::from_utf8(&header[..16])
            .map_err(|_| format!("non-UTF8 member name at offset {at}"))?;
        let size: usize = std::str::from_utf8(&header[48..58])
            .map_err(|_| format!("non-UTF8 member size at offset {at}"))?
            .trim()
            .parse()
            .map_err(|_| format!("bad member size at offset {at}"))?;
        at += HEADER_LEN;
        if at + size > bytes.len() {
            return Err("truncated member body".into());
        }
        let body = &bytes[at..at + size];
        at += size;
        if at % 2 == 1 {
            // Odd bodies are newline-padded to even alignment.
            if at >= bytes.len() || bytes[at] != b'\n' {
                return Err(format!("bad member padding at offset {at}"));
            }
            at += 1;
        }
        let name = raw_name.trim_end();
        if name == "//" {
            strings = body;
            continue;
        }
        if name == "/" || name == "/SYMDEF" {
            // SysV symbol index (never written by us, tolerated on read).
            continue;
        }
        let resolved = if let Some(off) = name.strip_prefix('/').filter(|s| !s.is_empty()) {
            let off: usize = off
                .parse()
                .map_err(|_| format!("bad string-table offset '{off}'"))?;
            let end = strings
                .get(off..)
                .and_then(|s| s.iter().position(|&c| c == b'/').map(|p| off + p))
                .ok_or_else(|| format!("string-table offset {off} out of range"))?;
            std::str::from_utf8(&strings[off..end])
                .map_err(|_| format!("non-UTF8 string-table name at {off}"))?
                .to_owned()
        } else {
            name.strip_suffix('/').unwrap_or(name).to_owned()
        };
        members.push(Member {
            name: resolved,
            bytes: body.to_vec(),
        });
    }
    Ok(members)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_short_names() {
        let members = vec![
            Member {
                name: "a.mcm1".into(),
                bytes: vec![1, 2, 3],
            },
            Member {
                name: "b.mcm1".into(),
                bytes: vec![4, 5],
            },
        ];
        let bytes = write(&members);
        assert_eq!(read(&bytes).expect("read back"), members);
    }

    #[test]
    fn roundtrip_long_name_via_string_table() {
        let members = vec![Member {
            name: "System.Private.CoreLib.mcm1".into(),
            bytes: vec![9; 7],
        }];
        let bytes = write(&members);
        assert_eq!(read(&bytes).expect("read back"), members);
    }

    #[test]
    fn odd_body_padding_skipped() {
        let members = vec![Member {
            name: "x".into(),
            bytes: vec![7],
        }];
        let bytes = write(&members);
        assert_eq!(bytes.len() % 2, 0, "archive stays even-sized");
        assert_eq!(read(&bytes).expect("read back"), members);
    }

    #[test]
    fn bad_magic_rejected() {
        assert!(read(b"definitely not an archive").is_err());
    }

    #[test]
    fn truncated_body_rejected() {
        let mut bytes = write(&[Member {
            name: "x".into(),
            bytes: vec![1, 2, 3, 4],
        }]);
        bytes.truncate(bytes.len() - 2);
        assert!(read(&bytes).is_err());
    }
}
