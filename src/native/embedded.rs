//! Reading type fragments embedded in binaries (`--embed-types`).
//!
//! When a library builds with embedding, its fragment rides inside the
//! artifact instead of a sidecar file: a non-loaded `.rubyc_types`
//! section in `.so` images, a `<member>.rci` member in `.a` archives.
//! Consumers prefer embedded bytes (in sync with their own image by
//! construction) and fall back to sidecar files. Parsing is manual,
//! mirroring `linker.rs` — no section-header library needed.

/// Name of the embedded fragment section in `.so` images.
pub const SECTION_NAME: &[u8] = b".rubyc_types";

/// Fixed preamble offset in our `.so` images (see the writer).
const SO_PREAMBLE_AT: usize = 64;

fn u16(bytes: &[u8], at: usize) -> Option<u16> {
    bytes
        .get(at..at + 2)
        .and_then(|s| <[u8; 2]>::try_from(s).ok())
        .map(u16::from_le_bytes)
}

fn u32(bytes: &[u8], at: usize) -> Option<u32> {
    bytes
        .get(at..at + 4)
        .and_then(|s| <[u8; 4]>::try_from(s).ok())
        .map(u32::from_le_bytes)
}

fn u64(bytes: &[u8], at: usize) -> Option<u64> {
    bytes
        .get(at..at + 8)
        .and_then(|s| <[u8; 8]>::try_from(s).ok())
        .map(u64::from_le_bytes)
}

/// Slice out the `.rubyc_types` section of an ELF image, if present.
/// `None` covers missing tables as well as missing sections (plain
/// images without embedding are the common case, not an error).
pub fn section_bytes(image: &[u8]) -> Option<&[u8]> {
    if image.len() < 64 || &image[0..4] != b"\x7fELF" {
        return None;
    }
    let shoff = u64(image, 40)? as usize;
    let shentsize = u16(image, 58)? as usize;
    let shnum = u16(image, 60)? as usize;
    let shstrndx = u16(image, 62)? as usize;
    if shentsize < 64 || shnum == 0 {
        return None;
    }
    let table = image.get(shoff..shoff + shnum * shentsize)?;
    let str_entry = table.get(shstrndx * shentsize..(shstrndx + 1) * shentsize)?;
    let str_off = u64(str_entry, 24)? as usize;
    let str_size = u64(str_entry, 32)? as usize;
    let strings = image.get(str_off..str_off + str_size)?;
    let name_at = |entry: &[u8]| {
        let off = u32(entry, 0)? as usize;
        let end = strings.get(off..)?.iter().position(|&c| c == 0)? + off;
        strings.get(off..end)
    };
    for i in 0..shnum {
        let entry = table.get(i * shentsize..(i + 1) * shentsize)?;
        if name_at(entry)? == SECTION_NAME {
            let off = u64(entry, 24)? as usize;
            let size = u64(entry, 32)? as usize;
            return image.get(off..off + size);
        }
    }
    None
}

/// Slice out the post-magic type table of an image (`MAGIC[8] +
/// length[u64 LE] + table`), if present. Tries the writer-placed fixed
/// offset first (O(1) on our artifacts), then scans. `None` covers
/// missing/short/truncated tables as well as plain images — only the
/// table layout counts as embedded; anything else falls through to
/// section and sidecar readers.
pub fn table_bytes(image: &[u8]) -> Option<&[u8]> {
    use crate::native::container::{self, Kind};
    let at = |off: usize| -> Option<&[u8]> {
        let parsed = container::parse(image.get(off..)?).ok()?;
        if parsed.kind != Kind::SharedLibraryWithTypes {
            return None;
        }
        let len_off = off + crate::native::container::PREAMBLE_LEN + 3;
        let len = u64(image, len_off)? as usize;
        image.get(len_off + 8..len_off + 8 + len)
    };
    if let Some(t) = at(SO_PREAMBLE_AT) {
        return Some(t);
    }
    // Scan fallback (foreign layouts): first 64KB only. A preamble
    // beyond that is pathological; bounding keeps debug-mode matrix
    // runs (which stat multi-GB stale images) from going quadratic.
    // Bounded by file length.
    (0..image.len().min(65536).saturating_sub(1))
        .filter(|i| *i != SO_PREAMBLE_AT)
        .find_map(|i| at(i))
}

/// Slice out the `<member>.rci` member of an archive, if present.
pub fn archive_member<'a>(members: &'a [crate::native::arlib::Member]) -> Option<&'a [u8]> {
    members
        .iter()
        .find(|m| m.name.ends_with(".rci"))
        .map(|m| m.bytes.as_slice())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_table_is_none_not_error() {
        assert!(section_bytes(b"\x7fELF").is_none());
        assert!(section_bytes(&[0u8; 64]).is_none());
        // Plain image without the section.
        let mut image = vec![0u8; 128];
        image[0..4].copy_from_slice(b"\x7fELF");
        assert!(section_bytes(&image).is_none());
    }

    #[test]
    fn archive_member_matches_rci_suffix() {
        let members = vec![
            crate::native::arlib::Member {
                name: "lib.mcm1".into(),
                bytes: vec![1],
            },
            crate::native::arlib::Member {
                name: "lib.rci".into(),
                bytes: vec![2, 3],
            },
        ];
        assert_eq!(archive_member(&members), Some([2, 3].as_slice()));
        assert!(archive_member(&members[..1]).is_none());
    }

    /// Hand-built minimal ELF: header + NULL + `.rubyc_types` + shstrtab.
    /// Pins the field widths (`st_name` is u32 — reading it as u64
    /// absorbs `st_type` and misses every section).
    #[test]
    fn section_roundtrip_minimal_elf() {        fn le16(v: u16) -> [u8; 2] {
            v.to_le_bytes()
        }
        fn le64(v: u64) -> [u8; 8] {
            v.to_le_bytes()
        }
        let payload = b"RCI1payload";
        let strings = b"\0.rubyc_types\0.shstrtab\0";
        let shoff = 64usize;
        let types_off = shoff + 3 * 64;
        let str_off = types_off + payload.len();
        let mut image = vec![0u8; str_off + strings.len()];
        image[0..4].copy_from_slice(b"\x7fELF");
        image[40..48].copy_from_slice(&le64(shoff as u64));
        image[58..60].copy_from_slice(&le16(64));
        image[60..62].copy_from_slice(&le16(3));
        image[62..64].copy_from_slice(&le16(2));
        // Section 1: .rubyc_types (name 1, PROGBITS, offset, size).
        let mut sh = |i: usize, name: u32, typ: u32, off: usize, size: usize| {
            let at = shoff + i * 64;
            image[at..at + 4].copy_from_slice(&name.to_le_bytes());
            image[at + 4..at + 8].copy_from_slice(&typ.to_le_bytes());
            image[at + 24..at + 32].copy_from_slice(&le64(off as u64));
            image[at + 32..at + 40].copy_from_slice(&le64(size as u64));
        };
        sh(1, 1, 1, types_off, payload.len());
        sh(2, 14, 3, str_off, strings.len());
        image[types_off..types_off + payload.len()].copy_from_slice(payload);
        image[str_off..str_off + strings.len()].copy_from_slice(strings);
        assert_eq!(section_bytes(&image), Some(payload.as_slice()));
    }

    /// Post-magic table: magic + u64 length + bytes at the fixed
    /// offset, plus truncated-length rejection (never panic, never
    /// over-read).
    #[test]
    fn table_roundtrip_and_truncation() {
        use crate::native::container::{MAGIC, PREAMBLE_LEN};
        let payload = b"RCI1table-bytes";
        let mut image = vec![0u8; 300];
        image[64..66].copy_from_slice(&[0xEB, 0x08]);
        image[66..74].copy_from_slice(&MAGIC);
        image[74] = 0x05;
        image[75..77].copy_from_slice(&1u16.to_le_bytes());
        let base = 64 + PREAMBLE_LEN + 3;
        image[base..base + 8].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        image[base + 8..base + 8 + payload.len()].copy_from_slice(payload);
        assert_eq!(table_bytes(&image), Some(payload.as_slice()));
        // Truncated length (overruns the file) is None, not a panic.
        let mut short = image.clone();
        short.truncate(base + 10);
        assert!(table_bytes(&short).is_none());
        // Plain kind byte is not a table.
        let mut plain = image.clone();
        plain[74] = 0x03;
        assert!(table_bytes(&plain).is_none());
    }
}
