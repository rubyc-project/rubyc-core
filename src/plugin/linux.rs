use std::collections::BTreeMap;

struct LoadedElf {
    base: u64,
    region: *mut u8,
    region_len: usize,
    symbols: BTreeMap<String, u64>,
}

pub fn load_elf(bytes: &[u8]) -> Result<(u64, BTreeMap<String, *const u8>, *mut u8, usize), String> {
    let loaded = load_elf_inner(bytes)?;
    let symbols: BTreeMap<String, *const u8> = loaded
        .symbols
        .iter()
        .map(|(name, addr)| (name.clone(), *addr as *const u8))
        .collect();
    let base = loaded.base;
    let region = loaded.region;
    let region_len = loaded.region_len;
    std::mem::forget(loaded);
    Ok((base, symbols, region, region_len))
}

fn load_elf_inner(bytes: &[u8]) -> Result<LoadedElf, String> {
    if bytes.len() < 64 {
        return Err("image too small for ELF header".to_string());
    }
    if &bytes[0..4] != &[0x7f, b'E', b'L', b'F'] {
        return Err("bad ELF magic".to_string());
    }
    if bytes[4] != 2 || bytes[5] != 1 {
        return Err("expected 64-bit little-endian ELF".to_string());
    }
    let e_type = u16::from_le_bytes([bytes[16], bytes[17]]);
    let e_machine = u16::from_le_bytes([bytes[18], bytes[19]]);
    if e_type != 3 {
        return Err("expected ET_DYN".to_string());
    }
    if e_machine != 0x3e {
        return Err("expected x86_64 machine".to_string());
    }

    let e_phoff = u64::from_le_bytes(bytes[32..40].try_into().unwrap()) as usize;
    let e_phentsize = u16::from_le_bytes([bytes[54], bytes[55]]) as usize;
    let e_phnum = u16::from_le_bytes([bytes[56], bytes[57]]) as usize;
    if e_phentsize < 56 {
        return Err("bad program header size".to_string());
    }

    let page = 0x1000usize;
    let mut region_len = 0usize;
    for i in 0..e_phnum {
        let off = e_phoff + i * e_phentsize;
        if off + 56 > bytes.len() {
            return Err("truncated program header".to_string());
        }
        let p_type = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        if p_type != 1 {
            continue;
        }
        let p_offset = u64::from_le_bytes(bytes[off + 8..off + 16].try_into().unwrap());
        let p_vaddr = u64::from_le_bytes(bytes[off + 16..off + 24].try_into().unwrap());
        let p_filesz = u64::from_le_bytes(bytes[off + 32..off + 40].try_into().unwrap());
        let p_memsz = u64::from_le_bytes(bytes[off + 40..off + 48].try_into().unwrap());
        if p_vaddr != p_offset {
            return Err("identity mapping required".to_string());
        }
        if p_offset as usize % page != 0 {
            return Err("segment offset not page aligned".to_string());
        }
        if p_offset as usize + p_filesz as usize > bytes.len() {
            return Err("truncated load segment".to_string());
        }
        let end = p_offset as usize + p_memsz.max(p_filesz) as usize;
        region_len = region_len.max(end);
    }
    if region_len == 0 {
        return Err("no PT_LOAD segment".to_string());
    }
    region_len = (region_len + page - 1) / page * page;

    const PROT_READ: i32 = 0x1;
    const PROT_WRITE: i32 = 0x2;
    const PROT_EXEC: i32 = 0x4;
    const MAP_PRIVATE: i32 = 0x02;
    const MAP_ANONYMOUS: i32 = 0x20;

    let region = unsafe {
        mmap(std::ptr::null_mut(), region_len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
    };
    if region.is_null() || region as usize == usize::MAX {
        return Err("mmap failed".to_string());
    }
    let base = region as u64;

    for i in 0..e_phnum {
        let off = e_phoff + i * e_phentsize;
        let p_type = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        if p_type != 1 {
            continue;
        }
        let p_offset = u64::from_le_bytes(bytes[off + 8..off + 16].try_into().unwrap()) as usize;
        let p_filesz = u64::from_le_bytes(bytes[off + 32..off + 40].try_into().unwrap()) as usize;
        let p_memsz = u64::from_le_bytes(bytes[off + 40..off + 48].try_into().unwrap()) as usize;
        let p_flags = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());

        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), region.add(p_offset), p_filesz);
        }

        let page_len = (p_memsz + page - 1) / page * page;
        let mut prot = PROT_READ;
        if p_flags & 0x2 != 0 {
            prot |= PROT_WRITE;
        }
        if p_flags & 0x1 != 0 {
            prot |= PROT_EXEC;
        }

        if unsafe { mprotect(region.add(p_offset), page_len, prot) } != 0 {
            unsafe {
                munmap(region as *mut std::os::raw::c_void, region_len);
            }
            return Err("mprotect failed".to_string());
        }
    }

    let symbols = read_symbols(bytes, region);

    Ok(LoadedElf {
        base,
        region,
        region_len,
        symbols,
    })
}

fn read_symbols(bytes: &[u8], region: *mut u8) -> BTreeMap<String, u64> {
    if bytes.len() < 64 {
        return BTreeMap::new();
    }
    let e_phoff = u64::from_le_bytes(bytes[32..40].try_into().unwrap()) as usize;
    let e_phentsize = u16::from_le_bytes([bytes[54], bytes[55]]) as usize;
    let e_phnum = u16::from_le_bytes([bytes[56], bytes[57]]) as usize;

    let mut dynsym_off: Option<usize> = None;
    let mut dynstr_off: Option<usize> = None;
    let mut symsz: Option<usize> = None;
    let mut strsz: Option<usize> = None;

    for i in 0..e_phnum {
        let off = e_phoff + i * e_phentsize;
        if off + 56 > bytes.len() {
            break;
        }
        let p_type = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());

        if p_type != 2 {
            continue;
        }
        let p_offset = u64::from_le_bytes(bytes[off + 8..off + 16].try_into().unwrap()) as usize;
        let p_filesz = u64::from_le_bytes(bytes[off + 32..off + 40].try_into().unwrap()) as usize;

        if p_offset + p_filesz > bytes.len() {
            continue;
        }
        let mut d = p_offset;
        while d + 16 <= p_offset + p_filesz {
            let tag = i64::from_le_bytes(bytes[d..d + 8].try_into().unwrap());
            let val = u64::from_le_bytes(bytes[d + 8..d + 16].try_into().unwrap()) as usize;
    
            if tag == 0 {
                break;
            }
            match tag {
                6 => dynsym_off = Some(val),
                5 => dynstr_off = Some(val),
                10 => symsz = Some(val),
                11 => strsz = Some(val),
                _ => {}
            }
            d += 16;
        }
    }

    let dynsym_off = match dynsym_off {
        Some(v) => v,
        None => return BTreeMap::new(),
    };
    let dynstr_off = match dynstr_off {
        Some(v) => v,
        None => return BTreeMap::new(),
    };
    let symsz = match symsz {
        Some(v) => v,
        None => return BTreeMap::new(),
    };
    let strsz = match strsz {
        Some(v) => v,
        None => return BTreeMap::new(),
    };

    let mut symbols = BTreeMap::new();
    for idx in 0..(symsz / 24) {
        let sym_off = dynsym_off + idx * 24;
        if sym_off + 24 > bytes.len() {
            break;
        }
        let st_name = u32::from_le_bytes(bytes[sym_off..sym_off + 4].try_into().unwrap()) as usize;
        let st_info = bytes[sym_off + 4];
        let st_value = u64::from_le_bytes(bytes[sym_off + 8..sym_off + 16].try_into().unwrap());
        let bind = st_info >> 4;
        let typ = st_info & 0xf;
        if bind == 0 || typ == 0 {
            continue;
        }
        if dynstr_off + st_name >= dynstr_off + strsz {
            continue;
        }
        let name = read_cstr(&bytes[dynstr_off + st_name..dynstr_off + strsz]);

        if name.is_empty() {
            continue;
        }
        let addr = (region as u64) + st_value;
        symbols.insert(name, addr);
    }

    symbols
}

fn read_cstr(slice: &[u8]) -> String {
    let end = slice.iter().position(|b| *b == 0).unwrap_or(slice.len());
    String::from_utf8_lossy(&slice[..end]).into_owned()
}

unsafe extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn mprotect(addr: *mut u8, len: usize, prot: i32) -> i32;
    fn munmap(addr: *mut std::os::raw::c_void, len: usize) -> i32;
}
