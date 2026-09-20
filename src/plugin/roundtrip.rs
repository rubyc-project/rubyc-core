const PAGE: usize = 0x1000;


const RODATA_OFF: usize = 0x200;
const NAME_OFF: usize = RODATA_OFF + 1;
const TRIPLE_OFF: usize = RODATA_OFF + 0x10;
const DYN_OFF: usize = 0x280;
const RELA_OFF: usize = 0x310;
const HASH_OFF: usize = 0x380;
const SYMTAB_OFF: usize = 0x400;
const STRTAB_OFF: usize = 0x500;
const GOT_OFF: usize = 0x1000;

const FN_NAME: usize = 0x180;
const FN_TRIPLE: usize = 0x188;
const FN_BASE: usize = 0x190;
const FN_LOWER: usize = 0x198;
const FN_WRITE_IMAGE: usize = 0x1A2;
const FN_RUN: usize = 0x1AE;

const NAME: &str = "roundtrip";
const TRIPLE: &str = "roundtrip-bare";

const DT_HASH: i64 = 4;
const DT_STRTAB: i64 = 5;
const DT_SYMTAB: i64 = 6;
const DT_RELA: i64 = 7;
const DT_SYMSZ: i64 = 10;
const DT_STRSZ: i64 = 11;
const DT_RELAENT: i64 = 55;
const DT_RELASZ: i64 = 60;
const DT_NULL: i64 = 0;

pub fn build_roundtrip_plugin() -> Vec<u8> {
    let mut img = vec![0u8; PAGE * 2];

    img[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
    img[4] = 2;
    img[5] = 1;
    img[6] = 1;
    img[7] = 0;
    put_u16(&mut img, 16, 3);
    put_u16(&mut img, 18, 0x3e);
    put_u32(&mut img, 20, 1);
    put_u64(&mut img, 24, 0);
    put_u64(&mut img, 32, 0x38);

    put_u64(&mut img, 40, 0);
    put_u32(&mut img, 48, 0);
    img[52] = 64;
    img[53] = 0;
    img[54] = 56;
    img[55] = 0;
    img[56] = 3;
    img[57] = 0;

    put_phdr(&mut img, 0x38, 1, 5, 0, 0, PAGE as u64, PAGE as u64);
    put_phdr(&mut img, 0x70, 1, 3, GOT_OFF as u64, GOT_OFF as u64, 8, PAGE as u64);
    put_phdr(&mut img, 0xA8, 2, 4, DYN_OFF as u64, DYN_OFF as u64, 8 * 16, 8 * 16);
    img[56] = 3;
    img[57] = 0;

    img[FN_NAME..FN_NAME + 2].copy_from_slice(&[0x48, 0xBE]);
    put_u32(&mut img, FN_NAME + 2, NAME_OFF as u32);
    img[FN_NAME + 6] = 0xC3;
    img[FN_LOWER..FN_LOWER + 9].copy_from_slice(&[0x48, 0x89, 0x3A, 0x48, 0x89, 0x31, 0x31, 0xC0, 0xC3]);
    img[FN_WRITE_IMAGE..FN_WRITE_IMAGE + 9].copy_from_slice(&[0x48, 0x89, 0x31, 0x4C, 0x89, 0x38, 0x31, 0xC0, 0xC3]);
    img[FN_RUN..FN_RUN + 6].copy_from_slice(&[0x48, 0x89, 0x32, 0x31, 0xC0, 0xC3]);

    img[NAME_OFF..NAME_OFF + NAME.len()].copy_from_slice(NAME.as_bytes());
    img[NAME_OFF + NAME.len()] = 0;
    img[TRIPLE_OFF..TRIPLE_OFF + TRIPLE.len()].copy_from_slice(TRIPLE.as_bytes());
    img[TRIPLE_OFF + TRIPLE.len()] = 0;

    let mut d = DYN_OFF;
    put_dyn(&mut img, &mut d, DT_HASH, HASH_OFF as u64);
    put_dyn(&mut img, &mut d, DT_STRTAB, STRTAB_OFF as u64);
    put_dyn(&mut img, &mut d, DT_SYMTAB, SYMTAB_OFF as u64);
    put_dyn(&mut img, &mut d, DT_SYMSZ, 7 * 24);
    put_dyn(&mut img, &mut d, DT_STRSZ, 73);
    put_dyn(&mut img, &mut d, DT_RELA, RELA_OFF as u64);
    put_dyn(&mut img, &mut d, DT_RELAENT, 24);
    put_dyn(&mut img, &mut d, DT_RELASZ, 24);
    put_dyn(&mut img, &mut d, DT_NULL, 0);

    put_u64(&mut img, RELA_OFF, GOT_OFF as u64);
    put_u64(&mut img, RELA_OFF + 8, 8);
    put_u64(&mut img, RELA_OFF + 16, 0);

    let names = [
        "",
        "rbxt_name",
        "rbxt_triple",
        "rbxt_base_addr",
        "rbxt_lower",
        "rbxt_write_image",
        "rbxt_run",
    ];
    let mut str_off = 0usize;
    let mut str_offs = Vec::new();
    for name in &names {
        str_offs.push(str_off);
        str_off += name.len() + 1;
    }

    let h = |name: &str| sysv_hash(name) as usize % 16;
    let mut buckets = vec![0usize; 16];
    let mut next = vec![0usize; 7];
    for (i, name) in names.iter().enumerate().skip(1) {
        let b = h(name);
        if buckets[b] == 0 {
            buckets[b] = i;
        } else {
            let mut cur = buckets[b];
            while next[cur] != 0 {
                cur = next[cur];
            }
            next[cur] = i;
        }
    }
    put_u32(&mut img, HASH_OFF, 16);
    for (i, v) in buckets.iter().enumerate() {
        put_u32(&mut img, HASH_OFF + 4 + i * 4, *v as u32);
    }
    for (i, v) in next.iter().enumerate() {
        put_u32(&mut img, HASH_OFF + 4 + 16 * 4 + i * 4, *v as u32);
    }

    let sym_values = [
        0usize,
        FN_NAME,
        FN_TRIPLE,
        FN_BASE,
        FN_LOWER,
        FN_WRITE_IMAGE,
        FN_RUN,
    ];
    for i in 0..7 {
        let o = SYMTAB_OFF + i * 24;
        put_u32(&mut img, o, str_offs[i] as u32);
        put_u8(&mut img, o + 4, if i == 0 { 0 } else { 0x11 });
        put_u8(&mut img, o + 5, 0);
        put_u64(&mut img, o + 8, sym_values[i] as u64);
        put_u32(&mut img, o + 16, 0);
        put_u32(&mut img, o + 20, 0);
    }

    let mut s = STRTAB_OFF;
    for name in &names {
        img[s..s + name.len()].copy_from_slice(name.as_bytes());
        img[s + name.len()] = 0;
        s += name.len() + 1;
    }

    img
}

fn put_u8(img: &mut [u8], off: usize, v: u8) {
    img[off] = v;
}

fn put_u16(img: &mut [u8], off: usize, v: u16) {
    img[off..off + 2].copy_from_slice(&v.to_le_bytes());
}

fn put_u32(img: &mut [u8], off: usize, v: u32) {
    img[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_u64(img: &mut [u8], off: usize, v: u64) {
    img[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

fn put_phdr(img: &mut [u8], off: usize, p_type: u32, p_flags: u32, p_offset: u64, p_vaddr: u64, p_filesz: u64, p_memsz: u64) {
    put_u32(img, off, p_type);
    put_u32(img, off + 4, p_flags);
    put_u64(img, off + 8, p_offset);
    put_u64(img, off + 16, p_vaddr);
    put_u64(img, off + 24, p_vaddr);
    put_u64(img, off + 32, p_filesz);
    put_u64(img, off + 40, p_memsz);
    put_u64(img, off + 48, PAGE as u64);
}

fn put_rip(img: &mut [u8], fn_off: usize, target: usize) {
    img[fn_off..fn_off + 2].copy_from_slice(&[0x48, 0xBE]);
    put_u32(img, fn_off + 2, target as u32);
    img[fn_off + 6] = 0xC3;
}

fn put_rip_mov(img: &mut [u8], fn_off: usize, target: usize) {
    let disp = (target as i64 - (fn_off + 6) as i64) as u32;
    img[fn_off..fn_off + 3].copy_from_slice(&[0x48, 0x8B, 0x05]);
    put_u32(img, fn_off + 3, disp);
    img[fn_off + 7] = 0xC3;
}

fn put_dyn(img: &mut [u8], off: &mut usize, tag: i64, val: u64) {
    img[*off..*off + 8].copy_from_slice(&tag.to_le_bytes());
    put_u64(img, *off + 8, val);
    *off += 16;
}

fn sysv_hash(name: &str) -> u32 {
    let mut h: u32 = 0;
    for &c in name.as_bytes() {
        h = h.wrapping_mul(16).wrapping_add(c as u32);
        let g = h & 0xf000_0000;
        if g != 0 {
            h ^= g >> 24;
            h &= !g;
        }
    }
    h
}
