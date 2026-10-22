//! Link-time symbol extraction from ELF shared objects.
//!
//! RubyC executables are fully static (no libc, no dynamic loader), so
//! `[LibraryImport]` stubs are resolved at BUILD time: the compiler reads
//! each `--lib` ELF, finds the exported symbol, and copies its machine
//! code into the output image.
//!
//! Relative (rel32) branches inside an extracted body stay valid because
//! the blob is copied contiguously. Limitations (deliberate for this
//! milestone): imported functions may not reference their library's data
//! sections or call symbols outside themselves; pure computational
//! functions — the target interop case — are fully supported.

use std::path::PathBuf;

const SHT_SYMTAB: u32 = 2;
const SHT_DYNSYM: u32 = 11;
const STT_FUNC: u8 = 2;
const STT_OBJECT: u8 = 1;
const SHN_UNDEF: u16 = 0;

struct Section {
    sh_type: u32,
    sh_addr: u64,
    sh_offset: u64,
    sh_size: u64,
    sh_link: usize,
    sh_entsize: u64,
}

fn u16le(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn u32le(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn u64le(b: &[u8], at: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(a)
}

/// Parse the section header table of a 64-bit little-endian ELF.
fn parse_sections(img: &[u8]) -> Result<Vec<Section>, String> {
    if img.len() < 64 || img[0..4] != *b"\x7fELF" {
        return Err("not an ELF file".into());
    }
    if img[4] != 2 {
        return Err("not a 64-bit ELF".into());
    }
    let shoff = u64le(img, 40) as usize;
    let shentsize = u16le(img, 58) as usize;
    let shnum = u16le(img, 60) as usize;
    if shoff == 0 || shentsize < 64 {
        return Err("ELF has no section headers".into());
    }
    let mut out = Vec::with_capacity(shnum);
    for i in 0..shnum {
        let at = shoff + i * shentsize;
        if at + 64 > img.len() {
            return Err("truncated section headers".into());
        }
        out.push(Section {
            sh_type: u32le(img, at + 4),
            sh_addr: u64le(img, at + 16),
            sh_offset: u64le(img, at + 24),
            sh_size: u64le(img, at + 32),
            sh_link: u32le(img, at + 40) as usize,
            sh_entsize: u64le(img, at + 56),
        });
    }
    Ok(out)
}

/// All defined symbols of one symbol table with their file ranges,
/// filtered by kind: `objects_only` selects `STT_OBJECT` data
/// symbols (static slots, vtable rows), otherwise `STT_FUNC` bodies.
///
/// For ET_DYN objects (`ld -shared` output) symbol values are offsets from
/// the load base, so the file offset is `sh_offset + st_value - sh_addr`
/// within the defining section.
fn read_symbols(
    img: &[u8],
    sections: &[Section],
    symtab: &Section,
    objects_only: bool,
) -> Vec<(String, std::ops::Range<usize>)> {
    let Some(strtab) = sections.get(symtab.sh_link) else {
        return Vec::new();
    };
    if symtab.sh_entsize == 0 || strtab.sh_offset == 0 {
        return Vec::new();
    }
    let entsz = symtab.sh_entsize as usize;
    let count = (symtab.sh_size / symtab.sh_entsize) as usize;
    let str_at = strtab.sh_offset as usize;
    let mut out = Vec::new();
    for i in 0..count {
        let at = symtab.sh_offset as usize + i * entsz;
        if at + 24 > img.len() {
            break;
        }
        let st_name = u32le(img, at) as usize;
        let st_info = img[at + 4];
        let st_shndx = u16le(img, at + 6);
        let st_value = u64le(img, at + 8);
        let st_size = u64le(img, at + 16);
        // Kind gate: `STT_FUNC` for copy-linkable bodies, `STT_OBJECT`
        // for data slots/vtable rows (see the push below).
        let want = if objects_only { STT_OBJECT } else { STT_FUNC };
        if st_name == 0 || st_size == 0 || st_shndx == SHN_UNDEF || st_info & 0xf != want {
            continue;
        }
        let Some(section) = sections.get(st_shndx as usize) else {
            continue;
        };
        let delta = (st_value - section.sh_addr) as i64;
        let file_start = section.sh_offset as i64 + delta;
        if file_start < 0 {
            continue;
        }
        let start = file_start as usize;
        let end = start + st_size as usize;
        if end > img.len() {
            continue;
        }
        let name_at = str_at + st_name;
        if name_at >= img.len() {
            continue;
        }
        let name_end = img[name_at..]
            .iter()
            .position(|&c| c == 0)
            .map(|p| name_at + p)
            .unwrap_or(name_at);
        let Ok(name) = std::str::from_utf8(&img[name_at..name_end]) else {
            continue;
        };
        if name.is_empty() {
            continue;
        }
        out.push((name.to_owned(), start..end));
    }
    out
}

/// OBJECT (`STT_OBJECT`) symbol byte ranges in one image (static
/// slots and vtable rows). Copy-link extracts only FUNCTION bodies;
/// data slots need an address, which the static path resolves by
/// name (see `merge_archives`), so they are looked up separately.
fn object_symbols(img: &[u8]) -> Result<Vec<(String, std::ops::Range<usize>)>, String> {
    let sections = parse_sections(img)?;
    let mut out = Vec::new();
    for section in &sections {
        if section.sh_type == SHT_DYNSYM || section.sh_type == SHT_SYMTAB {
            out.extend(read_symbols(img, &sections, section, true));
        }
    }
    Ok(out)
}

/// Every exported function symbol across all symbol tables of one image.
fn exported_functions(img: &[u8]) -> Result<Vec<(String, std::ops::Range<usize>)>, String> {
    let sections = parse_sections(img)?;
    let mut syms = Vec::new();
    for section in &sections {
        if section.sh_type == SHT_DYNSYM || section.sh_type == SHT_SYMTAB {
            syms.extend(read_symbols(img, &sections, section, false));
        }
    }
    Ok(syms)
}

/// True for a RubyC shared image (the 🐕🇨 container preamble sits right
/// after the 64-byte ELF header). Such a library is a RUNTIME dependency:
/// its exports are SysV→internal trampolines (`call body` is relative to
/// the home image), so copying the bytes into an executable is invalid.
/// C libraries do not carry the preamble and stay copy-linkable.
fn is_rubyc_shared_image(bytes: &[u8]) -> bool {
    use crate::native::container::{JUMP_OVER_MAGIC, MAGIC};
    bytes.len() >= 74
        && bytes[64..66] == JUMP_OVER_MAGIC
        && bytes[66..74] == MAGIC
}

/// Resolve dependency-image imports at LOAD time: for each `.so`-hinted
/// entry of `import_names`, `dlopen` the installed dep `.so` under
/// `dyn_dir` and `dlsym` the entry, writing the absolute address into
/// the parallel `slots` element. Non-dep entries are untouched.
/// Returns the loaded handles: the caller keeps them alive while the
/// consumer image runs (dropping a handle `dlclose`s the dep).
///
/// The dep `.so` files export both function symbols
/// (`<path>::<method>#<arity>`) and vtable-row OBJECT symbols
/// (`<path>::__vtable`), so `CallImport` and `LoadImport` slots resolve
/// through this one path. Verified on the VM only: `dlopen` of
/// hand-assembled images must never run on the host.
pub fn resolve_dep_imports(
    import_names: &[(String, String)],
    slots: &mut [u64],
    dyn_dir: &std::path::Path,
) -> Result<Vec<crate::libloader::Handle>, String> {
    use crate::libloader::Load as _;
    let mut handles = Vec::new();
    for (i, (lib_hint, entry)) in import_names.iter().enumerate() {
        if !lib_hint.ends_with(".so") {
            continue;
        }
        let path = dyn_dir.join(lib_hint);
        let handle = crate::libloader::load_path(&path)
            .map_err(|e| format!("cannot load dependency image {}: {e:?}", path.display()))?;
        let addr = handle
            .sym(entry)
            .ok_or_else(|| format!("import '{entry}' not exported by {}", path.display()))?;
        if let Some(slot) = slots.get_mut(i) {
            *slot = addr;
        }
        handles.push(handle);
    }
    Ok(handles)
}
/// Resolve a compiled image's dependency-image imports into its
/// `import_abs` table, for load-time relocation. For each `.so`-hinted
/// entry of `mc.import_names`, `dlopen`s the installed dep `.so` under
/// `dyn_dir` and records the `dlsym` address; non-dep entries and
/// already-extracted (static) imports are untouched. Returns the loaded
/// handles: the caller keeps them alive while the consumer image runs.
/// Verified on the VM only: `dlopen` of hand-assembled images must
/// never run on the host.
pub fn resolve_dep_imports_into(
    mc: &mut crate::native::target::MachineCode,
    dyn_dir: &std::path::Path,
) -> Result<Vec<crate::libloader::Handle>, String> {
    mc.import_abs.clear();
    mc.import_abs.resize(mc.import_names.len(), u64::MAX);
    resolve_dep_imports(&mc.import_names, &mut mc.import_abs, dyn_dir)
}
/// Static linking from `.a` archives: merge archive members into the main
/// image and resolve imports against member exports, so a standalone
/// executable carries every callee body (no loader, no `.so` at runtime).
///
/// Each member is one library's unlinked `MachineCode` (decoded from its
/// archive member by the target backend). Merging appends code/data and
/// remaps every index that addresses them: function bases, vtable rows,
/// static slots, string/exception-state data offsets, helper bodies, and
/// relocation symbols. Imports the merged members export — functions by
/// entry name, vtable rows and static slots by their object symbols — are
/// resolved in place, including `.so`-hinted dependency imports (a static
/// image has no loader, so those must consume a member body here).
///
/// Deliberate v1 semantics, all loud on violation:
/// - State is per-image, not shared: static slots and vtable rows merge
///   by copy, so two images statically linked against the same library
///   each own their statics (unlike dynamic linking, where the `.so`
///   is the single home). Diamond dependencies (two members exporting
///   the same symbol) resolve first-wins, and members exporting nothing
///   still unresolved never merge (dead code stays out of the image).
/// - Member `@init` functions (static field initializers) cannot chain
///   through the single `init_fn` slot: the first one wins, a second
///   is a hard error naming both members.
/// - Imported bodies must be self-contained within the merged image
///   (same bar as `--lib` body extraction): a body calling a symbol no
///   member exports fails naming the missing entry.
pub fn link_static_archives(
    mc: &mut crate::native::target::MachineCode,
    archives: Vec<(String, crate::native::target::MachineCode)>,
) -> Result<(), String> {
    use crate::native::target::{Reloc, Section, Sym};

    let mut funcs: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (name, idx, _) in mc.exports.iter() {
        if let Some(base) = mc.fn_bases.get(*idx) {
            funcs.entry(name.clone()).or_insert(*base);
        }
    }
    let mut objects: std::collections::HashMap<String, StaticObj> = std::collections::HashMap::new();
    for (row, path) in mc.vtable_offsets.iter().zip(mc.vtable_names.iter()) {
        objects
            .entry(crate::native::target::vtable_symbol(path))
            .or_insert(StaticObj::Vtable(*row));
    }
    for (k, sym) in mc.static_names.iter().enumerate() {
        objects.entry(sym.clone()).or_insert(StaticObj::Static(k));
    }
    // Import sidecars run parallel to the names.
    mc.import_bases.resize(mc.import_names.len(), usize::MAX);
    mc.import_abs.resize(mc.import_names.len(), u64::MAX);

    let unresolved = |mc: &crate::native::target::MachineCode| -> Vec<usize> {
        mc.import_names
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                let base_missing =
                    !matches!(mc.import_bases.get(*i).copied(), Some(b) if b != usize::MAX);
                let abs_missing = mc.import_abs.get(*i).copied().unwrap_or(u64::MAX) == u64::MAX;
                // `.so` hints are loader business in dynamic images, but a
                // static image has no loader: they must resolve here too.
                base_missing && abs_missing
            })
            .map(|(i, _)| i)
            .collect()
    };

    // Fixpoint: merge members exporting something still unresolved until
    // nothing new resolves. Each archive merges at most once.
    let mut merged: std::collections::HashSet<String> = std::collections::HashSet::new();
    loop {
        let need = unresolved(mc);
        if need.is_empty() {
            break;
        }
        let wanted: std::collections::HashSet<&str> =
            need.iter().map(|&i| mc.import_names[i].1.as_str()).collect();
        // Prefer the member whose archive stem matches the hint stem
        // (`Foo.so` prefers archive `Foo`), else any exporter.
        let mut pick: Option<usize> = None;
        for (ai, (stem, member)) in archives.iter().enumerate() {
            if merged.contains(stem) {
                continue;
            }
            let exports_name = |n: &str| {
                member.exports.iter().any(|(e, _, _)| e == n)
                    || member
                        .vtable_names
                        .iter()
                        .any(|p| crate::native::target::vtable_symbol(p) == n)
                    || member.static_names.iter().any(|s| s == n)
            };
            let any_match = wanted.iter().any(|w| exports_name(w));
            if !any_match {
                continue;
            }
            let hinted = need.iter().any(|&i| {
                let hint = mc.import_names[i].0.trim_end_matches(".so");
                !hint.is_empty() && (stem == hint || member_name_matches(stem, hint))
            });
            if hinted {
                pick = Some(ai);
                break;
            }
            if pick.is_none() {
                pick = Some(ai);
            }
        }
        let Some(ai) = pick else {
            let missing: Vec<String> = need
                .iter()
                .map(|&i| {
                    format!("'{}' (hint '{}')", mc.import_names[i].1, mc.import_names[i].0)
                })
                .collect();
            let searched: Vec<&str> = archives.iter().map(|(s, _)| s.as_str()).collect();
            return Err(format!(
                "static link failed: no archive member exports {} (searched: {})",
                missing.join(", "),
                searched.join(", ")
            ));
        };
        let (stem, member) = &archives[ai];
        let stem = stem.clone();
        // Merge member code/data with index remapping (code stays
        // 16-aligned like extracted bodies, data rows 8-aligned).
        while !mc.code.len().is_multiple_of(16) {
            mc.code.push(0xCC);
        }
        while !mc.data.len().is_multiple_of(8) {
            mc.data.push(0);
        }
        let code_base = mc.code.len();
        let data_base = mc.data.len();
        let fn_base = mc.fn_bases.len();
        let sb_base = mc.static_offsets.len();
        let imp_base = mc.import_names.len();
        mc.code.extend_from_slice(&member.code);
        mc.data.extend_from_slice(&member.data);
        mc.fn_bases
            .extend(member.fn_bases.iter().map(|b| b + code_base));
        mc.vtable_offsets
            .extend(member.vtable_offsets.iter().map(|o| o + data_base));
        mc.vtable_names.extend(member.vtable_names.iter().cloned());
        mc.static_offsets
            .extend(member.static_offsets.iter().map(|o| o + data_base));
        mc.static_names.extend(member.static_names.iter().cloned());
        // Helpers: adopt only when the merged image lacks one; the
        // member's own copy stays as dead weight otherwise (its relocs
        // address the adopted home through the shared slots below).
        macro_rules! adopt_helper {
            ($slot:ident) => {
                if mc.$slot.is_none() {
                    mc.$slot = member.$slot.map(|o| o + code_base);
                }
            };
        }
        adopt_helper!(helper_itoa);
        adopt_helper!(helper_itoa_err);
        adopt_helper!(helper_scan_int);
        adopt_helper!(helper_retain);
        adopt_helper!(helper_release);
        adopt_helper!(helper_alloc);
        adopt_helper!(helper_exc_enter);
        adopt_helper!(helper_exc_pop);
        adopt_helper!(helper_exc_throw);
        adopt_helper!(helper_exc_rethrow);
        adopt_helper!(helper_exc_load);
        adopt_helper!(helper_exc_clear);
        adopt_helper!(helper_argv);
        if mc.exc_state_offset.is_none() {
            mc.exc_state_offset = member.exc_state_offset.map(|o| o + data_base);
        }
        if let Some(f) = member.init_fn {
            if mc.init_fn.is_none() {
                mc.init_fn = Some(f + fn_base);
            } else {
                return Err(format!(
                    "static link failed: '{}' and the merged image both define @init (static field initializers cannot chain yet)",
                    stem
                ));
            }
        }
        for (name, idx, arity) in member.exports.iter() {
            funcs
                .entry(name.clone())
                .or_insert(code_base + member.fn_bases.get(*idx).copied().unwrap_or(0));
            mc.exports.push((name.clone(), idx + fn_base, *arity));
        }
        mc.export_takes_receiver
            .extend(member.export_takes_receiver.iter().copied());
        for (row, path) in member
            .vtable_offsets
            .iter()
            .zip(member.vtable_names.iter())
        {
            objects
                .entry(crate::native::target::vtable_symbol(path))
                .or_insert(StaticObj::Vtable(row + data_base));
        }
        for (k, sym) in member.static_names.iter().enumerate() {
            objects
                .entry(sym.clone())
                .or_insert(StaticObj::Static(sb_base + k));
        }
        // Member imports join the table (resolved ones rebased); their
        // relocs below are remapped onto the merged index spaces.
        // Backends may leave sidecars empty when nothing was ever
        // extracted: normalize to the names length first, or merged
        // slots go missing and later passes misread absence as resolved.
        mc.import_names.extend(member.import_names.iter().cloned());
        mc.import_bases.extend(
            member
                .import_bases
                .iter()
                .copied()
                .chain(std::iter::repeat(usize::MAX))
                .take(member.import_names.len())
                .map(|b| {
                    if b == usize::MAX {
                        usize::MAX
                    } else {
                        b + code_base
                    }
                }),
        );
        mc.import_abs.extend(
            member
                .import_abs
                .iter()
                .copied()
                .chain(std::iter::repeat(u64::MAX))
                .take(member.import_names.len()),
        );
        let remap = |sym: &Sym| -> Sym {
            match sym {
                Sym::Fn(i) => Sym::Fn(i + fn_base),
                Sym::String(o) => Sym::String(o + data_base),
                Sym::VtableOff(o) => Sym::VtableOff(o + data_base),
                Sym::Static(i) => Sym::Static(i + sb_base),
                Sym::Import(i) => Sym::Import(i + imp_base),
                Sym::ExcState(o) => Sym::ExcState(o + data_base),
                other => other.clone(),
            }
        };
        mc.relocs.extend(member.relocs.iter().map(|r| Reloc {
            kind: r.kind,
            section: r.section,
            site: r.site
                + match r.section {
                    Section::Code => code_base,
                    Section::Data => data_base,
                },
            sym: remap(&r.sym),
        }));
        mc.heap_size = mc.heap_size.max(member.heap_size);
        merged.insert(stem);
        // Resolve everything resolvable after this merge, then loop:
        // newly merged members can satisfy previously unresolvable
        // imports (transitive dependencies).
        resolve_against(mc, &funcs, &objects)?;
    }
    // Final sweep (covers imports the main image already satisfied).
    resolve_against(mc, &funcs, &objects)?;
    Ok(())
}

/// A merged linkable object: a vtable row by data offset, or a static
/// slot by static-table index.
#[derive(Clone, Copy)]
enum StaticObj {
    Vtable(usize),
    Static(usize),
}

/// Match an archive stem against an import hint stem loosely: exact
/// equality wins elsewhere, this only tolerates the `lib` prefix the
/// filesystem adds (`libFoo` archive satisfies hint `Foo`).
fn member_name_matches(stem: &str, hint: &str) -> bool {
    stem == hint || stem.strip_prefix("lib").unwrap_or(stem) == hint
}

/// Drop import slots no relocation references that also carry no body:
/// null function slots (and null `.so` hints) that nothing points at
/// can never be called, so resolving them only drags unreachable
/// natives into standalone images (or fails the link loudly for
/// dead code). Mirrors the drop rule inside `resolve_against`; the
/// executable path runs it unconditionally before `resolve_imports`
/// so plain (unoptimized) images link with only live stubs.
pub(crate) fn drop_unreferenced_null_slots(mc: &mut crate::native::target::MachineCode) {
    use crate::native::target::Sym;
    let mut referenced = vec![false; mc.import_names.len()];
    for r in mc.relocs.iter() {
        if let Sym::Import(i) = r.sym {
            if i < referenced.len() {
                referenced[i] = true;
            }
        }
    }
    let drop: Vec<bool> = mc
        .import_names
        .iter()
        .enumerate()
        .map(|(i, _)| {
            !referenced[i]
                && mc.import_bases.get(i).copied().unwrap_or(usize::MAX) == usize::MAX
                && mc.import_abs.get(i).copied().unwrap_or(u64::MAX) == u64::MAX
        })
        .collect();
    if !drop.iter().any(|&d| d) {
        return;
    }
    let mut remap = vec![0usize; mc.import_names.len()];
    let mut kept = 0;
    for (i, d) in drop.iter().enumerate() {
        if !d {
            remap[i] = kept;
            kept += 1;
        }
    }
    let mut new_names: Vec<(String, String)> = Vec::with_capacity(kept);
    let mut new_bases: Vec<usize> = Vec::with_capacity(kept);
    let mut new_abs: Vec<u64> = Vec::with_capacity(kept);
    for (i, d) in drop.iter().enumerate() {
        if !d {
            new_names.push(mc.import_names[i].clone());
            new_bases.push(mc.import_bases.get(i).copied().unwrap_or(usize::MAX));
            new_abs.push(mc.import_abs.get(i).copied().unwrap_or(u64::MAX));
        }
    }
    mc.import_names = new_names;
    mc.import_bases = new_bases;
    mc.import_abs = new_abs;
    for r in mc.relocs.iter_mut() {
        if let Sym::Import(i) = r.sym {
            r.sym = Sym::Import(remap[i]);
        }
    }
}

/// Resolve unresolved imports against the merged symbol tables: functions
/// point their slot at the merged body; objects rewrite every referencing
/// reloc onto the merged row/slot. Dead slots (consumed objects, or null
/// slots no reloc references) are compacted away and `Sym::Import`
/// indices remapped — a leftover null `.so`-hinted slot would trip the
/// standalone-image check, so only live slots survive.
fn resolve_against(
    mc: &mut crate::native::target::MachineCode,
    funcs: &std::collections::HashMap<String, usize>,
    objects: &std::collections::HashMap<String, StaticObj>,
) -> Result<(), String> {
    use crate::native::target::Sym;
    for (i, (_, entry)) in mc.import_names.iter().enumerate() {
        let base_missing = mc.import_bases.get(i).copied() == Some(usize::MAX);
        let abs_missing = mc.import_abs.get(i).copied().unwrap_or(u64::MAX) == u64::MAX;
        if base_missing && abs_missing
            && let Some(base) = funcs.get(entry)
        {
            if let Some(slot) = mc.import_bases.get_mut(i) {
                *slot = *base;
            }
        }
    }
    let mut consumed = vec![false; mc.import_names.len()];
    for r in mc.relocs.iter_mut() {
        if let Sym::Import(i) = r.sym {
            // Corrupt members (a relocation naming a slot beyond the
            // import table) fail here, loudly — never panic, never a
            // silently unpatched image.
            if i >= mc.import_names.len() {
                return Err(format!(
                    "static link failed: relocation references import slot {i} with only {} imports",
                    mc.import_names.len()
                ));
            }
            if let Some(obj) = objects.get(&mc.import_names[i].1) {
                r.sym = match obj {
                    StaticObj::Vtable(off) => Sym::VtableOff(*off),
                    StaticObj::Static(idx) => Sym::Static(*idx),
                };
                consumed[i] = true;
            }
        }
    }
    // Slots no reloc references are dead weight; drop them together
    // with consumed object slots (null function slots stay: the
    // standalone-image check reports them loudly).
    let mut referenced = vec![false; mc.import_names.len()];
    for r in mc.relocs.iter() {
        if let Sym::Import(i) = r.sym {
            referenced[i] = true;
        }
    }
    let drop: Vec<bool> = mc.import_names
        .iter()
        .enumerate()
        .map(|(i, _)| {
            consumed[i]
                || (!referenced[i]
                    && mc.import_bases.get(i).copied() == Some(usize::MAX)
                    && mc.import_abs.get(i).copied().unwrap_or(u64::MAX) == u64::MAX)
        })
        .collect();
    if drop.iter().any(|&d| d) {
        let mut remap = vec![0usize; mc.import_names.len()];
        let mut kept = 0;
        for (i, d) in drop.iter().enumerate() {
            if !d {
                remap[i] = kept;
                kept += 1;
            }
        }
        let mut names = Vec::with_capacity(kept);
        let mut bases = Vec::with_capacity(kept);
        let mut abs = Vec::with_capacity(kept);
        for (i, d) in drop.iter().enumerate() {
            if !d {
                names.push(mc.import_names[i].clone());
                bases.push(mc.import_bases.get(i).copied().unwrap_or(usize::MAX));
                abs.push(mc.import_abs.get(i).copied().unwrap_or(u64::MAX));
            }
        }
        mc.import_names = names;
        mc.import_bases = bases;
        mc.import_abs = abs;
        for r in mc.relocs.iter_mut() {
            if let Sym::Import(i) = r.sym {
                r.sym = Sym::Import(remap[i]);
            }
        }
    }
    Ok(())
}

/// Function-level dead-code elimination over a linked image (`--optimize`).
///
/// Reachability from roots: function 0 (the `main` entry), every export
/// (shared-library ABI keeps them all), `@init`, and every function whose
/// address is taken (a `Sym::Fn` target of a data-section relocation —
/// vtable rows, static function pointers). From there, transitively, every
/// code range live code references: functions, runtime helpers, and
/// extracted import bodies. Unreachable ranges leave the code blob and
/// every index space is remapped (`fn_bases`, exports, `init_fn`, helper
/// offsets, import slots, relocations). The data blob is kept whole —
/// strings, vtable rows, and static slots are address-taken by design.
///
/// Safety posture: this is an optimization, so on any structural surprise
/// (unsorted bases, out-of-range indices, relocations naming missing
/// slots) it bails out leaving the image untouched — the existing passes
/// report such images loudly in their own voice.
pub fn dead_code_eliminate(mc: &mut crate::native::target::MachineCode) {
    use crate::native::target::{Section, Sym};
    if mc.code.is_empty() || mc.fn_bases.is_empty() {
        return;
    }
    // Function bases arrive in emission order (backend extraction, then
    // link-time appends); anything else means this image was built by a
    // path DCE does not understand — leave it alone.
    if mc.fn_bases.windows(2).any(|w| w[0] > w[1]) {
        return;
    }

    /// One identity living at a code offset: a function (`fn_bases`
    /// index), a runtime helper (see `helper_id`), or an extracted
    /// import body (import slot).
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Anchor {
        Fn(usize),
        Helper(u8),
        Import(usize),
    }
    /// Helper field order: itoa, itoa_err, scan_int, retain, release,
    /// alloc, exc_enter, exc_pop, exc_throw, exc_rethrow, exc_load,
    /// exc_clear, argv.
    fn helper_get(mc: &crate::native::target::MachineCode, id: u8) -> Option<usize> {
        match id {
            0 => mc.helper_itoa,
            1 => mc.helper_itoa_err,
            2 => mc.helper_scan_int,
            3 => mc.helper_retain,
            4 => mc.helper_release,
            5 => mc.helper_alloc,
            6 => mc.helper_exc_enter,
            7 => mc.helper_exc_pop,
            8 => mc.helper_exc_throw,
            9 => mc.helper_exc_rethrow,
            10 => mc.helper_exc_load,
            11 => mc.helper_exc_clear,
            12 => mc.helper_argv,
            _ => None,
        }
    }
    fn helper_sym_id(sym: &Sym) -> Option<u8> {
        match sym {
            Sym::HelperItoa => Some(0),
            Sym::HelperItoaErr => Some(1),
            Sym::HelperScanInt => Some(2),
            Sym::HelperRetain => Some(3),
            Sym::HelperRelease => Some(4),
            Sym::HelperAlloc => Some(5),
            Sym::HelperExcEnter => Some(6),
            Sym::HelperExcPop => Some(7),
            Sym::HelperExcThrow => Some(8),
            Sym::HelperExcRethrow => Some(9),
            Sym::HelperExcLoad => Some(10),
            Sym::HelperExcClear => Some(11),
            _ => None,
        }
    }

    // Code anchors: every function base, every present helper, every
    // resolved import body. Equal offsets group into one range (two
    // identities sharing an address live and die together).
    let mut anchors: Vec<(usize, Anchor)> = Vec::new();
    for (i, b) in mc.fn_bases.iter().enumerate() {
        anchors.push((*b, Anchor::Fn(i)));
    }
    for id in 0..13u8 {
        if let Some(off) = helper_get(mc, id) {
            anchors.push((off, Anchor::Helper(id)));
        }
    }
    for (i, base) in mc.import_bases.iter().enumerate() {
        if *base != usize::MAX {
            anchors.push((*base, Anchor::Import(i)));
        }
    }
    anchors.sort_by_key(|(off, _)| *off);
    let mut groups: Vec<(usize, Vec<Anchor>)> = Vec::new();
    for (off, a) in anchors {
        if off > mc.code.len() {
            return;
        }
        if let Some(last) = groups.last_mut() {
            if last.0 == off {
                if !last.1.contains(&a) {
                    last.1.push(a);
                }
                continue;
            }
        }
        groups.push((off, vec![a]));
    }
    // Identity → group lookups for the fixpoint.
    let mut fn_group: Vec<Option<usize>> = vec![None; mc.fn_bases.len()];
    let mut helper_group: [Option<usize>; 13] = [None; 13];
    let mut import_group: Vec<Option<usize>> = vec![None; mc.import_names.len()];
    for (g, (_, ids)) in groups.iter().enumerate() {
        for a in ids {
            match a {
                Anchor::Fn(i) => fn_group[*i] = Some(g),
                Anchor::Helper(id) => helper_group[*id as usize] = Some(g),
                Anchor::Import(i) => {
                    if *i < import_group.len() {
                        import_group[*i] = Some(g);
                    }
                }
            }
        }
    }
    if fn_group.iter().any(|g| g.is_none()) {
        return;
    }
    let group_end = |g: usize| -> usize {
        groups
            .get(g + 1)
            .map(|(off, _)| *off)
            .unwrap_or(mc.code.len())
    };
    // Relocation owner: the group whose range holds a code-section site.
    // Ownerless relocations (out-of-range sites) keep their targets alive
    // conservatively instead of dropping reachable code.
    let mut owner: Vec<Option<usize>> = vec![None; mc.relocs.len()];
    for (ri, r) in mc.relocs.iter().enumerate() {
        if r.section != Section::Code {
            continue;
        }
        let mut lo = 0usize;
        let mut hi = groups.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if groups[mid].0 <= r.site {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo > 0 && r.site < group_end(lo - 1) {
            owner[ri] = Some(lo - 1);
        }
    }
    // Validate every index the walk will touch; bail (no-op) on corrupt
    // input rather than panicking or mislinking.
    for r in &mc.relocs {
        match &r.sym {
            Sym::Fn(j) => {
                if *j >= mc.fn_bases.len() {
                    return;
                }
            }
            Sym::Import(i) => {
                if *i >= mc.import_names.len() {
                    return;
                }
            }
            _ => {}
        }
    }
    for (_, idx, _) in mc.exports.iter() {
        if *idx >= mc.fn_bases.len() {
            return;
        }
    }
    if mc.init_fn.is_some_and(|f| f >= mc.fn_bases.len()) {
        return;
    }

    // Roots: entry fn, exports, @init, address-taken functions (data
    // relocs), and anything an ownerless/data reloc names. The argv
    // helper is reached positionally from the entry prologue (no `Sym`
    // edge), so it lives exactly when the entry does.
    let mut live = vec![false; groups.len()];
    let mut queue: Vec<usize> = Vec::new();
    let mut mark_group = |live: &mut Vec<bool>, queue: &mut Vec<usize>, g: usize| {
        if !live[g] {
            live[g] = true;
            queue.push(g);
        }
    };
    if let Some(g) = fn_group[0] {
        mark_group(&mut live, &mut queue, g);
        if let Some(ag) = helper_group[12] {
            mark_group(&mut live, &mut queue, ag);
        }
    }
    for (_, idx, _) in mc.exports.iter() {
        if let Some(g) = fn_group[*idx] {
            mark_group(&mut live, &mut queue, g);
        }
    }
    if let Some(f) = mc.init_fn {
        if let Some(g) = fn_group[f] {
            mark_group(&mut live, &mut queue, g);
        }
    }
    let mut mark_sym = |live: &mut Vec<bool>,
                        queue: &mut Vec<usize>,
                        fn_group: &[Option<usize>],
                        helper_group: &[Option<usize>; 13],
                        import_group: &[Option<usize>],
                        sym: &Sym| {
        match sym {
            Sym::Fn(j) => {
                if let Some(g) = fn_group[*j] {
                    if !live[g] {
                        live[g] = true;
                        queue.push(g);
                    }
                }
            }
            Sym::Import(i) => {
                if let Some(g) = import_group[*i] {
                    if !live[g] {
                        live[g] = true;
                        queue.push(g);
                    }
                }
            }
            s => {
                if let Some(id) = helper_sym_id(s) {
                    if let Some(g) = helper_group[id as usize] {
                        if !live[g] {
                            live[g] = true;
                            queue.push(g);
                        }
                    }
                }
            }
        }
    };
    for (ri, r) in mc.relocs.iter().enumerate() {
        if r.section != Section::Code || owner[ri].is_none() {
            mark_sym(
                &mut live,
                &mut queue,
                &fn_group,
                &helper_group,
                &import_group,
                &r.sym,
            );
        }
    }
    while let Some(g) = queue.pop() {
        for (ri, r) in mc.relocs.iter().enumerate() {
            if owner[ri] == Some(g) {
                mark_sym(
                    &mut live,
                    &mut queue,
                    &fn_group,
                    &helper_group,
                    &import_group,
                    &r.sym,
                );
            }
        }
    }

    // Rebuild: live ranges concatenated in offset order (16-aligned like
    // link-time appends), every index space remapped. Computed fully
    // before touching `mc`, so a late inconsistency still bails cleanly.
    let mut new_code: Vec<u8> = Vec::new();
    let mut new_off: Vec<Option<usize>> = vec![None; groups.len()];
    for (g, (off, _)) in groups.iter().enumerate() {
        if !live[g] {
            continue;
        }
        while !new_code.len().is_multiple_of(16) {
            new_code.push(0xCC);
        }
        let end = group_end(g);
        if end < *off || end > mc.code.len() {
            return;
        }
        new_off[g] = Some(new_code.len());
        new_code.extend_from_slice(&mc.code[*off..end]);
    }
    let mut fn_remap: Vec<Option<usize>> = vec![None; mc.fn_bases.len()];
    let mut new_fn_bases: Vec<usize> = Vec::new();
    for (i, g) in fn_group.iter().enumerate() {
        let g = match g {
            Some(g) => *g,
            None => return,
        };
        if !live[g] {
            continue;
        }
        fn_remap[i] = Some(new_fn_bases.len());
        new_fn_bases.push(new_off[g].expect("live group has an offset"));
    }
    // Every export names a live function (exports are roots); anything
    // else is an internal contradiction — bail untouched.
    let mut new_exports = Vec::with_capacity(mc.exports.len());
    for (name, idx, arity) in mc.exports.iter() {
        match fn_remap[*idx] {
            Some(n) => new_exports.push((name.clone(), n, *arity)),
            None => return,
        }
    }
    let new_init = match mc.init_fn {
        Some(f) => match fn_remap[f] {
            Some(n) => Some(n),
            None => return,
        },
        None => None,
    };
    let mut get_helper = |id: u8| -> Option<usize> {
        let g = helper_group[id as usize]?;
        if !live[g] {
            return None;
        }
        new_off[g]
    };
    // Import slots survive iff a kept relocation references them;
    // resolved bodies ride their (live) ranges, null slots stay null.
    let mut slot_live = vec![false; mc.import_names.len()];
    for (ri, r) in mc.relocs.iter().enumerate() {
        let keep = match r.section {
            Section::Data => true,
            Section::Code => owner[ri].is_some_and(|g| live[g]),
        };
        if keep {
            if let Sym::Import(i) = r.sym {
                slot_live[i] = true;
            }
        }
    }
    let mut slot_remap: Vec<Option<usize>> = vec![None; mc.import_names.len()];
    let mut new_names = Vec::new();
    let mut new_imp_bases: Vec<usize> = Vec::new();
    let mut new_abs = Vec::new();
    for (i, name) in mc.import_names.iter().enumerate() {
        if !slot_live[i] {
            continue;
        }
        // Resolved bodies ride their (live) ranges: rebase onto the new
        // group offset. A kept slot whose body range died is an internal
        // contradiction (a kept `Sym::Import` reloc marks the body group
        // live) — bail untouched. Null slots stay null.
        let old_base = mc.import_bases.get(i).copied().unwrap_or(usize::MAX);
        let new_base = if old_base == usize::MAX {
            usize::MAX
        } else {
            let g = match import_group[i] {
                Some(g) => g,
                None => return,
            };
            if !live[g] {
                return;
            }
            match new_off[g] {
                Some(o) => o,
                None => return,
            }
        };
        slot_remap[i] = Some(new_names.len());
        new_names.push(name.clone());
        new_imp_bases.push(new_base);
        new_abs.push(mc.import_abs.get(i).copied().unwrap_or(u64::MAX));
    }
    mc.code = new_code;
    mc.fn_bases = new_fn_bases;
    mc.exports = new_exports;
    mc.init_fn = new_init;
    mc.helper_itoa = get_helper(0);
    mc.helper_itoa_err = get_helper(1);
    mc.helper_scan_int = get_helper(2);
    mc.helper_retain = get_helper(3);
    mc.helper_release = get_helper(4);
    mc.helper_alloc = get_helper(5);
    mc.helper_exc_enter = get_helper(6);
    mc.helper_exc_pop = get_helper(7);
    mc.helper_exc_throw = get_helper(8);
    mc.helper_exc_rethrow = get_helper(9);
    mc.helper_exc_load = get_helper(10);
    mc.helper_exc_clear = get_helper(11);
    mc.helper_argv = get_helper(12);
    mc.import_names = new_names;
    mc.import_bases = new_imp_bases;
    mc.import_abs = new_abs;
    // Relocations: keep data relocs and live-owned code relocs, rebased
    // and remapped onto the new index spaces.
    let mut new_relocs = Vec::with_capacity(mc.relocs.len());
    for (ri, r) in mc.relocs.iter().enumerate() {
        let owned_live = r.section == Section::Code && owner[ri].is_some_and(|g| live[g]);
        if r.section != Section::Data && !owned_live {
            continue;
        }
        let mut nr = r.clone();
        if r.section == Section::Code {
            let g = owner[ri].expect("owned-live reloc has a group");
            let (off, _) = groups[g];
            nr.site = new_off[g].expect("live group has an offset") + (r.site - off);
        }
        nr.sym = match &r.sym {
            Sym::Fn(j) => match fn_remap[*j] {
                Some(n) => Sym::Fn(n),
                None => return,
            },
            Sym::Import(i) => match slot_remap[*i] {
                Some(n) => Sym::Import(n),
                None => return,
            },
            s => {
                if helper_sym_id(s).is_some() {
                    // Referenced helpers survive exactly when referenced:
                    // a kept reloc names a live helper group already.
                    s.clone()
                } else {
                    s.clone()
                }
            }
        };
        new_relocs.push(nr);
    }
    mc.relocs = new_relocs;
}
/// Copy-link `[LibraryImport]` bodies from `--lib` ELF files into
/// the code blob and filling `import_bases`. Fails with a readable message
/// listing any missing symbols.
///
/// `allow_loader_imports` is true only when the caller resolves
/// `.so`-hinted slots with a loader afterwards (the JIT's dlopen pass).
/// Executables and produced shared images have no loader: they pass false
/// and get a loud error instead of an image that calls a null slot.
pub fn resolve_imports(
    mc: &mut crate::native::target::MachineCode,
    libs: &[PathBuf],
    allow_loader_imports: bool,
) -> Result<(), String> {
    // Slots the static link step already resolved keep their bodies;
    // only null slots are (re)resolved here.
    if mc.import_bases.len() < mc.import_names.len() {
        mc.import_bases
            .resize(mc.import_names.len(), usize::MAX);
    }
    if mc.import_names.is_empty() {
        return Ok(());
    }

    // Per image: code symbols (copy-linkable bodies), data symbols
    // (static slots / vtable rows: address-only), and the bytes.
    // `LoadImport` may name a DATA symbol — a `.so`-owned static slot
    // this exe must read (e.g. CoreLib's
    // `OrdinalCasing::s_lowerCasingTable` when CoreLib was not merged
    // as an archive). Extracting its "body" spliced the data blob
    // into the CODE blob and the program executed data bytes
    // (`mov 0x18(%r10)`); instead copy the 8 data bytes into this
    // image's DATA section and point the import slot at that address.
    type SymbolTable = Vec<(String, std::ops::Range<usize>)>;
    let mut images: Vec<(PathBuf, Vec<u8>, SymbolTable, SymbolTable)> = Vec::new();
    let mut runtime_only: Vec<PathBuf> = Vec::new();
    for lib in libs {
        let bytes = std::fs::read(lib)
            .map_err(|e| format!("cannot read library {}: {e}", lib.display()))?;
        if is_rubyc_shared_image(&bytes) {
            runtime_only.push(lib.clone());
            continue;
        }
        let syms = exported_functions(&bytes).map_err(|e| format!("{}: {e}", lib.display()))?;
        let objs = object_symbols(&bytes).map_err(|e| format!("{}: {e}", lib.display()))?;
        images.push((lib.clone(), bytes, syms, objs));
    }

    let mut missing: Vec<(String, usize)> = Vec::new();
    let mut missing_dep: Vec<(String, String)> = Vec::new();
    for (i, (lib_hint, entry)) in mc.import_names.iter().enumerate() {
        // Slots the static link step already resolved keep their bodies.
        // Absent sidecar entries count as null (unresolved), never as
        // resolved: backends may leave them empty.
        if matches!(mc.import_bases.get(i).copied(), Some(b) if b != usize::MAX)
            || matches!(mc.import_abs.get(i).copied(), Some(a) if a != u64::MAX)
        {
            continue;
        }
        // Dependency-image imports (`lib_hint` is a `.so` file name, e.g.
        // `System.Private.CoreLib.so`): the callee's body stays in its
        // home image and is resolved by a loader at load time (dlopen +
        // dlsym). Extracting (copy-linking) the body here would break its
        // absolute references to the home image. Only a host with a
        // loader (the JIT) may leave the slot null; an executable or a
        // produced shared image has none, so it must report instead of
        // shipping a null call.
        if lib_hint.ends_with(".so") {
            if allow_loader_imports {
                continue;
            }
            missing_dep.push((entry.clone(), lib_hint.clone()));
            continue;
        }
        // Prefer an image whose file name contains the library hint
        // ("native_add" matches libnative_add.so); fall back to any image
        // exporting the entry point.
        let mut found = None;
        for (path, bytes, syms, objs) in &images {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !name.contains(lib_hint.as_str()) {
                continue;
            }
            if let Some((_, range)) = syms.iter().find(|(n, _)| n == entry) {
                found = Some((range.clone(), bytes));
                break;
            }
        }
        let (range, bytes) = match found.or_else(|| {
            images.iter().find_map(|(_, bytes, syms, _)| {
                syms.iter()
                    .find(|(name, _)| name == entry)
                    .map(|(_, range)| (range.clone(), bytes))
            })
        }) {
            Some(f) => f,
            None => {
                // Collect every missing import and report them together:
                // one-at-a-time errors turn stub authoring (host shims for
                // runtime natives) into a build-per-symbol grind.
                missing.push((entry.clone(), i));
                continue;
            }
        };
        // Align each extracted body to 16 bytes for cleanliness.
        while !mc.code.len().is_multiple_of(16) {
            mc.code.push(0xCC);
        }
        mc.import_bases[i] = mc.code.len();
        mc.code.extend_from_slice(&bytes[range]);
    }
    if !missing.is_empty() || !missing_dep.is_empty() {
        // A missing import is always an error. It used to be possible
        // to opt into fail-open linking (RUBYC_ALLOW_MISSING_IMPORTS),
        // which emitted a `xor eax,eax; ret` stub so every uncalled
        // symbol linked cleanly — and any *called* one silently
        // returned 0 at runtime instead of failing.
        let provided: Vec<String> = libs.iter().map(|p| p.display().to_string()).collect();
        let mut msg = String::new();
        if !missing.is_empty() {
            msg.push_str(&format!(
                "imports not found in provided libraries {provided:?}: {}",
                missing.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>().join(", ")
            ));
        }
        if !missing_dep.is_empty() {
            if !msg.is_empty() {
                msg.push_str("; ");
            }
            msg.push_str(&format!(
                "dependency-image imports need a loader, and this image has none: {} — run it in the JIT, link the matching .a archive, or build the dependency from sources",
                missing_dep
                    .iter()
                    .map(|(entry, hint)| format!("{entry} from {hint}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if !runtime_only.is_empty() {
            // .NET semantics: an assembly is a runtime dependency, not a
            // body donor. Say so instead of emitting a crashing copy.
            msg.push_str(&format!(
                "; RubyC shared libraries are runtime dependencies and are not copy-linked: {} — link the matching .a archive or build a dynamic consumer",
                runtime_only
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        return Err(msg);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_mc() -> crate::native::target::MachineCode {
        crate::native::target::MachineCode {
            code: Vec::new(),
            data: Vec::new(),
            fn_bases: Vec::new(),
            vtable_offsets: Vec::new(),
            vtable_names: Vec::new(),
            static_offsets: Vec::new(),
            static_names: Vec::new(),
            import_names: Vec::new(),
            import_bases: Vec::new(),
            import_abs: Vec::new(),
            exports: Vec::new(),
            export_takes_receiver: vec![],
            init_fn: None,
            relocs: Vec::new(),
            heap_size: 0,
            strip_symbols: false,
            helper_itoa: None,
            helper_itoa_err: None,
            helper_scan_int: None,
            helper_retain: None,
            helper_release: None,
            helper_alloc: None,
            helper_exc_enter: None,
            helper_exc_pop: None,
            helper_exc_throw: None,
            helper_exc_rethrow: None,
            helper_exc_load: None,
            helper_exc_clear: None,
            exc_state_offset: None,
            helper_argv: None,
        }
    }

    /// Minimal ELF64 with one `.symtab` holding a FUNC (`fn_body`) and
    /// an OBJECT (`data_slot`) symbol over a `.text`-style blob — the
    /// shape of a `--lib` image whose export table mixes code and
    /// data, as every real RubyC `.so` does.
    fn image_with_func_and_data_symbols() -> Vec<u8> {
        // Non-overlapping layout: bodies 0x600, strtab 0x620,
        // symtab 0x680, section headers 0x700 (4 × 64).
        const TEXT: usize = 0x600;
        let mut img = vec![0u8; 0x900];
        // ELF header: ET_REL(1), machine x86-64, section info.
        img[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
        img[4] = 2; // ELFCLASS64
        img[5] = 1; // little endian
        img[16..18].copy_from_slice(&1u16.to_le_bytes()); // e_type
        img[18..20].copy_from_slice(&[62, 0]); // e_machine x86-64
        // Bodies: 16 code bytes, then an 8-byte data slot.
        for (i, b) in [0x90u8; 16].iter().enumerate() {
            img[TEXT + i] = *b;
        }
        img[TEXT + 16..TEXT + 24].copy_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes());
        // strtab: "\0fn_body\0data_slot\0".
        let strtab_off = 0x620;
        let mut strings = vec![0u8];
        let fn_name = strings.len();
        strings.extend_from_slice(b"fn_body\0");
        let data_name = strings.len();
        strings.extend_from_slice(b"data_slot\0");
        img[strtab_off..strtab_off + strings.len()].copy_from_slice(&strings);
        // symtab at 0x680: null entry + one FUNC + one OBJECT.
        let symtab_off = 0x680;
        let mut entry = |at: usize, name: usize, info: u8, shndx: u16, value: u64, size: u64| {
            img[at..at + 4].copy_from_slice(&(name as u32).to_le_bytes());
            img[at + 4] = info;
            img[at + 6..at + 8].copy_from_slice(&shndx.to_le_bytes());
            img[at + 8..at + 16].copy_from_slice(&value.to_le_bytes());
            img[at + 16..at + 24].copy_from_slice(&size.to_le_bytes());
        };
        entry(symtab_off + 24, fn_name, 0x12, 3, TEXT as u64, 16); // STB_GLOBAL|STT_FUNC
        entry(symtab_off + 48, data_name, 0x11, 3, (TEXT + 16) as u64, 8); // STB_GLOBAL|STT_OBJECT
        // Section headers: null, .symtab(2), .strtab(3), .text(1).
        let shoff = 0x700usize;
        let mut shdr = |at: usize, sh_type: u32, addr: u64, offset: u64, size: u64, link: u32, entsize: u64| {
            // sh_name occupies 0..4; sh_type lives at +4.
            img[at + 4..at + 8].copy_from_slice(&sh_type.to_le_bytes());
            img[at + 16..at + 24].copy_from_slice(&addr.to_le_bytes());
            img[at + 24..at + 32].copy_from_slice(&offset.to_le_bytes());
            img[at + 32..at + 40].copy_from_slice(&size.to_le_bytes());
            img[at + 40..at + 44].copy_from_slice(&link.to_le_bytes());
            img[at + 56..at + 64].copy_from_slice(&entsize.to_le_bytes());
        };
        // Header 0 stays the ELF-mandated null entry; .symtab(1),
        // .strtab(2), .text(3) follow, so shndx=3 addresses .text.
        shdr(shoff + 64, 2, 0, symtab_off as u64, 72, 2, 24);
        shdr(shoff + 128, 3, 0, strtab_off as u64, strings.len() as u64, 0, 0);
        shdr(shoff + 192, 1, TEXT as u64, TEXT as u64, 64, 0, 0);
        img[0x28..0x30].copy_from_slice(&(shoff as u64).to_le_bytes());
        img[0x3A..0x3C].copy_from_slice(&64u16.to_le_bytes()); // e_shentsize
        img[0x3C..0x3E].copy_from_slice(&4u16.to_le_bytes()); // e_shnum
        img
    }

    #[test]
    fn function_and_data_symbols_are_classified_separately() {
        let img = image_with_func_and_data_symbols();
        let funcs: Vec<String> = exported_functions(&img)
            .expect("parses")
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        let objs: Vec<String> = object_symbols(&img)
            .expect("parses")
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert_eq!(funcs, vec!["fn_body".to_owned()], "code symbols only");
        assert_eq!(objs, vec!["data_slot".to_owned()], "data symbols only");
    }

    #[test]
    fn dep_so_imports_stay_null_for_loader() {
        let mut mc = empty_mc();
        mc.import_names = vec![
            ("DepLib.so".to_owned(), "Ns.Class::Method#0".to_owned()),
        ];
        resolve_imports(&mut mc, &[], true).expect("dep imports never fail link");
        assert_eq!(mc.import_bases, vec![usize::MAX]);
        assert!(mc.code.is_empty(), "no body extracted for dep imports");
    }

    #[test]
    fn static_link_merges_member_and_resolves_imports() {
        use crate::native::target::{Reloc, RelocKind, Section, Sym};
        // Member: one function body plus one vtable row.
        let mut member = empty_mc();
        member.code = vec![0x90; 16];
        member.data = vec![0; 8];
        member.fn_bases = vec![0];
        member.exports = vec![("M::F#0".to_owned(), 0, 0)];
        member.vtable_offsets = vec![0];
        member.vtable_names = vec!["M.C".to_owned()];
        // Main: a function import and an object import, both unresolved.
        let mut main = empty_mc();
        main.import_names = vec![
            ("M.a".to_owned(), "M::F#0".to_owned()),
            ("M.a".to_owned(), "M.C::__vtable".to_owned()),
        ];
        main.import_bases = vec![usize::MAX, usize::MAX];
        main.relocs = vec![
            Reloc { kind: RelocKind::Abs64, section: Section::Code, site: 0, sym: Sym::Import(0) },
            Reloc { kind: RelocKind::Abs64, section: Section::Code, site: 8, sym: Sym::Import(1) },
        ];
        main.code = vec![0; 16];
        link_static_archives(&mut main, vec![("M".to_owned(), member)])
            .expect("static link resolves");
        // Function slot points at the merged body (appended at offset 16).
        assert_eq!(main.import_bases[0], 16, "fn import base");
        assert_eq!(&main.code[16..32], &[0x90; 16], "member body merged");
        // Object slot is consumed: reloc rewritten onto the merged row,
        // dead slot compacted away (one live import remains).
        assert_eq!(main.import_names.len(), 1, "dead slot compacted");
        assert_eq!(main.import_names[0].1, "M::F#0");
        assert!(
            matches!(main.relocs[1].sym, Sym::VtableOff(0)),
            "object reloc rewritten: {:?}",
            main.relocs[1].sym
        );
        assert!(
            matches!(main.relocs[0].sym, Sym::Import(0)),
            "function reloc keeps remapped slot: {:?}",
            main.relocs[0].sym
        );
    }

    #[test]
    fn static_link_missing_export_errors_loudly() {
        let mut main = empty_mc();
        main.import_names = vec![("M.a".to_owned(), "Nope::Missing#0".to_owned())];
        main.import_bases = vec![usize::MAX];
        let member = empty_mc();
        let err = match link_static_archives(&mut main, vec![("M".to_owned(), member)]) {
            Ok(()) => panic!("missing export must fail"),
            Err(e) => e,
        };
        assert!(err.contains("Nope::Missing#0"), "names the entry: {err}");
        assert!(err.contains('M'), "names the searched archive: {err}");
    }

    #[test]
    fn dep_import_load_missing_image_errors() {
        // Success needs dlopen (VM-gated); the missing-file error is
        // local-safe and pins the lookup path.
        let names = vec![("Nope.so".to_owned(), "A::B#0".to_owned())];
        let mut slots = vec![0u64];
        let dir = std::path::Path::new("/nonexistent-rubyc-libraries");
        let err = match resolve_dep_imports(&names, &mut slots, dir) {
            Ok(_) => panic!("missing image must fail"),
            Err(e) => e,
        };
        assert!(err.contains("Nope.so"), "names the image: {err}");
        assert_eq!(slots, vec![0], "slots untouched on failure");
    }

    #[test]
    fn dep_import_load_skips_native_hints() {
        let names = vec![("native_add".to_owned(), "add".to_owned())];
        let mut slots = vec![0u64];
        let dir = std::path::Path::new("/nonexistent-rubyc-libraries");
        let handles =
            resolve_dep_imports(&names, &mut slots, dir).expect("native hints are not dep imports");
        assert!(handles.is_empty());
    }

    #[test]
    fn static_link_rebases_every_reloc_kind() {
        use crate::native::target::{Reloc, RelocKind, Section, Sym};
        // Non-empty main so all three index spaces shift: code 20 pads
        // to 32, data 10 pads to 16, one fn and one static slot each.
        let mut member = empty_mc();
        member.code = vec![0xAA; 8];
        member.data = vec![0xBB; 4];
        member.fn_bases = vec![0, 4];
        member.vtable_offsets = vec![0];
        member.static_offsets = vec![0];
        member.exports = vec![("M::F#0".to_owned(), 0, 0)];
        member.import_names = vec![("M.a".to_owned(), "M::G#0".to_owned())];
        member.import_bases = vec![4];
        member.relocs = vec![
            Reloc { kind: RelocKind::Abs64, section: Section::Code, site: 0, sym: Sym::Fn(0) },
            Reloc { kind: RelocKind::Rel32Call, section: Section::Code, site: 2, sym: Sym::String(2) },
            Reloc { kind: RelocKind::Abs64, section: Section::Code, site: 4, sym: Sym::HelperAlloc },
            Reloc { kind: RelocKind::Abs64, section: Section::Data, site: 0, sym: Sym::VtableOff(0) },
            Reloc { kind: RelocKind::Abs64, section: Section::Data, site: 2, sym: Sym::Static(0) },
            Reloc { kind: RelocKind::Abs64, section: Section::Data, site: 4, sym: Sym::Import(0) },
            Reloc { kind: RelocKind::Abs64, section: Section::Data, site: 6, sym: Sym::ExcState(1) },
            Reloc { kind: RelocKind::Abs64, section: Section::Data, site: 8, sym: Sym::HeapBase },
            Reloc { kind: RelocKind::Abs64, section: Section::Data, site: 10, sym: Sym::Name(3) },
        ];
        let mut main = empty_mc();
        main.code = vec![0; 20];
        main.data = vec![0; 10];
        main.fn_bases = vec![0];
        main.static_offsets = vec![0];
        main.import_names = vec![("M.a".to_owned(), "M::F#0".to_owned())];
        main.import_bases = vec![usize::MAX];
        link_static_archives(&mut main, vec![("M".to_owned(), member)])
            .expect("static link resolves");
        // code_base 32, data_base 16, fn_base 1, sb_base 1, imp_base 1.
        assert_eq!(&main.code[32..40], &[0xAA; 8]);
        assert_eq!(&main.data[16..20], &[0xBB; 4]);
        assert_eq!(main.fn_bases, vec![0, 32, 36], "member bodies appended");
        assert_eq!(main.vtable_offsets, vec![16]);
        assert_eq!(main.static_offsets, vec![0, 16]);
        let got: Vec<(usize, Sym)> = main.relocs.iter().map(|r| (r.site, r.sym.clone())).collect();
        assert!(got.contains(&(32, Sym::Fn(1))), "fn index rebased: {got:?}");
        assert!(got.contains(&(34, Sym::String(18))), "string offset rebased: {got:?}");
        assert!(
            got.contains(&(36, Sym::HelperAlloc)),
            "helper refs untouched: {got:?}"
        );
        assert!(got.contains(&(16, Sym::VtableOff(16))), "vtable offset rebased: {got:?}");
        assert!(got.contains(&(18, Sym::Static(1))), "static index rebased: {got:?}");
        assert!(got.contains(&(20, Sym::Import(1))), "import index rebased: {got:?}");
        assert!(got.contains(&(22, Sym::ExcState(17))), "excstate offset rebased: {got:?}");
        assert!(got.contains(&(24, Sym::HeapBase)), "heap ref untouched: {got:?}");
        assert!(got.contains(&(26, Sym::Name(3))), "name ref untouched: {got:?}");
        // Member's own resolved import rode along, rebased by code_base.
        assert_eq!(main.import_bases, vec![32, 36]);
    }

    #[test]
    fn static_link_rewrites_static_slot_import() {
        use crate::native::target::{Reloc, RelocKind, Section, Sym};
        let mut member = empty_mc();
        member.static_offsets = vec![4];
        member.static_names = vec!["M.C::counter".to_owned()];
        let mut main = empty_mc();
        main.import_names = vec![("M.a".to_owned(), "M.C::counter".to_owned())];
        main.import_bases = vec![usize::MAX];
        main.relocs = vec![
            Reloc { kind: RelocKind::Abs64, section: Section::Code, site: 0, sym: Sym::Import(0) },
        ];
        link_static_archives(&mut main, vec![("M".to_owned(), member)])
            .expect("static link resolves");
        assert!(
            matches!(main.relocs[0].sym, Sym::Static(0)),
            "object reloc rewritten: {:?}",
            main.relocs[0].sym
        );
        assert!(main.import_names.is_empty(), "consumed slot compacted");
    }

    #[test]
    fn static_link_helper_adopt_rules() {
        // Main has it, member has it: main's home wins.
        let mut member = empty_mc();
        member.code = vec![0x90; 16];
        member.helper_alloc = Some(0);
        let mut main = empty_mc();
        main.helper_alloc = Some(100);
        main.import_names = vec![("M.a".to_owned(), "M::F#0".to_owned())];
        main.import_bases = vec![usize::MAX];
        member.exports = vec![("M::F#0".to_owned(), 0, 0)];
        member.fn_bases = vec![0];
        link_static_archives(&mut main, vec![("M".to_owned(), member)])
            .expect("static link resolves");
        assert_eq!(main.helper_alloc, Some(100), "existing home wins");
        // Main lacks it, member has it: adopted, rebased to the merge.
        let mut member = empty_mc();
        member.code = vec![0x90; 16];
        member.helper_alloc = Some(4);
        member.exports = vec![("M::F#0".to_owned(), 0, 0)];
        member.fn_bases = vec![0];
        let mut main = empty_mc();
        main.import_names = vec![("M.a".to_owned(), "M::F#0".to_owned())];
        main.import_bases = vec![usize::MAX];
        link_static_archives(&mut main, vec![("M".to_owned(), member)])
            .expect("static link resolves");
        assert_eq!(main.helper_alloc, Some(4), "adopted rebased (code_base 0)");
        // Neither has it: stays absent.
        let mut member = empty_mc();
        member.exports = vec![("M::F#0".to_owned(), 0, 0)];
        member.fn_bases = vec![0];
        let mut main = empty_mc();
        main.import_names = vec![("M.a".to_owned(), "M::F#0".to_owned())];
        main.import_bases = vec![usize::MAX];
        link_static_archives(&mut main, vec![("M".to_owned(), member)])
            .expect("static link resolves");
        assert_eq!(main.helper_alloc, None);
    }

    #[test]
    fn static_link_init_conflict_errors_loudly() {
        let mut member = empty_mc();
        member.init_fn = Some(0);
        member.fn_bases = vec![0];
        let mut main = empty_mc();
        main.init_fn = Some(0);
        main.fn_bases = vec![0];
        main.import_names = vec![("M.a".to_owned(), "M::F#0".to_owned())];
        main.import_bases = vec![usize::MAX];
        member.exports = vec![("M::F#0".to_owned(), 0, 0)];
        let err = match link_static_archives(&mut main, vec![("M".to_owned(), member)]) {
            Ok(()) => panic!("double @init must fail"),
            Err(e) => e,
        };
        assert!(err.contains('M'), "names the member: {err}");
        assert!(err.contains("@init"), "names the conflict: {err}");
    }

    #[test]
    fn static_link_diamond_first_wins() {
        let mut first = empty_mc();
        first.code = vec![0x11; 16];
        first.fn_bases = vec![0];
        first.exports = vec![("D::F#0".to_owned(), 0, 0)];
        let mut second = empty_mc();
        second.code = vec![0x22; 16];
        second.fn_bases = vec![0];
        second.exports = vec![("D::F#0".to_owned(), 0, 0)];
        let mut main = empty_mc();
        main.import_names = vec![("D.a".to_owned(), "D::F#0".to_owned())];
        main.import_bases = vec![usize::MAX];
        link_static_archives(
            &mut main,
            vec![("D1".to_owned(), first), ("D2".to_owned(), second)],
        )
        .expect("static link resolves");
        // First member merged at base 0 and satisfied the only import;
        // the second exports nothing still wanted, so it never merges
        // (dead code stays out of the image).
        assert_eq!(main.import_bases[0], 0);
        assert_eq!(&main.code[..], &[0x11; 16]);
    }

    #[test]
    fn static_link_transitive_fixpoint() {
        // Main wants A::F, which lives in member A; A's body in turn
        // imports B::G from member B. One call merges both.
        let mut b = empty_mc();
        b.code = vec![0xBB; 16];
        b.fn_bases = vec![0];
        b.exports = vec![("B::G#0".to_owned(), 0, 0)];
        let mut a = empty_mc();
        a.code = vec![0xAA; 16];
        a.fn_bases = vec![0];
        a.exports = vec![("A::F#0".to_owned(), 0, 0)];
        a.import_names = vec![("B.a".to_owned(), "B::G#0".to_owned())];
        a.import_bases = vec![usize::MAX];
        a.relocs = vec![
            crate::native::target::Reloc {
                kind: crate::native::target::RelocKind::Abs64,
                section: crate::native::target::Section::Code,
                site: 0,
                sym: crate::native::target::Sym::Import(0),
            },
        ];
        let mut main = empty_mc();
        main.import_names = vec![("A.a".to_owned(), "A::F#0".to_owned())];
        main.import_bases = vec![usize::MAX];
        link_static_archives(
            &mut main,
            vec![("A".to_owned(), a), ("B".to_owned(), b)],
        )
        .expect("static link resolves transitively");
        assert_eq!(main.import_bases[0], 0, "A body at base 0");
        assert_eq!(main.import_bases[1], 16, "B body appended, A import rebased");
        assert_eq!(&main.code[0..16], &[0xAA; 16]);
        assert_eq!(&main.code[16..32], &[0xBB; 16]);
    }

    #[test]
    fn static_link_rebases_resolved_member_imports() {
        use crate::native::target::{Reloc, RelocKind, Section, Sym};
        // Member arrives with one import already resolved (points into
        // its own code); the base must shift with the merge.
        let mut member = empty_mc();
        member.code = vec![0x90; 32];
        member.fn_bases = vec![0, 16];
        member.exports = vec![("M::F#0".to_owned(), 0, 0)];
        member.import_names = vec![("M.a".to_owned(), "M::G#0".to_owned())];
        member.import_bases = vec![16];
        member.relocs = vec![
            Reloc { kind: RelocKind::Abs64, section: Section::Code, site: 0, sym: Sym::Import(0) },
        ];
        let mut main = empty_mc();
        main.code = vec![0; 16];
        main.import_names = vec![("M.a".to_owned(), "M::F#0".to_owned())];
        main.import_bases = vec![usize::MAX];
        link_static_archives(&mut main, vec![("M".to_owned(), member)])
            .expect("static link resolves");
        // code_base 16: member body at 16, resolved import 16+16=32,
        // member reloc remapped to slot 1.
        assert_eq!(main.import_bases[0], 16);
        assert_eq!(main.import_bases[1], 32, "resolved base rebased: {:?}", main.import_bases);
        assert!(
            matches!(main.relocs[0].sym, Sym::Import(1)),
            "member reloc remapped: {:?}",
            main.relocs[0].sym
        );
    }

    #[test]
    fn static_link_resolves_so_hinted_import() {
        // `.so` hints are loader business in dynamic images, but a
        // static image has no loader: the member body satisfies them.
        let mut member = empty_mc();
        member.code = vec![0x90; 16];
        member.fn_bases = vec![0];
        member.exports = vec![("M::F#0".to_owned(), 0, 0)];
        let mut main = empty_mc();
        main.import_names = vec![("M.so".to_owned(), "M::F#0".to_owned())];
        main.import_bases = vec![usize::MAX];
        link_static_archives(&mut main, vec![("M".to_owned(), member)])
            .expect("static link resolves .so hint");
        assert_eq!(main.import_bases[0], 0);
    }

    #[test]
    fn static_link_corrupt_member_errors_not_panics() {
        use crate::native::target::{Reloc, RelocKind, Section, Sym};
        // A member whose relocation names a slot beyond its own import
        // table is corrupt: loud error, never a panic, never a silent
        // unpatched image.
        let mut member = empty_mc();
        member.code = vec![0x90; 16];
        member.fn_bases = vec![0];
        member.exports = vec![("M::F#0".to_owned(), 0, 0)];
        member.relocs = vec![
            Reloc { kind: RelocKind::Abs64, section: Section::Code, site: 0, sym: Sym::Import(7) },
        ];
        let mut main = empty_mc();
        main.import_names = vec![("M.a".to_owned(), "M::F#0".to_owned())];
        main.import_bases = vec![usize::MAX];
        let err = match link_static_archives(&mut main, vec![("M".to_owned(), member)]) {
            Ok(()) => panic!("corrupt member must fail"),
            Err(e) => e,
        };
        assert!(err.contains("import slot 8"), "names the slot: {err}");
    }

    #[test]
    fn static_link_empty_member_sidecars_resolve() {
        // Real backends leave import sidecars empty when nothing was
        // ever extracted: merged slots must still appear (null), never
        // go missing or read as resolved.
        let mut member = empty_mc();
        member.code = vec![0x90; 16];
        member.fn_bases = vec![0];
        member.exports = vec![("M::F#0".to_owned(), 0, 0)];
        member.import_names = vec![("M.a".to_owned(), "M::G#0".to_owned())];
        assert!(member.import_bases.is_empty());
        member.relocs = vec![crate::native::target::Reloc {
            kind: crate::native::target::RelocKind::Abs64,
            section: crate::native::target::Section::Code,
            site: 0,
            sym: crate::native::target::Sym::Import(0),
        }];
        let mut other = empty_mc();
        other.code = vec![0x91; 16];
        other.fn_bases = vec![0];
        other.exports = vec![("M::G#0".to_owned(), 0, 0)];
        let mut main = empty_mc();
        main.import_names = vec![("M.a".to_owned(), "M::F#0".to_owned())];
        main.import_bases = vec![usize::MAX];
        link_static_archives(
            &mut main,
            vec![("M".to_owned(), member), ("N".to_owned(), other)],
        )
        .expect("transitive resolve with empty sidecars");
        assert_eq!(main.import_names.len(), 2);
        assert_eq!(main.import_bases.len(), 2);
        assert_ne!(main.import_bases[0], usize::MAX);
        assert_ne!(main.import_bases[1], usize::MAX);
    }

    #[test]
    fn static_link_appends_exports_helpers_state_heap() {
        let mut member = empty_mc();
        member.code = vec![0x90; 16];
        member.fn_bases = vec![0];
        member.exports = vec![("M::F#0".to_owned(), 0, 1)];
        member.export_takes_receiver = vec![false];
        member.exc_state_offset = Some(0);
        member.data = vec![0; 8];
        member.heap_size = 4096;
        let mut main = empty_mc();
        main.import_names = vec![("M.a".to_owned(), "M::F#0".to_owned())];
        main.import_bases = vec![usize::MAX];
        main.heap_size = 1024;
        link_static_archives(&mut main, vec![("M".to_owned(), member)])
            .expect("static link resolves");
        assert_eq!(main.exports, vec![("M::F#0".to_owned(), 0, 1)], "fn_idx remapped");
        assert_eq!(main.export_takes_receiver, vec![false]);
        assert_eq!(main.exc_state_offset, Some(0), "adopted with data_base 0");
        assert_eq!(main.heap_size, 4096, "heap takes the max");
    }

    /// Three 16-byte functions: fn0 calls fn2, fn1 is dead. The dead
    /// body leaves the blob and fn2's index compacts down.
    #[test]
    fn dce_drops_unreachable_function_and_remaps() {
        use crate::native::target::{Reloc, RelocKind, Section, Sym};
        let mut mc = empty_mc();
        mc.code = vec![0x90; 48];
        mc.fn_bases = vec![0, 16, 32];
        mc.relocs = vec![
            Reloc { kind: RelocKind::Rel32Call, section: Section::Code, site: 4, sym: Sym::Fn(2) },
        ];
        dead_code_eliminate(&mut mc);
        assert_eq!(mc.fn_bases, vec![0, 16], "dead middle function removed");
        assert_eq!(mc.code.len(), 32, "dead body gone");
        assert_eq!(mc.code, vec![0x90; 32]);
        assert_eq!(mc.relocs.len(), 1);
        assert_eq!(mc.relocs[0].site, 4, "site rebased into compacted blob");
        assert!(
            matches!(mc.relocs[0].sym, Sym::Fn(1)),
            "index remapped: {:?}",
            mc.relocs[0].sym
        );
    }

    /// Exports, `@init`, and address-taken functions are roots even with
    /// no in-image caller; an unreferenced helper is dropped.
    #[test]
    fn dce_keeps_roots_and_drops_dead_helper() {
        use crate::native::target::{Reloc, RelocKind, Section, Sym};
        let mut mc = empty_mc();
        mc.code = vec![0x90; 48];
        mc.fn_bases = vec![0, 16];
        mc.helper_itoa = Some(32);
        mc.exports = vec![("e".to_owned(), 0, 0)];
        mc.init_fn = Some(1);
        mc.data = vec![0; 8];
        mc.relocs = vec![
            Reloc { kind: RelocKind::Abs64, section: Section::Data, site: 0, sym: Sym::Fn(1) },
        ];
        dead_code_eliminate(&mut mc);
        assert_eq!(mc.fn_bases, vec![0, 16], "export + init + taken kept");
        assert_eq!(mc.exports, vec![("e".to_owned(), 0, 0)]);
        assert_eq!(mc.init_fn, Some(1));
        assert_eq!(mc.helper_itoa, None, "unreferenced helper dropped");
        assert_eq!(mc.code.len(), 32, "helper bytes gone");
        assert_eq!(mc.relocs.len(), 1, "data reloc kept");
    }

    /// An import slot referenced only from dead code leaves with it; a
    /// slot referenced from live code keeps its (rebased) body.
    #[test]
    fn dce_drops_dead_import_slot_and_body() {
        use crate::native::target::{Reloc, RelocKind, Section, Sym};
        let mut mc = empty_mc();
        mc.code = vec![0x90; 80];
        mc.fn_bases = vec![0, 16, 32];
        mc.import_names = vec![
            ("L".to_owned(), "Dead::F#0".to_owned()),
            ("L".to_owned(), "Live::G#0".to_owned()),
        ];
        mc.import_bases = vec![48, 64];
        mc.relocs = vec![
            Reloc { kind: RelocKind::Abs64, section: Section::Code, site: 4, sym: Sym::Import(1) },
            Reloc { kind: RelocKind::Abs64, section: Section::Code, site: 20, sym: Sym::Import(0) },
        ];
        dead_code_eliminate(&mut mc);
        // fn0 (root) references slot 1; fn1 (dead) referenced slot 0.
        // fn2 has no refs and no roots name it — dead too.
        assert_eq!(mc.fn_bases, vec![0], "only the entry survives");
        assert_eq!(mc.import_names.len(), 1, "dead slot dropped");
        assert_eq!(mc.import_names[0].1, "Live::G#0");
        assert_eq!(mc.import_bases, vec![16], "live body rebased after entry");
        assert_eq!(&mc.code[16..32], &[0x90; 16], "live body bytes kept");
        assert_eq!(mc.relocs.len(), 1);
        assert!(
            matches!(mc.relocs[0].sym, Sym::Import(0)),
            "slot remapped: {:?}",
            mc.relocs[0].sym
        );
    }

    /// Degenerate inputs are a no-op, never a mislink.
    #[test]
    fn dce_noop_on_empty_and_unsorted() {
        let mut mc = empty_mc();
        dead_code_eliminate(&mut mc);
        assert!(mc.code.is_empty() && mc.fn_bases.is_empty());
        let mut mc = empty_mc();
        mc.code = vec![0x90; 32];
        mc.fn_bases = vec![16, 0];
        dead_code_eliminate(&mut mc);
        assert_eq!(mc.fn_bases, vec![16, 0], "unsorted bases bail untouched");
        assert_eq!(mc.code.len(), 32);
    }

    /// Null import slots nothing references leave before resolution;
    /// referenced (and resolved) slots survive with remapped indices.
    #[test]
    fn drop_unreferenced_null_slots_keeps_live() {
        use crate::native::target::{Reloc, RelocKind, Section, Sym};
        let mut mc = empty_mc();
        mc.import_names = vec![
            ("L".to_owned(), "Dead::F#0".to_owned()),
            ("L".to_owned(), "Live::G#0".to_owned()),
            ("L".to_owned(), "Bodied::H#0".to_owned()),
        ];
        mc.import_bases = vec![usize::MAX, usize::MAX, 0];
        mc.import_abs = vec![u64::MAX, u64::MAX, u64::MAX];
        mc.relocs = vec![Reloc {
            kind: RelocKind::Abs64,
            section: Section::Code,
            site: 0,
            sym: Sym::Import(1),
        }];
        drop_unreferenced_null_slots(&mut mc);
        assert_eq!(mc.import_names.len(), 2, "dead null slot dropped");
        assert_eq!(mc.import_names[0].1, "Live::G#0");
        assert_eq!(mc.import_names[1].1, "Bodied::H#0");
        assert!(
            matches!(mc.relocs[0].sym, Sym::Import(0)),
            "survivor remapped: {:?}",
            mc.relocs[0].sym
        );
    }

    #[test]
    fn rubyc_shared_library_reports_runtime_dependency() {
        // A RubyC `.so` export is a trampoline whose `call body` is
        // image-relative; copying it crashed the consumer. The linker must
        // classify it as a runtime dependency, not a body donor.
        let dir = std::env::temp_dir().join(format!(
            "rubyc-shared-img-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut img = vec![0u8; 80];
        img[64..66].copy_from_slice(&crate::native::container::JUMP_OVER_MAGIC);
        img[66..74].copy_from_slice(&crate::native::container::MAGIC);
        let so = dir.join("Lib.so");
        std::fs::write(&so, &img).unwrap();
        let mut mc = empty_mc();
        mc.import_names = vec![("Lib".to_owned(), "answer".to_owned())];
        let err = resolve_imports(&mut mc, &[so], false).expect_err("runtime dependency, not a body");
        assert!(
            err.contains("runtime dependencies") && err.contains("dynamic consumer"),
            "the error must explain the model: {err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dep_so_import_without_a_loader_is_reported() {
        // A `.so`-hinted import used to be skipped unconditionally, so a
        // static executable linked with a null call. Only the JIT (which
        // dlopens afterwards) may leave it for load time.
        let mut mc = empty_mc();
        mc.import_names = vec![("Lib.so".to_owned(), "Lib::F#0".to_owned())];
        let err = resolve_imports(&mut mc, &[], false).expect_err("no loader in a static image");
        assert!(
            err.contains("need a loader")
                && err.contains("Lib::F#0")
                && err.contains("Lib.so"),
            "the error must name the slot and its image: {err}"
        );

        let mut mc = empty_mc();
        mc.import_names = vec![("Lib.so".to_owned(), "Lib::F#0".to_owned())];
        resolve_imports(&mut mc, &[], true).expect("the JIT resolves it with dlopen");
    }
}
