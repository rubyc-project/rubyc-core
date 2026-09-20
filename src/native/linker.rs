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

/// All defined function symbols of one symbol table with their file ranges.
///
/// For ET_DYN objects (`ld -shared` output) symbol values are offsets from
/// the load base, so the file offset is `sh_offset + st_value - sh_addr`
/// within the defining section.
fn read_symbols(
    img: &[u8],
    sections: &[Section],
    symtab: &Section,
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
        if st_name == 0 || st_size == 0 || st_shndx == SHN_UNDEF || st_info & 0xf != STT_FUNC {
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

/// Every exported function symbol across all symbol tables of one image.
fn exported_functions(img: &[u8]) -> Result<Vec<(String, std::ops::Range<usize>)>, String> {
    let sections = parse_sections(img)?;
    let mut syms = Vec::new();
    for section in &sections {
        if section.sh_type == SHT_DYNSYM || section.sh_type == SHT_SYMTAB {
            syms.extend(read_symbols(img, &sections, section));
        }
    }
    Ok(syms)
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
///   the same symbol) resolve first-wins.
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
                let base_missing = mc.import_bases.get(*i).copied() == Some(usize::MAX);
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
        mc.import_names.extend(member.import_names.iter().cloned());
        mc.import_bases.extend(member.import_bases.iter().map(|b| {
            if *b == usize::MAX {
                usize::MAX
            } else {
                b + code_base
            }
        }));
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
        if let Sym::Import(i) = r.sym
            && let Some(obj) = objects.get(&mc.import_names[i].1)
        {
            r.sym = match obj {
                StaticObj::Vtable(off) => Sym::VtableOff(*off),
                StaticObj::Static(idx) => Sym::Static(*idx),
            };
            consumed[i] = true;
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
/// Copy-link `[LibraryImport]` bodies from `--lib` ELF files into
/// the code blob and filling `import_bases`. Fails with a readable message
/// listing any missing symbols.
pub fn resolve_imports(
    mc: &mut crate::native::target::MachineCode,
    libs: &[PathBuf],
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

    type SymbolTable = Vec<(String, std::ops::Range<usize>)>;
    let mut images: Vec<(PathBuf, Vec<u8>, SymbolTable)> = Vec::new();
    for lib in libs {
        let bytes = std::fs::read(lib)
            .map_err(|e| format!("cannot read library {}: {e}", lib.display()))?;
        let syms = exported_functions(&bytes).map_err(|e| format!("{}: {e}", lib.display()))?;
        images.push((lib.clone(), bytes, syms));
    }

    for (i, (lib_hint, entry)) in mc.import_names.iter().enumerate() {
        // Slots the static link step already resolved keep their bodies.
        if mc.import_bases.get(i).copied() != Some(usize::MAX) {
            continue;
        }
        // Dependency-image imports (`lib_hint` is a `.so` file name, e.g.
        // `System.Private.CoreLib.so`): the callee's body stays in its
        // home image and is resolved by the loader at load time
        // (dlopen + dlsym). Extracting (copy-linking) the body here
        // would break its absolute references to the home image, so the
        // slot stays null — like shared-library builds with no `--lib`.
        if lib_hint.ends_with(".so") {
            continue;
        }
        // Prefer an image whose file name contains the library hint
        // ("native_add" matches libnative_add.so); fall back to any image
        // exporting the entry point.
        let mut found = None;
        for (path, bytes, syms) in &images {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.contains(lib_hint.as_str())
                && let Some((_, range)) = syms.iter().find(|(name, _)| name == entry)
            {
                found = Some((range.clone(), bytes));
                break;
            }
        }
        let Some((range, bytes)) = found.or_else(|| {
            images.iter().find_map(|(_, bytes, syms)| {
                syms.iter()
                    .find(|(name, _)| name == entry)
                    .map(|(_, range)| (range.clone(), bytes))
            })
        }) else {
            let provided: Vec<String> = libs.iter().map(|p| p.display().to_string()).collect();
            return Err(format!(
                "import '{entry}' not found in provided libraries {provided:?}"
            ));
        };
        // Align each extracted body to 16 bytes for cleanliness.
        while !mc.code.len().is_multiple_of(16) {
            mc.code.push(0xCC);
        }
        mc.import_bases[i] = mc.code.len();
        mc.code.extend_from_slice(&bytes[range]);
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

    #[test]
    fn dep_so_imports_stay_null_for_loader() {
        let mut mc = empty_mc();
        mc.import_names = vec![
            ("DepLib.so".to_owned(), "Ns.Class::Method#0".to_owned()),
        ];
        resolve_imports(&mut mc, &[]).expect("dep imports never fail link");
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
}
