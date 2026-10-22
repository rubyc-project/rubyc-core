//! Executable builds reject dependency-image imports loudly: a standalone
//! executable has no load-time dep loader. Uses stub backend/writer — no
//! target plugin needed.
use crate::native::ir::Program;
use crate::native::target::{CodeGenBackend, ImageWriter, MachineCode, TargetError};

struct StubBackend(&'static str);
impl CodeGenBackend for StubBackend {
    fn lower(&self, _program: &Program) -> Result<MachineCode, TargetError> {
        Ok(MachineCode {
            code: vec![0xc3],
            data: vec![],
            fn_bases: vec![0],
            vtable_offsets: vec![],
            vtable_names: vec![],
            static_offsets: vec![],
        static_names: vec![],
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
            import_names: vec![(self.0.to_owned(), "du.U::Helper#1".to_owned())],
            import_bases: vec![usize::MAX],
        import_abs: vec![],
            exports: vec![],
            export_takes_receiver: vec![],
            init_fn: None,
            relocs: vec![crate::native::target::Reloc {
                kind: crate::native::target::RelocKind::Abs64,
                section: crate::native::target::Section::Code,
                site: 0,
                sym: crate::native::target::Sym::Import(0),
            }],
            heap_size: 0,
            strip_symbols: false,
        })
    }
}

struct StubWriter;
impl ImageWriter for StubWriter {
    fn write_image(&self, _mc: &MachineCode, _entry_preamble: bool) -> Result<Vec<u8>, TargetError> {
        Ok(vec![])
    }
}

#[test]
fn exe_build_with_dep_import_fails_loud() {
    let unit = crate::parser::parse("class C { int Main() { return 0; } }").unwrap();
    let err = crate::native::compile_to_bytes_with_libs_preamble(
        &unit,
        &StubBackend("DepLib.so"),
        &StubWriter,
        &[],
        true,
        &[],
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("du.U::Helper#1"), "names the import: {msg}");
    assert!(
        msg.contains("from sources"),
        "points at a way to make the import resolvable: {msg}"
    );
}

struct LibWriter;
impl ImageWriter for LibWriter {
    fn write_image(&self, _mc: &MachineCode, _entry_preamble: bool) -> Result<Vec<u8>, TargetError> {
        Ok(vec![])
    }
    fn write_shared_library(&self, _mc: &MachineCode) -> Result<Vec<u8>, TargetError> {
        Ok(vec![])
    }
}

#[test]
fn shared_library_with_unlinked_import_fails_loud() {
    // A library declaring `[LibraryImport]` stubs used to warn and emit
    // with null import slots whenever no `--lib` was given ("like
    // `ld -shared`"). Every call through such a slot jumped through
    // address 0. An unresolvable import is a build failure for a
    // library exactly as it is for an executable.
    let unit = crate::parser::parse("class C { public int M() { return 0; } }").unwrap();
    let err = crate::native::compile_shared_library(
        &unit,
        &StubBackend("DepLib"),
        &LibWriter,
        &[],
        &[],
        None,
        Default::default(),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("LibraryImport") && msg.contains("--lib"),
        "names the stubs and the missing input: {msg}"
    );
}
