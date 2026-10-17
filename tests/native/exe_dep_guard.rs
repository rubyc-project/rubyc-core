//! Executable builds reject dependency-image imports loudly: a standalone
//! executable has no load-time dep loader yet (shared libraries keep null
//! slots by design). Uses stub backend/writer — no target plugin needed.
use crate::native::ir::Program;
use crate::native::target::{CodeGenBackend, ImageWriter, MachineCode, TargetError};

struct StubBackend;
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
            import_names: vec![("DepLib.so".to_owned(), "du.U::Helper#1".to_owned())],
            import_bases: vec![],
        import_abs: vec![],
            exports: vec![],
            export_takes_receiver: vec![],
            init_fn: None,
            relocs: vec![],
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
        &StubBackend,
        &StubWriter,
        &[],
        true,
        &[],
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("du.U::Helper#1"), "names the import: {msg}");
    assert!(msg.contains("shared library"), "points at the workaround: {msg}");
}
