//! Native-import marshaling (P/Invoke C ABI).
//!
//! Managed objects never cross a native boundary raw: `string` args
//! transcode to NUL-terminated UTF-8 temps (the object leads with its
//! LENGTH — file open read `"\021"` and died ENOENT on it), and
//! SafeHandle-family returns wrap to hollow live objects (downstream
//! vtable/field traffic on a small-int fd faulted at `[fd+0x58]`).
//! Managed-body cross-image calls keep their objects (same IR ABI).

use crate::native::ir::Program;

fn lower(src: &str) -> Result<Program, String> {
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).map_err(|e| format!("{e:?}"))
}

fn count_calls(p: &Program) -> usize {
    p.functions
        .iter()
        .filter(|f| !f.name.contains("@start"))
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| {
            matches!(
                i,
                crate::native::ir::Instr::CallImport { .. }
                    | crate::native::ir::Instr::CallVirt { .. }
                    | crate::native::ir::Instr::CallStatic { .. }
                    | crate::native::ir::Instr::CallFn { .. }
            )
        })
        .count()
}

fn count_imports(p: &Program) -> usize {
    p.functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(i, crate::native::ir::Instr::CallImport { .. }))
        .count()
}

fn count_alloc_dyn(p: &Program) -> usize {
    p.functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(i, crate::native::ir::Instr::AllocDyn { .. }))
        .count()
}

fn count_vtable_addr(p: &Program) -> usize {
    p.functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(i, crate::native::ir::Instr::VTableAddr { .. }))
        .count()
}

#[test]
fn import_string_arg_transcodes_to_temp() {
    // A `string` argument to a native import must allocate a UTF-8
    // temp (AllocDyn) — the raw object pointer leads with the length.
    let p = lower(
        r#"namespace t;
static class Interop {
    [LibraryImport("native", EntryPoint = "my_open")]
    internal static extern int MyOpen(string path, int flags, int mode);
}
class C {
    int Main() { return Interop.MyOpen("/tmp/x", 1, 2); }
}"#,
    )
    .unwrap();
    assert_eq!(count_imports(&p), 1, "native stub must stay an import");
    assert_eq!(count_calls(&p), 1, "only the import call, no managed body");
    assert!(
        count_alloc_dyn(&p) >= 1,
        "string arg must allocate a UTF-8 temp"
    );
}

#[test]
fn import_safehandle_return_wraps_to_object() {
    // A native import returning SafeHandle-family must build a live
    // object (VTableAddr) around the raw fd — not leak the int.
    let p = lower(
        r#"namespace t;
class SafeFileHandle {
    int handle;
}
static class Interop {
    [LibraryImport("native", EntryPoint = "my_open2")]
    internal static extern SafeFileHandle MyOpen2(string path, int flags);
}
class C {
    int Main() {
        SafeFileHandle h = Interop.MyOpen2("/tmp/x", 1);
        return 0;
    }
}"#,
    )
    .unwrap();
    assert_eq!(count_imports(&p), 1, "native stub must stay an import");
    assert!(
        count_vtable_addr(&p) >= 1,
        "SafeHandle return must wrap to a live object"
    );
    assert!(
        count_alloc_dyn(&p) >= 1,
        "string arg must still allocate a UTF-8 temp"
    );
}
