#[test]
fn dbg_direct_elem_printf_ir() {
    let src = r#"namespace t { class c { int Main(string[] args) { printf("{0}", args[1]); return 0; } } }"#;
    let unit = crate::parser::parse(src).unwrap();
    let p = crate::native::ir::lower(&unit).unwrap();
    for f in &p.functions {
        if f.name.contains("Main") {
            for b in &f.blocks {
                for i in &b.instrs {
                    eprintln!("  {i:?}");
                }
            }
        }
    }
}
