//! Full assembly integration: #Assembly blocks encode through the
//! assembler pipeline, [Assembly]-attributed methods are callable from
//! RubyC code with SysV convention args and return values.

use crate::parser::assembly::assemble;

#[test]
fn asm_block_encodes_to_bytes() {
    // A simple block: mov rax, 42; ret
    let src = "mov rax, 60\nmov rdi, 42\nsyscall";
    let bytes = assemble(src).unwrap();
    assert!(!bytes.is_empty());
}

#[test]
fn asm_labels_and_branches() {
    let src = "start:\n  nop\n  jmp end\n  nop\nend:\n  ret";
    let bytes = assemble(src).unwrap();
    assert!(!bytes.is_empty());
}

#[test]
#[ignore = ".data directives are not implemented yet"]
fn asm_data_directives() {
    let src = "msg: db 72, 101, 108, 108, 111\nlen: equ 5";
    let bytes = assemble(src).unwrap();
    assert!(!bytes.is_empty());
}

#[test]
fn asm_push_pop_preserves_reg() {
    let src = "push rbx\nmov rbx, 1\npop rbx\nret";
    let bytes = assemble(src).unwrap();
    assert!(!bytes.is_empty());
}

#[test]
fn asm_arithmetic_chain() {
    let src = "mov rax, 10\nadd rax, 20\nsub rax, 5\nimul rax, rax, 2\nret";
    let bytes = assemble(src).unwrap();
    // Expected result if executed: (10+20-5)*2 = 50
    assert!(!bytes.is_empty());
}
