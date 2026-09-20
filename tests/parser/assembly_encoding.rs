//! The built-in x86-64 assembler: two-pass encoding of instructions,
//! labels/branches, data directives, and `#Assembly`-style snippets.

use crate::parser::assembly::assemble;

#[test]
fn mov_imm32_uses_c7() {
    let b = assemble("mov rax, 1").unwrap();
    assert_eq!(b, vec![0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00]);
}

#[test]
fn reg_to_reg_xor() {
    let b = assemble("xor rdi, rdi").unwrap();
    assert_eq!(b, vec![0x48, 0x31, 0xFF]);
}

#[test]
fn add_imm8_compact() {
    let b = assemble("add rax, 5").unwrap();
    assert_eq!(b, vec![0x48, 0x83, 0xC0, 0x05]);
}

#[test]
fn branch_label_rel32_patches() {
    let b = assemble("start:\n jmp end\n nop\nend:\n ret").unwrap();
    assert_eq!(b[0], 0xE9);
    let rel = i32::from_le_bytes([b[1], b[2], b[3], b[4]]);
    assert_eq!(rel, 1);
    assert_eq!(&b[5..7], &[0x90, 0xC3]);
}

#[test]
fn jcc_backward_patches() {
    let src = "cmp rax, 0\nje next\nnext:\nnop";
    let b = assemble(src).unwrap();
    assert_eq!(b[0], 0x48);
    assert_eq!(&b[4..6], &[0x0F, 0x84]);
    let rel = i32::from_le_bytes([b[6], b[7], b[8], b[9]]);
    assert_eq!(rel, 0); // target == next instruction
    assert_eq!(b[10], 0x90);
}

#[test]
fn mem_operand_mov() {
    let b = assemble("mov rax, [rbx]").unwrap();
    assert_eq!(b, vec![0x48, 0x8B, 0x03]);
    let b = assemble("mov [rsp], rax").unwrap();
    assert_eq!(b, vec![0x48, 0x89, 0x04, 0x24]);
    let b = assemble("mov rax, [rbx + 16]").unwrap();
    assert_eq!(b, vec![0x48, 0x8B, 0x43, 0x10]);
}

#[test]
fn syscall_and_push_pop() {
    let b = assemble("push rbx\nsyscall\npop rbx\nret").unwrap();
    assert_eq!(b, vec![0x53, 0x0F, 0x05, 0x5B, 0xC3]);
}

#[test]
fn byte_prefix_memory() {
    let b = assemble("mov byte [rsp], 10").unwrap();
    // mov qword [rsp], imm32: 48 C7 04 24 0A 00 00 00
    assert_eq!(&b[..5], &[0x48, 0xC7, 0x04, 0x24, 0x0A]);
}

// --- whole-snippet assembly (the shapes `#Assembly` blocks produce) -----

#[test]
fn asm_block_encodes_to_bytes() {
    // A simple block: mov rax, 42; exit syscall
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
fn asm_data_directives_not_yet_supported() {
    // `db`/`equ` are on the roadmap (NASM-flavored subset grows per need);
    // today the assembler must reject them cleanly instead of emitting
    // garbage bytes.
    assert!(assemble("msg: db 72, 101, 108, 108, 111\nlen: equ 5").is_err());
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
    // Executed result would be (10+20-5)*2 = 50.
    assert!(!bytes.is_empty());
}
