//! Comments: line, block (non-nesting), mixes, and error cases.

use crate::models::{ClassMember, Expr, Literal, Member, Stmt};
use crate::parser;

#[test]
fn issue_header_block_comment_parses_verbatim() {
    // The exact header from the language design notes.
    let src = r#"/* ===========================================================
** = These comments should  work as well as the // comments
/* =========================================================== */
namespace t; class c { int main() { printf("ok"); } }
"#;
    let unit = parser::parse(src).expect("header must parse");
    assert!(!unit.namespaces.is_empty());
}

#[test]
fn block_comment_anywhere_is_trivia() {
    let src =
        "namespace /* mid */ t; class /* x */ c { int /* y */ main() { /* z */ printf(\"go\"); } }";
    let unit = parser::parse(src).unwrap();
    assert_eq!(unit.namespaces[0].name, "t");
}

#[test]
fn block_comment_spans_lines_and_hides_code() {
    let src = "namespace t;\n/* class ghost {\n int nope = 1;\n} */\nclass real { int main() { printf(\"yes\"); } }";
    let unit = parser::parse(src).unwrap();
    let ns = &unit.namespaces[0];
    assert_eq!(ns.members.len(), 1);
}

#[test]
fn line_and_block_comments_interleave() {
    let src = "// a\n/* b */ // c\nnamespace t; // d\nclass c { int main() { printf(\"x\"); } }";
    parser::parse(src).unwrap();
}

#[test]
fn slash_star_inside_string_is_not_a_comment() {
    let src = "namespace t; class c { int main() { printf(\"a/*b*/c\"); } }";
    let unit = parser::parse(src).unwrap();
    let ns = &unit.namespaces[0];
    let Member::Class(class) = &ns.members[0] else {
        panic!("expected class")
    };
    let ClassMember::Method(m) = &class.members[0] else {
        panic!("expected method")
    };
    match &m.body.as_ref().unwrap().stmts[0] {
        Stmt::Expression(Expr::Call { args, .. }) => {
            assert_eq!(args[0], Expr::Literal(Literal::Str("a/*b*/c".into())));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn unterminated_block_comment_reports_e0005() {
    let errors = parser::parse("namespace t; /* never closed").expect_err("should fail");
    assert!(errors.iter().any(|e| e.code == "E0005"), "{errors:?}");
}

#[test]
fn division_after_block_comment_still_works() {
    let body = "int q = 8; /* half */ printf(\"{0}\", q / 2);";
    let src = format!("namespace t; class c {{ int main() {{ {body} }} }}");
    let unit = parser::parse(&src).unwrap();
    assert_eq!(unit.namespaces[0].members.len(), 1);
}
