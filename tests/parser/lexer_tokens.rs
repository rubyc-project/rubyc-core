//! Lexer unit tests: token streams, keyword recognition, trivia skipping,
//! and span fidelity.

use crate::models::{Keyword, TokenKind};
use crate::parser::lexer::Lexer;

fn kinds(src: &str) -> Vec<TokenKind> {
    Lexer::new(src)
        .tokenize()
        .into_iter()
        .map(|t| t.kind)
        .collect()
}

#[test]
fn interp_escaped_braces_do_not_open_holes() {
    // `$"namespace {n} {{"`: the trailing `{{` is an escaped literal
    // brace, not a hole. The lexer used to open a phantom hole, treat
    // the closing `"` as nested, and swallow the rest of the file
    // (System.Private.Xml Compilation: "expected ')' ... found end
    // of file"). The raw body must survive intact and the tokens
    // after the string must lex normally.
    assert_eq!(
        kinds(r#"w.WriteLine($"namespace {n} {{");"#),
        vec![
            TokenKind::Identifier("w".into()),
            TokenKind::Dot,
            TokenKind::Identifier("WriteLine".into()),
            TokenKind::LeftParen,
            TokenKind::InterpolatedString {
                raw: "namespace {n} {{".into(),
                verbatim: false,
            },
            TokenKind::RightParen,
            TokenKind::Semicolon,
            TokenKind::Eof,
        ]
    );
}

#[test]
fn hello_world_token_stream() {
    assert_eq!(
        kinds(r#"printf("Hello World!");"#),
        vec![
            TokenKind::Identifier("printf".into()),
            TokenKind::LeftParen,
            TokenKind::StringLiteral("Hello World!".into()),
            TokenKind::RightParen,
            TokenKind::Semicolon,
            TokenKind::Eof,
        ]
    );
}

#[test]
fn keywords_are_recognized() {
    assert_eq!(
        kinds("namespace testnc; class testc"),
        vec![
            TokenKind::Keyword(Keyword::Namespace),
            TokenKind::Identifier("testnc".into()),
            TokenKind::Semicolon,
            TokenKind::Keyword(Keyword::Class),
            TokenKind::Identifier("testc".into()),
            TokenKind::Eof,
        ]
    );
}

#[test]
fn trivia_is_skipped_including_comments() {
    let src = "int main() // greet\n{ }";
    assert_eq!(kinds(src).len(), 7); // int main ( ) { } + Eof
}

#[test]
fn spans_cover_the_lexeme() {
    let tokens = Lexer::new("namespace").tokenize();
    assert_eq!(tokens[0].span.slice("namespace"), "namespace");
}

#[test]
fn float_literals_are_recognized() {
    assert_eq!(
        kinds("1.5 2.25f 1e10 0.0 3.5d 1.25E-3"),
        vec![
            TokenKind::FloatLiteral(1.5f64.to_bits()),
            TokenKind::FloatLiteral(2.25f64.to_bits()),
            TokenKind::FloatLiteral(1e10f64.to_bits()),
            TokenKind::FloatLiteral(0.0f64.to_bits()),
            TokenKind::FloatLiteral(3.5f64.to_bits()),
            TokenKind::FloatLiteral(1.25e-3f64.to_bits()),
            TokenKind::Eof,
        ]
    );
}

#[test]
fn nbsp_and_bom_are_trivia() {
    // Real dotnet sources contain NBSP (U+00A0, e.g. Task.cs) and
    // files may start with a UTF-8 BOM — both are trivia per the C#
    // whitespace rules, never stream-truncating unknown bytes.
    assert_eq!(
        kinds("\u{a0}int\u{a0}x;"),
        vec![
            TokenKind::Identifier("int".into()),
            TokenKind::Identifier("x".into()),
            TokenKind::Semicolon,
            TokenKind::Eof,
        ]
    );
    assert_eq!(
        kinds("\u{feff}int x;"),
        vec![
            TokenKind::Identifier("int".into()),
            TokenKind::Identifier("x".into()),
            TokenKind::Semicolon,
            TokenKind::Eof,
        ]
    );
}

#[test]
fn leading_dot_float_literals_are_recognized() {
    // `.90` is a C# double; `..` stays the range operator.
    assert_eq!(
        kinds(".90 .70e2"),
        vec![
            TokenKind::FloatLiteral(0.90f64.to_bits()),
            TokenKind::FloatLiteral(70.0f64.to_bits()),
            TokenKind::Eof,
        ]
    );
    assert_eq!(
        kinds("a[..]"),
        vec![
            TokenKind::Identifier("a".to_owned()),
            TokenKind::LeftBracket,
            TokenKind::DotDot,
            TokenKind::RightBracket,
            TokenKind::Eof,
        ]
    );
}

#[test]
fn integer_literals_are_not_mistaken_for_floats() {
    assert_eq!(
        kinds("7 1_000 0"),
        vec![
            TokenKind::IntLiteral(7),
            TokenKind::IntLiteral(1000),
            TokenKind::IntLiteral(0),
            TokenKind::Eof,
        ]
    );
}

#[test]
fn verbatim_identifiers_drop_at_and_skip_keyword_mapping() {
    // `@this` / `@class` are identifiers named `this` / `class`
    // (C# verbatim identifiers, e.g. SZArrayHelper's `T[] @this`),
    // never keywords and never `@`-prefixed.
    assert_eq!(
        kinds("@this @class @new"),
        vec![
            TokenKind::Identifier("this".into()),
            TokenKind::Identifier("class".into()),
            TokenKind::Identifier("new".into()),
            TokenKind::Eof,
        ]
    );
}

#[test]
fn hex_binary_and_suffix_integer_literals_parse_to_their_value() {
    // Hex, binary, and integral-suffixed literals (C# 11 forms) all parse to
    // the correct unsigned magnitude; a trailing suffix is not part of the value.
    assert_eq!(
        kinds("0xff 0xFF 0x1F 0b1010 0b1010L 255u 100 0"),
        vec![
            TokenKind::IntLiteral(0xff),
            TokenKind::IntLiteral(0xff),
            TokenKind::IntLiteral(0x1f),
            TokenKind::IntLiteral(0b1010),
            TokenKind::IntLiteral(0b1010),
            TokenKind::IntLiteral(255),
            TokenKind::IntLiteral(100),
            TokenKind::IntLiteral(0),
            TokenKind::Eof,
        ]
    );
}
