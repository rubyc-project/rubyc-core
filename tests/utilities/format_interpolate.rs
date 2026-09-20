//! printf-style interpolation: segment splitting, `{N}` placeholders,
//! `{{` escapes, argument validation and error reporting.

use crate::utilities::format::{interpolate, split_format, FormatError, Segment};

#[test]
fn plain_text_is_one_segment() {
    assert_eq!(
        split_format("hello").unwrap(),
        vec![Segment::Text("hello".into())]
    );
}

#[test]
fn placeholders_split_segments() {
    assert_eq!(
        split_format("a{0}b{1}").unwrap(),
        vec![
            Segment::Text("a".into()),
            Segment::Arg(0),
            Segment::Text("b".into()),
            Segment::Arg(1),
        ]
    );
}

#[test]
fn escaped_braces_become_literals() {
    assert_eq!(
        split_format("{{x}}").unwrap(),
        vec![Segment::Text("{x}".into())]
    );
}

#[test]
fn interpolate_joins_arguments_in_index_order() {
    let out = interpolate("{1} + {0} = ?", &["10".to_owned(), "20".to_owned()]).unwrap();
    assert_eq!(out, "20 + 10 = ?");
}

#[test]
fn errors_are_reported() {
    assert_eq!(
        split_format("oops {").unwrap_err(),
        FormatError::UnclosedBrace
    );
    assert_eq!(
        split_format("{x}").unwrap_err(),
        FormatError::InvalidPlaceholder("x".into())
    );
    assert!(interpolate("{3}", &[const { String::new() }; 2]).is_err());
}

#[test]
fn no_placeholders_needs_no_args() {
    assert!(interpolate("just text", &[]).is_ok());
}
