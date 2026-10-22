//! printf-style interpolation: segment splitting, `{N[,align][:spec]}`
//! placeholders, `{{` escapes, argument validation and error reporting.

use crate::utilities::format::{
    Align, FormatError, FormatItem, interpolate, split_format, Segment,
};

fn arg(index: usize) -> Segment {
    Segment::Arg(FormatItem {
        index,
        align: None,
        format: None,
    })
}

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
            arg(0),
            Segment::Text("b".into()),
            arg(1),
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
fn alignment_and_format_are_parsed() {
    assert_eq!(
        split_format("{0,8:F2}").unwrap(),
        vec![Segment::Arg(FormatItem {
            index: 0,
            align: Some(Align::Right(8)),
            format: Some("F2".into()),
        })]
    );
    assert_eq!(
        split_format("{1,-5:X}").unwrap(),
        vec![Segment::Arg(FormatItem {
            index: 1,
            align: Some(Align::Left(5)),
            format: Some("X".into()),
        })]
    );
    // A colon inside the format text belongs to the format string.
    assert_eq!(
        split_format("{0:00:00}").unwrap(),
        vec![Segment::Arg(FormatItem {
            index: 0,
            align: None,
            format: Some("00:00".into()),
        })]
    );
}

#[test]
fn interpolate_joins_arguments_in_index_order() {
    let out = interpolate("{1} + {0} = ?", &["10".to_owned(), "20".to_owned()]).unwrap();
    assert_eq!(out, "20 + 10 = ?");
}

#[test]
fn interpolate_applies_alignment() {
    let args = ["7".to_owned()];
    assert_eq!(interpolate("[{0,5}]", &args).unwrap(), "[    7]");
    assert_eq!(interpolate("[{0,-5}]", &args).unwrap(), "[7    ]");
    assert_eq!(interpolate("[{0,1}]", &args).unwrap(), "[7]");
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
    assert_eq!(
        split_format("{0,}").unwrap_err(),
        FormatError::InvalidAlignment(String::new())
    );
    assert_eq!(
        split_format("{0,1x}").unwrap_err(),
        FormatError::InvalidAlignment("1x".into())
    );
    assert!(interpolate("{3}", &[const { String::new() }; 2]).is_err());
    // Pre-rendered arguments cannot apply a numeric format.
    assert!(interpolate("{0:D3}", &["7".to_owned()]).is_err());
}

#[test]
fn no_placeholders_needs_no_args() {
    assert!(interpolate("just text", &[]).is_ok());
}
