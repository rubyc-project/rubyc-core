//! `did you mean …?` suggestions: edit distance, closest-candidate
//! selection, case handling, and the rendered help text.

use crate::utilities::suggest::{did_you_mean, edit_distance, suggest};

#[test]
fn distance_basics() {
    assert_eq!(edit_distance("kitten", "sitting"), 3);
    assert_eq!(edit_distance("abc", "abc"), 0);
    assert_eq!(edit_distance("", "ab"), 2);
}

#[test]
fn suggest_prefers_closest() {
    let c = ["printf", "print", "put_str"];
    assert_eq!(suggest("printfx", &c, 2), Some("printf"));
    assert_eq!(suggest("prints", &c, 2), Some("print"));
}

#[test]
fn case_insensitive_exact_wins() {
    let c = ["Value"];
    assert_eq!(suggest("value", &c, 2), Some("Value"));
}

#[test]
fn nothing_close_returns_none() {
    let c = ["alpha", "beta"];
    assert_eq!(suggest("zzzzzzzz", &c, 2), None);
}

#[test]
fn did_you_mean_formats() {
    assert_eq!(
        did_you_mean("valu", &["value"]),
        Some("did you mean 'value'?".into())
    );
    assert_eq!(did_you_mean("zzz", &["value"]), None);
}
