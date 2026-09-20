//! `--no-magic` round-trip tests: encoding without the RubyC preamble must
//! still decode (the decode path tolerates the absent header), and the raw
//! payload must carry no `jump-over-magic` / dog / kind / version header.

use crate::bytecode::traits::ToBytecode;
use crate::bytecode::{decode, encode_no_magic};
use crate::models::compare::StructuralEq;
use crate::native::container;
use crate::parser;
use crate::testdata::SAMPLES;

#[test]
fn every_sample_round_trips_without_preamble() {
    for sample in SAMPLES {
        let unit = parser::parse(sample.source)
            .unwrap_or_else(|e| panic!("sample {} failed to parse: {e:?}", sample.name));
        let bytes = encode_no_magic(&unit);
        let decoded = decode(&bytes)
            .unwrap_or_else(|e| panic!("sample {} no-magic decode failed: {e}", sample.name));
        assert!(
            decoded.structural_eq(&unit),
            "sample {} changed across no-magic round-trip",
            sample.name
        );
    }
}

#[test]
fn no_magic_payload_has_no_preamble() {
    let unit = parser::parse(SAMPLES[0].source).unwrap();
    let bytes = unit.to_bytecode();
    let raw = encode_no_magic(&unit);

    // The standard form starts with the jump-over-magic preamble.
    assert_eq!(&bytes[..container::PREAMBLE_LEN], &container::build(container::Kind::Bytecode, 1)[..container::PREAMBLE_LEN]);
    // The no-magic form must not contain the jump-over-magic bytes anywhere.
    assert!(
        !raw.windows(container::PREAMBLE_LEN)
            .any(|w| w == &container::JUMP_OVER_MAGIC[..]),
        "no-magic payload must not embed the jump-over-magic header"
    );
    // The no-magic payload is the raw unit encoding (shorter than the
    // preamble-prefixed form by exactly the full header: PREAMBLE_LEN + kind
    // byte + version bytes = 10 + 1 + 2 = 13).
    assert_eq!(bytes.len() - raw.len(), container::PREAMBLE_LEN + 3);
}

#[test]
fn no_magic_decode_is_tolerant_of_missing_header() {
    let unit = parser::parse(SAMPLES[0].source).unwrap();
    let raw = encode_no_magic(&unit);
    // Decoding the raw payload (no header) succeeds and round-trips.
    let decoded = decode(&raw).expect("raw payload decodes");
    assert!(decoded.structural_eq(&unit));
}

#[test]
fn empty_input_rejected() {
    assert_eq!(decode(&[]), Err(crate::models::DecodeError::BadMagic));
}
