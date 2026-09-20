//! Base64 codec: RFC 4648 vectors, padding tolerance, error cases, and
//! exhaustive round-trips (this carries the 🐕🇨 container bytes).

use crate::utilities::base64::{decode, encode, Base64Error};

#[test]
fn rfc4648_vectors() {
    assert_eq!(encode(b""), "");
    assert_eq!(encode(b"f"), "Zg==");
    assert_eq!(encode(b"fo"), "Zm8=");
    assert_eq!(encode(b"foo"), "Zm9v");
    assert_eq!(encode(b"foob"), "Zm9vYg==");
    assert_eq!(encode(b"fooba"), "Zm9vYmE=");
    assert_eq!(encode(b"foobar"), "Zm9vYmFy");
}

#[test]
fn decodes_rfc4648_vectors() {
    assert_eq!(decode("Zm9vYmFy").unwrap(), b"foobar");
    assert_eq!(decode("Zg==").unwrap(), b"f");
    assert_eq!(decode("Zm8=").unwrap(), b"fo");
}

#[test]
fn decode_without_padding_works() {
    assert_eq!(decode("Zm9vYmFy").unwrap(), b"foobar");
    assert_eq!(decode("Zg").unwrap(), b"f");
    assert_eq!(decode("Zm8").unwrap(), b"fo");
}

#[test]
fn whitespace_is_ignored_on_decode() {
    let wrapped = "6wjwn5CV\n8J+HqAEB\r\nAAEB";
    assert!(decode(wrapped).is_ok());
}

#[test]
fn invalid_symbols_rejected() {
    assert_eq!(decode("a*c"), Err(Base64Error::InvalidByte(b'*')));
    assert!(matches!(
        decode("hello world!").unwrap_err(),
        Base64Error::InvalidByte(_)
    ));
}

#[test]
fn lone_sextet_is_invalid_length() {
    assert_eq!(decode("A"), Err(Base64Error::InvalidLength));
}

#[test]
fn binary_data_round_trips() {
    // Every byte value, all lengths 0..=300: catches bit-shift bugs.
    for len in 0..300usize {
        let data: Vec<u8> = (0..len).map(|i| (i * 7 + len) as u8).collect();
        assert_eq!(decode(&encode(&data)).unwrap(), data, "len={len}");
    }
}

#[test]
fn emoji_bytecode_round_trip() {
    // The 🐕🇨 container magic plus a stray emoji — real payload shapes.
    let bytes = [0xF0, 0x9F, 0x90, 0x95, 0xF0, 0x9F, 0x87, 0xA8];
    assert_eq!(decode(&encode(&bytes)).unwrap(), bytes);
}
