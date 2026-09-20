use crate::TargetBackend;

struct FakeBackend;

impl TargetBackend for FakeBackend {
    fn name(&self) -> &'static str {
        "fake"
    }
    fn triple(&self) -> &'static str {
        "fake-unknown-none"
    }
    fn lower(&self, _program: &[u8]) -> Result<Vec<u8>, String> {
        Ok(vec![0u8; 4])
    }
    fn write_image(&self, _mc: &[u8], _entry_preamble: bool) -> Result<Vec<u8>, String> {
        Ok(vec![0u8; 8])
    }
}

#[test]
fn trait_methods_are_callable() {
    let b: Box<dyn TargetBackend> = Box::new(FakeBackend);
    assert_eq!(b.name(), "fake");
    assert_eq!(b.triple(), "fake-unknown-none");
    assert!(b.lower(&[]).is_ok());
    assert!(b.write_image(&[], false).is_ok());
}

#[test]
fn lower_returns_error_string_on_failure() {
    struct ErrBackend;
    impl TargetBackend for ErrBackend {
        fn name(&self) -> &'static str { "err" }
        fn triple(&self) -> &'static str { "err-unknown-none" }
        fn lower(&self, _program: &[u8]) -> Result<Vec<u8>, String> {
            Err("nope".into())
        }
        fn write_image(&self, _mc: &[u8], _entry_preamble: bool) -> Result<Vec<u8>, String> {
            Err("nope".into())
        }
    }
    let b: Box<dyn TargetBackend> = Box::new(ErrBackend);
    assert_eq!(b.lower(&[]).unwrap_err(), "nope");
    assert_eq!(b.write_image(&[], false).unwrap_err(), "nope");
}
