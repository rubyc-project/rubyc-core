//! Every context that switches class must name its class by PATH
//! (D-066). The access rule compares the viewer with the declaring
//! class, so a context that named a class by its bare name could not
//! be matched against a qualified declaring path — and a bare name is
//! ambiguous for a nested type (two outers may declare the same
//! nested name). These tests pin the contexts that switch class:
//! method bodies, property accessors, field initializers and local
//! functions.

use crate::native::ir::Program;

fn lower(src: &str) -> Result<Program, String> {
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).map_err(|e| e.to_string())
}

#[test]
fn local_function_in_nested_type_reads_private_member() {
    // A local function's body belongs to the class that declares it.
    // That class is named by its QUALIFIED path: the bare name of a
    // nested type is ambiguous (two outers below declare the same
    // nested name), so a bare viewer never matched the qualified
    // declaring path and the private read was denied.
    let src = r#"namespace t;
class Utf8 {
    class Handler {
        private int _provider;
        public int Read() {
            int Helper() { return _provider; }
            return Helper();
        }
    }
}
class Utf16 {
    class Handler {
        private int _provider;
    }
}
class C { int Main() { return 0; } }
"#;
    let r = lower(src);
    assert!(r.is_ok(), "a local function reads its own type's private member: {r:?}");
}

#[test]
fn property_accessor_reads_declaring_class_private_member() {
    // The getter's body belongs to the class that DECLARED the
    // property, which may be a base of the receiver's class. That
    // class is named by its qualified path.
    let src = r#"namespace t;
class Outer {
    class Base {
        private int _value;
        public int Value => _value;
    }
    class Derived : Base { }
}
class C { int Main() { return 0; } }
"#;
    let r = lower(src);
    assert!(
        r.is_ok(),
        "a property getter reads its declaring class's private member: {r:?}"
    );
}

#[test]
fn field_initializer_reads_own_private_member() {
    // A field initializer runs in the declaring class's context, which
    // is named by its qualified path (a nested type's bare name is
    // ambiguous).
    let src = r#"namespace t;
class Outer {
    class Inner {
        private int _seed = 7;
        private int _value = _seed;
        public int Value => _value;
    }
}
class C { int Main() { return 0; } }
"#;
    let r = lower(src);
    assert!(
        r.is_ok(),
        "a field initializer reads its own class's private member: {r:?}"
    );
}

#[test]
fn same_named_nested_types_keep_their_own_private_members() {
    // Two outers each nest a type with the same simple name. A read
    // of a private member inside each nested type must resolve to
    // THAT type's member, not to the first same-named entry.
    let src = r#"namespace t;
class A {
    class Shared {
        private int _a;
        public int Read() { return _a; }
    }
}
class B {
    class Shared {
        private int _b;
        public int Read() { return _b; }
    }
}
class C { int Main() { return 0; } }
"#;
    let r = lower(src);
    assert!(
        r.is_ok(),
        "same-named nested types keep their own private members: {r:?}"
    );
}
