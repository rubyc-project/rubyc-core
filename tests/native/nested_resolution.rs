//! Nested-type member resolution (C# enclosing-type lookup).
//!
//! - Members of a nested `partial` class resolve through the
//!   enclosing type, including `#if`-gated members
//!   (`AsyncProfiler.ContinuationWrapper.InitInfo` in CoreLib).
//! - `ref struct` nesting, block-form namespaces, and `ref`
//!   calls/params preserve resolution.
//! - Same-named nesteds in different outers: the ENCLOSING outer's
//!   nested wins over unit first-match, even when the other nested
//!   lacks the member entirely.

use crate::native::semantic::TypeTable;

fn table_of(src: &str) -> TypeTable {
    let pre = crate::preprocessor::preprocess(src, &[], "probe.rc").expect("preprocess");
    let unit = crate::parser::parse(&pre.source).unwrap();
    crate::native::semantic::analyze(&unit).expect("analyze")
}

fn methods_in(table: &TypeTable, path: &str) -> Vec<String> {
    let mut out = Vec::new();
    for c in &table.classes {
        if c.key.path == path {
            out.extend(c.methods.iter().map(|m| m.name.clone()));
        }
    }
    out
}

#[test]
fn nested_conditional_member_resolves() {
    let table = table_of(
        r#"namespace t;
class Outer {
    internal static void Init() { Nested.Go(); }
    internal static partial class Nested {
#if !SOMETHING_UNDEFINED
        public static void Go() { }
#endif
    }
    int main() { return 0; }
}"#,
    );
    let methods = methods_in(&table, "t.Outer.Nested");
    assert!(methods.contains(&"Go".to_string()), "conditional member kept: {methods:?}");
}

#[test]
fn nested_refstruct_refcall_in_block_ns_resolves() {
    let table = table_of(
        r#"namespace Ns
{
    static class Outer {
        internal ref struct Info
        {
            public int X;
            public ref int Table;
        }
        internal static void InitInfo(ref Info info)
        {
            info.X = 0;
            Nested.InitInfo(ref info);
        }
        internal static partial class Nested {
#if !SOMETHING_UNDEFINED
            public static void InitInfo(ref Info info)
            {
                info.X = 1;
            }
#endif
            public const byte COUNT = 32;
        }
        int main() { return 0; }
    }
}"#,
    );
    let methods = methods_in(&table, "Ns.Outer.Nested");
    assert!(methods.contains(&"InitInfo".to_string()), "ref InitInfo kept: {methods:?}");
}
#[test]
fn enclosing_nested_wins_over_samenamed_sibling() {
    let table = table_of(
        r#"namespace Ns
{
    static class OuterA {
        internal static void Init() { Nested.Go(1); }
        internal static partial class Nested {
            public static void Go(int x) { }
        }
        int main() { return 0; }
    }
    static class OuterB {
        internal static partial class Nested {
            public static void Other(string s) { }
        }
    }
}"#,
    );
    // Both nesteds coexist under their own paths with their own
    // members (no cross-contamination, no drops).
    let a = methods_in(&table, "Ns.OuterA.Nested");
    let b = methods_in(&table, "Ns.OuterB.Nested");
    assert_eq!(a, vec!["Go".to_string()], "OuterA.Nested: {a:?}");
    assert_eq!(b, vec!["Other".to_string()], "OuterB.Nested: {b:?}");
}

#[test]
fn nested_partial_parts_merge_members() {
    // Same-key nested `partial` parts in different outer parts
    // merge members (AsyncProfiler.AsyncCallstack split across the
    // libraries and CoreCLR parts in CoreLib): members from BOTH
    // parts are in the table. (Assert on the table: static-call
    // lowering fail-opens on unknown methods, so `lower().is_ok()`
    // alone cannot discriminate.)
    let src = r#"namespace t;
static partial class Outer {
    internal static partial class Nested {
        internal static void A() { }
    }
}
static partial class Outer {
    internal static partial class Nested {
        internal static void B() { }
    }
}
class User {
    int main() { return 0; }
}"#;
    let pre = crate::preprocessor::preprocess(src, &[], "probe.rc").expect("preprocess");
    let unit = crate::parser::parse(&pre.source).unwrap();
    let table = crate::native::semantic::analyze(&unit).expect("analyze");
    let mut methods: Vec<String> = Vec::new();
    for c in &table.classes {
        if c.key.path == "t.Outer.Nested" {
            methods.extend(c.methods.iter().map(|m| m.name.clone()));
        }
    }
    assert!(methods.contains(&"A".to_string()), "part 1 member kept: {methods:?}");
    assert!(methods.contains(&"B".to_string()), "part 2 member merged: {methods:?}");
}

#[test]
fn method_anchoring_prefers_own_nested_entry() {
    // Method-anchored `current_info` must resolve to the entry under
    // lowering, not table first-match: 25 same-named `Enumerator`s
    // share names and signatures, and a wrong anchor types members
    // wrong (`_enumerator` as `IAsyncEnumerator` instead of the
    // sibling nested struct in CoreLib's OrderedDictionary).
    let src = r#"namespace t;
class Wrong {
}
class Right {
    public bool Foo() { return true; }
}
class Beta {
    internal static class Outer {
        internal struct Enumerator {
            private Right _x;
            public bool MoveNext() => _x.Foo();
        }
    }
}
class Alpha {
    internal struct Enumerator {
        private Wrong _x;
        public bool MoveNext() { return true; }
    }
    int main() { return 0; }
}"#;
    let pre = crate::preprocessor::preprocess(src, &[], "probe.rc").expect("preprocess");
    let unit = crate::parser::parse(&pre.source).unwrap();
    crate::native::ir::lower(&unit).expect("own nested entry must anchor");
}
