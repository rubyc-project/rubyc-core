//! `out` variables with explicit types (`TypeNameParser` shape).
//!
//! - `out bool b` declares a bool (not var-inferred from the zero
//!   init, which would mistype it as int and break overload ranking
//!   for `bool` parameters).
//! - Unqualified calls through `using static` rank with the declared
//!   out-var type (`IsBeginningOfGenericArgs(ref span, out bool)`
//!   in CoreLib).

fn lower(src: &str) -> Result<crate::native::ir::Program, String> {
    let pre = crate::preprocessor::preprocess(src, &[], "probe.rc").expect("preprocess");
    let unit = crate::parser::parse(&pre.source).unwrap();
    crate::native::ir::lower(&unit).map_err(|e| format!("{e:?}"))
}

#[test]
fn out_var_keeps_declared_type_for_ranking() {
    let r = lower(
        r#"using static t.H;
namespace t {
    static class H {
        internal static bool Go(ref int x, out bool y) { y = true; return true; }
    }
    struct P {
        int _i;
        bool M() { return Go(ref _i, out bool b); }
        int main() { return 0; }
    }
}"#,
    );
    assert!(r.is_ok(), "out bool var must rank for bool param: {r:?}");
}

#[test]
fn out_discard_ranks_for_any_param() {
    // `out _` carries no type: it must not veto overloads with
    // non-nullable value-type params (`out bool` in CoreLib).
    let r = lower(
        r#"using static t.H;
namespace t {
    static class H {
        internal static bool Go(ref int x, out bool y) { y = true; return true; }
    }
    struct P {
        int _i;
        bool M() { return Go(ref _i, out _); }
        int main() { return 0; }
    }
}"#,
    );
    assert!(r.is_ok(), "out discard must rank: {r:?}");
}

#[test]
fn base_init_out_var_visible_in_body() {
    // `: base(M(k, out K dup))` declares `dup` for the constructor
    // body — and the out slot itself must not read before its
    // declaration lowers (the base call precedes the body, so a bare
    // read fails; the slot carries `default(T)` instead). Both the
    // explicit-type and `out var` forms (cf. CompositeMLDsaCng).
    for out_arg in ["out K dup", "out var dup"] {
        let src = format!(
            r#"namespace t {{
    class K {{ }}
    class Algo {{ }}
    class Base {{
        public Base(Algo a) {{ }}
    }}
    class P : Base {{
        private K _key;
        public P(K key) : base(M(key, {out_arg})) {{
            _key = dup;
        }}
        private static Algo M(K key, out K d) {{ d = key; return new Algo(); }}
        int main() {{ return 0; }}
    }}
}}"#
        );
        let r = lower(&src);
        assert!(r.is_ok(), "{out_arg} in base-init must lower: {r:?}");
    }
}
