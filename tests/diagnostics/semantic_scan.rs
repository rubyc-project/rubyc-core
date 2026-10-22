//! Semantic scan over a recovered AST: E0025 (unknown name) and
//! E0026 (unknown function), with the conservative "anything declared
//! anywhere in the unit is never flagged" rule, builtin/primitive
//! allowlists, and did-you-mean hints.

use crate::diagnostics_semantic::scan;
use crate::models::{Diagnostic, Severity};
use crate::parser;

fn parse(src: &str) -> crate::models::CompilationUnit {
    parser::parse(src)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}\nsrc:\n{src}"))
}

fn codes(diags: &[Diagnostic]) -> Vec<&'static str> {
    diags.iter().map(|d| d.code).collect()
}

// ---------- clean programs produce nothing ----------

#[test]
fn clean_program_no_diags() {
    let unit = parse(
        "namespace n {
            class c {
                int main() {
                    printf(\"hi\\n\");
                    return 0;
                }
            }
        }",
    );
    assert!(codes(&scan(&unit)).is_empty());
}

#[test]
fn used_local_and_params_clean() {
    let unit = parse(
        "namespace n {
            class c {
                public int add(int a, int b) {
                    int s = a + b;
                    return s;
                }
            }
        }",
    );
    assert!(codes(&scan(&unit)).is_empty());
}

#[test]
fn member_access_on_this_clean() {
    let unit = parse(
        "namespace n {
            class c {
                int value = 0;
                public int get() {
                    return this.value;
                }
            }
        }",
    );
    assert!(codes(&scan(&unit)).is_empty());
}

// ---------- E0025: unknown name ----------

#[test]
fn unknown_ident_is_e0025() {
    let unit = parse(
        "namespace n {
            class c {
                public int f() {
                    int x = nope;
                    return x;
                }
            }
        }",
    );
    let diags = scan(&unit);
    assert!(codes(&diags).contains(&"E0025"));
    let d = diags.iter().find(|d| d.code == "E0025").unwrap();
    assert_eq!(d.severity, Severity::Error);
    assert!(d.message.contains("nope"));
}

#[test]
fn unknown_assignment_target_is_e0025() {
    let unit = parse(
        "namespace n {
            class c {
                public int f() {
                    mystery = 3;
                    return 0;
                }
            }
        }",
    );
    let diags = scan(&unit);
    assert!(codes(&diags).contains(&"E0025"));
    assert!(diags.iter().any(|d| d.message.contains("mystery")));
}

#[test]
fn builtin_names_never_flagged() {
    let unit = parse(
        "namespace n {
            class c {
                public int f() {
                    printf(\"x\");
                    print(\"y\");
                    return 0;
                }
            }
        }",
    );
    assert!(codes(&scan(&unit)).is_empty());
}

#[test]
fn primitive_type_names_not_flagged() {
    let unit = parse(
        "namespace n {
            class c {
                public int f() {
                    int x = int.MaxValue;
                    return x;
                }
            }
        }",
    );
    assert!(codes(&scan(&unit)).is_empty());
}

#[test]
fn did_you_mean_hint_populated() {
    // `heler` is close to the method `helper` (a candidate) -> E0026 with a
    // did-you-mean help.
    let unit = parse(
        "namespace n {
            class c {
                public int f() {
                    return heler();
                }
                public int helper() {
                    return 1;
                }
            }
        }",
    );
    let diags = scan(&unit);
    let d = diags.iter().find(|d| d.code == "E0026").unwrap();
    assert!(d.help.is_some(), "expected a did-you-mean hint, got none");
}

// ---------- E0026: unknown function ----------

#[test]
fn unknown_call_is_e0026() {
    let unit = parse(
        "namespace n {
            class c {
                public int f() {
                    missing_fn(1);
                    return 0;
                }
            }
        }",
    );
    let diags = scan(&unit);
    assert!(codes(&diags).contains(&"E0026"));
    let d = diags.iter().find(|d| d.code == "E0026").unwrap();
    assert!(d.message.contains("missing_fn"));
}

#[test]
fn call_to_same_unit_method_not_flagged() {
    let unit = parse(
        "namespace n {
            class c {
                public int f() {
                    return helper();
                }
                public int helper() {
                    return 1;
                }
            }
        }",
    );
    // `helper` is declared in the same unit, so the conservative scan must
    // not flag it (precise overload/access is downstream).
    assert!(!codes(&scan(&unit)).contains(&"E0026"));
}

// ---------- control flow walk ----------

#[test]
fn unknown_in_if_branch_flagged() {
    let unit = parse(
        "namespace n {
            class c {
                public int f() {
                    if (true) {
                        return ghost;
                    }
                    return 0;
                }
            }
        }",
    );
    assert!(codes(&scan(&unit)).contains(&"E0025"));
}

#[test]
fn unknown_in_loop_body_flagged() {
    let unit = parse(
        "namespace n {
            class c {
                public int f() {
                    while (true) {
                        callme();
                        break;
                    }
                    return 0;
                }
            }
        }",
    );
    assert!(codes(&scan(&unit)).contains(&"E0026"));
}

// ---------- inheritance: base members not flagged ----------

#[test]
fn inherited_field_not_flagged() {
    let unit = parse(
        "namespace n {
            class base1 {
                int count = 0;
            }
            class child {
                public int f() {
                    return this.count;
                }
            }
        }",
    );
    // `child` does not declare `count`, but the effective member set walks
    // the base chain. If the parser records the base, this must be clean.
    // (Conservative pass: even if not inherited, `any_member` includes
    // `count` from `base1`, so it is never flagged.)
    assert!(!codes(&scan(&unit)).contains(&"E0025"));
}

#[test]
fn empty_unit_clean() {
    let unit = parse("");
    assert!(codes(&scan(&unit)).is_empty());
}

#[test]
fn lambda_body_is_walked_so_its_names_resolve() {
    // A lambda parses into a real node now, and the scan WALKS its body:
    // a body's reads are real references, so an unknown name inside one is
    // reported rather than hidden behind the closure.
    let unit = parse(
        "namespace n {
            class c {
                int main() {
                    int y = 5;
                    System.Func<int,int> f = x => x + y;
                    return 0;
                }
            }
        }",
    );
    let diags = scan(&unit);
    assert!(
        !diags.iter().any(|d| d.message.contains("'x'") || d.message.contains("'y'")),
        "lambda parameters and captured locals must resolve: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
    // An unknown name INSIDE the body is still reported.
    let unit2 = parse(
        "namespace n {
            class c {
                int main() { System.Func<int,int> f = x => x + missing_local; return 0; }
            }
        }",
    );
    assert!(
        scan(&unit2).iter().any(|d| d.code == "E0025"),
        "an unknown name in a lambda body must be reported"
    );
}

#[test]
fn generic_type_params_in_scope() {
    // Method-level (`M<TOther>`) and class-level (`Box<TDecimal>`)
    // type parameters are valid value references in bodies — no more
    // E0025 for `TOther.Zero`, `typeof(TDecimal)`, etc.
    let unit = parse(
        "namespace n {
            class Box<TDecimal> {
                TDecimal M<TOther>(TDecimal d) {
                    int t = typeof(TOther);
                    int u = typeof(TDecimal);
                    return d;
                }
            }
        }",
    );
    let diags = scan(&unit);
    assert!(
        !diags.iter().any(|d| d.message.contains("TOther") || d.message.contains("TDecimal")),
        "type params must be in scope: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

/// Parse `src`, asserting the ONLY diagnostic is the unsupported-pattern
/// one (E0036). These probes used to assert that a fabricated binding
/// resolved; the pattern that produced it is reported now.
fn parse_expecting_unsupported_pattern(src: &str) -> Vec<crate::models::Diagnostic> {
    crate::parser::parse(src)
        .expect_err("the pattern is reported")
        .into_iter()
        .map(|d| d.to_owned())
        .collect()
}

#[test]
fn is_not_generic_pattern_binding_resolves() {
    // `other is not Box<int> b` — the binding now receives the value
    // being tested (it used to be a zero placeholder, so `b` was always
    // 0 and the body read garbage). The subject param and the binding
    // must both resolve.
    let unit = parse(
        "namespace n {
            class Box<T> { public int value = 0; }
            class c {
                public int f(object other) {
                    if (other is not Box<int> b) { return 0; }
                    return b.value;
                }
            }
        }",
    );
    let diags = scan(&unit);
    assert!(
        !diags.iter().any(|d| d.message.contains("'b'") || d.message.contains("'other'")),
        "the pattern binding and subject must resolve: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

#[test]
fn out_var_in_condition_visible_after() {
    // `out` vars from an `if` condition stay in the enclosing block
    // (C# scoping) — usable after the statement.
    let unit = parse(
        "namespace n {
            class c {
                public bool Try(out int v) {
                    v = 1;
                    return true;
                }
                public int f() {
                    int i = 0;
                    if (Try(out int x)) {
                        i = x;
                    }
                    i = i + x;
                    return i;
                }
            }
        }",
    );
    let diags = scan(&unit);
    assert!(
        !diags.iter().any(|d| d.message.contains("'x'")),
        "condition out-var must stay in scope: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

#[test]
fn expression_bodied_pattern_binding_resolves() {
    // `=> (obj is Plane other) && Equals(other)` — the binding is
    // initialised from the subject, so `other` resolves in the body.
    let unit = parse(
        "namespace n {
            class Plane {
                public bool Equals(object obj) => (obj is Plane other) && Equals(other);
                public bool Same(object obj) => obj is Plane;
            }
        }",
    );
    let diags = scan(&unit);
    assert!(
        !diags.iter().any(|d| d.message.contains("other")),
        "expression-body bindings must resolve: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

#[test]
fn using_alias_names_in_scope() {
    // `using nint_t = System.Int64;` — the alias is a valid value
    // reference in bodies (e.g. `nint_t.MaxValue`, locals of that
    // type); the target resolves downstream in the type table.
    let unit = parse(
        "using nint_t = System.Int64;
        namespace n {
            class c {
                public int f() {
                    nint_t v = nint_t.MaxValue;
                    return v;
                }
            }
        }",
    );
    let diags = scan(&unit);
    assert!(
        !diags.iter().any(|d| d.message.contains("nint_t")),
        "using aliases must be in scope: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

#[test]
fn typeof_and_sizeof_args_are_types() {
    // `typeof(T[])`, `typeof(TKey)` — the argument is a TYPE, never a
    // value reference (broke every `typeof(T[])` in CoreLib: `T[]`).
    let unit = parse(
        "namespace n {
            class Box<TKey> {
                public bool f(object array) {
                    int t = typeof(TKey);
                    bool b = array.GetType() != typeof(TKey[]);
                    return b;
                }
            }
        }",
    );
    let diags = scan(&unit);
    assert!(
        !diags.iter().any(|d| d.message.contains("TKey")),
        "typeof args must not flag: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

#[test]
fn namespace_qualified_access_not_flagged() {
    // `System.Array.Copy(...)` — a dotted path THROUGH a known type is
    // qualification (`System` is a namespace here), not a value.
    // A path through nothing known still flags its root (`Nope.X`).
    let unit = parse(
        "namespace n {
            class Array {
                public static void Copy() {
                }
            }
            class c {
                public void f() {
                    System.Array.Copy();
                    Nope.Missing();
                }
            }
        }",
    );
    let diags = scan(&unit);
    assert!(
        !diags.iter().any(|d| d.message.contains("'System'")),
        "namespace root must not flag: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
    assert!(
        diags.iter().any(|d| d.message.contains("Nope")),
        "unknown roots must still flag: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

#[test]
fn verbatim_this_resolves() {
    // `T[] @this = ...; return @this[i];` — `@this` lexes as the
    // identifier `this`, which is always in scope.
    let unit = parse(
        "namespace n {
            class c {
                public int get_Item(int index) {
                    int[] @this = GetArr();
                    return @this[index];
                }
                public int[] GetArr() {
                    return this.arr;
                }
                public int[] arr = null;
            }
        }",
    );
    let diags = scan(&unit);
    assert!(
        !diags.iter().any(|d| d.message.contains("this")),
        "verbatim this must resolve: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

#[test]
fn using_static_members_in_scope() {
    // `using static Helpers;` — the target's constants and methods
    // resolve unqualified in bodies (`MaxStackAllocArgCount`).
    let unit = parse(
        "using static n.Helpers;
        namespace n {
            class Helpers {
                public const int Limit = 4;
                public static int Twice(int v) {
                    return v;
                }
            }
            class c {
                public int f() {
                    int v = Limit;
                    return Twice(v);
                }
            }
        }",
    );
    let diags = scan(&unit);
    assert!(
        !diags.iter().any(|d| d.message.contains("Limit")),
        "using-static members must be in scope: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

#[test]
fn double_underscore_keywords_resolve() {
    // `__reftype(value)` / `__makeref(x)` — undocumented
    // TypedReference keywords: always resolvable builtins.
    let unit = parse(
        "namespace n {
            class c {
                public int f(object value) {
                    int h = __reftype(value).GetHashCode();
                    return h;
                }
            }
        }",
    );
    let diags = scan(&unit);
    assert!(
        !diags.iter().any(|d| d.message.contains("__reftype")),
        "__reftype must resolve: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

#[test]
fn out_var_in_switch_subject_visible_after() {
    // `switch (Foo(x, out int consumed))` — the `out` var belongs to
    // the enclosing block (used after the switch).
    let unit = parse(
        "namespace n {
            class c {
                public int Foo(int a, out int b) {
                    b = 1;
                    return b;
                }
                public int M(int x) {
                    int y = 0;
                    switch (Foo(x, out int consumed)) {
                        case 1:
                            y = consumed;
                            break;
                        default:
                            break;
                    }
                    y = y + consumed;
                    return y;
                }
            }
        }",
    );
    let diags = scan(&unit);
    assert!(
        !diags.iter().any(|d| d.message.contains("consumed")),
        "switch-subject out-var must stay in scope: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

#[test]
fn partial_class_members_union_across_parts() {
    // `partial class Box` in two declarations: a field declared in
    // one part resolves in the other part's bodies (mirroring .NET,
    // e.g. `RuntimeType.ObjectType` living in the CoreCLR part).
    // Bases union too (`: B` + `: C`).
    let unit = parse(
        "namespace n {
            class B {
            }
            class C {
            }
            partial class Box : B {
                int shared = 1;
            }
            partial class Box : C {
                public int f() {
                    return shared;
                }
            }
        }",
    );
    let diags = scan(&unit);
    assert!(
        !diags.iter().any(|d| d.message.contains("shared")),
        "partial members must union: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

#[test]
fn empty_property_pattern_binding_parses_and_binds() {
    // `x is { } state` is the null test with a designation: it binds the
    // subject. It used to be reported as an unsupported property pattern,
    // which blocked every `is {} x` in the libraries.
    let unit = parser::parse(
        "namespace n {
            class c {
                public int f(object x) {
                    if (Get(x) is { } state) {
                        return state;
                    }
                    return 0;
                }
                public object Get(object o) { return o; }
            }
        }",
    )
    .expect("`is { } state` parses");
    let text = format!("{unit:?}");
    assert!(
        text.contains("__rubyc_nonnull"),
        "the empty property pattern must lower as a null test: {text}"
    );
    assert!(
        text.contains("name: \"state\""),
        "the designation must be declared: {text}"
    );
}

#[test]
fn fixed_statement_locals_in_scope() {
    // `fixed (byte* p = ...)` — the pinned name resolves in the body.
    let unit = parse(
        "namespace n {
            class c {
                public unsafe void f(byte[] array) {
                    fixed (byte* p = array) {
                        int x = *p;
                    }
                }
            }
        }",
    );
    let diags = scan(&unit);
    assert!(
        !diags.iter().any(|d| d.message.contains("'p'")),
        "fixed locals must resolve: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

#[test]
fn primary_ctor_params_in_scope() {
    // `ref struct Holder(nint addr, ...)` — primary-constructor
    // params resolve throughout the body.
    let unit = parse(
        "namespace n {
            class C {
                internal ref struct Holder(nint addr, nuint length) {
                    public void Free() {
                        MUnmap(addr, length);
                    }
                }
            }
        }",
    );
    let diags = scan(&unit);
    assert!(
        !diags.iter().any(|d| d.message.contains("addr") || d.message.contains("length")),
        "primary-ctor params must resolve: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

#[test]
fn field_keyword_resolves() {
    // `get => field;` — the `field` contextual keyword (backing
    // field) always resolves, like `this`.
    let unit = parse(
        "namespace n {
            class c {
                public int Value {
                    get => field;
                }
            }
        }",
    );
    let diags = scan(&unit);
    assert!(
        !diags.iter().any(|d| d.message.contains("'field'")),
        "field keyword must resolve: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

#[test]
fn switch_expr_arm_bindings_resolve() {
    // `x switch { TypeName a => a.Foo, ... }` — arm pattern
    // bindings resolve in arm bodies.
    let unit = parse(
        "namespace n {
            class C {
                public int M(object x) {
                    int v = x switch {
                        string a => a.Length,
                        int b => b + 1,
                        _ => 0,
                    };
                    return v;
                }
            }
        }",
    );
    let diags = scan(&unit);
    assert!(
        !diags.iter().any(|d| d.message.contains("'a'") || d.message.contains("'b'")),
        "switch-arm bindings must resolve: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

#[test]
fn nested_type_names_in_scope() {
    // Nested types (`Number.BigInteger`) are valid value references in
    // the enclosing class's bodies — no E0025 for `BigInteger.SetZero`.
    let unit = parse(
        "namespace System {
            static partial class Number {
                internal ref struct BigInteger {
                    public static int SetZero() {
                        return 0;
                    }
                }
                public static int Convert() {
                    int v = BigInteger.SetZero();
                    return v;
                }
            }
        }",
    );
    let diags = scan(&unit);
    assert!(
        !diags.iter().any(|d| d.message.contains("BigInteger")),
        "nested types must be in scope: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

#[test]
fn inherited_property_via_qualified_base_clean() {
    // Bare `Count` (implicit-this property inherited through a
    // QUALIFIED base spelling, `A.Base`) must not E0025: the
    // effective-member walk compares simple names.
    let unit = parse(
        "namespace A { class Base { public int Count { get { return 0; } } } }
         namespace t { class Derived : A.Base { int M() { return Count; } int main() { return 0; } } }",
    );
    assert!(codes(&scan(&unit)).is_empty());
}

#[test]
fn inner_pattern_designations_bind_the_subject() {
    // Every one of these patterns used to bind a variable to zero and
    // test false, then to be reported. Property and tuple patterns now
    // lower: members are tested against the subject and the designation
    // receives it.
    for src in [
        "class JsonTypeInfo { public bool IsNullable { get { return false; } } public JsonTypeInfo ElementTypeInfo { get { return null; } } } class C { int Main() { JsonTypeInfo typeInfo = null; if (typeInfo is { IsNullable: true, ElementTypeInfo: JsonTypeInfo elementTypeInfo }) { typeInfo = elementTypeInfo; } return 0; } }",
        "class C { int Main() { object v = null; object x = null; if (v is { HasDefaultValue: true, DefaultValue: object defaultValue }) { x = defaultValue; } return 0; } }",
        "class C { int Main() { object v = null; object x = null; if (v is (int a, string b)) { x = a; } return 0; } }",
        "class C { int Main() { object v = null; switch (v) { case System.Text.Json.JsonTypeInfo info: return 1; default: return 0; } } }",
    ] {
        let diags = crate::diagnostics_semantic::scan(
            &parser::parse(src).unwrap_or_else(|e| panic!("must parse: {e:?}")),
        );
        assert!(
            !diags.iter().any(|d| d.code == "E0036" || d.code == "E0025"),
            "no pattern diagnostic for:\n{src}\n{diags:?}"
        );
    }
}

#[test]
fn generic_bare_call_name_stripped_for_scope() {
    // `M<T>(...)` looks up `M` (explicit type args are not part of
    // the name). Regression: once the parser kept `<T>` on bare-call
    // callees, the scan flagged every generic call (`unknown function
    // 'Create<ulong>'`, thousands across numerics/Xml).
    let unit = parse(
        "class C { T M<T>() { return null; } int Main() { object x = M<int>(); return 0; } }",
    );
    assert!(codes(&scan(&unit)).is_empty());
}

#[test]
fn this_call_to_unknown_member_is_e0026() {
    // `this.M()` was routed through a placeholder hook that always
    // answered "known", so a typo on an explicit `this.` call could
    // never be reported. It resolves against the enclosing class's
    // effective (own + inherited) method set now.
    let unit = parse(
        "namespace n {
            class c {
                public int f() {
                    return this.heler();
                }
                public int helper() { return 1; }
            }
        }",
    );
    let diags = scan(&unit);
    let d = diags.iter().find(|d| d.code == "E0026").unwrap();
    assert!(d.message.contains("heler"), "names the member: {}", d.message);
}

#[test]
fn this_call_to_inherited_member_is_not_flagged() {
    // The member set is effective, so an inherited method is found on
    // a derived receiver.
    let unit = parse(
        "namespace n {
            class base_c { public int helper() { return 1; } }
            class c : base_c {
                public int f() { return this.helper(); }
            }
        }",
    );
    assert!(!codes(&scan(&unit)).contains(&"E0026"));
}

#[test]
fn lexer_reports_an_unexpected_byte_and_keeps_going() {
    // A byte with no lexical rule used to consume one byte and return
    // `Eof`, silently truncating the file: the declarations after it
    // vanished and the build "succeeded" on half a class. It is now
    // an error that names the byte, and lexing continues so the rest
    // of the file is still analyzed.
    let errs = crate::parser::parse(
        "namespace n {
            class c {
                public int f() { int x = 1 $ 2; return x; }
                public int g() { int y = 3 @ 4; return y; }
            }
        }",
    )
    .expect_err("an unexpected byte must not parse cleanly");
    let bytes: Vec<&str> = errs
        .iter()
        .filter(|e| e.code == "E0006")
        .map(|e| e.message.as_str())
        .collect();
    assert_eq!(
        bytes.len(),
        2,
        "both offending bytes are reported (lexing did not stop at the first): {bytes:?}"
    );
    assert!(bytes[0].contains("0x24"), "names `$`: {}", bytes[0]);
    assert!(bytes[1].contains("0x40"), "names `@`: {}", bytes[1]);
}
