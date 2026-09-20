//! Extension-method calls (`recv.Ext(args)` → `Class.Ext(recv, args)`).
//!
//! A static method whose first parameter carries `this` is callable
//! in member syntax on any receiver whose base type matches the
//! receiver parameter — including primitives and interface-typed
//! values with no class entry of their own.

use crate::native::ir::{Instr, Program};

fn lower(src: &str) -> Program {
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).unwrap()
}

fn has(p: &Program, pred: impl Fn(&Instr) -> bool) -> bool {
    p.functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .any(pred)
}

#[test]
fn extension_call_on_primitive_receiver() {
    let p = lower(
        r#"namespace t;
class Ext {
    public static int Twice(this int x) { return x; }
}
class C {
    int main() {
        int v = 3;
        return v.Twice();
    }
}"#,
    );
    // `v.Twice()` → direct static call `Ext.Twice(v)` (2 slots: one
    // function arg — the receiver — plus the `this` receiver vreg).
    assert!(
        has(
            &p,
            |i| matches!(i, Instr::CallStatic { args, .. } if args.len() == 1)
        ),
        "expected extension call to lower as CallStatic with receiver"
    );
}

#[test]
fn extension_call_fills_optional_tail() {
    let p = lower(
        r#"namespace t;
class Ext {
    public static int Add(this int x, int y = 5) { return x; }
}
class C {
    int main() {
        int v = 3;
        return v.Add();
    }
}"#,
    );
    // `v.Add()` → `Ext.Add(v, 5)`: receiver plus filled default.
    assert!(
        has(
            &p,
            |i| matches!(i, Instr::CallStatic { args, .. } if args.len() == 2)
        ),
        "expected extension call with filled optional"
    );
}

#[test]
fn member_beats_extension_with_same_name() {
    // A real member wins over an extension with the same name (C#
    // resolution order): the call dispatches virtually, not statically.
    let p = lower(
        r#"namespace t;
class Ext {
    public static int N(this Box b) { return 0; }
}
class Box {
    public int N() { return 1; }
}
class C {
    Box _b;
    int main() { return _b.N(); }
}"#,
    );
    let mut saw_virt = false;
    let mut static_targets = Vec::new();
    for f in &p.functions {
        for b in &f.blocks {
            for i in &b.instrs {
                match i {
                    Instr::CallVirt { .. } => saw_virt = true,
                    Instr::CallStatic { fidx, .. } => {
                        static_targets.push(p.functions[*fidx as usize].name.clone())
                    }
                    _ => {}
                }
            }
        }
    }
    assert!(saw_virt, "member call must dispatch virtually");
    assert!(
        !static_targets.iter().any(|n| n.contains("Ext::N")),
        "extension must not be selected: {static_targets:?}"
    );
}

#[test]
fn array_segment_converts_to_span_param() {
    // `ArraySegment<T>` converts implicitly to the span/memory views:
    // an `ArraySegment<byte>` argument ranks for `ReadOnlySpan<byte>`
    // (e.g. Crypto passing one to `WriteEncodedValueForCrypto`).
    let p = lower(
        r#"namespace t;
static class E {
    public static int Use(this C c, System.ReadOnlySpan<byte> s) { return 1; }
}
class C {
    int Go(System.ArraySegment<byte> seg) { return this.Use(seg); }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallStatic { .. })),
        "ArraySegment arg must select the span extension"
    );
}

#[test]
fn extension_chain_with_implicit_span_conversion() {
    // `s.AsSpan().StartsWith(t)`: the chained call keeps the declared
    // span result type, and a `string` argument converts to a span
    // parameter (C# implicit conversion).
    let p = lower(
        r#"namespace t;
class ReadOnlySpan { }
class Ext {
    public static ReadOnlySpan AsSpan(this string s) { ReadOnlySpan r = new ReadOnlySpan(); return r; }
    public static int StartsWith(this ReadOnlySpan a, ReadOnlySpan b) { return 1; }
}
class C {
    int main() {
        string s = "hi";
        string t = "ho";
        return s.AsSpan().StartsWith(t);
    }
}"#,
    );
    assert!(
        has(
            &p,
            |i| matches!(i, Instr::CallStatic { args, .. } if args.len() == 2)
        ),
        "expected chained extension call with converted arg"
    );
}

#[test]
fn extension_call_accepts_generic_param_arg() {
    // `r.StartsWith(5)` with `StartsWith<T>(this ROS<T>, T)`: a bare
    // method-generic parameter accepts any argument (generic arguments
    // are not recorded on types, so there is nothing to infer from —
    // full inference arrives with generic preservation).
    let p = lower(
        r#"namespace t;
class ROS<T> { }
class Ext {
    public static int StartsWith<T>(this ROS<T> a, T b) { return 1; }
}
class C {
    int main() {
        ROS<int> r = new ROS<int>();
        return r.StartsWith(5);
    }
}"#,
    );
    assert!(
        has(
            &p,
            |i| matches!(i, Instr::CallStatic { args, .. } if args.len() == 2)
        ),
        "expected generic extension call with inferred type"
    );
}

#[test]
fn extension_on_object_param_accepts_any_receiver() {
    // `this.GetRawData()` resolves through `GetRawData(this object)`:
    // every type implicitly converts to the `object` receiver.
    let p = lower(
        r#"namespace t;
class H {
    public static byte GetRawData(this object o) { return 0; }
}
class E {
    public void HasFlag() {
        byte b = this.GetRawData();
    }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(
            &p,
            |i| matches!(i, Instr::CallStatic { args, .. } if args.len() == 1)
        ),
        "expected object-receiver extension call"
    );
}

#[test]
fn unqualified_call_reaches_global_static() {
    // `using static` imports: an unqualified call with no local,
    // member, or enclosing-scope target resolves to any compatible
    // unit-wide static (`RotateLeft(...)` via `BitOperations`).
    let p = lower(
        r#"namespace t;
class BitOps {
    public static int Rol(int v, int s) { return v; }
}
class Marvin {
    public int Run(int p) {
        return Rol(p, 16);
    }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(
            &p,
            |i| matches!(i, Instr::CallStatic { args, .. } if args.len() == 2)
        ),
        "expected direct static call through global fallback"
    );
}

#[test]
fn span_converts_to_readonly_span_param() {
    // `ros.Overlaps(span)`: `Span<T>` implicitly converts to the
    // `ReadOnlySpan<T>` parameter (C# implicit conversion).
    let p = lower(
        r#"namespace t;
class Span { }
class ReadOnlySpan { }
class Ext {
    public static bool Overlaps(this ReadOnlySpan a, ReadOnlySpan b) { return true; }
}
class C {
    int main() {
        ReadOnlySpan ros = new ReadOnlySpan();
        Span sp = new Span();
        bool b = ros.Overlaps(sp);
        return 0;
    }
}"#,
    );
    assert!(
        has(
            &p,
            |i| matches!(i, Instr::CallStatic { args, .. } if args.len() == 2)
        ),
        "expected span-converted extension call"
    );
}

#[test]
fn extension_call_accepts_method_group_for_delegate_param() {
    // `c.GetKey<T>(Has)` passes a method group to `Predicate<Cert>`:
    // C# converts the group to the delegate. Shape mirrors
    // `X509Certificate2.GetPublicKey<ECDsa>(HasECDsaKeyUsage)`. The
    // delegate is DECLARED (as CoreLib declares `Predicate<T>`) —
    // convertibility resolves against declarations, never a name list.
    let p = lower(
        r#"namespace t;
class Cert { }
public delegate bool Predicate<T>(T obj);
static class E {
    public static bool GetKey<T>(this Cert c, Predicate<Cert>? p = null) { return true; }
    private static bool Has(Cert c) { return true; }
    public static bool Go<T>(Cert c) { return c.GetKey<T>(Has); }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallStatic { .. })),
        "method-group call must select the delegate extension"
    );
}

#[test]
fn method_group_rejected_for_non_delegate_param() {
    // A method group converts ONLY to compatible delegates: passing
    // one to an `int` parameter stays a loud arity error, not a
    // silent mis-rank.
    let src = r#"namespace t;
class Cert { }
static class E {
    public static bool TakeInt(this Cert c, int x) { return true; }
    private static bool Has(Cert c) { return true; }
    public static bool Go(Cert c) { return c.TakeInt(Has); }
    int main() { return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    assert!(
        crate::native::ir::lower(&unit).is_err(),
        "method group for int param must fail"
    );
}

#[test]
fn array_length_types_as_int_for_ranking() {
    // `arr.Length` (no class entry for arrays) types as `int`, not
    // `object` — otherwise extension overloads with `int` params
    // never rank (e.g. `BigInteger` arg + `Length` arg in Crypto's
    // `ExportKeyParameter` call).
    let p = lower(
        r#"namespace t;
class C {
    static byte[] Take(byte[] b, int n) { return b; }
    byte[] Go(byte[] arr) { return C.Take(arr, arr.Length); }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallStatic { .. })),
        "array Length arg must select the int overload"
    );
}

#[test]
fn using_static_property_types_argument() {
    // A bare name resolving to a unique unit-wide static property
    // (`Decimal_Ctor_Int32` via `using static ...CachedReflectionInfo`)
    // types as its declared type for ranking (file-scoped usings
    // approximated unit-wide; ambiguity falls back to legacy).
    let p = lower(
        r#"namespace t;
class CI { }
static class CRI {
    public static CI Ctor { get { return null; } }
}
static class X {
    public static int Call(CI c) { return 1; }
}
class P {
    int Go() { return X.Call(Ctor); }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallStatic { .. })),
        "using-static property must type the argument"
    );
}

#[test]
fn span_index_types_as_element() {
    // `bits[0]` on a `Span<int>` types as `int` (not `object`), so
    // overloads with `int` params rank (cf. `EmitPrimitive(bits[0])`
    // in Expressions' `EmitDecimal`). Both locals (declared type
    // keeps its generic arguments) and params.
    let p = lower(
        r#"namespace t;
class C {
    static void Take(int x) { }
    static void Go(System.Span<int> bits) { Take(bits[0]); }
    static void Loc() { System.Span<int> l = default; Take(l[0]); }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallStatic { .. } | Instr::CallFn { .. })),
        "span index must type as the element"
    );
}

#[test]
fn primitive_alias_receiver_types_return() {
    // `decimal.ToUInt32(v)` types as `uint` (not `object`): lowercase
    // primitive aliases resolve to their framework entries for
    // return-type inference, keeping downstream overload ranking
    // precise (cf. `EmitPrimitive(decimal.ToUInt32(value))`).
    let p = lower(
        r#"namespace t;
class Decimal {
    public static uint ToUInt32(decimal d) { return 0; }
}
static class D {
    public static void Take(uint x) { }
}
class P {
    void Go(decimal v) { D.Take(decimal.ToUInt32(v)); }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallStatic { .. })),
        "primitive static call must type its return"
    );
}

#[test]
fn bare_call_return_types_argument() {
    // `X.Call(GetIdx(i))`: a bare same-class call types as its
    // return type (not `object`), keeping extension overload ranking
    // precise (cf. `EmitLoadArg(GetLambdaArgument(index))`).
    let p = lower(
        r#"namespace t;
class ILG { }
static class X {
    public static void EmitLoadArg(this ILG il, int index) { }
}
class C {
    int GetIdx(int i) { return i; }
    ILG _ilg;
    void Go(int index) { _ilg.EmitLoadArg(GetIdx(index)); }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallStatic { .. })),
        "bare call return must type the extension argument"
    );
}

#[test]
fn nested_sibling_write_uses_method_entry() {
    // Same-named nested classes (`A.Builder` with `_count` vs
    // `S.Builder` without): a field write in `A.Builder.RemoveRange`
    // must use the method's own entry, not table first-match
    // (cf. `ImmutableArray.Builder.RemoveRange` vs
    // `ImmutableSortedSet.Builder`).
    let p = lower(
        r#"namespace t;
class A {
    public sealed class Builder {
        private int _count;
        public void RemoveRange(int index, int length) {
            this._count -= length;
        }
    }
}
class S {
    public sealed class Builder {
        public void Clear() { }
    }
}
class P {
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::StoreField { .. })),
        "anchored field write must lower"
    );
}

#[test]
fn using_static_const_types_argument() {
    // Same as properties, for `const` fields (which live in their own
    // table vec): `Prime64_1` types as `ulong` for ranking
    // (cf. `Multiply64To128(keyed, Prime64_1 + ...)`).
    let p = lower(
        r#"namespace t;
static class K {
    public const ulong Prime64_1 = 1;
    public static ulong Mix(ulong a, ulong b) { return a; }
}
class P {
    ulong Go(ulong keyed, uint n) { return K.Mix(keyed, Prime64_1 + n); }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallStatic { .. })),
        "using-static const must type the argument"
    );
}

#[test]
fn extension_wins_over_inaccessible_instance() {
    // `reader.GetUnescapedSpan()`: a PRIVATE instance method plus a
    // PUBLIC extension with the same name (System.Text.Json's
    // `Utf8JsonReader` + `JsonHelpers`). C# picks the extension —
    // the private instance is not a candidate from outside.
    let p = lower(
        r#"namespace t;
class Reader {
    private int Value() { return 1; }
}
class Helpers {
    public static int Value(this Reader r) { return 2; }
}
class C {
    int main() {
        Reader r = null;
        return r.Value();
    }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallStatic { .. })),
        "inaccessible instance must fall back to the visible extension"
    );
}

#[test]
fn range_slice_types_as_container() {
    // `span[..n].ToArray()`: a range slice has the CONTAINER's type
    // (`Span<T>`), not the element's — the call resolves against it
    // (`Array.FindAll`: `span[..foundCount].ToArray()` failed with
    // "'T' has no member 'ToArray'").
    let p = lower(
        r#"namespace t;
class Span<T> { }
static class MemoryExtensions {
    public static T[] ToArray<T>(this Span<T> span) { return null; }
}
class C {
    T[] M<T>(Span<T> span, int n) { return span[..n].ToArray(); }
    int Main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallStatic { .. })),
        "slice call must resolve against the container type"
    );
}

#[test]
fn extension_call_on_typevar_instantiated_interface() {
    // `source.Contains(value)` where `source` is `IEnumerable<T>?`
    // (an interface instantiated with a class type variable, assigned
    // inside a loop): the extension must resolve and emit a real call,
    // not a silent placeholder (companion to the `UnionIterator`
    // `base.Contains` shadowing fix — this locks the receiver side).
    let p = lower(
        r#"namespace System.Collections.Generic {
interface IEnumerable<T> { }
}
namespace System.Linq {
using System.Collections.Generic;
static class E {
    public static bool Contains<T>(this IEnumerable<T> s, T v) { return true; }
}
static class Host {
    class Inner<T> {
        internal abstract IEnumerable<T>? Get(int i);
        public bool Has(T v) {
            IEnumerable<T>? source;
            for (int i = 0; (source = Get(i)) is not null; i++) {
                if (source.Contains(v)) {
                    return true;
                }
            }
            return false;
        }
    }
}
}
namespace t {
class C {
    int Main() { return 0; }
}}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallStatic { .. })),
        "extension call on type-var interface must emit CallStatic"
    );
}
