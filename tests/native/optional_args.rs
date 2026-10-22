//! Optional parameters and field-typed receivers.
//!
//! - Calls omitting trailing defaulted params fill from the folded
//!   defaults (`M(1)` → `M(1, 5)`), for static, instance, and ctor
//!   calls (C# optional arguments, e.g. `ThrowIfNull(obj)`).
//! - An implicit-`this` field read types as the FIELD's declared type,
//!   not the current class (`_buffer.IndexOf(c, i)` resolves through
//!   `string`, not `StringParser`).

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
fn omitted_optional_static_arg_fills_default() {
    let p = lower(
        r#"namespace t; class A {
    public static int M(int x, int y = 5) { return x; }
    int main() { return A.M(1); }
}"#,
    );
    // `A.M(1)` lowers as a direct static call with BOTH args (the
    // omitted `y` filled from its folded default).
    assert!(
        has(
            &p,
            |i| matches!(i, Instr::CallStatic { args, .. } if args.len() == 2)
        ),
        "expected CallStatic with filled optional arg"
    );
}

#[test]
fn omitted_optional_instance_arg_fills_default() {
    let p = lower(
        r#"namespace t; class A {
    public int N(int x, int y = 5) { return x; }
    int main() { A a = new A(); return a.N(1); }
}"#,
    );
    // `a.N(1)` keeps virtual dispatch but passes the full arity.
    assert!(
        has(
            &p,
            |i| matches!(i, Instr::CallVirt { args, .. } if args.len() == 2)
        ),
        "expected CallVirt with filled optional arg"
    );
}

#[test]
fn omitted_optional_ctor_arg_fills_default() {
    let p = lower(
        r#"namespace t; class A {
    public A(int x = 3) { }
    int main() { A a = new A(); return 0; }
}"#,
    );
    // `new A()` passes obj + the filled default.
    assert!(
        has(
            &p,
            |i| matches!(i, Instr::CallFn { args, .. } if args.len() == 2)
        ),
        "expected ctor CallFn with filled optional arg"
    );
}

#[test]
fn field_receiver_resolves_through_field_type() {    let p = lower(
        r#"namespace t;
class B {
    public int N(int x) { return x; }
}
class A {
    B _b;
    int main() { return _b.N(1); }
}"#,
    );
    // `_b` types as `B` (not `A`): virtual dispatch to `B.N`.
    assert!(
        has(&p, |i| matches!(i, Instr::CallVirt { .. })),
        "expected CallVirt through the field's type"
    );
}

#[test]
fn forward_local_function_call_resolves() {    // Local-function bodies are discarded (calls yield zero), but the
    // NAME is visible throughout the enclosing block — including for
    // calls that precede the declaration (C# scoping).
    let p = lower(
        r#"namespace t; class A {
    int main() {
        int r = Helper(41);
        return r;
        static int Helper(int x) { return x + 1; }
    }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::Const { imm: 0, .. })),
        "expected forward local-function call to lower to zero"
    );
}

#[test]
fn bodyless_static_call_yields_zero() {    // `extern` statics generate no function; the call still lowers
    // (args evaluated, zero yielded) instead of erroring.
    let p = lower(
        r#"namespace t; class A {
    public static extern int Ext(int x);
    int main() { return A.Ext(1); }
}"#,
    );
    assert!(
        !has(&p, |i| matches!(i, Instr::CallStatic { .. })),
        "expected no direct call to a bodyless method"
    );
    assert!(
        has(&p, |i| matches!(i, Instr::Const { imm: 0, .. })),
        "expected zero placeholder for bodyless call"
    );
}

#[test]
fn implicit_object_base_resolves_inherited_call() {
    // Classes with no explicit base implicitly derive from `object`:
    // an unqualified `ToString()` dispatches virtually, not errors.
    let p = lower(
        r#"namespace t;
class Object {
    public string ToString() { return "o"; }
}
class A {
    int main() { ToString(); return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallVirt { .. })),
        "expected virtual dispatch through implicit object base"
    );
}

#[test]
fn implicit_object_base_is_transitive() {    // `Stream : Mid` with baseless `Mid` still sees `object` members
    // through the transitive chain.
    let p = lower(
        r#"namespace t;
class Object {
    public string Talk() { return "o"; }
}
class Mid { }
class Stream : Mid {
    int main() { Talk(); return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallVirt { .. })),
        "expected virtual dispatch through transitive object base"
    );
}

#[test]
fn out_var_declaration_precedes_call() {
    // `out int x` declares `x` for the call itself and everything
    // after it (evaluated as zero — out args are write-only).
    let p = lower(
        r#"namespace t; class M {
    int Try(out int v) { v = 1; return 0; }
    int main() {
        int r = Try(out int x);
        return x;
    }
}"#,
    );
    assert!(
        has(
            &p,
            |i| matches!(i, Instr::CallStatic { .. } | Instr::CallFn { .. } | Instr::CallVirt { .. })
        ),
        "expected the out-var call to lower"
    );
}

#[test]
fn overload_access_uses_selected_overload() {
    // Overloads may declare different access: a `private` 3-arg
    // overload must not deny a call that selects the `public` 2-arg
    // overload (name-only lookup conflates them).
    let p = lower(
        r#"namespace t;
class F {
    private static int M(string a, string b, string c) { return 2; }
    public static int M(string a, string b) { return 1; }
}
class D {
    int main() { return F.M("a", "b"); }
}"#,
    );
    assert!(
        has(
            &p,
            |i| matches!(i, Instr::CallStatic { args, .. } if args.len() == 2)
        ),
        "expected public overload call to lower"
    );
}

#[test]
fn overload_selection_prefers_type_compatible() {
    // Same-arity overloads with incompatible parameter types lose to a
    // defaults-fillable overload whose given args all match (enum
    // arguments compare as their enum, not the underlying `int`).
    let p = lower(
        r#"namespace t;
enum FileMode { Open, Create }
enum OpenFlags { ReadOnly }
class H {
    private static int Open(string path, OpenFlags flags, int mode, bool b, string e) { return 0; }
    internal static int Open(string fullPath, FileMode mode, int access, int share, int options, long prealloc, int ucm = 0, string e = null) { return 1; }
}
class D {
    int main() { return H.Open("p", FileMode.Open, 1, 1, 1, 2); }
}"#,
    );
    // The internal 8-param overload (6 given + 2 filled), not the
    // private 6-param one.
    assert!(
        has(
            &p,
            |i| matches!(i, Instr::CallStatic { args, .. } if args.len() == 8)
        ),
        "expected type-compatible overload with filled defaults"
    );
}

#[test]
fn ctor_selection_prefers_type_compatible() {
    // Same-arity ctors with different access/types: a `private`
    // same-arity overload must not shadow the applicable `public`
    // one (`new DateTimeOffset(ticks, default)`).
    let p = lower(
        r#"namespace t;
class T {
    public T(long ticks) { }
    private T(int a, int b) { }
    public T(long ticks, T off) { }
}
class D {
    int main() { T t = new T(5, default); return 0; }
}"#,
    );
    // The public 2-param ctor (object + 2 args).
    assert!(
        has(
            &p,
            |i| matches!(i, Instr::CallFn { args, .. } if args.len() == 3)
        ),
        "expected applicable public ctor call"
    );
}

#[test]
fn property_typed_arg_selects_compatible_overload() {
    // Explicit-property reads type as the DECLARED type (`Ticks` is
    // `long`, not `object`): the `public M(long)` overload wins over
    // the inapplicable `private M(string)` (which would deny access).
    let p = lower(
        r#"namespace t;
class D {
    public long Ticks { get { return 0; } }
}
class E {
    private static int M(string x) { return 0; }
    public static int M(long x) { return 1; }
}
class C {
    int main() { D d = new D(); return E.M(d.Ticks); }
}"#,
    );
    assert!(
        has(
            &p,
            |i| matches!(i, Instr::CallStatic { args, .. } if args.len() == 1)
        ),
        "expected property-typed overload call"
    );
}

#[test]
fn inherited_static_never_shadows_instance_call() {
    // C# never considers static methods for instance syntax: an
    // inherited `static M(object)` must not shadow (or error for) an
    // instance `c.M("hi")` that selects `M(string)`.
    let p = lower(
        r#"namespace t;
class Base {
    public static int M(object o) { return 0; }
}
class C : Base {
    public int M(string s) { return 1; }
    int main() { C c = new C(); return c.M("hi"); }
}"#,
    );
    assert!(
        has(
            &p,
            |i| matches!(i, Instr::CallVirt { args, .. } if args.len() == 1)
        ),
        "expected virtual dispatch to the instance overload"
    );
}

#[test]
fn primary_ctor_param_visible_in_field_initializer() {
    // `class R(H h)` with `H _h = h;`: the construction argument
    // binds for field initializers (no leaking past the `new`).
    let p = lower(
        r#"namespace t;
class H {
    public H() { }
}
class R(H h) {
    H _h = h;
    public H Get() { return h; }
}
class C {
    int main() {
        H h = new H();
        R r = new R(h);
        return 0;
    }
}"#,
    );
    assert!(
        has(
            &p,
            |i| matches!(i, Instr::CallFn { args, .. } if args.len() == 1)
        ),
        "expected primary construction to lower"
    );
}

#[test]
fn nested_static_call_lowers_to_function() {
    // Nested-class static methods lower to real functions: a
    // qualified `ContingentProperties.SetEvent(1)` call dispatches
    // statically instead of erroring as a missing member.
    let p = lower(
        r#"namespace t;
class Task {
    internal sealed class ContingentProperties {
        internal static void SetEvent(int m) { }
    }
    public void Dispose() {
        ContingentProperties.SetEvent(1);
    }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallStatic { .. })),
        "expected static call into nested class"
    );
}

#[test]
fn same_named_nested_fields_resolve_per_class() {
    // Two `DebugView`s nested in different siblings declare
    // different fields: each ctor's assignment resolves through the
    // entry actually declaring it (first-match alone would fail the
    // second). Qualified identity across same-named nesteds is
    // future work; first hit wins.
    let p = lower(
        r#"namespace t;
class Outer1 {
    class DebugView {
        int _x;
        public DebugView() { _x = 1; }
    }
}
class Outer2 {
    class DebugView {
        int _y;
        public DebugView() { _y = 2; }
    }
}
class C {
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::StoreField { .. })),
        "expected both nested field stores to lower"
    );
}

#[test]
fn plain_call_falls_back_to_same_named_sibling() {
    // Generic-arity siblings share a simple lowering name
    // (`WhenAllPromise<T>` lowers as `WhenAllPromise`): an
    // unqualified call missing from the first-match entry resolves
    // through a same-named sibling instead of erroring.
    let p = lower(
        r#"namespace t;
class Box<T> {
    public void Invoke() {
        bool b = Set(5);
    }
}
class Box {
    internal bool Set(int v) { return true; }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallVirt { .. })),
        "expected virtual dispatch through sibling overload"
    );
}

#[test]
fn interface_member_with_no_implementer_uses_the_shared_interface_slot() {
    // `IEnumerator<T>` sits beside a non-generic `IEnumerator`, and the
    // call is made on the non-generic one. Nothing in this unit
    // implements either interface, so no class row can supply a slot.
    // Interface members live in a range ABOVE every class slot, shared
    // by all implementers, so the call has a defined index even with no
    // implementer in the unit; the null receiver faults at run time,
    // exactly as C# does. Dispatching to a class slot (the old
    // fallback) would jump into whatever function held that index.
    let src = r#"namespace t;
interface IEnumerator<T> {
    T Current();
}
interface IEnumerator {
    bool MoveNext();
}
class C {
    int main() {
        IEnumerator e = null;
        bool b = e.MoveNext();
        return 0;
    }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    let table = crate::native::semantic::analyze(&unit).unwrap();
    let slot = crate::native::semantic::iface_slot_by_name(&table, "IEnumerator", "MoveNext")
        .expect("a declared interface member has a slot");
    assert!(
        slot >= crate::native::semantic::iface_slot_base(&table),
        "interface slots live above the class range (slot {slot})"
    );
    let p = crate::native::ir::lower(&unit).expect("interface call lowers");
    assert!(
        has(&p, |i| matches!(i, Instr::CallVirt { slot: s, .. } if *s as usize == slot)),
        "MoveNext must dispatch through interface slot {slot}"
    );
}

#[test]
fn interface_member_through_base_interface_uses_the_declaring_slot() {
    // `IAsyncEnum : IDisposable` inherits `Dispose` (the
    // `IAsyncEnumerator.DisposeAsync` shape). The member is declared on
    // the base, so the slot comes from there; the call must still lower
    // through the shared interface range.
    let src = r#"namespace t;
interface IDisposable {
    void Dispose();
}
interface IAsyncEnum : IDisposable {
    int Next();
}
class C {
    int main() {
        IAsyncEnum e = null;
        e.Dispose();
        return 0;
    }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    let table = crate::native::semantic::analyze(&unit).unwrap();
    let slot = crate::native::semantic::iface_slot_by_name(&table, "IAsyncEnum", "Dispose")
        .expect("an inherited interface member has a slot");
    let p = crate::native::ir::lower(&unit).expect("inherited interface call lowers");
    assert!(
        has(&p, |i| matches!(i, Instr::CallVirt { slot: s, .. } if *s as usize == slot)),
        "Dispose must dispatch through interface slot {slot}"
    );
}

#[test]
fn interface_member_with_an_implementer_dispatches_virtually() {
    // The real case the fallback was standing in for: a class implements
    // the interface, so a call through the interface-typed receiver has
    // an actual row to dispatch through.
    let p = lower(
        r#"namespace t;
interface IShape {
    int Area();
}
class Square : IShape {
    public int Area() { return 4; }
}
class C {
    int main() {
        IShape s = new Square();
        return s.Area();
    }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallVirt { .. })),
        "expected virtual dispatch to the implementing class"
    );
}

#[test]
fn field_held_delegate_invocation_dispatches() {
    // `m_action()` where `m_action` is an `Action` field used to be a
    // diagnostic; the invoke body is now shared with the local path, so
    // the call lowers to a function-address call (single or combined).
    let p = lower(
        r#"namespace t;
class A {
    System.Action _cb;
    public A() { }
    public void Run() {
        _cb();
    }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallFnAddr { .. })),
        "field delegate invocation must dispatch through the delegate"
    );
}

#[test]
fn nested_plain_call_reaches_outer_static() {
    // Unqualified calls from nested code resolve outward: an outer
    // `private static` lowers as a direct static call (instance
    // members are never candidates without a receiver).
    let p = lower(
        r#"namespace t;
class Outer {
    private int _fld;
    private static bool Helper(int x) { return true; }
    class Inner {
        internal Inner(Outer o) {
            bool b = Helper(o._fld);
        }
    }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(
            &p,
            |i| matches!(i, Instr::CallStatic { args, .. } if args.len() == 1)
        ),
        "expected direct static call into enclosing scope"
    );
}

#[test]
fn same_named_nested_resolves_via_enclosing_method() {
    // Two `Inner`s nested in different outers declare different
    // `Thing` properties: implicit-`this` member typing follows the
    // entry matching the method under lowering (here by ctor param
    // types), not first-match — even when the other entry comes
    // first in declaration order.
    let p = lower(
        r#"namespace t;
class BoxA {
    public int Count;
}
class BoxB {
}
class Outer2 {
    class Inner {
        public BoxB Thing { get; }
        public Inner(BoxB b) { int x = 0; }
    }
}
class Outer1 {
    class Inner {
        public BoxA Thing { get; }
        public Inner(BoxA b) { int x = Thing.Count; if (x != 0) { } }
    }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::LoadField { .. })),
        "expected member load through disambiguated entry"
    );
}

#[test]
fn interface_call_reaches_object_member() {
    // Every interface-typed value is an object: `System.Object`
    // members resolve on interfaces too (`e.GetType()` on an
    // `IEnumerable`).
    let p = lower(
        r#"namespace t;
class Object {
    public string Talk() { return "o"; }
}
interface IEnumerable {
    bool MoveNext();
}
class C {
    int main() {
        IEnumerable e = null;
        string s = e.Talk();
        return 0;
    }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallVirt { .. })),
        "expected virtual dispatch to object member via interface"
    );
}

#[test]
fn field_initializer_reads_constructed_statics() {
    // Field initializers evaluate in the CONSTRUCTED class's context:
    // `_currentElement = s_boxedFalse` reads the constructed nested
    // class's own static, never the caller's locals.
    let p = lower(
        r#"namespace t;
class Outer {
    class Inner {
        private static int _k = 41;
        int _v = _k;
        public Inner() { }
    }
    int main() {
        Inner i = new Inner();
        return 0;
    }
}"#,
    );
    assert!(
        has(&p, |i| matches!(
            i,
            Instr::LoadField { .. } | Instr::StoreField { .. }
        )),
        "expected field init stores to lower"
    );
}

#[test]
fn static_spelled_instance_call_placeholders() {
    // Static-style call matching only instance overloads (upstream's
    // `ResourceManager.GetString(key, null)` shape — receiver elision
    // unmodeled): evaluates args and yields zero instead of erroring.
    let p = lower(
        r#"namespace t;
class RM {
    public string GetString(string name) { return name; }
    public string GetString(string name, object culture) { return name; }
}
class SR {
    public static string Lookup(string key) {
        string s = RM.GetString(key, null);
        return s;
    }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::Const { imm: 0, .. })),
        "expected zero placeholder for static-spelled instance call"
    );
}

#[test]
fn assign_to_undeclared_implicitly_declares() {
    // Assignment to a name declared nowhere (upstream's `SR`
    // backout `s_resourceManager = null`, whose declaration lives
    // outside our model) implicitly declares a method-scoped
    // zero-initialized slot instead of erroring; later reads
    // resolve consistently.
    let p = lower(
        r#"namespace t;
class C {
    int main() {
        try {
            throw new System.Exception();
        } catch {
            undeclared_global = null;
        }
        return 0;
    }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::Const { imm: 0, .. })),
        "expected implicit declaration to lower"
    );
}

#[test]
fn static_interface_call_placeholders() {
    // Static interface members (`IRuntimeMethodInfo.GetValue`)
    // generate no functions: the call yields zero instead of
    // erroring as a static-through-type misuse.
    let p = lower(
        r#"namespace t;
interface IRWI {
    static int GetVal(IRWI m);
    int Instance();
}
class C {
    int main() {
        int x = IRWI.GetVal(null);
        return 0;
    }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::Const { imm: 0, .. })),
        "expected zero placeholder for static interface call"
    );
}

#[test]
fn visible_sibling_wins_over_private_first_match() {
    // Same-named generic-arity siblings (`GCHandle` with a private
    // 1-param ctor first, `GCHandle<T>` with a public 1-param ctor):
    // a visible applicable overload in another entry wins over an
    // inapplicable-or-inaccessible first match instead of erroring.
    let p = lower(
        r#"namespace t;
class GCHandle {
    private GCHandle(int h) { }
}
class GCHandle<T> {
    public GCHandle(object o) { }
}
class GC {
    public static void M(object o) {
        GCHandle h = new GCHandle(o);
    }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallFn { .. })),
        "expected visible sibling ctor call"
    );
}

#[test]
fn qualified_receiver_uses_exact_entry() {
    // `var e = new Interop.ErrorInfo(..)` types dotted; member
    // calls resolve through the qualified entry, not a simple-name
    // first match (which misses when unrelated same-named types
    // exist — the `Interop.ErrorInfo.GetErrorMessage` shape).
    let p = lower(
        r#"internal static partial class Interop {
    internal struct ErrorInfo {
        internal string GetErrorMessage() { return "x"; }
    }
}
class Other {
    class ErrorInfo {
        public int nope;
    }
}
class P {
    int main() {
        var e = new Interop.ErrorInfo(5);
        string s = e.GetErrorMessage();
        return 0;
    }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallVirt { .. })),
        "expected virtual dispatch on qualified receiver"
    );
}

#[test]
fn explicit_property_receiver_types_as_declared() {
    // Bare property names in value position type as the DECLARED
    // type (`TypeHandle` property, not the same-named struct), so
    // member calls on them resolve (`TypeHandle.Verify...` shape).
    let p = lower(
        r#"namespace t;
class H {
    public int Goo() { return 1; }
}
class C {
    public H Thing { get { return null; } }
    public void M() {
        int x = Thing.Goo();
    }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallVirt { .. })),
        "expected virtual dispatch through property type"
    );
}

#[test]
fn aliased_static_call_dispatches_to_declaring_function() {
    // `using M = MT; M.Go(5)` lowers to the `MT::Go` function, not a
    // lookup under the alias spelling (the `MdToken.IsNullToken`
    // shape).
    let p = lower(
        r#"namespace t;
using M = MT;
struct MT {
    public static bool Go(int t) { return true; }
}
class P {
    int main() {
        bool b = M.Go(5);
        return 0;
    }
}"#,
    );
    assert!(
        has(
            &p,
            |i| matches!(i, Instr::CallStatic { args, .. } if args.len() == 1)
        ),
        "expected static call through alias"
    );
}

#[test]
fn binary_result_types_select_compatible_overload() {
    // `x - y` (int) must not resolve as `object`-wildcard against a
    // `string` overload: the `public M(long)` overload wins over the
    // inapplicable `private M(string)` (which would deny access).
    let p = lower(
        r#"namespace t;
class C {
    private static int M(string a) { return 0; }
    public static int M(long a) { return 1; }
    int main() {
        int x = 1;
        int y = 2;
        return M(x - y);
    }
}"#,
    );
    assert!(
        has(
            &p,
            |i| matches!(i, Instr::CallFn { args, .. } if args.len() == 2)
        ),
        "expected arithmetically-typed overload call"
    );
}

#[test]
fn unknown_side_defers_to_known_numeric() {
    // `d - 2` with unknown `d` types as `int` (not `object`), so the
    // `public M(long)` overload wins over the inapplicable `private
    // M(string)` (span-index arithmetic shape).
    let p = lower(
        r#"namespace t;
class C {
    private static int M(string a) { return 0; }
    public static int M(long a) { return 1; }
    int main() {
        dynamic d = null;
        return M(d - 2);
    }
}"#,
    );
    assert!(
        has(
            &p,
            |i| matches!(i, Instr::CallFn { args, .. } if args.len() == 2)
        ),
        "expected deferred-numeric overload call"
    );
}

#[test]
fn member_position_property_writes_drop() {
    // Writes (plain and compound) through an explicit accessor-backed
    // property (`buf.Low64 = x`, `buf.Low64 += x`) evaluate for side
    // effects and drop — no setter function exists yet.
    let p = lower(
        r#"namespace t;
class Buf {
    public int Low { get { return 1; } set { } }
}
class C {
    int main() {
        Buf b = new Buf();
        b.Low = 5;
        b.Low += 2;
        return 0;
    }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallFn { .. })),
        "expected member property writes to lower"
    );
}

#[test]
fn enum_values_dispatch_object_members() {
    // Every enum value is an object: `Kind.A.ToString()` resolves
    // through inherited `System.Object` members.
    let p = lower(
        r#"namespace t;
class Object {
    public string ToString() { return "o"; }
}
enum Kind {
    A,
    B,
}
class C {
    int main() {
        Kind k = Kind.A;
        string s = k.ToString();
        return 0;
    }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallVirt { .. })),
        "expected virtual dispatch on enum value"
    );
}

#[test]
fn derived_hide_beats_inherited_private() {
    // Name hiding: a derived `private static Msg()` shadows the base's
    // identical-signature private — lookup must keep both copies and
    // prefer the derived one (the base copy is inaccessible and must
    // not shadow). The `GetDefaultMessage` shape.
    let p = lower(
        r#"namespace t;
class Base {
    private static string Msg() { return "b"; }
    public Base(string s) { }
}
class Derived : Base {
    private static string Msg() { return "d"; }
    public Derived() : base(Msg()) { }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallStatic { .. } | Instr::CallFn { .. })),
        "expected hidden derived call to lower"
    );
}

#[test]
fn nested_accesses_outer_protected() {
    // Nested types access enclosing protected members (the
    // `CompatDerivedImpl` → `Random.Sample` shape): same nesting
    // family allows, without derivation.
    let p = lower(
        r#"namespace t;
class Random {
    protected int Sample() { return 1; }
    class Inner {
        public int Next(Random r) {
            return r.Sample();
        }
    }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallVirt { .. })),
        "expected virtual dispatch through nesting family"
    );
}

#[test]
fn explicit_cast_types_as_target() {
    // `(nuint)offset` types as `nuint` (not `object`), so overloads
    // against `nuint` parameters stay applicable (the `StoreUnsafe`
    // shape).
    let p = lower(
        r#"namespace t;
class C {
    private static int M(string a) { return 0; }
    public static int M(long a) { return 1; }
    int main() {
        int offset = 5;
        return M((long)offset);
    }
}"#,
    );
    assert!(
        has(
            &p,
            |i| matches!(i, Instr::CallFn { args, .. } if args.len() == 2)
        ),
        "expected cast-typed overload call"
    );
}

#[test]
fn nested_reads_outer_static_field() {
    // Unqualified reads from nested code resolve outward to statics
    // (`s_utcTimeZone` read from nested `CachedData`).
    let p = lower(
        r#"namespace t;
class Outer {
    private static int _k = 41;
    class Inner {
        public int M() {
            return _k;
        }
    }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::LoadField { .. })),
        "expected outer static field load"
    );
}

#[test]
fn overwide_methods_skip_function_generation() {
    // Methods beyond 32 params (theoretical 64-element vector
    // factories) generate no function — the frame budget (`add rsp,
    // imm8`) cannot address them — instead of aborting the unit.
    // 16-element vector factories lower normally.
    let p = lower(
        r#"namespace t;
class Wide {
    public static int Slim(
        int p00, int p01, int p02, int p03,
        int p04, int p05, int p06, int p07,
        int p08, int p09, int p10, int p11,
        int p12, int p13, int p14, int p15) { return p00; }
    public static int TooWide(
        int p00, int p01, int p02, int p03,
        int p04, int p05, int p06, int p07,
        int p08, int p09, int p10, int p11,
        int p12, int p13, int p14, int p15,
        int p16, int p17, int p18, int p19,
        int p20, int p21, int p22, int p23,
        int p24, int p25, int p26, int p27,
        int p28, int p29, int p30, int p31,
        int p32) { return p00; }
    public static int Small(int x) { return x; }
    int main() { return Slim(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallFn { args, .. } if args.len() == 17)),
        "expected 16-wide call (plus receiver) to lower while >32 is skipped"
    );
}

#[test]
fn qualified_type_reference_resolves_by_suffix() {
    // `M3.Impl` (as written in a signature) resolves to
    // `Nums.M3.Impl` by suffix — not first-match `M4.Impl` — so
    // member access uses the right entry's members (the
    // `Matrix3x2.Impl` vs `Matrix4x4.Impl` shape).
    let p = lower(
        r#"namespace Nums;
class Box {
    public int Goo() { return 1; }
}
class M4 {
    struct Impl {
        public int X;
    }
}
class M3 {
    struct Impl {
        public Box X;
    }
    public static int M(M3.Impl m) {
        return m.X.Goo();
    }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallVirt { .. })),
        "expected virtual dispatch through suffix-resolved entry"
    );
}

#[test]
fn typed_reference_keywords_placeholder() {
    // Undocumented `__reftype`/`__makeref`/`__refvalue` keywords lower
    // to zero after evaluating args (mirrors the scan's known-list).
    let p = lower(
        r#"namespace t;
class C {
    int main() {
        object t = null;
        object r = __reftype(t);
        return 0;
    }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::Const { imm: 0, .. })),
        "expected zero placeholder for typed-reference keyword"
    );
}

#[test]
fn params_expanded_call_selects_overload() {
    // `M(3, E.A, E.B)` against `M(int, params E[])`: the trailing
    // args score against the element type (cf. MetadataSizes'
    // `IsReferenceSmall(3, TableIndex.MethodDef, ...)`).
    let p = lower(
        r#"namespace t;
enum E { A, B }
class C {
    bool Small(int tag, params E[] tables) { return true; }
    bool Go() { return Small(3, E.A, E.B); }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallFn { .. } | Instr::CallVirt { .. })),
        "params expanded call must select the overload"
    );
}

#[test]
fn params_empty_trailing_call_applies() {
    // Zero trailing args is legal (`M(3)` binds an empty array).
    let p = lower(
        r#"namespace t;
enum E { A, B }
class C {
    bool Small(int tag, params E[] tables) { return true; }
    bool Go() { return Small(3); }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallFn { .. } | Instr::CallVirt { .. })),
        "empty params call must apply"
    );
}

#[test]
fn params_nullable_array_expanded_call_applies() {
    // `M(m, a0..a4)` against `M(MI, params E[]?)`: the nullable
    // marker must not defeat element-type scoring (cf.
    // `Expression.Call(MethodInfo, ...)` overloads).
    let p = lower(
        r#"namespace t;
class MI { }
class E { }
static class X {
    public static int Call(MI m, params E[]? args) { return 1; }
}
class P {
    int Go(MI m, E a, E b, E c, E d, E e) { return X.Call(m, a, b, c, d, e); }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallStatic { .. })),
        "nullable-array params must score trailing args by element"
    );
}

#[test]
fn legacy_fallback_prefers_fewer_hard_mismatches() {
    // Same-arity legacy fallback ranks by mismatch awareness, not
    // declaration order: an unknown-name (`TVector`) arg wildcards
    // while an exact second arg selects — electing the vectorized
    // overload over the scalar one. Chained `.AsV4()` (V128-only
    // extension) proves the pick: scalar would leave a V4 receiver
    // with no applicable overload (cf. the real `Transform` wall).
    let p = lower(
        r#"namespace t;
class V128 { }
class V4 { }
class Quat { }
static class X {
    public static V4 Transform(V4 v, Quat r) { return v; }
    internal static V128 Transform(V128 v, V128 r) { return v; }
    public static V4 AsV4(this V128 v) { return default; }
}
class P {
    V4 Go(TVector t, V128 b) { return X.Transform(t, b).AsV4(); }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallStatic { .. })),
        "vectorized overload must win legacy ranking"
    );
}

#[test]
fn mixed_integer_binary_widens() {
    // `ulong + uint` types as `ulong` regardless of keyword vs
    // framework spelling (cf. `Prime64_1 + (length << 2)` in
    // XxHash128): the widening branch must see canonical names.
    let p = lower(
        r#"namespace t;
class C {
    static void Take(ulong x) { }
    static void Go(ulong a, uint b) { C.Take(a + b); }
    static void Go2(System.UInt64 a, uint b) { C.Take(a + b); }
    int main() { return 0; }
}"#,
    );
    assert!(
        has(&p, |i| matches!(i, Instr::CallStatic { .. })),
        "ulong+uint must rank the ulong overload"
    );
}
