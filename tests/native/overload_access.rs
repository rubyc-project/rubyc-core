//! Overload access filtering (C# candidate-set rule).
//!
//! Inaccessible overloads are removed BEFORE ranking: a private
//! `Equals(Holder)` must not outrank public `Equals(object)` for an
//! outside caller (`MetadataImport.Equals(MetadataImport)` vs
//! `Equals(object)` in CoreLib). When nothing is accessible, the full
//! set still ranks so the access check reports on the right member.

use crate::native::ir::Program;

fn lower(src: &str) -> Result<Program, String> {
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).map_err(|e| format!("{e:?}"))
}

#[test]
fn private_exact_overload_does_not_shadow_public_applicable() {
    // `h.Equals(h)` from outside `Holder`: the private exact-type
    // overload is not a candidate; the public `Equals(object)` wins
    // and lowering succeeds (no E0017).
    let r = lower(
        r#"namespace t;
class Holder {
    public bool Equals(object o) { return true; }
    private bool Equals(Holder h) { return true; }
}
class User {
    bool M(Holder h) { return h.Equals(h); }
    int main() { return 0; }
}"#,
    );
    assert!(r.is_ok(), "public Equals(object) should win: {r:?}");
}

#[test]
fn genuinely_private_call_still_errors() {    // Only a private overload exists: ranking falls back to the full
    // set and the access check stays loud.
    let r = lower(
        r#"namespace t;
class Holder {
    private bool Wink(Holder h) { return true; }
}
class User {
    bool M(Holder h) { return h.Wink(h); }
    int main() { return 0; }
}"#,
    );
    assert!(r.is_err(), "private-only call should fail access");
}

#[test]
fn internal_member_visible_across_classes_in_unit() {
    // One compilation unit is one assembly: `internal` members are
    // accessible from any class (`ModifiedType.Create` from
    // `RuntimeParameterInfo` in CoreLib).
    let r = lower(
        r#"namespace t;
class Maker {
    internal static int Create(int x, int y = 0) { return x; }
}
class User {
    int M() { return Maker.Create(1, parameterIndex: 2); }
    int main() { return 0; }
}"#,
    );
    assert!(r.is_ok(), "internal cross-class call should pass: {r:?}");
}

#[test]
fn protected_override_callable_via_base_slot() {
    // `JsonNode.ReplaceWith` calling `jsonArray.SetItem`: a protected
    // override of a protected virtual declared on the CALLER's own
    // class. Access checks against the base slot (visible here);
    // dispatch stays virtual so the override still runs.
    let r = lower(
        r#"namespace t;
class JsonNode {
    protected virtual void SetItem(int i, JsonNode n) { }
    public void ReplaceWith(JsonArray a) { a.SetItem(0, null); }
    int main() { return 0; }
}
class JsonArray : JsonNode {
    protected override void SetItem(int i, JsonNode n) { }
}"#,
    );
    assert!(r.is_ok(), "base-slot call should pass: {r:?}");
}

#[test]
fn protected_override_without_base_slot_still_errors() {
    // Same shape but the base declares nothing: no base slot redeems
    // the call, so the access check stays loud.
    let r = lower(
        r#"namespace t;
class Base { int main() { return 0; } }
class Derived : Base {
    protected void Hide(int i) { }
}
class User {
    int M(Derived d) { d.Hide(0); return 0; }
    int main() { return 0; }
}"#,
    );
    assert!(r.is_err(), "slot-less protected call should fail access");
}

#[test]
fn protected_field_accessible_from_nested_class() {
    // `Http2Stream` nested in `Http2Connection : HttpConnectionBase`
    // reads the protected `_pool` through an outer-typed reference:
    // nested classes share their enclosing types' access rights (was:
    // E0018 "'HttpConnectionBase._pool' is protected ...
    // [in Http2Stream.Http2Stream]").
    let r = lower(
        r#"namespace t;
class Base {
    protected int _pool;
}
class Outer : Base {
    public Outer() { _pool = 1; }
    class Inner {
        int M(Outer o) { return o._pool; }
    }
}
class C {
    int Main() { return 0; }
}"#,
    );
    assert!(r.is_ok(), "nested protected access should lower: {r:?}");
}

#[test]
fn access_check_prefers_matching_arity() {
    // `WeakReference` (arity 0, public `Target`) vs `WeakReference<T>`
    // (arity 1, private `Target`): access through the arg-less spelling
    // must resolve against arity 0 (was: E0017 "'WeakReference.Target'
    // is private ... [in TypeDescriptor.CreateAssociation]"). The
    // generic entry is declared FIRST so legacy first-match hits it.
    let r = lower(
        r#"namespace t;
class WeakRef<T> {
    private T? Target { get { return default; } }
}
class WeakRef {
    public object? Target { get { return null; } }
}
class C {
    object M(WeakRef w) { return w.Target; }
    int Main() { return 0; }
}"#,
    );
    assert!(r.is_ok(), "arity-0 access should lower: {r:?}");
}

#[test]
fn event_invoke_ignores_instance_property_decoy() {
    // `Changed?.Invoke(sender, e)` where `Changed` is an event: the
    // `using static` bare-name fallback must only consider STATIC
    // members — an unrelated instance property with the same name
    // (`ZipArchive.Changed`, type `ChangeState`) must not uniquely
    // resolve the receiver (was: "'ChangeState' has no member
    // 'Invoke' [in ComposablePartCatalogCollection.OnChanged]").
    // Event invocations lower as silent placeholders (established
    // behavior); only the mistype is fixed here.
    let r = lower(
        r#"namespace t;
class State { }
class Holder {
    public State Changed { get; set; }
}
delegate void Handler(object sender, object e);
class C {
    public event Handler Changed;
    void M(object sender, object e) { Changed?.Invoke(sender, e); }
    int Main() { return 0; }
}"#,
    );
    assert!(r.is_ok(), "event invoke past instance decoy should lower: {r:?}");
}

#[test]
fn explicit_type_args_resolve_bare_call() {
    // `GetOrCreateCacheEntry<BoxCache>().Box(...)`: the parser must
    // keep `<T>` on a bare-call callee (it was dropped, so no
    // substitution ran and the result stayed `T`), and lowering must
    // match overload `M` by the bare name. Asserts a real call is
    // emitted (not a silent zero placeholder, which masked the drop).
    use crate::native::ir::Instr;
    let p = lower(
        r#"namespace t;
class BoxCache { public object Box(object o) { return null; } }
class RuntimeType {
    internal T GetOrCreateCacheEntry<T>() where T : class { return null; }
    internal object DoBox() { return GetOrCreateCacheEntry<BoxCache>().Box(null); }
    int Main() { return 0; }
}"#,
    )
    .expect("generic bare call should resolve");
    let any_call = p.functions.iter().flat_map(|f| f.blocks.iter()).flat_map(|b| b.instrs.iter()).any(
        |i| matches!(
            i,
            Instr::CallVirt { .. } | Instr::CallStatic { .. } | Instr::CallFn { .. }
        ),
    );
    assert!(any_call, "expected a real call instruction");
}

#[test]
fn class_type_param_shadows_alias_in_member_call() {
    // `using T = XmlQueryTypeFactory;` (file-scoped alias) must not
    // hijack the class type parameter `T`: member calls on `T`-typed
    // values take the unknown-type path, not the alias target's
    // members (`MemberInfoCache<T>.MergeWithGlobalList` calling
    // `newMemberInfo.CacheEquals(...)` failed with "'T' has no
    // member 'Box'" style errors once the alias resolved).
    // Nested form mirrors the real shape (cache class nested in its
    // owner); both must degrade, not resolve via the alias.
    for src in [
        r#"namespace t;
using T = System.Xml.Xsl.XmlQueryTypeFactory;
class XmlQueryTypeFactory { }
class Cache<T> {
    int M(T item) { return item.CacheEquals(item); }
    int Main() { return 0; }
}"#,
        r#"namespace t;
using T = System.Xml.Xsl.XmlQueryTypeFactory;
class XmlQueryTypeFactory { }
class RuntimeType {
    private sealed class MemberInfoCache<T> where T : MemberInfo {
        int Merge(T a, T b) { return a.CacheEquals(b); }
    }
    int Main() { return 0; }
}
class MemberInfo { }"#,
    ] {
        let r = lower(src);
        assert!(r.is_ok(), "type variable must not resolve via alias: {r:?} for:\n{src}");
    }
}

#[test]
fn method_type_var_static_call_degrades() {
    // `T.IsZero(value)` (static abstract interface member, generic
    // math): a static call through a METHOD type parameter cannot
    // resolve and must degrade, not error (`ArgumentOutOfRangeException
    // .ThrowIfZero<T>` failed with "'T' has no member 'IsZero'").
    // The alias forces the issue: without the guard, `T` resolves
    // through it and the missing member errors.
    let r = lower(
        r#"namespace t;
using T = Ns.Foo;
class Foo { }
class Thrower {
    public static void ThrowIfZero<T>(T value) { if (T.IsZero(value)) { return; } }
    int Main() { return 0; }
}"#,
    );
    assert!(r.is_ok(), "static call through method type var must degrade: {r:?}");
}

#[test]
fn nullable_type_var_receiver_degrades() {
    // `Unsafe.Add(ref first, i)?.CompareTo(lookUp)`: the `?.` receiver
    // is a generic `ref` return — still an unbound variable, must
    // degrade (`SpanHelpers.SequenceCompareTo<T>` failed with "'T'
    // has no member 'CompareTo'"). The alias forces the issue: it
    // would otherwise resolve `T` to a concrete type missing the
    // member.
    let r = lower(
        r#"namespace t;
using T = Ns.Foo;
class Foo { }
static class Unsafe {
    public static ref T Add<T>(ref T source, int elementOffset) where T : allows ref struct { return ref source; }
    public static ref T Add<T>(ref T source, nint elementOffset) where T : allows ref struct { return ref source; }
}
class C {
    public static int M<T>(ref T first, T other) { return Unsafe.Add(ref first, 0)?.CompareTo(other) ?? -1; }
    int Main() { return 0; }
}"#,
    );
    assert!(r.is_ok(), "generic ref receiver must degrade: {r:?}");
}

#[test]
fn static_ctor_does_not_suppress_implicit_instance_ctor() {
    // `new TypeScope()` where the class declares only `static
    // TypeScope()`: C# still provides the implicit public
    // parameterless instance ctor (static ctors are never callable
    // via `new`). Regression: the static ctor matched instance-ctor
    // selection and failed access ("'TypeScope.TypeScope' is
    // private...").
    let r = lower(
        r#"namespace t;
class TypeScope {
    static TypeScope() { }
}
class User {
    TypeScope M() { return new TypeScope(); }
    int Main() { return 0; }
}"#,
    );
    assert!(r.is_ok(), "implicit instance ctor should win: {r:?}");
}

#[test]
fn self_assigning_declaration_binds_first() {
    // `MethodInfo? getMethod = getMethod = type.GetMethod(...)`: the
    // initializer assigns to the very name being declared (redundant
    // upstream, but legal). The name binds (zero) first so the
    // assignment target resolves (was: "use of undeclared variable
    // 'getMethod'").
    let r = lower(
        r#"namespace t;
class C {
    object M(object t) { object getMethod = getMethod = t; return getMethod; }
    int Main() { return 0; }
}"#,
    );
    assert!(r.is_ok(), "self-assigning declaration should lower: {r:?}");
}

#[test]
fn new_retries_samenamed_entries() {
    // `new Compiler()` where first-match `A.Compiler` has only a
    // 3-arg ctor but `B.Compiler` has the implicit zero-arg one:
    // construction retries same-named entries and adopts the first
    // that accepts (XslTransform with `using XsltOld`).
    let r = lower(
        r#"namespace A { class Compiler { public Compiler(int x) { } } }
namespace B { class Compiler { } }
class User {
    object M() { return new Compiler(); }
    int Main() { return 0; }
}"#,
    );
    assert!(r.is_ok(), "fallback entry should construct: {r:?}");
}

#[test]
fn new_unconstructible_still_errors() {
    // No same-named entry accepts the args: still a loud arity error,
    // not a mis-compile against an unrelated entry.
    let r = lower(
        r#"namespace A { class Compiler { public Compiler(int x) { } } }
namespace B { class Compiler { public Compiler(string s) { } } }
class User {
    object M() { return new Compiler(); }
    int Main() { return 0; }
}"#,
    );
    assert!(r.is_err(), "unconstructible call should fail");
}

#[test]
fn derived_field_shadows_base_field() {
    // `InvokeGenerator._fac` (XsltQilFactory) hides `QilCloneVisitor._fac`
    // (QilFactory): member calls resolve against the DERIVED field's
    // type (was: base field won, so `DefaultValueMarker` missed).
    let r = lower(
        r#"namespace t;
class QilFactory { }
class XsltQilFactory : QilFactory {
    public object DefaultValueMarker() { return null; }
}
class QilCloneVisitor {
    private readonly QilFactory _fac;
    public QilCloneVisitor(QilFactory f) { _fac = f; }
}
class InvokeGenerator : QilCloneVisitor {
    private readonly XsltQilFactory _fac;
    public InvokeGenerator(XsltQilFactory f) : base(f) { _fac = f; }
    public object GenerateInvoke() { return _fac.DefaultValueMarker(); }
    int Main() { return 0; }
}"#,
    );
    assert!(r.is_ok(), "derived field should shadow base: {r:?}");
}

#[test]
fn implicit_conversion_applies_to_overload() {
    // `ReportError(res, name, parent)` with `parent: DelayedQName`
    // into `(string, params string?[]?)`: the struct's `implicit
    // operator string` applies (System.Private.Xml `DelayedQName` →
    // `ReportError`). Regression: overload scoring knew no
    // user-defined conversions ("unknown call 'ReportError'").
    // Three args vs two params forces the params-expanded tier (an
    // exact-arity test would pass via legacy arity fallback).
    let r = lower(
        r#"namespace t;
struct DelayedQName {
    public static implicit operator string(DelayedQName qn) { return ""; }
}
class C {
    void ReportError(string res, params string?[]? args) { }
    void M(string res, string name, DelayedQName parent) { ReportError(res, name, parent); }
    int Main() { return 0; }
}"#,
    );
    assert!(r.is_ok(), "implicit conversion should apply: {r:?}");
}

#[test]
fn resx_generated_sr_types_params_call() {
    // `CheckError(c, SR.Key, a)` with `SR.Key: string` (resx-generated
    // `namespace System { partial class SR }`, System.Private.Xml):
    // string-typed args select the `(bool, string, params string[])`
    // overload. When SR was missing, everything typed `object` and the
    // `bool` slot rejected it ("unknown call 'CheckError'").
    let r = lower(
        r#"namespace System {
internal static partial class SR {
    public static string Key => "k.";
}
}
namespace t {
class C {
    void CheckError(bool cond, string res, params string?[]? args) { }
    void M(bool c, string a) { CheckError(c, System.SR.Key, a); }
    int Main() { return 0; }
}}"#,
    );
    assert!(r.is_ok(), "typed SR props should resolve the call: {r:?}");
}

#[test]
fn negated_call_types_as_bool() {
    // `CheckError(!_input.AttributeExists(1, "name"), ...)` (Xml
    // serialization): logical negation is always `bool`. Without it
    // the arg typed `object`, the `bool` slot rejected it, and the
    // overload mis-resolved ("unknown call 'CheckError'").
    let r = lower(
        r#"namespace t;
class Inp {
    public bool AttributeExists(int n, string s) { return true; }
}
class C {
    private Inp _input;
    void CheckError(bool cond, string res, params string?[]? args) { }
    void M() { CheckError(!_input.AttributeExists(1, "name"), "key"); }
    int Main() { return 0; }
}"#,
    );
    assert!(r.is_ok(), "negated call should type as bool: {r:?}");
}

#[test]
fn delegate_member_resolves_via_system_delegate() {
    // `continuation.GetTargetForSingleCastInstanceDelegate()` on an
    // `Action`-typed value: delegate declarations carry no members,
    // so lookup goes through the `System.Delegate` base (was: "'Action'
    // has no member 'GetTargetForSingleCastInstanceDelegate'").
    // Asserts a real virtual call is emitted, not a silent zero
    // placeholder (which is what the pre-fix degradation produced).
    use crate::native::ir::Instr;
    let p = lower(
        r#"namespace System {
public abstract class Delegate {
    internal object GetTarget() { return null; }
}
public delegate void Action();
class C {
    object M(Action continuation) { return continuation.GetTarget(); }
    int Main() { return 0; }
}}"#,
    )
    .expect("delegate base member should resolve");
    let callee = p
        .functions
        .iter()
        .find(|f| f.name == "C::M#1")
        .expect("C::M lowered");
    let any_call = callee
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .any(|i| {
            matches!(
                i,
                Instr::CallVirt { .. } | Instr::CallStatic { .. } | Instr::CallFn { .. }
            )
        });
    assert!(any_call, "expected a real call instruction in C::M");
}

#[test]
fn delegate_invoke_stays_placeholder() {
    // `handler.Invoke(...)` on a delegate-typed value is invocation,
    // not member lookup: it keeps the legacy placeholder path even
    // though delegate receivers otherwise resolve via
    // `System.Delegate` (which has no `Invoke` member — erroring
    // there broke `Progress.InvokeHandlers`).
    // The shadowing class variant covers `YieldAwaiter.RunAction`:
    // an unrelated same-named `Action` class must not capture the
    // call either.
    for src in [
        r#"namespace System {
public abstract class Delegate {
    internal object GetTarget() { return null; }
}
public delegate void Handler(int x);
class Progress {
    public void InvokeHandlers(Handler h, int v) { h.Invoke(v); }
    int Main() { return 0; }
}}"#,
        r#"namespace System {
public abstract class Delegate {
    internal object GetTarget() { return null; }
}
public delegate void Action();
}
namespace Other {
class Action {
    public int Unrelated() { return 0; }
}
}
namespace System {
class Progress2 {
    public void RunAction(Action a) { a.Invoke(); }
    int Main() { return 0; }
}}"#,
    ] {
        let r = lower(src);
        assert!(r.is_ok(), "delegate Invoke should stay placeholder: {r:?} for:\n{src}");
    }
}

#[test]
fn delegate_member_skips_unrelated_samenamed_class() {
    // Same shape, plus an unrelated same-named CLASS (`XsltOld.Action`
    // helper): first-match would select it and miss the member. The
    // divert only fires when the class entry has NO same-named method,
    // so the delegate base still wins here.
    use crate::native::ir::Instr;
    let p = lower(
        r#"namespace System {
public abstract class Delegate {
    internal object GetTarget() { return null; }
}
public delegate void Action();
class Helper {
    public int Unrelated() { return 0; }
}
}
namespace XsltOld {
class Action : System.Helper {
}
}
namespace System {
class C {
    object M(Action continuation) { return continuation.GetTarget(); }
    int Main() { return 0; }
}}"#,
    )
    .expect("delegate base member should resolve past the shadow");
    let callee = p
        .functions
        .iter()
        .find(|f| f.name == "C::M#1")
        .expect("C::M lowered");
    let any_call = callee
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .any(|i| {
            matches!(
                i,
                Instr::CallVirt { .. } | Instr::CallStatic { .. } | Instr::CallFn { .. }
            )
        });
    assert!(any_call, "expected a real call instruction in C::M");
}

#[test]
fn base_chain_method_group_arg_placeholders() {
    // `: base(x, cb)` with a bare (lowercase) method name as a delegate
    // argument (`FileSystemSecurity` passing `_HandleErrorCode`):
    // method groups have no value to evaluate — pass zero (was: "use
    // of undeclared variable '_HandleErrorCode'"). Lowercase matters:
    // uppercase-starting unknowns already placeholder via the
    // type-name rule; lowercase ones hit the undeclared error.
    let r = lower(
        r#"namespace t;
class Base {
    public Base(int x, object cb) { }
}
class Derived : Base {
    static object cb(int e) { return null; }
    public Derived() : base(0, cb) { }
    int Main() { return 0; }
}"#,
    );
    assert!(r.is_ok(), "method-group base arg should lower: {r:?}");
}

#[test]
fn regular_call_method_group_arg_placeholders() {
    // `M(cb)` with a bare (lowercase) method name as a delegate
    // argument: same fallback as the base-chain case — method groups
    // eval to zero.
    let r = lower(
        r#"namespace t;
class C {
    static object cb(int e) { return null; }
    void M(object x) { }
    void N() { M(cb); }
    int Main() { return 0; }
}"#,
    );
    assert!(r.is_ok(), "method-group call arg should lower: {r:?}");
}

#[test]
fn base_call_past_same_named_shadow() {
    // A baseless same-named STRUCT first-matches the short-name table
    // lookup, shadowing the derived class under lowering
    // (`System.Xml.Xsl.Runtime.UnionIterator` vs Linq's): `base.M(...)`
    // must type through the exact entry (`current_info`), not the
    // shadow (was: "'Object' has no member 'Contains'
    // [in UnionIterator.Contains]"). The struct sorts first so the
    // legacy lookup hits it.
    let r = lower(
        r#"namespace System {
    public class Object { }
}
namespace t;
struct UnionIterator {
    public int Count;
}
static class Host {
    class Base {
        public virtual bool Contains(int v) { return false; }
    }
    class UnionIterator<T> : Base {
        public override bool Contains(T v) { return base.Contains(0); }
    }
}
class C {
    int Main() { return 0; }
}"#,
    );
    assert!(r.is_ok(), "base call past same-named shadow should lower: {r:?}");
}

#[test]
fn params_span_accepts_single_values() {
    // `DumpHeaders(ref sb, _headers, _content?.Headers)` passes two
    // `HttpHeaders` VALUES to `params ReadOnlySpan<HttpHeaders?>`
    // (.NET 9 single-value span conversion): each expanded trailing
    // arg must be applicable (was: "'HeaderUtilities' has no member
    // 'DumpHeaders' [in HttpRequestMessage.ToString]").
    let r = lower(
        r#"namespace t;
struct ReadOnlySpan<T> { }
class Headers { }
class ReqHeaders : Headers { }
static class U {
    public static void Dump(int sb, params ReadOnlySpan<Headers> h) { }
}
class C {
    void M(ReqHeaders a, ReqHeaders b) { U.Dump(0, a, b); }
    int Main() { return 0; }
}"#,
    );
    assert!(r.is_ok(), "params-span single values should lower: {r:?}");
}

#[test]
fn inherited_property_types_member_chain() {
    // `mapping.TypeDesc.UseReflection` where `TypeDesc` lives on base
    // `Mapping`: member lookup must walk the base chain (was:
    // "'ReflectionAwareCodeGen' has no member 'GetStringForMethodInvoke'"
    // with the 4th arg typed `object` instead of `bool`, rejecting the
    // only overload). A decoy same-named base in another namespace
    // (`System.Runtime.InteropServices.TypeMapping` vs Xml's) must not
    // hijack chain resolution, and a decoy same-named `TypeDesc`
    // (`CompilerServices.TypeDesc`, first in table order) must not win
    // the spelling lookup when only the real one declares the member.
    let r = lower(
        r#"namespace System.Runtime.InteropServices {
    public static class TypeMapping { }
}
namespace System.Runtime.CompilerServices {
    public class TypeDesc {
        public bool IsCollectible;
    }
}
namespace t;
class TypeDesc {
    private int _flags;
    public bool UseReflection { get { return (_flags & 0x4000) != 0; } set { } }
}
class Mapping {
    private TypeDesc? _typeDesc;
    public TypeDesc? TypeDesc { get { return _typeDesc; } set { _typeDesc = value; } }
}
class TypeMapping : Mapping { }
class StructMapping : TypeMapping { }
static class U {
    public static string Get(string a, string b, string c, bool u, params string[] rest) { return a; }
}
class C {
    string M(StructMapping mapping) { return U.Get("o", "t", "m", mapping.TypeDesc.UseReflection); }
    int Main() { return 0; }
}"#,
    );
    assert!(r.is_ok(), "inherited property chain should lower: {r:?}");
}

#[test]
fn ternary_default_takes_other_branch_type() {
    // `SendAsync(buffer, messageType, endOfMessage ? Flags.EndOfMessage
    // : default, cancellationToken)` must select the `ValueTask`
    // overload (3rd arg `Flags`), not legacy-first-match the `Task`
    // overload declared first (was: "'Task' has no member 'AsTask'"
    // in `ManagedWebSocket.SendAsync`). Bare `default` converts to
    // the other branch's type (C# target typing); without this the
    // ternary degrades to `object` and ranking collapses.
    let r = lower(
        r#"namespace t;
enum F { A, B }
struct ArraySegment<T> { }
struct ReadOnlyMemory<T> { }
struct CancellationToken { }
struct ValueTask {
    public object AsTask() { return null; }
}
class Task { }
class Mid {
    public Task SendAsync(ArraySegment<byte> b, int m, bool e, CancellationToken c) { return null; }
    public ValueTask SendAsync(ReadOnlyMemory<byte> b, int m, F f, CancellationToken c) { return default; }
    public object Call(ArraySegment<byte> buffer, int messageType, bool endOfMessage, CancellationToken cancellationToken) {
        return SendAsync(buffer, messageType, endOfMessage ? F.A : default, cancellationToken).AsTask();
    }
    int Main() { return 0; }
}"#,
    );
    assert!(r.is_ok(), "default-typed ternary should select ValueTask overload: {r:?}");
}

#[test]
fn implicit_this_base_property_read_placeholders() {
    // `table` resolving to a base-class property through a derived
    // receiver (`new ConstNode(table!, ...)` in BinaryNode.Optimize,
    // `table` declared on ExpressionNode): properties occupy no
    // storage, so the read placeholders — but must SUCCEED, not
    // error as undeclared (was: "use of undeclared variable 'table'
    // [in BinaryNode.Optimize]").
    let r = lower(
        r#"namespace t;
class T { }
abstract class Base {
    private T? _t;
    protected T? Item
    {
        get { return _t; }
    }
}
class Derived : Base {
    object M() { return Item; }
    int Main() { return 0; }
}"#,
    );
    assert!(r.is_ok(), "base property read should lower: {r:?}");
}
