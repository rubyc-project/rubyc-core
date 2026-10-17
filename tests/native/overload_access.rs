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

#[test]
fn protected_base_property_readable_from_base_through_nullable() {
    // `task.Prop` where `task: Task?` and a derived class overrides
    // the `protected` property: C# binds through the static type
    // (`Task`), whose own declaration the viewer (`Task`) accesses
    // legally — denying on the derived override's account rejects
    // shipped code (`Task.AnyTaskRequiresNotifyDebuggerOfWaitCompletion`
    // broke the world build with E0018). Base-slot redemption allows
    // it; dispatch stays virtual.
    let r = lower(
        r#"namespace t;
class Task {
    protected virtual bool ShouldNotifyDebuggerOfWaitCompletion => true;
    internal static bool Any(Task?[] tasks) {
        foreach (Task? task in tasks) {
            if (task != null && task.ShouldNotifyDebuggerOfWaitCompletion) { return true; }
        }
        return false;
    }
    int Main() { return 0; }
}
class CompleteOnCountdownPromise : Task {
    protected override bool ShouldNotifyDebuggerOfWaitCompletion => false;
}"#,
    );
    assert!(r.is_ok(), "protected base property read should lower: {r:?}");
}



#[test]
fn nullable_ternary_array_coerces_to_span() {
    let p = lower(
        r#"namespace t;
class Span<T> {
    private T _reference;
    private int _length;
}
class C {
    int Main() {
        byte[]? nb = null;
        System.Span<byte> bb = (nb is not null) ? nb : new byte[1024];
        return (int)bb.Length;
    }
}"#,
    )
    .unwrap();
    // span_object emits Alloc{32} + VTableAddr + 2 field stores; a
    //MISS means the ternary result stayed a raw array (no coerce).
    let allocs = p
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| {
            matches!(i, crate::native::ir::Instr::Alloc { bytes: 32, .. })
        })
        .count();
    assert!(allocs > 0, "ternary array->Span must build a span object (no coerce happened?)");
}




#[test]
fn thread_init_interception_builds_hollow_object() {
    // `Thread.InitializeCurrentThread()` (bare call inside `Thread`)
    // must lower to an inline hollow object (Alloc + vtable), never a
    // call into the real body (whose QCalls have no smoke shims).
    let p = lower(
        r#"namespace t;
class Thread {
    private static Thread? t_currentThread;
    public static Thread Current {
        get {
            Thread? t = t_currentThread;
            if (t == null) { t = InitializeCurrentThread(); }
            return t;
        }
    }
    private static Thread InitializeCurrentThread() {
        Thread? thread = null;
        return thread;
    }
    int Main() { Thread t = Thread.Current; return 0; }
}"#,
    )
    .unwrap();
    let get = p
        .functions
        .iter()
        .find(|f| f.name.contains("Main"))
        .expect("caller must lower");
    let instrs: Vec<_> = get
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .collect();
    assert!(
        instrs.iter().any(|i| matches!(
            i,
            crate::native::ir::Instr::Alloc { .. }
        )),
        "must allocate the hollow Thread (no inline happened?)"
    );
    assert!(
        !instrs.iter().any(|i| matches!(
            i,
            crate::native::ir::Instr::CallFn { .. }
                | crate::native::ir::Instr::CallStatic { .. }
                | crate::native::ir::Instr::CallVirt { .. }
        )),
        "getter must not call anywhere (real body would hit missing QCalls)"
    );
}


#[test]
fn starts_with_inlines_no_call() {
    // `StartsWith(value)` on a closed span must inline (scalar
    // compare), never call the shared generic body (whose nullable
    // element gymnastics miscompile and faulted path checks).
    let p = lower(
        r#"namespace t;
class Span<T> {
    private T _reference;
    private int _length;
}
class C {
    int Main() {
        char[] ca = new char[4];
        System.ReadOnlySpan<char> r = ca;
        bool b = r.StartsWith('/');
        return b ? 1 : 0;
    }
}"#,
    )
    .unwrap();
    let bad = p
        .functions
        .iter()
        .filter(|f| !f.name.contains("@start"))
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| {
            matches!(
                i,
                crate::native::ir::Instr::CallVirt { .. }
                    | crate::native::ir::Instr::CallStatic { .. }
                    | crate::native::ir::Instr::CallFn { .. }
            )
        })
        .count();
    assert_eq!(bad, 0, "StartsWith must inline (calls={bad})");
}

#[test]
fn span_field_assign_coerces() {
    // `Span` field assigned from an array must coerce to a view
    // (not store the raw array — reads would see the header).
    let p = lower(
        r#"namespace t;
class Span<T> {
    private T _reference;
    private int _length;
}
class W {
    private System.Span<byte> _chars;
    public void Set(byte[] arr) { _chars = arr; }
    int Main() { return 0; }
}"#,
    )
    .unwrap();
    let spans = p
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(i, crate::native::ir::Instr::Alloc { bytes: 32, .. }))
        .count();
    assert!(spans > 0, "span field assign must build a span object");
}


#[test]
fn log2_fold_fires_no_call() {
    // `BitOperations.Log2` must fold (hardware paths misreport).
    let p = lower(
        r#"namespace t;
static class BitOperations {
    public static int Log2(uint value) { return 0; }
}
class C {
    int Main(uint v) { return BitOperations.Log2(v); }
}"#,
    )
    .unwrap();
    let calls = p
        .functions
        .iter()
        .filter(|f| !f.name.contains("@start"))
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| {
            matches!(
                i,
                crate::native::ir::Instr::CallVirt { .. }
                    | crate::native::ir::Instr::CallStatic { .. }
                    | crate::native::ir::Instr::CallFn { .. }
            )
        })
        .count();
    assert_eq!(calls, 0, "Log2 must fold (calls={calls})");
}

#[test]
fn scratch_log2_somearm_fires() {
    let p = lower(
        r#"namespace t;
static class BitOperations {
    public static int Log2(uint value) { return 0; }
}
class C {
    int Main(uint v) { return BitOperations.Log2(v); }
}"#,
    )
    .unwrap();
    let calls = p
        .functions
        .iter()
        .filter(|f| !f.name.contains("@start"))
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| {
            matches!(
                i,
                crate::native::ir::Instr::CallVirt { .. }
                    | crate::native::ir::Instr::CallStatic { .. }
                    | crate::native::ir::Instr::CallFn { .. }
            )
        })
        .count();
    assert_eq!(calls, 0, "Some-arm Log2 must fold (calls={calls})");
}

#[test]
fn scratch_probe_shape_printf_log2() {
    // Mirror `.probe/copyto/main.rc` exactly (printf + qualified
    // Log2 + `u` literal) with a local 3-overload BitOperations:
    // does the fold fire here?
    let p = lower(
        r#"namespace System.Numerics { static class BitOperations {
    public static int Log2(uint value) { return 0; }
    public static int Log2(ulong value) { return 0; }
    public static int Log2(nuint value) { return 0; }
} } namespace app { class Program {
    int main() { printf("a{0}", System.Numerics.BitOperations.Log2(15u)); return 0; }
} }"#,
    )
    .unwrap();
    let (calls, divs, consts) = (p
        .functions
        .iter()
        .filter(|f| !f.name.contains("@start"))
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| {
            matches!(
                i,
                crate::native::ir::Instr::CallVirt { .. }
                    | crate::native::ir::Instr::CallStatic { .. }
                    | crate::native::ir::Instr::CallFn { .. }
            )
        })
        .count(), p
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(i, crate::native::ir::Instr::Bin { op: crate::native::ir::BinOp::Div, .. }))
        .count(), 0);
    eprintln!("[rubyc] SCRATCH-PROBE calls={calls} divs={divs}");
    assert_eq!(calls, 0, "probe-shape Log2 must fold (calls={calls})");
    assert!(divs > 0, "fold must emit a Div loop (divs={divs})");
    let _ = consts;
}

#[test]
fn scratch_qualified_log2_with_object_entry() {
    // World-faithful repro: with `System.Object` in the table, the
    // qualified receiver degraded to `object` (the `System` namespace
    // ident types as `object`, which then RESOLVES, hiding the dotted
    // path) and the fold missed (`a0` on the VM). Must fold.
    let p = lower(
        r#"namespace System { class Object { } } namespace System.Numerics { static class BitOperations {
    public static int Log2(uint value) { return 0; }
    public static int Log2(ulong value) { return 0; }
    public static int Log2(nuint value) { return 0; }
} } namespace app { class Program {
    int main() { printf("a{0}", System.Numerics.BitOperations.Log2(15u)); return 0; }
} }"#,
    )
    .unwrap();
    let (calls, divs) = (p
        .functions
        .iter()
        .filter(|f| !f.name.contains("@start"))
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| {
            matches!(
                i,
                crate::native::ir::Instr::CallVirt { .. }
                    | crate::native::ir::Instr::CallStatic { .. }
                    | crate::native::ir::Instr::CallFn { .. }
            )
        })
        .count(), p
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(i, crate::native::ir::Instr::Bin { op: crate::native::ir::BinOp::Div, .. }))
        .count());
    eprintln!("[rubyc] SCRATCH-OBJECT calls={calls} divs={divs}");
    assert_eq!(calls, 0, "qualified Log2 with Object entry must fold (calls={calls})");
    assert!(divs > 0, "fold must emit a Div loop (divs={divs})");
}

#[test]
fn log2_qualified_spelling_folds_no_call() {
    // Qualified `System.Numerics.BitOperations.Log2` must fold too:
    // the real probe spells it fully, and the bare-name test above
    // stays green while the qualified path went to a zero placeholder.
    let p = lower(
        r#"namespace System.Numerics { static class BitOperations {
    public static int Log2(uint value) { return 0; }
} } namespace t { class C {
    int Main(uint v) { return System.Numerics.BitOperations.Log2(v); }
} }"#,
    )
    .unwrap();
    let calls = p
        .functions
        .iter()
        .filter(|f| !f.name.contains("@start"))
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| {
            matches!(
                i,
                crate::native::ir::Instr::CallVirt { .. }
                    | crate::native::ir::Instr::CallStatic { .. }
                    | crate::native::ir::Instr::CallFn { .. }
            )
        })
        .count();
    assert_eq!(calls, 0, "qualified Log2 must fold (calls={calls})");
}

#[test]
fn ref_marked_arg_elects_byref_twin_not_self() {
    // `WriteAtOffset` shape: a 3/3 value twin and a 3/3 `ref` twin.
    // A `ref`-marked call must elect the byref twin — electing self
    // recursed forever on identical args (file-I/O stack overflow).
    // Twins return distinct constants so the call edge is observable
    // by target body (overloads share IR names).
    let p = lower(
        r#"namespace t;
class R {
    static int M(int a, int b, long o) { return 111; }
    static int M(int a, int b, ref long o) { return 222; }
    static int Fwd(int a, int b, long o) { return M(a, b, ref o); }
    static int FwdV(int a, int b, long o) { return M(a, b, o); }
    static int Main() { return Fwd(1, 2, 3) + FwdV(1, 2, 3); }
}"#,
    )
    .unwrap();
    let target_consts = |caller: &str| -> Vec<i64> {
        let mut out = Vec::new();
        for f in p.functions.iter().filter(|f| f.name.contains(caller)) {
            for b in f.blocks.iter() {
                for i in b.instrs.iter() {
                    // Same-class statics lower direct (`CallFn`);
                    // cross-class statics go `CallStatic`.
                    let fidx = match i {
                        crate::native::ir::Instr::CallStatic { fidx, .. } => Some(*fidx),
                        crate::native::ir::Instr::CallFn { fidx, .. } => Some(*fidx),
                        _ => None,
                    };
                    if let Some(fidx) = fidx {
                        if let Some(t) = p.functions.get(fidx as usize) {
                            for tb in t.blocks.iter() {
                                for ti in tb.instrs.iter() {
                                    if let crate::native::ir::Instr::Const { imm, .. } = ti {
                                        out.push(*imm);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    };
    let fwd = target_consts("::Fwd#");
    let fwdv = target_consts("::FwdV#");
    assert!(
        fwd.contains(&222) && !fwd.contains(&111),
        "ref call must reach the byref twin (saw {fwd:?})"
    );
    assert!(
        fwdv.contains(&111) && !fwdv.contains(&222),
        "plain call must reach the value twin (saw {fwdv:?})"
    );
}

#[test]
fn ref_marked_memmove_still_folds_no_call() {
    // Guard: `ref`-marked span-copy args must keep hitting the
    // `SpanHelpers.Memmove` fold (MemCpy, no call) — the marker must
    // not strand previously-working shapes on placeholders.
    let p = lower(
        r#"namespace t;
static class SpanHelpers {
    public static void Memmove(ref byte d, ref byte s, int n) { }
}
class C {
    int Main(byte[] a) { SpanHelpers.Memmove(ref a[0], ref a[1], 3); return 0; }
}"#,
    )
    .unwrap();
    let calls = p
        .functions
        .iter()
        .filter(|f| !f.name.contains("@start"))
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| {
            matches!(
                i,
                crate::native::ir::Instr::CallVirt { .. }
                    | crate::native::ir::Instr::CallStatic { .. }
                    | crate::native::ir::Instr::CallFn { .. }
            )
        })
        .count();
    let memcpys = p
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(i, crate::native::ir::Instr::MemCpy { .. }))
        .count();
    assert_eq!(calls, 0, "marked Memmove must fold (calls={calls})");
    assert!(memcpys >= 1, "fold must emit MemCpy");
}

#[test]
fn scratch_private_ref_twin_elects() {
    // `Decimal.Round` shape: public value forwarders + a PRIVATE
    // ref twin. The marked call must elect the private twin (same
    // class ⇒ visible), not miss.
    let p = lower(
        r#"namespace t;
enum Mode { ToEven, Away }
class Decimal {
    public static decimal Round(decimal d) { return Round(ref d, 0, Mode.ToEven); }
    public static decimal Round(decimal d, int decimals) { return Round(ref d, decimals, Mode.ToEven); }
    public static decimal Round(decimal d, Mode mode) { return Round(ref d, 0, mode); }
    public static decimal Round(decimal d, int decimals, Mode mode) { return Round(ref d, decimals, mode); }
    private static decimal Round(ref decimal d, int decimals, Mode mode) { return d; }
    static int Main() { return 0; }
}"#,
    )
    .unwrap();
    let calls = p
        .functions
        .iter()
        .filter(|f| !f.name.contains("@start"))
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| {
            matches!(
                i,
                crate::native::ir::Instr::CallVirt { .. }
                    | crate::native::ir::Instr::CallStatic { .. }
                    | crate::native::ir::Instr::CallFn { .. }
            )
        })
        .count();
    eprintln!("[rubyc] SCRATCH-ROUND calls={calls}");
    assert!(calls >= 1, "forwarder must call the twin");
}

#[test]
fn scratch_real_decimal_round_methods() {
    // Parse the REAL System.Decimal source and list every Round
    // method with its byref flags: is the private ref-twin even in
    // the AST?
    let src = std::fs::read_to_string(
        "../dotnet-runtime/src/libraries/System.Private.CoreLib/src/System/Decimal.cs",
    )
    .or_else(|_| {
        std::fs::read_to_string(
            "dotnet-runtime/src/libraries/System.Private.CoreLib/src/System/Decimal.cs",
        )
    })
    .expect("Decimal.cs readable from test CWD");
    let unit = crate::parser::parse(&src).unwrap();
    let mut found = Vec::new();
    for ns in unit.namespaces.iter() {
        for m in ns.members.iter() {
            if let crate::models::ast::namespace::Member::Class(cls) = m {
                for mm in cls.members.iter() {
                    if let crate::models::ClassMember::Method(md) = mm {
                        if md.name == "Round" {
                            found.push((
                                cls.name.clone(),
                                md.params
                                    .iter()
                                    .map(|p| (p.type_name.clone(), p.is_byref))
                                    .collect::<Vec<_>>(),
                            ));
                        }
                    }
                }
            }
        }
    }
    eprintln!("[rubyc] SCRATCH-DECIMAL rounds={found:?}");
    assert!(
        found.iter().any(|(_, ps)| ps
            .iter()
            .any(|(_, b)| *b)),
        "ref twin must parse with byref flag (saw {found:?})"
    );
}

#[test]
fn scratch_real_decimal_preprocessed() {
    // Full front-end path on the REAL file: preprocess with the
    // CoreLib define set, then parse. Does the twin survive?
    let path = "../dotnet-runtime/src/libraries/System.Private.CoreLib/src/System/Decimal.rc";
    let src = std::fs::read_to_string(path)
        .or_else(|_| {
            std::fs::read_to_string(
                "dotnet-runtime/src/libraries/System.Private.CoreLib/src/System/Decimal.rc",
            )
        })
        .expect("Decimal.rc readable from test CWD");
    let defines = [
        "TARGET_LINUX",
        "TARGET_UNIX",
        "TARGET_64BIT",
        "TARGET_AMD64",
        "TARGET_LITTLE_ENDIAN",
        "CORECLR",
        "NET",
        "FEATURE_PERFTRACING",
        "SYSTEM_PRIVATE_CORELIB",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect::<Vec<_>>();
    let pre = crate::preprocessor::preprocess(&src, &defines, "Decimal.rc").unwrap();
    let expanded = crate::preprocessor::expand_macros(pre.source, &pre.macros);
    let unit = crate::parser::parse(&expanded).unwrap();
    let mut found = Vec::new();
    for ns in unit.namespaces.iter() {
        for m in ns.members.iter() {
            if let crate::models::ast::namespace::Member::Class(cls) = m {
                for mm in cls.members.iter() {
                    if let crate::models::ClassMember::Method(md) = mm {
                        if md.name == "Round" {
                            found.push((
                                cls.name.clone(),
                                md.params
                                    .iter()
                                    .map(|p| (p.type_name.clone(), p.is_byref))
                                    .collect::<Vec<_>>(),
                            ));
                        }
                    }
                }
            }
        }
    }
    eprintln!("[rubyc] SCRATCH-PREPROC rounds={found:?}");
    assert!(
        found.iter().any(|(_, ps)| ps.iter().any(|(_, b)| *b)),
        "preprocessed twin must keep byref flag (saw {found:?})"
    );
}

#[test]
fn scratch_real_decimal_table() {
    // Full parse→table on the REAL Decimal source: does the private
    // ref-twin survive semantic (no preloaded fragments)?
    let path = "../dotnet-runtime/src/libraries/System.Private.CoreLib/src/System/Decimal.rc";
    let src = std::fs::read_to_string(path)
        .or_else(|_| {
            std::fs::read_to_string(
                "dotnet-runtime/src/libraries/System.Private.CoreLib/src/System/Decimal.rc",
            )
        })
        .expect("Decimal.rc readable from test CWD");
    // Stub the interface bases (single-file semantic has no world
    // around it); the methods under test don't touch them. Sources
    // concatenate (duplicate usings are harmless).
    let stubs = r#"namespace System {
interface ISpanFormattable { }
interface IComparable { }
interface IConvertible { }
interface IComparable<T> { }
interface IEquatable<T> { }
interface ISerializable { }
interface IDeserializationCallback { }
interface IFloatingPoint<T> { }
interface IMinMaxValue<T> { }
interface IUtf8SpanFormattable { }
interface ISpanParsable<T> { }
interface IParsable<T> { }
interface IUtf8SpanParsable<T> { }
interface IAdditionOperators<A, B, C> { }
interface IAdditiveIdentity<A, B> { }
interface IComparisonOperators<A, B, C> { }
interface IDecrementOperators<A> { }
interface IDivisionOperators<A, B, C> { }
interface IEqualityOperators<A, B, C> { }
interface IFloatingPointConstants<A> { }
interface IIncrementOperators<A> { }
interface IMinMaxValue2<T> { }
interface IModulusOperators<A, B, C> { }
interface IMultiplicativeIdentity<A, B> { }
interface IMultiplyOperators<A, B, C> { }
interface INumber<A> { }
interface INumberBase<A> { }
interface ISignedNumber<A> { }
interface ISubtractionOperators<A, B, C> { }
interface IUnaryNegationOperators<A, B> { }
interface IUnaryPlusOperators<A, B> { }
}"#;
    // Merge by concatenating sources (usings dupes OK).
    let merged_src = format!("{stubs}\n{src}");
    let unit = crate::parser::parse(&merged_src).unwrap();
    let table = crate::native::semantic::analyze_with_owners(&unit, &Default::default())
        .unwrap_or_else(|errs| panic!("semantic analyzes Decimal.cs: {errs:?}"));
    let mut rounds = Vec::new();
    for c in table.classes.iter() {
        if c.name == "Decimal" || c.key.path == "System.Decimal" {
            for m in c.methods.iter().filter(|m| m.name == "Round") {
                rounds.push((
                    c.key.path.clone(),
                    m.param_types.clone(),
                    m.param_is_byref.clone(),
                    format!("{:?}", m.access),
                ));
            }
        }
    }
    eprintln!("[rubyc] SCRATCH-TABLE rounds={rounds:?}");
    assert!(
        rounds.iter().any(|(_, _, b, _)| b.iter().any(|x| *x)),
        "table must keep the ref twin (saw {rounds:?})"
    );
}

#[test]
fn interlocked_compare_exchange_stores_through_ref() {
    // `Interlocked.CompareExchange(ref staticField, …)` must STORE
    // (the `ref`-marked assign target normalizes to the inner Ident;
    // dropping it left every lazily-initialized singleton null →
    // null-virtual crashes deep in init).
    let p = lower(
        r#"namespace t;
static class Interlocked {
    public static int CompareExchange(ref int location1, int value, int comparand) { return 0; }
}
class C {
    static int _x;
    static int Main() { return Interlocked.CompareExchange(ref _x, 1, 0); }
}"#,
    )
    .unwrap();
    let stores = p
        .functions
        .iter()
        .filter(|f| !f.name.contains("@start"))
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(i, crate::native::ir::Instr::StoreField { .. }))
        .count();
    assert!(
        stores >= 1,
        "CompareExchange must emit the conditional store (saw {stores})"
    );
}

#[test]
fn using_parens_body_executes() {
    // `using (var r = Make()) { return Use(r); }` must lower the
    // body (both calls present) — the old shape skipped it whole,
    // so `File.ReadAllBytes` (implemented inside `using (...)`)
    // lowered to an empty body returning null.
    let p = lower(
        r#"namespace t;
class C {
    static int Make() { return 7; }
    static int Use(int x) { return x; }
    int Main() { using (var r = Make()) { return Use(r); } }
}"#,
    )
    .unwrap();
    let calls: Vec<String> = p
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter_map(|i| match i {
            crate::native::ir::Instr::CallFn { .. } => Some("fn".to_string()),
            crate::native::ir::Instr::CallStatic { .. } => Some("static".to_string()),
            _ => None,
        })
        .collect();
    assert!(
        calls.len() >= 2,
        "using body must execute (Make + Use calls, saw {calls:?})"
    );
}

#[test]
fn using_parens_bare_expr_evaluates() {
    // `using (Make()) { return 1; }` evaluates the resource for side
    // effects and runs the body.
    let p = lower(
        r#"namespace t;
class C {
    static int Make() { return 7; }
    int Main() { using (Make()) { return 1; } }
}"#,
    )
    .unwrap();
    let calls = p
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| {
            matches!(
                i,
                crate::native::ir::Instr::CallFn { .. }
                    | crate::native::ir::Instr::CallStatic { .. }
            )
        })
        .count();
    assert!(calls >= 1, "using resource must evaluate (saw {calls})");
}

/// True when some `return` in the matching function feeds on a value
/// the caller recognizes (`f` inspects that value's defining
/// instruction). `Interlocked` tests ask "did the emitted return read
/// through the element address?".
fn ret_src_defined_by(
    p: &Program,
    fsub: &str,
    mut f: impl FnMut(&crate::native::ir::Instr) -> bool,
) -> bool {
    use crate::native::ir::{Instr, Terminator};
    for func in p.functions.iter().filter(|x| x.name.contains(fsub)) {
        let mut defs = std::collections::BTreeMap::new();
        for b in func.blocks.iter() {
            for i in b.instrs.iter() {
                let dst = match i {
                    Instr::LoadField { dst, .. } => Some(dst.0),
                    Instr::CallStatic { dst, .. } => Some(dst.0),
                    Instr::CallFn { dst, .. } => Some(dst.0),
                    Instr::CallVirt { dst, .. } => Some(dst.0),
                    Instr::Const { dst, .. } => Some(dst.0),
                    _ => None,
                };
                if let Some(d) = dst {
                    defs.insert(d, i);
                }
            }
            if let Terminator::Ret(Some(src)) = &b.term {
                if let Some(def) = defs.get(&src.0) {
                    if f(def) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

#[test]
fn compare_exchange_ref_index_reads_value_not_address() {
    // `Interlocked.CompareExchange(ref arr[i], v, c)` must compare and
    // return the VALUE at the element, not the interior address:
    // call-site `ref` erases to the address, so the raw eval compared
    // address-vs-comparand (conditional store silently skipped —
    // `_buckets[i]` stayed null forever) and returned the address
    // downstream (`?? inst` picked a heap address whose null slot
    // virtual-dispatched into the file-I/O segfault). The returned
    // vreg must be defined by a dereferencing load.
    let p = lower(
        r#"namespace t;
static class Interlocked {
    public static int CompareExchange(ref int location1, int value, int comparand) { return 0; }
}
class C {
    static int Main() { int[] a = new int[1]; a[0] = 5; return Interlocked.CompareExchange(ref a[0], 9, 5); }
}"#,
    )
    .unwrap();
    assert!(
        ret_src_defined_by(&p, "Main", |i| matches!(
            i,
            crate::native::ir::Instr::LoadField { off: 0, .. }
        )),
        "CompareExchange must return the loaded element value"
    );
}

#[test]
fn exchange_ref_index_reads_value_not_address() {
    // Same class as above for `Exchange`: the returned old value must
    // come through the element address, never be the address itself.
    let p = lower(
        r#"namespace t;
static class Interlocked {
    public static int Exchange(ref int location1, int value) { return 0; }
}
class C {
    static int Main() { int[] a = new int[1]; a[0] = 5; return Interlocked.Exchange(ref a[0], 9); }
}"#,
    )
    .unwrap();
    assert!(
        ret_src_defined_by(&p, "Main", |i| matches!(
            i,
            crate::native::ir::Instr::LoadField { off: 0, .. }
        )),
        "Exchange must return the loaded element value"
    );
}

#[test]
fn bare_compare_exchange_ref_index_reads_value() {
    // Bare same-class path (`calls.rs` emulation, reached when local
    // resolution fails on the self-recursive placeholder body): the
    // same address-vs-value rule applies — the body below recurses,
    // so the inner call emulates with a `ref` array element.
    let p = lower(
        r#"namespace t;
static class Interlocked {
    public static int CompareExchange(ref int location1, int value, int comparand) {
        int[] a = new int[1]; a[0] = 5;
        return CompareExchange(ref a[0], value, comparand);
    }
}
class C {
    static int Main() { int[] a = new int[1]; return 0; }
}"#,
    )
    .unwrap();
    assert!(
        ret_src_defined_by(&p, "CompareExchange", |i| matches!(
            i,
            crate::native::ir::Instr::LoadField { off: 0, .. }
        )),
        "bare CompareExchange must return the loaded element value"
    );
}

#[test]
fn eventsource_isenabled_folds_false_no_call() {
    // No-ETW configuration: static images ship no event pipeline, so
    // `EventSource.IsEnabled()` is always false (real .NET without
    // listeners also yields false and skips the block). The call must
    // fold to zero with no dispatch: `ArrayPoolEventSource.Log` builds
    // a broken singleton (vtable slot holds -1) and the pooled-Rent
    // tail calls `log.IsEnabled()` on it (file-I/O segfault).
    // The body returns true so any real dispatch is observable.
    let p = lower(
        r#"namespace t;
class ScratchPoolEventSource {
    public bool IsEnabled() { return true; }
}
class C {
    bool Main(ScratchPoolEventSource log) { return log.IsEnabled(); }
}"#,
    )
    .unwrap();
    let calls = p
        .functions
        .iter()
        .filter(|f| !f.name.contains("@start"))
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| {
            matches!(
                i,
                crate::native::ir::Instr::CallVirt { .. }
                    | crate::native::ir::Instr::CallStatic { .. }
                    | crate::native::ir::Instr::CallFn { .. }
            )
        })
        .count();
    assert_eq!(calls, 0, "EventSource.IsEnabled must fold (calls={calls})");
}

#[test]
fn plain_isenabled_still_dispatches() {
    // Anti-overreach guard: `IsEnabled` on a non-EventSource type
    // keeps normal virtual dispatch (the fold above is family-gated).
    let p = lower(
        r#"namespace t;
class Plain {
    public bool IsEnabled() { return true; }
}
class C {
    bool Main(Plain p) { return p.IsEnabled(); }
}"#,
    )
    .unwrap();
    let calls = p
        .functions
        .iter()
        .filter(|f| !f.name.contains("@start"))
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| {
            matches!(
                i,
                crate::native::ir::Instr::CallVirt { .. }
                    | crate::native::ir::Instr::CallStatic { .. }
                    | crate::native::ir::Instr::CallFn { .. }
            )
        })
        .count();
    assert!(calls >= 1, "plain IsEnabled must dispatch (saw {calls})");
}

#[test]
fn objectheader_acquire_folds_no_call() {
    // No-header configuration: our objects carry no CoreCLR header
    // word (`[obj-4]` is foreign memory), so the thin-lock protocol
    // cannot run — `AcquireThinLock` reads garbage/faults there
    // (file-I/O segfault via `Partition.TryPush`). Static exes are
    // single-threaded, so acquisition is vacuous: fold to void with
    // no dispatch. The body returns a constant so any real dispatch
    // is observable.
    let p = lower(
        r#"namespace t;
static class ScratchObjectHeader {
    public static void AcquireThinLock(object o) { return; }
}
class C {
    static int Main() { object o = new object(); ScratchObjectHeader.AcquireThinLock(o); return 0; }
}"#,
    )
    .unwrap();
    let calls = p
        .functions
        .iter()
        .filter(|f| !f.name.contains("@start"))
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| {
            matches!(
                i,
                crate::native::ir::Instr::CallVirt { .. }
                    | crate::native::ir::Instr::CallStatic { .. }
                    | crate::native::ir::Instr::CallFn { .. }
            )
        })
        .count();
    assert_eq!(calls, 0, "ObjectHeader.AcquireThinLock must fold (calls={calls})");
}

#[test]
fn objectheader_tryacquire_folds_true_no_call() {
    // Same family: `TryAcquireThinLock` always succeeds uncontended.
    // Folds to true (const 1) with no dispatch, both arities.
    let p = lower(
        r#"namespace t;
static class ScratchObjectHeader {
    public static bool TryAcquireThinLock(object o) { return false; }
    public static bool TryAcquireThinLock(object o, int ms) { return false; }
}
class C {
    static int Main(object o) {
        bool a = ScratchObjectHeader.TryAcquireThinLock(o);
        bool b = ScratchObjectHeader.TryAcquireThinLock(o, 0);
        return 0;
    }
}"#,
    )
    .unwrap();
    let calls = p
        .functions
        .iter()
        .filter(|f| !f.name.contains("@start"))
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| {
            matches!(
                i,
                crate::native::ir::Instr::CallVirt { .. }
                    | crate::native::ir::Instr::CallStatic { .. }
                    | crate::native::ir::Instr::CallFn { .. }
            )
        })
        .count();
    assert_eq!(calls, 0, "ObjectHeader.TryAcquireThinLock must fold (calls={calls})");
}

#[test]
fn monitor_enter_delegates_to_folded_objectheader() {
    // `Monitor.Enter` bodies delegate to `ObjectHeader`; with the
    // fold, running the delegating body must emit no call to the
    // header primitive — only Main→Enter itself. (Stubs stand in
    // for the framework pair with the real shapes.)
    let p = lower(
        r#"namespace t;
static class ScratchObjectHeader {
    public static void AcquireThinLock(object o) { return; }
}
static class ScratchMonitor {
    public static void Enter(object o) { ScratchObjectHeader.AcquireThinLock(o); }
}
class C {
    static int Main() { ScratchMonitor.Enter(new object()); return 0; }
}"#,
    )
    .unwrap();
    let calls = p
        .functions
        .iter()
        .filter(|f| !f.name.contains("@start"))
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| {
            matches!(
                i,
                crate::native::ir::Instr::CallVirt { .. }
                    | crate::native::ir::Instr::CallStatic { .. }
                    | crate::native::ir::Instr::CallFn { .. }
            )
        })
        .count();
    assert_eq!(calls, 1, "only Main→Enter may dispatch (calls={calls})");
}

#[test]
fn plain_acquire_still_dispatches() {
    // Anti-overreach guard: `AcquireThinLock` on a non-ObjectHeader
    // type keeps normal dispatch (the fold is family-gated).
    let p = lower(
        r#"namespace t;
class Plain {
    public static void AcquireThinLock(object o) { return; }
}
class C {
    static int Main() { Plain.AcquireThinLock(new object()); return 0; }
}"#,
    )
    .unwrap();
    let calls = p
        .functions
        .iter()
        .filter(|f| !f.name.contains("@start"))
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| {
            matches!(
                i,
                crate::native::ir::Instr::CallVirt { .. }
                    | crate::native::ir::Instr::CallStatic { .. }
                    | crate::native::ir::Instr::CallFn { .. }
            )
        })
        .count();
    assert!(calls >= 1, "plain AcquireThinLock must dispatch (saw {calls})");
}

/// @start argument plumbing: the static ABI packs params from rdi
/// (VReg j = argv[j-1], VReg 0 = padding), so a static
/// `Main(string[])` must receive exactly [argv] — passing the entry
/// receiver first shifts every param by one (`args` aliases the
/// receiver object: Length read its vtable word, element reads went
/// wild — argv echo segfault). Returns the argument count @start uses
/// for its entry call.
fn start_main_call_argc(p: &Program) -> Option<usize> {
    use crate::native::ir::Instr;
    for func in p.functions.iter().filter(|x| x.name.contains("@start")) {
        for b in func.blocks.iter() {
            for i in b.instrs.iter() {
                if let Instr::CallFn { args, .. } = i {
                    return Some(args.len());
                }
            }
        }
    }
    None
}

#[test]
fn start_passes_argv_only_to_static_main() {
    let p = lower(
        r#"namespace t;
class Program {
    static int Main(string[] args) { return args.Length; }
}"#,
    )
    .unwrap();
    assert_eq!(
        start_main_call_argc(&p),
        Some(1),
        "@start must call static Main(string[]) with [argv] only"
    );
}

#[test]
fn start_passes_receiver_to_instance_main() {
    // Anti-overreach guard: instance `Main(string[])` keeps the
    // (receiver, argv) pair — only the static shape drops `obj`.
    let p = lower(
        r#"namespace t;
class Program {
    int Main(string[] args) { return args.Length; }
}"#,
    )
    .unwrap();
    assert_eq!(
        start_main_call_argc(&p),
        Some(2),
        "@start must call instance Main(string[]) with [obj, argv]"
    );
}
