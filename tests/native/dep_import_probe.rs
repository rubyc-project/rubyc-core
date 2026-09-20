//! Dynamic-linking consume probes:
//! - static cross-image calls lower to `CallImport` (dep `.so` lib hint);
//! - virtual cross-image calls stay `CallVirt`: the receiver object's
//!   home image owns the vtable row, and fragments carry canonical
//!   `vtable_slot`s, so no import is needed.
use std::collections::HashMap;

fn preload_of(lib: &str, class_path: &str, src: &str) -> crate::native::semantic::Preloaded {
    let unit = crate::parser::parse(src).unwrap();
    let table = crate::native::semantic::analyze(&unit).unwrap();
    let mut e = table
        .classes
        .iter()
        .find(|c| c.key.path == class_path)
        .unwrap()
        .clone();
    e.owner = lib.to_owned();
    let bytes = crate::native::rci::encode(lib, 1, &[e], &[], &[]);
    let frag = crate::native::rci::decode(&bytes).unwrap();
    crate::native::semantic::Preloaded {
        classes: frag.classes,
        interfaces: frag.interfaces,
        delegates: frag.delegates,
    }
}

#[test]
fn dep_static_call_imports_fragment() {
    let pre = preload_of(
        "DepLib",
        "du.U",
        "namespace du; static class U { public static int Helper(int x) { return x; } }",
    );
    let main_unit = crate::parser::parse(
        "namespace t { using du; class Caller { int M() { return U.Helper(1); } int Main() { return 0; } } }",
    )
    .unwrap();
    let owners: HashMap<crate::native::semantic::TypeKey, String> = HashMap::from([
        (
            crate::native::semantic::TypeKey::new("du.U", 0),
            "DepLib".to_owned(),
        ),
        (
            crate::native::semantic::TypeKey::new("t.Caller", 0),
            "MainLib".to_owned(),
        ),
    ]);
    let prog = crate::native::ir::lower_with_owners(&main_unit, true, &[], &owners, &pre)
        .expect("cross-image static call lowers");
    assert!(
        prog.imports
            .contains(&("DepLib.so".to_owned(), "du.U::Helper#1".to_owned())),
        "imports: {:?}",
        prog.imports
    );
}

#[test]
fn dep_virtual_call_needs_no_import() {
    let pre = preload_of(
        "DepLib",
        "du.V",
        "namespace du; class V { public virtual int Vf(int x) { return x; } }",
    );
    let main_unit = crate::parser::parse(
        "namespace t { using du; class Caller { int M(V v) { return v.Vf(2); } int Main() { return 0; } } }",
    )
    .unwrap();
    let owners: HashMap<crate::native::semantic::TypeKey, String> = HashMap::from([
        (
            crate::native::semantic::TypeKey::new("du.V", 0),
            "DepLib".to_owned(),
        ),
        (
            crate::native::semantic::TypeKey::new("t.Caller", 0),
            "MainLib".to_owned(),
        ),
    ]);
    let prog = crate::native::ir::lower_with_owners(&main_unit, true, &[], &owners, &pre)
        .expect("cross-image virtual call lowers");
    assert!(
        prog.imports.is_empty(),
        "virtual dispatch rides the receiver vtable; imports: {:?}",
        prog.imports
    );
}

#[test]
fn dep_ctor_call_imports_fragment() {
    let pre = preload_of(
        "DepLib",
        "du.W",
        "namespace du; class W { public W(int x) { } }",
    );
    let main_unit = crate::parser::parse(
        "namespace t { using du; class Caller { int M() { var w = new W(3); return 0; } int Main() { return 0; } } }",
    )
    .unwrap();
    let owners: HashMap<crate::native::semantic::TypeKey, String> = HashMap::from([
        (
            crate::native::semantic::TypeKey::new("du.W", 0),
            "DepLib".to_owned(),
        ),
        (
            crate::native::semantic::TypeKey::new("t.Caller", 0),
            "MainLib".to_owned(),
        ),
    ]);
    let prog = crate::native::ir::lower_with_owners(&main_unit, true, &[], &owners, &pre)
        .expect("cross-image ctor lowers");
    assert!(
        prog.imports.contains(&(
            "DepLib.so".to_owned(),
            "du.W::__vtable".to_owned()
        )),
        "vtable import: {:?}",
        prog.imports
    );
    assert!(
        prog.imports
            .iter()
            .any(|(_, s)| s == "du.W::W#1"),
        "imports: {:?}",
        prog.imports
    );
}

#[test]
fn dep_static_field_read_imports_slot() {
    let pre = preload_of(
        "DepLib",
        "du.U",
        "namespace du; static class U { public static int SF; public static int Helper(int x) { return x; } }",
    );
    let main_unit = crate::parser::parse(
        "namespace t { using du; class Caller { int M() { return U.SF; } int Main() { return 0; } } }",
    )
    .unwrap();
    let owners: HashMap<crate::native::semantic::TypeKey, String> = HashMap::from([
        (
            crate::native::semantic::TypeKey::new("du.U", 0),
            "DepLib".to_owned(),
        ),
        (
            crate::native::semantic::TypeKey::new("t.Caller", 0),
            "MainLib".to_owned(),
        ),
    ]);
    let prog = crate::native::ir::lower_with_owners(&main_unit, true, &[], &owners, &pre)
        .expect("cross-image static field read lowers");
    assert!(
        prog.imports
            .contains(&("DepLib.so".to_owned(), "du.U::SF".to_owned())),
        "imports: {:?}",
        prog.imports
    );
}

#[test]
fn dep_static_field_write_imports_slot() {
    let pre = preload_of(
        "DepLib",
        "du.U",
        "namespace du; static class U { public static int SF; }",
    );
    let main_unit = crate::parser::parse(
        "namespace t { using du; class Caller { void M() { U.SF = 7; } int Main() { return 0; } } }",
    )
    .unwrap();
    let owners: HashMap<crate::native::semantic::TypeKey, String> = HashMap::from([
        (
            crate::native::semantic::TypeKey::new("du.U", 0),
            "DepLib".to_owned(),
        ),
        (
            crate::native::semantic::TypeKey::new("t.Caller", 0),
            "MainLib".to_owned(),
        ),
    ]);
    let prog = crate::native::ir::lower_with_owners(&main_unit, true, &[], &owners, &pre)
        .expect("cross-image static field write lowers");
    assert!(
        prog.imports
            .contains(&("DepLib.so".to_owned(), "du.U::SF".to_owned())),
        "imports: {:?}",
        prog.imports
    );
}

#[test]
fn alias_spelled_ctor_resolves_target() {
    // `using WorkQueue = Queue<object>` (ThreadPoolWorkQueue): the
    // call-site spelling never matches ctor names — selection retries
    // with the resolved entry's own name instead of erroring ctor_arity.
    let src = "using WorkQueue = System.Collections.Generic.Queue<object>; namespace t { class C { WorkQueue M() { return new WorkQueue(); } int Main() { return 0; } } } namespace System.Collections.Generic { class Queue<T> { public Queue() { } public Queue(int n) { } } }";
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower_with_defines(&unit, true, &[])
        .expect("alias-spelled construction lowers");
}

#[test]
fn target_side_implicit_op_applies() {
    // `ErrArg.implicit operator ErrArg(string)` (CSharp ErrorHandling):
    // C# allows the operator on the TARGET side — overload resolution
    // must accept `string`/`CType` args for `ErrArg` params.
    let src = "namespace t { enum ErrorCode { A } class CType { } class ErrArg { public ErrArg(string s) { } public static implicit operator ErrArg(string s) { return new ErrArg(s); } public static implicit operator ErrArg(CType t) { return new ErrArg(\"\"); } } class Ex { } static class EH { public static Ex Error(ErrorCode id, params ErrArg[] args) { return new Ex(); } } class C { Ex M(ErrorCode id, string v, CType d) { throw EH.Error(id, v, d); } int Main() { return 0; } } }";
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower_with_defines(&unit, true, &[])
        .expect("target-side implicit conversions apply");
}

#[test]
fn same_arity_overloads_emit_distinct_targets() {
    // Same `{class}::{method}#{arity}` keys for different signatures
    // (Task#2 vs bool#2, Task#3 vs ValueTask#3) must not desync the
    // function index from the functions vector: the call to the
    // 4-param ValueTask overload (Sockets' SendToAsync) must target
    // its own function, not the first overload (which has no AsTask).
    // The `ArraySegment<T>` `T[]`-based implicit operator must NOT
    // claim the `Memory` argument (no shape match) — otherwise the
    // `Task` overload wins Tier 1 and the same failure recurs.
    let src = "namespace System { static class MemoryExtensions { public static System.Memory<T> AsMemory<T>(this T[]? array, int start, int length) { return new System.Memory<T>(); } } struct Memory<T> { } struct ReadOnlyMemory<T> { } struct ArraySegment<T> { public static implicit operator ArraySegment<T>(T[] array) { return new ArraySegment<T>(); } } } namespace System.Threading { struct CancellationToken { } } namespace System.Threading.Tasks { class ValueTask<T> { public Task<T> AsTask() { return new Task<T>(); } } class Task<T> { } } namespace System.Net.Sockets { enum SF { None } class EP { } class SAE { } partial class Sock { int Main() { return 0; } public System.Threading.Tasks.Task<int> SendToAsync(System.ArraySegment<byte> buffer, EP ep) { return new System.Threading.Tasks.Task<int>(); } public System.Threading.Tasks.Task<int> SendToAsync(System.ArraySegment<byte> buffer, SF flags, EP ep) { return new System.Threading.Tasks.Task<int>(); } public bool SendToAsync(SAE e) { return false; } public bool SendToAsync(SAE e, System.Threading.CancellationToken c) { return false; } } partial class Sock { public System.Threading.Tasks.ValueTask<int> SendToAsync(System.ReadOnlyMemory<byte> buffer, EP ep, System.Threading.CancellationToken ct = default) { return new System.Threading.Tasks.ValueTask<int>(); } public System.Threading.Tasks.ValueTask<int> SendToAsync(System.ReadOnlyMemory<byte> buffer, SF flags, EP ep, System.Threading.CancellationToken ct = default) { return new System.Threading.Tasks.ValueTask<int>(); } public System.Threading.Tasks.Task<int> M(byte[] buffer, int offset, int size, SF flags, EP ep) { return SendToAsync(buffer.AsMemory(offset, size), flags, ep).AsTask(); } } }";
    let unit = crate::parser::parse(src).unwrap();
    let prog = crate::native::ir::lower_with_defines(&unit, true, &[])
        .expect("same-arity overloads lower");
    let mut picked = Vec::new();
    for f in &prog.functions {
        if !f.name.contains("Sock::M#") {
            continue;
        }
        for b in &f.blocks {
            for i in &b.instrs {
                match i {
                    crate::native::ir::Instr::CallStatic { fidx, .. }
                    | crate::native::ir::Instr::CallFn { fidx, .. } => {
                        if let Some(t) = prog.functions.get(*fidx as usize) {
                            picked.push(t.name.clone());
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    assert!(
        picked.iter().any(|n| n == "Sock::SendToAsync#4"),
        "expected SendToAsync#4, got {picked:?}"
    );
}

#[test]
fn is_var_pattern_declares_binding() {
    // `while (A is var b && ...)` (FileProviders): the declaration
    // pattern must declare `b` in the enclosing scope — previously the
    // binding was consumed and dropped, failing downstream with
    // "undeclared variable 'var'".
    let src = "namespace t { class C { bool M(object o) { while (o is var child && child != null) { } return true; } int Main() { return 0; } } }";
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower_with_defines(&unit, true, &[])
        .expect("is-var pattern lowers with its binding declared");
}

#[test]
fn derives_from_fans_out_across_same_named_classes() {
    // `System.ComponentModel.ArrayConverter` (first alphabetically)
    // must not shadow `System.Text.Json.…ArrayConverter<TCollection,
    // TElement>` in protected-access walks: the Json one derives from
    // JsonCollectionConverter, so its protected members stay visible.
    let src = "namespace System.ComponentModel { class ArrayConverter { } } namespace System.Text.Json { class JsonCollectionConverter<TC, TE> { protected static int Helper() { return 0; } } class Mid<TC, TE> : JsonCollectionConverter<TC, TE> { } class ArrayConverter<TC, TE> : Mid<TE[], TE> { int M() { return Helper(); } } } namespace t { class K { int Main() { return 0; } } }";
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower_with_defines(&unit, true, &[])
        .expect("protected access through shadowed same-named class");
}

#[test]
fn params_object_packing_matches() {
    // `WriteEvent(2, level, …)` into `(int, params object?[])`
    // (LoggingEventSource): multi-arg packing must apply. The params
    // element strips to canonicalized `Object`, which the boxing
    // rules must accept like `object`.
    let src = "namespace t { static class C { public static void WriteEvent(int eventId, params object[] args) { } } class K { void M(int level, string name) { C.WriteEvent(2, level, name); } int Main() { return 0; } } }";
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower_with_defines(&unit, true, &[])
        .expect("multi-arg params packing lowers");
}

#[test]
fn explicit_generic_arity_shadows_nongeneric() {
    // `services.AddOptions<NoOp>()` must elect the generic overload
    // even though the non-generic `AddOptions()` is declared first:
    // explicit type arguments restrict candidates to matching generic
    // arity (C# errors otherwise), and the generic overload must
    // survive table dedup (same erased params). The chained
    // `.ValidateOnStart()` (only on `OptionsBuilder<T>`) proves the
    // right overload was elected — electing the non-generic one types
    // the chain as `IServiceCollection`, which has no such member
    // (Microsoft.Extensions.Diagnostics).
    let src = "namespace t { interface IServiceCollection { } class OptionsBuilder<T> { } static class Exts { public static IServiceCollection AddOptions(this IServiceCollection s) { return s; } public static OptionsBuilder<T> AddOptions<T>(this IServiceCollection s) where T : class { return new OptionsBuilder<T>(); } public static OptionsBuilder<T> ValidateOnStart<T>(this OptionsBuilder<T> b) where T : class { return b; } } class Impl : IServiceCollection { void M(IServiceCollection services) { services.AddOptions<NoOp>().ValidateOnStart(); } int Main() { return 0; } } class NoOp { } }";
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower_with_defines(&unit, true, &[])
        .expect("explicit generic arity elects the generic overload");
}

#[test]
fn union_dedup_respects_generic_arity() {
    // `M<T>(IServiceCollection)` vs `M<T,U>(IServiceCollection)` share
    // erased params but are distinct overloads: union merging must not
    // drop one as a duplicate of the other (which lost the 2-param
    // `TryAddSingleton` overload in Diagnostics' closure).
    use crate::models::{ClassMember, Member};
    let src = "namespace t { static class C { public static void M<T>(object o) { } public static void M<T, U>(object o) { } } class K { int Main() { return 0; } } }";
    let unit = crate::parser::parse(src).unwrap();
    let (mut one, mut two) = (Vec::new(), Vec::new());
    for ns in &unit.namespaces {
        for m in &ns.members {
            if let Member::Class(c) = m {
                for cm in &c.members {
                    if let ClassMember::Method(md) = cm {
                        if md.name == "M" && md.generic_params.len() == 1 {
                            one.push(cm.clone());
                        }
                        if md.name == "M" && md.generic_params.len() == 2 {
                            two.push(cm.clone());
                        }
                    }
                }
            }
        }
    }
    assert_eq!(one.len(), 1);
    assert_eq!(two.len(), 1);
    assert!(
        !crate::driver::class_member_is_dup(&one, &two[0]),
        "different generic arity is not a duplicate"
    );
}

#[test]
fn tuple_deconstruction_keeps_declared_types() {
    // `(HostingEnvironment h, PhysicalFileProvider p) = …` (Hosting):
    // destructured names must carry their DECLARED types. Previously
    // they synthesized as `var` (inferring `int` from the zero init),
    // so the extension call below resolved its receiver imprecisely
    // and missed — while a same-named member anywhere would still
    // resolve leniently. Assert the emitted target.
    let src = "namespace t { class H { } class P { } static class Exts { public static string Who(this P p) { return \"\"; } public static string Who(this H p) { return \"\"; } } class C { void M() { (H hostingEnvironment, P physicalFileProvider) = (new H(), new P()); string s = physicalFileProvider.Who(); } int Main() { return 0; } } }";
    let unit = crate::parser::parse(src).unwrap();
    let prog = crate::native::ir::lower_with_defines(&unit, true, &[])
        .expect("destructured names keep declared types");
    let mut picked = Vec::new();
    for f in &prog.functions {
        if !f.name.contains("C::M#") {
            continue;
        }
        for b in &f.blocks {
            for i in &b.instrs {
                match i {
                    crate::native::ir::Instr::CallStatic { fidx, .. }
                    | crate::native::ir::Instr::CallFn { fidx, .. } => {
                        if let Some(t) = prog.functions.get(*fidx as usize) {
                            picked.push(t.name.clone());
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    assert!(
        picked.iter().any(|n| n.contains("Who")),
        "expected a Who call, got {picked:?}"
    );
}

#[test]
fn nuint_arithmetic_types_numerically() {
    // `pos * 2` with `nuint pos` must type as `nuint`, not `object`:
    // otherwise the `ValueTask<int> Take(nuint)` overload misses, the
    // `Task<int> Take(object)` overload wins Tier 1 via the object
    // wildcard, and its result has no `AsTask` (HexConverter's
    // `StoreUnsafe(ref dest, pos * 2)` shape).
    let src = "namespace System.Threading.Tasks { class ValueTask<T> { public Task<T> AsTask() { return new Task<T>(); } } class Task<T> { } } namespace t { class C { public System.Threading.Tasks.Task<int> Take(object o) { return new System.Threading.Tasks.Task<int>(); } public System.Threading.Tasks.ValueTask<int> Take(nuint n) { return new System.Threading.Tasks.ValueTask<int>(); } public System.Threading.Tasks.Task<int> M(nuint pos) { return Take(pos * 2).AsTask(); } int Main() { return 0; } } }";
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower_with_defines(&unit, true, &[])
        .expect("nuint arithmetic resolves the nuint overload");
}

#[test]
fn static_methods_take_no_receiver() {
    // Static callees receive only declared args: a `This` receiver
    // shifts every static arg into garbage (params read zero at
    // runtime). Instance methods keep theirs.
    use crate::native::ir::ReceiverKind;
    let src = "namespace t { class U { public static int Helper(int x) { return x; } public int Id(int x) { return x; } } class K { int Main() { return 0; } } }";
    let unit = crate::parser::parse(src).unwrap();
    let prog = crate::native::ir::lower_with_defines(&unit, true, &[])
        .expect("lowers");
    let mut kinds = std::collections::HashMap::new();
    for f in &prog.functions {
        kinds.insert(f.name.clone(), f.receiver);
    }
    assert_eq!(
        kinds.get("U::Helper#1"),
        Some(&ReceiverKind::None),
        "static method takes no receiver: {:?}",
        kinds
    );
    assert_eq!(
        kinds.get("U::Id#1"),
        Some(&ReceiverKind::This),
        "instance method keeps its receiver: {:?}",
        kinds
    );
}

#[test]
fn static_initializer_uses_declaring_class_viewer() {
    // A static field initializer calling a private static method of
    // the same class (`SR.s_usingResourceKeys =
    // GetUsingResourceKeysSwitchValue()`): the initializer is the
    // declaring class's own code, so access checks run with that
    // class as viewer — not the entry class (`@start` context).
    let src = "namespace System { internal static partial class SR { private static readonly bool s_usingResourceKeys = GetUsingResourceKeysSwitchValue(); private static bool GetUsingResourceKeysSwitchValue() { return false; } } internal static partial class SR { public static string Foo => \"x\"; } } namespace t { class K { int Main() { return 0; } } }";
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower_with_defines(&unit, true, &[])
        .expect("same-class private calls in initializers lower");
}

#[test]
fn named_out_var_in_field_initializer() {
    // `s_flag = TryGetSwitch(switchName: "x", isEnabled: out bool
    // isEnabled) ? isEnabled : false` (TypeDescriptor): an `out`
    // variable declared inside a field initializer must be visible
    // to the rest of that initializer. Field declarations have no
    // statement list, so the binding rides on the field and lowers
    // first.
    let src = "namespace t { static class App { public static bool TryGetSwitch(string switchName, out bool isEnabled) { isEnabled = true; return true; } } class C { static bool s_flag = App.TryGetSwitch(switchName: \"x\", isEnabled: out bool isEnabled) ? isEnabled : false; int Main() { return 0; } } }";
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower_with_defines(&unit, true, &[])
        .expect("out-vars in field initializers lower");
}
