//! Semantic type tables: inheritance queries, C# access rules
//! (`public`/`private`/`protected`, default = private), and member lookup.

use crate::models::Access;
use crate::native::semantic::{
    visible_from, ClassInfo, FieldInfo, MethodInfo, TypeKey, TypeTable,
};

fn table() -> TypeTable {
    TypeTable {
        delegates: vec![],
        classes: vec![
            ClassInfo {
                name: "Animal".into(),
                key: TypeKey::new("Animal", 0),
                bases: vec![],
                fields: vec![FieldInfo {
                    name: "age".into(),
                    type_name: "int".into(),
                    access: Access::Protected,
                    offset_slots: 1,
                    is_static: false,
                    is_const: false,
                    const_value: None,
                    declaring_class: "Animal".into(),
                }],
                static_fields: vec![],
                const_fields: vec![],
                methods: vec![MethodInfo {
                    name: "Speak".into(),
                    ret_type: "void".into(),
                    param_types: vec![],
                    param_has_default: vec![],
                    param_defaults: vec![],
                    access: Access::Public,
                    is_static: false,
                    has_body: true,
                    declaring_class: "Animal".into(),
                    is_extension: false,
                    generic_params: vec![],
                    vtable_slot: 0,
                    conditional_symbol: None,
                    has_params: false,
                }],
                own_members: vec![
                    ("age".into(), true, Access::Protected),
                    ("Speak".into(), false, Access::Public),
                ],
                interfaces: vec![("ILiving".into(), 0)],
                properties: vec![],
                is_enum: false,
                generic_params: vec![],
                is_struct: false,
                owner: String::new(),
            },
            ClassInfo {
                name: "Dog".into(),
                key: TypeKey::new("Dog", 0),
                bases: vec!["Animal".into()],
                fields: vec![],
                static_fields: vec![],
                const_fields: vec![],
                methods: vec![MethodInfo {
                    name: "Speak".into(),
                    ret_type: "void".into(),
                    param_types: vec![],
                    param_has_default: vec![],
                    param_defaults: vec![],
                    access: Access::Public,
                    is_static: false,
                    has_body: true,
                    declaring_class: "Dog".into(),
                    is_extension: false,
                    generic_params: vec![],
                    vtable_slot: 0,
                    conditional_symbol: None,
                    has_params: false,
                }],
                own_members: vec![("Speak".into(), false, Access::Public)],
                interfaces: vec![],
                properties: vec![],
                is_enum: false,
                generic_params: vec![],
                is_struct: false,
                owner: String::new(),
            },
            ClassInfo {
                name: "Stranger".into(),
                key: TypeKey::new("Stranger", 0),
                bases: vec![],
                fields: vec![],
                static_fields: vec![],
                const_fields: vec![],
                methods: vec![],
                own_members: vec![],
                interfaces: vec![],
                properties: vec![],
                is_enum: false,
                generic_params: vec![],
                is_struct: false,
                owner: String::new(),
            },
        ],
        interfaces: vec![],
        imports: vec![],
        aliases: vec![],
    }
}

#[test]
fn class_lookup_finds_declared_classes_only() {
    let t = table();
    assert!(t.class("Dog").is_some());
    assert!(t.class("Cat").is_none());
}

#[test]
fn derives_from_walks_the_base_chain() {
    let t = table();
    // Reflexive.
    assert!(t.derives_from("Dog", "Dog"));
    // One hop up.
    assert!(t.derives_from("Dog", "Animal"));
    // Unrelated.
    assert!(!t.derives_from("Dog", "Stranger"));
    // Downward is not derivation.
    assert!(!t.derives_from("Animal", "Dog"));
    // Unknown classes derive from nothing but themselves.
    assert!(t.derives_from("Ghost", "Ghost"));
    assert!(!t.derives_from("Ghost", "Animal"));
}

#[test]
fn public_is_visible_everywhere() {
    let t = table();
    assert!(visible_from(&t, Access::Public, "Animal", "Stranger"));
}

#[test]
fn private_and_default_are_declaring_class_only() {
    let t = table();
    assert!(visible_from(&t, Access::Private, "Animal", "Animal"));
    assert!(!visible_from(&t, Access::Private, "Animal", "Dog"));
    // No modifier behaves like private (C# default).
    assert!(visible_from(&t, Access::Default, "Animal", "Animal"));
    assert!(!visible_from(&t, Access::Default, "Animal", "Stranger"));
}

#[test]
fn protected_extends_to_derived_classes() {
    let t = table();
    assert!(visible_from(&t, Access::Protected, "Animal", "Animal"));
    assert!(visible_from(&t, Access::Protected, "Animal", "Dog"));
    assert!(!visible_from(&t, Access::Protected, "Animal", "Stranger"));
}

#[test]
fn alias_resolves_to_target_class() {
    // `using Handle = Ns.Animal` — the alias resolves to the target
    // class (qualification stripped); unknown targets stay unknown.
    let mut t = table();
    t.aliases.push(("Handle".into(), "Ns.Animal".into()));
    assert_eq!(t.class("Handle").unwrap().name, "Animal");
    let mut t2 = table();
    t2.aliases.push(("Ghost".into(), "Ns.Missing".into()));
    assert!(t2.class("Ghost").is_none());
    assert!(t2.class("Missing").is_none());
}

#[test]
fn extern_bodyless_methods_pass_analysis() {
    // `[MethodImpl(InternalCall)] extern` methods (CoreLib FCalls)
    // are runtime-provided: no E0019,unlike plain bodyless methods.
    let unit = crate::parser::parse(
        "class C { internal static extern string FastAllocateString(int length); }",
    )
    .expect("parse");
    let table = crate::native::semantic::analyze(&unit)
        .expect("extern methods must not raise E0019");
    assert!(table.class("C").is_some());
    let unit2 = crate::parser::parse("class D { internal static string Missing(int length); }")
        .expect("parse");
    let errs = crate::native::semantic::analyze(&unit2).expect_err("plain bodyless must raise");
    assert!(
        errs.iter().any(|e| e.code == "E0019"),
        "expected E0019, got {:?}",
        errs.iter().map(|e| &e.code).collect::<Vec<_>>()
    );
}

#[test]
fn nested_and_qualified_bases_resolve() {
    // Nested interface bases (`: IValueLength`) and qualified ones
    // (`: Ns.IValidator`) resolve by simple name — no E0011.
    // (`RuntimeType.IGenericCacheEntry` in CoreLib.)
    let unit = crate::parser::parse(
        "namespace n {
            class Helpers {
                public interface IValueLength {
                }
                public struct Tiny : IValueLength {
                }
            }
            class Outer {
                public interface IValidator {
                }
            }
            class Uses : Outer.IValidator {
            }
        }",
    )
    .expect("parse");
    let res = crate::native::semantic::analyze(&unit);
    assert!(
        res.is_ok(),
        "nested/qualified bases must resolve: {:?}",
        res.err().map(|es| es
            .iter()
            .map(|e| format!("{}: {}", e.code, e.message))
            .collect::<Vec<_>>())
    );
}

#[test]
fn field_and_method_queries_resolve_by_name() {
    let animal = table().class("Animal").unwrap().clone();
    let age = animal.field("age").expect("field query");
    assert_eq!(age.offset_slots, 1);
    assert_eq!(animal.method("Speak").unwrap().vtable_slot, 0);
    assert!(animal.method("Missing").is_none());
    // declares_* mirrors field/method for own-member enforcement.
    assert!(animal.declares_field("age").is_some());
    assert!(animal.declares_method("Speak").is_some());
}

#[test]
fn nested_class_registers_with_methods() {
    // Nested classes (`Task.ContingentProperties`) register under a
    // qualified key with their own methods intact.
    let src = r#"namespace t;
class Task {
    internal sealed class ContingentProperties {
        internal static void SetEvent(int m) { }
    }
    public void Dispose() {
        ContingentProperties.SetEvent(1);
    }
    int main() { return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    let table = crate::native::semantic::analyze(&unit).unwrap();
    let nested = table
        .classes
        .iter()
        .find(|c| c.name == "ContingentProperties")
        .expect("nested class registers");
    assert!(
        nested
            .methods
            .iter()
            .any(|m| m.name == "SetEvent" && m.is_static),
        "nested static method registers"
    );
}

#[test]
fn nesting_family_checks_all_same_named_entries() {
    // Three outers each nesting an `AlternateLookup` (the
    // `HashSet`/`Dictionary`/`ConcurrentDictionary` shape):
    // private access between an outer and ANY same-named nested
    // entry in its family allows, regardless of declaration order.
    let src = r#"namespace t;
class HashSet {
    class AlternateLookup {
    }
}
class Dictionary {
    class AlternateLookup {
    }
}
class ConcurrentDictionary {
    private int _tables;
    class AlternateLookup {
    }
}
"#;
    let unit = crate::parser::parse(src).unwrap();
    let table = crate::native::semantic::analyze(&unit).unwrap();
    assert!(
        crate::native::semantic::visible_from(
            &table,
            crate::models::Access::Private,
            "ConcurrentDictionary",
            "AlternateLookup"
        ),
        "expected nesting family across same-named entries"
    );
    assert!(
        !crate::native::semantic::visible_from(
            &table,
            crate::models::Access::Private,
            "HashSet",
            "Dictionary"
        ),
        "unrelated classes stay denied"
    );
}

#[test]
fn qualified_and_simple_same_entry_allows() {
    // `System.IO.Stream.ReadWriteTask` (qualified, e.g. from a scoped
    // type reference) and `ReadWriteTask` (simple, the current class)
    // denote the same entry: private access allows.
    let src = r#"namespace System.IO {
class Stream {
    class ReadWriteTask {
        private int _x;
    }
}
}"#;
    let unit = crate::parser::parse(src).unwrap();
    let table = crate::native::semantic::analyze(&unit).unwrap();
    assert!(
        crate::native::semantic::visible_from(
            &table,
            crate::models::Access::Private,
            "System.IO.Stream.ReadWriteTask",
            "ReadWriteTask"
        ),
        "expected same-entry qualified/simple access"
    );
}

#[test]
fn const_string_null_folds() {
    // `const string X = null` is legal C# (Interop consts like
    // `Kernel32.LOCALE_NAME_USER_DEFAULT`): the null reference is a
    // valid constant value, not E0030.
    let src = r#"namespace t;
class Kernel32 {
    internal const string LOCALE_NAME_USER_DEFAULT = null;
    int main() { return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    let table = crate::native::semantic::analyze(&unit).expect("const null must analyze");
    let cls = table.class("Kernel32").expect("Kernel32 present");
    let f = cls
        .const_fields
        .iter()
        .find(|f| f.name == "LOCALE_NAME_USER_DEFAULT")
        .expect("const field present");
    assert_eq!(
        f.const_value,
        Some(crate::native::semantic::ConstValue::Null),
        "const null must fold to Null"
    );
}

#[test]
fn const_interpolated_string_folds() {
    // `$"..."` with all-constant holes is a valid const initializer
    // (C# 10+; e.g. XmlKeyHelper's `IEnumerableOfXElementTypeName`):
    // the holes fold and concatenate at analysis time, not E0030.
    let src = r#"namespace t;
class Functions {
    private const string XElementTypeName = "System.Xml.Linq.XElement" + ", System.Private.Xml.Linq";
    private const string IEnumerableOfXElementTypeName = $"System.Collections.Generic.IEnumerable`1[[{XElementTypeName}]], System.Runtime";
    int main() { return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    let table = crate::native::semantic::analyze(&unit).expect("const interpolated must analyze");
    let cls = table.class("Functions").expect("Functions present");
    let f = cls
        .const_fields
        .iter()
        .find(|f| f.name == "IEnumerableOfXElementTypeName")
        .expect("const field present");
    assert_eq!(
        f.const_value,
        Some(crate::native::semantic::ConstValue::Str(
            "System.Collections.Generic.IEnumerable`1[[System.Xml.Linq.XElement, System.Private.Xml.Linq]], System.Runtime".to_string()
        )),
        "const holes must fold and concatenate"
    );
}

#[test]
fn enum_receiver_falls_back_to_system_enum() {
    // Enum values ARE `System.Enum` instances (.NET): instance calls
    // on enum-typed receivers (`VarType.HasFlag(...)` in CoreLib)
    // resolve through the `Enum` entry.
    let src = r#"namespace System {
class Enum { public bool HasFlag(Enum f) { return true; } }
enum Color { Red, Blue }
class User {
    bool M(Color c) { return c.HasFlag(Color.Red); }
    int main() { return 0; }
}}"#;
    let unit = crate::parser::parse(src).unwrap();
    let table = crate::native::semantic::analyze(&unit).expect("analyze");
    assert!(table.class("Color").expect("Color present").is_enum);
    let program = crate::native::ir::lower(&unit).expect("enum HasFlag must lower");
    assert!(!program.functions.is_empty());
}

#[test]
fn delegate_field_invocation_placeholders() {
    // `helper._callback(...)` invokes a delegate-typed field (C#).
    // Delegates lower as placeholders (real dispatch is runtime
    // work): receiver+args evaluate, the call yields zero.
    let src = r#"namespace t;
delegate void Cb(int x);
class Holder {
    Cb _cb;
}
class User {
    int Fire(Holder h) { h._cb(1); return 0; }
    int main() { return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).expect("delegate invocation must lower");
}

#[test]
fn private_visible_via_half_qualified_spelling() {
    // Namespace-relative spellings (`WaitSubsystem.ThreadWaitInfo`
    // as-written) belong to the same nesting family as the simple
    // viewer spelling (`ThreadWaitInfo`): private access within the
    // family stays legal.
    let src = r#"namespace t;
static class Outer {
    internal static partial class Inner {
        private static int Secret() { return 1; }
    }
    int main() { return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    let table = crate::native::semantic::analyze(&unit).expect("analyze");
    assert!(
        crate::native::semantic::visible_from(
            &table,
            crate::models::Access::Private,
            "Outer.Inner",
            "Inner"
        ),
        "half-qualified spelling must be same-family"
    );
}

#[test]
fn implicit_default_ctor_ignores_samenamed_sibling() {
    // `new State()` inside PeriodicTimer means PeriodicTimer.State
    // (enclosing nested wins). It declares no ctors, so the implicit
    // public parameterless one applies — Lock.State's explicit ctor
    // must not force an arity error.
    let src = r#"namespace t;
static class Lock {
    internal struct State {
        internal State(int x) { }
    }
}
static class PeriodicTimer {
    internal sealed class State {
    }
    int Make() { object o = new State(); return 0; }
    int main() { return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).expect("implicit default ctor must lower");
}

#[test]
fn struct_keeps_implicit_default_ctor_with_explicit_ctors() {
    // Structs keep their implicit parameterless constructor even with
    // explicit ctors (e.g. Asn1's `AttributeAsn` declares only
    // `AttributeAsn(AsnEncodedData)` yet `new AttributeAsn()` is legal
    // C#); classes with explicit ctors stay exact.
    let src = r#"namespace t;
internal struct S {
    internal S(int x) { }
}
class P {
    object MakeS() { return new S(); }
    int main() { return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).expect("struct implicit default must lower");
    let bad = r#"namespace t;
class C {
    internal C(int x) { }
}
class Q {
    object MakeC() { return new C(); }
    int main() { return 0; }
}"#;
    let unit = crate::parser::parse(bad).unwrap();
    assert!(
        crate::native::ir::lower(&unit).is_err(),
        "class with only explicit ctors must reject new C()"
    );
}

#[test]
fn multi_ctor_with_defaults_all_recorded() {
    // Overloads differing only by defaulted trailing params must ALL
    // reach the table (cf. AssemblyNameInfo's 5-param ctor alongside
    // its 1-param one): overload-unaware dedup once dropped all but
    // the first and broke 5-arg construction.
    let src = r#"namespace t;
enum Flags { None }
class Ver { }
struct Arr { }
class A {
    public A(string name, Ver? version = null, string? culture = null, Flags flags = Flags.None, Arr tok = default) { }
    internal A(int parts) { }
    int main() { return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    let table = crate::native::semantic::analyze(&unit).expect("must analyze");
    let cls = table.class("A").expect("A present");
    let ctors: Vec<usize> = cls
        .methods
        .iter()
        .filter(|m| m.name == "A")
        .map(|m| m.param_types.len())
        .collect();
    assert!(
        ctors.contains(&5) && ctors.contains(&1),
        "both ctors must be recorded: {ctors:?}"
    );
}

#[test]
fn tuple_return_method_with_attrs_recorded() {
    // A method returning a generic tuple with attributes between
    // (cf. HexConverter.AsciiToHexVector128:
    // `internal static (Vector128<byte>, Vector128<byte>)
    // AsciiToHexVector128(...)`): the member must reach the table —
    // a silent drop surfaces downstream as a phantom "has no member".
    let src = r#"namespace t;
class V { }
static class H {
    [X(Y)]
    internal static (V, V) Pair(V a, V b) { return (a, b); }
    int main() { return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    let table = crate::native::semantic::analyze(&unit).expect("must analyze");
    let cls = table.class("H").expect("H present");
    assert!(
        cls.methods.iter().any(|m| m.name == "Pair" && m.param_types.len() == 2),
        "tuple-return method must be recorded: {:?}",
        cls.methods.iter().map(|m| &m.name).collect::<Vec<_>>()
    );
}

#[test]
fn conditional_call_omits_when_symbol_undefined() {
    // `[Conditional("DEBUG")]` calls omit unless the symbol is
    // defined (C# release semantics): `D.Assert(true)` vanishes
    // (including argument evaluation) while plain calls stay.
    use crate::native::ir::Instr;
    let src = r#"namespace t;
static class D {
    [Conditional("DEBUG")]
    public static void Assert(bool c) { }
    public static void Log(string m) { }
}
class P {
    int main() { D.Assert(true); D.Log("x"); return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    let p = crate::native::ir::lower(&unit).expect("omission must lower");
    let called: Vec<String> = p
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter_map(|i| match i {
            Instr::CallStatic { fidx, .. } => Some(p.functions[*fidx as usize].name.clone()),
            _ => None,
        })
        .collect();
    assert!(
        !called.iter().any(|n| n.contains("::Assert")),
        "conditional call must vanish: {called:?}"
    );
    assert!(
        called.iter().any(|n| n.contains("::Log")),
        "plain call must stay: {called:?}"
    );
}

#[test]
fn conditional_call_keeps_when_symbol_defined() {
    // Same unit with `DEBUG` defined: the conditional call lowers.
    let src = r#"namespace t;
static class D {
    [Conditional("DEBUG")]
    public static void Assert(bool c) { }
}
class P {
    int main() { D.Assert(true); return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    let p = crate::native::ir::lower_with_defines(&unit, true, &["DEBUG".to_string()])
        .expect("defined symbol must lower");
    let calls = p
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(i, crate::native::ir::Instr::CallStatic { .. }))
        .count();
    assert!(calls > 0, "defined symbol must keep the call");
}

#[test]
fn generic_ctor_binds_class_type_var() {
    // `new WeakGCHandle<NativeObjectWrapper>(wrapper)`: the class-level
    // type variable binds for overload applicability, so the public
    // `(T, bool = false)` ctor wins over the private `(IntPtr)` one
    // instead of erroring access on the arity-only fallback pick.
    let src = r#"namespace t;
class Box<T> {
    public Box(T target, bool flag = false) { }
    private Box(int handle) { }
}
class NativeObjectWrapper {
}
class User {
    int M(NativeObjectWrapper w) { Box<NativeObjectWrapper> b = new Box<NativeObjectWrapper>(w); return 0; }
    int main() { return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).expect("generic ctor must bind T");
}

#[test]
fn alias_fanout_across_files() {
    // `using`-aliases are file-scoped in C# but the unit merges files:
    // one name may map to several types (`NativeType` → `IntPtr` in
    // one file, `Double` in another). Selection tries every target's
    // entry — `NativeType.IsFinite` resolves through `Double` even
    // when the `IntPtr` target sorts first.
    let src = r#"using NativeType = System.IntPtr;
using NativeType = System.Double;
namespace System {
    class Double {
        public static bool IsFinite(double x) { return true; }
    }
    struct IntPtr {
    }
}
namespace t {
    class NFloat {
        NativeType _value;
        public static bool IsFinite(NFloat value) { return NativeType.IsFinite(value._value); }
        int main() { return 0; }
    }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).expect("alias fan-out must resolve");
}

#[test]
fn generic_call_result_substitutes_explicit_args() {
    // `Unsafe.As<string[]>(bucket)` types as `string[]`, not the unbound
    // `T`: unqualified overload ranking (`StartsWith(ref char, int,
    // string[])` via `using static`) needs the precise result type.
    let src = r#"using static t.H;
namespace t {
    static class U {
        public static T As<T>(object o) { return default; }
    }
    static class H {
        public static bool Go(ref char x, int y, string[] z) { return true; }
    }
    class User {
        bool M(ref char m, int len, object bucket) { return Go(ref m, len, U.As<string[]>(bucket)); }
        int main() { return 0; }
    }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).expect("substituted result must rank");
}

#[test]
fn generic_spelling_prefers_generic_definition() {
    // `Vec<byte>.Count` resolves through the generic `Vec<T>` entry
    // (arity match), not the same-named non-generic `Vec` statics
    // class (`Vector512<byte>.Count` in CoreLib). Unqualified ranking
    // has no arity-only fallback, so a mistyped (object) argument
    // errors instead of lowering.
    let src = r#"using static t.H;
namespace t {
    static class Vec {
        public static int Other() { return 0; }
    }
    class Vec<T> {
        public static int Count { get { return 4; } }
    }
    static class H {
        public static int Take(int x) { return 1; }
    }
    class User {
        int M() { return Take(Vec<byte>.Count); }
        int main() { return 0; }
    }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).expect("generic static member must type as int");
}

#[test]
fn lowercase_primitive_member_access() {
    // `s.Length` with `s: string` resolves through the `String` entry
    // (primitive spellings canonicalize for table lookup).
    let src = r#"using static t.H;
namespace System {
    class String {
        public int Length { get { return 0; } }
    }
}
namespace t {
    static class H {
        public static void TakeInt(int x) { }
    }
    class User {
        void M(string s) { TakeInt(s.Length); }
        int main() { return 0; }
    }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).expect("string.Length must resolve");
}

#[test]
fn delegate_property_invocation_placeholders() {
    // `info.Factory(...)` where `Factory` is a delegate-typed property
    // (TraceLoggingTypeInfo.PropertyValueFactory in CoreLib): invoke
    // the delegate value as a placeholder.
    let src = r#"namespace t;
delegate object Maker(object o);
class Info {
    Maker Factory { get { return null; } }
}
class User {
    object M(Info info, object o) { return info.Factory(o); }
    int main() { return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).expect("delegate property invocation must lower");
}

#[test]
fn derives_from_terminates_on_cycles() {
    // Base-graph cycles (reachable through per-call string spellings
    // even when `resolve_chain` reports them) must terminate, not
    // overflow the stack mid-build.
    let src = r#"namespace t;
class A {
    int main() { return 0; }
}
class B {
}"#;
    let unit = crate::parser::parse(src).unwrap();
    let mut table = crate::native::semantic::analyze(&unit).expect("analyze");
    for c in &mut table.classes {
        if c.name == "A" {
            c.bases = vec!["B".to_string()];
        } else if c.name == "B" {
            c.bases = vec!["A".to_string()];
        }
    }
    assert!(table.derives_from("A", "A"));
    assert!(table.derives_from("A", "B"));
    assert!(!table.derives_from("A", "Z"));
    assert!(!table.derives_from("B", "Z"));
}

#[test]
fn split_arity_ignores_tuple_commas() {
    // `IReadOnlyCollection<(TElement Element, TPriority Priority)>`
    // has arity 1: tuple element commas never separate generic args
    // (PriorityQueue.UnorderedItemsCollection in System.Collections).
    use crate::native::semantic::split_arity;
    assert_eq!(
        split_arity("IReadOnlyCollection<(TElement Element, TPriority Priority)>"),
        ("IReadOnlyCollection".to_string(), 1)
    );
    assert_eq!(
        split_arity("Dictionary<TKey, TValue>"),
        ("Dictionary".to_string(), 2)
    );
    assert_eq!(
        split_arity("G<A<B<C>>, D>"),
        ("G".to_string(), 2)
    );
}

#[test]
fn explicit_impl_outer_qualified_satisfies() {
    // `object Strategy.IAdapter.Make()` satisfies `IAdapter.Make`
    // (two-level explicit implementation names match by suffix).
    let src = r#"namespace t;
static class Strategy {
    internal interface IAdapter {
        object Make();
    }
}
class ComObject : Strategy.IAdapter {
    object Strategy.IAdapter.Make() { return null; }
    int main() { return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    crate::native::semantic::analyze(&unit).expect("outer-qualified explicit impl must satisfy");
}

#[test]
fn qualified_new_uses_qualified_entry() {
    // `new Pool.WorkStealingQueue()` resolves through the qualified
    // spelling: the entry without declared ctors supplies the implicit
    // default — a same-named sibling's explicit ctor (Portable's) must
    // not force an arity error.
    let src = r#"namespace t;
static class Portable {
    internal sealed class WorkStealingQueue {
        internal WorkStealingQueue(int size) { }
    }
}
static class Pool {
    internal sealed class WorkStealingQueue {
    }
    static class Locals {
        int Make() { object o = new Pool.WorkStealingQueue(); return 0; }
    }
}
class User {
    int main() { return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).expect("qualified new must use its own entry");
}

#[test]
fn interface_derives_transitively() {
    // `IList` → `ICollection` → `IEnumerable` (generic arguments
    // erased for identity, like the rest of applicability).
    let src = r#"namespace t;
interface IEnumerable<T> { }
interface ICollection<T> : IEnumerable<T> { }
interface IList<T> : ICollection<T> { }
class User {
    int main() { return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    let table = crate::native::semantic::analyze(&unit).expect("analyze");
    assert!(table.interface_derives_from("IList", "IEnumerable"));
    assert!(table.interface_derives_from("IList<T>", "IEnumerable<T>"));
    assert!(!table.interface_derives_from("IEnumerable", "IList"));
}

#[test]
fn extension_on_interface_base_accepts_derived() {
    // Extension methods on `IEnumerable<T>` (`TryGetSpan` in Linq)
    // accept `IList<T>` receivers via interface inheritance.
    let src = r#"using static t.H;
namespace t {
    interface IEnumerable<T> { }
    interface ICollection<T> : IEnumerable<T> { }
    interface IList<T> : ICollection<T> { }
    static class H {
        public static bool Go<T>(this IEnumerable<T> x) { return true; }
    }
    class User {
        bool M(IList<int> l) { return l.Go(); }
        int main() { return 0; }
    }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).expect("interface extension must apply");
}

#[test]
fn union_into_appends_non_matches() {
    // `union_into` must return non-matching members for appending —
    // never silently drop them (a `?`-conflation once discarded
    // thousands of dependency members, emptying CoreLib from merged
    // units). Matching entries merge instead.
    use crate::driver::{top_level_type_key, union_into};
    let parse = |src: &str| {
        let pre = crate::preprocessor::preprocess(src, &[], "t.cs").expect("preprocess");
        crate::parser::parse(&pre.source).expect("parse")
    };
    let main_u = parse(r#"namespace System { class Foo { int a; } }"#);
    let dep_u = parse(r#"namespace System { class Bar { int b; } class Foo { int c; } }"#);
    let mut combined: Vec<crate::models::Member> = vec![];
    for m in &main_u.namespaces[0].members {
        combined.push(m.clone());
    }
    let mut appended = 0;
    let mut merged = 0;
    let main_keys: std::collections::HashSet<(String, char, String, usize)> = Default::default();
    let mut deferred = Vec::new();
    for m in &dep_u.namespaces[0].members {
        match union_into(&mut combined, "System", m.clone(), &main_keys, &mut deferred) {
            None => merged += 1,
            Some(mm) => {
                appended += 1;
                combined.push(mm);
            }
        }
    }
    assert_eq!((appended, merged), (1, 1), "Bar appends, Foo merges");
    assert_eq!(combined.len(), 2, "nothing is lost");
    // The merged Foo keeps main's `a` and gains dep's `c`.
    let foo = combined
        .iter()
        .find(|m| m.name() == "Foo")
        .expect("Foo present");
    if let crate::models::Member::Class(c) = foo {
        let fields: Vec<String> = c
            .members
            .iter()
            .filter_map(|cm| match cm {
                crate::models::ClassMember::Field(f) => Some(f.name.clone()),
                _ => None,
            })
            .collect();
        assert!(fields.contains(&"a".to_string()), "main field kept: {fields:?}");
        assert!(fields.contains(&"c".to_string()), "dep field merged: {fields:?}");
    } else {
        panic!("Foo must stay a class");
    }
    // Sanity: keys used above are well-formed.
    assert!(top_level_type_key("System", foo).is_some());
}

#[test]
fn union_iface_adopts_weaker_bases() {
    // Same-key interfaces with differing bases (a polyfill stub vs the
    // real view, e.g. Numerics' `IEquatable`-based `IUtfChar<TSelf>` vs
    // CoreLib's `IBinaryInteger`-based one): methods union immediately,
    // bases resolve post-load to the weaker view, so every implementer
    // in the unit still satisfies the contract. Order-independent: the
    // wide view loads first below (dependencies load sorted).
    use crate::driver::{resolve_deferred_ifaces, union_into};
    use std::collections::HashSet;
    let parse = |src: &str| {
        let pre = crate::preprocessor::preprocess(src, &[], "t.cs").expect("preprocess");
        crate::parser::parse(&pre.source).expect("parse")
    };
    let wide = parse(r#"namespace System {
        interface IWide : IBig { int M(); }
        interface IBig { int A(); int B(); int C(); }
    }"#);
    let narrow = parse(r#"namespace System {
        interface IWide : ISmall { int M(); }
        interface ISmall { int A(); }
    }"#);
    for (first, second) in [(&wide, &narrow), (&narrow, &wide)] {
        let mut combined: Vec<crate::models::Member> = vec![];
        for m in &first.namespaces[0].members {
            combined.push(m.clone());
        }
        let main_keys: HashSet<(String, char, String, usize)> = Default::default();
        let mut deferred = Vec::new();
        for m in &second.namespaces[0].members {
            match union_into(&mut combined, "System", m.clone(), &main_keys, &mut deferred) {
                None => {}
                Some(mm) => combined.push(mm),
            }
        }
        assert_eq!(deferred.len(), 1, "exactly the IWide collision defers");
        let mut shell = parse("namespace System { class Anch { } }");
        shell.namespaces[0].members = combined;
        resolve_deferred_ifaces(&mut shell, &deferred);
        let entry = shell.namespaces[0]
            .members
            .iter()
            .find(|m| m.name() == "IWide")
            .expect("IWide present");
        if let crate::models::Member::Interface(i) = entry {
            assert_eq!(
                i.bases,
                vec!["ISmall".to_string()],
                "weaker bases win either order: {:?}",
                i.bases
            );
            assert!(i.methods.iter().any(|m| m.name == "M"), "methods unioned");
        } else {
            panic!("IWide must stay an interface");
        }
    }
}

#[test]
fn union_iface_main_bases_absolute() {
    // A main-project entry keeps its interface bases absolutely (C#
    // CS0436: source wins over reference), even against a weaker dep.
    use crate::driver::{resolve_deferred_ifaces, union_into};
    use std::collections::HashSet;
    let parse = |src: &str| {
        let pre = crate::preprocessor::preprocess(src, &[], "t.cs").expect("preprocess");
        crate::parser::parse(&pre.source).expect("parse")
    };
    let main_u = parse(r#"namespace System { interface IW : IB { int M(); } }"#);
    let dep_u = parse(r#"namespace System { interface IW : IS { int M(); } }"#);
    let mut combined: Vec<crate::models::Member> = vec![];
    for m in &main_u.namespaces[0].members {
        combined.push(m.clone());
    }
    let mut main_keys: HashSet<(String, char, String, usize)> = Default::default();
    main_keys.insert(("System".to_string(), 'I', "IW".to_string(), 0));
    let mut deferred = Vec::new();
    for m in &dep_u.namespaces[0].members {
        match union_into(&mut combined, "System", m.clone(), &main_keys, &mut deferred) {
            None => {}
            Some(mm) => combined.push(mm),
        }
    }
    assert_eq!(deferred.len(), 1, "collision still defers");
    let mut shell = parse("namespace System { class Anch { } }");
    shell.namespaces[0].members = combined;
    resolve_deferred_ifaces(&mut shell, &deferred);
    let entry = shell.namespaces[0]
        .members
        .iter()
        .find(|m| m.name() == "IW")
        .expect("IW present");
    if let crate::models::Member::Interface(i) = entry {
        assert_eq!(i.bases, vec!["IB".to_string()], "main bases kept: {:?}", i.bases);
    } else {
        panic!("IW must stay an interface");
    }
}

#[test]
fn open_type_param_satisfies_conformance() {    // An interface method parameter naming no known type is an open
    // generic variable (from an enclosing scope upstream): it matches
    // any concrete implementation type instead of E0016.
    let src = r#"namespace t;
interface IAdapter {
    object Make(T item);
}
class Impl : IAdapter {
    public object Make(string item) { return item; }
    int main() { return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    crate::native::semantic::analyze(&unit).expect("open type param must satisfy");
}

#[test]
fn extension_on_interface_accepts_implementing_class() {
    // Extension methods on `IEnumerable<T>` (`TryGetSpan` in Linq)
    // accept receivers whose class transitively implements it
    // (`Dictionary<K,V>`), not just interface-typed values.
    let src = r#"using static t.H;
namespace t {
    interface IEnumerable<T> { }
    interface ICollection<T> : IEnumerable<T> { }
    class Dict : ICollection<int> { }
    static class H {
        public static bool Go(this IEnumerable<int> x) { return true; }
    }
    class User {
        bool M(Dict d) { return d.Go(); }
        int main() { return 0; }
    }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).expect("class-to-interface conversion must apply");
}

#[test]
fn inherited_dim_satisfies_conformance() {
    // A default implementation inherited through the interface
    // hierarchy satisfies conformance (`IArrayElementMarshaler`
    // defaults `IArrayMarshaler.AllocateSpaceForManaged` for all its
    // implementers): no E0015.
    let src = r#"namespace t;
interface IBase {
    int Get(int x);
}
interface IDerived : IBase {
    int Get(int x) { return x; }
}
class Impl : IDerived {
    int main() { return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    crate::native::semantic::analyze(&unit).expect("inherited DIM must satisfy");
}

#[test]
fn generic_alias_prefers_generic_definition() {
    // `using WorkQueue = ...Queue<object>` resolves to generic
    // `Queue<T>`, not same-named non-generic `Queue` (ThreadPool
    // work queues in CoreLib).
    let src = r#"using WorkQueue = System.Collections.Generic.Queue<object>;
namespace System.Collections.Generic {
    class Queue {
        public bool Legacy() { return true; }
    }
    class Queue<T> {
        public bool TryDequeue(out T result) { result = default; return true; }
    }
}
namespace t {
    class User {
        int main() { return 0; }
    }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    let table = crate::native::semantic::analyze(&unit).expect("analyze");
    let e = table.class("WorkQueue").expect("alias resolves");
    assert_eq!(e.key.arity, 1, "must be generic Queue<T>");
    assert!(
        e.methods.iter().any(|m| m.name == "TryDequeue"),
        "generic entry carries TryDequeue"
    );
}

#[test]
fn qualified_enum_keys_resolve() {
    // Enum entries are keyed by qualified path like every other type,
    // so qualified spellings (`System.Threading.StackCrawlMark`) hit
    // them instead of erroring as unsupported variable types
    // (`Environment.FailFast` in CoreLib).
    let src = r#"namespace System.Threading {
    enum StackCrawlMark { LookForMe = 0, LookForMyCaller = 1 }
}
namespace t {
    class E {
        static void M() {
            System.Threading.StackCrawlMark mark = System.Threading.StackCrawlMark.LookForMyCaller;
        }
        int main() { return 0; }
    }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    let table = crate::native::semantic::analyze(&unit).expect("analyze");
    assert!(
        table
            .class("System.Threading.StackCrawlMark")
            .is_some_and(|c| c.is_enum),
        "qualified enum spelling must hit"
    );
    crate::native::ir::lower(&unit).expect("enum-typed local must lower");
}

#[test]
fn bare_method_group_in_cast_parses() {
    // `((Func<object[]>)Array.Empty<object>).GetMethodInfo()` — a bare
    // generic method group (no invocation) as a cast operand. The `<>`
    // must parse as generics, not a less-than comparison.
    let src = r#"namespace t;
delegate object D(object[] a);
class Arr {
    public static T[] Empty<T>() { return null; }
}
class User {
    object M() { return ((D)Arr.Empty<object>).ToString(); }
    int main() { return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).expect("method group cast must lower");
}

#[test]
fn unqualified_delegate_property_invocation_placeholders() {
    // `Impl(o, s)` where `Impl` is a delegate-typed property
    // (CookieParser.InternalSetNameMethod in its non-Primitives form):
    // invoke the delegate value as a placeholder.
    let src = r#"namespace t;
delegate bool Filter(object o, string s);
class Parser {
    Filter Impl { get { return null; } }
    bool Get(object o) { return Impl(o, "s"); }
    int main() { return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).expect("delegate property call must lower");
}

#[test]
fn interpolated_string_converts_to_handler() {
    // `$"{x}"` converts to interpolated-string-handler parameters
    // (`Span.TryWrite(..., ref TryWriteInterpolatedStringHandler, ...)`
    // in CoreLib): accept with below-exact score so precise overloads
    // still win.
    let src = r#"using static t.H;
namespace t {
    struct GoInterpolatedStringHandler {
    }
    static class H {
        public static bool Go(ref char x, GoInterpolatedStringHandler h, out int n) { n = 0; return true; }
    }
    class User {
        bool M(ref char m, int v) { int n; return Go(ref m, $"{v}", out n); }
        int main() { return 0; }
    }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).expect("handler conversion must apply");
}

#[test]
fn base_init_out_vars_scope_to_body() {
    // `out` variables declared in a `: base(...)` argument list are in
    // scope for the constructor body (`: base(M(p, out var u, ...))`
    // then `u` below; MLKemOpenSsl ctor in CoreLib).
    let src = r#"namespace t;
class Base {
    public Base(int x, out int u) { u = x; }
}
class Derived : Base {
    public Derived(int p) : base(p, out int u) {
        int v = u;
    }
    int main() { return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).expect("base-init out-vars must scope");
}

#[test]
fn nested_helper_does_not_shadow_top_level_base() {
    // A nested helper type must not win a bare base-name lookup by
    // sort order: `Derived : RealBase` binds the top-level
    // `System.RealBase` (with its members inherited), not the
    // unrelated nested `Other.Holder.RealBase` that sorts first
    // (cf. `DelegateHelpers.TypeInfo` shadowing
    // `System.Reflection.TypeInfo` for `RuntimeType`'s chain).
    let src = r#"namespace System {
    public class RealBase {
        internal int Root() { return 1; }
    }
}
namespace Other {
    public class Holder {
        internal sealed class RealBase {
        }
    }
    public class Derived : RealBase {
    }
    class P {
        int main() { return 0; }
    }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    let table = crate::native::semantic::analyze(&unit).expect("must analyze");
    let cls = table.class("Derived").expect("Derived present");
    assert!(
        cls.methods.iter().any(|m| m.name == "Root"),
        "top-level base members must inherit through the chain: {:?}",
        cls.methods.iter().map(|m| &m.name).collect::<Vec<_>>()
    );
}

#[test]
fn at_this_param_shadows_receiver() {
    // `@this` declares an ordinary parameter (cf.
    // `UpdateAndExecute1(CallSite @this, ...)`): uses type and
    // evaluate as the slot, never the enclosing class/receiver
    // (static methods don't even have one).
    let src = r#"namespace t;
class Site {
    internal int Get() { return 1; }
}
static class U {
    public static int Go(Site @this) { return @this.Get(); }
    int main() { return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).expect("@this must bind the parameter");
}

#[test]
fn partial_interface_parts_merge() {
    // `IDtdParserAdapter` across sync/async files (System.Private.Xml):
    // two `partial` parts merge members instead of E0012.
    let src = r#"namespace t {
partial interface IDtdParserAdapter { void A(); }
partial interface IDtdParserAdapter { void B(); }
class C { int main() { return 0; } }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    let table =
        crate::native::semantic::analyze(&unit).expect("partial interfaces should merge");
    let iface = table
        .interfaces
        .iter()
        .find(|i| i.name == "IDtdParserAdapter")
        .expect("merged interface present");
    assert_eq!(iface.methods.len(), 2);
}

#[test]
fn nonpartial_interface_duplicate_still_errors() {
    // Same name twice without `partial`: genuine duplicate (E0012),
    // not a merge — the guard that catches double-staged files.
    let src = r#"namespace t {
interface IDup { void A(); }
interface IDup { void B(); }
class C { int main() { return 0; } }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    let r = crate::native::semantic::analyze(&unit);
    assert!(r.is_err(), "non-partial duplicate should fail");
}

#[test]
fn file_scoped_alias_does_not_close_type_param() {
    // `using T = XmlQueryTypeFactory;` (System.Private.Xml XSLT files)
    // is file-scoped: elsewhere `T` stays an open generic variable,
    // so `FindItemIndex(string)` still satisfies `(T)` (frozen
    // string/int sets vs `IGenericSpecializedWrapper`). Regression:
    // the alias-resolving lookup made `T` concrete globally (E0016).
    let src = r#"namespace t {
using T = System.Xml.Xsl.XmlQueryTypeFactory;
class XmlQueryTypeFactory { }
class FrozenSet<T> {
    internal interface IWrap { int FindItemIndex(T item); }
    internal struct GSW : IWrap {
        public int FindItemIndex(string item) { return 0; }
    }
}
class C { int main() { return 0; } }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    crate::native::semantic::analyze(&unit).expect("alias T must not close the type param");
}

#[test]
fn owner_attribution_exact_and_nested() {
    // Dynamic-linking ownership: top-level entries attribute by exact
    // key hit; nested types inherit their file's lib through the
    // parent chain; unrecorded entries stay "" (treated as local).
    // The driver records main-project members first so merged
    // main+dep entries attribute to main (mirroring content main-wins).
    let src = r#"namespace t {
class Outer {
    class Inner { }
}
}
namespace u {
class Solo { }
}
namespace t {
enum E { A }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    let owners = std::collections::HashMap::from([
        (TypeKey::new("t.Outer", 0), "LibA".to_owned()),
        (TypeKey::new("u.Solo", 0), "LibB".to_owned()),
        (TypeKey::new("t.E", 0), "LibA".to_owned()),
    ]);
    let table = crate::native::semantic::analyze_with_owners(&unit, &owners)
        .expect("ownership probe unit must analyze");
    let owner = |path: &str| {
        table
            .classes
            .iter()
            .find(|c| c.key.path == path)
            .unwrap_or_else(|| panic!("missing entry {path}"))
            .owner
            .clone()
    };
    assert_eq!(owner("t.Outer"), "LibA");
    assert_eq!(owner("t.Outer.Inner"), "LibA", "nested inherits file lib");
    assert_eq!(owner("u.Solo"), "LibB");
    assert_eq!(owner("t.E"), "LibA");
}

#[test]
fn vtable_slots_order_independent() {
    // Cross-image agreement requires slots independent of table
    // content order: the same types declared in different file orders
    // must assign identical slots (global first-seen failed this —
    // per-class positional passes by construction).
    let a = crate::parser::parse(
        "namespace t; class X { public virtual int M() { return 1; } public virtual int N() { return 2; } } class Y { public virtual int P() { return 3; } }",
    )
    .unwrap();
    let b = crate::parser::parse(
        "namespace t; class Y { public virtual int P() { return 3; } } class X { public virtual int M() { return 1; } public virtual int N() { return 2; } }",
    )
    .unwrap();
    let slots = |unit: &crate::models::CompilationUnit| {
        let table = crate::native::semantic::analyze(unit).unwrap();
        let mut out = std::collections::BTreeMap::new();
        for c in &table.classes {
            for m in &c.methods {
                out.insert((c.key.path.clone(), m.name.clone()), m.vtable_slot);
            }
        }
        out
    };
    assert_eq!(slots(&a), slots(&b), "declaration order must not shift slots");
}

#[test]
fn vtable_slots_derived_share_base_prefix() {
    // Derived rows share their bases' prefix: every (name, slot) in
    // the base entry must appear identically in the derived entry, so
    // a base-typed call lands correctly on derived objects (and
    // cross-image, where the base side comes from `.rci`).
    let src = "namespace t; class A { public virtual int M() { return 1; } public virtual int N() { return 2; } } class B : A { public override int M() { return 3; } public virtual int P() { return 4; } }";
    let unit = crate::parser::parse(src).unwrap();
    let table = crate::native::semantic::analyze(&unit).unwrap();
    let get = |path: &str| {
        table
            .classes
            .iter()
            .find(|c| c.key.path == path)
            .unwrap_or_else(|| panic!("missing {path}"))
    };
    let a = get("t.A");
    let b = get("t.B");
    for m in &a.methods {
        let shared = b
            .methods
            .iter()
            .find(|x| x.name == m.name)
            .unwrap_or_else(|| panic!("derived missing {}", m.name));
        assert_eq!(
            shared.vtable_slot, m.vtable_slot,
            "derived must share base slot for {}",
            m.name
        );
    }
}

#[test]
fn preload_merges_fragment_verbatim() {
    // Dynamic-linking consume: a main-only unit plus a dependency
    // fragment must reproduce the full-source table for overlapping
    // entries — inherited members verbatim (offsets/slots/values),
    // const fixpoint seeded, chains resolved through the dep boundary.
    let full_src = "namespace t; class Base { public int F; public const int K = 7; public virtual int M() { return 1; } public int P { get { return 1; } } } class Derived : Base { public int G; }";
    let full_unit = crate::parser::parse(full_src).unwrap();
    let full = crate::native::semantic::analyze(&full_unit).unwrap();
    let base_full = full
        .classes
        .iter()
        .find(|c| c.key.path == "t.Base")
        .expect("full table has Base")
        .clone();
    // Simulate a dep build: Base owned by DepLib.
    let mut base_frag = base_full.clone();
    base_frag.owner = "DepLib".to_owned();
    let bytes = crate::native::rci::encode("DepLib", 1, &[base_frag], &[], &[]);
    let frag = crate::native::rci::decode(&bytes).expect("fragment decodes");
    assert_eq!(frag.classes.len(), 1);

    // Main-only unit (Base sources absent).
    let main_src = "namespace t; class Derived : Base { public int G; }";
    let main_unit = crate::parser::parse(main_src).unwrap();
    let owners = std::collections::HashMap::from([(
        crate::native::semantic::TypeKey::new("t.Derived", 0),
        "MainLib".to_owned(),
    )]);
    let pre = crate::native::semantic::Preloaded {
        classes: frag.classes,
        interfaces: frag.interfaces,
        delegates: frag.delegates,
    };
    let table =
        crate::native::semantic::analyze_with_preloaded(&main_unit, &owners, &pre)
            .expect("preload merge must analyze");
    // Dep entry present verbatim with its owner.
    let base = table
        .classes
        .iter()
        .find(|c| c.key.path == "t.Base")
        .expect("preloaded Base present");
    assert_eq!(base.owner, "DepLib");
    // Derived inherits verbatim: field offset, const value, method
    // slot, property type all match the full-source build.
    let derived = table
        .classes
        .iter()
        .find(|c| c.key.path == "t.Derived")
        .expect("compiled Derived present");
    assert_eq!(derived.owner, "MainLib");
    let derived_full = full
        .classes
        .iter()
        .find(|c| c.key.path == "t.Derived")
        .expect("full table has Derived");
    for f in &derived_full.fields {
        let got = derived
            .fields
            .iter()
            .find(|x| x.name == f.name)
            .unwrap_or_else(|| panic!("missing field {}", f.name));
        assert_eq!(got.offset_slots, f.offset_slots, "field {}", f.name);
        assert_eq!(got.type_name, f.type_name, "field {}", f.name);
    }
    for m in &derived_full.methods {
        let got = derived
            .methods
            .iter()
            .find(|x| x.name == m.name && x.param_types == m.param_types)
            .unwrap_or_else(|| panic!("missing method {}", m.name));
        assert_eq!(got.vtable_slot, m.vtable_slot, "method {}", m.name);
    }
    for c in &derived_full.const_fields {
        let got = derived
            .const_fields
            .iter()
            .find(|x| x.name == c.name)
            .unwrap_or_else(|| panic!("missing const {}", c.name));
        assert_eq!(got.const_value, c.const_value, "const {}", c.name);
    }
    for p in &derived_full.properties {
        let got = derived
            .properties
            .iter()
            .find(|x| x.name == p.name)
            .unwrap_or_else(|| panic!("missing property {}", p.name));
        assert_eq!(got.type_name, p.type_name, "property {}", p.name);
    }
}
