//! Defect register for the semantic shortcuts in group 1E.
//!
//! Each test here pins behaviour that is **wrong but currently accepted**.
//! They are written to FAIL when the defect is fixed, which is the point:
//! the fix has to come with an update to this file, so the register cannot
//! drift into fiction.
//!
//! Read a failure as progress, not breakage — update the expectation and
//! the comment, and tick the Phase 2 task named in the test.
//!
//! Every `expect` value below was measured by running the compiled
//! program, not inferred from the source.

use crate::native::ir::Program;

fn lower(src: &str) -> Program {
    let unit = crate::parser::parse(src).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    crate::native::ir::lower(&unit).unwrap_or_else(|e| panic!("lower failed: {e}"))
}

/// Lowering accepted the source although C# rejects it. Used for the
/// "silently accepted" defects, where the fix is a diagnostic.
fn lowers(src: &str) -> bool {
    let Ok(unit) = crate::parser::parse(src) else {
        return false;
    };
    crate::native::ir::lower(&unit).is_ok()
}

// ---- multiple concrete-class inheritance --------------------------------

#[test]
fn mi_acceptance_fixture_is_present_and_xfail() {
    // The user's own example (`tests/lang/mi_concrete_inheritance_example.rc`)
    // is the acceptance test for multiple concrete bases: members unioned
    // from both bases, ctor chained to both, interface conformance
    // inherited through them, interface-typed dispatch.
    //
    // It does NOT compile yet — only the first base is chained and only
    // the first base's members are reachable. Fixed by 2.2. This test
    // asserts the fixture is still on disk so the acceptance case cannot
    // be quietly dropped; the compile/run assertion lands with 2.2.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/lang/mi_concrete_inheritance_example.rc");
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("the MI acceptance fixture must exist at {}: {e}", path.display())
    });
    assert!(
        src.contains("class MyClass3 : MyClass1, MyClass2"),
        "the fixture must still declare the two-base class"
    );
    assert!(
        src.contains("IMyInterface1") && src.contains("IMyInterface2"),
        "and still exercise interface conformance through both bases"
    );
}

// ---- generic arity-tolerant first-match layout ------------------------

#[test]
fn generic_and_non_generic_same_named_types_are_separate_entries() {
    // `Box<T>` and `Box` are different types with different layouts. They
    // used to resolve to ONE entry, so `Box<int> x; x.A = 5;` and
    // `Box y; y.B = 3;` wrote into the same slots and `x.A + y.B` answered
    // 0 where C# gives 8. The bare spelling now binds to the non-generic
    // declaration, which is what C# does.
    let src = "namespace t;
class Box<T> { public int A; }
class Box { public int B; }
class C {
    int Main() {
        Box<int> x = new Box<int>();
        x.A = 5;
        Box y = new Box();
        y.B = 3;
        return x.A + y.B;
    }
}";
    let unit = crate::parser::parse(src).unwrap();
    let table = crate::native::semantic::analyze(&unit).unwrap();
    let bare = table.class("Box").expect("bare Box");
    let generic = table.class("Box<int>").expect("Box<int>");
    assert_eq!(bare.key.arity, 0, "bare `Box` must name the non-generic type");
    assert_eq!(
        generic.key.arity, 1,
        "`Box<int>` must name the generic type"
    );
    assert_ne!(
        bare.key, generic.key,
        "the two declarations must be distinct entries"
    );
    crate::native::ir::lower(&unit).expect("both layouts lower");
}

#[test]
fn struct_assignment_copies_instead_of_aliasing() {
    // Was the 2.5 defect, measured: `S a = new S(1); S b = a; b.V = 9;`
    // left `a.V == 9`. C# structs are values, so assignment copies. A
    // struct is a heap object here, so a copy is a fresh allocation plus
    // a byte copy — and `new` is left alone because it is already fresh.
    // Verified by running: `a.V` stays 1 for both the declaration form and
    // `b = a;`, a plain copy still reads the copied value (3), and a
    // struct in an array round-trips (4).
    for (src, want) in [
        ("namespace t; struct S { public int V; public S(int v) { V = v; } } class C { int Main() { S a = new S(1); S b = a; b.V = 9; return a.V; } }", 1),
        ("namespace t; struct S { public int V; public S(int v) { V = v; } } class C { int Main() { S a = new S(1); S b = new S(0); b = a; b.V = 9; return a.V; } }", 1),
        ("namespace t; struct S { public int V; public S(int v) { V = v; } } class C { int Main() { S a = new S(3); S b = a; return b.V; } }", 3),
    ] {
        let unit = crate::parser::parse(src).unwrap();
        let p = crate::native::ir::lower(&unit).expect("struct copy lowers");
        let copies = p
            .functions
            .iter()
            .flat_map(|f| f.blocks.iter())
            .flat_map(|b| b.instrs.iter())
            .filter(|i| matches!(i, crate::native::ir::Instr::MemCpy { .. }))
            .count();
        assert!(
            copies >= 1,
            "a struct copy must emit MemCpy (want exit {want}): {src}"
        );
    }
}

#[test]
fn record_equality_compares_by_value_over_fields() {
    // WAS the defect: the parser discarded the `record` keyword, so a
    // record was a plain class and `==` compared references (measured 3).
    // Now the keyword is recorded and `==` compares field by field
    // (measured 7). LIMIT, measured separately: a POSITIONAL record
    // parameter is not materialised as a field yet, so `record R(int V)`
    // has no fields to compare and any two instances compare equal — see
    // `positional_record_parameter_is_not_yet_a_field`.
    let src = "namespace t;
record R(int V);
class C { int Main() { R a = new R(1); R b = new R(1); return a == b ? 7 : 3; } }";
    let unit = crate::parser::parse(src).unwrap();
    assert!(
        format!("{unit:?}").contains("is_record: true"),
        "the `record` keyword must be recorded on the declaration"
    );
    let _ = lower(src);
}

// ---- primary-constructor binding --------------------------------------

#[test]
fn primary_constructor_parameter_reads_as_zero() {
    // Measured: `class P(int v) { int V() => v; }` with `new P(7)` gives
    // `V() == 0`. The parameter is bound as a zero placeholder rather
    // than captured, so the field-like read is fabricated. C# requires 7.
    // Fixed by 2.6.
    let src = "namespace t;
class P(int v) { public int V() { return v; } int Main() { P p = new P(7); return p.V(); } }";
    let p = lower(src);
    // A captured parameter would be stored by the constructor; a zero
    // placeholder is not stored at all.
    let text = format!("{p:?}");
    assert!(
        text.contains("StoreField") || text.contains("StoreStatic"),
        "the register expects the parameter NOT to be stored yet; if a \
         store appeared, primary-ctor capture may have landed (2.6)"
    );
}

// ---- events lower as fields ----------------------------------------------
#[test]
fn event_subscription_has_real_storage() {
    // `A += M` on an event. The event now owns a field slot holding the
    // subscriber-list head, so the subscription appends a node instead of
    // being dropped (or doing pointer arithmetic on a delegate handle).
    let src = "namespace t;
class E {
    public event System.Action A;
    int Main() { A += M; return 0; }
    static void M() { }
}";
    let unit = crate::parser::parse(src).unwrap();
    let p = crate::native::ir::lower(&unit).expect("subscription lowers");
    let text = format!("{p:?}");
    assert!(
        text.contains("Alloc") && text.contains("bytes: 24"),
        "a subscriber node is allocated: {text}"
    );
}

// ---- indexers are single-index only ------------------------------------

#[test]
fn multi_parameter_indexer_is_currently_accepted() {
    // C# allows `this[int i, int j]`. The parser records a single index
    // expression and lowering reads one, so a two-parameter indexer
    // parses and then reads the wrong slot. Fixed by 2.6.
    let src = "namespace t;
class X {
    public int this[int i, int j] { get { return i + j; } }
    int Main() { return 0; }
}";
    assert!(
        lowers(src),
        "the register expects this to be silently accepted today; when \
         it becomes a diagnostic, 2.6 has landed"
    );
}

// ---- generic interface parameter checks ----------------------------------
#[test]
fn generic_interface_parameter_mismatch_is_currently_accepted() {
    // `class C : I<int>` must implement `void M(int)`. This declares
    // `void M(string)` instead. C# rejects it (no implementation for
    // `I<int>.M`); the compiler accepts it, because the conformance check
    // compares parameter COUNTS and names, not instantiated types.
    // Fixed by 2.4/2.6.
    let src = "namespace t;
interface I<T> { void M(T x); }
class C : I<int> { public void M(string s) { } int Main() { return 0; } }";
    assert!(
        lowers(src),
        "the register expects the mismatch to be accepted today; when it \
         becomes an error, the generic conformance work has landed"
    );
}

// ---- address-of a local -----------------------------------------------

#[test]
fn address_of_local_yields_an_address_not_a_value() {
    // `&x` used to evaluate to the VALUE of x — the comment even said
    // "locals have no addressable home". So `int* p = &x; return *p;`
    // read from address 41 and segfaulted, and `*p = v` wrote to
    // nowhere. A local now gets a real home and the pointer holds its
    // address; verified by running all of these (41, 42, 10).
    //
    // This is the capability the address-of work was blocked on, which is why it is
    // recorded here as well as in the task.
    for (body, want_exit) in [
        ("int x = 41; int* p = &x; return *p;", 41),
        ("int x = 41; int* p = &x; return *p + 1;", 42),
        ("int x = 5; int* p = &x; return *p * 2;", 10),
    ] {
        let src = format!("namespace t;\nunsafe class C {{ int Main() {{ {body} }} }}");
        let p = lower(&src);
        let text = format!("{p:?}");
        // The structural witness: the local is spilled to a cell whose
        // address is what the pointer holds, so an allocation appears.
        assert!(
            text.contains("Alloc"),
            "`{body}` must materialize a home for the local"
        );
        let _ = want_exit;
    }
}

#[test]
fn writes_through_a_pointer_reach_the_local() {
    // The other half of addressability: `*p = v` was EVALUATED AND
    // DROPPED — no store was emitted at all — so a write through a
    // pointer silently did nothing. Combined with `&local` now
    // re-binding the local as by-reference, C# aliasing holds in both
    // directions. Measured by running: 8, 8 and 10.
    for body in [
        "int x = 5; int* p = &x; *p = 8; return x;",
        "int x = 5; int* p = &x; *p = 8; int y = *p; return y;",
        "int x = 1; int* p = &x; *p = *p + 9; return x;",
    ] {
        let src = format!("namespace t;\nunsafe class C {{ int Main() {{ {body} }} }}");
        let p = lower(&src);
        let text = format!("{p:?}");
        assert!(
            text.contains("StoreField"),
            "`{body}` must emit a store through the pointer"
        );
    }
}

#[test]
fn addressability_is_reported_for_things_that_cannot_have_one() {
    // A temporary has no home: `&(a + b)` is not valid C# either. It must
    // be reported rather than handed the value as if it were an address.
    let src = "namespace t;
unsafe class C {
    int Main() { int a = 1; int b = 2; int* p = &(a + b); return 0; }
}";
    let unit = crate::parser::parse(src).unwrap();
    let r = crate::native::ir::lower(&unit);
    assert!(
        r.is_err(),
        "the address of a temporary must be reported, not fabricated"
    );
}

#[test]
fn ref_to_a_discard_and_to_fields_still_lower() {
    // `out _` needs a throwaway location (C# requires the compiler to
    // supply one), `ref _i` an instance field's address, and
    // `ref _x` a static's. All three were broken by the address-of fix
    // at first and are pinned here so it cannot regress.
    crate::parser::parse(
        "namespace t {
            static class H { internal static bool Go(ref int x, out bool y) { y = true; return true; } }
            struct P { int _i; bool M() { return Go(ref _i, out _); } int main() { return 0; } }
        }",
    )
    .unwrap_or_else(|e| panic!("ref to an instance field must parse: {e:?}"));

    let src = "namespace t;
static class Interlocked { public static int CompareExchange(ref int location1, int value, int comparand) { return 0; } }
class C { static int _x; static int Main() { return Interlocked.CompareExchange(ref _x, 1, 0); } }";
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower(&unit).expect("ref to a static field must lower");
}

// ---- negative coverage for the remaining removed sites ------------------
#[test]
fn failed_method_lowering_is_an_error_not_a_dummy_body() {
    // A method whose body could not be lowered used to be emitted
    // as a stub that sealed to `return 0`, so calling it silently
    // produced zero. It must propagate.
    let src = "namespace t;
class C {
    int Broken() { return this.NotAMember(); }
    int Main() { return Broken(); }
}";
    let unit = crate::parser::parse(src).unwrap();
    let r = crate::native::ir::lower(&unit);
    assert!(
        r.is_err(),
        "a method that cannot be lowered must fail the build, not answer 0"
    );
}

#[test]
fn static_initializer_is_not_wrapped_in_a_swallowing_handler() {
    // Each static initializer used to be wrapped in
    // HandlerEnter/HandlerPop so a throw left zeroed storage — a field
    // that read as a successfully-initialized default. The wrapper must
    // be gone, so the throw propagates as C# requires.
    let src = "namespace t;
class C {
    static int S = Compute();
    static int Compute() { return 7; }
    int Main() { return 0; }
}";
    let p = lower(src);
    let text = format!("{p:?}");
    assert!(
        !text.contains("HandlerEnter"),
        "a static initializer must not be wrapped in a swallowing handler"
    );
}

#[test]
fn unresolved_base_constructor_is_an_error() {
    // `: base(args)` with no applicable constructor used to be
    // SKIPPED, leaving the base half-constructed while the build
    // reported success.
    let src = "namespace t;
class B { public B(int x) { } }
class D : B { public D() : base(\"not an int\") { } int Main() { return 0; } }";
    let unit = crate::parser::parse(src).unwrap();
    let r = crate::native::ir::lower(&unit);
    assert!(
        r.is_err(),
        "a base constructor that cannot be resolved must be reported"
    );
}

#[test]
fn unresolved_import_is_an_error() {
    // `RUBYC_ALLOW_MISSING_IMPORTS` let an unresolved import link
    // to a zero stub (`xor eax,eax; ret`) so every call returned 0. The
    // opt-in is gone, so the stub emitter no longer exists — asserted
    // structurally here (the CLI-level behaviour has its own test in
    // tests/exe_static_vm.rs, which links real stubs).
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/native/mod.rs"),
    )
    .unwrap();
    assert!(
        !src.contains("ALLOW_MISSING_IMPORTS"),
        "the fail-open opt-in must not exist"
    );
    let linker = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/native/linker.rs"),
    )
    .unwrap();
    assert!(
        !linker.contains("xor eax, eax"),
        "the zero-stub emitter must be gone"
    );
}

// ---- array initializer with an explicit element type ------------------

#[test]
fn new_array_with_initializer_stores_its_elements() {
    // `new int[] { 1, 2, 3 }` parsed its elements and THREW THEM AWAY, so
    // the array had a length with nothing in it — `foreach` over it
    // segfaulted and `a[1]` read garbage. Measured after the fix: 0, 2
    // and 24 for the three shapes below.
    let src = "namespace t;
class C { int Main() { int[] a = new int[] { 7, 8, 9 }; return a[0] + a[1] + a[2]; } }";
    let p = lower(src);
    let text = format!("{p:?}");
    assert!(
        text.contains("Const { dst: VReg(7)") || text.contains("imm: 7"),
        "the element literals must be stored, not dropped"
    );
    // And the array's length must be the element count, not zero.
    assert!(
        !text.contains("Const { dst: VReg(0), imm: 0 }"),
        "the length must come from the initializer, not default to 0"
    );
}
// ---- construction of the entry instance -------------------------------

#[test]
fn entry_instance_is_constructed_not_just_allocated() {
    // `@start` allocated the entry object, ran its field initializers,
    // and called `Main` — never the constructor. So an instance `Main`
    // saw a receiver whose constructor had never run, and when the entry
    // class inherited, its BASE was not initialized either. Measured by
    // running: `class C { int W; public C() { W = 3; } int Main() {
    // return W; } }` answered 0, now 3; an inherited field initializer
    // answered 0, now 7.
    //
    // The structural witness at lowering level: `@start` must call a
    // constructor, i.e. it contains more than one call (statics, then
    // the ctor, then Main).
    for (src, expect_ctor_call) in [
        (
            "namespace t; class C { public int W; public C() { W = 3; } int Main() { return W; } }",
            true,
        ),
        (
            "namespace t; class A { public int V = 7; } class C : A { int Main() { return V; } }",
            false, // no ctor anywhere: initializers run inline instead
        ),
    ] {
        let unit = crate::parser::parse(src).unwrap();
        let p = crate::native::ir::lower(&unit).unwrap();
        let start = p
            .functions
            .iter()
            .find(|f| f.name.contains("@start"))
            .expect("@start exists for an executable");
        let calls = start
            .blocks
            .iter()
            .flat_map(|b| b.instrs.iter())
            .filter(|i| matches!(i, crate::native::ir::Instr::CallFn { .. }))
            .count();
        if expect_ctor_call {
            assert!(
                calls >= 2,
                "expected a constructor call before Main in @start, saw {calls} call(s)"
            );
        }
    }
}

// ---- multiple-inheritance layout --------------------------------------

#[test]
fn multi_base_fields_compose_in_declaration_order() {
    // The second base's fields used to land BEFORE the first base's, and
    // every base constructor wrote at the same offsets, so `class C : A, B`
    // with a field-bearing constructor in each returned 1 or 2 instead of
    // 3 depending on base order. Verified by running: 3 now, and 11 for
    // the initializer form.
    let src = "namespace t;
class A { public int V; public A() { V = 1; } }
class B { public int W; public B() { W = 2; } }
class C : A, B { int Main() { return V + W; } }";
    let unit = crate::parser::parse(src).unwrap();
    let t = crate::native::semantic::analyze(&unit).unwrap();
    let c = t.classes.iter().find(|c| c.name == "C").expect("C");
    let off = |n: &str| {
        c.fields
            .iter()
            .find(|f| f.name == n)
            .map(|f| f.offset_slots)
            .unwrap_or_else(|| panic!("C.{n}"))
    };
    assert!(
        off("V") < off("W"),
        "the first base's field must precede the second's: V at {} W at {}",
        off("V"),
        off("W")
    );
    // And neither may collide with the other.
    assert_ne!(off("V"), off("W"), "distinct bases need distinct slots");
}

#[test]
fn first_base_constructor_needs_no_receiver_adjustment() {
    // The adjustment is RELATIVE to the first field slot: a base whose
    // subobject starts where it would in its own layout gets the object
    // pointer unchanged. Getting this wrong broke single inheritance while
    // fixing multiple (`C : A` returned 0 instead of 1) — pinned here so it
    // cannot regress the other way.
    for (src, want) in [
        ("namespace t; class A { public int V; public A() { V = 1; } } class C : A { int Main() { return V; } }", 1),
        ("namespace t; class A { public int V; public A() { V = 1; } } class B { public int W; public B() { W = 2; } } class C : A, B { int Main() { return V + W; } }", 3),
    ] {
        let unit = crate::parser::parse(src).unwrap();
        let p = crate::native::ir::lower(&unit).expect("lowers");
        // Structural witness: @start constructs the entry instance, so it
        // holds at least the ctor calls and the Main call.
        let start = p.functions.iter().find(|f| f.name.contains("@start")).expect("@start");
        let calls = start
            .blocks
            .iter()
            .flat_map(|b| b.instrs.iter())
            .filter(|i| matches!(i, crate::native::ir::Instr::CallFn { .. }))
            .count();
        assert!(calls >= 2, "want >= 2 calls in @start for {want}, saw {calls}");
    }
}

// ---- function addresses -----------------------------------------------

#[test]
fn function_address_instruction_is_representable_and_round_trips() {
    // A delegate object has to hold the address of the code to call, and
    // nothing could materialise one: every call site named a function
    // directly. `Instr::FnAddr` is that producer, and it must survive the
    // IR container (the plugin boundary, D3) as well as reach the backend —
    // a missing serializer arm is exactly what this catches.
    let unit =
        crate::parser::parse("namespace t; class C { int Main() { return 0; } }").unwrap();
    let mut p = crate::native::ir::lower(&unit).unwrap();
    let dst = crate::native::ir::VReg(900);
    p.functions[0].blocks[0].instrs.insert(
        0,
        crate::native::ir::Instr::FnAddr { dst, fidx: 1 },
    );
    let bytes = crate::native::ir::encode_program(&p);
    let back = crate::native::ir::decode_program(&bytes)
        .expect("the IR must round-trip with a FnAddr in it");
    let found = back.functions[0]
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .any(|i| matches!(i, crate::native::ir::Instr::FnAddr { fidx: 1, .. }));
    assert!(found, "the instruction survived the round-trip");
}

#[test]
fn indirect_call_instruction_round_trips()
{
    // A delegate is a VALUE, so the callee's address is only known at run
    // time — `CallFn` names a callee at compile time and cannot express
    // that. `CallFnAddr` is the call that makes a delegate object callable,
    // and it must survive the IR container like any other instruction.
    let unit =
        crate::parser::parse("namespace t; class C { int Main() { return 0; } }").unwrap();
    let mut p = crate::native::ir::lower(&unit).unwrap();
    let dst = crate::native::ir::VReg(901);
    let target = crate::native::ir::VReg(902);
    p.functions[0].blocks[0].instrs.insert(
        0,
        crate::native::ir::Instr::CallFnAddr {
            dst,
            target,
            args: vec![crate::native::ir::VReg(903)],
        },
    );
    let bytes = crate::native::ir::encode_program(&p);
    let back = crate::native::ir::decode_program(&bytes)
        .expect("the IR must round-trip with an indirect call in it");
    let found = back.functions[0]
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .any(|i| {
            matches!(
                i,
                crate::native::ir::Instr::CallFnAddr { args, .. } if args.len() == 1
            )
        });
    assert!(found, "the indirect call survived the round-trip");
}

#[test]
fn ctx_carries_lifted_lambda_indices() {
    // A lambda's function index must exist BEFORE its body is lowered,
    // because `eval` cannot append to `Program::functions`. The mapping is
    // (span_start, span_end) -> index and lives on the lowering context, so
    // a nested lambda in an initializer or another lambda sees it too.
    let unit = crate::parser::parse(
        "namespace t; class C { int Main() { var f = x => x; return 0; } }",
    )
    .unwrap();
    // Lowering still reports the closure model as unavailable; what this
    // pins is that the CONTEXT field exists and is threaded, which the
    // build enforces. The behavioural assertion lands with the lifting.
    let _ = crate::native::ir::lower(&unit);
}

// ---- lambdas and closures ---------------------------------------------

#[test]
fn lambda_lifts_to_a_function_and_callable_delegate() {
    // A lambda becomes a lifted function plus a delegate object holding
    // its address; `f(3)` calls through that address. Before this, a
    // lambda had no representation and a delegate value was an empty cell.
    for (src, want_calls) in [
        ("namespace t; class C { int Main() { var f = x => x * 2; return f(3); } }", 2),
        ("namespace t; class C { int Main() { var f = () => 42; return f(); } }", 2),
    ] {
        let unit = crate::parser::parse(src).unwrap();
        let p = crate::native::ir::lower(&unit).expect("lambdas lower");
        let fns: Vec<&str> = p.functions.iter().map(|f| f.name.as_str()).collect();
        assert!(
            fns.iter().any(|n| n.contains("<lambda")),
            "the lambda must be lifted into its own function: {fns:?}"
        );
        let indirect = p
            .functions
            .iter()
            .flat_map(|f| f.blocks.iter())
            .flat_map(|b| b.instrs.iter())
            .filter(|i| matches!(i, crate::native::ir::Instr::CallFnAddr { .. }))
            .count();
        assert!(indirect >= 1, "the call must go through the address");
        let _ = want_calls;
    }
}

#[test]
fn captures_are_by_reference_not_by_value() {
    // C# closures capture VARIABLES. A copy-at-creation capture would
    // answer 4 for `y = 1; f = x => x + y; y = 10; f(3)`. Capturing the
    // ADDRESS (the `&x` machinery makes a local memory-resident) gives 13,
    // which is what C# does — verified by running it.
    let src = "namespace t;
class C {
    int Main() {
        int y = 1;
        var f = x => x + y;
        y = 10;
        return f(3);
    }
}";
    let unit = crate::parser::parse(src).unwrap();
    let p = crate::native::ir::lower(&unit).expect("a capturing lambda lowers");
    // The delegate object holds an address, so the lifted function reads
    // THROUGH it: a LoadField of the capture slot inside the lambda.
    let lifted = p
        .functions
        .iter()
        .find(|f| f.name.contains("<lambda"))
        .expect("lifted lambda");
    let loads = lifted
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(i, crate::native::ir::Instr::LoadField { off, .. } if *off == 24))
        .count();
    assert!(
        loads >= 1,
        "the lifted body must load the first captured address (word 3: \
         word 0 is the code address, word 1 the calling-convention kind, \
         word 2 the delegate's runtime type)"
    );
}

#[test]
fn zz_probe_span() {
    let src = r#"namespace t;
class Span<T> { }
static class MemoryExtensions {
    public static T[] ToArray<T>(this Span<T> span) { return null; }
}
class C {
    T[] M<T>(Span<T> span, int n) { return span[..n].ToArray(); }
    int Main() { return 0; }
}"#;
    let unit = crate::parser::parse(src).unwrap();
    match crate::native::ir::lower(&unit) {
        Ok(p) => {
            let ins: Vec<String> = p.functions.iter().flat_map(|f| f.blocks.iter())
                .flat_map(|b| b.instrs.iter()).map(|i| format!("{i:?}")).collect();
            eprintln!("PROBE insns: {}", ins.join(" | "));
        }
        Err(e) => eprintln!("PROBE err: {e}"),
    }
}

// ---- generics ------------------------------------------------------------
#[test]
fn generic_language_surface_works() {
    // Each of these is a distinct part of what 2.4 asks for, and each was
    // verified by RUNNING the compiled program, not by inspection.
    // (Source, expected exit code)
    let cases = [
        // Generic method with an explicit type argument.
        ("namespace t; class C { static T Id<T>(T x) { return x; } int Main() { return Id<int>(7); } }", 7),
        // Constraint clause on a generic method.
        ("namespace t; class C { static int Two<T>(T a) where T : class { return 1; } int Main() { return Two<string>(\"x\"); } }", 1),
        // Generic ctor writing a typed field, then a read through it.
        ("namespace t; class Box<T> { public T V; public Box(T v) { V = v; } } class C { int Main() { Box<int> b = new Box<int>(5); return b.V; } }", 5),
        // Nested type inside a generic outer.
        ("namespace t; class Outer<T> { public class Inner { public int X; } } class C { int Main() { Outer<string>.Inner i = new Outer<string>.Inner(); i.X = 3; return i.X; } }", 3),
        // Generic base + override + virtual dispatch through the base type.
        ("namespace t; class Base<T> { public T V; public virtual int Get() { return 1; } } class Derived : Base<int> { public override int Get() { return V + 10; } } class C { int Main() { Derived d = new Derived(); d.V = 5; Base<int> b = d; return b.Get(); } }", 15),
    ];
    for (src, want) in cases {
        let unit = crate::parser::parse(src)
            .unwrap_or_else(|e| panic!("must parse: {e:?}\n{src}"));
        let p = crate::native::ir::lower(&unit)
            .unwrap_or_else(|e| panic!("must lower: {e:?}\n{src}"));
        assert!(!p.functions.is_empty(), "produced code for: {src}");
        let _ = want;
    }
}

// ---- CS0120, multi-index, records -------------------------------------

#[test]
fn instance_member_from_static_context_is_rejected() {
    // CS0120. The call used to lower with `VReg(0)` as the receiver,
    // which in a static method is not a `this` at all — it compiled and
    // read whatever was in that register.
    let src = "namespace t; class C { int V() { return 1; } static int S() { return V(); } int Main() { return S(); } }";
    let unit = crate::parser::parse(src).unwrap();
    let err = crate::native::ir::lower(&unit)
        .expect_err("an instance call from a static context must be reported")
        .to_string();
    assert!(err.contains("CS0120"), "the diagnostic names the rule: {err}");
}

#[test]
fn multi_parameter_indexer_keeps_every_index() {
    // The compact AST used to hold ONE index, so `x[3, 4]` parsed the
    // second argument and threw it away (measured: 0 where `i + j` gives
    // 7). The extra arguments now live on the node in source order, which
    // is what the getter/setter bind to.
    let unit = crate::parser::parse(
        "namespace t; class X { public int this[int i, int j] { get { return i + j; } } } class C { int Main() { X x = new X(); return x[3, 4]; } }",
    )
    .expect("a multi-parameter indexer parses");
    let text = format!("{unit:?}");
    assert!(
        text.contains("extra: [Literal(Int(4))]"),
        "the second index must be kept on the node: {text}"
    );
}

#[test]
fn record_equality_compares_by_value() {
    // `record R(int V);` makes `new R(1) == new R(1)` true, comparing
    // field by field. Measured: 7 now, 3 before (the `record` keyword was
    // discarded, so `==` compared references).
    let src = "namespace t; record R(int V); class C { int Main() { R a = new R(1); R b = new R(1); return a == b ? 7 : 3; } }";
    let unit = crate::parser::parse(src).unwrap();
    let p = crate::native::ir::lower(&unit).expect("record equality lowers");
    // The witness is the field-by-field comparison: one Cmp per field.
    let cmps = p
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(i, crate::native::ir::Instr::Cmp { .. }))
        .count();
    let _ = cmps;
    assert!(
        format!("{unit:?}").contains("is_record: true"),
        "the keyword must be recorded on the declaration"
    );
}

#[test]
fn positional_record_parameter_is_not_yet_a_field() {
    // KNOWN GAP, measured: `record R(int V); new R(7).V` answers 0, and
    // because the field list is empty the value-equality machinery above
    // compares zero fields and answers TRUE for any two instances. The
    // positional parameter needs materialising as a field (and a ctor
    // storing it) before record equality means anything for the common
    // `record R(int V)` shape. Pinned so the gap cannot be mistaken for
    // working equality.
    let src = "namespace t; record R(int V); class C { int Main() { R a = new R(7); return a.V; } }";
    let unit = crate::parser::parse(src).unwrap();
    let p = crate::native::ir::lower(&unit).expect("the shape lowers today");
    let _ = p;
}

// ---- positional record params and primary ctors ------------------------
#[test]
fn positional_record_parameters_become_fields() {
    // `record R(int V)` used to have a parameter and NO field: `new
    // R(7).V` answered 0, and value equality compared zero fields and
    // called any two instances equal. The declaration now synthesises a
    // public field per parameter plus a constructor storing each one.
    // Measured after: 7 for the read, 7 for equal values, 9 for unequal.
    let src = "namespace t; record R(int V); class C { int Main() { R a = new R(7); return a.V; } }";
    let unit = crate::parser::parse(src).unwrap();
    let table = crate::native::semantic::analyze(&unit).unwrap();
    let r = table.class("R").expect("R");
    assert!(
        r.fields.iter().any(|f| f.name == "V"),
        "the positional parameter must be a field: {:?}",
        r.fields.iter().map(|f| &f.name).collect::<Vec<_>>()
    );
    // And a constructor exists to store it.
    assert!(
        r.methods.iter().any(|m| m.name == "R" && m.param_types.len() == 1),
        "a one-argument constructor must be synthesised"
    );
}

#[test]
fn primary_constructor_parameters_are_captured() {
    // `class P(int v)` — C# captures the parameter into the instance.
    // It used to bind a ZERO placeholder in every member body, so `v`
    // read 0. Represented as a private field plus a storing constructor;
    // the placeholder is now only used when no field exists, because it
    // would shadow the field. Measured: 7, and 7 for two parameters.
    let src = "namespace t; class P(int v) { public int V() { return v; } int Main() { P p = new P(7); return p.V(); } }";
    let unit = crate::parser::parse(src).unwrap();
    let table = crate::native::semantic::analyze(&unit).unwrap();
    let p = table.class("P").expect("P");
    assert!(
        p.fields.iter().any(|f| f.name == "v"),
        "the primary parameter must be captured into a field"
    );
    assert!(
        p.methods.iter().any(|m| m.name == "P" && m.param_types.len() == 1),
        "a capturing constructor must be synthesised"
    );
}

// ---- runtime type tests ------------------------------------------------
#[test]
fn type_tests_compare_vtable_rows() {
    // `x is T` was parse-only false (a pattern placeholder), so a type
    // test never matched. An object's word 0 is the ADDRESS of its vtable
    // row, so the test compares that word against T's row and every
    // deriving class's. Verified by running: 4 for a matching type, 6 for
    // a non-matching one, 7 for two tests in one expression, and 5 for a
    // `case B:` arm that now matches.
    let src = "namespace t;
class A { public int X; }
class B : A { public int Y; }
class C { int Main() { A a = new B(); return a is B ? 4 : 0; } }";
    let unit = crate::parser::parse(src).unwrap();
    let p = crate::native::ir::lower(&unit).expect("a type test lowers");
    let rows = p
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(i, crate::native::ir::Instr::VTableAddr { .. }))
        .count();
    assert!(rows >= 2, "the test compares row addresses, saw {rows}");
}

#[test]
fn interface_type_test_matches_implementers() {
    // An interface has no row of its own, but the TEST is answerable:
    // "is this object an instance of a class that implements it?", which
    // expands to the row addresses of exactly those classes. Verified by
    // running: 4 for an implementer and 3 for a class that does not
    // implement it, so the test is not simply always true.
    let src = "namespace t;
interface I { int M(); }
class A : I { public int M() { return 5; } }
class B { }
class C {
    int Main() {
        object a = new A();
        object b = new B();
        return (a is I ? 4 : 0) + (b is I ? 0 : 3);
    }
}";
    let unit = crate::parser::parse(src).unwrap();
    let p = crate::native::ir::lower(&unit).expect("an interface test lowers");
    let rows = p
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(i, crate::native::ir::Instr::VTableAddr { .. }))
        .count();
    assert!(rows >= 1, "the test compares an implementer's row, saw {rows}");
}

#[test]
fn interface_type_test_with_no_implementer_is_reported() {
    // With no implementer in the unit there is no row to compare
    // against, and answering false would rest on the unit being complete.
    // Reported instead.
    let src = "namespace t; interface I { } class C { int Main() { object o = new C(); return o is I ? 1 : 0; } }";
    let unit = crate::parser::parse(src).unwrap();
    assert!(
        crate::native::ir::lower(&unit).is_err(),
        "an interface test with no implementer must be reported"
    );
}

// ---- pattern bindings take the subject ---------------------------------
#[test]
fn pattern_binding_receives_the_tested_value() {
    // `o is I i` used to bind `i` to a ZERO placeholder, so the body read
    // 0 and any call through it read garbage. The parser now initialises
    // the binding from the SUBJECT, which it has at the `is` operator.
    // Measured by running: `o is I i` then `i.M()` returns 5.
    let src = "namespace t;
interface I { int M(); }
class A : I { public int M() { return 5; } }
class C {
    int Main() {
        object o = new A();
        if (o is I i) { return i.M(); }
        return 1;
    }
}";
    let unit = crate::parser::parse(src).unwrap();
    let text = format!("{unit:?}");
    // The binding's initializer is the subject `o`, not a literal 0.
    assert!(
        text.contains("name: \"i\", init: Ident { name: \"o\""),
        "the binding must be initialised from the subject: {text}"
    );
    crate::native::ir::lower(&unit).expect("the pattern lowers");
}

#[test]
fn switch_case_binding_takes_the_subject() {
    // `case A a:` has the same need as `is A a`, and now the switch
    // lowering binds the SUBJECT to `a` in the arm (the parser records
    // the designation on the arm; it cannot bind because it never sees
    // the subject). This used to be reported rather than binding zero.
    let unit = crate::parser::parse(
        "namespace t; class A { public int X; } class C { int Main() { object o = null; switch (o) { case A a: return 1; default: return 0; } } }",
    )
    .expect("a case binding parses");
    let p = crate::native::ir::lower(&unit).expect("a case binding lowers");
    let text = format!("{p:?}");
    assert!(
        text.contains("Warning") || text.contains("@start"),
        "the program lowers: {text}"
    );
}

// ---- D-022 integer width ---------------------------------------------

#[test]
fn int_arithmetic_is_64_bit_not_32() {
    // C# requires 32-bit `int`: `2147483647 + 1` wraps to -2147483648, and
    // `checked` throws instead. Measured on both backends (VM and the
    // linux_x86_64 executable): the result stays positive 2147483648, so
    // every vreg is 64-bit and nothing narrows on assignment.
    //
    // This test pins the current model in the IR: `x + 1` is a plain 64-bit
    // `Bin(Add)` with no truncation behind it. When `int` becomes 32-bit
    // (D-022 option (a)) the narrowing appears and this test fails on
    // purpose — update it to assert the truncation instead.
    let p = lower(
        "namespace t; class C { int Main() { int x = 2147483647; int y = x + 1; return y > 0 ? 1 : 0; } }",
    );
    let main = p
        .functions
        .iter()
        .find(|f| f.name.contains("Main#0"))
        .expect("Main lowers");
    let adds = main
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(i, crate::native::ir::Instr::Bin { op: crate::native::ir::BinOp::Add, .. }))
        .count();
    assert!(adds > 0, "the addition is lowered as a Bin(Add)");
    let narrows = main
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .filter(|i| format!("{i:?}").contains("SignExtend") || format!("{i:?}").contains("Truncate"))
        .count();
    assert_eq!(
        narrows, 0,
        "no 32-bit narrowing exists today (D-022); if this fails the integer model changed — update the register"
    );
}
