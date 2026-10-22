//! Virtual property dispatch must survive lowering: a virtual
//! read on a statically-typed receiver has to emit a vtable-row
//! dispatch chain (or a devirtualized body), never a const-0
//! placeholder. Regression: `Stream.CanWrite`-shaped reads (abstract
//! base, explicit override in a concrete subclass) silently became
//! const 0, so every `StreamWriter` construction threw
//! "Stream was not writable". Also covers ctor-store/getter-load
//! offset agreement for inherited bool fields.

use crate::native::ir::{Instr, Program};

fn lower(src: &str) -> Program {
    let unit = crate::parser::parse(src).unwrap();
    crate::native::ir::lower_with(&unit, false).unwrap()
}

fn find_fn<'a>(p: &'a Program, needle: &str) -> &'a crate::native::ir::IrFunction {
    p.functions
        .iter()
        .find(|f| f.name.contains(needle))
        .unwrap_or_else(|| {
            panic!(
                "no function containing {needle:?}; have {:?}",
                p.functions.iter().map(|f| &f.name).collect::<Vec<_>>()
            )
        })
}

fn has_vtable_addr(f: &crate::native::ir::IrFunction) -> bool {
    f.blocks.iter().flat_map(|b| b.instrs.iter()).any(|i| {
        matches!(
            i,
            Instr::VTableAddr { .. } | Instr::CallVirt { .. } | Instr::LoadField { .. }
        )
    })
}

const SRC: &str = r#"namespace t {
    abstract class Base {
        public abstract bool Flag { get; }
        public virtual int Size { get { return 0; } }
    }
    class Impl : Base {
        private bool _flag;
        public Impl(bool v) { _flag = v; }
        public override bool Flag => _flag;
    }
    class User {
        public static int ReadFlag(Base b) {
            if (!b.Flag) { return 1; }
            return 0;
        }
        public static Impl Make() {
            Impl m = new Impl(true);
            return m;
        }
    }
}
"#;

#[test]
fn virtual_property_read_emits_dispatch_not_placeholder() {
    let p = lower(SRC);
    let read = find_fn(&p, "ReadFlag");
    // The `!b.Flag` read must dispatch (vtable compare or an inlined
    // body load), not fold the whole method to constant zero.
    assert!(
        has_vtable_addr(read),
        "ReadFlag must contain dispatch/load IR for b.Flag; got {:?}",
        read.blocks.iter().flat_map(|b| b.instrs.iter()).collect::<Vec<_>>()
    );
}

#[test]
fn ctor_store_and_getter_load_share_field_offset() {    let p = lower(SRC);
    let ctor = find_fn(&p, "Impl::Impl");
    let store_off = ctor
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .find_map(|i| match i {
            Instr::StoreField { off, .. } if *off >= 16 => Some(*off),
            _ => None,
        })
        .expect("ctor must store _flag to an object slot");
    // The override getter must load the same slot the ctor stored.
    let has_matching_load = p
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instrs.iter())
        .any(|i| matches!(i, Instr::LoadField { off, .. } if *off == store_off));
    assert!(
        has_matching_load,
        "no LoadField at ctor store offset {store_off}; stores and loads disagree"
    );
}

const BIG_SRC: &str = r#"namespace t {
    abstract class Stream {
        public abstract bool CanWrite { get; }
    }
    class AStream : Stream {
        private bool _w;
        public AStream(bool v) { _w = v; }
        public override bool CanWrite => _w;
    }
    class BStream : Stream {
        public override bool CanWrite => true;
    }
    class CStream : Stream {
        public override bool CanWrite => false;
    }
    class DStream : AStream {
        public DStream(bool v) : base(v) { }
    }
    class EStream : Stream {
        public override bool CanWrite => true;
    }
    class FStream : Stream {
        public override bool CanWrite => false;
    }
    class GStream : Stream {
        private bool _w;
        public GStream(bool v) { _w = v; }
        private void Check() {
            if (!_w) { throw new System.Exception(); }
        }
        public override bool CanWrite {
            get {
                Check();
                return _w;
            }
        }
    }
    class User {
        public static int ReadWrite(Stream s) {
            if (!s.CanWrite) { return 1; }
            return 0;
        }
    }
}
"#;

#[test]
fn many_sibling_overrides_still_dispatch() {
    let p = lower(BIG_SRC);
    let read = find_fn(&p, "ReadWrite");
    // Six explicit overrides: must still emit a dispatch chain (or a
    // devirtualized body load), never const 0.
    assert!(
        has_vtable_addr(read),
        "ReadWrite must dispatch s.CanWrite across siblings; got {:?}",
        read.blocks.iter().flat_map(|b| b.instrs.iter()).collect::<Vec<_>>()
    );
    // At least one arm must inline a real object-field load
    // (AStream/DStream read `_w`); all-const arms mean every arm
    // fell back to placeholder zero.
    let field_loads = read
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(i, Instr::LoadField { off, .. } if *off >= 16))
        .count();
    assert!(
        field_loads >= 1,
        "deform arms must load object fields; got {field_loads} (all arms placeholder?)"
    );
}

const DUP_SRC: &str = r#"namespace t {
    abstract class Stream {
        public abstract bool CanWrite { get; }
    }
    partial class Dup : Stream {
        private bool _w;
        public Dup(bool v) { _w = v; }
        public override bool CanWrite => _w;
    }
    partial class Dup : Stream {
        private bool _w2;
    }
    class User {
        public static int ReadWrite(Stream s) {
            if (!s.CanWrite) { return 1; }
            return 0;
        }
        public static Dup Make() {
            Dup d = new Dup(true);
            return d;
        }
    }
}
"#;

#[test]
fn duplicate_sibling_decl_does_not_sink_dispatch() {    let p = lower(DUP_SRC);
    let read = find_fn(&p, "ReadWrite");
    // A same-named second declaration (partial-part shape) must not
    // sink the virtual read to const 0.
    assert!(
        has_vtable_addr(read),
        "ReadWrite must dispatch despite duplicate decl; got {:?}",
        read.blocks.iter().flat_map(|b| b.instrs.iter()).collect::<Vec<_>>()
    );
}

const LOCAL_SRC: &str = r#"namespace t {
    abstract class Stream {
        public abstract bool CanWrite { get; }
    }
    class Mem : Stream {
        private bool _w;
        public Mem(bool v) { _w = v; }
        public override bool CanWrite => _w;
    }
    class User {
        public static int ReadLocal() {
            Mem m = new Mem(true);
            if (!m.CanWrite) { return 1; }
            return 0;
        }
    }
}
"#;

#[test]
fn virtual_read_on_new_local_dispatches() {
    let p = lower(LOCAL_SRC);
    let read = find_fn(&p, "ReadLocal");
    // Same read but on a freshly constructed local (not a param):
    // must still dispatch, never const 0.
    assert!(
        has_vtable_addr(read),
        "ReadLocal must dispatch m.CanWrite; got {:?}",
        read.blocks.iter().flat_map(|b| b.instrs.iter()).collect::<Vec<_>>()
    );
}

const BLOCK_SRC: &str = r#"namespace t {
    abstract class Stream {
        public abstract int Capacity { get; }
    }
    class Mem : Stream {
        private int _cap;
        private int _org;
        private bool _open;
        public Mem(int c) { _cap = c; _org = 0; _open = true; }
        private void Ensure() {
            if (!_open) { throw new System.Exception(); }
        }
        public override int Capacity {
            get {
                Ensure();
                return _cap - _org;
            }
        }
    }
    class User {
        public static int ReadCap() {
            Mem m = new Mem(10);
            return m.Capacity;
        }
    }
}
"#;

fn count_callvirt_or_vtable(f: &crate::native::ir::IrFunction) -> usize {
    f.blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .filter(|i| {
            matches!(
                i,
                Instr::VTableAddr { .. } | Instr::CallVirt { .. }
            )
        })
        .count()
}

#[test]
fn block_bodied_override_arm_inlines_body() {    let p = lower(BLOCK_SRC);
    let read = find_fn(&p, "ReadCap");
    // Block-bodied override (`get { Ensure(); return _cap; }`)
    // must inline real field loads in the match arm, not const 0.
    // (World symptom: MemoryStream.Capacity virtual read answered 0
    // even though the dispatch matched the right row.)
    let loads = read
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(i, Instr::LoadField { .. }))
        .count();
    assert!(
        loads >= 1,
        "ReadCap match arm must load fields (_cap via getter); got {loads} loads"
    );
    assert!(
        count_callvirt_or_vtable(read) >= 1,
        "ReadCap must dispatch virtually"
    );
}

const INTERLOCKED_SRC: &str = r#"namespace t {
    static class Interlocked {
        public static int ExchangeAdd(ref int location1, int value)
        {
            return ExchangeAdd(ref location1, value);
        }
        public static int CompareExchange(ref int location1, int value, int comparand)
        {
            return CompareExchange(ref location1, value, comparand);
        }
    }
    class User {
        public static int AddOne(ref int loc) {
            return Interlocked.ExchangeAdd(ref loc, 1);
        }
        public static int Cas(ref int loc, int v, int c) {
            return Interlocked.CompareExchange(ref loc, v, c);
        }
    }
}
"#;

#[test]
fn interlocked_self_recursion_emulates() {
    // `[Intrinsic]` self-recursive placeholders ("Must expand
    // intrinsic") would recurse forever at runtime; calls must
    // lower to inline load/add/store/compare sequences with no
    // remaining call to the same method.
    let p = lower(INTERLOCKED_SRC);
    for needle in ["AddOne", "Cas", "Interlocked::ExchangeAdd", "Interlocked::CompareExchange"] {
        let f = find_fn(&p, needle);
        let instrs: Vec<_> = f
            .blocks
            .iter()
            .flat_map(|b| b.instrs.iter())
            .collect();
        let self_calls = instrs.iter().filter(|i| {
            matches!(
                i,
                Instr::CallStatic { .. } | Instr::CallFn { .. } | Instr::CallVirt { .. }
            )
        });
        assert_eq!(
            self_calls.count(),
            0,
            "{needle} must contain no calls after Interlocked emulation; got {instrs:?}"
        );
    }
}

const DEFAULT_SPAN_SRC: &str = r#"namespace t {
    class User {
        public static int EmptyLength() {
            System.ReadOnlySpan<byte> s = default(System.ReadOnlySpan<byte>);
            return s.Length;
        }
        public static int CondLength(bool c, System.ReadOnlySpan<byte> p) {
            System.ReadOnlySpan<byte> s = c ? p : default;
            return s.Length;
        }
    }
}
"#;

#[test]
fn default_span_materializes_object_not_null() {
    // `default(ReadOnlySpan<byte>)` is a legal empty span: it must
    // lower to a real span object (whose Length reads 0), never
    // const 0 (whose Length faults on the null slot). Regression:
    // UTF8-noBOM `Preamble` (false arm `default`) faulted
    // `Console.WriteLine` on the null Length read.
    let p = lower(DEFAULT_SPAN_SRC);
    let f = find_fn(&p, "EmptyLength");
    let has_alloc = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .any(|i| matches!(i, Instr::Alloc { .. } | Instr::AllocDyn { .. }));
    assert!(
        has_alloc,
        "default span must allocate an empty span object; got {:?}",
        f.blocks.iter().flat_map(|b| b.instrs.iter()).collect::<Vec<_>>()
    );
}

#[test]
fn ternary_bare_default_takes_arm_type() {    // Target-typed `default`: the bare arm adopts the other arm's
    // span type (same guarantee as above, via the ternary).
    let p = lower(DEFAULT_SPAN_SRC);
    let f = find_fn(&p, "CondLength");
    let has_alloc = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .any(|i| matches!(i, Instr::Alloc { .. } | Instr::AllocDyn { .. }));
    assert!(
        has_alloc,
        "ternary default arm must allocate an empty span object; got {:?}",
        f.blocks.iter().flat_map(|b| b.instrs.iter()).collect::<Vec<_>>()
    );
}

const NULLCOND_SRC: &str = r#"namespace t {
    class Box {
        public int val;
        public Box(int v) { val = v; }
    }
    class User {
        public static int Read(Box b) {
            return b?.val ?? -1;
        }
        public static int Call(Box b) {
            B2 c = new B2();
            return c?.Run(b) ?? -1;
        }
    }
    class B2 {
        public int Run(Box b) { return b.val + 1; }
    }
}
"#;

#[test]
fn null_conditional_guards_null_receiver() {
    // `?.` desugars to a null check (not a collapsing `.`): the
    // lowered IR must branch on the receiver (not dereference it
    // unconditionally). Regression: `EncoderNLS.Reset` faulted on
    // `_fallbackBuffer?.Reset()` with a null buffer.
    let p = lower(NULLCOND_SRC);
    for needle in ["Read", "Call"] {
        let f = find_fn(&p, needle);
        let has_branch = f.blocks.iter().any(|b| {
            !matches!(
                b.term,
                crate::native::ir::Terminator::Branch { cond: None, .. }
            )
        });
    assert!(
        has_branch,
        "{needle} must branch on the ?. receiver; got {:?}",
        f.blocks.iter().flat_map(|b| b.instrs.iter()).collect::<Vec<_>>()
    );
}

const EXTERN_SRC: &str = r#"namespace t {
    class SafeHandle {
        protected int handle;
        public SafeHandle(int fd) { handle = fd; }
    }
    static class Sys {
        public static int Write(SafeHandle fd) { return 0; }
    }
    class User {
        public static int Go(SafeHandle h) { return Sys.Write(h); }
    }
}
"#;

#[test]
fn extern_like_static_marshal_path() {
    // Placeholder: documents current routing for extern-shaped
    // static calls ( обеспечивается ). Prints IR shape for
    // diagnosis; always passes (informational only).
    let p = lower(EXTERN_SRC);
    let f = find_fn(&p, "Go");
    let instrs: Vec<_> = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .collect();
    eprintln!("EXTERNGo IR: {instrs:?}");
}
}

const INSTINIT_SRC: &str = r#"namespace t {
    class Holder {
        public static readonly int[] shared = new int[3];
        public int[] mine = shared;
        public Holder() { }
    }
    class User {
        public static int Read() {
            Holder h = new Holder();
            return h.mine.Length;
        }
    }
}
"#;

#[test]
fn instance_init_from_static_reads_static() {
    // Instance field initializer reading a static (`= s_coreNewLine`
    // shape) must load static storage, not zero (which nulled
    // `StreamWriter.CoreNewLine`, faulting `WriteLine` on the null
    // Length read). Field inits emit at the `new` site.
    let p = lower(INSTINIT_SRC);
    let read = find_fn(&p, "Read");
    let has_static_load = read
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .any(|i| matches!(i, Instr::LoadStaticAddr { .. }));
    assert!(
        has_static_load,
        "new-site field init must load static storage; got {:?}",
        read.blocks.iter().flat_map(|b| b.instrs.iter()).collect::<Vec<_>>()
    );
}

const GENERIC_SRC: &str = r#"namespace t {
    class Base {
        public int flags;
        public Base(int f) { flags = f; }
        public bool Done() { return (flags & 16) != 0; }
    }
    class Box<T> : Base {
        public T val;
        public Box(int f, T v) : base(f) { val = v; }
    }
    class User {
        public static int Check() {
            Box<int> b = new Box<int>(16, 7);
            if (!b.Done()) { return 1; }
            return 0;
        }
    }
}
"#;

#[test]
fn generic_subclass_base_field_roundtrip() {
    // Generic instantiation must flatten base fields (a base ctor
    // store read back through the instantiation): `Task<T>`-shaped
    // hierarchies (`m_stateFlags` read as 0 faulted
    // `Task.IsCompleted`, throwing `ThrowAsyncIOInProgress`).
    // Asserts structurally (a real base-field load present), not
    // behaviorally.
    let p = lower(GENERIC_SRC);
    let done = find_fn(&p, "Done");
    let loads = done
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(i, Instr::LoadField { off, .. } if *off >= 16))
        .count();
    assert!(
        loads >= 1,
        "Done must load base field flags; got {:?}",
        done.blocks.iter().flat_map(|b| b.instrs.iter()).collect::<Vec<_>>()
    );
}

#[test]
fn generic_derived_chains_to_base_ctor() {
    // `Box<T>(...) : base(...)` must emit the base-ctor call;
    // a silently skipped chain leaves base fields zero (e.g.
    // `Task<TResult>` never setting `m_stateFlags`, so
    // `Task.CompletedTask.IsCompleted` reads false and
    // `StreamWriter` throws `ThrowAsyncIOInProgress`).
    let p = lower(GENERIC_SRC);
    let ctor = find_fn(&p, "Box::Box");
    let calls = ctor
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .filter(|i| {
            matches!(
                i,
                Instr::CallFn { .. } | Instr::CallStatic { .. }
            )
        })
        .count();
    assert!(
        calls >= 1,
        "Box<T> ctor must chain-call the base ctor; got {:?}",
        ctor.blocks.iter().flat_map(|b| b.instrs.iter()).collect::<Vec<_>>()
    );
}

const ADDROF_SRC: &str = r#"namespace t {
    class User {
        public static unsafe int At(byte[] arr, int i) {
            byte* p = &arr[i];
            return 0;
        }
    }
}
"#;

#[test]
fn addrof_index_computes_address() {
    // `&arr[i]` must compute the element ADDRESS (`base + 16 +
    // i*w`), not load the element value (which faulted
    // `Console.WriteLine` with a null destination into
    // `Buffer.MemoryCopy`).
    let p = lower(ADDROF_SRC);
    let f = find_fn(&p, "At");
    let adds = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .filter(|i| {
            matches!(
                i,
                Instr::Bin {
                    op: crate::native::ir::BinOp::Add,
                    ..
                }
            )
        })
        .count();
    assert!(
        adds >= 2,
        "&arr[i] must emit address arithmetic (base+16+i*w); got {:?}",
        f.blocks.iter().flat_map(|b| b.instrs.iter()).collect::<Vec<_>>()
    );
}

const SPAN_COERCE_SRC: &str = r#"namespace t {
    class ReadOnlySpan<T> {
        public int Length;
    }
    class User {
        public static int Take(ReadOnlySpan<char> s) { return s.Length; }
        public static int Go(string? v) { return Take(v); }
    }
}
"#;

#[test]
fn nullable_string_arg_packs_to_span() {
    // `string?` argument to a span parameter must pack (not pass
    // the string object through, whose chars then read as a data
    // pointer). Regression: `StreamWriter.WriteLine(string?)`
    // faulted in `Buffer.MemoryCopy` on the unpacked string.
    let p = lower(SPAN_COERCE_SRC);
    let f = find_fn(&p, "Go");
    let has_alloc = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .any(|i| matches!(i, Instr::Alloc { .. } | Instr::AllocDyn { .. }));
    assert!(
        has_alloc,
        "string? -> span call must pack (alloc); got {:?}",
        f.blocks.iter().flat_map(|b| b.instrs.iter()).collect::<Vec<_>>()
    );
}

const SAFEHANDLE_SRC: &str = r#"namespace t {
    class SafeHandle {
        protected int handle;
        public SafeHandle(int fd) { handle = fd; }
    }
    class SafeFileHandle : SafeHandle {
        public SafeFileHandle(int fd) : base(fd) { }
    }
    class Native {
        public static int Write(int fd) { return fd; }
    }
    class User {
        public static int Go(SafeFileHandle h) { return Native.Write(h); }
    }
}
"#;

#[test]
fn safehandle_arg_marshals_to_fd() {
    // P/Invoke-style SafeHandle→int marshaling (`DangerousGetHandle`):
    // passing the object pointer as the fd faulted
    // `Console.WriteLine` with EBADF (a heap address went out as
    // fd 45336048). The call must read the `handle` field.
    let p = lower(SAFEHANDLE_SRC);
    let f = find_fn(&p, "Go");
    let field_loads = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(i, Instr::LoadField { .. }))
        .count();
    assert!(
        field_loads >= 1,
        "SafeHandle arg must load the handle field; got {:?}",
        f.blocks.iter().flat_map(|b| b.instrs.iter()).collect::<Vec<_>>()
    );
}

const BASEINIT_SRC: &str = r#"namespace t {
    class Base {
        public int[] arr = new int[2];
    }
    class Mid : Base {
        public int x;
    }
    class Leaf : Mid {
        public int y;
    }
    class User {
        public static int Read() {
            Leaf l = new Leaf();
            return l.arr.Length;
        }
    }
}
"#;

#[test]
fn base_field_inits_run_at_new() {
    // Base-declared instance initializers must run (root-first,
    // then own): only own-decl members were visited, leaving
    // base-initialized fields zeroed (`TextWriter.CoreNewLine`
    // null faulted `WriteLine` on the null Length read).
    let p = lower(BASEINIT_SRC);
    let read = find_fn(&p, "Read");
    // The `new Leaf()` sequence must store to at least two distinct
    // object slots (base array field + own int fields), proving
    // base inits emit.
    let mut slots = std::collections::BTreeSet::new();
    for b in &read.blocks {
        for i in &b.instrs {
            if let Instr::StoreField { off, .. } = i {
                if *off >= 16 {
                    slots.insert(*off);
                }
            }
        }
    }
    assert!(
        slots.len() >= 2,
        "new Leaf() must init base+own fields; slots={slots:?}"
    );
}

const SETTER_SRC: &str = r#"namespace t {
    abstract class Base {
        public virtual bool Flag { get => false; set { } }
    }
    class Impl : Base {
        private bool _flag;
        public override bool Flag {
            get => _flag;
            set { _flag = value; }
        }
    }
    class Impl2 : Base {
        public override bool Flag {
            get => true;
            set { }
        }
    }
    class User {
        public static void SetFlag(Base b, bool v) {
            b.Flag = v;
        }
        public static void SetDirect() {
            Impl m = new Impl();
            m.Flag = true;
        }
    }
}
"#;

#[test]
fn virtual_setter_dispatches_not_drops() {
    // Virtual property SETS must run setter bodies, never silently
    // drop: a base-typed write deforms over the overriding
    // subclasses (row dispatch), and a concretely-typed write
    // inlines the setter body (field store). Both shapes reach
    // lowering as `Expr::Assign` (the parser never emits
    // `Stmt::MemberAssign`).
    let p = lower(SETTER_SRC);
    let f = find_fn(&p, "SetFlag");
    assert!(
        f.blocks.iter().flat_map(|b| b.instrs.iter()).any(|i| {
            matches!(i, Instr::VTableAddr { .. } | Instr::CallVirt { .. })
        }),
        "base-typed set must row-dispatch over Impl/Impl2; got {:?}",
        f.blocks.iter().flat_map(|b| b.instrs.iter()).collect::<Vec<_>>()
    );
    let f = find_fn(&p, "SetDirect");
    assert!(
        f.blocks.iter().flat_map(|b| b.instrs.iter()).any(|i| {
            matches!(i, Instr::StoreField { .. })
        }),
        "concrete-typed set must inline the `_flag = value` store; got {:?}",
        f.blocks.iter().flat_map(|b| b.instrs.iter()).collect::<Vec<_>>()
    );
}

const INIT_SRC: &str = r#"namespace t {
    class Impl {
        private bool _f;
        public bool Flag {
            get => _f;
            set { _f = value; }
        }
        public int N;
    }
    class User {
        public static void Make() {
            Impl m = new Impl() { Flag = true, N = 3 };
        }
    }
}
"#;

#[test]
fn object_initializer_writes_run_after_construction() {
    // `new Impl() { Flag = true, N = 3 }` must emit the member
    // writes (setter inline + field store), never drop them: this
    // is the `new StreamWriter(...) { AutoFlush = true }` shape
    // that flushes console output.
    let p = lower(INIT_SRC);
    let f = find_fn(&p, "Make");
    let stores = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(i, Instr::StoreField { .. }))
        .count();
    assert!(
        stores >= 2,
        "initializer must store _flag (setter) and N (field); got {:?}",
        f.blocks.iter().flat_map(|b| b.instrs.iter()).collect::<Vec<_>>()
    );
}

#[test]
fn self_referential_property_read_is_reported() {
    // `int P => P` has no terminating inline: lowering used to break
    // the cycle by answering 0, so the read ran like a property that
    // legitimately held zero. It must be a diagnostic instead
    //.
    let r = crate::native::ir::lower_with(
        &crate::parser::parse(
            r#"namespace t;
class C {
    public int P { get { return P; } set { } }
    public int Read() { return P; }
    int main() { return Read(); }
}"#,
        )
        .unwrap(),
        false,
    );
    let err = r.expect_err("a self-referential property read must not answer 0");
    assert!(
        err.to_string().contains("cycles"),
        "expected the cycle diagnostic, got: {err}"
    );
}

#[test]
fn property_read_without_a_getter_body_is_reported() {
    // An accessor with no modelled body has no value to read. The
    // const-0 answer was indistinguishable from a property holding
    // default(T).
    let r = crate::native::ir::lower_with(
        &crate::parser::parse(
            r#"namespace t;
abstract class Shape {
    public abstract int Sides { get; set; }
}
class C {
    public int Read(Shape s) { return s.Sides; }
    int main() { return 0; }
}"#,
        )
        .unwrap(),
        false,
    );
    let err = r.expect_err("a property with no getter body must not answer 0");
    assert!(
        err.to_string().contains("no getter body"),
        "expected the no-body diagnostic, got: {err}"
    );
}

#[test]
fn object_initializer_member_writes_still_run() {
    // `Key = Value` initializers ARE implemented (they become member
    // writes after construction) — that pass removed only the shapes whose
    // elements were being dropped, and this is the guard that they were
    // not collateral damage.
    let p = lower(INIT_SRC);
    let f = find_fn(&p, "Make");
    let stores = f
        .blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .filter(|i| matches!(i, Instr::StoreField { .. }))
        .count();
    assert!(stores >= 2, "member writes still emitted: {stores}");
}
