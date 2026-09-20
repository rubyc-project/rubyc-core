//! Virtual-register IR.
//!
//! Lowering produces three-address code over unbounded virtual registers;
//! the encoder runs linear-scan allocation and emits machine code. There is
//! no implicit accumulator and no memory protocol between instructions —
//! locals and temporaries are vregs, and the allocator decides which ones
//! live in registers versus stack slots.
//!
//! The IR is language-agnostic: it provides the primitives every language
//! needs (integers, floats, calls, dispatch, memory, control flow,
//! exceptions) without assuming a specific object model, memory strategy,
//! or calling convention. A language frontend lowers to this IR; a target
//! backend consumes it. The binary serialization format (see `serialize`)
//! is the plugin ABI boundary.

/// A virtual register. Identity is its index; the allocator maps each one
/// to a physical register or a stack slot for its whole live range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct VReg(pub u32);

pub type BlockId = u32;

/// How a function receives its object.
///
/// - `None` — free function, no receiver.
/// - `This` — implicit receiver in vreg 0 (C#/RubyC, C++).
/// - `Explicit` — receiver is a regular parameter (Rust `self`); the
///   backend places it according to the normal argument order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiverKind {
    None,
    This,
    Explicit,
}

/// Binary integer arithmetic. Division follows x86 `idiv` semantics:
/// quotient in the destination, hardware trap on zero divisor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    /// Remainder of `idiv` (rdx). Traps on zero divisor like Div.
    Rem,
    /// dst = a & b
    BitAnd,
    /// dst = a | b
    BitOr,
    /// dst = (a != 0) & (b != 0) — logical AND over normalized bools.
    And,
    /// dst = 1 when either operand is non-zero.
    Or,
    /// dst = a ^ b
    BitXor,
    /// dst = a << b (count masked to 6 bits by hardware)
    Shl,
    /// dst = a >> b — arithmetic shift, matching signed-int semantics.
    Shr,
    /// dst = a >>> b — logical (unsigned) shift.
    ShrShr,
    /// Type check: dst = (a is type) ? 1 : 0.
    Is,
}

/// Integer comparison producing `bool` as 0/1. Signed ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

/// Binary floating-point arithmetic (IEEE 754 double).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FBinOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Floating-point comparison producing `bool` as 0/1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FCmpOp {
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

/// One instruction. Every value-producing op names a destination vreg;
/// operands are always vregs or immediates, never memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instr {
    /// dst = imm (i64)
    Const { dst: VReg, imm: i64 },
    /// dst = imm (f64). The immediate is stored as its bit pattern (u64)
    /// so the instruction implements `Eq`.
    FConst { dst: VReg, bits: u64 },
    /// dst = a
    Mov { dst: VReg, a: VReg },
    /// Raw machine bytes emitted verbatim. Used by the built-in assembler
    /// (#Assembly blocks / .asm files). Clobbers everything.
    Raw(Vec<u8>),
    /// dst = a op b (i64)
    Bin {
        dst: VReg,
        op: BinOp,
        a: VReg,
        b: VReg,
    },
    /// dst = a op b (f64)
    FBin {
        dst: VReg,
        op: FBinOp,
        a: VReg,
        b: VReg,
    },
    /// dst = (a `op` b) as bool — 0 or 1 (i64)
    Cmp {
        dst: VReg,
        op: CmpOp,
        a: VReg,
        b: VReg,
    },
    /// dst = (a `op` b) as bool — 0 or 1 (f64)
    FCmp {
        dst: VReg,
        op: FCmpOp,
        a: VReg,
        b: VReg,
    },
    /// dst = -a (i64)
    Neg { dst: VReg, a: VReg },
    /// dst = -a (f64)
    FNeg { dst: VReg, a: VReg },

    /// Direct call. `args` is the full argument list already in calling
    /// order: `[receiver?] ++ params`. The ABI places them in registers,
    /// then stack.
    CallFn {
        dst: VReg,
        fidx: u32,
        args: Vec<VReg>,
    },
    /// Direct call to a `ReceiverKind::This` function with no receiver (a
    /// static method). The backend passes 0 in the receiver register.
    CallStatic {
        dst: VReg,
        fidx: u32,
        args: Vec<VReg>,
    },
    /// Virtual dispatch through `[recv_vtable + slot * 8]`.
    CallVirt {
        dst: VReg,
        slot: u32,
        recv: VReg,
        args: Vec<VReg>,
    },
    /// Dynamic dispatch by name. `name` indexes `Program::names`. The
    /// backend resolves the name at runtime (e.g. method table lookup,
    /// function pointer table, or a string-keyed dispatch table).
    CallByName {
        dst: VReg,
        recv: VReg,
        name: u32,
        args: Vec<VReg>,
    },
    /// External symbol call resolved by the linker.
    CallImport {
        dst: VReg,
        import: u32,
        args: Vec<VReg>,
    },
    /// dst = runtime address of an imported entity (a vtable row or a
    /// static slot owned by another image). The linker resolves
    /// `Program::imports[import]` (lib hint + symbol) to an absolute
    /// address at load time — the address-import twin of `CallImport`.
    LoadImport { dst: VReg, import: u32 },

    /// dst = bump-allocate `bytes` on the heap.
    Alloc { dst: VReg, bytes: u64 },
    /// dst = process argv as a `string[]` array (same layout as
    /// `NewArray`: length at offset 0, elements from offset 8).
    /// Each element is a heap string in literal layout (length at 0,
    /// refcount 1 at 8, bytes from 16). The backend reads the kernel
    /// argv via a startup-saved stack slot; contexts without one
    /// (JIT) yield an empty array rather than faulting.
    BuildArgv { dst: VReg },
    /// dst = bump-allocate `bytes` (a runtime value) on the heap. Used for
    /// runtime-sized objects such as concatenated strings.
    AllocDyn { dst: VReg, bytes: VReg },
    /// dst = allocate `bytes` on the stack frame. The backend must
    /// reserve the space in the frame prologue.
    StackAlloc { dst: VReg, bytes: u64 },
    /// dst = address of vtable row `row`
    VTableAddr { dst: VReg, row: u32 },
    /// dst = address of static slot `idx` in `Program::statics`. The
    /// backend resolves the absolute address of the slot in the data
    /// segment.
    LoadStaticAddr { dst: VReg, idx: u32 },
    /// dst = [base + off] where `off` is a byte offset.
    LoadField { dst: VReg, base: VReg, off: u32 },
    /// dst = (u8)[base + off] zero-extended to 64 bits.
    LoadByte { dst: VReg, base: VReg, off: u32 },
    /// [base + off] = src where `off` is a byte offset.
    StoreField { base: VReg, off: u32, src: VReg },

    /// write(1, string-at-data-offset, len)
    WriteStr { off: u32, len: u32 },
    /// dst = absolute address of the string at data offset `off`.
    LoadStrAddr { dst: VReg, off: u32 },
    /// memcpy: copy `len` bytes from `[src]` to `[dst]`. Both are addresses.
    MemCpy { dst: VReg, src: VReg, len: VReg },
    /// dst = 1 if `[a..a+len] == [b..b+len]`, else 0.
    MemCmp { dst: VReg, a: VReg, b: VReg, len: VReg },
    /// write(1, [src+16], [src+0]) — print a first-class string (layout:
    /// [0..8)=len, [8..16)=refcount, [16..)=data) to stdout.
    WriteString { src: VReg },
    /// print decimal of src (stdout)
    WriteInt { src: VReg },
    /// print decimal of src (stderr)
    EWriteInt { src: VReg },
    /// write(fd=2, string, len) — stderr twin of WriteStr
    EWriteStr { off: u32, len: u32 },
    /// read one line from stdin, parse signed decimal → dst
    ScanInt { dst: VReg },
    /// atomic refcount increment (share a heap reference).
    /// Backends with a different GC strategy may ignore this.
    Retain { src: VReg },
    /// atomic decrement; freelist-push the block on zero (drop last ref).
    /// `bytes` is the compile-time object size for the freelist capacity.
    /// Backends with a different GC strategy may ignore this.
    Release { src: VReg, bytes: u64 },
    /// Explicit destructor call at scope exit (Rust `drop`, C++ dtor).
    /// The backend calls the object's destructor. Distinct from `Release`
    /// (refcount): `Drop` is unconditional and runs exactly once.
    Drop { src: VReg },

    // ---- exception handling (setjmp/longjmp handler frames) ----

    /// Set up a `try` handler. Calls the runtime's `setjmp`; `dst` receives
    /// `0` on the normal first entry and `1` when control is transferred back
    /// here after an exception is thrown (via `longjmp`).
    HandlerEnter { dst: VReg },
    /// Remove the innermost active handler (normal completion of a try).
    HandlerPop,
    /// Load the in-flight exception payload into `dst`.
    LoadExc { dst: VReg },
    /// Clear the in-flight flag (a catch has taken ownership of the exception).
    ClearInflight,

    /// terminate the process (synthetic `@start` only)
    Exit,

    // ---- async / cooperative-coroutine instructions ----

    /// Allocate a task (coroutine frame) of `bytes` on the heap. `dst` is the
    /// pointer to the task. The task struct layout is managed by the runtime;
    /// the backend is responsible for initializing the control fields.
    ///
    /// Layout (managed by the runtime, not the compiler):
    ///   [0] = state (0 = Ready, 1 = Running, 2 = Suspended, 3 = Completed)
    ///   [1] = return value (i64)
    ///   [2] = saved instruction pointer / state-machine state
    ///   [3..] = saved local variables (size = bytes - 24)
    TaskAlloc { dst: VReg, bytes: u64 },

    /// Suspend the current coroutine, storing the continuation state.
    /// The backend saves the current state-machine state and local values
    /// into the task frame, then returns to the scheduler. When resumed,
    /// execution continues from the next state-machine block.
    Suspend { state: VReg },

    /// Resume a suspended coroutine. `task` is the task pointer. The backend
    /// restores local values from the task frame and jumps to the saved
    /// state-machine state. `dst` is the task's return value (valid after
    /// the coroutine completes).
    CoroutineResume { task: VReg, dst: VReg },
}

/// How a basic block ends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Terminator {
    /// return value (None = 0)
    Ret(Option<VReg>),
    /// synthetic exit with status 0
    Exit,
    /// synthetic exit with a runtime status (entry method's return value)
    ExitWith(VReg),
    /// Jump. `cond = None` is unconditional; otherwise the branch takes
    /// `if_true` when `cond != 0` and `if_false` otherwise.
    Branch {
        cond: Option<VReg>,
        if_true: BlockId,
        if_false: BlockId,
    },
    /// Throw the exception payload in `src`; never returns. The runtime walks
    /// the handler stack and `longjmp`s to the nearest enclosing handler.
    Unwind { src: VReg },
    /// Re-throw the in-flight exception (propagate to an enclosing handler);
    /// never returns.
    Rethrow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub instrs: Vec<Instr>,
    pub term: Terminator,
}

impl Block {
    pub fn empty() -> Self {
        Block {
            instrs: Vec::new(),
            term: Terminator::Ret(None),
        }
    }
}

/// One lowered function. `arity` counts parameters excluding the receiver;
/// the prologue materializes incoming registers into vregs where vreg 0 is
/// the receiver when [`ReceiverKind::This`] is set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrFunction {
    pub name: String,
    pub receiver: ReceiverKind,
    pub arity: usize,
    pub blocks: Vec<Block>,
    /// Total vregs; indices are dense.
    pub vregs: u32,
    /// True when this function is an async method lowered into a
    /// state-machine coroutine. The backend uses this to emit the
    /// coroutine prologue/epilogue and to handle `TaskAlloc`/`Suspend`/
    /// `Resume` instructions.
    pub is_async: bool,
}

impl IrFunction {
    pub fn new(
        name: impl Into<String>,
        receiver: ReceiverKind,
        arity: usize,
    ) -> Self {
        let mut f = IrFunction {
            name: name.into(),
            receiver,
            arity,
            blocks: vec![Block::empty()],
            vregs: 0,
            is_async: false,
        };
        if receiver == ReceiverKind::This {
            f.fresh_vreg(); // vreg 0 = this
        }
        for _ in 0..arity {
            f.fresh_vreg(); // params follow
        }
        f
    }

    pub fn fresh_vreg(&mut self) -> VReg {
        let v = VReg(self.vregs);
        self.vregs += 1;
        v
    }

    pub fn block(&mut self) -> BlockId {
        self.blocks.push(Block::empty());
        (self.blocks.len() - 1) as BlockId
    }

    pub fn cur(&mut self) -> &mut Block {
        let last = self.blocks.len() - 1;
        &mut self.blocks[last]
    }

    pub fn push(&mut self, i: Instr) {
        self.cur().instrs.push(i);
    }
}

/// Metadata for one static-field slot in `Program::statics`. The actual
/// storage lives in the data segment (see `MachineCode::static_offsets`);
/// this only carries the declaring class and name for reference and
/// diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrStatic {
    /// The field name.
    pub name: String,
    /// The class that declares (owns the storage of) this static.
    pub class: String,
    /// The field's type name (e.g. "int").
    pub type_name: String,
}

/// A lowered program ready for the backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    /// Function 0 is the synthetic `@start` for executables; shared
    /// libraries have none and `exports` indices stand alone.
    pub functions: Vec<IrFunction>,
    /// Static-field slots in declaration order (class order, then field
    /// order). `LoadStaticAddr::idx` indexes here.
    pub statics: Vec<IrStatic>,
    /// `(class_name, [function index per vtable slot])`.
    pub vtables: Vec<(String, Vec<usize>)>,
    /// `[LibraryImport]` stubs; `CallImport::import` indexes here.
    pub imports: Vec<(String, String)>,
    /// `(exported symbol, function index, arity)` — the shared-library ABI
    /// surface. Namespaces and classes never reach the ABI.
    pub exports: Vec<(String, usize, usize)>,
    /// Index of the synthetic `@init` function (shared libraries only):
    /// runs static field initializers once at load time. `None` for
    /// executables (their `@start` runs initializers instead) and for
    /// images without statics.
    pub init: Option<usize>,
    /// Literal-string bytes; `Sym::String` offsets address this blob.
    pub strings: Vec<u8>,
    /// Object heap bytes required at runtime.
    pub heap_size: u64,
    /// Method/field names for `CallByName` dynamic dispatch.
    pub names: Vec<String>,
    /// Entry point: index into `functions` of the function to run first.
    /// `None` for shared libraries or programs with no entry.
    pub entry: Option<usize>,
}
