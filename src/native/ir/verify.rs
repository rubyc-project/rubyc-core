//! IR verifier: every vreg read must be defined on every path that
//! reaches it.
//!
//! The lowering emits shared vregs (loop counters, join results) that are
//! assigned in more than one block, so "the definition dominates the use"
//! is too strict: a join target assigned in both arms of a branch is
//! defined on every incoming path without any single definition
//! dominating. The check here is definite assignment: for each use, either
//! one definition dominates it, or a per-vreg dataflow proves the value is
//! assigned on every path. A vreg defined in only one arm of a branch and
//! read in its sibling is neither, and that is exactly the miscompile this
//! guards against: the backend gives such a vreg one location, the sibling
//! path never writes it, and the read sees stack or heap garbage.

use super::instructions::{Instr, IrFunction, Program, ReceiverKind, Terminator, VReg};

/// Verify every function in `program`; `Err` names the first violation.
pub fn verify(program: &Program) -> Result<(), String> {
    for f in &program.functions {
        verify_function(f)?;
    }
    Ok(())
}

struct Sites {
    /// Definition sites per vreg: (block index, instruction index).
    defs: Vec<Vec<(usize, usize)>>,
    /// Uses: (vreg, block index, instruction index, is_terminator).
    uses: Vec<(u32, usize, usize, bool)>,
}

fn collect(f: &IrFunction) -> Result<Sites, String> {
    let mut defs = vec![Vec::new(); f.vregs as usize];
    let mut uses = Vec::new();
    for (bi, b) in f.blocks.iter().enumerate() {
        for (ii, ins) in b.instrs.iter().enumerate() {
            if let Some(d) = dst_of(ins) {
                if d.0 as usize >= f.vregs as usize {
                    return Err(format!(
                        "function '{}' block {bi} instruction {ii}: \
                         destination vreg {d:?} is beyond vregs={}",
                        f.name, f.vregs
                    ));
                }
                defs[d.0 as usize].push((bi, ii));
            }
            push_uses(ins, |v| uses.push((v.0, bi, ii, false)));
        }
        match &b.term {
            Terminator::Ret(Some(v)) | Terminator::ExitWith(v) | Terminator::Unwind { src: v } => {
                uses.push((v.0, bi, b.instrs.len(), true));
            }
            Terminator::Branch { cond: Some(v), .. } => uses.push((v.0, bi, b.instrs.len(), true)),
            _ => {}
        }
    }
    Ok(Sites { defs, uses })
}

fn dst_of(i: &Instr) -> Option<VReg> {
    Some(match i {
        Instr::Const { dst, .. }
        | Instr::FConst { dst, .. }
        | Instr::Mov { dst, .. }
        | Instr::Bin { dst, .. }
        | Instr::FBin { dst, .. }
        | Instr::Cmp { dst, .. }
        | Instr::FCmp { dst, .. }
        | Instr::Neg { dst, .. }
        | Instr::FNeg { dst, .. }
        | Instr::F2I { dst, .. }
        | Instr::I2F { dst, .. }
        | Instr::CallFn { dst, .. }
        | Instr::CallStatic { dst, .. }
        | Instr::CallVirt { dst, .. }
        | Instr::CallByName { dst, .. }
        | Instr::CallImport { dst, .. }
        | Instr::LoadImport { dst, .. }
        | Instr::Alloc { dst, .. }
        | Instr::BuildArgv { dst }
        | Instr::AllocDyn { dst, .. }
        | Instr::StackAlloc { dst, .. }
        | Instr::VTableAddr { dst, .. }
        | Instr::CallFnAddr { dst, .. }
        | Instr::FnAddr { dst, .. }
        | Instr::LoadStaticAddr { dst, .. }
        | Instr::LoadField { dst, .. }
        | Instr::LoadByte { dst, .. }
        | Instr::LoadStrAddr { dst, .. }
        | Instr::MemCmp { dst, .. }
        | Instr::ScanInt { dst }
        | Instr::HandlerEnter { dst }
        | Instr::LoadExc { dst }
        | Instr::TaskAlloc { dst, .. }
        | Instr::CoroutineResume { dst, .. } => *dst,
        _ => return None,
    })
}

fn push_uses(i: &Instr, mut use_v: impl FnMut(VReg)) {
    match i {
        Instr::Mov { a, .. } | Instr::Neg { a, .. } | Instr::FNeg { a, .. } => use_v(*a),
        Instr::Bin { a, b, .. }
        | Instr::FBin { a, b, .. }
        | Instr::Cmp { a, b, .. }
        | Instr::FCmp { a, b, .. } => {
            use_v(*a);
            use_v(*b);
        }
        Instr::F2I { src, .. } | Instr::I2F { src, .. } => use_v(*src),
        Instr::CallFn { args, .. }
        | Instr::CallStatic { args, .. }
        | Instr::CallImport { args, .. } => {
            for a in args {
                use_v(*a);
            }
        }
        Instr::CallVirt { recv, args, .. } | Instr::CallByName { recv, args, .. } => {
            use_v(*recv);
            for a in args {
                use_v(*a);
            }
        }
        Instr::CallFnAddr { target, args, .. } => {
            use_v(*target);
            for a in args {
                use_v(*a);
            }
        }
        Instr::AllocDyn { bytes, .. } => use_v(*bytes),
        Instr::LoadField { base, .. }
        | Instr::LoadByte { base, .. }
        | Instr::StoreField { base, .. } => {
            use_v(*base);
            if let Instr::StoreField { src, .. } = i {
                use_v(*src);
            }
        }
        Instr::MemCpy { dst, src, len } => {
            use_v(*dst);
            use_v(*src);
            use_v(*len);
        }
        Instr::MemCmp { a, b, len, .. } => {
            use_v(*a);
            use_v(*b);
            use_v(*len);
        }
        Instr::WriteString { src } | Instr::WriteInt { src } | Instr::EWriteInt { src } => {
            use_v(*src)
        }
        Instr::Retain { src } | Instr::Release { src, .. } | Instr::Drop { src } => use_v(*src),
        Instr::Suspend { state } => use_v(*state),
        Instr::CoroutineResume { task, .. } => use_v(*task),
        _ => {}
    }
}

fn verify_function(f: &IrFunction) -> Result<(), String> {
    let n = f.blocks.len();
    if n == 0 {
        return Err(format!("function '{}' has no blocks", f.name));
    }
    let sites = collect(f)?;
    let succs = successors(f);
    let reachable = reachable_from(&succs);
    let preds = predecessors(&succs, n);
    let rpo = reverse_postorder(&succs, &reachable);
    let idom = dominators(n, &preds, &rpo, &reachable);

    for &(v, bi, ii, term) in &sites.uses {
        let v = VReg(v);
        if v.0 as usize >= f.vregs as usize {
            return Err(format!(
                "function '{}' block {bi}: vreg {v:?} is beyond vregs={}",
                f.name, f.vregs
            ));
        }
        if !reachable[bi] {
            continue;
        }
        if definitely_assigned(f, &sites, &preds, &idom, &reachable, v, bi, ii, term) {
            continue;
        }
        let what = if term {
            format!("{:?}", f.blocks[bi].term)
        } else {
            format!("{:?}", f.blocks[bi].instrs[ii])
        };
        return Err(format!(
            "function '{}' block {bi} instruction {ii} ({what}): \
             vreg {v:?} is read but not assigned on every path",
            f.name
        ));
    }
    Ok(())
}

/// True when `v` is assigned before the use on every path.
#[allow(clippy::too_many_arguments)]
fn definitely_assigned(
    f: &IrFunction,
    sites: &Sites,
    preds: &[Vec<usize>],
    idom: &[Option<usize>],
    reachable: &[bool],
    v: VReg,
    bi: usize,
    ii: usize,
    term: bool,
) -> bool {
    let defs = &sites.defs[v.0 as usize];
    // A single definition that dominates the use (and precedes it in the
    // same block) is the common case: every lowered expression result.
    if defs.iter().any(|&(db, di)| {
        if db == bi {
            di < ii
        } else {
            dominates(idom, db, bi)
        }
    }) {
        return true;
    }
    // Multi-definition value (join target, loop-carried local): prove
    // definite assignment by dataflow. This is a "must" property, so it
    // is the greatest fixed point: start every reachable block at true
    // and lower it only when a predecessor can reach the block without
    // assigning the vreg. Starting from false instead would keep a loop
    // header false forever even though the loop body re-enters with the
    // value assigned on the entry path.
    let implicit = if f.receiver == ReceiverKind::None {
        // vreg 0 is reserved padding in static methods; parameters are
        // 1..=arity.
        v.0 >= 1 && v.0 <= f.arity as u32
    } else {
        // vreg 0 is the receiver, then the parameters.
        v.0 <= f.arity as u32
    };
    let mut in_def: Vec<bool> = (0..f.blocks.len())
        .map(|b| b != 0 && reachable[b])
        .collect();
    in_def[0] = implicit;
    let defines = |b: usize| defs.iter().any(|&(db, _)| db == b);
    let mut changed = true;
    while changed {
        changed = false;
        for b in 1..f.blocks.len() {
            if !reachable[b] {
                continue;
            }
            let mut val = true;
            for &p in &preds[b] {
                if !reachable[p] {
                    continue;
                }
                if !(in_def[p] || defines(p)) {
                    val = false;
                    break;
                }
            }
            if !val && in_def[b] {
                in_def[b] = false;
                changed = true;
            }
        }
    }
    if in_def[bi] {
        return true;
    }
    // Not assigned on entry to this block; a definition earlier in the
    // block still covers the use.
    defs.iter().any(|&(db, di)| db == bi && (term || di < ii))
}

fn successors(f: &IrFunction) -> Vec<Vec<usize>> {
    let n = f.blocks.len();
    let mut succs = vec![Vec::new(); n];
    for (bi, b) in f.blocks.iter().enumerate() {
        if let Terminator::Branch {
            if_true, if_false, ..
        } = &b.term
        {
            for &t in &[*if_true, *if_false] {
                let t = t as usize;
                if t < n && !succs[bi].contains(&t) {
                    succs[bi].push(t);
                }
            }
        }
    }
    succs
}

fn predecessors(succs: &[Vec<usize>], n: usize) -> Vec<Vec<usize>> {
    let mut preds = vec![Vec::new(); n];
    for (b, ss) in succs.iter().enumerate() {
        for &s in ss {
            preds[s].push(b);
        }
    }
    preds
}

fn reachable_from(succs: &[Vec<usize>]) -> Vec<bool> {
    let mut seen = vec![false; succs.len()];
    let mut stack = vec![0usize];
    seen[0] = true;
    while let Some(b) = stack.pop() {
        for &s in &succs[b] {
            if !seen[s] {
                seen[s] = true;
                stack.push(s);
            }
        }
    }
    seen
}

/// Entry-first order used by the dominator intersection.
fn reverse_postorder(succs: &[Vec<usize>], reachable: &[bool]) -> Vec<usize> {
    let n = succs.len();
    let mut post = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    // Iterative DFS so deep CFGs cannot overflow the compiler stack.
    let mut stack = vec![(0usize, 0usize)];
    seen[0] = true;
    while let Some((b, next)) = stack.pop() {
        if next < succs[b].len() {
            let s = succs[b][next];
            stack.push((b, next + 1));
            if !seen[s] {
                seen[s] = true;
                stack.push((s, 0));
            }
        } else {
            post.push(b);
        }
    }
    post.reverse();
    debug_assert!(post.iter().all(|&b| reachable[b]));
    post
}

fn dominators(
    n: usize,
    preds: &[Vec<usize>],
    rpo: &[usize],
    reachable: &[bool],
) -> Vec<Option<usize>> {
    let mut order = vec![usize::MAX; n];
    for (i, &b) in rpo.iter().enumerate() {
        order[b] = i;
    }
    let mut idom = vec![None; n];
    idom[0] = Some(0);
    let mut changed = true;
    while changed {
        changed = false;
        for &b in rpo.iter().skip(1) {
            let mut new: Option<usize> = None;
            for &p in &preds[b] {
                if !reachable[p] || idom[p].is_none() {
                    continue;
                }
                new = Some(match new {
                    None => p,
                    Some(x) => intersect(x, p, &idom, &order),
                });
            }
            if let Some(new) = new {
                if idom[b] != Some(new) {
                    idom[b] = Some(new);
                    changed = true;
                }
            }
        }
    }
    idom
}

fn intersect(mut a: usize, mut b: usize, idom: &[Option<usize>], order: &[usize]) -> usize {
    while a != b {
        while order[a] > order[b] {
            a = idom[a].expect("processed dominator");
        }
        while order[b] > order[a] {
            b = idom[b].expect("processed dominator");
        }
    }
    a
}

fn dominates(idom: &[Option<usize>], def: usize, user: usize) -> bool {
    let mut b = user;
    loop {
        if b == def {
            return true;
        }
        match idom[b] {
            Some(p) if p != b => b = p,
            _ => return false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::ir::instructions::{Block, IrFunction, ReceiverKind, Terminator};

    fn func(blocks: Vec<Block>, arity: usize) -> IrFunction {
        let mut f = IrFunction::new("t", ReceiverKind::None, arity);
        f.blocks = blocks;
        f.vregs = 8;
        f
    }

    fn b(instrs: Vec<Instr>, term: Terminator) -> Block {
        Block { instrs, term }
    }

    #[test]
    fn rejects_definition_in_sibling_arm() {
        // b0: c = const; branch(c, b1, b2)
        // b1: x = const; goto b3      b2: goto b3
        // b3: use x
        let x = VReg(3);
        let c = VReg(1);
        let f = func(
            vec![
                b(
                    vec![Instr::Const { dst: c, imm: 1 }],
                    Terminator::Branch {
                        cond: Some(c),
                        if_true: 1,
                        if_false: 2,
                    },
                ),
                b(
                    vec![Instr::Const { dst: x, imm: 7 }],
                    Terminator::Branch {
                        cond: None,
                        if_true: 3,
                        if_false: 3,
                    },
                ),
                b(
                    vec![],
                    Terminator::Branch {
                        cond: None,
                        if_true: 3,
                        if_false: 3,
                    },
                ),
                b(vec![], Terminator::Ret(Some(x))),
            ],
            0,
        );
        let err = verify_function(&f).expect_err("must be rejected");
        assert!(err.contains("not assigned on every path"), "{err}");
    }

    #[test]
    fn accepts_join_assigned_in_both_arms() {
        let x = VReg(3);
        let c = VReg(1);
        let f = func(
            vec![
                b(
                    vec![Instr::Const { dst: c, imm: 1 }],
                    Terminator::Branch {
                        cond: Some(c),
                        if_true: 1,
                        if_false: 2,
                    },
                ),
                b(
                    vec![Instr::Const { dst: x, imm: 7 }],
                    Terminator::Branch {
                        cond: None,
                        if_true: 3,
                        if_false: 3,
                    },
                ),
                b(
                    vec![Instr::Const { dst: x, imm: 9 }],
                    Terminator::Branch {
                        cond: None,
                        if_true: 3,
                        if_false: 3,
                    },
                ),
                b(vec![], Terminator::Ret(Some(x))),
            ],
            0,
        );
        verify_function(&f).expect("both arms assign");
    }

    #[test]
    fn rejects_undefined_vreg() {
        let f = func(vec![b(vec![], Terminator::Ret(Some(VReg(4))))], 0);
        let err = verify_function(&f).expect_err("must be rejected");
        assert!(err.contains("not assigned on every path"), "{err}");
    }

    #[test]
    fn accepts_parameter_used_in_a_loop() {
        // The loop back edge must not hide that the parameter is defined
        // on entry: a least-fixed-point dataflow kept the header false
        // forever and rejected every loop that used its parameter.
        let x = VReg(1);
        let f = func(
            vec![
                b(
                    vec![],
                    Terminator::Branch {
                        cond: None,
                        if_true: 1,
                        if_false: 1,
                    },
                ),
                b(
                    vec![],
                    Terminator::Branch {
                        cond: Some(x),
                        if_true: 2,
                        if_false: 3,
                    },
                ),
                b(
                    vec![],
                    Terminator::Branch {
                        cond: None,
                        if_true: 1,
                        if_false: 1,
                    },
                ),
                b(vec![], Terminator::Ret(Some(x))),
            ],
            1,
        );
        verify_function(&f).expect("parameter is assigned on entry");
    }
}
