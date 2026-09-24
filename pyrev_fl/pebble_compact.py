"""Compact (loop-based) compilation of Bennett's multi-level scheme to SRL.

``pebble.compile_pebbling`` unrolls a schedule, so its program has one copy of
the step per move: code size grows with the running time.  This module
compiles Bennett 1989's scheme with segment counts ``ms`` (``ms[0]`` is the
innermost level) into a program whose size is O(len(ms) + |step|),
independent of n = prod(ms):

* checkpoints live in arrays, one per state variable, ``<x>__ck[F + 1]``;
  cell 0 holds the input and cell F = sum(m - 1) + 1 the result, cells
  1..F-1 the intermediate checkpoints (level j owns m_j - 1 of them);
* level j is one ``from``-loop over t = 0 .. 2m_j - 2.  Iteration t runs the
  level below *once in the text*, under ``rif``: forward for t < m_j
  (placing checkpoint t + 1), backwards for t >= m_j (erasing checkpoint
  2m_j - 1 - t).  The source/target cells of the level below are computed
  from (t, own source, own target) before and uncomputed after;
* level 0 (the leaf) loads the source cell into scalars, runs the step,
  swaps the result into the target cell and unloads the source.  Run
  backwards, the same block erases the target cell.

The checker forbids ``X[d] += f(X[s])`` (one array on both sides), which is
why the leaf goes through the scalars ``<x>__in`` / ``<y>__out``.

Only Bennett's regular scheme is compiled this way; the fewest-move
schedules of ``pebble.optimal_schedule`` split unevenly and would need a
control stack.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod

from pyrev_fl.ast import (
    ArrayRef, Assign, BinOp, Binary, Block, Const, If, Loop, Program, Rif, Stmt,
    Swap, UpdateOp, Var,
)
from pyrev_fl.check import check_program
from pyrev_fl.invert import invert_block
from pyrev_fl.pebble import StepSpec, _rename_block

ADD, SUB, XOR = UpdateOp.ADD, UpdateOp.SUB, UpdateOp.XOR


def _c(v: int) -> Const:
    return Const(v)


def _v(n: str) -> Var:
    return Var(n)


def _b(op: BinOp, a, b) -> Binary:
    return Binary(op, a, b)


@dataclass
class CompactPebbling:
    program: Program
    ms: list[int]
    n: int                 # prod(ms)
    registers: int         # checkpoint cells besides the input: sum(m - 1) + 1
    output_vars: list[str]


def compile_bennett_compact(spec: StepSpec, ms: list[int]) -> CompactPebbling:
    if not ms or any(m < 2 for m in ms):
        raise ValueError("need at least one level, every level with m >= 2")
    xs, ys = spec.state, spec.next_state
    k = len(ms)
    F = sum(m - 1 for m in ms) + 1
    arr = {x: f"{x}__ck" for x in xs}
    xin = {x: f"{x}__in" for x in xs}
    yout = {y: f"{y}__out" for y in ys}
    outs = [f"{x}__final" for x in xs]
    src = [f"_pk_s{j}" for j in range(k + 1)]   # src[j]: source cell of a level-j run
    dst = [f"_pk_d{j}" for j in range(k + 1)]
    cnt = [f"_pk_t{j}" for j in range(k + 1)]   # cnt[j], idx[j] used by level j >= 1
    idx = [f"_pk_i{j}" for j in range(k + 1)]
    names = set(arr.values()) | set(xin.values()) | set(yout.values()) | set(outs)
    names |= set(src) | set(dst) | set(cnt[1:]) | set(idx[1:])
    clash = names & (set(xs) | set(ys) | set(spec.temps))
    if clash:
        raise ValueError(f"name clash with the step program: {sorted(clash)}")

    # level 0: one step from cell src[0] to cell dst[0]
    env = dict(xin)
    env.update(yout)
    leaf: list[Stmt] = [Assign(xin[x], XOR, ArrayRef(arr[x], _v(src[0]))) for x in xs]
    leaf += _rename_block(spec.program.body, env).stmts
    leaf += [Swap(ArrayRef(arr[x], _v(dst[0])), yout[y]) for x, y in zip(xs, ys)]
    leaf += [Assign(xin[x], XOR, ArrayRef(arr[x], _v(src[0]))) for x in xs]
    block = Block(leaf)

    base = 1
    for j in range(1, k + 1):
        m = ms[j - 1]
        t, i = _v(cnt[j]), _v(idx[j])
        fwd = _b(BinOp.LT, t, _c(m))
        setup = Block([
            If(fwd, Block([Assign(idx[j], ADD, _b(BinOp.ADD, t, _c(1)))]),
               Block([Assign(idx[j], ADD, _b(BinOp.SUB, _c(2 * m - 1), t))]), fwd),
            If(_b(BinOp.EQ, i, _c(1)), Block([Assign(src[j - 1], ADD, _v(src[j]))]),
               Block([Assign(src[j - 1], ADD, _b(BinOp.ADD, _c(base - 2), i))]),
               _b(BinOp.EQ, i, _c(1))),
            If(_b(BinOp.EQ, i, _c(m)), Block([Assign(dst[j - 1], ADD, _v(dst[j]))]),
               Block([Assign(dst[j - 1], ADD, _b(BinOp.ADD, _c(base - 1), i))]),
               _b(BinOp.EQ, i, _c(m))),
        ])
        back = _b(BinOp.GE, t, _c(m))
        body = setup.stmts + [Rif(back, block, back)] + invert_block(setup).stmts
        body.append(Assign(cnt[j], ADD, _c(1)))
        loop = Loop(_b(BinOp.EQ, t, _c(0)), Block(body), Block([]),
                    _b(BinOp.EQ, t, _c(2 * m - 1)))
        block = Block([loop, Assign(cnt[j], SUB, _c(2 * m - 1))])
        base += m - 1

    top: list[Stmt] = [Assign(ArrayRef(arr[x], _c(0)), XOR, _v(x)) for x in xs]
    top.append(Assign(dst[k], ADD, _c(F)))
    top += block.stmts
    top.append(Assign(dst[k], SUB, _c(F)))
    top += [Swap(ArrayRef(arr[x], _c(F)), o) for x, o in zip(xs, outs)]
    top += [Assign(ArrayRef(arr[x], _c(0)), XOR, _v(x)) for x in xs]

    temps = [f"{a}[{F + 1}]" for a in arr.values()] + list(xin.values()) + list(yout.values())
    temps += src + dst + cnt[1:] + idx[1:] + list(spec.temps)
    program = Program(inputs=list(xs), outputs=list(xs) + outs, temps=temps, body=Block(top))
    check_program(program)
    return CompactPebbling(program=program, ms=list(ms), n=prod(ms), registers=F,
                           output_vars=outs)


def control_overhead(ms: list[int], state_size: int) -> int:
    """Exact non-step time of ``compile_bennett_compact`` (tradeoff._measure accounting).

    A level-j run executes 2m_j - 1 iterations of 14 body steps (3 ifs and 3
    assignments, their inverses, the rif, the counter increment) and 2m_j
    loop-control steps (entry, 2m_j - 2 repetitions, exit), plus the counter
    reset: 15(2m_j - 1) + 2 in all.  A leaf run adds 3|X| steps
    (load, swap, unload) around the step; the top level adds 3|X| + 2."""
    runs = 1
    total = 3 * state_size + 2
    for m in reversed(ms):
        total += runs * ((2 * m - 1) * 15 + 2)
        runs *= 2 * m - 1
    return total + runs * 3 * state_size


def statement_count(block: Block) -> int:
    """Number of statements in the program text (control statements included)."""
    total = 0
    for st in block.stmts:
        total += 1
        if isinstance(st, If):
            total += statement_count(st.then_block) + statement_count(st.else_block)
        elif isinstance(st, Loop):
            total += statement_count(st.do_block) + statement_count(st.loop_block)
        elif isinstance(st, Rif):
            total += statement_count(st.body)
    return total
