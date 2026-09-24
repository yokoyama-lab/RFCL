"""Multi-level Bennett transformation via the reversible pebble game.

Bennett 1989 ("Time/Space Trade-offs for Reversible Computation") makes an
irreversible T-step computation x_0 -> x_1 -> ... -> x_T reversible without
keeping the whole history.  Each step x_{i+1} = f(x_i) is run *out of place*
into a fresh checkpoint register, and a checkpoint is erased by running the
same step backwards, which needs x_{i-1} to still be present.  Which
checkpoints exist at any time is a configuration of the reversible pebble
game on the path 0-1-...-n:

  * node 0 (the input) always carries a pebble;
  * a pebble may be put on, or taken off, node i only while node i-1 has one;
  * the game ends with a pebble on node n and on no other node except 0.

A *schedule* is the list of moves.  Its length is the number of step
applications (time) and its maximum number of simultaneous pebbles, not
counting node 0, is the number of checkpoint registers (space).

This module provides

  * three schedule families: ``linear_schedule`` (Bennett 1973, keep the whole
    history), ``bennett_schedule`` (Bennett 1989, k levels of m segments) and
    ``optimal_schedule`` (fewest moves under a pebble budget, by the recursion
    of Knill 1995), plus ``bfs_min_moves``, an exhaustive search used to check
    the recursion on small instances;
  * ``compile_pebbling``: turns a step program and a schedule into a clean SRL
    program (straight-line: one renamed copy of the step, or of its inverse,
    per move);
  * ``analyze_pebbling``: runs that program and reports measured time/space
    next to the schedule's exact counts.

The step program is an ordinary SRL program ``(X) (X Y) (temps)``: it reads
the state X, leaves X unchanged, writes the next state into Y (which starts at
zero, so Y must have the same length as X) and returns its temps to zero.
Only scalar state variables are supported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import count

from pyrev_fl.ast import (
    ArrayRef, Assign, Binary, Block, Const, Expr, If, Loop, Place, Pop, Program,
    Push, Rif, Stmt, Swap, Var,
)
from pyrev_fl.check import check_program
from pyrev_fl.invert import invert_block

Move = int  # +i puts a pebble on node i, -i removes it (i >= 1)


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScheduleStats:
    n: int            # length of the path (number of forward steps)
    moves: int        # step applications
    pebbles: int      # max simultaneous pebbles, excluding node 0


def validate_schedule(moves: list[Move], n: int) -> ScheduleStats:
    """Check that *moves* is a legal complete pebbling of 0..n; return its cost."""
    pebbled = {0}
    peak = 0
    for mv in moves:
        i = abs(mv)
        if mv == 0 or i > n:
            raise ValueError(f"move {mv} out of range 1..{n}")
        if i - 1 not in pebbled:
            raise ValueError(f"move {mv}: node {i - 1} carries no pebble")
        if mv > 0:
            if i in pebbled:
                raise ValueError(f"move {mv}: node {i} already pebbled")
            pebbled.add(i)
        else:
            if i not in pebbled:
                raise ValueError(f"move {mv}: node {i} carries no pebble")
            pebbled.remove(i)
        peak = max(peak, len(pebbled) - 1)
    if pebbled != {0, n}:
        raise ValueError(f"final configuration {sorted(pebbled)} is not {{0, {n}}}")
    return ScheduleStats(n=n, moves=len(moves), pebbles=peak)


def _shift(moves: list[Move], offset: int) -> list[Move]:
    return [mv + offset if mv > 0 else mv - offset for mv in moves]


def _reverse(moves: list[Move]) -> list[Move]:
    """The move sequence that undoes *moves* (removes what it placed)."""
    return [-mv for mv in reversed(moves)]


def linear_schedule(n: int) -> list[Move]:
    """Bennett 1973: compute all n checkpoints, then erase 1..n-1 backwards.

    2n - 1 moves, n pebbles."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return list(range(1, n + 1)) + [-i for i in range(n - 1, 0, -1)]


def bennett_schedule(k: int, m: int) -> list[Move]:
    """Bennett 1989 with k levels of m segments each; covers n = m**k steps.

    Level j advances m**j steps: it runs level j-1 m times forward, leaving
    m checkpoints, then erases the m-1 intermediate ones by running level j-1
    backwards.  (2m - 1)**k moves, k(m - 1) + 1 pebbles."""
    if k < 0 or m < 2:
        raise ValueError("need k >= 0 and m >= 2")
    if k == 0:
        return [1]
    sub = bennett_schedule(k - 1, m)
    width = m ** (k - 1)
    moves: list[Move] = []
    for i in range(m):
        moves += _shift(sub, i * width)
    for i in range(m - 2, -1, -1):
        moves += _shift(_reverse(sub), i * width)
    return moves


@lru_cache(maxsize=None)
def _opt(n: int, s: int) -> tuple[float, int]:
    """(min moves, best split) to pebble node n from node 0 with s pebbles.

    Knill 1995's recursion: pebble some m < n with s pebbles, pebble n from m
    with the s - 1 pebbles left, then remove m with s - 1 pebbles (n is held).
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    if s < 1 or n > 2 ** (s - 1):
        return float("inf"), 0
    if n == 1:
        return 1, 0
    best, arg = float("inf"), 0
    for m in range(1, n):
        c = _opt(m, s)[0] + _opt(n - m, s - 1)[0] + _opt(m, s - 1)[0]
        if c < best:
            best, arg = c, m
    return best, arg


def min_moves(n: int, s: int) -> float:
    """Fewest moves of the recursion above (inf when n > 2**(s-1)).

    With s pebbles (the final one included, node 0 excluded) the farthest
    reachable node is 2**(s-1); bfs_min_moves agrees on every n <= 16."""
    return _opt(n, s)[0]


def optimal_schedule(n: int, s: int) -> list[Move]:
    """A schedule reaching node n with at most s pebbles in min_moves(n, s) moves."""
    cost, m = _opt(n, s)
    if cost == float("inf"):
        raise ValueError(f"{s} pebbles reach at most node {2 ** (s - 1)}, not {n}")
    if n == 1:
        return [1]
    head = optimal_schedule(m, s)
    tail = _shift(optimal_schedule(n - m, s - 1), m)
    undo = _reverse(optimal_schedule(m, s - 1))
    return head + tail + undo


def bfs_min_moves(n: int, s: int) -> float:
    """Exact minimum number of moves, by breadth-first search over all
    configurations with at most s pebbles (exponential; for checking only)."""
    goal = 1 << (n - 1)
    frontier, seen, depth = [0], {0}, 0
    while frontier:
        if goal in seen:
            return depth
        nxt = []
        for conf in frontier:
            for i in range(1, n + 1):
                if i > 1 and not conf >> (i - 2) & 1:
                    continue
                c2 = conf ^ (1 << (i - 1))
                if c2 in seen or bin(c2).count("1") > s:
                    continue
                seen.add(c2)
                nxt.append(c2)
        frontier, depth = nxt, depth + 1
    return float("inf")


# ---------------------------------------------------------------------------
# Renaming
# ---------------------------------------------------------------------------

def _rn(name: str, env: dict[str, str]) -> str:
    return env.get(name, name)


def _rename_expr(e: Expr, env: dict[str, str]) -> Expr:
    if isinstance(e, Const):
        return e
    if isinstance(e, Var):
        return Var(_rn(e.name, env))
    if isinstance(e, ArrayRef):
        return ArrayRef(_rn(e.name, env), _rename_expr(e.index, env))
    if isinstance(e, Binary):
        return Binary(e.op, _rename_expr(e.left, env), _rename_expr(e.right, env))
    raise TypeError(f"unknown expression: {e!r}")


def _rename_place(p: str | Place, env: dict[str, str]) -> str | Place:
    if isinstance(p, str):
        return _rn(p, env)
    return _rename_expr(p, env)  # type: ignore[return-value]


def _rename_block(b: Block, env: dict[str, str]) -> Block:
    return Block([_rename_stmt(s, env) for s in b.stmts])


def _rename_stmt(s: Stmt, env: dict[str, str]) -> Stmt:
    if isinstance(s, Assign):
        return Assign(_rename_place(s.target, env), s.op, _rename_expr(s.expr, env))
    if isinstance(s, Swap):
        return Swap(_rename_place(s.left, env), _rename_place(s.right, env))
    if isinstance(s, If):
        return If(_rename_expr(s.test, env), _rename_block(s.then_block, env),
                  _rename_block(s.else_block, env), _rename_expr(s.assertion, env))
    if isinstance(s, Loop):
        return Loop(_rename_expr(s.entry_guard, env), _rename_block(s.do_block, env),
                    _rename_block(s.loop_block, env), _rename_expr(s.exit_guard, env))
    if isinstance(s, Rif):
        return Rif(_rename_expr(s.test, env), _rename_block(s.body, env),
                   _rename_expr(s.assertion, env))
    if isinstance(s, Push):
        return Push(_rn(s.var, env), _rn(s.stack, env))
    if isinstance(s, Pop):
        return Pop(_rn(s.var, env), _rn(s.stack, env))
    raise TypeError(f"unknown statement: {s!r}")


# ---------------------------------------------------------------------------
# Compilation to SRL
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StepSpec:
    """A step program split into its state (X), next-state (Y) and temps."""
    program: Program
    state: tuple[str, ...]
    next_state: tuple[str, ...]
    temps: tuple[str, ...]


def step_spec(program: Program) -> StepSpec:
    """Read a step program ``(X) (X Y) (temps)`` with |Y| = |X|, all scalar."""
    check_program(program)
    xs = list(program.inputs)
    outs = list(program.outputs)
    if outs[:len(xs)] != xs or len(outs) != 2 * len(xs):
        raise ValueError(
            "step program must have the form (X) (X Y) (temps) with |Y| = |X|; "
            f"got inputs {xs}, outputs {outs}")
    for name in xs + outs + list(program.temps):
        if "[" in name or ":" in name:
            raise ValueError(f"only scalar variables are supported: {name}")
    return StepSpec(program, tuple(xs), tuple(outs[len(xs):]), tuple(program.temps))


@dataclass
class Pebbling:
    program: Program                 # the compiled clean SRL program
    stats: ScheduleStats
    registers: int                   # checkpoint registers allocated (= stats.pebbles)
    output_vars: list[str]           # where x_n ends up
    moves: list[Move] = field(repr=False, default_factory=list)


def compile_pebbling(spec: StepSpec, moves: list[Move], n: int) -> Pebbling:
    """Expand a schedule into a straight-line clean SRL program.

    Node 0 is the program's input (the step's own state names).  Every other
    pebble lives in a checkpoint register ``<x>__p<r>``; registers are reused
    lowest-free-first, so the program declares exactly ``stats.pebbles`` of
    them.  Putting a pebble on i runs the step from i-1's register into a free
    register; removing it runs the inverted step on the same pair, which
    returns that register to zero.  The step's temps are shared by all moves
    (they are zero between steps).
    """
    stats = validate_schedule(moves, n)
    xs, ys = spec.state, spec.next_state
    taken = set(xs) | set(ys) | set(spec.temps)

    def reg_names(r: int) -> tuple[str, ...]:
        names = tuple(f"{x}__p{r}" for x in xs)
        clash = taken.intersection(names)
        if clash:
            raise ValueError(f"register name clash: {sorted(clash)}")
        return names

    fwd = spec.program.body
    bwd = invert_block(fwd)
    where: dict[int, tuple[str, ...]] = {0: xs}
    free: list[int] = []
    fresh = count(1)
    used_regs: list[int] = []
    stmts: list[Stmt] = []
    for mv in moves:
        i = abs(mv)
        if mv > 0:
            r = min(free) if free else next(fresh)
            if r in free:
                free.remove(r)
            else:
                used_regs.append(r)
            where[i] = reg_names(r)
            block = fwd
        else:
            block = bwd
        env = dict(zip(xs, where[i - 1]))
        env.update(zip(ys, where[i]))
        stmts.extend(_rename_block(block, env).stmts)
        if mv < 0:
            r = int(where.pop(i)[0].rsplit("__p", 1)[1])
            free.append(r)

    out = list(where[n])
    regs = [nm for r in used_regs for nm in reg_names(r)]
    temps = [nm for nm in regs if nm not in out] + list(spec.temps)
    program = Program(inputs=list(xs), outputs=list(xs) + out, temps=temps,
                      body=Block(stmts))
    check_program(program)
    return Pebbling(program=program, stats=stats, registers=len(used_regs),
                    output_vars=out, moves=list(moves))


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

@dataclass
class PebblingResult:
    stats: ScheduleStats
    time_steps: int             # measured statement executions
    peak_space: int             # measured peak of non-zero variables
    declared_space: int         # variables declared by the compiled program
    step_time: float            # measured time of the plain forward run / n
    baseline_time: int          # measured time of n forward steps (history kept)
    output: dict[str, int]      # x_n, keyed by the step's state names


def analyze_pebbling(spec: StepSpec, moves: list[Move], n: int,
                     inputs: list[int]) -> PebblingResult:
    """Compile, run and measure a pebbling; also measure the plain n-step run."""
    from pyrev_fl.interpreter import run_program
    from pyrev_fl.tradeoff import _measure  # local: tradeoff imports this module

    peb = compile_pebbling(spec, moves, n)
    metrics = _measure(peb.program, inputs)
    # Baseline: n forward steps with every intermediate state kept, i.e. the
    # history-keeping (Landauer) embedding.  It is not a complete pebbling,
    # so it is built by _forward_only rather than compile_pebbling.
    base_metrics = _measure(_forward_only(spec, n), inputs)
    store = run_program(peb.program, inputs)
    declared = len(set(peb.program.inputs) | set(peb.program.outputs)) + len(peb.program.temps)
    return PebblingResult(
        stats=peb.stats,
        time_steps=metrics.time_steps,
        peak_space=metrics.peak_space,
        declared_space=declared,
        step_time=base_metrics.time_steps / n,
        baseline_time=base_metrics.time_steps,
        output={x: store[v] for x, v in zip(spec.state, peb.output_vars)},
    )


def _forward_only(spec: StepSpec, n: int) -> Program:
    """n forward steps keeping every intermediate state (no uncomputation)."""
    xs, ys = spec.state, spec.next_state
    regs = [xs] + [tuple(f"{x}__h{i}" for x in xs) for i in range(1, n + 1)]
    stmts: list[Stmt] = []
    for i in range(1, n + 1):
        env = dict(zip(xs, regs[i - 1]))
        env.update(zip(ys, regs[i]))
        stmts.extend(_rename_block(spec.program.body, env).stmts)
    outs = [nm for r in regs[1:] for nm in r]
    return Program(inputs=list(xs), outputs=list(xs) + outs,
                   temps=list(spec.temps), body=Block(stmts))


def iterate_reference(spec: StepSpec, n: int, inputs: list[int]) -> list[int]:
    """x_n computed by running the step program n times (reference result)."""
    from pyrev_fl.interpreter import run_program
    state = list(inputs)
    for _ in range(n):
        store = run_program(spec.program, state)
        state = [store[y] for y in spec.next_state]
    return state
