"""Bennett time-space tradeoff analyzer (single level).

Measures the time (steps) and space (peak non-zero variables) of
1. the original computation and
2. its single-level Bennett transform (``bennett.make_reversible_program``),
and compares them with the exact counts of that transform:

- time  T_rev = 2T + |outputs| + 7 under this module's step accounting:
  2T for forward + reverse, |outputs| + 2 for the copies and the two flag
  updates, and 5 control steps (loop entry, the one loop iteration, loop
  exit, and one ``rif`` dispatch in each direction);
- space S_rev <= S + |outputs| + 1 (copies + flag); below it only when an
  output value is 0.

The +7 was measured in experiments/pebbling (28 runs, constant); earlier
versions predicted +2 and undercounted by exactly those 5 control steps.

Multi-level Bennett (the pebble game) is in ``pyrev_fl.pebble``.  With k
levels of m segments per level, Bennett 1989 gives, for n = m**k steps of a
computation with state size S,

- time  (2m - 1)**k step applications, i.e. T_rev = T * ((2m - 1)/m)**k,
- space k(m - 1) + 1 checkpoints, i.e. S_rev = S * (k(m - 1) + 1).

The often quoted "T * 2**k time, S + k*log2(T) space" matches neither
scheme: nested single-level embeddings double the time per level but add a
constant per level, and Bennett 1989 multiplies S rather than adding log T.

Reference: Bennett 1989, "Time/Space Trade-offs for Reversible Computation".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from pyrev_fl.ast import (
    Assign, Block, If, Loop, Pop, Program, Push, Rif, Stmt, Swap, UpdateOp, Var,
)
from pyrev_fl.bennett import make_reversible_program
from pyrev_fl.check import check_program, CheckError
from pyrev_fl.interface import build_layout
from pyrev_fl.interpreter import (
    EvalError, _eval_expr, _initialize_store, _read_place, _write_place,
    _validate_final_store,
)
from pyrev_fl.invert import invert_block

Store = dict[str, int]


@dataclass
class TSMetrics:
    """Time-space metrics for a single execution."""
    time_steps: int = 0
    peak_space: int = 0       # max non-zero variables at any point
    total_space: int = 0      # total variable slots used
    space_trace: list[int] = field(default_factory=list)


@dataclass
class TradeoffResult:
    """Complete tradeoff analysis."""
    original: TSMetrics
    bennett: TSMetrics
    time_ratio: float           # T_bennett / T_original
    space_ratio: float          # S_bennett / S_original
    theoretical_time_ratio: float   # predicted T_ratio
    theoretical_space_bound: int    # predicted S_bennett


def analyze_tradeoff(program: Program, inputs: list[int]) -> TradeoffResult:
    """Run both original and Bennett-transformed programs, compare metrics."""
    # Measure original
    orig_metrics = _measure(program, inputs)

    # Apply Bennett transform
    input_set = set(program.inputs)
    output_only = [o for o in program.outputs if o not in input_set]
    # Deduplicate temps to avoid checker errors
    seen: set[str] = set(program.inputs) | set(program.outputs)
    unique_temps: list[str] = []
    for t in output_only + list(program.temps):
        if t not in seen:
            seen.add(t)
            unique_temps.append(t)
    bennett_prog = make_reversible_program(
        body_stmts=list(program.body.stmts),
        inputs=list(program.inputs),
        outputs=list(program.outputs),
        temps=unique_temps,
    )
    bennett_metrics = _measure(bennett_prog, inputs)

    # Exact counts for single-level Bennett (see module docstring):
    # 2T forward + reverse, |outputs| copies, 2 flag updates, 5 control steps.
    T = max(orig_metrics.time_steps, 1)
    S = max(orig_metrics.peak_space, 1)
    theoretical_time = 2 * T + len(program.outputs) + 7
    n_outputs = len(program.outputs)

    time_ratio = bennett_metrics.time_steps / T if T > 0 else 0
    space_ratio = bennett_metrics.peak_space / S if S > 0 else 0

    return TradeoffResult(
        original=orig_metrics,
        bennett=bennett_metrics,
        time_ratio=time_ratio,
        space_ratio=space_ratio,
        theoretical_time_ratio=theoretical_time / T,
        theoretical_space_bound=S + n_outputs + 1,  # +outputs for copies, +1 for flag
    )


def _measure(program: Program, inputs: list[int]) -> TSMetrics:
    """Execute program and measure time/space."""
    layout = build_layout(program.inputs, program.outputs, program.temps)
    if len(inputs) != len(layout.inputs):
        raise EvalError(f"arity mismatch: expected {len(layout.inputs)}, got {len(inputs)}")
    try:
        check_program(program)
    except CheckError as exc:
        raise EvalError(str(exc)) from exc

    store = _initialize_store(program, inputs, layout)
    metrics = TSMetrics(total_space=len([k for k in store if not k.startswith("__")]))
    _measure_block(program.body, store, layout, metrics)
    _validate_final_store(store, layout)
    return metrics


def _count_nonzero(store: Store) -> int:
    """Count non-zero variables (excluding internal stack keys)."""
    return sum(1 for k, v in store.items() if not k.startswith("__") and v != 0)


def _record_space(store: Store, metrics: TSMetrics) -> None:
    nz = _count_nonzero(store)
    metrics.space_trace.append(nz)
    if nz > metrics.peak_space:
        metrics.peak_space = nz


def _measure_block(block: Block, store: Store, layout, metrics: TSMetrics) -> None:
    for stmt in block.stmts:
        _measure_stmt(stmt, store, layout, metrics)


def _measure_stmt(stmt: Stmt, store: Store, layout, metrics: TSMetrics) -> None:
    if isinstance(stmt, Assign):
        rhs = _eval_expr(stmt.expr, store, layout)
        old = _read_place(stmt.target, store, layout)
        if stmt.op is UpdateOp.ADD:
            _write_place(stmt.target, old + rhs, store, layout)
        elif stmt.op is UpdateOp.SUB:
            _write_place(stmt.target, old - rhs, store, layout)
        else:
            _write_place(stmt.target, old ^ rhs, store, layout)
        metrics.time_steps += 1
        _record_space(store, metrics)
        return

    if isinstance(stmt, Swap):
        l = _read_place(stmt.left, store, layout)
        r = _read_place(stmt.right, store, layout)
        _write_place(stmt.left, r, store, layout)
        _write_place(stmt.right, l, store, layout)
        metrics.time_steps += 1
        _record_space(store, metrics)
        return

    if isinstance(stmt, If):
        cond = _eval_expr(stmt.test, store, layout) != 0
        metrics.time_steps += 1
        _record_space(store, metrics)
        if cond:
            _measure_block(stmt.then_block, store, layout, metrics)
            if _eval_expr(stmt.assertion, store, layout) == 0:
                raise EvalError("if assertion must hold after then branch")
        else:
            _measure_block(stmt.else_block, store, layout, metrics)
            if _eval_expr(stmt.assertion, store, layout) != 0:
                raise EvalError("if assertion must be false after else branch")
        return

    if isinstance(stmt, Loop):
        if _eval_expr(stmt.entry_guard, store, layout) == 0:
            raise EvalError("loop entry guard must hold")
        metrics.time_steps += 1
        _record_space(store, metrics)
        _measure_block(stmt.do_block, store, layout, metrics)
        while _eval_expr(stmt.exit_guard, store, layout) == 0:
            metrics.time_steps += 1
            _measure_block(stmt.loop_block, store, layout, metrics)
            if _eval_expr(stmt.entry_guard, store, layout) != 0:
                raise EvalError("loop entry guard must be false between iterations")
            _measure_block(stmt.do_block, store, layout, metrics)
        metrics.time_steps += 1
        _record_space(store, metrics)
        return

    if isinstance(stmt, Rif):
        cond = _eval_expr(stmt.test, store, layout) != 0
        metrics.time_steps += 1
        _record_space(store, metrics)
        if cond:
            _measure_block(invert_block(stmt.body), store, layout, metrics)
            if _eval_expr(stmt.assertion, store, layout) == 0:
                raise EvalError("rif assertion failed")
        else:
            _measure_block(stmt.body, store, layout, metrics)
            if _eval_expr(stmt.assertion, store, layout) != 0:
                raise EvalError("rif assertion failed")
        return

    if isinstance(stmt, Push):
        val = store.get(stmt.var, 0)
        sk = f"__stack_{stmt.stack}"
        if sk not in store:
            store[sk] = []  # type: ignore
        store[sk].append(val)  # type: ignore
        store[stmt.var] = 0
        metrics.time_steps += 1
        _record_space(store, metrics)
        return

    if isinstance(stmt, Pop):
        if store.get(stmt.var, 0) != 0:
            raise EvalError(f"pop target must be zero: {stmt.var}")
        sk = f"__stack_{stmt.stack}"
        stack = store.get(sk, [])
        if not stack:
            raise EvalError(f"pop from empty stack")
        store[stmt.var] = stack.pop()  # type: ignore
        metrics.time_steps += 1
        _record_space(store, metrics)
        return

    raise TypeError(f"unknown statement: {stmt!r}")


def format_tradeoff(result: TradeoffResult) -> str:
    """Human-readable tradeoff report."""
    lines = [
        "Bennett Time-Space Tradeoff Analysis",
        "=" * 50,
        "",
        "Original computation:",
        f"  Time (steps):          {result.original.time_steps}",
        f"  Space (peak non-zero): {result.original.peak_space}",
        "",
        "Bennett-transformed computation:",
        f"  Time (steps):          {result.bennett.time_steps}",
        f"  Space (peak non-zero): {result.bennett.peak_space}",
        "",
        "Measured ratios:",
        f"  Time ratio:  {result.time_ratio:.2f}x",
        f"  Space ratio: {result.space_ratio:.2f}x",
        "",
        "Theoretical predictions (single-level Bennett):",
        f"  Time ratio:  {result.theoretical_time_ratio:.2f}x ((2T + |outputs| + 7) / T)",
        f"  Space bound: {result.theoretical_space_bound} (S + |outputs| + 1)",
        "",
        "Space trace (original):  " + _sparkline(result.original.space_trace),
        "Space trace (Bennett):   " + _sparkline(result.bennett.space_trace),
    ]
    return "\n".join(lines)


def _sparkline(values: list[int]) -> str:
    """Render a simple ASCII sparkline."""
    if not values:
        return "(empty)"
    mx = max(values) if values else 1
    if mx == 0:
        return "." * len(values)
    chars = " ▁▂▃▄▅▆▇█"
    return "".join(chars[min(int(v / mx * 8), 8)] for v in values)
