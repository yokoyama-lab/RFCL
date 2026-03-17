"""Landauer principle entropy counter for SRL programs.

Landauer's principle: every bit of information erased during computation
dissipates kT ln 2 energy.  In a truly reversible program no information
is erased, so the logical entropy change is zero.

This module instruments SRL program execution and computes
information-theoretic metrics at every step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from pyrev_fl.ast import (
    ArrayRef,
    Assign,
    Binary,
    Block,
    Const,
    Expr,
    If,
    Loop,
    Place,
    Pop,
    Program,
    Push,
    Rif,
    Stmt,
    Swap,
    UpdateOp,
    Var,
)
from pyrev_fl.check import check_program
from pyrev_fl.interface import InterfaceLayout, build_layout
from pyrev_fl.interpreter import (
    EvalError,
    _eval_binary,
    _eval_expr,
    _initialize_store,
    _read_place,
    _validate_final_store,
    _write_place,
)
from pyrev_fl.invert import invert_block

Store = dict[str, int]


@dataclass
class StepMetric:
    """Per-step entropy information."""

    stmt_text: str
    kind: str  # "assign", "swap", "push", "pop", "if_test", "loop_guard"
    bits_erased: float
    bits_created: float


@dataclass
class LandauerMetrics:
    """Full Landauer analysis result."""

    total_steps: int  # total number of executed statements
    reversible_steps: int  # steps that preserve all information
    info_erased_bits: float  # total bits of information erased
    info_created_bits: float  # total bits of information created
    net_entropy: float  # net entropy change (should be 0 for reversible)
    step_details: list[StepMetric] = field(default_factory=list)
    irreversible_baseline_bits: float = 0.0  # what an irreversible version would erase


def _bits_for_value(value: int) -> float:
    """Return log2(|value| + 1): information content of a value."""
    return math.log2(abs(value) + 1)


def analyze_landauer(program: Program, inputs: list[int]) -> LandauerMetrics:
    """Run *program* with *inputs* and compute Landauer metrics at each step."""
    layout = build_layout(program.inputs, program.outputs, program.temps)
    if len(inputs) != len(layout.inputs):
        raise EvalError(
            f"arity mismatch: expected {len(layout.inputs)}, got {len(inputs)}"
        )
    check_program(program)
    store = _initialize_store(program, inputs, layout)

    metrics = LandauerMetrics(
        total_steps=0,
        reversible_steps=0,
        info_erased_bits=0.0,
        info_created_bits=0.0,
        net_entropy=0.0,
    )

    _analyze_block(program.body, store, layout, metrics)
    _validate_final_store(store, layout)

    metrics.net_entropy = metrics.info_erased_bits - metrics.info_created_bits
    return metrics


# -- block / statement walkers ------------------------------------------------


def _analyze_block(
    block: Block,
    store: Store,
    layout: InterfaceLayout,
    metrics: LandauerMetrics,
) -> None:
    for stmt in block.stmts:
        _analyze_stmt(stmt, store, layout, metrics)


def _analyze_stmt(
    stmt: Stmt,
    store: Store,
    layout: InterfaceLayout,
    metrics: LandauerMetrics,
) -> None:
    if isinstance(stmt, Assign):
        _analyze_assign(stmt, store, layout, metrics)
        return
    if isinstance(stmt, Swap):
        _analyze_swap(stmt, store, layout, metrics)
        return
    if isinstance(stmt, If):
        _analyze_if(stmt, store, layout, metrics)
        return
    if isinstance(stmt, Loop):
        _analyze_loop(stmt, store, layout, metrics)
        return
    if isinstance(stmt, Rif):
        _analyze_rif(stmt, store, layout, metrics)
        return
    if isinstance(stmt, Push):
        _analyze_push(stmt, store, layout, metrics)
        return
    if isinstance(stmt, Pop):
        _analyze_pop(stmt, store, layout, metrics)
        return
    raise TypeError(f"unknown statement: {stmt!r}")


# -- individual statement analyzers -------------------------------------------


def _analyze_assign(
    stmt: Assign,
    store: Store,
    layout: InterfaceLayout,
    metrics: LandauerMetrics,
) -> None:
    old_value = _read_place(stmt.target, store, layout)
    rhs = _eval_expr(stmt.expr, store, layout)

    # Execute
    if stmt.op is UpdateOp.ADD:
        _write_place(stmt.target, old_value + rhs, store, layout)
    elif stmt.op is UpdateOp.SUB:
        _write_place(stmt.target, old_value - rhs, store, layout)
    elif stmt.op is UpdateOp.XOR:
        _write_place(stmt.target, old_value ^ rhs, store, layout)
    else:
        raise TypeError(f"unknown update op: {stmt.op!r}")

    # Reversible update: old value recoverable from new value + rhs => 0 bits erased
    irr_baseline = _bits_for_value(old_value)
    metrics.irreversible_baseline_bits += irr_baseline

    step = StepMetric(
        stmt_text=_render_assign(stmt),
        kind="assign",
        bits_erased=0.0,
        bits_created=0.0,
    )
    metrics.step_details.append(step)
    metrics.total_steps += 1
    metrics.reversible_steps += 1


def _analyze_swap(
    stmt: Swap,
    store: Store,
    layout: InterfaceLayout,
    metrics: LandauerMetrics,
) -> None:
    left = _read_place(stmt.left, store, layout)
    right = _read_place(stmt.right, store, layout)
    _write_place(stmt.left, right, store, layout)
    _write_place(stmt.right, left, store, layout)

    # Swap is a permutation: 0 bits erased
    # Irreversible baseline: both old values lost
    irr_baseline = _bits_for_value(left) + _bits_for_value(right)
    metrics.irreversible_baseline_bits += irr_baseline

    step = StepMetric(
        stmt_text=_render_swap(stmt),
        kind="swap",
        bits_erased=0.0,
        bits_created=0.0,
    )
    metrics.step_details.append(step)
    metrics.total_steps += 1
    metrics.reversible_steps += 1


def _analyze_if(
    stmt: If,
    store: Store,
    layout: InterfaceLayout,
    metrics: LandauerMetrics,
) -> None:
    cond = _truthy(_eval_expr(stmt.test, store, layout))

    # Record the test evaluation step
    step = StepMetric(
        stmt_text=f"if test: {'true' if cond else 'false'}",
        kind="if_test",
        bits_erased=0.0,
        bits_created=0.0,
    )
    metrics.step_details.append(step)
    metrics.total_steps += 1
    metrics.reversible_steps += 1

    # Execute the chosen branch
    if cond:
        _analyze_block(stmt.then_block, store, layout, metrics)
        if not _truthy(_eval_expr(stmt.assertion, store, layout)):
            raise EvalError("if assertion must hold after then branch")
    else:
        _analyze_block(stmt.else_block, store, layout, metrics)
        if _truthy(_eval_expr(stmt.assertion, store, layout)):
            raise EvalError("if assertion must be false after else branch")


def _analyze_loop(
    stmt: Loop,
    store: Store,
    layout: InterfaceLayout,
    metrics: LandauerMetrics,
) -> None:
    if not _truthy(_eval_expr(stmt.entry_guard, store, layout)):
        raise EvalError("loop entry guard must hold before entering")

    # Record entry guard evaluation
    step = StepMetric(
        stmt_text="loop entry_guard: true",
        kind="loop_guard",
        bits_erased=0.0,
        bits_created=0.0,
    )
    metrics.step_details.append(step)
    metrics.total_steps += 1
    metrics.reversible_steps += 1

    _analyze_block(stmt.do_block, store, layout, metrics)

    while not _truthy(_eval_expr(stmt.exit_guard, store, layout)):
        # Record loop continuation guard
        step = StepMetric(
            stmt_text="loop exit_guard: false (continue)",
            kind="loop_guard",
            bits_erased=0.0,
            bits_created=0.0,
        )
        metrics.step_details.append(step)
        metrics.total_steps += 1
        metrics.reversible_steps += 1

        _analyze_block(stmt.loop_block, store, layout, metrics)
        if _truthy(_eval_expr(stmt.entry_guard, store, layout)):
            raise EvalError("loop entry guard must be false between iterations")
        _analyze_block(stmt.do_block, store, layout, metrics)

    # Record exit guard true
    step = StepMetric(
        stmt_text="loop exit_guard: true (exit)",
        kind="loop_guard",
        bits_erased=0.0,
        bits_created=0.0,
    )
    metrics.step_details.append(step)
    metrics.total_steps += 1
    metrics.reversible_steps += 1


def _analyze_rif(
    stmt: Rif,
    store: Store,
    layout: InterfaceLayout,
    metrics: LandauerMetrics,
) -> None:
    cond = _truthy(_eval_expr(stmt.test, store, layout))

    step = StepMetric(
        stmt_text=f"rif test: {'true' if cond else 'false'}",
        kind="if_test",
        bits_erased=0.0,
        bits_created=0.0,
    )
    metrics.step_details.append(step)
    metrics.total_steps += 1
    metrics.reversible_steps += 1

    if cond:
        _analyze_block(invert_block(stmt.body), store, layout, metrics)
        if not _truthy(_eval_expr(stmt.assertion, store, layout)):
            raise EvalError("rif assertion must hold after reverse branch")
    else:
        _analyze_block(stmt.body, store, layout, metrics)
        if _truthy(_eval_expr(stmt.assertion, store, layout)):
            raise EvalError("rif assertion must be false after forward branch")


def _analyze_push(
    stmt: Push,
    store: Store,
    layout: InterfaceLayout,
    metrics: LandauerMetrics,
) -> None:
    val = store.get(stmt.var, 0)
    stack_key = f"__stack_{stmt.stack}"
    if stack_key not in store:
        store[stack_key] = []  # type: ignore[assignment]
    store[stack_key].append(val)  # type: ignore[union-attr]
    store[stmt.var] = 0

    # Push is reversible: value is saved on stack and var is zeroed.
    # Information is preserved (moved, not erased).
    irr_baseline = _bits_for_value(val)
    metrics.irreversible_baseline_bits += irr_baseline

    step = StepMetric(
        stmt_text=f"push {stmt.var} {stmt.stack}",
        kind="push",
        bits_erased=0.0,
        bits_created=0.0,
    )
    metrics.step_details.append(step)
    metrics.total_steps += 1
    metrics.reversible_steps += 1


def _analyze_pop(
    stmt: Pop,
    store: Store,
    layout: InterfaceLayout,
    metrics: LandauerMetrics,
) -> None:
    if store.get(stmt.var, 0) != 0:
        raise EvalError(f"pop target must be zero: {stmt.var}={store[stmt.var]}")
    stack_key = f"__stack_{stmt.stack}"
    stack = store.get(stack_key, [])
    if not stack:
        raise EvalError(f"pop from empty stack: {stmt.stack}")
    val = stack.pop()  # type: ignore[union-attr]
    store[stmt.var] = val

    irr_baseline = _bits_for_value(val)
    metrics.irreversible_baseline_bits += irr_baseline

    step = StepMetric(
        stmt_text=f"pop {stmt.var} {stmt.stack}",
        kind="pop",
        bits_erased=0.0,
        bits_created=0.0,
    )
    metrics.step_details.append(step)
    metrics.total_steps += 1
    metrics.reversible_steps += 1


# -- helpers -------------------------------------------------------------------


def _truthy(value: int) -> bool:
    return value != 0


def _render_place(place: str | Place) -> str:
    if isinstance(place, str):
        return place
    if isinstance(place, Var):
        return place.name
    if isinstance(place, ArrayRef):
        return f"{place.name}[...]"
    return repr(place)


def _render_assign(stmt: Assign) -> str:
    return f"{_render_place(stmt.target)} {stmt.op.value} ..."


def _render_swap(stmt: Swap) -> str:
    return f"{_render_place(stmt.left)} <=> {_render_place(stmt.right)}"


def format_metrics(metrics: LandauerMetrics) -> str:
    """Format *metrics* as a human-readable multi-line string."""
    lines = [
        "Landauer Entropy Analysis",
        "=" * 40,
        f"Total steps executed:     {metrics.total_steps}",
        f"Reversible steps:         {metrics.reversible_steps}",
        f"Information erased:       {metrics.info_erased_bits:.4f} bits",
        f"Information created:      {metrics.info_created_bits:.4f} bits",
        f"Net entropy change:       {metrics.net_entropy:.4f} bits",
        "",
        "Irreversible baseline",
        "-" * 40,
        f"Bits that WOULD be erased: {metrics.irreversible_baseline_bits:.4f}",
        f"Savings from reversibility: {metrics.irreversible_baseline_bits:.4f} bits",
    ]
    if metrics.step_details:
        lines.append("")
        lines.append("Step details:")
        for i, step in enumerate(metrics.step_details, 1):
            lines.append(
                f"  {i:3d}. [{step.kind:12s}] erased={step.bits_erased:.2f} "
                f"created={step.bits_created:.2f}  {step.stmt_text}"
            )
    return "\n".join(lines)


def metrics_to_dict(metrics: LandauerMetrics) -> dict:
    """Serialize *metrics* to a JSON-friendly dictionary."""
    return {
        "total_steps": metrics.total_steps,
        "reversible_steps": metrics.reversible_steps,
        "info_erased_bits": metrics.info_erased_bits,
        "info_created_bits": metrics.info_created_bits,
        "net_entropy": metrics.net_entropy,
        "irreversible_baseline_bits": metrics.irreversible_baseline_bits,
        "step_details": [
            {
                "stmt_text": s.stmt_text,
                "kind": s.kind,
                "bits_erased": s.bits_erased,
                "bits_created": s.bits_created,
            }
            for s in metrics.step_details
        ],
    }
