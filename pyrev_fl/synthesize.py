"""Enumerative program synthesizer for SRL.

Given input/output examples, enumerates candidate SRL programs of
increasing size and tests each against every example.  Returns the
first program that satisfies all examples, or ``None`` on timeout.

The search space is restricted to straight-line programs (no loops or
conditionals) built from:
    x += y, x -= y, x ^= y   (for distinct variables x, y)
    x += c, x -= c, x ^= c   (for small constants c in {1, 2})
    x <=> y                   (swap, for distinct variables)

Symmetry breaking
-----------------
* Within a sequence of statements, we skip orderings that are
  canonically "later" than an equivalent permutation.  Concretely, for
  independent statements (touching disjoint variables) we require
  lexicographic order of their string representations so that we
  enumerate only one of the equivalent orderings.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

from pyrev_fl.ast import (
    Assign,
    Block,
    Const,
    Program,
    Stmt,
    Swap,
    UpdateOp,
    Var,
)
from pyrev_fl.interpreter import EvalError, run_program
from pyrev_fl.interface import build_layout
from pyrev_fl.pretty import render_program


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass
class SynthExample:
    """One input/output example for the synthesizer."""
    inputs: list[int]
    expected_outputs: dict[str, int]


@dataclass
class SynthResult:
    """Result of a synthesis attempt."""
    program: Program | None
    programs_tested: int
    time_seconds: float


# ---------------------------------------------------------------------------
# Statement enumeration
# ---------------------------------------------------------------------------

_SMALL_CONSTS = (1, 2)
_UPDATE_OPS = (UpdateOp.ADD, UpdateOp.SUB, UpdateOp.XOR)


def _enumerate_single_stmts(all_vars: list[str]) -> list[Stmt]:
    """Generate all single statements over *all_vars*."""
    stmts: list[Stmt] = []

    for target in all_vars:
        # x op= y  for each other variable y
        others = [v for v in all_vars if v != target]
        for other in others:
            for op in _UPDATE_OPS:
                stmts.append(Assign(target, op, Var(other)))
        # x op= c  for small constants
        for c in _SMALL_CONSTS:
            for op in _UPDATE_OPS:
                stmts.append(Assign(target, op, Const(c)))

    # x <=> y  for each unordered pair
    for i, a in enumerate(all_vars):
        for b in all_vars[i + 1:]:
            stmts.append(Swap(a, b))

    return stmts


def _stmt_key(stmt: Stmt) -> str:
    """Deterministic string key for symmetry breaking."""
    if isinstance(stmt, Assign):
        target = stmt.target if isinstance(stmt.target, str) else repr(stmt.target)
        return f"A|{target}|{stmt.op.value}|{_expr_key(stmt.expr)}"
    if isinstance(stmt, Swap):
        left = stmt.left if isinstance(stmt.left, str) else repr(stmt.left)
        right = stmt.right if isinstance(stmt.right, str) else repr(stmt.right)
        return f"S|{left}|{right}"
    return repr(stmt)


def _expr_key(expr) -> str:
    from pyrev_fl.ast import Const as C, Var as V
    if isinstance(expr, C):
        return f"c{expr.value}"
    if isinstance(expr, V):
        return f"v{expr.name}"
    return repr(expr)


def _stmt_vars(stmt: Stmt) -> set[str]:
    """Return the set of variable names touched by *stmt*."""
    if isinstance(stmt, Assign):
        target_name = stmt.target if isinstance(stmt.target, str) else getattr(stmt.target, 'name', '')
        rhs_vars = _expr_vars(stmt.expr)
        return {target_name} | rhs_vars
    if isinstance(stmt, Swap):
        left = stmt.left if isinstance(stmt.left, str) else getattr(stmt.left, 'name', '')
        right = stmt.right if isinstance(stmt.right, str) else getattr(stmt.right, 'name', '')
        return {left, right}
    return set()


def _expr_vars(expr) -> set[str]:
    from pyrev_fl.ast import Const as C, Var as V, Binary as B
    if isinstance(expr, C):
        return set()
    if isinstance(expr, V):
        return {expr.name}
    if isinstance(expr, B):
        return _expr_vars(expr.left) | _expr_vars(expr.right)
    return set()


def _is_canonical_order(stmts: tuple[Stmt, ...]) -> bool:
    """Check whether independent adjacent statements are in canonical order.

    Two adjacent statements are *independent* when they touch disjoint
    variable sets.  For independent pairs we require that the first has
    a lexicographically smaller key than the second; this eliminates
    redundant permutations.
    """
    for i in range(len(stmts) - 1):
        if _stmt_vars(stmts[i]).isdisjoint(_stmt_vars(stmts[i + 1])):
            if _stmt_key(stmts[i]) > _stmt_key(stmts[i + 1]):
                return False
    return True


# ---------------------------------------------------------------------------
# Candidate testing
# ---------------------------------------------------------------------------

def _test_candidate(
    program: Program,
    examples: list[SynthExample],
) -> bool:
    """Return True if *program* produces correct outputs for every example."""
    layout = build_layout(program.inputs, program.outputs, program.temps)
    for ex in examples:
        try:
            store = run_program(program, ex.inputs)
        except (EvalError, ValueError):
            return False
        for name, expected in ex.expected_outputs.items():
            if store.get(name) != expected:
                return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize(
    input_names: list[str],
    output_names: list[str],
    examples: list[SynthExample],
    *,
    max_stmts: int = 3,
    max_depth: int = 1,
    timeout: float = 5.0,
) -> SynthResult:
    """Synthesize an SRL program from input/output examples.

    Enumerates programs of increasing size (1 statement, 2 statements, ...)
    and tests each against all examples.  Returns the first program that
    satisfies every example, or ``None`` if the timeout is reached.

    Parameters
    ----------
    input_names:
        Variable names for the program inputs.
    output_names:
        Variable names for the program outputs.
    examples:
        List of :class:`SynthExample` specifying desired behaviour.
    max_stmts:
        Maximum number of statements to try (default 3).
    max_depth:
        Reserved for future use (structured programs); currently ignored.
    timeout:
        Wall-clock seconds after which to give up.
    """
    start = time.monotonic()
    programs_tested = 0

    # Determine the set of all variables the synthesised program may use.
    # Variables that appear only in inputs (not outputs) must be zeroed,
    # so we need temps for certain patterns.  We add a small set of temp
    # variables automatically when inputs != outputs.
    all_names = list(dict.fromkeys(input_names + output_names))

    # Decide which variables are "temp" (declared in both inputs and
    # outputs is not temp; only in inputs is not temp either -- it is
    # zeroed automatically by the SRL interpreter).  We may need
    # additional temp variables for programs that require scratch space.
    input_set = set(input_names)
    output_set = set(output_names)
    temp_names: list[str] = []

    # Add synthetic temps if the variable space is very small.
    if len(all_names) < 3:
        for tname in ("_t0", "_t1"):
            if tname not in all_names:
                temp_names.append(tname)
                all_names.append(tname)

    single_stmts = _enumerate_single_stmts(all_names)

    # Also try the empty program (identity).
    empty_prog = Program(
        inputs=list(input_names),
        outputs=list(output_names),
        temps=list(temp_names),
        body=Block([]),
    )
    programs_tested += 1
    if _test_candidate(empty_prog, examples):
        elapsed = time.monotonic() - start
        return SynthResult(empty_prog, programs_tested, elapsed)

    for size in range(1, max_stmts + 1):
        if time.monotonic() - start > timeout:
            break
        for combo in itertools.product(single_stmts, repeat=size):
            if time.monotonic() - start > timeout:
                break

            # Symmetry breaking: skip non-canonical orderings.
            if not _is_canonical_order(combo):
                continue

            # Skip trivially redundant sequences: same statement twice
            # in a row is a no-op for XOR and cancels for +/-.
            if _has_adjacent_cancel(combo):
                continue

            prog = Program(
                inputs=list(input_names),
                outputs=list(output_names),
                temps=list(temp_names),
                body=Block(list(combo)),
            )

            programs_tested += 1
            if _test_candidate(prog, examples):
                elapsed = time.monotonic() - start
                return SynthResult(prog, programs_tested, elapsed)

    elapsed = time.monotonic() - start
    return SynthResult(None, programs_tested, elapsed)


def _has_adjacent_cancel(stmts: tuple[Stmt, ...]) -> bool:
    """Detect obviously redundant adjacent statement pairs.

    * ``x += y; x -= y`` (and vice-versa) cancel out.
    * ``x ^= y; x ^= y`` cancels out.
    * ``x <=> y; x <=> y`` cancels out (same swap twice).
    """
    for i in range(len(stmts) - 1):
        a, b = stmts[i], stmts[i + 1]
        # Duplicate swap
        if isinstance(a, Swap) and isinstance(b, Swap):
            if a == b:
                return True
        # Assign cancellation
        if isinstance(a, Assign) and isinstance(b, Assign):
            if a.target == b.target and a.expr == b.expr:
                # XOR self-inverse
                if a.op is UpdateOp.XOR and b.op is UpdateOp.XOR:
                    return True
                # ADD then SUB (or vice-versa)
                if {a.op, b.op} == {UpdateOp.ADD, UpdateOp.SUB}:
                    return True
    return False
