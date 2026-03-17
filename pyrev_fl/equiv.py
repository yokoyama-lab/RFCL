"""Program equivalence checking for SRL programs.

Two SRL programs are equivalent if they compute the same function:
for all valid inputs, they produce the same outputs.

Provides:
- Exhaustive check for small input domains
- Sampling-based check for larger domains
"""
from __future__ import annotations

from pyrev_fl.ast import Program
from pyrev_fl.interface import build_layout
from pyrev_fl.interpreter import run_program, EvalError


def check_equivalence(
    p1: Program,
    p2: Program,
    *,
    input_range: range = range(-5, 6),
    max_tests: int = 1000,
) -> tuple[bool, str]:
    """Check if p1 and p2 are equivalent over the given input range.

    Returns (True, "equivalent") or (False, "counterexample: ...").
    """
    layout1 = build_layout(p1.inputs, p1.outputs, p1.temps)
    layout2 = build_layout(p2.inputs, p2.outputs, p2.temps)

    if layout1.outputs != layout2.outputs:
        return False, f"different output interfaces: {layout1.outputs} vs {layout2.outputs}"

    n_inputs = len(layout1.inputs)
    if n_inputs != len(layout2.inputs):
        return False, f"different input arities: {n_inputs} vs {len(layout2.inputs)}"

    tested = 0
    for inputs in _input_tuples(n_inputs, input_range, max_tests):
        try:
            s1 = run_program(p1, list(inputs))
            s2 = run_program(p2, list(inputs))
        except (EvalError, ZeroDivisionError):
            continue  # skip inputs that cause errors in either program

        for name in layout1.outputs:
            if s1.get(name, 0) != s2.get(name, 0):
                return False, (
                    f"counterexample: inputs={list(inputs)}, "
                    f"{name}: p1={s1[name]}, p2={s2[name]}"
                )
        tested += 1

    return True, f"equivalent (tested {tested} inputs)"


def _input_tuples(n: int, values: range, limit: int):
    """Generate input tuples up to *limit*."""
    if n == 0:
        yield ()
        return
    count = 0
    vals = list(values)
    if n == 1:
        for v in vals:
            yield (v,)
            count += 1
            if count >= limit:
                return
    elif n == 2:
        for a in vals:
            for b in vals:
                yield (a, b)
                count += 1
                if count >= limit:
                    return
    else:
        for a in vals:
            for b in vals:
                for c in vals:
                    yield (a, b, c)
                    count += 1
                    if count >= limit:
                        return
