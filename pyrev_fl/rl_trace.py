from __future__ import annotations

from pyrev_fl.interface import build_layout
from pyrev_fl.rl_ast import Direction, Program
from pyrev_fl.rl_check import check_program
from pyrev_fl.rl_interpreter import EvalError, _entry_label, _eval_assignment, _eval_assignment_inverse, _eval_from, _eval_jump, _initialize_store, _validate_final_store


def trace_program(program: Program, inputs: list[int]) -> list[dict[str, object]]:
    layout = build_layout(program.inputs, program.outputs, program.temps)
    if len(inputs) != len(layout.inputs):
        raise EvalError(f"arity mismatch: expected {len(layout.inputs)}, got {len(inputs)}")
    check_program(program)
    block_map = {block.label: block for block in program.blocks}
    store = _initialize_store(program, inputs, layout)
    previous = "start"
    current = _entry_label(program)
    direction = Direction.FORWARD
    events: list[dict[str, object]] = []

    for _ in range(100000):
        if current == "halt" and direction is Direction.FORWARD:
            _validate_final_store(store, layout)
            return events

        block = block_map[current]
        before = dict(store)
        if direction is Direction.FORWARD:
            expected_previous, _ = _eval_from(block.from_, store, layout)
            if expected_previous != previous:
                raise EvalError(f"from mismatch at {block.label}: expected {expected_previous}, got {previous}")
            for assign in block.assigns:
                _eval_assignment(assign, store, layout)
            next_label, next_dir = _eval_jump(block.jump, store, layout)
        else:
            expected_previous, _ = _eval_jump(block.jump, store, layout)
            if expected_previous != previous:
                raise EvalError(f"jump mismatch at {block.label}: expected {expected_previous}, got {previous}")
            for assign in reversed(block.assigns):
                _eval_assignment_inverse(assign, store, layout)
            next_label, next_dir = _eval_from(block.from_, store, layout)

        events.append(
            {
                "direction": direction.value,
                "prev_label": previous,
                "current_label": current,
                "next_label": next_label,
                "next_direction": next_dir.value,
                "store_diff": _store_diff(before, store),
            }
        )
        previous, current, direction = current, next_label, next_dir

    raise EvalError("step limit exceeded")


def _store_diff(before: dict[str, int], after: dict[str, int]) -> dict[str, dict[str, int]]:
    diff: dict[str, dict[str, int]] = {}
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            diff[key] = {"before": before.get(key, 0), "after": after.get(key, 0)}
    return diff
