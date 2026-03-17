from __future__ import annotations

from pyrev_fl.dump_ast import dump_expr
from pyrev_fl.rl_ast import Assign, Block, Exit, FiFrom, FromEntry, FromLabel, Goto, IfGoto, Program, RFromLabel, RGoto, Swap


def dump_program(program: Program) -> dict[str, object]:
    return {
        "kind": "rl_program",
        "inputs": list(program.inputs),
        "outputs": list(program.outputs),
        "temps": list(program.temps),
        "blocks": [dump_block(block) for block in program.blocks],
    }


def dump_block(block: Block) -> dict[str, object]:
    return {
        "kind": "block",
        "label": block.label,
        "from": dump_from(block.from_),
        "assigns": [dump_assign(assign) for assign in block.assigns],
        "jump": dump_jump(block.jump),
    }


def dump_from(from_) -> dict[str, object]:
    if isinstance(from_, FromEntry):
        return {"kind": "entry"}
    if isinstance(from_, FromLabel):
        return {"kind": "from", "label": from_.label}
    if isinstance(from_, RFromLabel):
        return {"kind": "rfrom", "label": from_.label}
    if isinstance(from_, FiFrom):
        return {
            "kind": "fi_from",
            "expr": dump_expr(from_.expr),
            "true_label": from_.true_label,
            "false_label": from_.false_label,
        }
    raise TypeError(f"unknown from construct: {from_!r}")


def dump_assign(assign) -> dict[str, object]:
    if isinstance(assign, Assign):
        return {
            "kind": "assign",
            "target": dump_expr(assign.target) if not isinstance(assign.target, str) else {"kind": "var", "name": assign.target},
            "op": assign.op.value,
            "expr": dump_expr(assign.expr),
        }
    if isinstance(assign, Swap):
        return {
            "kind": "swap",
            "left": dump_expr(assign.left) if not isinstance(assign.left, str) else {"kind": "var", "name": assign.left},
            "right": dump_expr(assign.right) if not isinstance(assign.right, str) else {"kind": "var", "name": assign.right},
        }
    raise TypeError(f"unknown assignment: {assign!r}")


def dump_jump(jump) -> dict[str, object]:
    if isinstance(jump, Goto):
        return {"kind": "goto", "label": jump.label}
    if isinstance(jump, RGoto):
        return {"kind": "rgoto", "label": jump.label}
    if isinstance(jump, IfGoto):
        return {
            "kind": "if_goto",
            "expr": dump_expr(jump.expr),
            "true_label": jump.true_label,
            "false_label": jump.false_label,
        }
    if isinstance(jump, Exit):
        return {"kind": "exit"}
    raise TypeError(f"unknown jump construct: {jump!r}")
