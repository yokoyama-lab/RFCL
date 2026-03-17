from __future__ import annotations

from pyrev_fl.ast import ArrayRef, Binary, Const, Expr, Place, Var
from pyrev_fl.rl_ast import Assign, Block, Exit, FiFrom, FromEntry, FromLabel, Goto, IfGoto, Program, RFromLabel, RGoto, Swap


def render_program(program: Program) -> str:
    lines = [
        f"{_render_names(program.inputs)} {_render_names(program.outputs)} {_render_names(program.temps)}"
    ]
    for block in program.blocks:
        lines.extend(_render_block(block))
    return "\n".join(lines) + "\n"


def _render_names(names: list[str]) -> str:
    return f"({' '.join(names)})"


def _render_block(block: Block) -> list[str]:
    lines = [f"{block.label}: {_render_from(block.from_)}"]
    lines.extend(f"{_render_assignment(assign)}" for assign in block.assigns)
    lines.append(_render_jump(block.jump))
    return lines


def _render_from(from_) -> str:
    if isinstance(from_, FromEntry):
        return "entry;"
    if isinstance(from_, FromLabel):
        return f"from {from_.label};"
    if isinstance(from_, RFromLabel):
        return f"rfrom {from_.label};"
    if isinstance(from_, FiFrom):
        return f"fi {_render_expr(from_.expr)} from {from_.true_label} else {from_.false_label};"
    raise TypeError(f"unknown from construct: {from_!r}")


def _render_assignment(assign) -> str:
    if isinstance(assign, Assign):
        return f"{_render_place(assign.target)} {assign.op.value} {_render_expr(assign.expr)};"
    if isinstance(assign, Swap):
        return f"{_render_place(assign.left)} <=> {_render_place(assign.right)};"
    raise TypeError(f"unknown assignment: {assign!r}")


def _render_jump(jump) -> str:
    if isinstance(jump, Goto):
        return f"goto {jump.label};"
    if isinstance(jump, RGoto):
        return f"rgoto {jump.label};"
    if isinstance(jump, IfGoto):
        return f"if {_render_expr(jump.expr)} goto {jump.true_label} else {jump.false_label};"
    if isinstance(jump, Exit):
        return "exit;"
    raise TypeError(f"unknown jump construct: {jump!r}")


def _render_expr(expr: Expr) -> str:
    if isinstance(expr, Const):
        return str(expr.value)
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, ArrayRef):
        return f"{expr.name}[{_render_expr(expr.index)}]"
    if isinstance(expr, Binary):
        return f"({expr.op.value} {_render_expr(expr.left)} {_render_expr(expr.right)})"
    raise TypeError(f"unknown expression: {expr!r}")


def _render_place(place: str | Place) -> str:
    if isinstance(place, str):
        return place
    return _render_expr(place)
