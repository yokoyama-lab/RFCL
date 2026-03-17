"""RL → PLA (PISA-style reversible assembly) renderer.

PLA instruction set:
    ENTRY                       program entry point
    FROM label                  unconditional from
    RFROM label                 reverse from (direction change)
    FI expr FROM label ELSE label   conditional from
    ADD target, expr            target += expr
    SUB target, expr            target -= expr
    XOR target, expr            target ^= expr
    EXCH left, right            swap left and right
    BRA label                   unconditional branch (goto)
    RBRA label                  reverse branch (rgoto, direction change)
    BNEZ expr, label, label     conditional branch (if expr goto true else false)
    HALT                        program exit
"""
from __future__ import annotations

from pyrev_fl.rl_ast import (
    Assign, Block, Exit, FiFrom, FromEntry, FromLabel,
    Goto, IfGoto, Program, RFromLabel, RGoto, Swap,
)
from pyrev_fl.rl_pretty import _render_expr


def render_pla(program: Program) -> str:
    """Return PLA assembly text for *program*."""
    lines: list[str] = [
        f"# PLA (PISA-style reversible assembly)",
        f"({' '.join(program.inputs)}) ({' '.join(program.outputs)}) ({' '.join(program.temps)})",
        "",
    ]
    for block in program.blocks:
        lines.extend(_render_block(block))
        lines.append("")
    return "\n".join(lines)


def _render_block(block: Block) -> list[str]:
    lines = [f"{block.label}:"]
    lines.append(f"    {_render_from(block.from_)}")
    for assign in block.assigns:
        lines.append(f"    {_render_assign(assign)}")
    lines.append(f"    {_render_jump(block.jump)}")
    return lines


def _render_from(from_) -> str:
    if isinstance(from_, FromEntry):
        return "ENTRY"
    if isinstance(from_, FromLabel):
        return f"FROM {from_.label}"
    if isinstance(from_, RFromLabel):
        return f"RFROM {from_.label}"
    if isinstance(from_, FiFrom):
        return f"FI {_render_expr(from_.expr)} FROM {from_.true_label} ELSE {from_.false_label}"
    raise TypeError(f"unknown from: {from_!r}")


def _render_assign(assign) -> str:
    if isinstance(assign, Assign):
        op = {assign.op.ADD: "ADD", assign.op.SUB: "SUB", assign.op.XOR: "XOR"}[assign.op]
        return f"{op} {_render_place(assign.target)}, {_render_expr(assign.expr)}"
    if isinstance(assign, Swap):
        return f"EXCH {_render_place(assign.left)}, {_render_place(assign.right)}"
    raise TypeError(f"unknown assign: {assign!r}")


def _render_place(place) -> str:
    if isinstance(place, str):
        return place
    return _render_expr(place)


def _render_jump(jump) -> str:
    if isinstance(jump, Goto):
        return f"BRA {jump.label}"
    if isinstance(jump, RGoto):
        return f"RBRA {jump.label}"
    if isinstance(jump, IfGoto):
        return f"BNEZ {_render_expr(jump.expr)}, {jump.true_label}, {jump.false_label}"
    if isinstance(jump, Exit):
        return "HALT"
    raise TypeError(f"unknown jump: {jump!r}")
