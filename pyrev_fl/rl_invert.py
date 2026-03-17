from __future__ import annotations

from pyrev_fl.ast import UpdateOp
from pyrev_fl.rl_ast import Assign, Assignment, Block, Exit, FiFrom, FromEntry, FromLabel, Goto, IfGoto, Program, RFromLabel, RGoto, Swap


def invert_program(program: Program) -> Program:
    return Program(
        inputs=list(program.outputs),
        outputs=list(program.inputs),
        temps=list(program.temps),
        blocks=[invert_block(block) for block in program.blocks],
    )


def invert_block(block: Block) -> Block:
    return Block(
        label=block.label,
        from_=invert_jump(block.jump),
        assigns=[invert_assignment(assign) for assign in reversed(block.assigns)],
        jump=invert_from(block.from_),
    )


def invert_assignment(assign: Assignment) -> Assignment:
    if isinstance(assign, Assign):
        inverse = {
            UpdateOp.ADD: UpdateOp.SUB,
            UpdateOp.SUB: UpdateOp.ADD,
            UpdateOp.XOR: UpdateOp.XOR,
        }[assign.op]
        return Assign(assign.target, inverse, assign.expr)
    if isinstance(assign, Swap):
        return assign
    raise TypeError(f"unknown assignment: {assign!r}")


def invert_from(from_):
    if isinstance(from_, FromEntry):
        return Exit()
    if isinstance(from_, FromLabel):
        return Goto(from_.label)
    if isinstance(from_, RFromLabel):
        return RGoto(from_.label)
    if isinstance(from_, FiFrom):
        return IfGoto(from_.expr, from_.true_label, from_.false_label)
    raise TypeError(f"unknown from construct: {from_!r}")


def invert_jump(jump):
    if isinstance(jump, Exit):
        return FromEntry()
    if isinstance(jump, Goto):
        return FromLabel(jump.label)
    if isinstance(jump, RGoto):
        return RFromLabel(jump.label)
    if isinstance(jump, IfGoto):
        return FiFrom(jump.expr, jump.true_label, jump.false_label)
    raise TypeError(f"unknown jump construct: {jump!r}")

