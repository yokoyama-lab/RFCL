from __future__ import annotations

from pyrev_fl.ast import Assign, Block, If, Loop, Pop, Program, Push, Rif, Stmt, Swap, UpdateOp


def invert_program(program: Program) -> Program:
    return Program(
        inputs=list(program.outputs),
        outputs=list(program.inputs),
        temps=list(program.temps),
        body=invert_block(program.body),
    )


def invert_block(block: Block) -> Block:
    return Block([_invert_stmt(stmt) for stmt in reversed(block.stmts)])


def _invert_stmt(stmt: Stmt) -> Stmt:
    if isinstance(stmt, Assign):
        return Assign(stmt.target, _invert_update(stmt.op), stmt.expr)
    if isinstance(stmt, Swap):
        return stmt
    if isinstance(stmt, If):
        return If(
            test=stmt.assertion,
            then_block=invert_block(stmt.then_block),
            else_block=invert_block(stmt.else_block),
            assertion=stmt.test,
        )
    if isinstance(stmt, Loop):
        return Loop(
            entry_guard=stmt.exit_guard,
            do_block=invert_block(stmt.do_block),
            loop_block=invert_block(stmt.loop_block),
            exit_guard=stmt.entry_guard,
        )
    if isinstance(stmt, Rif):
        return Rif(
            test=stmt.assertion,
            body=invert_block(stmt.body),
            assertion=stmt.test,
        )
    if isinstance(stmt, Push):
        return Pop(stmt.var, stmt.stack)
    if isinstance(stmt, Pop):
        return Push(stmt.var, stmt.stack)
    raise TypeError(f"unknown statement: {stmt!r}")


def _invert_update(op: UpdateOp) -> UpdateOp:
    if op is UpdateOp.ADD:
        return UpdateOp.SUB
    if op is UpdateOp.SUB:
        return UpdateOp.ADD
    return UpdateOp.XOR
