"""Reversible program slicer for SRL.

Given an SRL program and a set of output variables of interest, computes the
minimal sub-program (slice) that produces those outputs.  Uses backward
dependency analysis: starting from the variables of interest, trace which
statements contribute to their values.
"""
from __future__ import annotations

from pyrev_fl.ast import (
    ArrayRef, Assign, Binary, Block, Const, Expr, If, Loop, Pop, Program, Push, Rif, Stmt, Swap, Var,
)


def slice_program(program: Program, targets: set[str]) -> Program:
    """Return a sliced program that only computes *targets*."""
    needed = set(targets)
    sliced_body = _slice_block(program.body, needed)
    return Program(program.inputs, program.outputs, program.temps, sliced_body)


def _slice_block(block: Block, needed: set[str]) -> Block:
    """Backward-slice a block: keep only statements that affect *needed* vars."""
    kept: list[Stmt] = []
    # Process statements in reverse (backward dependency)
    for stmt in reversed(block.stmts):
        defs, uses = _stmt_def_use(stmt)
        if defs & needed:
            needed -= defs
            needed |= uses
            kept.append(stmt)
        elif isinstance(stmt, (If, Loop, Rif)):
            # Control flow: always keep if any sub-statement is needed
            sub_needed = set(needed)
            sliced = _slice_control(stmt, sub_needed)
            if sliced is not None:
                needed |= _control_guard_vars(stmt)
                kept.append(sliced)
    kept.reverse()
    return Block(kept)


def _slice_control(stmt: Stmt, needed: set[str]) -> Stmt | None:
    """Slice a control-flow statement. Return None if entirely dead."""
    if isinstance(stmt, If):
        then_s = _slice_block(stmt.then_block, set(needed))
        else_s = _slice_block(stmt.else_block, set(needed))
        if not then_s.stmts and not else_s.stmts:
            return None
        return If(stmt.test, then_s, else_s, stmt.assertion)
    if isinstance(stmt, Loop):
        do_s = _slice_block(stmt.do_block, set(needed))
        loop_s = _slice_block(stmt.loop_block, set(needed))
        if not do_s.stmts and not loop_s.stmts:
            return None
        return Loop(stmt.entry_guard, do_s, loop_s, stmt.exit_guard)
    if isinstance(stmt, Rif):
        body_s = _slice_block(stmt.body, set(needed))
        if not body_s.stmts:
            return None
        return Rif(stmt.test, body_s, stmt.assertion)
    return None


def _stmt_def_use(stmt: Stmt) -> tuple[set[str], set[str]]:
    """Return (defined_vars, used_vars) for a statement."""
    if isinstance(stmt, Assign):
        target_name = _place_name(stmt.target)
        # Reversible assignments (x += e) use x too: x_new = x_old + e
        return {target_name}, _expr_vars(stmt.expr) | {target_name}
    if isinstance(stmt, Swap):
        left = _place_name(stmt.left)
        right = _place_name(stmt.right)
        return {left, right}, {left, right}
    if isinstance(stmt, Push):
        return {stmt.var, stmt.stack}, {stmt.var}
    if isinstance(stmt, Pop):
        return {stmt.var}, {stmt.stack}
    if isinstance(stmt, (If, Loop, Rif)):
        # Approximate: collect all defs and uses from sub-blocks
        defs: set[str] = set()
        uses: set[str] = set()
        for sub in _sub_blocks(stmt):
            for s in sub.stmts:
                d, u = _stmt_def_use(s)
                defs |= d
                uses |= u
        return defs, uses
    return set(), set()


def _control_guard_vars(stmt: Stmt) -> set[str]:
    """Variables used in control-flow guards."""
    if isinstance(stmt, If):
        return _expr_vars(stmt.test) | _expr_vars(stmt.assertion)
    if isinstance(stmt, Loop):
        return _expr_vars(stmt.entry_guard) | _expr_vars(stmt.exit_guard)
    if isinstance(stmt, Rif):
        return _expr_vars(stmt.test) | _expr_vars(stmt.assertion)
    return set()


def _sub_blocks(stmt: Stmt) -> list[Block]:
    if isinstance(stmt, If):
        return [stmt.then_block, stmt.else_block]
    if isinstance(stmt, Loop):
        return [stmt.do_block, stmt.loop_block]
    if isinstance(stmt, Rif):
        return [stmt.body]
    return []


def _expr_vars(expr: Expr) -> set[str]:
    if isinstance(expr, Const):
        return set()
    if isinstance(expr, Var):
        return {expr.name}
    if isinstance(expr, ArrayRef):
        return {expr.name} | _expr_vars(expr.index)
    if isinstance(expr, Binary):
        return _expr_vars(expr.left) | _expr_vars(expr.right)
    return set()


def _place_name(place) -> str:
    if isinstance(place, str):
        return place
    if isinstance(place, Var):
        return place.name
    if isinstance(place, ArrayRef):
        return place.name
    return str(place)
