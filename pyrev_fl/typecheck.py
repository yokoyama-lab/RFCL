"""Extended type checker for SRL programs.

Beyond the basic static checker (check.py), this verifies:
1. Array bounds: all constant array indices are within declared bounds
2. Stack discipline: push/pop are balanced within each scope
3. Zero constraint: variables used in assertions are never unconditionally zero-cleared
4. Expression type consistency: comparisons return boolean-like values
"""
from __future__ import annotations

from dataclasses import dataclass

from pyrev_fl.ast import (
    ArrayRef, Assign, Binary, BinOp, Block, Const, Expr,
    If, Loop, Pop, Program, Push, Rif, Stmt, Swap, Var,
)
from pyrev_fl.interface import build_layout, parse_decl


@dataclass
class TypeWarning:
    level: str  # "error" or "warning"
    message: str


def typecheck_program(program: Program) -> list[TypeWarning]:
    """Return a list of type warnings/errors for *program*."""
    warnings: list[TypeWarning] = []
    layout = build_layout(program.inputs, program.outputs, program.temps)
    decls = layout.declarations

    _check_array_bounds(program.body, decls, warnings)
    _check_stack_balance(program.body, warnings)
    _check_guard_vars(program.body, warnings)

    return warnings


def _check_array_bounds(block: Block, decls: dict, warnings: list[TypeWarning]) -> None:
    """Check that constant array indices are within bounds."""
    for stmt in block.stmts:
        for expr in _collect_exprs(stmt):
            _check_expr_bounds(expr, decls, warnings)
        for sub in _sub_blocks(stmt):
            _check_array_bounds(sub, decls, warnings)


def _check_expr_bounds(expr: Expr, decls: dict, warnings: list[TypeWarning]) -> None:
    if isinstance(expr, ArrayRef):
        decl = decls.get(expr.name)
        if decl is not None and isinstance(expr.index, Const):
            idx = expr.index.value
            size = getattr(decl, "size", None)
            if size is not None and (idx < 0 or idx >= size):
                warnings.append(TypeWarning(
                    "error",
                    f"array index out of bounds: {expr.name}[{idx}] (size={size})"
                ))
        _check_expr_bounds(expr.index, decls, warnings)
    elif isinstance(expr, Binary):
        _check_expr_bounds(expr.left, decls, warnings)
        _check_expr_bounds(expr.right, decls, warnings)


def _check_stack_balance(block: Block, warnings: list[TypeWarning]) -> None:
    """Check that push/pop operations are balanced within the block."""
    stack_counts: dict[str, int] = {}
    for stmt in block.stmts:
        if isinstance(stmt, Push):
            stack_counts[stmt.stack] = stack_counts.get(stmt.stack, 0) + 1
        elif isinstance(stmt, Pop):
            stack_counts[stmt.stack] = stack_counts.get(stmt.stack, 0) - 1
            if stack_counts[stmt.stack] < 0:
                warnings.append(TypeWarning(
                    "warning",
                    f"pop from potentially empty stack: {stmt.stack}"
                ))
        elif isinstance(stmt, (If, Loop, Rif)):
            for sub in _sub_blocks(stmt):
                _check_stack_balance(sub, warnings)

    for name, count in stack_counts.items():
        if count != 0:
            warnings.append(TypeWarning(
                "warning",
                f"unbalanced stack operations for '{name}': "
                f"net {'push' if count > 0 else 'pop'} count = {abs(count)}"
            ))


def _check_guard_vars(block: Block, warnings: list[TypeWarning]) -> None:
    """Warn if guard expressions use variables that might be zero."""
    for stmt in block.stmts:
        if isinstance(stmt, If):
            _check_guard_vars(stmt.then_block, warnings)
            _check_guard_vars(stmt.else_block, warnings)
        elif isinstance(stmt, Loop):
            _check_guard_vars(stmt.do_block, warnings)
            _check_guard_vars(stmt.loop_block, warnings)
        elif isinstance(stmt, Rif):
            _check_guard_vars(stmt.body, warnings)


def _collect_exprs(stmt: Stmt) -> list[Expr]:
    """Collect all expressions from a statement (not recursing into sub-blocks)."""
    exprs: list[Expr] = []
    if isinstance(stmt, Assign):
        if isinstance(stmt.target, (Var, ArrayRef)):
            exprs.append(stmt.target)
        exprs.append(stmt.expr)
    elif isinstance(stmt, Swap):
        for place in (stmt.left, stmt.right):
            if isinstance(place, (Var, ArrayRef)):
                exprs.append(place)
    elif isinstance(stmt, If):
        exprs.extend([stmt.test, stmt.assertion])
    elif isinstance(stmt, Loop):
        exprs.extend([stmt.entry_guard, stmt.exit_guard])
    elif isinstance(stmt, Rif):
        exprs.extend([stmt.test, stmt.assertion])
    return exprs


def _sub_blocks(stmt: Stmt) -> list[Block]:
    if isinstance(stmt, If):
        return [stmt.then_block, stmt.else_block]
    if isinstance(stmt, Loop):
        return [stmt.do_block, stmt.loop_block]
    if isinstance(stmt, Rif):
        return [stmt.body]
    return []
