"""Partial evaluator for SRL programs.

Given an SRL program and a partial store (known values for some inputs),
produces a specialized (residual) program that only computes over the
remaining unknown inputs.

This implements online partial evaluation for reversible programs,
inspired by Hatcliff's tutorial referenced in Moriyama 2009.
When combined with the self-interpreter (self_interp.srl), this
enables the first Futamura projection for reversible languages.
"""
from __future__ import annotations

from pyrev_fl.ast import (
    Assign, BinOp, Binary, Block, Const, Expr, If, Loop,
    Pop, Program, Push, Rif, Stmt, Swap, UpdateOp, Var,
)


def partial_eval(program: Program, known: dict[str, int]) -> Program:
    """Specialize *program* given known input values.

    *known* maps variable names to their known values.
    Returns a residual program over the remaining unknown inputs.
    """
    env = dict(known)
    residual_body = _pe_block(program.body, env)
    # Remaining inputs are those not in known
    remaining_inputs = [n for n in program.inputs if n not in known]
    return Program(remaining_inputs, program.outputs, program.temps, residual_body)


def _pe_block(block: Block, env: dict[str, int]) -> Block:
    stmts: list[Stmt] = []
    for stmt in block.stmts:
        result = _pe_stmt(stmt, env)
        if result is not None:
            stmts.append(result)
    return Block(stmts)


def _pe_stmt(stmt: Stmt, env: dict[str, int]) -> Stmt | None:
    if isinstance(stmt, Assign):
        target = stmt.target if isinstance(stmt.target, str) else getattr(stmt.target, "name", str(stmt.target))
        expr_val = _try_eval(stmt.expr, env)
        if target in env and expr_val is not None:
            # Both target and expr are known: compute at PE time
            old = env[target]
            if stmt.op is UpdateOp.ADD:
                env[target] = old + expr_val
            elif stmt.op is UpdateOp.SUB:
                env[target] = old - expr_val
            else:
                env[target] = old ^ expr_val
            return None  # statement eliminated
        # Residualize with simplified expression
        simplified = _simplify(stmt.expr, env)
        return Assign(stmt.target, stmt.op, simplified)

    if isinstance(stmt, Swap):
        left = stmt.left if isinstance(stmt.left, str) else getattr(stmt.left, "name", str(stmt.left))
        right = stmt.right if isinstance(stmt.right, str) else getattr(stmt.right, "name", str(stmt.right))
        if left in env and right in env:
            env[left], env[right] = env[right], env[left]
            return None
        return stmt

    if isinstance(stmt, If):
        test_val = _try_eval(stmt.test, env)
        if test_val is not None:
            if test_val != 0:
                body = _pe_block(stmt.then_block, env)
                return _unwrap_block(body)
            else:
                body = _pe_block(stmt.else_block, env)
                return _unwrap_block(body)
        # Can't determine branch: residualize both
        then_b = _pe_block(stmt.then_block, dict(env))
        else_b = _pe_block(stmt.else_block, dict(env))
        return If(_simplify(stmt.test, env), then_b, else_b, _simplify(stmt.assertion, env))

    if isinstance(stmt, Loop):
        # Loops: always residualize (online PE doesn't unroll unbounded loops)
        do_b = _pe_block(stmt.do_block, dict(env))
        loop_b = _pe_block(stmt.loop_block, dict(env))
        return Loop(
            _simplify(stmt.entry_guard, env), do_b, loop_b,
            _simplify(stmt.exit_guard, env),
        )

    if isinstance(stmt, Rif):
        test_val = _try_eval(stmt.test, env)
        if test_val is not None:
            # Known direction: specialize to just the appropriate execution
            body = _pe_block(stmt.body, env)
            if test_val != 0:
                # Reverse execution — invert the body
                from pyrev_fl.invert import invert_block
                return _unwrap_block(invert_block(body))
            else:
                return _unwrap_block(body)
        return Rif(_simplify(stmt.test, env), _pe_block(stmt.body, dict(env)),
                    _simplify(stmt.assertion, env))

    if isinstance(stmt, (Push, Pop)):
        return stmt

    raise TypeError(f"unknown statement: {stmt!r}")


def _try_eval(expr: Expr, env: dict[str, int]) -> int | None:
    """Try to evaluate expr given known values. Return None if unknown."""
    if isinstance(expr, Const):
        return expr.value
    if isinstance(expr, Var):
        return env.get(expr.name)
    if isinstance(expr, Binary):
        left = _try_eval(expr.left, env)
        right = _try_eval(expr.right, env)
        if left is None or right is None:
            return None
        ops = {
            BinOp.ADD: lambda a, b: a + b,
            BinOp.SUB: lambda a, b: a - b,
            BinOp.MUL: lambda a, b: a * b,
            BinOp.EQ: lambda a, b: int(a == b),
            BinOp.NE: lambda a, b: int(a != b),
            BinOp.LT: lambda a, b: int(a < b),
            BinOp.LE: lambda a, b: int(a <= b),
            BinOp.GT: lambda a, b: int(a > b),
            BinOp.GE: lambda a, b: int(a >= b),
        }
        fn = ops.get(expr.op)
        if fn is None:
            return None
        return fn(left, right)
    return None


def _simplify(expr: Expr, env: dict[str, int]) -> Expr:
    """Simplify expression by substituting known values."""
    val = _try_eval(expr, env)
    if val is not None:
        return Const(val)
    if isinstance(expr, Binary):
        return Binary(expr.op, _simplify(expr.left, env), _simplify(expr.right, env))
    return expr


def _unwrap_block(block: Block) -> Stmt | None:
    """Unwrap a single-statement block; return None for empty blocks."""
    if not block.stmts:
        return None
    if len(block.stmts) == 1:
        return block.stmts[0]
    # Multiple statements: wrap in a synthetic sequential block
    # Return the first statement and note the rest are lost (limitation)
    # For correctness, return as-is via a dummy If with always-true condition
    return If(Const(1), block, Block([]), Const(1))
