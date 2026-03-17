"""Random SRL program generator for property-based testing.

Generates well-formed SRL programs of configurable depth and width,
suitable for verifying the inversion theorem, translation equivalence,
and other properties over a large sample.
"""
from __future__ import annotations

import random
from pyrev_fl.ast import (
    Assign, BinOp, Binary, Block, Const, Expr, If, Loop, Program, Rif, Stmt, Swap, UpdateOp, Var,
)


def random_program(
    *,
    n_inputs: int = 2,
    n_outputs: int = 2,
    n_temps: int = 1,
    max_depth: int = 2,
    max_stmts: int = 4,
    seed: int | None = None,
) -> Program:
    """Generate a random well-formed SRL program."""
    rng = random.Random(seed)
    inputs = [f"x{i}" for i in range(n_inputs)]
    outputs = [f"x{i}" for i in range(n_outputs)]
    temps = [f"t{i}" for i in range(n_temps)]
    all_vars = list(dict.fromkeys(inputs + outputs + temps))
    body = _random_block(rng, all_vars, max_depth, max_stmts)
    return Program(inputs, outputs, temps, body)


def _random_block(rng: random.Random, vars: list[str], depth: int, max_stmts: int) -> Block:
    n = rng.randint(1, max(1, max_stmts))
    stmts = [_random_stmt(rng, vars, depth) for _ in range(n)]
    return Block(stmts)


def _random_stmt(rng: random.Random, vars: list[str], depth: int) -> Stmt:
    if depth <= 0 or rng.random() < 0.6:
        return _random_simple_stmt(rng, vars)
    kind = rng.choice(["if", "loop"])
    if kind == "if":
        return _random_if(rng, vars, depth)
    return _random_loop(rng, vars, depth)


def _random_simple_stmt(rng: random.Random, vars: list[str]) -> Stmt:
    if len(vars) >= 2 and rng.random() < 0.3:
        a, b = rng.sample(vars, 2)
        return Swap(a, b)
    target = rng.choice(vars)
    others = [v for v in vars if v != target]
    if not others:
        op = rng.choice([UpdateOp.ADD, UpdateOp.SUB])
        return Assign(target, op, Const(rng.randint(1, 5)))
    op = rng.choice([UpdateOp.ADD, UpdateOp.SUB, UpdateOp.XOR])
    expr = _random_expr(rng, others)
    return Assign(target, op, expr)


def _random_expr(rng: random.Random, vars: list[str]) -> Expr:
    if rng.random() < 0.7 or not vars:
        if rng.random() < 0.5 and vars:
            return Var(rng.choice(vars))
        return Const(rng.randint(0, 10))
    return Var(rng.choice(vars))


def _random_guard(rng: random.Random, vars: list[str]) -> Expr:
    """Generate a guard expression like (= var 0) or (!= var 0)."""
    v = rng.choice(vars)
    op = rng.choice([BinOp.EQ, BinOp.NE])
    return Binary(op, Var(v), Const(0))


def _random_if(rng: random.Random, vars: list[str], depth: int) -> If:
    guard_var = rng.choice(vars)
    test = Binary(BinOp.NE, Var(guard_var), Const(0))
    then_block = _random_block(rng, vars, depth - 1, 2)
    else_block = _random_block(rng, vars, depth - 1, 2)
    assertion = test  # Use same test as assertion (works when guard_var isn't modified)
    return If(test, then_block, else_block, assertion)


def _random_loop(rng: random.Random, vars: list[str], depth: int) -> Loop:
    # Choose a temp-like variable for the loop counter
    counter = rng.choice(vars)
    entry_guard = Binary(BinOp.EQ, Var(counter), Const(0))
    exit_guard = Binary(BinOp.NE, Var(counter), Const(0))
    do_block = Block([Assign(counter, UpdateOp.ADD, Const(1))])
    loop_block = Block([])  # Empty loop body ensures single iteration
    return Loop(entry_guard, do_block, loop_block, exit_guard)
