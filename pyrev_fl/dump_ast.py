from __future__ import annotations

from pyrev_fl.ast import ArrayRef, Assign, Binary, Block, Const, Expr, If, Loop, Place, Program, Rif, Stmt, Swap, Var


def dump_program(program: Program) -> dict[str, object]:
    return {
        "kind": "srl_program",
        "inputs": list(program.inputs),
        "outputs": list(program.outputs),
        "temps": list(program.temps),
        "body": dump_block(program.body),
    }


def dump_block(block: Block) -> dict[str, object]:
    return {
        "kind": "block",
        "stmts": [dump_stmt(stmt) for stmt in block.stmts],
    }


def dump_stmt(stmt: Stmt) -> dict[str, object]:
    if isinstance(stmt, Assign):
        return {
            "kind": "assign",
            "target": dump_place(stmt.target),
            "op": stmt.op.value,
            "expr": dump_expr(stmt.expr),
        }
    if isinstance(stmt, Swap):
        return {
            "kind": "swap",
            "left": dump_place(stmt.left),
            "right": dump_place(stmt.right),
        }
    if isinstance(stmt, If):
        return {
            "kind": "if",
            "test": dump_expr(stmt.test),
            "then": dump_block(stmt.then_block),
            "else": dump_block(stmt.else_block),
            "assertion": dump_expr(stmt.assertion),
        }
    if isinstance(stmt, Loop):
        return {
            "kind": "loop",
            "entry_guard": dump_expr(stmt.entry_guard),
            "do": dump_block(stmt.do_block),
            "loop": dump_block(stmt.loop_block),
            "exit_guard": dump_expr(stmt.exit_guard),
        }
    if isinstance(stmt, Rif):
        return {
            "kind": "rif",
            "test": dump_expr(stmt.test),
            "body": dump_block(stmt.body),
            "assertion": dump_expr(stmt.assertion),
        }
    raise TypeError(f"unknown statement: {stmt!r}")


def dump_expr(expr: Expr) -> dict[str, object]:
    if isinstance(expr, Const):
        return {"kind": "const", "value": expr.value}
    if isinstance(expr, Var):
        return {"kind": "var", "name": expr.name}
    if isinstance(expr, ArrayRef):
        return {"kind": "array_ref", "name": expr.name, "index": dump_expr(expr.index)}
    if isinstance(expr, Binary):
        return {
            "kind": "binary",
            "op": expr.op.value,
            "left": dump_expr(expr.left),
            "right": dump_expr(expr.right),
        }
    raise TypeError(f"unknown expression: {expr!r}")


def dump_place(place: str | Place) -> dict[str, object]:
    if isinstance(place, str):
        return {"kind": "var", "name": place}
    return dump_expr(place)
