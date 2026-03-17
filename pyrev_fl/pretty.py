from __future__ import annotations

from pyrev_fl.ast import ArrayRef, Assign, Binary, Block, Const, Expr, If, Loop, Place, Pop, Program, Push, Rif, Stmt, Swap, Var


def render_program(program: Program) -> str:
    parts = [
        _render_name_list(program.inputs),
        _render_name_list(program.outputs),
        _render_name_list(program.temps),
    ]
    lines = [" ".join(parts)]
    _render_block(lines, program.body, 0)
    return "\n".join(lines) + "\n"


def _render_name_list(names: list[str]) -> str:
    return f"({' '.join(names)})"


def _render_block(lines: list[str], block: Block, indent: int) -> None:
    for stmt in block.stmts:
        _render_stmt(lines, stmt, indent)


def _render_stmt(lines: list[str], stmt: Stmt, indent: int) -> None:
    prefix = " " * indent
    if isinstance(stmt, Assign):
        lines.append(f"{prefix}{_render_place(stmt.target)} {stmt.op.value} {_render_expr(stmt.expr)};")
        return
    if isinstance(stmt, Swap):
        lines.append(f"{prefix}{_render_place(stmt.left)} <=> {_render_place(stmt.right)};")
        return
    if isinstance(stmt, If):
        lines.append(f"{prefix}if {_render_expr(stmt.test)} then")
        _render_block(lines, stmt.then_block, indent + 2)
        lines.append(f"{prefix}else")
        _render_block(lines, stmt.else_block, indent + 2)
        lines.append(f"{prefix}fi {_render_expr(stmt.assertion)}")
        return
    if isinstance(stmt, Loop):
        lines.append(f"{prefix}from {_render_expr(stmt.entry_guard)} do")
        _render_block(lines, stmt.do_block, indent + 2)
        lines.append(f"{prefix}loop")
        _render_block(lines, stmt.loop_block, indent + 2)
        lines.append(f"{prefix}until {_render_expr(stmt.exit_guard)}")
        return
    if isinstance(stmt, Rif):
        lines.append(f"{prefix}rif {_render_expr(stmt.test)}")
        _render_block(lines, stmt.body, indent + 2)
        lines.append(f"{prefix}rfi {_render_expr(stmt.assertion)}")
        return
    if isinstance(stmt, Push):
        lines.append(f"{prefix}push {stmt.var} {stmt.stack};")
        return
    if isinstance(stmt, Pop):
        lines.append(f"{prefix}pop {stmt.var} {stmt.stack};")
        return
    raise TypeError(f"unknown statement: {stmt!r}")


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
