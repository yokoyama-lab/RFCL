"""SRL → Janus renderer.

Converts an SRL program to a Janus ``procedure main()`` that can be run
directly with the Jana interpreter.  The SRL interface (inputs / outputs /
temps) is stored in a comment header so that the round-trip parser can
reconstruct it.

Key difference from SRL:
- Janus uses infix expressions: ``x = 0``, ``(x + y)`` etc.
- Janus has no semicolons on statements.
- Janus uses ``skip`` for empty branches.
- ``rif (t) S rfi (a)`` is expanded to ``if t then invert(S) else S fi a``.
"""
from __future__ import annotations

from pyrev_fl.ast import Assign, Binary, BinOp, Block, Const, Expr, If, Loop, Program, Rif, Stmt, Swap, UpdateOp, Var
from pyrev_fl.invert import invert_block

_INDENT = "    "

_BIN_OP_JANUS: dict[BinOp, str] = {
    BinOp.ADD: "+",
    BinOp.SUB: "-",
    BinOp.MUL: "*",
    BinOp.DIV: "/",
    BinOp.EQ:  "=",
    BinOp.NE:  "!=",
    BinOp.LT:  "<",
    BinOp.LE:  "<=",
    BinOp.GT:  ">",
    BinOp.GE:  ">=",
}

_UPDATE_OP_JANUS: dict[UpdateOp, str] = {
    UpdateOp.ADD: "+=",
    UpdateOp.SUB: "-=",
    UpdateOp.XOR: "^=",
}


def render_janus(program: Program) -> str:
    """Return the Janus text for *program*."""
    lines: list[str] = [
        f"// SRL inputs=({' '.join(program.inputs)})"
        f" outputs=({' '.join(program.outputs)})"
        f" temps=({' '.join(program.temps)})",
        "procedure main()",
    ]
    # Declare all variables (deduplicated, inputs first, then extra outputs, then temps)
    seen: set[str] = set()
    for name in _ordered_unique(program.inputs, program.outputs, program.temps):
        lines.append(f"{_INDENT}int {_janus_ident(name)}")
        seen.add(name)
    for stmt in program.body.stmts:
        lines.extend(_render_stmt(stmt, indent=1))
    return "\n".join(lines) + "\n"


def _ordered_unique(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for group in groups:
        for name in group:
            if name not in seen:
                seen.add(name)
                result.append(name)
    return result


# ---------------------------------------------------------------------------
# Statement rendering
# ---------------------------------------------------------------------------

def _render_stmt(stmt: Stmt, indent: int) -> list[str]:
    prefix = _INDENT * indent
    if isinstance(stmt, Assign):
        op = _UPDATE_OP_JANUS[stmt.op]
        return [f"{prefix}{_render_place(stmt.target)} {op} {_render_expr(stmt.expr)}"]
    if isinstance(stmt, Swap):
        return [f"{prefix}{_render_place(stmt.left)} <=> {_render_place(stmt.right)}"]
    if isinstance(stmt, If):
        return _render_if(stmt.test, stmt.then_block, stmt.else_block, stmt.assertion, indent)
    if isinstance(stmt, Loop):
        return _render_loop(stmt.entry_guard, stmt.do_block, stmt.loop_block, stmt.exit_guard, indent)
    if isinstance(stmt, Rif):
        # rif (t) S rfi (a)  ≡  if t then invert(S) else S fi a
        return _render_if(stmt.test, invert_block(stmt.body), stmt.body, stmt.assertion, indent)
    raise TypeError(f"unknown statement: {stmt!r}")


def _render_if(test: Expr, then_block: Block, else_block: Block, assertion: Expr, indent: int) -> list[str]:
    prefix = _INDENT * indent
    lines = [f"{prefix}if {_render_expr(test)} then"]
    lines.extend(_render_block(then_block, indent + 1))
    lines.append(f"{prefix}else")
    lines.extend(_render_block(else_block, indent + 1))
    lines.append(f"{prefix}fi {_render_expr(assertion)}")
    return lines


def _render_loop(entry: Expr, do_block: Block, loop_block: Block, exit_guard: Expr, indent: int) -> list[str]:
    prefix = _INDENT * indent
    lines = [f"{prefix}from {_render_expr(entry)} do"]
    lines.extend(_render_block(do_block, indent + 1))
    lines.append(f"{prefix}loop")
    lines.extend(_render_block(loop_block, indent + 1))
    lines.append(f"{prefix}until {_render_expr(exit_guard)}")
    return lines


def _render_block(block: Block, indent: int) -> list[str]:
    if not block.stmts:
        return [_INDENT * indent + "skip"]
    lines: list[str] = []
    for stmt in block.stmts:
        lines.extend(_render_stmt(stmt, indent))
    return lines


# ---------------------------------------------------------------------------
# Expression rendering (SRL prefix → Janus infix)
# ---------------------------------------------------------------------------

def _render_expr(expr: Expr) -> str:
    if isinstance(expr, Const):
        return str(expr.value)
    if isinstance(expr, Var):
        return _janus_ident(expr.name)
    if isinstance(expr, Binary):
        op = _BIN_OP_JANUS[expr.op]
        return f"({_render_expr(expr.left)} {op} {_render_expr(expr.right)})"
    raise TypeError(f"unknown expression: {expr!r}")


def _janus_ident(name: str) -> str:
    """Convert SRL identifier to Jana-compatible identifier (hyphens → underscores)."""
    return name.replace("-", "_")


def _render_place(place) -> str:
    if isinstance(place, str):
        return _janus_ident(place)
    if isinstance(place, Var):
        return _janus_ident(place.name)
    return str(place)
