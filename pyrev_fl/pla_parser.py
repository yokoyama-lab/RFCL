"""PLA (PISA-style reversible assembly) → RL parser.

Parses the assembly format produced by pla_pretty.py back into an RL program.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from pyrev_fl.ast import BinOp, Binary, Const, Expr, UpdateOp, Var
from pyrev_fl.rl_ast import (
    Assign, Block, Exit, FiFrom, FromEntry, FromLabel,
    Goto, IfGoto, Program, RFromLabel, RGoto, Swap,
)


class PlaParseError(ValueError):
    pass


def parse_pla(source: str) -> Program:
    """Parse PLA assembly text and return an RL Program."""
    lines = source.splitlines()
    # Parse header
    header_line = None
    body_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("("):
            header_line = stripped
            body_start = i + 1
            break
    if header_line is None:
        raise PlaParseError("missing program header")

    inputs, outputs, temps = _parse_header(header_line)

    # Parse blocks
    blocks: list[Block] = []
    current_label: str | None = None
    current_from = None
    current_assigns: list = []
    current_jump = None

    for i in range(body_start, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            continue

        if line.endswith(":") and not line.startswith(("ADD", "SUB", "XOR", "EXCH", "BRA", "RBRA", "BNEZ", "HALT", "ENTRY", "FROM", "RFROM", "FI")):
            # New block label
            if current_label is not None and current_jump is not None:
                blocks.append(Block(current_label, current_from, current_assigns, current_jump))
            current_label = line[:-1].strip()
            current_from = None
            current_assigns = []
            current_jump = None
            continue

        if current_label is None:
            raise PlaParseError(f"instruction outside block: {line}")

        if line == "ENTRY":
            current_from = FromEntry()
        elif line.startswith("FROM "):
            current_from = FromLabel(line[5:].strip())
        elif line.startswith("RFROM "):
            current_from = RFromLabel(line[6:].strip())
        elif line.startswith("FI "):
            current_from = _parse_fi(line)
        elif line.startswith("ADD "):
            target, expr = _parse_binop(line[4:])
            current_assigns.append(Assign(target, UpdateOp.ADD, expr))
        elif line.startswith("SUB "):
            target, expr = _parse_binop(line[4:])
            current_assigns.append(Assign(target, UpdateOp.SUB, expr))
        elif line.startswith("XOR "):
            target, expr = _parse_binop(line[4:])
            current_assigns.append(Assign(target, UpdateOp.XOR, expr))
        elif line.startswith("EXCH "):
            left, right = line[5:].split(",", 1)
            current_assigns.append(Swap(left.strip(), right.strip()))
        elif line.startswith("BRA "):
            current_jump = Goto(line[4:].strip())
        elif line.startswith("RBRA "):
            current_jump = RGoto(line[5:].strip())
        elif line.startswith("BNEZ "):
            current_jump = _parse_bnez(line[5:])
        elif line == "HALT":
            current_jump = Exit()
        else:
            raise PlaParseError(f"unknown instruction: {line}")

    if current_label is not None and current_jump is not None:
        blocks.append(Block(current_label, current_from, current_assigns, current_jump))

    return Program(inputs, outputs, temps, blocks)


def _parse_header(line: str) -> tuple[list[str], list[str], list[str]]:
    groups = re.findall(r"\(([^)]*)\)", line)
    if len(groups) != 3:
        raise PlaParseError(f"expected 3 groups in header: {line}")
    return (
        [t for t in groups[0].split() if t],
        [t for t in groups[1].split() if t],
        [t for t in groups[2].split() if t],
    )


def _parse_fi(line: str) -> FiFrom:
    # FI expr FROM label1 ELSE label2
    m = re.match(r"FI\s+(.+?)\s+FROM\s+(\S+)\s+ELSE\s+(\S+)", line)
    if m is None:
        raise PlaParseError(f"invalid FI: {line}")
    expr = _parse_expr(m.group(1).strip())
    return FiFrom(expr, m.group(2), m.group(3))


def _parse_bnez(text: str) -> IfGoto:
    # expr, label1, label2
    parts = text.split(",")
    if len(parts) != 3:
        raise PlaParseError(f"BNEZ needs expr, label, label: {text}")
    expr = _parse_expr(parts[0].strip())
    return IfGoto(expr, parts[1].strip(), parts[2].strip())


def _parse_binop(text: str) -> tuple[str, Expr]:
    parts = text.split(",", 1)
    if len(parts) != 2:
        raise PlaParseError(f"expected target, expr: {text}")
    target = parts[0].strip()
    expr = _parse_expr(parts[1].strip())
    return target, expr


def _parse_expr(text: str) -> Expr:
    """Parse an expression in prefix notation: (op left right) or atom."""
    text = text.strip()
    if text.startswith("("):
        inner = text[1:-1].strip()
        tokens = _tokenize_expr(inner)
        if len(tokens) < 3:
            raise PlaParseError(f"invalid expression: {text}")
        op_str = tokens[0]
        # Find the split point for left and right subexpressions
        left_str, right_str = _split_binary_args(" ".join(tokens[1:]))
        op = _parse_op(op_str)
        return Binary(op, _parse_expr(left_str), _parse_expr(right_str))
    try:
        return Const(int(text))
    except ValueError:
        return Var(text)


def _tokenize_expr(text: str) -> list[str]:
    tokens = []
    current = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch.isspace() and depth == 0:
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(ch)
    if current:
        tokens.append("".join(current))
    return tokens


def _split_binary_args(text: str) -> tuple[str, str]:
    depth = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch.isspace() and depth == 0 and i > 0:
            return text[:i].strip(), text[i:].strip()
    raise PlaParseError(f"cannot split binary args: {text}")


_OP_MAP = {
    "+": BinOp.ADD, "-": BinOp.SUB, "*": BinOp.MUL, "/": BinOp.DIV,
    "=": BinOp.EQ, "!=": BinOp.NE,
    "<": BinOp.LT, "<=": BinOp.LE, ">": BinOp.GT, ">=": BinOp.GE,
}


def _parse_op(text: str) -> BinOp:
    op = _OP_MAP.get(text)
    if op is None:
        raise PlaParseError(f"unknown operator: {text}")
    return op
