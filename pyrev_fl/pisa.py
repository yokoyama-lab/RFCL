"""RL → PISA code generator.

Generates PISA (Pendulum Instruction Set Architecture) assembly from RL
programs, following Frank 1999. Each RL block is compiled to a sequence
of PISA instructions.

PISA register conventions:
  R0 = zero register (always 0)
  R1..Rn = program variables (mapped by name)
  PC = program counter (implicit)

PISA instructions used:
  ADD Ri, Rj, Rk    ; Ri += Rj (Rk unused, set to R0)
  SUB Ri, Rj, Rk    ; Ri -= Rj
  XOR Ri, Rj, Rk    ; Ri ^= Rj
  EXCH Ri, Rj       ; swap Ri and Rj
  BRA label          ; unconditional branch
  RBRA label         ; reverse branch (changes execution direction)
  BEQ Ri, Rj, label  ; branch if Ri == Rj
  BNE Ri, Rj, label  ; branch if Ri != Rj
  BLT Ri, Rj, label  ; branch if Ri < Rj
"""
from __future__ import annotations

from pyrev_fl.ast import BinOp, Binary, Const, Expr, UpdateOp, Var
from pyrev_fl.rl_ast import (
    Assign, Block, Exit, FiFrom, FromEntry, FromLabel,
    Goto, IfGoto, Program, RFromLabel, RGoto, Swap,
)


def compile_to_pisa(program: Program) -> str:
    """Compile an RL program to PISA assembly."""
    # Build register allocation: variable → register name
    all_vars: list[str] = []
    for name in program.inputs + program.outputs + program.temps:
        if name not in all_vars:
            all_vars.append(name)
    reg_map = {name: f"R{i+1}" for i, name in enumerate(all_vars)}

    lines: list[str] = [
        f"; PISA assembly generated from RL program",
        f"; Variables: {', '.join(f'{n}={r}' for n, r in reg_map.items())}",
        f"; R0 = zero register",
        "",
    ]

    for block in program.blocks:
        lines.extend(_compile_block(block, reg_map))
        lines.append("")

    return "\n".join(lines)


def _compile_block(block: Block, reg_map: dict[str, str]) -> list[str]:
    lines = [f"{block.label}:"]

    # From clause (assertion/entry — generates comment or check code)
    lines.append(f"    ; {_describe_from(block.from_)}")

    # Assignments
    for assign in block.assigns:
        lines.extend(_compile_assign(assign, reg_map))

    # Jump
    lines.extend(_compile_jump(block.jump, reg_map))

    return lines


def _compile_assign(assign, reg_map: dict[str, str]) -> list[str]:
    if isinstance(assign, Assign):
        target = _resolve_reg(assign.target, reg_map)
        src = _compile_expr_to_reg(assign.expr, reg_map)
        op = {UpdateOp.ADD: "ADD", UpdateOp.SUB: "SUB", UpdateOp.XOR: "XOR"}[assign.op]
        return [f"    {op} {target}, {src}, R0"]
    if isinstance(assign, Swap):
        left = _resolve_reg(assign.left, reg_map)
        right = _resolve_reg(assign.right, reg_map)
        return [f"    EXCH {left}, {right}"]
    return [f"    ; unknown assign: {assign!r}"]


def _compile_jump(jump, reg_map: dict[str, str]) -> list[str]:
    if isinstance(jump, Goto):
        return [f"    BRA {jump.label}"]
    if isinstance(jump, RGoto):
        return [f"    RBRA {jump.label}"]
    if isinstance(jump, IfGoto):
        return _compile_conditional(jump, reg_map)
    if isinstance(jump, Exit):
        return [f"    HALT"]
    return [f"    ; unknown jump: {jump!r}"]


def _compile_conditional(jump: IfGoto, reg_map: dict[str, str]) -> list[str]:
    expr = jump.expr
    if isinstance(expr, Binary):
        left = _compile_expr_to_reg(expr.left, reg_map)
        right = _compile_expr_to_reg(expr.right, reg_map)
        op_map = {
            BinOp.EQ: "BEQ", BinOp.NE: "BNE",
            BinOp.LT: "BLT", BinOp.GT: "BGT",
        }
        instr = op_map.get(expr.op, "BNE")
        return [
            f"    {instr} {left}, {right}, {jump.true_label}",
            f"    BRA {jump.false_label}",
        ]
    # Non-binary: test against zero
    src = _compile_expr_to_reg(expr, reg_map)
    return [
        f"    BNE {src}, R0, {jump.true_label}",
        f"    BRA {jump.false_label}",
    ]


def _compile_expr_to_reg(expr: Expr, reg_map: dict[str, str]) -> str:
    if isinstance(expr, Const):
        if expr.value == 0:
            return "R0"
        return f"#{expr.value}"  # immediate value
    if isinstance(expr, Var):
        return reg_map.get(expr.name, f"?{expr.name}")
    if isinstance(expr, Binary):
        # For simple cases, inline
        return f"({_compile_expr_to_reg(expr.left, reg_map)} {expr.op.value} {_compile_expr_to_reg(expr.right, reg_map)})"
    return f"?{expr!r}"


def _resolve_reg(place, reg_map: dict[str, str]) -> str:
    if isinstance(place, str):
        return reg_map.get(place, f"?{place}")
    if isinstance(place, Var):
        return reg_map.get(place.name, f"?{place.name}")
    return f"?{place!r}"


def _describe_from(from_) -> str:
    if isinstance(from_, FromEntry):
        return "entry"
    if isinstance(from_, FromLabel):
        return f"from {from_.label}"
    if isinstance(from_, RFromLabel):
        return f"rfrom {from_.label}"
    if isinstance(from_, FiFrom):
        return f"fi ... from {from_.true_label} else {from_.false_label}"
    return str(from_)
