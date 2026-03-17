from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Program:
    inputs: list[str]
    outputs: list[str]
    temps: list[str]
    body: "Block"


@dataclass(frozen=True)
class Block:
    stmts: list["Stmt"]


class UpdateOp(Enum):
    ADD = "+="
    SUB = "-="
    XOR = "^="


class BinOp(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    EQ = "="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="


@dataclass(frozen=True)
class Expr:
    pass


@dataclass(frozen=True)
class Place:
    pass


@dataclass(frozen=True)
class Const(Expr):
    value: int


@dataclass(frozen=True)
class Var(Expr, Place):
    name: str


@dataclass(frozen=True)
class ArrayRef(Expr, Place):
    name: str
    index: Expr


@dataclass(frozen=True)
class Binary(Expr):
    op: BinOp
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Stmt:
    pass


@dataclass(frozen=True)
class Assign(Stmt):
    target: str | Place
    op: UpdateOp
    expr: Expr


@dataclass(frozen=True)
class Swap(Stmt):
    left: str | Place
    right: str | Place


@dataclass(frozen=True)
class If(Stmt):
    test: Expr
    then_block: Block
    else_block: Block
    assertion: Expr


@dataclass(frozen=True)
class Loop(Stmt):
    entry_guard: Expr
    do_block: Block
    loop_block: Block
    exit_guard: Expr


@dataclass(frozen=True)
class Rif(Stmt):
    test: Expr
    body: Block
    assertion: Expr


@dataclass(frozen=True)
class Push(Stmt):
    """push var stack; — push var's value onto stack, then zero var."""
    var: str
    stack: str


@dataclass(frozen=True)
class Pop(Stmt):
    """pop var stack; — pop top of stack into var (var must be 0)."""
    var: str
    stack: str
