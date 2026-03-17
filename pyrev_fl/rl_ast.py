from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pyrev_fl.ast import BinOp, Expr, UpdateOp


@dataclass(frozen=True)
class Program:
    inputs: list[str]
    outputs: list[str]
    temps: list[str]
    blocks: list["Block"]


@dataclass(frozen=True)
class Assign:
    target: str
    op: UpdateOp
    expr: Expr


@dataclass(frozen=True)
class Swap:
    left: str
    right: str


Assignment = Assign | Swap


class Direction(Enum):
    FORWARD = "for"
    BACKWARD = "back"


@dataclass(frozen=True)
class FromEntry:
    pass


@dataclass(frozen=True)
class FromLabel:
    label: str


@dataclass(frozen=True)
class RFromLabel:
    label: str


@dataclass(frozen=True)
class FiFrom:
    expr: Expr
    true_label: str
    false_label: str


From = FromEntry | FromLabel | RFromLabel | FiFrom


@dataclass(frozen=True)
class Goto:
    label: str


@dataclass(frozen=True)
class RGoto:
    label: str


@dataclass(frozen=True)
class IfGoto:
    expr: Expr
    true_label: str
    false_label: str


@dataclass(frozen=True)
class Exit:
    pass


Jump = Goto | RGoto | IfGoto | Exit


@dataclass(frozen=True)
class Block:
    label: str
    from_: From
    assigns: list[Assignment]
    jump: Jump

