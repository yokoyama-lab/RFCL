from __future__ import annotations

from dataclasses import dataclass
import re


_DECL_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)(?:\[(\d+)\])?$")
_STACK_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*):stack$")


@dataclass(frozen=True)
class InterfaceDecl:
    name: str
    size: int | None = None
    is_stack: bool = False

    @property
    def is_array(self) -> bool:
        return self.size is not None and not self.is_stack

    def flatten(self) -> list[str]:
        if self.is_stack:
            return []  # stacks have no fixed cells
        if self.size is None:
            return [self.name]
        return [f"{self.name}[{index}]" for index in range(self.size)]


@dataclass(frozen=True)
class InterfaceLayout:
    declarations: dict[str, InterfaceDecl]
    inputs: list[str]
    outputs: list[str]
    temps: list[str]


def parse_decl(text: str) -> InterfaceDecl:
    stack_match = _STACK_RE.fullmatch(text)
    if stack_match is not None:
        return InterfaceDecl(stack_match.group(1), is_stack=True)
    match = _DECL_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"invalid interface declaration: {text}")
    name, raw_size = match.groups()
    if raw_size is None:
        return InterfaceDecl(name)
    size = int(raw_size)
    if size <= 0:
        raise ValueError(f"array size must be positive: {text}")
    return InterfaceDecl(name, size)


def build_layout(inputs: list[str], outputs: list[str], temps: list[str]) -> InterfaceLayout:
    declarations: dict[str, InterfaceDecl] = {}
    for group in (inputs, outputs, temps):
        for text in group:
            decl = parse_decl(text)
            existing = declarations.get(decl.name)
            if existing is not None and existing != decl:
                raise ValueError(f"inconsistent interface declaration: {decl.name}")
            declarations[decl.name] = decl
    return InterfaceLayout(
        declarations=declarations,
        inputs=flatten_interface(inputs),
        outputs=flatten_interface(outputs),
        temps=flatten_interface(temps),
    )


def flatten_interface(names: list[str]) -> list[str]:
    cells: list[str] = []
    for text in names:
        cells.extend(parse_decl(text).flatten())
    return cells

