from __future__ import annotations

from pyrev_fl.rl_ast import Block, Exit, FiFrom, FromEntry, Goto, IfGoto, Program, RGoto
from pyrev_fl.rl_pretty import _render_assignment, _render_expr, _render_from, _render_jump


def render_dot(program: Program) -> str:
    """Return a GraphViz DOT digraph string for *program*'s control-flow graph."""
    lines: list[str] = [
        "digraph rl {",
        '    rankdir=TB;',
        '    node [shape=record fontname="Courier"];',
        '    edge [fontname="Courier" fontsize=10];',
    ]

    for block in program.blocks:
        lines.append(_dot_node(block))

    for block in program.blocks:
        lines.extend(_dot_edges(block))

    lines.append("}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Node rendering
# ---------------------------------------------------------------------------

def _dot_node(block: Block) -> str:
    is_entry = isinstance(block.from_, FromEntry)
    is_exit = isinstance(block.jump, Exit)

    label_parts: list[str] = [
        _esc(block.label),
        _esc(_render_from(block.from_)),
    ]
    for assign in block.assigns:
        label_parts.append(_esc(_render_assignment(assign)))
    label_parts.append(_esc(_render_jump(block.jump)))

    label = r"\n".join(label_parts)

    attrs: list[str] = [f'label="{label}"']
    if is_entry or is_exit:
        attrs.append('style="bold"')
        attrs.append('shape="Mrecord"')

    return f'    {_dot_id(block.label)} [{" ".join(attrs)}];'


# ---------------------------------------------------------------------------
# Edge rendering
# ---------------------------------------------------------------------------

def _dot_edges(block: Block) -> list[str]:
    src = _dot_id(block.label)
    jump = block.jump

    if isinstance(jump, Goto):
        return [f'    {src} -> {_dot_id(jump.label)};']

    if isinstance(jump, RGoto):
        return [f'    {src} -> {_dot_id(jump.label)} [style=dashed label="R"];']

    if isinstance(jump, IfGoto):
        true_edge = f'    {src} -> {_dot_id(jump.true_label)} [label="T" color=darkgreen];'
        false_edge = f'    {src} -> {_dot_id(jump.false_label)} [label="F" style=dashed color=red];'
        return [true_edge, false_edge]

    if isinstance(jump, Exit):
        return []  # no outgoing edges from exit block

    raise TypeError(f"unknown jump: {jump!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dot_id(label: str) -> str:
    """Return a DOT-safe identifier for a block label."""
    # Replace characters that would break DOT identifiers
    safe = label.replace("-", "_").replace(".", "_")
    return f'"{safe}"'


def _esc(text: str) -> str:
    """Escape characters special to DOT record labels and HTML."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "\\<")
        .replace(">", "\\>")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace('"', '\\"')
        .replace("|", "\\|")
    )
