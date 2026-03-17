from __future__ import annotations

from pyrev_fl.ast import ArrayRef, Binary, Const, Expr, Place, Var
from pyrev_fl.interface import build_layout, parse_decl
from pyrev_fl.rl_ast import Assign, Block, Exit, FiFrom, FromEntry, FromLabel, Goto, IfGoto, Program, RFromLabel, RGoto, Swap


class CheckError(ValueError):
    pass


def check_program(program: Program) -> None:
    seen: set[str] = set()
    layout = build_layout(program.inputs, program.outputs, program.temps)
    declared = layout.declarations
    for block in program.blocks:
        if block.label in seen:
            raise CheckError(f"duplicate block label: {block.label}")
        seen.add(block.label)

    interface_names = {parse_decl(name).name for name in program.inputs + program.outputs}
    for name in program.temps:
        base = parse_decl(name).name
        if base in interface_names:
            raise CheckError(f"temporary variable overlaps interface: {base}")

    entry_count = sum(isinstance(block.from_, FromEntry) for block in program.blocks)
    exit_count = sum(isinstance(block.jump, Exit) for block in program.blocks)
    if entry_count != 1:
        raise CheckError(f"expected exactly one entry block, found {entry_count}")
    if exit_count != 1:
        raise CheckError(f"expected exactly one exit block, found {exit_count}")

    labels = {block.label for block in program.blocks}
    for block in program.blocks:
        _check_block(block, declared, labels)
    _check_from_jump_consistency(program.blocks)
    _check_reachability(program, labels)


def _check_block(block: Block, declared: dict[str, object], labels: set[str]) -> None:
    for assign in block.assigns:
        if isinstance(assign, Assign):
            _check_place(assign.target, declared)
            _check_expr(assign.expr, declared)
            if _expr_mentions_target(assign.expr, assign.target):
                raise CheckError("assignment expression mentions target")
        elif isinstance(assign, Swap):
            _check_place(assign.left, declared)
            _check_place(assign.right, declared)
            if assign.left == assign.right:
                raise CheckError(f"swap uses the same variable twice: {_place_name(assign.left)}")
            if _array_swap_may_alias(assign.left, assign.right):
                raise CheckError(f"swap may alias array elements: {_place_base_name(assign.left)}")

    _check_from(block.from_, declared, labels)
    _check_jump(block.jump, declared, labels)


def _check_from(from_, declared: dict[str, object], labels: set[str]) -> None:
    if isinstance(from_, FromEntry):
        return
    if isinstance(from_, FromLabel):
        _check_label(from_.label, labels)
        return
    if isinstance(from_, RFromLabel):
        _check_label(from_.label, labels)
        return
    if isinstance(from_, FiFrom):
        _check_expr(from_.expr, declared)
        _check_label(from_.true_label, labels)
        _check_label(from_.false_label, labels)
        return
    raise TypeError(f"unknown from construct: {from_!r}")


def _check_jump(jump, declared: dict[str, object], labels: set[str]) -> None:
    if isinstance(jump, Goto):
        _check_label(jump.label, labels)
        return
    if isinstance(jump, RGoto):
        _check_label(jump.label, labels)
        return
    if isinstance(jump, IfGoto):
        _check_expr(jump.expr, declared)
        _check_label(jump.true_label, labels)
        _check_label(jump.false_label, labels)
        return
    if isinstance(jump, Exit):
        return
    raise TypeError(f"unknown jump construct: {jump!r}")


def _check_expr(expr: Expr, declared: dict[str, object]) -> None:
    if isinstance(expr, Const):
        return
    if isinstance(expr, Var):
        _check_scalar(expr.name, declared)
        return
    if isinstance(expr, ArrayRef):
        _check_array(expr.name, declared)
        _check_expr(expr.index, declared)
        return
    if isinstance(expr, Binary):
        _check_expr(expr.left, declared)
        _check_expr(expr.right, declared)
        return
    raise TypeError(f"unknown expression: {expr!r}")


def _check_scalar(name: str, declared: dict[str, object]) -> None:
    decl = declared.get(name)
    if decl is None:
        raise CheckError(f"undeclared variable: {name}")
    if getattr(decl, "is_array", False):
        raise CheckError(f"array variable requires index: {name}")


def _check_array(name: str, declared: dict[str, object]) -> None:
    decl = declared.get(name)
    if decl is None:
        raise CheckError(f"undeclared variable: {name}")
    if not getattr(decl, "is_array", False):
        raise CheckError(f"scalar variable does not support indexing: {name}")


def _check_label(label: str, labels: set[str]) -> None:
    if label not in labels:
        raise CheckError(f"unknown block label: {label}")


def _check_place(place: str | Place, declared: dict[str, object]) -> None:
    if isinstance(place, str):
        _check_scalar(place, declared)
        return
    if isinstance(place, Var):
        _check_scalar(place.name, declared)
        return
    if isinstance(place, ArrayRef):
        _check_array(place.name, declared)
        _check_expr(place.index, declared)
        return
    raise TypeError(f"unknown place: {place!r}")


def _expr_mentions_target(expr: Expr, target: str | Place) -> bool:
    if isinstance(target, str):
        return _expr_mentions_scalar(expr, target)
    if isinstance(target, Var):
        return _expr_mentions_scalar(expr, target.name)
    if isinstance(target, ArrayRef):
        return _expr_mentions_array(expr, target.name)
    raise TypeError(f"unknown target: {target!r}")


def _expr_mentions_scalar(expr: Expr, target: str) -> bool:
    if isinstance(expr, Const):
        return False
    if isinstance(expr, Var):
        return expr.name == target
    if isinstance(expr, ArrayRef):
        return _expr_mentions_scalar(expr.index, target)
    if isinstance(expr, Binary):
        return _expr_mentions_scalar(expr.left, target) or _expr_mentions_scalar(expr.right, target)
    raise TypeError(f"unknown expression: {expr!r}")


def _expr_mentions_array(expr: Expr, target: str) -> bool:
    if isinstance(expr, Const):
        return False
    if isinstance(expr, Var):
        return False
    if isinstance(expr, ArrayRef):
        return expr.name == target or _expr_mentions_array(expr.index, target) or _expr_mentions_scalar(expr.index, target)
    if isinstance(expr, Binary):
        return _expr_mentions_array(expr.left, target) or _expr_mentions_array(expr.right, target)
    raise TypeError(f"unknown expression: {expr!r}")


def _array_swap_may_alias(left: str | Place, right: str | Place) -> bool:
    if not isinstance(left, ArrayRef) or not isinstance(right, ArrayRef):
        return False
    if left.name != right.name:
        return False
    left_index = _const_index(left.index)
    right_index = _const_index(right.index)
    if left_index is None or right_index is None:
        return True
    return left_index == right_index


def _const_index(expr: Expr) -> int | None:
    if isinstance(expr, Const):
        return expr.value
    return None


def _place_name(place: str | Place) -> str:
    if isinstance(place, str):
        return place
    if isinstance(place, Var):
        return place.name
    if isinstance(place, ArrayRef):
        return f"{place.name}[...]"
    raise TypeError(f"unknown place: {place!r}")


def _place_base_name(place: str | Place) -> str:
    if isinstance(place, str):
        return place
    if isinstance(place, Var):
        return place.name
    if isinstance(place, ArrayRef):
        return place.name
    raise TypeError(f"unknown place: {place!r}")


def _check_from_jump_consistency(blocks: list[Block]) -> None:
    """Static check: each block's from_ annotation must be consistent with its predecessor's jump.

    Checks:
    - FromLabel(A): A's jump must target this block (Goto or IfGoto).
    - FiFrom(e, T, F): T and F must each have Goto pointing to this block.
    - RFromLabel is not checked here (complex backward-to-forward semantics).
    """
    block_map = {b.label: b for b in blocks}
    for block in blocks:
        from_ = block.from_
        if isinstance(from_, FromLabel):
            pred = block_map.get(from_.label)
            if pred is not None and block.label not in _jump_targets(pred.jump):
                raise CheckError(
                    f"from/jump inconsistency: {block.label} claims predecessor {from_.label} "
                    f"but {from_.label} does not jump to {block.label}"
                )
        elif isinstance(from_, FiFrom):
            for pred_label in (from_.true_label, from_.false_label):
                pred = block_map.get(pred_label)
                if pred is not None and block.label not in _jump_targets(pred.jump):
                    raise CheckError(
                        f"fi-from/jump inconsistency: {block.label} claims predecessor {pred_label} "
                        f"but {pred_label} does not jump to {block.label}"
                    )


def _jump_targets(jump) -> set[str]:
    if isinstance(jump, (Goto, RGoto)):
        return {jump.label}
    if isinstance(jump, IfGoto):
        return {jump.true_label, jump.false_label}
    return set()


def _check_reachability(program: Program, labels: set[str]) -> None:
    entry = next(block.label for block in program.blocks if isinstance(block.from_, FromEntry))
    block_map = {block.label: block for block in program.blocks}
    adjacency = {label: set() for label in labels}
    for block in program.blocks:
        for next_label in _jump_successors(block):
            if next_label in labels:
                adjacency[block.label].add(next_label)
                adjacency[next_label].add(block.label)
        for prev_label in _from_neighbors(block):
            if prev_label in labels:
                adjacency[prev_label].add(block.label)
                adjacency[block.label].add(prev_label)

    seen: set[str] = set()
    stack = [entry]

    while stack:
        label = stack.pop()
        if label in seen:
            continue
        seen.add(label)
        stack.extend(sorted(adjacency[label] - seen))

    unreachable = labels - seen
    if unreachable:
        raise CheckError(f"unreachable blocks: {', '.join(sorted(unreachable))}")

    if not any(isinstance(block_map[label].jump, Exit) for label in seen):
        raise CheckError("exit block is not reachable from entry")


def _jump_successors(block: Block) -> list[str]:
    if isinstance(block.jump, Goto):
        return [block.jump.label]
    if isinstance(block.jump, RGoto):
        return [block.jump.label]
    if isinstance(block.jump, IfGoto):
        return [block.jump.true_label, block.jump.false_label]
    return []


def _from_neighbors(block: Block) -> list[str]:
    if isinstance(block.from_, FromLabel):
        return [block.from_.label]
    if isinstance(block.from_, RFromLabel):
        return [block.from_.label]
    if isinstance(block.from_, FiFrom):
        return [block.from_.true_label, block.from_.false_label]
    return []
