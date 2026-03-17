from __future__ import annotations

from pyrev_fl.ast import ArrayRef, Assign, Binary, Block, Const, Expr, If, Loop, Place, Pop, Program, Push, Rif, Stmt, Swap, Var
from pyrev_fl.interface import build_layout, parse_decl


class CheckError(ValueError):
    pass


def check_program(program: Program) -> None:
    _check_unique_group(program.inputs)
    _check_unique_group(program.outputs)
    _check_unique_group(program.temps)
    layout = build_layout(program.inputs, program.outputs, program.temps)
    declared = layout.declarations
    interface_names = {parse_decl(name).name for name in program.inputs + program.outputs}

    for name in program.temps:
        base = parse_decl(name).name
        if base in interface_names:
            raise CheckError(f"temporary variable overlaps interface: {base}")

    _check_block(program.body, declared)


def _check_unique_group(names: list[str]) -> None:
    seen: set[str] = set()
    for name in names:
        base = parse_decl(name).name
        if base in seen:
            raise CheckError(f"duplicate variable: {base}")
        seen.add(base)


def _check_block(block: Block, declared: dict[str, object]) -> None:
    for stmt in block.stmts:
        _check_stmt(stmt, declared)


def _check_stmt(stmt: Stmt, declared: dict[str, object]) -> None:
    if isinstance(stmt, Assign):
        _check_place(stmt.target, declared)
        _check_expr(stmt.expr, declared)
        if _expr_mentions_target(stmt.expr, stmt.target):
            raise CheckError("assignment expression mentions target")
        return
    if isinstance(stmt, Swap):
        _check_place(stmt.left, declared)
        _check_place(stmt.right, declared)
        if _same_place(stmt.left, stmt.right):
            raise CheckError(f"swap uses the same variable twice: {_place_name(stmt.left)}")
        if _array_swap_may_alias(stmt.left, stmt.right):
            raise CheckError(f"swap may alias array elements: {_place_base_name(stmt.left)}")
        return
    if isinstance(stmt, If):
        _check_expr(stmt.test, declared)
        _check_expr(stmt.assertion, declared)
        _check_block(stmt.then_block, declared)
        _check_block(stmt.else_block, declared)
        return
    if isinstance(stmt, Loop):
        _check_expr(stmt.entry_guard, declared)
        _check_expr(stmt.exit_guard, declared)
        _check_block(stmt.do_block, declared)
        _check_block(stmt.loop_block, declared)
        return
    if isinstance(stmt, Rif):
        _check_expr(stmt.test, declared)
        _check_expr(stmt.assertion, declared)
        _check_block(stmt.body, declared)
        return
    if isinstance(stmt, Push):
        _check_scalar(stmt.var, declared)
        decl = declared.get(stmt.stack)
        if decl is None or not getattr(decl, "is_stack", False):
            raise CheckError(f"push target is not a stack: {stmt.stack}")
        return
    if isinstance(stmt, Pop):
        _check_scalar(stmt.var, declared)
        decl = declared.get(stmt.stack)
        if decl is None or not getattr(decl, "is_stack", False):
            raise CheckError(f"pop source is not a stack: {stmt.stack}")
        return
    raise TypeError(f"unknown statement: {stmt!r}")


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


def _same_place(left: str | Place, right: str | Place) -> bool:
    return left == right


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
