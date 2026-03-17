from __future__ import annotations

from pyrev_fl.ast import ArrayRef, Assign, Binary, Block, Const, Expr, Place, If, Loop, Program, Rif, Stmt, Swap, Var
from pyrev_fl.check import check_program
from pyrev_fl.interface import build_layout
from pyrev_fl.interpreter import EvalError, _eval_binary, _initialize_store, _validate_final_store
from pyrev_fl.invert import invert_block

Store = dict[str, int]


def trace_program(program: Program, inputs: list[int]) -> list[dict[str, object]]:
    layout = build_layout(program.inputs, program.outputs, program.temps)
    if len(inputs) != len(layout.inputs):
        raise EvalError(f"arity mismatch: expected {len(layout.inputs)}, got {len(inputs)}")
    check_program(program)
    store = _initialize_store(program, inputs, layout)
    events: list[dict[str, object]] = []
    _trace_block(program.body, store, events, direction="forward", layout=layout)
    _validate_final_store(store, layout)
    return events


def _trace_block(block: Block, store: Store, events: list[dict[str, object]], direction: str, layout) -> None:
    for stmt in block.stmts:
        _trace_stmt(stmt, store, events, direction, layout)


def _trace_stmt(stmt: Stmt, store: Store, events: list[dict[str, object]], direction: str, layout) -> None:
    before = dict(store)
    event: dict[str, object] = {"direction": direction, "kind": _stmt_kind(stmt)}

    if isinstance(stmt, Assign):
        rhs = _eval_expr(stmt.expr, store, layout)
        current = _read_place(stmt.target, store, layout)
        if stmt.op.value == "+=":
            _write_place(stmt.target, current + rhs, store, layout)
        elif stmt.op.value == "-=":
            _write_place(stmt.target, current - rhs, store, layout)
        else:
            _write_place(stmt.target, current ^ rhs, store, layout)
        event["target"] = _place_label(stmt.target)
    elif isinstance(stmt, Swap):
        left = _read_place(stmt.left, store, layout)
        right = _read_place(stmt.right, store, layout)
        _write_place(stmt.left, right, store, layout)
        _write_place(stmt.right, left, store, layout)
        event["left"] = _place_label(stmt.left)
        event["right"] = _place_label(stmt.right)
    elif isinstance(stmt, If):
        cond = _truthy(_eval_expr(stmt.test, store, layout))
        event["branch"] = "then" if cond else "else"
        _trace_block(stmt.then_block if cond else stmt.else_block, store, events, direction, layout)
        assertion = _truthy(_eval_expr(stmt.assertion, store, layout))
        if cond and not assertion:
            raise EvalError("if assertion must hold after then branch")
        if not cond and assertion:
            raise EvalError("if assertion must be false after else branch")
    elif isinstance(stmt, Loop):
        if not _truthy(_eval_expr(stmt.entry_guard, store, layout)):
            raise EvalError("loop entry guard must hold before entering")
        iterations = 0
        _trace_block(stmt.do_block, store, events, direction, layout)
        iterations += 1
        while not _truthy(_eval_expr(stmt.exit_guard, store, layout)):
            _trace_block(stmt.loop_block, store, events, direction, layout)
            if _truthy(_eval_expr(stmt.entry_guard, store, layout)):
                raise EvalError("loop entry guard must be false between iterations")
            _trace_block(stmt.do_block, store, events, direction, layout)
            iterations += 1
        event["iterations"] = iterations
    elif isinstance(stmt, Rif):
        cond = _truthy(_eval_expr(stmt.test, store, layout))
        event["branch"] = "reverse" if cond else "forward"
        if cond:
            _trace_block(invert_block(stmt.body), store, events, direction="backward", layout=layout)
            if not _truthy(_eval_expr(stmt.assertion, store, layout)):
                raise EvalError("rif assertion must hold after reverse branch")
        else:
            _trace_block(stmt.body, store, events, direction="forward", layout=layout)
            if _truthy(_eval_expr(stmt.assertion, store, layout)):
                raise EvalError("rif assertion must be false after forward branch")
    else:
        raise TypeError(f"unknown statement: {stmt!r}")

    event["store_diff"] = _store_diff(before, store)
    events.append(event)


def _stmt_kind(stmt: Stmt) -> str:
    if isinstance(stmt, Assign):
        return "assign"
    if isinstance(stmt, Swap):
        return "swap"
    if isinstance(stmt, If):
        return "if"
    if isinstance(stmt, Loop):
        return "loop"
    if isinstance(stmt, Rif):
        return "rif"
    raise TypeError(f"unknown statement: {stmt!r}")


def _eval_expr(expr: Expr, store: Store, layout) -> int:
    if isinstance(expr, Const):
        return expr.value
    if isinstance(expr, Var):
        return _read_place(expr, store, layout)
    if isinstance(expr, ArrayRef):
        return _read_place(expr, store, layout)
    if isinstance(expr, Binary):
        return _eval_binary(expr.op, _eval_expr(expr.left, store, layout), _eval_expr(expr.right, store, layout))
    raise TypeError(f"unknown expression: {expr!r}")


def _truthy(value: int) -> bool:
    return value != 0


def _store_diff(before: Store, after: Store) -> dict[str, dict[str, int]]:
    diff: dict[str, dict[str, int]] = {}
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            diff[key] = {"before": before.get(key, 0), "after": after.get(key, 0)}
    return diff


def _read_place(place: str | Place, store: Store, layout) -> int:
    if isinstance(place, str):
        return store[place]
    if isinstance(place, Var):
        return store[place.name]
    if isinstance(place, ArrayRef):
        index = _eval_expr(place.index, store, layout)
        if index < 0 or layout.declarations[place.name].size is None or index >= layout.declarations[place.name].size:
            raise EvalError(f"array index out of bounds: {place.name}[{index}]")
        return store[f"{place.name}[{index}]"]
    raise TypeError(f"unknown place: {place!r}")


def _write_place(place: str | Place, value: int, store: Store, layout) -> None:
    if isinstance(place, str):
        store[place] = value
        return
    if isinstance(place, Var):
        store[place.name] = value
        return
    if isinstance(place, ArrayRef):
        index = _eval_expr(place.index, store, layout)
        if index < 0 or layout.declarations[place.name].size is None or index >= layout.declarations[place.name].size:
            raise EvalError(f"array index out of bounds: {place.name}[{index}]")
        store[f"{place.name}[{index}]"] = value
        return
    raise TypeError(f"unknown place: {place!r}")


def _place_label(place: str | Place) -> str:
    if isinstance(place, str):
        return place
    if isinstance(place, Var):
        return place.name
    if isinstance(place, ArrayRef):
        return f"{place.name}[...]"
    raise TypeError(f"unknown place: {place!r}")
