from __future__ import annotations

from pyrev_fl.ast import ArrayRef, Assign, BinOp, Binary, Block, Const, Expr, Place, If, Loop, Pop, Program, Push, Rif, Stmt, Swap, UpdateOp, Var
from pyrev_fl.check import CheckError, check_program
from pyrev_fl.interface import InterfaceLayout, build_layout
from pyrev_fl.invert import invert_block

Store = dict[str, int]


class EvalError(ValueError):
    pass


def run_program(program: Program, inputs: list[int]) -> Store:
    layout = build_layout(program.inputs, program.outputs, program.temps)
    if len(inputs) != len(layout.inputs):
        raise EvalError(f"arity mismatch: expected {len(layout.inputs)}, got {len(inputs)}")

    try:
        check_program(program)
    except CheckError as exc:
        raise EvalError(str(exc)) from exc

    store = _initialize_store(program, inputs, layout)
    eval_block(program.body, store, layout)
    _validate_final_store(store, layout)
    return store


def eval_block(block: Block, store: Store, layout: InterfaceLayout) -> None:
    for stmt in block.stmts:
        _eval_stmt(stmt, store, layout)


def _initialize_store(program: Program, inputs: list[int], layout: InterfaceLayout) -> Store:
    store: Store = {}
    for name, value in zip(layout.inputs, inputs, strict=True):
        store[name] = value
    for name in layout.inputs + layout.outputs + layout.temps:
        store.setdefault(name, 0)
    return store


def _eval_stmt(stmt: Stmt, store: Store, layout: InterfaceLayout) -> None:
    if isinstance(stmt, Assign):
        rhs = _eval_expr(stmt.expr, store, layout)
        current = _read_place(stmt.target, store, layout)
        if stmt.op is UpdateOp.ADD:
            _write_place(stmt.target, current + rhs, store, layout)
        elif stmt.op is UpdateOp.SUB:
            _write_place(stmt.target, current - rhs, store, layout)
        elif stmt.op is UpdateOp.XOR:
            _write_place(stmt.target, current ^ rhs, store, layout)
        else:
            raise TypeError(f"unknown update op: {stmt.op!r}")
        return

    if isinstance(stmt, Swap):
        left = _read_place(stmt.left, store, layout)
        right = _read_place(stmt.right, store, layout)
        _write_place(stmt.left, right, store, layout)
        _write_place(stmt.right, left, store, layout)
        return

    if isinstance(stmt, If):
        cond = _truthy(_eval_expr(stmt.test, store, layout))
        if cond:
            eval_block(stmt.then_block, store, layout)
            if not _truthy(_eval_expr(stmt.assertion, store, layout)):
                raise EvalError("if assertion must hold after then branch")
        else:
            eval_block(stmt.else_block, store, layout)
            if _truthy(_eval_expr(stmt.assertion, store, layout)):
                raise EvalError("if assertion must be false after else branch")
        return

    if isinstance(stmt, Loop):
        if not _truthy(_eval_expr(stmt.entry_guard, store, layout)):
            raise EvalError("loop entry guard must hold before entering")

        eval_block(stmt.do_block, store, layout)
        while not _truthy(_eval_expr(stmt.exit_guard, store, layout)):
            eval_block(stmt.loop_block, store, layout)
            if _truthy(_eval_expr(stmt.entry_guard, store, layout)):
                raise EvalError("loop entry guard must be false between iterations")
            eval_block(stmt.do_block, store, layout)
        return

    if isinstance(stmt, Rif):
        cond = _truthy(_eval_expr(stmt.test, store, layout))
        if cond:
            eval_block(invert_block(stmt.body), store, layout)
            if not _truthy(_eval_expr(stmt.assertion, store, layout)):
                raise EvalError("rif assertion must hold after reverse branch")
        else:
            eval_block(stmt.body, store, layout)
            if _truthy(_eval_expr(stmt.assertion, store, layout)):
                raise EvalError("rif assertion must be false after forward branch")
        return

    if isinstance(stmt, Push):
        val = store.get(stmt.var, 0)
        stack_key = f"__stack_{stmt.stack}"
        if stack_key not in store:
            store[stack_key] = []  # type: ignore[assignment]
        store[stack_key].append(val)  # type: ignore[union-attr]
        store[stmt.var] = 0
        return

    if isinstance(stmt, Pop):
        if store.get(stmt.var, 0) != 0:
            raise EvalError(f"pop target must be zero: {stmt.var}={store[stmt.var]}")
        stack_key = f"__stack_{stmt.stack}"
        stack = store.get(stack_key, [])
        if not stack:
            raise EvalError(f"pop from empty stack: {stmt.stack}")
        store[stmt.var] = stack.pop()  # type: ignore[union-attr]
        return

    raise TypeError(f"unknown statement: {stmt!r}")


def _eval_expr(expr: Expr, store: Store, layout: InterfaceLayout) -> int:
    if isinstance(expr, Const):
        return expr.value
    if isinstance(expr, Var):
        return _read_scalar(expr.name, store, layout)
    if isinstance(expr, ArrayRef):
        return _read_array(expr, store, layout)
    if isinstance(expr, Binary):
        left = _eval_expr(expr.left, store, layout)
        right = _eval_expr(expr.right, store, layout)
        return _eval_binary(expr.op, left, right)
    raise TypeError(f"unknown expression: {expr!r}")


def _eval_binary(op: BinOp, left: int, right: int) -> int:
    if op is BinOp.ADD:
        return left + right
    if op is BinOp.SUB:
        return left - right
    if op is BinOp.MUL:
        return left * right
    if op is BinOp.DIV:
        if right == 0:
            raise EvalError("division by zero")
        return left // right
    if op is BinOp.EQ:
        return int(left == right)
    if op is BinOp.NE:
        return int(left != right)
    if op is BinOp.LT:
        return int(left < right)
    if op is BinOp.LE:
        return int(left <= right)
    if op is BinOp.GT:
        return int(left > right)
    if op is BinOp.GE:
        return int(left >= right)
    raise TypeError(f"unknown binary op: {op!r}")


def _read_scalar(store_name: str, store: Store, layout: InterfaceLayout) -> int:
    decl = layout.declarations.get(store_name)
    if decl is not None and decl.is_array:
        raise EvalError(f"array variable requires index: {store_name}")
    return _read_cell(store, store_name)


def _read_array(expr: ArrayRef, store: Store, layout: InterfaceLayout) -> int:
    return _read_cell(store, _resolve_array_cell(expr.name, expr.index, store, layout))


def _read_place(place: str | Place, store: Store, layout: InterfaceLayout) -> int:
    if isinstance(place, str):
        return _read_scalar(place, store, layout)
    if isinstance(place, Var):
        return _read_scalar(place.name, store, layout)
    if isinstance(place, ArrayRef):
        return _read_array(place, store, layout)
    raise TypeError(f"unknown place: {place!r}")


def _write_place(place: str | Place, value: int, store: Store, layout: InterfaceLayout) -> None:
    if isinstance(place, str):
        decl = layout.declarations.get(place)
        if decl is not None and decl.is_array:
            raise EvalError(f"array variable requires index: {place}")
        store[place] = value
        return
    if isinstance(place, Var):
        _write_place(place.name, value, store, layout)
        return
    if isinstance(place, ArrayRef):
        store[_resolve_array_cell(place.name, place.index, store, layout)] = value
        return
    raise TypeError(f"unknown place: {place!r}")


def _resolve_array_cell(name: str, index: Expr, store: Store, layout: InterfaceLayout) -> str:
    decl = layout.declarations.get(name)
    if decl is None:
        raise EvalError(f"unknown variable: {name}")
    if not decl.is_array:
        raise EvalError(f"scalar variable does not support indexing: {name}")
    index_value = _eval_expr(index, store, layout)
    if index_value < 0 or decl.size is None or index_value >= decl.size:
        raise EvalError(f"array index out of bounds: {name}[{index_value}]")
    return f"{name}[{index_value}]"


def _read_cell(store: Store, name: str) -> int:
    try:
        return store[name]
    except KeyError as exc:
        raise EvalError(f"unknown variable: {name}") from exc


def _truthy(value: int) -> bool:
    return value != 0


def _validate_final_store(store: Store, layout: InterfaceLayout) -> None:
    must_be_zero = set(layout.temps) | (set(layout.inputs) - set(layout.outputs))
    for name in sorted(must_be_zero):
        if store.get(name, 0) != 0:
            raise EvalError(f"variable must be zero at exit: {name}={store[name]}")
