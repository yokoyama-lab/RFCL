from __future__ import annotations

from pyrev_fl.ast import ArrayRef, BinOp, Binary, Const, Expr, Place, UpdateOp, Var
from pyrev_fl.interface import InterfaceLayout, build_layout
from pyrev_fl.rl_ast import Assign, Assignment, Block, Direction, Exit, FiFrom, FromEntry, FromLabel, Goto, IfGoto, Jump, Program, RFromLabel, RGoto, Swap
from pyrev_fl.rl_check import CheckError, check_program

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

    block_map = {block.label: block for block in program.blocks}
    current = _entry_label(program)
    previous = "start"
    direction = Direction.FORWARD
    store = _initialize_store(program, inputs, layout)

    for _ in range(100000):
        if current == "halt" and direction is Direction.FORWARD:
            _validate_final_store(store, layout)
            return store
        block = block_map.get(current)
        if block is None:
            raise EvalError(f"unknown block: {current}")
        previous, current, direction = _step(block, previous, direction, store, layout)

    raise EvalError("step limit exceeded")


def _initialize_store(program: Program, inputs: list[int], layout: InterfaceLayout) -> Store:
    store: Store = {}
    for name, value in zip(layout.inputs, inputs, strict=True):
        store[name] = value
    for name in layout.inputs + layout.outputs + layout.temps:
        store.setdefault(name, 0)
    return store


def _entry_label(program: Program) -> str:
    for block in program.blocks:
        if isinstance(block.from_, FromEntry):
            return block.label
    raise EvalError("missing entry block")


def _step(block: Block, previous: str, direction: Direction, store: Store, layout: InterfaceLayout) -> tuple[str, str, Direction]:
    if direction is Direction.FORWARD:
        expected_previous, _ = _eval_from(block.from_, store, layout)
        if expected_previous != previous:
            raise EvalError(f"from mismatch at {block.label}: expected {expected_previous}, got {previous}")
        for assign in block.assigns:
            _eval_assignment(assign, store, layout)
        next_label, next_dir = _eval_jump(block.jump, store, layout)
        return block.label, next_label, next_dir

    expected_previous, _ = _eval_jump(block.jump, store, layout)
    if expected_previous != previous:
        raise EvalError(f"jump mismatch at {block.label}: expected {expected_previous}, got {previous}")
    for assign in reversed(block.assigns):
        _eval_assignment_inverse(assign, store, layout)
    next_label, next_dir = _eval_from(block.from_, store, layout)
    return block.label, next_label, next_dir


def _eval_from(from_, store: Store, layout: InterfaceLayout) -> tuple[str, Direction]:
    if isinstance(from_, FromEntry):
        return "start", Direction.BACKWARD
    if isinstance(from_, FromLabel):
        return from_.label, Direction.BACKWARD
    if isinstance(from_, RFromLabel):
        return from_.label, Direction.FORWARD
    if isinstance(from_, FiFrom):
        cond = _truthy(_eval_expr(from_.expr, store, layout))
        return (from_.true_label if cond else from_.false_label, Direction.BACKWARD)
    raise TypeError(f"unknown from construct: {from_!r}")


def _eval_jump(jump: Jump, store: Store, layout: InterfaceLayout) -> tuple[str, Direction]:
    if isinstance(jump, Goto):
        return jump.label, Direction.FORWARD
    if isinstance(jump, RGoto):
        return jump.label, Direction.BACKWARD
    if isinstance(jump, IfGoto):
        cond = _truthy(_eval_expr(jump.expr, store, layout))
        return (jump.true_label if cond else jump.false_label, Direction.FORWARD)
    if isinstance(jump, Exit):
        return "halt", Direction.FORWARD
    raise TypeError(f"unknown jump construct: {jump!r}")


def _eval_assignment(assign: Assignment, store: Store, layout: InterfaceLayout) -> None:
    if isinstance(assign, Assign):
        rhs = _eval_expr(assign.expr, store, layout)
        current = _read_place(assign.target, store, layout)
        if assign.op is UpdateOp.ADD:
            _write_place(assign.target, current + rhs, store, layout)
        elif assign.op is UpdateOp.SUB:
            _write_place(assign.target, current - rhs, store, layout)
        else:
            _write_place(assign.target, current ^ rhs, store, layout)
        return
    if isinstance(assign, Swap):
        left = _read_place(assign.left, store, layout)
        right = _read_place(assign.right, store, layout)
        _write_place(assign.left, right, store, layout)
        _write_place(assign.right, left, store, layout)
        return
    raise TypeError(f"unknown assignment: {assign!r}")


def _eval_assignment_inverse(assign: Assignment, store: Store, layout: InterfaceLayout) -> None:
    if isinstance(assign, Assign):
        inverse = {
            UpdateOp.ADD: UpdateOp.SUB,
            UpdateOp.SUB: UpdateOp.ADD,
            UpdateOp.XOR: UpdateOp.XOR,
        }[assign.op]
        _eval_assignment(Assign(assign.target, inverse, assign.expr), store, layout)
        return
    _eval_assignment(assign, store, layout)


def _eval_expr(expr: Expr, store: Store, layout: InterfaceLayout) -> int:
    if isinstance(expr, Const):
        return expr.value
    if isinstance(expr, Var):
        return _read_scalar(expr.name, store, layout)
    if isinstance(expr, ArrayRef):
        return _read_cell(store, _resolve_array_cell(expr.name, expr.index, store, layout))
    if isinstance(expr, Binary):
        left = _eval_expr(expr.left, store, layout)
        right = _eval_expr(expr.right, store, layout)
        if expr.op is BinOp.ADD:
            return left + right
        if expr.op is BinOp.SUB:
            return left - right
        if expr.op is BinOp.MUL:
            return left * right
        if expr.op is BinOp.DIV:
            if right == 0:
                raise EvalError("division by zero")
            return left // right
        if expr.op is BinOp.EQ:
            return int(left == right)
        if expr.op is BinOp.NE:
            return int(left != right)
        if expr.op is BinOp.LT:
            return int(left < right)
        if expr.op is BinOp.LE:
            return int(left <= right)
        if expr.op is BinOp.GT:
            return int(left > right)
        if expr.op is BinOp.GE:
            return int(left >= right)
    raise TypeError(f"unknown expression: {expr!r}")


def _read_scalar(name: str, store: Store, layout: InterfaceLayout) -> int:
    decl = layout.declarations.get(name)
    if decl is not None and decl.is_array:
        raise EvalError(f"array variable requires index: {name}")
    return _read_cell(store, name)


def _read_place(place: str | Place, store: Store, layout: InterfaceLayout) -> int:
    if isinstance(place, str):
        return _read_scalar(place, store, layout)
    if isinstance(place, Var):
        return _read_scalar(place.name, store, layout)
    if isinstance(place, ArrayRef):
        return _read_cell(store, _resolve_array_cell(place.name, place.index, store, layout))
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
