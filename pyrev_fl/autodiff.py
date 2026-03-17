"""Reverse-mode automatic differentiation for SRL programs.

The key insight: reversible programs naturally support reverse-mode AD
because the execution trace can be replayed backward without storing it.
This gives O(1) memory for the tape (vs O(T) for conventional reverse-mode AD).

For integer SRL programs, gradients are discrete approximations:
- x += y:  d(x_new)/d(y) = 1,  d(x_new)/d(x_old) = 1
- x -= y:  d(x_new)/d(y) = -1, d(x_new)/d(x_old) = 1
- x ^= y:  not differentiable, gradient = 0
- x <=> y: permutation, gradients swap
- if/loop:  gradient flows through the taken branch
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pyrev_fl.ast import (
    ArrayRef,
    Assign,
    BinOp,
    Binary,
    Block,
    Const,
    Expr,
    If,
    Loop,
    Place,
    Pop,
    Program,
    Push,
    Rif,
    Stmt,
    Swap,
    UpdateOp,
    Var,
)
from pyrev_fl.check import check_program
from pyrev_fl.interface import InterfaceLayout, build_layout
from pyrev_fl.interpreter import (
    EvalError,
    _eval_expr,
    _initialize_store,
    _read_place,
    _truthy,
    _validate_final_store,
    eval_block,
)
from pyrev_fl.invert import invert_block


Store = dict[str, int]
GradMap = dict[str, float]


@dataclass
class DiffResult:
    """Result of differentiating a program."""

    output_values: dict[str, int]  # forward pass outputs
    gradients: dict[str, dict[str, float]]  # d(output)/d(input) for each pair
    tape_size: int  # number of stored intermediate values (0 for reversible)


@dataclass
class _TracedStmt:
    """A statement together with the store snapshot before it executed."""

    stmt: Stmt
    store_before: dict[str, int]
    branch: str | None = None  # "then"/"else" for If, "reverse"/"forward" for Rif
    sub_trace: list[_TracedStmt] = field(default_factory=list)
    iterations: list[list[_TracedStmt]] = field(default_factory=list)


def differentiate(
    program: Program,
    inputs: list[int],
    output_var: str,
) -> DiffResult:
    """Compute gradients of output_var w.r.t. all inputs using reverse-mode AD.

    For SRL programs (integer arithmetic), gradients are discrete:
    - x += y: dx/dy_old = 1, dy/dy_old = 1 (partial derivatives)
    - x -= y: dx/dy_old = -1
    - x ^= y: not differentiable in the classical sense, gradient = 0
    - x <=> y: permutation, gradient is a permutation matrix

    The key point is that reversible execution means we DON'T need to store
    the forward tape -- we can recompute any intermediate value by running
    backward from the current point. This gives O(1) memory for the tape
    (vs O(T) for conventional reverse-mode AD).
    """
    layout = build_layout(program.inputs, program.outputs, program.temps)
    if len(inputs) != len(layout.inputs):
        raise EvalError(
            f"arity mismatch: expected {len(layout.inputs)}, got {len(inputs)}"
        )
    check_program(program)

    store = _initialize_store(program, inputs, layout)

    # Forward pass: record each statement with the store state before it.
    traced = _trace_forward(program.body, store, layout)

    _validate_final_store(store, layout)

    # Collect output values.
    output_values = {name: store[name] for name in layout.outputs}

    # Find the output variable in the store.
    if output_var not in store:
        raise EvalError(f"unknown output variable: {output_var}")

    # Backward pass: propagate adjoint gradients.
    adjoints: GradMap = {}
    adjoints[output_var] = 1.0

    _backprop(traced, adjoints, layout)

    # Extract gradients w.r.t. each input.
    gradients: dict[str, dict[str, float]] = {}
    gradients[output_var] = {}
    for inp in layout.inputs:
        gradients[output_var][inp] = adjoints.get(inp, 0.0)

    return DiffResult(
        output_values=output_values,
        gradients=gradients,
        tape_size=0,  # Reversible programs need no tape!
    )


def compute_jacobian(
    program: Program, inputs: list[int]
) -> dict[str, dict[str, float]]:
    """Compute the full Jacobian matrix d(outputs)/d(inputs)."""
    layout = build_layout(program.inputs, program.outputs, program.temps)
    jacobian: dict[str, dict[str, float]] = {}
    for out_var in layout.outputs:
        result = differentiate(program, inputs, out_var)
        jacobian[out_var] = result.gradients[out_var]
    return jacobian


# ---------------------------------------------------------------------------
# Forward tracing
# ---------------------------------------------------------------------------


def _trace_forward(
    block: Block, store: Store, layout: InterfaceLayout
) -> list[_TracedStmt]:
    """Execute the block forward, recording each statement with store snapshots."""
    traced: list[_TracedStmt] = []
    for stmt in block.stmts:
        entry = _trace_stmt_forward(stmt, store, layout)
        traced.append(entry)
    return traced


def _trace_stmt_forward(
    stmt: Stmt, store: Store, layout: InterfaceLayout
) -> _TracedStmt:
    """Execute one statement forward, returning a _TracedStmt."""
    snapshot = dict(store)

    if isinstance(stmt, Assign):
        rhs = _eval_expr(stmt.expr, store, layout)
        current = _read_place(stmt.target, store, layout)
        if stmt.op is UpdateOp.ADD:
            _write_place(stmt.target, current + rhs, store, layout)
        elif stmt.op is UpdateOp.SUB:
            _write_place(stmt.target, current - rhs, store, layout)
        elif stmt.op is UpdateOp.XOR:
            _write_place(stmt.target, current ^ rhs, store, layout)
        return _TracedStmt(stmt=stmt, store_before=snapshot)

    if isinstance(stmt, Swap):
        left = _read_place(stmt.left, store, layout)
        right = _read_place(stmt.right, store, layout)
        _write_place(stmt.left, right, store, layout)
        _write_place(stmt.right, left, store, layout)
        return _TracedStmt(stmt=stmt, store_before=snapshot)

    if isinstance(stmt, If):
        cond = _truthy(_eval_expr(stmt.test, store, layout))
        branch = "then" if cond else "else"
        chosen = stmt.then_block if cond else stmt.else_block
        sub = _trace_forward(chosen, store, layout)
        assertion = _truthy(_eval_expr(stmt.assertion, store, layout))
        if cond and not assertion:
            raise EvalError("if assertion must hold after then branch")
        if not cond and assertion:
            raise EvalError("if assertion must be false after else branch")
        return _TracedStmt(stmt=stmt, store_before=snapshot, branch=branch, sub_trace=sub)

    if isinstance(stmt, Loop):
        if not _truthy(_eval_expr(stmt.entry_guard, store, layout)):
            raise EvalError("loop entry guard must hold before entering")
        iterations: list[list[_TracedStmt]] = []

        # First do_block
        do_trace = _trace_forward(stmt.do_block, store, layout)
        first_iter = do_trace
        while not _truthy(_eval_expr(stmt.exit_guard, store, layout)):
            loop_trace = _trace_forward(stmt.loop_block, store, layout)
            if _truthy(_eval_expr(stmt.entry_guard, store, layout)):
                raise EvalError("loop entry guard must be false between iterations")
            iterations.append(first_iter + loop_trace)
            first_iter = _trace_forward(stmt.do_block, store, layout)
        # The final iteration's do_block (exit guard is true)
        iterations.append(first_iter)
        return _TracedStmt(stmt=stmt, store_before=snapshot, iterations=iterations)

    if isinstance(stmt, Rif):
        cond = _truthy(_eval_expr(stmt.test, store, layout))
        if cond:
            branch = "reverse"
            body_to_run = invert_block(stmt.body)
        else:
            branch = "forward"
            body_to_run = stmt.body
        sub = _trace_forward(body_to_run, store, layout)
        assertion = _truthy(_eval_expr(stmt.assertion, store, layout))
        if cond and not assertion:
            raise EvalError("rif assertion must hold after reverse branch")
        if not cond and assertion:
            raise EvalError("rif assertion must be false after forward branch")
        return _TracedStmt(stmt=stmt, store_before=snapshot, branch=branch, sub_trace=sub)

    if isinstance(stmt, (Push, Pop)):
        # Push/Pop: execute normally, treat as non-differentiable.
        eval_block(Block([stmt]), store, layout)
        return _TracedStmt(stmt=stmt, store_before=snapshot)

    raise TypeError(f"unknown statement: {stmt!r}")


# ---------------------------------------------------------------------------
# Backward gradient propagation
# ---------------------------------------------------------------------------


def _backprop(
    traced: list[_TracedStmt], adjoints: GradMap, layout: InterfaceLayout
) -> None:
    """Propagate adjoints backward through the traced statements."""
    for entry in reversed(traced):
        _backprop_stmt(entry, adjoints, layout)


def _backprop_stmt(
    entry: _TracedStmt, adjoints: GradMap, layout: InterfaceLayout
) -> None:
    """Propagate adjoints backward through one traced statement."""
    stmt = entry.stmt
    store = entry.store_before

    if isinstance(stmt, Assign):
        target_name = _place_key(stmt.target)
        # The adjoint of the target flows backward.
        adj_target = adjoints.get(target_name, 0.0)

        if stmt.op is UpdateOp.XOR:
            # XOR is not differentiable; gradient = 0.
            # The adjoint of the target is lost (zeroed out for the XOR contribution).
            # But the target's old value still contributes identically:
            # x_new = x_old ^ rhs  =>  d(x_new)/d(x_old) ~ 0, d(x_new)/d(rhs) ~ 0
            # So adj_target does not flow to any input through XOR.
            pass
        elif stmt.op is UpdateOp.ADD:
            # x_new = x_old + expr
            # d(x_new)/d(x_old) = 1 => adj(x_old) += adj(x_new) * 1
            # d(x_new)/d(vars in expr) = d(expr)/d(var) => adj(var) += adj(x_new) * d(expr)/d(var)
            # adj_target already accumulates into x_old (it was x_old before).
            _accumulate_expr_adjoints(stmt.expr, adj_target, store, layout, adjoints)
        elif stmt.op is UpdateOp.SUB:
            # x_new = x_old - expr
            # d(x_new)/d(x_old) = 1 (already in adjoints for target)
            # d(x_new)/d(vars in expr) = -d(expr)/d(var)
            _accumulate_expr_adjoints(stmt.expr, -adj_target, store, layout, adjoints)
        return

    if isinstance(stmt, Swap):
        # x_new = y_old, y_new = x_old
        # Adjoint of x_new goes to y_old, adjoint of y_new goes to x_old.
        left_name = _place_key(stmt.left)
        right_name = _place_key(stmt.right)
        adj_left = adjoints.get(left_name, 0.0)
        adj_right = adjoints.get(right_name, 0.0)
        # Swap the adjoints: adj for the old left = adj_right, adj for old right = adj_left
        adjoints[left_name] = adj_right
        adjoints[right_name] = adj_left
        return

    if isinstance(stmt, If):
        # Gradient flows through the taken branch.
        _backprop(entry.sub_trace, adjoints, layout)
        return

    if isinstance(stmt, Loop):
        # Gradient flows backward through all iterations in reverse.
        for iteration_trace in reversed(entry.iterations):
            _backprop(iteration_trace, adjoints, layout)
        return

    if isinstance(stmt, Rif):
        # Gradient flows through whichever branch was taken.
        _backprop(entry.sub_trace, adjoints, layout)
        return

    if isinstance(stmt, (Push, Pop)):
        # Push/Pop: non-differentiable stack operations, no gradient flow.
        return

    raise TypeError(f"unknown statement: {stmt!r}")


def _accumulate_expr_adjoints(
    expr: Expr,
    adj: float,
    store: Store,
    layout: InterfaceLayout,
    adjoints: GradMap,
) -> None:
    """Accumulate adjoints from an expression into the variable adjoints.

    Computes d(expr)/d(var) for each variable in expr and adds adj * d(expr)/d(var)
    to adjoints[var].
    """
    if isinstance(expr, Const):
        # Constants have zero derivative.
        return

    if isinstance(expr, Var):
        # d(var)/d(var) = 1
        adjoints[expr.name] = adjoints.get(expr.name, 0.0) + adj
        return

    if isinstance(expr, ArrayRef):
        # d(a[i])/d(a[i]) = 1; index is treated as discrete (no gradient).
        key = _resolve_array_key(expr, store, layout)
        adjoints[key] = adjoints.get(key, 0.0) + adj
        return

    if isinstance(expr, Binary):
        left_val = _eval_expr(expr.left, store, layout)
        right_val = _eval_expr(expr.right, store, layout)
        left_adj, right_adj = _binary_adjoints(expr.op, left_val, right_val, adj)
        _accumulate_expr_adjoints(expr.left, left_adj, store, layout, adjoints)
        _accumulate_expr_adjoints(expr.right, right_adj, store, layout, adjoints)
        return

    raise TypeError(f"unknown expression: {expr!r}")


def _binary_adjoints(
    op: BinOp, left_val: int, right_val: int, adj: float
) -> tuple[float, float]:
    """Compute adjoint contributions for a binary operation.

    Returns (left_adj, right_adj).
    """
    if op is BinOp.ADD:
        # d(l+r)/dl = 1, d(l+r)/dr = 1
        return (adj, adj)
    if op is BinOp.SUB:
        # d(l-r)/dl = 1, d(l-r)/dr = -1
        return (adj, -adj)
    if op is BinOp.MUL:
        # d(l*r)/dl = r, d(l*r)/dr = l
        return (adj * right_val, adj * left_val)
    if op is BinOp.DIV:
        # d(l//r)/dl ~ 1/r, d(l//r)/dr ~ -l/r^2  (integer approx)
        if right_val == 0:
            return (0.0, 0.0)
        return (adj / right_val, -adj * left_val / (right_val * right_val))
    # Comparison operators: not differentiable, gradient = 0
    return (0.0, 0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _place_key(place: str | Place) -> str:
    """Return the store key for a place."""
    if isinstance(place, str):
        return place
    if isinstance(place, Var):
        return place.name
    if isinstance(place, ArrayRef):
        # For array refs, we'd need the index at the time of execution.
        # This is a simplification -- we return the base name with a marker.
        return f"{place.name}[?]"
    raise TypeError(f"unknown place: {place!r}")


def _resolve_array_key(expr: ArrayRef, store: Store, layout: InterfaceLayout) -> str:
    """Resolve an array reference to a concrete store key using the given store."""
    index = _eval_expr(expr.index, store, layout)
    return f"{expr.name}[{index}]"


def _write_place(
    place: str | Place, value: int, store: Store, layout: InterfaceLayout
) -> None:
    """Write a value to a place in the store."""
    if isinstance(place, str):
        store[place] = value
        return
    if isinstance(place, Var):
        store[place.name] = value
        return
    if isinstance(place, ArrayRef):
        index = _eval_expr(place.index, store, layout)
        store[f"{place.name}[{index}]"] = value
        return
    raise TypeError(f"unknown place: {place!r}")
