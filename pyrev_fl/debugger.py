from __future__ import annotations

from dataclasses import dataclass

from pyrev_fl.ast import (
    Assign,
    Block,
    If,
    Loop,
    Pop,
    Program,
    Push,
    Rif,
    Stmt,
    Swap,
)
from pyrev_fl.check import check_program
from pyrev_fl.interface import build_layout
from pyrev_fl.interpreter import (
    EvalError,
    Store,
    _eval_expr,
    _eval_stmt,
    _initialize_store,
)
from pyrev_fl.invert import invert_block
from pyrev_fl.pretty import _render_expr, _render_place


@dataclass
class DebugState:
    store: dict[str, int]
    position: int
    stmt_text: str


# -- internal pseudo-statements for deferred control flow checks -----------

@dataclass(frozen=True)
class _IfAssertionCheck(Stmt):
    original: If
    took_then: bool


@dataclass(frozen=True)
class _LoopContinuation(Stmt):
    original: Loop
    first: bool


@dataclass(frozen=True)
class _LoopEntryCheck(Stmt):
    original: Loop


@dataclass(frozen=True)
class _RifAssertionCheck(Stmt):
    original: Rif
    took_reverse: bool


_INTERNAL_STMTS = (_IfAssertionCheck, _LoopContinuation, _LoopEntryCheck, _RifAssertionCheck)


@dataclass
class _PendingItem:
    stmt: Stmt
    direction: str  # "forward" or "backward"


class DebugSession:
    def __init__(self, program: Program, inputs: list[int]):
        layout = build_layout(program.inputs, program.outputs, program.temps)
        if len(inputs) != len(layout.inputs):
            raise EvalError(
                f"arity mismatch: expected {len(layout.inputs)}, got {len(inputs)}"
            )
        check_program(program)

        self._layout = layout
        self._store = _initialize_store(program, inputs, layout)
        self._program = program
        self._position = 0
        self._history: list[DebugState] = []
        self._snapshots: list[dict[str, object]] = []
        self._queue: list[_PendingItem] = []
        self._expand_block(program.body, "forward")

    # -- public API --------------------------------------------------------

    def step_forward(self) -> DebugState:
        if not self._queue:
            raise EvalError("execution already finished")

        snap = self._snapshot_store()
        queue_snap = list(self._queue)

        item = self._queue.pop(0)
        stmt, direction = item.stmt, item.direction

        if isinstance(stmt, If):
            stmt_text = _describe_stmt(stmt, direction)
            self._step_if(stmt, direction)
        elif isinstance(stmt, Loop):
            stmt_text = _describe_stmt(stmt, direction)
            self._step_loop(stmt, direction)
        elif isinstance(stmt, Rif):
            stmt_text = _describe_stmt(stmt, direction)
            self._step_rif(stmt, direction)
        elif isinstance(stmt, _INTERNAL_STMTS):
            stmt_text = self._step_internal(stmt)
        else:
            _eval_stmt(stmt, self._store, self._layout)
            stmt_text = _describe_stmt(stmt, direction)

        self._snapshots.append({"store": snap, "queue": queue_snap})
        state = DebugState(
            store=self._user_store(),
            position=self._position,
            stmt_text=stmt_text,
        )
        self._position += 1
        self._history.append(state)
        return state

    def step_backward(self) -> DebugState:
        if not self._snapshots:
            raise EvalError("no previous state to return to")

        saved = self._snapshots.pop()
        self._restore_store(saved["store"])  # type: ignore[arg-type]
        self._queue = saved["queue"]  # type: ignore[assignment]
        self._position -= 1
        popped = self._history.pop()
        return DebugState(
            store=self._user_store(),
            position=self._position,
            stmt_text=f"(undo) {popped.stmt_text}",
        )

    def current_state(self) -> DebugState:
        last_text = self._history[-1].stmt_text if self._history else "(initial)"
        return DebugState(
            store=self._user_store(),
            position=self._position,
            stmt_text=last_text,
        )

    def is_done(self) -> bool:
        return len(self._queue) == 0

    def run_to_end(self) -> DebugState:
        while not self.is_done():
            self.step_forward()
        return self.current_state()

    @property
    def history(self) -> list[DebugState]:
        return list(self._history)

    # -- control flow expansion --------------------------------------------

    def _expand_block(self, block: Block, direction: str) -> None:
        items = [_PendingItem(s, direction) for s in block.stmts]
        self._queue = items + self._queue

    def _step_if(self, stmt: If, direction: str) -> None:
        cond = _truthy(_eval_expr(stmt.test, self._store, self._layout))
        block = stmt.then_block if cond else stmt.else_block
        self._expand_block(block, direction)
        self._queue.insert(
            len(block.stmts),
            _PendingItem(_IfAssertionCheck(stmt, cond), direction),
        )

    def _step_loop(self, stmt: Loop, direction: str) -> None:
        if not _truthy(_eval_expr(stmt.entry_guard, self._store, self._layout)):
            raise EvalError("loop entry guard must hold before entering")
        items = [_PendingItem(s, direction) for s in stmt.do_block.stmts]
        items.append(_PendingItem(_LoopContinuation(stmt, first=True), direction))
        self._queue = items + self._queue

    def _step_rif(self, stmt: Rif, direction: str) -> None:
        cond = _truthy(_eval_expr(stmt.test, self._store, self._layout))
        if cond:
            block = invert_block(stmt.body)
            child_dir = "backward"
        else:
            block = stmt.body
            child_dir = "forward"
        self._expand_block(block, child_dir)
        self._queue.insert(
            len(block.stmts),
            _PendingItem(_RifAssertionCheck(stmt, cond), direction),
        )

    def _step_internal(self, stmt: Stmt) -> str:
        if isinstance(stmt, _IfAssertionCheck):
            assertion = _truthy(
                _eval_expr(stmt.original.assertion, self._store, self._layout)
            )
            if stmt.took_then and not assertion:
                raise EvalError("if assertion must hold after then branch")
            if not stmt.took_then and assertion:
                raise EvalError("if assertion must be false after else branch")
            branch = "then" if stmt.took_then else "else"
            return f"fi (checked {branch} branch)"

        if isinstance(stmt, _LoopContinuation):
            original = stmt.original
            if _truthy(_eval_expr(original.exit_guard, self._store, self._layout)):
                return "until (loop exit)"
            items: list[_PendingItem] = []
            for s in original.loop_block.stmts:
                items.append(_PendingItem(s, "forward"))
            items.append(_PendingItem(_LoopEntryCheck(original), "forward"))
            for s in original.do_block.stmts:
                items.append(_PendingItem(s, "forward"))
            items.append(
                _PendingItem(_LoopContinuation(original, first=False), "forward")
            )
            self._queue = items + self._queue
            return "loop (continuing)"

        if isinstance(stmt, _LoopEntryCheck):
            if _truthy(
                _eval_expr(stmt.original.entry_guard, self._store, self._layout)
            ):
                raise EvalError("loop entry guard must be false between iterations")
            return "from (entry guard check)"

        if isinstance(stmt, _RifAssertionCheck):
            assertion = _truthy(
                _eval_expr(stmt.original.assertion, self._store, self._layout)
            )
            if stmt.took_reverse and not assertion:
                raise EvalError("rif assertion must hold after reverse branch")
            if not stmt.took_reverse and assertion:
                raise EvalError("rif assertion must be false after forward branch")
            branch = "reverse" if stmt.took_reverse else "forward"
            return f"rfi (checked {branch} branch)"

        raise TypeError(f"unknown internal statement: {stmt!r}")

    # -- snapshot helpers --------------------------------------------------

    def _snapshot_store(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, val in self._store.items():
            if isinstance(val, list):
                result[key] = list(val)
            else:
                result[key] = val
        return result

    def _restore_store(self, snap: dict[str, object]) -> None:
        self._store.clear()
        for key, val in snap.items():
            if isinstance(val, list):
                self._store[key] = list(val)
            else:
                self._store[key] = val  # type: ignore[assignment]

    def _user_store(self) -> dict[str, int]:
        return {k: v for k, v in self._store.items() if not k.startswith("__stack_")}


def _truthy(value: int) -> bool:
    return value != 0


def _describe_stmt(stmt: Stmt, direction: str) -> str:
    if isinstance(stmt, Assign):
        return f"{_render_place(stmt.target)} {stmt.op.value} {_render_expr(stmt.expr)}"
    if isinstance(stmt, Swap):
        return f"{_render_place(stmt.left)} <=> {_render_place(stmt.right)}"
    if isinstance(stmt, If):
        return f"if {_render_expr(stmt.test)}"
    if isinstance(stmt, Loop):
        return f"from {_render_expr(stmt.entry_guard)}"
    if isinstance(stmt, Rif):
        return f"rif {_render_expr(stmt.test)}"
    if isinstance(stmt, Push):
        return f"push {stmt.var} {stmt.stack}"
    if isinstance(stmt, Pop):
        return f"pop {stmt.var} {stmt.stack}"
    if isinstance(stmt, _IfAssertionCheck):
        branch = "then" if stmt.took_then else "else"
        return f"fi (checked {branch} branch)"
    if isinstance(stmt, _LoopContinuation):
        return "until (loop check)"
    if isinstance(stmt, _LoopEntryCheck):
        return "from (entry guard check)"
    if isinstance(stmt, _RifAssertionCheck):
        branch = "reverse" if stmt.took_reverse else "forward"
        return f"rfi (checked {branch} branch)"
    return repr(stmt)


def debug_trace(program: Program, inputs: list[int]) -> list[dict[str, object]]:
    session = DebugSession(program, inputs)
    trace: list[dict[str, object]] = []
    trace.append({
        "step": 0,
        "action": "init",
        "store": dict(session.current_state().store),
    })
    step = 0
    while not session.is_done():
        state = session.step_forward()
        step += 1
        trace.append({
            "step": step,
            "action": "forward",
            "stmt": state.stmt_text,
            "store": dict(state.store),
        })

    # Step all the way back to demonstrate reversibility
    while session._snapshots:
        state = session.step_backward()
        step += 1
        trace.append({
            "step": step,
            "action": "backward",
            "stmt": state.stmt_text,
            "store": dict(state.store),
        })

    return trace
