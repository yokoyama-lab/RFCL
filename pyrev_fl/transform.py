from __future__ import annotations

from dataclasses import dataclass

from pyrev_fl.ast import BinOp, Binary, Const, Expr
from pyrev_fl.ast import Assign as SAssign
from pyrev_fl.ast import Block as SBlock
from pyrev_fl.ast import If as SIf
from pyrev_fl.ast import Loop as SLoop
from pyrev_fl.ast import Program as SProgram
from pyrev_fl.ast import Rif as SRif
from pyrev_fl.ast import Stmt as SStmt
from pyrev_fl.ast import Swap as SSwap
from pyrev_fl.invert import invert_block
from pyrev_fl.rl_ast import Assign as RAssign
from pyrev_fl.rl_ast import Block as RBlock
from pyrev_fl.rl_ast import Exit, FiFrom, FromEntry, FromLabel, Goto, IfGoto, Program as RProgram, RFromLabel, RGoto, Swap as RSwap


class TransformError(ValueError):
    pass


@dataclass
class _LowerCtx:
    counter: int = 0

    def fresh(self, prefix: str) -> str:
        label = f"{prefix}_{self.counter}"
        self.counter += 1
        return label


@dataclass
class _Lowered:
    blocks: list[RBlock]
    exit_predecessor: str


@dataclass
class _RaisedPattern:
    stmts: list[SStmt]
    next_label: str | None
    consumed: set[str]


@dataclass
class _Cfg:
    block_map: dict[str, RBlock]
    forward_succ: dict[str, set[str]]
    forward_pred: dict[str, set[str]]
    structural_neighbors: dict[str, set[str]]
    idom: dict[str, str | None]

    def reachable_forward(self, start: str, excluded: str | None = None) -> set[str]:
        reachable: set[str] = set()
        stack = [start]
        while stack:
            label = stack.pop()
            if label in reachable or label not in self.block_map or label == excluded:
                continue
            reachable.add(label)
            stack.extend(sorted(self.forward_succ[label] - reachable))
        return reachable

    def dominates(self, a: str, b: str) -> bool:
        """Return True if *a* dominates *b* (i.e. every path from entry to *b* passes through *a*)."""
        current: str | None = b
        while current is not None:
            if current == a:
                return True
            current = self.idom.get(current)
        return False

    def back_edges(self) -> list[tuple[str, str]]:
        """Return all back-edges ``(src, target)`` where *target* dominates *src*."""
        edges: list[tuple[str, str]] = []
        for src, targets in self.forward_succ.items():
            for tgt in targets:
                if self.dominates(tgt, src):
                    edges.append((src, tgt))
        return edges

    def natural_loop_body(self, header: str, back_src: str) -> set[str]:
        """Compute the set of blocks in the natural loop for back-edge *back_src* → *header*."""
        body = {header}
        worklist = [back_src]
        while worklist:
            b = worklist.pop()
            if b in body:
                continue
            body.add(b)
            for pred in self.forward_pred.get(b, set()):
                if pred not in body:
                    worklist.append(pred)
        return body


def lower_srl_to_rl(program: SProgram) -> RProgram:
    ctx = _LowerCtx()
    entry_label = ctx.fresh("entry")
    start_label = ctx.fresh("start")
    exit_label = ctx.fresh("exit")
    lowered = _lower_block(program.body, entry_label, start_label, exit_label, ctx)
    entry_block = RBlock(entry_label, FromEntry(), [], Goto(start_label))
    if not lowered.blocks:
        blocks = [entry_block, RBlock(start_label, FromLabel(entry_label), [], Goto(exit_label))]
        predecessor = start_label
    else:
        blocks = [entry_block, *lowered.blocks]
        predecessor = lowered.exit_predecessor
    blocks.append(RBlock(exit_label, FromLabel(predecessor), [], Exit()))
    return RProgram(program.inputs, program.outputs, program.temps, blocks)


def _lower_block(block: SBlock, prev_label: str, start_label: str, next_label: str, ctx: _LowerCtx) -> _Lowered:
    if not block.stmts:
        return _Lowered([RBlock(start_label, FromLabel(prev_label), [], Goto(next_label))], start_label)

    labels = [start_label] + [ctx.fresh("seq") for _ in range(len(block.stmts) - 1)]
    blocks: list[RBlock] = []
    incoming = prev_label
    exit_predecessor = start_label
    for index, stmt in enumerate(block.stmts):
        current_label = labels[index]
        outgoing = next_label if index == len(block.stmts) - 1 else labels[index + 1]
        lowered = _lower_stmt(stmt, incoming, current_label, outgoing, ctx)
        blocks.extend(lowered.blocks)
        incoming = lowered.exit_predecessor
        exit_predecessor = lowered.exit_predecessor
    return _Lowered(blocks, exit_predecessor)


def _lower_stmt(stmt: SStmt, prev_label: str, label: str, next_label: str, ctx: _LowerCtx) -> _Lowered:
    if isinstance(stmt, SAssign):
        return _Lowered([RBlock(label, FromLabel(prev_label), [RAssign(stmt.target, stmt.op, stmt.expr)], Goto(next_label))], label)
    if isinstance(stmt, SSwap):
        return _Lowered([RBlock(label, FromLabel(prev_label), [RSwap(stmt.left, stmt.right)], Goto(next_label))], label)
    if isinstance(stmt, SIf):
        then_label = ctx.fresh("then")
        else_label = ctx.fresh("else")
        join_label = ctx.fresh("join")
        then_blocks = _lower_block(stmt.then_block, label, then_label, join_label, ctx)
        else_blocks = _lower_block(stmt.else_block, label, else_label, join_label, ctx)
        join_prev_true = then_blocks.exit_predecessor
        join_prev_false = else_blocks.exit_predecessor
        return _Lowered(
            [
                RBlock(label, FromLabel(prev_label), [], IfGoto(stmt.test, then_label, else_label)),
                *then_blocks.blocks,
                *else_blocks.blocks,
                RBlock(join_label, FiFrom(stmt.assertion, join_prev_true, join_prev_false), [], Goto(next_label)),
            ],
            join_label,
        )
    if isinstance(stmt, SLoop):
        do_start = label
        after_do = ctx.fresh("after_do")
        loop_body_start = ctx.fresh("loop_body")
        loop_back = ctx.fresh("loop_back")

        do_blocks = _lower_block(stmt.do_block, prev_label, do_start, after_do, ctx)
        first_do = do_blocks.blocks[0]
        do_blocks.blocks[0] = RBlock(
            first_do.label,
            FiFrom(stmt.entry_guard, prev_label, loop_back),
            first_do.assigns,
            first_do.jump,
        )
        do_last = do_blocks.exit_predecessor

        loop_body_blocks = _lower_block(stmt.loop_block, after_do, loop_body_start, loop_back, ctx)
        loop_body_last = loop_body_blocks.exit_predecessor

        return _Lowered(
            [
                *do_blocks.blocks,
                RBlock(after_do, FromLabel(do_last), [], IfGoto(stmt.exit_guard, next_label, loop_body_start)),
                *loop_body_blocks.blocks,
                RBlock(loop_back, FromLabel(loop_body_last), [], Goto(do_start)),
            ],
            after_do,
        )
    if isinstance(stmt, SRif):
        fwd_label = ctx.fresh("rif_fwd")
        rev_label = ctx.fresh("rif_rev")
        join_label = ctx.fresh("rif_join")
        fwd_lowered = _lower_block(stmt.body, label, fwd_label, join_label, ctx)
        rev_lowered = _lower_block(invert_block(stmt.body), label, rev_label, join_label, ctx)
        fwd_exit = fwd_lowered.exit_predecessor
        rev_exit = rev_lowered.exit_predecessor
        return _Lowered(
            [
                RBlock(label, FromLabel(prev_label), [], IfGoto(stmt.test, rev_label, fwd_label)),
                *fwd_lowered.blocks,
                *rev_lowered.blocks,
                RBlock(join_label, FiFrom(stmt.assertion, rev_exit, fwd_exit), [], Goto(next_label)),
            ],
            join_label,
        )
    raise TransformError("lowering currently supports only assign, swap, if, loop, and rif")


def raise_rl_to_srl(program: RProgram) -> SProgram:
    block_map = {block.label: block for block in program.blocks}
    cfg = _build_cfg(block_map)
    entry = next((block for block in program.blocks if isinstance(block.from_, FromEntry)), None)
    if entry is None:
        raise TransformError("missing entry block")
    body, _ = _raise_linear(entry.label, cfg, set(), stop_labels=set())
    return SProgram(program.inputs, program.outputs, program.temps, body)


def _raise_linear(
    label: str,
    cfg: _Cfg,
    seen: set[str],
    stop_labels: set[str],
    *,
    detect_loops: bool = True,
) -> tuple[SBlock, str | None]:
    stmts: list[SStmt] = []
    current: str | None = label
    while current is not None and current not in seen:
        if current in stop_labels:
            return SBlock(stmts), current
        seen.add(current)
        block = cfg.block_map[current]

        # --- Pattern matchers that consume entire multi-block regions ---

        reverse_match = _match_bennett_reverse_jump_stmt(current, cfg)
        if reverse_match is not None:
            stmts.extend(reverse_match.stmts)
            seen.update(reverse_match.consumed)
            current = reverse_match.next_label
            continue
        if isinstance(block.from_, RFromLabel):
            raise TransformError("raising does not support rfrom blocks yet")
        if isinstance(block.jump, RGoto):
            raise TransformError("raising does not support rgoto jumps yet")

        loop_match = _match_generated_loop(current, cfg) if detect_loops else None
        if loop_match is not None:
            loop_stmt, next_label, consumed = loop_match
            stmts.append(loop_stmt)
            seen.update(consumed)
            current = next_label
            continue

        # --- Process all assignments in this block ---

        stmts.extend(_raise_assignments(block.assigns))

        # --- Handle the jump ---

        if isinstance(block.jump, IfGoto):
            join_label, assertion = _find_join_for_if(cfg, block.jump.true_label, block.jump.false_label)
            then_block, then_stop = _raise_linear(block.jump.true_label, cfg, set(seen), {join_label})
            else_block, else_stop = _raise_linear(block.jump.false_label, cfg, set(seen), {join_label})
            if then_stop != join_label or else_stop != join_label:
                raise TransformError("raising only supports RL if-shapes with a single join block")
            # Detect rif pattern: true branch = invert(false branch)
            if invert_block(then_block) == else_block:
                stmts.append(SRif(block.jump.expr, else_block, assertion))
            else:
                stmts.append(SIf(block.jump.expr, then_block, else_block, assertion))
            join_block = cfg.block_map[join_label]
            seen.add(join_label)
            stmts.extend(_raise_assignments(join_block.assigns))
            if isinstance(join_block.jump, Goto):
                current = join_block.jump.label if join_block.jump.label in cfg.block_map else None
            elif isinstance(join_block.jump, Exit):
                return SBlock(stmts), None
            else:
                raise TransformError("unsupported join block shape")
            continue

        if isinstance(block.jump, Goto):
            current = block.jump.label if block.jump.label in cfg.block_map else None
            continue
        if isinstance(block.jump, Exit):
            return SBlock(stmts), None
        raise TransformError("unsupported block shape")

    return SBlock(stmts), current


def _find_after_do_in_loop(do_start: str, loop_back: str, cfg: _Cfg) -> str | None:
    """Find the after_do block for a loop.

    The after_do block has IfGoto where:
    - true branch exits the loop (does not reach loop_back)
    - false branch enters the loop body (eventually reaches loop_back)

    Works for do_blocks with any number of statements, including nested if/rif.
    """
    reachable = cfg.reachable_forward(do_start, loop_back)
    for candidate in reachable:
        if candidate not in cfg.block_map:
            continue
        blk = cfg.block_map[candidate]
        if not isinstance(blk.jump, IfGoto):
            continue
        false_label = blk.jump.false_label
        true_label = blk.jump.true_label
        # False branch must lead to loop_back; true branch must not
        false_reaches_back = loop_back in cfg.reachable_forward(false_label, do_start)
        true_reaches_back = loop_back in cfg.reachable_forward(true_label, do_start)
        if false_reaches_back and not true_reaches_back:
            return candidate
    return None


def _match_generated_loop(
    start_label: str,
    cfg: _Cfg,
) -> tuple[SLoop, str | None, set[str]] | None:
    start_block = cfg.block_map[start_label]
    if not isinstance(start_block.from_, FiFrom):
        return None

    loop_back_label = start_block.from_.false_label
    if loop_back_label not in cfg.block_map:
        return None
    loop_back_block = cfg.block_map[loop_back_label]
    if not isinstance(loop_back_block.jump, Goto) or loop_back_block.jump.label != start_label:
        return None

    after_do_label = _find_after_do_in_loop(start_label, loop_back_label, cfg)
    if after_do_label is None:
        return None
    after_do_block = cfg.block_map[after_do_label]
    if not isinstance(after_do_block.jump, IfGoto):
        return None

    loop_body_start = after_do_block.jump.false_label
    next_label = after_do_block.jump.true_label
    if loop_body_start not in cfg.block_map:
        return None

    do_block, do_stop = _raise_linear(
        start_label,
        cfg,
        set(),
        {after_do_label},
        detect_loops=False,
    )
    if do_stop != after_do_label:
        return None
    # Include after_do's assignments at the end of do_block (hand-written RL
    # may put assignments in the same block as the exit-test IfGoto).
    after_do_assigns = _raise_assignments(after_do_block.assigns)
    if after_do_assigns:
        do_block = SBlock(do_block.stmts + after_do_assigns)

    loop_block, loop_stop = _raise_linear(
        loop_body_start,
        cfg,
        set(),
        {loop_back_label},
        detect_loops=False,
    )
    if loop_stop != loop_back_label:
        return None
    # Include loop_back's assignments (usually empty for lowered RL).
    loop_back_assigns = _raise_assignments(loop_back_block.assigns)
    if loop_back_assigns:
        loop_block = SBlock(loop_block.stmts + loop_back_assigns)

    loop_stmt = SLoop(
        entry_guard=start_block.from_.expr,
        do_block=do_block,
        loop_block=loop_block,
        exit_guard=after_do_block.jump.expr,
    )
    # Use reachability for consumed set (handles nested if/fi within loop).
    consumed = cfg.reachable_forward(start_label, excluded=after_do_label)
    consumed.add(after_do_label)
    consumed.add(loop_back_label)
    if loop_body_start in cfg.block_map and loop_body_start != loop_back_label:
        consumed.update(cfg.reachable_forward(loop_body_start, excluded=loop_back_label))
    return loop_stmt, next_label, consumed



def _find_join_for_if(cfg: _Cfg, then_label: str, else_label: str) -> tuple[str, object]:
    """Find the FiFrom join block for an if/rif.

    For each candidate FiFrom block B, computes forward reachability from each
    branch *excluding B itself* (to avoid cycles from loops). The correct join
    has one predecessor reachable only from the then-branch and the other only
    from the else-branch.
    """
    for block in cfg.block_map.values():
        if not isinstance(block.from_, FiFrom):
            continue
        t, f = block.from_.true_label, block.from_.false_label
        excluded = block.label
        then_r = cfg.reachable_forward(then_label, excluded)
        else_r = cfg.reachable_forward(else_label, excluded)
        if (t in then_r and f in else_r) or (t in else_r and f in then_r):
            return block.label, block.from_.expr
    raise TransformError("raising only supports generated RL if-shapes with fi/from join blocks")


def _match_bennett_reverse_jump_stmt(
    start_label: str,
    cfg: _Cfg,
) -> _RaisedPattern | None:
    init = cfg.block_map.get(start_label)
    if init is None or not isinstance(init.from_, FiFrom) or not isinstance(init.jump, Goto):
        return None

    pre_end = cfg.block_map.get(init.from_.false_label)
    if pre_end is None or pre_end.assigns or not isinstance(pre_end.from_, RFromLabel) or not isinstance(pre_end.jump, Goto):
        return None
    if pre_end.jump.label != init.label:
        return None

    end = cfg.block_map.get(pre_end.from_.label)
    if end is None or not isinstance(end.from_, RFromLabel):
        return None
    if not isinstance(end.jump, (Goto, Exit)):
        return None
    if end.from_.label != pre_end.label:
        return None

    target = cfg.block_map.get(init.jump.label)
    if target is None:
        return None

    # ------------------------------------------------------------------
    # Case 1: Full Bennett pattern with inner loop (init → test → loop/copy)
    # ------------------------------------------------------------------
    if (
        not target.assigns
        and isinstance(target.from_, FiFrom)
        and isinstance(target.jump, IfGoto)
        and target.from_.true_label == init.label
        and target.from_.false_label == target.jump.false_label
    ):
        test = target
        loop_chain = _raise_linear_assign_chain(
            start_label=test.jump.false_label,
            cfg=cfg,
            expected_prev=test.label,
            terminal=Goto(test.label),
        )
        if loop_chain is None:
            return None
        loop_stmts, loop_tail, loop_consumed = loop_chain

        copy_entry = cfg.block_map.get(test.jump.true_label)
        if copy_entry is None or copy_entry.assigns or not isinstance(copy_entry.from_, FromLabel) or not isinstance(copy_entry.jump, IfGoto):
            return None
        if copy_entry.from_.label != test.label:
            return None
        if copy_entry.jump.expr != init.from_.expr:
            return None

        result = _match_bennett_copy_section(copy_entry, cfg)
        if result is None:
            return None
        copy_stmts, copy_consumed = result

        outer_entry_guard = _normalize_zero_comparison(init.from_.expr)
        rif_condition = _negate_expr(outer_entry_guard)
        inner_loop = SLoop(
            entry_guard=test.from_.expr,
            do_block=SBlock(loop_stmts),
            loop_block=SBlock([]),
            exit_guard=test.jump.expr,
        )
        rif_stmt = SRif(
            test=rif_condition,
            body=SBlock([*_raise_assignments(init.assigns), inner_loop]),
            assertion=rif_condition,
        )
        outer_loop = SLoop(
            entry_guard=outer_entry_guard,
            do_block=SBlock([rif_stmt]),
            loop_block=SBlock(copy_stmts),
            exit_guard=_negate_expr(copy_entry.jump.expr),
        )
        next_label = end.jump.label if isinstance(end.jump, Goto) else None
        return _RaisedPattern(
            stmts=[outer_loop, *_raise_assignments(end.assigns)],
            next_label=next_label,
            consumed={
                init.label, test.label, copy_entry.label,
                pre_end.label, end.label,
                *loop_consumed, *copy_consumed,
            },
        )

    # ------------------------------------------------------------------
    # Case 2: Simple Bennett pattern without inner loop (init → copy-entry)
    # ------------------------------------------------------------------
    copy_entry = target
    if (
        not copy_entry.assigns
        and isinstance(copy_entry.from_, FromLabel)
        and isinstance(copy_entry.jump, IfGoto)
        and copy_entry.from_.label == init.label
        and copy_entry.jump.expr == init.from_.expr
    ):
        result = _match_bennett_copy_section(copy_entry, cfg)
        if result is None:
            return None
        copy_stmts, copy_consumed = result

        outer_entry_guard = _normalize_zero_comparison(init.from_.expr)
        rif_body_stmts = _raise_assignments(init.assigns)
        rif_condition = _negate_expr(outer_entry_guard)

        do_stmts: list[SStmt] = []
        if rif_body_stmts:
            do_stmts.append(SRif(test=rif_condition, body=SBlock(rif_body_stmts), assertion=rif_condition))
        outer_loop = SLoop(
            entry_guard=outer_entry_guard,
            do_block=SBlock(do_stmts),
            loop_block=SBlock(copy_stmts),
            exit_guard=_negate_expr(copy_entry.jump.expr),
        )
        next_label = end.jump.label if isinstance(end.jump, Goto) else None
        return _RaisedPattern(
            stmts=[outer_loop, *_raise_assignments(end.assigns)],
            next_label=next_label,
            consumed={
                init.label, copy_entry.label,
                pre_end.label, end.label,
                *copy_consumed,
            },
        )

    return None


def _match_bennett_copy_section(
    copy_entry: RBlock, cfg: _Cfg,
) -> tuple[list[SStmt], set[str]] | None:
    """Match the copy-main/copy-exit rgoto pair within a Bennett pattern."""
    if not isinstance(copy_entry.jump, IfGoto):
        return None
    copy_exit = cfg.block_map.get(copy_entry.jump.false_label)
    if copy_exit is None:
        return None
    copy_chain = _raise_linear_assign_chain(
        start_label=copy_entry.jump.true_label,
        cfg=cfg,
        expected_prev=copy_entry.label,
        terminal=RGoto(copy_exit.label),
    )
    if copy_chain is None:
        return None
    copy_stmts, copy_tail, copy_consumed = copy_chain
    if not isinstance(copy_exit.from_, FromLabel) or copy_exit.from_.label != copy_entry.label:
        return None
    if copy_exit.assigns or not isinstance(copy_exit.jump, RGoto) or copy_exit.jump.label != copy_tail:
        return None
    copy_consumed.add(copy_exit.label)
    return copy_stmts, copy_consumed


def _raise_assignments(assigns: list[RAssign | RSwap]) -> list[SStmt]:
    stmts: list[SStmt] = []
    for assign in assigns:
        if isinstance(assign, RAssign):
            stmts.append(SAssign(assign.target, assign.op, assign.expr))
        else:
            stmts.append(SSwap(assign.left, assign.right))
    return stmts


def _raise_linear_assign_chain(
    start_label: str,
    cfg: _Cfg,
    expected_prev: str,
    terminal,
) -> tuple[list[SStmt], str, set[str]] | None:
    stmts: list[SStmt] = []
    consumed: set[str] = set()
    current = start_label
    prev = expected_prev
    while current in cfg.block_map and current not in consumed:
        block = cfg.block_map[current]
        if not isinstance(block.from_, FromLabel) or block.from_.label != prev:
            return None
        if not block.assigns:
            return None
        stmts.extend(_raise_assignments(block.assigns))
        consumed.add(current)
        if block.jump == terminal:
            return stmts, current, consumed
        if not isinstance(block.jump, Goto):
            return None
        prev, current = current, block.jump.label
    return None


def _negate_expr(expr: Expr) -> Expr:
    if not isinstance(expr, Binary):
        raise TransformError("reverse-jump raising only supports binary guard expressions")
    negated = {
        BinOp.EQ: BinOp.NE,
        BinOp.NE: BinOp.EQ,
        BinOp.LT: BinOp.GE,
        BinOp.LE: BinOp.GT,
        BinOp.GT: BinOp.LE,
        BinOp.GE: BinOp.LT,
    }.get(expr.op)
    if negated is None:
        raise TransformError("reverse-jump raising only supports comparable guard expressions")
    return Binary(negated, expr.left, expr.right)


def _normalize_zero_comparison(expr: Expr) -> Expr:
    if not isinstance(expr, Binary):
        return expr
    if expr.op not in {BinOp.EQ, BinOp.NE}:
        return expr
    if isinstance(expr.right, Const) and expr.right.value == 0 and not isinstance(expr.left, Const):
        return Binary(expr.op, expr.right, expr.left)
    return expr


def _build_cfg(block_map: dict[str, RBlock]) -> _Cfg:
    labels = set(block_map)
    forward_succ: dict[str, set[str]] = {label: set() for label in labels}
    forward_pred: dict[str, set[str]] = {label: set() for label in labels}
    structural_neighbors: dict[str, set[str]] = {label: set() for label in labels}
    entry_label: str | None = None
    for label, block in block_map.items():
        if isinstance(block.from_, FromEntry):
            entry_label = label
        for target in _forward_jump_targets(block.jump):
            if target in labels:
                forward_succ[label].add(target)
                forward_pred[target].add(label)
                structural_neighbors[label].add(target)
                structural_neighbors[target].add(label)
        for neighbor in _from_neighbors(block):
            if neighbor in labels:
                structural_neighbors[label].add(neighbor)
                structural_neighbors[neighbor].add(label)
    idom = _compute_idom(forward_pred, labels, entry_label) if entry_label else {}
    return _Cfg(block_map, forward_succ, forward_pred, structural_neighbors, idom)


def _compute_idom(
    pred: dict[str, set[str]], all_blocks: set[str], entry: str
) -> dict[str, str | None]:
    """Compute immediate dominators using the iterative dataflow algorithm."""
    # Step 1: Compute dominator sets
    dom: dict[str, set[str]] = {entry: {entry}}
    for b in all_blocks:
        if b != entry:
            dom[b] = set(all_blocks)

    changed = True
    while changed:
        changed = False
        for b in all_blocks:
            if b == entry:
                continue
            preds = pred.get(b, set())
            pred_doms = [dom[p] for p in preds if p in dom and dom[p] != all_blocks]
            if not pred_doms:
                continue
            new_dom = set.intersection(*pred_doms) | {b}
            if new_dom != dom[b]:
                dom[b] = new_dom
                changed = True

    # Step 2: Extract immediate dominators
    idom: dict[str, str | None] = {entry: None}
    for b in all_blocks:
        if b == entry:
            continue
        strict_doms = dom.get(b, set()) - {b}
        if not strict_doms:
            continue
        # idom(b) is the strict dominator d closest to b:
        # every other strict dominator of b also dominates d.
        for d in strict_doms:
            if all(e in dom.get(d, set()) for e in strict_doms if e != d):
                idom[b] = d
                break
    return idom


def _forward_jump_targets(jump) -> set[str]:
    if isinstance(jump, Goto):
        return {jump.label}
    if isinstance(jump, IfGoto):
        return {jump.true_label, jump.false_label}
    return set()


def _from_neighbors(block: RBlock) -> set[str]:
    if isinstance(block.from_, (FromLabel, RFromLabel)):
        return {block.from_.label}
    if isinstance(block.from_, FiFrom):
        return {block.from_.true_label, block.from_.false_label}
    return set()
