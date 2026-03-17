"""Bennett transformation: compile irreversible computations into clean reversible programs.

Bennett's trick produces a garbage-free reversible version of an irreversible computation:
  1. Forward: compute the original body, producing outputs + garbage in temps
  2. Copy: XOR each output into a fresh copy variable
  3. Reverse: uncompute the original body (via rif), clearing all temps back to zero
  4. Result: only the copied outputs remain, with no garbage

The generated pattern follows Moriyama 2009, Figure 2.15:

    from (= 0 _bf) do
      rif (!= 0 _bf)
        <original computation>
      rfi (!= 0 _bf)
    loop
      <copy output vars via XOR>
      _bf ^= 1;
    until (!= _bf 0)
    _bf ^= 1;

where _bf (bennett flag) controls the forward/reverse direction.
"""

from __future__ import annotations

from pyrev_fl.ast import (
    Assign,
    BinOp,
    Binary,
    Block,
    Const,
    Loop,
    Program,
    Rif,
    Stmt,
    UpdateOp,
    Var,
)


_BENNETT_FLAG = "_bf"
_COPY_SUFFIX = "_copy"


def _fresh_flag(used: set[str]) -> str:
    """Return a bennett-flag name that does not collide with *used* names."""
    candidate = _BENNETT_FLAG
    i = 0
    while candidate in used:
        i += 1
        candidate = f"{_BENNETT_FLAG}{i}"
    return candidate


def _fresh_copy_name(name: str, used: set[str]) -> str:
    """Return a copy-variable name for *name* that does not collide with *used*."""
    candidate = f"{name}{_COPY_SUFFIX}"
    i = 0
    while candidate in used:
        i += 1
        candidate = f"{name}{_COPY_SUFFIX}{i}"
    return candidate


def bennett_transform(
    body: Block,
    output_vars: list[str],
    *,
    used_names: set[str] | None = None,
) -> tuple[Block, str, list[str]]:
    """Apply Bennett's trick to make a block garbage-free.

    Given a block that computes *output_vars* (plus possibly garbage in temps),
    produce a new block that:
      1. Runs the original computation forward
      2. Copies output_vars to fresh variables using XOR
      3. Runs the original computation in reverse (uncomputing garbage)
      4. Returns only the copied outputs

    Parameters
    ----------
    body : Block
        The original (potentially irreversible) computation body.
    output_vars : list[str]
        Names of variables that carry the desired outputs after *body* runs.
    used_names : set[str] | None
        All variable names already in use.  Fresh names are chosen to avoid
        collisions.  If ``None``, the set is derived from *output_vars* alone.

    Returns
    -------
    tuple[Block, str, list[str]]
        ``(transformed_block, flag_name, copy_names)`` where *flag_name* is the
        bennett flag temp and *copy_names* are the fresh copy variables (one per
        output_var, in the same order).
    """
    if used_names is None:
        used_names = set(output_vars)
    else:
        used_names = set(used_names)

    # Choose fresh names
    flag = _fresh_flag(used_names)
    used_names.add(flag)

    copy_names: list[str] = []
    for name in output_vars:
        cn = _fresh_copy_name(name, used_names)
        copy_names.append(cn)
        used_names.add(cn)

    # Build the "not-equal zero" expression for the flag
    flag_ne_zero = Binary(BinOp.NE, Const(0), Var(flag))
    flag_eq_zero = Binary(BinOp.EQ, Const(0), Var(flag))

    # rif block: forward on first pass, reverse on second
    rif_stmt = Rif(
        test=flag_ne_zero,
        body=body,
        assertion=flag_ne_zero,
    )

    # Copy statements: copy_var ^= output_var
    copy_stmts: list[Stmt] = []
    for out_name, cp_name in zip(output_vars, copy_names):
        copy_stmts.append(Assign(cp_name, UpdateOp.XOR, Var(out_name)))
    # Flip the flag
    copy_stmts.append(Assign(flag, UpdateOp.XOR, Const(1)))

    # Build the from/do/loop/until
    loop_stmt = Loop(
        entry_guard=flag_eq_zero,
        do_block=Block([rif_stmt]),
        loop_block=Block(copy_stmts),
        exit_guard=flag_ne_zero,
    )

    # Final reset of the flag
    reset_flag = Assign(flag, UpdateOp.XOR, Const(1))

    return Block([loop_stmt, reset_flag]), flag, copy_names


def make_reversible_program(
    body_stmts: list[Stmt],
    inputs: list[str],
    outputs: list[str],
    temps: list[str],
) -> Program:
    """Create a complete reversible SRL program from a list of statements.

    Wraps the statements in Bennett's trick, ensuring all temporary variables
    return to zero.  The resulting program takes the same inputs and produces
    the same outputs as the original computation, but is fully reversible.

    Parameters
    ----------
    body_stmts : list[Stmt]
        The original (potentially irreversible) computation as a list of statements.
    inputs : list[str]
        Input variable names.
    outputs : list[str]
        Output variable names -- these are the "real" results to preserve.
    temps : list[str]
        Temporary variable names used by the computation (will be zero-cleared).

    Returns
    -------
    Program
        A clean reversible SRL program.
    """
    body = Block(body_stmts)

    all_names = set(inputs) | set(outputs) | set(temps)
    transformed, flag, copy_names = bennett_transform(
        body, outputs, used_names=all_names,
    )

    # After Bennett's trick:
    #   - Input variables are restored to their original values.
    #   - Original output variables are zeroed by uncomputation (unless they
    #     are also inputs, in which case they are restored to their input value).
    #   - Temporary variables are zeroed by uncomputation.
    #   - Copy variables hold the desired output values.
    #   - The bennett flag is zeroed by the final reset statement.
    #
    # SRL convention:
    #   inputs  = original inputs
    #   outputs = original inputs + copy variables
    #   temps   = original outputs that are NOT inputs + original temps + flag

    input_set = set(inputs)
    # Only outputs that are purely outputs (not also inputs) become temps
    output_only_temps = [o for o in outputs if o not in input_set]

    new_temps = list(temps) + output_only_temps + [flag]

    return Program(
        inputs=list(inputs),
        outputs=list(inputs) + copy_names,
        temps=new_temps,
        body=transformed,
    )
