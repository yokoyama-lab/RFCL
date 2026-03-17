"""Tests for Bennett transformation (irreversible -> reversible compilation).

Verifies that bennett_transform and make_reversible_program correctly apply
Bennett's trick: forward compute, copy outputs, reverse (uncompute), yielding
a clean reversible program with no garbage.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from subprocess import run as subprocess_run

from pyrev_fl.ast import (
    Assign,
    BinOp,
    Binary,
    Block,
    Const,
    Loop,
    Program,
    Rif,
    Swap,
    UpdateOp,
    Var,
)
from pyrev_fl.bennett import bennett_transform, make_reversible_program
from pyrev_fl.interface import build_layout
from pyrev_fl.interpreter import run_program
from pyrev_fl.invert import invert_program
from pyrev_fl.parser import parse_program
from pyrev_fl.pretty import render_program


EXAMPLES = Path(__file__).parent.parent / "examples"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_and_get_outputs(program: Program, inputs: list[int]) -> dict[str, int]:
    """Run a program and return the output variable values."""
    layout = build_layout(program.inputs, program.outputs, program.temps)
    store = run_program(program, inputs)
    return {name: store[name] for name in layout.outputs}


def _srl_round_trip(program: Program, inputs: list[int]) -> list[int]:
    """Run program forward, then run its inverse on the outputs."""
    layout = build_layout(program.inputs, program.outputs, program.temps)
    store_fwd = run_program(program, inputs)
    output_vals = [store_fwd[name] for name in layout.outputs]

    inv = invert_program(program)
    inv_layout = build_layout(inv.inputs, inv.outputs, inv.temps)
    store_rev = run_program(inv, output_vals)
    return [store_rev[name] for name in inv_layout.outputs]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class BennettTransformTests(unittest.TestCase):

    def test_bennett_simple_add(self):
        """Transform y += x into a clean reversible program.

        Original: (x) -> (y=x) with y as output.
        Bennett'd: inputs x, outputs x and y_copy.
        """
        body = Block([Assign("y", UpdateOp.ADD, Var("x"))])
        prog = make_reversible_program(
            body_stmts=body.stmts,
            inputs=["x"],
            outputs=["y"],
            temps=[],
        )
        for x in [0, 1, 3, 7, -5]:
            with self.subTest(x=x):
                result = _run_and_get_outputs(prog, [x])
                self.assertEqual(result["x"], x, "input should be preserved")
                self.assertEqual(result["y_copy"], x, "output copy should equal original output")

    def test_bennett_multi_stmt(self):
        """Transform a multi-statement computation.

        Original: t = x * 2; y = t + 1  (using SRL-style updates)
        """
        body = Block([
            Assign("t", UpdateOp.ADD, Var("x")),
            Assign("t", UpdateOp.ADD, Var("x")),  # t = 2*x
            Assign("y", UpdateOp.ADD, Var("t")),
            Assign("y", UpdateOp.ADD, Const(1)),   # y = 2*x + 1
        ])
        prog = make_reversible_program(
            body_stmts=body.stmts,
            inputs=["x"],
            outputs=["y"],
            temps=["t"],
        )
        for x in [0, 1, 5, 10]:
            with self.subTest(x=x):
                result = _run_and_get_outputs(prog, [x])
                self.assertEqual(result["x"], x)
                self.assertEqual(result["y_copy"], 2 * x + 1)

    def test_bennett_preserves_outputs(self):
        """Original and Bennett'd programs produce same logical outputs."""
        # Original (non-reversible) computation: y = x + 3
        original = Program(
            inputs=["x"],
            outputs=["x", "y"],
            temps=[],
            body=Block([Assign("y", UpdateOp.ADD, Var("x")),
                        Assign("y", UpdateOp.ADD, Const(3))]),
        )
        bennett = make_reversible_program(
            body_stmts=original.body.stmts,
            inputs=["x"],
            outputs=["y"],
            temps=[],
        )
        for x in [0, 2, 7, -3]:
            with self.subTest(x=x):
                orig_out = run_program(original, [x])
                benn_out = _run_and_get_outputs(bennett, [x])
                self.assertEqual(orig_out["y"], benn_out["y_copy"],
                                 "Bennett'd output should match original")

    def test_bennett_cleans_temps(self):
        """All temps are zero after Bennett'd program runs.

        We verify this by checking that run_program succeeds (it validates
        that all temps are zero at exit).
        """
        # Computation: t1 = x, t2 = x + 1, y = t1 + t2
        body_stmts = [
            Assign("t1", UpdateOp.ADD, Var("x")),
            Assign("t2", UpdateOp.ADD, Var("x")),
            Assign("t2", UpdateOp.ADD, Const(1)),
            Assign("y", UpdateOp.ADD, Var("t1")),
            Assign("y", UpdateOp.ADD, Var("t2")),
        ]
        prog = make_reversible_program(
            body_stmts=body_stmts,
            inputs=["x"],
            outputs=["y"],
            temps=["t1", "t2"],
        )
        # run_program will raise EvalError if any temp is non-zero
        for x in [0, 1, 5, 10]:
            with self.subTest(x=x):
                store = run_program(prog, [x])
                # Also explicitly check
                layout = build_layout(prog.inputs, prog.outputs, prog.temps)
                for t in layout.temps:
                    self.assertEqual(store[t], 0, f"temp {t} should be zero")

    def test_bennett_is_reversible(self):
        """The Bennett'd program satisfies the inversion theorem.

        run(invert(prog), outputs(run(prog, inputs))) == inputs
        """
        body_stmts = [
            Assign("y", UpdateOp.ADD, Var("x")),
            Assign("y", UpdateOp.ADD, Const(5)),
        ]
        prog = make_reversible_program(
            body_stmts=body_stmts,
            inputs=["x"],
            outputs=["y"],
            temps=[],
        )
        for x in [0, 3, 7, -2, 100]:
            with self.subTest(x=x):
                recovered = _srl_round_trip(prog, [x])
                self.assertEqual(recovered, [x])

    def test_bennett_with_loop(self):
        """Transform a computation involving a loop (factorial-like accumulation).

        Compute sum 1+2+...+n using a from/do/loop/until.
        """
        # The loop: from (= acc 0) do acc += n; loop n -= 1; acc += n; until (= n 0)
        # This computes acc = n + (n-1) + ... + 1 = n*(n+1)/2
        loop_body = Block([
            Loop(
                entry_guard=Binary(BinOp.EQ, Var("acc"), Const(0)),
                do_block=Block([Assign("acc", UpdateOp.ADD, Var("n"))]),
                loop_block=Block([
                    Assign("n", UpdateOp.SUB, Const(1)),
                    Assign("acc", UpdateOp.ADD, Var("n")),
                ]),
                exit_guard=Binary(BinOp.EQ, Var("n"), Const(0)),
            ),
        ])
        prog = make_reversible_program(
            body_stmts=loop_body.stmts,
            inputs=["n"],
            outputs=["acc"],
            temps=[],
        )
        # The loop computes acc = n + 2*(n-1) + 2*(n-2) + ... + 2*0 = n^2
        for n, expected in [(1, 1), (2, 4), (3, 9), (4, 16), (5, 25)]:
            with self.subTest(n=n):
                result = _run_and_get_outputs(prog, [n])
                self.assertEqual(result["n"], n, "input must be preserved")
                self.assertEqual(result["acc_copy"], expected)

    def test_make_reversible_program(self):
        """End-to-end test: make_reversible_program -> render -> parse -> run."""
        body_stmts = [
            Assign("y", UpdateOp.ADD, Var("x")),
            Assign("y", UpdateOp.ADD, Var("x")),  # y = 2*x
        ]
        prog = make_reversible_program(
            body_stmts=body_stmts,
            inputs=["x"],
            outputs=["y"],
            temps=[],
        )
        # Render, re-parse, and run — verify the program survives serialization
        source = render_program(prog)
        reparsed = parse_program(source)
        # Verify structural equivalence of the interface
        self.assertEqual(reparsed.inputs, prog.inputs)
        self.assertEqual(reparsed.outputs, prog.outputs)
        self.assertEqual(reparsed.temps, prog.temps)
        # Verify functional equivalence
        result = _run_and_get_outputs(reparsed, [7])
        self.assertEqual(result["x"], 7)
        self.assertEqual(result["y_copy"], 14)

    def test_bennett_matches_handwritten_fib(self):
        """The handwritten fib_bennett.srl should produce the same results as
        manually applying Bennett's trick to a fibonacci loop."""
        handwritten = parse_program((EXAMPLES / "fib_bennett.srl").read_text())
        for n in [1, 2, 3, 4, 5]:
            with self.subTest(n=n):
                result = _run_and_get_outputs(handwritten, [n])
                # Fibonacci: F(1)=1, F(2)=1, F(3)=2, F(4)=3, F(5)=5
                fibs = {1: 1, 2: 1, 3: 2, 4: 3, 5: 5}
                self.assertEqual(result["output"], fibs[n])
                self.assertEqual(result["n"], n, "input preserved")

    def test_bennett_name_collision_avoidance(self):
        """Fresh names avoid collisions with existing variable names."""
        # Use names that would collide with default bennett names
        body = Block([Assign("_bf", UpdateOp.ADD, Var("x"))])
        prog = make_reversible_program(
            body_stmts=body.stmts,
            inputs=["x"],
            outputs=["_bf"],
            temps=[],
        )
        # The flag variable should NOT be "_bf" since that's already used
        # Check that the program runs correctly
        result = _run_and_get_outputs(prog, [42])
        self.assertEqual(result["x"], 42)
        # The copy of _bf should be 42
        # Find the copy name (it should be _bf_copy)
        found_copy = [n for n in prog.outputs if n not in prog.inputs]
        self.assertEqual(len(found_copy), 1)
        self.assertEqual(result[found_copy[0]], 42)

    def test_bennett_multiple_outputs(self):
        """Bennett transform with multiple output variables."""
        # Compute y1 = x + 1, y2 = x + 2
        body_stmts = [
            Assign("y1", UpdateOp.ADD, Var("x")),
            Assign("y1", UpdateOp.ADD, Const(1)),
            Assign("y2", UpdateOp.ADD, Var("x")),
            Assign("y2", UpdateOp.ADD, Const(2)),
        ]
        prog = make_reversible_program(
            body_stmts=body_stmts,
            inputs=["x"],
            outputs=["y1", "y2"],
            temps=[],
        )
        for x in [0, 3, 10]:
            with self.subTest(x=x):
                result = _run_and_get_outputs(prog, [x])
                self.assertEqual(result["x"], x)
                self.assertEqual(result["y1_copy"], x + 1)
                self.assertEqual(result["y2_copy"], x + 2)

    def test_bennett_reversibility_with_multiple_inputs(self):
        """Bennett'd program with multiple inputs is reversible."""
        # y = a + b
        body_stmts = [Assign("y", UpdateOp.ADD, Var("a")),
                       Assign("y", UpdateOp.ADD, Var("b"))]
        prog = make_reversible_program(
            body_stmts=body_stmts,
            inputs=["a", "b"],
            outputs=["y"],
            temps=[],
        )
        for a, b in [(1, 2), (0, 0), (5, -3), (100, 200)]:
            with self.subTest(a=a, b=b):
                recovered = _srl_round_trip(prog, [a, b])
                self.assertEqual(recovered, [a, b])


class BennettCliTests(unittest.TestCase):
    """Test the ``bennett`` CLI subcommand."""

    def test_cli_bennett_basic(self):
        result = subprocess_run(
            ["python3", "-m", "pyrev_fl.cli", "bennett", "examples/copy.srl"],
            check=True,
            capture_output=True,
            text=True,
        )
        # Output should be a valid SRL program
        prog = parse_program(result.stdout)
        # Running it should work
        store = run_program(prog, [5])
        layout = build_layout(prog.inputs, prog.outputs, prog.temps)
        outputs = {n: store[n] for n in layout.outputs}
        self.assertEqual(outputs["x"], 5)

    def test_cli_bennett_with_output_vars(self):
        result = subprocess_run(
            ["python3", "-m", "pyrev_fl.cli", "bennett",
             "--output-vars", "y",
             "examples/copy.srl"],
            check=True,
            capture_output=True,
            text=True,
        )
        prog = parse_program(result.stdout)
        store = run_program(prog, [5])
        layout = build_layout(prog.inputs, prog.outputs, prog.temps)
        outputs = {n: store[n] for n in layout.outputs}
        self.assertEqual(outputs["x"], 5)
        self.assertEqual(outputs["y_copy"], 5)


if __name__ == "__main__":
    unittest.main()
