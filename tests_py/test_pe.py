"""Tests for the partial evaluator."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from pyrev_fl.interpreter import run_program
from pyrev_fl.parser import parse_program
from pyrev_fl.pe import partial_eval
from pyrev_fl.pretty import render_program

EXAMPLES = Path(__file__).parent.parent / "examples"


class PartialEvalTests(unittest.TestCase):

    def test_fully_known_eliminates_all(self):
        """All inputs known → empty body (all computed at PE time)."""
        src = "(x y) (x y) ()\nx += y;\n"
        prog = parse_program(src)
        residual = partial_eval(prog, {"x": 5, "y": 3})
        self.assertEqual(residual.body.stmts, [])
        self.assertEqual(residual.inputs, [])

    def test_partially_known_simplifies(self):
        """Known y=3 specializes x += y to x += 3."""
        src = "(x y) (x y) ()\nx += y;\n"
        prog = parse_program(src)
        residual = partial_eval(prog, {"y": 3})
        rendered = render_program(residual)
        self.assertIn("x += 3", rendered)
        self.assertEqual(residual.inputs, ["x"])

    def test_if_known_branch_eliminated(self):
        """Known flag=1 eliminates the else branch."""
        src = (
            "(x flag) (x y flag) ()\n"
            "if (!= flag 0) then\n  y ^= x;\nelse\nfi (!= flag 0)\n"
        )
        prog = parse_program(src)
        residual = partial_eval(prog, {"flag": 1})
        rendered = render_program(residual)
        self.assertIn("y ^= x", rendered)
        self.assertNotIn("if", rendered)
        self.assertNotIn("fi", rendered)

    def test_if_false_branch_eliminated(self):
        """Known flag=0 eliminates the then branch."""
        src = (
            "(x flag) (x y flag) ()\n"
            "if (!= flag 0) then\n  y ^= x;\nelse\n  y -= x;\nfi (!= flag 0)\n"
        )
        prog = parse_program(src)
        residual = partial_eval(prog, {"flag": 0})
        rendered = render_program(residual)
        self.assertIn("y -= x", rendered)
        self.assertNotIn("y ^= x", rendered)

    def test_residual_preserves_semantics(self):
        """Specialized program produces same output as original."""
        src = "(x y) (x y) ()\nx += y;\ny += 1;\n"
        prog = parse_program(src)
        residual = partial_eval(prog, {"y": 3})
        orig = run_program(prog, [5, 3])
        # Residual inputs: [x=5], y is eliminated
        res = run_program(residual, [5])
        self.assertEqual(orig["x"], res["x"])

    def test_swap_known_values(self):
        """Swap of two known values is computed at PE time."""
        src = "(x y) (x y) ()\nx <=> y;\n"
        prog = parse_program(src)
        residual = partial_eval(prog, {"x": 5, "y": 3})
        self.assertEqual(residual.body.stmts, [])

    def test_cli_pe(self):
        result = subprocess.run(
            [sys.executable, "-m", "pyrev_fl.cli", "pe",
             str(EXAMPLES / "copy.srl"), "--known", "x=5"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("y += 5", result.stdout)
