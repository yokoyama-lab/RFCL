"""Tests for the reversible program slicer."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from pyrev_fl.interpreter import run_program
from pyrev_fl.interface import build_layout
from pyrev_fl.parser import parse_program
from pyrev_fl.pretty import render_program
from pyrev_fl.slicer import slice_program

EXAMPLES = Path(__file__).parent.parent / "examples"


class SlicerTests(unittest.TestCase):

    def test_slice_removes_dead_code(self):
        """Slicing for y removes statements that only affect x."""
        src = "(x y) (x y) ()\nx += 1;\ny += 2;\nx += 3;\n"
        prog = parse_program(src)
        sliced = slice_program(prog, {"y"})
        rendered = render_program(sliced)
        self.assertIn("y += 2", rendered)
        self.assertNotIn("x += 1", rendered)
        self.assertNotIn("x += 3", rendered)

    def test_slice_keeps_dependencies(self):
        """Slicing for y keeps x += 1 if y depends on x."""
        src = "(x y) (x y) ()\nx += 1;\ny += x;\n"
        prog = parse_program(src)
        sliced = slice_program(prog, {"y"})
        rendered = render_program(sliced)
        self.assertIn("x += 1", rendered)
        self.assertIn("y += x", rendered)

    def test_slice_preserves_semantics(self):
        """Sliced program produces the same value for target variables."""
        src = "(a b c) (a b c) ()\na += 1;\nb += 2;\nc += a;\nc += b;\n"
        prog = parse_program(src)
        sliced = slice_program(prog, {"c"})
        orig = run_program(prog, [0, 0, 0])
        sliced_store = run_program(sliced, [0, 0, 0])
        self.assertEqual(orig["c"], sliced_store["c"])

    def test_slice_all_targets_is_identity(self):
        """Slicing for all outputs keeps all statements."""
        prog = parse_program((EXAMPLES / "copy.srl").read_text())
        sliced = slice_program(prog, {"x", "y"})
        self.assertEqual(render_program(prog), render_program(sliced))

    def test_slice_if_statement(self):
        """Slice through if/fi keeps relevant branches."""
        src = (
            "(x flag) (x y flag) ()\n"
            "if (!= flag 0) then\n  y ^= x;\nelse\nfi (!= flag 0)\n"
        )
        prog = parse_program(src)
        sliced = slice_program(prog, {"y"})
        self.assertIn("if", render_program(sliced))

    def test_cli_slice(self):
        result = subprocess.run(
            [sys.executable, "-m", "pyrev_fl.cli", "slice",
             str(EXAMPLES / "copy.srl"), "--targets", "y"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("y += x", result.stdout)
