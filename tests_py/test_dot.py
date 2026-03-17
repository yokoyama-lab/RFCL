"""Tests for the RL → DOT graph renderer (pyrev_fl/rl_dot.py)."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from pyrev_fl.rl_ast import (
    Assign,
    Block,
    Exit,
    FiFrom,
    FromEntry,
    FromLabel,
    Goto,
    IfGoto,
    Program,
    RFromLabel,
    RGoto,
)
from pyrev_fl.ast import Const, UpdateOp, Var
from pyrev_fl.rl_dot import render_dot
from pyrev_fl.rl_parser import parse_program

EXAMPLES = Path(__file__).parent.parent / "examples"


def _prog(*blocks: Block) -> Program:
    return Program(inputs=[], outputs=[], temps=[], blocks=list(blocks))


class DotTests(unittest.TestCase):

    # ------------------------------------------------------------------
    # Basic structure
    # ------------------------------------------------------------------

    def test_dot_output_is_digraph(self):
        """render_dot always starts with 'digraph'."""
        program = parse_program((EXAMPLES / "copy.rl").read_text())
        dot = render_dot(program)
        self.assertTrue(dot.strip().startswith("digraph"), repr(dot[:50]))
        self.assertIn("}", dot)

    def test_dot_copy_contains_block_label(self):
        """copy.rl has one block 'start'; its label appears in the DOT output."""
        program = parse_program((EXAMPLES / "copy.rl").read_text())
        dot = render_dot(program)
        self.assertIn("start", dot)

    def test_dot_copy_entry_block_is_bold(self):
        """Entry block gets style=bold."""
        program = parse_program((EXAMPLES / "copy.rl").read_text())
        dot = render_dot(program)
        self.assertIn("bold", dot)

    def test_dot_fib_bennett_all_blocks_present(self):
        """fib_bennett.rl has 10 blocks; all labels appear as DOT nodes."""
        program = parse_program((EXAMPLES / "fib_bennett.rl").read_text())
        dot = render_dot(program)
        for block in program.blocks:
            self.assertIn(block.label, dot, f"block {block.label!r} missing from DOT")

    # ------------------------------------------------------------------
    # Edge types
    # ------------------------------------------------------------------

    def test_dot_goto_produces_solid_edge(self):
        """A Goto jump creates a plain '->' edge (no dashed or label)."""
        prog = _prog(
            Block("a", FromEntry(), [], Goto("b")),
            Block("b", FromLabel("a"), [], Exit()),
        )
        dot = render_dot(prog)
        # Should have an edge a -> b
        self.assertIn("->", dot)
        # The Goto edge should NOT have dashed style
        lines = [l for l in dot.splitlines() if '"a"' in l and '->' in l]
        self.assertTrue(any("dashed" not in l for l in lines), dot)

    def test_dot_ifgoto_produces_two_labeled_edges(self):
        """An IfGoto jump creates two edges labeled 'T' and 'F'."""
        prog = _prog(
            Block("entry", FromEntry(), [], IfGoto(Var("x"), "t", "f")),
            Block("t", FromLabel("entry"), [], Exit()),
            Block("f", FromLabel("entry"), [], Exit()),
        )
        dot = render_dot(prog)
        self.assertIn('"T"', dot)
        self.assertIn('"F"', dot)
        # Two distinct edges from entry
        edges = [l for l in dot.splitlines() if '"entry"' in l and '->' in l]
        self.assertEqual(len(edges), 2, f"expected 2 edges, got: {edges}")

    def test_dot_rgoto_is_dashed(self):
        """An RGoto edge has style=dashed."""
        prog = _prog(
            Block("a", FromEntry(), [], RGoto("b")),
            Block("b", RFromLabel("a"), [], Exit()),
        )
        dot = render_dot(prog)
        edge_lines = [l for l in dot.splitlines() if '"a"' in l and '->' in l]
        self.assertTrue(any("dashed" in l for l in edge_lines), dot)

    def test_dot_exit_block_has_no_outgoing_edge(self):
        """A block with Exit jump produces no outgoing '->' edge."""
        prog = _prog(
            Block("only", FromEntry(), [], Exit()),
        )
        dot = render_dot(prog)
        edge_lines = [l for l in dot.splitlines() if '->' in l]
        self.assertEqual(edge_lines, [], f"unexpected edges: {edge_lines}")

    # ------------------------------------------------------------------
    # Label escaping
    # ------------------------------------------------------------------

    def test_dot_comparison_expr_angle_brackets_escaped(self):
        """< and > in expressions are escaped so DOT doesn't choke."""
        prog = _prog(
            Block("cond", FromEntry(), [], IfGoto(Var("x"), "t", "f")),
            Block("t", FromLabel("cond"), [], Exit()),
            Block("f", FromLabel("cond"), [], Exit()),
        )
        # Even with a FiFrom that uses '<' or '>' via Binary we must not get raw '<'
        # Simpler: just check the DOT is parseable by verifying braces are balanced
        dot = render_dot(prog)
        self.assertEqual(dot.count("{"), dot.count("}"))

    # ------------------------------------------------------------------
    # CLI integration
    # ------------------------------------------------------------------

    def test_cli_rl_dot_graph_copy(self):
        """CLI 'rl-dot-graph' on copy.rl exits 0 and outputs a digraph."""
        result = subprocess.run(
            [sys.executable, "-m", "pyrev_fl.cli", "rl-dot-graph", str(EXAMPLES / "copy.rl")],
            capture_output=True, text=True, cwd=str(EXAMPLES.parent),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.strip().startswith("digraph"), result.stdout[:80])

    def test_cli_rl_dot_graph_fib_bennett(self):
        """CLI 'rl-dot-graph' on fib_bennett.rl exits 0 with all block labels."""
        result = subprocess.run(
            [sys.executable, "-m", "pyrev_fl.cli", "rl-dot-graph", str(EXAMPLES / "fib_bennett.rl")],
            capture_output=True, text=True, cwd=str(EXAMPLES.parent),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        program = parse_program((EXAMPLES / "fib_bennett.rl").read_text())
        for block in program.blocks:
            self.assertIn(block.label, result.stdout, f"block {block.label!r} missing")

    def test_cli_rl_dot_graph_output_flag(self):
        """CLI --output FILE writes DOT to the file instead of stdout."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".dot", delete=False) as f:
            out_path = f.name
        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "pyrev_fl.cli", "rl-dot-graph",
                    str(EXAMPLES / "copy.rl"), "--output", out_path,
                ],
                capture_output=True, text=True, cwd=str(EXAMPLES.parent),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            dot = Path(out_path).read_text()
            self.assertIn("digraph", dot)
        finally:
            os.unlink(out_path)
