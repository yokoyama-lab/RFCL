"""Tests for the general RL → SRL raiser (Phase 2.2).

Tests hand-written RL programs that the original pattern-matching raiser
could not handle (multi-assignment blocks, compact structures).
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from pyrev_fl.interface import build_layout
from pyrev_fl.interpreter import run_program
from pyrev_fl.rl_interpreter import run_program as rl_run_program
from pyrev_fl.rl_parser import parse_program as rl_parse_program
from pyrev_fl.pretty import render_program
from pyrev_fl.transform import raise_rl_to_srl, lower_srl_to_rl, _build_cfg, _compute_idom

EXAMPLES = Path(__file__).parent.parent / "examples"


class DominatorTests(unittest.TestCase):
    """Test dominator computation and CFG analysis."""

    def test_linear_chain_dominators(self):
        """In a linear chain A→B→C, A dominates B and C, B dominates C."""
        rl = rl_parse_program((EXAMPLES / "hand_copy_multi.rl").read_text())
        block_map = {b.label: b for b in rl.blocks}
        cfg = _build_cfg(block_map)
        # start is the only block; it dominates itself
        self.assertTrue(cfg.dominates("start", "start"))

    def test_if_diamond_dominators(self):
        """In an if/fi diamond, the entry dominates all blocks."""
        rl = rl_parse_program((EXAMPLES / "hand_if_multi.rl").read_text())
        block_map = {b.label: b for b in rl.blocks}
        cfg = _build_cfg(block_map)
        for label in block_map:
            self.assertTrue(cfg.dominates("start", label), f"start should dominate {label}")
        # join is post-dominated by start but tbranch does NOT dominate ebranch
        self.assertFalse(cfg.dominates("tbranch", "ebranch"))
        self.assertFalse(cfg.dominates("ebranch", "tbranch"))

    def test_loop_back_edges(self):
        """Loop programs have a back-edge from back → header."""
        rl = rl_parse_program((EXAMPLES / "hand_countdown.rl").read_text())
        block_map = {b.label: b for b in rl.blocks}
        cfg = _build_cfg(block_map)
        back_edges = cfg.back_edges()
        # back → loop is a back-edge (loop dominates back)
        self.assertIn(("back", "loop"), back_edges)

    def test_natural_loop_body(self):
        """Natural loop body includes header, step, back, and the exit test."""
        rl = rl_parse_program((EXAMPLES / "hand_countdown.rl").read_text())
        block_map = {b.label: b for b in rl.blocks}
        cfg = _build_cfg(block_map)
        body = cfg.natural_loop_body("loop", "back")
        self.assertIn("loop", body)
        self.assertIn("step", body)
        self.assertIn("back", body)
        # done is outside the loop
        self.assertNotIn("done", body)
        self.assertNotIn("init", body)


class MultiAssignRaisingTests(unittest.TestCase):
    """Test raising RL programs with multiple assignments per block."""

    def test_raise_multi_assign_copy(self):
        """hand_copy_multi.rl: single block with 2 assigns → raises to SRL."""
        rl = rl_parse_program((EXAMPLES / "hand_copy_multi.rl").read_text())
        srl = raise_rl_to_srl(rl)
        rendered = render_program(srl)
        self.assertIn("y ^= x", rendered)
        self.assertIn("z ^= x", rendered)

    def test_multi_assign_copy_semantics(self):
        """Raised SRL produces same outputs as the original RL."""
        rl = rl_parse_program((EXAMPLES / "hand_copy_multi.rl").read_text())
        for x in [0, 3, 7, -2]:
            with self.subTest(x=x):
                rl_store = rl_run_program(rl, [x])
                srl = raise_rl_to_srl(rl)
                srl_store = run_program(srl, [x])
                layout = build_layout(rl.inputs, rl.outputs, rl.temps)
                for name in layout.outputs:
                    self.assertEqual(srl_store[name], rl_store[name])

    def test_raise_multi_assign_if_branches(self):
        """hand_if_multi.rl: if/fi with multi-assign branches → raises to SRL."""
        rl = rl_parse_program((EXAMPLES / "hand_if_multi.rl").read_text())
        srl = raise_rl_to_srl(rl)
        rendered = render_program(srl)
        self.assertIn("if", rendered)
        self.assertIn("fi", rendered)

    def test_multi_assign_if_semantics(self):
        """Raised SRL from hand_if_multi.rl matches RL outputs."""
        rl = rl_parse_program((EXAMPLES / "hand_if_multi.rl").read_text())
        srl = raise_rl_to_srl(rl)
        layout = build_layout(rl.inputs, rl.outputs, rl.temps)
        for x, flag in [(3, 1), (5, 0), (0, 1), (1, 0)]:
            with self.subTest(x=x, flag=flag):
                rl_store = rl_run_program(rl, [x, flag])
                srl_store = run_program(srl, [x, flag])
                for name in layout.outputs:
                    self.assertEqual(srl_store[name], rl_store[name])


class CompactLoopRaisingTests(unittest.TestCase):
    """Test raising compact hand-written loop RL programs."""

    def test_raise_compact_countdown(self):
        """hand_countdown.rl: compact loop with multi-assign body → raises to SRL."""
        rl = rl_parse_program((EXAMPLES / "hand_countdown.rl").read_text())
        srl = raise_rl_to_srl(rl)
        rendered = render_program(srl)
        self.assertIn("from", rendered)
        self.assertIn("until", rendered)

    def test_compact_countdown_semantics(self):
        """Raised SRL from hand_countdown.rl matches RL outputs."""
        rl = rl_parse_program((EXAMPLES / "hand_countdown.rl").read_text())
        srl = raise_rl_to_srl(rl)
        layout = build_layout(rl.inputs, rl.outputs, rl.temps)
        for n in [1, 2, 3, 5]:
            with self.subTest(n=n):
                rl_store = rl_run_program(rl, [n])
                srl_store = run_program(srl, [n])
                for name in layout.outputs:
                    self.assertEqual(srl_store[name], rl_store[name], f"{name}")

    def test_compact_countdown_matches_original_srl(self):
        """hand_countdown.rl should raise to equivalent of countdown_clean.srl."""
        from pyrev_fl.parser import parse_program as srl_parse
        original = srl_parse((EXAMPLES / "countdown_clean.srl").read_text())
        rl = rl_parse_program((EXAMPLES / "hand_countdown.rl").read_text())
        raised = raise_rl_to_srl(rl)
        # Same semantic results
        for n in [1, 3, 5]:
            with self.subTest(n=n):
                orig_store = run_program(original, [n])
                raised_store = run_program(raised, [n])
                self.assertEqual(orig_store["acc"], raised_store["acc"])


class RoundTripGeneralTests(unittest.TestCase):
    """Verify lower → raise round-trips for programs that exercise the general raiser."""

    def test_lowered_then_raised_copy_multi(self):
        """Raise hand_copy_multi.rl, lower it back to RL, run both, compare."""
        rl = rl_parse_program((EXAMPLES / "hand_copy_multi.rl").read_text())
        srl = raise_rl_to_srl(rl)
        rl2 = lower_srl_to_rl(srl)
        for x in [0, 5, -3]:
            with self.subTest(x=x):
                s1 = rl_run_program(rl, [x])
                s2 = rl_run_program(rl2, [x])
                layout = build_layout(rl.inputs, rl.outputs, rl.temps)
                for name in layout.outputs:
                    self.assertEqual(s1[name], s2[name])


class CliGeneralRaiserTests(unittest.TestCase):
    """CLI integration tests for general raiser."""

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "pyrev_fl.cli", *args],
            capture_output=True, text=True, cwd=str(EXAMPLES.parent),
        )

    def test_cli_raise_hand_copy_multi(self):
        result = self._run("raise-rl-to-srl", str(EXAMPLES / "hand_copy_multi.rl"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("y ^= x", result.stdout)
        self.assertIn("z ^= x", result.stdout)

    def test_cli_raise_hand_countdown(self):
        result = self._run("raise-rl-to-srl", str(EXAMPLES / "hand_countdown.rl"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("from", result.stdout)
        self.assertIn("until", result.stdout)

    def test_cli_raise_hand_if_multi(self):
        result = self._run("raise-rl-to-srl", str(EXAMPLES / "hand_if_multi.rl"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("if", result.stdout)
        self.assertIn("fi", result.stdout)

    def test_cli_run_hand_countdown(self):
        """rl-run on hand_countdown.rl gives same result as countdown_clean.srl."""
        r1 = self._run("run", str(EXAMPLES / "countdown_clean.srl"), "3")
        r2 = self._run("rl-run", str(EXAMPLES / "hand_countdown.rl"), "3")
        self.assertEqual(r1.returncode, 0)
        self.assertEqual(r2.returncode, 0)
        self.assertEqual(r1.stdout.strip(), r2.stdout.strip())
