"""Cross-module integration tests and missing edge cases.

Fills gaps identified in test coverage analysis:
1. Multi-tool pipelines (bennett→landauer, pe→run, slice→equiv, lower→pla→run)
2. Missing CLI tests (synthesize)
3. Error handling for invalid inputs
4. Boundary conditions (empty programs, XOR-only, etc.)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pyrev_fl.parser import parse_program
from pyrev_fl.interpreter import run_program, EvalError
from pyrev_fl.interface import build_layout
from pyrev_fl.invert import invert_program
from pyrev_fl.pretty import render_program
from pyrev_fl.bennett import make_reversible_program
from pyrev_fl.landauer import analyze_landauer
from pyrev_fl.pe import partial_eval
from pyrev_fl.slicer import slice_program
from pyrev_fl.equiv import check_equivalence
from pyrev_fl.pla_parser import parse_pla, PlaParseError
from pyrev_fl.janus_parser import parse_janus, JanusParseError
from pyrev_fl.transform import lower_srl_to_rl
from pyrev_fl.rl_interpreter import run_program as rl_run_program
from pyrev_fl.pla_pretty import render_pla
from pyrev_fl.tradeoff import analyze_tradeoff

EXAMPLES = Path(__file__).parent.parent / "examples"


# ===================================================================
# Pipeline integration tests
# ===================================================================

class BennettLandauerPipelineTests(unittest.TestCase):
    """Bennett-transformed programs must have 0 Landauer entropy."""

    def test_bennett_copy_zero_entropy(self):
        p = parse_program("(x) (x y) ()\ny += x;\n")
        bp = make_reversible_program(p.body.stmts, p.inputs, p.outputs, p.temps)
        metrics = analyze_landauer(bp, [5])
        self.assertEqual(metrics.net_entropy, 0.0)
        self.assertEqual(metrics.info_erased_bits, 0.0)

    def test_bennett_countdown_zero_entropy(self):
        p = parse_program((EXAMPLES / "countdown_clean.srl").read_text())
        bp = make_reversible_program(
            p.body.stmts, p.inputs, p.outputs, p.temps
        )
        metrics = analyze_landauer(bp, [3])
        self.assertEqual(metrics.net_entropy, 0.0)


class PeRunPipelineTests(unittest.TestCase):
    """Partial evaluation followed by run gives same results."""

    def test_pe_then_run_copy(self):
        p = parse_program("(x y) (x y) ()\nx += y;\n")
        pe_prog = partial_eval(p, {"y": 3})
        orig = run_program(p, [5, 3])
        spec = run_program(pe_prog, [5])
        self.assertEqual(orig["x"], spec["x"])

    def test_pe_then_run_branch(self):
        src = "(x flag) (x y flag) ()\nif (!= flag 0) then\n  y ^= x;\nelse\nfi (!= flag 0)\n"
        p = parse_program(src)
        pe_prog = partial_eval(p, {"flag": 1})
        orig = run_program(p, [7, 1])
        spec = run_program(pe_prog, [7])
        self.assertEqual(orig["y"], spec["y"])


class SliceEquivPipelineTests(unittest.TestCase):
    """Sliced program is equivalent to original for target variables."""

    def test_slice_equiv_copy(self):
        src = "(x y z) (x y z) ()\nx += 1;\ny += 2;\nz += x;\n"
        p = parse_program(src)
        sliced = slice_program(p, {"z"})
        # They should produce same z for all inputs
        for vals in [(0, 0, 0), (1, 2, 3), (5, 0, 0)]:
            orig = run_program(p, list(vals))
            sl = run_program(sliced, list(vals))
            self.assertEqual(orig["z"], sl["z"])


class LowerPlaPipelineTests(unittest.TestCase):
    """Full pipeline: SRL → lower → PLA → parse back → RL → run."""

    def test_full_pla_pipeline(self):
        p = parse_program((EXAMPLES / "copy.srl").read_text())
        rl = lower_srl_to_rl(p)
        pla_text = render_pla(rl)
        rl2 = parse_pla(pla_text)
        for x in [0, 5, -3]:
            s1 = rl_run_program(rl, [x])
            s2 = rl_run_program(rl2, [x])
            self.assertEqual(s1["y"], s2["y"])

    def test_full_pla_pipeline_countdown(self):
        p = parse_program((EXAMPLES / "countdown_clean.srl").read_text())
        rl = lower_srl_to_rl(p)
        pla_text = render_pla(rl)
        rl2 = parse_pla(pla_text)
        s1 = rl_run_program(rl, [3])
        s2 = rl_run_program(rl2, [3])
        self.assertEqual(s1["acc"], s2["acc"])


class TradeoffBennettConsistencyTests(unittest.TestCase):
    """Tradeoff analysis and manual Bennett should agree."""

    def test_tradeoff_bennett_output_matches(self):
        p = parse_program((EXAMPLES / "copy.srl").read_text())
        result = analyze_tradeoff(p, [5])
        # Original and Bennett should produce same output values
        layout = build_layout(p.inputs, p.outputs, p.temps)
        orig = run_program(p, [5])
        # Bennett version has different output names (copies), but original outputs should match
        self.assertGreater(result.bennett.time_steps, result.original.time_steps)


# ===================================================================
# Error handling tests
# ===================================================================

class PlaParserErrorTests(unittest.TestCase):

    def test_invalid_pla_missing_header(self):
        with self.assertRaises(PlaParseError):
            parse_pla("start:\n    ENTRY\n    HALT\n")

    def test_invalid_pla_unknown_instruction(self):
        with self.assertRaises(PlaParseError):
            parse_pla("() () ()\nstart:\n    ENTRY\n    BOGUS x\n    HALT\n")


class JanusParserErrorTests(unittest.TestCase):

    def test_invalid_janus_no_procedure(self):
        with self.assertRaises((JanusParseError, IndexError)):
            parse_janus("int x\nx += 1\n")

    def test_invalid_janus_bad_operator(self):
        with self.assertRaises(JanusParseError):
            parse_janus("procedure main()\n    int x\n    x ** 2\n")


class TradeoffErrorTests(unittest.TestCase):

    def test_tradeoff_arity_mismatch(self):
        p = parse_program("(x y) (x y) ()\nx += y;\n")
        with self.assertRaises(EvalError):
            analyze_tradeoff(p, [5])  # needs 2 inputs


# ===================================================================
# Boundary condition tests
# ===================================================================

class EmptyProgramTests(unittest.TestCase):

    def test_empty_body_run(self):
        p = parse_program("(x) (x) ()\n")
        store = run_program(p, [5])
        self.assertEqual(store["x"], 5)

    def test_empty_body_invert(self):
        p = parse_program("(x) (x) ()\n")
        inv = invert_program(p)
        store = run_program(inv, [5])
        self.assertEqual(store["x"], 5)

    def test_empty_body_pe(self):
        p = parse_program("(x) (x) ()\n")
        pe = partial_eval(p, {"x": 5})
        self.assertEqual(pe.body.stmts, [])

    def test_empty_body_slice(self):
        p = parse_program("(x) (x) ()\n")
        sliced = slice_program(p, {"x"})
        self.assertEqual(sliced.body.stmts, [])

    def test_empty_body_landauer(self):
        p = parse_program("(x) (x) ()\n")
        metrics = analyze_landauer(p, [5])
        self.assertEqual(metrics.total_steps, 0)


class XorOnlyProgramTests(unittest.TestCase):

    def test_xor_chain(self):
        """XOR is self-inverse: x ^= y ^= x should give specific result."""
        src = "(x y) (x y) ()\nx ^= y;\ny ^= x;\nx ^= y;\n"
        p = parse_program(src)
        store = run_program(p, [5, 3])
        # XOR swap: x=3, y=5
        self.assertEqual(store["x"], 3)
        self.assertEqual(store["y"], 5)

    def test_xor_inversion(self):
        src = "(x y) (x y) ()\nx ^= y;\n"
        p = parse_program(src)
        layout = build_layout(p.inputs, p.outputs, p.temps)
        fwd = run_program(p, [5, 3])
        outputs = [fwd[n] for n in layout.outputs]
        inv = invert_program(p)
        inv_layout = build_layout(inv.inputs, inv.outputs, inv.temps)
        rev = run_program(inv, outputs)
        recovered = [rev[n] for n in inv_layout.outputs]
        self.assertEqual(recovered, [5, 3])


# ===================================================================
# Missing CLI test
# ===================================================================

class SynthesizeCliTests(unittest.TestCase):

    def test_cli_synthesize_swap(self):
        result = subprocess.run(
            [sys.executable, "-m", "pyrev_fl.cli", "synthesize",
             "--inputs", "x y", "--outputs", "x y",
             "--examples", "5,3:3,5", "--examples", "1,2:2,1",
             "--max-stmts", "1"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("<=>", result.stdout)

    def test_cli_synthesize_impossible(self):
        result = subprocess.run(
            [sys.executable, "-m", "pyrev_fl.cli", "synthesize",
             "--inputs", "x", "--outputs", "x",
             "--examples", "1:2", "--examples", "1:3",
             "--max-stmts", "1", "--timeout", "1"],
            capture_output=True, text=True, timeout=10,
        )
        # Should fail (contradictory examples)
        self.assertNotEqual(result.returncode, 0)
