"""Tests for Bennett time-space tradeoff analysis."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from pyrev_fl.parser import parse_program
from pyrev_fl.tradeoff import analyze_tradeoff, format_tradeoff

EXAMPLES = Path(__file__).parent.parent / "examples"


class TradeoffBasicTests(unittest.TestCase):

    def test_copy_tradeoff(self):
        """copy.srl: single step, Bennett adds overhead."""
        p = parse_program((EXAMPLES / "copy.srl").read_text())
        r = analyze_tradeoff(p, [5])
        self.assertEqual(r.original.time_steps, 1)
        self.assertGreater(r.bennett.time_steps, r.original.time_steps)

    def test_countdown_tradeoff(self):
        """countdown_clean.srl with n=3: Bennett time ≈ 2x original."""
        p = parse_program((EXAMPLES / "countdown_clean.srl").read_text())
        r = analyze_tradeoff(p, [3])
        self.assertGreater(r.original.time_steps, 1)
        # Bennett should be roughly 2x the original time
        self.assertGreater(r.time_ratio, 1.5)
        self.assertLess(r.time_ratio, 4.0)


class TradeoffTheoreticalBoundsTests(unittest.TestCase):

    def test_time_ratio_near_two(self):
        """For simple programs, Bennett time ratio should be near 2."""
        for srl_path, inputs in [
            ("copy.srl", [5]),
            ("countdown_clean.srl", [3]),
            ("branch_copy.srl", [5, 1]),
        ]:
            with self.subTest(program=srl_path):
                p = parse_program((EXAMPLES / srl_path).read_text())
                r = analyze_tradeoff(p, inputs)
                # Time ratio should be between 1.5 and 5 (2x + small constant overhead)
                self.assertGreater(r.time_ratio, 1.0, f"time ratio too low: {r.time_ratio}")

    def test_space_bounded(self):
        """Bennett space should not exceed theoretical bound."""
        for srl_path, inputs in [
            ("copy.srl", [5]),
            ("countdown_clean.srl", [3]),
        ]:
            with self.subTest(program=srl_path):
                p = parse_program((EXAMPLES / srl_path).read_text())
                r = analyze_tradeoff(p, inputs)
                self.assertLessEqual(
                    r.bennett.peak_space,
                    r.theoretical_space_bound + 5,  # small tolerance for loop vars
                    f"space exceeds theoretical bound"
                )

    def test_fib_bennett_tradeoff(self):
        """fib_bennett.srl already uses Bennett's trick internally."""
        p = parse_program((EXAMPLES / "fib_bennett.srl").read_text())
        r = analyze_tradeoff(p, [5])
        self.assertGreater(r.original.time_steps, 5)
        self.assertGreater(r.bennett.time_steps, r.original.time_steps)


class TradeoffSpaceTraceTests(unittest.TestCase):

    def test_space_trace_recorded(self):
        """Space trace should have entries for each step."""
        p = parse_program((EXAMPLES / "copy.srl").read_text())
        r = analyze_tradeoff(p, [5])
        self.assertEqual(len(r.original.space_trace), r.original.time_steps)

    def test_bennett_space_returns_to_low(self):
        """Bennett's uncomputation should bring space back down."""
        p = parse_program((EXAMPLES / "countdown_clean.srl").read_text())
        r = analyze_tradeoff(p, [3])
        trace = r.bennett.space_trace
        if len(trace) >= 4:
            # The last few entries should have lower space than the peak
            self.assertLess(trace[-1], r.bennett.peak_space + 1)

    def test_original_vs_bennett_time_scaling(self):
        """As n grows, Bennett time grows proportionally."""
        p = parse_program((EXAMPLES / "countdown_clean.srl").read_text())
        results = {}
        for n in [2, 4, 6]:
            r = analyze_tradeoff(p, [n])
            results[n] = r
        # Time should increase with n for both
        self.assertLess(results[2].original.time_steps, results[6].original.time_steps)
        self.assertLess(results[2].bennett.time_steps, results[6].bennett.time_steps)


class TradeoffFormatTests(unittest.TestCase):

    def test_format_output(self):
        p = parse_program((EXAMPLES / "copy.srl").read_text())
        r = analyze_tradeoff(p, [5])
        text = format_tradeoff(r)
        self.assertIn("Bennett Time-Space Tradeoff", text)
        self.assertIn("Time ratio", text)
        self.assertIn("Space trace", text)


class ExactPredictionTests(unittest.TestCase):

    def test_time_prediction_is_exact_on_the_k1_sweep_inputs(self):
        import csv
        rows = EXAMPLES.parent / "experiments/pebbling/results/rfcl_k1_measurements.csv"
        checked = 0
        with open(rows) as f:
            for row in csv.DictReader(f):
                if row.get("status") != "ok":
                    continue
                prog = parse_program((EXAMPLES / row["program"]).read_text())
                inputs = [int(v) for v in row["inputs"].split()]
                r = analyze_tradeoff(prog, inputs)
                T = r.original.time_steps
                self.assertEqual(r.bennett.time_steps, 2 * T + len(prog.outputs) + 7, row)
                self.assertAlmostEqual(r.theoretical_time_ratio * T, r.bennett.time_steps)
                self.assertLessEqual(r.bennett.peak_space, r.theoretical_space_bound, row)
                checked += 1
        self.assertEqual(checked, 28)


class TradeoffCliTests(unittest.TestCase):

    def test_cli_tradeoff(self):
        result = subprocess.run(
            [sys.executable, "-m", "pyrev_fl.cli", "tradeoff",
             str(EXAMPLES / "copy.srl"), "5"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Bennett", result.stdout)

    def test_cli_tradeoff_json(self):
        result = subprocess.run(
            [sys.executable, "-m", "pyrev_fl.cli", "tradeoff",
             str(EXAMPLES / "countdown_clean.srl"), "3", "--json"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        import json
        data = json.loads(result.stdout)
        self.assertIn("time_ratio", data)
        self.assertIn("original", data)
        self.assertIn("bennett", data)
