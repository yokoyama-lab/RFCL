"""Tests for the Landauer principle entropy counter."""
import json
import unittest
from pathlib import Path
from subprocess import run

from pyrev_fl.landauer import analyze_landauer, format_metrics, metrics_to_dict
from pyrev_fl.parser import parse_program
from pyrev_fl.trace import trace_program


class LandauerTests(unittest.TestCase):
    # ------------------------------------------------------------------ #
    # Core property: every SRL program has 0 net entropy                  #
    # ------------------------------------------------------------------ #

    def test_reversible_program_zero_entropy(self) -> None:
        """All well-formed SRL programs must have 0 net entropy change."""
        examples = [
            ("examples/copy.srl", [3]),
            ("examples/countdown_clean.srl", [4]),
            ("examples/fib_bennett.srl", [5]),
            ("examples/branch_copy.srl", [5, 1]),
            ("examples/stack_demo.srl", [3, 7]),
        ]
        for path_str, inputs in examples:
            with self.subTest(program=path_str):
                source = Path(path_str).read_text(encoding="utf-8")
                program = parse_program(source)
                metrics = analyze_landauer(program, inputs)
                self.assertAlmostEqual(
                    metrics.net_entropy,
                    0.0,
                    places=10,
                    msg=f"{path_str}: net entropy should be 0",
                )
                self.assertAlmostEqual(metrics.info_erased_bits, 0.0, places=10)
                self.assertAlmostEqual(metrics.info_created_bits, 0.0, places=10)
                self.assertEqual(metrics.total_steps, metrics.reversible_steps)

    # ------------------------------------------------------------------ #
    # Per-program metrics                                                 #
    # ------------------------------------------------------------------ #

    def test_copy_metrics(self) -> None:
        """copy.srl is a single assignment: 1 step, 0 bits erased."""
        source = Path("examples/copy.srl").read_text(encoding="utf-8")
        program = parse_program(source)
        metrics = analyze_landauer(program, [3])
        # copy.srl: y += x;  — one assignment step
        self.assertEqual(metrics.total_steps, 1)
        self.assertAlmostEqual(metrics.info_erased_bits, 0.0)
        self.assertEqual(len(metrics.step_details), 1)
        self.assertEqual(metrics.step_details[0].kind, "assign")

    def test_countdown_metrics(self) -> None:
        """countdown has multiple steps, all reversible."""
        source = Path("examples/countdown_clean.srl").read_text(encoding="utf-8")
        program = parse_program(source)
        metrics = analyze_landauer(program, [4])
        # The loop runs multiple iterations — must have more than 1 step
        self.assertGreater(metrics.total_steps, 1)
        self.assertEqual(metrics.total_steps, metrics.reversible_steps)
        self.assertAlmostEqual(metrics.info_erased_bits, 0.0)
        # Should have loop_guard steps and assign steps
        kinds = {s.kind for s in metrics.step_details}
        self.assertIn("assign", kinds)
        self.assertIn("loop_guard", kinds)

    def test_fib_bennett_metrics(self) -> None:
        """Complex program (rif + loop), still 0 entropy."""
        source = Path("examples/fib_bennett.srl").read_text(encoding="utf-8")
        program = parse_program(source)
        metrics = analyze_landauer(program, [5])
        self.assertGreater(metrics.total_steps, 5)
        self.assertAlmostEqual(metrics.net_entropy, 0.0)
        self.assertEqual(metrics.total_steps, metrics.reversible_steps)

    # ------------------------------------------------------------------ #
    # Step count matches trace length                                     #
    # ------------------------------------------------------------------ #

    def test_step_count_matches_trace(self) -> None:
        """The number of steps in Landauer analysis matches the trace event count."""
        source = Path("examples/copy.srl").read_text(encoding="utf-8")
        program = parse_program(source)
        metrics = analyze_landauer(program, [3])
        trace_events = trace_program(program, [3])
        self.assertEqual(metrics.total_steps, len(trace_events))

    # ------------------------------------------------------------------ #
    # Irreversible baseline is positive                                   #
    # ------------------------------------------------------------------ #

    def test_irreversible_baseline_positive(self) -> None:
        """The hypothetical irreversible baseline shows positive erasure for
        programs that modify non-zero variables."""
        source = Path("examples/countdown_clean.srl").read_text(encoding="utf-8")
        program = parse_program(source)
        metrics = analyze_landauer(program, [4])
        # countdown modifies acc and n, both non-zero at some point
        self.assertGreater(metrics.irreversible_baseline_bits, 0.0)
        # But the actual SRL program erases nothing
        self.assertAlmostEqual(metrics.info_erased_bits, 0.0)

    # ------------------------------------------------------------------ #
    # Stack operations                                                    #
    # ------------------------------------------------------------------ #

    def test_stack_demo_metrics(self) -> None:
        """Stack operations (push/pop) are reversible: 0 bits erased."""
        source = Path("examples/stack_demo.srl").read_text(encoding="utf-8")
        program = parse_program(source)
        metrics = analyze_landauer(program, [3, 7])
        kinds = [s.kind for s in metrics.step_details]
        self.assertIn("push", kinds)
        self.assertIn("pop", kinds)
        self.assertAlmostEqual(metrics.info_erased_bits, 0.0)
        self.assertEqual(metrics.total_steps, 4)  # 2 push + 2 pop

    # ------------------------------------------------------------------ #
    # Serialization round-trips                                           #
    # ------------------------------------------------------------------ #

    def test_metrics_to_dict_and_format(self) -> None:
        """metrics_to_dict produces a JSON-serializable dict; format_metrics produces text."""
        source = Path("examples/copy.srl").read_text(encoding="utf-8")
        program = parse_program(source)
        metrics = analyze_landauer(program, [3])
        d = metrics_to_dict(metrics)
        self.assertEqual(d["total_steps"], 1)
        self.assertAlmostEqual(d["net_entropy"], 0.0)
        # Should be JSON-serializable
        json.dumps(d)
        # format_metrics produces non-empty text
        text = format_metrics(metrics)
        self.assertIn("Landauer", text)
        self.assertIn("Reversible", text)

    # ------------------------------------------------------------------ #
    # CLI integration                                                     #
    # ------------------------------------------------------------------ #

    def test_cli_landauer_text(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "landauer", "examples/copy.srl", "3"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("Landauer", result.stdout)
        self.assertIn("0.0000", result.stdout)

    def test_cli_landauer_json(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "landauer", "--json", "examples/copy.srl", "3"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["kind"], "landauer")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["total_steps"], 1)
        self.assertAlmostEqual(payload["net_entropy"], 0.0)


if __name__ == "__main__":
    unittest.main()
