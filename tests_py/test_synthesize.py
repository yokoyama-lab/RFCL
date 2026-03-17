"""Tests for the enumerative SRL program synthesizer."""

from __future__ import annotations

import unittest

from pyrev_fl.interface import build_layout
from pyrev_fl.interpreter import run_program
from pyrev_fl.invert import invert_program
from pyrev_fl.pretty import render_program
from pyrev_fl.synthesize import SynthExample, SynthResult, synthesize


class SynthesizeTests(unittest.TestCase):
    """Core synthesis tests."""

    def test_synthesize_copy(self) -> None:
        """Given (x=5)→(y=5), synthesize y ^= x  or  y += x."""
        result = synthesize(
            input_names=["x"],
            output_names=["x", "y"],
            examples=[SynthExample(inputs=[5], expected_outputs={"x": 5, "y": 5})],
            max_stmts=2,
            timeout=5.0,
        )
        self.assertIsNotNone(result.program)
        assert result.program is not None
        # Verify it actually works
        store = run_program(result.program, [5])
        self.assertEqual(store["y"], 5)
        self.assertEqual(store["x"], 5)
        # Also check a different input
        store2 = run_program(result.program, [10])
        self.assertEqual(store2["y"], 10)

    def test_synthesize_increment(self) -> None:
        """Given (x=0)→(x=1), (x=5)→(x=6), synthesize x += 1."""
        result = synthesize(
            input_names=["x"],
            output_names=["x"],
            examples=[
                SynthExample(inputs=[0], expected_outputs={"x": 1}),
                SynthExample(inputs=[5], expected_outputs={"x": 6}),
            ],
            max_stmts=2,
            timeout=5.0,
        )
        self.assertIsNotNone(result.program)
        assert result.program is not None
        store = run_program(result.program, [7])
        self.assertEqual(store["x"], 8)

    def test_synthesize_swap(self) -> None:
        """Given (x=3,y=5)→(x=5,y=3), synthesize x <=> y."""
        result = synthesize(
            input_names=["x", "y"],
            output_names=["x", "y"],
            examples=[
                SynthExample(inputs=[3, 5], expected_outputs={"x": 5, "y": 3}),
                SynthExample(inputs=[1, 2], expected_outputs={"x": 2, "y": 1}),
            ],
            max_stmts=2,
            timeout=5.0,
        )
        self.assertIsNotNone(result.program)
        assert result.program is not None
        store = run_program(result.program, [10, 20])
        self.assertEqual(store["x"], 20)
        self.assertEqual(store["y"], 10)

    def test_synthesize_negate(self) -> None:
        """Given (x=3,y=0)→(x=3,y=-3), synthesize y -= x."""
        result = synthesize(
            input_names=["x", "y"],
            output_names=["x", "y"],
            examples=[
                SynthExample(inputs=[3, 0], expected_outputs={"x": 3, "y": -3}),
                SynthExample(inputs=[7, 0], expected_outputs={"x": 7, "y": -7}),
            ],
            max_stmts=2,
            timeout=5.0,
        )
        self.assertIsNotNone(result.program)
        assert result.program is not None
        store = run_program(result.program, [5, 0])
        self.assertEqual(store["y"], -5)
        self.assertEqual(store["x"], 5)

    def test_synthesize_add(self) -> None:
        """Given (a=3,b=5)→(a=3,b=8), (a=1,b=2)→(a=1,b=3), synthesize b += a."""
        result = synthesize(
            input_names=["a", "b"],
            output_names=["a", "b"],
            examples=[
                SynthExample(inputs=[3, 5], expected_outputs={"a": 3, "b": 8}),
                SynthExample(inputs=[1, 2], expected_outputs={"a": 1, "b": 3}),
            ],
            max_stmts=2,
            timeout=5.0,
        )
        self.assertIsNotNone(result.program, "should find an addition program")
        assert result.program is not None
        store = run_program(result.program, [4, 10])
        self.assertEqual(store["a"], 4)
        self.assertEqual(store["b"], 14)

    def test_synthesize_impossible(self) -> None:
        """No valid 2-statement program maps (x=0)→(x=1) AND (x=0)→(x=2)."""
        result = synthesize(
            input_names=["x"],
            output_names=["x"],
            examples=[
                SynthExample(inputs=[0], expected_outputs={"x": 1}),
                SynthExample(inputs=[0], expected_outputs={"x": 2}),
            ],
            max_stmts=2,
            timeout=2.0,
        )
        self.assertIsNone(result.program)
        self.assertGreater(result.programs_tested, 0)

    def test_synthesized_programs_are_reversible(self) -> None:
        """All synthesized programs must satisfy the inversion theorem.

        For each example: run(invert(p), outputs(run(p, inputs))) == inputs.
        """
        specs = [
            # (input_names, output_names, examples, description)
            (
                ["x"],
                ["x"],
                [SynthExample(inputs=[0], expected_outputs={"x": 1}),
                 SynthExample(inputs=[5], expected_outputs={"x": 6})],
                "increment",
            ),
            (
                ["x", "y"],
                ["x", "y"],
                [SynthExample(inputs=[3, 5], expected_outputs={"x": 5, "y": 3})],
                "swap",
            ),
            (
                ["x", "y"],
                ["x", "y"],
                [SynthExample(inputs=[3, 0], expected_outputs={"x": 3, "y": -3})],
                "negate",
            ),
        ]

        for input_names, output_names, examples, desc in specs:
            with self.subTest(desc=desc):
                result = synthesize(
                    input_names=input_names,
                    output_names=output_names,
                    examples=examples,
                    max_stmts=2,
                    timeout=5.0,
                )
                self.assertIsNotNone(result.program, f"synthesis failed for {desc}")
                assert result.program is not None

                prog = result.program
                inv = invert_program(prog)

                for ex in examples:
                    layout = build_layout(prog.inputs, prog.outputs, prog.temps)
                    fwd_store = run_program(prog, ex.inputs)
                    output_vals = [fwd_store[name] for name in layout.outputs]

                    inv_layout = build_layout(inv.inputs, inv.outputs, inv.temps)
                    rev_store = run_program(inv, output_vals)
                    recovered = [rev_store[name] for name in inv_layout.outputs]
                    self.assertEqual(recovered, ex.inputs,
                                     f"inversion failed for {desc}")

    def test_synthesize_result_fields(self) -> None:
        """SynthResult contains correct metadata."""
        result = synthesize(
            input_names=["x"],
            output_names=["x"],
            examples=[SynthExample(inputs=[0], expected_outputs={"x": 1})],
            max_stmts=1,
            timeout=5.0,
        )
        self.assertIsNotNone(result.program)
        self.assertGreater(result.programs_tested, 0)
        self.assertGreater(result.time_seconds, 0.0)

    def test_synthesize_deterministic(self) -> None:
        """Same examples produce the same program across runs."""
        examples = [
            SynthExample(inputs=[3, 5], expected_outputs={"x": 5, "y": 3}),
        ]
        r1 = synthesize(["x", "y"], ["x", "y"], examples, max_stmts=2, timeout=5.0)
        r2 = synthesize(["x", "y"], ["x", "y"], examples, max_stmts=2, timeout=5.0)
        self.assertIsNotNone(r1.program)
        self.assertIsNotNone(r2.program)
        assert r1.program is not None and r2.program is not None
        self.assertEqual(render_program(r1.program), render_program(r2.program))

    def test_synthesize_identity(self) -> None:
        """The empty program (identity) is found when outputs == inputs."""
        result = synthesize(
            input_names=["x"],
            output_names=["x"],
            examples=[SynthExample(inputs=[42], expected_outputs={"x": 42})],
            max_stmts=2,
            timeout=5.0,
        )
        self.assertIsNotNone(result.program)
        assert result.program is not None
        self.assertEqual(len(result.program.body.stmts), 0,
                         "identity should be the empty program")

    def test_synthesize_xor_constant(self) -> None:
        """Given x=0→x=2, x=2→x=0, synthesize x ^= 2."""
        result = synthesize(
            input_names=["x"],
            output_names=["x"],
            examples=[
                SynthExample(inputs=[0], expected_outputs={"x": 2}),
                SynthExample(inputs=[2], expected_outputs={"x": 0}),
            ],
            max_stmts=1,
            timeout=5.0,
        )
        self.assertIsNotNone(result.program)
        assert result.program is not None
        store = run_program(result.program, [5])
        self.assertEqual(store["x"], 5 ^ 2)


if __name__ == "__main__":
    unittest.main()
