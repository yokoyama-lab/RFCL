"""Tests for theoretical properties from Moriyama 2009.

Verifies key theorems about the relationship between SRL/RL,
inversion, and translation (lowering/raising).

1. Inversion-lowering commutativity:
   lower(invert(p)) and rl_invert(lower(p)) are semantically equivalent.
   (Inversion commutes with translation.)

2. Translation equivalence:
   run_srl(p, inputs) == run_rl(lower(p), inputs) for all SRL programs.
   (Lowering preserves semantics.)

3. Round-trip identity:
   run_srl(raise(lower(p)), inputs) == run_srl(p, inputs)
   (Lower then raise preserves semantics.)

4. Double inversion identity:
   invert(invert(p)) == p for both SRL and RL programs.
   (Inversion is an involution.)
"""
from __future__ import annotations

import unittest
from pathlib import Path

from pyrev_fl.interface import build_layout
from pyrev_fl.interpreter import run_program
from pyrev_fl.invert import invert_program
from pyrev_fl.parser import parse_program
from pyrev_fl.pretty import render_program
from pyrev_fl.rl_interpreter import run_program as rl_run_program
from pyrev_fl.rl_invert import invert_program as rl_invert_program
from pyrev_fl.rl_parser import parse_program as rl_parse_program
from pyrev_fl.rl_pretty import render_program as rl_render_program
from pyrev_fl.transform import lower_srl_to_rl, raise_rl_to_srl

EXAMPLES = Path(__file__).parent.parent / "examples"

# All SRL programs with their valid test inputs
SRL_PROGRAMS = {
    "copy.srl": [[5], [0], [-3]],
    "countdown_clean.srl": [[1], [3], [5]],
    "branch_copy.srl": [[3, 1], [5, 0]],  # x!=0 for flag=0
    "bennett_rif.srl": [[0, 2, 3], [1, 5, 3]],
    "fib_bennett.srl": [[1], [3], [5]],
}


class InversionLoweringCommutativity(unittest.TestCase):
    """Theorem: lower(invert(p)) ≡ rl_invert(lower(p)).

    Inversion and lowering commute: inverting an SRL program then lowering
    gives semantically the same RL program as lowering then inverting.
    """

    def _check(self, srl_path: str, inputs: list[int]) -> None:
        p = parse_program((EXAMPLES / srl_path).read_text())
        layout = build_layout(p.inputs, p.outputs, p.temps)

        # Path A: invert SRL, then lower to RL
        inv_srl = invert_program(p)
        path_a = lower_srl_to_rl(inv_srl)

        # Path B: lower to RL, then invert RL
        lowered = lower_srl_to_rl(p)
        path_b = rl_invert_program(lowered)

        # Both should produce the same outputs when run on the OUTPUTS of p
        fwd_store = run_program(p, inputs)
        outputs = [fwd_store[name] for name in layout.outputs]

        inv_layout = build_layout(inv_srl.inputs, inv_srl.outputs, inv_srl.temps)

        store_a = rl_run_program(path_a, outputs)
        store_b = rl_run_program(path_b, outputs)

        for name in inv_layout.outputs:
            self.assertEqual(
                store_a[name], store_b[name],
                f"{srl_path} input={inputs}: {name} differs: "
                f"lower(invert)={store_a[name]} vs rl_invert(lower)={store_b[name]}"
            )

    def test_copy(self):
        for inputs in SRL_PROGRAMS["copy.srl"]:
            with self.subTest(inputs=inputs):
                self._check("copy.srl", inputs)

    def test_countdown(self):
        for inputs in SRL_PROGRAMS["countdown_clean.srl"]:
            with self.subTest(inputs=inputs):
                self._check("countdown_clean.srl", inputs)

    def test_branch_copy(self):
        for inputs in SRL_PROGRAMS["branch_copy.srl"]:
            with self.subTest(inputs=inputs):
                self._check("branch_copy.srl", inputs)

    def test_bennett_rif(self):
        for inputs in SRL_PROGRAMS["bennett_rif.srl"]:
            with self.subTest(inputs=inputs):
                self._check("bennett_rif.srl", inputs)

    def test_fib_bennett(self):
        for inputs in SRL_PROGRAMS["fib_bennett.srl"]:
            with self.subTest(inputs=inputs):
                self._check("fib_bennett.srl", inputs)


class TranslationEquivalence(unittest.TestCase):
    """Theorem: run_srl(p, inputs) == run_rl(lower(p), inputs).

    Lowering preserves program semantics.
    """

    def _check(self, srl_path: str, inputs: list[int]) -> None:
        p = parse_program((EXAMPLES / srl_path).read_text())
        layout = build_layout(p.inputs, p.outputs, p.temps)
        lowered = lower_srl_to_rl(p)

        srl_store = run_program(p, inputs)
        rl_store = rl_run_program(lowered, inputs)

        for name in layout.outputs:
            self.assertEqual(
                srl_store[name], rl_store[name],
                f"{srl_path} input={inputs}: {name} differs"
            )

    def test_all_srl_programs(self):
        """Lowering preserves semantics for all example programs."""
        for srl_path, input_sets in SRL_PROGRAMS.items():
            for inputs in input_sets:
                with self.subTest(program=srl_path, inputs=inputs):
                    self._check(srl_path, inputs)


class RoundTripIdentity(unittest.TestCase):
    """Theorem: run(raise(lower(p)), inputs) == run(p, inputs).

    Lower → raise round-trip preserves semantics.
    """

    def _check(self, srl_path: str, inputs: list[int]) -> None:
        p = parse_program((EXAMPLES / srl_path).read_text())
        layout = build_layout(p.inputs, p.outputs, p.temps)

        lowered = lower_srl_to_rl(p)
        raised = raise_rl_to_srl(lowered)

        orig_store = run_program(p, inputs)
        rt_store = run_program(raised, inputs)

        for name in layout.outputs:
            self.assertEqual(
                orig_store[name], rt_store[name],
                f"{srl_path} input={inputs}: {name} differs"
            )

    def test_all_srl_programs(self):
        """Lower-raise round-trip preserves semantics for all examples."""
        for srl_path, input_sets in SRL_PROGRAMS.items():
            for inputs in input_sets:
                with self.subTest(program=srl_path, inputs=inputs):
                    self._check(srl_path, inputs)


class DoubleInversionIdentity(unittest.TestCase):
    """Theorem: invert(invert(p)) == p (syntactically).

    Double inversion is the identity for both SRL and RL programs.
    """

    def test_srl_double_inversion(self):
        """invert(invert(p)) == p for all SRL example programs."""
        for srl_path in SRL_PROGRAMS:
            with self.subTest(program=srl_path):
                p = parse_program((EXAMPLES / srl_path).read_text())
                double_inv = invert_program(invert_program(p))
                # Compare rendered text (canonical form)
                self.assertEqual(
                    render_program(p),
                    render_program(double_inv),
                    f"{srl_path}: double inversion not identity"
                )

    def test_rl_double_inversion(self):
        """rl_invert(rl_invert(p)) == p for RL example programs."""
        for rl_path in ["copy.rl", "fib_bennett.rl"]:
            with self.subTest(program=rl_path):
                p = rl_parse_program((EXAMPLES / rl_path).read_text())
                double_inv = rl_invert_program(rl_invert_program(p))
                self.assertEqual(
                    rl_render_program(p),
                    rl_render_program(double_inv),
                    f"{rl_path}: double inversion not identity"
                )

    def test_srl_double_inversion_lowered(self):
        """rl_invert(rl_invert(lower(p))) == lower(p) for all SRL programs."""
        for srl_path in SRL_PROGRAMS:
            with self.subTest(program=srl_path):
                p = parse_program((EXAMPLES / srl_path).read_text())
                lowered = lower_srl_to_rl(p)
                double_inv = rl_invert_program(rl_invert_program(lowered))
                self.assertEqual(
                    rl_render_program(lowered),
                    rl_render_program(double_inv),
                    f"{srl_path}: RL double inversion not identity"
                )
