"""Property-based tests using randomly generated SRL programs.

Verifies fundamental properties over many random programs:
1. Inversion theorem: run(invert(p), out(run(p, in))) == in
2. Double inversion: invert(invert(p)) == p
3. Translation equivalence: run_srl(p) == run_rl(lower(p))
4. Lower-raise round-trip: run(raise(lower(p))) == run(p)
"""
from __future__ import annotations

import unittest

from pyrev_fl.gen import random_program
from pyrev_fl.interface import build_layout
from pyrev_fl.interpreter import run_program, EvalError
from pyrev_fl.invert import invert_program
from pyrev_fl.pretty import render_program
from pyrev_fl.rl_interpreter import run_program as rl_run_program
from pyrev_fl.transform import lower_srl_to_rl, raise_rl_to_srl

N_SAMPLES = 50


class RandomInversionTests(unittest.TestCase):
    """run(invert(p), outputs(run(p, inputs))) == inputs for random programs."""

    def test_inversion_theorem(self):
        passed = 0
        for seed in range(N_SAMPLES * 3):
            prog = random_program(
                n_inputs=2, n_outputs=2, n_temps=1,
                max_depth=1, max_stmts=3, seed=seed,
            )
            inputs = [seed % 5, (seed * 3 + 1) % 7]
            try:
                layout = build_layout(prog.inputs, prog.outputs, prog.temps)
                fwd = run_program(prog, inputs)
                outputs = [fwd[n] for n in layout.outputs]
                inv = invert_program(prog)
                inv_layout = build_layout(inv.inputs, inv.outputs, inv.temps)
                rev = run_program(inv, outputs)
                recovered = [rev[n] for n in inv_layout.outputs]
                self.assertEqual(recovered, inputs, f"seed={seed}")
                passed += 1
                if passed >= N_SAMPLES:
                    break
            except (EvalError, ZeroDivisionError):
                continue  # skip programs that hit runtime errors
        self.assertGreaterEqual(passed, N_SAMPLES // 2, "too few programs succeeded")


class RandomDoubleInversionTests(unittest.TestCase):
    """invert(invert(p)) == p (syntactic) for random programs."""

    def test_double_inversion_identity(self):
        for seed in range(N_SAMPLES):
            prog = random_program(max_depth=2, max_stmts=3, seed=seed)
            double = invert_program(invert_program(prog))
            self.assertEqual(
                render_program(prog),
                render_program(double),
                f"seed={seed}",
            )


class RandomTranslationTests(unittest.TestCase):
    """run_srl(p) == run_rl(lower(p)) for random programs."""

    def test_translation_equivalence(self):
        passed = 0
        for seed in range(N_SAMPLES * 3):
            prog = random_program(
                n_inputs=2, n_outputs=2, n_temps=1,
                max_depth=1, max_stmts=3, seed=seed,
            )
            inputs = [seed % 5, (seed * 3 + 1) % 7]
            try:
                layout = build_layout(prog.inputs, prog.outputs, prog.temps)
                srl_store = run_program(prog, inputs)
                rl_prog = lower_srl_to_rl(prog)
                rl_store = rl_run_program(rl_prog, inputs)
                for name in layout.outputs:
                    self.assertEqual(
                        srl_store[name], rl_store[name],
                        f"seed={seed}, var={name}",
                    )
                passed += 1
                if passed >= N_SAMPLES:
                    break
            except (EvalError, ZeroDivisionError):
                continue
        self.assertGreaterEqual(passed, N_SAMPLES // 2)


class RandomRoundTripTests(unittest.TestCase):
    """run(raise(lower(p))) == run(p) for random programs."""

    def test_lower_raise_round_trip(self):
        passed = 0
        for seed in range(N_SAMPLES * 3):
            prog = random_program(
                n_inputs=2, n_outputs=2, n_temps=1,
                max_depth=1, max_stmts=3, seed=seed,
            )
            inputs = [seed % 5, (seed * 3 + 1) % 7]
            try:
                layout = build_layout(prog.inputs, prog.outputs, prog.temps)
                orig = run_program(prog, inputs)
                raised = raise_rl_to_srl(lower_srl_to_rl(prog))
                rt = run_program(raised, inputs)
                for name in layout.outputs:
                    self.assertEqual(orig[name], rt[name], f"seed={seed}")
                passed += 1
                if passed >= N_SAMPLES:
                    break
            except (EvalError, ZeroDivisionError, ValueError):
                continue
        self.assertGreaterEqual(passed, N_SAMPLES // 2)
