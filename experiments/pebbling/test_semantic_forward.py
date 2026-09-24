"""Checks of semantic_forward.py on hand-made cases (needs PyJanus; skips otherwise).

Run from this directory:  PYJANUS=/path/to/PyJanus python3 -m unittest test_semantic_forward
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = HERE / "semantic_cases"
sys.path.insert(0, str(HERE))


def _pyjanus_available() -> bool:
    env = os.environ.get("PYJANUS")
    parent = HERE.parent.parent.parent
    return bool(env and Path(env, "jana_py").is_dir()) or any(
        (parent / d / "jana_py").is_dir() for d in ("PyJanus", "pyjanus"))


@unittest.skipUnless(_pyjanus_available(), "PyJanus checkout not found")
class SemanticForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import measure_corpus as mc
        import semantic_forward as sf
        mc.MeasuringRuntime = sf.SemanticRuntime
        cls.mc = mc

    def run_case(self, name: str) -> dict:
        _, res = self.mc.measure_file(str(CASES / name), ["jana2014"], 30, False)
        self.assertEqual(res["status"], "ok", res.get("error"))
        return res

    def test_ccu_pairs_the_uncall(self):
        r = self.run_case("ccu.j")
        # 1 init + 4 call + 1 copy + 4 uncall; the uncall (body + statement) is reverse
        self.assertEqual((r["total_steps"], r["sem_uncompute_steps"], r["n_paired"]), (10, 4, 1))

    def test_inverse_as_main_is_all_forward(self):
        r = self.run_case("inverse_main.j")
        self.assertEqual(r["sem_uncompute_steps"], 0)
        self.assertEqual(r["n_unpaired_uncalls"], 1)
        self.assertGreater(r["time_ratio"], 2)          # the syntactic misreading
        self.assertEqual(r["time_ratio_sem"], 1.0)

    def test_call_that_undoes_an_uncall_is_reverse_work(self):
        r = self.run_case("reversed_ccu.j")
        self.assertEqual((r["n_paired"], r["n_paired_calls"]), (1, 1))
        self.assertEqual(r["sem_uncompute_steps"], 4)

    def test_uncall_on_other_data_is_not_an_uncomputation(self):
        r = self.run_case("not_inverse.j")
        self.assertEqual((r["n_paired"], r["sem_uncompute_steps"]), (0, 0))
        self.assertEqual(r["n_partial"], 0)

    def test_continuing_call_is_not_an_uncomputation(self):
        r = self.run_case("repeat.j")
        self.assertEqual((r["n_paired"], r["sem_uncompute_steps"]), (0, 0))
        self.assertEqual((r["n_partial"], r["sem_uncompute_steps_loose"]), (0, 0))

    def test_garbage_clearing_uncall_is_partial(self):
        r = self.run_case("sort_idiom.j")
        self.assertEqual((r["n_paired"], r["n_partial"]), (0, 1))
        self.assertEqual(r["sem_uncompute_steps"], 0)
        self.assertEqual(r["sem_uncompute_steps_loose"], 3)   # 2 statements + the uncall

    def test_strict_pairs_count_in_loose_too(self):
        r = self.run_case("ccu.j")
        self.assertEqual(r["sem_uncompute_steps_loose"], r["sem_uncompute_steps"])

    def test_literal_to_constant_parameter_still_pairs(self):
        r = self.run_case("const_arg.j")
        self.assertEqual((r["n_paired"], r["n_partial"]), (1, 0))

    def test_nested_pairs_are_not_counted_twice(self):
        # 18 steps; strict reverse work = 3 (inner uncall in the call of outer)
        # + 8 (the uncall of outer, which contains another inner pair) = 11,
        # not 3 + 3 + 8 (proofs/lean: run_eq)
        r = self.run_case("nested.j")
        self.assertEqual((r["total_steps"], r["sem_uncompute_steps"], r["n_paired"]), (18, 11, 3))

    def test_self_inverse_procedure_called_twice_pairs(self):
        r = self.run_case("twice.j")
        self.assertEqual((r["n_paired"], r["n_paired_same_dir"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
