"""Tests for multi-level Bennett / reversible pebbling (pyrev_fl.pebble)."""
from __future__ import annotations

import unittest
from pathlib import Path

from pyrev_fl.interpreter import run_program
from pyrev_fl.invert import invert_program
from pyrev_fl.parser import parse_program
from pyrev_fl.pebble import (
    analyze_pebbling, bennett_schedule, bfs_min_moves, compile_pebbling,
    iterate_reference, linear_schedule, min_moves, optimal_schedule, step_spec,
    validate_schedule,
)

EXAMPLES = Path(__file__).parent.parent / "examples"


def _spec(name: str):
    return step_spec(parse_program((EXAMPLES / name).read_text()))


class ScheduleTests(unittest.TestCase):
    def test_linear_schedule_costs_2n_minus_1_moves_and_n_pebbles(self):
        for n in range(1, 20):
            st = validate_schedule(linear_schedule(n), n)
            self.assertEqual((st.moves, st.pebbles), (2 * n - 1, n))

    def test_bennett_schedule_matches_bennett_1989_counts(self):
        for k in range(0, 5):
            for m in range(2, 6):
                if m ** k > 700:
                    continue
                st = validate_schedule(bennett_schedule(k, m), m ** k)
                self.assertEqual(st.moves, (2 * m - 1) ** k)
                self.assertEqual(st.pebbles, k * (m - 1) + 1)

    def test_optimal_recursion_equals_exhaustive_search(self):
        for n in range(1, 13):
            for s in range(1, n + 1):
                self.assertEqual(min_moves(n, s), bfs_min_moves(n, s), (n, s))

    def test_optimal_schedule_is_legal_and_achieves_min_moves(self):
        for n in range(1, 40):
            for s in range(1, 8):
                if min_moves(n, s) == float("inf"):
                    continue
                st = validate_schedule(optimal_schedule(n, s), n)
                self.assertEqual(st.moves, min_moves(n, s))
                self.assertLessEqual(st.pebbles, s)

    def test_s_pebbles_reach_exactly_node_2_to_the_s_minus_1(self):
        for s in range(1, 6):
            reach = 2 ** (s - 1)
            self.assertLess(bfs_min_moves(reach, s), float("inf"))
            self.assertEqual(bfs_min_moves(reach + 1, s), float("inf"))

    def test_optimum_beats_binary_bennett_at_its_own_design_point(self):
        # n = 2^(s-1) with s pebbles: Bennett k = s-1, m = 2 takes 3^(s-1).
        self.assertEqual(bfs_min_moves(8, 4), 25)      # vs 27
        self.assertEqual(bfs_min_moves(16, 5), 71)     # vs 81

    def test_validate_rejects_illegal_moves(self):
        with self.assertRaises(ValueError):
            validate_schedule([2], 2)               # node 1 empty
        with self.assertRaises(ValueError):
            validate_schedule([1, 2], 2)            # node 1 left pebbled
        with self.assertRaises(ValueError):
            validate_schedule([1, -1, 1, 2, -2], 2)  # final node empty


class CompileTests(unittest.TestCase):
    def test_compiled_program_computes_x_n_for_every_family(self):
        cases = [("step_fib.srl", [0, 1]), ("step_lcg.srl", [7]),
                 ("step_tri.srl", [1000])]
        n = 16
        for fname, x0 in cases:
            spec = _spec(fname)
            ref = iterate_reference(spec, n, x0)
            for moves in (linear_schedule(n), bennett_schedule(4, 2),
                          bennett_schedule(2, 4), optimal_schedule(n, 5)):
                peb = compile_pebbling(spec, moves, n)
                store = run_program(peb.program, x0)
                self.assertEqual([store[v] for v in peb.output_vars], ref, fname)

    def test_compiled_program_declares_one_register_per_pebble(self):
        spec = _spec("step_fib.srl")
        for s in (5, 6, 9):
            peb = compile_pebbling(spec, optimal_schedule(16, s), 16)
            self.assertEqual(peb.registers, peb.stats.pebbles)
            self.assertEqual(len(peb.program.temps) + len(peb.output_vars),
                             2 * peb.registers)

    def test_compiled_program_is_invertible(self):
        spec = _spec("step_lcg.srl")
        peb = compile_pebbling(spec, bennett_schedule(3, 2), 8)
        fwd = run_program(peb.program, [11])
        inv = invert_program(peb.program)
        back = run_program(inv, [fwd[v] for v in inv.inputs])
        self.assertEqual(back["x"], 11)

    def test_straight_line_step_time_is_moves_times_step_length(self):
        spec = _spec("step_fib.srl")
        for moves, n in ((linear_schedule(8), 8), (bennett_schedule(3, 2), 8),
                         (optimal_schedule(8, 4), 8)):
            r = analyze_pebbling(spec, moves, n, [0, 1])
            self.assertEqual(r.time_steps, 3 * r.stats.moves)
            self.assertEqual(r.baseline_time, 3 * n)
            self.assertLessEqual(r.peak_space, r.declared_space)

    def test_output_is_keyed_by_state_names_for_every_schedule(self):
        spec = _spec("step_fib.srl")
        for moves in (optimal_schedule(16, 5), bennett_schedule(2, 4), linear_schedule(16)):
            r = analyze_pebbling(spec, moves, 16, [0, 1])
            self.assertEqual(r.output, {"a": 987, "b": 1597})

    def test_step_program_shape_is_checked(self):
        with self.assertRaises(ValueError):
            step_spec(parse_program("(x) (y) ()\ny += x;"))


if __name__ == "__main__":
    unittest.main()
