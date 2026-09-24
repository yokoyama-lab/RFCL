"""Tests for the loop-based compilation of multi-level Bennett (pyrev_fl.pebble_compact)."""
from __future__ import annotations

import unittest
from pathlib import Path

from pyrev_fl.interface import build_layout
from pyrev_fl.interpreter import run_program
from pyrev_fl.invert import invert_program
from pyrev_fl.parser import parse_program
from pyrev_fl.pebble import (
    bennett_mixed_schedule, bennett_schedule, compile_pebbling, iterate_reference,
    step_spec, validate_schedule,
)
from pyrev_fl.pebble_compact import (
    compile_bennett_compact, control_overhead, statement_count,
)
from pyrev_fl.tradeoff import _measure

EXAMPLES = Path(__file__).parent.parent / "examples"
STEPS = [("step_fib.srl", [0, 1]), ("step_lcg.srl", [7]), ("step_tri.srl", [100000])]
SHAPES = [[2], [3], [2, 2], [4, 4], [2, 3, 4], [2] * 5, [5, 2]]


def _spec(name: str):
    return step_spec(parse_program((EXAMPLES / name).read_text()))


class MixedScheduleTests(unittest.TestCase):
    def test_uniform_mixed_schedule_is_bennett_schedule(self):
        for k in range(4):
            for m in (2, 3, 4):
                self.assertEqual(bennett_mixed_schedule([m] * k), bennett_schedule(k, m))

    def test_mixed_schedule_counts(self):
        st = validate_schedule(bennett_mixed_schedule([2, 3, 4]), 24)
        self.assertEqual((st.moves, st.pebbles), (3 * 5 * 7, 1 + 2 + 3 + 1))


class CompactTests(unittest.TestCase):
    def test_compact_program_computes_x_n(self):
        for fname, x0 in STEPS:
            spec = _spec(fname)
            for ms in SHAPES:
                c = compile_bennett_compact(spec, ms)
                store = run_program(c.program, x0)
                self.assertEqual([store[o] for o in c.output_vars],
                                 iterate_reference(spec, c.n, x0), (fname, ms))

    def test_time_is_unrolled_time_plus_exact_control_overhead(self):
        for fname, x0 in STEPS:
            spec = _spec(fname)
            for ms in SHAPES:
                c = compile_bennett_compact(spec, ms)
                straight = compile_pebbling(spec, bennett_mixed_schedule(ms), c.n)
                self.assertEqual(
                    _measure(c.program, x0).time_steps,
                    _measure(straight.program, x0).time_steps
                    + control_overhead(ms, len(spec.state)), (fname, ms))

    def test_code_size_grows_with_levels_not_with_n(self):
        spec = _spec("step_lcg.srl")
        sizes = [statement_count(compile_bennett_compact(spec, [2] * k).program.body)
                 for k in range(1, 9)]
        steps = {b - a for a, b in zip(sizes, sizes[1:])}
        self.assertEqual(steps, {22})                   # 22 statements per level
        big = compile_bennett_compact(spec, [2] * 8)     # n = 256, 6561 moves
        self.assertLess(statement_count(big.program.body), 200)
        self.assertEqual(sizes[0], statement_count(
            compile_bennett_compact(spec, [9]).program.body))  # m does not matter either

    def test_registers_match_bennett_pebbles(self):
        spec = _spec("step_fib.srl")
        for ms in SHAPES:
            c = compile_bennett_compact(spec, ms)
            st = validate_schedule(bennett_mixed_schedule(ms), c.n)
            self.assertEqual(c.registers, st.pebbles)

    def test_compact_program_is_invertible(self):
        spec = _spec("step_tri.srl")
        c = compile_bennett_compact(spec, [2, 3])
        fwd = run_program(c.program, [4321])
        inv = invert_program(c.program)
        back = run_program(inv, [fwd[v] for v in inv.inputs])
        self.assertEqual(back["x"], 4321)

    def test_peak_space_within_declared_cells(self):
        spec = _spec("step_fib.srl")
        c = compile_bennett_compact(spec, [2, 2, 2])
        m = _measure(c.program, [0, 1])
        layout = build_layout(c.program.inputs, c.program.outputs, c.program.temps)
        cells = set(layout.inputs) | set(layout.outputs) | set(layout.temps)
        self.assertLessEqual(m.peak_space, len(cells))
        self.assertEqual(c.program.temps[:2], ["a__ck[5]", "b__ck[5]"])  # F + 1 = 5 cells

    def test_rendered_program_parses_back_and_runs(self):
        from pyrev_fl.pretty import render_program
        spec = _spec("step_tri.srl")
        c = compile_bennett_compact(spec, [2, 3, 2])
        again = parse_program(render_program(c.program))
        self.assertEqual(run_program(again, [100000])["x__final"],
                         iterate_reference(spec, 12, [100000])[0])

    def test_rejects_degenerate_shapes(self):
        spec = _spec("step_lcg.srl")
        with self.assertRaises(ValueError):
            compile_bennett_compact(spec, [])
        with self.assertRaises(ValueError):
            compile_bennett_compact(spec, [1, 2])


if __name__ == "__main__":
    unittest.main()
