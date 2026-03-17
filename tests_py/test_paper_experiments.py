"""Additional experiments from Moriyama 2009.

Experiment 3: rgoto/rfrom examples beyond Bennett-Fibonacci
Experiment 4: Execution trace verification (operational semantics)
Experiment 5: RL static checker constraint coverage
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from pyrev_fl.interface import build_layout
from pyrev_fl.interpreter import run_program
from pyrev_fl.invert import invert_program
from pyrev_fl.parser import parse_program
from pyrev_fl.rl_interpreter import run_program as rl_run_program
from pyrev_fl.rl_invert import invert_program as rl_invert_program
from pyrev_fl.rl_parser import parse_program as rl_parse_program
from pyrev_fl.rl_check import check_program as rl_check_program, CheckError
from pyrev_fl.trace import trace_program
from pyrev_fl.rl_trace import trace_program as rl_trace_program

EXAMPLES = Path(__file__).parent.parent / "examples"


# ===================================================================
# Experiment 3: rgoto/rfrom examples
# ===================================================================

class RgotoCopyTests(unittest.TestCase):
    """Test the Bennett-style copy using rgoto/rfrom (rgoto_copy.rl)."""

    @classmethod
    def setUpClass(cls):
        cls.prog = rl_parse_program((EXAMPLES / "rgoto_copy.rl").read_text())

    def test_copy_positive(self):
        for x in [1, 5, 42, 100]:
            with self.subTest(x=x):
                store = rl_run_program(self.prog, [x])
                self.assertEqual(store["y"], x)

    def test_copy_zero(self):
        store = rl_run_program(self.prog, [0])
        self.assertEqual(store["y"], 0)

    def test_copy_negative(self):
        store = rl_run_program(self.prog, [-7])
        self.assertEqual(store["y"], -7)

    def test_temps_are_zero(self):
        """The copied flag must be zero at exit."""
        store = rl_run_program(self.prog, [5])
        self.assertEqual(store["copied"], 0)

    def test_inversion_recovers_input(self):
        """run(invert(p), outputs(run(p, x))) == x."""
        for x in [0, 3, -5]:
            with self.subTest(x=x):
                layout = build_layout(
                    self.prog.inputs, self.prog.outputs, self.prog.temps
                )
                fwd = rl_run_program(self.prog, [x])
                outputs = [fwd[name] for name in layout.outputs]
                inv = rl_invert_program(self.prog)
                inv_layout = build_layout(inv.inputs, inv.outputs, inv.temps)
                rev = rl_run_program(inv, outputs)
                recovered = [rev[name] for name in inv_layout.outputs]
                self.assertEqual(recovered, [x])

    def test_double_inversion_identity(self):
        """rl_invert(rl_invert(p)) == p."""
        from pyrev_fl.rl_pretty import render_program as rl_render
        double_inv = rl_invert_program(rl_invert_program(self.prog))
        self.assertEqual(
            rl_render(self.prog),
            rl_render(double_inv),
        )

    def test_rgoto_copy_matches_srl_copy(self):
        """rgoto_copy.rl produces the same result as copy.srl."""
        srl = parse_program((EXAMPLES / "copy.srl").read_text())
        for x in [0, 5, -3]:
            with self.subTest(x=x):
                rl_store = rl_run_program(self.prog, [x])
                srl_store = run_program(srl, [x])
                self.assertEqual(rl_store["y"], srl_store["y"])


class RgotoCopyRaisingTests(unittest.TestCase):
    """Test raising rgoto_copy.rl (simple Bennett) to SRL."""

    def test_raise_rgoto_copy_to_srl(self):
        from pyrev_fl.transform import raise_rl_to_srl
        from pyrev_fl.pretty import render_program
        rl = rl_parse_program((EXAMPLES / "rgoto_copy.rl").read_text())
        srl = raise_rl_to_srl(rl)
        rendered = render_program(srl)
        self.assertIn("from", rendered)
        self.assertIn("until", rendered)

    def test_raised_rgoto_copy_semantics(self):
        """Raised SRL from rgoto_copy.rl matches RL outputs."""
        from pyrev_fl.transform import raise_rl_to_srl
        rl = rl_parse_program((EXAMPLES / "rgoto_copy.rl").read_text())
        srl = raise_rl_to_srl(rl)
        layout = build_layout(rl.inputs, rl.outputs, rl.temps)
        for x in [0, 5, -3, 42]:
            with self.subTest(x=x):
                rl_store = rl_run_program(rl, [x])
                srl_store = run_program(srl, [x])
                for name in layout.outputs:
                    self.assertEqual(srl_store[name], rl_store[name])

    def test_raised_rgoto_copy_round_trip(self):
        """raise → lower → run = original RL run."""
        from pyrev_fl.transform import raise_rl_to_srl, lower_srl_to_rl
        rl = rl_parse_program((EXAMPLES / "rgoto_copy.rl").read_text())
        srl = raise_rl_to_srl(rl)
        rl2 = lower_srl_to_rl(srl)
        for x in [0, 5, 42]:
            with self.subTest(x=x):
                s1 = rl_run_program(rl, [x])
                s2 = rl_run_program(rl2, [x])
                self.assertEqual(s1["y"], s2["y"])


class FibBennettRlRgotoTests(unittest.TestCase):
    """Verify the paper's Fibonacci example (p.14-17) with rgoto/rfrom."""

    @classmethod
    def setUpClass(cls):
        cls.prog = rl_parse_program((EXAMPLES / "fib_bennett.rl").read_text())

    def test_fib_values(self):
        """fib(1)=1, fib(2)=1, fib(3)=2, fib(4)=3, fib(5)=5."""
        expected = {1: 1, 2: 1, 3: 2, 4: 3, 5: 5}
        for n, fib_n in expected.items():
            with self.subTest(n=n):
                store = rl_run_program(self.prog, [n])
                self.assertEqual(store["output"], fib_n)

    def test_srl_rl_equivalence(self):
        """fib_bennett.rl and fib_bennett.srl produce the same output."""
        srl = parse_program((EXAMPLES / "fib_bennett.srl").read_text())
        for n in [1, 2, 3, 5]:
            with self.subTest(n=n):
                rl_store = rl_run_program(self.prog, [n])
                srl_store = run_program(srl, [n])
                self.assertEqual(rl_store["output"], srl_store["output"])

    def test_temps_zero_at_exit(self):
        """v1, v2, output-copied must be zero at exit."""
        store = rl_run_program(self.prog, [5])
        self.assertEqual(store["v1"], 0)
        self.assertEqual(store["v2"], 0)
        self.assertEqual(store["output-copied"], 0)


# ===================================================================
# Experiment 4: Execution trace verification
# ===================================================================

class SrlTraceTests(unittest.TestCase):
    """Verify SRL traces match the operational semantics (Figures 2.12-2.14)."""

    def test_copy_trace_single_step(self):
        """copy.srl (y += x) should have exactly one step in the trace."""
        prog = parse_program((EXAMPLES / "copy.srl").read_text())
        events = trace_program(prog, [5])
        # Trace should show: initial state, the assignment, final state
        self.assertTrue(len(events) > 0)
        # Should contain assignment event
        assign_events = [e for e in events if e.get("kind") == "assign"]
        self.assertEqual(len(assign_events), 1)

    def test_countdown_trace_loop_iterations(self):
        """countdown_clean.srl with n=3 should iterate the loop 3 times."""
        prog = parse_program((EXAMPLES / "countdown_clean.srl").read_text())
        events = trace_program(prog, [3])
        # Count loop-body events (n -= 1 happens 3 times)
        loop_assigns = [
            e for e in events
            if e.get("kind") == "assign" and e.get("target") == "n"
        ]
        self.assertEqual(len(loop_assigns), 3)

    def test_branch_copy_trace_then_branch(self):
        """branch_copy.srl with flag=1 takes the then branch."""
        prog = parse_program((EXAMPLES / "branch_copy.srl").read_text())
        events = trace_program(prog, [5, 1])
        # Should see if-then event
        if_events = [e for e in events if e.get("kind") == "if"]
        self.assertTrue(len(if_events) > 0)
        self.assertTrue(if_events[0].get("branch") == "then")


class RlTraceTests(unittest.TestCase):
    """Verify RL traces show direction changes for rgoto/rfrom."""

    def test_fib_bennett_direction_changes(self):
        """fib_bennett.rl should have direction changes (for → back → for)."""
        prog = rl_parse_program((EXAMPLES / "fib_bennett.rl").read_text())
        events = rl_trace_program(prog, [1])
        directions = [e.get("direction") for e in events if "direction" in e]
        # Must see both forward and backward execution
        self.assertIn("for", directions)
        self.assertIn("back", directions)

    def test_rgoto_copy_direction_changes(self):
        """rgoto_copy.rl should have direction changes."""
        prog = rl_parse_program((EXAMPLES / "rgoto_copy.rl").read_text())
        events = rl_trace_program(prog, [5])
        directions = [e.get("direction") for e in events if "direction" in e]
        self.assertIn("for", directions)
        self.assertIn("back", directions)

    def test_copy_rl_no_direction_change(self):
        """copy.rl (no rgoto) stays in forward direction throughout."""
        prog = rl_parse_program((EXAMPLES / "copy.rl").read_text())
        events = rl_trace_program(prog, [5])
        directions = [e.get("direction") for e in events if "direction" in e]
        for d in directions:
            self.assertEqual(d, "for")


# ===================================================================
# Experiment 5: RL static checker constraint coverage
# ===================================================================

class RlCheckerConstraintTests(unittest.TestCase):
    """Verify the RL static checker catches all constraint violations."""

    def _check_fails(self, source: str, expected_msg: str) -> None:
        prog = rl_parse_program(source)
        with self.assertRaises(CheckError) as ctx:
            rl_check_program(prog)
        self.assertIn(expected_msg, str(ctx.exception))

    def test_duplicate_label(self):
        self._check_fails(
            "(x) (x) ()\nA: entry; exit;\nA: from A; exit;\n",
            "duplicate block label"
        )

    def test_missing_entry(self):
        self._check_fails(
            "(x) (x) ()\nA: from B; exit;\nB: from A; exit;\n",
            "expected exactly one entry block"
        )

    def test_missing_exit(self):
        self._check_fails(
            "(x) (x) ()\nA: entry; goto B;\nB: from A; goto A;\n",
            "expected exactly one exit block"
        )

    def test_unknown_label_in_goto(self):
        self._check_fails(
            "(x) (x) ()\nA: entry; goto MISSING;\nB: from A; exit;\n",
            "unknown block label"
        )

    def test_unknown_label_in_from(self):
        self._check_fails(
            "(x) (x) ()\nA: entry; goto B;\nB: from MISSING; exit;\n",
            "unknown block label"
        )

    def test_unreachable_block(self):
        # C and D form a disconnected component unreachable from entry.
        self._check_fails(
            "(x) (x) ()\nA: entry; exit;\nC: from D; goto D;\nD: from C; goto C;\n",
            "unreachable"
        )

    def test_from_jump_inconsistency(self):
        """Block C claims predecessor A, but A jumps to B not C."""
        self._check_fails(
            "(x) (x) ()\nA: entry; goto B;\nB: from A; exit;\nC: from A; goto B;\n",
            "inconsistency"
        )

    def test_undeclared_variable(self):
        self._check_fails(
            "(x) (x) ()\nA: entry; z += 1; exit;\n",
            "undeclared variable"
        )

    def test_self_referential_assignment(self):
        self._check_fails(
            "(x) (x) ()\nA: entry; x += x; exit;\n",
            "assignment expression mentions target"
        )

    def test_same_swap_variable(self):
        self._check_fails(
            "(x) (x) ()\nA: entry; x <=> x; exit;\n",
            "swap uses the same variable twice"
        )
