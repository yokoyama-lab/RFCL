"""Inversion theorem property tests.

Verifies the core reversibility invariant for both SRL and RL programs:

    run(invert(p), outputs(run(p, inputs))) == inputs

for a representative set of programs and inputs.
"""
from __future__ import annotations

from pathlib import Path
import unittest

from pyrev_fl.interface import build_layout
from pyrev_fl.interpreter import run_program
from pyrev_fl.invert import invert_program
from pyrev_fl.parser import parse_program
from pyrev_fl.rl_interpreter import run_program as rl_run_program
from pyrev_fl.rl_invert import invert_program as rl_invert_program
from pyrev_fl.rl_parser import parse_program as rl_parse_program
from pyrev_fl.transform import lower_srl_to_rl

EXAMPLES = Path(__file__).parent.parent / "examples"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _srl_round_trip(program, inputs: list[int]) -> list[int]:
    """Run program forward, then run its inverse on the outputs. Return recovered inputs."""
    layout = build_layout(program.inputs, program.outputs, program.temps)
    store_fwd = run_program(program, inputs)
    output_vals = [store_fwd[name] for name in layout.outputs]

    inv = invert_program(program)
    inv_layout = build_layout(inv.inputs, inv.outputs, inv.temps)
    store_rev = run_program(inv, output_vals)
    return [store_rev[name] for name in inv_layout.outputs]


def _rl_round_trip(program, inputs: list[int]) -> list[int]:
    """RL version of the round-trip helper."""
    layout = build_layout(program.inputs, program.outputs, program.temps)
    store_fwd = rl_run_program(program, inputs)
    output_vals = [store_fwd[name] for name in layout.outputs]

    inv = rl_invert_program(program)
    inv_layout = build_layout(inv.inputs, inv.outputs, inv.temps)
    store_rev = rl_run_program(inv, output_vals)
    return [store_rev[name] for name in inv_layout.outputs]


# ---------------------------------------------------------------------------
# SRL inversion tests
# ---------------------------------------------------------------------------

class SrlInversionTests(unittest.TestCase):

    def test_copy_inversion(self):
        """copy.srl: y += x  →  run(inv, (x, y)) recovers x."""
        p = parse_program((EXAMPLES / "copy.srl").read_text())
        for x in [0, 1, 5, -3, 100]:
            with self.subTest(x=x):
                self.assertEqual(_srl_round_trip(p, [x]), [x])

    def test_branch_copy_flag1(self):
        """branch_copy.srl with flag=1: XOR copy branch."""
        p = parse_program((EXAMPLES / "branch_copy.srl").read_text())
        for x in [1, 3, 7, 10]:
            with self.subTest(x=x, flag=1):
                self.assertEqual(_srl_round_trip(p, [x, 1]), [x, 1])

    def test_branch_copy_flag0(self):
        """branch_copy.srl with flag=0: else branch (no-op on y).
        Requires x != 0 so that the assertion (= y x) is false (y starts at 0).
        """
        p = parse_program((EXAMPLES / "branch_copy.srl").read_text())
        for x in [2, 5, -1]:
            with self.subTest(x=x, flag=0):
                self.assertEqual(_srl_round_trip(p, [x, 0]), [x, 0])

    def test_countdown_inversion(self):
        """countdown_clean.srl: accumulate n steps."""
        p = parse_program((EXAMPLES / "countdown_clean.srl").read_text())
        for n in [1, 2, 3, 5, 8]:
            with self.subTest(n=n):
                self.assertEqual(_srl_round_trip(p, [n]), [n])

    def test_bennett_rif_flag0(self):
        """bennett_rif.srl with flag=0: body runs forward (x += y)."""
        p = parse_program((EXAMPLES / "bennett_rif.srl").read_text())
        for x, y in [(0, 3), (2, 5), (10, 1)]:
            with self.subTest(flag=0, x=x, y=y):
                self.assertEqual(_srl_round_trip(p, [0, x, y]), [0, x, y])

    def test_bennett_rif_flag1(self):
        """bennett_rif.srl with flag=1: body runs backward (x -= y)."""
        p = parse_program((EXAMPLES / "bennett_rif.srl").read_text())
        # When flag=1, rif runs invert(body) = x -= y.
        # Round-trip must still recover original inputs.
        for x, y in [(5, 3), (7, 2), (10, 4)]:
            with self.subTest(flag=1, x=x, y=y):
                self.assertEqual(_srl_round_trip(p, [1, x, y]), [1, x, y])

    def test_fib_bennett_inversion(self):
        """fib_bennett.srl: Fibonacci with Bennett's trick; inv recovers n."""
        p = parse_program((EXAMPLES / "fib_bennett.srl").read_text())
        for n in [1, 2, 3, 4, 5]:
            with self.subTest(n=n):
                self.assertEqual(_srl_round_trip(p, [n]), [n])

    def test_inline_swap_inversion(self):
        """Inline: swap x <=> y is its own inverse."""
        src = "(x y) (x y) ()\nx <=> y;\n"
        p = parse_program(src)
        for x, y in [(0, 0), (1, 2), (3, -1), (7, 7)]:
            with self.subTest(x=x, y=y):
                self.assertEqual(_srl_round_trip(p, [x, y]), [x, y])

    def test_inline_nested_if_inversion(self):
        """Inline: nested if/fi structure recovers inputs."""
        src = (
            "(a b flag) (a b flag) ()\n"
            "if (!= flag 0) then\n"
            "  a += b;\n"
            "else\n"
            "  b -= a;\n"
            "fi (!= flag 0)\n"
        )
        p = parse_program(src)
        for a, b in [(1, 2), (5, 3), (0, 7)]:
            with self.subTest(a=a, b=b, flag=1):
                self.assertEqual(_srl_round_trip(p, [a, b, 1]), [a, b, 1])
            with self.subTest(a=a, b=b, flag=0):
                self.assertEqual(_srl_round_trip(p, [a, b, 0]), [a, b, 0])

    def test_inline_xor_assign_inversion(self):
        """Inline: XOR is self-inverse; round-trip is identity."""
        src = "(x y) (x y) ()\nx ^= y;\n"
        p = parse_program(src)
        for x, y in [(0, 0), (3, 5), (255, 255), (0b1010, 0b1100)]:
            with self.subTest(x=x, y=y):
                self.assertEqual(_srl_round_trip(p, [x, y]), [x, y])


# ---------------------------------------------------------------------------
# RL inversion tests
# ---------------------------------------------------------------------------

class RlInversionTests(unittest.TestCase):

    def test_copy_rl_inversion(self):
        """copy.rl: single-block RL program; inv recovers input."""
        p = rl_parse_program((EXAMPLES / "copy.rl").read_text())
        for x in [0, 1, 5, -3, 42]:
            with self.subTest(x=x):
                self.assertEqual(_rl_round_trip(p, [x]), [x])

    def test_fib_bennett_rl_inversion(self):
        """fib_bennett.rl: multi-block RL program; inv recovers n."""
        p = rl_parse_program((EXAMPLES / "fib_bennett.rl").read_text())
        for n in [1, 2, 3, 4, 5]:
            with self.subTest(n=n):
                self.assertEqual(_rl_round_trip(p, [n]), [n])


# ---------------------------------------------------------------------------
# Lowered SRL → RL inversion consistency
# ---------------------------------------------------------------------------

class LoweredRlInversionTests(unittest.TestCase):
    """Verify that lowering preserves the inversion property:
    a SRL program and its lowered RL counterpart must both satisfy the theorem.
    """

    def _check_both(self, srl_prog, inputs: list[int]) -> None:
        recovered_srl = _srl_round_trip(srl_prog, inputs)
        self.assertEqual(recovered_srl, inputs, f"SRL inv failed for inputs={inputs}")

        rl_prog = lower_srl_to_rl(srl_prog)
        recovered_rl = _rl_round_trip(rl_prog, inputs)
        self.assertEqual(recovered_rl, inputs, f"RL inv failed for inputs={inputs}")

    def test_copy_lowered(self):
        p = parse_program((EXAMPLES / "copy.srl").read_text())
        for x in [0, 3, 7]:
            self._check_both(p, [x])

    def test_countdown_lowered(self):
        p = parse_program((EXAMPLES / "countdown_clean.srl").read_text())
        for n in [1, 3, 5]:
            self._check_both(p, [n])

    def test_fib_bennett_lowered(self):
        p = parse_program((EXAMPLES / "fib_bennett.srl").read_text())
        for n in [1, 2, 4]:
            self._check_both(p, [n])

    def test_bennett_rif_lowered(self):
        p = parse_program((EXAMPLES / "bennett_rif.srl").read_text())
        for flag, x, y in [(0, 2, 3), (1, 7, 4)]:
            self._check_both(p, [flag, x, y])
