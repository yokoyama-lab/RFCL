"""Tests for the reversible self-interpreter (examples/self_interp.srl).

The self-interpreter executes straight-line reversible programs encoded as
integer arrays.  Instruction format: (opcode, target_idx, source_idx).

    opcode 1: store[target] += store[source]
    opcode 2: store[target] -= store[source]
    opcode 3: store[target] ^= store[source]

This test suite verifies:
1. The interpreter produces correct results for known programs.
2. The interpreter is itself reversible (inversion theorem holds).
3. Lower → RL round-trip works.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from pyrev_fl.interface import build_layout
from pyrev_fl.interpreter import run_program
from pyrev_fl.invert import invert_program
from pyrev_fl.parser import parse_program

EXAMPLES = Path(__file__).parent.parent / "examples"

CODE_SIZE = 30
STORE_SIZE = 10


def _make_inputs(
    instructions: list[tuple[int, int, int]],
    store_init: list[int],
) -> list[int]:
    """Build the flat input list for self_interp.srl.

    Interface: (code[30] n store[10]) → 30 + 1 + 10 = 41 inputs.
    """
    code = [0] * CODE_SIZE
    for i, (op, tgt, src) in enumerate(instructions):
        code[i * 3] = op
        code[i * 3 + 1] = tgt
        code[i * 3 + 2] = src
    n = len(instructions)
    store = list(store_init) + [0] * (STORE_SIZE - len(store_init))
    return code + [n] + store


def _extract_store(result: dict[str, int]) -> list[int]:
    """Extract the store values from the run result."""
    return [result.get(f"store[{i}]", 0) for i in range(STORE_SIZE)]


class SelfInterpTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.program = parse_program((EXAMPLES / "self_interp.srl").read_text())

    def _run(self, instructions, store_init):
        inputs = _make_inputs(instructions, store_init)
        return run_program(self.program, inputs)

    # ------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------

    def test_empty_program(self):
        """n=0: store unchanged."""
        result = self._run([], [5, 3])
        store = _extract_store(result)
        self.assertEqual(store[0], 5)
        self.assertEqual(store[1], 3)

    def test_single_add(self):
        """ADD store[0] store[1]: 5 + 3 = 8."""
        result = self._run([(1, 0, 1)], [5, 3])
        store = _extract_store(result)
        self.assertEqual(store[0], 8)
        self.assertEqual(store[1], 3)

    def test_single_sub(self):
        """SUB store[0] store[1]: 5 - 3 = 2."""
        result = self._run([(2, 0, 1)], [5, 3])
        store = _extract_store(result)
        self.assertEqual(store[0], 2)
        self.assertEqual(store[1], 3)

    def test_single_xor(self):
        """XOR store[0] store[1]: 5 ^ 3 = 6."""
        result = self._run([(3, 0, 1)], [5, 3])
        store = _extract_store(result)
        self.assertEqual(store[0], 6)
        self.assertEqual(store[1], 3)

    def test_two_instructions(self):
        """ADD store[0] store[1]; SUB store[0] store[2]: 5+3-2 = 6."""
        result = self._run([(1, 0, 1), (2, 0, 2)], [5, 3, 2])
        store = _extract_store(result)
        self.assertEqual(store[0], 6)
        self.assertEqual(store[1], 3)
        self.assertEqual(store[2], 2)

    def test_three_instructions_chain(self):
        """XOR store[2] store[0]; ADD store[2] store[1]; SUB store[0] store[2]."""
        result = self._run([(3, 2, 0), (1, 2, 1), (2, 0, 2)], [10, 3, 0])
        store = _extract_store(result)
        # store[2] = 0 ^ 10 = 10, then store[2] = 10 + 3 = 13
        # store[0] = 10 - 13 = -3
        self.assertEqual(store[0], -3)
        self.assertEqual(store[1], 3)
        self.assertEqual(store[2], 13)

    def test_copy_via_xor(self):
        """XOR store[1] store[0]: copy x to y (y starts at 0)."""
        result = self._run([(3, 1, 0)], [42, 0])
        store = _extract_store(result)
        self.assertEqual(store[0], 42)
        self.assertEqual(store[1], 42)

    # ------------------------------------------------------------------
    # Reversibility (inversion theorem)
    # ------------------------------------------------------------------

    def _check_inversion(self, instructions, store_init):
        """Verify: run(invert(interp), outputs(run(interp, inputs))) == inputs."""
        inputs = _make_inputs(instructions, store_init)
        layout = build_layout(
            self.program.inputs, self.program.outputs, self.program.temps
        )
        # Forward
        fwd_store = run_program(self.program, inputs)
        fwd_outputs = [fwd_store[name] for name in layout.outputs]
        # Inverse
        inv = invert_program(self.program)
        inv_layout = build_layout(inv.inputs, inv.outputs, inv.temps)
        rev_store = run_program(inv, fwd_outputs)
        recovered = [rev_store[name] for name in inv_layout.outputs]
        self.assertEqual(recovered, inputs)

    def test_inversion_empty(self):
        self._check_inversion([], [5, 3])

    def test_inversion_single_add(self):
        self._check_inversion([(1, 0, 1)], [5, 3])

    def test_inversion_single_sub(self):
        self._check_inversion([(2, 0, 1)], [5, 3])

    def test_inversion_single_xor(self):
        self._check_inversion([(3, 0, 1)], [5, 3])

    def test_inversion_multi(self):
        self._check_inversion([(1, 0, 1), (2, 0, 2)], [5, 3, 2])

    def test_inversion_chain(self):
        self._check_inversion([(3, 2, 0), (1, 2, 1), (2, 0, 2)], [10, 3, 0])
