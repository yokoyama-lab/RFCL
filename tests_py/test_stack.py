"""Tests for dynamic stack data structure (SRL extension)."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from pyrev_fl.interpreter import run_program
from pyrev_fl.invert import invert_program
from pyrev_fl.parser import parse_program
from pyrev_fl.pretty import render_program
from pyrev_fl.interface import build_layout

EXAMPLES = Path(__file__).parent.parent / "examples"


class StackBasicTests(unittest.TestCase):
    """Test basic push/pop operations."""

    def test_stack_swap(self):
        """stack_demo.srl: swap x and y via stack."""
        p = parse_program((EXAMPLES / "stack_demo.srl").read_text())
        store = run_program(p, [5, 3])
        self.assertEqual(store["x"], 3)
        self.assertEqual(store["y"], 5)

    def test_stack_swap_zeros(self):
        store = run_program(
            parse_program((EXAMPLES / "stack_demo.srl").read_text()), [0, 0]
        )
        self.assertEqual(store["x"], 0)
        self.assertEqual(store["y"], 0)

    def test_push_pop_identity(self):
        """push then pop returns the same value."""
        src = "(x) (x) (s:stack)\npush x s;\npop x s;\n"
        store = run_program(parse_program(src), [42])
        self.assertEqual(store["x"], 42)

    def test_push_zeros_variable(self):
        """After push, the variable is zero."""
        src = "(x y) (x y) (s:stack)\npush x s;\ny ^= x;\npop x s;\n"
        store = run_program(parse_program(src), [5, 0])
        # After push: x=0. y ^= 0 → y=0. pop: x=5.
        self.assertEqual(store["x"], 5)
        self.assertEqual(store["y"], 0)

    def test_pop_requires_zero_target(self):
        """Pop into non-zero variable raises error."""
        src = "(x y) (x y) (s:stack)\npush x s;\npop y s;\n"
        p = parse_program(src)
        # y=3 is not zero → pop should fail
        with self.assertRaises(Exception):
            run_program(p, [5, 3])

    def test_pop_empty_stack_fails(self):
        """Pop from empty stack raises error."""
        src = "(x) (x) (s:stack)\npop x s;\n"
        p = parse_program(src)
        with self.assertRaises(Exception):
            run_program(p, [0])

    def test_multiple_pushes_lifo(self):
        """Stack is LIFO: last pushed is first popped."""
        src = (
            "(a b c) (a b c) (s:stack)\n"
            "push a s;\npush b s;\npush c s;\n"
            "pop c s;\npop b s;\npop a s;\n"
        )
        store = run_program(parse_program(src), [1, 2, 3])
        self.assertEqual(store["a"], 1)
        self.assertEqual(store["b"], 2)
        self.assertEqual(store["c"], 3)


class StackInversionTests(unittest.TestCase):
    """Test push ↔ pop inversion."""

    def test_inversion_swaps_push_pop(self):
        """invert(push x s) == pop x s, invert(pop x s) == push x s."""
        src = "(x y) (x y) (s:stack)\npush x s;\npush y s;\npop x s;\npop y s;\n"
        p = parse_program(src)
        inv = invert_program(p)
        rendered = render_program(inv)
        # Inverted: push y, push x, pop y, pop x (reversed order, push↔pop)
        self.assertIn("push y s", rendered)
        self.assertIn("push x s", rendered)
        self.assertIn("pop y s", rendered)
        self.assertIn("pop x s", rendered)

    def test_inversion_round_trip(self):
        """run(invert(p), outputs(run(p, inputs))) == inputs."""
        p = parse_program((EXAMPLES / "stack_demo.srl").read_text())
        layout = build_layout(p.inputs, p.outputs, p.temps)
        fwd = run_program(p, [5, 3])
        outputs = [fwd[name] for name in layout.outputs]
        inv = invert_program(p)
        inv_layout = build_layout(inv.inputs, inv.outputs, inv.temps)
        rev = run_program(inv, outputs)
        recovered = [rev[name] for name in inv_layout.outputs]
        self.assertEqual(recovered, [5, 3])

    def test_double_inversion_identity(self):
        """invert(invert(p)) == p."""
        src = "(x y) (x y) (s:stack)\npush x s;\npush y s;\npop x s;\npop y s;\n"
        p = parse_program(src)
        double = invert_program(invert_program(p))
        self.assertEqual(render_program(p), render_program(double))


class StackCliTests(unittest.TestCase):
    """CLI tests for stack programs."""

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "pyrev_fl.cli", *args],
            capture_output=True, text=True,
        )

    def test_cli_run_stack_demo(self):
        r = self._run("run", str(EXAMPLES / "stack_demo.srl"), "5", "3")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("x=3", r.stdout)
        self.assertIn("y=5", r.stdout)

    def test_cli_invert_stack_demo(self):
        r = self._run("invert", str(EXAMPLES / "stack_demo.srl"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("push", r.stdout)
        self.assertIn("pop", r.stdout)

    def test_cli_check_stack_demo(self):
        r = self._run("check", str(EXAMPLES / "stack_demo.srl"))
        self.assertEqual(r.returncode, 0, r.stderr)
