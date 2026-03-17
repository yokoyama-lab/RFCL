"""Tests for the extended type checker."""
from __future__ import annotations

import subprocess
import sys
import unittest

from pyrev_fl.parser import parse_program
from pyrev_fl.typecheck import typecheck_program


class TypecheckArrayBoundsTests(unittest.TestCase):

    def test_valid_array_access(self):
        src = "(a[5]) (a[5]) ()\na[0] += 1;\na[4] += 2;\n"
        warnings = typecheck_program(parse_program(src))
        errors = [w for w in warnings if w.level == "error"]
        self.assertEqual(errors, [])

    def test_out_of_bounds_access(self):
        src = "(a[3]) (a[3]) ()\na[5] += 1;\n"
        warnings = typecheck_program(parse_program(src))
        errors = [w for w in warnings if w.level == "error"]
        self.assertTrue(len(errors) > 0)
        self.assertIn("out of bounds", errors[0].message)

    def test_negative_index(self):
        # Negative constant index (parser may not produce this, but check anyway)
        from pyrev_fl.ast import Assign, ArrayRef, Block, Const, Program, UpdateOp
        prog = Program(
            inputs=["a[3]"], outputs=["a[3]"], temps=[],
            body=Block([Assign(ArrayRef("a", Const(-1)), UpdateOp.ADD, Const(1))]),
        )
        warnings = typecheck_program(prog)
        errors = [w for w in warnings if w.level == "error"]
        self.assertTrue(len(errors) > 0)


class TypecheckStackBalanceTests(unittest.TestCase):

    def test_balanced_push_pop(self):
        src = "(x) (x) (s:stack)\npush x s;\npop x s;\n"
        warnings = typecheck_program(parse_program(src))
        balance_warnings = [w for w in warnings if "unbalanced" in w.message]
        self.assertEqual(balance_warnings, [])

    def test_unbalanced_push(self):
        src = "(x) (x) (s:stack)\npush x s;\n"
        warnings = typecheck_program(parse_program(src))
        balance_warnings = [w for w in warnings if "unbalanced" in w.message]
        self.assertTrue(len(balance_warnings) > 0)

    def test_extra_pop_warning(self):
        src = "(x) (x) (s:stack)\npop x s;\n"
        warnings = typecheck_program(parse_program(src))
        pop_warnings = [w for w in warnings if "empty stack" in w.message or "unbalanced" in w.message]
        self.assertTrue(len(pop_warnings) > 0)


class TypecheckNoIssuesTests(unittest.TestCase):

    def test_clean_programs_no_warnings(self):
        """Well-formed programs without stacks or arrays produce no warnings."""
        for src in [
            "(x) (x y) ()\ny += x;\n",
            "(x y) (x y) ()\nx <=> y;\n",
        ]:
            with self.subTest(src=src[:20]):
                warnings = typecheck_program(parse_program(src))
                self.assertEqual(warnings, [])


class TypecheckCliTests(unittest.TestCase):

    def test_cli_typecheck_clean(self):
        import tempfile, os
        src = "(x) (x y) ()\ny += x;\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srl", delete=False) as f:
            f.write(src)
            path = f.name
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pyrev_fl.cli", "typecheck", path],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0)
            self.assertIn("ok", result.stdout)
        finally:
            os.unlink(path)

    def test_cli_typecheck_json(self):
        import tempfile, os
        src = "(a[3]) (a[3]) ()\na[5] += 1;\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srl", delete=False) as f:
            f.write(src)
            path = f.name
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pyrev_fl.cli", "typecheck", path, "--json"],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("out of bounds", result.stdout)
        finally:
            os.unlink(path)
