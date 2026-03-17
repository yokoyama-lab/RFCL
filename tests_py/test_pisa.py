"""Tests for PISA code generation."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from pyrev_fl.pisa import compile_to_pisa
from pyrev_fl.rl_parser import parse_program as rl_parse_program

EXAMPLES = Path(__file__).parent.parent / "examples"


class PisaCodeGenTests(unittest.TestCase):

    def test_copy_pisa(self):
        rl = rl_parse_program((EXAMPLES / "copy.rl").read_text())
        pisa = compile_to_pisa(rl)
        self.assertIn("XOR", pisa)
        self.assertIn("HALT", pisa)
        self.assertIn("R0", pisa)

    def test_fib_bennett_pisa(self):
        rl = rl_parse_program((EXAMPLES / "fib_bennett.rl").read_text())
        pisa = compile_to_pisa(rl)
        self.assertIn("BRA", pisa)
        self.assertIn("RBRA", pisa)
        self.assertIn("ADD", pisa)
        self.assertIn("SUB", pisa)
        self.assertIn("EXCH", pisa)

    def test_register_allocation(self):
        """Variables are mapped to registers R1, R2, ..."""
        rl = rl_parse_program((EXAMPLES / "copy.rl").read_text())
        pisa = compile_to_pisa(rl)
        self.assertIn("R1", pisa)
        self.assertIn("R2", pisa)

    def test_rgoto_generates_rbra(self):
        rl = rl_parse_program((EXAMPLES / "rgoto_copy.rl").read_text())
        pisa = compile_to_pisa(rl)
        self.assertIn("RBRA", pisa)

    def test_conditional_branch(self):
        rl = rl_parse_program((EXAMPLES / "hand_if_multi.rl").read_text())
        pisa = compile_to_pisa(rl)
        self.assertIn("BNE", pisa)

    def test_all_rl_examples_compile(self):
        """All RL examples compile to PISA without errors."""
        for path in sorted(EXAMPLES.glob("*.rl")):
            with self.subTest(file=path.name):
                rl = rl_parse_program(path.read_text())
                pisa = compile_to_pisa(rl)
                self.assertTrue(len(pisa) > 0)

    def test_cli_rl_to_pisa(self):
        result = subprocess.run(
            [sys.executable, "-m", "pyrev_fl.cli", "rl-to-pisa",
             str(EXAMPLES / "copy.rl")],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PISA", result.stdout)
