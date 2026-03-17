"""Tests for program equivalence checking."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import os
import unittest
from pathlib import Path

from pyrev_fl.equiv import check_equivalence
from pyrev_fl.parser import parse_program
from pyrev_fl.invert import invert_program
from pyrev_fl.transform import lower_srl_to_rl, raise_rl_to_srl

EXAMPLES = Path(__file__).parent.parent / "examples"


class EquivTests(unittest.TestCase):

    def test_identity_equivalent(self):
        """A program is equivalent to itself."""
        p = parse_program((EXAMPLES / "copy.srl").read_text())
        ok, msg = check_equivalence(p, p)
        self.assertTrue(ok, msg)

    def test_raised_lowered_equivalent(self):
        """raise(lower(p)) is equivalent to p."""
        p = parse_program((EXAMPLES / "copy.srl").read_text())
        rt = raise_rl_to_srl(lower_srl_to_rl(p))
        ok, msg = check_equivalence(p, rt)
        self.assertTrue(ok, msg)

    def test_different_programs_not_equivalent(self):
        """Two different programs are not equivalent."""
        p1 = parse_program("(x) (x y) ()\ny += x;\n")
        p2 = parse_program("(x) (x y) ()\ny -= x;\n")
        ok, msg = check_equivalence(p1, p2)
        self.assertFalse(ok)
        self.assertIn("counterexample", msg)

    def test_countdown_round_trip_equivalent(self):
        """countdown_clean.srl is equivalent to its raise(lower()) version."""
        p = parse_program((EXAMPLES / "countdown_clean.srl").read_text())
        rt = raise_rl_to_srl(lower_srl_to_rl(p))
        ok, msg = check_equivalence(p, rt, input_range=range(1, 6))
        self.assertTrue(ok, msg)

    def test_semantically_equal_different_syntax(self):
        """Two syntactically different but semantically equal programs."""
        p1 = parse_program("(x y) (x y) ()\nx += y;\nx += y;\n")
        p2 = parse_program("(x y) (x y) ()\nx += y;\nx += y;\n")
        ok, msg = check_equivalence(p1, p2)
        self.assertTrue(ok, msg)

    def test_cli_equiv_same(self):
        result = subprocess.run(
            [sys.executable, "-m", "pyrev_fl.cli", "equiv",
             str(EXAMPLES / "copy.srl"), str(EXAMPLES / "copy.srl")],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("equivalent", result.stdout)

    def test_cli_equiv_different(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srl", delete=False) as f:
            f.write("(x) (x y) ()\ny -= x;\n")
            path = f.name
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pyrev_fl.cli", "equiv",
                 str(EXAMPLES / "copy.srl"), path],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("counterexample", result.stdout)
        finally:
            os.unlink(path)
