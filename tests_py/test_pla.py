"""Tests for PLA (PISA-style reversible assembly) ↔ RL conversion."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from pyrev_fl.pla_parser import parse_pla
from pyrev_fl.pla_pretty import render_pla
from pyrev_fl.rl_interpreter import run_program as rl_run_program
from pyrev_fl.rl_parser import parse_program as rl_parse_program
from pyrev_fl.rl_pretty import render_program as rl_render_program

EXAMPLES = Path(__file__).parent.parent / "examples"


class PlaPrettyTests(unittest.TestCase):
    """Test RL → PLA rendering."""

    def test_copy_pla(self):
        rl = rl_parse_program((EXAMPLES / "copy.rl").read_text())
        pla = render_pla(rl)
        self.assertIn("ENTRY", pla)
        self.assertIn("XOR", pla)
        self.assertIn("HALT", pla)

    def test_fib_bennett_pla(self):
        rl = rl_parse_program((EXAMPLES / "fib_bennett.rl").read_text())
        pla = render_pla(rl)
        self.assertIn("BRA", pla)
        self.assertIn("RBRA", pla)
        self.assertIn("RFROM", pla)
        self.assertIn("BNEZ", pla)

    def test_hand_countdown_pla(self):
        rl = rl_parse_program((EXAMPLES / "hand_countdown.rl").read_text())
        pla = render_pla(rl)
        self.assertIn("ADD", pla)
        self.assertIn("SUB", pla)

    def test_rgoto_copy_pla(self):
        rl = rl_parse_program((EXAMPLES / "rgoto_copy.rl").read_text())
        pla = render_pla(rl)
        self.assertIn("RBRA", pla)
        self.assertIn("RFROM", pla)


class PlaRoundTripTests(unittest.TestCase):
    """Test RL → PLA → RL round-trip."""

    def _check_round_trip(self, path: Path) -> None:
        rl = rl_parse_program(path.read_text())
        pla = render_pla(rl)
        rl2 = parse_pla(pla)
        self.assertEqual(
            rl_render_program(rl),
            rl_render_program(rl2),
            f"Round-trip failed for {path.name}",
        )

    def test_copy_round_trip(self):
        self._check_round_trip(EXAMPLES / "copy.rl")

    def test_fib_bennett_round_trip(self):
        self._check_round_trip(EXAMPLES / "fib_bennett.rl")

    def test_rgoto_copy_round_trip(self):
        self._check_round_trip(EXAMPLES / "rgoto_copy.rl")

    def test_hand_countdown_round_trip(self):
        self._check_round_trip(EXAMPLES / "hand_countdown.rl")


class PlaSemanticTests(unittest.TestCase):
    """Verify PLA round-tripped RL programs produce the same outputs."""

    def _check_semantics(self, path: Path, inputs: list[int]) -> None:
        rl = rl_parse_program(path.read_text())
        pla = render_pla(rl)
        rl2 = parse_pla(pla)
        store1 = rl_run_program(rl, inputs)
        store2 = rl_run_program(rl2, inputs)
        from pyrev_fl.interface import build_layout
        layout = build_layout(rl.inputs, rl.outputs, rl.temps)
        for name in layout.outputs:
            self.assertEqual(store1[name], store2[name])

    def test_copy_semantics(self):
        self._check_semantics(EXAMPLES / "copy.rl", [5])

    def test_fib_bennett_semantics(self):
        for n in [1, 3, 5]:
            with self.subTest(n=n):
                self._check_semantics(EXAMPLES / "fib_bennett.rl", [n])

    def test_rgoto_copy_semantics(self):
        for x in [0, 5, -3]:
            with self.subTest(x=x):
                self._check_semantics(EXAMPLES / "rgoto_copy.rl", [x])


class PlaCliTests(unittest.TestCase):
    """CLI integration tests for PLA commands."""

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "pyrev_fl.cli", *args],
            capture_output=True, text=True,
        )

    def test_rl_to_pla_cli(self):
        r = self._run("rl-to-pla", str(EXAMPLES / "copy.rl"))
        self.assertEqual(r.returncode, 0)
        self.assertIn("ENTRY", r.stdout)

    def test_pla_to_rl_pipe(self):
        """rl-to-pla | pla-to-rl round-trip via CLI."""
        import tempfile, os
        r1 = self._run("rl-to-pla", str(EXAMPLES / "copy.rl"))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pla", delete=False) as f:
            f.write(r1.stdout)
            path = f.name
        try:
            r2 = self._run("pla-to-rl", path)
            self.assertEqual(r2.returncode, 0)
            self.assertIn("entry", r2.stdout)
        finally:
            os.unlink(path)
