"""Tests for visualization output (DOT → SVG) and paper figure generation.

Verifies that all RL programs produce valid DOT output that GraphViz can render.
Also generates SVG files in output/ for paper use.
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from pyrev_fl.rl_dot import render_dot
from pyrev_fl.rl_parser import parse_program as rl_parse_program
from pyrev_fl.parser import parse_program
from pyrev_fl.transform import lower_srl_to_rl

EXAMPLES = Path(__file__).parent.parent / "examples"
OUTPUT = Path(__file__).parent.parent / "output"


class DotValidityTests(unittest.TestCase):
    """All RL programs produce syntactically valid DOT."""

    def _check_valid_dot(self, dot: str, label: str) -> None:
        self.assertTrue(dot.strip().startswith("digraph"), f"{label}: not a digraph")
        self.assertIn("}", dot, f"{label}: missing closing brace")
        # Check balanced braces
        self.assertEqual(dot.count("{"), dot.count("}"), f"{label}: unbalanced braces")

    def test_all_rl_examples(self):
        for path in sorted(EXAMPLES.glob("*.rl")):
            with self.subTest(file=path.name):
                prog = rl_parse_program(path.read_text())
                dot = render_dot(prog)
                self._check_valid_dot(dot, path.name)

    def test_all_lowered_srl(self):
        for path in sorted(EXAMPLES.glob("*.srl")):
            if "stack" in path.name:
                continue  # stack programs can't be lowered to RL
            with self.subTest(file=path.name):
                srl = parse_program(path.read_text())
                rl = lower_srl_to_rl(srl)
                dot = render_dot(rl)
                self._check_valid_dot(dot, f"lower({path.name})")


class DotToSvgTests(unittest.TestCase):
    """Verify DOT renders to SVG with GraphViz (if available)."""

    @classmethod
    def setUpClass(cls):
        result = subprocess.run(["which", "dot"], capture_output=True)
        if result.returncode != 0:
            raise unittest.SkipTest("GraphViz not installed")

    def _render_svg(self, dot: str) -> str:
        result = subprocess.run(
            ["dot", "-Tsvg"],
            input=dot, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_fib_bennett_renders(self):
        prog = rl_parse_program((EXAMPLES / "fib_bennett.rl").read_text())
        svg = self._render_svg(render_dot(prog))
        self.assertIn("<svg", svg)

    def test_rgoto_copy_renders(self):
        prog = rl_parse_program((EXAMPLES / "rgoto_copy.rl").read_text())
        svg = self._render_svg(render_dot(prog))
        self.assertIn("<svg", svg)

    def test_lowered_fib_bennett_renders(self):
        srl = parse_program((EXAMPLES / "fib_bennett.srl").read_text())
        rl = lower_srl_to_rl(srl)
        svg = self._render_svg(render_dot(rl))
        self.assertIn("<svg", svg)


class SvgOutputGenerationTests(unittest.TestCase):
    """Generate SVG output files for the paper (side effect: writes to output/)."""

    @classmethod
    def setUpClass(cls):
        result = subprocess.run(["which", "dot"], capture_output=True)
        if result.returncode != 0:
            raise unittest.SkipTest("GraphViz not installed")
        OUTPUT.mkdir(exist_ok=True)

    def _write_svg(self, name: str, dot: str) -> Path:
        result = subprocess.run(
            ["dot", "-Tsvg"], input=dot, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        path = OUTPUT / f"{name}.svg"
        path.write_text(result.stdout)
        return path

    def test_generate_all_rl_svgs(self):
        for path in sorted(EXAMPLES.glob("*.rl")):
            name = path.stem
            prog = rl_parse_program(path.read_text())
            svg_path = self._write_svg(name, render_dot(prog))
            self.assertTrue(svg_path.exists())

    def test_generate_all_lowered_svgs(self):
        for path in sorted(EXAMPLES.glob("*.srl")):
            if "stack" in path.name:
                continue  # stack programs can't be lowered to RL
            name = f"{path.stem}_lowered"
            srl = parse_program(path.read_text())
            rl = lower_srl_to_rl(srl)
            svg_path = self._write_svg(name, render_dot(rl))
            self.assertTrue(svg_path.exists())
