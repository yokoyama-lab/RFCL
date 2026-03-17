"""Tests for the SRL ↔ Janus conversion."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from pyrev_fl.interface import build_layout
from pyrev_fl.interpreter import run_program
from pyrev_fl.janus_parser import parse_janus
from pyrev_fl.janus_pretty import render_janus
from pyrev_fl.parser import parse_program
from pyrev_fl.pretty import render_program

EXAMPLES = Path(__file__).parent.parent / "examples"


class SrlToJanusTests(unittest.TestCase):
    """render_janus: SRL → Janus text."""

    def _srl(self, text: str):
        return parse_program(text)

    def test_header_contains_procedure_main(self):
        p = self._srl("(x) (x y) ()\ny += x;\n")
        janus = render_janus(p)
        self.assertIn("procedure main()", janus)

    def test_interface_comment(self):
        p = self._srl("(x) (x y) (t)\ny += x;\n")
        janus = render_janus(p)
        self.assertIn("SRL inputs=(x)", janus)
        self.assertIn("outputs=(x y)", janus)
        self.assertIn("temps=(t)", janus)

    def test_variable_declarations(self):
        p = self._srl("(x) (x y) (t)\ny += x;\n")
        janus = render_janus(p)
        self.assertIn("int x", janus)
        self.assertIn("int y", janus)
        self.assertIn("int t", janus)

    def test_assign_add(self):
        p = self._srl("(x y) (x y) ()\nx += y;\n")
        janus = render_janus(p)
        self.assertIn("x += y", janus)

    def test_assign_sub(self):
        p = self._srl("(x y) (x y) ()\nx -= y;\n")
        janus = render_janus(p)
        self.assertIn("x -= y", janus)

    def test_assign_xor(self):
        p = self._srl("(x y) (x y) ()\nx ^= y;\n")
        janus = render_janus(p)
        self.assertIn("x ^= y", janus)

    def test_swap(self):
        p = self._srl("(x y) (x y) ()\nx <=> y;\n")
        janus = render_janus(p)
        self.assertIn("x <=> y", janus)

    def test_if_fi(self):
        p = self._srl(
            "(x flag) (x y flag) ()\n"
            "if (!= flag 0) then\n  y ^= x;\nelse\nfi (!= flag 0)\n"
        )
        janus = render_janus(p)
        self.assertIn("if", janus)
        self.assertIn("then", janus)
        self.assertIn("else", janus)
        self.assertIn("fi", janus)

    def test_empty_else_becomes_skip(self):
        p = self._srl(
            "(x flag) (x y flag) ()\n"
            "if (!= flag 0) then\n  y ^= x;\nelse\nfi (!= flag 0)\n"
        )
        janus = render_janus(p)
        self.assertIn("skip", janus)

    def test_loop(self):
        p = self._srl(
            "(n) (acc) ()\n"
            "from (= acc 0) do\n  acc += 1;\nloop\n  n -= 1;\nuntil (= n 0)\n"
        )
        janus = render_janus(p)
        self.assertIn("from", janus)
        self.assertIn("do", janus)
        self.assertIn("loop", janus)
        self.assertIn("until", janus)

    def test_rif_expanded_as_if(self):
        """rif (t) S rfi (a) is rendered as if t then invert(S) else S fi a."""
        p = self._srl(
            "(flag x y) (flag x y) ()\n"
            "rif (!= flag 0)\n  x += y;\nrfi (!= flag 0)\n"
        )
        janus = render_janus(p)
        # Should NOT contain "rif" keyword — expanded to if/fi
        self.assertNotIn("rif", janus)
        self.assertIn("if", janus)
        # Both the forward body (x += y) and its inverse (x -= y) should appear
        self.assertIn("x += y", janus)
        self.assertIn("x -= y", janus)

    def test_infix_eq_operator(self):
        """SRL (= x 0) → Janus (x = 0) infix."""
        p = self._srl(
            "(n) (acc) ()\n"
            "from (= acc 0) do\n  acc += 1;\nloop\n  n -= 1;\nuntil (= n 0)\n"
        )
        janus = render_janus(p)
        self.assertIn("= 0", janus)

    def test_no_semicolons(self):
        """Janus statements have no trailing semicolons."""
        p = self._srl("(x y) (x y) ()\nx += y;\nx <=> y;\n")
        janus = render_janus(p)
        # Remove the comment line and check no semicolons remain
        code_lines = [l for l in janus.splitlines() if not l.strip().startswith("//")]
        for line in code_lines:
            self.assertNotIn(";", line, f"semicolon found in: {line!r}")


class JanusToSrlTests(unittest.TestCase):
    """parse_janus: Janus text → SRL Program."""

    def test_simple_assign(self):
        src = "procedure main()\n    int x\n    int y\n    y += x\n"
        p = parse_janus(src)
        self.assertEqual(set(p.inputs), {"x", "y"})
        srl = render_program(p)
        self.assertIn("y += x", srl)

    def test_interface_comment_preserved(self):
        """If SRL interface comment is present, it sets inputs/outputs/temps."""
        src = (
            "// SRL inputs=(x) outputs=(x y) temps=()\n"
            "procedure main()\n"
            "    int x\n"
            "    int y\n"
            "    y += x\n"
        )
        p = parse_janus(src)
        self.assertEqual(p.inputs, ["x"])
        self.assertEqual(p.outputs, ["x", "y"])
        self.assertEqual(p.temps, [])

    def test_swap(self):
        src = "procedure main()\n    int x\n    int y\n    x <=> y\n"
        p = parse_janus(src)
        srl = render_program(p)
        self.assertIn("<=>", srl)

    def test_if_fi(self):
        src = (
            "procedure main()\n"
            "    int x\n"
            "    int y\n"
            "    int flag\n"
            "    if flag != 0 then\n"
            "        y += x\n"
            "    else\n"
            "        skip\n"
            "    fi flag != 0\n"
        )
        p = parse_janus(src)
        srl = render_program(p)
        self.assertIn("if", srl)
        self.assertIn("fi", srl)

    def test_loop(self):
        src = (
            "procedure main()\n"
            "    int n\n"
            "    int acc\n"
            "    from acc = 0 do\n"
            "        acc += 1\n"
            "    loop\n"
            "        n -= 1\n"
            "    until n = 0\n"
        )
        p = parse_janus(src)
        srl = render_program(p)
        self.assertIn("from", srl)
        self.assertIn("until", srl)

    def test_expression_infix(self):
        """Janus infix expression (x + y) parsed to Binary(ADD, Var(x), Var(y))."""
        src = (
            "procedure main()\n"
            "    int x\n"
            "    int y\n"
            "    int z\n"
            "    z += (x + y)\n"
        )
        p = parse_janus(src)
        srl = render_program(p)
        self.assertIn("z +=", srl)

    def test_skip_ignored(self):
        """skip produces an empty block; does not appear in SRL output."""
        src = (
            "// SRL inputs=(x flag) outputs=(x y flag) temps=()\n"
            "procedure main()\n"
            "    int x\n"
            "    int y\n"
            "    int flag\n"
            "    if flag != 0 then\n"
            "        y ^= x\n"
            "    else\n"
            "        skip\n"
            "    fi flag != 0\n"
        )
        p = parse_janus(src)
        # else branch should be empty Block
        from pyrev_fl.ast import If
        if_stmt = p.body.stmts[0]
        self.assertIsInstance(if_stmt, If)
        self.assertEqual(if_stmt.else_block.stmts, [])


class RoundTripTests(unittest.TestCase):
    """SRL → Janus → SRL round-trips."""

    def _roundtrip(self, srl_text: str) -> str:
        p1 = parse_program(srl_text)
        janus = render_janus(p1)
        p2 = parse_janus(janus)
        return render_program(p2)

    def _check_semantics(self, srl_text: str, inputs: list[int], expected_outputs: dict[str, int]) -> None:
        """Verify that the Janus round-tripped program produces the same outputs."""
        p1 = parse_program(srl_text)
        janus = render_janus(p1)
        p2 = parse_janus(janus)
        layout = build_layout(p2.inputs, p2.outputs, p2.temps)
        store = run_program(p2, inputs)
        for name, val in expected_outputs.items():
            self.assertEqual(store[name], val, f"{name}: expected {val}, got {store[name]}")

    def test_roundtrip_copy(self):
        srl = "(x) (x y) ()\ny += x;\n"
        rt = self._roundtrip(srl)
        self.assertIn("y += x", rt)

    def test_roundtrip_interface(self):
        """Interface (inputs/outputs/temps) survives round-trip."""
        srl = "(x) (x y) (t)\ny += x;\n"
        p1 = parse_program(srl)
        janus = render_janus(p1)
        p2 = parse_janus(janus)
        self.assertEqual(p2.inputs, p1.inputs)
        self.assertEqual(p2.outputs, p1.outputs)
        self.assertEqual(p2.temps, p1.temps)

    def test_semantics_copy(self):
        self._check_semantics("(x) (x y) ()\ny += x;\n", [5], {"x": 5, "y": 5})

    def test_semantics_countdown(self):
        srl = (EXAMPLES / "countdown_clean.srl").read_text()
        self._check_semantics(srl, [3], {"acc": 7})  # countdown: acc = 2n+1

    def test_semantics_branch_copy_flag1(self):
        srl = (EXAMPLES / "branch_copy.srl").read_text()
        self._check_semantics(srl, [7, 1], {"x": 7, "y": 7, "flag": 1})

    def test_roundtrip_rif(self):
        """rif is expanded to if/fi in Janus; the Janus version has no rif."""
        srl = (EXAMPLES / "bennett_rif.srl").read_text()
        p1 = parse_program(srl)
        janus = render_janus(p1)
        self.assertNotIn("rif", janus)
        # Can be parsed back to SRL (as if/fi, not rif)
        p2 = parse_janus(janus)
        self.assertIsNotNone(p2)


class CliTests(unittest.TestCase):
    """CLI integration tests for srl-to-janus and janus-to-srl."""

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "pyrev_fl.cli", *args],
            capture_output=True, text=True, cwd=str(EXAMPLES.parent),
        )

    def test_srl_to_janus_copy(self):
        result = self._run("srl-to-janus", str(EXAMPLES / "copy.srl"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("procedure main()", result.stdout)
        self.assertIn("y += x", result.stdout)

    def test_srl_to_janus_fib_bennett(self):
        result = self._run("srl-to-janus", str(EXAMPLES / "fib_bennett.srl"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("procedure main()", result.stdout)
        self.assertNotIn("rif", result.stdout)  # rif expanded

    def test_janus_to_srl_simple(self):
        import tempfile, os
        src = (
            "// SRL inputs=(x) outputs=(x y) temps=()\n"
            "procedure main()\n"
            "    int x\n"
            "    int y\n"
            "    y += x\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".janus", delete=False) as f:
            f.write(src)
            path = f.name
        try:
            result = self._run("janus-to-srl", path)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("y += x", result.stdout)
        finally:
            os.unlink(path)

    def test_janus_to_srl_roundtrip_cli(self):
        """srl-to-janus | janus-to-srl produces parseable SRL."""
        r1 = self._run("srl-to-janus", str(EXAMPLES / "countdown_clean.srl"))
        self.assertEqual(r1.returncode, 0, r1.stderr)
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".janus", delete=False) as f:
            f.write(r1.stdout)
            path = f.name
        try:
            r2 = self._run("janus-to-srl", path)
            self.assertEqual(r2.returncode, 0, r2.stderr)
            self.assertIn("from", r2.stdout)
            self.assertIn("until", r2.stdout)
        finally:
            os.unlink(path)
