"""Integration tests: SRL → Janus → Jana execution.

Verifies that converted Janus programs produce correct results when run
through the Jana interpreter. This validates the Janus conversion against
an independent implementation.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from pyrev_fl.janus_pretty import render_janus
from pyrev_fl.parser import parse_program

EXAMPLES = Path(__file__).parent.parent / "examples"
JANA = Path("/home/a/.local/bin/jana")


def _has_jana() -> bool:
    return JANA.exists()


def _run_jana(janus_src: str, init_stmts: str = "") -> dict[str, int]:
    """Run a Janus program through Jana, return variable values."""
    # Convert to procedure + main with init
    # Parse the procedure body from the Janus source
    lines = janus_src.strip().splitlines()
    # Replace "procedure main()" with "procedure prog(...)" and create a wrapper
    vars_section = []
    body_lines = []
    in_vars = True
    for line in lines[2:]:  # skip comment and "procedure main()"
        stripped = line.strip()
        if in_vars and stripped.startswith("int "):
            vars_section.append(stripped)
        else:
            in_vars = False
            body_lines.append(line)

    var_names = [v.replace("int ", "").strip() for v in vars_section]
    params = ", ".join(f"int {v}" for v in var_names)
    prog_body = "\n".join(body_lines)

    full_src = f"procedure prog({params})\n{prog_body}\n\n"
    full_src += "procedure main()\n"
    for v in var_names:
        full_src += f"    int {v}\n"
    if init_stmts:
        full_src += init_stmts + "\n"
    full_src += f"    call prog({', '.join(var_names)})\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".janus", delete=False) as f:
        f.write(full_src)
        path = f.name
    try:
        result = subprocess.run(
            [str(JANA), path],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Jana error: {result.stderr}")
        # Parse output: "var = value" lines
        vals = {}
        for line in result.stdout.strip().splitlines():
            if "=" in line:
                name, val = line.split("=", 1)
                vals[name.strip()] = int(val.strip())
        return vals
    finally:
        os.unlink(path)


@unittest.skipUnless(_has_jana(), "Jana interpreter not found")
class JanaIntegrationTests(unittest.TestCase):
    """Run converted SRL programs through Jana and verify results."""

    def test_copy_via_jana(self):
        """copy.srl: y += x, with x=7."""
        janus = render_janus(parse_program((EXAMPLES / "copy.srl").read_text()))
        vals = _run_jana(janus, "    x += 7")
        self.assertEqual(vals["x"], 7)
        self.assertEqual(vals["y"], 7)

    def test_countdown_via_jana(self):
        """countdown_clean.srl: acc = 2n+1 for n=3."""
        janus = render_janus(parse_program((EXAMPLES / "countdown_clean.srl").read_text()))
        vals = _run_jana(janus, "    n += 3")
        self.assertEqual(vals["acc"], 7)

    def test_bennett_rif_flag0_via_jana(self):
        """bennett_rif.srl with flag=0: x += y (forward body)."""
        janus = render_janus(parse_program((EXAMPLES / "bennett_rif.srl").read_text()))
        vals = _run_jana(janus, "    x += 2\n    y += 3")
        self.assertEqual(vals["x"], 5)  # x=2+3=5
        self.assertEqual(vals["y"], 3)

    def test_fib_bennett_via_jana(self):
        """fib_bennett.srl: fib(5)=5."""
        janus = render_janus(parse_program((EXAMPLES / "fib_bennett.srl").read_text()))
        vals = _run_jana(janus, "    n += 5")
        self.assertEqual(vals["output"], 5)
        self.assertEqual(vals["n"], 5)

    def test_fib_bennett_multiple_values(self):
        """fib_bennett.srl: verify fib(1)..fib(5) via Jana."""
        expected = {1: 1, 2: 1, 3: 2, 4: 3, 5: 5}
        janus = render_janus(parse_program((EXAMPLES / "fib_bennett.srl").read_text()))
        for n, fib_n in expected.items():
            with self.subTest(n=n):
                vals = _run_jana(janus, f"    n += {n}")
                self.assertEqual(vals["output"], fib_n, f"fib({n})")

    def test_branch_copy_flag1_via_jana(self):
        """branch_copy.srl with flag=1: y ^= x."""
        janus = render_janus(parse_program((EXAMPLES / "branch_copy.srl").read_text()))
        vals = _run_jana(janus, "    x += 7\n    flag += 1")
        self.assertEqual(vals["y"], 7)
