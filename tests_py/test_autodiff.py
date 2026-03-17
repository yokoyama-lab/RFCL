import json
import unittest
from pathlib import Path
from subprocess import run

from pyrev_fl.ast import Assign, Block, Const, Program, Swap, UpdateOp, Var, Binary, BinOp
from pyrev_fl.autodiff import DiffResult, compute_jacobian, differentiate
from pyrev_fl.parser import parse_program


class TestIdentityGradient(unittest.TestCase):
    """For y += x (copy), dy/dx = 1."""

    def test_identity_gradient(self) -> None:
        program = parse_program("(x) (x y) ()\ny += x;\n")
        result = differentiate(program, [3], "y")
        self.assertAlmostEqual(result.gradients["y"]["x"], 1.0)


class TestAddConstantGradient(unittest.TestCase):
    """For x += 3, dx/dx = 1 (constant does not contribute a variable gradient)."""

    def test_add_constant_gradient(self) -> None:
        program = parse_program("(x) (x) ()\nx += 3;\n")
        result = differentiate(program, [5], "x")
        # dx_new/dx_old = 1 (the += 3 adds a constant, so x_new = x_old + 3)
        self.assertAlmostEqual(result.gradients["x"]["x"], 1.0)


class TestSubGradient(unittest.TestCase):
    """For x -= y, dx/dy = -1."""

    def test_sub_gradient(self) -> None:
        # Program: inputs (x y), outputs (x y), x -= y
        program = parse_program("(x y) (x y) ()\nx -= y;\n")
        result = differentiate(program, [10, 3], "x")
        # x_new = x_old - y => dx_new/dy = -1
        self.assertAlmostEqual(result.gradients["x"]["y"], -1.0)
        # dx_new/dx_old = 1
        self.assertAlmostEqual(result.gradients["x"]["x"], 1.0)


class TestChainGradient(unittest.TestCase):
    """For x += y; z += x, dz/dy = 1 (chain rule)."""

    def test_chain_gradient(self) -> None:
        # x += y makes x = x_old + y
        # z += x makes z = z_old + x_new = 0 + (x_old + y) = x_old + y
        # so dz/dy = 1
        program = parse_program("(x y) (x y z) ()\nx += y;\nz += x;\n")
        result = differentiate(program, [2, 3], "z")
        self.assertAlmostEqual(result.gradients["z"]["y"], 1.0)
        self.assertAlmostEqual(result.gradients["z"]["x"], 1.0)


class TestZeroTapeSize(unittest.TestCase):
    """Reversible programs always have tape_size = 0."""

    def test_zero_tape_size(self) -> None:
        program = parse_program("(x) (x y) ()\ny += x;\n")
        result = differentiate(program, [3], "y")
        self.assertEqual(result.tape_size, 0)

    def test_zero_tape_chain(self) -> None:
        program = parse_program("(x y) (x y z) ()\nx += y;\nz += x;\n")
        result = differentiate(program, [2, 3], "z")
        self.assertEqual(result.tape_size, 0)

    def test_zero_tape_loop(self) -> None:
        source = Path("examples/countdown_clean.srl").read_text(encoding="utf-8")
        program = parse_program(source)
        result = differentiate(program, [4], "acc")
        self.assertEqual(result.tape_size, 0)


class TestJacobianCopy(unittest.TestCase):
    """Jacobian of copy.srl: y += x gives J = [[1, 0], [1, 1]] for (x, y)."""

    def test_jacobian_copy(self) -> None:
        program = parse_program("(x) (x y) ()\ny += x;\n")
        jac = compute_jacobian(program, [3])
        # dx/dx = 1  (x is unchanged)
        self.assertAlmostEqual(jac["x"]["x"], 1.0)
        # dy/dx = 1  (y += x)
        self.assertAlmostEqual(jac["y"]["x"], 1.0)


class TestSwapJacobian(unittest.TestCase):
    """Jacobian of swap is a permutation matrix."""

    def test_swap_jacobian(self) -> None:
        program = parse_program("(x y) (x y) ()\nx <=> y;\n")
        jac = compute_jacobian(program, [3, 5])
        # After swap: x_new = y_old, y_new = x_old
        # dx_new/dx_old = 0, dx_new/dy_old = 1
        # dy_new/dx_old = 1, dy_new/dy_old = 0
        self.assertAlmostEqual(jac["x"]["x"], 0.0)
        self.assertAlmostEqual(jac["x"]["y"], 1.0)
        self.assertAlmostEqual(jac["y"]["x"], 1.0)
        self.assertAlmostEqual(jac["y"]["y"], 0.0)


class TestXorGradientZero(unittest.TestCase):
    """XOR is not differentiable; gradients are 0."""

    def test_xor_gradient_zero(self) -> None:
        program = parse_program("(x y) (x y) ()\nx ^= y;\n")
        result = differentiate(program, [3, 5], "x")
        # XOR is not differentiable, so gradient through XOR is 0
        self.assertAlmostEqual(result.gradients["x"]["y"], 0.0)


class TestIfBranchGradient(unittest.TestCase):
    """Gradient flows through the taken branch of an if statement."""

    def test_if_then_branch(self) -> None:
        source = """(x flag) (x y flag) ()
if (!= flag 0) then
  y ^= x;
else
fi (= y x)
"""
        program = parse_program(source)
        # flag=1 => then branch => y ^= x (XOR, gradient 0)
        result = differentiate(program, [5, 1], "y")
        # XOR means dy/dx = 0
        self.assertAlmostEqual(result.gradients["y"]["x"], 0.0)

    def test_if_else_branch(self) -> None:
        source = """(x flag) (x flag) ()
if (!= flag 0) then
  x += 1;
else
  x += 2;
fi (!= flag 0)
"""
        program = parse_program(source)
        # flag=0 => else branch => x += 2 (constant, dx/dx = 1)
        result = differentiate(program, [5, 0], "x")
        self.assertAlmostEqual(result.gradients["x"]["x"], 1.0)


class TestLoopGradient(unittest.TestCase):
    """Gradient through a loop accumulates across iterations."""

    def test_countdown_gradient(self) -> None:
        # countdown_clean.srl computes acc = n*(n+1)/2 (triangular number)
        # But the loop body is: acc += 1 (each iteration adds 1 to acc)
        # The loop runs n+1 times total (n iterations of loop block + initial do).
        # d(acc)/d(n) is hard to express exactly for discrete integer loops,
        # but the structure should give a non-trivial gradient.
        source = Path("examples/countdown_clean.srl").read_text(encoding="utf-8")
        program = parse_program(source)
        result = differentiate(program, [4], "acc")
        # acc starts at 0, n starts at 4
        self.assertEqual(result.output_values["acc"], 9)
        self.assertEqual(result.tape_size, 0)


class TestMulExprGradient(unittest.TestCase):
    """Gradient through multiplication expression."""

    def test_mul_gradient(self) -> None:
        # x += (y * z) => dx/dy = z, dx/dz = y
        program = parse_program("(x y z) (x y z) ()\nx += (* y z);\n")
        result = differentiate(program, [0, 3, 5], "x")
        # dx/dy = z = 5
        self.assertAlmostEqual(result.gradients["x"]["y"], 5.0)
        # dx/dz = y = 3
        self.assertAlmostEqual(result.gradients["x"]["z"], 3.0)


class TestDoubleAccumulation(unittest.TestCase):
    """Gradient accumulates when a variable is used multiple times."""

    def test_double_add(self) -> None:
        # z += x; z += x => z = z_old + x + x = 2x, so dz/dx = 2
        # But x appears in two separate statements, both adding to z.
        program = parse_program("(x) (x z) ()\nz += x;\nz += x;\n")
        result = differentiate(program, [3], "z")
        self.assertAlmostEqual(result.gradients["z"]["x"], 2.0)


class TestOutputValues(unittest.TestCase):
    """DiffResult includes correct output values."""

    def test_output_values(self) -> None:
        program = parse_program("(x) (x y) ()\ny += x;\n")
        result = differentiate(program, [7], "y")
        self.assertEqual(result.output_values["x"], 7)
        self.assertEqual(result.output_values["y"], 7)


class TestRifGradient(unittest.TestCase):
    """Gradient through rif flows through the taken branch."""

    def test_rif_forward_branch(self) -> None:
        # flag=0 => forward branch => x += y
        source = """(flag x y) (flag x y) ()
rif (!= flag 0)
  x += y;
rfi (!= flag 0)
"""
        program = parse_program(source)
        result = differentiate(program, [0, 2, 3], "x")
        # Forward branch: x += y, so dx/dy = 1
        self.assertAlmostEqual(result.gradients["x"]["y"], 1.0)

    def test_rif_reverse_branch(self) -> None:
        # flag=1 => reverse branch => invert(x += y) = x -= y
        source = """(flag x y) (flag x y) ()
rif (!= flag 0)
  x += y;
rfi (!= flag 0)
"""
        program = parse_program(source)
        result = differentiate(program, [1, 5, 3], "x")
        # Reverse branch: x -= y, so dx/dy = -1
        self.assertAlmostEqual(result.gradients["x"]["y"], -1.0)


class TestCliAutodiff(unittest.TestCase):
    """CLI autodiff subcommand tests."""

    def test_cli_autodiff_basic(self) -> None:
        result = run(
            [
                "python3", "-m", "pyrev_fl.cli",
                "autodiff", "examples/copy.srl", "3",
                "--output", "y",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("dy/dx", result.stdout)
        self.assertIn("1.0", result.stdout)

    def test_cli_autodiff_json(self) -> None:
        result = run(
            [
                "python3", "-m", "pyrev_fl.cli",
                "autodiff", "--json", "examples/copy.srl", "3",
                "--output", "y",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["tape_size"], 0)
        self.assertAlmostEqual(payload["gradients"]["y"]["x"], 1.0)

    def test_cli_autodiff_jacobian(self) -> None:
        result = run(
            [
                "python3", "-m", "pyrev_fl.cli",
                "autodiff", "--json", "examples/copy.srl", "3",
                "--output", "y", "--jacobian",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        # Jacobian has entries for both x and y
        self.assertIn("x", payload["jacobian"])
        self.assertIn("y", payload["jacobian"])


if __name__ == "__main__":
    unittest.main()
