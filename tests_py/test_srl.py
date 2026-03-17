import unittest
import json
from pathlib import Path
from subprocess import run

from pyrev_fl.ast import Assign, BinOp, Binary, Block, Const, If, Loop, Program, Rif, UpdateOp, Var
from pyrev_fl.check import CheckError, check_program
from pyrev_fl.interpreter import EvalError, run_program
from pyrev_fl.invert import invert_program
from pyrev_fl.parser import ParseError, parse_program
from pyrev_fl.pretty import render_program
from pyrev_fl.rl_interpreter import run_program as rl_run_program
from pyrev_fl.rl_parser import parse_program as rl_parse_program
from pyrev_fl.transform import lower_srl_to_rl, raise_rl_to_srl


class SrlTests(unittest.TestCase):
    def test_invert_swaps_io_and_reverses_block_order(self) -> None:
        program = Program(
            inputs=["x"],
            outputs=["y"],
            temps=["t"],
            body=Block(
                [
                    Assign("x", UpdateOp.ADD, Const(1)),
                    Assign("y", UpdateOp.XOR, Var("x")),
                ]
            ),
        )

        inverted = invert_program(program)

        self.assertEqual(inverted.inputs, ["y"])
        self.assertEqual(inverted.outputs, ["x"])
        self.assertIsInstance(inverted.body.stmts[0], Assign)
        self.assertIs(inverted.body.stmts[1].op, UpdateOp.SUB)

    def test_eval_basic_assignment(self) -> None:
        program = parse_program("(x) (x y) ()\ny += x;\n")
        store = run_program(program, [3])
        self.assertEqual(store["x"], 3)
        self.assertEqual(store["y"], 3)

    def test_eval_if_uses_test_and_assertion_pair(self) -> None:
        program = Program(
            inputs=["x"],
            outputs=["x"],
            temps=[],
            body=Block(
                [
                    If(
                        test=Binary(BinOp.EQ, Var("x"), Const(0)),
                        then_block=Block([]),
                        else_block=Block([]),
                        assertion=Binary(BinOp.EQ, Var("x"), Const(0)),
                    )
                ]
            ),
        )

        store = run_program(program, [0])
        self.assertEqual(store["x"], 0)

    def test_check_rejects_assignment_that_mentions_target(self) -> None:
        program = Program(
            inputs=["x"],
            outputs=["y"],
            temps=[],
            body=Block([Assign("x", UpdateOp.ADD, Binary(BinOp.ADD, Var("x"), Const(1)))]),
        )

        with self.assertRaises(CheckError):
            check_program(program)

    def test_check_rejects_undeclared_variable(self) -> None:
        program = parse_program("(x) (y) ()\ny += z;\n")
        with self.assertRaises(CheckError):
            check_program(program)

    def test_check_rejects_array_self_reference(self) -> None:
        program = parse_program("(a[2] i) (a[2]) ()\na[i] += a[0];\n")
        with self.assertRaises(CheckError):
            check_program(program)

    def test_check_rejects_degenerate_swap(self) -> None:
        program = parse_program("(x) (x) ()\nx <=> x;\n")
        with self.assertRaises(CheckError):
            check_program(program)

    def test_eval_rejects_array_index_out_of_bounds(self) -> None:
        program = parse_program("(a[2]) (out) ()\nout += a[2];\n")
        with self.assertRaises(EvalError):
            run_program(program, [3, 4])

    def test_parse_and_render_round_trip(self) -> None:
        source = """(x) (x y) ()
if (= x 0) then
  y += 1;
else
  y ^= x;
fi (!= y 0)
"""
        program = parse_program(source)
        reparsed = parse_program(render_program(program))
        self.assertEqual(program, reparsed)

    def test_parse_loop_program(self) -> None:
        source = """(n acc) (n acc) ()
from (= acc 0) do
  acc += 1;
loop
  n -= 1;
  acc += 1;
until (= n 0)
"""
        program = parse_program(source)
        self.assertIsInstance(program.body.stmts[0], Loop)

    def test_parse_and_eval_array_access(self) -> None:
        source = """(a[2] idx) (a[2] idx out) ()
out += a[idx];
a[0] += 1;
"""
        program = parse_program(source)
        reparsed = parse_program(render_program(program))
        self.assertEqual(program, reparsed)
        store = run_program(program, [5, 7, 1])
        self.assertEqual(store["a[0]"], 6)
        self.assertEqual(store["a[1]"], 7)
        self.assertEqual(store["idx"], 1)
        self.assertEqual(store["out"], 7)

    def test_parse_and_eval_rif(self) -> None:
        source = """(flag x y) (flag x y) ()
rif (!= flag 0)
  x += y;
rfi (!= flag 0)
"""
        program = parse_program(source)
        self.assertIsInstance(program.body.stmts[0], Rif)
        store = run_program(program, [0, 2, 3])
        self.assertEqual(store["x"], 5)

    def test_rejects_non_zero_temp_at_exit(self) -> None:
        program = parse_program("(x) (y) (t)\nt += 1;\ny += x;\n")
        with self.assertRaises(EvalError):
            run_program(program, [3])

    def test_run_branch_copy_example(self) -> None:
        source = Path("examples/branch_copy.srl").read_text(encoding="utf-8")
        store = run_program(parse_program(source), [5, 1])
        self.assertEqual(store["x"], 5)
        self.assertEqual(store["y"], 5)
        self.assertEqual(store["flag"], 1)

    def test_cli_run_copy_example(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "run", "examples/copy.srl", "3"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip().splitlines(), ["x=3", "y=3"])

    def test_cli_run_copy_example_json(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "run", "--json", "examples/copy.srl", "3"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["kind"], "srl_run")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["outputs"], {"x": 3, "y": 3})

    def test_cli_invert_json(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "invert", "--json", "examples/copy.srl"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["kind"], "srl_invert")
        self.assertTrue(payload["ok"])
        self.assertIn("(x y) (x) ()", payload["program"])

    def test_run_countdown_example(self) -> None:
        source = Path("examples/countdown_clean.srl").read_text(encoding="utf-8")
        store = run_program(parse_program(source), [4])
        self.assertEqual(store["acc"], 9)
        self.assertEqual(store["n"], 0)

    def test_cli_dump_ast(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "dump-ast", "examples/copy.srl"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["kind"], "srl_program")
        self.assertEqual(payload["body"]["stmts"][0]["kind"], "assign")

    def test_cli_check(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "check", "examples/countdown_clean.srl"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "ok")

    def test_cli_check_json(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "check", "--json", "examples/countdown_clean.srl"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["kind"], "srl_check")
        self.assertTrue(payload["ok"])

    def test_cli_trace(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "trace", "examples/copy.srl", "3"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload[0]["kind"], "assign")
        self.assertEqual(payload[0]["store_diff"]["y"]["after"], 3)

    def test_cli_trace_jsonl(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "trace", "--jsonl", "examples/copy.srl", "3"],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        self.assertGreaterEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertEqual(payload["kind"], "assign")
        self.assertEqual(payload["store_diff"]["y"]["after"], 3)

    def test_lower_srl_to_rl_cli(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "lower-srl-to-rl", "examples/copy.srl"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("entry;", result.stdout)
        self.assertIn("y += x;", result.stdout)

    def test_lower_srl_to_rl_cli_json(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "lower-srl-to-rl", "--json", "examples/copy.srl"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["kind"], "lower_srl_to_rl")
        self.assertTrue(payload["ok"])
        self.assertIn("entry;", payload["program"])

    def test_lower_loop_srl_to_rl_runs(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "lower-srl-to-rl", "examples/countdown_clean.srl"],
            check=True,
            capture_output=True,
            text=True,
        )
        rl_program = rl_parse_program(result.stdout)
        store = rl_run_program(rl_program, [4])
        self.assertEqual(store["acc"], 9)

    def test_lower_array_srl_to_rl_runs(self) -> None:
        source = """(a[2] idx) (a[2] idx out) ()
out += a[idx];
a[0] += 1;
"""
        rl_program = lower_srl_to_rl(parse_program(source))
        store = rl_run_program(rl_program, [5, 7, 1])
        self.assertEqual(store["a[0]"], 6)
        self.assertEqual(store["a[1]"], 7)
        self.assertEqual(store["idx"], 1)
        self.assertEqual(store["out"], 7)

    def test_parse_error_reports_position(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            parse_program("(x) (y) ()\ny += ;\n")
        self.assertIn("line 2, column 6", str(ctx.exception))

    def test_lower_rif_to_rl_false_branch(self) -> None:
        """rif with flag=0 (test false): runs body forward."""
        source = Path("examples/bennett_rif.srl").read_text(encoding="utf-8")
        srl_prog = parse_program(source)
        rl_prog = lower_srl_to_rl(srl_prog)
        store = rl_run_program(rl_prog, [0, 2, 3])
        self.assertEqual(store["x"], 5)  # body x+=y executed forward

    def test_lower_rif_to_rl_true_branch(self) -> None:
        """rif with flag=1 (test true): runs body backward (invert: x-=y)."""
        source = Path("examples/bennett_rif.srl").read_text(encoding="utf-8")
        srl_prog = parse_program(source)
        rl_prog = lower_srl_to_rl(srl_prog)
        store = rl_run_program(rl_prog, [1, 5, 3])
        self.assertEqual(store["x"], 2)  # x-=y executed (invert of x+=y)

    def test_lower_rif_round_trip(self) -> None:
        """lower then raise a rif program reproduces the original."""
        source = Path("examples/bennett_rif.srl").read_text(encoding="utf-8")
        srl_prog = parse_program(source)
        raised = raise_rl_to_srl(lower_srl_to_rl(srl_prog))
        self.assertEqual(srl_prog, raised)

    def test_lower_rif_cli(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "lower-srl-to-rl", "examples/bennett_rif.srl"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("rif_fwd", result.stdout)
        self.assertIn("rif_rev", result.stdout)

    def test_lower_fib_bennett_round_trip(self) -> None:
        """fib_bennett.srl (rif nested in loop) survives lower→raise round-trip."""
        source = Path("examples/fib_bennett.srl").read_text(encoding="utf-8")
        srl_prog = parse_program(source)
        raised = raise_rl_to_srl(lower_srl_to_rl(srl_prog))
        self.assertEqual(srl_prog, raised)

    def test_cli_run_fib_bennett_example(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "run", "examples/fib_bennett.srl", "5"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip().splitlines(), ["n=5", "output=5"])

    def test_cli_lower_then_raise_fib_bennett(self) -> None:
        lowered = run(
            ["python3", "-m", "pyrev_fl.cli", "lower-srl-to-rl", "examples/fib_bennett.srl"],
            check=True,
            capture_output=True,
            text=True,
        )
        raised = run(
            ["python3", "-m", "pyrev_fl.cli", "raise-rl-to-srl", "/dev/stdin"],
            input=lowered.stdout,
            check=True,
            capture_output=True,
            text=True,
        )
        original = parse_program(Path("examples/fib_bennett.srl").read_text(encoding="utf-8"))
        self.assertEqual(parse_program(raised.stdout), original)

    def test_cli_check_json_error(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "check", "--json", "/dev/stdin"],
            input="(x) (y) ()\ny += z;\n",
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ok"])
        self.assertIn("undeclared variable", payload["error"])

    def test_cli_run_array_json(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "run", "--json", "/dev/stdin", "5", "7", "1"],
            input="(a[2] idx) (a[2] idx out) ()\nout += a[idx];\na[0] += 1;\n",
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["outputs"], {"a[0]": 6, "a[1]": 7, "idx": 1, "out": 7})

    def test_cli_run_json_error(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "run", "--json", "/dev/stdin", "3"],
            input="(x) (y) (t)\nt += 1;\ny += x;\n",
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ok"])
        self.assertIn("must be zero at exit", payload["error"])


if __name__ == "__main__":
    unittest.main()
