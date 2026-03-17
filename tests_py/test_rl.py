import unittest
import json
from pathlib import Path
from subprocess import run

from pyrev_fl.parser import parse_program as parse_srl_program
from pyrev_fl.rl_ast import RGoto
from pyrev_fl.rl_check import CheckError, check_program
from pyrev_fl.rl_interpreter import EvalError, run_program
from pyrev_fl.rl_invert import invert_program
from pyrev_fl.rl_parser import ParseError, parse_program
from pyrev_fl.rl_pretty import render_program
from pyrev_fl.transform import raise_rl_to_srl


class RlTests(unittest.TestCase):
    def test_parse_and_render_round_trip(self) -> None:
        source = """(x) (x y) ()
start: entry;
y += x;
exit;
"""
        program = parse_program(source)
        reparsed = parse_program(render_program(program))
        self.assertEqual(program, reparsed)

    def test_parse_reverse_jump_block(self) -> None:
        source = """(x) (x) ()
a: from b;
rgoto c;
"""
        program = parse_program(source)
        self.assertIsInstance(program.blocks[0].jump, RGoto)

    def test_run_paper_fibonacci_example(self) -> None:
        with open("examples/fib_bennett.rl", "r", encoding="utf-8") as fh:
            program = parse_program(fh.read())
        store = run_program(program, [5])
        self.assertEqual(store["n"], 5)
        self.assertEqual(store["output"], 5)

    def test_invert_swaps_program_io(self) -> None:
        source = """(x) (y) ()
a: entry;
y ^= x;
exit;
"""
        program = parse_program(source)
        inverted = invert_program(program)
        self.assertEqual(inverted.inputs, ["y"])
        self.assertEqual(inverted.outputs, ["x"])

    def test_rejects_non_zero_temp_at_exit(self) -> None:
        source = """(x) (y) (t)
a: entry;
t += 1;
y ^= x;
exit;
"""
        program = parse_program(source)
        with self.assertRaises(EvalError):
            run_program(program, [3])

    def test_check_rejects_unknown_label(self) -> None:
        source = """(x) (x) ()
a: entry;
goto missing;
"""
        with self.assertRaises(CheckError):
            check_program(parse_program(source))

    def test_check_rejects_unreachable_block(self) -> None:
        source = """(x) (x) ()
a: entry;
exit;
b: from a;
exit;
"""
        with self.assertRaises(CheckError):
            check_program(parse_program(source))

    def test_check_rejects_undeclared_variable(self) -> None:
        source = """(x) (y) ()
a: entry;
y ^= z;
exit;
"""
        with self.assertRaises(CheckError):
            check_program(parse_program(source))

    def test_check_rejects_array_swap_alias(self) -> None:
        source = """(a[2] i j) (a[2]) ()
a: entry;
a[i] <=> a[j];
exit;
"""
        with self.assertRaises(CheckError):
            check_program(parse_program(source))

    def test_run_copy_example(self) -> None:
        source = Path("examples/copy.rl").read_text(encoding="utf-8")
        store = run_program(parse_program(source), [7])
        self.assertEqual(store["x"], 7)
        self.assertEqual(store["y"], 7)

    def test_run_array_program(self) -> None:
        source = """(a[2] idx) (a[2] idx out) ()
start: entry;
out += a[idx];
a[0] += 1;
exit;
"""
        store = run_program(parse_program(source), [5, 7, 1])
        self.assertEqual(store["a[0]"], 6)
        self.assertEqual(store["a[1]"], 7)
        self.assertEqual(store["idx"], 1)
        self.assertEqual(store["out"], 7)

    def test_cli_rl_run_copy_example(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "rl-run", "examples/copy.rl", "7"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip().splitlines(), ["x=7", "y=7"])

    def test_cli_rl_run_copy_example_json(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "rl-run", "--json", "examples/copy.rl", "7"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["kind"], "rl_run")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["outputs"], {"x": 7, "y": 7})

    def test_cli_rl_run_array_json(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "rl-run", "--json", "/dev/stdin", "5", "7", "1"],
            input="(a[2] idx) (a[2] idx out) ()\nstart: entry;\nout += a[idx];\na[0] += 1;\nexit;\n",
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["outputs"], {"a[0]": 6, "a[1]": 7, "idx": 1, "out": 7})

    def test_cli_rl_invert_json(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "rl-invert", "--json", "examples/copy.rl"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["kind"], "rl_invert")
        self.assertTrue(payload["ok"])
        self.assertIn("(x y) (x) ()", payload["program"])

    def test_cli_rl_dump_ast(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "rl-dump-ast", "examples/copy.rl"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["kind"], "rl_program")
        self.assertEqual(payload["blocks"][0]["jump"]["kind"], "exit")

    def test_cli_rl_check(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "rl-check", "examples/copy.rl"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "ok")

    def test_cli_rl_check_json(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "rl-check", "--json", "examples/copy.rl"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["kind"], "rl_check")
        self.assertTrue(payload["ok"])

    def test_cli_rl_trace(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "rl-trace", "examples/copy.rl", "7"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload[0]["current_label"], "start")
        self.assertEqual(payload[0]["store_diff"]["y"]["after"], 7)

    def test_cli_rl_trace_jsonl(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "rl-trace", "--jsonl", "examples/copy.rl", "7"],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        self.assertGreaterEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertEqual(payload["current_label"], "start")
        self.assertEqual(payload["store_diff"]["y"]["after"], 7)

    def test_parse_error_reports_position(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            parse_program("(x) (y) ()\na: entry;\ny += ;\nexit;\n")
        self.assertIn("line 3, column 6", str(ctx.exception))

    def test_raise_rl_to_srl_cli(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "raise-rl-to-srl", "examples/copy.rl"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("y ^= x;", result.stdout)

    def test_raise_rl_to_srl_cli_json(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "raise-rl-to-srl", "--json", "examples/copy.rl"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["kind"], "raise_rl_to_srl")
        self.assertTrue(payload["ok"])
        self.assertIn("y ^= x;", payload["program"])

    def test_raise_generated_loop_rl_to_srl(self) -> None:
        lowered = run(
            ["python3", "-m", "pyrev_fl.cli", "lower-srl-to-rl", "examples/countdown_clean.srl"],
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
        self.assertIn("from (= acc 0) do", raised.stdout)
        self.assertIn("until (= n 0)", raised.stdout)

    def test_raise_bennett_reverse_jump_program(self) -> None:
        rl_program = parse_program(Path("examples/fib_bennett.rl").read_text(encoding="utf-8"))
        raised = raise_rl_to_srl(rl_program)
        expected = parse_srl_program(Path("examples/fib_bennett.srl").read_text(encoding="utf-8"))
        self.assertEqual(raised, expected)

    def test_raise_bennett_reverse_jump_cli(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "raise-rl-to-srl", "examples/fib_bennett.rl"],
            check=True,
            capture_output=True,
            text=True,
        )
        raised = parse_srl_program(result.stdout)
        expected = parse_srl_program(Path("examples/fib_bennett.srl").read_text(encoding="utf-8"))
        self.assertEqual(raised, expected)

    def test_raise_embedded_bennett_reverse_jump_program(self) -> None:
        rl_source = """(a n) (a n output out2) (v1 v2 output-copied)
start: entry;
out2 ^= a;
goto pre_init;
pre_init: from start;
goto init;
init: fi (= output-copied 0) from pre_init else pre_end;
v1 ^= 0;
v2 ^= 1;
goto test;
test: fi (= v1 0) from init else loop;
if (= n 0) goto copy_entry else loop;
loop: from test;
v1 += v2;
v1 <=> v2;
n -= 1;
goto test;
copy_entry: from test;
if (= output-copied 0) goto copy_main else copy_exit;
copy_main: from copy_entry;
output ^= v1;
output-copied ^= 1;
rgoto copy_exit;
copy_exit: from copy_entry;
rgoto copy_main;
pre_end: rfrom end;
goto init;
end: rfrom pre_end;
output-copied ^= 1;
goto done;
done: from end;
out2 ^= output;
exit;
"""
        expected_source = """(a n) (a n output out2) (v1 v2 output-copied)
out2 ^= a;
from (= 0 output-copied) do
  rif (!= 0 output-copied)
    v1 ^= 0;
    v2 ^= 1;
    from (= v1 0) do
      v1 += v2;
      v1 <=> v2;
      n -= 1;
    loop
    until (= n 0)
  rfi (!= 0 output-copied)
loop
  output ^= v1;
  output-copied ^= 1;
until (!= output-copied 0)
output-copied ^= 1;
out2 ^= output;
"""
        raised = raise_rl_to_srl(parse_program(rl_source))
        self.assertEqual(raised, parse_srl_program(expected_source))

    def test_raise_embedded_bennett_reverse_jump_cli(self) -> None:
        rl_source = """(a n) (a n output out2) (v1 v2 output-copied)
start: entry;
out2 ^= a;
goto pre_init;
pre_init: from start;
goto init;
init: fi (= output-copied 0) from pre_init else pre_end;
v1 ^= 0;
v2 ^= 1;
goto test;
test: fi (= v1 0) from init else loop;
if (= n 0) goto copy_entry else loop;
loop: from test;
v1 += v2;
v1 <=> v2;
n -= 1;
goto test;
copy_entry: from test;
if (= output-copied 0) goto copy_main else copy_exit;
copy_main: from copy_entry;
output ^= v1;
output-copied ^= 1;
rgoto copy_exit;
copy_exit: from copy_entry;
rgoto copy_main;
pre_end: rfrom end;
goto init;
end: rfrom pre_end;
output-copied ^= 1;
goto done;
done: from end;
out2 ^= output;
exit;
"""
        expected_source = """(a n) (a n output out2) (v1 v2 output-copied)
out2 ^= a;
from (= 0 output-copied) do
  rif (!= 0 output-copied)
    v1 ^= 0;
    v2 ^= 1;
    from (= v1 0) do
      v1 += v2;
      v1 <=> v2;
      n -= 1;
    loop
    until (= n 0)
  rfi (!= 0 output-copied)
loop
  output ^= v1;
  output-copied ^= 1;
until (!= output-copied 0)
output-copied ^= 1;
out2 ^= output;
"""
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "raise-rl-to-srl", "/dev/stdin"],
            input=rl_source,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(parse_srl_program(result.stdout), parse_srl_program(expected_source))

    def test_raise_bennett_reverse_jump_split_copy_chain(self) -> None:
        rl_source = """(a n) (a n output out2) (v1 v2 output-copied)
start: entry;
out2 ^= a;
goto pre_init;
pre_init: from start;
goto init;
init: fi (= output-copied 0) from pre_init else pre_end;
v1 ^= 0;
v2 ^= 1;
goto test;
test: fi (= v1 0) from init else loop;
if (= n 0) goto copy_entry else loop;
loop: from test;
v1 += v2;
v1 <=> v2;
n -= 1;
goto test;
copy_entry: from test;
if (= output-copied 0) goto copy_main else copy_exit;
copy_main: from copy_entry;
output ^= v1;
goto copy_tail;
copy_tail: from copy_main;
output-copied ^= 1;
rgoto copy_exit;
copy_exit: from copy_entry;
rgoto copy_tail;
pre_end: rfrom end;
goto init;
end: rfrom pre_end;
output-copied ^= 1;
goto done;
done: from end;
out2 ^= output;
exit;
"""
        expected_source = """(a n) (a n output out2) (v1 v2 output-copied)
out2 ^= a;
from (= 0 output-copied) do
  rif (!= 0 output-copied)
    v1 ^= 0;
    v2 ^= 1;
    from (= v1 0) do
      v1 += v2;
      v1 <=> v2;
      n -= 1;
    loop
    until (= n 0)
  rfi (!= 0 output-copied)
loop
  output ^= v1;
  output-copied ^= 1;
until (!= output-copied 0)
output-copied ^= 1;
out2 ^= output;
"""
        self.assertEqual(raise_rl_to_srl(parse_program(rl_source)), parse_srl_program(expected_source))

    def test_raise_bennett_reverse_jump_split_loop_chain(self) -> None:
        rl_source = """(a n) (a n output out2) (v1 v2 output-copied)
start: entry;
out2 ^= a;
goto pre_init;
pre_init: from start;
goto init;
init: fi (= output-copied 0) from pre_init else pre_end;
v1 ^= 0;
v2 ^= 1;
goto test;
test: fi (= v1 0) from init else loop_head;
if (= n 0) goto copy_entry else loop_head;
loop_head: from test;
v1 += v2;
goto loop_tail;
loop_tail: from loop_head;
v1 <=> v2;
n -= 1;
goto test;
copy_entry: from test;
if (= output-copied 0) goto copy_main else copy_exit;
copy_main: from copy_entry;
output ^= v1;
output-copied ^= 1;
rgoto copy_exit;
copy_exit: from copy_entry;
rgoto copy_main;
pre_end: rfrom end;
goto init;
end: rfrom pre_end;
output-copied ^= 1;
goto done;
done: from end;
out2 ^= output;
exit;
"""
        expected_source = """(a n) (a n output out2) (v1 v2 output-copied)
out2 ^= a;
from (= 0 output-copied) do
  rif (!= 0 output-copied)
    v1 ^= 0;
    v2 ^= 1;
    from (= v1 0) do
      v1 += v2;
      v1 <=> v2;
      n -= 1;
    loop
    until (= n 0)
  rfi (!= 0 output-copied)
loop
  output ^= v1;
  output-copied ^= 1;
until (!= output-copied 0)
output-copied ^= 1;
out2 ^= output;
"""
        self.assertEqual(raise_rl_to_srl(parse_program(rl_source)), parse_srl_program(expected_source))

    def test_raise_rejects_rgoto(self) -> None:
        source = """(x) (x) ()
a: entry;
rgoto b;
b: rfrom a;
exit;
"""
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "raise-rl-to-srl", "/dev/stdin"],
            input=source,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rgoto", result.stderr)

    def test_raise_json_error(self) -> None:
        source = """(x) (x) ()
a: entry;
rgoto b;
b: rfrom a;
exit;
"""
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "raise-rl-to-srl", "--json", "/dev/stdin"],
            input=source,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ok"])
        self.assertIn("rgoto", payload["error"])

    def test_cli_rl_check_json_error(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "rl-check", "--json", "/dev/stdin"],
            input="(x) (x) ()\na: entry;\ngoto missing;\nb: from a;\nexit;\n",
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ok"])
        self.assertIn("unknown block label", payload["error"])

    def test_cli_rl_run_json_error(self) -> None:
        result = run(
            ["python3", "-m", "pyrev_fl.cli", "rl-run", "--json", "/dev/stdin", "3"],
            input="(x) (y) (t)\na: entry;\nt += 1;\ny ^= x;\nexit;\n",
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ok"])
        self.assertIn("must be zero at exit", payload["error"])

    def test_check_rejects_from_jump_inconsistency(self) -> None:
        # Block b claims predecessor a, but a gotos c (not b)
        source = """(x) (x) ()
a: entry;
goto c;
b: from a;
goto c;
c: fi (= x 0) from b else b;
exit;
"""
        with self.assertRaises(CheckError):
            check_program(parse_program(source))

    def test_check_rejects_fifrom_jump_inconsistency(self) -> None:
        # fi-from claims predecessors b and c, but c gotos d (not this join block)
        source = """(x) (x) ()
a: entry;
if (= x 0) goto b else c;
b: from a;
goto join;
c: from a;
goto d;
d: from c;
goto join;
join: fi (= x 0) from b else c;
exit;
"""
        # join claims c as predecessor but c gotos d, not join → inconsistency
        with self.assertRaises(CheckError):
            check_program(parse_program(source))

    def test_check_accepts_consistent_fifrom(self) -> None:
        # Properly structured if: both branches goto join, fi-from claims them as predecessors
        source = """(x) (x y) ()
start: entry;
if (= x 0) goto then_b else else_b;
then_b: from start;
y += 1;
goto join;
else_b: from start;
goto join;
join: fi (= x 0) from then_b else else_b;
exit;
"""
        check_program(parse_program(source))  # should not raise


if __name__ == "__main__":
    unittest.main()
