from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from pyrev_fl.debugger import DebugSession, DebugState, debug_trace
from pyrev_fl.parser import parse_program

EXAMPLES = Path(__file__).parent.parent / "examples"


class DebugStepForwardTests(unittest.TestCase):
    def test_step_forward_copy(self):
        program = parse_program((EXAMPLES / "copy.srl").read_text())
        session = DebugSession(program, [3])
        # copy.srl has one statement: y += x;
        state = session.step_forward()
        self.assertEqual(state.store["y"], 3)
        self.assertEqual(state.store["x"], 3)
        self.assertIn("y", state.stmt_text)
        self.assertIn("+=", state.stmt_text)
        self.assertTrue(session.is_done())

    def test_step_backward_undoes(self):
        program = parse_program((EXAMPLES / "copy.srl").read_text())
        session = DebugSession(program, [3])
        session.step_forward()
        self.assertEqual(session.current_state().store["y"], 3)
        state = session.step_backward()
        self.assertEqual(state.store["y"], 0)
        self.assertEqual(state.store["x"], 3)
        self.assertIn("(undo)", state.stmt_text)

    def test_full_forward_backward(self):
        source = "(x) (x y) ()\ny += x;\nx += 1;\n"
        program = parse_program(source)
        session = DebugSession(program, [3])

        # Step all the way forward
        session.step_forward()
        session.step_forward()
        self.assertTrue(session.is_done())
        self.assertEqual(session.current_state().store["y"], 3)
        self.assertEqual(session.current_state().store["x"], 4)

        # Step all the way back
        session.step_backward()
        self.assertEqual(session.current_state().store["x"], 3)
        self.assertEqual(session.current_state().store["y"], 3)

        session.step_backward()
        self.assertEqual(session.current_state().store["x"], 3)
        self.assertEqual(session.current_state().store["y"], 0)

    def test_history_records_states(self):
        source = "(x) (x y) ()\ny += x;\nx += 1;\n"
        program = parse_program(source)
        session = DebugSession(program, [3])
        session.run_to_end()
        history = session.history
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].store["y"], 3)
        self.assertEqual(history[0].store["x"], 3)
        self.assertEqual(history[1].store["y"], 3)
        self.assertEqual(history[1].store["x"], 4)

    def test_branch_copy_debug(self):
        source = (EXAMPLES / "branch_copy.srl").read_text()
        program = parse_program(source)
        session = DebugSession(program, [5, 1])

        # First step should be the if statement (branch decision)
        state = session.step_forward()
        self.assertIn("if", state.stmt_text)

        # Step through the then-branch body
        session.run_to_end()
        final = session.current_state()
        self.assertEqual(final.store["x"], 5)
        self.assertEqual(final.store["y"], 5)
        self.assertEqual(final.store["flag"], 1)

    def test_countdown_debug(self):
        source = (EXAMPLES / "countdown_clean.srl").read_text()
        program = parse_program(source)
        # countdown_clean with n=3: loop iterates, acc accumulates
        session = DebugSession(program, [3])
        session.run_to_end()
        final = session.current_state()
        self.assertEqual(final.store["n"], 0)
        self.assertEqual(final.store["acc"], 7)

        # Verify we can step all the way back to initial state
        while session._snapshots:
            session.step_backward()
        initial = session.current_state()
        self.assertEqual(initial.store["n"], 3)
        self.assertEqual(initial.store["acc"], 0)


class DebugTraceTests(unittest.TestCase):
    def test_debug_trace_copy(self):
        program = parse_program((EXAMPLES / "copy.srl").read_text())
        trace = debug_trace(program, [3])
        # First entry is init, then forward steps, then backward steps
        self.assertEqual(trace[0]["action"], "init")
        self.assertEqual(trace[0]["store"]["y"], 0)
        forwards = [e for e in trace if e["action"] == "forward"]
        backwards = [e for e in trace if e["action"] == "backward"]
        self.assertGreaterEqual(len(forwards), 1)
        self.assertEqual(len(forwards), len(backwards))

    def test_run_to_end_then_back(self):
        source = "(a b) (a b) ()\na <=> b;\n"
        program = parse_program(source)
        session = DebugSession(program, [10, 20])
        session.run_to_end()
        self.assertEqual(session.current_state().store["a"], 20)
        self.assertEqual(session.current_state().store["b"], 10)
        session.step_backward()
        self.assertEqual(session.current_state().store["a"], 10)
        self.assertEqual(session.current_state().store["b"], 20)


class DebugCliTests(unittest.TestCase):
    def test_cli_debug_copy(self):
        result = subprocess.run(
            [sys.executable, "-m", "pyrev_fl.cli", "debug", "--json",
             str(EXAMPLES / "copy.srl"), "3"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        self.assertGreater(len(payload["trace"]), 0)
        # First trace entry should be init
        self.assertEqual(payload["trace"][0]["action"], "init")

    def test_cli_debug_countdown(self):
        result = subprocess.run(
            [sys.executable, "-m", "pyrev_fl.cli", "debug", "--json",
             str(EXAMPLES / "countdown_clean.srl"), "3"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        forward_steps = [e for e in payload["trace"] if e["action"] == "forward"]
        backward_steps = [e for e in payload["trace"] if e["action"] == "backward"]
        self.assertEqual(len(forward_steps), len(backward_steps))


if __name__ == "__main__":
    unittest.main()
