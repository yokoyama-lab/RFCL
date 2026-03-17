"""Tests for the SRL -> quantum circuit compiler."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from pyrev_fl.ast import Assign, Block, Const, Program, Swap, UpdateOp, Var
from pyrev_fl.parser import parse_program
from pyrev_fl.quantum import QuantumCircuit, QuantumGate, compile_to_circuit

EXAMPLES = Path(__file__).parent.parent / "examples"

BITS = 4  # use a small bit-width for concise tests


class XorCircuitTests(unittest.TestCase):
    """XOR (^=) should produce one CNOT per bit."""

    def test_xor_generates_cnots(self):
        # y ^= x  with 4-bit registers  -> 4 CNOT gates
        program = Program(
            inputs=["x"],
            outputs=["x", "y"],
            temps=[],
            body=Block([Assign("y", UpdateOp.XOR, Var("x"))]),
        )
        circuit = compile_to_circuit(program, [5], bits=BITS)
        cnots = [g for g in circuit.gates if g.kind == "CNOT"]
        # One CNOT per bit position
        self.assertEqual(len(cnots), BITS)
        # Each CNOT has the right control/target pattern
        for i, gate in enumerate(cnots):
            self.assertEqual(gate.controls, [f"x_b{i}"])
            self.assertEqual(gate.targets, [f"y_b{i}"])


class SwapCircuitTests(unittest.TestCase):
    """Swap should produce SWAP gates."""

    def test_swap_generates_swap_gates(self):
        program = Program(
            inputs=["x", "y"],
            outputs=["x", "y"],
            temps=[],
            body=Block([Swap("x", "y")]),
        )
        circuit = compile_to_circuit(program, [3, 7], bits=BITS)
        swaps = [g for g in circuit.gates if g.kind == "SWAP"]
        self.assertEqual(len(swaps), BITS)
        for i, gate in enumerate(swaps):
            self.assertIn(f"x_b{i}", gate.targets)
            self.assertIn(f"y_b{i}", gate.targets)


class AddCircuitTests(unittest.TestCase):
    """ADD (+=) should produce an abstract adder gate."""

    def test_add_generates_adder(self):
        program = Program(
            inputs=["x"],
            outputs=["x", "y"],
            temps=[],
            body=Block([Assign("y", UpdateOp.ADD, Var("x"))]),
        )
        circuit = compile_to_circuit(program, [2], bits=BITS)
        adds = [g for g in circuit.gates if g.kind == "ADD"]
        self.assertEqual(len(adds), 1)
        self.assertEqual(adds[0].targets, [f"y_b{i}" for i in range(BITS)])
        self.assertEqual(adds[0].controls, [f"x_b{i}" for i in range(BITS)])


class CopyCircuitTests(unittest.TestCase):
    """The copy.srl example should compile to a valid circuit."""

    def test_copy_circuit(self):
        program = parse_program((EXAMPLES / "copy.srl").read_text())
        circuit = compile_to_circuit(program, [42], bits=8)
        # copy.srl is: y += x;  — produces one ADD gate plus input encoding
        self.assertGreater(len(circuit.gates), 0)
        self.assertGreater(len(circuit.qubits), 0)
        # The circuit should mention both x and y qubits
        qubit_names = set(circuit.qubits)
        self.assertTrue(any("x_b" in q for q in qubit_names))
        self.assertTrue(any("y_b" in q for q in qubit_names))


class QasmOutputTests(unittest.TestCase):
    """QASM output should be syntactically reasonable."""

    def test_circuit_qasm_output(self):
        program = Program(
            inputs=["x"],
            outputs=["x", "y"],
            temps=[],
            body=Block([Assign("y", UpdateOp.XOR, Var("x"))]),
        )
        circuit = compile_to_circuit(program, [3], bits=BITS)
        qasm = circuit.to_qasm()
        # Check required OpenQASM 2.0 elements
        self.assertIn("OPENQASM 2.0;", qasm)
        self.assertIn('include "qelib1.inc";', qasm)
        self.assertIn("qreg q[", qasm)
        self.assertIn("creg c[", qasm)
        self.assertIn("measure", qasm)
        # CNOT should appear as "cx"
        self.assertIn("cx ", qasm)


class GateCountTests(unittest.TestCase):
    """Gate counts should be reasonable for simple programs."""

    def test_gate_count(self):
        # y ^= x  with 4-bit: 4 CNOTs + input-encoding NOTs
        program = Program(
            inputs=["x"],
            outputs=["x", "y"],
            temps=[],
            body=Block([Assign("y", UpdateOp.XOR, Var("x"))]),
        )
        circuit = compile_to_circuit(program, [5], bits=BITS)
        counts = circuit.gate_count()
        self.assertEqual(counts.get("CNOT", 0), BITS)
        # Input value 5 = 0b0101 -> 2 NOT gates
        self.assertEqual(counts.get("NOT", 0), 2)

    def test_gate_count_multiple_statements(self):
        # y ^= x; y ^= x => 2*BITS CNOTs
        program = Program(
            inputs=["x"],
            outputs=["x", "y"],
            temps=[],
            body=Block([
                Assign("y", UpdateOp.XOR, Var("x")),
                Assign("y", UpdateOp.XOR, Var("x")),
            ]),
        )
        circuit = compile_to_circuit(program, [0], bits=BITS)
        counts = circuit.gate_count()
        self.assertEqual(counts.get("CNOT", 0), 2 * BITS)


class CircuitDepthTests(unittest.TestCase):
    """Circuit depth calculation."""

    def test_circuit_depth(self):
        # A single layer of CNOTs on disjoint qubits can run in parallel
        # y ^= x: all CNOTs operate on distinct (x_bi, y_bi) pairs -> depth 1 for body
        # But input encoding may add depth.
        program = Program(
            inputs=["x"],
            outputs=["x", "y"],
            temps=[],
            body=Block([Assign("y", UpdateOp.XOR, Var("x"))]),
        )
        # Input 0 — no encoding NOTs, so all CNOTs are parallel -> depth 1
        circuit = compile_to_circuit(program, [0], bits=BITS)
        self.assertEqual(circuit.depth(), 1)

    def test_circuit_depth_sequential(self):
        # x ^= y; then y ^= x — second set of CNOTs depends on first
        # x_b0: CNOT target, then CNOT control -> depth 2
        program = Program(
            inputs=["x", "y"],
            outputs=["x", "y"],
            temps=[],
            body=Block([
                Assign("x", UpdateOp.XOR, Var("y")),
                Assign("y", UpdateOp.XOR, Var("x")),
            ]),
        )
        circuit = compile_to_circuit(program, [0, 0], bits=BITS)
        self.assertEqual(circuit.depth(), 2)

    def test_empty_circuit_depth(self):
        circuit = QuantumCircuit(qubits=["q0", "q1"])
        self.assertEqual(circuit.depth(), 0)


class SubCircuitTests(unittest.TestCase):
    """SUB (-=) should produce an abstract subtractor gate."""

    def test_sub_generates_subtractor(self):
        program = Program(
            inputs=["x", "y"],
            outputs=["x", "y"],
            temps=[],
            body=Block([Assign("x", UpdateOp.SUB, Var("y"))]),
        )
        circuit = compile_to_circuit(program, [5, 3], bits=BITS)
        subs = [g for g in circuit.gates if g.kind == "SUB"]
        self.assertEqual(len(subs), 1)


class XorConstCircuitTests(unittest.TestCase):
    """XOR with a constant should produce NOT gates."""

    def test_xor_const_generates_nots(self):
        # x ^= 3;  3 = 0b11 -> 2 NOT gates on x_b0 and x_b1
        program = Program(
            inputs=["x"],
            outputs=["x"],
            temps=[],
            body=Block([Assign("x", UpdateOp.XOR, Const(3))]),
        )
        circuit = compile_to_circuit(program, [0], bits=BITS)
        # Only body gates (input is 0 so no encoding NOTs)
        nots = [g for g in circuit.gates if g.kind == "NOT"]
        self.assertEqual(len(nots), 2)
        self.assertEqual(nots[0].targets, ["x_b0"])
        self.assertEqual(nots[1].targets, ["x_b1"])


class CliToCircuitTests(unittest.TestCase):
    """CLI to-circuit subcommand."""

    def test_cli_to_circuit_text(self):
        result = subprocess.run(
            [sys.executable, "-m", "pyrev_fl.cli", "to-circuit",
             str(EXAMPLES / "copy.srl"), "5", "--bits", "4"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Quantum circuit:", result.stdout)
        self.assertIn("Gate counts:", result.stdout)

    def test_cli_to_circuit_qasm(self):
        result = subprocess.run(
            [sys.executable, "-m", "pyrev_fl.cli", "to-circuit",
             str(EXAMPLES / "copy.srl"), "5", "--bits", "4", "--qasm"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OPENQASM 2.0;", result.stdout)


if __name__ == "__main__":
    unittest.main()
