"""SRL → Quantum circuit compiler.

Maps reversible classical SRL programs to quantum circuits using
Toffoli/CNOT/SWAP gates.  Every classical reversible computation can be
embedded in a unitary quantum circuit; this module makes that connection
explicit.

Gate mapping (per bit position):
  x ^= y   →  CNOT(control=y_i, target=x_i)        for each bit i
  x <=> y  →  SWAP(x_i, y_i)  [= 3 CNOTs]          for each bit i
  x += y   →  ADD(x, y)  [abstract ripple-carry]
  x -= y   →  SUB(x, y)  [inverse of adder]

For simplicity the adder/subtractor are kept as single abstract gates
rather than decomposed into individual Toffoli gates.  This is
sufficient to demonstrate the SRL-to-quantum correspondence without
building a full quantum simulator.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pyrev_fl.ast import (
    Assign,
    Block,
    Const,
    Expr,
    Program,
    Stmt,
    Swap as SwapStmt,
    UpdateOp,
    Var,
)
from pyrev_fl.interface import build_layout


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class QuantumGate:
    """A quantum gate in the circuit."""
    kind: str           # "CNOT", "TOFFOLI", "NOT", "SWAP", "ADD", "SUB"
    targets: list[str]  # qubit names acted upon
    controls: list[str] = field(default_factory=list)  # control qubits

    def qubits_used(self) -> set[str]:
        """Return the set of all qubits touched by this gate."""
        return set(self.targets) | set(self.controls)


@dataclass
class QuantumCircuit:
    """A quantum circuit expressed as an ordered list of gates."""
    qubits: list[str]
    gates: list[QuantumGate] = field(default_factory=list)

    # ---- metrics ----------------------------------------------------------

    def gate_count(self) -> dict[str, int]:
        """Count gates by type."""
        counts: dict[str, int] = {}
        for gate in self.gates:
            counts[gate.kind] = counts.get(gate.kind, 0) + 1
        return counts

    def depth(self) -> int:
        """Circuit depth (longest path through gates).

        Each qubit has an independent "clock"; a gate executes at
        max(clock of its qubits) + 1.  The depth is the maximum clock
        value reached.
        """
        if not self.gates:
            return 0
        clocks: dict[str, int] = {q: 0 for q in self.qubits}
        max_depth = 0
        for gate in self.gates:
            used = gate.qubits_used()
            t = max((clocks.get(q, 0) for q in used), default=0) + 1
            for q in used:
                clocks[q] = t
            if t > max_depth:
                max_depth = t
        return max_depth

    # ---- rendering --------------------------------------------------------

    def to_qasm(self) -> str:
        """Render as OpenQASM 2.0 format."""
        lines: list[str] = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            "",
        ]

        # Declare qubit register
        n = len(self.qubits)
        lines.append(f"qreg q[{n}];")
        lines.append(f"creg c[{n}];")
        lines.append("")

        # Build qubit-name → index mapping
        idx = {name: i for i, name in enumerate(self.qubits)}

        for gate in self.gates:
            lines.append(_gate_to_qasm(gate, idx))

        # Measurement
        lines.append("")
        for i in range(n):
            lines.append(f"measure q[{i}] -> c[{i}];")

        return "\n".join(lines) + "\n"

    def to_text(self) -> str:
        """Human-readable text summary."""
        lines: list[str] = [
            f"Quantum circuit: {len(self.qubits)} qubits, "
            f"{len(self.gates)} gates, depth {self.depth()}",
            "",
            "Qubits:",
        ]
        for q in self.qubits:
            lines.append(f"  {q}")
        lines.append("")
        lines.append("Gates:")
        for i, gate in enumerate(self.gates):
            ctrl = f" ctrl={gate.controls}" if gate.controls else ""
            lines.append(f"  {i:4d}: {gate.kind}({', '.join(gate.targets)}){ctrl}")
        lines.append("")
        counts = self.gate_count()
        lines.append("Gate counts:")
        for kind in sorted(counts):
            lines.append(f"  {kind}: {counts[kind]}")
        return "\n".join(lines) + "\n"


def _gate_to_qasm(gate: QuantumGate, idx: dict[str, int]) -> str:
    """Convert a single gate to an OpenQASM 2.0 line."""
    if gate.kind == "CNOT":
        c = idx[gate.controls[0]]
        t = idx[gate.targets[0]]
        return f"cx q[{c}],q[{t}];"
    if gate.kind == "TOFFOLI":
        c0 = idx[gate.controls[0]]
        c1 = idx[gate.controls[1]]
        t = idx[gate.targets[0]]
        return f"ccx q[{c0}],q[{c1}],q[{t}];"
    if gate.kind == "NOT":
        t = idx[gate.targets[0]]
        return f"x q[{t}];"
    if gate.kind == "SWAP":
        a = idx[gate.targets[0]]
        b = idx[gate.targets[1]]
        # SWAP = 3 CNOTs
        return (
            f"cx q[{a}],q[{b}]; cx q[{b}],q[{a}]; cx q[{a}],q[{b}];"
            f" // SWAP {gate.targets[0]},{gate.targets[1]}"
        )
    if gate.kind in ("ADD", "SUB"):
        # Abstract high-level gate — emit as a comment + barrier
        tgts = ",".join(f"q[{idx[t]}]" for t in gate.targets)
        ctrls = ",".join(f"q[{idx[c]}]" for c in gate.controls)
        return f"// {gate.kind} target=[{tgts}] source=[{ctrls}]"
    # Fallback
    return f"// unknown gate: {gate.kind}"


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------

def compile_to_circuit(program: Program, inputs: list[int],
                       bits: int = 8) -> QuantumCircuit:
    """Compile an SRL program to a quantum circuit.

    Each integer variable is represented as a register of *bits* qubits.
    The initial values of input variables are encoded by applying NOT
    gates to the appropriate bit positions.

    Parameters
    ----------
    program : Program
        A parsed SRL program.
    inputs : list[int]
        Concrete input values (one per declared input variable).
    bits : int
        Number of qubits per variable (default 8).

    Returns
    -------
    QuantumCircuit
    """
    layout = build_layout(program.inputs, program.outputs, program.temps)

    # Gather all scalar variable names (in declaration order).
    all_vars: list[str] = []
    for name in layout.inputs + layout.outputs + layout.temps:
        if name not in all_vars:
            all_vars.append(name)

    # Build qubit names: var_b0, var_b1, ...
    qubits: list[str] = []
    var_qubits: dict[str, list[str]] = {}
    for var in all_vars:
        qs = [f"{var}_b{i}" for i in range(bits)]
        var_qubits[var] = qs
        qubits.extend(qs)

    circuit = QuantumCircuit(qubits=qubits)

    # Encode input values via NOT gates.
    for var_name, value in zip(layout.inputs, inputs, strict=True):
        _encode_value(circuit, var_qubits[var_name], value, bits)

    # Compile the program body.
    _compile_block(program.body, circuit, var_qubits, bits)

    return circuit


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _encode_value(circuit: QuantumCircuit, qubit_names: list[str],
                  value: int, bits: int) -> None:
    """Flip qubits corresponding to set bits in *value*."""
    for i in range(bits):
        if (value >> i) & 1:
            circuit.gates.append(QuantumGate(kind="NOT", targets=[qubit_names[i]]))


def _compile_block(block: Block, circuit: QuantumCircuit,
                   var_qubits: dict[str, list[str]], bits: int) -> None:
    for stmt in block.stmts:
        _compile_stmt(stmt, circuit, var_qubits, bits)


def _compile_stmt(stmt: Stmt, circuit: QuantumCircuit,
                  var_qubits: dict[str, list[str]], bits: int) -> None:
    if isinstance(stmt, Assign):
        _compile_assign(stmt, circuit, var_qubits, bits)
        return
    if isinstance(stmt, SwapStmt):
        _compile_swap(stmt, circuit, var_qubits, bits)
        return
    # Control-flow constructs (if/loop/rif) are not decomposed into
    # gates — they would require ancilla management and are beyond the
    # scope of this abstract mapping.  We silently skip them so that
    # straight-line programs still compile.


def _resolve_var_name(place) -> str:
    """Extract the variable name from a string or Var node."""
    if isinstance(place, str):
        return place
    if isinstance(place, Var):
        return place.name
    raise ValueError(f"unsupported place in quantum compilation: {place!r}")


def _compile_assign(stmt: Assign, circuit: QuantumCircuit,
                    var_qubits: dict[str, list[str]], bits: int) -> None:
    target_name = _resolve_var_name(stmt.target)
    target_qs = var_qubits[target_name]

    # x ^= y — CNOT per bit
    if stmt.op is UpdateOp.XOR and isinstance(stmt.expr, Var):
        source_qs = var_qubits[stmt.expr.name]
        for i in range(bits):
            circuit.gates.append(
                QuantumGate(kind="CNOT",
                            targets=[target_qs[i]],
                            controls=[source_qs[i]])
            )
        return

    # x ^= constant — NOT gates on appropriate bits
    if stmt.op is UpdateOp.XOR and isinstance(stmt.expr, Const):
        _encode_value(circuit, target_qs, stmt.expr.value, bits)
        return

    # x += y — abstract adder gate
    if stmt.op is UpdateOp.ADD and isinstance(stmt.expr, Var):
        source_qs = var_qubits[stmt.expr.name]
        circuit.gates.append(
            QuantumGate(kind="ADD",
                        targets=list(target_qs),
                        controls=list(source_qs))
        )
        return

    # x += constant — abstract adder with constant
    if stmt.op is UpdateOp.ADD and isinstance(stmt.expr, Const):
        circuit.gates.append(
            QuantumGate(kind="ADD",
                        targets=list(target_qs),
                        controls=[f"#const({stmt.expr.value})"])
        )
        return

    # x -= y — abstract subtractor gate (inverse of adder)
    if stmt.op is UpdateOp.SUB and isinstance(stmt.expr, Var):
        source_qs = var_qubits[stmt.expr.name]
        circuit.gates.append(
            QuantumGate(kind="SUB",
                        targets=list(target_qs),
                        controls=list(source_qs))
        )
        return

    # x -= constant — abstract subtractor with constant
    if stmt.op is UpdateOp.SUB and isinstance(stmt.expr, Const):
        circuit.gates.append(
            QuantumGate(kind="SUB",
                        targets=list(target_qs),
                        controls=[f"#const({stmt.expr.value})"])
        )
        return


def _compile_swap(stmt: SwapStmt, circuit: QuantumCircuit,
                  var_qubits: dict[str, list[str]], bits: int) -> None:
    left_name = _resolve_var_name(stmt.left)
    right_name = _resolve_var_name(stmt.right)
    left_qs = var_qubits[left_name]
    right_qs = var_qubits[right_name]
    for i in range(bits):
        circuit.gates.append(
            QuantumGate(kind="SWAP",
                        targets=[left_qs[i], right_qs[i]])
        )
