#!/usr/bin/env python3
"""Single-level (k=1) Bennett sweep over the RFCL SRL examples.

For every ``examples/*.srl`` program, run ``pyrev_fl.tradeoff.analyze_tradeoff``
on several increasing inputs and record measured vs. predicted time/space:

  T_pred            = 2*T_orig + |outputs| + 2      (RFCL's own single-level prediction)
  S_pred            = S_orig + |outputs| + 1        (RFCL's own single-level prediction)
  T_pred_klevel_k1  = 2*T_orig                      (Bennett k-level formula, k=1)
  S_pred_klevel_k1  = S_orig + log2(T_orig)         (Bennett k-level formula, k=1)

Only SRL is swept: the ``tradeoff`` CLI subcommand (pyrev_fl/cli.py::_tradeoff)
parses with ``pyrev_fl.parser.parse_program`` only, so ``.rl`` files are not
accepted by the analyzer.

Usage:  python3 experiments/pebbling/rfcl_k1_sweep.py   (from the repo root, or anywhere)
Output: experiments/pebbling/results/rfcl_k1_measurements.csv
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from pyrev_fl.parser import parse_program  # noqa: E402
from pyrev_fl.tradeoff import analyze_tradeoff  # noqa: E402

EXAMPLES = REPO / "examples"
OUT = HERE / "results" / "rfcl_k1_measurements.csv"

FIELDS = [
    "program", "inputs", "T_orig", "S_orig", "T_bennett", "S_bennett",
    "T_pred", "S_pred", "T_pred_klevel_k1", "S_pred_klevel_k1",
    "time_ratio", "space_ratio", "status", "error",
]


def _self_interp_inputs(n_instr: int) -> list[int]:
    """Flat input vector for self_interp.srl: code[30] n store[10].

    Encodes ``n_instr`` straight-line instructions (opcode, target, source)
    over a 10-cell store; targets differ from sources as the program requires.
    """
    ops = [1, 3, 2, 1, 3, 1, 2, 3, 1, 2]           # ADD / XOR / SUB cycle
    code: list[int] = []
    for i in range(n_instr):
        tgt = (i + 1) % 10
        src = (i + 2) % 10
        code += [ops[i], tgt, src]
    code += [0] * (30 - len(code))
    store = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    return code + [n_instr] + store


# program -> list of input vectors (increasing "size" for each program)
SWEEP: dict[str, list[list[int]]] = {
    "bennett_rif.srl":     [[1, 5, 7], [1, 10, 20], [0, 20, 40], [1, 40, 80], [0, 80, 160]],
    "branch_copy.srl":     [[5, 1], [10, 1], [20, 0], [40, 1], [80, 0]],
    "copy.srl":            [[5], [10], [20], [40], [80]],
    "countdown_clean.srl": [[5], [10], [20], [40]],
    "fib_bennett.srl":     [[5], [10], [15], [20]],
    "stack_demo.srl":      [[1, 2], [5, 10], [20, 40], [40, 80], [100, 200]],
    "self_interp.srl":     [_self_interp_inputs(k) for k in (1, 2, 3, 4, 5)],
}


def sweep() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    programs = sorted(EXAMPLES.glob("*.srl"))
    for path in programs:
        name = path.name
        input_sets = SWEEP.get(name)
        if input_sets is None:
            rows.append(dict.fromkeys(FIELDS, "") | {
                "program": name, "status": "error",
                "error": "no input vectors configured for this program",
            })
            continue
        for inputs in input_sets:
            row: dict[str, object] = dict.fromkeys(FIELDS, "")
            row["program"] = name
            row["inputs"] = " ".join(str(v) for v in inputs)
            try:
                program = parse_program(path.read_text())
                n_out = len(program.outputs)
                res = analyze_tradeoff(program, inputs)
                T, S = res.original.time_steps, res.original.peak_space
                Tb, Sb = res.bennett.time_steps, res.bennett.peak_space
                row.update({
                    "T_orig": T, "S_orig": S,
                    "T_bennett": Tb, "S_bennett": Sb,
                    "T_pred": 2 * T + n_out + 2,
                    "S_pred": S + n_out + 1,
                    "T_pred_klevel_k1": 2 * T,
                    "S_pred_klevel_k1": f"{S + math.log2(T):.3f}" if T > 0 else "",
                    "time_ratio": f"{Tb / T:.3f}" if T else "",
                    "space_ratio": f"{Sb / S:.3f}" if S else "",
                    "status": "ok",
                })
            except Exception as exc:  # checker / arity / transform errors
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    return rows


def main() -> int:
    rows = sweep()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"wrote {OUT} ({len(rows)} rows, {ok} ok, {len(rows) - ok} error)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
