#!/usr/bin/env python3
"""Time-space tradeoff curves of multi-level Bennett, measured in RFCL.

For each path length n and each pebble budget s, take the fewest-move
schedule (pyrev_fl.pebble.optimal_schedule), and for n = m**k also Bennett
1989's (k, m) schedule and the linear (history-keeping) one; compile each to
SRL with three step programs from examples/ and run it.  Writes

  results/schedules.csv    one row per (n, schedule): moves, pebbles
  results/measured.csv     one row per (n, schedule, step): measured time and
                           space, and the time predicted exactly from the
                           schedule (sum over moves of the step's cost there)

Run from this directory:  python3 sweep.py
"""
from __future__ import annotations

import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from pyrev_fl.parser import parse_program  # noqa: E402
from pyrev_fl.pebble import (  # noqa: E402
    analyze_pebbling, bennett_schedule, linear_schedule,
    min_moves, optimal_schedule, step_spec, validate_schedule,
)
from pyrev_fl.ast import Program  # noqa: E402
from pyrev_fl.interpreter import run_program  # noqa: E402
from pyrev_fl.tradeoff import _measure  # noqa: E402

STEPS = {  # step file -> x_0
    "step_fib.srl": [0, 1],
    "step_lcg.srl": [7],
    "step_tri.srl": [100000],
}
NS = [16, 64, 256]
MEASURE_NS = {16, 64}          # n = 256 is only counted, not run, except the rows below
MEASURE_256 = {"linear", "bennett(k=8,m=2)", "bennett(k=4,m=4)", "bennett(k=2,m=16)",
               "optimal(s=9)", "optimal(s=12)", "optimal(s=16)", "optimal(s=24)"}


def bennett_points(n: int):
    for m in range(2, n + 1):
        k = round(math.log(n, m))
        if k >= 1 and m ** k == n and m < n:
            yield k, m


def schedules(n: int):
    yield "linear", linear_schedule(n)
    for k, m in bennett_points(n):
        yield f"bennett(k={k},m={m})", bennett_schedule(k, m)
    s_min = math.ceil(math.log2(n)) + 1
    for s in range(s_min, n + 1):
        if min_moves(n, s) == min_moves(n, s - 1) and s > s_min:
            continue  # the curve is flat here; keep only its corners
        yield f"optimal(s={s})", optimal_schedule(n, s)


def step_costs(spec, n, x0):
    """Measured cost of the forward step and of the inverse step at each node."""
    from pyrev_fl.invert import invert_block
    fwd, bwd = [], []
    state = list(x0)
    for _ in range(n):
        prog = spec.program
        fwd.append(_measure(prog, state).time_steps)
        store = run_program(prog, state)
        nxt = [store[y] for y in spec.next_state]
        inv = Program(inputs=list(prog.outputs), outputs=list(prog.inputs),
                      temps=list(prog.temps), body=invert_block(prog.body))
        bwd.append(_measure(inv, state + nxt).time_steps)
        state = nxt
    return fwd, bwd


def main() -> int:
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    specs = {f: step_spec(parse_program(open(os.path.join(ROOT, "examples", f)).read()))
             for f in STEPS}
    sched_rows, meas_rows = [], []
    for n in NS:
        costs = {f: step_costs(specs[f], n, STEPS[f]) for f in STEPS}
        for label, moves in schedules(n):
            st = validate_schedule(moves, n)
            family = label.split("(")[0]
            sched_rows.append({"n": n, "schedule": label, "family": family,
                               "moves": st.moves, "pebbles": st.pebbles,
                               "moves_per_step": round(st.moves / n, 4)})
            if n not in MEASURE_NS and label not in MEASURE_256:
                continue
            for f, x0 in STEPS.items():
                r = analyze_pebbling(specs[f], moves, n, x0)
                fwd, bwd = costs[f]
                pred = sum(fwd[mv - 1] if mv > 0 else bwd[-mv - 1] for mv in moves)
                meas_rows.append({
                    "n": n, "schedule": label, "family": family, "step": f,
                    "moves": st.moves, "pebbles": st.pebbles,
                    "state_size": len(specs[f].state), "temps": len(specs[f].temps),
                    "time": r.time_steps, "time_pred": pred,
                    "baseline_time": r.baseline_time,
                    "time_ratio": round(r.time_steps / r.baseline_time, 4),
                    "peak_space": r.peak_space, "declared_space": r.declared_space,
                    "space_pred": len(specs[f].state) * (st.pebbles + 1) + len(specs[f].temps),
                    "output_ok": list(r.output.values()) == _ref(specs[f], n, x0),
                })
            print(f"n={n:4d} {label:22s} moves={st.moves:6d} pebbles={st.pebbles:4d}", flush=True)
    for name, rows in (("schedules.csv", sched_rows), ("measured.csv", meas_rows)):
        with open(os.path.join(out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    bad = [r for r in meas_rows if r["time"] != r["time_pred"] or not r["output_ok"]
           or r["declared_space"] != r["space_pred"]]
    print(f"{len(meas_rows)} measured rows, {len(bad)} deviating from the exact prediction")
    return 1 if bad else 0


_REF: dict = {}


def _ref(spec, n, x0):
    from pyrev_fl.pebble import iterate_reference
    key = (id(spec), n, tuple(x0))
    if key not in _REF:
        _REF[key] = iterate_reference(spec, n, x0)
    return _REF[key]


if __name__ == "__main__":
    sys.exit(main())
