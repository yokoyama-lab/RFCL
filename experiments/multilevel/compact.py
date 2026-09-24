#!/usr/bin/env python3
"""Code size and time of the loop-based compilation vs. the unrolled one.

For Bennett shapes ms = [2]*k (k = 1..10) and [4]*k (k = 1..5), i.e. n up to
1024, compile each step program both ways (pyrev_fl.pebble_compact and
pyrev_fl.pebble.compile_pebbling), run both, and write results/compact.csv.
Exits 1 if an output is wrong or the measured compact time differs from
unrolled time + control_overhead(ms, |X|).

Run from this directory:  python3 compact.py
"""
from __future__ import annotations

import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from pyrev_fl.interpreter import run_program  # noqa: E402
from pyrev_fl.parser import parse_program  # noqa: E402
from pyrev_fl.pebble import (  # noqa: E402
    bennett_mixed_schedule, compile_pebbling, iterate_reference, step_spec,
    validate_schedule,
)
from pyrev_fl.pebble_compact import (  # noqa: E402
    compile_bennett_compact, control_overhead, statement_count,
)
from pyrev_fl.tradeoff import _measure  # noqa: E402

STEPS = {"step_fib.srl": [0, 1], "step_lcg.srl": [7], "step_tri.srl": [100000]}
SHAPES = [[2] * k for k in range(1, 11)] + [[4] * k for k in range(1, 6)]


def main() -> int:
    rows, bad = [], 0
    for f, x0 in STEPS.items():
        spec = step_spec(parse_program(open(os.path.join(ROOT, "examples", f)).read()))
        for ms in SHAPES:
            t0 = time.monotonic()
            c = compile_bennett_compact(spec, ms)
            moves = bennett_mixed_schedule(ms)
            st = validate_schedule(moves, c.n)
            straight = compile_pebbling(spec, moves, c.n)
            mc, ms_ = _measure(c.program, x0), _measure(straight.program, x0)
            out = run_program(c.program, x0)
            ok = [out[o] for o in c.output_vars] == iterate_reference(spec, c.n, x0)
            exact = mc.time_steps == ms_.time_steps + control_overhead(ms, len(spec.state))
            bad += (not ok) + (not exact)
            rows.append({
                "step": f, "m": ms[0], "k": len(ms), "n": c.n, "moves": st.moves,
                "pebbles": st.pebbles,
                "size_compact": statement_count(c.program.body),
                "size_unrolled": statement_count(straight.program.body),
                "time_compact": mc.time_steps, "time_unrolled": ms_.time_steps,
                "overhead_per_move": round((mc.time_steps - ms_.time_steps) / st.moves, 3),
                "peak_compact": mc.peak_space, "peak_unrolled": ms_.peak_space,
                "output_ok": ok, "time_exact": exact,
            })
            r = rows[-1]
            print(f"{f:14s} m={ms[0]} k={len(ms):2d} n={c.n:5d} size {r['size_compact']:4d} vs "
                  f"{r['size_unrolled']:7d}  time {r['time_compact']:8d} vs {r['time_unrolled']:7d} "
                  f"({time.monotonic() - t0:.1f}s)", flush=True)
    with open(os.path.join(HERE, "results", "compact.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows, {bad} failures")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
