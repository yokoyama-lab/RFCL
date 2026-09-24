# Multi-level Bennett (pebble game) tradeoff curves

Measures `pyrev_fl/pebble.py` (multi-level Bennett transform) on three step
programs (`examples/step_fib.srl`, `step_lcg.srl`, `step_tri.srl`): time and
space against the number of pebbles, for the fewest-move schedules, Bennett
1989's (k, m) schedules and the linear one. Result: `REPORT.md` (1 figure, 1 table).

Files: `sweep.py` (compiles and runs every schedule, writes
`results/schedules.csv`, `results/measured.csv`; exits 1 if a measured time
differs from the schedule's prediction), `analyze.py` (writes
`results/fig_tradeoff.{png,pdf,svg}` and `results/bennett_vs_optimal.md`;
needs matplotlib).

    python3 sweep.py
    python3 analyze.py
    python3 compact.py   # loop-based compilation vs. unrolled (results/compact.csv)

Pure RFCL, no external checkouts needed.
