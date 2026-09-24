#!/usr/bin/env python3
"""Figure and table for the multi-level Bennett sweep (reads results/*.csv).

Writes results/fig_tradeoff.{png,pdf,svg} and results/bennett_vs_optimal.md.
Needs matplotlib.  Run from this directory:  python3 analyze.py
"""
from __future__ import annotations

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from pyrev_fl.pebble import min_moves  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = os.path.join(HERE, "results")


def load(name):
    with open(os.path.join(RES, name)) as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    sched = load("schedules.csv")
    meas = load("measured.csv")
    ns = sorted({int(r["n"]) for r in sched})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    colors = dict(zip(ns, ["#1f77b4", "#d62728", "#2ca02c"]))
    for n in ns:
        s_lo = max(1, (n - 1).bit_length() + 1)
        xs = list(range(s_lo, n + 1))
        ax.step(xs, [min_moves(n, s) / n for s in xs], where="post",
                color=colors[n], lw=1.4, label=f"optimal, n={n}")
        pts = [r for r in sched if int(r["n"]) == n and r["family"] == "bennett"]
        ax.scatter([int(r["pebbles"]) for r in pts], [int(r["moves"]) / n for r in pts],
                   marker="s", color=colors[n], edgecolor="k", zorder=3,
                   label=f"Bennett 1989 (k, m), n={n}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("pebbles = checkpoint registers (space / |state|)")
    ax.set_ylabel("step applications / n (time ratio)")
    ax.set_title("(a) time-space tradeoff of the pebble game")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for f, mk in (("step_fib.srl", "o"), ("step_lcg.srl", "^"), ("step_tri.srl", "x")):
        rows = [r for r in meas if r["step"] == f]
        ax.scatter([int(r["time_pred"]) for r in rows], [int(r["time"]) for r in rows],
                   marker=mk, s=18, label=f)
    lim = max(int(r["time"]) for r in meas) * 1.5
    ax.plot([1, lim], [1, lim], "k--", lw=0.8, label="measured = predicted")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time predicted from the schedule (sum of step costs)")
    ax.set_ylabel("time measured by running the compiled SRL")
    ax.set_title(f"(b) {len(meas)} compiled programs")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(RES, f"fig_tradeoff.{ext}"), dpi=150,
                    metadata={"Date": None} if ext == "pdf" else
                    ({"Date": None} if ext == "svg" else None))

    lines = ["| n | Bennett (k, m) | pebbles | Bennett moves | optimal moves, same pebbles | saving | "
             "pebbles optimal needs for Bennett's moves |", "|---|---|---|---|---|---|---|"]
    for r in sched:
        if r["family"] != "bennett":
            continue
        n, p, mv = int(r["n"]), int(r["pebbles"]), int(r["moves"])
        opt = min_moves(n, p)
        need = next(s for s in range(1, n + 1) if min_moves(n, s) <= mv)
        lines.append(f"| {n} | {r['schedule'][8:-1]} | {p} | {mv} | {int(opt)} | "
                     f"{100 * (mv - opt) / mv:.1f} % | {need} |")
    ok = sum(r["time"] == r["time_pred"] and r["output_ok"] == "True" for r in meas)
    lines += ["", f"Measured: {ok}/{len(meas)} compiled programs have measured time equal to "
              "the schedule's prediction and the correct x_n."]
    with open(os.path.join(RES, "bennett_vs_optimal.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
