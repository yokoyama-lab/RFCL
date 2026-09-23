#!/usr/bin/env python3
"""analyze.py -- classify the corpus measurements, fit the scaling series and
draw the one figure + one table of the "Bennett k-level predictions vs. real
reversible programs" experiment.

Reads   results/corpus_measurements.csv
        results/scaling_measurements.csv
        results/rfcl_k1_measurements.csv
Writes  results/classification.csv
        results/deviation_table.md
        results/fig_bennett_vs_measured.{png,pdf,svg}
        results/summary.json

Only the standard library, numpy and matplotlib are used.

Classes (applied in this order, first match wins; ok reversible-algorithms rows only)
  E  inverse-as-main     n_uncalls > 0 and fwd_steps < 0.10 * total_steps
  D  nested Bennett      time_ratio > 2.05
  C  single-level CCU    1.6 <= time_ratio <= 2.05
  B  partial uncompute   n_uncalls > 0 and time_ratio < 1.6
  A2 DP/table in-place   n_uncalls == 0 and family in {dp, floyd_warshall, bellman_ford, apsp}
  A1 in-place            every remaining n_uncalls == 0 program
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

CORPUS_CSV = RES / "corpus_measurements.csv"
SCALING_CSV = RES / "scaling_measurements.csv"
K1_CSV = RES / "rfcl_k1_measurements.csv"

CLASS_ORDER = ["A1", "A2", "B", "C", "D", "E"]
CLASS_NAME = {
    "A1": "in-place",
    "A2": "DP/table in-place",
    "B": "partial uncompute",
    "C": "single-level CCU",
    "D": "nested Bennett",
    "E": "inverse-as-main",
}
CLASS_RULE = {
    "A1": "n_uncalls = 0, other families",
    "A2": "n_uncalls = 0, family in {dp, floyd_warshall, bellman_ford, apsp}",
    "B": "n_uncalls > 0, time_ratio < 1.6",
    "C": "1.6 <= time_ratio <= 2.05",
    "D": "time_ratio > 2.05",
    "E": "n_uncalls > 0, fwd_steps < 0.10 * total_steps",
}
DP_FAMILIES = {"dp", "floyd_warshall", "bellman_ford", "apsp"}
CCU_LO, CCU_HI = 1.6, 2.05

# ---------------------------------------------------------------- dataviz tokens
# Categorical slots 1-3 of the reference palette validate all-pairs (scatter).
# Colour carries the Bennett *regime*; the marker shape carries the class.
BLUE, AQUA, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"
YELLOW, MAGENTA, GREEN, VIOLET, RED = "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

CLASS_STYLE = {          # (colour, marker)
    "A1": (BLUE, "o"),
    "A2": (BLUE, "s"),
    "B": (AQUA, "^"),
    "C": (AQUA, "D"),
    "D": (ORANGE, "P"),
    "E": (ORANGE, ">"),
}
# 8 scaling programs: adjacent-validated 8-slot order (lines).
PROGRAM_STYLE = {
    "numeric/gcd.j": (BLUE, "o", "gcd"),
    "others/code/factorial.j": (ORANGE, "s", "factorial"),
    "dp/lcs.j": (AQUA, "^", "lcs"),
    "dp/knapsack.j": (YELLOW, "v", "knapsack"),
    "dp/lis.j": (MAGENTA, "D", "lis"),
    "bsort/bsort2.j": (GREEN, "P", "bsort2"),
    "msort/msort1.j": (VIOLET, "X", "msort1"),
    "others/examples/fact.j": (RED, "*", "fact (recursive)"),
}
PROGRAM_ORDER = list(PROGRAM_STYLE)

# short labels for panel (a)
ANNOTATE = {
    "numeric/gcd.j": "gcd",
    "dp/knapsack.j": "knapsack",
    "dp/lcs.j": "lcs",
    "bsort/bsort2.j": "bsort2",
    "msort/msort1.j": "msort1",
    "hsort/code/hsort1.j": "hsort1",
    "others/examples/fact.j": "fact",
    "rank/rank.j": "rank",
    "rank/rank_lexicographic.j": "rank_lexicographic",
    "perm/code/perm2decfac.j": "perm2decfac",
}

X_CLIP = 3.2   # panel (a): class-E points beyond this are drawn at the right edge


# ---------------------------------------------------------------- helpers
def fnum(s: str) -> float:
    return float(s) if s not in ("", None) else math.nan


def inum(s: str) -> int:
    return int(float(s)) if s not in ("", None) else 0


def short(path: str) -> str:
    return Path(path).stem


def median(xs) -> float:
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return statistics.median(xs) if xs else math.nan


def lsq(x, y):
    """Least-squares slope, intercept of y on x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.ptp(x) == 0:
        return math.nan, math.nan
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def classify(r: dict) -> str:
    n_unc = inum(r["n_uncalls"])
    tr = fnum(r["time_ratio"])
    fwd, tot = inum(r["fwd_steps"]), inum(r["total_steps"])
    if n_unc > 0 and fwd < 0.10 * tot:
        return "E"
    if tr > CCU_HI:
        return "D"
    if CCU_LO <= tr <= CCU_HI:
        return "C"
    if n_unc > 0 and tr < CCU_LO:
        return "B"
    if n_unc == 0 and r["family"] in DP_FAMILIES:
        return "A2"
    return "A1"


# ---------------------------------------------------------------- load
def load_corpus():
    with CORPUS_CSV.open() as f:
        rows = list(csv.DictReader(f))
    ok = [r for r in rows if r["corpus"] == "reversible-algorithms" and r["status"] == "ok"]
    for r in ok:
        r["class"] = classify(r)
    return rows, ok


def load_scaling():
    with SCALING_CSV.open() as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    series: dict[str, list[dict]] = {}
    for r in rows:
        series.setdefault(r["program"], []).append(r)
    for v in series.values():
        v.sort(key=lambda r: int(r["size_value"]))
    return series


def load_k1():
    with K1_CSV.open() as f:
        rows = list(csv.DictReader(f))
    ok = [r for r in rows if r["status"] == "ok"]
    t_off = sorted({inum(r["T_bennett"]) - inum(r["T_pred"]) for r in ok})
    s_off = sorted({inum(r["S_bennett"]) - inum(r["S_pred"]) for r in ok})
    k1_t = sorted({inum(r["T_bennett"]) - inum(r["T_pred_klevel_k1"]) for r in ok})
    k1_s_err = [inum(r["S_bennett"]) - fnum(r["S_pred_klevel_k1"]) for r in ok]
    s_exact = sum(1 for r in ok if inum(r["S_bennett"]) == inum(r["S_pred"]))
    return {
        "n_S_exact": s_exact,
        "n_S_below_pred": len(ok) - s_exact,
        "S_below_pred_rows": [f'{r["program"]} [{r["inputs"]}]' for r in ok if inum(r["S_bennett"]) != inum(r["S_pred"])],
        "n_ok_rows": len(ok),
        "n_error_rows": len(rows) - len(ok),
        "programs_ok": sorted({r["program"] for r in ok}),
        "programs_error": sorted({r["program"] for r in rows if r["status"] != "ok"}),
        "T_bennett_minus_T_pred_values": t_off,
        "S_bennett_minus_S_pred_values": s_off,
        "T_bennett_minus_2T_values": k1_t,
        "S_bennett_minus_(S+log2T)_min_max": [round(min(k1_s_err), 3), round(max(k1_s_err), 3)],
        "T_pred_formula": "2*T + |outputs| + 2 (tradeoff.py:81)",
        "T_measured_formula": "2*T + |outputs| + 7  (= T_pred + 5)",
        "plus5_breakdown": {
            "loop_entry_guard": 1, "loop_iteration_test": 1, "loop_exit": 1,
            "rif_dispatch_forward": 1, "rif_dispatch_reverse": 1,
        },
    }


# ---------------------------------------------------------------- tables
def write_classification(ok):
    cols = ["path", "family", "class", "total_steps", "fwd_steps", "time_ratio",
            "transient_peak", "log2_T", "k_space", "k_time", "n_uncalls", "call_depth_max"]
    order = {c: i for i, c in enumerate(CLASS_ORDER)}
    rows = sorted(ok, key=lambda r: (order[r["class"]], r["path"]))
    with (RES / "classification.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])


PREDICTION = {
    "A1": "k = 0: reversible as written, no uncomputation; T_rev = T, S_extra = 0 (Bennett not applicable)",
    "A2": "k = 0 likewise; a pebbling view would charge the DP table as garbage (S_extra ~ table size) or log T for re-computation",
    "B": "between k = 0 and k = 1: T_rev = T + T_uncalled part, S_extra = O(|outputs of uncalled part|)",
    "C": "k = 1: T_rev = 2T; docstring S_extra = log2 T, Bennett 1989 S_extra = O(|outputs|) (constant)",
    "D": "k levels: docstring T_rev = T*2^k, S_extra = k*log2 T; nested CCU gives S_extra = k * (ancillas per level)",
    "E": "no prediction: T_rev/T is undefined when the algorithm *is* an uncall",
}
CAUSE = {
    "A1": "hand-written in-place algorithms need no CCU embedding; the ancilla is a constant number of loop/index variables",
    "A2": "the DP table is the *output* (live_final 5-11), not garbage; nothing is ever uncomputed, so time_ratio = 1 and the transient is 2 index variables",
    "B": "only a sub-procedure is call/uncalled (e.g. a search or heapify); the bulk of main stays forward",
    "C": "the CCU is real but main has extra forward work and the uncall repeats only the called body, so T_rev/T sits at 1.6-1.95, and the ancilla is a constant (median 4), not log2 T",
    "D": "recursion with call+uncall at every level doubles the work per level (k = depth) but adds ~1 ancilla per level, so space is k*const, not k*log2 T",
    "E": "the forward work is an `uncall` of a generating procedure; fwd/uncall counting inverts the roles and time_ratio is an accounting artifact",
}


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def class_stats(ok):
    stats = {}
    for c in CLASS_ORDER:
        rs = [r for r in ok if r["class"] == c]
        stats[c] = {
            "count": len(rs),
            "median_time_ratio": round(median([fnum(r["time_ratio"]) for r in rs]), 3) if rs else None,
            "median_transient_peak": median([fnum(r["transient_peak"]) for r in rs]) if rs else None,
            "median_k_space": round(median([fnum(r["k_space"]) for r in rs]), 3) if rs else None,
            "median_k_time": round(median([fnum(r["k_time"]) for r in rs]), 3) if rs else None,
            "members": [r["path"] for r in rs],
        }
        if c == "A1":
            stats[c]["transient_peak_eq_0"] = sum(1 for r in rs if inum(r["transient_peak"]) == 0)
            stats[c]["transient_peak_gt_0"] = sum(1 for r in rs if inum(r["transient_peak"]) > 0)
    return stats


def representatives(c, rs):
    pick = {
        "A1": ["numeric/gcd.j", "bsort/code/bsort1.j", "hsort/hsort2.j", "avl/avl_insert.j"],
        "A2": ["dp/knapsack.j", "dp/lcs.j", "dp/edit_distance.j", "floyd_warshall/floyd_warshall.j"],
        "B": ["dijkstra/dijkstra.j", "qsort/qsort2.j", "minheap/minheap_Nohr2015.j", "avl/avl_search.j"],
        "C": ["bsort/bsort2.j", "msort/msort1.j", "hsort/code/hsort1.j", "dp/lis.j"],
        "D": ["others/examples/fact.j", "perm/code/perm2decfac.j", "rank/rank.j", "isort/code/isort1.j"],
        "E": ["rank/rank_lexicographic.j", "others/code/decfac2rank.j", "ssort/code/ssort1.j"],
    }[c]
    have = {r["path"] for r in rs}
    return ", ".join(short(p) for p in pick if p in have)


def scaling_fits(series):
    fits = {}
    for prog, rs in series.items():
        sizes = [int(r["size_value"]) for r in rs]
        log2T = [fnum(r["log2_T"]) for r in rs]
        tp = [fnum(r["transient_peak"]) for r in rs]
        tr = [fnum(r["time_ratio"]) for r in rs]
        s_tp, _ = lsq(log2T, tp)
        s_lt, i_lt = lsq(sizes, log2T)
        s_ltr, i_ltr = lsq(sizes, np.log2(tr))
        fits[prog] = {
            "sizes": sizes,
            "time_ratio_first_last": [tr[0], tr[-1]],
            "transient_peak_first_last": [tp[0], tp[-1]],
            "slope_transient_peak_vs_log2T": round(s_tp, 3),
            "slope_log2T_vs_size": round(s_lt, 3),
            "intercept_log2T_vs_size": round(i_lt, 3),
            "base_T_vs_size": round(2 ** s_lt, 3),
            "slope_log2_time_ratio_vs_size": round(s_ltr, 3),
            "intercept_log2_time_ratio_vs_size": round(i_ltr, 3),
            "base_time_ratio_vs_size": round(2 ** s_ltr, 3),
        }
    return fits


def write_deviation_table(ok, stats, series, fits, k1):
    lines = ["# Bennett k-level predictions vs. measured reversible programs", ""]
    lines.append(f"Corpus: reversible-algorithms, {len(ok)} programs that run under PyJanus (jana2014 dialect). "
                 "Metrics: time_ratio = total_steps/fwd_steps, transient_peak = max_live_vars - max(live_init, live_final), "
                 "k_space = transient_peak/log2(total_steps), k_time = log2(time_ratio). Medians over the class.")
    lines.append("")
    lines.append("## Table 1 - deviation of each program class from the k-level prediction")
    lines.append("")
    rows = []
    for c in CLASS_ORDER:
        s = stats[c]
        rs = [r for r in ok if r["class"] == c]
        cnt = str(s["count"])
        if c == "A1":
            cnt += f" ({s['transient_peak_eq_0']} with transient_peak = 0, {s['transient_peak_gt_0']} with > 0)"
        sig = (f"median time_ratio {s['median_time_ratio']:.2f}, transient_peak {s['median_transient_peak']:g}, "
               f"k_space {s['median_k_space']:.2f}")
        rows.append([f"{c} {CLASS_NAME[c]}", CLASS_RULE[c], cnt, representatives(c, rs), sig, PREDICTION[c], CAUSE[c]])
    lines.append(md_table(["class", "rule", "count", "representative programs", "measured signature",
                           "Bennett k-level prediction for this shape", "cause of the deviation"], rows))
    lines.append("")
    lines.append("## Table 2 - every D and E program, and the dp family, individually")
    lines.append("")
    rows = []
    for r in sorted(ok, key=lambda r: (CLASS_ORDER.index(r["class"]), -fnum(r["time_ratio"]))):
        if r["class"] in ("D", "E") or r["family"] == "dp":
            rows.append([r["path"], r["class"], r["time_ratio"], r["transient_peak"], r["k_space"],
                         r["k_time"], r["call_depth_max"], r["n_uncalls"]])
    lines.append(md_table(["path", "class", "time_ratio", "transient_peak", "k_space", "k_time",
                           "call_depth_max", "n_uncalls"], rows))
    lines.append("")
    lines.append("## Table 3 - scaling series (6 input sizes per program)")
    lines.append("")
    rows = []
    for prog in PROGRAM_ORDER:
        f = fits[prog]
        rows.append([prog, f"{f['sizes'][0]} -> {f['sizes'][-1]}",
                     f"{f['time_ratio_first_last'][0]:.3g} -> {f['time_ratio_first_last'][1]:.3g}",
                     f"{f['transient_peak_first_last'][0]:g} -> {f['transient_peak_first_last'][1]:g}",
                     f"{f['slope_transient_peak_vs_log2T']:+.3f}",
                     f"{f['slope_log2T_vs_size']:.3f} (T x{f['base_T_vs_size']:.2f} per +1)"])
    lines.append(md_table(["program", "size", "time_ratio first -> last", "transient_peak first -> last",
                           "slope transient_peak vs log2 T", "slope log2 T vs size"], rows))
    lines.append("")
    lines.append("Reading: a k-level Bennett scheme with k growing as log2 T would show slope 1 in column 5; "
                 "a fixed single-level CCU shows 0. fact.j's slope is per unit of *n* (recursion depth), "
                 "which is the nested-embedding regime k = n, S_extra = k * 1 ancilla.")
    lines.append("")
    lines.append("## RFCL single-level check (rfcl_k1_measurements.csv)")
    lines.append("")
    lines.append(f"{k1['n_ok_rows']} ok rows over {len(k1['programs_ok'])} SRL programs: "
                 f"S_pred = S + |out| + 1 is an upper bound that is tight on {k1['n_S_exact']} rows; on the other {k1['n_S_below_pred']} rows "
                 f"({'; '.join(k1['S_below_pred_rows'])}) an output value is 0, so peak_space (a non-zero count) sits below it "
                 f"(S_bennett - S_pred in {k1['S_bennett_minus_S_pred_values']}); "
                 f"T_bennett - T_pred in {k1['T_bennett_minus_T_pred_values']} (constant +5), "
                 f"T_bennett - 2T in {k1['T_bennett_minus_2T_values']}.")
    (RES / "deviation_table.md").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


# ---------------------------------------------------------------- figure
def screen_angle(ax, p0, p1):
    """Screen-space angle (degrees) of the segment p0->p1 given in data coordinates."""
    (x0, y0), (x1, y1) = ax.transData.transform([p0, p1])
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, width=0.8, length=3)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def draw_figure(ok, stats, series, fits):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(15, 4.8),
                                           gridspec_kw={"width_ratios": [1.25, 1, 1], "wspace": 0.28})
    for ax in (ax_a, ax_b, ax_c):
        style_axes(ax)

    # ---- (a) k_time vs k_space
    ax = ax_a
    ax.axvspan(math.log2(CCU_LO), math.log2(CCU_HI), color=AQUA, alpha=0.10, lw=0, zorder=0)
    ax.text((math.log2(CCU_LO) + math.log2(CCU_HI)) / 2, -0.03, "time ratio\n1.6-2.05", ha="center",
            va="bottom", fontsize=7, color=INK2)
    xs_line = np.linspace(0, 1.45, 2)
    ax.plot(xs_line, xs_line, ls="--", lw=1.2, color=MUTED, zorder=1)
    ax.text(1.38, 1.47, "y = x: docstring k-level prediction\n(2^k time, k·log₂T space)",
            fontsize=6.8, color=INK2, ha="left", va="center")
    ax.plot([1], [1], marker="*", ms=15, color=INK, mec=SURFACE, mew=1.2, ls="none", zorder=6)
    ax.annotate("single-level Bennett (k = 1)", (1, 1), xytext=(1.1, 0.7), fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=0, shrinkB=6))

    rng = np.random.default_rng(3)
    overflow = []
    for c in CLASS_ORDER:
        col, mk = CLASS_STYLE[c]
        rs = [r for r in ok if r["class"] == c]
        xv = np.array([fnum(r["k_time"]) for r in rs])
        yv = np.array([fnum(r["k_space"]) for r in rs])
        # small horizontal jitter only for the x == 0 stack (values identical by construction)
        xj = xv + np.where(xv == 0, rng.uniform(-0.025, 0.025, len(xv)), 0.0)
        inside = xv <= X_CLIP
        ax.plot(xj[inside], yv[inside], ls="none", marker=mk, ms=6.5 if mk != "P" else 7.5, color=col,
                mec=SURFACE, mew=0.9, alpha=0.9, zorder=4, label=f"{c}  {CLASS_NAME[c]} (n = {len(rs)})")
        for r, x, y in zip(rs, xv, yv):
            if x > X_CLIP:
                overflow.append((r, x, y, col))
    for r, x, y, col in overflow:
        ax.plot([X_CLIP + 0.12], [y], marker=">", ms=8, color=col, mec=SURFACE, mew=0.9, ls="none", zorder=5)
    ax.axvline(X_CLIP, color=AXIS, lw=0.8, ls=(0, (2, 3)))
    ax.text(X_CLIP + 0.24, 1.55, "x clipped at 3.2;\nE points drawn at the\nedge, true k_time in label", fontsize=6.3,
            color=MUTED, ha="right", va="top")

    # annotations (short names), offsets tuned by hand
    by_path = {r["path"]: r for r in ok}
    offsets = {
        "gcd": (0.06, -0.035), "knapsack": (0.06, 0.02), "lcs": (0.06, -0.06),
        "bsort2": (-0.02, 0.06), "msort1": (0.07, -0.01), "hsort1": (0.04, 0.03),
        "fact": (0.05, -0.02), "rank": (0.05, 0.00), "rank_lexicographic": (-0.08, 0.04),
        "perm2decfac": (0.05, 0.00),
    }
    for p, name in ANNOTATE.items():
        r = by_path[p]
        x, y = fnum(r["k_time"]), fnum(r["k_space"])
        label = name
        if x > X_CLIP:
            label = f"{name} (k_time = {x:.1f})"
            x = X_CLIP + 0.12
        dx, dy = offsets[name]
        ha = "right" if dx < 0 else "left"
        ax.annotate(label, (x, y), xytext=(x + dx, y + dy), fontsize=7, color=INK, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=0, shrinkB=2), zorder=7)
    ax.set_xlim(-0.12, X_CLIP + 0.25)
    ax.set_ylim(-0.05, 1.56)
    ax.set_xlabel("k_time = log₂(total_steps / fwd_steps)")
    ax.set_ylabel("k_space = transient_peak / log₂(total_steps)")
    ax.set_title("(a) 120 reversible-algorithms programs, by class", loc="left")
    handles = [Line2D([], [], ls="none", marker=CLASS_STYLE[c][1], color=CLASS_STYLE[c][0], mec=SURFACE,
                      ms=6.5, label=f"{c}  {CLASS_NAME[c]} (n = {stats[c]['count']})") for c in CLASS_ORDER]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.0), ncol=1,
              handletextpad=0.4, labelcolor=INK2, borderaxespad=0.3, labelspacing=0.35)

    # ---- (b) transient_peak vs log2 T
    ax = ax_b
    xg = np.linspace(4, 16, 2)
    ax.plot(xg, xg, ls="--", lw=1.0, color=MUTED, zorder=1)
    ax.plot(xg, 2 * xg, ls="--", lw=1.0, color=MUTED, zorder=1)
    ref_labels = [
        ax.text(6.3, 6.3, "k = 1: S_extra = log₂T", fontsize=7, color=INK2, ha="center", va="center",
                rotation_mode="anchor", clip_on=True,
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none")),
        ax.text(5.6, 11.2, "k = 2: 2·log₂T", fontsize=7, color=INK2, ha="center", va="center",
                rotation_mode="anchor", clip_on=True,
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none")),
    ]
    for prog in PROGRAM_ORDER:
        col, mk, name = PROGRAM_STYLE[prog]
        rs = series[prog]
        x = [fnum(r["log2_T"]) for r in rs]
        y = [fnum(r["transient_peak"]) for r in rs]
        ax.plot(x, y, color=col, lw=2, marker=mk, ms=6.5 if mk != "*" else 9, mec=SURFACE, mew=0.9,
                solid_capstyle="round", label=name, zorder=4)
    # direct labels for the two rising series and one flat group
    ms1 = series["msort/msort1.j"][-1]
    ax.annotate("msort1: +1 per doubling of n", (fnum(ms1["log2_T"]), fnum(ms1["transient_peak"])),
                xytext=(9.0, 12.6), fontsize=7, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    fa = series["others/examples/fact.j"][3]
    ax.annotate("fact: n − 1 ancillas\n(one per recursion level)", (fnum(fa["log2_T"]), fnum(fa["transient_peak"])),
                xytext=(11.9, 6.9), fontsize=7, color=INK, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkB=4))
    ax.text(12.6, 5.35, "bsort2, lis: 5", fontsize=7, color=INK2, va="bottom")
    ax.text(4.1, 2.35, "factorial, lcs, knapsack: 2", fontsize=7, color=INK2, va="bottom")
    ax.text(6.9, 0.2, "gcd: 0", fontsize=7, color=INK2, va="bottom")
    ax.set_xlim(4, 16.2)
    ax.set_ylim(-0.4, 15.5)
    fig.canvas.draw()
    ref_labels[0].set_rotation(screen_angle(ax, (4, 4), (8, 8)))
    ref_labels[1].set_rotation(screen_angle(ax, (4, 8), (6, 12)))
    ax.set_xlabel("log₂(total_steps)")
    ax.set_ylabel("transient_peak (ancilla variables beyond inputs/outputs)")
    ax.set_title("(b) space scaling, 8 programs × 6 input sizes", loc="left")
    prog_handles = [Line2D([], [], color=PROGRAM_STYLE[p][0], lw=2, marker=PROGRAM_STYLE[p][1],
                           ms=6.5 if PROGRAM_STYLE[p][1] != "*" else 9, mec=SURFACE, mew=0.9,
                           label=PROGRAM_STYLE[p][2]) for p in PROGRAM_ORDER]

    # ---- (c) time_ratio vs input size (log y)
    ax = ax_c
    ax.set_yscale("log")
    ax.axhline(2, ls="--", lw=1.0, color=MUTED, zorder=1)
    ax.text(185, 2.12, "k = 1: T_rev = 2T", fontsize=7, color=INK2, ha="right", va="bottom")
    ff = fits["others/examples/fact.j"]
    n = np.linspace(2, 12, 50)
    ax.plot(n, 2 ** (ff["intercept_log2_time_ratio_vs_size"] + ff["slope_log2_time_ratio_vs_size"] * n),
            ls=":", lw=1.2, color=RED, zorder=2)
    for prog in PROGRAM_ORDER:
        col, mk, name = PROGRAM_STYLE[prog]
        rs = series[prog]
        x = [int(r["size_value"]) for r in rs]
        y = [fnum(r["time_ratio"]) for r in rs]
        ax.plot(x, y, color=col, lw=2, marker=mk, ms=6.5 if mk != "*" else 9, mec=SURFACE, mew=0.9,
                solid_capstyle="round", label=name, zorder=4)
    ax.annotate(f"fact: fitted {ff['base_time_ratio_vs_size']:.2f}ⁿ\n(T itself grows {ff['base_T_vs_size']:.2f}ⁿ)",
                (12, fits["others/examples/fact.j"]["time_ratio_first_last"][1]), xytext=(16, 180),
                fontsize=7, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.annotate("gcd, factorial, lcs,\nknapsack: 1 (no uncall;\ninline uncompute is\ncounted as forward)",
                (40, 1.0), xytext=(18, 11), fontsize=7, color=INK2, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkB=4))
    li = series["dp/lis.j"][4]
    ax.annotate("bsort2, msort1, lis: 1.57 → 1.98,\nconverging to 2 from below", (int(li["size_value"]), fnum(li["time_ratio"])),
                xytext=(11, 4.2), fontsize=7, color=INK2, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkB=4))
    ax.set_xscale("log", base=2)
    ax.set_xlim(1.7, 200)
    ax.set_ylim(0.85, 800)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels(["2", "4", "8", "16", "32", "64", "128"])
    ax.set_xlabel("input size parameter (n, items, Fibonacci index)")
    ax.set_ylabel("time_ratio = total_steps / fwd_steps")
    ax.set_title("(c) time scaling, same programs", loc="left")

    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.2)
    fig.legend(handles=prog_handles, loc="lower center", bbox_to_anchor=(0.67, 0.005), ncol=8,
               columnspacing=1.2, handletextpad=0.5, handlelength=2.4, labelcolor=INK2,
               title="programs in (b) and (c)", title_fontsize=7.5)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(RES / f"fig_bennett_vs_measured.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    all_rows, ok = load_corpus()
    series = load_scaling()
    k1 = load_k1()
    stats = class_stats(ok)
    fits = scaling_fits(series)

    print(f"reversible-algorithms ok programs: {len(ok)}")
    for c in CLASS_ORDER:
        s = stats[c]
        extra = f" (transient_peak=0: {s['transient_peak_eq_0']}, >0: {s['transient_peak_gt_0']})" if c == "A1" else ""
        print(f"  {c:2s} {CLASS_NAME[c]:<20s} n={s['count']:3d}  median time_ratio={s['median_time_ratio']}"
              f"  transient_peak={s['median_transient_peak']}  k_space={s['median_k_space']}{extra}")
    print("A2 members:", ", ".join(stats["A2"]["members"]))
    print("D members: ", ", ".join(stats["D"]["members"]))
    print("E members: ", ", ".join(stats["E"]["members"]))

    write_classification(ok)
    write_deviation_table(ok, stats, series, fits, k1)
    draw_figure(ok, stats, series, fits)

    corpus_counts = {}
    for r in all_rows:
        corpus_counts.setdefault(r["corpus"], {}).setdefault(r["status"], 0)
        corpus_counts[r["corpus"]][r["status"]] += 1
    all_ok = [r for r in ok]
    summary = {
        "corpus_status_counts": corpus_counts,
        "n_classified": len(ok),
        "class_rule": CLASS_RULE,
        "class_stats": stats,
        "ccu_band_time_ratio": [CCU_LO, CCU_HI],
        "ccu_band_k_time": [round(math.log2(CCU_LO), 3), round(math.log2(CCU_HI), 3)],
        "corpus_wide": {
            "median_transient_peak": median([fnum(r["transient_peak"]) for r in all_ok]),
            "max_transient_peak": max(inum(r["transient_peak"]) for r in all_ok),
            "median_k_space": round(median([fnum(r["k_space"]) for r in all_ok]), 3),
            "n_k_space_ge_1": sum(1 for r in all_ok if fnum(r["k_space"]) >= 1),
            "n_time_ratio_eq_1": sum(1 for r in all_ok if fnum(r["time_ratio"]) == 1.0),
            "n_uncalls_gt_0": sum(1 for r in all_ok if inum(r["n_uncalls"]) > 0),
        },
        "scaling_fits": fits,
        "rfcl_k1": k1,
        "panel_a_x_clip": X_CLIP,
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", RES / "classification.csv", RES / "deviation_table.md", RES / "summary.json",
          RES / "fig_bennett_vs_measured.{png,pdf,svg}")


if __name__ == "__main__":
    main()
