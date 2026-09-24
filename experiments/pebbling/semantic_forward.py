#!/usr/bin/env python3
"""semantic_forward.py -- "forward work" defined by effect, not by syntax.

measure_corpus.py counts a step as reverse work iff it runs inside an
`uncall`.  That misreads programs whose main computation *is* an uncall
(class E in REPORT.md: `uncall unrank(...)` computes a rank), and it would
equally misread `uncall g; copy; call g`, where the `call` is the cleanup.

Semantic definition used here.  A procedure invocation I (call or uncall)
*uncomputes* an earlier completed invocation J iff

  * I and J invoke the same procedure,
  * their parameters are bound to the same storage (same variables / array
    cells; value arguments are matched by position),
  * the parameters' values when I starts equal those when J ended, and the
    values when I ends equal those when J started, and
  * J changed something (its entry and exit values differ).

I.e. I is an inverse of J on the same data, whichever direction it runs in.
Each J is undone at most once, and the most recent candidate is taken.  The
steps of I (its body plus the call statement itself) are *reverse work*;
all other steps are *forward work*, including an uncall that nobody pairs
with.  Steps already counted inside a nested uncomputing invocation are not
counted twice.  This is the *strict* count.

A *loose* count also treats I as reverse work when it only partly undoes J:
some parameter storage that J changed is shared with I and I takes it from
J's exit value back to J's entry value, while other parameters differ.  This
is the reversible-sorting idiom `call sort(a, g); uncall sort(ord, g)`: the
uncall clears the garbage g *and* computes a new output in ord.  Strict and
loose bracket the reverse work an invocation-level analysis can attribute.

What it still cannot see: uncomputation written inline as statements (no
invocation to pair), the same limitation as measure_corpus.py.

Writes results/semantic_measurements.csv (one row per runnable program of
the reversible-algorithms corpus) and results/semantic_classes.md.

Usage (same PyJanus lookup as measure_corpus.py):
  PYJANUS=/path/to/PyJanus python3 semantic_forward.py /path/to/reversible-algorithms
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import measure_corpus as mc  # noqa: E402  (locates and imports PyJanus)
from jana_py.runtime import ArraySliceProxy, CellProxy, StructFieldProxy  # noqa: E402

MAX_DONE_PER_PROC = 4096


def _storage_key(pname: str, cell, value_cells: set[int]):
    """Identity of the storage a parameter is bound to."""
    c = mc._unwrap(cell)
    if id(cell) in value_cells or id(c) in value_cells:
        return ("value", pname)
    if isinstance(c, (CellProxy, StructFieldProxy, ArraySliceProxy)):
        parent = getattr(c, "array", None)
        if parent is None:
            parent = getattr(c, "struct_value", None)
        return ("proxy", id(parent), getattr(c, "index", None),
                getattr(c, "field_name", None), getattr(c, "offset", None))
    return ("cell", id(c))


class _Inv:
    __slots__ = ("proc", "direction", "keys", "cells", "keep", "pre", "post",
                 "step0", "unc0", "unc0_loose")


class SemanticRuntime(mc.MeasuringRuntime):
    """MeasuringRuntime plus effect-based pairing of invocations."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._pending_dir = 0
        self._inv_stack: list[_Inv] = []
        self._done: dict[str, list[_Inv]] = defaultdict(list)
        self._sem_unc = 0
        self._sem_unc_loose = 0
        self._n_partial = 0
        self._n_paired = 0
        self._n_paired_same_dir = 0
        self._n_paired_calls = 0      # a `call` that undoes an earlier uncall
        self._n_unpaired_uncalls = 0

    def _bind_args(self, caller, name, args, pos):
        proc, frame, checks = super()._bind_args(caller, name, args, pos)
        value_cells = {id(cell) for _, cell in checks}
        rec = _Inv()
        rec.proc = name
        rec.direction = self._pending_dir
        items = list(frame.vars.items())
        rec.keys = tuple(_storage_key(p, c, value_cells) for p, c in items)
        rec.cells = [mc._unwrap(c) for _, c in items]
        # keep the parent containers alive so their ids cannot be reused
        rec.keep = [getattr(c, "array", None) or getattr(c, "struct_value", None) or c
                    for c in rec.cells]
        rec.pre = [copy.deepcopy(c.value) for c in rec.cells]
        rec.step0 = self._step_count
        rec.unc0 = self._sem_unc
        rec.unc0_loose = self._sem_unc_loose
        self._inv_stack.append(rec)
        return proc, frame, checks

    def _invoke(self, direction, base, caller, name, args, pos, record_stmt, record_nested):
        self._pending_dir = direction
        depth = len(self._inv_stack)
        try:
            base(caller, name, args, pos, record_stmt=record_stmt, record_nested=record_nested)
        except BaseException:
            del self._inv_stack[depth:]
            raise
        rec = self._inv_stack.pop()
        del self._inv_stack[depth:]
        self._finish(rec)

    def _call_proc(self, caller, name, args, pos, record_stmt=True, record_nested=False):
        self._invoke(+1, super()._call_proc, caller, name, args, pos, record_stmt, record_nested)

    def _uncall_proc(self, caller, name, args, pos, record_stmt=True, record_nested=False):
        self._invoke(-1, super()._uncall_proc, caller, name, args, pos, record_stmt, record_nested)

    @staticmethod
    def _full_inverse(earlier: _Inv, later: _Inv) -> bool:
        return earlier.keys == later.keys and earlier.post == later.pre and earlier.pre == later.post

    @staticmethod
    def _partial_inverse(earlier: _Inv, later: _Inv) -> bool:
        where = {k: i for i, k in enumerate(earlier.keys) if k[0] != "value"}
        for ii, k in enumerate(later.keys):
            jj = where.get(k)
            if (jj is not None and earlier.pre[jj] != earlier.post[jj]
                    and later.pre[ii] == earlier.post[jj] and later.post[ii] == earlier.pre[jj]):
                return True
        return False

    def _find(self, done: list[_Inv], rec: _Inv, test) -> int | None:
        for j in range(len(done) - 1, -1, -1):
            if test(done[j], rec):
                return j
        return None

    def _finish(self, rec: _Inv) -> None:
        rec.post = [copy.deepcopy(c.value) for c in rec.cells]
        done = self._done[rec.proc]
        steps = self._step_count - rec.step0 + 1  # + the call/uncall statement itself
        full = self._find(done, rec, self._full_inverse)
        part = None if full is not None else self._find(done, rec, self._partial_inverse)
        if full is None and part is None:
            if rec.direction < 0:
                self._n_unpaired_uncalls += 1
            if rec.pre != rec.post:
                done.append(rec)
                if len(done) > MAX_DONE_PER_PROC:
                    del done[0]
            return
        J = done.pop(full if full is not None else part)
        self._sem_unc_loose += steps - (self._sem_unc_loose - rec.unc0_loose)
        if full is None:
            self._n_partial += 1
            return
        self._n_paired += 1
        if J.direction == rec.direction:
            self._n_paired_same_dir += 1
        if rec.direction > 0:
            self._n_paired_calls += 1
        self._sem_unc += steps - (self._sem_unc - rec.unc0)

    def metrics(self) -> dict:
        m = super().metrics()
        T = self._step_count
        fwd = T - self._sem_unc
        ratio = T / fwd if fwd > 0 else None
        fwd_l = T - self._sem_unc_loose
        ratio_l = T / fwd_l if fwd_l > 0 else None
        m.update({
            "sem_uncompute_steps_loose": self._sem_unc_loose,
            "time_ratio_sem_loose": round(ratio_l, 4) if ratio_l is not None else "",
            "n_partial": self._n_partial,
            "sem_uncompute_steps": self._sem_unc,
            "sem_fwd_steps": fwd,
            "time_ratio_sem": round(ratio, 4) if ratio is not None else "",
            "k_time_sem": round(math.log2(ratio), 3) if ratio is not None else "",
            "n_paired": self._n_paired,
            "n_paired_same_dir": self._n_paired_same_dir,
            "n_paired_calls": self._n_paired_calls,
            "n_unpaired_uncalls": self._n_unpaired_uncalls,
        })
        return m


SEM_COLUMNS = ["sem_uncompute_steps_loose", "time_ratio_sem_loose", "n_partial",
               "sem_uncompute_steps", "sem_fwd_steps", "time_ratio_sem", "k_time_sem",
               "n_paired", "n_paired_same_dir", "n_paired_calls", "n_unpaired_uncalls"]

CCU_LO, CCU_HI = 1.6, 2.05
DP_FAMILIES = {"dp", "floyd_warshall", "bellman_ford", "apsp"}


def classify(n_unc: int, tr: float, fwd: int, tot: int, family: str) -> str:
    """analyze.py's rules, applied to either accounting."""
    if n_unc > 0 and fwd < 0.10 * tot:
        return "E"
    if tr > CCU_HI:
        return "D"
    if CCU_LO <= tr <= CCU_HI:
        return "C"
    if n_unc > 0 and tr < CCU_LO:
        return "B"
    if n_unc == 0 and family in DP_FAMILIES:
        return "A2"
    return "A1"


def run(root: str, timeout: float) -> list[dict]:
    mc.MeasuringRuntime = SemanticRuntime  # measure_source instantiates this name
    rows = []
    for path in mc.iter_corpus_files(root):
        rel = os.path.relpath(path, root)
        std, res = mc.measure_file(path, mc.DIALECTS_BY_EXT[os.path.splitext(path)[1]],
                                   timeout, keep_timeline=False)
        if res.get("status") != "ok":
            continue
        row = {"path": rel, "family": mc.family_of("reversible-algorithms", rel),
               "dialect_used": std}
        for c in ("total_steps", "fwd_steps", "time_ratio", "n_calls", "n_uncalls",
                  "call_depth_max") + tuple(SEM_COLUMNS):
            row[c] = res.get(c, "")
        T = int(row["total_steps"])
        row["class_syntactic"] = classify(int(row["n_uncalls"]), float(row["time_ratio"] or 1),
                                          int(row["fwd_steps"]), T, row["family"])
        row["class_semantic"] = classify(int(row["n_paired"]), float(row["time_ratio_sem"] or 1),
                                         int(row["sem_fwd_steps"]), T, row["family"])
        row["class_semantic_loose"] = classify(
            int(row["n_paired"]) + int(row["n_partial"]), float(row["time_ratio_sem_loose"] or 1),
            T - int(row["sem_uncompute_steps_loose"]), T, row["family"])
        rows.append(row)
        print(f"{rel:60s} {row['class_syntactic']:>2} -> {row['class_semantic']:<2} "
              f"/ {row['class_semantic_loose']:<2} ratio {row['time_ratio']} -> "
              f"{row['time_ratio_sem']} / {row['time_ratio_sem_loose']}", flush=True)
    return rows


def write_outputs(rows: list[dict], out_dir: str) -> None:
    with open(os.path.join(out_dir, "semantic_measurements.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    classes = ["A1", "A2", "B", "C", "D", "E"]
    lines = ["# Syntactic vs. semantic forward work",
             "",
             f"{len(rows)} runnable programs of reversible-algorithms (commit pinned in "
             "REPORT.md). *Syntactic*: inside `uncall` = reverse (measure_corpus.py). "
             "*Strict*: reverse = invocations that exactly undo an earlier one on the same "
             "storage. *Loose*: also invocations that restore some storage an earlier one "
             "changed (garbage clearing). Same class rules as analyze.py.",
             "",
             "## Class sizes", "",
             "| class | syntactic | strict | loose |", "|---|---|---|---|"]
    for c in classes:
        lines.append(f"| {c} | " + " | ".join(
            str(sum(r[k] == c for r in rows))
            for k in ("class_syntactic", "class_semantic", "class_semantic_loose")) + " |")
    for key, title in (("class_semantic", "strict"), ("class_semantic_loose", "loose")):
        trans = Counter((r["class_syntactic"], r[key]) for r in rows)
        lines += ["", f"## Transitions syntactic -> {title}", "",
                  f"| syntactic \\ {title} | " + " | ".join(classes) + " | total |",
                  "|---|" + "---|" * (len(classes) + 1)]
        for a in classes:
            cnt = [trans[(a, b)] for b in classes]
            if sum(cnt):
                lines.append(f"| {a} | " + " | ".join(str(c) if c else "" for c in cnt)
                             + f" | {sum(cnt)} |")
    lines += ["", "## Programs whose ratio changes by more than 1 % under either count", "",
              "| path | class syn / strict / loose | time_ratio syn | strict | loose | "
              "uncalls | full pairs | partial | unpaired uncalls | pairing calls |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["path"]):
        a = float(r["time_ratio"] or 1)
        b = float(r["time_ratio_sem"] or 1)
        c = float(r["time_ratio_sem_loose"] or 1)
        if abs(a - b) > 0.01 * a or abs(a - c) > 0.01 * a:
            lines.append(f"| {r['path']} | {r['class_syntactic']} / {r['class_semantic']} / "
                         f"{r['class_semantic_loose']} | {a:g} | {b:g} | {c:g} | "
                         f"{r['n_uncalls']} | {r['n_paired']} | {r['n_partial']} | "
                         f"{r['n_unpaired_uncalls']} | {r['n_paired_calls']} |")
    with open(os.path.join(out_dir, "semantic_classes.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", help="reversible-algorithms checkout")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--timeout", type=float, default=60)
    args = ap.parse_args(argv)
    rows = run(args.root, args.timeout)
    write_outputs(rows, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
