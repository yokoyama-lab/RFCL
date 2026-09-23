#!/usr/bin/env python3
"""measure_corpus.py -- Bennett/pebbling measurements of real reversible programs.

Runs every Janus program of one or more corpora under the PyJanus interpreter
with an extended space profiler and writes

  results/corpus_measurements.csv    one row per source file
  results/scaling_measurements.csv   input-size scaling of a few programs
  results/timelines/<name>.jsonl     per-step (step, live_vars) timelines

Usage:
  PYJANUS=/path/to/PyJanus python3 measure_corpus.py CORPUS_ROOT [CORPUS_ROOT ...]
      [--out results] [--timeout 60] [--no-scaling] [--no-timelines]
      [--scratch DIR] [--summary-only]

PyJanus is imported from $PYJANUS (default /home/claude/bennett/pyjanus); no
file of PyJanus or of the corpora is ever modified.  Scaling variants are
written to --scratch (temp copies only).

Measured quantities (per run)
-----------------------------
total_steps      statement executions (PyJanus pebble.py semantics: every
                 statement, including the call/uncall statement itself, counts
                 one step *after* its body has executed)
fwd_steps        steps recorded while the uncall-nesting counter is 0
uncall_steps     total_steps - fwd_steps
live_vars        number of non-zero variables reachable from the *active*
                 frame stack (main frame + frames of calls in progress), with
                 aliases de-duplicated: a by-reference parameter is the same
                 storage as the caller's variable and is counted once.  This
                 deliberately differs from pebble.py's own _count_all_frames,
                 which keeps every frame ever entered and counts aliases in
                 each frame separately (see NOTE in MeasuringRuntime).
live_bits        sum of bit-widths of those non-zero values
live_init/final  live_vars at the "init" snapshot / at the last snapshot
transient_peak   max_live_vars - max(live_init, live_final)
k_space          transient_peak / log2(total_steps)
k_time           log2(total_steps / fwd_steps)

LIMITATION: fwd_steps only recognises reverse execution that goes through an
`uncall` statement.  Hand-written inline inverse code (e.g. a second loop that
undoes the first one statement by statement, or a `local ... delocal` cleanup
sequence) is indistinguishable from forward computation and is counted as
forward.  Programs with n_uncalls == 0 therefore always get time_ratio == 1.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import re
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict

PYJANUS = os.environ.get("PYJANUS", "/home/claude/bennett/pyjanus")
sys.path.insert(0, PYJANUS)

from jana_py.cli import parse_for_std  # noqa: E402
from jana_py.errors import JanaError  # noqa: E402
from jana_py.pebble import _ProfilingRuntime, _is_zero, _value_bits  # noqa: E402
from jana_py.preprocess import preprocess_text  # noqa: E402
from jana_py.runtime import (  # noqa: E402
    ArraySliceProxy,
    CellProxy,
    ConstantParamProxy,
    Frame,
    Runtime,
    StructFieldProxy,
)
from jana_py.validate import validate_program  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "results")
DEFAULT_SCRATCH = (
    "/tmp/claude-0/-home-claude/06ba6152-b045-59b6-8290-c44e49ea1de1/scratchpad/measureA"
)

# dialects tried, in order, per file extension (first success wins; on total
# failure the attempt that got furthest -- execution > validate > parse -- is
# reported, ties going to the earliest dialect in the list)
DIALECTS_BY_EXT = {
    ".j": ["jana2014"],                                              # reversible-algorithms
    ".j2": ["jana2014", "janus2026", "janus1982", "janus1982ext"],   # kato 2013 examples
    ".j1": ["janus1982", "janus1982ext", "jana2014", "janus2026"],   # watanabe 1982 transcriptions
}

METRIC_COLUMNS = [
    "status", "error",
    "total_steps", "fwd_steps", "uncall_steps", "time_ratio",
    "max_live_vars", "max_live_bits", "live_init", "live_final", "transient_peak",
    "log2_T", "k_space", "k_time",
    "call_depth_max", "local_var_max", "n_calls", "n_uncalls",
    "peak_step", "peak_fraction", "wall_seconds",
]
CORPUS_COLUMNS = ["corpus", "path", "family", "dialect_used"] + METRIC_COLUMNS + ["catalog_status"]
SCALING_COLUMNS = ["program", "size_param", "size_value"] + METRIC_COLUMNS

TIMELINE_PROGRAMS = {
    # relative path inside reversible-algorithms -> timeline file stem
    "numeric/gcd.j": "gcd",
    "bsort/bsort2.j": "bsort2",
    "dp/knapsack.j": "knapsack",
    "dp/lcs.j": "lcs",
    "apsp/apsp.j": "apsp",
    "others/code/factorial.j": "factorial",
    "msort/msort1.j": "msort1",
    "qsort/qsort2.j": "qsort2",
    # NOTE: no rod-cutting ("cutrod") DP exists in either corpus
    # (grep -riE 'cut.?rod|rod.?cut|\brod\b' finds nothing); the closest
    # 1-D "fold over all previous cells" DP is dp/lis.j, dumped instead.
    "dp/lis.j": "lis",
}


class TimeoutAbort(Exception):
    pass


# ---------------------------------------------------------------------------
# Corrected live-variable counting
# ---------------------------------------------------------------------------

def _unwrap(cell):
    while isinstance(cell, ConstantParamProxy):
        cell = cell._inner
    return cell


def count_distinct_live(frames: list[Frame]) -> tuple[int, int]:
    """Count non-zero variables over `frames`, each storage location once.

    A whole array (a plain Cell whose value is a list/dict) is one variable;
    an element/field/slice proxy into an array that is already counted as a
    whole is skipped; distinct proxies into the same array position are
    counted once."""
    plain, proxies = [], []
    for fr in frames:
        for cell in fr.vars.values():
            cell = _unwrap(cell)
            if isinstance(cell, (CellProxy, StructFieldProxy, ArraySliceProxy)):
                proxies.append(cell)
            else:
                plain.append(cell)
    seen: set = set()
    containers: set = set()
    live = bits = 0
    for c in plain:
        if id(c) in seen:
            continue
        seen.add(id(c))
        v = c.value
        if isinstance(v, (list, dict)):
            containers.add(id(v))
        if not _is_zero(v):
            live += 1
            bits += _value_bits(v)
    for p in proxies:
        parent = getattr(p, "array", None)
        if parent is None:
            parent = getattr(p, "struct_value", None)
        if id(parent) in containers:
            continue
        key = (id(parent), getattr(p, "index", None), getattr(p, "field_name", None),
               getattr(p, "offset", None))
        if key in seen:
            continue
        seen.add(key)
        v = p.value
        if not _is_zero(v):
            live += 1
            bits += _value_bits(v)
    return live, bits


# ---------------------------------------------------------------------------
# Measuring runtime
# ---------------------------------------------------------------------------

class MeasuringRuntime(_ProfilingRuntime):
    """pebble._ProfilingRuntime plus forward/uncall accounting.

    NOTE on the base class: _ProfilingRuntime._exec_stmt_impl appends every
    frame it ever sees to self._frames and never removes it, and
    _count_all_frames then counts by-reference parameters once per frame.
    For gcd.j that reports a peak of 6 live variables where only 3 distinct
    non-zero variables exist (a, b, log; aliased in the callee frame).  The
    `frame not in self._frames` test is also a linear scan with dataclass
    equality, i.e. O(steps x frames).  We therefore bypass that override and
    keep our own active-frame stack (root frame + frames of calls in
    progress, captured in _bind_args and popped when the call returns).
    """

    CHECK_EVERY = 256

    def __init__(self, program, timeout: float | None = None,
                 keep_timeline: bool = False, **kwargs):
        super().__init__(program, **kwargs)
        self._timeout = timeout
        self._t0 = time.monotonic()
        self._keep_timeline = keep_timeline
        self._call_stack: list[Frame] = []
        self._uncall_depth = 0
        self._fwd_steps = 0
        self._n_calls = 0
        self._n_uncalls = 0
        self._live_init: int | None = None
        self._live_final: int | None = None
        self._peak_step = 0
        self._timeline: list[tuple[int, int, int, int]] = []
        self._timed_out = False

    # -- frame tracking --------------------------------------------------
    def _bind_args(self, caller, name, args, pos):
        proc, frame, checks = super()._bind_args(caller, name, args, pos)
        self._call_stack.append(frame)
        return proc, frame, checks

    def _call_proc(self, caller, name, args, pos, record_stmt=True, record_nested=False):
        depth = len(self._call_stack)
        self._n_calls += 1
        try:
            super()._call_proc(caller, name, args, pos,
                               record_stmt=record_stmt, record_nested=record_nested)
        finally:
            del self._call_stack[depth:]

    def _uncall_proc(self, caller, name, args, pos, record_stmt=True, record_nested=False):
        depth = len(self._call_stack)
        self._n_uncalls += 1
        self._uncall_depth += 1
        try:
            super()._uncall_proc(caller, name, args, pos,
                                 record_stmt=record_stmt, record_nested=record_nested)
        finally:
            self._uncall_depth -= 1
            del self._call_stack[depth:]

    # -- statement hook (bypasses _ProfilingRuntime's frame bookkeeping) ---
    def _exec_stmt_impl(self, frame, stmt, allow_break, record_stmt, record_nested=False):
        Runtime._exec_stmt_impl(
            self, frame, stmt, allow_break=False,
            record_stmt=record_stmt, record_nested=record_nested,
        )
        self._step_count += 1
        if self._timeout is not None and self._step_count % self.CHECK_EVERY == 0:
            if time.monotonic() - self._t0 > self._timeout:
                self._timed_out = True
                raise TimeoutAbort()
        self._record_snapshot(line=stmt.pos.line, event=self._event_name(stmt))

    # -- snapshot ----------------------------------------------------------
    def _active_frames(self) -> list[Frame]:
        root = self._frames[:1]
        return root + self._call_stack

    def _record_snapshot(self, line: int, event: str) -> None:
        live_vars, live_bits = count_distinct_live(self._active_frames())
        step = self._step_count
        if event == "init":
            self._live_init = live_vars
        elif self._uncall_depth == 0:
            self._fwd_steps += 1
        if live_vars > self._peak_live_vars:
            self._peak_live_vars = live_vars
            self._peak_step = step
        if live_bits > self._peak_live_bits:
            self._peak_live_bits = live_bits
        self._live_final = live_vars
        if self._keep_timeline:
            self._timeline.append((step, live_vars, live_bits, 1 if self._uncall_depth else 0))

    # -- results -----------------------------------------------------------
    def metrics(self) -> dict:
        T = self._step_count
        fwd = self._fwd_steps
        li = self._live_init or 0
        lf = self._live_final or 0
        peak = self._peak_live_vars
        transient = peak - max(li, lf)
        log2T = math.log2(T) if T > 0 else 0.0
        ratio = (T / fwd) if fwd > 0 else None
        return {
            "total_steps": T,
            "fwd_steps": fwd,
            "uncall_steps": T - fwd,
            "time_ratio": round(ratio, 4) if ratio is not None else "",
            "max_live_vars": peak,
            "max_live_bits": self._peak_live_bits,
            "live_init": li,
            "live_final": lf,
            "transient_peak": transient,
            "log2_T": round(log2T, 3),
            "k_space": round(transient / log2T, 3) if log2T > 0 else "",
            "k_time": round(math.log2(ratio), 3) if ratio is not None else "",
            "call_depth_max": self._peak_call_depth,
            "local_var_max": self._peak_local_vars,
            "n_calls": self._n_calls,
            "n_uncalls": self._n_uncalls,
            "peak_step": self._peak_step,
            "peak_fraction": round(self._peak_step / T, 4) if T > 0 else "",
        }


# ---------------------------------------------------------------------------
# Running one program
# ---------------------------------------------------------------------------

def _first_line(exc: BaseException) -> str:
    if isinstance(exc, JanaError):
        msg = exc.message.strip().splitlines()[0] if exc.message.strip() else type(exc).__name__
        return f"line {exc.pos.line}: {msg}"
    if isinstance(exc, RecursionError):
        return "maximum recursion depth exceeded"
    s = str(exc).strip().splitlines()
    return (s[0] if s else type(exc).__name__)[:200]


def _run_in_big_stack(fn):
    """Run fn() in a thread with a large stack so deep Janus recursion works."""
    result: dict = {}

    def target():
        try:
            result["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            result["exc"] = exc

    old = threading.stack_size(512 * 1024 * 1024)
    try:
        t = threading.Thread(target=target)
        t.start()
        t.join()
    finally:
        threading.stack_size(old)
    if "exc" in result:
        raise result["exc"]
    return result.get("value")


def measure_source(filename: str, text: str, std: str, timeout: float,
                   keep_timeline: bool = False, stdin_text: str = "") -> dict:
    """Parse + validate + run `text` under dialect `std`. Never raises."""
    out = {"status": "error", "error": "", "phase": "preprocess", "timeline": None}
    t0 = time.monotonic()
    rt = None
    try:
        pre = preprocess_text(filename, text, include_dirs=[], std=std)
        out["phase"] = "parse"
        program = parse_for_std(std, filename, pre.text, pre.line_origins)
        out["phase"] = "validate"
        validate_program(program, require_main=False)
        if program.main is None or not program.main.stmts:
            out["status"] = "no_main"
            out["error"] = "no main procedure (or empty main)"
            return out
        out["phase"] = "execution"
        rt = MeasuringRuntime(program, timeout=timeout, keep_timeline=keep_timeline, std=std)

        def go():
            saved = (sys.stdin, sys.stdout, sys.stderr)
            sys.stdin = io.StringIO(stdin_text)
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            try:
                rt.run()
            finally:
                sys.stdin, sys.stdout, sys.stderr = saved

        _run_in_big_stack(go)
        out["status"] = "ok"
    except TimeoutAbort:
        out["status"] = "timeout"
        out["error"] = f"timeout after {timeout:g}s"
    except (SystemExit, KeyboardInterrupt) as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    except BaseException as exc:  # noqa: BLE001
        out["error"] = _first_line(exc)
    out["wall_seconds"] = round(time.monotonic() - t0, 3)
    if rt is not None and out["phase"] == "execution":
        out.update(rt.metrics())
        if keep_timeline:
            out["timeline"] = rt._timeline
    return out


def metric_row(res: dict) -> dict:
    row = {c: "" for c in METRIC_COLUMNS}
    row["status"] = res.get("status", "error")
    row["error"] = res.get("error", "")
    row["wall_seconds"] = res.get("wall_seconds", "")
    # partial metrics are kept for timeouts (the run is meaningful up to the
    # abort) but blanked for errors, so a crashed run never looks like data
    if "total_steps" in res and row["status"] in ("ok", "timeout"):
        for c in METRIC_COLUMNS:
            if c in res and c not in ("status", "error"):
                row[c] = res[c]
    return row


PHASE_RANK = {"preprocess": 0, "parse": 1, "validate": 2, "execution": 3}


def measure_file(path: str, dialects: list[str], timeout: float, keep_timeline: bool) -> tuple[str, dict]:
    """Try the dialects in order; return (dialect_used, result) for the best."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    attempts: list[tuple[str, dict]] = []
    variants = [(text, "")]
    if re.search(r"^\s*procedure\s+main_fwd\b", text, re.M):
        # Watanabe/J1 transcriptions carry main_fwd/main_bwd instead of main
        variants.append((re.sub(r"^(\s*procedure\s+)main_fwd\b", r"\1main", text, flags=re.M),
                         "(main_fwd->main)"))
    for src, tag in variants:
        for std in dialects:
            res = measure_source(path, src, std, timeout, keep_timeline)
            attempts.append((std + tag, res))
            if res["status"] == "ok":
                return attempts[-1]

    def score(a):
        st = a[1]["status"]
        return (st == "ok", st == "timeout", st == "no_main", PHASE_RANK[a[1]["phase"]])

    return max(attempts, key=score)


# ---------------------------------------------------------------------------
# Corpus walk
# ---------------------------------------------------------------------------

def corpus_name(root: str) -> str:
    return os.path.basename(os.path.normpath(root))


def family_of(corpus: str, rel: str) -> str:
    top = rel.split("/")[0]
    if corpus == "janus-examples":
        if top.startswith("kato"):
            return "kato"
        if top.startswith("watanabe"):
            return "watanabe"
    return top


def load_catalog(root: str) -> dict:
    p = os.path.join(root, "catalog.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as f:
            recs = json.load(f).get("records", [])
        return {r["file"]: r.get("status", "") for r in recs}
    except Exception:  # noqa: BLE001
        return {}


def iter_corpus_files(root: str) -> list[str]:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for fn in sorted(filenames):
            if fn.endswith((".j", ".j1", ".j2")):
                files.append(os.path.join(dirpath, fn))
    return files


def run_corpus(roots: list[str], out_dir: str, timeout: float, do_timelines: bool) -> list[dict]:
    rows: list[dict] = []
    tl_dir = os.path.join(out_dir, "timelines")
    os.makedirs(tl_dir, exist_ok=True)
    for root in roots:
        corpus = corpus_name(root)
        catalog = load_catalog(root)
        for path in iter_corpus_files(root):
            rel = os.path.relpath(path, root)
            ext = os.path.splitext(path)[1]
            dialects = DIALECTS_BY_EXT[ext]
            illegal = "examples_illegal" in rel.split("/")
            want_tl = do_timelines and corpus == "reversible-algorithms" and rel in TIMELINE_PROGRAMS
            print(f"[{corpus}] {rel} ...", end=" ", flush=True)
            std, res = measure_file(path, dialects, timeout, keep_timeline=want_tl)
            row = {"corpus": corpus, "path": rel, "family": family_of(corpus, rel),
                   "dialect_used": std, "catalog_status": catalog.get(rel, "")}
            row.update(metric_row(res))
            if illegal:
                row["status"] = "illegal_by_design"
                if not row["error"]:
                    row["error"] = "ran without error under " + std
            print(row["status"], row["error"] or "", f"T={row['total_steps']}", flush=True)
            rows.append(row)
            if want_tl and res.get("timeline") is not None:
                with open(os.path.join(tl_dir, TIMELINE_PROGRAMS[rel] + ".jsonl"), "w") as f:
                    for step, lv, lb, inu in res["timeline"]:
                        f.write(json.dumps({"step": step, "live_vars": lv, "live_bits": lb,
                                            "in_uncall": inu}, separators=(",", ":")) + "\n")
    rows.sort(key=lambda r: (r["corpus"], r["path"]))
    return rows


# ---------------------------------------------------------------------------
# Input-size scaling
# ---------------------------------------------------------------------------

def _rng(seed: int) -> random.Random:
    return random.Random(1000003 * seed + 17)


def _inits(name: str, values, base: int = 0) -> str:
    return "".join(f"  {name}[{i + base}] += {v}\n" for i, v in enumerate(values) if v)


def _identity(name: str, n: int) -> str:
    return "".join(f"  {name}[{i}] += {i}\n" for i in range(n))


MAIN_RE = r"procedure main\(\)[\s\S]*\Z"   # whole main() up to end of file


def _fib(k: int) -> int:
    a, b = 0, 1
    for _ in range(k):
        a, b = b, a + b
    return a


def _lcs_main(n: int) -> str:
    r = _rng(n)
    x = [r.randint(1, 4) for _ in range(n)]
    y = [r.randint(1, 4) for _ in range(n)]
    return (f"procedure main()\n  int x[{n + 1}]\n  int y[{n + 1}]\n  int m\n  int n\n"
            f"  int L[{(n + 1) * (n + 1)}]\n  m += {n}\n  n += {n}\n"
            + _inits("x", x, 1) + _inits("y", y, 1) + "  call lcs(x,m,y,n,L)\n")


def _knapsack_main(n: int) -> str:
    r = _rng(n)
    cap = 2 * n
    wt = [r.randint(1, n) for _ in range(n)]
    val = [r.randint(1, 3 * n) for _ in range(n)]
    return (f"procedure main()\n  int wt[{n + 1}]\n  int val[{n + 1}]\n  int n\n  int cap\n"
            f"  int K[{(n + 1) * (cap + 1)}]\n  n += {n}\n  cap += {cap}\n"
            + _inits("wt", wt, 1) + _inits("val", val, 1) + "  call knapsack(wt,val,n,cap,K)\n")


def _lis_main(n: int) -> str:
    r = _rng(n)
    a = [r.randint(1, 100) for _ in range(n)]
    return (f"procedure main()\n  int a[{n}]\n  int n\n  int L[{n}]\n  int res\n  n += {n}\n"
            + _inits("a", a) + "  call lis(a,n,L,res)\n")


def _bsort2_main(n: int) -> str:
    r = _rng(n)
    a = r.sample(range(1, 10 * n + 1), n)
    return (f"procedure main()\n  int a[{n}]\n  int sz\n  stack gb\n  int ord[{n}]\n  sz += {n}\n"
            + _inits("a", a) + "  call bsort(a,sz,gb)\n" + _identity("ord", n)
            + "  uncall bsort(ord,sz,gb)\n")


def _msort1_main(n: int) -> str:
    r = _rng(n)
    a = [r.randint(1, 20) for _ in range(n)]
    return (f"procedure main()\n  int a[{n}]\n  int ord[{n}]\n  stack gbg\n  int gbfg\n  int p\n  int r\n"
            + _inits("a", a) + f"  r += {n - 1}\n  call msort(a,p,r,gbg)\n" + _identity("ord", n)
            + "  uncall msort(ord,p,r,gbg)\n")


# Each recipe: corpus-relative path, size parameter name, list of sizes and a
# list of (regex, replacement(size)) substitutions applied to the file text.
SCALING_RECIPES = [
    {   # subtractive Euclid on consecutive Fibonacci pairs (F_{k+1}, F_k)
        "name": "numeric/gcd.j", "size_param": "fib_index_k",
        "sizes": [6, 9, 12, 15, 18, 21],
        "subs": [(r"a \+= 48", lambda k: f"a += {_fib(k + 1)}"),
                 (r"b \+= 36", lambda k: f"b += {_fib(k)}")],
    },
    {   # iterative factorial, n
        "name": "others/code/factorial.j", "size_param": "n",
        "sizes": [5, 10, 20, 40, 80, 160],
        "subs": [(r"n \+= 5", lambda n: f"n += {n}")],
    },
    {   # recursive factorial with call/uncall (Bennett-style), n
        "name": "others/examples/fact.j", "size_param": "n",
        "sizes": [2, 4, 6, 8, 10, 12],
        "subs": [(r"n \+= 5", lambda n: f"n += {n}")],
    },
    {   # LCS of two random strings of length n over {1..4}
        "name": "dp/lcs.j", "size_param": "n",
        "sizes": [3, 5, 8, 12, 16, 24],
        "subs": [(MAIN_RE, _lcs_main)],
    },
    {   # 0/1 knapsack, n items, capacity 2n
        "name": "dp/knapsack.j", "size_param": "n_items",
        "sizes": [2, 4, 6, 8, 12, 16],
        "subs": [(MAIN_RE, _knapsack_main)],
    },
    {   # longest increasing subsequence of n random values
        "name": "dp/lis.j", "size_param": "n",
        "sizes": [4, 8, 12, 16, 24, 32],
        "subs": [(MAIN_RE, _lis_main)],
    },
    {   # bubble sort with garbage stack + uncall to clean up, n keys
        "name": "bsort/bsort2.j", "size_param": "n",
        "sizes": [4, 6, 8, 12, 16, 24],
        "subs": [(MAIN_RE, _bsort2_main)],
    },
    {   # merge sort with garbage stack + uncall, n keys (powers of two)
        "name": "msort/msort1.j", "size_param": "n",
        "sizes": [2, 4, 8, 16, 32, 64],
        "subs": [(MAIN_RE, _msort1_main)],
    },
]


def run_scaling(ra_root: str, scratch: str, timeout: float) -> list[dict]:
    os.makedirs(scratch, exist_ok=True)
    rows: list[dict] = []
    for rec in SCALING_RECIPES:
        src_path = os.path.join(ra_root, rec["name"])
        with open(src_path, encoding="utf-8") as f:
            base_text = f.read()
        for size in rec["sizes"]:
            text = base_text
            for pattern, repl in rec["subs"]:
                text, n_sub = re.subn(pattern, lambda m, s=size: repl(s), text, count=1)
                if n_sub != 1:
                    raise RuntimeError(f"{rec['name']}: pattern {pattern!r} did not match")
            stem = rec["name"].replace("/", "__").replace(".j", "")
            tmp = os.path.join(scratch, f"{stem}__{rec['size_param']}{size}.j")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[scaling] {rec['name']} {rec['size_param']}={size} ...", end=" ", flush=True)
            res = measure_source(tmp, text, "jana2014", timeout)
            row = {"program": rec["name"], "size_param": rec["size_param"], "size_value": size}
            row.update(metric_row(res))
            print(row["status"], row["error"] or "", f"T={row['total_steps']}", flush=True)
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Output + summaries
# ---------------------------------------------------------------------------

def write_csv(path: str, rows: list[dict], columns: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "  (none)"
    widths = [max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols]
    lines = ["  " + "  ".join(c.ljust(w) for c, w in zip(cols, widths))]
    for r in rows:
        lines.append("  " + "  ".join(str(r.get(c, "")).ljust(w) for c, w in zip(cols, widths)))
    return "\n".join(lines)


SHOW = ["corpus", "path", "status", "total_steps", "fwd_steps", "time_ratio", "max_live_vars",
        "live_init", "live_final", "transient_peak", "log2_T", "k_space", "k_time",
        "n_calls", "n_uncalls", "peak_fraction"]


def summarize(rows: list[dict], scaling_rows: list[dict] | None, out_dir: str) -> None:
    print("\n" + "=" * 78)
    print("CSV header:", ",".join(CORPUS_COLUMNS))
    print("\n== status counts per corpus ==")
    per = defaultdict(Counter)
    for r in rows:
        per[r["corpus"]][r["status"]] += 1
    for c in sorted(per):
        print(f"  {c}: total={sum(per[c].values())} " +
              " ".join(f"{k}={v}" for k, v in sorted(per[c].items())))

    ok = [r for r in rows if r["status"] in ("ok", "timeout") and _num(r["total_steps"]) is not None]

    def top(key, n=10):
        return sorted((r for r in ok if _num(r[key]) is not None),
                      key=lambda r: -_num(r[key]))[:n]

    print("\n== 10 largest total_steps ==")
    print(_table(top("total_steps"), SHOW))
    print("\n== 10 largest transient_peak ==")
    print(_table(top("transient_peak"), SHOW))
    print("\n== 10 largest k_space ==")
    print(_table(top("k_space"), SHOW))
    print("\n== all rows of family dp ==")
    print(_table([r for r in rows if r["family"] == "dp"], SHOW))
    print("\n== rows with time_ratio >= 1.8 ==")
    print(_table([r for r in ok if (_num(r["time_ratio"]) or 0) >= 1.8], SHOW))
    print("\n== in-place family: n_uncalls == 0 and transient_peak == 0 (count per family) ==")
    inplace = [r for r in ok if _num(r["n_uncalls"]) == 0 and _num(r["transient_peak"]) == 0]
    cnt = Counter((r["corpus"], r["family"]) for r in inplace)
    for (c, fam), n in sorted(cnt.items()):
        print(f"  {c}/{fam}: {n}")
    print(f"  total: {len(inplace)} of {len(ok)} measured rows")
    if scaling_rows is not None:
        print("\n== scaling_measurements.csv ==")
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=SCALING_COLUMNS)
        w.writeheader()
        for r in scaling_rows:
            w.writerow({c: r.get(c, "") for c in SCALING_COLUMNS})
        print(buf.getvalue())
    print("\n== programs that did not run (status != ok) ==")
    print(_table([r for r in rows if r["status"] != "ok"],
                 ["corpus", "path", "dialect_used", "status", "catalog_status", "error"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", help="corpus root directories")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--scratch", default=DEFAULT_SCRATCH)
    ap.add_argument("--no-scaling", action="store_true")
    ap.add_argument("--no-timelines", action="store_true")
    ap.add_argument("--summary-only", action="store_true", help="re-print summaries from existing CSVs")
    args = ap.parse_args(argv)
    sys.setrecursionlimit(200000)

    corpus_csv = os.path.join(args.out, "corpus_measurements.csv")
    scaling_csv = os.path.join(args.out, "scaling_measurements.csv")
    if args.summary_only:
        rows = read_csv(corpus_csv)
        srows = read_csv(scaling_csv) if os.path.exists(scaling_csv) else None
        summarize(rows, srows, args.out)
        return 0
    if not args.roots:
        ap.error("at least one corpus root is required")

    rows = run_corpus(args.roots, args.out, args.timeout, not args.no_timelines)
    write_csv(corpus_csv, rows, CORPUS_COLUMNS)
    print(f"\nwrote {corpus_csv} ({len(rows)} rows)")

    srows = None
    ra = [r for r in args.roots if corpus_name(r) == "reversible-algorithms"]
    if not args.no_scaling and ra:
        srows = run_scaling(ra[0], args.scratch, args.timeout)
        write_csv(scaling_csv, srows, SCALING_COLUMNS)
        print(f"wrote {scaling_csv} ({len(srows)} rows)")
    summarize(rows, srows, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
