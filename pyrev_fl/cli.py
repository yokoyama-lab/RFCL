from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pyrev_fl.check import check_program
from pyrev_fl.dump_ast import dump_program as dump_srl_program
from pyrev_fl.interface import build_layout
from pyrev_fl.interpreter import run_program
from pyrev_fl.invert import invert_program
from pyrev_fl.parser import parse_program
from pyrev_fl.pretty import render_program
from pyrev_fl.rl_check import check_program as rl_check_program
from pyrev_fl.rl_dump_ast import dump_program as dump_rl_program
from pyrev_fl.rl_interpreter import run_program as rl_run_program
from pyrev_fl.rl_invert import invert_program as rl_invert_program
from pyrev_fl.rl_parser import parse_program as rl_parse_program
from pyrev_fl.rl_pretty import render_program as rl_render_program
from pyrev_fl.rl_trace import trace_program as rl_trace_program
from pyrev_fl.trace import trace_program
from pyrev_fl.janus_parser import parse_janus
from pyrev_fl.janus_pretty import render_janus
from pyrev_fl.equiv import check_equivalence
from pyrev_fl.pe import partial_eval
from pyrev_fl.typecheck import typecheck_program
from pyrev_fl.pisa import compile_to_pisa
from pyrev_fl.pla_parser import parse_pla
from pyrev_fl.pla_pretty import render_pla
from pyrev_fl.slicer import slice_program
from pyrev_fl.rl_dot import render_dot
from pyrev_fl.transform import lower_srl_to_rl, raise_rl_to_srl
from pyrev_fl.bennett import make_reversible_program
from pyrev_fl.autodiff import compute_jacobian, differentiate
from pyrev_fl.debugger import debug_trace
from pyrev_fl.landauer import analyze_landauer, format_metrics, metrics_to_dict
from pyrev_fl.quantum import compile_to_circuit
from pyrev_fl.tradeoff import analyze_tradeoff, format_tradeoff
from pyrev_fl.synthesize import SynthExample, synthesize


def main() -> int:
    parser = argparse.ArgumentParser(prog="pyrev_fl")
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("file")
    run_cmd.add_argument("inputs", nargs="*")
    run_cmd.add_argument("--json", action="store_true", dest="json_output")

    invert_cmd = sub.add_parser("invert")
    invert_cmd.add_argument("file")
    invert_cmd.add_argument("--json", action="store_true", dest="json_output")

    dump_cmd = sub.add_parser("dump-ast")
    dump_cmd.add_argument("file")

    check_cmd = sub.add_parser("check")
    check_cmd.add_argument("file")
    check_cmd.add_argument("--json", action="store_true", dest="json_output")

    trace_cmd = sub.add_parser("trace")
    trace_cmd.add_argument("file")
    trace_cmd.add_argument("inputs", nargs="*")
    trace_cmd.add_argument("--jsonl", action="store_true")

    lower_cmd = sub.add_parser("lower-srl-to-rl")
    lower_cmd.add_argument("file")
    lower_cmd.add_argument("--json", action="store_true", dest="json_output")

    rl_run_cmd = sub.add_parser("rl-run")
    rl_run_cmd.add_argument("file")
    rl_run_cmd.add_argument("inputs", nargs="*")
    rl_run_cmd.add_argument("--json", action="store_true", dest="json_output")

    rl_invert_cmd = sub.add_parser("rl-invert")
    rl_invert_cmd.add_argument("file")
    rl_invert_cmd.add_argument("--json", action="store_true", dest="json_output")

    rl_dump_cmd = sub.add_parser("rl-dump-ast")
    rl_dump_cmd.add_argument("file")

    rl_check_cmd = sub.add_parser("rl-check")
    rl_check_cmd.add_argument("file")
    rl_check_cmd.add_argument("--json", action="store_true", dest="json_output")

    rl_trace_cmd = sub.add_parser("rl-trace")
    rl_trace_cmd.add_argument("file")
    rl_trace_cmd.add_argument("inputs", nargs="*")
    rl_trace_cmd.add_argument("--jsonl", action="store_true")

    raise_cmd = sub.add_parser("raise-rl-to-srl")
    raise_cmd.add_argument("file")
    raise_cmd.add_argument("--json", action="store_true", dest="json_output")

    dot_cmd = sub.add_parser("rl-dot-graph")
    dot_cmd.add_argument("file")
    dot_cmd.add_argument("--output", default=None, metavar="FILE")

    srl2j_cmd = sub.add_parser("srl-to-janus")
    srl2j_cmd.add_argument("file")

    j2srl_cmd = sub.add_parser("janus-to-srl")
    j2srl_cmd.add_argument("file")

    rl2pla_cmd = sub.add_parser("rl-to-pla")
    rl2pla_cmd.add_argument("file")

    pla2rl_cmd = sub.add_parser("pla-to-rl")
    pla2rl_cmd.add_argument("file")

    slice_cmd = sub.add_parser("slice")
    slice_cmd.add_argument("file")
    slice_cmd.add_argument("--targets", required=True, help="comma-separated output variables")

    equiv_cmd = sub.add_parser("equiv")
    equiv_cmd.add_argument("file1")
    equiv_cmd.add_argument("file2")
    equiv_cmd.add_argument("--range", default="5", type=int, dest="input_range")

    pe_cmd = sub.add_parser("pe")
    pe_cmd.add_argument("file")
    pe_cmd.add_argument("--known", required=True, help="var=val,var=val,...")

    tc_cmd = sub.add_parser("typecheck")
    tc_cmd.add_argument("file")
    tc_cmd.add_argument("--json", action="store_true", dest="json_output")

    pisa_cmd = sub.add_parser("rl-to-pisa")
    pisa_cmd.add_argument("file")

    bennett_cmd = sub.add_parser("bennett")
    bennett_cmd.add_argument("file")
    bennett_cmd.add_argument("--output-vars", default=None, help="comma-separated output variables")

    landauer_cmd = sub.add_parser("landauer")
    landauer_cmd.add_argument("file")
    landauer_cmd.add_argument("inputs", nargs="*")
    landauer_cmd.add_argument("--json", action="store_true", dest="json_output")

    synth_cmd = sub.add_parser("synthesize")
    synth_cmd.add_argument("--inputs", required=True, help="space-separated input variable names")
    synth_cmd.add_argument("--outputs", required=True, help="space-separated output variable names")
    synth_cmd.add_argument("--examples", action="append", required=True,
                           help="input1,input2:output1,output2  (repeatable)")
    synth_cmd.add_argument("--max-stmts", type=int, default=3)
    synth_cmd.add_argument("--timeout", type=float, default=5.0)

    debug_cmd = sub.add_parser("debug")
    debug_cmd.add_argument("file")
    debug_cmd.add_argument("inputs", nargs="*")
    debug_cmd.add_argument("--json", action="store_true", dest="json_output")

    circuit_cmd = sub.add_parser("to-circuit")
    circuit_cmd.add_argument("file")
    circuit_cmd.add_argument("inputs", nargs="*")
    circuit_cmd.add_argument("--bits", type=int, default=8)
    circuit_cmd.add_argument("--qasm", action="store_true")

    ad_cmd = sub.add_parser("autodiff")
    ad_cmd.add_argument("file")
    ad_cmd.add_argument("inputs", nargs="*")
    ad_cmd.add_argument("--output", required=True, dest="output_var")
    ad_cmd.add_argument("--json", action="store_true", dest="json_output")
    ad_cmd.add_argument("--jacobian", action="store_true")

    tradeoff_cmd = sub.add_parser("tradeoff")
    tradeoff_cmd.add_argument("file")
    tradeoff_cmd.add_argument("inputs", nargs="*")
    tradeoff_cmd.add_argument("--json", action="store_true", dest="json_output")

    pebble_cmd = sub.add_parser("pebble", help="multi-level Bennett via the pebble game")
    pebble_cmd.add_argument("file", help="step program (X) (X Y) (temps)")
    pebble_cmd.add_argument("inputs", nargs="*")
    pebble_cmd.add_argument("--steps", type=int, required=True, help="number of steps n")
    group = pebble_cmd.add_mutually_exclusive_group()
    group.add_argument("--pebbles", type=int, help="fewest moves with at most this many pebbles")
    group.add_argument("--levels", type=int, help="Bennett 1989 levels k (needs --segments; n = m**k)")
    group.add_argument("--linear", action="store_true", help="keep the whole history (2n-1 moves)")
    group.add_argument("--mixed", help="Bennett 1989 with per-level segments, innermost first, e.g. 2,3,4 (n = product)")
    pebble_cmd.add_argument("--compact", action="store_true",
                            help="loop-based program (code size independent of n); needs --levels or --mixed")
    pebble_cmd.add_argument("--segments", type=int, default=2, help="Bennett 1989 segments m per level")
    pebble_cmd.add_argument("--emit", action="store_true", help="print the compiled SRL program")
    pebble_cmd.add_argument("--json", action="store_true", dest="json_output")

    args = parser.parse_args()
    try:
        if args.command == "run":
            return _run(Path(args.file), args.inputs, args.json_output)
        if args.command == "invert":
            return _invert(Path(args.file), args.json_output)
        if args.command == "dump-ast":
            return _dump_ast(Path(args.file))
        if args.command == "check":
            return _check(Path(args.file), args.json_output)
        if args.command == "trace":
            return _trace(Path(args.file), args.inputs, args.jsonl)
        if args.command == "lower-srl-to-rl":
            return _lower(Path(args.file), args.json_output)
        if args.command == "rl-run":
            return _rl_run(Path(args.file), args.inputs, args.json_output)
        if args.command == "rl-invert":
            return _rl_invert(Path(args.file), args.json_output)
        if args.command == "rl-dump-ast":
            return _rl_dump_ast(Path(args.file))
        if args.command == "rl-check":
            return _rl_check(Path(args.file), args.json_output)
        if args.command == "rl-trace":
            return _rl_trace(Path(args.file), args.inputs, args.jsonl)
        if args.command == "raise-rl-to-srl":
            return _raise(Path(args.file), args.json_output)
        if args.command == "rl-dot-graph":
            return _rl_dot_graph(Path(args.file), args.output)
        if args.command == "srl-to-janus":
            return _srl_to_janus(Path(args.file))
        if args.command == "janus-to-srl":
            return _janus_to_srl(Path(args.file))
        if args.command == "rl-to-pla":
            return _rl_to_pla(Path(args.file))
        if args.command == "pla-to-rl":
            return _pla_to_rl(Path(args.file))
        if args.command == "slice":
            return _slice(Path(args.file), args.targets)
        if args.command == "equiv":
            return _equiv(Path(args.file1), Path(args.file2), args.input_range)
        if args.command == "pe":
            return _pe(Path(args.file), args.known)
        if args.command == "typecheck":
            return _typecheck(Path(args.file), getattr(args, "json_output", False))
        if args.command == "rl-to-pisa":
            return _rl_to_pisa(Path(args.file))
        if args.command == "bennett":
            return _bennett(Path(args.file), args.output_vars)
        if args.command == "landauer":
            return _landauer(Path(args.file), args.inputs, args.json_output)
        if args.command == "synthesize":
            return _synthesize(args.inputs, args.outputs, args.examples,
                               args.max_stmts, args.timeout)
        if args.command == "debug":
            return _debug(Path(args.file), args.inputs, args.json_output)
        if args.command == "to-circuit":
            return _to_circuit(Path(args.file), args.inputs, args.bits, args.qasm)
        if args.command == "autodiff":
            return _autodiff(Path(args.file), args.inputs, args.output_var, args.json_output, args.jacobian)
        if args.command == "tradeoff":
            return _tradeoff(Path(args.file), args.inputs, args.json_output)
        if args.command == "pebble":
            return _pebble(args)
        parser.exit(2, "unknown command\n")
    except ValueError as exc:
        if getattr(args, "json_output", False):
            print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
            return 1
        parser.exit(1, f"{exc}\n")


def _run(path: Path, raw_inputs: list[str], json_output: bool = False) -> int:
    program = parse_program(path.read_text())
    layout = build_layout(program.inputs, program.outputs, program.temps)
    inputs = [int(value) for value in raw_inputs]
    store = run_program(program, inputs)
    if json_output:
        print(
            json.dumps(
                {
                    "kind": "srl_run",
                    "ok": True,
                    "outputs": {name: store[name] for name in layout.outputs},
                },
                sort_keys=True,
            )
        )
    else:
        for name in layout.outputs:
            print(f"{name}={store[name]}")
    return 0


def _invert(path: Path, json_output: bool = False) -> int:
    program = parse_program(path.read_text())
    rendered = render_program(invert_program(program))
    if json_output:
        print(json.dumps({"kind": "srl_invert", "ok": True, "program": rendered}, sort_keys=True))
    else:
        print(rendered, end="")
    return 0


def _dump_ast(path: Path) -> int:
    program = parse_program(path.read_text())
    print(json.dumps(dump_srl_program(program), indent=2, sort_keys=True))
    return 0


def _check(path: Path, json_output: bool = False) -> int:
    program = parse_program(path.read_text())
    check_program(program)
    if json_output:
        print(json.dumps({"kind": "srl_check", "ok": True}, sort_keys=True))
    else:
        print("ok")
    return 0


def _trace(path: Path, raw_inputs: list[str], jsonl: bool = False) -> int:
    program = parse_program(path.read_text())
    inputs = [int(value) for value in raw_inputs]
    events = trace_program(program, inputs)
    if jsonl:
        for event in events:
            print(json.dumps(event, sort_keys=True))
    else:
        print(json.dumps(events, indent=2, sort_keys=True))
    return 0


def _lower(path: Path, json_output: bool = False) -> int:
    program = parse_program(path.read_text())
    rendered = rl_render_program(lower_srl_to_rl(program))
    if json_output:
        print(json.dumps({"kind": "lower_srl_to_rl", "ok": True, "program": rendered}, sort_keys=True))
    else:
        print(rendered, end="")
    return 0


def _rl_run(path: Path, raw_inputs: list[str], json_output: bool = False) -> int:
    program = rl_parse_program(path.read_text())
    layout = build_layout(program.inputs, program.outputs, program.temps)
    inputs = [int(value) for value in raw_inputs]
    store = rl_run_program(program, inputs)
    if json_output:
        print(
            json.dumps(
                {
                    "kind": "rl_run",
                    "ok": True,
                    "outputs": {name: store[name] for name in layout.outputs},
                },
                sort_keys=True,
            )
        )
    else:
        for name in layout.outputs:
            print(f"{name}={store[name]}")
    return 0


def _rl_invert(path: Path, json_output: bool = False) -> int:
    program = rl_parse_program(path.read_text())
    rendered = rl_render_program(rl_invert_program(program))
    if json_output:
        print(json.dumps({"kind": "rl_invert", "ok": True, "program": rendered}, sort_keys=True))
    else:
        print(rendered, end="")
    return 0


def _rl_dump_ast(path: Path) -> int:
    program = rl_parse_program(path.read_text())
    print(json.dumps(dump_rl_program(program), indent=2, sort_keys=True))
    return 0


def _rl_check(path: Path, json_output: bool = False) -> int:
    program = rl_parse_program(path.read_text())
    rl_check_program(program)
    if json_output:
        print(json.dumps({"kind": "rl_check", "ok": True}, sort_keys=True))
    else:
        print("ok")
    return 0


def _rl_trace(path: Path, raw_inputs: list[str], jsonl: bool = False) -> int:
    program = rl_parse_program(path.read_text())
    inputs = [int(value) for value in raw_inputs]
    events = rl_trace_program(program, inputs)
    if jsonl:
        for event in events:
            print(json.dumps(event, sort_keys=True))
    else:
        print(json.dumps(events, indent=2, sort_keys=True))
    return 0


def _raise(path: Path, json_output: bool = False) -> int:
    program = rl_parse_program(path.read_text())
    rendered = render_program(raise_rl_to_srl(program))
    if json_output:
        print(json.dumps({"kind": "raise_rl_to_srl", "ok": True, "program": rendered}, sort_keys=True))
    else:
        print(rendered, end="")
    return 0


def _srl_to_janus(path: Path) -> int:
    program = parse_program(path.read_text())
    print(render_janus(program), end="")
    return 0


def _janus_to_srl(path: Path) -> int:
    program = parse_janus(path.read_text())
    print(render_program(program), end="")
    return 0


def _pe(path: Path, known_str: str) -> int:
    program = parse_program(path.read_text())
    known = {}
    for pair in known_str.split(","):
        name, val = pair.split("=", 1)
        known[name.strip()] = int(val.strip())
    residual = partial_eval(program, known)
    print(render_program(residual), end="")
    return 0


def _typecheck(path: Path, json_output: bool = False) -> int:
    program = parse_program(path.read_text())
    warnings = typecheck_program(program)
    if json_output:
        print(json.dumps([{"level": w.level, "message": w.message} for w in warnings], indent=2))
    elif warnings:
        for w in warnings:
            print(f"[{w.level}] {w.message}")
    else:
        print("ok")
    return 1 if any(w.level == "error" for w in warnings) else 0


def _slice(path: Path, targets_str: str) -> int:
    program = parse_program(path.read_text())
    targets = {t.strip() for t in targets_str.split(",")}
    sliced = slice_program(program, targets)
    print(render_program(sliced), end="")
    return 0


def _equiv(path1: Path, path2: Path, input_range: int) -> int:
    p1 = parse_program(path1.read_text())
    p2 = parse_program(path2.read_text())
    ok, msg = check_equivalence(p1, p2, input_range=range(-input_range, input_range + 1))
    print(msg)
    return 0 if ok else 1


def _rl_to_pisa(path: Path) -> int:
    program = rl_parse_program(path.read_text())
    print(compile_to_pisa(program), end="")
    return 0


def _rl_to_pla(path: Path) -> int:
    program = rl_parse_program(path.read_text())
    print(render_pla(program), end="")
    return 0


def _pla_to_rl(path: Path) -> int:
    program = parse_pla(path.read_text())
    print(rl_render_program(program), end="")
    return 0


def _bennett(path: Path, output_vars_str: str | None = None) -> int:
    program = parse_program(path.read_text())
    if output_vars_str is not None:
        output_vars = [v.strip() for v in output_vars_str.split(",")]
    else:
        output_vars = list(program.outputs)
    # Determine which variables are temps (not inputs or desired outputs)
    input_set = set(program.inputs)
    output_set = set(output_vars)
    # temps = everything declared that isn't an input or desired output
    all_declared = set(program.inputs) | set(program.outputs) | set(program.temps)
    temps = [v for v in (list(program.outputs) + list(program.temps))
             if v not in input_set and v not in output_set]
    # Remove duplicates while preserving order
    seen: set[str] = set()
    unique_temps: list[str] = []
    for t in temps:
        if t not in seen:
            seen.add(t)
            unique_temps.append(t)

    result = make_reversible_program(
        body_stmts=program.body.stmts,
        inputs=list(program.inputs),
        outputs=output_vars,
        temps=unique_temps,
    )
    print(render_program(result), end="")
    return 0


def _debug(path: Path, raw_inputs: list[str], json_output: bool = False) -> int:
    program = parse_program(path.read_text())
    inputs = [int(value) for value in raw_inputs]
    trace = debug_trace(program, inputs)
    if json_output:
        print(json.dumps({"kind": "srl_debug", "ok": True, "trace": trace}, sort_keys=True))
    else:
        for entry in trace:
            step = entry.get("step", "?")
            action = entry["action"]
            stmt = entry.get("stmt", "")
            store = entry["store"]
            print(f"  #{step} [{action}] {stmt}  {store}")
    return 0


def _to_circuit(path: Path, raw_inputs: list[str], bits: int, qasm: bool) -> int:
    program = parse_program(path.read_text())
    inputs = [int(value) for value in raw_inputs]
    circuit = compile_to_circuit(program, inputs, bits=bits)
    if qasm:
        print(circuit.to_qasm(), end="")
    else:
        print(circuit.to_text(), end="")
    return 0


def _autodiff(path: Path, raw_inputs: list[str], output_var: str, json_output: bool = False, jacobian: bool = False) -> int:
    program = parse_program(path.read_text())
    inputs = [int(value) for value in raw_inputs]
    result = differentiate(program, inputs, output_var)
    if json_output:
        data = {"kind": "autodiff", "ok": True, "output_values": result.output_values,
                "gradients": result.gradients, "tape_size": result.tape_size}
        if jacobian:
            data["jacobian"] = compute_jacobian(program, inputs)
        print(json.dumps(data, sort_keys=True))
    else:
        print(f"Output values: {result.output_values}")
        print(f"Tape size: {result.tape_size} (reversible: no tape needed)")
        print(f"\nGradients of {output_var}:")
        for inp, grad in sorted(result.gradients.get(output_var, {}).items()):
            print(f"  d{output_var}/d{inp} = {grad}")
    return 0


def _pebble(args) -> int:
    from pyrev_fl.pebble import (
        analyze_pebbling, bennett_mixed_schedule, compile_pebbling, linear_schedule,
        optimal_schedule, step_spec,
    )
    from pyrev_fl.pretty import render_program

    spec = step_spec(parse_program(Path(args.file).read_text()))
    n = args.steps
    ms = None
    if args.levels is not None:
        ms = [args.segments] * args.levels
    elif args.mixed is not None:
        ms = [int(v) for v in args.mixed.split(",")]
    if ms is not None:
        if math.prod(ms) != n:
            raise ValueError(f"--steps must equal the product of the segments = {math.prod(ms)}")
        moves, label = bennett_mixed_schedule(ms), f"bennett(m={','.join(map(str, ms))})"
    elif args.compact:
        raise ValueError("--compact needs --levels or --mixed")
    elif args.pebbles is not None:
        moves, label = optimal_schedule(n, args.pebbles), f"optimal(s={args.pebbles})"
    else:
        moves, label = linear_schedule(n), "linear"
    if args.compact:
        return _pebble_compact(spec, ms, label, args)
    if args.emit:
        print(render_program(compile_pebbling(spec, moves, n).program), end="")
        return 0
    inputs = [int(v) for v in args.inputs]
    r = analyze_pebbling(spec, moves, n, inputs)
    if args.json_output:
        print(json.dumps({
            "kind": "pebble", "ok": True, "schedule": label, "n": n,
            "moves": r.stats.moves, "pebbles": r.stats.pebbles,
            "time": r.time_steps, "baseline_time": r.baseline_time,
            "peak_space": r.peak_space, "declared_space": r.declared_space,
            "output": r.output,
        }))
        return 0
    print(f"schedule        {label}, n = {n}")
    print(f"moves           {r.stats.moves}  ({r.stats.moves / n:.2f} per step)")
    print(f"pebbles         {r.stats.pebbles}  (checkpoint registers)")
    print(f"time            {r.time_steps}  (forward-only run: {r.baseline_time}, ratio {r.time_steps / max(r.baseline_time, 1):.2f})")
    print(f"space           peak {r.peak_space} non-zero of {r.declared_space} declared")
    print("output          " + " ".join(f"{k}={v}" for k, v in r.output.items()))
    return 0


def _pebble_compact(spec, ms, label, args) -> int:
    from pyrev_fl.interpreter import run_program
    from pyrev_fl.pebble_compact import compile_bennett_compact, statement_count
    from pyrev_fl.pretty import render_program
    from pyrev_fl.tradeoff import _measure

    c = compile_bennett_compact(spec, ms)
    if args.emit:
        print(render_program(c.program), end="")
        return 0
    inputs = [int(v) for v in args.inputs]
    metrics = _measure(c.program, inputs)
    store = run_program(c.program, inputs)
    output = {x: store[o] for x, o in zip(spec.state, c.output_vars)}
    size = statement_count(c.program.body)
    if args.json_output:
        print(json.dumps({
            "kind": "pebble", "ok": True, "schedule": label + " compact", "n": c.n,
            "pebbles": c.registers, "time": metrics.time_steps,
            "peak_space": metrics.peak_space, "statements": size, "output": output,
        }))
        return 0
    print(f"schedule        {label}, compact, n = {c.n}")
    print(f"pebbles         {c.registers}  (checkpoint cells)")
    print(f"time            {metrics.time_steps}")
    print(f"space           peak {metrics.peak_space} non-zero")
    print(f"program size    {size} statements")
    print("output          " + " ".join(f"{k}={v}" for k, v in output.items()))
    return 0


def _tradeoff(path: Path, raw_inputs: list[str], json_output: bool = False) -> int:
    program = parse_program(path.read_text())
    inputs = [int(value) for value in raw_inputs]
    result = analyze_tradeoff(program, inputs)
    if json_output:
        print(json.dumps({
            "kind": "tradeoff", "ok": True,
            "original": {"time": result.original.time_steps, "space": result.original.peak_space},
            "bennett": {"time": result.bennett.time_steps, "space": result.bennett.peak_space},
            "time_ratio": result.time_ratio,
            "space_ratio": result.space_ratio,
            "theoretical_time_ratio": result.theoretical_time_ratio,
            "theoretical_space_bound": result.theoretical_space_bound,
        }, sort_keys=True))
    else:
        print(format_tradeoff(result))
    return 0


def _landauer(path: Path, raw_inputs: list[str], json_output: bool = False) -> int:
    program = parse_program(path.read_text())
    inputs = [int(value) for value in raw_inputs]
    metrics = analyze_landauer(program, inputs)
    if json_output:
        print(json.dumps({"kind": "landauer", "ok": True, **metrics_to_dict(metrics)}, sort_keys=True))
    else:
        print(format_metrics(metrics))
    return 0


def _synthesize(
    inputs_str: str,
    outputs_str: str,
    raw_examples: list[str],
    max_stmts: int,
    timeout: float,
) -> int:
    input_names = inputs_str.split()
    output_names = outputs_str.split()
    examples: list[SynthExample] = []
    for raw in raw_examples:
        if ":" not in raw:
            raise ValueError(f"bad example format (expected 'in1,in2:out1,out2'): {raw}")
        in_part, out_part = raw.split(":", 1)
        in_vals = [int(v.strip()) for v in in_part.split(",")]
        out_vals = [int(v.strip()) for v in out_part.split(",")]
        if len(out_vals) != len(output_names):
            raise ValueError(
                f"output count mismatch: expected {len(output_names)}, got {len(out_vals)}"
            )
        expected = dict(zip(output_names, out_vals))
        examples.append(SynthExample(inputs=in_vals, expected_outputs=expected))

    result = synthesize(
        input_names=input_names,
        output_names=output_names,
        examples=examples,
        max_stmts=max_stmts,
        timeout=timeout,
    )

    if result.program is not None:
        print(render_program(result.program), end="")
        print(f"\n# Tested {result.programs_tested} programs in {result.time_seconds:.3f}s",
              file=__import__("sys").stderr)
        return 0
    else:
        print(f"No program found (tested {result.programs_tested} in {result.time_seconds:.3f}s)",
              file=__import__("sys").stderr)
        return 1


def _rl_dot_graph(path: Path, output: str | None = None) -> int:
    program = rl_parse_program(path.read_text())
    dot = render_dot(program)
    if output:
        Path(output).write_text(dot)
    else:
        print(dot, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
