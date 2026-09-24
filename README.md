# RFCL — Two Reversible Flowchart Languages: SRL & RL

A complete implementation of the structured reversible language (SRL) and
unstructured reversible language (RL) from Yokoyama, Axelsen, and Glück's
reversible flowchart framework, extended with the reverse jump (`rgoto`/`rfrom`)
and conditional reverse (`rif`/`rfi`) constructs described by Moriyama (2009).

## Features

### Core Languages
- **SRL**: structured reversible language with `if`/`fi`, `from`/`do`/`loop`/`until`, `rif`/`rfi`
- **RL**: unstructured reversible language with labeled blocks, `goto`, `rgoto`/`rfrom`
- Integer variables, fixed-size arrays, dynamic stacks (`push`/`pop`)
- Parser, pretty printer, interpreter, program inverter, static checker, execution trace

### Transforms
- **SRL → RL lowering**: compiles structured programs to labeled blocks
- **RL → SRL raising**: reconstructs structured programs from CFGs using dominator analysis
- **SRL ↔ Janus**: bidirectional conversion with the Janus reversible language
- **RL ↔ PLA**: PISA-style reversible assembly format
- **RL → PISA**: register-based assembly code generation

### Research Tools
- **Reversible debugger**: step forward and backward through execution
- **Landauer entropy analysis**: measure information erasure per step (always 0 for reversible programs)
- **Bennett transformation**: automatically convert computations to garbage-free reversible form
- **Time-space tradeoff analyzer**: measure and compare with the exact single-level counts (T_rev = 2T + |outputs| + 7)
- **Multi-level Bennett (pebble game)**: compile a step program and a pebbling schedule (linear, Bennett 1989 (k, m), or fewest moves under a pebble budget) into a clean SRL program (`pebble` subcommand, `--compact` for a loop-based program whose size does not grow with n; `experiments/multilevel/`)
- **Quantum circuit compiler**: compile SRL programs to Toffoli/CNOT circuits (OpenQASM 2.0)
- **Reverse-mode automatic differentiation**: tape-free gradient computation via reversibility
- **Program synthesis**: enumerate SRL programs from input/output examples
- **Partial evaluator**: specialize programs with known input values
- **Program slicer**: extract minimal sub-programs for target variables
- **Equivalence checker**: exhaustive input-space comparison of two programs
- **Extended type checker**: array bounds, stack balance verification
- **CFG visualization**: generate GraphViz DOT/SVG diagrams of RL control flow
- **Reversible self-interpreter**: SRL program that interprets SRL (r-Turing completeness)

## Quick Start

```sh
# Run an SRL program
python3 -m pyrev_fl.cli run examples/fib_bennett.srl 5

# Invert a program
python3 -m pyrev_fl.cli invert examples/copy.srl

# Lower SRL to RL, then raise back
python3 -m pyrev_fl.cli lower-srl-to-rl examples/countdown_clean.srl
python3 -m pyrev_fl.cli raise-rl-to-srl examples/hand_countdown.rl

# Visualize RL control flow
python3 -m pyrev_fl.cli rl-dot-graph examples/fib_bennett.rl | dot -Tsvg -o fib.svg

# Convert to Janus and run with Jana
python3 -m pyrev_fl.cli srl-to-janus examples/fib_bennett.srl

# Reversible debugger
python3 -m pyrev_fl.cli debug examples/copy.srl 5 --json

# Bennett time-space tradeoff
python3 -m pyrev_fl.cli tradeoff examples/countdown_clean.srl 5

# Landauer entropy analysis
python3 -m pyrev_fl.cli landauer examples/copy.srl 5

# Quantum circuit compilation
python3 -m pyrev_fl.cli to-circuit examples/copy.srl 5 --qasm

# Automatic differentiation
python3 -m pyrev_fl.cli autodiff examples/copy.srl 5 --output y
```

## Examples

| File | Description |
|------|-------------|
| `copy.srl` | Minimal SRL assignment (`y += x`) |
| `branch_copy.srl` | SRL conditional (`if`/`fi`) |
| `countdown_clean.srl` | SRL loop (`from`/`do`/`loop`/`until`) |
| `bennett_rif.srl` | Conditional reverse (`rif`/`rfi`) |
| `fib_bennett.srl` | Fibonacci via Bennett's trick with `rif` |
| `self_interp.srl` | Reversible self-interpreter |
| `stack_demo.srl` | Stack-based swap via `push`/`pop` |
| `copy.rl` | Minimal RL program |
| `fib_bennett.rl` | Fibonacci with reverse jump (`rgoto`/`rfrom`) |
| `rgoto_copy.rl` | Bennett-style copy via `rgoto`/`rfrom` |
| `hand_countdown.rl` | Hand-written RL loop (multi-assignment blocks) |

## Tests

```sh
python3 -m unittest discover -s tests_py -v
```

392 tests covering: core interpreters, inversion theorem, translation equivalence,
round-trip identity, Landauer entropy, Bennett tradeoff, quantum circuits,
automatic differentiation, program synthesis, and 50+ random programs.

## References

- T. Yokoyama, H. B. Axelsen, R. Glück. "Principles of a reversible programming language." Proc. CF'08, ACM, 2008.
- K. Moriyama. "Theoretical properties of reversible flowchart programming languages." DIKU, University of Copenhagen, 2009.
- C. H. Bennett. "Time/Space trade-offs for reversible computation." SIAM J. Comput., 18(4):766–776, 1989.
- M. P. Frank. "Reversibility for efficient computing." PhD thesis, MIT, 1999.
- T. Yokoyama, R. Glück. "A reversible programming language and its invertible self-interpreter." Proc. PEPM'07, ACM, 2007.

## License

MIT
