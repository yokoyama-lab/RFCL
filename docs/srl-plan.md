# SRL/RL Implementation Plan

Based on Moriyama 2009.

## Implemented

### SRL core
- integer variables
- fixed-size arrays
- assignments: `+=`, `-=`, `^=`, `swap`
- sequencing
- conditional: `if e1 then b1 else b2 fi e2`
- loop: `from e1 do b1 loop b2 until e2`
- program inversion
- static checker (undeclared variables, degenerate swap, self-referential assignment)
- temporary variable zeroing constraint at exit
- parser and pretty printer
- CLI (`run`, `invert`, `check`, `trace`, `dump-ast`)

### SRL extension
- `rif e1 b rfi e2` (conditional reverse / reverse conditional)

### RL core
- blocks, labels, `goto`, `from`, `fi ... from ... else`
- reverse jump: `rgoto` / `rfrom`
- fixed-size arrays
- program inversion
- static checker (unknown labels, unreachable blocks, undeclared variables, from/jump consistency)
- parser and pretty printer
- CLI (`rl-run`, `rl-invert`, `rl-check`, `rl-trace`, `rl-dump-ast`)

### Transforms
- `lower-srl-to-rl`: SRL `assign`, `swap`, `if`, `loop`, `rif` → RL
- `raise-rl-to-srl`: toolchain-generated RL for `assign`, `swap`, `if`, `loop`, `rif` → SRL
- `raise-rl-to-srl`: hand-written RL with multi-assignment blocks and compact loops → SRL
- `raise-rl-to-srl`: Bennett-style reverse-jump RL subgraphs (full and simple patterns) → SRL
- dominator tree computation for CFG analysis

### Visualization
- `rl-dot-graph`: RL program → GraphViz DOT → SVG (12 figures generated)

### Janus interoperability
- `srl-to-janus`: SRL → Janus (rif expanded to if/fi, hyphenated identifiers converted)
- `janus-to-srl`: Janus → SRL parser (simplified subset)
- Validated against Jana interpreter (fib_bennett, copy, countdown, branch_copy, bennett_rif)

### Theoretical property verification (214 tests)
- Inversion theorem: `run(invert(p), out(run(p, in))) == in` for SRL, RL, and lowered programs
- Translation equivalence: `run_srl(p) == run_rl(lower(p))` for all examples
- Round-trip identity: `run(raise(lower(p))) == run(p)` for all examples
- Inversion-lowering commutativity: `lower(invert(p)) ≡ rl_invert(lower(p))`
- Double inversion identity: `invert(invert(p)) == p` (syntactic)
- RL trace direction changes verified for rgoto/rfrom programs

### Self-interpreter
- `self_interp.srl`: reversible interpreter for straight-line programs (ADD/SUB/XOR)
- Demonstrates r-Turing completeness
- Inversion theorem verified for the self-interpreter itself

## Deferred

- raising arbitrary RL graphs with non-Bennett `rgoto`/`rfrom` patterns
- dynamic data structures (stacks) for r-Turing completeness (paper's Chapter 3)
- translation between PLA and RL (paper's concluding remarks)

## Semantic constraints (from paper)

- execution must be forward and backward deterministic
- conditionals use a test/assertion pair
- loops use an entry/exit condition pair
- temporary variables must return to zero at program exit
