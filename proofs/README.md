# Machine-checked facts behind `experiments/pebbling/semantic_forward.py`

`lean/RFCLProofs/Pairing.lean` (Lean 4, no Mathlib). Build and audit:

    cd proofs/lean
    lake build                      # produces .lake/build/lib/lean/RFCLProofs/Pairing.olean
    lake env lean AxiomCheck.lean   # only propext, Quot.sound, Classical.choice

CI runs both (`.github/workflows/ci.yml`, job `lean`).

| theorem | statement | where it is used |
|---|---|---|
| `run_eq`, `run_zero` | the post-order accumulation `acc += steps − (acc − acc_at_entry)` over marked invocations equals the number of steps inside at least one marked invocation | `SemanticRuntime._finish`: `self._sem_unc += steps - (self._sem_unc - rec.unc0)` (and the same for `_sem_unc_loose`) |
| `full_to_partial` | an exact inverse of an invocation that changed something is a partial inverse | loose marking ⊇ strict marking |
| `cov_mono`, `strict_le_loose_le_total` | covered steps are monotone in the marking; strict ≤ loose ≤ total | the two columns of `results/semantic_measurements.csv` |
| `uncall_pairs` | if `uncall` computes a left inverse of `call`, an uncall starting at a call's exit values ends at its entry values | compute–copy–uncompute on untouched arguments is always strictly paired |
| `same_direction_pair_is_involutive` | two paired `call`s of `f` imply `f (f v) = v` | `n_paired_same_dir` (case `semantic_cases/twice.j`) |

The model and the code, side by side.

* An invocation is a tree node: `own` = its steps not inside a nested
  invocation, plus 1 for the call/uncall statement, which PyJanus counts after
  the body (`steps = step_count − step0 + 1` in `_finish`). Nested
  invocations are the children, in execution order. PyJanus finishes an
  invocation after all invocations nested in it, which is the post-order
  of `run`.
* The mark of a node is "the pairing test succeeded when it finished". The
  theorems hold for *any* marking, so they do not depend on how often the
  test succeeds, which invocation it picks as the partner, or on
  `MAX_DONE_PER_PROC`.
* `Rec` compares parameter values by position. That the storage keys are
  equal is assumed, i.e. it is the key test `earlier.keys == later.keys`.
  Whether `_storage_key` gives the right identity (aliasing, value
  arguments, constant literals) is *not* proved. It is checked by the cases
  in `experiments/pebbling/semantic_cases/`.

Checked: `lake build` succeeds and `Pairing.olean` is rewritten; the axiom
check lists only the standard axioms. Replacing the accumulation in `run` by
`a + (own + fsteps ks)` (count nested pairs again) makes `lake build` fail.
`semantic_cases/nested.j` exercises the same point in the Python code: 18
steps, 11 strict reverse steps (3 + 8), not 14.
