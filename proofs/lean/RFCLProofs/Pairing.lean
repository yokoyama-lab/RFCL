/-!
# Soundness of the invocation pairing in `experiments/pebbling/semantic_forward.py`

Three facts the analysis relies on, proved without Mathlib.

1. **Accounting** (`run_eq`).  The runtime walks the invocation tree in
   post-order (an invocation finishes after all invocations nested in it) and,
   for each paired ("marked") invocation, adds its step count minus what was
   already added while it ran (`self._sem_unc += steps - (self._sem_unc -
   rec.unc0)`).  The total equals `cov`: the number of steps lying inside at
   least one marked invocation.  Nothing is counted twice, nothing is lost.
2. **Strict ≤ loose** (`full_to_partial`, `cov_mono`).  An exact inverse of an
   invocation that changed something is also a partial inverse, so the loose
   marking contains the strict one, and `cov` is monotone in the marking.
3. **Compute/uncompute is always paired** (`uncall_pairs`).  If a procedure's
   meaning `f` has a left inverse `g` (what `uncall` runs), an `uncall` that
   starts where a `call` ended ends where the call started: the strict test
   holds.  A same-direction pair (`call p; ...; call p`) can only occur at a
   point where `f` undoes itself (`same_direction_pair_is_involutive`).

The tree model: an invocation has `own` steps not inside any nested
invocation (including the call statement itself), a mark, and its nested
invocations in order.  How `semantic_forward.py` maps onto it is in
`proofs/README.md`.
-/

namespace RFCL.Pairing

mutual
inductive Tree where
  | node (own : Nat) (marked : Bool) (kids : Forest)
inductive Forest where
  | nil
  | cons (t : Tree) (rest : Forest)
end

mutual
/-- All steps of an invocation. -/
def steps : Tree → Nat
  | .node own _ ks => own + fsteps ks
def fsteps : Forest → Nat
  | .nil => 0
  | .cons t r => steps t + fsteps r
end

mutual
/-- Specification: steps inside at least one marked invocation. -/
def cov : Tree → Nat
  | .node own m ks => if m then own + fsteps ks else fcov ks
def fcov : Forest → Nat
  | .nil => 0
  | .cons t r => cov t + fcov r
end

mutual
/-- The implementation: post-order accumulation, as in `SemanticRuntime._finish`. -/
def run (acc : Nat) : Tree → Nat
  | .node own m ks =>
    let a := frun acc ks
    if m then a + ((own + fsteps ks) - (a - acc)) else a
def frun (acc : Nat) : Forest → Nat
  | .nil => acc
  | .cons t r => frun (run acc t) r
end

mutual
theorem cov_le : ∀ t : Tree, cov t ≤ steps t
  | .node own m ks => by
    have h := fcov_le ks
    cases m <;> simp [cov, steps] <;> omega
theorem fcov_le : ∀ f : Forest, fcov f ≤ fsteps f
  | .nil => by simp [fcov, fsteps]
  | .cons t r => by
    have h1 := cov_le t
    have h2 := fcov_le r
    simp only [fcov, fsteps]; omega
end

mutual
/-- The accumulation computes exactly the covered steps. -/
theorem run_eq : ∀ (acc : Nat) (t : Tree), run acc t = acc + cov t
  | acc, .node own m ks => by
    have h := frun_eq acc ks
    have hle := fcov_le ks
    cases m <;> simp [run, cov, h] <;> omega
theorem frun_eq : ∀ (acc : Nat) (f : Forest), frun acc f = acc + fcov f
  | acc, .nil => by simp [frun, fcov]
  | acc, .cons t r => by
    have h1 := run_eq acc t
    have h2 := frun_eq (run acc t) r
    rw [frun, h2, h1, fcov]; omega
end

/-- Starting from zero, the counter ends at the covered steps. -/
theorem run_zero (t : Tree) : run 0 t = cov t := by
  simp [run_eq]

/-! ## Monotonicity in the marking -/

mutual
/-- `Below t u`: same shape and step counts, and every node marked in `t` is
marked in `u` (strict marking below loose marking). -/
inductive Below : Tree → Tree → Prop where
  | node {own : Nat} {m m' : Bool} {ks ks' : Forest} :
      (m = true → m' = true) → FBelow ks ks' → Below (.node own m ks) (.node own m' ks')
inductive FBelow : Forest → Forest → Prop where
  | nil : FBelow .nil .nil
  | cons {t t' : Tree} {r r' : Forest} : Below t t' → FBelow r r' → FBelow (.cons t r) (.cons t' r')
end

mutual
theorem steps_eq_of_below : ∀ {t u : Tree}, Below t u → steps t = steps u
  | _, _, .node _ hk => by
    simp only [steps, fsteps_eq_of_below hk]
theorem fsteps_eq_of_below : ∀ {f g : Forest}, FBelow f g → fsteps f = fsteps g
  | _, _, .nil => rfl
  | _, _, .cons ht hr => by
    simp only [fsteps, steps_eq_of_below ht, fsteps_eq_of_below hr]
end

mutual
theorem cov_mono : ∀ {t u : Tree}, Below t u → cov t ≤ cov u
  | .node own m ks, .node _ m' ks', .node hm hk => by
    have hs := fsteps_eq_of_below hk
    have hc := fcov_mono hk
    have hle := fcov_le ks
    cases m <;> cases m' <;> simp_all [cov]
    all_goals omega
theorem fcov_mono : ∀ {f g : Forest}, FBelow f g → fcov f ≤ fcov g
  | _, _, .nil => Nat.le_refl _
  | _, _, .cons ht hr => by
    have h1 := cov_mono ht
    have h2 := fcov_mono hr
    simp only [fcov]; omega
end

/-- strict count ≤ loose count ≤ all steps. -/
theorem strict_le_loose_le_total {s l : Tree} (h : Below s l) :
    run 0 s ≤ run 0 l ∧ run 0 l ≤ steps l := by
  rw [run_zero, run_zero]
  exact ⟨cov_mono h, cov_le l⟩

/-! ## The pairing tests themselves -/

/-- The parameter values of one invocation at entry and exit, by position
(the storage keys are assumed equal; that is the key test). -/
structure Rec (α : Type) (n : Nat) where
  pre : Fin n → α
  post : Fin n → α

variable {α : Type} {n : Nat}

/-- `_full_inverse`: `later` exactly undoes `earlier`. -/
def FullInverse (earlier later : Rec α n) : Prop :=
  earlier.post = later.pre ∧ earlier.pre = later.post

/-- `_partial_inverse`: some position `earlier` changed is taken back. -/
def PartialInverse (earlier later : Rec α n) : Prop :=
  ∃ i, earlier.pre i ≠ earlier.post i ∧ later.pre i = earlier.post i ∧
    later.post i = earlier.pre i

theorem full_to_partial {J I : Rec α n} (hchg : J.pre ≠ J.post) (h : FullInverse J I) :
    PartialInverse J I := by
  have : ∃ i, J.pre i ≠ J.post i := by
    apply Classical.byContradiction
    intro hno
    apply hchg
    funext i
    apply Classical.byContradiction
    intro hi
    exact hno ⟨i, hi⟩
  obtain ⟨i, hi⟩ := this
  exact ⟨i, hi, by rw [← h.1], by rw [← h.2]⟩

/-- `call p` computes `f`, `uncall p` computes `g` with `g (f v) = v`: an
uncall that starts where the call ended is an exact inverse of it. -/
theorem uncall_pairs (f g : (Fin n → α) → (Fin n → α)) (hgf : ∀ v, g (f v) = v)
    {J I : Rec α n} (hJ : J.post = f J.pre) (hstart : I.pre = J.post)
    (hI : I.post = g I.pre) : FullInverse J I := by
  refine ⟨hstart.symm, ?_⟩
  rw [hI, hstart, hJ, hgf]

/-- If two `call`s of the same procedure pair, `f` maps the second call's
output back: `f (f v) = v` at the first call's entry `v`. -/
theorem same_direction_pair_is_involutive (f : (Fin n → α) → (Fin n → α))
    {J I : Rec α n} (hJ : J.post = f J.pre) (hI : I.post = f I.pre)
    (h : FullInverse J I) : f (f J.pre) = J.pre := by
  rw [← hJ, h.1, ← hI, ← h.2]

end RFCL.Pairing
