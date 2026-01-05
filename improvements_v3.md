# Minix improvements plan

_Last updated: 2026-01-03_

This document consolidates the current status, observed issues, and a concrete improvement roadmap for **Minix** with the goal of becoming a **top-tier convex conic optimization solver** and achieving a **meaningful wallclock speed lead** over **Clarabel** (and ultimately commercial-grade baselines such as **MOSEK**) on real workloads.

It is written to be **actionable**: each section identifies (1) what’s wrong, (2) why it matters for convergence or wallclock time, and (3) what changes to implement.

---

## 1. North-star goal

**Goal:** Deliver a production-grade HSDE + predictor–corrector interior-point solver that is:

- **Fast in wallclock time** on real benchmark suites and CVXPY workloads  
- **Robust** (rare numerical failures, reliable infeasibility detection, stable termination)
- **High accuracy** (tight residuals and gap at termination)
- **Feature-complete** for modern conic modeling:
  - LP / QP / SOCP
  - EXP / POW (and ideally generalized power cone)
  - PSD (with chordal decomposition for scale)

**Key competitive target:** “significant lead” over Clarabel or MOSEK (whichever is faster) for meaningful slices of problems (not cherry-picked).

---

## 2. Current status and observed issues

### 2.1 HSDE / Mehrotra updates already integrated

Recent changes (as of the latest round of work):

- HSDE changes in `solver-core/src/ipm/predcorr.rs`:
  - μ_aff-based σ with clipping
  - Combined step uses `(1-σ)` weighted residuals (with a small floor)
  - MCC correction uses a delta adjustment
  - alpha-stall recovery (σ bump + reg bump + refinement bump)
  - fallback diagonal scaling fix (no `sqrt`, smaller eps)
  - relaxed NT scaling interior checks for scaling purposes
- Defaults adjusted in `solver-core/src/problem.rs`:
  - `mcc_iters = 0`, `line_search_max_iters = 0` (they currently hurt DUAL2)
- Python plumbing:
  - `solver-py/src/lib.rs`, `python/__init__.py`, `python/cvxpy_backend.py`

### 2.2 Convergence issue: DUAL2 stalls at the end

DUAL2 still hits **MaxIters**. With defaults (MCC/line search off):

- μ drops to ~`6e-13`
- scaled dual residual stalls around ~`8e-4`
- **alpha collapses** because `alpha_sz` goes tiny (s/z boundary)
- unscaled dual residual remains large (~`30`)
- diagnostics confirm **s/z step-to-boundary** is the limiter, **not τ/κ**

Interpretation: complementarity is essentially solved, but we are not getting dual feasibility.

### 2.3 Benchmarks: iterations vs wallclock are telling different stories

#### Iterations (older view)
- Problems tested: 12  
- MINIX wins: 9 (75%)  
- Geometric mean: 0.28× (MINIX ~3.5× faster by iteration count)

#### Wallclock (the metric we actually care about)
- Problems tested: 12  
- MINIX wins: 4 (33%)  
- Total time: SCS 162ms, MINIX 414ms  
- Geometric mean: 0.70× (**MINIX is 1.4× slower**)

SCS wins most medium/large problems because its iterations are cheap.

**Conclusion:** The solver needs big wins in **per-iteration cost** (and also must fix the tail-stall on some QPs), or else “few iterations” won’t translate into wallclock leadership.

---

## 3. Root causes (what’s actually holding us back)

### 3.1 Tail-stall and alpha collapse is (very likely) a scaling + RHS issue

When μ is tiny but dual feasibility does not improve and α becomes tiny due to s/z:

- The Newton direction is likely “too feasibility aggressive” in some components, producing large negative `ds`/`dz` entries → `alpha_sz` becomes microscopic.
- Once α becomes tiny, progress in feasibility norms stalls even though complementarity is effectively solved.

**Two immediate suspects (both visible in current code):**

1) **Combined step RHS uses full residuals**  
   The design doc recommends `(1-σ) * residuals` for the combined-step RHS. Full residuals can be overly aggressive when σ is near 1, which is exactly when we need a centering-dominant step that stays in the cone interior.

2) **NT scaling interior checks are too strict late**  
   Example: `NonNegCone::INTERIOR_TOL = 1e-12` (and SOC uses a similarly strict tolerance). In late iterations it is normal for some entries of `s` or `z` to be extremely small but still positive; scaling should not fail or fall back in that regime.

**Status:** Both items above have been addressed (weighted RHS + relaxed scaling interior checks). If DUAL2 still stalls, the next suspects are KKT regularization and termination behavior.

---

### 3.2 A correctness bug exists in the fallback diagonal scaling (P0)

In `solver-core/src/ipm/predcorr.rs`, when NT scaling fails, the fallback diagonal scaling is built as:

- `d[i] = sqrt(max(s_i / z_i, 1e-8))`

But `ScalingBlock::Diagonal { d }` represents **H** (not **W**) and the rest of the code (and design doc) assumes:

- for NonNeg NT scaling: `H = diag(s ./ z)`

Using `sqrt(s/z)` is dimensionally inconsistent and breaks the condensed KKT system whenever this fallback path is activated.

This was fixed by removing the `sqrt` and using a smaller epsilon. The fallback now matches the expected H scaling.

---

### 3.3 Per-iteration overhead is currently dominated by allocations + KKT rebuild

The design doc’s “no allocations in hot loop” requirement is not met:

- Predictor–corrector step allocates many `Vec<f64>` each iteration (`rhs_*`, `dx*`, `dz*`, `ds*`, temporary buffers).
- KKT build uses `TriMat` and `to_csc()` each iteration (`solver-core/src/linalg/kkt.rs`), incurring:
  - repeated allocations
  - sorting/compression
  - repeated cloning (matrix stored for refinement)
- Linear solves allocate permuted RHS/solution buffers each call.
- QDLDL numeric factorization allocates L/D/workspaces each iteration and copies the matrix values to add static regularization.
- Termination checks allocate `x_bar` and `px` and compute objectives each iteration.

This matches the benchmark reality: Minix needs fewer iterations, but each iteration is currently “too expensive.”

---

### 3.4 SOC scaling path is very expensive (dense block materialization)

For SOC blocks, `ScalingBlock::SocStructured { w }` is **structured**, but the KKT assembly path currently materializes a full dense `dim×dim` block each factorization by computing columns via `P(w) e_i` (allocating `e_i` and `col_i` per column).

This is O(dim²) work and can destroy sparsity/structure.

---

### 3.5 Regularization strategy is not yet “Clarabel-class”

In `solve_ipm`, the solver enforces:

- `static_reg >= 1e-4` for LP / sparse QP
- `static_reg >= 1e-6` otherwise

This can stabilize early solves but can also prevent high-accuracy convergence and distort the Newton direction late in the solve.

There is also an additional “extra_reg” injected into diagonal H blocks when `min(s,z)` approaches `μ/100`. This can become large (capped at `1e-2`) and may be too disruptive late.

---

### 3.6 Iterative refinement is not true refinement today (matrix mismatch)

`KktSolver::factor()` stores `self.kkt_mat = Some(kkt.clone())` for refinement residual matvecs.

But `QdldlSolver::numeric_factorization()` **copies** the matrix values and adds `static_reg` to the diagonal inside `a_x`.

So:
- the factorization solves **(KKT + static_reg·I)**,
- but refinement residuals are computed against **the unshifted KKT** stored in `kkt_mat`.

This has been fixed by including the same diagonal shift in refinement matvecs, so the residuals match the factorized system.

---

### 3.7 Termination checks are currently done on the *scaled* problem, not unscaled

The design doc specifies that termination should be evaluated on unscaled problem data after undoing Ruiz scaling, so user tolerances mean what they think they mean.

Today, `check_termination` is called on the scaled problem, so feasibility/gap checks are on scaled quantities. This can cause:
- over-solving (wasting wallclock)
- under-solving (declaring success but with large unscaled residuals)

This has been fixed by evaluating termination on unscaled x/s/z with original problem data.

---

## 4. Improvement roadmap (prioritized)

### P0 — Fix correctness & convergence blockers (stop-the-world)

#### P0.1 Fix fallback diagonal scaling (critical correctness)
**Where:** `solver-core/src/ipm/predcorr.rs`, fallback when NT scaling fails.

**Change:**
- Replace `sqrt(max(s_i/z_i, eps))` with `max(s_i / z_i, eps)` **without sqrt**.
- Use a much smaller `eps` (e.g., `1e-18`), or better: `eps` tied to μ and norms.

**Expected impact:**  
Avoids incorrect H; should reduce alpha collapse and late-iteration weirdness.

---

#### P0.2 Prevent NT scaling from failing near convergence
**Where:** `NonNegCone::is_interior_*`, `SocCone::is_interior_*`, and scaling code that rejects points as “not interior”.

**Change options:**
- Add a separate “scaling interior” predicate that only checks strict positivity (`> tiny_abs`) rather than `> tol*||s||∞`.
- Or keep strict barrier interior checks but in scaling update do:
  - if values are positive but small, accept NT scaling anyway,
  - and optionally clamp only for division safety (not for correctness).

**Expected impact:**  
Stops spurious fallback paths and avoids scaling failures in the tail.

---

#### P0.3 Implement combined-step RHS weighting `(1-σ)` (design doc parity)
**Where:** `solver-core/src/ipm/predcorr.rs`.

**Current:** `rhs_x_comb = -r_x`, `rhs_z_comb = d_s - r_z`, etc.

**Change:** Use:
- `rhs_x_comb = -(1 - σ) * r_x`
- `rhs_z_comb` should incorporate the same `(1-σ)` feasibility weighting (consistent with the design doc)
- make the feasibility weight a knob

**Expected impact:**  
Improves late-iteration step quality and reduces step-to-boundary collapse when σ is large.

---

#### P0.4 Make alpha-stall a first-class recovery path
**Where:** `predictor_corrector_step` after computing α.

**Recovery actions (in order):**
1) If `alpha_sz` is tiny and μ is already small: force σ closer to 1 (centering step).
2) Increase KKT regularization *in the solve* (not by injecting huge extra_reg into H).
3) Increase refinement iterations when residuals stagnate.
4) Optional: do a cheap “recentering” step.

**Expected impact:**  
Turns “stuck at the end” into controlled convergence instead of α→0 forever.

---

#### P0.5 Fix iterative refinement matrix mismatch
**Where:** `solver-core/src/linalg/kkt.rs` and `solver-core/src/linalg/qdldl.rs`.

**Change options:**
- Move static regularization into KKT assembly (so `kkt_mat` == factorized matrix), and remove diagonal-shift injection inside `numeric_factorization`, **or**
- Keep current scheme but make refinement residual matvec include the same diagonal shifts used during factorization.

**Expected impact:**  
Refinement becomes real, improving tail accuracy and stability.

---

#### P0.6 Termination on unscaled data (design doc parity)
**Where:** `solver-core/src/ipm/termination.rs` and `solver-core/src/ipm/mod.rs`.

**Change:**
- Compute termination metrics on unscaled x/s/z by applying the stored Ruiz scaling.
- Keep a separate “debug/verbose” block for scaled diagnostics if needed.

**Expected impact:**  
Correctness + benchmark fairness + reduces wasted iterations.

---

### P1 — Wallclock performance: remove overhead and match Clarabel-class engineering

#### P1.1 Add real timing and a per-iteration profile breakdown
Add timers for:
- residual computation
- scaling computation
- KKT assembly/update
- factorization
- solves (+ refinement)
- cone operations
- termination checks

Without this, you cannot reliably target wallclock wins.

---

#### P1.2 Make the IPM loop allocation-free
Introduce a `Workspace` struct that owns all per-iteration buffers and replace `Vec::new()` in the hot loop with reuse.

---

#### P1.3 Stop rebuilding the KKT matrix from triplets every iteration
Build the KKT sparsity pattern once and update numeric values in place.

This requires:
- precomputing write maps from scaling blocks to CSC positions
- for SOC, a strategy that avoids per-iteration dense rebuilds where possible

---

#### P1.4 Batch the two RHS solves (true “two-solve strategy”)
**Where:** `KktSolver::solve_two_rhs` and callers.

Currently: just calls `solve()` twice (two allocations + two permutations).

Target:
- One permutation pass for both RHS vectors
- One solve pass reusing buffers
- Optional: implement forward/back substitution loops that process 2 RHS vectors together

---

#### P1.5 Reuse QDLDL numeric factorization workspaces across iterations
Avoid per-iteration allocations:
- reuse `l_p/l_i/l_x`, `d/d_inv`, and work arrays
- avoid copying `a_x` just to add static reg (precompute diagonal indices)

---

#### P1.6 SOC scaling / KKT insertion redesign
At minimum:
- eliminate per-column allocations in SOC dense assembly (preallocate `e_i` and `col_i` buffers)
- write dense block entries into fixed CSC positions without rebuilding everything

Longer term:
- avoid dense SOC blocks in KKT altogether (structured/low-rank update approach), if SOC dimensions can be large.

---

#### P1.7 Termination: avoid allocations and expensive objective computations every iteration
Compute objective/gap less frequently and reuse buffers.

---

### P2 — Feature parity then lead features

- EXP / POW with correct nonsymmetric scaling + 3rd-order correction
- PSD + chordal decomposition
- multiple linear solver backends (supernodal / Pardiso / etc.)
- warm starts + problem data updates
- mixed precision / GPU (longer-term)

---

## 5. Concrete “next steps” checklist (recommended order)

### Week 0: Fix DUAL2 stall + correctness
- [x] Fix fallback diagonal scaling sqrt bug
- [x] Prevent NT scaling “false failures” near convergence
- [x] Implement `(1-σ)` combined-step RHS weighting (make it tunable)
- [x] Add alpha-stall recovery (σ bump + reg bump + refinement bump)
- [x] Fix iterative refinement matrix mismatch
- [x] Move termination checks to unscaled quantities

### Week 1: Measure and remove obvious overhead
- [ ] Add timing breakdown in SolveInfo + verbose perf output
- [ ] Add predictor-corrector workspace to remove allocations
- [ ] Make KKT solve buffers reusable (no per-solve Vec allocations)

### Week 2+: Make per-iteration cost competitive with Clarabel
- [ ] In-place KKT numeric update (no TriMat rebuild)
- [ ] Implement real multi-RHS solves
- [ ] Reuse QDLDL numeric factorization workspaces

### Later: feature parity and lead features
- [ ] EXP / POW (BFGS scaling + 3rd order correction)
- [ ] PSD + chordal decomposition
- [ ] Alternative linear solvers
- [ ] Warm starts / problem updates
- [ ] GPU / mixed precision

---

## 6. Success criteria (what “done” means)

### Reliability
- No MaxIters on standard suites except truly pathological cases
- Correct statuses for infeasible/unbounded cases
- Stable behavior as μ → tiny (no “α → 0 forever” loops)

### Accuracy
- Achieve target tolerances (e.g., 1e-8 feasibility / gap) on typical QP/SOCP sets
- No “μ small but residual large” false convergence

### Wallclock
- For CPU baselines:
  - Match or beat Clarabel on QP/SOCP in geometric mean on a fair benchmark set
  - Beat SCS on medium-size problems at moderate-to-high accuracy
- For repeated solves (future):
  - demonstrate strong wins with warm starts / problem updates

---

## 7. Notes / references used for this plan

- Project design doc:
  - condensed KKT form + “two-solve strategy”
  - `(1-σ)` combined RHS weighting
  - performance requirement: allocation-free hot loop
  - warning about large SOC dense blocks
  - termination on unscaled data after Ruiz

- Current implementation:
  - `solver-core/src/ipm/predcorr.rs` (scaling fallback + combined RHS + step-size behavior)
  - `solver-core/src/linalg/kkt.rs` (KKT assembly + SOC dense materialization)
  - `solver-core/src/linalg/qdldl.rs` (static reg injection + per-iteration allocations)
  - `solver-core/src/ipm/termination.rs` (scaled termination + allocations)
