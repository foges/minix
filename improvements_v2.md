# Minix improvements plan

_Last updated: 2026-01-03_

This document consolidates the current status, observed issues, and a concrete improvement roadmap for **Minix** with the goal of becoming a **top-tier convex conic optimization solver** and achieving a **meaningful speed lead** over **Clarabel** (and ultimately commercial-grade baselines such as **MOSEK**) on real workloads.

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
  - Combined step uses **full residuals**
  - MCC correction uses a delta adjustment
  - alpha-stall diagnostics (confirm s/z step-to-boundary is the limiter)
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
- new diagnostics confirm **s/z step-to-boundary** is the limiter, **not τ/κ**

Interpretation: complementarity is essentially solved, but we are not getting dual feasibility.

### 2.3 Benchmarks: iterations vs wallclock are telling different stories

#### Iterations (older view)

- Problems tested: 12  
- MINIX wins: 9 (75%)  
- Geometric mean: 0.28× (MINIX ~3.5× faster by iteration count)

Standouts:
- QPCBLEND: 11 vs 2000 iterations
- HS21: 7 vs 275
- DUALC1: 12 vs 200
- ZECEVIC2: 5 vs 75

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

- The computed Newton direction is likely “too feasibility aggressive” in some components, producing large negative `ds`/`dz` entries → `alpha_sz` becomes tiny.
- Once α becomes tiny, progress in feasibility norms stalls even though the method keeps “centering” or driving complementarity down.

**Two immediate suspects (both visible in current code):**

1) **Combined step RHS uses full residuals**  
   The design doc recommends `(1-σ) * residuals` for combined-step linear RHS. Full residuals can be overly aggressive when σ is near 1, which is exactly when we need a centering-dominant step that stays in the cone interior.

2) **NT scaling “interior” checks are too strict for late iterations**  
   Example: `NonNegCone::INTERIOR_TOL = 1e-12` and SOC uses the same order.  
   In late iterations it is normal for some entries of `s` or `z` to be very small; this should not automatically be treated as “not interior” as long as positivity holds. Strict checks can trigger scaling failure/recovery behavior at exactly the worst time.

### 3.2 A correctness bug exists in the fallback diagonal scaling

In `solver-core/src/ipm/predcorr.rs`, when NT scaling fails, the fallback diagonal scaling is built as:

- `d[i] = sqrt(max(s_i / z_i, 1e-8))`

But `ScalingBlock::Diagonal` represents **H** (not **W**), and for NonNeg NT scaling we want:

- `H = diag(s ./ z)`

Using `sqrt(s/z)` is dimensionally inconsistent and will break the condensed system if this fallback path is activated (which can happen when `is_interior_*` fails).

This is a high-priority fix because it can create exactly the “alpha collapses near the end” behavior.

### 3.3 Per-iteration overhead is currently dominated by allocations + KKT rebuild

The design doc’s “no allocations in hot loop” requirement is not met:

- Predictor–corrector step allocates many `Vec<f64>` each iteration (`rhs_*`, `dx*`, `dz*`, `ds*`, temporary buffers).
- KKT build uses `TriMat` and `to_csc()` each iteration (`solver-core/src/linalg/kkt.rs`), incurring:
  - repeated allocations
  - sorting/compression
  - repeated cloning (matrix stored for refinement)
- Linear solves allocate permuted RHS/solution buffers each call.
- QDLDL numeric factorization allocates L/D/workspaces each iteration.
- Termination computes objectives and constructs `x_bar`, `z_bar`, `px` allocating multiple vectors each check.

This is consistent with wallclock losses: IPM iterations are expensive, and we currently pay extra overhead beyond the factorization itself.

### 3.4 SOC scaling path is very expensive (and currently materializes dense blocks)

`ScalingBlock::SocStructured::to_matrix()` constructs a full dense matrix by applying the quadratic representation to basis vectors, allocating per column. That is O(d²) work with a lot of allocation and will dominate runtime for larger SOC blocks.

Design doc already warned about this (“dense H blocks can destroy sparsity”).

### 3.5 Regularization strategy is not yet “Clarabel-class”

Current heuristics in `solve_ipm` enforce `static_reg >= 1e-4` for LP / sparse QP. That can stabilize early solves, but can also:

- bias the Newton system late in the solve
- prevent high-accuracy convergence for hard instances
- create poor step directions in the tail

We also add “extra_reg” into diagonal scaling blocks when `min(s)` or `min(z)` becomes small relative to μ; this can become large (capped at `1e-2`) and may be too disruptive late in the solve.

---

## 4. Improvement roadmap (prioritized)

### P0 — Fix correctness & convergence blockers (must do first)

#### P0.1 Fix fallback diagonal scaling (critical correctness)
**Where:** `solver-core/src/ipm/predcorr.rs`, fallback when NT scaling fails.

**Change:**
- Replace `sqrt(max(s_i/z_i, eps))` with `max(s_i / z_i, eps)` **without sqrt**.
- Use a much smaller `eps` (e.g., `1e-18`) or better: `eps = min_eps * (mu / max(z_i,1))` style.

**Expected impact:**  
Avoids incorrect H; should reduce alpha collapse and late-iteration weirdness.

---

#### P0.2 Relax “interior” checks for scaling purposes
**Where:** `NonNegCone::INTERIOR_TOL`, `SocCone::INTERIOR_TOL`, and scaling code that rejects points as “not interior”.

**Change:**
- Separate **“barrier interior”** vs **“numerical scaling interior”**:
  - Barrier operations can remain conservative.
  - Scaling should accept very small positive values as interior.
- For NonNeg: treat interior as `s_i > 0` (or `> tiny_abs`, e.g. `1e-300`) for scaling.
- Alternatively: define `tol = eps * max(1, ||s||∞, ||z||∞, μ)` and set eps near `1e-18`.

**Expected impact:**  
Prevents spurious scaling failure late in the solve; reduces need for fallback paths and “push_to_interior” recovery.

---

#### P0.3 Implement combined-step RHS weighting `(1-σ)` (design doc parity)
**Where:** `solver-core/src/ipm/predcorr.rs`.

**Current:** `rhs_x_comb = -r_x`, `rhs_z_comb = -r_z`, `rhs_tau_comb = -r_tau`.

**Change:** Use:
- `rhs_x_comb = -(1 - σ) * r_x`
- `rhs_z_comb = -(1 - σ) * r_z`
- `rhs_tau_comb = -(1 - σ) * r_tau`

Make the feasibility weight a tunable knob (e.g., `feas_weight = 1 - σ`, with optional floor like `max(1-σ, 0.05)`).

**Expected impact:**  
Improves late-iteration step quality and reduces step-to-boundary collapse when σ is large.

---

#### P0.4 Make alpha-stall a first-class recovery path
**Where:** `predictor_corrector_step` after computing α.

**Detection:** You already have diagnostics showing `alpha_sz` is the limiter.

**Recovery actions (in order):**
1) Increase σ (more centering) when α is tiny (`α < 1e-3`) and μ is already small.
2) Increase regularization *in the KKT solve* (not by perturbing H massively).
3) Increase iterative refinement iterations when residuals stagnate.
4) As a last resort, do a “recentering step” (pure centering direction) to move away from boundary.

**Expected impact:**  
Turns “stuck at the end” into a controlled, convergent behavior.

---

#### P0.5 Revisit “extra_reg” injected into H
**Where:** `predcorr.rs` scaling block construction.

**Change:**
- Replace the current “if min(s,z) < μ/100 then add up to 1e-2 to H diagonal” with:
  - a gentler schedule tied to `static_reg` / `dynamic_reg_min_pivot`
  - or move regularization into the KKT diagonal blocks (`P + ρI`, `H + δI`) with principled scaling.
- Make it measurable: report how often and how large extra_reg becomes.

**Expected impact:**  
Prevents late-iteration direction distortion and reduces “μ small but residual large”.

---

### P1 — Wallclock performance: remove overhead and match Clarabel-class engineering

#### P1.1 Add real timing and a per-iteration profile breakdown
**Where:** `solver-core/src/ipm/mod.rs` and `SolveInfo`.

Add timers for:
- residual computation
- scaling computation
- KKT assembly/update
- factorization
- solves (+ refinement)
- cone operations
- termination checks

**Why:** Without this you can’t tell if you’re losing to SCS because of factorization, matrix build, refinement, or pure overhead.

---

#### P1.2 Make the IPM loop allocation-free
**Where:** `predcorr.rs`, `termination.rs`, `kkt.rs`.

Approach:
- Introduce a `Workspace` struct that owns all per-iteration buffers:
  - RHS vectors
  - dx/dz intermediate directions (affine and basis directions)
  - ds/dz scratch
  - temporary vectors used for SOC quadratic rep, matvecs, etc.
- Replace `Vec::new()` in the hot loop with `.fill(0.0)` or `.copy_from_slice()` on preallocated buffers.

**Expected impact:**  
Large speedup for small/medium problems where allocation dominates.

---

#### P1.3 Stop rebuilding the KKT matrix from triplets every iteration
**Where:** `solver-core/src/linalg/kkt.rs` (`build_kkt_matrix`).

Current approach:
- build `TriMat` → `to_csc()` every iteration.

Target approach:
- Build the KKT **sparsity pattern once** (CSC `indptr/indices`) during `initialize`.
- Keep a mutable `values` array and update numeric values in place each iteration:
  - update P block values (constant)
  - update A / Aᵀ block values (constant)
  - update H block values (changes each iteration, but positions are fixed)

This requires:
- precomputing “write maps” from cone scaling blocks → positions in the CSC structure
- for SOC, decide whether to keep dense block materialization or use a structured update path (see P1.6)

**Expected impact:**  
Often the single biggest wallclock win.

---

#### P1.4 Batch the two RHS solves (true “two-solve strategy”)
**Where:** `KktSolver::solve_two_rhs` and callers.

Currently:
- `solve_two_rhs` just calls `solve` twice, each doing permutations + allocations.

Target:
- One permutation pass for both RHS vectors
- One solve pass that reuses the same allocated work buffers
- Optional: pack RHS into a small dense matrix and run forward/back-sub with BLAS-2 style loops (even without BLAS this improves cache locality)

**Expected impact:**  
Notable reduction in solve overhead per iteration.

---

#### P1.5 Reuse QDLDL factorization workspaces across iterations
**Where:** `solver-core/src/linalg/qdldl.rs`.

Current:
- allocates `l_p, l_i, l_x, d, d_inv, bwork, iwork, fwork` every factorization.

Target:
- allocate these once after symbolic factorization (since `nnz_l` is known)
- reuse across all numeric factorizations
- avoid copying `a_x` just to add static reg:
  - either add static reg directly into the KKT values array
  - or precompute diagonal positions and apply a cheap in-place update

**Expected impact:**  
Large speedup; reduces allocator + memory traffic.

---

#### P1.6 SOC scaling and KKT insertion must be redesigned for speed
**Where:** `ScalingBlock::SocStructured::to_matrix()` and KKT assembly.

Options:
1) **Small SOC blocks:** keep dense representation, but compute/update entries in-place without allocations.
2) **Large SOC blocks:** avoid dense H entirely:
   - use a structured representation (Jordan algebra) to apply H and H⁻¹ without materializing the full block
   - redesign KKT solve so that the SOC contribution does not destroy sparsity (e.g., specialized augmentation / extra variables / low-rank updates)

At minimum: remove the current “apply quad rep to basis vectors” method; compute the dense block directly and write it into fixed CSC positions.

**Expected impact:**  
Prevents SOC-heavy problems from becoming intractable.

---

#### P1.7 Termination: stop allocating and computing expensive objectives every iteration
**Where:** `solver-core/src/ipm/termination.rs`.

Changes:
- Do not allocate `x_bar`, `s_bar`, `z_bar` each call.
- Do not build `px` vector each call.
- Compute objective/gap less frequently (e.g., every 5–10 iterations) unless close to termination.
- For feasibility-driven termination, focus on residual norms (already computed).

**Expected impact:**  
Small-to-medium speedup; also reduces noise in profiling.

---

### P2 — Feature parity with Clarabel, then “lead” features

Once P0 and P1 are done, Minix can compete on wallclock. To get a **lead**, we need more.

#### P2.1 Complete cone support
Implement per design doc:

- Exponential cone (EXP) with correct nonsymmetric scaling + third-order correction
- Power cone (POW) (and consider generalized power cone if targeting Clarabel feature parity)
- PSD cone:
  - start with dense blocks
  - then add chordal decomposition / clique merging

#### P2.2 Multiple linear solver backends
Clarabel exposes multiple direct solvers (QDLDL, CHOLMOD, Pardiso, MA57, …). Matching this matters because “fastest” depends strongly on matrix structure.

For Minix:
- Keep a fast pure-Rust baseline (QDLDL-class) for small/medium problems.
- Add optional feature-gated backends:
  - SuiteSparse CHOLMOD (supernodal)
  - Pardiso / MKL (if licensing permits for your distribution)
  - MA57 / HSL (optional)

#### P2.3 Problem updates and warm starts
To win on embedded / MPC / repeated solves:
- allow updating q/b (and possibly A) without rebuilding everything
- warm-start x/s/z with safe interior recovery

#### P2.4 GPU / mixed precision (longer-term)
Clarabel has a GPU variant (CuClarabel). Beating that requires:
- GPU sparse direct solve or hybrid strategy
- mixed precision factorization + refinement (FP32/TF32 with FP64 refinement)
- kernel fusion for cone operations

This is a major project, but it’s a plausible “lead” once CPU parity is achieved.

---

## 5. Concrete “next steps” checklist (recommended order)

### Week 0: Fix DUAL2 stall + correctness
- [ ] Fix fallback diagonal scaling sqrt bug
- [ ] Relax interior checks for scaling / late iterations
- [ ] Implement `(1-σ)` combined-step RHS weighting (make it tunable)
- [ ] Add alpha-stall recovery (σ bump + reg bump + refinement bump)
- [ ] Reassess/remove aggressive `extra_reg` injected into H

### Week 1: Measure and remove obvious overhead
- [ ] Add timing breakdown in SolveInfo + verbose perf output
- [ ] Build a predictor-corrector workspace to remove allocations
- [ ] Make KKT solve buffers reusable (no per-solve Vec allocations)

### Week 2+: Make per-iteration cost competitive with Clarabel
- [ ] In-place KKT numeric update (no TriMat rebuild)
- [ ] Batch the two RHS solves correctly
- [ ] Reuse QDLDL numeric factorization workspaces

### Later: feature parity and lead features
- [ ] EXP / POW with correct scaling + 3rd order correction
- [ ] PSD + chordal decomposition
- [ ] Alternative linear solvers
- [ ] Warm starts / problem updates
- [ ] GPU strategy

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

- The project’s design doc spells out:
  - condensed KKT form + “two-solve strategy”
  - `(1-σ)` combined RHS weighting
  - performance requirement: allocation-free hot loop
  - warning about large SOC dense blocks

- Clarabel public docs describe:
  - supported cone types (including EXP/POW and PSD triangle)
  - multiple linear solver backends (QDLDL, CHOLMOD, Pardiso, MA57, …)
  - chordal decomposition and problem data updates

(Links intentionally omitted here; see project notes or add a References section if you want this file to be self-contained.)

