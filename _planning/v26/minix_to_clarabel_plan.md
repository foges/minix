# Minix → Clarabel Performance Plan

This is a concrete, engineering-focused plan to close the performance and iteration-count gap vs Clarabel for the SOCP / HSDE / IPM2 path.

It is written assuming the current code layout in `solver-core` matches the `rustfiles-solver-core` index you pasted (notably `src/ipm2/predcorr.rs`, `src/linalg/kkt.rs`, `src/scaling/nt.rs`).

---

## 0) Restate the observed gap (so we optimize the right thing)

From your timings:

- **PRIMALC8**: 1183ms vs Clarabel 5.65ms (**~209× slower**) with **23 vs 8 iterations**.
  - 80% of Minix time is **KKT factorization + solve** (943ms / 1183ms).
  - Per-iteration: ~51ms vs ~0.7ms.

- **DUALC1**: 22ms vs 1.31ms (**~17× slower**) with **30 vs 5 iterations**.

This means we must simultaneously:

1. **Fix the linear algebra (per-iter)** – PRIMALC8 is dominated by KKT factor/solve.
2. **Fix the step quality / neighborhood logic (iterations)** – DUALC1 (and others) are doing far too many iterations.

---

## 1) Diagnosis: what is “terribly wrong”

### 1.1 Dense SOC blocks are being inserted into the global KKT

In `src/linalg/kkt.rs`, SOC scaling is updated via:

- `update_soc_block_in_place(...)`

This function **fills every entry of the SOC block** (upper triangle) by looping over columns and calling `quad_rep_soc_in_place` per column. That implies:

- Update cost per SOC cone is **O(k²)** writes (k = cone dim).
- The global KKT sparsity pattern now contains a **dense k×k block**.

For “QP → SOCP epigraph” conversions, you often end up with a **single huge SOC cone** of dimension ~`n + 2`.
A dense block that large will:

- explode KKT nnz
- explode fill-in
- make LDL factorization and solve orders of magnitude slower

This is consistent with your PRIMALC8 profile: ~550ms factorization and ~393ms solves.

**Bottom line:** even with SuiteSparse, factoring a matrix that contains a dense SOC block of dimension ~O(n) is not going to be competitive.

### 1.2 SOC centrality / neighborhood is (effectively) not enforced in line search

`src/ipm2/predcorr.rs::centrality_ok_nonneg_trial` currently checks only NonNeg cones.
For SOC cones it does:

- identify they are not NonNeg
- increment offset
- `continue`  

So SOC cones are *never* required to satisfy a neighborhood condition during backtracking.

This yields classic failure mode:

- iterates drift to SOC boundary
- complementarity becomes ill-conditioned
- step sizes collapse
- iterations explode

You already saw this manifest as “rides the SOC boundary” and `alpha → 0` behavior.

### 1.3 SOC spectral decomposition cancellation exists in *two* places

You fixed the cancellation in `src/scaling/nt.rs` (great), but **IPM2 has its own in-place SOC spectral decomposition** in `src/ipm2/predcorr.rs`:

- `spectral_decomposition_in_place` does `lambda[1] = t - x_norm;` (catastrophic near boundary).

If IPM2 uses the in-place path (it does), you can still get garbage eigenvalues and step directions even if `src/scaling/nt.rs` is fixed.

### 1.4 Proximity control is currently “NonNeg-centric”

`apply_proximity_step_control` computes proximity using `s_i * z_i`, which is correct for NonNeg but **not** for SOC (Jordan product / spectral values / NT-scaled complementarity).
This makes the step controller blind to SOC badness.

---

## 2) Target architecture (what we should converge to)

To get Clarabel-like performance, we need to match two high-level design choices:

### 2.1 Don’t put dense SOC scaling blocks into the sparse KKT

Instead, represent the SOC scaling contribution as either:

- **(Preferred) Low-rank augmentation** (rank-2 update per SOC block) so nnz grows O(k), not O(k²)
- **or a reduced / condensed system** (Schur complement) that only needs fast `H^{-1}·v` applies, not explicit H

### 2.2 Enforce SOC neighborhood with an escape hatch (sigma/MCC)

A pure “halve alpha until centrality ok” strategy fails if you are already outside the neighborhood.
When SOC centrality fails, you must **re-center**:

- increase `sigma` (more centering)
- or run **multiple correction steps (MCC)**

Clarabel does not get stuck shrinking alpha forever because it has a mechanism to restore centrality.

---

## 3) Phase plan (ordered by ROI)

### Phase A — correctness & stability (must do first)

#### A1) Fix SOC lambda2 cancellation in IPM2 in-place spectral decomposition

**Patch:** `0001_ipm2_stable_lambda2_inplace.patch`

- File: `solver-core/src/ipm2/predcorr.rs`
- Function: `spectral_decomposition_in_place`
- Replace `lambda[1] = t - x_norm` with stable `(t² - ||x||²) / (t + ||x||)`.

This eliminates the “step direction exploded” class of failures for any code path that uses IPM2 in-place scaling.

#### A2) Add SOC sanity assertions (debug-only)

Add `debug_assert!` checks (behind an env flag if you prefer) that:

- For any SOC block used in scaling: `discriminant(v) > 0` (interior)
- For `jordan_sqrt`: eigenvalues >= 0

These will catch early “we left the SOC interior” bugs.

---

### Phase B — iteration count (SOC neighborhood + adaptive centering)

Goal: bring DUALC1 from ~30 iters → single digits, and PRIMALC* from ~23 → ~8–12.

#### B1) Implement SOC neighborhood check in line search using NT-scaled complementarity

**Where:** `src/ipm2/predcorr.rs::centrality_ok_nonneg_trial`

For SOC blocks, check the spectral values of the *scaled* complementarity.
A robust way (what you already started doing) is:

1. form `tmp = Q_{sqrt(s)}(z)` (quadratic representation)
2. take spectral values of `tmp` → these are (roughly) `λ_s * λ_z`
3. compare `sqrt(tmp_eigs)` (geometric mean) to `mu_trial`

Neighborhood condition:

- `comp_lo >= beta * mu_trial`
- `comp_hi <= gamma * mu_trial`

**Important implementation details:**

- No allocations inside line search.
  Use a scratch buffer in `IpmWorkspace` (you already have `SocScratch`).
- Use the **stable lambda2** everywhere.
- Return a structured violation (not just bool) so the caller can decide whether to backtrack alpha or increase sigma.

#### B2) Add the missing “escape hatch”: sigma bump (or MCC) when SOC centrality fails

**Why:** if the current iterate is already outside the SOC neighborhood, shrinking alpha cannot fix it.

**Implementation sketch (minimal churn):**

- Change `centrality_ok_nonneg_trial(...) -> bool` into something like:

  - `centrality_check_trial(...) -> Result<(), CentralityViolation>`

- In the line search loop:

  - If failure is NonNeg: keep halving `alpha` (current behavior)
  - If failure is SOC:
    - if `alpha` is still “large”: you can backtrack a few times
    - but if alpha would drop below `alpha_min` **or** the violation persists at small alpha:
      - set a flag `needs_more_centering = true`
      - break out of line search

- In `predictor_corrector_step_in_place` (or the outer solve loop):

  - if `needs_more_centering`:
    - increase `sigma` (e.g. `sigma = max(sigma, sigma_floor)` where `sigma_floor` depends on violation)
    - recompute the corrector RHS and resolve KKT
    - retry line search

This is the simplest version of “MCC-like” behavior and usually gets you from “alpha collapses” → “makes progress”.

#### B3) Fix proximity step control to include SOC

Update `apply_proximity_step_control` so its proximity measure respects cone type:

- NonNeg: keep current elementwise logic
- SOC: use the same `sqrt(eigs(Q_{sqrt(s)}(z)))` metric
- (optional later) PSD: use eigenvalues / trace-based proxy

This is important because even if you add SOC centrality checks, the proximity controller can still push you into bad regions if it’s blind to SOC.

---

### Phase C — linear algebra (the real performance work)

Goal: make PRIMALC8 per-iteration cost look like Clarabel.

#### C0) Ensure we are using the fastest backend

Before major refactors:

- Build with `suitesparse-ldl` feature enabled.
- Make SuiteSparse the default backend for large problems.

This is a “cheap win” that can easily shave large constant factors.

#### C1) Stop inserting dense SOC blocks into KKT (core fix)

There are two viable approaches.

##### Option 1 (recommended): SOC low-rank augmentation (rank-2) with 2 auxiliary variables per SOC cone

Key idea:

- The SOC scaling/Hessian-like operator has only **three distinct eigenvalues**: two special directions + one repeated `(k-2)` times.
- That means it can be represented as:

  `H = a * I + b * u uᵀ + c * v vᵀ`  (rank-2 update)

Instead of materializing dense H inside the global KKT, represent the rank-2 part via bordering:

Schur complement identity:

- `D - U Uᵀ` can be represented by the augmented matrix:

  ```
  [ D   U ]
  [ Uᵀ  I ]
  ```

If you pick `D = -aI` and `U = [sqrt(b) u, sqrt(c) v]`, then eliminating the 2 aux vars produces `-H`.

**What changes in Minix:**

- KKT dimension increases by `2 * (#soc_cones)`.
- KKT nnz increases by **O(sum soc_dim)**, not O(sum soc_dim²).
- Factorization becomes comparable to Clarabel.

**Where to implement:**

- `src/linalg/kkt.rs`
  - modify `compute_h_block_positions` and SOC block handling
  - stop allocating dense SOC block positions
  - instead allocate:
    - diagonal for SOC variables
    - 2 “border” columns/rows per SOC cone
- Update RHS packing/unpacking accordingly.

##### Option 2: Condensed system / Schur complement solve

Eliminate slack variables using `H^{-1}` and factor a smaller system.

This is algorithmically clean but requires:

- efficient sparse formation of the condensed matrix
- and careful handling for SOC `H^{-1}` (still low-rank)

Option 1 is usually simpler to integrate with an existing sparse LDL KKT solver.

#### C2) Add instrumentation to prove the fix

Add debug/perf prints (behind env flags) for:

- KKT dimension
- KKT nnz
- fill ratio after factorization (if available)
- factorization time
- solve time

This lets you verify that the SOC dense block is gone.

---

## 4) Benchmark expectations & acceptance criteria

After Phase A+B (no KKT refactor yet):

- DUALC1 iters should drop substantially (goal: ≤10).
- The solver should no longer “ride the SOC boundary” / stall at alpha≈0.

After Phase C (SOC low-rank KKT):

- PRIMALC8 should drop from ~1.2s to **tens of ms**.
- With iterations reduced to ~8–12, total should be in the same ballpark as Clarabel.

If the gap remains large after C1, the remaining suspects are:

- backend mismatch (QDLDL vs SuiteSparse)
- ordering (AMD / CAMD) differences
- repeated symbolic factorization
- extra solves/refinement per iteration

---

## 5) Patch list

### Already in your workspace / prior work

- `stable_lambda2_soc.patch` (fixes SOC cancellation in `src/scaling/nt.rs`)

### New patch (drop-in)

- `0001_ipm2_stable_lambda2_inplace.patch`
  - fixes the same cancellation in `src/ipm2/predcorr.rs::spectral_decomposition_in_place`

### Next patches to implement (non-trivial, but directly mapped to the plan)

- `0002_soc_neighborhood_check_ipm2.patch`
  - SOC neighborhood enforcement in line search (no allocations, uses workspace scratch)

- `0003_soc_sigma_bump_on_centrality_fail.patch`
  - when SOC centrality fails, increase sigma / re-center instead of infinite alpha backtracking

- `0004_soc_lowrank_kkt.patch`
  - removes dense SOC block insertion and replaces it with rank-2 augmentation

---

## 6) Implementation notes / “gotchas”

1. **Avoid allocations in line search.**
   Your earlier SOC neighborhood patch allocated vectors each call, which will show up in profiles once KKT is fixed.

2. **Don’t compare SOCP-epigraph vs QP-direct without noting it.**
   If Clarabel is being run in direct-QP mode while Minix is run in SOCP-epigraph mode, the comparison is inherently unfair.
   (Clarabel explicitly optimizes for direct quadratic objectives; epigraph reformulations can increase fill.)

3. **SOC neighborhood check must be paired with re-centering.**
   Otherwise you just create early alpha stalls.

4. **Once SOC dense blocks are removed, iteration count becomes the main lever.**
   That’s when SOC centrality/proximity work pays off.

---

## 7) Quick command cheat sheet

Apply patches:

```bash
git apply /path/to/stable_lambda2_soc.patch
git apply /path/to/0001_ipm2_stable_lambda2_inplace.patch
```

Re-run regression:

```bash
MINIX_VERBOSE=2 cargo run --release -p solver-bench -- regression --socp-only --filter PRIMALC8
```

