# Minix improvements plan (convergence + wallclock)

_Last updated: 2026-01-03_

This document consolidates:

1. **Where MINIX is today** (benchmarks + observed failure modes).
2. **What must change to beat Clarabel/MOSEK on wallclock** (not iterations).
3. **Concrete engineering work items** (P0/P1/P2) with measurable success criteria.
4. **Patch: wallclock killers** (what the accompanying patch does and why).

---

## Current state summary

### What improved recently
Recent HSDE / Mehrotra changes substantially improved convergence on many Maros–Mészáros QP instances:

- Big iteration-count wins on several instances (e.g., QPCBLEND, HS21, DUALC1, ZECEVIC2).
- Remaining failures are “endgame stalls”: **μ and gap go to ~0**, but feasibility (especially dual) stalls and **α collapses**.

### What still fails
DUAL2 (and a few others) still hit MaxIters in the “endgame”:

- μ drops extremely small (~1e-13 range).
- Scaled dual residual stalls around ~8e-4.
- Unscaled dual residual stays large (tens).
- α collapses because the **s/z step-to-boundary** becomes tiny (confirmed by alpha-stall diagnostics).

### Wallclock reality check (important)
When measured in **wallclock**, SCS often wins on medium/large instances:

- IPM needs far fewer iterations, but each iteration is dominated by **KKT assembly + factorization + solves**.
- In MINIX, the **per-iteration fixed costs** and **memory traffic** are currently too high.

This means: even if convergence is “fixed”, MINIX still loses unless we eliminate the current wallclock killers.

---

## What “winning on wallclock” requires

To beat Clarabel (and compete with MOSEK), MINIX needs:

1. **Near-zero allocations in the iteration loop** (including KKT assembly/factorization/solve path).
2. **A KKT pipeline that updates numeric values without rebuilding sparsity** whenever structure is fixed (typical LP/QP).
3. **A faster sparse factorization backend** (or an extremely well-tuned QDLDL path) plus careful ordering.
4. **Low-overhead multi-RHS solves** (HSDE requires multiple solves per iteration).
5. **Well-chosen regularization + scaling** that avoids “tiny-step endgame stalls” without exploding iteration count.
6. **End-to-end profiling discipline**: every change must be justified by perf/flamegraphs and regressions.

---

## Wallclock killers (ranked)

### WK1 — Rebuilding KKT sparsity each iteration (TriMat → CSC)
Reconstructing the KKT matrix from triplets every iteration is expensive:
- allocs + pushes + sorting/coalescing + cache misses
- dominates wallclock for many sizes

**Fix:** cache sparsity pattern and update only the numeric values whenever possible.

### WK2 — KKT solve path allocates several O(n+m) vectors per solve
Iterative refinement was allocating multiple large vectors **per solve**; HSDE calls KKT solves multiple times per iteration, multiplying the damage.

**Fix:** keep a persistent solve workspace inside `KktSolver`.

### WK3 — QDLDL numeric factorization allocates many buffers per factorization
Even if factorization is heavy, repeated allocation is pure overhead and causes allocator contention + cache churn.

**Fix:** reuse factorization buffers, workspace arrays, and cached diagonal positions for static regularization.

### WK4 — Cloning the full KKT matrix for refinement
The code was cloning the entire CSC matrix just to keep it around for refinement residual evaluation.

**Fix:** store the matrix once and factorize from a reference.

### WK5 — SOC dense block assembly allocates per column
SOC “dense” blocks were creating new vectors for each column during KKT assembly.

**Fix:** reuse per-block scratch vectors.

### WK6 — Avoidable per-iteration RHS allocations (e.g., `-q`, `b.clone()`)
Small in isolation, but they show up in profiles and add noise.

**Fix:** precompute constant RHS (like `-q`) once per solve, and stop cloning `b`.

---

## Patch: `minix_wallclock_killers.patch`

This patch directly targets WK1–WK6. Concretely it:

### 1) Allocation-free solve + refinement workspace (WK2)
`KktSolver` now holds a `SolveWorkspace` with:
- `rhs_perm`, `sol_perm`, `kx`, `res`, `delta`

So repeated solves (and refinement iterations) do **no heap allocations**.

### 2) No KKT clone on factorization (WK4)
`factor()` now stores the built KKT matrix in `self.kkt_mat` once and calls QDLDL on a reference.

### 3) Diagonal-H fast path: update H diagonal in-place (WK1)
When all scaling blocks are `Zero` or `Diagonal` (typical LP/QP):
- the KKT sparsity is fixed
- only the diagonal of the `-H` block changes each iteration

The patch:
- caches the data-index positions of those diagonal entries
- updates them in-place each `factor()` call
- avoids TriMat→CSC rebuild for those problems

This is the single biggest wallclock win for the Maros–Mészáros LP/QP-style benchmarks.

### 4) QDLDL buffer reuse + cached diagonal positions (WK3)
`QdldlSolver` now:
- caches diagonal positions for fast “add static_reg to diagonal”
- reuses `a_x_work` (matrix values buffer)
- reuses `bwork/iwork/fwork` and factorization buffers across iterations

### 5) SOC assembly scratch reuse (WK5)
SOC structured block assembly now reuses scratch vectors per block instead of allocating per column.

### 6) Precompute `-q`, avoid `b.clone()` (WK6)
`solve_ipm` precomputes `neg_q = -q` once for the scaled problem and passes it into `predictor_corrector_step`.
`predictor_corrector_step` now uses `&prob.b` rather than cloning.

---

## What this patch does **not** solve yet

These are the next wallclock killers to tackle after this patch:

### Next WK — Multi-RHS triangular solves
Even with allocation-free solves, HSDE still performs multiple solves per iteration.
A major next step is true **multi-RHS solve** support in the factorization backend (solve 2–4 RHS with one forward/back sweep).

### Next WK — Supernodal / BLAS-backed sparse factorization
QDLDL is a great baseline, but Clarabel/MOSEK performance comes from:
- high-quality sparse factorizations (supernodal, cache-aware)
- tuned BLAS kernels and vectorization
- careful ordering and numerical pivot strategies

Consider:
- adding optional backends (CHOLMOD, KLU, Pardiso, etc.)
- or writing a dedicated supernodal LDLᵀ for quasi-definite KKT systems

### Next WK — Threading
For large problems:
- symbolic analysis is single-threaded
- numeric factorization and solves are single-threaded

Parallelism (carefully applied) can be decisive.

---

## How to validate (must-do)

1. **Flamegraph before/after** on:
   - DUAL2
   - a medium QP where SCS wins on wallclock
2. Confirm:
   - allocations/iter ≈ 0 in the hot path
   - KKT build time drops sharply on diagonal-H problems
3. Track separate timers:
   - cone scaling update
   - KKT numeric update
   - factorization
   - solves (affine + combined + dtau)
   - termination/residual computation

---

## Next convergence work (still required)

Even after winning wallclock, MINIX must also converge robustly to high accuracy:

- Fix endgame stalls where μ→0 but feasibility doesn’t improve.
- Improve step-to-boundary behavior for s/z-limited α collapse:
  - better scaling regularization near boundary
  - endgame heuristic to switch to “feasibility restoration” mode
  - smarter σ/centering when μ is tiny but residuals persist
  - consider switching to a **more robust stopping / fallback strategy** when complementarity is tiny

These belong in a convergence patch series (separate from wallclock engineering), to keep performance changes isolated and measurable.
