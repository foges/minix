# Minix improvements plan

_Last updated: 2026-01-03_

This document consolidates the current status, observed issues, and a concrete improvement roadmap for **Minix** with the goal of becoming a **top-tier convex conic optimization solver** and achieving a **meaningful wallclock speed lead** over **Clarabel** (and ultimately commercial-grade baselines such as **MOSEK**) on real workloads.

It is written to be **actionable**: each section identifies (1) what’s wrong, (2) why it matters for convergence or wallclock time, and (3) what changes to implement.

---

## 1. What “beating Clarabel/MOSEK” actually requires (wallclock, not iterations)

Minix is already showing the “IPM advantage” in **iteration count** on many problems, but is losing on **wallclock** on medium/large instances because its per-iteration costs are currently too high.

To get a meaningful lead over Clarabel/MOSEK on realistic workloads, Minix needs three things simultaneously:

1) **Correctness + robustness**  
   - Reliable convergence to tight feasibility + gap
   - Reliable infeasibility detection (HSDE certificates)
   - Termination criteria evaluated on *unscaled* quantities (so user tolerances mean what users think)

2) **A sparse KKT stack that is “production IPM-grade”**  
   - KKT assembly must be *pattern-reused* and *allocation-free* in the hot loop
   - Sparse LDLᵀ must reuse symbolic analysis and reuse numeric workspace
   - Accurate solves (iterative refinement aligned with the exact matrix being solved)
   - Regularization that stabilizes early iterations without destroying tail accuracy

3) **Systematic, measured optimization**  
   - Timing breakdowns (factorization/solve/cones) per benchmark
   - Profiling-driven work: remove allocations, remove repeated conversions, remove redundant solves

---

## 2. Current status

### 2.1 Algorithmic improvements already integrated (good progress)

Recent HSDE / predictor-corrector work is already in-tree (not exhaustive):

- μ_aff-based σ with clipping
- Combined step feasibility weighting (1−σ) with floor
- MCC “delta” correction wiring + alpha-stall diagnostics
- Defaults adjusted so MCC + line search are currently off (they hurt DUAL2 as implemented)

### 2.2 The main functional blocker: DUAL2 tail-stall

DUAL2 still hits `MaxIters`. Empirically:

- μ becomes extremely small (complementarity essentially solved)
- dual feasibility stalls
- step length collapses due to **s/z step-to-boundary** (alpha_sz tiny), not τ/κ

Interpretation: we are “centrally feasible” in complementarity but are not making progress in the feasibility residuals because the Newton direction is producing ds/dz components that force a microscopic fraction-to-boundary step.

---

## 3. Immediate correctness / robustness patches (included)

These are “must-fix” because they directly affect tail accuracy and termination correctness.

### 3.1 Iterative refinement matrix mismatch (fix included)

**Problem:** the factorization solves a *regularized* matrix, but refinement was computing residuals against the *unshifted* stored KKT matrix. That makes refinement far less effective (worst-case: actively misleading).

**Fix:** in refinement, add the static diagonal shift back into `Kx` before forming the residual.

### 3.2 Termination criteria are evaluated on scaled quantities (fix included)

The design doc requires termination on **unscaled** variables after undoing Ruiz scaling:

- x̄ = x/τ, s̄ = s/τ, z̄ = z/τ
- r_p = A x̄ + s̄ − b, r_d = P x̄ + Aᵀ z̄ + q
- objectives and gap on the **unscaled** problem

**Fix:** make `check_termination` take `RuizScaling`, unscale x̄/s̄/z̄, and evaluate residuals + gap per design doc §16.

### 3.3 Patch artifact

A patch-style diff implementing 3.1 and 3.2 is provided as:

- `minix_p0_fixes.patch`

(See the accompanying file next to this doc.)

---

## 4. Fixing the DUAL2 tail-stall (next convergence work)

The stall signature (“μ tiny, alpha_sz tiny, feasibility stuck”) is classic “bad direction near the boundary.”
In practice, there are a few high-probability causes:

### 4.1 KKT solve accuracy and regularization are distorting the direction

**What to do:**

- **Turn iterative refinement into a real weapon**
  - after the refinement-mismatch fix lands, sweep `kkt_refine_iters = 1, 2, 3, 5`
  - confirm whether DUAL2 tail-stall disappears (often it does if the direction was solve-error dominated)

- **Revisit static regularization schedule**
  - today Minix enforces a large minimum static reg in some cases
  - implement a *decaying* reg schedule tied to μ and factorization stability:
    - start with safe reg
    - if factorization succeeds for N iterations and μ is decreasing, gradually reduce reg
    - if pivot problems occur, bump reg again

### 4.2 MCC (multiple centrality correction) should be a tail fix, but currently hurts

MCC is exactly the tool to fix “alpha_sz collapses due to a few bad components,” but it must be done carefully:

**What to do:**
- Validate the MCC math against the design doc:
  - MCC should only perturb the complementarity equation (Δw) while preserving feasibility RHS
- Add unit tests on a toy LP/QP where a single component would otherwise limit alpha_sz
- Only then turn MCC back on by default

### 4.3 Add a central-neighborhood line search (not just positivity)

Right now, positivity alone can allow the iterate to drift into a region where the Newton direction becomes pathological.

Implement a neighborhood condition (Clarabel-style):
- enforce bounds on the scaled complementarity ratios
- shrink alpha until the neighborhood test passes

---

## 5. Wallclock performance plan (where the real speed is)

Your wallclock benchmark results are exactly what you expect from an IPM whose iteration loop is not yet optimized:
- very few iterations
- but expensive iterations due to KKT build + factorization + allocations

To beat Clarabel/MOSEK in wallclock, you need to treat the KKT path as *the product*.

### 5.1 Make the iteration loop allocation-free

**Target:** no `Vec` allocations inside the main iteration loop after warm start.

Concrete work:
- add an `IpmWorkspace` struct holding:
  - RHS buffers (aff/corr/comb)
  - dx/dz/ds scratch
  - cone work vectors
  - alpha diagnostics scratch
- plumb it through `predcorr_step`

### 5.2 Reuse the KKT sparsity pattern

Right now KKT assembly uses `TriMat` + `to_csc()` each iteration (heavy).

Instead:
- build a fixed CSC pattern once (symbolic stage)
- store an array of “value slots”
- each iteration, fill only the numeric values in-place
- reuse the factorization symbolic analysis across iterations

### 5.3 Upgrade the sparse LDL backend strategy

If the goal is “significantly faster than Clarabel,” you will almost certainly need either:
- a much faster Rust LDL (supernodal, multithreaded), or
- an optional external backend (SuiteSparse LDL/CHOLMOD, Pardiso, HSL, etc.)

Minix should have a trait for KKT factorization so you can swap implementations without touching IPM logic.

### 5.4 SOC scaling / KKT block assembly: remove dense materialization

Avoid forming dense H blocks for SOC in the KKT.
Exploit structured formulas so the SOC contribution is applied without O(d²) dense columns.

---

## 6. Benchmarking + profiling plan (so we don’t fly blind)

### 6.1 Required metrics (every run)

Report and log:
- total solve time
- time in KKT factorization
- time in triangular solves (all solves per iter)
- time in cone kernels
- allocations per iteration (debug build instrumentation)

### 6.2 Compare against the right baselines

- Clarabel (primary IPM baseline)
- MOSEK (commercial target)
- SCS (first-order baseline)
- ECOS (older IPM baseline)

Use:
- wallclock
- accuracy (unscaled residuals + gap at termination)
- robustness (failure rate)

---

## 7. Recommended next sequence (minimal risk, maximum signal)

1) Land the P0 patch (refinement shift + unscaled termination)
2) Re-run DUAL2 and the Maros-Mészáros slice with refinement sweeps
3) If DUAL2 still stalls:
   - debug MCC math and re-enable it
   - add neighborhood line search
4) Start the KKT rewrite: pattern reuse + no allocations
5) Only then chase “fastest possible LDL” (backend swap / supernodal / threading)
