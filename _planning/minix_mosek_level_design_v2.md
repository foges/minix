# Minix → MOSEK‑close Conic IPM: Engineering Design Doc (v2)

_Last updated: 2026-01-03_

This document is a **concrete engineering plan** to take the current Minix solver from “works on many instances” to **MOSEK‑close** behavior on continuous conic problems.

It is explicitly based on the same algorithm class as MOSEK’s conic interior-point optimizer (homogeneous/self-dual primal–dual IPM), and targets the same “production properties”: robust certificates, presolve+scaling, high-quality sparse KKT solves, and deterministic performance. citeturn0search0turn0search1turn0search11

---

## 0. What “MOSEK‑close” means (so this is falsifiable)

MOSEK has decades of tuning, presolve, and advanced linear algebra. “Matching MOSEK everywhere” is not a realistic near-term objective.

**Definition we will use:**

### 0.1 MOSEK‑close for LP/QP/SOCP (first milestone)
On a curated benchmark set representative of your workloads (e.g., Maros–Mészáros QP, Mittelmann SOCP subsets, CVXPY workloads):

- **Robustness:** Minix returns the same status class as MOSEK (Optimal / PrimalInf / DualInf) for ≥ 99% of instances where MOSEK returns a status.
- **Accuracy:** Unscaled feasibility + gap meet tolerances at termination (default 1e‑8 or tighter).
- **Wallclock:** Geometric-mean wallclock within **2×** of MOSEK on that set (single-thread, comparable tolerances), and within **1.2×** on small/medium cases where QDLDL-class methods should compete.

### 0.2 MOSEK‑close for EXP/POW/PSD (second milestone)
Once symmetric cones are stable and fast:

- Implement EXP/POW with correct nonsymmetric scaling + third-order correction.
- Implement PSD + chordal decomposition for sparse PSD.
- Achieve robustness and “within 2–3×” wallclock on cone-heavy suites (this is a harder target).

---

## 1. Start fresh or build on current code?

**Recommendation: build on the existing codebase.**  
You already have:
- HSDE/IPM iteration structure
- cone kernel framework
- KKT solver abstraction and factorization backends
- Python/CVXPY integration plumbing

**But**: the refactors needed (workspace, KKT caching, backend trait) are large. Do them behind a parallel path:

- [DONE] Create `solver-core/src/ipm2/` and route solves through it via a feature flag or settings knob.
- Keep the current solver as baseline until `ipm2` matches correctness and beats it on wallclock.

This prevents a multi-week “broken main branch” situation.

---

## 2. Reality check: why BOYD1/BOYD2 are the right forcing functions

### 2.1 BOYD1 failure mode (“polish blocked”)
- μ + gap collapse and primal feasibility is excellent
- dual residual stalls above tolerance
- solver spends huge wallclock to hit MaxIters

This is almost always: **linear solve accuracy / regularization discipline / end-game mode missing**.

### 2.2 BOYD2 failure mode (“structure punished”)
- huge bound/row structure → enormous KKT
- conditioning is brutal
- iterations may be fine but wallclock explodes

This is almost always: **presolve + structure elimination missing**.

---

## 3. Design principles (non-negotiables)

### 3.1 All termination and reported metrics are unscaled
Compute `r_p`, `r_d`, objectives and gap **after undoing Ruiz scaling**. (User tolerances must mean what they think they mean.)

### 3.2 No ad-hoc Newton system perturbations in the hot loop
Stability should come from:
- small static regularization (scale-aware)
- dynamic pivot protection
- iterative refinement against the *actual solved matrix*
- explicit stall recovery and polish modes

### 3.3 Allocation-free iteration loop
After initialization:
- no allocations in `predcorr`, `termination`, KKT solve (including refinement), or cone kernels.

### 3.4 KKT structure reuse
- CSC sparsity pattern built once
- symbolic factorization reused
- only numeric `values` updated per iteration

---

## 4. Workstream A: Linear solve quality (highest leverage)

### A1 — Fix iterative refinement matrix mismatch
Ensure residuals are computed against the *same* shifted matrix factorized (e.g., include `static_reg` in matvec).

### A2 — Centralize regularization policy
Implement `RegularizationPolicy` that produces:
- `static_reg_eff` (scale-aware)
- `dynamic_min_pivot` rules
- adaptive rules for polish mode (lower static reg late, raise refinement)

### A3 — Add solve accuracy telemetry
Per iteration (diagnostics):
- `||Kx-b||∞` for both RHS solves (post-refinement)
- effective regularization parameters
- dynamic pivot bump count

### A4 — Add an end-game polish mode
Trigger when:
- μ is already tiny but feasibility/gap not met,
and respond with:
- increased refinement iters
- RHS mode switch
- (optionally) reduced static reg (if stable)

---

## 5. Workstream B: Presolve + postsolve (MOSEK-grade requirement)

MOSEK does presolve and scaling; “MOSEK-close” requires you to do meaningful reductions too. citeturn0search1turn0search11

### B1 — Transformation framework
Implement a `PresolveTransform` stack:
- each transform records enough info to undo in postsolve
- postsolve reconstructs original x/s/z and certificates

### B2 — Fixed variable elimination
- detect `l == u`
- substitute out variable and update b/q/constants
- shrink matrices

### B3 — Bounds handling without “bounds as rows” blowup
Prefer:
- variable shifts `x = t + l` for lower bounds
- encode only remaining one-sided bounds as cone rows (if needed)

### B4 — Singleton/identity-row elimination
Detect rows with exactly one nonzero (common for bounds and simple equalities) and eliminate them into diagonal updates (Schur complement), shrinking KKT dimension dramatically.

---

## 6. Workstream C: Step logic, stall recovery, end-game correctness

### C1 — Combined-step RHS weighting `(1-σ)`
Expose a knob:
- `rhs_mode = Full` vs `ScaledByOneMinusSigma`
Default to `(1-σ)` with a small floor (e.g., `max(1-σ, 0.05)`).

### C2 — Alpha-stall detector and mode switch
If:
- `alpha < 1e-6` repeatedly, or
- residual improvement stalls while μ is small,
switch to:
- high σ (centering)
- RHS `(1-σ)` mode
- increased refinement
- potentially polish mode

---

## 7. Workstream D: Wallclock performance engineering

### D1 — `IpmWorkspace` and allocation-free predcorr/termination
Preallocate:
- RHS vectors, direction vectors, scratch buffers, termination buffers.

### D2 — Persistent solve workspace inside KKT solver
Preallocate:
- permuted RHS/solution buffers
- refinement scratch vectors

### D3 — Reuse CSC pattern; numeric updates in-place
Build KKT sparsity once and update `values` only.

### D4 — Real multi-RHS solve
Permute once, solve twice, unpermute once.

### D5 — SOC scaling: eliminate dense materialization where possible
Short-term: preallocate and write dense blocks into fixed positions without per-column allocations.
Long-term: structured SOC scaling representation that avoids dense insertion entirely.

---

## 8. Workstream G: Linear solver backends + ordering + determinism

MOSEK’s performance comes heavily from advanced sparse direct solvers, presolve, and hardware/parallel exploitation. citeturn0search11

### G1 — Backend trait
Define `LinearSolverBackend` so you can plug:
- QDLDL baseline (pure Rust)
- CHOLMOD / Pardiso / etc. (optional, depending on licensing and distribution)

### G2 — Workspace reuse + diagonal shift application
Even QDLDL must be engineered:
- reuse factor buffers across numeric factorizations
- apply shifts without copying matrices

### G3 — Ordering improvements
Expose AMD / METIS selection and keep deterministic tie-break rules.

### G4 — Deterministic parallelism strategy
- If you parallelize cone blocks/residuals, ensure deterministic chunking and stable reduction order.
- Backend factorization parallelism is backend-dependent; document determinism guarantees.

---

## 9. Workstream E: Bench gates and regression discipline

### E1 — Correctness harness
- curated suite with known hard cases (BOYD1/BOYD2/DUAL2 + cone coverage)
- assert unscaled termination metrics

### E2 — Performance gates
Track:
- total time
- KKT update time
- factor time
- solve time
- cone time
- allocations (where measurable)

---

## 10. Can this get us MOSEK-close?

**Yes, for LP/QP/SOCP**, this plan is the right path to get **close**—provided you also do:
- at least one “serious” sparse direct backend OR extremely optimized QDLDL path, and
- real presolve reductions (bounds/fixed vars/singleton elimination), and
- polish/stall modes to avoid end-game spin.

Those items are explicitly included above.

**For full MOSEK parity across EXP/POW/PSD at scale**, this plan is still the right architecture, but expect:
- more kernel complexity (nonsymmetric scaling/oracles/corrections),
- more presolve (especially for SDP),
- and stronger linear algebra backends.

This v2 doc now includes those as explicit workstreams so “MOSEK-close” is not hand-wavy.

---

## 11. Appendix: concrete task list (high ROI order)

1) A1, A3 (solve accuracy correctness + telemetry)  
2) A2, A4 (reg policy + polish mode)  
3) C1, C2 (RHS weighting + stall recovery)  
4) D1–D4 (allocation-free + KKT cache + multi-RHS solve)  
5) B1–B4 (presolve framework + structural elimination)  
6) G1–G4 (backend trait + ordering + determinism/parallelism)  
7) Cones beyond symmetric (EXP/POW/PSD) once symmetric cones are stable/fast

