# Clarabel comparison: what MINIX can learn

This note distills *actionable* lessons from CLARABEL’s behavior (fast + consistent) versus MINIX’s behavior (often very fast, but with rare catastrophic outliers that dominate total wallclock).

It’s written to help you close two gaps:

1. **Robustness gap** (avoid the rare “DUAL* / HS*” stalls that dominate totals)
2. **Engineering gap** (reduce per-solve overhead so IPM iterations aren’t “expensive by construction”)

---

## 1) Interpreting your benchmark: why MINIX can “win” many but lose total time

Your results (MINIX best geometric mean, worst total time) strongly suggest a **heavy‑tail runtime distribution**:

- **Geometric mean** ≈ “typical” solve time.
- **Total time** ≈ “tail risk” (a few bad instances dominate).

This is a *classic* solver-development pattern: early iterations of an algorithm are fine, but a few instances hit:
- max iterations,
- very small step sizes (α → 0),
- repeated refactorizations with little progress,
- or numerical issues (KKT conditioning / regularization / scaling).

**Implication:** If you want “MOSEK-like” behavior, focus first on eliminating the tail. A 10–50× slowdown on 1–2 problems will erase dozens of small wins.

---

## 2) What CLARABEL is doing that explains “consistency”

### 2.1 Data equilibration is default and well-bounded

CLARABEL performs Ruiz equilibration **on all matrix-valued data**, and exposes bounds like min/max scaling and max equilibration iterations.

Even if MINIX has Ruiz, *the details matter*:
- Min/max scaling caps prevent extreme scaling that can create near‑singular KKT blocks.
- Consistent unscaling rules are critical (especially for dual feasibility norms).

**MINIX check:** ensure your Ruiz step has *both*:
- sensible caps (e.g., 1e-4..1e4), and
- a cost scaling strategy that doesn’t distort termination checks.

### 2.2 Initial iterate strategy (very likely a big reason for tail robustness)

CLARABEL describes multiple initialization strategies; for symmetric cones it follows CVXOPT-style ideas to start “near the central path” rather than just “inside the cone”.

A modern IPM often does something like:
1) Solve an auxiliary KKT system to make equality residuals small
2) Set s and z from that solution
3) Shift s and z into the cone interior by adding a multiple of the cone identity element

**Why this matters for your tail:** When you initialize with x=0, z=0 for equality rows, etc., you can start *far* from the central path on certain structured instances (like Maros/Mészáros DUAL*). Then the algorithm may:
- drive complementarity down “too early” in some components,
- slam into the s/z boundary,
- and lose the ability to reduce feasibility residuals without taking α≈0 steps.

### 2.3 Linear solver stabilization: static + dynamic regularization

CLARABEL emphasizes:
- **static regularization** to keep the system quasi-definite
- **dynamic regularization** that *bounds pivots away from 0* (i.e., stabilizes LDL)

This is exactly the regime where MINIX shows issues: μ becomes tiny, but feasibility residuals stall and α collapses.

**Key idea:** near convergence, the Newton system becomes numerically delicate; if directions are even slightly wrong, they can “point out of the cone” on tiny slack/dual entries, forcing α→0.

### 2.4 Allocation-free / low-overhead solver loop

CLARABEL explicitly targets a structure where, once the sparsity pattern is set:
- symbolic factorization is reused
- the solver can avoid repeated allocation / assembly overhead
- KKT solves are highly streamlined

This matters for wallclock: even if iteration count is low, if each iteration does a full sparse rebuild + new allocations, MINIX will look slow versus a tuned solver.

---

## 3) Concrete lessons to port into MINIX

### 3.1 Make MINIX’s “bad tail” the primary KPI (not mean/median)

Track at least these metrics:
- shifted geometric mean (reduces sensitivity to tiny instances)
- p95 / p99 wallclock
- performance profile (Dolan–Moré style)
- “tail count”: how many problems exceed 5× best solver time

This makes regressions obvious.

### 3.2 Implement CLARABEL/QOCO-style initialization for symmetric cones

Add a new initialization mode for symmetric cones (Zero/NonNeg/SOC/PSD) that:
1) Solves a KKT system (or least-squares + equality-constrained QP) to reduce residuals
2) Forms s, z from that solution
3) Applies a **cone shift** so s and z are *comfortably* interior, not barely interior

Then benchmark the DUAL* set again. This is one of the highest-leverage “robustness” changes.

### 3.3 Iterative refinement on KKT solves (small code, huge robustness)

When μ is small, do:
1) solve K d = rhs
2) compute r = rhs − K d
3) solve K δ = r
4) d ← d + δ

Even 1–2 refinement steps can dramatically reduce wrong-sign components in ds/dz that cause α to collapse.

### 3.4 Dynamic regularization must be applied “in the factorization”, not as an afterthought

If your LDL backend “bumps” pivots only after computing L, the resulting solve corresponds to a different matrix than the one factored, and can produce bad Newton directions.

You want pivot bounding applied where it affects elimination and L, not just Dinv.

(If this is already fixed in your updated branch, great — but it’s the first thing to double-check.)

### 3.5 Stop conditions: add a “reduced tolerance” regime

CLARABEL exposes both full and reduced tolerances.

For production, this matters because:
- the last digits of feasibility can be expensive,
- and some instances are essentially solved but fight numerical noise.

Add:
- `reduced_tol_feas`, `reduced_tol_gap`, etc.
- a rule: if progress stalls and μ is already tiny, accept reduced tolerances.

This is a practical way to kill outliers without harming typical performance.

### 3.6 Engineering for wallclock: don’t rebuild the KKT sparse structure every iter

If MINIX currently rebuilds CSC from triplets each iteration:
- prebuild the sparsity pattern once
- keep `(indptr, indices)` fixed
- update only numerical values each iteration (mostly the −H block)

This alone can easily cut wallclock by multiples on medium problems.

### 3.7 Expose and tune “max_step_fraction” and line-search in a principled way

CLARABEL defaults max step fraction ≈ 0.99.

MINIX’s “α stall” diagnostics are telling you the step-to-boundary constraint is the limiter.
That doesn’t mean line search is bad; it may mean:
- the merit function is poorly scaled,
- the acceptance criterion fights the cone step,
- or it’s only useful for nonsymmetric cones.

Make line search optional per cone type, and default it only where it helps.

---

## 4) A prioritized roadmap, based on “kill the tail first”

### P0 (days): robustness quick wins
- Add iterative refinement on KKT solve
- Add reduced-tolerance early exit when μ is tiny + residual stalls
- Add “restart / recenter” logic when α collapses for multiple iterations

### P1 (1–2 weeks): match CLARABEL initialization and stabilization
- Implement symmetric-cone initialization via auxiliary KKT solve + cone shift
- Ensure dynamic pivot bounding is applied correctly in LDL
- Tighten/validate unscaling and stopping norms (scaled vs unscaled consistency)

### P2 (weeks–months): close the engineering/performance gap
- Preassemble KKT pattern and update values in-place
- Replace/augment LDL backend (supernodal LDL or SuiteSparse-style factorization)
- Multi-threaded factorization (if sizes warrant)
- Optional mixed precision / GPU path (CuClarabel shows this can help, but it’s a big project)

---

## 5) What to look for in the next benchmark after these changes

If you adopt the above, success should look like:
- MINIX still best or near-best geometric mean
- total time improves sharply (tail cases stop dominating)
- “bad” instances no longer hit max iterations
- α does not collapse to machine zero late in solve
- feasibility residuals continue decreasing when μ is already tiny

---

## 6) References worth reading (for implementation details)

- Clarabel paper (arXiv:2405.12762) — initialization, regularization, KKT structure, scaling, performance evaluation
- Clarabel documentation — settings and safeguard defaults (max step fraction, equilibration bounds, reduced tolerances)
- QOCO paper (arXiv:2503.12658) — clear description of predictor–corrector, σ computation, step fraction, initialization, and performance profiles
- CuClarabel (GPU) paper — for long-term parallelism / mixed precision ideas

