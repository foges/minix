# BOYD1 still MaxIters + QGROW vs Clarabel: next moves ("MOSEK hat")

This is an actionable debugging + performance plan based on your most recent BOYD1 trace and the benchmark deltas vs Clarabel.

---

## 1) BOYD1: what your trace really says

At iter 30 you reported:

- `rel_p = 3.833e-14`
- `gap_rel = 4.646e-7` and `μ = 4.326e-8`
- `rel_d = 8.399e-4` with `r_d_inf ≈ 2.236e2` and `dual_scale ≈ 2.662e5`

Interpretation:

- You are already essentially **optimal in objective + primal feasibility**.
- You are in the classic IPM endgame where only **dual feasibility** remains.
- The dual residual is no longer improving because it has hit a numerical floor.

This is *exactly* the regime where mature commercial solvers switch tactics (and/or relax termination) because the barrier KKT becomes extremely ill-conditioned.

### Why this happens (in one paragraph)

As `μ → 0`, inactive nonneg/bound constraints have `z_i → 0` while `s_i` stays positive. For NonNeg NT scaling, the KKT diagonal uses `H_i = s_i / z_i`, which can become enormous. Even with singleton elimination (which removes bound rows from the KKT), the remaining system is still affected by extreme diagonal updates. At that point, factorization quality, ordering, and iterative refinement determine how far `r_d` can be pushed. If you also have any static regularization that is not scaled correctly, you get a hard floor.

---

## 2) What MOSEK would do here (concretely)

### (A) Active-set / basis identification polish

MOSEK has a notion of “basis identification / crossover” for many problems: once the barrier iterate is very close, it tries to identify which inequalities are active and solves a reduced equality-constrained problem to obtain high-quality multipliers. For NonNeg problems, this is a straightforward active-set QP solve.

**What to implement in Minix:**

1. Detect likely active inequalities:
   - active if `s_i` is tiny or `z_i` is relatively large.
   - cap the number of actives (keep the biggest `z_i` if needed).
2. Solve the equality-constrained QP with constraints = original equalities + detected actives.
3. If any active multipliers come out negative, drop them and re-solve (2–3 passes).
4. Reconstruct full `(s,z)`:
   - active: `s_i = 0`, `z_i = y_i` (>=0)
   - inactive: `z_i = 0`, `s_i = b_i - A_i x` (clamp to >=0)

This bypasses the nasty `H = s/z` endgame and often drops the dual residual by orders of magnitude.

### (B) Enter polish earlier (scale it with requested accuracy)

A fixed `polish_mu_thresh = 1e-10` is brittle. Tie it to the user’s requested gap tolerance:

- `polish_mu_thresh = max(1e-12, 100 * tol_gap)`

This starts polish when the barrier has already done its job.

### (C) Stronger linear algebra (speed + accuracy)

To compete with Clarabel/MOSEK on ill-conditioned problems you eventually need:

- a high-quality sparse LDL backend (SuiteSparse LDL / CHOLMOD style)
- stable ordering (CAMD/AMD) and robust handling of fill-in explosions
- more effective refinement (and diagnostics on KKT residuals)

Even if the algorithmic steps are correct, a weaker factorization backend will manifest as both:

- slower per-iteration time, and
- dual-feasibility floors.

### (D) MOSEK-style result semantics

Commercial solvers tend to distinguish:

- OPTIMAL
- NEAR_OPTIMAL
- STALLED_NEAR_OPTIMUM

Minix currently returns MaxIters even when the solution is very high quality. For users this is confusing, and for benchmarking it makes you look worse than you are.

Recommendation:

- Keep strict `Optimal` check.
- Add a “near-opt” classification when:
  - primal OK and gap OK,
  - dual is within a configurable relaxed tolerance,
  - and progress has stalled.

You can still report the true residuals in `SolveInfo`.

---

## 3) QGROW: why Clarabel is 47× faster and how to close the gap

Your table:

- QGROW22: minix 3408ms vs clarabel 73ms

This kind of gap usually means one (or both) of:

1) **Minix takes far more IPM iterations** (endgame stalls, bad centering/regularization/scaling).
2) **Per-iteration KKT cost is far worse** (ordering/fill-in/backend), so even similar iters lose.

### What to do next (in order)

1. Add a one-line summary log per problem:
   - iters, factorization time, solve time, nnz(L), whether CAMD perm used, #dynamic bumps, max(H_i) and min(H_i).

2. Run QGROW with:
   - current backend
   - increased `kkt_refine_iters`
   - smaller static reg in polish
   - active-set polish enabled

3. If iters are the issue:
   - increase scaling effort (more Ruiz iters, or adaptive until row/col norms stabilize)
   - revisit sigma/clipping policies specifically for ill-conditioned QPs
   - introduce solve-form selection (primal vs dual) like MOSEK’s “dualizer”

4. If per-iter is the issue:
   - you need a stronger sparse LDL backend (SuiteSparse) and/or better ordering fallback

---

## 4) ipm vs ipm2: can we delete one?

Yes, but not instantly.

Best practice:

1. Route the default `solver_core::solve` entry point to **ipm2**.
2. Keep ipm1 as an explicit A/B option in solver-bench.
3. Once ipm2 is strictly better across your regression suite, delete ipm1 (or leave a thin compatibility wrapper for one release).

This prevents two solver implementations from diverging.

---

## 5) What the patch in this bundle contains

The patch file included alongside this note does:

- Adds `solver-core/src/ipm2/polish.rs` implementing NonNeg active-set polish (Zero + NonNeg cones only).
- Hooks polish into `solve_ipm2`:
  - recompute metrics on the recovered/original problem (with explicit bounds)
  - run polish only when primal+gap OK but dual not
  - upgrade status to Optimal if polish reaches tolerances
- Makes polish trigger threshold scale with `tol_gap`.
- Routes `solver_core::solve` to ipm2 by default.
- Keeps the solver-bench ability to run ipm1 explicitly, but defaults to ipm2.

---

## 6) Task list (short)

1. Land the active-set polish + earlier polish trigger (this patch).
2. Add KKT quality instrumentation (nnz(L), perm used, residual of KKT solve).
3. Add a SuiteSparse LDL backend toggle and benchmark BOYD1/QGROW.
4. Add solve-form selection heuristic (primal vs dual) and benchmark QGROW.
5. Add NearOptimal/Stalled status (don’t hide residuals; just classify better).
