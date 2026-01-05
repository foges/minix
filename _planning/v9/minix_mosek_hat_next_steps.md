# Minix: “What would MOSEK do” next steps (BOYD1 convergence + QGROW speed)

## 1) What your BOYD1 log is telling us

From the run you pasted (30 iterations, `BOYD1`, `n=93261`, `m=93279`):

- **Complementarity is basically solved**: `μ ≈ 2.39e-6` and `gap_rel ≈ 4.53e-6`.
- **Primal feasibility is solved**: `rel_p ≈ 1.23e-14`.
- **But dual feasibility stalls**: `rel_d ≈ 1e-3` (unscaled `r_d_inf ≈ 2.67e2`).

That combination (“μ tiny, primal tiny, dual stuck”) is a classic sign that:

1. **The Newton system is being solved accurately enough to follow the central path (μ↓)**,
2. but **not accurately enough to drive the stationarity residual down further**, or
3. some part of the model transformation (presolve / elimination / scaling) makes the *dual stationarity* the hardest piece to satisfy in finite precision.

MOSEK’s own docs explicitly note that PFEAS/DFEAS “should converge … but may stall at low level due to rounding errors”. Your stall level (`~1e-3`) is just too high to accept for a solver that wants to compete with MOSEK/Clarabel.

## 2) The two biggest levers MOSEK has that Minix still needs

### Lever A — More controlled scaling (equilibration clamps)
Ill-conditioned QPs like QGROW generally need **aggressive but bounded** row/column scaling.

Clarabel does this explicitly via `equilibrate_min_scaling` / `equilibrate_max_scaling` defaults (1e-4, 1e4), rather than “if norm < eps, skip scaling”.

Skipping scaling for “tiny norm rows/cols” often leaves the worst-conditioned parts unscaled, which:

- increases iteration count (slow), and
- increases numerical error in KKT solves (dual stall).

### Lever B — Polishing / crossover to reduce stationarity residual
A production solver will often do a **post-solve polishing step** when:

- primal feasibility is already very good,
- gap is good,
- but dual feasibility is lagging.

For bound-heavy problems (like BOYD1, where your KKT presolve says ~93k singleton rows), the most effective polish is **a “bound/active-set aware” dual recovery**:

- identify inactive bounds (slack > tol) and force their multipliers to 0,
- solve a small least-squares system for equality multipliers,
- recover bound multipliers from stationarity.

This is conceptually the same reason simplex crossover exists in QP solvers: it produces a numerically consistent dual.

## 3) Speed: why QGROW is 47× slower than Clarabel

Even if iteration counts were identical, you can lose a lot of time if your inner loop is doing any of:

- rebuilding a sparse CSC matrix from triplets every iteration,
- allocating / sorting / deduplicating repeatedly.

For diagonal-only scaling blocks (LP/QP with only Zero + Nonneg cones), you want:

- **fixed symbolic factorization once**,
- **in-place numeric updates** of only the changing diagonal entries, then
- numeric refactor.

That is the lowest-hanging “wallclock killer” relative to Clarabel.

## 4) Included patches

### Patch 1 — Ruiz scaling clamps (helps QGROW iteration count + dual stall)
File: `minix_ruiz_clamp_scaling.patch`

- Adds `RUIZ_MIN_SCALING=1e-4` and `RUIZ_MAX_SCALING=1e4`.
- Replaces the “if norm > 1e-12 else 1” logic with a **clamped inverse-sqrt scaling**.

### Patch 2 — Fast diagonal-only KKT updates (big speed win for QP/LP)
File: `minix_kkt_diag_cache.patch`

- Caches the positions of the **KKT diagonal entries for the slack block** when the scaling blocks are only `Zero` or `Diagonal`.
- In subsequent iterations, it updates only those diagonal entries in-place, and avoids rebuilding the sparse matrix.

### Patch 3 — Combined
File: `minix_mosek_hat_perf_and_scaling.patch`

- Just patch 1 + patch 2 merged.

## 5) What I would do next (beyond these patches)

These are the “MOSEK hat” next steps I’d prioritize for BOYD1 and QGROW:

1. **Add a targeted polish stage for singleton/bound-heavy constraints**
   - Trigger when `rel_p` and `gap_rel` are good but `rel_d` is not.
   - Use active-set identification based on slack.
   - Solve for equality multipliers (tiny system), then recover bound multipliers.

2. **Drop static KKT regularization closer to Clarabel’s defaults**
   - Clarabel’s defaults are on the order of `1e-8` for static regularization.
   - If Minix uses `1e-6` (or worse for “sparse QP path”), you will get a dual residual floor.

3. **Make refinement adaptive**
   - If `μ` is tiny but `rel_d` isn’t improving, increase iterative refinement iterations for KKT solves.

4. **(If you really need to beat Clarabel)** add a faster factorization backend
   - Supernodal LDL (SuiteSparse LDL / CHOLMOD or Pardiso) and keep symbolic factorization cached.

## 6) About `ipm` vs `ipm2`

Recommendation:

- **Keep `ipm2` as the only solver you actively evolve.**
- Keep `ipm` temporarily as a fallback behind a feature flag (or compile-time option) until `ipm2` is stable.
- Do **not** delete `ipm` until `ipm2`:
  - passes the full benchmark set,
  - matches or exceeds accuracy,
  - matches or exceeds speed.

Once that’s true, delete `ipm` (or move it to `legacy/` or a separate crate). Deleting too early will slow you down during regression debugging.

---

## Appendix: quick “success criteria” for the next BOYD1 attempt

After these patches + a polish stage, BOYD1 should look like:

- `rel_p <= 1e-8`
- `rel_d <= 1e-8`
- `gap_rel <= 1e-8`

and the final few iterations should show `DFEAS` falling rather than stalling.

## Appendix: skeleton for “singleton / bound dual recovery” polish

This is the simplest polishing step that’s *specifically* effective for BOYD1-style models where:

- almost all inequality rows are singleton rows, and
- the nonneg cone is huge.

High-level:

1. Split constraints into `A_eq x = b_eq` (Zero cone) and singleton inequality rows `a_i x_{j(i)} + s_i = b_i`.
2. Mark a singleton row **inactive** if its slack is comfortably positive (e.g. `s_i > s_tol`).
3. For inactive rows, enforce `z_i = 0` and use the stationarity equations on those coordinates to estimate the equality dual `y`:

   `P x + q + A_eq^T y ≈ 0` on inactive coordinates.

4. With that `y`, recover singleton duals:

   `z_i = max(0, -(grad_{j(i)} + (A_eq^T y)_{j(i)}) / a_i)`

This gives you dual variables that are **algebraically consistent** with the primal point `x`.

Pseudo-code sketch (not drop-in, but close):

```rust
// Inputs:
// - x: primal (already unscaled)
// - s: slack for cone rows (unscaled)
// - problem: has P, q, A_eq
// - singleton_rows: Vec<(row_index_in_A, col_j, a_ij)>
// - eq_rows: 0..m_eq

fn polish_singleton_duals(...) -> (y_eq, z_single) {
    // 1) Compute grad = P x + q
    let mut grad = vec![0.0; n];
    // grad += P * x
    // grad += q

    // 2) Build normal equations for y via inactive set
    let mut M = DMatrix::<f64>::zeros(m_eq, m_eq);
    let mut rhs = DVector::<f64>::zeros(m_eq);

    for (row, col_j, a) in singleton_rows.iter().copied() {
        let s_i = s[row];
        if s_i > s_tol {
            // inactive -> z_i = 0, so enforce (A_eq^T y)_j = -grad_j
            // column j of A_eq gives equation c_j^T y = -grad_j
            let c_j = a_eq_column(j); // sparse vector length m_eq
            // M += c_j * c_j^T
            // rhs += c_j * (-grad_j)
        }
    }

    // regularize M a bit in case the inactive set is rank-deficient
    for k in 0..m_eq { M[(k,k)] += 1e-12; }

    let y = M.cholesky().unwrap().solve(&rhs);

    // 3) Compute aeqty = A_eq^T y
    let aeqty = compute_a_eq_transpose_times_y(y);

    // 4) Recover singleton z
    let mut z = vec![0.0; m_total];
    for (row, col_j, a) in singleton_rows.iter().copied() {
        let s_i = s[row];
        if s_i > s_tol {
            z[row] = 0.0;
        } else {
            z[row] = (-(grad[col_j] + aeqty[col_j]) / a).max(0.0);
        }
    }

    (y, z)
}
```

Pragmatic trigger:

- Only run this when `rel_p` and `gap_rel` are already within tolerance (or close), and `rel_d` is not.
- It should be *cheap* here because the equality system is only `m_eq × m_eq` (18×18 for BOYD1).

