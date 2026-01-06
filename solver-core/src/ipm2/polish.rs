//! Active-set polishing utilities.
//!
//! These are **optional** post-processing steps aimed at the classic IPM
//! endgame failure mode on large NonNeg blocks:
//!
//! - μ is tiny and the primal residual is excellent
//! - but the solver cannot reduce the (unscaled) dual residual further because
//!   the KKT system becomes extremely ill-conditioned (H = diag(s/z) spans many
//!   orders of magnitude).
//!
//! What MOSEK (and many production IPM solvers) do in this regime is a form of
//! **crossover / polishing**:
//!
//! 1. Identify a candidate active set (constraints with small slack or large
//!    multipliers).
//! 2. Solve an equality-constrained QP using only those constraints as
//!    equalities.
//! 3. Drop any constraints whose multiplier comes out negative (since NonNeg
//!    dual multipliers must be >= 0), and resolve.
//!
//! This file implements a conservative version of that idea for problems that
//! contain **only Zero + NonNeg cones** (including bounds that were converted to
//! NonNeg rows).

use crate::linalg::kkt::KktSolver;
use crate::linalg::sparse;
use crate::problem::{ConeSpec, ProblemData, SolverSettings};
use crate::scaling::ScalingBlock;

#[derive(Debug, Clone)]
pub struct PolishResult {
    pub x: Vec<f64>,
    pub s: Vec<f64>,
    pub z: Vec<f64>,
}

#[inline]
fn inf_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
}

/// Attempt an active-set polish for Zero + NonNeg problems.
///
/// Returns `Some(PolishResult)` on success, `None` if:
/// - the cone set includes anything other than Zero/NonNeg
/// - the active-set construction is empty
/// - the KKT solve fails (numerical issues)
pub fn polish_nonneg_active_set(
    prob: &ProblemData,
    x0: &[f64],
    s0: &[f64],
    z0: &[f64],
    settings: &SolverSettings,
) -> Option<PolishResult> {
    let diag_enabled = std::env::var("MINIX_DIAGNOSTICS").is_ok();
    let n = prob.num_vars();
    let m = prob.num_constraints();
    if x0.len() != n || s0.len() != m || z0.len() != m {
        if diag_enabled {
            eprintln!("polish: dimension mismatch: x0={} vs n={}, s0={} vs m={}, z0={} vs m={}",
                x0.len(), n, s0.len(), m, z0.len(), m);
        }
        return None;
    }

    // Only handle Zero + NonNeg for now.
    if prob
        .cones
        .iter()
        .any(|c| !matches!(c, ConeSpec::Zero { .. } | ConeSpec::NonNeg { .. }))
    {
        if diag_enabled {
            eprintln!("polish: unsupported cone types, found: {:?}",
                prob.cones.iter().filter(|c| !matches!(c, ConeSpec::Zero { .. } | ConeSpec::NonNeg { .. })).collect::<Vec<_>>());
        }
        return None;
    }

    // Collect equality rows (Zero) and inequality rows (NonNeg).
    let mut eq_rows = Vec::new();
    let mut ineq_rows = Vec::new();
    let mut offset = 0usize;
    for cone in &prob.cones {
        match *cone {
            ConeSpec::Zero { dim } => {
                eq_rows.extend(offset..offset + dim);
                offset += dim;
            }
            ConeSpec::NonNeg { dim } => {
                ineq_rows.extend(offset..offset + dim);
                offset += dim;
            }
            _ => unreachable!(),
        }
    }
    debug_assert_eq!(offset, m);

    if ineq_rows.is_empty() {
        if diag_enabled {
            eprintln!("polish: no inequality rows");
        }
        return None;
    }

    if diag_enabled {
        eprintln!("polish: eq_rows={} ineq_rows={}", eq_rows.len(), ineq_rows.len());
    }

    // Conservative thresholds based on current magnitudes.
    // Be very conservative - only select constraints that are DEFINITELY active.
    // A constraint is active if: s is very small AND z is positive (indicating binding).
    let s_norm = inf_norm(s0).max(1.0);
    let z_norm = inf_norm(z0).max(1.0);
    // Much more conservative: require s < 1e-4 * ||s|| AND z > 0
    let s_thresh = 1e-4 * s_norm;

    // Candidate active set: small slack AND positive multiplier.
    let mut active: Vec<usize> = ineq_rows
        .iter()
        .copied()
        .filter(|&i| s0[i].abs() <= s_thresh && z0[i] > 1e-10 * z_norm)
        .collect();

    if active.is_empty() {
        if diag_enabled {
            eprintln!("polish: no active constraints (s_thresh={:.3e})", s_thresh);
        }
        return None;
    }

    if diag_enabled {
        eprintln!("polish: candidate active set size={}", active.len());
    }

    // Cap active set size: a huge active set makes the polish KKT ill-posed.
    // Keep the most "active" constraints by multiplier magnitude.
    let max_active = n.max(64);
    if active.len() > max_active {
        if diag_enabled {
            eprintln!("polish: capping active set from {} to {}", active.len(), max_active);
        }
        active.sort_by(|&a, &b| z0[b].abs().partial_cmp(&z0[a].abs()).unwrap());
        active.truncate(max_active);
    }

    // Iterative pruning: if a constraint comes out with a negative multiplier,
    // drop it and re-solve.
    let mut active_set = active;
    let max_passes = 3usize;
    // Negative multiplier tolerance: a constraint shouldn't be in active set if z < 0
    // Use a small relative tolerance.
    let neg_mult_tol = -1e-8 * z_norm;

    for pass in 0..max_passes {
        let (a_eq, b_eq, row_ids) = build_equality_system(prob, &eq_rows, &active_set);
        let m_eq = row_ids.len();
        if m_eq == 0 {
            if diag_enabled {
                eprintln!("polish pass {}: empty equality system", pass);
            }
            return None;
        }

        if diag_enabled {
            eprintln!("polish pass {}: m_eq={} (eq_rows={} active={})", pass, m_eq, eq_rows.len(), active_set.len());
        }

        // Solve the equality-QP KKT:
        //   P x + A_eq^T y + q = 0
        //   A_eq x = b_eq
        // using the standard KKT form with H=0 and quasi-definite regularization.
        // Use very small regularization for polish to get accurate constraint satisfaction.
        let h_blocks = vec![ScalingBlock::Zero { dim: m_eq }];
        let polish_static_reg = 1e-12;  // Much smaller than normal solver
        let mut kkt = KktSolver::new(n, m_eq, polish_static_reg, settings.dynamic_reg_min_pivot);
        if kkt.initialize(prob.P.as_ref(), &a_eq, &h_blocks).is_err() {
            if diag_enabled {
                eprintln!("polish pass {}: KKT initialize failed", pass);
            }
            return None;
        }
        if kkt.update_numeric(prob.P.as_ref(), &a_eq, &h_blocks).is_err() {
            if diag_enabled {
                eprintln!("polish pass {}: KKT update_numeric failed", pass);
            }
            return None;
        }
        let factor = match kkt.factorize() {
            Ok(f) => f,
            Err(e) => {
                if diag_enabled {
                    eprintln!("polish pass {}: KKT factorize failed: {:?}", pass, e);
                }
                return None;
            }
        };

        let rhs_x: Vec<f64> = prob.q.iter().map(|&v| -v).collect();
        let rhs_z = b_eq;
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; m_eq];
        kkt.solve_refined(
            &factor,
            &rhs_x,
            &rhs_z,
            &mut x,
            &mut y,
            settings.kkt_refine_iters.max(4),
        );

        // Identify any "active" NonNeg constraints with negative multipliers.
        // Those should not be treated as active; drop and try again.
        let mut dropped_any = false;
        if !active_set.is_empty() {
            let active_offset = eq_rows.len();
            let mut new_active = Vec::with_capacity(active_set.len());
            for (k, &row) in active_set.iter().enumerate() {
                let mult = y[active_offset + k];
                if mult < neg_mult_tol {
                    dropped_any = true;
                } else {
                    new_active.push(row);
                }
            }
            if dropped_any {
                active_set = new_active;
                if active_set.is_empty() {
                    return None;
                }
                continue;
            }
        }

        // Reconstruct full (s,z).
        let mut z = vec![0.0; m];
        // Equality duals are free.
        for (k, &row) in eq_rows.iter().enumerate() {
            z[row] = y[k];
        }
        // Active inequality duals must be >= 0.
        for (k, &row) in active_set.iter().enumerate() {
            z[row] = y[eq_rows.len() + k].max(0.0);
        }

        let mut s = compute_slack(prob, &x);
        // Enforce s=0 on equality rows and active rows.
        for &row in &eq_rows {
            s[row] = 0.0;
        }
        for &row in &active_set {
            s[row] = 0.0;
        }
        // Project remaining NonNeg slacks to >= 0.
        for &row in &ineq_rows {
            if s[row] < 0.0 {
                s[row] = 0.0;
            }
        }

        if diag_enabled {
            eprintln!("polish: success after {} passes", pass + 1);
        }
        return Some(PolishResult { x, s, z });
    }

    if diag_enabled {
        eprintln!("polish: failed after {} passes (max_passes reached)", max_passes);
    }
    None
}

fn build_equality_system(
    prob: &ProblemData,
    eq_rows: &[usize],
    active_ineq_rows: &[usize],
) -> (sparse::SparseCsc, Vec<f64>, Vec<usize>) {
    let n = prob.num_vars();

    // Row ids in the new system (for debugging / future extensions).
    let mut row_ids = Vec::with_capacity(eq_rows.len() + active_ineq_rows.len());
    row_ids.extend_from_slice(eq_rows);
    row_ids.extend_from_slice(active_ineq_rows);

    // Map old row -> new row index.
    let mut row_map = vec![None; prob.num_constraints()];
    for (new_i, &old_i) in row_ids.iter().enumerate() {
        row_map[old_i] = Some(new_i);
    }

    let m_eq = row_ids.len();
    let mut triplets = Vec::with_capacity(prob.A.nnz());
    for (val, (row, col)) in prob.A.iter() {
        if let Some(new_row) = row_map[row] {
            triplets.push((new_row, col, *val));
        }
    }

    let a_eq = sparse::from_triplets(m_eq, n, triplets);
    let b_eq: Vec<f64> = row_ids.iter().map(|&r| prob.b[r]).collect();
    (a_eq, b_eq, row_ids)
}

fn compute_slack(prob: &ProblemData, x: &[f64]) -> Vec<f64> {
    let m = prob.num_constraints();
    let n = prob.num_vars();
    debug_assert_eq!(x.len(), n);

    // s = b - A x
    let mut s = prob.b.clone();
    for (val, (row, col)) in prob.A.iter() {
        s[row] -= (*val) * x[col];
    }
    debug_assert_eq!(s.len(), m);
    s
}

/// Primal projection polish: project x onto active constraints with large violations.
///
/// This handles the case where dual/gap are converged but primal residual is stuck
/// on a few active constraints. We find a minimum-norm correction Δx such that
/// A_active * (x + Δx) = b_active for those constraints.
///
/// Returns `Some(PolishResult)` if successful, `None` otherwise.
pub fn polish_primal_projection(
    prob: &ProblemData,
    x0: &[f64],
    s0: &[f64],
    z0: &[f64],
    rp: &[f64],
    tol_feas: f64,
) -> Option<PolishResult> {
    let diag_enabled = std::env::var("MINIX_DIAGNOSTICS").is_ok();
    let n = prob.num_vars();
    let m = prob.num_constraints();

    if x0.len() != n || s0.len() != m || z0.len() != m || rp.len() != m {
        return None;
    }

    // Only handle Zero + NonNeg cones
    if prob.cones.iter().any(|c| !matches!(c, ConeSpec::Zero { .. } | ConeSpec::NonNeg { .. })) {
        return None;
    }

    // Find rows that are:
    // 1. Active (s ≈ 0)
    // 2. Have significant primal violation (|rp| close to the max violation)
    //
    // The key insight: we only want to project onto the rows that dominate the
    // primal residual, not all rows with small slack. Use the max |rp| to filter.
    let rp_max = rp.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
    let rp_thresh = rp_max * 0.9; // Only rows with |rp| >= 90% of max
    let s_thresh = tol_feas * 0.001; // Very small slack = active (s < 1e-11)

    let mut violating_active: Vec<usize> = (0..m)
        .filter(|&i| s0[i].abs() < s_thresh && rp[i].abs() >= rp_thresh)
        .collect();

    if violating_active.is_empty() {
        if diag_enabled {
            eprintln!("primal_polish: no violating active constraints");
        }
        return None;
    }

    // Sort by violation magnitude (largest first) and take top rows
    // Using fewer rows means smaller Δx, which means less dual disruption
    violating_active.sort_by(|&a, &b| {
        rp[b].abs().partial_cmp(&rp[a].abs()).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Adaptive row limit: start conservative (16 rows) to minimize dual disruption
    // The caller can iterate with more rows if needed, but we found empirically
    // that fixing ~15-20 rows balances primal improvement vs dual degradation
    let max_rows = 16; // Power of 2 for potential SIMD alignment benefit
    if violating_active.len() > max_rows {
        violating_active.truncate(max_rows);
        if diag_enabled {
            eprintln!("primal_polish: limiting to top {} violating rows", max_rows);
        }
    }

    let k = violating_active.len();
    if diag_enabled {
        eprintln!("primal_polish: {} violating active constraints", k);
    }

    // Build A_active (k × n) as dense rows
    // Each row is the constraint coefficients for a violating row
    let mut a_rows: Vec<Vec<f64>> = vec![vec![0.0; n]; k];
    for (&val, (row, col)) in prob.A.iter() {
        if let Some(idx) = violating_active.iter().position(|&r| r == row) {
            a_rows[idx][col] = val;
        }
    }

    // Build rhs = -rp_active (the residual we want to eliminate)
    let rhs: Vec<f64> = violating_active.iter().map(|&i| -rp[i]).collect();

    // Solve min ||Δx||² s.t. A_active * Δx = rhs
    // Solution: Δx = A^T * (A * A^T)^{-1} * rhs
    //
    // First compute G = A * A^T (k × k dense)
    let mut g = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..=i {
            let dot: f64 = (0..n).map(|c| a_rows[i][c] * a_rows[j][c]).sum();
            g[i][j] = dot;
            g[j][i] = dot;
        }
    }

    // Add small regularization for numerical stability
    for i in 0..k {
        g[i][i] += 1e-12;
    }

    // Solve G * y = rhs using Cholesky (G is SPD)
    let y = match cholesky_solve(&g, &rhs) {
        Some(y) => y,
        None => {
            if diag_enabled {
                eprintln!("primal_polish: Cholesky solve failed");
            }
            return None;
        }
    };

    // Δx = A^T * y
    let mut dx = vec![0.0; n];
    for (i, &yi) in y.iter().enumerate() {
        for j in 0..n {
            dx[j] += a_rows[i][j] * yi;
        }
    }

    // Check correction magnitude - don't apply huge corrections
    let dx_norm = dx.iter().map(|x| x * x).sum::<f64>().sqrt();
    let x_norm = x0.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.0);
    if dx_norm > 0.1 * x_norm {
        if diag_enabled {
            eprintln!("primal_polish: correction too large ({:.3e} vs {:.3e}), rejecting", dx_norm, x_norm);
        }
        return None;
    }

    // Apply correction
    let mut x = x0.to_vec();
    for i in 0..n {
        x[i] += dx[i];
    }

    // Recompute s = b - Ax
    let s = compute_slack(prob, &x);

    // Keep z unchanged (dual is already good)
    let z = z0.to_vec();

    if diag_enabled {
        // Verify improvement
        let new_rp: Vec<f64> = (0..m).map(|i| prob.b[i] - s[i] - {
            let mut ax_i = 0.0;
            for (&val, (row, col)) in prob.A.iter() {
                if row == i {
                    ax_i += val * x[col];
                }
            }
            ax_i
        }).collect();
        let new_rp_inf = inf_norm(&new_rp);
        let old_rp_inf = inf_norm(rp);
        eprintln!("primal_polish: |rp| {:.3e} -> {:.3e}, |dx|={:.3e}", old_rp_inf, new_rp_inf, dx_norm);
    }

    Some(PolishResult { x, s, z })
}

/// Combined primal + dual polish.
///
/// First applies primal projection to fix primal residuals, then computes a dual
/// correction to mitigate the dual degradation caused by changing x.
///
/// The dual residual after primal correction is rd_new = P*x_new + q - A^T*z.
/// We want to find Δz such that A^T*Δz ≈ P*Δx to minimize dual degradation.
pub fn polish_primal_and_dual(
    prob: &ProblemData,
    x0: &[f64],
    s0: &[f64],
    z0: &[f64],
    rp: &[f64],
    tol_feas: f64,
) -> Option<PolishResult> {
    let diag_enabled = std::env::var("MINIX_DIAGNOSTICS").is_ok();
    let n = prob.num_vars();
    let m = prob.num_constraints();

    // First, get the primal correction
    let primal_result = polish_primal_projection(prob, x0, s0, z0, rp, tol_feas)?;

    // Compute Δx from primal correction
    let dx: Vec<f64> = primal_result.x.iter()
        .zip(x0.iter())
        .map(|(&xp, &x0)| xp - x0)
        .collect();

    // Compute P*Δx (the dual residual change)
    let mut p_dx = vec![0.0; n];
    if let Some(ref p) = prob.P {
        for col in 0..n {
            if let Some(col_view) = p.outer_view(col) {
                for (row, &val) in col_view.iter() {
                    p_dx[row] += val * dx[col];
                }
            }
        }
    }

    let p_dx_norm = p_dx.iter().map(|x| x * x).sum::<f64>().sqrt();
    if p_dx_norm < 1e-14 {
        // No dual correction needed
        return Some(primal_result);
    }

    if diag_enabled {
        eprintln!("dual_polish: |P*dx|={:.3e}", p_dx_norm);
    }

    // Solve for Δz: A^T*Δz = P*Δx  (n equations, m unknowns)
    // Minimum-norm solution: Δz = A * (A^T*A)^{-1} * (P*Δx)
    //
    // But A^T*A is n×n dense which is expensive. For now, use a simplified approach:
    // Only adjust z for rows where the constraint is active (s ≈ 0).
    //
    // For active row i: z[i] adjustment affects rd via -A[i,:] (the i-th row of A).
    // We want: -Σ_i A[i,j]*Δz[i] ≈ P*Δx[j] for each j.
    //
    // This is still a least-squares problem but over active rows only.
    let s_thresh = tol_feas * 0.001;
    let active_rows: Vec<usize> = (0..m)
        .filter(|&i| s0[i].abs() < s_thresh)
        .collect();

    if active_rows.is_empty() || active_rows.len() > 500 {
        // Too many active rows, skip dual correction
        if diag_enabled {
            eprintln!("dual_polish: skipping, active_rows={}", active_rows.len());
        }
        return Some(primal_result);
    }

    let k = active_rows.len();
    if diag_enabled {
        eprintln!("dual_polish: {} active rows for dual correction", k);
    }

    // Build A_active^T (n × k) and then compute (A_active * A_active^T)^{-1} * A_active * p_dx
    // Build A_active as k×n dense
    let mut a_active: Vec<Vec<f64>> = vec![vec![0.0; n]; k];
    for (&val, (row, col)) in prob.A.iter() {
        if let Some(idx) = active_rows.iter().position(|&r| r == row) {
            a_active[idx][col] = val;
        }
    }

    // Compute A_active * p_dx (k-vector)
    let a_pdx: Vec<f64> = (0..k)
        .map(|i| (0..n).map(|j| a_active[i][j] * p_dx[j]).sum())
        .collect();

    // Compute G = A_active * A_active^T (k × k)
    let mut g = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..=i {
            let dot: f64 = (0..n).map(|c| a_active[i][c] * a_active[j][c]).sum();
            g[i][j] = dot;
            g[j][i] = dot;
        }
    }

    // Add regularization
    for i in 0..k {
        g[i][i] += 1e-10;
    }

    // Solve G * y = A_active * p_dx
    let y = cholesky_solve(&g, &a_pdx)?;

    // Δz for active rows: Δz[active_rows[i]] = y[i]
    let mut dz = vec![0.0; m];
    for (i, &row) in active_rows.iter().enumerate() {
        dz[row] = y[i];
    }

    // Apply dual correction
    let mut z = primal_result.z.clone();
    for i in 0..m {
        z[i] += dz[i];
    }

    // Ensure z stays non-negative for NonNeg cones
    // (This is important - z must be in the dual cone)
    let mut offset = 0;
    for cone in &prob.cones {
        match *cone {
            ConeSpec::Zero { dim } => {
                // z can be anything for Zero cone (it's the free dual)
                offset += dim;
            }
            ConeSpec::NonNeg { dim } => {
                // z must be >= 0 for NonNeg cone
                for i in offset..offset + dim {
                    z[i] = z[i].max(0.0);
                }
                offset += dim;
            }
            _ => {
                // Skip for other cones
                offset += cone.dim();
            }
        }
    }

    if diag_enabled {
        let dz_norm = dz.iter().map(|x| x * x).sum::<f64>().sqrt();
        eprintln!("dual_polish: |dz|={:.3e}", dz_norm);
    }

    Some(PolishResult {
        x: primal_result.x,
        s: primal_result.s,
        z,
    })
}

/// Simple Cholesky solve for small dense SPD systems
fn cholesky_solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 || b.len() != n {
        return None;
    }

    // Cholesky factorization: A = L * L^T
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 0.0 {
                    return None; // Not positive definite
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    // Forward solve: L * y = b
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i][j] * y[j];
        }
        y[i] = sum / l[i][i];
    }

    // Backward solve: L^T * x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j][i] * x[j];
        }
        x[i] = sum / l[i][i];
    }

    Some(x)
}
