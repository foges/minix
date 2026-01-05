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
