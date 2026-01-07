//! Termination criteria for the IPM solver.
//!
//! Checks for:
//! - Optimality: Primal/dual feasibility + small duality gap
//! - Primal infeasibility: τ → 0 with b^T z < 0
//! - Dual infeasibility: τ → 0 with q^T x < 0
//! - Numerical errors: NaN, factorization failure, stalled progress
//!
//! IMPORTANT: All termination checks should be done on **unscaled** data
//! (after undoing Ruiz scaling). See design doc §16.

use super::hsde::HsdeState;
use crate::presolve::ruiz::RuizScaling;
use crate::problem::{ConeSpec, ProblemData, SolveStatus};

/// Termination criteria.
#[derive(Debug, Clone)]
pub struct TerminationCriteria {
    /// Tolerance for primal/dual feasibility
    pub tol_feas: f64,

    /// Tolerance for absolute duality gap
    pub tol_gap: f64,

    /// Tolerance for relative duality gap (gap / max(|primal_obj|, |dual_obj|, 1))
    pub tol_gap_rel: f64,

    /// Tolerance for infeasibility detection
    pub tol_infeas: f64,

    /// Minimum τ threshold for infeasibility detection
    pub tau_min: f64,

    /// Maximum iterations
    pub max_iter: usize,

    /// Minimum progress threshold (μ reduction per iteration)
    pub min_progress: f64,
}

impl Default for TerminationCriteria {
    fn default() -> Self {
        Self {
            tol_feas: 1e-8,
            tol_gap: 1e-8,
            tol_gap_rel: 1e-8,  // Match Clarabel: was 1e-3 (too loose!)
            tol_infeas: 1e-8,
            tau_min: 1e-8,
            max_iter: 200,
            min_progress: 1e-12,
        }
    }
}

#[inline]
fn inf_norm(v: &[f64]) -> f64 {
    v.iter()
        .map(|x| x.abs())
        .fold(0.0_f64, f64::max)
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// Check termination conditions.
///
/// Returns `Some(status)` if solver should terminate, `None` otherwise.
pub fn check_termination(
    prob: &ProblemData,
    scaling: &RuizScaling,
    state: &HsdeState,
    iter: usize,
    criteria: &TerminationCriteria,
) -> Option<SolveStatus> {
    // Check for NaN
    if state.tau.is_nan() || state.kappa.is_nan() {
        return Some(SolveStatus::NumericalError);
    }

    for &xi in &state.x {
        if xi.is_nan() {
            return Some(SolveStatus::NumericalError);
        }
    }

    // Check max iterations
    if iter >= criteria.max_iter {
        return Some(SolveStatus::MaxIters);
    }

    // τ ≈ 0: check infeasibility certificates.
    if state.tau < criteria.tau_min {
        return check_infeasibility(prob, scaling, state, criteria);
    }

    // Unscale solution by τ and undo Ruiz scaling.
    let inv_tau = 1.0 / state.tau;
    let x_bar_scaled: Vec<f64> = state.x.iter().map(|xi| xi * inv_tau).collect();
    let s_bar_scaled: Vec<f64> = state.s.iter().map(|si| si * inv_tau).collect();
    let z_bar_scaled: Vec<f64> = state.z.iter().map(|zi| zi * inv_tau).collect();

    let x_bar = scaling.unscale_x(&x_bar_scaled);
    let s_bar = scaling.unscale_s(&s_bar_scaled);
    let z_bar = scaling.unscale_z(&z_bar_scaled);

    let n = prob.num_vars();
    let m = prob.num_constraints();
    debug_assert_eq!(x_bar.len(), n);
    debug_assert_eq!(s_bar.len(), m);
    debug_assert_eq!(z_bar.len(), m);

    // Residuals on unscaled data:
    //   r_p = A x̄ + s̄ - b
    //   r_d = P x̄ + A^T z̄ + q
    let mut r_p = s_bar.clone();
    for i in 0..m {
        r_p[i] -= prob.b[i];
    }
    for (&val, (row, col)) in prob.A.iter() {
        r_p[row] += val * x_bar[col];
    }

    let mut p_x = vec![0.0; n];
    if let Some(ref p) = prob.P {
        for col in 0..n {
            if let Some(col_view) = p.outer_view(col) {
                for (row, &val) in col_view.iter() {
                    if row == col {
                        p_x[row] += val * x_bar[col];
                    } else {
                        p_x[row] += val * x_bar[col];
                        p_x[col] += val * x_bar[row];
                    }
                }
            }
        }
    }

    let mut r_d = vec![0.0; n];
    for i in 0..n {
        r_d[i] = p_x[i] + prob.q[i];
    }
    for (&val, (row, col)) in prob.A.iter() {
        r_d[col] += val * z_bar[row];
    }

    let rp_inf = inf_norm(&r_p);
    let rd_inf = inf_norm(&r_d);

    if !rp_inf.is_finite() || !rd_inf.is_finite() {
        return Some(SolveStatus::NumericalError);
    }

    // Feasibility scaling.
    let b_inf = inf_norm(&prob.b);
    let q_inf = inf_norm(&prob.q);
    let x_inf = inf_norm(&x_bar);
    let s_inf = inf_norm(&s_bar);
    let z_inf = inf_norm(&z_bar);

    let primal_scale = (b_inf + x_inf + s_inf).max(1.0);
    let dual_scale = (q_inf + x_inf + z_inf).max(1.0);

    let primal_ok = rp_inf <= criteria.tol_feas * primal_scale;
    let dual_ok = rd_inf <= criteria.tol_feas * dual_scale;

    // Objectives on unscaled data.
    let xpx = dot(&x_bar, &p_x);
    let qtx = dot(&prob.q, &x_bar);
    let btz = dot(&prob.b, &z_bar);

    let primal_obj = 0.5 * xpx + qtx;
    let dual_obj = -0.5 * xpx - btz;
    let gap = (primal_obj - dual_obj).abs();

    // Absolute gap scaling: max(1, min(|g_p|, |g_d|)).
    let gap_scale_abs = primal_obj.abs().min(dual_obj.abs()).max(1.0);
    let gap_ok_abs = gap <= criteria.tol_gap * gap_scale_abs;

    // Relative gap fallback.
    let gap_scale_rel = primal_obj.abs().max(dual_obj.abs()).max(1.0);
    let gap_rel = gap / gap_scale_rel;
    let gap_ok = gap_ok_abs || gap_rel <= criteria.tol_gap_rel;

    if primal_ok && dual_ok && gap_ok {
        return Some(SolveStatus::Optimal);
    }

    None
}

/// Check for infeasibility certificates when τ ≈ 0.
fn check_infeasibility(
    prob: &ProblemData,
    scaling: &RuizScaling,
    state: &HsdeState,
    criteria: &TerminationCriteria,
) -> Option<SolveStatus> {
    if state.tau > criteria.tau_min {
        return None;
    }

    let has_unsupported_cone = prob.cones.iter().any(|cone| {
        !matches!(
            cone,
            ConeSpec::Zero { .. }
                | ConeSpec::NonNeg { .. }
                | ConeSpec::Soc { .. }
                | ConeSpec::Psd { .. }
                | ConeSpec::Exp { .. }
                | ConeSpec::Pow { .. }
        )
    });
    if has_unsupported_cone {
        return Some(SolveStatus::NumericalError);
    }

    // Use unnormalized variables (x, s, z) and undo Ruiz scaling.
    let x = scaling.unscale_x(&state.x);
    let s = scaling.unscale_s(&state.s);
    let z = scaling.unscale_z(&state.z);

    let n = prob.num_vars();
    let m = prob.num_constraints();
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(s.len(), m);
    debug_assert_eq!(z.len(), m);

    let x_inf = inf_norm(&x);
    let s_inf = inf_norm(&s);
    let z_inf = inf_norm(&z);

    // Primal infeasibility certificate:
    //  - b^T z < -eps_abs
    //  - ||A^T z||_inf <= eps_rel * max(1, ||x||_inf + ||z||_inf) * |b^T z|
    let btz = dot(&prob.b, &z);
    if btz < -criteria.tol_infeas {
        let mut atz = vec![0.0; n];
        for (&val, (row, col)) in prob.A.iter() {
            atz[col] += val * z[row];
        }
        let atz_inf = inf_norm(&atz);
        let bound = criteria.tol_infeas * (x_inf + z_inf).max(1.0) * btz.abs();
        let z_cone_ok = dual_cone_ok(prob, &z, criteria.tol_infeas);

        if atz_inf <= bound && z_cone_ok {
            return Some(SolveStatus::PrimalInfeasible);
        }
    }

    // Dual infeasibility certificate:
    //  - q^T x < -eps_abs
    //  - ||P x||_inf <= eps_rel * max(1, ||x||_inf) * |q^T x|
    //  - ||A x + s||_inf <= eps_rel * max(1, ||x||_inf + ||s||_inf) * |q^T x|
    let qtx = dot(&prob.q, &x);
    if qtx < -criteria.tol_infeas {
        let mut p_x = vec![0.0; n];
        if let Some(ref p) = prob.P {
            for col in 0..n {
                if let Some(col_view) = p.outer_view(col) {
                    for (row, &val) in col_view.iter() {
                        if row == col {
                            p_x[row] += val * x[col];
                        } else {
                            p_x[row] += val * x[col];
                            p_x[col] += val * x[row];
                        }
                    }
                }
            }
        }
        let p_x_inf = inf_norm(&p_x);
        let px_bound = criteria.tol_infeas * x_inf.max(1.0) * qtx.abs();

        let mut ax_s = s.clone();
        for (&val, (row, col)) in prob.A.iter() {
            ax_s[row] += val * x[col];
        }
        let ax_s_inf = inf_norm(&ax_s);
        let axs_bound = criteria.tol_infeas * (x_inf + s_inf).max(1.0) * qtx.abs();

        if p_x_inf <= px_bound && ax_s_inf <= axs_bound {
            return Some(SolveStatus::DualInfeasible);
        }
    }

    Some(SolveStatus::NumericalError)
}

fn dual_cone_ok(prob: &ProblemData, z: &[f64], tol: f64) -> bool {
    let mut offset = 0;
    for cone in &prob.cones {
        match *cone {
            ConeSpec::Zero { dim } => {
                offset += dim;
            }
            ConeSpec::NonNeg { dim } => {
                if z[offset..offset + dim].iter().any(|&v| v < -tol) {
                    return false;
                }
                offset += dim;
            }
            _ => {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::sparse;
    use crate::presolve::ruiz::RuizScaling;
    use crate::problem::ConeSpec;

    #[test]
    fn test_termination_optimal() {
        // Simple LP
        let prob = ProblemData {
            P: None,
            q: vec![1.0, 1.0],
            A: sparse::from_triplets(1, 2, vec![(0, 0, 1.0), (0, 1, 1.0)]),
            b: vec![1.0],
            cones: vec![ConeSpec::Zero { dim: 1 }],
            var_bounds: None,
            integrality: None,
        };

        // At optimality: primal obj = q'x = 1.0, dual obj = -b'z
        // Strong duality: q'x = -b'z => 1.0 = -z => z = -1.0
        let state = HsdeState {
            x: vec![0.5, 0.5],
            s: vec![0.0],
            z: vec![-1.0],  // Fixed: was 1.0, should be -1.0 for strong duality
            tau: 1.0,
            kappa: 1e-10,   // Near-complementarity (was 0.0)
            xi: vec![0.5, 0.5],  // ξ = x/τ
        };

        let criteria = TerminationCriteria::default();

        let scaling = RuizScaling::identity(prob.num_vars(), prob.num_constraints());
        let status = check_termination(&prob, &scaling, &state, 10, &criteria);

        // Should detect optimality
        assert!(matches!(status, Some(SolveStatus::Optimal)));
    }

    #[test]
    fn test_termination_max_iter() {
        let prob = ProblemData {
            P: None,
            q: vec![1.0],
            A: sparse::from_triplets(1, 1, vec![(0, 0, 1.0)]),
            b: vec![1.0],
            cones: vec![ConeSpec::Zero { dim: 1 }],
            var_bounds: None,
            integrality: None,
        };

        let state = HsdeState::new(1, 1);
        let criteria = TerminationCriteria {
            max_iter: 50,
            ..Default::default()
        };

        let scaling = RuizScaling::identity(prob.num_vars(), prob.num_constraints());
        let status = check_termination(&prob, &scaling, &state, 51, &criteria);

        assert!(matches!(status, Some(SolveStatus::MaxIters)));
    }

    #[test]
    fn test_termination_primal_infeasible() {
        // Primal infeasible problem (no x satisfies Ax = b, x >= 0)
        let prob = ProblemData {
            P: None,
            q: vec![0.0],
            A: sparse::from_triplets(1, 1, vec![]), // A = 0, so Ax = b is infeasible if b != 0
            b: vec![-1.0],
            cones: vec![ConeSpec::NonNeg { dim: 1 }],
            var_bounds: None,
            integrality: None,
        };

        let state = HsdeState {
            x: vec![0.0],
            s: vec![0.0],
            z: vec![1.0], // z > 0
            tau: 1e-10,   // τ → 0
            kappa: 1.0,
            xi: vec![0.0],  // ξ = x/τ (but x=0 anyway)
        };

        let criteria = TerminationCriteria::default();

        let scaling = RuizScaling::identity(prob.num_vars(), prob.num_constraints());
        let status = check_termination(&prob, &scaling, &state, 10, &criteria);

        // Should detect primal infeasibility (b^T z = -1 * 1 = -1 < 0)
        assert!(matches!(status, Some(SolveStatus::PrimalInfeasible)));
    }
}
