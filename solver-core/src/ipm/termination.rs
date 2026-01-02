//! Termination criteria for the IPM solver.
//!
//! Checks for:
//! - Optimality: Primal/dual feasibility + small duality gap
//! - Primal infeasibility: τ → 0 with b^T z < 0
//! - Dual infeasibility: τ → 0 with q^T x < 0
//! - Numerical errors: NaN, factorization failure, stalled progress

use super::hsde::{HsdeState, HsdeResiduals};
use crate::problem::{ProblemData, SolveStatus};

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
            tol_gap_rel: 1e-3,  // 0.1% relative gap tolerance
            tol_infeas: 1e-8,
            tau_min: 1e-8,
            max_iter: 200,
            min_progress: 1e-12,
        }
    }
}

/// Check termination conditions.
///
/// Returns `Some(status)` if solver should terminate, `None` otherwise.
pub fn check_termination(
    prob: &ProblemData,
    state: &HsdeState,
    residuals: &HsdeResiduals,
    mu: f64,
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

    // Unscale solution by τ
    if state.tau < criteria.tau_min {
        // τ ≈ 0: Check for infeasibility certificates
        return check_infeasibility(prob, state, criteria);
    }

    // Compute unscaled quantities
    let x_bar: Vec<f64> = state.x.iter().map(|xi| xi / state.tau).collect();
    let _s_bar: Vec<f64> = state.s.iter().map(|si| si / state.tau).collect();
    let z_bar: Vec<f64> = state.z.iter().map(|zi| zi / state.tau).collect();

    // Compute primal objective: 0.5 * x^T P x + q^T x
    let mut primal_obj = 0.0;

    if let Some(ref p) = prob.P {
        // x^T P x
        let mut px = vec![0.0; prob.num_vars()];
        for col in 0..prob.num_vars() {
            if let Some(col_view) = p.outer_view(col) {
                for (row, &val) in col_view.iter() {
                    px[row] += val * x_bar[col];
                    if row != col {
                        px[col] += val * x_bar[row]; // Symmetric
                    }
                }
            }
        }
        for i in 0..prob.num_vars() {
            primal_obj += 0.5 * x_bar[i] * px[i];
        }
    }

    // q^T x
    for i in 0..prob.num_vars() {
        primal_obj += prob.q[i] * x_bar[i];
    }

    // Compute dual objective: -b^T z
    let mut dual_obj = 0.0;
    for i in 0..prob.num_constraints() {
        dual_obj -= prob.b[i] * z_bar[i];
    }

    // Compute residuals (unscaled)
    let (rx_norm, rz_norm, _) = residuals.norms();

    // Scale norms by τ for comparison
    let primal_res = rx_norm / state.tau.max(1.0);
    let dual_res = rz_norm / state.tau.max(1.0);
    let gap = (primal_obj - dual_obj).abs();

    // Compute relative gap: gap / max(|primal_obj|, |dual_obj|, 1)
    let scale = primal_obj.abs().max(dual_obj.abs()).max(1.0);
    let gap_rel = gap / scale;

    // Check optimality (either absolute or relative gap tolerance met)
    let gap_ok = gap < criteria.tol_gap || gap_rel < criteria.tol_gap_rel;
    if primal_res < criteria.tol_feas && dual_res < criteria.tol_feas && gap_ok {
        return Some(SolveStatus::Optimal);
    }

    // Check for stalled progress
    if mu < criteria.min_progress && iter > 10 {
        // Consider this "solved" if residuals are reasonable
        if primal_res < 10.0 * criteria.tol_feas && dual_res < 10.0 * criteria.tol_feas {
            return Some(SolveStatus::Optimal);
        }
    }

    None
}

/// Check for infeasibility certificates when τ ≈ 0.
fn check_infeasibility(
    prob: &ProblemData,
    state: &HsdeState,
    criteria: &TerminationCriteria,
) -> Option<SolveStatus> {
    if state.tau > criteria.tau_min {
        return None;
    }

    // Check primal infeasibility: b^T z < 0
    let btz: f64 = prob.b.iter().zip(state.z.iter()).map(|(bi, zi)| bi * zi).sum();

    if btz < -1e-8 {
        return Some(SolveStatus::PrimalInfeasible);
    }

    // Check dual infeasibility: q^T x < 0
    let qtx: f64 = prob.q.iter().zip(state.x.iter()).map(|(qi, xi)| qi * xi).sum();

    if qtx < -1e-8 {
        return Some(SolveStatus::DualInfeasible);
    }

    // τ ≈ 0 but no clear certificate
    Some(SolveStatus::NumericalError)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::sparse;

    #[test]
    fn test_termination_optimal() {
        // Simple LP
        let prob = ProblemData {
            P: None,
            q: vec![1.0, 1.0],
            A: sparse::from_triplets(1, 2, vec![(0, 0, 1.0), (0, 1, 1.0)]),
            b: vec![1.0],
            cones: vec![],
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

        let mut residuals = HsdeResiduals::new(2, 1);
        // Make residuals small (near-optimal)
        residuals.r_x = vec![1e-9, 1e-9];
        residuals.r_z = vec![1e-9];
        residuals.r_tau = 1e-9;

        let criteria = TerminationCriteria::default();

        let status = check_termination(&prob, &state, &residuals, 1e-9, 10, &criteria);

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
            cones: vec![],
            var_bounds: None,
            integrality: None,
        };

        let state = HsdeState::new(1, 1);
        let residuals = HsdeResiduals::new(1, 1);
        let criteria = TerminationCriteria {
            max_iter: 50,
            ..Default::default()
        };

        let status = check_termination(&prob, &state, &residuals, 1.0, 51, &criteria);

        assert!(matches!(status, Some(SolveStatus::MaxIters)));
    }

    #[test]
    fn test_termination_primal_infeasible() {
        // Primal infeasible problem (no x satisfies Ax = b, x >= 0)
        let prob = ProblemData {
            P: None,
            q: vec![0.0],
            A: sparse::from_triplets(1, 1, vec![(0, 0, 1.0)]),
            b: vec![-1.0], // Infeasible for x >= 0
            cones: vec![],
            var_bounds: None,
            integrality: None,
        };

        let state = HsdeState {
            x: vec![0.0],
            s: vec![1.0],
            z: vec![1.0], // z > 0
            tau: 1e-10,   // τ → 0
            kappa: 1.0,
            xi: vec![0.0],  // ξ = x/τ (but x=0 anyway)
        };

        let residuals = HsdeResiduals::new(1, 1);
        let criteria = TerminationCriteria::default();

        let status = check_termination(&prob, &state, &residuals, 1.0, 10, &criteria);

        // Should detect primal infeasibility (b^T z = -1 * 1 = -1 < 0)
        assert!(matches!(status, Some(SolveStatus::PrimalInfeasible)));
    }
}
