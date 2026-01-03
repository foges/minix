//! Master problem backend trait and types.

use crate::error::MipResult;
use crate::model::MipProblem;

/// Status of master problem solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MasterStatus {
    /// Optimal solution found.
    Optimal,

    /// Master LP/QP is infeasible (node can be pruned).
    Infeasible,

    /// Master is unbounded (shouldn't happen with proper bounds).
    Unbounded,

    /// Numerical difficulties.
    NumericalError,
}

/// Result from solving the master problem.
#[derive(Debug, Clone)]
pub struct MasterResult {
    /// Solve status.
    pub status: MasterStatus,

    /// Primal solution x.
    pub x: Vec<f64>,

    /// Primal objective value.
    pub obj_val: f64,

    /// Dual objective value (lower bound).
    pub dual_obj: f64,

    /// Slack variables s (for constraint Ax + s = b).
    pub s: Vec<f64>,

    /// Dual variables z.
    pub z: Vec<f64>,
}

impl MasterResult {
    /// Create an infeasible result.
    pub fn infeasible() -> Self {
        Self {
            status: MasterStatus::Infeasible,
            x: Vec::new(),
            obj_val: f64::INFINITY,
            dual_obj: f64::INFINITY,
            s: Vec::new(),
            z: Vec::new(),
        }
    }
}

/// Source of a cut (for tracking and debugging).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutSource {
    /// K* certificate cut from infeasible conic subproblem.
    KStarCertificate {
        /// Which cone block generated this cut.
        cone_idx: usize,
    },

    /// SOC tangent cut.
    SocTangent {
        /// Which SOC cone.
        cone_idx: usize,
    },

    /// Disaggregated cut from a specific cone block.
    Disaggregated {
        /// Cone index.
        cone_idx: usize,
        /// Block within the cone (for product cones).
        block: usize,
    },

    /// User-provided cut.
    User,
}

/// A linear cut: a^T x <= rhs.
#[derive(Debug, Clone)]
pub struct LinearCut {
    /// Coefficient vector (dense, length n).
    pub coefs: Vec<f64>,

    /// Right-hand side.
    pub rhs: f64,

    /// Optional name for debugging.
    pub name: Option<String>,

    /// Source of this cut.
    pub source: CutSource,
}

impl LinearCut {
    /// Create a new cut.
    pub fn new(coefs: Vec<f64>, rhs: f64, source: CutSource) -> Self {
        Self {
            coefs,
            rhs,
            name: None,
            source,
        }
    }

    /// Create a cut with a name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Compute violation: a^T x - rhs (positive means violated).
    pub fn violation(&self, x: &[f64]) -> f64 {
        let lhs: f64 = self.coefs.iter().zip(x.iter()).map(|(a, x)| a * x).sum();
        lhs - self.rhs
    }

    /// Check if cut is violated by more than tolerance.
    pub fn is_violated(&self, x: &[f64], tol: f64) -> bool {
        self.violation(x) > tol
    }

    /// Normalize the cut so that ||a||_inf = 1.
    pub fn normalize(&mut self) {
        let max_coef = self
            .coefs
            .iter()
            .map(|c| c.abs())
            .fold(0.0_f64, f64::max);

        if max_coef > 1e-12 {
            for c in &mut self.coefs {
                *c /= max_coef;
            }
            self.rhs /= max_coef;
        }
    }

    /// Check if cut has valid coefficients (not all zeros, finite).
    pub fn is_valid(&self) -> bool {
        let has_nonzero = self.coefs.iter().any(|c| c.abs() > 1e-12);
        let all_finite = self.coefs.iter().all(|c| c.is_finite()) && self.rhs.is_finite();
        has_nonzero && all_finite
    }
}

/// Trait for master problem backends (LP/QP solvers).
///
/// The master backend maintains the current LP/QP relaxation of the MIP,
/// including variable bounds and cuts. It supports:
/// - Solving the relaxation
/// - Adding/removing cuts
/// - Updating variable bounds (for branching)
pub trait MasterBackend {
    /// Initialize the backend with the base problem.
    ///
    /// This creates the initial LP/QP master by:
    /// - Keeping Zero and NonNeg cones
    /// - Relaxing SOC/other cones (to be enforced via cuts)
    /// - Setting up variable bounds
    fn initialize(&mut self, prob: &MipProblem) -> MipResult<()>;

    /// Add a linear cut: a^T x <= rhs.
    ///
    /// Returns an identifier for the cut (for later removal).
    fn add_cut(&mut self, cut: &LinearCut) -> usize;

    /// Add multiple cuts efficiently.
    fn add_cuts(&mut self, cuts: &[LinearCut]) -> Vec<usize> {
        cuts.iter().map(|c| self.add_cut(c)).collect()
    }

    /// Remove cuts by their identifiers.
    fn remove_cuts(&mut self, cut_ids: &[usize]);

    /// Update variable bounds (for branching).
    fn set_var_bounds(&mut self, var: usize, lb: f64, ub: f64);

    /// Solve the current master LP/QP.
    fn solve(&mut self) -> MipResult<MasterResult>;

    /// Get the number of active cuts.
    fn num_cuts(&self) -> usize;

    /// Get the number of variables.
    fn num_vars(&self) -> usize;

    /// Get the number of constraints (excluding cuts).
    fn num_base_constraints(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cut_violation() {
        // Cut: x0 + x1 <= 1
        let cut = LinearCut::new(vec![1.0, 1.0], 1.0, CutSource::User);

        // (0.5, 0.5) satisfies: 0.5 + 0.5 = 1 <= 1
        assert!(!cut.is_violated(&[0.5, 0.5], 1e-6));

        // (0.6, 0.6) violates: 0.6 + 0.6 = 1.2 > 1
        assert!(cut.is_violated(&[0.6, 0.6], 1e-6));

        let viol = cut.violation(&[0.6, 0.6]);
        assert!((viol - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_cut_normalization() {
        let mut cut = LinearCut::new(vec![2.0, 4.0], 6.0, CutSource::User);
        cut.normalize();

        // After normalization: 0.5*x0 + 1.0*x1 <= 1.5
        assert!((cut.coefs[0] - 0.5).abs() < 1e-10);
        assert!((cut.coefs[1] - 1.0).abs() < 1e-10);
        assert!((cut.rhs - 1.5).abs() < 1e-10);
    }
}
