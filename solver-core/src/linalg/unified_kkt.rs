//! Unified KKT solver interface.
//!
//! Provides a common interface for different KKT solving strategies:
//! - Standard augmented KKT system (sparse LDL)
//! - Normal equations for tall problems (dense Cholesky)
//!
//! This module implements `KktSolverTrait` by dispatching to the appropriate
//! backend based on problem structure.

use super::backend::BackendError;
use super::kkt::KktSolver;
use super::kkt_trait::KktSolverTrait;
use super::normal_eqns::{NormalEqnsFactor, NormalEqnsSolver};
use super::sparse::{SparseCsc, SparseSymmetricCsc};
use crate::linalg::qdldl::QdldlFactorization;
use crate::problem::ConeSpec;
use crate::scaling::ScalingBlock;

/// Unified factor token that wraps the underlying solver's factorization.
pub enum UnifiedFactor {
    /// Factorization from standard augmented KKT solver (sparse LDL)
    Standard(QdldlFactorization),
    /// Factorization from normal equations solver (dense Cholesky, stored in solver)
    NormalEqns(NormalEqnsFactor),
}

/// Unified KKT solver that auto-selects between standard and normal equations.
pub enum UnifiedKktSolver {
    /// Standard augmented KKT system (for general problems)
    Standard(KktSolver),
    /// Normal equations for tall problems (m >> n with diagonal H)
    NormalEqns(NormalEqnsSolver),
}

/// Check if problem is suitable for normal equations.
///
/// Returns true if:
/// - MINIX_NORMAL_EQNS=1 environment variable is set (opt-in for now)
/// - m > 5*n (tall problem)
/// - n <= 500 (dense ops are fast)
/// - All cones are Zero or NonNeg (diagonal H)
///
/// Note: Normal equations solver is disabled by default until properly validated.
/// Enable with MINIX_NORMAL_EQNS=1 for testing.
pub fn should_use_normal_equations(n: usize, m: usize, cones: &[ConeSpec]) -> bool {
    // Opt-in for now - normal equations needs more testing
    let enabled = std::env::var("MINIX_NORMAL_EQNS")
        .map(|v| v != "0")
        .unwrap_or(false);
    if !enabled {
        return false;
    }

    if m <= 5 * n || n > 500 {
        return false;
    }

    // Check all cones are Zero or NonNeg
    for cone in cones {
        match cone {
            ConeSpec::Zero { .. } | ConeSpec::NonNeg { .. } => {}
            _ => return false,
        }
    }

    true
}

impl UnifiedKktSolver {
    /// Create a new unified KKT solver, auto-selecting the best strategy.
    pub fn new(
        n: usize,
        m: usize,
        static_reg: f64,
        dynamic_reg_min_pivot: f64,
        p: Option<&SparseSymmetricCsc>,
        a: &SparseCsc,
        h_blocks: &[ScalingBlock],
        cones: &[ConeSpec],
    ) -> Self {
        if should_use_normal_equations(n, m, cones) {
            if std::env::var("MINIX_DIAGNOSTICS").ok().as_deref() == Some("1") {
                eprintln!(
                    "Using normal equations solver (n={}, m={}, ratio={:.1}x)",
                    n, m, m as f64 / n as f64
                );
            }
            let solver = NormalEqnsSolver::new(n, m, p, a, static_reg);
            UnifiedKktSolver::NormalEqns(solver)
        } else {
            let kkt = KktSolver::new_with_singleton_elimination(
                n,
                m,
                static_reg,
                dynamic_reg_min_pivot,
                a,
                h_blocks,
            );
            UnifiedKktSolver::Standard(kkt)
        }
    }

    /// Check if using normal equations.
    pub fn is_normal_equations(&self) -> bool {
        matches!(self, UnifiedKktSolver::NormalEqns(_))
    }
}

impl KktSolverTrait for UnifiedKktSolver {
    type Factor = UnifiedFactor;

    fn initialize(
        &mut self,
        p: Option<&SparseSymmetricCsc>,
        a: &SparseCsc,
        h_blocks: &[ScalingBlock],
    ) -> Result<(), BackendError> {
        match self {
            UnifiedKktSolver::Standard(kkt) => kkt.initialize(p, a, h_blocks),
            UnifiedKktSolver::NormalEqns(solver) => solver.initialize(p, a, h_blocks),
        }
    }

    fn update_numeric(
        &mut self,
        p: Option<&SparseSymmetricCsc>,
        a: &SparseCsc,
        h_blocks: &[ScalingBlock],
    ) -> Result<(), BackendError> {
        match self {
            UnifiedKktSolver::Standard(kkt) => kkt.update_numeric(p, a, h_blocks),
            UnifiedKktSolver::NormalEqns(solver) => solver.update_numeric(p, a, h_blocks),
        }
    }

    fn factorize(&mut self) -> Result<Self::Factor, BackendError> {
        match self {
            UnifiedKktSolver::Standard(kkt) => {
                let factor = KktSolverTrait::factorize(kkt)?;
                Ok(UnifiedFactor::Standard(factor))
            }
            UnifiedKktSolver::NormalEqns(solver) => {
                let factor = KktSolverTrait::factorize(solver)?;
                Ok(UnifiedFactor::NormalEqns(factor))
            }
        }
    }

    fn solve_refined(
        &mut self,
        factor: &Self::Factor,
        rhs_x: &[f64],
        rhs_z: &[f64],
        sol_x: &mut [f64],
        sol_z: &mut [f64],
        refine_iters: usize,
    ) {
        match (self, factor) {
            (UnifiedKktSolver::Standard(kkt), UnifiedFactor::Standard(f)) => {
                kkt.solve_refined(f, rhs_x, rhs_z, sol_x, sol_z, refine_iters);
            }
            (UnifiedKktSolver::NormalEqns(solver), UnifiedFactor::NormalEqns(f)) => {
                solver.solve_refined(f, rhs_x, rhs_z, sol_x, sol_z, refine_iters);
            }
            _ => panic!("Mismatched solver and factor types"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_two_rhs_refined_tagged(
        &mut self,
        factor: &Self::Factor,
        rhs_x1: &[f64],
        rhs_z1: &[f64],
        rhs_x2: &[f64],
        rhs_z2: &[f64],
        sol_x1: &mut [f64],
        sol_z1: &mut [f64],
        sol_x2: &mut [f64],
        sol_z2: &mut [f64],
        refine_iters: usize,
        tag1: &'static str,
        tag2: &'static str,
    ) {
        match (self, factor) {
            (UnifiedKktSolver::Standard(kkt), UnifiedFactor::Standard(f)) => {
                kkt.solve_two_rhs_refined_tagged(
                    f, rhs_x1, rhs_z1, rhs_x2, rhs_z2,
                    sol_x1, sol_z1, sol_x2, sol_z2,
                    refine_iters, tag1, tag2,
                );
            }
            (UnifiedKktSolver::NormalEqns(solver), UnifiedFactor::NormalEqns(f)) => {
                solver.solve_two_rhs_refined_tagged(
                    f, rhs_x1, rhs_z1, rhs_x2, rhs_z2,
                    sol_x1, sol_z1, sol_x2, sol_z2,
                    refine_iters, tag1, tag2,
                );
            }
            _ => panic!("Mismatched solver and factor types"),
        }
    }

    fn static_reg(&self) -> f64 {
        match self {
            UnifiedKktSolver::Standard(kkt) => KktSolverTrait::static_reg(kkt),
            UnifiedKktSolver::NormalEqns(solver) => KktSolverTrait::static_reg(solver),
        }
    }

    fn set_static_reg(&mut self, reg: f64) -> Result<(), BackendError> {
        match self {
            UnifiedKktSolver::Standard(kkt) => KktSolverTrait::set_static_reg(kkt, reg),
            UnifiedKktSolver::NormalEqns(solver) => KktSolverTrait::set_static_reg(solver, reg),
        }
    }

    fn bump_static_reg(&mut self, min_reg: f64) -> Result<bool, BackendError> {
        match self {
            UnifiedKktSolver::Standard(kkt) => KktSolverTrait::bump_static_reg(kkt, min_reg),
            UnifiedKktSolver::NormalEqns(solver) => KktSolverTrait::bump_static_reg(solver, min_reg),
        }
    }

    fn dynamic_bumps(&self) -> u64 {
        match self {
            UnifiedKktSolver::Standard(kkt) => KktSolverTrait::dynamic_bumps(kkt),
            UnifiedKktSolver::NormalEqns(solver) => KktSolverTrait::dynamic_bumps(solver),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_use_normal_equations() {
        // Tall problem with only NonNeg cones - should use
        let cones = vec![ConeSpec::NonNeg { dim: 100 }];
        assert!(should_use_normal_equations(10, 100, &cones));

        // Not tall enough - should not use
        assert!(!should_use_normal_equations(10, 30, &cones));

        // Has SOC cone - should not use
        let cones_soc = vec![ConeSpec::NonNeg { dim: 50 }, ConeSpec::Soc { dim: 10 }];
        assert!(!should_use_normal_equations(10, 100, &cones_soc));

        // n too large - should not use
        let cones_large = vec![ConeSpec::NonNeg { dim: 10000 }];
        assert!(!should_use_normal_equations(600, 10000, &cones_large));
    }
}
