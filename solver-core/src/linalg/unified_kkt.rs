//! Unified KKT solver interface.
//!
//! Provides a common interface for different KKT solving strategies:
//! - Standard augmented KKT system (sparse LDL)
//! - Normal equations for tall problems (dense Cholesky)

use super::kkt::KktSolver;
use super::normal_eqns::NormalEqnsSolver;
use super::sparse::{SparseCsc, SparseSymmetricCsc};
use crate::problem::ConeSpec;
use crate::scaling::ScalingBlock;

/// Unified KKT solver that auto-selects between standard and normal equations.
pub enum UnifiedKktSolver {
    /// Standard augmented KKT system (for general problems)
    Standard(KktSolver),
    /// Normal equations for tall problems (m >> n with diagonal H)
    NormalEqns(NormalEqnsSolverWrapper),
}

/// Wrapper around NormalEqnsSolver with additional state for the unified interface.
pub struct NormalEqnsSolverWrapper {
    solver: NormalEqnsSolver,
    n: usize,
    m: usize,
    /// Current H diagonal values
    h_diag: Vec<f64>,
    /// Factorization token (just a marker)
    factored: bool,
}

/// Check if problem is suitable for normal equations.
///
/// Returns true if:
/// - m > 5*n (tall problem)
/// - n <= 500 (dense ops are fast)
/// - All cones are Zero or NonNeg (diagonal H)
pub fn should_use_normal_equations(n: usize, m: usize, cones: &[ConeSpec]) -> bool {
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
            eprintln!(
                "Using normal equations solver (n={}, m={}, ratio={:.1}x)",
                n, m, m as f64 / n as f64
            );
            let solver = NormalEqnsSolver::new(n, m, p, a, static_reg);
            UnifiedKktSolver::NormalEqns(NormalEqnsSolverWrapper {
                solver,
                n,
                m,
                h_diag: vec![1.0; m],
                factored: false,
            })
        } else {
            let kkt = KktSolver::new_with_singleton_elimination(
                n,
                m,
                static_reg,
                dynamic_reg_min_pivot,
                a,
                h_blocks,
            );
            // Initialize will be called separately
            UnifiedKktSolver::Standard(kkt)
        }
    }

    /// Initialize the solver (symbolic factorization for standard, no-op for normal eqns).
    pub fn initialize(
        &mut self,
        p: Option<&SparseSymmetricCsc>,
        a: &SparseCsc,
        h_blocks: &[ScalingBlock],
    ) -> Result<(), String> {
        match self {
            UnifiedKktSolver::Standard(kkt) => {
                kkt.initialize(p, a, h_blocks)
                    .map_err(|e| format!("{}", e))
            }
            UnifiedKktSolver::NormalEqns(_) => Ok(()), // Already initialized in new()
        }
    }

    /// Update numeric values and factorize.
    pub fn update_and_factor(
        &mut self,
        p: Option<&SparseSymmetricCsc>,
        a: &SparseCsc,
        h_blocks: &[ScalingBlock],
    ) -> Result<(), String> {
        match self {
            UnifiedKktSolver::Standard(kkt) => {
                kkt.update_numeric(p, a, h_blocks)
                    .map_err(|e| format!("{}", e))?;
                kkt.factorize().map_err(|e| format!("{}", e))?;
                Ok(())
            }
            UnifiedKktSolver::NormalEqns(wrapper) => {
                // Extract H diagonal from scaling blocks
                let mut offset = 0;
                for block in h_blocks {
                    match block {
                        ScalingBlock::Zero { dim } => {
                            // Zero cone: H is effectively infinite, H^{-1} = 0
                            for i in 0..*dim {
                                wrapper.h_diag[offset + i] = 1e20; // Large value
                            }
                            offset += dim;
                        }
                        ScalingBlock::Diagonal { d } => {
                            for (i, &val) in d.iter().enumerate() {
                                wrapper.h_diag[offset + i] = val;
                            }
                            offset += d.len();
                        }
                        _ => {
                            return Err("Normal equations only supports Zero and NonNeg cones".to_string());
                        }
                    }
                }

                wrapper.solver.update_and_factor(&wrapper.h_diag)?;
                wrapper.factored = true;
                Ok(())
            }
        }
    }

    /// Solve the KKT system.
    pub fn solve(
        &mut self,
        rhs_x: &[f64],
        rhs_z: &[f64],
        sol_x: &mut [f64],
        sol_z: &mut [f64],
    ) {
        match self {
            UnifiedKktSolver::Standard(kkt) => {
                // For standard solver, we need the factor token
                // This is a simplified interface - full integration would pass the factor
                let factor = kkt.factorize().expect("Factorization failed");
                kkt.solve(&factor, rhs_x, rhs_z, sol_x, sol_z);
            }
            UnifiedKktSolver::NormalEqns(wrapper) => {
                assert!(wrapper.factored, "Must call update_and_factor first");
                wrapper.solver.solve(&wrapper.h_diag, rhs_x, rhs_z, sol_x, sol_z);
            }
        }
    }

    /// Solve with two RHS vectors (for predictor-corrector).
    pub fn solve_two_rhs(
        &mut self,
        rhs_x1: &[f64],
        rhs_z1: &[f64],
        rhs_x2: &[f64],
        rhs_z2: &[f64],
        sol_x1: &mut [f64],
        sol_z1: &mut [f64],
        sol_x2: &mut [f64],
        sol_z2: &mut [f64],
    ) {
        match self {
            UnifiedKktSolver::Standard(kkt) => {
                let factor = kkt.factorize().expect("Factorization failed");
                kkt.solve_two_rhs(
                    &factor,
                    rhs_x1, rhs_z1,
                    rhs_x2, rhs_z2,
                    sol_x1, sol_z1,
                    sol_x2, sol_z2,
                );
            }
            UnifiedKktSolver::NormalEqns(wrapper) => {
                assert!(wrapper.factored, "Must call update_and_factor first");
                wrapper.solver.solve(&wrapper.h_diag, rhs_x1, rhs_z1, sol_x1, sol_z1);
                wrapper.solver.solve(&wrapper.h_diag, rhs_x2, rhs_z2, sol_x2, sol_z2);
            }
        }
    }

    /// Get static regularization value.
    pub fn static_reg(&self) -> f64 {
        match self {
            UnifiedKktSolver::Standard(kkt) => kkt.static_reg(),
            UnifiedKktSolver::NormalEqns(_) => 1e-8, // Default
        }
    }

    /// Set static regularization value.
    pub fn set_static_reg(&mut self, reg: f64) -> Result<(), String> {
        match self {
            UnifiedKktSolver::Standard(kkt) => {
                kkt.set_static_reg(reg).map_err(|e| format!("{}", e))
            }
            UnifiedKktSolver::NormalEqns(_) => Ok(()), // Normal eqns uses fixed reg
        }
    }

    /// Check if using normal equations.
    pub fn is_normal_equations(&self) -> bool {
        matches!(self, UnifiedKktSolver::NormalEqns(_))
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
