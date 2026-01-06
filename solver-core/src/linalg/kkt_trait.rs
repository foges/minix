//! KKT solver trait for unified interface.
//!
//! This trait abstracts over different KKT solving strategies:
//! - Standard augmented KKT system (sparse LDL)
//! - Normal equations for tall problems (dense Cholesky)

use super::backend::BackendError;
use super::sparse::{SparseCsc, SparseSymmetricCsc};
use crate::scaling::ScalingBlock;

/// Trait for KKT system solvers.
///
/// This provides a common interface for the predictor-corrector algorithm
/// to use different linear algebra backends.
pub trait KktSolverTrait {
    /// Factor type returned by factorize().
    type Factor;

    /// Perform symbolic factorization (one-time setup).
    fn initialize(
        &mut self,
        p: Option<&SparseSymmetricCsc>,
        a: &SparseCsc,
        h_blocks: &[ScalingBlock],
    ) -> Result<(), BackendError>;

    /// Update numeric values in the KKT matrix.
    fn update_numeric(
        &mut self,
        p: Option<&SparseSymmetricCsc>,
        a: &SparseCsc,
        h_blocks: &[ScalingBlock],
    ) -> Result<(), BackendError>;

    /// Compute numeric factorization.
    fn factorize(&mut self) -> Result<Self::Factor, BackendError>;

    /// Solve a single KKT system with refinement.
    fn solve_refined(
        &mut self,
        factor: &Self::Factor,
        rhs_x: &[f64],
        rhs_z: &[f64],
        sol_x: &mut [f64],
        sol_z: &mut [f64],
        refine_iters: usize,
    );

    /// Solve two KKT systems with the same factorization (for predictor-corrector).
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
    );

    /// Get static regularization value.
    fn static_reg(&self) -> f64;

    /// Set static regularization value.
    fn set_static_reg(&mut self, reg: f64) -> Result<(), BackendError>;

    /// Increase static regularization to at least min_reg.
    fn bump_static_reg(&mut self, min_reg: f64) -> Result<bool, BackendError>;

    /// Get count of dynamic regularization bumps.
    fn dynamic_bumps(&self) -> u64;
}
