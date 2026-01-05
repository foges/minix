//! Minix: A state-of-the-art convex optimization solver
//!
//! This library provides a production-grade implementation of an interior point method
//! for convex conic optimization problems. It supports:
//!
//! - **Linear Programming (LP)**: Zero and nonnegative cones
//! - **Quadratic Programming (QP)**: Convex quadratic objectives
//! - **Second-Order Cone Programming (SOCP)**: Lorentz cones
//! - **Exponential Cone Programming**: Relative entropy, logistic regression
//! - **Power Cone Programming**: Geometric programming
//! - **Semidefinite Programming (SDP)**: Positive semidefinite matrix constraints
//!
//! # Algorithm
//!
//! The solver uses a **homogeneous self-dual embedding (HSDE)** interior point method
//! with predictor-corrector steps. Key features:
//!
//! - **Nesterov-Todd scaling** for symmetric cones (LP, SOC, PSD)
//! - **BFGS primal-dual scaling** for nonsymmetric cones (EXP, POW)
//! - **Robust regularization** for quasi-definite KKT systems
//! - **Infeasibility certificates** for ill-posed problems
//!
//! # Example
//!
//! ```ignore
//! use solver_core::{ProblemData, ConeSpec, SolverSettings, solve};
//!
//! // Minimize 0.5 * x^T P x + q^T x
//! // subject to A x + s = b, s ∈ K
//!
//! let prob = ProblemData {
//!     P: None,  // LP (no quadratic term)
//!     q: vec![1.0, 1.0],
//!     A: /* sparse matrix */,
//!     b: vec![1.0],
//!     cones: vec![ConeSpec::NonNeg { dim: 1 }],
//!     var_bounds: None,
//!     integrality: None,
//! };
//!
//! let settings = SolverSettings::default();
//! let result = solve(&prob, &settings)?;
//!
//! println!("Status: {:?}", result.status);
//! println!("Optimal value: {}", result.obj_val);
//! println!("Solution: {:?}", result.x);
//! ```
//!
//! # References
//!
//! This implementation follows the design outlined in the accompanying
//! engineering specification document. Key algorithmic references:
//!
//! - Clarabel.rs: Interior point method for conic QPs
//! - MOSEK: Commercial-grade nonsymmetric cone handling
//! - ECOS: Embedded conic solver (baseline for comparison)

#![allow(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]  // IPM algorithms need many parameters

pub mod problem;
pub mod cones;
pub mod scaling;
pub mod linalg;
pub mod ipm;
pub mod ipm2;
pub mod presolve;
pub mod postsolve;
pub mod util;

// Re-export main types
pub use problem::{
    ProblemData, ConeSpec, Pow3D, VarBound, VarType,
    SolverSettings, SolveResult, SolveStatus, SolveInfo, WarmStart,
};

/// Main solve entry point.
///
/// Solves a convex conic optimization problem.
///
/// # Example
///
/// ```ignore
/// use solver_core::{ProblemData, ConeSpec, SolverSettings, solve};
/// use solver_core::linalg::sparse;
///
/// // min x1 + x2 s.t. x1 + x2 = 1
/// let prob = ProblemData {
///     P: None,
///     q: vec![1.0, 1.0],
///     A: sparse::from_triplets(1, 2, vec![(0, 0, 1.0), (0, 1, 1.0)]),
///     b: vec![1.0],
///     cones: vec![ConeSpec::Zero { dim: 1 }],
///     var_bounds: None,
///     integrality: None,
/// };
///
/// let settings = SolverSettings::default();
/// let result = solve(&prob, &settings)?;
/// ```
pub fn solve(
    problem: &ProblemData,
    settings: &SolverSettings,
) -> Result<SolveResult, Box<dyn std::error::Error>> {
    // ipm2 is the active development track. Keep ipm1 for A/B/regression,
    // but route the default entry point to ipm2.
    ipm2::solve_ipm2(problem, settings)
}
