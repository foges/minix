//! Cone kernel trait definition.
//!
//! This module defines the core interface that all cone implementations must satisfy.
//! The trait provides barrier function evaluations, interior tests, step-to-boundary
//! calculations, and initialization points.

/// Core cone kernel interface.
///
/// All cone types (Zero, NonNeg, SOC, EXP, POW, PSD) must implement this trait
/// to be used in the IPM solver. The trait methods are designed to be
/// allocation-free and suitable for performance-critical inner loops.
///
/// # Coordinate Convention
///
/// All methods operate on contiguous slices of the global s/z vectors.
/// The cone kernel is responsible for a specific range [offset .. offset+dim].
///
/// # Barrier Function
///
/// Each cone (except Zero) has a logarithmically homogeneous self-concordant
/// barrier function f(s). The methods provide:
/// - `barrier_value(s)`: compute f(s)
/// - `barrier_grad_primal(s, grad)`: compute ∇f(s)
/// - `barrier_hess_apply_primal(s, v, out)`: compute ∇²f(s) * v
///
/// For nonsymmetric cones (EXP, POW), dual barrier methods are also provided.
///
/// # Safety and Numerical Stability
///
/// - All barrier methods assume s is in the **strict interior** of the cone.
/// - Callers must check `is_interior_primal` before calling barrier methods.
/// - Implementations should use numerically stable formulas to avoid overflow/underflow.
pub trait ConeKernel: Send + Sync + std::any::Any {
    // ========================================================================
    // Basic properties
    // ========================================================================

    /// Dimension of this cone in the m-dimensional slack/dual space.
    fn dim(&self) -> usize;

    /// Barrier degree ν for this cone (used in μ calculation).
    ///
    /// - Zero: 0
    /// - NonNeg(n): n
    /// - SOC: 2 (regardless of dimension)
    /// - PSD(n): n
    /// - EXP: 3 (per block)
    /// - POW: 3 (per block)
    fn barrier_degree(&self) -> usize;

    // ========================================================================
    // Interior tests
    // ========================================================================

    /// Check if s is in the strict interior of the primal cone K.
    ///
    /// Returns true if s ∈ int(K), with a safety margin for numerical stability.
    ///
    /// # Safety margin
    ///
    /// Implementations should use a tolerance relative to ||s|| to avoid
    /// boundary issues. A typical margin is 1e-12 * max(1, ||s||).
    fn is_interior_primal(&self, s: &[f64]) -> bool;

    /// Check if z is in the strict interior of the dual cone K*.
    ///
    /// For self-dual cones (Zero, NonNeg, SOC, PSD), this is the same as
    /// the primal interior test. For nonsymmetric cones (EXP, POW), this
    /// uses the dual cone definition.
    fn is_interior_dual(&self, z: &[f64]) -> bool;

    // ========================================================================
    // Step-to-boundary
    // ========================================================================

    /// Compute maximum step size α such that s + α * ds remains in int(K).
    ///
    /// Returns α_max ∈ [0, ∞). If the direction ds points into the interior,
    /// returns +∞ (represented as f64::INFINITY).
    ///
    /// # Requirements
    ///
    /// - s must be in int(K) (checked by caller)
    /// - α_max is computed so that s + α_max * ds is **on the boundary** of K
    /// - The IPM will then apply a safety factor (e.g., 0.99 * α_max)
    fn step_to_boundary_primal(&self, s: &[f64], ds: &[f64]) -> f64;

    /// Compute maximum step size α such that z + α * dz remains in int(K*).
    fn step_to_boundary_dual(&self, z: &[f64], dz: &[f64]) -> f64;

    // ========================================================================
    // Barrier function (primal)
    // ========================================================================

    /// Evaluate the barrier function f(s).
    ///
    /// # Requirements
    ///
    /// - s must be in int(K)
    /// - Returns a finite value (NaN/Inf indicates a bug or numerical issue)
    fn barrier_value(&self, s: &[f64]) -> f64;

    /// Compute the barrier gradient ∇f(s).
    ///
    /// Writes the gradient to `grad_out` (same length as s).
    ///
    /// # Requirements
    ///
    /// - s must be in int(K)
    /// - grad_out.len() == s.len()
    fn barrier_grad_primal(&self, s: &[f64], grad_out: &mut [f64]);

    /// Compute the barrier Hessian-vector product ∇²f(s) * v.
    ///
    /// Writes the result to `out` (same length as s).
    ///
    /// # Requirements
    ///
    /// - s must be in int(K)
    /// - v.len() == s.len()
    /// - out.len() == s.len()
    ///
    /// # Implementation note
    ///
    /// This method should NOT materialize the full Hessian matrix.
    /// Instead, implement the matrix-vector product directly using
    /// the barrier structure (e.g., rank-1 updates for SOC).
    fn barrier_hess_apply_primal(&self, s: &[f64], v: &[f64], out: &mut [f64]);

    // ========================================================================
    // Barrier function (dual) - for nonsymmetric cones
    // ========================================================================

    /// Compute the dual barrier gradient ∇f*(z).
    ///
    /// For symmetric cones, this can delegate to the primal gradient.
    /// For nonsymmetric cones (EXP, POW), this requires the dual map oracle.
    ///
    /// Writes the gradient to `grad_out` (same length as z).
    fn barrier_grad_dual(&self, z: &[f64], grad_out: &mut [f64]);

    /// Compute the dual barrier Hessian-vector product ∇²f*(z) * v.
    ///
    /// Writes the result to `out` (same length as z).
    fn barrier_hess_apply_dual(&self, z: &[f64], v: &[f64], out: &mut [f64]);

    // ========================================================================
    // Dual map oracle (for nonsymmetric cones)
    // ========================================================================

    /// Compute the dual map for nonsymmetric cones.
    ///
    /// Given z ∈ int(K*), solve:
    ///     x_z = argmin_{x ∈ int(K)} { z^T x + f(x) }
    ///
    /// Returns:
    /// - x_out: the minimizer x_z (also equals -∇f*(z))
    /// - h_star: ∇²f*(z) as a 3×3 matrix (row-major) for 3D cones
    ///
    /// # For symmetric cones
    ///
    /// This method is not used (can panic or return dummy values).
    ///
    /// # For nonsymmetric cones (EXP, POW)
    ///
    /// This is computed via a small Newton solve with backtracking line search.
    /// The implementation should:
    /// - Warm-start from the previous iteration
    /// - Converge to tolerance ~1e-10
    /// - Use at most 10-20 Newton steps
    ///
    /// # Requirements
    ///
    /// - z must be in int(K*)
    /// - x_out.len() == 3 (for EXP/POW)
    /// - h_star.len() == 9 (3×3 row-major)
    fn dual_map(&self, z: &[f64], x_out: &mut [f64], h_star: &mut [f64; 9]);

    // ========================================================================
    // Initialization
    // ========================================================================

    /// Compute a well-centered unit initialization point (s₀, z₀).
    ///
    /// Returns interior points that are:
    /// 1. In int(K) × int(K*)
    /// 2. Well-centered (far from boundary)
    /// 3. Scaled appropriately for the cone
    ///
    /// # Initialization points (from design doc)
    ///
    /// - Zero: no initialization needed
    /// - NonNeg: s₀ = z₀ = ones
    /// - SOC: s₀ = z₀ = (1, 0, ..., 0)
    /// - PSD: s₀ = z₀ = I (identity in svec)
    /// - EXP: s₀ = z₀ = (-1.051383, 0.556409, 1.258967)
    /// - POW(α): s₀ = z₀ = (√(1+α), √(2-α), 0)
    ///
    /// # Requirements
    ///
    /// - s_out.len() == dim()
    /// - z_out.len() == dim()
    fn unit_initialization(&self, s_out: &mut [f64], z_out: &mut [f64]);
}
