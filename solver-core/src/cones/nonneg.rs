//! Nonnegative orthant cone.
//!
//! The nonnegative cone K = ℝ₊^n = {s : s_i ≥ 0 for all i} is the simplest
//! self-dual cone with a barrier function.
//!
//! # Barrier Function
//!
//! f(s) = -∑ᵢ log(s_i)
//!
//! This is the standard logarithmic barrier for the nonnegative orthant.
//!
//! # Derivatives
//!
//! - Gradient: (∇f)_i = -1/s_i
//! - Hessian: (∇²f)_{ij} = δ_{ij} / s_i²
//!
//! The Hessian is diagonal, making all operations very efficient.

use super::traits::ConeKernel;

/// Nonnegative orthant cone ℝ₊^n.
///
/// This cone represents simple nonnegativity constraints s ≥ 0.
#[derive(Debug, Clone)]
pub struct NonNegCone {
    /// Dimension of the cone
    dim: usize,
}

impl NonNegCone {
    /// Create a new nonnegative cone of the given dimension.
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "NonNeg cone must have positive dimension");
        Self { dim }
    }

    /// Interior tolerance: s_i > tol * max(1, ||s||_∞)
    const INTERIOR_TOL: f64 = 1e-12;
}

impl ConeKernel for NonNegCone {
    fn dim(&self) -> usize {
        self.dim
    }

    fn barrier_degree(&self) -> usize {
        self.dim  // ν = n for ℝ₊^n
    }

    fn is_interior_primal(&self, s: &[f64]) -> bool {
        assert_eq!(s.len(), self.dim);

        // Check for NaN
        if s.iter().any(|&x| x.is_nan()) {
            return false;
        }

        // Compute tolerance relative to ||s||_∞
        let s_max = s.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        let tol = Self::INTERIOR_TOL * s_max.max(1.0);

        // All components must be > tol
        s.iter().all(|&x| x > tol)
    }

    fn is_interior_dual(&self, z: &[f64]) -> bool {
        // NonNeg cone is self-dual
        self.is_interior_primal(z)
    }

    fn step_to_boundary_primal(&self, s: &[f64], ds: &[f64]) -> f64 {
        assert_eq!(s.len(), self.dim);
        assert_eq!(ds.len(), self.dim);

        let mut alpha_max = f64::INFINITY;

        for i in 0..self.dim {
            if ds[i] < 0.0 {
                // Need s_i + α ds_i > 0
                // α < -s_i / ds_i
                let alpha_i = -s[i] / ds[i];
                alpha_max = alpha_max.min(alpha_i);
            }
            // If ds[i] >= 0, no constraint from this component
        }

        alpha_max
    }

    fn step_to_boundary_dual(&self, z: &[f64], dz: &[f64]) -> f64 {
        // Self-dual
        self.step_to_boundary_primal(z, dz)
    }

    fn barrier_value(&self, s: &[f64]) -> f64 {
        assert_eq!(s.len(), self.dim);

        // f(s) = -∑ log(s_i)
        s.iter().map(|&x| -x.ln()).sum()
    }

    fn barrier_grad_primal(&self, s: &[f64], grad_out: &mut [f64]) {
        assert_eq!(s.len(), self.dim);
        assert_eq!(grad_out.len(), self.dim);

        // ∇f = -1 ./ s (elementwise)
        for i in 0..self.dim {
            grad_out[i] = -1.0 / s[i];
        }
    }

    fn barrier_hess_apply_primal(&self, s: &[f64], v: &[f64], out: &mut [f64]) {
        assert_eq!(s.len(), self.dim);
        assert_eq!(v.len(), self.dim);
        assert_eq!(out.len(), self.dim);

        // ∇²f is diagonal: (∇²f)_{ii} = 1/s_i²
        // (∇²f v)_i = v_i / s_i²
        for i in 0..self.dim {
            out[i] = v[i] / (s[i] * s[i]);
        }
    }

    fn barrier_grad_dual(&self, z: &[f64], grad_out: &mut [f64]) {
        // Self-dual
        self.barrier_grad_primal(z, grad_out)
    }

    fn barrier_hess_apply_dual(&self, z: &[f64], v: &[f64], out: &mut [f64]) {
        // Self-dual
        self.barrier_hess_apply_primal(z, v, out)
    }

    fn dual_map(&self, _z: &[f64], _x_out: &mut [f64], _h_star: &mut [f64; 9]) {
        panic!("NonNeg cone is self-dual; dual_map not needed");
    }

    fn unit_initialization(&self, s_out: &mut [f64], z_out: &mut [f64]) {
        assert_eq!(s_out.len(), self.dim);
        assert_eq!(z_out.len(), self.dim);

        // Initialize to ones
        for i in 0..self.dim {
            s_out[i] = 1.0;
            z_out[i] = 1.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nonneg_basic() {
        let cone = NonNegCone::new(5);
        assert_eq!(cone.dim(), 5);
        assert_eq!(cone.barrier_degree(), 5);
    }

    #[test]
    fn test_nonneg_interior() {
        let cone = NonNegCone::new(3);

        // Interior points
        assert!(cone.is_interior_primal(&[1.0, 2.0, 3.0]));
        assert!(cone.is_interior_primal(&[0.1, 0.1, 0.1]));

        // Boundary points (should fail with tolerance)
        assert!(!cone.is_interior_primal(&[0.0, 1.0, 1.0]));
        assert!(!cone.is_interior_primal(&[1.0, 0.0, 1.0]));

        // Exterior points
        assert!(!cone.is_interior_primal(&[-1.0, 1.0, 1.0]));
        assert!(!cone.is_interior_primal(&[1.0, -0.5, 1.0]));

        // NaN
        assert!(!cone.is_interior_primal(&[f64::NAN, 1.0, 1.0]));
    }

    #[test]
    fn test_nonneg_step_to_boundary() {
        let cone = NonNegCone::new(3);

        // Test case 1: moving away from boundary
        let s = vec![1.0, 2.0, 3.0];
        let ds = vec![1.0, 1.0, 1.0];
        assert_eq!(cone.step_to_boundary_primal(&s, &ds), f64::INFINITY);

        // Test case 2: moving toward boundary
        let ds = vec![-0.5, -1.0, -2.0];
        let alpha = cone.step_to_boundary_primal(&s, &ds);
        // Most restrictive: s[1] + α ds[1] = 0 → 2 - α = 0 → α = 2
        // Also: s[0] + α ds[0] = 0 → 1 - 0.5α = 0 → α = 2
        // Also: s[2] + α ds[2] = 0 → 3 - 2α = 0 → α = 1.5
        // So α_max = 1.5
        assert!((alpha - 1.5).abs() < 1e-10);

        // Test case 3: mixed directions
        let ds = vec![1.0, -1.0, 0.0];
        let alpha = cone.step_to_boundary_primal(&s, &ds);
        // Only constraint from ds[1] < 0: 2 - α = 0 → α = 2
        assert_eq!(alpha, 2.0);
    }

    #[test]
    fn test_nonneg_barrier_value() {
        let cone = NonNegCone::new(3);

        let s = vec![1.0, 1.0, 1.0];
        let f = cone.barrier_value(&s);
        // f = -log(1) - log(1) - log(1) = 0
        assert!((f - 0.0).abs() < 1e-10);

        let s = vec![2.0, 2.0, 2.0];
        let f = cone.barrier_value(&s);
        // f = -3 * log(2)
        let expected = -3.0 * 2.0f64.ln();
        assert!((f - expected).abs() < 1e-10);
    }

    #[test]
    fn test_nonneg_barrier_gradient() {
        let cone = NonNegCone::new(3);

        let s = vec![1.0, 2.0, 4.0];
        let mut grad = vec![0.0; 3];
        cone.barrier_grad_primal(&s, &mut grad);

        // ∇f = [-1/s_i] = [-1, -0.5, -0.25]
        assert!((grad[0] - (-1.0)).abs() < 1e-10);
        assert!((grad[1] - (-0.5)).abs() < 1e-10);
        assert!((grad[2] - (-0.25)).abs() < 1e-10);
    }

    #[test]
    fn test_nonneg_barrier_hessian() {
        let cone = NonNegCone::new(3);

        let s = vec![1.0, 2.0, 4.0];
        let v = vec![1.0, 1.0, 1.0];
        let mut out = vec![0.0; 3];
        cone.barrier_hess_apply_primal(&s, &v, &mut out);

        // (∇²f v)_i = v_i / s_i² = [1/1, 1/4, 1/16]
        assert!((out[0] - 1.0).abs() < 1e-10);
        assert!((out[1] - 0.25).abs() < 1e-10);
        assert!((out[2] - 0.0625).abs() < 1e-10);
    }

    #[test]
    fn test_nonneg_initialization() {
        let cone = NonNegCone::new(4);
        let mut s = vec![0.0; 4];
        let mut z = vec![0.0; 4];

        cone.unit_initialization(&mut s, &mut z);

        assert_eq!(s, vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(z, vec![1.0, 1.0, 1.0, 1.0]);

        // Verify they're interior
        assert!(cone.is_interior_primal(&s));
        assert!(cone.is_interior_dual(&z));
    }

    #[test]
    fn test_nonneg_self_dual() {
        let cone = NonNegCone::new(3);
        let s = vec![1.0, 2.0, 3.0];

        // Interior test should be the same for primal and dual
        assert_eq!(
            cone.is_interior_primal(&s),
            cone.is_interior_dual(&s)
        );

        // Step-to-boundary should be the same
        let ds = vec![-0.5, -1.0, -0.5];
        assert_eq!(
            cone.step_to_boundary_primal(&s, &ds),
            cone.step_to_boundary_dual(&s, &ds)
        );
    }
}
