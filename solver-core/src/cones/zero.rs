//! Zero cone: equality constraints.
//!
//! The zero cone K = {0}^n represents equality constraints in the optimization problem.
//! It has no interior points (except trivially s=0) and no barrier function.
//! Special handling is required in the KKT system.

use super::traits::ConeKernel;

/// Zero cone for equality constraints.
///
/// The zero cone {0}^n is used to represent equality constraints A x = b.
/// Since there are no interior points, barrier-related methods should not be called.
#[derive(Debug, Clone)]
pub struct ZeroCone {
    /// Dimension of the zero cone
    dim: usize,
}

impl ZeroCone {
    /// Create a new zero cone of the given dimension
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "Zero cone must have positive dimension");
        Self { dim }
    }
}

impl ConeKernel for ZeroCone {
    fn dim(&self) -> usize {
        self.dim
    }

    fn barrier_degree(&self) -> usize {
        0 // No barrier for zero cone
    }

    fn is_interior_primal(&self, _s: &[f64]) -> bool {
        // Zero cone has no interior (only s=0 is in the cone)
        false
    }

    fn is_interior_dual(&self, _z: &[f64]) -> bool {
        // Dual of zero cone is all of ℝ^n, so always interior
        true
    }

    fn step_to_boundary_primal(&self, _s: &[f64], _ds: &[f64]) -> f64 {
        // No interior, no step to take
        0.0
    }

    fn step_to_boundary_dual(&self, _z: &[f64], _dz: &[f64]) -> f64 {
        // Dual cone is all of ℝ^n, no boundary
        f64::INFINITY
    }

    fn barrier_value(&self, _s: &[f64]) -> f64 {
        // No barrier for zero cone
        panic!("Zero cone has no barrier function");
    }

    fn barrier_grad_primal(&self, _s: &[f64], _grad_out: &mut [f64]) {
        panic!("Zero cone has no barrier function");
    }

    fn barrier_hess_apply_primal(&self, _s: &[f64], _v: &[f64], _out: &mut [f64]) {
        panic!("Zero cone has no barrier function");
    }

    fn barrier_grad_dual(&self, _z: &[f64], _grad_out: &mut [f64]) {
        panic!("Zero cone has no dual barrier function");
    }

    fn barrier_hess_apply_dual(&self, _z: &[f64], _v: &[f64], _out: &mut [f64]) {
        panic!("Zero cone has no dual barrier function");
    }

    fn dual_map(&self, _z: &[f64], _x_out: &mut [f64], _h_star: &mut [f64; 9]) {
        panic!("Zero cone has no dual map");
    }

    fn unit_initialization(&self, s_out: &mut [f64], z_out: &mut [f64]) {
        // Initialize to zero (though this will be handled specially in the IPM)
        for i in 0..self.dim {
            s_out[i] = 0.0;
            z_out[i] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_cone_basic() {
        let cone = ZeroCone::new(5);
        assert_eq!(cone.dim(), 5);
        assert_eq!(cone.barrier_degree(), 0);
    }

    #[test]
    fn test_zero_cone_interior() {
        let cone = ZeroCone::new(3);
        let s = vec![0.0, 0.0, 0.0];
        let z = vec![1.0, 2.0, 3.0];

        // Zero cone has no interior
        assert!(!cone.is_interior_primal(&s));

        // Dual cone is all of ℝ^n
        assert!(cone.is_interior_dual(&z));
    }

    #[test]
    fn test_zero_cone_initialization() {
        let cone = ZeroCone::new(4);
        let mut s = vec![0.0; 4];
        let mut z = vec![0.0; 4];

        cone.unit_initialization(&mut s, &mut z);

        // Should initialize to zeros
        assert_eq!(s, vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(z, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "Zero cone has no barrier function")]
    fn test_zero_cone_barrier_panics() {
        let cone = ZeroCone::new(3);
        let s = vec![0.0, 0.0, 0.0];
        cone.barrier_value(&s);
    }
}
