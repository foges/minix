//! Exponential cone.
//!
//! Not yet implemented.

use super::traits::ConeKernel;

/// Exponential cone (placeholder)
#[derive(Debug, Clone)]
pub struct ExpCone {
    count: usize,
}

impl ExpCone {
    /// Create a new exponential cone with `count` 3D blocks
    pub fn new(count: usize) -> Self {
        Self { count }
    }
}

impl ConeKernel for ExpCone {
    fn dim(&self) -> usize {
        3 * self.count
    }
    fn barrier_degree(&self) -> usize {
        3 * self.count
    }
    fn is_interior_primal(&self, _s: &[f64]) -> bool {
        unimplemented!()
    }
    fn is_interior_dual(&self, _z: &[f64]) -> bool {
        unimplemented!()
    }
    fn step_to_boundary_primal(&self, _s: &[f64], _ds: &[f64]) -> f64 {
        unimplemented!()
    }
    fn step_to_boundary_dual(&self, _z: &[f64], _dz: &[f64]) -> f64 {
        unimplemented!()
    }
    fn barrier_value(&self, _s: &[f64]) -> f64 {
        unimplemented!()
    }
    fn barrier_grad_primal(&self, _s: &[f64], _grad_out: &mut [f64]) {
        unimplemented!()
    }
    fn barrier_hess_apply_primal(&self, _s: &[f64], _v: &[f64], _out: &mut [f64]) {
        unimplemented!()
    }
    fn barrier_grad_dual(&self, _z: &[f64], _grad_out: &mut [f64]) {
        unimplemented!()
    }
    fn barrier_hess_apply_dual(&self, _z: &[f64], _v: &[f64], _out: &mut [f64]) {
        unimplemented!()
    }
    fn dual_map(&self, _z: &[f64], _x_out: &mut [f64], _h_star: &mut [f64; 9]) {
        unimplemented!()
    }
    fn unit_initialization(&self, _s_out: &mut [f64], _z_out: &mut [f64]) {
        unimplemented!()
    }
}
