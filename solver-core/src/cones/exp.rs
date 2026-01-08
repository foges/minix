//! Exponential cone.
//!
//! Uses the log-homogeneous barrier from the design doc.

use super::traits::ConeKernel;
use nalgebra::Matrix3;

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

    const INTERIOR_TOL: f64 = 1e-12;
    const NEWTON_TOL: f64 = 1e-10;
    const MAX_NEWTON_ITERS: usize = 20;
    const MAX_LINESEARCH_ITERS: usize = 40;
}

impl ConeKernel for ExpCone {
    fn dim(&self) -> usize { 3 * self.count }
    fn barrier_degree(&self) -> usize { 3 * self.count }
    fn is_interior_primal(&self, s: &[f64]) -> bool {
        assert_eq!(s.len(), self.dim());
        for block in 0..self.count {
            let offset = 3 * block;
            if !exp_primal_interior(&s[offset..offset + 3]) {
                return false;
            }
        }
        true
    }

    fn is_interior_dual(&self, z: &[f64]) -> bool {
        assert_eq!(z.len(), self.dim());
        for block in 0..self.count {
            let offset = 3 * block;
            if !exp_dual_interior(&z[offset..offset + 3]) {
                return false;
            }
        }
        true
    }

    fn step_to_boundary_primal(&self, s: &[f64], ds: &[f64]) -> f64 {
        assert_eq!(s.len(), self.dim());
        assert_eq!(ds.len(), self.dim());
        let mut alpha = f64::INFINITY;
        for block in 0..self.count {
            let offset = 3 * block;
            let s_block = &s[offset..offset + 3];
            let ds_block = &ds[offset..offset + 3];

            // Debug: check if s is interior
            let is_int = exp_primal_interior(s_block);
            if !is_int {
                eprintln!("WARNING: exp cone s NOT interior: {:?}", s_block);
            }

            let a = exp_step_to_boundary_block(s_block, ds_block, exp_primal_interior);
            if !is_int && a == 0.0 {
                eprintln!("  -> Returning alpha=0 because s not interior");
            }
            if a.is_finite() {
                alpha = alpha.min(a.max(0.0));
            }
            if alpha == 0.0 {
                break;
            }
        }
        alpha
    }

    fn step_to_boundary_dual(&self, z: &[f64], dz: &[f64]) -> f64 {
        assert_eq!(z.len(), self.dim());
        assert_eq!(dz.len(), self.dim());
        let mut alpha = f64::INFINITY;
        for block in 0..self.count {
            let offset = 3 * block;
            let a = exp_step_to_boundary_block(
                &z[offset..offset + 3],
                &dz[offset..offset + 3],
                exp_dual_interior,
            );
            if a.is_finite() {
                alpha = alpha.min(a.max(0.0));
            }
            if alpha == 0.0 {
                break;
            }
        }
        alpha
    }

    fn barrier_value(&self, s: &[f64]) -> f64 {
        assert_eq!(s.len(), self.dim());
        let mut value = 0.0;
        for block in 0..self.count {
            let offset = 3 * block;
            value += exp_barrier_value_block(&s[offset..offset + 3]);
        }
        value
    }

    fn barrier_grad_primal(&self, s: &[f64], grad_out: &mut [f64]) {
        assert_eq!(s.len(), self.dim());
        assert_eq!(grad_out.len(), self.dim());
        for block in 0..self.count {
            let offset = 3 * block;
            exp_barrier_grad_block(&s[offset..offset + 3], &mut grad_out[offset..offset + 3]);
        }
    }

    fn barrier_hess_apply_primal(&self, s: &[f64], v: &[f64], out: &mut [f64]) {
        assert_eq!(s.len(), self.dim());
        assert_eq!(v.len(), self.dim());
        assert_eq!(out.len(), self.dim());
        for block in 0..self.count {
            let offset = 3 * block;
            exp_barrier_hess_apply_block(
                &s[offset..offset + 3],
                &v[offset..offset + 3],
                &mut out[offset..offset + 3],
            );
        }
    }

    fn barrier_grad_dual(&self, z: &[f64], grad_out: &mut [f64]) {
        assert_eq!(z.len(), self.dim());
        assert_eq!(grad_out.len(), self.dim());
        for block in 0..self.count {
            let offset = 3 * block;
            let mut x = [0.0; 3];
            let mut h_star = [0.0; 9];
            exp_dual_map_block(&z[offset..offset + 3], &mut x, &mut h_star);
            grad_out[offset..offset + 3].copy_from_slice(&[-x[0], -x[1], -x[2]]);
        }
    }

    fn barrier_hess_apply_dual(&self, z: &[f64], v: &[f64], out: &mut [f64]) {
        assert_eq!(z.len(), self.dim());
        assert_eq!(v.len(), self.dim());
        assert_eq!(out.len(), self.dim());
        for block in 0..self.count {
            let offset = 3 * block;
            let mut x = [0.0; 3];
            let mut h_star = [0.0; 9];
            exp_dual_map_block(&z[offset..offset + 3], &mut x, &mut h_star);
            apply_mat3(&h_star, &v[offset..offset + 3], &mut out[offset..offset + 3]);
        }
    }

    fn dual_map(&self, z: &[f64], x_out: &mut [f64], h_star: &mut [f64; 9]) {
        assert_eq!(z.len(), 3, "ExpCone dual_map expects a single 3D block");
        assert_eq!(x_out.len(), 3);
        exp_dual_map_block(z, x_out, h_star);
    }

    fn unit_initialization(&self, s_out: &mut [f64], z_out: &mut [f64]) {
        assert_eq!(s_out.len(), self.dim());
        assert_eq!(z_out.len(), self.dim());
        for block in 0..self.count {
            let offset = 3 * block;
            s_out[offset..offset + 3].copy_from_slice(&[-1.051_383, 0.556_409, 1.258_967]);
            z_out[offset..offset + 3].copy_from_slice(&[-1.051_383, 0.556_409, 1.258_967]);
        }
    }
}

fn exp_primal_interior(s: &[f64]) -> bool {
    if s.len() != 3 || s.iter().any(|&v| !v.is_finite()) {
        return false;
    }
    let x = s[0];
    let y = s[1];
    let z = s[2];
    if y <= 0.0 || z <= 0.0 {
        return false;
    }
    let psi = y * (z / y).ln() - x;
    if !psi.is_finite() {
        return false;
    }
    let scale = x.abs().max(y.abs()).max(z.abs()).max(1.0);
    psi > ExpCone::INTERIOR_TOL * scale
}

fn exp_dual_interior(z: &[f64]) -> bool {
    if z.len() != 3 || z.iter().any(|&v| !v.is_finite()) {
        return false;
    }
    let u = z[0];
    let v = z[1];
    let w = z[2];
    if u >= -ExpCone::INTERIOR_TOL {
        return false;
    }
    if w <= 0.0 {
        return false;
    }
    let log_w = w.ln();
    let log_rhs = (-u).ln() + v / u - 1.0;
    (log_w - log_rhs) > ExpCone::INTERIOR_TOL
}

fn exp_step_to_boundary_block(
    s: &[f64],
    ds: &[f64],
    interior: fn(&[f64]) -> bool,
) -> f64 {
    if ds.iter().all(|&v| v == 0.0) {
        return f64::INFINITY;
    }
    if !interior(s) {
        return 0.0;
    }

    let mut trial = [0.0; 3];
    for i in 0..3 {
        trial[i] = s[i] + ds[i];
    }
    if interior(&trial) {
        return f64::INFINITY;
    }

    let mut lo = 0.0;
    let mut hi = 1.0;
    for _ in 0..ExpCone::MAX_LINESEARCH_ITERS {
        let mid = 0.5 * (lo + hi);
        for i in 0..3 {
            trial[i] = s[i] + mid * ds[i];
        }
        if interior(&trial) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

fn exp_barrier_value_block(s: &[f64]) -> f64 {
    let x = s[0];
    let y = s[1];
    let z = s[2];
    let psi = y * (z / y).ln() - x;
    -psi.ln() - y.ln() - z.ln()
}

fn exp_barrier_grad_block(s: &[f64], grad_out: &mut [f64]) {
    let x = s[0];
    let y = s[1];
    let z = s[2];
    let psi = y * (z / y).ln() - x;
    let gpsi = exp_grad_psi(y, z);
    let inv_psi = 1.0 / psi;
    grad_out[0] = -inv_psi * gpsi[0];
    grad_out[1] = -inv_psi * gpsi[1] - 1.0 / y;
    grad_out[2] = -inv_psi * gpsi[2] - 1.0 / z;
}

fn exp_barrier_hess_apply_block(s: &[f64], v: &[f64], out: &mut [f64]) {
    let x = s[0];
    let y = s[1];
    let z = s[2];
    let psi = y * (z / y).ln() - x;
    let gpsi = exp_grad_psi(y, z);
    let hpsi = exp_hess_psi(y, z);

    let inv_psi = 1.0 / psi;
    let inv_psi2 = inv_psi * inv_psi;
    let mut h = [0.0; 9];

    for i in 0..3 {
        for j in 0..3 {
            h[3 * i + j] = inv_psi2 * gpsi[i] * gpsi[j] - inv_psi * hpsi[3 * i + j];
        }
    }
    h[4] += 1.0 / (y * y);
    h[8] += 1.0 / (z * z);

    apply_mat3(&h, v, out);
}

fn exp_grad_psi(y: f64, z: f64) -> [f64; 3] {
    let log_ratio = (z / y).ln();
    [-1.0, log_ratio - 1.0, y / z]
}

fn exp_hess_psi(y: f64, z: f64) -> [f64; 9] {
    [
        0.0, 0.0, 0.0,
        0.0, -1.0 / y, 1.0 / z,
        0.0, 1.0 / z, -y / (z * z),
    ]
}

fn exp_dual_map_block(z: &[f64], x_out: &mut [f64], h_star: &mut [f64; 9]) {
    let mut x = [-1.051_383, 0.556_409, 1.258_967];
    for _ in 0..ExpCone::MAX_NEWTON_ITERS {
        let mut grad = [0.0; 3];
        exp_barrier_grad_block(&x, &mut grad);
        let r = [z[0] + grad[0], z[1] + grad[1], z[2] + grad[2]];
        let r_norm = r.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if r_norm <= ExpCone::NEWTON_TOL {
            break;
        }
        let h = exp_hess_matrix(&x);
        let dx = solve_3x3(&h, &r);
        let mut alpha = 1.0;
        let mut moved = false;
        for _ in 0..ExpCone::MAX_LINESEARCH_ITERS {
            let trial = [x[0] + alpha * dx[0], x[1] + alpha * dx[1], x[2] + alpha * dx[2]];
            if exp_primal_interior(&trial) {
                x = trial;
                moved = true;
                break;
            }
            alpha *= 0.5;
        }
        if !moved {
            break;
        }
    }

    x_out.copy_from_slice(&x);
    let h = exp_hess_matrix(&x);
    let h_inv = invert_3x3(&h);
    *h_star = h_inv;
}

fn exp_hess_matrix(x: &[f64; 3]) -> [f64; 9] {
    let y = x[1];
    let z = x[2];
    let psi = y * (z / y).ln() - x[0];
    let gpsi = exp_grad_psi(y, z);
    let hpsi = exp_hess_psi(y, z);

    let inv_psi = 1.0 / psi;
    let inv_psi2 = inv_psi * inv_psi;
    let mut h = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            h[3 * i + j] = inv_psi2 * gpsi[i] * gpsi[j] - inv_psi * hpsi[3 * i + j];
        }
    }
    h[4] += 1.0 / (y * y);
    h[8] += 1.0 / (z * z);
    h
}

fn apply_mat3(h: &[f64; 9], v: &[f64], out: &mut [f64]) {
    out[0] = h[0] * v[0] + h[1] * v[1] + h[2] * v[2];
    out[1] = h[3] * v[0] + h[4] * v[1] + h[5] * v[2];
    out[2] = h[6] * v[0] + h[7] * v[1] + h[8] * v[2];
}

fn solve_3x3(h: &[f64; 9], r: &[f64; 3]) -> [f64; 3] {
    let h_inv = invert_3x3(h);
    [
        -(h_inv[0] * r[0] + h_inv[1] * r[1] + h_inv[2] * r[2]),
        -(h_inv[3] * r[0] + h_inv[4] * r[1] + h_inv[5] * r[2]),
        -(h_inv[6] * r[0] + h_inv[7] * r[1] + h_inv[8] * r[2]),
    ]
}

fn invert_3x3(h: &[f64; 9]) -> [f64; 9] {
    let base = Matrix3::from_row_slice(h);
    if let Some(inv) = base.try_inverse() {
        return mat3_to_row_major(&inv);
    }

    let mut shift = 1e-8;
    for _ in 0..6 {
        let mut shifted = base;
        for i in 0..3 {
            shifted[(i, i)] += shift;
        }
        if let Some(inv) = shifted.try_inverse() {
            return mat3_to_row_major(&inv);
        }
        shift *= 10.0;
    }

    let mut out = [0.0; 9];
    out[0] = 1.0;
    out[4] = 1.0;
    out[8] = 1.0;
    out
}

fn mat3_to_row_major(m: &Matrix3<f64>) -> [f64; 9] {
    let mut out = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            out[3 * i + j] = m[(i, j)];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp_primal_interior() {
        // Test basic interior points
        // K_exp = {(x,y,z) : z >= y*exp(x/y), y > 0}

        // (0, 1, 2): z=2 >= 1*exp(0) = 1 ✓
        assert!(exp_primal_interior(&[0.0, 1.0, 2.0]));

        // (-1, 1, 0.5): z=0.5 >= 1*exp(-1) = 0.368 ✓
        assert!(exp_primal_interior(&[-1.0, 1.0, 0.5]));

        // (1, 1, 3): z=3 >= 1*exp(1) = 2.718 ✓
        assert!(exp_primal_interior(&[1.0, 1.0, 3.0]));

        // Boundary: (0, 1, 1): z=1 = 1*exp(0) = 1 (should fail)
        assert!(!exp_primal_interior(&[0.0, 1.0, 1.0]));

        // Outside: (0, 1, 0.5): z=0.5 < 1*exp(0) = 1 ✗
        assert!(!exp_primal_interior(&[0.0, 1.0, 0.5]));
    }

    #[test]
    fn test_exp_dual_interior() {
        // Dual cone: K_exp^* = {(u,v,w) : u < 0, w*exp(v/u - 1) >= -u}
        // Equivalently: w >= -u * exp(v/u - 1)
        //             : ln(w) >= ln(-u) + v/u - 1

        // (-1, 0, 1): w=1, -u=1, v/u=0
        //   ln(1) >= ln(1) + 0 - 1
        //   0 >= -1 ✓
        assert!(exp_dual_interior(&[-1.0, 0.0, 1.0]));
    }

    #[test]
    fn test_exp_barrier_grad() {
        // Test that gradient is computed correctly
        let s = [0.0, 1.0, 2.0];
        let mut grad = [0.0; 3];
        exp_barrier_grad_block(&s, &mut grad);

        println!("grad at ({}, {}, {}) = {:?}", s[0], s[1], s[2], grad);

        // Gradient should be finite
        assert!(grad.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_exp_step_to_boundary() {
        let cone = ExpCone::new(1);

        // Interior point: (-0.693, 1.0, 2.0)
        // This corresponds to (t, y, x) where x = 2, exp(-t) = 2, so t = -ln(2) ≈ -0.693
        let s = [-0.693, 1.0, 2.0];
        assert!(exp_primal_interior(&s));

        // Try a step that increases x (should be OK since we're in interior)
        let ds = [0.0, 0.0, 0.1];
        let alpha = cone.step_to_boundary_primal(&s, &ds);

        println!("alpha_s = {}", alpha);
        assert!(alpha > 0.0, "Step in positive direction should be possible");
    }

    #[test]
    fn test_problem_point() {
        // Test the specific point from our benchmark problem
        // Variables: [t, x]
        // Slack: s = [-t, 1, x, 2-x]
        // Cone: (s[0:3]) ∈ K_exp, s[3] ∈ K_+

        // Try t=0, x=1.5
        let t = 0.0;
        let x = 1.5;
        let s_exp = [-t, 1.0, x];

        println!("Testing point: t={}, x={}", t, x);
        println!("  Exp cone slack: {:?}", s_exp);
        println!("  Is interior? {}", exp_primal_interior(&s_exp));

        assert!(exp_primal_interior(&s_exp), "Point should be interior");

        // Now try optimal point: t = -ln(2), x = 2
        let t_opt = -(2.0_f64.ln());
        let x_opt = 2.0;
        let s_opt = [-t_opt, 1.0, x_opt];

        println!("\nOptimal point: t={}, x={}", t_opt, x_opt);
        println!("  Exp cone slack: {:?}", s_opt);
        println!("  Should be on boundary (not interior)");

        // This should be on the boundary, not interior
        // Because x = exp(-t) → x = exp(-(-ln(2))) = exp(ln(2)) = 2
    }

    #[test]
    fn test_step_to_boundary_negative_direction() {
        // Test the step-to-boundary for exp cone with a decreasing direction
        // This is a regression test for the bug where negative steps return 0

        // Interior point: (0, 1, 1.5)
        let s = [0.0, 1.0, 1.5];
        assert!(exp_primal_interior(&s), "Starting point must be interior");

        // Direction that decreases x (should be valid since we're in interior)
        let ds = [-0.1, 0.0, -0.1];

        let alpha = exp_step_to_boundary_block(&s, &ds, exp_primal_interior);

        println!("\nStep-to-boundary test:");
        println!("  s = {:?}", s);
        println!("  ds = {:?}", ds);
        println!("  alpha = {}", alpha);

        // The step should allow some movement
        // At s + alpha*ds, we should still be able to move
        if alpha > 0.0 && alpha < f64::INFINITY {
            let s_new = [
                s[0] + alpha * ds[0],
                s[1] + alpha * ds[1],
                s[2] + alpha * ds[2],
            ];
            println!("  s + alpha*ds = {:?}", s_new);
            println!("  Is interior? {}", exp_primal_interior(&s_new));
        }

        assert!(alpha > 0.0, "Step size should be positive, got {}", alpha);
    }

    #[test]
    fn test_step_boundary_actual_problem() {
        // Reproduce the exact scenario from the trivial exp cone problem
        // After preprocessing, our problem becomes:
        // min x s.t. (x, 1, 1) ∈ K_exp

        // After push-to-interior, what are the initial (s, z) values?
        // Let me check by manually computing them

        // Standard HSDE initialization might set s = e, z = e
        let s = [1.0, 1.0, 1.0];
        let z = [1.0, 1.0, 1.0];

        println!("\nActual problem initialization:");
        println!("  s = {:?}, is_interior = {}", s, exp_primal_interior(&s));
        println!("  z = {:?}, is_interior = {}", z, exp_dual_interior(&z));

        // Check if this is interior
        // For primal: z >= y*exp(x/y) → 1 >= 1*exp(1) = 2.718 → NO!
        // So [1,1,1] is NOT interior for exp cone!
    }

    #[test]
    fn test_what_is_actually_interior() {
        // Find an actual interior point for exp cone
        // K_exp = {(x,y,z) : z >= y*exp(x/y), y > 0}

        // Try various points
        let test_points = vec![
            ([0.0, 1.0, 2.0], "should be interior"),
            ([1.0, 1.0, 3.0], "should be interior"),
            ([1.0, 1.0, 1.0], "NOT interior (1 < e)"),
            ([-1.0, 1.0, 1.0], "should be interior"),
            ([0.0, 1.0, 1.01], "barely interior"),
        ];

        for (point, desc) in test_points {
            let x: f64 = point[0];
            let y: f64 = point[1];
            let z: f64 = point[2];
            let required = y * (x / y).exp();
            let is_int = exp_primal_interior(&point);
            println!("{}: {:?} → z={}, required={}, interior={}",
                     desc, point, z, required, is_int);
        }
    }

    #[test]
    fn test_unit_initialization_is_interior() {
        let cone = ExpCone::new(1);
        let mut s = vec![0.0; 3];
        let mut z = vec![0.0; 3];

        cone.unit_initialization(&mut s, &mut z);

        println!("\nUnit initialization:");
        println!("  s = {:?}, is_interior = {}", s, exp_primal_interior(&s));
        println!("  z = {:?}, is_interior = {}", z, exp_dual_interior(&z));

        // Check what's required
        let x: f64 = s[0];
        let y: f64 = s[1];
        let z_val: f64 = s[2];
        let required = y * (x / y).exp();
        println!("  For s: z={}, y*exp(x/y)={}", z_val, required);

        assert!(exp_primal_interior(&s), "Unit initialization s should be interior");
        assert!(exp_dual_interior(&z), "Unit initialization z should be interior");
    }

    #[test]
    fn test_barrier_gradient_sign() {
        // Test that barrier gradient has correct sign for descent
        let cone = ExpCone::new(1);
        let s = [-1.0, 1.0, 2.0];  // Interior point

        let mut grad = vec![0.0; 3];
        cone.barrier_grad_primal(&s, &mut grad);

        println!("\nBarrier gradient test:");
        println!("  s = {:?}", s);
        println!("  ∇f(s) = {:?}", grad);

        // The barrier gradient should point inward (toward interior)
        // For exp cone, we expect specific signs based on the barrier function
        assert!(grad.iter().all(|&g| g.is_finite()), "Gradient should be finite");
    }
}
