//! Power cone.
//!
//! Uses the log-homogeneous barrier from the design doc.

use super::traits::ConeKernel;
use nalgebra::Matrix3;

/// Power cone (placeholder)
#[derive(Debug, Clone)]
pub struct PowCone {
    alphas: Vec<f64>,
}

impl PowCone {
    /// Create a new power cone with given alpha parameters
    pub fn new(alphas: Vec<f64>) -> Self {
        Self { alphas }
    }

    const INTERIOR_TOL: f64 = 1e-12;
    const NEWTON_TOL: f64 = 1e-10;
    const MAX_NEWTON_ITERS: usize = 20;
    const MAX_LINESEARCH_ITERS: usize = 40;
}

impl ConeKernel for PowCone {
    fn dim(&self) -> usize { 3 * self.alphas.len() }
    fn barrier_degree(&self) -> usize { 3 * self.alphas.len() }
    fn is_interior_primal(&self, s: &[f64]) -> bool {
        assert_eq!(s.len(), self.dim());
        for (block, &alpha) in self.alphas.iter().enumerate() {
            let offset = 3 * block;
            if !pow_primal_interior(&s[offset..offset + 3], alpha) {
                return false;
            }
        }
        true
    }

    fn is_interior_dual(&self, z: &[f64]) -> bool {
        assert_eq!(z.len(), self.dim());
        for (block, &alpha) in self.alphas.iter().enumerate() {
            let offset = 3 * block;
            if !pow_dual_interior(&z[offset..offset + 3], alpha) {
                return false;
            }
        }
        true
    }

    fn step_to_boundary_primal(&self, s: &[f64], ds: &[f64]) -> f64 {
        assert_eq!(s.len(), self.dim());
        assert_eq!(ds.len(), self.dim());
        let mut alpha = f64::INFINITY;
        for (block, &a) in self.alphas.iter().enumerate() {
            let offset = 3 * block;
            let step = pow_step_to_boundary_block(
                &s[offset..offset + 3],
                &ds[offset..offset + 3],
                a,
                pow_primal_interior,
            );
            if step.is_finite() {
                alpha = alpha.min(step.max(0.0));
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
        for (block, &a) in self.alphas.iter().enumerate() {
            let offset = 3 * block;
            let step = pow_step_to_boundary_block(
                &z[offset..offset + 3],
                &dz[offset..offset + 3],
                a,
                pow_dual_interior,
            );
            if step.is_finite() {
                alpha = alpha.min(step.max(0.0));
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
        for (block, &alpha) in self.alphas.iter().enumerate() {
            let offset = 3 * block;
            value += pow_barrier_value_block(&s[offset..offset + 3], alpha);
        }
        value
    }

    fn barrier_grad_primal(&self, s: &[f64], grad_out: &mut [f64]) {
        assert_eq!(s.len(), self.dim());
        assert_eq!(grad_out.len(), self.dim());
        for (block, &alpha) in self.alphas.iter().enumerate() {
            let offset = 3 * block;
            pow_barrier_grad_block(&s[offset..offset + 3], alpha, &mut grad_out[offset..offset + 3]);
        }
    }

    fn barrier_hess_apply_primal(&self, s: &[f64], v: &[f64], out: &mut [f64]) {
        assert_eq!(s.len(), self.dim());
        assert_eq!(v.len(), self.dim());
        assert_eq!(out.len(), self.dim());
        for (block, &alpha) in self.alphas.iter().enumerate() {
            let offset = 3 * block;
            pow_barrier_hess_apply_block(
                &s[offset..offset + 3],
                &v[offset..offset + 3],
                alpha,
                &mut out[offset..offset + 3],
            );
        }
    }

    fn barrier_grad_dual(&self, z: &[f64], grad_out: &mut [f64]) {
        assert_eq!(z.len(), self.dim());
        assert_eq!(grad_out.len(), self.dim());
        for (block, &alpha) in self.alphas.iter().enumerate() {
            let offset = 3 * block;
            let mut x = [0.0; 3];
            let mut h_star = [0.0; 9];
            pow_dual_map_block(&z[offset..offset + 3], alpha, &mut x, &mut h_star);
            grad_out[offset..offset + 3].copy_from_slice(&[-x[0], -x[1], -x[2]]);
        }
    }

    fn barrier_hess_apply_dual(&self, z: &[f64], v: &[f64], out: &mut [f64]) {
        assert_eq!(z.len(), self.dim());
        assert_eq!(v.len(), self.dim());
        assert_eq!(out.len(), self.dim());
        for (block, &alpha) in self.alphas.iter().enumerate() {
            let offset = 3 * block;
            let mut x = [0.0; 3];
            let mut h_star = [0.0; 9];
            pow_dual_map_block(&z[offset..offset + 3], alpha, &mut x, &mut h_star);
            apply_mat3(&h_star, &v[offset..offset + 3], &mut out[offset..offset + 3]);
        }
    }

    fn dual_map(&self, z: &[f64], x_out: &mut [f64], h_star: &mut [f64; 9]) {
        assert_eq!(z.len(), 3, "PowCone dual_map expects a single 3D block");
        assert_eq!(x_out.len(), 3);
        assert_eq!(self.alphas.len(), 1, "PowCone dual_map requires a single alpha");
        pow_dual_map_block(z, self.alphas[0], x_out, h_star);
    }

    fn unit_initialization(&self, s_out: &mut [f64], z_out: &mut [f64]) {
        assert_eq!(s_out.len(), self.dim());
        assert_eq!(z_out.len(), self.dim());
        for (block, &alpha) in self.alphas.iter().enumerate() {
            let offset = 3 * block;
            let x = (1.0 + alpha).sqrt();
            let y = (2.0 - alpha).sqrt();
            s_out[offset] = x;
            s_out[offset + 1] = y;
            s_out[offset + 2] = 0.0;
            z_out[offset] = x;
            z_out[offset + 1] = y;
            z_out[offset + 2] = 0.0;
        }
    }
}

fn pow_primal_interior(s: &[f64], alpha: f64) -> bool {
    if s.len() != 3 || s.iter().any(|&v| !v.is_finite()) {
        return false;
    }
    let x = s[0];
    let y = s[1];
    let z = s[2];
    if x <= 0.0 || y <= 0.0 {
        return false;
    }
    let (a, b) = pow_ab(alpha);
    let log_p = a * x.ln() + b * y.ln();
    let p = log_p.exp();
    let psi = p - z * z;
    if !psi.is_finite() {
        return false;
    }
    let scale = x.abs().max(y.abs()).max(z.abs()).max(1.0);
    psi > PowCone::INTERIOR_TOL * scale
}

fn pow_dual_interior(z: &[f64], alpha: f64) -> bool {
    if z.len() != 3 || z.iter().any(|&v| !v.is_finite()) {
        return false;
    }
    let u = z[0];
    let v = z[1];
    let w = z[2];
    if u <= PowCone::INTERIOR_TOL || v <= PowCone::INTERIOR_TOL {
        return false;
    }
    let w_abs = w.abs();
    if w_abs == 0.0 {
        return true;
    }
    let log_p = alpha * (u / alpha).ln() + (1.0 - alpha) * (v / (1.0 - alpha)).ln();
    (log_p - w_abs.ln()) > PowCone::INTERIOR_TOL
}

fn pow_step_to_boundary_block(
    s: &[f64],
    ds: &[f64],
    alpha: f64,
    interior: fn(&[f64], f64) -> bool,
) -> f64 {
    if ds.iter().all(|&v| v == 0.0) {
        return f64::INFINITY;
    }
    if !interior(s, alpha) {
        return 0.0;
    }

    let mut trial = [0.0; 3];
    for i in 0..3 {
        trial[i] = s[i] + ds[i];
    }
    if interior(&trial, alpha) {
        return f64::INFINITY;
    }

    let mut lo = 0.0;
    let mut hi = 1.0;
    for _ in 0..PowCone::MAX_LINESEARCH_ITERS {
        let mid = 0.5 * (lo + hi);
        for i in 0..3 {
            trial[i] = s[i] + mid * ds[i];
        }
        if interior(&trial, alpha) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

fn pow_barrier_value_block(s: &[f64], alpha: f64) -> f64 {
    let x = s[0];
    let y = s[1];
    let z = s[2];
    let (a, b) = pow_ab(alpha);
    let log_p = a * x.ln() + b * y.ln();
    let p = log_p.exp();
    let psi = p - z * z;
    -psi.ln() - (1.0 - alpha) * x.ln() - alpha * y.ln()
}

fn pow_barrier_grad_block(s: &[f64], alpha: f64, grad_out: &mut [f64]) {
    let x = s[0];
    let y = s[1];
    let z = s[2];
    let (a, b) = pow_ab(alpha);
    let log_p = a * x.ln() + b * y.ln();
    let p = log_p.exp();
    let psi = p - z * z;
    let inv_psi = 1.0 / psi;

    let g1 = a * p / x;
    let g2 = b * p / y;
    let g3 = -2.0 * z;

    grad_out[0] = -inv_psi * g1 - (1.0 - alpha) / x;
    grad_out[1] = -inv_psi * g2 - alpha / y;
    grad_out[2] = -inv_psi * g3;
}

fn pow_barrier_hess_apply_block(s: &[f64], v: &[f64], alpha: f64, out: &mut [f64]) {
    let x = s[0];
    let y = s[1];
    let z = s[2];
    let (a, b) = pow_ab(alpha);
    let log_p = a * x.ln() + b * y.ln();
    let p = log_p.exp();
    let psi = p - z * z;
    let inv_psi = 1.0 / psi;
    let inv_psi2 = inv_psi * inv_psi;

    let g1 = a * p / x;
    let g2 = b * p / y;
    let g3 = -2.0 * z;
    let g = [g1, g2, g3];

    let h11 = a * (a - 1.0) * p / (x * x);
    let h22 = b * (b - 1.0) * p / (y * y);
    let h12 = a * b * p / (x * y);
    let h33 = -2.0;

    let mut h = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            h[3 * i + j] = inv_psi2 * g[i] * g[j];
        }
    }
    h[0] -= inv_psi * h11;
    h[4] -= inv_psi * h22;
    h[1] -= inv_psi * h12;
    h[3] -= inv_psi * h12;
    h[8] -= inv_psi * h33;

    h[0] += (1.0 - alpha) / (x * x);
    h[4] += alpha / (y * y);

    apply_mat3(&h, v, out);
}

fn pow_dual_map_block(z: &[f64], alpha: f64, x_out: &mut [f64], h_star: &mut [f64; 9]) {
    let mut x = [(1.0 + alpha).sqrt(), (2.0 - alpha).sqrt(), 0.0];
    for _ in 0..PowCone::MAX_NEWTON_ITERS {
        let mut grad = [0.0; 3];
        pow_barrier_grad_block(&x, alpha, &mut grad);
        let r = [z[0] + grad[0], z[1] + grad[1], z[2] + grad[2]];
        let r_norm = r.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if r_norm <= PowCone::NEWTON_TOL {
            break;
        }
        let h = pow_hess_matrix(&x, alpha);
        let dx = solve_3x3(&h, &r);
        let mut alpha_ls = 1.0;
        let mut moved = false;
        for _ in 0..PowCone::MAX_LINESEARCH_ITERS {
            let trial = [
                x[0] + alpha_ls * dx[0],
                x[1] + alpha_ls * dx[1],
                x[2] + alpha_ls * dx[2],
            ];
            if pow_primal_interior(&trial, alpha) {
                x = trial;
                moved = true;
                break;
            }
            alpha_ls *= 0.5;
        }
        if !moved {
            break;
        }
    }

    x_out.copy_from_slice(&x);
    let h = pow_hess_matrix(&x, alpha);
    let h_inv = invert_3x3(&h);
    *h_star = h_inv;
}

fn pow_hess_matrix(x: &[f64; 3], alpha: f64) -> [f64; 9] {
    let x0 = x[0];
    let y0 = x[1];
    let z0 = x[2];
    let (a, b) = pow_ab(alpha);
    let log_p = a * x0.ln() + b * y0.ln();
    let p = log_p.exp();
    let psi = p - z0 * z0;
    let inv_psi = 1.0 / psi;
    let inv_psi2 = inv_psi * inv_psi;

    let g1 = a * p / x0;
    let g2 = b * p / y0;
    let g3 = -2.0 * z0;
    let g = [g1, g2, g3];

    let h11 = a * (a - 1.0) * p / (x0 * x0);
    let h22 = b * (b - 1.0) * p / (y0 * y0);
    let h12 = a * b * p / (x0 * y0);
    let h33 = -2.0;

    let mut h = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            h[3 * i + j] = inv_psi2 * g[i] * g[j];
        }
    }
    h[0] -= inv_psi * h11;
    h[4] -= inv_psi * h22;
    h[1] -= inv_psi * h12;
    h[3] -= inv_psi * h12;
    h[8] -= inv_psi * h33;
    h[0] += (1.0 - alpha) / (x0 * x0);
    h[4] += alpha / (y0 * y0);
    h
}

fn pow_ab(alpha: f64) -> (f64, f64) {
    let a = 2.0 * alpha;
    let b = 2.0 - a;
    (a, b)
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
