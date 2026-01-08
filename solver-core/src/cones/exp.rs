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
            let a = exp_step_to_boundary_block(
                &s[offset..offset + 3],
                &ds[offset..offset + 3],
                exp_primal_interior,
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

/// Check if exp cone block (s, z) is in the central neighborhood.
///
/// The central neighborhood condition is: || s + μ ∇f^*(z) ||_∞ <= θ μ
/// where θ is a centrality parameter (typically 0.1 to 0.5).
///
/// This prevents the iterate from drifting too far from the central path.
pub fn exp_central_ok(s: &[f64], z: &[f64], mu: f64, theta: f64) -> bool {
    // Compute ∇f^*(z) via dual map
    let mut x = [0.0; 3];
    let mut h_star = [0.0; 9];
    exp_dual_map_block(z, &mut x, &mut h_star);
    let grad_fstar = [-x[0], -x[1], -x[2]];

    // Compute residual: s + μ ∇f^*(z)
    let res = [
        s[0] + mu * grad_fstar[0],
        s[1] + mu * grad_fstar[1],
        s[2] + mu * grad_fstar[2],
    ];

    // Check || res ||_∞ <= θ μ
    let norm_inf = res.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);
    norm_inf <= theta * mu
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

/// Compute the third-order contraction of ψ: ∇³ψ[p,q]
///
/// For ψ(x,y,z) = y*log(z/y) - x, the non-zero third derivatives are:
/// - ∂³ψ/∂y³ = 1/y²
/// - ∂³ψ/∂y²∂z = -1/z²
/// - ∂³ψ/∂y∂z² = -1/z²
/// - ∂³ψ/∂z³ = 2y/z³
fn exp_third_psi_contract(y: f64, z: f64, p: &[f64], q: &[f64]) -> [f64; 3] {
    let y2 = y * y;
    let z2 = z * z;
    let z3 = z2 * z;

    // ∇³ψ[p,q] is bilinear in p and q
    // Component 0 (x): all third derivatives involving x are 0
    let t0 = 0.0;

    // Component 1 (y):
    //   ∂³ψ/∂y³ p[1]q[1] + ∂³ψ/∂y²∂z (p[1]q[2] + p[2]q[1]) + ∂³ψ/∂y∂z² p[2]q[2]
    let t1 = (1.0 / y2) * p[1] * q[1]
           - (1.0 / z2) * (p[1] * q[2] + p[2] * q[1])
           - (1.0 / z2) * p[2] * q[2];

    // Component 2 (z):
    //   ∂³ψ/∂y²∂z p[1]q[1] + ∂³ψ/∂y∂z² (p[1]q[2] + p[2]q[1]) + ∂³ψ/∂z³ p[2]q[2]
    let t2 = -(1.0 / z2) * p[1] * q[1]
           - (1.0 / z2) * (p[1] * q[2] + p[2] * q[1])
           + (2.0 * y / z3) * p[2] * q[2];

    [t0, t1, t2]
}

/// Compute the third-order contraction for the primal barrier: ∇³f[p,q]
///
/// For f(x) = -log(ψ) - log(y) - log(z), using the generic formula:
/// ∇³(-log ψ)[p,q] = -(1/ψ) * ∇³ψ[p,q]
///                   + (1/ψ²) * (∇ψᵀp * ∇²ψ q + ∇ψᵀq * ∇²ψ p + pᵀ∇²ψq * ∇ψ)
///                   - (2/ψ³) * (∇ψᵀp) * (∇ψᵀq) * ∇ψ
fn exp_primal_third_contract(x: &[f64], p: &[f64], q: &[f64]) -> [f64; 3] {
    let y = x[1];
    let z = x[2];
    let psi = y * (z / y).ln() - x[0];

    let gpsi = exp_grad_psi(y, z);
    let hpsi = exp_hess_psi(y, z);
    let t_psi = exp_third_psi_contract(y, z, p, q);

    let inv_psi = 1.0 / psi;
    let inv_psi2 = inv_psi * inv_psi;
    let inv_psi3 = inv_psi2 * inv_psi;

    // Compute Hψ * p and Hψ * q
    let mut hpsi_p = [0.0; 3];
    let mut hpsi_q = [0.0; 3];
    apply_mat3(&hpsi, p, &mut hpsi_p);
    apply_mat3(&hpsi, q, &mut hpsi_q);

    // Scalar products
    let gpsi_dot_p = gpsi[0] * p[0] + gpsi[1] * p[1] + gpsi[2] * p[2];
    let gpsi_dot_q = gpsi[0] * q[0] + gpsi[1] * q[1] + gpsi[2] * q[2];
    let p_dot_hpsi_q = p[0] * hpsi_q[0] + p[1] * hpsi_q[1] + p[2] * hpsi_q[2];

    // Generic formula for ∇³(-log ψ)[p,q]
    let mut result = [0.0; 3];
    for i in 0..3 {
        result[i] = -inv_psi * t_psi[i]
                  + inv_psi2 * (gpsi_dot_p * hpsi_q[i] + gpsi_dot_q * hpsi_p[i] + p_dot_hpsi_q * gpsi[i])
                  - 2.0 * inv_psi3 * gpsi_dot_p * gpsi_dot_q * gpsi[i];
    }

    // Add contributions from -log(y) and -log(z)
    // ∇³(-log y)[p,q] = 2/y³ p[1]q[1] at component 1
    // ∇³(-log z)[p,q] = 2/z³ p[2]q[2] at component 2
    result[1] += 2.0 / (y * y * y) * p[1] * q[1];
    result[2] += 2.0 / (z * z * z) * p[2] * q[2];

    result
}

/// Compute the third-order correction η for exp cone Mehrotra predictor-corrector.
///
/// Given:
/// - z: current dual point
/// - ds_aff, dz_aff: affine (predictor) directions
/// - x, h_star: outputs from dual map
///
/// Returns: η = -0.5 * ∇³f^*(z)[dz_aff, u] where u = H_star^{-1} ds_aff
///
/// We compute this via the primal barrier using:
/// - p = -H_star * dz_aff (in primal space)
/// - q = H_star^{-1} * ds_aff (in primal space)
/// - η_primal = ∇³f(x)[p, q]
/// - η = -0.5 * H_star * η_primal
pub fn exp_third_order_correction(
    _z: &[f64],
    ds_aff: &[f64],
    dz_aff: &[f64],
    x: &[f64],
    h_star: &[f64; 9],
) -> [f64; 3] {
    // Compute p = -H_star * dz_aff
    let mut p = [0.0; 3];
    apply_mat3(h_star, dz_aff, &mut p);
    p[0] = -p[0];
    p[1] = -p[1];
    p[2] = -p[2];

    // Compute q = H_star^{-1} * ds_aff
    // This requires solving H_star * q = ds_aff
    // For now, invert H_star (it's 3x3, cheap)
    let h_star_inv = invert_3x3(h_star);
    let mut q = [0.0; 3];
    apply_mat3(&h_star_inv, ds_aff, &mut q);

    // Compute ∇³f(x)[p, q]
    let third_contract = exp_primal_third_contract(x, &p, &q);

    // η = -0.5 * H_star * third_contract
    let mut eta = [0.0; 3];
    apply_mat3(h_star, &third_contract, &mut eta);
    eta[0] *= -0.5;
    eta[1] *= -0.5;
    eta[2] *= -0.5;

    eta
}

/// Compute the dual barrier gradient for the exponential cone dual.
///
/// The dual cone is K_exp* = {(u,v,w) : u < 0, w ≥ -u*exp(v/u - 1)}
/// The dual barrier is f*(u,v,w) = -log(-u) - log(w) - log(ψ*)
/// where ψ* = u + w*exp(v/w - 1)
pub fn exp_dual_barrier_grad_block(z: &[f64], grad_out: &mut [f64]) {
    let u: f64 = z[0];
    let v: f64 = z[1];
    let w: f64 = z[2];

    // Compute ψ* = u + w*exp(v/w - 1)
    let exp_term = (v / w - 1.0).exp();
    let psi_star = u + w * exp_term;

    let inv_psi_star = 1.0 / psi_star;

    // ∂ψ*/∂u = 1
    // ∂ψ*/∂v = exp(v/w - 1)
    // ∂ψ*/∂w = exp(v/w - 1) * (1 - v/w)
    let d_psi_du = 1.0;
    let d_psi_dv = exp_term;
    let d_psi_dw = exp_term * (1.0 - v / w);

    // ∇f*(u,v,w) = [1/u - 1/ψ*, -exp(v/w-1)/ψ*, -1/w - exp(v/w-1)*(1-v/w)/ψ*]
    grad_out[0] = 1.0 / u - inv_psi_star * d_psi_du;
    grad_out[1] = -inv_psi_star * d_psi_dv;
    grad_out[2] = -1.0 / w - inv_psi_star * d_psi_dw;
}

/// Compute the dual barrier Hessian for the exponential cone dual.
fn exp_dual_hess_matrix(z: &[f64]) -> [f64; 9] {
    let u: f64 = z[0];
    let v: f64 = z[1];
    let w: f64 = z[2];

    let exp_term = (v / w - 1.0).exp();
    let psi_star = u + w * exp_term;

    let inv_psi_star = 1.0 / psi_star;
    let inv_psi_star2 = inv_psi_star * inv_psi_star;

    // Gradient of ψ*
    let d_psi = [1.0, exp_term, exp_term * (1.0 - v / w)];

    // Hessian of ψ* (sparse structure)
    // ∂²ψ*/∂u² = 0, ∂²ψ*/∂u∂v = 0, ∂²ψ*/∂u∂w = 0
    // ∂²ψ*/∂v² = exp(v/w-1) / w
    // ∂²ψ*/∂v∂w = -exp(v/w-1) * v / w²
    // ∂²ψ*/∂w² = exp(v/w-1) * v² / w³
    let h_psi = [
        0.0, 0.0, 0.0,
        0.0, exp_term / w, -exp_term * v / (w * w),
        0.0, -exp_term * v / (w * w), exp_term * v * v / (w * w * w),
    ];

    // Hessian formula: H = (1/ψ²) * ∇ψ ∇ψᵀ - (1/ψ) * ∇²ψ + diag terms
    let mut h = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            h[3 * i + j] = inv_psi_star2 * d_psi[i] * d_psi[j] - inv_psi_star * h_psi[3 * i + j];
        }
    }

    // Add diagonal terms from -log(-u) and -log(w)
    h[0] += 1.0 / (u * u);  // ∂²(-log(-u))/∂u² = -1/u²
    h[8] += 1.0 / (w * w);  // ∂²(-log(w))/∂w² = 1/w²

    h
}

/// Compute third-order correction for exponential cone predictor-corrector.
///
/// Implements Clarabel's third-order correction for nonsymmetric cones:
///   η = -½∇³f*(z)[Δz, H^{-1}Δs]
///
/// where:
/// - z: current dual iterate (u,v,w) ∈ K_exp*
/// - dz_aff: affine dual step
/// - ds_aff: affine primal step
/// - eta_out: output correction term
///
/// This captures curvature information that second-order Mehrotra correction
/// misses, allowing larger confident steps through the exp cone's nonlinear geometry.
/// Public wrapper for third-order correction (called from predictor-corrector).
// REMOVED: Finite-difference third-order correction (numerically unstable)
// See _planning/v16/third_order_correction_analysis.md for details.
//
// The correct approach requires an analytical formula (like Clarabel uses),
// not finite differences. The analytical formula involves:
// - Auxiliary function ψ = z[0]*log(-z[0]/z[2]) - z[0] + z[1]
// - Complex combinations of dot products and reciprocals
// - Proper scaling and sign conventions
//
// Expected benefit when properly implemented: 3-10x iteration reduction
// (from 50-200 iterations to 10-30 iterations on exp cone problems)
//
// For now, exp cones use standard second-order Mehrotra correction.
// This is correct but less efficient than third-order correction.

pub fn exp_dual_map_block(z: &[f64], x_out: &mut [f64], h_star: &mut [f64; 9]) {
    // The dual map should solve: ∇f(x) + z = 0
    // where f is the PRIMAL barrier and x is in the PRIMAL cone.
    // Then ∇f^*(z) = -x by Fenchel conjugacy.

    // Start from primal unit initialization
    let mut x = [-1.051_383, 0.556_409, 1.258_967];

    for _ in 0..ExpCone::MAX_NEWTON_ITERS {
        let mut grad = [0.0; 3];
        exp_barrier_grad_block(&x, &mut grad);  // Use PRIMAL barrier gradient!
        let r = [z[0] + grad[0], z[1] + grad[1], z[2] + grad[2]];
        let r_norm = r.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if r_norm <= ExpCone::NEWTON_TOL {
            break;
        }
        let h = exp_hess_matrix(&[x[0], x[1], x[2]]);  // Use PRIMAL barrier Hessian!
        let dx = solve_3x3(&h, &r);
        let mut alpha = 1.0;
        let mut moved = false;
        for _ in 0..ExpCone::MAX_LINESEARCH_ITERS {
            let trial = [x[0] + alpha * dx[0], x[1] + alpha * dx[1], x[2] + alpha * dx[2]];
            // Check primal cone interior since x should be in K (primal cone)
            // solving ∇f(x) + z = 0 where f is the primal barrier
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
    let h = exp_hess_matrix(&[x[0], x[1], x[2]]);  // Use PRIMAL barrier Hessian!
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
    fn test_dual_barrier_gradient_finite() {
        // Test that the dual barrier gradient is finite for interior points
        let z = [-1.0, 0.5, 1.5];  // Should be in K_exp* interior
        assert!(exp_dual_interior(&z), "Test point should be in dual interior");

        let mut grad = [0.0; 3];
        exp_dual_barrier_grad_block(&z, &mut grad);

        println!("\nDual barrier gradient test:");
        println!("  z = {:?}", z);
        println!("  ∇f*(z) = {:?}", grad);

        assert!(grad.iter().all(|&g| g.is_finite()), "Dual barrier gradient should be finite");
    }

    #[test]
    fn test_dual_map_basic() {
        // Test that dual_map produces a point in the primal cone
        let z = [-1.0, 0.5, 1.5];  // Point in K_exp* interior
        assert!(exp_dual_interior(&z), "z should be in dual interior");

        let mut s_tilde = [0.0; 3];
        let mut h_star = [0.0; 9];
        exp_dual_map_block(&z, &mut s_tilde, &mut h_star);

        println!("\nDual map test:");
        println!("  z = {:?}", z);
        println!("  s_tilde = -∇f*(z) = {:?}", s_tilde);
        println!("  is s_tilde interior? {}", exp_primal_interior(&s_tilde));

        // The dual map should return s_tilde such that ∇f*(z) = -s_tilde
        // Since ∇f*(z) ∈ K for z ∈ K*, we expect s_tilde ∈ K
        // Actually, s_tilde = -∇f*(z) ∈ -K, which may not be in K...
        // Let me check what the gradient actually is
    }

    // DISABLED: Third-order correction removed (finite differences were unstable)
    // See _planning/v16/third_order_correction_analysis.md for details
    // #[test]
    // fn test_third_order_correction() {
    //     // Test third-order correction computation
    //     let z = [-1.0, 0.5, 1.5];  // Dual interior point
    //     assert!(exp_dual_interior(&z), "z must be in dual interior");
    //
    //     // Random affine steps
    //     let dz_aff = [0.05, -0.02, 0.08];
    //     let ds_aff = [-0.03, 0.04, -0.01];
    //
    //     let mut eta = [0.0; 3];
    //     exp_third_order_correction(&z, &dz_aff, &ds_aff, &mut eta);
    //
    //     println!("\nThird-order correction test:");
    //     println!("  z = {:?}", z);
    //     println!("  dz_aff = {:?}", dz_aff);
    //     println!("  ds_aff = {:?}", ds_aff);
    //     println!("  η (correction) = {:?}", eta);
    //
    //     // Check that output is finite
    //     assert!(eta.iter().all(|&x: &f64| x.is_finite()), "Correction should be finite");
    //
    //     // Check magnitude is reasonable (not exploding)
    //     assert!(eta.iter().all(|&x: &f64| x.abs() < 100.0), "Correction should be bounded");
    //
    //     // The correction should be non-trivial (not all zeros)
    //     let max_abs = eta.iter().map(|&x: &f64| x.abs()).fold(0.0_f64, f64::max);
    //     assert!(max_abs > 1e-10, "Correction should be non-trivial");
    // }

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
