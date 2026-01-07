//! Second-order (Lorentz) cone.
//!
//! The second-order cone (also called Lorentz cone or ice cream cone) is defined as:
//!
//! K_SOC = {(t, x) ∈ ℝ × ℝ^{d-1} : t ≥ ||x||₂}
//!
//! This is a self-dual cone and is fundamental for SOCP (second-order cone programming).
//!
//! # Barrier Function
//!
//! f(t, x) = -log(t² - ||x||²)
//!
//! # Jordan Algebra
//!
//! The SOC has a Jordan algebra structure with:
//! - Product: (t,x) ∘ (u,v) = (tu + x^T v, tv + ux)
//! - Identity: e = (1, 0, ..., 0)
//! - Spectral decomposition: λ₁ = t + ||x||, λ₂ = t - ||x||
//!
//! This structure is used for Nesterov-Todd scaling in the IPM.

use super::traits::ConeKernel;

/// Second-order (Lorentz) cone.
///
/// Represents the constraint t ≥ ||x||₂ where the first component is t
/// and the remaining components form the vector x.
#[derive(Debug, Clone)]
pub struct SocCone {
    /// Total dimension (d = 1 + length of x vector)
    dim: usize,
}

impl SocCone {
    /// Create a new second-order cone of the given dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - Total dimension (must be at least 2: one for t, at least one for x)
    pub fn new(dim: usize) -> Self {
        assert!(dim >= 2, "SOC cone must have dimension >= 2");
        Self { dim }
    }

    /// Interior tolerance
    const INTERIOR_TOL: f64 = 1e-12;

    /// Scaling interior tolerance: accept very small positive values.
    const SCALING_INTERIOR_TOL: f64 = 1e-30;

    /// Relaxed interior check for scaling computations.
    pub(crate) fn is_interior_scaling(&self, s: &[f64]) -> bool {
        assert_eq!(s.len(), self.dim);
        if s.iter().any(|&x| !x.is_finite()) {
            return false;
        }

        let t = s[0];
        if t <= 0.0 {
            return false;
        }

        let x_norm = Self::x_norm(s);
        let tol = Self::SCALING_INTERIOR_TOL * t.abs().max(1.0);
        t - x_norm > tol
    }

    /// Compute t² - ||x||² (the discriminant used throughout)
    #[inline]
    fn discriminant(s: &[f64]) -> f64 {
        let t = s[0];
        let x_norm_sq: f64 = s[1..].iter().map(|&xi| xi * xi).sum();
        t * t - x_norm_sq
    }

    /// Compute ||x||₂
    #[inline]
    fn x_norm(s: &[f64]) -> f64 {
        s[1..].iter().map(|&xi| xi * xi).sum::<f64>().sqrt()
    }

    /// Compute inner product x^T y
    #[inline]
    #[allow(dead_code)]
    fn x_dot(s: &[f64], v: &[f64]) -> f64 {
        s[1..].iter().zip(&v[1..]).map(|(&si, &vi)| si * vi).sum()
    }
}

// ============================================================================
// Jordan Algebra Operations
// ============================================================================

/// Jordan product: (t,x) ∘ (u,v) = (tu + x^T v, tv + ux)
#[allow(dead_code)]
fn jordan_product(s: &[f64], other: &[f64], out: &mut [f64]) {
    let t = s[0];
    let u = other[0];

    // out[0] = t*u + x^T v
    out[0] = t * u + s[1..].iter().zip(&other[1..]).map(|(&si, &oi)| si * oi).sum::<f64>();

    // out[1..] = t*v + u*x
    for i in 1..s.len() {
        out[i] = t * other[i] + u * s[i];
    }
}

/// Spectral decomposition: compute eigenvalues λ₁ = t + ||x||, λ₂ = t - ||x||
#[allow(dead_code)]
fn spectral_values(s: &[f64]) -> (f64, f64) {
    let t = s[0];
    let x_norm = SocCone::x_norm(s);
    (t + x_norm, t - x_norm)
}

/// Jordan square root: compute w such that w ∘ w = s
///
/// Uses spectral decomposition: if s has eigenvalues (λ₁, λ₂) with eigenvectors (c₁, c₂),
/// then √s has eigenvalues (√λ₁, √λ₂) with the same eigenvectors.
#[allow(dead_code)]
fn jordan_sqrt(s: &[f64], out: &mut [f64]) {
    let t = s[0];
    let x_norm = SocCone::x_norm(s);

    let lambda1 = t + x_norm;
    let lambda2 = t - x_norm;

    assert!(lambda2 > 0.0, "Cannot take square root of point not in interior");

    let sqrt_lambda1 = lambda1.sqrt();
    let sqrt_lambda2 = lambda2.sqrt();

    // Reconstruct: w_t = (√λ₁ + √λ₂)/2, w_x = (√λ₁ - √λ₂)/(2||x||) * x
    out[0] = (sqrt_lambda1 + sqrt_lambda2) / 2.0;

    if x_norm > 1e-12 {
        let scale = (sqrt_lambda1 - sqrt_lambda2) / (2.0 * x_norm);
        for i in 1..s.len() {
            out[i] = scale * s[i];
        }
    } else {
        // If x ≈ 0, then s ≈ (t, 0), and √s = (√t, 0)
        for i in 1..s.len() {
            out[i] = 0.0;
        }
    }
}

/// Jordan inverse: compute w such that w ∘ s = e (identity)
#[allow(dead_code)]
fn jordan_inv(s: &[f64], out: &mut [f64]) {
    let t = s[0];
    let x_norm_sq: f64 = s[1..].iter().map(|&xi| xi * xi).sum();
    let det = t * t - x_norm_sq;

    assert!(det > 0.0, "Cannot invert point not in interior");

    // w = (t, -x) / det
    out[0] = t / det;
    for i in 1..s.len() {
        out[i] = -s[i] / det;
    }
}

/// Quadratic representation: P(w)y = 2w ∘ (w ∘ y) - (w ∘ w) ∘ y
///
/// This is used in NT scaling computations.
#[allow(dead_code)]
fn quad_rep(w: &[f64], y: &[f64], out: &mut [f64]) {
    let n = w.len();
    let mut w_circ_y = vec![0.0; n];
    let mut w_circ_w = vec![0.0; n];
    let mut temp = vec![0.0; n];

    jordan_product(w, y, &mut w_circ_y);
    jordan_product(w, w, &mut w_circ_w);
    jordan_product(w, &w_circ_y, &mut temp);
    jordan_product(&w_circ_w, y, out);

    for i in 0..n {
        out[i] = 2.0 * temp[i] - out[i];
    }
}

// ============================================================================
// ConeKernel Implementation
// ============================================================================

impl ConeKernel for SocCone {
    fn dim(&self) -> usize {
        self.dim
    }

    fn barrier_degree(&self) -> usize {
        2  // SOC always has barrier degree 2
    }

    fn is_interior_primal(&self, s: &[f64]) -> bool {
        assert_eq!(s.len(), self.dim);

        // Check for NaN
        if s.iter().any(|&x| x.is_nan()) {
            return false;
        }

        // Compute discriminant u = t² - ||x||²
        let u = Self::discriminant(s);

        // Need t > 0 and u > 0
        let s_norm = s.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        let tol = Self::INTERIOR_TOL * s_norm.max(1.0);

        s[0] > tol && u > tol * tol
    }

    fn is_interior_dual(&self, z: &[f64]) -> bool {
        // SOC is self-dual
        self.is_interior_primal(z)
    }

    fn step_to_boundary_primal(&self, s: &[f64], ds: &[f64]) -> f64 {
        assert_eq!(s.len(), self.dim);
        assert_eq!(ds.len(), self.dim);

        // We want the maximum α such that (t + α Δt)² - ||x + α Δx||² ≥ 0
        //
        // Expand: t² + 2tα(Δt) + α²(Δt)² - ||x||² - 2α(x^T Δx) - α²||Δx||² ≥ 0
        //
        // Rearrange: aα² + bα + c ≥ 0, where:
        //   a = (Δt)² - ||Δx||²
        //   b = 2(t Δt - x^T Δx)
        //   c = t² - ||x||² > 0 (since s is interior)

        let t = s[0];
        let dt = ds[0];

        let x_norm_sq: f64 = s[1..].iter().map(|&xi| xi * xi).sum();
        let dx_norm_sq: f64 = ds[1..].iter().map(|&dxi| dxi * dxi).sum();
        let x_dot_dx: f64 = s[1..].iter().zip(&ds[1..]).map(|(&xi, &dxi)| xi * dxi).sum();

        let a = dt * dt - dx_norm_sq;
        let b = 2.0 * (t * dt - x_dot_dx);
        let c = t * t - x_norm_sq;

        if c <= 0.0 || !c.is_finite() {
            return 0.0;
        }

        // Solve aα² + bα + c = 0
        // If a ≈ 0 relative to other coefficients, use linear case: α = -c/b
        let coef_scale = a.abs().max(b.abs()).max(c.abs()).max(1.0);
        if a.abs() < 1e-12 * coef_scale {
            // Linear case: aα² term is negligible relative to bα + c
            if b < 0.0 {
                return -c / b;
            } else {
                return f64::INFINITY;
            }
        }

        // Quadratic formula: α = (-b ± √(b² - 4ac)) / (2a)
        let discriminant = b * b - 4.0 * a * c;

        // Due to floating point precision, discriminant can be slightly negative
        // when mathematically it should be zero (or very small positive).
        // Use a relative tolerance based on the magnitude of b² and 4ac.
        let disc_scale = (b * b).abs().max((4.0 * a * c).abs()).max(1e-300);
        let disc_tol = 1e-12 * disc_scale;

        if discriminant < -disc_tol {
            // Definitely no real roots: direction points into interior
            return f64::INFINITY;
        }

        // Clamp small negative discriminants to zero
        let sqrt_disc = discriminant.max(0.0).sqrt();

        // Use Citardauq formula to avoid catastrophic cancellation.
        // Standard formula (-b ± √disc) / 2a loses precision when b ≈ ±√disc.
        // Instead, compute one root directly and the other via c = a*α1*α2.
        let (alpha1, alpha2) = if b >= 0.0 {
            // b positive: -b - √disc has no cancellation (both negative)
            let q = -0.5 * (b + sqrt_disc);
            if q.abs() < 1e-300 {
                // Degenerate case: both roots are ~0
                (0.0, 0.0)
            } else {
                (q / a, c / q)
            }
        } else {
            // b negative: -b + √disc has no cancellation (both positive)
            let q = -0.5 * (b - sqrt_disc);
            if q.abs() < 1e-300 {
                (0.0, 0.0)
            } else {
                (q / a, c / q)
            }
        };

        // We want the smallest positive root
        let mut alpha_max = f64::INFINITY;

        if alpha1 > 0.0 {
            alpha_max = alpha_max.min(alpha1);
        }
        if alpha2 > 0.0 {
            alpha_max = alpha_max.min(alpha2);
        }

        // Also need t + α Δt > 0
        if dt < 0.0 {
            let alpha_t = -t / dt;
            alpha_max = alpha_max.min(alpha_t);
        }

        alpha_max
    }

    fn step_to_boundary_dual(&self, z: &[f64], dz: &[f64]) -> f64 {
        // Self-dual
        self.step_to_boundary_primal(z, dz)
    }

    fn barrier_value(&self, s: &[f64]) -> f64 {
        assert_eq!(s.len(), self.dim);

        // f(t, x) = -log(t² - ||x||²)
        let u = Self::discriminant(s);
        assert!(u > 0.0, "s not in interior");

        -u.ln()
    }

    fn barrier_grad_primal(&self, s: &[f64], grad_out: &mut [f64]) {
        assert_eq!(s.len(), self.dim);
        assert_eq!(grad_out.len(), self.dim);

        let t = s[0];
        let u = Self::discriminant(s);

        // ∇f = [-2t/u, 2x/u]
        grad_out[0] = -2.0 * t / u;
        for i in 1..self.dim {
            grad_out[i] = 2.0 * s[i] / u;
        }
    }

    fn barrier_hess_apply_primal(&self, s: &[f64], v: &[f64], out: &mut [f64]) {
        assert_eq!(s.len(), self.dim);
        assert_eq!(v.len(), self.dim);
        assert_eq!(out.len(), self.dim);

        // ∇²f = (2/u) * [[-1, 0], [0, I]] + (4/u²) * [[t], [-x]] * [[t], [-x]]^T
        //
        // (∇²f v) = (2/u) * [[-v_t], [v_x]] + (4/u²) * (t v_t - x^T v_x) * [[t], [-x]]

        let t = s[0];
        let u = Self::discriminant(s);

        let v_t = v[0];
        let x_dot_v: f64 = s[1..].iter().zip(&v[1..]).map(|(&xi, &vi)| xi * vi).sum();

        let a = t * v_t - x_dot_v;  // = [[t], [-x]]^T * v

        // out_t = (2/u) * (-v_t) + (4/u²) * a * t
        out[0] = (-2.0 / u) * v_t + (4.0 / (u * u)) * t * a;

        // out_x = (2/u) * v_x + (4/u²) * a * (-x)
        for i in 1..self.dim {
            out[i] = (2.0 / u) * v[i] + (4.0 / (u * u)) * (-s[i]) * a;
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
        panic!("SOC is self-dual; dual_map not needed");
    }

    fn unit_initialization(&self, s_out: &mut [f64], z_out: &mut [f64]) {
        assert_eq!(s_out.len(), self.dim);
        assert_eq!(z_out.len(), self.dim);

        // Initialize to (1, 0, ..., 0)
        s_out[0] = 1.0;
        z_out[0] = 1.0;

        for i in 1..self.dim {
            s_out[i] = 0.0;
            z_out[i] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soc_basic() {
        let cone = SocCone::new(3);
        assert_eq!(cone.dim(), 3);
        assert_eq!(cone.barrier_degree(), 2);
    }

    #[test]
    fn test_soc_interior() {
        let cone = SocCone::new(3);

        // Interior: t = 2, x = (1, 0), ||x|| = 1 < 2 ✓
        assert!(cone.is_interior_primal(&[2.0, 1.0, 0.0]));

        // Interior: t = 5, x = (3, 4), ||x|| = 5 = t (boundary)
        // Should fail due to tolerance
        assert!(!cone.is_interior_primal(&[5.0, 3.0, 4.0]));

        // Interior: t = 5.1, x = (3, 4), ||x|| = 5 < 5.1 ✓
        assert!(cone.is_interior_primal(&[5.1, 3.0, 4.0]));

        // Exterior: t = 1, x = (2, 0), ||x|| = 2 > 1 ✗
        assert!(!cone.is_interior_primal(&[1.0, 2.0, 0.0]));

        // Negative t
        assert!(!cone.is_interior_primal(&[-1.0, 0.0, 0.0]));

        // NaN
        assert!(!cone.is_interior_primal(&[f64::NAN, 0.0, 0.0]));
    }

    #[test]
    fn test_soc_discriminant() {
        // t=3, x=(1,2), ||x||² = 5, u = 9 - 5 = 4
        let s = vec![3.0, 1.0, 2.0];
        assert!((SocCone::discriminant(&s) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_soc_barrier_value() {
        let cone = SocCone::new(3);

        // t=3, x=(1,2), u=4, f=-log(4)
        let s = vec![3.0, 1.0, 2.0];
        let f = cone.barrier_value(&s);
        let expected = -(4.0f64).ln();
        assert!((f - expected).abs() < 1e-10);
    }

    #[test]
    fn test_soc_step_to_boundary() {
        let cone = SocCone::new(3);

        // Start at s = (2, 0, 0), move in direction ds = (1, 0, 0)
        // This moves away from boundary: α = ∞
        let s = vec![2.0, 0.0, 0.0];
        let ds = vec![1.0, 0.0, 0.0];
        assert_eq!(cone.step_to_boundary_primal(&s, &ds), f64::INFINITY);

        // Start at s = (2, 0, 0), move in direction ds = (-1, 1, 0)
        // Need (2-α)² ≥ α², which gives 4 - 4α + α² ≥ α², so 4 ≥ 4α, α ≤ 1
        // Also need 2 - α > 0, so α < 2
        // Boundary at α = 1: (1, 1, 0) has ||x|| = 1 = t
        let ds = vec![-1.0, 1.0, 0.0];
        let alpha = cone.step_to_boundary_primal(&s, &ds);
        assert!((alpha - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_soc_jordan_product() {
        // (2, [1,0]) ∘ (3, [0,1]) = (2*3 + 1*0 + 0*1, 2*[0,1] + 3*[1,0])
        //                          = (6, [3, 2])
        let s = vec![2.0, 1.0, 0.0];
        let other = vec![3.0, 0.0, 1.0];
        let mut out = vec![0.0; 3];

        jordan_product(&s, &other, &mut out);

        assert!((out[0] - 6.0).abs() < 1e-10);
        assert!((out[1] - 3.0).abs() < 1e-10);
        assert!((out[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_soc_spectral_values() {
        // t=5, x=(3,4), ||x||=5
        // λ₁ = 5+5=10, λ₂ = 5-5=0 (boundary)
        let s = vec![5.0, 3.0, 4.0];
        let (l1, l2) = spectral_values(&s);
        assert!((l1 - 10.0).abs() < 1e-10);
        assert!(l2.abs() < 1e-10);

        // t=3, x=(1,2), ||x||=√5
        let s = vec![3.0, 1.0, 2.0];
        let (l1, l2) = spectral_values(&s);
        let sqrt5 = 5.0f64.sqrt();
        assert!((l1 - (3.0 + sqrt5)).abs() < 1e-10);
        assert!((l2 - (3.0 - sqrt5)).abs() < 1e-10);
    }

    #[test]
    fn test_soc_initialization() {
        let cone = SocCone::new(5);
        let mut s = vec![0.0; 5];
        let mut z = vec![0.0; 5];

        cone.unit_initialization(&mut s, &mut z);

        assert_eq!(s[0], 1.0);
        assert_eq!(z[0], 1.0);
        for i in 1..5 {
            assert_eq!(s[i], 0.0);
            assert_eq!(z[i], 0.0);
        }

        assert!(cone.is_interior_primal(&s));
        assert!(cone.is_interior_dual(&z));
    }

    #[test]
    fn test_soc_step_citardauq_edge_case() {
        // Test case where standard quadratic formula would lose precision
        // due to catastrophic cancellation: b ≈ √discriminant
        let cone = SocCone::new(3);

        // Construct a case where b and √disc are nearly equal
        // s = (2.0, 0.5, 0.5), ds = (-1.0, 0.5, 0.5)
        // This creates a case where we're moving toward the boundary
        let s = vec![2.0, 0.5, 0.5];
        let ds = vec![-1.0, 0.5, 0.5];

        let alpha = cone.step_to_boundary_primal(&s, &ds);

        // Verify result is positive and finite
        assert!(alpha > 0.0);
        assert!(alpha < 100.0);

        // Verify that s + alpha*ds is on the boundary (t = ||x||)
        let t_new = s[0] + alpha * ds[0];
        let x_new: Vec<f64> = (1..3).map(|i| s[i] + alpha * ds[i]).collect();
        let x_norm = (x_new[0] * x_new[0] + x_new[1] * x_new[1]).sqrt();

        // Should be on or very close to boundary
        assert!((t_new - x_norm).abs() < 1e-10);
    }

    #[test]
    fn test_soc_step_scaled_problem() {
        // Test that relative threshold works for scaled problems
        // Scale all coefficients by 1e8
        let cone = SocCone::new(3);

        let scale = 1e8;
        let s = vec![2.0 * scale, 0.0, 0.0];
        let ds = vec![-1.0 * scale, 1.0 * scale, 0.0];

        let alpha = cone.step_to_boundary_primal(&s, &ds);

        // Should get α = 1 (same as unscaled case)
        assert!((alpha - 1.0).abs() < 1e-8);

        // Verify boundary
        let t_new = s[0] + alpha * ds[0];
        let x_new = s[1] + alpha * ds[1];
        assert!((t_new - x_new.abs()).abs() < 1e-6 * scale);
    }

    #[test]
    fn test_soc_step_tiny_quadratic_coef() {
        // Test linear case detection when a ≈ 0 relative to other coefficients
        let cone = SocCone::new(3);

        // Case where the quadratic term is negligible
        // s = (10.0, 1.0, 0.0), ds = (0.0, -1.0, 0.0)
        // dt = 0, dx = (-1, 0), so a = dt² - ||dx||² = -1
        // This is not the linear case, but let's construct one

        // For linear case: we need dt² ≈ ||dx||²
        // s = (10.0, 0.0, 0.0), ds = (1.0, 1.0, 0.0)
        // dt = 1, ||dx||² = 1, a = 1-1 = 0 (exactly linear)
        let s = vec![10.0, 0.0, 0.0];
        let ds = vec![1.0, 1.0, 0.0];

        let alpha = cone.step_to_boundary_primal(&s, &ds);

        // Moving (10,0,0) by α*(1,1,0) gives (10+α, α, 0)
        // Boundary at t = ||x||, so 10+α = α, which is impossible
        // Actually, t > ||x|| always in this direction, so α = ∞
        assert_eq!(alpha, f64::INFINITY);
    }

    #[test]
    fn test_soc_step_boundary_cases() {
        let cone = SocCone::new(3);

        // Case: already on boundary (should return 0)
        let s_boundary = vec![1.0, 1.0, 0.0]; // t = ||x|| = 1
        let ds = vec![-1.0, 0.0, 0.0]; // moving into exterior
        let alpha = cone.step_to_boundary_primal(&s_boundary, &ds);
        assert_eq!(alpha, 0.0);

        // Case: moving tangent to cone surface (along boundary)
        // At (2, √2, √2), ||x|| = √(2+2) = 2 = t (on boundary)
        let sqrt2 = 2.0f64.sqrt();
        let s_boundary2 = vec![2.0, sqrt2, sqrt2];
        // Direction d = (1, √2/2, √2/2): new t = 2+α, new ||x|| = √((√2+α√2/2)²*2) = 2+α
        // So t = ||x|| for all α - this is tangent to the cone surface
        let ds2 = vec![1.0, sqrt2 / 2.0, sqrt2 / 2.0];
        let alpha2 = cone.step_to_boundary_primal(&s_boundary2, &ds2);
        // Tangent direction: stays on boundary, so α = ∞ (never hits boundary again)
        // or α = 0 depending on numerical precision at the boundary
        assert!(alpha2.is_infinite() || alpha2 == 0.0);
    }
}
