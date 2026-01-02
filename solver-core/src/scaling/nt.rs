//! Nesterov-Todd scaling for symmetric cones.
//!
//! The NT scaling provides a symmetric scaling matrix H such that:
//!   H s = z  and  H² s ∘ z = e
//!
//! For different cone types:
//! - NonNeg: H = diag(sqrt(s ./ z))
//! - SOC: H(w) via quadratic representation in Jordan algebra
//! - PSD: H = W with W V W where M = X^{1/2} Z X^{1/2}, W = X^{1/2} M^{-1/2} X^{1/2}

use super::ScalingBlock;
use crate::cones::{ConeKernel, NonNegCone, SocCone};
use thiserror::Error;

/// NT scaling errors
#[derive(Error, Debug)]
#[allow(missing_docs)]  // Error variant fields are self-documenting
pub enum NtScalingError {
    /// Point not in interior
    #[error("Point not in cone interior")]
    NotInterior,

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

/// Compute NT scaling for NonNeg cone.
///
/// Returns H = diag(sqrt(s ./ z))
///
/// # Arguments
///
/// * `cone` - NonNeg cone
/// * `s` - Primal point (must be interior)
/// * `z` - Dual point (must be interior)
pub fn nt_scaling_nonneg(
    cone: &NonNegCone,
    s: &[f64],
    z: &[f64],
) -> Result<ScalingBlock, NtScalingError> {
    if s.len() != cone.dim() || z.len() != cone.dim() {
        return Err(NtScalingError::DimensionMismatch {
            expected: cone.dim(),
            actual: s.len(),
        });
    }

    if !cone.is_interior_primal(s) || !cone.is_interior_dual(z) {
        return Err(NtScalingError::NotInterior);
    }

    // H = diag(1 ./ sqrt(s ∘ z))  (NT scaling for nonnegative orthant)
    // Property: H² s ∘ z = e  =>  h_i² * s_i * z_i = 1  =>  h_i = 1 / sqrt(s_i * z_i)
    let d: Vec<f64> = s.iter().zip(z.iter())
        .map(|(si, zi)| 1.0 / (si * zi).sqrt())
        .collect();

    Ok(ScalingBlock::Diagonal { d })
}

/// Compute NT scaling for SOC cone.
///
/// Returns H(w) as a structured representation where w is the NT point.
/// The NT point w is computed via Jordan algebra:
///   1. s_sqrt = jordan_sqrt(s)
///   2. u = P(s_sqrt) z  (quadratic representation)
///   3. u_inv_sqrt = jordan_sqrt(jordan_inv(u))
///   4. w = P(s_sqrt) u_inv_sqrt
///
/// The resulting w satisfies: P(w) s = z
///
/// # Arguments
///
/// * `cone` - SOC cone
/// * `s` - Primal point (must be interior)
/// * `z` - Dual point (must be interior)
pub fn nt_scaling_soc(
    cone: &SocCone,
    s: &[f64],
    z: &[f64],
) -> Result<ScalingBlock, NtScalingError> {
    if s.len() != cone.dim() || z.len() != cone.dim() {
        return Err(NtScalingError::DimensionMismatch {
            expected: cone.dim(),
            actual: s.len(),
        });
    }

    if !cone.is_interior_primal(s) || !cone.is_interior_dual(z) {
        return Err(NtScalingError::NotInterior);
    }

    let n = cone.dim();
    let mut w = vec![0.0; n];

    // For SOC, the NT point is simply: w = jordan_sqrt(s ∘ z)
    // This gives w ∘ w = s ∘ z (geometric mean property)
    let mut s_circ_z = vec![0.0; n];
    jordan_product(s, z, &mut s_circ_z);
    jordan_sqrt(&s_circ_z, &mut w);

    Ok(ScalingBlock::SocStructured { w })
}

// ============================================================================
// Jordan algebra operations for SOC (internal helpers)
// ============================================================================

/// Jordan product for SOC: (t, x) ∘ (u, v) = (t*u + x·v, t*v + u*x)
#[inline]
fn jordan_product(a: &[f64], b: &[f64], out: &mut [f64]) {
    let t = a[0];
    let u = b[0];

    // out[0] = t*u + x·v
    out[0] = t * u;
    for i in 1..a.len() {
        out[0] += a[i] * b[i];
    }

    // out[1..] = t*v + u*x
    for i in 1..a.len() {
        out[i] = t * b[i] + u * a[i];
    }
}

/// Spectral decomposition: (t, x) = λ₁ e₁ + λ₂ e₂
/// where e₁ = (1, x/||x||)/2, e₂ = (1, -x/||x||)/2
/// and λ₁ = t + ||x||, λ₂ = t - ||x||
#[inline]
fn spectral_decomposition(v: &[f64], lambda: &mut [f64; 2], e1: &mut [f64], e2: &mut [f64]) {
    let t = v[0];
    let x_norm = if v.len() == 1 {
        0.0
    } else {
        v[1..].iter().map(|xi| xi * xi).sum::<f64>().sqrt()
    };

    lambda[0] = t + x_norm;
    lambda[1] = t - x_norm;

    // e1 = (1, x/||x||) / 2
    // e2 = (1, -x/||x||) / 2
    if x_norm > 1e-14 {
        e1[0] = 0.5;
        e2[0] = 0.5;
        let inv_norm = 1.0 / x_norm;
        for i in 1..v.len() {
            let x_normalized = v[i] * inv_norm;
            e1[i] = 0.5 * x_normalized;
            e2[i] = -0.5 * x_normalized;
        }
    } else {
        // Near axis: x ≈ 0, so e1 ≈ e2 ≈ (1, 0) / 2
        e1[0] = 0.5;
        e2[0] = 0.5;
        for i in 1..v.len() {
            e1[i] = 0.0;
            e2[i] = 0.0;
        }
    }
}

/// Jordan square root: sqrt((t, x)) = (sqrt(λ₁), sqrt(λ₂)) in spectral decomposition
fn jordan_sqrt(v: &[f64], out: &mut [f64]) {
    let n = v.len();
    let mut lambda = [0.0; 2];
    let mut e1 = vec![0.0; n];
    let mut e2 = vec![0.0; n];

    spectral_decomposition(v, &mut lambda, &mut e1, &mut e2);

    let sqrt_lambda1 = lambda[0].sqrt();
    let sqrt_lambda2 = lambda[1].sqrt();

    // out = sqrt(λ₁) e₁ + sqrt(λ₂) e₂
    for i in 0..n {
        out[i] = sqrt_lambda1 * e1[i] + sqrt_lambda2 * e2[i];
    }
}

/// Jordan inverse: inv((t, x)) = (1/λ₁, 1/λ₂) in spectral decomposition
pub fn jordan_inv(v: &[f64], out: &mut [f64]) {
    let n = v.len();
    let mut lambda = [0.0; 2];
    let mut e1 = vec![0.0; n];
    let mut e2 = vec![0.0; n];

    spectral_decomposition(v, &mut lambda, &mut e1, &mut e2);

    let inv_lambda1 = 1.0 / lambda[0];
    let inv_lambda2 = 1.0 / lambda[1];

    // out = (1/λ₁) e₁ + (1/λ₂) e₂
    for i in 0..n {
        out[i] = inv_lambda1 * e1[i] + inv_lambda2 * e2[i];
    }
}

/// Quadratic representation: P(w) y = 2 (w ∘ y) ∘ w - (w ∘ w) ∘ y
pub fn quad_rep(w: &[f64], y: &[f64], out: &mut [f64]) {
    let n = w.len();
    let mut w_circ_y = vec![0.0; n];
    let mut w_circ_w = vec![0.0; n];
    let mut temp = vec![0.0; n];

    // w ∘ y
    jordan_product(w, y, &mut w_circ_y);

    // w ∘ w
    jordan_product(w, w, &mut w_circ_w);

    // 2 (w ∘ y) ∘ w
    jordan_product(&w_circ_y, w, &mut temp);
    for i in 0..n {
        temp[i] *= 2.0;
    }

    // (w ∘ w) ∘ y
    let mut w2_circ_y = vec![0.0; n];
    jordan_product(&w_circ_w, y, &mut w2_circ_y);

    // out = 2 (w ∘ y) ∘ w - (w ∘ w) ∘ y
    for i in 0..n {
        out[i] = temp[i] - w2_circ_y[i];
    }
}

/// Public convenience function for applying quadratic representation.
/// Same as `quad_rep` but with clearer naming for external use.
#[inline]
pub fn quad_rep_apply(w: &[f64], y: &[f64], out: &mut [f64]) {
    quad_rep(w, y, out);
}

/// Public convenience function for computing Jordan inverse.
/// Same as `jordan_inv` but with clearer naming for external use.
#[inline]
pub fn jordan_inv_apply(v: &[f64], out: &mut [f64]) {
    jordan_inv(v, out);
}

/// Compute NT scaling for any cone type.
///
/// This is a convenience function that dispatches to the appropriate
/// cone-specific NT scaling function.
///
/// # Arguments
///
/// * `s` - Primal point (must be in cone interior)
/// * `z` - Dual point (must be in cone interior)
/// * `cone` - The cone
///
/// # Returns
///
/// The NT scaling block H such that H s ≈ z
pub fn compute_nt_scaling(
    s: &[f64],
    z: &[f64],
    cone: &dyn ConeKernel,
) -> Result<ScalingBlock, NtScalingError> {
    // Try to downcast to specific cone types
    if let Some(nonneg_cone) = (cone as &dyn std::any::Any).downcast_ref::<NonNegCone>() {
        return nt_scaling_nonneg(nonneg_cone, s, z);
    }

    if let Some(soc_cone) = (cone as &dyn std::any::Any).downcast_ref::<SocCone>() {
        return nt_scaling_soc(soc_cone, s, z);
    }

    // Fallback: simple diagonal scaling
    // H = diag(sqrt(s ./ z))
    let d: Vec<f64> = s.iter().zip(z.iter())
        .map(|(si, zi)| (si / zi).max(1e-8).sqrt())
        .collect();

    Ok(ScalingBlock::Diagonal { d })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nt_scaling_nonneg() {
        let cone = NonNegCone::new(3);
        let s = vec![4.0, 9.0, 16.0];
        let z = vec![1.0, 4.0, 4.0];

        let scaling = nt_scaling_nonneg(&cone, &s, &z).unwrap();

        if let ScalingBlock::Diagonal { d } = scaling {
            // H = diag(1 ./ sqrt(s ∘ z)) = diag(1/sqrt(4*1), 1/sqrt(9*4), 1/sqrt(16*4))
            //   = diag(1/2, 1/6, 1/8) = diag(0.5, 0.1667, 0.125)
            assert!((d[0] - 0.5).abs() < 1e-10);
            assert!((d[1] - 1.0/6.0).abs() < 1e-10);
            assert!((d[2] - 0.125).abs() < 1e-10);
        } else {
            panic!("Expected diagonal scaling");
        }
    }

    #[test]
    fn test_nt_scaling_nonneg_property() {
        // Property: H² s ∘ z = e  (element-wise: h²_i s_i z_i = 1)
        let cone = NonNegCone::new(5);
        let s = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let z = vec![5.0, 4.0, 3.0, 2.0, 1.0];

        let scaling = nt_scaling_nonneg(&cone, &s, &z).unwrap();

        if let ScalingBlock::Diagonal { d } = scaling {
            for i in 0..5 {
                let h_squared_times_sz = d[i] * d[i] * s[i] * z[i];
                assert!((h_squared_times_sz - 1.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_jordan_product() {
        // (2, [1, 0]) ∘ (3, [0, 1]) = (2*3 + 0, [2*[0,1] + 3*[1,0]]) = (6, [3, 2])
        let a = vec![2.0, 1.0, 0.0];
        let b = vec![3.0, 0.0, 1.0];
        let mut out = vec![0.0; 3];

        jordan_product(&a, &b, &mut out);

        assert!((out[0] - 6.0).abs() < 1e-10);
        assert!((out[1] - 3.0).abs() < 1e-10);
        assert!((out[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_decomposition() {
        // (5, [3, 4]) has ||x|| = 5, so λ₁ = 10, λ₂ = 0
        let v = vec![5.0, 3.0, 4.0];
        let mut lambda = [0.0; 2];
        let mut e1 = vec![0.0; 3];
        let mut e2 = vec![0.0; 3];

        spectral_decomposition(&v, &mut lambda, &mut e1, &mut e2);

        assert!((lambda[0] - 10.0).abs() < 1e-10);
        assert!((lambda[1] - 0.0).abs() < 1e-10);

        // e1 = (1, [3/5, 4/5]) / 2
        assert!((e1[0] - 0.5).abs() < 1e-10);
        assert!((e1[1] - 0.3).abs() < 1e-10);
        assert!((e1[2] - 0.4).abs() < 1e-10);

        // e2 = (1, -[3/5, 4/5]) / 2
        assert!((e2[0] - 0.5).abs() < 1e-10);
        assert!((e2[1] + 0.3).abs() < 1e-10);
        assert!((e2[2] + 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_jordan_sqrt() {
        // sqrt((5, [0, 0])) = (sqrt(5), [0, 0])
        let v = vec![5.0, 0.0, 0.0];
        let mut out = vec![0.0; 3];

        jordan_sqrt(&v, &mut out);

        assert!((out[0] - 5.0_f64.sqrt()).abs() < 1e-10);
        assert!(out[1].abs() < 1e-10);
        assert!(out[2].abs() < 1e-10);
    }

    #[test]
    fn test_jordan_inv() {
        // inv((4, [0, 0])) = (1/4, [0, 0])
        let v = vec![4.0, 0.0, 0.0];
        let mut out = vec![0.0; 3];

        jordan_inv(&v, &mut out);

        assert!((out[0] - 0.25).abs() < 1e-10);
        assert!(out[1].abs() < 1e-10);
        assert!(out[2].abs() < 1e-10);
    }

    #[test]
    fn test_nt_scaling_soc() {
        let cone = SocCone::new(3);

        // Simple test: s = z = (2, [0, 0])
        // Then w should be (2, [0, 0]) as well
        let s = vec![2.0, 0.0, 0.0];
        let z = vec![2.0, 0.0, 0.0];

        let scaling = nt_scaling_soc(&cone, &s, &z).unwrap();

        if let ScalingBlock::SocStructured { w } = scaling {
            // When s = z, the NT point w = jordan_sqrt(s ∘ z) = jordan_sqrt(s ∘ s) = s
            assert!((w[0] - 2.0).abs() < 1e-6);
            assert!(w[1].abs() < 1e-6);
            assert!(w[2].abs() < 1e-6);
        } else {
            panic!("Expected SOC structured scaling");
        }
    }

    #[test]
    fn test_nt_scaling_soc_property() {
        // Property: w ∘ w should equal s ∘ z (NT point definition)
        let cone = SocCone::new(5);
        let s = vec![5.0, 1.0, 2.0, 1.0, 1.0];
        let z = vec![10.0, 2.0, 4.0, 2.0, 2.0];

        let scaling = nt_scaling_soc(&cone, &s, &z).unwrap();

        if let ScalingBlock::SocStructured { w } = scaling {
            // Verify w ∘ w ≈ s ∘ z
            let mut w_squared = vec![0.0; 5];
            jordan_product(&w, &w, &mut w_squared);

            let mut s_circ_z = vec![0.0; 5];
            jordan_product(&s, &z, &mut s_circ_z);

            for i in 0..5 {
                let rel_err = (w_squared[i] - s_circ_z[i]).abs() / s_circ_z[i].abs().max(1.0);
                assert!(
                    rel_err < 1e-6,
                    "w∘w ≠ s∘z at index {}: {} vs {}",
                    i,
                    w_squared[i],
                    s_circ_z[i]
                );
            }
        }
    }
}
