//! BFGS primal-dual scaling for nonsymmetric cones.
//!
//! This module implements quasi-Newton scaling for nonsymmetric cones
//! (Exponential and Power cones). Two approaches are available:
//!
//! 1. **Rank-3 scaling** (Clarabel-style, default): More stable and efficient
//!    - Hs = s·s^T/⟨s,z⟩ + δs·δs^T/⟨δs,δz⟩ + t·axis·axis^T
//!    - Uses perturbed iterates and orthogonal axis
//!    - Falls back to dual-only scaling when stability checks fail
//!
//! 2. **Rank-4 scaling** (Tunçel's general formula): More general but less stable
//!    - H = Z(Z^T S)^(-1)Z^T + H_a - H_a S(S^T H_a S)^(-1)S^T H_a
//!    - Uses shadow iterates directly
//!
//! The resulting H approximately satisfies:
//!   H z = s,  H ∇f(s) = ∇f*(z)
//! where z̃ = -∇f(s), s̃ = -∇f*(z).

use crate::cones::ConeKernel;
use crate::scaling::ScalingBlock;
use nalgebra::DMatrix;
use nalgebra::linalg::SymmetricEigen;
use thiserror::Error;

/// Errors that can arise while computing BFGS scaling.
#[derive(Debug, Error)]
#[allow(missing_docs)]
pub enum BfgsScalingError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("failed to compute dual map")]
    DualMapFailed,
}

/// Compute BFGS scaling for a single 3D nonsymmetric cone block.
///
/// This uses the rank-3 formula by default (Clarabel-style), with fallback
/// to dual-only scaling (μ*H_dual) if stability checks fail.
pub fn bfgs_scaling_3d(
    s: &[f64],
    z: &[f64],
    cone: &dyn ConeKernel,
) -> Result<ScalingBlock, BfgsScalingError> {
    // Try rank-3 scaling first (more stable), fallback to dual-only scaling if needed
    match bfgs_scaling_3d_rank3(s, z, cone) {
        Ok(scaling) => Ok(scaling),
        Err(_) => {
            // Fallback: use dual-only scaling Hs = μ * H_dual (like Clarabel)
            // This is more accurate than μ*I and matches Clarabel's fallback
            dual_only_scaling_3d(s, z, cone)
        }
    }
}

/// Compute dual-only scaling: Hs = μ * H_dual
///
/// This is Clarabel's fallback when primal-dual scaling fails.
/// Uses the actual Hessian of the dual barrier, scaled by μ.
fn dual_only_scaling_3d(
    s: &[f64],
    z: &[f64],
    cone: &dyn ConeKernel,
) -> Result<ScalingBlock, BfgsScalingError> {
    let s_dot_z = dot3(s, z);
    let mu = (s_dot_z / 3.0).abs().max(1e-12);

    // Get the Hessian of the dual barrier at z
    let mut s_tilde = [0.0; 3];
    let mut h_dual = [0.0; 9];
    cone.dual_map(z, &mut s_tilde, &mut h_dual);

    // Hs = μ * H_dual
    let mut h = [0.0; 9];
    for i in 0..9 {
        h[i] = mu * h_dual[i];
    }

    // Symmetrize (should be symmetric, but numerical errors)
    let h = symmetrize_mat3(&h);

    // Sanity check: ensure H is positive definite
    let min_eig = min_eigenvalue(&h);
    if !min_eig.is_finite() || min_eig <= 1e-12 {
        // Last resort: use μ*I
        return Ok(ScalingBlock::Dense3x3 {
            h: [mu, 0.0, 0.0, 0.0, mu, 0.0, 0.0, 0.0, mu],
        });
    }

    Ok(ScalingBlock::Dense3x3 { h })
}

/// Compute rank-3 BFGS scaling (Clarabel-style).
///
/// Formula: Hs = s·s^T/⟨s,z⟩ + δs·δs^T/⟨δs,δz⟩ + t·axis·axis^T
///
/// where:
/// - δs = s + μ·s̃ (perturbed primal)
/// - δz = z + μ·z̃ (perturbed dual)
/// - axis = cross(z, z̃) / ||cross(z, z̃)|| (orthogonal direction)
/// - t is a computed scaling coefficient
///
/// This approach is more numerically stable and efficient than rank-4.
fn bfgs_scaling_3d_rank3(
    s: &[f64],
    z: &[f64],
    cone: &dyn ConeKernel,
) -> Result<ScalingBlock, BfgsScalingError> {
    if s.len() != 3 || z.len() != 3 {
        return Err(BfgsScalingError::DimensionMismatch {
            expected: 3,
            actual: s.len().max(z.len()),
        });
    }

    // Compute shadow iterates
    let mut grad_primal = [0.0; 3];
    cone.barrier_grad_primal(s, &mut grad_primal);
    let z_tilde = [-grad_primal[0], -grad_primal[1], -grad_primal[2]];

    let mut s_tilde = [0.0; 3];
    let mut h_dual = [0.0; 9];
    cone.dual_map(z, &mut s_tilde, &mut h_dual);

    // Compute barrier parameters
    let s_dot_z = dot3(s, z);
    let st_dot_zt = dot3(&s_tilde, &z_tilde);
    let mu = s_dot_z / 3.0;
    let mu_tilde = st_dot_zt / 3.0;

    // Stability checks (from Clarabel)
    let eps = 1e-10;
    let eps_sqrt = 1e-5;

    // Check 1: Centrality (|μ·μ̃ - 1| > √ε)
    let de1 = (mu * mu_tilde - 1.0).abs();
    if de1 <= eps_sqrt {
        return Err(BfgsScalingError::DualMapFailed);  // Too close to central path
    }

    // Check 2: Definiteness (H_dual.quad_form(z̃, z̃) - 3μ̃² > ε)
    let ht_zt = mat3_vec(&h_dual, &z_tilde);
    let quad_form = dot3(&z_tilde, &ht_zt);
    let de2 = quad_form - 3.0 * mu_tilde * mu_tilde;
    if de2 <= eps {
        return Err(BfgsScalingError::DualMapFailed);  // Not positive definite enough
    }

    // Compute perturbed iterates
    let s_arr = [s[0], s[1], s[2]];
    let z_arr = [z[0], z[1], z[2]];
    let delta_s = add_vec(&s_arr, &scale_vec(&s_tilde, mu));
    let delta_z = add_vec(&z_arr, &scale_vec(&z_tilde, mu));

    // Check 3: Positivity (⟨s,z⟩ > 0 and ⟨δs,δz⟩ > 0)
    let ds_dot_dz = dot3(&delta_s, &delta_z);
    if s_dot_z <= 0.0 || ds_dot_dz <= 0.0 {
        return Err(BfgsScalingError::DualMapFailed);  // Lost positivity
    }

    // Compute orthogonal axis via cross product
    let axis_z = cross_product(&z_arr, &z_tilde);
    let axis_norm = norm3(&axis_z);
    if axis_norm < eps {
        return Err(BfgsScalingError::DualMapFailed);  // Vectors are parallel
    }
    let axis_z_normalized = scale_vec(&axis_z, 1.0 / axis_norm);

    // Compute scaling coefficient t
    // t = μ · ||H_dual - s̃·s̃^T/3 - tmp·tmp^T/de2||_F
    let mut h_correction = h_dual;

    // Subtract s̃·s̃^T/3
    for i in 0..3 {
        for j in 0..3 {
            h_correction[3*i + j] -= s_tilde[i] * s_tilde[j] / 3.0;
        }
    }

    // Compute tmp = H_dual·z̃ - μ̃·s̃
    let h_zt = mat3_vec(&h_dual, &z_tilde);
    let tmp = [
        h_zt[0] - mu_tilde * s_tilde[0],
        h_zt[1] - mu_tilde * s_tilde[1],
        h_zt[2] - mu_tilde * s_tilde[2],
    ];

    // Subtract tmp·tmp^T/de2
    for i in 0..3 {
        for j in 0..3 {
            h_correction[3*i + j] -= tmp[i] * tmp[j] / de2;
        }
    }

    // Frobenius norm
    let frobenius_norm_sq: f64 = h_correction.iter().map(|x| x * x).sum();
    let t = mu * frobenius_norm_sq.sqrt();

    // Build rank-3 scaling: Hs = s·s^T/⟨s,z⟩ + δs·δs^T/⟨δs,δz⟩ + t·axis·axis^T
    let mut h = [0.0; 9];

    // Term 1: s·s^T/⟨s,z⟩
    for i in 0..3 {
        for j in 0..3 {
            h[3*i + j] += s_arr[i] * s_arr[j] / s_dot_z;
        }
    }

    // Term 2: δs·δs^T/⟨δs,δz⟩
    for i in 0..3 {
        for j in 0..3 {
            h[3*i + j] += delta_s[i] * delta_s[j] / ds_dot_dz;
        }
    }

    // Term 3: t·axis·axis^T
    for i in 0..3 {
        for j in 0..3 {
            h[3*i + j] += t * axis_z_normalized[i] * axis_z_normalized[j];
        }
    }

    // Symmetrize (should already be symmetric, but numerical errors)
    let h = symmetrize_mat3(&h);

    // Ensure bounded condition number for numerical stability
    // The KKT system can become ill-conditioned if H has extreme eigenvalue spread.
    // We limit the condition number to prevent overflow in the solve.
    let min_eig = min_eigenvalue(&h);
    let max_eig = max_eigenvalue(&h);
    let max_cond = 1e6;  // Allow higher condition number (dual-only fallback handles edge cases)

    if !min_eig.is_finite() || !max_eig.is_finite() || min_eig <= 1e-12 {
        // Return error so outer function uses dual-only scaling fallback
        return Err(BfgsScalingError::DualMapFailed);
    }

    let cond = max_eig / min_eig.max(1e-15);
    if cond > max_cond {
        // Condition number too high - return error for dual-only fallback
        return Err(BfgsScalingError::DualMapFailed);
    }

    Ok(ScalingBlock::Dense3x3 { h })
}

/// Compute rank-4 BFGS scaling (Tunçel's general formula).
///
/// This is the original implementation using the general rank-4 formula.
/// Less stable than rank-3 but more general.
fn bfgs_scaling_3d_rank4(
    s: &[f64],
    z: &[f64],
    cone: &dyn ConeKernel,
) -> Result<ScalingBlock, BfgsScalingError> {
    if s.len() != 3 || z.len() != 3 {
        return Err(BfgsScalingError::DimensionMismatch {
            expected: 3,
            actual: s.len().max(z.len()),
        });
    }

    let mut grad = [0.0; 3];
    cone.barrier_grad_primal(s, &mut grad);
    let z_tilde = [-grad[0], -grad[1], -grad[2]];

    let mut s_tilde = [0.0; 3];
    let mut h_star = [0.0; 9];
    cone.dual_map(z, &mut s_tilde, &mut h_star);

    let s_dot_z = dot3(s, z);
    let mut mu = (s_dot_z / 3.0).abs();
    if !mu.is_finite() || mu <= 1e-12 {
        mu = 1.0;
    }

    let mut h_a = [0.0; 9];
    for i in 0..9 {
        h_a[i] = mu * h_star[i];
    }

    // Fallback to identity-like scaling for early returns
    let mu_fallback = mu.max(1.0);
    let fallback = [mu_fallback, 0.0, 0.0, 0.0, mu_fallback, 0.0, 0.0, 0.0, mu_fallback];

    let zts = [
        dot3(z, s),
        dot3(z, &s_tilde),
        dot3(&z_tilde, s),
        dot3(&z_tilde, &s_tilde),
    ];
    let Some(inv_zts) = inv_2x2(zts) else {
        return Ok(ScalingBlock::Dense3x3 { h: fallback });
    };

    let hs0 = mat3_vec(&h_a, s);
    let hs1 = mat3_vec(&h_a, &s_tilde);

    let shas = [
        dot3(s, &hs0),
        dot3(s, &hs1),
        dot3(&s_tilde, &hs0),
        dot3(&s_tilde, &hs1),
    ];
    let Some(inv_shas) = inv_2x2(shas) else {
        return Ok(ScalingBlock::Dense3x3 { h: fallback });
    };

    let col0 = add_vec(&scale_vec(z, inv_zts[0]), &scale_vec(&z_tilde, inv_zts[2]));
    let col1 = add_vec(&scale_vec(z, inv_zts[1]), &scale_vec(&z_tilde, inv_zts[3]));
    let term1 = outer_sum(&col0, z, &col1, &z_tilde);

    let temp0 = add_vec(&scale_vec(&hs0, inv_shas[0]), &scale_vec(&hs1, inv_shas[2]));
    let temp1 = add_vec(&scale_vec(&hs0, inv_shas[1]), &scale_vec(&hs1, inv_shas[3]));
    let term2 = outer_sum(&temp0, &hs0, &temp1, &hs1);

    let mut h = [0.0; 9];
    for i in 0..9 {
        h[i] = term1[i] + h_a[i] - term2[i];
    }

    let h = symmetrize_mat3(&h);
    let min_eig = min_eigenvalue(&h);
    let max_eig = max_eigenvalue(&h);
    let max_cond = 1e3;  // Maximum allowed condition number (very tight to avoid KKT overflow)

    if !min_eig.is_finite() || !max_eig.is_finite() || min_eig <= 1e-10 {
        return Ok(ScalingBlock::Dense3x3 { h: fallback });
    }

    let cond = max_eig / min_eig.max(1e-15);
    if cond > max_cond {
        // Condition number way too high - use identity fallback
        return Ok(ScalingBlock::Dense3x3 { h: fallback });
    }

    Ok(ScalingBlock::Dense3x3 { h })
}

fn dot3(a: &[f64], b: &[f64]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn inv_2x2(m: [f64; 4]) -> Option<[f64; 4]> {
    let det = m[0] * m[3] - m[1] * m[2];
    if !det.is_finite() || det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([m[3] * inv_det, -m[1] * inv_det, -m[2] * inv_det, m[0] * inv_det])
}

fn mat3_vec(h: &[f64; 9], v: &[f64]) -> [f64; 3] {
    [
        h[0] * v[0] + h[1] * v[1] + h[2] * v[2],
        h[3] * v[0] + h[4] * v[1] + h[5] * v[2],
        h[6] * v[0] + h[7] * v[1] + h[8] * v[2],
    ]
}

fn scale_vec(v: &[f64], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn add_vec(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn outer_sum(a0: &[f64; 3], b0: &[f64], a1: &[f64; 3], b1: &[f64]) -> [f64; 9] {
    let mut out = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            out[3 * i + j] = a0[i] * b0[j] + a1[i] * b1[j];
        }
    }
    out
}

fn cross_product(a: &[f64], b: &[f64]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm3(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn symmetrize_mat3(h: &[f64; 9]) -> [f64; 9] {
    let mut out = *h;
    for i in 0..3 {
        for j in (i + 1)..3 {
            let avg = 0.5 * (h[3 * i + j] + h[3 * j + i]);
            out[3 * i + j] = avg;
            out[3 * j + i] = avg;
        }
    }
    out
}

fn min_eigenvalue(h: &[f64; 9]) -> f64 {
    let m = DMatrix::<f64>::from_row_slice(3, 3, h);
    let eig = SymmetricEigen::new(m);
    eig.eigenvalues.iter().copied().fold(f64::INFINITY, f64::min)
}

fn max_eigenvalue(h: &[f64; 9]) -> f64 {
    let m = DMatrix::<f64>::from_row_slice(3, 3, h);
    let eig = SymmetricEigen::new(m);
    eig.eigenvalues.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}
