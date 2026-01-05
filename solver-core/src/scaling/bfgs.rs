//! BFGS primal-dual scaling for nonsymmetric cones.
//!
//! This module implements the BFGS-based quasi-Newton scaling for
//! nonsymmetric cones (Exponential and Power cones). The scaling
//! is constructed using shadow points and a BFGS rank-2 update.
//!
//! The resulting H satisfies:
//!   H z = s,  H \tilde z = \tilde s
//! where \tilde z = -∇f(s), \tilde s = -∇f*(z).

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
pub fn bfgs_scaling_3d(
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

    let zts = [
        dot3(z, s),
        dot3(z, &s_tilde),
        dot3(&z_tilde, s),
        dot3(&z_tilde, &s_tilde),
    ];
    let Some(inv_zts) = inv_2x2(zts) else {
        return Ok(ScalingBlock::Dense3x3 { h: symmetrize_mat3(&h_a) });
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
        return Ok(ScalingBlock::Dense3x3 { h: symmetrize_mat3(&h_a) });
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

    let mut h = symmetrize_mat3(&h);
    let min_eig = min_eigenvalue(&h);
    if !min_eig.is_finite() || min_eig <= 1e-10 {
        let shift = (1e-6 - min_eig).max(1e-6);
        h[0] += shift;
        h[4] += shift;
        h[8] += shift;
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
