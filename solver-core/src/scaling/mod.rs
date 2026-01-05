//! Scaling matrices for cone IPM.
//!
//! This module implements scaling updates for symmetric cones (Nesterov-Todd)
//! and nonsymmetric cones (BFGS primal-dual scaling).

pub mod nt;
pub mod bfgs;

use crate::cones::psd::{mat_to_svec, svec_to_mat};
use nalgebra::DMatrix;
use nalgebra::linalg::SymmetricEigen;

/// Scaling block representation for the H matrix in the KKT system.
#[derive(Debug, Clone)]
#[allow(missing_docs)]  // Enum variant fields are self-documenting
pub enum ScalingBlock {
    /// Zero cone (no scaling needed)
    Zero { dim: usize },

    /// Diagonal scaling (for NonNeg cone)
    Diagonal { d: Vec<f64> },

    /// Dense 3×3 scaling (for EXP/POW cones)
    Dense3x3 { h: [f64; 9] },

    /// Structured SOC scaling (quadratic representation)
    SocStructured { w: Vec<f64> },

    /// Structured PSD scaling (W factor)
    PsdStructured { w_factor: Vec<f64>, n: usize },
}

impl ScalingBlock {
    /// Apply H to a vector: out = H * v
    pub fn apply(&self, v: &[f64], out: &mut [f64]) {
        match self {
            ScalingBlock::Zero { .. } => {
                // H = 0 for zero cone
                out.fill(0.0);
            }
            ScalingBlock::Diagonal { d } => {
                for i in 0..d.len() {
                    out[i] = d[i] * v[i];
                }
            }
            ScalingBlock::Dense3x3 { h } => {
                // 3×3 dense matrix-vector product (row-major)
                out[0] = h[0] * v[0] + h[1] * v[1] + h[2] * v[2];
                out[1] = h[3] * v[0] + h[4] * v[1] + h[5] * v[2];
                out[2] = h[6] * v[0] + h[7] * v[1] + h[8] * v[2];
            }
            ScalingBlock::SocStructured { w } => {
                // H(w) v = P(w) v (quadratic representation)
                nt::quad_rep_apply(w, v, out);
            }
            ScalingBlock::PsdStructured { w_factor, n } => {
                let w = DMatrix::<f64>::from_row_slice(*n, *n, w_factor);
                let v_mat = svec_to_mat(v, *n);
                let out_mat = &w * v_mat * &w;
                mat_to_svec(&out_mat, out);
            }
        }
    }

    /// Apply H^{-1} to a vector: out = H^{-1} * v
    pub fn apply_inv(&self, v: &[f64], out: &mut [f64]) {
        match self {
            ScalingBlock::Zero { .. } => {
                // H^{-1} undefined for zero cone (should not be called)
                panic!("Cannot apply inverse scaling to zero cone");
            }
            ScalingBlock::Diagonal { d } => {
                for i in 0..d.len() {
                    out[i] = v[i] / d[i];
                }
            }
            ScalingBlock::Dense3x3 { h } => {
                // Solve 3×3 system (use direct formula or small LU)
                // For now, use Cramer's rule (to be optimized)
                let det = h[0] * (h[4] * h[8] - h[5] * h[7])
                    - h[1] * (h[3] * h[8] - h[5] * h[6])
                    + h[2] * (h[3] * h[7] - h[4] * h[6]);

                let inv_det = 1.0 / det;

                let h_inv = [
                    (h[4] * h[8] - h[5] * h[7]) * inv_det,
                    (h[2] * h[7] - h[1] * h[8]) * inv_det,
                    (h[1] * h[5] - h[2] * h[4]) * inv_det,
                    (h[5] * h[6] - h[3] * h[8]) * inv_det,
                    (h[0] * h[8] - h[2] * h[6]) * inv_det,
                    (h[2] * h[3] - h[0] * h[5]) * inv_det,
                    (h[3] * h[7] - h[4] * h[6]) * inv_det,
                    (h[1] * h[6] - h[0] * h[7]) * inv_det,
                    (h[0] * h[4] - h[1] * h[3]) * inv_det,
                ];

                out[0] = h_inv[0] * v[0] + h_inv[1] * v[1] + h_inv[2] * v[2];
                out[1] = h_inv[3] * v[0] + h_inv[4] * v[1] + h_inv[5] * v[2];
                out[2] = h_inv[6] * v[0] + h_inv[7] * v[1] + h_inv[8] * v[2];
            }
            ScalingBlock::SocStructured { w } => {
                // H(w)^{-1} v = P(w^{-1}) v
                // First compute w_inv = jordan_inv(w)
                let mut w_inv = vec![0.0; w.len()];
                nt::jordan_inv_apply(w, &mut w_inv);
                // Then apply P(w_inv) to v
                nt::quad_rep_apply(&w_inv, v, out);
            }
            ScalingBlock::PsdStructured { w_factor, n } => {
                let w = DMatrix::<f64>::from_row_slice(*n, *n, w_factor);
                let w_inv = w.clone().try_inverse().unwrap_or_else(|| {
                    let eig = SymmetricEigen::new(w);
                    let inv_vals = eig.eigenvalues.map(|v| 1.0 / v.max(1e-18));
                    &eig.eigenvectors
                        * DMatrix::<f64>::from_diagonal(&inv_vals)
                        * eig.eigenvectors.transpose()
                });
                let v_mat = svec_to_mat(v, *n);
                let out_mat = &w_inv * v_mat * &w_inv;
                mat_to_svec(&out_mat, out);
            }
        }
    }
}
