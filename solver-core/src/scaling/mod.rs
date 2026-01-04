//! Scaling matrices for cone IPM.
//!
//! This module implements scaling updates for symmetric cones (Nesterov-Todd)
//! and nonsymmetric cones (BFGS primal-dual scaling).

pub mod bfgs;
pub mod nt;

/// Scaling block representation for the H matrix in the KKT system.
#[derive(Debug, Clone)]
#[allow(missing_docs)] // Enum variant fields are self-documenting
pub enum ScalingBlock {
    /// Zero cone (no scaling needed)
    Zero { dim: usize },

    /// Diagonal scaling (for NonNeg cone)
    Diagonal { d: Vec<f64> },

    /// Dense 3×3 scaling (for EXP/POW cones)
    Dense3x3 { h: [f64; 9] },

    /// Structured SOC scaling (quadratic representation)
    SocStructured { w: Vec<f64>, diag_reg: f64 },

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
            ScalingBlock::SocStructured { w, diag_reg } => {
                // H(w) v = P(w) v + diag_reg * v
                nt::quad_rep_apply(w, v, out);
                if *diag_reg != 0.0 {
                    for i in 0..v.len() {
                        out[i] += diag_reg * v[i];
                    }
                }
            }
            ScalingBlock::PsdStructured { .. } => {
                unimplemented!("PSD structured scaling not yet implemented")
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
                let det = h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6])
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
            ScalingBlock::SocStructured { w, diag_reg } => {
                // H(w)^{-1} v = (P(w) + diag_reg*I)^{-1} v
                // Use a small CG solve when diag_reg > 0 (kept small for perf).
                let n = w.len();
                if *diag_reg == 0.0 {
                    let mut w_inv = vec![0.0; n];
                    nt::jordan_inv_apply(w, &mut w_inv);
                    nt::quad_rep_apply(&w_inv, v, out);
                } else {
                    // Conjugate gradient on SPD operator: (P(w) + diag_reg I)
                    let mut x = vec![0.0; n];
                    let mut r = v.to_vec();
                    let mut p = r.clone();
                    let mut ap = vec![0.0; n];
                    let mut rs_old = r.iter().map(|ri| ri * ri).sum::<f64>();

                    for _ in 0..8 {
                        nt::quad_rep_apply(w, &p, &mut ap);
                        for i in 0..n {
                            ap[i] += diag_reg * p[i];
                        }
                        let denom = p
                            .iter()
                            .zip(ap.iter())
                            .map(|(pi, api)| pi * api)
                            .sum::<f64>();
                        if denom.abs() < 1e-18 {
                            break;
                        }
                        let alpha = rs_old / denom;
                        for i in 0..n {
                            x[i] += alpha * p[i];
                            r[i] -= alpha * ap[i];
                        }
                        let rs_new = r.iter().map(|ri| ri * ri).sum::<f64>();
                        if rs_new < 1e-20 * rs_old.max(1.0) {
                            break;
                        }
                        let beta = rs_new / rs_old;
                        for i in 0..n {
                            p[i] = r[i] + beta * p[i];
                        }
                        rs_old = rs_new;
                    }

                    out.copy_from_slice(&x);
                }
            }
            ScalingBlock::PsdStructured { .. } => {
                unimplemented!("PSD structured scaling inverse not yet implemented")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ScalingBlock;

    #[test]
    fn test_soc_scaling_diag_reg_apply() {
        let block = ScalingBlock::SocStructured {
            w: vec![1.0, 0.0, 0.0],
            diag_reg: 0.5,
        };

        let v = vec![2.0, -1.0, 4.0];
        let mut out = vec![0.0; 3];
        block.apply(&v, &mut out);

        // For w = (1,0,0), P(w) is identity, so H v = (1 + diag_reg) * v.
        assert!((out[0] - 3.0).abs() < 1e-12);
        assert!((out[1] + 1.5).abs() < 1e-12);
        assert!((out[2] - 6.0).abs() < 1e-12);
    }
}
