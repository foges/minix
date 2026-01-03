//! Dual certificates for K* cut generation.
//!
//! When a conic subproblem is infeasible, the dual variables provide a
//! certificate that can be used to generate valid cuts.

use solver_core::ConeSpec;

use crate::master::LinearCut;

/// Dual certificate from a conic subproblem.
///
/// For problem: min q^T x s.t. Ax + s = b, s ∈ K
/// A dual certificate y satisfies: y ∈ K* (dual cone)
///
/// Any y ∈ K* gives a valid inequality: (A^T y)^T x <= b^T y
#[derive(Debug, Clone)]
pub struct DualCertificate {
    /// Full dual vector y (length m).
    pub y: Vec<f64>,

    /// Per-cone-block dual information.
    pub cone_duals: Vec<ConeDual>,

    /// Certificate value b^T y.
    pub certificate_value: f64,
}

/// Dual information for a single cone block.
#[derive(Debug, Clone)]
pub struct ConeDual {
    /// Index of the cone in the problem's cone list.
    pub cone_idx: usize,

    /// Type of cone.
    pub cone_type: ConeSpec,

    /// Starting offset of this cone in the constraint vector.
    pub offset: usize,

    /// Dimension of this cone block.
    pub dim: usize,

    /// Dual vector for this block (slice of full y).
    pub y_block: Vec<f64>,

    /// Violation of this block: -y^T (b - Ax) for the block.
    /// Positive means the block contributes to infeasibility.
    pub violation: f64,
}

impl DualCertificate {
    /// Create a dual certificate from a full dual vector and cone specification.
    pub fn from_dual(
        z: &[f64],
        b: &[f64],
        cones: &[ConeSpec],
    ) -> Self {
        let mut cone_duals = Vec::with_capacity(cones.len());
        let mut offset = 0;

        for (idx, cone) in cones.iter().enumerate() {
            let dim = cone.dim();
            if dim == 0 {
                continue;
            }

            let y_block: Vec<f64> = z[offset..offset + dim].to_vec();
            let b_block = &b[offset..offset + dim];

            // Compute b_block^T y_block
            let block_value: f64 = b_block
                .iter()
                .zip(&y_block)
                .map(|(bi, yi)| bi * yi)
                .sum();

            cone_duals.push(ConeDual {
                cone_idx: idx,
                cone_type: cone.clone(),
                offset,
                dim,
                y_block,
                violation: block_value, // Will be updated with actual violation
            });

            offset += dim;
        }

        // Total certificate value
        let certificate_value: f64 = z.iter().zip(b.iter()).map(|(yi, bi)| yi * bi).sum();

        Self {
            y: z.to_vec(),
            cone_duals,
            certificate_value,
        }
    }

    /// Update violations based on current slack values s = b - Ax.
    ///
    /// Violation for block i: -y_i^T s_i (positive means violated).
    pub fn update_violations(&mut self, s: &[f64]) {
        for cone_dual in &mut self.cone_duals {
            let s_block = &s[cone_dual.offset..cone_dual.offset + cone_dual.dim];
            let violation: f64 = cone_dual
                .y_block
                .iter()
                .zip(s_block)
                .map(|(yi, si)| -yi * si)
                .sum();
            cone_dual.violation = violation;
        }
    }

    /// Check if the dual vector is valid (all components in respective dual cones).
    ///
    /// For self-dual cones (NonNeg, SOC), y must be in the cone.
    /// For Zero cone, the dual is all of R^n (no constraint).
    pub fn is_valid(&self) -> bool {
        for cone_dual in &self.cone_duals {
            if !is_in_dual_cone(&cone_dual.y_block, &cone_dual.cone_type) {
                return false;
            }
        }
        true
    }

    /// Get cone blocks sorted by violation (most violated first).
    pub fn sorted_by_violation(&self) -> Vec<&ConeDual> {
        let mut sorted: Vec<&ConeDual> = self.cone_duals.iter().collect();
        sorted.sort_by(|a, b| {
            b.violation
                .partial_cmp(&a.violation)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// Get the most violated cone blocks (up to max_blocks).
    pub fn most_violated(&self, max_blocks: usize) -> Vec<&ConeDual> {
        self.sorted_by_violation()
            .into_iter()
            .filter(|cd| cd.violation > 1e-8)
            .take(max_blocks)
            .collect()
    }
}

/// Check if a vector is in the dual cone.
///
/// For self-dual cones (NonNeg, SOC, PSD), dual cone = primal cone.
/// For Zero cone, dual cone = R^n (all vectors valid).
fn is_in_dual_cone(y: &[f64], cone: &ConeSpec) -> bool {
    match cone {
        ConeSpec::Zero { .. } => {
            // Dual of {0}^n is R^n - all vectors valid
            true
        }
        ConeSpec::NonNeg { .. } => {
            // Self-dual: y >= 0
            y.iter().all(|&yi| yi >= -1e-10)
        }
        ConeSpec::Soc { .. } => {
            // Self-dual: y[0] >= ||y[1:]||
            if y.is_empty() {
                return true;
            }
            let t = y[0];
            let x_norm_sq: f64 = y[1..].iter().map(|xi| xi * xi).sum();
            t >= x_norm_sq.sqrt() - 1e-10
        }
        ConeSpec::Psd { .. } => {
            // Self-dual (checking requires eigendecomposition, skip for now)
            true
        }
        ConeSpec::Exp { .. } | ConeSpec::Pow { .. } => {
            // Non-self-dual cones - would need proper dual cone check
            // For now, assume valid
            true
        }
    }
}

/// Extract cuts from a sparse matrix and dual certificate.
pub struct CutExtractor {
    /// Number of variables.
    n: usize,
}

impl CutExtractor {
    /// Create a new cut extractor.
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    /// Generate a single K* cut from the full certificate.
    ///
    /// Cut: (A^T y)^T x <= b^T y
    pub fn extract_full_cut(
        &self,
        cert: &DualCertificate,
        a: &sprs::CsMat<f64>,
        b: &[f64],
    ) -> LinearCut {
        // Compute a_cut = A^T y
        let mut a_cut = vec![0.0; self.n];
        for (col_idx, col) in a.outer_iterator().enumerate() {
            for (row_idx, &val) in col.iter() {
                a_cut[col_idx] += val * cert.y[row_idx];
            }
        }

        // rhs = b^T y
        let rhs: f64 = b.iter().zip(&cert.y).map(|(bi, yi)| bi * yi).sum();

        let mut cut = LinearCut::new(
            a_cut,
            rhs,
            crate::master::CutSource::KStarCertificate { cone_idx: 0 },
        );
        cut.normalize();
        cut
    }

    /// Generate disaggregated cuts (one per cone block).
    ///
    /// For each cone block i with dual y_i:
    /// Cut: (A_i^T y_i)^T x <= b_i^T y_i
    pub fn extract_disaggregated_cuts(
        &self,
        cert: &DualCertificate,
        a: &sprs::CsMat<f64>,
        b: &[f64],
        max_cuts: usize,
    ) -> Vec<LinearCut> {
        let mut cuts = Vec::new();

        // Get most violated blocks
        let violated_blocks = cert.most_violated(max_cuts);

        for cone_dual in violated_blocks {
            let offset = cone_dual.offset;
            let dim = cone_dual.dim;

            // Compute a_cut = A_block^T y_block
            let mut a_cut = vec![0.0; self.n];
            for (col_idx, col) in a.outer_iterator().enumerate() {
                for (row_idx, &val) in col.iter() {
                    if row_idx >= offset && row_idx < offset + dim {
                        let local_idx = row_idx - offset;
                        a_cut[col_idx] += val * cone_dual.y_block[local_idx];
                    }
                }
            }

            // rhs = b_block^T y_block
            let b_block = &b[offset..offset + dim];
            let rhs: f64 = b_block
                .iter()
                .zip(&cone_dual.y_block)
                .map(|(bi, yi)| bi * yi)
                .sum();

            let mut cut = LinearCut::new(
                a_cut,
                rhs,
                crate::master::CutSource::Disaggregated {
                    cone_idx: cone_dual.cone_idx,
                    block: 0,
                },
            );
            cut.normalize();

            if cut.is_valid() {
                cuts.push(cut);
            }
        }

        cuts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_cone_membership() {
        // NonNeg: y >= 0
        assert!(is_in_dual_cone(&[1.0, 2.0, 0.0], &ConeSpec::NonNeg { dim: 3 }));
        assert!(!is_in_dual_cone(&[1.0, -0.1, 0.0], &ConeSpec::NonNeg { dim: 3 }));

        // SOC: y[0] >= ||y[1:]||
        assert!(is_in_dual_cone(&[2.0, 1.0, 1.0], &ConeSpec::Soc { dim: 3 })); // 2 >= sqrt(2)
        assert!(!is_in_dual_cone(&[1.0, 1.0, 1.0], &ConeSpec::Soc { dim: 3 })); // 1 < sqrt(2)

        // Zero: all of R^n
        assert!(is_in_dual_cone(&[1.0, -1.0, 100.0], &ConeSpec::Zero { dim: 3 }));
    }

    #[test]
    fn test_certificate_creation() {
        let z = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.5, 1.0, 1.5, 2.0];
        let cones = vec![
            ConeSpec::NonNeg { dim: 2 },
            ConeSpec::Soc { dim: 2 },
        ];

        let cert = DualCertificate::from_dual(&z, &b, &cones);

        assert_eq!(cert.y.len(), 4);
        assert_eq!(cert.cone_duals.len(), 2);
        assert_eq!(cert.cone_duals[0].offset, 0);
        assert_eq!(cert.cone_duals[0].dim, 2);
        assert_eq!(cert.cone_duals[1].offset, 2);
        assert_eq!(cert.cone_duals[1].dim, 2);
    }
}
