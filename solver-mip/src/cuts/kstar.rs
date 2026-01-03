//! K* certificate cut generation.
//!
//! For y ∈ K* (dual cone), generates valid inequality: (A^T y)^T x <= b^T y
//!
//! These cuts are derived from dual certificates of infeasible conic subproblems.

use solver_core::ConeSpec;

use crate::master::LinearCut;
use crate::model::MipProblem;
use crate::oracle::{CutExtractor, DualCertificate};

/// K* cut generator settings.
#[derive(Debug, Clone)]
pub struct KStarSettings {
    /// Maximum cuts to generate per oracle call.
    pub max_cuts_per_round: usize,

    /// Use disaggregated cuts (one per violated cone block).
    pub disaggregate: bool,

    /// Minimum violation for a cut to be added.
    pub min_violation: f64,

    /// Normalize cuts before adding.
    pub normalize: bool,
}

impl Default for KStarSettings {
    fn default() -> Self {
        Self {
            max_cuts_per_round: 10,
            disaggregate: true,
            min_violation: 1e-8,
            normalize: true,
        }
    }
}

/// K* certificate cut generator.
///
/// Generates valid cuts from dual certificates when the conic oracle
/// returns infeasibility.
pub struct KStarCutGenerator {
    /// Settings.
    settings: KStarSettings,

    /// Cut extractor for matrix operations.
    extractor: CutExtractor,

    /// Statistics.
    stats: KStarStats,
}

/// Statistics for K* cut generation.
#[derive(Debug, Default, Clone)]
pub struct KStarStats {
    /// Total cuts generated.
    pub cuts_generated: usize,

    /// Full cuts (from complete certificate).
    pub full_cuts: usize,

    /// Disaggregated cuts (per cone block).
    pub disaggregated_cuts: usize,

    /// Cuts rejected (invalid or too weak).
    pub cuts_rejected: usize,
}

impl KStarCutGenerator {
    /// Create a new K* cut generator.
    pub fn new(num_vars: usize, settings: KStarSettings) -> Self {
        Self {
            settings,
            extractor: CutExtractor::new(num_vars),
            stats: KStarStats::default(),
        }
    }

    /// Generate cuts from a dual certificate.
    ///
    /// # Arguments
    ///
    /// * `cert` - Dual certificate from infeasible conic subproblem
    /// * `prob` - MIP problem data
    /// * `x` - Current LP solution (for violation computation)
    ///
    /// # Returns
    ///
    /// Vector of valid cuts to add to the master problem.
    pub fn generate(
        &mut self,
        cert: &DualCertificate,
        prob: &MipProblem,
        x: &[f64],
    ) -> Vec<LinearCut> {
        let mut cuts = Vec::new();

        // Validate certificate
        if !cert.is_valid() {
            log::warn!("Invalid dual certificate (not in K*)");
            self.stats.cuts_rejected += 1;
            return cuts;
        }

        // Update violations based on current point
        let mut cert_mut = cert.clone();
        let s: Vec<f64> = compute_slack(prob, x);
        cert_mut.update_violations(&s);

        if self.settings.disaggregate {
            // Generate per-cone-block cuts
            let disagg_cuts = self.extractor.extract_disaggregated_cuts(
                &cert_mut,
                &prob.conic.A,
                &prob.conic.b,
                self.settings.max_cuts_per_round,
            );

            for cut in disagg_cuts {
                if self.is_violated(&cut, x) {
                    cuts.push(cut);
                    self.stats.disaggregated_cuts += 1;
                } else {
                    self.stats.cuts_rejected += 1;
                }
            }
        }

        // Always generate the full cut if we haven't hit the limit
        if cuts.len() < self.settings.max_cuts_per_round {
            let full_cut = self.extractor.extract_full_cut(&cert_mut, &prob.conic.A, &prob.conic.b);

            if self.is_violated(&full_cut, x) {
                cuts.push(full_cut);
                self.stats.full_cuts += 1;
            }
        }

        self.stats.cuts_generated += cuts.len();
        cuts
    }

    /// Generate a single K* cut from raw dual variables.
    ///
    /// Simpler interface when you just have the dual vector z.
    pub fn generate_simple(
        &mut self,
        z: &[f64],
        prob: &MipProblem,
    ) -> Option<LinearCut> {
        let n = prob.num_vars();
        let mut a = vec![0.0; n];

        // Compute a = A^T z
        for (col_idx, col) in prob.conic.A.outer_iterator().enumerate() {
            for (row_idx, &val) in col.iter() {
                a[col_idx] += val * z[row_idx];
            }
        }

        // Compute rhs = b^T z
        let rhs: f64 = prob.conic.b.iter().zip(z.iter()).map(|(b, z)| b * z).sum();

        let mut cut = LinearCut::new(
            a,
            rhs,
            crate::master::CutSource::KStarCertificate { cone_idx: 0 },
        );

        if self.settings.normalize {
            cut.normalize();
        }

        if cut.is_valid() {
            self.stats.cuts_generated += 1;
            self.stats.full_cuts += 1;
            Some(cut)
        } else {
            self.stats.cuts_rejected += 1;
            None
        }
    }

    /// Check if a cut is violated at the current point.
    fn is_violated(&self, cut: &LinearCut, x: &[f64]) -> bool {
        let violation = cut.violation(x);
        violation > self.settings.min_violation
    }

    /// Get generation statistics.
    pub fn stats(&self) -> &KStarStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = KStarStats::default();
    }
}

/// Compute slack vector s = b - Ax.
fn compute_slack(prob: &MipProblem, x: &[f64]) -> Vec<f64> {
    let m = prob.conic.b.len();
    let mut s = prob.conic.b.clone();

    // s = b - Ax
    for (col_idx, col) in prob.conic.A.outer_iterator().enumerate() {
        for (row_idx, &val) in col.iter() {
            s[row_idx] -= val * x[col_idx];
        }
    }

    s
}

/// Check if a dual vector is in the dual cone K*.
///
/// For self-dual cones (NonNeg, SOC, PSD), K* = K.
pub fn is_in_dual_cone(y: &[f64], cones: &[ConeSpec]) -> bool {
    let mut offset = 0;
    for cone in cones {
        let dim = cone.dim();
        if dim == 0 {
            continue;
        }

        let y_block = &y[offset..offset + dim];

        let valid = match cone {
            ConeSpec::Zero { .. } => true, // Dual is R^n
            ConeSpec::NonNeg { .. } => y_block.iter().all(|&yi| yi >= -1e-10),
            ConeSpec::Soc { .. } => {
                if y_block.is_empty() {
                    true
                } else {
                    let t = y_block[0];
                    let x_norm: f64 = y_block[1..].iter().map(|xi| xi * xi).sum::<f64>().sqrt();
                    t >= x_norm - 1e-10
                }
            }
            ConeSpec::Psd { .. } => true, // Would need eigendecomposition
            ConeSpec::Exp { .. } | ConeSpec::Pow { .. } => true, // Non-self-dual, assume valid
        };

        if !valid {
            return false;
        }

        offset += dim;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use solver_core::ConeSpec;

    #[test]
    fn test_dual_cone_check() {
        // NonNeg cone
        let cones = vec![ConeSpec::NonNeg { dim: 3 }];
        assert!(is_in_dual_cone(&[1.0, 2.0, 0.0], &cones));
        assert!(!is_in_dual_cone(&[1.0, -1.0, 0.0], &cones));

        // SOC cone
        let cones = vec![ConeSpec::Soc { dim: 3 }];
        assert!(is_in_dual_cone(&[2.0, 1.0, 1.0], &cones)); // 2 >= sqrt(2)
        assert!(!is_in_dual_cone(&[1.0, 1.0, 1.0], &cones)); // 1 < sqrt(2)

        // Zero cone (dual is all of R^n)
        let cones = vec![ConeSpec::Zero { dim: 3 }];
        assert!(is_in_dual_cone(&[100.0, -100.0, 0.0], &cones));
    }

    #[test]
    fn test_slack_computation() {
        use solver_core::{ProblemData, VarType};
        use sprs::CsMat;

        // Simple problem: Ax = [1, 1] * [x0, x1]^T
        let a = CsMat::new_csc(
            (1, 2),
            vec![0, 1, 2],
            vec![0, 0],
            vec![1.0, 1.0],
        );
        let prob = MipProblem::new(ProblemData {
            P: None,
            q: vec![1.0, 1.0],
            A: a,
            b: vec![2.0],
            cones: vec![ConeSpec::NonNeg { dim: 1 }],
            var_bounds: None,
            integrality: Some(vec![VarType::Binary, VarType::Continuous]),
        })
        .unwrap();

        let x = vec![0.5, 0.5];
        let s = compute_slack(&prob, &x);

        // s = b - Ax = 2.0 - 1.0 = 1.0
        assert!((s[0] - 1.0).abs() < 1e-10);
    }
}
