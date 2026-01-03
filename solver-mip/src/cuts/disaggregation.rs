//! Per-cone-block cut disaggregation utilities.
//!
//! Disaggregated cuts are generated separately for each cone block rather than
//! using the full certificate. This can produce tighter approximations and
//! helps identify which constraints are violated.

use solver_core::ConeSpec;

use crate::master::{CutSource, LinearCut};
use crate::model::MipProblem;

/// Information about a cone block for disaggregation.
#[derive(Debug, Clone)]
pub struct ConeBlock {
    /// Index in the cone list.
    pub cone_idx: usize,

    /// Type of cone.
    pub cone_type: ConeSpec,

    /// Starting row in the constraint matrix.
    pub row_start: usize,

    /// Dimension of this block.
    pub dim: usize,

    /// Current violation (positive = violated).
    pub violation: f64,
}

/// Analyzes the conic structure of a problem.
pub struct ConeAnalyzer {
    /// Cone blocks in the problem.
    blocks: Vec<ConeBlock>,

    /// Total constraint dimension.
    total_dim: usize,
}

impl ConeAnalyzer {
    /// Create a new analyzer from problem data.
    pub fn new(prob: &MipProblem) -> Self {
        let mut blocks = Vec::new();
        let mut offset = 0;

        for (idx, cone) in prob.conic.cones.iter().enumerate() {
            let dim = cone.dim();
            if dim > 0 {
                blocks.push(ConeBlock {
                    cone_idx: idx,
                    cone_type: cone.clone(),
                    row_start: offset,
                    dim,
                    violation: 0.0,
                });
            }
            offset += dim;
        }

        Self {
            blocks,
            total_dim: offset,
        }
    }

    /// Get all cone blocks.
    pub fn blocks(&self) -> &[ConeBlock] {
        &self.blocks
    }

    /// Get SOC blocks only.
    pub fn soc_blocks(&self) -> impl Iterator<Item = &ConeBlock> {
        self.blocks
            .iter()
            .filter(|b| matches!(b.cone_type, ConeSpec::Soc { .. }))
    }

    /// Get NonNeg blocks only.
    pub fn nonneg_blocks(&self) -> impl Iterator<Item = &ConeBlock> {
        self.blocks
            .iter()
            .filter(|b| matches!(b.cone_type, ConeSpec::NonNeg { .. }))
    }

    /// Update violations from slack vector s = b - Ax.
    pub fn update_violations(&mut self, s: &[f64]) {
        for block in &mut self.blocks {
            block.violation = compute_block_violation(block, s);
        }
    }

    /// Get the most violated blocks (sorted by violation).
    pub fn most_violated(&self, max_count: usize) -> Vec<&ConeBlock> {
        let mut sorted: Vec<&ConeBlock> = self
            .blocks
            .iter()
            .filter(|b| b.violation > 1e-8)
            .collect();

        sorted.sort_by(|a, b| {
            b.violation
                .partial_cmp(&a.violation)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        sorted.into_iter().take(max_count).collect()
    }

    /// Get total constraint dimension.
    pub fn total_dim(&self) -> usize {
        self.total_dim
    }
}

/// Compute violation for a cone block.
fn compute_block_violation(block: &ConeBlock, s: &[f64]) -> f64 {
    let s_block = &s[block.row_start..block.row_start + block.dim];

    match &block.cone_type {
        ConeSpec::Zero { .. } => {
            // Zero cone: s should be 0
            s_block.iter().map(|x| x.abs()).fold(0.0, f64::max)
        }
        ConeSpec::NonNeg { .. } => {
            // NonNeg: s >= 0, violation is max(-s_i, 0)
            s_block.iter().map(|x| (-x).max(0.0)).fold(0.0, f64::max)
        }
        ConeSpec::Soc { .. } => {
            // SOC: t >= ||x||, violation is ||x|| - t
            if s_block.is_empty() {
                return 0.0;
            }
            let t = s_block[0];
            let x_norm: f64 = s_block[1..].iter().map(|x| x * x).sum::<f64>().sqrt();
            (x_norm - t).max(0.0)
        }
        ConeSpec::Psd { .. } => {
            // PSD: would need eigenvalue check
            0.0
        }
        ConeSpec::Exp { .. } | ConeSpec::Pow { .. } => {
            // Non-elementary cones
            0.0
        }
    }
}

/// Generate a disaggregated cut for a single cone block.
///
/// For block i with dual y_i, the cut is: (A_i^T y_i)^T x <= b_i^T y_i
pub fn generate_block_cut(
    block: &ConeBlock,
    y: &[f64],
    prob: &MipProblem,
) -> Option<LinearCut> {
    let n = prob.num_vars();
    let y_block = &y[block.row_start..block.row_start + block.dim];

    // Compute a = A_block^T y_block
    let mut a = vec![0.0; n];
    for (col_idx, col) in prob.conic.A.outer_iterator().enumerate() {
        for (row_idx, &val) in col.iter() {
            if row_idx >= block.row_start && row_idx < block.row_start + block.dim {
                let local_idx = row_idx - block.row_start;
                a[col_idx] += val * y_block[local_idx];
            }
        }
    }

    // Compute rhs = b_block^T y_block
    let b_block = &prob.conic.b[block.row_start..block.row_start + block.dim];
    let rhs: f64 = b_block.iter().zip(y_block).map(|(b, y)| b * y).sum();

    let mut cut = LinearCut::new(
        a,
        rhs,
        CutSource::Disaggregated {
            cone_idx: block.cone_idx,
            block: 0,
        },
    );

    cut.normalize();

    if cut.is_valid() {
        Some(cut)
    } else {
        None
    }
}

/// Lifted disaggregation for SOC constraints.
///
/// For an SOC block (t, x) where t >= ||x||, this generates multiple
/// tangent hyperplanes to better approximate the cone.
pub struct LiftedDisaggregation {
    /// Number of tangent directions to use per SOC.
    pub num_tangents: usize,

    /// Minimum norm to generate tangent (avoid apex singularity).
    pub min_norm: f64,
}

impl Default for LiftedDisaggregation {
    fn default() -> Self {
        Self {
            num_tangents: 4,
            min_norm: 1e-8,
        }
    }
}

impl LiftedDisaggregation {
    /// Generate lifted cuts for SOC blocks.
    pub fn generate_soc_cuts(
        &self,
        prob: &MipProblem,
        s: &[f64],
        analyzer: &ConeAnalyzer,
    ) -> Vec<LinearCut> {
        let mut cuts = Vec::new();

        for block in analyzer.soc_blocks() {
            if block.violation < 1e-8 {
                continue;
            }

            let s_block = &s[block.row_start..block.row_start + block.dim];
            if s_block.len() < 2 {
                continue;
            }

            let t = s_block[0];
            let x = &s_block[1..];
            let x_norm: f64 = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();

            // Generate tangent at current point
            if x_norm > self.min_norm {
                if let Some(cut) = self.soc_tangent_cut(prob, block, t, x, x_norm) {
                    cuts.push(cut);
                }
            }

            // Generate additional tangents at perturbed directions
            if self.num_tangents > 1 && x.len() > 1 {
                for i in 0..(self.num_tangents - 1).min(x.len()) {
                    let mut x_perturbed = x.to_vec();
                    x_perturbed[i] += x_norm * 0.1;
                    let perturbed_norm: f64 =
                        x_perturbed.iter().map(|xi| xi * xi).sum::<f64>().sqrt();

                    if let Some(cut) = self.soc_tangent_cut(prob, block, t, &x_perturbed, perturbed_norm) {
                        cuts.push(cut);
                    }
                }
            }
        }

        cuts
    }

    /// Generate a single SOC tangent cut.
    fn soc_tangent_cut(
        &self,
        prob: &MipProblem,
        block: &ConeBlock,
        _t: f64,
        x: &[f64],
        x_norm: f64,
    ) -> Option<LinearCut> {
        if x_norm < self.min_norm {
            return None;
        }

        let n = prob.num_vars();
        let offset = block.row_start;
        let dim = block.dim;

        // Normalized direction
        let x_hat: Vec<f64> = x.iter().map(|xi| xi / x_norm).collect();

        // Cut: -A[t_row,:] x + sum_i (x_hat[i] * A[x_row+i,:]) x <= -b[t_row] + sum_i x_hat[i] * b[x_row+i]
        let mut a_cut = vec![0.0; n];

        for (col_idx, col) in prob.conic.A.outer_iterator().enumerate() {
            for (row_idx, &val) in col.iter() {
                if row_idx == offset {
                    // t row: negate
                    a_cut[col_idx] -= val;
                } else if row_idx > offset && row_idx < offset + dim {
                    let local_idx = row_idx - offset - 1;
                    a_cut[col_idx] += x_hat[local_idx] * val;
                }
            }
        }

        // RHS
        let mut rhs = -prob.conic.b[offset];
        for (i, &xi_hat) in x_hat.iter().enumerate() {
            rhs += xi_hat * prob.conic.b[offset + 1 + i];
        }

        let mut cut = LinearCut::new(
            a_cut,
            rhs,
            CutSource::SocTangent {
                cone_idx: block.cone_idx,
            },
        );

        cut.normalize();

        if cut.is_valid() {
            Some(cut)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solver_core::{ProblemData, VarType};
    use sprs::CsMat;

    fn simple_soc_problem() -> MipProblem {
        // min x0
        // s.t. (1, x0) in SOC means 1 >= |x0|
        let a = CsMat::new_csc(
            (2, 1),
            vec![0, 1],
            vec![1],
            vec![-1.0],
        );

        MipProblem::new(ProblemData {
            P: None,
            q: vec![1.0],
            A: a,
            b: vec![1.0, 0.0],
            cones: vec![ConeSpec::Soc { dim: 2 }],
            var_bounds: None,
            integrality: Some(vec![VarType::Binary]),
        })
        .unwrap()
    }

    #[test]
    fn test_cone_analyzer() {
        let prob = simple_soc_problem();
        let analyzer = ConeAnalyzer::new(&prob);

        assert_eq!(analyzer.blocks().len(), 1);
        assert_eq!(analyzer.total_dim(), 2);

        let soc_blocks: Vec<_> = analyzer.soc_blocks().collect();
        assert_eq!(soc_blocks.len(), 1);
        assert_eq!(soc_blocks[0].dim, 2);
    }

    #[test]
    fn test_soc_violation() {
        let prob = simple_soc_problem();
        let mut analyzer = ConeAnalyzer::new(&prob);

        // s = (1, 0.5) - feasible: 1 >= |0.5|
        analyzer.update_violations(&[1.0, 0.5]);
        assert!(analyzer.blocks()[0].violation < 1e-8);

        // s = (0.5, 1.0) - infeasible: 0.5 < |1.0|
        analyzer.update_violations(&[0.5, 1.0]);
        assert!(analyzer.blocks()[0].violation > 0.4); // violation = 1.0 - 0.5 = 0.5
    }
}
