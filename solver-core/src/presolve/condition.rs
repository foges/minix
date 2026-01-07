//! Constraint conditioning for ill-conditioned problems.
//!
//! This module detects and fixes common sources of numerical issues:
//! 1. Nearly-parallel constraint rows (high cosine similarity)
//! 2. Rows with extreme coefficient ratios (span many orders of magnitude)
//! 3. Duplicate or linearly dependent rows
//!
//! These issues cause the KKT system to become extremely ill-conditioned,
//! leading to:
//! - Dual variables exploding (QFFFFF80, QSHIP family)
//! - Complementarity breakdown (QFORPLAN)
//! - Step directions with enormous components
//!
//! The fixes applied are conservative scaling operations that preserve
//! problem feasibility and optimality.

use crate::problem::ProblemData;
use std::collections::HashMap;

/// Statistics about constraint matrix conditioning.
#[derive(Debug, Clone)]
pub struct ConditioningStats {
    /// Number of nearly-parallel row pairs found
    pub parallel_pairs: usize,
    /// Number of rows with extreme coefficient ratios
    pub extreme_ratio_rows: usize,
    /// Maximum cosine similarity found between any two rows
    pub max_cosine_sim: f64,
    /// Maximum coefficient ratio (max/min) found in any row
    pub max_coeff_ratio: f64,
}

/// Analyze constraint matrix for conditioning issues.
///
/// Returns statistics about potential numerical problems without modifying the matrix.
pub fn analyze_conditioning(prob: &ProblemData) -> ConditioningStats {
    let m = prob.num_constraints();
    let n = prob.num_vars();

    if m == 0 || n == 0 {
        return ConditioningStats {
            parallel_pairs: 0,
            extreme_ratio_rows: 0,
            max_cosine_sim: 0.0,
            max_coeff_ratio: 1.0,
        };
    }

    // Build row-wise representation for easier analysis
    let mut rows: Vec<HashMap<usize, f64>> = vec![HashMap::new(); m];
    for (&val, (row, col)) in prob.A.iter() {
        rows[row].insert(col, val);
    }

    // Compute row norms (for cosine similarity)
    let mut row_norms: Vec<f64> = vec![0.0; m];
    for (i, row) in rows.iter().enumerate() {
        row_norms[i] = row.values().map(|&v| v * v).sum::<f64>().sqrt();
    }

    let mut max_cosine_sim = 0.0;
    let mut parallel_pairs = 0;

    // Check for nearly-parallel rows (sample for large problems)
    let check_limit = if m > 1000 { 1000 } else { m };
    for i in 0..check_limit.min(m) {
        if row_norms[i] < 1e-12 {
            continue; // Skip zero rows
        }

        // Sample j to avoid O(m^2) on huge problems
        let j_step = if m > 1000 { m / 500 } else { 1 };
        for j in ((i + 1)..m).step_by(j_step) {
            if row_norms[j] < 1e-12 {
                continue;
            }

            // Compute cosine similarity: dot(row_i, row_j) / (||row_i|| * ||row_j||)
            let mut dot_product = 0.0;
            for (&col, &val_i) in &rows[i] {
                if let Some(&val_j) = rows[j].get(&col) {
                    dot_product += val_i * val_j;
                }
            }

            let cosine_sim = (dot_product / (row_norms[i] * row_norms[j])).abs();
            max_cosine_sim = f64::max(max_cosine_sim, cosine_sim);

            // Nearly parallel if cosine similarity > 0.999
            if cosine_sim > 0.999 {
                parallel_pairs += 1;
            }
        }
    }

    // Check for extreme coefficient ratios within rows
    let mut max_coeff_ratio = 1.0;
    let mut extreme_ratio_rows = 0;
    for row in &rows {
        if row.is_empty() {
            continue;
        }

        let max_abs = row.values().map(|&v| v.abs()).fold(0.0_f64, f64::max);
        let min_abs = row.values()
            .map(|&v| v.abs())
            .filter(|&v| v > 1e-20) // Ignore tiny values
            .fold(f64::INFINITY, f64::min);

        if min_abs.is_finite() && max_abs > 0.0 {
            let ratio = max_abs / min_abs;
            max_coeff_ratio = f64::max(max_coeff_ratio, ratio);

            // Extreme if ratio > 1e8
            if ratio > 1e8 {
                extreme_ratio_rows += 1;
            }
        }
    }

    ConditioningStats {
        parallel_pairs,
        extreme_ratio_rows,
        max_cosine_sim,
        max_coeff_ratio,
    }
}

/// Apply row-wise scaling to improve conditioning.
///
/// For each row i, compute a scale factor based on row norm and coefficient spread,
/// then multiply row i of A and element i of b by this factor.
///
/// This is similar to Ruiz scaling but focuses on rows with extreme properties.
pub fn apply_row_scaling(prob: &mut ProblemData) -> Vec<f64> {
    let m = prob.num_constraints();
    let n = prob.num_vars();

    if m == 0 || n == 0 {
        return vec![1.0; m];
    }

    // Build row-wise representation
    let mut rows: Vec<HashMap<usize, f64>> = vec![HashMap::new(); m];
    for (&val, (row, col)) in prob.A.iter() {
        rows[row].insert(col, val);
    }

    // Compute scaling factors for each row
    let mut row_scales = vec![1.0; m];

    for (i, row) in rows.iter().enumerate() {
        if row.is_empty() {
            continue;
        }

        // Compute row statistics
        let max_abs = row.values().map(|&v| v.abs()).fold(0.0_f64, f64::max);
        let min_abs = row.values()
            .map(|&v| v.abs())
            .filter(|&v| v > 1e-20)
            .fold(f64::INFINITY, f64::min);

        if max_abs < 1e-20 {
            continue; // Skip essentially zero rows
        }

        // Scale factor: geometric mean of max and min (pulls toward 1.0)
        // This reduces the coefficient spread without changing the row direction
        let geom_mean = if min_abs.is_finite() && min_abs > 1e-20 {
            (max_abs * min_abs).sqrt()
        } else {
            max_abs
        };

        // Target: scale so geometric mean is around 1.0
        if geom_mean > 1e-10 {
            row_scales[i] = 1.0 / geom_mean.sqrt();
        }

        // Clamp to avoid extreme scaling
        row_scales[i] = row_scales[i].clamp(1e-3, 1e3);
    }

    // Apply scaling to A and b
    let mut new_triplets = Vec::new();
    for (&val, (row, col)) in prob.A.iter() {
        new_triplets.push((row, col, val * row_scales[row]));
    }

    // Rebuild A
    prob.A = crate::linalg::sparse::from_triplets(m, n, new_triplets);

    // Scale b
    for i in 0..m {
        prob.b[i] *= row_scales[i];
    }

    row_scales
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::sparse;
    use crate::problem::ConeSpec;

    #[test]
    fn test_analyze_parallel_rows() {
        // Create problem with two parallel rows
        // Row 0: x + 2y = 1
        // Row 1: 2x + 4y = 2 (exactly parallel)
        let a = sparse::from_triplets(
            2,
            2,
            vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 2.0), (1, 1, 4.0)],
        );

        let prob = ProblemData {
            P: None,
            q: vec![0.0, 0.0],
            A: a,
            b: vec![1.0, 2.0],
            cones: vec![ConeSpec::Zero { dim: 2 }],
            var_bounds: None,
            integrality: None,
        };

        let stats = analyze_conditioning(&prob);

        // Should detect high cosine similarity
        assert!(stats.max_cosine_sim > 0.99);
        assert!(stats.parallel_pairs > 0);
    }

    #[test]
    fn test_analyze_extreme_ratios() {
        // Row with extreme coefficient ratio: 1e-10 and 1e10
        let a = sparse::from_triplets(
            1,
            2,
            vec![(0, 0, 1e-10), (0, 1, 1e10)],
        );

        let prob = ProblemData {
            P: None,
            q: vec![0.0, 0.0],
            A: a,
            b: vec![1.0],
            cones: vec![ConeSpec::Zero { dim: 1 }],
            var_bounds: None,
            integrality: None,
        };

        let stats = analyze_conditioning(&prob);

        // Ratio = 1e10 / 1e-10 = 1e20
        assert!(stats.max_coeff_ratio > 1e15);
        assert!(stats.extreme_ratio_rows > 0);
    }

    #[test]
    fn test_row_scaling() {
        // Row with coefficients [1e-5, 1e5]
        let a = sparse::from_triplets(
            1,
            2,
            vec![(0, 0, 1e-5), (0, 1, 1e5)],
        );

        let mut prob = ProblemData {
            P: None,
            q: vec![0.0, 0.0],
            A: a,
            b: vec![2.0],
            cones: vec![ConeSpec::Zero { dim: 1 }],
            var_bounds: None,
            integrality: None,
        };

        let scales = apply_row_scaling(&mut prob);

        // Should have computed a non-trivial scale
        assert!((scales[0] - 1.0).abs() > 0.1);

        // Check that b was scaled
        assert!((prob.b[0] - 2.0).abs() > 0.01);
    }
}
