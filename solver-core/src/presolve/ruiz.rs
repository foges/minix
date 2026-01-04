//! Ruiz equilibration for matrix conditioning.
//!
//! Ruiz equilibration iteratively scales the rows and columns of the constraint
//! matrix A to improve conditioning. This helps the IPM converge faster and
//! more reliably by balancing the magnitudes of matrix entries.
//!
//! The algorithm:
//! 1. For each iteration:
//!    - Compute row scaling: d_r[i] = 1/sqrt(||A[i,:]||_∞)
//!    - Compute column scaling: d_c[j] = 1/sqrt(||A[:,j]||_∞)
//!    - Apply: A ← diag(d_r) * A * diag(d_c)
//!    - Accumulate: R *= d_r, C *= d_c
//! 2. Scale P, q, b accordingly
//! 3. After solving, unscale x, s, z

use crate::linalg::sparse::{SparseCsc, SparseSymmetricCsc};
use crate::problem::ConeSpec;
use sprs::TriMat;

/// Result of Ruiz equilibration containing scaled problem data and scaling factors.
#[derive(Clone)]
pub struct RuizScaling {
    /// Row scaling factors (length m), applied to A and b
    pub row_scale: Vec<f64>,
    /// Column scaling factors (length n), applied to A, P, and q
    pub col_scale: Vec<f64>,
    /// Cost scaling factor for numerical stability
    pub cost_scale: f64,
}

impl RuizScaling {
    /// Create identity scaling (no scaling applied).
    pub fn identity(n: usize, m: usize) -> Self {
        Self {
            row_scale: vec![1.0; m],
            col_scale: vec![1.0; n],
            cost_scale: 1.0,
        }
    }

    /// Unscale the primal solution x.
    /// x_original = diag(col_scale) * x_scaled
    pub fn unscale_x(&self, x_scaled: &[f64]) -> Vec<f64> {
        x_scaled.iter()
            .zip(self.col_scale.iter())
            .map(|(&xi, &ci)| ci * xi)
            .collect()
    }

    /// Unscale the slack variables s.
    /// Given A_scaled = R * A * C and b_scaled = R * b,
    /// the scaled slack is s_scaled = R * s, so s_original = s_scaled / R
    pub fn unscale_s(&self, s_scaled: &[f64]) -> Vec<f64> {
        s_scaled.iter()
            .zip(self.row_scale.iter())
            .map(|(&si, &ri)| si / ri)
            .collect()
    }

    /// Unscale the dual variables z.
    /// Given the dual equation scales as A^T z → C * A^T * R * z_scaled,
    /// we have z_original = cost_scale * R * z_scaled
    pub fn unscale_z(&self, z_scaled: &[f64]) -> Vec<f64> {
        z_scaled.iter()
            .zip(self.row_scale.iter())
            .map(|(&zi, &ri)| self.cost_scale * ri * zi)
            .collect()
    }

    /// Unscale the objective value.
    /// obj_original = obj_scaled / cost_scale
    pub fn unscale_obj(&self, obj_scaled: f64) -> f64 {
        obj_scaled / self.cost_scale
    }
}

/// Apply Ruiz equilibration to the problem data.
///
/// # Arguments
///
/// * `a` - Constraint matrix (m × n)
/// * `p` - Optional quadratic cost matrix (n × n)
/// * `q` - Linear cost vector (length n)
/// * `b` - Constraint RHS (length m)
/// * `iters` - Number of Ruiz iterations
/// * `cones` - Cone partition (used for block-aware row scaling)
///
/// # Returns
///
/// Tuple of (scaled_A, scaled_P, scaled_q, scaled_b, scaling)
#[allow(non_snake_case)]
pub fn equilibrate(
    A: &SparseCsc,
    P: Option<&SparseSymmetricCsc>,
    q: &[f64],
    b: &[f64],
    iters: usize,
    cones: &[ConeSpec],
) -> (SparseCsc, Option<SparseSymmetricCsc>, Vec<f64>, Vec<f64>, RuizScaling) {
    let m = A.rows();
    let n = A.cols();

    if iters == 0 {
        return (
            A.clone(),
            P.cloned(),
            q.to_vec(),
            b.to_vec(),
            RuizScaling::identity(n, m),
        );
    }

    // Accumulated scaling factors
    let mut row_scale = vec![1.0; m];
    let mut col_scale = vec![1.0; n];

    // Work with mutable copies
    let mut A_scaled = A.clone();
    let mut P_scaled = P.cloned();

    for _ in 0..iters {
        // Compute row infinity norms of A
        let mut row_norms = vec![0.0_f64; m];
        for (&val, (row, _col)) in A_scaled.iter() {
            row_norms[row] = row_norms[row].max(val.abs());
        }

        // Compute column infinity norms of A (and P if present)
        let mut col_norms = vec![0.0_f64; n];
        for (&val, (_row, col)) in A_scaled.iter() {
            col_norms[col] = col_norms[col].max(val.abs());
        }
        if let Some(ref p) = P_scaled {
            for (&val, (row, col)) in p.iter() {
                // P is symmetric, stored as upper triangle
                col_norms[col] = col_norms[col].max(val.abs());
                if row != col {
                    col_norms[row] = col_norms[row].max(val.abs());
                }
            }
        }

        // Compute scaling factors: d = 1/sqrt(norm), avoiding division by zero
        let mut d_row: Vec<f64> = row_norms.iter()
            .map(|&norm| if norm > 1e-12 { 1.0 / norm.sqrt() } else { 1.0 })
            .collect();
        let d_col: Vec<f64> = col_norms.iter()
            .map(|&norm| if norm > 1e-12 { 1.0 / norm.sqrt() } else { 1.0 })
            .collect();

        // For non-separable cones (SOC/PSD/EXP/POW), enforce uniform row scaling
        // within each cone block to preserve cone geometry.
        if !cones.is_empty() {
            let mut offset = 0;
            for cone in cones {
                let dim = match cone {
                    ConeSpec::Zero { dim } => *dim,
                    ConeSpec::NonNeg { dim } => *dim,
                    ConeSpec::Soc { dim } => *dim,
                    ConeSpec::Psd { n } => n * (n + 1) / 2,
                    ConeSpec::Exp { count } => 3 * count,
                    ConeSpec::Pow { cones } => 3 * cones.len(),
                };

                if dim == 0 {
                    continue;
                }

                if offset + dim > m {
                    break;
                }

                let uniform_block = matches!(
                    cone,
                    ConeSpec::Soc { .. } | ConeSpec::Psd { .. } | ConeSpec::Exp { .. } | ConeSpec::Pow { .. }
                );

                if uniform_block {
                    let mut block_norm = 0.0_f64;
                    for i in offset..offset + dim {
                        block_norm = block_norm.max(row_norms[i]);
                    }
                    let block_scale = if block_norm > 1e-12 { 1.0 / block_norm.sqrt() } else { 1.0 };
                    for i in offset..offset + dim {
                        d_row[i] = block_scale;
                    }
                }

                offset += dim;
            }
        }

        // Apply scaling to A: A = diag(d_row) * A * diag(d_col)
        A_scaled = scale_matrix(&A_scaled, &d_row, &d_col);

        // Apply scaling to P: P = diag(d_col) * P * diag(d_col)
        if let Some(ref p) = P_scaled {
            P_scaled = Some(scale_symmetric_matrix(p, &d_col));
        }

        // Accumulate scaling
        for i in 0..m {
            row_scale[i] *= d_row[i];
        }
        for j in 0..n {
            col_scale[j] *= d_col[j];
        }
    }

    // Scale q: q_scaled = diag(col_scale) * q
    let q_scaled: Vec<f64> = q.iter()
        .zip(col_scale.iter())
        .map(|(&qi, &ci)| ci * qi)
        .collect();

    // Scale b: b_scaled = diag(row_scale) * b
    let b_scaled: Vec<f64> = b.iter()
        .zip(row_scale.iter())
        .map(|(&bi, &ri)| ri * bi)
        .collect();

    // Compute cost scaling for numerical stability
    // Scale so that ||q||_∞ and ||P||_∞ are O(1)
    let q_norm = q_scaled.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let p_norm = if let Some(ref p) = P_scaled {
        p.iter().map(|(v, _)| v.abs()).fold(0.0_f64, f64::max)
    } else {
        0.0
    };
    let max_cost_norm = q_norm.max(p_norm);
    let cost_scale = if max_cost_norm > 1e-12 { max_cost_norm } else { 1.0 };

    // Apply cost scaling
    let q_scaled: Vec<f64> = q_scaled.iter().map(|&qi| qi / cost_scale).collect();
    let P_scaled = P_scaled.map(|p| scale_matrix_scalar(&p, 1.0 / cost_scale));

    let scaling = RuizScaling {
        row_scale,
        col_scale,
        cost_scale,
    };

    (A_scaled, P_scaled, q_scaled, b_scaled, scaling)
}

/// Scale a sparse matrix: result = diag(row_scale) * A * diag(col_scale)
fn scale_matrix(mat: &SparseCsc, row_scale: &[f64], col_scale: &[f64]) -> SparseCsc {
    let m = mat.rows();
    let n = mat.cols();
    let mut tri = TriMat::new((m, n));

    for (&val, (row, col)) in mat.iter() {
        tri.add_triplet(row, col, val * row_scale[row] * col_scale[col]);
    }

    tri.to_csc()
}

/// Scale a symmetric matrix: result = diag(scale) * P * diag(scale)
fn scale_symmetric_matrix(mat: &SparseSymmetricCsc, scale: &[f64]) -> SparseSymmetricCsc {
    let n = mat.rows();
    let mut tri = TriMat::new((n, n));

    for (&val, (row, col)) in mat.iter() {
        tri.add_triplet(row, col, val * scale[row] * scale[col]);
    }

    tri.to_csc()
}

/// Scale a matrix by a scalar
fn scale_matrix_scalar(mat: &SparseCsc, scalar: f64) -> SparseCsc {
    let m = mat.rows();
    let n = mat.cols();
    let mut tri = TriMat::new((m, n));

    for (&val, (row, col)) in mat.iter() {
        tri.add_triplet(row, col, val * scalar);
    }

    tri.to_csc()
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::linalg::sparse;
    use crate::problem::ConeSpec;

    #[test]
    fn test_identity_scaling() {
        let scaling = RuizScaling::identity(3, 2);

        let x = vec![1.0, 2.0, 3.0];
        let x_unscaled = scaling.unscale_x(&x);
        assert_eq!(x, x_unscaled);

        let s = vec![4.0, 5.0];
        let s_unscaled = scaling.unscale_s(&s);
        assert_eq!(s, s_unscaled);

        let z = vec![6.0, 7.0];
        let z_unscaled = scaling.unscale_z(&z);
        assert_eq!(z, z_unscaled);
    }

    #[test]
    fn test_equilibrate_no_iters() {
        let A = sparse::from_triplets(2, 3, vec![
            (0, 0, 1.0), (0, 1, 2.0),
            (1, 1, 3.0), (1, 2, 4.0),
        ]);
        let q = vec![1.0, 2.0, 3.0];
        let b = vec![5.0, 6.0];

        let (A_scaled, _, q_scaled, b_scaled, scaling) =
            equilibrate(&A, None, &q, &b, 0, &[ConeSpec::NonNeg { dim: 2 }]);

        // With 0 iterations, should be identity
        assert_eq!(A_scaled.nnz(), A.nnz());
        assert_eq!(q_scaled, q);
        assert_eq!(b_scaled, b);
        assert_eq!(scaling.row_scale, vec![1.0; 2]);
        assert_eq!(scaling.col_scale, vec![1.0; 3]);
    }

    #[test]
    fn test_equilibrate_balances_norms() {
        // Matrix with very different row/column magnitudes
        let A = sparse::from_triplets(2, 2, vec![
            (0, 0, 1000.0), (0, 1, 1.0),
            (1, 0, 1.0), (1, 1, 0.001),
        ]);
        let q = vec![1.0, 1.0];
        let b = vec![1.0, 1.0];

        let (A_scaled, _, _, _, _) = equilibrate(&A, None, &q, &b, 10, &[ConeSpec::NonNeg { dim: 2 }]);

        // After equilibration, row and column norms should be more balanced
        let mut row_norms = vec![0.0_f64; 2];
        let mut col_norms = vec![0.0_f64; 2];
        for (&val, (row, col)) in A_scaled.iter() {
            row_norms[row] = row_norms[row].max(val.abs());
            col_norms[col] = col_norms[col].max(val.abs());
        }

        // Check that max/min ratio is much smaller than original (1000000:1)
        let row_ratio = row_norms[0].max(row_norms[1]) / row_norms[0].min(row_norms[1]);
        let col_ratio = col_norms[0].max(col_norms[1]) / col_norms[0].min(col_norms[1]);

        assert!(row_ratio < 100.0, "Row ratio should be balanced: {}", row_ratio);
        assert!(col_ratio < 100.0, "Col ratio should be balanced: {}", col_ratio);
    }

    #[test]
    fn test_unscale_roundtrip() {
        let A = sparse::from_triplets(2, 3, vec![
            (0, 0, 100.0), (0, 1, 0.01),
            (1, 1, 10.0), (1, 2, 0.1),
        ]);
        let q = vec![1.0, 2.0, 3.0];
        let b = vec![5.0, 6.0];

        let (_, _, _, _, scaling) = equilibrate(&A, None, &q, &b, 5, &[ConeSpec::NonNeg { dim: 2 }]);

        // Test x roundtrip: x_scaled = x / C, unscale gives x = C * x_scaled
        let x_orig = vec![1.0, 2.0, 3.0];
        let x_scaled: Vec<f64> = x_orig.iter()
            .zip(scaling.col_scale.iter())
            .map(|(&xi, &ci)| xi / ci)
            .collect();
        let x_unscaled = scaling.unscale_x(&x_scaled);
        for i in 0..3 {
            assert!((x_orig[i] - x_unscaled[i]).abs() < 1e-10,
                "x roundtrip failed at {}: {} vs {}", i, x_orig[i], x_unscaled[i]);
        }

        // Test s roundtrip: s_scaled = R * s, unscale gives s = s_scaled / R
        let s_orig = vec![1.0, 2.0];
        let s_scaled: Vec<f64> = s_orig.iter()
            .zip(scaling.row_scale.iter())
            .map(|(&si, &ri)| ri * si)
            .collect();
        let s_unscaled = scaling.unscale_s(&s_scaled);
        for i in 0..2 {
            assert!((s_orig[i] - s_unscaled[i]).abs() < 1e-10,
                "s roundtrip failed at {}: {} vs {}", i, s_orig[i], s_unscaled[i]);
        }

        // Test z roundtrip: z_scaled = z / (cost_scale * R), unscale gives z = cost_scale * R * z_scaled
        let z_orig = vec![1.0, 2.0];
        let z_scaled: Vec<f64> = z_orig.iter()
            .zip(scaling.row_scale.iter())
            .map(|(&zi, &ri)| zi / (scaling.cost_scale * ri))
            .collect();
        let z_unscaled = scaling.unscale_z(&z_scaled);
        for i in 0..2 {
            assert!((z_orig[i] - z_unscaled[i]).abs() < 1e-10,
                "z roundtrip failed at {}: {} vs {}", i, z_orig[i], z_unscaled[i]);
        }
    }

    #[test]
    fn test_equilibrate_with_p() {
        let A = sparse::from_triplets(2, 2, vec![
            (0, 0, 1.0), (0, 1, 2.0),
            (1, 0, 3.0), (1, 1, 4.0),
        ]);
        let P = sparse::from_triplets(2, 2, vec![
            (0, 0, 100.0),
            (0, 1, 10.0),
            (1, 1, 1.0),
        ]);
        let q = vec![1.0, 2.0];
        let b = vec![5.0, 6.0];

        let (A_scaled, P_scaled, q_scaled, b_scaled, scaling) =
            equilibrate(&A, Some(&P), &q, &b, 5, &[ConeSpec::NonNeg { dim: 2 }]);

        // Verify dimensions are preserved
        assert_eq!(A_scaled.rows(), 2);
        assert_eq!(A_scaled.cols(), 2);
        assert!(P_scaled.is_some());
        let P_scaled = P_scaled.unwrap();
        assert_eq!(P_scaled.rows(), 2);
        assert_eq!(P_scaled.cols(), 2);
        assert_eq!(q_scaled.len(), 2);
        assert_eq!(b_scaled.len(), 2);

        // Verify scaling factors are positive
        for &r in &scaling.row_scale {
            assert!(r > 0.0);
        }
        for &c in &scaling.col_scale {
            assert!(c > 0.0);
        }
        assert!(scaling.cost_scale > 0.0);
    }
}
