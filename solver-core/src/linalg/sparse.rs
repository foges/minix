//! Sparse matrix types and operations.
//!
//! This module provides wrappers and utilities for sparse matrices in CSC
//! (Compressed Sparse Column) format, which is the standard format for
//! sparse direct solvers.

use sprs::{CsMat, TriMat};

/// Sparse matrix in CSC format (general, not necessarily symmetric).
pub type SparseCsc = CsMat<f64>;

/// Sparse symmetric matrix in CSC format (upper triangle only).
pub type SparseSymmetricCsc = CsMat<f64>;

/// Triplet format sparse matrix builder.
pub type SparseTriMat = TriMat<f64>;

/// Build a sparse CSC matrix from triplets (row, col, value).
///
/// # Arguments
///
/// * `nrows` - Number of rows
/// * `ncols` - Number of columns
/// * `triplets` - Iterator of (row, col, value) tuples
pub fn from_triplets<I>(nrows: usize, ncols: usize, triplets: I) -> SparseCsc
where
    I: IntoIterator<Item = (usize, usize, f64)>,
{
    let mut tri = TriMat::new((nrows, ncols));
    for (i, j, v) in triplets {
        tri.add_triplet(i, j, v);
    }
    tri.to_csc()
}

/// Build a symmetric sparse CSC matrix from upper triangle triplets.
///
/// Only stores the upper triangle. Assumes triplets satisfy j >= i.
pub fn from_triplets_symmetric<I>(n: usize, triplets: I) -> SparseSymmetricCsc
where
    I: IntoIterator<Item = (usize, usize, f64)>,
{
    let mut tri = TriMat::new((n, n));
    for (i, j, v) in triplets {
        assert!(j >= i, "Symmetric matrix must only contain upper triangle");
        tri.add_triplet(i, j, v);
    }
    tri.to_csc()
}

/// Create a diagonal matrix in CSC format.
pub fn diagonal(diag: &[f64]) -> SparseCsc {
    let n = diag.len();
    let triplets = diag.iter().enumerate().map(|(i, &v)| (i, i, v));
    from_triplets(n, n, triplets)
}

/// Create an identity matrix in CSC format.
pub fn identity(n: usize) -> SparseCsc {
    diagonal(&vec![1.0; n])
}

/// Sparse matrix-vector product: y = alpha * A * x + beta * y
pub fn spmv(a: &SparseCsc, x: &[f64], y: &mut [f64], alpha: f64, beta: f64) {
    assert_eq!(a.cols(), x.len());
    assert_eq!(a.rows(), y.len());

    // Scale y by beta
    if beta == 0.0 {
        y.fill(0.0);
    } else if beta != 1.0 {
        for yi in y.iter_mut() {
            *yi *= beta;
        }
    }

    // Add alpha * A * x
    if alpha != 0.0 {
        for (val, (row, col)) in a.iter() {
            y[row] += alpha * (*val) * x[col];
        }
    }
}

/// Transpose-vector product: y = alpha * A^T * x + beta * y
pub fn spmv_transpose(a: &SparseCsc, x: &[f64], y: &mut [f64], alpha: f64, beta: f64) {
    assert_eq!(a.rows(), x.len());
    assert_eq!(a.cols(), y.len());

    // For CSC, A^T is equivalent to treating columns as rows
    // Scale y by beta
    if beta == 0.0 {
        y.fill(0.0);
    } else if beta != 1.0 {
        for yi in y.iter_mut() {
            *yi *= beta;
        }
    }

    // Add alpha * A^T * x
    if alpha != 0.0 {
        for col_idx in 0..a.cols() {
            let col = a.outer_view(col_idx).unwrap();
            for (row_idx, &val) in col.iter() {
                y[col_idx] += alpha * val * x[row_idx];
            }
        }
    }
}

/// Stack two sparse matrices vertically: [A; B]
pub fn vstack(a: &SparseCsc, b: &SparseCsc) -> SparseCsc {
    assert_eq!(a.cols(), b.cols(), "Matrices must have same number of columns");

    let nrows = a.rows() + b.rows();
    let ncols = a.cols();

    let mut tri = TriMat::new((nrows, ncols));

    // Add entries from A
    for (val, (row, col)) in a.iter() {
        tri.add_triplet(row, col, *val);
    }

    // Add entries from B (offset rows by a.rows())
    for (val, (row, col)) in b.iter() {
        tri.add_triplet(row + a.rows(), col, *val);
    }

    tri.to_csc()
}

/// Stack two sparse matrices horizontally: [A, B]
pub fn hstack(a: &SparseCsc, b: &SparseCsc) -> SparseCsc {
    assert_eq!(a.rows(), b.rows(), "Matrices must have same number of rows");

    let nrows = a.rows();
    let ncols = a.cols() + b.cols();

    let mut tri = TriMat::new((nrows, ncols));

    // Add entries from A
    for (val, (row, col)) in a.iter() {
        tri.add_triplet(row, col, *val);
    }

    // Add entries from B (offset cols by a.cols())
    for (val, (row, col)) in b.iter() {
        tri.add_triplet(row, col + a.cols(), *val);
    }

    tri.to_csc()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_triplets() {
        let triplets = vec![
            (0, 0, 1.0),
            (1, 1, 2.0),
            (0, 1, 3.0),
        ];
        let mat = from_triplets(2, 2, triplets);

        assert_eq!(mat.rows(), 2);
        assert_eq!(mat.cols(), 2);
        assert_eq!(mat.nnz(), 3);
    }

    #[test]
    fn test_diagonal() {
        let diag = vec![1.0, 2.0, 3.0];
        let mat = diagonal(&diag);

        assert_eq!(mat.rows(), 3);
        assert_eq!(mat.cols(), 3);
        assert_eq!(mat.nnz(), 3);

        // Check diagonal values
        for i in 0..3 {
            let col = mat.outer_view(i).unwrap();
            let val = col.iter().next().unwrap();
            assert_eq!(*val.1, diag[i]);
        }
    }

    #[test]
    fn test_identity() {
        let mat = identity(5);

        assert_eq!(mat.rows(), 5);
        assert_eq!(mat.cols(), 5);
        assert_eq!(mat.nnz(), 5);
    }

    #[test]
    fn test_spmv() {
        // 2x2 matrix: [[1, 2], [3, 4]]
        let triplets = vec![
            (0, 0, 1.0), (0, 1, 2.0),
            (1, 0, 3.0), (1, 1, 4.0),
        ];
        let mat = from_triplets(2, 2, triplets);

        let x = vec![1.0, 2.0];
        let mut y = vec![0.0; 2];

        spmv(&mat, &x, &mut y, 1.0, 0.0);

        // y = [[1, 2], [3, 4]] * [1, 2] = [5, 11]
        assert!((y[0] - 5.0).abs() < 1e-10);
        assert!((y[1] - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_vstack() {
        // A = [[1, 2]]  (1x2)
        // B = [[3, 4]]  (1x2)
        // [A; B] = [[1, 2], [3, 4]]  (2x2)

        let a = from_triplets(1, 2, vec![(0, 0, 1.0), (0, 1, 2.0)]);
        let b = from_triplets(1, 2, vec![(0, 0, 3.0), (0, 1, 4.0)]);

        let stacked = vstack(&a, &b);

        assert_eq!(stacked.rows(), 2);
        assert_eq!(stacked.cols(), 2);
        assert_eq!(stacked.nnz(), 4);
    }
}
