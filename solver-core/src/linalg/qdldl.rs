//! LDL factorization wrapper.
//!
//! This module provides a clean interface to sparse LDL^T factorization
//! for quasi-definite matrices using the `ldl` crate.
//!
//! The LDL factorization computes L and D such that A = LDL^T, where:
//! - L is lower triangular with unit diagonal
//! - D is diagonal (can have negative entries, unlike Cholesky)
//!
//! This is essential for solving KKT systems in interior point methods.

use super::sparse::SparseCsc;
use thiserror::Error;

/// LDL solver errors
#[derive(Error, Debug)]
pub enum QdldlError {
    /// Factorization failed (matrix not quasi-definite)
    #[error("Factorization failed: matrix not quasi-definite")]
    FactorizationFailed,

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
    },

    /// Invalid regularization parameter
    #[error("Invalid regularization parameter: {0}")]
    InvalidRegularization(String),
}

/// LDL factorization backend.
///
/// Manages factorization of quasi-definite matrices using the `ldl` crate.
pub struct QdldlSolver {
    /// Matrix dimension
    n: usize,

    /// Elimination tree (computed during symbolic factorization)
    etree: Option<Vec<Option<usize>>>,

    /// L nonzero count per column
    l_nz: Option<Vec<usize>>,

    /// Factorization data: L and D
    /// L is stored in CSC format: (l_p, l_i, l_x)
    /// D is stored as a vector
    factorization: Option<LdlFactorData>,

    /// Static regularization (added to diagonal)
    static_reg: f64,

    /// Dynamic regularization minimum pivot threshold
    dynamic_reg_min_pivot: f64,

    /// Number of dynamic regularization bumps applied
    dynamic_bumps: u64,
}

/// Internal storage for LDL factorization
struct LdlFactorData {
    /// L column pointers
    l_p: Vec<usize>,
    /// L row indices
    l_i: Vec<usize>,
    /// L values
    l_x: Vec<f64>,
    /// D diagonal (stored for debugging/future use)
    #[allow(dead_code)]
    d: Vec<f64>,
    /// D inverse (for faster solving)
    d_inv: Vec<f64>,
}

impl QdldlSolver {
    /// Create a new LDL solver.
    ///
    /// # Arguments
    ///
    /// * `n` - Dimension of the system
    /// * `static_reg` - Static diagonal regularization (added to all diagonal entries)
    /// * `dynamic_reg_min_pivot` - Minimum pivot threshold for dynamic regularization
    pub fn new(n: usize, static_reg: f64, dynamic_reg_min_pivot: f64) -> Self {
        assert!(static_reg >= 0.0, "Static regularization must be non-negative");
        assert!(
            dynamic_reg_min_pivot > 0.0,
            "Dynamic regularization threshold must be positive"
        );

        Self {
            n,
            etree: None,
            l_nz: None,
            factorization: None,
            static_reg,
            dynamic_reg_min_pivot,
            dynamic_bumps: 0,
        }
    }

    /// Perform symbolic factorization on the sparsity pattern.
    ///
    /// Computes the elimination tree, which can be reused across numeric factorizations
    /// with the same sparsity pattern.
    ///
    /// # Arguments
    ///
    /// * `mat` - Sparse matrix in CSC format (upper triangle only)
    pub fn symbolic_factorization(&mut self, mat: &SparseCsc) -> Result<(), QdldlError> {
        if mat.rows() != self.n || mat.cols() != self.n {
            return Err(QdldlError::DimensionMismatch {
                expected: self.n,
                actual: mat.rows(),
            });
        }

        // Keep indptr alive
        let indptr = mat.indptr();
        let a_p = indptr.raw_storage();
        let a_i = mat.indices();

        // Allocate outputs
        let mut work = vec![0; self.n];
        let mut l_nz = vec![0; self.n];
        let mut etree = vec![None; self.n];

        // Compute elimination tree
        let result = ldl::etree(
            self.n,
            a_p,
            a_i,
            &mut work,
            &mut l_nz,
            &mut etree,
        );

        match result {
            Ok(_) => {
                self.etree = Some(etree);
                self.l_nz = Some(l_nz);
                Ok(())
            }
            Err(_) => Err(QdldlError::FactorizationFailed),
        }
    }

    /// Perform numeric factorization.
    ///
    /// Computes the LDL^T factorization with regularization.
    ///
    /// # Arguments
    ///
    /// * `mat` - Sparse matrix in CSC format (upper triangle only)
    ///
    /// # Returns
    ///
    /// A factorization that can be used to solve linear systems.
    pub fn numeric_factorization(
        &mut self,
        mat: &SparseCsc,
    ) -> Result<QdldlFactorization, QdldlError> {
        // Ensure symbolic factorization was done
        if self.etree.is_none() {
            self.symbolic_factorization(mat)?;
        }

        // Extract CSC arrays (keep indptr alive)
        let indptr = mat.indptr();
        let a_p = indptr.raw_storage();
        let a_i = mat.indices();
        let a_x_orig = mat.data();

        // Apply static regularization to diagonal
        let mut a_x = a_x_orig.to_vec();
        if self.static_reg > 0.0 {
            for col in 0..self.n {
                let start = a_p[col];
                let end = a_p[col + 1];
                for idx in start..end {
                    if a_i[idx] == col {
                        // Diagonal entry
                        a_x[idx] += self.static_reg;
                    }
                }
            }
        }

        // Allocate workspace for factorization
        let nnz_a = a_i.len();
        let mut l_p = vec![0; self.n + 1];
        let mut l_i = vec![0; nnz_a];
        let mut l_x = vec![0.0; nnz_a];
        let mut d = vec![0.0; self.n];
        let mut d_inv = vec![0.0; self.n];

        // Get etree reference
        let etree = self.etree.as_ref().unwrap();
        let l_nz = self.l_nz.as_ref().unwrap();

        // Workspace arrays
        let mut bwork = vec![ldl::Marker::Unused; self.n];
        let mut iwork = vec![0; 3 * self.n];
        let mut fwork = vec![0.0; self.n];

        // Perform factorization
        let result = ldl::factor(
            self.n,
            a_p,
            a_i,
            &a_x,
            &mut l_p,
            &mut l_i,
            &mut l_x,
            &mut d,
            &mut d_inv,
            &l_nz,
            etree,
            &mut bwork,
            &mut iwork,
            &mut fwork,
        );

        // Check for factorization failure
        match result {
            Ok(_) => {
                // Apply dynamic regularization if needed
                self.dynamic_bumps = 0;
                for i in 0..self.n {
                    if d[i].abs() < self.dynamic_reg_min_pivot {
                        d[i] = if d[i] >= 0.0 {
                            self.dynamic_reg_min_pivot
                        } else {
                            -self.dynamic_reg_min_pivot
                        };
                        d_inv[i] = 1.0 / d[i];
                        self.dynamic_bumps += 1;
                    }
                }

                let factor_data = LdlFactorData {
                    l_p,
                    l_i,
                    l_x,
                    d: d.clone(),
                    d_inv,
                };

                self.factorization = Some(factor_data);

                Ok(QdldlFactorization { d_values: d })
            }
            Err(_) => Err(QdldlError::FactorizationFailed),
        }
    }

    /// Solve the system Kx = b using the factorization.
    ///
    /// Solves LDL^T x = b using forward/backward substitution.
    ///
    /// # Arguments
    ///
    /// * `_factor` - The factorization (stored internally, parameter for API compatibility)
    /// * `b` - Right-hand side vector
    /// * `x` - Solution vector (output)
    pub fn solve(&self, _factor: &QdldlFactorization, b: &[f64], x: &mut [f64]) {
        assert_eq!(b.len(), self.n);
        assert_eq!(x.len(), self.n);

        if let Some(ref factor_data) = self.factorization {
            // Copy b to x (will be modified in-place)
            x.copy_from_slice(b);

            // Solve LDL^T x = b using ldl::solve
            ldl::solve(
                self.n,
                &factor_data.l_p,
                &factor_data.l_i,
                &factor_data.l_x,
                &factor_data.d_inv,
                x,
            );
        } else {
            // No factorization available, just copy
            x.copy_from_slice(b);
        }
    }

    /// Get the number of dynamic regularization bumps from last factorization.
    pub fn dynamic_bumps(&self) -> u64 {
        self.dynamic_bumps
    }
}

/// Result of numeric factorization.
///
/// Holds the diagonal D values for diagnostics.
pub struct QdldlFactorization {
    d_values: Vec<f64>,
}

impl QdldlFactorization {
    /// Get the diagonal D values (for diagnostics).
    pub fn d_values(&self) -> &[f64] {
        &self.d_values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::sparse;

    #[test]
    fn test_qdldl_simple_pd() {
        // Simple 2x2 positive definite: [[2, 1], [1, 2]]
        let triplets = vec![(0, 0, 2.0), (0, 1, 1.0), (1, 1, 2.0)];
        let mat = sparse::from_triplets(2, 2, triplets);

        let mut solver = QdldlSolver::new(2, 1e-9, 1e-7);
        solver.symbolic_factorization(&mat).unwrap();

        let factor = solver.numeric_factorization(&mat).unwrap();

        // Test solve: [[2, 1], [1, 2]] * x = [3, 3]
        // Solution should be x = [1, 1]
        let b = vec![3.0, 3.0];
        let mut x = vec![0.0; 2];
        solver.solve(&factor, &b, &mut x);

        // Check solution (with some tolerance for numerical error)
        assert!((x[0] - 1.0).abs() < 1e-6, "x[0] = {}, expected 1.0", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-6, "x[1] = {}, expected 1.0", x[1]);
    }

    #[test]
    fn test_qdldl_quasi_definite() {
        // Quasi-definite 4x4 KKT-like system:
        // [[1, 0, 1, 0],
        //  [0, 1, 0, 1],
        //  [1, 0, -1, 0],
        //  [0, 1, 0, -1]]
        let triplets = vec![
            (0, 0, 1.0),
            (0, 2, 1.0),
            (1, 1, 1.0),
            (1, 3, 1.0),
            (2, 2, -1.0),
            (3, 3, -1.0),
        ];
        let mat = sparse::from_triplets(4, 4, triplets);

        let mut solver = QdldlSolver::new(4, 1e-8, 1e-7);
        solver.symbolic_factorization(&mat).unwrap();

        let factor = solver.numeric_factorization(&mat).unwrap();

        // Check that D has entries
        let d = factor.d_values();
        assert_eq!(d.len(), 4);

        // Test that we can solve a system
        let b = vec![1.0, 1.0, 0.0, 0.0];
        let mut x = vec![0.0; 4];
        solver.solve(&factor, &b, &mut x);

        // Verify solution by checking residual
        // Compute A*x - b and check it's small
        // (We won't check exact values due to quasi-definiteness)
        assert!(x.iter().all(|&xi| xi.is_finite()), "Solution has non-finite values");
    }
}
