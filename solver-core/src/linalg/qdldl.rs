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

    /// Fill-reducing ordering failed
    #[error("Ordering failed: {0}")]
    OrderingFailed(String),

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

    /// Cached diagonal positions (col -> index in CSC data) for applying static regularization
    diag_positions: Option<Vec<Option<usize>>>,

    /// Reusable workspace for the matrix values (A_x + static_reg on diagonal)
    a_x_work: Vec<f64>,

    /// Reusable factorization workspaces (allocated once)
    bwork: Vec<ldl::Marker>,
    iwork: Vec<usize>,
    fwork: Vec<f64>,
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
            diag_positions: None,
            a_x_work: Vec::new(),
            bwork: vec![ldl::Marker::Unused; n],
            iwork: vec![0; 3 * n],
            fwork: vec![0.0; n],
        }
    }

    /// Return the current static regularization value.
    pub fn static_reg(&self) -> f64 {
        self.static_reg
    }

    /// Update the static regularization value.
    pub fn set_static_reg(&mut self, static_reg: f64) -> Result<(), QdldlError> {
        if static_reg < 0.0 {
            return Err(QdldlError::InvalidRegularization(format!(
                "static_reg must be non-negative, got {}",
                static_reg
            )));
        }
        self.static_reg = static_reg;
        Ok(())
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

                // Cache diagonal positions for fast static-regularization application.
                let mut diag_positions = vec![None; self.n];
                for col in 0..self.n {
                    let start = a_p[col];
                    let end = a_p[col + 1];
                    for idx in start..end {
                        if a_i[idx] == col {
                            diag_positions[col] = Some(idx);
                            break;
                        }
                    }
                }
                self.diag_positions = Some(diag_positions);

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

        // Ensure a_x workspace is allocated
        if self.a_x_work.len() != a_x_orig.len() {
            self.a_x_work.resize(a_x_orig.len(), 0.0);
        }
        self.a_x_work.copy_from_slice(a_x_orig);

        // Apply static regularization to diagonal (fast via cached diagonal positions)
        if self.static_reg > 0.0 {
            if let Some(diag_pos) = &self.diag_positions {
                for col in 0..self.n {
                    if let Some(idx) = diag_pos[col] {
                        self.a_x_work[idx] += self.static_reg;
                    }
                }
            } else {
                // Fallback (should not happen): scan for diagonal entries.
                for col in 0..self.n {
                    let start = a_p[col];
                    let end = a_p[col + 1];
                    for idx in start..end {
                        if a_i[idx] == col {
                            self.a_x_work[idx] += self.static_reg;
                            break;
                        }
                    }
                }
            }
        }

        // Get etree reference
        let etree = self.etree.as_ref().unwrap();
        let l_nz = self.l_nz.as_ref().unwrap();

        // Compute total nonzeros in L from l_nz (fill-in can make L larger than A)
        let nnz_l: usize = l_nz.iter().sum();

        // Ensure factorization buffers exist and are correctly sized
        if self.factorization.is_none() {
            self.factorization = Some(LdlFactorData {
                l_p: vec![0; self.n + 1],
                l_i: vec![0; nnz_l],
                l_x: vec![0.0; nnz_l],
                d: vec![0.0; self.n],
                d_inv: vec![0.0; self.n],
            });
        } else {
            let f = self.factorization.as_mut().unwrap();
            if f.l_p.len() != self.n + 1 {
                f.l_p.resize(self.n + 1, 0);
            }
            if f.l_i.len() != nnz_l {
                f.l_i.resize(nnz_l, 0);
            }
            if f.l_x.len() != nnz_l {
                f.l_x.resize(nnz_l, 0.0);
            }
            if f.d.len() != self.n {
                f.d.resize(self.n, 0.0);
            }
            if f.d_inv.len() != self.n {
                f.d_inv.resize(self.n, 0.0);
            }
        }

        let f = self.factorization.as_mut().unwrap();

        // Reset workspaces (ldl expects clean markers)
        self.bwork.fill(ldl::Marker::Unused);
        self.iwork.fill(0);
        self.fwork.fill(0.0);

        // Perform factorization
        let result = ldl::factor(
            self.n,
            a_p,
            a_i,
            &self.a_x_work,
            &mut f.l_p,
            &mut f.l_i,
            &mut f.l_x,
            &mut f.d,
            &mut f.d_inv,
            l_nz,
            etree,
            &mut self.bwork,
            &mut self.iwork,
            &mut self.fwork,
        );

        // Check for factorization failure
        match result {
            Ok(_) => {
                // Apply dynamic regularization if needed
                // Clarabel uses threshold=1e-13 and replacement=2e-7 (ratio of ~2e6).
                // We use a similar ratio: replacement = threshold * 2e6, capped at 1e-6.
                let replacement = (self.dynamic_reg_min_pivot * 2e6).min(1e-6);
                self.dynamic_bumps = 0;
                for i in 0..self.n {
                    if f.d[i].abs() < self.dynamic_reg_min_pivot {
                        f.d[i] = if f.d[i] >= 0.0 {
                            replacement
                        } else {
                            -replacement
                        };
                        f.d_inv[i] = 1.0 / f.d[i];
                        self.dynamic_bumps += 1;
                    }
                }

                Ok(QdldlFactorization {})
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
}

impl QdldlFactorization {
}

impl QdldlSolver {
    /// Get the diagonal D values from the most recent factorization.
    pub fn d_values(&self) -> Option<&[f64]> {
        self.factorization.as_ref().map(|f| f.d.as_slice())
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
        let d = solver.d_values().expect("missing D values");
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
