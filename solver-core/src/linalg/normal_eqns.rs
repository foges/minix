//! Normal equations solver for tall problems (m >> n).
//!
//! When the constraint matrix A is tall (m >> n), it's more efficient to
//! solve the Schur complement (normal equations) system instead of the
//! full augmented KKT system:
//!
//! Standard KKT (n+m × n+m):
//! ```text
//! [P + εI    A^T  ] [dx]   [-r_x]
//! [A      -(H+εI)] [dz] = [r_z]
//! ```
//!
//! Normal equations (n × n):
//! ```text
//! S = P + A^T * H^{-1} * A + εI
//! S * dx = -r_x + A^T * H^{-1} * r_z
//! dz = H^{-1} * (r_z + A * dx)
//! ```
//!
//! For KSIP with n=20, m=1000, this reduces from 1020×1020 to 20×20.

use super::sparse::{SparseCsc, SparseSymmetricCsc};
use nalgebra::{DMatrix, DVector, Cholesky};

/// Normal equations KKT solver for tall problems.
pub struct NormalEqnsSolver {
    n: usize,
    m: usize,
    static_reg: f64,

    /// Dense P matrix (base for Schur complement)
    p_dense: DMatrix<f64>,

    /// Dense Schur complement matrix S = P + A^T * H^{-1} * A
    schur: DMatrix<f64>,

    /// Cached A^T as dense matrix for fast matvec
    at_dense: DMatrix<f64>,

    /// Cached A as dense matrix
    a_dense: DMatrix<f64>,

    /// Workspace for H^{-1} * v
    h_inv_work: Vec<f64>,

    /// Workspace for A^T * v
    at_v: DVector<f64>,

    /// Workspace for A * v
    a_v: DVector<f64>,

    /// Cholesky factorization of S
    chol: Option<Cholesky<f64, nalgebra::Dyn>>,
}

impl NormalEqnsSolver {
    /// Create a new normal equations solver.
    ///
    /// Only supports diagonal H blocks (Zero and NonNeg cones).
    pub fn new(
        n: usize,
        m: usize,
        p: Option<&SparseSymmetricCsc>,
        a: &SparseCsc,
        static_reg: f64,
    ) -> Self {
        // Convert A to dense
        let mut a_dense = DMatrix::zeros(m, n);
        for (&val, (row, col)) in a.iter() {
            a_dense[(row, col)] = val;
        }

        // A^T
        let at_dense = a_dense.transpose();

        // Convert P to dense (symmetric, stored upper triangle)
        let mut p_dense = DMatrix::zeros(n, n);
        if let Some(p_mat) = p {
            for col in 0..n {
                if let Some(col_view) = p_mat.outer_view(col) {
                    for (row, &val) in col_view.iter() {
                        p_dense[(row, col)] += val;
                        if row != col {
                            p_dense[(col, row)] += val;
                        }
                    }
                }
            }
        }

        // Add static regularization to P diagonal
        for i in 0..n {
            p_dense[(i, i)] += static_reg;
        }

        let schur = DMatrix::zeros(n, n);

        Self {
            n,
            m,
            static_reg,
            p_dense,
            schur,
            at_dense,
            a_dense,
            h_inv_work: vec![0.0; m],
            at_v: DVector::zeros(n),
            a_v: DVector::zeros(m),
            chol: None,
        }
    }

    /// Check if normal equations are beneficial for these dimensions.
    pub fn should_use(n: usize, m: usize) -> bool {
        // Use normal equations when m > 5*n and n is small enough for dense ops
        m > 5 * n && n <= 500
    }

    /// Update H^{-1} diagonal and refactorize.
    ///
    /// `h_diag` contains the diagonal of H (scaling block values).
    /// For Zero cone: h_diag[i] = 0 (will be treated as large, making H^{-1} ≈ 0)
    /// For NonNeg cone: h_diag[i] = s[i]/z[i] (the NT scaling)
    pub fn update_and_factor(&mut self, h_diag: &[f64]) -> Result<(), String> {
        assert_eq!(h_diag.len(), self.m);

        // Compute H^{-1} diagonal (with regularization)
        for i in 0..self.m {
            let h_val = h_diag[i] + self.static_reg;
            self.h_inv_work[i] = if h_val.abs() > 1e-14 {
                1.0 / h_val
            } else {
                0.0 // Zero cone or near-zero
            };
        }

        // Build S = P + εI + A^T * H^{-1} * A
        // Start with P + εI (stored in p_dense)
        self.schur.copy_from(&self.p_dense);

        // Add A^T * diag(h_inv) * A
        // S[i,j] += sum_k A[k,i] * h_inv[k] * A[k,j]
        // This is O(n^2 * m) but n is small
        for i in 0..self.n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..self.m {
                    sum += self.at_dense[(i, k)] * self.h_inv_work[k] * self.a_dense[(k, j)];
                }
                self.schur[(i, j)] += sum;
                if i != j {
                    self.schur[(j, i)] += sum;
                }
            }
        }

        // Cholesky factorization
        self.chol = Cholesky::new(self.schur.clone());

        if self.chol.is_none() {
            return Err("Normal equations Cholesky factorization failed".to_string());
        }

        Ok(())
    }

    /// Solve the system given RHS vectors.
    ///
    /// Solves:
    /// ```text
    /// S * dx = rhs_x + A^T * H^{-1} * rhs_z
    /// dz = H^{-1} * (rhs_z + A * dx)
    /// ```
    pub fn solve(
        &mut self,
        h_diag: &[f64],
        rhs_x: &[f64],
        rhs_z: &[f64],
        sol_x: &mut [f64],
        sol_z: &mut [f64],
    ) {
        let chol = self.chol.as_ref().expect("Must call update_and_factor first");

        // Compute H^{-1} * rhs_z
        for i in 0..self.m {
            let h_val = h_diag[i] + self.static_reg;
            self.h_inv_work[i] = if h_val.abs() > 1e-14 {
                rhs_z[i] / h_val
            } else {
                0.0
            };
        }

        // Compute rhs_reduced = rhs_x + A^T * H^{-1} * rhs_z
        // at_v = A^T * h_inv_work
        let h_inv_vec = DVector::from_column_slice(&self.h_inv_work);
        self.at_v = &self.at_dense * &h_inv_vec;

        let mut rhs_reduced = DVector::from_column_slice(rhs_x);
        rhs_reduced += &self.at_v;

        // Solve S * dx = rhs_reduced
        let dx = chol.solve(&rhs_reduced);

        // Copy dx to sol_x
        for i in 0..self.n {
            sol_x[i] = dx[i];
        }

        // Compute dz = H^{-1} * (rhs_z + A * dx)
        // a_v = A * dx
        self.a_v = &self.a_dense * &dx;

        for i in 0..self.m {
            let h_val = h_diag[i] + self.static_reg;
            sol_z[i] = if h_val.abs() > 1e-14 {
                (rhs_z[i] + self.a_v[i]) / h_val
            } else {
                0.0
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::sparse;

    #[test]
    fn test_normal_eqns_simple() {
        // Simple 2-var, 4-constraint problem
        // A = [[1, 0], [0, 1], [1, 1], [1, -1]]
        let n = 2;
        let m = 4;

        let a_triplets = vec![
            (0, 0, 1.0),
            (1, 1, 1.0),
            (2, 0, 1.0), (2, 1, 1.0),
            (3, 0, 1.0), (3, 1, -1.0),
        ];
        let a = sparse::from_triplets(m, n, a_triplets);

        let mut solver = NormalEqnsSolver::new(n, m, None, &a, 1e-8);

        // H = diag([1, 1, 1, 1])
        let h_diag = vec![1.0, 1.0, 1.0, 1.0];
        solver.update_and_factor(&h_diag).unwrap();

        // Simple RHS
        let rhs_x = vec![1.0, 1.0];
        let rhs_z = vec![0.0, 0.0, 0.0, 0.0];
        let mut sol_x = vec![0.0; n];
        let mut sol_z = vec![0.0; m];

        solver.solve(&h_diag, &rhs_x, &rhs_z, &mut sol_x, &mut sol_z);

        // Verify solution satisfies the original KKT system approximately
        // [εI    A^T] [dx]   [rhs_x]
        // [A    -H  ] [dz] = [rhs_z]

        // Check: A * dx - H * dz ≈ rhs_z
        let mut resid = vec![0.0; m];
        for (&val, (row, col)) in a.iter() {
            resid[row] += val * sol_x[col];
        }
        for i in 0..m {
            resid[i] -= h_diag[i] * sol_z[i];
        }

        let resid_norm: f64 = resid.iter().map(|x| x*x).sum::<f64>().sqrt();
        assert!(resid_norm < 1e-6, "Residual too large: {}", resid_norm);
    }
}
