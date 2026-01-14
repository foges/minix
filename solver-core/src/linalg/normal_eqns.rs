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

use super::backend::BackendError;
use super::kkt_trait::KktSolverTrait;
use super::sparse::{SparseCsc, SparseSymmetricCsc};
use crate::scaling::ScalingBlock;
use nalgebra::{DMatrix, DVector, Cholesky};

/// Marker type for normal equations factorization.
///
/// The actual Cholesky factor is stored inside the solver.
#[derive(Debug, Clone)]
pub struct NormalEqnsFactor;

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

    /// Cached H diagonal values from last update_numeric
    h_diag: Vec<f64>,

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
            h_diag: vec![0.0; m],
            h_inv_work: vec![0.0; m],
            at_v: DVector::zeros(n),
            a_v: DVector::zeros(m),
            chol: None,
        }
    }

    /// Extract H diagonal from scaling blocks.
    fn extract_h_diag(h_blocks: &[ScalingBlock], h_diag: &mut [f64]) {
        let mut offset = 0;
        for block in h_blocks {
            match block {
                ScalingBlock::Zero { dim } => {
                    for i in 0..*dim {
                        h_diag[offset + i] = 0.0;
                    }
                    offset += dim;
                }
                ScalingBlock::Diagonal { d } => {
                    h_diag[offset..offset + d.len()].copy_from_slice(d);
                    offset += d.len();
                }
                _ => panic!("Normal equations only support Zero and Diagonal (NonNeg) cones"),
            }
        }
    }

    /// Build the Schur complement matrix from current h_diag.
    fn build_schur(&mut self) {
        // Compute H^{-1} diagonal (with regularization)
        for i in 0..self.m {
            let h_val = self.h_diag[i] + self.static_reg;
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
        self.h_diag.copy_from_slice(h_diag);
        self.build_schur();

        // Cholesky factorization
        self.chol = Cholesky::new(self.schur.clone());

        if self.chol.is_none() {
            return Err("Normal equations Cholesky factorization failed".to_string());
        }

        Ok(())
    }

    /// Get current static regularization.
    pub fn static_reg(&self) -> f64 {
        self.static_reg
    }

    /// Set static regularization (requires rebuilding P diagonal).
    pub fn set_static_reg(&mut self, reg: f64) {
        // Adjust P diagonal: remove old reg, add new
        let diff = reg - self.static_reg;
        for i in 0..self.n {
            self.p_dense[(i, i)] += diff;
        }
        self.static_reg = reg;
    }

    /// Solve the system given RHS vectors.
    ///
    /// Solves the KKT system:
    /// ```text
    /// [P    A^T] [dx]   [rhs_x]
    /// [A   -H  ] [dz] = [rhs_z]
    /// ```
    /// Using normal equations:
    /// ```text
    /// S * dx = rhs_x + A^T * H^{-1} * rhs_z   where S = P + A^T * H^{-1} * A
    /// dz = H^{-1} * (A * dx - rhs_z)
    /// ```
    pub fn solve(
        &mut self,
        rhs_x: &[f64],
        rhs_z: &[f64],
        sol_x: &mut [f64],
        sol_z: &mut [f64],
    ) {
        let chol = self.chol.as_ref().expect("Must call update_and_factor first");

        // Compute H^{-1} * rhs_z
        for i in 0..self.m {
            let h_val = self.h_diag[i] + self.static_reg;
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

        // Compute dz = H^{-1} * (A * dx - rhs_z)
        // From KKT: A*dx - H*dz = rhs_z  =>  dz = H^{-1}*(A*dx - rhs_z)
        // a_v = A * dx
        self.a_v = &self.a_dense * &dx;

        for i in 0..self.m {
            let h_val = self.h_diag[i] + self.static_reg;
            sol_z[i] = if h_val.abs() > 1e-14 {
                (self.a_v[i] - rhs_z[i]) / h_val
            } else {
                0.0
            };
        }
    }

    /// Solve with old API that takes h_diag parameter (for backward compatibility).
    #[allow(dead_code)]
    pub fn solve_with_h_diag(
        &mut self,
        h_diag: &[f64],
        rhs_x: &[f64],
        rhs_z: &[f64],
        sol_x: &mut [f64],
        sol_z: &mut [f64],
    ) {
        self.h_diag.copy_from_slice(h_diag);
        self.solve(rhs_x, rhs_z, sol_x, sol_z);
    }
}

impl KktSolverTrait for NormalEqnsSolver {
    type Factor = NormalEqnsFactor;

    fn initialize(
        &mut self,
        _p: Option<&SparseSymmetricCsc>,
        _a: &SparseCsc,
        _h_blocks: &[ScalingBlock],
    ) -> Result<(), BackendError> {
        // For normal equations, initialization is done in the constructor.
        // The sparsity pattern is converted to dense at construction time.
        Ok(())
    }

    fn update_numeric(
        &mut self,
        _p: Option<&SparseSymmetricCsc>,
        _a: &SparseCsc,
        h_blocks: &[ScalingBlock],
    ) -> Result<(), BackendError> {
        // Extract h_diag from scaling blocks
        Self::extract_h_diag(h_blocks, &mut self.h_diag);
        // Build the Schur complement matrix
        self.build_schur();
        Ok(())
    }

    fn factorize(&mut self) -> Result<Self::Factor, BackendError> {
        // Cholesky factorization
        self.chol = Cholesky::new(self.schur.clone());

        if self.chol.is_none() {
            return Err(BackendError::Message(
                "Normal equations Cholesky factorization failed".to_string(),
            ));
        }

        Ok(NormalEqnsFactor)
    }

    fn solve_refined(
        &mut self,
        _factor: &Self::Factor,
        rhs_x: &[f64],
        rhs_z: &[f64],
        sol_x: &mut [f64],
        sol_z: &mut [f64],
        _refine_iters: usize,
    ) {
        // For dense Cholesky, refinement is not typically needed
        // (the factorization is quite stable).
        self.solve(rhs_x, rhs_z, sol_x, sol_z);
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_two_rhs_refined_tagged(
        &mut self,
        _factor: &Self::Factor,
        rhs_x1: &[f64],
        rhs_z1: &[f64],
        rhs_x2: &[f64],
        rhs_z2: &[f64],
        sol_x1: &mut [f64],
        sol_z1: &mut [f64],
        sol_x2: &mut [f64],
        sol_z2: &mut [f64],
        _refine_iters: usize,
        _tag1: &'static str,
        _tag2: &'static str,
    ) {
        // For dense systems, just call solve twice - dense ops are fast
        self.solve(rhs_x1, rhs_z1, sol_x1, sol_z1);
        self.solve(rhs_x2, rhs_z2, sol_x2, sol_z2);
    }

    fn static_reg(&self) -> f64 {
        self.static_reg
    }

    fn set_static_reg(&mut self, reg: f64) -> Result<(), BackendError> {
        NormalEqnsSolver::set_static_reg(self, reg);
        Ok(())
    }

    fn bump_static_reg(&mut self, min_reg: f64) -> Result<bool, BackendError> {
        if min_reg > self.static_reg {
            NormalEqnsSolver::set_static_reg(self, min_reg);
            return Ok(true);
        }
        Ok(false)
    }

    fn dynamic_bumps(&self) -> u64 {
        // Dense Cholesky doesn't need dynamic regularization bumps
        0
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

        solver.solve(&rhs_x, &rhs_z, &mut sol_x, &mut sol_z);

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

    /// Test the normal equations solve with non-zero rhs_z.
    ///
    /// This test specifically verifies the sign in the dz computation:
    ///   dz = H^{-1} * (A * dx - rhs_z)
    ///
    /// A sign error here would cause wrong dual residuals in the IPM,
    /// leading to poor convergence on tall problems like KSIP.
    #[test]
    fn test_normal_eqns_nonzero_rhs_z() {
        // Create a small tall problem (m > n)
        // This is similar to the KSIP structure that exposed the sign bug
        let n = 3;
        let m = 10;

        // A is a random tall matrix
        let mut a_triplets = vec![];
        for i in 0..m {
            for j in 0..n {
                // Deterministic "random" pattern based on indices
                let val = ((i * n + j) % 7) as f64 - 3.0;
                if val != 0.0 {
                    a_triplets.push((i, j, val));
                }
            }
        }
        let a = sparse::from_triplets(m, n, a_triplets.clone());

        // Small P matrix for regularization
        let p_triplets = vec![(0, 0, 0.1), (1, 1, 0.1), (2, 2, 0.1)];
        let p = sparse::from_triplets_symmetric(n, p_triplets);

        let static_reg = 1e-8;
        let mut solver = NormalEqnsSolver::new(n, m, Some(&p), &a, static_reg);

        // H = diag of positive values
        let h_diag: Vec<f64> = (0..m).map(|i| 1.0 + (i as f64) * 0.1).collect();
        solver.update_and_factor(&h_diag).unwrap();

        // Non-trivial RHS
        let rhs_x = vec![1.0, -0.5, 0.3];
        let rhs_z: Vec<f64> = (0..m).map(|i| (i as f64) * 0.2 - 1.0).collect();
        let mut sol_x = vec![0.0; n];
        let mut sol_z = vec![0.0; m];

        solver.solve(&rhs_x, &rhs_z, &mut sol_x, &mut sol_z);

        // Verify full KKT residual:
        // [P+εI    A^T  ] [dx]   [rhs_x]
        // [A    -(H+εI)] [dz] = [rhs_z]

        // Residual 1: (P+εI)*dx + A^T*dz - rhs_x
        let mut resid_x = vec![0.0; n];
        // P*dx (only diagonal since our P is diagonal)
        for i in 0..n {
            resid_x[i] += (0.1 + static_reg) * sol_x[i];
        }
        // A^T*dz
        for &(row, col, val) in &a_triplets {
            resid_x[col] += val * sol_z[row];
        }
        // -rhs_x
        for i in 0..n {
            resid_x[i] -= rhs_x[i];
        }

        let resid_x_norm: f64 = resid_x.iter().map(|x| x * x).sum::<f64>().sqrt();

        // Residual 2: A*dx - (H+εI)*dz - rhs_z
        let mut resid_z = vec![0.0; m];
        // A*dx
        for &(row, col, val) in &a_triplets {
            resid_z[row] += val * sol_x[col];
        }
        // -(H+εI)*dz
        for i in 0..m {
            resid_z[i] -= (h_diag[i] + static_reg) * sol_z[i];
        }
        // -rhs_z
        for i in 0..m {
            resid_z[i] -= rhs_z[i];
        }

        let resid_z_norm: f64 = resid_z.iter().map(|x| x * x).sum::<f64>().sqrt();

        // Both residuals should be small
        assert!(
            resid_x_norm < 1e-6,
            "dx residual too large: {} (should be < 1e-6)",
            resid_x_norm
        );
        assert!(
            resid_z_norm < 1e-6,
            "dz residual too large: {} - sign error in dz = H^{{-1}}*(A*dx - rhs_z)?",
            resid_z_norm
        );
    }

    /// Test that normal equations produce the same solution as direct KKT for a small problem.
    /// This helps catch sign or formula errors in the Schur complement reduction.
    #[test]
    fn test_normal_eqns_matches_direct_kkt() {
        let n = 2;
        let m = 5;

        // Simple A matrix
        let a_triplets = vec![
            (0, 0, 2.0), (0, 1, 1.0),
            (1, 0, 1.0), (1, 1, 3.0),
            (2, 0, 1.0),
            (3, 1, 1.0),
            (4, 0, 1.0), (4, 1, 1.0),
        ];
        let a = sparse::from_triplets(m, n, a_triplets.clone());

        let static_reg = 1e-6;
        let mut solver = NormalEqnsSolver::new(n, m, None, &a, static_reg);

        // Simple diagonal H
        let h_diag = vec![2.0, 3.0, 1.0, 4.0, 2.0];
        solver.update_and_factor(&h_diag).unwrap();

        let rhs_x = vec![1.0, 2.0];
        let rhs_z = vec![0.5, -0.5, 1.0, 0.0, -1.0];
        let mut sol_x = vec![0.0; n];
        let mut sol_z = vec![0.0; m];

        solver.solve(&rhs_x, &rhs_z, &mut sol_x, &mut sol_z);

        // Build and solve the direct KKT system for comparison
        // [εI    A^T  ] [dx]   [rhs_x]
        // [A  -(H+εI)] [dz] = [rhs_z]
        use nalgebra::{DMatrix, DVector};
        let kkt_dim = n + m;
        let mut kkt = DMatrix::zeros(kkt_dim, kkt_dim);

        // Top-left: εI
        for i in 0..n {
            kkt[(i, i)] = static_reg;
        }
        // Top-right: A^T
        for &(row, col, val) in &a_triplets {
            kkt[(col, n + row)] = val;
        }
        // Bottom-left: A
        for &(row, col, val) in &a_triplets {
            kkt[(n + row, col)] = val;
        }
        // Bottom-right: -(H+εI)
        for i in 0..m {
            kkt[(n + i, n + i)] = -(h_diag[i] + static_reg);
        }

        let mut rhs = DVector::zeros(kkt_dim);
        for i in 0..n {
            rhs[i] = rhs_x[i];
        }
        for i in 0..m {
            rhs[n + i] = rhs_z[i];
        }

        // Solve directly using LU decomposition
        let sol_direct = kkt.lu().solve(&rhs).expect("Direct KKT solve failed");

        // Compare solutions
        for i in 0..n {
            let diff = (sol_x[i] - sol_direct[i]).abs();
            assert!(
                diff < 1e-6,
                "sol_x[{}] mismatch: normal_eqns={:.6}, direct={:.6}",
                i, sol_x[i], sol_direct[i]
            );
        }
        for i in 0..m {
            let diff = (sol_z[i] - sol_direct[n + i]).abs();
            assert!(
                diff < 1e-6,
                "sol_z[{}] mismatch: normal_eqns={:.6}, direct={:.6}",
                i, sol_z[i], sol_direct[n + i]
            );
        }
    }
}
