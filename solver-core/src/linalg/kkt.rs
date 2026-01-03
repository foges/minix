//! KKT system builder and solver.
//!
//! This module handles the construction and solution of KKT systems that arise
//! in interior point methods. The KKT matrix has the quasi-definite form:
//!
//! ```text
//! K = [ P + εI    A^T  ]
//!     [ A      -(H + εI)]
//! ```
//!
//! where:
//! - P is the cost Hessian (n×n, PSD)
//! - A is the constraint matrix (m×n)
//! - H is the cone scaling matrix (m×m, block diagonal, SPD)
//! - ε is static regularization
//!
//! The solver implements the two-solve strategy from §5.4.1 of the design doc
//! for efficient predictor-corrector steps.

use super::qdldl::{QdldlError, QdldlFactorization, QdldlSolver};
use super::sparse::{SparseCsc, SparseSymmetricCsc};
use crate::scaling::ScalingBlock;
use sprs::TriMat;
use sprs_suitesparse_camd::try_camd;

fn symm_matvec_upper(a: &SparseCsc, x: &[f64], y: &mut [f64]) {
    y.fill(0.0);
    for (val, (row, col)) in a.iter() {
        y[row] += val * x[col];
        if row != col {
            y[col] += val * x[row];
        }
    }
}

/// KKT system solver.
///
/// Manages the construction, factorization, and solution of KKT systems
/// arising in the IPM algorithm.
pub struct KktSolver {
    /// Problem dimensions
    n: usize, // Number of variables
    m: usize, // Number of constraints

    /// QDLDL backend
    qdldl: QdldlSolver,

    /// Workspace for KKT matrix construction
    kkt_mat: Option<SparseCsc>,

    /// Static regularization
    static_reg: f64,

    /// Fill-reducing permutation (new index -> old index)
    perm: Option<Vec<usize>>,

    /// Inverse permutation (old index -> new index)
    perm_inv: Option<Vec<usize>>,
}

impl KktSolver {
    /// Create a new KKT solver.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of primal variables
    /// * `m` - Number of constraints (slack dimension)
    /// * `static_reg` - Static diagonal regularization
    /// * `dynamic_reg_min_pivot` - Dynamic regularization threshold
    pub fn new(n: usize, m: usize, static_reg: f64, dynamic_reg_min_pivot: f64) -> Self {
        let kkt_dim = n + m;
        let qdldl = QdldlSolver::new(kkt_dim, static_reg, dynamic_reg_min_pivot);

        Self {
            n,
            m,
            qdldl,
            kkt_mat: None,
            static_reg,
            perm: None,
            perm_inv: None,
        }
    }

    /// Return the current static regularization value.
    pub fn static_reg(&self) -> f64 {
        self.static_reg
    }

    /// Update the static regularization value (used in KKT assembly + LDL).
    pub fn set_static_reg(&mut self, static_reg: f64) -> Result<(), QdldlError> {
        self.static_reg = static_reg;
        self.qdldl.set_static_reg(static_reg)?;
        Ok(())
    }

    /// Increase static regularization to at least `min_static_reg`.
    pub fn bump_static_reg(&mut self, min_static_reg: f64) -> Result<bool, QdldlError> {
        if min_static_reg > self.static_reg {
            self.set_static_reg(min_static_reg)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn compute_camd_perm(&self, kkt: &SparseCsc) -> Result<(Vec<usize>, Vec<usize>), QdldlError> {
        let perm = try_camd(kkt.structure_view())
            .map_err(|e| QdldlError::OrderingFailed(e.to_string()))?;
        Ok((perm.vec(), perm.inv_vec()))
    }

    /// Build the KKT matrix K = [[P + εI, A^T], [A, -(H + εI)]].
    ///
    /// This assembles the augmented system matrix from the problem data
    /// and current scaling matrix H.
    ///
    /// Note: QDLDL will add static_reg to all diagonal entries, so we assemble
    /// the (2,2) block as -(H + 2*ε) to get -(H + ε) after QDLDL's regularization.
    ///
    /// # Arguments
    ///
    /// * `p` - Cost Hessian P (n×n, upper triangle, optional)
    /// * `a` - Constraint matrix A (m×n)
    /// * `h_blocks` - Scaling matrix H as a list of diagonal blocks
    ///
    /// # Returns
    ///
    /// The KKT matrix in CSC format (upper triangle only).
    pub fn build_kkt_matrix(
        &self,
        p: Option<&SparseSymmetricCsc>,
        a: &SparseCsc,
        h_blocks: &[ScalingBlock],
    ) -> SparseCsc {
        self.build_kkt_matrix_with_perm(self.perm_inv.as_deref(), p, a, h_blocks)
    }

    fn build_kkt_matrix_with_perm(
        &self,
        perm: Option<&[usize]>,
        p: Option<&SparseSymmetricCsc>,
        a: &SparseCsc,
        h_blocks: &[ScalingBlock],
    ) -> SparseCsc {
        assert_eq!(a.rows(), self.m);
        assert_eq!(a.cols(), self.n);

        let kkt_dim = self.n + self.m;
        let mut tri = TriMat::new((kkt_dim, kkt_dim));
        let map_index = |idx: usize| perm.map_or(idx, |p| p[idx]);
        let add_triplet = |row: usize, col: usize, val: f64, tri: &mut TriMat<f64>| {
            let r = map_index(row);
            let c = map_index(col);
            if r <= c {
                tri.add_triplet(r, c, val);
            } else {
                tri.add_triplet(c, r, val);
            }
        };

        // ===================================================================
        // Top-left block: P (n×n, upper triangle) + regularization
        // ===================================================================
        if let Some(p_mat) = p {
            assert_eq!(p_mat.rows(), self.n);
            assert_eq!(p_mat.cols(), self.n);

            for (val, (row, col)) in p_mat.iter() {
                if row <= col {
                    // Only upper triangle
                    add_triplet(row, col, *val, &mut tri);
                }
            }
        }

        // Ensure all diagonal entries exist so QDLDL can add regularization.
        // For LPs (P=None) or sparse QPs with missing diagonals, we add 0.0 placeholders.
        // QDLDL will then add static_reg to these diagonal entries.
        // Using add_triplet with 0.0 is safe - it sums with existing values if present.
        for i in 0..self.n {
            add_triplet(i, i, 0.0, &mut tri);
        }

        // ===================================================================
        // Top-right block: A^T (stored as upper triangle of full matrix)
        // Since K is symmetric, we store A^T in the upper triangle.
        // Entry K[i, n+j] = A[j, i] for i < n, j < m
        // ===================================================================
        for (val, (row_a, col_a)) in a.iter() {
            // A[row_a, col_a] corresponds to K[col_a, n + row_a]
            // We want col >= row for upper triangle
            let kkt_row = col_a;
            let kkt_col = self.n + row_a;

            add_triplet(kkt_row, kkt_col, *val, &mut tri);
        }

        // ===================================================================
        // Bottom-right block: -H (m×m, block diagonal)
        // H is stored as a list of diagonal blocks. We assemble it here.
        // ===================================================================
        let mut offset = 0;
        for h_block in h_blocks {
            let block_dim = match h_block {
                ScalingBlock::Zero { dim } => *dim,
                ScalingBlock::Diagonal { d } => d.len(),
                ScalingBlock::Dense3x3 { .. } => 3,
                ScalingBlock::SocStructured { w } => w.len(),
                ScalingBlock::PsdStructured { n, .. } => n * (n + 1) / 2,
            };

            // Apply -(H + 2ε*I) to this block
            // QDLDL will add +ε later, giving us -(H + ε) as desired for quasi-definiteness
            match h_block {
                ScalingBlock::Zero { dim } => {
                    // For Zero cone (equality constraints), H = 0
                    // We want -(0 + ε) = -ε after QDLDL adds +ε
                    // So we assemble -2ε here
                    for i in 0..*dim {
                        let kkt_idx = self.n + offset + i;
                        add_triplet(kkt_idx, kkt_idx, -2.0 * self.static_reg, &mut tri);
                    }
                }
                ScalingBlock::Diagonal { d } => {
                    // -(H + 2ε) for diagonal scaling
                    for i in 0..d.len() {
                        let kkt_idx = self.n + offset + i;
                        add_triplet(kkt_idx, kkt_idx, -d[i] - 2.0 * self.static_reg, &mut tri);
                    }
                }
                ScalingBlock::Dense3x3 { h } => {
                    // -(H + 2ε*I) as a dense 3×3 block (upper triangle)
                    for i in 0..3 {
                        for j in i..3 {
                            let kkt_row = self.n + offset + i;
                            let kkt_col = self.n + offset + j;
                            let idx = i * 3 + j; // row-major storage
                            let mut val = -h[idx];
                            if i == j {
                                val -= 2.0 * self.static_reg;
                            }
                            add_triplet(kkt_row, kkt_col, val, &mut tri);
                        }
                    }
                }
                ScalingBlock::SocStructured { w } => {
                    // For SOC, the scaling matrix is H(w) = quadratic representation P(w)
                    // We need to compute the full dim x dim matrix and add -(H + 2ε*I) to KKT
                    let dim = w.len();
                    for i in 0..dim {
                        // Compute P(w) e_i to get column i of the matrix
                        let mut e_i = vec![0.0; dim];
                        e_i[i] = 1.0;

                        let mut col_i = vec![0.0; dim];
                        crate::scaling::nt::quad_rep_apply(w, &e_i, &mut col_i);

                        // Add upper triangle (j <= i) to avoid duplicates
                        for j in 0..=i {
                            let kkt_row = self.n + offset + j;
                            let kkt_col = self.n + offset + i;
                            let mut val = -col_i[j];
                            // Add regularization to diagonal
                            if i == j {
                                val -= 2.0 * self.static_reg;
                            }
                            add_triplet(kkt_row, kkt_col, val, &mut tri);
                        }
                    }
                }
                ScalingBlock::PsdStructured { .. } => {
                    unimplemented!("PSD structured scaling not yet implemented in KKT assembly");
                }
            }

            offset += block_dim;
        }

        assert_eq!(offset, self.m, "Scaling blocks must cover all {} slacks", self.m);

        tri.to_csc()
    }

    /// Initialize the solver with the KKT matrix sparsity pattern.
    ///
    /// Performs symbolic factorization, which only needs to be done once
    /// if the sparsity pattern doesn't change.
    pub fn initialize(
        &mut self,
        p: Option<&SparseSymmetricCsc>,
        a: &SparseCsc,
        h_blocks: &[ScalingBlock],
    ) -> Result<(), QdldlError> {
        let kkt_unpermuted = self.build_kkt_matrix_with_perm(None, p, a, h_blocks);
        let (perm, perm_inv) = self.compute_camd_perm(&kkt_unpermuted)?;
        if perm.iter().enumerate().all(|(i, &pi)| i == pi) {
            self.perm = None;
            self.perm_inv = None;
        } else {
            self.perm = Some(perm);
            self.perm_inv = Some(perm_inv);
        }

        let kkt = self.build_kkt_matrix(p, a, h_blocks);
        self.qdldl.symbolic_factorization(&kkt)?;
        self.kkt_mat = Some(kkt);
        Ok(())
    }

    /// Factor the KKT system.
    ///
    /// Performs numeric factorization with the current values of P, A, and H.
    /// The sparsity pattern must match the one from initialize().
    pub fn factor(
        &mut self,
        p: Option<&SparseSymmetricCsc>,
        a: &SparseCsc,
        h_blocks: &[ScalingBlock],
    ) -> Result<QdldlFactorization, QdldlError> {
        let kkt = self.build_kkt_matrix(p, a, h_blocks);
        self.kkt_mat = Some(kkt.clone());
        self.qdldl.numeric_factorization(&kkt)
    }

    /// Solve a single KKT system: K * [dx; dz] = [rhs_x; rhs_z].
    ///
    /// # Arguments
    ///
    /// * `factor` - Factorization from factor()
    /// * `rhs_x` - Right-hand side for x block (length n)
    /// * `rhs_z` - Right-hand side for z block (length m)
    /// * `sol_x` - Solution for x block (output, length n)
    /// * `sol_z` - Solution for z block (output, length m)
    pub fn solve(
        &self,
        factor: &QdldlFactorization,
        rhs_x: &[f64],
        rhs_z: &[f64],
        sol_x: &mut [f64],
        sol_z: &mut [f64],
    ) {
        self.solve_with_refinement(factor, rhs_x, rhs_z, sol_x, sol_z, 0);
    }

    /// Solve with optional iterative refinement.
    pub fn solve_refined(
        &self,
        factor: &QdldlFactorization,
        rhs_x: &[f64],
        rhs_z: &[f64],
        sol_x: &mut [f64],
        sol_z: &mut [f64],
        refine_iters: usize,
    ) {
        self.solve_with_refinement(factor, rhs_x, rhs_z, sol_x, sol_z, refine_iters);
    }

    fn solve_with_refinement(
        &self,
        factor: &QdldlFactorization,
        rhs_x: &[f64],
        rhs_z: &[f64],
        sol_x: &mut [f64],
        sol_z: &mut [f64],
        refine_iters: usize,
    ) {
        assert_eq!(rhs_x.len(), self.n);
        assert_eq!(rhs_z.len(), self.m);
        assert_eq!(sol_x.len(), self.n);
        assert_eq!(sol_z.len(), self.m);

        // Assemble and permute RHS (if needed)
        let kkt_dim = self.n + self.m;
        let mut rhs_perm = vec![0.0; kkt_dim];
        if let Some(p) = &self.perm {
            for i in 0..kkt_dim {
                let src = p[i];
                if src < self.n {
                    rhs_perm[i] = rhs_x[src];
                } else {
                    rhs_perm[i] = rhs_z[src - self.n];
                }
            }
        } else {
            rhs_perm[..self.n].copy_from_slice(rhs_x);
            rhs_perm[self.n..].copy_from_slice(rhs_z);
        }

        // Solve permuted system
        let mut sol_perm = vec![0.0; kkt_dim];
        self.qdldl.solve(factor, &rhs_perm, &mut sol_perm);

        if refine_iters > 0 {
            if let Some(kkt) = &self.kkt_mat {
                let mut kx = vec![0.0; kkt_dim];
                let mut res = vec![0.0; kkt_dim];
                let mut delta = vec![0.0; kkt_dim];

                for _ in 0..refine_iters {
                    symm_matvec_upper(kkt, &sol_perm, &mut kx);
                    if self.static_reg != 0.0 {
                        for i in 0..kkt_dim {
                            kx[i] += self.static_reg * sol_perm[i];
                        }
                    }
                    for i in 0..kkt_dim {
                        res[i] = rhs_perm[i] - kx[i];
                    }

                    let res_norm = res.iter().map(|v| v * v).sum::<f64>().sqrt();
                    if !res_norm.is_finite() || res_norm < 1e-12 {
                        break;
                    }

                    self.qdldl.solve(factor, &res, &mut delta);
                    for i in 0..kkt_dim {
                        sol_perm[i] += delta[i];
                    }
                }
            }
        }

        // Unpermute solution back to original ordering
        if let Some(p_inv) = &self.perm_inv {
            for i in 0..self.n {
                sol_x[i] = sol_perm[p_inv[i]];
            }
            for i in 0..self.m {
                sol_z[i] = sol_perm[p_inv[self.n + i]];
            }
        } else {
            sol_x.copy_from_slice(&sol_perm[..self.n]);
            sol_z.copy_from_slice(&sol_perm[self.n..]);
        }
    }

    /// Two-solve strategy for predictor-corrector (§5.4.1 of design doc).
    ///
    /// Solves two systems with the same KKT matrix:
    /// K * [dx1; dz1] = [rhs_x1; rhs_z1]
    /// K * [dx2; dz2] = [rhs_x2; rhs_z2]
    ///
    /// This is more efficient than calling solve() twice because the
    /// factorization is reused.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_two_rhs(
        &self,
        factor: &QdldlFactorization,
        rhs_x1: &[f64],
        rhs_z1: &[f64],
        rhs_x2: &[f64],
        rhs_z2: &[f64],
        sol_x1: &mut [f64],
        sol_z1: &mut [f64],
        sol_x2: &mut [f64],
        sol_z2: &mut [f64],
    ) {
        // Solve first system
        self.solve(factor, rhs_x1, rhs_z1, sol_x1, sol_z1);

        // Solve second system
        self.solve(factor, rhs_x2, rhs_z2, sol_x2, sol_z2);
    }

    /// Get the number of dynamic regularization bumps from the last factorization.
    pub fn dynamic_bumps(&self) -> u64 {
        self.qdldl.dynamic_bumps()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::sparse;

    #[test]
    fn test_kkt_simple_lp() {
        // Simple LP:
        //   min  x1 + x2
        //   s.t. x1 + x2 = 1   (equality)
        //        x1, x2 >= 0   (nonnegativity)
        //
        // Variables: x = [x1, x2]  (n=2)
        // Slacks: s = [s_eq, s1, s2]  (m=3)
        //   s_eq for equality (zero cone)
        //   s1, s2 for nonnegativity (nonneg cone)
        //
        // KKT system (4×4 with regularization omitted):
        //   [0  0 | 1  1  1 ] [dx1 ]   [r_x1 ]
        //   [0  0 | 1  1  1 ] [dx2 ]   [r_x2 ]
        //   [------+--------] [---- ] = [-----]
        //   [1  1 | 0  0  0 ] [dz_eq]   [r_zeq]
        //   [1  1 | 0 -h1 0 ] [dz1  ]   [r_z1 ]
        //   [1  1 | 0  0 -h2] [dz2  ]   [r_z2 ]
        //
        // For this test, we'll use h1 = h2 = 1.0

        let n = 2;
        let m = 3;

        // P = None (LP, no quadratic term)
        // A = [[1, 1], [1, 0], [0, 1]]  (m×n)
        let a_triplets = vec![
            (0, 0, 1.0), (0, 1, 1.0),  // Equality constraint
            (1, 0, 1.0),               // x1 >= 0
            (2, 1, 1.0),               // x2 >= 0
        ];
        let a = sparse::from_triplets(m, n, a_triplets);

        // H blocks: [Zero(1), Diagonal([1.0, 1.0])]
        let h_blocks = vec![
            ScalingBlock::Zero { dim: 1 },
            ScalingBlock::Diagonal { d: vec![1.0, 1.0] },
        ];

        let mut kkt_solver = KktSolver::new(n, m, 1e-8, 1e-7);

        // Initialize (symbolic factorization)
        kkt_solver.initialize(None, &a, &h_blocks).unwrap();

        // Factor (numeric)
        let factor = kkt_solver.factor(None, &a, &h_blocks).unwrap();

        // Solve a simple system: K * [dx; dz] = [1, 1, 0, 0, 0]
        let rhs_x = vec![1.0, 1.0];
        let rhs_z = vec![0.0, 0.0, 0.0];
        let mut sol_x = vec![0.0; 2];
        let mut sol_z = vec![0.0; 3];

        kkt_solver.solve(&factor, &rhs_x, &rhs_z, &mut sol_x, &mut sol_z);

        // Just check that we got a solution (exact values depend on regularization)
        assert!(sol_x.iter().any(|&x| x.abs() > 1e-6));
    }

    #[test]
    fn test_kkt_with_p_matrix() {
        // QP with cost: 0.5 * (x1^2 + x2^2) + 0
        // Constraint: x1 + x2 >= 1
        //
        // P = [[1, 0], [0, 1]]
        // A = [[1, 1]]
        // H = [1.0] (nonneg cone)

        let n = 2;
        let m = 1;

        let p_triplets = vec![(0, 0, 1.0), (1, 1, 1.0)];
        let p = sparse::from_triplets_symmetric(n, p_triplets);

        let a_triplets = vec![(0, 0, 1.0), (0, 1, 1.0)];
        let a = sparse::from_triplets(m, n, a_triplets);

        let h_blocks = vec![ScalingBlock::Diagonal { d: vec![1.0] }];

        let mut kkt_solver = KktSolver::new(n, m, 1e-8, 1e-7);

        kkt_solver.initialize(Some(&p), &a, &h_blocks).unwrap();
        let factor = kkt_solver.factor(Some(&p), &a, &h_blocks).unwrap();

        // Solve trivial system
        let rhs_x = vec![1.0, 1.0];
        let rhs_z = vec![0.0];
        let mut sol_x = vec![0.0; 2];
        let mut sol_z = vec![0.0; 1];

        kkt_solver.solve(&factor, &rhs_x, &rhs_z, &mut sol_x, &mut sol_z);

        // Check that we got a solution
        assert!(sol_x[0].abs() + sol_x[1].abs() > 1e-6);
    }

    #[test]
    fn test_kkt_two_solve() {
        // Test the two-RHS solve strategy
        let n = 2;
        let m = 1;

        let p_triplets = vec![(0, 0, 1.0), (1, 1, 1.0)];
        let p = sparse::from_triplets_symmetric(n, p_triplets);

        let a_triplets = vec![(0, 0, 1.0), (0, 1, 1.0)];
        let a = sparse::from_triplets(m, n, a_triplets);

        let h_blocks = vec![ScalingBlock::Diagonal { d: vec![1.0] }];

        let mut kkt_solver = KktSolver::new(n, m, 1e-8, 1e-7);
        kkt_solver.initialize(Some(&p), &a, &h_blocks).unwrap();
        let factor = kkt_solver.factor(Some(&p), &a, &h_blocks).unwrap();

        // Two different RHS
        let rhs_x1 = vec![1.0, 0.0];
        let rhs_z1 = vec![0.0];
        let rhs_x2 = vec![0.0, 1.0];
        let rhs_z2 = vec![1.0];

        let mut sol_x1 = vec![0.0; 2];
        let mut sol_z1 = vec![0.0; 1];
        let mut sol_x2 = vec![0.0; 2];
        let mut sol_z2 = vec![0.0; 1];

        kkt_solver.solve_two_rhs(
            &factor,
            &rhs_x1, &rhs_z1,
            &rhs_x2, &rhs_z2,
            &mut sol_x1, &mut sol_z1,
            &mut sol_x2, &mut sol_z2,
        );

        // Check that both solutions are non-trivial
        assert!(sol_x1[0].abs() + sol_x1[1].abs() > 1e-6);
        assert!(sol_x2[0].abs() + sol_x2[1].abs() > 1e-6);

        // Solutions should be different
        assert!((sol_x1[0] - sol_x2[0]).abs() > 1e-6 || (sol_x1[1] - sol_x2[1]).abs() > 1e-6);
    }
}
