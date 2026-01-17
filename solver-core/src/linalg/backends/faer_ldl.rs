//! Faer sparse LDL backend for high-performance parallel factorization.
//!
//! This module provides a KktBackend implementation using faer's supernodal
//! sparse LDL factorization, which leverages parallelism via rayon.

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::cholesky::ldlt::factor::{LdltParams, LdltRegularization};
use faer::sparse::linalg::amd::Control as AmdControl;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, LdltRef, SymbolicCholesky,
    SymmetricOrdering,
};
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Conj, MatMut, Par, Side, Spec};

use crate::linalg::backend::{BackendError, KktBackend};
use crate::linalg::sparse::SparseCsc;

/// Faer LDL backend for KKT systems.
///
/// Uses faer's supernodal sparse LDL factorization with dynamic regularization
/// for quasi-definite systems. This backend enables parallel factorization
/// through rayon.
pub struct FaerLdlBackend {
    n: usize,
    static_reg: f64,
    dynamic_reg_eps: f64,
    dynamic_reg_delta: f64,

    // Permuted matrix storage (faer wants sorted columns)
    perm: Vec<usize>,
    iperm: Vec<usize>,
    perm_colptr: Vec<usize>,
    perm_rowval: Vec<usize>,
    perm_nzval: Vec<f64>,
    perm_map: Vec<usize>, // Maps original nzval index to permuted index

    // Cached diagonal positions in perm_nzval (for fast static_reg addition)
    perm_diag_positions: Vec<usize>,

    // Diagonal sign pattern for dynamic regularization (permuted order)
    perm_dsigns: Vec<i8>,

    // Symbolic factorization (computed once)
    symbolic: Option<SymbolicCholesky<usize>>,

    // Numeric factor values
    ld_vals: Vec<f64>,

    // Workspace for solve
    work: Option<MemBuffer>,
    bperm: Vec<f64>,

    // Parallelism setting
    parallelism: Par,

    // LDL parameters
    ldlt_params: Spec<LdltParams, f64>,

    // Track dynamic regularization bumps
    dynamic_bump_count: u64,
}

impl FaerLdlBackend {
    /// Permute symmetric matrix PAP' using inverse permutation.
    /// Returns permuted colptr, rowval, nzval, and a mapping from original to permuted indices.
    fn permute_symmetric(
        colptr: &[usize],
        rowval: &[usize],
        nzval: &[f64],
        iperm: &[usize],
    ) -> (Vec<usize>, Vec<usize>, Vec<f64>, Vec<usize>) {
        let n = colptr.len() - 1;
        let nnz = nzval.len();

        // Count entries per column in permuted matrix
        let mut col_counts = vec![0usize; n];
        for (old_row, old_col) in iter_csc(colptr, rowval) {
            let new_row = iperm[old_row];
            let new_col = iperm[old_col];
            // Keep upper triangle
            let (_nr, nc) = if new_row <= new_col {
                (new_row, new_col)
            } else {
                (new_col, new_row)
            };
            col_counts[nc] += 1;
        }

        // Build column pointers
        let mut new_colptr = vec![0usize; n + 1];
        for i in 0..n {
            new_colptr[i + 1] = new_colptr[i] + col_counts[i];
        }

        // Fill values
        let mut new_rowval = vec![0usize; nnz];
        let mut new_nzval = vec![0.0; nnz];
        let mut perm_map = vec![0usize; nnz];

        col_counts.fill(0);
        for (orig_idx, (old_row, old_col)) in iter_csc(colptr, rowval).enumerate() {
            let new_row = iperm[old_row];
            let new_col = iperm[old_col];
            let (nr, nc) = if new_row <= new_col {
                (new_row, new_col)
            } else {
                (new_col, new_row)
            };
            let pos = new_colptr[nc] + col_counts[nc];
            new_rowval[pos] = nr;
            new_nzval[pos] = nzval[orig_idx];
            perm_map[orig_idx] = pos;
            col_counts[nc] += 1;
        }

        // Sort each column by row index
        let mut temp_rows = Vec::new();
        let mut temp_vals = Vec::new();
        let mut temp_orig = Vec::new();

        // We need to track original indices through the sort
        // Create inverse perm_map to update after sorting
        let mut inv_perm_map = vec![0usize; nnz];
        for (orig, &perm_pos) in perm_map.iter().enumerate() {
            inv_perm_map[perm_pos] = orig;
        }

        for c in 0..n {
            let start = new_colptr[c];
            let end = new_colptr[c + 1];
            let len = end - start;
            if len <= 1 {
                continue;
            }

            temp_rows.clear();
            temp_vals.clear();
            temp_orig.clear();

            for i in start..end {
                temp_rows.push(new_rowval[i]);
                temp_vals.push(new_nzval[i]);
                temp_orig.push(inv_perm_map[i]);
            }

            // Sort by row index
            let mut indices: Vec<usize> = (0..len).collect();
            indices.sort_by_key(|&i| temp_rows[i]);

            for (i, &idx) in indices.iter().enumerate() {
                new_rowval[start + i] = temp_rows[idx];
                new_nzval[start + i] = temp_vals[idx];
                perm_map[temp_orig[idx]] = start + i;
            }
        }

        (new_colptr, new_rowval, new_nzval, perm_map)
    }

    fn compute_amd_ordering(kkt: &SparseCsc) -> (Vec<usize>, Vec<usize>) {
        use sprs_suitesparse_camd::try_camd;

        // Try CAMD ordering using structure view (no values needed)
        match try_camd(kkt.structure_view()) {
            Ok(perm) => (perm.vec(), perm.inv_vec()),
            Err(_) => {
                // Fall back to identity
                let n = kkt.cols();
                let perm: Vec<usize> = (0..n).collect();
                let iperm = perm.clone();
                (perm, iperm)
            }
        }
    }

    fn setup_dsigns_and_diag_positions(&mut self) {
        // For quasi-definite KKT system: positive definite in x block, negative definite in z block
        // The KKT matrix is [P+εI, A'; A, -(H+εI)]
        // We infer the sign pattern from the diagonal values:
        // - Positive diagonal entries get sign +1 (regularize positively if too small)
        // - Negative diagonal entries get sign -1 (regularize negatively if too small)
        //
        // Also cache diagonal positions for fast static_reg addition.

        self.perm_dsigns = vec![1i8; self.n];
        self.perm_diag_positions = vec![usize::MAX; self.n]; // MAX means not found

        // Extract diagonal values, set signs, and cache positions (in permuted order)
        for col in 0..self.n {
            let start = self.perm_colptr[col];
            let end = self.perm_colptr[col + 1];
            for i in start..end {
                if self.perm_rowval[i] == col {
                    // Found diagonal entry
                    self.perm_diag_positions[col] = i;
                    if self.perm_nzval[i] < 0.0 {
                        self.perm_dsigns[col] = -1;
                    }
                    break;
                }
            }
        }
    }

}

/// Iterate over (row, col) pairs in a CSC matrix.
fn iter_csc<'a>(
    colptr: &'a [usize],
    rowval: &'a [usize],
) -> impl Iterator<Item = (usize, usize)> + 'a {
    let n = colptr.len() - 1;
    (0..n).flat_map(move |col| {
        let start = colptr[col];
        let end = colptr[col + 1];
        rowval[start..end].iter().map(move |&row| (row, col))
    })
}

impl KktBackend for FaerLdlBackend {
    type Factorization = ();

    fn new(n: usize, static_reg: f64, dynamic_reg_min_pivot: f64) -> Self {
        // Determine parallelism from environment or use all cores
        let parallelism = if let Ok(threads) = std::env::var("MINIX_THREADS") {
            match threads.parse::<usize>() {
                Ok(0) => Par::rayon(0), // 0 means auto
                Ok(1) => Par::Seq,
                Ok(t) => Par::rayon(t),
                Err(_) => Par::rayon(0),
            }
        } else {
            Par::rayon(0) // Use all available threads
        };

        Self {
            n,
            static_reg,
            dynamic_reg_eps: dynamic_reg_min_pivot,
            dynamic_reg_delta: dynamic_reg_min_pivot * 1e-4,
            perm: Vec::new(),
            iperm: Vec::new(),
            perm_colptr: Vec::new(),
            perm_rowval: Vec::new(),
            perm_nzval: Vec::new(),
            perm_map: Vec::new(),
            perm_diag_positions: Vec::new(),
            perm_dsigns: Vec::new(),
            symbolic: None,
            ld_vals: Vec::new(),
            work: None,
            bperm: Vec::new(),
            parallelism,
            ldlt_params: Spec::default(),
            dynamic_bump_count: 0,
        }
    }

    fn set_static_reg(&mut self, static_reg: f64) -> Result<(), BackendError> {
        if !static_reg.is_finite() || static_reg < 0.0 {
            return Err(BackendError::Message(format!(
                "invalid static_reg {}",
                static_reg
            )));
        }
        self.static_reg = static_reg;
        Ok(())
    }

    fn static_reg(&self) -> f64 {
        self.static_reg
    }

    fn symbolic_factorization(&mut self, kkt: &SparseCsc) -> Result<(), BackendError> {
        self.n = kkt.cols();

        // Compute AMD ordering
        let (perm, iperm) = Self::compute_amd_ordering(kkt);
        self.perm = perm;
        self.iperm = iperm;

        // Permute matrix
        let indptr = kkt.indptr();
        let colptr = indptr.raw_storage();
        let rowval = kkt.indices();
        let nzval = kkt.data();

        let (perm_colptr, perm_rowval, perm_nzval, perm_map) =
            Self::permute_symmetric(colptr, rowval, nzval, &self.iperm);

        self.perm_colptr = perm_colptr;
        self.perm_rowval = perm_rowval;
        self.perm_nzval = perm_nzval;
        self.perm_map = perm_map;

        // Setup diagonal signs and cache diagonal positions for fast regularization
        self.setup_dsigns_and_diag_positions();

        // Build symbolic structure
        let symb_mat = SymbolicSparseColMatRef::new_checked(
            self.n,
            self.n,
            &self.perm_colptr,
            None,
            &self.perm_rowval,
        );

        let cholesky_params = CholeskySymbolicParams {
            supernodal_flop_ratio_threshold: SupernodalThreshold::AUTO,
            amd_params: AmdControl::default(),
            ..Default::default()
        };

        let symbolic = factorize_symbolic_cholesky(
            symb_mat,
            Side::Upper,
            SymmetricOrdering::Identity, // We've already permuted
            cholesky_params,
        )
        .map_err(|e| BackendError::Message(format!("faer symbolic factorization failed: {:?}", e)))?;

        // Allocate space for L*D values
        self.ld_vals = vec![0.0; symbolic.len_val()];

        // Allocate workspace
        let req_factor = symbolic.factorize_numeric_ldlt_scratch::<f64>(self.parallelism, self.ldlt_params);
        let req_solve = symbolic.solve_in_place_scratch::<f64>(1, self.parallelism);
        let req = StackReq::any_of(&[req_factor, req_solve]);
        self.work = Some(MemBuffer::new(req));

        self.bperm = vec![0.0; self.n];
        self.symbolic = Some(symbolic);

        Ok(())
    }

    fn numeric_factorization(&mut self, kkt: &SparseCsc) -> Result<Self::Factorization, BackendError> {
        // Update permuted values from KKT
        let nzval = kkt.data();
        for (orig_idx, &perm_idx) in self.perm_map.iter().enumerate() {
            self.perm_nzval[perm_idx] = nzval[orig_idx];
        }

        // Add static regularization using cached diagonal positions (O(n) instead of O(nnz))
        let static_reg = self.static_reg;
        if static_reg != 0.0 {
            for &diag_pos in &self.perm_diag_positions {
                if diag_pos != usize::MAX {
                    self.perm_nzval[diag_pos] += static_reg;
                }
            }
        }

        // Create sparse matrix view
        let symb_mat = SymbolicSparseColMatRef::new_checked(
            self.n,
            self.n,
            &self.perm_colptr,
            None,
            &self.perm_rowval,
        );
        let mat = SparseColMatRef::new(symb_mat, &self.perm_nzval);

        // Setup regularization
        let regularizer = LdltRegularization {
            dynamic_regularization_signs: Some(&self.perm_dsigns),
            dynamic_regularization_delta: self.dynamic_reg_delta,
            dynamic_regularization_epsilon: self.dynamic_reg_eps,
        };

        // Factorize
        let symbolic = self.symbolic.as_ref().ok_or_else(|| {
            BackendError::Message("symbolic factorization not performed".to_string())
        })?;

        let work = self.work.as_mut().ok_or_else(|| {
            BackendError::Message("workspace not allocated".to_string())
        })?;

        symbolic
            .factorize_numeric_ldlt(
                &mut self.ld_vals,
                mat,
                Side::Upper,
                regularizer,
                self.parallelism,
                MemStack::new(work),
                self.ldlt_params,
            )
            .map_err(|e| BackendError::Message(format!("faer numeric factorization failed: {:?}", e)))?;

        Ok(())
    }

    fn solve(&self, _factor: &Self::Factorization, rhs: &[f64], sol: &mut [f64]) {
        let symbolic = match self.symbolic.as_ref() {
            Some(s) => s,
            None => {
                sol.copy_from_slice(rhs);
                return;
            }
        };

        // Use sol as scratch space for permuted RHS (avoids allocation)
        // Permute RHS: sol[i] = rhs[perm[i]]
        for (i, &p) in self.perm.iter().enumerate() {
            sol[i] = rhs[p];
        }

        // Solve in place using pre-allocated workspace
        let ldlt = LdltRef::new(symbolic, &self.ld_vals);

        // Use pre-allocated work buffer (we need to get a mutable reference)
        // Since self is &self, we need a local buffer. The work buffer is sized
        // for both factor and solve, so we can reuse it. But since we're &self,
        // we need a small allocation here. This is much smaller than n.
        let req_solve = symbolic.solve_in_place_scratch::<f64>(1, self.parallelism);
        let mut work = MemBuffer::new(req_solve);

        let mut rhs_mat = MatMut::from_column_major_slice_mut(sol, self.n, 1);

        ldlt.solve_in_place_with_conj(
            Conj::No,
            rhs_mat.as_mut(),
            self.parallelism,
            MemStack::new(&mut work),
        );

        // Now sol contains the permuted solution. We need to inverse permute.
        // Use the pre-allocated bperm buffer by copying first.
        // But wait - we can't mutate self. Let's do it in-place using sol.
        //
        // We need: final_sol[perm[i]] = current_sol[i]
        // This requires a temporary buffer because indices may overlap.
        //
        // Alternative: inverse permute in-place using cycle following.
        // For simplicity, use a small stack allocation for the temp copy.
        let temp: Vec<f64> = sol[..self.n].to_vec();
        for (i, &p) in self.perm.iter().enumerate() {
            sol[p] = temp[i];
        }
    }

    fn dynamic_bumps(&self) -> u64 {
        self.dynamic_bump_count
    }

    fn estimate_condition_number(&self) -> Option<f64> {
        // Get D values from the LDL factorization
        if self.ld_vals.is_empty() {
            return None;
        }

        // In faer's supernodal format, extracting just D is complex
        // For now, return None - this is optional and QDLDL provides it
        // TODO: Extract diagonal from supernodal structure if needed
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::sparse;

    #[test]
    fn test_faer_backend_simple() {
        // Simple 2x2 positive definite system
        // K = [2, 1; 1, 2]
        let triplets = vec![(0, 0, 2.0), (0, 1, 1.0), (1, 1, 2.0)];
        let kkt = sparse::from_triplets(2, 2, triplets);

        let mut backend = FaerLdlBackend::new(2, 1e-10, 1e-10);
        backend.symbolic_factorization(&kkt).unwrap();
        let factor = backend.numeric_factorization(&kkt).unwrap();

        let rhs = vec![3.0, 4.0];
        let mut sol = vec![0.0; 2];

        backend.solve(&factor, &rhs, &mut sol);

        // Check: K * sol ≈ rhs
        // [2, 1; 1, 2] * [x1, x2] = [3, 4]
        // Solution should be approximately [2/3, 5/3]
        let expected = [2.0 / 3.0, 5.0 / 3.0];
        for i in 0..2 {
            assert!(
                (sol[i] - expected[i]).abs() < 1e-8,
                "sol[{}] = {} != {}",
                i,
                sol[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_faer_backend_quasi_definite() {
        // Quasi-definite system (like KKT)
        // K = [1, 0, 1; 0, 1, 1; 1, 1, -1]
        // This is [P, A'; A, -H] form
        let triplets = vec![
            (0, 0, 1.0),
            (1, 1, 1.0),
            (0, 2, 1.0),
            (1, 2, 1.0),
            (2, 2, -1.0),
        ];
        let kkt = sparse::from_triplets(3, 3, triplets);

        let mut backend = FaerLdlBackend::new(3, 1e-10, 1e-6);
        backend.symbolic_factorization(&kkt).unwrap();
        let factor = backend.numeric_factorization(&kkt).unwrap();

        let rhs = vec![1.0, 2.0, 1.0];
        let mut sol = vec![0.0; 3];

        backend.solve(&factor, &rhs, &mut sol);

        // Verify solution by computing residual
        // Note: the matrix is symmetric so we use full form for checking
        let full_triplets = vec![
            (0, 0, 1.0),
            (1, 1, 1.0),
            (0, 2, 1.0),
            (2, 0, 1.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (2, 2, -1.0),
        ];
        let full_kkt = sparse::from_triplets(3, 3, full_triplets);

        let mut kx = vec![0.0; 3];
        for (val, (row, col)) in full_kkt.iter() {
            kx[row] += val * sol[col];
        }

        let mut residual = 0.0;
        for i in 0..3 {
            residual += (kx[i] - rhs[i]).powi(2);
        }
        residual = residual.sqrt();

        assert!(residual < 1e-6, "residual = {} too large", residual);
    }
}
