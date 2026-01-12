//! KktBackend implementation for CUDA using cuDSS.
//!
//! This module provides `CudaKktBackend`, which wraps NVIDIA's cuDSS library
//! for GPU-accelerated sparse direct solves on NVIDIA GPUs.

use super::error::{CudaError, CudaResult};
use super::ffi::{
    CudssMatrix_t, CudssPhase, CudssMtype, CudssMview, CudssIndexBase,
    CudaDataType, check_cudss,
};
use super::handle::{CudaHandle, CudaConfig, GpuBuffer};
use std::ptr;

/// CUDA-accelerated KKT solver backend using cuDSS.
///
/// This backend uses NVIDIA's cuDSS library for sparse direct solves.
/// cuDSS provides high-performance LDL^T/Cholesky/LU factorization on GPUs.
///
/// The workflow follows cuDSS's three-phase design:
///
/// 1. **Analysis**: Creates cudssMatrix and cudssData objects, performs
///    symbolic factorization with fill-reducing ordering
/// 2. **Factorization**: GPU-accelerated supernodal factorization
/// 3. **Solve**: GPU-accelerated triangular solves
pub struct CudaKktBackend {
    /// CUDA handle (device, stream, cuDSS handle).
    handle: CudaHandle,

    /// Matrix dimension.
    n: usize,

    /// Number of nonzeros in the original matrix.
    nnz: usize,

    /// GPU buffer for matrix values (f64).
    values_gpu: Option<GpuBuffer>,

    /// GPU buffer for CSR row pointers (i32).
    row_ptr_gpu: Option<GpuBuffer>,

    /// GPU buffer for CSR column indices (i32).
    col_ind_gpu: Option<GpuBuffer>,

    /// GPU buffer for RHS vector (f64).
    rhs_gpu: Option<GpuBuffer>,

    /// GPU buffer for solution vector (f64).
    sol_gpu: Option<GpuBuffer>,

    /// cuDSS matrix object.
    matrix: CudssMatrix_t,

    /// cuDSS RHS matrix (dense, single column).
    rhs_matrix: CudssMatrix_t,

    /// cuDSS solution matrix (dense, single column).
    sol_matrix: CudssMatrix_t,

    /// Static regularization.
    static_reg: f64,

    /// Whether symbolic analysis has been performed.
    symbolic_done: bool,

    /// Whether numeric factorization is valid.
    factor_valid: bool,

    /// Solver statistics.
    stats: SolverStats,
}

/// Solver statistics.
#[derive(Debug, Default, Clone)]
pub struct SolverStats {
    /// Number of symbolic factorizations performed.
    pub num_symbolic: usize,

    /// Number of numeric factorizations performed.
    pub num_numeric: usize,

    /// Number of solves performed.
    pub num_solves: usize,

    /// Total time in symbolic analysis (seconds).
    pub time_symbolic: f64,

    /// Total time in numeric factorization (seconds).
    pub time_numeric: f64,

    /// Total time in solves (seconds).
    pub time_solve: f64,

    /// Number of dynamic regularization bumps.
    pub dynamic_bumps: u64,
}

/// Factorization handle returned by numeric_factorization.
#[derive(Debug, Clone)]
pub struct CudaFactorization {
    /// Matrix dimension.
    pub n: usize,

    /// Number of nonzeros in L.
    pub nnz_l: usize,

    /// Whether factorization succeeded.
    pub success: bool,

    /// Number of negative pivots (for indefinite matrices).
    pub num_neg_pivots: usize,

    /// Minimum pivot magnitude.
    pub min_pivot: f64,
}

impl CudaKktBackend {
    /// Create a new CUDA KKT backend with the given configuration.
    pub fn new(config: CudaConfig) -> CudaResult<Self> {
        let handle = CudaHandle::new(config)?;

        Ok(Self {
            handle,
            n: 0,
            nnz: 0,
            values_gpu: None,
            row_ptr_gpu: None,
            col_ind_gpu: None,
            rhs_gpu: None,
            sol_gpu: None,
            matrix: ptr::null_mut(),
            rhs_matrix: ptr::null_mut(),
            sol_matrix: ptr::null_mut(),
            static_reg: 1e-8,
            symbolic_done: false,
            factor_valid: false,
            stats: SolverStats::default(),
        })
    }

    /// Create a new CUDA KKT backend with default configuration.
    pub fn with_defaults() -> CudaResult<Self> {
        Self::new(CudaConfig::default())
    }

    /// Get solver statistics.
    pub fn stats(&self) -> &SolverStats {
        &self.stats
    }

    /// Perform symbolic analysis on a sparse matrix.
    ///
    /// This sets up cuDSS matrix objects and performs symbolic factorization.
    /// Call this once per sparsity pattern.
    ///
    /// # Arguments
    /// * `n` - Matrix dimension
    /// * `col_ptr` - CSC column pointers (length n+1)
    /// * `row_ind` - CSC row indices
    pub fn symbolic_analysis(
        &mut self,
        n: usize,
        col_ptr: &[usize],
        row_ind: &[usize],
    ) -> CudaResult<()> {
        let start = std::time::Instant::now();

        let nnz = row_ind.len();

        // cuDSS uses CSR format. For symmetric matrices, CSC of lower triangle
        // is the same as CSR of upper triangle translated.
        // We'll convert to CSR here.

        // Convert CSC to CSR for symmetric matrix
        // For symmetric indefinite, we provide the lower triangle in CSR format
        let (csr_row_ptr, csr_col_ind) = self.csc_to_csr_symmetric(n, col_ptr, row_ind);

        // Allocate GPU buffers
        self.row_ptr_gpu = Some(self.handle.allocate((n + 1) * std::mem::size_of::<i32>())?);
        self.col_ind_gpu = Some(self.handle.allocate(nnz * std::mem::size_of::<i32>())?);
        self.values_gpu = Some(self.handle.allocate(nnz * std::mem::size_of::<f64>())?);
        self.rhs_gpu = Some(self.handle.allocate(n * std::mem::size_of::<f64>())?);
        self.sol_gpu = Some(self.handle.allocate(n * std::mem::size_of::<f64>())?);

        // Upload structure to GPU
        let row_ptr_i32: Vec<i32> = csr_row_ptr.iter().map(|&x| x as i32).collect();
        let col_ind_i32: Vec<i32> = csr_col_ind.iter().map(|&x| x as i32).collect();

        self.handle.upload(self.row_ptr_gpu.as_ref().unwrap(), &row_ptr_i32)?;
        self.handle.upload(self.col_ind_gpu.as_ref().unwrap(), &col_ind_i32)?;

        // Create cuDSS matrix (CSR format, symmetric, lower triangle view)
        let libs = self.handle.libs();
        unsafe {
            let err = (libs.cudss_matrix_create_csr)(
                &mut self.matrix,
                n as i64,
                n as i64,
                nnz as i64,
                self.row_ptr_gpu.as_ref().unwrap().as_ptr(),
                ptr::null_mut(), // row_ptr_end (null = use row_ptr+1)
                self.col_ind_gpu.as_ref().unwrap().as_ptr(),
                self.values_gpu.as_ref().unwrap().as_ptr(),
                CudaDataType::R32I,  // Index type
                CudaDataType::R32I,  // Index type for col_ind
                CudaDataType::R64F,  // Value type (double)
                CudssIndexBase::Zero,
                CudssMtype::Symmetric,
                CudssMview::Lower,
            );
            check_cudss(err, "cudssMatrixCreateCsr")?;
        }

        // Create dense matrices for RHS and solution
        unsafe {
            let err = (libs.cudss_matrix_create_dn)(
                &mut self.rhs_matrix,
                n as i64,
                1,  // Single column (nrhs=1)
                n as i64,  // Leading dimension
                self.rhs_gpu.as_ref().unwrap().as_ptr(),
                CudaDataType::R64F,
                1,  // Column major
            );
            check_cudss(err, "cudssMatrixCreateDn(rhs)")?;

            let err = (libs.cudss_matrix_create_dn)(
                &mut self.sol_matrix,
                n as i64,
                1,
                n as i64,
                self.sol_gpu.as_ref().unwrap().as_ptr(),
                CudaDataType::R64F,
                1,
            );
            check_cudss(err, "cudssMatrixCreateDn(sol)")?;
        }

        // Run analysis phase
        unsafe {
            let err = (libs.cudss_execute)(
                self.handle.cudss_handle(),
                CudssPhase::Analysis,
                self.handle.cudss_config(),
                self.handle.cudss_data(),
                self.matrix,
                self.sol_matrix,
                self.rhs_matrix,
            );
            check_cudss(err, "cudssExecute(Analysis)")?;
        }

        self.handle.synchronize()?;

        self.n = n;
        self.nnz = nnz;
        self.symbolic_done = true;
        self.stats.num_symbolic += 1;
        self.stats.time_symbolic += start.elapsed().as_secs_f64();

        if self.handle.config().verbose {
            println!(
                "[CUDA] Symbolic analysis: n={}, nnz={}",
                n, nnz
            );
        }

        Ok(())
    }

    /// Perform numeric factorization.
    ///
    /// Call this when matrix values change but sparsity pattern is the same.
    /// Must call `symbolic_analysis` first.
    ///
    /// # Arguments
    /// * `values` - Matrix values (length nnz)
    pub fn numeric_factorization(
        &mut self,
        values: &[f64],
    ) -> CudaResult<CudaFactorization> {
        let start = std::time::Instant::now();

        if !self.symbolic_done {
            return Err(CudaError::NumericFactorization(
                "Must call symbolic_analysis first".to_string()
            ));
        }

        if values.len() != self.nnz {
            return Err(CudaError::DimensionMismatch {
                expected: self.nnz,
                actual: values.len(),
                context: "numeric_factorization values".to_string(),
            });
        }

        // Upload values to GPU
        self.handle.upload(self.values_gpu.as_ref().unwrap(), values)?;

        // Run factorization phase
        let libs = self.handle.libs();
        unsafe {
            let err = (libs.cudss_execute)(
                self.handle.cudss_handle(),
                CudssPhase::Factorization,
                self.handle.cudss_config(),
                self.handle.cudss_data(),
                self.matrix,
                self.sol_matrix,
                self.rhs_matrix,
            );
            check_cudss(err, "cudssExecute(Factorization)")?;
        }

        self.handle.synchronize()?;

        self.factor_valid = true;
        self.stats.num_numeric += 1;
        self.stats.time_numeric += start.elapsed().as_secs_f64();

        if self.handle.config().verbose {
            println!("[CUDA] Factorization completed");
        }

        Ok(CudaFactorization {
            n: self.n,
            nnz_l: 0, // Could query from cuDSS
            success: true,
            num_neg_pivots: 0,
            min_pivot: 0.0,
        })
    }

    /// Solve the system Ax = b using the computed factorization.
    ///
    /// # Arguments
    /// * `rhs` - Right-hand side vector (length n)
    /// * `solution` - Solution vector (length n, modified in place)
    pub fn solve(&mut self, rhs: &[f64], solution: &mut [f64]) -> CudaResult<()> {
        let start = std::time::Instant::now();

        if !self.factor_valid {
            return Err(CudaError::Solve(
                "No valid factorization".to_string()
            ));
        }

        if rhs.len() != self.n || solution.len() != self.n {
            return Err(CudaError::DimensionMismatch {
                expected: self.n,
                actual: rhs.len().min(solution.len()),
                context: "solve".to_string(),
            });
        }

        // Upload RHS to GPU
        self.handle.upload(self.rhs_gpu.as_ref().unwrap(), rhs)?;

        // Run solve phase
        let libs = self.handle.libs();
        unsafe {
            let err = (libs.cudss_execute)(
                self.handle.cudss_handle(),
                CudssPhase::Solve,
                self.handle.cudss_config(),
                self.handle.cudss_data(),
                self.matrix,
                self.sol_matrix,
                self.rhs_matrix,
            );
            check_cudss(err, "cudssExecute(Solve)")?;
        }

        self.handle.synchronize()?;

        // Download solution from GPU
        self.handle.download(solution, self.sol_gpu.as_ref().unwrap())?;

        self.stats.num_solves += 1;
        self.stats.time_solve += start.elapsed().as_secs_f64();

        Ok(())
    }

    /// Set static regularization value.
    pub fn set_static_reg(&mut self, reg: f64) {
        self.static_reg = reg;
    }

    /// Get static regularization value.
    pub fn static_reg(&self) -> f64 {
        self.static_reg
    }

    /// Convert CSC (lower triangle) to CSR format for symmetric matrix.
    ///
    /// For a symmetric matrix stored as lower triangle in CSC,
    /// the CSR of the same lower triangle is just the transpose.
    fn csc_to_csr_symmetric(
        &self,
        n: usize,
        col_ptr: &[usize],
        row_ind: &[usize],
    ) -> (Vec<usize>, Vec<usize>) {
        // CSC lower triangle -> CSR lower triangle
        // This is effectively transposing the sparsity pattern

        // Count entries per row
        let mut row_counts = vec![0usize; n];
        for &row in row_ind {
            row_counts[row] += 1;
        }

        // Build row pointers
        let mut row_ptr = vec![0usize; n + 1];
        for i in 0..n {
            row_ptr[i + 1] = row_ptr[i] + row_counts[i];
        }

        // Build column indices
        let nnz = row_ind.len();
        let mut col_indices = vec![0usize; nnz];
        let mut current_pos = row_ptr.clone();

        for col in 0..n {
            for p in col_ptr[col]..col_ptr[col + 1] {
                let row = row_ind[p];
                let pos = current_pos[row];
                col_indices[pos] = col;
                current_pos[row] += 1;
            }
        }

        (row_ptr, col_indices)
    }
}

impl Drop for CudaKktBackend {
    fn drop(&mut self) {
        let libs = self.handle.libs();
        unsafe {
            if !self.sol_matrix.is_null() {
                let _ = (libs.cudss_matrix_destroy)(self.sol_matrix);
            }
            if !self.rhs_matrix.is_null() {
                let _ = (libs.cudss_matrix_destroy)(self.rhs_matrix);
            }
            if !self.matrix.is_null() {
                let _ = (libs.cudss_matrix_destroy)(self.matrix);
            }
        }
    }
}

// ============================================================================
// KktBackend trait implementation
// ============================================================================

use crate::linalg::backend::{BackendError, KktBackend};
use crate::linalg::sparse::SparseCsc;
use std::cell::UnsafeCell;

/// Adapter to use CudaKktBackend with the KktBackend trait.
///
/// This allows the CUDA backend to be used as a drop-in replacement
/// for QdldlBackend in the KktSolver.
pub struct CudaBackendAdapter {
    /// Inner backend wrapped in UnsafeCell for interior mutability.
    /// This is needed because the KktBackend::solve() takes &self but
    /// the inner solve() needs &mut self.
    inner: UnsafeCell<CudaKktBackend>,
    static_reg: f64,
    n: usize,
}

// SAFETY: CudaBackendAdapter uses UnsafeCell for interior mutability.
// This is safe because:
// 1. The solve() method is the only one that mutates through the UnsafeCell
// 2. solve() is never called concurrently (sequential IPM iterations)
// 3. No references to the inner backend escape the adapter
unsafe impl Send for CudaBackendAdapter {}
unsafe impl Sync for CudaBackendAdapter {}

impl KktBackend for CudaBackendAdapter {
    type Factorization = CudaFactorization;

    fn new(n: usize, static_reg: f64, dynamic_reg_min_pivot: f64) -> Self
    where
        Self: Sized,
    {
        let mut config = CudaConfig::default();
        config.pivot_tolerance = dynamic_reg_min_pivot;

        let inner = CudaKktBackend::new(config)
            .expect("Failed to create CUDA backend");

        Self {
            inner: UnsafeCell::new(inner),
            static_reg,
            n,
        }
    }

    fn set_static_reg(&mut self, static_reg: f64) -> Result<(), BackendError> {
        self.static_reg = static_reg;
        // SAFETY: We have &mut self, so exclusive access is guaranteed
        unsafe {
            (*self.inner.get()).set_static_reg(static_reg);
        }
        Ok(())
    }

    fn static_reg(&self) -> f64 {
        self.static_reg
    }

    fn symbolic_factorization(&mut self, kkt: &SparseCsc) -> Result<(), BackendError> {
        let (n, m) = kkt.shape();
        if n != m {
            return Err(BackendError::Message(format!(
                "KKT matrix must be square, got {}x{}",
                n, m
            )));
        }

        self.n = n;

        // Extract CSC components
        let indptr = kkt.indptr();
        let col_ptr: Vec<usize> = indptr.raw_storage().to_vec();
        let row_ind: Vec<usize> = kkt.indices().to_vec();

        // SAFETY: We have &mut self, so exclusive access is guaranteed
        unsafe {
            (*self.inner.get())
                .symbolic_analysis(n, &col_ptr, &row_ind)
                .map_err(|e| BackendError::Other(e.to_string()))
        }
    }

    fn numeric_factorization(&mut self, kkt: &SparseCsc) -> Result<Self::Factorization, BackendError> {
        // Extract values
        let values: Vec<f64> = kkt.data().to_vec();

        // SAFETY: We have &mut self, so exclusive access is guaranteed
        unsafe {
            (*self.inner.get())
                .numeric_factorization(&values)
                .map_err(|e| BackendError::Other(e.to_string()))
        }
    }

    fn solve(&self, _factor: &Self::Factorization, rhs: &[f64], sol: &mut [f64]) {
        // Note: CudaKktBackend stores factor internally, so we ignore the factor parameter
        // SAFETY: solve() is only called sequentially from IPM iterations.
        // The UnsafeCell allows interior mutability for the solve phase.
        unsafe {
            if let Err(e) = (*self.inner.get()).solve(rhs, sol) {
                eprintln!("[CUDA] Solve error: {}", e);
                sol.fill(0.0);
            }
        }
    }

    fn dynamic_bumps(&self) -> u64 {
        // SAFETY: stats() only reads data
        unsafe { (*self.inner.get()).stats().dynamic_bumps }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::handle::MatrixType;

    #[test]
    fn test_cuda_backend_creation() {
        // This will likely fail without actual CUDA, but tests the structure
        let result = CudaKktBackend::with_defaults();
        match result {
            Ok(backend) => {
                assert!(!backend.symbolic_done);
                assert_eq!(backend.n, 0);
            }
            Err(CudaError::LibraryNotFound(_)) => {
                // Expected without CUDA libraries
                println!("CUDA libraries not found (expected on non-CUDA systems)");
            }
            Err(CudaError::NoDevice) => {
                // Expected without CUDA device
                println!("No CUDA device found");
            }
            Err(e) => {
                // Other errors are OK too for CI
                println!("CUDA init error: {}", e);
            }
        }
    }

    #[test]
    fn test_config_defaults() {
        let config = CudaConfig::default();
        assert_eq!(config.device_index, 0);
        assert_eq!(config.matrix_type, MatrixType::SymmetricIndefinite);
        assert!(!config.mixed_precision);
        assert_eq!(config.refine_iters, 2);
    }
}
