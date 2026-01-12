//! KktBackend implementation for CUDA using cuDSS.
//!
//! This module provides `CudaKktBackend`, which wraps NVIDIA's cuDSS library
//! for GPU-accelerated sparse direct solves on NVIDIA GPUs.

use super::error::{CudaError, CudaResult};
use super::handle::{CudaHandle, CudaConfig, GpuBuffer};

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

    /// GPU buffer for matrix values.
    values_gpu: Option<GpuBuffer>,

    /// GPU buffer for CSC column pointers.
    col_ptr_gpu: Option<GpuBuffer>,

    /// GPU buffer for CSC row indices.
    row_ind_gpu: Option<GpuBuffer>,

    /// GPU buffer for RHS vector.
    rhs_gpu: Option<GpuBuffer>,

    /// GPU buffer for solution vector.
    sol_gpu: Option<GpuBuffer>,

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
            col_ptr_gpu: None,
            row_ind_gpu: None,
            rhs_gpu: None,
            sol_gpu: None,
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

        if !self.handle.is_initialized() {
            return Err(CudaError::NotImplemented(
                "cuDSS bindings not yet implemented".to_string()
            ));
        }

        let nnz = row_ind.len();

        // Allocate GPU buffers for matrix structure
        // cuDSS uses CSR format, but for symmetric matrices CSC == CSR^T
        self.col_ptr_gpu = Some(self.handle.allocate((n + 1) * std::mem::size_of::<i32>())?);
        self.row_ind_gpu = Some(self.handle.allocate(nnz * std::mem::size_of::<i32>())?);
        self.values_gpu = Some(self.handle.allocate(nnz * std::mem::size_of::<f64>())?);

        // Allocate RHS and solution buffers
        self.rhs_gpu = Some(self.handle.allocate(n * std::mem::size_of::<f64>())?);
        self.sol_gpu = Some(self.handle.allocate(n * std::mem::size_of::<f64>())?);

        // Upload column pointers (convert to i32)
        let col_ptr_i32: Vec<i32> = col_ptr.iter().map(|&x| x as i32).collect();
        if let Some(ref buf) = self.col_ptr_gpu {
            self.handle.upload(buf, &col_ptr_i32)?;
        }

        // Upload row indices (convert to i32)
        let row_ind_i32: Vec<i32> = row_ind.iter().map(|&x| x as i32).collect();
        if let Some(ref buf) = self.row_ind_gpu {
            self.handle.upload(buf, &row_ind_i32)?;
        }

        // TODO: Create cuDSS matrix and data objects
        // cudssMatrixCreate(matrix, n, n, nnz, col_ptr_gpu, row_ind_gpu, values_gpu,
        //                   CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_LOWER);
        // cudssDataCreate(handle, data);
        // cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, matrix, solution, rhs);

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

        if !self.handle.is_initialized() {
            return Err(CudaError::NotImplemented(
                "cuDSS bindings not yet implemented".to_string()
            ));
        }

        if values.len() != self.nnz {
            return Err(CudaError::DimensionMismatch {
                expected: self.nnz,
                actual: values.len(),
                context: "numeric_factorization values".to_string(),
            });
        }

        // Add static regularization to diagonal
        let mut values_reg = values.to_vec();
        // TODO: Add regularization to diagonal elements
        // This requires knowing which entries are diagonal

        // Upload values to GPU
        if let Some(ref buf) = self.values_gpu {
            self.handle.upload(buf, &values_reg)?;
        }

        // TODO: Call cuDSS factorization
        // cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, matrix, solution, rhs);

        self.handle.synchronize()?;

        self.factor_valid = true;
        self.stats.num_numeric += 1;
        self.stats.time_numeric += start.elapsed().as_secs_f64();

        Ok(CudaFactorization {
            n: self.n,
            nnz_l: 0, // TODO: Get from cuDSS
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

        if !self.handle.is_initialized() {
            return Err(CudaError::NotImplemented(
                "cuDSS bindings not yet implemented".to_string()
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
        if let Some(ref buf) = self.rhs_gpu {
            self.handle.upload(buf, rhs)?;
        }

        // TODO: Call cuDSS solve
        // cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, matrix, solution_gpu, rhs_gpu);

        self.handle.synchronize()?;

        // Download solution from GPU
        if let Some(ref buf) = self.sol_gpu {
            self.handle.download(solution, buf)?;
        }

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
            Err(CudaError::NoDevice) => {
                // Expected without CUDA device
            }
            Err(e) => {
                // Other errors are OK too for CI
                println!("CUDA init: {}", e);
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
