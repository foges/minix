//! KktBackend implementation for Metal.
//!
//! This module provides `MetalKktBackend`, which implements the `KktBackend` trait
//! for GPU-accelerated KKT system solves on Apple Silicon.

use super::error::{MetalError, MetalResult};
use super::handle::{DssHandle, DssConfig};
use super::symbolic::SymbolicAnalysis;

/// Metal-accelerated KKT solver backend.
///
/// This backend uses Metal compute shaders for sparse direct solves.
/// It implements a cuDSS-like three-phase workflow:
///
/// 1. **Analysis**: Symbolic factorization (ordering, etree, supernodes)
/// 2. **Factorization**: Numeric LDL^T factorization
/// 3. **Solve**: Triangular solves
#[cfg(target_os = "macos")]
pub struct MetalKktBackend {
    /// Metal handle (device, queue, pipelines).
    handle: DssHandle,

    /// Symbolic analysis result (set after symbolic_factorization).
    symbolic: Option<SymbolicAnalysis>,

    /// GPU buffers for factor storage.
    factor_buffers: Option<FactorBuffers>,

    /// GPU buffers for solve workspace.
    solve_buffers: Option<SolveBuffers>,

    /// Statistics.
    stats: SolverStats,
}

/// GPU buffers for storing the LDL^T factorization.
#[cfg(target_os = "macos")]
struct FactorBuffers {
    /// L factor values (CSR or dense supernodal panels).
    l_values: metal::Buffer,

    /// D diagonal values.
    d_values: metal::Buffer,

    /// L row indices (for CSR storage).
    l_row_ind: metal::Buffer,

    /// L column pointers (for CSR storage).
    l_col_ptr: metal::Buffer,

    /// Permutation vector.
    perm: metal::Buffer,

    /// Inverse permutation vector.
    perm_inv: metal::Buffer,

    /// Inverse diagonal (1/D) for solve phase.
    d_inv: metal::Buffer,
}

/// GPU buffers for solve workspace.
#[cfg(target_os = "macos")]
struct SolveBuffers {
    /// Temporary vector for permuted RHS.
    temp_rhs: metal::Buffer,

    /// Temporary vector for intermediate results.
    temp_sol: metal::Buffer,

    /// Level schedule: pointers to level starts.
    level_ptr: metal::Buffer,

    /// Level schedule: row indices per level.
    level_rows_lower: metal::Buffer,

    /// Level schedule: row indices per level (upper solve).
    level_rows_upper: metal::Buffer,

    /// Number of levels.
    num_levels: usize,

    /// Level sizes.
    level_sizes_lower: Vec<usize>,
    level_sizes_upper: Vec<usize>,
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
pub struct MetalFactorization {
    /// Matrix dimension.
    pub n: usize,

    /// Number of nonzeros in L.
    pub nnz_l: usize,

    /// Whether factorization succeeded.
    pub success: bool,

    /// Minimum pivot encountered.
    pub min_pivot: f64,

    /// Maximum pivot encountered.
    pub max_pivot: f64,
}

#[cfg(target_os = "macos")]
impl MetalKktBackend {
    /// Create a new Metal KKT backend with the given configuration.
    pub fn new(config: DssConfig) -> MetalResult<Self> {
        let handle = DssHandle::new(config)?;

        Ok(Self {
            handle,
            symbolic: None,
            factor_buffers: None,
            solve_buffers: None,
            stats: SolverStats::default(),
        })
    }

    /// Create a new Metal KKT backend with default configuration.
    pub fn with_defaults() -> MetalResult<Self> {
        Self::new(DssConfig::default())
    }

    /// Get solver statistics.
    pub fn stats(&self) -> &SolverStats {
        &self.stats
    }

    /// Get the symbolic analysis result.
    pub fn symbolic(&self) -> Option<&SymbolicAnalysis> {
        self.symbolic.as_ref()
    }

    /// Perform symbolic analysis on a sparse matrix.
    ///
    /// This analyzes the sparsity pattern and prepares for numeric factorization.
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
    ) -> MetalResult<()> {
        let start = std::time::Instant::now();

        // Run symbolic analysis on CPU
        let symbolic = SymbolicAnalysis::analyze(
            n,
            col_ptr,
            row_ind,
            self.handle.config().ordering,
        )?;

        if self.handle.config().verbose {
            println!(
                "[Metal] Symbolic analysis: n={}, nnz_l={}, supernodes={}",
                symbolic.n,
                symbolic.nnz_l,
                symbolic.supernodes.len()
            );
        }

        // Allocate GPU buffers for factor storage
        self.allocate_factor_buffers(&symbolic)?;

        // Allocate GPU buffers for solve workspace
        self.allocate_solve_buffers(&symbolic)?;

        self.symbolic = Some(symbolic);
        self.stats.num_symbolic += 1;
        self.stats.time_symbolic += start.elapsed().as_secs_f64();

        Ok(())
    }

    /// Perform numeric factorization.
    ///
    /// Call this when matrix values change but sparsity pattern is the same.
    /// Must call `symbolic_analysis` first.
    ///
    /// # Arguments
    /// * `col_ptr` - CSC column pointers
    /// * `row_ind` - CSC row indices
    /// * `values` - Matrix values
    pub fn numeric_factorization(
        &mut self,
        col_ptr: &[usize],
        row_ind: &[usize],
        values: &[f64],
    ) -> MetalResult<MetalFactorization> {
        let start = std::time::Instant::now();

        let symbolic = self.symbolic.as_ref()
            .ok_or(MetalError::NumericFactorization(
                "Must call symbolic_analysis first".to_string()
            ))?;

        let n = symbolic.n;

        // For now, implement a simple CPU factorization and upload to GPU
        // TODO: Implement GPU supernodal factorization

        // Convert to f32 for GPU
        let values_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();

        // Permute and factorize on CPU (placeholder)
        let (l_values, d_values) = self.cpu_ldlt_factorize(
            n,
            col_ptr,
            row_ind,
            &values_f32,
            symbolic,
        )?;

        // Upload to GPU
        self.upload_factor_to_gpu(&l_values, &d_values)?;

        let min_pivot = d_values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_pivot = d_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        self.stats.num_numeric += 1;
        self.stats.time_numeric += start.elapsed().as_secs_f64();

        Ok(MetalFactorization {
            n,
            nnz_l: symbolic.nnz_l,
            success: true,
            min_pivot: min_pivot as f64,
            max_pivot: max_pivot as f64,
        })
    }

    /// Solve the system Ax = b using the computed factorization.
    ///
    /// # Arguments
    /// * `rhs` - Right-hand side vector (length n)
    /// * `solution` - Solution vector (length n, modified in place)
    pub fn solve(&mut self, rhs: &[f64], solution: &mut [f64]) -> MetalResult<()> {
        let start = std::time::Instant::now();

        let symbolic = self.symbolic.as_ref()
            .ok_or(MetalError::Solve("No symbolic analysis".to_string()))?;

        let n = symbolic.n;

        if rhs.len() != n || solution.len() != n {
            return Err(MetalError::DimensionMismatch {
                expected: n,
                actual: rhs.len().min(solution.len()),
                context: "solve".to_string(),
            });
        }

        // For now, implement CPU solve (placeholder for GPU solve)
        // TODO: Implement GPU level-scheduled triangular solve

        // Convert to f32
        let rhs_f32: Vec<f32> = rhs.iter().map(|&v| v as f32).collect();
        let mut sol_f32 = vec![0.0f32; n];

        self.cpu_solve(&rhs_f32, &mut sol_f32, symbolic)?;

        // Convert back to f64
        for (i, &v) in sol_f32.iter().enumerate() {
            solution[i] = v as f64;
        }

        self.stats.num_solves += 1;
        self.stats.time_solve += start.elapsed().as_secs_f64();

        Ok(())
    }

    // ========================================================================
    // Private helper methods
    // ========================================================================

    fn allocate_factor_buffers(&mut self, symbolic: &SymbolicAnalysis) -> MetalResult<()> {
        let n = symbolic.n;
        let nnz_l = symbolic.nnz_l;

        let l_values = self.handle.create_buffer(nnz_l * std::mem::size_of::<f32>())?;
        let d_values = self.handle.create_buffer(n * std::mem::size_of::<f32>())?;
        let d_inv = self.handle.create_buffer(n * std::mem::size_of::<f32>())?;
        let l_row_ind = self.handle.create_buffer(nnz_l * std::mem::size_of::<u32>())?;
        let l_col_ptr = self.handle.create_buffer((n + 1) * std::mem::size_of::<u32>())?;
        let perm = self.handle.create_buffer_with_data(
            &symbolic.perm.iter().map(|&p| p as u32).collect::<Vec<_>>()
        )?;
        let perm_inv = self.handle.create_buffer_with_data(
            &symbolic.perm_inv.iter().map(|&p| p as u32).collect::<Vec<_>>()
        )?;

        self.factor_buffers = Some(FactorBuffers {
            l_values,
            d_values,
            l_row_ind,
            l_col_ptr,
            perm,
            perm_inv,
            d_inv,
        });

        Ok(())
    }

    fn allocate_solve_buffers(&mut self, symbolic: &SymbolicAnalysis) -> MetalResult<()> {
        let n = symbolic.n;

        let temp_rhs = self.handle.create_buffer(n * std::mem::size_of::<f32>())?;
        let temp_sol = self.handle.create_buffer(n * std::mem::size_of::<f32>())?;

        // Build level schedule for triangular solves
        // For now, use simple row-by-row schedule (no parallelism)
        // TODO: Compute proper level schedule from etree

        let level_ptr = self.handle.create_buffer((n + 1) * std::mem::size_of::<u32>())?;
        let level_rows_lower = self.handle.create_buffer(n * std::mem::size_of::<u32>())?;
        let level_rows_upper = self.handle.create_buffer(n * std::mem::size_of::<u32>())?;

        self.solve_buffers = Some(SolveBuffers {
            temp_rhs,
            temp_sol,
            level_ptr,
            level_rows_lower,
            level_rows_upper,
            num_levels: n, // One level per row (no parallelism yet)
            level_sizes_lower: vec![1; n],
            level_sizes_upper: vec![1; n],
        });

        Ok(())
    }

    fn cpu_ldlt_factorize(
        &self,
        n: usize,
        col_ptr: &[usize],
        row_ind: &[usize],
        values: &[f32],
        symbolic: &SymbolicAnalysis,
    ) -> MetalResult<(Vec<f32>, Vec<f32>)> {
        // Simple dense LDL^T factorization for now (placeholder)
        // TODO: Implement proper sparse supernodal factorization

        let pivot_min = self.handle.config().pivot_min;

        // Build dense matrix (only for small n, for testing)
        if n > 1000 {
            return Err(MetalError::NotImplemented(
                "Sparse factorization not yet implemented; matrix too large for dense fallback".to_string()
            ));
        }

        let mut a = vec![0.0f32; n * n];

        // Fill dense matrix from sparse
        for j in 0..n {
            let pj = symbolic.perm_inv[j]; // Original column for permuted column j
            for p in col_ptr[pj]..col_ptr[pj + 1] {
                let pi = row_ind[p];
                let i = symbolic.perm[pi]; // Permuted row
                let v = values[p];

                a[i * n + j] = v;
                a[j * n + i] = v; // Symmetric
            }
        }

        // Dense LDL^T factorization (in-place on lower triangle)
        let mut d = vec![0.0f32; n];

        for k in 0..n {
            // Compute D[k]
            let mut dkk = a[k * n + k];
            for j in 0..k {
                let ljk = a[k * n + j];
                dkk -= d[j] * ljk * ljk;
            }

            // Clamp pivot
            if dkk.abs() < pivot_min {
                dkk = if dkk >= 0.0 { pivot_min } else { -pivot_min };
            }
            d[k] = dkk;

            // Compute L[i,k] for i > k
            for i in (k + 1)..n {
                let mut lik = a[i * n + k];
                for j in 0..k {
                    lik -= d[j] * a[i * n + j] * a[k * n + j];
                }
                a[i * n + k] = lik / dkk;
            }
        }

        // Extract L values (lower triangle, column by column)
        let mut l_values = Vec::with_capacity(symbolic.nnz_l);
        for j in 0..n {
            for i in j..n {
                if i == j {
                    l_values.push(1.0f32); // Unit diagonal
                } else {
                    l_values.push(a[i * n + j]);
                }
            }
        }

        Ok((l_values, d))
    }

    fn upload_factor_to_gpu(&mut self, l_values: &[f32], d_values: &[f32]) -> MetalResult<()> {
        let buffers = self.factor_buffers.as_mut()
            .ok_or(MetalError::NumericFactorization("No factor buffers".to_string()))?;

        // Upload L values
        let l_ptr = buffers.l_values.contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(l_values.as_ptr(), l_ptr, l_values.len());
        }

        // Upload D values
        let d_ptr = buffers.d_values.contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(d_values.as_ptr(), d_ptr, d_values.len());
        }

        // Compute and upload D^{-1}
        let d_inv: Vec<f32> = d_values.iter().map(|&d| 1.0 / d).collect();
        let d_inv_ptr = buffers.d_inv.contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(d_inv.as_ptr(), d_inv_ptr, d_inv.len());
        }

        Ok(())
    }

    fn cpu_solve(
        &self,
        rhs: &[f32],
        solution: &mut [f32],
        symbolic: &SymbolicAnalysis,
    ) -> MetalResult<()> {
        let n = symbolic.n;

        // Get factor from GPU (for now, we keep a CPU copy implicitly through dense factorization)
        // TODO: Proper GPU solve

        let buffers = self.factor_buffers.as_ref()
            .ok_or(MetalError::Solve("No factor buffers".to_string()))?;

        // Read L and D from GPU buffers
        let l_ptr = buffers.l_values.contents() as *const f32;
        let d_inv_ptr = buffers.d_inv.contents() as *const f32;

        // Permute RHS: y = P * b
        let mut y = vec![0.0f32; n];
        for i in 0..n {
            y[i] = rhs[symbolic.perm_inv[i]];
        }

        // Forward solve: L * z = y
        // With dense L stored column-wise (lower triangle)
        let mut z = y.clone();
        let mut l_idx = 0;
        for j in 0..n {
            // L[j,j] = 1 (unit diagonal)
            l_idx += 1;

            let zj = z[j];
            for i in (j + 1)..n {
                let lij = unsafe { *l_ptr.add(l_idx) };
                z[i] -= lij * zj;
                l_idx += 1;
            }
        }

        // Diagonal solve: D^{-1} * z
        for i in 0..n {
            let d_inv_i = unsafe { *d_inv_ptr.add(i) };
            z[i] *= d_inv_i;
        }

        // Backward solve: L^T * x = z
        // Process columns in reverse
        let mut x = z;
        for j in (0..n).rev() {
            // Compute L column start index
            let col_start: usize = (0..j).map(|k| n - k).sum();

            for i in (j + 1)..n {
                let lij = unsafe { *l_ptr.add(col_start + 1 + (i - j - 1)) };
                x[j] -= lij * x[i];
            }
            // L[j,j] = 1, so no division needed
        }

        // Inverse permute: solution = P^T * x
        for i in 0..n {
            solution[symbolic.perm[i]] = x[i];
        }

        Ok(())
    }
}

// Stub for non-macOS
#[cfg(not(target_os = "macos"))]
pub struct MetalKktBackend;

#[cfg(not(target_os = "macos"))]
impl MetalKktBackend {
    pub fn new(_config: DssConfig) -> MetalResult<Self> {
        Err(MetalError::NoDevice)
    }

    pub fn with_defaults() -> MetalResult<Self> {
        Err(MetalError::NoDevice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "macos")]
    fn test_metal_backend_creation() {
        let result = MetalKktBackend::with_defaults();
        // May fail if no Metal device, that's OK for CI
        if let Ok(backend) = result {
            assert!(backend.symbolic.is_none());
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_simple_solve() {
        let backend = match MetalKktBackend::with_defaults() {
            Ok(b) => b,
            Err(_) => return, // Skip if no Metal device
        };

        // 2x2 SPD matrix:
        // [4 1]
        // [1 3]
        let n = 2;
        let col_ptr = vec![0, 2, 4];
        let row_ind = vec![0, 1, 0, 1];
        let values: Vec<f64> = vec![4.0, 1.0, 1.0, 3.0];

        let mut backend = backend;

        // Symbolic analysis
        backend.symbolic_analysis(n, &col_ptr, &row_ind).unwrap();

        // Numeric factorization
        let _factor = backend.numeric_factorization(&col_ptr, &row_ind, &values).unwrap();

        // Solve Ax = b where b = [5, 4] (solution should be x = [1, 1])
        let rhs = vec![5.0, 4.0];
        let mut solution = vec![0.0; 2];
        backend.solve(&rhs, &mut solution).unwrap();

        // Check solution (with tolerance for f32 precision)
        assert!((solution[0] - 1.0).abs() < 1e-5, "x[0] = {}", solution[0]);
        assert!((solution[1] - 1.0).abs() < 1e-5, "x[1] = {}", solution[1]);
    }
}
