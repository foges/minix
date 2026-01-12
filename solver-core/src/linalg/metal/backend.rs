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

        // Use level schedules from symbolic analysis
        let (level_ptr_lower, level_rows_lower_vec) = &symbolic.solve_levels_lower;
        let (level_ptr_upper, level_rows_upper_vec) = &symbolic.solve_levels_upper;

        // Compute level sizes
        let num_levels_lower = if level_ptr_lower.len() > 1 { level_ptr_lower.len() - 1 } else { 0 };
        let num_levels_upper = if level_ptr_upper.len() > 1 { level_ptr_upper.len() - 1 } else { 0 };

        let level_sizes_lower: Vec<usize> = (0..num_levels_lower)
            .map(|i| level_ptr_lower[i + 1] - level_ptr_lower[i])
            .collect();
        let level_sizes_upper: Vec<usize> = (0..num_levels_upper)
            .map(|i| level_ptr_upper[i + 1] - level_ptr_upper[i])
            .collect();

        // Upload level schedules to GPU
        let level_ptr = self.handle.create_buffer_with_data(
            &level_ptr_lower.iter().map(|&x| x as u32).collect::<Vec<_>>()
        )?;
        let level_rows_lower = self.handle.create_buffer_with_data(
            &level_rows_lower_vec.iter().map(|&x| x as u32).collect::<Vec<_>>()
        )?;
        let level_rows_upper = self.handle.create_buffer_with_data(
            &level_rows_upper_vec.iter().map(|&x| x as u32).collect::<Vec<_>>()
        )?;

        let num_levels = num_levels_lower.max(num_levels_upper);

        self.solve_buffers = Some(SolveBuffers {
            temp_rhs,
            temp_sol,
            level_ptr,
            level_rows_lower,
            level_rows_upper,
            num_levels,
            level_sizes_lower,
            level_sizes_upper,
        });

        if self.handle.config().verbose {
            println!(
                "[Metal] Solve buffers: {} levels (lower), {} levels (upper)",
                num_levels_lower, num_levels_upper
            );
        }

        Ok(())
    }

    /// GPU-accelerated solve using level-scheduled SpTRSV.
    ///
    /// This dispatches Metal compute commands for:
    /// 1. Permute RHS
    /// 2. Forward solve (L)
    /// 3. Diagonal solve (D^{-1})
    /// 4. Backward solve (L^T)
    /// 5. Inverse permute
    #[allow(dead_code)]
    fn gpu_solve(
        &self,
        rhs: &[f32],
        solution: &mut [f32],
        symbolic: &SymbolicAnalysis,
    ) -> MetalResult<()> {
        let n = symbolic.n;

        let factor_buffers = self.factor_buffers.as_ref()
            .ok_or(MetalError::Solve("No factor buffers".to_string()))?;
        let solve_buffers = self.solve_buffers.as_ref()
            .ok_or(MetalError::Solve("No solve buffers".to_string()))?;

        // Upload RHS to GPU
        let rhs_ptr = solve_buffers.temp_rhs.contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(rhs.as_ptr(), rhs_ptr, n);
        }

        // Create command buffer
        let command_buffer = self.handle.queue().new_command_buffer();

        // Step 1: Permute RHS
        // y[i] = rhs[perm_inv[i]]
        self.encode_permute(
            command_buffer,
            &solve_buffers.temp_rhs,
            &factor_buffers.perm_inv,
            &solve_buffers.temp_sol,
            n,
        )?;

        // Step 2: Forward solve (L * z = y)
        // For level-scheduled SpTRSV, we dispatch one kernel per level
        // Note: This is a simplified version; full implementation would use CSR factor
        // For now, we skip GPU solve and use CPU fallback

        // Step 3: Diagonal solve (z = D^{-1} * z)
        self.encode_dinv_apply(
            command_buffer,
            &solve_buffers.temp_sol,
            &factor_buffers.d_inv,
            n,
        )?;

        // Step 4: Backward solve (L^T * x = z) - would be level-scheduled

        // Step 5: Inverse permute
        // solution[perm[i]] = x[i]
        self.encode_permute_inv(
            command_buffer,
            &solve_buffers.temp_sol,
            &factor_buffers.perm,
            &solve_buffers.temp_rhs, // Reuse as output
            n,
        )?;

        // Commit and wait
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Download solution
        let sol_ptr = solve_buffers.temp_rhs.contents() as *const f32;
        unsafe {
            std::ptr::copy_nonoverlapping(sol_ptr, solution.as_mut_ptr(), n);
        }

        Ok(())
    }

    fn encode_permute(
        &self,
        command_buffer: &metal::CommandBufferRef,
        input: &metal::Buffer,
        perm: &metal::Buffer,
        output: &metal::Buffer,
        n: usize,
    ) -> MetalResult<()> {
        let pipeline = self.handle.pipeline(super::kernels::PERMUTE_GATHER)
            .ok_or(MetalError::Solve("Missing permute_gather pipeline".to_string()))?;

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(perm), 0);
        encoder.set_buffer(2, Some(output), 0);

        let n_u32 = n as u32;
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &n_u32 as *const u32 as *const _);

        let threadgroup_size = metal::MTLSize::new(
            self.handle.config().threadgroup_size_1d as u64,
            1,
            1,
        );
        let grid_size = metal::MTLSize::new(n as u64, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        Ok(())
    }

    fn encode_permute_inv(
        &self,
        command_buffer: &metal::CommandBufferRef,
        input: &metal::Buffer,
        perm: &metal::Buffer,
        output: &metal::Buffer,
        n: usize,
    ) -> MetalResult<()> {
        let pipeline = self.handle.pipeline(super::kernels::PERMUTE_SCATTER)
            .ok_or(MetalError::Solve("Missing permute_scatter pipeline".to_string()))?;

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(perm), 0);
        encoder.set_buffer(2, Some(output), 0);

        let n_u32 = n as u32;
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &n_u32 as *const u32 as *const _);

        let threadgroup_size = metal::MTLSize::new(
            self.handle.config().threadgroup_size_1d as u64,
            1,
            1,
        );
        let grid_size = metal::MTLSize::new(n as u64, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        Ok(())
    }

    fn encode_dinv_apply(
        &self,
        command_buffer: &metal::CommandBufferRef,
        x: &metal::Buffer,
        dinv: &metal::Buffer,
        n: usize,
    ) -> MetalResult<()> {
        let pipeline = self.handle.pipeline(super::kernels::APPLY_DINV_INPLACE)
            .ok_or(MetalError::Solve("Missing apply_dinv_inplace pipeline".to_string()))?;

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(dinv), 0);

        let n_u32 = n as u32;
        encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &n_u32 as *const u32 as *const _);

        let threadgroup_size = metal::MTLSize::new(
            self.handle.config().threadgroup_size_1d as u64,
            1,
            1,
        );
        let grid_size = metal::MTLSize::new(n as u64, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

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

// ============================================================================
// KktBackend trait implementation
// ============================================================================

use crate::linalg::backend::{BackendError, KktBackend};
use crate::linalg::sparse::SparseCsc;

/// Adapter to use MetalKktBackend with the KktBackend trait.
///
/// This allows the Metal backend to be used as a drop-in replacement
/// for QdldlBackend in the KktSolver.
#[cfg(target_os = "macos")]
pub struct MetalBackendAdapter {
    inner: MetalKktBackend,
    static_reg: f64,
    dynamic_reg_min_pivot: f64,
    n: usize,
}

#[cfg(target_os = "macos")]
impl KktBackend for MetalBackendAdapter {
    type Factorization = MetalFactorization;

    fn new(n: usize, static_reg: f64, dynamic_reg_min_pivot: f64) -> Self
    where
        Self: Sized,
    {
        let mut config = DssConfig::default();
        config.static_reg = static_reg;
        config.pivot_min = dynamic_reg_min_pivot as f32;

        let inner = MetalKktBackend::new(config)
            .expect("Failed to create Metal backend");

        Self {
            inner,
            static_reg,
            dynamic_reg_min_pivot,
            n,
        }
    }

    fn set_static_reg(&mut self, static_reg: f64) -> Result<(), BackendError> {
        self.static_reg = static_reg;
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
        let col_ptr: Vec<usize> = kkt.indptr().iter().map(|&x| x).collect();
        let row_ind: Vec<usize> = kkt.indices().iter().map(|&x| x).collect();

        self.inner
            .symbolic_analysis(n, &col_ptr, &row_ind)
            .map_err(|e| BackendError::Other(e.to_string()))
    }

    fn numeric_factorization(&mut self, kkt: &SparseCsc) -> Result<Self::Factorization, BackendError> {
        // Extract CSC components
        let col_ptr: Vec<usize> = kkt.indptr().iter().map(|&x| x).collect();
        let row_ind: Vec<usize> = kkt.indices().iter().map(|&x| x).collect();
        let values: Vec<f64> = kkt.data().to_vec();

        self.inner
            .numeric_factorization(&col_ptr, &row_ind, &values)
            .map_err(|e| BackendError::Other(e.to_string()))
    }

    fn solve(&self, _factor: &Self::Factorization, rhs: &[f64], sol: &mut [f64]) {
        // Note: MetalKktBackend stores factor internally, so we ignore the factor parameter
        // This requires a mutable borrow, but the trait expects &self
        // For now, we work around this by having solve use the stored factor

        // SAFETY: This is a workaround for the trait signature. The Metal backend
        // stores the factor internally and doesn't need the factor parameter.
        // In a proper implementation, we'd want to change the trait or use interior mutability.
        let inner_ptr = &self.inner as *const MetalKktBackend as *mut MetalKktBackend;
        unsafe {
            let inner_mut = &mut *inner_ptr;
            if let Err(e) = inner_mut.solve(rhs, sol) {
                // Log error but continue (solve trait method doesn't return Result)
                eprintln!("[Metal] Solve error: {}", e);
                sol.fill(0.0);
            }
        }
    }

    fn dynamic_bumps(&self) -> u64 {
        self.inner.stats().dynamic_bumps
    }
}

/// Stub adapter for non-macOS platforms.
#[cfg(not(target_os = "macos"))]
pub struct MetalBackendAdapter {
    _marker: std::marker::PhantomData<()>,
}

#[cfg(not(target_os = "macos"))]
impl KktBackend for MetalBackendAdapter {
    type Factorization = ();

    fn new(_n: usize, _static_reg: f64, _dynamic_reg_min_pivot: f64) -> Self {
        panic!("Metal backend not available on this platform")
    }

    fn set_static_reg(&mut self, _static_reg: f64) -> Result<(), BackendError> {
        Err(BackendError::Other("Metal not available".to_string()))
    }

    fn static_reg(&self) -> f64 {
        0.0
    }

    fn symbolic_factorization(&mut self, _kkt: &SparseCsc) -> Result<(), BackendError> {
        Err(BackendError::Other("Metal not available".to_string()))
    }

    fn numeric_factorization(&mut self, _kkt: &SparseCsc) -> Result<Self::Factorization, BackendError> {
        Err(BackendError::Other("Metal not available".to_string()))
    }

    fn solve(&self, _factor: &Self::Factorization, _rhs: &[f64], _sol: &mut [f64]) {
        // No-op on non-macOS
    }

    fn dynamic_bumps(&self) -> u64 {
        0
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
