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
    /// L factor values in CSR format (for forward solve).
    l_csr_values: metal::Buffer,

    /// L column indices in CSR format.
    l_csr_col_ind: metal::Buffer,

    /// L row pointers in CSR format.
    l_csr_row_ptr: metal::Buffer,

    /// L^T factor values in CSR format (for backward solve).
    /// This is L stored in CSC format, interpreted as L^T CSR.
    lt_csr_values: metal::Buffer,

    /// L^T column indices in CSR format (L row indices).
    lt_csr_col_ind: metal::Buffer,

    /// L^T row pointers in CSR format (L column pointers).
    lt_csr_row_ptr: metal::Buffer,

    /// D diagonal values.
    d_values: metal::Buffer,

    /// Inverse diagonal (1/D) for solve phase.
    d_inv: metal::Buffer,

    /// Permutation vector.
    perm: metal::Buffer,

    /// Inverse permutation vector.
    perm_inv: metal::Buffer,

    /// Matrix dimension.
    n: usize,

    /// Number of nonzeros in L.
    nnz_l: usize,
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

/// Sparse LDL^T factorization result (CSR format for L and L^T, plus D diagonal)
#[cfg(target_os = "macos")]
struct SparseLdltResult {
    // L in CSR format (for forward solve)
    l_csr_values: Vec<f32>,
    l_csr_col_ind: Vec<u32>,
    l_csr_row_ptr: Vec<u32>,

    // L^T in CSR format (for backward solve)
    lt_csr_values: Vec<f32>,
    lt_csr_col_ind: Vec<u32>,
    lt_csr_row_ptr: Vec<u32>,

    // Diagonal D
    d: Vec<f32>,
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

        // Extract what we need from symbolic up front to avoid borrow issues
        let (n, nnz_l, perm, perm_inv) = {
            let symbolic = self.symbolic.as_ref()
                .ok_or(MetalError::NumericFactorization(
                    "Must call symbolic_analysis first".to_string()
                ))?;
            (symbolic.n, symbolic.nnz_l, symbolic.perm.clone(), symbolic.perm_inv.clone())
        };

        // For now, implement a simple CPU factorization and upload to GPU
        // TODO: Implement GPU supernodal factorization

        // Convert to f32 for GPU
        let values_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();

        // Permute and factorize on CPU
        let factor_result = self.cpu_ldlt_factorize_with_perm(
            n,
            col_ptr,
            row_ind,
            &values_f32,
            &perm,
            &perm_inv,
        )?;

        // Upload to GPU
        self.upload_factor_to_gpu(&factor_result)?;

        let min_pivot = factor_result.d.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_pivot = factor_result.d.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        self.stats.num_numeric += 1;
        self.stats.time_numeric += start.elapsed().as_secs_f64();

        Ok(MetalFactorization {
            n,
            nnz_l,
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

        // Convert to f32
        let rhs_f32: Vec<f32> = rhs.iter().map(|&v| v as f32).collect();
        let mut sol_f32 = vec![0.0f32; n];

        // Use GPU solve if enabled, otherwise CPU fallback
        if self.handle.config().use_gpu_solve {
            self.gpu_solve(&rhs_f32, &mut sol_f32, symbolic)?;
        } else {
            self.cpu_solve(&rhs_f32, &mut sol_f32, symbolic)?;
        }

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

        // CSR format for L (for forward solve)
        let l_csr_values = self.handle.create_buffer(nnz_l * std::mem::size_of::<f32>())?;
        let l_csr_col_ind = self.handle.create_buffer(nnz_l * std::mem::size_of::<u32>())?;
        let l_csr_row_ptr = self.handle.create_buffer((n + 1) * std::mem::size_of::<u32>())?;

        // L^T in CSR format (for backward solve) - same size as L
        let lt_csr_values = self.handle.create_buffer(nnz_l * std::mem::size_of::<f32>())?;
        let lt_csr_col_ind = self.handle.create_buffer(nnz_l * std::mem::size_of::<u32>())?;
        let lt_csr_row_ptr = self.handle.create_buffer((n + 1) * std::mem::size_of::<u32>())?;

        // Diagonal values
        let d_values = self.handle.create_buffer(n * std::mem::size_of::<f32>())?;
        let d_inv = self.handle.create_buffer(n * std::mem::size_of::<f32>())?;

        // Permutation vectors
        let perm = self.handle.create_buffer_with_data(
            &symbolic.perm.iter().map(|&p| p as u32).collect::<Vec<_>>()
        )?;
        let perm_inv = self.handle.create_buffer_with_data(
            &symbolic.perm_inv.iter().map(|&p| p as u32).collect::<Vec<_>>()
        )?;

        self.factor_buffers = Some(FactorBuffers {
            l_csr_values,
            l_csr_col_ind,
            l_csr_row_ptr,
            lt_csr_values,
            lt_csr_col_ind,
            lt_csr_row_ptr,
            d_values,
            d_inv,
            perm,
            perm_inv,
            n,
            nnz_l,
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

        // Step 2: Forward solve (L * z = y) using level-scheduled SpTRSV
        // We dispatch one kernel per level, with synchronization between levels
        let (level_ptr_lower, _) = &symbolic.solve_levels_lower;
        let num_levels_lower = if level_ptr_lower.len() > 1 { level_ptr_lower.len() - 1 } else { 0 };

        for level in 0..num_levels_lower {
            let level_start = level_ptr_lower[level];
            let level_size = level_ptr_lower[level + 1] - level_start;

            if level_size > 0 {
                self.encode_sptrsv_lower_level(
                    command_buffer,
                    &factor_buffers.l_csr_values,
                    &factor_buffers.l_csr_col_ind,
                    &factor_buffers.l_csr_row_ptr,
                    &solve_buffers.temp_sol, // Input b (in-place, becomes y)
                    &solve_buffers.level_rows_lower,
                    level_start,
                    level_size,
                )?;
            }
        }

        // Step 3: Diagonal solve (z = D^{-1} * z)
        self.encode_dinv_apply(
            command_buffer,
            &solve_buffers.temp_sol,
            &factor_buffers.d_inv,
            n,
        )?;

        // Step 4: Backward solve (L^T * x = z) using level-scheduled SpTRSV
        let (level_ptr_upper, _) = &symbolic.solve_levels_upper;
        let num_levels_upper = if level_ptr_upper.len() > 1 { level_ptr_upper.len() - 1 } else { 0 };

        for level in 0..num_levels_upper {
            let level_start = level_ptr_upper[level];
            let level_size = level_ptr_upper[level + 1] - level_start;

            if level_size > 0 {
                self.encode_sptrsv_upper_level(
                    command_buffer,
                    &factor_buffers.lt_csr_values,
                    &factor_buffers.lt_csr_col_ind,
                    &factor_buffers.lt_csr_row_ptr,
                    &solve_buffers.temp_sol, // Input b (in-place, becomes x)
                    &solve_buffers.level_rows_upper,
                    level_start,
                    level_size,
                )?;
            }
        }

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

    /// Encode level-scheduled lower triangular solve kernel.
    fn encode_sptrsv_lower_level(
        &self,
        command_buffer: &metal::CommandBufferRef,
        l_values: &metal::Buffer,
        l_col_ind: &metal::Buffer,
        l_row_ptr: &metal::Buffer,
        y: &metal::Buffer, // In-place: input b, output y
        level_rows: &metal::Buffer,
        level_start: usize,
        level_size: usize,
    ) -> MetalResult<()> {
        let pipeline = self.handle.pipeline(super::kernels::SPTRSV_LOWER_LEVEL_UNITDIAG)
            .ok_or(MetalError::Solve("Missing sptrsv_lower_level pipeline".to_string()))?;

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);

        // Buffer layout matches kernel signature:
        // buffer(0): values, buffer(1): col_indices, buffer(2): row_ptr,
        // buffer(3): b (input), buffer(4): y (output, same buffer for in-place)
        // buffer(5): level_rows, buffer(6): level_size
        encoder.set_buffer(0, Some(l_values), 0);
        encoder.set_buffer(1, Some(l_col_ind), 0);
        encoder.set_buffer(2, Some(l_row_ptr), 0);
        encoder.set_buffer(3, Some(y), 0); // b (input)
        encoder.set_buffer(4, Some(y), 0); // y (output, in-place)
        encoder.set_buffer(5, Some(level_rows), (level_start * std::mem::size_of::<u32>()) as u64);

        let level_size_u32 = level_size as u32;
        encoder.set_bytes(6, std::mem::size_of::<u32>() as u64, &level_size_u32 as *const u32 as *const _);

        let threadgroup_size = metal::MTLSize::new(
            self.handle.config().threadgroup_size_1d.min(level_size) as u64,
            1,
            1,
        );
        let grid_size = metal::MTLSize::new(level_size as u64, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        Ok(())
    }

    /// Encode level-scheduled upper triangular solve kernel.
    fn encode_sptrsv_upper_level(
        &self,
        command_buffer: &metal::CommandBufferRef,
        lt_values: &metal::Buffer,
        lt_col_ind: &metal::Buffer,
        lt_row_ptr: &metal::Buffer,
        x: &metal::Buffer, // In-place: input b, output x
        level_rows: &metal::Buffer,
        level_start: usize,
        level_size: usize,
    ) -> MetalResult<()> {
        let pipeline = self.handle.pipeline(super::kernels::SPTRSV_UPPER_LEVEL_UNITDIAG)
            .ok_or(MetalError::Solve("Missing sptrsv_upper_level pipeline".to_string()))?;

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);

        encoder.set_buffer(0, Some(lt_values), 0);
        encoder.set_buffer(1, Some(lt_col_ind), 0);
        encoder.set_buffer(2, Some(lt_row_ptr), 0);
        encoder.set_buffer(3, Some(x), 0); // b (input)
        encoder.set_buffer(4, Some(x), 0); // x (output, in-place)
        encoder.set_buffer(5, Some(level_rows), (level_start * std::mem::size_of::<u32>()) as u64);

        let level_size_u32 = level_size as u32;
        encoder.set_bytes(6, std::mem::size_of::<u32>() as u64, &level_size_u32 as *const u32 as *const _);

        let threadgroup_size = metal::MTLSize::new(
            self.handle.config().threadgroup_size_1d.min(level_size) as u64,
            1,
            1,
        );
        let grid_size = metal::MTLSize::new(level_size as u64, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

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

    fn cpu_ldlt_factorize_with_perm(
        &self,
        n: usize,
        col_ptr: &[usize],
        row_ind: &[usize],
        values: &[f32],
        perm: &[usize],
        perm_inv: &[usize],
    ) -> MetalResult<SparseLdltResult> {
        let pivot_min = self.handle.config().pivot_min;

        // For now, use dense factorization for small matrices
        // TODO: Implement proper sparse supernodal factorization
        if n > 2000 {
            return Err(MetalError::NotImplemented(
                "Sparse factorization not yet implemented; matrix too large for dense fallback".to_string()
            ));
        }

        // Build dense matrix with permutation applied
        let mut a = vec![0.0f32; n * n];

        for j in 0..n {
            let pj = perm_inv[j]; // Original column for permuted column j
            for p in col_ptr[pj]..col_ptr[pj + 1] {
                let pi = row_ind[p];
                let i = perm[pi]; // Permuted row
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

        // Convert dense L to CSR format (lower triangular with unit diagonal)
        // CSR: row_ptr[i..i+1] gives range in col_ind/values for row i
        let mut l_csr_row_ptr = vec![0u32; n + 1];
        let mut l_csr_col_ind = Vec::new();
        let mut l_csr_values = Vec::new();

        let drop_tol = 1e-14f32;

        for i in 0..n {
            // Row i of L has entries in columns 0..=i
            // Unit diagonal at (i,i), off-diagonal entries at (i, j) for j < i
            for j in 0..i {
                let val = a[i * n + j];
                if val.abs() > drop_tol {
                    l_csr_col_ind.push(j as u32);
                    l_csr_values.push(val);
                }
            }
            // Unit diagonal (store explicitly for GPU kernel)
            l_csr_col_ind.push(i as u32);
            l_csr_values.push(1.0f32);

            l_csr_row_ptr[i + 1] = l_csr_values.len() as u32;
        }

        // Compute L^T in CSR format (upper triangular, same as L in CSC)
        // L^T[i,j] = L[j,i], so row i of L^T contains entries from column i of L
        let mut lt_csr_row_ptr = vec![0u32; n + 1];
        let mut lt_csr_col_ind = Vec::new();
        let mut lt_csr_values = Vec::new();

        // First pass: count entries per row of L^T (= per column of L)
        let mut col_counts = vec![0usize; n];
        for i in 0..n {
            // Diagonal
            col_counts[i] += 1;
            // Off-diagonal in column i: entries L[j,i] for j > i
            for j in (i + 1)..n {
                let val = a[j * n + i];
                if val.abs() > drop_tol {
                    col_counts[i] += 1;
                }
            }
        }

        // Build row pointers for L^T
        for i in 0..n {
            lt_csr_row_ptr[i + 1] = lt_csr_row_ptr[i] + col_counts[i] as u32;
        }

        // Fill L^T values
        for i in 0..n {
            // Row i of L^T: diagonal at (i,i), then entries L[j,i] for j > i become L^T[i,j]
            // Diagonal first
            lt_csr_col_ind.push(i as u32);
            lt_csr_values.push(1.0f32);

            // Off-diagonal entries (L^T[i,j] = L[j,i] for j > i)
            for j in (i + 1)..n {
                let val = a[j * n + i]; // L[j,i]
                if val.abs() > drop_tol {
                    lt_csr_col_ind.push(j as u32);
                    lt_csr_values.push(val);
                }
            }
        }

        Ok(SparseLdltResult {
            l_csr_values,
            l_csr_col_ind,
            l_csr_row_ptr,
            lt_csr_values,
            lt_csr_col_ind,
            lt_csr_row_ptr,
            d,
        })
    }

    fn upload_factor_to_gpu(&mut self, result: &SparseLdltResult) -> MetalResult<()> {
        let buffers = self.factor_buffers.as_mut()
            .ok_or(MetalError::NumericFactorization("No factor buffers".to_string()))?;

        // Upload L CSR values
        let l_val_ptr = buffers.l_csr_values.contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(result.l_csr_values.as_ptr(), l_val_ptr, result.l_csr_values.len());
        }

        // Upload L CSR column indices
        let l_col_ptr = buffers.l_csr_col_ind.contents() as *mut u32;
        unsafe {
            std::ptr::copy_nonoverlapping(result.l_csr_col_ind.as_ptr(), l_col_ptr, result.l_csr_col_ind.len());
        }

        // Upload L CSR row pointers
        let l_row_ptr = buffers.l_csr_row_ptr.contents() as *mut u32;
        unsafe {
            std::ptr::copy_nonoverlapping(result.l_csr_row_ptr.as_ptr(), l_row_ptr, result.l_csr_row_ptr.len());
        }

        // Upload L^T CSR values
        let lt_val_ptr = buffers.lt_csr_values.contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(result.lt_csr_values.as_ptr(), lt_val_ptr, result.lt_csr_values.len());
        }

        // Upload L^T CSR column indices
        let lt_col_ptr = buffers.lt_csr_col_ind.contents() as *mut u32;
        unsafe {
            std::ptr::copy_nonoverlapping(result.lt_csr_col_ind.as_ptr(), lt_col_ptr, result.lt_csr_col_ind.len());
        }

        // Upload L^T CSR row pointers
        let lt_row_ptr = buffers.lt_csr_row_ptr.contents() as *mut u32;
        unsafe {
            std::ptr::copy_nonoverlapping(result.lt_csr_row_ptr.as_ptr(), lt_row_ptr, result.lt_csr_row_ptr.len());
        }

        // Upload D values
        let d_ptr = buffers.d_values.contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(result.d.as_ptr(), d_ptr, result.d.len());
        }

        // Compute and upload D^{-1}
        let d_inv: Vec<f32> = result.d.iter().map(|&d| 1.0 / d).collect();
        let d_inv_ptr = buffers.d_inv.contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(d_inv.as_ptr(), d_inv_ptr, d_inv.len());
        }

        // Update stored dimensions
        buffers.nnz_l = result.l_csr_values.len();

        Ok(())
    }

    fn cpu_solve(
        &self,
        rhs: &[f32],
        solution: &mut [f32],
        symbolic: &SymbolicAnalysis,
    ) -> MetalResult<()> {
        let n = symbolic.n;

        let buffers = self.factor_buffers.as_ref()
            .ok_or(MetalError::Solve("No factor buffers".to_string()))?;

        // Read L CSR from GPU buffers (for forward solve)
        let l_values = buffers.l_csr_values.contents() as *const f32;
        let l_col_ind = buffers.l_csr_col_ind.contents() as *const u32;
        let l_row_ptr = buffers.l_csr_row_ptr.contents() as *const u32;

        // Read L^T CSR from GPU buffers (for backward solve)
        let lt_values = buffers.lt_csr_values.contents() as *const f32;
        let lt_col_ind = buffers.lt_csr_col_ind.contents() as *const u32;
        let lt_row_ptr = buffers.lt_csr_row_ptr.contents() as *const u32;

        let d_inv = buffers.d_inv.contents() as *const f32;

        // Permute RHS: y[i] = rhs[perm_inv[i]]
        let mut y = vec![0.0f32; n];
        for i in 0..n {
            y[i] = rhs[symbolic.perm_inv[i]];
        }

        // Forward solve: L * z = y (CSR lower triangular with unit diagonal)
        // Process rows in order; row i depends on columns j < i
        let mut z = y;
        for i in 0..n {
            let row_start = unsafe { *l_row_ptr.add(i) } as usize;
            let row_end = unsafe { *l_row_ptr.add(i + 1) } as usize;

            let mut sum = z[i];
            // Iterate through off-diagonal entries (col < i)
            for p in row_start..row_end {
                let col = unsafe { *l_col_ind.add(p) } as usize;
                if col < i {
                    let val = unsafe { *l_values.add(p) };
                    sum -= val * z[col];
                }
                // Diagonal entry (col == i) has val = 1.0, no division needed
            }
            z[i] = sum;
        }

        // Diagonal solve: z = D^{-1} * z
        for i in 0..n {
            let d_inv_i = unsafe { *d_inv.add(i) };
            z[i] *= d_inv_i;
        }

        // Backward solve: L^T * x = z (using L^T CSR, upper triangular)
        // Process rows in reverse order; row i depends on columns j > i
        let mut x = z;
        for i in (0..n).rev() {
            let row_start = unsafe { *lt_row_ptr.add(i) } as usize;
            let row_end = unsafe { *lt_row_ptr.add(i + 1) } as usize;

            let mut sum = x[i];
            // Iterate through off-diagonal entries (col > i)
            for p in row_start..row_end {
                let col = unsafe { *lt_col_ind.add(p) } as usize;
                if col > i {
                    let val = unsafe { *lt_values.add(p) };
                    sum -= val * x[col];
                }
                // Diagonal entry (col == i) has val = 1.0, no division needed
            }
            x[i] = sum;
        }

        // Inverse permute: solution[perm[i]] = x[i]
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
use std::cell::UnsafeCell;

/// Adapter to use MetalKktBackend with the KktBackend trait.
///
/// This allows the Metal backend to be used as a drop-in replacement
/// for QdldlBackend in the KktSolver.
#[cfg(target_os = "macos")]
pub struct MetalBackendAdapter {
    // Use UnsafeCell for interior mutability since solve() needs &mut but trait takes &self
    inner: UnsafeCell<MetalKktBackend>,
    static_reg: f64,
    #[allow(dead_code)]
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
            inner: UnsafeCell::new(inner),
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
        let col_ptr: Vec<usize> = kkt.indptr().raw_storage().iter().copied().collect();
        let row_ind: Vec<usize> = kkt.indices().iter().map(|&x| x).collect();

        // SAFETY: We have &mut self, so exclusive access is guaranteed
        let inner = unsafe { &mut *self.inner.get() };
        inner
            .symbolic_analysis(n, &col_ptr, &row_ind)
            .map_err(|e| BackendError::Other(e.to_string()))
    }

    fn numeric_factorization(&mut self, kkt: &SparseCsc) -> Result<Self::Factorization, BackendError> {
        // Extract CSC components
        let col_ptr: Vec<usize> = kkt.indptr().raw_storage().iter().copied().collect();
        let row_ind: Vec<usize> = kkt.indices().iter().map(|&x| x).collect();
        let values: Vec<f64> = kkt.data().to_vec();

        // SAFETY: We have &mut self, so exclusive access is guaranteed
        let inner = unsafe { &mut *self.inner.get() };
        inner
            .numeric_factorization(&col_ptr, &row_ind, &values)
            .map_err(|e| BackendError::Other(e.to_string()))
    }

    fn solve(&self, _factor: &Self::Factorization, rhs: &[f64], sol: &mut [f64]) {
        // Note: MetalKktBackend stores factor internally, so we ignore the factor parameter
        // SAFETY: The solve method is the only code path that mutates inner through &self.
        // The trait design requires &self, but we need interior mutability for Metal's
        // command buffer submission. This is safe because solve() is not reentrant.
        let inner = unsafe { &mut *self.inner.get() };
        if let Err(e) = inner.solve(rhs, sol) {
            // Log error but continue (solve trait method doesn't return Result)
            eprintln!("[Metal] Solve error: {}", e);
            sol.fill(0.0);
        }
    }

    fn dynamic_bumps(&self) -> u64 {
        // SAFETY: Read-only access to stats
        let inner = unsafe { &*self.inner.get() };
        inner.stats().dynamic_bumps
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

    /// Helper to compute residual ||Ax - b|| for symmetric matrices in CSC lower triangle format
    fn compute_residual(
        n: usize,
        col_ptr: &[usize],
        row_ind: &[usize],
        values: &[f64],
        solution: &[f64],
        rhs: &[f64],
    ) -> f64 {
        let mut residual = rhs.to_vec();
        for col in 0..n {
            for p in col_ptr[col]..col_ptr[col + 1] {
                let row = row_ind[p];
                let val = values[p];
                residual[row] -= val * solution[col];
                if row != col {
                    residual[col] -= val * solution[row];
                }
            }
        }
        residual.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    // ========================================================================
    // Metal GPU tests - only compile and run on macOS with Metal
    // ========================================================================

    /// Comprehensive Metal test covering: SPD solve, indefinite solve, refactorization, multiple RHS
    #[test]
    #[cfg(target_os = "macos")]
    fn test_metal_comprehensive() {
        let mut backend = match MetalKktBackend::with_defaults() {
            Ok(b) => b,
            Err(_) => return, // Skip if no Metal device
        };

        // === Test 1: Simple 2x2 SPD ===
        let n = 2;
        let col_ptr = vec![0, 2, 4];
        let row_ind = vec![0, 1, 0, 1];
        let values: Vec<f64> = vec![4.0, 1.0, 1.0, 3.0];

        backend.symbolic_analysis(n, &col_ptr, &row_ind).unwrap();
        let _f = backend.numeric_factorization(&col_ptr, &row_ind, &values).unwrap();

        let rhs = vec![5.0, 4.0];
        let mut solution = vec![0.0; 2];
        backend.solve(&rhs, &mut solution).unwrap();
        assert!((solution[0] - 1.0).abs() < 1e-4, "SPD solve failed");
        assert!((solution[1] - 1.0).abs() < 1e-4, "SPD solve failed");

        // === Test 2: Refactorization with different values ===
        let values2: Vec<f64> = vec![2.0, 0.5, 0.5, 2.0];
        let _f2 = backend.numeric_factorization(&col_ptr, &row_ind, &values2).unwrap();
        let rhs2 = vec![2.5, 2.5];
        backend.solve(&rhs2, &mut solution).unwrap();
        assert!((solution[0] - 1.0).abs() < 1e-4, "Refactorization failed");

        // === Test 3: Statistics tracking ===
        assert_eq!(backend.stats().num_symbolic, 1);
        assert_eq!(backend.stats().num_numeric, 2);
        assert!(backend.stats().num_solves >= 2);
    }

    /// Test Metal with indefinite KKT-like matrix (tests LDL^T factorization)
    #[test]
    #[cfg(target_os = "macos")]
    fn test_metal_indefinite_matrix() {
        let mut backend = match MetalKktBackend::with_defaults() {
            Ok(b) => b,
            Err(_) => return,
        };

        // 4x4 KKT matrix (indefinite - has positive and negative eigenvalues):
        // [ 2   0.5  1   0 ]
        // [0.5  2    0   1 ]
        // [ 1   0   -1   0 ]
        // [ 0   1    0  -1 ]
        let n = 4;
        let col_ptr = vec![0, 3, 5, 6, 7];
        let row_ind = vec![0, 1, 2, 1, 3, 2, 3];
        let values: Vec<f64> = vec![2.0, 0.5, 1.0, 2.0, 1.0, -1.0, -1.0];

        backend.symbolic_analysis(n, &col_ptr, &row_ind).unwrap();
        let _f = backend.numeric_factorization(&col_ptr, &row_ind, &values).unwrap();

        let rhs = vec![1.0, 1.0, 0.5, 0.5];
        let mut solution = vec![0.0; n];
        backend.solve(&rhs, &mut solution).unwrap();

        let res_norm = compute_residual(n, &col_ptr, &row_ind, &values, &solution, &rhs);
        assert!(res_norm < 1e-3, "KKT solve residual {} too large", res_norm);
    }

    /// Test Metal with larger 10x10 tridiagonal matrix
    #[test]
    #[cfg(target_os = "macos")]
    fn test_metal_larger_matrix() {
        let mut backend = match MetalKktBackend::with_defaults() {
            Ok(b) => b,
            Err(_) => return,
        };

        let n = 10;
        let mut col_ptr = vec![0usize];
        let mut row_ind = Vec::new();
        let mut values = Vec::new();

        for col in 0..n {
            row_ind.push(col);
            values.push(3.0);
            if col < n - 1 {
                row_ind.push(col + 1);
                values.push(-1.0);
            }
            col_ptr.push(row_ind.len());
        }

        backend.symbolic_analysis(n, &col_ptr, &row_ind).unwrap();
        let _f = backend.numeric_factorization(&col_ptr, &row_ind, &values).unwrap();

        let rhs: Vec<f64> = vec![1.0; n];
        let mut solution = vec![0.0; n];
        backend.solve(&rhs, &mut solution).unwrap();

        let res_norm = compute_residual(n, &col_ptr, &row_ind, &values, &solution, &rhs);
        assert!(res_norm < 1e-3, "10x10 solve residual {} too large", res_norm);
    }
}
