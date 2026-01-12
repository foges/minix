//! Metal device handle and configuration.
//!
//! This module provides the core infrastructure for interacting with Metal:
//! - `DssHandle`: Manages device, command queue, compiled library, and pipeline cache.
//! - `DssConfig`: Configuration options for the solver (ordering, precision, tuning).

use super::error::{MetalError, MetalResult};
use super::kernels;
use std::collections::HashMap;

#[cfg(target_os = "macos")]
use metal::{Device, CommandQueue, Library, ComputePipelineState, CompileOptions};

/// Configuration for the Metal sparse direct solver.
#[derive(Debug, Clone)]
pub struct DssConfig {
    /// Ordering method for fill reduction.
    pub ordering: OrderingMethod,

    /// Minimum pivot magnitude (for numerical stability).
    /// Pivots smaller than this are clamped. Set to 0 to disable.
    pub pivot_min: f32,

    /// Static regularization to add to diagonal.
    pub static_reg: f64,

    /// Number of iterative refinement steps after solve.
    pub refine_iters: usize,

    /// Threadgroup size for 1D kernels.
    pub threadgroup_size_1d: usize,

    /// Tile size for dense SYRK update (portable kernel).
    pub tile_size: usize,

    /// Use SIMD-group matrix ops when available (requires Apple GPU family 7+).
    pub use_simd_group_matrix: bool,

    /// Enable verbose logging for debugging.
    pub verbose: bool,
}

impl Default for DssConfig {
    fn default() -> Self {
        Self {
            ordering: OrderingMethod::Amd,
            pivot_min: 1e-12,
            static_reg: 1e-8,
            refine_iters: 2,
            threadgroup_size_1d: 256,
            tile_size: 16,
            use_simd_group_matrix: true,
            verbose: false,
        }
    }
}

/// Fill-reducing ordering method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderingMethod {
    /// No reordering (identity permutation).
    Natural,
    /// Approximate Minimum Degree (good general-purpose choice).
    Amd,
    /// Nested dissection (best for very large problems, requires external library).
    NestedDissection,
}

/// Handle to Metal device, command queue, and compiled pipelines.
///
/// This is analogous to cuDSS's handle + data objects combined.
/// Create once and reuse across multiple solves.
#[cfg(target_os = "macos")]
pub struct DssHandle {
    /// Metal device (GPU).
    pub(crate) device: Device,

    /// Command queue for dispatching work.
    pub(crate) queue: CommandQueue,

    /// Compiled shader library.
    pub(crate) library: Library,

    /// Cached compute pipeline states, keyed by kernel name.
    pub(crate) pipelines: HashMap<&'static str, ComputePipelineState>,

    /// Configuration.
    pub(crate) config: DssConfig,
}

#[cfg(target_os = "macos")]
impl DssHandle {
    /// Create a new Metal handle with the given configuration.
    ///
    /// This will:
    /// 1. Get the default Metal device
    /// 2. Create a command queue
    /// 3. Compile the shader library
    /// 4. Create compute pipelines for all kernels
    pub fn new(config: DssConfig) -> MetalResult<Self> {
        // Get default device
        let device = Device::system_default()
            .ok_or(MetalError::NoDevice)?;

        if config.verbose {
            println!("[Metal] Using device: {}", device.name());
        }

        // Create command queue
        let queue = device.new_command_queue();

        // Compile shader source
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(super::METAL_SHADER_SOURCE, &options)
            .map_err(|e| MetalError::ShaderCompilation(e.to_string()))?;

        if config.verbose {
            println!("[Metal] Shader library compiled successfully");
        }

        // Create pipelines for all kernels
        let mut pipelines = HashMap::new();

        let kernel_names = [
            // Sparse
            kernels::CSR_SPMV,
            kernels::CSR_SPMV_ADD,
            kernels::CSR_ROW_SUMSQUARES,
            // Reductions
            kernels::DOT_PARTIAL,
            kernels::REDUCE_SUM_PARTIAL,
            // Vector ops
            kernels::VEC_ADD,
            kernels::VEC_SUB,
            kernels::VEC_COPY,
            kernels::VEC_SET,
            kernels::VEC_SCALE_INPLACE,
            kernels::VEC_AXPY_INPLACE,
            kernels::VEC_XPAY_INPLACE,
            // Cone projections
            kernels::PROJ_NONNEG_INPLACE,
            kernels::PROJ_SOC_INPLACE,
            // Permutations
            kernels::PERMUTE_GATHER,
            kernels::PERMUTE_SCATTER,
            kernels::PERMUTE_VEC,
            kernels::PERMUTE_VEC_INV,
            // Triangular solves
            kernels::SPTRSV_LOWER_LEVEL,
            kernels::SPTRSV_UPPER_LEVEL,
            kernels::APPLY_DINV_INPLACE,
            // Dense supernodal
            kernels::DENSE_LDLT_BATCHED,
            kernels::DENSE_TRSM_RIGHT,
            kernels::DENSE_TRSM_RIGHT_CACHED,
            kernels::DENSE_COL_SCALE,
            kernels::DENSE_SYRK_UPDATE,
            kernels::DENSE_SYRK_UPDATE_SIMD,
            // Assembly
            kernels::GATHER,
            kernels::SCATTER_SET,
            kernels::SCATTER_ADD,
            // Debug
            kernels::CSR_TO_DENSE,
        ];

        for name in kernel_names {
            let function = library.get_function(name, None)
                .map_err(|e| MetalError::PipelineCreation {
                    kernel: name.to_string(),
                    reason: format!("Function not found: {}", e),
                })?;

            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| MetalError::PipelineCreation {
                    kernel: name.to_string(),
                    reason: e.to_string(),
                })?;

            pipelines.insert(name, pipeline);
        }

        if config.verbose {
            println!("[Metal] Created {} compute pipelines", pipelines.len());
        }

        Ok(Self {
            device,
            queue,
            library,
            pipelines,
            config,
        })
    }

    /// Get a compute pipeline by kernel name.
    pub fn pipeline(&self, name: &'static str) -> Option<&ComputePipelineState> {
        self.pipelines.get(name)
    }

    /// Get the Metal device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the command queue.
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// Get the configuration.
    pub fn config(&self) -> &DssConfig {
        &self.config
    }

    /// Check if SIMD-group matrix operations are supported.
    pub fn supports_simd_group_matrix(&self) -> bool {
        // Apple GPU family 7+ supports simdgroup_matrix
        // This is available on M1 and later
        self.device.supports_family(metal::MTLGPUFamily::Apple7)
    }

    /// Create a GPU buffer with the given size in bytes.
    pub fn create_buffer(&self, size: usize) -> MetalResult<metal::Buffer> {
        let buffer = self.device.new_buffer(
            size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        Ok(buffer)
    }

    /// Create a GPU buffer initialized with data.
    pub fn create_buffer_with_data<T: Copy>(&self, data: &[T]) -> MetalResult<metal::Buffer> {
        let size = std::mem::size_of_val(data);
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        Ok(buffer)
    }
}

/// Stub implementation for non-macOS platforms.
#[cfg(not(target_os = "macos"))]
pub struct DssHandle {
    config: DssConfig,
}

#[cfg(not(target_os = "macos"))]
impl DssHandle {
    pub fn new(config: DssConfig) -> MetalResult<Self> {
        Err(MetalError::NoDevice)
    }

    pub fn config(&self) -> &DssConfig {
        &self.config
    }
}
