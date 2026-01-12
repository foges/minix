//! CUDA handle and cuDSS configuration.
//!
//! This module provides the core infrastructure for interacting with CUDA and cuDSS:
//! - `CudaHandle`: Manages CUDA context, stream, and cuDSS handles.
//! - `CudaConfig`: Configuration options for the solver.

use super::error::{CudaError, CudaResult};
use super::ffi::{
    self, CudaLibraries, CudaStream_t, CudssHandle_t, CudssConfig_t, CudssData_t,
    CudaMemcpyKind, CudssConfigParam, CudssAlgType, check_cuda, check_cudss,
};
use std::ffi::c_void;
use std::ptr;

/// Configuration for the CUDA sparse direct solver.
#[derive(Debug, Clone)]
pub struct CudaConfig {
    /// CUDA device index (0 for default GPU).
    pub device_index: i32,

    /// Matrix type for cuDSS.
    pub matrix_type: MatrixType,

    /// Ordering method for fill reduction.
    pub ordering: CudaOrdering,

    /// Use mixed precision (FP32 factorization, FP64 refinement).
    pub mixed_precision: bool,

    /// Number of iterative refinement steps.
    pub refine_iters: usize,

    /// Pivot tolerance for numerical stability.
    pub pivot_tolerance: f64,

    /// Enable verbose logging.
    pub verbose: bool,
}

impl Default for CudaConfig {
    fn default() -> Self {
        Self {
            device_index: 0,
            matrix_type: MatrixType::SymmetricIndefinite,
            ordering: CudaOrdering::Metis,
            mixed_precision: false,
            refine_iters: 2,
            pivot_tolerance: 1e-12,
            verbose: false,
        }
    }
}

/// Matrix type for cuDSS solver selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixType {
    /// Symmetric positive definite (Cholesky).
    SymmetricPositiveDefinite,
    /// Symmetric indefinite (LDL^T with pivoting).
    SymmetricIndefinite,
    /// General unsymmetric (LU).
    General,
}

/// Fill-reducing ordering method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaOrdering {
    /// No reordering.
    Natural,
    /// Approximate Minimum Degree.
    Amd,
    /// METIS nested dissection (best for large problems).
    Metis,
    /// Let cuDSS choose automatically.
    Auto,
}

/// Handle to CUDA context and cuDSS library.
///
/// This manages the CUDA device, stream, and cuDSS handles.
/// Create once and reuse across multiple solves.
pub struct CudaHandle {
    /// Reference to loaded CUDA libraries.
    libs: &'static CudaLibraries,

    /// CUDA device index.
    device_index: i32,

    /// CUDA stream for async operations.
    stream: CudaStream_t,

    /// cuDSS handle.
    cudss_handle: CudssHandle_t,

    /// cuDSS configuration.
    cudss_config: CudssConfig_t,

    /// cuDSS data object (stores factorization state).
    cudss_data: CudssData_t,

    /// Configuration.
    config: CudaConfig,

    /// Whether cuDSS is initialized.
    initialized: bool,
}

impl CudaHandle {
    /// Create a new CUDA handle with the given configuration.
    ///
    /// This will:
    /// 1. Load CUDA and cuDSS libraries dynamically
    /// 2. Initialize CUDA on the specified device
    /// 3. Create a CUDA stream
    /// 4. Initialize cuDSS
    pub fn new(config: CudaConfig) -> CudaResult<Self> {
        // Load CUDA libraries dynamically
        let libs = ffi::get_cuda_libs()?;

        // Check device count
        let mut device_count: i32 = 0;
        unsafe {
            let err = (libs.cuda_get_device_count)(&mut device_count);
            check_cuda(libs, err, "cudaGetDeviceCount")?;
        }

        if device_count == 0 {
            return Err(CudaError::NoDevice);
        }

        if config.device_index >= device_count {
            return Err(CudaError::NoDevice);
        }

        // Set device
        unsafe {
            let err = (libs.cuda_set_device)(config.device_index);
            check_cuda(libs, err, "cudaSetDevice")?;
        }

        if config.verbose {
            println!("[CUDA] Using device {}", config.device_index);
        }

        // Create CUDA stream
        let mut stream: CudaStream_t = ptr::null_mut();
        unsafe {
            let err = (libs.cuda_stream_create)(&mut stream);
            check_cuda(libs, err, "cudaStreamCreate")?;
        }

        // Create cuDSS handle
        let mut cudss_handle: CudssHandle_t = ptr::null_mut();
        unsafe {
            let err = (libs.cudss_create)(&mut cudss_handle);
            check_cudss(err, "cudssCreate")?;
        }

        // Set stream for cuDSS
        unsafe {
            let err = (libs.cudss_set_stream)(cudss_handle, stream);
            check_cudss(err, "cudssSetStream")?;
        }

        // Create cuDSS config
        let mut cudss_config: CudssConfig_t = ptr::null_mut();
        unsafe {
            let err = (libs.cudss_config_create)(&mut cudss_config);
            check_cudss(err, "cudssConfigCreate")?;
        }

        // Configure algorithm based on matrix type
        let alg_type = match config.matrix_type {
            MatrixType::SymmetricPositiveDefinite => CudssAlgType::Chol,
            MatrixType::SymmetricIndefinite => CudssAlgType::Ldlt,
            MatrixType::General => CudssAlgType::Lu,
        };

        unsafe {
            let alg_val = alg_type as i32;
            let err = (libs.cudss_config_set)(
                cudss_config,
                CudssConfigParam::FactorizationAlg,
                &alg_val as *const i32 as *const c_void,
                std::mem::size_of::<i32>(),
            );
            check_cudss(err, "cudssConfigSet(FactorizationAlg)")?;

            // Set number of iterative refinement steps
            let ir_steps = config.refine_iters as i32;
            let err = (libs.cudss_config_set)(
                cudss_config,
                CudssConfigParam::IrNSteps,
                &ir_steps as *const i32 as *const c_void,
                std::mem::size_of::<i32>(),
            );
            check_cudss(err, "cudssConfigSet(IrNSteps)")?;
        }

        // Create cuDSS data object
        let mut cudss_data: CudssData_t = ptr::null_mut();
        unsafe {
            let err = (libs.cudss_data_create)(cudss_handle, &mut cudss_data);
            check_cudss(err, "cudssDataCreate")?;
        }

        if config.verbose {
            println!("[CUDA] cuDSS initialized successfully");
        }

        Ok(Self {
            libs,
            device_index: config.device_index,
            stream,
            cudss_handle,
            cudss_config,
            cudss_data,
            config,
            initialized: true,
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &CudaConfig {
        &self.config
    }

    /// Check if cuDSS is properly initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get the cuDSS handle.
    pub fn cudss_handle(&self) -> CudssHandle_t {
        self.cudss_handle
    }

    /// Get the cuDSS config.
    pub fn cudss_config(&self) -> CudssConfig_t {
        self.cudss_config
    }

    /// Get the cuDSS data object.
    pub fn cudss_data(&self) -> CudssData_t {
        self.cudss_data
    }

    /// Get the CUDA stream.
    pub fn stream(&self) -> CudaStream_t {
        self.stream
    }

    /// Get the loaded libraries.
    pub fn libs(&self) -> &'static CudaLibraries {
        self.libs
    }

    /// Allocate GPU memory.
    pub fn allocate(&self, size: usize) -> CudaResult<GpuBuffer> {
        let mut ptr: *mut c_void = ptr::null_mut();
        unsafe {
            let err = (self.libs.cuda_malloc)(&mut ptr, size);
            check_cuda(self.libs, err, "cudaMalloc")?;
        }

        Ok(GpuBuffer {
            ptr,
            size,
            libs: self.libs,
        })
    }

    /// Copy data from host to device.
    pub fn upload<T: Copy>(&self, dst: &GpuBuffer, src: &[T]) -> CudaResult<()> {
        let byte_size = src.len() * std::mem::size_of::<T>();
        if byte_size > dst.size {
            return Err(CudaError::AllocationFailed {
                size: byte_size,
                reason: format!("Buffer too small: {} < {}", dst.size, byte_size),
            });
        }

        unsafe {
            let err = (self.libs.cuda_memcpy)(
                dst.ptr,
                src.as_ptr() as *const c_void,
                byte_size,
                CudaMemcpyKind::HostToDevice,
            );
            check_cuda(self.libs, err, "cudaMemcpy(HtoD)")?;
        }

        Ok(())
    }

    /// Copy data from host to device asynchronously.
    pub fn upload_async<T: Copy>(&self, dst: &GpuBuffer, src: &[T]) -> CudaResult<()> {
        let byte_size = src.len() * std::mem::size_of::<T>();
        if byte_size > dst.size {
            return Err(CudaError::AllocationFailed {
                size: byte_size,
                reason: format!("Buffer too small: {} < {}", dst.size, byte_size),
            });
        }

        unsafe {
            let err = (self.libs.cuda_memcpy_async)(
                dst.ptr,
                src.as_ptr() as *const c_void,
                byte_size,
                CudaMemcpyKind::HostToDevice,
                self.stream,
            );
            check_cuda(self.libs, err, "cudaMemcpyAsync(HtoD)")?;
        }

        Ok(())
    }

    /// Copy data from device to host.
    pub fn download<T: Copy>(&self, dst: &mut [T], src: &GpuBuffer) -> CudaResult<()> {
        let byte_size = dst.len() * std::mem::size_of::<T>();
        if byte_size > src.size {
            return Err(CudaError::DimensionMismatch {
                expected: src.size,
                actual: byte_size,
                context: "download".to_string(),
            });
        }

        unsafe {
            let err = (self.libs.cuda_memcpy)(
                dst.as_mut_ptr() as *mut c_void,
                src.ptr,
                byte_size,
                CudaMemcpyKind::DeviceToHost,
            );
            check_cuda(self.libs, err, "cudaMemcpy(DtoH)")?;
        }

        Ok(())
    }

    /// Copy data from device to host asynchronously.
    pub fn download_async<T: Copy>(&self, dst: &mut [T], src: &GpuBuffer) -> CudaResult<()> {
        let byte_size = dst.len() * std::mem::size_of::<T>();
        if byte_size > src.size {
            return Err(CudaError::DimensionMismatch {
                expected: src.size,
                actual: byte_size,
                context: "download_async".to_string(),
            });
        }

        unsafe {
            let err = (self.libs.cuda_memcpy_async)(
                dst.as_mut_ptr() as *mut c_void,
                src.ptr,
                byte_size,
                CudaMemcpyKind::DeviceToHost,
                self.stream,
            );
            check_cuda(self.libs, err, "cudaMemcpyAsync(DtoH)")?;
        }

        Ok(())
    }

    /// Synchronize the CUDA stream.
    pub fn synchronize(&self) -> CudaResult<()> {
        unsafe {
            let err = (self.libs.cuda_stream_synchronize)(self.stream);
            check_cuda(self.libs, err, "cudaStreamSynchronize")?;
        }
        Ok(())
    }
}

impl Drop for CudaHandle {
    fn drop(&mut self) {
        if self.initialized {
            unsafe {
                // Destroy cuDSS data
                if !self.cudss_data.is_null() {
                    let _ = (self.libs.cudss_data_destroy)(self.cudss_handle, self.cudss_data);
                }

                // Destroy cuDSS config
                if !self.cudss_config.is_null() {
                    let _ = (self.libs.cudss_config_destroy)(self.cudss_config);
                }

                // Destroy cuDSS handle
                if !self.cudss_handle.is_null() {
                    let _ = (self.libs.cudss_destroy)(self.cudss_handle);
                }

                // Destroy CUDA stream
                if !self.stream.is_null() {
                    let _ = (self.libs.cuda_stream_destroy)(self.stream);
                }
            }
        }
    }
}

/// GPU memory buffer.
pub struct GpuBuffer {
    ptr: *mut c_void,
    size: usize,
    libs: &'static CudaLibraries,
}

impl GpuBuffer {
    /// Get the raw pointer.
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Get the buffer size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = (self.libs.cuda_free)(self.ptr);
            }
        }
    }
}

// SAFETY: GPU buffers can be sent between threads (operations are synchronized via stream)
unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}
