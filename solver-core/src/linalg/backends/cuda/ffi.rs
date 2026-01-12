//! Dynamic FFI bindings for CUDA and cuDSS.
//!
//! This module provides dynamic loading of CUDA runtime and cuDSS libraries,
//! allowing the code to compile and run on systems without CUDA installed.
//! When CUDA is not available, operations will return appropriate errors.

use super::error::{CudaError, CudaResult};
use libloading::{Library, Symbol};
use std::ffi::c_void;
use std::sync::OnceLock;

// ============================================================================
// CUDA Types
// ============================================================================

/// CUDA error codes
pub type CudaError_t = i32;

/// CUDA stream handle
pub type CudaStream_t = *mut c_void;

/// CUDA memory copy kind
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CudaMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
}

// cuDSS types
pub type CudssHandle_t = *mut c_void;
pub type CudssMatrix_t = *mut c_void;
pub type CudssData_t = *mut c_void;
pub type CudssConfig_t = *mut c_void;

/// cuDSS matrix type
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CudssMtype {
    General = 0,
    Symmetric = 1,
    Hermitian = 2,
    Spd = 3,  // Symmetric positive definite
    Hpd = 4,  // Hermitian positive definite
}

/// cuDSS matrix view (which triangle to use)
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CudssMview {
    Full = 0,
    Lower = 1,
    Upper = 2,
}

/// cuDSS index base
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CudssIndexBase {
    Zero = 0,
    One = 1,
}

/// cuDSS matrix format
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CudssLayout {
    Csr = 0,  // Compressed Sparse Row
    Csc = 1,  // Compressed Sparse Column
}

/// cuDSS phases
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CudssPhase {
    Analysis = 1,
    Factorization = 2,
    Solve = 3,
    SolveForward = 4,
    SolveDiagonal = 5,
    SolveBackward = 6,
}

/// cuDSS algorithm for factorization
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CudssAlgType {
    Default = 0,
    Chol = 1,   // Cholesky
    Ldlt = 2,   // LDL^T
    Lu = 3,     // LU
}

/// cuDSS configuration parameter
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CudssConfigParam {
    ReorderingAlg = 1,
    FactorizationAlg = 2,
    SolveAlg = 3,
    MatchingType = 4,
    SolveMode = 5,
    IrNSteps = 6,
    IrTol = 7,
    PivotType = 8,
    PivotThreshold = 9,
    PivotEpsilon = 10,
    MaxLuNnz = 11,
    // ... more params exist but these are the main ones we need
}

/// cuDSS data parameter (for getting info after operations)
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CudssDataParam {
    LuNnz = 1,
    NpivotsCorrected = 2,
    InertiaPos = 3,
    InertiaNeg = 4,
    PermReorderRow = 5,
    PermReorderCol = 6,
    DiagModified = 7,
    // ... more params
}

/// CUDA data type
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CudaDataType {
    R32F = 0,   // float
    R64F = 1,   // double
    R32I = 10,  // int32
    R64I = 11,  // int64
}

// ============================================================================
// Function pointer types
// ============================================================================

// CUDA Runtime
type CudaSetDeviceFn = unsafe extern "C" fn(device: i32) -> CudaError_t;
type CudaGetDeviceCountFn = unsafe extern "C" fn(count: *mut i32) -> CudaError_t;
type CudaStreamCreateFn = unsafe extern "C" fn(stream: *mut CudaStream_t) -> CudaError_t;
type CudaStreamDestroyFn = unsafe extern "C" fn(stream: CudaStream_t) -> CudaError_t;
type CudaStreamSynchronizeFn = unsafe extern "C" fn(stream: CudaStream_t) -> CudaError_t;
type CudaMallocFn = unsafe extern "C" fn(ptr: *mut *mut c_void, size: usize) -> CudaError_t;
type CudaFreeFn = unsafe extern "C" fn(ptr: *mut c_void) -> CudaError_t;
type CudaMemcpyFn = unsafe extern "C" fn(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: CudaMemcpyKind,
) -> CudaError_t;
type CudaMemcpyAsyncFn = unsafe extern "C" fn(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: CudaMemcpyKind,
    stream: CudaStream_t,
) -> CudaError_t;
type CudaGetErrorStringFn = unsafe extern "C" fn(error: CudaError_t) -> *const i8;

// cuDSS
type CudssCreateFn = unsafe extern "C" fn(handle: *mut CudssHandle_t) -> CudaError_t;
type CudssDestroyFn = unsafe extern "C" fn(handle: CudssHandle_t) -> CudaError_t;
type CudssSetStreamFn = unsafe extern "C" fn(handle: CudssHandle_t, stream: CudaStream_t) -> CudaError_t;

type CudssConfigCreateFn = unsafe extern "C" fn(config: *mut CudssConfig_t) -> CudaError_t;
type CudssConfigDestroyFn = unsafe extern "C" fn(config: CudssConfig_t) -> CudaError_t;
type CudssConfigSetFn = unsafe extern "C" fn(
    config: CudssConfig_t,
    param: CudssConfigParam,
    value: *const c_void,
    size: usize,
) -> CudaError_t;

type CudssDataCreateFn = unsafe extern "C" fn(handle: CudssHandle_t, data: *mut CudssData_t) -> CudaError_t;
type CudssDataDestroyFn = unsafe extern "C" fn(handle: CudssHandle_t, data: CudssData_t) -> CudaError_t;
type CudssDataGetFn = unsafe extern "C" fn(
    handle: CudssHandle_t,
    data: CudssData_t,
    param: CudssDataParam,
    value: *mut c_void,
    size: usize,
) -> CudaError_t;

type CudssMatrixCreateCsrFn = unsafe extern "C" fn(
    matrix: *mut CudssMatrix_t,
    nrows: i64,
    ncols: i64,
    nnz: i64,
    row_ptr: *mut c_void,
    row_ptr_end: *mut c_void,  // Can be null
    col_ind: *mut c_void,
    values: *mut c_void,
    row_ptr_type: CudaDataType,
    col_ind_type: CudaDataType,
    value_type: CudaDataType,
    index_base: CudssIndexBase,
    mtype: CudssMtype,
    mview: CudssMview,
) -> CudaError_t;

type CudssMatrixCreateDnFn = unsafe extern "C" fn(
    matrix: *mut CudssMatrix_t,
    nrows: i64,
    ncols: i64,
    lda: i64,
    values: *mut c_void,
    value_type: CudaDataType,
    layout: i32,  // CUDSS_LAYOUT_COL_MAJOR = 1
) -> CudaError_t;

type CudssMatrixDestroyFn = unsafe extern "C" fn(matrix: CudssMatrix_t) -> CudaError_t;

type CudssExecuteFn = unsafe extern "C" fn(
    handle: CudssHandle_t,
    phase: CudssPhase,
    config: CudssConfig_t,
    data: CudssData_t,
    matrix: CudssMatrix_t,
    solution: CudssMatrix_t,
    rhs: CudssMatrix_t,
) -> CudaError_t;

// ============================================================================
// Library wrapper
// ============================================================================

/// Dynamically loaded CUDA and cuDSS libraries.
pub struct CudaLibraries {
    _cuda_rt: Library,
    _cudss: Library,

    // CUDA Runtime functions
    pub cuda_set_device: CudaSetDeviceFn,
    pub cuda_get_device_count: CudaGetDeviceCountFn,
    pub cuda_stream_create: CudaStreamCreateFn,
    pub cuda_stream_destroy: CudaStreamDestroyFn,
    pub cuda_stream_synchronize: CudaStreamSynchronizeFn,
    pub cuda_malloc: CudaMallocFn,
    pub cuda_free: CudaFreeFn,
    pub cuda_memcpy: CudaMemcpyFn,
    pub cuda_memcpy_async: CudaMemcpyAsyncFn,
    pub cuda_get_error_string: CudaGetErrorStringFn,

    // cuDSS functions
    pub cudss_create: CudssCreateFn,
    pub cudss_destroy: CudssDestroyFn,
    pub cudss_set_stream: CudssSetStreamFn,
    pub cudss_config_create: CudssConfigCreateFn,
    pub cudss_config_destroy: CudssConfigDestroyFn,
    pub cudss_config_set: CudssConfigSetFn,
    pub cudss_data_create: CudssDataCreateFn,
    pub cudss_data_destroy: CudssDataDestroyFn,
    pub cudss_data_get: CudssDataGetFn,
    pub cudss_matrix_create_csr: CudssMatrixCreateCsrFn,
    pub cudss_matrix_create_dn: CudssMatrixCreateDnFn,
    pub cudss_matrix_destroy: CudssMatrixDestroyFn,
    pub cudss_execute: CudssExecuteFn,
}

// SAFETY: The function pointers are loaded from shared libraries that are
// designed to be called from multiple threads. CUDA handles synchronization.
unsafe impl Send for CudaLibraries {}
unsafe impl Sync for CudaLibraries {}

impl CudaLibraries {
    /// Try to load CUDA and cuDSS libraries.
    pub fn load() -> CudaResult<Self> {
        // Try different library names based on platform
        let cuda_rt_names = if cfg!(target_os = "windows") {
            vec!["cudart64_12.dll", "cudart64_11.dll", "cudart64.dll"]
        } else if cfg!(target_os = "macos") {
            vec!["libcudart.dylib"]
        } else {
            vec!["libcudart.so.12", "libcudart.so.11", "libcudart.so"]
        };

        let cudss_names = if cfg!(target_os = "windows") {
            vec!["cudss64_0.dll", "cudss64.dll"]
        } else if cfg!(target_os = "macos") {
            vec!["libcudss.dylib"]
        } else {
            vec!["libcudss.so.0", "libcudss.so"]
        };

        // Load CUDA runtime
        let cuda_rt = Self::load_library(&cuda_rt_names)
            .map_err(|_| CudaError::LibraryNotFound(
                "CUDA runtime library not found. Install CUDA toolkit.".to_string()
            ))?;

        // Load cuDSS
        let cudss = Self::load_library(&cudss_names)
            .map_err(|_| CudaError::LibraryNotFound(
                "cuDSS library not found. Install cuDSS from NVIDIA.".to_string()
            ))?;

        // Load CUDA runtime symbols - must extract raw pointers before moving libraries
        let cuda_set_device;
        let cuda_get_device_count;
        let cuda_stream_create;
        let cuda_stream_destroy;
        let cuda_stream_synchronize;
        let cuda_malloc;
        let cuda_free;
        let cuda_memcpy;
        let cuda_memcpy_async;
        let cuda_get_error_string;

        unsafe {
            cuda_set_device = *cuda_rt.get::<CudaSetDeviceFn>(b"cudaSetDevice")
                .map_err(|e| CudaError::LibraryNotFound(format!("cudaSetDevice: {}", e)))?;
            cuda_get_device_count = *cuda_rt.get::<CudaGetDeviceCountFn>(b"cudaGetDeviceCount")
                .map_err(|e| CudaError::LibraryNotFound(format!("cudaGetDeviceCount: {}", e)))?;
            cuda_stream_create = *cuda_rt.get::<CudaStreamCreateFn>(b"cudaStreamCreate")
                .map_err(|e| CudaError::LibraryNotFound(format!("cudaStreamCreate: {}", e)))?;
            cuda_stream_destroy = *cuda_rt.get::<CudaStreamDestroyFn>(b"cudaStreamDestroy")
                .map_err(|e| CudaError::LibraryNotFound(format!("cudaStreamDestroy: {}", e)))?;
            cuda_stream_synchronize = *cuda_rt.get::<CudaStreamSynchronizeFn>(b"cudaStreamSynchronize")
                .map_err(|e| CudaError::LibraryNotFound(format!("cudaStreamSynchronize: {}", e)))?;
            cuda_malloc = *cuda_rt.get::<CudaMallocFn>(b"cudaMalloc")
                .map_err(|e| CudaError::LibraryNotFound(format!("cudaMalloc: {}", e)))?;
            cuda_free = *cuda_rt.get::<CudaFreeFn>(b"cudaFree")
                .map_err(|e| CudaError::LibraryNotFound(format!("cudaFree: {}", e)))?;
            cuda_memcpy = *cuda_rt.get::<CudaMemcpyFn>(b"cudaMemcpy")
                .map_err(|e| CudaError::LibraryNotFound(format!("cudaMemcpy: {}", e)))?;
            cuda_memcpy_async = *cuda_rt.get::<CudaMemcpyAsyncFn>(b"cudaMemcpyAsync")
                .map_err(|e| CudaError::LibraryNotFound(format!("cudaMemcpyAsync: {}", e)))?;
            cuda_get_error_string = *cuda_rt.get::<CudaGetErrorStringFn>(b"cudaGetErrorString")
                .map_err(|e| CudaError::LibraryNotFound(format!("cudaGetErrorString: {}", e)))?;
        }

        // Load cuDSS symbols
        let cudss_create;
        let cudss_destroy;
        let cudss_set_stream;
        let cudss_config_create;
        let cudss_config_destroy;
        let cudss_config_set;
        let cudss_data_create;
        let cudss_data_destroy;
        let cudss_data_get;
        let cudss_matrix_create_csr;
        let cudss_matrix_create_dn;
        let cudss_matrix_destroy;
        let cudss_execute;

        unsafe {
            cudss_create = *cudss.get::<CudssCreateFn>(b"cudssCreate")
                .map_err(|e| CudaError::LibraryNotFound(format!("cudssCreate: {}", e)))?;
            cudss_destroy = *cudss.get::<CudssDestroyFn>(b"cudssDestroy")
                .map_err(|e| CudaError::LibraryNotFound(format!("cudssDestroy: {}", e)))?;
            cudss_set_stream = *cudss.get::<CudssSetStreamFn>(b"cudssSetStream")
                .map_err(|e| CudaError::LibraryNotFound(format!("cudssSetStream: {}", e)))?;
            cudss_config_create = *cudss.get::<CudssConfigCreateFn>(b"cudssConfigCreate")
                .map_err(|e| CudaError::LibraryNotFound(format!("cudssConfigCreate: {}", e)))?;
            cudss_config_destroy = *cudss.get::<CudssConfigDestroyFn>(b"cudssConfigDestroy")
                .map_err(|e| CudaError::LibraryNotFound(format!("cudssConfigDestroy: {}", e)))?;
            cudss_config_set = *cudss.get::<CudssConfigSetFn>(b"cudssConfigSet")
                .map_err(|e| CudaError::LibraryNotFound(format!("cudssConfigSet: {}", e)))?;
            cudss_data_create = *cudss.get::<CudssDataCreateFn>(b"cudssDataCreate")
                .map_err(|e| CudaError::LibraryNotFound(format!("cudssDataCreate: {}", e)))?;
            cudss_data_destroy = *cudss.get::<CudssDataDestroyFn>(b"cudssDataDestroy")
                .map_err(|e| CudaError::LibraryNotFound(format!("cudssDataDestroy: {}", e)))?;
            cudss_data_get = *cudss.get::<CudssDataGetFn>(b"cudssDataGet")
                .map_err(|e| CudaError::LibraryNotFound(format!("cudssDataGet: {}", e)))?;
            cudss_matrix_create_csr = *cudss.get::<CudssMatrixCreateCsrFn>(b"cudssMatrixCreateCsr")
                .map_err(|e| CudaError::LibraryNotFound(format!("cudssMatrixCreateCsr: {}", e)))?;
            cudss_matrix_create_dn = *cudss.get::<CudssMatrixCreateDnFn>(b"cudssMatrixCreateDn")
                .map_err(|e| CudaError::LibraryNotFound(format!("cudssMatrixCreateDn: {}", e)))?;
            cudss_matrix_destroy = *cudss.get::<CudssMatrixDestroyFn>(b"cudssMatrixDestroy")
                .map_err(|e| CudaError::LibraryNotFound(format!("cudssMatrixDestroy: {}", e)))?;
            cudss_execute = *cudss.get::<CudssExecuteFn>(b"cudssExecute")
                .map_err(|e| CudaError::LibraryNotFound(format!("cudssExecute: {}", e)))?;
        }

        Ok(Self {
            _cuda_rt: cuda_rt,
            _cudss: cudss,
            cuda_set_device,
            cuda_get_device_count,
            cuda_stream_create,
            cuda_stream_destroy,
            cuda_stream_synchronize,
            cuda_malloc,
            cuda_free,
            cuda_memcpy,
            cuda_memcpy_async,
            cuda_get_error_string,
            cudss_create,
            cudss_destroy,
            cudss_set_stream,
            cudss_config_create,
            cudss_config_destroy,
            cudss_config_set,
            cudss_data_create,
            cudss_data_destroy,
            cudss_data_get,
            cudss_matrix_create_csr,
            cudss_matrix_create_dn,
            cudss_matrix_destroy,
            cudss_execute,
        })
    }

    fn load_library(names: &[&str]) -> Result<Library, libloading::Error> {
        let mut last_error = None;
        for name in names {
            match unsafe { Library::new(name) } {
                Ok(lib) => return Ok(lib),
                Err(e) => last_error = Some(e),
            }
        }
        Err(last_error.unwrap())
    }
}

// ============================================================================
// Global library instance
// ============================================================================

static CUDA_LIBS: OnceLock<CudaResult<CudaLibraries>> = OnceLock::new();

/// Get the global CUDA libraries instance.
///
/// This lazily loads the CUDA and cuDSS libraries on first call.
/// Returns an error if the libraries are not available.
pub fn get_cuda_libs() -> CudaResult<&'static CudaLibraries> {
    CUDA_LIBS.get_or_init(CudaLibraries::load)
        .as_ref()
        .map_err(|e| CudaError::LibraryNotFound(e.to_string()))
}

/// Check if CUDA is available without initializing.
///
/// This is a quick check that just tries to load the libraries.
pub fn is_cuda_available() -> bool {
    get_cuda_libs().is_ok()
}

// ============================================================================
// Helper functions
// ============================================================================

/// Check CUDA error and convert to CudaResult.
pub fn check_cuda(libs: &CudaLibraries, err: CudaError_t, context: &str) -> CudaResult<()> {
    if err == 0 {
        Ok(())
    } else {
        let msg = unsafe {
            let ptr = (libs.cuda_get_error_string)(err);
            if ptr.is_null() {
                format!("Unknown error {}", err)
            } else {
                std::ffi::CStr::from_ptr(ptr)
                    .to_string_lossy()
                    .into_owned()
            }
        };
        Err(CudaError::DriverError {
            code: err,
            message: format!("{}: {}", context, msg),
        })
    }
}

/// Check cuDSS error and convert to CudaResult.
pub fn check_cudss(err: CudaError_t, phase: &str) -> CudaResult<()> {
    if err == 0 {
        Ok(())
    } else {
        Err(CudaError::CudssError {
            code: err,
            phase: phase.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_availability() {
        // This test just checks that the availability check works
        // It will return false on systems without CUDA, which is fine
        let available = is_cuda_available();
        println!("CUDA available: {}", available);
    }
}
