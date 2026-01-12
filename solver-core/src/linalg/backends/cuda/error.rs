//! Error types for the CUDA backend.

use std::fmt;

/// Result type for CUDA operations.
pub type CudaResult<T> = Result<T, CudaError>;

/// Errors that can occur in the CUDA backend.
#[derive(Debug)]
pub enum CudaError {
    /// No CUDA device available.
    NoDevice,

    /// CUDA driver error.
    DriverError {
        code: i32,
        message: String,
    },

    /// cuDSS error.
    CudssError {
        code: i32,
        phase: String,
    },

    /// Failed to allocate GPU memory.
    AllocationFailed {
        size: usize,
        reason: String,
    },

    /// Symbolic analysis failed.
    SymbolicAnalysis(String),

    /// Numeric factorization failed.
    NumericFactorization(String),

    /// Solve failed.
    Solve(String),

    /// Matrix is singular or nearly singular.
    SingularMatrix {
        pivot_index: usize,
        pivot_value: f64,
    },

    /// Invalid matrix structure.
    InvalidMatrix(String),

    /// Dimension mismatch.
    DimensionMismatch {
        expected: usize,
        actual: usize,
        context: String,
    },

    /// cuDSS library not available.
    LibraryNotFound(String),

    /// Feature not yet implemented.
    NotImplemented(String),
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaError::NoDevice => {
                write!(f, "No CUDA device available")
            }
            CudaError::DriverError { code, message } => {
                write!(f, "CUDA driver error {}: {}", code, message)
            }
            CudaError::CudssError { code, phase } => {
                write!(f, "cuDSS error {} during {}", code, phase)
            }
            CudaError::AllocationFailed { size, reason } => {
                write!(f, "Failed to allocate {} bytes on GPU: {}", size, reason)
            }
            CudaError::SymbolicAnalysis(msg) => {
                write!(f, "Symbolic analysis failed: {}", msg)
            }
            CudaError::NumericFactorization(msg) => {
                write!(f, "Numeric factorization failed: {}", msg)
            }
            CudaError::Solve(msg) => {
                write!(f, "Solve failed: {}", msg)
            }
            CudaError::SingularMatrix { pivot_index, pivot_value } => {
                write!(
                    f,
                    "Matrix is singular: pivot {} has value {}",
                    pivot_index, pivot_value
                )
            }
            CudaError::InvalidMatrix(msg) => {
                write!(f, "Invalid matrix structure: {}", msg)
            }
            CudaError::DimensionMismatch { expected, actual, context } => {
                write!(
                    f,
                    "Dimension mismatch in {}: expected {}, got {}",
                    context, expected, actual
                )
            }
            CudaError::LibraryNotFound(msg) => {
                write!(f, "cuDSS library not found: {}", msg)
            }
            CudaError::NotImplemented(msg) => {
                write!(f, "Not implemented: {}", msg)
            }
        }
    }
}

impl std::error::Error for CudaError {}

impl From<CudaError> for crate::linalg::backend::BackendError {
    fn from(e: CudaError) -> Self {
        crate::linalg::backend::BackendError::Other(e.to_string())
    }
}
