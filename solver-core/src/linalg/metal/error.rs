//! Error types for the Metal backend.

use std::fmt;

/// Result type for Metal operations.
pub type MetalResult<T> = Result<T, MetalError>;

/// Errors that can occur in the Metal backend.
#[derive(Debug)]
pub enum MetalError {
    /// No Metal device available (not on macOS/iOS or no GPU).
    NoDevice,

    /// Failed to create command queue.
    CommandQueueCreation,

    /// Failed to compile Metal shader source.
    ShaderCompilation(String),

    /// Failed to create compute pipeline for a kernel.
    PipelineCreation {
        kernel: String,
        reason: String,
    },

    /// Failed to create GPU buffer.
    BufferCreation {
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

    /// GPU command execution failed.
    CommandExecution(String),

    /// Feature not yet implemented.
    NotImplemented(String),
}

impl fmt::Display for MetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetalError::NoDevice => {
                write!(f, "No Metal device available")
            }
            MetalError::CommandQueueCreation => {
                write!(f, "Failed to create Metal command queue")
            }
            MetalError::ShaderCompilation(msg) => {
                write!(f, "Metal shader compilation failed: {}", msg)
            }
            MetalError::PipelineCreation { kernel, reason } => {
                write!(f, "Failed to create pipeline for kernel '{}': {}", kernel, reason)
            }
            MetalError::BufferCreation { size, reason } => {
                write!(f, "Failed to create GPU buffer of size {}: {}", size, reason)
            }
            MetalError::SymbolicAnalysis(msg) => {
                write!(f, "Symbolic analysis failed: {}", msg)
            }
            MetalError::NumericFactorization(msg) => {
                write!(f, "Numeric factorization failed: {}", msg)
            }
            MetalError::Solve(msg) => {
                write!(f, "Solve failed: {}", msg)
            }
            MetalError::SingularMatrix { pivot_index, pivot_value } => {
                write!(
                    f,
                    "Matrix is singular: pivot {} has value {}",
                    pivot_index, pivot_value
                )
            }
            MetalError::InvalidMatrix(msg) => {
                write!(f, "Invalid matrix structure: {}", msg)
            }
            MetalError::DimensionMismatch { expected, actual, context } => {
                write!(
                    f,
                    "Dimension mismatch in {}: expected {}, got {}",
                    context, expected, actual
                )
            }
            MetalError::CommandExecution(msg) => {
                write!(f, "GPU command execution failed: {}", msg)
            }
            MetalError::NotImplemented(msg) => {
                write!(f, "Not implemented: {}", msg)
            }
        }
    }
}

impl std::error::Error for MetalError {}

impl From<MetalError> for crate::linalg::backend::BackendError {
    fn from(e: MetalError) -> Self {
        crate::linalg::backend::BackendError::Other(e.to_string())
    }
}
