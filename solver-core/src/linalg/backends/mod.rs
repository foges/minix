#[cfg(feature = "suitesparse-ldl")]
mod suitesparse_ldl;

#[cfg(feature = "suitesparse-ldl")]
pub use suitesparse_ldl::SuiteSparseLdlBackend;

// Re-export Metal backend when available
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use crate::linalg::metal::{MetalBackendAdapter, MetalKktBackend, DssConfig};

// CUDA backend (when enabled)
#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::{CudaBackendAdapter, CudaKktBackend, CudaConfig};
