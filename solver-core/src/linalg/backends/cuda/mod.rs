//! CUDA GPU backend for sparse direct KKT solves using cuDSS.
//!
//! This module provides GPU-accelerated sparse direct solving for NVIDIA GPUs
//! using NVIDIA's cuDSS (CUDA Direct Sparse Solver) library.
//!
//! # Architecture
//!
//! Unlike the Metal backend which implements custom kernels, this backend
//! wraps cuDSS which provides:
//!
//! 1. **Analysis**: Symbolic factorization with fill-reducing ordering
//! 2. **Factorization**: GPU-accelerated supernodal LDL^T/Cholesky
//! 3. **Solve**: GPU-accelerated triangular solves
//!
//! # Feature Gate
//!
//! This module requires the `cuda` feature and the cuDSS library:
//!
//! ```toml
//! [dependencies]
//! solver-core = { version = "...", features = ["cuda"] }
//! ```
//!
//! # Example
//!
//! ```ignore
//! use solver_core::linalg::backends::cuda::{CudaKktBackend, CudaConfig};
//!
//! let config = CudaConfig::default();
//! let mut backend = CudaKktBackend::new(config)?;
//!
//! // Analysis phase
//! backend.symbolic_factorization(&kkt_matrix)?;
//!
//! // Factorization phase
//! let factor = backend.numeric_factorization(&kkt_matrix)?;
//!
//! // Solve phase
//! backend.solve(&factor, &rhs, &mut solution);
//! ```

mod error;
mod handle;
mod backend;

pub use error::{CudaError, CudaResult};
pub use handle::{CudaHandle, CudaConfig};
pub use backend::{CudaKktBackend, CudaBackendAdapter, CudaFactorization};
