//! Linear algebra layer.
//!
//! Sparse matrix operations, KKT system building, and factorization backends.

pub mod sparse;
pub mod kkt;
pub mod kkt_trait;
pub mod backend;
pub mod backends;
pub mod qdldl;
pub mod normal_eqns;
pub mod unified_kkt;

/// Metal GPU backend for Apple Silicon.
/// Only available on macOS with the `metal` feature enabled.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metal;

pub use backend::BackendError;
