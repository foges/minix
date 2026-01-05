//! Linear algebra layer.
//!
//! Sparse matrix operations, KKT system building, and factorization backends.

pub mod sparse;
pub mod kkt;
pub mod backend;
pub mod backends;
pub mod qdldl;
pub mod normal_eqns;
pub mod unified_kkt;
