//! Conic subproblem oracle for validating integer solutions.

mod conic;
pub mod certificate;

pub use conic::{ConicOracle, OracleResult};
pub use certificate::{ConeDual, CutExtractor, DualCertificate};
