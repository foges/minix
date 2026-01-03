//! Master problem (LP/QP relaxation) management.

mod backend;
mod ipm_backend;

pub use backend::{CutSource, LinearCut, MasterBackend, MasterResult, MasterStatus};
pub use ipm_backend::IpmMasterBackend;
