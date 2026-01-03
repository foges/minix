//! Error types for the MIP solver.

use thiserror::Error;

/// Errors that can occur during MIP solving.
#[derive(Error, Debug)]
pub enum MipError {
    /// Problem validation failed
    #[error("Invalid problem: {0}")]
    InvalidProblem(String),

    /// Master LP/QP solve failed
    #[error("Master solve failed: {0}")]
    MasterSolveError(String),

    /// Conic oracle (subproblem) failed
    #[error("Oracle failed: {0}")]
    OracleError(String),

    /// Numerical issues in cut generation
    #[error("Cut generation failed: {0}")]
    CutGenerationError(String),

    /// Internal solver error
    #[error("Internal error: {0}")]
    InternalError(String),

    /// Time limit exceeded
    #[error("Time limit exceeded")]
    TimeLimit,

    /// Node limit exceeded
    #[error("Node limit exceeded")]
    NodeLimit,

    /// Solver-core error
    #[error("Solver core error: {0}")]
    SolverCore(#[from] Box<dyn std::error::Error + Send + Sync>),
}

/// Result type for MIP operations.
pub type MipResult<T> = Result<T, MipError>;
