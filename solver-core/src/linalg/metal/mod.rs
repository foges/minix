//! Metal GPU backend for sparse direct KKT solves.
//!
//! This module provides a cuDSS-like sparse direct solver for Apple GPUs using Metal.
//! It implements the `KktBackend` trait for integration with minix's IPM solver.
//!
//! # Architecture
//!
//! The solver follows a three-phase design (mirroring cuDSS):
//!
//! 1. **Analysis (symbolic)**: Ordering, elimination tree, supernodes, level scheduling.
//!    Runs once per sparsity pattern. Mostly CPU work.
//!
//! 2. **Factorization (numeric)**: LDL^T factorization using supernodal dense kernels.
//!    Runs when matrix values change. GPU-heavy.
//!
//! 3. **Solve**: Forward/backward triangular solves with level scheduling.
//!    Runs per RHS. GPU-heavy.
//!
//! # Feature Gate
//!
//! This module is only available when the `metal` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! solver-core = { version = "...", features = ["metal"] }
//! ```
//!
//! # Example
//!
//! ```ignore
//! use solver_core::linalg::metal::{MetalKktBackend, DssConfig};
//!
//! let config = DssConfig::default();
//! let mut backend = MetalKktBackend::new(config)?;
//!
//! // Analysis phase (once per sparsity pattern)
//! backend.symbolic_factorization(&kkt_matrix)?;
//!
//! // Factorization phase (when values change)
//! let factor = backend.numeric_factorization(&kkt_matrix)?;
//!
//! // Solve phase (per RHS)
//! backend.solve(&factor, &rhs, &mut solution);
//! ```

mod error;
mod handle;
mod symbolic;
mod backend;

pub use error::{MetalError, MetalResult};
pub use handle::{DssHandle, DssConfig};
pub use symbolic::{SymbolicAnalysis, EliminationTree, Supernode, LevelSchedule};
pub use backend::{MetalKktBackend, MetalBackendAdapter, MetalFactorization, SolverStats};

/// Embedded Metal shader source.
/// Compiled at runtime using `new_library_with_source`.
pub const METAL_SHADER_SOURCE: &str = include_str!("kkt_all_kernels.metal");

/// Kernel names for pipeline creation.
pub mod kernels {
    // Sparse kernels
    pub const CSR_SPMV: &str = "csr_spmv_f32";
    pub const CSR_SPMV_ADD: &str = "csr_spmv_add_f32";
    pub const CSR_ROW_SUMSQUARES: &str = "csr_row_sumsquares_f32";

    // Reductions
    pub const DOT_PARTIAL: &str = "dot_partial_f32";
    pub const REDUCE_SUM_PARTIAL: &str = "reduce_sum_partial_f32";

    // Vector ops
    pub const VEC_ADD: &str = "vec_add_f32";
    pub const VEC_SUB: &str = "vec_sub_f32";
    pub const VEC_COPY: &str = "vec_copy_f32";
    pub const VEC_SET: &str = "vec_set_f32";
    pub const VEC_SCALE_INPLACE: &str = "vec_scale_inplace_f32";
    pub const VEC_AXPY_INPLACE: &str = "vec_axpy_inplace_f32";
    pub const VEC_XPAY_INPLACE: &str = "vec_xpay_inplace_f32";

    // Cone projections
    pub const PROJ_NONNEG_INPLACE: &str = "proj_nonneg_inplace_f32";
    pub const PROJ_SOC_INPLACE: &str = "proj_soc_inplace_f32";

    // Permutations
    pub const PERMUTE_GATHER: &str = "permute_gather_f32";
    pub const PERMUTE_SCATTER: &str = "permute_scatter_f32";
    pub const PERMUTE_VEC: &str = "permute_vec_f32";
    pub const PERMUTE_VEC_INV: &str = "permute_vec_inv_f32";

    // Triangular solves (level-scheduled)
    pub const SPTRSV_LOWER_LEVEL: &str = "sptrsv_lower_level_f32";
    pub const SPTRSV_UPPER_LEVEL: &str = "sptrsv_upper_level_f32";
    pub const APPLY_DINV_INPLACE: &str = "apply_dinv_inplace_f32";

    // Dense supernodal kernels
    pub const DENSE_LDLT_BATCHED: &str = "dense_ldlt_nopivot_inplace_f32_batched";
    pub const DENSE_TRSM_RIGHT: &str = "dense_trsm_right_unit_upper_from_unit_lower_f32";
    pub const DENSE_TRSM_RIGHT_CACHED: &str = "dense_trsm_right_unit_upper_from_unit_lower_f32_cached";
    pub const DENSE_COL_SCALE: &str = "dense_col_scale_f32";
    pub const DENSE_SYRK_UPDATE: &str = "dense_syrk_ldlt_update_f32";
    pub const DENSE_SYRK_UPDATE_SIMD: &str = "dense_syrk_ldlt_update_f32_simdgroup64";

    // Assembly helpers
    pub const GATHER: &str = "gather_f32";
    pub const SCATTER_SET: &str = "scatter_set_f32";
    pub const SCATTER_ADD: &str = "scatter_add_f32";

    // Debug
    pub const CSR_TO_DENSE: &str = "csr_to_dense_f32";
}
