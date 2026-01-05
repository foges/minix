//! Main IPM solver module (ipm2).
//!
//! Implements a predictor-corrector interior point method with:
//! - HSDE (Homogeneous Self-Dual Embedding) formulation
//! - Ruiz equilibration for problem scaling
//! - NT (Nesterov-Todd) scaling for cone operations
//! - Active-set polishing for bound-heavy QP problems
#![allow(missing_docs)]

pub mod diagnostics;
pub mod metrics;
pub mod modes;
pub mod polish;
pub mod predcorr;
pub mod perf;
pub mod regularization;
pub mod solve;
pub mod workspace;

pub use diagnostics::DiagnosticsConfig;
pub use metrics::{UnscaledMetrics, compute_unscaled_metrics};
pub use modes::{SolveMode, StallDetector};
pub use polish::polish_nonneg_active_set;
pub use perf::{PerfSection, PerfTimers};
pub use regularization::{RegularizationPolicy, RegularizationState};
pub use solve::solve_ipm2;
pub use workspace::IpmWorkspace;
