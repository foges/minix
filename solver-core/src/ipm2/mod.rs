//! Experimental "ipm2" module: staging ground for MOSEK-close refactors.
//!
//! This module is intentionally additive: it can live side-by-side with the current ipm.
//! Wire it in via a feature flag or a settings knob so you can A/B against the existing solver.
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
