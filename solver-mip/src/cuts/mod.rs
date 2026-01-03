//! Cut generation for conic outer approximation.
//!
//! This module provides cut generators for the OA algorithm:
//! - K* certificate cuts from dual cones
//! - SOC tangent cuts for direct approximation
//! - Cut pool management
//! - Per-cone-block disaggregation

pub mod disaggregation;
pub mod kstar;
mod pool;
mod soc;

pub use disaggregation::{ConeAnalyzer, ConeBlock, LiftedDisaggregation};
pub use kstar::{KStarCutGenerator, KStarSettings};
pub use pool::{CutPool, CutPoolSettings, CutStatus, PooledCut};
pub use soc::{SocTangentGenerator, SocTangentSettings};
