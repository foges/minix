//! Cone kernel implementations.
//!
//! This module provides implementations of cone kernels (barrier functions,
//! interior tests, step-to-boundary, and scaling) for all supported cone types.

pub mod exp;
pub mod nonneg;
pub mod pow;
pub mod psd;
pub mod soc;
pub mod traits;
pub mod zero;

pub use exp::ExpCone;
pub use nonneg::NonNegCone;
pub use pow::PowCone;
pub use psd::PsdCone;
pub use soc::SocCone;
pub use traits::ConeKernel;
pub use zero::ZeroCone;
