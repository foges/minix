//! Cone kernel implementations.
//!
//! This module provides implementations of cone kernels (barrier functions,
//! interior tests, step-to-boundary, and scaling) for all supported cone types.

pub mod traits;
pub mod zero;
pub mod nonneg;
pub mod soc;
pub mod exp;
pub mod pow;
pub mod psd;

pub use traits::ConeKernel;
pub use zero::ZeroCone;
pub use nonneg::NonNegCone;
pub use soc::SocCone;
pub use exp::ExpCone;
pub use pow::PowCone;
pub use psd::PsdCone;
