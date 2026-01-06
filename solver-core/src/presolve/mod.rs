//! Presolve and scaling.
//!
//! Problem preprocessing, Ruiz equilibration, and future chordal decomposition.

pub mod ruiz;
pub mod singleton;
pub mod bounds;
pub mod eliminate;

use crate::problem::ProblemData;
use crate::presolve::bounds::{PresolveResult, shift_bounds_and_eliminate_fixed_with_postsolve};
use crate::presolve::eliminate::eliminate_singleton_rows;

pub fn apply_presolve(prob: &ProblemData) -> PresolveResult {
    let presolved = eliminate_singleton_rows(prob);
    shift_bounds_and_eliminate_fixed_with_postsolve(&presolved.problem, presolved.postsolve)
}
