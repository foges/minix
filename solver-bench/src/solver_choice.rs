use clap::ValueEnum;
use solver_core::{solve, ProblemData, SolveResult, SolverSettings};
use solver_core::ipm2::solve_ipm2;

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum SolverChoice {
    Ipm1,
    Ipm2,
}

pub fn solve_with_choice(
    prob: &ProblemData,
    settings: &SolverSettings,
    choice: SolverChoice,
) -> Result<SolveResult, Box<dyn std::error::Error>> {
    match choice {
        SolverChoice::Ipm1 => solve(prob, settings),
        SolverChoice::Ipm2 => solve_ipm2(prob, settings),
    }
}
