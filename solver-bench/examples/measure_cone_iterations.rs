// Standalone example - copy test_problems logic here since examples can't import from bin crates

use solver_core::SolverSettings;

#[path = "../src/test_problems.rs"]
mod test_problems;

fn main() {
    let mut settings = SolverSettings::default();
    settings.max_iter = 200;

    println!("Measuring exact iteration counts for cone problems (max_iter=200)...\n");

    for prob in test_problems::synthetic_test_problems() {
        let problem_data = (prob.builder)();
        match solver_core::solve(&problem_data, &settings) {
            Ok(res) => {
                println!("\"{}\" => Some({}),  // {:?}",
                    prob.name, res.info.iters, res.status);
            }
            Err(e) => {
                println!("\"{}\" => None,  // ERROR: {}", prob.name, e);
            }
        }
    }
}
