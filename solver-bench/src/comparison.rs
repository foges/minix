//! Multi-solver comparison and win matrix generation.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use solver_core::SolveStatus;

use crate::maros_meszaros::{BenchmarkResult, BenchmarkSummary};

/// Results from a single solver run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverResults {
    /// Solver name
    pub solver_name: String,
    /// Results for each problem
    pub results: Vec<BenchmarkResult>,
    /// Summary statistics
    pub summary: BenchmarkSummary,
}

impl SolverResults {
    /// Create from a list of benchmark results
    pub fn new(solver_name: String, results: Vec<BenchmarkResult>) -> Self {
        let summary = crate::maros_meszaros::compute_summary(&results);
        Self {
            solver_name,
            results,
            summary,
        }
    }

    /// Save to JSON file
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path.as_ref())
            .with_context(|| format!("Failed to create file {}", path.as_ref().display()))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)
            .with_context(|| format!("Failed to write JSON to {}", path.as_ref().display()))?;
        Ok(())
    }

    /// Load from JSON file
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open file {}", path.as_ref().display()))?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader)
            .with_context(|| format!("Failed to parse JSON from {}", path.as_ref().display()))
    }
}

/// Comparison between multiple solvers
pub struct SolverComparison {
    /// Results from each solver
    pub solvers: Vec<SolverResults>,
}

impl SolverComparison {
    /// Create a new comparison
    pub fn new(solvers: Vec<SolverResults>) -> Self {
        Self { solvers }
    }

    /// Get all problem names that appear in any solver results
    fn all_problems(&self) -> HashSet<String> {
        let mut problems = HashSet::new();
        for solver in &self.solvers {
            for result in &solver.results {
                problems.insert(result.name.clone());
            }
        }
        problems
    }

    /// Build a map from problem name to result for a solver
    fn problem_map(results: &[BenchmarkResult]) -> HashMap<String, &BenchmarkResult> {
        results.iter().map(|r| (r.name.clone(), r)).collect()
    }

    /// Check if a result is considered "solved" (Optimal or AlmostOptimal)
    fn is_solved(status: SolveStatus) -> bool {
        matches!(status, SolveStatus::Optimal | SolveStatus::AlmostOptimal)
    }

    /// Print summary table comparing all solvers
    pub fn print_summary(&self) {
        println!("\n{}", "=".repeat(80));
        println!("Multi-Solver Comparison Summary");
        println!("{}", "=".repeat(80));

        println!(
            "\n{:<20} {:>10} {:>10} {:>10} {:>15} {:>15}",
            "Solver", "Optimal", "AlmostOpt", "Combined", "Geom Mean Time", "Pass Rate"
        );
        println!("{}", "-".repeat(80));

        for solver in &self.solvers {
            let combined = solver.summary.optimal + solver.summary.almost_optimal;
            let pass_rate = 100.0 * combined as f64 / solver.summary.total as f64;

            println!(
                "{:<20} {:>10} {:>10} {:>10} {:>15.2}ms {:>14.1}%",
                solver.solver_name,
                solver.summary.optimal,
                solver.summary.almost_optimal,
                combined,
                solver.summary.geom_mean_time_ms,
                pass_rate
            );
        }

        println!("{}", "=".repeat(80));
    }

    /// Print win matrix showing head-to-head problem solving
    pub fn print_win_matrix(&self) {
        println!("\n{}", "=".repeat(80));
        println!("Win Matrix: Problems Solved by Each Solver Pair");
        println!("{}", "=".repeat(80));
        println!("Format: A vs B shows (problems A solves that B doesn't | both solve | B solves that A doesn't)");
        println!("{}", "-".repeat(80));

        let n = self.solvers.len();

        // Print header
        print!("\n{:<20}", "");
        for solver in &self.solvers {
            print!(" {:>20}", &solver.solver_name[..solver.solver_name.len().min(20)]);
        }
        println!();
        println!("{}", "-".repeat(20 + n * 21));

        // For each row solver
        for (i, solver_a) in self.solvers.iter().enumerate() {
            print!("{:<20}", &solver_a.solver_name[..solver_a.solver_name.len().min(20)]);

            let map_a = Self::problem_map(&solver_a.results);

            // For each column solver
            for (j, solver_b) in self.solvers.iter().enumerate() {
                if i == j {
                    // Diagonal: show total solved
                    let solved = solver_a.summary.optimal + solver_a.summary.almost_optimal;
                    print!(" {:>20}", format!("({} solved)", solved));
                } else {
                    let map_b = Self::problem_map(&solver_b.results);

                    let mut a_only = 0;
                    let mut both = 0;
                    let mut b_only = 0;

                    for problem in self.all_problems() {
                        let a_solved = map_a
                            .get(&problem)
                            .map(|r| Self::is_solved(r.status))
                            .unwrap_or(false);
                        let b_solved = map_b
                            .get(&problem)
                            .map(|r| Self::is_solved(r.status))
                            .unwrap_or(false);

                        match (a_solved, b_solved) {
                            (true, false) => a_only += 1,
                            (true, true) => both += 1,
                            (false, true) => b_only += 1,
                            (false, false) => {}
                        }
                    }

                    print!(" {:>20}", format!("{}|{}|{}", a_only, both, b_only));
                }
            }
            println!();
        }

        println!("{}", "=".repeat(80));
    }

    /// Print performance comparison on commonly solved problems
    pub fn print_performance_comparison(&self) {
        println!("\n{}", "=".repeat(80));
        println!("Performance Comparison (Geometric Mean Time on Commonly Solved Problems)");
        println!("{}", "=".repeat(80));

        let n = self.solvers.len();

        // For each pair of solvers
        for i in 0..n {
            for j in (i + 1)..n {
                let solver_a = &self.solvers[i];
                let solver_b = &self.solvers[j];

                let map_a = Self::problem_map(&solver_a.results);
                let map_b = Self::problem_map(&solver_b.results);

                // Find commonly solved problems
                let mut common_times_a = Vec::new();
                let mut common_times_b = Vec::new();

                for problem in self.all_problems() {
                    if let (Some(res_a), Some(res_b)) = (map_a.get(&problem), map_b.get(&problem)) {
                        if Self::is_solved(res_a.status) && Self::is_solved(res_b.status) {
                            common_times_a.push(res_a.solve_time_ms);
                            common_times_b.push(res_b.solve_time_ms);
                        }
                    }
                }

                if common_times_a.is_empty() {
                    println!(
                        "\n{} vs {}: No commonly solved problems",
                        solver_a.solver_name, solver_b.solver_name
                    );
                    continue;
                }

                // Compute shifted geometric mean
                let geom_mean_a = Self::geom_mean(&common_times_a);
                let geom_mean_b = Self::geom_mean(&common_times_b);

                println!(
                    "\n{} vs {} ({} common problems):",
                    solver_a.solver_name, solver_b.solver_name, common_times_a.len()
                );
                println!("  {:<20} {:.2}ms", solver_a.solver_name, geom_mean_a);
                println!("  {:<20} {:.2}ms", solver_b.solver_name, geom_mean_b);

                if geom_mean_a < geom_mean_b {
                    println!("  {} is {:.2}x faster", solver_a.solver_name, geom_mean_b / geom_mean_a);
                } else if geom_mean_b < geom_mean_a {
                    println!("  {} is {:.2}x faster", solver_b.solver_name, geom_mean_a / geom_mean_b);
                } else {
                    println!("  Same performance");
                }
            }
        }

        println!("{}", "=".repeat(80));
    }

    /// Compute shifted geometric mean
    fn geom_mean(times: &[f64]) -> f64 {
        if times.is_empty() {
            return 0.0;
        }

        let log_sum: f64 = times.iter().map(|&t| (t + 1.0).ln()).sum();
        (log_sum / times.len() as f64).exp() - 1.0
    }

    /// Print detailed problem-by-problem comparison
    pub fn print_detailed_comparison(&self, limit: Option<usize>) {
        println!("\n{}", "=".repeat(100));
        println!("Detailed Problem-by-Problem Comparison");
        println!("{}", "=".repeat(100));

        let problems: Vec<String> = self.all_problems().into_iter().collect();
        let mut problems = problems;
        problems.sort();

        // Build maps for each solver
        let solver_maps: Vec<HashMap<String, &BenchmarkResult>> =
            self.solvers.iter().map(|s| Self::problem_map(&s.results)).collect();

        // Print header
        print!("\n{:<15}", "Problem");
        for solver in &self.solvers {
            print!(" {:>15}", &solver.solver_name[..solver.solver_name.len().min(15)]);
        }
        println!();

        print!("{:<15}", "");
        for _ in &self.solvers {
            print!(" {:>7} {:>7}", "Status", "Time(ms)");
        }
        println!();
        println!("{}", "-".repeat(15 + self.solvers.len() * 16));

        let show_limit = limit.unwrap_or(problems.len());
        for problem in problems.iter().take(show_limit) {
            print!("{:<15}", problem);

            for map in &solver_maps {
                if let Some(result) = map.get(problem) {
                    let status_str = match result.status {
                        SolveStatus::Optimal => "Opt",
                        SolveStatus::AlmostOptimal => "AlmOpt",
                        SolveStatus::MaxIters => "MaxIt",
                        SolveStatus::NumericalError => "NumErr",
                        _ => "Other",
                    };
                    print!(" {:>7} {:>7.1}", status_str, result.solve_time_ms);
                } else {
                    print!(" {:>7} {:>7}", "-", "-");
                }
            }
            println!();
        }

        if problems.len() > show_limit {
            println!("... and {} more problems", problems.len() - show_limit);
        }

        println!("{}", "=".repeat(100));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solver_core::SolveStatus;

    fn make_test_result(name: &str, status: SolveStatus, time_ms: f64) -> BenchmarkResult {
        BenchmarkResult {
            name: name.to_string(),
            n: 10,
            m: 5,
            status,
            iterations: 10,
            obj_val: 1.0,
            mu: 1e-8,
            solve_time_ms: time_ms,
            error: None,
        }
    }

    #[test]
    fn test_win_matrix() {
        let solver_a = SolverResults::new(
            "SolverA".to_string(),
            vec![
                make_test_result("P1", SolveStatus::Optimal, 1.0),
                make_test_result("P2", SolveStatus::Optimal, 2.0),
                make_test_result("P3", SolveStatus::MaxIters, 3.0),
            ],
        );

        let solver_b = SolverResults::new(
            "SolverB".to_string(),
            vec![
                make_test_result("P1", SolveStatus::Optimal, 1.5),
                make_test_result("P2", SolveStatus::MaxIters, 2.5),
                make_test_result("P3", SolveStatus::Optimal, 3.5),
            ],
        );

        let comparison = SolverComparison::new(vec![solver_a, solver_b]);

        // Just test that it doesn't panic
        comparison.print_win_matrix();
        comparison.print_performance_comparison();
    }
}
