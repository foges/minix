//! Benchmark suite for exponential cone problems.
//!
//! Tests convergence quality and wall-clock performance on various
//! exp cone problems including entropy maximization, KL divergence,
//! and log-sum-exp constraints.

use solver_core::linalg::sparse;
use solver_core::{ConeSpec, ProblemData, SolverSettings, SolveStatus, solve};
use std::time::Instant;

/// Entropy maximization: maximize ∑ x_i log(x_i) subject to ∑ x_i = 1, x ≥ 0
///
/// Reformulated as exponential cone problem:
/// minimize -∑ t_i subject to (t_i, x_i, x_i) ∈ K_exp, ∑ x_i = 1
pub fn entropy_maximization(n: usize) -> ProblemData {
    // Variables: [x_1, ..., x_n, t_1, ..., t_n]
    let num_vars = 2 * n;

    // Objective: minimize -∑ t_i  (equivalent to maximizing ∑ t_i)
    let mut q = vec![0.0; num_vars];
    for i in 0..n {
        q[n + i] = -1.0; // Coefficients for t_i
    }

    // Constraints:
    // Rows 0..3n: n exp cones (t_i, x_i, x_i) ∈ K_exp
    // Row 3n: ∑ x_i = 1 (equality constraint)
    let mut triplets = Vec::new();

    // Exp cone constraints: (t_i, x_i, x_i) ∈ K_exp
    for i in 0..n {
        let row_offset = 3 * i;
        // Row 0: -t_i + s_0 = 0  =>  s_0 = t_i
        triplets.push((row_offset, n + i, -1.0));
        // Row 1: -x_i + s_1 = 0  =>  s_1 = x_i
        triplets.push((row_offset + 1, i, -1.0));
        // Row 2: -x_i + s_2 = 0  =>  s_2 = x_i
        triplets.push((row_offset + 2, i, -1.0));
    }

    // Equality constraint: ∑ x_i + s = 1
    let eq_row = 3 * n;
    for i in 0..n {
        triplets.push((eq_row, i, 1.0));
    }

    let A = sparse::from_triplets(3 * n + 1, num_vars, triplets);
    let mut b = vec![0.0; 3 * n + 1];
    b[3 * n] = 1.0; // ∑ x_i = 1

    let mut cones = Vec::new();
    for _ in 0..n {
        cones.push(ConeSpec::Exp { count: 1 });
    }
    cones.push(ConeSpec::Zero { dim: 1 });

    ProblemData {
        P: None,
        q,
        A,
        b,
        cones,
        var_bounds: None,
        integrality: None,
    }
}

/// KL divergence minimization: min KL(x || p) = ∑ x_i log(x_i/p_i)
/// subject to ∑ x_i = 1, x ≥ 0
///
/// Reformulated: min ∑ t_i - ∑ x_i log(p_i)
/// subject to (t_i, x_i, x_i) ∈ K_exp, ∑ x_i = 1
pub fn kl_divergence(n: usize, p: &[f64]) -> ProblemData {
    assert_eq!(p.len(), n, "p must have length n");
    assert!(p.iter().all(|&pi| pi > 0.0), "p must be positive");

    // Variables: [x_1, ..., x_n, t_1, ..., t_n]
    let num_vars = 2 * n;

    // Objective: minimize ∑ t_i - ∑ x_i log(p_i)
    let mut q = vec![0.0; num_vars];
    for i in 0..n {
        q[i] = -p[i].ln(); // Coefficient for x_i
        q[n + i] = 1.0;     // Coefficient for t_i
    }

    // Same structure as entropy_maximization
    let mut triplets = Vec::new();
    for i in 0..n {
        let row_offset = 3 * i;
        triplets.push((row_offset, n + i, -1.0));
        triplets.push((row_offset + 1, i, -1.0));
        triplets.push((row_offset + 2, i, -1.0));
    }
    let eq_row = 3 * n;
    for i in 0..n {
        triplets.push((eq_row, i, 1.0));
    }

    let A = sparse::from_triplets(3 * n + 1, num_vars, triplets);
    let mut b = vec![0.0; 3 * n + 1];
    b[3 * n] = 1.0;

    let mut cones = Vec::new();
    for _ in 0..n {
        cones.push(ConeSpec::Exp { count: 1 });
    }
    cones.push(ConeSpec::Zero { dim: 1 });

    ProblemData {
        P: None,
        q,
        A,
        b,
        cones,
        var_bounds: None,
        integrality: None,
    }
}

/// Log-sum-exp constraint: log(∑ exp(a_i)) ≤ b
///
/// Reformulated: ∑ y_i ≤ exp(b), (a_i, 1, y_i) ∈ K_exp
pub fn log_sum_exp(n: usize, a: &[f64], b: f64) -> ProblemData {
    assert_eq!(a.len(), n);

    // Variables: [y_1, ..., y_n]
    let num_vars = n;

    // Objective: minimize 0 (feasibility problem)
    let q = vec![0.0; num_vars];

    // Constraints:
    // Rows 0..3n: n exp cones (a_i, 1, y_i) ∈ K_exp
    // Row 3n: ∑ y_i ≤ exp(b) (NonNeg cone)
    let mut triplets = Vec::new();

    for i in 0..n {
        let row_offset = 3 * i;
        // Row 0: s_0 = a_i (fixed)
        // Row 1: s_1 = 1 (fixed)
        // Row 2: -y_i + s_2 = 0
        triplets.push((row_offset + 2, i, -1.0));
    }

    // Inequality: ∑ y_i + s = exp(b)
    let ineq_row = 3 * n;
    for i in 0..n {
        triplets.push((ineq_row, i, 1.0));
    }

    let A = sparse::from_triplets(3 * n + 1, num_vars, triplets);
    let mut b_vec = Vec::new();
    for i in 0..n {
        b_vec.push(a[i]);  // s_0 = a_i
        b_vec.push(1.0);   // s_1 = 1
        b_vec.push(0.0);   // s_2 = y_i
    }
    b_vec.push(b.exp()); // ∑ y_i + s = exp(b)

    let mut cones = Vec::new();
    for _ in 0..n {
        cones.push(ConeSpec::Exp { count: 1 });
    }
    cones.push(ConeSpec::NonNeg { dim: 1 });

    ProblemData {
        P: None,
        q,
        A,
        b: b_vec,
        cones,
        var_bounds: None,
        integrality: None,
    }
}

/// Portfolio optimization with exponential utility
///
/// maximize E[R] - λ/2 Var[R] where R = r'x is portfolio return
/// subject to ∑ x_i = 1, x ≥ 0 (fully invested, long-only)
///
/// For exponential utility: max ∑ r_i x_i - λ ∑ x_i log(x_i)
pub fn portfolio_exp_utility(n: usize, returns: &[f64], lambda: f64) -> ProblemData {
    assert_eq!(returns.len(), n);

    // Variables: [x_1, ..., x_n, t_1, ..., t_n]
    let num_vars = 2 * n;

    // Objective: minimize -∑ r_i x_i + λ ∑ t_i
    let mut q = vec![0.0; num_vars];
    for i in 0..n {
        q[i] = -returns[i];  // -r_i for x_i
        q[n + i] = lambda;   // λ for t_i
    }

    // Constraints: (t_i, x_i, x_i) ∈ K_exp, ∑ x_i = 1
    let mut triplets = Vec::new();
    for i in 0..n {
        let row_offset = 3 * i;
        triplets.push((row_offset, n + i, -1.0));
        triplets.push((row_offset + 1, i, -1.0));
        triplets.push((row_offset + 2, i, -1.0));
    }
    let eq_row = 3 * n;
    for i in 0..n {
        triplets.push((eq_row, i, 1.0));
    }

    let A = sparse::from_triplets(3 * n + 1, num_vars, triplets);
    let mut b = vec![0.0; 3 * n + 1];
    b[3 * n] = 1.0;

    let mut cones = Vec::new();
    for _ in 0..n {
        cones.push(ConeSpec::Exp { count: 1 });
    }
    cones.push(ConeSpec::Zero { dim: 1 });

    ProblemData {
        P: None,
        q,
        A,
        b,
        cones,
        var_bounds: None,
        integrality: None,
    }
}

pub struct BenchResult {
    pub name: String,
    pub n: usize,
    pub status: String,
    pub iters: usize,
    pub obj_val: f64,
    pub solve_time_ms: f64,
    pub feasible: bool,
    pub optimal: bool,
}

pub fn run_exp_cone_benchmarks(verbose: bool) -> Vec<BenchResult> {
    let mut results = Vec::new();

    // Test sizes
    let sizes = vec![5, 10, 20, 50];

    for &n in &sizes {
        // Entropy maximization
        {
            let prob = entropy_maximization(n);
            let mut settings = SolverSettings::default();
            settings.verbose = verbose;
            settings.max_iter = 100;

            let start = Instant::now();
            let result = solve(&prob, &settings).unwrap();
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            let optimal = matches!(result.status, SolveStatus::Optimal);
            let feasible = result.info.primal_res <= 1e-6 && result.info.dual_res <= 1e-6;

            results.push(BenchResult {
                name: format!("entropy_max_n{}", n),
                n,
                status: format!("{:?}", result.status),
                iters: result.info.iters,
                obj_val: result.obj_val,
                solve_time_ms: elapsed,
                feasible,
                optimal,
            });
        }

        // KL divergence (uniform prior)
        {
            let p = vec![1.0 / n as f64; n];
            let prob = kl_divergence(n, &p);
            let mut settings = SolverSettings::default();
            settings.verbose = verbose;
            settings.max_iter = 100;

            let start = Instant::now();
            let result = solve(&prob, &settings).unwrap();
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            let optimal = matches!(result.status, SolveStatus::Optimal);
            let feasible = result.info.primal_res <= 1e-6 && result.info.dual_res <= 1e-6;

            results.push(BenchResult {
                name: format!("kl_divergence_n{}", n),
                n,
                status: format!("{:?}", result.status),
                iters: result.info.iters,
                obj_val: result.obj_val,
                solve_time_ms: elapsed,
                feasible,
                optimal,
            });
        }

        // Portfolio with exponential utility
        if n <= 20 {
            let returns: Vec<f64> = (0..n).map(|i| 0.05 + 0.02 * (i as f64) / (n as f64)).collect();
            let prob = portfolio_exp_utility(n, &returns, 0.5);
            let mut settings = SolverSettings::default();
            settings.verbose = verbose;
            settings.max_iter = 100;

            let start = Instant::now();
            let result = solve(&prob, &settings).unwrap();
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            let optimal = matches!(result.status, SolveStatus::Optimal);
            let feasible = result.info.primal_res <= 1e-6 && result.info.dual_res <= 1e-6;

            results.push(BenchResult {
                name: format!("portfolio_exp_n{}", n),
                n,
                status: format!("{:?}", result.status),
                iters: result.info.iters,
                obj_val: result.obj_val,
                solve_time_ms: elapsed,
                feasible,
                optimal,
            });
        }
    }

    results
}

pub fn print_benchmark_table(results: &[BenchResult]) {
    println!("\n{:=<100}", "");
    println!("EXPONENTIAL CONE BENCHMARK RESULTS");
    println!("{:=<100}", "");
    println!(
        "{:<30} {:>6} {:>10} {:>8} {:>15} {:>12} {:>8}",
        "Problem", "n", "Status", "Iters", "Objective", "Time (ms)", "Quality"
    );
    println!("{:-<100}", "");

    for r in results {
        let quality = if r.optimal {
            "Optimal"
        } else if r.feasible {
            "Feasible"
        } else {
            "Infeas"
        };

        println!(
            "{:<30} {:>6} {:>10} {:>8} {:>15.6e} {:>12.2} {:>8}",
            r.name, r.n, r.status, r.iters, r.obj_val, r.solve_time_ms, quality
        );
    }

    println!("{:=<100}", "");

    // Summary statistics
    let optimal_count = results.iter().filter(|r| r.optimal).count();
    let feasible_count = results.iter().filter(|r| r.feasible).count();
    let avg_iters = results.iter().map(|r| r.iters).sum::<usize>() as f64 / results.len() as f64;
    let avg_time = results.iter().map(|r| r.solve_time_ms).sum::<f64>() / results.len() as f64;

    println!("Summary:");
    println!("  Optimal: {}/{} ({:.1}%)", optimal_count, results.len(),
             100.0 * optimal_count as f64 / results.len() as f64);
    println!("  Feasible: {}/{} ({:.1}%)", feasible_count, results.len(),
             100.0 * feasible_count as f64 / results.len() as f64);
    println!("  Avg iterations: {:.1}", avg_iters);
    println!("  Avg solve time: {:.2} ms", avg_time);
    println!("{:=<100}\n", "");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_maximization_small() {
        let prob = entropy_maximization(3);
        let mut settings = SolverSettings::default();
        settings.verbose = true;
        settings.max_iter = 50;

        let result = solve(&prob, &settings).unwrap();
        println!("\n=== Entropy Maximization (n=3) ===");
        println!("Status: {:?}", result.status);
        println!("Iterations: {}", result.info.iters);
        println!("Objective: {:.6}", result.obj_val);
        println!("Solution x: {:?}", &result.x[0..3]);

        // Optimal solution should be uniform: x_i = 1/3
        // Objective = -3 * (1/3) * log(1/3) = log(3) ≈ 1.0986
        assert!(result.info.primal_res <= 1e-5, "Primal residual too high");
        assert!(result.info.dual_res <= 1e-5, "Dual residual too high");
    }

    #[test]
    fn test_kl_divergence_small() {
        let n = 3;
        let p = vec![0.5, 0.3, 0.2];
        let prob = kl_divergence(n, &p);
        let mut settings = SolverSettings::default();
        settings.verbose = true;
        settings.max_iter = 50;

        let result = solve(&prob, &settings).unwrap();
        println!("\n=== KL Divergence (n=3) ===");
        println!("Status: {:?}", result.status);
        println!("Iterations: {}", result.info.iters);
        println!("Objective: {:.6}", result.obj_val);
        println!("Solution x: {:?}", &result.x[0..3]);
        println!("Target  p: {:?}", p);

        // Optimal solution should be x = p (minimize KL(x||p))
        assert!(result.info.primal_res <= 1e-5, "Primal residual too high");
        assert!(result.info.dual_res <= 1e-5, "Dual residual too high");
    }

    #[test]
    fn test_run_all_benchmarks() {
        let results = run_exp_cone_benchmarks(false);
        print_benchmark_table(&results);

        // Check that at least some problems solved optimally
        let optimal_count = results.iter().filter(|r| r.optimal).count();
        assert!(optimal_count > 0, "No problems solved optimally");
    }
}
