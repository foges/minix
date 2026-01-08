//! Comprehensive exponential cone benchmark suite

use solver_core::{ConeSpec, ProblemData, SolverSettings, solve, SolveStatus};
use solver_core::linalg::sparse;
use std::time::Instant;

// Re-use benchmark problem generators
mod bench_problems {
    use super::*;

    pub fn trivial() -> ProblemData {
        let num_vars = 1;
        let q = vec![1.0];
        let triplets = vec![(0, 0, -1.0)];
        let A = sparse::from_triplets(3, num_vars, triplets);
        let b = vec![0.0, 1.0, 1.0];
        ProblemData {
            P: None,
            q,
            A,
            b,
            cones: vec![ConeSpec::Exp { count: 1 }],
            var_bounds: None,
            integrality: None,
        }
    }

    pub fn cvxpy_style() -> ProblemData {
        let num_vars = 3;
        let q = vec![1.0, 1.0, 1.0];
        let e = std::f64::consts::E;
        let triplets = vec![
            (0, 0, -1.0), (1, 1, -1.0), (2, 2, -1.0),
            (3, 1, 1.0), (4, 2, 1.0),
        ];
        let A = sparse::from_triplets(5, num_vars, triplets);
        let b = vec![0.0, 0.0, 0.0, 1.0, e];
        ProblemData {
            P: None,
            q,
            A,
            b,
            cones: vec![ConeSpec::Exp { count: 1 }, ConeSpec::Zero { dim: 2 }],
            var_bounds: None,
            integrality: None,
        }
    }

    /// Create n independent copies of the trivial exp cone problem
    pub fn trivial_multi(n: usize) -> ProblemData {
        let num_vars = n;
        let q = vec![1.0; num_vars];

        // Each exp cone has 3 rows, constraint: s_i = [-x_i, 1, 1] ∈ K_exp
        let mut triplets = Vec::new();
        for i in 0..n {
            let row_base = 3 * i;
            triplets.push((row_base, i, -1.0));
        }

        let num_rows = 3 * n;
        let A = sparse::from_triplets(num_rows, num_vars, triplets);

        let mut b = Vec::new();
        for _ in 0..n {
            b.push(0.0);  // s[3i] = 0
            b.push(1.0);  // s[3i+1] = 1
            b.push(1.0);  // s[3i+2] = 1
        }

        ProblemData {
            P: None,
            q,
            A,
            b,
            cones: vec![ConeSpec::Exp { count: n }],
            var_bounds: None,
            integrality: None,
        }
    }

    pub fn entropy_maximization(n: usize) -> ProblemData {
        let num_vars = 2 * n;
        let mut q = vec![0.0; num_vars];
        for i in 0..n {
            q[n + i] = -1.0;
        }

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

    pub fn kl_divergence(n: usize) -> ProblemData {
        // Use uniform target distribution
        let p: Vec<f64> = (0..n).map(|_| 1.0 / n as f64).collect();

        let num_vars = 2 * n;
        let mut q = vec![0.0; num_vars];
        for i in 0..n {
            q[i] = -p[i].ln();
            q[n + i] = 1.0;
        }

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

    pub fn log_sum_exp(n: usize) -> ProblemData {
        let a: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
        let b = 1.0_f64;

        let num_vars = n;
        let q = vec![0.0; num_vars];

        let mut triplets = Vec::new();
        for i in 0..n {
            let row_offset = 3 * i;
            triplets.push((row_offset + 2, i, -1.0));
        }
        let ineq_row = 3 * n;
        for i in 0..n {
            triplets.push((ineq_row, i, 1.0));
        }

        let A = sparse::from_triplets(3 * n + 1, num_vars, triplets);
        let mut b_vec = Vec::new();
        for i in 0..n {
            b_vec.push(a[i]);
            b_vec.push(1.0);
            b_vec.push(0.0);
        }
        b_vec.push(b.exp());

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
}

struct BenchResult {
    name: String,
    n: usize,
    status: SolveStatus,
    iters: usize,
    time_ms: f64,
    obj: f64,
}

fn main() {
    // Use properly-sized multi-cone problems
    let problems: Vec<(&str, Box<dyn Fn() -> ProblemData>, usize)> = vec![
        ("trivial-1", Box::new(|| bench_problems::trivial()), 1),
        ("cvxpy-3", Box::new(|| bench_problems::cvxpy_style()), 3),
        ("trivial-multi-2", Box::new(|| bench_problems::trivial_multi(2)), 2),
        ("trivial-multi-5", Box::new(|| bench_problems::trivial_multi(5)), 5),
        ("trivial-multi-10", Box::new(|| bench_problems::trivial_multi(10)), 10),
    ];

    let mut settings = SolverSettings::default();
    settings.max_iter = 250;  // Allow more iterations for harder problems
    settings.verbose = false;

    println!("\n{:=<80}", "");
    println!("EXPONENTIAL CONE BENCHMARK SUITE");
    println!("{:=<80}\n", "");

    println!("{:<20} {:>6} {:>8} {:>12} {:>12} {:>10}",
             "Problem", "n", "Status", "Iters", "Time (ms)", "Objective");
    println!("{:-<80}", "");

    let mut results = Vec::new();

    for (name, prob_fn, n) in problems {
        let prob = prob_fn();

        // Warm-up
        let _ = solve(&prob, &settings);

        // Timed solve
        let start = Instant::now();
        let result = solve(&prob, &settings);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let (status, iters, obj) = match result {
            Ok(sol) => (sol.status, sol.info.iters, sol.obj_val),
            Err(_) => (SolveStatus::NumericalError, 0, f64::NAN),
        };

        let status_str = match status {
            SolveStatus::Optimal => "Optimal",
            SolveStatus::AlmostOptimal => "AlmostOpt",
            SolveStatus::PrimalInfeasible => "PrInfeas",
            SolveStatus::DualInfeasible => "DuInfeas",
            SolveStatus::MaxIters => "MaxIter",
            SolveStatus::NumericalError => "NumError",
            _ => "Other",
        };

        println!("{:<20} {:>6} {:>8} {:>12} {:>12.2} {:>12.4}",
                 name, n, status_str, iters, elapsed, obj);

        results.push(BenchResult {
            name: name.to_string(),
            n,
            status,
            iters,
            time_ms: elapsed,
            obj,
        });
    }

    println!("{:-<80}", "");

    // Summary statistics
    let optimal_count = results.iter().filter(|r| r.status == SolveStatus::Optimal).count();
    let total_time: f64 = results.iter().map(|r| r.time_ms).sum();
    let avg_time = total_time / results.len() as f64;
    let avg_iters: f64 = results.iter().map(|r| r.iters as f64).sum::<f64>() / results.len() as f64;

    println!("\nSummary:");
    println!("  Solved: {}/{}", optimal_count, results.len());
    println!("  Avg time: {:.2} ms", avg_time);
    println!("  Avg iters: {:.1}", avg_iters);
    println!("  Total time: {:.2} ms\n", total_time);

    println!("{:=<80}\n", "");
}
