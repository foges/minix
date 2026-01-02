//! Benchmarking CLI for minix solver.

use solver_core::{solve, ProblemData, ConeSpec, SolverSettings};
use solver_core::linalg::sparse;
use std::time::Instant;

/// Generate a random LP:
///   minimize    c^T x
///   subject to  Ax = b
///               x >= 0
///
/// where A is m x n, with density `sparsity`.
fn generate_random_lp(n: usize, m: usize, sparsity: f64, seed: u64) -> ProblemData {
    // Simple LCG random number generator
    let mut rng_state = seed;
    let mut rand = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f64) / (u32::MAX as f64)
    };

    // Generate cost vector c (random positive values)
    let q: Vec<f64> = (0..n).map(|_| rand() + 0.1).collect();

    // Generate constraint matrix A with given sparsity
    // Format: Ax + s = b, s in Zero cone (equality), then x >= 0 via NonNeg
    // We'll use: [A | I_m] [x; s_eq] = b for equalities
    //            [-I_n | 0] [x; s_eq] + s_bound = 0 for x >= 0

    // Actually, let's use the standard form:
    // Ax + s_eq = b (equality constraints, s_eq = 0)
    // -x + s_bound = 0 (bound constraints, s_bound >= 0, so x >= 0)
    //
    // Combined A matrix is (m + n) x n:
    // [  A  ]
    // [ -I  ]
    //
    // b = [b_eq; 0]
    // cones = [Zero(m), NonNeg(n)]

    let total_constraints = m + n;

    // Generate A part (m x n with sparsity)
    let mut triplets = Vec::new();
    for i in 0..m {
        for j in 0..n {
            if rand() < sparsity {
                let val = 2.0 * rand() - 1.0; // Random in [-1, 1]
                triplets.push((i, j, val));
            }
        }
        // Ensure at least one nonzero per row for feasibility
        let j = (rand() * n as f64) as usize;
        let j = j.min(n - 1);
        triplets.push((i, j, rand() + 0.5));
    }

    // Add -I part for bound constraints
    for j in 0..n {
        triplets.push((m + j, j, -1.0));
    }

    let a = sparse::from_triplets(total_constraints, n, triplets);

    // Generate RHS b
    // For feasibility, compute b = A * x_feas for some feasible x
    let x_feas: Vec<f64> = (0..n).map(|_| rand() + 0.1).collect();
    let mut b = vec![0.0; total_constraints];

    // b[0..m] = A * x_feas
    for col in 0..n {
        if let Some(col_view) = a.outer_view(col) {
            for (row, &val) in col_view.iter() {
                if row < m {
                    b[row] += val * x_feas[col];
                }
            }
        }
    }
    // b[m..m+n] = 0 (bound constraints -x + s = 0)

    ProblemData {
        P: None,
        q,
        A: a,
        b,
        cones: vec![
            ConeSpec::Zero { dim: m },
            ConeSpec::NonNeg { dim: n },
        ],
        var_bounds: None,
        integrality: None,
    }
}

/// Generate a portfolio optimization LP:
///   minimize    -r^T x  (maximize return)
///   subject to  1^T x = 1  (weights sum to 1)
///               x >= 0     (long only)
fn generate_portfolio_lp(n: usize, seed: u64) -> ProblemData {
    let mut rng_state = seed;
    let mut rand = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f64) / (u32::MAX as f64)
    };

    // Expected returns (negative because we minimize)
    let q: Vec<f64> = (0..n).map(|_| -(rand() * 0.2 + 0.05)).collect();

    // Constraints:
    // 1. Sum of weights = 1: [1, 1, ..., 1] x + s_eq = 1, s_eq = 0
    // 2. x >= 0: -I x + s_bound = 0, s_bound >= 0

    let mut triplets = Vec::new();

    // Row 0: sum constraint
    for j in 0..n {
        triplets.push((0, j, 1.0));
    }

    // Rows 1..n+1: -I for bounds
    for j in 0..n {
        triplets.push((1 + j, j, -1.0));
    }

    let a = sparse::from_triplets(1 + n, n, triplets);
    let mut b = vec![0.0; 1 + n];
    b[0] = 1.0; // Sum = 1

    ProblemData {
        P: None,
        q,
        A: a,
        b,
        cones: vec![
            ConeSpec::Zero { dim: 1 },
            ConeSpec::NonNeg { dim: n },
        ],
        var_bounds: None,
        integrality: None,
    }
}

fn run_benchmark(name: &str, prob: &ProblemData, settings: &SolverSettings) {
    let n = prob.num_vars();
    let m = prob.num_constraints();
    let nnz = prob.A.nnz();

    println!("\n{}", "=".repeat(60));
    println!("{}", name);
    println!("{}", "=".repeat(60));
    println!("Variables (n):    {}", n);
    println!("Constraints (m):  {}", m);
    println!("A nonzeros:       {} ({:.2}% dense)", nnz, 100.0 * nnz as f64 / (n * m) as f64);
    println!();

    let start = Instant::now();
    let result = solve(prob, settings);
    let elapsed = start.elapsed();

    match result {
        Ok(res) => {
            println!("Status:           {:?}", res.status);
            println!("Iterations:       {}", res.info.iters);
            println!("Objective:        {:.6e}", res.obj_val);
            println!("Final μ:          {:.6e}", res.info.mu);
            println!("Solve time:       {:.3} ms", elapsed.as_secs_f64() * 1000.0);
            println!("Time/iteration:   {:.3} ms", elapsed.as_secs_f64() * 1000.0 / res.info.iters as f64);
        }
        Err(e) => {
            println!("ERROR: {}", e);
        }
    }
}

fn main() {
    println!("Minix Solver Benchmarks");
    println!("=======================\n");

    let settings = SolverSettings {
        verbose: false,
        max_iter: 200,
        tol_feas: 1e-6,
        tol_gap: 1e-6,
        ..Default::default()
    };

    // Small portfolio LP
    let prob = generate_portfolio_lp(50, 12345);
    run_benchmark("Portfolio LP (n=50)", &prob, &settings);

    // Medium portfolio LP
    let prob = generate_portfolio_lp(200, 12345);
    run_benchmark("Portfolio LP (n=200)", &prob, &settings);

    // Large portfolio LP
    let prob = generate_portfolio_lp(500, 12345);
    run_benchmark("Portfolio LP (n=500)", &prob, &settings);

    // Random sparse LP (small)
    let prob = generate_random_lp(100, 50, 0.3, 12345);
    run_benchmark("Random LP (n=100, m=50, 30% dense)", &prob, &settings);

    // Random sparse LP (medium)
    let prob = generate_random_lp(500, 200, 0.1, 12345);
    run_benchmark("Random LP (n=500, m=200, 10% dense)", &prob, &settings);

    // Random sparse LP (large)
    let prob = generate_random_lp(1000, 500, 0.05, 12345);
    run_benchmark("Random LP (n=1000, m=500, 5% dense)", &prob, &settings);

    println!("\n{}", "=".repeat(60));
    println!("Benchmarks complete");
    println!("{}", "=".repeat(60));
}
