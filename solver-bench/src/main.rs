//! Benchmarking CLI for minix solver.

mod maros_meszaros;
mod qps;

use clap::{Parser, Subcommand};
use solver_core::{solve, ConeSpec, ProblemData, SolverSettings};
use solver_core::linalg::sparse;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "solver-bench")]
#[command(about = "Benchmarking CLI for minix solver")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run random generated benchmarks
    Random {
        /// Maximum iterations
        #[arg(long, default_value = "200")]
        max_iter: usize,
    },
    /// Run Maros-Meszaros QP benchmark suite
    MarosMeszaros {
        /// Maximum number of problems to run (default: all 138)
        #[arg(long)]
        limit: Option<usize>,
        /// Maximum iterations per problem
        #[arg(long, default_value = "200")]
        max_iter: usize,
        /// Run a single problem by name
        #[arg(long)]
        problem: Option<String>,
        /// Show detailed results table
        #[arg(long)]
        table: bool,
    },
    /// Parse and show info about a QPS file
    Info {
        /// Path to QPS file
        path: String,
    },
}

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

    let total_constraints = m + n;

    // Generate A part (m x n with sparsity)
    let mut triplets = Vec::new();
    for i in 0..m {
        for j in 0..n {
            if rand() < sparsity {
                let val = 2.0 * rand() - 1.0;
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
    let x_feas: Vec<f64> = (0..n).map(|_| rand() + 0.1).collect();
    let mut b = vec![0.0; total_constraints];

    for col in 0..n {
        if let Some(col_view) = a.outer_view(col) {
            for (row, &val) in col_view.iter() {
                if row < m {
                    b[row] += val * x_feas[col];
                }
            }
        }
    }

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

/// Generate a portfolio optimization LP
fn generate_portfolio_lp(n: usize, seed: u64) -> ProblemData {
    let mut rng_state = seed;
    let mut rand = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f64) / (u32::MAX as f64)
    };

    let q: Vec<f64> = (0..n).map(|_| -(rand() * 0.2 + 0.05)).collect();

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
    b[0] = 1.0;

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

fn run_random_benchmarks(max_iter: usize) {
    println!("Minix Solver Benchmarks");
    println!("=======================\n");

    let settings = SolverSettings {
        verbose: false,
        max_iter,
        tol_feas: 1e-6,
        tol_gap: 1e-6,
        ..Default::default()
    };

    // Portfolio LPs
    let prob = generate_portfolio_lp(50, 12345);
    run_benchmark("Portfolio LP (n=50)", &prob, &settings);

    let prob = generate_portfolio_lp(200, 12345);
    run_benchmark("Portfolio LP (n=200)", &prob, &settings);

    let prob = generate_portfolio_lp(500, 12345);
    run_benchmark("Portfolio LP (n=500)", &prob, &settings);

    // Random LPs
    let prob = generate_random_lp(100, 50, 0.3, 12345);
    run_benchmark("Random LP (n=100, m=50, 30% dense)", &prob, &settings);

    let prob = generate_random_lp(500, 200, 0.1, 12345);
    run_benchmark("Random LP (n=500, m=200, 10% dense)", &prob, &settings);

    let prob = generate_random_lp(1000, 500, 0.05, 12345);
    run_benchmark("Random LP (n=1000, m=500, 5% dense)", &prob, &settings);

    println!("\n{}", "=".repeat(60));
    println!("Benchmarks complete");
    println!("{}", "=".repeat(60));
}

fn run_maros_meszaros(limit: Option<usize>, max_iter: usize, problem: Option<String>, show_table: bool) {
    let settings = SolverSettings {
        verbose: false,
        max_iter,
        tol_feas: 1e-8,
        tol_gap: 1e-8,
        ..Default::default()
    };

    if let Some(name) = problem {
        // Run single problem
        println!("Running single problem: {}", name);
        let result = maros_meszaros::run_single(&name, &settings);

        if let Some(err) = &result.error {
            println!("Error: {}", err);
        } else {
            println!("Status:     {:?}", result.status);
            println!("Variables:  {}", result.n);
            println!("Constraints:{}", result.m);
            println!("Iterations: {}", result.iterations);
            println!("Objective:  {:.6e}", result.obj_val);
            println!("Final μ:    {:.6e}", result.mu);
            println!("Time:       {:.3} ms", result.solve_time_ms);
        }
    } else {
        // Run full suite
        println!("Running Maros-Meszaros QP Benchmark Suite");
        println!("=========================================\n");

        let results = maros_meszaros::run_full_suite(&settings, limit);
        let summary = maros_meszaros::compute_summary(&results);

        if show_table {
            maros_meszaros::print_results_table(&results);
        }

        maros_meszaros::print_summary(&summary);
    }
}

fn show_qps_info(path: &str) {
    match qps::parse_qps(path) {
        Ok(qps) => {
            println!("QPS Problem: {}", qps.name);
            println!("Variables:   {}", qps.n);
            println!("Constraints: {}", qps.m);
            println!("Q nonzeros:  {}", qps.p_triplets.len());
            println!("A nonzeros:  {}", qps.a_triplets.len());

            println!("\nVariable bounds:");
            for (i, name) in qps.var_names.iter().enumerate().take(5) {
                println!("  {}: [{}, {}]", name, qps.var_lower[i], qps.var_upper[i]);
            }
            if qps.n > 5 {
                println!("  ... ({} more)", qps.n - 5);
            }

            println!("\nConstraint bounds:");
            for (i, name) in qps.con_names.iter().enumerate().take(5) {
                println!("  {}: [{}, {}]", name, qps.con_lower[i], qps.con_upper[i]);
            }
            if qps.m > 5 {
                println!("  ... ({} more)", qps.m - 5);
            }

            // Try converting to conic form
            match qps.to_problem_data() {
                Ok(prob) => {
                    println!("\nConic form:");
                    println!("  Variables:   {}", prob.num_vars());
                    println!("  Constraints: {}", prob.num_constraints());
                    println!("  Cones:       {:?}", prob.cones);
                }
                Err(e) => {
                    println!("\nFailed to convert to conic form: {}", e);
                }
            }
        }
        Err(e) => {
            eprintln!("Error parsing QPS file: {}", e);
        }
    }
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Random { max_iter }) => {
            run_random_benchmarks(max_iter);
        }
        Some(Commands::MarosMeszaros { limit, max_iter, problem, table }) => {
            run_maros_meszaros(limit, max_iter, problem, table);
        }
        Some(Commands::Info { path }) => {
            show_qps_info(&path);
        }
        None => {
            // Default: run random benchmarks
            run_random_benchmarks(200);
        }
    }
}
