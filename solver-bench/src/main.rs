//! Benchmarking CLI for minix solver.

mod conic_benchmarks;
mod exp_cone_bench;
mod maros_meszaros;
mod qps;
mod regression;
mod sdplib;
mod solver_choice;
mod test_problems;

use clap::{Parser, Subcommand, ValueEnum};
use solver_choice::{solve_with_choice, SolverChoice};
use solver_core::{ConeSpec, ProblemData, SolveStatus, SolverSettings};
use solver_core::linalg::sparse;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "solver-bench")]
#[command(about = "Benchmarking CLI for minix solver")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

/// Available benchmark suites
#[derive(Debug, Clone, Copy, ValueEnum)]
enum BenchmarkSuite {
    /// Maros-Meszaros QP benchmark suite (136 problems)
    Mm,
    /// SDPLIB SDP benchmark suite (~90 problems)
    Sdplib,
    /// Exponential cone benchmarks (entropy, KL, portfolio)
    Exp,
}

#[derive(Subcommand)]
enum Commands {
    /// Run random generated benchmarks
    Random {
        /// Maximum iterations
        #[arg(long, default_value = "200")]
        max_iter: usize,
        /// Solver backend to use
        #[arg(long, value_enum, default_value = "ipm2")]
        solver: SolverChoice,
    },
    /// Run benchmark suite (Maros-Meszaros QP or SDPLIB SDP)
    Benchmark {
        /// Benchmark suite to run
        #[arg(long, value_enum)]
        suite: BenchmarkSuite,
        /// Path to problem files (default: auto-detect from _data/)
        #[arg(long)]
        path: Option<String>,
        /// Maximum number of problems to run
        #[arg(long)]
        limit: Option<usize>,
        /// Maximum iterations per problem
        #[arg(long, default_value = "100")]
        max_iter: usize,
        /// Run a single problem by name
        #[arg(long)]
        problem: Option<String>,
        /// Show detailed results table (MM only)
        #[arg(long)]
        table: bool,
        /// Verbose output (show iteration progress)
        #[arg(long, short)]
        verbose: bool,
        /// Solver backend to use
        #[arg(long, value_enum, default_value = "ipm2")]
        solver: SolverChoice,
    },
    /// Parse and show info about a QPS file
    Info {
        /// Path to QPS file
        path: String,
    },
    /// Run regression suite (local QPS cache + synthetic cases)
    Regression {
        /// Maximum iterations per problem (for passing problems)
        #[arg(long, default_value = "200")]
        max_iter: usize,
        /// Maximum iterations for expected-to-fail problems
        #[arg(long, default_value = "50")]
        max_iter_fail: usize,
        /// Require cached QPS files (fail if missing)
        #[arg(long)]
        require_cache: bool,
        /// Solver backend to use
        #[arg(long, value_enum, default_value = "ipm2")]
        solver: SolverChoice,
        /// Read performance baseline JSON and gate regressions
        #[arg(long)]
        baseline_in: Option<String>,
        /// Write performance baseline JSON
        #[arg(long)]
        baseline_out: Option<String>,
        /// Allowed regression ratio (0.2 = 20% slower)
        #[arg(long, default_value = "0.2")]
        max_regression: f64,
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

fn run_benchmark(name: &str, prob: &ProblemData, settings: &SolverSettings, solver: SolverChoice) {
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
    let result = solve_with_choice(prob, settings, solver);
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

fn run_random_benchmarks(max_iter: usize, solver: SolverChoice) {
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
    run_benchmark("Portfolio LP (n=50)", &prob, &settings, solver);

    let prob = generate_portfolio_lp(200, 12345);
    run_benchmark("Portfolio LP (n=200)", &prob, &settings, solver);

    let prob = generate_portfolio_lp(500, 12345);
    run_benchmark("Portfolio LP (n=500)", &prob, &settings, solver);

    // Random LPs
    let prob = generate_random_lp(100, 50, 0.3, 12345);
    run_benchmark("Random LP (n=100, m=50, 30% dense)", &prob, &settings, solver);

    let prob = generate_random_lp(500, 200, 0.1, 12345);
    run_benchmark("Random LP (n=500, m=200, 10% dense)", &prob, &settings, solver);

    let prob = generate_random_lp(1000, 500, 0.05, 12345);
    run_benchmark("Random LP (n=1000, m=500, 5% dense)", &prob, &settings, solver);

    println!("\n{}", "=".repeat(60));
    println!("Benchmarks complete");
    println!("{}", "=".repeat(60));
}

fn run_maros_meszaros(
    limit: Option<usize>,
    max_iter: usize,
    problem: Option<String>,
    show_table: bool,
    solver: SolverChoice,
) {
    // Check for direct mode via environment variable
    let direct_mode = std::env::var("MINIX_DIRECT_MODE")
        .map(|v| v != "0")
        .unwrap_or(false);

    let settings = SolverSettings {
        verbose: false,
        max_iter,
        tol_feas: 1e-8,
        tol_gap: 1e-8,
        direct_mode,
        ..Default::default()
    };

    if let Some(name) = problem {
        // Run single problem
        println!("Running single problem: {}", name);
        let result = maros_meszaros::run_single(&name, &settings, solver);

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

        let results = maros_meszaros::run_full_suite(&settings, limit, solver);
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

fn run_regression_suite(
    max_iter: usize,
    max_iter_fail: usize,
    solver: SolverChoice,
    require_cache: bool,
    baseline_in: Option<String>,
    baseline_out: Option<String>,
    max_regression: f64,
) {
    let mut settings = SolverSettings::default();
    settings.max_iter = max_iter;

    let results = regression::run_regression_suite(&settings, solver, require_cache, max_iter_fail);
    let mut failed = 0usize;
    let mut skipped = 0usize;

    for res in &results {
        if res.skipped {
            skipped += 1;
            println!("{}: SKIP (missing cache)", res.name);
            continue;
        }
        if res.status != SolveStatus::Optimal
            || !res.rel_p.is_finite()
            || !res.rel_d.is_finite()
            || !res.gap_rel.is_finite()
        {
            failed += 1;
            println!(
                "{}: FAIL status={:?} rel_p={:.2e} rel_d={:.2e} gap_rel={:.2e} {}",
                res.name,
                res.status,
                res.rel_p,
                res.rel_d,
                res.gap_rel,
                res.error.as_deref().unwrap_or(""),
            );
            continue;
        }

        // Use practical tolerances for unscaled metrics
        let tol_feas = 1e-6;
        let tol_gap = 1e-3;
        if res.rel_p > tol_feas || res.rel_d > tol_feas || res.gap_rel > tol_gap {
            failed += 1;
            println!(
                "{}: FAIL rel_p={:.2e} rel_d={:.2e} gap_rel={:.2e}",
                res.name,
                res.rel_p,
                res.rel_d,
                res.gap_rel,
            );
        } else {
            println!(
                "{}: OK iters={} rel_p={:.2e} rel_d={:.2e} gap_rel={:.2e}",
                res.name,
                res.iterations,
                res.rel_p,
                res.rel_d,
                res.gap_rel,
            );
        }
    }

    println!(
        "summary: total={} failed={} skipped={}",
        results.len(),
        failed,
        skipped
    );

    if failed == 0 {
        if let Some(path) = baseline_out.as_ref() {
            let summary = regression::perf_summary(&results);
            let payload = serde_json::to_string_pretty(&summary)
                .expect("failed to serialize perf summary");
            if let Err(e) = std::fs::write(path, payload) {
                eprintln!("failed to write baseline {}: {}", path, e);
                std::process::exit(1);
            }
        }

        if let Some(path) = baseline_in.as_ref() {
            let Ok(contents) = std::fs::read_to_string(path) else {
                eprintln!("failed to read baseline {}", path);
                std::process::exit(1);
            };
            let baseline: regression::PerfSummary = match serde_json::from_str(&contents) {
                Ok(val) => val,
                Err(e) => {
                    eprintln!("failed to parse baseline {}: {}", path, e);
                    std::process::exit(1);
                }
            };
            let summary = regression::perf_summary(&results);
            let perf_failures =
                regression::compare_perf_baseline(&baseline, &summary, max_regression);
            if !perf_failures.is_empty() {
                for msg in perf_failures {
                    eprintln!("perf regression: {}", msg);
                }
                std::process::exit(1);
            }
        }
    }

    if failed > 0 || (require_cache && skipped > 0) {
        std::process::exit(1);
    }
}

fn run_exp_benchmarks(_max_iter: usize, verbose: bool) {
    use exp_cone_bench::{run_exp_cone_benchmarks, print_benchmark_table};

    println!("Exponential Cone Benchmark Suite");
    println!("=================================\n");

    let results = run_exp_cone_benchmarks(verbose);
    print_benchmark_table(&results);
}

fn run_sdplib_benchmarks(
    path: Option<String>,
    limit: Option<usize>,
    max_iter: usize,
    problem: Option<String>,
    download: Option<String>,
    verbose: bool,
) {
    use sdplib::{load_sdpa_file, sdpa_to_conic, sdplib_reference_values, solve_sdpa};
    use std::path::PathBuf;

    // Handle download request
    if let Some(dir) = download {
        println!("To download SDPLIB problems:");
        println!("  curl -L http://euler.nmt.edu/~brian/sdplib/sdplib.zip -o sdplib.zip");
        println!("  unzip sdplib.zip -d {}", dir);
        println!("\nAlternatively, visit: http://euler.nmt.edu/~brian/sdplib/sdplib.html");
        return;
    }

    let settings = SolverSettings {
        verbose,
        max_iter,
        tol_feas: 1e-6,
        tol_gap: 1e-6,
        ..Default::default()
    };

    let reference = sdplib_reference_values();

    // Single problem mode
    if let Some(name) = problem {
        let file_path = if let Some(ref dir) = path {
            PathBuf::from(dir).join(format!("{}.dat-s", name))
        } else {
            PathBuf::from(format!("{}.dat-s", name))
        };

        if !file_path.exists() {
            eprintln!("Problem file not found: {}", file_path.display());
            return;
        }

        println!("Running single problem: {}", name);

        match load_sdpa_file(&file_path) {
            Ok(sdpa) => {
                println!("Problem: {}", sdpa.name);
                println!("Constraints (m): {}", sdpa.m_dim);
                println!("Blocks: {:?}", sdpa.block_struct);

                match sdpa_to_conic(&sdpa) {
                    Ok(prob) => {
                        println!("Variables (svec): {}", prob.num_vars());

                        let start = Instant::now();
                        match solve_sdpa(&sdpa, &settings) {
                            Ok(result) => {
                                let elapsed = start.elapsed();
                                println!("Status: {:?}", result.status);
                                println!("Primal obj: {:.6e}", result.primal_obj);
                                println!("Dual obj: {:.6e}", result.dual_obj);
                                println!("Iterations: {}", result.iterations);
                                println!("Time: {:.3} ms", elapsed.as_secs_f64() * 1000.0);

                                if let Some(&ref_val) = reference.get(sdpa.name.as_str()) {
                                    let rel_err = (result.primal_obj - ref_val).abs() / (1.0 + ref_val.abs());
                                    println!("Reference: {:.6e}", ref_val);
                                    println!("Rel error: {:.2e}", rel_err);
                                }
                            }
                            Err(e) => println!("Solve error: {}", e),
                        }
                    }
                    Err(e) => println!("Conversion error: {}", e),
                }
            }
            Err(e) => println!("Parse error: {}", e),
        }
        return;
    }

    // Full benchmark mode
    let dir = path.unwrap_or_else(|| ".".to_string());
    let dir_path = PathBuf::from(&dir);

    if !dir_path.exists() {
        eprintln!("Directory not found: {}", dir);
        eprintln!("\nTo run SDPLIB benchmarks, specify the path to SDPLIB .dat-s files:");
        eprintln!("  cargo run --release -p solver-bench -- sdplib --path /path/to/sdplib");
        return;
    }

    // Find all .dat-s files
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir_path)
        .expect("Failed to read directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p: &PathBuf| p.extension().map_or(false, |e| e == "dat-s"))
        .collect();

    files.sort();

    if files.is_empty() {
        eprintln!("No .dat-s files found in {}", dir);
        return;
    }

    let files: Vec<_> = if let Some(n) = limit {
        files.into_iter().take(n).collect()
    } else {
        files
    };

    println!("SDPLIB Benchmark Suite");
    println!("======================");
    println!("Found {} problems in {}\n", files.len(), dir);

    // Results table
    println!(
        "{:<20} {:>8} {:>8} {:>12} {:>12} {:>10} {:>8}",
        "Problem", "Status", "Iters", "Primal Obj", "Reference", "Rel Err", "Time(ms)"
    );
    println!("{}", "-".repeat(90));

    let mut total = 0;
    let mut optimal = 0;
    let mut almost_optimal = 0;
    let mut failed = 0;
    let mut total_time = 0.0;

    for file in &files {
        let name = file.file_stem().unwrap().to_str().unwrap();
        total += 1;

        match load_sdpa_file(file) {
            Ok(sdpa) => {
                let start = Instant::now();
                match solve_sdpa(&sdpa, &settings) {
                    Ok(result) => {
                        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                        total_time += elapsed;

                        let status_str = match result.status {
                            SolveStatus::Optimal => {
                                optimal += 1;
                                "Optimal"
                            }
                            SolveStatus::AlmostOptimal => {
                                almost_optimal += 1;
                                "Almost"
                            }
                            _ => {
                                failed += 1;
                                "FAILED"
                            }
                        };

                        let (ref_str, err_str) = if let Some(&ref_val) = reference.get(name) {
                            let rel_err = (result.primal_obj - ref_val).abs() / (1.0 + ref_val.abs());
                            (format!("{:.4e}", ref_val), format!("{:.2e}", rel_err))
                        } else {
                            ("-".to_string(), "-".to_string())
                        };

                        println!(
                            "{:<20} {:>8} {:>8} {:>12.4e} {:>12} {:>10} {:>8.1}",
                            name, status_str, result.iterations, result.primal_obj, ref_str, err_str, elapsed
                        );
                    }
                    Err(e) => {
                        failed += 1;
                        println!("{:<20} {:>8} {:>8} {:>12} {:>12} {:>10} {:>8}", name, "ERROR", "-", "-", "-", "-", "-");
                        eprintln!("  Error: {}", e);
                    }
                }
            }
            Err(e) => {
                failed += 1;
                println!("{:<20} {:>8} {:>8} {:>12} {:>12} {:>10} {:>8}", name, "PARSE", "-", "-", "-", "-", "-");
                eprintln!("  Parse error: {}", e);
            }
        }
    }

    println!("{}", "-".repeat(90));
    println!("\nSummary:");
    println!("  Total:        {}", total);
    println!("  Optimal:      {} ({:.1}%)", optimal, 100.0 * optimal as f64 / total as f64);
    println!("  Almost:       {} ({:.1}%)", almost_optimal, 100.0 * almost_optimal as f64 / total as f64);
    println!("  Failed:       {} ({:.1}%)", failed, 100.0 * failed as f64 / total as f64);
    println!("  Total time:   {:.1} ms", total_time);
    println!("  Avg time:     {:.1} ms/problem", total_time / total as f64);
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Random { max_iter, solver }) => {
            run_random_benchmarks(max_iter, solver);
        }
        Some(Commands::Benchmark { suite, path, limit, max_iter, problem, table, verbose, solver }) => {
            match suite {
                BenchmarkSuite::Mm => {
                    // Auto-detect path for Maros-Meszaros
                    let _path = path; // MM uses its own path logic via maros_meszaros module
                    run_maros_meszaros(limit, max_iter, problem, table, solver);
                }
                BenchmarkSuite::Sdplib => {
                    // Auto-detect path for SDPLIB
                    let sdplib_path = path.or_else(|| {
                        let default = std::path::Path::new("_data/sdplib");
                        if default.exists() {
                            Some("_data/sdplib".to_string())
                        } else {
                            None
                        }
                    });
                    run_sdplib_benchmarks(sdplib_path, limit, max_iter, problem, None, verbose);
                }
                BenchmarkSuite::Exp => {
                    run_exp_benchmarks(max_iter, verbose);
                }
            }
        }
        Some(Commands::Info { path }) => {
            show_qps_info(&path);
        }
        Some(Commands::Regression {
            max_iter,
            max_iter_fail,
            require_cache,
            solver,
            baseline_in,
            baseline_out,
            max_regression,
        }) => {
            run_regression_suite(
                max_iter,
                max_iter_fail,
                solver,
                require_cache,
                baseline_in,
                baseline_out,
                max_regression,
            );
        }
        None => {
            // Default: run random benchmarks with ipm1
            run_random_benchmarks(200, SolverChoice::Ipm1);
        }
    }
}
