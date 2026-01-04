//! Maros-Meszaros QP benchmark suite runner.
//!
//! Downloads and runs the standard Maros-Meszaros test set of 138 QP problems.
//! Prefers local MAT files from ClarabelBenchmarks if available.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use solver_core::{solve, ProblemData, SolveResult, SolveStatus, SolverSettings};

use crate::matparser;
use crate::qps::{parse_qps, QpsProblem};

/// URL for Maros-Meszaros QPS files (from GitHub mirror)
const MM_BASE_URL: &str =
    "https://raw.githubusercontent.com/YimingYAN/QP-Test-Problems/master/QPS_Files";

/// Known Maros-Meszaros problem names (138 problems)
const MM_PROBLEMS: &[&str] = &[
    "AUG2D", "AUG2DC", "AUG2DCQP", "AUG2DQP", "AUG3D", "AUG3DC", "AUG3DCQP", "AUG3DQP", "BOYD1",
    "BOYD2", "CONT-050", "CONT-100", "CONT-101", "CONT-200", "CONT-201", "CONT-300", "CVXQP1_L",
    "CVXQP1_M", "CVXQP1_S", "CVXQP2_L", "CVXQP2_M", "CVXQP2_S", "CVXQP3_L", "CVXQP3_M", "CVXQP3_S",
    "DPKLO1", "DTOC3", "DUAL1", "DUAL2", "DUAL3", "DUAL4", "DUALC1", "DUALC2", "DUALC5", "DUALC8",
    "EXDATA", "GOULDQP2", "GOULDQP3", "HS118", "HS21", "HS268", "HS35", "HS35MOD", "HS51", "HS52",
    "HS53", "HS76", "HUES-MOD", "HUESTIS", "KSIP", "LASER", "LISWET1", "LISWET10", "LISWET11",
    "LISWET12", "LISWET2", "LISWET3", "LISWET4", "LISWET5", "LISWET6", "LISWET7", "LISWET8",
    "LISWET9", "LOTSCHD", "MOSARQP1", "MOSARQP2", "POWELL20", "PRIMAL1", "PRIMAL2", "PRIMAL3",
    "PRIMAL4", "PRIMALC1", "PRIMALC2", "PRIMALC5", "PRIMALC8", "Q25FV47", "QADLITTL", "QAFIRO",
    "QBANDM", "QBEACONF", "QBORE3D", "QBRANDY", "QCAPRI", "QE226", "QETAMACR", "QFFFFF80",
    "QFORPLAN", "QGFRDXPN", "QGROW15", "QGROW22", "QGROW7", "QISRAEL", "QPCBLEND", "QPCBOEI1",
    "QPCBOEI2", "QPCSTAIR", "QPILOTNO", "QRECIPE", "QSC205", "QSCAGR25", "QSCAGR7", "QSCFXM1",
    "QSCFXM2", "QSCFXM3", "QSCORPIO", "QSCRS8", "QSCSD1", "QSCSD6", "QSCSD8", "QSCTAP1", "QSCTAP2",
    "QSCTAP3", "QSEBA", "QSHARE1B", "QSHARE2B", "QSHELL", "QSHIP04L", "QSHIP04S", "QSHIP08L",
    "QSHIP08S", "QSHIP12L", "QSHIP12S", "QSIERRA", "QSTAIR", "QSTANDAT", "S268", "STADAT1",
    "STADAT2", "STADAT3", "STCQP1", "STCQP2", "TAME", "UBH1", "VALUES", "YAO", "ZECEVIC2",
];

#[inline]
fn inf_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

fn print_diagnostics(name: &str, prob: &ProblemData, res: &SolveResult) {
    let n = prob.num_vars();
    let m = prob.num_constraints();

    let mut r_p = res.s.clone();
    for i in 0..m {
        r_p[i] -= prob.b[i];
    }
    for (&val, (row, col)) in prob.A.iter() {
        r_p[row] += val * res.x[col];
    }

    let mut p_x = vec![0.0; n];
    if let Some(ref p) = prob.P {
        for col in 0..n {
            if let Some(col_view) = p.outer_view(col) {
                for (row, &val) in col_view.iter() {
                    if row == col {
                        p_x[row] += val * res.x[col];
                    } else {
                        p_x[row] += val * res.x[col];
                        p_x[col] += val * res.x[row];
                    }
                }
            }
        }
    }

    let mut r_d = vec![0.0; n];
    for i in 0..n {
        r_d[i] = p_x[i] + prob.q[i];
    }
    for (&val, (row, col)) in prob.A.iter() {
        r_d[col] += val * res.z[row];
    }

    let rp_inf = inf_norm(&r_p);
    let rd_inf = inf_norm(&r_d);
    let x_inf = inf_norm(&res.x);
    let s_inf = inf_norm(&res.s);
    let z_inf = inf_norm(&res.z);
    let b_inf = inf_norm(&prob.b);
    let q_inf = inf_norm(&prob.q);
    let primal_scale = (b_inf + x_inf + s_inf).max(1.0);
    let dual_scale = (q_inf + x_inf + z_inf).max(1.0);

    let xpx = dot(&res.x, &p_x);
    let qtx = dot(&prob.q, &res.x);
    let btz = dot(&prob.b, &res.z);
    let primal_obj = 0.5 * xpx + qtx;
    let dual_obj = -0.5 * xpx - btz;
    let gap = (primal_obj - dual_obj).abs();
    let gap_scale = primal_obj.abs().max(dual_obj.abs()).max(1.0);

    println!("Diagnostics for {}:", name);
    println!(
        "  r_p_inf={:.3e} (scale {:.3e}), r_d_inf={:.3e} (scale {:.3e})",
        rp_inf, primal_scale, rd_inf, dual_scale
    );
    println!(
        "  rel_p={:.3e}, rel_d={:.3e}",
        rp_inf / primal_scale,
        rd_inf / dual_scale
    );
    println!(
        "  gap={:.3e}, gap_rel={:.3e}, obj_p={:.3e}, obj_d={:.3e}",
        gap,
        gap / gap_scale,
        primal_obj,
        dual_obj
    );
}

/// Result of running a single benchmark problem
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Problem name
    pub name: String,
    /// Number of variables
    pub n: usize,
    /// Number of constraints
    pub m: usize,
    /// Solve status
    pub status: SolveStatus,
    /// Number of iterations
    pub iterations: usize,
    /// Objective value
    pub obj_val: f64,
    /// Final mu
    pub mu: f64,
    /// Solve time in milliseconds
    pub solve_time_ms: f64,
    /// Error message if any
    pub error: Option<String>,
}

/// Summary statistics for benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkSummary {
    /// Total problems attempted
    pub total: usize,
    /// Problems solved to optimality
    pub optimal: usize,
    /// Problems hitting max iterations
    pub max_iters: usize,
    /// Problems with numerical errors
    pub numerical_errors: usize,
    /// Problems that failed to parse
    pub parse_errors: usize,
    /// Total solve time in seconds
    pub total_time_s: f64,
    /// Geometric mean of iterations (for solved problems)
    pub geom_mean_iters: f64,
}

/// Get the cache directory for benchmark problems
fn get_cache_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("minix-bench")
        .join("maros-meszaros")
}

/// Download a QPS file if not cached
fn download_qps(name: &str) -> Result<PathBuf> {
    let cache_dir = get_cache_dir();
    fs::create_dir_all(&cache_dir)?;

    let filename = format!("{}.QPS", name);
    let cached_path = cache_dir.join(&filename);

    if cached_path.exists() {
        return Ok(cached_path);
    }

    // Try downloading from GitHub mirror (no .gz)
    let url = format!("{}/{}.QPS", MM_BASE_URL, name);

    eprintln!("Downloading {}...", name);

    let output = std::process::Command::new("curl")
        .args(["-sL", "--max-time", "30", &url])
        .output()
        .context("Failed to run curl")?;

    if output.status.success() && !output.stdout.is_empty() {
        // Check if it's valid QPS content (starts with NAME or has ROWS section)
        let content = String::from_utf8_lossy(&output.stdout);
        if content.contains("ROWS") || content.starts_with("NAME") {
            fs::write(&cached_path, &output.stdout)?;
            return Ok(cached_path);
        }
    }

    // Try lowercase
    let url = format!("{}/{}.qps", MM_BASE_URL, name);
    let output = std::process::Command::new("curl")
        .args(["-sL", "--max-time", "30", &url])
        .output()
        .context("Failed to run curl")?;

    if output.status.success() && !output.stdout.is_empty() {
        let content = String::from_utf8_lossy(&output.stdout);
        if content.contains("ROWS") || content.starts_with("NAME") {
            fs::write(&cached_path, &output.stdout)?;
            return Ok(cached_path);
        }
    }

    Err(anyhow::anyhow!(
        "Failed to download {} - file not found or invalid",
        name
    ))
}

/// Get the local ClarabelBenchmarks MAT directory if available.
fn get_local_mat_dir() -> Option<PathBuf> {
    // Check relative to crate directory first
    if let Ok(crate_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let repo_dir = PathBuf::from(&crate_dir)
            .parent()
            .map(|p| p.join("ClarabelBenchmarks/src/problem_sets/maros/targets/mat"));
        if let Some(ref dir) = repo_dir {
            if dir.exists() {
                return repo_dir;
            }
        }
    }

    // Check relative to current working directory
    if let Ok(cwd) = std::env::current_dir() {
        let cwd_repo = cwd.join("ClarabelBenchmarks/src/problem_sets/maros/targets/mat");
        if cwd_repo.exists() {
            return Some(cwd_repo);
        }
    }

    None
}

/// Load a problem, preferring MAT files from ClarabelBenchmarks.
/// Note: MAT file loading may fail for sparse matrices (matfile crate limitation).
pub fn load_problem(name: &str) -> Result<QpsProblem> {
    // Try local MAT file from ClarabelBenchmarks first
    if let Some(mat_dir) = get_local_mat_dir() {
        let mat_path = mat_dir.join(format!("{}.mat", name));
        if mat_path.exists() {
            // Try to load from MAT file (may fail if matrices are sparse)
            match matparser::parse_mat(&mat_path) {
                Ok(osqp) => {
                    return Ok(QpsProblem {
                        name: osqp.name,
                        n: osqp.n,
                        m: osqp.m,
                        obj_sense: 1.0, // OSQP format is minimization
                        p_triplets: osqp.p_triplets,
                        a_triplets: osqp.a_triplets,
                        q: osqp.q,
                        con_lower: osqp.l,
                        con_upper: osqp.u,
                        var_lower: vec![-1e20; osqp.n], // No explicit var bounds in MAT format
                        var_upper: vec![1e20; osqp.n],
                        var_names: (0..osqp.n).map(|i| format!("x{}", i)).collect(),
                        con_names: (0..osqp.m).map(|i| format!("c{}", i)).collect(),
                    });
                }
                Err(_) => {
                    // MAT loading failed (likely sparse matrices), fall back to QPS
                }
            }
        }
    }

    // Check for local QPS file
    let local_paths = [
        PathBuf::from(format!("{}.QPS", name)),
        PathBuf::from(format!("{}.qps", name)),
        PathBuf::from(format!("data/{}.QPS", name)),
        PathBuf::from(format!("data/{}.qps", name)),
    ];

    for path in &local_paths {
        if path.exists() {
            return parse_qps(path);
        }
    }

    // Try cache or download QPS
    let path = download_qps(name)?;
    parse_qps(&path)
}

/// Run a single benchmark problem
pub fn run_single(name: &str, settings: &SolverSettings) -> BenchmarkResult {
    // Load and parse problem
    let qps = match load_problem(name) {
        Ok(q) => q,
        Err(e) => {
            return BenchmarkResult {
                name: name.to_string(),
                n: 0,
                m: 0,
                status: SolveStatus::NumericalError,
                iterations: 0,
                obj_val: f64::NAN,
                mu: f64::NAN,
                solve_time_ms: 0.0,
                error: Some(format!("Parse error: {}", e)),
            };
        }
    };

    // Convert to conic form
    let prob = match qps.to_problem_data() {
        Ok(p) => p,
        Err(e) => {
            return BenchmarkResult {
                name: name.to_string(),
                n: qps.n,
                m: qps.m,
                status: SolveStatus::NumericalError,
                iterations: 0,
                obj_val: f64::NAN,
                mu: f64::NAN,
                solve_time_ms: 0.0,
                error: Some(format!("Conversion error: {}", e)),
            };
        }
    };

    // Solve
    let start = Instant::now();
    let result = solve(&prob, settings);
    let elapsed = start.elapsed();

    let diagnostics_enabled = std::env::var("MINIX_DIAGNOSTICS").is_ok();

    match result {
        Ok(res) => {
            if diagnostics_enabled || res.status != SolveStatus::Optimal {
                print_diagnostics(name, &prob, &res);
            }

            BenchmarkResult {
                name: name.to_string(),
                n: prob.num_vars(),
                m: prob.num_constraints(),
                status: res.status,
                iterations: res.info.iters,
                obj_val: res.obj_val,
                mu: res.info.mu,
                solve_time_ms: elapsed.as_secs_f64() * 1000.0,
                error: None,
            }
        }
        Err(e) => BenchmarkResult {
            name: name.to_string(),
            n: prob.num_vars(),
            m: prob.num_constraints(),
            status: SolveStatus::NumericalError,
            iterations: 0,
            obj_val: f64::NAN,
            mu: f64::NAN,
            solve_time_ms: elapsed.as_secs_f64() * 1000.0,
            error: Some(e.to_string()),
        },
    }
}

/// Run full Maros-Meszaros benchmark suite
pub fn run_full_suite(
    settings: &SolverSettings,
    max_problems: Option<usize>,
) -> Vec<BenchmarkResult> {
    let problems: Vec<&str> = MM_PROBLEMS
        .iter()
        .take(max_problems.unwrap_or(MM_PROBLEMS.len()))
        .copied()
        .collect();

    let mut results = Vec::with_capacity(problems.len());

    for (i, name) in problems.iter().enumerate() {
        eprint!("[{}/{}] {} ... ", i + 1, problems.len(), name);
        let result = run_single(name, settings);

        let status_str = match result.status {
            SolveStatus::Optimal => "✓",
            SolveStatus::MaxIters => "M",
            SolveStatus::NumericalError => "N",
            _ => "?",
        };

        if result.error.is_some() {
            eprintln!("ERROR");
        } else {
            eprintln!(
                "{} ({} iters, {:.1}ms)",
                status_str, result.iterations, result.solve_time_ms
            );
        }

        results.push(result);
    }

    results
}

/// Compute summary statistics
pub fn compute_summary(results: &[BenchmarkResult]) -> BenchmarkSummary {
    let total = results.len();
    let mut optimal = 0;
    let mut max_iters = 0;
    let mut numerical_errors = 0;
    let mut parse_errors = 0;
    let mut total_time_s = 0.0;
    let mut iter_log_sum = 0.0;
    let mut iter_count = 0;

    for r in results {
        total_time_s += r.solve_time_ms / 1000.0;

        if r.error.is_some() && r.error.as_ref().unwrap().contains("Parse") {
            parse_errors += 1;
            continue;
        }

        match r.status {
            SolveStatus::Optimal => {
                optimal += 1;
                if r.iterations > 0 {
                    iter_log_sum += (r.iterations as f64).ln();
                    iter_count += 1;
                }
            }
            SolveStatus::MaxIters => max_iters += 1,
            SolveStatus::NumericalError => numerical_errors += 1,
            _ => {}
        }
    }

    let geom_mean_iters = if iter_count > 0 {
        (iter_log_sum / iter_count as f64).exp()
    } else {
        0.0
    };

    BenchmarkSummary {
        total,
        optimal,
        max_iters,
        numerical_errors,
        parse_errors,
        total_time_s,
        geom_mean_iters,
    }
}

/// Print results summary
pub fn print_summary(summary: &BenchmarkSummary) {
    println!("\n{}", "=".repeat(60));
    println!("Maros-Meszaros Benchmark Summary");
    println!("{}", "=".repeat(60));
    println!("Total problems:      {}", summary.total);
    println!(
        "Optimal:             {} ({:.1}%)",
        summary.optimal,
        100.0 * summary.optimal as f64 / summary.total as f64
    );
    println!("Max iterations:      {}", summary.max_iters);
    println!("Numerical errors:    {}", summary.numerical_errors);
    println!("Parse errors:        {}", summary.parse_errors);
    println!("Total time:          {:.2}s", summary.total_time_s);
    println!("Geom mean iters:     {:.1}", summary.geom_mean_iters);
    println!("{}", "=".repeat(60));
}

/// Print detailed results table
pub fn print_results_table(results: &[BenchmarkResult]) {
    println!(
        "\n{:<15} {:>6} {:>8} {:>8} {:>10} {:>12} {:>10}",
        "Problem", "n", "m", "Status", "Iters", "Obj", "Time(ms)"
    );
    println!("{}", "-".repeat(75));

    for r in results {
        let status_str = match r.status {
            SolveStatus::Optimal => "Optimal",
            SolveStatus::MaxIters => "MaxIter",
            SolveStatus::NumericalError => "NumErr",
            SolveStatus::PrimalInfeasible => "PrimInf",
            SolveStatus::DualInfeasible => "DualInf",
            _ => "Other",
        };

        if r.error.is_some() {
            println!(
                "{:<15} {:>6} {:>8} {:>8} {:>10} {:>12} {:>10}",
                r.name, "-", "-", "Error", "-", "-", "-"
            );
        } else {
            println!(
                "{:<15} {:>6} {:>8} {:>8} {:>10} {:>12.4e} {:>10.1}",
                r.name, r.n, r.m, status_str, r.iterations, r.obj_val, r.solve_time_ms
            );
        }
    }
}
