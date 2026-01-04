//! Mészáros lptestset benchmark collections from SuiteSparse.
//!
//! These collections are useful for testing robustness and edge cases:
//! - INFEAS: Infeasibility detection tests
//! - PROBLEMATIC: Numerically challenging problems
//!
//! Source: https://sparse.tamu.edu/Meszaros

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use solver_core::{solve, SolveStatus, SolverSettings};

use crate::qps::{parse_qps, QpsProblem};

/// SuiteSparse Matrix Market base URL
const SUITESPARSE_BASE: &str = "https://suitesparse-collection-website.herokuapp.com/MM/Meszaros";

/// Mészáros INFEAS collection - infeasibility detection tests.
/// These problems are known to be infeasible.
pub const MESZAROS_INFEAS: &[&str] = &[
    "bgdbg1", "bgetam", "bgprtr", "box1", "chemcom", "cplex1", "cplex2", "ex72a", "ex73a",
    "forest6", "galenet", "gosh", "gran", "itest2", "itest6", "klein1", "klein2", "klein3",
    "mondou2", "pang", "pilot4i", "qual", "reactor", "refinery", "vol1", "woodinfe",
];

/// Mészáros PROBLEMATIC collection - numerically challenging problems.
/// These problems are feasible but may be ill-conditioned or degenerate.
pub const MESZAROS_PROBLEMATIC: &[&str] = &[
    "bas1lp", "cq5", "cq9", "deter0", "deter1", "deter2", "deter3", "deter4", "deter5", "deter6",
    "deter7", "deter8", "ex3sta1", "farm", "gams10a", "gams10am", "gams30a", "gams30am", "gams60a",
    "gams60am", "gas11", "iiasa", "jendrec1", "kleemin3", "kleemin4", "kleemin5", "kleemin6",
    "kleemin7", "kleemin8", "model1", "model10", "model2", "model3", "model4", "model5", "model6",
    "model7", "model8", "model9", "nemsemm1", "nsct1", "nsct2", "nsic1", "nsic2", "nsir1", "nsir2",
    "nug05", "nug06", "nug07", "nug08", "nug12", "nug15", "o9", "p0033", "p0040", "p0201", "p0282",
    "p0291", "p0548", "p2756", "primagaz", "problem", "progas", "qiulp", "reses2", "reses3",
    "rosen1", "rosen10", "rosen2", "rosen7", "rosen8", "route", "seymourl", "seymourl", "slptsk",
    "stoch1", "stoch2", "stoch3",
];

/// Expected status for a test problem.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExpectedStatus {
    Optimal,
    Infeasible,
    Unbounded,
    Unknown, // For problematic cases where status is uncertain
}

/// Result of running a single benchmark.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub n: usize,
    pub m: usize,
    pub nnz: usize,
    pub expected_status: ExpectedStatus,
    pub status: SolveStatus,
    pub status_correct: bool,
    pub iterations: usize,
    pub obj_val: f64,
    pub mu: f64,
    pub solve_time_ms: f64,
    pub error: Option<String>,
}

/// Summary statistics for a benchmark suite.
#[derive(Debug, Clone)]
pub struct BenchmarkSummary {
    pub total: usize,
    pub correct_status: usize,
    pub optimal: usize,
    pub infeasible: usize,
    pub max_iters: usize,
    pub numerical_error: usize,
    pub other: usize,
    pub parse_errors: usize,
    pub avg_iters: f64,
    pub avg_time_ms: f64,
}

/// Get the cache directory for Mészáros problems.
fn get_cache_dir(collection: &str) -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(format!(".cache/minix-bench/meszaros/{}", collection))
}

/// Download an MPS file from SuiteSparse.
fn download_mps(collection: &str, name: &str) -> Result<PathBuf> {
    let cache_dir = get_cache_dir(collection);
    fs::create_dir_all(&cache_dir)?;

    let mps_path = cache_dir.join(format!("{}.mps", name));

    // Check if already cached
    if mps_path.exists() {
        return Ok(mps_path);
    }

    // SuiteSparse stores files as .tar.gz containing problem/problem.mps
    let url = format!("{}/{}/{}.tar.gz", SUITESPARSE_BASE, collection, name);

    eprintln!("Downloading {}...", url);

    let tar_gz_path = cache_dir.join(format!("{}.tar.gz", name));

    let output = Command::new("curl")
        .args(["-sL", "--max-time", "60", "-o"])
        .arg(&tar_gz_path)
        .arg(&url)
        .output()
        .context("Failed to run curl")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("curl failed: {}", stderr);
    }

    if !tar_gz_path.exists() || fs::metadata(&tar_gz_path)?.len() == 0 {
        bail!("Download failed: empty or missing file");
    }

    // Extract MPS file from tar.gz
    // The archive contains: name/name.mps
    let _ = Command::new("tar")
        .args(["-xzf"])
        .arg(&tar_gz_path)
        .arg("-C")
        .arg(&cache_dir)
        .arg("--strip-components=1")
        .arg(format!("{}/{}.mps", name, name))
        .output()
        .context("Failed to extract tar.gz")?;

    // Clean up tar.gz
    let _ = fs::remove_file(&tar_gz_path);

    if !mps_path.exists() {
        // Try alternative extraction (some files may have different structure)
        let _ = Command::new("tar")
            .args(["-xzf"])
            .arg(&tar_gz_path)
            .arg("-C")
            .arg(&cache_dir)
            .output();

        // Try to find the MPS file
        if let Ok(entries) = fs::read_dir(&cache_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let inner_mps = path.join(format!("{}.mps", name));
                    if inner_mps.exists() {
                        fs::rename(&inner_mps, &mps_path)?;
                        let _ = fs::remove_dir_all(&path);
                        break;
                    }
                }
            }
        }
    }

    if !mps_path.exists() {
        bail!("Failed to extract MPS file from archive");
    }

    Ok(mps_path)
}

/// Load a problem from a Mészáros collection.
pub fn load_problem(collection: &str, name: &str) -> Result<QpsProblem> {
    let path = download_mps(collection, name)?;
    parse_qps(&path)
}

/// Run a single benchmark with expected status.
pub fn run_single_with_expected(
    collection: &str,
    name: &str,
    expected: ExpectedStatus,
    settings: &SolverSettings,
) -> BenchmarkResult {
    // Load problem
    let qps = match load_problem(collection, name) {
        Ok(qps) => qps,
        Err(e) => {
            return BenchmarkResult {
                name: name.to_string(),
                n: 0,
                m: 0,
                nnz: 0,
                expected_status: expected,
                status: SolveStatus::NumericalError,
                status_correct: false,
                iterations: 0,
                obj_val: 0.0,
                mu: 0.0,
                solve_time_ms: 0.0,
                error: Some(format!("Load error: {}", e)),
            };
        }
    };

    // Convert to ProblemData
    let prob = match qps.to_problem_data() {
        Ok(p) => p,
        Err(e) => {
            return BenchmarkResult {
                name: name.to_string(),
                n: qps.n,
                m: qps.m,
                nnz: qps.a_triplets.len(),
                expected_status: expected,
                status: SolveStatus::NumericalError,
                status_correct: false,
                iterations: 0,
                obj_val: 0.0,
                mu: 0.0,
                solve_time_ms: 0.0,
                error: Some(format!("Conversion error: {}", e)),
            };
        }
    };

    let n = prob.num_vars();
    let m = prob.num_constraints();
    let nnz = prob.A.nnz();

    // Solve
    let start = Instant::now();
    let result = solve(&prob, settings);
    let elapsed = start.elapsed();

    match result {
        Ok(res) => {
            let status_correct = match expected {
                ExpectedStatus::Optimal => res.status == SolveStatus::Optimal,
                ExpectedStatus::Infeasible => {
                    res.status == SolveStatus::PrimalInfeasible
                        || res.status == SolveStatus::DualInfeasible
                }
                ExpectedStatus::Unbounded => res.status == SolveStatus::DualInfeasible,
                ExpectedStatus::Unknown => true, // Accept any status for unknown
            };

            BenchmarkResult {
                name: name.to_string(),
                n,
                m,
                nnz,
                expected_status: expected,
                status: res.status,
                status_correct,
                iterations: res.info.iters,
                obj_val: res.obj_val,
                mu: res.info.mu,
                solve_time_ms: elapsed.as_secs_f64() * 1000.0,
                error: None,
            }
        }
        Err(e) => BenchmarkResult {
            name: name.to_string(),
            n,
            m,
            nnz,
            expected_status: expected,
            status: SolveStatus::NumericalError,
            status_correct: false,
            iterations: 0,
            obj_val: 0.0,
            mu: 0.0,
            solve_time_ms: elapsed.as_secs_f64() * 1000.0,
            error: Some(format!("Solve error: {}", e)),
        },
    }
}

/// Run the INFEAS suite (infeasibility detection tests).
pub fn run_infeas_suite(settings: &SolverSettings, limit: Option<usize>) -> Vec<BenchmarkResult> {
    let problems: Vec<_> = if let Some(limit) = limit {
        MESZAROS_INFEAS.iter().take(limit).collect()
    } else {
        MESZAROS_INFEAS.iter().collect()
    };

    let mut results = Vec::with_capacity(problems.len());

    for (i, name) in problems.iter().enumerate() {
        eprint!("[{}/{}] {}... ", i + 1, problems.len(), name);

        let result = run_single_with_expected("INFEAS", name, ExpectedStatus::Infeasible, settings);

        if let Some(ref err) = result.error {
            eprintln!("ERROR: {}", err);
        } else {
            let correctness = if result.status_correct { "✓" } else { "✗" };
            eprintln!(
                "{} {:?} (expected {:?}) in {} iters, {:.1}ms",
                correctness,
                result.status,
                result.expected_status,
                result.iterations,
                result.solve_time_ms
            );
        }

        results.push(result);
    }

    results
}

/// Run the PROBLEMATIC suite (numerically challenging problems).
pub fn run_problematic_suite(
    settings: &SolverSettings,
    limit: Option<usize>,
) -> Vec<BenchmarkResult> {
    let problems: Vec<_> = if let Some(limit) = limit {
        MESZAROS_PROBLEMATIC.iter().take(limit).collect()
    } else {
        MESZAROS_PROBLEMATIC.iter().collect()
    };

    let mut results = Vec::with_capacity(problems.len());

    for (i, name) in problems.iter().enumerate() {
        eprint!("[{}/{}] {}... ", i + 1, problems.len(), name);

        // PROBLEMATIC problems may have various outcomes
        let result =
            run_single_with_expected("PROBLEMATIC", name, ExpectedStatus::Unknown, settings);

        if let Some(ref err) = result.error {
            eprintln!("ERROR: {}", err);
        } else {
            eprintln!(
                "{:?} in {} iters, {:.1}ms",
                result.status, result.iterations, result.solve_time_ms
            );
        }

        results.push(result);
    }

    results
}

/// Compute summary statistics.
pub fn compute_summary(results: &[BenchmarkResult]) -> BenchmarkSummary {
    let mut summary = BenchmarkSummary {
        total: results.len(),
        correct_status: 0,
        optimal: 0,
        infeasible: 0,
        max_iters: 0,
        numerical_error: 0,
        other: 0,
        parse_errors: 0,
        avg_iters: 0.0,
        avg_time_ms: 0.0,
    };

    let mut total_iters = 0;
    let mut total_time = 0.0;
    let mut solved_count = 0;

    for r in results {
        if r.error.is_some() {
            summary.parse_errors += 1;
            continue;
        }

        if r.status_correct {
            summary.correct_status += 1;
        }

        match r.status {
            SolveStatus::Optimal => summary.optimal += 1,
            SolveStatus::PrimalInfeasible | SolveStatus::DualInfeasible => summary.infeasible += 1,
            SolveStatus::MaxIters => summary.max_iters += 1,
            SolveStatus::NumericalError => summary.numerical_error += 1,
            _ => summary.other += 1,
        }

        total_iters += r.iterations;
        total_time += r.solve_time_ms;
        solved_count += 1;
    }

    if solved_count > 0 {
        summary.avg_iters = total_iters as f64 / solved_count as f64;
        summary.avg_time_ms = total_time / solved_count as f64;
    }

    summary
}

/// Print results table.
pub fn print_results_table(results: &[BenchmarkResult]) {
    println!("\n{:-<100}", "");
    println!(
        "{:<20} {:>8} {:>8} {:>12} {:>10} {:>12} {:>10}",
        "Problem", "n", "m", "Expected", "Status", "Correct?", "Time(ms)"
    );
    println!("{:-<100}", "");

    for r in results {
        let status_str = if r.error.is_some() {
            "ERROR".to_string()
        } else {
            format!("{:?}", r.status)
        };

        let correct_str = if r.error.is_some() {
            "-".to_string()
        } else if r.status_correct {
            "✓".to_string()
        } else {
            "✗".to_string()
        };

        println!(
            "{:<20} {:>8} {:>8} {:>12?} {:>10} {:>12} {:>10.1}",
            r.name, r.n, r.m, r.expected_status, status_str, correct_str, r.solve_time_ms
        );
    }

    println!("{:-<100}", "");
}

/// Print summary.
pub fn print_summary(summary: &BenchmarkSummary, suite_name: &str) {
    println!("\nMészáros {} Summary", suite_name);
    println!("{}", "=".repeat(30 + suite_name.len()));
    println!("Total problems:     {}", summary.total);
    println!("Parse errors:       {}", summary.parse_errors);
    println!();
    println!(
        "Correct status:     {} ({:.1}%)",
        summary.correct_status,
        100.0 * summary.correct_status as f64
            / (summary.total - summary.parse_errors).max(1) as f64
    );
    println!();
    println!("Optimal:            {}", summary.optimal);
    println!("Infeasible:         {}", summary.infeasible);
    println!("Max iterations:     {}", summary.max_iters);
    println!("Numerical error:    {}", summary.numerical_error);
    println!("Other:              {}", summary.other);
    println!();
    println!("Avg iterations:     {:.1}", summary.avg_iters);
    println!("Avg solve time:     {:.1} ms", summary.avg_time_ms);
}
