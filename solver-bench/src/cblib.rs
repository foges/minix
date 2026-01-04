//! CBLIB (Conic Benchmark Library) benchmark infrastructure.
//!
//! Runs SOCP benchmarks from CBLIB. Prefers local files from ClarabelBenchmarks
//! repo if available, otherwise downloads from https://cblib.zib.de/

use crate::cbf;
use anyhow::{bail, Context, Result};
use flate2::read::GzDecoder;
use solver_core::{solve, SolveStatus, SolverSettings};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

const CBLIB_BASE_URL: &str = "https://cblib.zib.de/download/all";

/// SOCP problems from CBLIB (no SDP, no integer variables, no power cones).
/// Uses Q (second-order) and QR (rotated second-order) cones.
/// Reference: https://cblib.zib.de/download/readme.txt
///
/// File names use underscores (e.g., chainsing_1000_1) to match ClarabelBenchmarks.
pub const CBLIB_SOCP_PROBLEMS: &[&str] = &[
    // Chained singular function (Kobayashi, Kim, Kojima 2008)
    // Uses QR (rotated second-order) cones + NonPos constraints
    "chainsing_1000_1",
    "chainsing_1000_2",
    "chainsing_1000_3",
    // Antenna array calibration (Coleman & Vanderbei 1999)
    // SOC cones, ~2.4k vars
    "nb",
    "nb_L1",
    "nb_L2_bessel",
    // Loaded plastic plates collapse states (Andersen 1998, Christiansen 1999)
    // SOC cones, various sizes
    "nql30",
    "nql60",
    "nql90",
    "qssp30",
    "qssp60",
    // Structural engineering (shear wall)
    "db_shear_wall",
    // Surgical scheduling (SOC relaxation)
    "sched_50_50_orig",
    "sched_50_50_scaled",
    "sched_100_50_orig",
    "sched_100_50_scaled",
    "sched_100_100_orig",
    // Note: sambal has very small SOC cones that work well
    "sambal",
];

/// Large SOCP problems (may require more memory/time)
pub const CBLIB_SOCP_LARGE: &[&str] = &[
    // Larger chainsing variants
    "chainsing_10000_1",
    "chainsing_10000_2",
    "chainsing_10000_3",
    // Larger plate problems
    "nql180",
    "qssp90",
    "qssp180",
    // Larger scheduling
    "sched_100_100_scaled",
    "sched_200_100_orig",
    "sched_200_100_scaled",
    // Structural problems
    "beam7",
    "db_joint_soerensen",
    "db_plane_strain_prism",
    "db_plate_yield_line",
    "db_plate_yield_line_fox",
    // Antenna nb_L2
    "nb_L2",
    // Strictmin problems
    "strictmin_2D_43_dual",
    "strictmin_2D_43_primal",
];

/// Mittelmann "Large SOCP Benchmark" curated subset.
/// From: https://plato.asu.edu/ftp/socp.html
/// These are the larger, more challenging problems used for solver comparison.
pub const CBLIB_MITTELMANN: &[&str] = &[
    // Very large chainsing (from Mittelmann benchmark)
    "chainsing_50000_1",
    "chainsing_50000_2",
    "chainsing_50000_3",
    // Joint FC problems (Femur bone finite element)
    "joint_FC_5",
    "joint_FC_7",
    "joint_FC_8",
    "joint_FC_9",
    "joint_FC_12",
    // Joint HO problems (Hip bone finite element)
    "joint_HO_01",
    "joint_HO_02",
    "joint_HO_03",
    "joint_HO_04",
    "joint_HO_05",
    "joint_HO_12",
    "joint_HO_13",
    "joint_HO_14",
    "joint_HO_18",
    "joint_HO_23",
    "joint_HO_24",
    "joint_HO_25",
    "joint_HO_26",
    "joint_HO_27",
    "joint_HO_28",
    "joint_HO_29",
];

/// Result of running a single CBLIB benchmark.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub n: usize,
    pub m: usize,
    pub nnz: usize,
    pub status: SolveStatus,
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
    pub optimal: usize,
    pub max_iters: usize,
    pub numerical_error: usize,
    pub other: usize,
    pub parse_errors: usize,
    pub avg_iters: f64,
    pub avg_time_ms: f64,
}

/// Get the cache directory for CBLIB problems.
fn get_cache_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache/minix-bench/cblib")
}

/// Get the local ClarabelBenchmarks directory if available.
fn get_local_cblib_dir() -> Option<PathBuf> {
    // Check relative to crate directory first
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let repo_dir = PathBuf::from(crate_dir)
        .parent()?
        .join("ClarabelBenchmarks/src/problem_sets/cblib/targets/socp");

    if repo_dir.exists() {
        return Some(repo_dir);
    }

    // Check relative to current working directory
    let cwd = std::env::current_dir().ok()?;
    let cwd_repo = cwd.join("ClarabelBenchmarks/src/problem_sets/cblib/targets/socp");
    if cwd_repo.exists() {
        return Some(cwd_repo);
    }

    None
}

/// Find a CBF file, preferring local ClarabelBenchmarks copy.
fn find_or_download_cbf(name: &str) -> Result<PathBuf> {
    let cache_dir = get_cache_dir();
    fs::create_dir_all(&cache_dir)?;

    let cbf_path = cache_dir.join(format!("{}.cbf", name));

    // Check if already cached (uncompressed)
    if cbf_path.exists() {
        return Ok(cbf_path);
    }

    // Try to find in local ClarabelBenchmarks repo
    if let Some(local_dir) = get_local_cblib_dir() {
        // Check standard directory
        let local_gz = local_dir.join(format!("{}.cbf.gz", name));
        if local_gz.exists() {
            return decompress_to_cache(&local_gz, &cbf_path);
        }

        // Check large subdirectory
        let large_gz = local_dir.join("large").join(format!("{}.cbf.gz", name));
        if large_gz.exists() {
            return decompress_to_cache(&large_gz, &cbf_path);
        }
    }

    // Fall back to downloading
    download_cbf_from_url(name, &cbf_path)
}

/// Decompress a .cbf.gz file to the cache directory.
fn decompress_to_cache(gz_path: &PathBuf, cbf_path: &PathBuf) -> Result<PathBuf> {
    eprintln!("Loading from local: {}", gz_path.display());

    let gz_file = File::open(gz_path)?;
    let mut decoder = GzDecoder::new(gz_file);
    let mut content = Vec::new();
    decoder
        .read_to_end(&mut content)
        .context("Failed to decompress CBF file")?;

    let mut cbf_file = File::create(cbf_path)?;
    cbf_file.write_all(&content)?;

    Ok(cbf_path.clone())
}

/// Download a CBF file from CBLIB website.
fn download_cbf_from_url(name: &str, cbf_path: &PathBuf) -> Result<PathBuf> {
    let cache_dir = cbf_path.parent().unwrap();

    // Try underscore version first (ClarabelBenchmarks naming)
    let name_underscore = name.replace('-', "_");
    let name_hyphen = name.replace('_', "-");

    for try_name in [&name_underscore, &name_hyphen] {
        let url = format!("{}/{}.cbf.gz", CBLIB_BASE_URL, try_name);
        let gz_path = cache_dir.join(format!("{}.cbf.gz", try_name));

        eprintln!("Downloading {}...", url);

        let output = Command::new("curl")
            .args(["-sL", "--max-time", "60", "-o"])
            .arg(&gz_path)
            .arg(&url)
            .output()
            .context("Failed to run curl")?;

        if output.status.success() && gz_path.exists() && fs::metadata(&gz_path)?.len() > 0 {
            // Decompress
            let gz_file = File::open(&gz_path)?;
            let mut decoder = GzDecoder::new(gz_file);
            let mut content = Vec::new();
            if decoder.read_to_end(&mut content).is_ok() {
                let mut cbf_file = File::create(cbf_path)?;
                cbf_file.write_all(&content)?;
                let _ = fs::remove_file(&gz_path);
                return Ok(cbf_path.clone());
            }
        }

        let _ = fs::remove_file(&gz_path);
    }

    bail!("Download failed for {}: could not fetch from CBLIB", name);
}

/// Load a CBLIB problem.
pub fn load_problem(name: &str) -> Result<cbf::CbfProblem> {
    let path = find_or_download_cbf(name)?;
    cbf::parse_cbf(&path)
}

/// Run a single CBLIB benchmark.
pub fn run_single(name: &str, settings: &SolverSettings) -> BenchmarkResult {
    // Load problem
    let cbf = match load_problem(name) {
        Ok(cbf) => cbf,
        Err(e) => {
            return BenchmarkResult {
                name: name.to_string(),
                n: 0,
                m: 0,
                nnz: 0,
                status: SolveStatus::NumericalError,
                iterations: 0,
                obj_val: 0.0,
                mu: 0.0,
                solve_time_ms: 0.0,
                error: Some(format!("Parse error: {}", e)),
            };
        }
    };

    // Convert to ProblemData
    let prob = match cbf::to_problem_data(&cbf) {
        Ok(p) => p,
        Err(e) => {
            return BenchmarkResult {
                name: name.to_string(),
                n: cbf.n,
                m: cbf.m,
                nnz: cbf.a_triplets.len(),
                status: SolveStatus::NumericalError,
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
        Ok(res) => BenchmarkResult {
            name: name.to_string(),
            n,
            m,
            nnz,
            status: res.status,
            iterations: res.info.iters,
            obj_val: res.obj_val,
            mu: res.info.mu,
            solve_time_ms: elapsed.as_secs_f64() * 1000.0,
            error: None,
        },
        Err(e) => BenchmarkResult {
            name: name.to_string(),
            n,
            m,
            nnz,
            status: SolveStatus::NumericalError,
            iterations: 0,
            obj_val: 0.0,
            mu: 0.0,
            solve_time_ms: elapsed.as_secs_f64() * 1000.0,
            error: Some(format!("Solve error: {}", e)),
        },
    }
}

/// Run the standard CBLIB SOCP suite.
pub fn run_full_suite(settings: &SolverSettings, limit: Option<usize>) -> Vec<BenchmarkResult> {
    run_problem_list(CBLIB_SOCP_PROBLEMS, settings, limit)
}

/// Run the large CBLIB SOCP suite.
pub fn run_large_suite(settings: &SolverSettings, limit: Option<usize>) -> Vec<BenchmarkResult> {
    run_problem_list(CBLIB_SOCP_LARGE, settings, limit)
}

/// Run the Mittelmann "Large SOCP Benchmark" curated subset.
pub fn run_mittelmann_suite(
    settings: &SolverSettings,
    limit: Option<usize>,
) -> Vec<BenchmarkResult> {
    run_problem_list(CBLIB_MITTELMANN, settings, limit)
}

/// Run all CBLIB SOCP problems (standard + large).
pub fn run_all_suites(settings: &SolverSettings, limit: Option<usize>) -> Vec<BenchmarkResult> {
    let all_problems: Vec<&str> = CBLIB_SOCP_PROBLEMS
        .iter()
        .chain(CBLIB_SOCP_LARGE.iter())
        .copied()
        .collect();
    run_problem_list(&all_problems, settings, limit)
}

/// Run complete CBLIB (standard + large + mittelmann).
pub fn run_complete_suite(settings: &SolverSettings, limit: Option<usize>) -> Vec<BenchmarkResult> {
    let all_problems: Vec<&str> = CBLIB_SOCP_PROBLEMS
        .iter()
        .chain(CBLIB_SOCP_LARGE.iter())
        .chain(CBLIB_MITTELMANN.iter())
        .copied()
        .collect();
    run_problem_list(&all_problems, settings, limit)
}

/// Run a list of CBLIB problems.
fn run_problem_list(
    problems: &[&str],
    settings: &SolverSettings,
    limit: Option<usize>,
) -> Vec<BenchmarkResult> {
    let problems: Vec<_> = if let Some(limit) = limit {
        problems.iter().take(limit).collect()
    } else {
        problems.iter().collect()
    };

    let mut results = Vec::with_capacity(problems.len());

    for (i, name) in problems.iter().enumerate() {
        eprint!("[{}/{}] {}... ", i + 1, problems.len(), name);

        let result = run_single(name, settings);

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
        optimal: 0,
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

        match r.status {
            SolveStatus::Optimal => summary.optimal += 1,
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
    println!("\n{:-<90}", "");
    println!(
        "{:<30} {:>8} {:>8} {:>10} {:>8} {:>12} {:>10}",
        "Problem", "n", "m", "Status", "Iters", "Objective", "Time(ms)"
    );
    println!("{:-<90}", "");

    for r in results {
        let status_str = if r.error.is_some() {
            "ERROR".to_string()
        } else {
            format!("{:?}", r.status)
        };

        println!(
            "{:<30} {:>8} {:>8} {:>10} {:>8} {:>12.4e} {:>10.1}",
            r.name, r.n, r.m, status_str, r.iterations, r.obj_val, r.solve_time_ms
        );
    }

    println!("{:-<90}", "");
}

/// Print summary.
pub fn print_summary(summary: &BenchmarkSummary) {
    println!("\nCBLIB Benchmark Summary");
    println!("=======================");
    println!("Total problems:     {}", summary.total);
    println!("Parse errors:       {}", summary.parse_errors);
    println!();
    println!(
        "Optimal:            {} ({:.1}%)",
        summary.optimal,
        100.0 * summary.optimal as f64 / (summary.total - summary.parse_errors).max(1) as f64
    );
    println!("Max iterations:     {}", summary.max_iters);
    println!("Numerical error:    {}", summary.numerical_error);
    println!("Other:              {}", summary.other);
    println!();
    println!("Avg iterations:     {:.1}", summary.avg_iters);
    println!("Avg solve time:     {:.1} ms", summary.avg_time_ms);
}
