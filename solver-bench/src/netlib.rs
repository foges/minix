//! NETLIB LP benchmark suite runner.
//!
//! Downloads and runs the classic NETLIB LP test set.
//! Reference: https://www.netlib.org/lp/

use std::fs::{self, File};
use std::io::Read;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use flate2::read::GzDecoder;
use solver_core::{solve, SolveStatus, SolverSettings};

use crate::qps::{parse_qps, QpsProblem};

/// Primary URL for NETLIB LP files (from HiGHS solver's test instances)
const NETLIB_HIGHS_URL: &str =
    "https://raw.githubusercontent.com/ERGO-Code/HiGHS/master/check/instances";

/// Secondary URL: COIN-OR's MPS files
const NETLIB_COINOR_URL: &str =
    "https://raw.githubusercontent.com/coin-or-tools/Data-Netlib/master";

/// Curated subset of NETLIB problems for quick testing.
/// These are representative problems that solve quickly.
pub const NETLIB_CLASSIC: &[&str] = &[
    // Small problems (< 100 constraints)
    "afiro",    // 27 x 32, classic tiny
    "sc50a",    // 50 x 48
    "sc50b",    // 50 x 48
    "adlittle", // 56 x 97
    "blend",    // 74 x 83
    "sc105",    // 105 x 103
    // Medium problems (100-500 constraints)
    "israel",   // 174 x 142
    "sc205",    // 205 x 203
    "share1b",  // 117 x 225
    "share2b",  // 96 x 79
    "lotfi",    // 153 x 308
    "e226",     // 223 x 282
    "bandm",    // 305 x 472
    "scorpion", // 388 x 358
    // Larger problems (500+ constraints)
    "etamacro", // 400 x 688
    "25fv47",   // 821 x 1571
    "greenbea", // 2392 x 5405
];

/// Full NETLIB feasible LP set (108 problems from ClarabelBenchmarks).
/// Source: ClarabelBenchmarks/src/problem_sets/netlib/feasibleLP/
pub const NETLIB_FULL: &[&str] = &[
    "25fv47", "80bau3b", "adlittle", "afiro", "agg", "agg2", "agg3", "bandm", "beaconfd", "blend",
    "bnl1", "bnl2", "bore3d", "brandy", "capri", "cre_a", "cre_c", "cycle", "czprob", "d2q06c",
    "d6cube", "degen2", "degen3", "dfl001", "e226", "etamacro", "fffff800", "finnis", "fit1d",
    "fit1p", "fit2d", "fit2p", "ganges", "gfrd_pnc", "greenbea", "greenbeb", "grow15", "grow22",
    "grow7", "israel", "kb2", "ken_07", "ken_11", "ken_13", "lotfi", "maros", "maros_r7",
    "modszk1", "nug05", "nug06", "nug07", "nug08", "nug12", "nug15", "osa_07", "pds_02", "pds_06",
    "pds_10", "perold", "pilot", "pilot_ja", "pilot_we", "pilot4", "pilot87", "pilotnov", "qap12",
    "qap15", "qap8", "recipe", "sc105", "sc205", "sc50a", "sc50b", "scagr25", "scagr7", "scfxm1",
    "scfxm2", "scfxm3", "scorpion", "scrs8", "scsd1", "scsd6", "scsd8", "sctap1", "sctap2",
    "sctap3", "share1b", "share2b", "shell", "ship04l", "ship04s", "ship08l", "ship08s", "ship12l",
    "ship12s", "sierra", "stair", "standata", "standgub", "standmps", "stocfor1", "stocfor2",
    "stocfor3", "truss", "tuff", "vtp_base", "wood1p", "woodw",
];

/// Default problem set (classic subset for quick testing).
pub const NETLIB_PROBLEMS: &[&str] = NETLIB_CLASSIC;

/// Result of running a single NETLIB benchmark.
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

/// Get the cache directory for NETLIB problems.
fn get_cache_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache/minix-bench/netlib")
}

/// Try to download an MPS file from a URL.
fn try_download_mps(url: &str, mps_path: &PathBuf) -> Result<bool> {
    let output = Command::new("curl")
        .args(["-sL", "--fail", "--max-time", "60", "-o"])
        .arg(mps_path)
        .arg(url)
        .output()
        .context("Failed to run curl")?;

    if !output.status.success() {
        let _ = fs::remove_file(mps_path);
        return Ok(false);
    }

    // Check if file was downloaded and is valid
    if !mps_path.exists() {
        return Ok(false);
    }

    let metadata = fs::metadata(mps_path)?;
    if metadata.len() == 0 {
        let _ = fs::remove_file(mps_path);
        return Ok(false);
    }

    // Check if we got an HTML error page instead of MPS file
    // MPS files must start with "NAME" or a comment (spaces + NAME)
    let mut content = String::new();
    File::open(mps_path)?.read_to_string(&mut content)?;
    let trimmed = content.trim_start();
    if !trimmed.starts_with("NAME") && !trimmed.starts_with("*") {
        let _ = fs::remove_file(mps_path);
        return Ok(false);
    }

    Ok(true)
}

/// Try to download and decompress a .mps.gz file.
fn try_download_mps_gz(url: &str, mps_path: &PathBuf) -> Result<bool> {
    let gz_path = mps_path.with_extension("mps.gz");

    let output = Command::new("curl")
        .args(["-sL", "--fail", "--max-time", "60", "-o"])
        .arg(&gz_path)
        .arg(url)
        .output()
        .context("Failed to run curl")?;

    if !output.status.success() || !gz_path.exists() {
        let _ = fs::remove_file(&gz_path);
        return Ok(false);
    }

    // Decompress
    let gz_file = File::open(&gz_path)?;
    let mut decoder = GzDecoder::new(gz_file);
    let mut content = String::new();
    if decoder.read_to_string(&mut content).is_err() {
        let _ = fs::remove_file(&gz_path);
        return Ok(false);
    }

    // Validate MPS format
    let trimmed = content.trim_start();
    if !trimmed.starts_with("NAME") && !trimmed.starts_with("*") {
        let _ = fs::remove_file(&gz_path);
        return Ok(false);
    }

    // Write decompressed content
    fs::write(mps_path, &content)?;
    let _ = fs::remove_file(&gz_path);

    Ok(true)
}

/// Download an MPS file from NETLIB (tries multiple sources).
fn download_mps(name: &str) -> Result<PathBuf> {
    let cache_dir = get_cache_dir();
    fs::create_dir_all(&cache_dir)?;

    let mps_path = cache_dir.join(format!("{}.mps", name));

    // Check if already cached
    if mps_path.exists() {
        return Ok(mps_path);
    }

    // Try HiGHS repository first (uncompressed)
    let url = format!("{}/{}.mps", NETLIB_HIGHS_URL, name);
    eprintln!("Downloading {}...", url);
    if try_download_mps(&url, &mps_path)? {
        return Ok(mps_path);
    }

    // Try COIN-OR repository (compressed .mps.gz)
    let url = format!("{}/{}.mps.gz", NETLIB_COINOR_URL, name);
    eprintln!("Trying {}...", url);
    if try_download_mps_gz(&url, &mps_path)? {
        return Ok(mps_path);
    }

    bail!(
        "Download failed: could not find {} at HiGHS or COIN-OR repositories",
        name
    )
}

/// Load a NETLIB problem.
pub fn load_problem(name: &str) -> Result<QpsProblem> {
    let path = download_mps(name)?;
    parse_qps(&path)
}

/// Run a single NETLIB benchmark.
pub fn run_single(name: &str, settings: &SolverSettings) -> BenchmarkResult {
    // Load problem
    let qps = match load_problem(name) {
        Ok(qps) => qps,
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

/// Run the classic NETLIB LP suite (curated subset).
pub fn run_full_suite(settings: &SolverSettings, limit: Option<usize>) -> Vec<BenchmarkResult> {
    run_problem_list(NETLIB_CLASSIC, settings, limit)
}

/// Run the extended NETLIB LP suite (all 108 problems).
pub fn run_extended_suite(settings: &SolverSettings, limit: Option<usize>) -> Vec<BenchmarkResult> {
    run_problem_list(NETLIB_FULL, settings, limit)
}

/// Run a list of problems.
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
        "{:<20} {:>8} {:>8} {:>10} {:>8} {:>12} {:>10}",
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
            "{:<20} {:>8} {:>8} {:>10} {:>8} {:>12.4e} {:>10.1}",
            r.name, r.n, r.m, status_str, r.iterations, r.obj_val, r.solve_time_ms
        );
    }

    println!("{:-<90}", "");
}

/// Print summary.
pub fn print_summary(summary: &BenchmarkSummary) {
    println!("\nNETLIB LP Benchmark Summary");
    println!("===========================");
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
