//! QPLIB (Quadratic Programming Library) benchmark infrastructure.
//!
//! Downloads and runs QP benchmarks from https://qplib.zib.de/
//! Uses CPLEX LP format for parsing.

use crate::qps::QpsProblem;
use anyhow::{bail, Context, Result};
use solver_core::{solve, SolveStatus, SolverSettings};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

const QPLIB_BASE_URL: &str = "https://qplib.zib.de";

/// QP problems from QPLIB (continuous, convex quadratic).
/// These are QP problems without integer variables and with convex objectives.
pub const QPLIB_PROBLEMS: &[&str] = &[
    // Small convex QP problems
    "QPLIB_0018",
    "QPLIB_0031",
    "QPLIB_0076",
    "QPLIB_0118",
    "QPLIB_0254",
    // Medium convex QP problems
    "QPLIB_0396",
    "QPLIB_0586",
    "QPLIB_0711",
    // Larger problems (may require more iterations)
    "QPLIB_0976",
    "QPLIB_1493",
];

/// Result of running a single QPLIB benchmark.
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

/// Get the cache directory for QPLIB problems.
fn get_cache_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache/minix-bench/qplib")
}

/// Download an LP file from QPLIB.
fn download_lp(name: &str) -> Result<PathBuf> {
    let cache_dir = get_cache_dir();
    fs::create_dir_all(&cache_dir)?;

    let lp_path = cache_dir.join(format!("{}.lp", name));

    // Check if already cached
    if lp_path.exists() {
        return Ok(lp_path);
    }

    // Download LP file (format: https://qplib.zib.de/lp/QPLIB_XXXX.lp)
    let url = format!("{}/lp/{}.lp", QPLIB_BASE_URL, name);

    eprintln!("Downloading {}...", url);

    let output = Command::new("curl")
        .args(["-sL", "--max-time", "60", "-o"])
        .arg(&lp_path)
        .arg(&url)
        .output()
        .context("Failed to run curl")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("curl failed: {}", stderr);
    }

    // Check if file was downloaded
    if !lp_path.exists() || fs::metadata(&lp_path)?.len() == 0 {
        bail!("Download failed: empty or missing file");
    }

    Ok(lp_path)
}

/// Parse CPLEX LP format file.
///
/// LP format is a human-readable format with sections:
/// - Minimize/Maximize: objective function
/// - Subject To: linear constraints
/// - Bounds: variable bounds
/// - End
pub fn parse_lp<P: AsRef<std::path::Path>>(path: P) -> Result<QpsProblem> {
    let file = File::open(path.as_ref())
        .with_context(|| format!("Failed to open LP file: {:?}", path.as_ref()))?;
    let reader = BufReader::new(file);

    let name = path
        .as_ref()
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let mut obj_sense = 1.0; // minimize by default
    let mut var_map: HashMap<String, usize> = HashMap::new();
    let mut con_map: HashMap<String, usize> = HashMap::new();
    let mut n = 0;
    let mut m = 0;

    let mut q_coeffs: HashMap<usize, f64> = HashMap::new();
    let mut p_triplets: Vec<(usize, usize, f64)> = Vec::new();
    let mut a_triplets: Vec<(usize, usize, f64)> = Vec::new();
    let mut con_lower: Vec<f64> = Vec::new();
    let mut con_upper: Vec<f64> = Vec::new();

    #[derive(Debug, Clone, Copy, PartialEq)]
    #[allow(dead_code)]
    enum Section {
        None,
        Objective,
        Constraints,
        Bounds,
    }

    let mut section = Section::None;

    // Helper to get or create variable index
    let mut get_var = |name: &str| -> usize {
        if let Some(&idx) = var_map.get(name) {
            idx
        } else {
            let idx = n;
            var_map.insert(name.to_string(), idx);
            n += 1;
            idx
        }
    };

    for line_result in reader.lines() {
        let line = line_result?;
        let trimmed = line.trim();

        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with('\\') {
            continue;
        }

        // Check for section headers
        let upper = trimmed.to_uppercase();
        if upper.starts_with("MINIMIZE") || upper.starts_with("MINIMUM") || upper.starts_with("MIN")
        {
            section = Section::Objective;
            obj_sense = 1.0;
            continue;
        } else if upper.starts_with("MAXIMIZE")
            || upper.starts_with("MAXIMUM")
            || upper.starts_with("MAX")
        {
            section = Section::Objective;
            obj_sense = -1.0;
            continue;
        } else if upper.starts_with("SUBJECT TO")
            || upper.starts_with("ST")
            || upper.starts_with("S.T.")
        {
            section = Section::Constraints;
            continue;
        } else if upper.starts_with("BOUNDS") {
            section = Section::Bounds;
            continue;
        } else if upper.starts_with("END") {
            break;
        } else if upper.starts_with("GENERAL")
            || upper.starts_with("BINARY")
            || upper.starts_with("INTEGER")
            || upper.starts_with("SEMI")
        {
            // Skip integer/binary sections - we only handle continuous
            section = Section::None;
            continue;
        }

        match section {
            Section::Objective => {
                // Parse objective terms like "x1 + 2 x2 - 3 x3" or "[ x1^2 + 2 x1*x2 ]/2"
                // This is simplified - full LP parsing would need a proper lexer
                parse_objective_terms(trimmed, &mut get_var, &mut q_coeffs, &mut p_triplets)?;
            }
            Section::Constraints => {
                // Parse constraint like "c1: x1 + 2 x2 <= 10"
                if let Some((con_name, expr, bound)) = parse_constraint_line(trimmed) {
                    let con_idx = m;
                    con_map.insert(con_name.clone(), con_idx);
                    m += 1;

                    // Parse expression coefficients
                    for (var_name, coef) in parse_linear_expr(&expr) {
                        let var_idx = get_var(&var_name);
                        a_triplets.push((con_idx, var_idx, coef));
                    }

                    // Set bounds based on constraint type
                    match bound {
                        ConstraintBound::Le(val) => {
                            con_lower.push(f64::NEG_INFINITY);
                            con_upper.push(val);
                        }
                        ConstraintBound::Ge(val) => {
                            con_lower.push(val);
                            con_upper.push(f64::INFINITY);
                        }
                        ConstraintBound::Eq(val) => {
                            con_lower.push(val);
                            con_upper.push(val);
                        }
                    }
                }
            }
            Section::Bounds => {
                // Parse bound like "0 <= x1 <= 10" or "x2 free"
                // Will be processed after we know all variables
            }
            _ => {}
        }
    }

    // Initialize variable bounds (defaults to [0, +inf))
    let var_lower = vec![0.0; n];
    let var_upper = vec![f64::INFINITY; n];

    // Build linear cost vector
    let mut q = vec![0.0; n];
    for (idx, coef) in q_coeffs {
        if idx < n {
            q[idx] = coef;
        }
    }

    Ok(QpsProblem {
        name,
        n,
        m,
        obj_sense,
        q,
        p_triplets,
        a_triplets,
        con_lower,
        con_upper,
        var_lower,
        var_upper,
        var_names: var_map
            .iter()
            .map(|(name, &idx)| (idx, name.clone()))
            .collect::<HashMap<_, _>>()
            .into_iter()
            .collect::<Vec<_>>()
            .into_iter()
            .map(|(_, name)| name)
            .collect(),
        con_names: con_map
            .iter()
            .map(|(name, &idx)| (idx, name.clone()))
            .collect::<HashMap<_, _>>()
            .into_iter()
            .collect::<Vec<_>>()
            .into_iter()
            .map(|(_, name)| name)
            .collect(),
    })
}

#[derive(Debug)]
enum ConstraintBound {
    Le(f64),
    Ge(f64),
    Eq(f64),
}

/// Parse objective terms (simplified - handles basic linear terms).
fn parse_objective_terms(
    line: &str,
    get_var: &mut impl FnMut(&str) -> usize,
    q_coeffs: &mut HashMap<usize, f64>,
    _p_triplets: &mut Vec<(usize, usize, f64)>,
) -> Result<()> {
    // Skip objective name if present (e.g., "obj:")
    let expr = if let Some(colon_pos) = line.find(':') {
        &line[colon_pos + 1..]
    } else {
        line
    };

    // Parse linear terms
    for (var_name, coef) in parse_linear_expr(expr) {
        let var_idx = get_var(&var_name);
        *q_coeffs.entry(var_idx).or_insert(0.0) += coef;
    }

    Ok(())
}

/// Parse a constraint line, returning (name, expression, bound).
fn parse_constraint_line(line: &str) -> Option<(String, String, ConstraintBound)> {
    // Try to find constraint name
    let (name, rest) = if let Some(colon_pos) = line.find(':') {
        let name = line[..colon_pos].trim().to_string();
        let rest = line[colon_pos + 1..].trim();
        (name, rest.to_string())
    } else {
        (format!("c{}", line.len()), line.to_string())
    };

    // Find comparison operator
    if let Some(pos) = rest.find("<=") {
        let expr = rest[..pos].trim().to_string();
        let val: f64 = rest[pos + 2..].trim().parse().ok()?;
        Some((name, expr, ConstraintBound::Le(val)))
    } else if let Some(pos) = rest.find(">=") {
        let expr = rest[..pos].trim().to_string();
        let val: f64 = rest[pos + 2..].trim().parse().ok()?;
        Some((name, expr, ConstraintBound::Ge(val)))
    } else if let Some(pos) = rest.find('=') {
        let expr = rest[..pos].trim().to_string();
        let val: f64 = rest[pos + 1..].trim().parse().ok()?;
        Some((name, expr, ConstraintBound::Eq(val)))
    } else {
        None
    }
}

/// Parse a linear expression into (variable_name, coefficient) pairs.
fn parse_linear_expr(expr: &str) -> Vec<(String, f64)> {
    let mut result = Vec::new();
    let mut current_coef: Option<f64> = None;
    let mut sign = 1.0;

    // Tokenize: split by spaces but keep +/-
    let tokens: Vec<&str> = expr.split_whitespace().collect();

    for token in tokens {
        if token == "+" {
            sign = 1.0;
        } else if token == "-" {
            sign = -1.0;
        } else if let Ok(val) = token.parse::<f64>() {
            current_coef = Some(sign * val);
            sign = 1.0;
        } else if token.starts_with('+') || token.starts_with('-') {
            // Handle "+2" or "-3"
            if let Ok(val) = token.parse::<f64>() {
                current_coef = Some(val);
            }
        } else {
            // Variable name
            let coef = current_coef.unwrap_or(sign);
            result.push((token.to_string(), coef));
            current_coef = None;
            sign = 1.0;
        }
    }

    result
}

/// Load a QPLIB problem.
pub fn load_problem(name: &str) -> Result<QpsProblem> {
    let path = download_lp(name)?;
    parse_lp(&path)
}

/// Run a single QPLIB benchmark.
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

/// Run the full QPLIB suite.
pub fn run_full_suite(settings: &SolverSettings, limit: Option<usize>) -> Vec<BenchmarkResult> {
    let problems: Vec<_> = if let Some(limit) = limit {
        QPLIB_PROBLEMS.iter().take(limit).collect()
    } else {
        QPLIB_PROBLEMS.iter().collect()
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
    println!("\nQPLIB Benchmark Summary");
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
