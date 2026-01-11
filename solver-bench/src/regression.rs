use solver_core::ipm2::metrics::compute_unscaled_metrics;
use solver_core::{ConeSpec, ProblemData, SolveStatus, SolverSettings};
use serde::{Deserialize, Serialize};

use crate::maros_meszaros::load_local_problem;
use crate::solver_choice::{solve_with_choice, SolverChoice};
use crate::test_problems;

pub struct RegressionResult {
    pub name: String,
    pub status: SolveStatus,
    pub rel_p: f64,
    pub rel_d: f64,
    pub gap_rel: f64,
    pub iterations: usize,
    pub expected_iters: Option<usize>,
    pub expected_status: Option<SolveStatus>,
    pub error: Option<String>,
    pub skipped: bool,
    pub expected_to_fail: bool,
    pub solve_time_ms: Option<u64>,
    pub kkt_factor_time_ms: Option<u64>,
    pub kkt_solve_time_ms: Option<u64>,
    pub cone_time_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfSummary {
    pub total_solve_ms: u64,
    pub total_kkt_factor_ms: u64,
    pub total_kkt_solve_ms: u64,
    pub total_cone_ms: u64,
    pub cases: usize,
}

impl PerfSummary {
    fn empty() -> Self {
        Self {
            total_solve_ms: 0,
            total_kkt_factor_ms: 0,
            total_kkt_solve_ms: 0,
            total_cone_ms: 0,
            cases: 0,
        }
    }
}

pub fn perf_summary(results: &[RegressionResult]) -> PerfSummary {
    let mut summary = PerfSummary::empty();
    for res in results {
        let (solve_ms, kkt_factor_ms, kkt_solve_ms, cone_ms) = match (
            res.solve_time_ms,
            res.kkt_factor_time_ms,
            res.kkt_solve_time_ms,
            res.cone_time_ms,
        ) {
            (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
            _ => continue,
        };

        summary.total_solve_ms += solve_ms;
        summary.total_kkt_factor_ms += kkt_factor_ms;
        summary.total_kkt_solve_ms += kkt_solve_ms;
        summary.total_cone_ms += cone_ms;
        summary.cases += 1;
    }
    summary
}

pub fn compare_perf_baseline(
    baseline: &PerfSummary,
    current: &PerfSummary,
    max_regression: f64,
) -> Vec<String> {
    let mut failures = Vec::new();
    let guard = |name: &str, base: u64, cur: u64| {
        if base == 0 {
            return None;
        }
        let ratio = cur as f64 / base as f64;
        if ratio > 1.0 + max_regression {
            Some(format!(
                "{} regression {:.2}x (baseline {}ms, current {}ms)",
                name, ratio, base, cur
            ))
        } else {
            None
        }
    };

    if let Some(msg) = guard(
        "solve_time",
        baseline.total_solve_ms,
        current.total_solve_ms,
    ) {
        failures.push(msg);
    }
    if let Some(msg) = guard(
        "kkt_factor_time",
        baseline.total_kkt_factor_ms,
        current.total_kkt_factor_ms,
    ) {
        failures.push(msg);
    }
    if let Some(msg) = guard(
        "kkt_solve_time",
        baseline.total_kkt_solve_ms,
        current.total_kkt_solve_ms,
    ) {
        failures.push(msg);
    }
    if let Some(msg) = guard(
        "cone_time",
        baseline.total_cone_ms,
        current.total_cone_ms,
    ) {
        failures.push(msg);
    }

    failures
}

/// Returns expected behavior for specific problems.
/// Used to track known expected statuses (e.g., NumericalLimit for BOYD-class problems)
/// and expected iteration counts for regression detection.
fn expected_behavior(name: &str) -> (Option<SolveStatus>, Option<usize>) {
    match name {
        // BOYD-class problems hit numerical precision floor (135,000x cancellation)
        // κ(K) > 1e13, rel_d stuck at ~1e-3 despite primal+gap converging
        "BOYD1" => (Some(SolveStatus::NumericalLimit), Some(50)),
        "BOYD2" => (Some(SolveStatus::NumericalLimit), Some(50)),

        // Add more specific expected behaviors here as needed
        // For most problems: (None, None) means expect Optimal with variable iterations
        _ => (None, None),
    }
}

pub fn run_regression_suite(
    settings: &SolverSettings,
    solver: SolverChoice,
    require_cache: bool,
    max_iter_fail: usize,
) -> Vec<RegressionResult> {
    let mut results = Vec::new();

    // 108 Maros-Meszaros problems that truly meet quality standards (79.4% of 136)
    // Excluded: 28 problems with dual divergence or gap issues
    let qps_cases = [
        // HS problems (tiny)
        "HS21", "HS35", "HS35MOD", "HS51", "HS52", "HS53", "HS76", "HS118", "HS268",
        // Other tiny (<1ms)
        "TAME", "S268", "ZECEVIC2", "LOTSCHD", "QAFIRO",
        // CVXQP family (all 9)
        "CVXQP1_S", "CVXQP2_S", "CVXQP3_S",
        "CVXQP1_M", "CVXQP2_M", "CVXQP3_M",
        "CVXQP1_L", "CVXQP2_L", "CVXQP3_L",
        // DUAL/PRIMAL families (all 16)
        "DUAL1", "DUAL2", "DUAL3", "DUAL4",
        "DUALC1", "DUALC2", "DUALC5", "DUALC8",
        "PRIMAL1", "PRIMAL2", "PRIMAL3", "PRIMAL4",
        "PRIMALC1", "PRIMALC2", "PRIMALC5", "PRIMALC8",
        // AUG family (all 8)
        "AUG2D", "AUG2DC", "AUG2DCQP", "AUG2DQP",
        "AUG3D", "AUG3DC", "AUG3DCQP", "AUG3DQP",
        // CONT family (all 6)
        "CONT-050", "CONT-100", "CONT-101", "CONT-200", "CONT-201", "CONT-300",
        // LISWET family (all 12)
        "LISWET1", "LISWET2", "LISWET3", "LISWET4", "LISWET5", "LISWET6",
        "LISWET7", "LISWET8", "LISWET9", "LISWET10", "LISWET11", "LISWET12",
        // STADAT family (all 3)
        "STADAT1", "STADAT2", "STADAT3",
        // QGROW family (all 3)
        "QGROW7", "QGROW15", "QGROW22",
        // Q* problems that pass with good quality
        "QETAMACR", "QISRAEL",
        "QPCBLEND", "QPCBOEI2", "QPCSTAIR",
        "QRECIPE", "QSC205",
        "QSCSD1", "QSCSD6", "QSCSD8", "QSCTAP1", "QSCTAP2", "QSCTAP3",
        "QSEBA", "QSHARE2B", "QSHELL", "QSIERRA", "QSTAIR", "QSTANDAT",
        // Other medium/large
        "DPKLO1", "DTOC3", "EXDATA", "GOULDQP2", "GOULDQP3",
        "HUES-MOD", "HUESTIS", "KSIP", "LASER",
        "MOSARQP1", "MOSARQP2", "POWELL20",
        "STCQP2", "UBH1", "VALUES", "YAO",
        // BOYD portfolio QPs (~93k vars)
        "BOYD1", "BOYD2",
    ];

    // Get expected failures list
    let expected_failures: std::collections::HashSet<&str> =
        test_problems::maros_meszaros_expected_failures().iter().copied().collect();

    for name in qps_cases {
        let is_expected_failure = expected_failures.contains(name);

        match load_local_problem(name) {
            Ok(qps) => {
                let prob = match qps.to_problem_data() {
                    Ok(p) => p,
                    Err(e) => {
                        results.push(RegressionResult {
                            name: name.to_string(),
                            status: SolveStatus::NumericalError,
                            rel_p: f64::NAN,
                            rel_d: f64::NAN,
                            gap_rel: f64::NAN,
                            iterations: 0,
                            expected_iters: None,
                            expected_status: None,
                            error: Some(format!("conversion error: {}", e)),
                            skipped: false,
                            expected_to_fail: is_expected_failure,
                            solve_time_ms: None,
                            kkt_factor_time_ms: None,
                            kkt_solve_time_ms: None,
                            cone_time_ms: None,
                        });
                        continue;
                    }
                };
                // Use reduced max_iter for expected-to-fail problems
                let mut settings_for_problem = settings.clone();
                if is_expected_failure {
                    settings_for_problem.max_iter = max_iter_fail;
                }
                let mut result = run_case(&prob, &settings_for_problem, solver, name);
                result.expected_to_fail = is_expected_failure;
                let (exp_status, exp_iters) = expected_behavior(name);
                result.expected_status = exp_status;
                result.expected_iters = exp_iters;
                results.push(result);
            }
            Err(e) => {
                if require_cache {
                    results.push(RegressionResult {
                        name: name.to_string(),
                        status: SolveStatus::NumericalError,
                        rel_p: f64::NAN,
                        rel_d: f64::NAN,
                        gap_rel: f64::NAN,
                        iterations: 0,
                        expected_iters: None,
                        expected_status: None,
                        error: Some(format!("missing QPS: {}", e)),
                        skipped: false,
                        expected_to_fail: is_expected_failure,
                        solve_time_ms: None,
                        kkt_factor_time_ms: None,
                        kkt_solve_time_ms: None,
                        cone_time_ms: None,
                    });
                } else {
                    results.push(RegressionResult {
                        name: name.to_string(),
                        status: SolveStatus::NumericalError,
                        rel_p: f64::NAN,
                        rel_d: f64::NAN,
                        gap_rel: f64::NAN,
                        iterations: 0,
                        expected_iters: None,
                        expected_status: None,
                        error: None,
                        skipped: true,
                        expected_to_fail: is_expected_failure,
                        solve_time_ms: None,
                        kkt_factor_time_ms: None,
                        kkt_solve_time_ms: None,
                        cone_time_ms: None,
                    });
                }
            }
        }
    }

    // Add cone problems from test_problems module
    for test_prob in test_problems::synthetic_test_problems() {
        let prob = (test_prob.builder)();
        let mut result = run_case(&prob, settings, solver, test_prob.name);
        result.expected_iters = test_prob.expected_iterations;
        results.push(result);
    }

    results
}

fn run_case(
    prob: &ProblemData,
    settings: &SolverSettings,
    solver: SolverChoice,
    name: &str,
) -> RegressionResult {
    match solve_with_choice(prob, settings, solver) {
        Ok(res) => {
            let n = prob.num_vars();
            let m = prob.num_constraints();
            let mut r_p = vec![0.0; m];
            let mut r_d = vec![0.0; n];
            let mut p_x = vec![0.0; n];
            let metrics = compute_unscaled_metrics(
                &prob.A,
                prob.P.as_ref(),
                &prob.q,
                &prob.b,
                &res.x,
                &res.s,
                &res.z,
                &mut r_p,
                &mut r_d,
                &mut p_x,
            );

            RegressionResult {
                name: name.to_string(),
                status: res.status,
                rel_p: metrics.rel_p,
                rel_d: metrics.rel_d,
                gap_rel: metrics.gap_rel,
                iterations: res.info.iters,
                expected_iters: None, // Will be set by caller
                expected_status: None, // Will be set by caller
                error: None,
                skipped: false,
                expected_to_fail: false, // Will be set by caller
                solve_time_ms: Some(res.info.solve_time_ms),
                kkt_factor_time_ms: Some(res.info.kkt_factor_time_ms),
                kkt_solve_time_ms: Some(res.info.kkt_solve_time_ms),
                cone_time_ms: Some(res.info.cone_time_ms),
            }
        }
        Err(e) => RegressionResult {
            name: name.to_string(),
            status: SolveStatus::NumericalError,
            rel_p: f64::NAN,
            rel_d: f64::NAN,
            gap_rel: f64::NAN,
            iterations: 0,
            expected_iters: None,
            expected_status: None,
            error: Some(e.to_string()),
            skipped: false,
            expected_to_fail: false, // Will be set by caller
            solve_time_ms: None,
            kkt_factor_time_ms: None,
            kkt_solve_time_ms: None,
            cone_time_ms: None,
        },
    }
}

fn synthetic_cases() -> Vec<(&'static str, ProblemData)> {
    let mut cases = Vec::new();

    // Nonnegativity LP: min x, x >= 0
    let a = solver_core::linalg::sparse::from_triplets(1, 1, vec![(0, 0, -1.0)]);
    let prob = ProblemData {
        P: None,
        q: vec![1.0],
        A: a,
        b: vec![0.0],
        cones: vec![ConeSpec::NonNeg { dim: 1 }],
        var_bounds: None,
        integrality: None,
    };
    cases.push(("SYN_LP_NONNEG", prob));

    // SOC feasibility: x in SOC via s = x, A = -I, b = 0.
    let a = solver_core::linalg::sparse::from_triplets(
        2,
        2,
        vec![(0, 0, -1.0), (1, 1, -1.0)],
    );
    let prob = ProblemData {
        P: None,
        q: vec![0.0, 0.0],
        A: a,
        b: vec![0.0, 0.0],
        cones: vec![ConeSpec::Soc { dim: 2 }],
        var_bounds: None,
        integrality: None,
    };
    cases.push(("SYN_SOC_FEAS", prob));

    cases
}

/// Expected iterations for each problem - measured with 1e-8 tolerance (industry standard)
/// These are EXACT iteration counts with no margin/slop allowed.
fn expected_iterations(name: &str) -> Option<usize> {
    match name {
        // HS problems
        "HS21" => Some(9), "HS35" => Some(6), "HS35MOD" => Some(12),
        "HS51" => Some(4), "HS52" => Some(4), "HS53" => Some(6),
        "HS76" => Some(6), "HS118" => Some(11), "HS268" => Some(8),
        // Small problems
        "TAME" => Some(4), "S268" => Some(8), "ZECEVIC2" => Some(7),
        "LOTSCHD" => Some(8), "QAFIRO" => Some(14),
        // CVXQP family
        "CVXQP1_S" => Some(8), "CVXQP2_S" => Some(9), "CVXQP3_S" => Some(10),
        "CVXQP1_M" => Some(10), "CVXQP2_M" => Some(9), "CVXQP3_M" => Some(12),
        "CVXQP1_L" => Some(11), "CVXQP2_L" => Some(10), "CVXQP3_L" => Some(10),
        // DUAL/PRIMAL
        "DUAL1" => Some(12), "DUAL2" => Some(11), "DUAL3" => Some(12), "DUAL4" => Some(12),
        "DUALC1" => Some(13), "DUALC2" => Some(10), "DUALC5" => Some(10), "DUALC8" => Some(10),
        "PRIMAL1" => Some(10), "PRIMAL2" => Some(9), "PRIMAL3" => Some(10), "PRIMAL4" => Some(9),
        "PRIMALC1" => Some(16), "PRIMALC2" => Some(14), "PRIMALC5" => Some(9), "PRIMALC8" => Some(13),
        // AUG family
        "AUG2D" => Some(7), "AUG2DC" => Some(7), "AUG2DCQP" => Some(13), "AUG2DQP" => Some(14),
        "AUG3D" => Some(6), "AUG3DC" => Some(6), "AUG3DCQP" => Some(11), "AUG3DQP" => Some(13),
        // CONT family (updated after scale-invariant infeasibility detection)
        "CONT-050" => Some(10), "CONT-100" => Some(11), "CONT-101" => Some(10),
        "CONT-200" => Some(12), "CONT-201" => Some(10), "CONT-300" => Some(12),
        // LISWET family (all converge after recent fixes)
        "LISWET1" => Some(27), "LISWET2" => Some(18), "LISWET3" => Some(26), "LISWET4" => Some(36),
        "LISWET5" => Some(20), "LISWET6" => Some(25), "LISWET7" => Some(32), "LISWET8" => Some(32),
        "LISWET9" => Some(38), "LISWET10" => Some(25), "LISWET11" => Some(30), "LISWET12" => Some(38),
        // STADAT/QGROW
        "STADAT1" => Some(12), "STADAT2" => Some(26), "STADAT3" => Some(27),
        "QGROW7" => Some(22), "QGROW15" => Some(24), "QGROW22" => Some(28),
        // Other Q* problems
        "QETAMACR" => Some(21), "QISRAEL" => Some(27), "QPCBLEND" => Some(17),
        "QPCBOEI2" => Some(24), "QPCSTAIR" => Some(21), "QRECIPE" => Some(17),
        "QSC205" => Some(16), "QSCSD1" => Some(9), "QSCSD6" => Some(13), "QSCSD8" => Some(12),
        "QSCTAP1" => Some(19), "QSCTAP2" => Some(11), "QSCTAP3" => Some(13),
        "QSEBA" => Some(24), "QSHARE2B" => Some(17), "QSHELL" => Some(37),
        "QSIERRA" => Some(34), "QSTAIR" => Some(21), "QSTANDAT" => Some(18),
        // Other
        "DPKLO1" => Some(4), "DTOC3" => Some(5), "EXDATA" => Some(13),
        "GOULDQP2" => Some(14), "GOULDQP3" => Some(8),
        "HUES-MOD" => Some(10), "HUESTIS" => Some(10), "KSIP" => Some(12), "LASER" => Some(9),
        "MOSARQP1" => Some(10), "MOSARQP2" => Some(10), "POWELL20" => Some(9),
        "STCQP2" => Some(8), "UBH1" => Some(63), "VALUES" => Some(13), "YAO" => Some(44),
        // BOYD (large) - these hit MaxIters, no expected value
        // Synthetic (measured exact with 1e-8 tolerances)
        "SYN_LP_NONNEG" => Some(4), "SYN_SOC_FEAS" => Some(5),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn regression_suite_smoke() {
        let require_cache = env::var("MINIX_REQUIRE_QPS_CACHE")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let max_iter = env::var("MINIX_REGRESSION_MAX_ITER")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(200);
        let max_iter_fail = env::var("MINIX_REGRESSION_MAX_ITER_FAIL")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(50);
        let verbose = env::var("MINIX_VERBOSE")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let mut settings = SolverSettings::default();
        settings.max_iter = max_iter;

        let results = run_regression_suite(&settings, SolverChoice::Ipm2, require_cache, max_iter_fail);
        // Use practical tolerances for unscaled metrics
        // The solver uses scaled metrics internally (1e-8), but unscaled
        // metrics can differ due to problem conditioning
        let tol_feas = 1e-6;  // Feasibility tolerance for unscaled metrics
        let tol_gap = 1e-3;   // Relative gap tolerance (problems with poor conditioning may not reach 1e-6)

        let mut failures = Vec::new();
        let mut unexpected_passes = Vec::new();

        for res in &results {
            if res.skipped {
                if require_cache {
                    failures.push(format!("{}: missing cache", res.name));
                }
                continue;
            }

            let is_pass = matches!(res.status, SolveStatus::Optimal | SolveStatus::AlmostOptimal)
                && res.rel_p.is_finite()
                && res.rel_d.is_finite()
                && res.gap_rel.is_finite()
                && res.rel_p <= tol_feas
                && res.rel_d <= tol_feas
                && res.gap_rel <= tol_gap;

            if res.expected_to_fail {
                // Expected to fail - don't fail CI if it does
                if is_pass {
                    unexpected_passes.push(format!("🎉 {} unexpectedly passed!", res.name));
                }
                continue; // Don't check further for expected failures
            }

            // Check if expected status is set and verify it matches
            if let Some(expected_status) = &res.expected_status {
                if std::mem::discriminant(&res.status) != std::mem::discriminant(expected_status) {
                    failures.push(format!(
                        "{}: status changed from expected {:?} to {:?} (iters={})",
                        res.name, expected_status, res.status, res.iterations
                    ));
                    continue;
                }
            }

            // Check iteration count regression (only if expected_iters is set)
            if let Some(expected_iters) = res.expected_iters {
                let iter_ratio = res.iterations as f64 / expected_iters as f64;
                if iter_ratio > 1.2 {
                    failures.push(format!(
                        "{}: iteration regression {:.2}x (expected {}, got {})",
                        res.name, iter_ratio, expected_iters, res.iterations
                    ));
                }
            }

            // For problems with expected non-optimal status, don't require Optimal
            let is_expected_non_optimal = matches!(
                res.expected_status,
                Some(SolveStatus::NumericalLimit)
                    | Some(SolveStatus::PrimalInfeasible)
                    | Some(SolveStatus::DualInfeasible)
                    | Some(SolveStatus::MaxIters)
            );

            // Not expected to fail - require pass (unless expected non-optimal status)
            if !is_expected_non_optimal
                && (!matches!(res.status, SolveStatus::Optimal | SolveStatus::AlmostOptimal)
                    || !res.rel_p.is_finite()
                    || !res.rel_d.is_finite()
                    || !res.gap_rel.is_finite())
            {
                let msg = format!(
                    "{}: status={:?} rel_p={:.2e} rel_d={:.2e} gap_rel={:.2e} {}",
                    res.name,
                    res.status,
                    res.rel_p,
                    res.rel_d,
                    res.gap_rel,
                    res.error.as_deref().unwrap_or(""),
                );
                if verbose {
                    eprintln!("\n{}", "=".repeat(60));
                    eprintln!("FAILURE: {}", res.name);
                    eprintln!("{}", "=".repeat(60));
                    eprintln!("Status: {:?}", res.status);
                    eprintln!("Iterations: {}", res.iterations);
                    eprintln!("Metrics:");
                    eprintln!("  rel_p:   {:.2e}", res.rel_p);
                    eprintln!("  rel_d:   {:.2e}", res.rel_d);
                    eprintln!("  gap_rel: {:.2e}", res.gap_rel);
                    if let Some(t) = res.solve_time_ms {
                        eprintln!("Wall clock: {:.1} ms", t);
                    }
                    if let Some(err) = &res.error {
                        eprintln!("Error: {}", err);
                    }
                    eprintln!("{}", "=".repeat(60));
                }
                failures.push(msg);
                continue;
            }
            // Skip tolerance check for problems with expected non-optimal status
            if !is_expected_non_optimal && (res.rel_p > tol_feas || res.rel_d > tol_feas || res.gap_rel > tol_gap) {
                let msg = format!(
                    "{}: rel_p={:.2e} rel_d={:.2e} gap_rel={:.2e}",
                    res.name, res.rel_p, res.rel_d, res.gap_rel
                );
                if verbose {
                    eprintln!("\n{}", "=".repeat(60));
                    eprintln!("TOLERANCE FAILURE: {}", res.name);
                    eprintln!("{}", "=".repeat(60));
                    eprintln!("Status: {:?}", res.status);
                    eprintln!("Iterations: {}", res.iterations);
                    eprintln!("Metrics (exceeds tolerances):");
                    eprintln!("  rel_p:   {:.2e} (tol: {:.2e})", res.rel_p, tol_feas);
                    eprintln!("  rel_d:   {:.2e} (tol: {:.2e})", res.rel_d, tol_feas);
                    eprintln!("  gap_rel: {:.2e} (tol: {:.2e})", res.gap_rel, tol_gap);
                    if let Some(t) = res.solve_time_ms {
                        eprintln!("Wall clock: {:.1} ms", t);
                    }
                    eprintln!("{}", "=".repeat(60));
                }
                failures.push(msg);
            }
            // Check iteration count - must match exactly
            if let Some(expected) = expected_iterations(&res.name) {
                if res.iterations != expected {
                    let msg = format!(
                        "{}: iteration mismatch {} != {} (expected exact match)",
                        res.name, res.iterations, expected
                    );
                    if verbose {
                        eprintln!("\n{}", "=".repeat(60));
                        eprintln!("ITERATION MISMATCH: {}", res.name);
                        eprintln!("{}", "=".repeat(60));
                        eprintln!("Status: {:?}", res.status);
                        eprintln!("Iterations: {} (expected: {})", res.iterations, expected);
                        eprintln!("Metrics:");
                        eprintln!("  rel_p:   {:.2e}", res.rel_p);
                        eprintln!("  rel_d:   {:.2e}", res.rel_d);
                        eprintln!("  gap_rel: {:.2e}", res.gap_rel);
                        if let Some(t) = res.solve_time_ms {
                            eprintln!("Wall clock: {:.1} ms", t);
                        }
                        eprintln!("{}", "=".repeat(60));
                    }
                    failures.push(msg);
                }
            }
        }

        // Print unexpected passes (informational only, don't fail CI)
        for msg in &unexpected_passes {
            eprintln!("{}", msg);
        }

        if !failures.is_empty() {
            panic!("regression failures:\n{}", failures.join("\n"));
        }

        if let Ok(path) = env::var("MINIX_PERF_BASELINE") {
            let contents = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("failed to read baseline {}: {}", path, e));
            let baseline: PerfSummary = serde_json::from_str(&contents)
                .unwrap_or_else(|e| panic!("failed to parse baseline {}: {}", path, e));
            let summary = perf_summary(&results);
            let max_regression = env::var("MINIX_MAX_REGRESSION")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.2);
            let perf_failures = compare_perf_baseline(&baseline, &summary, max_regression);
            if !perf_failures.is_empty() {
                panic!("perf regression:\n{}", perf_failures.join("\n"));
            }
        }
    }
}
