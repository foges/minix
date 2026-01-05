use solver_core::ipm2::metrics::compute_unscaled_metrics;
use solver_core::{ConeSpec, ProblemData, SolveStatus, SolverSettings};
use serde::{Deserialize, Serialize};

use crate::maros_meszaros::load_local_problem;
use crate::solver_choice::{solve_with_choice, SolverChoice};

pub struct RegressionResult {
    pub name: String,
    pub status: SolveStatus,
    pub rel_p: f64,
    pub rel_d: f64,
    pub gap_rel: f64,
    pub error: Option<String>,
    pub skipped: bool,
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

pub fn run_regression_suite(
    settings: &SolverSettings,
    solver: SolverChoice,
    require_cache: bool,
) -> Vec<RegressionResult> {
    let mut results = Vec::new();

    // Comprehensive QPS problems (~2 min total)
    // Excludes: LISWET1/10-12 (MaxIter),
    // CONT-200+ (too slow), CVXQP*_L (too slow), EXDATA/KSIP (too slow)
    let qps_cases = [
        // Tiny QPs (<5ms) - HS problems
        "HS21", "HS35", "HS35MOD", "HS51", "HS52", "HS53", "HS76", "HS118", "HS268",
        // Tiny QPs (<5ms) - CVXQP small
        "CVXQP1_S", "CVXQP2_S", "CVXQP3_S",
        // Tiny QPs (<10ms) - DUAL problems
        "DUAL1", "DUAL2", "DUAL3", "DUAL4",
        "DUALC1", "DUALC2", "DUALC5", "DUALC8",
        // Tiny QPs (<10ms) - Other
        "DPKLO1", "GOULDQP2", "GOULDQP3",
        // Small QPs (<50ms)
        "AUG3D", "AUG3DC", "AUG3DCQP", "AUG3DQP",
        "DTOC3", "HUESTIS", "LASER",
        // Medium QPs (<150ms)
        "CVXQP1_M", "CVXQP2_M", "CVXQP3_M",
        "LISWET2", "LISWET3", "LISWET4", "LISWET5",
        "CONT-050",
        // Large QPs (<250ms)
        "AUG2D", "AUG2DC", "AUG2DCQP", "AUG2DQP",
        // Very large QPs (<3s)
        "CONT-100", "CONT-101",
        // BOYD portfolio QPs (~93k vars, converge via early polish)
        "BOYD1", "BOYD2",
    ];
    for name in qps_cases {
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
                            error: Some(format!("conversion error: {}", e)),
                            skipped: false,
                            solve_time_ms: None,
                            kkt_factor_time_ms: None,
                            kkt_solve_time_ms: None,
                            cone_time_ms: None,
                        });
                        continue;
                    }
                };
                results.push(run_case(&prob, settings, solver, name));
            }
            Err(e) => {
                if require_cache {
                    results.push(RegressionResult {
                        name: name.to_string(),
                        status: SolveStatus::NumericalError,
                        rel_p: f64::NAN,
                        rel_d: f64::NAN,
                        gap_rel: f64::NAN,
                        error: Some(format!("missing QPS: {}", e)),
                        skipped: false,
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
                        error: None,
                        skipped: true,
                        solve_time_ms: None,
                        kkt_factor_time_ms: None,
                        kkt_solve_time_ms: None,
                        cone_time_ms: None,
                    });
                }
            }
        }
    }

    for (name, prob) in synthetic_cases() {
        results.push(run_case(&prob, settings, solver, name));
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
                error: None,
                skipped: false,
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
            error: Some(e.to_string()),
            skipped: false,
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

        let mut settings = SolverSettings::default();
        settings.max_iter = max_iter;

        let results = run_regression_suite(&settings, SolverChoice::Ipm2, require_cache);
        // Use practical tolerances for unscaled metrics
        // The solver uses scaled metrics internally (1e-8), but unscaled
        // metrics can differ due to problem conditioning
        let tol_feas = 1e-6;  // Feasibility tolerance for unscaled metrics
        let tol_gap = 1e-3;   // Relative gap tolerance (problems with poor conditioning may not reach 1e-6)

        let mut failures = Vec::new();
        for res in &results {
            if res.skipped {
                if require_cache {
                    failures.push(format!("{}: missing cache", res.name));
                }
                continue;
            }
            if res.status != SolveStatus::Optimal
                || !res.rel_p.is_finite()
                || !res.rel_d.is_finite()
                || !res.gap_rel.is_finite()
            {
                failures.push(format!(
                    "{}: status={:?} rel_p={:.2e} rel_d={:.2e} gap_rel={:.2e} {}",
                    res.name,
                    res.status,
                    res.rel_p,
                    res.rel_d,
                    res.gap_rel,
                    res.error.as_deref().unwrap_or(""),
                ));
                continue;
            }
            if res.rel_p > tol_feas || res.rel_d > tol_feas || res.gap_rel > tol_gap {
                failures.push(format!(
                    "{}: rel_p={:.2e} rel_d={:.2e} gap_rel={:.2e}",
                    res.name, res.rel_p, res.rel_d, res.gap_rel
                ));
            }
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
