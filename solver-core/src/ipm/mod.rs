//! Interior point method solver.
//!
//! HSDE formulation, predictor-corrector algorithm, and termination criteria.

pub mod hsde;
pub mod predcorr;
pub mod termination;

use crate::cones::{ConeKernel, ZeroCone, NonNegCone, SocCone, ExpCone, PowCone, PsdCone};
use crate::linalg::kkt::KktSolver;
use crate::presolve::apply_presolve;
use crate::presolve::ruiz::equilibrate;
use crate::presolve::singleton::detect_singleton_rows;
use crate::problem::{ProblemData, ConeSpec, SolverSettings, SolveResult, SolveStatus, SolveInfo};
use crate::ipm2::metrics::compute_unscaled_metrics;
use crate::scaling::ScalingBlock;
use hsde::{HsdeState, HsdeResiduals, compute_residuals, compute_mu};
use predcorr::{predictor_corrector_step, StepTimings};
use termination::{TerminationCriteria, check_termination};
use std::time::Instant;
use std::sync::OnceLock;

fn diagnostics_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        // Check new unified MINIX_VERBOSE first (level >= 2 means verbose)
        if let Ok(v) = std::env::var("MINIX_VERBOSE") {
            return v.parse::<u8>().map(|n| n >= 2).unwrap_or(false);
        }
        // Legacy: check MINIX_DIAGNOSTICS
        std::env::var("MINIX_DIAGNOSTICS")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false)
    })
}

fn min_slice(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

/// Main IPM solver.
///
/// Solves a convex conic optimization problem using the HSDE interior point method
/// with predictor-corrector steps.
///
/// # Arguments
///
/// * `prob` - Problem data
/// * `settings` - Solver settings
///
/// # Returns
///
/// `SolveResult` with solution, status, and diagnostics.
pub fn solve_ipm(
    prob: &ProblemData,
    settings: &SolverSettings,
) -> Result<SolveResult, Box<dyn std::error::Error>> {
    // Validate problem
    prob.validate()?;

    let orig_prob = prob.clone();
    let presolved = apply_presolve(prob);
    let prob = presolved.problem;
    let postsolve = presolved.postsolve;

    // Convert var_bounds to explicit constraints if present
    let prob = prob.with_bounds_as_constraints();

    let n = prob.num_vars();
    let m = prob.num_constraints();
    let orig_n = orig_prob.num_vars();

    // Apply Ruiz equilibration for numerical stability
    let (a_scaled, p_scaled, q_scaled, b_scaled, scaling) = equilibrate(
        &prob.A,
        prob.P.as_ref(),
        &prob.q,
        &prob.b,
        settings.ruiz_iters,
        &prob.cones,
    );

    // Create scaled problem
    let scaled_prob = ProblemData {
        P: p_scaled,
        q: q_scaled,
        A: a_scaled,
        b: b_scaled,
        cones: prob.cones.clone(),
        var_bounds: prob.var_bounds.clone(),
        integrality: prob.integrality.clone(),
    };

    let singleton_partition = detect_singleton_rows(&scaled_prob.A);
    if settings.verbose {
        eprintln!(
            "presolve: singleton_rows={} non_singleton_rows={}",
            singleton_partition.singleton_rows.len(),
            singleton_partition.non_singleton_rows.len(),
        );
    }

    // Precompute constant RHS used by the two-solve dtau strategy: rhs_x2 = -q.
    let neg_q: Vec<f64> = scaled_prob.q.iter().map(|&v| -v).collect();

    // Build cone kernels from cone specs
    let cones = build_cones(&scaled_prob.cones)?;

    // Compute total barrier degree
    let barrier_degree: usize = cones.iter().map(|c| c.barrier_degree()).sum();

    // Initialize HSDE state
    let mut state = HsdeState::new(n, m);
    state.initialize_with_prob(&cones, &scaled_prob);
    if let Some(warm) = settings.warm_start.as_ref() {
        state.apply_warm_start(warm, &postsolve, &scaling, &cones);
    }

    // Initialize residuals
    let mut residuals = HsdeResiduals::new(n, m);

    // Initialize KKT solver
    // For LPs (P=None) or very sparse QPs, use higher regularization to stabilize.
    // The (1,1) block is only εI for LPs. With small ε, solving
    //   [εI, A^T] [dx]   [rhs_x]
    //   [A,  -(H)] [dz] = [rhs_z]
    // gives dx ≈ rhs_x/ε, which blows up for small ε.
    // Use a small ε floor for stability while preserving high-accuracy convergence.
    let mut static_reg = settings.static_reg.max(1e-8);

    // Build initial scaling structure for KKT assembly.
    let initial_scaling: Vec<ScalingBlock> = cones.iter().map(|cone| {
        let dim = cone.dim();
        if cone.barrier_degree() == 0 {
            ScalingBlock::Zero { dim }
        } else if (cone.as_ref() as &dyn std::any::Any).downcast_ref::<SocCone>().is_some() {
            // SOC creates a dense block in KKT
            ScalingBlock::SocStructured { w: vec![1.0; dim] }
        } else if (cone.as_ref() as &dyn std::any::Any).downcast_ref::<ExpCone>().is_some()
            || (cone.as_ref() as &dyn std::any::Any).downcast_ref::<PowCone>().is_some()
        {
            ScalingBlock::Dense3x3 { h: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] }
        } else if let Some(psd) = (cone.as_ref() as &dyn std::any::Any).downcast_ref::<PsdCone>() {
            let n = psd.size();
            let mut w_factor = vec![0.0; n * n];
            for i in 0..n {
                w_factor[i * n + i] = 1.0;
            }
            ScalingBlock::PsdStructured { w_factor, n }
        } else {
            // NonNeg uses diagonal scaling
            ScalingBlock::Diagonal { d: vec![1.0; dim] }
        }
    }).collect();

    let mut kkt = KktSolver::new_with_singleton_elimination(
        n,
        m,
        static_reg,
        settings.dynamic_reg_min_pivot,
        &scaled_prob.A,
        &initial_scaling,
    );

    // Perform symbolic factorization once with initial scaling structure.
    // This determines the sparsity pattern of L and the elimination tree.
    // Subsequent calls to factor() reuse this symbolic factorization.

    if let Err(e) = kkt.initialize(scaled_prob.P.as_ref(), &scaled_prob.A, &initial_scaling) {
        return Err(format!("KKT symbolic factorization failed: {}", e).into());
    }

    // Termination criteria
    let criteria = TerminationCriteria {
        tol_feas: settings.tol_feas,
        tol_gap: settings.tol_gap,
        tol_gap_rel: settings.tol_gap,  // Use same tolerance for relative gap
        tol_infeas: settings.tol_infeas,
        max_iter: settings.max_iter,
        ..Default::default()
    };

    // Initial barrier parameter
    let mut mu = compute_mu(&state, barrier_degree);

    let mut status = SolveStatus::NumericalError;  // Will be overwritten
    let mut iter = 0;
    let mut consecutive_failures = 0;
    const MAX_CONSECUTIVE_FAILURES: usize = 3;
    let mut timings = StepTimings::default();
    let mut last_dynamic_bumps = 0;
    let start = Instant::now();

    if settings.verbose {
        println!("Minix IPM Solver");
        println!("================");
        println!("Problem: n = {}, m = {}, cones = {:?}", n, m, scaled_prob.cones.len());
        if settings.ruiz_iters > 0 {
            println!("Ruiz equilibration: {} iterations", settings.ruiz_iters);
        }
        println!("Barrier degree: {}", barrier_degree);
        println!("Initial state: x={:?}, s={:?}, z={:?}, tau={}, kappa={}",
                 state.x, state.s, state.z, state.tau, state.kappa);
        println!("Initial mu: {}", mu);
        println!();
        println!(
            "{:>4} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>10}",
            "Iter", "μ", "Primal Res", "Dual Res", "GapObj", "GapComp", "TauKappa", "Alpha"
        );
        println!("{}", "-".repeat(100));
    }

    // Main IPM loop
    while iter < settings.max_iter {
        // Compute residuals
        compute_residuals(&scaled_prob, &state, &mut residuals);

        // Check termination
        if let Some(term_status) = check_termination(&prob, &scaling, &state, iter, &criteria) {
            status = term_status;
            break;
        }

        // Take predictor-corrector step
        let step_result = match predictor_corrector_step(
            &mut kkt,
            &scaled_prob,
            &neg_q,
            &mut state,
            &residuals,
            &cones,
            mu,
            barrier_degree,
            settings,
            &mut timings,
        ) {
            Ok(result) => {
                consecutive_failures = 0;  // Reset on success
                result
            }
            Err(e) => {
                consecutive_failures += 1;

                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                    if settings.verbose {
                        eprintln!("IPM step failed {} times: {}", consecutive_failures, e);
                    }
                    status = SolveStatus::NumericalError;
                    break;
                }

                // Infeasible-start recovery: push state back to cone interior
                if settings.verbose {
                    eprintln!("IPM step failed (attempt {}), recovering: {}", consecutive_failures, e);
                }

                // Push s and z back to interior with larger margin
                let recovery_margin = (mu * 0.1).clamp(1e-4, 1e4);
                state.push_to_interior(&cones, recovery_margin);

                // Recompute mu after recovery
                mu = compute_mu(&state, barrier_degree);

                // Skip to next iteration (will recompute residuals and retry)
                iter += 1;
                continue;
            }
        };

        // Update mu
        mu = step_result.mu_new;

        // Check for divergence or numerical issues
        if !mu.is_finite() || mu > 1e15 {
            consecutive_failures += 1;
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                if settings.verbose {
                    eprintln!("Divergence detected: μ = {}", mu);
                }
                status = SolveStatus::NumericalError;
                break;
            }

            // Recovery: push back to interior
            if settings.verbose {
                eprintln!("Numerical issue detected (μ = {}), recovering", mu);
            }
            state.push_to_interior(&cones, 1e-2);
            mu = compute_mu(&state, barrier_degree);
        }

        // Normalize tau+kappa to prevent HSDE drift (keep tau+kappa near 2.0)
        // This prevents kappa explosion on problems like QFORPLAN
        // Use tau+kappa normalization instead of tau-only to bound both variables
        state.normalize_tau_kappa_if_needed(0.5, 50.0, 2.0);

        if diagnostics_enabled() {
            let min_s = min_slice(&state.s);
            let min_z = min_slice(&state.z);
            eprintln!(
                "iter {:4} alpha={:.3e} alpha_sz={:.3e} min_s={:.3e} min_z={:.3e} mu={:.3e}",
                iter,
                step_result.alpha,
                step_result.alpha_sz,
                min_s,
                min_z,
                mu
            );
        }

        // Verbose output
        if settings.verbose {
            let (rx_norm, rz_norm, _) = residuals.norms();
            let primal_res = rz_norm / state.tau.max(1.0);
            let dual_res = rx_norm / state.tau.max(1.0);

            // Compute gap (on scaled problem)
            let x_bar: Vec<f64> = state.x.iter().map(|xi| xi / state.tau).collect();
            let z_bar: Vec<f64> = state.z.iter().map(|zi| zi / state.tau).collect();

            // Compute x^T P x - only process upper triangle (row <= col) to handle both
            // upper-triangular and full symmetric matrix storage
            let mut xpx = 0.0;
            if let Some(ref p) = scaled_prob.P {
                for col in 0..n {
                    if let Some(col_view) = p.outer_view(col) {
                        for (row, &val) in col_view.iter() {
                            if row < col {
                                // Upper triangle off-diagonal
                                xpx += 2.0 * x_bar[row] * val * x_bar[col];
                            } else if row == col {
                                // Diagonal
                                xpx += x_bar[row] * val * x_bar[col];
                            }
                            // Skip lower triangle (row > col)
                        }
                    }
                }
            }

            let qtx: f64 = scaled_prob.q.iter().zip(x_bar.iter()).map(|(qi, xi)| qi * xi).sum();
            let btz: f64 = scaled_prob.b.iter().zip(z_bar.iter()).map(|(bi, zi)| bi * zi).sum();
            let gap_obj = (xpx + qtx + btz).abs();

            let s_dot_z: f64 = state
                .s
                .iter()
                .zip(state.z.iter())
                .map(|(si, zi)| si * zi)
                .sum();
            let tau_kappa = state.tau * state.kappa;
            let gap_comp = if state.tau > 0.0 {
                s_dot_z / (state.tau * state.tau)
            } else {
                s_dot_z
            };

            println!(
                "{:4} {:12.4e} {:12.4e} {:12.4e} {:12.4e} {:12.4e} {:12.4e} {:10.4}",
                iter, mu, primal_res, dual_res, gap_obj, gap_comp, tau_kappa, step_result.alpha
            );
        }

        last_dynamic_bumps = kkt.dynamic_bumps();
        static_reg = kkt.static_reg();
        iter += 1;
    }

    if iter >= settings.max_iter && status == SolveStatus::NumericalError {
        status = SolveStatus::MaxIters;
    }

    if settings.verbose {
        println!("{}", "-".repeat(72));
        println!("Status: {:?}", status);
        println!("Iterations: {}", iter);
        println!();
    }

    // Extract solution in scaled space
    let x_scaled: Vec<f64> = if state.tau > 1e-8 {
        state.x.iter().map(|xi| xi / state.tau).collect()
    } else {
        vec![0.0; n]
    };

    let s_scaled: Vec<f64> = if state.tau > 1e-8 {
        state.s.iter().map(|si| si / state.tau).collect()
    } else {
        vec![0.0; m]
    };

    let z_scaled: Vec<f64> = if state.tau > 1e-8 {
        state.z.iter().map(|zi| zi / state.tau).collect()
    } else {
        vec![0.0; m]
    };

    // Unscale solution back to original coordinates
    let x_unscaled = scaling.unscale_x(&x_scaled);
    let s_unscaled = scaling.unscale_s(&s_scaled);
    let z_unscaled = scaling.unscale_z(&z_scaled);

    let x = postsolve.recover_x(&x_unscaled);
    let s = postsolve.recover_s(&s_unscaled, &x);
    let z = postsolve.recover_z(&z_unscaled);

    // Compute objective value using ORIGINAL (unscaled) problem data
    // Only process upper triangle (row <= col) to handle both upper-triangular
    // and full symmetric matrix storage
    let mut obj_val = 0.0;
    if let Some(ref p) = orig_prob.P {
        let mut px = vec![0.0; orig_n];
        for col in 0..orig_n {
            if let Some(col_view) = p.outer_view(col) {
                for (row, &val) in col_view.iter() {
                    if row <= col {
                        px[row] += val * x[col];
                        if row != col {
                            px[col] += val * x[row];
                        }
                    }
                    // Skip lower triangle (row > col)
                }
            }
        }
        for i in 0..orig_n {
            obj_val += 0.5 * x[i] * px[i];
        }
    }
    for i in 0..orig_n {
        obj_val += orig_prob.q[i] * x[i];
    }

    let orig_prob_bounds = orig_prob.with_bounds_as_constraints();
    let (primal_res, dual_res, gap) = {
        let mut r_p = vec![0.0; orig_prob_bounds.num_constraints()];
        let mut r_d = vec![0.0; orig_prob_bounds.num_vars()];
        let mut p_x = vec![0.0; orig_prob_bounds.num_vars()];
        let metrics = compute_unscaled_metrics(
            &orig_prob_bounds.A,
            orig_prob_bounds.P.as_ref(),
            &orig_prob_bounds.q,
            &orig_prob_bounds.b,
            &x,
            &s,
            &z,
            &mut r_p,
            &mut r_d,
            &mut p_x,
        );
        (metrics.rel_p, metrics.rel_d, metrics.gap_rel)
    };

    Ok(SolveResult {
        status,
        x,
        s,
        z,
        obj_val,
        info: SolveInfo {
            iters: iter,
            solve_time_ms: start.elapsed().as_millis() as u64,
            kkt_factor_time_ms: timings.kkt_factor.as_millis() as u64,
            kkt_solve_time_ms: timings.kkt_solve.as_millis() as u64,
            cone_time_ms: timings.cone.as_millis() as u64,
            primal_res,
            dual_res,
            gap,
            mu,
            reg_static: static_reg,
            reg_dynamic_bumps: last_dynamic_bumps,
        },
    })
}

/// Build cone kernels from cone specifications.
fn build_cones(specs: &[ConeSpec]) -> Result<Vec<Box<dyn ConeKernel>>, Box<dyn std::error::Error>> {
    let mut cones: Vec<Box<dyn ConeKernel>> = Vec::new();

    for spec in specs {
        match spec {
            ConeSpec::Zero { dim } => {
                cones.push(Box::new(ZeroCone::new(*dim)));
            }
            ConeSpec::NonNeg { dim } => {
                cones.push(Box::new(NonNegCone::new(*dim)));
            }
            ConeSpec::Soc { dim } => {
                cones.push(Box::new(SocCone::new(*dim)));
            }
            ConeSpec::Psd { n } => {
                cones.push(Box::new(PsdCone::new(*n)));
            }
            ConeSpec::Exp { count } => {
                for _ in 0..*count {
                    cones.push(Box::new(ExpCone::new(1)));
                }
            }
            ConeSpec::Pow { cones: pow_cones } => {
                for pow in pow_cones {
                    cones.push(Box::new(PowCone::new(vec![pow.alpha])));
                }
            }
        }
    }

    Ok(cones)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::sparse;

    #[test]
    fn test_solve_simple_lp() {
        // min x1 + x2
        // s.t. x1 + x2 = 1
        //      x1, x2 >= 0
        //
        // Optimal: any point with x1 + x2 = 1, x >= 0, e.g., [0.5, 0.5], obj = 1.0
        //
        // Reformulated with bounds:
        //   x1 + x2 + s_eq = 1, s_eq = 0  (equality)
        //   -x1 + s_1 = 0, s_1 >= 0       (bound x1 >= 0)
        //   -x2 + s_2 = 0, s_2 >= 0       (bound x2 >= 0)

        // A is 3x2: [equality, bound x1, bound x2]
        let a_triplets = vec![
            (0, 0, 1.0), (0, 1, 1.0),  // x1 + x2 = 1
            (1, 0, -1.0),              // -x1 + s_1 = 0
            (2, 1, -1.0),              // -x2 + s_2 = 0
        ];

        let prob = ProblemData {
            P: None,
            q: vec![1.0, 1.0],
            A: sparse::from_triplets(3, 2, a_triplets),
            b: vec![1.0, 0.0, 0.0],
            cones: vec![
                ConeSpec::Zero { dim: 1 },    // equality constraint
                ConeSpec::NonNeg { dim: 2 },  // bounds x >= 0
            ],
            var_bounds: None,
            integrality: None,
        };

        let settings = SolverSettings {
            verbose: true,
            max_iter: 50,
            tol_feas: 1e-6,
            tol_gap: 1e-6,
            ..Default::default()
        };

        let result = solve_ipm(&prob, &settings).expect("Solve failed");

        println!("Result: {:?}", result);
        println!("x = {:?}", result.x);
        println!("obj = {}", result.obj_val);

        // Check status
        assert!(matches!(result.status, SolveStatus::Optimal | SolveStatus::MaxIters));

        // Check solution satisfies constraints
        if result.status == SolveStatus::Optimal {
            let sum = result.x[0] + result.x[1];
            assert!((sum - 1.0).abs() < 0.1, "Constraint not satisfied: {}", sum);
            assert!(result.x[0] >= -0.1);
            assert!(result.x[1] >= -0.1);
            assert!((result.obj_val - 1.0).abs() < 0.1);
        }
    }
}
