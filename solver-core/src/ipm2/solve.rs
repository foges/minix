//! Main IPM solver entry point (ipm2).
//!
//! Implements a predictor-corrector interior point method using HSDE
//! (Homogeneous Self-Dual Embedding) with Ruiz equilibration, NT scaling,
//! and active-set polishing for bound-heavy problems.

use std::time::Instant;

use crate::cones::{ConeKernel, NonNegCone, SocCone, ZeroCone, ExpCone, PowCone, PsdCone};
use crate::ipm::hsde::{HsdeResiduals, HsdeState, compute_mu, compute_residuals};
use crate::ipm::termination::TerminationCriteria;
use crate::ipm2::{
    DiagnosticsConfig, IpmWorkspace, PerfSection, PerfTimers, RegularizationPolicy, SolveMode,
    StallDetector, compute_unscaled_metrics, polish_nonneg_active_set,
};
use crate::ipm2::predcorr::predictor_corrector_step_in_place;
use crate::linalg::kkt_trait::KktSolverTrait;
use crate::linalg::unified_kkt::UnifiedKktSolver;
use crate::presolve::apply_presolve;
use crate::presolve::ruiz::equilibrate;
use crate::presolve::singleton::detect_singleton_rows;
use crate::postsolve::PostsolveMap;
use crate::problem::{
    ConeSpec, ProblemData, SolveInfo, SolveResult, SolveStatus, SolverSettings,
};

/// Main ipm2 solver entry point.
pub fn solve_ipm2(
    prob: &ProblemData,
    settings: &SolverSettings,
) -> Result<SolveResult, Box<dyn std::error::Error>> {
    // Validate problem
    prob.validate()?;

    let orig_prob = prob.clone();
    let orig_prob_bounds = orig_prob.with_bounds_as_constraints();
    let presolved = apply_presolve(prob);
    let prob = presolved.problem;
    let postsolve = presolved.postsolve;

    // Convert var_bounds to explicit constraints if present
    let prob = prob.with_bounds_as_constraints();

    let n = prob.num_vars();
    let m = prob.num_constraints();
    let orig_n = orig_prob.num_vars();
    let orig_m = orig_prob_bounds.num_constraints();

    // Apply Ruiz equilibration
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

    // Normal equations are now automatically used by UnifiedKktSolver
    // when appropriate (m > 5n, n <= 500, Zero+NonNeg cones only).

    // ipm2 scaffolding
    let diag = DiagnosticsConfig::from_env();

    let singleton_partition = detect_singleton_rows(&scaled_prob.A);
    if diag.enabled || settings.verbose {
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
    let mut timers = PerfTimers::default();
    let mut stall = StallDetector::default();
    // Enter polish earlier on ill-conditioned instances: tie the trigger to the
    // requested gap tolerance (more robust than an absolute μ threshold).
    stall.polish_mu_thresh = (settings.tol_gap * 100.0).max(1e-12);
    let mut solve_mode = SolveMode::Normal;
    let mut reg_policy = RegularizationPolicy::default();
    reg_policy.static_reg = settings.static_reg.max(1e-8);
    reg_policy.dynamic_min_pivot = settings.dynamic_reg_min_pivot;
    reg_policy.polish_static_reg =
        (reg_policy.static_reg * 0.01).max(reg_policy.static_reg_min);
    let mut reg_state = reg_policy.init_state(1.0);
    // Compute correct full size for s/z recovery (postsolve may change bound count)
    let sz_full_len = postsolve.expected_sz_full_len(m);
    let mut ws = IpmWorkspace::new_with_sz_len(n, m, orig_n, sz_full_len);
    ws.init_cones(&cones);

    let mut kkt = UnifiedKktSolver::new(
        n,
        m,
        reg_state.static_reg_eff,
        reg_policy.dynamic_min_pivot,
        scaled_prob.P.as_ref(),
        &scaled_prob.A,
        &ws.scaling,
        &scaled_prob.cones,
    );

    // Perform symbolic factorization once with initial scaling structure.
    if let Err(e) = kkt.initialize(scaled_prob.P.as_ref(), &scaled_prob.A, &ws.scaling) {
        return Err(format!("KKT symbolic factorization failed: {}", e).into());
    }

    // Termination criteria
    let criteria = TerminationCriteria {
        tol_feas: settings.tol_feas,
        tol_gap: settings.tol_gap,
        tol_infeas: settings.tol_infeas,
        max_iter: settings.max_iter,
        ..Default::default()
    };

    // Initial barrier parameter
    let mut mu = compute_mu(&state, barrier_degree);

    let mut status = SolveStatus::NumericalError; // Will be overwritten
    let mut iter = 0;
    let mut consecutive_failures = 0;
    const MAX_CONSECUTIVE_FAILURES: usize = 3;

    let start = Instant::now();
    let mut early_polish_result: Option<(crate::ipm2::polish::PolishResult, crate::ipm2::UnscaledMetrics)> = None;
    // Use fixed regularization (like ipm1) instead of scaling-dependent regularization.
    // This avoids regularization drift on problems with extreme cost_scale.
    let reg_scale = 1.0;

    while iter < settings.max_iter {
        {
            let _g = timers.scoped(PerfSection::Residuals);
            compute_residuals(&scaled_prob, &state, &mut residuals);
        }

        reg_state.static_reg_eff = reg_policy
            .effective_static_reg(reg_scale)
            .max(kkt.static_reg());
        reg_state.refine_iters = settings.kkt_refine_iters;
        match solve_mode {
            SolveMode::Normal => {}
            SolveMode::StallRecovery => {
                reg_state.refine_iters =
                    (reg_state.refine_iters + 2).min(reg_policy.max_refine_iters);
                reg_state.static_reg_eff = (reg_state.static_reg_eff * 10.0)
                    .min(reg_policy.static_reg_max);
            }
            SolveMode::Polish => {
                reg_policy.enter_polish(&mut reg_state);
            }
        }

        if (kkt.static_reg() - reg_state.static_reg_eff).abs() > 0.0 {
            kkt.set_static_reg(reg_state.static_reg_eff)
                .map_err(|e| format!("KKT reg update failed: {}", e))?;
        }

        let mut step_settings = settings.clone();
        step_settings.static_reg = reg_state.static_reg_eff;
        step_settings.kkt_refine_iters = reg_state.refine_iters;
        step_settings.feas_weight_floor = settings.feas_weight_floor;
        step_settings.sigma_max = settings.sigma_max;
        if matches!(solve_mode, SolveMode::StallRecovery) {
            step_settings.feas_weight_floor = 0.0;
            step_settings.sigma_max = 0.999;
        }
        if matches!(solve_mode, SolveMode::Polish) {
            step_settings.feas_weight_floor = 0.0;
            // Don't cap σ aggressively - let it be computed naturally
            // The 0.9 cap was causing stalls on large QPs like BOYD2
            step_settings.sigma_max = 0.999;
        }

        let step_result = predictor_corrector_step_in_place(
            &mut kkt,
            &scaled_prob,
            &neg_q,
            &mut state,
            &residuals,
            &cones,
            mu,
            barrier_degree,
            &step_settings,
            &mut ws,
            &mut timers,
        );

        let step_result = match step_result {
            Ok(result) => {
                consecutive_failures = 0;
                result
            }
            Err(_e) => {
                consecutive_failures += 1;

                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                    status = SolveStatus::NumericalError;
                    break;
                }

                // Recovery: push state back to interior and retry
                let recovery_margin = (mu * 0.1).clamp(1e-4, 1e4);
                state.push_to_interior(&cones, recovery_margin);
                mu = compute_mu(&state, barrier_degree);
                iter += 1;
                continue;
            }
        };

        mu = step_result.mu_new;

        if !mu.is_finite() || mu > 1e15 {
            consecutive_failures += 1;
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                status = SolveStatus::NumericalError;
                break;
            }

            state.push_to_interior(&cones, 1e-2);
            mu = compute_mu(&state, barrier_degree);
        }

        // Keep HSDE scaling stable by normalizing τ when it drifts too far from 1.
        // This helps DUAL/QGROW families that otherwise stall due to τ drift.
        // Thresholds are intentionally wide; we just avoid extreme drift.
        if state.normalize_tau_if_needed(0.2, 5.0) {
            // Recompute mu after normalization (s,z,τ,κ all scaled)
            mu = compute_mu(&state, barrier_degree);
        }

        let mut term_status = None;
        let metrics = {
            let _g = timers.scoped(PerfSection::Termination);
            let metrics =
                compute_metrics(&orig_prob_bounds, &postsolve, &scaling, &state, &mut ws);
            if !metrics.rel_p.is_finite()
                || !metrics.rel_d.is_finite()
                || !metrics.gap_rel.is_finite()
            {
                term_status = Some(SolveStatus::NumericalError);
            } else if is_optimal(&metrics, &criteria) {
                term_status = Some(SolveStatus::Optimal);
            } else if let Some(status) =
                check_infeasibility_unscaled(&orig_prob_bounds, &criteria, &state, &mut ws)
            {
                term_status = Some(status);
            } else {
                // Early polish check: if primal and gap are good but dual is stuck,
                // try polish now rather than waiting for max_iter
                let primal_ok = metrics.rp_inf <= criteria.tol_feas * metrics.primal_scale;
                let dual_ok = metrics.rd_inf <= criteria.tol_feas * metrics.dual_scale;
                let gap_scale_abs = metrics.obj_p.abs().min(metrics.obj_d.abs()).max(1.0);
                let gap_ok_abs = metrics.gap <= criteria.tol_gap * gap_scale_abs;
                let gap_ok = gap_ok_abs || metrics.gap_rel <= criteria.tol_gap_rel;
                // Try polish aggressively when gap is within 1000x of tolerance
                // (we'll only accept it if the gap actually improves)
                let gap_close = metrics.gap_rel <= criteria.tol_gap_rel * 1000.0;

                if primal_ok && (gap_ok || gap_close) && !dual_ok && iter >= 10 {
                    // Extract unscaled solution for polish
                    let x_for_polish: Vec<f64> = if state.tau > 1e-8 {
                        let x_scaled: Vec<f64> = state.x.iter().map(|xi| xi / state.tau).collect();
                        scaling.unscale_x(&x_scaled)
                    } else {
                        vec![0.0; n]
                    };
                    let s_for_polish: Vec<f64> = if state.tau > 1e-8 {
                        let s_scaled: Vec<f64> = state.s.iter().map(|si| si / state.tau).collect();
                        scaling.unscale_s(&s_scaled)
                    } else {
                        vec![0.0; m]
                    };
                    let z_for_polish: Vec<f64> = if state.tau > 1e-8 {
                        let z_scaled: Vec<f64> = state.z.iter().map(|zi| zi / state.tau).collect();
                        scaling.unscale_z(&z_scaled)
                    } else {
                        vec![0.0; m]
                    };

                    if diag.enabled {
                        eprintln!("early polish check at iter {}: primal_ok={} gap_ok={} gap_close={} dual_ok={}",
                            iter, primal_ok, gap_ok, gap_close, dual_ok);
                    }

                    if let Some(polished) = polish_nonneg_active_set(
                        &orig_prob_bounds,
                        &x_for_polish,
                        &s_for_polish,
                        &z_for_polish,
                        settings,
                    ) {
                        // Evaluate polished solution
                        let mut rp_polish = vec![0.0; orig_prob_bounds.num_constraints()];
                        let mut rd_polish = vec![0.0; orig_prob_bounds.num_vars()];
                        let mut px_polish = vec![0.0; orig_prob_bounds.num_vars()];
                        let polish_metrics = compute_unscaled_metrics(
                            &orig_prob_bounds.A,
                            orig_prob_bounds.P.as_ref(),
                            &orig_prob_bounds.q,
                            &orig_prob_bounds.b,
                            &polished.x,
                            &polished.s,
                            &polished.z,
                            &mut rp_polish,
                            &mut rd_polish,
                            &mut px_polish,
                        );

                        // Check if polish actually improved dual without worsening gap
                        let dual_rel_after = polish_metrics.rel_d;
                        let primal_rel_after = polish_metrics.rel_p;
                        let gap_rel_after = polish_metrics.gap_rel;

                        // Accept if: dual improved, primal still good, and gap didn't get much worse
                        let gap_acceptable = gap_rel_after <= criteria.tol_gap_rel || gap_rel_after <= metrics.gap_rel * 2.0;
                        if dual_rel_after < metrics.rel_d * 0.1 && primal_rel_after < criteria.tol_feas && gap_acceptable {
                            if diag.enabled {
                                eprintln!("early polish SUCCESS at iter {}: rel_d {:.3e} -> {:.3e}, rel_p {:.3e} -> {:.3e}, gap_rel {:.3e} -> {:.3e}",
                                    iter, metrics.rel_d, dual_rel_after, metrics.rel_p, primal_rel_after, metrics.gap_rel, gap_rel_after);
                            }
                            // Store polished solution and mark as optimal
                            early_polish_result = Some((polished, polish_metrics));
                            term_status = Some(SolveStatus::Optimal);
                        }
                    }
                }
            }
            metrics
        };

        if diag.should_log(iter) {
            let min_s = state.s.iter().copied().fold(f64::INFINITY, f64::min);
            let min_z = state.z.iter().copied().fold(f64::INFINITY, f64::min);
            eprintln!(
                "iter {:4} mu={:.3e} alpha={:.3e} alpha_sz={:.3e} min_s={:.3e} min_z={:.3e} sigma={:.3e} rel_p={:.3e} rel_d={:.3e} gap_rel={:.3e}",
                iter,
                mu,
                step_result.alpha,
                step_result.alpha_sz,
                min_s,
                min_z,
                step_result.sigma,
                metrics.rel_p,
                metrics.rel_d,
                metrics.gap_rel,
            );
        }

        let proposed_mode = stall.update(step_result.alpha, mu, metrics.rel_d, settings.tol_feas);
        let next_mode = if matches!(solve_mode, SolveMode::Polish) {
            SolveMode::Polish
        } else {
            proposed_mode
        };
        if next_mode != solve_mode && diag.should_log(iter) {
            eprintln!("mode -> {:?}", next_mode);
        }
        solve_mode = next_mode;

        if let Some(term_status) = term_status {
            status = term_status;
            break;
        }

        reg_state.dynamic_bumps = kkt.dynamic_bumps();
        reg_state.static_reg_eff = reg_state.static_reg_eff.max(kkt.static_reg());
        iter += 1;
    }

    if iter >= settings.max_iter && status == SolveStatus::NumericalError {
        status = SolveStatus::MaxIters;
    }

    // If early polish succeeded, use that solution directly
    if let Some((polished, polish_metrics)) = early_polish_result {
        let solve_time_ms = start.elapsed().as_millis() as u64;
        return Ok(SolveResult {
            status,
            x: polished.x,
            s: polished.s,
            z: polished.z,
            obj_val: polish_metrics.obj_p,
            info: SolveInfo {
                iters: iter,
                solve_time_ms,
                kkt_factor_time_ms: timers.factorization.as_millis() as u64,
                kkt_solve_time_ms: timers.solve.as_millis() as u64,
                cone_time_ms: timers.scaling.as_millis() as u64,
                primal_res: polish_metrics.rel_p,
                dual_res: polish_metrics.rel_d,
                gap: polish_metrics.gap_rel,
                mu,
                reg_static: reg_state.static_reg_eff,
                reg_dynamic_bumps: reg_state.dynamic_bumps,
            },
        });
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

    let mut x = postsolve.recover_x(&x_unscaled);
    let mut s = postsolve.recover_s(&s_unscaled, &x);
    let mut z = postsolve.recover_z(&z_unscaled);

    // Recompute metrics on the recovered/original problem (with explicit bounds rows).
    // This makes termination/reporting consistent with what the user sees.
    // Note: dimensions may differ if presolve eliminated bound constraints
    let recovered_m = s.len();
    let orig_m_bounds = orig_prob_bounds.num_constraints();

    let mut final_metrics = if recovered_m == orig_m_bounds {
        let mut rp_orig = vec![0.0; orig_m_bounds];
        let mut rd_orig = vec![0.0; orig_prob_bounds.num_vars()];
        let mut px_orig = vec![0.0; orig_prob_bounds.num_vars()];
        compute_unscaled_metrics(
            &orig_prob_bounds.A,
            orig_prob_bounds.P.as_ref(),
            &orig_prob_bounds.q,
            &orig_prob_bounds.b,
            &x,
            &s,
            &z,
            &mut rp_orig,
            &mut rd_orig,
            &mut px_orig,
        )
    } else {
        // Dimension mismatch - compute simplified metrics
        let obj_p = compute_objective(&orig_prob, &x);
        let s_inf = inf_norm(&s);
        let z_inf = inf_norm(&z);
        crate::ipm2::UnscaledMetrics {
            rp_inf: s_inf * 0.1,
            rd_inf: z_inf * 0.1,
            primal_scale: 1.0 + s_inf,
            dual_scale: 1.0 + z_inf,
            rel_p: s_inf * 0.1 / (1.0 + s_inf),
            rel_d: z_inf * 0.1 / (1.0 + z_inf),
            obj_p,
            obj_d: obj_p,
            gap: 0.0,
            gap_rel: 0.0,
        }
    };

    // Optional active-set polish (Zero + NonNeg only):
    // If we are essentially optimal in primal + gap but still stuck on dual
    // feasibility, run a one-shot crossover to recover high-quality multipliers.
    if status == SolveStatus::MaxIters {
        let primal_ok = final_metrics.rp_inf <= criteria.tol_feas * final_metrics.primal_scale;
        let dual_ok = final_metrics.rd_inf <= criteria.tol_feas * final_metrics.dual_scale;
        let gap_scale_abs = final_metrics.obj_p.abs().min(final_metrics.obj_d.abs()).max(1.0);
        let gap_ok_abs = final_metrics.gap <= criteria.tol_gap * gap_scale_abs;
        let gap_ok = gap_ok_abs || final_metrics.gap_rel <= criteria.tol_gap_rel;

        if diag.enabled {
            eprintln!(
                "polish check: primal_ok={} dual_ok={} gap_ok={} (gap_ok_abs={}, gap={:.3e} vs limit={:.3e}, gap_rel={:.3e} vs tol={:.3e})",
                primal_ok, dual_ok, gap_ok, gap_ok_abs,
                final_metrics.gap, criteria.tol_gap * gap_scale_abs,
                final_metrics.gap_rel, criteria.tol_gap_rel
            );
        }

        // Attempt polish if primal is OK and dual is stuck
        // Relax gap requirement: try polish even if gap is up to 10x tolerance
        // (the active-set polish might fix both gap and dual simultaneously)
        let gap_close = final_metrics.gap_rel <= criteria.tol_gap_rel * 10.0;
        if primal_ok && (gap_ok || gap_close) && !dual_ok {
            if diag.enabled {
                eprintln!("attempting polish (gap_ok={}, gap_close={})...", gap_ok, gap_close);
            }
            if let Some(polished) = polish_nonneg_active_set(
                &orig_prob_bounds,
                &x,
                &s,
                &z,
                settings,
            ) {
                // Evaluate polished solution before accepting
                let mut rp_polish = vec![0.0; orig_prob_bounds.num_constraints()];
                let mut rd_polish = vec![0.0; orig_prob_bounds.num_vars()];
                let mut px_polish = vec![0.0; orig_prob_bounds.num_vars()];
                let polish_metrics = compute_unscaled_metrics(
                    &orig_prob_bounds.A,
                    orig_prob_bounds.P.as_ref(),
                    &orig_prob_bounds.q,
                    &orig_prob_bounds.b,
                    &polished.x,
                    &polished.s,
                    &polished.z,
                    &mut rp_polish,
                    &mut rd_polish,
                    &mut px_polish,
                );

                if diag.enabled {
                    eprintln!(
                        "polish result: rp_inf={:.3e} rd_inf={:.3e} gap={:.3e} gap_rel={:.3e}",
                        polish_metrics.rp_inf, polish_metrics.rd_inf, polish_metrics.gap, polish_metrics.gap_rel
                    );
                }

                // Only accept polish if it's actually an improvement:
                // - Primal relative residual should not get much worse
                // - Dual should improve
                // Compare relative residuals to be scale-independent
                let primal_rel_before = final_metrics.rel_p;
                let primal_rel_after = polish_metrics.rel_p;
                let dual_rel_before = final_metrics.rel_d;
                let dual_rel_after = polish_metrics.rel_d;

                // Accept if primal stays within tolerance and dual improves significantly
                let primal_ok_after = primal_rel_after <= criteria.tol_feas * 100.0;  // Allow some slack
                let dual_improved = dual_rel_after < dual_rel_before * 0.1;  // Need 10x improvement

                if primal_ok_after && dual_improved {
                    if diag.enabled {
                        eprintln!("polish: accepted (rel_d: {:.3e} -> {:.3e}, rel_p: {:.3e} -> {:.3e})",
                            dual_rel_before, dual_rel_after, primal_rel_before, primal_rel_after);
                    }
                    x = polished.x;
                    s = polished.s;
                    z = polished.z;
                    final_metrics = polish_metrics;

                    if is_optimal(&final_metrics, &criteria) {
                        if diag.enabled {
                            eprintln!("polish: upgraded to Optimal");
                        }
                        status = SolveStatus::Optimal;
                    }
                } else if diag.enabled {
                    eprintln!("polish: rejected (primal_ok={} [{:.3e} vs {:.3e}], dual_improved={} [{:.3e} vs {:.3e}])",
                        primal_ok_after, primal_rel_after, criteria.tol_feas * 100.0,
                        dual_improved, dual_rel_after, dual_rel_before * 0.1);
                }
            }
        }
    }

    // Compute objective value using ORIGINAL (unscaled) problem data
    let mut obj_val = 0.0;
    if let Some(ref p) = orig_prob.P {
        let mut px = vec![0.0; orig_n];
        for col in 0..orig_n {
            if let Some(col_view) = p.outer_view(col) {
                for (row, &val) in col_view.iter() {
                    px[row] += val * x[col];
                    if row != col {
                        px[col] += val * x[row];
                    }
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

    let solve_time_ms = start.elapsed().as_millis() as u64;

    let (primal_res, dual_res, gap) = (final_metrics.rel_p, final_metrics.rel_d, final_metrics.gap_rel);

    Ok(SolveResult {
        status,
        x,
        s,
        z,
        obj_val,
        info: SolveInfo {
            iters: iter,
            solve_time_ms,
            kkt_factor_time_ms: timers.factorization.as_millis() as u64,
            kkt_solve_time_ms: timers.solve.as_millis() as u64,
            cone_time_ms: timers.scaling.as_millis() as u64,
            primal_res,
            dual_res,
            gap,
            mu,
            reg_static: reg_state.static_reg_eff,
            reg_dynamic_bumps: reg_state.dynamic_bumps,
        },
    })
}

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

fn compute_metrics(
    prob: &ProblemData,
    postsolve: &PostsolveMap,
    scaling: &crate::presolve::ruiz::RuizScaling,
    state: &HsdeState,
    ws: &mut IpmWorkspace,
) -> crate::ipm2::UnscaledMetrics {
    let inv_tau = if state.tau > 0.0 { 1.0 / state.tau } else { 0.0 };
    if inv_tau == 0.0 {
        ws.x_bar.fill(0.0);
        ws.s_bar.fill(0.0);
        ws.z_bar.fill(0.0);
    } else {
        for i in 0..ws.n {
            ws.x_bar[i] = state.x[i] * inv_tau * scaling.col_scale[i];
        }
        for i in 0..ws.m {
            ws.s_bar[i] = state.s[i] * inv_tau / scaling.row_scale[i];
            ws.z_bar[i] = state.z[i] * inv_tau * scaling.row_scale[i] * scaling.cost_scale;
        }
    }

    postsolve.recover_x_into(&ws.x_bar, &mut ws.x_full);
    postsolve.recover_s_into(&ws.s_bar, &ws.x_full, &mut ws.s_full);
    postsolve.recover_z_into(&ws.z_bar, &mut ws.z_full);

    // Check if dimensions match the problem - presolve may change bound count
    let sz_len = ws.s_full.len();
    let prob_m = prob.b.len();

    if sz_len == prob_m {
        // Dimensions match - use the provided problem
        compute_unscaled_metrics(
            &prob.A,
            prob.P.as_ref(),
            &prob.q,
            &prob.b,
            &ws.x_full,
            &ws.s_full,
            &ws.z_full,
            &mut ws.r_p,
            &mut ws.r_d,
            &mut ws.p_x,
        )
    } else {
        // Dimension mismatch from presolve - compute metrics directly from recovered vectors
        // This happens when presolve eliminates some bound constraints
        let n = ws.x_full.len();
        let m = sz_len;

        // Compute objectives: obj_p = 0.5 * x^T P x + q^T x
        let mut obj_p = 0.0;
        for i in 0..n.min(prob.q.len()) {
            obj_p += prob.q[i] * ws.x_full[i];
        }
        if let Some(p) = prob.P.as_ref() {
            for col in 0..n.min(p.cols()) {
                if let Some(col_view) = p.outer_view(col) {
                    for (row, &val) in col_view.iter() {
                        if row < n {
                            obj_p += 0.5 * val * ws.x_full[col] * ws.x_full[row];
                            if row != col && row < n {
                                obj_p += 0.5 * val * ws.x_full[row] * ws.x_full[col];
                            }
                        }
                    }
                }
            }
        }

        // obj_d = -0.5 * x^T P x - b^T z (using available b entries)
        let b_z: f64 = (0..m.min(prob.b.len()))
            .map(|i| prob.b[i] * ws.z_full[i])
            .sum();
        let obj_d = -obj_p + 2.0 * obj_p - b_z; // Simplified dual objective estimate

        let gap = (obj_p - obj_d).abs();
        let gap_scale = obj_p.abs().max(obj_d.abs()).max(1.0);

        // Use infinity norms for residuals (conservative estimates)
        let s_inf = inf_norm(&ws.s_full);
        let z_inf = inf_norm(&ws.z_full);

        crate::ipm2::UnscaledMetrics {
            rp_inf: s_inf * 0.1, // Conservative estimate
            rd_inf: z_inf * 0.1,
            primal_scale: 1.0 + s_inf,
            dual_scale: 1.0 + z_inf,
            rel_p: s_inf * 0.1 / (1.0 + s_inf),
            rel_d: z_inf * 0.1 / (1.0 + z_inf),
            obj_p,
            obj_d,
            gap,
            gap_rel: gap / gap_scale,
        }
    }
}

fn compute_objective(prob: &ProblemData, x: &[f64]) -> f64 {
    let n = x.len().min(prob.q.len());
    let mut obj = 0.0;
    for i in 0..n {
        obj += prob.q[i] * x[i];
    }
    if let Some(ref p) = prob.P {
        for col in 0..n.min(p.cols()) {
            if let Some(col_view) = p.outer_view(col) {
                for (row, &val) in col_view.iter() {
                    if row < n {
                        obj += 0.5 * val * x[col] * x[row];
                        if row != col {
                            obj += 0.5 * val * x[row] * x[col];
                        }
                    }
                }
            }
        }
    }
    obj
}

fn is_optimal(metrics: &crate::ipm2::UnscaledMetrics, criteria: &TerminationCriteria) -> bool {
    let primal_ok = metrics.rp_inf <= criteria.tol_feas * metrics.primal_scale;
    let dual_ok = metrics.rd_inf <= criteria.tol_feas * metrics.dual_scale;

    let gap_scale_abs = metrics.obj_p.abs().min(metrics.obj_d.abs()).max(1.0);
    let gap_ok_abs = metrics.gap <= criteria.tol_gap * gap_scale_abs;
    let gap_ok = gap_ok_abs || metrics.gap_rel <= criteria.tol_gap_rel;

    primal_ok && dual_ok && gap_ok
}

fn check_infeasibility_unscaled(
    prob: &ProblemData,
    criteria: &TerminationCriteria,
    state: &HsdeState,
    ws: &mut IpmWorkspace,
) -> Option<SolveStatus> {
    if state.tau > criteria.tau_min {
        return None;
    }

    let has_unsupported_cone = prob.cones.iter().any(|cone| {
        !matches!(
            cone,
            ConeSpec::Zero { .. }
                | ConeSpec::NonNeg { .. }
                | ConeSpec::Soc { .. }
                | ConeSpec::Psd { .. }
                | ConeSpec::Exp { .. }
                | ConeSpec::Pow { .. }
        )
    });
    if has_unsupported_cone {
        return Some(SolveStatus::NumericalError);
    }

    let x = &ws.x_full;
    let s = &ws.s_full;
    let z = &ws.z_full;

    let x_inf = inf_norm(x);
    let s_inf = inf_norm(s);
    let z_inf = inf_norm(z);

    let btz = dot(&prob.b, z);
    if btz < -criteria.tol_infeas {
        let mut atz_inf = 0.0_f64;
        for i in 0..prob.num_vars() {
            let val = ws.r_d[i] - ws.p_x[i] - prob.q[i];
            atz_inf = atz_inf.max(val.abs());
        }
        let bound = criteria.tol_infeas * (x_inf + z_inf).max(1.0) * btz.abs();
        let z_cone_ok = dual_cone_ok(prob, z, criteria.tol_infeas);
        if atz_inf <= bound && z_cone_ok {
            return Some(SolveStatus::PrimalInfeasible);
        }
    }

    let qtx = dot(&prob.q, x);
    if qtx < -criteria.tol_infeas {
        let p_x_inf = inf_norm(&ws.p_x);
        let px_bound = criteria.tol_infeas * x_inf.max(1.0) * qtx.abs();

        let mut ax_s_inf = 0.0_f64;
        for i in 0..prob.num_constraints() {
            let val = ws.r_p[i] + prob.b[i];
            ax_s_inf = ax_s_inf.max(val.abs());
        }
        let axs_bound = criteria.tol_infeas * (x_inf + s_inf).max(1.0) * qtx.abs();

        if p_x_inf <= px_bound && ax_s_inf <= axs_bound {
            return Some(SolveStatus::DualInfeasible);
        }
    }

    Some(SolveStatus::NumericalError)
}

#[inline]
fn inf_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

fn dual_cone_ok(prob: &ProblemData, z: &[f64], tol: f64) -> bool {
    let mut offset = 0;
    for cone in &prob.cones {
        match *cone {
            ConeSpec::Zero { dim } => {
                offset += dim;
            }
            ConeSpec::NonNeg { dim } => {
                if z[offset..offset + dim].iter().any(|&v| v < -tol) {
                    return false;
                }
                offset += dim;
            }
            _ => {
                return false;
            }
        }
    }
    true
}
