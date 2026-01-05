//! Simplified IPM solver for tall problems using normal equations.
//!
//! This module provides a fast path for LP/QP problems where m >> n
//! and all cones are Zero or NonNeg. In this case, we can solve the
//! normal equations (n×n dense) instead of the full KKT system ((n+m)×(n+m) sparse).

use std::time::Instant;

use crate::cones::{ConeKernel, NonNegCone, ZeroCone};
use crate::ipm::hsde::{HsdeResiduals, HsdeState, compute_residuals};
use crate::ipm::termination::TerminationCriteria;
use crate::ipm2::{
    DiagnosticsConfig, compute_unscaled_metrics,
};
use crate::linalg::normal_eqns::NormalEqnsSolver;
use crate::postsolve::PostsolveMap;
use crate::presolve::ruiz::RuizScaling;
use crate::problem::{
    ConeSpec, ProblemData, SolveInfo, SolveResult, SolveStatus, SolverSettings,
};

/// Simplified predictor-corrector step for normal equations path.
/// Returns (alpha, mu_new) on success.
fn normal_eqns_step(
    solver: &mut NormalEqnsSolver,
    prob: &ProblemData,
    state: &mut HsdeState,
    residuals: &HsdeResiduals,
    mu: f64,
    barrier_degree: usize,
    h_diag: &mut [f64],
    // Workspace vectors (reused across iterations)
    dx_aff: &mut [f64],
    dz_aff: &mut [f64],
    ds_aff: &mut [f64],
    dx: &mut [f64],
    dz: &mut [f64],
    ds: &mut [f64],
) -> Result<(f64, f64), String> {
    let n = prob.num_vars();
    let m = prob.num_constraints();

    // Update NT scaling diagonal (s[i]/z[i] for NonNeg cones)
    let mut offset = 0;
    for cone in &prob.cones {
        match cone {
            ConeSpec::Zero { dim } => {
                for i in 0..*dim {
                    h_diag[offset + i] = 1e20; // Large value for zero cone
                }
                offset += dim;
            }
            ConeSpec::NonNeg { dim } => {
                for i in 0..*dim {
                    let s_i = state.s[offset + i];
                    let z_i = state.z[offset + i];
                    h_diag[offset + i] = if s_i > 1e-14 && z_i > 1e-14 {
                        (s_i / z_i).clamp(1e-18, 1e18)
                    } else {
                        1.0
                    };
                }
                offset += dim;
            }
            _ => return Err("Normal equations only supports Zero and NonNeg cones".to_string()),
        }
    }

    // Factor the normal equations system
    solver.update_and_factor(h_diag)?;

    // Build RHS for affine step
    // rhs_x = -r_x, rhs_z = s - r_z
    let mut rhs_x: Vec<f64> = residuals.r_x.iter().map(|&r| -r).collect();
    let mut rhs_z: Vec<f64> = (0..m).map(|i| state.s[i] - residuals.r_z[i]).collect();

    // Solve affine direction
    solver.solve(h_diag, &rhs_x, &rhs_z, dx_aff, dz_aff);

    // Compute ds_aff = -s - H * dz_aff for NonNeg cones
    offset = 0;
    for cone in &prob.cones {
        match cone {
            ConeSpec::Zero { dim } => {
                for i in 0..*dim {
                    ds_aff[offset + i] = 0.0;
                }
                offset += dim;
            }
            ConeSpec::NonNeg { dim } => {
                for i in 0..*dim {
                    let idx = offset + i;
                    ds_aff[idx] = -state.s[idx] - h_diag[idx] * dz_aff[idx];
                }
                offset += dim;
            }
            _ => unreachable!(),
        }
    }

    // Compute affine step size
    let mut alpha_aff: f64 = 1.0;
    for i in 0..m {
        if ds_aff[i] < 0.0 && state.s[i] > 0.0 {
            alpha_aff = alpha_aff.min(-state.s[i] / ds_aff[i]);
        }
        if dz_aff[i] < 0.0 && state.z[i] > 0.0 {
            alpha_aff = alpha_aff.min(-state.z[i] / dz_aff[i]);
        }
    }
    alpha_aff = (0.99 * alpha_aff).min(1.0);

    // Compute centering parameter σ
    let mut s_dot_z_aff = 0.0;
    offset = 0;
    for cone in &prob.cones {
        let dim = match cone {
            ConeSpec::Zero { dim } | ConeSpec::NonNeg { dim } => *dim,
            _ => unreachable!(),
        };
        if matches!(cone, ConeSpec::NonNeg { .. }) {
            for i in 0..dim {
                let idx = offset + i;
                let s_trial = state.s[idx] + alpha_aff * ds_aff[idx];
                let z_trial = state.z[idx] + alpha_aff * dz_aff[idx];
                s_dot_z_aff += s_trial * z_trial;
            }
        }
        offset += dim;
    }
    let mu_aff = s_dot_z_aff / barrier_degree as f64;
    let sigma = if mu > 1e-14 && mu_aff > 0.0 {
        (mu_aff / mu).powi(3).clamp(1e-3, 0.99)
    } else {
        0.1
    };

    // Build RHS for combined step with Mehrotra correction
    let target_mu = sigma * mu;
    for i in 0..n {
        rhs_x[i] = -residuals.r_x[i];
    }

    offset = 0;
    for cone in &prob.cones {
        match cone {
            ConeSpec::Zero { dim } => {
                for i in 0..*dim {
                    rhs_z[offset + i] = -residuals.r_z[offset + i];
                }
                offset += dim;
            }
            ConeSpec::NonNeg { dim } => {
                for i in 0..*dim {
                    let idx = offset + i;
                    let z_safe = state.z[idx].max(1e-14);
                    // Mehrotra correction: ds_aff * dz_aff / z
                    let correction = ds_aff[idx] * dz_aff[idx] / z_safe;
                    // d_s_comb = (s*z + ds_aff*dz_aff - sigma*mu) / z
                    let d_s_comb = (state.s[idx] * state.z[idx] + ds_aff[idx] * dz_aff[idx] - target_mu) / z_safe;
                    rhs_z[idx] = d_s_comb - residuals.r_z[idx];
                }
                offset += dim;
            }
            _ => unreachable!(),
        }
    }

    // Solve combined direction
    solver.solve(h_diag, &rhs_x, &rhs_z, dx, dz);

    // Compute ds from dz
    offset = 0;
    for cone in &prob.cones {
        match cone {
            ConeSpec::Zero { dim } => {
                for i in 0..*dim {
                    ds[offset + i] = 0.0;
                }
                offset += dim;
            }
            ConeSpec::NonNeg { dim } => {
                for i in 0..*dim {
                    let idx = offset + i;
                    let z_safe = state.z[idx].max(1e-14);
                    let d_s_comb = (state.s[idx] * state.z[idx] + ds_aff[idx] * dz_aff[idx] - target_mu) / z_safe;
                    ds[idx] = -d_s_comb - h_diag[idx] * dz[idx];
                }
                offset += dim;
            }
            _ => unreachable!(),
        }
    }

    // Compute step size
    let mut alpha: f64 = 1.0;
    for i in 0..m {
        if ds[i] < 0.0 && state.s[i] > 0.0 {
            alpha = alpha.min(-state.s[i] / ds[i]);
        }
        if dz[i] < 0.0 && state.z[i] > 0.0 {
            alpha = alpha.min(-state.z[i] / dz[i]);
        }
    }
    alpha = (0.99 * alpha).min(1.0);

    // Update state
    for i in 0..n {
        state.x[i] += alpha * dx[i];
    }
    offset = 0;
    for cone in &prob.cones {
        let dim = match cone {
            ConeSpec::Zero { dim } | ConeSpec::NonNeg { dim } => *dim,
            _ => unreachable!(),
        };
        for i in 0..dim {
            let idx = offset + i;
            if matches!(cone, ConeSpec::NonNeg { .. }) {
                state.s[idx] += alpha * ds[idx];
            }
            state.z[idx] += alpha * dz[idx];
        }
        offset += dim;
    }

    // Ensure positivity
    for i in 0..m {
        if state.s[i] < 1e-14 {
            state.s[i] = 1e-14;
        }
        if state.z[i] < 1e-14 {
            state.z[i] = 1e-14;
        }
    }

    // Compute new mu
    let mut s_dot_z = 0.0;
    offset = 0;
    for cone in &prob.cones {
        let dim = match cone {
            ConeSpec::Zero { dim } | ConeSpec::NonNeg { dim } => *dim,
            _ => unreachable!(),
        };
        if matches!(cone, ConeSpec::NonNeg { .. }) {
            for i in 0..dim {
                s_dot_z += state.s[offset + i] * state.z[offset + i];
            }
        }
        offset += dim;
    }
    let mu_new = s_dot_z / barrier_degree as f64;

    Ok((alpha, mu_new))
}

/// Solve using normal equations for tall problems.
/// This is a simplified IPM that works when m >> n and only Zero/NonNeg cones are present.
pub fn solve_normal_equations(
    prob: &ProblemData,
    scaled_prob: &ProblemData,
    settings: &SolverSettings,
    postsolve: &PostsolveMap,
    scaling: &RuizScaling,
    orig_prob_bounds: &ProblemData,
) -> Result<SolveResult, Box<dyn std::error::Error>> {
    let diag = DiagnosticsConfig::from_env();
    let n = scaled_prob.num_vars();
    let m = scaled_prob.num_constraints();

    // Create normal equations solver
    let mut solver = NormalEqnsSolver::new(
        n, m,
        scaled_prob.P.as_ref(),
        &scaled_prob.A,
        settings.static_reg,
    );

    // Build cone kernels and compute barrier degree
    let mut cones: Vec<Box<dyn ConeKernel>> = Vec::new();
    let mut barrier_degree = 0usize;
    for spec in &scaled_prob.cones {
        match spec {
            ConeSpec::Zero { dim } => {
                cones.push(Box::new(ZeroCone::new(*dim)));
            }
            ConeSpec::NonNeg { dim } => {
                cones.push(Box::new(NonNegCone::new(*dim)));
                barrier_degree += dim;
            }
            _ => return Err("Normal equations only supports Zero and NonNeg cones".into()),
        }
    }

    // Initialize state
    let mut state = HsdeState::new(n, m);
    state.initialize_with_prob(&cones, scaled_prob);

    // Workspace
    let mut h_diag = vec![1.0; m];
    let mut dx_aff = vec![0.0; n];
    let mut dz_aff = vec![0.0; m];
    let mut ds_aff = vec![0.0; m];
    let mut dx = vec![0.0; n];
    let mut dz = vec![0.0; m];
    let mut ds = vec![0.0; m];
    let mut residuals = HsdeResiduals::new(n, m);

    let criteria = TerminationCriteria {
        tol_feas: settings.tol_feas,
        tol_gap: settings.tol_gap,
        tol_infeas: settings.tol_infeas,
        max_iter: settings.max_iter,
        ..Default::default()
    };

    // Compute initial mu
    let mut mu = {
        let mut s_dot_z = 0.0;
        let mut offset = 0;
        for cone in &scaled_prob.cones {
            let dim = match cone {
                ConeSpec::Zero { dim } | ConeSpec::NonNeg { dim } => *dim,
                _ => unreachable!(),
            };
            if matches!(cone, ConeSpec::NonNeg { .. }) {
                for i in 0..dim {
                    s_dot_z += state.s[offset + i] * state.z[offset + i];
                }
            }
            offset += dim;
        }
        s_dot_z / barrier_degree as f64
    };

    let start = Instant::now();
    let mut iter = 0;
    let mut status = SolveStatus::MaxIters;

    while iter < settings.max_iter {
        // Compute residuals
        compute_residuals(scaled_prob, &state, &mut residuals);

        // Compute unscaled metrics for termination check
        let x_unscaled = scaling.unscale_x(&state.x);
        let s_unscaled = scaling.unscale_s(&state.s);
        let z_unscaled = scaling.unscale_z(&state.z);
        let x_full = postsolve.recover_x(&x_unscaled);
        let s_full = postsolve.recover_s(&s_unscaled, &x_full);
        let z_full = postsolve.recover_z(&z_unscaled);

        let mut rp = vec![0.0; orig_prob_bounds.num_constraints()];
        let mut rd = vec![0.0; orig_prob_bounds.num_vars()];
        let mut px = vec![0.0; orig_prob_bounds.num_vars()];
        let metrics = compute_unscaled_metrics(
            &orig_prob_bounds.A,
            orig_prob_bounds.P.as_ref(),
            &orig_prob_bounds.q,
            &orig_prob_bounds.b,
            &x_full,
            &s_full,
            &z_full,
            &mut rp,
            &mut rd,
            &mut px,
        );

        if diag.should_log(iter) {
            eprintln!(
                "iter {:4} mu={:.3e} rel_p={:.3e} rel_d={:.3e} gap_rel={:.3e}",
                iter, mu, metrics.rel_p, metrics.rel_d, metrics.gap_rel
            );
        }

        // Check termination
        let primal_ok = metrics.rel_p <= criteria.tol_feas;
        let dual_ok = metrics.rel_d <= criteria.tol_feas;
        let gap_ok = metrics.gap_rel <= criteria.tol_gap_rel;

        if primal_ok && dual_ok && gap_ok {
            status = SolveStatus::Optimal;
            break;
        }

        // Take a step
        let step_result = normal_eqns_step(
            &mut solver,
            scaled_prob,
            &mut state,
            &residuals,
            mu,
            barrier_degree,
            &mut h_diag,
            &mut dx_aff,
            &mut dz_aff,
            &mut ds_aff,
            &mut dx,
            &mut dz,
            &mut ds,
        );

        match step_result {
            Ok((alpha, mu_new)) => {
                if diag.should_log(iter) {
                    eprintln!("  alpha={:.3e} mu_new={:.3e}", alpha, mu_new);
                }
                mu = mu_new;
            }
            Err(e) => {
                eprintln!("Normal equations step failed: {}", e);
                status = SolveStatus::NumericalError;
                break;
            }
        }

        iter += 1;
    }

    // Extract final solution
    let x_unscaled = scaling.unscale_x(&state.x);
    let s_unscaled = scaling.unscale_s(&state.s);
    let z_unscaled = scaling.unscale_z(&state.z);
    let x = postsolve.recover_x(&x_unscaled);
    let s = postsolve.recover_s(&s_unscaled, &x);
    let z = postsolve.recover_z(&z_unscaled);

    // Compute final metrics
    let mut rp = vec![0.0; orig_prob_bounds.num_constraints()];
    let mut rd = vec![0.0; orig_prob_bounds.num_vars()];
    let mut px = vec![0.0; orig_prob_bounds.num_vars()];
    let final_metrics = compute_unscaled_metrics(
        &orig_prob_bounds.A,
        orig_prob_bounds.P.as_ref(),
        &orig_prob_bounds.q,
        &orig_prob_bounds.b,
        &x,
        &s,
        &z,
        &mut rp,
        &mut rd,
        &mut px,
    );

    // Compute objective
    let mut obj_val = 0.0;
    if let Some(ref p) = prob.P {
        let mut px = vec![0.0; prob.num_vars()];
        for col in 0..prob.num_vars() {
            if let Some(col_view) = p.outer_view(col) {
                for (row, &val) in col_view.iter() {
                    px[row] += val * x[col];
                    if row != col {
                        px[col] += val * x[row];
                    }
                }
            }
        }
        for i in 0..prob.num_vars() {
            obj_val += 0.5 * x[i] * px[i];
        }
    }
    for i in 0..prob.num_vars() {
        obj_val += prob.q[i] * x[i];
    }

    let solve_time_ms = start.elapsed().as_millis() as u64;

    Ok(SolveResult {
        status,
        x,
        s,
        z,
        obj_val,
        info: SolveInfo {
            iters: iter,
            solve_time_ms,
            kkt_factor_time_ms: 0, // Not tracked separately for normal eqns
            kkt_solve_time_ms: 0,
            cone_time_ms: 0,
            primal_res: final_metrics.rel_p,
            dual_res: final_metrics.rel_d,
            gap: final_metrics.gap_rel,
            mu,
            reg_static: settings.static_reg,
            reg_dynamic_bumps: 0,
        },
    })
}
