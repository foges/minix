//! Interior point method solver.
//!
//! HSDE formulation, predictor-corrector algorithm, and termination criteria.

pub mod hsde;
pub mod predcorr;
pub mod termination;

use crate::cones::{ConeKernel, ZeroCone, NonNegCone, SocCone};
use crate::linalg::kkt::KktSolver;
use crate::presolve::ruiz::equilibrate;
use crate::problem::{ProblemData, ConeSpec, SolverSettings, SolveResult, SolveStatus, SolveInfo};
use crate::scaling::ScalingBlock;
use hsde::{HsdeState, HsdeResiduals, compute_residuals, compute_mu};
use predcorr::predictor_corrector_step;
use termination::{TerminationCriteria, check_termination};

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

    let n = prob.num_vars();
    let m = prob.num_constraints();

    // Apply Ruiz equilibration for numerical stability
    let (a_scaled, p_scaled, q_scaled, b_scaled, scaling) = equilibrate(
        &prob.A,
        prob.P.as_ref(),
        &prob.q,
        &prob.b,
        settings.ruiz_iters,
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

    // Build cone kernels from cone specs
    let cones = build_cones(&scaled_prob.cones)?;

    // Compute total barrier degree
    let barrier_degree: usize = cones.iter().map(|c| c.barrier_degree()).sum();

    // Initialize HSDE state
    let mut state = HsdeState::new(n, m);
    state.initialize_with_prob(&cones, &scaled_prob);

    // Initialize residuals
    let mut residuals = HsdeResiduals::new(n, m);

    // Initialize KKT solver
    // For LPs (P=None) or very sparse QPs, use higher regularization to stabilize.
    // The (1,1) block is only εI for LPs. With small ε, solving
    //   [εI, A^T] [dx]   [rhs_x]
    //   [A,  -(H)] [dz] = [rhs_z]
    // gives dx ≈ rhs_x/ε, which blows up for small ε.
    // Using ε=1e-4 provides stability while allowing good convergence.
    let p_is_sparse = scaled_prob.P.as_ref().map_or(true, |p| {
        p.nnz() < n / 2  // Less than 50% diagonal fill
    });
    let static_reg = if p_is_sparse {
        settings.static_reg.max(1e-4)  // LP or sparse QP: use at least 1e-4
    } else {
        settings.static_reg.max(1e-6)  // Dense QP: use at least 1e-6
    };

    let mut kkt = KktSolver::new(
        n,
        m,
        static_reg,
        settings.dynamic_reg_min_pivot,
    );

    // Perform symbolic factorization once with initial scaling structure.
    // This determines the sparsity pattern of L and the elimination tree.
    // Subsequent calls to factor() reuse this symbolic factorization.
    let initial_scaling: Vec<ScalingBlock> = cones.iter().map(|cone| {
        let dim = cone.dim();
        if cone.barrier_degree() == 0 {
            ScalingBlock::Zero { dim }
        } else if (cone.as_ref() as &dyn std::any::Any).downcast_ref::<SocCone>().is_some() {
            // SOC creates a dense block in KKT
            ScalingBlock::SocStructured { w: vec![1.0; dim] }
        } else {
            // NonNeg uses diagonal scaling
            ScalingBlock::Diagonal { d: vec![1.0; dim] }
        }
    }).collect();

    if let Err(e) = kkt.initialize(scaled_prob.P.as_ref(), &scaled_prob.A, &initial_scaling) {
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

    let mut status = SolveStatus::NumericalError;  // Will be overwritten
    let mut iter = 0;
    let mut consecutive_failures = 0;
    const MAX_CONSECUTIVE_FAILURES: usize = 3;

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
        println!("{:>4} {:>12} {:>12} {:>12} {:>12} {:>8}",
                 "Iter", "μ", "Primal Res", "Dual Res", "Gap", "Alpha");
        println!("{}", "-".repeat(72));
    }

    // Main IPM loop
    while iter < settings.max_iter {
        // Compute residuals
        compute_residuals(&scaled_prob, &state, &mut residuals);

        // Check termination
        if let Some(term_status) = check_termination(&scaled_prob, &state, &residuals, mu, iter, &criteria) {
            status = term_status;
            break;
        }

        // Take predictor-corrector step
        let step_result = match predictor_corrector_step(
            &mut kkt,
            &scaled_prob,
            &mut state,
            &residuals,
            &cones,
            mu,
            barrier_degree,
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
                let recovery_margin = (mu * 0.1).max(1e-4);
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

        // Verbose output
        if settings.verbose {
            let (rx_norm, rz_norm, _) = residuals.norms();
            let primal_res = rx_norm / state.tau.max(1.0);
            let dual_res = rz_norm / state.tau.max(1.0);

            // Compute gap (on scaled problem)
            let x_bar: Vec<f64> = state.x.iter().map(|xi| xi / state.tau).collect();
            let z_bar: Vec<f64> = state.z.iter().map(|zi| zi / state.tau).collect();

            let qtx: f64 = scaled_prob.q.iter().zip(x_bar.iter()).map(|(qi, xi)| qi * xi).sum();
            let btz: f64 = scaled_prob.b.iter().zip(z_bar.iter()).map(|(bi, zi)| bi * zi).sum();
            let gap = (qtx + btz).abs();

            println!("{:4} {:12.4e} {:12.4e} {:12.4e} {:12.4e} {:8.4}",
                     iter, mu, primal_res, dual_res, gap, step_result.alpha);
        }

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
    let x = scaling.unscale_x(&x_scaled);
    let s = scaling.unscale_s(&s_scaled);
    let z = scaling.unscale_z(&z_scaled);

    // Compute objective value using ORIGINAL (unscaled) problem data
    let mut obj_val = 0.0;
    if let Some(ref p) = prob.P {
        let mut px = vec![0.0; n];
        for col in 0..n {
            if let Some(col_view) = p.outer_view(col) {
                for (row, &val) in col_view.iter() {
                    px[row] += val * x[col];
                    if row != col {
                        px[col] += val * x[row];
                    }
                }
            }
        }
        for i in 0..n {
            obj_val += 0.5 * x[i] * px[i];
        }
    }
    for i in 0..n {
        obj_val += prob.q[i] * x[i];
    }

    Ok(SolveResult {
        status,
        x,
        s,
        z,
        obj_val,
        info: SolveInfo {
            iters: iter,
            solve_time_ms: 0,  // TODO: Add timing
            kkt_factor_time_ms: 0,
            kkt_solve_time_ms: 0,
            cone_time_ms: 0,
            primal_res: 0.0,  // TODO: Record final residuals
            dual_res: 0.0,
            gap: 0.0,
            mu,
            reg_static: static_reg,
            reg_dynamic_bumps: 0,
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
            ConeSpec::Psd { .. } => {
                return Err("PSD cone not yet implemented".into());
            }
            ConeSpec::Exp { .. } => {
                return Err("Exponential cone not yet implemented".into());
            }
            ConeSpec::Pow { .. } => {
                return Err("Power cone not yet implemented".into());
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

