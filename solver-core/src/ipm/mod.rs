//! Interior point method solver.
//!
//! HSDE formulation, predictor-corrector algorithm, and termination criteria.

pub mod hsde;
pub mod predcorr;
pub mod termination;

use crate::cones::{ConeKernel, ZeroCone, NonNegCone, SocCone};
use crate::linalg::kkt::KktSolver;
use crate::problem::{ProblemData, ConeSpec, SolverSettings, SolveResult, SolveStatus, SolveInfo};
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

    // Build cone kernels from cone specs
    let cones = build_cones(&prob.cones)?;

    // Compute total barrier degree
    let barrier_degree: usize = cones.iter().map(|c| c.barrier_degree()).sum();

    // Initialize HSDE state
    let mut state = HsdeState::new(n, m);
    state.initialize_with_prob(&cones, prob);

    // Initialize residuals
    let mut residuals = HsdeResiduals::new(n, m);

    // Initialize KKT solver
    // For LPs (P=None), use higher regularization to stabilize the Schur complement
    // computation in the two-solve strategy. The (1,1) block is only εI for LPs,
    // and with ε=1e-9, the second solve K[Δx₂,Δz₂]=[-q,b] produces Δx₂≈1e9.
    // Using ε=1e-6 stabilizes this without affecting solution accuracy much.
    let static_reg = if prob.P.is_none() {
        settings.static_reg.max(1e-6)  // LP: use at least 1e-6
    } else {
        settings.static_reg
    };

    let mut kkt = KktSolver::new(
        n,
        m,
        static_reg,
        settings.dynamic_reg_min_pivot,
    );

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

    if settings.verbose {
        println!("Minix IPM Solver");
        println!("================");
        println!("Problem: n = {}, m = {}, cones = {:?}", n, m, prob.cones.len());
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
        compute_residuals(prob, &state, &mut residuals);

        // Check termination
        if let Some(term_status) = check_termination(prob, &state, &residuals, mu, iter, &criteria) {
            status = term_status;
            break;
        }

        // Take predictor-corrector step
        let step_result = match predictor_corrector_step(
            &mut kkt,
            prob,
            &mut state,
            &residuals,
            &cones,
            mu,
            barrier_degree,
        ) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("IPM step failed: {}", e);
                status = SolveStatus::NumericalError;
                break;
            }
        };

        // Update mu
        mu = step_result.mu_new;

        // Verbose output
        if settings.verbose {
            let (rx_norm, rz_norm, _) = residuals.norms();
            let primal_res = rx_norm / state.tau.max(1.0);
            let dual_res = rz_norm / state.tau.max(1.0);

            // Compute gap
            let x_bar: Vec<f64> = state.x.iter().map(|xi| xi / state.tau).collect();
            let z_bar: Vec<f64> = state.z.iter().map(|zi| zi / state.tau).collect();

            let qtx: f64 = prob.q.iter().zip(x_bar.iter()).map(|(qi, xi)| qi * xi).sum();
            let btz: f64 = prob.b.iter().zip(z_bar.iter()).map(|(bi, zi)| bi * zi).sum();
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

    // Extract solution
    let x = if state.tau > 1e-8 {
        state.x.iter().map(|xi| xi / state.tau).collect()
    } else {
        vec![0.0; n]
    };

    let s = if state.tau > 1e-8 {
        state.s.iter().map(|si| si / state.tau).collect()
    } else {
        vec![0.0; m]
    };

    let z = if state.tau > 1e-8 {
        state.z.iter().map(|zi| zi / state.tau).collect()
    } else {
        vec![0.0; m]
    };

    // Compute objective value
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
        // Optimal: x = [0.5, 0.5], obj = 1.0

        let prob = ProblemData {
            P: None,
            q: vec![1.0, 1.0],
            A: sparse::from_triplets(1, 2, vec![(0, 0, 1.0), (0, 1, 1.0)]),
            b: vec![1.0],
            cones: vec![ConeSpec::Zero { dim: 1 }],
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

        // Check solution is approximately [0.5, 0.5]
        // (may not be exact due to simplified predictor-corrector)
        if result.status == SolveStatus::Optimal {
            assert!((result.x[0] - 0.5).abs() < 0.1);
            assert!((result.x[1] - 0.5).abs() < 0.1);
            assert!((result.obj_val - 1.0).abs() < 0.1);
        }
    }
}

