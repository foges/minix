//! Comprehensive diagnostics for SOC, SDP, and Exp cone problems
//! Shows iteration 25-30 details for problematic cases

use solver_core::{solve, ConeSpec, ProblemData, SolverSettings, SolveStatus};
use solver_core::linalg::sparse;

fn exp_cone_trivial() -> ProblemData {
    // min x
    // s.t. s = [-x, 1, 1] ∈ K_exp
    let q = vec![1.0];
    let triplets = vec![(0, 0, -1.0)];
    let a = sparse::from_triplets(3, 1, triplets);
    let b = vec![0.0, 1.0, 1.0];
    ProblemData {
        P: None,
        q,
        A: a,
        b,
        cones: vec![ConeSpec::Exp { count: 1 }],
        var_bounds: None,
        integrality: None,
    }
}

fn soc_problem() -> ProblemData {
    // min t
    // s.t. ||(x1, x2)|| <= t, t >= 1
    let q = vec![1.0, 0.0, 0.0];  // min t
    let a = sparse::from_triplets(
        4,
        3,
        vec![
            (0, 0, -1.0), // -t + s1 = -1
            (1, 0, 1.0),  // t + s2 = 0 (SOC first)
            (2, 1, 1.0),  // x1 + s3 = 0
            (3, 2, 1.0),  // x2 + s4 = 0
        ],
    );
    let b = vec![-1.0, 0.0, 0.0, 0.0];
    ProblemData {
        P: None,
        q,
        A: a,
        b,
        cones: vec![ConeSpec::NonNeg { dim: 1 }, ConeSpec::Soc { dim: 3 }],
        var_bounds: None,
        integrality: None,
    }
}

fn main() {
    println!("\n");
    println!("================================================================================");
    println!("COMPREHENSIVE CONE DIAGNOSTICS - ITERATION 25-30 DETAILS");
    println!("================================================================================");
    println!();

    // Test Exp Cone
    println!("################################################################################");
    println!("## EXP CONE: Trivial Problem");
    println!("################################################################################");
    println!("Problem: min x s.t. s = [-x, 1, 1] ∈ K_exp");
    println!("Expected: x = 0, obj = 0");
    println!();

    let prob_exp = exp_cone_trivial();
    let settings_exp = SolverSettings {
        verbose: true,
        max_iter: 35,
        tol_feas: 1e-8,
        tol_gap: 1e-8,
        ..Default::default()
    };

    match solve(&prob_exp, &settings_exp) {
        Ok(sol) => {
            println!("\n--- FINAL RESULT ---");
            println!("Status: {:?}", sol.status);
            println!("Iterations: {}", sol.info.iters);
            println!("Objective: {:.10e}", sol.obj_val);
            println!("x = {:?}", sol.x);
            println!("Residuals:");
            println!("  primal_res: {:.6e}", sol.info.primal_res);
            println!("  dual_res:   {:.6e}", sol.info.dual_res);
            println!("  gap:        {:.6e}", sol.info.gap);
            println!("  mu:         {:.6e}", sol.info.mu);
        },
        Err(e) => println!("ERROR: {:?}", e),
    }

    println!();
    println!();

    // Test SOC
    println!("################################################################################");
    println!("## SOC: Simple SOCP Problem");
    println!("################################################################################");
    println!("Problem: min t s.t. ||(x1, x2)|| <= t, t >= 1");
    println!("Expected: t = 1, x1 = x2 = 0, obj = 1");
    println!();

    let prob_soc = soc_problem();
    let settings_soc = SolverSettings {
        verbose: true,
        max_iter: 35,
        tol_feas: 1e-8,
        tol_gap: 1e-8,
        ..Default::default()
    };

    match solve(&prob_soc, &settings_soc) {
        Ok(sol) => {
            println!("\n--- FINAL RESULT ---");
            println!("Status: {:?}", sol.status);
            println!("Iterations: {}", sol.info.iters);
            println!("Objective: {:.10e}", sol.obj_val);
            println!("x = {:?}", sol.x);
            println!("Residuals:");
            println!("  primal_res: {:.6e}", sol.info.primal_res);
            println!("  dual_res:   {:.6e}", sol.info.dual_res);
            println!("  gap:        {:.6e}", sol.info.gap);
            println!("  mu:         {:.6e}", sol.info.mu);
        },
        Err(e) => println!("ERROR: {:?}", e),
    }

    println!();
    println!();
    println!("================================================================================");
    println!("END OF DIAGNOSTICS");
    println!("================================================================================");
}
