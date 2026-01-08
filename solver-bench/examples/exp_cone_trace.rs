//! Trace exp cone solver iterations to see where it gets stuck

use solver_core::{ConeSpec, ProblemData, SolverSettings, solve};
use solver_core::linalg::sparse;

fn trivial() -> ProblemData {
    // min x
    // s.t. s = [-x, 1, 1] ∈ K_exp
    //
    // Optimal: x = 0 (since s = [0, 1, 1] satisfies 1*exp(0/1) = 1 ≤ 1)
    let num_vars = 1;
    let q = vec![1.0];
    let triplets = vec![(0, 0, -1.0)];
    let a = sparse::from_triplets(3, num_vars, triplets);
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

fn main() {
    println!("\n=== Exp Cone Trivial Problem ===");
    println!("min x s.t. s = [-x, 1, 1] ∈ K_exp");
    println!("Expected: x = 0, obj = 0");
    println!();

    let prob = trivial();
    let settings = SolverSettings {
        verbose: true,  // Enable verbose to see iteration progress
        max_iter: 20,   // Just 20 iterations to see initial behavior
        tol_feas: 1e-8,
        tol_gap: 1e-8,
        ..Default::default()
    };

    match solve(&prob, &settings) {
        Ok(sol) => {
            println!("\n=== Solution ===");
            println!("Status: {:?}", sol.status);
            println!("Iterations: {}", sol.info.iters);
            println!("x = {:?}", sol.x);
            println!("s = {:?}", sol.s);
            println!("z = {:?}", sol.z);
            println!("Objective: {:.6e}", sol.obj_val);
            println!("\nResiduals:");
            println!("  primal_res: {:.6e}", sol.info.primal_res);
            println!("  dual_res:   {:.6e}", sol.info.dual_res);
            println!("  gap:        {:.6e}", sol.info.gap);
            println!("  mu:         {:.6e}", sol.info.mu);
        },
        Err(e) => println!("Error: {:?}", e),
    }
}
