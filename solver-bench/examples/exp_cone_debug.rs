//! Debug exp cone multi-problem issues

use solver_core::{ConeSpec, ProblemData, SolverSettings, solve};
use solver_core::linalg::sparse;

fn trivial() -> ProblemData {
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

fn trivial_multi(n: usize) -> ProblemData {
    let num_vars = n;
    let q = vec![1.0; num_vars];

    // Each exp cone has 3 rows, constraint: s_i = [-x_i, 1, 1] ∈ K_exp
    let mut triplets = Vec::new();
    for i in 0..n {
        let row_base = 3 * i;
        triplets.push((row_base, i, -1.0));
    }

    let num_rows = 3 * n;
    let a = sparse::from_triplets(num_rows, num_vars, triplets);

    let mut b = Vec::new();
    for _ in 0..n {
        b.push(0.0);  // s[3i] = 0
        b.push(1.0);  // s[3i+1] = 1
        b.push(1.0);  // s[3i+2] = 1
    }

    ProblemData {
        P: None,
        q,
        A: a,
        b,
        cones: vec![ConeSpec::Exp { count: n }],
        var_bounds: None,
        integrality: None,
    }
}

fn main() {
    println!("\n=== Testing Trivial Problem (max_iter=1000, tol=1e-8) ===");
    let prob1 = trivial();
    let mut settings1 = SolverSettings::default();
    settings1.max_iter = 1000;
    settings1.verbose = false;

    match solve(&prob1, &settings1) {
        Ok(sol) => {
            println!("Status: {:?}, Iters: {}, Obj: {:.4e}", sol.status, sol.info.iters, sol.obj_val);
            println!("primal_res: {:.4e}, dual_res: {:.4e}, gap: {:.4e}, mu: {:.4e}",
                     sol.info.primal_res, sol.info.dual_res, sol.info.gap, sol.info.mu);
        },
        Err(e) => println!("Error: {:?}", e),
    }

    println!("\n=== Testing Trivial-Multi-2 (max_iter=1000, tol=1e-8) ===");
    let prob2 = trivial_multi(2);
    let mut settings2 = SolverSettings::default();
    settings2.max_iter = 1000;
    settings2.verbose = false;

    match solve(&prob2, &settings2) {
        Ok(sol) => {
            println!("Status: {:?}, Iters: {}, Obj: {:.4e}", sol.status, sol.info.iters, sol.obj_val);
            println!("primal_res: {:.4e}, dual_res: {:.4e}, gap: {:.4e}, mu: {:.4e}",
                     sol.info.primal_res, sol.info.dual_res, sol.info.gap, sol.info.mu);
        },
        Err(e) => println!("Error: {:?}", e),
    }

}
