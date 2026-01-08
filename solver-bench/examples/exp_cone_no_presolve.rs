//! Test exp cone WITHOUT presolve

use solver_core::{solve, ConeSpec, ProblemData, SolverSettings};
use solver_core::linalg::sparse;

fn trivial() -> ProblemData {
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

fn main() {
    println!("\n=== Exp Cone WITHOUT Presolve ===\n");

    let prob = trivial();
    let settings = SolverSettings {
        verbose: true,
        max_iter: 50,
        tol_feas: 1e-8,
        tol_gap: 1e-8,
        ruiz_iters: 0,  // Disable Ruiz
        ..Default::default()
    };

    match solve(&prob, &settings) {
        Ok(sol) => {
            println!("Status: {:?}", sol.status);
            println!("Iterations: {}", sol.info.iters);
            println!("x = {:?}", sol.x);
            println!("s = {:?}", sol.s);
            println!("z = {:?}", sol.z);
        },
        Err(e) => println!("ERROR: {:?}", e),
    }
}
