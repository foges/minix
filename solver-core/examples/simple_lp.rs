//! Simple LP example demonstrating the Minix solver.
//!
//! Solves:
//!   minimize    x1 + x2
//!   subject to  x1 + x2 = 1
//!               x1, x2 >= 0
//!
//! Optimal solution: x1 = 0.5, x2 = 0.5, objective = 1.0

use solver_core::linalg::sparse;
use solver_core::{solve, ConeSpec, ProblemData, SolverSettings};

fn main() {
    println!("Minix Solver - Simple LP Example");
    println!("=================================");
    println!("NOTE: This is a work-in-progress implementation.");
    println!("The simplified predictor-corrector may not fully converge.");
    println!();

    // Problem: min x1 + x2 s.t. x1 + x2 = 1, x1 >= 0, x2 >= 0
    //
    // In standard form:
    //   minimize q^T x
    //   subject to A x + s = b, s ∈ K
    //
    // Variables: x = [x1, x2] (n=2)
    // Constraints (m=3):
    //   1. x1 + x2 + s1 = 1, s1 ∈ {0} (equality)
    //   2. -x1 + s2 = 0, s2 >= 0 (x1 >= 0)
    //   3. -x2 + s3 = 0, s3 >= 0 (x2 >= 0)
    //
    // A = [ 1   1]    b = [1]    cones: Zero(1), NonNeg(2)
    //     [-1   0]        [0]
    //     [ 0  -1]        [0]

    let prob = ProblemData {
        P: None,           // No quadratic term (LP)
        q: vec![1.0, 1.0], // Objective: x1 + x2
        A: sparse::from_triplets(
            3,
            2,
            vec![
                (0, 0, 1.0),
                (0, 1, 1.0),  // Row 0: x1 + x2
                (1, 0, -1.0), // Row 1: -x1
                (2, 1, -1.0), // Row 2: -x2
            ],
        ),
        b: vec![1.0, 0.0, 0.0],
        cones: vec![
            ConeSpec::Zero { dim: 1 },   // s1 = 0 (equality constraint)
            ConeSpec::NonNeg { dim: 2 }, // s2, s3 >= 0 (variable bounds)
        ],
        var_bounds: None,
        integrality: None,
    };

    // Solver settings
    let settings = SolverSettings {
        verbose: true,
        max_iter: 100, // Converges in ~91 iterations with default tolerances
        tol_feas: 1e-7,
        tol_gap: 1e-7,
        ..Default::default()
    };

    // Solve
    match solve(&prob, &settings) {
        Ok(result) => {
            println!("\n=== Solution ===");
            println!("Status: {:?}", result.status);
            println!("x1 = {:.6}", result.x[0]);
            println!("x2 = {:.6}", result.x[1]);
            println!("s  = {:?}", result.s);
            println!("z  = {:?}", result.z);
            println!("Objective value: {:.6}", result.obj_val);
            println!("Iterations: {}", result.info.iters);

            // Verify constraint
            let sum = result.x[0] + result.x[1];
            println!(
                "\nConstraint verification: x1 + x2 = {:.6} (should be 1.0)",
                sum
            );

            // Compute gap
            let qtx = result.x[0] + result.x[1]; // q = [1, 1]
            let btz = result.z[0]; // b = [1, 0, 0]
            println!(
                "Gap: q'x + b'z = {:.6} + {:.6} = {:.6}",
                qtx,
                btz,
                qtx + btz
            );
        }
        Err(e) => {
            eprintln!("Solver failed: {}", e);
            std::process::exit(1);
        }
    }
}
