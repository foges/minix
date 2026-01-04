use solver_core::{solve, ConeSpec, ProblemData, SolverSettings};
use sprs::CsMat;

fn main() {
    println!("=== Testing solver-core on simple LPs ===\n");

    // Test 1: Simple LP relaxation of binary problem
    // min -x0 - x1
    // s.t. x0 + x1 <= 1 (NonNeg cone)
    //      0 <= x0 <= 1
    //      0 <= x1 <= 1
    println!("--- Test 1: Simple LP with bounds ---");

    // Formulate with bounds as separate constraints:
    // Row 0: x0 + x1 + s0 = 1 (s0 >= 0 gives x0 + x1 <= 1)
    // Row 1: -x0 + s1 = 0 (s1 >= 0 gives x0 >= 0)
    // Row 2: -x1 + s2 = 0 (s2 >= 0 gives x1 >= 0)
    // Row 3: x0 + s3 = 1 (s3 >= 0 gives x0 <= 1)
    // Row 4: x1 + s4 = 1 (s4 >= 0 gives x1 <= 1)

    let a = CsMat::new_csc(
        (5, 2),
        vec![0, 3, 6],                        // col pointers
        vec![0, 1, 3, 0, 2, 4],               // row indices
        vec![1.0, -1.0, 1.0, 1.0, -1.0, 1.0], // values
    );

    let prob = ProblemData {
        P: None,
        q: vec![-1.0, -1.0],
        A: a,
        b: vec![1.0, 0.0, 0.0, 1.0, 1.0],
        cones: vec![ConeSpec::NonNeg { dim: 5 }],
        var_bounds: None,
        integrality: None,
    };

    println!("n={}, m={}", prob.num_vars(), prob.num_constraints());

    let settings = SolverSettings::default();
    match solve(&prob, &settings) {
        Ok(result) => {
            println!("Status: {:?}", result.status);
            println!("Obj: {:.6}", result.obj_val);
            println!("x: {:?}", result.x);
        }
        Err(e) => println!("Error: {}", e),
    }

    println!();

    // Test 2: Even simpler LP without bounds
    // min -x0 - x1
    // s.t. x0 + x1 <= 1
    println!("--- Test 2: Simpler LP ---");

    let a2 = CsMat::new_csc((1, 2), vec![0, 1, 2], vec![0, 0], vec![1.0, 1.0]);

    let prob2 = ProblemData {
        P: None,
        q: vec![-1.0, -1.0],
        A: a2,
        b: vec![1.0],
        cones: vec![ConeSpec::NonNeg { dim: 1 }],
        var_bounds: None,
        integrality: None,
    };

    println!("n={}, m={}", prob2.num_vars(), prob2.num_constraints());

    match solve(&prob2, &settings) {
        Ok(result) => {
            println!("Status: {:?}", result.status);
            println!("Obj: {:.6}", result.obj_val);
            println!("x: {:?}", result.x);
        }
        Err(e) => println!("Error: {}", e),
    }

    println!();

    // Test 3: Use var_bounds instead of explicit constraints
    println!("--- Test 3: LP with var_bounds ---");

    let a3 = CsMat::new_csc((1, 2), vec![0, 1, 2], vec![0, 0], vec![1.0, 1.0]);

    let prob3 = ProblemData {
        P: None,
        q: vec![-1.0, -1.0],
        A: a3,
        b: vec![1.0],
        cones: vec![ConeSpec::NonNeg { dim: 1 }],
        var_bounds: Some(vec![
            solver_core::VarBound {
                var: 0,
                lower: Some(0.0),
                upper: Some(1.0),
            },
            solver_core::VarBound {
                var: 1,
                lower: Some(0.0),
                upper: Some(1.0),
            },
        ]),
        integrality: None,
    };

    println!("n={}, m={}", prob3.num_vars(), prob3.num_constraints());

    match solve(&prob3, &settings) {
        Ok(result) => {
            println!("Status: {:?}", result.status);
            println!("Obj: {:.6}", result.obj_val);
            println!("x: {:?}", result.x);
        }
        Err(e) => println!("Error: {}", e),
    }
}
