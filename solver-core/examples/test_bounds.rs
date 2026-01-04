use solver_core::{solve, ConeSpec, ProblemData, SolverSettings, VarBound};
use sprs::CsMat;

fn main() {
    println!("=== Testing solver-core bound enforcement ===\n");

    // min -x0 - x1
    // s.t. x0 + x1 <= 1
    // x0 in [0, 1], x1 in [0, 0]  (x1 fixed to 0)

    let a = CsMat::new_csc((1, 2), vec![0, 1, 2], vec![0, 0], vec![1.0, 1.0]);

    let prob = ProblemData {
        P: None,
        q: vec![-1.0, -1.0],
        A: a,
        b: vec![1.0],
        cones: vec![ConeSpec::NonNeg { dim: 1 }],
        var_bounds: Some(vec![
            VarBound {
                var: 0,
                lower: Some(0.0),
                upper: Some(1.0),
            },
            VarBound {
                var: 1,
                lower: Some(0.0),
                upper: Some(0.0),
            }, // x1 = 0
        ]),
        integrality: None,
    };

    println!("Solving with x1 fixed to 0...");
    println!("Expected: x0 = 1, x1 = 0, obj = -1");

    let settings = SolverSettings::default();
    match solve(&prob, &settings) {
        Ok(result) => {
            println!("Status: {:?}", result.status);
            println!("Obj: {:.6}", result.obj_val);
            println!("x: {:?}", result.x);
        }
        Err(e) => println!("Error: {}", e),
    }
}
