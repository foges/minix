//! Exp cone wall-clock timing benchmark

use solver_core::{ConeSpec, ProblemData, SolverSettings, solve};
use solver_core::linalg::sparse;
use std::time::Instant;

fn exp_cone_trivial() -> ProblemData {
    let num_vars = 1;
    let q = vec![1.0];
    let triplets = vec![(0, 0, -1.0)];
    let A = sparse::from_triplets(3, num_vars, triplets);
    let b = vec![0.0, 1.0, 1.0];
    let cones = vec![ConeSpec::Exp { count: 1 }];

    ProblemData {
        P: None,
        q,
        A,
        b,
        cones,
        var_bounds: None,
        integrality: None,
    }
}

fn exp_cone_cvxpy() -> ProblemData {
    let num_vars = 3;
    let q = vec![1.0, 1.0, 1.0];
    let e = std::f64::consts::E;

    let triplets = vec![
        (0, 0, -1.0),
        (1, 1, -1.0),
        (2, 2, -1.0),
        (3, 1, 1.0),
        (4, 2, 1.0),
    ];

    let A = sparse::from_triplets(5, num_vars, triplets);
    let b = vec![0.0, 0.0, 0.0, 1.0, e];

    let cones = vec![
        ConeSpec::Exp { count: 1 },
        ConeSpec::Zero { dim: 2 },
    ];

    ProblemData {
        P: None,
        q,
        A,
        b,
        cones,
        var_bounds: None,
        integrality: None,
    }
}

fn main() {
    println!("\n{:=<80}", "");
    println!("EXPONENTIAL CONE WALL-CLOCK TIMING BENCHMARK");
    println!("{:=<80}\n", "");

    let problems = vec![
        ("Trivial (n=1, m=3)", exp_cone_trivial(), 50),
        ("CVXPY-style (n=3, m=5)", exp_cone_cvxpy(), 200),
    ];

    println!("{:<30} {:>10} {:>12} {:>15} {:>10}",
             "Problem", "Iters", "Time (ms)", "Objective", "µs/iter");
    println!("{:-<80}", "");

    for (name, prob, max_iter) in problems {
        let mut settings = SolverSettings::default();
        settings.verbose = false;
        settings.max_iter = max_iter;
        settings.use_proximity_step_control = false;  // Disabled

        // Warmup run
        let _ = solve(&prob, &settings);

        // Timed runs (5 iterations for average)
        let mut times = Vec::new();
        let mut iters = 0;
        let mut obj = 0.0;

        for _ in 0..5 {
            let start = Instant::now();
            let result = solve(&prob, &settings).unwrap();
            let elapsed = start.elapsed();
            times.push(elapsed.as_micros() as f64 / 1000.0);
            iters = result.info.iters;
            obj = result.obj_val;
        }

        let avg_time = times.iter().sum::<f64>() / times.len() as f64;
        let time_per_iter = (avg_time * 1000.0) / iters as f64; // microseconds

        println!("{:<30} {:>10} {:>12.2} {:>15.6} {:>10.1}",
                 name, iters, avg_time, obj, time_per_iter);
    }

    println!("{:=<80}\n", "");
}
