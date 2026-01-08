//! Baseline benchmark for exp cone improvements

use solver_core::{ConeSpec, ProblemData, SolverSettings, solve};
use solver_core::linalg::sparse;
use std::time::Instant;

fn trivial() -> ProblemData {
    let num_vars = 1;
    let q = vec![1.0];
    let triplets = vec![(0, 0, -1.0)];
    let A = sparse::from_triplets(3, num_vars, triplets);
    let b = vec![0.0, 1.0, 1.0];
    ProblemData {
        P: None,
        q,
        A,
        b,
        cones: vec![ConeSpec::Exp { count: 1 }],
        var_bounds: None,
        integrality: None,
    }
}

fn cvxpy_style() -> ProblemData {
    let num_vars = 3;
    let q = vec![1.0, 1.0, 1.0];
    let e = std::f64::consts::E;
    let triplets = vec![
        (0, 0, -1.0), (1, 1, -1.0), (2, 2, -1.0),
        (3, 1, 1.0), (4, 2, 1.0),
    ];
    let A = sparse::from_triplets(5, num_vars, triplets);
    let b = vec![0.0, 0.0, 0.0, 1.0, e];
    ProblemData {
        P: None,
        q,
        A,
        b,
        cones: vec![ConeSpec::Exp { count: 1 }, ConeSpec::Zero { dim: 2 }],
        var_bounds: None,
        integrality: None,
    }
}

fn main() {
    println!("\n{:=<80}", "");
    println!("EXPONENTIAL CONE BASELINE BENCHMARK");
    println!("{:=<80}\n", "");

    let problems = vec![
        ("trivial", trivial(), 50),
        ("cvxpy", cvxpy_style(), 200),
    ];

    println!("{:<20} {:>8} {:>12} {:>12} {:>10} {:>12}",
             "Problem", "Iters", "Time (ms)", "Objective", "µs/iter", "Status");
    println!("{:-<80}", "");

    let mut total_time = 0.0;

    for (name, prob, expected_iters) in problems {
        let mut settings = SolverSettings::default();
        settings.max_iter = expected_iters;
        settings.verbose = false;

        // Warmup
        let _ = solve(&prob, &settings);

        // Timed solve (average of 5 runs)
        let mut times = vec![];
        let mut iters = 0;
        let mut obj = 0.0;
        let mut status = solver_core::SolveStatus::MaxIters;

        for _ in 0..5 {
            let start = Instant::now();
            let result = solve(&prob, &settings).unwrap();
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            times.push(elapsed);
            iters = result.info.iters;
            obj = result.obj_val;
            status = result.status;
        }

        let avg_time = times.iter().sum::<f64>() / times.len() as f64;
        let time_per_iter = (avg_time * 1000.0) / iters as f64;

        let status_str = match status {
            solver_core::SolveStatus::Optimal => "Optimal",
            solver_core::SolveStatus::AlmostOptimal => "AlmostOpt",
            solver_core::SolveStatus::MaxIters => "MaxIter",
            _ => "Other",
        };

        println!("{:<20} {:>8} {:>12.2} {:>12.4} {:>10.1} {:>12}",
                 name, iters, avg_time, obj, time_per_iter, status_str);

        total_time += avg_time;
    }

    println!("{:-<80}", "");
    println!("Total time: {:.2} ms\n", total_time);
    println!("{:=<80}\n", "");
}
