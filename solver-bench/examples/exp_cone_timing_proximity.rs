use solver_core::{sparse, ConeSpec, ProblemData, SolverSettings, solve};

fn main() {
    // Test problem 1: Trivial exp cone problem
    let num_vars = 1;
    let triplets = vec![(0, 0, 1.0), (1, 0, 0.0), (2, 0, 0.0)];
    let A = sparse::from_triplets(3, num_vars, triplets);
    let b = vec![0.0, 1.0, 1.0];
    let q = vec![1.0];
    let prob1 = ProblemData {
        P: sparse::zeros(num_vars, num_vars),
        q,
        A,
        b,
        cones: vec![ConeSpec::ExponentialCone { count: 1 }],
    };

    // Test problem 2: CVXPY-style exp cone problem
    let num_vars = 3;
    let triplets = vec![
        (0, 0, 1.0), (1, 0, 0.0), (2, 0, 0.0),
        (0, 1, 0.0), (1, 1, 1.0), (2, 1, 0.0),
        (3, 2, 1.0), (4, 2, 1.0),
    ];
    let A = sparse::from_triplets(5, num_vars, triplets);
    let b = vec![0.0, 0.5, 1.0, 1.0, 2.0];
    let q = vec![-1.0, 0.0, 0.0];
    let prob2 = ProblemData {
        P: sparse::zeros(num_vars, num_vars),
        q,
        A,
        b,
        cones: vec![ConeSpec::ExponentialCone { count: 1 }, ConeSpec::Zero { dim: 2 }],
    };

    println!("================================================================================");
    println!("EXPONENTIAL CONE TIMING WITH PROXIMITY STEP CONTROL");
    println!("================================================================================");
    println!();
    println!("{:35} {:>5}    {:>8}       {:>10}    {:>7}", "Problem", "Iters", "Time (ms)", "Objective", "µs/iter");
    println!("--------------------------------------------------------------------------------");

    // Enable proximity step control
    let mut settings = SolverSettings::default();
    settings.max_iter = 200;
    settings.use_proximity_step_control = true;

    // Benchmark trivial problem
    let start = std::time::Instant::now();
    let result1 = solve(&prob1, &settings).unwrap();
    let time1 = start.elapsed().as_secs_f64() * 1000.0;
    let us_per_iter1 = (time1 * 1000.0) / (result1.info.iters as f64);

    println!("{:35} {:>5}    {:>8.2}    {:>12.6}    {:>7.1}",
             "Trivial (n=1, m=3)",
             result1.info.iters,
             time1,
             result1.info.obj_val,
             us_per_iter1);

    // Benchmark CVXPY-style problem
    let start = std::time::Instant::now();
    let result2 = solve(&prob2, &settings).unwrap();
    let time2 = start.elapsed().as_secs_f64() * 1000.0;
    let us_per_iter2 = (time2 * 1000.0) / (result2.info.iters as f64);

    println!("{:35} {:>5}    {:>8.2}    {:>12.6}    {:>7.1}",
             "CVXPY-style (n=3, m=5)",
             result2.info.iters,
             time2,
             result2.info.obj_val,
             us_per_iter2);

    println!("================================================================================");
}
