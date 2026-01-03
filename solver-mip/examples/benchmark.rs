//! Benchmark the MIP solver on classic optimization problems.
//!
//! Run with: cargo run --release -p solver-mip --example benchmark

use solver_core::{ConeSpec, ProblemData, VarBound, VarType};
use solver_mip::{solve_mip, MipSettings, MipStatus};
use sprs::CsMat;
use std::time::Instant;

fn main() {
    println!("=== MIP Solver Benchmark ===\n");

    // Run benchmarks
    benchmark_knapsack(10);
    benchmark_knapsack(15);
    benchmark_knapsack(20);

    benchmark_set_cover(10, 5);
    benchmark_set_cover(15, 8);

    benchmark_facility_location(5, 10);

    benchmark_portfolio_misocp(5);
    benchmark_portfolio_misocp(10);
}

/// 0-1 Knapsack Problem
///
/// max  sum_i v[i] * x[i]
/// s.t. sum_i w[i] * x[i] <= capacity
///      x[i] binary
fn benchmark_knapsack(n: usize) {
    println!("--- Knapsack Problem (n={}) ---", n);

    // Generate random-ish instance
    let values: Vec<f64> = (0..n).map(|i| (i * 7 + 3) as f64 % 20.0 + 5.0).collect();
    let weights: Vec<f64> = (0..n).map(|i| (i * 11 + 5) as f64 % 15.0 + 3.0).collect();
    let capacity: f64 = weights.iter().sum::<f64>() * 0.5;

    println!(
        "  Items: {}, Capacity: {:.1}, Total weight: {:.1}",
        n,
        capacity,
        weights.iter().sum::<f64>()
    );

    // Constraint: sum_i w[i] * x[i] + s = capacity, s >= 0
    let a = CsMat::new_csc(
        (1, n),
        (0..=n).collect(),
        (0..n).map(|_| 0).collect(),
        weights.clone(),
    );

    // Objective: max sum_i v[i] * x[i] => min -sum_i v[i] * x[i]
    let q: Vec<f64> = values.iter().map(|v| -v).collect();

    // Binary variables need [0,1] bounds for LP relaxation
    let var_bounds: Vec<VarBound> = (0..n)
        .map(|i| VarBound {
            var: i,
            lower: Some(0.0),
            upper: Some(1.0),
        })
        .collect();

    let prob = ProblemData {
        P: None,
        q,
        A: a,
        b: vec![capacity],
        cones: vec![ConeSpec::NonNeg { dim: 1 }],
        var_bounds: Some(var_bounds),
        integrality: Some(vec![VarType::Binary; n]),
    };

    run_mip_benchmark("Knapsack", &prob);
}

/// Set Cover Problem
///
/// min  sum_j c[j] * x[j]  (minimize cost of selected sets)
/// s.t. sum_{j: i in S_j} x[j] >= 1 for all elements i
///      x[j] binary
fn benchmark_set_cover(num_elements: usize, num_sets: usize) {
    println!(
        "--- Set Cover (elements={}, sets={}) ---",
        num_elements, num_sets
    );

    // Generate set membership (each set covers some elements)
    let mut membership = vec![vec![0.0; num_sets]; num_elements];
    for j in 0..num_sets {
        for i in 0..num_elements {
            // Set j covers element i with some pattern
            if (i + j * 3) % 4 < 2 || j == i % num_sets {
                membership[i][j] = 1.0;
            }
        }
    }

    // Costs: slightly favor smaller-indexed sets
    let costs: Vec<f64> = (0..num_sets).map(|j| 1.0 + (j as f64) * 0.1).collect();

    // Constraint: -sum_{j: i in S_j} x[j] + s_i = -1, s_i >= 0
    // This gives: sum_{j: i in S_j} x[j] >= 1
    let mut row_indices = Vec::new();
    let mut col_ptrs = vec![0usize];
    let mut values = Vec::new();

    for j in 0..num_sets {
        for i in 0..num_elements {
            if membership[i][j] > 0.5 {
                row_indices.push(i);
                values.push(-1.0);
            }
        }
        col_ptrs.push(row_indices.len());
    }

    let a = CsMat::new_csc(
        (num_elements, num_sets),
        col_ptrs,
        row_indices,
        values,
    );

    let prob = ProblemData {
        P: None,
        q: costs,
        A: a,
        b: vec![-1.0; num_elements],
        cones: vec![ConeSpec::NonNeg { dim: num_elements }],
        var_bounds: None,
        integrality: Some(vec![VarType::Binary; num_sets]),
    };

    run_mip_benchmark("Set Cover", &prob);
}

/// Uncapacitated Facility Location
///
/// min  sum_i sum_j c[i][j] * x[i][j] + sum_j f[j] * y[j]
/// s.t. sum_j x[i][j] = 1 for all customers i
///      x[i][j] <= y[j] for all i,j
///      y[j] binary, x[i][j] >= 0
fn benchmark_facility_location(num_facilities: usize, num_customers: usize) {
    println!(
        "--- Facility Location (facilities={}, customers={}) ---",
        num_facilities, num_customers
    );

    // Variables: x[i][j] for i in customers, j in facilities, then y[j]
    let num_x = num_customers * num_facilities;
    let num_y = num_facilities;
    let n = num_x + num_y;

    // Assignment costs
    let mut q = Vec::with_capacity(n);
    for i in 0..num_customers {
        for j in 0..num_facilities {
            // Distance-like cost
            let cost = ((i as f64 - j as f64 * 2.0).abs() + 1.0) * 0.5;
            q.push(cost);
        }
    }
    // Facility opening costs
    for j in 0..num_facilities {
        q.push(5.0 + j as f64);
    }

    // Constraints:
    // 1. sum_j x[i][j] = 1 for each customer (Zero cone)
    // 2. x[i][j] <= y[j] => -x[i][j] + y[j] + s >= 0 (NonNeg cone)

    let num_assignment = num_customers;
    let num_capacity = num_x;
    let m = num_assignment + num_capacity;

    let mut row_indices = Vec::new();
    let mut col_ptrs = vec![0usize];
    let mut values = Vec::new();

    // x variables
    for i in 0..num_customers {
        for j in 0..num_facilities {
            // Assignment constraint row i: coefficient 1
            row_indices.push(i);
            values.push(1.0);

            // Capacity constraint row num_assignment + i*num_facilities + j: coefficient -1
            row_indices.push(num_assignment + i * num_facilities + j);
            values.push(-1.0);

            col_ptrs.push(row_indices.len());
        }
    }

    // y variables
    for j in 0..num_facilities {
        for i in 0..num_customers {
            // Capacity constraint: coefficient 1
            row_indices.push(num_assignment + i * num_facilities + j);
            values.push(1.0);
        }
        col_ptrs.push(row_indices.len());
    }

    let a = CsMat::new_csc((m, n), col_ptrs, row_indices, values);

    let mut b = vec![1.0; num_assignment]; // Assignment: sum = 1
    b.extend(vec![0.0; num_capacity]); // Capacity: -x + y + s = 0

    let cones = vec![
        ConeSpec::Zero { dim: num_assignment },
        ConeSpec::NonNeg { dim: num_capacity },
    ];

    // x >= 0, y binary
    let mut integrality = vec![VarType::Continuous; num_x];
    integrality.extend(vec![VarType::Binary; num_y]);

    // Bounds: x >= 0 (implicit from NonNeg on capacity slack)
    let var_bounds: Vec<VarBound> = (0..num_x)
        .map(|i| VarBound {
            var: i,
            lower: Some(0.0),
            upper: Some(1.0),
        })
        .collect();

    let prob = ProblemData {
        P: None,
        q,
        A: a,
        b,
        cones,
        var_bounds: Some(var_bounds),
        integrality: Some(integrality),
    };

    run_mip_benchmark("Facility Location", &prob);
}

/// Portfolio optimization with cardinality constraint (MISOCP)
///
/// min  -mu^T x + lambda * ||Sigma^{1/2} x||_2
/// s.t. sum_i x[i] = 1
///      x[i] <= y[i]  (can only invest if selected)
///      sum_i y[i] <= k (at most k assets)
///      x[i] >= 0, y[i] binary
fn benchmark_portfolio_misocp(num_assets: usize) {
    println!("--- Portfolio MISOCP (assets={}) ---", num_assets);

    let k = (num_assets + 1) / 2; // Max assets to select
    let lambda = 0.5; // Risk aversion

    // Expected returns
    let mu: Vec<f64> = (0..num_assets)
        .map(|i| 0.05 + 0.02 * (i as f64))
        .collect();

    // Risk (simplified: diagonal covariance)
    let sigma_sqrt: Vec<f64> = (0..num_assets)
        .map(|i| 0.1 + 0.05 * (i as f64))
        .collect();

    // Variables: x[0..n], y[0..n], t (SOC auxiliary)
    let n_x = num_assets;
    let n_y = num_assets;
    let n = n_x + n_y + 1; // +1 for t

    // Objective: -mu^T x + lambda * t
    let mut q = vec![0.0; n];
    for i in 0..num_assets {
        q[i] = -mu[i];
    }
    q[n - 1] = lambda; // coefficient for t

    // Constraints:
    // 1. sum_i x[i] = 1 (Zero)
    // 2. x[i] - y[i] <= 0 (NonNeg) => -x[i] + y[i] + s = 0
    // 3. sum_i y[i] <= k (NonNeg) => sum_i y[i] + s = k
    // 4. SOC: (t, sigma_sqrt[0]*x[0], ..., sigma_sqrt[n-1]*x[n-1]) in SOC

    let m_budget = 1;
    let m_link = num_assets;
    let m_cardinality = 1;
    let m_soc = 1 + num_assets; // t + scaled x

    let m = m_budget + m_link + m_cardinality + m_soc;

    let mut row_indices = Vec::new();
    let mut col_ptrs = vec![0usize];
    let mut values = Vec::new();

    // x variables
    for i in 0..num_assets {
        // Budget: coefficient 1
        row_indices.push(0);
        values.push(1.0);

        // Link: -x[i]
        row_indices.push(m_budget + i);
        values.push(-1.0);

        // SOC: -sigma_sqrt[i] * x[i] for row m_budget + m_link + m_cardinality + 1 + i
        row_indices.push(m_budget + m_link + m_cardinality + 1 + i);
        values.push(-sigma_sqrt[i]);

        col_ptrs.push(row_indices.len());
    }

    // y variables
    for i in 0..num_assets {
        // Link: +y[i]
        row_indices.push(m_budget + i);
        values.push(1.0);

        // Cardinality: +y[i]
        row_indices.push(m_budget + m_link);
        values.push(1.0);

        col_ptrs.push(row_indices.len());
    }

    // t variable
    // SOC: -t for row m_budget + m_link + m_cardinality
    row_indices.push(m_budget + m_link + m_cardinality);
    values.push(-1.0);
    col_ptrs.push(row_indices.len());

    let a = CsMat::new_csc((m, n), col_ptrs, row_indices, values);

    let mut b = vec![0.0; m];
    b[0] = 1.0; // Budget = 1
    // Link: 0
    b[m_budget + m_link] = k as f64; // Cardinality <= k
    // SOC: 0

    let cones = vec![
        ConeSpec::Zero { dim: m_budget },
        ConeSpec::NonNeg { dim: m_link + m_cardinality },
        ConeSpec::Soc { dim: m_soc },
    ];

    let mut integrality = vec![VarType::Continuous; n_x];
    integrality.extend(vec![VarType::Binary; n_y]);
    integrality.push(VarType::Continuous); // t

    // Bounds: x in [0,1], y in [0,1] (binary), t >= 0
    let mut var_bounds: Vec<VarBound> = (0..num_assets)
        .map(|i| VarBound {
            var: i,
            lower: Some(0.0),
            upper: Some(1.0),
        })
        .collect();
    // y variables: implicit [0,1] from binary
    for i in 0..num_assets {
        var_bounds.push(VarBound {
            var: n_x + i,
            lower: Some(0.0),
            upper: Some(1.0),
        });
    }
    // t >= 0 (norm is always non-negative) - critical for bounded LP relaxation
    var_bounds.push(VarBound {
        var: n - 1,
        lower: Some(0.0),
        upper: None,
    });

    let prob = ProblemData {
        P: None,
        q,
        A: a,
        b,
        cones,
        var_bounds: Some(var_bounds),
        integrality: Some(integrality),
    };

    run_mip_benchmark("Portfolio MISOCP", &prob);
}

fn run_mip_benchmark(name: &str, prob: &ProblemData) {
    // Debug: test LP relaxation first for MISOCP problems
    if name.contains("MISOCP") {
        let lp_prob = ProblemData {
            P: prob.P.clone(),
            q: prob.q.clone(),
            A: prob.A.clone(),
            b: prob.b.clone(),
            cones: prob.cones.clone(),
            var_bounds: prob.var_bounds.clone(),
            integrality: None, // Continuous relaxation
        };
        let lp_settings = solver_core::SolverSettings { verbose: false, max_iter: 200, ..Default::default() };
        match solver_core::solve(&lp_prob, &lp_settings) {
            Ok(r) => println!("  LP relaxation: {:?}, obj={:.6}", r.status, r.obj_val),
            Err(e) => println!("  LP relaxation error: {}", e),
        }
    }

    let settings = MipSettings {
        verbose: false,
        max_nodes: 10000,
        gap_tol: 1e-4,
        ..Default::default()
    };

    let start = Instant::now();
    let result = solve_mip(prob, &settings);
    let elapsed = start.elapsed();

    match result {
        Ok(sol) => {
            println!("  Status: {:?}", sol.status);
            if sol.status.has_solution() {
                println!("  Objective: {:.6}", sol.obj_val);
                println!("  Bound: {:.6}", sol.bound);
                println!("  Gap: {:.2}%", sol.gap * 100.0);
            }
            println!("  Nodes explored: {}", sol.nodes_explored);
            println!("  Cuts added: {}", sol.cuts_added);
            println!("  Time: {:.3}s", elapsed.as_secs_f64());
        }
        Err(e) => {
            println!("  Error: {}", e);
            println!("  Time: {:.3}s", elapsed.as_secs_f64());
        }
    }
    println!();
}
