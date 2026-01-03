//! Simple benchmark to debug MIP solver
//!
//! Run with: cargo run --release -p solver-mip --example simple_bench

use solver_core::{ConeSpec, ProblemData, VarBound, VarType};
use solver_mip::{solve_mip, MipSettings};
use sprs::CsMat;
use std::time::Instant;

fn main() {
    println!("=== Simple MIP Solver Test ===\n");

    // Test 1: Very simple binary LP
    test_simple_binary_lp();

    // Test 2: Small knapsack
    test_small_knapsack();
}

/// Simple binary LP:
/// max x0 + x1
/// s.t. x0 + x1 <= 1
///      x0, x1 in {0,1}
fn test_simple_binary_lp() {
    println!("--- Test 1: Simple Binary LP ---");
    println!("max x0 + x1 s.t. x0 + x1 <= 1, x binary");

    // Constraint: x0 + x1 + s = 1, s >= 0 (so x0 + x1 <= 1)
    let a = CsMat::new_csc(
        (1, 2),
        vec![0, 1, 2],
        vec![0, 0],
        vec![1.0, 1.0],
    );

    let prob = ProblemData {
        P: None,
        q: vec![-1.0, -1.0], // max => min -
        A: a,
        b: vec![1.0],
        cones: vec![ConeSpec::NonNeg { dim: 1 }],
        var_bounds: Some(vec![
            VarBound { var: 0, lower: Some(0.0), upper: Some(1.0) },
            VarBound { var: 1, lower: Some(0.0), upper: Some(1.0) },
        ]),
        integrality: Some(vec![VarType::Binary, VarType::Binary]),
    };

    run_solve(&prob);
}

/// Small knapsack:
/// max 3x0 + 2x1 + 4x2
/// s.t. 2x0 + x1 + 3x2 <= 4
///      x binary
fn test_small_knapsack() {
    println!("--- Test 2: Small Knapsack ---");
    println!("max 3x0 + 2x1 + 4x2 s.t. 2x0 + x1 + 3x2 <= 4, x binary");

    // Constraint: 2x0 + x1 + 3x2 + s = 4, s >= 0
    let a = CsMat::new_csc(
        (1, 3),
        vec![0, 1, 2, 3],
        vec![0, 0, 0],
        vec![2.0, 1.0, 3.0],
    );

    let prob = ProblemData {
        P: None,
        q: vec![-3.0, -2.0, -4.0], // max => min -
        A: a,
        b: vec![4.0],
        cones: vec![ConeSpec::NonNeg { dim: 1 }],
        var_bounds: Some(vec![
            VarBound { var: 0, lower: Some(0.0), upper: Some(1.0) },
            VarBound { var: 1, lower: Some(0.0), upper: Some(1.0) },
            VarBound { var: 2, lower: Some(0.0), upper: Some(1.0) },
        ]),
        integrality: Some(vec![VarType::Binary, VarType::Binary, VarType::Binary]),
    };

    run_solve(&prob);
}

fn run_solve(prob: &ProblemData) {
    // First test the LP relaxation directly with solver-core
    println!("Testing LP relaxation with solver-core...");

    let lp_settings = solver_core::SolverSettings {
        verbose: true,
        ..Default::default()
    };

    // Create LP relaxation (same problem but without integrality)
    let lp_prob = ProblemData {
        P: prob.P.clone(),
        q: prob.q.clone(),
        A: prob.A.clone(),
        b: prob.b.clone(),
        cones: prob.cones.clone(),
        var_bounds: prob.var_bounds.clone(),
        integrality: None,
    };

    match solver_core::solve(&lp_prob, &lp_settings) {
        Ok(result) => {
            println!("LP Status: {:?}", result.status);
            println!("LP Obj: {:.6}", result.obj_val);
            println!("LP x: {:?}", result.x);
        }
        Err(e) => {
            println!("LP Error: {}", e);
        }
    }

    println!("\nNow testing MIP solver...");

    let settings = MipSettings {
        verbose: true,
        max_nodes: 1000,
        gap_tol: 1e-4,
        log_freq: 1,
        ..Default::default()
    };

    println!("Problem: n={}, m={}", prob.num_vars(), prob.num_constraints());

    let start = Instant::now();
    let result = solve_mip(prob, &settings);
    let elapsed = start.elapsed();

    match result {
        Ok(sol) => {
            println!("Status: {:?}", sol.status);
            if sol.status.has_solution() {
                println!("Objective: {:.6} (maximizing: {:.6})", sol.obj_val, -sol.obj_val);
                println!("Solution: {:?}", sol.x);
                println!("Bound: {:.6}", sol.bound);
                println!("Gap: {:.4}%", sol.gap * 100.0);
            }
            println!("Nodes: {}, Cuts: {}", sol.nodes_explored, sol.cuts_added);
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }
    println!("Time: {:.3}s\n", elapsed.as_secs_f64());
}
