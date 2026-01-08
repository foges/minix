//! Diagnose BOYD1 and BOYD2 problems

use solver_core::{solve, SolverSettings};
use std::path::Path;

fn main() {
    println!("\n=== BOYD Problem Diagnostics ===\n");

    for name in &["BOYD1", "BOYD2"] {
        println!("Testing {}...", name);

        let prob = match maros_meszaros::load_problem(name) {
            Ok(p) => p,
            Err(e) => {
                println!("  Error loading: {:?}\n", e);
                continue;
            }
        };

        let settings = SolverSettings {
            verbose: false,
            max_iter: 30,
            tol_feas: 1e-8,
            tol_gap: 1e-8,
            ..Default::default()
        };

        match solve(&prob, &settings) {
            Ok(sol) => {
                println!("  Status: {:?}", sol.status);
                println!("  Iterations: {}", sol.info.iters);
                println!("  Objective: {:.6e}", sol.obj_val);
                println!("  Residuals:");
                println!("    primal_res: {:.6e}", sol.info.primal_res);
                println!("    dual_res:   {:.6e}", sol.info.dual_res);
                println!("    gap:        {:.6e}", sol.info.gap);
                println!("    mu:         {:.6e}", sol.info.mu);
            },
            Err(e) => {
                println!("  Error: {:?}", e);
            }
        }
        println!();
    }
}
