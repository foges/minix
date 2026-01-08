//! Check if exp cone initialization is truly interior

use solver_core::cones::{ExpCone, ConeKernel};

fn main() {
    println!("\n=== Exp Cone Interior Check ===\n");

    let cone = ExpCone::new(1);
    let mut s = vec![0.0; 3];
    let mut z = vec![0.0; 3];

    // Get unit initialization
    cone.unit_initialization(&mut s, &mut z);

    println!("Initialization values:");
    println!("  s = [{:.6}, {:.6}, {:.6}]", s[0], s[1], s[2]);
    println!("  z = [{:.6}, {:.6}, {:.6}]", z[0], z[1], z[2]);
    println!();

    // Check if interior
    let s_interior = cone.is_interior_primal(&s);
    let z_interior = cone.is_interior_dual(&z);

    println!("Interior checks:");
    println!("  s is interior (primal): {}", s_interior);
    println!("  z is interior (dual):   {}", z_interior);
    println!();

    // Compute ψ for primal cone
    let x = s[0];
    let y = s[1];
    let z_val = s[2];
    if y > 0.0 && z_val > 0.0 {
        let psi = y * (z_val / y).ln() - x;
        println!("Primal cone check:");
        println!("  x = {:.6}", x);
        println!("  y = {:.6}", y);
        println!("  z = {:.6}", z_val);
        println!("  ψ = y*log(z/y) - x = {:.6}", psi);
        println!("  ψ > 0? {} (required for interior)", psi > 0.0);
        println!("  exp(x/y) = {:.6}", (x/y).exp());
        println!("  y*exp(x/y) = {:.6}", y * (x/y).exp());
        println!("  y*exp(x/y) < z? {} (equivalent condition)", y * (x/y).exp() < z_val);
    }
    println!();

    // Compute barrier value
    let barrier = cone.barrier_value(&s);
    println!("Barrier value: {:.6}", barrier);
    println!("  Is finite? {}", barrier.is_finite());
    println!();

    // Check a simple direction
    let ds = vec![0.1, 0.0, 0.0];  // Small step in x direction
    let alpha = cone.step_to_boundary_primal(&s, &ds);
    println!("Step to boundary test:");
    println!("  ds = [{:.6}, {:.6}, {:.6}]", ds[0], ds[1], ds[2]);
    println!("  alpha = {:.6}", alpha);
    println!();

    // Try the step
    let s_new = vec![s[0] + 0.5 * ds[0], s[1] + 0.5 * ds[1], s[2] + 0.5 * ds[2]];
    let s_new_interior = cone.is_interior_primal(&s_new);
    println!("After 50% step:");
    println!("  s_new = [{:.6}, {:.6}, {:.6}]", s_new[0], s_new[1], s_new[2]);
    println!("  is interior? {}", s_new_interior);
}
