//! Cone kernel unit tests with finite difference validation.
//!
//! This module provides comprehensive testing for all cone implementations,
//! including finite difference checking of gradients and Hessians.

use solver_core::cones::{ConeKernel, NonNegCone, SocCone, ZeroCone};

/// Finite difference tolerance for gradient checking
const FD_GRAD_TOL: f64 = 1e-6;

/// Finite difference tolerance for Hessian checking
const FD_HESS_TOL: f64 = 1e-5;

/// Compute finite difference approximation of gradient.
///
/// Uses central differences: ∂f/∂x_i ≈ (f(x + εe_i) - f(x - εe_i)) / (2ε)
fn finite_diff_gradient<K: ConeKernel>(cone: &K, s: &[f64], grad_fd: &mut [f64]) {
    let n = s.len();
    let mut s_plus = s.to_vec();
    let mut s_minus = s.to_vec();

    for i in 0..n {
        // Choose ε relative to s[i]
        let eps = 1e-6 * s[i].abs().max(1.0);

        // f(s + εe_i)
        s_plus[i] = s[i] + eps;
        let f_plus = cone.barrier_value(&s_plus);
        s_plus[i] = s[i]; // restore

        // f(s - εe_i)
        s_minus[i] = s[i] - eps;
        let f_minus = cone.barrier_value(&s_minus);
        s_minus[i] = s[i]; // restore

        // Central difference
        grad_fd[i] = (f_plus - f_minus) / (2.0 * eps);
    }
}

/// Compute finite difference approximation of Hessian-vector product.
///
/// Uses central differences: ∇²f(x) v ≈ (∇f(x + εv) - ∇f(x - εv)) / (2ε)
fn finite_diff_hessian_apply<K: ConeKernel>(cone: &K, s: &[f64], v: &[f64], hess_v_fd: &mut [f64]) {
    let n = s.len();
    let mut s_plus = vec![0.0; n];
    let mut s_minus = vec![0.0; n];
    let mut grad_plus = vec![0.0; n];
    let mut grad_minus = vec![0.0; n];

    // Choose ε relative to ||s||
    let s_norm = s.iter().map(|x| x * x).sum::<f64>().sqrt();
    let eps = 1e-6 * s_norm.max(1.0);

    // s + εv
    for i in 0..n {
        s_plus[i] = s[i] + eps * v[i];
    }
    cone.barrier_grad_primal(&s_plus, &mut grad_plus);

    // s - εv
    for i in 0..n {
        s_minus[i] = s[i] - eps * v[i];
    }
    cone.barrier_grad_primal(&s_minus, &mut grad_minus);

    // Central difference
    for i in 0..n {
        hess_v_fd[i] = (grad_plus[i] - grad_minus[i]) / (2.0 * eps);
    }
}

/// Check gradient via finite differences.
fn check_gradient<K: ConeKernel>(cone: &K, s: &[f64], tol: f64) -> bool {
    let n = s.len();
    let mut grad = vec![0.0; n];
    let mut grad_fd = vec![0.0; n];

    cone.barrier_grad_primal(s, &mut grad);
    finite_diff_gradient(cone, s, &mut grad_fd);

    // Check relative error
    for i in 0..n {
        let err = (grad[i] - grad_fd[i]).abs();
        let scale = grad[i].abs().max(grad_fd[i].abs()).max(1.0);
        let rel_err = err / scale;

        if rel_err > tol {
            eprintln!(
                "Gradient check failed at index {}: analytic={}, fd={}, rel_err={}",
                i, grad[i], grad_fd[i], rel_err
            );
            return false;
        }
    }

    true
}

/// Check Hessian-vector product via finite differences.
fn check_hessian<K: ConeKernel>(cone: &K, s: &[f64], v: &[f64], tol: f64) -> bool {
    let n = s.len();
    let mut hess_v = vec![0.0; n];
    let mut hess_v_fd = vec![0.0; n];

    cone.barrier_hess_apply_primal(s, v, &mut hess_v);
    finite_diff_hessian_apply(cone, s, v, &mut hess_v_fd);

    // Check relative error
    for i in 0..n {
        let err = (hess_v[i] - hess_v_fd[i]).abs();
        let scale = hess_v[i].abs().max(hess_v_fd[i].abs()).max(1.0);
        let rel_err = err / scale;

        if rel_err > tol {
            eprintln!(
                "Hessian check failed at index {}: analytic={}, fd={}, rel_err={}",
                i, hess_v[i], hess_v_fd[i], rel_err
            );
            return false;
        }
    }

    true
}

// ============================================================================
// NonNeg Cone Tests
// ============================================================================

#[test]
fn test_nonneg_gradient_fd() {
    let cone = NonNegCone::new(5);

    // Test at several interior points
    let test_points = vec![
        vec![1.0, 1.0, 1.0, 1.0, 1.0],
        vec![0.5, 1.0, 2.0, 3.0, 4.0],
        vec![0.1, 0.2, 0.3, 0.4, 0.5],
        vec![10.0, 20.0, 30.0, 40.0, 50.0],
    ];

    for s in test_points {
        assert!(cone.is_interior_primal(&s), "Test point not interior");
        assert!(
            check_gradient(&cone, &s, FD_GRAD_TOL),
            "Gradient check failed at {:?}",
            s
        );
    }
}

#[test]
fn test_nonneg_hessian_fd() {
    let cone = NonNegCone::new(5);

    let s = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let test_vectors = vec![
        vec![1.0, 0.0, 0.0, 0.0, 0.0],   // unit vector
        vec![1.0, 1.0, 1.0, 1.0, 1.0],   // ones
        vec![0.5, -0.5, 1.0, -1.0, 2.0], // mixed
        vec![1.0, 2.0, 3.0, 4.0, 5.0],   // arbitrary
    ];

    for v in test_vectors {
        assert!(
            check_hessian(&cone, &s, &v, FD_HESS_TOL),
            "Hessian check failed with v={:?}",
            v
        );
    }
}

#[test]
fn test_nonneg_gradient_various_dimensions() {
    for dim in [1, 2, 5, 10, 50] {
        let cone = NonNegCone::new(dim);
        let s: Vec<f64> = (1..=dim).map(|i| i as f64).collect();

        assert!(
            check_gradient(&cone, &s, FD_GRAD_TOL),
            "Gradient check failed for dim={}",
            dim
        );
    }
}

// ============================================================================
// Property-based tests (random points)
// ============================================================================

#[test]
fn test_nonneg_gradient_random() {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(12345);
    let cone = NonNegCone::new(10);

    for _ in 0..20 {
        // Generate random interior point
        let s: Vec<f64> = (0..10).map(|_| rng.gen_range(0.1..10.0)).collect();

        assert!(cone.is_interior_primal(&s));
        assert!(
            check_gradient(&cone, &s, FD_GRAD_TOL),
            "Random gradient check failed"
        );
    }
}

#[test]
fn test_nonneg_hessian_random() {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(54321);
    let cone = NonNegCone::new(10);

    for _ in 0..20 {
        // Generate random interior point and direction
        let s: Vec<f64> = (0..10).map(|_| rng.gen_range(0.1..10.0)).collect();
        let v: Vec<f64> = (0..10).map(|_| rng.gen_range(-1.0..1.0)).collect();

        assert!(cone.is_interior_primal(&s));
        assert!(
            check_hessian(&cone, &s, &v, FD_HESS_TOL),
            "Random Hessian check failed"
        );
    }
}

// ============================================================================
// Zero Cone Tests (no derivatives to check)
// ============================================================================

#[test]
fn test_zero_cone_properties() {
    let cone = ZeroCone::new(10);
    assert_eq!(cone.dim(), 10);
    assert_eq!(cone.barrier_degree(), 0);

    let z = vec![1.0; 10];
    assert!(cone.is_interior_dual(&z));
    assert!(!cone.is_interior_primal(&z));
}

// ============================================================================
// SOC Cone Tests
// ============================================================================

#[test]
fn test_soc_gradient_fd() {
    let cone = SocCone::new(5);

    // Test at several interior points
    let test_points = vec![
        vec![2.0, 0.0, 0.0, 0.0, 0.0],  // (2, 0...)
        vec![3.0, 1.0, 1.0, 1.0, 0.0],  // t=3, ||x||=√3
        vec![10.0, 1.0, 2.0, 3.0, 4.0], // t=10, ||x||=√30
        vec![5.0, 2.0, 1.0, 0.5, 0.5],  // t=5, ||x||≈2.3
    ];

    for s in test_points {
        assert!(
            cone.is_interior_primal(&s),
            "Test point not interior: {:?}",
            s
        );
        assert!(
            check_gradient(&cone, &s, FD_GRAD_TOL),
            "Gradient check failed at {:?}",
            s
        );
    }
}

#[test]
fn test_soc_hessian_fd() {
    let cone = SocCone::new(5);

    let s = vec![5.0, 1.0, 2.0, 1.0, 1.0];
    let test_vectors = vec![
        vec![1.0, 0.0, 0.0, 0.0, 0.0],   // t-direction
        vec![0.0, 1.0, 0.0, 0.0, 0.0],   // x-direction
        vec![1.0, 1.0, 1.0, 1.0, 1.0],   // ones
        vec![0.5, -0.5, 1.0, -1.0, 2.0], // mixed
        vec![2.0, 1.0, 1.0, 1.0, 1.0],   // arbitrary
    ];

    for v in test_vectors {
        assert!(
            check_hessian(&cone, &s, &v, FD_HESS_TOL),
            "Hessian check failed with v={:?}",
            v
        );
    }
}

#[test]
fn test_soc_gradient_various_dimensions() {
    for dim in [2, 3, 5, 10, 20] {
        let cone = SocCone::new(dim);

        // Create interior point: t = dim, x = (1, 1, ..., 1)
        // ||x|| = √(dim-1), need t > √(dim-1)
        let mut s = vec![1.0; dim];
        s[0] = (dim as f64).sqrt() + 1.0; // Safely interior

        assert!(
            cone.is_interior_primal(&s),
            "Point not interior for dim={}",
            dim
        );
        assert!(
            check_gradient(&cone, &s, FD_GRAD_TOL),
            "Gradient check failed for dim={}",
            dim
        );
    }
}

#[test]
fn test_soc_gradient_random() {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(11111);
    let cone = SocCone::new(10);

    for _ in 0..20 {
        // Generate random interior point
        // x ~ uniform[0, 1], t = ||x|| + margin
        let x: Vec<f64> = (0..9).map(|_| rng.gen_range(0.0..1.0)).collect();
        let x_norm = x.iter().map(|&xi| xi * xi).sum::<f64>().sqrt();
        let margin = rng.gen_range(0.5..2.0);

        let mut s = vec![0.0; 10];
        s[0] = x_norm + margin;
        s[1..].copy_from_slice(&x);

        assert!(cone.is_interior_primal(&s));
        assert!(
            check_gradient(&cone, &s, FD_GRAD_TOL),
            "Random gradient check failed"
        );
    }
}

#[test]
fn test_soc_hessian_random() {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(22222);
    let cone = SocCone::new(10);

    for _ in 0..20 {
        // Generate random interior point
        let x: Vec<f64> = (0..9).map(|_| rng.gen_range(0.0..1.0)).collect();
        let x_norm = x.iter().map(|&xi| xi * xi).sum::<f64>().sqrt();
        let margin = rng.gen_range(0.5..2.0);

        let mut s = vec![0.0; 10];
        s[0] = x_norm + margin;
        s[1..].copy_from_slice(&x);

        // Random direction
        let v: Vec<f64> = (0..10).map(|_| rng.gen_range(-1.0..1.0)).collect();

        assert!(cone.is_interior_primal(&s));
        assert!(
            check_hessian(&cone, &s, &v, FD_HESS_TOL),
            "Random Hessian check failed"
        );
    }
}

#[test]
fn test_soc_near_axis() {
    // Test case where x ≈ 0 (near the axis of the cone)
    let cone = SocCone::new(5);

    let s = vec![1.0, 1e-8, 1e-8, 1e-8, 1e-8];
    assert!(cone.is_interior_primal(&s));

    // Gradient should still be accurate
    assert!(
        check_gradient(&cone, &s, FD_GRAD_TOL * 10.0), // Slightly relaxed tolerance
        "Gradient check failed near axis"
    );
}

#[test]
fn test_soc_step_to_boundary() {
    let cone = SocCone::new(3);

    // Test 1: Interior point, direction pointing outward
    // s = (2, 1, 0), ||x|| = 1, so s is interior (2 > 1)
    let s = vec![2.0, 1.0, 0.0];
    // ds = (-1, 0, 0), this decreases t
    let ds = vec![-1.0, 0.0, 0.0];

    let alpha = cone.step_to_boundary_primal(&s, &ds);
    eprintln!("Test 1: s={:?}, ds={:?}, alpha={}", s, ds, alpha);

    // At alpha = 1, new t = 2 - 1 = 1, still >= ||x|| = 1
    // But we want the point where t = ||x|| exactly
    // t + alpha*dt = ||x + alpha*dx||
    // 2 - alpha = 1 => alpha = 1
    assert!(
        (alpha - 1.0).abs() < 1e-10,
        "Expected alpha=1, got {}",
        alpha
    );

    // Verify: s + alpha*ds should be on boundary
    let s_trial: Vec<f64> = s
        .iter()
        .zip(ds.iter())
        .map(|(&si, &dsi)| si + alpha * dsi)
        .collect();
    let t_trial = s_trial[0];
    let x_norm_trial = (s_trial[1] * s_trial[1] + s_trial[2] * s_trial[2]).sqrt();
    eprintln!(
        "  s_trial={:?}, t={}, ||x||={}",
        s_trial, t_trial, x_norm_trial
    );
    assert!((t_trial - x_norm_trial).abs() < 1e-10, "Not on boundary");

    // Test 2: Direction pointing into interior (should give infinity)
    let ds2 = vec![1.0, 0.0, 0.0]; // Increase t
    let alpha2 = cone.step_to_boundary_primal(&s, &ds2);
    eprintln!("Test 2: s={:?}, ds={:?}, alpha={}", s, ds2, alpha2);
    assert!(alpha2 == f64::INFINITY, "Expected infinity, got {}", alpha2);

    // Test 3: Check that result is actually on boundary
    // s = (3, 2, 0), ds = (-2, 1, 0)
    // At boundary: t + alpha*dt = ||x + alpha*dx||
    // 3 - 2*alpha = |2 + alpha| (assuming positive)
    // 3 - 2*alpha = 2 + alpha => 1 = 3*alpha => alpha = 1/3
    let s3 = vec![3.0, 2.0, 0.0];
    let ds3 = vec![-2.0, 1.0, 0.0];
    let alpha3 = cone.step_to_boundary_primal(&s3, &ds3);
    eprintln!("Test 3: s={:?}, ds={:?}, alpha={}", s3, ds3, alpha3);

    // Verify on boundary
    let t3 = s3[0] + alpha3 * ds3[0];
    let x3_norm = ((s3[1] + alpha3 * ds3[1]).powi(2) + (s3[2] + alpha3 * ds3[2]).powi(2)).sqrt();
    eprintln!("  t_trial={}, ||x||_trial={}", t3, x3_norm);
    assert!(
        (t3 - x3_norm).abs() < 1e-10,
        "Not on boundary: t={}, ||x||={}",
        t3,
        x3_norm
    );

    // Test 4: Apply 0.99 fraction and verify still interior
    let alpha_safe = 0.99 * alpha3;
    let t4 = s3[0] + alpha_safe * ds3[0];
    let x4_norm =
        ((s3[1] + alpha_safe * ds3[1]).powi(2) + (s3[2] + alpha_safe * ds3[2]).powi(2)).sqrt();
    eprintln!(
        "Test 4: alpha_safe={}, t={}, ||x||={}",
        alpha_safe, t4, x4_norm
    );
    assert!(t4 > x4_norm, "Point should be interior after 0.99 fraction");

    // Test 5: Exact failing case from solver - t decreasing, x = 0
    // z_pre = [0.000279, 0, 0], dz = [-0.000426, 0, 0]
    // After step with alpha=0.99: t = 0.000279 - 0.99*0.000426 = -0.000143 < 0 (OUTSIDE!)
    let z5 = vec![0.0002792600436585457, 0.0, 0.0];
    let dz5 = vec![-0.0004260934378502775, 0.0, 0.0];

    // Manual calculation of step_to_boundary:
    let t = z5[0];
    let dt = dz5[0];
    let x_norm_sq: f64 = z5[1..].iter().map(|&xi| xi * xi).sum();
    let dx_norm_sq: f64 = dz5[1..].iter().map(|&dxi| dxi * dxi).sum();
    let x_dot_dx: f64 = z5[1..]
        .iter()
        .zip(&dz5[1..])
        .map(|(&xi, &dxi)| xi * dxi)
        .sum();

    let a = dt * dt - dx_norm_sq;
    let b = 2.0 * (t * dt - x_dot_dx);
    let c = t * t - x_norm_sq;

    eprintln!("Test 5: Manual calculation:");
    eprintln!("  t={:.6e}, dt={:.6e}", t, dt);
    eprintln!(
        "  x_norm_sq={:.6e}, dx_norm_sq={:.6e}, x_dot_dx={:.6e}",
        x_norm_sq, dx_norm_sq, x_dot_dx
    );
    eprintln!("  a={:.6e}, b={:.6e}, c={:.6e}", a, b, c);

    let discriminant = b * b - 4.0 * a * c;
    eprintln!("  discriminant={:.6e}", discriminant);

    if discriminant >= 0.0 {
        let sqrt_disc = discriminant.sqrt();
        let alpha1 = (-b - sqrt_disc) / (2.0 * a);
        let alpha2 = (-b + sqrt_disc) / (2.0 * a);
        eprintln!("  sqrt_disc={:.6e}", sqrt_disc);
        eprintln!("  alpha1={:.6e}, alpha2={:.6e}", alpha1, alpha2);

        // t positivity check
        if dt < 0.0 {
            let alpha_t = -t / dt;
            eprintln!("  alpha_t (t positivity)={:.6e}", alpha_t);
        }
    }

    // Step to boundary should be alpha = t / |dt| = 0.000279 / 0.000426 = 0.655
    let alpha5 = cone.step_to_boundary_dual(&z5, &dz5);
    eprintln!("  alpha_boundary (from function)={}", alpha5);

    // Verify the boundary point
    let t5_boundary = z5[0] + alpha5 * dz5[0];
    let x5_boundary_norm =
        ((z5[1] + alpha5 * dz5[1]).powi(2) + (z5[2] + alpha5 * dz5[2]).powi(2)).sqrt();
    eprintln!(
        "  at boundary: t={}, ||x||={}",
        t5_boundary, x5_boundary_norm
    );

    // alpha5 should be around 0.655, NOT infinity
    assert!(alpha5.is_finite(), "alpha should be finite, got {}", alpha5);
    assert!(
        alpha5 < 1.0,
        "alpha should be < 1.0 since step overshoots, got {}",
        alpha5
    );
    assert!(alpha5 > 0.0, "alpha should be positive");

    // Verify: with 0.99 * alpha5, we should be in interior
    let alpha_safe5 = 0.99 * alpha5;
    let t5_safe = z5[0] + alpha_safe5 * dz5[0];
    assert!(t5_safe > 0.0, "t should stay positive with safe step");
}
