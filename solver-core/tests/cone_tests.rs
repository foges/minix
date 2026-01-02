//! Cone kernel unit tests with finite difference validation.
//!
//! This module provides comprehensive testing for all cone implementations,
//! including finite difference checking of gradients and Hessians.

use solver_core::cones::{ConeKernel, ZeroCone, NonNegCone, SocCone};

/// Finite difference tolerance for gradient checking
const FD_GRAD_TOL: f64 = 1e-6;

/// Finite difference tolerance for Hessian checking
const FD_HESS_TOL: f64 = 1e-5;

/// Compute finite difference approximation of gradient.
///
/// Uses central differences: ∂f/∂x_i ≈ (f(x + εe_i) - f(x - εe_i)) / (2ε)
fn finite_diff_gradient<K: ConeKernel>(
    cone: &K,
    s: &[f64],
    grad_fd: &mut [f64],
) {
    let n = s.len();
    let mut s_plus = s.to_vec();
    let mut s_minus = s.to_vec();

    for i in 0..n {
        // Choose ε relative to s[i]
        let eps = 1e-6 * s[i].abs().max(1.0);

        // f(s + εe_i)
        s_plus[i] = s[i] + eps;
        let f_plus = cone.barrier_value(&s_plus);
        s_plus[i] = s[i];  // restore

        // f(s - εe_i)
        s_minus[i] = s[i] - eps;
        let f_minus = cone.barrier_value(&s_minus);
        s_minus[i] = s[i];  // restore

        // Central difference
        grad_fd[i] = (f_plus - f_minus) / (2.0 * eps);
    }
}

/// Compute finite difference approximation of Hessian-vector product.
///
/// Uses central differences: ∇²f(x) v ≈ (∇f(x + εv) - ∇f(x - εv)) / (2ε)
fn finite_diff_hessian_apply<K: ConeKernel>(
    cone: &K,
    s: &[f64],
    v: &[f64],
    hess_v_fd: &mut [f64],
) {
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
        vec![1.0, 0.0, 0.0, 0.0, 0.0],  // unit vector
        vec![1.0, 1.0, 1.0, 1.0, 1.0],  // ones
        vec![0.5, -0.5, 1.0, -1.0, 2.0], // mixed
        vec![1.0, 2.0, 3.0, 4.0, 5.0],  // arbitrary
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
        let s: Vec<f64> = (0..10)
            .map(|_| rng.gen_range(0.1..10.0))
            .collect();

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
        let s: Vec<f64> = (0..10)
            .map(|_| rng.gen_range(0.1..10.0))
            .collect();
        let v: Vec<f64> = (0..10)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

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
        assert!(cone.is_interior_primal(&s), "Test point not interior: {:?}", s);
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
        vec![1.0, 0.0, 0.0, 0.0, 0.0],  // t-direction
        vec![0.0, 1.0, 0.0, 0.0, 0.0],  // x-direction
        vec![1.0, 1.0, 1.0, 1.0, 1.0],  // ones
        vec![0.5, -0.5, 1.0, -1.0, 2.0], // mixed
        vec![2.0, 1.0, 1.0, 1.0, 1.0],  // arbitrary
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
        s[0] = (dim as f64).sqrt() + 1.0;  // Safely interior

        assert!(
            cone.is_interior_primal(&s),
            "Point not interior for dim={}", dim
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
        check_gradient(&cone, &s, FD_GRAD_TOL * 10.0),  // Slightly relaxed tolerance
        "Gradient check failed near axis"
    );
}
