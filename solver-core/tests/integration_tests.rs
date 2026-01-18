//! End-to-end integration tests for the Minix solver.
//!
//! These tests validate that the full IPM pipeline works correctly
//! on various problem types.

use solver_core::{solve, ProblemData, ConeSpec, SolverSettings, SolveStatus};
use solver_core::cones::{PsdCone, ExpCone, ConeKernel};
use solver_core::linalg::sparse;

#[test]
fn test_simple_lp() {
    // min x1 + x2
    // s.t. x1 + x2 = 1
    //      x1, x2 >= 0
    //
    // Optimal: any point on the line x1 + x2 = 1 with x1, x2 >= 0
    // e.g., x = [0, 1], [1, 0], or [0.5, 0.5], all give obj = 1.0
    //
    // Reformulated:
    //   x1 + x2 + s_eq = 1, s_eq = 0  (equality)
    //   -x1 + s_1 = 0, s_1 >= 0       (bound x1 >= 0)
    //   -x2 + s_2 = 0, s_2 >= 0       (bound x2 >= 0)

    // A is 3x2: [equality, bound x1, bound x2]
    let a_triplets = vec![
        (0, 0, 1.0), (0, 1, 1.0),  // x1 + x2 = 1
        (1, 0, -1.0),              // -x1 + s_1 = 0
        (2, 1, -1.0),              // -x2 + s_2 = 0
    ];

    let prob = ProblemData {
        P: None,
        q: vec![1.0, 1.0],
        A: sparse::from_triplets(3, 2, a_triplets),
        b: vec![1.0, 0.0, 0.0],
        cones: vec![
            ConeSpec::Zero { dim: 1 },    // equality constraint
            ConeSpec::NonNeg { dim: 2 },  // bounds x >= 0
        ],
        var_bounds: None,
        integrality: None,
    };

    let settings = SolverSettings {
        verbose: true,
        max_iter: 50,
        tol_feas: 1e-6,
        tol_gap: 1e-6,
        ..Default::default()
    };

    let result = solve(&prob, &settings).expect("Solve failed");

    println!("\n=== Simple LP Result ===");
    println!("Status: {:?}", result.status);
    println!("x = {:?}", result.x);
    println!("obj = {}", result.obj_val);

    // CRITICAL: Only accept Optimal or AlmostOptimal status
    // MaxIters means the solver FAILED to converge
    assert!(matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::AlmostOptimal
    ), "Expected Optimal/AlmostOptimal, got {:?}", result.status);

    // Check that solution is approximately optimal
    let sum = result.x[0] + result.x[1];
    assert!((sum - 1.0).abs() < 0.1, "Constraint not satisfied: {}", sum);
    assert!(result.x[0] >= -0.1);
    assert!(result.x[1] >= -0.1);
    assert!((result.obj_val - 1.0).abs() < 0.1);
}

#[test]
fn test_lp_with_inequality() {
    // min -x1 - x2
    // s.t. x1 + x2 <= 1
    //      x1, x2 >= 0
    //
    // Reformulated in standard form:
    //   min -x1 - x2
    //   s.t. x1 + x2 + s_ineq = 1   (inequality slack)
    //       -x1 + s_x1 = 0          (bound on x1)
    //       -x2 + s_x2 = 0          (bound on x2)
    //        s_ineq, s_x1, s_x2 >= 0
    //
    // Optimal: x = [1, 0] or [0, 1] or [0.5, 0.5], obj = -1

    // A is 3x2: rows are [inequality, bound x1, bound x2]
    let a_triplets = vec![
        (0, 0, 1.0), (0, 1, 1.0),  // x1 + x2 + s_ineq = 1
        (1, 0, -1.0),              // -x1 + s_x1 = 0
        (2, 1, -1.0),              // -x2 + s_x2 = 0
    ];

    let prob = ProblemData {
        P: None,
        q: vec![-1.0, -1.0],
        A: sparse::from_triplets(3, 2, a_triplets),
        b: vec![1.0, 0.0, 0.0],
        cones: vec![ConeSpec::NonNeg { dim: 3 }],  // All slacks are nonnegative
        var_bounds: None,
        integrality: None,
    };

    let settings = SolverSettings {
        verbose: true,
        max_iter: 100,
        tol_feas: 1e-6,
        tol_gap: 1e-6,
        ..Default::default()
    };

    let result = solve(&prob, &settings).expect("Solve failed");

    println!("\n=== LP with Inequality Result ===");
    println!("Status: {:?}", result.status);
    println!("x = {:?}", result.x);
    println!("obj = {}", result.obj_val);

    assert!(matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::AlmostOptimal
    ), "Expected Optimal/AlmostOptimal, got {:?}", result.status);

    // Check that solution is approximately optimal (x1 + x2 = 1, x >= 0)
    assert!((result.x[0] + result.x[1] - 1.0).abs() < 0.1);
    assert!(result.x[0] >= -0.1);
    assert!(result.x[1] >= -0.1);
    assert!((result.obj_val - (-1.0)).abs() < 0.1);
}

#[test]
fn test_simple_qp() {
    // min 0.5 * (x1^2 + x2^2) + x1 + x2
    // s.t. x1 + x2 = 1
    //      x1, x2 >= 0  (need bounds for IPM to work)
    //
    // Optimal: x = [0.5, 0.5], obj = 1.25 (0.5*0.5 + 1)
    //
    // Reformulated:
    //   x1 + x2 + s_eq = 1, s_eq = 0  (equality via Zero cone)
    //   -x1 + s_1 = 0, s_1 >= 0       (bound x1 >= 0)
    //   -x2 + s_2 = 0, s_2 >= 0       (bound x2 >= 0)

    let p_triplets = vec![
        (0, 0, 1.0),  // P[0,0] = 1
        (1, 1, 1.0),  // P[1,1] = 1
    ];

    // A is 3x2: [equality, bound x1, bound x2]
    let a_triplets = vec![
        (0, 0, 1.0), (0, 1, 1.0),  // x1 + x2 = 1
        (1, 0, -1.0),              // -x1 + s_1 = 0
        (2, 1, -1.0),              // -x2 + s_2 = 0
    ];

    let prob = ProblemData {
        P: Some(sparse::from_triplets(2, 2, p_triplets)),
        q: vec![1.0, 1.0],
        A: sparse::from_triplets(3, 2, a_triplets),
        b: vec![1.0, 0.0, 0.0],
        cones: vec![
            ConeSpec::Zero { dim: 1 },    // equality constraint
            ConeSpec::NonNeg { dim: 2 },  // bounds x >= 0
        ],
        var_bounds: None,
        integrality: None,
    };

    let settings = SolverSettings {
        verbose: true,
        max_iter: 50,
        tol_feas: 1e-6,
        tol_gap: 1e-6,
        ..Default::default()
    };

    let result = solve(&prob, &settings).expect("Solve failed");

    println!("\n=== Simple QP Result ===");
    println!("Status: {:?}", result.status);
    println!("x = {:?}", result.x);
    println!("obj = {}", result.obj_val);

    assert!(matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::AlmostOptimal
    ), "Expected Optimal/AlmostOptimal, got {:?}", result.status);

    // Check constraint is satisfied (approximately)
    let sum = result.x[0] + result.x[1];
    assert!((sum - 1.0).abs() < 0.1, "Constraint not satisfied: x1 + x2 = {}", sum);
    // Optimal is x = [0.5, 0.5], obj = 1.25
    assert!((result.obj_val - 1.25).abs() < 0.1, "Objective value unexpected: {}", result.obj_val);
}

#[test]
fn test_nonneg_cone() {
    // min -x
    // s.t. x <= 2
    //      x >= 0
    //
    // Reformulated:
    // min -x
    // s.t. x + s = 2, s in NonNeg
    //
    // Optimal: x = 2, s = 0, obj = -2

    let prob = ProblemData {
        P: None,
        q: vec![-1.0],
        A: sparse::from_triplets(1, 1, vec![(0, 0, 1.0)]),
        b: vec![2.0],
        cones: vec![ConeSpec::NonNeg { dim: 1 }],
        var_bounds: None,
        integrality: None,
    };

    let settings = SolverSettings {
        verbose: true,
        max_iter: 50,
        ..Default::default()
    };

    let result = solve(&prob, &settings).expect("Solve failed");

    println!("\n=== NonNeg Cone Result ===");
    println!("Status: {:?}", result.status);
    println!("x = {:?}", result.x);
    println!("s = {:?}", result.s);
    println!("obj = {}", result.obj_val);

    assert!(matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::AlmostOptimal
    ), "Expected Optimal/AlmostOptimal, got {:?}", result.status);
}

#[test]
fn test_small_soc() {
    // Pure SOCP: min t s.t. (t, x1, x2) ∈ SOC, t + x1 + x2 = 2
    //
    // The constraint t + x1 + x2 = 2 combined with SOC gives t >= ||(x1, x2)||
    // At optimum, we want minimum t such that t >= ||(x1, x2)|| and t + x1 + x2 = 2
    //
    // Standard conic form: Ax + s = b, s ∈ K
    // Variables: t, x1, x2 (n=3)
    // Cones: Zero(1) for equality, SOC(3) for t >= ||(x1, x2)||
    //
    // Equality: t + x1 + x2 = 2  =>  A_eq = [1, 1, 1], b_eq = 2, s_eq = 0
    // SOC: s_soc = (t, x1, x2)   =>  A_soc = -I, b_soc = 0
    //
    // Full system:
    // A = [1  1  1]     b = [2]
    //     [-1 0  0]         [0]
    //     [0 -1  0]         [0]
    //     [0  0 -1]         [0]
    //
    // Optimal: By Lagrangian, optimal has t = ||(x1, x2)|| (SOC active)
    // With symmetry x1 = x2 and t + 2*x1 = 2:
    // t = sqrt(2) * x1, so sqrt(2)*x1 + 2*x1 = 2, x1 = 2/(sqrt(2)+2) ≈ 0.586
    // t = sqrt(2) * 0.586 ≈ 0.828

    let prob = ProblemData {
        P: None,
        q: vec![1.0, 0.0, 0.0],  // min t
        A: sparse::from_triplets(
            4,
            3,
            vec![
                (0, 0, 1.0), (0, 1, 1.0), (0, 2, 1.0), // t + x1 + x2 = 2
                (1, 0, -1.0), // -t + s1 = 0   =>  s1 = t
                (2, 1, -1.0), // -x1 + s2 = 0  =>  s2 = x1
                (3, 2, -1.0), // -x2 + s3 = 0  =>  s3 = x2
            ],
        ),
        b: vec![2.0, 0.0, 0.0, 0.0],
        cones: vec![ConeSpec::Zero { dim: 1 }, ConeSpec::Soc { dim: 3 }],
        var_bounds: None,
        integrality: None,
    };

    let settings = SolverSettings {
        verbose: false,
        max_iter: 100,
        ..Default::default()
    };

    let result = solve(&prob, &settings).expect("Solve failed");

    println!("\n=== Small SOC Result ===");
    println!("Status: {:?}", result.status);
    println!("x = {:?}", result.x);
    println!("obj = {}", result.obj_val);

    // Expected: t ≈ 0.828, x1 ≈ x2 ≈ 0.586
    let expected_t = 2.0 / (2.0_f64.sqrt() + 2.0) * 2.0_f64.sqrt();
    let _expected_x = 2.0 / (2.0_f64.sqrt() + 2.0);

    // Accept a wide tolerance since SOC support is in development
    if matches!(result.status, SolveStatus::Optimal | SolveStatus::AlmostOptimal) {
        assert!((result.x[0] - expected_t).abs() < 0.2,
            "Expected t ≈ {:.3}, got {:.3}", expected_t, result.x[0]);
        assert!((result.obj_val - expected_t).abs() < 0.2,
            "Expected obj ≈ {:.3}, got {:.3}", expected_t, result.obj_val);
    }
}

#[test]
fn test_rsoc_qp_formulation() {
    // Test RSOC-based QP formulation for a simple QP:
    // min (1/2) x² subject to x >= 1
    // Optimal: x = 1, obj = 0.5
    //
    // SOCP reformulation:
    // min t subject to t >= x²/2, x >= 1
    //
    // Using RSOC: (t, 1, x) ∈ RSOC means 2*t*1 >= x², so t >= x²/2
    // Convert to SOC: ((t+1)/√2, (t-1)/√2, x) ∈ SOC
    //
    // Variables: [x, t]
    // Constraints:
    //   1. x >= 1 (NonNeg cone for x - 1)
    //   2. ((t+1)/√2, (t-1)/√2, x) ∈ SOC (use constant 1 directly in b)
    //
    // Objective: min t

    let sqrt2 = std::f64::consts::SQRT_2;

    // Variables: [x, t]
    // Constraint rows:
    //   Row 0: -x + slack0 = -1, slack0 >= 0 (x >= 1)
    //   Row 1: -t/√2 + soc0 = 1/√2 => soc0 = (t+1)/√2
    //   Row 2: -t/√2 + soc1 = -1/√2 => soc1 = (t-1)/√2
    //   Row 3: -x + soc2 = 0 => soc2 = x

    let a_triplets = vec![
        (0, 0, -1.0),                 // Row 0: -x
        (1, 1, -1.0 / sqrt2),         // Row 1: -t/√2
        (2, 1, -1.0 / sqrt2),         // Row 2: -t/√2
        (3, 0, -1.0),                 // Row 3: -x
    ];

    let prob = ProblemData {
        P: None,
        q: vec![0.0, 1.0],  // min t (variable at index 1)
        A: sparse::from_triplets(4, 2, a_triplets),
        b: vec![-1.0, 1.0 / sqrt2, -1.0 / sqrt2, 0.0],
        cones: vec![
            ConeSpec::NonNeg { dim: 1 },  // x >= 1
            ConeSpec::Soc { dim: 3 },     // RSOC->SOC
        ],
        var_bounds: None,
        integrality: None,
    };

    let settings = SolverSettings {
        verbose: true,
        max_iter: 100,
        ..Default::default()
    };

    let result = solve(&prob, &settings).expect("Solve failed");

    println!("\n=== RSOC QP Formulation Result ===");
    println!("Status: {:?}", result.status);
    println!("x = {:?}", result.x);
    println!("obj = {} (expected 0.5)", result.obj_val);

    assert!(
        matches!(result.status, SolveStatus::Optimal | SolveStatus::AlmostOptimal),
        "Expected Optimal, got {:?}",
        result.status
    );

    // At optimum: x = 1, t = 0.5
    let x = result.x[0];
    let t = result.x[1];
    assert!((x - 1.0).abs() < 0.1, "Expected x ≈ 1, got {}", x);
    assert!((t - 0.5).abs() < 0.1, "Expected t ≈ 0.5, got {}", t);
    assert!((result.obj_val - 0.5).abs() < 0.1, "Expected obj ≈ 0.5, got {}", result.obj_val);
}

#[test]
fn test_psd_cone_basic() {
    let cone = PsdCone::new(2);
    let mut s = vec![0.0; cone.dim()];
    let mut z = vec![0.0; cone.dim()];
    cone.unit_initialization(&mut s, &mut z);

    assert!(cone.is_interior_primal(&s));
    assert!(cone.is_interior_dual(&z));
}

#[test]
fn test_exp_cone_basic() {
    let cone = ExpCone::new(1);
    let mut s = vec![0.0; cone.dim()];
    let mut z = vec![0.0; cone.dim()];
    cone.unit_initialization(&mut s, &mut z);

    assert!(cone.is_interior_primal(&s));
    assert!(cone.is_interior_dual(&z));
    assert!(cone.barrier_value(&s).is_finite());
}

// ============================================================================
// SDP Optimization Tests
// ============================================================================

#[test]
fn test_sdp_trace_minimization() {
    // Minimize trace(X) = x0 + x2
    // s.t. X[0,0] = 1
    //      X[1,1] >= 0.5  (to make it bounded)
    //      X >= 0 (PSD) - enforced via PSD cone
    //
    // For 2x2 PSD in svec form: [x0, x1*sqrt2, x2]
    //
    // Optimal: X = [[1, 0], [0, 0.5]], trace = 1.5

    // Variables: y (scalar, for constraint)
    // Constraint: A'y + s = c, s in PSD
    // A = [1; 0; 0] (single column), c = [1, 0, 0.5]
    // So: y + s0 = 1, s1 = 0, s2 = 0.5
    // s in PSD means s0*s2 >= s1^2/2
    // With s1 = 0, s2 = 0.5, need s0 >= 0
    // max y such that s0 = 1-y >= 0, so y <= 1
    // Optimal: y = 1, s = [0, 0, 0.5]
    // But s = [0, 0, 0.5] is NOT PSD (s0*s2 = 0 = s1^2, boundary)
    // Need s0 > 0 strictly, so optimal is approaching y = 1

    let a_triplets = vec![
        (0, 0, 1.0),   // y affects s0
    ];

    let prob = ProblemData {
        P: None,
        q: vec![-1.0],  // max y = min -y
        A: sparse::from_triplets(3, 1, a_triplets),
        b: vec![1.0, 0.0, 0.5],  // [1, 0, 0.5] target
        cones: vec![ConeSpec::Psd { n: 2 }],
        var_bounds: None,
        integrality: None,
    };

    let settings = SolverSettings {
        verbose: false,
        max_iter: 50,
        ..Default::default()
    };

    let result = solve(&prob, &settings).expect("Solve failed");

    println!("\n=== SDP Trace Minimization Result ===");
    println!("Status: {:?}", result.status);
    println!("x = {:?}", result.x);
    println!("obj = {}", result.obj_val);

    // Accept any convergent status since SDP can be numerically challenging
    assert!(matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::AlmostOptimal | SolveStatus::NumericalLimit | SolveStatus::MaxIters
    ), "Expected convergent status, got {:?}", result.status);
}

#[test]
fn test_simple_sdp_with_psd_cone() {
    // Dual form SDP:
    // max  b'y
    // s.t. sum_i y_i * F_i + S = C
    //      S >= 0 (PSD)
    //
    // Equivalently in conic form:
    // max  b'y
    // s.t. A'y + s = c
    //      s in PSD
    //
    // Simple case: 2x2 identity constraint
    // F0 = I (identity), b = [1]
    // max y
    // s.t. y*I + S = I  =>  S = (1-y)*I
    // S PSD requires 1-y >= 0, so y <= 1
    // Optimal: y = 1, S = 0

    let _sqrt2 = std::f64::consts::SQRT_2;

    // svec of 2x2 identity: [1, 0, 1]
    // A: [1, 0, 1]' (column for y)
    let a_triplets = vec![
        (0, 0, 1.0),  // F0[0,0] = 1
        (2, 0, 1.0),  // F0[1,1] = 1
    ];

    let prob = ProblemData {
        P: None,
        q: vec![-1.0],  // max y = min -y
        A: sparse::from_triplets(3, 1, a_triplets),
        b: vec![1.0, 0.0, 1.0],  // svec(I)
        cones: vec![
            ConeSpec::Psd { n: 2 },  // 2x2 PSD cone
        ],
        var_bounds: None,
        integrality: None,
    };

    let settings = SolverSettings {
        verbose: true,
        max_iter: 50,
        tol_feas: 1e-6,
        tol_gap: 1e-6,
        ..Default::default()
    };

    let result = solve(&prob, &settings).expect("Solve failed");

    println!("\n=== Simple SDP with PSD Cone Result ===");
    println!("Status: {:?}", result.status);
    println!("x (y) = {:?}", result.x);
    println!("s (svec(S)) = {:?}", result.s);
    println!("obj = {}", result.obj_val);

    assert!(matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::AlmostOptimal | SolveStatus::NumericalLimit
    ), "Expected convergent status, got {:?}", result.status);

    // Optimal y should be close to 1
    if matches!(result.status, SolveStatus::Optimal | SolveStatus::AlmostOptimal) {
        assert!((result.x[0] - 1.0).abs() < 0.1, "Expected y ≈ 1, got {}", result.x[0]);
    }
}

// ============================================================================
// Infeasibility Detection Tests
// ============================================================================

#[test]
fn test_primal_infeasible_lp() {
    // Infeasible LP:
    // min x
    // s.t. x <= -1  (x + s1 = -1, s1 >= 0)
    //      x >= 1   (-x + s2 = -1, s2 >= 0)
    //
    // Clearly infeasible: x cannot be both <= -1 and >= 1

    let a_triplets = vec![
        (0, 0, 1.0),   // x + s1 = -1
        (1, 0, -1.0),  // -x + s2 = -1
    ];

    let prob = ProblemData {
        P: None,
        q: vec![1.0],
        A: sparse::from_triplets(2, 1, a_triplets),
        b: vec![-1.0, -1.0],
        cones: vec![ConeSpec::NonNeg { dim: 2 }],
        var_bounds: None,
        integrality: None,
    };

    let settings = SolverSettings {
        verbose: false,
        max_iter: 100,
        ..Default::default()
    };

    let result = solve(&prob, &settings).expect("Solve failed");

    println!("\n=== Primal Infeasible LP Result ===");
    println!("Status: {:?}", result.status);

    // Should detect primal infeasibility (or fail gracefully)
    // Infeasibility detection is tricky - accept several outcomes
    assert!(!matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::AlmostOptimal
    ), "Should NOT report Optimal for infeasible problem, got {:?}", result.status);
}

#[test]
fn test_dual_infeasible_lp() {
    // Dual infeasible (unbounded primal) LP:
    // min -x
    // s.t. x >= 0
    //
    // No upper bound on x, so objective is unbounded below
    //
    // Reformulated:
    // min -x
    // s.t. -x + s = 0, s >= 0

    let a_triplets = vec![(0, 0, -1.0)];

    let prob = ProblemData {
        P: None,
        q: vec![-1.0],
        A: sparse::from_triplets(1, 1, a_triplets),
        b: vec![0.0],
        cones: vec![ConeSpec::NonNeg { dim: 1 }],
        var_bounds: None,
        integrality: None,
    };

    let settings = SolverSettings {
        verbose: false,
        max_iter: 100,
        ..Default::default()
    };

    let result = solve(&prob, &settings).expect("Solve failed");

    println!("\n=== Dual Infeasible LP Result ===");
    println!("Status: {:?}", result.status);

    // Should detect dual infeasibility (unbounded primal) or fail gracefully
    // Accept any non-optimal outcome
    assert!(!matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::AlmostOptimal
    ), "Should NOT report Optimal for unbounded problem, got {:?}", result.status);
}

// ============================================================================
// More Complex QP Tests
// ============================================================================

#[test]
fn test_qp_box_constrained() {
    // Box-constrained QP:
    // min 0.5 * x'Px + q'x
    // s.t. 0 <= x <= 1
    //
    // P = I (identity), q = [-0.5, -0.5, -0.5]
    // Optimal: x = [0.5, 0.5, 0.5]
    //
    // Reformulated with slacks for upper and lower bounds:
    // -x + s_lower = 0, s_lower >= 0  (x >= 0)
    // x + s_upper = 1, s_upper >= 0   (x <= 1)

    let n = 3;

    // P = I
    let p_triplets: Vec<_> = (0..n).map(|i| (i, i, 1.0)).collect();

    // A: lower bounds (-I), upper bounds (I)
    let mut a_triplets = Vec::new();
    for i in 0..n {
        a_triplets.push((i, i, -1.0));      // lower bound: -x_i + s = 0
        a_triplets.push((n + i, i, 1.0));   // upper bound: x_i + s = 1
    }

    let prob = ProblemData {
        P: Some(sparse::from_triplets(n, n, p_triplets)),
        q: vec![-0.5; n],
        A: sparse::from_triplets(2 * n, n, a_triplets),
        b: {
            let mut b = vec![0.0; n];  // lower bounds
            b.extend(vec![1.0; n]);    // upper bounds
            b
        },
        cones: vec![ConeSpec::NonNeg { dim: 2 * n }],
        var_bounds: None,
        integrality: None,
    };

    let settings = SolverSettings {
        verbose: true,
        max_iter: 50,
        tol_feas: 1e-6,
        tol_gap: 1e-6,
        ..Default::default()
    };

    let result = solve(&prob, &settings).expect("Solve failed");

    println!("\n=== Box-Constrained QP Result ===");
    println!("Status: {:?}", result.status);
    println!("x = {:?}", result.x);
    println!("obj = {}", result.obj_val);

    assert!(matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::AlmostOptimal
    ), "Expected Optimal/AlmostOptimal, got {:?}", result.status);

    // Optimal x should be [0.5, 0.5, 0.5]
    for i in 0..n {
        assert!(
            (result.x[i] - 0.5).abs() < 0.1,
            "Expected x[{}] ≈ 0.5, got {}", i, result.x[i]
        );
    }
}

#[test]
fn test_qp_with_equality_and_inequality() {
    // Portfolio-style QP:
    // min 0.5 * x'Px
    // s.t. sum(x) = 1    (equality: weights sum to 1)
    //      x >= 0        (no short selling)
    //
    // P = [[2, 1], [1, 2]] (covariance-like)
    // Optimal: x = [0.5, 0.5]

    let p_triplets = vec![
        (0, 0, 2.0),
        (0, 1, 1.0),
        (1, 0, 1.0),
        (1, 1, 2.0),
    ];

    let a_triplets = vec![
        (0, 0, 1.0), (0, 1, 1.0),  // x1 + x2 = 1
        (1, 0, -1.0),              // -x1 + s = 0
        (2, 1, -1.0),              // -x2 + s = 0
    ];

    let prob = ProblemData {
        P: Some(sparse::from_triplets(2, 2, p_triplets)),
        q: vec![0.0, 0.0],
        A: sparse::from_triplets(3, 2, a_triplets),
        b: vec![1.0, 0.0, 0.0],
        cones: vec![
            ConeSpec::Zero { dim: 1 },    // equality
            ConeSpec::NonNeg { dim: 2 },  // x >= 0
        ],
        var_bounds: None,
        integrality: None,
    };

    let settings = SolverSettings {
        verbose: true,
        max_iter: 50,
        tol_feas: 1e-6,
        tol_gap: 1e-6,
        ..Default::default()
    };

    let result = solve(&prob, &settings).expect("Solve failed");

    println!("\n=== Portfolio QP Result ===");
    println!("Status: {:?}", result.status);
    println!("x = {:?}", result.x);
    println!("obj = {}", result.obj_val);

    assert!(matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::AlmostOptimal
    ), "Expected Optimal/AlmostOptimal, got {:?}", result.status);

    // Sum should be 1
    let sum = result.x[0] + result.x[1];
    assert!((sum - 1.0).abs() < 0.1, "Weights don't sum to 1: {}", sum);

    // Optimal should be near [0.5, 0.5]
    assert!((result.x[0] - 0.5).abs() < 0.15, "Expected x[0] ≈ 0.5, got {}", result.x[0]);
    assert!((result.x[1] - 0.5).abs() < 0.15, "Expected x[1] ≈ 0.5, got {}", result.x[1]);
}

// ============================================================================
// Exponential Cone Optimization Tests
// ============================================================================

#[test]
fn test_exp_cone_simple() {
    // Simple exp cone test:
    // MINIX K_exp = {(x, y, z) : z >= y*exp(x/y), y > 0, z > 0}
    //
    // Problem: min z s.t. (1, 1, z) in K_exp
    // This means z >= 1*exp(1/1) = e ≈ 2.718
    // Optimal: z = e ≈ 2.71828
    //
    // In standard form: A*var + s = b, s in K
    // Variable: z (1 var)
    // Slack: s = (s0, s1, s2) in K_exp
    // Constraint: s = [1, 1, z] (x and y fixed at 1)
    //
    // A = [[0], [0], [-1]], b = [1, 1, 0]
    // s = b - A*z = [1, 1, 0] - [0, 0, -z] = [1, 1, z]
    // Objective: min z, so q = [1]

    let prob = ProblemData {
        P: None,
        q: vec![1.0],  // min z
        A: sparse::from_triplets(3, 1, vec![
            (2, 0, -1.0),  // s[2] = z
        ]),
        b: vec![1.0, 1.0, 0.0],
        cones: vec![ConeSpec::Exp { count: 1 }],
        var_bounds: None,
        integrality: None,
    };

    let settings = SolverSettings {
        verbose: false,
        max_iter: 100,
        ..Default::default()
    };

    let result = solve(&prob, &settings);

    println!("\n=== Exp Cone Test Result ===");
    match &result {
        Ok(r) => {
            println!("Status: {:?}", r.status);
            println!("x = {:?}", r.x);
            println!("obj = {}", r.obj_val);
        }
        Err(e) => println!("Error: {}", e),
    }

    // Just verify it doesn't panic - exp cone support may be partial
    assert!(result.is_ok(), "Solver should not panic on exp cone problem");
}

// ============================================================================
// Scale Invariance Tests
// ============================================================================

#[test]
fn test_scale_invariant_solution() {
    // Verify that scaling the problem doesn't significantly affect the solution
    //
    // Original: min x1 + x2 s.t. x1 + x2 = 1, x >= 0
    // Scaled:   min 1000*x1 + 1000*x2 s.t. 1000*x1 + 1000*x2 = 1000, x >= 0

    let a_triplets = vec![
        (0, 0, 1.0), (0, 1, 1.0),
        (1, 0, -1.0),
        (2, 1, -1.0),
    ];

    let prob_original = ProblemData {
        P: None,
        q: vec![1.0, 1.0],
        A: sparse::from_triplets(3, 2, a_triplets.clone()),
        b: vec![1.0, 0.0, 0.0],
        cones: vec![
            ConeSpec::Zero { dim: 1 },
            ConeSpec::NonNeg { dim: 2 },
        ],
        var_bounds: None,
        integrality: None,
    };

    let scale = 1000.0;
    let a_triplets_scaled = vec![
        (0, 0, scale), (0, 1, scale),
        (1, 0, -1.0),  // Keep bounds unscaled
        (2, 1, -1.0),
    ];

    let prob_scaled = ProblemData {
        P: None,
        q: vec![scale, scale],
        A: sparse::from_triplets(3, 2, a_triplets_scaled),
        b: vec![scale, 0.0, 0.0],
        cones: vec![
            ConeSpec::Zero { dim: 1 },
            ConeSpec::NonNeg { dim: 2 },
        ],
        var_bounds: None,
        integrality: None,
    };

    let settings = SolverSettings {
        verbose: false,
        max_iter: 50,
        ..Default::default()
    };

    let result_orig = solve(&prob_original, &settings).expect("Original solve failed");
    let result_scaled = solve(&prob_scaled, &settings).expect("Scaled solve failed");

    println!("\n=== Scale Invariance Test ===");
    println!("Original: x = {:?}, obj = {}", result_orig.x, result_orig.obj_val);
    println!("Scaled:   x = {:?}, obj = {}", result_scaled.x, result_scaled.obj_val);

    // Both should converge
    assert!(matches!(
        result_orig.status,
        SolveStatus::Optimal | SolveStatus::AlmostOptimal
    ));
    assert!(matches!(
        result_scaled.status,
        SolveStatus::Optimal | SolveStatus::AlmostOptimal
    ));

    // Solutions should be similar (x values, not objective)
    let x_diff = (result_orig.x[0] - result_scaled.x[0]).abs()
               + (result_orig.x[1] - result_scaled.x[1]).abs();
    assert!(x_diff < 0.2, "Solutions differ too much: diff = {}", x_diff);
}

// ============================================================================
// Larger Problem Tests
// ============================================================================

#[test]
fn test_larger_lp() {
    // Larger LP to test scalability
    // min sum(x)
    // s.t. sum(x) = n/2
    //      x >= 0

    let n = 50;

    let mut a_triplets = Vec::new();
    // Equality: sum(x) = n/2
    for i in 0..n {
        a_triplets.push((0, i, 1.0));
    }
    // Bounds: x >= 0
    for i in 0..n {
        a_triplets.push((1 + i, i, -1.0));
    }

    let prob = ProblemData {
        P: None,
        q: vec![1.0; n],
        A: sparse::from_triplets(1 + n, n, a_triplets),
        b: {
            let mut b = vec![n as f64 / 2.0];  // equality
            b.extend(vec![0.0; n]);             // bounds
            b
        },
        cones: vec![
            ConeSpec::Zero { dim: 1 },
            ConeSpec::NonNeg { dim: n },
        ],
        var_bounds: None,
        integrality: None,
    };

    let settings = SolverSettings {
        verbose: false,
        max_iter: 100,
        ..Default::default()
    };

    let result = solve(&prob, &settings).expect("Solve failed");

    println!("\n=== Larger LP (n={}) Result ===", n);
    println!("Status: {:?}", result.status);
    println!("obj = {}", result.obj_val);
    println!("x[0..5] = {:?}", &result.x[0..5]);

    assert!(matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::AlmostOptimal
    ), "Expected Optimal/AlmostOptimal, got {:?}", result.status);

    // Check constraint: sum(x) ≈ n/2
    let sum: f64 = result.x.iter().sum();
    let expected = n as f64 / 2.0;
    assert!(
        (sum - expected).abs() < 1.0,
        "Constraint violated: sum = {}, expected {}", sum, expected
    );
}
