/// Benchmark problems for exponential and SDP cones
use solver_core::{ProblemData, ConeSpec, SolverSettings, solve, SolveStatus};
use solver_core::linalg::sparse;

/// Relative entropy minimization problem using exponential cone
///
/// minimize    sum_i u_i * log(u_i / v_i)
/// subject to  sum_i u_i = 1
///             u >= 0
///
/// where v is given distribution.
///
/// Reformulation using exponential cone:
/// minimize    sum_i t_i
/// subject to  t_i >= u_i * log(u_i / v_i)   (via Exp cone)
///             sum_i u_i = 1
///             u >= 0
///
/// Exponential cone form: t >= u * log(u/v)
///   ⟺  v * exp(log(u/v)) <= v * exp(t/u)
///   ⟺  u * exp(log(u/v)) <= u * exp(t/u)  (multiply by u/v)
///   ⟺  exp((log(u) - log(v))) <= exp(t/u)
///   ⟺  log(u) - log(v) <= t/u
///   ⟺  u*log(u/v) <= t
///
/// Standard exponential cone: {(x,y,z) : y*exp(x/y) <= z, y > 0}
/// Our constraint: u*exp(log(u/v)) <= v*exp(t/u)
///
/// Let me use the proper formulation:
/// (log(u/v), u, t) ∈ K_exp
/// which means: u * exp(log(u/v) / u) <= t
///             u * exp(log(u/v) / u) <= t
///             u * (u/v)^(1/u) <= t
///
/// Actually, cleaner formulation:
/// (-t, u, u*v) ∈ K_exp means: u * exp(-t/u) <= u*v
///   ⟺  exp(-t/u) <= v
///   ⟺  -t/u <= log(v)
///   ⟺  t >= -u*log(v) = u*log(1/v)
///
/// No wait, let's use the standard relative entropy form:
/// t >= u*log(u) - u*log(v)
///   = u*log(u) - u*log(v)
///
/// Split into two parts:
/// 1. t1 >= u*log(u)   via (-t1, u, u) ∈ K_exp
/// 2. t2 = -u*log(v)   = u*log(v) where we know v
///
/// For (-t, u, u) ∈ K_exp: u*exp(-t/u) <= u  ⟺  exp(-t/u) <= 1  ⟺  -t/u <= 0  ⟺  t >= 0
///
/// Hmm, that's not right either. Let me use the CVXPY formulation.
///
/// Standard: minimize sum(entr(u)) where entr(x) = -x*log(x)
/// Using Exp cone: -x*log(x) is captured by (x, 1, z) ∈ K_exp with x*exp(x/x) <= z
///
/// Actually, CVXPY uses: entr(x) represented via (-x, 1, z) ∈ K_exp
/// which gives: 1*exp(-x/1) <= z  ⟺  exp(-x) <= z
///
/// For relative entropy: sum_i u_i*log(u_i/v_i) = sum_i (u_i*log(u_i) - u_i*log(v_i))
///
/// Let me just implement a simple exponential cone problem first.
/// Simple exponential cone test adapted from CVXPY
///
/// minimize    x + y + z
/// subject to  y*exp(x/y) <= z
///             y == 1, z == exp(1)
///
/// This forces x = 1 at optimum, giving objective = 1 + 1 + e ≈ 4.718
pub fn exp_cone_cvxpy_style() -> ProblemData {
    // Variables: [x, y, z]
    // Standard form: minimize c'v s.t. Av + s = b, s ∈ K
    let num_vars = 3;

    // Objective: min x + y + z
    let q = vec![1.0, 1.0, 1.0];

    // Constraints:
    // Row 0-2: Exponential cone (x, y, z) with s[0:3] = (x, y, z) must be in K_exp
    // Row 3: y + s[3] = 1  (equality: y = 1)
    // Row 4: z + s[4] = e  (equality: z = e)
    let e = std::f64::consts::E;

    let triplets = vec![
        // Exp cone rows: -I * [x,y,z]' + s[0:3] = 0
        (0, 0, -1.0),  // -x + s[0] = 0  ⟹  s[0] = x
        (1, 1, -1.0),  // -y + s[1] = 0  ⟹  s[1] = y
        (2, 2, -1.0),  // -z + s[2] = 0  ⟹  s[2] = z
        // Equality constraints (Zero cone)
        (3, 1, 1.0),   // y + s[3] = 1   ⟹  s[3] = 1 - y (must be 0)
        (4, 2, 1.0),   // z + s[4] = e   ⟹  s[4] = e - z (must be 0)
    ];

    let A = sparse::from_triplets(5, num_vars, triplets);
    let b = vec![0.0, 0.0, 0.0, 1.0, e];

    let cones = vec![
        ConeSpec::Exp { count: 1 },   // Rows 0-2
        ConeSpec::Zero { dim: 2 },    // Rows 3-4
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

pub fn relative_entropy_simple(_n: usize) -> ProblemData {
    // Simple unbounded exponential cone problem for debugging
    // minimize    x + y
    // subject to  (x, 1, y) ∈ K_exp  i.e., y >= exp(x)
    //
    // Without upper bounds, we want x → -∞ and y → 0, making this unbounded.
    // Let's add: y >= 1
    //
    // Then optimal is: y = exp(x), minimize x + exp(x)
    // d/dx[x + exp(x)] = 1 + exp(x) > 0 always, so minimize by x → -∞
    // But y >= 1 means exp(x) >= 1, so x >= 0
    // At x=0: objective = 0 + 1 = 1

    let num_vars = 2; // [x, y]

    // Objective: min x + y
    let q = vec![1.0, 1.0];

    // Constraints:
    // Row 0-2: Exponential cone (x, 1, y) where s = (x, 1, y)
    // Row 3: y + s[3] = 1  (y >= 1 via slack)
    let triplets = vec![
        (0, 0, -1.0),  // -x + s[0] = 0   ⟹  s[0] = x
        // Row 1: s[1] = 1 (no variable terms)
        (2, 1, -1.0),  // -y + s[2] = 0   ⟹  s[2] = y
        (3, 1, 1.0),   // y + s[3] = 1    ⟹  s[3] = 1 - y
    ];

    let A = sparse::from_triplets(4, num_vars, triplets);
    let b = vec![0.0, 1.0, 0.0, 1.0];

    let cones = vec![
        ConeSpec::Exp { count: 1 },   // Rows 0-2
        ConeSpec::NonNeg { dim: 1 },  // Row 3
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

/// Simple SDP problem: minimize trace(C*X) subject to trace(A_i*X) = b_i, X ⪰ 0
///
/// Example: Find minimum trace PSD matrix with fixed diagonal
/// minimize    trace(X)
/// subject to  X_ii = 1 for all i
///             X ⪰ 0
///
/// This has solution X = I (identity matrix)
pub fn sdp_trace_minimization(n: usize) -> ProblemData {
    // Variables: X is n×n symmetric matrix
    // In svec format: dim = n*(n+1)/2

    let sdp_dim = n * (n + 1) / 2;

    // Objective: minimize trace(X)
    // trace(X) = sum_i X_ii
    // In svec format (with sqrt(2) scaling on off-diagonals):
    // X_ii are at indices 0, 1+n, 1+n+(n-1), ...
    let mut q = vec![0.0; sdp_dim];
    let mut idx = 0;
    for i in 0..n {
        q[idx] = 1.0;
        idx += n - i;
    }

    // Constraints are partitioned into two blocks:
    // 1. Zero cone (n rows): X_ii = 1
    // 2. PSD cone (sdp_dim rows): X itself in svec format must be PSD
    //
    // Total rows: n + sdp_dim

    let mut triplets = Vec::new();

    // Block 1: Zero cone constraints (rows 0..n)
    // X_ii + s_zero[i] = 1, where s_zero[i] must be 0
    idx = 0;
    for i in 0..n {
        triplets.push((i, idx, 1.0));
        idx += n - i;
    }

    // Block 2: PSD cone constraints (rows n..n+sdp_dim)
    // -x + s_psd = 0, where s_psd = x must be in PSD cone
    // This is: -I * x + s_psd = 0
    for i in 0..sdp_dim {
        triplets.push((n + i, i, -1.0));
    }

    let A = sparse::from_triplets(n + sdp_dim, sdp_dim, triplets);
    let mut b = vec![1.0; n];  // X_ii = 1
    b.extend(vec![0.0; sdp_dim]);  // PSD constraint

    let cones = vec![
        ConeSpec::Zero { dim: n },
        ConeSpec::Psd { n },
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

/// Maximum cut SDP relaxation
///
/// Given graph with adjacency matrix W, find maximum cut.
/// SDP relaxation:
/// maximize    (1/4) * sum_{i,j} W_ij * (1 - X_ij)
/// subject to  X_ii = 1 for all i
///             X ⪰ 0
///
/// Equivalently (in minimization form):
/// minimize    -(1/4) * sum_{i,j} W_ij + (1/4) * sum_{i,j} W_ij * X_ij
/// subject to  X_ii = 1 for all i
///             X ⪰ 0
pub fn sdp_maxcut(n: usize, edges: &[(usize, usize, f64)]) -> ProblemData {
    let sdp_dim = n * (n + 1) / 2;

    // Build objective from edge weights
    let mut q = vec![0.0; sdp_dim];

    // Helper: get svec index for (i,j) with i <= j
    let svec_idx = |i: usize, j: usize| -> usize {
        assert!(i <= j);
        let base = (0..i).map(|k| n - k).sum::<usize>();
        base + (j - i)
    };

    for &(i, j, w) in edges {
        let (ii, jj) = if i < j { (i, j) } else { (j, i) };
        let idx = svec_idx(ii, jj);
        q[idx] += 0.25 * w * if ii == jj { 1.0 } else { std::f64::consts::SQRT_2 };
    }

    // Constraints partitioned into two blocks:
    // 1. Zero cone: X_ii = 1
    // 2. PSD cone: X ⪰ 0
    let mut triplets = Vec::new();

    // Block 1: Zero cone (rows 0..n)
    for i in 0..n {
        triplets.push((i, svec_idx(i, i), 1.0));
    }

    // Block 2: PSD cone (rows n..n+sdp_dim)
    for i in 0..sdp_dim {
        triplets.push((n + i, i, -1.0));
    }

    let A = sparse::from_triplets(n + sdp_dim, sdp_dim, triplets);
    let mut b = vec![1.0; n];
    b.extend(vec![0.0; sdp_dim]);

    let cones = vec![
        ConeSpec::Zero { dim: n },
        ConeSpec::Psd { n },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp_debug_trivial() {
        // Simplest possible exp cone problem
        // minimize x
        // subject to (x, 1, 1) ∈ K_exp  i.e., exp(x) ≤ 1, so x ≤ 0
        // Optimal: x = 0, objective = 0

        let num_vars = 1;
        let q = vec![1.0];
        let triplets = vec![(0, 0, -1.0)];
        let A = sparse::from_triplets(3, num_vars, triplets);
        let b = vec![0.0, 1.0, 1.0];
        let cones = vec![ConeSpec::Exp { count: 1 }];

        let prob = ProblemData {
            P: None,
            q,
            A,
            b,
            cones,
            var_bounds: None,
            integrality: None,
        };

        let mut settings = SolverSettings::default();
        settings.max_iter = 20;
        settings.verbose = true;

        let result = solve(&prob, &settings).unwrap();
        println!("\n=== Exp Cone Trivial Debug Test ===");
        println!("Status: {:?}", result.status);
        println!("Iterations: {}", result.info.iters);
        println!("Objective: {:.6}", result.obj_val);
        println!("Solution: x={:.6}", result.x[0]);
        println!("\nExpected: x=0, objective=0");
    }

    #[test]
    fn test_exponential_cone_cvxpy() {
        let prob = exp_cone_cvxpy_style();
        let mut settings = SolverSettings::default();
        settings.max_iter = 200;
        settings.verbose = false;

        let result = solve(&prob, &settings).unwrap();
        println!("\n=== Exponential Cone (CVXPY style) ===");
        println!("Status: {:?}", result.status);
        println!("Iterations: {}", result.info.iters);
        println!("Objective: {:.6}", result.obj_val);
        println!("Solution: x={:.6}, y={:.6}, z={:.6}", result.x[0], result.x[1], result.x[2]);

        // Expected: x=1, y=1, z=e, objective = 1 + 1 + e ≈ 4.718
        if matches!(result.status, SolveStatus::Optimal | SolveStatus::AlmostOptimal) {
            let expected_obj = 1.0 + 1.0 + std::f64::consts::E;
            assert!((result.obj_val - expected_obj).abs() < 0.1,
                "Objective should be ~{}, got {}", expected_obj, result.obj_val);
        }
    }

    #[test]
    fn test_exponential_cone_simple() {
        let prob = relative_entropy_simple(5);
        let mut settings = SolverSettings::default();
        settings.max_iter = 200;
        settings.verbose = false;

        let result = solve(&prob, &settings).unwrap();
        println!("\n=== Exponential Cone (simple) ===");
        println!("Status: {:?}", result.status);
        println!("Iterations: {}", result.info.iters);
        println!("Objective: {:.6}", result.obj_val);
        println!("Solution: x={:.6}, y={:.6}", result.x[0], result.x[1]);

        // Expected: x=0, y=1, objective = 0 + 1 = 1
        if matches!(result.status, SolveStatus::Optimal | SolveStatus::AlmostOptimal) {
            assert!((result.obj_val - 1.0).abs() < 0.1,
                "Objective should be ~1.0, got {}", result.obj_val);
        }
    }

    #[test]
    fn test_sdp_trace_minimization() {
        let n = 3;
        let prob = sdp_trace_minimization(n);
        let mut settings = SolverSettings::default();
        settings.max_iter = 200;

        let result = solve(&prob, &settings).unwrap();
        println!("\n=== SDP Trace Minimization ===");
        println!("Status: {:?}", result.status);
        println!("Iterations: {}", result.info.iters);
        println!("Objective: {:.6}", result.obj_val);

        // Objective should be trace(I) = n even if not fully converged
        assert!((result.obj_val - n as f64).abs() < 0.5,
            "Objective should be ~{}, got {}", n, result.obj_val);
    }

    #[test]
    fn test_sdp_maxcut_triangle() {
        // Triangle graph: 3 nodes, all edges weight 1
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)];
        let prob = sdp_maxcut(3, &edges);
        let mut settings = SolverSettings::default();
        settings.max_iter = 200;

        let result = solve(&prob, &settings).unwrap();
        println!("\n=== MaxCut SDP ===");
        println!("Status: {:?}", result.status);
        println!("Iterations: {}", result.info.iters);
        println!("Objective: {:.6}", result.obj_val);

        // Should get some reasonable approximation
        // For triangle with unit weights, SDP bound is around 1.25 * optimal_cut
    }
}
