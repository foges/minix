//! Benchmarking CLI for minix solver.

mod cbf;
mod cblib;
mod maros_meszaros;
mod matparser;
mod meszaros;
mod netlib;
mod pglib;
mod qplib;
mod qps;

use clap::{Parser, Subcommand};
use solver_core::linalg::sparse;
use solver_core::{solve, ConeSpec, ProblemData, SolverSettings};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "solver-bench")]
#[command(about = "Benchmarking CLI for minix solver")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run random generated benchmarks
    Random {
        /// Maximum iterations
        #[arg(long, default_value = "200")]
        max_iter: usize,
    },
    /// Run Maros-Meszaros QP benchmark suite
    MarosMeszaros {
        /// Maximum number of problems to run (default: all 138)
        #[arg(long)]
        limit: Option<usize>,
        /// Maximum iterations per problem
        #[arg(long, default_value = "200")]
        max_iter: usize,
        /// Run a single problem by name
        #[arg(long)]
        problem: Option<String>,
        /// Show detailed results table
        #[arg(long)]
        table: bool,
    },
    /// Parse and show info about a QPS file
    Info {
        /// Path to QPS file
        path: String,
    },
    /// Run CBLIB conic benchmark suite (SOCP problems)
    Cblib {
        /// Maximum number of problems to run (default: all)
        #[arg(long)]
        limit: Option<usize>,
        /// Maximum iterations per problem
        #[arg(long, default_value = "200")]
        max_iter: usize,
        /// Run a single problem by name
        #[arg(long)]
        problem: Option<String>,
        /// Show detailed results table
        #[arg(long)]
        table: bool,
        /// Run the large problem suite
        #[arg(long)]
        large: bool,
        /// Run Mittelmann "Large SOCP Benchmark" curated subset
        #[arg(long)]
        mittelmann: bool,
        /// Run all problems (standard + large)
        #[arg(long)]
        all: bool,
        /// Enable verbose solver output
        #[arg(long, short)]
        verbose: bool,
    },
    /// Run NETLIB LP benchmark suite
    Netlib {
        /// Maximum number of problems to run (default: all)
        #[arg(long)]
        limit: Option<usize>,
        /// Maximum iterations per problem
        #[arg(long, default_value = "200")]
        max_iter: usize,
        /// Run a single problem by name
        #[arg(long)]
        problem: Option<String>,
        /// Show detailed results table
        #[arg(long)]
        table: bool,
        /// Run full 108-problem extended suite
        #[arg(long)]
        full: bool,
        /// Enable verbose solver output
        #[arg(long, short)]
        verbose: bool,
    },
    /// Run QPLIB QP benchmark suite
    Qplib {
        /// Maximum number of problems to run (default: all)
        #[arg(long)]
        limit: Option<usize>,
        /// Maximum iterations per problem
        #[arg(long, default_value = "200")]
        max_iter: usize,
        /// Run a single problem by name
        #[arg(long)]
        problem: Option<String>,
        /// Show detailed results table
        #[arg(long)]
        table: bool,
    },
    /// Run PGLib-OPF power grid SOCP benchmark suite
    Pglib {
        /// Maximum number of problems to run (default: all)
        #[arg(long)]
        limit: Option<usize>,
        /// Maximum iterations per problem
        #[arg(long, default_value = "200")]
        max_iter: usize,
        /// Run a single problem by name
        #[arg(long)]
        problem: Option<String>,
        /// Show detailed results table
        #[arg(long)]
        table: bool,
        /// Enable verbose solver output
        #[arg(long, short)]
        verbose: bool,
    },
    /// Run Mészáros lptestset benchmark suites (SuiteSparse)
    Meszaros {
        /// Maximum number of problems to run (default: all)
        #[arg(long)]
        limit: Option<usize>,
        /// Maximum iterations per problem
        #[arg(long, default_value = "200")]
        max_iter: usize,
        /// Show detailed results table
        #[arg(long)]
        table: bool,
        /// Run INFEAS suite (infeasibility detection tests)
        #[arg(long)]
        infeas: bool,
        /// Run PROBLEMATIC suite (numerically challenging)
        #[arg(long)]
        problematic: bool,
        /// Enable verbose solver output
        #[arg(long, short)]
        verbose: bool,
    },
}

/// Generate a random LP:
///   minimize    c^T x
///   subject to  Ax = b
///               x >= 0
///
/// where A is m x n, with density `sparsity`.
fn generate_random_lp(n: usize, m: usize, sparsity: f64, seed: u64) -> ProblemData {
    // Simple LCG random number generator
    let mut rng_state = seed;
    let mut rand = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f64) / (u32::MAX as f64)
    };

    // Generate cost vector c (random positive values)
    let q: Vec<f64> = (0..n).map(|_| rand() + 0.1).collect();

    let total_constraints = m + n;

    // Generate A part (m x n with sparsity)
    let mut triplets = Vec::new();
    for i in 0..m {
        for j in 0..n {
            if rand() < sparsity {
                let val = 2.0 * rand() - 1.0;
                triplets.push((i, j, val));
            }
        }
        // Ensure at least one nonzero per row for feasibility
        let j = (rand() * n as f64) as usize;
        let j = j.min(n - 1);
        triplets.push((i, j, rand() + 0.5));
    }

    // Add -I part for bound constraints
    for j in 0..n {
        triplets.push((m + j, j, -1.0));
    }

    let a = sparse::from_triplets(total_constraints, n, triplets);

    // Generate RHS b
    let x_feas: Vec<f64> = (0..n).map(|_| rand() + 0.1).collect();
    let mut b = vec![0.0; total_constraints];

    for col in 0..n {
        if let Some(col_view) = a.outer_view(col) {
            for (row, &val) in col_view.iter() {
                if row < m {
                    b[row] += val * x_feas[col];
                }
            }
        }
    }

    ProblemData {
        P: None,
        q,
        A: a,
        b,
        cones: vec![ConeSpec::Zero { dim: m }, ConeSpec::NonNeg { dim: n }],
        var_bounds: None,
        integrality: None,
    }
}

/// Generate a simple SOCP problem:
///   minimize    c'x
///   subject to  (t, x_rest) ∈ SOC
///               sum(x) = 1
///
/// This is a norm minimization problem: minimize ||x_rest|| with constraints.
fn generate_simple_socp(n: usize, _seed: u64) -> ProblemData {
    // Objective: minimize first variable (the SOC "t" component)
    let mut q = vec![0.0; n];
    q[0] = 1.0;

    // Constraints: sum of remaining variables = 1, plus SOC constraint
    // The SOC constraint is: t >= ||x_rest||, i.e., (t, x_1, ..., x_{n-1}) ∈ SOC
    // We add: -I * x + s = 0 where s is in SOC
    let mut triplets = Vec::new();

    // First row: equality constraint sum(x[1..]) = 1
    for j in 1..n {
        triplets.push((0, j, 1.0));
    }

    // SOC constraint: -I for the SOC block
    for j in 0..n {
        triplets.push((1 + j, j, -1.0));
    }

    let a = sparse::from_triplets(1 + n, n, triplets);

    // RHS: equality = 1, SOC slack = 0
    let mut b = vec![0.0; 1 + n];
    b[0] = 1.0;

    ProblemData {
        P: None,
        q,
        A: a,
        b,
        cones: vec![
            ConeSpec::Zero { dim: 1 }, // equality constraint
            ConeSpec::Soc { dim: n },  // SOC constraint
        ],
        var_bounds: None,
        integrality: None,
    }
}

/// Generate a robust portfolio optimization problem (SOCP).
///
/// Problem: Maximize expected return subject to risk budget.
///   maximize    μ'x
///   subject to  ||Σ^{1/2} x|| <= γ  (risk constraint)
///               1'x = 1             (budget constraint)
///               x >= 0              (no short selling)
///
/// This is a classic SOCP formulation for Markowitz portfolio optimization.
fn generate_portfolio_socp(n: usize, seed: u64) -> ProblemData {
    let mut rng_state = seed;
    let mut rand = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f64) / (u32::MAX as f64)
    };

    // Expected returns (random, between 5% and 15%)
    let mu: Vec<f64> = (0..n).map(|_| -(0.05 + rand() * 0.10)).collect();

    // Risk budget (allow moderate risk)
    let gamma = 0.5;

    // Generate a random covariance factor (for Σ = F'F + D)
    // For simplicity, use diagonal + small correlations
    let num_factors = (n / 5).max(3).min(n);

    // Constraints:
    // 1. Budget: sum(x) = 1
    // 2. Risk: ||y|| <= gamma where y = Σ^{1/2} x (approximated as Fx)
    // 3. Non-negativity: x >= 0

    let m_eq = 1;
    let m_soc = num_factors + 1; // (gamma, Fx) in SOC
    let m_nonneg = n;
    let total_m = m_eq + m_soc + m_nonneg;

    let mut triplets = Vec::new();
    let mut b = vec![0.0; total_m];

    // Budget constraint: sum(x) = 1
    for j in 0..n {
        triplets.push((0, j, 1.0));
    }
    b[0] = 1.0;

    // Risk constraint: (gamma, Fx) in SOC
    // Row 1: -0 + s_0 = gamma  (s_0 = gamma)
    b[m_eq] = gamma;

    // Rows 2..m_eq+num_factors: -F_ij * x_j + s_i = 0
    for i in 0..num_factors {
        for j in 0..n {
            let f_ij = 0.1 * (rand() - 0.5); // Small random factor loading
            if f_ij.abs() > 0.01 {
                triplets.push((m_eq + 1 + i, j, -f_ij));
            }
        }
    }

    // Non-negativity: -x + s = 0, s >= 0
    for j in 0..n {
        triplets.push((m_eq + m_soc + j, j, -1.0));
    }

    let a = sparse::from_triplets(total_m, n, triplets);

    ProblemData {
        P: None,
        q: mu,
        A: a,
        b,
        cones: vec![
            ConeSpec::Zero { dim: m_eq },
            ConeSpec::Soc { dim: m_soc },
            ConeSpec::NonNeg { dim: m_nonneg },
        ],
        var_bounds: None,
        integrality: None,
    }
}

/// Generate a LASSO regression problem (SOCP formulation).
///
/// Problem: min (1/2)||Ax - b||^2 + λ||x||_1
///
/// Reformulated as SOCP:
///   min  t + λ * sum(u)
///   s.t. ||(r, 1)|| <= t + 1   (epigraph of ||r||^2)
///        r = Ax - b
///        -u <= x <= u           (absolute value)
fn generate_lasso_socp(n: usize, m: usize, lambda: f64, seed: u64) -> ProblemData {
    let mut rng_state = seed;
    let mut rand = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f64) / (u32::MAX as f64)
    };

    // Variables: [x (n), u (n), t (1), r (m)]
    let n_vars = n + n + 1 + m; // x, u, t, r
    let x_off = 0;
    let u_off = n;
    let t_off = 2 * n;
    let r_off = 2 * n + 1;

    // Objective: t + λ * sum(u)
    let mut q = vec![0.0; n_vars];
    q[t_off] = 1.0; // minimize t
    for j in 0..n {
        q[u_off + j] = lambda; // + λ * u
    }

    // Constraints:
    // 1. r = Ax - b  (equality, m rows)
    // 2. ||(t+1, r)|| <= t+1  i.e., (t+1, r) in SOC  (need reformulation)
    //    Actually: ||(1, r)||^2 <= (t+1)^2 when t >= ||r||^2/2 - 1/2
    //    Simpler: use (s, r) in SOC where s = sqrt(2t + 1) ... complex
    //    Use standard form: minimize t s.t. ||r||^2 <= 2t
    // 3. x <= u and -x <= u  (2n inequality constraints)

    // Simplified: min t s.t. (t, r) in SOC (||r|| <= t), r = Ax - b
    // This gives min ||Ax-b|| + λ||x||_1

    // Constraints:
    // 1. r - Ax = -b (equality, m rows)
    // 2. (t, r) in SOC (m+1 rows)
    // 3. x - u <= 0 (n rows) : x + s = u, s >= 0
    // 4. -x - u <= 0 (n rows) : -x + s = u, s >= 0

    let m_eq = m; // r = Ax - b
    let m_soc = 1 + m; // (t, r) in SOC
    let m_ineq = 2 * n; // |x| <= u
    let total_m = m_eq + m_soc + m_ineq;

    let mut triplets = Vec::new();
    let mut b_vec = vec![0.0; total_m];

    // Generate random design matrix A and observations
    let true_x: Vec<f64> = (0..n)
        .map(|_| {
            if rand() < 0.1 {
                rand() * 2.0 - 1.0
            } else {
                0.0
            }
        })
        .collect();
    let mut obs = vec![0.0; m];
    for i in 0..m {
        for j in 0..n {
            let a_ij = rand() * 2.0 - 1.0;
            obs[i] += a_ij * true_x[j];
            // Constraint: r_i - a_ij * x_j = -b_i
            triplets.push((i, r_off + i, 1.0)); // r_i
            triplets.push((i, x_off + j, -a_ij)); // -A_ij * x_j
        }
        obs[i] += 0.1 * (rand() - 0.5); // noise
        b_vec[i] = -obs[i]; // = -b
    }

    // SOC: (t, r) in SOC
    // -t + s_0 = 0 => s_0 = t
    triplets.push((m_eq, t_off, -1.0));
    // -r_i + s_{i+1} = 0 => s_{i+1} = r_i
    for i in 0..m {
        triplets.push((m_eq + 1 + i, r_off + i, -1.0));
    }

    // x <= u: x - u + s = 0, s >= 0 => x <= u
    for j in 0..n {
        triplets.push((m_eq + m_soc + j, x_off + j, 1.0));
        triplets.push((m_eq + m_soc + j, u_off + j, -1.0));
    }

    // -x <= u: -x - u + s = 0, s >= 0 => -x <= u
    for j in 0..n {
        triplets.push((m_eq + m_soc + n + j, x_off + j, -1.0));
        triplets.push((m_eq + m_soc + n + j, u_off + j, -1.0));
    }

    let a = sparse::from_triplets(total_m, n_vars, triplets);

    ProblemData {
        P: None,
        q,
        A: a,
        b: b_vec,
        cones: vec![
            ConeSpec::Zero { dim: m_eq },
            ConeSpec::Soc { dim: m_soc },
            ConeSpec::NonNeg { dim: m_ineq },
        ],
        var_bounds: None,
        integrality: None,
    }
}

/// Generate a multi-SOC problem with k SOC constraints on disjoint variable blocks.
///
/// Problem structure:
///   minimize    sum(t_i)  (minimize the SOC "t" components)
///   subject to  sum(x) = 1  (equality constraint)
///               (t_i, x_block_i) ∈ SOC  for each block
///
/// Variables are partitioned into k disjoint SOC blocks.
fn generate_multi_socp(n: usize, k: usize, _seed: u64) -> ProblemData {
    // Each SOC block needs at least 3 dimensions (t + 2 x components)
    let soc_dim = (n / k).max(3);
    let num_socs = (n / soc_dim).min(k).max(1);
    let total_vars = soc_dim * num_socs;

    // Objective: minimize sum of t components (first element of each SOC block)
    let mut q = vec![0.0; total_vars];
    for i in 0..num_socs {
        q[i * soc_dim] = 1.0; // Cost on t_i
    }

    // Constraints:
    // 1. Equality: sum of all x components (not t) = 1
    // 2. SOC: -I * vars + s = 0, s ∈ SOC (for each block)
    let num_eq_rows = 1;
    let num_soc_rows = total_vars;
    let num_rows = num_eq_rows + num_soc_rows;

    let mut triplets = Vec::new();

    // Equality constraint: sum of non-t components = 1
    for block in 0..num_socs {
        for j in 1..soc_dim {
            // Skip t component (index 0 of each block)
            let var_idx = block * soc_dim + j;
            triplets.push((0, var_idx, 1.0));
        }
    }

    // SOC constraints: -I for each variable
    for j in 0..total_vars {
        triplets.push((num_eq_rows + j, j, -1.0));
    }

    let a = sparse::from_triplets(num_rows, total_vars, triplets);

    // RHS: equality = 1, SOC slacks = 0
    let mut b = vec![0.0; num_rows];
    b[0] = 1.0;

    // Cones: one Zero (equality), then k SOC cones
    let mut cones = vec![ConeSpec::Zero { dim: 1 }];
    for _ in 0..num_socs {
        cones.push(ConeSpec::Soc { dim: soc_dim });
    }

    ProblemData {
        P: None,
        q,
        A: a,
        b,
        cones,
        var_bounds: None,
        integrality: None,
    }
}

/// Generate a radial network OPF problem using SOCP relaxation.
///
/// This models a simple radial (tree) network with n buses:
/// - Bus 0 is the substation (reference bus)
/// - Power balance at each bus
/// - Voltage magnitude bounds
/// - Line power flow limits (SOCP constraint)
///
/// SOCP relaxation variables for each line (i,j):
///   P_ij, Q_ij = active/reactive power flow
///   l_ij = squared current magnitude
///   v_i = squared voltage magnitude at bus i
///
/// Key SOCP constraint for each line:
///   ||(2*P_ij, 2*Q_ij, l_ij - v_i)||_2 <= l_ij + v_i
fn generate_power_flow_socp(n_buses: usize, seed: u64) -> ProblemData {
    let mut rng_state = seed;
    let mut rand = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f64) / (u32::MAX as f64)
    };

    // For a radial network with n buses, we have n-1 lines
    let n_lines = n_buses - 1;

    // Variables:
    // - P_ij: active power flow on each line (n-1)
    // - Q_ij: reactive power flow on each line (n-1)
    // - l_ij: squared current on each line (n-1)
    // - v_i: squared voltage at each bus (n)
    // - p_gen: generation at substation (1)
    let n_p = n_lines;
    let n_q = n_lines;
    let n_l = n_lines;
    let n_v = n_buses;
    let n_gen = 1;
    let n_vars = n_p + n_q + n_l + n_v + n_gen;

    // Variable offsets
    let p_off = 0;
    let q_off = n_p;
    let l_off = n_p + n_q;
    let v_off = n_p + n_q + n_l;
    let gen_off = n_p + n_q + n_l + n_v;

    // Objective: minimize generation cost (p_gen)
    let mut q = vec![0.0; n_vars];
    q[gen_off] = 1.0; // minimize total generation

    // Constraints:
    // 1. Power balance at each non-substation bus (equality): n-1 rows
    // 2. Substation power balance (equality): 1 row
    // 3. Voltage limits: v_min <= v_i <= v_max (NonNeg): 2*n rows
    // 4. Line flow SOCP: 4*(n-1) rows (each SOC has dim 4)

    let n_eq = n_buses; // power balance
    let n_vbounds = 2 * n_buses; // voltage bounds
    let n_soc_total = 4 * n_lines; // SOCP constraints

    let total_m = n_eq + n_vbounds + n_soc_total;

    let mut triplets = Vec::new();
    let mut b_vec = vec![0.0; total_m];

    // Generate random loads at each bus (except substation)
    let loads: Vec<f64> = (0..n_buses)
        .map(|i| if i == 0 { 0.0 } else { 0.5 + rand() * 0.5 })
        .collect();

    // Power balance at substation (bus 0):
    // p_gen - sum of outgoing P = 0
    triplets.push((0, gen_off, 1.0));
    // For simplicity, assume bus 0 connects to bus 1
    triplets.push((0, p_off, -1.0)); // P_01 outgoing

    // Power balance at other buses (bus i, i > 0):
    // P_incoming - P_outgoing - load = 0
    for i in 1..n_buses {
        // Incoming power from parent (bus i-1 -> i)
        triplets.push((i, p_off + (i - 1), 1.0));

        // Outgoing power to child (bus i -> i+1) if not leaf
        if i < n_buses - 1 {
            triplets.push((i, p_off + i, -1.0));
        }

        // Load (goes to RHS)
        b_vec[i] = loads[i];
    }

    // Voltage bounds: v_min = 0.9, v_max = 1.1
    // v_i - 0.9 >= 0 => v_i + s1 = 0.9, s1 <= 0 => use v_i - s1 = 0.9, s1 >= 0
    // Actually: v_i >= 0.9 => -v_i + s = -0.9, s >= 0
    // And: v_i <= 1.1 => v_i + s = 1.1, s >= 0
    let v_min = 0.9;
    let v_max = 1.1;
    for i in 0..n_buses {
        // v_i >= v_min: -v_i + s = -v_min
        triplets.push((n_eq + 2 * i, v_off + i, -1.0));
        b_vec[n_eq + 2 * i] = -v_min;

        // v_i <= v_max: v_i + s = v_max
        triplets.push((n_eq + 2 * i + 1, v_off + i, 1.0));
        b_vec[n_eq + 2 * i + 1] = v_max;
    }

    // Reference voltage at substation
    // v_0 = 1.0: already handled by bounds

    // SOCP constraints for each line:
    // ||(2P, 2Q, l - v_parent)|| <= l + v_parent
    // Reformulated as: (l + v_parent, 2P, 2Q, l - v_parent) in SOC
    // s_0 = l + v_parent, s_1 = 2P, s_2 = 2Q, s_3 = l - v_parent
    let soc_base = n_eq + n_vbounds;
    for line in 0..n_lines {
        let parent = line; // Bus i
        let soc_off = soc_base + 4 * line;

        // s_0 = l + v_parent => -l - v_parent + s_0 = 0
        triplets.push((soc_off, l_off + line, -1.0));
        triplets.push((soc_off, v_off + parent, -1.0));

        // s_1 = 2P => -2P + s_1 = 0
        triplets.push((soc_off + 1, p_off + line, -2.0));

        // s_2 = 2Q => -2Q + s_2 = 0
        triplets.push((soc_off + 2, q_off + line, -2.0));

        // s_3 = l - v_parent => -l + v_parent + s_3 = 0
        triplets.push((soc_off + 3, l_off + line, -1.0));
        triplets.push((soc_off + 3, v_off + parent, 1.0));
    }

    let a = sparse::from_triplets(total_m, n_vars, triplets);

    // Cones: Zero (equality), NonNeg (voltage bounds), then SOC for each line
    let mut cones = vec![
        ConeSpec::Zero { dim: n_eq },
        ConeSpec::NonNeg { dim: n_vbounds },
    ];
    for _ in 0..n_lines {
        cones.push(ConeSpec::Soc { dim: 4 });
    }

    ProblemData {
        P: None,
        q,
        A: a,
        b: b_vec,
        cones,
        var_bounds: None,
        integrality: None,
    }
}

/// Generate a portfolio optimization LP
fn generate_portfolio_lp(n: usize, seed: u64) -> ProblemData {
    let mut rng_state = seed;
    let mut rand = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f64) / (u32::MAX as f64)
    };

    let q: Vec<f64> = (0..n).map(|_| -(rand() * 0.2 + 0.05)).collect();

    let mut triplets = Vec::new();

    // Row 0: sum constraint
    for j in 0..n {
        triplets.push((0, j, 1.0));
    }

    // Rows 1..n+1: -I for bounds
    for j in 0..n {
        triplets.push((1 + j, j, -1.0));
    }

    let a = sparse::from_triplets(1 + n, n, triplets);
    let mut b = vec![0.0; 1 + n];
    b[0] = 1.0;

    ProblemData {
        P: None,
        q,
        A: a,
        b,
        cones: vec![ConeSpec::Zero { dim: 1 }, ConeSpec::NonNeg { dim: n }],
        var_bounds: None,
        integrality: None,
    }
}

fn run_benchmark(name: &str, prob: &ProblemData, settings: &SolverSettings) {
    let n = prob.num_vars();
    let m = prob.num_constraints();
    let nnz = prob.A.nnz();

    println!("\n{}", "=".repeat(60));
    println!("{}", name);
    println!("{}", "=".repeat(60));
    println!("Variables (n):    {}", n);
    println!("Constraints (m):  {}", m);
    println!(
        "A nonzeros:       {} ({:.2}% dense)",
        nnz,
        100.0 * nnz as f64 / (n * m) as f64
    );
    println!();

    let start = Instant::now();
    let result = solve(prob, settings);
    let elapsed = start.elapsed();

    match result {
        Ok(res) => {
            println!("Status:           {:?}", res.status);
            println!("Iterations:       {}", res.info.iters);
            println!("Objective:        {:.6e}", res.obj_val);
            println!("Final μ:          {:.6e}", res.info.mu);
            println!("Solve time:       {:.3} ms", elapsed.as_secs_f64() * 1000.0);
            println!(
                "Time/iteration:   {:.3} ms",
                elapsed.as_secs_f64() * 1000.0 / res.info.iters as f64
            );
        }
        Err(e) => {
            println!("ERROR: {}", e);
        }
    }
}

fn run_random_benchmarks(max_iter: usize) {
    println!("Minix Solver Benchmarks");
    println!("=======================\n");

    let settings = SolverSettings {
        verbose: false,
        max_iter,
        tol_feas: 1e-6,
        tol_gap: 1e-6,
        ..Default::default()
    };

    // Portfolio LPs
    let prob = generate_portfolio_lp(50, 12345);
    run_benchmark("Portfolio LP (n=50)", &prob, &settings);

    let prob = generate_portfolio_lp(200, 12345);
    run_benchmark("Portfolio LP (n=200)", &prob, &settings);

    let prob = generate_portfolio_lp(500, 12345);
    run_benchmark("Portfolio LP (n=500)", &prob, &settings);

    // Random LPs
    let prob = generate_random_lp(100, 50, 0.3, 12345);
    run_benchmark("Random LP (n=100, m=50, 30% dense)", &prob, &settings);

    let prob = generate_random_lp(500, 200, 0.1, 12345);
    run_benchmark("Random LP (n=500, m=200, 10% dense)", &prob, &settings);

    let prob = generate_random_lp(1000, 500, 0.05, 12345);
    run_benchmark("Random LP (n=1000, m=500, 5% dense)", &prob, &settings);

    // SOCP benchmarks
    let prob = generate_simple_socp(10, 12345);
    run_benchmark("Simple SOCP (n=10, 1 SOC)", &prob, &settings);

    let prob = generate_simple_socp(50, 12345);
    run_benchmark("Simple SOCP (n=50, 1 SOC)", &prob, &settings);

    let prob = generate_multi_socp(100, 10, 12345);
    run_benchmark("Multi SOCP (n=100, 10 SOCs)", &prob, &settings);

    // Portfolio optimization (SOCP)
    let prob = generate_portfolio_socp(50, 12345);
    run_benchmark("Portfolio SOCP (n=50 assets)", &prob, &settings);

    let prob = generate_portfolio_socp(200, 12345);
    run_benchmark("Portfolio SOCP (n=200 assets)", &prob, &settings);

    // LASSO regression (SOCP)
    let prob = generate_lasso_socp(20, 100, 0.1, 12345);
    run_benchmark("LASSO SOCP (n=20, m=100)", &prob, &settings);

    let prob = generate_lasso_socp(50, 200, 0.1, 12345);
    run_benchmark("LASSO SOCP (n=50, m=200)", &prob, &settings);

    // Power flow OPF SOCP (radial network)
    let prob = generate_power_flow_socp(10, 12345);
    run_benchmark("OPF SOCP (10-bus radial)", &prob, &settings);

    let prob = generate_power_flow_socp(30, 12345);
    run_benchmark("OPF SOCP (30-bus radial)", &prob, &settings);

    let prob = generate_power_flow_socp(100, 12345);
    run_benchmark("OPF SOCP (100-bus radial)", &prob, &settings);

    println!("\n{}", "=".repeat(60));
    println!("Benchmarks complete");
    println!("{}", "=".repeat(60));
}

fn run_maros_meszaros(
    limit: Option<usize>,
    max_iter: usize,
    problem: Option<String>,
    show_table: bool,
) {
    let settings = SolverSettings {
        verbose: false,
        max_iter,
        tol_feas: 1e-8,
        tol_gap: 1e-8,
        ..Default::default()
    };

    if let Some(name) = problem {
        // Run single problem
        println!("Running single problem: {}", name);
        let result = maros_meszaros::run_single(&name, &settings);

        if let Some(err) = &result.error {
            println!("Error: {}", err);
        } else {
            println!("Status:     {:?}", result.status);
            println!("Variables:  {}", result.n);
            println!("Constraints:{}", result.m);
            println!("Iterations: {}", result.iterations);
            println!("Objective:  {:.6e}", result.obj_val);
            println!("Final μ:    {:.6e}", result.mu);
            println!("Time:       {:.3} ms", result.solve_time_ms);
        }
    } else {
        // Run full suite
        println!("Running Maros-Meszaros QP Benchmark Suite");
        println!("=========================================\n");

        let results = maros_meszaros::run_full_suite(&settings, limit);
        let summary = maros_meszaros::compute_summary(&results);

        if show_table {
            maros_meszaros::print_results_table(&results);
        }

        maros_meszaros::print_summary(&summary);
    }
}

fn show_qps_info(path: &str) {
    match qps::parse_qps(path) {
        Ok(qps) => {
            println!("QPS Problem: {}", qps.name);
            println!("Variables:   {}", qps.n);
            println!("Constraints: {}", qps.m);
            println!("Q nonzeros:  {}", qps.p_triplets.len());
            println!("A nonzeros:  {}", qps.a_triplets.len());

            println!("\nVariable bounds:");
            for (i, name) in qps.var_names.iter().enumerate().take(5) {
                println!("  {}: [{}, {}]", name, qps.var_lower[i], qps.var_upper[i]);
            }
            if qps.n > 5 {
                println!("  ... ({} more)", qps.n - 5);
            }

            println!("\nConstraint bounds:");
            for (i, name) in qps.con_names.iter().enumerate().take(5) {
                println!("  {}: [{}, {}]", name, qps.con_lower[i], qps.con_upper[i]);
            }
            if qps.m > 5 {
                println!("  ... ({} more)", qps.m - 5);
            }

            // Try converting to conic form
            match qps.to_problem_data() {
                Ok(prob) => {
                    println!("\nConic form:");
                    println!("  Variables:   {}", prob.num_vars());
                    println!("  Constraints: {}", prob.num_constraints());
                    println!("  Cones:       {:?}", prob.cones);
                }
                Err(e) => {
                    println!("\nFailed to convert to conic form: {}", e);
                }
            }
        }
        Err(e) => {
            eprintln!("Error parsing QPS file: {}", e);
        }
    }
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Random { max_iter }) => {
            run_random_benchmarks(max_iter);
        }
        Some(Commands::MarosMeszaros {
            limit,
            max_iter,
            problem,
            table,
        }) => {
            run_maros_meszaros(limit, max_iter, problem, table);
        }
        Some(Commands::Info { path }) => {
            show_qps_info(&path);
        }
        Some(Commands::Cblib {
            limit,
            max_iter,
            problem,
            table,
            large,
            mittelmann,
            all,
            verbose,
        }) => {
            run_cblib(
                limit, max_iter, problem, table, large, mittelmann, all, verbose,
            );
        }
        Some(Commands::Netlib {
            limit,
            max_iter,
            problem,
            table,
            full,
            verbose,
        }) => {
            run_netlib(limit, max_iter, problem, table, full, verbose);
        }
        Some(Commands::Qplib {
            limit,
            max_iter,
            problem,
            table,
        }) => {
            run_qplib(limit, max_iter, problem, table);
        }
        Some(Commands::Pglib {
            limit,
            max_iter,
            problem,
            table,
            verbose,
        }) => {
            run_pglib(limit, max_iter, problem, table, verbose);
        }
        Some(Commands::Meszaros {
            limit,
            max_iter,
            table,
            infeas,
            problematic,
            verbose,
        }) => {
            run_meszaros(limit, max_iter, table, infeas, problematic, verbose);
        }
        None => {
            // Default: run random benchmarks
            run_random_benchmarks(200);
        }
    }
}

fn run_qplib(limit: Option<usize>, max_iter: usize, problem: Option<String>, show_table: bool) {
    let settings = SolverSettings {
        verbose: false,
        max_iter,
        tol_feas: 1e-8,
        tol_gap: 1e-8,
        ..Default::default()
    };

    if let Some(name) = problem {
        // Run single problem
        println!("Running QPLIB problem: {}", name);
        let result = qplib::run_single(&name, &settings);

        if let Some(err) = &result.error {
            println!("Error: {}", err);
        } else {
            println!("Status:     {:?}", result.status);
            println!("Variables:  {}", result.n);
            println!("Constraints:{}", result.m);
            println!("Nonzeros:   {}", result.nnz);
            println!("Iterations: {}", result.iterations);
            println!("Objective:  {:.6e}", result.obj_val);
            println!("Final μ:    {:.6e}", result.mu);
            println!("Time:       {:.3} ms", result.solve_time_ms);
        }
    } else {
        // Run full suite
        println!("Running QPLIB QP Benchmark Suite");
        println!("================================\n");

        let results = qplib::run_full_suite(&settings, limit);
        let summary = qplib::compute_summary(&results);

        if show_table {
            qplib::print_results_table(&results);
        }

        qplib::print_summary(&summary);
    }
}

fn run_netlib(
    limit: Option<usize>,
    max_iter: usize,
    problem: Option<String>,
    show_table: bool,
    full: bool,
    verbose: bool,
) {
    let settings = SolverSettings {
        verbose,
        max_iter,
        tol_feas: 1e-8,
        tol_gap: 1e-8,
        ..Default::default()
    };

    if let Some(name) = problem {
        // Run single problem
        println!("Running NETLIB problem: {}", name);
        let result = netlib::run_single(&name, &settings);

        if let Some(err) = &result.error {
            println!("Error: {}", err);
        } else {
            println!("Status:     {:?}", result.status);
            println!("Variables:  {}", result.n);
            println!("Constraints:{}", result.m);
            println!("Nonzeros:   {}", result.nnz);
            println!("Iterations: {}", result.iterations);
            println!("Objective:  {:.6e}", result.obj_val);
            println!("Final μ:    {:.6e}", result.mu);
            println!("Time:       {:.3} ms", result.solve_time_ms);
        }
    } else {
        // Run suite
        let suite_name = if full {
            "NETLIB LP Extended Benchmark Suite (108 problems)"
        } else {
            "NETLIB LP Classic Benchmark Suite (17 problems)"
        };
        println!("Running {}", suite_name);
        println!("{}\n", "=".repeat(suite_name.len() + 8));

        let results = if full {
            netlib::run_extended_suite(&settings, limit)
        } else {
            netlib::run_full_suite(&settings, limit)
        };
        let summary = netlib::compute_summary(&results);

        if show_table {
            netlib::print_results_table(&results);
        }

        netlib::print_summary(&summary);
    }
}

fn run_cblib(
    limit: Option<usize>,
    max_iter: usize,
    problem: Option<String>,
    show_table: bool,
    large: bool,
    mittelmann: bool,
    all: bool,
    verbose: bool,
) {
    let settings = SolverSettings {
        verbose,
        max_iter,
        tol_feas: 1e-8,
        tol_gap: 1e-8,
        ..Default::default()
    };

    if let Some(name) = problem {
        // Run single problem
        println!("Running CBLIB problem: {}", name);
        let result = cblib::run_single(&name, &settings);

        if let Some(err) = &result.error {
            println!("Error: {}", err);
        } else {
            println!("Status:     {:?}", result.status);
            println!("Variables:  {}", result.n);
            println!("Constraints:{}", result.m);
            println!("Nonzeros:   {}", result.nnz);
            println!("Iterations: {}", result.iterations);
            println!("Objective:  {:.6e}", result.obj_val);
            println!("Final μ:    {:.6e}", result.mu);
            println!("Time:       {:.3} ms", result.solve_time_ms);
        }
    } else {
        // Run suite
        let suite_name = if all {
            "CBLIB SOCP Complete Benchmark Suite (Standard + Large + Mittelmann)"
        } else if mittelmann {
            "CBLIB Mittelmann Large SOCP Benchmark"
        } else if large {
            "CBLIB SOCP Large Benchmark Suite"
        } else {
            "CBLIB SOCP Benchmark Suite"
        };

        println!("Running {}", suite_name);
        println!("{}\n", "=".repeat(suite_name.len() + 8));

        let results = if all {
            cblib::run_complete_suite(&settings, limit)
        } else if mittelmann {
            cblib::run_mittelmann_suite(&settings, limit)
        } else if large {
            cblib::run_large_suite(&settings, limit)
        } else {
            cblib::run_full_suite(&settings, limit)
        };
        let summary = cblib::compute_summary(&results);

        if show_table {
            cblib::print_results_table(&results);
        }

        cblib::print_summary(&summary);
    }
}

fn run_pglib(
    limit: Option<usize>,
    max_iter: usize,
    problem: Option<String>,
    show_table: bool,
    verbose: bool,
) {
    let settings = SolverSettings {
        verbose,
        max_iter,
        tol_feas: 1e-8,
        tol_gap: 1e-8,
        ..Default::default()
    };

    if let Some(name) = problem {
        // Run single problem
        println!("Running PGLib-OPF problem: {}", name);
        let result = pglib::run_single(&name, &settings);

        if let Some(err) = &result.error {
            println!("Error: {}", err);
        } else {
            println!("Status:     {:?}", result.status);
            println!(
                "Network:    {} buses, {} gens, {} branches",
                result.n_buses, result.n_gens, result.n_branches
            );
            println!("Variables:  {}", result.n);
            println!("Constraints:{}", result.m);
            println!("Nonzeros:   {}", result.nnz);
            println!("Iterations: {}", result.iterations);
            println!("Objective:  {:.6e}", result.obj_val);
            println!("Final μ:    {:.6e}", result.mu);
            println!("Time:       {:.3} ms", result.solve_time_ms);
        }
    } else {
        // Run full suite
        println!("Running PGLib-OPF SOCP Benchmark Suite");
        println!("======================================\n");

        let results = pglib::run_full_suite(&settings, limit);
        let summary = pglib::compute_summary(&results);

        if show_table {
            pglib::print_results_table(&results);
        }

        pglib::print_summary(&summary);
    }
}

fn run_meszaros(
    limit: Option<usize>,
    max_iter: usize,
    show_table: bool,
    infeas: bool,
    problematic: bool,
    verbose: bool,
) {
    let settings = SolverSettings {
        verbose,
        max_iter,
        tol_feas: 1e-8,
        tol_gap: 1e-8,
        ..Default::default()
    };

    // Default to INFEAS if no suite specified
    let run_infeas = infeas || (!infeas && !problematic);
    let run_problematic = problematic;

    if run_infeas {
        println!("Running Mészáros INFEAS Suite (Infeasibility Detection)");
        println!("=======================================================\n");

        let results = meszaros::run_infeas_suite(&settings, limit);
        let summary = meszaros::compute_summary(&results);

        if show_table {
            meszaros::print_results_table(&results);
        }

        meszaros::print_summary(&summary, "INFEAS");
    }

    if run_problematic {
        if run_infeas {
            println!("\n");
        }
        println!("Running Mészáros PROBLEMATIC Suite (Numerically Challenging)");
        println!("============================================================\n");

        let results = meszaros::run_problematic_suite(&settings, limit);
        let summary = meszaros::compute_summary(&results);

        if show_table {
            meszaros::print_results_table(&results);
        }

        meszaros::print_summary(&summary, "PROBLEMATIC");
    }
}
