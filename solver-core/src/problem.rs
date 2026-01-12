//! Problem data structures and validation.
//!
//! This module defines the canonical optimization problem representation
//! and all associated types.

use std::fmt;

/// Sparse symmetric matrix in CSC format (upper triangle only).
///
/// For a positive semidefinite matrix P, we store only the upper triangular part
/// to save memory and ensure consistency.
pub type SparseSymmetricCsc = sprs::CsMatI<f64, usize>;

/// Sparse matrix in CSC format.
pub type SparseCsc = sprs::CsMatI<f64, usize>;

/// Optimization problem in canonical form.
///
/// The solver works with the canonical formulation:
///
/// ```text
/// minimize    (1/2) x^T P x + q^T x
/// subject to  A x + s = b
///             s ∈ K
/// ```
///
/// where K is a Cartesian product of cones.
///
/// # Dimensions
///
/// - `n`: number of primal variables (length of x)
/// - `m`: number of constraints (length of b, number of rows in A)
/// - P: n × n (optional, PSD)
/// - q: n
/// - A: m × n
/// - b: m
/// - s, z: m (partitioned by cones)
#[derive(Debug, Clone)]
#[allow(non_snake_case)]  // P and A are standard mathematical notation
pub struct ProblemData {
    /// Quadratic cost matrix P (n × n, PSD, upper triangle in CSC).
    /// If None, this is a linear program.
    pub P: Option<SparseSymmetricCsc>,

    /// Linear cost vector q (length n)
    pub q: Vec<f64>,

    /// Constraint matrix A (m × n, CSC format)
    pub A: SparseCsc,

    /// Constraint right-hand side b (length m)
    pub b: Vec<f64>,

    /// Cone specifications partitioning the m-dimensional slack/dual space
    pub cones: Vec<ConeSpec>,

    /// Optional variable bounds (can be represented via cone constraints)
    pub var_bounds: Option<Vec<VarBound>>,

    /// Optional integrality constraints for mixed-integer problems
    pub integrality: Option<Vec<VarType>>,
}

/// Cone specification.
///
/// Each cone type corresponds to a block in the Cartesian product K = K₁ × K₂ × ... × Kₙ.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]  // Enum variant fields are self-documenting
pub enum ConeSpec {
    /// Zero cone: {0}^dim (equality constraints).
    /// No barrier, treated specially in KKT system.
    Zero { dim: usize },

    /// Nonnegative orthant: ℝ₊^dim
    NonNeg { dim: usize },

    /// Second-order (Lorentz) cone: {(t, x) : t ≥ ||x||₂}
    /// Dimension must be at least 2.
    Soc { dim: usize },

    /// Positive semidefinite cone: S₊^n (n × n symmetric matrices)
    /// Stored in svec format: dimension = n(n+1)/2
    Psd { n: usize },

    /// Exponential cone: closure{(x,y,z) : y > 0, y exp(x/y) ≤ z}
    /// Always 3D per block; `count` specifies number of blocks
    Exp { count: usize },

    /// 3D power cone: {(x,y,z) : x ≥ 0, y ≥ 0, x^α y^(1-α) ≥ |z|}
    /// Each cone has its own α ∈ (0,1)
    Pow { cones: Vec<Pow3D> },
}

/// 3D power cone with parameter α ∈ (0,1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pow3D {
    /// Exponent parameter, must be in (0, 1)
    pub alpha: f64,
}

/// Variable bound specification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarBound {
    /// Variable index
    pub var: usize,
    /// Lower bound (None = -∞)
    pub lower: Option<f64>,
    /// Upper bound (None = +∞)
    pub upper: Option<f64>,
}

/// Variable type for mixed-integer problems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarType {
    /// Continuous variable
    Continuous,
    /// Integer variable
    Integer,
    /// Binary variable (0 or 1)
    Binary,
}

/// Optional warm-start data (unscaled, original problem coordinates).
#[derive(Debug, Clone, Default)]
pub struct WarmStart {
    /// Primal variables x (length n)
    pub x: Option<Vec<f64>>,
    /// Slack variables s (length m)
    pub s: Option<Vec<f64>>,
    /// Dual variables z (length m)
    pub z: Option<Vec<f64>>,
    /// Homogenization variable tau (optional)
    pub tau: Option<f64>,
    /// Dual homogenization variable kappa (optional)
    pub kappa: Option<f64>,
}

/// Solver settings and parameters.
#[derive(Debug, Clone)]
pub struct SolverSettings {
    /// Maximum number of IPM iterations
    pub max_iter: usize,

    /// Time limit in milliseconds (None = no limit)
    pub time_limit_ms: Option<u64>,

    /// Enable verbose logging
    pub verbose: bool,

    /// Primal/dual feasibility tolerance
    pub tol_feas: f64,

    /// Duality gap tolerance
    pub tol_gap: f64,

    /// Infeasibility detection tolerance
    pub tol_infeas: f64,

    /// Number of Ruiz equilibration iterations
    pub ruiz_iters: usize,

    /// Static regularization for KKT system (added to diagonal)
    pub static_reg: f64,

    /// Minimum pivot threshold for dynamic regularization
    pub dynamic_reg_min_pivot: f64,

    /// Iterative refinement steps for KKT solves
    pub kkt_refine_iters: usize,

    /// Minimum feasibility weight for combined-step RHS (0 = pure (1-σ))
    pub feas_weight_floor: f64,

    /// Multiple centrality correction iterations
    pub mcc_iters: usize,

    /// Centrality lower bound (sᵢ zᵢ >= β μ)
    pub centrality_beta: f64,

    /// Centrality upper bound (sᵢ zᵢ <= γ μ)
    pub centrality_gamma: f64,

    /// Maximum centering parameter σ (cap for combined step)
    pub sigma_max: f64,

    /// Max backtracking steps for centrality line search
    pub line_search_max_iters: usize,

    /// Optional warm-start values for repeated solves
    pub warm_start: Option<WarmStart>,

    /// Use direct solve mode (τ=1, κ=0) instead of full HSDE.
    /// Faster for well-posed problems but loses infeasibility detection.
    /// Falls back to HSDE automatically if divergence detected.
    pub direct_mode: bool,

    /// Enable constraint conditioning (detect and fix ill-conditioned rows).
    /// Helps with problems that have nearly-parallel constraints or extreme coefficient ratios.
    /// None = use default (true), Some(false) = disable.
    pub enable_conditioning: Option<bool>,

    /// Use proximity-based step size control to keep iterates near central path.
    /// This can reduce iteration count at the cost of more step size computation.
    /// Experimental feature - may help with exponential cones.
    pub use_proximity_step_control: bool,

    /// Proximal regularization strength for free variables.
    /// Adds P[j,j] += rho for variables with zero A-column and zero q[j].
    /// Helps stabilize Newton system for degenerate SDPs. Set to 0 to disable.
    pub proximal_rho: f64,

    /// Enable chordal decomposition for sparse SDPs.
    /// Decomposes large PSD cones into smaller overlapping ones based on sparsity.
    /// None = auto (enabled for PSD cones >= 10), Some(false) = disable.
    pub chordal_decomp: Option<bool>,
}

impl Default for SolverSettings {
    fn default() -> Self {
        // Allow environment variable override for refinement iterations
        // Default to 10 (matching CLARABEL) for better accuracy on ill-conditioned problems
        let kkt_refine_iters = std::env::var("MINIX_REFINE_ITERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(10);

        // Allow environment variable override for Ruiz scaling iterations
        // Set MINIX_RUIZ_ITERS=0 to disable scaling entirely
        let ruiz_iters = std::env::var("MINIX_RUIZ_ITERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(10);
        let mcc_iters = std::env::var("MINIX_MCC_ITERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);

        Self {
            max_iter: 100,
            time_limit_ms: None,
            verbose: false,
            tol_feas: 1e-8,    // Match Clarabel/OSQP industry standard
            tol_gap: 1e-8,     // Match Clarabel/OSQP industry standard
            tol_infeas: 1e-8,  // Infeasibility detection tolerance
            ruiz_iters,
            static_reg: 1e-8,
            dynamic_reg_min_pivot: 1e-13,
            kkt_refine_iters,
            feas_weight_floor: 0.05,
            mcc_iters,
            centrality_beta: 0.1,
            centrality_gamma: 10.0,
            sigma_max: 0.999,
            line_search_max_iters: 0,
            warm_start: None,
            // Allow environment variable to enable direct mode (no HSDE tau/kappa)
            direct_mode: std::env::var("MINIX_DIRECT_MODE")
                .ok()
                .map(|s| s == "1")
                .unwrap_or(false),
            enable_conditioning: None,  // Defaults to true
            use_proximity_step_control: false,  // Experimental, opt-in
            // Proximal regularization for free variables (zero A-column + zero q)
            // Set MINIX_PROXIMAL_RHO=0.0 to disable, or a positive value like 1e-4
            proximal_rho: std::env::var("MINIX_PROXIMAL_RHO")
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(1e-4),  // Default: enabled with rho=1e-4
            // Chordal decomposition for sparse SDPs - auto-enable by default
            chordal_decomp: std::env::var("MINIX_CHORDAL")
                .ok()
                .map(|s| s != "0" && s.to_lowercase() != "false"),
        }
    }
}

/// Solution status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveStatus {
    /// Optimal solution found
    Optimal,

    /// Almost optimal - meets reduced accuracy thresholds (like Clarabel)
    /// Tolerances: gap=5e-5, feas=1e-4 (vs strict: gap=1e-8, feas=1e-8)
    AlmostOptimal,

    /// Primal problem is infeasible (certificate available)
    PrimalInfeasible,

    /// Dual problem is infeasible (certificate available)
    DualInfeasible,

    /// Problem is unbounded (dual infeasible implies primal unbounded)
    Unbounded,

    /// Maximum iterations reached
    MaxIters,

    /// Time limit reached
    TimeLimit,

    /// Numerical error encountered
    NumericalError,

    /// Numerical precision limit reached (primal+gap OK, dual at precision floor)
    /// This indicates the dual residual floor is due to ill-conditioning and
    /// catastrophic cancellation, not an algorithmic issue.
    NumericalLimit,
}

impl fmt::Display for SolveStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolveStatus::Optimal => write!(f, "Optimal"),
            SolveStatus::AlmostOptimal => write!(f, "AlmostOptimal"),
            SolveStatus::PrimalInfeasible => write!(f, "Primal Infeasible"),
            SolveStatus::DualInfeasible => write!(f, "Dual Infeasible"),
            SolveStatus::Unbounded => write!(f, "Unbounded"),
            SolveStatus::MaxIters => write!(f, "MaxIters"),
            SolveStatus::TimeLimit => write!(f, "Time Limit"),
            SolveStatus::NumericalError => write!(f, "Numerical Error"),
            SolveStatus::NumericalLimit => write!(f, "NumericalLimit"),
        }
    }
}

/// Solve result with solution and diagnostics.
#[derive(Debug, Clone)]
pub struct SolveResult {
    /// Solution status
    pub status: SolveStatus,

    /// Primal solution x (length n, unscaled)
    pub x: Vec<f64>,

    /// Slack variables s (length m, unscaled)
    pub s: Vec<f64>,

    /// Dual variables z (length m, unscaled)
    pub z: Vec<f64>,

    /// Objective value at solution
    pub obj_val: f64,

    /// Detailed solve information and diagnostics
    pub info: SolveInfo,
}

/// Detailed solve information and diagnostics.
#[derive(Debug, Clone)]
pub struct SolveInfo {
    /// Number of IPM iterations completed
    pub iters: usize,

    /// Total solve time (milliseconds)
    pub solve_time_ms: u64,

    /// Time spent in KKT factorization (milliseconds)
    pub kkt_factor_time_ms: u64,

    /// Time spent in KKT solves (milliseconds)
    pub kkt_solve_time_ms: u64,

    /// Time spent in cone operations (milliseconds)
    pub cone_time_ms: u64,

    /// Final primal residual norm
    pub primal_res: f64,

    /// Final dual residual norm
    pub dual_res: f64,

    /// Final duality gap
    pub gap: f64,

    /// Final barrier parameter μ
    pub mu: f64,

    /// Static regularization used
    pub reg_static: f64,

    /// Number of dynamic regularization bumps applied
    pub reg_dynamic_bumps: u64,
}

impl ProblemData {
    /// Get the number of primal variables (n)
    pub fn num_vars(&self) -> usize {
        self.q.len()
    }

    /// Get the number of constraints (m)
    pub fn num_constraints(&self) -> usize {
        self.b.len()
    }

    /// Validate problem dimensions and cone partitioning
    pub fn validate(&self) -> Result<(), String> {
        let n = self.num_vars();
        let m = self.num_constraints();

        // Check q dimension
        if self.q.len() != n {
            return Err(format!("q has length {}, expected {}", self.q.len(), n));
        }

        // Check P dimensions if present
        if let Some(ref p) = self.P {
            if p.rows() != n || p.cols() != n {
                return Err(format!(
                    "P has shape {}×{}, expected {}×{}",
                    p.rows(), p.cols(), n, n
                ));
            }
        }

        // Check A dimensions
        if self.A.rows() != m {
            return Err(format!(
                "A has {} rows, expected {}",
                self.A.rows(), m
            ));
        }
        if self.A.cols() != n {
            return Err(format!(
                "A has {} cols, expected {}",
                self.A.cols(), n
            ));
        }

        // Check b dimension
        if self.b.len() != m {
            return Err(format!("b has length {}, expected {}", self.b.len(), m));
        }

        // Check cone dimensions sum to m
        let cone_total_dim: usize = self.cones.iter().map(|c| c.dim()).sum();
        if cone_total_dim != m {
            return Err(format!(
                "Cone dimensions sum to {}, expected {}",
                cone_total_dim, m
            ));
        }

        // Validate individual cones
        for cone in &self.cones {
            cone.validate()?;
        }
        // POW cones not yet supported
        if self.cones.iter().any(|cone| {
            matches!(cone, ConeSpec::Pow { .. })
        }) {
            return Err("POW cones are not supported yet".to_string());
        }

        // Check variable bounds if present
        if let Some(ref bounds) = self.var_bounds {
            for bound in bounds {
                if bound.var >= n {
                    return Err(format!(
                        "Bound on variable {} out of range (n={})",
                        bound.var, n
                    ));
                }
                if let (Some(l), Some(u)) = (bound.lower, bound.upper) {
                    if l > u {
                        return Err(format!(
                            "Variable {} has lower bound {} > upper bound {}",
                            bound.var, l, u
                        ));
                    }
                }
            }
        }

        // Check integrality if present
        if let Some(ref int_types) = self.integrality {
            if int_types.len() != n {
                return Err(format!(
                    "Integrality vector has length {}, expected {}",
                    int_types.len(), n
                ));
            }
        }

        Ok(())
    }

    /// Convert variable bounds to explicit cone constraints.
    ///
    /// This creates a new problem with var_bounds = None, where bounds are
    /// represented as NonNeg cone constraints:
    /// - x >= lb becomes -x + s = -lb with s >= 0
    /// - x <= ub becomes  x + s = ub with s >= 0
    pub fn with_bounds_as_constraints(&self) -> Self {
        let Some(ref bounds) = self.var_bounds else {
            // No bounds, return clone
            return self.clone();
        };

        // Count lower and upper bounds
        let mut num_lb = 0;
        let mut num_ub = 0;
        for b in bounds {
            if b.lower.is_some() {
                num_lb += 1;
            }
            if b.upper.is_some() {
                num_ub += 1;
            }
        }

        if num_lb + num_ub == 0 {
            return self.clone();
        }

        let n = self.num_vars();
        let m = self.num_constraints();
        let m_new = m + num_lb + num_ub;

        // Build new A matrix with bound constraints appended
        use sprs::TriMat;
        let mut tri = TriMat::new((m_new, n));

        // Copy existing A
        for (col_idx, col) in self.A.outer_iterator().enumerate() {
            for (row_idx, &val) in col.iter() {
                tri.add_triplet(row_idx, col_idx, val);
            }
        }

        // Add lower bound rows: -x + s = -lb with s >= 0 means x >= lb
        let mut row = m;
        for b in bounds {
            if b.lower.is_some() {
                tri.add_triplet(row, b.var, -1.0);
                row += 1;
            }
        }

        // Add upper bound rows: x + s = ub with s >= 0 means x <= ub
        for b in bounds {
            if b.upper.is_some() {
                tri.add_triplet(row, b.var, 1.0);
                row += 1;
            }
        }

        let a_new = tri.to_csc();

        // Build new b vector
        let mut b_new = Vec::with_capacity(m_new);
        b_new.extend_from_slice(&self.b);

        // Lower bound RHS: -lb
        for b in bounds {
            if let Some(lb) = b.lower {
                b_new.push(-lb);
            }
        }

        // Upper bound RHS: ub
        for b in bounds {
            if let Some(ub) = b.upper {
                b_new.push(ub);
            }
        }

        // Add NonNeg cone for bounds
        let mut cones_new = self.cones.clone();
        if num_lb + num_ub > 0 {
            cones_new.push(ConeSpec::NonNeg { dim: num_lb + num_ub });
        }

        ProblemData {
            P: self.P.clone(),
            q: self.q.clone(),
            A: a_new,
            b: b_new,
            cones: cones_new,
            var_bounds: None,
            integrality: self.integrality.clone(),
        }
    }
}

impl ConeSpec {
    /// Get the dimension of this cone in the m-dimensional space
    pub fn dim(&self) -> usize {
        match self {
            ConeSpec::Zero { dim } => *dim,
            ConeSpec::NonNeg { dim } => *dim,
            ConeSpec::Soc { dim } => *dim,
            ConeSpec::Psd { n } => n * (n + 1) / 2,  // svec dimension
            ConeSpec::Exp { count } => 3 * count,
            ConeSpec::Pow { cones } => 3 * cones.len(),
        }
    }

    /// Get the barrier degree ν of this cone
    pub fn barrier_degree(&self) -> usize {
        match self {
            ConeSpec::Zero { .. } => 0,
            ConeSpec::NonNeg { dim } => *dim,
            ConeSpec::Soc { .. } => 2,  // SOC always has degree 2
            ConeSpec::Psd { n } => *n,
            ConeSpec::Exp { count } => 3 * count,
            ConeSpec::Pow { cones } => 3 * cones.len(),
        }
    }

    /// Validate this cone specification
    pub fn validate(&self) -> Result<(), String> {
        match self {
            ConeSpec::Zero { dim } => {
                if *dim == 0 {
                    return Err("Zero cone must have positive dimension".to_string());
                }
            }
            ConeSpec::NonNeg { dim } => {
                if *dim == 0 {
                    return Err("NonNeg cone must have positive dimension".to_string());
                }
            }
            ConeSpec::Soc { dim } => {
                if *dim < 2 {
                    return Err(format!(
                        "SOC cone must have dimension >= 2, got {}",
                        dim
                    ));
                }
            }
            ConeSpec::Psd { n } => {
                if *n == 0 {
                    return Err("PSD cone must have positive size".to_string());
                }
            }
            ConeSpec::Exp { count } => {
                if *count == 0 {
                    return Err("Exp cone must have positive count".to_string());
                }
            }
            ConeSpec::Pow { cones } => {
                if cones.is_empty() {
                    return Err("Pow cone must have at least one block".to_string());
                }
                for pow in cones {
                    if !(0.0 < pow.alpha && pow.alpha < 1.0) {
                        return Err(format!(
                            "Power cone alpha must be in (0,1), got {}",
                            pow.alpha
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cone_dim() {
        assert_eq!(ConeSpec::Zero { dim: 5 }.dim(), 5);
        assert_eq!(ConeSpec::NonNeg { dim: 10 }.dim(), 10);
        assert_eq!(ConeSpec::Soc { dim: 7 }.dim(), 7);
        assert_eq!(ConeSpec::Psd { n: 3 }.dim(), 6);  // 3*4/2
        assert_eq!(ConeSpec::Exp { count: 2 }.dim(), 6);
        assert_eq!(
            ConeSpec::Pow { cones: vec![Pow3D { alpha: 0.5 }, Pow3D { alpha: 0.3 }] }.dim(),
            6
        );
    }

    #[test]
    fn test_cone_barrier_degree() {
        assert_eq!(ConeSpec::Zero { dim: 5 }.barrier_degree(), 0);
        assert_eq!(ConeSpec::NonNeg { dim: 10 }.barrier_degree(), 10);
        assert_eq!(ConeSpec::Soc { dim: 100 }.barrier_degree(), 2);
        assert_eq!(ConeSpec::Psd { n: 5 }.barrier_degree(), 5);
        assert_eq!(ConeSpec::Exp { count: 3 }.barrier_degree(), 9);
    }

    #[test]
    fn test_cone_validation() {
        // Valid cones
        assert!(ConeSpec::Zero { dim: 1 }.validate().is_ok());
        assert!(ConeSpec::NonNeg { dim: 1 }.validate().is_ok());
        assert!(ConeSpec::Soc { dim: 2 }.validate().is_ok());
        assert!(ConeSpec::Psd { n: 2 }.validate().is_ok());
        assert!(ConeSpec::Exp { count: 1 }.validate().is_ok());
        assert!(ConeSpec::Pow { cones: vec![Pow3D { alpha: 0.5 }] }.validate().is_ok());

        // Invalid cones
        assert!(ConeSpec::Zero { dim: 0 }.validate().is_err());
        assert!(ConeSpec::Soc { dim: 1 }.validate().is_err());
        assert!(ConeSpec::Pow { cones: vec![Pow3D { alpha: 0.0 }] }.validate().is_err());
        assert!(ConeSpec::Pow { cones: vec![Pow3D { alpha: 1.0 }] }.validate().is_err());
        assert!(ConeSpec::Pow { cones: vec![Pow3D { alpha: 1.5 }] }.validate().is_err());
    }
}
