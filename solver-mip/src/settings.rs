//! Configuration settings for the MIP solver.

use solver_core::SolverSettings;

/// Branching variable selection rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BranchingRule {
    /// Select variable with fractional part closest to 0.5.
    #[default]
    MostFractional,

    /// Use pseudocost estimates from previous branches.
    Pseudocost,

    /// Strong branching: solve LP relaxations to evaluate candidates.
    StrongBranching {
        /// Number of candidate variables to evaluate.
        candidates: usize,
    },
}

/// Node selection strategy for the B&B tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NodeSelection {
    /// Always select node with best (lowest) dual bound.
    #[default]
    BestBound,

    /// Depth-first search (helps find feasible solutions quickly).
    DepthFirst,

    /// Select by estimated objective value.
    BestEstimate,

    /// Hybrid: alternate between diving and best-bound.
    Hybrid {
        /// How often to dive (every N nodes).
        dive_freq: usize,
    },
}

/// MIP solver settings.
#[derive(Debug, Clone)]
pub struct MipSettings {
    // === Termination criteria ===
    /// Maximum number of nodes to explore.
    pub max_nodes: u64,

    /// Time limit in milliseconds (None = unlimited).
    pub time_limit_ms: Option<u64>,

    /// Relative optimality gap tolerance.
    /// Stop when (incumbent - bound) / |incumbent| <= gap_tol.
    pub gap_tol: f64,

    /// Absolute optimality gap tolerance.
    pub gap_abs_tol: f64,

    /// Integer feasibility tolerance.
    /// A variable is considered integer if |x - round(x)| <= int_feas_tol.
    pub int_feas_tol: f64,

    // === Search strategy ===
    /// Branching variable selection rule.
    pub branching_rule: BranchingRule,

    /// Node selection strategy.
    pub node_selection: NodeSelection,

    // === Cut settings ===
    /// Maximum cuts to add per separation round.
    pub cuts_per_round: usize,

    /// How often to clean up inactive cuts (every N nodes).
    pub cut_cleanup_freq: usize,

    /// Generate disaggregated K* cuts (one per cone block).
    pub disaggregate_cuts: bool,

    /// Minimum violation for a cut to be added.
    pub cut_violation_tol: f64,

    // === Solver settings ===
    /// Settings for the master LP/QP solver.
    pub master_settings: SolverSettings,

    /// Settings for the conic oracle (subproblem solver).
    pub oracle_settings: SolverSettings,

    // === Output ===
    /// Print progress information.
    pub verbose: bool,

    /// Log frequency (print every N nodes).
    pub log_freq: u64,
}

impl Default for MipSettings {
    fn default() -> Self {
        let mut master_settings = SolverSettings::default();
        // Master is LP/QP, can use tighter tolerances
        master_settings.tol_feas = 1e-8;
        master_settings.tol_gap = 1e-8;

        let mut oracle_settings = SolverSettings::default();
        // Oracle validates conic feasibility
        oracle_settings.tol_feas = 1e-7;
        oracle_settings.tol_gap = 1e-7;

        Self {
            // Termination
            max_nodes: 1_000_000,
            time_limit_ms: None,
            gap_tol: 1e-4,
            gap_abs_tol: 1e-6,
            int_feas_tol: 1e-6,

            // Search
            branching_rule: BranchingRule::default(),
            node_selection: NodeSelection::default(),

            // Cuts
            cuts_per_round: 100,
            cut_cleanup_freq: 100,
            disaggregate_cuts: true,
            cut_violation_tol: 1e-7,

            // Solver
            master_settings,
            oracle_settings,

            // Output
            verbose: false,
            log_freq: 100,
        }
    }
}

impl MipSettings {
    /// Create settings with verbose output enabled.
    pub fn verbose() -> Self {
        let mut s = Self::default();
        s.verbose = true;
        s.log_freq = 1;
        s
    }

    /// Set time limit in seconds.
    pub fn with_time_limit(mut self, seconds: f64) -> Self {
        self.time_limit_ms = Some((seconds * 1000.0) as u64);
        self
    }

    /// Set maximum nodes.
    pub fn with_max_nodes(mut self, nodes: u64) -> Self {
        self.max_nodes = nodes;
        self
    }

    /// Set optimality gap tolerance.
    pub fn with_gap_tol(mut self, tol: f64) -> Self {
        self.gap_tol = tol;
        self
    }
}
