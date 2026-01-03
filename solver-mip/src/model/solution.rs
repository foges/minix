//! MIP solution types.

/// Status of the MIP solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MipStatus {
    /// Optimal solution found within tolerance.
    Optimal,

    /// Problem is infeasible.
    Infeasible,

    /// Problem is unbounded.
    Unbounded,

    /// Node limit reached, best solution returned.
    NodeLimit,

    /// Time limit reached, best solution returned.
    TimeLimit,

    /// Gap limit reached (solution within gap_tol of optimal).
    GapLimit,

    /// Numerical difficulties encountered.
    NumericalError,

    /// Solver was interrupted.
    Interrupted,
}

impl MipStatus {
    /// Returns true if a feasible solution was found.
    pub fn has_solution(&self) -> bool {
        matches!(
            self,
            MipStatus::Optimal | MipStatus::NodeLimit | MipStatus::TimeLimit | MipStatus::GapLimit
        )
    }

    /// Returns true if optimality was proven.
    pub fn is_optimal(&self) -> bool {
        matches!(self, MipStatus::Optimal | MipStatus::GapLimit)
    }
}

/// Complete MIP solution with diagnostics.
#[derive(Debug, Clone)]
pub struct MipSolution {
    /// Solve status.
    pub status: MipStatus,

    /// Primal solution (if found).
    pub x: Vec<f64>,

    /// Objective value of best solution (primal bound).
    pub obj_val: f64,

    /// Best dual bound (from LP relaxations).
    pub bound: f64,

    /// Relative optimality gap: (obj_val - bound) / |obj_val|.
    pub gap: f64,

    /// Number of B&B nodes explored.
    pub nodes_explored: u64,

    /// Number of cuts added.
    pub cuts_added: u64,

    /// Total solve time in milliseconds.
    pub solve_time_ms: u64,

    /// Number of times incumbent was updated.
    pub incumbent_updates: u64,
}

impl Default for MipSolution {
    fn default() -> Self {
        Self {
            status: MipStatus::Infeasible,
            x: Vec::new(),
            obj_val: f64::INFINITY,
            bound: f64::NEG_INFINITY,
            gap: f64::INFINITY,
            nodes_explored: 0,
            cuts_added: 0,
            solve_time_ms: 0,
            incumbent_updates: 0,
        }
    }
}

impl MipSolution {
    /// Create a solution indicating infeasibility.
    pub fn infeasible() -> Self {
        Self {
            status: MipStatus::Infeasible,
            ..Default::default()
        }
    }

    /// Create an optimal solution.
    pub fn optimal(x: Vec<f64>, obj_val: f64, bound: f64) -> Self {
        Self {
            status: MipStatus::Optimal,
            x,
            obj_val,
            bound,
            gap: Self::compute_gap(obj_val, bound),
            ..Default::default()
        }
    }

    /// Compute relative gap.
    pub fn compute_gap(primal: f64, dual: f64) -> f64 {
        if primal.is_infinite() || dual.is_infinite() {
            return f64::INFINITY;
        }
        let denom = primal.abs().max(1e-10);
        (primal - dual).abs() / denom
    }
}

/// Tracks the best known feasible solution (incumbent).
#[derive(Debug, Clone)]
pub struct IncumbentTracker {
    /// Current best solution (if any).
    pub solution: Option<Vec<f64>>,

    /// Objective value of incumbent (primal bound).
    /// Initialized to +∞ for minimization.
    pub obj_val: f64,

    /// Number of times incumbent was updated.
    pub update_count: u64,
}

impl Default for IncumbentTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl IncumbentTracker {
    /// Create a new incumbent tracker.
    pub fn new() -> Self {
        Self {
            solution: None,
            obj_val: f64::INFINITY,
            update_count: 0,
        }
    }

    /// Check if we have an incumbent.
    pub fn has_incumbent(&self) -> bool {
        self.solution.is_some()
    }

    /// Try to update incumbent with a new solution.
    ///
    /// Returns true if the incumbent was improved.
    pub fn update(&mut self, x: &[f64], obj: f64) -> bool {
        // For minimization, accept if strictly better
        if obj < self.obj_val - 1e-9 {
            self.solution = Some(x.to_vec());
            self.obj_val = obj;
            self.update_count += 1;
            true
        } else {
            false
        }
    }

    /// Compute relative gap to a dual bound.
    pub fn gap(&self, dual_bound: f64) -> f64 {
        MipSolution::compute_gap(self.obj_val, dual_bound)
    }

    /// Check if gap is within tolerance.
    pub fn gap_closed(&self, dual_bound: f64, tol: f64) -> bool {
        self.gap(dual_bound) <= tol
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incumbent_tracker() {
        let mut tracker = IncumbentTracker::new();

        assert!(!tracker.has_incumbent());
        assert_eq!(tracker.obj_val, f64::INFINITY);

        // First solution
        assert!(tracker.update(&[1.0, 2.0], 10.0));
        assert!(tracker.has_incumbent());
        assert_eq!(tracker.obj_val, 10.0);
        assert_eq!(tracker.update_count, 1);

        // Worse solution (rejected)
        assert!(!tracker.update(&[2.0, 3.0], 15.0));
        assert_eq!(tracker.obj_val, 10.0);
        assert_eq!(tracker.update_count, 1);

        // Better solution (accepted)
        assert!(tracker.update(&[0.5, 1.0], 5.0));
        assert_eq!(tracker.obj_val, 5.0);
        assert_eq!(tracker.update_count, 2);
    }

    #[test]
    fn test_gap_computation() {
        // Gap = |10 - 8| / |10| = 0.2
        let gap = MipSolution::compute_gap(10.0, 8.0);
        assert!((gap - 0.2).abs() < 1e-10);

        // Gap near zero
        let gap = MipSolution::compute_gap(10.0, 9.9999);
        assert!(gap < 0.001);
    }

    #[test]
    fn test_status_methods() {
        assert!(MipStatus::Optimal.has_solution());
        assert!(MipStatus::NodeLimit.has_solution());
        assert!(!MipStatus::Infeasible.has_solution());

        assert!(MipStatus::Optimal.is_optimal());
        assert!(MipStatus::GapLimit.is_optimal());
        assert!(!MipStatus::NodeLimit.is_optimal());
    }
}
