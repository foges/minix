//! Branching variable selection.

use super::BoundChange;
use crate::model::MipProblem;
use crate::settings::BranchingRule;

/// A branching decision.
#[derive(Debug, Clone)]
pub struct BranchDecision {
    /// Variable to branch on.
    pub var: usize,

    /// Current (fractional) value.
    pub value: f64,

    /// Bound change for "down" branch (x <= floor(value)).
    pub down_branch: BoundChange,

    /// Bound change for "up" branch (x >= ceil(value)).
    pub up_branch: BoundChange,

    /// Score of this decision (for logging/debugging).
    pub score: f64,
}

/// Branching variable selector.
pub struct BranchingSelector {
    /// Branching rule to use.
    rule: BranchingRule,

    /// Pseudocost statistics (for pseudocost branching).
    /// pseudocosts_down[i] = average objective change per unit decrease
    /// pseudocosts_up[i] = average objective change per unit increase
    pseudocosts_down: Vec<f64>,
    pseudocosts_up: Vec<f64>,

    /// Number of times each variable has been branched on (down direction).
    branch_count_down: Vec<u64>,

    /// Number of times each variable has been branched on (up direction).
    branch_count_up: Vec<u64>,

    /// Total nodes processed (for hybrid switching).
    nodes_processed: u64,

    /// Whether we have found an incumbent (for two-phase).
    has_incumbent: bool,
}

impl BranchingSelector {
    /// Create a new branching selector.
    pub fn new(rule: BranchingRule, num_vars: usize) -> Self {
        Self {
            rule,
            pseudocosts_down: vec![1.0; num_vars], // Initialize to 1 (neutral)
            pseudocosts_up: vec![1.0; num_vars],
            branch_count_down: vec![0; num_vars],
            branch_count_up: vec![0; num_vars],
            nodes_processed: 0,
            has_incumbent: false,
        }
    }

    /// Notify that a node has been processed.
    pub fn node_processed(&mut self) {
        self.nodes_processed += 1;
    }

    /// Notify that an incumbent has been found.
    pub fn set_has_incumbent(&mut self, has: bool) {
        self.has_incumbent = has;
    }

    /// Get the total branch count for a variable.
    pub fn branch_count(&self, var: usize) -> u64 {
        self.branch_count_down[var] + self.branch_count_up[var]
    }

    /// Check if pseudocosts for a variable are reliable.
    pub fn is_reliable(&self, var: usize, min_count: u64) -> bool {
        self.branch_count_down[var] >= min_count && self.branch_count_up[var] >= min_count
    }

    /// Select a branching variable.
    ///
    /// Returns None if the solution is integer-feasible.
    pub fn select(
        &self,
        x: &[f64],
        prob: &MipProblem,
        tol: f64,
    ) -> Option<BranchDecision> {
        // Find fractional integer variables
        let fractional = prob.get_fractional_vars(x, tol);

        if fractional.is_empty() {
            return None;
        }

        match self.rule {
            BranchingRule::MostFractional => self.select_most_fractional(&fractional, prob),
            BranchingRule::Pseudocost => self.select_pseudocost(&fractional, prob),
            BranchingRule::StrongBranching { candidates } => {
                // Strong branching: evaluate top candidates
                self.select_strong_branching(&fractional, prob, candidates)
            }
            BranchingRule::Reliability {
                candidates,
                reliability_count,
                max_sb_iters: _,
            } => {
                self.select_reliability(&fractional, prob, candidates, reliability_count)
            }
            BranchingRule::Hybrid { switch_after_nodes } => {
                if self.nodes_processed < switch_after_nodes {
                    self.select_most_fractional(&fractional, prob)
                } else {
                    self.select_pseudocost(&fractional, prob)
                }
            }
        }
    }

    /// Select variable closest to 0.5 (most fractional).
    fn select_most_fractional(
        &self,
        fractional: &[(usize, f64, f64)],
        prob: &MipProblem,
    ) -> Option<BranchDecision> {
        // Select variable with fractionality closest to 0.5
        let (var, value, frac) = fractional
            .iter()
            .max_by(|(_, _, f1), (_, _, f2)| f1.partial_cmp(f2).unwrap())
            .copied()?;

        Some(self.make_decision(var, value, frac, prob))
    }

    /// Select variable with best pseudocost score.
    fn select_pseudocost(
        &self,
        fractional: &[(usize, f64, f64)],
        prob: &MipProblem,
    ) -> Option<BranchDecision> {
        // Score = product(down_cost, up_cost) - maximizing minimum improvement
        let (var, value, score) = fractional
            .iter()
            .map(|(v, val, _)| (*v, *val, self.pseudocost_score(*v, *val)))
            .max_by(|(_, _, s1), (_, _, s2)| s1.partial_cmp(s2).unwrap())?;

        Some(self.make_decision(var, value, score, prob))
    }

    /// Select using strong branching (evaluate LP bounds for top candidates).
    fn select_strong_branching(
        &self,
        fractional: &[(usize, f64, f64)],
        prob: &MipProblem,
        max_candidates: usize,
    ) -> Option<BranchDecision> {
        // For now, use pseudocost as a proxy for strong branching
        // (full implementation would require LP solves)
        // Select top candidates by fractionality, then score by pseudocost
        let mut candidates: Vec<_> = fractional.to_vec();
        candidates.sort_by(|(_, _, f1), (_, _, f2)| {
            f2.partial_cmp(f1).unwrap() // Most fractional first
        });
        candidates.truncate(max_candidates);

        // Score each candidate
        let (var, value, score) = candidates
            .iter()
            .map(|(v, val, _)| (*v, *val, self.pseudocost_score(*v, *val)))
            .max_by(|(_, _, s1), (_, _, s2)| s1.partial_cmp(s2).unwrap())?;

        Some(self.make_decision(var, value, score, prob))
    }

    /// Select using reliability branching.
    ///
    /// Uses strong branching for unreliable variables, pseudocost for reliable ones.
    fn select_reliability(
        &self,
        fractional: &[(usize, f64, f64)],
        prob: &MipProblem,
        max_candidates: usize,
        reliability_count: u64,
    ) -> Option<BranchDecision> {
        // Partition into reliable and unreliable
        let (reliable, unreliable): (Vec<_>, Vec<_>) = fractional
            .iter()
            .partition(|(v, _, _)| self.is_reliable(*v, reliability_count));

        if !unreliable.is_empty() {
            // Strong branch on unreliable candidates
            self.select_strong_branching(&unreliable, prob, max_candidates)
        } else {
            // All reliable - use pseudocost
            self.select_pseudocost(&reliable, prob)
        }
    }

    /// Compute pseudocost score for a variable.
    fn pseudocost_score(&self, var: usize, value: f64) -> f64 {
        let frac = value.fract().abs();
        let down_frac = frac;
        let up_frac = 1.0 - frac;

        let down_cost = down_frac * self.pseudocosts_down[var];
        let up_cost = up_frac * self.pseudocosts_up[var];

        // Use product score (common in MIP solvers)
        // This prefers balanced improvements in both directions
        (down_cost * up_cost).max(1e-10)
    }

    /// Create a branch decision for a variable.
    fn make_decision(&self, var: usize, value: f64, score: f64, prob: &MipProblem) -> BranchDecision {
        let old_lb = prob.var_lb[var];
        let old_ub = prob.var_ub[var];

        BranchDecision {
            var,
            value,
            down_branch: BoundChange::down_branch(var, old_lb, old_ub, value),
            up_branch: BoundChange::up_branch(var, old_lb, old_ub, value),
            score,
        }
    }

    /// Update pseudocosts after branching.
    ///
    /// Called after solving child nodes to update pseudocost estimates.
    pub fn update_pseudocosts(
        &mut self,
        var: usize,
        value: f64,
        down_obj_change: Option<f64>,
        up_obj_change: Option<f64>,
    ) {
        let frac = value.fract().abs();
        let down_frac = frac;
        let up_frac = 1.0 - frac;

        if let Some(change) = down_obj_change {
            if down_frac > 1e-6 && change > 0.0 {
                let pc = change / down_frac;
                // Running average with more weight on recent observations
                let count = self.branch_count_down[var] as f64;
                self.pseudocosts_down[var] =
                    (self.pseudocosts_down[var] * count + pc) / (count + 1.0);
                self.branch_count_down[var] += 1;
            }
        }

        if let Some(change) = up_obj_change {
            if up_frac > 1e-6 && change > 0.0 {
                let pc = change / up_frac;
                let count = self.branch_count_up[var] as f64;
                self.pseudocosts_up[var] =
                    (self.pseudocosts_up[var] * count + pc) / (count + 1.0);
                self.branch_count_up[var] += 1;
            }
        }
    }

    /// Get pseudocost statistics for a variable.
    pub fn get_pseudocosts(&self, var: usize) -> (f64, f64, u64, u64) {
        (
            self.pseudocosts_down[var],
            self.pseudocosts_up[var],
            self.branch_count_down[var],
            self.branch_count_up[var],
        )
    }

    /// Initialize pseudocosts from objective coefficients.
    ///
    /// This provides a reasonable starting point before any branching.
    pub fn init_from_objective(&mut self, q: &[f64]) {
        for (i, &qi) in q.iter().enumerate() {
            if i < self.pseudocosts_down.len() {
                // Use absolute value of objective coefficient as initial estimate
                let init_cost = qi.abs().max(0.1);
                self.pseudocosts_down[i] = init_cost;
                self.pseudocosts_up[i] = init_cost;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solver_core::{ConeSpec, ProblemData, VarType};
    use sprs::CsMat;

    fn simple_mip() -> MipProblem {
        let n = 3;
        let m = 1;
        let a = CsMat::new_csc((m, n), vec![0, 1, 2, 3], vec![0, 0, 0], vec![1.0, 1.0, 1.0]);

        let prob = ProblemData {
            P: None,
            q: vec![1.0, 1.0, 1.0],
            A: a,
            b: vec![2.0],
            cones: vec![ConeSpec::NonNeg { dim: 1 }],
            var_bounds: Some(vec![
                solver_core::VarBound { var: 0, lower: Some(0.0), upper: Some(1.0) },
                solver_core::VarBound { var: 1, lower: Some(0.0), upper: Some(1.0) },
                solver_core::VarBound { var: 2, lower: Some(0.0), upper: None },
            ]),
            integrality: Some(vec![VarType::Binary, VarType::Binary, VarType::Continuous]),
        };

        MipProblem::new(prob).unwrap()
    }

    #[test]
    fn test_most_fractional() {
        let prob = simple_mip();
        let selector = BranchingSelector::new(BranchingRule::MostFractional, 3);

        // x0 = 0.3, x1 = 0.7, x2 = 1.0
        // Fractionalities: x0 = 0.3, x1 = 0.3
        // Both equally fractional, either is valid
        let x = vec![0.3, 0.7, 1.0];
        let decision = selector.select(&x, &prob, 1e-6);

        assert!(decision.is_some());
        let d = decision.unwrap();
        assert!(d.var == 0 || d.var == 1);
    }

    #[test]
    fn test_integer_feasible() {
        let prob = simple_mip();
        let selector = BranchingSelector::new(BranchingRule::MostFractional, 3);

        // All integers at integer values
        let x = vec![1.0, 0.0, 1.0];
        let decision = selector.select(&x, &prob, 1e-6);

        assert!(decision.is_none());
    }

    #[test]
    fn test_branch_decision() {
        let prob = simple_mip();
        let selector = BranchingSelector::new(BranchingRule::MostFractional, 3);

        let x = vec![0.5, 0.0, 1.0];
        let decision = selector.select(&x, &prob, 1e-6).unwrap();

        assert_eq!(decision.var, 0);
        assert_eq!(decision.value, 0.5);

        // Down branch: x0 <= 0
        assert_eq!(decision.down_branch.new_ub, 0.0);

        // Up branch: x0 >= 1
        assert_eq!(decision.up_branch.new_lb, 1.0);
    }
}
