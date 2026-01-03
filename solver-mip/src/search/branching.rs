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

    /// Number of times each variable has been branched on.
    branch_count: Vec<u64>,
}

impl BranchingSelector {
    /// Create a new branching selector.
    pub fn new(rule: BranchingRule, num_vars: usize) -> Self {
        Self {
            rule,
            pseudocosts_down: vec![1.0; num_vars], // Initialize to 1 (neutral)
            pseudocosts_up: vec![1.0; num_vars],
            branch_count: vec![0; num_vars],
        }
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
            BranchingRule::StrongBranching { candidates: _ } => {
                // For now, fall back to most fractional
                // TODO: implement strong branching
                self.select_most_fractional(&fractional, prob)
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
        let (var, value, _) = fractional
            .iter()
            .max_by(|(_, _, f1), (_, _, f2)| f1.partial_cmp(f2).unwrap())
            .copied()?;

        Some(self.make_decision(var, value, prob))
    }

    /// Select variable with best pseudocost score.
    fn select_pseudocost(
        &self,
        fractional: &[(usize, f64, f64)],
        prob: &MipProblem,
    ) -> Option<BranchDecision> {
        // Score = min(frac * pc_down, (1-frac) * pc_up)
        // This estimates the minimum objective increase from branching
        let (var, value, _) = fractional
            .iter()
            .max_by(|(v1, val1, _), (v2, val2, _)| {
                let score1 = self.pseudocost_score(*v1, *val1);
                let score2 = self.pseudocost_score(*v2, *val2);
                score1.partial_cmp(&score2).unwrap()
            })
            .copied()?;

        Some(self.make_decision(var, value, prob))
    }

    /// Compute pseudocost score for a variable.
    fn pseudocost_score(&self, var: usize, value: f64) -> f64 {
        let frac = value.fract();
        let down_frac = frac;
        let up_frac = 1.0 - frac;

        let down_cost = down_frac * self.pseudocosts_down[var];
        let up_cost = up_frac * self.pseudocosts_up[var];

        // Use product score (common in MIP solvers)
        (down_cost * up_cost).max(1e-10)
    }

    /// Create a branch decision for a variable.
    fn make_decision(&self, var: usize, value: f64, prob: &MipProblem) -> BranchDecision {
        let old_lb = prob.var_lb[var];
        let old_ub = prob.var_ub[var];

        BranchDecision {
            var,
            value,
            down_branch: BoundChange::down_branch(var, old_lb, old_ub, value),
            up_branch: BoundChange::up_branch(var, old_lb, old_ub, value),
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
        let frac = value.fract();
        let down_frac = frac;
        let up_frac = 1.0 - frac;

        if let Some(change) = down_obj_change {
            if down_frac > 1e-6 {
                let pc = change / down_frac;
                // Running average
                let count = self.branch_count[var] as f64;
                self.pseudocosts_down[var] =
                    (self.pseudocosts_down[var] * count + pc) / (count + 1.0);
            }
        }

        if let Some(change) = up_obj_change {
            if up_frac > 1e-6 {
                let pc = change / up_frac;
                let count = self.branch_count[var] as f64;
                self.pseudocosts_up[var] =
                    (self.pseudocosts_up[var] * count + pc) / (count + 1.0);
            }
        }

        self.branch_count[var] += 1;
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
