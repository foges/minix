//! Branch-and-bound tree controller.

use std::time::Instant;

use super::{BranchDecision, BranchingSelector, NodeQueue, SearchNode};
use crate::model::{IncumbentTracker, MipProblem, MipSolution, MipStatus};
use crate::settings::MipSettings;

/// Branch-and-bound tree controller.
///
/// Manages the B&B tree, node queue, incumbent, and termination.
pub struct BranchAndBound {
    /// Node queue.
    queue: NodeQueue,

    /// Branching variable selector.
    branching: BranchingSelector,

    /// Incumbent solution tracker.
    pub incumbent: IncumbentTracker,

    /// Next node ID to assign.
    next_node_id: u64,

    /// Total nodes explored.
    nodes_explored: u64,

    /// Nodes pruned.
    nodes_pruned: u64,

    /// Cuts added.
    cuts_added: u64,

    /// Start time.
    start_time: Option<Instant>,

    /// Settings.
    settings: MipSettings,
}

impl BranchAndBound {
    /// Create a new B&B controller.
    pub fn new(settings: MipSettings, num_vars: usize) -> Self {
        Self {
            queue: NodeQueue::new(settings.node_selection),
            branching: BranchingSelector::new(settings.branching_rule, num_vars),
            incumbent: IncumbentTracker::new(),
            next_node_id: 1, // 0 reserved for root
            nodes_explored: 0,
            nodes_pruned: 0,
            cuts_added: 0,
            start_time: None,
            settings,
        }
    }

    /// Initialize with the root node.
    pub fn initialize(&mut self, root_bound: f64) {
        self.start_time = Some(Instant::now());

        let mut root = SearchNode::root();
        root.dual_bound = root_bound;
        root.estimate = root_bound;

        self.queue.push(root);
    }

    /// Get the next node to process.
    pub fn next_node(&mut self) -> Option<SearchNode> {
        self.queue.pop()
    }

    /// Mark a node as explored.
    pub fn node_explored(&mut self) {
        self.nodes_explored += 1;
    }

    /// Record that a node was pruned.
    pub fn node_pruned(&mut self) {
        self.nodes_pruned += 1;
    }

    /// Record cuts added.
    pub fn cuts_added(&mut self, count: usize) {
        self.cuts_added += count as u64;
    }

    /// Create child nodes from a branching decision.
    ///
    /// Returns the two child nodes (down, up).
    pub fn branch(&mut self, parent: &SearchNode, decision: BranchDecision) -> (SearchNode, SearchNode) {
        let down_id = self.next_node_id;
        let up_id = self.next_node_id + 1;
        self.next_node_id += 2;

        let down_child = parent.child(down_id, decision.down_branch);
        let up_child = parent.child(up_id, decision.up_branch);

        (down_child, up_child)
    }

    /// Add a node to the queue.
    pub fn enqueue(&mut self, node: SearchNode) {
        self.queue.push(node);
    }

    /// Select a branching variable.
    pub fn select_branching(
        &self,
        x: &[f64],
        prob: &MipProblem,
    ) -> Option<BranchDecision> {
        self.branching.select(x, prob, self.settings.int_feas_tol)
    }

    /// Update incumbent with a new solution.
    ///
    /// Returns true if incumbent was improved.
    pub fn update_incumbent(&mut self, x: &[f64], obj: f64) -> bool {
        let improved = self.incumbent.update(x, obj);

        if improved {
            // Prune nodes dominated by new incumbent
            let pruned = self.queue.prune_by_bound(obj);
            self.nodes_pruned += pruned as u64;

            if self.settings.verbose {
                log::info!(
                    "New incumbent: obj={:.6e}, pruned {} nodes",
                    obj,
                    pruned
                );
            }
        }

        improved
    }

    /// Get the current optimality gap.
    pub fn gap(&self) -> f64 {
        self.incumbent.gap(self.queue.best_bound())
    }

    /// Get the best dual bound.
    pub fn best_bound(&self) -> f64 {
        self.queue.best_bound()
    }

    /// Get elapsed time in milliseconds.
    pub fn elapsed_ms(&self) -> u64 {
        self.start_time
            .map(|t| t.elapsed().as_millis() as u64)
            .unwrap_or(0)
    }

    /// Check if time limit is exceeded.
    pub fn time_limit_exceeded(&self) -> bool {
        if let Some(limit) = self.settings.time_limit_ms {
            self.elapsed_ms() >= limit
        } else {
            false
        }
    }

    /// Check termination conditions.
    ///
    /// Returns Some(status) if we should terminate, None otherwise.
    pub fn check_termination(&self) -> Option<MipStatus> {
        // Time limit
        if self.time_limit_exceeded() {
            return Some(if self.incumbent.has_incumbent() {
                MipStatus::TimeLimit
            } else {
                MipStatus::TimeLimit
            });
        }

        // Node limit
        if self.nodes_explored >= self.settings.max_nodes {
            return Some(if self.incumbent.has_incumbent() {
                MipStatus::NodeLimit
            } else {
                MipStatus::NodeLimit
            });
        }

        // Gap closed
        if self.incumbent.has_incumbent() && self.gap() <= self.settings.gap_tol {
            return Some(MipStatus::GapLimit);
        }

        // Queue empty
        if self.queue.is_empty() {
            return Some(if self.incumbent.has_incumbent() {
                MipStatus::Optimal
            } else {
                MipStatus::Infeasible
            });
        }

        None
    }

    /// Finalize the solve and return the solution.
    pub fn finalize(&self, status: MipStatus) -> MipSolution {
        MipSolution {
            status,
            x: self.incumbent.solution.clone().unwrap_or_default(),
            obj_val: self.incumbent.obj_val,
            bound: self.queue.best_bound(),
            gap: self.gap(),
            nodes_explored: self.nodes_explored,
            cuts_added: self.cuts_added,
            solve_time_ms: self.elapsed_ms(),
            incumbent_updates: self.incumbent.update_count,
        }
    }

    /// Log progress (if verbose).
    pub fn log_progress(&self) {
        if !self.settings.verbose {
            return;
        }

        if self.nodes_explored % self.settings.log_freq != 0 {
            return;
        }

        log::info!(
            "Nodes: {} ({} open) | Bound: {:.6e} | Incumbent: {:.6e} | Gap: {:.2}% | Cuts: {} | Time: {:.1}s",
            self.nodes_explored,
            self.queue.len(),
            self.queue.best_bound(),
            self.incumbent.obj_val,
            self.gap() * 100.0,
            self.cuts_added,
            self.elapsed_ms() as f64 / 1000.0,
        );
    }

    /// Get statistics for display.
    pub fn stats(&self) -> TreeStats {
        TreeStats {
            nodes_explored: self.nodes_explored,
            nodes_pruned: self.nodes_pruned,
            nodes_open: self.queue.len() as u64,
            cuts_added: self.cuts_added,
            incumbent_updates: self.incumbent.update_count,
            best_bound: self.queue.best_bound(),
            incumbent_obj: self.incumbent.obj_val,
            gap: self.gap(),
            elapsed_ms: self.elapsed_ms(),
        }
    }
}

/// Statistics from the B&B tree.
#[derive(Debug, Clone)]
pub struct TreeStats {
    pub nodes_explored: u64,
    pub nodes_pruned: u64,
    pub nodes_open: u64,
    pub cuts_added: u64,
    pub incumbent_updates: u64,
    pub best_bound: f64,
    pub incumbent_obj: f64,
    pub gap: f64,
    pub elapsed_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::MipSettings;

    #[test]
    fn test_tree_initialization() {
        let settings = MipSettings::default();
        let mut tree = BranchAndBound::new(settings, 10);

        tree.initialize(0.0);

        assert!(tree.next_node().is_some());
        assert!(tree.next_node().is_none()); // Queue now empty
    }

    #[test]
    fn test_incumbent_update() {
        let settings = MipSettings::default();
        let mut tree = BranchAndBound::new(settings, 10);
        tree.initialize(0.0);

        // First incumbent
        assert!(tree.update_incumbent(&vec![1.0; 10], 100.0));
        assert_eq!(tree.incumbent.obj_val, 100.0);

        // Worse solution rejected
        assert!(!tree.update_incumbent(&vec![2.0; 10], 150.0));
        assert_eq!(tree.incumbent.obj_val, 100.0);

        // Better solution accepted
        assert!(tree.update_incumbent(&vec![0.5; 10], 50.0));
        assert_eq!(tree.incumbent.obj_val, 50.0);
    }

    #[test]
    fn test_termination_gap() {
        let mut settings = MipSettings::default();
        settings.gap_tol = 0.1; // 10% gap

        let mut tree = BranchAndBound::new(settings, 10);
        tree.initialize(0.0);

        // Set incumbent to 100
        tree.update_incumbent(&vec![1.0; 10], 100.0);

        // Pop root so queue is empty, best_bound becomes infinity
        tree.next_node();

        // Queue empty with incumbent -> optimal
        assert_eq!(tree.check_termination(), Some(MipStatus::Optimal));
    }
}
