//! Search node representation.

/// Status of a search node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    /// Node is waiting to be processed.
    Pending,

    /// Node is currently being processed.
    Processing,

    /// Node was pruned (bound >= incumbent).
    Pruned,

    /// Node LP relaxation is infeasible.
    Infeasible,

    /// Node produced an integer-feasible solution.
    IntegerFeasible,

    /// Node was branched (children created).
    Branched,
}

/// A bound change from branching.
#[derive(Debug, Clone, Copy)]
pub struct BoundChange {
    /// Variable index.
    pub var: usize,

    /// Previous lower bound.
    pub old_lb: f64,

    /// Previous upper bound.
    pub old_ub: f64,

    /// New lower bound.
    pub new_lb: f64,

    /// New upper bound.
    pub new_ub: f64,
}

impl BoundChange {
    /// Create a "down" branch: x <= floor(value).
    pub fn down_branch(var: usize, old_lb: f64, old_ub: f64, value: f64) -> Self {
        Self {
            var,
            old_lb,
            old_ub,
            new_lb: old_lb,
            new_ub: value.floor(),
        }
    }

    /// Create an "up" branch: x >= ceil(value).
    pub fn up_branch(var: usize, old_lb: f64, old_ub: f64, value: f64) -> Self {
        Self {
            var,
            old_lb,
            old_ub,
            new_lb: value.ceil(),
            new_ub: old_ub,
        }
    }

    /// Check if the bound change creates an empty domain.
    pub fn is_infeasible(&self) -> bool {
        self.new_lb > self.new_ub + 1e-9
    }
}

/// A node in the B&B search tree.
#[derive(Debug, Clone)]
pub struct SearchNode {
    /// Unique node identifier.
    pub id: u64,

    /// Parent node ID (None for root).
    pub parent_id: Option<u64>,

    /// Depth in the tree (0 for root).
    pub depth: usize,

    /// Bound changes from parent to this node.
    pub bound_changes: Vec<BoundChange>,

    /// Dual bound at this node (from master LP).
    /// Lower bound on optimal objective in this subtree.
    pub dual_bound: f64,

    /// Estimate of best integer solution reachable.
    pub estimate: f64,

    /// Node processing status.
    pub status: NodeStatus,
}

impl SearchNode {
    /// Create the root node.
    pub fn root() -> Self {
        Self {
            id: 0,
            parent_id: None,
            depth: 0,
            bound_changes: Vec::new(),
            dual_bound: f64::NEG_INFINITY,
            estimate: f64::NEG_INFINITY,
            status: NodeStatus::Pending,
        }
    }

    /// Create a child node from a bound change.
    pub fn child(&self, id: u64, bound_change: BoundChange) -> Self {
        Self {
            id,
            parent_id: Some(self.id),
            depth: self.depth + 1,
            bound_changes: vec![bound_change],
            dual_bound: self.dual_bound, // Inherit parent's bound initially
            estimate: self.estimate,
            status: NodeStatus::Pending,
        }
    }

    /// Check if this node can be pruned by an incumbent.
    ///
    /// A node can be pruned if its dual bound >= incumbent objective.
    pub fn can_prune(&self, incumbent_obj: f64) -> bool {
        self.dual_bound >= incumbent_obj - 1e-9
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_root_node() {
        let root = SearchNode::root();
        assert_eq!(root.id, 0);
        assert!(root.parent_id.is_none());
        assert_eq!(root.depth, 0);
        assert!(root.bound_changes.is_empty());
        assert_eq!(root.status, NodeStatus::Pending);
    }

    #[test]
    fn test_child_node() {
        let root = SearchNode::root();
        let bc = BoundChange::down_branch(0, 0.0, 1.0, 0.5);
        let child = root.child(1, bc);

        assert_eq!(child.id, 1);
        assert_eq!(child.parent_id, Some(0));
        assert_eq!(child.depth, 1);
        assert_eq!(child.bound_changes.len(), 1);
    }

    #[test]
    fn test_bound_changes() {
        // Down branch on x with value 2.7: x <= 2
        let down = BoundChange::down_branch(0, 0.0, 5.0, 2.7);
        assert_eq!(down.new_lb, 0.0);
        assert_eq!(down.new_ub, 2.0);
        assert!(!down.is_infeasible());

        // Up branch on x with value 2.7: x >= 3
        let up = BoundChange::up_branch(0, 0.0, 5.0, 2.7);
        assert_eq!(up.new_lb, 3.0);
        assert_eq!(up.new_ub, 5.0);
        assert!(!up.is_infeasible());

        // Infeasible bound change
        let bad = BoundChange::down_branch(0, 3.0, 5.0, 2.7);
        assert!(bad.is_infeasible()); // new_ub = 2 < new_lb = 3
    }

    #[test]
    fn test_pruning() {
        let mut node = SearchNode::root();
        node.dual_bound = 10.0;

        // Incumbent 15: cannot prune (10 < 15)
        assert!(!node.can_prune(15.0));

        // Incumbent 10: can prune (10 >= 10)
        assert!(node.can_prune(10.0));

        // Incumbent 8: can prune (10 >= 8)
        assert!(node.can_prune(8.0));
    }
}
