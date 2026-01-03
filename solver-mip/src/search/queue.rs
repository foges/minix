//! Node priority queue for B&B tree exploration.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::SearchNode;
use crate::settings::NodeSelection;

/// Entry in the node queue with priority.
struct QueuedNode {
    node: SearchNode,
    priority: f64, // Higher = selected first
}

impl PartialEq for QueuedNode {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for QueuedNode {}

impl PartialOrd for QueuedNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueuedNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first
        self.priority
            .partial_cmp(&other.priority)
            .unwrap_or(Ordering::Equal)
    }
}

/// Priority queue for B&B nodes.
pub struct NodeQueue {
    /// Node selection strategy.
    strategy: NodeSelection,

    /// Priority queue (max-heap by priority).
    heap: BinaryHeap<QueuedNode>,

    /// Count of nodes added.
    nodes_added: u64,

    /// Count of nodes popped.
    nodes_popped: u64,

    /// Best (lowest) dual bound in queue.
    best_bound: f64,

    /// ID of last processed node (for plunging).
    last_node_id: Option<u64>,

    /// Whether we have found an incumbent (for two-phase).
    has_incumbent: bool,

    /// Current plunge depth.
    plunge_depth: usize,
}

impl NodeQueue {
    /// Create a new node queue with the given strategy.
    pub fn new(strategy: NodeSelection) -> Self {
        Self {
            strategy,
            heap: BinaryHeap::new(),
            nodes_added: 0,
            nodes_popped: 0,
            best_bound: f64::NEG_INFINITY,
            last_node_id: None,
            has_incumbent: false,
            plunge_depth: 0,
        }
    }

    /// Notify that an incumbent has been found.
    pub fn set_has_incumbent(&mut self, has: bool) {
        self.has_incumbent = has;
    }

    /// Reset plunge depth (called when backtracking).
    pub fn reset_plunge(&mut self) {
        self.plunge_depth = 0;
    }

    /// Add a node to the queue.
    pub fn push(&mut self, node: SearchNode) {
        let priority = self.compute_priority(&node);

        // Update best bound
        if node.dual_bound < self.best_bound || self.heap.is_empty() {
            self.best_bound = node.dual_bound;
        }

        self.heap.push(QueuedNode { node, priority });
        self.nodes_added += 1;
    }

    /// Get the next node to process.
    pub fn pop(&mut self) -> Option<SearchNode> {
        let queued = self.pop_with_strategy()?;
        self.nodes_popped += 1;
        self.last_node_id = Some(queued.node.id);

        // Update plunge depth
        if let Some(last_id) = self.last_node_id {
            if queued.node.parent_id == Some(last_id) {
                self.plunge_depth += 1;
            } else {
                self.plunge_depth = 0;
            }
        }

        // Recompute best bound
        self.recompute_best_bound();

        Some(queued.node)
    }

    /// Pop with strategy-specific logic.
    fn pop_with_strategy(&mut self) -> Option<QueuedNode> {
        match self.strategy {
            NodeSelection::TwoPhase => {
                if !self.has_incumbent {
                    // Depth-first until incumbent found
                    self.pop_by_depth()
                } else {
                    // Best-bound after incumbent
                    self.heap.pop()
                }
            }
            NodeSelection::Plunging { max_plunge_depth } => {
                // Try to continue plunging if possible
                if self.plunge_depth < max_plunge_depth {
                    if let Some(child) = self.pop_child_of_last() {
                        return Some(child);
                    }
                }
                // Fall back to best-bound
                self.heap.pop()
            }
            NodeSelection::Restarts { restart_freq } => {
                if self.nodes_popped > 0 && self.nodes_popped % restart_freq == 0 {
                    // Restart: pick best-bound
                    self.heap.pop()
                } else {
                    // Normal: depth-first
                    self.pop_by_depth()
                }
            }
            _ => {
                // Other strategies use priority-based selection
                self.heap.pop()
            }
        }
    }

    /// Pop the deepest node (for depth-first variants).
    fn pop_by_depth(&mut self) -> Option<QueuedNode> {
        if self.heap.is_empty() {
            return None;
        }

        // Find deepest node
        let mut deepest_idx = 0;
        let mut max_depth = 0;
        for (i, q) in self.heap.iter().enumerate() {
            if q.node.depth > max_depth {
                max_depth = q.node.depth;
                deepest_idx = i;
            }
        }

        // Remove and return (inefficient, but simple)
        let nodes: Vec<_> = self.heap.drain().collect();
        let mut result = None;
        for (i, node) in nodes.into_iter().enumerate() {
            if i == deepest_idx {
                result = Some(node);
            } else {
                self.heap.push(node);
            }
        }
        result
    }

    /// Pop a child of the last processed node (for plunging).
    fn pop_child_of_last(&mut self) -> Option<QueuedNode> {
        let last_id = self.last_node_id?;

        // Find a child of the last node
        let child_idx = self
            .heap
            .iter()
            .position(|q| q.node.parent_id == Some(last_id));

        if let Some(idx) = child_idx {
            // Remove and return the child
            let nodes: Vec<_> = self.heap.drain().collect();
            let mut result = None;
            for (i, node) in nodes.into_iter().enumerate() {
                if i == idx {
                    result = Some(node);
                } else {
                    self.heap.push(node);
                }
            }
            result
        } else {
            None
        }
    }

    /// Peek at the next node without removing it.
    pub fn peek(&self) -> Option<&SearchNode> {
        self.heap.peek().map(|q| &q.node)
    }

    /// Get the best (lowest) dual bound across all nodes.
    pub fn best_bound(&self) -> f64 {
        self.best_bound
    }

    /// Prune nodes that are dominated by the incumbent.
    ///
    /// Returns the number of pruned nodes.
    pub fn prune_by_bound(&mut self, incumbent_obj: f64) -> usize {
        let before = self.heap.len();

        // Drain heap, keeping only nodes that can't be pruned
        let remaining: Vec<QueuedNode> = self
            .heap
            .drain()
            .filter(|q| !q.node.can_prune(incumbent_obj))
            .collect();

        self.heap = remaining.into_iter().collect();
        self.recompute_best_bound();

        before - self.heap.len()
    }

    /// Check if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Get the number of nodes in the queue.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Get the total number of nodes added.
    pub fn total_added(&self) -> u64 {
        self.nodes_added
    }

    /// Get the total number of nodes popped.
    pub fn total_popped(&self) -> u64 {
        self.nodes_popped
    }

    /// Compute priority for a node based on selection strategy.
    fn compute_priority(&self, node: &SearchNode) -> f64 {
        match self.strategy {
            NodeSelection::BestBound => {
                // Lowest dual bound first (negate for max-heap)
                -node.dual_bound
            }
            NodeSelection::DepthFirst => {
                // Deepest first
                node.depth as f64
            }
            NodeSelection::BestEstimate => {
                // Lowest estimate first
                -node.estimate
            }
            NodeSelection::Hybrid { dive_freq } => {
                // Alternate between diving and best-bound
                if self.nodes_popped % dive_freq as u64 == 0 {
                    node.depth as f64
                } else {
                    -node.dual_bound
                }
            }
            NodeSelection::TwoPhase => {
                // Priority based on phase (handled in pop_with_strategy)
                if self.has_incumbent {
                    -node.dual_bound
                } else {
                    node.depth as f64
                }
            }
            NodeSelection::Plunging { .. } => {
                // Prefer children of current node, then best-bound
                // Priority is mainly for fallback
                -node.dual_bound
            }
            NodeSelection::Restarts { .. } => {
                // Use depth for normal selection
                node.depth as f64
            }
        }
    }

    /// Recompute best bound after removal.
    fn recompute_best_bound(&mut self) {
        self.best_bound = self
            .heap
            .iter()
            .map(|q| q.node.dual_bound)
            .fold(f64::INFINITY, f64::min);

        if self.heap.is_empty() {
            self.best_bound = f64::INFINITY;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_best_bound_selection() {
        let mut queue = NodeQueue::new(NodeSelection::BestBound);

        let mut n1 = SearchNode::root();
        n1.id = 1;
        n1.dual_bound = 10.0;

        let mut n2 = SearchNode::root();
        n2.id = 2;
        n2.dual_bound = 5.0;

        let mut n3 = SearchNode::root();
        n3.id = 3;
        n3.dual_bound = 15.0;

        queue.push(n1);
        queue.push(n2);
        queue.push(n3);

        assert_eq!(queue.best_bound(), 5.0);

        // Best bound (lowest) should come first
        let first = queue.pop().unwrap();
        assert_eq!(first.id, 2);
        assert_eq!(first.dual_bound, 5.0);

        let second = queue.pop().unwrap();
        assert_eq!(second.id, 1);

        let third = queue.pop().unwrap();
        assert_eq!(third.id, 3);

        assert!(queue.is_empty());
    }

    #[test]
    fn test_depth_first_selection() {
        let mut queue = NodeQueue::new(NodeSelection::DepthFirst);

        let mut n1 = SearchNode::root();
        n1.id = 1;
        n1.depth = 0;

        let mut n2 = SearchNode::root();
        n2.id = 2;
        n2.depth = 2;

        let mut n3 = SearchNode::root();
        n3.id = 3;
        n3.depth = 1;

        queue.push(n1);
        queue.push(n2);
        queue.push(n3);

        // Deepest first
        assert_eq!(queue.pop().unwrap().id, 2);
        assert_eq!(queue.pop().unwrap().id, 3);
        assert_eq!(queue.pop().unwrap().id, 1);
    }

    #[test]
    fn test_pruning() {
        let mut queue = NodeQueue::new(NodeSelection::BestBound);

        for i in 0..5 {
            let mut node = SearchNode::root();
            node.id = i;
            node.dual_bound = i as f64 * 10.0; // 0, 10, 20, 30, 40
            queue.push(node);
        }

        assert_eq!(queue.len(), 5);

        // Prune nodes with bound >= 25
        let pruned = queue.prune_by_bound(25.0);
        assert_eq!(pruned, 2); // nodes with bound 30 and 40
        assert_eq!(queue.len(), 3);
    }
}
