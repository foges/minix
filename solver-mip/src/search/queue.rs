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
        }
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
        let queued = self.heap.pop()?;
        self.nodes_popped += 1;

        // Recompute best bound
        self.recompute_best_bound();

        Some(queued.node)
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
