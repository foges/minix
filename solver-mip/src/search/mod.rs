//! Branch-and-bound search tree management.

mod node;
mod queue;
mod branching;
mod tree;

pub use node::{BoundChange, NodeStatus, SearchNode};
pub use queue::NodeQueue;
pub use branching::{BranchDecision, BranchingSelector};
pub use tree::BranchAndBound;
