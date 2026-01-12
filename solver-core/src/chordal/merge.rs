//! Clique merging strategies for chordal decomposition.
//!
//! After finding maximal cliques, we may want to merge some small or
//! highly overlapping cliques to reduce overhead. The trade-off is:
//! - More cliques = smaller cones = better conditioning
//! - Fewer cliques = less overlap constraints = simpler problem

use super::cliques::{Clique, CliqueTree};
use std::collections::BinaryHeap;

/// Strategy for merging cliques.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// No merging - keep all cliques separate
    None,
    /// Parent-child merge based on fill-in
    ParentChild,
    /// Clique graph merge based on complexity weight (default)
    CliqueGraph,
}

impl Default for MergeStrategy {
    fn default() -> Self {
        Self::CliqueGraph
    }
}

/// Merge cliques according to the given strategy.
pub fn merge_cliques(tree: &CliqueTree, strategy: MergeStrategy) -> CliqueTree {
    match strategy {
        MergeStrategy::None => tree.clone(),
        MergeStrategy::ParentChild => parent_child_merge(tree),
        MergeStrategy::CliqueGraph => clique_graph_merge(tree),
    }
}

/// Parent-child merge: greedily merge children into parents.
fn parent_child_merge(tree: &CliqueTree) -> CliqueTree {
    if tree.cliques.len() <= 1 {
        return tree.clone();
    }

    let mut cliques = tree.cliques.clone();
    let mut parent = tree.parent.clone();
    let mut merged = vec![false; cliques.len()];

    // Process from leaves to root
    let mut order: Vec<usize> = (0..cliques.len()).collect();
    order.sort_by_key(|&i| {
        // Leaves first (no children)
        let num_children = tree.children[i].len();
        std::cmp::Reverse(num_children)
    });

    for &child_idx in &order {
        if merged[child_idx] {
            continue;
        }

        if let Some(parent_idx) = parent[child_idx] {
            if merged[parent_idx] {
                continue;
            }

            // Check if merge is beneficial
            let child_size = cliques[child_idx].size();
            let parent_size = cliques[parent_idx].size();
            let union_size = cliques[child_idx].union(&cliques[parent_idx]).len();

            // Compute complexity change
            let before_cost = complexity_cost(child_size) + complexity_cost(parent_size);
            let after_cost = complexity_cost(union_size);

            // Merge if it reduces or doesn't significantly increase cost
            // Use integer math: after_cost * 2 <= before_cost * 3
            if after_cost * 2 <= before_cost * 3 {
                // Merge child into parent
                let merged_vertices = cliques[child_idx].union(&cliques[parent_idx]);
                cliques[parent_idx] = Clique::new(merged_vertices);
                merged[child_idx] = true;

                // Update parent pointers for child's children
                for &grandchild in &tree.children[child_idx] {
                    if !merged[grandchild] {
                        parent[grandchild] = Some(parent_idx);
                    }
                }
            }
        }
    }

    // Build new tree from non-merged cliques
    rebuild_tree(&cliques, &parent, &merged)
}

/// Clique graph merge: merge based on complexity weight.
///
/// Weight formula: w(Ci, Cj) = |Ci|^3 + |Cj|^3 - |Ci ∪ Cj|^3
/// Higher weight = more beneficial to merge (saves more computation).
fn clique_graph_merge(tree: &CliqueTree) -> CliqueTree {
    if tree.cliques.len() <= 1 {
        return tree.clone();
    }

    let mut cliques = tree.cliques.clone();
    let mut active: Vec<bool> = vec![true; cliques.len()];

    // Build priority queue of merge candidates
    // Entry: (weight, clique_i, clique_j)
    #[derive(Debug, Clone, PartialEq)]
    struct MergeCandidate {
        weight: i64,
        i: usize,
        j: usize,
    }

    impl Eq for MergeCandidate {}

    impl PartialOrd for MergeCandidate {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for MergeCandidate {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            // Higher weight = higher priority
            self.weight.cmp(&other.weight)
        }
    }

    let mut heap = BinaryHeap::new();

    // Initialize with all adjacent pairs in tree
    for (i, p) in tree.parent.iter().enumerate() {
        if let Some(j) = *p {
            let weight = merge_weight(&cliques[i], &cliques[j]);
            heap.push(MergeCandidate { weight, i, j });
        }
    }

    // Also consider non-adjacent pairs with overlap
    for i in 0..cliques.len() {
        for j in i + 1..cliques.len() {
            if tree.parent[i] != Some(j) && tree.parent[j] != Some(i) {
                let overlap = cliques[i].intersection(&cliques[j]);
                if !overlap.is_empty() {
                    let weight = merge_weight(&cliques[i], &cliques[j]);
                    heap.push(MergeCandidate { weight, i, j });
                }
            }
        }
    }

    // Greedily merge while beneficial
    while let Some(candidate) = heap.pop() {
        let i = candidate.i;
        let j = candidate.j;

        if !active[i] || !active[j] {
            continue;
        }

        // Only merge if weight is non-negative (saves computation)
        if candidate.weight < 0 {
            break;
        }

        // Merge j into i
        let merged_vertices = cliques[i].union(&cliques[j]);
        cliques[i] = Clique::new(merged_vertices);
        active[j] = false;

        // Add new merge candidates for the merged clique
        for k in 0..cliques.len() {
            if k != i && active[k] {
                let overlap = cliques[i].intersection(&cliques[k]);
                if !overlap.is_empty() {
                    let weight = merge_weight(&cliques[i], &cliques[k]);
                    heap.push(MergeCandidate { weight, i, j: k });
                }
            }
        }
    }

    // Build new tree from active cliques
    let mut new_cliques = Vec::new();
    let mut old_to_new = vec![usize::MAX; cliques.len()];

    for (old_idx, clique) in cliques.into_iter().enumerate() {
        if active[old_idx] {
            old_to_new[old_idx] = new_cliques.len();
            new_cliques.push(clique);
        }
    }

    // Rebuild parent relationships
    let mut new_parent = vec![None; new_cliques.len()];

    // Use maximum spanning tree on intersection sizes
    if new_cliques.len() > 1 {
        let mut edges: Vec<(usize, usize, usize)> = Vec::new();
        for i in 0..new_cliques.len() {
            for j in i + 1..new_cliques.len() {
                let intersection = new_cliques[i].intersection(&new_cliques[j]);
                if !intersection.is_empty() {
                    edges.push((intersection.len(), i, j));
                }
            }
        }
        edges.sort_by(|a, b| b.0.cmp(&a.0));

        // Kruskal's for maximum spanning tree
        let mut uf: Vec<usize> = (0..new_cliques.len()).collect();
        fn find(uf: &mut [usize], i: usize) -> usize {
            if uf[i] != i {
                uf[i] = find(uf, uf[i]);
            }
            uf[i]
        }

        let mut tree_adj: Vec<Vec<usize>> = vec![vec![]; new_cliques.len()];
        for (_, i, j) in edges {
            let ri = find(&mut uf, i);
            let rj = find(&mut uf, j);
            if ri != rj {
                uf[ri] = rj;
                tree_adj[i].push(j);
                tree_adj[j].push(i);
            }
        }

        // BFS to assign parents (root at 0)
        let mut visited = vec![false; new_cliques.len()];
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(0);
        visited[0] = true;
        while let Some(v) = queue.pop_front() {
            for &u in &tree_adj[v] {
                if !visited[u] {
                    visited[u] = true;
                    new_parent[u] = Some(v);
                    queue.push_back(u);
                }
            }
        }
    }

    // Build separators and children
    let mut new_separator = vec![vec![]; new_cliques.len()];
    let mut new_children = vec![vec![]; new_cliques.len()];

    for (i, p) in new_parent.iter().enumerate() {
        if let Some(p_idx) = *p {
            new_separator[i] = new_cliques[i].intersection(&new_cliques[p_idx]);
            new_children[p_idx].push(i);
        }
    }

    CliqueTree {
        cliques: new_cliques,
        parent: new_parent,
        separator: new_separator,
        children: new_children,
    }
}

/// Compute merge weight: |Ci|^3 + |Cj|^3 - |Ci ∪ Cj|^3
/// Positive weight means merge saves computation.
fn merge_weight(a: &Clique, b: &Clique) -> i64 {
    let size_a = a.size() as i64;
    let size_b = b.size() as i64;
    let size_union = a.union(b).len() as i64;

    size_a * size_a * size_a + size_b * size_b * size_b - size_union * size_union * size_union
}

/// Compute complexity cost for a clique (roughly O(n^3) operations).
fn complexity_cost(n: usize) -> usize {
    n * n * n
}

/// Rebuild clique tree from surviving cliques.
fn rebuild_tree(
    cliques: &[Clique],
    old_parent: &[Option<usize>],
    merged: &[bool],
) -> CliqueTree {
    let mut new_cliques = Vec::new();
    let mut old_to_new = vec![usize::MAX; cliques.len()];

    for (old_idx, clique) in cliques.iter().enumerate() {
        if !merged[old_idx] {
            old_to_new[old_idx] = new_cliques.len();
            new_cliques.push(clique.clone());
        }
    }

    if new_cliques.is_empty() {
        return CliqueTree {
            cliques: vec![],
            parent: vec![],
            separator: vec![],
            children: vec![],
        };
    }

    // Map parent relationships
    let mut new_parent = vec![None; new_cliques.len()];
    for (old_idx, p) in old_parent.iter().enumerate() {
        if merged[old_idx] {
            continue;
        }
        if let Some(old_p) = *p {
            // Find non-merged ancestor
            let mut ancestor = old_p;
            while merged[ancestor] {
                if let Some(next) = old_parent[ancestor] {
                    ancestor = next;
                } else {
                    break;
                }
            }
            if !merged[ancestor] && ancestor != old_idx {
                new_parent[old_to_new[old_idx]] = Some(old_to_new[ancestor]);
            }
        }
    }

    // Build separators and children
    let mut new_separator = vec![vec![]; new_cliques.len()];
    let mut new_children = vec![vec![]; new_cliques.len()];

    for (i, p) in new_parent.iter().enumerate() {
        if let Some(p_idx) = *p {
            new_separator[i] = new_cliques[i].intersection(&new_cliques[p_idx]);
            new_children[p_idx].push(i);
        }
    }

    CliqueTree {
        cliques: new_cliques,
        parent: new_parent,
        separator: new_separator,
        children: new_children,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_weight_positive() {
        // Two size-2 cliques merging to size-3
        // 2^3 + 2^3 - 3^3 = 8 + 8 - 27 = -11 (not beneficial)
        let c1 = Clique::new(vec![0, 1]);
        let c2 = Clique::new(vec![1, 2]);
        let weight = merge_weight(&c1, &c2);
        assert!(weight < 0);
    }

    #[test]
    fn test_merge_weight_large_overlap() {
        // Two size-4 cliques with 3-element overlap (union = 5)
        // 4^3 + 4^3 - 5^3 = 64 + 64 - 125 = 3 (slightly beneficial)
        let c1 = Clique::new(vec![0, 1, 2, 3]);
        let c2 = Clique::new(vec![1, 2, 3, 4]);
        let weight = merge_weight(&c1, &c2);
        assert!(weight > 0);
    }
}
