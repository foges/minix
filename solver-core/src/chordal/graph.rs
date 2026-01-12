//! Sparsity graph construction and chordal completion.
//!
//! A graph G = (V, E) is chordal if every cycle of length > 3 has a chord.
//! The sparsity graph for an SDP has vertices 1..n (matrix indices) and edges
//! for every nonzero position in the constraint matrices.

use std::collections::BTreeSet;
use crate::linalg::sparse::SparseCsc;

/// Sparsity graph for a symmetric matrix.
#[derive(Debug, Clone)]
pub struct SparsityGraph {
    /// Matrix dimension
    pub n: usize,
    /// Adjacency list (sorted for each vertex)
    pub adj: Vec<BTreeSet<usize>>,
    /// Number of edges
    pub num_edges: usize,
}

impl SparsityGraph {
    /// Create empty sparsity graph.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            adj: vec![BTreeSet::new(); n],
            num_edges: 0,
        }
    }

    /// Create fully connected (dense) sparsity graph.
    pub fn dense(n: usize) -> Self {
        let mut g = Self::new(n);
        for i in 0..n {
            for j in i + 1..n {
                g.add_edge(i, j);
            }
        }
        g
    }

    /// Add an undirected edge.
    pub fn add_edge(&mut self, i: usize, j: usize) {
        if i != j && !self.adj[i].contains(&j) {
            self.adj[i].insert(j);
            self.adj[j].insert(i);
            self.num_edges += 1;
        }
    }

    /// Check if edge exists.
    pub fn has_edge(&self, i: usize, j: usize) -> bool {
        i != j && self.adj[i].contains(&j)
    }

    /// Get degree of vertex.
    pub fn degree(&self, v: usize) -> usize {
        self.adj[v].len()
    }

    /// Get neighbors of vertex.
    pub fn neighbors(&self, v: usize) -> &BTreeSet<usize> {
        &self.adj[v]
    }

    /// Build sparsity graph from constraint matrix columns.
    ///
    /// For columns in the range [offset, offset + svec_dim), extract the
    /// sparsity pattern and convert svec indices to matrix (i, j) positions.
    pub fn from_constraints(a: &SparseCsc, offset: usize, svec_dim: usize, n: usize) -> Self {
        let mut g = Self::new(n);

        // For each column in the constraint matrix that touches this PSD cone
        for col in 0..a.cols() {
            if let Some(col_view) = a.outer_view(col) {
                for (row, _val) in col_view.iter() {
                    if row >= offset && row < offset + svec_dim {
                        // Convert svec index to matrix (i, j)
                        let svec_idx = row - offset;
                        let (i, j) = svec_to_ij(svec_idx, n);
                        // Add edge (i, j) to sparsity graph
                        g.add_edge(i, j);
                    }
                }
            }
        }

        // Also add diagonal (always present in PSD)
        // No edges needed for diagonal - they're self-loops

        g
    }

    /// Check if graph is chordal using maximum cardinality search.
    pub fn is_chordal(&self) -> bool {
        if self.n <= 3 {
            return true;
        }

        // Use MCS to get perfect elimination ordering
        let ordering = self.maximum_cardinality_search();

        // Check if ordering is perfect elimination ordering
        self.is_perfect_elimination_ordering(&ordering)
    }

    /// Maximum cardinality search - produces perfect elimination ordering for chordal graphs.
    pub fn maximum_cardinality_search(&self) -> Vec<usize> {
        let mut ordering = Vec::with_capacity(self.n);
        let mut in_ordering = vec![false; self.n];
        let mut cardinality = vec![0usize; self.n];

        for _ in 0..self.n {
            // Find vertex with maximum cardinality not yet in ordering
            let v = (0..self.n)
                .filter(|&u| !in_ordering[u])
                .max_by_key(|&u| cardinality[u])
                .unwrap();

            ordering.push(v);
            in_ordering[v] = true;

            // Update cardinalities of neighbors
            for &u in &self.adj[v] {
                if !in_ordering[u] {
                    cardinality[u] += 1;
                }
            }
        }

        ordering
    }

    /// Check if ordering is a perfect elimination ordering.
    fn is_perfect_elimination_ordering(&self, ordering: &[usize]) -> bool {
        let n = self.n;
        let mut position = vec![0usize; n];
        for (pos, &v) in ordering.iter().enumerate() {
            position[v] = pos;
        }

        // For each vertex v, check that its earlier neighbors form a clique
        for (pos, &v) in ordering.iter().enumerate() {
            let earlier_neighbors: Vec<usize> = self.adj[v]
                .iter()
                .filter(|&&u| position[u] < pos)
                .copied()
                .collect();

            // Check all pairs of earlier neighbors are adjacent
            for i in 0..earlier_neighbors.len() {
                for j in i + 1..earlier_neighbors.len() {
                    if !self.has_edge(earlier_neighbors[i], earlier_neighbors[j]) {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Compute fill-in edges needed to make graph chordal.
    pub fn compute_fill_in(&self) -> Vec<(usize, usize)> {
        let mut fill_in = Vec::new();
        let mut g = self.clone();

        // Use minimum degree elimination
        let ordering = g.minimum_degree_ordering();

        for &v in &ordering {
            // Get neighbors of v that haven't been eliminated
            let neighbors: Vec<usize> = g.adj[v].iter().copied().collect();

            // Make neighbors into clique (add fill-in edges)
            for i in 0..neighbors.len() {
                for j in i + 1..neighbors.len() {
                    let u = neighbors[i];
                    let w = neighbors[j];
                    if !g.has_edge(u, w) {
                        g.add_edge(u, w);
                        fill_in.push((u.min(w), u.max(w)));
                    }
                }
            }

            // "Eliminate" v by removing all its edges
            for &u in &neighbors {
                g.adj[u].remove(&v);
            }
            g.adj[v].clear();
        }

        fill_in
    }

    /// Minimum degree ordering for fill-reducing elimination.
    fn minimum_degree_ordering(&self) -> Vec<usize> {
        let mut ordering = Vec::with_capacity(self.n);
        let mut eliminated = vec![false; self.n];
        let mut degree: Vec<usize> = (0..self.n).map(|v| self.degree(v)).collect();
        let mut adj = self.adj.clone();

        for _ in 0..self.n {
            // Find vertex with minimum degree
            let v = (0..self.n)
                .filter(|&u| !eliminated[u])
                .min_by_key(|&u| degree[u])
                .unwrap();

            ordering.push(v);
            eliminated[v] = true;

            // Update degrees for fill-in
            let neighbors: Vec<usize> = adj[v].iter().copied().collect();
            for i in 0..neighbors.len() {
                for j in i + 1..neighbors.len() {
                    let u = neighbors[i];
                    let w = neighbors[j];
                    if !adj[u].contains(&w) {
                        adj[u].insert(w);
                        adj[w].insert(u);
                        degree[u] += 1;
                        degree[w] += 1;
                    }
                }
            }

            // Remove v from neighbors' adjacency
            for &u in &neighbors {
                adj[u].remove(&v);
                degree[u] = degree[u].saturating_sub(1);
            }
        }

        ordering
    }
}

/// Chordal graph with perfect elimination ordering.
#[derive(Debug, Clone)]
pub struct ChordalGraph {
    /// Underlying sparsity graph (now chordal)
    pub graph: SparsityGraph,
    /// Perfect elimination ordering
    pub ordering: Vec<usize>,
    /// Fill-in edges that were added
    pub fill_in: Vec<(usize, usize)>,
}

impl ChordalGraph {
    /// Create chordal graph from sparsity graph, adding fill-in if needed.
    pub fn from_sparsity(mut sparsity: SparsityGraph) -> Self {
        // Check if already chordal
        let ordering = sparsity.maximum_cardinality_search();
        if sparsity.is_perfect_elimination_ordering(&ordering) {
            return Self {
                graph: sparsity,
                ordering,
                fill_in: vec![],
            };
        }

        // Compute and add fill-in edges
        let fill_in = sparsity.compute_fill_in();
        for &(i, j) in &fill_in {
            sparsity.add_edge(i, j);
        }

        // Get new perfect elimination ordering
        let ordering = sparsity.maximum_cardinality_search();
        debug_assert!(sparsity.is_perfect_elimination_ordering(&ordering));

        Self {
            graph: sparsity,
            ordering,
            fill_in,
        }
    }

    /// Get the graph dimension.
    pub fn n(&self) -> usize {
        self.graph.n
    }
}

/// Convert svec index to matrix (i, j) position.
/// svec uses column-major upper triangular: (0,0), (0,1), (1,1), (0,2), (1,2), (2,2), ...
fn svec_to_ij(idx: usize, n: usize) -> (usize, usize) {
    // Find column j such that j*(j+1)/2 <= idx < (j+1)*(j+2)/2
    let mut j = 0;
    while (j + 1) * (j + 2) / 2 <= idx {
        j += 1;
    }
    let i = idx - j * (j + 1) / 2;
    (i, j)
}

/// Convert matrix (i, j) position to svec index.
pub fn ij_to_svec(i: usize, j: usize) -> usize {
    let (i, j) = if i <= j { (i, j) } else { (j, i) };
    j * (j + 1) / 2 + i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svec_to_ij() {
        // For n=3: (0,0)=0, (0,1)=1, (1,1)=2, (0,2)=3, (1,2)=4, (2,2)=5
        assert_eq!(svec_to_ij(0, 3), (0, 0));
        assert_eq!(svec_to_ij(1, 3), (0, 1));
        assert_eq!(svec_to_ij(2, 3), (1, 1));
        assert_eq!(svec_to_ij(3, 3), (0, 2));
        assert_eq!(svec_to_ij(4, 3), (1, 2));
        assert_eq!(svec_to_ij(5, 3), (2, 2));
    }

    #[test]
    fn test_ij_to_svec() {
        assert_eq!(ij_to_svec(0, 0), 0);
        assert_eq!(ij_to_svec(0, 1), 1);
        assert_eq!(ij_to_svec(1, 0), 1); // symmetric
        assert_eq!(ij_to_svec(1, 1), 2);
        assert_eq!(ij_to_svec(2, 2), 5);
    }

    #[test]
    fn test_complete_graph_is_chordal() {
        let g = SparsityGraph::dense(5);
        assert!(g.is_chordal());
    }

    #[test]
    fn test_cycle_not_chordal() {
        // 4-cycle: 0-1-2-3-0
        let mut g = SparsityGraph::new(4);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(3, 0);
        assert!(!g.is_chordal());
    }

    #[test]
    fn test_chordal_completion() {
        // 4-cycle needs one fill-in edge
        let mut g = SparsityGraph::new(4);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(3, 0);

        let chordal = ChordalGraph::from_sparsity(g);
        assert!(chordal.graph.is_chordal());
        assert!(!chordal.fill_in.is_empty());
    }
}
