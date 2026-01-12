//! Maximal clique enumeration and clique tree construction.
//!
//! For chordal graphs, maximal cliques can be found efficiently using the
//! perfect elimination ordering. The clique tree satisfies the "running
//! intersection property": for any two cliques, their intersection is
//! contained in all cliques on the path between them.

use super::graph::ChordalGraph;

/// A maximal clique in the chordal graph.
#[derive(Debug, Clone)]
pub struct Clique {
    /// Vertices in this clique (sorted)
    pub vertices: Vec<usize>,
}

impl Clique {
    /// Create a new clique from vertices.
    pub fn new(mut vertices: Vec<usize>) -> Self {
        vertices.sort_unstable();
        Self { vertices }
    }

    /// Size of the clique.
    pub fn size(&self) -> usize {
        self.vertices.len()
    }

    /// Check if vertex is in clique.
    pub fn contains(&self, v: usize) -> bool {
        self.vertices.binary_search(&v).is_ok()
    }

    /// Compute intersection with another clique.
    pub fn intersection(&self, other: &Clique) -> Vec<usize> {
        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;
        while i < self.vertices.len() && j < other.vertices.len() {
            match self.vertices[i].cmp(&other.vertices[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    result.push(self.vertices[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        result
    }

    /// Compute union with another clique.
    pub fn union(&self, other: &Clique) -> Vec<usize> {
        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;
        while i < self.vertices.len() || j < other.vertices.len() {
            if i >= self.vertices.len() {
                result.push(other.vertices[j]);
                j += 1;
            } else if j >= other.vertices.len() {
                result.push(self.vertices[i]);
                i += 1;
            } else if self.vertices[i] < other.vertices[j] {
                result.push(self.vertices[i]);
                i += 1;
            } else if self.vertices[i] > other.vertices[j] {
                result.push(other.vertices[j]);
                j += 1;
            } else {
                result.push(self.vertices[i]);
                i += 1;
                j += 1;
            }
        }
        result
    }

    /// Compute svec dimension for this clique.
    pub fn svec_dim(&self) -> usize {
        let n = self.size();
        n * (n + 1) / 2
    }
}

/// Clique tree with running intersection property.
#[derive(Debug, Clone)]
pub struct CliqueTree {
    /// Maximal cliques
    pub cliques: Vec<Clique>,
    /// Parent of each clique in the tree (None for root)
    pub parent: Vec<Option<usize>>,
    /// Separator (intersection with parent) for each clique
    pub separator: Vec<Vec<usize>>,
    /// Children of each clique
    pub children: Vec<Vec<usize>>,
}

impl CliqueTree {
    /// Build clique tree from chordal graph using perfect elimination ordering.
    pub fn from_chordal(chordal: &ChordalGraph) -> Self {
        let n = chordal.n();
        if n == 0 {
            return Self {
                cliques: vec![],
                parent: vec![],
                separator: vec![],
                children: vec![],
            };
        }

        // Find maximal cliques using perfect elimination ordering
        let cliques = Self::find_maximal_cliques(chordal);

        if cliques.is_empty() {
            return Self {
                cliques: vec![],
                parent: vec![],
                separator: vec![],
                children: vec![],
            };
        }

        // Build clique tree using maximum spanning tree on intersection sizes
        let (parent, separator, children) = Self::build_tree(&cliques);

        Self {
            cliques,
            parent,
            separator,
            children,
        }
    }

    /// Find maximal cliques using perfect elimination ordering.
    fn find_maximal_cliques(chordal: &ChordalGraph) -> Vec<Clique> {
        let n = chordal.n();
        let ordering = &chordal.ordering;
        let graph = &chordal.graph;

        // Position in ordering (inverse permutation)
        let mut position = vec![0usize; n];
        for (pos, &v) in ordering.iter().enumerate() {
            position[v] = pos;
        }

        let mut cliques: Vec<Clique> = Vec::new();
        let mut clique_of_vertex: Vec<Option<usize>> = vec![None; n];

        // Process vertices in reverse elimination order
        for &v in ordering.iter().rev() {
            // Get neighbors that come later in ordering
            let later_neighbors: Vec<usize> = graph.adj[v]
                .iter()
                .filter(|&&u| position[u] > position[v])
                .copied()
                .collect();

            if later_neighbors.is_empty() {
                // v forms a singleton clique (or is absorbed into existing)
                // Check if there's an existing clique containing just v
                let mut found = false;
                for (idx, c) in cliques.iter().enumerate() {
                    if c.size() == 1 && c.vertices[0] == v {
                        found = true;
                        clique_of_vertex[v] = Some(idx);
                        break;
                    }
                }
                if !found {
                    clique_of_vertex[v] = Some(cliques.len());
                    cliques.push(Clique::new(vec![v]));
                }
            } else {
                // Find the clique that contains v and all its later neighbors
                // This is v ∪ later_neighbors
                let mut clique_vertices = vec![v];
                clique_vertices.extend(later_neighbors.iter().copied());
                clique_vertices.sort_unstable();

                // Check if this is a subset of an existing clique
                let mut is_maximal = true;
                for (idx, c) in cliques.iter().enumerate() {
                    if clique_vertices.iter().all(|&u| c.contains(u)) {
                        // This clique is contained in existing clique
                        is_maximal = false;
                        clique_of_vertex[v] = Some(idx);
                        break;
                    }
                }

                if is_maximal {
                    clique_of_vertex[v] = Some(cliques.len());
                    cliques.push(Clique::new(clique_vertices));
                }
            }
        }

        // Remove non-maximal cliques (those contained in others)
        let mut maximal = vec![true; cliques.len()];
        for i in 0..cliques.len() {
            for j in 0..cliques.len() {
                if i != j && maximal[i] && maximal[j] {
                    // Check if clique i is subset of clique j
                    if cliques[i].vertices.iter().all(|v| cliques[j].contains(*v)) {
                        maximal[i] = false;
                    }
                }
            }
        }

        cliques
            .into_iter()
            .enumerate()
            .filter(|(i, _)| maximal[*i])
            .map(|(_, c)| c)
            .collect()
    }

    /// Build clique tree using maximum spanning tree on intersection sizes.
    fn build_tree(cliques: &[Clique]) -> (Vec<Option<usize>>, Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let num_cliques = cliques.len();
        if num_cliques == 0 {
            return (vec![], vec![], vec![]);
        }
        if num_cliques == 1 {
            return (vec![None], vec![vec![]], vec![vec![]]);
        }

        // Compute intersection sizes between all pairs
        let mut edges: Vec<(usize, usize, usize)> = Vec::new();
        for i in 0..num_cliques {
            for j in i + 1..num_cliques {
                let intersection = cliques[i].intersection(&cliques[j]);
                if !intersection.is_empty() {
                    edges.push((intersection.len(), i, j));
                }
            }
        }

        // Sort by intersection size (descending) for maximum spanning tree
        edges.sort_by(|a, b| b.0.cmp(&a.0));

        // Kruskal's algorithm for maximum spanning tree
        let mut parent = vec![None; num_cliques];
        let mut uf_parent: Vec<usize> = (0..num_cliques).collect();
        let mut uf_rank = vec![0usize; num_cliques];
        let mut tree_edges = Vec::new();

        fn find(uf_parent: &mut [usize], i: usize) -> usize {
            if uf_parent[i] != i {
                uf_parent[i] = find(uf_parent, uf_parent[i]);
            }
            uf_parent[i]
        }

        fn union(uf_parent: &mut [usize], uf_rank: &mut [usize], i: usize, j: usize) -> bool {
            let ri = find(uf_parent, i);
            let rj = find(uf_parent, j);
            if ri == rj {
                return false;
            }
            if uf_rank[ri] < uf_rank[rj] {
                uf_parent[ri] = rj;
            } else if uf_rank[ri] > uf_rank[rj] {
                uf_parent[rj] = ri;
            } else {
                uf_parent[rj] = ri;
                uf_rank[ri] += 1;
            }
            true
        }

        for (_, i, j) in edges {
            if union(&mut uf_parent, &mut uf_rank, i, j) {
                tree_edges.push((i, j));
                if tree_edges.len() == num_cliques - 1 {
                    break;
                }
            }
        }

        // Convert undirected tree edges to parent-child relationships (root at 0)
        let mut adj: Vec<Vec<usize>> = vec![vec![]; num_cliques];
        for &(i, j) in &tree_edges {
            adj[i].push(j);
            adj[j].push(i);
        }

        // BFS from root to assign parents
        let mut visited = vec![false; num_cliques];
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(0);
        visited[0] = true;

        while let Some(v) = queue.pop_front() {
            for &u in &adj[v] {
                if !visited[u] {
                    visited[u] = true;
                    parent[u] = Some(v);
                    queue.push_back(u);
                }
            }
        }

        // Compute separators and children
        let mut separator = vec![vec![]; num_cliques];
        let mut children = vec![vec![]; num_cliques];

        for (i, p) in parent.iter().enumerate() {
            if let Some(p_idx) = *p {
                separator[i] = cliques[i].intersection(&cliques[p_idx]);
                children[p_idx].push(i);
            }
        }

        (parent, separator, children)
    }

    /// Get total number of cliques.
    pub fn num_cliques(&self) -> usize {
        self.cliques.len()
    }

    /// Check if decomposition is trivial (single clique = original cone).
    pub fn is_trivial(&self) -> bool {
        self.cliques.len() <= 1
    }

    /// Compute total svec dimension across all cliques.
    pub fn total_svec_dim(&self) -> usize {
        self.cliques.iter().map(|c| c.svec_dim()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chordal::graph::SparsityGraph;

    #[test]
    fn test_clique_intersection() {
        let c1 = Clique::new(vec![0, 1, 2]);
        let c2 = Clique::new(vec![1, 2, 3]);
        assert_eq!(c1.intersection(&c2), vec![1, 2]);
    }

    #[test]
    fn test_clique_union() {
        let c1 = Clique::new(vec![0, 1, 2]);
        let c2 = Clique::new(vec![1, 2, 3]);
        assert_eq!(c1.union(&c2), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_complete_graph_single_clique() {
        let g = SparsityGraph::dense(4);
        let chordal = super::super::graph::ChordalGraph::from_sparsity(g);
        let tree = CliqueTree::from_chordal(&chordal);

        // Complete graph has exactly one maximal clique
        assert_eq!(tree.num_cliques(), 1);
        assert_eq!(tree.cliques[0].size(), 4);
    }

    #[test]
    fn test_path_graph_cliques() {
        // Path 0-1-2-3: cliques are {0,1}, {1,2}, {2,3}
        let mut g = SparsityGraph::new(4);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);

        let chordal = super::super::graph::ChordalGraph::from_sparsity(g);
        let tree = CliqueTree::from_chordal(&chordal);

        assert_eq!(tree.num_cliques(), 3);
        for c in &tree.cliques {
            assert_eq!(c.size(), 2);
        }
    }
}
