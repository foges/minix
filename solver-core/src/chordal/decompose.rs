//! Problem transformation for chordal decomposition.
//!
//! Transforms the original problem by replacing large PSD cones with
//! smaller overlapping PSD cones corresponding to maximal cliques.

use super::cliques::{Clique, CliqueTree};
use super::graph::ij_to_svec;
use super::ChordalAnalysis;
use crate::linalg::sparse;
use crate::problem::ProblemData;
use crate::ConeSpec;
use std::collections::HashMap;

/// Entry selector mapping between clique and original matrix.
#[derive(Debug, Clone)]
pub struct EntrySelector {
    /// Clique index
    pub clique_idx: usize,
    /// Size of clique (n for PSD(n))
    pub clique_n: usize,
    /// Maps svec index in clique to svec index in original
    pub to_original: Vec<usize>,
    /// Maps svec index in original to svec index in clique (if present)
    pub from_original: HashMap<usize, usize>,
}

impl EntrySelector {
    /// Create entry selector for a clique.
    pub fn new(clique_idx: usize, clique: &Clique, original_n: usize) -> Self {
        let clique_n = clique.size();
        let clique_svec_dim = clique_n * (clique_n + 1) / 2;

        let mut to_original = Vec::with_capacity(clique_svec_dim);
        let mut from_original = HashMap::new();

        // Map (i_clique, j_clique) in clique to (i_orig, j_orig) in original
        for j_clique in 0..clique_n {
            for i_clique in 0..=j_clique {
                let i_orig = clique.vertices[i_clique];
                let j_orig = clique.vertices[j_clique];
                let orig_svec_idx = ij_to_svec(i_orig, j_orig);
                let clique_svec_idx = ij_to_svec(i_clique, j_clique);

                assert_eq!(to_original.len(), clique_svec_idx);
                to_original.push(orig_svec_idx);
                from_original.insert(orig_svec_idx, clique_svec_idx);
            }
        }

        Self {
            clique_idx,
            clique_n,
            to_original,
            from_original,
        }
    }

    /// Get svec dimension for this clique.
    pub fn svec_dim(&self) -> usize {
        self.to_original.len()
    }
}

/// Overlap constraint between two cliques.
#[derive(Debug, Clone)]
pub struct OverlapConstraint {
    /// First clique index
    pub clique_a: usize,
    /// Second clique index
    pub clique_b: usize,
    /// Overlapping vertex indices (in original numbering)
    pub overlap_vertices: Vec<usize>,
    /// Maps (i,j) in overlap to svec index in clique A
    pub a_indices: Vec<usize>,
    /// Maps (i,j) in overlap to svec index in clique B
    pub b_indices: Vec<usize>,
}

impl OverlapConstraint {
    /// Create overlap constraint between two cliques.
    pub fn new(
        clique_a: usize,
        clique_b: usize,
        cliques: &[Clique],
        selectors: &[EntrySelector],
    ) -> Option<Self> {
        let overlap_vertices = cliques[clique_a].intersection(&cliques[clique_b]);
        if overlap_vertices.is_empty() {
            return None;
        }

        let selector_a = &selectors[clique_a];
        let selector_b = &selectors[clique_b];

        let mut a_indices = Vec::new();
        let mut b_indices = Vec::new();

        // For each (i, j) pair in the overlap, find svec indices in both cliques
        let overlap_size = overlap_vertices.len();
        for j_idx in 0..overlap_size {
            for i_idx in 0..=j_idx {
                let i_orig = overlap_vertices[i_idx];
                let j_orig = overlap_vertices[j_idx];
                let orig_svec = ij_to_svec(i_orig, j_orig);

                if let (Some(&a_idx), Some(&b_idx)) = (
                    selector_a.from_original.get(&orig_svec),
                    selector_b.from_original.get(&orig_svec),
                ) {
                    a_indices.push(a_idx);
                    b_indices.push(b_idx);
                }
            }
        }

        Some(Self {
            clique_a,
            clique_b,
            overlap_vertices,
            a_indices,
            b_indices,
        })
    }

    /// Number of overlap constraints (svec entries).
    pub fn num_constraints(&self) -> usize {
        self.a_indices.len()
    }
}

/// Decomposition data for a single PSD cone.
#[derive(Debug, Clone)]
pub struct PsdDecomposition {
    /// Original matrix size
    pub original_n: usize,
    /// Original svec dimension
    pub original_svec_dim: usize,
    /// Offset in original constraint matrix
    pub offset: usize,
    /// Clique tree after merging
    pub clique_tree: CliqueTree,
    /// Entry selectors for each clique
    pub selectors: Vec<EntrySelector>,
    /// Overlap constraints between adjacent cliques
    pub overlaps: Vec<OverlapConstraint>,
}

impl PsdDecomposition {
    /// Create decomposition from merged clique tree.
    pub fn new(original_n: usize, clique_tree: CliqueTree, offset: usize) -> Self {
        let original_svec_dim = original_n * (original_n + 1) / 2;

        // Create entry selectors
        let selectors: Vec<EntrySelector> = clique_tree
            .cliques
            .iter()
            .enumerate()
            .map(|(idx, c)| EntrySelector::new(idx, c, original_n))
            .collect();

        // Create overlap constraints for adjacent cliques in tree
        let mut overlaps = Vec::new();
        for (child_idx, parent_opt) in clique_tree.parent.iter().enumerate() {
            if let Some(parent_idx) = *parent_opt {
                if let Some(overlap) =
                    OverlapConstraint::new(child_idx, parent_idx, &clique_tree.cliques, &selectors)
                {
                    overlaps.push(overlap);
                }
            }
        }

        Self {
            original_n,
            original_svec_dim,
            offset,
            clique_tree,
            selectors,
            overlaps,
        }
    }

    /// Check if decomposition is beneficial.
    pub fn is_beneficial(&self) -> bool {
        // Beneficial if we have multiple cliques and they're smaller than original
        if self.clique_tree.num_cliques() <= 1 {
            return false;
        }

        // Check that largest clique is significantly smaller than original
        let max_clique_size = self
            .clique_tree
            .cliques
            .iter()
            .map(|c| c.size())
            .max()
            .unwrap_or(0);

        max_clique_size < self.original_n
    }

    /// Get list of new PSD cone specs.
    pub fn cone_specs(&self) -> Vec<ConeSpec> {
        self.clique_tree
            .cliques
            .iter()
            .map(|c| ConeSpec::Psd { n: c.size() })
            .collect()
    }

    /// Get total svec dimension across all cliques.
    pub fn total_svec_dim(&self) -> usize {
        self.selectors.iter().map(|s| s.svec_dim()).sum()
    }

    /// Get total number of overlap constraints.
    pub fn total_overlap_constraints(&self) -> usize {
        self.overlaps.iter().map(|o| o.num_constraints()).sum()
    }
}

/// Container for all decomposition data.
#[derive(Debug, Clone)]
pub struct DecomposedPsd {
    /// Decomposition for each decomposed PSD cone
    pub decompositions: Vec<PsdDecomposition>,
    /// Original cone indices that were decomposed
    pub original_cone_indices: Vec<usize>,
    /// Mapping from new cone index to (decomp_idx, clique_idx)
    pub cone_mapping: Vec<(usize, usize)>,
    /// Offset of each new PSD cone in the decomposed slack vector
    pub new_cone_offsets: Vec<usize>,
    /// Number of new overlap equality constraints added
    pub num_overlap_constraints: usize,
}

/// Transform a problem using chordal decomposition.
pub fn transform_problem(
    prob: &ProblemData,
    analysis: &ChordalAnalysis,
) -> (ProblemData, DecomposedPsd) {
    if !analysis.beneficial || analysis.decompositions.is_empty() {
        // No decomposition - return original problem
        let decomposed = DecomposedPsd {
            decompositions: vec![],
            original_cone_indices: vec![],
            cone_mapping: vec![],
            new_cone_offsets: vec![],
            num_overlap_constraints: 0,
        };
        return (prob.clone(), decomposed);
    }

    // Build new cone list
    let mut new_cones = Vec::new();
    let mut cone_mapping = Vec::new();
    let mut new_cone_offsets = Vec::new();
    let mut current_offset = 0;

    let decomposed_set: std::collections::HashSet<usize> =
        analysis.decomposed_cones.iter().copied().collect();

    for (cone_idx, cone) in prob.cones.iter().enumerate() {
        if decomposed_set.contains(&cone_idx) {
            // Find the decomposition for this cone
            let decomp_idx = analysis
                .decomposed_cones
                .iter()
                .position(|&i| i == cone_idx)
                .unwrap();
            let decomp = &analysis.decompositions[decomp_idx];

            // Add decomposed PSD cones
            for (clique_idx, clique) in decomp.clique_tree.cliques.iter().enumerate() {
                new_cones.push(ConeSpec::Psd { n: clique.size() });
                cone_mapping.push((decomp_idx, clique_idx));
                new_cone_offsets.push(current_offset);
                current_offset += clique.svec_dim();
            }
        } else {
            // Keep original cone
            new_cones.push(cone.clone());
            cone_mapping.push((usize::MAX, cone_idx)); // Mark as non-decomposed
            new_cone_offsets.push(current_offset);
            current_offset += cone.dim();
        }
    }

    // Calculate total overlap constraints
    let num_overlap_constraints: usize = analysis
        .decompositions
        .iter()
        .map(|d| d.total_overlap_constraints())
        .sum();

    // Build new constraint matrix A
    // Original constraints + overlap constraints
    let new_m = prob.A.rows() + num_overlap_constraints;
    let new_n = prob.A.cols(); // Variables unchanged

    // For now, we'll build the new A matrix by:
    // 1. Copying original constraints with column mappings for decomposed cones
    // 2. Adding overlap constraints

    let mut triplets = Vec::new();

    // Copy original constraints
    for col in 0..prob.A.cols() {
        if let Some(col_view) = prob.A.outer_view(col) {
            for (row, &val) in col_view.iter() {
                triplets.push((row, col, val));
            }
        }
    }

    // Add overlap constraints
    // For each overlap, add constraint: x_a[i] - x_b[i] = 0
    // These constrain the slack variables to agree on overlapping entries
    // Note: This is a simplified placeholder. Full implementation would
    // properly integrate overlap constraints into the cone structure.
    let _ = num_overlap_constraints; // Suppress unused warning for now

    // Build new b vector
    let mut new_b = prob.b.clone();
    new_b.extend(vec![0.0; num_overlap_constraints]);

    // Build new problem (simplified - full impl would handle slacks properly)
    let new_a = sparse::from_triplets(new_m, new_n, triplets);

    let new_prob = ProblemData {
        P: prob.P.clone(),
        q: prob.q.clone(),
        A: new_a,
        b: new_b,
        cones: new_cones,
        var_bounds: prob.var_bounds.clone(),
        integrality: prob.integrality.clone(),
    };

    let decomposed = DecomposedPsd {
        decompositions: analysis.decompositions.clone(),
        original_cone_indices: analysis.decomposed_cones.clone(),
        cone_mapping,
        new_cone_offsets,
        num_overlap_constraints,
    };

    (new_prob, decomposed)
}

/// Recover original solution from decomposed solution.
pub fn recover_solution(
    decomposed: &DecomposedPsd,
    x: &[f64],
    s: &[f64],
    z: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if decomposed.decompositions.is_empty() {
        return (x.to_vec(), s.to_vec(), z.to_vec());
    }

    // For now, return as-is
    // Full implementation would:
    // 1. Extract clique solutions
    // 2. Assemble original s from overlapping clique solutions
    // 3. Complete z using PSD completion

    (x.to_vec(), s.to_vec(), z.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entry_selector() {
        let clique = Clique::new(vec![0, 2, 3]); // Indices 0, 2, 3 from 4x4 matrix
        let selector = EntrySelector::new(0, &clique, 4);

        assert_eq!(selector.clique_n, 3);
        assert_eq!(selector.svec_dim(), 6); // 3*4/2 = 6

        // (0,0) in clique -> (0,0) in original = svec idx 0
        assert_eq!(selector.to_original[0], 0);
        // (0,1) in clique -> (0,2) in original = svec idx 3
        assert_eq!(selector.to_original[1], ij_to_svec(0, 2));
    }
}
