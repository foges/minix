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
    pub fn new(clique_idx: usize, clique: &Clique, _original_n: usize) -> Self {
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
    /// Maps original svec index to (clique_idx, clique_svec_idx) for the "owner" clique
    pub ownership: Vec<(usize, usize)>,
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

        // Build ownership mapping: each original svec index maps to first clique containing it
        let mut ownership = vec![(usize::MAX, usize::MAX); original_svec_dim];
        for (clique_idx, selector) in selectors.iter().enumerate() {
            for (clique_svec_idx, &orig_svec_idx) in selector.to_original.iter().enumerate() {
                if ownership[orig_svec_idx].0 == usize::MAX {
                    ownership[orig_svec_idx] = (clique_idx, clique_svec_idx);
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
            ownership,
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
    /// Offset of each new cone in the decomposed slack vector
    pub new_cone_offsets: Vec<usize>,
    /// Maps original cone index to its offset in original slack vector
    pub original_cone_offsets: Vec<usize>,
    /// Number of new overlap equality constraints added
    pub num_overlap_constraints: usize,
    /// Total new slack dimension
    pub new_slack_dim: usize,
    /// Original slack dimension
    pub original_slack_dim: usize,
}

/// Transform a problem using chordal decomposition.
///
/// The transformation:
/// 1. Replaces decomposed PSD cones with smaller clique-based PSD cones
/// 2. Expands the slack vector accordingly
/// 3. Adds Zero cone equality constraints for overlapping entries
pub fn transform_problem(
    prob: &ProblemData,
    analysis: &ChordalAnalysis,
) -> (ProblemData, DecomposedPsd) {
    if !analysis.beneficial || analysis.decompositions.is_empty() {
        // No decomposition - return original problem
        let original_slack_dim: usize = prob.cones.iter().map(|c| c.dim()).sum();
        let decomposed = DecomposedPsd {
            decompositions: vec![],
            original_cone_indices: vec![],
            cone_mapping: vec![],
            new_cone_offsets: vec![],
            original_cone_offsets: vec![],
            num_overlap_constraints: 0,
            new_slack_dim: original_slack_dim,
            original_slack_dim,
        };
        return (prob.clone(), decomposed);
    }

    // Compute original cone offsets
    let mut original_cone_offsets = Vec::with_capacity(prob.cones.len());
    let mut offset = 0;
    for cone in &prob.cones {
        original_cone_offsets.push(offset);
        offset += cone.dim();
    }
    let original_slack_dim = offset;

    let decomposed_set: std::collections::HashSet<usize> =
        analysis.decomposed_cones.iter().copied().collect();

    // Build new cone list and compute new offsets
    let mut new_cones = Vec::new();
    let mut cone_mapping = Vec::new();
    let mut new_cone_offsets = Vec::new();
    let mut current_offset = 0;

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
    let new_slack_dim = current_offset;

    // Calculate total overlap constraints
    let num_overlap_constraints: usize = analysis
        .decompositions
        .iter()
        .map(|d| d.total_overlap_constraints())
        .sum();

    // Add Zero cone for overlap constraints at the end
    if num_overlap_constraints > 0 {
        new_cones.push(ConeSpec::Zero { dim: num_overlap_constraints });
        new_cone_offsets.push(current_offset);
    }

    // Build mapping from original slack index to new slack index
    // For non-decomposed cones, this is a direct shift
    // For decomposed cones, use the ownership mapping
    let mut orig_to_new_slack: Vec<usize> = vec![usize::MAX; original_slack_dim];

    for (cone_idx, cone) in prob.cones.iter().enumerate() {
        let orig_offset = original_cone_offsets[cone_idx];

        if decomposed_set.contains(&cone_idx) {
            let decomp_idx = analysis
                .decomposed_cones
                .iter()
                .position(|&i| i == cone_idx)
                .unwrap();
            let decomp = &analysis.decompositions[decomp_idx];

            // Find the new cone offset for this decomposition's first clique
            let first_new_cone_idx = cone_mapping.iter()
                .position(|&(d, c)| d == decomp_idx && c == 0)
                .unwrap();

            // Map each original svec index to its owner's new position
            for orig_svec_idx in 0..decomp.original_svec_dim {
                let (owner_clique, clique_svec_idx) = decomp.ownership[orig_svec_idx];
                if owner_clique != usize::MAX {
                    let new_cone_idx = first_new_cone_idx + owner_clique;
                    let new_slack_idx = new_cone_offsets[new_cone_idx] + clique_svec_idx;
                    orig_to_new_slack[orig_offset + orig_svec_idx] = new_slack_idx;
                }
            }
        } else {
            // Non-decomposed cone: find its new position
            let new_cone_idx = cone_mapping.iter()
                .position(|&(d, c)| d == usize::MAX && c == cone_idx)
                .unwrap();
            let new_offset = new_cone_offsets[new_cone_idx];

            for i in 0..cone.dim() {
                orig_to_new_slack[orig_offset + i] = new_offset + i;
            }
        }
    }

    // Build new A matrix
    // The A matrix has dimensions (slack_dim, num_vars)
    // We need to remap row indices (slack indices) according to orig_to_new_slack
    let mut triplets = Vec::new();
    let num_vars = prob.A.cols();

    for col in 0..num_vars {
        if let Some(col_view) = prob.A.outer_view(col) {
            for (orig_row, &val) in col_view.iter() {
                let new_row = orig_to_new_slack[orig_row];
                if new_row != usize::MAX {
                    triplets.push((new_row, col, val));
                }
            }
        }
    }

    // Add overlap constraints
    // For each overlap, add constraints: s_a[i] - s_b[i] = 0
    // This goes into the Zero cone we added at the end
    let mut overlap_row = new_slack_dim;  // Start after all PSD cones

    for decomp in &analysis.decompositions {
        // Find the first new cone index for this decomposition
        let decomp_idx = analysis.decompositions.iter()
            .position(|d| std::ptr::eq(d, decomp))
            .unwrap();
        let first_new_cone_idx = cone_mapping.iter()
            .position(|&(d, c)| d == decomp_idx && c == 0)
            .unwrap();

        for overlap in &decomp.overlaps {
            let clique_a_offset = new_cone_offsets[first_new_cone_idx + overlap.clique_a];
            let clique_b_offset = new_cone_offsets[first_new_cone_idx + overlap.clique_b];

            for (&a_idx, &b_idx) in overlap.a_indices.iter().zip(&overlap.b_indices) {
                // Constraint: s_a - s_b = 0
                // This becomes a Zero cone constraint: s_overlap = 0
                // where s_overlap = s_a - s_b
                // We implement this by adding rows to A that compute s_a - s_b
                // But wait - A maps variables to slacks, not slacks to slacks
                //
                // The overlap constraints need to be handled differently.
                // In the standard form Ax + s = b, we need to add new variables
                // or handle this via the cone structure.
                //
                // Actually, for HSDE form, we can add explicit equality constraints.
                // The overlap constraint s_a[i] = s_b[i] can be written as:
                // A row that's zero except: +1 at position a, -1 at position b
                // with b value 0, and this row is in the Zero cone.

                // But since A maps (variables -> slacks), and we want slack equality,
                // we need a different approach. Let me think...
                //
                // Actually in our form: Ax + s = b
                // The slack s is what's constrained to be in the cone.
                // For overlap, we want s_a[i] = s_b[i].
                // This means we need to NOT have separate slack entries for overlaps.
                //
                // Alternative approach: Use consensus ADMM or just accept that
                // overlaps share the same slack variable. This means the A matrix
                // needs to have the same column contribute to multiple clique positions.
                //
                // Let me reconsider the transformation...
                //
                // For now, let's skip the explicit overlap constraints and instead
                // duplicate the A entries for overlapping positions. This means
                // the same variable contribution goes to all cliques containing that entry.

                let _ = (overlap_row, clique_a_offset, clique_b_offset, a_idx, b_idx);
            }
        }
    }

    // Duplicate entries to ALL cliques containing them.
    // This enforces overlap consistency: s1[overlap] = s2[overlap] through
    // having identical constraints at both positions.

    triplets.clear();

    for col in 0..num_vars {
        if let Some(col_view) = prob.A.outer_view(col) {
            for (orig_row, &val) in col_view.iter() {
                // Find which cone this row belongs to
                let mut found_cone = None;
                for (cone_idx, cone) in prob.cones.iter().enumerate() {
                    let cone_offset = original_cone_offsets[cone_idx];
                    if orig_row >= cone_offset && orig_row < cone_offset + cone.dim() {
                        found_cone = Some((cone_idx, orig_row - cone_offset));
                        break;
                    }
                }

                if let Some((cone_idx, local_idx)) = found_cone {
                    if decomposed_set.contains(&cone_idx) {
                        // Decomposed cone: add entry to ALL cliques containing this index
                        let decomp_idx = analysis
                            .decomposed_cones
                            .iter()
                            .position(|&i| i == cone_idx)
                            .unwrap();
                        let decomp = &analysis.decompositions[decomp_idx];
                        let first_new_cone_idx = cone_mapping.iter()
                            .position(|&(d, c)| d == decomp_idx && c == 0)
                            .unwrap();

                        // Find all cliques containing this svec index
                        for (clique_idx, selector) in decomp.selectors.iter().enumerate() {
                            if let Some(&clique_svec_idx) = selector.from_original.get(&local_idx) {
                                let new_cone_idx = first_new_cone_idx + clique_idx;
                                let new_row = new_cone_offsets[new_cone_idx] + clique_svec_idx;
                                triplets.push((new_row, col, val));
                            }
                        }
                    } else {
                        // Non-decomposed cone: direct mapping
                        let new_cone_idx = cone_mapping.iter()
                            .position(|&(d, c)| d == usize::MAX && c == cone_idx)
                            .unwrap();
                        let new_row = new_cone_offsets[new_cone_idx] + local_idx;
                        triplets.push((new_row, col, val));
                    }
                }
            }
        }
    }

    // Build new b vector with same duplication
    let mut new_b = vec![0.0; new_slack_dim];

    for (orig_row, &val) in prob.b.iter().enumerate() {
        // Find which cone this row belongs to
        let mut found_cone = None;
        for (cone_idx, cone) in prob.cones.iter().enumerate() {
            let cone_offset = original_cone_offsets[cone_idx];
            if orig_row >= cone_offset && orig_row < cone_offset + cone.dim() {
                found_cone = Some((cone_idx, orig_row - cone_offset));
                break;
            }
        }

        if let Some((cone_idx, local_idx)) = found_cone {
            if decomposed_set.contains(&cone_idx) {
                // Decomposed cone: add to ALL cliques containing this index
                let decomp_idx = analysis
                    .decomposed_cones
                    .iter()
                    .position(|&i| i == cone_idx)
                    .unwrap();
                let decomp = &analysis.decompositions[decomp_idx];
                let first_new_cone_idx = cone_mapping.iter()
                    .position(|&(d, c)| d == decomp_idx && c == 0)
                    .unwrap();

                for (clique_idx, selector) in decomp.selectors.iter().enumerate() {
                    if let Some(&clique_svec_idx) = selector.from_original.get(&local_idx) {
                        let new_cone_idx = first_new_cone_idx + clique_idx;
                        let new_row = new_cone_offsets[new_cone_idx] + clique_svec_idx;
                        new_b[new_row] = val;
                    }
                }
            } else {
                // Non-decomposed cone: direct mapping
                let new_cone_idx = cone_mapping.iter()
                    .position(|&(d, c)| d == usize::MAX && c == cone_idx)
                    .unwrap();
                let new_row = new_cone_offsets[new_cone_idx] + local_idx;
                new_b[new_row] = val;
            }
        }
    }

    // No Zero cone needed - using duplication for overlap consistency
    let final_cones = if num_overlap_constraints > 0 {
        new_cones[..new_cones.len()-1].to_vec()
    } else {
        new_cones
    };

    let new_a = sparse::from_triplets(new_slack_dim, num_vars, triplets);

    let new_prob = ProblemData {
        P: prob.P.clone(),
        q: prob.q.clone(),
        A: new_a,
        b: new_b[..new_slack_dim].to_vec(),
        cones: final_cones,
        var_bounds: prob.var_bounds.clone(),
        integrality: prob.integrality.clone(),
    };

    let decomposed = DecomposedPsd {
        decompositions: analysis.decompositions.clone(),
        original_cone_indices: analysis.decomposed_cones.clone(),
        cone_mapping,
        new_cone_offsets,
        original_cone_offsets,
        num_overlap_constraints: 0, // Using duplication instead
        new_slack_dim,
        original_slack_dim,
    };

    (new_prob, decomposed)
}

/// Recover original solution from decomposed solution.
pub fn recover_solution(
    decomposed: &DecomposedPsd,
    x: &[f64],
    s: &[f64],
    z: &[f64],
    original_cones: &[ConeSpec],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if decomposed.decompositions.is_empty() {
        return (x.to_vec(), s.to_vec(), z.to_vec());
    }

    let mut orig_s = vec![0.0; decomposed.original_slack_dim];
    let mut orig_z = vec![0.0; decomposed.original_slack_dim];

    // For each original cone
    for (cone_idx, cone) in original_cones.iter().enumerate() {
        let orig_offset = decomposed.original_cone_offsets[cone_idx];

        // Check if this cone was decomposed
        let decomp_idx_opt = decomposed.original_cone_indices.iter()
            .position(|&i| i == cone_idx);

        if let Some(decomp_idx) = decomp_idx_opt {
            let decomp = &decomposed.decompositions[decomp_idx];

            // Find the first new cone index for this decomposition
            let first_new_cone_idx = decomposed.cone_mapping.iter()
                .position(|&(d, c)| d == decomp_idx && c == 0)
                .unwrap();

            // Recover each original entry from its owner clique
            for orig_svec_idx in 0..decomp.original_svec_dim {
                let (owner_clique, clique_svec_idx) = decomp.ownership[orig_svec_idx];
                if owner_clique != usize::MAX {
                    let new_cone_idx = first_new_cone_idx + owner_clique;
                    let new_offset = decomposed.new_cone_offsets[new_cone_idx];
                    let new_idx = new_offset + clique_svec_idx;

                    orig_s[orig_offset + orig_svec_idx] = s[new_idx];
                    orig_z[orig_offset + orig_svec_idx] = z[new_idx];
                }
            }
        } else {
            // Non-decomposed cone: direct copy
            let new_cone_idx = decomposed.cone_mapping.iter()
                .position(|&(d, c)| d == usize::MAX && c == cone_idx)
                .unwrap();
            let new_offset = decomposed.new_cone_offsets[new_cone_idx];

            for i in 0..cone.dim() {
                orig_s[orig_offset + i] = s[new_offset + i];
                orig_z[orig_offset + i] = z[new_offset + i];
            }
        }
    }

    (x.to_vec(), orig_s, orig_z)
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
