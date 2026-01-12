//! Chordal decomposition for sparse semidefinite programs.
//!
//! This module implements chordal decomposition to exploit sparsity in SDP constraints.
//! Instead of enforcing `X ∈ PSD(n)`, we enforce `X[Cₖ, Cₖ] ∈ PSD(|Cₖ|)` for each
//! maximal clique Cₖ in the aggregate sparsity pattern.
//!
//! # Algorithm Overview
//!
//! 1. Build aggregate sparsity graph from constraint matrices
//! 2. Make graph chordal (if not already) via minimum degree ordering
//! 3. Find maximal cliques using perfect elimination ordering
//! 4. Build clique tree with running intersection property
//! 5. Optionally merge small/overlapping cliques
//! 6. Transform problem: replace large PSD with smaller overlapping PSDs
//! 7. After solving, complete dual variables via PSD completion
//!
//! # References
//!
//! - Vandenberghe & Andersen: "Chordal Graphs and Semidefinite Optimization"
//! - Zheng et al: "Chordal decomposition in operator-splitting methods for sparse SDPs"

mod graph;
mod cliques;
mod decompose;
mod merge;
mod completion;

pub use graph::{SparsityGraph, ChordalGraph};
pub use cliques::{Clique, CliqueTree};
pub use decompose::{DecomposedPsd, PsdDecomposition};
pub use merge::{MergeStrategy, merge_cliques};
pub use completion::complete_dual;

use crate::problem::ProblemData;
use crate::ConeSpec;

/// Settings for chordal decomposition.
#[derive(Debug, Clone)]
pub struct ChordalSettings {
    /// Enable chordal decomposition (default: true for PSD cones)
    pub enabled: bool,
    /// Minimum PSD cone size to consider decomposition (default: 10)
    pub min_size: usize,
    /// Merge strategy (default: CliqueGraph)
    pub merge_strategy: MergeStrategy,
    /// Enable compact form assembly (default: true)
    pub compact: bool,
    /// Complete dual variables after solve (default: true)
    pub complete_dual: bool,
}

impl Default for ChordalSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            min_size: 10,
            merge_strategy: MergeStrategy::CliqueGraph,
            compact: true,
            complete_dual: true,
        }
    }
}

/// Result of analyzing a problem for chordal decomposition.
#[derive(Debug)]
pub struct ChordalAnalysis {
    /// Original PSD cone indices that were decomposed
    pub decomposed_cones: Vec<usize>,
    /// Decomposition data for each decomposed cone
    pub decompositions: Vec<PsdDecomposition>,
    /// Total number of cliques after decomposition
    pub total_cliques: usize,
    /// Whether decomposition is beneficial
    pub beneficial: bool,
}

/// Analyze a problem for chordal decomposition opportunities.
pub fn analyze_problem(prob: &ProblemData, settings: &ChordalSettings) -> ChordalAnalysis {
    if !settings.enabled {
        return ChordalAnalysis {
            decomposed_cones: vec![],
            decompositions: vec![],
            total_cliques: 0,
            beneficial: false,
        };
    }

    let mut decomposed_cones = Vec::new();
    let mut decompositions = Vec::new();
    let mut total_cliques = 0;

    for (idx, cone) in prob.cones.iter().enumerate() {
        if let ConeSpec::Psd { n } = cone {
            if *n >= settings.min_size {
                // Build sparsity graph for this PSD cone
                if let Some(decomp) = analyze_psd_cone(prob, idx, *n, settings) {
                    if decomp.is_beneficial() {
                        total_cliques += decomp.clique_tree.cliques.len();
                        decomposed_cones.push(idx);
                        decompositions.push(decomp);
                    }
                }
            }
        }
    }

    let beneficial = !decompositions.is_empty();

    ChordalAnalysis {
        decomposed_cones,
        decompositions,
        total_cliques,
        beneficial,
    }
}

/// Analyze a single PSD cone for decomposition.
fn analyze_psd_cone(
    prob: &ProblemData,
    cone_idx: usize,
    n: usize,
    settings: &ChordalSettings,
) -> Option<PsdDecomposition> {
    // Find the offset of this PSD cone in the constraint matrix
    let mut offset = 0;
    for (i, cone) in prob.cones.iter().enumerate() {
        if i == cone_idx {
            break;
        }
        offset += cone.dim();
    }

    let svec_dim = n * (n + 1) / 2;

    // Build sparsity graph from constraint matrix columns
    let sparsity = SparsityGraph::from_constraints(&prob.A, offset, svec_dim, n);

    // Make chordal if needed
    let chordal = ChordalGraph::from_sparsity(sparsity);

    // Find maximal cliques
    let clique_tree = CliqueTree::from_chordal(&chordal);

    // Apply merge strategy
    let merged = merge_cliques(&clique_tree, settings.merge_strategy);

    Some(PsdDecomposition::new(n, merged, offset))
}

/// Transform a problem using chordal decomposition.
pub fn decompose_problem(
    prob: &ProblemData,
    analysis: &ChordalAnalysis,
) -> (ProblemData, DecomposedPsd) {
    decompose::transform_problem(prob, analysis)
}

/// Recover original solution from decomposed solution.
pub fn recover_solution(
    decomposed: &DecomposedPsd,
    x: &[f64],
    s: &[f64],
    z: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    decompose::recover_solution(decomposed, x, s, z)
}
