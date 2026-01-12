//! Symbolic analysis for sparse LDL^T factorization.
//!
//! This module implements the CPU-side symbolic analysis phase:
//! - Fill-reducing ordering (AMD via SuiteSparse CAMD)
//! - Elimination tree construction
//! - Supernode detection
//! - Level scheduling for parallel execution

use super::error::{MetalError, MetalResult};
use super::handle::OrderingMethod;
use crate::linalg::sparse::SparseCsc;
use sprs_suitesparse_camd::try_camd;

/// Complete symbolic analysis result.
///
/// Contains all information needed for numeric factorization and solve.
#[derive(Debug, Clone)]
pub struct SymbolicAnalysis {
    /// Matrix dimension.
    pub n: usize,

    /// Number of nonzeros in the factor L.
    pub nnz_l: usize,

    /// Fill-reducing permutation: new_index = perm[old_index].
    pub perm: Vec<usize>,

    /// Inverse permutation: old_index = perm_inv[new_index].
    pub perm_inv: Vec<usize>,

    /// Elimination tree.
    pub etree: EliminationTree,

    /// Supernodes (groups of columns with same structure).
    pub supernodes: Vec<Supernode>,

    /// Level schedule for supernodes (parallel factorization).
    pub level_schedule: LevelSchedule,

    /// Column pointers for factor L in CSC format.
    pub l_col_ptr: Vec<usize>,

    /// Row indices for factor L in CSC format (allocated, filled during numeric).
    pub l_row_ind: Vec<usize>,

    /// Column counts (number of nonzeros in each column of L).
    pub col_counts: Vec<usize>,

    /// Level schedule for rows (lower triangular solve).
    /// (level_ptr, level_rows) - rows at level i are level_rows[level_ptr[i]..level_ptr[i+1]]
    pub solve_levels_lower: (Vec<usize>, Vec<usize>),

    /// Level schedule for rows (upper triangular solve / L^T solve).
    pub solve_levels_upper: (Vec<usize>, Vec<usize>),
}

impl SymbolicAnalysis {
    /// Perform symbolic analysis on a symmetric sparse matrix.
    ///
    /// # Arguments
    /// * `n` - Matrix dimension
    /// * `col_ptr` - CSC column pointers (length n+1)
    /// * `row_ind` - CSC row indices
    /// * `ordering` - Fill-reducing ordering method
    pub fn analyze(
        n: usize,
        col_ptr: &[usize],
        row_ind: &[usize],
        ordering: OrderingMethod,
    ) -> MetalResult<Self> {
        if col_ptr.len() != n + 1 {
            return Err(MetalError::InvalidMatrix(format!(
                "col_ptr length {} != n+1 = {}",
                col_ptr.len(),
                n + 1
            )));
        }

        // Step 1: Compute fill-reducing ordering
        let (perm, perm_inv) = match ordering {
            OrderingMethod::Natural => {
                let perm: Vec<usize> = (0..n).collect();
                let perm_inv = perm.clone();
                (perm, perm_inv)
            }
            OrderingMethod::Amd => {
                compute_amd_ordering(n, col_ptr, row_ind)?
            }
            OrderingMethod::NestedDissection => {
                // Fall back to AMD for now
                compute_amd_ordering(n, col_ptr, row_ind)?
            }
        };

        // Step 2: Build elimination tree on permuted matrix
        let etree = EliminationTree::build(n, col_ptr, row_ind, &perm, &perm_inv)?;

        // Step 3: Compute column counts for factor L
        let col_counts = compute_column_counts(n, col_ptr, row_ind, &perm, &perm_inv, &etree)?;

        // Step 4: Detect supernodes
        let supernodes = detect_supernodes(n, &etree, &col_counts)?;

        // Step 5: Build level schedule for supernodes
        let level_schedule = LevelSchedule::build(&etree, &supernodes)?;

        // Step 6: Allocate factor storage structure
        let nnz_l: usize = col_counts.iter().sum();
        let mut l_col_ptr = vec![0usize; n + 1];
        for j in 0..n {
            l_col_ptr[j + 1] = l_col_ptr[j] + col_counts[j];
        }
        let l_row_ind = vec![0usize; nnz_l];

        // Step 7: Compute row-level schedules for triangular solves
        // Note: For now we use the original matrix structure; in practice we'd use L's structure
        let solve_levels_lower = compute_row_level_schedule(n, col_ptr, row_ind, true);
        let solve_levels_upper = compute_row_level_schedule(n, col_ptr, row_ind, false);

        Ok(Self {
            n,
            nnz_l,
            perm,
            perm_inv,
            etree,
            supernodes,
            level_schedule,
            l_col_ptr,
            l_row_ind,
            col_counts,
            solve_levels_lower,
            solve_levels_upper,
        })
    }

    /// Perform symbolic analysis from a SparseCsc matrix.
    pub fn analyze_csc(mat: &SparseCsc, ordering: OrderingMethod) -> MetalResult<Self> {
        let (n, m) = mat.shape();
        if n != m {
            return Err(MetalError::InvalidMatrix(format!(
                "Matrix must be square, got {}x{}",
                n, m
            )));
        }

        // Extract CSC components
        let col_ptr: Vec<usize> = mat.indptr().raw_storage().iter().copied().collect();
        let row_ind: Vec<usize> = mat.indices().iter().map(|&x| x).collect();

        Self::analyze(n, &col_ptr, &row_ind, ordering)
    }
}

/// Elimination tree for sparse Cholesky/LDL factorization.
#[derive(Debug, Clone)]
pub struct EliminationTree {
    /// Parent of each column in the etree. parent[j] = k means column k is the parent of j.
    /// parent[j] = n (or usize::MAX) means j is a root.
    pub parent: Vec<usize>,

    /// Children of each node (for bottom-up traversal).
    pub children: Vec<Vec<usize>>,

    /// Post-order of the etree (for cache-friendly traversal).
    pub postorder: Vec<usize>,

    /// Depth of each node in the etree.
    pub depth: Vec<usize>,
}

impl EliminationTree {
    /// Build the elimination tree for a symmetric matrix.
    ///
    /// Uses the standard algorithm: for each column j, parent[j] is the minimum
    /// row index > j in column j of the factor L.
    pub fn build(
        n: usize,
        col_ptr: &[usize],
        row_ind: &[usize],
        perm: &[usize],
        perm_inv: &[usize],
    ) -> MetalResult<Self> {
        let mut parent = vec![n; n]; // n means "no parent" (root)
        let mut ancestor = vec![0usize; n];

        // Standard etree algorithm using path compression
        for j in 0..n {
            ancestor[j] = j;

            // Get the original column corresponding to permuted column j
            let orig_col = perm_inv[j];

            // Iterate over entries in the original column
            for p in col_ptr[orig_col]..col_ptr[orig_col + 1] {
                let orig_row = row_ind[p];
                let i = perm[orig_row]; // Permuted row index

                // Only consider entries in the lower triangle (i > j in permuted ordering)
                if i < j {
                    // Find root of i's subtree with path compression
                    let mut r = i;
                    while ancestor[r] != r && ancestor[r] != j {
                        r = ancestor[r];
                    }

                    if ancestor[r] != j {
                        parent[r] = j;
                        ancestor[r] = j;
                    }

                    // Path compression
                    let mut s = i;
                    while s != r {
                        let next = ancestor[s];
                        ancestor[s] = j;
                        s = next;
                    }
                }
            }
        }

        // Build children lists
        let mut children = vec![Vec::new(); n];
        for j in 0..n {
            if parent[j] < n {
                children[parent[j]].push(j);
            }
        }

        // Compute postorder
        let postorder = compute_postorder(n, &parent, &children);

        // Compute depth
        let mut depth = vec![0usize; n];
        for &j in &postorder {
            if parent[j] < n {
                depth[j] = depth[parent[j]] + 1;
            }
        }

        Ok(Self {
            parent,
            children,
            postorder,
            depth,
        })
    }

    /// Get root nodes (columns with no parent).
    pub fn roots(&self) -> Vec<usize> {
        let n = self.parent.len();
        self.parent
            .iter()
            .enumerate()
            .filter(|(_, &p)| p >= n)
            .map(|(j, _)| j)
            .collect()
    }
}

/// A supernode: a group of consecutive columns with the same row structure.
#[derive(Debug, Clone)]
pub struct Supernode {
    /// Index of this supernode.
    pub index: usize,

    /// First column in this supernode.
    pub first_col: usize,

    /// Number of columns in this supernode.
    pub num_cols: usize,

    /// Number of rows in the supernode panel (including diagonal).
    pub num_rows: usize,

    /// Row indices for this supernode (in permuted ordering).
    pub row_indices: Vec<usize>,

    /// Parent supernode index (or usize::MAX if root).
    pub parent: usize,

    /// Child supernode indices.
    pub children: Vec<usize>,
}

impl Supernode {
    /// Get the last column in this supernode (inclusive).
    pub fn last_col(&self) -> usize {
        self.first_col + self.num_cols - 1
    }

    /// Get the columns in this supernode.
    pub fn columns(&self) -> std::ops::Range<usize> {
        self.first_col..self.first_col + self.num_cols
    }
}

/// Level schedule for parallel execution of supernodes.
#[derive(Debug, Clone)]
pub struct LevelSchedule {
    /// Number of levels.
    pub num_levels: usize,

    /// Start index in `level_supernodes` for each level.
    /// Level i contains supernodes level_supernodes[level_ptr[i]..level_ptr[i+1]].
    pub level_ptr: Vec<usize>,

    /// Supernode indices ordered by level.
    pub level_supernodes: Vec<usize>,

    /// Level of each supernode.
    pub supernode_level: Vec<usize>,
}

impl LevelSchedule {
    /// Build level schedule from elimination tree and supernodes.
    ///
    /// Supernodes at the same level have no dependencies between them
    /// and can be processed in parallel.
    pub fn build(etree: &EliminationTree, supernodes: &[Supernode]) -> MetalResult<Self> {
        let num_supernodes = supernodes.len();
        if num_supernodes == 0 {
            return Ok(Self {
                num_levels: 0,
                level_ptr: vec![0],
                level_supernodes: vec![],
                supernode_level: vec![],
            });
        }

        // Compute level for each supernode (leaves are level 0)
        let mut supernode_level = vec![0usize; num_supernodes];

        // Process in reverse topological order (children before parents)
        for s in 0..num_supernodes {
            let sn = &supernodes[s];
            for &child in &sn.children {
                supernode_level[s] = supernode_level[s].max(supernode_level[child] + 1);
            }
        }

        // Find max level
        let num_levels = supernode_level.iter().max().map_or(0, |&m| m + 1);

        // Group supernodes by level
        let mut level_counts = vec![0usize; num_levels];
        for &level in &supernode_level {
            level_counts[level] += 1;
        }

        let mut level_ptr = vec![0usize; num_levels + 1];
        for i in 0..num_levels {
            level_ptr[i + 1] = level_ptr[i] + level_counts[i];
        }

        let mut level_supernodes = vec![0usize; num_supernodes];
        let mut level_next = level_ptr.clone();
        for s in 0..num_supernodes {
            let level = supernode_level[s];
            level_supernodes[level_next[level]] = s;
            level_next[level] += 1;
        }

        Ok(Self {
            num_levels,
            level_ptr,
            level_supernodes,
            supernode_level,
        })
    }

    /// Get supernodes at a given level.
    pub fn supernodes_at_level(&self, level: usize) -> &[usize] {
        if level >= self.num_levels {
            return &[];
        }
        &self.level_supernodes[self.level_ptr[level]..self.level_ptr[level + 1]]
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Compute AMD (Approximate Minimum Degree) ordering using SuiteSparse CAMD.
fn compute_amd_ordering(
    n: usize,
    col_ptr: &[usize],
    row_ind: &[usize],
) -> MetalResult<(Vec<usize>, Vec<usize>)> {
    // Build a sprs CSC matrix for CAMD
    // CAMD expects the structure view of a symmetric matrix
    let nnz = col_ptr[n];
    let values = vec![1.0f64; nnz]; // Dummy values, CAMD only uses structure

    let col_ptr_i: Vec<usize> = col_ptr.to_vec();
    let row_ind_i: Vec<usize> = row_ind.to_vec();

    // Create CSC matrix using sprs
    let mat = sprs::CsMat::new(
        (n, n),
        col_ptr_i,
        row_ind_i,
        values,
    );

    // Run CAMD ordering
    let perm = try_camd(mat.structure_view())
        .map_err(|e| MetalError::SymbolicAnalysis(format!("CAMD ordering failed: {}", e)))?;

    Ok((perm.vec(), perm.inv_vec()))
}

/// Compute level schedule for rows of a triangular matrix.
///
/// This is used for level-scheduled SpTRSV on GPU.
/// Returns (level_ptr, level_rows) where level i contains rows level_rows[level_ptr[i]..level_ptr[i+1]].
pub fn compute_row_level_schedule(
    n: usize,
    col_ptr: &[usize],
    row_ind: &[usize],
    is_lower: bool,
) -> (Vec<usize>, Vec<usize>) {
    // Compute level for each row based on dependencies
    let mut row_level = vec![0usize; n];

    if is_lower {
        // For lower triangular: row i depends on rows j < i where L[i,j] != 0
        for i in 0..n {
            let mut max_dep_level = 0usize;
            // Find columns j < i in row i
            for j in 0..n {
                for p in col_ptr[j]..col_ptr[j + 1] {
                    if row_ind[p] == i && j < i {
                        max_dep_level = max_dep_level.max(row_level[j] + 1);
                    }
                }
            }
            row_level[i] = max_dep_level;
        }
    } else {
        // For upper triangular: row i depends on rows j > i where U[i,j] != 0
        for i in (0..n).rev() {
            let mut max_dep_level = 0usize;
            // Find columns j > i in row i
            for j in (i + 1)..n {
                for p in col_ptr[j]..col_ptr[j + 1] {
                    if row_ind[p] == i {
                        max_dep_level = max_dep_level.max(row_level[j] + 1);
                    }
                }
            }
            row_level[i] = max_dep_level;
        }
    }

    // Find number of levels
    let num_levels = row_level.iter().max().map_or(0, |&m| m + 1);

    // Group rows by level
    let mut level_counts = vec![0usize; num_levels];
    for &level in &row_level {
        level_counts[level] += 1;
    }

    let mut level_ptr = vec![0usize; num_levels + 1];
    for i in 0..num_levels {
        level_ptr[i + 1] = level_ptr[i] + level_counts[i];
    }

    let mut level_rows = vec![0usize; n];
    let mut level_next = level_ptr.clone();

    if is_lower {
        // Process in order for lower triangular
        for i in 0..n {
            let level = row_level[i];
            level_rows[level_next[level]] = i;
            level_next[level] += 1;
        }
    } else {
        // Process in reverse order for upper triangular
        for i in (0..n).rev() {
            let level = row_level[i];
            level_rows[level_next[level]] = i;
            level_next[level] += 1;
        }
    }

    (level_ptr, level_rows)
}

/// Simplified degree-based ordering fallback (used when CAMD fails or for testing)
#[allow(dead_code)]
fn compute_degree_ordering(
    n: usize,
    col_ptr: &[usize],
    row_ind: &[usize],
) -> MetalResult<(Vec<usize>, Vec<usize>)> {
    // Compute degrees
    let mut degree: Vec<usize> = (0..n)
        .map(|j| col_ptr[j + 1] - col_ptr[j])
        .collect();

    // For symmetric matrix, count both upper and lower
    for j in 0..n {
        for p in col_ptr[j]..col_ptr[j + 1] {
            let i = row_ind[p];
            if i != j {
                degree[i] += 1;
            }
        }
    }

    // Sort by degree (minimum degree heuristic)
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&j| degree[j]);

    // Build inverse permutation
    let mut perm_inv = vec![0usize; n];
    for (new_idx, &old_idx) in order.iter().enumerate() {
        perm_inv[old_idx] = new_idx;
    }

    // perm[new_idx] = old_idx
    let perm = order;

    // For proper AMD, we'd iteratively:
    // 1. Pick minimum degree node
    // 2. Eliminate it (update graph)
    // 3. Update degrees of neighbors
    // 4. Repeat
    //
    // This simple version just sorts by initial degree, which is a rough approximation.

    Ok((perm, perm_inv))
}

/// Compute column counts for factor L.
fn compute_column_counts(
    n: usize,
    col_ptr: &[usize],
    row_ind: &[usize],
    perm: &[usize],
    perm_inv: &[usize],
    etree: &EliminationTree,
) -> MetalResult<Vec<usize>> {
    // Use skeleton counting algorithm
    // Each column j of L has structure: {j} ∪ (struct(L[:,parent[j]]) \ {parent[j]})
    // plus any new fill from row entries

    let mut col_counts = vec![1usize; n]; // Each column has at least the diagonal

    // First pass: count entries from original matrix
    for j in 0..n {
        let orig_col = perm_inv[j];
        for p in col_ptr[orig_col]..col_ptr[orig_col + 1] {
            let orig_row = row_ind[p];
            let i = perm[orig_row];
            if i > j {
                // Entry in lower triangle contributes to column j
                col_counts[j] += 1;
            }
        }
    }

    // Second pass: propagate fill through etree
    // This is a simplified version; proper symbolic factorization is more complex
    for &j in &etree.postorder {
        if etree.parent[j] < n {
            let parent = etree.parent[j];
            // Parent inherits structure from child (minus the child's column)
            // This is approximate; exact counting requires more bookkeeping
            if col_counts[j] > 1 {
                col_counts[parent] = col_counts[parent].max(col_counts[j] - 1 + 1);
            }
        }
    }

    Ok(col_counts)
}

/// Detect supernodes (groups of columns with identical structure).
fn detect_supernodes(
    n: usize,
    etree: &EliminationTree,
    col_counts: &[usize],
) -> MetalResult<Vec<Supernode>> {
    if n == 0 {
        return Ok(vec![]);
    }

    // Fundamental supernode detection:
    // Columns j and j+1 are in the same supernode if:
    // 1. parent[j] == j+1 in the etree
    // 2. col_count[j+1] == col_count[j] - 1

    let mut supernode_start = vec![true; n]; // Is this column the start of a supernode?

    for j in 0..n - 1 {
        // Check if j and j+1 should be in the same supernode
        if etree.parent[j] == j + 1 && col_counts[j + 1] == col_counts[j].saturating_sub(1) {
            supernode_start[j + 1] = false;
        }
    }

    // Build supernodes
    let mut supernodes = Vec::new();
    let mut current_start = 0;

    for j in 1..=n {
        if j == n || supernode_start[j] {
            // End current supernode
            let num_cols = j - current_start;
            let num_rows = col_counts[current_start];

            supernodes.push(Supernode {
                index: supernodes.len(),
                first_col: current_start,
                num_cols,
                num_rows,
                row_indices: Vec::new(), // Filled during numeric phase
                parent: usize::MAX,
                children: Vec::new(),
            });

            current_start = j;
        }
    }

    // Build supernode tree from column etree
    let num_supernodes = supernodes.len();

    // Map columns to supernodes
    let mut col_to_supernode = vec![0usize; n];
    for (s, sn) in supernodes.iter().enumerate() {
        for col in sn.columns() {
            col_to_supernode[col] = s;
        }
    }

    // Find parent supernode
    for s in 0..num_supernodes {
        let last_col = supernodes[s].last_col();
        if etree.parent[last_col] < n {
            let parent_col = etree.parent[last_col];
            let parent_sn = col_to_supernode[parent_col];
            supernodes[s].parent = parent_sn;
        }
    }

    // Build children lists
    for s in 0..num_supernodes {
        let parent = supernodes[s].parent;
        if parent < num_supernodes {
            supernodes[parent].children.push(s);
        }
    }

    Ok(supernodes)
}

/// Compute postorder traversal of the etree.
fn compute_postorder(n: usize, parent: &[usize], children: &[Vec<usize>]) -> Vec<usize> {
    let mut postorder = Vec::with_capacity(n);
    let mut visited = vec![false; n];

    // Find roots
    let roots: Vec<usize> = parent
        .iter()
        .enumerate()
        .filter(|(_, &p)| p >= n)
        .map(|(j, _)| j)
        .collect();

    // DFS from each root
    for root in roots {
        postorder_dfs(root, children, &mut visited, &mut postorder);
    }

    postorder
}

fn postorder_dfs(
    node: usize,
    children: &[Vec<usize>],
    visited: &mut [bool],
    postorder: &mut Vec<usize>,
) {
    if visited[node] {
        return;
    }
    visited[node] = true;

    for &child in &children[node] {
        postorder_dfs(child, children, visited, postorder);
    }

    postorder.push(node);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_etree_simple() {
        // Simple tridiagonal matrix: identity permutation
        // Pattern:
        //   x x . .
        //   x x x .
        //   . x x x
        //   . . x x
        let n = 4;
        let col_ptr = vec![0, 2, 5, 8, 10];
        let row_ind = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];

        let perm: Vec<usize> = (0..n).collect();
        let perm_inv = perm.clone();

        let etree = EliminationTree::build(n, &col_ptr, &row_ind, &perm, &perm_inv).unwrap();

        // For tridiagonal: parent[j] = j+1
        assert_eq!(etree.parent[0], 1);
        assert_eq!(etree.parent[1], 2);
        assert_eq!(etree.parent[2], 3);
        assert!(etree.parent[3] >= n); // Root
    }

    #[test]
    fn test_supernode_detection() {
        let n = 4;
        let col_counts = vec![4, 3, 2, 1]; // Typical for dense lower triangle

        let mut parent = vec![n; n];
        parent[0] = 1;
        parent[1] = 2;
        parent[2] = 3;

        let children = vec![vec![], vec![0], vec![1], vec![2]];
        let postorder = vec![0, 1, 2, 3];
        let depth = vec![3, 2, 1, 0];

        let etree = EliminationTree {
            parent,
            children,
            postorder,
            depth,
        };

        let supernodes = detect_supernodes(n, &etree, &col_counts).unwrap();

        // With perfect chain structure and decreasing col_counts,
        // all columns should be in one supernode
        assert_eq!(supernodes.len(), 1);
        assert_eq!(supernodes[0].num_cols, 4);
    }
}
