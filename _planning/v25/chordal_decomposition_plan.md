# Chordal Decomposition for MINIX SDP

## Problem

MINIX fails on larger PSD blocks (10x10+) due to ill-conditioned KKT systems:
- control1 (10x10 + 5x5): 7% error vs CLARABEL's 1.84e-7
- truss5 (33x 10x10): 72x slower than CLARABEL, same accuracy

CLARABEL uses chordal decomposition to break large PSD cones into smaller overlapping ones.

## What is Chordal Decomposition?

**Core Theorem** (Agler/Grone): A sparse symmetric matrix S is PSD if and only if all its principal submatrices corresponding to maximal cliques are PSD.

This means: Instead of enforcing `X ∈ PSD(n)`, we can enforce `X[Cₖ, Cₖ] ∈ PSD(|Cₖ|)` for each maximal clique Cₖ.

**Benefits:**
- Smaller PSD cones = better conditioned KKT
- Fewer iterations (19-26 vs 73-100)
- O(Σ|Cₖ|³) vs O(n³) per iteration

## Algorithm Steps

### 1. Aggregate Sparsity Pattern
```
Input: Constraint matrices F₀, F₁, ..., Fₘ for PSD cone
Output: Sparsity graph G(V, E)

V = {1, ..., n}  (matrix indices)
E = {}
for each Fᵢ:
    for each nonzero (j, k) in Fᵢ:
        E = E ∪ {(j, k)}
```

### 2. Chordal Completion
If G is not chordal, add fill-in edges to make it chordal.
- Use minimum degree ordering or similar heuristic
- Result: chordal graph G' ⊇ G

### 3. Maximal Clique Enumeration
Find all maximal cliques {C₁, ..., Cₚ} in G'.
- For chordal graphs: at most n cliques
- Use perfect elimination ordering

### 4. Clique Tree Construction
Build tree T where:
- Nodes = maximal cliques
- Edges satisfy "running intersection property"
- Cᵢ ∩ Cⱼ ⊆ all cliques on path between Cᵢ and Cⱼ

### 5. Clique Merging (Optional)
Merge small/overlapping cliques to reduce overhead.
- CliqueGraphMerge: weight = |Cᵢ|³ + |Cⱼ|³ - |Cᵢ ∪ Cⱼ|³
- Merge pairs with highest (most negative) weight

### 6. Problem Transformation
Replace original problem:
```
Original: X ∈ PSD(n)
Decomposed: Xₖ ∈ PSD(|Cₖ|) for k = 1..p
            Xₖ[overlap] = Xₗ[overlap] for overlapping cliques
```

Entry selector matrices Tₖ map: `Xₖ = Tₖ X Tₖᵀ`

### 7. Dual Variable Completion
After solving, recover full dual Y from decomposed Yₖ.
- Non-trivial: must maintain PSD while filling structural zeros
- Use positive semidefinite matrix completion algorithm

## Data Structures Needed

```rust
/// Aggregate sparsity pattern
struct SparsityGraph {
    n: usize,
    edges: HashSet<(usize, usize)>,
}

/// Maximal clique
struct Clique {
    indices: Vec<usize>,  // sorted indices in original matrix
}

/// Clique tree for running intersection property
struct CliqueTree {
    cliques: Vec<Clique>,
    parent: Vec<Option<usize>>,  // parent in tree
    separator: Vec<Vec<usize>>,  // overlap with parent
}

/// Entry selector for one clique
struct EntrySelector {
    clique_idx: usize,
    n_clique: usize,
    /// Maps (i,j) in clique to index in original svec
    to_original: Vec<usize>,
    /// Maps index in original svec to (i,j) in clique (if present)
    from_original: HashMap<usize, usize>,
}

/// Decomposed PSD cone
struct DecomposedPsd {
    original_n: usize,
    cliques: Vec<Clique>,
    selectors: Vec<EntrySelector>,
    /// Overlap constraints between cliques
    overlaps: Vec<OverlapConstraint>,
}
```

## Implementation Phases

### Phase 1: Sparsity Analysis (~200 LOC)
- [ ] Parse constraint matrices to build sparsity graph
- [ ] Implement chordal check
- [ ] Implement chordal completion (minimum degree)

### Phase 2: Clique Enumeration (~300 LOC)
- [ ] Perfect elimination ordering
- [ ] Maximal clique enumeration
- [ ] Clique tree construction

### Phase 3: Problem Transformation (~400 LOC)
- [ ] Entry selector matrices
- [ ] Transform PSD constraints to decomposed form
- [ ] Handle overlap constraints (as equality constraints)

### Phase 4: Clique Merging (~200 LOC)
- [ ] Implement CliqueGraphMerge strategy
- [ ] Tune merge threshold

### Phase 5: Dual Completion (~200 LOC)
- [ ] PSD matrix completion algorithm
- [ ] Integrate with solution recovery

### Phase 6: Integration (~200 LOC)
- [ ] Auto-detect when decomposition is beneficial
- [ ] Settings for enable/disable, merge strategy
- [ ] Benchmarking and validation

**Total estimate: ~1500 LOC**

## Key References

1. [Vandenberghe & Andersen: Chordal Graphs and Semidefinite Optimization](https://www.seas.ucla.edu/~vandenbe/publications/chordalsdp.pdf)
2. [COSMO.jl Decomposition Docs](https://oxfordcontrol.github.io/COSMO.jl/stable/decomposition/)
3. [Zheng et al: Chordal decomposition in operator-splitting methods](https://link.springer.com/article/10.1007/s10107-019-01366-3)
4. [CLARABEL Chordal Guide](https://clarabel.org/stable/user_guide_chordal/)

## Alternative: Simpler First Pass

Before full chordal decomposition, we could try:
1. **Block-diagonal detection**: If constraint matrices are block-diagonal, split into independent PSD cones (no overlap constraints needed)
2. **Dense fallback**: Skip decomposition for small n (e.g., n ≤ 20)

This would handle cases like truss5 (33 independent 10x10 blocks) without full clique machinery.
