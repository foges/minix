# Constraint Conditioning Results

## Summary

Constraint row conditioning via geometric mean scaling was implemented and tested. **Result: Harmful - decreased pass rate from 79.4% to 76.5%.**

## Implementation

Added `solver-core/src/presolve/condition.rs` with:
- `analyze_conditioning()` - detects nearly-parallel rows (cosine sim > 0.999) and extreme coefficient ratios
- `apply_row_scaling()` - applies geometric mean scaling: `scale[i] = 1/sqrt(geom_mean(max_abs, min_abs))`

Integration in `solve.rs`:
- Runs before Ruiz equilibration
- Triggers when: `extreme_ratio_rows > 0 OR max_ratio > 1e5 OR parallel_pairs > 5`

## Benchmark Results

### Full Benchmark (50 iters max)
- **Baseline (v15)**: 108/136 Optimal (79.4%)
- **With conditioning**: 104/136 Optimal (76.5%)
- **Change**: -4 problems (-2.9%)

### Individual Problem Analysis

**QFFFFF80** (22 parallel pairs, max_ratio=1.090e5):
- Baseline: rel_p=9.120e-9, rel_d=7.682e2
- With conditioning: rel_p=3.935e-1, rel_d=7.233e2
- **Result**: Primal residual degraded 4e7x, dual slightly better

**QFORPLAN** (26 parallel pairs, max_ratio=2.873e1):
- With conditioning: rel_p=3.074e-14, rel_d=1.531e0
- Still fails (MaxIters)

**QGFRDXPN** (211 parallel pairs, max_ratio=1.095e3):
- With conditioning: rel_p=2.849e-15, rel_d=3.589e0
- Still fails (MaxIters)

**QSHIP04S** (2 parallel pairs, max_ratio=2.118):
- With conditioning: rel_p=4.240e-10, rel_d=5.398e-1
- Still fails (MaxIters)

## Why It Failed

### Root Cause: Interferes with Ruiz Equilibration

The geometric mean row scaling disrupts the problem structure in ways that make Ruiz equilibration less effective:

1. **Ruiz expects raw problem**: Ruiz iteratively balances row/column norms. Pre-scaling rows changes the starting point in ways that don't align with Ruiz's convergence.

2. **No benefit for parallel rows**: Nearly-parallel rows indicate rank deficiency in the constraint matrix. Scaling them doesn't fix the fundamental issue that the KKT matrix is near-singular.

3. **Breaking dual structure**: The dual explosion problems (QFFFFF80, QSHIP) have pathological A^T*z components. Scaling rows changes the dual scaling but doesn't address why certain dual variables become huge.

### What Actually Helps Parallel Rows

Scaling is the wrong tool. Better approaches:
1. **Elimination**: Remove truly redundant constraints (if row_i ≈ α·row_j, eliminate one)
2. **Combination**: Merge near-parallel constraints
3. **Robust KKT**: Use regularization or iterative refinement to handle near-singularity
4. **Dual recovery**: Accept primal solution, recover dual via least squares (already implemented)

## Correct Diagnosis

Problems have different root causes:

| Problem | Parallel Pairs | Issue | Real Fix |
|---------|----------------|-------|----------|
| QFFFFF80 | 22 | Dual explosion (A^Tz) | Dual recovery, better KKT |
| QFORPLAN | 26 | μ explosion (s·z) | Barrier parameter control |
| QGFRDXPN | 211 | Dual stuck | Possibly truly infeasible |
| QSHIP04S | 2 | Dual directions huge | Dual recovery (already works) |

## Conclusion

**Conditioning via row scaling: Failed experiment.**

- Decreases pass rate: 108 → 104 problems
- Disrupts Ruiz equilibration
- Doesn't address root causes (rank deficiency, dual explosion)

**Next steps:**
1. Revert to baseline (disable conditioning by default)
2. Keep detection code for analysis
3. Focus on approaches that showed promise:
   - Dual recovery (already implemented, shows improvement)
   - Better barrier parameter control for μ explosion
   - Robust KKT solvers with regularization
