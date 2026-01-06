# BOYD2 Convergence Analysis and Improvement Plan

## Current State

**Problem:** BOYD2 requires ~100 iterations to converge, then relies on post-hoc polish to achieve "Optimal" status. This is a hack, not true algorithmic convergence.

**Goal:** True convergence in under 30 iterations, like production solvers (MOSEK, Gurobi, Clarabel).

## Problem Characteristics

- **Variables:** 93,263
- **Constraints:** 279,794 (3x BOYD1)
- **Structure:** Heavy bound constraints (likely variable bounds expanded as NonNeg rows)
- **Known issue:** μ stalls/increases during iterations 10-30

## Observed Symptoms

From diagnostic traces:

1. **μ stalls:** At iteration 17, μ = 1.43. At iteration 29, μ = 0.53. Only 2.7x reduction in 12 iterations (should be orders of magnitude).

2. **μ increases:** At iteration 18, μ went from 1.43 to 1.86 (30% increase!). This violates the fundamental IPM property that μ should decrease monotonically.

3. **High σ values:** σ = 0.98 at iteration 18, meaning almost no centering reduction. This happens when mu_aff ≈ mu, i.e., the affine step doesn't reduce μ.

4. **Small affine step sizes:** α_aff often limited to 0.07-0.3 during iterations 15-30, indicating cone constraints are blocking progress.

5. **Gap stalls:** gap_rel oscillates between 1.3-2.0 for 20+ iterations instead of monotonically decreasing.

## Root Cause Analysis

### Why is α_aff small?

The affine step α_aff is computed as the largest step that keeps (s + α·Δs, z + α·Δz) in the cone interior. For NonNeg cones, this means:

```
s_i + α·Δs_i > 0  for all i
z_i + α·Δz_i > 0  for all i
```

If Δs_i or Δz_i is large and negative for some component while s_i or z_i is small, α_aff gets limited.

**Likely cause:** Some constraint has s_i ≈ 0 (active bound) or z_i ≈ 0 (inactive bound with zero multiplier), and the Newton direction wants to push it more negative.

### Why does μ increase?

μ = (s'z + τκ) / (ν + 1)

For μ to increase after a step, the product s'z must increase. This can happen when:
1. The corrector step overcorrects
2. Numerical errors in the KKT solve
3. The centering parameter σ is too high (we're adding centering but not reducing)

### Comparison with ipm1

At iteration 29:
- ipm1: μ = 0.215
- ipm2: μ = 0.528

ipm2 is 2.5x worse! Something in ipm2 is different/broken for this problem.

## What Production Solvers Do

### MOSEK's Approach

1. **Homogeneous Self-Dual Embedding (HSDE):** Handles infeasibility detection naturally. We do this.

2. **NT Scaling:** Uses Nesterov-Todd scaling for symmetric cones. We do this, but may have bugs.

3. **Mehrotra Predictor-Corrector:** Standard approach, but MOSEK uses:
   - Higher-order correctors (Gondzio-style) for ill-conditioned problems
   - Adaptive centering parameter based on predicted vs actual μ reduction
   - Multiple centrality corrector (MCC) steps

4. **Basis Identification/Crossover:** When interior-point gets close, identifies active set and does crossover to simplex-style solution. This is what our polish attempts.

5. **Regularization Strategy:**
   - Proximal-point regularization
   - Dynamic regularization based on KKT solve quality
   - Different regularization for primal vs dual blocks

6. **Scaling/Preconditioning:**
   - Ruiz equilibration (we do this)
   - Additional problem-specific scaling
   - Constraint aggregation for redundant/near-parallel constraints

### Gurobi's Approach

1. **Barrier Method:** Similar predictor-corrector IPM.

2. **Crossover:** Strong emphasis on crossover to basic solution.

3. **Presolve:** Aggressive problem reduction:
   - Bound tightening
   - Redundant constraint removal
   - Variable fixing
   - Aggregation

4. **Numerical Stability:**
   - Robust linear algebra (uses multiple precision if needed)
   - Careful handling of near-degenerate problems

### Clarabel's Approach (Rust reference)

1. **Standard predictor-corrector** with Mehrotra centering
2. **Sparse LDL factorization** with AMD ordering
3. **Iterative refinement** for KKT solves
4. **Combined step** (predictor + corrector in one)

## Improvement Plan

### Phase 1: Diagnostics (understand the problem)

1. **Add detailed per-iteration logging:**
   - Components limiting α_aff (which constraint, s or z side)
   - Condition number of KKT system
   - KKT residual before/after refinement
   - Actual vs predicted μ reduction

2. **Compare ipm1 vs ipm2 step-by-step:**
   - Why does ipm1 make better progress?
   - What's different in the predictor-corrector?

3. **Analyze BOYD2 structure:**
   - How many constraints are bounds?
   - What's the sparsity pattern?
   - Are there near-redundant constraints?

### Phase 2: Fix Known Issues

1. **Interior tolerance bug (from v6 patch):**
   - Apply the fix for relative vs absolute interior tolerance
   - This affects NT scaling fallback

2. **NT scaling fallback:**
   - Current fallback uses sqrt(s/z) which is wrong
   - Should use s/z directly (clamped)

3. **μ monotonicity:**
   - Add safeguard: reject steps that increase μ
   - Fall back to shorter step or pure centering

4. **σ computation:**
   - When mu_aff ≈ mu, current formula gives σ ≈ 1
   - Consider: cap σ more aggressively, or use alternative formula

### Phase 3: Algorithmic Improvements

1. **Multiple Centrality Correctors (MCC):**
   - After predictor-corrector, do additional corrector steps
   - Each corrector improves the step direction
   - Gondzio-style: usually 1-3 extra correctors

2. **Adaptive σ:**
   - Track predicted vs actual μ reduction
   - If actual << predicted, reduce σ next iteration
   - If actual ≈ predicted, can be more aggressive

3. **Better step size computation:**
   - Current: simple backtracking
   - Better: Mehrotra-style asymmetric step sizes for (s,z)
   - Consider: different α for primal vs dual

4. **Higher-order corrections:**
   - Second-order cone: can compute third-order terms
   - Helps near the central path

### Phase 4: Linear Algebra Improvements

1. **KKT solve quality:**
   - Monitor residual ||Kx - b||
   - Adaptive refinement iterations
   - Consider mixed-precision refinement

2. **Ordering:**
   - AMD/CAMD for fill-in reduction
   - Compare with current ordering

3. **Sparse LDL backend:**
   - Consider SuiteSparse integration
   - Better numerical stability

### Phase 5: Presolve Improvements

1. **Bound tightening:**
   - Propagate bounds through constraints
   - Fix variables at bounds when possible

2. **Redundant constraint detection:**
   - Near-parallel rows
   - Dominated constraints

3. **Scaling refinement:**
   - Problem-specific scaling after Ruiz
   - Balance constraint magnitudes

## Specific Investigation for BOYD2

### Key Questions

1. **Why does ipm2 stall while ipm1 doesn't (as badly)?**
   - Compare predictor-corrector implementations line by line
   - Check sigma computation differences
   - Check step size computation differences

2. **What constraints are limiting α_aff?**
   - Add logging to identify limiting constraints
   - Are they bounds or general inequalities?

3. **Is the KKT solve accurate?**
   - Log KKT residuals
   - Try more refinement iterations

4. **Is NT scaling failing?**
   - Check for NT fallback triggers
   - Log scaling block condition numbers

### Immediate Next Steps

1. Apply v6 interior tolerance fix to ipm2
2. Add detailed diagnostic logging for BOYD2
3. Compare ipm1 vs ipm2 predictor-corrector step by step
4. Identify why μ increases at iteration 18

## Success Criteria

- BOYD2 converges to Optimal in ≤30 iterations
- No post-hoc polish hack needed
- μ decreases monotonically (or nearly so)
- gap_rel decreases monotonically
- No worse than Clarabel on iteration count

## References

- MOSEK optimization manual (interior-point chapter)
- Andersen et al., "Implementing Interior Point Methods for Linear Programming"
- Gondzio, "Multiple centrality corrections in a primal-dual method for linear programming"
- Wright, "Primal-Dual Interior-Point Methods" (SIAM)
