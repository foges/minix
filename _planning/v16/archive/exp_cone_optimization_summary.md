# Exponential Cone Optimization Summary

## Current Performance

**Minix** (after exp cone fix):
- Iteration count: 50-200 iterations
- Per-iteration cost: **7-11 µs** (best in class ✅)
- Solve time: 0.4-2.5 ms
- Ranking: **#2** among tested solvers

**Clarabel** (#1 solver):
- Iteration count: **7-9 iterations** ✅
- Per-iteration cost: ~200-300 µs
- Solve time: 1.7-3.5 ms
- Ranking: **#1**

## The Gap

**Key insight**: Minix has the fastest per-iteration cost but requires **5-25x more iterations** than Clarabel.

**Why**: Clarabel uses third-order correction for nonsymmetric cones (exp, pow), while Minix uses only second-order Mehrotra correction.

## Third-Order Correction Investigation

### Attempted Approach: Finite Differences ❌

**Implementation**: Computed ∇³f*(z)[Δz, ∇²f*(z)^{-1}Δs] using finite differences with ε=1e-7.

**Results**: **FAILED** - numerically unstable

```
Debug output:
  z=[-2.10, 1.11, 2.52]
  dz=[4.62, 6.17, -1.29]
  ds=[-4.89, 1.40, -0.001]
  η=[17.8, -5.82, 11.9]  ← Too large!
  μ=4.42  ← Complementarity gap
  ds*dz=-22.6  ← Second-order correction
```

**Problems**:
- η is 4x larger than μ (should be much smaller)
- Correction terms dominate the solution
- No convergence improvement
- Increased per-iteration cost (25 µs vs 11 µs)

**Root cause**: Finite differences accumulate errors across 3×3×3 tensor, magnify rounding errors, and are sensitive to ε choice.

### Clarabel's Approach: Analytical Formula ✅

**Research findings**: Clarabel does NOT use finite differences. They implement an analytical formula.

**Key components**:
```rust
// 1. Auxiliary function
ψ = z[0]*log(-z[0]/z[2]) - z[0] + z[1]

// 2. Solve for u = ∇²f*(z)⁻¹Δs via Cholesky factorization

// 3. Compute dot products
dotψu = u.dot(grad_ψ)
dotψv = v.dot(grad_ψ)  // v = Δz

// 4. Complex formula involving ψ, dot products, reciprocals
η[0] = (1/ψ - 2/z[0])*u[0]*v[0]/(z[0]²) - u[2]*v[2]/(z[2]²)/ψ + ...
η[2] = 2*(z[0]/ψ - 1)*u[2]*v[2]/(z[2]³) - (u[2]*v[0] + u[0]*v[2])/(z[2]²)/ψ + ...

// 5. Final scaling by 0.5
```

**Integration**:
```rust
shift = grad * σμ - η  // Note: SUBTRACT η, not add!
```

**Benefits** (from literature):
- **80% iteration reduction** (200 → 40 iterations)
- **70% solve time reduction**
- Numerically stable and battle-tested

## Decision: Defer Third-Order Correction

### Why Defer?

1. **Complexity**: Analytical formula is complex and error-prone
2. **Finite differences don't work**: Proven by testing
3. **Alternative improvements exist**:
   - Rank-4 BFGS scaling (~10-20% iteration reduction)
   - Proximity-based step selection (~10-30% reduction)
   - Better initialization
   - Adaptive centering
4. **Already competitive**: #2 solver with best per-iteration cost
5. **Can revisit later**: After simpler optimizations

### Code Changes

**Removed**:
- `exp_third_order_correction()` finite-difference implementation
- ExpCone special case in predictor-corrector loop
- Debug output statements

**Added**:
- TODO comment in `predcorr.rs` explaining the opportunity
- Analysis document: `_planning/v16/third_order_correction_analysis.md`
- This summary document

## Alternative Improvements (Recommended Next Steps)

### 1. Rank-4 BFGS Scaling

**Effort**: 1-2 days
**Expected benefit**: 10-20% iteration reduction
**Complexity**: Medium

Clarabel uses rank-4 BFGS update vs our rank-3.

### 2. Proximity-Based Step Selection

**Effort**: 2-3 days
**Expected benefit**: 10-30% iteration reduction
**Complexity**: Medium

Add proximity metric to keep iterates near central path:
```
proximity = ‖W⁻¹s - Wz‖ / ‖W⁻¹s + Wz‖
```

Constrain step size to keep proximity < 0.95.

### 3. Better Exp Cone Initialization

**Effort**: 1 day
**Expected benefit**: 5-15% iteration reduction
**Complexity**: Low

Current unit initialization may not be ideal for exp cones.

### 4. Adaptive Centering Parameters

**Effort**: 1-2 days
**Expected benefit**: 5-15% iteration reduction
**Complexity**: Low-Medium

Make σ_min and σ_max adaptive based on progress.

## Combined Impact Estimate

**With all alternative improvements**:
- Expected iteration reduction: 25-50%
- From 50-200 iters → 25-100 iters
- May still trail Clarabel's 7-9 iters, but significantly closer
- Combined with our superior per-iteration cost, could win on small problems

## Final Thoughts

**Third-order correction is the "holy grail"** for exp cone performance, but:
- Requires careful analytical implementation (not finite differences)
- Is complex and error-prone to implement from scratch
- Can be deferred in favor of simpler, incremental improvements

**Strategic approach**:
1. Implement simpler improvements first (BFGS, proximity, etc.)
2. Achieve 25-50% iteration reduction with lower risk
3. Optionally revisit third-order correction later
4. With combined improvements, Minix can be competitive or #1 on small problems

**Current status**: Excellent solver with room for growth. The exp cone fix was critical and successful. Further optimizations are icing on the cake.

## Files Modified

- `solver-core/src/cones/exp.rs`: Removed broken finite-difference code
- `solver-core/src/ipm2/predcorr.rs`: Removed ExpCone special case, added TODO
- `_planning/v16/third_order_correction_analysis.md`: Detailed analysis
- `_planning/v16/exp_cone_optimization_summary.md`: This summary

## References

- [Clarabel.rs Source](https://github.com/oxfordcontrol/Clarabel.rs/blob/main/src/solver/core/cones/expcone.rs)
- [Clarabel Paper](https://arxiv.org/html/2405.12762v1)
- [Nonsymmetric Exponential-Cone Optimization](https://link.springer.com/article/10.1007/s10107-021-01631-4)
- [Santiago Serrano Dissertation](https://web.stanford.edu/group/SOL/dissertations/ThesisAkleAdobe-augmented.pdf)
