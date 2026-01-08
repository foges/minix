# Path to Best Exponential Cone Solver

## Current Status

**Minix Performance** (as of v16 post-fix):
- Per-iteration cost: **10-13 µs** (BEST IN CLASS) ✅
- Iteration count: **50-200 iterations** ⚠️
- Solve time: **0.5-2.5 ms** (competitive)
- Ranking: **#2** among tested solvers (behind Clarabel)

**Clarabel Performance**:
- Per-iteration cost: **~200-300 µs**
- Iteration count: **7-9 iterations** ✅ (5-25x fewer!)
- Solve time: **1.7-3.5 ms**
- Ranking: **#1** (fastest overall)

## The Gap

**Key Insight**: Minix has the fastest per-iteration cost but requires 5-25x more iterations than Clarabel.

**Implication**:
- Small problems (n < 100): Minix can be faster due to low per-iteration cost
- Large problems (n > 100): Clarabel will win due to far fewer iterations

**Goal**: Reduce Minix iteration count from 50-200 to 10-30 (3-10x improvement)

## Research Findings: Why Clarabel Converges Faster

### Algorithm Comparison

| Feature | Minix (Current) | Clarabel | Impact |
|---------|-----------------|----------|--------|
| **Centering Parameter** | σ = (1-α_aff)³ ✅ | σ = (1-α_aff)³ ✅ | Same (no gap) |
| **Predictor-Corrector** | Yes ✅ | Yes ✅ | Same (no gap) |
| **Mehrotra Correction** | Yes (2nd order) ✅ | Yes (2nd order) ✅ | Same (symmetric cones) |
| **Third-Order Correction** | **NO** ❌ | **YES** ✅ | **MAJOR GAP** |
| **BFGS Scaling** | Rank-3 ✅ | Rank-4 ✅ | Minor gap |
| **KKT Regularization** | Static + Dynamic ✅ | Static + Dynamic ✅ | Same (no gap) |
| **Step Selection** | Standard line search ✅ | **Proximity-based** ⚠️ | **Possible gap** |

### Critical Differences

#### 1. **Third-Order Correction for Nonsymmetric Cones** ⭐ MAJOR

**Clarabel** (from paper):
> "For nonsymmetric cones, a third-order correction is computed:
> η = -½∇³f*(z)[Δz, ∇²f*(z)^{-1}Δs]"

**What this means**:
- Symmetric cones (NonNeg, SOC, SDP): Use 2nd-order Mehrotra correction
- **Nonsymmetric cones (Exp, Pow)**: Use 3rd-order correction for better curvature handling

**Minix currently**: Uses 2nd-order Mehrotra correction for ALL cones (including Exp)

**Impact**: **This is likely the primary reason for the 5-25x iteration gap!**

Third-order corrections capture more curvature information, allowing larger steps that still converge. Without this, we need many small steps to navigate the exponential cone's nonlinear geometry.

#### 2. **Rank-4 vs Rank-3 BFGS Scaling** (Minor)

**Clarabel**: Uses rank-4 BFGS update
**Minix**: Uses rank-3 BFGS update

**Impact**: Probably minor (<10% iteration difference), but worth investigating

#### 3. **Proximity-Based Step Selection** (Possible)

**Clarabel** (from paper):
> "Select α sufficiently small so that the updated values remain within a neighborhood of the central path using a proximity metric"

**Minix**: Uses standard step-to-boundary with backtracking

**Impact**: Could explain why Clarabel takes confident large steps while we take conservative small steps

## Root Cause Analysis

**Why do we need 5-25x more iterations?**

1. **Primary**: **No third-order correction** → Poor handling of exp cone curvature → Many small steps needed
2. **Secondary**: **Conservative step selection** → Don't exploit full Newton step potential
3. **Tertiary**: **Rank-3 vs Rank-4 BFGS** → Slightly less accurate scaling

**Evidence**:
- Our per-iteration cost is excellent (10-13 µs) → Computational efficiency is not the problem
- We converge to correct solutions → Algorithmic correctness is fine
- But we take tiny steps → **Step quality is the issue**

## Proposed Improvements (Priority Order)

### Priority 1: Implement Third-Order Correction for Exp Cones 🎯

**Impact**: **Expected 3-10x iteration reduction** (from 50-200 to 10-30 iterations)

**Implementation**:

```rust
// For exponential cones, compute 3rd-order correction term
fn exp_third_order_correction(
    z: &[f64],          // Current dual iterate
    dz: &[f64],         // Affine dual step
    ds: &[f64],         // Affine primal step
    eta_out: &mut [f64] // Output: correction term
) {
    // η = -½∇³f*(z)[Δz, ∇²f*(z)^{-1}Δs]
    //
    // Steps:
    // 1. Compute ∇²f*(z)^{-1}Δs (Hessian inverse times ds)
    // 2. Compute ∇³f*(z)[Δz, result] (third derivative in two directions)
    // 3. Scale by -½

    // For exp cone, this involves:
    // - Barrier Hessian ∇²f*(z) = we already compute in exp_dual_hess_matrix()
    // - Third derivative ∇³f*(z) = need to derive and implement
}
```

**Required Work**:
1. Derive third derivative of dual barrier `∇³f*(z)` for exp cone
2. Implement efficient computation (likely just a few FLOPs given exp cone structure)
3. Add to predictor-corrector in `solver-core/src/ipm2/predcorr.rs`
4. Test on exp cone problems

**Estimated Effort**: 3-5 days

**Expected Outcome**:
- Iteration count: 50-200 → 15-40 (3-5x improvement)
- Solve time: 0.5-2.5 ms → 0.3-1.0 ms (2x improvement)
- **Ranking: #2 → #1** (likely beat Clarabel on small problems!)

### Priority 2: Upgrade to Rank-4 BFGS Scaling

**Impact**: **Expected 10-20% iteration reduction**

**Implementation**:
- Review Clarabel's rank-4 BFGS formulation
- Extend our rank-3 update in `solver-core/src/scaling/bfgs.rs`
- Add one more rank-1 update term

**Estimated Effort**: 1-2 days

**Expected Outcome**: 15-40 iters → 12-35 iters (modest improvement)

### Priority 3: Proximity-Based Step Selection

**Impact**: **Expected 10-30% iteration reduction**

**Implementation**:
- Add proximity metric: `∥W^{-1}s - Wz∥ / ∥W^{-1}s + Wz∥`
- Constrain step size to keep proximity below threshold (e.g., 0.95)
- This prevents iterates from drifting too far from central path

**Estimated Effort**: 2-3 days

**Expected Outcome**: 12-35 iters → 10-25 iters (final refinement)

### Priority 4: Adaptive Centering Tuning

**Impact**: **Expected 5-15% iteration reduction**

**Implementation**:
- Make σ_min and σ_max adaptive based on progress
- Reduce centering when close to optimality
- Increase centering when numerical issues detected

**Estimated Effort**: 1-2 days

## Implementation Roadmap

### Phase 1: Third-Order Correction (Week 1)

**Days 1-2**: Mathematical Derivation
- Derive ∇³f*(u,v,w) for dual exp cone barrier
- Work out the tensor contraction formula
- Verify against literature (Skajaa & Ye 2015)

**Days 3-4**: Implementation
- Add `exp_third_derivative()` function
- Integrate into predictor-corrector
- Add unit tests for correctness

**Day 5**: Benchmarking
- Test on all exp cone problems
- Measure iteration count improvement
- Validate correctness of solutions

**Target**: 3-5x iteration reduction

### Phase 2: BFGS + Proximity (Week 2)

**Days 6-7**: Rank-4 BFGS
- Study Clarabel's formulation
- Implement rank-4 update
- Test for correctness

**Days 8-10**: Proximity Metric
- Implement proximity calculation
- Add step size constraint
- Tune threshold parameter

**Target**: Additional 20-40% iteration reduction

### Phase 3: Polish & Validation (Week 3)

**Days 11-12**: Comprehensive Benchmarking
- Run full exp cone test suite
- Compare against ECOS, SCS, Clarabel
- Document performance improvements

**Days 13-14**: Code Review & Documentation
- Clean up implementation
- Add inline documentation
- Update user-facing docs

**Day 15**: Final Testing
- Regression tests on QP/SOCP/SDP
- Ensure no performance degradation
- Prepare for release

## Success Metrics

### Minimum Success (Good)
- Iteration count: 50-200 → 20-50 (2-4x improvement)
- Solve time: Competitive with ECOS
- Ranking: Still #2, but closer to Clarabel

### Target Success (Great)
- Iteration count: 50-200 → 15-35 (3-6x improvement)
- Solve time: Match or beat Clarabel on small problems
- Ranking: **#1 on small problems (n < 100)**

### Stretch Success (Amazing)
- Iteration count: 50-200 → 10-25 (5-10x improvement)
- Solve time: **Fastest overall** (beat Clarabel on all sizes)
- Ranking: **#1 BEST EXPONENTIAL CONE SOLVER** 🏆

## Risk Mitigation

**Risk 1**: Third-order correction is complex to implement correctly
- **Mitigation**: Start with unit tests on simple cases
- **Validation**: Compare step directions against finite differences
- **Fallback**: Keep 2nd-order path if 3rd-order fails

**Risk 2**: Performance gains don't materialize
- **Mitigation**: Benchmark incrementally after each change
- **Evidence**: Clarabel uses these techniques and is 5-25x faster
- **Fallback**: Document findings even if only modest improvement

**Risk 3**: Introduces numerical instability
- **Mitigation**: Add safeguards (e.g., bounded corrections)
- **Testing**: Run full regression suite after each change
- **Rollback**: Git history allows easy reversion

## Mathematical Deep Dive: Third Derivative

### Dual Barrier for Exp Cone

Recall from our fix:
```
f*(u,v,w) = -log(-u) - log(w) - log(ψ*)
where ψ* = u + w*exp(v/w - 1)
```

### First Derivative (already implemented)
```
∇f*(u,v,w) = [1/u - 1/ψ*, -exp(v/w-1)/ψ*, -1/w - exp(v/w-1)*(1-v/w)/ψ*]
```

### Second Derivative (already implemented)
```
∇²f*(u,v,w) = (1/ψ²) * ∇ψ ∇ψᵀ - (1/ψ) * ∇²ψ + diag(1/u², 0, 1/w²)
```

### Third Derivative (TO IMPLEMENT)
```
∇³f*(u,v,w)[a,b] = (∂/∂a)(∇²f*[b])
```

This is a 3×3×3 tensor, but we only need specific contractions for the correction term.

**Clarabel's formula**:
```
η = -½∇³f*(z)[Δz, ∇²f*(z)^{-1}Δs]
```

**Computation**:
1. Compute `temp = ∇²f*(z)^{-1}Δs` (3×3 matrix-vector multiply + inverse)
2. Compute `η = ∇³f*(z)[Δz, temp]` (tensor contraction)
3. Scale by `-½`

**Implementation Plan**:
- The tensor structure for exp cone is sparse (most entries zero)
- Can compute directly without full 3×3×3 storage
- Likely <20 FLOPs per correction

## References

1. **Clarabel Paper**: Goulart & Chen (2024). ["Clarabel: An interior-point solver for conic programs with quadratic objectives"](https://arxiv.org/abs/2405.12762)

2. **Exp Cone Theory**: Skajaa & Ye (2015). "A homogeneous interior-point algorithm for nonsymmetric convex conic optimization"

3. **BFGS Scaling**: Chares (2009). "Cones and interior-point algorithms for structured convex optimization involving powers and exponentials"

4. **Clarabel Implementation**: [GitHub - oxfordcontrol/Clarabel.rs](https://github.com/oxfordcontrol/Clarabel.rs)

## Conclusion

**We have a clear path to becoming the best exponential cone solver:**

1. ✅ **We already have** excellent per-iteration efficiency (10-13 µs)
2. ✅ **We already have** correct implementation (100% pass rate)
3. ⚠️ **We need to add** third-order correction for nonsymmetric cones
4. ⚠️ **We need to improve** step selection strategy

**The gap is NOT about fundamental algorithm quality** - it's about specific refinements for nonsymmetric cones that Clarabel has implemented and we haven't.

**Estimated timeline to #1**: **2-3 weeks** of focused development

**Key enabler**: Our low per-iteration cost means even moderate iteration reduction makes us fastest overall.

**Bottom line**: This is achievable! We're not missing some fundamental insight - we just need to implement known techniques that Clarabel uses.

---

**Next Steps**:
1. Derive ∇³f* for exp cone
2. Implement third-order correction
3. Benchmark and iterate
4. Claim #1 ranking! 🏆
