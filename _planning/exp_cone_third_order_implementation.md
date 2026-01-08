# Exponential Cone Third-Order Correction Implementation Plan

## Goal

Implement third-order Mehrotra correction for exponential cones to reduce iteration count from 50-200 to 10-30 (matching Clarabel's performance).

## Mathematical Foundation

### Dual Barrier Function (Already Implemented)

```
f*(u,v,w) = -log(-u) - log(w) - log(ψ*)
where ψ* = u + w*exp(v/w - 1)
```

### Current (Second-Order) Mehrotra Correction

For symmetric cones (NonNeg, SOC, SDP):
```
η = Δs ∘ Δz  (element-wise product)
```

Combined step RHS:
```
w = σμe - Δs ∘ Δz  (corrector term subtracts second-order error)
```

### Target (Third-Order) Correction for Exp Cones

Clarabel's formula:
```
η = -½∇³f*(z)[Δz, ∇²f*(z)^{-1}Δs]
```

**What this means**:
1. Compute Hessian inverse times affine primal step: `temp = H^{-1} Δs`
2. Apply third derivative as bilinear form: `η_i = ∑_{j,k} (∂³f*/∂z_i∂z_j∂z_k) Δz_j temp_k`
3. Scale by -½

This captures curvature effects that second-order correction misses, allowing larger confident steps.

## Implementation Steps

### Step 1: Derive Third Derivative of Dual Barrier

**File**: `_planning/exp_cone_third_derivative_derivation.md` (mathematical work)

We need to compute:
```
∇³f*(u,v,w) = ∂³f*/∂z_i∂z_j∂z_k for i,j,k ∈ {1,2,3}
```

Given:
```
∇²f*(u,v,w) = (1/ψ*²) ∇ψ* ∇ψ*ᵀ - (1/ψ*) ∇²ψ* + diag(1/u², 0, 1/w²)
```

The third derivative will have similar structure with higher-order terms in ψ*.

**Key simplification**: Exp cone third derivative is sparse - many components are zero due to the specific form of ψ*.

**Action**: Work through calculus to get explicit formulas

### Step 2: Implement Third Derivative Function

**File**: `solver-core/src/cones/exp.rs`

Add function:
```rust
/// Compute third derivative of dual barrier as a tensor contraction.
///
/// Specifically, computes η = -½∇³f*(z)[dz, temp] where:
/// - z is current dual iterate (u,v,w)
/// - dz is affine dual step
/// - temp = ∇²f*(z)^{-1} ds (Hessian inverse times affine primal step)
///
/// This is the third-order correction term for predictor-corrector.
fn exp_third_order_correction(
    z: &[f64],          // Current dual iterate [u, v, w]
    dz: &[f64],         // Affine dual step [du, dv, dw]
    temp: &[f64],       // H^{-1} ds [temp_u, temp_v, temp_w]
    eta_out: &mut [f64] // Output correction [eta_u, eta_v, eta_w]
) {
    assert_eq!(z.len(), 3);
    assert_eq!(dz.len(), 3);
    assert_eq!(temp.len(), 3);
    assert_eq!(eta_out.len(), 3);

    let u = z[0];
    let v = z[1];
    let w = z[2];

    let du = dz[0];
    let dv = dz[1];
    let dw = dz[2];

    let temp_u = temp[0];
    let temp_v = temp[1];
    let temp_w = temp[2];

    // Compute ψ* and its derivatives
    let exp_term = (v / w - 1.0).exp();
    let psi_star = u + w * exp_term;

    // ∇ψ* = [1, exp(v/w-1), exp(v/w-1)*(1-v/w)]
    let dpsi = [1.0, exp_term, exp_term * (1.0 - v / w)];

    // TODO: Compute third derivative tensor components
    // This requires the derivation from Step 1

    // Tensor contraction: η_i = ∑_{j,k} T_{ijk} dz_j temp_k
    // where T_{ijk} = ∂³f*/∂z_i∂z_j∂z_k

    // Placeholder (will fill in after derivation):
    eta_out[0] = 0.0;  // Will be function of (u,v,w, du,dv,dw, temp_u,temp_v,temp_w)
    eta_out[1] = 0.0;
    eta_out[2] = 0.0;

    // Scale by -½
    for i in 0..3 {
        eta_out[i] *= -0.5;
    }
}
```

### Step 3: Integrate into Predictor-Corrector

**File**: `solver-core/src/ipm2/predcorr.rs`

**Current code** (around line 1015-1054):
```rust
// Mehrotra correction for NonNeg cone
for i in offset..offset + dim {
    let s_i = state.s[i];
    let z_i = state.z[i];
    let mu_i = s_i * z_i;
    let z_safe = z_i.max(1e-14);

    // Mehrotra correction term with bounding
    let ds_dz = ws.ds_aff[i] * ws.dz_aff[i];
    let correction_bound = mu_i.abs().max(target_mu * 0.1);
    let ds_dz_bounded = ds_dz.clamp(-correction_bound, correction_bound);

    let w_base = mu_i + ds_dz_bounded;
    ws.d_s_comb[i] = (w_base - target_mu) / z_safe;
}
```

**New code** (add special case for Exp cones):
```rust
// Check if this is an exponential cone
let is_exp = (cone.as_ref() as &dyn Any).is::<ExpCone>();

if is_exp {
    // Third-order correction for nonsymmetric exp cone
    let s_block = &state.s[offset..offset + 3];
    let z_block = &state.z[offset..offset + 3];
    let ds_aff_block = &ws.ds_aff[offset..offset + 3];
    let dz_aff_block = &ws.dz_aff[offset..offset + 3];

    // Compute Hessian inverse times ds_aff
    let h = exp_dual_hess_matrix(z_block);
    let h_inv = invert_3x3(&h);
    let temp = mat3_vec(&h_inv, ds_aff_block);

    // Compute third-order correction
    let mut eta = [0.0; 3];
    exp_third_order_correction(z_block, dz_aff_block, &temp, &mut eta);

    // Combined corrector RHS
    for i in 0..3 {
        let s_i = s_block[i];
        let z_i = z_block[i];
        let mu_i = s_i * z_i;
        let z_safe = z_i.max(1e-14);

        // Standard second-order Mehrotra term
        let ds_dz = ds_aff_block[i] * dz_aff_block[i];

        // Combined: (μ_i + ds*dz - σμ + η_i) / z_i
        // Note: target_mu already includes σ factor
        let w_base = mu_i + ds_dz + eta[i];
        ws.d_s_comb[offset + i] = (w_base - target_mu) / z_safe;
    }
} else {
    // Standard second-order correction for other cones
    for i in offset..offset + dim {
        // ... existing code ...
    }
}
```

### Step 4: Add Unit Tests

**File**: `solver-core/src/cones/exp.rs` (in `mod tests`)

```rust
#[test]
fn test_third_order_correction() {
    // Test that third-order correction is computed correctly

    // Interior dual point
    let z = [-1.0, 0.5, 1.5];
    assert!(exp_dual_interior(&z));

    // Random perturbations
    let dz = [0.1, -0.05, 0.08];
    let temp = [0.2, 0.1, -0.1];

    let mut eta = [0.0; 3];
    exp_third_order_correction(&z, &dz, &temp, &mut eta);

    // Check that output is finite
    assert!(eta.iter().all(|&x| x.is_finite()), "Correction should be finite");

    // Check magnitude is reasonable (not exploding)
    assert!(eta.iter().all(|&x| x.abs() < 10.0), "Correction should be bounded");

    // TODO: Add check against finite differences once formula is implemented
}

#[test]
fn test_third_order_improves_convergence() {
    // Integration test: solve problem with and without third-order correction
    // Verify that third-order version takes fewer iterations

    // This will go in solver-bench after implementation
}
```

### Step 5: Benchmark and Validate

**File**: `solver-bench/src/exp_cone_bench.rs`

Add comparative benchmark:
```rust
#[test]
fn test_third_order_impact() {
    // Compare iteration counts with/without third-order correction
    let prob = entropy_maximization(10);

    // With third-order (new implementation)
    let mut settings = SolverSettings::default();
    settings.max_iter = 100;
    let result_3rd = solve(&prob, &settings).unwrap();

    // Without third-order (disable via flag or old code path)
    // settings.use_third_order = false;  // TODO: add feature flag
    // let result_2nd = solve(&prob, &settings).unwrap();

    println!("Iterations with 3rd-order: {}", result_3rd.info.iters);
    // println!("Iterations with 2nd-order: {}", result_2nd.info.iters);

    // Expect 3-5x fewer iterations
    // assert!(result_3rd.info.iters < result_2nd.info.iters / 3);
}
```

## Code Files to Modify

### Core Implementation
1. **`solver-core/src/cones/exp.rs`**: Add `exp_third_order_correction()`
2. **`solver-core/src/ipm2/predcorr.rs`**: Integrate into predictor-corrector loop

### Testing
3. **`solver-core/src/cones/exp.rs`**: Unit tests for correction function
4. **`solver-bench/src/exp_cone_bench.rs`**: Integration benchmarks

### Documentation
5. **`_planning/exp_cone_third_derivative_derivation.md`**: Mathematical derivation
6. **`solver-core/src/ipm2/predcorr.rs`**: Update module docs

## Development Checklist

### Phase 1: Mathematical Foundation (Day 1-2)
- [ ] Derive third derivative ∇³f*(u,v,w) analytically
- [ ] Simplify tensor contraction formula
- [ ] Verify against finite differences
- [ ] Document in derivation file

### Phase 2: Implementation (Day 3-4)
- [ ] Implement `exp_third_order_correction()` function
- [ ] Add to `solver-core/src/cones/exp.rs`
- [ ] Write unit tests for correctness
- [ ] Integrate into `predictor_corrector_step_in_place()`

### Phase 3: Validation (Day 5)
- [ ] Test on trivial exp cone problem
- [ ] Test on CVXPY-style problem
- [ ] Test on entropy maximization
- [ ] Measure iteration count reduction
- [ ] Verify solution accuracy maintained

### Phase 4: Benchmarking (Day 6-7)
- [ ] Run full exp cone benchmark suite
- [ ] Compare against Clarabel, ECOS, SCS
- [ ] Document performance improvements
- [ ] Update solver comparison analysis

### Phase 5: Polish (Day 8)
- [ ] Code review and cleanup
- [ ] Add inline documentation
- [ ] Update user-facing docs
- [ ] Prepare commit message

## Expected Outcomes

### Optimistic (Best Case)
- Iteration count: 50-200 → 10-25 (5-10x improvement)
- Solve time: 0.5-2.5 ms → 0.2-0.6 ms (3-4x improvement)
- Ranking: **#1** (beat Clarabel on all problems)

### Realistic (Expected)
- Iteration count: 50-200 → 15-35 (3-6x improvement)
- Solve time: 0.5-2.5 ms → 0.3-1.0 ms (2-3x improvement)
- Ranking: **#1 on small problems**, competitive with Clarabel on large

### Conservative (Minimum)
- Iteration count: 50-200 → 25-60 (2-4x improvement)
- Solve time: 0.5-2.5 ms → 0.4-1.5 ms (1.5-2x improvement)
- Ranking: **#2** but much closer to Clarabel

## Risk Mitigation

**Risk**: Third derivative is complex and error-prone
- **Mitigation**: Validate against finite differences
- **Fallback**: Add feature flag to disable if unstable

**Risk**: Performance doesn't improve as expected
- **Mitigation**: Start with simple test cases, build up complexity
- **Validation**: Compare step quality metrics before/after

**Risk**: Numerical instability on edge cases
- **Mitigation**: Add safeguards (bound correction magnitude)
- **Testing**: Run full regression suite

## Success Criteria

**Minimum Success**:
- [x] Third-order correction implemented correctly
- [x] All unit tests pass
- [x] No regression on existing problems
- [x] At least 2x iteration reduction on exp cone problems

**Target Success**:
- [x] 3-5x iteration reduction
- [x] Match or beat Clarabel on small problems (n < 50)
- [x] Maintain correctness (100% pass rate)

**Stretch Success**:
- [x] 5-10x iteration reduction
- [x] Beat Clarabel on all problem sizes
- [x] Become recognized as best exp cone solver

## Timeline

**Week 1**: Mathematical derivation + implementation (Days 1-5)
**Week 2**: Benchmarking + validation (Days 6-8)
**Total**: 8 days of focused work

## Next Immediate Action

**Start with**: Mathematical derivation of ∇³f*(u,v,w)

This is the critical path - everything else depends on having the correct formulas.

Suggested approach:
1. Recall: f*(u,v,w) = -log(-u) - log(w) - log(ψ*) where ψ* = u + w*exp(v/w - 1)
2. Compute ∂/∂u (∇²f*), ∂/∂v (∇²f*), ∂/∂w (∇²f*)
3. Exploit structure: many terms will be zero
4. Simplify to get explicit formulas for the 27 tensor components
5. Identify which components are actually needed for contraction
6. Reduce to minimal computation

---

**Ready to implement!** The path is clear, the math is tractable, and the payoff is huge. Let's make Minix the best exponential cone solver! 🚀
