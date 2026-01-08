# Exponential Cone Bug Investigation - ONGOING

## Status: PARTIALLY SOLVED - Non-symmetric cone issue identified

## Summary

Exp cones completely diverge in optimization (objectives off by millions), but all primitive tests pass. Root cause appears to be related to exp cones being **non-symmetric** (K ≠ K*).

## Findings

### ✅ What Works
1. **Exp cone primitives**: All unit tests pass
   - `exp_primal_interior()`: Correctly identifies interior points
   - `exp_dual_interior()`: Correctly identifies dual interior points
   - `exp_barrier_grad()`: Finite gradients
   - `step_to_boundary()`: Returns valid alpha

2. **Unit initialization**: `[-1.051383, 0.556409, 1.258967]` is interior for both primal and dual

3. **Preprocessing**: Correctly skips exp cone rows (doesn't remove them)

4. **SDP cones**: Work perfectly (proves general infrastructure is sound)

### ❌ What Fails
1. **All optimization problems with exp cones diverge**:
   - Trivial problem: obj = -204M instead of ~0
   - CVXPY problem: obj = -1.2e25 instead of ~4.7
   - Simple problem: obj = -351M instead of ~1.0

2. **Symptom**: `alpha_sz = 0` every iteration (step-to-boundary returns 0)

### Attempted Fixes (didn't work)
1. ✅ Added `push_to_interior()` call after initialization
2. ✅ Verified preprocessing doesn't break cone structure
3. ✅ Added extensive debug logging

### Key Clue: Non-Symmetric Cones
User pointed out: **Exp cones are non-symmetric** (K_exp ≠ K_exp*)

- Primal cone: K_exp = {(x,y,z) : z ≥ y*exp(x/y), y > 0}
- Dual cone: K_exp* = {(u,v,w) : u < 0, w ≥ -u*exp(v/u - 1)}

For symmetric cones (Zero, NonNeg, SOC, PSD): K = K*
For non-symmetric cones (Exp, Pow): K ≠ K*

## Hypothesis

The bug is likely in how primal vs. dual cones are handled in:
1. **KKT system assembly**: May be using wrong cone for z
2. **NT scaling (BFGS)**: Uses dual map which may have sign errors
3. **Barrier gradient application**: May have wrong signs

## Diagnostic Tests Added

Created comprehensive test suite in `solver-bench/src/conic_benchmarks.rs`:
```rust
test_exp_debug_trivial()       // Simplest: min x s.t. (x,1,1) ∈ K_exp
test_exponential_cone_cvxpy()  // CVXPY-style test
test_exponential_cone_simple() // Basic exp cone problem
```

Created unit tests in `solver-core/src/cones/exp.rs`:
```rust
test_exp_primal_interior()     // ✅ PASS
test_exp_dual_interior()       // ✅ PASS
test_exp_barrier_grad()        // ✅ PASS
test_step_to_boundary()        // ✅ PASS
test_unit_initialization_is_interior() // ✅ PASS
test_barrier_gradient_sign()   // NEW - check gradient correctness
```

## Next Steps

1. **Check BFGS scaling for sign errors**:
   - File: `solver-core/src/scaling/bfgs.rs`
   - The dual map and NT scaling for nonsymmetric cones is complex
   - May have wrong signs in gradient or Hessian

2. **Verify barrier function matches literature**:
   - Current: `f(s) = -ln(ψ) - ln(y) - ln(z)` where `ψ = y*ln(z/y) - x`
   - Check against ECOS/SCS implementations

3. **Check KKT system for primal/dual cone confusion**:
   - Standard form: Ax + s = b, s ∈ K (primal)
   - Dual: A'y + z = c, z ∈ K* (dual)
   - Verify z is checked against K*, not K

4. **Try reference problem from CVXPY/ECOS**:
   - Get known-good problem + solution
   - Verify our barrier gives same KKT residuals

## Files Modified

- `solver-core/src/cones/exp.rs`: Added 8 unit tests + debug logging
- `solver-core/src/ipm2/solve.rs`: Added `push_to_interior()` call
- `solver-core/src/ipm/hsde.rs`: Added debug logging to `push_to_interior()`
- `solver-bench/src/conic_benchmarks.rs`: Added 3 exp cone benchmark problems
- `solver-core/src/presolve/eliminate.rs`: Added debug logging (verified correct)

## Code Changes Summary

### Added push_to_interior() after initialization
```rust
// solver-core/src/ipm2/solve.rs
state.initialize_with_prob(&cones, &scaled_prob);
state.push_to_interior(&cones, 1e-2);  // NEW - ensure interior
```

### Added debug logging
```rust
// solver-core/src/cones/exp.rs:step_to_boundary_primal()
if !is_int {
    eprintln!("WARNING: exp cone s NOT interior: {:?}", s_block);
}
```

## References

- ECOS solver: https://github.com/embotech/ecos
- SCS solver: https://github.com/cvxgrp/scs
- Exponential cone barrier: Skajaa & Ye (2015) "A homogeneous interior-point algorithm for nonsymmetric convex conic optimization"
- CVXPY exp cone docs: https://www.cvxpy.org/tutorial/advanced/index.html#exponential-cone

## Test Output Example

```
=== Exp Cone Trivial Debug Test ===
Status: MaxIters
Iterations: 20
Objective: -204627059.627428
Solution: x=-204627059.627428
Expected: x=0, objective=0

presolve: skipping Exp { count: 1 } cone row 0  ← Correctly skipped!
presolve: singleton_rows=1 non_singleton_rows=2
alpha stall: alpha_sz=0.000e0 (every iteration!)  ← Bug symptom
```

## Recommendation

Given the complexity of non-symmetric cone handling and that SDP (symmetric) cones work perfectly, the bug is likely in:
1. BFGS scaling sign errors
2. Barrier gradient sign convention
3. Primal/dual cone confusion in KKT system

Suggest comparing step-by-step with a reference implementation (ECOS or SCS) on a simple exp cone problem to find the discrepancy.
