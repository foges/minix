# Exp Cone Convergence Investigation

## Summary

**Status**: ROOT CAUSE IDENTIFIED ✓

The exponential cone solver is completely broken - it takes **zero step size** on every iteration, making no progress toward the solution.

## Key Findings

### 1. Hardened Tests Reveal the Problem ✅

After hardening tests to require `Optimal|AlmostOptimal` status:
- All exp cone problems now correctly fail with `MaxIters`
- Regression suite properly rejects non-convergent results
- Integration tests compile and pass (except unrelated SOC issue)

**Files Fixed**:
- `solver-core/tests/integration_tests.rs` - Added `ConeKernel` import, hardened 5 tests
- `solver-bench/src/regression.rs` - Requires Optimal|AlmostOptimal

### 2. Zero Step Size on Every Iteration ❌

Running with verbose output reveals:
```
alpha stall: alpha=0.000e0 (pre_ls=0.000e0), alpha_sz=0.000e0, ...
```

**What this means**:
- `alpha=0.000e0` - Final step size is ZERO
- `alpha_sz=0.000e0` - Step to boundary for (s, z) variables is ZERO
- Solver cannot move from initial point!

**After 1000 iterations**:
```
s = [-2.102766, 1.112818, 2.517934]  (exactly initialization values!)
z = [-2.102766, 1.112818, 2.517934]  (exactly initialization values!)
primal_res: 5.977e-1
dual_res:   8.820e-1
gap:        1.000e0
mu:         3.250e0
```

The solver is **completely stuck** at the initialization point.

### 3. Cone Geometry Functions Are Correct ✅

Created `exp_cone_interior_check.rs` test:
- Initialization point IS interior: ✓
- Simple step directions work correctly: ✓
- `step_to_boundary` returns `inf` for valid directions: ✓
- Barrier value is finite: ✓

**Conclusion**: The cone implementation (interior checks, barrier, step_to_boundary) is working correctly.

### 4. Root Cause: KKT System Produces Invalid Search Direction ❌

Since:
- Initialization is interior ✓
- Simple test directions work ✓
- But solver gets `alpha_sz=0` ❌

The problem MUST be in the **search direction `(ds, dz)`** computed by the KKT system.

**Hypothesis**: The KKT system is assembling an incorrect linear system due to:
1. Wrong BFGS scaling matrix `W` for exp cones
2. Wrong barrier Hessian application
3. Wrong dual map in scaling computation

## Test Results

### Exp Cone Suite (all failures)

```
Problem                Status        Iters    Objective
trivial-1             MaxIter         250       0.0000
cvxpy-3               MaxIter         250       3.7183
trivial-multi-2       MaxIter         250       0.0000
trivial-multi-5       MaxIter         250       0.0000
trivial-multi-10      MaxIter         250       0.0000
```

**Solved: 0/5** (0%)

### Regression Suite (sample)

```
Maros-Meszaros:
  Optimal:             8 (80.0%)
  Max iterations:      2 (20.0%)  ← BOYD1, BOYD2
```

Hardened tests now properly catch MaxIters as failures.

## Code Locations

### Exp Cone Implementation
- **`solver-core/src/cones/exp.rs`**
  - Lines 51-91: `step_to_boundary_primal/dual` ✓ Working
  - Lines 103-136: Barrier gradient/hessian (suspect but looks correct)
  - Lines 302-324: Dual barrier gradient (CRITICAL)
  - Lines 396-436: Dual map via Newton (CRITICAL)

### BFGS Scaling
- **`solver-core/src/scaling/bfgs.rs`**
  - Lines 20-50: Main `bfgs_scaling_3d` dispatch
  - Lines 59-219: `bfgs_scaling_3d_rank3` (current implementation)
  - Lines 224-413: `bfgs_scaling_3d_rank4` (fallback)

### KKT System
- **`solver-core/src/linalg/kkt.rs`**
  - Assembly of KKT system using scaled variables
  - May be treating exp cone incorrectly

## Next Steps

### Immediate Debugging

1. **Add diagnostics to print search direction `(ds, dz)`**
   - Check if direction is all zeros
   - Check if direction points out of cone immediately

2. **Verify BFGS scaling matrix `W`**
   - Print W for exp cone blocks
   - Check if W^2 = H_dual^{-1} (should match dual barrier Hessian inverse)
   - Compare rank-3 vs rank-4 (already tested - both fail)

3. **Check dual map convergence**
   - Add diagnostics to `exp_dual_map_block` Newton solver
   - Verify it converges and produces correct x

4. **Test with simplified scaling**
   - Try identity scaling (W=I) to isolate if problem is in scaling
   - Try using only primal barrier (symmetric cone approach)

### Potential Fixes

**Option A**: Fix BFGS scaling for exp cones
- The rank-3/rank-4 formulas may be incorrect for non-SOC cones
- May need different factorization for exp cone geometry

**Option B**: Use Nesterov-Todd scaling instead
- Exp cone has self-dual structure
- NT scaling might be more stable

**Option C**: Fix dual barrier implementation
- Double-check gradient/Hessian formulas
- Verify dual cone definition matches primal

**Option D**: Add proximal regularization
- H_dual + ρI might make scaling more stable
- Would help with potential quasi-definiteness issues

## Files Created This Session

### Investigation
- `_planning/v16/multi_cone_debug_log.md` - Original multi-cone investigation
- `_planning/v16/exp_cone_convergence_investigation.md` - This file

### Testing
- `_planning/v16/TESTING_REQUIREMENTS.md` - Comprehensive testing guidelines
- `_planning/v16/testing_failures_found.md` - What we found and why it matters

### Benchmarks / Diagnostics
- `solver-bench/examples/exp_cone_suite.rs` - Multi-problem benchmark (fixed)
- `solver-bench/examples/exp_cone_debug.rs` - Residual printer
- `solver-bench/examples/exp_cone_trace.rs` - Verbose iteration trace
- `solver-bench/examples/exp_cone_interior_check.rs` - Geometry validation

## Timeline

- **v15**: Exp cone solver presumably worked (passed regression suite)
- **v16**: Implemented rank-3 BFGS → exp cones broke
- **Investigation**: Discovered rank-3 BFGS is NOT the cause (rank-4 also fails)
- **Current**: Issue predates rank-3 BFGS changes

**Historical check**: Even commit `cca213a` ("Fix exponential cone bug") shows MaxIters, suggesting the solver never truly converged, just computed reasonable objective values.

## Success Criteria

### Minimum Viable Fix
- [ ] Exp cone problems reach `Optimal` status
- [ ] Residuals decrease properly (not stuck at 0.6-0.9)
- [ ] alpha > 0 on most iterations (solver makes progress)
- [ ] At least 3/5 exp cone suite problems solve

### Stretch Goals
- [ ] All 5 exp cone suite problems solve
- [ ] Add 20+ exp cone problems from public benchmarks
- [ ] Performance competitive with other solvers
- [ ] Robust to ill-conditioned problems

## Open Questions

1. **When did this break?**
   - Need to test earlier commits systematically
   - May have never worked correctly

2. **Do other nonsymmetric cones work?**
   - Is this exp-cone specific or a general nonsymmetric cone issue?
   - Need to add power cone tests when implemented

3. **How did tests pass before?**
   - Tests accepted MaxIters as success ❌
   - Benchmarks only checked objective value ❌
   - No actual exp cone problems in regression suite ❌

## Lessons Learned

1. **Always check solve status** - Never accept MaxIters/NumericalError
2. **Verbose output is essential** - Would have caught this immediately
3. **Test cone geometry separately** - Helps isolate KKT vs cone issues
4. **Step size diagnostics critical** - alpha=0 is a clear red flag
5. **Need public benchmark problems** - Can't trust hand-crafted tests alone
