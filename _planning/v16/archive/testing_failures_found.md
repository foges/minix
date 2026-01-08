# Critical Testing Failures Found - Session v16

## Problem Summary

The exp cone solver has been **completely broken** for an unknown period, but our tests didn't catch it because they were checking the wrong things.

**Symptom**: Solver runs for 50-1000 iterations without converging (primal_res=0.6, dual_res=0.9, gap=1.0, μ=3.25 - completely stuck), then returns MaxIters status with the "correct" objective value.

**Why tests didn't catch it**: Tests accepted `MaxIters` status as passing, or only checked that solve() returned Ok() without validating the status field.

## Testing Failures Found

### 1. Integration Tests Accept MaxIters ❌

**File**: `solver-core/tests/integration_tests.rs`
**Line**: 59-62

```rust
assert!(matches!(
    result.status,
    SolveStatus::Optimal | SolveStatus::MaxIters  // ❌ WRONG!
));
```

**Problem**: Test PASSES when solver hits iteration limit without converging!

**Fix**: Only accept Optimal or AlmostOptimal.

### 2. Regression Test Only Checked Optimal ⚠️

**File**: `solver-bench/src/regression.rs`
**Line**: 425

```rust
if res.status != SolveStatus::Optimal  // Missing AlmostOptimal
```

**Problem**: Correct to reject MaxIters, but should also accept AlmostOptimal.

**Fix**: ✅ Already fixed to `matches!(res.status, SolveStatus::Optimal | SolveStatus::AlmostOptimal)`

### 3. No Status Checks in Exp Cone Benchmarks ❌

**Files**:
- `solver-bench/examples/exp_cone_baseline.rs` - Didn't print status until today
- `solver-bench/examples/exp_cone_suite.rs` - Prints status but doesn't fail on MaxIters
- `solver-bench/examples/exp_cone_timing.rs` - No status check

**Problem**: Benchmarks run and report "success" even when solver fails to converge.

**Result**: We were celebrating "34% faster" performance when the solver was actually broken!

## Root Cause

The exp cone solver is fundamentally not converging:
- ✅ Computes correct objective value
- ❌ Never reaches convergence (residuals stay at ~0.6-0.9)
- ❌ Makes zero progress (μ doesn't decrease)
- ❌ Returns MaxIters status after hitting iteration limit

**When this started**: Unknown - possibly never worked correctly, or broke in an earlier commit

**Why we didn't know**: Tests accepted MaxIters as success

## What Needs To Be Fixed

### Immediate (Testing Infrastructure)

1. ✅ **Regression tests**: Fixed to require Optimal|AlmostOptimal
2. ❌ **Integration tests**: Fix all tests that accept MaxIters
3. ❌ **Benchmark harness**: Add status validation, fail loudly on non-optimal
4. ❌ **Test helper functions**: Create `assert_solved_optimally()` helper

### Next (Actually Fix The Solver)

5. ❌ **Debug exp cone convergence**: Why is solver stuck?
6. ❌ **Add exp cone regression problems**: Need actual exp cone problems in test suite
7. ❌ **Verify other cone types**: Are SOC/PSD also broken?

## Files Modified This Session

### Tests Fixed
- `solver-bench/src/regression.rs`: ✅ Now requires Optimal|AlmostOptimal

### Tests Still Broken
- `solver-core/tests/integration_tests.rs`: ❌ Accepts MaxIters
- `solver-bench/examples/exp_cone_*.rs`: ❌ No status validation

### New Test Files Created (For Debugging)
- `solver-bench/examples/exp_cone_debug.rs`: Debug harness with residual printing
- `_planning/v16/multi_cone_debug_log.md`: Investigation log

## Lessons Learned

1. **NEVER accept MaxIters as success** - it means the solver failed to converge
2. **Always check solve status** - solve() returning Ok() doesn't mean optimal solution
3. **Print residuals in debug mode** - would have caught this immediately
4. **Status must be part of test assertions** - not just logged
5. **Benchmarks must validate correctness** - fast broken code is worse than slow correct code

## Action Items

- [ ] Fix all integration tests to require Optimal|AlmostOptimal
- [ ] Add `assert_optimal!()` macro for tests
- [ ] Add pre-commit hook to grep for "MaxIters.*assert"
- [ ] Document testing requirements in CONTRIBUTING.md
- [ ] Run full regression suite and FIX any exp cone failures
- [ ] Investigate why exp cone solver is stuck (separate debugging session)

## Status Meanings (For Reference)

**PASS**:
- `Optimal`: Meets strict tolerances (1e-8)
- `AlmostOptimal`: Meets relaxed tolerances (1e-4)

**FAIL**:
- `MaxIters`: Hit iteration limit WITHOUT converging ❌
- `NumericalError`: Solver encountered numerical issues ❌
- `PrimalInfeasible`: Problem is infeasible ⚠️
- `DualInfeasible`: Problem is unbounded ⚠️

Only Optimal and AlmostOptimal should pass tests for feasible problems.
