# Multi-Cone Exponential Cone Debug Log

## Issue

When expanding exp cone benchmarks beyond single-cone problems, multi-cone variants fail with `NumericalError`.

**Failing problems:**
- trivial-multi-2 (2 exp cones)
- trivial-multi-5 (5 exp cones)
- cvxpy-multi-2 (2 exp cones + 4 zero constraints)

**Working problems:**
- trivial-1 (1 exp cone)
- cvxpy-3 (1 exp cone + 2 zero constraints)

## Investigation Plan

1. Examine problem formulations in `exp_cone_suite.rs`
2. Run simplest failing case (trivial-multi-2) with verbose output
3. Check if problem data is correctly formed (matrix dimensions, cone specs)
4. Test with single iteration to see initialization
5. Identify root cause

## Test 1: Examine Problem Formulations

### Root Cause Found! ✅

**Problem**: The multi-cone variants are incorrectly constructed.

**Current (WRONG) approach**:
```rust
let mut prob = bench_problems::trivial();  // 3 rows, 1 var, 1 exp cone
prob.cones = vec![ConeSpec::Exp { count: 2 }];  // Says 2 cones (6 rows needed)
```

**Issue**:
- Trivial problem has A matrix: 3 rows × 1 col, b vector: 3 elements
- Changing cone spec to `count: 2` tells solver to expect 6 rows (2 exp cones × 3 rows each)
- **Dimension mismatch** → NumericalError

**Fix needed**:
- Must properly construct multi-cone problems with correct matrix dimensions
- Cannot just change cone count on existing single-cone problem

## Test 2: Create Proper Multi-Cone Problems

Fixed multi-cone problem formulations with correct matrix dimensions.

Result: No more NumericalError! ✅

**But discovered critical issue**: Even trivial-1 and cvxpy-3 return status `MaxIters`, not `Optimal`. ❌

## Test 3: Check Baseline Benchmark Status

Modified exp_cone_baseline.rs to print solve status.

**CRITICAL FINDING**: ❌❌❌

```
Problem                 Iters    Time (ms)    Objective    µs/iter       Status
trivial                    50         0.46       0.0000        9.2      MaxIter
cvxpy                     200         1.20       3.7183        6.0      MaxIter
```

**The exp cone solver is NOT converging!**

Both problems hit MaxIters without reaching optimal solution. The previous "successful" results were misleading - we were just running until iteration limit.

## Root Cause Analysis

Possible reasons exp cone problems don't converge:

1. **Incorrect exp cone projection** - BFGS scaling might be wrong
2. **Incorrect exp cone barrier** - gradient/hessian formulas
3. **Incorrect cone membership test** - problems declared "solved" prematurely
4. **Wrong problem formulation** - trivial/cvxpy problems might be infeasible
5. **Tolerance too tight** - default 1e-8 might be unreachable for exp cones
6. **Initialization issue** - starting point causes divergence

## Test 4: Check Residuals and Convergence

Ran trivial problem with max_iter=50 and both tight (1e-8) and loose (1e-4) tolerances.

**CRITICAL**: Solver makes **ZERO** progress! ❌❌❌

```
=== Testing Trivial Problem (max_iter=50, tol=1e-8) ===
Status: MaxIters, Iters: 50, Obj: 0.0000e0
primal_res: 6.0000e-1, dual_res: 7.5000e-1, gap: 1.0000e0, mu: 3.7500e0

=== Testing Trivial Problem (max_iter=50, tol=1e-4) ===
Status: MaxIters, Iters: 50, Obj: 0.0000e0
primal_res: 6.0000e-1, dual_res: 7.5000e-1, gap: 1.0000e-1, mu: 3.7500e0
```

**Analysis**:
- Residuals are HUGE (0.6, 0.75, 1.0) after 50 iterations
- μ is still 3.75 (not decreasing towards 0)
- Same residuals for both tolerances → NOT a tolerance issue
- **The solver is stuck and not converging at all**

## Hypothesis: Rank-3 BFGS Broke Exp Cone Solver

Timeline:
1. v15: Exp cone solver worked (passed regression suite)
2. v16: Implemented rank-3 BFGS scaling
3. Now: Exp cone solver completely broken

**Test**: Disable rank-3 BFGS and test with rank-4 fallback...

## Test 5: Disable Rank-3 BFGS (Test with Rank-4 Only)

Disabled rank-3 BFGS, reverted to rank-4 only.

**Result**: Still broken! ❌

```
=== Testing Trivial Problem (max_iter=50, tol=1e-8) ===
Status: MaxIters, Iters: 50, Obj: 0.0000e0
primal_res: 6.0000e-1, dual_res: 7.5000e-1, gap: 1.0000e0, mu: 3.7500e0
```

**Conclusion**: Rank-3 BFGS is NOT the problem. Something else broke the solver.

## Test 6: Check Exp Cone Initialization

Reverted exp cone initialization to original values `[-1.051_383, 0.556_409, 1.258_967]`.

**Result**: Still broken! ❌

## Test 7: Test with Higher max_iter

Tested with max_iter=1000 to see if solver eventually converges.

**Result**: SOLVER IS COMPLETELY STUCK! ❌❌❌

```
=== Testing Trivial Problem (max_iter=1000, tol=1e-8) ===
Status: MaxIters, Iters: 1000, Obj: 0.0000e0
primal_res: 5.9773e-1, dual_res: 8.8199e-1, gap: 1.0000e0, mu: 3.2500e0
```

**After 1000 iterations**:
- Residuals are UNCHANGED (still ~0.6-0.9)
- μ is UNCHANGED (still ~3.25)
- The solver makes ZERO progress

## Test 8: Check Clean Code (Before This Session)

Hypothesis: Maybe I broke it earlier in this session?

Tested clean main branch (before any of my changes today).

**Result**: ALSO BROKEN! ❌

Even on the clean code (commit c4b3ff3), the solver hits MaxIter with huge residuals. This means the exp cone solver has been broken for a while, NOT from my changes today.

## Test 9: Check Historical Commits

Checked commit cca213a ("Fix exponential cone bug: Implement proper dual barrier").

**Result**: ALSO BROKEN! ❌

Even at the commit that supposedly "fixed" exp cones, the solver still hits MaxIter.

## ROOT CAUSE ANALYSIS

The exp cone solver is **fundamentally broken** - it's not converging at all, just stuck.

**What the "fix" actually fixed**:
- Before cca213a: Solver diverged to obj=-1e25 (completely wrong)
- After cca213a: Solver gets correct objective (0.0, 3.72) but doesn't converge

**Current status**:
- ✅ Computes correct objective value
- ❌ Never reaches Optimal status (residuals stay huge)
- ❌ Makes zero progress after initial iterations

**Next**: Need to check regression suite to see if ANY exp cone problems actually reach Optimal status...

## Test 10: Fix All Tests to Require Optimal Status

**Action**: Systematically fixed all tests to reject MaxIters status.

### Files Fixed

1. **solver-bench/src/regression.rs** ✅
   - Changed line 425-427 from `status != Optimal` to `!matches!(status, Optimal | AlmostOptimal)`
   - Added comment: "CRITICAL: Require Optimal or AlmostOptimal status"

2. **solver-core/tests/integration_tests.rs** ✅
   - Fixed 5 tests that accepted MaxIters:
     - `test_simple_lp` (line 59-64)
     - `test_lp_with_inequality` (line 121-124)
     - `test_simple_qp` (line 186-189)
     - `test_nonneg_cone` (line 234-237)
     - `test_small_soc` (line 294-297) - kept NumericalError for incomplete SOC support
   - Removed conditional checks (if status == Optimal) - now ALL checks run
   - Added descriptive error messages

3. **Documentation** ✅
   - Created `_planning/v16/TESTING_REQUIREMENTS.md` - Comprehensive testing guidelines
   - Created `_planning/v16/testing_failures_found.md` - What we found and why it matters
   - Updated `_planning/v16/multi_cone_debug_log.md` - This file

### Result

**All tests now properly validate solver convergence!**

Running `cargo test` will NOW FAIL if:
- Exp cone problems hit MaxIters (which they currently do)
- Any solver returns non-convergent status
- Residuals are large even with "correct" objective

This is GOOD - we want tests to fail loudly when the solver is broken.

## Summary

### What Was Wrong

1. **Solver**: Exp cone completely broken (residuals stuck at ~0.6-0.9, never converges)
2. **Tests**: Accepted MaxIters as passing, hid the problem
3. **Benchmarks**: Reported "performance improvements" on broken solver
4. **Root cause**: Unknown when it broke (possibly never worked right)

### What We Fixed

1. ✅ Regression tests now require Optimal|AlmostOptimal
2. ✅ Integration tests now reject MaxIters
3. ✅ Documentation of testing requirements
4. ✅ Re-enabled rank-3 BFGS (wasn't the problem)
5. ✅ Reverted initialization changes (wasn't the problem)

### What's Still Broken

1. ❌ Exp cone solver doesn't converge (THE BIG ISSUE)
2. ❌ Benchmark examples don't validate status
3. ❌ No exp cone problems in regression suite

### Next Steps

1. **Run cargo test** - Will fail on exp cone tests (expected!)
2. **Debug exp cone convergence** - Why is solver stuck?
3. **Add exp cone regression problems** - Need test coverage
4. **Fix the actual solver** - Make it converge

## Lessons Learned

1. **Testing revealed a systemic failure** - Not catching non-convergence is a HUGE gap
2. **Status field is critical** - Must always check it
3. **Benchmarks must validate correctness** - Fast broken code means nothing
4. **Tests are our safety net** - When they're wrong, everything fails silently
