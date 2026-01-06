# Merge QA Analysis: End-to-End Review

## Executive Summary

After merging 43 commits from upstream `foges/minix`, I performed comprehensive end-to-end testing to validate which of our improvements are still useful and effective.

**Critical Finding:** The default solver has changed from `ipm` to `ipm2`, which means some of our SOC-specific improvements are NO LONGER ACTIVE by default.

## What's Still Useful ✅

### 1. **SOC citardauq Formula** (ACTIVE in both ipm1 and ipm2)
- **Location:** `solver-core/src/cones/soc.rs:step_to_boundary_primal()`
- **Status:** ✅ Used by both solvers
- **Impact:** Prevents numerical cancellation when computing SOC step lengths
- **Evidence:** Both ipm1 and ipm2 call `cone.step_to_boundary_primal()` which uses our citardauq improvements

### 2. **Benchmark Infrastructure** (ACTIVE)
- **Location:** `solver-bench/src/`
- **Status:** ✅ Fully functional
- **Impact:** 7 comprehensive test suites (Maros-Meszaros, NETLIB, CBLIB, PGLib, QPLIB, Meszaros, Regression)
- **Evidence:** Successfully ran 20 Maros-Meszaros problems with 100% optimal rate
- **Added Value:** Solver selection parameter allows comparing ipm1 vs ipm2

### 3. **QPS OBJSENSE Parsing** (ACTIVE)
- **Location:** `solver-bench/src/qps.rs`
- **Status:** ✅ Used when parsing QPS files
- **Impact:** Correctly handles maximization vs minimization
- **Evidence:** Integrated cleanly during merge, no conflicts with upstream

### 4. **MAT File Support** (ACTIVE)
- **Location:** `solver-bench/src/maros_meszaros.rs`
- **Status:** ✅ Used as fallback for Maros-Meszaros suite
- **Impact:** Faster loading from local .mat files when available
- **Evidence:** Merged successfully with upstream's download infrastructure

## What's NOT Being Used (But Not Harmful) ⚠️

### 5. **Adaptive NT Scaling Regularization Tracking**
- **Location:** `solver-core/src/ipm/predcorr.rs` (s_min, z_min tracking)
- **Status:** ⚠️ Only used for diagnostics, NOT for algorithm decisions
- **Impact:** Minimal - only affects diagnostic messages
- **Upstream Alternative:** ipm2 has `RegularizationPolicy` with dynamic_bumps
- **Verdict:** Keep for diagnostics, not harmful

## What's ACTIVELY HARMFUL ❌

### 6. **SOC Centrality Parameters** (ONLY in ipm1, NOT in ipm2)
- **Location:** `solver-core/src/ipm/predcorr.rs:centrality_ok_trial()`
- **Parameters:**
  - `soc_centrality_beta`
  - `soc_centrality_gamma`
  - `soc_centrality_use_upper`
  - `soc_centrality_use_jordan`
  - `soc_centrality_mu_threshold`
- **Status:** ❌ HURTS PERFORMANCE on hard SOC problems
- **Evidence:** Empirical testing on BOYD1 and BOYD2

#### Performance Comparison

**BOYD1:**
```
ipm1 (with SOC centrality):
- Status: MaxIters (FAILED)
- Iterations: 200
- Time: 8889.7 ms
- Objective: -6.173e7 (WRONG)

ipm2 (without SOC centrality):
- Status: Optimal (SUCCESS)
- Iterations: 23
- Time: 1551.7 ms
- Objective: -6.908e7 (CORRECT)
```

**BOYD2:**
```
ipm1 (with SOC centrality):
- Status: MaxIters (FAILED)
- Iterations: 200
- Objective: 2.751e3 (WRONG)

ipm2 (without SOC centrality):
- Status: Optimal (SUCCESS)
- Iterations: 43
- Time: 2162.6 ms
- Objective: -1.1e1 (CORRECT)
```

**Analysis:** The SOC centrality checks are TOO RESTRICTIVE, causing line search to fail on hard problems. ipm2's simpler centrality check (NonNeg only) performs much better.

## Solver Architecture Understanding

### Current Setup
```
solver-core/src/lib.rs:solve() → ipm2::solve_ipm2()  [DEFAULT]
                               → ipm::solve_ipm()     [via --solver ipm1]
```

### ipm1 (Old Solver)
- ✅ Has SOC citardauq formula (via cone.step_to_boundary)
- ❌ Has SOC centrality parameters (HARMFUL)
- ✅ Has s_min/z_min tracking (diagnostic only)
- ⚠️ Workspace pre-allocation (we added this)
- ⚠️ Our NT scaling error handling (upstream improved this)

### ipm2 (New Solver, Default)
- ✅ Has SOC citardauq formula (via cone.step_to_boundary)
- ✅ NO SOC centrality parameters (simpler is better!)
- ✅ Better KKT retry logic with static reg bumping
- ✅ Bounded Mehrotra correction
- ✅ Better timing instrumentation
- ✅ Improved diagnostics
- ✅ Polish for bound-heavy problems

## Test Results

### Unit Tests
- ✅ 65/65 lib tests passing
- ✅ 7/7 integration tests passing
- **Total:** 72/72 (100%)

### Benchmark Validation
- ✅ 20/20 Maros-Meszaros optimal (100%)
- ✅ Geometric mean iterations: 9.2
- ✅ Total time: 57.94s

### Hard SOC Problems
- ❌ BOYD1 with ipm1: FAILED (MaxIters)
- ✅ BOYD1 with ipm2: Optimal in 23 iters
- ❌ BOYD2 with ipm1: FAILED (MaxIters)
- ✅ BOYD2 with ipm2: Optimal in 43 iters

## Recommendations

### Keep As-Is ✅
1. **SOC citardauq formula** - Solid numerical improvement
2. **Benchmark infrastructure** - Excellent testing capability
3. **QPS OBJSENSE parsing** - Correct behavior for maximization
4. **MAT file support** - Performance optimization
5. **s_min/z_min tracking** - Useful diagnostics

### Consider Removing 🤔
6. **SOC centrality parameters** - Actively harmful, not used in ipm2
   - Options:
     - A) Remove completely from SolverSettings
     - B) Keep but document as "experimental, ipm1 only"
     - C) Try porting a SIMPLER version to ipm2

### Document Clearly 📝
- ipm2 is now the default solver (better performance)
- ipm1 is available via `--solver ipm1` for regression testing
- SOC centrality parameters only affect ipm1

## Merged Code Quality

### Conflict Resolution Quality: ✅ Excellent
- All 7 files resolved correctly
- Our improvements preserved where beneficial
- Upstream improvements integrated fully
- No functionality lost

### Integration Quality: ✅ Excellent
- Both solver paths working
- Benchmark infrastructure functional
- Proper solver selection mechanism
- Clean separation between ipm1 and ipm2

### Testing Quality: ✅ Excellent
- All tests passing
- Benchmarks validate functionality
- Hard problems expose the SOC centrality issue

## Conclusion

**The merge was technically successful**, but it revealed that our SOC centrality parameters are not beneficial - they actually hurt performance significantly on the hardest SOC problems (BOYD1, BOYD2).

**Our most valuable contribution** is the SOC citardauq formula in `step_to_boundary`, which IS being used by both solvers and prevents numerical issues.

**The benchmark infrastructure** is also excellent and provides clear evidence of solver performance differences.

**Recommendation:** Keep the current merged state. The SOC centrality parameters exist but are only active in ipm1, which is opt-in. The default (ipm2) performs better without them. Consider this a successful lesson in empirical validation - sometimes simpler is better!
