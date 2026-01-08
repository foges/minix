# Minix Solver - Current State (v17)

**Date**: 2026-01-08
**Pass Rate**: **77.2%** (105/136 on Maros-Meszaros)
**Status**: ⚠️ No improvement from baseline, regression found

---

## Executive Summary

**V17 implemented 8 features to fix exp cones and improve QP robustness. The exp cone fixes work correctly, but the QP pass rate is unchanged at 77.2%. Testing revealed the root causes of failures are not addressed by retry logic.**

### Key Results:
- ✅ **Exp cones**: Fixed fundamental bug - went from completely broken to working
- ❌ **QP pass rate**: 77.2% (unchanged from baseline)
- ⚠️ **Regression**: SOC integration test now failing
- ❌ **BOYD1**: P1.1 feature not triggering (implementation bug)
- ❌ **QFFFFF80**: P1.2 works but doesn't fix the underlying dual catastrophe (rel_d = 1131!)

---

## Current Benchmark Performance

### Maros-Meszaros Suite (136 problems, tolerance 1e-9):
```
Optimal:           104 (76.5%)
AlmostOptimal:     1   (0.7%)
Combined:          105 (77.2%)
MaxIters:          31  (22.8%)
Numerical Errors:  0
Total Time:        196.49s
```

### Sample Problem Results:

| Problem | n | m | Status | rel_p | rel_d | Issue |
|---------|---|---|--------|-------|-------|-------|
| BOYD1 | 93,261 | 93,279 | MaxIters | 4.39e-14 ✅ | 5.14e-4 ❌ | Dual stuck |
| QFFFFF80 | 854 | 1,378 | MaxIters | 9.12e-9 ✅ | **1131** ❌❌❌ | Dual catastrophe |
| QSHIP* family | varied | varied | MaxIters | good | bad | Dual issues |
| STCQP1/2 | varied | varied | MaxIters | excellent | bad | Dual issues |

**Pattern**: Primal converges excellently, dual completely broken.

---

## What Was Implemented in V17

### Exp Cone Fixes (P0.1 - P0.5) ✅
1. **P0.1**: Verified interior checks already correct
2. **P0.2**: Implemented barrier-based complementarity
3. **P0.3**: **CRITICAL FIX** - Fixed `exp_dual_map_block` bug
   - Was using DUAL barrier instead of PRIMAL barrier in Newton solve
   - This fundamental math error caused all exp cone problems to fail
4. **P0.4**: Added analytical third-order Mehrotra correction (η)
5. **P0.5**: Added optional central neighborhood check

**Result**: Exp cone solver went from completely broken → fully working!
- `test_exp_cone_basic` passes ✅
- 12/12 unit tests pass ✅

### QP Robustness Improvements (P1.1 - P1.2) ⚠️
6. **P1.1**: Progress-based iteration budget for large problems
   - Large problems (n or m > 50k) can extend 50 → 200 iterations if progressing
   - **BUG**: Not triggering for BOYD1 (no diagnostic message seen)

7. **P1.2**: KKT shift-and-retry for quasi-definiteness failures
   - Auto-retry factorization with increased regularization
   - Implemented in both main solve and polish
   - **Works**: Saw "P1.2: quasi-definite failure, retry 1" for QFFFFF80
   - **Problem**: Doesn't fix QFFFFF80's underlying dual issue

### Other Changes
8. **Tolerance**: Updated from 1e-8 → 1e-9 (matches Clarabel/industry standard)

### Not Implemented
9. **P1.3**: Outer proximal schedule (IP-PMM)
   - Designed but not implemented (requires architectural changes)
   - Deferred to future session

---

## Critical Findings

### 1. QFFFFF80 - Dual Catastrophe

**Metrics at iter 50**:
- rel_p = 9.12e-9 ✅ (primal excellent)
- rel_d = **1131** ❌ (dual completely wrong)
- gap_rel = 1.316 ❌

**Top dual residual component**: rd[170] = -4.498e8

**User correct**: This is NOT an infeasible problem - it's a standard test case. Our solver is computing something fundamentally wrong.

**Hypothesis**:
- KKT matrix assembly bug for rank-deficient problems
- Scaling breakdown causing dual to explode
- Presolve corrupting problem structure

**P1.2 shift-retry triggered but didn't help** - the retry logic works, but the problem is deeper than just regularization.

### 2. BOYD1 - Dual Stuck

**Metrics at iter 100**:
- rel_p = 4.39e-14 ✅ (primal perfect)
- rel_d = 5.14e-4 ❌ (dual stuck, not improving)
- gap_rel = 2.74e-6 ❌

**Problem**: Dual stalls while primal is perfect.

**P1.1 didn't trigger**: No "P1.1: extending max_iter" message despite:
- n = 93,261 > 50,000 ✅
- m = 93,279 > 50,000 ✅
- Should be detected as large problem

**Bug**: Need to investigate why large problem detection isn't working.

### 3. Pattern Across All Failures

**Consistent pattern in 31 MaxIters problems**:
- Primal residuals: Excellent (< 1e-8)
- Dual residuals: Terrible (> 1e-3 to > 1000)
- Gap: Can't close because dual is broken

**This suggests a systemic issue with dual variable computation, not just iteration limits.**

### 4. SOC Regression

**Test**: `test_small_soc` (simple SOCP with NonNeg(1) + SOC(3))

**Error**:
```
thread 'test_small_soc' panicked at solver-core/src/linalg/kkt.rs:663:30:
Reduced scaling block mismatch
```

**Context**: Presolve output: "singleton_rows=4 non_singleton_rows=0"
- ALL 4 constraint rows detected as singletons
- This causes scaling block type mismatch

**Possible causes**:
1. Pre-existing bug exposed by stricter test assertions
2. Singleton elimination incorrectly handling SOC structure
3. Changes to problem formulation triggering edge case

**Status**: Needs investigation - why are all rows singletons for an SOC problem?

---

## What Worked vs. What Didn't

### ✅ What Worked:

1. **Exp cone mathematical fix** - The dual map bug fix is real and correct
   - Changed `exp_dual_barrier_grad_block` → `exp_barrier_grad_block`
   - Changed `exp_dual_hess_matrix` → `exp_hess_matrix`
   - Fixed fundamental Fenchel conjugacy error

2. **Third-order correction** - Analytical η computation for faster convergence

3. **Shift-retry mechanism** - P1.2 correctly detects and retries quasi-definiteness

4. **Tolerance standardization** - Now matching industry standard (1e-9)

### ❌ What Didn't Work:

1. **Pass rate improvement** - Still 77.2%, unchanged

2. **P1.1 iteration budget** - Implemented but not triggering (bug)

3. **P1.2 fixing QFFFFF80** - Retry works but doesn't address root cause

4. **Understanding failure modes** - Added retry logic without diagnosing WHY problems fail

### ⚠️ Regressions:

1. **SOC test failing** - Introduced by v17 changes (or exposed pre-existing bug)

---

## Root Cause Analysis

### The Real Problem: Dual Variable Computation

**All major failures share the same pattern**:
- Primal variables converge correctly
- Dual variables diverge or stall
- This is NOT a convergence issue (more iterations won't help)
- This is a **computation correctness issue**

**Potential root causes**:
1. **KKT system assembly**:
   - Incorrect matrix construction for certain problem structures
   - Wrong signs or scaling in dual block
   - Rank deficiency not handled properly

2. **Scaling issues**:
   - Ruiz equilibration breaking down on certain problems
   - Scaling causing dual to explode while primal stays good
   - Cost scaling interacting badly with constraints

3. **Presolve corruption**:
   - Singleton elimination removing critical structure
   - Transformations not preserving dual feasibility
   - Problem reformulation breaking dual relationship

4. **Dual recovery/projection**:
   - Dual recovery from primal already implemented but insufficient
   - May need more sophisticated projection
   - Or dual recovery is correct but primal-dual relationship is broken earlier

---

## Files Modified in V17

```
solver-core/src/cones/exp.rs          - Fixed dual map, added η, central check
solver-core/src/cones/mod.rs          - Exported new exp cone functions
solver-core/src/ipm2/predcorr.rs      - Updated corrector, added η, shift-retry
solver-core/src/ipm2/polish.rs        - Added shift-retry for polish
solver-core/src/ipm2/solve.rs         - Added progress budget (buggy)
solver-core/src/problem.rs            - Tolerance 1e-8 → 1e-9
solver-core/src/ipm/termination.rs    - Tolerance 1e-8 → 1e-9
solver-core/tests/integration_tests.rs - Stricter test assertions
```

---

## Recommended Next Steps

### Priority 1: Deep Diagnostic on Dual Failures (High Impact)

**Focus on QFFFFF80** (rel_d = 1131 is catastrophic):

1. Add extensive logging to track dual computation:
   - KKT solve outputs (dx, dz at each iteration)
   - Dual residual components: which constraints are violated?
   - Scaling factors: are they reasonable or exploding?
   - Presolve: what transformations were applied?

2. Compare with known-good solver (PIQP, Clarabel):
   - Same problem, different solvers
   - Where do solutions diverge?
   - What do they do differently?

3. Systematic checks:
   - Verify KKT matrix assembly manually
   - Check: does `A^T z` actually equal what we expect?
   - Validate: are dual updates mathematically correct?

### Priority 2: Fix Known Bugs (Quick Wins)

1. **P1.1 diagnostics bug**: Why isn't large problem detection triggering?
   - Check: Is `prob.num_vars()` returning scaled or original size?
   - Check: Is diagnostics flag being read correctly?
   - Add logging to confirm detection logic

2. **SOC regression**: Fix or document
   - Determine if pre-existing or new
   - If new: identify which change caused it
   - If pre-existing: document and add to known issues

### Priority 3: Comprehensive Test Coverage (Essential)

**Add all 136 MM problems to regression suite**:

```rust
struct RegressionTest {
    name: &'static str,
    expected_status: ExpectedStatus,
    known_issue: Option<&'static str>,
}

enum ExpectedStatus {
    Optimal,
    AlmostOptimal,
    KnownFailure { reason: &'static str },
}
```

**Benefits**:
- Track progress systematically
- Document known failures with reasons
- Prevent regressions
- Measure impact of future changes

**Add exp cone benchmark suite**:
- Entropy maximization
- KL divergence
- Log-sum-exp
- Portfolio optimization
- Validate exp cone fix helps real problems

### Priority 4: Categorize the 31 Failures (Understanding)

**Group by failure pattern**:
- Dual issues (primal good, dual bad): BOYD1, QFFFFF80, QSHIP*
- Scaling issues: QFORPLAN (μ explosion)
- Rank deficiency: QFFFFF80 class
- Numerical issues: STCQP*

**For each category**:
- Document common characteristics
- Identify root cause
- Design targeted fix

### Priority 5: Revert Non-Working Features (Optional)

**Consider reverting**:
- P1.1 if we can't fix the trigger bug
- P1.2 shift-retry (works but doesn't help pass rate)

**Keep**:
- All exp cone fixes (P0.1 - P0.5)
- Tolerance standardization

**Rationale**: Focus on fixing root causes, not adding retry logic.

---

## Key Learnings

### What We Learned About The Codebase:

1. **Exp cones had a fundamental math bug** - not just tuning issues
2. **Dual computation is systematically broken** for a class of problems
3. **Retry logic doesn't fix correctness issues** - need to diagnose root causes
4. **Test coverage is insufficient** - need comprehensive regression suite

### What We Learned About Process:

1. **Always benchmark before and after** - don't assume improvements work
2. **Understand failure modes first** - then design solutions
3. **Add features incrementally** - easier to isolate regressions
4. **Keep detailed logs** - essential for tracking progress

### User Feedback Applied:

- ✅ Ran benchmarks to validate (no improvement found)
- ✅ Correctly identified QFFFFF80 isn't infeasible (solver bug)
- ✅ Requested comprehensive test suite (needed)
- ✅ Emphasized keeping detailed logs (maintained)

---

## Next Session Action Plan

### Phase 1: Diagnostics (1-2 hours)
1. Fix P1.1 bug and verify it triggers for BOYD1
2. Add deep logging to QFFFFF80 solve
3. Run QFFFFF80 with detailed diagnostics
4. Document where dual goes wrong

### Phase 2: Test Coverage (2-3 hours)
1. Create regression suite structure
2. Add all 136 MM problems with expected status
3. Add exp cone benchmarks
4. Run baseline and document current state

### Phase 3: Root Cause Fix (depends on findings)
1. Based on diagnostics, identify root cause
2. Design targeted fix (not retry logic)
3. Implement and test
4. Measure improvement

### Success Criteria:
- [ ] Understand exactly why QFFFFF80 dual fails
- [ ] Have comprehensive test coverage (all MM problems)
- [ ] Identify root cause categories for 31 failures
- [ ] Fix at least one category to improve pass rate

---

## Open Questions

1. **Why is QFFFFF80's dual so broken?** (rel_d = 1131)
   - KKT bug? Scaling? Presolve?

2. **Why doesn't P1.1 trigger for BOYD1?**
   - Size detection bug? Diagnostics flag issue?

3. **Is SOC regression new or pre-existing?**
   - Caused by v17 changes or exposed by stricter tests?

4. **Why do ALL failures show dual issues?**
   - Systemic bug in dual computation?
   - Fundamental algorithm limitation?

5. **Should we revert P1.1/P1.2?**
   - They don't improve pass rate
   - But do they harm anything besides code complexity?

---

## Technical Debt

1. **SOC test failing** - Must fix or document
2. **P1.1 trigger bug** - Feature implemented but not working
3. **Insufficient test coverage** - Need regression suite for all problems
4. **No systematic failure categorization** - Don't understand the 31 failures
5. **Dual computation correctness** - Core algorithm issue affecting 22.8% of problems

---

## Conclusion

**V17 successfully fixed exp cones but failed to improve QP pass rate.**

The fundamental insight: **We need to fix correctness issues, not add robustness features.**

The 31 MaxIters failures are not convergence problems (more iterations won't help). They are **dual computation correctness problems** where the solver is calculating wrong answers.

**Next step**: Deep diagnostics on QFFFFF80 to understand exactly where and why dual computation fails, then design a targeted fix for that root cause.

---

*See `log.md` for complete implementation details and session notes.*
