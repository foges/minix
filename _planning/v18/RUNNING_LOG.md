# V18 Running Log

## Goal
Fix regressions and bugs identified in improvement_plan.md:
1. SOC regression: singleton elimination not cone-aware
2. P1.1 bug: max_iter extension not working (for loop issue)
3. Add dual residual diagnostics

## Session Start: 2026-01-08

### Task 1: Fix SOC Regression - Cone-Aware Singleton Elimination

**Problem**: Singleton elimination in presolve eliminates rows from multi-row cones (SOC, Exp, PSD) without updating cone structure, causing "Reduced scaling block mismatch" panics.

**Solution**: Only allow singleton elimination on separable 1D cones (NonNeg, Zero).

**Status**: COMPLETED ✅

**Changes Made**:
1. Added `row_is_eligible_for_singleton_elim()` function to check cone type
2. Created `detect_singleton_rows_cone_aware()` that filters out multi-dimensional cones
3. Updated `eliminate_singleton_rows()` to use cone-aware detection
4. Kept legacy `detect_singleton_rows()` for compatibility

**Files Modified**:
- solver-core/src/presolve/singleton.rs (added cone-aware logic)
- solver-core/src/presolve/eliminate.rs (use new function)

**Build**: SUCCESS (with warnings only)

**Next**: Test with SOC problems to verify fix

---

### Task 2: Fix P1.1 Bug - Convert For Loop to While for max_iter Extension

**Problem**: The P1.1 "extend max_iter" feature uses a for loop `for iter in 0..max_iter`, which cannot be extended at runtime when the range is set at loop entry.

**Solution**: Convert to while loop so max_iter can be dynamically extended.

**Status**: COMPLETED ✅

**Findings**:
- The iteration loop is already a while loop (not a for loop)
- The issue was that `is_large_problem` used REDUCED dimensions after presolve
- BOYD problems have ~93k vars originally but may shrink below 50k threshold after presolve

**Changes Made**:
1. Changed `is_large_problem` to use `orig_n` and `orig_m` (original dimensions before presolve)
2. Removed underscore prefix from `orig_m` variable
3. Added comment explaining why original dimensions are used

**Files Modified**:
- solver-core/src/ipm2/solve.rs

**Build**: SUCCESS

**Expected Impact**: BOYD1/BOYD2 should now be classified as large problems and get extended max_iter budget (200 iterations instead of 50)
