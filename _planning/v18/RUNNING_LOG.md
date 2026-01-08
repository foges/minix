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
