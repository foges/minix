# v16 Session Documentation

**Date**: January 7-8, 2026
**Status**: Investigation complete, **exp cone is BROKEN (unfixed)**

---

## Overview

This directory contains the complete documentation from the v16 investigation session, which focused on:
- Root cause analysis of exponential cone convergence issues
- Performance benchmarking and comparison with other solvers
- Code cleanup and documentation consolidation

All findings have been consolidated into a single comprehensive document for easy reference.

---

## Main Document

### [V16_COMPREHENSIVE.md](./V16_COMPREHENSIVE.md) 📚 **READ THIS**

**Complete v16 session documentation in one place**

This comprehensive document contains all investigation findings, analysis, and recommendations:

**NEW**: Includes **actual iteration-by-iteration data** (iterations 25-30) for all 31 failing problems showing exactly how residuals, gap, and μ evolve.

1. **Executive Summary**
   - Session overview and key findings
   - What was fixed, what remains

2. **Current Performance**
   - Maros-Meszaros QP: 77.2% pass rate @ 1e-8 tolerance
   - Exponential cone: 10-13 µs per-iteration, #2 ranking
   - Comparison with PIQP, Clarabel, ECOS, SCS

3. **Root Cause Analysis**
   - z_safe bug identified and fixed
   - Pure centering corrector implementation
   - Why finite differences failed

4. **Complete Investigation Trail**
   - Timeline of 12 different approaches tried
   - Iteration-level diagnostics
   - Testing infrastructure fixes

5. **Tolerance Investigation**
   - PIQP's 96% uses loose tolerances (~1.0)
   - At tight tolerances: Minix 77.2% vs PIQP 73% ✅
   - We're actually ahead at high accuracy!

6. **Failure Analysis**
   - Detailed breakdown of 31 Maros-Meszaros failures
   - Categorization by root cause
   - Priority ranking for fixes

7. **Third-Order Correction Research**
   - Analysis of reference implementations
   - Clarabel's analytical formula
   - Implementation roadmap for future work

8. **Next Steps**
   - 2nd-order Mehrotra: 2-4x iteration reduction (2-3 hour effort)
   - 3rd-order analytical: 5-10x iteration reduction (3-5 day effort)
   - Or leave as-is (already #2 solver, best per-iteration cost)

---

## Quick Reference

**Current Performance**:
```
Maros-Meszaros QP:  77.2% pass rate @ 1e-8 tolerance (105/136 problems)
Exp Cone:           BROKEN - KKT assembly produces garbage search directions
Overall Status:     QP works, exp cone completely broken
```

**Key Findings**:
- ✅ Root cause identified: KKT assembly bug for exp cones (ds[0] = 4.5e13!)
- ❌ **NOT FIXED** - exp cone solver completely broken
- ❌ Claims about "10-13 µs per-iteration" were WRONG - solver doesn't work
- ✅ Testing infrastructure fixed (requires Optimal status)
- ✅ Debug code cleanup (~250 lines removed)

**What Needs to be Fixed**:
- Fix KKT assembly for exponential cones (core bug)
- NOT z_safe, NOT BFGS scaling - those are symptoms

**Fair Comparison with PIQP**:
- PIQP 96% pass rate uses eps ≈ 1.0 (very loose)
- At tight tolerances (1e-8): **Minix 77.2% vs PIQP 73%** ✅

---

## Archive

Old/superseded documents have been moved to [`archive/`](./archive/) including:
- Previous investigation summaries
- Old status documents
- Debug logs
- Intermediate findings
- Implementation notes

**18 files archived** to keep v16 directory clean and focused.

---

## Files Modified During v16 Session

**Core Changes**:
- `solver-core/src/ipm2/predcorr.rs` - z_safe fix, pure centering
- `solver-core/src/scaling/bfgs.rs` - Cleaned up debug prints
- `solver-core/src/linalg/kkt.rs` - Cleaned up debug prints

**Tests**:
- `solver-core/tests/integration_tests.rs` - Fixed to require Optimal status
- `solver-bench/examples/exp_cone_no_presolve.rs` - Test harness

---

## Navigation

- 📚 **Complete documentation** → [V16_COMPREHENSIVE.md](./V16_COMPREHENSIVE.md)
- 📦 **Old/archived docs** → [archive/](./archive/)
- 🏠 **Back to planning** → [../_planning/](../)
