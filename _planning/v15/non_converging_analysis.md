# Non-Converging Problems Analysis (v15)

## Executive Summary

After v15 improvements, **108/136 (79.4%)** problems converge optimally on Maros-Meszaros.
**28 problems** do not converge within 200 iterations:
- 27 hit MaxIters
- 1 hits NumericalError (QCAPRI)

## Problem Categories

### Category 1: HSDE Scalar Explosion (1 problem)

**Problem:** QFORPLAN

**Symptoms:**
- μ explodes to 1e26+
- rel_p and rel_d stuck at 0.1 (10%)
- gap_rel stuck at 0.96

**Diagnostics:**
```
iter 43 mu=1.470e26 rel_p=1.000e-1 rel_d=1.000e-1 gap_rel=9.609e-1
iter 49 mu=1.120e24 rel_p=1.000e-1 rel_d=1.000e-1 gap_rel=9.609e-1
Final: rel_p=5.461e-14, rel_d=8.564e-1  (note: different from iter metrics!)
```

**Root Cause:** The HSDE homogenization scalars (τ, κ) are exploding. The solver is following a "scaling ray" rather than converging to optimality.

**Attempted Fixes:**
1. ✅ Merit function check (100x μ growth rejection) - prevents some runaway but doesn't fix QFORPLAN
2. ✅ τ+κ normalization - tested but interferes with infeasibility detection
3. ❌ τ-only normalization (0.2, 5.0 bounds) - already in place, not sufficient

**Status:** Not solved. Likely needs fundamental HSDE redesign (e.g., different normalization strategy, or dropping HSDE for direct IPM).

---

### Category 2: Dual Explosion with Good Primal (2 problems)

**Problems:** QFFFFF80, QGFRDXPN

**Symptoms:**
- Primal excellent (rel_p ~ 1e-9)
- Dual explodes (rel_d ~ 700)
- Gap large (gap_rel ~ 1.3)

**QFFFFF80 Diagnostics:**
```
iter 49 mu=4.938e-7 rel_p=9.120e-9 rel_d=7.682e2 gap_rel=1.316e0
Final: rel_p=9.120e-9, rel_d=7.682e2
```

**Root Cause:** The dual variable z for constraint 170 is exploding (A^Tz = -2.671e8). This single constraint is driving the entire dual residual. The problem has a near-singular constraint structure.

**Attempted Fixes:**
1. ✅ Skip active-set polish when dual severely bad (>100x tolerance)
2. ✅ Try LP dual polish instead - doesn't help (gap too large to trigger)
3. ❌ Early polish - rejected because primal degrades

**Status:** Not solved. May need constraint analysis to identify and handle degenerate constraints.

---

### Category 3: Step Size Collapse (QSHIP family - 6 problems)

**Problems:** QSHIP04L, QSHIP04S, QSHIP08L, QSHIP08S, QSHIP12L, QSHIP12S

**Symptoms:**
- Primal excellent (rel_p ~ 1e-10)
- Dual stuck at ~0.5
- Step size α collapses to 1e-40 and below
- μ frozen at 5e-14

**QSHIP04S Diagnostics:**
```
iter 43 mu=5.195e-14 alpha=1.378e-41 rel_p=4.240e-10 rel_d=5.398e-1
iter 49 mu=5.195e-14 alpha=1.378e-53 rel_p=4.240e-10 rel_d=5.398e-1
```

**Root Cause:** The KKT system becomes extremely ill-conditioned. The step direction is computed but the step size is limited by cone boundaries to essentially zero.

**Attempted Fixes:**
1. ✅ Anti-stall σ cap (0.1 for dual stall) - ablation shows no impact
2. ✅ Regularization bumps - already applied
3. ❌ Polish - can't trigger (gap not close enough)

**Status:** Not solved. Fundamental numerical conditioning issue.

---

### Category 4: Numerical Error (1 problem)

**Problem:** QCAPRI

**Symptoms:**
- Hits NumericalError at iteration 78
- Usually due to non-finite values or factorization failure

**Status:** Need deeper investigation. Likely KKT factorization breakdown.

---

### Category 5: General MaxIters - Mixed Pathologies (18 problems)

**Problems:** Q25FV47, QADLITTL, QBANDM, QBEACONF, QBORE3D, QBRANDY, QE226, QPCBOEI1, QPILOTNO, QSCAGR7, QSCAGR25, QSCFXM1, QSCFXM2, QSCFXM3, QSCORPIO, QSCRS8, QSHARE1B, STCQP1

These problems have varying pathologies but generally:
- Primal converges (rel_p < 1e-6)
- Dual stuck (rel_d ~ 0.1 to 1.0)
- Gap may or may not be close

---

## Summary of v15 Fixes Applied

| Fix | Purpose | Impact |
|-----|---------|--------|
| Merit function check | Reject μ explosion steps | Prevents some HSDE runaway |
| τ+κ normalization | Keep HSDE scalars bounded | Tested, interferes with infeasibility detection |
| Polish when dual bad | Skip fragile active-set polish | Prevents quasi-definite errors |
| LP dual polish fallback | More robust for bad dual | Limited impact (gap threshold) |

## What Would Help

### For QFORPLAN (HSDE explosion):
- Direct IPM without homogenization
- Alternative HSDE normalization (normalize τ only, or use merit function on HSDE scalars)
- Infeasibility detection bypass for this problem type

### For QFFFFF80 (dual explosion):
- Constraint preconditioning/scaling
- Identify and handle degenerate constraints
- Alternative dual update formulas

### For QSHIP family (step collapse):
- Better KKT regularization for ill-conditioned matrices
- Active set methods (crossover to simplex)
- Iterative refinement with higher precision

### For all:
- Hybrid approach: switch to different algorithm when IPM stalls
- More aggressive presolve to simplify problem structure
- Warm-start from OSQP/Clarabel solution

## Benchmark Context

These 28 problems are known to be difficult:
- MOSEK and Clarabel also struggle with some of them
- Many have condition numbers > 1e12
- Some have degenerate/near-singular constraint matrices

The current 79.4% pass rate is reasonable for a research solver without:
- Crossover to simplex
- Active set refinement
- Presolve (beyond Ruiz scaling)
