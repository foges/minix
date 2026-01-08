# Plan: Tackling Remaining 20 Maros-Meszaros Failures

## Current Status
- **116/136 (85.3%) optimal** after recent fixes
- **11 MaxIters**: QSHIP family + similar dual-stuck problems
- **9 NumericalError**: Problems with overflow or unconverged gaps

## Problem Categories

### Category A: QSHIP Family (6 problems)
**Problems**: QSHIP04S, QSHIP04L, QSHIP08S, QSHIP08L, QSHIP12S, QSHIP12L

**Symptoms**:
- Excellent primal (1e-9 to 1e-15)
- Terrible dual (0.5-0.9) - stuck and degrading
- Moderate gap (1e-3 to 0.16)
- Polish fixes dual but destroys primal

**Root Cause Hypothesis**:
The QSHIP problems have very sparse P matrices (56 triplets for 1458 vars). Some variables have no P coupling and minimal A coverage, making dual residual structurally difficult.

**Experiments**:
1. **Analyze problem structure**: Check which variables have sparse A columns
2. **Adaptive tolerance**: Per-variable dual tolerance based on A column density
3. **Augmented Lagrangian**: Add penalty for dual infeasibility
4. **Direct dual solve**: Solve A^T z = Px + q as least-squares with cone projection

### Category B: Large Gap Problems (5+ problems)
**Problems**: QPILOTNO, QSCAGR25, Q25FV47, others with gap > 10%

**Symptoms**:
- Gap doesn't converge (10-100%+ of objective)
- May have numerical overflow (gap=4e22)
- Both primal and dual stuck

**Root Cause Hypothesis**:
Poor problem conditioning or scaling issues. The IPM step directions don't make progress on the gap.

**Experiments**:
1. **Better Ruiz scaling**: More iterations or row/column equilibration variants
2. **Scaled termination**: Check convergence in scaled space before unscaling
3. **Aggressive centering recovery**: When gap stalls, increase σ temporarily
4. **Multi-start**: Try different initial points

### Category C: Numerical Overflow (2-3 problems)
**Symptoms**:
- Values become 1e20+
- gap, obj_p, obj_d overflow

**Root Cause Hypothesis**:
τ/κ or state variables drift to extreme values without proper recovery.

**Experiments**:
1. **Better τ/κ normalization**: More aggressive bounds on homogeneous variables
2. **Early overflow detection**: Catch large values before they explode
3. **Fallback to direct mode**: Disable HSDE when τ drifts

## Experiment Priority

### High Priority (Quick Wins)
1. **Analyze QSHIP structure** - understand the structural issue
2. **Better overflow detection** - catch early and terminate gracefully
3. **Relaxed gap tolerance for almost-optimal** - currently 10x, try 100x for specific patterns

### Medium Priority (Moderate Effort)
4. **Direct dual solve** - implement efficient A^T z = target solver
5. **Scaled termination** - check convergence in scaled coordinates
6. **Augmented Lagrangian term** - penalize dual infeasibility

### Low Priority (Major Changes)
7. **Improved Ruiz scaling** - block-aware or adaptive iterations
8. **Multi-start strategies** - different initial points
9. **Crossover to simplex** - for final basis identification

## Metrics to Track

For each experiment:
- Number of problems solved (out of 20)
- Regression on currently-passing problems
- Iteration count changes
- Time impact

## Success Criteria

- **Stretch goal**: 130/136 (95%+) optimal
- **Realistic goal**: 125/136 (92%) optimal
- **Minimum**: No regressions on current 116 passing
