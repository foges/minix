# Detailed Failure Analysis (v15, 50 iterations)

## Summary

**Pass rate:** 108/136 (79.4%)
**Failing:** 28 problems (all MaxIters)
**Iteration limit:** 50 (enforced)

## All Failing Problems with Metrics

| # | Problem | rel_p | rel_d | gap_rel | Category |
|---|---------|-------|-------|---------|----------|
| 1 | Q25FV47 | 1.3e-16 | 4.2e-1 | 1.7e-1 | Dual slow |
| 2 | QADLITTL | 1.1e-16 | 1.1e-2 | 3.0e-3 | Dual slow |
| 3 | QBANDM | 1.7e-16 | 1.0e-1 | 7.1e-1 | Dual slow |
| 4 | QBEACONF | 9.2e-15 | 1.0e0 | 1.0e-1 | Dual stall |
| 5 | QBORE3D | 5.8e-13 | 1.8e0 | 8.9e-8 | Dual stall, gap tiny |
| 6 | QBRANDY | 1.6e-16 | 9.8e-2 | 1.2e-1 | Dual slow |
| 7 | QCAPRI | 1.9e-11 | 2.8e-1 | 5.8e-2 | Dual slow |
| 8 | QE226 | 1.5e-16 | 3.2e-2 | 7.6e-2 | Dual slow |
| 9 | **QFFFFF80** | **9.1e-9** | **768** | **1.3** | **Dual explosion** |
| 10 | **QFORPLAN** | **5.5e-14** | **0.86** | **0.96** | **HSDE explosion** |
| 11 | QGFRDXPN | 2.5e-9 | 3.7e-3 | 2.2e-12 | Gap tiny, dual OK |
| 12 | QPCBOEI1 | 1.2e-9 | 2.6e-3 | 3.4e-15 | Gap tiny, dual OK |
| 13 | QPILOTNO | 2.4e-10 | 2.8e-5 | 1.4e-1 | Primal great, gap large |
| 14 | QSCAGR25 | 1.4e-16 | 2.0e0 | 2.8e-1 | Dual stall |
| 15 | QSCAGR7 | 1.2e-16 | 1.7e-1 | 1.2e-1 | Dual slow |
| 16 | QSCFXM1 | 2.1e-9 | 8.3e-1 | 1.5e-1 | Dual slow |
| 17 | QSCFXM2 | 2.4e-9 | 9.0e-1 | 2.1e-1 | Dual slow |
| 18 | QSCFXM3 | 3.6e-9 | 9.1e-1 | 2.0e-1 | Dual slow |
| 19 | QSCORPIO | 6.5e-17 | 1.4e-1 | 1.2e-3 | Dual slow, gap tiny |
| 20 | QSCRS8 | 1.2e-16 | 8.4e-1 | 3.1e-13 | Dual slow, gap tiny |
| 21 | QSHARE1B | 2.7e-16 | 3.1e-4 | 1.8e-2 | Gap moderate |
| 22 | **QSHIP04L** | **4.2e-10** | **0.27** | **3.7e-3** | **Step collapse** |
| 23 | **QSHIP04S** | **4.2e-10** | **0.54** | **0.16** | **Step collapse** |
| 24 | **QSHIP08L** | **1.2e-9** | **0.38** | **8.2e-3** | **Step collapse** |
| 25 | **QSHIP08S** | **1.9e-9** | **0.87** | **0.37** | **Step collapse** |
| 26 | **QSHIP12L** | **5.8e-10** | **3.3e-2** | **9.5e-2** | **Step collapse** |
| 27 | **QSHIP12S** | **7.2e-10** | **5.3e-2** | **0.73** | **Step collapse** |
| 28 | STCQP1 | 2.7e-16 | 11.6 | 5.0e-1 | Dual explosion (moderate) |

## Category Breakdown

### Category 1: HSDE Scalar Explosion (1 problem)

**QFORPLAN** - The homogenization scaling completely breaks down:
- At 50 iters: rel_p=5.5e-14, rel_d=0.86, gap_rel=0.96
- μ explodes to 1e26+ when run longer
- τ/κ runaway despite normalization guards

**Root cause:** HSDE scaling ray divergence on this problem's structure

**What's been tried:**
- ✅ Merit function check (100x μ growth rejection)
- ✅ τ normalization (bounds: 0.2, 5.0)
- ⚠️ τ+κ normalization (interferes with infeasibility detection)

**Next steps:**
1. Detect HSDE scaling ray escape early and abort to direct IPM
2. Implement non-HSDE QP solver path for known-feasible problems
3. Alternative normalization: normalize on μ decomposition balance


### Category 2: Dual Explosion (2 problems)

**QFFFFF80** - Single constraint drives dual to infinity:
- rel_p=9.1e-9 (excellent), rel_d=768, gap_rel=1.3
- Constraint 170: A^Tz = -2.67e8 (one dual component dominates)

**QGFRDXPN** - Similar pattern but milder:
- rel_p=2.5e-9, rel_d=3.7e-3, gap_rel=2.2e-12 (gap essentially 0!)
- Dual is reasonable, but gap is numerically zero - why doesn't it terminate?

**Root cause:** Near-degenerate constraint matrix, non-unique dual

**What's been tried:**
- ✅ Skip polish when rel_d > 100x tolerance
- ✅ LP dual polish fallback

**Next steps:**
1. For gap_rel < 1e-10 but dual bad: accept solution anyway (primal is good)
2. Constraint preconditioning: identify and rescale problematic rows
3. Dual-only recovery via regularized least squares on active set


### Category 3: Step Size Collapse (6 problems - QSHIP family)

All 6 QSHIP problems show identical pathology:
- Primal converges to 1e-9 or better
- Dual stalls at 0.03–0.87
- Step size α collapses to 1e-40+
- μ frozen in range 1e-13 to 1e-14

**QSHIP04L:** rel_p=4.2e-10, rel_d=0.27, gap_rel=3.7e-3
**QSHIP04S:** rel_p=4.2e-10, rel_d=0.54, gap_rel=0.16
**QSHIP08L:** rel_p=1.2e-9, rel_d=0.38, gap_rel=8.2e-3
**QSHIP08S:** rel_p=1.9e-9, rel_d=0.87, gap_rel=0.37
**QSHIP12L:** rel_p=5.8e-10, rel_d=3.3e-2, gap_rel=9.5e-2
**QSHIP12S:** rel_p=7.2e-10, rel_d=5.3e-2, gap_rel=0.73

**Root cause:** Extreme KKT ill-conditioning; step direction computed but cone geometry forces α→0

**What's been tried:**
- ✅ Anti-stall σ cap (ablation shows: no impact)
- ✅ Regularization bumps

**Next steps:**
1. Active-set polish: fix primal near-zero slacks, solve dual-only KKT
2. Crossover to simplex when step size < 1e-8
3. Iterative refinement with quad precision for KKT solve


### Category 4: Dual Slow Convergence (15 problems)

General pattern: primal converges well, dual stuck at 0.01–2.0

**Problems:**
- Q25FV47: rel_p=1.3e-16, rel_d=0.42
- QADLITTL, QBANDM, QBRANDY, QCAPRI, QE226: dual 0.01–0.1
- QBEACONF, QBORE3D, QSCAGR25: dual 1.0–2.0 (severe stall)
- QSCFXM1/2/3: dual 0.83–0.91 (very slow)
- QSCAGR7, QSCORPIO: dual 0.14–0.17
- QPCBOEI1, QSCRS8: gap ~ 0 but still running

**Root cause:** Dual updates ineffective due to problem structure (degeneracy, ill-conditioning)

**Next steps:**
1. Identify problems where gap_rel < 1e-12 and terminate early
2. Increase centering (σ) when dual stalls but primal good
3. Dual step length modification (independent α for dual)


### Category 5: Other (4 problems)

**QPILOTNO:** rel_p=2.4e-10, rel_d=2.8e-5, gap_rel=0.14
- Both residuals excellent but gap large - likely objective scaling issue

**QSHARE1B:** rel_p=2.7e-16, rel_d=3.1e-4, gap_rel=0.018
- Almost there, needs a few more iterations

**STCQP1:** rel_p=2.7e-16, rel_d=11.6, gap_rel=0.50
- Similar to QFFFFF80 but milder dual explosion

**Next steps:**
1. QSHARE1B: increase iter limit to 60 for borderline cases
2. QPILOTNO: check objective scaling (c^Tx vs x^TPx imbalance)


## High-Level Recommendations

### Quick wins (likely 3–5 more problems):
1. **Early termination:** Accept solution when gap_rel < 1e-10 regardless of dual
   - Would fix: QGFRDXPN, QPCBOEI1, QSCRS8, possibly QSCORPIO
2. **Borderline iter bump:** Increase to 60 iters for problems "almost there"
   - Would fix: QSHARE1B, possibly QE226
3. **Gap scaling check:** Normalize gap by max(|obj_p|, |obj_d|, 1)
   - Would fix: QPILOTNO if gap formula is too strict

### Medium effort (5–10 more problems):
4. **Active-set dual recovery:** When primal excellent but dual bad, solve dual-only LS
   - Would fix: QFFFFF80, QSHIP family (6 problems), STCQP1
5. **HSDE escape detection:** Switch to direct IPM when τ/κ diverge
   - Would fix: QFORPLAN

### Fundamental changes (would need major work):
6. **Crossover to simplex** when IPM stalls
7. **Presolve/constraint analysis** to identify degenerate structure
8. **Quad precision KKT** for extremely ill-conditioned problems

## Implementation Priority

**Phase 1 (target: 113–116 / 136 = 83–85%)**
- Early termination on gap_rel < 1e-10
- Borderline iter bump to 60
- Gap scaling normalization

**Phase 2 (target: 120–125 / 136 = 88–92%)**
- Active-set dual recovery
- HSDE escape detection

**Phase 3 (research)**
- Simplex crossover
- Constraint preconditioning
