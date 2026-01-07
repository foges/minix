# Diagnostic Findings (Phase 1 Complete)

## Summary

Ran diagnostics on the three main failure categories. All failures are **genuine** - not tolerance issues, not "almost there." The solver is mathematically stuck.

## Finding 1: QFORPLAN - Primal-Dual Breakdown (Not HSDE!)

**Test:** Added μ decomposition logging and tested with/without HSDE

**Results:**
```
iter 8:  μ=4.8e11   μ_sz=6.3e13    μ_tk=3.1e11    ratio=202x
iter 10: μ=3.4e15   μ_sz=4.5e17    μ_tk=2.4e15    ratio=189x
iter 14: μ=2.0e25   μ_sz=2.7e27    μ_tk=9.7e22    ratio=27,400x
```

**Also tested with direct mode (MINIX_DIRECT_MODE=1):**
- Still explodes: μ → 1e24, rd[31] → -4.1e21
- **Conclusion: NOT an HSDE problem!**

**Root cause:**
- Both s·z and τκ are exploding
- **s·z component dominates** by 100-27,000x
- This is **fundamental primal-dual complementarity breakdown**
- The problem itself may be ill-posed or near-infeasible

**Implication:** No amount of HSDE normalization will fix this. The KKT system is producing directions that increase complementarity gap rather than decrease it.

---

## Finding 2: QSHIP04S - Catastrophic KKT Directions

**Test:** Added step blocking diagnostics to show what limits α

**Results at iters 46-49 (completely stuck, α → 1e-53):**

**Primal blocking:**
```
idx=280:  s=5.7e-13  ds=-0.045     (s at machine precision, large negative step)
idx=1740: s=8.5e-56  ds=-8.7e-16   (s way beyond machine limits!)
```

**Dual blocking:**
```
idx=278:  z=0.07     dz=-3.3e10    (reasonable z, ENORMOUS negative direction!)
idx=279:  z=0.09     dz=-4.2e9     (huge direction)
idx=1740: z=0.0017   dz=-3.5e7     (huge direction)
```

**Root cause:**
- The KKT solve produces mathematically "valid" directions
- But dz is **10 billion** times larger than z itself!
- Even α=1e-10 would make z negative → infeasible
- The KKT matrix is so ill-conditioned that the solution is numerically garbage

**Why this happens:**
1. Primal converges → s → 0 at some indices
2. As s → 0, the KKT matrix becomes more ill-conditioned
3. Regularization helps but not enough
4. The computed (dy, dz) is wildly wrong
5. Step size forced to ~0 to avoid violating z ≥ 0

**Implication:**
- Can't fix with better step size rules (direction itself is bad)
- Can't fix with more regularization (already at recovery level)
- **Need fundamentally different approach when primal converged but dual stuck**

---

## Finding 3: "Dual Slow" Problems Are Actually Stuck

**Test:** Ran QBANDM, QBRANDY, QE226 with 50 vs 100 iterations

**Results:**
| Problem | rel_d @ 50 iters | rel_d @ 100 iters | Change |
|---------|------------------|-------------------|--------|
| QBANDM  | 1.004e-1        | 1.004e-1          | 0.000e0 ✗ |
| QBRANDY | 9.832e-2        | 9.832e-2          | 0.000e0 ✗ |
| QE226   | 3.203e-2        | 3.203e-2          | 0.000e0 ✗ |

**Conclusion:**
- **ZERO improvement** from doubling iterations
- These aren't "slow" - they're **frozen**
- rel_d identical to machine precision (1e-16 relative difference)

**Implication:** More iterations won't help. Need algorithmic change.

---

## What These Findings Mean for Fixes

### ❌ Won't Work:
1. **HSDE normalization improvements** → QFORPLAN explodes even in direct mode
2. **More iterations** → Problems literally frozen (rel_d doesn't budge)
3. **Better step size rules** → QSHIP has ~0 step size because direction is garbage
4. **More regularization** → Already at maximum recovery levels

### ✅ High-Leverage Fixes:

#### 1. Active-Set Dual Recovery (Highest Priority)
**For:** QSHIP family (6), QFFFFF80, possibly 10-15 "dual slow" problems

**Approach:** When primal excellent (rel_p < 1e-6) but dual bad (rel_d > 1):
```rust
// Fix x where s < 1e-6 (active constraints)
// Solve SPD system for dual only:
//   minimize ||Px + q + A'z||^2 + rho ||z||^2
// This avoids the saddle-point KKT system entirely
// Uses Cholesky (always stable) instead of LDL (can fail)
```

**Expected impact:** Fix 7-15 problems → **85-91% pass rate**

#### 2. Early Termination with Certificates
**For:** QFORPLAN, other "exploding" problems

**Approach:**
- Detect when μ grows 1000x while residuals flat
- Check if iterate proves infeasibility (κ/τ → ∞ with theory)
- Return "Infeasible" or "NumericalError" instead of MaxIters

**Expected impact:** Cleaner failure modes, no change to pass rate

#### 3. Presolve: Identify Degenerate Constraints
**For:** All stuck problems

**Approach:**
- Before IPM, compute row condition numbers of A
- Identify nearly-parallel constraints
- Scale or combine them

**Expected impact:** 2-5 more problems → **87-94% pass rate**

---

## Recommended Implementation Order

**Week 2-3: Active-set dual recovery**
- Should fix 7-15 problems with excellent primal but bad dual
- Most problems fall into this category
- Algorithmic breakthrough, not just tuning

**Week 4: Testing and analysis**
- Re-run full suite
- Analyze remaining failures
- Document what's genuinely "too hard"

**Not pursuing:**
- HSDE improvements (doesn't help QFORPLAN)
- Iteration limit increases (proven ineffective)
- Step size tuning (direction is the problem, not step size)

---

## Files Added

**Code changes:**
- `solver-core/src/ipm2/solve.rs`: μ decomposition logging when μ > 1e10
- `solver-core/src/ipm2/predcorr.rs`: Step blocking diagnostics when α < 1e-8

**Enable with:**
```bash
MINIX_DIAGNOSTICS=1 cargo run --release -p solver-bench -- maros-meszaros --problem QFORPLAN
```

---

## Next Steps

1. ✅ **Phase 1 complete:** Diagnostics show root causes
2. ⏭️  **Phase 2 starting:** Implement active-set dual recovery
3. **Phase 3:** Test on all 28 failures, measure impact
