# V15 Session 2: Constraint Conditioning Experiment

## What Was Done

### 1. Implemented Constraint Conditioning
- Created `solver-core/src/presolve/condition.rs` (~220 lines)
- Detection: nearly-parallel rows (cosine sim > 0.999), extreme coefficient ratios
- Fix attempt: geometric mean row scaling before Ruiz equilibration
- Integration in `solve.rs` with `enable_conditioning` setting

### 2. Testing & Results
- Full benchmark: **104/136 Optimal (76.5%)** vs baseline **108/136 (79.4%)**
- **Verdict: Harmful** - decreased pass rate by 2.9% (4 problems)
- Individual problem analysis showed primal residuals degraded significantly

### 3. Root Cause Analysis
**Why row scaling failed:**
- Interferes with Ruiz equilibration (Ruiz expects raw problem structure)
- Doesn't fix parallel rows (rank deficiency still exists)
- Disrupts dual variable scaling without addressing root causes
- Breaking existing convergence patterns for marginal/no dual improvement

### 4. Documentation
- `conditioning_results.md` - full analysis of failure
- `presolve_implementation.md` - updated with DISABLED status
- Code cleaned up: disabled by default, diagnostics behind verbose flag

## Key Insights

### Problem Categories Confirmed

From diagnostic session and conditioning analysis:

| Category | Count | Root Cause | Example |
|----------|-------|------------|---------|
| Dual explosion | ~6 | KKT produces huge A^T*z | QFFFFF80 |
| Dual directions huge | ~5 | Step direction dz >> z | QSHIP04S |
| Dual stuck | ~15 | Zero dual progress across iters | QFORPLAN |
| μ explosion | ~2 | s·z grows unbounded | QFORPLAN |
| Unknown | ~5 | Need investigation | Various |

### What Doesn't Work

1. ❌ **Row scaling before Ruiz** - interferes with equilibration
2. ❌ **Lowering tolerances** - against rigorous standards principle
3. ❌ **Scaling parallel rows** - doesn't fix rank deficiency

### What Shows Promise

1. ✅ **Dual recovery** (already implemented)
   - QSHIP04S: 0.54 → 0.22 rel_d improvement
   - Accepts excellent primal, recovers dual via least squares
   - Currently triggers when: `rel_p < 1e-6 AND rel_d > 0.1 AND iter >= 20`

2. ✅ **μ decomposition logging** (already implemented)
   - Diagnoses s·z vs τκ explosions
   - Enables targeted fixes per problem type

3. ✅ **Step blocking diagnostics** (already implemented)
   - Identifies which variable blocks step
   - Reveals dual direction pathologies

## Recommended Next Steps

### Option A: Extend Dual Recovery (Most Promising)

**Current status:** Implemented but conservative thresholds

**Improvements to try:**
1. **Earlier trigger** - currently waits until iter 20, try iter 10-15
2. **Multiple attempts** - try recovery every 5-10 iters if dual stuck
3. **Relaxed dual threshold** - currently `rel_d > 0.1`, try `rel_d > 0.05`
4. **Primal threshold** - currently `rel_p < 1e-6`, try `rel_p < 1e-5` for early recovery

**Expected impact:** +5-10 problems (reach ~113-118/136, 83-87%)

**Implementation:**
```rust
// In solve.rs, around line 392-460
// Current: iter >= 20 && rel_p < 1e-6 && rel_d > 0.1
// Try: iter >= 10 && rel_p < 1e-5 && rel_d > 0.05
// Multiple attempts: also trigger if stuck for N iters
```

### Option B: Better Barrier Parameter Control

**Target:** μ explosion problems (QFORPLAN)

**Current issue:** μ = s·z/m grows unbounded despite centering

**Potential fixes:**
1. **Adaptive centering** - detect μ growth, increase σ (more aggressive centering)
2. **Mehrotra heuristic tuning** - adjust formula when μ >> τκ
3. **Primal-dual splitting** - take different step sizes for primal/dual

**Expected impact:** +2-3 problems

**Risk:** High - barrier parameter control is delicate, easy to break convergence

### Option C: Robust KKT Solving

**Target:** Problems where KKT solver produces pathological directions

**Approaches:**
1. **Regularization** - add δI to KKT diagonal when near-singular
2. **Iterative refinement** - detect large residuals, refine KKT solution
3. **Direction validation** - detect explosion (||Δ|| > threshold), reject step

**Expected impact:** +3-5 problems

**Complexity:** Medium-high - requires careful tuning of thresholds

### Option D: Redundant Constraint Elimination (True Presolve)

**Target:** Problems with truly parallel rows (QGFRDXPN: 211 pairs!)

**Approach:**
1. Detect parallel rows: cosine similarity > 0.9999, check b values
2. Eliminate redundant rows entirely (not just scale)
3. Postsolve: recover eliminated duals from active constraints

**Expected impact:** +2-4 problems

**Complexity:** High - need robust detection, postsolve mapping

## Recommended Action Plan

### Phase 1: Low-Hanging Fruit (1-2 hours)
**Extend dual recovery with relaxed thresholds**

1. Change trigger in `solve.rs`:
   - Earlier: `iter >= 10` (from 20)
   - Relaxed: `rel_p < 1e-5` (from 1e-6), `rel_d > 0.05` (from 0.1)
2. Add periodic retry: every 10 iters if dual stuck
3. Benchmark and measure impact

**Expected:** ~110-113/136 (81-83%)

### Phase 2: Medium Effort (3-5 hours)
**Add KKT direction validation**

1. Detect pathological directions in `predcorr.rs`:
   ```rust
   let dx_norm = dx.iter().map(|v| v*v).sum::<f64>().sqrt();
   if dx_norm > 1e8 * x_norm {
       // Apply damping or reject
   }
   ```
2. Add regularization option: `KKT + δI` when ill-conditioned
3. Benchmark and measure

**Expected:** ~113-118/136 (83-87%)

### Phase 3: High Effort (1-2 days)
**Implement redundant constraint elimination**

1. Add to presolve pipeline (after singleton elimination)
2. True QR-based detection of linear dependence
3. Postsolve dual recovery
4. Benchmark and measure

**Expected:** ~118-122/136 (87-90%)

## Files Modified This Session

### New Files
- `solver-core/src/presolve/condition.rs` (~220 lines) - kept for analysis
- `_planning/v15/conditioning_results.md` - failure analysis
- `_planning/v15/session2_summary.md` - this document

### Modified Files
- `solver-core/src/ipm2/solve.rs` - conditioning integration (disabled)
- `solver-core/src/presolve/mod.rs` - added condition module
- `solver-core/src/problem.rs` - added `enable_conditioning` field
- `_planning/v15/presolve_implementation.md` - marked as disabled

## Current State

**Status:** Baseline restored (108/136, 79.4%)

**Active features:**
- ✅ Dual recovery (iter >= 20, conservative thresholds)
- ✅ μ decomposition diagnostics
- ✅ Step blocking diagnostics
- ❌ Constraint conditioning (disabled - harmful)

**Recommended next:** Extend dual recovery (Option A, Phase 1)
