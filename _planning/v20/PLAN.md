# V20 Plan: Compensated Summation + Condition-Aware Acceptance

## Executive Summary

**Goal**: Enhance BOYD-class problem handling with:
1. **Compensated (Kahan) summation** for A^T*z to detect catastrophic cancellation
2. **Condition-aware acceptance** as fallback when hitting numerical precision limits

**Context**: V19 added condition number diagnostics and confirmed BOYD has κ(K) growing from 1e12 → 3e16. Crossover polish already exists (from v8) and is being applied. The remaining issue is distinguishing between:
- Algorithmic failure (solver should do better)
- Numerical precision limit (double-precision floor)

## User Requirements

From detailed guidance:
- "BOYD1/BOYD2 are exactly the kind of instances where a 'plain' primal–dual IPM can end up primal-feasible + small gap, but the reported dual feasibility stalls..."
- "Compensated (Kahan) for A^T * z lets you measure: am I stuck because of cancellation error, or because the multipliers really are correct?"
- "If cancellation_factor > 100, you know the floor is numerical precision, not a solver bug"
- "Condition-aware acceptance: If κ(K) > 1e13 AND primal+gap converged AND dual stalled for 5 iterations, report AlmostOptimal (or NumericalLimit)"

## What We Already Have

✅ **From v8 patch (already applied)**:
- `polish_nonneg_active_set()` for active-set crossover polish
- Early polish trigger (line 590 in solve.rs)
- Final polish after MaxIters (line 1068 in solve.rs)
- Multiple polish variants (primal_projection, primal_and_dual, lp_dual, dual_only)

✅ **From v19**:
- Condition number estimation: `estimate_condition_number()`
- Warnings when κ > 1e12 (ill-conditioned) or κ > 1e15 (severely ill-conditioned)
- Dynamic regularization in Polish+dual stall mode

✅ **From metrics.rs**:
- `diagnose_dual_residual()` function (lines 157-247)
- Already decomposes r_d = P*x + A^T*z + q
- BUT: uses standard summation (lines 191-200), not compensated

## What We Need to Implement

### Phase 1: Compensated Summation for A^T*z (2-3 hours)

**Concept**: Kahan (compensated) summation tracks the accumulated rounding error:
```rust
let mut sum = 0.0;
let mut c = 0.0;  // Running compensation for lost low-order bits
for &val in values {
    let y = val - c;
    let t = sum + y;
    c = (t - sum) - y;  // (t - sum) recovers high-order bits, subtract y = error
    sum = t;
}
```

**Implementation**:

1. **New function in metrics.rs**: `compute_atz_with_kahan()`
   ```rust
   pub struct AtzResult {
       pub atz: Vec<f64>,                    // Final A^T*z values
       pub atz_magnitude: Vec<f64>,          // Sum of |val * z| (cancellation-free)
       pub cancellation_factor: Vec<f64>,    // atz_magnitude / |atz|
       pub max_cancellation: f64,            // max(cancellation_factor)
   }

   pub fn compute_atz_with_kahan(a: &CsMat<f64>, z: &[f64]) -> AtzResult {
       let n = a.cols();
       let mut atz = vec![0.0; n];
       let mut atz_compensation = vec![0.0; n];
       let mut atz_magnitude = vec![0.0; n];

       for col in 0..n {
           if let Some(col_view) = a.outer_view(col) {
               for (row, &val) in col_view.iter() {
                   let contrib = val * z[row];

                   // Kahan summation for atz
                   let y = contrib - atz_compensation[col];
                   let t = atz[col] + y;
                   atz_compensation[col] = (t - atz[col]) - y;
                   atz[col] = t;

                   // Magnitude sum (no cancellation)
                   atz_magnitude[col] += contrib.abs();
               }
           }
       }

       // Compute cancellation factor for each variable
       let mut cancellation_factor = vec![1.0; n];
       for i in 0..n {
           if atz[i].abs() > 1e-20 {
               cancellation_factor[i] = atz_magnitude[i] / atz[i].abs();
           }
       }

       let max_cancellation = cancellation_factor.iter()
           .copied()
           .fold(0.0f64, f64::max);

       AtzResult { atz, atz_magnitude, cancellation_factor, max_cancellation }
   }
   ```

2. **Update diagnose_dual_residual()** to use Kahan summation and report cancellation:
   ```rust
   let atz_result = compute_atz_with_kahan(a, z_bar);

   eprintln!("Cancellation Analysis:");
   eprintln!("  Max cancellation factor: {:.1f}x", atz_result.max_cancellation);
   if atz_result.max_cancellation > 100.0 {
       eprintln!("  ⚠️  SEVERE CANCELLATION: dual residual floor is due to numerical precision");
   }

   // Show top 5 variables with worst cancellation
   let mut indexed: Vec<(usize, f64)> = atz_result.cancellation_factor.iter()
       .enumerate()
       .map(|(i, &v)| (i, v))
       .collect();
   indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

   eprintln!("  Top 5 variables with cancellation:");
   for i in 0..5.min(indexed.len()) {
       let idx = indexed[i].0;
       let factor = indexed[i].1;
       if factor > 10.0 {
           eprintln!("    x[{}]: {:.1f}x cancellation (atz={:.3e}, magnitude={:.3e})",
               idx, factor, atz_result.atz[idx], atz_result.atz_magnitude[idx]);
       }
   }
   ```

3. **Optional**: Update `compute_unscaled_metrics()` to also use Kahan summation for production A^T*z computation
   - This would make the metrics more accurate but may have performance cost
   - Could gate behind `MINIX_USE_KAHAN` env var

### Phase 2: Condition-Aware Acceptance (1-2 hours)

**Trigger conditions** (from user's guidance):
- `rel_p <= tol_feas` (primal feasible)
- `gap_rel <= tol_gap` (gap converged)
- `rel_d > tol_feas` (dual stuck)
- `κ(K) > 1e13` (ill-conditioned)
- Dual stalled for 5+ iterations (no improvement)
- Optional: `max_cancellation > 100` (cancellation-dominated)

**Implementation in solve.rs**:

1. **New SolveStatus variant** in `problem.rs`:
   ```rust
   pub enum SolveStatus {
       Optimal,
       MaxIters,
       NumericalLimit,  // NEW: hit numerical precision floor
       // ... existing variants
   }
   ```

2. **Condition-aware acceptance logic** (in solve.rs, before returning MaxIters):
   ```rust
   // Check for numerical precision limit
   if status == SolveStatus::MaxIters {
       let primal_ok = metrics.rel_p <= criteria.tol_feas;
       let gap_ok = metrics.gap_rel <= criteria.tol_gap_rel;
       let dual_stuck = metrics.rel_d > criteria.tol_feas;

       // Check condition number from last iteration
       let cond_number = kkt.estimate_condition_number().unwrap_or(1.0);
       let ill_conditioned = cond_number > 1e13;

       // Check dual stall (no improvement in last 5 iterations)
       let dual_stalled = stall.dual_stall_count >= 5;

       if primal_ok && gap_ok && dual_stuck && ill_conditioned && dual_stalled {
           if diag.enabled {
               eprintln!("\nCondition-aware acceptance:");
               eprintln!("  rel_p={:.3e} (✓), gap_rel={:.3e} (✓), rel_d={:.3e} (✗)",
                   metrics.rel_p, metrics.gap_rel, metrics.rel_d);
               eprintln!("  κ(K)={:.3e} (ill-conditioned)", cond_number);
               eprintln!("  Dual stalled for {} iterations", stall.dual_stall_count);
               eprintln!("  → Accepting as NumericalLimit (double-precision floor)");
           }
           status = SolveStatus::NumericalLimit;
       }
   }
   ```

3. **Enhanced diagnostics with cancellation check**:
   ```rust
   // If condition-aware acceptance triggered, diagnose cancellation
   if status == SolveStatus::NumericalLimit && diag.enabled {
       let atz_result = compute_atz_with_kahan(&orig_prob_bounds.A, &z);
       if atz_result.max_cancellation > 100.0 {
           eprintln!("  Cancellation factor: {:.1f}x (confirms numerical precision limit)",
               atz_result.max_cancellation);
       }

       diagnose_dual_residual(
           &orig_prob_bounds.A,
           orig_prob_bounds.P.as_ref(),
           &orig_prob_bounds.q,
           &x, &z, &rd_orig,
           "NumericalLimit"
       );
   }
   ```

### Phase 3: Testing & Validation (1 hour)

1. **Run BOYD1 with new diagnostics**:
   ```bash
   MINIX_DIAGNOSTICS=1 cargo run --release -p solver-bench -- qps BOYD1
   ```

   Expected output:
   - Condition number: ~3e16 (severely ill-conditioned)
   - Cancellation factor: >100x (confirms numerical floor)
   - Status: NumericalLimit (not MaxIters)

2. **Run BOYD2 for confirmation**:
   ```bash
   MINIX_DIAGNOSTICS=1 cargo run --release -p solver-bench -- qps BOYD2
   ```

3. **Regression suite** (ensure no status changes for good problems):
   ```bash
   MINIX_REGRESSION_MAX_ITER=200 cargo test -p solver-bench regression_suite_smoke --release
   ```

   - All 108 passing problems should still report Optimal (NOT NumericalLimit)
   - BOYD1/BOYD2 should report NumericalLimit (NOT MaxIters)

### Phase 4: Documentation (30 min)

1. Update `_planning/v20/RUNNING_LOG.md` with implementation notes
2. Update `_planning/v20/SUMMARY.md` with:
   - Compensated summation for cancellation detection
   - Condition-aware acceptance for numerical precision limits
   - BOYD status changed from MaxIters to NumericalLimit

## File Changes

**solver-core/src/problem.rs**:
- Add `SolveStatus::NumericalLimit` variant

**solver-core/src/ipm2/metrics.rs**:
- Add `struct AtzResult`
- Add `pub fn compute_atz_with_kahan()`
- Update `diagnose_dual_residual()` to use Kahan and report cancellation

**solver-core/src/ipm2/solve.rs**:
- Add condition-aware acceptance logic before returning MaxIters
- Call cancellation diagnostics when NumericalLimit triggered

**Documentation**:
- `_planning/v20/PLAN.md` (this file)
- `_planning/v20/RUNNING_LOG.md`
- `_planning/v20/SUMMARY.md`

## Success Criteria

✅ Kahan summation implemented for A^T*z computation
✅ Cancellation factor reported in diagnose_dual_residual()
✅ SolveStatus::NumericalLimit variant added
✅ Condition-aware acceptance logic triggers for BOYD1/BOYD2
✅ BOYD shows max_cancellation > 100x (confirms numerical floor)
✅ Regression suite: 108 Optimal unchanged, BOYD → NumericalLimit
✅ Diagnostics clearly distinguish algorithmic vs numerical issues

## Expected Impact

**BOYD1/BOYD2**:
- Status: MaxIters → **NumericalLimit** (more honest reporting)
- Diagnostics: Show max_cancellation ~1000x (cancellation-dominated)
- User clarity: "This is a numerical precision limit, not a solver bug"

**Other problems**:
- No status changes (only triggers for ill-conditioned + dual-stuck cases)
- Enhanced diagnostics available via MINIX_DIAGNOSTICS=1

**Research value**:
- Quantifies when dual residual floor is due to cancellation vs conditioning
- Provides concrete evidence for "this problem is fundamentally hard for double-precision IPM"

## Time Estimate

- Phase 1 (Kahan summation): 2-3 hours
- Phase 2 (Condition-aware acceptance): 1-2 hours
- Phase 3 (Testing): 1 hour
- Phase 4 (Documentation): 30 min

**Total**: 4.5-6.5 hours

## Notes

- Kahan summation overhead is negligible (only called in diagnostics by default)
- Condition-aware acceptance is conservative (only triggers with multiple checks)
- This does NOT change the actual solver algorithm, only status reporting and diagnostics
- BOYD-class problems remain in expected-to-fail set (v18), but now with clearer reason
