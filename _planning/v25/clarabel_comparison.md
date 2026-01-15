# Clarabel vs Minix: Detailed Implementation Analysis

## Executive Summary

Minix achieves 87.5% (119/136) on Maros-Meszaros vs Clarabel's ~100%. The 9 MaxIters failures are:
BOYD2, QBEACONF, QBORE3D, QSC205, QSCFXM2, QSCFXM3, QSHARE1B, QSIERRA, YAO

Key differences identified through code analysis:

## 1. Centering Parameter (σ) Computation - CRITICAL DIFFERENCE

### Clarabel (simpler, more robust)
```rust
fn centering_parameter(&self, α: T) -> T {
    T::powi(T::one() - α, 3)  // σ = (1 - α)³
}
```
- Only depends on affine step length α
- Small α (blocked step) → σ ≈ 1 (heavy centering)
- Large α (free step) → σ ≈ 0 (aggressive toward boundary)
- **Geometrically motivated**: step length captures how "blocked" we are

### Minix (classic Mehrotra)
```rust
let sigma = (mu_aff / mu).powi(3);  // σ = (μ_aff/μ)³
```
- Depends on ratio of predicted to current duality gap
- More state-dependent, can be unstable when μ_aff is poorly estimated
- On ill-conditioned problems, μ_aff can be garbage

**Impact**: Clarabel's formula is more robust because it only depends on geometry (step length), not on potentially ill-conditioned μ estimates. This is likely a major factor in convergence differences.

## 2. Default Parameters

| Parameter | Clarabel | Minix | Notes |
|-----------|----------|-------|-------|
| max_iter | **200** | **100** | **2x difference - easy fix** |
| tol_gap_abs/rel | 1e-8 | 1e-8 | Same |
| tol_feas | 1e-8 | 1e-8 | Same |
| static_reg_constant | 1e-8 | 1e-8 | Same |
| static_reg_proportional | ε² ≈ 5e-32 | N/A | Minix lacks this |
| IR reltol | 1e-13 | N/A | Fixed iters in Minix |
| IR abstol | 1e-12 | N/A | Fixed iters in Minix |
| IR max_iter | 10 | 10 | Same |
| IR stop_ratio | 5.0 | N/A | Early stop if < 5x improvement |
| linesearch_backtrack | 0.8 | N/A | Minix uses different LS |
| min_terminate_step | 1e-4 | 1e-4 | Same |
| min_switch_step | 0.1 | N/A | For scaling strategy |
| equilibrate_max_iter | 10 | 10 | Same |
| equilibrate_min/max | 1e-4 / 1e4 | unbounded | **Minix doesn't bound** |
| max_step_fraction | 0.99 | ~0.99 | Similar |

## 3. Scaling Strategy Checkpointing (Clarabel unique)

Clarabel has a sophisticated strategy switching system:

1. **PrimalDual** scaling (default for symmetric cones)
2. **Dual** scaling (fallback for asymmetric cones)

When issues occur:
- Small step (α < 0.1) → switch PrimalDual → Dual
- KKT solve failure → switch PrimalDual → Dual
- Insufficient progress → restore previous iterate, try Dual

```rust
fn strategy_checkpoint_small_step(&mut self, α: T, scaling: ScalingStrategy) {
    if !self.cones.is_symmetric()
        && scaling == ScalingStrategy::PrimalDual
        && α < self.settings.core().min_switch_step_length  // 0.1
    {
        return StrategyCheckpoint::Update(ScalingStrategy::Dual);
    }
    // ...
}
```

**Minix**: No scaling strategy switching. Uses NT scaling throughout.

## 4. Iterative Refinement - Different Approach

### Clarabel (adaptive)
```rust
// Stop early if improvement ratio < stop_ratio (5.0)
let improved_ratio = lastnorme / norme;
if improved_ratio < stopratio {
    if improved_ratio > T::one() {
        std::mem::swap(x, dx);  // Accept if any improvement
    }
    break;  // Stop refinement early
}
```
- Tolerance-based stopping (reltol=1e-13, abstol=1e-12)
- Early exit when improvements stall
- Accepts partial improvements

### Minix (fixed)
- Fixed number of iterations (default 10)
- No early stopping
- No tolerance-based termination

## 5. Static Regularization

### Clarabel (adaptive)
```rust
let eps = settings.static_regularization_constant
        + settings.static_regularization_proportional * maxdiag;
// constant = 1e-8, proportional = ε² ≈ 5e-32 (essentially 0)
```

### Minix (fixed)
```rust
reg_state.static_reg_eff = settings.static_reg;  // Fixed 1e-8
```

Note: The proportional term ε² ≈ 5e-32 is essentially zero, so this is nearly equivalent. Not a major difference.

## 6. First-Iteration Mehrotra Dampening

### Clarabel
```rust
// Make a reduced Mehrotra correction in the first iteration
// to accommodate badly centred starting points
let m = if iter > 1 {T::one()} else {α};

self.step_rhs.combined_step_rhs(
    &self.residuals, &self.variables, &mut self.cones,
    &mut self.step_lhs,
    σ, μ, m  // <-- 'm' scales the Mehrotra correction
);
```

On iter=1, Mehrotra correction is scaled by α (typically < 1).
This prevents large corrections from badly centered initial points.

### Minix
No such dampening. Full Mehrotra correction from iteration 1.

## 7. Insufficient Progress Recovery

### Clarabel
```rust
fn strategy_checkpoint_insufficient_progress(&mut self, scaling: ScalingStrategy) {
    if self.info.get_status() == SolverStatus::InsufficientProgress {
        // Recover old iterate since "insufficient progress" often
        // involves actual degradation of results
        self.info.reset_to_prev_iterate(&mut self.variables, &self.prev_vars);

        if !self.cones.is_symmetric() && scaling == ScalingStrategy::PrimalDual {
            return StrategyCheckpoint::Update(ScalingStrategy::Dual);
        }
        return StrategyCheckpoint::Fail;
    }
}
```

Clarabel explicitly:
1. Saves previous iterate before each step
2. Restores it when progress stalls
3. Tries alternative scaling strategy

### Minix
Has stall detection but different recovery:
- Increases regularization
- Has StallRecovery and Polish modes
- Tracks consecutive failures
- Does NOT restore previous iterate

## 8. Equilibration Bounds

### Clarabel
```rust
equilibrate_min_scaling: 1e-4,
equilibrate_max_scaling: 1e4,
```
Bounds scaling factors to [1e-4, 1e4] (ratio 1e8 max).

### Minix
No explicit bounds on scaling factors.

**Impact**: BOYD2 has extreme scaling. Unbounded equilibration may cause numerical issues.

## 9. Summary of Missing Features in Minix

| Feature | Impact | Effort |
|---------|--------|--------|
| max_iter=200 | Medium | Trivial |
| σ = (1-α)³ centering | **High** | Easy |
| Equilibration bounds | Medium | Easy |
| First-iter Mehrotra dampening | Low-Medium | Easy |
| Adaptive IR with early stop | Low | Medium |
| Scaling strategy switching | Medium | Hard |
| Previous iterate restoration | Medium | Medium |

## Recommendations

### Immediate (should help most)

1. **Change max_iter from 100 to 200**
   - File: `solver-core/src/problem.rs`
   - Just change the default

2. **Implement Clarabel's centering formula**
   - File: `solver-core/src/ipm2/predcorr.rs`
   - Change `sigma = (mu_aff/mu).powi(3)` to `sigma = (1.0 - alpha_aff).powi(3)`
   - This is the most impactful change

3. **Add equilibration bounds**
   - File: `solver-core/src/presolve/ruiz.rs`
   - Clamp scaling factors to [1e-4, 1e4]

### Later

4. First-iteration Mehrotra dampening
5. Adaptive iterative refinement
6. Previous iterate tracking and restoration

## Test Protocol

After each change, run:
```bash
cargo run --release -p solver-bench -- maros-meszaros --problem QSCFXM2
cargo run --release -p solver-bench -- maros-meszaros --problem QSHARE1B
cargo run --release -p solver-bench -- maros-meszaros --problem YAO
```

These are representative of the failing problems.
