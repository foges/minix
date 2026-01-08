# Non-HSDE Direct Solve Mode Plan

## Problem Statement

Several Maros-Meszaros problems hit MaxIters (200) because:
1. **Primal residual stalls** (e.g., YAO: rel_p stuck at 6e-6)
2. **Large duality gap** (e.g., STCQP1: gap_rel=50%, QSHIP04S: gap_rel=16%)

The current HSDE approach adds overhead via τ/κ variables that may not be needed for feasible, bounded problems.

## What Clarabel Does Differently

From [Clarabel paper](https://arxiv.org/html/2405.12762v1):

1. **Modified homogeneous embedding** - keeps P in objective directly (no epigraph)
2. **Still uses τ/κ** for homogeneity and infeasibility detection
3. **Key difference**: Direct handling of quadratics avoids computing P^(1/2)

So Clarabel is NOT abandoning HSDE - it's using a smarter formulation.

## Proposed Approach: Hybrid Mode

Rather than replacing HSDE, add a **direct primal-dual mode** that:
1. Fixes τ = 1, κ = 0 (no homogenization)
2. Uses simpler KKT system
3. Falls back to full HSDE if trouble detected

### Architecture

```
solve_ipm2()
├── SolveMode::Direct     <- NEW: τ=1, κ=0, simpler residuals
├── SolveMode::Normal     <- existing HSDE
├── SolveMode::StallRecovery
└── SolveMode::Polish
```

### Implementation Steps

#### Phase 1: Add Direct Mode Flag

```rust
// In SolverSettings
pub direct_mode: bool,  // default: false (opt-in)

// In solve_ipm2()
let use_direct = settings.direct_mode && is_suitable_for_direct(&prob);
```

#### Phase 2: Simplified Residual Computation

For direct mode (τ=1, κ=0):
```
r_x = P x + A^T z + q        (vs. P x + A^T z + q τ)
r_z = A x + s - b            (vs. A x + s - b τ)
r_τ = not computed           (vs. complex formula)
```

#### Phase 3: Simplified Step Computation

Direct mode KKT system:
```
[P + δI    A^T  ] [Δx]   [r_x]
[A        -H^-1] [Δz] = [r_z]
```

No τ/κ coupling, no homogeneous corrections.

#### Phase 4: Fallback Detection

Switch to HSDE if:
1. Primal or dual residual increases 10x
2. μ increases 10x
3. Step size α < 0.01 for 5 consecutive iterations
4. Iteration count exceeds threshold without progress

### Files to Modify

| File | Changes |
|------|---------|
| `problem.rs` | Add `direct_mode: bool` to SolverSettings |
| `ipm2/solve.rs` | Add mode selection logic, direct residual computation |
| `ipm2/predcorr.rs` | Add `predictor_corrector_direct()` variant |
| `ipm/hsde.rs` | Add `DirectState` struct (no τ/κ) |

### Testing Strategy

1. **Regression safety**: All 92 existing tests must pass
2. **New problems**: Test STCQP1/2, YAO, QSHIP* with direct mode
3. **Fallback**: Verify HSDE fallback works when direct mode fails

### Expected Benefits

| Problem | Current | Direct Mode (expected) |
|---------|---------|----------------------|
| YAO | MaxIters (rel_p=6e-6) | Optimal (simpler steps) |
| STCQP1 | MaxIters (gap=50%) | Potentially better |
| QSHIP04S | MaxIters (gap=16%) | Potentially better |

### Risks

1. **Infeasibility detection lost** in direct mode (mitigated by fallback)
2. **Code complexity** increases with two paths
3. **Testing burden** doubles

### Alternative: Just Improve HSDE

Instead of adding direct mode, could improve existing HSDE:
1. Better step size selection (Mehrotra predictor-corrector tuning)
2. Multiple centrality corrections
3. Warm-start from polish solution

This is less invasive but may not solve the fundamental τ/κ overhead issue.

## Recommendation

**Phase 1**: Implement direct mode as opt-in (`direct_mode=true`)
**Phase 2**: If successful, make it default for LP/QP with auto-detection
**Phase 3**: Keep HSDE for SOCP/SDP and infeasibility detection

## Questions to Resolve

1. Should direct mode use different termination criteria?
2. Should we share KKT solver between modes or have separate instances?
3. How aggressively should we trigger HSDE fallback?
