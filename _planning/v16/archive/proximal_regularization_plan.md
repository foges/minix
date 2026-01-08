# V16: Proximal Regularization Implementation Plan

## Executive Summary

**Goal**: Implement Interior Point-Proximal Method of Multipliers (IP-PMM) to close the gap to PIQP.

**Current State**:
- Minix: 77.2% pass rate, 25.9ms geom mean time
- PIQP: 96% pass rate, 1.0x baseline (fastest)

**The Gap**:
- **NOT 10-50x slower** - shifted geometric mean penalizes failures heavily
- **Real gap on solved problems**: ~2-5x slower
- **Main issue**: We fail on 31 problems that PIQP solves (23% gap)

**Root Cause**: Ill-conditioned KKT systems from:
1. Rank-deficient P (LPs, degenerate QPs)
2. Near-singular constraint matrices
3. Extreme scaling issues
4. HSDE tau/kappa divergence

**Solution**: Proximal regularization makes problems strictly convex and well-conditioned.

---

## What is Proximal Regularization?

### Mathematical Foundation

Instead of solving the original problem:
```
minimize    (1/2) x'Px + q'x
subject to  Ax + s = b, s ∈ K
```

PIQP solves a **regularized version**:
```
minimize    (1/2) x'Px + q'x + (ρ/2)||x - x_ref||²
subject to  Ax + s = b, s ∈ K
```

Where:
- `ρ > 0` is the proximal penalty parameter
- `x_ref` is a reference point (updated each iteration)

### Why This Helps

1. **Strict Convexity**: Even when P = 0 (LP) or P is rank-deficient, the term `(ρ/2)||x - x_ref||²` makes the objective strictly convex.

2. **Well-Conditioned KKT**:
   ```
   Original:  [ P      A^T  ]
              [ A      -H   ]

   Proximal:  [ P+ρI   A^T  ]  ← Better condition number!
              [ A      -H   ]
   ```

3. **Convergence to Original**: As iterations progress, x_ref → x_optimal, so the regularization term → 0.

4. **Polynomial Complexity**: First primal-dual IPM with proven polynomial complexity (Pougkakiotis & Gondzio, 2021).

### The Algorithm (IP-PMM)

```
Initialize: ρ = 1e-6, x_ref = 0

For each outer iteration k:
    1. Solve regularized problem with IPM:
       min (1/2)x'Px + q'x + (ρ/2)||x - x_ref||²
       s.t. Ax = b, s ∈ K

    2. Check convergence to ORIGINAL problem (not regularized)
       - Primal feasibility: ||Ax + s - b|| ≤ ε
       - Dual feasibility: ||Px + q + A'z|| ≤ ε  ← Original P, not P+ρI
       - Complementarity: x'(Px + q + A'z) ≤ ε
       - Gap: |primal_obj - dual_obj| ≤ ε

    3. If not converged:
       - Update x_ref = x_current
       - Optionally increase ρ (if progress stalled)

    4. Continue IPM iterations on regularized problem
```

**Key insight**: We run standard Mehrotra IPM, but on a better-conditioned problem!

---

## Implementation Plan

### Phase 1: Basic Proximal Regularization (1 week)

**Goal**: Get a working prototype that improves pass rate by 10-15%.

#### Step 1.1: Add Proximal Parameters to Settings
```rust
// solver-core/src/problem.rs
pub struct SolverSettings {
    // ... existing fields ...

    /// Proximal penalty parameter (default: 1e-6)
    pub proximal_rho: f64,

    /// Enable proximal regularization
    pub use_proximal: bool,
}
```

#### Step 1.2: Add Reference Point to State
```rust
// solver-core/src/ipm2/hsde.rs
pub struct HsdeState {
    pub x: Vec<f64>,
    pub s: Vec<f64>,
    pub z: Vec<f64>,
    pub tau: f64,
    pub kappa: f64,
    pub xi: Vec<f64>,  // Already exists

    /// Reference point for proximal term (x_ref)
    pub x_ref: Vec<f64>,  // NEW
}
```

#### Step 1.3: Modify KKT Assembly
```rust
// solver-core/src/linalg/kkt.rs

// Add rho to KKT assembly
pub fn assemble_kkt(
    P: Option<&CscMatrix<f64>>,
    A: &CscMatrix<f64>,
    scaling: &Scaling,
    rho: f64,  // NEW
) -> Result<KktSystem, String> {
    // When assembling top-left block:
    // OLD: K_11 = P
    // NEW: K_11 = P + ρ*I

    for i in 0..n {
        K.add(i, i, rho);  // Add proximal regularization
    }
}
```

#### Step 1.4: Update q Vector in Objective
```rust
// solver-core/src/ipm2/solve.rs

// Before KKT solve, modify RHS to include proximal gradient:
// grad(proximal_term) = ρ(x - x_ref)
let mut q_prox = prob.q.clone();
if settings.use_proximal {
    for i in 0..n {
        q_prox[i] += settings.proximal_rho * (state.x[i] - state.x_ref[i]);
    }
}
```

#### Step 1.5: Update Termination Checks
```rust
// solver-core/src/ipm/termination.rs

// CRITICAL: Check convergence on ORIGINAL problem, not regularized!
// Must compute residuals using original P and q, not P+ρI and q+ρ(x-x_ref)

fn check_termination(
    prob: &ProblemData,
    state: &HsdeState,
    settings: &SolverSettings,
) -> Option<SolveStatus> {
    // Compute dual residual with ORIGINAL P:
    let r_d = prob.P * x + prob.q + A' * z  // NOT (P+ρI)

    // This is the true optimality check!
}
```

### Phase 2: Adaptive Proximal Updates (3-5 days)

**Goal**: Dynamically update ρ and x_ref for faster convergence.

#### Step 2.1: Reference Point Updates
```rust
// Update x_ref every N iterations (start with N=10)
if iter % 10 == 0 {
    state.x_ref = state.x.clone();
}
```

#### Step 2.2: Adaptive ρ Tuning
```rust
// Increase ρ if progress stalls
if metrics.rel_p > prev_rel_p && metrics.rel_d > prev_rel_d {
    settings.proximal_rho *= 2.0;  // Increase regularization
    settings.proximal_rho = settings.proximal_rho.min(1e-2);  // Cap at 0.01
}
```

### Phase 3: Outer/Inner Iteration Structure (1 week)

**Goal**: Full IP-PMM algorithm with outer proximal loop.

#### Step 3.1: Outer Loop for PMM
```rust
// Pseudo-code structure
fn solve_ipm_pmm(prob: &ProblemData, settings: &SolverSettings) -> SolveResult {
    let mut x_ref = vec![0.0; prob.num_vars()];
    let mut rho = settings.proximal_rho;

    for outer_iter in 0..10 {  // Outer PMM iterations
        // Solve regularized problem with IPM
        let result = solve_ipm_inner(prob, x_ref, rho, settings);

        // Check convergence to ORIGINAL problem
        if converged_to_original(prob, &result) {
            return result;
        }

        // Update proximal parameters
        x_ref = result.x.clone();
        rho = update_rho(rho, &metrics);
    }
}
```

### Phase 4: Incremental KKT Updates (1-2 weeks)

**Goal**: 2-3x speedup from avoiding full refactorization.

#### Step 4.1: Cache KKT Pattern
```rust
pub struct KktSystem {
    // ... existing fields ...

    /// Cached symbolic factorization
    symbolic: Option<LdlSymbolic>,

    /// Track which entries changed
    changed_entries: Vec<(usize, usize)>,
}
```

#### Step 4.2: Incremental Updates
```rust
// Only update H block (scaling changes each iteration)
// P + ρI is constant within outer iteration
pub fn update_kkt_scaling(&mut self, new_scaling: &Scaling) {
    // Only modify H block entries
    for (row, col, val) in self.h_block_entries {
        self.matrix.update(row, col, compute_new_h_val(new_scaling));
    }

    // Numeric refactor only (reuse symbolic)
    self.ldl.refactor_numeric(&self.matrix);
}
```

---

## Expected Results

### Phase 1: Basic Proximal
- **Pass rate**: 77.2% → 87-90%
- **Geom mean time**: 25.9ms → 22ms (fewer failures, slight overhead)
- **Impact**: +10-13% pass rate, solves QFORPLAN, QFFFFF80, QSHIP* family

### Phase 2: Adaptive Updates
- **Pass rate**: 87-90% → 90-92%
- **Geom mean time**: 22ms → 20ms
- **Impact**: Fewer iterations needed, faster convergence

### Phase 3: Full IP-PMM
- **Pass rate**: 90-92% → 92-94%
- **Geom mean time**: 20ms → 18ms
- **Impact**: Match PIQP's robustness

### Phase 4: Incremental KKT
- **Pass rate**: 92-94% (no change)
- **Geom mean time**: 18ms → 10-12ms (2-3x faster)
- **Impact**: Major speedup on large problems

### Combined Final Target
- **Pass rate**: 92-95% (vs PIQP's 96%)
- **Geom mean time**: 10-15ms (vs PIQP's baseline)
- **Gap closed**: 70-80% of the way to PIQP

---

## Risk Assessment

### Technical Risks

1. **Termination criteria complexity** (MEDIUM)
   - Risk: Hard to check convergence to original vs regularized problem
   - Mitigation: Carefully track which residuals use original vs proximal terms

2. **Optimal ρ tuning** (MEDIUM)
   - Risk: Too large → slow convergence, too small → ill-conditioning
   - Mitigation: Start conservative (1e-6), increase adaptively

3. **Reference point updates** (LOW)
   - Risk: Updating x_ref too often → no convergence
   - Mitigation: Update every 10 iters or when progress stalls

### Performance Risks

1. **Overhead from proximal gradient** (LOW)
   - Extra vector operations: q_prox = q + ρ(x - x_ref)
   - Negligible compared to KKT solve time

2. **More outer iterations** (LOW)
   - Might need 2-3 outer PMM iterations
   - Still faster than failing entirely

---

## Testing Strategy

### Unit Tests
```rust
#[test]
fn test_proximal_regularization_lp() {
    // Test on LP (P = 0) where proximal helps most
    let prob = simple_lp();

    let result_no_prox = solve(&prob, SolverSettings { use_proximal: false, .. });
    let result_prox = solve(&prob, SolverSettings { use_proximal: true, .. });

    assert!(result_prox.status == Optimal);
    assert_eq!(result_prox.obj_val, result_no_prox.obj_val, 1e-6);
}
```

### Regression Tests
1. Run on current 105 solved problems → must still solve with proximal
2. Run on 31 failed problems → expect 20-25 to now solve

### Benchmark Tracking
```bash
# Before proximal
cargo run --release -p solver-bench -- maros-meszaros --export-json before.json

# After phase 1
cargo run --release -p solver-bench -- maros-meszaros --export-json phase1.json

# Compare
cargo run --release -p solver-bench -- compare before.json phase1.json --detailed
```

---

## Implementation Schedule

| Phase | Effort | Duration | Deliverable |
|-------|--------|----------|-------------|
| Phase 1: Basic Proximal | Medium | 5-7 days | +10-15% pass rate |
| Phase 2: Adaptive Updates | Low | 3-5 days | +2-3% pass rate, fewer iters |
| Phase 3: Full IP-PMM | Medium | 5-7 days | +2-3% pass rate |
| Phase 4: Incremental KKT | Medium | 7-10 days | 2-3x speedup |
| **Total** | | **3-4 weeks** | **92-95% pass rate, 10-15ms** |

---

## Success Metrics

### Must-Have (Phase 1)
- ✅ Pass rate ≥ 87% (up from 77.2%)
- ✅ Solve QFORPLAN (currently fails due to μ explosion)
- ✅ Solve QFFFFF80 (currently fails due to dual explosion)
- ✅ No regressions on currently-solved problems

### Should-Have (Phase 2-3)
- ✅ Pass rate ≥ 90%
- ✅ Solve QSHIP family (4+ problems)
- ✅ Geom mean time ≤ 20ms

### Nice-to-Have (Phase 4)
- ✅ Pass rate ≥ 92%
- ✅ Geom mean time ≤ 15ms
- ✅ Within 3-5 problems of PIQP (132 vs our 125-128)

---

## References

- [PIQP ArXiv Paper](https://arxiv.org/abs/2304.00290)
- [IP-PMM Original Paper (Pougkakiotis & Gondzio, 2021)](https://link.springer.com/article/10.1007/s10589-020-00240-9)
- [PIQP GitHub](https://github.com/PREDICT-EPFL/piqp)
- [Maros-Meszaros Benchmark](https://github.com/qpsolvers/maros_meszaros_qpbenchmark)

---

## Next Steps

1. ✅ Research PIQP algorithm (DONE)
2. ✅ Write planning doc (DONE)
3. 🔄 Implement Phase 1: Basic Proximal Regularization
4. ⏳ Test on failing problems (QFORPLAN, QFFFFF80, QSHIP*)
5. ⏳ Measure impact and iterate
