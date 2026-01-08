# V19 Implementation Plan: Industry-Standard Robustness Features

## Executive Summary

**Goal**: Close the iteration count and robustness gap vs MOSEK/Clarabel by implementing literature-standard IPM features that address conditioning-limited problems like BOYD.

**Current State**: 78.3% pass rate (108/138 MM problems)
**Target**: 85-90% pass rate by fixing conditioning-limited failures

**Key Insight**: BOYD failure (dual stalls at ||r_d||=134 despite excellent primal) is a textbook case where iterative refinement + dynamic regularization are the standard solution.

---

## Literature-Based Diagnosis

### What Top Solvers Do That Minix Doesn't

From the comprehensive literature analysis:

**Iteration Quality** (why MOSEK/Clarabel converge in fewer iterations):
1. **Iterative refinement** on ill-conditioned KKT solves (1-2 passes standard)
2. **Dynamic regularization** (detect conditioning issues → adapt regularization)
3. **Multiple correctors** (not just 1 affine + 1 corrector)
4. **Nonsymmetric cone higher-order corrections** (3rd order for exp cone)

**Wallclock Speed** (why they're faster per iteration):
1. **Supernodal/multifrontal factorization** (not basic QDLDL)
2. **Ordering reuse** (symbolic factorization once, reuse elimination tree)
3. **Multithreading** in factorization
4. **Vectorized cone primitives** (SIMD)

### Minix Current State

**Has** ✅:
- HSDE formulation
- Predictor-corrector (single corrector)
- NT scaling for symmetric cones
- Static regularization
- Polish phase
- Ruiz equilibration

**Missing** ❌:
- Iterative refinement (literature standard)
- Dynamic regularization (adapt to conditioning)
- Multiple correctors
- Higher-order exp cone corrections
- Supernodal factorization
- Ordering reuse

---

## Three-Tier Implementation Plan

### Tier 1: Robustness (Highest ROI for BOYD-class problems)

These directly address the "dual residual floor" problem on ill-conditioned systems.

#### 1.1 Iterative Refinement on KKT Solves

**Priority**: CRITICAL (P0)
**Estimated Impact**: +5-10% pass rate (BOYD + similar conditioning-limited problems)
**Complexity**: Low (1-2 hours)

**Implementation**:
```rust
// After LDL backsolve in solve_normal.rs and solve_direct.rs
fn solve_with_refinement(
    kkt: &KKTMatrix,
    ldl: &LDLFactorization,
    rhs: &[f64],
    x: &mut [f64],
    max_refinement_iters: usize,  // typically 1-2
) {
    // Initial solve
    ldl.solve(x, rhs);

    // Refinement passes
    for _ in 0..max_refinement_iters {
        // Compute residual: r = rhs - K*x
        let residual = compute_residual(kkt, x, rhs);

        // Check if refinement helped
        let residual_norm = inf_norm(&residual);
        if residual_norm < 1e-14 * rhs_norm {
            break;  // Converged to machine precision
        }

        // Solve correction: K * delta_x = residual
        let mut delta_x = vec![0.0; x.len()];
        ldl.solve(&mut delta_x, &residual);

        // Update: x += delta_x
        for i in 0..x.len() {
            x[i] += delta_x[i];
        }
    }
}
```

**Where to apply**:
- `solver-core/src/ipm2/solve_normal.rs` - normal equations solve
- `solver-core/src/ipm2/solve_direct.rs` - direct KKT solve (if exists)
- Enable by default for all problems (cheap: just 1-2 backsolves)

**Expected result**: BOYD dual residual improves from ||r_d||=134 to <1e-6

---

#### 1.2 Dynamic Regularization

**Priority**: HIGH (P1)
**Estimated Impact**: +2-5% pass rate
**Complexity**: Medium (2-3 hours)

**Problem**: Minix uses static regularization (fixed ε_p, ε_d). When conditioning is terrible, these may be too small.

**Solution**: Detect dual stall → increase regularization adaptively.

**Implementation**:
```rust
struct AdaptiveRegularization {
    eps_primal: f64,
    eps_dual: f64,
    last_rel_d: f64,
    stall_count: usize,
}

impl AdaptiveRegularization {
    fn update(&mut self, metrics: &UnscaledMetrics) {
        // Detect dual stall
        let dual_improving = metrics.rel_d < self.last_rel_d * 0.9;

        if !dual_improving && metrics.rel_d > 1e-3 {
            self.stall_count += 1;

            if self.stall_count >= 3 {
                // Increase dual regularization
                self.eps_dual *= 10.0;
                self.eps_dual = self.eps_dual.min(1e-4);  // cap at 1e-4

                eprintln!("Dual stall detected, increasing eps_dual to {:.2e}", self.eps_dual);
                self.stall_count = 0;
            }
        } else {
            self.stall_count = 0;
        }

        self.last_rel_d = metrics.rel_d;
    }
}
```

**Integration**: Add to main IPM loop in `solve.rs`

**Expected result**: Automatically recover from dual stalls by improving KKT conditioning

---

#### 1.3 Condition Number Estimation & Diagnostics

**Priority**: MEDIUM (P2)
**Estimated Impact**: 0% pass rate (diagnostics only)
**Complexity**: Low (1 hour)

**Goal**: Measure when KKT is ill-conditioned so we know when refinement/regularization is needed.

**Implementation**:
```rust
fn estimate_condition_number(ldl: &LDLFactorization) -> f64 {
    // Cheap estimate from diagonal of D in LDL
    let d_max = ldl.D.iter().fold(0.0, |acc, &x| acc.max(x.abs()));
    let d_min = ldl.D.iter().fold(f64::INFINITY, |acc, &x| {
        if x.abs() > 1e-20 {
            acc.min(x.abs())
        } else {
            acc
        }
    });
    d_max / d_min
}
```

**Usage**: Log when condition number > 1e12 (warning threshold)

---

### Tier 2: Multiple Correctors & Exp Cone Improvements

#### 2.1 Multiple Corrector Steps

**Priority**: MEDIUM (P2)
**Estimated Impact**: +2-5% pass rate (iteration count reduction)
**Complexity**: Medium (3-4 hours)

**Current**: Minix does 1 affine + 1 corrector
**Literature**: Modern solvers do 1 affine + 2-3 correctors when beneficial

**Implementation**:
```rust
// In predcorr.rs
for corrector_iter in 0..max_correctors {
    // Compute corrector RHS
    let sigma = compute_centering_parameter(...);
    let rhs_corr = compute_corrector_rhs(sigma, ...);

    // Solve corrector step
    solve_kkt(kkt, &rhs_corr, &mut delta_corr);

    // Check if corrector improves merit
    let merit_after = compute_merit(x + delta_corr, ...);
    if merit_after < merit_before {
        // Accept corrector
        apply_step(&mut x, &delta_corr);
        merit_before = merit_after;
    } else {
        // Corrector didn't help, stop
        break;
    }
}
```

**Heuristic**: Do max 2-3 correctors, stop when merit stops improving

---

#### 2.2 Higher-Order Exp Cone Corrections

**Priority**: LOW (P3)
**Estimated Impact**: +1-2% pass rate (exp cone specific)
**Complexity**: High (4-6 hours)

**Problem**: Nonsymmetric cones (exp, power) need special treatment

**Solution**: Add 3rd-order correction terms for exp cone (see Clarabel papers)

**Status**: DEFERRED (Minix has few exp cone problems currently)

---

### Tier 3: Factorization Backend & Wallclock Optimization

#### 3.1 Ordering Reuse

**Priority**: HIGH (P1) for wallclock
**Estimated Impact**: 20-40% wallclock reduction
**Complexity**: Low (1-2 hours)

**Problem**: Minix recomputes symbolic factorization every iteration (wasteful)

**Solution**: Compute ordering once, reuse elimination tree

**Implementation**:
```rust
struct CachedFactorization {
    ordering: Option<Vec<usize>>,
    elim_tree: Option<EliminationTree>,
}

// First iteration: compute ordering
if cached.ordering.is_none() {
    cached.ordering = Some(compute_amd_ordering(kkt));
    cached.elim_tree = Some(symbolic_factorization(kkt, &cached.ordering));
}

// Subsequent iterations: reuse
numeric_factorization(kkt, &cached.ordering, &cached.elim_tree);
```

**Expected result**: 20-40% wallclock speedup (symbolic factorization is expensive)

---

#### 3.2 Supernodal Factorization (Future Work)

**Priority**: LOW (P4)
**Estimated Impact**: 50-200% wallclock improvement (problem-dependent)
**Complexity**: Very High (weeks)

**Status**: OUT OF SCOPE for v19 (requires new factorization backend)

Options:
- Integrate Intel MKL Pardiso
- Integrate CHOLMOD (supernodal)
- Write custom supernodal LDL (major project)

---

## Implementation Order

### Phase 1: Iterative Refinement (Day 1, 2-3 hours)
1. Implement `solve_with_refinement()` function
2. Add to normal equations path
3. Test on BOYD1/BOYD2
4. Measure improvement in dual residual

**Success Metric**: BOYD1 rel_d < 1e-6 (from 8e-4)

---

### Phase 2: Dynamic Regularization (Day 1, 2-3 hours)
1. Add `AdaptiveRegularization` struct
2. Integrate into main IPM loop
3. Test on QFFFFF80 (known dual stall)
4. Test on BOYD1/BOYD2

**Success Metric**: Fewer MaxIters failures due to dual stall

---

### Phase 3: Condition Number Diagnostics (Day 1, 1 hour)
1. Add `estimate_condition_number()` from LDL diagonal
2. Log warnings when condition > 1e12
3. Correlate with refinement iterations

**Success Metric**: Clear diagnostic output showing when refinement helps

---

### Phase 4: Ordering Reuse (Day 2, 2-3 hours)
1. Cache AMD ordering after first iteration
2. Reuse for numeric factorization
3. Benchmark wallclock time improvement
4. Run full regression suite

**Success Metric**: 20-40% wallclock reduction on medium/large problems

---

### Phase 5: Multiple Correctors (Day 2-3, 3-4 hours)
1. Extend corrector loop to allow 2-3 iterations
2. Add merit function check (stop if not improving)
3. Test iteration count reduction
4. Run full regression suite

**Success Metric**: 5-10% fewer iterations on average

---

### Phase 6: Full Regression + Comparison (Day 3, 2 hours)
1. Run full regression suite with all improvements
2. Measure pass rate improvement
3. Compare wallclock vs baseline
4. Update documentation

**Success Metric**: 85%+ pass rate (up from 78.3%)

---

## Testing Strategy

### Unit Tests
- Iterative refinement: synthetic ill-conditioned systems
- Dynamic regularization: mock dual stall scenarios
- Condition number estimation: known matrices

### Integration Tests
- BOYD1/BOYD2 (conditioning-limited)
- QFFFFF80 (dual stall)
- Full regression suite (108 passing + 30 expected-to-fail)

### Benchmarks
- Wallclock time: before/after on 20 representative problems
- Iteration counts: geometric mean across all problems
- Condition number distributions: histogram

---

## Success Criteria

**Minimum Success** (v19 shipped):
- ✅ Iterative refinement implemented and enabled
- ✅ Dynamic regularization implemented
- ✅ Pass rate: 85%+ (110+/138 MM problems)
- ✅ BOYD1 passes (rel_d < 1e-6)

**Stretch Goals**:
- 🎯 Ordering reuse (wallclock speedup)
- 🎯 Multiple correctors (iteration reduction)
- 🎯 Pass rate: 90%+ (124+/138)

---

## Files to Modify

**Core Implementation**:
- `solver-core/src/ipm2/solve_normal.rs` - Add iterative refinement
- `solver-core/src/ipm2/solve.rs` - Add dynamic regularization
- `solver-core/src/linalg/kkt.rs` - Add condition number estimation
- `solver-core/src/ipm2/predcorr.rs` - Add multiple correctors
- `solver-core/src/settings.rs` - Add refinement/corrector settings

**Testing**:
- `solver-bench/src/regression.rs` - Track improvements
- `solver-bench/tests/refinement_tests.rs` - New unit tests

**Documentation**:
- `_planning/v19/RUNNING_LOG.md` - Session log
- `_planning/v19/RESULTS.md` - Test results
- `_planning/v19/ANALYSIS.md` - Before/after comparison

---

## Timeline

**Estimated Total**: 2-3 days full-time

- **Day 1 AM**: Iterative refinement (2-3 hours)
- **Day 1 PM**: Dynamic regularization + diagnostics (3-4 hours)
- **Day 2 AM**: Ordering reuse (2-3 hours)
- **Day 2 PM**: Multiple correctors (3-4 hours)
- **Day 3**: Testing, regression, documentation (6-8 hours)

---

## Risk Assessment

**Low Risk**:
- Iterative refinement (well-established, cheap, safe)
- Condition number estimation (diagnostics only)

**Medium Risk**:
- Dynamic regularization (could hurt if heuristic is wrong)
- Multiple correctors (could waste iterations if not beneficial)

**Mitigation**:
- Add environment variable toggles: `MINIX_REFINEMENT_ITERS`, `MINIX_DYNAMIC_REG`, `MINIX_MAX_CORRECTORS`
- Default to conservative settings (1 refinement iter, dynamic reg enabled, 2 correctors max)
- Test on regression suite at each step

---

## Literature References

**Iterative Refinement**:
- Standard technique in numerical linear algebra (Wilkinson, Golub)
- Explicitly mentioned in modern IPM solver papers (QOCO, etc.)

**Dynamic Regularization**:
- Discussed in regularization strategies for ill-conditioned KKT systems
- Adaptive methods standard in commercial solvers

**Multiple Correctors**:
- Mehrotra predictor-corrector extensions
- Common in modern conic IPM implementations

**Nonsymmetric Cone Scaling**:
- Literature on exp/power cone IPM methods
- Clarabel papers on higher-order corrections

---

## Next Steps

1. Create v19 directory structure
2. Implement Phase 1 (iterative refinement)
3. Test on BOYD1
4. Continue to Phase 2-6

**Let's begin execution.**
