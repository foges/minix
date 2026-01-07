# Next Steps for v15 (Rigorous Approach)

## Current Status

**Pass rate:** 108/136 (79.4%) with 50-iteration limit
**All 28 failures are genuine** - they violate tolerance (rel_d, rel_p, or gap_rel > 1e-8)

## Key Findings from Initial Exploration

### 1. QFORPLAN: Not an HSDE problem!
- Tested with `MINIX_DIRECT_MODE=1` → still explodes (μ=1e24)
- Variable rd[31] explodes in both HSDE and direct mode
- **This is a problem-specific numerical breakdown**, not a solver mode issue
- Likely: extreme ill-conditioning or near-infeasibility

### 2. QFFFFF80: Single variable ruins dual
- rd[170] = -3e8 (one variable out of 854)
- This variable has 51 constraint connections
- Suggests: nearly dependent constraints or scaling issue

### 3. QSHIP family: Step size collapse is real
- μ → 1e-14 (machine precision)
- Primal excellent (rel_p ~ 1e-10)
- Dual stuck (rel_d ~ 0.5)
- Step size α → 0 due to cone geometry blocking direction
- **Not a tolerance issue** - gap is still 10-70% of objective!

### 4. Dual slow convergence: Need more data
- 15 problems with excellent primal, slow dual
- Unclear if they're truly stuck or just need more iterations
- Need experiment: run 1-2 to 100 iters to see trajectory

## Immediate Next Actions

### Action 1: Understand the 3 hard problems better

**QFORPLAN diagnostics (1-2 hours):**
```rust
// In ipm/predcorr.rs, add after iteration loop:
if name == "QFORPLAN" && iter % 10 == 0 {
    // Check μ decomposition
    let mu_sz = compute_barrier_term(s, z, cones);
    let mu_tau_kappa = tau * kappa;
    println!("μ decomp: sz={:.2e} τκ={:.2e} ratio={:.2f}",
             mu_sz, mu_tau_kappa, mu_sz / mu_tau_kappa);

    // Check constraint matrix for rd[31]
    // Which rows contribute to this variable?
    // Are any nearly parallel?
}
```

**QFFFFF80 constraint analysis (1 hour):**
```rust
// For rd[170] with 51 connections:
// - Compute row norms of contributing constraints
// - Check for near-linear-dependence using QR or SVD
// - Try rescaling just that variable
```

**QSHIP04S blocking analysis (1 hour):**
```rust
// In step computation:
if alpha_primal < 1e-8 || alpha_dual < 1e-8 {
    println!("BLOCK primal: idx={} s={:.2e} ds={:.2e} would_be={:.2e}",
             blocking_idx_p, s[blocking_idx_p], ds[blocking_idx_p],
             s[blocking_idx_p] + alpha_primal * ds[blocking_idx_p]);
    println!("BLOCK dual: idx={} z={:.2e} dz={:.2e} would_be={:.2e}",
             blocking_idx_d, z[blocking_idx_d], dz[blocking_idx_d],
             z[blocking_idx_d] + alpha_dual * dz[blocking_idx_d]);
}
```

### Action 2: Test iteration scaling (30 min)

```bash
# Are "dual slow" problems actually progressing?
for prob in QBANDM QBRANDY QE226; do
    echo "=== $prob at 50 iters ==="
    cargo run --release -p solver-bench -- maros-meszaros --max-iter 50 --problem $prob | grep "rel_d"

    echo "=== $prob at 100 iters ==="
    cargo run --release -p solver-bench -- maros-meszaros --max-iter 100 --problem $prob | grep "rel_d"
done
```

If rel_d improves significantly (e.g., 0.1 → 0.01), then these just need more time.
If rel_d is flat (0.1 → 0.099), then they're truly stuck.

### Action 3: Implement active-set dual recovery (4-6 hours)

For problems with:
- rel_p < 1e-6 (primal converged)
- rel_d > 1.0 (dual bad)
- gap_rel > 0.1 (not close)

Try fixing primal and solving dual-only:

```rust
pub fn recover_dual_from_primal(
    prob: &ProblemData,
    x: &[f64],
    s: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), String> {
    // Identify active constraints (s < epsilon)
    let active: Vec<usize> = s.iter().enumerate()
        .filter(|(_, &si)| si < 1e-6)
        .map(|(i, _)| i)
        .collect();

    // Solve: minimize ||Px + q + A'z||^2 + rho ||z||^2
    // This is always SPD, no KKT saddle point needed

    // Build normal equations: (AA' + rho I) z = -(A * (Px + q))
    // Solve with Cholesky (guaranteed to work)

    // Project z onto cone constraints

    // Compute y from equality constraints

    Ok((y, z))
}
```

Test on: QFFFFF80, QSHIP family (7 problems)
Expected: 5-7 more problems pass (70-80% → 74-85%)

### Action 4: Independent primal/dual step sizes (2-3 hours)

Current: single α for all variables
Better: α_primal for (x,s), α_dual for (y,z)

```rust
let alpha_primal = 0.99 * max_step_primal;
let alpha_dual = if rel_p < 1e-6 && rel_d > 0.1 {
    // Primal converged, push dual harder
    min(0.995, 2.0 * alpha_primal) * max_step_dual
} else {
    0.99 * max_step_dual
};
```

Test on: 15 "dual slow" problems
Expected: 2-5 more problems pass

## Timeline Estimate

**Week 1 (Diagnostics & Understanding):**
- Day 1-2: Add diagnostics, run experiments on QFORPLAN/QFFFFF80/QSHIP04S
- Day 3: Test iteration scaling on "dual slow" problems
- Day 4-5: Analyze results, identify root causes

**Week 2 (Active-set dual recovery):**
- Day 1-2: Implement dual recovery via LS
- Day 3: Test and debug on QFFFFF80
- Day 4-5: Test on QSHIP family, tune regularization

Expected: **113-120/136 (83-88%)**

**Week 3 (Independent step sizes):**
- Day 1: Implement separate α_primal, α_dual
- Day 2-3: Test on all 28 failures
- Day 4-5: Tune heuristics, validate doesn't break passing tests

Expected: **118-125/136 (87-92%)**

**Week 4 (Problem-specific fixes):**
- QFORPLAN: May need to declare "numerically too hard" or implement special handling
- Remaining ~10 problems: Individual analysis

## Definition of Success

**Acceptable outcomes for a problem:**
1. ✅ **Optimal:** rel_p < 1e-8, rel_d < 1e-8, gap_rel < 1e-8
2. ✅ **Infeasible (proven):** HSDE detects κ/τ → ∞ with theory guarantees
3. ✅ **Unbounded (proven):** Similar
4. ⚠️ **NumericalError:** Solver breaks but honestly reports it

**Unacceptable outcomes:**
1. ❌ Returning "Optimal" when tolerances violated
2. ❌ Silently accepting bad solutions
3. ❌ Ignoring dual just because primal is good

## Success Metrics

**By end of exploration:**
- Target: 85-90% (116-122/136) genuinely optimal
- Understand root cause of each remaining failure
- Document which problems are "too hard for current IPM"
- No relaxed tolerances, no silent failures

**Quality bar:**
- All 116+ passing problems have:
  - rel_p < 1e-8 ✓
  - rel_d < 1e-8 ✓
  - gap_rel < 1e-8 ✓
- All failing problems have clear diagnostic why they fail
- Regression suite still at 110/110
