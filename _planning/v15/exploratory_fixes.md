# Exploratory Fixes for v15 Failures

**Philosophy:** Fix root causes, don't lower the bar. All 28 failures are genuine - they produce solutions that violate tolerances (rel_d > 1e-8, gap_rel > 1e-8).

## Real Failure Categories

### 1. QFFFFF80: Dual Component Explosion (rel_d=768)

**Status at 50 iters:**
- rel_p = 9.1e-9 ✓
- rel_d = 768 ✗ (76,800x too large!)
- gap_rel = 1.3 ✗
- obj_p = +8.7e5, obj_d = -2.8e6 (wrong sign!)

**Root cause:** Variable `rd[170]` = -3.05e8 dominates, driven by 51 constraint connections. This single variable is destroying stationarity.

**Exploratory directions:**
1. **Why is row 170 exploding?** Check constraint matrix structure for rd[170]
   - Is the constraint nearly dependent on others?
   - Is there a scaling issue (large vs tiny coefficients)?

2. **Active-set identification:** At rel_p=1e-9, which constraints are truly active (s ≈ 0)?
   - Can we fix those and solve a reduced KKT system for dual only?
   - Would eliminate ill-conditioned rows from dual update

3. **Constraint preconditioning:** Identify problematic rows BEFORE IPM
   - Compute row condition numbers
   - Rescale or combine nearly-dependent constraints

4. **Why doesn't polish trigger?** gap_rel = 1.3 >> 1e-3 threshold
   - Should we allow "primal-only polish" when rel_p tiny but rel_d huge?
   - Fix primal variables, solve for dual from scratch

**Experiment 1: Dump constraint matrix around rd[170]**
```bash
# Add diagnostic: print A[row] for all rows that feed into rd[170]
# Check for near-linear-dependence
```

**Experiment 2: Force active-set dual recovery at iter 50**
```rust
if rel_p < 1e-6 && rel_d > 10.0 && gap_rel > 0.1 {
    // Fix x where s < 1e-6
    // Solve: minimize ||Px + q + A'z||^2 + rho ||z||^2
    // This is SPD, always solvable
}
```

---

### 2. QSHIP Family (6 problems): Step Size Collapse

**QSHIP04S at 50 iters:**
- rel_p = 4.2e-10 ✓
- rel_d = 0.54 ✗ (54x too large)
- gap_rel = 0.16 ✗
- μ = 5.2e-14 (frozen!)

**Root cause:** μ has hit machine precision, but dual is still 50% away from feasible. The KKT system is so ill-conditioned that the computed step direction (even if mathematically correct) results in step size α → 0 due to cone geometry.

**The step collapse mechanism:**
1. KKT solve produces a direction `(dx, ds, dy, dz)`
2. Direction may be "correct" (reduces residuals in theory)
3. But `s + α*ds` or `z + α*dz` would violate cone constraints
4. Fraction-to-boundary rule forces α → 1e-40
5. Iterate doesn't move, stuck forever

**Exploratory directions:**

1. **Is the KKT direction actually bad?**
   - Print `||KKT_matrix * step - rhs||` to verify solve accuracy
   - If solve residual is large → regularization isn't helping
   - If solve residual is small → direction is "correct" but step is blocked

2. **What's blocking the step?**
   - Is it primal `s` going negative? Or dual `z`?
   - Which specific cone constraint forces α → 0?
   - Print: `alpha_primal`, `alpha_dual` separately, and blocking index

3. **Can we modify the step direction?**
   - Project step onto "safe subspace" that won't hit boundaries
   - Use iterative refinement to get better KKT solution
   - Switch to different linear solver (conjugate gradient?)

4. **Should we declare victory early?**
   - When μ < 1e-12 and rel_p < 1e-8 but step size < 1e-10 for 5+ iters
   - Accept that primal is optimal, try dual-only recovery
   - Then test if solution actually satisfies KKT conditions

**Experiment 1: Diagnose blocking**
```rust
// In step size computation:
let (alpha_p, idx_p, constraint_p) = max_step_to_boundary(&s, &ds, ...);
let (alpha_d, idx_d, constraint_d) = max_step_to_boundary(&z, &dz, ...);
if alpha_p < 1e-6 || alpha_d < 1e-6 {
    println!("BLOCKING: alpha_p={:.2e} at idx {} (s={:.2e}, ds={:.2e})",
             alpha_p, idx_p, s[idx_p], ds[idx_p]);
    println!("BLOCKING: alpha_d={:.2e} at idx {} (z={:.2e}, dz={:.2e})",
             alpha_d, idx_d, z[idx_d], dz[idx_d]);
}
```

**Experiment 2: Dual-only recovery when stuck**
```rust
if mu < 1e-12 && rel_p < 1e-8 && alpha < 1e-8 && stall_count > 5 {
    // Primal is excellent, step size collapsed
    // Fix x and s, solve for (y, z) only:
    //   minimize ||Px + q + A'z||^2 + ||Ax - b + s||^2
    //   subject to s, z in cones
    // This is a simpler problem
}
```

---

### 3. QFORPLAN: HSDE Scaling Ray Divergence (μ → 1e26)

**At 50 iters:**
- rel_p = 5.5e-14 ✓ (excellent!)
- rel_d = 0.86 ✗
- gap_rel = 0.96 ✗
- μ = ??? (grows to 1e26 if allowed to continue)

**Root cause:** HSDE homogenization scalars (τ, κ) are following a "scaling ray" where the embedded problem is feasible but the recovered solution (x/τ, y/τ) doesn't converge.

**Why HSDE can diverge:**
- The embedded system is: `H * v = c` where v = (x,s,y,z,τ,κ)
- Scaling `v → γv` for any γ > 0 is still a solution
- If τ → 0 or τ → ∞, the recovered solution x/τ becomes meaningless
- Merit function checks help but don't fix fundamental scaling issue

**Exploratory directions:**

1. **What's the μ decomposition?**
   - μ = (s·z + τ·κ) / (num_barriers + 1)
   - When μ explodes, is it s·z or τ·κ blowing up?
   - If τ·κ → huge: HSDE problem
   - If s·z → huge: primal-dual problem

2. **Can we detect the scaling ray early?**
   - Track: τ, κ, s·z, τ·κ over iterations
   - If τ oscillates wildly or κ grows unbounded → escaping
   - Switch to direct IPM (no HSDE) for this problem

3. **Why doesn't normalization work?**
   - Current: renormalize when τ ∉ [0.2, 5.0]
   - But this might be fighting the wrong battle
   - Maybe the problem is actually infeasible or near-infeasible?

4. **Is the problem actually feasible?**
   - Run direct IPM (MINIX_DIRECT_MODE=1)
   - If direct IPM fails → problem is hard even without HSDE
   - If direct IPM succeeds → HSDE is the culprit

**Experiment 1: Run without HSDE**
```bash
MINIX_DIRECT_MODE=1 cargo run --release -p solver-bench -- \
    maros-meszaros --max-iter 50 --problem QFORPLAN
```

**Experiment 2: Early HSDE escape detection**
```rust
// Track HSDE health metric
let tau_ratio = tau_new / tau_old;
let kappa_ratio = kappa_new / kappa_old;
let mu_ratio = mu_new / mu_old;

if mu_ratio > 10.0 && (tau_ratio > 5.0 || kappa_ratio > 5.0) {
    // HSDE is diverging, switch to direct IPM mid-solve
    warn!("HSDE escape detected, switching to direct mode");
    // Reconstruct (x,y,z,s) from current iterate
    // Continue with direct IPM
}
```

---

### 4. Dual Slow Convergence (15 problems)

**Example: Q25FV47**
- rel_p = 1.3e-16 ✓ (machine precision!)
- rel_d = 0.42 ✗ (42x too large)
- gap_rel = 0.17 ✗

**Root cause:** Primal converges fast, dual converges slowly or not at all. Common in:
- Degenerate problems (many optimal solutions)
- Ill-conditioned problems (large condition number)
- Problems with inactive constraints

**Exploratory directions:**

1. **Is the dual actually stuck, or just slow?**
   - Run one of these to 100 iters (2x current)
   - Does rel_d continue decreasing, just slowly?
   - Or is it truly flat?

2. **What's the dual update formula?**
   - In predictor-corrector, dual update comes from KKT solve
   - If KKT matrix is ill-conditioned → dual update is noisy
   - Can we use a different dual update (e.g., gradient-based)?

3. **Should we use different step sizes for primal and dual?**
   - Currently: same α for both
   - Many IPM implementations use separate α_primal, α_dual
   - When primal is perfect, let dual take aggressive steps

4. **Active-set detection:**
   - Identify constraints with s ≈ 0 (active)
   - These should have "large" z (tight constraints)
   - Constraints with s >> 0 should have z ≈ 0
   - Is this pattern holding? Or are inactive constraints getting nonzero z?

**Experiment 1: Independent step sizes**
```rust
// Instead of single alpha:
let alpha_primal = 0.99 * max_step_primal(s, ds);
let alpha_dual = if rel_p < 1e-6 && rel_d > 0.1 {
    // Primal converged, dual stuck → aggressive dual step
    0.995 * max_step_dual(z, dz)
} else {
    0.99 * max_step_dual(z, dz)
};

x += alpha_primal * dx;
s += alpha_primal * ds;
y += alpha_dual * dy;
z += alpha_dual * dz;
```

**Experiment 2: Test on QBANDM with 100 iters**
```bash
cargo run --release -p solver-bench -- \
    maros-meszaros --max-iter 100 --problem QBANDM
```

---

## Prioritized Experiments

### Week 1: Diagnostics
1. Run QFORPLAN with direct mode (no HSDE)
2. Run QBANDM to 100 iters (is dual slow or stuck?)
3. Add blocking diagnostics to QSHIP04S (what stops the step?)
4. Dump A matrix structure for QFFFFF80 row 170

### Week 2: Active-set dual recovery
- Implement "fix primal, solve dual LS" for stuck cases
- Test on: QFFFFF80, QSHIP family (7 problems total)
- Should fix problems with rel_p excellent but rel_d bad

### Week 3: HSDE escape detection
- Implement mid-solve HSDE → direct IPM switching
- Test on: QFORPLAN
- Should prevent μ explosion

### Week 4: Dual step modifications
- Implement independent α_primal, α_dual
- Test on: 15 "dual slow" problems
- Might accelerate convergence without breaking anything

---

## Success Criteria

**Do NOT accept solutions that violate:**
- rel_p > 1e-8
- rel_d > 1e-8
- gap_rel > 1e-8

**Valid reasons to mark SOLVED:**
- All three criteria met ✓
- Problem proved infeasible (separate status)
- Problem proved unbounded (separate status)

**Invalid reasons:**
- "Close enough"
- "Industry solver also struggles"
- "Gap is tiny" (if dual is garbage)
