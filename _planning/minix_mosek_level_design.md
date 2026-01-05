# Minix → MOSEK‑class Conic IPM: Engineering Design Doc

_Last updated: 2026-01-03_

This document turns the current “gap analysis” into a **followable engineering plan** with **concrete tasks** and **implementation snippets**. It is intended to be used directly as a work tracker.

This plan builds on the existing project design direction (HSDE/IPM, two-solve strategy, scaling/termination requirements). fileciteturn0file0

---

## 0. Executive summary

Minix is already in the right algorithm family (HSDE + predictor–corrector IPM). The remaining gap to “MOSEK‑class” is **not** “more iterations” or “better σ formula”. It is:

1) **Linear solve quality & regularization discipline** (the #1 source of “μ→0 but dual residual stalls”, e.g., BOYD1).  
2) **Structural problem handling / presolve** (BOYD2-type problems explode in size/conditioning when bounds are encoded as rows).  
3) **Performance engineering** (allocation-free iteration loop, KKT pattern reuse, real multi-RHS solves, avoid dense SOC materialization).  
4) **End-game polish & stall recovery** (solver must not spin to MaxIters when it’s “nearly optimal”).

This doc specifies concrete tasks to close those gaps.

---

## 1. Design principles (non-negotiables)

### P1.1 “No hacks”: every stabilization must be principled and measurable
- Avoid ad-hoc perturbations of the Newton system (e.g., “if min(s,z) small, add huge shift into H”).
- Prefer:
  - **small static reg** + **dynamic pivot protection**
  - **iterative refinement** against the *same* matrix actually solved
  - **stall policies** (mode switches) over “keep iterating”

### P1.2 Termination is evaluated on **unscaled** metrics
As required by the design doc, feasibility, gap, and certificates must be computed after undoing Ruiz scaling. fileciteturn0file0

### P1.3 Hot loop is allocation-free
After the first iteration:
- no `Vec::new()` in `predcorr`, `kkt.solve`, `termination`, or cone kernels.

### P1.4 KKT structure is reused
- symbolic factorization is reused
- CSC pattern is reused
- only numeric values update each iteration

---

## 2. Problem statement: what’s failing today (BOYD1/BOYD2)

### BOYD1 symptom (observed)
- `μ ~ 4e-10`, `gap_rel ~ 5e-9`, `rel_p ~ 1e-14` ✅
- `rel_d ~ 1e-3` stalls ❌
- solver hits `MaxIters` after spending ~20s

**Interpretation:** “Almost optimal” but **dual feasibility polishing is blocked by a solve-accuracy floor** and/or regularization.

### BOYD2 symptom (observed)
- huge problem with tons of constraints/bounds
- complementarity not collapsing fast, residuals limited by conditioning

**Interpretation:** this is as much a **modeling/presolve + structure exploitation** problem as it is an IPM problem.

---

## 3. Architecture changes (what we will add)

### 3.1 New core structs

#### 3.1.1 `IpmWorkspace` (allocation-free iteration)
Own all buffers needed for one iteration.

```rust
pub struct IpmWorkspace {
    // RHS vectors (KKT dimension = n + m_eff)
    pub rhs1: Vec<f64>,
    pub rhs2: Vec<f64>,

    // Solutions for the two RHS solves
    pub sol1: Vec<f64>,
    pub sol2: Vec<f64>,

    // Cone scratch (shared, sized to max block)
    pub cone_tmp: Vec<f64>,

    // Termination scratch
    pub r_p: Vec<f64>,
    pub r_d: Vec<f64>,
    pub p_x: Vec<f64>,
}
```

#### 3.1.2 `KktCache` (pattern reuse + numeric updates)
The KKT matrix structure is fixed for a solve (after presolve/scaling). Only numeric values change.

```rust
pub struct KktCache {
    pub n: usize,
    pub m_eff: usize,

    // KKT CSC structure: indptr/indices are constant
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub values: Vec<f64>,

    // Fast numeric update maps
    pub h_diag_value_ix: Vec<usize>,      // positions of -H diagonal entries
    pub p_diag_value_ix: Vec<usize>,      // optional: positions of P diag for shifts
}
```

#### 3.1.3 `RegularizationPolicy` (stop hardcoding floors in random places)
Centralize all regularization rules.

```rust
pub struct RegularizationPolicy {
    pub static_reg: f64,
    pub dynamic_min_pivot: f64,

    // Adaptive end-game / stall knobs
    pub polish_static_reg: f64,
    pub max_refine_iters: usize,
}
```

---

## 4. Workstream A: Linear solve quality (the BOYD1 killer)

### A0. Goal
Eliminate the “accuracy floor” where μ collapses but `r_d` stalls.

### A1. Task: Make refinement correct against the *factorized* matrix
**Problem:** if factorization uses `K + εI`, refinement residual must also use `K + εI`.

**Implementation**
- Store the applied shifts (`static_reg`, dynamic bumps if any) in the solver.
- During residual matvec, compute: `Kx + static_reg * x` (+ any other shifts).

```rust
// Inside refinement loop:
symm_matvec_upper(kkt_unshifted, &x, &mut kx);
for i in 0..kkt_dim {
    kx[i] += static_reg * x[i];
}
res[i] = rhs[i] - kx[i];
```

**Acceptance**
- On BOYD1, increasing `kkt_refine_iters` must actually reduce KKT residual norms (observable in diagnostics).

---

### A2. Task: Replace “static_reg floors” with a principled policy
**Problem:** forcing `static_reg >= 1e-4` is effectively solving a different problem and creates a residual floor.

**Implementation approach**
- Default: `static_reg = 1e-8` (double precision)
- Allow problem-based scaling: `static_reg_eff = static_reg * scale_factor`
- Use dynamic pivot protection to ensure solvability instead of huge static reg.

```rust
pub fn effective_static_reg(settings: &SolverSettings, scale: f64) -> f64 {
    (settings.static_reg * scale).max(1e-12).min(1e-4)
}
```

Where `scale` might be `max(1, ||diag(P)||_∞, ||H||_∞)`.

**Acceptance**
- BOYD1 should be able to polish `rel_d` by at least 2–3 orders of magnitude without `MaxIters`.

---

### A3. Task: Add a “polish mode” when μ is already tiny
**Problem:** if μ is below tolerance but `r_d` is not, continuing standard predictor–corrector may not help.

**Trigger**
- `mu < tol_gap * 1e-2` AND `dual_res > tol_feas * 10` for `k` consecutive iterations.

**Polish mode behavior**
- Increase refinement iters (`kkt_refine_iters = 5..10`)
- Reduce static reg (down to `polish_static_reg`, e.g. `1e-10`)
- Use combined RHS weighting that prioritizes feasibility (or specifically dual feasibility)

```rust
if polish_mode {
    reg.static_reg = reg.polish_static_reg;
    refine_iters = reg.max_refine_iters;
    rhs_mode = CombinedRhsMode::ScaledByOneMinusSigma;
}
```

**Acceptance**
- BOYD1 terminates `Optimal` (or at least stops within < max_iter) with `rel_d <= tol_feas` at default tolerances.

---

### A4. Task: Instrument KKT solve accuracy (so we stop guessing)
Add diagnostics (env flag `MINIX_DIAGNOSTICS=1`) per iteration:
- `||Kx - rhs||_∞` for both RHS solves (after refinement)
- number of dynamic pivot bumps
- effective `static_reg`

**Snippet**
```rust
if diagnostics {
    eprintln!(
        "kkt: static_reg={:.2e} bumps={} res1={:.2e} res2={:.2e}",
        static_reg, bumps, kkt_res1_inf, kkt_res2_inf
    );
}
```

**Acceptance**
- When BOYD1 stalls, you must be able to tell if it’s a solve-accuracy floor vs step-size pathology.

---

## 5. Workstream B: Structural exploitation for bounds and identity rows (the BOYD2 killer)

### B0. Goal
Prevent “tons of bound rows” from turning into gigantic ill-conditioned KKT systems.

### B1. Task: Detect identity-row constraint blocks
Many modeling pipelines encode bounds as rows where each row has exactly one nonzero in A.

**Detection**
Scan A row-by-row:
- count nonzeros per row
- if exactly one nonzero, record `(row, col, val)` as “singleton row”

Partition constraints:
- `A = [A_gen; A_singleton]`

**Snippet**
```rust
pub struct SingletonRow {
    pub row: usize,
    pub col: usize,
    pub val: f64,
}

pub fn detect_singleton_rows(a: &SparseCsc<f64>) -> Vec<SingletonRow> { /* ... */ }
```

### B2. Task: Eliminate singleton rows from the KKT (Schur-style)
If a row is `val * x[col] + s[row] = b[row]`, then in the condensed KKT:

`A_singleton Δx - H_singleton Δz_singleton = rhs`

Solve for `Δz_singleton` and substitute into the first equation, producing a **diagonal update** to the `P` block:

- Add to `P_eff[col,col] += val^2 / H_row`
- Add to RHS: `rhs_x[col] += val * rhs_row / H_row`

This reduces KKT dimension dramatically.

**Pseudo**
```text
for each singleton row r with (col, val):
    h = H[r]            // scalar
    rhs = rhs_z[r]      // scalar (post (d_z-d_s) handling)
    P_eff[col,col] += val*val / h
    rhs_x[col] += val * rhs / h
```

**Acceptance**
- BOYD2 KKT dimension drops from (n+m) to roughly (n+m_gen).
- Wallclock improves massively (even if iterations stay similar).

---

### B3. Task: Preserve correct dual recovery
When eliminating singleton rows, you must recover `Δz_singleton` after solving reduced system:

`Δz_singleton = (val*Δx[col] - rhs_row) / h`

Store this so the full `Δz` vector remains available for cone step-to-boundary checks and updates.

---

## 6. Workstream C: IPM step selection, stall recovery, and end-game correctness

### C0. Goal
Stop “α → 0 forever” and stop “μ tiny but residual not done” from burning wallclock.

### C1. Task: Implement combined-step RHS feasibility weighting `(1-σ)`
The design doc specifies weighted residuals in combined step. fileciteturn0file0

Make it a knob:

```rust
pub enum CombinedRhsMode {
    Full,
    ScaledByOneMinusSigma,
}

let feas_weight = match rhs_mode {
    CombinedRhsMode::Full => 1.0,
    CombinedRhsMode::ScaledByOneMinusSigma => (1.0 - sigma).max(0.05),
};
rhs_x = -feas_weight * r_x;
rhs_z = -feas_weight * r_z;
rhs_tau = -feas_weight * r_tau;
```

**Acceptance**
- On BOYD1, when α_aff becomes tiny late, the combined step should become more centering-dominant and avoid s/z-driven α collapse.

---

### C2. Task: Add explicit alpha-stall mode switch
Detect stall:
- `alpha < 1e-6` for 5 consecutive iterations OR
- `dual_res` not improving for 10 iterations while `mu` is already small

Actions:
- force `sigma = 0.99`
- enable `(1-σ)` RHS mode
- increase refinement iters
- reduce static reg if in polish mode

**Snippet**
```rust
if alpha < 1e-6 && stall_count >= 5 {
    sigma = 0.99;
    rhs_mode = CombinedRhsMode::ScaledByOneMinusSigma;
    refine_iters = refine_iters.max(5);
}
```

**Acceptance**
- DUAL2/BOYD1-class failures must stop “spinning” at the end; they must either converge or terminate with a meaningful `NumericalError` (with diagnostics).

---

### C3. Task: Termination on unscaled metrics (strict requirement)
Implement:
- compute `x̄, s̄, z̄ = (x,s,z)/τ`
- undo Ruiz scaling and evaluate `r_p, r_d, gap` on unscaled problem data.

**Implementation skeleton**
```rust
pub fn unscaled_metrics(
    prob: &ProblemData,
    scaling: &RuizScaling,
    state: &HsdeState,
    ws: &mut IpmWorkspace,
) -> Metrics {
    // fill ws.r_p, ws.r_d without allocations
    // compute inf norms + objectives
    Metrics { /* ... */ }
}
```

**Acceptance**
- Diagnostics print the same “rel_p/rel_d/gap_rel” as your bench hook currently does.
- Termination thresholds match user expectations across scaled problems.

---

## 7. Workstream D: Wallclock performance engineering (allocation-free + reuse)

### D0. Goal
Convert “few iterations” into “fast wallclock” by removing overhead.

### D1. Task: Allocation-free `predictor_corrector_step`
Replace per-iteration `Vec` creation with workspace slices.

**Pattern**
```rust
let rhs1 = &mut ws.rhs1[..kkt_dim];
let rhs2 = &mut ws.rhs2[..kkt_dim];
rhs1.fill(0.0);
rhs2.fill(0.0);

// fill RHS values
// call kkt.solve_two_rhs(rhs1, rhs2, ws_solve)
```

**Acceptance**
- `cargo instruments` / perf shows no allocator activity in hot loop.

---

### D2. Task: Build KKT CSC pattern once
Right now KKT assembly uses triplets → CSC each iteration (expensive).

New design:
- Build CSC pattern once (`indptr`, `indices`)
- Keep `values` mutable and update numeric entries in place

**Snippet**
```rust
impl KktCache {
    pub fn update_h_diagonal(&mut self, h_diag: &[f64]) {
        for (i, &ix) in self.h_diag_value_ix.iter().enumerate() {
            self.values[ix] = -h_diag[i];
        }
    }
}
```

**Acceptance**
- KKT rebuild time drops to near zero for NonNeg/QP problems.
- BOYD1 wallclock drops sharply even before convergence improves.

---

### D3. Task: Real multi-RHS solve (avoid permuting twice)
Implement a solve that permutes/triangular-solves two RHS at once:

```rust
pub fn solve_two_rhs_in_place(
    &self,
    rhs1: &mut [f64],
    rhs2: &mut [f64],
    ws: &mut SolveWorkspace,
) {
    self.permute_in(rhs1, &mut ws.rhs_perm1);
    self.permute_in(rhs2, &mut ws.rhs_perm2);

    self.ldl.solve_in_place(&mut ws.rhs_perm1);
    self.ldl.solve_in_place(&mut ws.rhs_perm2);

    self.permute_out(&ws.rhs_perm1, rhs1);
    self.permute_out(&ws.rhs_perm2, rhs2);
}
```

**Acceptance**
- solve time per iteration decreases measurably on medium instances.

---

### D4. Task: SOC scaling without dense materialization in KKT
Current SOC path materializes dense `H` blocks. This is a long-term performance trap.

Short-term:
- preallocate SOC scratch vectors, avoid per-column allocations.
- write dense SOC block into fixed CSC positions directly.

Long-term:
- represent SOC scaling in structured form and avoid dense insertion entirely (requires solver changes).

**Acceptance**
- SOC-heavy problems no longer show O(d²) spikes per iteration due to allocations.

---

## 8. Workstream E: Regression tests + benchmark gates (MOSEK-grade discipline)

### E1. Task: Build a “solver correctness harness”
For each benchmark instance:
- run Minix
- compute unscaled metrics
- assert termination status implies metric bounds

Add “known-hard” instances:
- BOYD1, BOYD2
- DUAL2
- a SOC-heavy case
- a sparse LP-ish case

### E2. Task: Add performance budgets in CI (soft-fail initially)
Track:
- total time
- factorization time
- KKT build time
- allocations count (if you can hook allocator)

Budget examples (illustrative; tune later):
- “No more than X allocations per solve”
- “KKT build < 5% of runtime on QP suite”

---

## 9. Concrete task list (ordered by ROI)

### Phase 1 — Stop correctness pathologies and make BOYD measurable
1. **A4** add KKT accuracy instrumentation
2. **A1** ensure refinement matches shifted KKT
3. **C3** unscaled termination
4. **C1/C2** combined RHS weighting + alpha-stall mode switch
5. **A2** replace static_reg floors with policy
6. **A3** polish mode (end-game)

### Phase 2 — BOYD2 structural win + wallclock
7. **B1/B2/B3** singleton-row elimination
8. **D1** workspace for predcorr
9. **D2** reuse KKT CSC pattern
10. **D3** real two-RHS solve

### Phase 3 — “MOSEK-class” backend quality
11. Add backend trait + integrate at least one serious sparse solver
12. Improve SOC handling (structured scaling)
13. Presolve reductions (fixed vars, redundant rows, etc.)
14. Expand cone support (EXP/POW/PSD) with robust scaling/corrections (per main design doc). fileciteturn0file0

---

## 10. Debugging playbook for BOYD1/BOYD2 (what to print)

When `MINIX_DIAGNOSTICS=1`, print per iteration:

- `mu`, `tau`, `kappa`
- `primal_res`, `dual_res`, `gap_rel`
- `alpha_aff`, `alpha`, `sigma`
- `min(s)`, `min(z)` for NonNeg blocks
- `static_reg_eff`, `dynamic_bumps`
- `kkt_residual_rhs1`, `kkt_residual_rhs2`

Example:

```text
iter=142 mu=4.2e-10 tau=1.0 kappa=...
  rel_p=... rel_d=... gap_rel=...
  alpha_aff=1.2e-3 alpha=9.9e-4 sigma=0.99 mode=POLISH
  reg=1e-10 bumps=3 kkt_res1=2e-11 kkt_res2=3e-11
```

This immediately tells you whether:
- you are solve-limited (KKT residuals large), or
- step-limited (alpha collapses), or
- logic-limited (mode selection wrong).

---

## 11. “Definition of done” for MOSEK-class behavior

Minix can be called “MOSEK-class” only when it consistently exhibits:

### Correctness / robustness
- Does not spin at the end: either converges or returns a meaningful `NumericalError`.
- Correct statuses on infeasible/unbounded instances (HSDE certificates).
- Termination metrics computed on **unscaled** quantities.

### Accuracy
- Polishes dual feasibility when μ is tiny (BOYD1 must finish).
- No “μ tiny but rel_d stuck at 1e-3” on common QP/LP classes.

### Performance
- Hot loop allocation-free
- KKT pattern reused
- Multi-RHS solves implemented
- Structural elimination prevents bound rows from dominating runtime (BOYD2 must not be a 20s solve)

---

## 12. Appendix: Minimal code skeleton for the main solve loop

```rust
pub fn solve_ipm(prob: &ProblemData, settings: &SolverSettings) -> SolveResult {
    let (scaled_prob, scaling) = prescale(prob, settings);
    let mut state = initialize(&scaled_prob, settings);

    let mut ws = IpmWorkspace::new(&scaled_prob, /* m_eff etc */);
    let mut kkt = KktSolver::new(&scaled_prob, settings);

    for iter in 0..settings.max_iter {
        // 1) residuals (scaled)
        compute_residuals(&scaled_prob, &state, &mut ws);

        // 2) scaling update (cone kernels)
        update_scaling(&scaled_prob, &state, &mut ws);

        // 3) factorize (reuse pattern, update values)
        kkt.factor_in_place(&scaled_prob, &state, &ws)?;

        // 4) predictor-corrector step using workspace + two RHS solves
        predictor_corrector_step(&scaled_prob, &mut state, &mut ws, &mut kkt, settings);

        // 5) termination (unscaled)
        let metrics = unscaled_metrics(prob, &scaling, &state, &mut ws);
        if metrics.is_optimal(settings) {
            return finalize_optimal(metrics, state, scaling);
        }
        if metrics.is_infeasible_certificate(settings) {
            return finalize_infeasible(metrics, state, scaling);
        }

        // 6) stall policy / polish mode
        update_modes(&mut state, &metrics, settings);
    }

    finalize_max_iters(state, scaling)
}
```

---

### Notes
- This plan is intentionally concrete and biased toward the failure modes you are seeing now.
- Once BOYD1/BOYD2 are fixed, the remaining work to reach MOSEK-level is primarily “scale to more cones + better factorization backends”.

