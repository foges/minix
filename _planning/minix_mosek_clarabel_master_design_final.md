# Minix “ipm2” Master Design Doc
## MOSEK‑close and *significantly faster than Clarabel* (wallclock)

_Last updated: 2026‑01‑04_

This is a **single, detailed, followable** design doc that combines:

- the deep, code-oriented plan from the original “MOSEK‑class” doc fileciteturn0file1  
- the milestone/definition-of-success framing from v2 fileciteturn0file0  
- the algorithm spec and “equation-complete” guidance from the main project design doc fileciteturn0file3  
- and the **actual `ipm2/` scaffolding that already exists** (workspace, metrics, stall detector, timers, regularization policy). fileciteturn0file2  

The intent is that you can implement this **incrementally** while continuously A/B testing against the current solver. You should **not scrap** the existing solver until `ipm2` is demonstrably superior on your benchmark gates.

---

## 0) North star and success criteria

### 0.1 The competitive goal
You asked for:

- **MOSEK‑close** behavior (robust, accurate, stable)  
- **a significant wallclock lead over Clarabel** (or MOSEK, whichever is faster) on your target workloads.

This is achievable only if we do *both*:
1) make the solver **robust** in finite precision (no BOYD1 tail‑spin, correct statuses, good certificates), and  
2) make each IPM iteration **cheap enough** that “few iterations” converts into wallclock wins.

### 0.2 Definition of “MOSEK‑close” (falsifiable)
We will use the following milestone definitions (adapted from v2):

**Milestone M1 (LP/QP/SOCP, symmetric cones only):**
- **Robustness:** status agreement with MOSEK on ≥ 99% of instances in a curated suite where MOSEK returns a status.
- **Accuracy:** termination on **unscaled** residuals and gap at default tolerances (e.g. 1e‑8 feasibility and gap).
- **Wallclock:** geometric mean within **2×** of MOSEK on that suite (single-thread, comparable tolerances).  
  For small/medium problems (where overhead dominates), target **≤ 1.2×** of MOSEK.

**Milestone M2 (beat Clarabel):**
- On the same suite + your CVXPY workloads, show a **significant wallclock lead** over Clarabel:
  - target: **≥ 1.5×** faster in geometric mean, and
  - **≥ 2×** faster on at least one meaningful “bucket” (e.g., bound-heavy QPs, repeated solves, or high-accuracy solves).
- **Rule:** we measure **wallclock**, not iterations.

**Milestone M3 (nonsymmetric cones + PSD):**
- EXP/POW robust and stable (correct oracles, scaling, 3rd-order correction).
- PSD with chordal decomposition (for sparse PSD).
- Then: aim for “MOSEK‑close” within **2–3×** on cone-heavy suites (harder).

> Why Clarabel lead is plausible: you can win via **structural elimination** (singleton/bound rows), **KKT caching**, **backend choice**, and **repeated-solve ergonomics**—not only via algorithmic novelty.

---

## 1) Strategy: keep the old solver as baseline; build `ipm2` as the future

### 1.1 Do *not* restart from scratch
Your current codebase already contains:
- the right algorithm family (HSDE + predictor–corrector IPM),
- cone plumbing and scaling machinery,
- linear algebra infrastructure,
- Python/CVXPY integration.

Starting over deletes your safety net and slows iteration.

### 1.2 The correct “freshness” move: parallel implementation track
You already have an `ipm2/` scaffolding bundle. fileciteturn0file2  
Treat `ipm2` as the staging ground for “MOSEK-ish engineering”:

- `ipm1`: keep as baseline
- `ipm2`: new architecture (workspace, timers, stall/polish, unscaled metrics), gradually absorbing the best pieces
- A/B gate every change against:
  - correctness suite
  - performance suite
  - BOYD1/BOYD2
  - and your real CVXPY workloads.

---

## 2) Canonical problem form and HSDE recap (what we’re solving)

We solve the standard conic QP form (your design doc’s canonical internal form): fileciteturn0file3

\[
\min_{x,s}\ \frac12 x^\top P x + q^\top x\quad
\text{s.t.}\ Ax + s = b,\ \ s\in K
\]

HSDE state uses \((x,s,z,\tau,\kappa)\), and when \(\tau>0\) we recover:

\[
\bar x=x/\tau,\quad \bar s=s/\tau,\quad \bar z=z/\tau.
\]

### 2.1 Residuals (scaled-domain)
Define (in the scaled problem):
- \(r_x = P x + A^\top z + q\tau\)
- \(r_z = A x + s - b\tau\)
- \(r_\tau = \frac{1}{\tau}x^\top P x + q^\top x + b^\top z + \kappa\)

These drive the affine and combined Newton RHS.

### 2.2 Condensed KKT and the “two-solve strategy”
Use the quasi-definite condensed KKT system and compute \(\Delta x, \Delta z\) via two RHS solves (your design doc §5.4.1). fileciteturn0file3

This is non-negotiable for speed:
- factorize KKT once per iteration
- solve for 2 RHS vectors using triangular solves (or packed multi-RHS).

---

## 3) `ipm2` scaffolding: what already exists and how to use it

You already have these `ipm2` modules: fileciteturn0file2

- `ipm2/workspace.rs`: `IpmWorkspace` (preallocated vectors)
- `ipm2/metrics.rs`: `compute_unscaled_metrics` (allocation-free, given unscaled x̄,s̄,z̄)
- `ipm2/modes.rs`: `StallDetector` and `SolveMode` (Normal/StallRecovery/Polish)
- `ipm2/regularization.rs`: `RegularizationPolicy/State`
- `ipm2/diagnostics.rs`: env-driven logging (`MINIX_DIAGNOSTICS`, etc.)
- `ipm2/perf.rs`: `PerfTimers` with scoped sections

### 3.1 How `ipm2` becomes the real solver: add `ipm2/solve.rs`
The main missing file is the orchestrator.

**New file to create:** `solver-core/src/ipm2/solve.rs`

Responsibilities:
1) prescale (Ruiz + presolve transforms)
2) initialize HSDE state (strict interior)
3) allocate `IpmWorkspace`, timers, diagnostics config
4) run IPM iteration loop:
   - residuals (scaled)
   - scaling update (cone kernels)
   - update KKT numeric values (cached CSC pattern)
   - factorize
   - solve two RHS
   - assemble step
   - step-to-boundary / backtracking
   - update state
   - compute **unscaled** metrics (via `compute_unscaled_metrics`)
   - termination/certificates
   - stall/polish mode selection and reg policy updates

### 3.2 `ipm2` main loop skeleton (code)
Below is a target structure that *uses existing ipm2 code*.

```rust
use crate::ipm2::{
    DiagnosticsConfig, IpmWorkspace, PerfSection, PerfTimers,
    RegularizationPolicy, StallDetector, SolveMode,
    compute_unscaled_metrics,
};

pub fn solve_ipm2(prob: &ProblemData, settings: &SolverSettings) -> SolveResult {
    // 0) prescale + presolve
    let (scaled, ruiz) = prescale_ruiz(prob, settings);

    // 1) init HSDE state (x,s,z,tau,kappa) strictly interior
    let mut st = hsde_initialize(&scaled, settings);

    // 2) ipm2 scaffolding
    let diag = DiagnosticsConfig::from_env();
    let mut timers = PerfTimers::default();
    let mut stall = StallDetector::default();

    let reg_policy = RegularizationPolicy::default();
    let mut reg_state = reg_policy.init_state(/*scale=*/1.0);

    let (n, m) = (scaled.num_vars(), scaled.num_constraints());
    let mut ws = IpmWorkspace::new(n, m);

    // 3) KKT caching + solver (to be implemented in Workstream D)
    let mut kkt = KktSystem::new_cached(&scaled, settings);

    for iter in 0..settings.max_iter {
        // Residuals
        {
            let _g = timers.scoped(PerfSection::Residuals);
            compute_residuals_scaled(&scaled, &st /*, write into ws scratch */);
        }

        // Scaling update (H blocks)
        {
            let _g = timers.scoped(PerfSection::Scaling);
            update_scaling_blocks(&scaled, &st /* cone kernels, write into scaling state */);
        }

        // KKT numeric update + factorize
        {
            let _g = timers.scoped(PerfSection::KktUpdate);
            kkt.update_numeric(&scaled, &st, reg_state.static_reg_eff);
        }
        {
            let _g = timers.scoped(PerfSection::Factorization);
            kkt.factorize()?;
        }

        // Predictor–corrector step:
        // - fill ws.rhs1/ws.rhs2 without allocation
        // - solve 2 RHS
        // - compute Δτ and combine
        {
            let _g = timers.scoped(PerfSection::Solve);
            predictor_corrector_two_solve(&scaled, &mut st, &mut ws, &mut kkt, settings);
        }

        // Compute unscaled x̄,s̄,z̄ into ws.x_bar/ws.s_bar/ws.z_bar
        // (divide by tau, undo Ruiz scaling)
        unscale_into_workspace(prob, &ruiz, &st, &mut ws);

        // Unscaled termination metrics
        let metrics = {
            let _g = timers.scoped(PerfSection::Termination);
            compute_unscaled_metrics(
                &prob.A, prob.P.as_ref(), &prob.q, &prob.b,
                &ws.x_bar, &ws.s_bar, &ws.z_bar,
                &mut ws.r_p, &mut ws.r_d, &mut ws.p_x,
            )
        };

        if diag.should_log(iter) {
            eprintln!(
                "iter={iter} rel_p={:.3e} rel_d={:.3e} gap_rel={:.3e} mode=? reg={:.1e}",
                metrics.rel_p, metrics.rel_d, metrics.gap_rel, reg_state.static_reg_eff
            );
        }

        // Termination + certificates (unscaled)
        if is_optimal(&metrics, settings) {
            return finalize_optimal(prob, &ruiz, &st, metrics, timers);
        }
        if let Some(status) = check_certificates(prob, &ruiz, &st, settings) {
            return finalize_status(prob, &ruiz, &st, status, metrics, timers);
        }

        // Stall/polish mode selection
        let mode = stall.update(/*alpha=*/st.last_alpha, /*mu=*/st.mu, metrics.rel_d, settings.tol_feas);
        match mode {
            SolveMode::Normal => {}
            SolveMode::StallRecovery => {
                // e.g. force σ high, switch RHS weighting, raise refinement
                enable_stall_recovery(&mut st, &mut reg_state);
            }
            SolveMode::Polish => {
                reg_policy.enter_polish(&mut reg_state);
                enable_polish_mode(&mut st);
            }
        }
    }

    finalize_max_iters(prob, &ruiz, &st, timers)
}
```

This is the core “glue” that ties your algorithm spec to the `ipm2` code you already have.

---

## 4) The real reasons you’re not Clarabel/MOSEK yet (and how we close them)

Your benchmark observations (BOYD1/BOYD2, DUAL2 endgame) are typical of a solver that:

- is “mostly correct” on the macro level (μ/gap drop, primal feasible),
- but hits a **finite-precision accuracy floor** in the KKT solve and/or step selection,
- and burns wallclock because hot-loop engineering isn’t finished.

### 4.1 BOYD1 class (“μ tiny, rel_d stalls”)
Root causes tend to be:
- KKT solve accuracy (refinement mismatch, too-large regularization, insufficient refinement)
- no explicit end-game polish mode
- bad feasibility weighting in combined step late

### 4.2 BOYD2 class (“structure punished”)
Root causes tend to be:
- modeling/presolve: bounds encoded as rows and not eliminated
- KKT dimension ballooning and conditioning

### 4.3 Clarabel lead: where you can realistically win
Clarabel is strong, so “beat by margin” requires **leverage**:

- **Structural elimination** (singleton/bounds rows → diagonal Schur updates)
- **KKT CSC caching** (no rebuilds, numeric update only)
- **Packed multi-RHS solves**
- **Polish mode** to avoid endgame spin
- **Repeated-solve support** (warm starts + param updates) for CVXPY workloads

---

## 5) Workstreams (detailed and implementable)

This section is the “how” with concrete changes, code snippets, and acceptance tests.

### Workstream A — KKT solve quality + regularization discipline (BOYD1)
**Goal:** eliminate the accuracy floor that blocks dual feasibility polishing.

#### A1) Make iterative refinement mathematically correct
If the factorization solves \((K + \epsilon I)x=b\), refinement residual must be computed against the **same** matrix.

**Implementation detail**
- Store `static_reg_eff` (and any dynamic bump info if applicable) on the KKT solver.
- During `Kx` matvec, add the diagonal shift term.

```rust
symm_matvec_upper(kkt_unshifted, &x, &mut kx);
for i in 0..kkt_dim {
    kx[i] += static_reg_eff * x[i];
}
res[i] = rhs[i] - kx[i];
```

**Acceptance**
- On BOYD1, increasing `refine_iters` must decrease `||Kx-b||∞` reliably.

#### A2) Centralize regularization policy (stop scattering floors)
Use `ipm2::RegularizationPolicy` as the *only* place that decides:
- effective static regularization
- dynamic minimum pivot
- polish mode adjustments

The policy already exists in `ipm2/regularization.rs`. fileciteturn0file2

**What to change in the solver**
- Delete ad-hoc `if sparse_qp { static_reg = max(static_reg, ...) }` logic in random files.
- Instead, compute a scale factor once per iteration and call:

```rust
let scale = estimate_kkt_scale(&scaled, &scaling_state);
reg_state.static_reg_eff = reg_policy.effective_static_reg(scale);
```

**Acceptance**
- You can print `static_reg_eff` each iteration; it changes only via the policy.
- BOYD1 can polish more (lower `rel_d`) without instability.

#### A3) Solve-accuracy telemetry (so you stop guessing)
Under `MINIX_DIAGNOSTICS_KKT=1`, print:
- `||Kx-b||∞` for rhs1/rhs2 after refinement
- `static_reg_eff`, dynamic bump count
- refine iters

This should be added to `KktSystem::solve_two_rhs_in_place()`.

**Acceptance**
- When something stalls, logs tell you if you’re solve-limited.

#### A4) “Polish mode” (end-game)
Use `StallDetector` and `SolveMode::Polish` (already in `ipm2/modes.rs`). fileciteturn0file2

**Trigger**
- `mu < polish_mu_thresh` and `rel_d > polish_dual_mult * tol_feas`

**Behavior**
- increase refinement iters (up to `policy.max_refine_iters`)
- optionally reduce static reg toward `policy.polish_static_reg`
- switch RHS weighting to `(1-σ)` mode
- optionally do a feasibility-focused Newton step

**Acceptance**
- BOYD1 must terminate `Optimal` at default tolerances (not MaxIters).
- DUAL2 endgame must not spin with α≈0 for hundreds of iterations.

---

### Workstream B — Presolve + structure elimination (BOYD2)
**Goal:** avoid KKT blowups due to bounds/singletons.

#### B1) Presolve transform stack with postsolve
Implement a transform stack that records enough to reconstruct:
- original x from reduced x
- original residuals/certificates

This is MOSEK‑style “presolve + postsolve”.

#### B2) Bounds shifting (eliminate lower bounds without adding rows)
Instead of encoding both lower and upper bounds as rows, shift variables:
- if `x_i >= l_i`, replace `x_i = t_i + l_i`, with new variable `t_i >= 0`.
This reduces RHS magnitude and helps conditioning.

#### B3) Singleton row elimination (identity-like rows)
Detect constraints rows with exactly one nonzero in A (typical for bounds and simple equalities). Then eliminate them from KKT with a diagonal Schur update:

For row `r` with entry `val * x[col] + s[r] = b[r]`:
- add to effective P diagonal: `P[col,col] += val^2 / H[r]`
- update RHS accordingly

**Acceptance**
- BOYD2 KKT dimension drops sharply.
- wallclock improves drastically on bound-heavy instances.

---

### Workstream C — Step logic + stall recovery (DUAL2/BOYD endgame)
**Goal:** stop α collapse / feasibility stall.

#### C1) Combined-step RHS weighting `(1-σ)`
Your design doc recommends `(1-σ)` feasibility weighting for combined RHS. fileciteturn0file3  
Implement it as a knob in `predcorr`:

```rust
let feas_weight = (1.0 - sigma).max(0.05);
rhs_x = -feas_weight * r_x;
rhs_z = -feas_weight * r_z;
rhs_tau = -feas_weight * r_tau;
```

#### C2) Explicit stall-recovery mode
When `SolveMode::StallRecovery` triggers:
- force σ close to 1 (centering)
- enable `(1-σ)` RHS weighting
- increase refinement iters
- optionally adjust regularization slightly (but do not “hack” H)

**Acceptance**
- if α becomes tiny, solver switches modes quickly and either converges or exits meaningfully.

---

### Workstream D — Wallclock performance engineering (beat Clarabel)
**Goal:** reduce per-iteration cost enough to win wallclock.

#### D1) Allocation-free hot loop (use `IpmWorkspace`)
`ipm2::IpmWorkspace` already preallocates the main vectors. fileciteturn0file2  
The remaining work is to route all per-iteration routines to use these buffers and stop allocating.

**Acceptance**
- allocator activity in the iteration loop ~0 after init.

#### D2) KKT CSC caching (no `TriMat -> CSC` rebuilds)
Build KKT sparsity once, then update numeric values only.

Key data structure (from the detailed doc):
- store CSC `indptr`, `indices`, `values`
- store index maps for:
  - `H` diagonal positions (and optionally SOC dense block positions)
  - regularization diagonal positions

**Acceptance**
- KKT build/update becomes <5% of runtime on QP suite.

#### D3) Packed multi-RHS solve
Two-solve strategy needs 2 RHS vectors. Implement:
- permute both RHS once,
- triangular solve both,
- unpermute both once.

**Acceptance**
- solve time per iter decreases measurably.

#### D4) Reuse factorization workspaces
Ensure QDLDL (or any backend) reuses:
- numeric factorization buffers
- work arrays
- diagonal index maps for shifts

**Acceptance**
- factorization bucket decreases; no per-iter allocations inside factorize.

---

### Workstream E — Benchmarking gates (you can’t optimize what you can’t measure)
**Goal:** make “beat Clarabel / close to MOSEK” a continuous, measurable target.

#### E1) Standard suite buckets
Maintain separate buckets:
- small/medium/large QP (Maros–Mészáros)
- SOC heavy
- bound-heavy
- ill-conditioned
- CVXPY real workloads

#### E2) Always record time breakdown
Use `ipm2/perf.rs` timers. fileciteturn0file2  
Log per solve:
- residuals, scaling, KKT update, factorization, solve, termination

#### E3) CI gates (soft → hard)
Start with soft gates (warn on regressions), then hard gates:
- correctness status/regression
- performance budgets per bucket

---

## 6) The “Clarabel lead” playbook (what to implement first)

If the *explicit* goal is “significant margin over Clarabel”, the highest ROI path is:

1) **Eliminate singleton/bound rows** (BOYD2 and similar)  
2) **KKT caching + no allocations** (per-iteration cost collapse)  
3) **Multi-RHS solve** (two-solve strategy becomes real)  
4) **Polish mode** (stop tail spinning and wasting wallclock)  
5) **Warm starts + param updates** for CVXPY workloads (repeated solves)

This combination is how you create a “meaningful bucket” where you can be 2×+ faster.

---

## 7) What remains for “true MOSEK parity”
After M1/M2 are achieved, MOSEK parity comes from:

- richer presolve reductions (redundant rows, scaling heuristics, etc.)
- more advanced sparse linear algebra backends (supernodal, multithread)
- full cone coverage (EXP/POW/PSD) and the kernel math from the main design doc fileciteturn0file3  
- extensive testing and “boring” robustness engineering

`ipm2` is the right vehicle for this, but it will be an iterative journey.

---

## Appendix A) Mapping: doc tasks → `ipm2` modules

| Need | `ipm2` module | What exists | What to add |
|------|---------------|------------|-------------|
| Allocation-free buffers | `workspace.rs` | ✅ `IpmWorkspace` | add cone scratch buffers and direction buffers as needed |
| Unscaled metrics | `metrics.rs` | ✅ `compute_unscaled_metrics` | wire `unscale_into_workspace()` (divide by τ, undo Ruiz) |
| Stall/polish logic | `modes.rs` | ✅ `StallDetector` | integrate with step-size/σ logic; track α, μ |
| Regularization | `regularization.rs` | ✅ policy + state | connect to KKT update, dynamic pivot info |
| Diagnostics config | `diagnostics.rs` | ✅ env-based | add structured logging options (JSONL) |
| Perf timing | `perf.rs` | ✅ `PerfTimers` | emit time breakdown into SolveInfo |

---

## Appendix B) “Where to start tomorrow” (implementation order)

1) Create `ipm2/solve.rs` and route the solver-bench runner to call it behind a flag.
2) Wire unscaled termination metrics using `compute_unscaled_metrics` (already done).
3) Add KKT solve residual telemetry.
4) Implement polish mode + stall recovery (hook `StallDetector`).
5) Implement KKT CSC caching (largest wallclock win).
6) Implement singleton row elimination (largest structural win).
7) Implement packed multi-RHS solves and buffer reuse.

Once those are in, you will have a solver that:
- stops wasting 20s on BOYD1 endgames,
- doesn’t explode on BOYD2 structures,
- and starts competing seriously on wallclock (which is what you care about).

