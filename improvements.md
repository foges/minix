# Minix IPM Solver — Convergence & Robustness Improvements (DUAL* Stalls)

This document is a **thorough, actionable** set of changes to address the behavior you observed:

- Minix is **very fast** on some Maros–Mészáros problems (e.g. LOTSCHD / DPKLO1),
- but **hits max iterations** (or stalls with `alpha → 0`) on the **DUAL*** family,
- where **primal/dual residuals improve** but the **“gap” / complementarity** does not.

It is written as a **fix list + engineering plan**, aligned with the solver design doc’s HSDE + predictor–corrector approach.

---

## 0. TL;DR prioritized fix list (do these first)

### P0 — Correctness bugs / HSDE coupling (very likely causes of the stall)

1. **Apply the τ-direction in the *combined/corrector* step** (two-solve strategy)
   - After computing `dtau`, you must do:
     - `dx ← dx1 + dtau * dx2`
     - `dz ← dz1 + dtau * dz2`
   - The affine step does this; the combined step currently does **not**.
   - Impact: the combined direction becomes inconsistent with the HSDE Newton system and can stall or drive step sizes to ~0.

2. **Stop “freezing τ” for LPs / P=None**
   - HSDE τ/κ dynamics are needed for LPs too.
   - If dtau is gated off (e.g. by an `is_qp` check), τ stays ≈1 and κ/μ can stop decreasing.

3. **Use ONE step size α for (x,s,z,τ,κ)**
   - Compute α to keep **all interior variables** positive: `(s,z,τ,κ)`.
   - Do **not** update κ with a different alpha (and do not clamp κ after the step).
   - Using a smaller `alpha_for_kappa` breaks the Newton step and can prevent μ from reaching tolerance.

### P0 — Instrumentation that prevents misdiagnosis

4. **Fix the “gap” that is printed in verbose mode for QPs**
   - For QPs, `|qᵀx + bᵀz|` is **not** the primal-dual gap. It will generally converge to `|xᵀPx|` at optimality.
   - Print:
     - `gap_obj = |pobj - dobj|`
     - `gap_comp = sᵀz` (and separately `tau*kappa`)
   - Otherwise you can think you’re stuck even when you aren’t, or miss when κ is the true blocker.

---

## 1. What “DUAL* stalls” look like in an HSDE IPM

When the implementation is correct, on feasible problems you typically see:

- `||r_x||`, `||r_z||` decrease,
- `sᵀz → 0`,
- `τ stays > 0` and `κ → 0`,
- μ decreases smoothly (often superlinearly late in solve),
- and α usually stays “reasonable” (often ~0.5–1.0 after the first few iters).

Your pattern (“residuals improve, gap/μ stuck, α → 0”) is consistent with:

- **τ/κ coupling not being applied correctly**, so the step solves the wrong linearized system, or
- **κ not decreasing** because it is updated with a different step size, or
- **centrality breakdown** on a degenerate LP/QP where basic predictor–corrector needs extra recentering (multiple centrality corrections / neighborhood control).

---

## 2. P0: Fix HSDE τ/κ dynamics so the algorithm matches the design doc

### 2.1 Apply the τ-direction in the combined step (critical)

**Where:** `solver-core/src/ipm/predcorr.rs` (combined step section)

**Symptom it causes:** the solver “sort of” reduces residuals, but cannot reduce complementarity reliably; α gets driven down; τ may remain fixed; κ may behave erratically.

**Fix:**
- After you compute `dtau` for the combined step, you must update the already-solved directions with the precomputed `(dx2, dz2)`.

You already have helper `apply_tau_direction(...)` used for the affine step. Use it in the combined step too.

**Expected improvement:**
- DUAL* family should stop “mysteriously stalling” because the combined step actually satisfies
  `PΔx + AᵀΔz + qΔτ = ...` and `AΔx - HΔz - bΔτ = ...`
  instead of implicitly assuming `Δτ = 0`.

---

### 2.2 Remove / relax gating that disables dtau (especially for LPs)

If there is logic like:

- “only compute dtau for QPs”,
- or “dtau only when P is present”,

that is **not** HSDE anymore. HSDE dynamics matter for LPs and conic problems with `P = 0`.

**Fix options (ranked):**

1. **Always compute dtau**, but add guardrails:
   - if denominator is too small / not finite:
     - treat as a numerical failure and trigger the existing recovery path, or
     - bump regularization and refactor, then retry.

2. If you must gate dtau, gate it on **denominator conditioning**, not on “QP vs LP”.

---

### 2.3 Step-size selection must include κ (and must be unified)

**Where:** `solver-core/src/ipm/predcorr.rs`

Right now the code:
- computes `alpha` from cone step-to-boundary + τ positivity,
- then computes a second `alpha_for_kappa`,
- updates κ with a different step and clamps it if needed.

This breaks the Newton step consistency and can stall μ.

**Fix:**
- Extend step-size selection to include κ exactly like τ:
  - if `dkappa < 0`, then `alpha_kappa = 0.99 * (-kappa / dkappa)`, else `inf`
- Final alpha:
  - `alpha = min(alpha_s, alpha_z, alpha_tau, alpha_kappa)`
- Update all variables using the **same** alpha:
  - `x,s,z,tau,kappa`.


### 2.3.1 Minimal patch sketch (combined step)

**Goal:** Make the combined step match the design doc’s §5.4.1 two-solve update:
\[
\Delta x = \Delta x_1 + \Delta\tau\,\Delta x_2,\quad
\Delta z = \Delta z_1 + \Delta\tau\,\Delta z_2
\]
and use a *single* `alpha` that preserves positivity for **s, z, τ, κ**.

In `solver-core/src/ipm/predcorr.rs` around the combined-step dtau computation (roughly near the current `dtau = ...` block):

```rust
// 1) Solve for dx1,dz1 (already done): kkt.solve(..., &mut dx, &mut dz);

// 2) Compute dtau from Schur complement (do NOT gate this on "is_qp")
let dtau = compute_dtau(numerator_corr, denominator, state.tau, denom_scale);

// 3) Apply tau direction so dx,dz become the FULL Newton direction
apply_tau_direction(&mut dx, &mut dz, dtau, &dx2, &dz2);

// 4) Now compute ds using the UPDATED dz (since ds depends on dz)
compute_ds_from_complementarity(...);

// 5) Compute dkappa direction (BEFORE step-size selection)
let dkappa = -(d_kappa_corr + state.kappa * dtau) / state.tau;

// 6) Compute alpha INCLUDING tau and kappa
let mut alpha = compute_step_size(&state.s, &ds, &state.z, &dz, cones, 0.99);
if dtau < 0.0 { alpha = alpha.min(0.99 * (-state.tau / dtau)); }
if dkappa < 0.0 { alpha = alpha.min(0.99 * (-state.kappa / dkappa)); }
alpha = alpha.min(1.0);

// 7) Update ALL variables with the SAME alpha
state.x[i]     += alpha * dx[i];
state.s[j]     += alpha * ds[j];  // skip Zero cone as now
state.z[j]     += alpha * dz[j];
state.tau      += alpha * dtau;
state.kappa    += alpha * dkappa;
```

If you do only one thing to fix DUAL* stalls: do **(2) + (3) + (5) + (6) + (7)**.


**If κ would cross ≤ 0 frequently:**
- That’s a signal the direction is “too aggressive” (or numerical issues in dtau/dkappa).
- Use:
  - stronger centering (clip σ up),
  - multiple centrality corrections (see §4.1),
  - or a backtracking line search with a neighborhood condition.

---

### 2.4 Make μ and “gap” consistent with HSDE

For HSDE:
- μ should be computed as
  - `mu = (sᵀz + τκ) / (ν + 1)`  (already done)
- For a feasible optimum:
  - `τ > 0`, `κ → 0`, `sᵀz → 0`.

**Add debug checks (in verbose / debug mode):**
- Print:
  - `tau`, `kappa`, `tau*kappa`, `s_dot_z`, `mu`
- Assert monotonic “trend” checks (soft warnings):
  - if `alpha < 1e-8` for ~10 iters, log a “stagnation” warning and dump a snapshot.

This quickly tells you whether the stall is:
- κ refusing to decrease, or
- s/z approaching boundary too fast, or
- a linear solve stability issue (KKT residuals not improving).

---

## 3. P0: Fix reporting and termination metrics (prevents false conclusions)

### 3.1 Fix verbose gap for QP

In `solver-core/src/ipm/mod.rs`, the printed `gap = |qᵀx̄ + bᵀz̄|` is correct for LP, but **not** for QP.

For QP, the primal-dual objective gap is:

- `gap_obj = |pobj - dobj|`
- which equals `|x̄ᵀP x̄ + qᵀx̄ + bᵀz̄|`.

**Recommendation: print three gaps:**
- `gap_obj` (objective gap)
- `gap_comp = s̄ᵀ z̄` (complementarity in recovered variables)
- `gap_hsde = gap_comp + τκ / τ²` (optional, but makes τκ visible)

This will immediately reveal whether:
- the algorithm is actually converging but your logged metric was wrong, or
- the method is truly stuck.

---

### 3.2 Align termination residual scaling with “best practice” metrics

Termination currently uses scaled residuals (good), but ensure:
- primal feasibility uses `||A x̄ + s̄ - b|| / max(1, ||b||)`
- dual feasibility uses `||P x̄ + Aᵀ z̄ + q|| / max(1, ||q||)`
- objective gap uses `max(1,|pobj|,|dobj|)` for relative scaling.

If any residual uses the wrong denominator vector (e.g., b_norm vs q_norm), fix it to avoid:
- early termination on some problems,
- or never terminating on others.

---

## 4. P1: Add “top-tier” IPM techniques that prevent α → 0 stagnation

Once P0 fixes are applied, if DUAL* still shows stalls (common for degenerate LP/QP), these are the next levers.

### 4.1 Multiple Centrality Corrections (Gondzio MCC)

**When to use:** when:
- α becomes tiny,
- complementarity stops decreasing,
- iterates drift out of a central neighborhood.

**Idea:** after the predictor direction, take **multiple inexpensive correction steps** using the *same KKT factorization* to re-center the iterate before the main step. This is widely used in robust IPM implementations.

**Implementation sketch:**
- After affine step, compute a candidate combined step.
- If centrality measure is bad (e.g. max(sᵢ zᵢ)/μ too large), perform up to `k` corrections:
  - modify only the complementarity RHS terms
  - solve KKT again (triangular solves only)
  - accumulate the correction direction
- Stop when centrality is acceptable or correction budget hit.

**Why it helps your case:**
- DUAL* problems often behave like **degenerate LPs** where basic Mehrotra may produce directions that are too “boundary seeking”; MCC keeps the iterates inside the neighborhood and prevents α collapse.

---

### 4.2 Neighborhood control / line search (symmetric-cone version)

Modern solvers often enforce a symmetric neighborhood, e.g.:

- For NonNeg: enforce `sᵢ zᵢ ∈ [(1-θ)μ, (1+θ)μ]` for all i, or a norm-based variant.

If violated:
- backtrack α (or increase σ), and retry.

This is especially helpful when:
- you see α dropping late in iterations,
- or s and z become extremely unbalanced.

---

### 4.3 Adaptive σ beyond `(1-α_aff)^3`

Your design doc already mentions the Clarabel-style `σ=(1-α_aff)^3`, which is good and stable.

Still, two improvements are common:

1. **Use `σ = (μ_aff/μ)^3` when μ_aff is reliable**, otherwise fallback to `(1-α_aff)^3`.
2. **Clip σ** into `[σ_min, σ_max]` (e.g. `[1e-3, 0.999]`) to avoid extreme behavior.

This can reduce “flip-flopping” where the method alternates between:
- too aggressive (bad centrality, α tiny)
- too centered (slow progress).

---

### 4.4 Iterative refinement for the KKT solve

If the KKT solve is slightly inaccurate, the IPM can look like it is “making progress on residuals but not on gap”, because complementarity is very sensitive.

Add 1–3 rounds of iterative refinement:

- solve KKT for Δ,
- compute linear system residual `r = RHS - KΔ`,
- solve KKT again for correction `δ`,
- set `Δ ← Δ + δ`.

This is inexpensive once factorization is available, and is a standard robustness feature in high-end solvers.

---

## 5. P1: Regularization strategy improvements (especially for equality-heavy problems)

DUAL* has many equalities + inequalities; this can make KKT systems very ill-conditioned.

### 5.1 Dynamic regularization tied to step quality

Trigger increased regularization when:
- α is consistently tiny,
- dtau denominator is near zero,
- LDL pivot is tiny,
- or refinement detects large residual after solve.

Strategy:
- bump `static_reg` / `dynamic_reg_min_pivot` geometrically,
- refactor,
- retry the same iteration.

### 5.2 Separate treatment for LP (P = 0)

LP KKT is particularly sensitive because the (1,1) block is “just εI”. You already bumped ε for sparse P, which is good.

Still consider:
- a dedicated LP path that forms a Schur complement in the constraint space, or
- stronger diagonal perturbations on the equality rows to preserve quasi-definiteness.

---

## 6. P2: Starting point & scaling upgrades (helps hard instances and stability)

### 6.1 Better HSDE starting point for τ/κ

Even with τ=κ=1, you can do better:

- choose τ, κ so that the initial residuals (r_x, r_z, r_tau) are balanced,
- avoid starting with τκ dominating μ (which creates a floor for μ until κ drops).

If you add warm-start later, this becomes even more important.

### 6.2 Make Ruiz scaling “aware” of equality vs inequality blocks

You already enforce uniform scaling within SOC/PSD/EXP/POW blocks (good).

For equality blocks (Zero cone), consider:
- not scaling them as aggressively as inequality rows,
- or scaling them with a different target norm, because they act like “hard” constraints.

This can reduce numerical issues on equality-heavy problems.

---

## 7. Regression tests & debugging checklist (so this doesn’t come back)

### 7.1 Add a “DUAL2 regression” test

Add a test that:
- loads DUAL2,
- runs Minix with deterministic settings,
- asserts:
  - status == Optimal,
  - μ < tol,
  - τ > 1e-6,
  - κ < 1e-8,
  - primal/dual residuals < tol,
  - objective matches a reference solver to 1e-5 or so.

### 7.2 Add invariant checks inside the IPM loop (debug builds)

- `s ∈ int(K)` and `z ∈ int(K*)` after each update.
- `tau > 0`, `kappa > 0` always.
- dtau direction should not be silently ignored in combined step.
- `mu_new` should be finite and usually decrease.

### 7.3 Log “why α is small”

When α < 1e-6, print:
- min step components:
  - α_s, α_z, α_tau, α_kappa
- cone block and index that hit the boundary
- min(s), min(z), and the corresponding ds/dz entries

This pinpoints whether:
- one inequality row is degenerately scaling,
- dtau/dkappa is too aggressive,
- or a KKT solve produced a bad direction.

---

## 8. Longer-term: matching “top-of-the-line” solver stacks

Once DUAL* is fixed, the next tier to reach “Clarabel/MOSEK-like” behavior is:

- robust homogeneous embedding everywhere,
- NT scaling for symmetric cones,
- BFGS scaling + third-order correction for exp/pow,
- aggressive-but-safe step selection with neighborhood control,
- iterative refinement and smart regularization,
- optional warm-start (especially valuable for MIP node solves).

These are exactly the components highlighted in the design doc and in modern solver literature.

---

## 9. Reference implementations / literature pointers

(Links placed in code blocks to keep them copy-paste friendly.)

```text
Clarabel (IPM, homogeneous embedding, NT scaling, QP support):
https://arxiv.org/abs/2405.12762

MOSEK homogeneous & self-dual algorithm (LP/conic optimizer docs):
https://docs.mosek.com/latest/rmosek/solving-conic.html

MOSEK whitepaper: homogeneous self-dual model for LP:
https://docs.mosek.com/whitepapers/homolo.pdf

MOSEK exponential cone paper (homogeneous model + NT + Mehrotra PC):
https://docs.mosek.com/whitepapers/expcone.pdf

SCS paper (operator splitting on homogeneous self-dual embedding):
https://stanford.edu/~boyd/papers/pdf/scs_long.pdf

Gondzio 1996: Multiple centrality corrections (MCC):
https://link.springer.com/article/10.1007/BF00249643

Ruiz 2001: matrix equilibration (Ruiz scaling):
https://www.numerical.rl.ac.uk/media/reports/drRAL2001034.pdf
```

---

## Appendix A: “What to check first” quick diagnostic script idea

For a failing problem (e.g. DUAL2), print per-iteration:

- `alpha`
- `||r_x||`, `||r_z||`, `|r_tau|`
- `s_dot_z`, `tau*kappa`, `mu`
- `pobj`, `dobj`, `gap_obj`
- `tau`, `kappa`, `dtau`, `dkappa`
- `min(s)`, `min(z)`

If:
- `tau` never moves and/or `tau*kappa` doesn’t go down → HSDE dynamics are effectively disabled.
- `tau` moves but `dx,dz` do not include dtau correction → missing `apply_tau_direction` in combined step.
- `kappa` decreases with a different alpha than other variables → unify α to include κ.
