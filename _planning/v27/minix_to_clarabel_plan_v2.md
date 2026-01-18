# Minix → Clarabel performance plan (SOCP-heavy)

This plan assumes the current state is already close (or better) than Clarabel on *many* QP / mixed-cone workloads, but still has two clear gaps:

1) **Detecting infeasible / unbounded SOCPs fast** (Minix sometimes runs many extra iterations).
2) **Iteration-count gap on some SOCP reformulations** (e.g., QP epigraph / PRIMALC8 class).

It also addresses a correctness-quality issue:

3) **Postsolve recovery occasionally degrades reported primal feasibility** (`rel_p` worsens after postsolve).

---

## What “done” looks like

### Performance targets
- **QP / QP+SOC / Portfolio / LASSO**: within ±10% of Clarabel wall time on the existing suite.
- **Pure SOCP infeasible/unbounded**: **within 1.0–1.5×** Clarabel time and (most importantly) **stop in roughly the same iteration regime** (no “runs to max iters” cases).

### Correctness targets
- Postsolve should **never** worsen primal residuals by orders of magnitude when dimensions match.
- Infeasible / unbounded status should be detected reliably for SOC-only instances (at minimum), with no regression on feasible problems.

---

## Phase 0 — Lock in baselines (no solver changes)

**Goal:** Make sure we can attribute wins/regressions.

- Record (per problem):
  - iterations
  - time breakdown: KKT update / factor / solve / misc
  - termination reason and final residuals
- Compare against Clarabel with identical data + scaling settings.

No patch needed (workflow), but do it before/after each phase below.

---

## Phase 1 — Fix the two “obvious” issues (patches included)

### 1A) Postsolve: enforce `Ax + s = b` after recovery

**Symptom:** `rel_p` sometimes increases after postsolve.

**Cause (likely):** the postsolve map recovers `x` and `s` independently; tiny drift in either can show up as a larger primal residual after mapping back.

**Fix:** If dimensions match the original-with-bounds form, recompute:
- `s := b - A x` (in recovered/original space)

This makes reported feasibility consistent and eliminates a whole class of “postsolve regression” noise.

**Patch:** `0002_postsolve_recompute_slack.patch`

---

### 1B) Enable SOC dual-cone checks for infeasibility certificates

**Symptom:** SOCP infeasible problems can’t be certified (or take far too long), because `dual_cone_ok(...)` returns `false` for SOC cones.

**Fix:** Add SOC membership test to `dual_cone_ok`:
- For SOC: `t >= ||x||` (with tolerance)

This unblocks *primal infeasible* certification for SOC-only problems.

**Patches:**
- `0003_soc_infeasibility_detection.patch` (ipm2 path)
- `0004_soc_infeasibility_detection_ipm.patch` (legacy ipm path)

---

## Phase 2 — Make infeasible/unbounded detection match Clarabel

Once SOC dual-cone membership is wired in, the next step is to ensure we stop at the same point Clarabel does.

### 2A) Add cone checks for the *dual infeasible* certificate (optional but recommended)
Today Minix checks:
- `P x ≈ 0`
- `A x + s ≈ 0`
- `qᵀx < 0`

It does **not** validate `s ∈ K` as part of the certificate. For SOC/NN this is easy and cheap, and makes the certificate more robust.

**Outcome:** fewer false negatives/positives; easier to relax “when to check”.

### 2B) Earlier “kappa-dominant” trigger (if needed)
If we still see cases where Minix runs many more iterations than Clarabel on infeasible/unbounded:
- Trigger certificate checks when `kappa/(tau+kappa)` is high (ratio threshold), instead of waiting for very small absolute `tau`.

**This is only necessary if Phase 1 doesn’t close the gap**, but it’s a common reason HSDE solvers differ on “infeasible speed”.

Patch TBD (depends on observed traces).

---

## Phase 3 — Close iteration-count gaps on SOCP reformulations

The remaining iteration gaps (e.g., PRIMALC8-type) are usually *algorithmic*, not KKT-performance.

### 3A) Prefer rotated SOC epigraphs for quadratic terms
Many QPs are turned into SOCPs via an epigraph constraint. Using **RSOC** often improves conditioning and reduces iterations vs. the “standard SOC via (t±1)” trick.

Options:
- Implement RSOC as a first-class cone (best long-term, bigger change).
- Or introduce a presolve transformation that preserves equivalence but improves scaling (smaller change, less general).

**Acceptance:** PRIMALC8 iteration count close to Clarabel (e.g., 20 → ~8), without regressions on other SOCPs.

Patch TBD (this is a larger feature).

### 3B) SOC neighborhood / centrality control that doesn’t collapse alpha
If we enable SOC neighborhood checks, we must pair them with:
- sigma escalation / MCC-style correction when the direction is “too affine”
- no brittle upper-bound that crushes alpha
- stable spectral math (already started with stable `λ₂` handling)

This can improve iteration count *and* reduce “stall” behavior near optimality.

Patch TBD (requires careful tuning + regression testing).

---

## Phase 4 — Clean up: tests + guardrails

- Add regression tests:
  - Postsolve `Ax+s=b` consistency (Phase 1A)
  - SOC infeasibility certificate triggers on known infeasible SOCPs (Phase 1B)
- Add debug counters:
  - “certificate attempted / succeeded”
  - “postsolve slack recomputed”
- Keep the optimizations behind settings if needed, but aim to make good behavior default.

---

## Patch list (apply in order)

1. `0002_postsolve_recompute_slack.patch`
2. `0003_soc_infeasibility_detection.patch`
3. `0004_soc_infeasibility_detection_ipm.patch`

