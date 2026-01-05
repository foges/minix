# Minix fine-tooth comb report (Jan 2026)

This report is based on the current `rustfiles` snapshot you provided and the failure symptoms you pasted:

- `nt scaling fallback: blocks=1, s_min=-1.156e8, z_min=2.508e-215, mu=2.321e10, extra_reg=1.000e-5`
- BOYD1 ending at `MaxIters` with huge residuals and huge `μ`.

That **specific diagnostic line is already the smoking gun**: NT scaling for the **nonnegative cone** only “fails” when *some* slack or dual component is **non-positive / non-finite**. In other words: the iterate is leaving the cone, and the solver is continuing anyway.

Below are the issues I found, ordered by “most likely to directly cause your BOYD1 failure”.

---

## Patches produced

- `minix_finetoothe_predcorr.patch`  
  Fixes step-size computation around cone boundaries, adds an interior sanity check before scaling/factorization, and stops silently continuing after NT scaling failure.

- `minix_finetoothe_nt.patch`  
  Makes NonNeg NT scaling use the cone’s scaling-interior test and clamps `s/z` to a safe range to avoid `inf`/`0` scalings.

- `minix_finetoothe_qps_obj_sense.patch`  
  Fixes objective-sense handling for quadratic problems (MAX → MIN must negate both **q and P**, not just **q**).

---

## Issue 1 (Critical): `compute_step_size` ignores α=0, so once any component hits the boundary the solver can take an invalid step

### Where
`solver-core/src/ipm/predcorr.rs` → `fn compute_step_size(...)`

### What happens
The previous code did:

```rust
if alpha_p > 0.0 && alpha_p < alpha { alpha = alpha_p; }
```

That means:
- If a cone returns `alpha_p == 0.0` (typical when `s_i == 0` and `ds_i < 0`), **it gets ignored**.
- If `s` or `z` are *already* non-interior (negative / zero), many cones return `0.0` or even negative values. Those were ignored too.
- Result: `alpha` can remain `∞`, so the function returns `1.0`, and the solver applies a **full step** even though it should take **no step**.

Once *one* component underflows to `0` (or drifts slightly negative), this bug can immediately produce exactly what you’re seeing:
- slack minimum becomes a large negative number (e.g., `s_min=-1e8`)
- scaling fails and you get the “nt scaling fallback” line

### Fix in patch
In `minix_finetoothe_predcorr.patch`:
- Barrier-free cones (e.g. `Zero`) are skipped for step-size restriction.
- For barrier cones, finite α values are respected **including α=0**.
- Negative α is treated as `0` (i.e., “do not move further in that direction”).
- If the direction contains NaN/inf, it returns `0.0` rather than “full step”.

This prevents the solver from stepping *out of the cone* when a component is at/near the boundary.

---

## Issue 2 (Critical): NT scaling failure is treated as “fallback and continue”, allowing non-interior iterates to propagate and explode

### Where
`solver-core/src/ipm/predcorr.rs` in the scaling construction loop:
```rust
let scale = match nt::compute_nt_scaling(...) { ... Err(_) => { ... fallback ... } };
```

### Why it’s a bug
For NonNeg:
- NT scaling fails basically only if `s_i <= 0` or `z_i <= 0` (or non-finite).
- “Fallback and continue” is not a safe recovery; you’re building a KKT system from a point that violates the cone domain.

That’s why your diagnostics show repeated fallback with increasing regularization: the solver is *already off the central path* but continues.

### Fix in patch
In `minix_finetoothe_predcorr.patch`:
- NT scaling failure now returns `Err(...)` with detailed block diagnostics (cone type, offset, dim, s_min, z_min).
- The outer loop in `ipm/mod.rs` already has a recovery path for `Err` (it calls `push_to_interior`).

This is the correct behavior: **stop the step and recover**, don’t proceed with fake scaling.

---

## Issue 3 (High): No “pre-step” interior / finite sanity check (so once state is corrupted, you keep computing with it)

### Where
`solver-core/src/ipm/predcorr.rs` at the top of `predictor_corrector_step(...)`

### Why it matters
Once any of:
- `tau <= 0`
- `kappa <= 0`
- `s` or `z` contains NaN/inf
- any barrier cone slice is non-interior

…then the subsequent KKT system / step-size logic is no longer meaningful.

### Fix in patch
`minix_finetoothe_predcorr.patch` adds:

- `check_state_interior_for_step(...)` which verifies:
  - `tau` and `kappa` positive + finite
  - `x,s,z` all finite
  - for NonNeg and SOC: checks “scaling interior” (not the stricter `is_interior_primal()`), to avoid false positives when components are tiny but valid

If this check fails, the step returns `Err`, and the outer loop recovers.

---

## Issue 4 (High): NonNeg NT scaling computes `s/z` without guarding against overflow/underflow → can produce `inf` scaling entries

### Where
`solver-core/src/scaling/nt.rs` → `fn nt_scaling_nonneg(...)`

### Why it matters
Even when `s` and `z` are positive:
- `s/z` can overflow to `inf` on ill-scaled instances (or underflow to `0`),
- resulting in a KKT matrix with non-finite values,
- which cascades into NaNs and then non-interior iterates.

### Fix in patch
`minix_finetoothe_nt.patch`:
- Uses `NonNegCone::is_interior_scaling(...)` (strictly > 1e-300) before building scaling.
- Clamps `s/z` into `[1e-18, 1e18]`.

This makes KKT assembly significantly more robust on badly scaled problems.

---

## Issue 5 (Medium): Quadratic MAX problems are converted incorrectly (only `q` is negated, not `P`)

### Where
`solver-bench/src/qps.rs`

### Symptom
For QP objectives `0.5 x'P x + q'x`, converting MAX to MIN requires negating **both** the linear and quadratic terms.

The code previously did:
```rust
let q = q * obj_sense;
```
but built `P` unchanged.

### Fix in patch
`minix_finetoothe_qps_obj_sense.patch`:
- multiplies each triplet value in `P` by `obj_sense` too.

Note: If `obj_sense == -1` and `P` is PSD, negating it makes the problem non-convex; your solver likely can’t solve it. But the current behavior was silently wrong either way.

---

## Additional issues found (not patched)

These aren’t necessarily causing your BOYD1 blow-up, but they’re correctness/robustness gaps worth addressing.

### A. Recovery push can be extremely aggressive when μ is huge
`ipm/mod.rs` uses:
```rust
let recovery_margin = (mu * 0.1).max(1e-4);
```
If μ has exploded (like your log shows), this can set slacks/duals to ~1e9 or larger, potentially making the next iter worse. Consider capping the recovery margin.

### B. Dynamic regularization in QDLDL is applied *after* factorization by clamping D
`solver-core/src/linalg/qdldl.rs` modifies D and D_inv after `ldl::factor()` but leaves L unchanged. That corresponds to solving a *different* matrix than the one you assembled (and not the usual “A + δI” regularization). This can significantly distort Newton directions when many pivots get clamped.

### C. Many cones are stubbed / unimplemented
Several cone methods return `Err` or panic (`ExpCone`, `PowCone`, `PsdCone`). That’s fine if you never hit them, but it’s a landmine.

### D. Performance: QPS conversion builds per-row constraints with O(m·nnz) scans
In `solver-bench/src/qps.rs`, the inequality construction loops each row and scans all A triplets each time. This can dominate runtime on large instances.

### E. `SolveInfo` fields are mostly TODO
In `ipm/mod.rs`, many `SolveInfo` fields are returned as `0.0`. That makes diagnostics misleading.

---

## How these fixes relate to your log

Your failure line:
```
nt scaling fallback: blocks=1, s_min=-1.156e8 ...
```
means: **your iterate left the NonNeg cone**.

The two direct enablers in code were:
1) step size logic ignored α=0 (so a boundary component doesn’t constrain α),
2) scaling failure was “handled” by fallback (so the solver continued from an invalid iterate).

The supplied patches remove both failure modes.
