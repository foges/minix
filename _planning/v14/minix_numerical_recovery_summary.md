# NumericalError robustness tweaks (IPM2)

This patch targets the remaining `NumericalError` exits by making two “be conservative instead of dying” changes:

## 1) Robust `dtau` update (HSDE scalar step)

File: `solver-core/src/ipm2/predcorr.rs`

`compute_dtau()` previously returned an error when the denominator was judged ill‑conditioned:
- `|denom| <= 1e-10 * scale`

That error bubbles up as a failed predictor–corrector step and can quickly trip the **consecutive failure** limit, exiting with `Status::NumericalError`.

**Change:** when the denominator is ill‑conditioned, we now **skip the tau update** by returning `Ok(0.0)` (no-op `dtau`) instead of erroring.

Rationale:
- In practice, this scalar update comes from a nearly-singular 2×2 coupling; when it’s unreliable, damp/skip is a standard IPM robustness trick.
- The step still updates the main primal/dual directions; tau/kappa can recover on subsequent iterations.

## 2) Failure-driven “numeric recovery” ramp (regularization + refinement)

File: `solver-core/src/ipm2/solve.rs`

Added a small state variable:
- `numeric_recovery_level` (capped at `MAX_NUMERIC_RECOVERY_LEVEL = 6`)

Behavior:
- On any step failure (`predictor_corrector_step_in_place` returning `Err`) **or** when `mu` becomes non-finite/huge, increment `numeric_recovery_level`.
- On the next iteration, if `numeric_recovery_level > 0`:
  - Multiply static KKT regularization by `10^level` (clamped to `reg_policy.static_reg_max`)
  - Add `2*level` extra iterative refinement iterations (clamped to `reg_policy.max_refine_iters`)
  - Force more conservative step parameters (`feas_weight_floor=0`, `sigma_max=0.999`)
- On a successful step, reset `numeric_recovery_level` back to 0.

Rationale:
- Many `NumericalError` cases are “just barely failing” factorizations/solves. Increasing regularization and refinement *only when needed* often flips these into solvable (possibly slower) cases.
- Because polish exists, we can accept slightly more regularization during recovery and still clean up accuracy later.

## Suggested local testing

1. Run the previously-problematic suite and count `N (` again:
   - `cargo run --release -p solver-bench -- maros-meszaros --limit 150 2>&1 | tee /tmp/bench.txt | tr '\r' '\n' | grep ' N (' | wc -l`

2. Spot-check that the regression suite remains green:
   - `cargo run --release -p solver-bench -- regression`

3. If you want to see when numeric recovery triggers, enable diagnostics logging in your settings (the patch logs `numeric recovery: ...` via `DiagnosticsLogger`).

## Extra diagnostics

When `MINIX_DIAGNOSTICS=1` (or equivalent diagnostics enabling), step failures now print the underlying error string:
- `predictor-corrector step failed at iter ...`

This makes it much easier to distinguish KKT factorization issues vs scalar-update pathologies.
