# BOYD1/BOYD2 convergence: what’s still broken + next fix

## Symptom recap

- **BOYD1**: `μ` and `gap_rel` collapse to ~1e-10 / 1e-9, and `rel_p` is essentially 0, but **`rel_d` stalls ~1e-3** and the solver hits `MaxIters`.
- **BOYD2**: behaves like a huge LP / “almost-LP” (very sparse `P`), and either `μ`/gap don’t move or dual feasibility stalls.

This pattern (“gap/primal OK, dual feasibility stalls”) is a classic sign of **KKT solve bias / over-regularization** (especially when the solver reports tiny complementarity but not feasibility).

## Root causes I’d focus on

### 1) Static regularization floor is way too large for sparse/LP-like problems

`solver-core/src/ipm/mod.rs` currently uses a heuristic that bumps `static_reg` to **1e-4** for LP / sparse-`P` systems.

That is *orders of magnitude* larger than typical IPM defaults (Clarabel’s docs show a default static regularization of **1e-8** and dynamic regularization threshold around **1e-13**). citeturn2view0

When `P≈0` (LP-like), adding `εI` is required for quasi-definiteness, but **if `ε` is too large** it becomes a “ridge” term that can:
- cap attainable dual feasibility (stationarity error floors at roughly `ε * ||x||`),
- make the solver look “optimal” in gap while never reaching feasibility tolerances.

That aligns eerily well with BOYD1: you’re seeing a dual residual that is “small-ish” but refuses to go below ~1e-3 relative.

### 2) The “extra_reg” hack is biasing the Newton system near the boundary

In `predcorr.rs`, there’s logic that detects `min(s,z) < μ/100` and **adds a uniform `extra_reg` directly to the NT scaling `H`**.

This is not a standard stabilization trick (and it changes the Newton equations), so it can easily create the exact failure mode you’re seeing:
- `μ` keeps shrinking (because complementarity is being pushed),
- feasibility (especially dual feasibility) stalls because the linear system is no longer the correct Jacobian.

### 3) Default dynamic pivot floor is extremely aggressive

Your default `dynamic_reg_min_pivot` is **1e-7**, which is huge if it triggers (it effectively perturbs the factorization/solve).
Clarabel’s dynamic regularization threshold default is much smaller (**1e-13**). citeturn2view0

Even if your dynamic reg is implemented differently, the default floor should be *much* smaller than 1e-7.

## Proposed fix (patch)

The attached patch does three things:

1) **Removes the LP/sparse-`P` 1e-4 static_reg floor** and replaces it with a small floor (`1e-8`).
2) **Deletes the “extra_reg modifies H” path** in `predcorr.rs`.
3) Adjusts defaults closer to Clarabel:
   - `static_reg: 1e-8`
   - `dynamic_reg_min_pivot: 1e-13`
   - `kkt_refine_iters: 2`

## How to validate quickly

Run:

- `BOYD1` with diagnostics:
  - expect `rel_d` to keep decreasing and eventually pass tol, or at least move *significantly* below ~1e-3.
- `BOYD2`:
  - expect `μ` to start collapsing reliably (no longer stuck at O(1e-3) if tolerances require more).

If you still get end-game “alpha collapses due to s/z boundary” after this:
- that’s the next layer (centrality neighborhood / MCC / polishing),
- but it’s much easier to reason about *after* removing large regularization bias.
