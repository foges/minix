# Minix Solver Second-Pass Analysis

## Scope
- Read updated code in `solver-core/src/ipm/predcorr.rs`, `solver-core/src/ipm/hsde.rs`, `solver-core/src/ipm/mod.rs`, `solver-core/src/ipm/termination.rs`, `solver-core/src/scaling/nt.rs`, `solver-core/src/presolve/ruiz.rs`, and `solver-core/src/linalg/kkt.rs`.
- No external research used.

## What’s Improved
- Predictor/corrector KKT RHS now includes the complementarity term (`d_s`), and `ds` is derived from `-d_s - H dz`, which fixes a core inconsistency in the Newton step.
- The combined step now scales feasibility residuals by `(1 - sigma)`, and sigma uses the stable `(1 - alpha_aff)^3` heuristic.
- Ruiz equilibration is implemented and hooked into `solve_ipm`.
- Basic recovery via `push_to_interior` exists, which helps avoid hard failures when iterates drift.
- Dual objective and residual scaling in termination are more consistent with QP structure.

## Remaining Issues / Needed Improvements

### 1) Ruiz scaling breaks cone geometry for SOC (and future PSD/EXP/POW)
- `equilibrate` scales each row independently (`solver-core/src/presolve/ruiz.rs`). This is fine for NonNeg/Zero cones, but **invalid for SOC blocks**, where all components must be scaled together to preserve the cone.
- Effect: the solver can silently solve a *different* problem for SOC, and recovery/step-to-boundary may behave erratically.
- Concrete action: make Ruiz row scaling block-aware: one scalar per cone block (uniform scale for each SOC/PSD block) rather than per-row scaling.

### 2) Unscale mappings for `s` and `z` are inverted
- Current unscaling:
  - `s_original = row_scale * s_scaled`
  - `z_original = z_scaled / (row_scale * cost_scale)`
- Given `A_scaled = R A C`, `b_scaled = R b`, the correct mapping is:
  - `s_original = s_scaled / R`
  - `z_original = cost_scale * R * z_scaled`
- Effect: returned `s` and `z` are wrong even when the solver converges.
- Concrete action: fix `RuizScaling::unscale_s` and `RuizScaling::unscale_z` in `solver-core/src/presolve/ruiz.rs`.

### 3) SOC NT scaling is still approximate, and SOC Mehrotra correction is missing
- `nt_scaling_soc` still uses `w = sqrt(s \circ z)` (`solver-core/src/scaling/nt.rs`), which is only exact when s and z commute.
- Combined-step `d_s` uses the **NonNeg formula** for all cones (`solver-core/src/ipm/predcorr.rs`). This is incorrect for SOC because division by `z_i` and elementwise products are not valid in Jordan algebra.
- Effect: SOC steps can be distorted and convergence will slow or fail on general SOC problems.
- Concrete action:
  - Implement full NT scaling for SOC (the 4-step formula from the design doc).
  - Implement SOC-specific Mehrotra correction using Jordan products and NT scaling (`W`, `lambda`, `eta`).

### 4) HSDE is effectively disabled (tau fixed to 1)
- Both affine and corrector steps force `dtau = 0` (`solver-core/src/ipm/predcorr.rs`). This removes the HSDE dynamics and makes infeasibility detection unreliable.
- Termination also ignores `r_tau`, so the embedding equation is not enforced.
- Effect: infeasible problems may be misclassified or stall; HSDE benefits are lost.
- Concrete action: either (a) fully restore HSDE updates (dtau, dkappa) or (b) formally switch to a non-HSDE PDIP path and update `mu`, termination, and residual checks accordingly.

### 5) `push_to_interior` does not restore SOC interior
- The recovery step only fixes non-positive entries (`solver-core/src/ipm/hsde.rs`). For SOC, interior requires `t > ||x||`, so all components can be positive and still be *outside* the cone.
- Effect: recovery can leave SOC blocks non-interior and cause repeated failures.
- Concrete action: implement cone-specific recovery (e.g., project to SOC interior or reset SOC blocks to scaled unit initialization).

### 6) `mu` definition still includes `tau*kappa` even though `tau` is fixed
- With `tau` fixed, `mu = (s·z + tau*kappa)/(nu+1)` (`solver-core/src/ipm/hsde.rs`) can slow progress vs. the standard `mu = (s·z)/nu` used by non-HSDE IPMs.
- Effect: slower reduction in `mu` and a mismatch with the step-size heuristic.
- Concrete action: if HSDE remains disabled, switch to the standard `mu` formula and update kappa handling accordingly.

### 7) Symbolic factorization reuse still missing
- KKT is refactorized each iteration without reusing a symbolic factorization (`solver-core/src/linalg/kkt.rs`, `solver-core/src/ipm/predcorr.rs`).
- Effect: unnecessary overhead in larger problems; not a correctness issue.
- Concrete action: call `KktSolver::initialize` once and reuse the symbolic structure across iterations.

## Summary
The core direction logic is much improved, and Ruiz scaling plus centering updates are a major step forward. The remaining blockers are mostly (1) cone-structure correctness (SOC scaling and correction), (2) scaling/unscaling math (Ruiz), and (3) the choice to freeze tau, which undermines HSDE’s benefits. Fixing those should yield both correct SOC behavior and faster convergence on harder instances.
