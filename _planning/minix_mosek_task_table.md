# Minix → “MOSEK‑close” task breakdown

**Recommendation:** **build off the existing codebase**, *do not restart from scratch*.  
However, do the large refactors behind a **parallel implementation track** (e.g., `ipm2/` module or a feature flag) so you can A/B against the current solver and avoid destabilizing progress.

---

## Task table

> Columns:
> - **Build strategy**: *In-place* (edit existing modules), *Add-on* (new modules + minimal wiring), *Parallel* (new `ipm2/` path that runs side-by-side).
> - **Acceptance criteria**: should be checked on *unscaled* metrics and measured in *wallclock*.

| ID | Phase | Build strategy | Area | Task | Primary code location(s) | Acceptance criteria | Depends on |
|---:|:------|:---------------|:-----|:-----|:--------------------------|:--------------------|:-----------|
| 0.1 | Setup | Parallel | Project hygiene | Create `ipm2/` (side-by-side) path so refactors can be staged safely | `solver-core/src/ipm2/*`, minimal wiring in `lib.rs` or `ipm/mod.rs` | `ipm2` can solve at least one small QP end-to-end using existing linear algebra + cone kernels | — |
| 0.2 | Setup | In-place | Benchmarking | Standardize diagnostics output (unscaled `r_p`, `r_d`, `gap`, `μ`, `α`, `σ`) behind `MINIX_DIAGNOSTICS=1` | `solver-bench`, `ipm/*` | For every run, you can print the same metrics; BOYD1 shows `rel_p`, `rel_d`, `gap_rel`, `μ` every N iters | — |
| 0.3 | Setup | In-place | Profiling | Add per-iteration timers (residuals / scaling / KKT update / factor / solves / termination) | `ipm/mod.rs`, `linalg/kkt.rs` | You can attribute >95% runtime into named buckets | — |
| A1 | Phase 1 | In-place | Correctness | Termination checks on **unscaled** quantities after undoing Ruiz scaling | `ipm/termination.rs`, `ipm/mod.rs`, `presolve/ruiz.rs` | Diagnostics and termination agree (no “scaled OK but unscaled bad”); exit statuses match expectations | 0.2 |
| A2 | Phase 1 | In-place | Linear solve accuracy | Fix iterative refinement so residuals are computed against the **same shifted matrix** actually solved | `linalg/kkt.rs`, `linalg/qdldl.rs` | Increasing `kkt_refine_iters` decreases `||Kx-b||∞` reliably on ill-conditioned cases | 0.2 |
| A3 | Phase 1 | In-place | Regularization | Replace ad-hoc static-reg floors with a centralized `RegularizationPolicy` | `ipm/mod.rs`, new `ipm/regularization.rs` | BOYD1 can polish `rel_d` lower than before (≥ 2–3 orders) without instability; reg values are reported | A2 |
| A4 | Phase 1 | In-place | Robustness | Add KKT residual diagnostics (post-refinement) for both RHS solves | `linalg/kkt.rs`, `ipm/mod.rs` | When a solve stalls, logs clearly show whether it is solve-limited vs step-limited | 0.2 |
| A5 | Phase 1 | In-place | End-game | Implement **Polish mode** when `μ` is tiny but feasibility not met (raise refinement, adjust reg, adjust RHS mode) | `ipm/mod.rs`, `ipm/predcorr.rs` | BOYD1 terminates `Optimal` with default tolerances, not `MaxIters` | A1–A4 |
| C1 | Phase 1 | In-place | Step logic | Implement combined-step feasibility weighting `(1-σ)` as a knob (design-doc parity) | `ipm/predcorr.rs` | Tail iterations stop driving `α` to ~0 on “polishing” problems; fewer MaxIters near optimum | A1 |
| C2 | Phase 1 | In-place | Stall recovery | Alpha-stall detector + mode switch (force centering, raise refinement, switch RHS mode) | `ipm/mod.rs`, `ipm/predcorr.rs` | DUAL2/BOYD1-type runs never spend >X iters with `α < 1e-8` without switching modes | C1 |
| D1 | Phase 2 | Add-on | Performance | Add `IpmWorkspace` so predictor–corrector and termination are allocation-free | new `ipm/workspace.rs`, refactor `predcorr.rs`, `termination.rs` | Allocation count in hot loop ~0 (after init); wallclock improves measurably on medium instances | 0.3 |
| D2 | Phase 2 | In-place | Performance | Add persistent `SolveWorkspace` inside `KktSolver` (no per-solve allocations) | `linalg/kkt.rs` | Solver runs show minimal allocator activity in KKT solve | D1 |
| D3 | Phase 2 | In-place | Performance | Reuse KKT CSC pattern; update numeric values in place (no `TriMat -> CSC` per iter) | `linalg/kkt.rs`, new `linalg/kkt_cache.rs` | KKT build/update bucket <5% runtime on QP suite | D1 |
| D4 | Phase 2 | In-place | Performance | True multi-RHS solve path (permute once, solve twice) | `linalg/kkt.rs`, backend trait optional | Solve bucket decreases measurably; total time decreases on medium instances | D2 |
| D5 | Phase 2 | In-place | Performance | SOC scaling: eliminate per-column allocations; write into fixed CSC positions | `linalg/kkt.rs`, `cones/soc.rs` or scaling code | SOC-heavy instances no longer show O(d²) spikes from allocation/materialization | D1 |
| B1 | Phase 2 | In-place | Structure | Detect singleton (identity-like) constraint rows | `presolve/*`, `problem.rs` | Produces a stable partition of rows: `A = [A_gen; A_singleton]` | A1 |
| B2 | Phase 2 | In-place | Structure | Eliminate singleton rows from KKT via diagonal Schur updates and RHS updates | `linalg/kkt.rs` or `presolve/eliminate.rs` | BOYD2 KKT dimension reduces massively; solve time drops dramatically | B1, D3 |
| B3 | Phase 2 | In-place | Presolve | Add bounds shifting and fixed-variable elimination (avoid “bounds as rows” blowup) | `presolve/bounds.rs`, `postsolve/*` | Large-bound problems stop exploding KKT size; recovered solution matches original vars | B2 |
| E1 | Phase 2 | In-place | Testing | Correctness harness: cross-check unscaled metrics, statuses, and certificates on curated suite | `solver-bench`, `tests/` | CI fails if regression in status or unscaled residuals/gap | A1 |
| E2 | Phase 2 | In-place | Testing/Perf | Perf gates: record time buckets, allocations, and compare to baselines | CI + `solver-bench` | Detects wallclock regressions automatically | 0.3, E1 |
| G1 | Phase 3 | Add-on | Linear algebra | Introduce `LinearSolverBackend` trait (QDLDL baseline) | `linalg/backend.rs`, `linalg/qdldl.rs` | Backends swappable without changing IPM logic | D3 |
| G2 | Phase 3 | In-place | Linear algebra | Reuse QDLDL numeric factor buffers across iterations | `linalg/qdldl.rs` | Factorization bucket decreases; no repeated allocations in factorize | G1 |
| G3 | Phase 3 | Add-on | Linear algebra | Add a “serious” sparse direct backend (CHOLMOD / Pardiso / etc. depending on licensing) | `linalg/backends/*` | Large sparse instances show big wallclock improvements vs QDLDL | G1 |
| G4 | Phase 3 | In-place | Linear algebra | Ordering improvements (AMD / METIS option); deterministic tie-breaks | `linalg/ordering.rs` | Better fill-in and speed on large cases; runs are deterministic | G1 |
| H1 | Phase 3 | In-place | Parallelism | Parallelize cone kernels + residual computation with deterministic chunking | `cones/*`, `ipm/*` | Speedup on multi-core; results stable run-to-run | D1 |
| H2 | Phase 3 | Add-on | Parallelism | Optional parallel factorization (via backend) | `linalg/backends/*` | Clear speedup on large problems when enabled | G3 |
| F1 | Phase 4 | In-place | Cones | EXP/POW cones: primal/dual interior tests, dual-map oracle, BFGS scaling, 3rd-order correction + tests | `cones/exp.rs`, `cones/pow.rs`, scaling modules | Pass finite-diff derivative tests; solve EXP/POW benchmark set robustly | E1 |
| F2 | Phase 4 | In-place | Cones | PSD cone dense path + chordal decomposition | `cones/psd.rs`, `presolve/chordal.rs` | Can solve medium SDP sets; chordal reduces runtime/memory | E2 |
| M1 | Optional | Parallel | Strategy | Hybrid algorithm selection (IPM vs first-order) for huge low-accuracy cases | `solve.rs` dispatcher | Stops losing wallclock to SCS-like methods on giant instances | E2 |

---

## Start fresh or build on existing?

**Build on existing.** The current codebase already has:
- HSDE/IPM iteration structure
- cone kernel scaffolding
- sparse KKT factorization hooks
- CVXPY integration plumbing

Starting from scratch would delay learning and reintroduce the same numerical pitfalls.

**But**: do the largest refactors (workspace, KKT cache, backend trait) either:
- behind an `ipm2/` module, or
- behind feature flags (`--features ipm2`)

so you can continuously compare against the current solver on the same suites.

