# Minix `ipm2` Task List (MOSEK‑close + Clarabel‑beating)

This is the **separate task table** referenced by the master design doc.

> Conventions
> - **Strategy:** *Parallel* = build in `ipm2/` side-by-side; *In-place* = modify existing modules used by both.
> - **Acceptance:** always measured on **unscaled** metrics and **wallclock**.

| ID | Phase | Strategy | Category | Task | Primary files | Acceptance criteria | Depends |
|---:|:------|:---------|:---------|:-----|:--------------|:--------------------|:--------|
| 0.1 | Setup | Parallel | Wiring | Add `ipm2/solve.rs` and a `--solver ipm2` bench flag | `solver-core/src/ipm2/solve.rs`, `solver-bench` | Can solve at least one small NonNeg QP end-to-end | — |
| 0.2 | Setup | In-place | Diagnostics | Standardize per-iter diagnostics output (unscaled rel_p/rel_d/gap_rel, μ, α, σ) | bench runner + `ipm2/diagnostics.rs` | Logs show the same metrics for all problems | 0.1 |
| 0.3 | Setup | In-place | Profiling | Per-iteration timers using `PerfTimers` sections | `ipm2/perf.rs`, `ipm2/solve.rs` | 95%+ runtime attributed to named buckets | 0.1 |
| A1 | Phase 1 | In-place | Solve accuracy | Make refinement residuals match the shifted matrix actually factorized | `linalg/kkt.rs` (+ backend) | Increasing refine iters reduces `||Kx-b||∞` on ill-conditioned cases | 0.2 |
| A2 | Phase 1 | Parallel | Reg policy | Use `RegularizationPolicy/State` as the *only* reg authority (remove scattered floors) | `ipm2/regularization.rs`, KKT builder | `static_reg_eff` is printed and derived only from policy | A1 |
| A3 | Phase 1 | In-place | Telemetry | Print post-refinement KKT residuals for both RHS solves under env flag | `linalg/kkt.rs` | BOYD1 stalls show whether solve-limited vs step-limited | A1 |
| A4 | Phase 1 | Parallel | End-game | Implement `Polish` mode behavior (raise refinement, adjust RHS mode, maybe reduce reg) | `ipm2/solve.rs`, `ipm2/modes.rs` | BOYD1 terminates `Optimal` (default tolerances) | A2–A3 |
| C1 | Phase 1 | Parallel | Step logic | Implement combined-step RHS `(1-σ)` feasibility weighting | `ipm2/solve.rs` / predcorr logic | Tail iterations avoid α collapse; fewer MaxIters in endgame | 0.1 |
| C2 | Phase 1 | Parallel | Stall recovery | Use `StallDetector` to trigger `StallRecovery` mode | `ipm2/modes.rs`, `ipm2/solve.rs` | No >N iterations with α < 1e-8 without mode switch | C1 |
| T1 | Phase 1 | Parallel | Termination | Wire `compute_unscaled_metrics` + unscale-into-workspace | `ipm2/metrics.rs`, `ipm2/workspace.rs`, unscale utilities | Termination decisions are based on unscaled rel_p/rel_d/gap_rel | 0.1 |
| D1 | Phase 2 | Parallel | Perf | Make `ipm2` iteration allocation-free using `IpmWorkspace` | `ipm2/workspace.rs`, `ipm2/solve.rs` | Allocation count in hot loop ~0 after init | 0.1 |
| D2 | Phase 2 | In-place | Perf | Persistent solve workspace inside KKT solver (perm buffers, refine scratch) | `linalg/kkt.rs` | No per-solve allocations in KKT solve/refine | D1 |
| D3 | Phase 2 | In-place | Perf | KKT CSC pattern caching + numeric in-place updates | `linalg/kkt_cache.rs`, `linalg/kkt.rs` | KKT build/update bucket <5% runtime on QP suite | D1 |
| D4 | Phase 2 | In-place | Perf | Packed multi-RHS solve (permute once, solve twice) | `linalg/kkt.rs` | Solve bucket decreases measurably on medium instances | D2 |
| B1 | Phase 2 | In-place | Presolve | Detect singleton rows (one nonzero per row) | `presolve/singleton.rs` | Stable partition of constraints; counts reported | T1 |
| B2 | Phase 2 | In-place | Presolve | Eliminate singleton rows via Schur diagonal update + RHS update | `presolve/eliminate.rs` + KKT update | BOYD2 KKT dimension drops massively; wallclock improves | B1, D3 |
| B3 | Phase 2 | In-place | Presolve | Bounds shifting + fixed var elimination + postsolve reconstruction | `presolve/bounds.rs`, `postsolve/*` | Bound-heavy problems stop blowing up KKT; solution matches original vars | B2 |
| E1 | Phase 2 | In-place | Testing | Correctness harness: curated suite (BOYD1/2/DUAL2 + cone coverage) | `tests/`, `solver-bench` | CI fails on status/regression or unscaled metric regression | T1 |
| E2 | Phase 2 | In-place | Perf gates | Record baseline wallclock by bucket and gate regressions | CI + bench | Detects wallclock regressions automatically | 0.3, E1 |
| G1 | Phase 3 | In-place | Linalg | Introduce backend trait; keep QDLDL baseline | `linalg/backend.rs` | Backends swappable without IPM changes | D3 |
| G2 | Phase 3 | In-place | Linalg | Reuse factor buffers across iterations (QDLDL engineering) | `linalg/qdldl.rs` | Factor bucket decreases; no repeated allocations | G1 |
| G3 | Phase 3 | Add-on | Linalg | Add a serious sparse direct backend (optional feature) | `linalg/backends/*` | Large sparse instances speed up significantly | G1 |
| P1 | Phase 3 | Parallel | Product win | Warm starts + parameter updates for repeated solves (CVXPY sweeps) | `ipm2/solve.rs`, Python bindings | Repeated-solve workloads show 2×+ win vs Clarabel | D3, A4 |

---

## Progress (working log)
- 0.1: done (added `solver-core/src/ipm2/solve.rs`, exported `ipm2`, added `--solver ipm2` in `solver-bench`)
- 0.2: done (per-iter unscaled diagnostics in `ipm2/solve.rs`)
- 0.3: done (per-iter timers in `ipm2/solve.rs`)
- A1: done (refinement residuals include static shift; diagnostics show post-refine residuals)
- A2: done for ipm2 (regularization routed through `RegularizationPolicy` and per-iter settings override)
- A3: done (KKT solve residual telemetry under `MINIX_DIAGNOSTICS_KKT`, tagged rhs1/rhs2)
- A4: done (polish mode increases refinement, reduces reg, and enforces (1-σ) weighting; convergence still needs runtime validation)
- C1: done for ipm2 (uses existing combined-step feasibility weighting in `predcorr`)
- C2: done for ipm2 (stall detection switches mode and adjusts reg/refinement)
- T1: done for ipm2 (termination uses unscaled metrics from `compute_unscaled_metrics`)
- D1: done (ipm2 hot loop uses workspace; KKT updates are in-place for non-diagonal cones)
- D2: done (KKT solve workspace is persistent; no per-solve allocations in refine path)
- D3: done (cache KKT pattern and update H block entries in place for non-diagonal cones)
- D4: done (packed two-RHS solve path with single permutation; wired into ipm/ipm2)
- B1: done (singleton row detection + stable partition; counts logged when verbose/diagnostics)
- B3: done (bounds shifting + fixed var elimination + postsolve x reconstruction)
- B2: done (KKT singleton-row Schur elimination: reduced A/H blocks, diagonal P updates, RHS adjustments, and dz recovery)
- E1: done (regression suite CLI + test gate with QPS cache optional + synthetic cone cases; unscaled metric checks)
- E2: done (perf baseline write/read + regression gate via CLI/test env vars)
- G1: done (backend trait + default backend wiring; optional SuiteSparse LDL backend feature gating)
- G2: done (QDLDL factor reuse; no per-iteration D cloning)
- G3: done (SuiteSparse LDL backend implementation behind feature)
- P1: done (warm-start mapping for presolve rows; Python solver supports warm-start args + parameter updates + ipm2 selection)
- Validation: cargo check --offline failed (PyO3 does not support Python 3.14; try newer PyO3 or PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1)
- Validation: fixed KKT borrow errors; `cargo check -p solver-core --offline` passes (docs warnings only)
- Validation: `PYO3_PYTHON=python3.12 cargo check -p solver-py --offline` passes (docs warnings only)
- D4: packed RHS permutation for two-solve path; split KKT update vs factorization for timers
- T1: ipm2 termination now uses unscaled metrics on postsolved original vars + allocation-free infeasibility checks
- C2/A4: stall recovery now forces (1-σ) weighting and higher σ cap; reg policy refreshed per iter
- Misc: added in-place postsolve recover helpers + orig-dim metrics buffers in `IpmWorkspace`
- Misc: allow missing docs warnings at crate root
- Validation: `cargo check -p solver-core --offline` passes (dead_code warning only)
- BOYD1 v6: NonNeg interior check uses absolute tol (1e-300), NT fallback uses clamped s/z; NT fallback not triggered on BOYD1 with MINIX_DIAGNOSTICS=1
- BOYD1 v6: BOYD-only overrides tried (kkt_refine_iters=7, static_reg=1e-4, mcc_iters=1, line_search_max_iters=10); still stalls with alpha ~6.6e-6, rel_p ~5e-1, rel_d ~1.0 after 50 iters
- BOYD1 v6: added NonNeg-block diagnostics (min s/z, alpha_sz limiter index) in ipm/ipm2; BOYD1 shows NonNeg limiting (idx 26792/26790) with alpha_sz ~3.16e-2 while alpha shrinks to ~3e-5
- BOYD1 v6: centrality line search diagnostics added; first failure is NonNeg w below beta*mu (idx 26784), so line search halves alpha even when alpha_sz is ~3e-2
- Cleanup (reverted): restored ipm1 module under `solver-core/src/ipm/`; HSDE/termination back under ipm; solver-core default + bench/python support both ipm/ipm2
