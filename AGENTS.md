# Repository Guidelines

## Project Structure & Module Organization
- `solver-core/` holds the Rust IPM solver implementation (cones, KKT, scaling, IPM loops).
- `solver-bench/` provides benchmark tooling and the Maros-Meszaros runner.
- `solver-py/` contains PyO3-based Python bindings; `solver-ffi/` and `solver-mip/` are planned extensions.
- `_data/` stores datasets and artifacts; `_planning/` contains research notes and patches.
- `target/` and `benches/` are build/generated outputs and should not be edited manually.

## Build, Test, and Development Commands
- `cargo build --release` builds all workspace crates in release mode.
- `cargo test` runs the full Rust test suite across the workspace.
- `cargo test --lib -p solver-core` runs only solver-core library tests.
- `cargo run --release -p solver-bench -- maros-meszaros --limit 10` runs a small benchmark slice.
- `cd solver-py && maturin develop --release` builds and installs the Python extension for local use.

## Coding Style & Naming Conventions
- Rust 2021 edition; indent with 4 spaces and follow standard rustfmt defaults.
- Names follow Rust conventions: `snake_case` for functions/modules, `CamelCase` for types, `SCREAMING_SNAKE_CASE` for constants.
- `solver-core` enables `#![warn(clippy::all)]`; keep new code clippy-clean where practical.
- Prefer explicit error handling and small, focused modules for numerical routines.

## Testing Guidelines
- Unit and integration tests live in `solver-core/tests/` (e.g., `cone_tests.rs`).
- Example-based checks live under `solver-core/examples/`.
- Use `cargo test` for validation; for benchmark regressions, add/refresh runs in `solver-bench/`.
- Property testing is available via `proptest`; approximate comparisons use `approx`.

## Commit & Pull Request Guidelines
- Commit subjects in history are short, imperative verbs (e.g., “Fix …”, “Add …”, “Bump …”).
- Keep commits focused; split solver changes from benchmarks or docs when possible.
- PRs should include: a concise summary, tests run (commands + results), and notes on solver accuracy/perf impact when relevant.
- Link related issues or benchmark artifacts if a change affects convergence or runtime.

## Benchmarking & Comparison

### Maros-Meszaros QP Test Suite
- 136 problems downloaded to `~/.cache/minix-bench/maros-meszaros/*.QPS`
- Run all: `cargo run --release -p solver-bench -- benchmark --suite mm`
- Run single: `cargo run --release -p solver-bench -- benchmark --suite mm --problem HS21`

**Current Results (2026-01-17):**
- Minix: 135/136 (99.3%) solved, ~40s total
- Clarabel: 131/136 (96.3%) solved, ~24s total (but excludes 4 timeouts)
- On 128 problems both can solve: Minix ~16s vs Clarabel ~24s (Minix 1.5x faster)

### Clarabel Comparison Scripts
Located in `/tmp/claude/` (reusable across sessions):
- `bench_clarabel_mm.py` - QPS parser + Clarabel conversion functions
- `run_clarabel_all.py` - Run Clarabel on all MM problems with timeout
- Results saved to `/tmp/claude/clarabel_results.json`

To run Clarabel comparison:
```bash
source solver-py/.venv/bin/activate
python /tmp/claude/run_clarabel_all.py
```

### Tolerance Standards (Clarabel-compatible)
Both solver and benchmark use these formulas:
- `primal_scale = max(1, ||b|| + ||x|| + ||s||)`
- `dual_scale = max(1, ||q|| + ||x|| + ||z||)`
- `gap_rel = |obj_p - obj_d| / max(1, min(|obj_p|, |obj_d|))`
- Default tolerances: `tol_feas=1e-8`, `tol_gap_rel=1e-8`

### Known Hard Problems
- BOYD2: Only unsolved problem - hits MaxIters (200), gap improves but very slowly
- NumLimit problems (QE226, QGFRDXPN, QSC205): Severe numerical cancellation in A^T*z
  - These are accepted via condition-aware acceptance with relaxed tolerances
- CONT-200/201/300: Minix solves in 10-30s, Clarabel times out
