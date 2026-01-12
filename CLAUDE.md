# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
cargo build --release              # Build all crates
cargo test                         # Run all tests
cargo test --lib -p solver-core    # Run solver-core unit tests only
cargo test -p solver-core test_foo # Run a single test by name

# GPU backends (feature-gated)
cargo build -p solver-core --features metal --release   # Metal (macOS)
cargo build -p solver-core --features cuda              # CUDA (NVIDIA)
cargo test -p solver-core --features metal --release -- --test-threads=1

# Benchmarks
cargo run --release -p solver-bench -- maros-meszaros --limit 10
cargo run --release -p solver-bench -- maros-meszaros --problem HS21
cargo run --release -p solver-bench -- regression
```

## Python Bindings

```bash
cd solver-py
source .venv/bin/activate  # or: uv venv && source .venv/bin/activate
maturin develop --release
python -c "import minix; print(minix.version())"
```

## Project Structure

```
solver-core/     # Core IPM solver (the main library)
solver-bench/    # Benchmark runner (Maros-Meszaros QP test set)
solver-py/       # Python bindings via PyO3
solver-mip/      # Mixed-integer extension
solver-ffi/      # C FFI bindings
_planning/       # Design docs and ablation analysis
```

## Architecture

### Solver Entry Point
`solve()` in `solver-core/src/lib.rs` routes to `ipm2::solve_ipm2()` - the active implementation.

### IPM Pipeline (ipm2/)
1. **solve.rs** - Main solve loop, orchestrates presolve → IPM iterations → postsolve
2. **predcorr.rs** - Predictor-corrector interior point method with Mehrotra heuristics
3. **workspace.rs** - Pre-allocated iteration workspace (avoids allocations in hot loop)
4. **regularization.rs** - Adaptive KKT regularization policy
5. **polish.rs** - Solution polishing via active-set detection
6. **solve_normal.rs** - Normal equations fast path for tall LP/QP (m >> n)

### Cones (cones/)
Each cone implements the `Cone` trait (`traits.rs`):
- **zero.rs** - Equality constraints (s = 0)
- **nonneg.rs** - Non-negative orthant (s ≥ 0)
- **soc.rs** - Second-order cone (Lorentz cone)
- **psd.rs** - Positive semidefinite matrices
- **exp.rs** - Exponential cone (for entropy/logistic)
- **pow.rs** - Power cone (for geometric programming)

### Linear Algebra (linalg/)
- **kkt.rs** - KKT system assembly (augmented system form)
- **qdldl.rs** - Default LDL factorization via QDLDL
- **sparse.rs** - CSC sparse matrix utilities
- **backends/** - Alternative factorization backends:
  - `metal/` - Apple Silicon GPU via Metal (feature: `metal`)
  - `cuda/` - NVIDIA GPU via cuDSS dynamic loading (feature: `cuda`)

### Preprocessing (presolve/)
- **ruiz.rs** - Ruiz equilibration for matrix conditioning
- **bounds.rs** - Variable bound extraction
- **singleton.rs** - Singleton row/column elimination

### Scaling (scaling/)
- **nt.rs** - Nesterov-Todd scaling for symmetric cones
- **bfgs.rs** - BFGS-based scaling for nonsymmetric cones (exp, pow)

## Problem Format

```
minimize    (1/2) x'Px + q'x
subject to  Ax + s = b
            s ∈ K
```
where K is a Cartesian product of cones (Zero, NonNeg, SOC, PSD, Exp).

## Key Types

- `ProblemData` - Input problem (P, q, A, b, cones)
- `SolverSettings` - Tolerances, max iterations, verbosity
- `SolveResult` - Solution (x, s, y, z), status, objective value
- `ConeSpec` - Cone specification enum (Zero, NonNeg, SOC, PSD, Exp, Pow3D)
