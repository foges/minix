# Minix - Conic Optimization Solver

Rust implementation of an interior-point method (IPM) solver for convex conic optimization problems supporting QP, SOCP, SDP, and exponential cones.

## Project Structure

```
solver-core/     # Core IPM solver implementation
solver-bench/    # Benchmark runner (600+ problems across 7 test suites)
solver-py/       # Python bindings via PyO3
solver-mip/      # Mixed-integer extension (future)
solver-ffi/      # C FFI bindings (future)
```

## Build Commands

```bash
# Build all crates
cargo build --release

# Run tests
cargo test

# Run specific solver-core tests
cargo test --lib -p solver-core

# Run benchmarks (7 test suites available)
cargo run --release -p solver-bench -- maros-meszaros --limit 10
cargo run --release -p solver-bench -- maros-meszaros --problem HS21
cargo run --release -p solver-bench -- netlib --limit 10
cargo run --release -p solver-bench -- cblib --limit 10
cargo run --release -p solver-bench -- pglib --limit 5
cargo run --release -p solver-bench -- qplib --limit 10
cargo run --release -p solver-bench -- meszaros --limit 10
```

## Python Bindings

```bash
cd solver-py
source .venv/bin/activate  # or: uv venv && source .venv/bin/activate
maturin develop --release
python -c "import minix; print(minix.version())"
```

## Key Modules in solver-core

- `ipm/predcorr.rs` - Predictor-corrector IPM main loop
- `ipm/workspace.rs` - Pre-allocated workspace for iteration vectors
- `ipm/hsde.rs` - Homogeneous self-dual embedding utilities
- `cones/` - Cone implementations (zero, nonneg, SOC, PSD, exp)
- `linalg/kkt.rs` - KKT system assembly and solve
- `linalg/qdldl.rs` - LDL factorization wrapper
- `presolve/ruiz.rs` - Ruiz equilibration for conditioning
- `scaling/nt.rs` - Nesterov-Todd scaling

## Problem Format

The solver handles problems of the form:
```
minimize    (1/2) x'Px + q'x
subject to  Ax + s = b
            s ∈ K
```
where K is a Cartesian product of cones (Zero, NonNeg, SOC, PSD, Exp).

## Known Issues

1. Ruiz scaling doesn't preserve SOC geometry (needs block-aware scaling)
2. HSDE tau/kappa updates are frozen (tau=1)
