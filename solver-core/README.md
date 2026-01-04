# Minix Solver Core

A state-of-the-art convex optimization solver implemented in Rust, targeting MOSEK-class quality with support for linear programming (LP), quadratic programming (QP), and second-order cone programming (SOCP).

## Features

### Implemented ✅

- **Core Algorithm**: Homogeneous Self-Dual Embedding (HSDE) interior point method with predictor-corrector steps
- **Cone Support**:
  - Zero cone (equality constraints)
  - NonNeg cone (inequality constraints)
  - Second-Order Cone (SOC/Lorentz cone) with full Jordan algebra
- **Scaling**: Nesterov-Todd (NT) scaling for symmetric cones
- **Linear Algebra**:
  - QDLDL sparse LDL^T factorization
  - Static and dynamic regularization for robustness
  - Efficient two-RHS solve for predictor-corrector
  - Pre-allocated workspace for zero-allocation iterations
- **Termination**:
  - Optimality detection (primal/dual feasibility + duality gap)
  - Infeasibility certificates (primal/dual infeasible detection)
  - Numerical error handling
- **Presolve**: Ruiz equilibration for problem conditioning
- **Testing**: Comprehensive unit tests with finite difference validation
- **Benchmarking**: 600+ problems across 7 standard test suites

### Planned 🚧

- **Additional Cones**: Exponential, Power, PSD
- **BFGS Scaling**: For nonsymmetric cones
- **MIP Support**: Branch-and-bound for mixed-integer problems
- **Python/C Bindings**: Foreign function interfaces

## Quick Start

### Problem Format

Minix solves problems in the form:

```
minimize    (1/2) x^T P x + q^T x
subject to  A x + s = b
            s ∈ K
```

where `K` is a Cartesian product of cones (Zero, NonNeg, SOC, etc.).

### Example: Simple LP

```rust
use solver_core::{solve, ProblemData, ConeSpec, SolverSettings};
use solver_core::linalg::sparse;

// min x1 + x2
// s.t. x1 + x2 = 1
let prob = ProblemData {
    P: None,  // No quadratic term
    q: vec![1.0, 1.0],
    A: sparse::from_triplets(1, 2, vec![(0, 0, 1.0), (0, 1, 1.0)]),
    b: vec![1.0],
    cones: vec![ConeSpec::Zero { dim: 1 }],
    var_bounds: None,
    integrality: None,
};

let settings = SolverSettings::default();
let result = solve(&prob, &settings)?;

println!("Status: {:?}", result.status);
println!("Solution: {:?}", result.x);
println!("Objective: {}", result.obj_val);
```

### Example: Quadratic Program

```rust
// min 0.5 * (x1^2 + x2^2) + x1 + x2
// s.t. x1 + x2 = 1

let p_triplets = vec![(0, 0, 1.0), (1, 1, 1.0)];

let prob = ProblemData {
    P: Some(sparse::from_triplets(2, 2, p_triplets)),
    q: vec![1.0, 1.0],
    A: sparse::from_triplets(1, 2, vec![(0, 0, 1.0), (0, 1, 1.0)]),
    b: vec![1.0],
    cones: vec![ConeSpec::Zero { dim: 1 }],
    var_bounds: None,
    integrality: None,
};
```

### Example: Second-Order Cone Program

```rust
// min t
// s.t. || x || <= t
//      t + x1 + x2 = 1

let prob = ProblemData {
    P: None,
    q: vec![1.0, 0.0, 0.0],  // min t (first component)
    A: sparse::from_triplets(1, 3, vec![(0, 0, 1.0), (0, 1, 1.0), (0, 2, 1.0)]),
    b: vec![1.0],
    cones: vec![
        ConeSpec::Soc { dim: 3 },      // (t, x1, x2) in SOC
        ConeSpec::Zero { dim: 1 },     // Equality constraint
    ],
    var_bounds: None,
    integrality: None,
};
```

## Architecture

### Module Structure

```
solver-core/
├── src/
│   ├── problem.rs          # Problem data structures
│   ├── cones/              # Cone kernel implementations
│   │   ├── traits.rs       # ConeKernel trait
│   │   ├── zero.rs         # Zero cone (equality)
│   │   ├── nonneg.rs       # NonNeg cone (inequality)
│   │   └── soc.rs          # SOC cone with Jordan algebra
│   ├── scaling/            # NT and BFGS scaling
│   │   ├── mod.rs          # ScalingBlock enum
│   │   ├── nt.rs           # Nesterov-Todd scaling
│   │   └── bfgs.rs         # BFGS (placeholder)
│   ├── linalg/             # Linear algebra layer
│   │   ├── sparse.rs       # Sparse matrix utilities
│   │   ├── qdldl.rs        # QDLDL wrapper
│   │   └── kkt.rs          # KKT system builder
│   ├── ipm/                # Interior point method
│   │   ├── hsde.rs         # HSDE formulation
│   │   ├── predcorr.rs     # Predictor-corrector
│   │   ├── workspace.rs    # Pre-allocated iteration vectors
│   │   ├── termination.rs  # Termination criteria
│   │   └── mod.rs          # Main solver loop
│   └── lib.rs              # Public API
└── tests/
    ├── cone_tests.rs       # Finite difference validation
    └── integration_tests.rs # End-to-end tests
```

### Key Components

#### ConeKernel Trait

All cones implement a unified interface:

```rust
pub trait ConeKernel {
    fn dim(&self) -> usize;
    fn barrier_degree(&self) -> usize;
    fn is_interior_primal(&self, s: &[f64]) -> bool;
    fn barrier_value(&self, s: &[f64]) -> f64;
    fn barrier_grad_primal(&self, s: &[f64], grad_out: &mut [f64]);
    fn barrier_hess_apply_primal(&self, s: &[f64], v: &[f64], out: &mut [f64]);
    fn step_to_boundary_primal(&self, s: &[f64], ds: &[f64]) -> f64;
    fn unit_initialization(&self, s_out: &mut [f64], z_out: &mut [f64]);
    // ... and dual variants
}
```

#### HSDE State

The solver maintains variables `(x, s, z, τ, κ)`:

- `x`: Primal variables
- `s`: Cone slack variables
- `z`: Dual variables
- `τ`: Homogenization (scaling) variable
- `κ`: Dual homogenization variable

When `τ > 0`, the solution is `x̄ = x/τ`, `s̄ = s/τ`, `z̄ = z/τ`.
When `τ → 0`, the problem is infeasible.

#### KKT System

The solver constructs and factors:

```
K = [ P + εI    A^T   ]
    [ A      -(H + εI) ]
```

where `H` is the Hessian scaling matrix (NT or BFGS).

## Algorithm

### Predictor-Corrector IPM

1. **Initialize**: Set `(x, s, z, τ, κ)` using cone unit initializations
2. **Loop**:
   - Compute NT scaling: `H` such that `H s = z`
   - Factor KKT system with current `H`
   - **Affine step**: Solve with `σ = 0` (pure Newton)
   - Compute step size `α_aff` and centering `σ = (1 - α_aff)³`
   - **Corrector step**: Solve with Mehrotra correction
   - Update `(x, s, z, τ, κ)` with fraction-to-boundary rule
   - Check termination (optimality, infeasibility, max iterations)

### Termination Criteria

- **Optimal**: Primal residual, dual residual, and duality gap all small
- **Primal Infeasible**: `τ → 0` and `b^T z < 0`
- **Dual Infeasible**: `τ → 0` and `q^T x < 0`
- **Max Iterations**: Iteration limit reached
- **Numerical Error**: NaN detected

## Running Tests

```bash
# Unit tests (cone kernels, KKT, etc.)
cargo test

# Integration tests
cargo test --test integration_tests -- --nocapture

# Specific test with verbose output
cargo test test_simple_lp -- --nocapture
```

## Running Examples

```bash
cargo run --example simple_lp
```

## Performance Considerations

### Current Status

The implementation prioritizes **correctness and clarity** over performance:

- ✅ Correct HSDE formulation
- ✅ Proper NT scaling with Jordan algebra
- ✅ Robust QDLDL factorization
- ✅ Comprehensive testing

### Known Limitations

1. **Simplified Predictor-Corrector**: The current implementation uses a basic version. Full Mehrotra correction with proper RHS construction is planned.

2. **Symbolic Factorization**: Not yet reused across iterations.

3. **No Parallelization**: Single-threaded execution.

### Planned Optimizations

- Reuse symbolic factorization across iterations
- Implement full Mehrotra correction
- Profile and optimize hot paths
- Parallel KKT assembly (optional)

## Comparison to Other Solvers

| Feature | Minix | ECOS | Clarabel | MOSEK |
|---------|-------|------|----------|-------|
| Language | Rust | C | Rust | C |
| LP/QP | ✅ | ✅ | ✅ | ✅ |
| SOCP | ✅ | ✅ | ✅ | ✅ |
| Exponential | 🚧 | ✅ | ✅ | ✅ |
| Power | 🚧 | ❌ | ❌ | ✅ |
| SDP | 🚧 | ❌ | ✅ | ✅ |
| Open Source | ✅ | ✅ | ✅ | ❌ |
| Production Ready | 🚧 | ✅ | ✅ | ✅ |

**Goal**: Match MOSEK-class quality for continuous convex optimization while remaining fully open source.

## Design Philosophy

1. **Equation-Complete**: Every implementation follows design doc equations exactly
2. **Test-Driven**: Finite difference validation for all cone kernels
3. **Type-Safe**: Leverage Rust's type system for correctness
4. **Modular**: Clean separation of concerns (cones, scaling, linalg, IPM)
5. **Deterministic**: Fixed orderings, stable summation, reproducible results
6. **Diagnostic-Rich**: Comprehensive iteration info, residuals, regularization tracking

## References

- **Design Document**: See `convex_mip_solver_design_final_final.md` for full algorithmic details
- **HSDE Method**: Self-dual homogeneous embedding for infeasibility detection
- **NT Scaling**: Nesterov-Todd scaling via Jordan algebra
- **QDLDL**: Quasi-definite LDL^T factorization for KKT systems
- **Clarabel.rs**: Reference implementation for Rust convex solver
- **MOSEK**: Commercial solver (quality target)

## Contributing

This solver is under active development. Contributions welcome in:

- Additional cone types (Exponential, Power, PSD)
- Performance optimizations
- Benchmark suite development
- Documentation improvements

## License

TBD

## Status

**Current**: Working IPM solver for LP, QP, and SOCP problems with Zero, NonNeg, and SOC cones.

**Next Steps**:
1. Implement Exponential and Power cones
2. Add PSD cone support
3. Performance tuning and optimization
4. Python bindings
