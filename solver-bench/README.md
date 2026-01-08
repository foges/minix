# Minix Solver Benchmark Suite

Comprehensive benchmarking tools for the Minix convex optimization solver, with support for multi-solver comparison and win matrix analysis.

## Features

- **Maros-Meszaros QP Benchmark**: Run the standard 138-problem QP test suite
- **JSON Export**: Export benchmark results for external analysis
- **Multi-Solver Comparison**: Compare Minix against other solvers (Clarabel, OSQP, etc.)
- **Win Matrix**: Head-to-head problem solving comparison between solvers
- **Performance Analysis**: Geometric mean timing on commonly solved problems

## Quick Start

### Run Maros-Meszaros Benchmark

```bash
# Run full suite (138 problems) with IPM2 solver
cargo run --release -p solver-bench -- maros-meszaros

# Run first 10 problems only
cargo run --release -p solver-bench -- maros-meszaros --limit 10

# Run with IPM1 solver instead
cargo run --release -p solver-bench -- maros-meszaros --solver ipm1

# Show detailed results table
cargo run --release -p solver-bench -- maros-meszaros --table

# Run single problem for debugging
cargo run --release -p solver-bench -- maros-meszaros --problem QFORPLAN
```

### Export Results to JSON

```bash
# Run benchmark and export to JSON
cargo run --release -p solver-bench -- maros-meszaros \
    --export-json results/minix-ipm2.json \
    --solver-name "Minix-IPM2"

# Export with different solver
cargo run --release -p solver-bench -- maros-meszaros \
    --solver ipm1 \
    --export-json results/minix-ipm1.json \
    --solver-name "Minix-IPM1"
```

### Compare Multiple Solvers

```bash
# Compare two solver runs
cargo run --release -p solver-bench -- compare \
    results/minix-ipm2.json \
    results/minix-ipm1.json

# Compare with detailed problem-by-problem breakdown
cargo run --release -p solver-bench -- compare \
    results/minix-ipm2.json \
    results/minix-ipm1.json \
    --detailed

# Limit detailed output to first 20 problems
cargo run --release -p solver-bench -- compare \
    results/*.json \
    --detailed \
    --limit 20
```

## Comparison Output

The `compare` command provides three types of analysis:

### 1. Summary Table

Shows overall performance metrics for each solver:

```
================================================================================
Multi-Solver Comparison Summary
================================================================================

Solver                  Optimal  AlmostOpt   Combined  Geom Mean Time       Pass Rate
--------------------------------------------------------------------------------
Minix-IPM2                  104          1        105          103.41ms           77.2%
Clarabel                     63          0         63          165.23ms           46.0%
OSQP                         58          2         60           45.12ms           43.8%
================================================================================
```

### 2. Win Matrix

Head-to-head comparison showing:
- Problems solver A solves that B doesn't
- Problems both solvers solve
- Problems solver B solves that A doesn't

```
================================================================================
Win Matrix: Problems Solved by Each Solver Pair
================================================================================
Format: A vs B shows (A only | both | B only)
--------------------------------------------------------------------------------

                               Minix-IPM2             Clarabel                 OSQP
---------------------------------------------------------------------------------
Minix-IPM2                     (105 solved)            42|63|0              45|60|0
Clarabel                           42|63|0           (63 solved)             3|60|0
OSQP                               45|60|0               3|60|0          (60 solved)
================================================================================
```

### 3. Performance Comparison

Geometric mean solve time on problems that both solvers successfully solve:

```
================================================================================
Performance Comparison (Geometric Mean Time on Commonly Solved Problems)
================================================================================

Minix-IPM2 vs Clarabel (63 common problems):
  Minix-IPM2           103.41ms
  Clarabel             165.23ms
  Minix-IPM2 is 1.60x faster

Minix-IPM2 vs OSQP (60 common problems):
  Minix-IPM2           103.41ms
  OSQP                  45.12ms
  OSQP is 2.29x faster
================================================================================
```

### 4. Detailed Problem-by-Problem (Optional)

With `--detailed` flag, shows status and timing for each problem:

```
================================================================================
Detailed Problem-by-Problem Comparison
================================================================================

Problem              Minix-IPM2        Clarabel            OSQP
                 Status Time(ms)  Status Time(ms)  Status Time(ms)
---------------------------------------------------------------
AUG2D               Opt   119.8     Opt   165.2     Opt    45.1
AUG2DC              Opt   118.4   MaxIt   200.3     Opt    48.2
QFORPLAN          MaxIt    50.1   MaxIt    50.0   MaxIt    50.3
...
================================================================================
```

## External Solver Integration

### Python Script for External Solvers

The `run_external_solvers.py` script provides a framework for running external solvers and exporting results in the same JSON format:

```bash
# Install required packages
pip install clarabel osqp scipy numpy

# Run Clarabel and export results
python solver-bench/run_external_solvers.py \
    --solver clarabel \
    --export results/clarabel.json \
    --max-iter 50

# Run OSQP
python solver-bench/run_external_solvers.py \
    --solver osqp \
    --export results/osqp.json
```

**Note**: The Python script is currently a framework/stub. To fully implement:
1. Add proper QPS parser (or use `qpsolvers` package)
2. Convert QP format to solver-specific formats
3. Map solver statuses to standard statuses

Alternatively, you can manually create JSON files matching the format (see JSON Format section below).

## JSON Format

Results are exported in the following JSON format:

```json
{
  "solver_name": "Minix-IPM2",
  "results": [
    {
      "name": "AUG2D",
      "n": 20200,
      "m": 10000,
      "status": "Optimal",
      "iterations": 7,
      "obj_val": 1677511.7517468934,
      "mu": 0.0012204188342120983,
      "solve_time_ms": 119.819458,
      "error": null
    },
    ...
  ],
  "summary": {
    "total": 136,
    "optimal": 104,
    "almost_optimal": 1,
    "max_iters": 31,
    "numerical_errors": 0,
    "parse_errors": 0,
    "total_time_s": 14.123,
    "geom_mean_iters": 12.5,
    "geom_mean_time_ms": 103.4
  }
}
```

### Status Values

- `"Optimal"` - Meets strict tolerances (gap=1e-8, feas=1e-8)
- `"AlmostOptimal"` - Meets relaxed tolerances (gap=5e-5, feas=1e-4)
- `"MaxIters"` - Hit iteration limit without converging
- `"NumericalError"` - Numerical issues encountered
- `"PrimalInfeasible"` - Problem is primal infeasible
- `"DualInfeasible"` - Problem is dual infeasible

## Advanced Usage

### Environment Variables

- `MINIX_DIAGNOSTICS=1` - Enable detailed diagnostics output
- `MINIX_DIRECT_MODE=1` - Use direct solve instead of iterative refinement

### Regression Testing

```bash
# Run regression suite and save baseline
cargo run --release -p solver-bench -- regression \
    --baseline-out baseline.json

# Run again and check for regressions
cargo run --release -p solver-bench -- regression \
    --baseline-in baseline.json \
    --max-regression 0.2  # Allow 20% slowdown
```

## Performance Metrics

### Geometric Mean

The geometric mean is used for timing because:
1. Less sensitive to outliers than arithmetic mean
2. More robust for multiplicative phenomena (performance ratios)
3. Shifted version handles zero times: `exp(mean(log(t + 1))) - 1`

### Pass Rate

Percentage of problems solved to "Optimal" or "AlmostOptimal" status:
```
Pass Rate = (Optimal + AlmostOptimal) / Total * 100%
```

## Benchmark Results

Current Minix performance on Maros-Meszaros (138 problems):

| Metric | Value |
|--------|-------|
| Optimal | 104/136 (76.5%) |
| AlmostOptimal | 1/136 (0.7%) |
| Combined | 105/136 (77.2%) |
| Geom Mean Time | ~103ms |

Comparison to other solvers (default accuracy):

| Solver | Pass Rate | Relative Time |
|--------|-----------|---------------|
| **Minix** | **77.2%** | - |
| PIQP | 96% | 1.0x (fastest) |
| ProxQP | 73% | 57.1x |
| HiGHS | 67% | - |
| SCS | 62% | - |
| Clarabel | 46% | 163.8x |
| OSQP | 43% | 16.3x |

## Contributing

To add support for a new solver:

1. Create a JSON file with benchmark results in the format above
2. Use `cargo run -- compare` to analyze against Minix
3. Or extend `run_external_solvers.py` with proper solver integration

## References

- [Maros-Meszaros QP Test Set](https://github.com/YimingYAN/QP-Test-Problems)
- [qpsolvers Benchmark](https://github.com/qpsolvers/maros_meszaros_qpbenchmark)
- [Clarabel Documentation](https://clarabel.org/)
- [OSQP Documentation](https://osqp.org/)
