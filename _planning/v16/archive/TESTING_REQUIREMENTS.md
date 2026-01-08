# Testing Requirements for Minix Solver

## CRITICAL: Always Check Solve Status

**RULE**: Tests MUST NEVER accept `MaxIters` or other failure statuses as passing.

### Acceptable Statuses ✅

- `SolveStatus::Optimal` - Meets strict tolerances (tol_feas=1e-8, tol_gap=1e-8)
- `SolveStatus::AlmostOptimal` - Meets relaxed tolerances (tol_feas=1e-4, tol_gap_rel=5e-5)

### Failure Statuses ❌

- `SolveStatus::MaxIters` - Hit iteration limit WITHOUT converging
- `SolveStatus::NumericalError` - Solver encountered numerical issues
- `SolveStatus::PrimalInfeasible` - Problem is infeasible
- `SolveStatus::DualInfeasible` - Problem is unbounded

## Test Assertion Patterns

### ✅ CORRECT: Require convergence

```rust
let result = solve(&prob, &settings).expect("Solve failed");

// ALWAYS check status
assert!(matches!(
    result.status,
    SolveStatus::Optimal | SolveStatus::AlmostOptimal
), "Expected Optimal/AlmostOptimal, got {:?}", result.status);

// Then check solution quality
assert!((result.obj_val - expected_obj).abs() < tol);
```

### ❌ WRONG: Accept MaxIters

```rust
// NEVER DO THIS!
assert!(matches!(
    result.status,
    SolveStatus::Optimal | SolveStatus::MaxIters  // ❌ BAD!
));
```

### ❌ WRONG: Only check solve() doesn't error

```rust
// NEVER DO THIS!
let result = solve(&prob, &settings).unwrap();  // ❌ Doesn't check status!
assert!((result.obj_val - expected_obj).abs() < tol);
```

## Regression Test Requirements

### Maros-Meszaros Suite

```rust
// From solver-bench/src/regression.rs
if !matches!(res.status, SolveStatus::Optimal | SolveStatus::AlmostOptimal) {
    failures.push(format!(
        "{}: status={:?} rel_p={:.2e} rel_d={:.2e} gap_rel={:.2e}",
        res.name, res.status, res.rel_p, res.rel_d, res.gap_rel
    ));
}
```

### Cone-Specific Tests

For problems with experimental cone support (e.g., SOC with incomplete scaling):

```rust
// May accept NumericalError if cone implementation is incomplete
// But NEVER accept MaxIters!
assert!(matches!(
    result.status,
    SolveStatus::Optimal | SolveStatus::AlmostOptimal | SolveStatus::NumericalError
), "Expected Optimal/AlmostOptimal/NumericalError, got {:?}", result.status);
```

## Benchmark Requirements

### Timing Benchmarks

Benchmarks MUST validate correctness before reporting timing:

```rust
let result = solve(&prob, &settings).unwrap();

// Check status FIRST
let status_str = match result.status {
    SolveStatus::Optimal => "Optimal",
    SolveStatus::AlmostOptimal => "AlmostOpt",
    _ => "FAILED",  // Don't hide failures!
};

// Only report timing for successful solves
if !matches!(result.status, SolveStatus::Optimal | SolveStatus::AlmostOptimal) {
    println!("FAILED: {:?}", result.status);
    panic!("Benchmark problem failed to solve");
}

println!("{:<20} {:>8} {:>12} {:>12}",
         name, result.info.iters, elapsed_ms, status_str);
```

### Comparison Benchmarks

When comparing to other solvers:

```rust
// Count ONLY Optimal and AlmostOptimal as solved
let solved_count = results.iter()
    .filter(|r| matches!(r.status, SolveStatus::Optimal | SolveStatus::AlmostOptimal))
    .count();

println!("Solved: {}/{} ({:.1}%)",
         solved_count, total, 100.0 * solved_count as f64 / total as f64);
```

## Files Fixed

### ✅ Fixed Files
- `solver-bench/src/regression.rs` - Now requires Optimal|AlmostOptimal
- `solver-core/tests/integration_tests.rs` - All 5 tests fixed to reject MaxIters

### ❌ Still Need Fixing
- `solver-bench/examples/exp_cone_*.rs` - No status validation
- Any custom benchmarks or scripts

## Common Mistakes to Avoid

1. **"It returned Ok() so it must be fine"** ❌
   - solve() returns Ok() even when status is MaxIters

2. **"The objective value looks right"** ❌
   - Solver can get close to the right objective without converging

3. **"Residuals are small enough"** ❌
   - Check the status field, not just residuals

4. **"It's just a benchmark, correctness doesn't matter"** ❌
   - Fast broken code is worse than slow correct code

5. **"MaxIters just means it needs more iterations"** ❌
   - MaxIters means the solver is stuck or making no progress
   - If solver is healthy, increasing max_iter will help
   - If solver is broken (like exp cones now), more iterations won't help

## Pre-Commit Checklist

Before committing changes:

- [ ] All integration tests check for Optimal|AlmostOptimal
- [ ] Regression tests require convergence
- [ ] Benchmarks validate status before reporting timing
- [ ] No tests accept MaxIters as passing
- [ ] Run `cargo test` and verify all tests pass

## Debugging Failed Tests

When a test fails with MaxIters:

```rust
// Add diagnostic output
println!("Status: {:?}", result.status);
println!("Iterations: {}", result.info.iters);
println!("Residuals: primal={:.2e}, dual={:.2e}, gap={:.2e}",
         result.info.primal_res, result.info.dual_res, result.info.gap);
println!("μ: {:.2e}", result.info.mu);
```

Look for:
- Residuals not decreasing → Solver is stuck
- μ not decreasing → Barrier parameter update broken
- Residuals ~0.6-0.9 → Exp cone issue (known bug)
- Iterations exactly at max_iter → Hit limit

## Future Improvements

- [ ] Add `assert_solved_optimally!()` macro
- [ ] Add pre-commit hook to detect MaxIters in tests
- [ ] Add CI check for status validation
- [ ] Create test helper: `require_optimal_status(result)`
- [ ] Add verbose mode to print status in all tests
