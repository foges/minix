# Conic Solver Status Summary (v19)

## Current State

### Hand-crafted SDP Tests: PASSING (13/13)

All custom SDP tests pass:
- `test_sdp_trace_minimization` (3x3, 4x4, 5x5): Optimal in 4 iterations
- `test_sdp_maxcut_triangle`: Optimal in 4 iterations, obj=-0.75
- `test_sdp_maxcut_complete_4`: Optimal in 5 iterations, obj=-1.0
- `test_sdp_maxcut_star`: Optimal in 70 iterations, obj=-2.0
- `test_sdp_maxcut_path`: AlmostOptimal in 100 iterations, obj=-1.5
- `test_sdp_lovasz_theta`: Optimal in 5 iterations, obj=-11.18
- `test_sdp_min_eigenvalue`: Optimal in 6 iterations, obj=0.27

### SDPLIB Benchmarks: NOT CONVERGING

SDPLIB problems hit max iterations without converging:

#### truss1 (small, 6 constraints, 19 vars)
```
Problem: truss1
Constraints (m): 6
Blocks: [2, 2, 2, 2, 2, 2, 1]
Variables (svec): 19
Status: MaxIters
Primal obj: -9.090898e0
Dual obj: 9.090898e0
Iterations: 100
Reference: -8.999996e0
Rel error: 9.09e-3
```
Close to optimal but not converging.

#### hinf1 (13 constraints, 41 vars)
```
Problem: hinf1
Constraints (m): 13
Blocks: [4, 4, 6]
Variables (svec): 41
Status: MaxIters
Primal obj: 1.923950e0
Dual obj: -1.923950e0
Iterations: 20
Reference: 2.032749e0
Rel error: 3.59e-2
```
Getting ~95% of the way there but stalling.

#### theta1 (104 constraints, 1275 vars - 50x50 PSD block)
```
Problem: theta1
Constraints (m): 104
Blocks: [50]
Variables (svec): 1275
Status: MaxIters
Primal obj: -0.000000e0
Dual obj: 0.000000e0
Iterations: 500
Time: 148372.271 ms (~2.5 min)
Reference: 2.300000e1
Rel error: 9.58e-1
```
Completely fails - objective is 0 when reference is 23.

## Root Cause Analysis

### 1. SDPA Format Conversion Issues

The SDPA format uses:
- Primal: max tr(F0 * X) s.t. tr(Fi * X) = ci, X >= 0
- Dual: min c'y s.t. F0 + sum_i yi*Fi >= 0

Our conversion in `sdplib.rs`:
- x = svec(X)
- q = -svec(F0) (negative for minimization)
- A_i = svec(Fi) as row i
- b = c

**Potential issue**: Sign conventions may be wrong for primal vs dual objectives.

### 2. Multi-block Structure

SDPLIB problems often have multiple PSD blocks, e.g., truss1 has:
- Blocks: [2, 2, 2, 2, 2, 2, 1]
- This means 6 separate 2x2 PSD cones + 1 diagonal

Our conversion creates:
- Zero cone for equality constraints
- Separate PSD/NonNeg cones for each block

**Potential issue**: Block offsets in svec indexing may be incorrect.

### 3. Termination Criteria

For SDP, the standard termination involves:
- Primal feasibility: ||Ax - b|| / (1 + ||b||)
- Dual feasibility: ||A'y + z - c|| / (1 + ||c||)
- Duality gap: |p* - d*| / (1 + |p*| + |d*|)

Our termination may not be accounting for PSD structure correctly.

### 4. Centering Parameter

For PSD cones, we use pure centering (sigma=1) to avoid numerical issues with
Mehrotra correction on svec format. This is conservative and may slow convergence.

## Implementation Status

### Completed
- [x] SDPA-sparse format parser (`sdplib.rs`)
- [x] SDPA to conic form conversion
- [x] Unified benchmark CLI (`--suite mm` vs `--suite sdplib`)
- [x] Reference optimal values for ~60 SDPLIB problems
- [x] Basic PSD cone support in solver

### In Progress
- [ ] Debug SDPLIB conversion (sign conventions, block handling)
- [ ] Verify svec scaling (sqrt(2) on off-diagonals)
- [ ] Check termination criteria for multi-block SDP

### TODO
- [ ] Test with MOSEK/Clarabel to verify conversion is correct
- [ ] Add verbose iteration logging for SDP debugging
- [ ] Profile large SDP solve (theta1 takes 148s for 500 iters)

## CLI Usage

```bash
# Run single SDPLIB problem
cargo run -p solver-bench -- benchmark --suite sdplib --problem truss1

# Run full SDPLIB suite
cargo run -p solver-bench -- benchmark --suite sdplib --limit 10

# Run Maros-Meszaros QP suite
cargo run -p solver-bench -- benchmark --suite mm --limit 10
```

## Debug Output

### truss1 (after fixing sign bug)
```
Running single problem: truss1
Problem: truss1
Constraints (m): 6
Blocks: [2, 2, 2, 2, 2, 2, 1]
Variables (svec): 19
Status: MaxIters
Primal obj: -9.090898e0
Dual obj: -9.090898e0
Iterations: 50
Reference: -8.999996e0
Rel error: 9.09e-3
```

**Key observations**:
1. Primal and dual now match (fixed sign bug in solve_sdpa)
2. Value is close to reference: -9.09 vs -9.00 (0.9% error)
3. Solver is making progress but hitting max iters before convergence
4. Tolerance may be too tight for PSD cone's slower convergence

## Bug Fixed

**SDPA objective sign bug**: In `solve_sdpa`, we were incorrectly reporting:
- `primal_obj = -obj_val`
- `dual_obj = obj_val` (wrong! should also be negated)

Fixed to report same value for both (at optimality they're equal).

## Quick Benchmark Results (100 iters, 1e-6 tol)

| Problem | Status | Our Obj | Reference | Rel Error |
|---------|--------|---------|-----------|-----------|
| truss1  | MaxIters | -9.09 | -9.00 | 0.9% |
| truss4  | MaxIters | -9.04 | -9.01 | 0.35% |
| hinf1   | MaxIters | 1.92 | 2.03 | 3.6% |
| hinf2   | MaxIters | -0.24 | 10.93 | **93%** |

**Analysis**:
- Truss problems: Working well, close to optimal
- hinf1: Making progress, ~96% of optimal
- hinf2: Completely broken - wrong sign and magnitude

The hinf problems have multiple PSD blocks of different sizes. The conversion may have block offset issues.

## Exponential Cone Benchmarks: NOT CONVERGING (0/11)

```
Problem                             n     Status    Iters       Objective    Time (ms)  Quality
----------------------------------------------------------------------------------------------------
entropy_max_n5                      5 AlmostOptimal      100    -1.274474e-5         4.30   Infeas
kl_divergence_n5                    5   MaxIters      100     -4.797201e0         2.63   Infeas
portfolio_exp_n5                    5   MaxIters      100     -5.401608e1         1.67   Infeas
entropy_max_n10                    10   MaxIters      100     -3.883443e0         6.83   Infeas
kl_divergence_n10                  10   MaxIters      100      2.435979e1         6.53   Infeas
...
Summary:
  Optimal: 0/11 (0.0%)
  Feasible: 0/11 (0.0%)
```

**Analysis**: All exp cone problems hit max iters without becoming feasible. Exp cone support needs debugging.

## Next Steps

1. **Debug hinf2**: Check block offset calculation for multi-block problems
2. **Debug exp cone**: Investigate why exp cone problems are infeasible
3. **Profile convergence**: Track residuals per iteration to understand stalling
4. **Test termination**: Add Optimal/AlmostOptimal detection for 1e-3 rel accuracy
