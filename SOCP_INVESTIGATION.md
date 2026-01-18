# SOCP Convergence Investigation

## Summary

When QP problems from Maros-Meszaros are reformulated as SOCP (Second-Order Cone Programs), Minix fails to converge or returns `AlmostOptimal` on many problems that Clarabel solves to full precision (`Optimal`) in 5-11 iterations.

## What's Working

- Native QP solving (with P matrix) works well on most problems
- LP and simple SOCP problems converge normally
- Initial iterations of SOCP-reformulated problems show good progress

## What's Not Working

- SOCP-reformulated QP problems fail on several test cases:
  - `DUALC1_SOCP`: MaxIters FAIL
  - `DUALC8_SOCP`: AlmostOptimal (rel_d=2.3e-5)
  - `PRIMALC1_SOCP`: MaxIters FAIL
  - `PRIMALC2_SOCP`: MaxIters FAIL
  - `PRIMALC5_SOCP`: MaxIters FAIL
  - `PRIMALC8_SOCP`: MaxIters FAIL

- Clarabel solves ALL these problems to Optimal with residuals ~1e-10 in 5-11 iterations

## Clarabel Comparison Results

| Problem | Clarabel Status | Iters | r_prim | r_dual |
|---------|-----------------|-------|--------|--------|
| HS21_SOCP | Solved | 8 | 7.29e-10 | 6.97e-10 |
| HS35_SOCP | Solved | 8 | 2.27e-10 | 4.24e-10 |
| HS268_SOCP | Solved | 7 | 5.82e-10 | 3.21e-10 |
| DUAL1_SOCP | Solved | 8 | 6.03e-10 | 5.76e-10 |
| DUAL4_SOCP | Solved | 9 | 3.99e-10 | 3.69e-10 |
| DUALC1_SOCP | **Solved** | 5 | 4.93e-10 | 3.75e-10 |
| DUALC8_SOCP | **Solved** | 5 | 5.68e-10 | 4.79e-10 |
| PRIMALC1_SOCP | **Solved** | 6 | 4.49e-10 | 4.25e-10 |
| PRIMALC2_SOCP | **Solved** | 7 | 2.60e-09 | 1.61e-10 |
| PRIMALC5_SOCP | **Solved** | 11 | 2.97e-09 | 4.89e-10 |
| PRIMALC8_SOCP | **Solved** | 8 | 4.28e-09 | 1.32e-10 |

## Iteration Trace (DUALC1_SOCP)

The solver gets close to the solution but then experiences numerical breakdown:

```
iter    p_obj      d_obj      gap        rel_p      rel_d      gap_rel
...
  16    3.7048e-1  3.7047e-1  1.07e-5    4.23e-9    5.96e-6    1.20e-6
  17    3.7046e-1  3.7046e-1  8.96e-7    1.04e-8    4.98e-7    1.02e-7    ← Nearly converged
  18    3.7045e-1  3.7045e-1  1.03e-7    1.44e-7    1.02e-7    1.54e-7    ← BREAKDOWN STARTS HERE
  BLOCK primal: idx=233 s=5.218e2 ds=-2.335e14 alpha_p_raw=1.181e-12
  BLOCK dual: idx=233 z=4.994e0 dz=-1.946e11 alpha_d_raw=1.374e-14
  BLOCK primal: idx=233 s=5.218e2 ds=-2.698e24 alpha_p_raw=1.135e-22  ← Exploding step directions
  BLOCK dual: idx=233 z=4.994e0 dz=-2.325e21 alpha_d_raw=1.185e-24
  19    3.6880e-1  3.7033e-1  1.53e-3    8.31e-6    2.15e-7    8.05e-6    ← Residuals get worse
  ...
  29    3.7048e-1  3.6503e-1  5.45e-3    3.15e-6    1.72e-5    6.39e-5    ← Never recovers
```

Key observations:
1. Iteration 17: Nearly converged with `rel_p=1.04e-8`, very close to tolerance
2. Iteration 18: Step size collapses to `1.88e-27` due to massive step directions at index 233
3. Condition number explodes from `6.12e20` to `2.86e21`
4. After iteration 18, the solver oscillates and never recovers

## Root Cause Analysis

### The SOCP Formulation

QPs are reformulated to SOCP using rotated second-order cone (RSOC):

```
QP:   min (1/2) x'Px + q'x   s.t. Ax = b, x >= 0

SOCP: min t + q'x   s.t. Ax = b, x >= 0, ((t+1)/√2, (t-1)/√2, Lx) ∈ SOC
      where P = L'L (Cholesky)
```

The rotated SOC `2tv >= ||Lx||²` with `v=1` gives `t >= ||Lx||²/2`, which is equivalent to the QP objective.

### The Problem: Near-Boundary Behavior

The SOC constraint requires:
```
((t+1)/√2)² >= ((t-1)/√2)² + ||Lx||²
```

When the optimal `t ≈ ||Lx||²/2` is small (close to 0 or 1):
- The `(t-1)/√2` term approaches the boundary
- At index 233, the `(t-1)/√2` component has `s[233] = 5.218e2` but the step direction `ds[233] = -2.335e14`
- This causes line search to collapse (alpha → 0)

### Likely Bugs

1. **NT Scaling Breakdown**: The Nesterov-Todd scaling for SOC uses Jordan algebra operations. When components are near the boundary (small discriminant `t² - ||x||²`), the scaling becomes ill-conditioned.

2. **Step Direction Explosion**: The KKT system produces massive step directions at iteration 18, suggesting numerical issues in the linear algebra (condition number ~10²¹).

3. **Missing Recovery Mechanism**: Unlike Clarabel, Minix doesn't have a robust mechanism to handle near-boundary situations in SOC cones.

## Relevant Files

### SOCP Formulation
- `solver-bench/src/qps.rs` (lines 225-391) - `to_socp_form()` method

### SOC Cone Implementation
- `solver-core/src/cones/soc.rs` - SOC barrier, gradient, step calculations

### NT Scaling for SOC
- `solver-core/src/scaling/nt.rs` (lines 91-131) - `nt_scaling_soc()` using Jordan algebra
  - `jordan_sqrt()`, `quad_rep_apply()`, `jordan_inv()` operations

### IPM Main Loop
- `solver-core/src/ipm2/predcorr.rs` - Predictor-corrector iterations
- `solver-core/src/ipm2/solve.rs` - Main solve loop, step computation

### KKT System
- `solver-core/src/linalg/kkt.rs` - KKT assembly
- `solver-core/src/linalg/qdldl.rs` - LDL factorization

### Test Script
- `solver-bench/compare_clarabel.py` - Python script comparing Minix vs Clarabel on SOCP problems

## Next Steps

1. **Investigate NT scaling stability**: Check if `jordan_sqrt`, `jordan_inv` produce stable results when SOC discriminant is small

2. **Add regularization**: Consider adding regularization to the KKT system when condition number is high

3. **Study Clarabel's approach**: Clarabel uses a different formulation or has better numerical safeguards

4. **Consider alternative SOCP formulation**: The rotated SOC conversion may not be numerically optimal; direct RSOC support could be better

## Running the Comparison

```bash
# Run Clarabel comparison
cd solver-bench
python compare_clarabel.py

# Run Minix regression on SOCP
MINIX_VERBOSE=2 cargo run --release -p solver-bench -- regression --socp-only --filter DUALC1

# Run full SOCP regression
cargo run --release -p solver-bench -- regression --socp-only
```
