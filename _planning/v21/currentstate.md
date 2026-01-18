# Current PSD/SDP Solver State (v21)

## Summary

After implementing the PSD Jordan corrector with Sylvester solve, results are mixed:
- **truss1**: ✅ Works well - 7 iterations, correct objective
- **control1**: ❌ 100 iterations (MaxIters), wrong objective

## Root Cause Analysis (NEW)

After extensive debugging, the root cause has been identified:

### Problem: Rank-Deficient Constraint Matrix

control1 has **10 zero columns** in the constraint matrix A:
- Columns 26, 33, 34, 41, 42, 43, 50, 51, 52, 53
- These correspond to off-diagonal entries X1[(6,7), (6,8), (7,8), (6,9), (7,9), (8,9), (6,10), (7,10), (8,10), (9,10)]
- The lower-right 5×5 block of X1's off-diagonals are **not in any constraint**

This is by design in SDPA format - some matrix entries are only constrained by:
1. The PSD constraint (X1 ⪰ 0)
2. The trace constraint (tr(X1) = 1)

But they don't appear in constraints F1-F20 that link X1 and X2.

### Why This Causes Problems

1. **Ill-conditioning**: Without Ruiz scaling, κ = 6.4e15 (catastrophically ill-conditioned)
2. **KKT degeneracy**: Zero A columns mean:
   - Dual constraint A[:,j]'*y + z[j] = q[j] simplifies to z[j] = q[j] = 0
   - These z entries push toward zero, conflicting with PSD interior requirement
3. **Residual scaling issue**: Large ||x|| or ||s|| inflates the denominator of rel_p, making constraint violations look small

### Evidence

From Python analysis:
```
A col norms:
  zero columns: [26, 33, 34, 41, 42, 43, 50, 51, 52, 53]
  min col norm: 0.0
  max col norm: 14642

Without Ruiz: condition number κ = 6.4e15
With Ruiz: condition number manageable but problem remains
```

The solver converges (gap → 0) but to the WRONG objective (131 vs 17.78) because:
- The constraints are approximately satisfied (small rel_p)
- But the scaling hides large absolute violations
- The objective depends on X2, which is determined by X1 through constraints
- If those constraints are violated, X2 takes wrong values

### Why truss1 Works but control1 Doesn't

truss1 structure:
- 7 blocks: [2, 2, 2, 2, 2, 2, 1] (all 2×2 PSD or diagonal)
- Every variable is constrained by some F_i matrix
- No zero columns in A

control1 structure:
- 2 blocks: [10, 5] (10×10 and 5×5 PSD)
- 10 out of 70 variables (14%) have no A constraints
- Lower-right corner of X1 is effectively "free"

## Potential Fixes

1. **Presolve detection**: Detect zero A columns and warn/fail for SDP problems
2. **Regularization**: Add small regularization to zero columns
3. **Column elimination**: Remove unconstrained variables and handle via PSD projection
4. **Different formulation**: Use a formulation that doesn't create zero columns

## SDPA Format Reference

The SDPA FORMAT file defines:
```
(P) min c'y  s.t.  Σ y_i*F_i - F0 = X,  X ⪰ 0    (primal, y variables)
(D) max tr(F0*Y)  s.t.  tr(F_i*Y) = c_i,  Y ⪰ 0  (dual, Y matrix)
```

Our solver converts to the dual formulation (Y = decision variable).
Reference objective 17.78 is for the SDPA dual = tr(F0*Y).

## Detailed Iteration Analysis

Running control1 with MINIX_VERBOSE=3 reveals the exact convergence behavior:

### Key Observations from Iteration Data

**1. Primal residual is STUCK from the start:**
```
iter    0: rel_p=1.446e-1, r_p (abs)≈6.7e-1 (initial, before Ruiz unscaling)
iter    8: rel_p=2.993e-5, r_p (abs)≈225
iter   50: rel_p=3.065e-5, r_p (abs)≈222
iter   99: rel_p=4.819e-5, r_p (abs)≈223
```
The **relative** primal residual looks like it's improving, but the **absolute** primal residual r_p stays around 220-250 throughout! The relative residual appears small only because ||x||, ||s||, ||b|| are large (scaling effect).

**2. Dual slack Z becomes increasingly infeasible (min eigenvalue grows negative):**
```
iter    0: min_z = -184
iter   25: min_z = -974   <-- something goes wrong here
iter   50: min_z = -1322
iter   75: min_z = -1532
iter   99: min_z = -1647
```
For a valid PSD solution, Z ⪰ 0 (all eigenvalues non-negative). But min_z reaches **-1647** - the dual solution is massively infeasible!

**3. Gap and dual residual DO converge:**
```
iter    0: gap_rel=6.365e-1, rel_d=1.610e-1
iter   50: gap_rel=1.327e-6, rel_d=3.189e-6
iter   99: gap_rel=1.740e-7, rel_d=5.646e-7
```
The solver thinks it's converging because gap → 0 and dual residual → 0.

**4. Critical transition at iterations 25-30:**
```
iter   24: mu=4.602e-4, min_z=-186  (OK)
iter   25: mu=8.733e-3, min_z=-974  (JUMP!)
iter   26: mu=5.923e-3, min_z=-998
```
Around iteration 25, there's a dramatic transition where:
- mu (barrier parameter) spikes from 4.6e-4 to 8.7e-3
- min_z (PSD eigenvalue) jumps from -186 to -974
- This is when the solver diverges to the wrong solution

### Full Iteration Log (sampled)

```
iter    0 mu=6.463e3 α=0.947 min_s=-1.4 min_z=-184 rel_p=1.45e-1 rel_d=1.61e-1 gap_rel=6.4e-1
iter    5 mu=1.214e-1 α=0.903 min_s=-73  min_z=-147 rel_p=2.46e-5 rel_d=8.43e-5 gap_rel=6.5e-4
iter   10 mu=1.655e-3 α=0.615 min_s=-42  min_z=-146 rel_p=2.90e-5 rel_d=6.25e-5 gap_rel=1.3e-5
                                                     ^-- rel_p stuck around 2-4e-5

iter   25 mu=8.733e-3 α=0.759 min_s=-104 min_z=-974 rel_p=2.53e-5 rel_d=1.43e-5 gap_rel=1.2e-5
                                                     ^-- min_z JUMPS to -974!

iter   50 mu=7.843e-4 α=0.017 min_s=-43  min_z=-1322 rel_p=3.09e-5 gap_rel=1.5e-6
iter   75 mu=1.632e-4 α=0.028 min_s=-25  min_z=-1532 rel_p=4.38e-5 gap_rel=1.7e-6
iter   99 mu=3.285e-5 α=0.826 min_s=-17  min_z=-1647 rel_p=4.82e-5 gap_rel=1.7e-7

Final absolute residuals (from "Iter" output):
  r_p = 223 (huge! should be ~0)
  r_d = 73
```

### Interpretation

1. **The solver converges to a KKT point** - gap and dual residual go to zero
2. **But it's the WRONG KKT point** because:
   - The primal constraints Ax + s = b are violated (r_p = 223)
   - The PSD constraint Z ⪰ 0 is violated (min eigenvalue = -1647)
3. **Relative scaling hides the truth** - rel_p = 4.8e-5 looks acceptable, but abs(r_p) = 223 is huge
4. **The zero columns in A cause degeneracy** - variables in those columns can drift freely without affecting the apparent residuals

### Root Cause Confirmation

The zero-column analysis from Python confirms 10 variables have no linear constraints:
- Columns 26, 33, 34, 41, 42, 43, 50, 51, 52, 53 have zero A column norm
- These correspond to X1 off-diagonal entries in the lower-right 5×5 subblock
- They affect the KKT system via the PSD constraint only, not via Ax = b

## Original Observations (for reference)

### Changes Made
1. Replaced pure centering with proper Jordan-algebra Mehrotra corrector for PSD cones
2. Implemented Sylvester equation solve using eigendecomposition: `λU + Uλ = 2v`
3. The implementation follows the same pattern as SOC but uses matrix operations

### Test Results

#### truss1 (Works)
```
Problem: truss1
Constraints (m): 6
Blocks: [2, 2, 2, 2, 2, 2, 1]
Variables (svec): 19
Status: Optimal
Iterations: 7
Primal obj: -9.000025e0
Reference: -8.999996e0
Rel error: 2.92e-6
```

#### control1 (Broken)
```
Problem: control1
Constraints (m): 21
Blocks: [10, 5]
Variables (svec): 70
Status: AlmostOptimal (100 iters)
Primal obj: 1.310923e2
Reference: 1.778463e1
Rel error: 6.03e0  (HUGE!)
```

## Reference Solver Comparison

### CLARABEL (via CVXPY) - WORKS!
```
Solving with CLARABEL...
iter    pcost        dcost       gap       pres      dres      k/t        μ       step
  0  +1.8475e-01  +1.8475e-01  8.33e-17  7.18e-01  9.90e-01  1.00e+00  5.57e+00   ------
  5  -1.7397e+02  -1.7046e+02  2.06e-02  1.11e-03  2.37e-03  3.60e+00  4.32e-02  5.21e-01
 10  -8.8125e+01  -8.4467e+01  4.33e-02  3.81e-04  4.40e-05  3.67e+00  1.51e-03  7.33e-01
 15  -3.3116e+01  -3.1802e+01  4.13e-02  8.06e-05  9.66e-07  1.31e+00  3.16e-05  2.27e-01
 20  -1.8150e+01  -1.8076e+01  4.12e-03  4.21e-05  7.39e-09  7.44e-02  2.37e-07  8.55e-01
 25  -1.7785e+01  -1.7785e+01  1.77e-07  7.49e-07  2.62e-13  3.15e-06  8.39e-12  9.32e-01
 28  -1.7785e+01  -1.7785e+01  3.01e-11  3.24e-09  1.39e-16  5.35e-10  1.51e-15  9.74e-01
Terminated with status = Solved
solve time = 71.344374ms

Status: optimal
Optimal value: 17.784627  ✅ CORRECT!
```

### SCS (via CVXPY) - FAILS!
```
Status: optimal_inaccurate
Optimal value: 235.322408  ❌ WRONG (after 5000 iterations)
```

### Our Solver (minix) - FAILS!
```
Status: AlmostOptimal
Optimal value: 131.0923  ❌ WRONG (after 100 iterations)
```

### Key Observations

1. **CLARABEL converges in 28 iterations** to the correct answer
2. **CLARABEL's primal residual converges smoothly**: 7e-1 → 1e-3 → 4e-5 → 3e-9
3. **Our solver's primal residual gets stuck** at ~4e-5 (relative) = 223 (absolute)
4. **CLARABEL uses CVXPY's reformulation** which may handle the problem differently

### What CLARABEL Does Differently

From Clarabel's output:
- Uses equilibration: `min_scale = 1.0e-4, max_scale = 1.0e4`
- Uses iterative refinement: `max iter = 10, reltol = 1.0e-13`
- Uses static reg: `ϵ1 = 1.0e-8, ϵ2 = 4.9e-32`
- Uses dynamic reg: `ϵ = 1.0e-13, δ = 2.0e-7`

Key difference: **CVXPY reformulates the problem** before passing to Clarabel. The raw SDPA problem has:
- 21 constraints, 70 svec variables
- 10 zero columns in A

After CVXPY reformulation:
- 91 constraints (21 equality + 70 cone membership)
- The formulation may avoid the zero-column degeneracy

## Key Findings (Final)

### 1. Our Formulation IS Correct
```
Our A matrix: (91, 70) - 21 equality rows + 70 identity embedding rows
Zero columns: 0 (the identity embedding fills them in)
```
Lines 199-203 of sdplib.rs correctly add `-I` for PSD embedding:
```rust
for i in 0..total_dim {
    triplets.push((sdpa.m_dim + i, i, -1.0));
}
```

### 2. CVXPY Uses Different Scaling in Identity Embedding
CVXPY/Clarabel uses:
- `-1.0` for diagonal entries (matrix X_ii)
- `-sqrt(2)` for off-diagonal entries (matrix X_ij, i≠j)

We use `-1.0` for all entries. This might matter for the PSD cone geometry.

### 3. The Problem is NOT Zero Columns
Earlier analysis was done on just the 21×70 equality part. The full 91×70 matrix has no zero columns.

### 4. IPM Internals Differ
From trace output, our solver:
- Computes correct NT scaling W
- Computes correct Hessian H
- But min_z (dual slack eigenvalue) goes increasingly negative: -184 → -974 → -1647

Clarabel keeps dual slack strictly positive throughout.

## Root Cause Hypotheses

1. **Step size computation**: We may be taking too-large steps in the PSD direction, leaving the cone
2. **Regularization**: Clarabel uses different static/dynamic regularization strategy
3. **Iterative refinement**: Clarabel uses 10 refinement iterations with tight tolerance (1e-13)
4. **svec scaling mismatch**: The `-sqrt(2)` vs `-1` in identity embedding might affect conditioning

## Next Steps (Prioritized)

1. **Fix identity embedding scaling** (QUICK TEST)
   - Change line 201-202 to apply `svec_scale()` to identity rows
   - This matches CVXPY's formulation exactly

2. **Add absolute residual monitoring**
   - Detect when rel_p looks good but abs(r_p) is huge
   - Fail early on false convergence

3. **Increase iterative refinement**
   - Try 10 iterations like Clarabel
   - Tighter tolerance (1e-13)

4. **Compare with Clarabel's regularization**
   - Static: ε1 = 1e-8, ε2 = 4.9e-32
   - Dynamic: ε = 1e-13, δ = 2e-7
