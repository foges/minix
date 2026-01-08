# Exponential Cone Bug - ROOT CAUSE FOUND

**Date**: 2026-01-08
**Status**: ROOT CAUSE IDENTIFIED

---

## The Smoking Gun

### Issue 1: Massive Search Direction Values

The KKT system produces search directions with **astronomical values**:

```
ds = [4.512610e13, 1.916872e0, 6.917798e-1]
     ^^^^^^^^^^^ 45 TRILLION!

dz = [4.512657e5, 1.402548e6, -1.050917e5]
     ALL in the hundreds of thousands
```

This causes `step_to_boundary` to return `alpha=0` because even an infinitesimal step would violate the cone interior.

### Issue 2: s and z Remain Identical Throughout

More critically, **s and z NEVER diverge** from their initial values:

```
Iteration 0:
  s = [-2.102766, 1.112818, 2.517934]
  z = [-2.102766, 1.112818, 2.517934]  ← SAME

Iteration 1000:
  s = [-2.102766, 1.112818, 2.517934]  ← UNCHANGED!
  z = [-2.102766, 1.112818, 2.517934]  ← UNCHANGED!
```

In a working IPM:
- Initialization: s ≈ z is OK (HSDE uses symmetric initialization)
- After KKT solve: s and z should DIVERGE
- **Current behavior**: s and z stay locked together forever!

### Issue 3: BFGS Scaling Inputs Look Suspicious

```
s_tilde (from dual map) = [-5.837608e-1, 1.184489e-12, 5.788857e-1]
                                         ^^^^^^^^^^^^ Near zero!

mu = 3.999993e0
mu_tilde = 1.695702e-1
mu * mu_tilde = 0.68 ← Should be ≈ 1.0 (centrality violation)
```

The dual map produces `s_tilde[1] ≈ 0`, and the centrality check fails (mu*mu_tilde should ≈ 1).

---

## Root Cause Analysis

### Theory 1: KKT System Assembly Bug (MOST LIKELY)

**Hypothesis**: The KKT system for exp cones is assembled incorrectly, producing garbage search directions.

**Evidence**:
1. ds[0] = 4.5e13 is completely unreasonable
2. s and z never move from initialization
3. Works for NonNeg/Zero cones, fails for Exp

**Where to look**:
- `solver-core/src/linalg/kkt.rs` - KKT assembly for exp cones
- How are exp cone blocks added to the KKT matrix?
- Is the scaling matrix W being used correctly?

### Theory 2: BFGS Scaling Matrix is Wrong

**Hypothesis**: The W matrix for exp cones is ill-conditioned or incorrect.

**Evidence**:
1. s_tilde[1] ≈ 0 from dual map
2. Centrality check fails (mu*mu_tilde = 0.68, not 1.0)
3. Huge search direction values suggest numerical blow-up

**Where to look**:
- `solver-core/src/scaling/bfgs.rs:62-200` - rank-3 BFGS
- Is the cross product computation stable?
- Are there divide-by-zero or near-zero issues?

### Theory 3: Dual Map is Broken

**Hypothesis**: `exp_dual_map_block` doesn't converge or produces wrong values.

**Evidence**:
1. s_tilde[1] ≈ 0 is suspicious
2. If dual map is wrong, scaling will be wrong, and KKT will fail

**Where to look**:
- `solver-core/src/cones/exp.rs:396-436` - dual map Newton solver
- Add diagnostics to check Newton convergence
- Verify the output satisfies ∇f*(x) = -z

---

## Next Steps

### Step 1: Add More KKT Diagnostics

Print the full KKT matrix structure for exp cones to see if it's being assembled correctly.

### Step 2: Test with Identity Scaling

Bypass BFGS and use W=I to see if the problem is in scaling or KKT assembly:

```rust
// In bfgs.rs, temporarily force identity scaling for exp cones
ScalingBlock::Diagonal(vec![1.0, 1.0, 1.0])
```

If this works, the bug is in BFGS. If not, it's in KKT assembly.

### Step 3: Verify Dual Map

Add diagnostics to `exp_dual_map_block` to check:
- Does Newton iteration converge?
- Is the residual ∥∇f*(x) + z∥ small?
- Why is s_tilde[1] ≈ 0?

### Step 4: Compare with Working Solver

Check how ECOS, SCS, or Clarabel initialize and scale exp cones. Look for differences in:
- Initialization strategy
- Scaling approach
- KKT structure

---

## Test Cases

### Simplest Failing Case

```
minimize    x
subject to  s = [-x, 1, 1] ∈ K_exp

Expected: x = 0, obj = 0
Actual:   MaxIters (alpha=0 on every iteration)
```

### What Should Happen (Iteration 1)

```
Start:
  s = [-2.103, 1.113, 2.518]
  z = [-2.103, 1.113, 2.518]

After KKT solve:
  ds = [small values, e.g., 0.1, 0.05, 0.02]
  dz = [small values, different from ds]
  alpha = 0.9 (healthy step)

After step:
  s_new = s + 0.9*ds  (moved!)
  z_new = z + 0.9*dz  (moved!, different from s_new)
```

### What Actually Happens

```
Start:
  s = [-2.103, 1.113, 2.518]
  z = [-2.103, 1.113, 2.518]

After KKT solve:
  ds = [4.5e13, 1.92, 0.69]  ← GARBAGE!
  dz = [4.5e5, 1.4e6, -1e5]  ← GARBAGE!
  alpha = 0.0 (can't move!)

After step:
  s_new = s (UNCHANGED)
  z_new = z (UNCHANGED)
```

---

## Files Involved

### Primary Suspects

1. **`solver-core/src/linalg/kkt.rs`**
   - How exp cone blocks are added to KKT matrix
   - Possibly wrong structure or indices

2. **`solver-core/src/scaling/bfgs.rs`**
   - Lines 62-200: rank-3 BFGS scaling
   - May produce ill-conditioned W matrix

3. **`solver-core/src/cones/exp.rs`**
   - Lines 396-436: dual_map_block
   - May not converge or produce wrong output

### Supporting Code

4. **`solver-core/src/ipm2/predcorr.rs`**
   - Line 1154: Calls `compute_step_size`
   - Line 1379-1450: `compute_step_size` function

5. **`solver-core/src/ipm2/solve.rs`** (or solve_normal.rs)
   - Where KKT system is solved
   - Calls into kkt.rs

---

## Why This Went Undetected

1. **Tests only checked objective value**, not solve status
2. **No verbose output** by default - would have shown alpha=0 immediately
3. **No public benchmark exp cone problems** to validate against
4. **Integration tests accepted MaxIters** as passing

---

## Success Criteria

### Minimal Fix

- [ ] Exp cone problems take alpha > 0 on most iterations
- [ ] s and z diverge after first KKT solve
- [ ] ds and dz have reasonable magnitudes (< 100)
- [ ] At least 3/5 exp cone problems reach Optimal

### Full Fix

- [ ] All 5 exp cone suite problems solve
- [ ] Residuals decrease monotonically
- [ ] Comparable performance to ECOS/SCS
- [ ] Add 20+ exp cone problems from CBLIB

---

## Timeline

- **2026-01-07**: Exp cone "optimizations" added (but didn't work)
- **2026-01-08 AM**: Testing hardened, failures discovered
- **2026-01-08 PM**: Root cause identified (this document)
- **2026-01-08 Evening**: Fix implementation (in progress)

---

## Confidence Level

**95% confident** the bug is in KKT assembly or BFGS scaling, not in cone geometry functions.

**Evidence**:
- ✅ Cone geometry functions tested independently - all work
- ✅ Initialization creates interior points
- ✅ step_to_boundary works correctly
- ❌ KKT produces garbage search directions
- ❌ s and z locked together (KKT should separate them)

**Next**: Test with identity scaling (W=I) to isolate whether it's BFGS or KKT.
## Identity Scaling Test Result

With W=I (identity scaling):
- s and z diverge ✓
- Solution completely wrong ❌  
- x = -5.72e8 (should be 0)
- ds/dz still astronomical
- primal_res: 1.8e-9 (good)
- dual_res: 1.5e-8 (good)  
- gap: 1.0 (bad)
- μ: 1.12e9 (exploded)

**Conclusion**: Bug is in KKT assembly, not BFGS scaling.

