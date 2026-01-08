# Exponential Cone Dual Barrier Analysis

## Problem Statement

The exp cone implementation uses the **primal barrier** in `dual_map()`, but for non-symmetric cones it should use the **dual barrier**.

## Cone Definitions

**Primal cone:**
```
K_exp = {(x,y,z) : z ≥ y*exp(x/y), y > 0}
```

**Dual cone:**
```
K_exp* = {(u,v,w) : u < 0, w ≥ -u*exp(v/u - 1)}
```

## Barrier Functions

**Primal barrier** (currently implemented):
```
f(x,y,z) = -log(y) - log(z) - log(ψ)
where ψ = y*log(z/y) - x
```

**Dual barrier** (MISSING):
```
f*(u,v,w) = -log(-u) - log(w) - log(ψ*)
where ψ* = u + w*exp(v/w - 1)
```

## Verification of ψ*

For (u,v,w) ∈ K_exp*, we need w ≥ -u*exp(v/u - 1), i.e., u < 0 and w > -u*exp(v/u - 1).

Check that ψ* > 0 ⟺ (u,v,w) ∈ int(K_exp*):
```
ψ* > 0
u + w*exp(v/w - 1) > 0
w*exp(v/w - 1) > -u
exp(v/w - 1) > -u/w  (since w > 0)
v/w - 1 > log(-u/w)  (since u < 0, so -u > 0)
v/w > log(-u/w) + 1
v/w > log(-u) - log(w) + 1
exp(v/w) > exp(log(-u) - log(w) + 1)
exp(v/w) > (-u)/w * e
w*exp(v/w)/e > -u
w*exp(v/w - 1) > -u  ✓
```

So ψ* > 0 is equivalent to w > -u*exp(v/w - 1), which is the interior of K_exp*.

## Dual Barrier Gradient

Need to compute ∇f*(u,v,w):
```
f*(u,v,w) = -log(-u) - log(w) - log(ψ*)
where ψ* = u + w*exp(v/w - 1)
```

Partial derivatives:
```
∂ψ*/∂u = 1
∂ψ*/∂v = w*exp(v/w - 1) * (1/w) = exp(v/w - 1)
∂ψ*/∂w = exp(v/w - 1) + w*exp(v/w - 1)*(-v/w²)
        = exp(v/w - 1) * (1 - v/w)
```

So:
```
∇f*(u,v,w) = [
    -1/(-u) - (1/ψ*) * ∂ψ*/∂u,
    0       - (1/ψ*) * ∂ψ*/∂v,
    -1/w    - (1/ψ*) * ∂ψ*/∂w
]
= [
    1/u - 1/ψ*,
    -exp(v/w - 1)/ψ*,
    -1/w - exp(v/w - 1)*(1 - v/w)/ψ*
]
```

## Implementation Plan

1. Implement `exp_dual_barrier_grad_block(z, grad_out)`:
   - Takes (u,v,w) ∈ K_exp*
   - Computes ψ* = u + w*exp(v/w - 1)
   - Computes gradient as above

2. Update `exp_dual_map_block()`:
   - Currently solves: ∇f(x) = -z using **primal barrier**
   - Should solve: ∇f*(x) = -z using **dual barrier**
   - Use Newton's method with dual barrier gradient

3. Test:
   - Unit test that ∇f*(z) + ∇f(s) = 0 when properly paired
   - Check that trivial exp cone problem converges

## Expected Impact

With correct dual barrier:
- BFGS scaling will compute correct H matrix
- KKT steps will preserve cone membership
- Exp cone problems will converge instead of diverging
