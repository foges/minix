# Minix v16+ — Next Steps Still Worth Trying (with Pseudocode / Patch Sketches)

**Context (as of 2026-01-08)**  
- QP suite: ~77% pass rate at strict tolerance (but Boyd problems are *not* actually passing after tightening).  
- Symmetric cones (nonneg/SOC/PSD/“SPD”): mostly working.  
- **Exponential cone: currently broken** (does not converge; directions/stepsize logic not producing progress).

This document focuses on **high-leverage, still-worth-trying** changes that have a realistic chance of:
1) Getting exp cones to “actually solve”, and  
2) Nudging QP-suite pass rate up a few points without destabilizing already-good cases.

---

## How to use this doc

Each proposal has:
- **Why it’s worth trying**
- **Trigger / scope**
- **Implementation sketch** (pseudocode + patch-like snippets)
- **Tests** (what to run and what success looks like)

I’ve deliberately avoided vague “maybe try X” advice—everything below is written so you can implement it directly.

---

# P0 — Fix exponential cone (make it solve, not just “avoid explosions”)

Right now the exp-cone path is in “fails to make a step / alpha=0 / wrong answer” territory.  
For exp cones, the biggest mistake I’ve seen repeatedly (including in your summary) is: **treating exp cone complementarity like elementwise `s_i * z_i ≈ μ`.** That works for `R_+` but **not** for nonsymmetric cones. For exp cones, you need the **barrier-gradient + higher-order correction** approach (Clarabel-style), plus **line search that checks exp-cone interior**.

## P0.1 Add robust exp-cone *interior* checks and use them in step selection

### Why it’s worth trying
If alpha is repeatedly 0, you’re likely failing the interior test or computing step-to-boundary incorrectly.

### Core interior tests
For primal exp cone point `s = (x,y,z)` in the *interior*:
- `y > 0`
- `z > 0`
- `ψ(s) = y * log(z/y) - x > 0`

For dual exp cone point `z = (u,v,w)` in the *interior* (one common characterization):
- `u < 0`
- `w > 0`
- `-u * exp(v/u) < e * w` (equivalently `log(-u) + v/u < 1 + log(w)`)

### Patch sketch: exp interior helpers

```rust
fn exp_primal_interior(s: [f64; 3]) -> bool {
    let (x,y,z) = (s[0], s[1], s[2]);
    if !(y > 0.0 && z > 0.0) { return false; }
    // ψ = y*log(z/y) - x
    let psi = y * (z/y).ln() - x;
    psi > 0.0
}

fn exp_dual_interior(z: [f64; 3]) -> bool {
    let (u,v,w) = (z[0], z[1], z[2]);
    if !(u < 0.0 && w > 0.0) { return false; }

    // check: log(-u) + v/u < 1 + log(w)
    // do it in log space to avoid overflow/underflow:
    let lhs = (-u).ln() + v/u;
    let rhs = 1.0 + w.ln();
    lhs < rhs
}
```

### Patch sketch: exp step size via backtracking (simple + reliable)
Instead of trying to derive a closed-form boundary step (which is error-prone for exp),
do **backtracking** on the exp blocks.

```rust
fn alpha_backtrack_exp(
    s: [f64;3], ds: [f64;3],
    z: [f64;3], dz: [f64;3],
    mut alpha: f64, // start with alpha from other cones or 1.0
) -> f64 {
    let beta = 0.8;        // shrink
    let min_alpha = 1e-16; // hard floor

    while alpha > min_alpha {
        let s_new = [s[0] + alpha*ds[0], s[1] + alpha*ds[1], s[2] + alpha*ds[2]];
        let z_new = [z[0] + alpha*dz[0], z[1] + alpha*dz[1], z[2] + alpha*dz[2]];

        if exp_primal_interior(s_new) && exp_dual_interior(z_new) {
            return alpha;
        }
        alpha *= beta;
    }
    0.0
}
```

### Tests
- Run your smallest exp cone benchmark (`exp_cone_no_presolve`) and log:
  - `alpha` at each iter should stop being pinned at 0.
  - `ψ(s)` should remain positive and not collapse to ~0 every iter.

---

## P0.2 Stop using elementwise exp-cone “μ_i = s_i*z_i” complementarity

### Why it’s worth trying
For nonsymmetric cones, complementarity is expressed through the barrier:
- `s = -μ ∇f^*(z)` (dual barrier gradient)
- and the Mehrotra-style corrector requires a higher-order term (`η`) derived from the 3rd derivative.

Your summary already points in this direction (“Clarabel uses analytical formula; shift = grad*σμ - η”).  
This is *the* central missing piece if exp cones are “totally broken”.

### Implementation: build the exp-cone *combined* RHS using barrier objects

At the level where you assemble the predictor-corrector RHS for the cone blocks, implement:

> For nonsymmetric cones, use  
> `d_s = s + σ μ ∇f^*(z) + η`,  
> where `η = -1/2 ∇³ f^*(z)[Δz, (∇² f^*(z))^{-1} Δs]`.

This matches the design doc guidance (and Clarabel’s implementation pattern).

#### Pseudocode structure per exp-cone block

```rust
// Inputs (per exp block):
// - current z (dual) and s (slack)
// - affine directions dz_aff, ds_aff
// - sigma, mu
// - ability to evaluate exp dual barrier oracle: (grad_fstar, hess_fstar, third_contract)

fn exp_ds_comb(
    s: [f64;3],
    z: [f64;3],
    ds_aff: [f64;3],
    dz_aff: [f64;3],
    sigma: f64,
    mu: f64,
) -> [f64;3] {

    // 1) Dual barrier oracle at z:
    //    grad_fstar = ∇ f^*(z)
    //    Hstar       = ∇² f^*(z)   (3x3 SPD)
    let (grad_fstar, Hstar) = exp_dual_oracle(z);

    // 2) u = (Hstar)^{-1} ds_aff
    let u = solve_3x3(Hstar, ds_aff);

    // 3) eta = -0.5 * ∇³ f^*(z)[dz_aff, u]
    let eta = exp_third_contract_fstar(z, dz_aff, u).scale(-0.5);

    // 4) d_s = s + σ μ grad_fstar + eta
    [
        s[0] + sigma*mu*grad_fstar[0] + eta[0],
        s[1] + sigma*mu*grad_fstar[1] + eta[1],
        s[2] + sigma*mu*grad_fstar[2] + eta[2],
    ]
}
```

### Where to hook it
Search for the place where you currently have special-casing like:
- “Exp cone: pure centering”
- or code that sets `d_s_comb[i] = (mu_i - target_mu)/z_safe`

Replace that exp-cone branch with the **block-based** computation above.

---

## P0.3 Implement the exp dual-barrier oracle (grad/Hess) via the “dual map” (Newton solve), with caching

### Why it’s worth trying
The exp cone dual barrier `f^*(z)` is inconvenient to evaluate directly, but Clarabel and others do:
- solve a small Newton problem to compute the **dual map** `x = -∇f^*(z)` (a primal point),
- then reuse primal derivatives at `x` to obtain `∇f^*(z)` and `∇² f^*(z)`.

Caching/warm-starting the Newton solve per exp block is a massive practical win.

### Pseudocode: exp dual map oracle
```rust
struct ExpDualCache {
    // store last x for warm-start; one per exp cone block
    x_prev: [f64; 3],
    valid: bool,
}

fn exp_dual_oracle(z: [f64;3], cache: &mut ExpDualCache) -> ([f64;3], [[f64;3];3]) {
    // 1) Solve for x = dual_map(z):  ∇f(x) + z = 0
    //    Use 5-15 Newton steps with damping and interior checks.
    let mut x = if cache.valid { cache.x_prev } else { exp_default_dual_map_start(z) };

    for _ in 0..MAX_NEWTON {
        // Compute primal grad/Hess at x for f(x) = -log(ψ) - log(y) - log(z)
        let (g, H) = exp_primal_grad_hess(x);

        // residual: r = g + z
        let r = [g[0] + z[0], g[1] + z[1], g[2] + z[2]];
        if norm_inf(r) < 1e-12 { break; }

        // Newton step: H * dx = -r
        let dx = solve_3x3(H, [-r[0], -r[1], -r[2]]);

        // Damped step to keep x interior
        let mut alpha = 1.0;
        while alpha > 1e-16 {
            let x_new = [x[0] + alpha*dx[0], x[1] + alpha*dx[1], x[2] + alpha*dx[2]];
            if exp_primal_interior(x_new) { x = x_new; break; }
            alpha *= 0.5;
        }
        if alpha <= 1e-16 { break; }
    }

    cache.x_prev = x;
    cache.valid = true;

    // 2) grad f^*(z) = -x
    let grad_fstar = [-x[0], -x[1], -x[2]];

    // 3) Hess f^*(z) = (Hess f(x))^{-1}
    let (_, Hx) = exp_primal_grad_hess(x);
    let Hstar = invert_3x3(Hx);

    (grad_fstar, Hstar)
}
```

### Tests
- Add a unit test that checks the involution property:
  - if `x = dual_map(z)`, then `z ≈ -∇f(x)` and `x` is interior.
- Add a test that `Hstar` is SPD (eigs > 0), and that `Hstar * Hx ≈ I`.

---

## P0.4 Implement the exp-cone higher-order correction `η` analytically (Clarabel-style)

### Why it’s worth trying
Finite differences for third derivatives are numerically awful (you already saw this).
The analytical contraction is the “real” fix.

You already have the key pattern:
- compute `u = (∇² f^*(z))^{-1} ds_aff`
- compute `η = -0.5 ∇³ f^*(z)[dz_aff, u]`
- then use `shift = σμ grad - η`

### Practical suggestion
Don’t try to symbolically implement *everything* at once. Do it in layers:

1) Implement `T[p,q] = ∇³(-log ψ)(x)[p,q]` using the **generic contraction** formula.  
2) Plug into `η = 0.5 H^* u` where `u = ∇³ f(x)[p,q]` and `p = -H^* dz_aff`.

### Pseudocode: generic contraction for `-log(ψ)`
Given ψ = ψ(x), define:
- `g = ∇ψ`
- `H = ∇²ψ`
- `Tψ[p,q] = ∇³ψ[p,q]` (for exp, this is sparse/cheap)

Then:
```text
∇³(-log ψ)[p,q] = -(1/ψ) * Tψ[p,q]
                  + (1/ψ²) * ( gᵀ p * H q + gᵀ q * H p + (pᵀ H q) g )
                  - (2/ψ³) * (gᵀ p) (gᵀ q) g
```

This formula is easy to code and easy to unit-test.

### Tests
- Random interior `x`, random `p,q`; compare contraction against a *very carefully scaled* finite difference just for sanity (not production).
- Validate η is “small-ish” relative to σμ∇f^*(z) for typical iterates (not orders of magnitude larger).

---

## P0.5 Add exp-cone “central neighborhood” check to line search (prevents drift)

### Why it’s worth trying
Even with correct oracles, exp cones can drift out of the central path unless you enforce some neighborhood.
A simple and effective check is:

`|| s + μ ∇f^*(z) ||_∞ <= θ μ`

(or use 2-norm). You can check this after tentative step, and backtrack if violated.

```rust
fn exp_central_ok(s: [f64;3], z: [f64;3], mu: f64, theta: f64, cache: &mut ExpDualCache) -> bool {
    let (grad_fstar, _) = exp_dual_oracle(z, cache);
    let res = [
        s[0] + mu*grad_fstar[0],
        s[1] + mu*grad_fstar[1],
        s[2] + mu*grad_fstar[2],
    ];
    norm_inf(res) <= theta*mu
}
```

Then in line search:
- require `exp_primal_interior`, `exp_dual_interior`, **and** `exp_central_ok`.

---

# P1 — QP suite: small pass-rate bump with “surgical” robustness

You’re already competitive, so I’d only chase improvements that are:
- **triggered only on failure/stall**
- **don’t perturb already-solving cases**

## P1.1 Make Boyd problems pass without harming others: “progress-based iteration budget”

### Why it’s worth trying
Boyd problems are huge; you don’t want to raise `max_iter` globally.
But if they’re making monotonic progress, you *do* want to keep iterating.

### Trigger
- Very large `n` or `m` (e.g., > 50k)
- Residual decreases consistently over a window (e.g., 5–10 iters)

### Pseudocode
```rust
let huge = (n > 50_000) || (m > 50_000);

if huge {
    // default: 50, but allow extension up to 200
    let hard_cap = 200;

    if iter >= 50 && is_making_progress(history, window=8, min_factor=1.05) {
        // keep going
    } else if iter >= 50 {
        // stop (stall)
        return MaxIters;
    }
}
```

Where `is_making_progress` can be something like:
- `rel_p` or `rel_d` reduced by ≥5% over last 8 iters, OR
- gap reduced by ≥5%, OR
- merit function reduced.

### Tests
- Full suite runtime should not blow up (only 1–2 huge problems extend).
- Boyd1/2 should at least *not* stop at 50 when trending down.

---

## P1.2 Quasi-definite KKT failures (QFFFFF80): add “diagonal shift and retry” inside factorization

This is copied from the earlier next-steps doc because it remains one of the most plausible +1 improvements.

### Patch sketch
When LDL / factorization fails with “not quasi-definite”:
1) Add diagonal regularization to primal block (and optionally dual block)
2) Retry factorization a few times with increasing shift

```rust
let mut delta = 1e-10 * diag_scale;
for _ in 0..MAX_RETRIES {
    match try_factorize() {
        Ok(_) => break,
        Err(NotQuasiDefinite) => {
            add_diag_shift_to_primal_block(delta);
            add_diag_shift_to_dual_block(delta); // optional
            delta *= 100.0;
        }
        Err(e) => return Err(e),
    }
}
```

### Testing
- Re-run QFFFFF80 with diagnostics: should no longer die when entering polish.

---

## P1.3 “Outer proximal loop” (real IP-PMM), not a fixed tiny ρ

### Why it’s worth trying
A fixed `ρ = 1e-6` can be “too weak to help, strong enough to hurt”.
PIQP’s real strength is *iterating* proximal problems and reducing ρ.

### Trigger
Only enable outer loop when:
- Factorization fails OR
- You detect stall with μ at floor but residuals not decreasing.

### Pseudocode
```rust
let rho_schedule = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 0.0];

let mut x_ref = zeros(n);
for rho in rho_schedule {
    set_proximal_rho(rho);
    set_x_ref(x_ref);

    let result = solve_inner(max_iter=50);

    if result.status == Optimal || result.status == AlmostOptimal {
        return result;
    }

    // warm start next outer loop
    x_ref = result.x;
}
return last_result;
```

### Key detail
- When `rho > 0`, **skip polish** (or delay it until `rho == 0`), otherwise you’ll fight the proximal term.

### Expected gain
- +1–3 problems if tuned well (especially quasi-definite / degenerate LP-ish ones)

---

## P1.4 Frozen residuals with μ ~ machine epsilon: add “restart with bumped μ and higher reg”

### Why it’s worth trying
In many “frozen” logs you have:
- μ tiny (sometimes ~1e-14 or 1e-15)
- r_p, r_d not improving at all

At that point, pushing μ smaller is useless; you need a **restoration/restart**.

### Trigger
If for `k` iterations:
- `rel_p` and `rel_d` change < 1%,
- but `min(mu) < 1e-12`,
then do a restart:

### Restart idea (minimal)
- reset μ target upward (e.g., multiply by 1e3 but cap)
- increase KKT regularization (static_reg *= 1e2)
- keep x,y warm-started, but reinitialize (s,z) to a safer interior point

```rust
if stalled && mu < 1e-12 {
    static_reg *= 100.0;
    mu = max(mu, 1e-6);

    // reinitialize s,z away from boundary
    s = max(s, s_floor);
    z = max(z, z_floor);

    continue; // keep iterating
}
```

This is intentionally “dumb but effective”.

---

## P1.5 HSDE QFORPLAN: add “ray detection” + correct classification (even if you can’t fix it)

### Why it’s worth doing
Even if you can’t make QFORPLAN solve, you can:
- stop it from blowing up numerically
- return a meaningful status (infeasible/unbounded) rather than MaxIters

### Trigger
During HSDE iterations:
- τ → 0, κ large, or dual residual explodes while primal residual stays tiny
- μ exploding

### Pseudocode detection
```rust
if mu > 1e12 && rel_p < 1e-6 && rel_d > 1e6 {
    return SolverStatus::PrimalInfeasible; // or DualInfeasible depending on HSDE theory
}
```

Even if the benchmark counts it as “fail”, it’s still a correctness improvement and avoids wasted runtime.

---

# P2 — Engineering hygiene that prevents “false confidence”

## P2.1 Make tolerance handling impossible to accidentally loosen

### Why it’s worth doing
You already got bitten by this with Boyd.

### Patch sketch
In solver-bench:
- print the exact tolerances at start of every run
- dump into the JSON results (so you can’t misread later)

Also add a regression test:
- asserts the solver config uses `tol = 1e-8` when you say it does.

---

## P2.2 Add “exp cone smoke test” that fails on MaxIters

You already fixed some tests that accepted MaxIters; do the same here.

Minimal assertion:
```rust
assert!(matches!(result.status, Optimal | AlmostOptimal));
```

---

# Suggested order of attack (fastest path to real wins)

1) **P0.1 + P0.2**: interior checks + stop using elementwise exp complementarity  
2) **P0.3**: dual-map oracle with caching  
3) **P0.5**: neighborhood check (stability)  
4) **P0.4**: analytical η (this is the “finish the job” step)  
5) **P1.1**: progress-based iter budget (Boyd)  
6) **P1.2**: KKT shift-and-retry (QFFFFF80)  
7) **P1.3**: outer proximal schedule (only on stall/failure)

---

# What I would *not* do next

- Global `max_iter` increase for everything  
- Fixed `ρ` proximal for all problems  
- Heavy new heuristics that change behavior on the 105 already-good problems  

Keep changes:
- **localized**
- **triggered**
- **observable** (with diagnostics + regression tests)

---

## Appendix: quick “patch targeting” checklist for exp cones

When you’re debugging exp cones, always log per-iteration for one exp block:
- `psi(s)` and `psi(z)` (or dual interior margin)
- `alpha` and which condition fails (primal interior / dual interior / neighborhood)
- `||s + μ ∇f^*(z)||` (centrality)
- whether `ds_aff, dz_aff` are basically zero or exploding
- whether your dual-map Newton solver converges (residual norm)

If any of these are “nonsense”, fix that before trying more advanced corrections.
