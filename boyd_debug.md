# BOYD1 / BOYD2 Debug Notes (MINIX)

This note focuses specifically on the Maros–Mészáros BOYD* failures you reported.

## What your diagnostics imply

### BOYD1
- `rel_p ≈ 8.75e-15` and `gap_rel ≈ 4.77e-9` with `μ ≈ 4.3e-10`.
- The run is effectively optimal **except** for dual feasibility: `rel_d ≈ 1.41e-3`.

Interpretation: the algorithm is getting all the way to “μ ~ 0” but can’t polish the **dual residual**. That is a classic symptom of **(a)** inaccurate KKT solves near the end, and/or **(b)** a bad scaling block `H` near the boundary.

### BOYD2
- `μ ≈ 6.8e1`, `gap_rel ≈ 2e-1` and doesn’t improve.
- This is a “near-LP” (almost empty Q) with massive constraints + bounds-as-rows.

Interpretation: either the Newton directions are being distorted (regularization/scaling issues), or step sizes are collapsing because some `s/z` components are being driven incorrectly.

## What is most likely broken (code-level)

### 1) NonNeg NT scaling fallback is wrong in `predcorr.rs`
In `solver-core/src/ipm/predcorr.rs`, the NT scaling fallback currently builds

- `d[i] = sqrt(s[i]/z[i])`

but `ScalingBlock::Diagonal { d }` is treated everywhere as **H = diag(d)**, and for the LP/QP NonNeg cone the correct NT scaling is

- **H = diag(s ./ z)**

Using `sqrt(s/z)` produces the wrong Newton system when NT scaling errors out.

Why this is relevant to BOYD1/BOYD2:
- `nt_scaling_nonneg()` currently refuses to scale if the cone’s “interior” test fails.
- The NonNeg cone interior test uses a *relative-to-max* tolerance; if `s_max` is huge (common when you have many bound slacks) and some components are tiny-but-positive (common late in the solve), it can trip the “not interior” condition.
- That forces the fallback path, which currently builds the wrong `H`.

This can produce exactly what you see:
- BOYD1: the method gets close, but dual feasibility polishing stalls.
- BOYD2: complementarity / progress can be completely derailed.

### 2) NonNeg NT scaling is overly strict near the boundary (`nt.rs`)
`nt_scaling_nonneg()` uses `cone.is_interior_*()` which can fail purely due to *dynamic range*, not actual non-positivity.

For NT scaling, what we actually need is simply:
- `s_i > 0` and `z_i > 0` (finite)

If those hold, `s/z` is well-defined.

### 3) Iterative refinement residual mismatch in KKT solver
Your KKT solver uses QDLDL with a **static diagonal shift** (`static_reg`) to guarantee quasi-definiteness. QDLDL is solving a *shifted* system

- `(K + ε I) x = b`

but the refinement code computes residuals using `Kx`, not `(K + εI)x`. That makes refinement far less effective exactly when you need it (late-iteration polishing).

This is consistent with “dual residual stalls while μ is tiny”.

### 4) Static regularization floor is very large for sparse/LP-ish problems
`ipm/mod.rs` forces `static_reg >= 1e-4` on sparse-QP/LP-ish instances and `>= 1e-6` even on dense QPs.

For BOYD2 this is plausibly too strong (distorts directions). For BOYD1 it can directly prevent high-accuracy polishing.

## Patch provided

The patch below targets the above “most likely broken” items.

**File:** `minix_boyd_fixes.patch`

### What it changes
- `predcorr.rs`: fallback scaling now uses **H = s/z** (with conservative clamps), and prints when fallback is actually being used (`MINIX_DIAGNOSTICS=1`).
- `nt.rs`: NonNeg NT scaling now treats “interior” as `s_i>0, z_i>0` (finite), avoiding false-negative interior checks caused by huge dynamic range.
- `kkt.rs`: iterative refinement residual uses the **shifted** matrix `(K + εI)`.
- `ipm/mod.rs`: reduces the forced `static_reg` floors, so your chosen `settings.static_reg` actually matters for polishing.

## How to use it

1) Apply the patch.
2) Re-run the two bad cases with diagnostics:

```bash
MINIX_DIAGNOSTICS=1 cargo run --release -p solver-bench -- maros-meszaros --problem BOYD1 --max-iter 300
MINIX_DIAGNOSTICS=1 cargo run --release -p solver-bench -- maros-meszaros --problem BOYD2 --max-iter 300
```

### What you should look for
- Do you see `nt scaling fallback: ...` lines?
  - If yes, that was almost certainly poisoning the Newton system.
  - After the patch, those should become rare; and when they happen, they should no longer be catastrophic.

- Does BOYD1’s `rel_d` drop rapidly once μ is small?
  - If yes: you were solve-accuracy / scaling limited.

- Does BOYD2 start decreasing μ/gap (even slowly)?
  - If yes: the previous combination of (forced) regularization and scaling was distorting the steps.

## If BOYD1 still stalls after this
The next *targeted* knob to try is increasing solve accuracy only near the end:
- bump `kkt_refine_iters` to 3–5.
- lower `dynamic_reg_min_pivot` to ~1e-10 (so dynamic bumps don’t dominate late).

If that fixes it, the longer-term solution is:
- adaptive refinement based on residual norms, and
- split regularization parameters for `P`-block vs `H`-block (Clarabel-style `delta`/`rho`), rather than a single global shift.

## If BOYD2 still fails after this
The long-term fix is structural:
- bounds-as-rows should be eliminated (or handled as a separate diagonal block) so you don’t inflate the KKT system by ~n rows and introduce identity-structure constraints into the sparse factorization.

But the patch above should still help diagnose whether you’re failing because of scaling/regularization vs something else.
