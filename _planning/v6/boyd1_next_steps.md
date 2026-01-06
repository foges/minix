# BOYD1 still hitting MaxIters — likely root cause + next patches

Your BOYD1 trace is **not** the “polishing / last‑mile dual feasibility” failure anymore.  
It’s diverging early:

- `mu` grows from ~8e7 → ~3e11 in 50 iters (should trend down).
- step sizes collapse to ~1e‑7–1e‑5 very early.
- `rel_p ≈ O(1)` and `gap_rel ≈ O(1)` after 50 iters → you’re **not** close to feasible/optimal.

Also: `min_s=0.0` and `min_z<0` in your per‑iter printout are **not** automatically a bug:
- BOYD1 has **18 equality rows** (`Zero` cone). Equality slack `s_eq` is fixed at 0 → global `min_s` will always be 0.
- Equality dual variables are **free**, so `z_eq` can be negative → global `min_z` can be negative.

So the actionable question is: **what is happening on the NonNeg (bounds) block?**  
Your `alpha_sz` is tiny, meaning some NonNeg component hits the step‑to‑boundary limiter.

## The biggest “code smell” I see: NonNeg NT scaling can spuriously fail, and the fallback is wrong

In the current codebase we reviewed, two things combine into a catastrophe on large, bound‑heavy problems like BOYD1:

### 1) `NonNegCone::is_interior_primal` uses a *relative* threshold
```rust
tol = 1e-12 * max(1, ||s||_∞)
s_i > tol
```

With huge dynamic range (very common on BOYD1: some entries tiny, some huge), this test flags *small-but-positive* entries as “not interior”. That causes **NT scaling to error** even when `s>0, z>0`.

### 2) The NT-scaling fallback in `predcorr.rs` uses `sqrt(s/z)` for `ScalingBlock::Diagonal`

But for NonNeg, `ScalingBlock::Diagonal` is used as **H = S Z^{-1} = s/z** in the condensed KKT and in `ds = -d_s - H dz`.

Using `sqrt(s/z)` instead means you’re solving the *wrong Newton system* (wrong KKT matrix and wrong ds), which can absolutely produce:
- exploding `mu`
- extremely small fraction-to-boundary steps
- no progress in feasibility/objective

## Patch: make NonNeg interior checks absolute + fix diagonal fallback to use s/z

Apply this patch:

- `solver-core/src/cones/nonneg.rs`
  - Make interior check **absolute** (`x.is_finite() && x > 1e-300`)
- `solver-core/src/ipm/predcorr.rs`
  - Fix fallback diagonal scaling to use **clamped s/z** (NOT sqrt)

**Patch file:** `minix_boyd1_nt_interior_fix.patch` (download link below)

## What to do right after applying

1) Rerun BOYD1 with your diagnostics:
```bash
MINIX_DIAGNOSTICS=1 cargo run --release -p solver-bench -- maros-meszaros --problem BOYD1 --max-iter 50
```

2) Add/verify a one-line diagnostic when the NT scaling fallback triggers (if you don’t already have this):
- For BOYD1, you want: **fallback never triggers for NonNeg** unless you truly left the cone.

3) If BOYD1 still stalls after this patch:
- bump KKT refinement:
  - `kkt_refine_iters = 5` (or 7)
- try a bigger base reg just for BOYD-class problems:
  - `static_reg = 1e-5` or `1e-4`
- enable centrality helpers *only for BOYD*:
  - `mcc_iters = 1`
  - `line_search_max_iters = 10`
  - `centrality_beta=0.1`, `centrality_gamma=10.0`

(Those are “algorithmic robustness knobs” not “bugs”, but BOYD is exactly the kind of instance where MOSEK uses very aggressive centrality/linear algebra tricks.)

## Why this aligns with your trace

Your per‑iter log has **no sign of converging complementarity** (mu is increasing) while steps are boundary‑limited. That is a classic symptom of:
- a wrong H (scaling) being used in the KKT and ds recovery, OR
- KKT solve producing directions that destroy centrality because of very poor conditioning.

The patch attacks the first (more “buggy”) category directly.

