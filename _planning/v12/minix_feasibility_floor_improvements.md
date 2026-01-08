# Feasibility-floor debugging + small cleanup (ipm2)

This patch focuses on **actionable diagnostics** for the “primal feasibility floor” regime (e.g. YAO) and fixes two warnings you showed:

- `solver-core/src/ipm2/solve.rs`: `orig_m` unused
- `solver-core/src/ipm2/solve_normal.rs`: `correction` unused

## What the patch changes

### 1) Print *which rows* dominate the primal residual
At the end of a solve (in the existing diagnostics print block), when `primal_ok == false`, it now prints the **top-5 rows by |r_p|** where:

- `r_p = A x + s - b` (the same residual used in the metrics)
- Each printed row is tagged as:
  - `orig`  → index `< orig_m` (constraint came from the original problem)
  - `bound` → index `>= orig_m` (constraint was introduced by “bounds-as-constraints”)

This directly answers the question: *“Are the singleton / bound-derived rows dominating the max residual?”*.

### 2) Print top dual residual components when `dual_ok == false`
Same idea, but for `r_d` (top-5 components).

### 3) Keep `r_p` / `r_d` vectors around for reporting
The final-metrics block is restructured to return `(final_metrics, rp_report, rd_report)` instead of only `final_metrics`, so we can print the violating indices.

### 4) Warning cleanup
- `orig_m` is now used (and we also compute `full_m` for nicer diagnostics output).
- `correction` in `solve_normal.rs` is renamed to `_correction` to silence the unused-variable warning (until you decide to turn that into a real Gondzio/MCC term).

## How to apply

From repo root:

```bash
git apply minix_feasibility_floor_improvements.patch
```

Re-run your bench:

```bash
MINIX_DIAGNOSTICS=1 cargo run --release -p solver-bench -- maros-meszaros --problem YAO --max-iter 50
```

## Expected output change
When `primal_ok=false`, you should now also see something like:

- `top |r_p| rows ... (kind=orig|bound)`
- a list of 5 row indices and signed residual values

If the top violations are all `kind=bound`, that strongly suggests either:

- bounds postsolve inconsistency (recover / scaling / sign convention), or
- singleton elimination interaction with reg/refinement.

## Next steps (not in this patch)

If this diagnostic shows that the primal residual is concentrated in `kind=orig`, then the likely culprit is a **linear-solve accuracy / degeneracy** issue rather than presolve.

If it concentrates in `kind=bound`, focus on:

- “bounds-as-constraints” construction and sign conventions,
- postsolve recovery of `s,z` for those rows,
- whether those rows are the ones eliminated by singleton elimination.
