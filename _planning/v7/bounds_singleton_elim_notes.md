# Bounds / singleton elimination patch

This patch focuses on two concrete (and safe) improvements that are easy to parallelize with the NT-scaling work:

1. **Fix `initial_scaling` ordering in `solve_ipm`**  
   `KktSolver::new_with_singleton_elimination(...)` needs the *initial scaling structure* (diagonal vs SOC blocks) at construction time so it can decide which singleton rows are safe to eliminate (only diagonal blocks).  
   In the current file ordering, `initial_scaling` was declared **after** it was used. This patch moves the `initial_scaling` block earlier so the code compiles and the intended singleton elimination path is actually active.

2. **Clean up `SingletonElim` to remove the `row_map` dead field warning**  
   `row_map` was only needed transiently to build the reduced `A` once; it is not required afterward (we expand solutions using `kept_rows` and the stored singleton metadata). The patch removes the field to eliminate the warning.

3. **Diagnostics so you can verify the elimination is actually happening**  
   If you run with `MINIX_DIAGNOSTICS=1`, construction of the KKT solver prints a single line like:
   ```
   kkt presolve: singleton elimination enabled: m_full=93279 m_reduced=18 eliminated=93261 diag_update_cols=93261
   ```
   This is the easiest way to confirm that BOYD1/BOYD2 bounds rows are being removed from the KKT factorization.

## Why this matters for BOYD1/BOYD2

Those problems have **tens of thousands of explicit bound rows** (each a singleton `±x_i + s = b`, `s >= 0`).  
Eliminating them from the KKT system turns the KKT dimension from roughly:

- **Before:** `(n + m_full)` where `m_full ≈ n + m_eq`
- **After:** `(n + m_reduced)` where `m_reduced ≈ m_eq` (often only ~18 for BOYD1)

This reduces factorization size/fill and usually improves numerical conditioning too.

## How to apply

From repo root:
```bash
git apply minix_bounds_singleton_elim.patch
```

## How to verify

Run BOYD1 with diagnostics enabled:
```bash
MINIX_DIAGNOSTICS=1 cargo run --release -p solver-bench -- maros-meszaros --problem BOYD1 --max-iter 5
```

Look for the `kkt presolve:` line. If you see a large `m_full -> m_reduced` drop, the elimination is active.

## Notes / follow-ups (optional)

- If you still see `m_reduced == m_full`, then either:
  - bounds are not being added as singleton rows (unlikely for QPS), or
  - those rows are not in a **diagonal scaling block** (e.g. they were placed in a non-diagonal cone by mistake).
