# V26 Work Log

## Session Summary (2026-01-14)

### Major Accomplishment: SOC Arrowhead Optimization

**Problem**: PRIMALC8 was 209x slower than Clarabel (1183ms vs 5.65ms), with 80% of time in KKT factorization.

**Root Cause**: Dense SOC blocks were being inserted into the global KKT matrix. For SOC cones, the scaling matrix Q_w is dense (O(dim²) entries). For QP-to-SOCP epigraph reformulations, this creates massive fill-in.

**Solution Implemented**: Arrowhead factorization for SOC cones.

The key insight is that Q_w can be decomposed as:
```
Q_w = Q_arrowhead + 2 * w̄ * w̄ᵀ
```

Where Q_arrowhead has only O(dim) entries (first row + diagonal), and w̄ is the tail vector of the NT scaling point.

Implementation details:
1. **New `HBlockPositions::SocArrowhead` variant** in `kkt.rs`:
   - Stores positions for first row and diagonal only
   - Used for SOC cones with dim >= 8

2. **Sherman-Morrison/Woodbury correction**:
   - After solving K_arrowhead * y = b, apply rank-1 correction
   - Formula: For (A - uu')⁻¹b = A⁻¹b + z*(u'y)/(1 - u'z)
   - Key sign fix: negative rank-1 update requires different formula than positive

3. **Critical bug fix**: Initial implementation had sign error in Sherman-Morrison formula
   - Changed Gram matrix from `1.0 + dot` to `1.0 - dot`
   - Changed final step from subtraction to addition
   - Without this fix, DUALC1 failed with NumericalError

**Results**:
- PRIMALC8: 1183ms → 23ms (**51x speedup**)
- PRIMALC8 iterations: 23 → 20
- Committed as a48a063

### Phase B Investigation (SOC Centrality)

Investigated Phase B items from the planning document:

1. **B1: SOC neighborhood check in line search** - Already implemented in `centrality_ok_nonneg_trial` (lines 595-655)
   - Uses NT-scaled complementarity eigenvalues
   - Relaxed bounds: beta_soc = beta * 0.1, gamma_soc = gamma * 10

2. **B2: Sigma bump escape hatch** - Already implemented (lines 1806-1809)
   - When alpha stalls with centrality limiter, sigma bumped to max(sigma, 0.3)
   - Second retry uses sigma_cap

3. **B3: SOC proximity step control** - Already implemented (lines 2205-2243)
   - Uses same NT-scaled complementarity approach

**Conclusion**: All Phase B items were already implemented. The iteration count difference (20 vs Clarabel's 8) is likely due to:
- Different problem formulations (epigraph SOCP vs direct QP)
- Clarabel optimizes for direct quadratic objectives

### DUALC1 Investigation

Found that DUALC1 converges well internally (rel_p=4.04e-10 at iteration 21) but the **postsolve recovery** degrades this to rel_p=4.85e-7.

Key observations:
- KKT condition number is 10³³-10³⁵ (severely ill-conditioned)
- min_z reaches 1.2e-13 (near machine epsilon)
- Final status is `~OK` with 30 iterations (stalled from iter 21)

This is a postsolve/recovery issue, not an IPM issue.

### Current SOCP Regression Results (2026-01-14)

**PRIMALC Series** (all pass):
```
PRIMALC1_SOCP: OK    iters=28 rel_p=4.07e-13 rel_d=4.17e-9 gap_rel=4.00e-10 time=14ms
PRIMALC2_SOCP: OK    iters=22 rel_p=3.61e-13 rel_d=6.96e-9 gap_rel=1.88e-10 time=9ms
PRIMALC5_SOCP: OK    iters=14 rel_p=7.44e-12 rel_d=7.05e-9 gap_rel=2.81e-11 time=6ms
PRIMALC8_SOCP: ~OK   iters=20 rel_p=1.60e-11 rel_d=7.18e-8 gap_rel=4.01e-11 time=22ms
```

**DUALC Series** (1 failure):
```
DUALC1_SOCP: ~OK     iters=30 rel_p=4.85e-7 rel_d=9.40e-13 gap_rel=9.49e-12 time=12ms
DUALC2_SOCP: FAIL    status=MaxIters rel_p=2.22e3 rel_d=2.15e1 gap_rel=1.04e0
DUALC5_SOCP: ~OK     iters=15 rel_p=3.63e-8 rel_d=1.73e-9 gap_rel=7.94e-11 time=4ms
DUALC8_SOCP: ~OK     iters=30 rel_p=1.66e-5 rel_d=3.26e-9 gap_rel=1.41e-11 time=26ms
```

**Notes**:
- DUALC2 fails completely - needs investigation
- DUALC1, DUALC8 show postsolve degradation (internal metrics much better)
- All PRIMALC problems pass with excellent performance

### Pre-Arrowhead Performance

| Problem | Time (before) | Time (after) | Iterations | Notes |
|---------|--------------|--------------|------------|-------|
| PRIMALC8 | 1183ms | 22ms | 20 | **51x speedup** |
| DUALC1 | ~22ms | 12ms | 30 | Postsolve issue |

### Remaining Work

1. **Fix DUALC2 failure**
   - Currently fails with MaxIters and rel_p=2.22e3
   - Needs investigation - likely a different issue than other DUALC problems

2. **Investigate postsolve recovery** for DUALC1, DUALC8
   - DUALC1: rel_p degrades from 4e-10 (internal) to 4.85e-7 (final)
   - DUALC8: rel_p=1.66e-5 despite good gap convergence
   - Could be numerical precision in unscaling/recovery

3. **Consider direct QP path**
   - For fair comparison with Clarabel, support direct P matrix
   - Avoids epigraph SOCP reformulation overhead

4. **Further iteration reduction**
   - Current: 20 iterations for PRIMALC8
   - Target: ~8 (Clarabel's level)
   - May require different predictor-corrector strategy

### Direct QP Mode Works (No Need for SOCP Reformulation)

**Key Finding**: The solver already supports direct QP mode with P matrix. The `--socp` flag is only needed for testing SOCP cone support, NOT for running QP problems efficiently.

**Performance Comparison on AUG2D:**
- Direct QP: **197ms, 7 iterations**
- SOCP mode: **>30 seconds** (killed)

**Full MM Suite (136 problems, Direct QP mode):**
```
Total problems:      136
Optimal:             83 (61.0%)
AlmostOptimal:       30 (22.1%)
Combined (Opt+Almost): 113 (83.1%)
Max iterations:      13 (9.6%)
Total time:          234.58s
Geom mean time:      46.2ms
```

**Why SOCP is slower:**
The epigraph SOCP reformulation `min t s.t. (t,1,Lx) ∈ RSOC` embeds the dense Cholesky factor L in the constraint matrix A. This creates O(n²) fill-in even though the arrowhead optimization keeps the SOC scaling block H sparse at O(n).

**Recommendation:** Use direct QP mode (without `--socp`) for QP problems. Only use `--socp` to test SOC cone support on problems that naturally have SOC constraints.

### Time Limit Feature Added

Added `--time-limit-ms` CLI option to the benchmark runner:
- Added `time_limit_ms: Option<u64>` to `SolverSettings`
- Implemented timeout check at start of each IPM iteration
- Returns `SolveStatus::TimeLimit` when exceeded

**Limitation**: The timeout only triggers at iteration boundaries. If a single KKT factorization takes >5s (common for large SOCP problems like AUG2D), we can't interrupt it. A proper mid-iteration timeout would require threading or signals.

**Usage**:
```bash
cargo run --release -p solver-bench -- benchmark --suite mm --socp --max-iter 200 --time-limit-ms 5000
```

### Files Modified

- `solver-core/src/ipm2/solve.rs`:
  - Added time limit check at start of each iteration
  - Returns `TimeLimit` status when exceeded

- `solver-bench/src/main.rs`:
  - Added `--time-limit-ms` CLI argument
  - Added SOCP mode for full MM suite (not just single problems)

- `solver-core/src/linalg/kkt.rs`:
  - Added `HBlockPositions::SocArrowhead` enum variant
  - Added `ShermanMorrisonWorkspace` struct
  - Added `update_soc_arrowhead_in_place` function
  - Modified `build_kkt_matrix_with_perm` for arrowhead sparsity
  - Added Woodbury correction in solve methods

### Technical Notes

**Sherman-Morrison for negative rank-1 updates**:
When KKT has -Q_w (negative), the correction is:
```
K = K_arrowhead - u*uᵀ
```
For (A - uu')⁻¹b:
```
y = A⁻¹b
z = A⁻¹u
x = y + z * (u'y) / (1 - u'z)
```
Note the **plus** sign and **minus** in denominator.

**SOC eigenvalue computation (stable)**:
```
λ_max = t + ||x̄||
λ_min = (t² - ||x̄||²) / (t + ||x̄||)  // avoids cancellation
```

**SOC neighborhood condition**:
```
β_soc · μ ≤ sqrt(λ_min(s) · λ_min(z))
sqrt(λ_max(s) · λ_max(z)) ≤ γ_soc · μ
```

### CVXPY Interface: P + SOC Works Correctly (2026-01-14)

**Critical Finding**: CVXPY sends P matrix directly to solvers, NOT the epigraph SOCP form.

**Investigation**:
- Checked CVXPY's [clarabel_conif.py](https://github.com/cvxpy/cvxpy/blob/master/cvxpy/reductions/solvers/conic_solvers/clarabel_conif.py)
- CVXPY extracts P from data and passes it directly: `_solver = clarabel.DefaultSolver(P, q, A, b, cones, _settings)`
- Clarabel's KKT system has form `[P + εI, A'; A, -H]` - same as ours

**Our solver already supports this**:
- `ProblemData` accepts optional `P` with any cone types
- `build_kkt_matrix` places P directly in top-left block
- Python bindings accept `p_indptr`, `p_indices`, `p_data` for P matrix
- No reformulation needed

**Verification**: Added tests in `integration_tests.rs`:
1. `test_qp_with_soc_constraint` - P + SOC (passes)
2. `test_larger_qp_with_soc` - P + SOC + equality (passes)

**Why `--socp` benchmark was slow**:
The `to_socp_form()` function in `qps.rs` artificially converts QP to epigraph form:
```
min t + q'x  s.t. (t, 1, Lx) ∈ RSOC
```
This embeds dense L (Cholesky of P) into A matrix, creating O(n²) fill-in.
This is **NOT** what CVXPY does - it's a worst-case test scenario.

**Conclusion**: For CVXPY compatibility, no changes needed. Our solver handles P + SOC directly.
The `--socp` benchmark flag tests an artificial worst case. Real CVXPY usage sends P directly.

### CVXPY Benchmark: Minix vs Clarabel (2026-01-14)

Created CVXPY solver interface (`solver-py/minix_cvxpy.py`) and benchmark script.

**Results (Minix is 35% faster overall):**

| Problem      | Clarabel (ms) | Minix (ms) | Ratio |
|--------------|---------------|------------|-------|
| QP-50        | 30.7          | 21.4       | 0.70x |
| QP-100       | 26.8          | 23.8       | 0.89x |
| QP-200       | 101.4         | 71.4       | 0.70x |
| SOCP-50      | 9.4           | 6.7        | 0.71x |
| SOCP-100     | 28.1          | 19.9       | 0.71x |
| QP+SOC-50    | 4.4           | 6.1        | 1.39x |
| QP+SOC-100   | 4.2           | 5.8        | 1.40x |
| Portfolio-50 | 10.0          | 4.8        | 0.47x |
| Portfolio-100| 24.7          | 12.8       | 0.52x |
| LASSO-100x50 | 9.1           | 12.1       | 1.32x |
| LASSO-200x100| 124.9         | 59.8       | 0.48x |
| **Total**    | **373.7ms**   | **244.5ms**| **0.65x** |

**Key findings:**
- Minix faster on: QP, SOCP, Portfolio, large LASSO
- Clarabel slightly faster on: QP+SOC, small LASSO
- Both solvers give same objective values (< 1e-4 difference)

**Files created:**
- `solver-py/minix_cvxpy.py` - CVXPY solver interface for Minix
- `solver-py/benchmark_cvxpy.py` - Benchmark script

### Sherman-Morrison Precomputation Fix (2026-01-14)

**Problem**: QP+SOC-200 was 4x slower than Clarabel, and performance degraded rapidly with problem size:
- QP+SOC-50: 1.39x slower
- QP+SOC-100: 1.40x slower
- QP+SOC-200: 4.38x slower
- QP+SOC-500: 36x slower (extrapolated)

**Root Cause**: In `apply_woodbury_correction`, we were computing `z_i = K^{-1} u_i` on EVERY solve call. This involved k KKT solves (one per SOC block) per RHS, giving O(k × num_solves) expensive operations per IPM iteration.

**Solution**: Precompute z_i vectors once per factorization:
1. Added `precomputed: bool` flag to `ShermanMorrisonWorkspace`
2. Split into two methods:
   - `precompute_woodbury_data`: Computes z_i and Gram matrix G = I - U'Z once
   - `apply_woodbury_correction`: Only applies the correction (no KKT solves)
3. Call precompute immediately after factorization
4. Reset `precomputed = false` in `reset_woodbury_workspace`

**Results (after fix):**

| Problem      | Clarabel (ms) | Minix (ms) | Ratio |
|--------------|---------------|------------|-------|
| QP+SOC-50    | 15.4          | 4.1        | 0.27x |
| QP+SOC-100   | 3.4           | 3.9        | 1.15x |
| QP+SOC-200   | 3.8           | 4.7        | 1.23x |

**Improvement**: QP+SOC-200 went from 4.38x slower to 1.23x (similar performance). QP+SOC-50 is now 3.7x faster than Clarabel.

**Full Updated Benchmark Results (2026-01-14):**

| Problem        | Clarabel (ms) | Minix (ms) | Ratio | Notes |
|----------------|---------------|------------|-------|-------|
| QP-50          | 20.6          | 14.4       | 0.70x | |
| QP-100         | 22.5          | 19.6       | 0.87x | |
| QP-200         | 74.8          | 99.9       | 1.33x | |
| QP-500         | 1146.8        | 1072.3     | 0.94x | |
| SOCP-50        | 50.6          | 30.6       | 0.60x | unbounded |
| SOCP-100       | 42.3          | 245.0      | 5.79x | unbounded, slow detection |
| SOCP-200       | 84.9          | 4131.5     | 48.6x | unbounded, hit max_iter |
| QP+SOC-50      | 15.4          | 4.1        | 0.27x | **fixed** |
| QP+SOC-100     | 3.4           | 3.9        | 1.15x | **fixed** |
| QP+SOC-200     | 3.8           | 4.7        | 1.23x | **fixed** |
| Portfolio-50   | 12.6          | 5.4        | 0.43x | |
| Portfolio-100  | 23.4          | 11.1       | 0.48x | |
| Portfolio-200  | 98.3          | 46.0       | 0.47x | |
| LASSO-100x50   | 15.6          | 12.1       | 0.78x | |
| LASSO-200x100  | 103.7         | 57.0       | 0.55x | |
| LASSO-500x250  | 542.5         | 511.2      | 0.94x | |

**Remaining Issue**: Pure SOCP unboundedness detection is slow (SOCP-100: 5.79x, SOCP-200: 48.6x slower). This is a different issue - related to how we detect dual infeasibility certificates, not the Woodbury correction.

**Files Modified:**
- `solver-core/src/linalg/kkt.rs`:
  - Added `precomputed: bool` to `ShermanMorrisonWorkspace`
  - Added `precompute_woodbury_data` method
  - Modified `apply_woodbury_correction` to only apply (no solves)
  - Call precompute after `factor_kkt` in all backends
  - Reset flag in `reset_woodbury_workspace`
- `solver-py/minix_cvxpy.py`:
  - Fixed STATUS_MAP: "max_iterations" instead of "max_iters"
