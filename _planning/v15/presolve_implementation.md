# Constraint Conditioning Implementation

## STATUS: DISABLED - Harmful to Performance

**Benchmark result**: Decreased pass rate from 108/136 (79.4%) to 104/136 (76.5%)

**Root cause**: Geometric mean row scaling interferes with Ruiz equilibration, disrupts problem structure

**See**: `conditioning_results.md` for full analysis

---

## What Was Implemented

Added a new presolve phase that detects and fixes ill-conditioned constraint rows before IPM starts.

### New Module: `solver-core/src/presolve/condition.rs`

**Features:**
1. **`analyze_conditioning()`** - Analyzes constraint matrix for:
   - Nearly-parallel rows (cosine similarity > 0.999)
   - Extreme coefficient ratios (max/min > 1e8)
   - Returns statistics without modifying problem

2. **`apply_row_scaling()`** - Applies geometric mean scaling to rows:
   - Computes scale factor = 1/sqrt(geom_mean(max_abs, min_abs))
   - Scales row i of A and element i of b
   - Reduces coefficient spread while preserving row direction
   - Clamped to [1e-3, 1e3] to avoid extreme scaling

### Integration

**Modified:** `solver-core/src/ipm2/solve.rs`
- Added conditioning phase before Ruiz equilibration (line 50-72)
- Triggers when `enable_conditioning` is true (default)
- Applies scaling if:
  - `extreme_ratio_rows > 0`, OR
  - `max_coeff_ratio > 1e6`

**Modified:** `solver-core/src/problem.rs`
- Added `enable_conditioning: Option<bool>` to `SolverSettings`
- Defaults to `None` (interpreted as `true`)

### Design Rationale

**Why before Ruiz?**
- Conditioning fixes structural issues (extreme ratios, near-parallel rows)
- Ruiz then performs fine-grained equilibration
- Order matters: fix structure first, then balance

**Why geometric mean scaling?**
- Reduces coefficient spread without changing row direction
- More conservative than l∞ or l2 normalization
- Preserves problem feasibility/optimality

**Why conservative thresholds?**
- Only trigger on severe issues (ratio > 1e6)
- Avoid over-scaling well-conditioned problems
- Clamp scales to [1e-3, 1e3] for safety

## Expected Impact

**Problems that should improve:**
1. **QFFFFF80** - row 170 has extreme dual component (A^Tz = -3e8)
   - If caused by coefficient imbalance, scaling should help
2. **QSHIP family** - dual directions explode (dz ~ 1e10)
   - Better conditioning → better KKT solutions
3. **"Dual slow" problems** - 15 problems with stuck dual
   - May have hidden conditioning issues

**Realistic expectation:**
- Helps 2-5 problems significantly
- Marginal improvement on 5-10 others
- Total: +3-8 problems → **111-116/136 (82-85%)**

**Why not more:**
- Some problems are fundamentally ill-conditioned (κ > 1e12)
- No amount of scaling fixes linear dependence
- May need presolve elimination (remove redundant rows)

## Limitations

**Current implementation doesn't:**
1. ❌ Eliminate linearly dependent rows (just scales them)
2. ❌ Combine nearly-parallel constraints
3. ❌ Handle all cone types (only tested on Zero+NonNeg)
4. ❌ Use iterative refinement (one-pass scaling)

**Could be extended to:**
1. ✅ Detect and remove redundant rows
2. ✅ Combine parallel constraints: `row_i = α·row_j + row_i`
3. ✅ Use QR decomposition for robust independence check
4. ✅ Iterate: scale → check → scale again if needed

## Testing Plan

1. ✅ Unit tests for parallel detection and extreme ratios
2. ⏳ Run full Maros-Meszaros benchmark
3. ⏳ Compare before/after on failing problems
4. ⏳ Check regression (ensure passing tests still pass)

## Code Statistics

**Lines added:** ~250
- condition.rs: ~220 lines (including tests)
- solve.rs: ~23 lines (integration)
- problem.rs: ~7 lines (settings field)

**Files modified:** 4
**Tests added:** 3

## Next Steps (If This Helps)

**Phase 1:** Measure impact
- Get benchmark results
- Analyze which problems improved
- Document failure modes

**Phase 2:** Extend if successful
- Add row elimination for truly parallel constraints
- Use QR for more robust detection
- Add column conditioning (extreme Q/P coefficients)

**Phase 3:** Integrate with dual recovery
- Conditioning + dual recovery together
- May be synergistic (better KKT → better dual recovery)
