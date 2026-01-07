# Research: Clarabel and MOSEK Approaches

## Clarabel AlmostOptimal Status

**Source:** [Clarabel API Settings](https://clarabel.org/stable/api_settings/)

### Full Accuracy Thresholds (Optimal)
- `tol_gap_abs = 1e-8` - absolute duality gap
- `tol_gap_rel = 1e-8` - relative duality gap
- `tol_feas = 1e-8` - feasibility
- `tol_ktratio = 1e-6` - KT ratio

### Reduced Accuracy Thresholds (AlmostOptimal)
- `reduced_tol_gap_abs = 5e-5` (500x looser)
- `reduced_tol_gap_rel = 5e-5` (500x looser)
- `reduced_tol_feas = 1e-4` (12.5x looser)
- `reduced_tol_ktratio = 1e-4` (10x looser)

**Key insight:** AlmostOptimal is 500x looser on gap, 12.5x on feasibility. This tracks "close but not quite" solutions.

**Use case:** Branch-and-bound, numerically challenging problems with wide value ranges.

---

## Clarabel KKT Regularization

**Source:** [Clarabel arXiv paper](https://arxiv.org/html/2405.12762v1), [Clarabel control parameters](https://cran.r-project.org/web//packages/clarabel/clarabel.pdf)

### Static Regularization (Always On)
- `static_regularization_enable = TRUE` (default)
- `static_regularization_constant = 1e-8`
- **Effect:** Adds `1e-8 * I` to KKT diagonal for numerical stability

### Dynamic Regularization (Adaptive)
- `dynamic_regularization_enable = TRUE` (default)
- `dynamic_regularization_eps = 1e-13` (minimum eigenvalue threshold)
- `dynamic_regularization_delta = 2e-7` (regularization increment)
- **Effect:** Detects ill-conditioning, adds adaptive δ to diagonal

### Iterative Refinement
- `iterative_refinement_enable = TRUE` (default)
- **Effect:** After KKT solve, refines solution to reduce residual

**Performance note:** "Both ClarabelGPU and ClarabelRs are faster and more numerically stable than Mosek even with the presolve step"

**Warning:** "KKT matrix K is extremely ill-conditioned. Caution is required when using mixed precision for numerically hard problems, e.g., conic programs with exponential cones."

---

## MOSEK Infeasibility Detection

**Source:** [MOSEK infeasibility docs](https://docs.mosek.com/9.2/toolbox/debugging-infeas.html)

### Farkas Certificates
- **Method:** MOSEK uses Farkas certificates to prove primal/dual infeasibility
- **Parameter:** `MSK_DPAR_INTPNT_CO_TOL_INFEAS` controls conservativeness
- Smaller value = more conservative about declaring infeasible

### Dual Infeasibility Reports
- Lists constraints involved in infeasibility
- Nonzero values in certificate indicate problematic constraints
- Enable with: `MSK_IPAR_INFEAS_REPORT_AUTO = MSK_ON`

### Certificate Quality
- "Infeasible problem status may be just an artifact of numerical issues appearing when the problem is badly-scaled, barely feasible or otherwise ill-conditioned"
- "This may be visible in the solution summary if the infeasibility certificate has poor quality"

**Key insight:** MOSEK is careful about numerical issues vs. true infeasibility

---

## How Our Solver Compares

### Current Minix Thresholds (from code)
Looking at our code, we likely use:
- Primal feasibility: ~1e-8
- Dual feasibility: ~1e-8
- Gap tolerance: ~1e-8

**Issue:** We have no "AlmostOptimal" tier, so solutions at 1e-5 are treated same as solutions at 1e-1

### Current Minix KKT Handling
- ✅ We use QDLDL for KKT factorization
- ❌ No static regularization (no `+ δI` term)
- ❌ No dynamic regularization (no ill-conditioning detection)
- ❌ No iterative refinement

### What We Should Add

1. **AlmostOptimal Status (Easy, High Value)**
   - Track solutions meeting 5e-5 gap, 1e-4 feasibility
   - Better visibility into "almost working" fixes
   - Helps identify which problems are close

2. **Static KKT Regularization (Medium, High Value)**
   - Add `1e-8 * I` to KKT diagonal
   - Should stabilize ill-conditioned systems
   - Clarabel defaults this to ON

3. **Dynamic Regularization (Hard, High Value)**
   - Detect near-singular KKT (condition number estimate)
   - Adaptively increase δ when needed
   - Requires eigenvalue estimation or SVD

4. **Iterative Refinement (Medium, Medium Value)**
   - After KKT solve, compute residual
   - If large, refine: `x' = x + solve(KKT, r)`
   - Cheap if residual is small

---

## Implementation Priority

### Phase 1: AlmostOptimal Status (30 min)
- Add status enum variant
- Track reduced thresholds in metrics
- Update benchmark output
- **Expected:** Better visibility, no performance change

### Phase 2: Static KKT Regularization (1-2 hours)
- Modify KKT assembly to add `δI` (default 1e-8)
- Add `kkt_regularization` to SolverSettings
- Benchmark with different δ values
- **Expected:** +3-8 problems from stabilization

### Phase 3: Extended Dual Recovery (1 hour)
- Relax thresholds (iter >= 10, rel_p < 1e-5)
- Add periodic retry logic
- **Expected:** +5-10 problems

### Phase 4: Iterative Refinement (2-3 hours)
- Compute KKT residual after solve
- Refine if ||r|| > threshold
- **Expected:** +2-5 problems

### Phase 5: Dynamic Regularization (1 day)
- Add condition number estimation
- Adaptive δ based on conditioning
- **Expected:** +5-10 problems

---

## Sources
- [Clarabel API Settings](https://clarabel.org/stable/api_settings/)
- [Clarabel arXiv paper](https://arxiv.org/html/2405.12762v1)
- [Clarabel control parameters (CRAN)](https://cran.r-project.org/web//packages/clarabel/clarabel.pdf)
- [MOSEK infeasibility debugging](https://docs.mosek.com/9.2/toolbox/debugging-infeas.html)
- [Julia discussion: AlmostOptimal status](https://discourse.julialang.org/t/resolving-almost-optimal-solution-in-clarabel-tulip-does-well/112319)
