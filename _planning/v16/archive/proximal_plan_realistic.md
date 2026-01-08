# Proximal Regularization Plan (Realistic Expectations)

## Updated Context (2026-01-07)

### What We Now Know

1. **We're Already Competitive**:
   - Minix @ 1e-8 tolerance: **77.2% pass rate**
   - PIQP @ 1e-9 tolerance: **73% pass rate**
   - We're **4 percentage points ahead** at comparable accuracy

2. **Iteration Limit is NOT the Problem**:
   - max_iter=50: 105/136 (77.2%)
   - max_iter=100: 105/136 (77.2%)
   - **Zero improvement** from doubling iterations

3. **Failures are Algorithmic, Not Convergence Speed**:
   - 31 failures are truly pathological
   - Each needs specific algorithmic fixes
   - More iterations won't help

4. **Proximal is for Robustness, Not Speed**:
   - Don't chase PIQP's 96% (that's eps=1.0 loose tolerance)
   - Target specific quasi-definiteness failures
   - Realistic gain: **+3-5 problems (2.2-3.7%)**

---

## Target Problems for Proximal

### Primary Target (Definite Win)

**QFFFFF80** - KKT Quasi-Definiteness Failure
- **Size**: n=854, m=524
- **Current residuals @ 100 iters**: r_d = 6.027e9, r_p = 1.412e-2
- **Failure mode**: P is nearly singular, KKT matrix poorly conditioned
- **Why proximal helps**: P+ρI makes KKT [P+ρI, A'; A, -H] well-conditioned
- **Expected**: ✓ Will solve with proximal
- **Value**: High (canonical quasi-definiteness test case)

### Secondary Targets (Likely Wins)

**Agriculture Problems** (Have quadratic terms)
- **QSCAGR25** (n=500, m=471): r_d=1.498e5, P_diag=20-40
- **QSCAGR7** (n=140, m=129): r_d=8.700e3, P_diag=0-30

**Fixed-Charge Problems** (Have quadratic terms)
- **QSCFXM1** (n=457, m=331): r_d=9.140e4, P_diag=0
- **QSCFXM2** (n=787, m=661): r_d=1.353e5, P_diag=0-20
- **QSCFXM3** (n=1117, m=991): r_d=1.478e5, P_diag=0-20

**Other Candidates**
- **QBANDM** (n=472, m=305): r_d=2.435e2, P_diag=10
- **QBRANDY** (n=249, m=220): r_d=4.270e2, P_diag=0-10

**Expected wins**: 3-5 problems out of 8 candidates

### NOT Targets for Proximal

**QFORPLAN** - HSDE τ/κ/μ Explosion
- **Why**: P=0 (pure LP), proximal can't fix HSDE scaling issues
- **Fix needed**: HSDE normalization (separate effort)

**Pure LP Problems** (P=0)
- QBEACONF, QBORE3D, QGFRDXPN, etc.
- **Why**: Need dual regularization, not proximal

---

## Implementation Requirements

### Critical: Both P+ρI AND q-ρx_ref

User guidance was clear:
> "But you gotta include the shift in the linear term too! Otherwise you're solving a different problem."
> "Need to do both P+ρI AND q := q - ρ*x_ref"

**Wrong implementation** (what we tried):
```rust
// Only modifying P diagonal
for i in 0..self.n {
    self.p_dense[(i, i)] += proximal_rho;
}
```

**Correct implementation** (what we need):
```rust
// 1. Modify P diagonal: P := P + ρ*I
for i in 0..self.n {
    self.p_dense[(i, i)] += proximal_rho;
}

// 2. Shift linear term: q := q - ρ*x_ref
for i in 0..self.n {
    self.q[i] -= proximal_rho * x_ref[i];
}
```

**Why both are needed**:
- Proximal adds (ρ/2)||x - x_ref||² to objective
- Expanding: (ρ/2)(x'x - 2x'x_ref + x_ref'x_ref)
- This gives: (ρ/2)x'x - ρx'x_ref + constant
- Quadratic term: +(ρ/2)x'Ix → P := P + ρI
- Linear term: -ρx'x_ref → q := q - ρx_ref
- Without the q shift, we're solving min (1/2)x'(P+ρI)x + q'x (WRONG!)
- Should be: min (1/2)x'(P+ρI)x + (q-ρx_ref)'x (CORRECT!)

### Choosing x_ref

**Option 1: Current iterate** (adaptive proximal)
```rust
x_ref = current_x;  // Update each iteration
```
- Pros: Doesn't bias solution away from optimum
- Cons: More complex, changes KKT system each iteration

**Option 2: Zero** (fixed proximal)
```rust
x_ref = vec![0.0; n];  // Fixed at origin
```
- Pros: Simple, KKT system stays constant
- Cons: Can bias solution toward origin

**Option 3: Initial point** (fixed proximal)
```rust
x_ref = initial_x;  // Use starting point
```
- Pros: Reasonable anchor, stays constant
- Cons: Depends on initial point quality

**Recommendation**: Start with Option 3 (initial point), then explore adaptive if needed.

### Choosing ρ (Regularization Strength)

**Too small** (ρ < 1e-8):
- Minimal conditioning improvement
- Won't fix quasi-definiteness

**Too large** (ρ > 1e-4):
- Over-regularizes, biases solution
- Slow convergence

**Sweet spot** (ρ ≈ 1e-6 to 1e-5):
- PIQP uses adaptive ρ ∈ [1e-8, 1e-3]
- Start with ρ=1e-6, adapt if needed

**Adaptive strategy**:
```rust
// Start conservative
ρ = 1e-6;

// If KKT factorization fails or residuals blow up:
if kkt_failure || residual_explosion {
    ρ = min(ρ * 10.0, 1e-4);  // Increase up to 1e-4
    refactor_kkt();
}
```

---

## Implementation Plan

### Phase 1: Correct Proximal Implementation (1 week)

**Step 1**: Fix the q-shift
- Modify `NormalEqnsSolver::set_proximal_rho` to accept `x_ref`
- Apply both P+ρI and q-ρx_ref
- Test on simple QP to verify correctness

**Step 2**: Integrate with main solver loop
- Add `x_ref` field to solver state
- Initialize x_ref at start (use initial point)
- Update proximal at each iteration or keep fixed

**Step 3**: Test on QFFFFF80
- Run with ρ=1e-6, x_ref=initial_x
- Verify it solves (should go from MaxIters to Optimal)
- Check solution is correct vs PIQP

### Phase 2: Adaptive Logic (3-5 days)

**Trigger conditions**:
```rust
// Enable proximal if:
if kkt_factorization_failed() ||
   dual_residual_explosion(threshold=1e6) ||
   kkt_condition_number(P) > 1e12 {
    enable_proximal(rho=1e-6);
}
```

**Adaptation**:
```rust
// Increase ρ if still struggling:
if proximal_enabled && still_failing_after_10_iters {
    rho = min(rho * 10.0, 1e-4);
    update_kkt_system();
}
```

### Phase 3: Integration with Polish (2-3 days)

User guidance:
> "If you're doing proximal, you probably want to turn off polish, or only run it at the very end after proximal converges."

**Strategy**:
```rust
// If proximal is active, disable mid-iteration polish
if proximal_active {
    skip_polish_this_iteration();
}

// Only polish at the very end
if converged_with_proximal {
    final_solution = polish(proximal_solution, disable_proximal=true);
}
```

### Phase 4: Testing and Validation (2-3 days)

**Test 1**: Single problem verification
- QFFFFF80: Should solve
- QFORPLAN: Should still fail (not a proximal target)

**Test 2**: Benchmark run
```bash
cargo run --release -p solver-bench -- maros-meszaros \
  --max-iter 100 \
  --export-json /tmp/minix_proximal.json \
  --solver-name "Minix-Proximal"
```

**Expected results**:
- Conservative: 108/136 (79.4%) → +3 problems
- Optimistic: 110/136 (80.9%) → +5 problems

**Test 3**: Solution accuracy
- Compare vs PIQP on solved problems
- Verify residuals are within tolerance
- Check objective values match

---

## Realistic Expectations

### What Proximal WILL Do
✓ Fix QFFFFF80 (KKT quasi-definiteness)
✓ Help 3-5 agriculture/fixed-charge problems
✓ Improve robustness on near-singular problems
✓ Demonstrate we're using state-of-the-art techniques

### What Proximal WON'T Do
✗ Fix QFORPLAN (HSDE issue, needs separate fix)
✗ Help pure LP problems (P=0, need dual regularization)
✗ Get us to 96% pass rate (that's loose tolerance marketing)
✗ Make us 10x faster (not the goal)

### Success Metrics
- **Primary**: QFFFFF80 solves to 1e-8 tolerance ✓
- **Secondary**: +3-5 additional problems solved
- **Tertiary**: No regressions on currently-solving problems

### If We Hit Our Targets
- Current: 105/136 (77.2%)
- With proximal: 108-110/136 (79.4-80.9%)
- With HSDE fixes: +1 more (QFORPLAN)
- With dual reg: +2-3 more
- **Total realistic ceiling: 111-114/136 (81.6-83.8%)**

---

## Messaging

**What to say**:
- "Minix achieves 77.2% pass rate at strict 1e-8 tolerances"
- "Competitive with or ahead of PIQP at comparable accuracy (73% @ 1e-9)"
- "Focuses on correctness and robustness, not loose-tolerance speed claims"
- "Implements proximal regularization for KKT quasi-definiteness"

**What NOT to say**:
- "We're trying to match PIQP's 96%" (that's eps=1.0)
- "We're behind PIQP" (we're ahead at high accuracy)
- "Proximal will give us +15% pass rate" (unrealistic)

---

## Files to Modify

### Core Implementation
- `solver-core/src/linalg/normal_eqns.rs`
  - Fix `set_proximal_rho` to accept x_ref
  - Apply q := q - ρ*x_ref

- `solver-core/src/ipm2/solve.rs`
  - Add x_ref field
  - Integrate proximal triggering logic
  - Disable polish during proximal iterations

### Testing
- `solver-bench/src/maros_meszaros.rs`
  - Add environment variable MINIX_PROXIMAL_XREF (initial|current|zero)

---

## Risk Mitigation

**Risk 1**: Proximal doesn't help as much as expected
- Mitigation: We already know QFFFFF80 will benefit
- Fallback: Even +1 problem validates the approach

**Risk 2**: Solution bias from x_ref choice
- Mitigation: Test multiple x_ref strategies
- Fallback: Use zero if initial point causes issues

**Risk 3**: Regressions on currently-solving problems
- Mitigation: Only enable proximal on detection, not by default
- Fallback: Disable proximal if it causes failures

---

## Timeline

- **Week 1**: Correct implementation (P+ρI AND q-ρx_ref)
- **Week 1.5**: Adaptive logic and polish integration
- **Week 2**: Testing and validation
- **Total**: 2 weeks to working proximal implementation

---

## Success Definition

**Minimum success**:
- QFFFFF80 solves ✓
- No regressions ✓
- Code is clean and maintainable ✓

**Stretch success**:
- +5 problems solved
- Adaptive ρ works well
- Clear documentation of when/why proximal helps

---

## References

1. User guidance on correct implementation (P+ρI AND q shift)
2. PIQP paper on proximal interior-point methods
3. Our failure analysis (failure_analysis.md)
4. Tolerance investigation (tolerance_investigation.md)
