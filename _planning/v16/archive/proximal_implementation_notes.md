# Proximal Regularization Implementation Notes

## Date: 2026-01-07

## Key Discovery: Implementation Already Complete!

When investigating the proximal regularization implementation, I discovered that **both components** (P+ρI AND q-ρx_ref) are **already fully implemented** in the codebase.

### Where the Implementation Lives

#### 1. P+ρI Modification (KKT Solver)

**File**: `solver-core/src/linalg/kkt.rs`

**Lines 1062-1064** (KKT matrix assembly):
```rust
// Add proximal_rho here (before QDLDL's static_reg) for IP-PMM.
// Total P diagonal regularization = proximal_rho + static_reg
for i in 0..self.n {
    add_triplet(i, i, self.proximal_rho, &mut tri);
}
```

**Lines 1266-1267** (P diagonal base):
```rust
fn fill_p_diag_base(&mut self, p: Option<&SparseSymmetricCsc>) {
    // Start with proximal regularization on all diagonals
    self.p_diag_base.fill(self.proximal_rho);
```

**Line 982-984** (Setter):
```rust
pub fn set_proximal_rho(&mut self, proximal_rho: f64) {
    self.proximal_rho = proximal_rho;
}
```

#### 2. q-ρx_ref Modification (Main Solver)

**File**: `solver-core/src/ipm2/solve.rs`

**Lines 302-312** (Gradient adjustment):
```rust
// Update neg_q_working to include proximal gradient if enabled
if settings.use_proximal {
    // For proximal objective: min (1/2)x'Px + q'x + (ρ/2)||x - x_ref||²
    // Gradient: ∇f = Px + q + ρ(x - x_ref) = (P+ρI)x + (q - ρ*x_ref)
    // We already added ρI to P in KKT, so modify q: neg_q_eff = -(q - ρ*x_ref) = -q + ρ*x_ref
    for i in 0..n {
        neg_q_working[i] = neg_q_base[i] + settings.proximal_rho * state.x_ref[i];
    }
} else {
    // No proximal: use base neg_q
    neg_q_working.copy_from_slice(&neg_q_base);
}
```

**Line 773** (x_ref update):
```rust
state.x_ref.copy_from_slice(&state.x);
```

#### 3. State Management (HSDE State)

**File**: `solver-core/src/ipm/hsde.rs`

**Lines 50-53** (x_ref field):
```rust
/// Reference point for proximal regularization (n-dimensional).
/// Used in IP-PMM to add (ρ/2)||x - x_ref||² to objective.
/// Updated periodically during iterations to track the solution.
pub x_ref: Vec<f64>,
```

**Line 66** (Initialization):
```rust
x_ref: vec![0.0; n],  // Initialize reference point at origin
```

#### 4. Settings Management

**File**: `solver-core/src/problem.rs`

**Lines 211-219** (Settings fields):
```rust
/// Enable proximal regularization (Interior Point-Proximal Method of Multipliers).
/// Adds (ρ/2)||x - x_ref||² to objective, improving conditioning and robustness.
/// Based on PIQP's IP-PMM algorithm (Schwan et al. 2023, arXiv:2304.00290).
pub use_proximal: bool,

/// Proximal penalty parameter ρ for regularization term (ρ/2)||x - x_ref||².
/// Larger values improve conditioning but may slow convergence.
/// Typical range: 1e-8 to 1e-2. Default: 1e-6.
pub proximal_rho: f64,

/// Update x_ref (reference point) every N iterations.
/// 0 = never update (x_ref stays at initial value)
pub proximal_update_interval: usize,
```

**Lines 252-254** (Default values):
```rust
use_proximal: false,  // Opt-in for now (default off until validated)
proximal_rho: 1e-6,
proximal_update_interval: 10,
```

#### 5. Benchmark Integration

**File**: `solver-bench/src/main.rs`

**Lines 291-303** (Environment variable support):
```rust
let use_proximal = std::env::var("MINIX_USE_PROXIMAL")
    .map(|v| v != "0")
    .unwrap_or(false);

let proximal_rho = std::env::var("MINIX_PROXIMAL_RHO")
    .ok()
    .and_then(|v| v.parse().ok())
    .unwrap_or(1e-6);

let proximal_update_interval = std::env::var("MINIX_PROXIMAL_UPDATE_INTERVAL")
    .ok()
    .and_then(|v| v.parse().ok())
    .unwrap_or(10);
```

**Lines 322-324** (Settings application):
```rust
use_proximal,
proximal_rho,
proximal_update_interval,
```

---

## Implementation Verification

### Correctness Check

The implementation follows the correct mathematical formulation:

**Original problem**:
```
minimize (1/2)x'Px + q'x
subject to Ax + s = b, s ∈ K
```

**With proximal regularization**:
```
minimize (1/2)x'Px + q'x + (ρ/2)||x - x_ref||²
```

**Expanded objective**:
```
= (1/2)x'Px + q'x + (ρ/2)(x'x - 2x'x_ref + x_ref'x_ref)
= (1/2)x'Px + q'x + (ρ/2)x'Ix - ρx'x_ref + constant
= (1/2)x'(P + ρI)x + (q - ρx_ref)'x + constant
```

**Therefore**:
1. Modify P: P_proximal = P + ρI ✓ (done in kkt.rs)
2. Modify q: q_proximal = q - ρx_ref ✓ (done in solve.rs)

The implementation is **mathematically correct**.

---

## How to Use

### Via Environment Variables (Benchmark)

```bash
MINIX_USE_PROXIMAL=1 \
MINIX_PROXIMAL_RHO=1e-6 \
MINIX_PROXIMAL_UPDATE_INTERVAL=10 \
cargo run --release -p solver-bench -- maros-meszaros
```

### Via SolverSettings (API)

```rust
let settings = SolverSettings {
    use_proximal: true,
    proximal_rho: 1e-6,
    proximal_update_interval: 10,
    ..Default::default()
};
```

---

## Implementation Flow

1. **Initialization** (solve.rs:157-166):
   - Create UnifiedKktSolver with problem data
   - Symbolic factorization

2. **Proximal Setup** (solve.rs:173-179):
   ```rust
   if settings.use_proximal {
       kkt.set_proximal_rho(settings.proximal_rho);  // Sets ρ in KKT solver
   }
   ```

3. **Main Iteration Loop**:

   a. **Update x_ref** (every proximal_update_interval iterations):
   ```rust
   if settings.use_proximal && iter % settings.proximal_update_interval == 0 {
       state.x_ref.copy_from_slice(&state.x);  // x_ref ← current x
   }
   ```

   b. **Apply q-shift** (solve.rs:302-312):
   ```rust
   if settings.use_proximal {
       for i in 0..n {
           neg_q_working[i] = neg_q_base[i] + settings.proximal_rho * state.x_ref[i];
       }
   }
   ```

   c. **KKT Assembly** (predcorr.rs:751):
   ```rust
   kkt.update_numeric(prob.P.as_ref(), &prob.A, &ws.scaling)?;
   ```
   This calls the KKT matrix assembly which includes P+ρI (kkt.rs:1063)

   d. **Factor and Solve**:
   ```rust
   let factor = kkt.factorize()?;
   kkt.solve_refined(&factor, ...)?;
   ```

---

## Testing Strategy

### Test 1: QFFFFF80 (Primary Target)
- **Problem**: n=854, m=1378, KKT quasi-definiteness failure
- **Baseline**: MaxIters @ 50 iterations, r_d = 6.027e9
- **With proximal**: Testing...
- **Expected**: Should solve or significantly reduce residuals

### Test 2: QSCAGR25/7 (Agriculture Problems)
- **Problems**: Degenerate with quadratic terms
- **Expected**: May benefit from better conditioning

### Test 3: QSCFXM1/2/3 (Fixed-Charge Networks)
- **Problems**: Network flow with fixed charges
- **Expected**: May benefit from regularization

### Test 4: Full Benchmark
- **Command**: `MINIX_USE_PROXIMAL=1 MINIX_PROXIMAL_RHO=1e-6 cargo run --release -p solver-bench -- maros-meszaros`
- **Expected gain**: +3-5 problems (conservative)
- **Baseline**: 105/136 (77.2%)
- **Target**: 108-110/136 (79.4-80.9%)

---

## Potential Issues & Solutions

### Issue 1: x_ref Initialization
- **Current**: x_ref starts at origin (zeros)
- **Problem**: May bias solution toward origin on first few iterations
- **Solution**: Could initialize x_ref = initial_x instead
- **Status**: Acceptable for now, test if needed

### Issue 2: Update Interval
- **Current**: x_ref updated every 10 iterations
- **Problem**: May update too frequently or infrequently
- **Solution**: Adaptive update based on ||x - x_ref|| or residual changes
- **Status**: Acceptable for now, optimize if needed

### Issue 3: ρ Value Selection
- **Current**: Fixed ρ=1e-6
- **Problem**: May be too small for some problems, too large for others
- **Solution**: Adaptive ρ based on KKT condition number or factorization success
- **Status**: Test with different fixed values first

### Issue 4: Polish Integration
- **Current**: Polish may interfere with proximal
- **User guidance**: "Turn off polish during proximal, or only run at end"
- **Solution**: Disable polish when proximal is active, or add final polish step
- **Status**: Need to implement

---

## Next Steps

1. ✓ Verify implementation is correct (DONE)
2. ⚙ Run full benchmark with proximal (IN PROGRESS)
3. Compare results vs baseline
4. Analyze which problems benefit
5. Tune ρ value if needed
6. Implement polish integration
7. Consider adaptive ρ and x_ref updates

---

## References

1. PIQP Paper (Schwan et al. 2023): "A Proximal Interior-Point Method for Conic Optimization"
2. IP-PMM Algorithm: Interior-Point Proximal Method of Multipliers
3. User guidance on correct implementation (P+ρI AND q-ρx_ref)
4. Our failure analysis: QFFFFF80 as primary target
