# Minix: Non-Embedding “Direct” Solve Mode (and what we can learn from Clarabel/MOSEK)

## Context

You’re already on the right track: the biggest step forward so far was fixing *model/presolve interactions* (bounds + singleton elimination + postsolve sizing) and then adding targeted *numerical safety valves* (τ normalization, adaptive refinement, and early polish). The BOYD1 trace you shared is exactly the kind of progress that usually unlocks the next tier of robustness.

What’s left (QGROW/KSIP speed gap; a handful of small inaccurate/failed cases; DUAL* being slow) strongly suggests:

- **Implementation / linear-algebra efficiency** is the main remaining differentiator vs Clarabel.
- Some **edge-case convergence / termination** issues remain (YAO-like “μ tiny, dual OK, primal stalls”; QSC205-like “dual stalls”).
- A “non-embedding” direct solve is *not* the reason Clarabel is fast — but **a direct solve mode can still be a useful tool** (especially as a *refinement/polish stage* or as a *fast path for likely-feasible problems*).

This doc focuses on:  
1) “non-HSDE takeaways” you can adopt **without rewriting everything**, and  
2) a **clear design** for a true non-embedding *direct* solve mode, implemented alongside the current homogeneous embedding solver.

---

## 1) What Clarabel actually does (why “non-HSDE” is a bit of a trap)

Clarabel’s core contribution is *not* “no embedding.” It’s that for **conic programs with quadratic objectives**, it uses a **homogeneous embedding that is *not self-dual*** (because keeping the quadratic term breaks self-duality), and then solves that embedding directly with a primal-dual IPM.

This matters because it implies:

- Switching Minix to a non-embedding direct method will **not automatically align you with Clarabel’s algorithmic choices**.
- The speed delta is overwhelmingly likely to come from **KKT assembly/factorization caching, ordering, refinement, and low-allocation numerics**, not from the presence/absence of (τ, κ).

---

## 2) “Non-HSDE takeaways” you can apply without rewriting everything

### Takeaway A: Treat “direct mode” as **a refinement stage first**
Most of your problematic traces look like:

- complementarity is tiny (`μ` ~ 1e-12 or smaller),
- one of the residuals stops improving (often primal),
- step sizes get clipped by a tiny set of near-active entries, and
- you burn iterations “stirring” around a nearly-feasible point.

A *small*, high-ROI tactic is:

**After the embedding method gets close, run a few Newton/refinement steps on the *dehomogenized* primal-dual KKT system (τ fixed, κ dropped).**

You already have the machinery:
- scaling,
- KKT factorization,
- iterative refinement,
- and an active-set polish.

This “direct refinement” is much less invasive than a full new solver mode and directly addresses YAO/QSC-style stalls.

Practical trigger:
- `μ < μ_refine` **and** `min(rel_p, rel_d)` stalls for `k` iters
- OR `gap_rel` good and one residual is good and the other is not (classic endgame stall)

Recommended default:
- 1–5 refinement iterations,
- no centering (σ≈0) or very mild centering,
- accept only if it improves the stalled residual without breaking the other.

---

### Takeaway B: Termination should be computed on **unscaled, dehomogenized** variables and data
Even if the internal algorithm works in a scaled space, termination should be measured in the user’s original units (or at least “problem-scaled” norms that track those units). Otherwise you get:
- tiny `μ`,
- tiny scaled residuals,
- but nontrivial primal/dual infeasibility in original units.

(From your logs, you already print both “inf” and “scale”. That’s good. The remaining work is to ensure the *solver’s internal stop decisions* are based on those same quantities.)

---

### Takeaway C: Reuse KKT structure and avoid “per-iter rebuild”
The QGROW22 magnitude gap (tens of ms vs seconds) almost cannot be algorithmic. For ~3k KKT dimension, factorization should not be multi-second unless:

- the matrix is rebuilt/allocated repeatedly,
- pattern is re-sorted each iter,
- symbolic analysis is repeated,
- or the numeric update path is not actually “values-only.”

Clarabel (and MOSEK) aggressively separate:
- **symbolic factorization** (pattern + ordering + etree) and
- **numeric factorization** (values only).

If Minix is still rebuilding sparse structures each iteration, this is the next “MOSEK hat” priority.

---

### Takeaway D: Add “side selection” for Schur complement / elimination order
For LP/QP-like problems, you can often choose to eliminate x or z first:

- If `n << m`: forming / factoring an `n×n` Schur complement can be cheaper.
- If `m << n`: keep current approach (factor `m×m` or quasi-definite KKT).

Clarabel does this implicitly via its KKT strategy selection; MOSEK absolutely does.

Minix can add a heuristic:
- if `n < n_threshold` and `P` is diagonal-ish / easy: consider normal equations path,
- else do quasi-definite LDL.

This is a performance feature, not a correctness change.

---

### Takeaway E: Use direct-mode ideas to improve **polish**
Your polish is already paying dividends on BOYD1. Two more MOSEK-ish upgrades:

1) **Robust active-set selection**  
   Combine:
   - smallest `s_i` (likely active), and
   - largest `z_i` (strongly binding),
   and include a “safety band” to avoid missing actives.

2) **Accept polish if it improves max-residual and keeps the others within tolerance**, not only if it strictly improves all metrics.  
   Some problems need polish to fix the stalled residual, and it may slightly perturb the other residual while still within tolerance.

---

## 3) Should Minix implement a true non-embedding Direct Solve mode?

### Short answer
**Yes, but do it in phases**:
1) **Direct refinement stage** (low risk, immediate value).  
2) **Direct solve mode** behind a flag (feasible-problem fast path).  
3) **AUTO mode**: start direct, fall back to embedding when needed.

### Why it can help (even if Clarabel embeds)
- The embedded system adds coupling via τ/κ, and while it’s usually cheap, it can interact poorly with
  - very ill-conditioned problems,
  - very small-scale objectives,
  - or strict termination requirements.
- A non-embedding direct mode is a good *engineering tool*:
  - to debug,
  - to refine,
  - and as a performance option when you’re confident the problem is feasible.

### Why it’s risky
- Without embedding, **infeasibility/unboundedness detection becomes heuristic** unless you add a second mechanism for certificates.
- You can “spin” indefinitely on infeasible instances.

This is why MOSEK/Clarabel-like products keep a homogeneous model available.

---

## 4) Design: How a non-embedding Direct Solve mode would live alongside HSDE/H-embedding

### 4.1 Solver modes

Introduce:

```rust
pub enum SolveMode {
    /// Current default: homogeneous embedding (HSDE / non-self-dual H for QP)
    Homogeneous,
    /// New: primal-dual solve on original KKT conditions (no τ, κ)
    Direct,
    /// Try Direct, fall back to Homogeneous on suspicion of infeasibility/stagnation
    Auto,
}
```

Recommended default:
- `Auto` (for the solver-bench), but
- keep library default as `Homogeneous` until you trust `Auto`.

---

### 4.2 Shared components (no duplication)

Both modes should share:
- presolve (bounds, singleton elimination),
- scaling (Ruiz + cone scaling / NT scaling),
- KKT symbolic analysis caching,
- numeric factorization and iterative refinement,
- step-size computation, and
- polishing.

The key is to make the “outer loop” generic over a small interface:

```rust
trait NewtonModel {
    /// Computes residual vectors and scalar summaries (μ, gap, etc.)
    fn residuals(&self, state: &State, prob: &ScaledProblem) -> Residuals;

    /// Builds RHS terms for affine/corrector steps
    fn build_rhs(&self, state: &State, res: &Residuals, step_type: StepType) -> KktRhs;

    /// Updates state after step (including any mode-specific normalization)
    fn apply_step(&self, state: &mut State, step: &KktStep, alpha: f64);
}
```

Then:
- `HomogeneousModel: NewtonModel` (existing),
- `DirectModel: NewtonModel` (new).

Everything else (KKT assembly, factor, refinement, line search, termination, polish triggers) can stay in one place.

---

### 4.3 State structs

**Homogeneous state (existing)**  
- x, s, z, τ, κ (plus any embedding-specific residual scalars)

**Direct state (new)**  
- x, s, z only

Conversion helpers:

- From homogeneous to direct (when τ>0):
  - `x_d = x / τ`, `s_d = s / τ`, `z_d = z / τ`
- From direct to homogeneous (for fallback):
  - `τ = 1`, `κ = κ0` (small), `x = x_d`, `s = s_d`, `z = z_d`

---

### 4.4 Direct-mode Newton system (what we solve)

For conic QP in “slack form”:

- primal feasibility: `A x + s = b`, `s ∈ K`
- dual feasibility: `P x + q + Aᵀ z = 0`, `z ∈ K*`
- complementarity: cone-scaled (NT) `s ∘ z = μ e`

The KKT linearization looks like:

- `P Δx + Aᵀ Δz = -r_d`
- `A Δx + Δs = -r_p`
- `Z Δs + S Δz = r_c`  (LP case; general cones use NT scaling blocks)

With NT scaling, you typically reduce to the quasi-definite system:

`[ P + δI     Aᵀ ] [Δx] = [rhs_x]`
`[ A       -(H+δI)] [Δz]   [rhs_z]`

where `H` is your current scaling operator (block diagonal).

**Key point:** this is the *same* KKT form you already have, just with different RHS construction and without τ/κ coupling.

So Direct mode should be able to reuse your `KktSolver` and scaling blocks nearly verbatim.

---

### 4.5 Infeasibility/unboundedness handling in Direct mode

Direct mode alone cannot reliably certify infeasibility. So:

- **Direct mode is “feasible-first”**:
  - It targets solutions when they exist.
  - It detects trouble and yields `SuspectInfeasible` (internal) rather than claiming infeasible.

In `Auto` mode:
- Run Direct up to:
  - `k_fail` iterations OR
  - until a “suspicion trigger” fires
- Then restart with Homogeneous mode.

Suspicion triggers (examples):
- `μ` not decreasing for `k` iters AND residuals not improving
- τ-equivalent metrics in direct mode suggest no feasible point:
  - e.g., infeasibility residual lower bounds flat while step sizes collapse
- scaling becomes extreme (a hint of infeasibility/ill-conditioning)
- KKT becomes singular repeatedly even with reg (common in infeasible instances)

When fallback happens:
- warm-start Homogeneous from the current direct iterate.

---

## 5) Roadmap (phased, “don’t rewrite everything”)

### Phase 0: Direct refinement stage (recommended first)
- Add a small function that:
  1) dehomogenizes (x,s,z),
  2) runs 1–5 direct Newton/refinement iterations,
  3) re-checks termination on unscaled metrics,
  4) accepts if it improves the stalled residual.

This is low risk and likely fixes YAO-like endgame stalls.

### Phase 1: Direct solve mode (flagged)
- Implement `DirectModel: NewtonModel`.
- Add CLI/config option `--mode direct`.
- Add tests on “known feasible” problems.

### Phase 2: Auto mode
- Direct by default, fallback to Homogeneous.
- Track “fallback rate” in benchmarks (should be low on the usual feasible suites).

---

## 6) Practical guidance: when to expect Direct mode to help

Direct mode is most likely to help when:
- problems are feasible,
- the endgame stall is from embedding scalars interacting with scaling / termination,
- or you want a faster solution path without certificates.

Direct mode is least helpful when:
- the real bottleneck is KKT assembly/factorization overhead,
- or the problem family contains infeasible/unbounded cases where certificates matter.

Given your current benchmark story, **performance wins vs Clarabel will primarily come from linear algebra**, but **robustness wins** (YAO/QSC/HS* stragglers) may come from direct-refinement/polish.

---

## Appendix: What I would check “with a MOSEK hat” before implementing Direct mode

1) **Are we already solving Clarabel’s “H embedding” correctly for QP?**  
   If not, Direct mode might mask issues rather than fix them.

2) **Are we repeating symbolic analysis / permutation / pattern sorting per iteration?**  
   If yes, fix this first — it’s likely the QGROW22/KSIP gap.

3) **Do we have an elimination-order heuristic?**  
   Add `n vs m` heuristics and possibly a normal-equations path.

4) **Do we have aggressive allocation avoidance in the solve loop?**  
   Replace Vec reallocations with preallocated workspaces and `copy_from_slice`.

5) **Do we have stable refinement rules for both primal and dual?**  
   Your current adaptive refinement triggers on dual stall; add symmetric logic for primal stall.

