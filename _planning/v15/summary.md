You’ve basically arrived at the “last 1–2%” where test-set weirdness (and a couple design choices) show up hard. The two failures you describe actually look like two different classes of pathology:
	•	QFORPLAN: HSDE scaling-ray runaway (μ blowing up while residuals don’t move)
	•	QFFFFF80: Dual drift + brittle polish linear algebra (primal good, dual huge; polish triggers non–quasi-definite KKT)

Also worth noting: both problems are “known annoying” in the CUTEst/Maros–Mészáros ecosystem. CUTEr/CUTEst explicitly includes the Maros–Mészáros QP test set.  ￼
And in a recent benchmark paper, MOSEK’s residual for QFORPLAN is catastrophically bad while other methods get ~1e‑6, and QFFFFF80 also shows weaker MOSEK residual than the other methods.
That doesn’t prove “your solver is fine,” but it does support the idea that these are not random, and that an interior-point implementation needs extra guardrails here.

⸻

1) QFORPLAN: HSDE scalar explosion (μ → 1e23), residuals stuck at ~1e‑1

What this pattern usually means in HSDE

In a homogeneous self-dual embedding, the iterates live on a positive scaling ray: scaling
(x,s,y,z,\tau,\kappa)\leftarrow \gamma (x,s,y,z,\tau,\kappa),\;\gamma>0
doesn’t change feasibility of the embedded equations, and the recovered solution depends on ratios like x/\tau, y/\tau, etc.

So if you don’t actively normalize the HSDE scalars (τ, κ) and/or you allow a step that doesn’t reduce a merit/potential measure, it’s possible to get:
	•	residuals not improving (your rel_p/rel_d ~ 0.1 flat),
	•	step sizes not obviously “failing” (α ~ 0.99),
	•	while the embedding scalars and complementarity measure blow up.

Given QFORPLAN’s reputation for poor numerical behavior under some IPM settings (e.g., MOSEK’s residual in that benchmark is ~7.4\times 10^{6} on QFORPLAN), this is consistent with “HSDE runaway due to scaling/conditioning,” not necessarily a sign error.

The 3 cheapest, highest-leverage things to do

A) Log the decomposition of μ and the HSDE scalars
Before changing anything, add one diagnostic line (just for this problem family / when μ grows):
	•	τ, κ
	•	μ components:
	•	\mu_{sz} = \frac{s^\top z}{\nu} (restricted to barrier cones)
	•	\mu_{\tau\kappa} = \frac{\tau\kappa}{\nu}
	•	μ growth ratio: μ_{k+1}/μ_k
	•	σ (centering), and μ_aff if you compute it

This instantly tells you whether the blow-up is dominated by τκ or by s·z, and whether σ is “doing something crazy.”

B) Add HSDE renormalization every iteration (or when τ, κ leave a band)
Because HSDE is homogeneous, you can do this without changing the recovered solution quality.

Two common normalizations (pick one):

Option 1: τ-normalize (keep τ ≈ 1)
If \tau drifts outside a band (say τ > 10 or τ < 0.1), do:

\gamma = 1/\tau,\qquad
(x,s,y,z,\tau,\kappa)\leftarrow \gamma (x,s,y,z,\tau,\kappa)

Key detail: κ must be scaled too (κ ← κ/τ). If κ is not scaled consistently, μ and the central-path equations get distorted and you can create exactly the kind of runaway you’re seeing.

Option 2: (τ+κ)-normalize (keep τ+κ ≈ 1)
\gamma = 1/(\tau+\kappa),\qquad
(\cdot)\leftarrow \gamma(\cdot)
This tends to keep both scalars in a sane range.

Either strategy is a “guardrail,” not a theoretical change.

C) Add a merit/potential decrease check to stop the “runaway steps”
Curtis & Nocedal’s IPM paper (for QPs) explicitly frames globalization via a merit function and step-size selection strategies rather than relying purely on fraction-to-the-boundary.  ￼

You don’t need a full-blown line search across everything to get a win here. A minimal variant:
	•	Define a scalar “progress” measure, e.g.
\Phi = \|r_p\|_2 + \|r_d\|_2 + \eta\,\mu
(or squared norms if you prefer).
	•	After computing a candidate step, reject it (shrink α) if:
	•	\Phi increases by more than, say, 10–20%, or
	•	μ grows by > 10× while residuals barely change.

This will prevent the exact exponential growth you’re seeing even if the direction is numerically questionable.

⸻

2) QFFFFF80: primal great (~1e‑8), dual huge (~700), “matrix not quasi-definite” in Polish

What your symptoms suggest

You have two intertwined issues:
	1.	Dual drift (y/z blowing up or not converging) even while primal feasibility improves.
This is a known behavior in degenerate / ill-conditioned QPs where multipliers are non-unique or the stationarity system is badly conditioned.
	2.	Polish is attempting a linear solve that requires quasi-definiteness, but the matrix you assemble in Polish mode violates that structure. That can happen when:

	•	you drop or zero-out the diagonal terms that make the KKT matrix quasi-definite,
	•	you reduce regularization too aggressively in Polish,
	•	or your active-set construction changes the block structure (e.g., making the dual block effectively “0” instead of “−D”, which breaks quasi-definiteness).

The fact that MOSEK’s residual reported for QFFFFF80 is notably worse than other methods in that benchmark suggests there really is numerical sensitivity in that instance.
And it’s also a named CUTEst/QP benchmark used in IPM step-length strategy studies.  ￼

Cheap fixes that often unblock this class

A) Don’t enter Polish when dual is clearly not under control
Right now you enter Polish with rel_d ~ 600–700. That’s a strong smell: even a perfect polish can’t always salvage a wildly drifting dual if the underlying stationarity system is unstable.

Rule of thumb:
	•	Only allow mode -> Polish if both rel_p and rel_d are below some threshold (e.g., 1e‑3 or 1e‑4), OR if rel_d is at least decreasing over the last k iterations.

If rel_d is exploding, just keep iterating (or trigger a “dual stabilization” mode), but don’t switch to a numerically fragile subproblem solve.

B) In Polish, keep quasi-definiteness by construction
If your factorization backend expects quasi-definite matrices, then in Polish you must maintain:
	•	primal block positive definite (P + reg ≻ 0),
	•	dual block negative definite (−D ≺ 0).

Two practical ways:
	•	Regularization floor in Polish: never let the PD regularization go to ~0.
E.g. polish_reg = max(base_reg, reg_floor) where reg_floor might be 1e‑10 to 1e‑8 times a scaling norm.
	•	Dual block floor: if Polish changes D (or eliminates it for “active” constraints), enforce D_i >= d_min so that −D stays strictly negative definite.

This alone can eliminate “not quasi-definite.”

C) When factorization fails: retry with stronger reg or fallback to normal equations
You already added numerical recovery mechanisms elsewhere; Polish should use the same philosophy:
	•	On “not quasi-definite,” do:
	1.	multiply reg by 10 (and/or add a dual-diagonal floor),
	2.	retry factorization,
	3.	if still failing after a few tries, exit Polish and continue standard IPM, or fallback to a more general solver path (normal equations, iterative refinement).

This turns a hard failure into a degrade-gracefully path.

D) A targeted “dual-only recovery” can fix the “primal good / dual bad” endgame
If x (and s) are truly excellent, you can often recompute better multipliers by solving a different (and more numerically stable) system than your current Polish KKT:
	•	Identify active constraints from s (e.g., s_i ≤ ε).
	•	Solve a regularized least squares system for multipliers on the active set that minimizes stationarity residual:
\min_\lambda \|P x + q + A_{\text{act}}^\top \lambda\|_2^2 + \rho\|\lambda\|_2^2
	•	Then reconstruct z on inactive/active sets appropriately (inactive z=0; active z from residual, projected onto ≥0 if you want).

This doesn’t require quasi-definite factorization of the saddle KKT; it uses SPD normal equations, which you can regularize.

⸻

A concrete “do this next” checklist

For QFORPLAN (fast triage → likely fix)
	1.	Print τ, κ, σ, μ_aff, and split μ into (s·z) vs (τκ).
	2.	Add HSDE normalization (τ=1 or τ+κ=1) each iter, or when τ/κ leaves [1e‑2, 1e2].
	3.	Add a minimal merit check: reject steps that increase Φ too much or cause μ explosion.

For QFFFFF80 (fast triage → likely fix)
	1.	Prevent Polish entry unless rel_d is small or decreasing.
	2.	Add Polish reg floors to preserve quasi-definiteness.
	3.	On factorization failure: retry with stronger reg, else exit Polish / fallback.
	4.	(Optional but powerful) implement dual-only recovery via regularized LS on active constraints.

⸻

Why I think these are the right levers
	•	Your logs show no “step feasibility” failure, meaning your fraction-to-boundary logic isn’t what’s stopping progress; it’s that you’re accepting steps that don’t reduce anything meaningful (QFORPLAN), and/or you’re switching into a mode whose linear algebra assumptions stop holding (QFFFFF80).
	•	Both problems have published evidence of being numerically touchy in practice (especially QFORPLAN), with at least one mainstream IPM struggling badly on them.
	•	Step-length/globalization details matter on these QP test sets; this is exactly what Curtis–Nocedal investigate.  ￼

⸻

If you want, I can also draft a small "debug print block" template (what to print and where) that's low-noise but pins down the root cause in 1–2 runs—especially for confirming whether QFORPLAN's μ explosion is coming from τκ or from s·z.

---

# Implementation Status (v15)

## What Was Implemented

### For QFORPLAN (HSDE scalar explosion)

| Suggestion | Status | Result |
|------------|--------|--------|
| A) Log μ decomposition (s·z vs τκ) | ✅ Done | Added `mu_decomposition()` helper in hsde.rs |
| B) HSDE τ-normalization | ✅ Already present | Thresholds (0.2, 5.0), helps but doesn't fix QFORPLAN |
| B) HSDE τ+κ normalization | ⚠️ Tried | Interferes with infeasibility detection (drives τ too small) |
| C) Merit/potential check | ✅ Done | Reject steps with 100x μ growth + large residuals |

**Result:** QFORPLAN still fails. The μ explosion is primarily from s·z, not τκ. The merit check prevents some runaway but doesn't fix the fundamental issue.

### For QFFFFF80 (dual explosion + brittle polish)

| Suggestion | Status | Result |
|------------|--------|--------|
| A) Don't enter Polish when dual bad | ✅ Done | Skip active-set polish when rel_d > 100x tolerance |
| B) Polish reg floors | ❌ Not done | Could help but polish rarely triggers |
| C) Factorization failure retry | ❌ Not done | Polish not triggering due to gap threshold |
| D) Dual-only recovery (LS) | ❌ Not done | More complex, not implemented |

**Result:** QFFFFF80 still fails. The gap (1.3) is too large for polish to trigger. Even if it did, the primal would degrade significantly.

## Commits Made

1. **7129002** - Add v15 robustness improvements: merit check, normalization, polish guards
2. **7a915e0** - Add ablation analysis: anti-stall has no measurable impact

## Current State

- **Maros-Meszaros:** 108/136 (79.4%) optimal **with 50-iteration limit**
- **Regression Suite:** 110 tests passing
- **28 MaxIters failures** - see `detailed_failures.md` for complete breakdown
- **Key pathologies:**
  - **QFORPLAN:** HSDE scalar explosion, μ → 1e26
  - **QFFFFF80:** Dual explosion (rel_p=9e-9, rel_d=768)
  - **QSHIP family (6):** Step size collapse (α → 1e-40)
  - **15 others:** Dual slow convergence

## Remaining Work

To fix the remaining 28 problems, would likely need:

1. **Direct IPM** (no HSDE) - avoids τ/κ scaling issues entirely
2. **Constraint preconditioning** - identify and scale degenerate constraints
3. **Active-set crossover** - switch to simplex when IPM stalls
4. **Higher-precision arithmetic** - for ill-conditioned KKT systems

See `non_converging_analysis.md` for detailed breakdown of all 28 failing problems.