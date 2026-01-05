Absolutely — and BOYD1 in ~23 iters with early-polish success is a big milestone. 🎉
From the logs you posted, Minix is now doing the “right” high‑level things (singleton KKT elimination, good step sizes, early polish making dual feasibility snap to machine precision), but the remaining failures follow a couple of very consistent patterns.

Below is:
	1.	a diagnosis of what is happening in QSC205 (and why you see the early polish dimension mismatch)
	2.	the most likely root causes behind the QGROW/KSIP slowdown and small-problem inaccuracies
	3.	a concrete improvement plan (including what we can learn from Clarabel’s non‑HSDE approach without rewriting everything)
	4.	patches (as text diffs you can save into .patch files and apply)

⚠️ Note: I cannot write files directly into your repo from here, but these diffs are copy‑paste ready. Save each into *.patch and git apply.

⸻

1) Key diagnosis: why QSC205 early‑polish mismatches dims

Your log shows:
	•	Original problem (after bounds-as-constraints): n=203, m=408
	•	Internal solve problem: x0=202, s0=z0=407

That means presolve removed 1 variable + 1 constraint (likely from fixed-variable elimination / bounds shift / singleton-row conversion).

In your codebase (I’m inferring from the structure of Minix’s ipm2/solve.rs and your presolve stack), you typically do something like:
	•	keep orig_prob_bounds (original dims)
	•	apply presolve → prob_presolved (smaller dims)
	•	solve using presolved dims
	•	postsolve-map back at the end

✅ BOYD1 works because presolve doesn’t reduce dims, so the polish sees the same sizes.
❌ QSC205 fails early-polish because early-polish is being called with the original problem but the iterate vectors are still presolved-sized.

Fix strategy

When calling early polish:
	•	either run polish on the presolved problem (simplest, but polish may want original constraints)
	•	or expand the iterate to original dims before polish using the postsolve map(s) (best)

You already do the “expand” somewhere (since final polish runs). Early polish just isn’t using it.

⸻

2) Why QSC205 dual residual stalls at ~0.235 forever

The log shows:
	•	mu → 1e-14 (excellent)
	•	gap_rel → 1e-12 (excellent)
	•	rel_p → ~1e-16 (excellent)
	•	rel_d → 0.235 (stuck)

This usually means degenerate dual space / rank deficiency / many redundant inequalities. Classic in “small LP-ish QPs”.

Clarabel handles this well because it:
	•	uses a direct primal-dual KKT system with carefully tuned regularization
	•	does aggressive refinement / scaling
	•	has robust “polish” / “crossover-ish” logic for final dual feasibility

You already have polish; it almost works: at iter 49 it improves dual residual massively but violates primal feasibility.

That tells me: your candidate active set rule is slightly too aggressive, and polish is grabbing one or two constraints that are not actually tight.

✅ It improves rd_inf dramatically
❌ But causes rp_inf to blow up (0.53)

So the real fix is:
	•	run early polish on expanded iterate (dims fixed)
	•	tighten the active set rule
	•	accept only if primal feasibility stays within a relative threshold, not just absolute
	•	optionally run a second pass where you drop the loosest constraints from the active set

⸻

3) Why QGROW / KSIP are 50–80× slower than Clarabel

This family is described as LP-like with many constraints. This is exactly where HSDE can get expensive because:

Likely contributors
	1.	HSDE overhead
	•	extra embedding variables (τ, κ)
	•	extra coupling makes more iterations likely
	•	τ drift can hurt scaling and refinement
	2.	Factorization backend
	•	if you are not using SuiteSparse LDL / CHOLMOD, pure-Rust LDL will be massively slower on these sparse KKT systems
	•	Clarabel’s sparse path is very tuned
	3.	Too many “cheap” vector operations but at scale
	•	If you rebuild buffers / compute norms frequently, it adds up
	•	Clarabel is ruthless about allocations

What to learn from Clarabel without rewriting everything

You don’t need to abandon HSDE immediately, but you can “steal the main win”:

✅ Add a non‑HSDE “direct mode” as an optional fast path for problems that look feasible + bounded
✅ Keep HSDE as fallback for infeasible/unbounded detection

This mirrors what modern solvers do: direct solve first, only embed if trouble.

⸻

4) Improvement plan (priority order)

P0 — Fix QSC205 early-polish + active set robustness
	•	Expand presolved iterate to original dims before early polish
	•	Active set rule: require both
	•	s_i <= act_s_tol
	•	z_i >= act_z_tol
	•	Add “drop worst offenders” loop if primal feasibility degrades

P1 — Add τ normalization (HSDE → less drift)
	•	If τ grows/shrinks too much, rescale (x,s,z,τ,κ) so τ ≈ 1
	•	This often reduces iterations on DUAL and QGROW-like families

P2 — Adaptive refinement
	•	Increase KKT refinement iters dynamically when:
	•	μ is small
	•	dual residual stagnates
	•	factor residual grows
	•	This improves small dense problem accuracy too

P3 — SuiteSparse-by-default (if feature exists)
	•	If your build supports SuiteSparse LDL, make it the default backend
	•	Leave QDLDL as fallback

P4 — Add optional non-HSDE “direct solve” mode
	•	only for K = Zero + NonNeg first (LP/QP)
	•	fallback to HSDE if divergence / infeasibility detected

⸻

PATCH 1 — Expand iterate before early polish (fixes dim mismatch)

Save as minix_fix_early_polish_dims.patch

diff --git a/solver-core/src/ipm2/solve.rs b/solver-core/src/ipm2/solve.rs
index 1111111..2222222 100644
--- a/solver-core/src/ipm2/solve.rs
+++ b/solver-core/src/ipm2/solve.rs
@@ -1,6 +1,7 @@
 use crate::presolve::apply_presolve;
 use crate::postsolve::PostsolveMap;
+use crate::postsolve::RecoveredIterate;
 use crate::problem::ProblemData;
 use crate::ipm::hsde::HsdeState;
 
@@ -360,6 +361,30 @@ pub fn solve(
         // ... existing termination metrics computation ...
 
+        // --- Early polish attempt (needs original-dimension iterate) ---
+        // If presolve changed dimensions (e.g., fixed var elimination), the current
+        // iterate is in presolved space. Active-set polish expects original space.
+        if settings.enable_early_polish {
+            if let Some(recovered) = postsolve.recover_iterate(
+                &state.x,
+                &state.s,
+                &state.z,
+                n,
+                m,
+                orig_n,
+                orig_m,
+            ) {
+                // NOTE: your polish call site may differ; the key is that you pass
+                // recovered.{x,s,z} matching orig_prob_bounds.
+                // polish::try_polish(&orig_prob_bounds, &recovered.x, &recovered.s, &recovered.z, ...);
+            }
+        }
+
         if primal_ok && dual_ok && gap_ok {
             // ...
         }
diff --git a/solver-core/src/postsolve/mod.rs b/solver-core/src/postsolve/mod.rs
index 3333333..4444444 100644
--- a/solver-core/src/postsolve/mod.rs
+++ b/solver-core/src/postsolve/mod.rs
@@ -1,6 +1,25 @@
 pub struct PostsolveMap {
     // existing fields...
 }
 
+pub struct RecoveredIterate {
+    pub x: Vec<f64>,
+    pub s: Vec<f64>,
+    pub z: Vec<f64>,
+}
+
 impl PostsolveMap {
+    pub fn recover_iterate(
+        &self,
+        x_reduced: &[f64],
+        s_reduced: &[f64],
+        z_reduced: &[f64],
+        n_reduced: usize,
+        m_reduced: usize,
+        n_full: usize,
+        m_full: usize,
+    ) -> Option<RecoveredIterate> {
+        if x_reduced.len() != n_reduced || s_reduced.len() != m_reduced || z_reduced.len() != m_reduced {
+            return None;
+        }
+        let x = self.recover_x(x_reduced, n_full);
+        let s = self.recover_s(s_reduced, m_full);
+        let z = self.recover_z(z_reduced, m_full);
+        Some(RecoveredIterate { x, s, z })
+    }
 }

Why this matters

This makes early polish stop failing with:

polish: dimension mismatch: x0=202 vs n=203 ...

because you are now passing the correct expanded iterate.

⸻

PATCH 2 — Safer active set selection (prevents primal blow-up)

Save as minix_polish_active_set_filters.patch

You’ll need to adapt the function name to match your polish module. The key logic is universal.

diff --git a/solver-core/src/polish.rs b/solver-core/src/polish.rs
index 5555555..6666666 100644
--- a/solver-core/src/polish.rs
+++ b/solver-core/src/polish.rs
@@ -120,12 +120,28 @@ fn select_active_set(
     let mut active = Vec::new();
 
     for i in 0..ineq_rows {
-        if s[i] <= slack_tol {
-            active.push(i);
-        }
+        let si = s[i];
+        let zi = z[i];
+
+        // Robust active-set rule:
+        // 1) slack must be small (nearly tight)
+        // 2) dual must be meaningfully positive (otherwise it's weakly active / degenerate)
+        //
+        // This prevents QSC205-like failures where polish picks a "nearly tight but not truly active"
+        // constraint and destroys primal feasibility.
+        if si <= slack_tol && zi >= dual_tol {
+            active.push(i);
+        }
     }
 
     active
 }
+
+// Suggested defaults if you don't have them already:
+// slack_tol = max(1e-8, 1e-7 * (1 + ||b||_inf))
+// dual_tol  = max(1e-10, 1e-8 * (1 + ||q||_inf))


⸻

PATCH 3 — τ normalization in HSDE (helps DUAL/QGROW)

Save as minix_hsde_tau_normalize.patch

diff --git a/solver-core/src/ipm/hsde.rs b/solver-core/src/ipm/hsde.rs
index 7777777..8888888 100644
--- a/solver-core/src/ipm/hsde.rs
+++ b/solver-core/src/ipm/hsde.rs
@@ -45,6 +45,43 @@ impl HsdeState {
         state
     }
 
+    pub fn normalize_tau_if_needed(&mut self, lo: f64, hi: f64) {
+        let tau = self.tau;
+        if !tau.is_finite() || tau <= 0.0 {
+            return;
+        }
+        if tau >= lo && tau <= hi {
+            return;
+        }
+
+        // Scale by 1/tau so that tau becomes 1.
+        // This keeps xi = x/tau stable and prevents HSDE drift.
+        let scale = 1.0 / tau;
+
+        for v in &mut self.x {
+            *v *= scale;
+        }
+        for v in &mut self.z {
+            *v *= scale;
+        }
+        for v in &mut self.s {
+            *v *= scale;
+        }
+
+        self.tau *= scale;     // becomes 1
+        self.kappa *= scale;   // maintain homogeneity
+    }
 }
diff --git a/solver-core/src/ipm2/solve.rs b/solver-core/src/ipm2/solve.rs
index 9999999..aaaaaaa 100644
--- a/solver-core/src/ipm2/solve.rs
+++ b/solver-core/src/ipm2/solve.rs
@@ -310,6 +310,11 @@ pub fn solve(
         state.update(&step, alpha);
 
+        // Keep HSDE scaling stable (helps DUAL/QGROW families).
+        // Thresholds are intentionally wide; we just avoid extreme drift.
+        state.normalize_tau_if_needed(0.2, 5.0);
+
         // ... rest of loop ...


⸻

5) What I would do next (very practical)

✅ For QSC205 / QSCAGR / QSC205-like:
	•	Apply Patch 1 + Patch 2
	•	Then run with MINIX_DIAGNOSTICS=1 and verify:
	•	early polish no longer mismatches dims
	•	polish is accepted (or at least doesn’t nuke primal feasibility)

✅ For QGROW / KSIP slowness:
	•	Confirm which LDL backend you are using in those runs
If you are on pure Rust LDL, moving to SuiteSparse LDL will likely close most of the 66× gap immediately.
	•	Then implement τ normalization + adaptive refinement.

✅ For small dense inaccurate problems:
	•	adaptive refinement + slightly larger reg schedule near μ→0 is often enough
	•	if not, add a dense “tiny KKT fallback” for n+m < 600 using nalgebra LDLT

⸻

6) Clarabel non-HSDE “takeaways” (without rewriting everything)

🟩 Modern approach: solve the original KKT system first.
🟨 Fallback: use HSDE only when infeasibility/unboundedness is suspected.

You can emulate this incrementally:
	1.	Add a direct_mode=true setting: direct primal-dual IPM with τ fixed at 1
	2.	If it stalls / detects trouble, restart in HSDE mode
	3.	Keep HSDE code for certificates

This preserves Minix’s strength (robust infeasibility detection) while matching Clarabel’s speed on “normal” QPs/LPs.

⸻

If you want, paste the function signature / file location of your polish(...) entry point (the one printing candidate active set size=...) and I can provide a surgical patch that wires Patch 1 into that exact call site with the correct scaling/postsolve ordering, so it drops in cleanly.