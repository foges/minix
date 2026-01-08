Got it. Given what you’re seeing in v17, there are three distinct buckets:
	1.	A real regression/bug (SOC test + singleton elimination / reduced-scaling mismatch)
	2.	A feature bug (P1.1 “extend max_iter” not actually extending)
	3.	The hard part: “primal fine, dual broken” on a stubborn subset (QFFFFF80/QSHIP/STCQP/BOYD), which won’t move from retry logic alone.

Below is what I’d do next, in priority order, with very concrete patch-style guidance and diagnostics.

⸻

0) Reframe the situation

Your v17 results are actually consistent with:
	•	Exp cone fixes were correctness fixes → they work, great.
	•	P1.2 shift-and-retry only helps when you can’t factor or the solve is numerically unstable.
But your failures are largely “dual residual won’t go down / blows up” while the factorization usually works (or gets retried successfully).
	•	P1.1 would only help BOYD (and maybe a couple others) if it actually extends iterations and if the dual is genuinely converging slowly (not stuck at a linear-solve accuracy floor).

So: fix the two bugs first, then you can tell which problems are (A) slow, (B) stuck, (C) wrong.

⸻

1) SOC regression: singleton elimination must be cone-aware

Symptom you reported

singleton_rows=4 non_singleton_rows=0 on a tiny SOC test, then:

panic ... kkt.rs:663: Reduced scaling block mismatch

That is a massive red flag that presolve eliminated rows that belong to a multi-row cone block, or it eliminated rows but did not update cone/scaling block structure consistently.

Why it happens

Row-wise singleton elimination is only “obviously safe” for separable 1D cones (NonNeg, Zero, maybe Box if represented as two independent nonneg rows).

For SOC / Exp / PSD, the rows are not independent coordinates — they’re coordinates of a coupled cone constraint. Eliminating one row inside a SOC block without rewriting the cone is basically guaranteed to break your scaling-block bookkeeping (and can silently break correctness).

Quick, safe fix (recommended)

Only allow singleton elimination on rows that belong to separable cones.
Concretely:
	•	Build a row -> cone_block_id map (or row -> (cone_kind, cone_dim, block_start)).
	•	When scanning for singleton rows, skip any row whose cone block has dim > 1.

Patch-style pseudocode

// presolve/singleton_elim.rs (or wherever you decide singleton rows)
//
// Build once:
let row_cone = build_row_cone_map(&cones, m); 
// row_cone[i] -> { kind, block_start, block_len }

fn row_is_eligible_singleton(row: usize, row_cone: &RowConeInfo) -> bool {
    // Only separable 1D constraints are safe for rowwise elimination.
    matches!(row_cone.kind, ConeKind::NonNeg | ConeKind::Zero)
        && row_cone.block_len == 1
}

// When detecting/eliminating:
for row in 0..m {
    if row_nnz[row] != 1 { continue; }
    if !row_is_eligible_singleton(row, &row_cone[row]) {
        continue; // IMPORTANT: don't touch SOC/Exp/PSD rows
    }
    eliminate_singleton_row(row);
}

Better fix (if you want singleton elim on NonNeg blocks)

If you want to eliminate inside a NonNeg block of length > 1 (totally fine), you also need to update the cone list after elimination (shrink/split the NonNeg block as rows are removed). That’s doable, but the quick fix above is enough to fix the SOC regression now.

Debugging confirmation steps

Run the SOC test with:
	•	Presolve disabled → if it passes, regression is 100% presolve logic.
	•	Then re-enable presolve but with the “skip cone_dim>1” filter → should pass.

Also add a very blunt invariant after presolve:

debug_assert_eq!(reduced_cones.total_dim(), A_reduced.nrows());

(or m_reduced) and fail loudly.

⸻

2) P1.1 bug: your “extend max_iter” likely can’t extend due to a for loop

This is the most common implementation trap in Rust:

for iter in 0..max_iter {
    if should_extend { max_iter = 200; }
}

That cannot extend the loop bounds. The range is fixed at loop entry.

Fix

Switch the main iteration loop to a while loop (or loop {} with explicit break).

Patch-style pseudocode

let mut iter: usize = 0;
let mut max_iter: usize = params.max_iter;

while iter < max_iter {
    // ... do one IPM iteration ...

    if should_extend_budget(&state, &metrics, &orig_dims, iter, max_iter) {
        let new_max = params.max_iter_extended; // e.g. 200
        if new_max > max_iter {
            if diagnostics {
                eprintln!(
                    "P1.1: extending max_iter {} -> {} (progress ok)",
                    max_iter, new_max
                );
            }
            max_iter = new_max;
        }
    }

    iter += 1;
}

Make “large problem” detection use original dimensions

If presolve shrinks BOYD’s dimensions, and you’re checking prob.num_vars() after presolve, you can accidentally not classify it as “large”.

Do this once up front:

let orig_n = orig_prob.num_vars();
let orig_m = orig_prob.num_constraints();
let is_large = orig_n > 50_000 || orig_m > 50_000;

…and store is_large (don’t recompute later from reduced problem).

Why this matters

If you can get BOYD1 + BOYD2 to pass, that’s an immediate +2 problems (assuming they really are “slow monotonic” not “dual stuck”).

But: if BOYD dual residual is limited by linear solve accuracy, more iterations won’t help. Which leads to the next section.

⸻

3) “Primal perfect, dual awful” failures: separate “bug” from “degeneracy + ill-conditioning”

You’re currently lumping these together, but you should split them with one diagnostic.

3.1 Add a dual residual decomposition print (for QFFFFF80 first)

For QP stationarity residual, you want something like:
	•	g = P x + q
	•	h = A^T y (or G^T z, depending on your formulation)
	•	r_d = g + h + ... (signs depend on your conventions)

When rd[170] = -4.498e8, you need to know if it’s:
	•	because g[170] is ~1e8 (objective term), OR
	•	because A^T y is ~1e8 (dual blow-up), OR
	•	because some recovery/scaling term is wrong.

Add this diagnostic (pseudo)

let g = P.mul(&x) + &q;        // length n
let aty = A.transpose_mul(&y); // length n
let rd = &g + &aty - &z_x;     // or whatever your stationarity form is

print_top_k("rd", &rd, 10, |j| {
    format!(
        "j={j} rd={:+.3e} g={:+.3e} Aty={:+.3e} zterm={:+.3e} x={:+.3e}",
        rd[j], g[j], aty[j], z_x[j], x[j],
    )
});

If g is reasonable but A^T y is insane → this is “dual blow-up” (conditioning / degeneracy / wrong updates / presolve-recovery bug).
If g is insane → you may have scaling issues or data magnitude issues.

3.2 Run three toggles for QFFFFF80

For QFFFFF80 only, run:
	1.	presolve OFF
	2.	scaling OFF (or minimum scaling)
	3.	polish OFF

That gives a matrix:

Presolve	Scaling	Polish	What it tells you
OFF	ON	ON	presolve might be corrupting / recovery wrong
ON	OFF	ON	scaling interacting badly with degeneracy
ON	ON	OFF	polish is destabilizing / making KKT indefinite
OFF	OFF	OFF	“pure core IPM” behavior

If dual catastrophe only happens when presolve ON, then your earlier instinct (“presolve corrupts structure”) is likely right. If it happens regardless, it’s deeper.

⸻

4) Likely high-leverage solver changes for QFFFFF80-class problems

I agree with your updated conclusion: retry logic won’t fix correctness.

But I don’t think the right next move is “give up on proximal.”
The important nuance is:
	•	Fixed tiny rho everywhere is bad (you observed it).
	•	Adaptive, targeted regularization is exactly what helps the “primal OK, dual bad / quasi-definite issues” class.

4.1 Make proximal “event-driven”, not global

Trigger proximal only when one of these happens:
	•	rel_d exceeds some threshold and is increasing (dual blow-up)
	•	or KKT “not quasi-definite”
	•	or iterative refinement saturates and residual won’t decrease

Suggested trigger

let dual_catastrophe = rel_d > 1e2 && rel_d > 10.0 * rel_d_best;
let kkt_failed = matches!(kkt_status, KktStatus::NotQuasiDefinite | KktStatus::FactorizationFailed);
let stuck = stall_counter > STALL_LIMIT;
if dual_catastrophe || kkt_failed || stuck {
    enable_or_increase_prox();
}

4.2 Choose rho based on problem scale

You correctly noted 1e-6 is invisible versus diag(P) ~ 10.

A robust heuristic:
	•	p_scale = max(1.0, max_diag(P))
	•	start rho = 1e-4 * p_scale (not 1e-6)
	•	if still failing, multiply by 10 per retry up to rho_max (maybe 1e0 * p_scale)

Pseudocode

rho = max(rho, 1e-4 * p_scale);
rho = min(rho * 10.0, rho_max);
x_ref = x_current.clone();        // keep regularization “centered”
q_eff = q - rho * x_ref;          // consistent with IP-PMM proximal point
P_eff = P + rho * I;

4.3 Critical: disable polish while proximal is active

Polish often expects the original KKT structure / complementarity regime.
If you’re in a regularized prox phase, polish can absolutely destabilize and cause the “not quasi-definite” loop you saw.

Simple rule:
	•	If rho > 0, skip polish, or only run polish after you turn rho back down.

⸻

5) BOYD1/2 specifically: if dual stalls at ~1e-4, this might be a linear-solve accuracy floor

The “primal 1e-14, dual 5e-4” shape is classic “linear solves aren’t accurate enough for the dual”.

Two things to try:

5.1 Dual-stall should boost refinement (not just sigma)

If you already have “adaptive refinement” but it triggers only on primal stall, extend it:

if dual_stall_detected {
    refine_iters = min(refine_iters + 2, refine_iters_max);
    static_reg = max(static_reg, static_reg * 10.0);
}

5.2 Force a stricter linear solve tolerance for “large mode”

If BOYD uses iterative solves (CG) anywhere, tighten:
	•	CG tol for normal equations
	•	refinement residual thresholds

Even with a direct LDL, you can do more iterative refinement passes on large instances.

⸻

6) What I would do next (very concrete)

Step A — Fix regressions/bugs first
	1.	SOC regression: make singleton elimination cone-aware (skip dim>1 cones).
	2.	P1.1 bug: convert iteration loop to while, make “large problem” check use original dims, and ensure the extension actually changes the loop bound.

Step B — Re-evaluate pass rate
	•	Re-run MM suite.
	•	If BOYD1/2 now pass: you should see +2 immediately.

Step C — QFFFFF80 diagnosis and one targeted fix
	1.	Add dual residual decomposition.
	2.	Run QFFFFF80 with presolve/scaling/polish toggles.
	3.	Implement event-driven proximal with meaningful rho scaling + “disable polish while rho>0”.
	4.	Re-test QFFFFF80 and the few most similar failing problems (QSCAGR*, QSCFXM*, STCQP*).

⸻

If you want, paste just the SOC presolve output + the exact elimination stats (how many rows eliminated, and what cone blocks remain) and I’ll tell you which of the two “singleton elimination” fixes you need:
	•	“skip dim>1 cones” only, or
	•	also “update cone list after eliminating NonNeg rows”.

But even without that, the two patches above (cone-aware singleton elimination + while loop for P1.1) are the highest-confidence, immediate-value moves.