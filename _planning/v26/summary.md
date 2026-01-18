I put the full plan into a Markdown file and also included the key “drop‑in” patch that I’m confident you should apply immediately.

Download the plan:
Minix → Clarabel Performance Plan (Markdown)￼

Download the new patch (important, low-risk):
0001_ipm2_stable_lambda2_inplace.patch￼

Related existing patches you already have in your sandbox (for convenience):
	•	stable_lambda2_soc.patch￼
	•	minix_unregularized_kkt_refinement.patch￼
	•	minix_psd_reg_floor.patch￼
	•	minix_soc_scaled_neighborhood.patch￼ (good direction, but needs sigma/MCC handling so it doesn’t stall)

The 핵심 diagnosis (why PRIMALC8 is 209× slower)

Two things are simultaneously killing you:
	1.	You’re explicitly building dense SOC blocks in the global KKT.
In src/linalg/kkt.rs::update_soc_block_in_place, you materialize the full dense SOC scaling block by looping over columns and computing quad_rep_soc_in_place(w, e_i); that is O(k^2) to form the block and (much worse) it destroys sparsity so the LDL factorization cost blows up. This is exactly the kind of structural mistake that turns a millisecond solve into hundreds of milliseconds.
	2.	SOC centrality is not enforced (so you need many more iterations).
src/ipm2/predcorr.rs::centrality_ok_nonneg_trial currently skips SOC cones entirely, so the solver can “ride” the SOC boundary. That typically produces: (a) tiny steps later, (b) poor predictor directions, and (c) tons of iterations.

Also: Clarabel’s own write-up explicitly calls out that epigraph SOCP reformulations of quadratic objectives can increase KKT fill-in, and that handling quadratic objectives directly yields a more efficient condensed system.
So one major “performance parity” item is making sure you’re comparing the same formulation (QP vs epigraph-SOCP) and/or adding the same structural optimization Clarabel is using.

What the plan tells you to do first
	•	Immediate correctness + robustness: apply stable SOC lambda2 fix everywhere it exists (you already fixed scaling/nt.rs; my new patch fixes the in-place SOC spectral decomposition used in IPM2). This removes a real numerical landmine.
	•	Immediate perf sanity check: ensure you are actually using the SuiteSparse LDL backend in the benchmark build (if you’re on QDLDL, you’re basically guaranteed to be way behind Clarabel on big KKT’s).
	•	Then: implement SOC neighborhood/proximity checks with a sigma/MCC escape hatch (the reason your first SOC neighborhood patch stalled is that “backtracking alpha” can’t fix “already outside the neighborhood”; you need to re-center the direction).
	•	Real fix for PRIMALC8 wallclock: stop materializing dense SOC blocks in the KKT. You need a condensed / SOC-aware linear system strategy (Schur complement / low-rank / augmented formulation) so SOC scaling updates don’t explode fill.

All of that is laid out step-by-step in the Markdown with milestones, acceptance criteria, and what to instrument.

If you want, paste your latest PRIMALC8 solver log (just the per-iter metrics + any KKT size/nnz info you have). I can then tighten the “which linear-system strategy to implement first” choice (SOC-augmented KKT vs Schur complement) to whatever will land fastest in your codebase.