Here are (a) concrete Rust snippets for each of the 4 items you listed, and (b) a patch + .md summary that implements the “top violating rows/components” diagnostics (item 4) and cleans up the two warnings you showed.
	•	Download patch￼
	•	Download summary￼

⸻

1) Extend adaptive refinement to primal stalling

The pattern you want is: track progress in rel_p (and rel_d), and when μ is already tiny but rel_p isn’t improving, boost iterative refinement iterations (or tighten the solve tolerance).

A compact controller you can drop into solve.rs (or wherever you run the IPM loop):

#[derive(Debug, Clone)]
struct StallRefineCtl {
    base_refine_iters: usize,
    max_extra_iters: usize,
    // how small mu must be before we consider stalling to be a "linear solve accuracy" issue
    mu_thresh: f64,
    // improvement factor threshold (e.g. 2x improvement required)
    min_improve_factor: f64,
    // how many consecutive "not improving" observations before we boost
    stall_window: usize,

    last_rel_p: Option<f64>,
    last_rel_d: Option<f64>,
    stall_p: usize,
    stall_d: usize,

    extra_iters: usize,
}

impl StallRefineCtl {
    fn new(base_refine_iters: usize) -> Self {
        Self {
            base_refine_iters,
            max_extra_iters: 24,
            mu_thresh: 1e-8,
            min_improve_factor: 2.0,
            stall_window: 3,
            last_rel_p: None,
            last_rel_d: None,
            stall_p: 0,
            stall_d: 0,
            extra_iters: 0,
        }
    }

    fn update(&mut self, mu: f64, rel_p: f64, rel_d: f64) {
        if mu > self.mu_thresh {
            self.last_rel_p = Some(rel_p);
            self.last_rel_d = Some(rel_d);
            self.stall_p = 0;
            self.stall_d = 0;
            // don't reset extra_iters aggressively; let it decay naturally below
            self.decay_on_progress(rel_p, rel_d);
            return;
        }

        // Primal stall update
        if let Some(prev) = self.last_rel_p {
            let improve = prev / rel_p.max(1e-300);
            if improve < self.min_improve_factor {
                self.stall_p += 1;
            } else {
                self.stall_p = 0;
            }
        }
        self.last_rel_p = Some(rel_p);

        // Dual stall update
        if let Some(prev) = self.last_rel_d {
            let improve = prev / rel_d.max(1e-300);
            if improve < self.min_improve_factor {
                self.stall_d += 1;
            } else {
                self.stall_d = 0;
            }
        }
        self.last_rel_d = Some(rel_d);

        // Boost if either residual is stalling
        if self.stall_p >= self.stall_window || self.stall_d >= self.stall_window {
            // step-up fast
            self.extra_iters = (self.extra_iters + 4).min(self.max_extra_iters);
        } else {
            // decay slowly if not stalling
            self.decay_on_progress(rel_p, rel_d);
        }
    }

    fn decay_on_progress(&mut self, rel_p: f64, rel_d: f64) {
        // If you want: only decay when making *very* good progress.
        let good = |r: f64| r < 1e-2; // arbitrary "we're not in the weeds" gate
        if self.extra_iters > 0 && (good(rel_p) || good(rel_d)) {
            self.extra_iters -= 1;
        }
    }

    fn refine_iters(&self) -> usize {
        self.base_refine_iters + self.extra_iters
    }

    fn primal_stalled(&self) -> bool {
        self.stall_p >= self.stall_window
    }
}

Integration pattern in your solve loop:

let mut ctl = StallRefineCtl::new(settings.kkt_refine_iters);

for iter in 0..settings.max_iter {
    // compute metrics first
    // metrics: { mu, rel_p, rel_d, ... }

    ctl.update(metrics.mu, metrics.rel_p, metrics.rel_d);

    let refine_iters = ctl.refine_iters();

    // pass refine_iters into your KKT solve / step routine:
    // predictor_corrector_step_in_place(..., refine_iters, ...)
}

If you already have “dual stall → boost refinement”, literally mirror that logic for rel_p.

⸻

2) Add σ anti-stall (cap σ when feasibility is stuck)

Once μ is tiny, a σ ≈ 1 step is very centered and tends to preserve an r_p plateau in degenerate cases.

Minimal heuristic:

let mut sigma = (mu_aff / mu).powi(3);
// if you already clamp sigma, do it before this
sigma = sigma.clamp(settings.sigma_min, settings.sigma_max);

// Anti-stall: when mu is tiny and primal is stalling, don't over-center
if metrics.mu < 1e-8 && ctl.primal_stalled() && metrics.rel_p > settings.tol_feas {
    sigma = sigma.min(0.5);
}

A slightly more “progress-driven” version (still heuristic, but behaves nicely):

if metrics.mu < 1e-8 && ctl.primal_stalled() {
    // punish sigma if we're not reducing rel_p
    // (assumes ctl.last_rel_p was updated already)
    sigma = sigma.min(0.5);
}

Where to put this:
	•	Right after you compute sigma from mu_aff / mu.
	•	Before you build the corrector RHS and solve for the combined direction.

⸻

3) Gondzio / multiple-correction steps (cheap, same factorization)

You already have the tell-tale oscillation: alternating tiny α and large α. This is exactly where MCC helps.

You can implement MCC as: after you compute your combined direction (dx, ds, dz), do k extra backsolves with modified complementarity RHS, reusing the same factorization.

A common practical variant is the “clamp complementarity products” strategy:

fn compute_delta_w_nonneg(
    s: &[f64],
    z: &[f64],
    ds: &[f64],
    dz: &[f64],
    alpha: f64,
    mu: f64,
    beta: f64,
    gamma: f64,
    delta_w: &mut [f64],
) -> bool {
    let mut any = false;
    let lo = beta * mu;
    let hi = gamma * mu;

    for i in 0..s.len() {
        let w_trial = (s[i] + alpha * ds[i]) * (z[i] + alpha * dz[i]);
        let w_clamped = w_trial.clamp(lo, hi);
        let dw = w_clamped - w_trial;
        delta_w[i] = dw;
        any |= dw != 0.0;
    }
    any
}

Then in your step routine:

// after you have dx, ds, dz and a candidate alpha (or alpha_aff)
for _ in 0..settings.mcc_iters {
    if !compute_delta_w_nonneg(
        &s, &z,
        &ds, &dz,
        alpha,
        mu,
        settings.centrality_beta,
        settings.centrality_gamma,
        &mut delta_w,
    ) {
        break;
    }

    // Build MCC RHS by adding delta_w to the complementarity RHS
    // Conceptually:
    // rhs_sz = sigma*mu*e - ds_aff*dz_aff + delta_w
    //
    // Then solve the SAME KKT matrix with the updated RHS:
    // kkt.solve_refined(rhs_mcc, sol_mcc, refine_iters);

    dx.axpy(1.0, &dx_mcc);
    ds.axpy(1.0, &ds_mcc);
    dz.axpy(1.0, &dz_mcc);
}

Key point: no refactorization; only additional solves with a different RHS. If your KKT interface is “one RHS → one solution”, MCC is straightforward.

If your solve_normal.rs currently computes something like:

let correction = ds_aff[idx] * dz_aff[idx] / z_safe;

that’s a good hook location to start threading in a real MCC term; right now it’s unused (and your warning shows it).

⸻

4) Diagnose whether singleton/bound-derived rows dominate r_p

The patch I’m attaching implements exactly this as an end-of-solve diagnostic:
	•	prints the top-5 |r_p| rows and tags each as:
	•	orig (row index < orig_m)
	•	bound (row index >= orig_m, i.e., introduced by bounds-as-constraints)

That’s not quite “singleton-eliminated vs not”, but in your YAO logs m_full=2005, m_orig≈2000 and singleton-eliminated rows are almost certainly the bound-derived ones—so this usually answers the question in practice.

If you want to go further and explicitly tag “singleton-eliminated”, the next step is: expose the eliminated-row set from your singleton eliminator and check membership when printing.

⸻

Files
	•	minix_feasibility_floor_improvements.patch￼
	•	minix_feasibility_floor_improvements.md￼

If you apply the patch and re-run YAO with MINIX_DIAGNOSTICS=1, the added “top |r_p| rows” output should immediately tell you whether the residual plateau is concentrated in bound-derived rows (very actionable) or in original constraints (more likely linear-solve accuracy / degeneracy → refinement + sigma + MCC are the right levers).