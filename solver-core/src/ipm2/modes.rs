#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum SolveMode {
    Normal,
    StallRecovery,
    Polish,
}

#[derive(Debug, Clone)]
pub struct StallDetector {
    alpha_small_count: usize,
    dual_stall_count: usize,
    primal_stall_count: usize,
    last_dual_res: f64,
    last_primal_res: f64,
    polish_trigger_count: usize,

    pub alpha_small_thresh: f64,
    pub alpha_small_iters: usize,

    pub dual_stall_iters: usize,
    pub dual_stall_rel_impr: f64,

    pub primal_stall_iters: usize,
    pub primal_stall_rel_impr: f64,
    pub primal_stall_mu_thresh: f64,

    pub polish_mu_thresh: f64,
    pub polish_dual_mult: f64,
    pub polish_trigger_iters: usize,
}

impl Default for StallDetector {
    fn default() -> Self {
        Self {
            alpha_small_count: 0,
            dual_stall_count: 0,
            primal_stall_count: 0,
            last_dual_res: f64::INFINITY,
            last_primal_res: f64::INFINITY,
            polish_trigger_count: 0,

            alpha_small_thresh: 1e-6,
            alpha_small_iters: 5,

            dual_stall_iters: 10,
            dual_stall_rel_impr: 1e-3,

            primal_stall_iters: 5, // Require 5 consecutive non-improving iters
            primal_stall_rel_impr: 2.0, // Require 2x improvement to reset stall
            primal_stall_mu_thresh: 1e-10, // Only trigger when mu is very tiny

            polish_mu_thresh: 1e-10,
            polish_dual_mult: 10.0,
            polish_trigger_iters: 3,
        }
    }
}

impl StallDetector {
    pub fn update(&mut self, alpha: f64, mu: f64, primal_res: f64, dual_res: f64, tol_feas: f64) -> SolveMode {
        // Alpha stall
        if alpha.is_finite() && alpha < self.alpha_small_thresh {
            self.alpha_small_count += 1;
        } else {
            self.alpha_small_count = 0;
        }

        // Dual residual stall: count as stalling if either:
        // 1. The improvement is very small (< threshold), OR
        // 2. The residual is getting WORSE (negative improvement)
        // This catches both "stuck" and "degrading" cases (e.g., QSHIP family)
        if self.last_dual_res.is_finite() && dual_res.is_finite() {
            let rel_impr = (self.last_dual_res - dual_res) / self.last_dual_res.max(1e-18);
            // Stalling if improvement < threshold (includes negative = getting worse)
            if rel_impr < self.dual_stall_rel_impr {
                self.dual_stall_count += 1;
            } else {
                self.dual_stall_count = 0;
            }
        }
        self.last_dual_res = dual_res;

        // Primal residual stall (only track when μ is tiny)
        if mu.is_finite() && mu < self.primal_stall_mu_thresh {
            if self.last_primal_res.is_finite() && primal_res.is_finite() {
                // Improvement factor: prev / current (higher is better)
                let impr_factor = self.last_primal_res / primal_res.max(1e-18);
                if impr_factor < self.primal_stall_rel_impr {
                    // Not improving by at least 2x -> stalling
                    self.primal_stall_count += 1;
                } else {
                    self.primal_stall_count = 0;
                }
            }
        } else {
            // mu not small enough yet, don't count as primal stall
            self.primal_stall_count = 0;
        }
        self.last_primal_res = primal_res;

        let polish_trigger = mu.is_finite()
            && mu < self.polish_mu_thresh
            && dual_res.is_finite()
            && dual_res > self.polish_dual_mult * tol_feas;

        if polish_trigger {
            self.polish_trigger_count += 1;
        } else {
            self.polish_trigger_count = 0;
        }

        if self.polish_trigger_count >= self.polish_trigger_iters {
            return SolveMode::Polish;
        }

        if self.alpha_small_count >= self.alpha_small_iters || self.dual_stall_count >= self.dual_stall_iters {
            return SolveMode::StallRecovery;
        }

        SolveMode::Normal
    }

    /// Returns true if primal feasibility is stalling (not improving for several iterations
    /// when μ is already tiny). This indicates potential need for σ anti-stall cap.
    pub fn primal_stalling(&self) -> bool {
        self.primal_stall_count >= self.primal_stall_iters
    }

    /// Returns true if dual residual is stalling (not improving for several iterations).
    /// This indicates potential need for σ anti-stall cap.
    pub fn dual_stalling(&self) -> bool {
        self.dual_stall_count >= self.dual_stall_iters
    }

    /// Returns the number of consecutive iterations where dual residual is stalling.
    pub fn dual_stall_count(&self) -> usize {
        self.dual_stall_count
    }
}
