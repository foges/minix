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
    last_dual_res: f64,
    polish_trigger_count: usize,

    pub alpha_small_thresh: f64,
    pub alpha_small_iters: usize,

    pub dual_stall_iters: usize,
    pub dual_stall_rel_impr: f64,

    pub polish_mu_thresh: f64,
    pub polish_dual_mult: f64,
    pub polish_trigger_iters: usize,
}

impl Default for StallDetector {
    fn default() -> Self {
        Self {
            alpha_small_count: 0,
            dual_stall_count: 0,
            last_dual_res: f64::INFINITY,
            polish_trigger_count: 0,

            alpha_small_thresh: 1e-6,
            alpha_small_iters: 5,

            dual_stall_iters: 10,
            dual_stall_rel_impr: 1e-3,

            polish_mu_thresh: 1e-10,
            polish_dual_mult: 10.0,
            polish_trigger_iters: 3,
        }
    }
}

impl StallDetector {
    pub fn update(&mut self, alpha: f64, mu: f64, dual_res: f64, tol_feas: f64) -> SolveMode {
        // Alpha stall
        if alpha.is_finite() && alpha < self.alpha_small_thresh {
            self.alpha_small_count += 1;
        } else {
            self.alpha_small_count = 0;
        }

        // Dual residual stall
        if self.last_dual_res.is_finite() && dual_res.is_finite() {
            let rel_impr = (self.last_dual_res - dual_res) / self.last_dual_res.max(1e-18);
            if rel_impr.abs() < self.dual_stall_rel_impr {
                self.dual_stall_count += 1;
            } else {
                self.dual_stall_count = 0;
            }
        }
        self.last_dual_res = dual_res;

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
}
