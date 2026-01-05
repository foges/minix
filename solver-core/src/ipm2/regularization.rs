#[derive(Debug, Clone)]
pub struct RegularizationPolicy {
    pub static_reg: f64,
    pub static_reg_min: f64,
    pub static_reg_max: f64,
    pub dynamic_min_pivot: f64,

    // End-game / polish knobs
    pub polish_static_reg: f64,
    pub max_refine_iters: usize,
}

impl Default for RegularizationPolicy {
    fn default() -> Self {
        Self {
            static_reg: 1e-8,
            static_reg_min: 1e-12,
            static_reg_max: 1e-4,
            dynamic_min_pivot: 1e-13,
            polish_static_reg: 1e-10,
            max_refine_iters: 8,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct RegularizationState {
    pub static_reg_eff: f64,
    pub dynamic_bumps: u64,
    pub refine_iters: usize,
}

impl RegularizationPolicy {
    pub fn init_state(&self, scale: f64) -> RegularizationState {
        RegularizationState {
            static_reg_eff: self.effective_static_reg(scale),
            dynamic_bumps: 0,
            refine_iters: 1,
        }
    }

    #[inline]
    pub fn effective_static_reg(&self, scale: f64) -> f64 {
        let s = if scale.is_finite() { scale.max(1.0) } else { 1.0 };
        (self.static_reg * s).clamp(self.static_reg_min, self.static_reg_max)
    }

    #[inline]
    pub fn enter_polish(&self, st: &mut RegularizationState) {
        st.static_reg_eff = st.static_reg_eff.min(self.polish_static_reg);
        st.refine_iters = st.refine_iters.max(self.max_refine_iters);
    }
}

