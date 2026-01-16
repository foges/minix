use crate::linalg::sparse::SparseSymmetricCsc;

/// Clarabel-style proportional regularization constants.
/// Static regularization = ε1 + ε2 * max_diag
/// where ε1 is absolute minimum and ε2 scales with problem diagonal.
pub const PROP_REG_EPS1: f64 = 1e-8;  // Absolute minimum regularization
pub const PROP_REG_EPS2: f64 = 1e-16; // Proportional factor: eps * max_diag gives O(eps) regularization

#[derive(Debug, Clone)]
pub struct RegularizationPolicy {
    pub static_reg: f64,
    pub static_reg_min: f64,
    pub static_reg_max: f64,
    pub dynamic_min_pivot: f64,

    // End-game / polish knobs
    pub polish_static_reg: f64,
    pub max_refine_iters: usize,

    // Proportional regularization (Clarabel-style)
    pub use_proportional: bool,
    pub max_diag: f64,  // Maximum diagonal element from KKT system
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
            use_proportional: true,  // Enable by default
            max_diag: 1.0,           // Will be set from P diagonal
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
        // Use proportional regularization if enabled
        let base_reg = if self.use_proportional {
            self.proportional_static_reg()
        } else {
            self.static_reg
        };
        let s = if scale.is_finite() { scale.max(1.0) } else { 1.0 };
        (base_reg * s).clamp(self.static_reg_min, self.static_reg_max)
    }

    /// Compute proportional static regularization (Clarabel-style).
    /// Returns ε1 + ε2 * max_diag, clamped to [static_reg_min, static_reg_max].
    #[inline]
    pub fn proportional_static_reg(&self) -> f64 {
        let prop_reg = PROP_REG_EPS1 + PROP_REG_EPS2 * self.max_diag;
        prop_reg.clamp(self.static_reg_min, self.static_reg_max)
    }

    /// Set max_diag from the P matrix diagonal.
    pub fn set_max_diag_from_p(&mut self, p: Option<&SparseSymmetricCsc>) {
        let mut max_diag = 1.0_f64;
        if let Some(p_mat) = p {
            for (&val, (row, col)) in p_mat.iter() {
                if row == col {
                    max_diag = max_diag.max(val.abs());
                }
            }
        }
        self.max_diag = max_diag;
    }

    #[inline]
    pub fn enter_polish(&self, st: &mut RegularizationState) {
        st.static_reg_eff = st.static_reg_eff.min(self.polish_static_reg);
        st.refine_iters = st.refine_iters.max(self.max_refine_iters);
    }
}

