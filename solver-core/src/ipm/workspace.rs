//! Pre-allocated workspace for predictor-corrector IPM.
//!
//! This module provides reusable buffers to eliminate per-iteration allocations
//! in the hot path of the interior point method.

use crate::cones::ConeKernel;

/// Cone type for fast dispatch (avoids runtime Any downcasting).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConeType {
    /// Zero cone (equality constraint)
    Zero,
    /// Non-negative orthant
    NonNeg,
    /// Second-order cone (Lorentz cone)
    Soc,
}

/// Workspace for predictor-corrector algorithm.
///
/// All buffers are pre-allocated based on problem dimensions and reused
/// across iterations to avoid allocation overhead.
pub struct PredCorrWorkspace {
    // Problem dimensions
    n: usize, // number of variables
    m: usize, // number of constraints

    // ========================================================================
    // Per-iteration vectors (allocated once, reused each iteration)
    // ========================================================================
    /// Affine direction for x
    pub dx_aff: Vec<f64>,
    /// Affine direction for z
    pub dz_aff: Vec<f64>,
    /// Affine direction for s
    pub ds_aff: Vec<f64>,

    /// Combined direction for x
    pub dx: Vec<f64>,
    /// Combined direction for z
    pub dz: Vec<f64>,
    /// Combined direction for s
    pub ds: Vec<f64>,

    /// Mehrotra correction d_s
    pub d_s_comb: Vec<f64>,

    /// Second solve for Schur complement: dx2
    pub dx2: Vec<f64>,
    /// Second solve for Schur complement: dz2
    pub dz2: Vec<f64>,

    /// P*xi product
    pub mul_p_xi: Vec<f64>,
    /// 2*P*xi + q product
    pub mul_p_xi_q: Vec<f64>,

    /// RHS for affine solve (x part)
    pub rhs_x_aff: Vec<f64>,
    /// RHS for affine solve (z part)
    pub rhs_z_aff: Vec<f64>,

    /// RHS for combined solve (x part)
    pub rhs_x_comb: Vec<f64>,
    /// RHS for combined solve (z part)
    pub rhs_z_comb: Vec<f64>,

    /// MCC delta accumulator
    pub mcc_delta: Vec<f64>,

    // ========================================================================
    // SOC-specific buffers (sized to max SOC dimension)
    // ========================================================================
    max_soc_dim: usize,

    /// W^{1/2} scaling vector
    pub soc_w_half: Vec<f64>,
    /// W^{-1/2} scaling vector
    pub soc_w_half_inv: Vec<f64>,
    /// Lambda = W*z
    pub soc_lambda: Vec<f64>,
    /// W^{-1} ds
    pub soc_w_inv_ds: Vec<f64>,
    /// W dz
    pub soc_w_dz: Vec<f64>,
    /// Eta = (W^{-1} ds) ∘ (W dz)
    pub soc_eta: Vec<f64>,
    /// Lambda squared
    pub soc_lambda_sq: Vec<f64>,
    /// v vector for Mehrotra correction
    pub soc_v: Vec<f64>,
    /// u vector (solution to λ ∘ u = v)
    pub soc_u: Vec<f64>,
    /// d_s block output
    pub soc_d_s_block: Vec<f64>,
    /// H*dz temporary
    pub soc_h_dz: Vec<f64>,

    // ========================================================================
    // Centrality check buffers (for line search)
    // ========================================================================
    /// Trial s for centrality check
    pub cent_s_trial: Vec<f64>,
    /// Trial z for centrality check
    pub cent_z_trial: Vec<f64>,
    /// Jordan product w = s ∘ z for SOC centrality
    pub cent_w: Vec<f64>,

    // ========================================================================
    // Jordan algebra temporaries (for spectral decomposition)
    // ========================================================================
    /// Spectral e1 vector
    pub jordan_e1: Vec<f64>,
    /// Spectral e2 vector
    pub jordan_e2: Vec<f64>,
    /// Jordan product temporary 1
    pub jordan_temp1: Vec<f64>,
    /// Jordan product temporary 2
    pub jordan_temp2: Vec<f64>,
    /// Jordan product temporary 3
    pub jordan_temp3: Vec<f64>,

    // ========================================================================
    // Problem structure cache
    // ========================================================================
    /// Whether problem has any SOC cones
    pub has_soc: bool,
    /// Whether problem has any NonNeg cones
    pub has_nonneg: bool,
    /// Index ranges for each SOC cone: (start, end, dim)
    pub soc_ranges: Vec<(usize, usize, usize)>,
    /// Index ranges for each NonNeg cone: (start, end, dim)
    pub nonneg_ranges: Vec<(usize, usize)>,
    /// Cone types in order (avoids runtime type checks)
    pub cone_types: Vec<ConeType>,
    /// Cone dimensions in order
    pub cone_dims: Vec<usize>,
    /// Cone offsets in order
    pub cone_offsets: Vec<usize>,
}

impl PredCorrWorkspace {
    /// Create a new workspace for the given problem dimensions.
    pub fn new(n: usize, m: usize, cones: &[Box<dyn ConeKernel>]) -> Self {
        use std::any::Any;

        // Find max SOC dimension and cache cone structure
        let mut max_soc_dim = 0usize;
        let mut has_soc = false;
        let mut has_nonneg = false;
        let mut soc_ranges = Vec::new();
        let mut nonneg_ranges = Vec::new();
        let mut cone_types = Vec::with_capacity(cones.len());
        let mut cone_dims = Vec::with_capacity(cones.len());
        let mut cone_offsets = Vec::with_capacity(cones.len());

        let mut offset = 0;
        for cone in cones {
            let dim = cone.dim();
            cone_offsets.push(offset);
            cone_dims.push(dim);

            if dim == 0 {
                cone_types.push(ConeType::Zero);
                continue;
            }

            let is_soc = (cone.as_ref() as &dyn Any).is::<crate::cones::SocCone>();
            let is_nonneg = (cone.as_ref() as &dyn Any).is::<crate::cones::NonNegCone>();

            if is_soc {
                has_soc = true;
                max_soc_dim = max_soc_dim.max(dim);
                soc_ranges.push((offset, offset + dim, dim));
                cone_types.push(ConeType::Soc);
            } else if is_nonneg {
                has_nonneg = true;
                nonneg_ranges.push((offset, offset + dim));
                cone_types.push(ConeType::NonNeg);
            } else {
                // Zero cone or unknown - treat as Zero
                cone_types.push(ConeType::Zero);
            }

            offset += dim;
        }

        Self {
            n,
            m,

            // Per-iteration vectors
            dx_aff: vec![0.0; n],
            dz_aff: vec![0.0; m],
            ds_aff: vec![0.0; m],
            dx: vec![0.0; n],
            dz: vec![0.0; m],
            ds: vec![0.0; m],
            d_s_comb: vec![0.0; m],
            dx2: vec![0.0; n],
            dz2: vec![0.0; m],
            mul_p_xi: vec![0.0; n],
            mul_p_xi_q: vec![0.0; n],
            rhs_x_aff: vec![0.0; n],
            rhs_z_aff: vec![0.0; m],
            rhs_x_comb: vec![0.0; n],
            rhs_z_comb: vec![0.0; m],
            mcc_delta: vec![0.0; m],

            // SOC-specific buffers
            max_soc_dim,
            soc_w_half: vec![0.0; max_soc_dim],
            soc_w_half_inv: vec![0.0; max_soc_dim],
            soc_lambda: vec![0.0; max_soc_dim],
            soc_w_inv_ds: vec![0.0; max_soc_dim],
            soc_w_dz: vec![0.0; max_soc_dim],
            soc_eta: vec![0.0; max_soc_dim],
            soc_lambda_sq: vec![0.0; max_soc_dim],
            soc_v: vec![0.0; max_soc_dim],
            soc_u: vec![0.0; max_soc_dim],
            soc_d_s_block: vec![0.0; max_soc_dim],
            soc_h_dz: vec![0.0; max_soc_dim],

            // Centrality check buffers
            cent_s_trial: vec![0.0; max_soc_dim],
            cent_z_trial: vec![0.0; max_soc_dim],
            cent_w: vec![0.0; max_soc_dim],

            // Jordan algebra temporaries
            jordan_e1: vec![0.0; max_soc_dim],
            jordan_e2: vec![0.0; max_soc_dim],
            jordan_temp1: vec![0.0; max_soc_dim],
            jordan_temp2: vec![0.0; max_soc_dim],
            jordan_temp3: vec![0.0; max_soc_dim],

            // Problem structure
            has_soc,
            has_nonneg,
            soc_ranges,
            nonneg_ranges,
            cone_types,
            cone_dims,
            cone_offsets,
        }
    }

    /// Reset all iteration buffers to zero.
    #[inline]
    pub fn reset_iteration(&mut self) {
        // Only reset what's necessary - most buffers are overwritten
        self.d_s_comb.fill(0.0);
        self.mcc_delta.fill(0.0);
    }

    /// Get n (number of variables).
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get m (number of constraints).
    #[inline]
    pub fn m(&self) -> usize {
        self.m
    }

    /// Get max SOC dimension.
    #[inline]
    pub fn max_soc_dim(&self) -> usize {
        self.max_soc_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cones::{NonNegCone, SocCone};

    #[test]
    fn test_workspace_creation() {
        let cones: Vec<Box<dyn ConeKernel>> = vec![
            Box::new(NonNegCone::new(10)),
            Box::new(SocCone::new(5)),
            Box::new(SocCone::new(8)),
        ];

        let ws = PredCorrWorkspace::new(20, 23, &cones);

        assert!(ws.has_nonneg);
        assert!(ws.has_soc);
        assert_eq!(ws.max_soc_dim, 8);
        assert_eq!(ws.soc_ranges.len(), 2);
        assert_eq!(ws.nonneg_ranges.len(), 1);
        assert_eq!(ws.dx.len(), 20);
        assert_eq!(ws.dz.len(), 23);
    }

    #[test]
    fn test_workspace_qp_only() {
        let cones: Vec<Box<dyn ConeKernel>> = vec![Box::new(NonNegCone::new(10))];

        let ws = PredCorrWorkspace::new(10, 10, &cones);

        assert!(ws.has_nonneg);
        assert!(!ws.has_soc);
        assert_eq!(ws.max_soc_dim, 0);
    }
}
