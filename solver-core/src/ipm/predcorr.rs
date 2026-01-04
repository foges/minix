//! Predictor-corrector steps for HSDE interior point method.
//!
//! The predictor-corrector algorithm has two phases per iteration:
//! 1. **Affine step**: Solve KKT system with σ = 0 (pure Newton step)
//! 2. **Combined step**: Solve with Mehrotra correction (adds centering)
//!
//! This implementation follows §7 of the design doc.
//!
//! ## Performance Optimizations
//!
//! This module uses a pre-allocated workspace to eliminate per-iteration
//! memory allocations. Key optimizations:
//! - All direction vectors (dx, dz, ds) are reused across iterations
//! - SOC-specific buffers are sized to max cone dimension
//! - Line search uses cached trial vectors instead of allocating
//! - Jordan algebra operations use workspace temporaries

use super::hsde::{compute_mu, HsdeResiduals, HsdeState};
use super::workspace::{ConeType, PredCorrWorkspace};
use crate::cones::{ConeKernel, NonNegCone, SocCone};
use crate::linalg::kkt::KktSolver;
use crate::problem::{ProblemData, SolverSettings};
use crate::scaling::{nt, ScalingBlock};
use std::any::Any;

/// Predictor-corrector step result.
#[derive(Debug)]
pub struct StepResult {
    /// Step size taken
    pub alpha: f64,

    /// Centering parameter used
    pub sigma: f64,

    /// New barrier parameter after step
    pub mu_new: f64,

    /// Number of line-search backtracks
    pub line_search_backtracks: u64,
}

/// Inline dot product - avoids iterator overhead in hot paths.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Compute both SOC eigenvalues in one pass (avoids duplicate norm computation).
#[inline]
fn soc_eigs(v: &[f64]) -> (f64, f64) {
    let t = v[0];
    let mut norm_sq = 0.0;
    for i in 1..v.len() {
        norm_sq += v[i] * v[i];
    }
    let norm = norm_sq.sqrt();
    (t - norm, t + norm) // (min_eig, max_eig)
}

#[inline]
fn soc_min_eig(v: &[f64]) -> f64 {
    soc_eigs(v).0
}

#[inline]
fn soc_jordan_product(a: &[f64], b: &[f64], out: &mut [f64]) {
    out[0] = a[0] * b[0];
    for i in 1..a.len() {
        out[0] += a[i] * b[i];
    }
    for i in 1..a.len() {
        out[i] = a[0] * b[i] + b[0] * a[i];
    }
}

fn compute_dtau(
    numerator: f64,
    denominator: f64,
    tau: f64,
    denom_scale: f64,
) -> Result<f64, String> {
    if !numerator.is_finite() || !denominator.is_finite() || !tau.is_finite() {
        return Err("dtau inputs not finite".to_string());
    }
    if tau <= 0.0 {
        return Err(format!("tau non-positive (tau={:.3e})", tau));
    }

    let scale = denom_scale.max(1.0);
    if denominator.abs() <= 1e-10 * scale {
        return Err(format!(
            "dtau denominator ill-conditioned (denom={:.3e}, scale={:.3e})",
            denominator, scale
        ));
    }

    let raw_dtau = numerator / denominator;
    let max_dtau = 2.0 * tau;
    Ok(raw_dtau.max(-max_dtau).min(max_dtau))
}

fn apply_tau_direction(dx: &mut [f64], dz: &mut [f64], dtau: f64, dx2: &[f64], dz2: &[f64]) {
    if dtau == 0.0 {
        return;
    }

    for i in 0..dx.len() {
        dx[i] += dtau * dx2[i];
    }
    for i in 0..dz.len() {
        dz[i] += dtau * dz2[i];
    }
}

fn clamp_complementarity_nonneg(
    state: &HsdeState,
    ds: &[f64],
    dz: &[f64],
    cones: &[Box<dyn ConeKernel>],
    beta: f64,
    gamma: f64,
    mu: f64,
) -> Option<Vec<f64>> {
    if mu <= 0.0 {
        return None;
    }

    let mut has_nonneg = false;
    let mut changed = false;
    let mut delta_w = vec![0.0; state.s.len()];
    let mut offset = 0;

    for cone in cones {
        let dim = cone.dim();
        if dim == 0 {
            continue;
        }

        let is_nonneg = (cone.as_ref() as &dyn Any).is::<NonNegCone>();
        if !is_nonneg {
            offset += dim;
            continue;
        }

        has_nonneg = true;
        for i in 0..dim {
            let idx = offset + i;
            let w = (state.s[idx] + ds[idx]) * (state.z[idx] + dz[idx]);
            let w_clamped = w.max(beta * mu).min(gamma * mu);
            let delta = w_clamped - w;
            if delta.abs() > 0.0 {
                changed = true;
            }
            delta_w[idx] = delta;
        }

        offset += dim;
    }

    if !has_nonneg || !changed {
        return None;
    }

    Some(delta_w)
}

/// Check centrality condition for trial step.
///
/// Uses pre-allocated workspace buffers to avoid allocations in line search.
/// The workspace buffers (cent_s_trial, cent_z_trial, cent_w) are sized to
/// max_soc_dim and reused across line search iterations.
#[inline]
fn centrality_ok_trial(
    state: &HsdeState,
    ds: &[f64],
    dz: &[f64],
    dtau: f64,
    dkappa: f64,
    cones: &[Box<dyn ConeKernel>],
    beta: f64,
    gamma: f64,
    soc_beta: f64,
    soc_gamma: f64,
    barrier_degree: usize,
    alpha: f64,
    enable_soc: bool,
    soc_use_upper: bool,
    soc_use_jordan: bool,
    soc_mu_threshold: f64,
    // Workspace buffers (pre-allocated, reused across line search iterations)
    s_trial_buf: &mut [f64],
    z_trial_buf: &mut [f64],
    w_buf: &mut [f64],
) -> bool {
    if barrier_degree == 0 {
        return true;
    }

    let tau_trial = state.tau + alpha * dtau;
    let kappa_trial = state.kappa + alpha * dkappa;
    if tau_trial <= 0.0 || kappa_trial <= 0.0 {
        return false;
    }

    // Compute s·z in a single pass (avoid iterator chain allocation)
    let mut s_dot_z = 0.0;
    for i in 0..state.s.len() {
        let s_i = state.s[i] + alpha * ds[i];
        let z_i = state.z[i] + alpha * dz[i];
        s_dot_z += s_i * z_i;
    }

    let mu_trial = (s_dot_z + tau_trial * kappa_trial) / (barrier_degree as f64 + 1.0);
    if mu_trial <= 0.0 {
        return false;
    }

    let mut offset = 0;
    for cone in cones {
        let dim = cone.dim();
        if dim == 0 {
            continue;
        }

        let is_nonneg = (cone.as_ref() as &dyn Any).is::<NonNegCone>();
        let is_soc = (cone.as_ref() as &dyn Any).is::<SocCone>();

        if is_nonneg {
            // NonNeg centrality: check each component
            for i in 0..dim {
                let idx = offset + i;
                let s_i = state.s[idx] + alpha * ds[idx];
                let z_i = state.z[idx] + alpha * dz[idx];
                let w = s_i * z_i;
                if w < beta * mu_trial || w > gamma * mu_trial {
                    return false;
                }
            }
        } else if is_soc && enable_soc {
            if mu_trial < soc_mu_threshold {
                offset += dim;
                continue;
            }

            let s_block = &state.s[offset..offset + dim];
            let z_block = &state.z[offset..offset + dim];
            let ds_block = &ds[offset..offset + dim];
            let dz_block = &dz[offset..offset + dim];

            // Use workspace buffers instead of allocating
            let s_trial = &mut s_trial_buf[..dim];
            let z_trial = &mut z_trial_buf[..dim];
            for i in 0..dim {
                s_trial[i] = s_block[i] + alpha * ds_block[i];
                z_trial[i] = z_block[i] + alpha * dz_block[i];
            }

            if soc_use_jordan {
                let w = &mut w_buf[..dim];
                soc_jordan_product(s_trial, z_trial, w);
                let (w_min, w_max) = soc_eigs(w);

                if w_min < soc_beta * mu_trial {
                    return false;
                }
                if soc_use_upper && w_max > soc_gamma * mu_trial {
                    return false;
                }
            } else {
                let (s_min, s_max) = soc_eigs(s_trial);
                let (z_min, z_max) = soc_eigs(z_trial);
                let lower = (soc_beta * mu_trial).sqrt();

                if s_min < lower || z_min < lower {
                    return false;
                }
                if soc_use_upper && s_max * z_max > soc_gamma * mu_trial {
                    return false;
                }
            }
        }

        offset += dim;
    }

    true
}

/// Take a predictor-corrector step.
///
/// Implements the Mehrotra predictor-corrector algorithm with:
/// - Affine step to predict progress
/// - Adaptive centering parameter σ
/// - Combined corrector step
/// - Fraction-to-boundary step size selection
///
/// Uses pre-allocated workspace to eliminate per-iteration memory allocations.
///
/// # Returns
///
/// The step result with alpha, sigma, and new mu.
pub fn predictor_corrector_step(
    kkt: &mut KktSolver,
    prob: &ProblemData,
    neg_q: &[f64],
    state: &mut HsdeState,
    residuals: &HsdeResiduals,
    cones: &[Box<dyn ConeKernel>],
    mu: f64,
    barrier_degree: usize,
    settings: &SolverSettings,
    ws: &mut PredCorrWorkspace,
) -> Result<StepResult, String> {
    let n = prob.num_vars();
    let m = prob.num_constraints();

    assert_eq!(neg_q.len(), n, "neg_q must have length n");

    // ======================================================================
    // Step 1: Compute NT scaling for all cones with adaptive regularization
    // ======================================================================
    let mut scaling: Vec<ScalingBlock> = Vec::new();
    let mut offset = 0;

    // Track minimum interior measures for adaptive regularization
    let mut s_min = f64::INFINITY;
    let mut z_min = f64::INFINITY;

    for cone in cones {
        let dim = cone.dim();
        if dim == 0 {
            scaling.push(ScalingBlock::Zero { dim: 0 });
            continue;
        }

        // Skip NT scaling for Zero cone (equality constraints have no barrier)
        if cone.barrier_degree() == 0 {
            scaling.push(ScalingBlock::Zero { dim });
            offset += dim;
            continue;
        }

        let s = &state.s[offset..offset + dim];
        let z = &state.z[offset..offset + dim];

        // Track minimum interior measures for adaptive regularization.
        let is_nonneg = (cone.as_ref() as &dyn Any).is::<NonNegCone>();
        let is_soc = (cone.as_ref() as &dyn Any).is::<SocCone>();
        if is_nonneg {
            for &si in s.iter() {
                if si < s_min {
                    s_min = si;
                }
            }
            for &zi in z.iter() {
                if zi < z_min {
                    z_min = zi;
                }
            }
        } else if is_soc {
            let s_eig_min = soc_min_eig(s);
            let z_eig_min = soc_min_eig(z);
            if s_eig_min < s_min {
                s_min = s_eig_min;
            }
            if z_eig_min < z_min {
                z_min = z_eig_min;
            }
        } else {
            for &si in s.iter() {
                if si < s_min {
                    s_min = si;
                }
            }
            for &zi in z.iter() {
                if zi < z_min {
                    z_min = zi;
                }
            }
        }

        // Compute NT scaling based on cone type
        let scale = nt::compute_nt_scaling(s, z, cone.as_ref()).unwrap_or_else(|_| {
            // Fallback to simple diagonal scaling if NT fails
            let eps = 1e-18;
            let d: Vec<f64> = s
                .iter()
                .zip(z.iter())
                .map(|(si, zi)| {
                    let num = si.max(eps);
                    let den = zi.max(eps);
                    (num / den).max(eps)
                })
                .collect();
            ScalingBlock::Diagonal { d }
        });

        scaling.push(scale);
        offset += dim;
    }

    // Adaptive regularization: gently increase when near boundaries
    // When min(s, z) < μ/100, the scaling can become ill-conditioned.
    let conditioning_threshold = mu / 100.0;
    let min_sz = s_min.min(z_min);
    let needs_extra_reg = min_sz.is_finite() && min_sz < conditioning_threshold;
    let base_reg = settings.static_reg.max(settings.dynamic_reg_min_pivot);
    let extra_reg = if needs_extra_reg {
        let denom = min_sz.max(1e-300);
        let ratio = conditioning_threshold / denom;
        let scale = ratio.sqrt().min(100.0);
        (base_reg * scale).min(1e-4)
    } else {
        0.0
    };

    if settings.verbose && extra_reg > 0.0 {
        eprintln!(
            "extra_reg={:.2e} (s_min={:.2e}, z_min={:.2e}, mu={:.2e})",
            extra_reg, s_min, z_min, mu
        );
    }

    // Apply extra regularization by modifying the scaling
    if extra_reg > 0.0 {
        for block in scaling.iter_mut() {
            match block {
                ScalingBlock::Diagonal { d } => {
                    for di in d.iter_mut() {
                        // Add regularization to H directly (H_reg = H + extra_reg * I)
                        *di += extra_reg;
                    }
                }
                ScalingBlock::SocStructured { diag_reg, .. } => {
                    if settings.enable_soc_adaptive_reg {
                        *diag_reg += extra_reg * settings.soc_adaptive_reg_scale;
                    }
                }
                _ => {}
            }
        }
    }

    // ======================================================================
    // Step 2: Factor KKT system
    // ======================================================================
    let factor = kkt
        .factor(prob.P.as_ref(), &prob.A, &scaling)
        .map_err(|e| format!("KKT factorization failed: {}", e))?;

    // ======================================================================
    // Step 3: Affine step (σ = 0)
    // ======================================================================
    // Newton step to drive residuals toward 0.
    //
    // The linearized equations give:
    //   P Δx + A^T Δz + q Δτ = -r_x  (Newton step to reduce r_x to 0)
    //   A Δx - H Δz = -r_z + s       (combining primal feasibility with complementarity)
    //
    // The complementarity equation H Δz + Δs = -d_s gives:
    //   Δs = -d_s - H Δz = -s - H*dz  (for affine step where d_s = s)

    // Use workspace buffers instead of allocating
    let dx_aff = &mut ws.dx_aff[..];
    let dz_aff = &mut ws.dz_aff[..];
    dx_aff.fill(0.0);
    dz_aff.fill(0.0);
    let dtau_aff;

    // Affine RHS: build directly into workspace buffers (avoid iterator allocations)
    //   rhs_x = -r_x (Newton step to reduce dual residual)
    //   rhs_z = s - r_z (combining -r_z from primal + s from complementarity)
    let rhs_x_aff = &mut ws.rhs_x_aff[..];
    let rhs_z_aff = &mut ws.rhs_z_aff[..];
    for i in 0..n {
        rhs_x_aff[i] = -residuals.r_x[i];
    }
    for i in 0..m {
        rhs_z_aff[i] = state.s[i] - residuals.r_z[i];
    }

    kkt.solve_refined(
        &factor,
        rhs_x_aff,
        rhs_z_aff,
        dx_aff,
        dz_aff,
        settings.kkt_refine_iters,
    );

    // Compute dtau via two-solve Schur complement strategy (design doc §5.4.1)
    // This replaces the old heuristic dtau = -(q'dx + b'dz)

    // First, compute mul_p_xi = P*ξ (if P exists)
    // Use workspace buffer instead of allocating
    let mul_p_xi = &mut ws.mul_p_xi[..];
    mul_p_xi.fill(0.0);
    if let Some(ref p) = prob.P {
        // P is symmetric upper triangle, do symmetric matvec
        // Optimization: process non-diagonal entries with single branch
        for col in 0..n {
            if let Some(col_view) = p.outer_view(col) {
                let xi_col = state.xi[col];
                for (row, &val) in col_view.iter() {
                    let contribution = val * xi_col;
                    mul_p_xi[row] += contribution;
                    if row != col {
                        mul_p_xi[col] += val * state.xi[row];
                    }
                }
            }
        }
    }

    // Compute mul_p_xi_q = 2*P*ξ + q (use workspace buffer)
    let mul_p_xi_q = &mut ws.mul_p_xi_q[..];
    for i in 0..n {
        mul_p_xi_q[i] = 2.0 * mul_p_xi[i] + prob.q[i];
    }

    // Second solve for Schur complement: K [Δx₂, Δz₂] = [-q, b]
    // (design doc §5.4.1)
    let dx2 = &mut ws.dx2[..];
    let dz2 = &mut ws.dz2[..];
    dx2.fill(0.0);
    dz2.fill(0.0);
    let rhs_x2 = neg_q;
    let rhs_z2 = &prob.b;

    kkt.solve_refined(
        &factor,
        rhs_x2,
        rhs_z2,
        dx2,
        dz2,
        settings.kkt_refine_iters,
    );

    // Compute dtau via Schur complement formula (design doc §5.4.1)
    // Numerator: d_τ - d_κ/τ + (2Pξ+q)ᵀΔx₁ + bᵀΔz₁
    // Denominator: κ/τ + ξᵀPξ - (2Pξ+q)ᵀΔx₂ - bᵀΔz₂
    //
    // Note: For LPs (P=None), we use higher regularization (≥1e-6) to stabilize
    // the second solve. This is set in ipm/mod.rs.

    // d_tau = r_tau (affine direction for tau)
    let d_tau = residuals.r_tau;

    // d_kappa for affine step (design doc §7.1): d_kappa = κ * τ
    let d_kappa = state.kappa * state.tau;

    let dot_mul_p_xi_q_dx1 = dot(mul_p_xi_q, dx_aff);
    let dot_b_dz1 = dot(&prob.b, dz_aff);
    let numerator = d_tau - d_kappa / state.tau + dot_mul_p_xi_q_dx1 + dot_b_dz1;

    let dot_xi_mul_p_xi = dot(&state.xi, mul_p_xi);
    let dot_mul_p_xi_q_dx2 = dot(mul_p_xi_q, dx2);
    let dot_b_dz2 = dot(&prob.b, dz2);
    let denominator = state.kappa / state.tau + dot_xi_mul_p_xi - dot_mul_p_xi_q_dx2 - dot_b_dz2;

    let denom_scale = (state.kappa / state.tau).abs().max(dot_xi_mul_p_xi.abs());
    dtau_aff = compute_dtau(numerator, denominator, state.tau, denom_scale)
        .map_err(|e| format!("affine dtau failed: {}", e))?;

    apply_tau_direction(dx_aff, dz_aff, dtau_aff, dx2, dz2);

    let dkappa_aff = -(d_kappa + state.kappa * dtau_aff) / state.tau;

    // Debug output disabled by default
    // #[cfg(debug_assertions)]
    // eprintln!("  [dtau_aff] = {:.6e}", dtau_aff);

    // Compute ds_aff from complementarity equation (design doc §5.4):
    //   Δs = -d_s - H Δz
    // For affine step, d_s = s, so:
    //   ds_aff = -s - H*dz_aff
    // Use workspace buffer instead of allocating
    let ds_aff = &mut ws.ds_aff[..];
    ds_aff.fill(0.0);
    let mut offset = 0;
    for (cone_idx, cone) in cones.iter().enumerate() {
        let dim = cone.dim();
        if dim == 0 {
            continue;
        }

        if cone.barrier_degree() == 0 {
            // Zero cone: ds = 0 always (s must remain 0)
            for i in offset..offset + dim {
                ds_aff[i] = 0.0;
            }
        } else {
            // Apply ds = -s - H*dz using the scaling block
            match &scaling[cone_idx] {
                ScalingBlock::Diagonal { d } => {
                    for i in 0..dim {
                        // H_ii = d[i], so ds = -s - H*dz = -s - d[i]*dz
                        ds_aff[offset + i] = -state.s[offset + i] - d[i] * dz_aff[offset + i];
                    }
                }
                ScalingBlock::SocStructured { w, diag_reg } => {
                    // For SOC, H = P(w) (quadratic representation)
                    // ds = -s - (P(w) + diag_reg*I)*dz
                    // Use workspace buffer instead of allocating
                    let dz_slice = &dz_aff[offset..offset + dim];
                    let h_dz = &mut ws.soc_h_dz[..dim];
                    crate::scaling::nt::quad_rep_apply(w, dz_slice, h_dz);
                    for i in 0..dim {
                        ds_aff[offset + i] =
                            -state.s[offset + i] - h_dz[i] - diag_reg * dz_slice[i];
                    }
                }
                _ => {
                    // Fallback: assume diagonal with H = s/z
                    for i in 0..dim {
                        let h_ii = state.s[offset + i] / state.z[offset + i].max(1e-14);
                        ds_aff[offset + i] = -state.s[offset + i] - h_ii * dz_aff[offset + i];
                    }
                }
            }
        }
        offset += dim;
    }

    // Compute affine step size (step-to-boundary)
    let mut alpha_aff = compute_step_size(&state.s, &ds_aff, &state.z, &dz_aff, cones, 1.0);
    if dtau_aff < 0.0 {
        alpha_aff = alpha_aff.min(-state.tau / dtau_aff);
    }
    if dkappa_aff < 0.0 {
        alpha_aff = alpha_aff.min(-state.kappa / dkappa_aff);
    }

    // ======================================================================
    // Step 4: Compute centering parameter σ
    // ======================================================================
    let mu_aff = compute_mu_aff(
        state,
        &ds_aff,
        &dz_aff,
        dtau_aff,
        dkappa_aff,
        alpha_aff,
        barrier_degree,
    );
    let sigma = compute_centering_parameter(alpha_aff, mu, mu_aff, barrier_degree);

    // ======================================================================
    // Step 5: Combined corrector step (+ step size, with stall recovery)
    // ======================================================================
    // From design doc §7.3:
    //   d_x = (1-σ) r_x
    //   d_z = (1-σ) r_z
    //   d_tau = (1-σ) r_tau
    //   d_kappa = κτ + Δκ_aff Δτ_aff - σμ
    //   d_s = Mehrotra correction (§7.3.1 for symmetric cones)
    //
    // KKT RHS:
    //   rhs_x = d_x
    //   rhs_z = d_s - d_z
    //
    // Use workspace buffers instead of allocating
    let dx = &mut ws.dx[..];
    let dz = &mut ws.dz[..];
    let ds = &mut ws.ds[..];
    let d_s_comb = &mut ws.d_s_comb[..];
    dx.fill(0.0);
    dz.fill(0.0);
    ds.fill(0.0);
    let mut dtau = 0.0;
    let mut dkappa = 0.0;

    let mut alpha = 0.0;
    let mut alpha_sz = f64::INFINITY;
    let mut alpha_tau = f64::INFINITY;
    let mut alpha_kappa = f64::INFINITY;
    let mut alpha_pre_ls = 0.0;

    let mut sigma_used = sigma;
    let mut sigma_eff = sigma;
    let mut feas_weight_floor = 0.05;
    let mut refine_iters = settings.kkt_refine_iters;
    let mut final_feas_weight = 0.0;
    let mut line_search_backtracks = 0u64;

    let max_retries = 2usize;
    for attempt in 0..=max_retries {
        sigma_used = sigma_eff;
        let feas_weight = (1.0 - sigma_eff).max(feas_weight_floor);
        final_feas_weight = feas_weight;
        let target_mu = sigma_eff * mu;

        let d_kappa_corr = state.kappa * state.tau + dkappa_aff * dtau_aff - target_mu;

        // Build RHS for combined step (use workspace buffers)
        let rhs_x_comb = &mut ws.rhs_x_comb[..];
        for i in 0..n {
            rhs_x_comb[i] = -feas_weight * residuals.r_x[i];
        }

        let mut mcc_delta: Option<Vec<f64>> = None;
        for corr_iter in 0..=settings.mcc_iters {
            d_s_comb.fill(0.0);
            let mut offset = 0;
            for (cone_idx, cone) in cones.iter().enumerate() {
                let dim = cone.dim();
                if dim == 0 {
                    continue;
                }

                if cone.barrier_degree() == 0 {
                    // Zero cone: d_s = 0
                    offset += dim;
                    continue;
                }

                // Use cached cone types to avoid runtime type checks
                let cone_type = ws.cone_types[cone_idx];

                if cone_type == ConeType::Soc {
                    if let ScalingBlock::SocStructured { w, .. } = &scaling[cone_idx] {
                        let z_slice = &state.z[offset..offset + dim];
                        let ds_aff_slice = &ds_aff[offset..offset + dim];
                        let dz_aff_slice = &dz_aff[offset..offset + dim];

                        // Use workspace buffers instead of allocating (critical for performance)
                        // All SOC buffers are sized to max_soc_dim
                        let w_half = &mut ws.soc_w_half[..dim];
                        let w_half_inv = &mut ws.soc_w_half_inv[..dim];
                        let lambda = &mut ws.soc_lambda[..dim];
                        let w_inv_ds = &mut ws.soc_w_inv_ds[..dim];
                        let w_dz = &mut ws.soc_w_dz[..dim];
                        let eta = &mut ws.soc_eta[..dim];
                        let lambda_sq = &mut ws.soc_lambda_sq[..dim];
                        let v = &mut ws.soc_v[..dim];
                        let u = &mut ws.soc_u[..dim];
                        let d_s_block = &mut ws.soc_d_s_block[..dim];

                        // Build W = P(w^{1/2}) and W^{-1} = P(w^{-1/2})
                        nt::jordan_sqrt_apply(w, w_half);
                        nt::jordan_inv_apply(w_half, w_half_inv);

                        // λ = W z
                        nt::quad_rep_apply(w_half, z_slice, lambda);

                        // η = (W^{-1} ds_aff) ∘ (W dz_aff)
                        nt::quad_rep_apply(w_half_inv, ds_aff_slice, w_inv_ds);
                        nt::quad_rep_apply(w_half, dz_aff_slice, w_dz);
                        nt::jordan_product_apply(w_inv_ds, w_dz, eta);

                        // v = λ∘λ + η - σμ e, with e = (1, 0, ..., 0)
                        nt::jordan_product_apply(lambda, lambda, lambda_sq);

                        v[0] = lambda_sq[0] + eta[0] - target_mu;
                        for i in 1..dim {
                            v[i] = lambda_sq[i] + eta[i];
                        }

                        // u solves λ ∘ u = v
                        nt::jordan_solve_apply(lambda, v, u);

                        // d_s = W^T u (W is self-adjoint for SOC)
                        nt::quad_rep_apply(w_half, u, d_s_block);

                        d_s_comb[offset..offset + dim].copy_from_slice(d_s_block);
                    } else {
                        // Fallback: use diagonal correction if scaling block isn't SOC
                        for i in offset..offset + dim {
                            let z_i = state.z[i].max(1e-14);
                            let w_base = state.s[i] * state.z[i] + ds_aff[i] * dz_aff[i];
                            d_s_comb[i] = (w_base - target_mu) / z_i;
                        }
                    }
                } else {
                    for i in offset..offset + dim {
                        let z_i = state.z[i].max(1e-14);
                        let w_base = state.s[i] * state.z[i] + ds_aff[i] * dz_aff[i];
                        let delta = if cone_type == ConeType::NonNeg {
                            mcc_delta.as_ref().map_or(0.0, |d| d[i])
                        } else {
                            0.0
                        };
                        d_s_comb[i] = (w_base - target_mu - delta) / z_i;
                    }
                }

                offset += dim;
                let _ = cone_idx;
            }

            // rhs_z = d_s - d_z (weighted feasibility residual) - use workspace buffer
            let rhs_z_comb = &mut ws.rhs_z_comb[..];
            for i in 0..m {
                rhs_z_comb[i] = d_s_comb[i] - feas_weight * residuals.r_z[i];
            }

            kkt.solve_refined(
                &factor,
                rhs_x_comb,
                rhs_z_comb,
                dx,
                dz,
                refine_iters,
            );

            // Compute dtau for corrector step using Schur complement formula
            // From design doc §7.3:
            //   d_tau = r_tau
            //   d_kappa = κτ + Δκ_aff Δτ_aff - σμ
            //
            // Schur complement numerator: d_tau - d_kappa/τ + (2Pξ+q)ᵀΔx + bᵀΔz
            let d_tau_corr = feas_weight * residuals.r_tau;

            let dot_mul_p_xi_q_dx = dot(mul_p_xi_q, dx);
            let dot_b_dz = dot(&prob.b, dz);
            let numerator_corr =
                d_tau_corr - d_kappa_corr / state.tau + dot_mul_p_xi_q_dx + dot_b_dz;

            dtau = compute_dtau(numerator_corr, denominator, state.tau, denom_scale)
                .map_err(|e| format!("corrector dtau failed: {}", e))?;

            apply_tau_direction(dx, dz, dtau, dx2, dz2);

            // Compute ds from complementarity equation (design doc §5.4):
            //   Δs = -d_s - H Δz
            let mut offset = 0;
            for (cone_idx, cone) in cones.iter().enumerate() {
                let dim = cone.dim();
                if dim == 0 {
                    continue;
                }

                if cone.barrier_degree() == 0 {
                    // Zero cone: ds = 0 always (s must remain 0)
                    for i in offset..offset + dim {
                        ds[i] = 0.0;
                    }
                } else {
                    // Apply ds = -d_s - H*dz using the scaling block
                    match &scaling[cone_idx] {
                        ScalingBlock::Diagonal { d } => {
                            for i in 0..dim {
                                // ds = -d_s - H*dz
                                ds[offset + i] = -d_s_comb[offset + i] - d[i] * dz[offset + i];
                            }
                        }
                        ScalingBlock::SocStructured { w, diag_reg } => {
                            // For SOC, H = P(w) (quadratic representation)
                            // ds = -d_s - (P(w) + diag_reg*I)*dz
                            // Use workspace buffer instead of allocating
                            let dz_slice = &dz[offset..offset + dim];
                            let h_dz = &mut ws.soc_h_dz[..dim];
                            crate::scaling::nt::quad_rep_apply(w, dz_slice, h_dz);
                            for i in 0..dim {
                                ds[offset + i] =
                                    -d_s_comb[offset + i] - h_dz[i] - diag_reg * dz_slice[i];
                            }
                        }
                        _ => {
                            // Fallback: assume diagonal with H = s/z
                            for i in 0..dim {
                                let h_ii = state.s[offset + i] / state.z[offset + i].max(1e-14);
                                ds[offset + i] = -d_s_comb[offset + i] - h_ii * dz[offset + i];
                            }
                        }
                    }
                }
                offset += dim;
                let _ = cone_idx;
            }

            if corr_iter == settings.mcc_iters {
                break;
            }

            let next_delta = clamp_complementarity_nonneg(
                state,
                &ds,
                &dz,
                cones,
                settings.centrality_beta,
                settings.centrality_gamma,
                mu,
            );
            if next_delta.is_none() {
                break;
            }
            mcc_delta = next_delta;
        }

        // Compute step size with fraction-to-boundary
        let tau_old = state.tau;
        dkappa = -(d_kappa_corr + state.kappa * dtau) / tau_old;

        alpha_sz = compute_step_size(&state.s, &ds, &state.z, &dz, cones, 1.0);
        alpha = alpha_sz;
        alpha_tau = f64::INFINITY;
        alpha_kappa = f64::INFINITY;
        if dtau < 0.0 {
            alpha_tau = -state.tau / dtau;
            alpha = alpha.min(alpha_tau);
        }
        if dkappa < 0.0 {
            alpha_kappa = -state.kappa / dkappa;
            alpha = alpha.min(alpha_kappa);
        }

        // Apply fraction-to-boundary and cap at 1.0 (never take more than a full Newton step)
        alpha = (0.99 * alpha).min(1.0);
        alpha_pre_ls = alpha;

        if settings.line_search_max_iters > 0
            && settings.centrality_gamma > settings.centrality_beta
            && settings.centrality_beta > 0.0
        {
            for _ in 0..settings.line_search_max_iters {
                // Use workspace buffers for centrality check (eliminates allocations in loop)
                if centrality_ok_trial(
                    state,
                    ds,
                    dz,
                    dtau,
                    dkappa,
                    cones,
                    settings.centrality_beta,
                    settings.centrality_gamma,
                    settings.soc_centrality_beta,
                    settings.soc_centrality_gamma,
                    barrier_degree,
                    alpha,
                    settings.enable_soc_centrality,
                    settings.soc_centrality_use_upper,
                    settings.soc_centrality_use_jordan,
                    settings.soc_centrality_mu_threshold,
                    &mut ws.cent_s_trial,
                    &mut ws.cent_z_trial,
                    &mut ws.cent_w,
                ) {
                    break;
                }
                alpha *= 0.5;
                line_search_backtracks += 1;
            }
        }

        let alpha_limiter_sz = alpha_sz <= alpha_tau.min(alpha_kappa);
        let alpha_stall = alpha < 1e-3 && mu < 1e-6 && alpha_limiter_sz;
        if !alpha_stall || attempt == max_retries {
            break;
        }

        if settings.verbose {
            eprintln!(
                "alpha stall detected: alpha={:.3e} (pre_ls={:.3e}), alpha_sz={:.3e}, alpha_tau={:.3e}, alpha_kappa={:.3e}, sigma={:.3e}, attempt={}",
                alpha,
                alpha_pre_ls,
                alpha_sz,
                alpha_tau,
                alpha_kappa,
                sigma_eff,
                attempt + 1,
            );
        }

        if attempt == 0 {
            let base_reg = settings.static_reg.max(settings.dynamic_reg_min_pivot);
            let bump_reg = (base_reg * 10.0).min(1e-4);
            if bump_reg > 0.0 {
                let changed = kkt
                    .bump_static_reg(bump_reg)
                    .map_err(|e| format!("KKT reg bump failed: {}", e))?;
                if changed && settings.verbose {
                    eprintln!(
                        "bumped KKT static_reg to {:.2e} after alpha stall",
                        bump_reg
                    );
                }
            }
            sigma_eff = (sigma_eff + 0.2).min(0.999);
            refine_iters = refine_iters.saturating_add(2);
        } else {
            sigma_eff = 0.999;
            feas_weight_floor = 0.0;
            refine_iters = refine_iters.saturating_add(2);
        }
    }

    if settings.verbose && alpha < 1e-8 {
        eprintln!(
            "alpha stall: alpha={:.3e} (pre_ls={:.3e}), alpha_sz={:.3e}, alpha_tau={:.3e}, alpha_kappa={:.3e}, sigma={:.3e}, feas_weight={:.3e}, tau={:.3e}, kappa={:.3e}, dtau={:.3e}, dkappa={:.3e}",
            alpha,
            alpha_pre_ls,
            alpha_sz,
            alpha_tau,
            alpha_kappa,
            sigma_used,
            final_feas_weight,
            state.tau,
            state.kappa,
            dtau,
            dkappa,
        );
    }

    // ======================================================================
    // Step 7: Update state
    // ======================================================================
    for i in 0..n {
        state.x[i] += alpha * dx[i];
    }

    // Update s and z, but skip Zero cone slacks (they should remain 0)
    let mut offset = 0;
    for cone in cones {
        let dim = cone.dim();
        if dim > 0 {
            if cone.barrier_degree() == 0 {
                // Zero cone: keep s = 0, but update z (dual is free)
                for i in offset..offset + dim {
                    state.s[i] = 0.0; // Keep at 0
                    state.z[i] += alpha * dz[i];
                }
            } else {
                // Normal cones: update both s and z
                for i in offset..offset + dim {
                    state.s[i] += alpha * ds[i];
                    state.z[i] += alpha * dz[i];
                }
            }
        }
        offset += dim;
    }

    state.tau += alpha * dtau;

    // Update κ via Newton step (design doc §5.4):
    //   Δκ = -(d_κ + κΔτ)/τ
    // For combined step, d_κ = κτ + Δκ_aff Δτ_aff - σμ
    // IMPORTANT: Use tau_old (pre-update) as per the Newton step formula
    state.kappa += alpha * dkappa;

    // Safety clamp - use very small floor to allow κ → 0 for infeasibility detection
    // while preventing exact zero which could cause division issues.
    if state.kappa < 1e-30 {
        state.kappa = 1e-30;
    }

    // Update ξ = x/τ for next iteration's Schur complement
    for i in 0..n {
        state.xi[i] = state.x[i] / state.tau;
    }

    // Compute new μ
    let mu_new = compute_mu(state, barrier_degree);

    // DEBUG: Verify z stayed in cone
    if settings.verbose {
        let mut offset = 0;
        for cone in cones {
            let dim = cone.dim();
            if dim == 0 {
                continue;
            }
            let is_soc = (cone.as_ref() as &dyn Any).is::<SocCone>();
            if is_soc && dim >= 2 {
                let z_block = &state.z[offset..offset + dim];
                let dz_block = &dz[offset..offset + dim];
                let t = z_block[0];
                let x_norm = z_block[1..].iter().map(|xi| xi * xi).sum::<f64>().sqrt();
                let eig_min = t - x_norm;
                if eig_min < 0.0 {
                    // Compute what z was before the step
                    let z_pre: Vec<f64> = z_block
                        .iter()
                        .zip(dz_block.iter())
                        .map(|(&zi, &dzi)| zi - alpha * dzi)
                        .collect();
                    let t_pre = z_pre[0];
                    let x_norm_pre = z_pre[1..].iter().map(|xi| xi * xi).sum::<f64>().sqrt();
                    let eig_min_pre = t_pre - x_norm_pre;

                    // Compute step_to_boundary for this cone
                    let alpha_boundary = cone.step_to_boundary_dual(&z_pre, dz_block);

                    eprintln!("BUG: After step, z outside SOC!");
                    eprintln!("  z_pre={:?}, eig_min_pre={:.6e}", z_pre, eig_min_pre);
                    eprintln!("  dz={:?}", dz_block);
                    eprintln!(
                        "  alpha_used={:.6e}, alpha_boundary={:.6e}",
                        alpha, alpha_boundary
                    );
                    eprintln!("  z_post={:?}, eig_min_post={:.6e}", z_block, eig_min);
                }
            }
            offset += dim;
        }
    }

    Ok(StepResult {
        alpha,
        sigma: sigma_used,
        mu_new,
        line_search_backtracks,
    })
}

/// Compute step size using fraction-to-boundary rule.
///
/// Returns the maximum α such that (s + α Δs, z + α Δz) stays in the cone interior.
fn compute_step_size(
    s: &[f64],
    ds: &[f64],
    z: &[f64],
    dz: &[f64],
    cones: &[Box<dyn ConeKernel>],
    fraction: f64,
) -> f64 {
    let mut alpha = f64::INFINITY;
    let mut offset = 0;

    for cone in cones {
        let dim = cone.dim();
        if dim == 0 {
            continue;
        }

        let s_slice = &s[offset..offset + dim];
        let ds_slice = &ds[offset..offset + dim];
        let z_slice = &z[offset..offset + dim];
        let dz_slice = &dz[offset..offset + dim];

        // Primal step-to-boundary
        let alpha_p = cone.step_to_boundary_primal(s_slice, ds_slice);
        if alpha_p > 0.0 && alpha_p < alpha {
            alpha = alpha_p;
        }

        // Dual step-to-boundary
        let alpha_d = cone.step_to_boundary_dual(z_slice, dz_slice);
        if alpha_d > 0.0 && alpha_d < alpha {
            alpha = alpha_d;
        }

        offset += dim;
    }

    // Apply fraction-to-boundary safety factor and cap at 1.0
    // (Newton step should never be > 1)
    if alpha.is_finite() {
        (fraction * alpha).min(1.0)
    } else {
        1.0
    }
}

/// Compute centering parameter σ.
///
/// Uses the robust formula from design doc §7.2:
///   σ = (1 - α_aff)³
///
/// This is simple, stable, and works well in practice.
/// It gives σ ≈ 0 when affine step is large (aggressive progress)
/// and σ ≈ 1 when affine step is small (conservative centering).
fn compute_mu_aff(
    state: &HsdeState,
    ds_aff: &[f64],
    dz_aff: &[f64],
    dtau_aff: f64,
    dkappa_aff: f64,
    alpha_aff: f64,
    barrier_degree: usize,
) -> f64 {
    if barrier_degree == 0 {
        return 0.0;
    }

    let tau_aff = state.tau + alpha_aff * dtau_aff;
    let kappa_aff = state.kappa + alpha_aff * dkappa_aff;
    if !tau_aff.is_finite() || !kappa_aff.is_finite() || tau_aff <= 0.0 || kappa_aff <= 0.0 {
        return f64::NAN;
    }

    let mut s_dot_z = 0.0;
    for i in 0..state.s.len() {
        let s_i = state.s[i] + alpha_aff * ds_aff[i];
        let z_i = state.z[i] + alpha_aff * dz_aff[i];
        s_dot_z += s_i * z_i;
    }

    (s_dot_z + tau_aff * kappa_aff) / (barrier_degree as f64 + 1.0)
}

/// Compute centering parameter σ using μ_aff when reliable.
fn compute_centering_parameter(alpha_aff: f64, mu: f64, mu_aff: f64, barrier_degree: usize) -> f64 {
    // Special case: no barrier (only Zero cones)
    if barrier_degree == 0 {
        return 0.0;
    }

    let sigma_min = 1e-3;
    let sigma_max = 0.999;
    let sigma = if mu_aff.is_finite() && mu_aff > 0.0 && mu.is_finite() && mu > 0.0 {
        let ratio = (mu_aff / mu).max(0.0);
        ratio.powi(3)
    } else {
        (1.0 - alpha_aff).powi(3)
    };

    sigma.max(sigma_min).min(sigma_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_centering_parameter() {
        // If μ_aff << μ, σ should clip to the lower bound.
        let sigma = compute_centering_parameter(
            0.99, // large alpha_aff (good progress)
            1.0,  // current mu
            1e-6, // very small mu_aff
            3,
        );
        assert!(
            sigma >= 1e-3 && sigma <= 1.1e-3,
            "σ should clip near 1e-3 for tiny mu_aff, got {}",
            sigma
        );

        // Test that σ → 1 when affine step makes poor progress
        let sigma = compute_centering_parameter(
            0.01, // small alpha_aff (poor progress)
            1.0,  // current mu
            1.0,  // mu_aff ~ mu
            3,
        );
        assert!(
            sigma > 0.9,
            "σ should be large for small affine step, got {}",
            sigma
        );
    }

    #[test]
    fn test_compute_step_size() {
        let cones: Vec<Box<dyn ConeKernel>> = vec![Box::new(NonNegCone::new(2))];

        // Test that step size is limited by cone boundary
        let s = vec![1.0, 2.0];
        let ds = vec![-0.5, -1.0]; // Would reach boundary at α = 2 for first component
        let z = vec![1.0, 1.0];
        let dz = vec![-0.5, -0.5]; // Would reach boundary at α = 2

        let alpha = compute_step_size(&s, &ds, &z, &dz, &cones, 1.0);

        // Should be at most 2.0 (when s[0] + 2*(-0.5) = 0)
        assert!(alpha <= 2.0, "Step size should be limited by cone boundary");
        assert!(alpha > 0.0, "Step size should be positive");
    }

    #[test]
    fn test_centrality_ok_soc_trial() {
        let cones: Vec<Box<dyn ConeKernel>> = vec![Box::new(SocCone::new(3))];

        let state = HsdeState {
            x: vec![],
            s: vec![1.0, 0.0, 0.0],
            z: vec![1.0, 0.0, 0.0],
            tau: 1.0,
            kappa: 1.0,
            xi: vec![],
        };

        let ds = vec![0.0; 3];
        let dz = vec![0.0; 3];

        // Workspace buffers for centrality check
        let mut s_trial = vec![0.0; 3];
        let mut z_trial = vec![0.0; 3];
        let mut w_buf = vec![0.0; 3];

        let ok = centrality_ok_trial(
            &state, &ds, &dz, 0.0, 0.0, &cones, 0.1, 10.0, 0.1, 10.0, 2, 1.0, true, true, true, 0.0,
            &mut s_trial, &mut z_trial, &mut w_buf,
        );
        assert!(ok, "SOC centrality should pass for loose bounds");

        let not_ok = centrality_ok_trial(
            &state, &ds, &dz, 0.0, 0.0, &cones, 0.9, 1.1, 0.9, 1.1, 2, 1.0, true, true, true, 0.0,
            &mut s_trial, &mut z_trial, &mut w_buf,
        );
        assert!(!not_ok, "SOC centrality should fail for tight bounds");
    }
}
