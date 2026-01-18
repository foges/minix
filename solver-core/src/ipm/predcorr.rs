//! Predictor-corrector steps for HSDE interior point method.
//!
//! The predictor-corrector algorithm has two phases per iteration:
//! 1. **Affine step**: Solve KKT system with σ = 0 (pure Newton step)
//! 2. **Combined step**: Solve with Mehrotra correction (adds centering)
//!
//! This implementation follows §7 of the design doc.

use super::hsde::{HsdeState, HsdeResiduals, compute_mu};
use crate::cones::{ConeKernel, NonNegCone, SocCone, PsdCone};
use crate::cones::psd::{mat_to_svec, svec_to_mat};
use crate::linalg::kkt::KktSolver;
use crate::scaling::{ScalingBlock, nt};
use crate::problem::{ProblemData, SolverSettings};
use nalgebra::DMatrix;
use std::any::Any;
use std::time::{Duration, Instant};

fn diagnostics_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        // Check new unified MINIX_VERBOSE first (level >= 2 means verbose)
        if let Ok(v) = std::env::var("MINIX_VERBOSE") {
            return v.parse::<u8>().map(|n| n >= 2).unwrap_or(false);
        }
        // Legacy: check MINIX_DIAGNOSTICS
        std::env::var("MINIX_DIAGNOSTICS")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false)
    })
}

fn trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        // Check new unified MINIX_VERBOSE first (level >= 4 means trace)
        if let Ok(v) = std::env::var("MINIX_VERBOSE") {
            return v.parse::<u8>().map(|n| n >= 4).unwrap_or(false);
        }
        // Legacy: check PSD-specific debug vars
        std::env::var("MINIX_DEBUG_PSD_DS").ok().as_deref() == Some("1")
    })
}

#[derive(Debug, Clone, Copy)]
struct NonNegStepDiag {
    min_s: f64,
    min_z: f64,
    min_ratio: f64,
    alpha_lim: f64,
    alpha_lim_idx: usize,
    alpha_lim_side: &'static str,
}

fn nonneg_step_diagnostics(
    s: &[f64],
    ds: &[f64],
    z: &[f64],
    dz: &[f64],
    cones: &[Box<dyn ConeKernel>],
) -> Option<NonNegStepDiag> {
    let mut found = false;
    let mut min_s = f64::INFINITY;
    let mut min_z = f64::INFINITY;
    let mut min_ratio = f64::INFINITY;
    let mut alpha_lim = f64::INFINITY;
    let mut alpha_lim_idx = usize::MAX;
    let mut alpha_lim_side = "n/a";
    let mut offset = 0usize;

    for cone in cones {
        let dim = cone.dim();
        if dim == 0 {
            continue;
        }
        if cone.barrier_degree() == 0 {
            offset += dim;
            continue;
        }

        if (cone.as_ref() as &dyn Any).is::<NonNegCone>() {
            found = true;
            for i in 0..dim {
                let idx = offset + i;
                let si = s[idx];
                let zi = z[idx];
                let dsi = ds[idx];
                let dzi = dz[idx];

                if si.is_finite() {
                    if min_s.is_nan() {
                        min_s = si;
                    } else {
                        min_s = min_s.min(si);
                    }
                } else {
                    min_s = f64::NAN;
                }

                if zi.is_finite() {
                    if min_z.is_nan() {
                        min_z = zi;
                    } else {
                        min_z = min_z.min(zi);
                    }
                } else {
                    min_z = f64::NAN;
                }

                if si.is_finite() && zi.is_finite() && zi > 0.0 {
                    let ratio = si / zi;
                    if ratio.is_finite() {
                        min_ratio = min_ratio.min(ratio);
                    }
                }

                if dsi.is_finite() && dsi < 0.0 && si.is_finite() {
                    let alpha = -si / dsi;
                    if alpha.is_finite() && alpha >= 0.0 && alpha < alpha_lim {
                        alpha_lim = alpha;
                        alpha_lim_idx = idx;
                        alpha_lim_side = "s";
                    }
                }

                if dzi.is_finite() && dzi < 0.0 && zi.is_finite() {
                    let alpha = -zi / dzi;
                    if alpha.is_finite() && alpha >= 0.0 && alpha < alpha_lim {
                        alpha_lim = alpha;
                        alpha_lim_idx = idx;
                        alpha_lim_side = "z";
                    }
                }
            }
        }

        offset += dim;
    }

    if !found {
        return None;
    }

    if !min_ratio.is_finite() {
        min_ratio = f64::NAN;
    }
    if !alpha_lim.is_finite() {
        alpha_lim = f64::NAN;
    }

    Some(NonNegStepDiag {
        min_s,
        min_z,
        min_ratio,
        alpha_lim,
        alpha_lim_idx,
        alpha_lim_side,
    })
}

fn min_slice(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

fn all_finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn cone_type_name(cone: &dyn ConeKernel) -> &'static str {
    let any = cone as &dyn Any;
    if any.is::<NonNegCone>() {
        "NonNeg"
    } else if any.is::<SocCone>() {
        "SOC"
    } else {
        "Other"
    }
}

fn check_state_interior_for_step(
    state: &HsdeState,
    cones: &[Box<dyn ConeKernel>],
) -> Result<(), String> {
    if !state.tau.is_finite() || state.tau <= 0.0 {
        return Err(format!("tau is not positive finite (tau={})", state.tau));
    }
    if !state.kappa.is_finite() || state.kappa <= 0.0 {
        return Err(format!("kappa is not positive finite (kappa={})", state.kappa));
    }
    if !all_finite(&state.x) {
        return Err("x contains non-finite values".to_string());
    }
    if !all_finite(&state.s) {
        return Err("s contains non-finite values".to_string());
    }
    if !all_finite(&state.z) {
        return Err("z contains non-finite values".to_string());
    }

    let mut offset = 0usize;
    for cone in cones.iter() {
        let dim = cone.dim();
        if dim == 0 {
            continue;
        }
        let s_slice = &state.s[offset..offset + dim];
        let z_slice = &state.z[offset..offset + dim];

        if cone.barrier_degree() == 0 {
            offset += dim;
            continue;
        }

        let any = cone.as_ref() as &dyn Any;
        if let Some(nonneg) = any.downcast_ref::<NonNegCone>() {
            if !nonneg.is_interior_scaling(s_slice) || !nonneg.is_interior_scaling(z_slice) {
                return Err(format!(
                    "NonNeg cone not interior (offset={}, dim={}, s_min={:.3e}, z_min={:.3e})",
                    offset,
                    dim,
                    min_slice(s_slice),
                    min_slice(z_slice)
                ));
            }
        } else if let Some(soc) = any.downcast_ref::<SocCone>() {
            if !soc.is_interior_scaling(s_slice) || !soc.is_interior_scaling(z_slice) {
                return Err(format!(
                    "SOC cone not interior (offset={}, dim={}, s_min={:.3e}, z_min={:.3e})",
                    offset,
                    dim,
                    min_slice(s_slice),
                    min_slice(z_slice)
                ));
            }
        } else {
            if !cone.is_interior_primal(s_slice) || !cone.is_interior_dual(z_slice) {
                return Err(format!(
                    "{} cone not interior (offset={}, dim={})",
                    cone_type_name(cone.as_ref()),
                    offset,
                    dim
                ));
            }
        }

        offset += dim;
    }

    Ok(())
}

/// Predictor-corrector step result.
#[derive(Debug)]
pub struct StepResult {
    /// Step size taken
    pub alpha: f64,

    /// Step size limited by cone boundaries
    pub alpha_sz: f64,

    /// Centering parameter used
    pub sigma: f64,

    /// New barrier parameter after step
    pub mu_new: f64,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct StepTimings {
    pub kkt_factor: Duration,
    pub kkt_solve: Duration,
    pub cone: Duration,
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

fn centrality_ok_nonneg_trial(
    state: &HsdeState,
    ds: &[f64],
    dz: &[f64],
    dtau: f64,
    dkappa: f64,
    cones: &[Box<dyn ConeKernel>],
    beta: f64,
    gamma: f64,
    barrier_degree: usize,
    alpha: f64,
) -> bool {
    if barrier_degree == 0 {
        return true;
    }

    let tau_trial = state.tau + alpha * dtau;
    let kappa_trial = state.kappa + alpha * dkappa;
    if tau_trial <= 0.0 || kappa_trial <= 0.0 {
        return false;
    }

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

    let mut has_nonneg = false;
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
            let s_i = state.s[idx] + alpha * ds[idx];
            let z_i = state.z[idx] + alpha * dz[idx];
            let w = s_i * z_i;
            if w < beta * mu_trial || w > gamma * mu_trial {
                return false;
            }
        }

        offset += dim;
    }

    if !has_nonneg {
        return true;
    }

    true
}

#[derive(Debug, Clone, Copy)]
struct CentralityViolation {
    idx: usize,
    side: &'static str,
    w: f64,
    lower: f64,
    upper: f64,
    s_i: f64,
    z_i: f64,
    mu_trial: f64,
    tau_trial: f64,
    kappa_trial: f64,
}

fn centrality_nonneg_violation(
    state: &HsdeState,
    ds: &[f64],
    dz: &[f64],
    dtau: f64,
    dkappa: f64,
    cones: &[Box<dyn ConeKernel>],
    beta: f64,
    gamma: f64,
    barrier_degree: usize,
    alpha: f64,
) -> Option<CentralityViolation> {
    if barrier_degree == 0 {
        return None;
    }

    let tau_trial = state.tau + alpha * dtau;
    let kappa_trial = state.kappa + alpha * dkappa;
    if tau_trial <= 0.0 || kappa_trial <= 0.0 {
        return Some(CentralityViolation {
            idx: usize::MAX,
            side: "tau_kappa",
            w: f64::NAN,
            lower: f64::NAN,
            upper: f64::NAN,
            s_i: f64::NAN,
            z_i: f64::NAN,
            mu_trial: f64::NAN,
            tau_trial,
            kappa_trial,
        });
    }

    let mut s_dot_z = 0.0;
    for i in 0..state.s.len() {
        let s_i = state.s[i] + alpha * ds[i];
        let z_i = state.z[i] + alpha * dz[i];
        s_dot_z += s_i * z_i;
    }

    let mu_trial = (s_dot_z + tau_trial * kappa_trial) / (barrier_degree as f64 + 1.0);
    if mu_trial <= 0.0 {
        return Some(CentralityViolation {
            idx: usize::MAX,
            side: "mu",
            w: f64::NAN,
            lower: f64::NAN,
            upper: f64::NAN,
            s_i: f64::NAN,
            z_i: f64::NAN,
            mu_trial,
            tau_trial,
            kappa_trial,
        });
    }

    let lower = beta * mu_trial;
    let upper = gamma * mu_trial;

    let mut offset = 0;
    for cone in cones {
        let dim = cone.dim();
        if dim == 0 {
            continue;
        }

        if cone.barrier_degree() == 0 {
            offset += dim;
            continue;
        }

        if (cone.as_ref() as &dyn Any).is::<NonNegCone>() {
            for i in 0..dim {
                let idx = offset + i;
                let s_i = state.s[idx] + alpha * ds[idx];
                let z_i = state.z[idx] + alpha * dz[idx];
                let w = s_i * z_i;
                if w < lower {
                    return Some(CentralityViolation {
                        idx,
                        side: "low",
                        w,
                        lower,
                        upper,
                        s_i,
                        z_i,
                        mu_trial,
                        tau_trial,
                        kappa_trial,
                    });
                }
                if w > upper {
                    return Some(CentralityViolation {
                        idx,
                        side: "high",
                        w,
                        lower,
                        upper,
                        s_i,
                        z_i,
                        mu_trial,
                        tau_trial,
                        kappa_trial,
                    });
                }
            }
        }

        offset += dim;
    }

    None
}

/// Take a predictor-corrector step.
///
/// Implements the Mehrotra predictor-corrector algorithm with:
/// - Affine step to predict progress
/// - Adaptive centering parameter σ
/// - Combined corrector step
/// - Fraction-to-boundary step size selection
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
    timings: &mut StepTimings,
) -> Result<StepResult, String> {
    let n = prob.num_vars();
    let m = prob.num_constraints();
    check_state_interior_for_step(state, cones)?;

    assert_eq!(neg_q.len(), n, "neg_q must have length n");

    // ======================================================================
    // Step 1: Compute NT scaling for all cones with adaptive regularization
    // ======================================================================
    let cone_start = Instant::now();
    let mut scaling: Vec<ScalingBlock> = Vec::new();
    let mut offset = 0;

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

        // Compute NT scaling based on cone type
        let scale = match nt::compute_nt_scaling(s, z, cone.as_ref()) {
            Ok(scale) => scale,
            Err(e) => {
                let s_block_min = min_slice(s);
                let z_block_min = min_slice(z);
                if (cone.as_ref() as &dyn Any).is::<NonNegCone>() {
                    if diagnostics_enabled() {
                        eprintln!(
                            "nt scaling fallback: cone={}, offset={}, dim={}, s_min={:.3e}, z_min={:.3e}: {}",
                            cone_type_name(cone.as_ref()),
                            offset,
                            dim,
                            s_block_min,
                            z_block_min,
                            e
                        );
                    }
                    // ScalingBlock::Diagonal represents H = S Z^{-1} for NonNeg.
                    let d: Vec<f64> = s
                        .iter()
                        .zip(z.iter())
                        .map(|(si, zi)| {
                            let ratio = si / zi;
                            if ratio.is_finite() && ratio > 0.0 {
                                ratio.clamp(1e-12, 1e12)
                            } else {
                                1.0
                            }
                        })
                        .collect();
                    ScalingBlock::Diagonal { d }
                } else {
                    if diagnostics_enabled() {
                        eprintln!(
                            "nt scaling error: cone={}, offset={}, dim={}, s_min={:.3e}, z_min={:.3e}: {}",
                            cone_type_name(cone.as_ref()),
                            offset,
                            dim,
                            s_block_min,
                            z_block_min,
                            e
                        );
                    }
                    return Err(format!(
                        "NT scaling failed for cone={} (offset={}, dim={}, s_min={:.3e}, z_min={:.3e}): {}",
                        cone_type_name(cone.as_ref()),
                        offset,
                        dim,
                        s_block_min,
                        z_block_min,
                        e
                    ));
                }
            }
        };

        scaling.push(scale);
        offset += dim;
    }

    timings.cone += cone_start.elapsed();

    // ======================================================================
    // Step 2: Factor KKT system
    // ======================================================================
    let factor = {
        const MAX_REG_RETRIES: usize = 3;
        const MAX_STATIC_REG: f64 = 1e-2;
        let mut retries = 0usize;
        loop {
            let start = Instant::now();
            let factor = kkt
                .factor(prob.P.as_ref(), &prob.A, &scaling)
                .map_err(|e| format!("KKT factorization failed: {}", e))?;
            timings.kkt_factor += start.elapsed();

            let bumps = kkt.dynamic_bumps();
            if bumps == 0 || retries >= MAX_REG_RETRIES {
                break factor;
            }

            let next_reg = (kkt.static_reg() * 10.0).min(MAX_STATIC_REG);
            if next_reg <= kkt.static_reg() {
                break factor;
            }
            kkt.set_static_reg(next_reg)
                .map_err(|e| format!("KKT reg update failed: {}", e))?;
            retries += 1;
        }
    };

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
    let mut dx_aff = vec![0.0; n];
    let mut dz_aff = vec![0.0; m];
    let dtau_aff;

    // Affine RHS:
    //   rhs_x = -r_x (Newton step to reduce dual residual)
    //   rhs_z = s - r_z (combining -r_z from primal + s from complementarity)
    let rhs_x_aff: Vec<f64> = residuals.r_x.iter().map(|&r| -r).collect();
    let rhs_z_aff: Vec<f64> = state.s.iter().zip(residuals.r_z.iter())
        .map(|(si, ri)| si - ri)
        .collect();

    // Compute dtau via two-solve Schur complement strategy (design doc §5.4.1)
    // This replaces the old heuristic dtau = -(q'dx + b'dz)

    // First, compute mul_p_xi = P*ξ (if P exists)
    // Only process upper triangle (row <= col) to handle both upper-triangular
    // and full symmetric matrix storage
    let mut mul_p_xi = vec![0.0; n];
    if let Some(ref p) = prob.P {
        for col in 0..n {
            if let Some(col_view) = p.outer_view(col) {
                for (row, &val) in col_view.iter() {
                    if row <= col {
                        mul_p_xi[row] += val * state.xi[col];
                        if row != col {
                            mul_p_xi[col] += val * state.xi[row];
                        }
                    }
                    // Skip lower triangle (row > col)
                }
            }
        }
    }

    // Compute mul_p_xi_q = 2*P*ξ + q
    let mul_p_xi_q: Vec<f64> = mul_p_xi.iter()
        .zip(prob.q.iter())
        .map(|(pxi, qi)| 2.0 * pxi + qi)
        .collect();

    // Second solve for Schur complement: K [Δx₂, Δz₂] = [-q, b]
    // (design doc §5.4.1)
    let mut dx2 = vec![0.0; n];
    let mut dz2 = vec![0.0; m];
    let rhs_x2 = neg_q;
    let rhs_z2 = &prob.b;

    {
        let start = Instant::now();
        kkt.solve_two_rhs_refined_tagged(
            &factor,
            &rhs_x_aff,
            &rhs_z_aff,
            rhs_x2,
            rhs_z2,
            &mut dx_aff,
            &mut dz_aff,
            &mut dx2,
            &mut dz2,
            settings.kkt_refine_iters,
            "rhs1",
            "rhs2",
        );
        timings.kkt_solve += start.elapsed();
    }

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

    let dot_mul_p_xi_q_dx1: f64 = mul_p_xi_q.iter().zip(dx_aff.iter()).map(|(a, b)| a * b).sum();
    let dot_b_dz1: f64 = prob.b.iter().zip(dz_aff.iter()).map(|(a, b)| a * b).sum();
    let numerator = d_tau - d_kappa / state.tau + dot_mul_p_xi_q_dx1 + dot_b_dz1;

    let dot_xi_mul_p_xi: f64 = state.xi.iter().zip(mul_p_xi.iter()).map(|(a, b)| a * b).sum();
    let dot_mul_p_xi_q_dx2: f64 = mul_p_xi_q.iter().zip(dx2.iter()).map(|(a, b)| a * b).sum();
    let dot_b_dz2: f64 = prob.b.iter().zip(dz2.iter()).map(|(a, b)| a * b).sum();
    let denominator = state.kappa / state.tau + dot_xi_mul_p_xi - dot_mul_p_xi_q_dx2 - dot_b_dz2;

    let denom_scale = (state.kappa / state.tau).abs().max(dot_xi_mul_p_xi.abs());
    dtau_aff = compute_dtau(numerator, denominator, state.tau, denom_scale)
        .map_err(|e| format!("affine dtau failed: {}", e))?;

    apply_tau_direction(&mut dx_aff, &mut dz_aff, dtau_aff, &dx2, &dz2);

    let dkappa_aff = -(d_kappa + state.kappa * dtau_aff) / state.tau;

    // Debug output disabled by default
    // #[cfg(debug_assertions)]
    // eprintln!("  [dtau_aff] = {:.6e}", dtau_aff);

    // Compute ds_aff from complementarity equation (design doc §5.4):
    //   Δs = -d_s - H Δz
    // For affine step, d_s = s, so:
    //   ds_aff = -s - H*dz_aff
    let mut ds_aff = vec![0.0; m];
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
                ScalingBlock::SocStructured { w } => {
                    // For SOC, H = P(w) (quadratic representation)
                    // ds = -s - P(w)*dz
                    let dz_slice = &dz_aff[offset..offset + dim];
                    let mut h_dz = vec![0.0; dim];
                    crate::scaling::nt::quad_rep_apply(w, dz_slice, &mut h_dz);
                    for i in 0..dim {
                        ds_aff[offset + i] = -state.s[offset + i] - h_dz[i];
                    }
                }
                ScalingBlock::PsdStructured { w_factor, n } => {
                    // For PSD, H = W ⊗ W (Kronecker product in svec form)
                    // ds = -s - H*dz = -s - W*Dz*W (where Dz is matrix form of dz)
                    let w = DMatrix::<f64>::from_row_slice(*n, *n, w_factor);
                    let dz_slice = &dz_aff[offset..offset + dim];
                    let dz_mat = svec_to_mat(dz_slice, *n);
                    let h_dz_mat = &w * &dz_mat * &w;
                    let mut h_dz = vec![0.0; dim];
                    mat_to_svec(&h_dz_mat, &mut h_dz);
                    for i in 0..dim {
                        ds_aff[offset + i] = -state.s[offset + i] - h_dz[i];
                    }

                    // Debug output at trace level (MINIX_VERBOSE=4)
                    let debug = trace_enabled();
                    if debug {
                        eprintln!("PSD ds_aff: dz={:?}", dz_slice);
                        eprintln!("  H*dz={:?}", h_dz);
                        eprintln!("  s={:?}", &state.s[offset..offset + dim]);
                        eprintln!("  ds_aff={:?}", &ds_aff[offset..offset + dim]);
                    }
                }
                _ => {
                    // Fallback for unknown scaling blocks - should not happen
                    // This now handles Zero blocks which are filtered above,
                    // Dense3x3 (exp/pow cones), etc.
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
        cones,
    );
    let sigma_cap = settings.sigma_max.min(0.999);
    let sigma = compute_centering_parameter(alpha_aff, mu, mu_aff, barrier_degree).min(sigma_cap);


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
    let mut dx = vec![0.0; n];
    let mut dz = vec![0.0; m];
    let mut ds = vec![0.0; m];
    let mut d_s_comb = vec![0.0; m];
    let mut dtau = 0.0;
    let mut dkappa = 0.0;

    let mut alpha = 0.0;
    let mut alpha_sz = f64::INFINITY;
    let mut alpha_tau = f64::INFINITY;
    let mut alpha_kappa = f64::INFINITY;
    let mut alpha_pre_ls = 0.0;

    let mut sigma_used = sigma;
    let mut sigma_eff = sigma;
    let mut feas_weight_floor = settings.feas_weight_floor.clamp(0.0, 1.0);
    let mut refine_iters = settings.kkt_refine_iters;
    let mut final_feas_weight = 0.0;

    let max_retries = 2usize;
    for attempt in 0..=max_retries {
        sigma_used = sigma_eff;
        let feas_weight = (1.0 - sigma_eff).max(feas_weight_floor);
        final_feas_weight = feas_weight;
        let target_mu = sigma_eff * mu;

        let d_kappa_corr = state.kappa * state.tau + dkappa_aff * dtau_aff - target_mu;

        // Build RHS for combined step
        let rhs_x_comb: Vec<f64> = residuals.r_x.iter().map(|&r| -feas_weight * r).collect();

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

                let is_soc = (cone.as_ref() as &dyn Any).is::<SocCone>();
                let is_nonneg = (cone.as_ref() as &dyn Any).is::<NonNegCone>();
                let is_psd = (cone.as_ref() as &dyn Any).is::<PsdCone>();

                if is_psd {
                    // PSD cone: use pure centering (no Mehrotra correction)
                    // d_s_comb = s - σμ * svec(I)
                    // This avoids the numerical issues from the diagonal fallback
                    if let ScalingBlock::PsdStructured { n, .. } = &scaling[cone_idx] {
                        let n_psd = *n;
                        // Copy s
                        for i in 0..dim {
                            d_s_comb[offset + i] = state.s[offset + i];
                        }
                        // Subtract σμ from diagonal elements
                        // Diagonal (k,k) is at svec index k*(k+3)/2 for k=0..n-1
                        for k in 0..n_psd {
                            let diag_idx = k * (k + 3) / 2;
                            d_s_comb[offset + diag_idx] -= target_mu;
                        }
                    } else {
                        // Fallback: just use s for d_s_comb
                        for i in 0..dim {
                            d_s_comb[offset + i] = state.s[offset + i];
                        }
                    }
                } else if is_soc {
                    if let ScalingBlock::SocStructured { w } = &scaling[cone_idx] {
                        let z_slice = &state.z[offset..offset + dim];
                        let ds_aff_slice = &ds_aff[offset..offset + dim];
                        let dz_aff_slice = &dz_aff[offset..offset + dim];

                        // Build W = P(w^{1/2}) and W^{-1} = P(w^{-1/2})
                        let mut w_half = vec![0.0; dim];
                        nt::jordan_sqrt_apply(w, &mut w_half);

                        let mut w_half_inv = vec![0.0; dim];
                        nt::jordan_inv_apply(&w_half, &mut w_half_inv);

                        // λ = W z
                        let mut lambda = vec![0.0; dim];
                        nt::quad_rep_apply(&w_half, z_slice, &mut lambda);

                        // η = (W^{-1} ds_aff) ∘ (W dz_aff)
                        let mut w_inv_ds = vec![0.0; dim];
                        nt::quad_rep_apply(&w_half_inv, ds_aff_slice, &mut w_inv_ds);

                        let mut w_dz = vec![0.0; dim];
                        nt::quad_rep_apply(&w_half, dz_aff_slice, &mut w_dz);

                        let mut eta = vec![0.0; dim];
                        nt::jordan_product_apply(&w_inv_ds, &w_dz, &mut eta);

                        // v = λ∘λ + η - σμ e, with e = (1, 0, ..., 0)
                        let mut lambda_sq = vec![0.0; dim];
                        nt::jordan_product_apply(&lambda, &lambda, &mut lambda_sq);

                        let mut v = vec![0.0; dim];
                        v[0] = lambda_sq[0] + eta[0] - target_mu;
                        for i in 1..dim {
                            v[i] = lambda_sq[i] + eta[i];
                        }

                        // u solves λ ∘ u = v
                        let mut u = vec![0.0; dim];
                        nt::jordan_solve_apply(&lambda, &v, &mut u);

                        // d_s = W^T u (W is self-adjoint for SOC)
                        let mut d_s_block = vec![0.0; dim];
                        nt::quad_rep_apply(&w_half, &u, &mut d_s_block);

                        d_s_comb[offset..offset + dim].copy_from_slice(&d_s_block);
                    } else {
                        // Fallback: diagonal correction with bounded Mehrotra term
                        for i in offset..offset + dim {
                            let s_i = state.s[i];
                            let z_i = state.z[i];
                            let mu_i = s_i * z_i;
                            let z_safe = z_i.max(1e-14);

                            let ds_dz = ds_aff[i] * dz_aff[i];
                            let correction_bound = mu_i.abs().max(target_mu * 0.1);
                            let ds_dz_bounded = ds_dz.clamp(-correction_bound, correction_bound);

                            let w_base = mu_i + ds_dz_bounded;
                            d_s_comb[i] = (w_base - target_mu) / z_safe;
                        }
                    }
                } else {
                    // Mehrotra correction for NonNeg cone
                    // Use bounded correction to prevent numerical blow-up near boundaries
                    for i in offset..offset + dim {
                        let s_i = state.s[i];
                        let z_i = state.z[i];
                        let mu_i = s_i * z_i;
                        let z_safe = z_i.max(1e-14);

                        // Mehrotra correction term with bounding
                        let ds_dz = ds_aff[i] * dz_aff[i];
                        let correction_bound = mu_i.abs().max(target_mu * 0.1);
                        let ds_dz_bounded = ds_dz.clamp(-correction_bound, correction_bound);

                        // MCC delta if present
                        let delta = if is_nonneg {
                            mcc_delta.as_ref().map_or(0.0, |d| d[i])
                        } else {
                            0.0
                        };

                        let w_base = mu_i + ds_dz_bounded;
                        d_s_comb[i] = (w_base - target_mu - delta) / z_safe;
                    }
                }

                offset += dim;
                let _ = cone_idx;
            }

            // rhs_z = d_s - d_z (weighted feasibility residual)
            let rhs_z_comb: Vec<f64> = d_s_comb.iter().zip(residuals.r_z.iter())
                .map(|(ds_i, rz_i)| ds_i - feas_weight * rz_i)
                .collect();

            {
                let start = Instant::now();
                kkt.solve_refined(
                    &factor,
                    &rhs_x_comb,
                    &rhs_z_comb,
                    &mut dx,
                    &mut dz,
                    refine_iters,
                );
                timings.kkt_solve += start.elapsed();
            }

            // Compute dtau for corrector step using Schur complement formula
            // From design doc §7.3:
            //   d_tau = r_tau
            //   d_kappa = κτ + Δκ_aff Δτ_aff - σμ
            //
            // Schur complement numerator: d_tau - d_kappa/τ + (2Pξ+q)ᵀΔx + bᵀΔz
            let d_tau_corr = feas_weight * residuals.r_tau;

            let dot_mul_p_xi_q_dx: f64 = mul_p_xi_q.iter().zip(dx.iter()).map(|(a, b)| a * b).sum();
            let dot_b_dz: f64 = prob.b.iter().zip(dz.iter()).map(|(a, b)| a * b).sum();
            let numerator_corr = d_tau_corr - d_kappa_corr / state.tau + dot_mul_p_xi_q_dx + dot_b_dz;

            dtau = compute_dtau(numerator_corr, denominator, state.tau, denom_scale)
                .map_err(|e| format!("corrector dtau failed: {}", e))?;

            apply_tau_direction(&mut dx, &mut dz, dtau, &dx2, &dz2);

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
                        ScalingBlock::SocStructured { w } => {
                            // For SOC, H = P(w) (quadratic representation)
                            // ds = -d_s - P(w)*dz
                            let dz_slice = &dz[offset..offset + dim];
                            let mut h_dz = vec![0.0; dim];
                            crate::scaling::nt::quad_rep_apply(w, dz_slice, &mut h_dz);
                            for i in 0..dim {
                                ds[offset + i] = -d_s_comb[offset + i] - h_dz[i];
                            }
                        }
                        ScalingBlock::PsdStructured { w_factor, n } => {
                            // For PSD, H = W ⊗ W (Kronecker product in svec form)
                            // ds = -d_s - H*dz = -d_s - W*Dz*W
                            let w = DMatrix::<f64>::from_row_slice(*n, *n, w_factor);
                            let dz_mat = svec_to_mat(&dz[offset..offset + dim], *n);
                            let h_dz_mat = &w * &dz_mat * &w;
                            let mut h_dz = vec![0.0; dim];
                            mat_to_svec(&h_dz_mat, &mut h_dz);
                            for i in 0..dim {
                                ds[offset + i] = -d_s_comb[offset + i] - h_dz[i];
                            }
                        }
                        _ => {
                            // Fallback for Dense3x3 (exp/pow cones), etc.
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
            let mut ls_reported = false;
            for _ in 0..settings.line_search_max_iters {
                if centrality_ok_nonneg_trial(
                    state,
                    &ds,
                    &dz,
                    dtau,
                    dkappa,
                    cones,
                    settings.centrality_beta,
                    settings.centrality_gamma,
                    barrier_degree,
                    alpha,
                ) {
                    break;
                }
                if diagnostics_enabled() && !ls_reported {
                    if let Some(violation) = centrality_nonneg_violation(
                        state,
                        &ds,
                        &dz,
                        dtau,
                        dkappa,
                        cones,
                        settings.centrality_beta,
                        settings.centrality_gamma,
                        barrier_degree,
                        alpha,
                    ) {
                        let idx_str = if violation.idx == usize::MAX {
                            "n/a".to_string()
                        } else {
                            violation.idx.to_string()
                        };
                        eprintln!(
                            "centrality ls fail: alpha={:.3e} side={} idx={} w={:.3e} bounds=[{:.3e},{:.3e}] s={:.3e} z={:.3e} mu_trial={:.3e} tau_trial={:.3e} kappa_trial={:.3e}",
                            alpha,
                            violation.side,
                            idx_str,
                            violation.w,
                            violation.lower,
                            violation.upper,
                            violation.s_i,
                            violation.z_i,
                            violation.mu_trial,
                            violation.tau_trial,
                            violation.kappa_trial
                        );
                    } else {
                        eprintln!(
                            "centrality ls fail: alpha={:.3e} (no nonneg violation found)",
                            alpha
                        );
                    }
                    ls_reported = true;
                }
                alpha *= 0.5;
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
                    eprintln!("bumped KKT static_reg to {:.2e} after alpha stall", bump_reg);
                }
            }
            sigma_eff = (sigma_eff + 0.2).min(sigma_cap);
            refine_iters = refine_iters.saturating_add(2);
        } else {
            sigma_eff = sigma_cap;
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

    if diagnostics_enabled() {
        if let Some(diag) = nonneg_step_diagnostics(&state.s, &ds, &state.z, &dz, cones) {
            let lim_idx = if diag.alpha_lim_idx == usize::MAX {
                "none".to_string()
            } else {
                diag.alpha_lim_idx.to_string()
            };
            let nonneg_limits = diag.alpha_lim.is_finite()
                && alpha_sz.is_finite()
                && (diag.alpha_lim - alpha_sz).abs() <= 1e-12 * alpha_sz.max(1.0);
            eprintln!(
                "nonneg diag: min_s={:.3e} min_z={:.3e} min_s_over_z={:.3e} alpha_nonneg={:.3e} lim_idx={} lim_side={} alpha_sz={:.3e} alpha={:.3e} nonneg_limits={}",
                diag.min_s,
                diag.min_z,
                diag.min_ratio,
                diag.alpha_lim,
                lim_idx,
                diag.alpha_lim_side,
                alpha_sz,
                alpha,
                nonneg_limits
            );
        }
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
                    state.s[i] = 0.0;  // Keep at 0
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

    // Safety clamp (should rarely trigger now with proper step size)
    if state.kappa < 1e-12 {
        state.kappa = 1e-12;
    }

    // Update ξ = x/τ for next iteration's Schur complement
    for i in 0..n {
        state.xi[i] = state.x[i] / state.tau;
    }

    // Compute new μ
    let mu_new = compute_mu(state, barrier_degree);

    Ok(StepResult {
        alpha,
        alpha_sz,
        sigma: sigma_used,
        mu_new,
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
    let mut offset = 0usize;

    for cone in cones.iter() {
        let dim = cone.dim();
        if dim == 0 {
            continue;
        }

        let s_slice = &s[offset..offset + dim];
        let ds_slice = &ds[offset..offset + dim];
        let z_slice = &z[offset..offset + dim];
        let dz_slice = &dz[offset..offset + dim];

        // Barrier-free cones (e.g., Zero) don't constrain step size.
        if cone.barrier_degree() == 0 {
            offset += dim;
            continue;
        }

        // Non-finite directions -> safest possible step is 0.0.
        if !all_finite(ds_slice) || !all_finite(dz_slice) {
            return 0.0;
        }

        let alpha_p = cone.step_to_boundary_primal(s_slice, ds_slice);
        let alpha_d = cone.step_to_boundary_dual(z_slice, dz_slice);

        if alpha_p.is_finite() {
            alpha = alpha.min(alpha_p.max(0.0));
        }
        if alpha_d.is_finite() {
            alpha = alpha.min(alpha_d.max(0.0));
        }

        if alpha == 0.0 {
            break;
        }

        offset += dim;
    }

    if alpha.is_finite() {
        (fraction * alpha).min(1.0)
    } else {
        1.0
    }
}

/// Compute μ_aff = complementarity after affine step.
///
/// IMPORTANT: Only cones with barrier_degree > 0 (NonNeg, SOC) contribute.
/// Zero cones (equalities) must be excluded or they can pollute μ_aff
/// with large residual values, causing σ to saturate incorrectly.
fn compute_mu_aff(
    state: &HsdeState,
    ds_aff: &[f64],
    dz_aff: &[f64],
    dtau_aff: f64,
    dkappa_aff: f64,
    alpha_aff: f64,
    barrier_degree: usize,
    cones: &[Box<dyn ConeKernel>],
) -> f64 {
    if barrier_degree == 0 {
        return 0.0;
    }

    let tau_aff = state.tau + alpha_aff * dtau_aff;
    let kappa_aff = state.kappa + alpha_aff * dkappa_aff;
    if !tau_aff.is_finite() || !kappa_aff.is_finite() || tau_aff <= 0.0 || kappa_aff <= 0.0 {
        return f64::NAN;
    }

    // Iterate by cone blocks, only including cones with barrier_degree > 0
    let mut s_dot_z = 0.0;
    let mut offset = 0;
    for cone in cones {
        let dim = cone.dim();
        if dim == 0 {
            continue;
        }

        // Skip Zero cones (barrier_degree == 0) - they shouldn't contribute
        if cone.barrier_degree() > 0 {
            for i in offset..offset + dim {
                let s_i = state.s[i] + alpha_aff * ds_aff[i];
                let z_i = state.z[i] + alpha_aff * dz_aff[i];
                s_dot_z += s_i * z_i;
            }
        }
        offset += dim;
    }

    (s_dot_z + tau_aff * kappa_aff) / (barrier_degree as f64 + 1.0)
}

/// Compute centering parameter σ using μ_aff when reliable.
fn compute_centering_parameter(
    alpha_aff: f64,
    mu: f64,
    mu_aff: f64,
    barrier_degree: usize,
) -> f64 {
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
        assert!(sigma > 0.9, "σ should be large for small affine step, got {}", sigma);
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
}
