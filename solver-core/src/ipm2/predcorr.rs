//! Predictor-corrector steps for HSDE interior point method (ipm2, allocation-free).
//!
//! The predictor-corrector algorithm has two phases per iteration:
//! 1. **Affine step**: Solve KKT system with σ = 0 (pure Newton step)
//! 2. **Combined step**: Solve with Mehrotra correction (adds centering)
//!
//! This implementation follows §7 of the design doc, but reuses workspace buffers
//! to avoid per-iteration allocations.

use std::any::Any;

use crate::cones::{ConeKernel, NonNegCone, SocCone, ExpCone, PowCone, PsdCone, exp_dual_map_block, exp_central_ok, exp_third_order_correction};
use crate::cones::psd::{mat_to_svec, svec_to_mat};
use crate::ipm::hsde::{compute_mu, HsdeResiduals, HsdeState};
use crate::ipm2::{IpmWorkspace, PerfSection, PerfTimers};
use crate::ipm2::workspace::SocScratch;
use crate::linalg::kkt_trait::KktSolverTrait;
use crate::linalg::unified_kkt::UnifiedKktSolver;
use crate::problem::{ProblemData, SolverSettings};
use crate::scaling::{ScalingBlock, nt, bfgs};
use nalgebra::DMatrix;
use nalgebra::linalg::SymmetricEigen;

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
        // Legacy: check cone-specific debug vars
        std::env::var("MINIX_EXP_DEBUG").is_ok()
            || std::env::var("MINIX_EXP_CENTRAL_CHECK").is_ok()
            || std::env::var("MINIX_QFORPLAN_DIAG").is_ok()
    })
}

/// Check if tau should be frozen to 1.0 (for debugging tau dynamics issues).
/// Set MINIX_FREEZE_TAU=1 to enable. This prevents tau drift that can cause
/// primal residual floors in HSDE. See: _planning/v22/LOG.md
fn freeze_tau_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("MINIX_FREEZE_TAU")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false)
    })
}

/// Check if full feasibility weighting should be used (feas_weight=1.0).
///
/// The default Mehrotra predictor-corrector uses feas_weight = 1 - sigma,
/// which can downweight feasibility when sigma is high. This causes
/// "chasing complementarity while ignoring feasibility" failures on SDPs.
///
/// MINIX_FULL_FEAS=0 to use original formula; otherwise use feas_weight=1.0.
/// See: _planning/v23/improvements
fn full_feas_weight_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("MINIX_FULL_FEAS")
            .map(|v| v != "0")
            .unwrap_or(true)  // Default: full feasibility weighting
    })
}

fn psd_reg_cap_value() -> f64 {
    static CAP: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        let floor = std::env::var("MINIX_PSD_REG_FLOOR")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(1e-8);
        let mut cap = std::env::var("MINIX_PSD_REG_CAP")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(1e-6);
        if !cap.is_finite() || cap <= 0.0 {
            cap = 1e-6;
        }
        cap.max(floor)
    })
}

fn psd_reg_cap_for_cones(cones: &[Box<dyn ConeKernel>]) -> Option<f64> {
    let has_psd = cones.iter().any(|c| {
        (c.as_ref() as &dyn Any).is::<PsdCone>()
    });
    if has_psd { Some(psd_reg_cap_value()) } else { None }
}

fn all_finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
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

        let is_nonneg = (cone.as_ref() as &dyn Any).is::<NonNegCone>();
        let is_soc = (cone.as_ref() as &dyn Any).is::<SocCone>();

        if is_nonneg {
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
        } else if is_soc {
            // SOC: check NT-scaled complementarity eigenvalues (geometric means)
            // Use relaxed bounds matching centrality_ok_nonneg_trial
            let beta_soc = beta * 0.1;
            let gamma_soc = (gamma * 10.0).min(1000.0);
            let lower_soc = beta_soc * mu_trial;
            let upper_soc = gamma_soc * mu_trial;

            let s0 = state.s[offset] + alpha * ds[offset];
            let z0 = state.z[offset] + alpha * dz[offset];

            let mut s_norm_sq = 0.0;
            let mut z_norm_sq = 0.0;
            for i in 1..dim {
                let si = state.s[offset + i] + alpha * ds[offset + i];
                let zi = state.z[offset + i] + alpha * dz[offset + i];
                s_norm_sq += si * si;
                z_norm_sq += zi * zi;
            }
            let s_norm = s_norm_sq.sqrt();
            let z_norm = z_norm_sq.sqrt();

            let s_hi = s0 + s_norm;
            let s_lo = if s0 <= s_norm { s0 - s_norm } else {
                let d = s0 + s_norm; if d == 0.0 { 0.0 } else { s0.mul_add(s0, -s_norm_sq) / d }
            };
            let z_hi = z0 + z_norm;
            let z_lo = if z0 <= z_norm { z0 - z_norm } else {
                let d = z0 + z_norm; if d == 0.0 { 0.0 } else { z0.mul_add(z0, -z_norm_sq) / d }
            };

            if s_lo <= 0.0 || z_lo <= 0.0 {
                return Some(CentralityViolation {
                    idx: offset,
                    side: "soc_not_interior",
                    w: s_lo.min(z_lo),
                    lower: lower_soc,
                    upper: upper_soc,
                    s_i: s0,
                    z_i: z0,
                    mu_trial,
                    tau_trial,
                    kappa_trial,
                });
            }

            let comp_hi = (s_hi * z_hi).sqrt();
            let comp_lo = (s_lo * z_lo).sqrt();

            if comp_lo < lower_soc {
                return Some(CentralityViolation {
                    idx: offset,
                    side: "soc_low",
                    w: comp_lo,
                    lower: lower_soc,
                    upper: upper_soc,
                    s_i: s0,
                    z_i: z0,
                    mu_trial,
                    tau_trial,
                    kappa_trial,
                });
            }
            if comp_hi > upper_soc {
                return Some(CentralityViolation {
                    idx: offset,
                    side: "soc_high",
                    w: comp_hi,
                    lower: lower_soc,
                    upper: upper_soc,
                    s_i: s0,
                    z_i: z0,
                    mu_trial,
                    tau_trial,
                    kappa_trial,
                });
            }
        }

        offset += dim;
    }

    None
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

fn compute_dtau(
    numerator: f64,
    denominator: f64,
    tau: f64,
    denom_scale: f64,
) -> Result<f64, String> {
    if !numerator.is_finite() || !denominator.is_finite() || !tau.is_finite() || !denom_scale.is_finite() {
        return Err("dtau inputs not finite".to_string());
    }
    if tau <= 0.0 {
        return Err(format!("tau non-positive (tau={:.3e})", tau));
    }

    // If the denominator is ill-conditioned, treat the update as unreliable and
    // fall back to a no-op step for tau. This is more robust than failing the
    // entire iteration, and mirrors the common IPM practice of dampening or
    // skipping scalar updates when the underlying 2x2 system is nearly singular.
    let scale = denom_scale.max(1.0);
    if denominator.abs() <= 1e-10 * scale {
        return Ok(0.0);
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

fn clamp_complementarity_nonneg_in_place(
    state: &HsdeState,
    ds: &[f64],
    dz: &[f64],
    cones: &[Box<dyn ConeKernel>],
    beta: f64,
    gamma: f64,
    mu: f64,
    delta_w: &mut [f64],
) -> bool {
    if mu <= 0.0 {
        delta_w.fill(0.0);
        return false;
    }

    let mut has_nonneg = false;
    let mut changed = false;
    delta_w.fill(0.0);
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

    has_nonneg && changed
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

    let mut offset = 0;
    for cone in cones {
        let dim = cone.dim();
        if dim == 0 {
            continue;
        }

        let is_nonneg = (cone.as_ref() as &dyn Any).is::<NonNegCone>();
        let is_soc = (cone.as_ref() as &dyn Any).is::<SocCone>();

        if is_nonneg {
            // NonNeg: check w = s_i * z_i ∈ [β·μ, γ·μ] for each component
            for i in 0..dim {
                let idx = offset + i;
                let s_i = state.s[idx] + alpha * ds[idx];
                let z_i = state.z[idx] + alpha * dz[idx];
                let w = s_i * z_i;
                if w < beta * mu_trial || w > gamma * mu_trial {
                    return false;
                }
            }
        } else if is_soc {
            // For SOC cones, enforce the central neighborhood using NT-scaled complementarity.
            // Instead of checking raw s∘z eigenvalues (which can be outside SOC even when
            // s,z are interior), we check the geometric means: sqrt(λ_i(s) * λ_i(z)).
            // This is the correct symmetric-cone neighborhood check (Clarabel-style).
            //
            // SOC cones use relaxed centrality bounds compared to NonNeg:
            // - beta_soc = beta * 0.1 (10x looser lower bound)
            // - gamma_soc = gamma * 10 (10x looser upper bound, clamped to 1000)
            // This accounts for the more complex geometry of SOC cones and avoids
            // overly restricting steps on ill-conditioned problems.
            let beta_soc = beta * 0.1;
            let gamma_soc = (gamma * 10.0).min(1000.0);

            let s0 = state.s[offset] + alpha * ds[offset];
            let z0 = state.z[offset] + alpha * dz[offset];

            // Compute ||s̄|| and ||z̄||
            let mut s_norm_sq = 0.0;
            let mut z_norm_sq = 0.0;
            for i in 1..dim {
                let si = state.s[offset + i] + alpha * ds[offset + i];
                let zi = state.z[offset + i] + alpha * dz[offset + i];
                s_norm_sq += si * si;
                z_norm_sq += zi * zi;
            }
            let s_norm = s_norm_sq.sqrt();
            let z_norm = z_norm_sq.sqrt();

            // Eigenvalues of s: λ_max(s) = s0 + ||s̄||, λ_min(s) = s0 - ||s̄||
            let s_hi = s0 + s_norm;
            let s_lo = if s0 <= s_norm {
                s0 - s_norm
            } else {
                let denom = s0 + s_norm;
                if denom == 0.0 { 0.0 } else { s0.mul_add(s0, -s_norm_sq) / denom }
            };

            // Eigenvalues of z: λ_max(z) = z0 + ||z̄||, λ_min(z) = z0 - ||z̄||
            let z_hi = z0 + z_norm;
            let z_lo = if z0 <= z_norm {
                z0 - z_norm
            } else {
                let denom = z0 + z_norm;
                if denom == 0.0 { 0.0 } else { z0.mul_add(z0, -z_norm_sq) / denom }
            };

            // Check interior: both s and z must be strictly interior
            if s_lo <= 0.0 || z_lo <= 0.0 {
                return false;
            }

            // NT-scaled complementarity eigenvalues (geometric means)
            let comp_hi = (s_hi * z_hi).sqrt();
            let comp_lo = (s_lo * z_lo).sqrt();

            // Check neighborhood with relaxed SOC bounds: β_soc·μ ≤ comp_lo and comp_hi ≤ γ_soc·μ
            if comp_lo < beta_soc * mu_trial || comp_hi > gamma_soc * mu_trial {
                return false;
            }
        }

        offset += dim;
    }

    true
}

// SOC helpers (allocation-free)
#[inline]
fn soc_x_norm(v: &[f64]) -> f64 {
    v[1..].iter().map(|&xi| xi * xi).sum::<f64>().sqrt()
}

fn spectral_decomposition_in_place(v: &[f64], lambda: &mut [f64; 2], e1: &mut [f64], e2: &mut [f64]) {
    let t = v[0];
    let x_norm = if v.len() == 1 { 0.0 } else { soc_x_norm(v) };

    lambda[0] = t + x_norm;

    // Compute lambda[1] stably to avoid catastrophic cancellation when t ≈ ||x||:
    // λ₂ = t - ||x|| = (t² - ||x||²) / (t + ||x||).
    let lambda0 = lambda[0];
    if lambda0 != 0.0 {
        let scale = t.abs().max(x_norm);
        let det = if scale == 0.0 {
            0.0
        } else {
            let ts = t / scale;
            let xs = x_norm / scale;
            ts.mul_add(ts, -(xs * xs)) * (scale * scale)
        };
        lambda[1] = (det / lambda0).max(0.0);
    } else {
        // Fallback (should be unreachable for interior points).
        lambda[1] = t - x_norm;
    }

    if x_norm > 1e-14 {
        let inv_norm = 1.0 / x_norm;
        e1[0] = 0.5;
        e2[0] = 0.5;
        for i in 1..v.len() {
            let x_normalized = v[i] * inv_norm;
            e1[i] = 0.5 * x_normalized;
            e2[i] = -0.5 * x_normalized;
        }
    } else {
        e1[0] = 0.5;
        e2[0] = 0.5;
        for i in 1..v.len() {
            e1[i] = 0.0;
            e2[i] = 0.0;
        }
    }
}

fn jordan_product_in_place(a: &[f64], b: &[f64], out: &mut [f64]) {
    let t = a[0];
    let u = b[0];

    out[0] = t * u;
    for i in 1..a.len() {
        out[0] += a[i] * b[i];
    }

    for i in 1..a.len() {
        out[i] = t * b[i] + u * a[i];
    }
}

fn jordan_sqrt_in_place(v: &[f64], out: &mut [f64], e1: &mut [f64], e2: &mut [f64]) {
    let mut lambda = [0.0; 2];
    spectral_decomposition_in_place(v, &mut lambda, e1, e2);

    let sqrt_l1 = lambda[0].sqrt();
    let sqrt_l2 = lambda[1].sqrt();
    for i in 0..v.len() {
        out[i] = sqrt_l1 * e1[i] + sqrt_l2 * e2[i];
    }
}

fn jordan_inv_in_place(v: &[f64], out: &mut [f64], e1: &mut [f64], e2: &mut [f64]) {
    let mut lambda = [0.0; 2];
    spectral_decomposition_in_place(v, &mut lambda, e1, e2);

    let inv_l1 = 1.0 / lambda[0];
    let inv_l2 = 1.0 / lambda[1];
    for i in 0..v.len() {
        out[i] = inv_l1 * e1[i] + inv_l2 * e2[i];
    }
}

fn quad_rep_in_place(
    w: &[f64],
    y: &[f64],
    out: &mut [f64],
    w_circ_y: &mut [f64],
    w_circ_w: &mut [f64],
    temp: &mut [f64],
    w2_circ_y: &mut [f64],
) {
    jordan_product_in_place(w, y, w_circ_y);
    jordan_product_in_place(w, w, w_circ_w);

    jordan_product_in_place(w_circ_y, w, temp);
    for i in 0..w.len() {
        temp[i] *= 2.0;
    }

    jordan_product_in_place(w_circ_w, y, w2_circ_y);

    for i in 0..w.len() {
        out[i] = temp[i] - w2_circ_y[i];
    }
}

fn jordan_solve_in_place(lambda: &[f64], v: &[f64], out: &mut [f64], e1: &mut [f64], e2: &mut [f64]) {
    let mut eigen = [0.0; 2];
    spectral_decomposition_in_place(lambda, &mut eigen, e1, e2);

    let e1_dot: f64 = e1.iter().zip(e1.iter()).map(|(a, b)| a * b).sum();
    let e2_dot: f64 = e2.iter().zip(e2.iter()).map(|(a, b)| a * b).sum();

    let v1: f64 = v.iter().zip(e1.iter()).map(|(vi, ei)| vi * ei).sum::<f64>() / e1_dot;
    let v2: f64 = v.iter().zip(e2.iter()).map(|(vi, ei)| vi * ei).sum::<f64>() / e2_dot;

    let inv_l1 = 1.0 / eigen[0].max(1e-14);
    let inv_l2 = 1.0 / eigen[1].max(1e-14);

    for i in 0..lambda.len() {
        out[i] = (v1 * inv_l1) * e1[i] + (v2 * inv_l2) * e2[i];
    }
}

fn nt_scaling_nonneg_in_place(s: &[f64], z: &[f64], d: &mut [f64]) -> Result<(), ()> {
    if s.iter().any(|&x| !x.is_finite() || x <= 0.0)
        || z.iter().any(|&x| !x.is_finite() || x <= 0.0)
    {
        return Err(());
    }

    // Clamp to numerically safe range (matches nt_scaling_nonneg in nt.rs)
    for i in 0..s.len() {
        d[i] = (s[i] / z[i]).clamp(1e-18, 1e18);
    }

    Ok(())
}

fn nt_scaling_soc_in_place(
    cone: &SocCone,
    s: &[f64],
    z: &[f64],
    w: &mut [f64],
    scratch: &mut SocScratch,
) -> Result<(), ()> {
    if !cone.is_interior_scaling(s) || !cone.is_interior_scaling(z) {
        return Err(());
    }

    let dim = cone.dim();
    let s_sqrt = &mut scratch.s_sqrt[..dim];
    let u = &mut scratch.u[..dim];
    let u_inv = &mut scratch.u_inv[..dim];
    let u_inv_sqrt = &mut scratch.u_inv_sqrt[..dim];
    let e1 = &mut scratch.e1[..dim];
    let e2 = &mut scratch.e2[..dim];
    let w_circ_y = &mut scratch.w_circ_y[..dim];
    let w_circ_w = &mut scratch.w_circ_w[..dim];
    let temp = &mut scratch.temp[..dim];
    let w2_circ_y = &mut scratch.w2_circ_y[..dim];

    jordan_sqrt_in_place(s, s_sqrt, e1, e2);
    quad_rep_in_place(s_sqrt, z, u, w_circ_y, w_circ_w, temp, w2_circ_y);
    jordan_inv_in_place(u, u_inv, e1, e2);
    jordan_sqrt_in_place(u_inv, u_inv_sqrt, e1, e2);
    quad_rep_in_place(s_sqrt, u_inv_sqrt, w, w_circ_y, w_circ_w, temp, w2_circ_y);

    Ok(())
}

/// Allocation-free predictor-corrector step using workspace buffers.
pub fn predictor_corrector_step_in_place(
    kkt: &mut UnifiedKktSolver,
    prob: &ProblemData,
    neg_q: &[f64],
    state: &mut HsdeState,
    residuals: &HsdeResiduals,
    cones: &[Box<dyn ConeKernel>],
    mu: f64,
    barrier_degree: usize,
    settings: &SolverSettings,
    ws: &mut IpmWorkspace,
    timers: &mut PerfTimers,
) -> Result<StepResult, String> {
    let n = prob.num_vars();
    let m = prob.num_constraints();

    assert_eq!(neg_q.len(), n, "neg_q must have length n");

    // ======================================================================
    // Step 1: Compute NT scaling for all cones with adaptive regularization
    // ======================================================================
    {
        let _g = timers.scoped(PerfSection::Scaling);
        let mut offset = 0;
        let mut nt_fallbacks: usize = 0;

        for (cone_idx, cone) in cones.iter().enumerate() {
            let dim = cone.dim();
            if dim == 0 {
                continue;
            }

            if cone.barrier_degree() == 0 {
                offset += dim;
                continue;
            }

            let s = &state.s[offset..offset + dim];
            let z = &state.z[offset..offset + dim];

            let is_soc = (cone.as_ref() as &dyn Any).is::<SocCone>();

            let update_ok = match &mut ws.scaling[cone_idx] {
                ScalingBlock::Diagonal { d } => nt_scaling_nonneg_in_place(s, z, d).is_ok(),
                ScalingBlock::SocStructured { w } => {
                    if let Some(soc_cone) = (cone.as_ref() as &dyn Any).downcast_ref::<SocCone>() {
                        nt_scaling_soc_in_place(soc_cone, s, z, w, &mut ws.soc_scratch).is_ok()
                    } else {
                        false
                    }
                }
                ScalingBlock::Dense3x3 { .. } => {
                    if (cone.as_ref() as &dyn Any).is::<ExpCone>()
                        || (cone.as_ref() as &dyn Any).is::<PowCone>()
                    {
                        if let Ok(block) = bfgs::bfgs_scaling_3d(s, z, cone.as_ref()) {
                            ws.scaling[cone_idx] = block;
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
                ScalingBlock::PsdStructured { .. } => {
                    if let Some(psd_cone) = (cone.as_ref() as &dyn Any).downcast_ref::<PsdCone>() {
                        if let Ok(block) = nt::nt_scaling_psd(psd_cone, s, z) {
                            ws.scaling[cone_idx] = block;
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
                ScalingBlock::Zero { .. } => true,
            };

            if !update_ok {
                nt_fallbacks += 1;
                if is_soc {
                    // Fallback to diagonal scaling for SOC if NT fails (reuse SOC buffer).
                    let mut d = match std::mem::replace(
                        &mut ws.scaling[cone_idx],
                        ScalingBlock::Zero { dim },
                    ) {
                        ScalingBlock::SocStructured { w } => w,
                        ScalingBlock::Diagonal { d } => d,
                        other => {
                            ws.scaling[cone_idx] = other;
                            offset += dim;
                            continue;
                        }
                    };
                    if d.len() != dim {
                        d.resize(dim, 0.0);
                    }
                    for i in 0..dim {
                        let ratio = s[i] / z[i];
                        // Match ipm1 fallback: use 1.0 for invalid ratios
                        d[i] = if ratio.is_finite() && ratio > 0.0 {
                            ratio.clamp(1e-12, 1e12)
                        } else {
                            1.0
                        };
                    }
                    ws.scaling[cone_idx] = ScalingBlock::Diagonal { d };
                } else if matches!(ws.scaling[cone_idx], ScalingBlock::Dense3x3 { .. }) {
                    // Fallback to identity for nonsymmetric cones (exp/pow) when BFGS fails
                    // This gives a well-conditioned but less accurate scaling
                    ws.scaling[cone_idx] = ScalingBlock::Dense3x3 {
                        h: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    };
                } else if let ScalingBlock::Diagonal { d } = &mut ws.scaling[cone_idx] {
                    for i in 0..dim {
                        let ratio = s[i] / z[i];
                        // Match ipm1 fallback: use 1.0 for invalid ratios
                        d[i] = if ratio.is_finite() && ratio > 0.0 {
                            ratio.clamp(1e-12, 1e12)
                        } else {
                            1.0
                        };
                    }
                }
            }

            offset += dim;
        }

        if diagnostics_enabled() && nt_fallbacks > 0 {
            eprintln!("nt scaling fallback: blocks={}, mu={:.3e}", nt_fallbacks, mu);
        }
    }

    // ======================================================================
    // Step 2: Factor KKT system
    // ======================================================================
    let factor = {
        const MAX_REG_RETRIES: usize = 3;
        const MAX_STATIC_REG: f64 = 1e-2;
        let max_static_reg = psd_reg_cap_for_cones(cones).unwrap_or(MAX_STATIC_REG);
        let mut retries = 0usize;
        loop {
            {
                let _g = timers.scoped(PerfSection::KktUpdate);
                kkt.update_numeric(prob.P.as_ref(), &prob.A, &ws.scaling)
                    .map_err(|e| format!("KKT update failed: {}", e))?;
            }

            // P1.2: Try factorization with shift-and-retry for quasi-definiteness failures
            let factor_result = {
                let _g = timers.scoped(PerfSection::Factorization);
                kkt.factorize()
            };

            let factor = match factor_result {
                Ok(f) => f,
                Err(e) => {
                    // Check if this is a quasi-definiteness failure
                    let is_qd_failure = e.to_string().contains("not quasi-definite");

                    if is_qd_failure && retries < MAX_REG_RETRIES {
                        // P1.2: Increase regularization and retry for quasi-definiteness failures
                        let current_reg = kkt.static_reg();
                        let next_reg = if current_reg < 1e-10 {
                            1e-10  // Start with small shift if reg is tiny
                        } else {
                            (current_reg * 100.0).min(max_static_reg)
                        };

                        if diagnostics_enabled() {
                            eprintln!(
                                "P1.2: quasi-definite failure, retry {} with reg {:.3e} -> {:.3e}",
                                retries + 1, current_reg, next_reg
                            );
                        }

                        kkt.set_static_reg(next_reg)
                            .map_err(|e| format!("KKT reg update failed: {}", e))?;
                        retries += 1;
                        continue; // Retry factorization
                    } else {
                        // Not a QD failure, or exhausted retries - propagate error
                        return Err(format!("KKT factorization failed: {}", e).into());
                    }
                }
            };

            let bumps = kkt.dynamic_bumps();
            if bumps == 0 || retries >= MAX_REG_RETRIES {
                break factor;
            }

            let next_reg = (kkt.static_reg() * 10.0).min(max_static_reg);
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
    for i in 0..n {
        ws.rhs_x[i] = -residuals.r_x[i];
    }
    for i in 0..m {
        ws.rhs_z[i] = state.s[i] - residuals.r_z[i];
    }

    {
        let _g = timers.scoped(PerfSection::Solve);
        kkt.solve_two_rhs_refined_tagged(
            &factor,
            &ws.rhs_x,
            &ws.rhs_z,
            neg_q,
            &prob.b,
            &mut ws.dx_aff,
            &mut ws.dz_aff,
            &mut ws.dx2,
            &mut ws.dz2,
            settings.kkt_refine_iters,
            "rhs1",
            "rhs2",
        );
    }

    // Compute mul_p_xi = P * xi (if P exists)
    ws.mul_p_xi.fill(0.0);
    if let Some(ref p) = prob.P {
        for col in 0..n {
            if let Some(col_view) = p.outer_view(col) {
                for (row, &val) in col_view.iter() {
                    if row == col {
                        ws.mul_p_xi[row] += val * state.xi[col];
                    } else {
                        ws.mul_p_xi[row] += val * state.xi[col];
                        ws.mul_p_xi[col] += val * state.xi[row];
                    }
                }
            }
        }
    }

    for i in 0..n {
        ws.mul_p_xi_q[i] = 2.0 * ws.mul_p_xi[i] + prob.q[i];
    }

    // Compute dtau via Schur complement formula (design doc §5.4.1)
    let d_tau = residuals.r_tau;
    let d_kappa = state.kappa * state.tau;

    let dot_mul_p_xi_q_dx1: f64 = ws
        .mul_p_xi_q
        .iter()
        .zip(ws.dx_aff.iter())
        .map(|(a, b)| a * b)
        .sum();
    let dot_b_dz1: f64 = prob.b.iter().zip(ws.dz_aff.iter()).map(|(a, b)| a * b).sum();
    let numerator = d_tau - d_kappa / state.tau + dot_mul_p_xi_q_dx1 + dot_b_dz1;

    let dot_xi_mul_p_xi: f64 = state
        .xi
        .iter()
        .zip(ws.mul_p_xi.iter())
        .map(|(a, b)| a * b)
        .sum();
    let dot_mul_p_xi_q_dx2: f64 = ws
        .mul_p_xi_q
        .iter()
        .zip(ws.dx2.iter())
        .map(|(a, b)| a * b)
        .sum();
    let dot_b_dz2: f64 = prob.b.iter().zip(ws.dz2.iter()).map(|(a, b)| a * b).sum();
    let denominator = state.kappa / state.tau + dot_xi_mul_p_xi - dot_mul_p_xi_q_dx2 - dot_b_dz2;

    let denom_scale = (state.kappa / state.tau).abs().max(dot_xi_mul_p_xi.abs());
    let dtau_aff = compute_dtau(numerator, denominator, state.tau, denom_scale)
        .map_err(|e| format!("affine dtau failed: {}", e))?;

    apply_tau_direction(&mut ws.dx_aff, &mut ws.dz_aff, dtau_aff, &ws.dx2, &ws.dz2);

    let dkappa_aff = -(d_kappa + state.kappa * dtau_aff) / state.tau;

    // Compute ds_aff from complementarity equation
    let mut offset = 0;
    for (cone_idx, cone) in cones.iter().enumerate() {
        let dim = cone.dim();
        if dim == 0 {
            continue;
        }

        if cone.barrier_degree() == 0 {
            for i in offset..offset + dim {
                ws.ds_aff[i] = 0.0;
            }
        } else {
            if let ScalingBlock::SocStructured { w } = &ws.scaling[cone_idx] {
                let scratch = &mut ws.soc_scratch;
                let w_circ_y = &mut scratch.w_circ_y[..dim];
                let w_circ_w = &mut scratch.w_circ_w[..dim];
                let temp = &mut scratch.temp[..dim];
                let w2_circ_y = &mut scratch.w2_circ_y[..dim];
                let h_dz = &mut scratch.h_dz[..dim];
                quad_rep_in_place(w, &ws.dz_aff[offset..offset + dim], h_dz, w_circ_y, w_circ_w, temp, w2_circ_y);
                for i in 0..dim {
                    ws.ds_aff[offset + i] = -state.s[offset + i] - h_dz[i];
                }
            } else {
                let dz_slice = &ws.dz_aff[offset..offset + dim];
                let ds_slice = &mut ws.ds_aff[offset..offset + dim];
                ws.scaling[cone_idx].apply(dz_slice, ds_slice);
                for i in 0..dim {
                    ds_slice[i] = -state.s[offset + i] - ds_slice[i];
                }
            }
        }

        offset += dim;
    }

    // Compute affine step size
    let mut alpha_aff = compute_step_size(&state.s, &ws.ds_aff, &state.z, &ws.dz_aff, cones, 1.0);
    if dtau_aff < 0.0 {
        alpha_aff = alpha_aff.min(-state.tau / dtau_aff);
    }
    if dkappa_aff < 0.0 {
        alpha_aff = alpha_aff.min(-state.kappa / dkappa_aff);
    }

    // Compute centering parameter σ
    let mu_aff = compute_mu_aff(
        state,
        &ws.ds_aff,
        &ws.dz_aff,
        dtau_aff,
        dkappa_aff,
        alpha_aff,
        barrier_degree,
        cones,
    );
    let sigma_cap = settings.sigma_max.min(0.999);
    let sigma = compute_centering_parameter(
        alpha_aff,
        mu,
        mu_aff,
        barrier_degree,
    ).min(sigma_cap);

    // ======================================================================
    // Step 5: Combined corrector step (+ step size, with stall recovery)
    // ======================================================================
    ws.dx.fill(0.0);
    ws.dz.fill(0.0);
    ws.ds.fill(0.0);
    ws.d_s_comb.fill(0.0);

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

    // Full feasibility weighting: don't downweight feasibility RHS based on sigma.
    // This prevents "chasing complementarity while ignoring feasibility" which
    // causes SDP convergence issues (e.g., control1).
    // MINIX_FULL_FEAS=1 (or unset) uses feas_weight=1.0; =0 uses original formula.
    let use_full_feas = full_feas_weight_enabled();

    let max_retries = 2usize;
    for attempt in 0..=max_retries {
        let mut has_mcc = false;
        sigma_used = sigma_eff;
        let feas_weight = if use_full_feas {
            1.0
        } else {
            (1.0 - sigma_eff).max(feas_weight_floor)
        };
        final_feas_weight = feas_weight;
        let target_mu = sigma_eff * mu;

        let d_kappa_corr = state.kappa * state.tau + dkappa_aff * dtau_aff - target_mu;

        for i in 0..n {
            ws.rhs_x[i] = -feas_weight * residuals.r_x[i];
        }

        for corr_iter in 0..=settings.mcc_iters {
            ws.d_s_comb.fill(0.0);
            let mut offset = 0;
            for (cone_idx, cone) in cones.iter().enumerate() {
                let dim = cone.dim();
                if dim == 0 {
                    continue;
                }

                if cone.barrier_degree() == 0 {
                    offset += dim;
                    continue;
                }

                let is_soc = (cone.as_ref() as &dyn Any).is::<SocCone>();
                let is_nonneg = (cone.as_ref() as &dyn Any).is::<NonNegCone>();
                let is_psd = (cone.as_ref() as &dyn Any).is::<PsdCone>();

                // TODO: Implement analytical third-order correction for exponential cones
                // Research shows this requires a complex analytical formula (not finite differences).
                // See: _planning/v16/third_order_correction_analysis.md
                // Expected benefit: 3-10x iteration reduction (from 50-200 to 10-30 iters)

                if is_psd {
                    // PSD cone: proper Jordan-algebra Mehrotra corrector with Sylvester solve
                    // This is the PSD analogue of the SOC corrector below
                    if let ScalingBlock::PsdStructured { w_factor, n } = &ws.scaling[cone_idx] {
                        if trace_enabled() && corr_iter == 0 {
                            eprintln!("PSD Mehrotra correction: block {} (n={})", cone_idx, *n);
                        }
                        let n_psd = *n;
                        debug_assert_eq!(dim, n_psd * (n_psd + 1) / 2);

                        let z_slice = &state.z[offset..offset + dim];
                        let ds_aff_slice = &ws.ds_aff[offset..offset + dim];
                        let dz_aff_slice = &ws.dz_aff[offset..offset + dim];

                        // svec -> symmetric matrices
                        let z_mat = svec_to_mat(z_slice, n_psd);
                        let ds_aff_mat = svec_to_mat(ds_aff_slice, n_psd);
                        let dz_aff_mat = svec_to_mat(dz_aff_slice, n_psd);

                        // W from scaling block (NT scaling matrix)
                        let w_raw = DMatrix::from_row_slice(n_psd, n_psd, w_factor);
                        let w_mat = 0.5 * (&w_raw + w_raw.transpose());

                        // W^{1/2}, W^{-1/2} via eigendecomposition
                        let eig_w = SymmetricEigen::new(w_mat);
                        let q_w = &eig_w.eigenvectors;
                        let d_w = &eig_w.eigenvalues;

                        let d_sqrt = d_w.map(|x| x.max(1e-30).sqrt());
                        let d_inv_sqrt = d_w.map(|x| 1.0 / x.max(1e-30).sqrt());
                        let q_w_t = q_w.transpose();

                        let w_half = q_w * DMatrix::from_diagonal(&d_sqrt) * &q_w_t;
                        let w_half_inv = q_w * DMatrix::from_diagonal(&d_inv_sqrt) * &q_w_t;

                        // λ = W^{1/2} Z W^{1/2}
                        let lambda_raw = &w_half * &z_mat * &w_half;
                        let lambda = 0.5 * (&lambda_raw + lambda_raw.transpose());
                        // A = W^{-1/2} dS_aff W^{-1/2}
                        let a = &w_half_inv * &ds_aff_mat * &w_half_inv;
                        // B = W^{1/2} dZ_aff W^{1/2}
                        let b = &w_half * &dz_aff_mat * &w_half;

                        // η = (AB + BA) / 2 (Jordan product)
                        let eta = (&a * &b + &b * &a) * 0.5;

                        // v = λ² + η - σμ I
                        let mut v = &lambda * &lambda + eta;
                        for i in 0..n_psd {
                            v[(i, i)] -= target_mu;
                        }

                        // Solve λ U + U λ = 2v using eigendecomposition of λ (Sylvester equation)
                        let eig_l = SymmetricEigen::new(lambda);
                        let q_l = &eig_l.eigenvectors;
                        let d_l = &eig_l.eigenvalues;
                        let q_l_t = q_l.transpose();

                        let v_hat = &q_l_t * &v * q_l;
                        let mut u_hat = DMatrix::zeros(n_psd, n_psd);
                        for i in 0..n_psd {
                            for j in 0..n_psd {
                                let denom = d_l[i] + d_l[j];
                                // denom should be > 0 if λ is PD; guard anyway
                                u_hat[(i, j)] = if denom > 1e-30 {
                                    2.0 * v_hat[(i, j)] / denom
                                } else {
                                    0.0
                                };
                            }
                        }
                        let u = q_l * u_hat * &q_l_t;

                        // d_s_comb = W^{1/2} U W^{1/2}
                        let ds_comb_mat = &w_half * u * &w_half;

                        // Back to svec
                        mat_to_svec(&ds_comb_mat, &mut ws.d_s_comb[offset..offset + dim]);
                    } else {
                        // Fallback: pure centering (no Mehrotra correction)
                        for i in 0..dim {
                            ws.d_s_comb[offset + i] = state.s[offset + i];
                        }
                        // Subtract σμ from diagonal elements if we can detect n
                        // This is a simple fallback; the structured path above is preferred
                    }
                } else if is_soc {
                    if let ScalingBlock::SocStructured { w } = &ws.scaling[cone_idx] {
                        let z_slice = &state.z[offset..offset + dim];
                        let ds_aff_slice = &ws.ds_aff[offset..offset + dim];
                        let dz_aff_slice = &ws.dz_aff[offset..offset + dim];

                        let scratch = &mut ws.soc_scratch;
                        let w_half = &mut scratch.w_half[..dim];
                        let w_half_inv = &mut scratch.w_half_inv[..dim];
                        let lambda = &mut scratch.lambda[..dim];
                        let w_inv_ds = &mut scratch.w_inv_ds[..dim];
                        let w_dz = &mut scratch.w_dz[..dim];
                        let eta = &mut scratch.eta[..dim];
                        let lambda_sq = &mut scratch.lambda_sq[..dim];
                        let v = &mut scratch.v[..dim];
                        let u = &mut scratch.u_vec[..dim];
                        let d_s_block = &mut scratch.d_s_block[..dim];
                        let e1 = &mut scratch.e1[..dim];
                        let e2 = &mut scratch.e2[..dim];
                        let w_circ_y = &mut scratch.w_circ_y[..dim];
                        let w_circ_w = &mut scratch.w_circ_w[..dim];
                        let temp = &mut scratch.temp[..dim];
                        let w2_circ_y = &mut scratch.w2_circ_y[..dim];

                        jordan_sqrt_in_place(w, w_half, e1, e2);
                        jordan_inv_in_place(w_half, w_half_inv, e1, e2);

                        quad_rep_in_place(w_half, z_slice, lambda, w_circ_y, w_circ_w, temp, w2_circ_y);
                        quad_rep_in_place(w_half_inv, ds_aff_slice, w_inv_ds, w_circ_y, w_circ_w, temp, w2_circ_y);
                        quad_rep_in_place(w_half, dz_aff_slice, w_dz, w_circ_y, w_circ_w, temp, w2_circ_y);

                        jordan_product_in_place(w_inv_ds, w_dz, eta);
                        jordan_product_in_place(lambda, lambda, lambda_sq);

                        v[0] = lambda_sq[0] + eta[0] - target_mu;
                        for i in 1..dim {
                            v[i] = lambda_sq[i] + eta[i];
                        }

                        jordan_solve_in_place(lambda, v, u, e1, e2);
                        quad_rep_in_place(w_half, u, d_s_block, w_circ_y, w_circ_w, temp, w2_circ_y);

                        ws.d_s_comb[offset..offset + dim].copy_from_slice(d_s_block);
                    } else {
                        // Fallback: diagonal correction with bounded Mehrotra term
                        for i in offset..offset + dim {
                            let s_i = state.s[i];
                            let z_i = state.z[i];
                            let mu_i = s_i * z_i;
                            // FIX: Handle negative z (for nonsymmetric cones)
                            let z_safe = if z_i.abs() < 1e-14 {
                                1e-14 * z_i.signum()
                            } else {
                                z_i
                            };

                            // Bound the Mehrotra correction to prevent numerical blow-up
                            let ds_dz = ws.ds_aff[i] * ws.dz_aff[i];
                            let correction_bound = mu_i.abs().max(target_mu * 0.1);
                            let ds_dz_bounded = ds_dz.clamp(-correction_bound, correction_bound);

                            let w_base = mu_i + ds_dz_bounded;
                            ws.d_s_comb[i] = (w_base - target_mu) / z_safe;
                        }
                    }
                } else {
                    // Check if this is a nonsymmetric cone (Dense3x3 = Exp/Pow)
                    let is_nonsym = matches!(ws.scaling[cone_idx], ScalingBlock::Dense3x3 { .. });

                    if is_nonsym {
                        // For nonsymmetric cones (Exp/Pow), use barrier-based complementarity
                        // Complementarity is: s + μ ∇f^*(z) ≈ 0
                        // So the corrector shift is: d_s = s + σ μ ∇f^*(z)

                        // Process each 3D block
                        for block in 0..(dim / 3) {
                            let block_offset = offset + 3 * block;
                            let s_block = [
                                state.s[block_offset],
                                state.s[block_offset + 1],
                                state.s[block_offset + 2],
                            ];
                            let z_block = [
                                state.z[block_offset],
                                state.z[block_offset + 1],
                                state.z[block_offset + 2],
                            ];

                            // Compute ∇f^*(z) via dual map for this exp cone block
                            // The dual map solves ∇f(x) + z = 0, then ∇f^*(z) = -x
                            let mut x = [0.0; 3];
                            let mut h_star = [0.0; 9];
                            exp_dual_map_block(&z_block, &mut x, &mut h_star);
                            let grad_fstar = [-x[0], -x[1], -x[2]];

                            // Extract affine directions for this block
                            let ds_aff_block = [
                                ws.ds_aff[block_offset],
                                ws.ds_aff[block_offset + 1],
                                ws.ds_aff[block_offset + 2],
                            ];
                            let dz_aff_block = [
                                ws.dz_aff[block_offset],
                                ws.dz_aff[block_offset + 1],
                                ws.dz_aff[block_offset + 2],
                            ];

                            // Compute third-order correction η
                            let eta = exp_third_order_correction(
                                &z_block,
                                &ds_aff_block,
                                &dz_aff_block,
                                &x,
                                &h_star,
                            );

                            // Barrier-based corrector with third-order correction:
                            // d_s = s + σ μ ∇f^*(z) + η
                            for j in 0..3 {
                                let i = block_offset + j;
                                ws.d_s_comb[i] = s_block[j] + sigma * target_mu * grad_fstar[j] + eta[j];
                            }

                            // Diagnostic logging at trace level (MINIX_VERBOSE=4)
                            if trace_enabled() {
                                eprintln!("Exp cone block {} corrector:", block);
                                eprintln!("  s = {:?}", s_block);
                                eprintln!("  z = {:?}", z_block);
                                eprintln!("  ∇f^*(z) = {:?}", grad_fstar);
                                eprintln!("  sigma = {:.3e}, mu = {:.3e}", sigma, target_mu);
                                eprintln!("  d_s_comb = [{:.3e}, {:.3e}, {:.3e}]",
                                    ws.d_s_comb[block_offset],
                                    ws.d_s_comb[block_offset + 1],
                                    ws.d_s_comb[block_offset + 2]
                                );
                            }
                        }
                    } else {
                        // Mehrotra correction for NonNeg cones
                        for i in offset..offset + dim {
                            let s_i = state.s[i];
                            let z_i = state.z[i];
                            let mu_i = s_i * z_i;
                            let z_safe = z_i.max(1e-14);

                            // Mehrotra correction term with bounding
                            let ds_dz = ws.ds_aff[i] * ws.dz_aff[i];
                            let correction_bound = mu_i.abs().max(target_mu * 0.1);
                            let ds_dz_bounded = ds_dz.clamp(-correction_bound, correction_bound);

                            // MCC delta if present
                            let delta = if is_nonneg && has_mcc { ws.mcc_delta[i] } else { 0.0 };

                            let w_base = mu_i + ds_dz_bounded;
                            ws.d_s_comb[i] = (w_base - target_mu - delta) / z_safe;
                        }
                    }
                }

                offset += dim;
            }

            for i in 0..m {
                ws.rhs_z[i] = ws.d_s_comb[i] - feas_weight * residuals.r_z[i];
            }

            kkt.solve_refined(
                &factor,
                &ws.rhs_x,
                &ws.rhs_z,
                &mut ws.dx,
                &mut ws.dz,
                refine_iters,
            );

            let d_tau_corr = feas_weight * residuals.r_tau;

            let dot_mul_p_xi_q_dx: f64 = ws
                .mul_p_xi_q
                .iter()
                .zip(ws.dx.iter())
                .map(|(a, b)| a * b)
                .sum();
            let dot_b_dz: f64 = prob.b.iter().zip(ws.dz.iter()).map(|(a, b)| a * b).sum();
            let numerator_corr = d_tau_corr - d_kappa_corr / state.tau + dot_mul_p_xi_q_dx + dot_b_dz;

            dtau = compute_dtau(numerator_corr, denominator, state.tau, denom_scale)
                .map_err(|e| format!("corrector dtau failed: {}", e))?;

            apply_tau_direction(&mut ws.dx, &mut ws.dz, dtau, &ws.dx2, &ws.dz2);

            let mut offset = 0;
            for (cone_idx, cone) in cones.iter().enumerate() {
                let dim = cone.dim();
                if dim == 0 {
                    continue;
                }

                if cone.barrier_degree() == 0 {
                    for i in offset..offset + dim {
                        ws.ds[i] = 0.0;
                    }
                } else {
                    if let ScalingBlock::SocStructured { w } = &ws.scaling[cone_idx] {
                        let scratch = &mut ws.soc_scratch;
                        let w_circ_y = &mut scratch.w_circ_y[..dim];
                        let w_circ_w = &mut scratch.w_circ_w[..dim];
                        let temp = &mut scratch.temp[..dim];
                        let w2_circ_y = &mut scratch.w2_circ_y[..dim];
                        let h_dz = &mut scratch.h_dz[..dim];
                        quad_rep_in_place(w, &ws.dz[offset..offset + dim], h_dz, w_circ_y, w_circ_w, temp, w2_circ_y);
                        for i in 0..dim {
                            ws.ds[offset + i] = -ws.d_s_comb[offset + i] - h_dz[i];
                        }
                    } else {
                        let dz_slice = &ws.dz[offset..offset + dim];
                        let ds_slice = &mut ws.ds[offset..offset + dim];
                        ws.scaling[cone_idx].apply(dz_slice, ds_slice);
                        for i in 0..dim {
                            ds_slice[i] = -ws.d_s_comb[offset + i] - ds_slice[i];
                        }
                    }
                }

                offset += dim;
            }

            if corr_iter < settings.mcc_iters {
                has_mcc = clamp_complementarity_nonneg_in_place(
                    state,
                    &ws.ds,
                    &ws.dz,
                    cones,
                    settings.centrality_beta,
                    settings.centrality_gamma,
                    mu,
                    &mut ws.mcc_delta,
                );
                if !has_mcc {
                    break;
                }
            }
        }

        let tau_old = state.tau;
        dkappa = -(d_kappa_corr + state.kappa * dtau) / tau_old;

        alpha_sz = compute_step_size(&state.s, &ws.ds, &state.z, &ws.dz, cones, 1.0);
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

        alpha = (0.99 * alpha).min(1.0);
        let alpha_pre_prox = alpha;

        // Proximity-based step size reduction (experimental)
        // This helps keep iterates close to the central path, reducing iteration count
        if settings.use_proximity_step_control {
            alpha = apply_proximity_step_control(
                state,
                &ws.ds,
                &ws.dz,
                dtau,
                dkappa,
                cones,
                barrier_degree,
                alpha,
                0.95,  // proximity threshold
            );
        }
        let alpha_post_prox = alpha;
        alpha_pre_ls = alpha;

        if settings.line_search_max_iters > 0
            && settings.centrality_gamma > settings.centrality_beta
            && settings.centrality_beta > 0.0
        {
            let mut ls_reported = false;
            for _ in 0..settings.line_search_max_iters {
                if centrality_ok_nonneg_trial(
                    state,
                    &ws.ds,
                    &ws.dz,
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
                        &ws.ds,
                        &ws.dz,
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
                // Minimum alpha floor to prevent complete stalling
                // If we hit this floor, accept the step and let sigma adjustment handle centering
                if alpha < 1e-4 {
                    alpha = 1e-4;
                    break;
                }
            }
        }

        // Exp cone central neighborhood check (P0.5)
        // Backtrack if the step would violate the central neighborhood condition
        // Enabled at trace level (MINIX_VERBOSE=4)
        if trace_enabled() {
            let theta = 0.3; // centrality parameter (0.1 to 0.5 typical)
            let mut offset = 0usize;
            let max_backtrack = 10;

            for _ in 0..max_backtrack {
                let mut central_ok = true;

                for cone in cones.iter() {
                    let dim = cone.dim();
                    if dim == 0 {
                        offset += dim;
                        continue;
                    }

                    // Check if this is an exp cone (3D blocks)
                    if (&**cone as &dyn std::any::Any).downcast_ref::<ExpCone>().is_some() {
                        // Check each 3D block
                        for block in 0..(dim / 3) {
                            let block_offset = offset + 3 * block;
                            let s_trial = [
                                state.s[block_offset] + alpha * ws.ds[block_offset],
                                state.s[block_offset + 1] + alpha * ws.ds[block_offset + 1],
                                state.s[block_offset + 2] + alpha * ws.ds[block_offset + 2],
                            ];
                            let z_trial = [
                                state.z[block_offset] + alpha * ws.dz[block_offset],
                                state.z[block_offset + 1] + alpha * ws.dz[block_offset + 1],
                                state.z[block_offset + 2] + alpha * ws.dz[block_offset + 2],
                            ];

                            if !exp_central_ok(&s_trial, &z_trial, target_mu, theta) {
                                central_ok = false;
                                if diagnostics_enabled() {
                                    eprintln!(
                                        "exp central check fail: block={} alpha={:.3e} theta={:.2}",
                                        block, alpha, theta
                                    );
                                }
                                break;
                            }
                        }
                    }

                    offset += dim;
                    if !central_ok {
                        break;
                    }
                }

                if central_ok {
                    break;
                }

                // Backtrack
                alpha *= 0.7;
                offset = 0; // reset for next iteration
            }
        }

        let alpha_post_ls = alpha;

        let alpha_limiter_sz = alpha_sz <= alpha_tau.min(alpha_kappa);
        let alpha_limiter_proximity = alpha_post_prox < 0.99 * alpha_pre_prox;
        let alpha_limiter_ls = alpha_post_ls < 0.99 * alpha_pre_ls;
        let alpha_limiter_centrality = alpha_limiter_proximity || alpha_limiter_ls;
        let centrality_emergency = alpha_limiter_centrality && alpha <= 1e-4;

        let alpha_stall = alpha < 1e-3
            && (alpha_limiter_sz || alpha_limiter_centrality)
            && (mu < 1e-6 || centrality_emergency);

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
            let mut bump_reg = (base_reg * 10.0).min(1e-4);
            if let Some(reg_cap) = psd_reg_cap_for_cones(cones) {
                bump_reg = bump_reg.min(reg_cap);
            }
            if bump_reg > 0.0 {
                let changed = kkt
                    .bump_static_reg(bump_reg)
                    .map_err(|e| format!("KKT reg bump failed: {}", e))?;
                if changed && settings.verbose {
                    eprintln!("bumped KKT static_reg to {:.2e} after alpha stall", bump_reg);
                }
            }
            sigma_eff = (sigma_eff + 0.2).min(sigma_cap);
            // If centrality checks crushed alpha, bump sigma more aggressively
            if alpha_limiter_centrality {
                sigma_eff = sigma_eff.max(0.3);
            }
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
        if let Some(diag) = nonneg_step_diagnostics(&state.s, &ws.ds, &state.z, &ws.dz, cones) {
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
        state.x[i] += alpha * ws.dx[i];
    }

    let mut offset = 0;
    for cone in cones {
        let dim = cone.dim();
        if dim > 0 {
            if cone.barrier_degree() == 0 {
                for i in offset..offset + dim {
                    state.s[i] = 0.0;
                    state.z[i] += alpha * ws.dz[i];
                }
            } else {
                for i in offset..offset + dim {
                    state.s[i] += alpha * ws.ds[i];
                    state.z[i] += alpha * ws.dz[i];
                }
            }
        }
        offset += dim;
    }

    // In direct mode, freeze tau=1 and kappa=0 (no homogeneous embedding updates)
    if settings.direct_mode {
        state.tau = 1.0;
        state.kappa = 0.0;
    } else {
        state.tau += alpha * dtau;
        state.kappa += alpha * dkappa;

        if state.kappa < 1e-12 {
            state.kappa = 1e-12;
        }
    }

    for i in 0..n {
        state.xi[i] = state.x[i] / state.tau;
    }

    let mu_new = compute_mu(state, barrier_degree);

    Ok(StepResult {
        alpha,
        alpha_sz,
        sigma: sigma_used,
        mu_new,
    })
}

/// Compute step size using fraction-to-boundary rule.
fn compute_step_size(
    s: &[f64],
    ds: &[f64],
    z: &[f64],
    dz: &[f64],
    cones: &[Box<dyn ConeKernel>],
    fraction: f64,
) -> f64 {
    let mut alpha = f64::INFINITY;
    let mut alpha_p_min = f64::INFINITY;
    let mut alpha_d_min = f64::INFINITY;
    let mut blocking_p_idx = None;
    let mut blocking_d_idx = None;
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

        if alpha_p.is_finite() && alpha_p < alpha_p_min {
            alpha_p_min = alpha_p.max(0.0);
            // Find which index is blocking in this cone
            for i in 0..dim {
                let idx = offset + i;
                if ds_slice[i] < 0.0 {
                    let ratio = -s_slice[i] / ds_slice[i];
                    if (ratio - alpha_p).abs() < 1e-10 * (ratio.abs() + 1.0) {
                        blocking_p_idx = Some((idx, s_slice[i], ds_slice[i]));
                        break;
                    }
                }
            }
        }

        if alpha_d.is_finite() && alpha_d < alpha_d_min {
            alpha_d_min = alpha_d.max(0.0);
            // Find which index is blocking in this cone
            for i in 0..dim {
                let idx = offset + i;
                if dz_slice[i] < 0.0 {
                    let ratio = -z_slice[i] / dz_slice[i];
                    if (ratio - alpha_d).abs() < 1e-10 * (ratio.abs() + 1.0) {
                        blocking_d_idx = Some((idx, z_slice[i], dz_slice[i]));
                        break;
                    }
                }
            }
        }

        alpha = alpha.min(alpha_p_min).min(alpha_d_min);

        if alpha == 0.0 {
            break;
        }

        offset += dim;
    }

    let alpha_final = if alpha.is_finite() {
        (fraction * alpha).min(1.0)
    } else {
        1.0
    };

    // Log blocking info when step size is very small
    if diagnostics_enabled() && alpha_final < 1e-8 {
        if let Some((idx, s, ds)) = blocking_p_idx {
            eprintln!(
                "  BLOCK primal: idx={} s={:.3e} ds={:.3e} alpha_p_raw={:.3e} would_be={:.3e}",
                idx, s, ds, alpha_p_min, s + alpha_p_min * ds
            );
        }
        if let Some((idx, z, dz)) = blocking_d_idx {
            eprintln!(
                "  BLOCK dual: idx={} z={:.3e} dz={:.3e} alpha_d_raw={:.3e} would_be={:.3e}",
                idx, z, dz, alpha_d_min, z + alpha_d_min * dz
            );
        }
    }

    alpha_final
}

/// Compute μ_aff = (s_aff · z_aff + τ_aff κ_aff) / (ν + 1) after affine step.
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

fn compute_centering_parameter(
    alpha_aff: f64,
    mu: f64,
    mu_aff: f64,
    barrier_degree: usize,
) -> f64 {
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

/// Adaptive centering parameter that reduces centering when close to convergence.
///
/// This allows the solver to take more aggressive steps (less centering) when
/// residuals and complementarity gap are small, speeding up convergence.
fn compute_centering_parameter_adaptive(
    alpha_aff: f64,
    mu: f64,
    mu_aff: f64,
    barrier_degree: usize,
    residuals: &HsdeResiduals,
) -> f64 {
    if barrier_degree == 0 {
        return 0.0;
    }

    // Base centering parameter (Mehrotra's formula)
    let sigma_base = if mu_aff.is_finite() && mu_aff > 0.0 && mu.is_finite() && mu > 0.0 {
        let ratio = (mu_aff / mu).max(0.0);
        ratio.powi(3)
    } else {
        (1.0 - alpha_aff).powi(3)
    };

    // Adaptive sigma_min based on progress
    // When close to convergence (small mu and small residuals), use smaller sigma_min
    // to allow less aggressive centering
    let r_x_norm = residuals.r_x.iter().map(|&x| x.abs()).fold(0.0_f64, f64::max);
    let r_z_norm = residuals.r_z.iter().map(|&x| x.abs()).fold(0.0_f64, f64::max);
    let res_norm = r_x_norm.max(r_z_norm).max(residuals.r_tau.abs());

    // Compute adaptive sigma_min:
    // - Far from convergence (res > 1e-4 or mu > 1e-4): sigma_min = 1e-3 (standard)
    // - Close to convergence (res < 1e-6 and mu < 1e-6): sigma_min = 1e-5 (aggressive)
    // - In between: interpolate
    let sigma_min = if res_norm > 1e-4 || mu > 1e-4 {
        1e-3  // Standard centering far from optimum
    } else if res_norm < 1e-6 && mu < 1e-6 {
        1e-5  // Aggressive (less centering) near optimum
    } else {
        // Interpolate between 1e-5 and 1e-3 based on progress
        let progress = ((res_norm.max(mu) - 1e-6) / (1e-4 - 1e-6)).clamp(0.0, 1.0);
        1e-5 + progress * (1e-3 - 1e-5)
    };

    let sigma_max = 0.999;
    sigma_base.max(sigma_min).min(sigma_max)
}

/// Apply proximity-based step size control to keep iterates close to central path.
///
/// This function reduces the step size if the trial iterate would have a large
/// proximity metric (neighborhood parameter), which indicates being far from
/// the central path.
///
/// The proximity metric used is:
///   proximity = ||s ⊙ z - μe||_∞ / μ
///
/// If proximity > threshold, we reduce alpha until proximity <= threshold.
fn apply_proximity_step_control(
    state: &HsdeState,
    ds: &[f64],
    dz: &[f64],
    dtau: f64,
    dkappa: f64,
    cones: &[Box<dyn ConeKernel>],
    barrier_degree: usize,
    alpha_init: f64,
    proximity_threshold: f64,
) -> f64 {
    let mut alpha = alpha_init;
    let backtrack_factor = 0.8;
    let max_backtrack = 10;

    for _ in 0..max_backtrack {
        // Compute trial iterate
        let tau_trial = state.tau + alpha * dtau;
        let kappa_trial = state.kappa + alpha * dkappa;
        let mut s_dot_z_trial = 0.0;

        let mut offset = 0;
        for cone in cones.iter() {
            let dim = cone.dim();
            if dim == 0 || cone.barrier_degree() == 0 {
                offset += dim;
                continue;
            }

            for i in 0..dim {
                let idx = offset + i;
                let s_trial = state.s[idx] + alpha * ds[idx];
                let z_trial = state.z[idx] + alpha * dz[idx];
                s_dot_z_trial += s_trial * z_trial;
            }

            offset += dim;
        }

        // Compute trial mu
        let mu_trial = (s_dot_z_trial + tau_trial * kappa_trial) / (barrier_degree as f64 + 1.0);

        if !mu_trial.is_finite() || mu_trial <= 0.0 {
            alpha *= backtrack_factor;
            continue;
        }

        // Compute proximity (infinity norm of (s⊙z - μe) / μ)
        let mut proximity = 0.0_f64;
        offset = 0;

        for cone in cones.iter() {
            let dim = cone.dim();
            if dim == 0 || cone.barrier_degree() == 0 {
                offset += dim;
                continue;
            }

            let is_soc = (cone.as_ref() as &dyn Any).is::<SocCone>();
            if is_soc {
                // For SOC cones, measure proximity using NT-scaled complementarity eigenvalues
                let s0 = state.s[offset] + alpha * ds[offset];
                let z0 = state.z[offset] + alpha * dz[offset];

                let mut s_norm_sq = 0.0;
                let mut z_norm_sq = 0.0;
                for i in 1..dim {
                    let si = state.s[offset + i] + alpha * ds[offset + i];
                    let zi = state.z[offset + i] + alpha * dz[offset + i];
                    s_norm_sq += si * si;
                    z_norm_sq += zi * zi;
                }
                let s_norm = s_norm_sq.sqrt();
                let z_norm = z_norm_sq.sqrt();

                let s_hi = s0 + s_norm;
                let s_lo = if s0 <= s_norm { s0 - s_norm } else {
                    let d = s0 + s_norm; if d == 0.0 { 0.0 } else { s0.mul_add(s0, -s_norm_sq) / d }
                };
                let z_hi = z0 + z_norm;
                let z_lo = if z0 <= z_norm { z0 - z_norm } else {
                    let d = z0 + z_norm; if d == 0.0 { 0.0 } else { z0.mul_add(z0, -z_norm_sq) / d }
                };

                if s_lo > 0.0 && z_lo > 0.0 {
                    let comp_hi = (s_hi * z_hi).sqrt();
                    let comp_lo = (s_lo * z_lo).sqrt();
                    let deviation = (comp_hi - mu_trial).abs().max((comp_lo - mu_trial).abs()) / mu_trial;
                    proximity = proximity.max(deviation);
                } else {
                    // Not interior - max deviation
                    proximity = f64::MAX;
                }

                offset += dim;
                continue;
            }

            for i in 0..dim {
                let idx = offset + i;
                let s_trial = state.s[idx] + alpha * ds[idx];
                let z_trial = state.z[idx] + alpha * dz[idx];
                let complementarity = s_trial * z_trial;
                let deviation = (complementarity - mu_trial).abs() / mu_trial;
                proximity = proximity.max(deviation);
            }

            offset += dim;
        }

        if proximity <= proximity_threshold {
            return alpha;
        }

        // Reduce step size and try again
        alpha *= backtrack_factor;
    }

    // If we exhausted backtracks, return the reduced alpha
    alpha
}
