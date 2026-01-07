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
use std::time::{Duration, Instant};

fn diagnostics_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("MINIX_DIAGNOSTICS")
            .map(|v| v != "0")
            .unwrap_or(false)
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

        // Compute NT scaling based on cone type - use upstream's improved error handling
        let scale = match nt::compute_nt_scaling(s, z, cone.as_ref()) {
            Ok(scale) => scale,
            Err(e) => {
                let s_block_min = min_slice(s);
                let z_block_min = min_slice(z);
                if is_nonneg {
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
    // Note: SOC adaptive regularization was removed as upstream removed diag_reg from SocStructured
    if extra_reg > 0.0 {
        for block in scaling.iter_mut() {
            match block {
                ScalingBlock::Diagonal { d } => {
                    for di in d.iter_mut() {
                        // Add regularization to H directly (H_reg = H + extra_reg * I)
                        *di += extra_reg;
                    }
                }
                _ => {}
            }
        }
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

    // Combined two-RHS solve for efficiency (upstream improvement)
    {
        let start = Instant::now();
        kkt.solve_two_rhs_refined_tagged(
            &factor,
            rhs_x_aff,
            rhs_z_aff,
            rhs_x2,
            rhs_z2,
            dx_aff,
            dz_aff,
            dx2,
            dz2,
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
                ScalingBlock::SocStructured { w } => {
                    // For SOC, H = P(w) (quadratic representation)
                    // ds = -s - P(w)*dz
                    // Use workspace buffer instead of allocating
                    let dz_slice = &dz_aff[offset..offset + dim];
                    let h_dz = &mut ws.soc_h_dz[..dim];
                    crate::scaling::nt::quad_rep_apply(w, dz_slice, h_dz);
                    for i in 0..dim {
                        ds_aff[offset + i] =
                            -state.s[offset + i] - h_dz[i];
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
    let mut feas_weight_floor = settings.feas_weight_floor.clamp(0.0, 1.0);
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
                        let delta = if cone_type == ConeType::NonNeg {
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

            // rhs_z = d_s - d_z (weighted feasibility residual) - use workspace buffer
            let rhs_z_comb = &mut ws.rhs_z_comb[..];
            for i in 0..m {
                rhs_z_comb[i] = d_s_comb[i] - feas_weight * residuals.r_z[i];
            }

            {
                let start = Instant::now();
                kkt.solve_refined(
                    &factor,
                    rhs_x_comb,
                    rhs_z_comb,
                    dx,
                    dz,
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
                        ScalingBlock::SocStructured { w } => {
                            // For SOC, H = P(w) (quadratic representation)
                            // ds = -d_s - P(w)*dz
                            // Use workspace buffer instead of allocating
                            let dz_slice = &dz[offset..offset + dim];
                            let h_dz = &mut ws.soc_h_dz[..dim];
                            crate::scaling::nt::quad_rep_apply(w, dz_slice, h_dz);
                            for i in 0..dim {
                                ds[offset + i] =
                                    -d_s_comb[offset + i] - h_dz[i];
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
            let mut ls_reported = false;
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
                    0.0,   // soc_centrality_beta (disabled - SOC centrality is harmful)
                    100.0, // soc_centrality_gamma (disabled)
                    barrier_degree,
                    alpha,
                    false, // enable_soc_centrality (disabled - causes BOYD1/BOYD2 failures)
                    false, // soc_centrality_use_upper (disabled)
                    false, // soc_centrality_use_jordan (disabled)
                    0.0,   // soc_centrality_mu_threshold (disabled)
                    &mut ws.cent_s_trial,
                    &mut ws.cent_z_trial,
                    &mut ws.cent_w,
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
        alpha_sz,
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
