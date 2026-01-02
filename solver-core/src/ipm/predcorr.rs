//! Predictor-corrector steps for HSDE interior point method.
//!
//! The predictor-corrector algorithm has two phases per iteration:
//! 1. **Affine step**: Solve KKT system with σ = 0 (pure Newton step)
//! 2. **Combined step**: Solve with Mehrotra correction (adds centering)
//!
//! This implementation follows §7 of the design doc.

use super::hsde::{HsdeState, HsdeResiduals, compute_mu};
use crate::cones::ConeKernel;
use crate::linalg::kkt::KktSolver;
use crate::scaling::{ScalingBlock, nt};
use crate::problem::ProblemData;

/// Predictor-corrector step result.
#[derive(Debug)]
pub struct StepResult {
    /// Step size taken
    pub alpha: f64,

    /// Centering parameter used
    pub sigma: f64,

    /// New barrier parameter after step
    pub mu_new: f64,
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
    state: &mut HsdeState,
    residuals: &HsdeResiduals,
    cones: &[Box<dyn ConeKernel>],
    mu: f64,
    barrier_degree: usize,
) -> Result<StepResult, String> {
    let n = prob.num_vars();
    let m = prob.num_constraints();

    // ======================================================================
    // Step 1: Compute NT scaling for all cones
    // ======================================================================
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
        let scale = nt::compute_nt_scaling(s, z, cone.as_ref())
            .unwrap_or_else(|_| {
                // Fallback to simple diagonal scaling if NT fails
                let d: Vec<f64> = s.iter().zip(z.iter())
                    .map(|(si, zi)| (si / zi).max(1e-8).sqrt())
                    .collect();
                ScalingBlock::Diagonal { d }
            });

        scaling.push(scale);
        offset += dim;
    }

    // ======================================================================
    // Step 2: Factor KKT system
    // ======================================================================
    let factor = kkt.factor(prob.P.as_ref(), &prob.A, &scaling)
        .map_err(|e| format!("KKT factorization failed: {}", e))?;

    // ======================================================================
    // Step 3: Affine step (σ = 0)
    // ======================================================================
    let mut dx_aff = vec![0.0; n];
    let mut dz_aff = vec![0.0; m];
    let dtau_aff;

    // Affine RHS: negative residuals (Newton direction)
    let rhs_x: Vec<f64> = residuals.r_x.iter().map(|&r| -r).collect();
    let rhs_z: Vec<f64> = residuals.r_z.iter().map(|&r| -r).collect();

    kkt.solve(&factor, &rhs_x, &rhs_z, &mut dx_aff, &mut dz_aff);

    // Compute dtau via two-solve Schur complement strategy (design doc §5.4.1)
    // This replaces the old heuristic dtau = -(q'dx + b'dz)

    // First, compute mul_p_xi = P*ξ (if P exists)
    let mut mul_p_xi = vec![0.0; n];
    if let Some(ref p) = prob.P {
        // P is symmetric upper triangle, do symmetric matvec
        for col in 0..n {
            if let Some(col_view) = p.outer_view(col) {
                for (row, &val) in col_view.iter() {
                    if row == col {
                        mul_p_xi[row] += val * state.xi[col];
                    } else {
                        mul_p_xi[row] += val * state.xi[col];
                        mul_p_xi[col] += val * state.xi[row];
                    }
                }
            }
        }
    }

    // Compute mul_p_xi_q = 2*P*ξ + q
    let mul_p_xi_q: Vec<f64> = mul_p_xi.iter()
        .zip(prob.q.iter())
        .map(|(pxi, qi)| 2.0 * pxi + qi)
        .collect();

    // Second solve: K [Δx₂, Δz₂] = [-q, b]
    let mut dx2 = vec![0.0; n];
    let mut dz2 = vec![0.0; m];
    let rhs_x2: Vec<f64> = prob.q.iter().map(|&qi| -qi).collect();
    let rhs_z2 = prob.b.clone();

    kkt.solve(&factor, &rhs_x2, &rhs_z2, &mut dx2, &mut dz2);

    // Compute dtau via Schur complement formula (design doc §5.4.1)
    // Numerator: d_τ - d_κ/τ + (2Pξ+q)ᵀΔx₁ + bᵀΔz₁
    // Denominator: κ/τ + ξᵀPξ - (2Pξ+q)ᵀΔx₂ - bᵀΔz₂
    //
    // Note: For LPs (P=None), we use higher regularization (≥1e-6) to stabilize
    // the second solve. This is set in ipm/mod.rs.

    // d_tau = -r_tau (affine direction for tau)
    let d_tau = -residuals.r_tau;

    // d_kappa: from complementarity τκ ≈ μ, we want κ to track μ/τ
    // For affine step (σ=0), d_kappa = -state.kappa (drive to 0)
    let d_kappa = -state.kappa;

    let dot_mul_p_xi_q_dx1: f64 = mul_p_xi_q.iter().zip(dx_aff.iter()).map(|(a, b)| a * b).sum();
    let dot_b_dz1: f64 = prob.b.iter().zip(dz_aff.iter()).map(|(a, b)| a * b).sum();
    let numerator = d_tau - d_kappa / state.tau + dot_mul_p_xi_q_dx1 + dot_b_dz1;

    let dot_xi_mul_p_xi: f64 = state.xi.iter().zip(mul_p_xi.iter()).map(|(a, b)| a * b).sum();
    let dot_mul_p_xi_q_dx2: f64 = mul_p_xi_q.iter().zip(dx2.iter()).map(|(a, b)| a * b).sum();
    let dot_b_dz2: f64 = prob.b.iter().zip(dz2.iter()).map(|(a, b)| a * b).sum();
    let denominator = state.kappa / state.tau + dot_xi_mul_p_xi - dot_mul_p_xi_q_dx2 - dot_b_dz2;

    // Compute dtau (with safeguards)
    // For problems with mixed cones (barrier_degree > 0 and Zero cones present),
    // the Schur complement can be unstable. Keep tau fixed for stability.
    let has_zero_cone = cones.iter().any(|c| c.barrier_degree() == 0 && c.dim() > 0);
    let has_barrier_cone = barrier_degree > 0;
    let is_mixed = has_zero_cone && has_barrier_cone;

    // Compute dtau with safeguards
    // For mixed cone problems, the Schur complement is unstable even with the correct
    // ds formula. Keep tau fixed for now - the algorithm converges correctly but slowly.
    dtau_aff = if is_mixed {
        0.0
    } else if denominator.abs() > 1e-8 {
        let raw_dtau = numerator / denominator;
        let max_dtau = 2.0 * state.tau;
        raw_dtau.max(-max_dtau).min(max_dtau)
    } else {
        0.0
    };

    // Debug output disabled by default
    // #[cfg(debug_assertions)]
    // eprintln!("  [dtau_aff] = {:.6e}", dtau_aff);

    // Compute ds_aff from primal constraint Newton step:
    // A×dx + ds - b×dτ = -r_z
    // With dτ = dtau_aff: ds = -r_z - A×dx + b×dtau_aff
    let mut ds_aff = vec![0.0; m];
    for i in 0..m {
        ds_aff[i] = -residuals.r_z[i] + prob.b[i] * dtau_aff;
    }
    for (val, (row, col)) in prob.A.iter() {
        ds_aff[row] -= (*val) * dx_aff[col];
    }

    // Compute affine step size (step-to-boundary)
    let alpha_aff = compute_step_size(&state.s, &ds_aff, &state.z, &dz_aff, cones, 1.0);

    // ======================================================================
    // Step 4: Compute centering parameter σ
    // ======================================================================
    let sigma = compute_centering_parameter_with_cones(
        alpha_aff, mu, &state.s, &ds_aff, &state.z, &dz_aff, cones, barrier_degree
    );


    // ======================================================================
    // Step 5: Combined corrector step
    // ======================================================================
    let mut dx = vec![0.0; n];
    let mut dz = vec![0.0; m];
    let dtau;

    // Build corrected RHS with Mehrotra correction (design doc §7.3.1)
    //
    // For the complementarity condition s ∘ z = μe, the Newton system gives:
    //   s ∘ dz + ds ∘ z = -s ∘ z + σμe - ds_aff ∘ dz_aff
    //
    // The Mehrotra correction term η = ds_aff ∘ dz_aff accounts for second-order effects.
    // For NonNeg cones, ∘ is elementwise multiplication.
    // For SOC cones, ∘ is the Jordan product.
    //
    // The RHS modification for the KKT system is:
    //   rhs_z_corr[i] = rhs_z[i] + (σμ - ds_aff[i] * dz_aff[i]) / s[i]  (for NonNeg)
    //
    let target_mu = sigma * mu;
    let mut rhs_z_corr = rhs_z.to_vec();

    // Apply Mehrotra correction per cone
    let mut offset = 0;
    for cone in cones {
        let dim = cone.dim();
        if dim == 0 {
            continue;
        }

        if cone.barrier_degree() == 0 {
            // Zero cone: no correction needed
            offset += dim;
            continue;
        }

        // For NonNeg cones: full Mehrotra correction
        // The Mehrotra correction is: (σμ - ds_aff_i * dz_aff_i) / s_i
        // This includes:
        // 1. Centering term: σμ / s_i (pushes toward s*z = σμ)
        // 2. Second-order correction: -ds_aff_i * dz_aff_i / s_i (accounts for curvature)
        for i in offset..offset + dim {
            let s_i = state.s[i].max(1e-12);  // Avoid division by zero
            // Full Mehrotra correction
            let correction = (target_mu - ds_aff[i] * dz_aff[i]) / s_i;
            rhs_z_corr[i] += correction;
        }

        offset += dim;
    }

    kkt.solve(&factor, &rhs_x, &rhs_z_corr, &mut dx, &mut dz);

    // Compute dtau for corrector step using same Schur complement formula
    // For corrector step, d_kappa incorporates centering: d_kappa = σμ/τ - κ
    let d_tau_corr = -residuals.r_tau;
    let d_kappa_corr = target_mu / state.tau - state.kappa;

    let dot_mul_p_xi_q_dx: f64 = mul_p_xi_q.iter().zip(dx.iter()).map(|(a, b)| a * b).sum();
    let dot_b_dz: f64 = prob.b.iter().zip(dz.iter()).map(|(a, b)| a * b).sum();
    let numerator_corr = d_tau_corr - d_kappa_corr / state.tau + dot_mul_p_xi_q_dx + dot_b_dz;

    // Denominator is the same as affine step (only depends on current state)
    // Apply same safeguards as affine step
    dtau = if is_mixed {
        0.0  // Keep tau fixed for mixed cone problems
    } else if denominator.abs() > 1e-8 {
        let raw_dtau = numerator_corr / denominator;
        let max_dtau = 2.0 * state.tau;
        raw_dtau.max(-max_dtau).min(max_dtau)
    } else {
        0.0
    };

    // Debug output disabled by default
    // #[cfg(debug_assertions)]
    // eprintln!("  [dtau_corr] = {:.6e}", dtau);

    // Compute ds from primal constraint Newton step:
    // A×dx + ds - b×dτ = -r_z
    // With dτ = dtau: ds = -r_z - A×dx + b×dtau
    let mut ds = vec![0.0; m];
    for i in 0..m {
        ds[i] = -residuals.r_z[i] + prob.b[i] * dtau;
    }
    for (val, (row, col)) in prob.A.iter() {
        ds[row] -= (*val) * dx[col];
    }

    // ======================================================================
    // Step 6: Compute step size with fraction-to-boundary
    // ======================================================================
    let mut alpha = compute_step_size(&state.s, &ds, &state.z, &dz, cones, 0.99);

    // Also check step-to-boundary for tau (must remain positive)
    if dtau < 0.0 {
        let alpha_tau = -state.tau / dtau;
        // Use standard fraction-to-boundary (0.99) for tau
        let conservative_alpha_tau = 0.99 * alpha_tau;
        if conservative_alpha_tau < alpha {
            alpha = conservative_alpha_tau;
        }
    }

    // Cap alpha at 1.0 (never take more than a full Newton step)
    alpha = alpha.min(1.0);

    // No need to check kappa step-to-boundary since we recompute it from mu/tau

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

    // Update κ to maintain complementarity τκ ≈ μ
    state.kappa = mu / state.tau;

    // Update ξ = x/τ for next iteration's Schur complement
    for i in 0..n {
        state.xi[i] = state.x[i] / state.tau;
    }

    // Compute new μ
    let mu_new = compute_mu(state, barrier_degree);

    Ok(StepResult {
        alpha,
        sigma,
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

/// Compute centering parameter σ using Mehrotra's heuristic.
///
/// σ = (μ_aff / μ)^3 where μ_aff = <s + α_aff Δs, z + α_aff Δz> / ν
///
/// For problems with only Zero cones (barrier_degree = 0), returns 0.
///
/// Note: Only barrier cone components contribute to μ_aff. Zero cone components
/// (where s=0) are excluded since they don't have a barrier.
fn compute_centering_parameter_with_cones(
    alpha_aff: f64,
    mu: f64,
    s: &[f64],
    ds: &[f64],
    z: &[f64],
    dz: &[f64],
    cones: &[Box<dyn ConeKernel>],
    barrier_degree: usize,
) -> f64 {
    // Special case: no barrier (only Zero cones)
    if barrier_degree == 0 {
        return 0.0;
    }

    // Compute μ after affine step, only for barrier cone components
    let mut dot_product = 0.0;
    let mut offset = 0;

    for cone in cones {
        let dim = cone.dim();
        if dim == 0 {
            continue;
        }

        // Skip Zero cones (they don't contribute to μ)
        if cone.barrier_degree() == 0 {
            offset += dim;
            continue;
        }

        for i in offset..offset + dim {
            let s_new = s[i] + alpha_aff * ds[i];
            let z_new = z[i] + alpha_aff * dz[i];
            dot_product += s_new * z_new;
        }

        offset += dim;
    }

    let mu_aff = dot_product / (barrier_degree as f64);

    // σ = (μ_aff / μ)^3
    let ratio = mu_aff / mu.max(1e-10);
    let sigma = ratio.powi(3);

    // Clamp to [0, 1]
    sigma.max(0.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cones::NonNegCone;

    #[test]
    fn test_compute_centering_parameter() {
        // Create NonNeg cones for testing (3-dimensional)
        let cones: Vec<Box<dyn ConeKernel>> = vec![Box::new(NonNegCone::new(3))];

        // Test that σ → 0 when affine step makes good progress
        let sigma = compute_centering_parameter_with_cones(
            0.99, // large alpha_aff (good progress)
            1.0,  // current mu
            &vec![1.0; 3],
            &vec![-0.5; 3], // negative direction (approaching boundary)
            &vec![1.0; 3],
            &vec![-0.5; 3],
            &cones,
            3,
        );
        assert!(sigma < 0.5, "σ should be small for good affine progress, got {}", sigma);

        // Test that σ → 1 when affine step makes poor progress
        let sigma = compute_centering_parameter_with_cones(
            0.01, // small alpha_aff (poor progress)
            1.0,
            &vec![1.0; 3],
            &vec![-0.5; 3],
            &vec![1.0; 3],
            &vec![-0.5; 3],
            &cones,
            3,
        );
        assert!(sigma > 0.9, "σ should be large for poor affine progress, got {}", sigma);
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
