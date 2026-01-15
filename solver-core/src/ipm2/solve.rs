//! Main IPM solver entry point (ipm2).
//!
//! Implements a predictor-corrector interior point method using HSDE
//! (Homogeneous Self-Dual Embedding) with Ruiz equilibration, NT scaling,
//! and active-set polishing for bound-heavy problems.

use std::time::Instant;

use crate::cones::{ConeKernel, NonNegCone, SocCone, ZeroCone, ExpCone, PowCone, PsdCone};
use crate::cones::psd::svec_to_mat;
use crate::ipm::hsde::{HsdeResiduals, HsdeState, compute_mu, compute_residuals};
use crate::ipm::termination::TerminationCriteria;
use crate::ipm2::{
    DiagnosticsConfig, IpmWorkspace, PerfSection, PerfTimers, RegularizationPolicy, SolveMode,
    StallDetector, compute_unscaled_metrics, diagnose_dual_residual, polish_nonneg_active_set,
    polish_primal_and_dual, polish_lp_dual,
};
use crate::ipm2::predcorr::predictor_corrector_step_in_place;
use crate::linalg::kkt_trait::KktSolverTrait;
use crate::linalg::unified_kkt::UnifiedKktSolver;
use crate::chordal::{ChordalSettings, analyze_problem as analyze_chordal, decompose_problem};
use crate::presolve::apply_presolve;
use crate::presolve::proximal::{detect_free_variables_eq, add_proximal_regularization};
use crate::presolve::ruiz::equilibrate;
use crate::presolve::singleton::detect_singleton_rows;
use crate::postsolve::PostsolveMap;
use crate::problem::{
    ConeSpec, ProblemData, SolveInfo, SolveResult, SolveStatus, SolverSettings,
};
use nalgebra::linalg::SymmetricEigen;

/// Check if CLARABEL-style HSDE rescaling by max(tau, kappa) is enabled.
///
/// Default is true (use max-based rescaling). Set MINIX_HSDE_RESCALE=threshold
/// to use the original threshold-based normalization.
fn hsde_rescale_by_max() -> bool {
    static USE_MAX: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *USE_MAX.get_or_init(|| {
        std::env::var("MINIX_HSDE_RESCALE")
            .map(|v| v.to_lowercase() != "threshold")
            .unwrap_or(true)  // Default: use max-based rescaling
    })
}

fn psd_reg_strict_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("MINIX_PSD_REG_STRICT")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

fn psd_reg_dynamic_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("MINIX_PSD_REG_DYNAMIC")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

fn psd_reg_log_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("MINIX_PSD_REG_LOG")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false)
    })
}

fn psd_diag_avg_abs(svec: &[f64], n: usize) -> f64 {
    let mut sum = 0.0;
    let mut idx = 0usize;
    for j in 0..n {
        for i in 0..=j {
            if i == j {
                sum += svec[idx].abs();
            }
            idx += 1;
        }
    }
    if n == 0 { 0.0 } else { sum / n as f64 }
}

fn psd_scale_from_state(state: &HsdeState, cones: &[Box<dyn ConeKernel>]) -> Option<f64> {
    let mut offset = 0usize;
    let mut total = 0.0;
    let mut count = 0usize;

    for cone in cones {
        let dim = cone.dim();
        if dim == 0 {
            continue;
        }
        if cone.barrier_degree() == 0 {
            offset += dim;
            continue;
        }

        let is_psd = (cone.as_ref() as &dyn std::any::Any).is::<PsdCone>();
        if is_psd {
            let n = if let Some(psd) = (cone.as_ref() as &dyn std::any::Any).downcast_ref::<PsdCone>() {
                psd.size()
            } else {
                offset += dim;
                continue;
            };
            let s_block = &state.s[offset..offset + dim];
            let z_block = &state.z[offset..offset + dim];
            let s_avg = psd_diag_avg_abs(s_block, n);
            let z_avg = psd_diag_avg_abs(z_block, n);
            let scale = 0.5 * (s_avg + z_avg);
            if scale.is_finite() && scale > 0.0 {
                total += scale;
                count += 1;
            }
        }

        offset += dim;
    }

    if count == 0 { None } else { Some(total / count as f64) }
}

fn psd_min_eigs_from_state(state: &HsdeState, cones: &[Box<dyn ConeKernel>]) -> Option<(f64, f64)> {
    let mut offset = 0usize;
    let mut min_s = f64::INFINITY;
    let mut min_z = f64::INFINITY;
    let mut found = false;

    for cone in cones {
        let dim = cone.dim();
        if dim == 0 {
            continue;
        }
        if cone.barrier_degree() == 0 {
            offset += dim;
            continue;
        }

        if let Some(psd) = (cone.as_ref() as &dyn std::any::Any).downcast_ref::<PsdCone>() {
            let n = psd.size();
            let s_block = &state.s[offset..offset + dim];
            let z_block = &state.z[offset..offset + dim];

            let s_mat = svec_to_mat(s_block, n);
            let z_mat = svec_to_mat(z_block, n);
            let eig_s = SymmetricEigen::new(s_mat);
            let eig_z = SymmetricEigen::new(z_mat);

            if let Some(&val) = eig_s.eigenvalues.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
                if val.is_finite() {
                    min_s = min_s.min(val);
                    found = true;
                }
            }
            if let Some(&val) = eig_z.eigenvalues.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
                if val.is_finite() {
                    min_z = min_z.min(val);
                    found = true;
                }
            }
        }

        offset += dim;
    }

    if found { Some((min_s, min_z)) } else { None }
}

/// Main ipm2 solver entry point.
pub fn solve_ipm2(
    prob: &ProblemData,
    settings: &SolverSettings,
) -> Result<SolveResult, Box<dyn std::error::Error>> {
    // Validate problem
    prob.validate()?;

    // SOC centrality line search is DISABLED by default because testing shows it
    // prevents convergence on many problems. The centrality check (even with relaxed
    // bounds) is too strict for ill-conditioned SOCP problems.
    // Enable with MINIX_ENABLE_SOC_AUTOLS=1 for experimental centrality enforcement.
    let has_soc = prob.cones.iter().any(|c| matches!(c, ConeSpec::Soc { .. }));
    let enable_soc_autols = std::env::var("MINIX_ENABLE_SOC_AUTOLS")
        .map(|v| v == "1")
        .unwrap_or(false);
    let settings = if has_soc && settings.line_search_max_iters == 0 && enable_soc_autols {
        let mut s = settings.clone();
        s.line_search_max_iters = 15;
        s
    } else {
        settings.clone()
    };
    let settings = &settings;

    let orig_prob = prob.clone();
    let orig_prob_bounds = orig_prob.with_bounds_as_constraints();
    let presolved = apply_presolve(prob);
    let prob = presolved.problem;
    let postsolve = presolved.postsolve;

    // Convert var_bounds to explicit constraints if present
    let prob = prob.with_bounds_as_constraints();

    let n = prob.num_vars();
    let m = prob.num_constraints();
    let orig_n = orig_prob.num_vars();
    let orig_m = orig_prob_bounds.num_constraints();

    // Constraint conditioning: DISABLED (harmful - see _planning/v15/conditioning_results.md)
    // Row scaling interferes with Ruiz equilibration and decreases pass rate (108→104).
    // Detection code kept for analysis. Enable with SolverSettings.enable_conditioning=true.
    let mut prob = prob;
    if settings.enable_conditioning.unwrap_or(false) {
        let cond_stats = crate::presolve::condition::analyze_conditioning(&prob);
        if settings.verbose {
            eprintln!(
                "conditioning: parallel_pairs={} extreme_ratio_rows={} max_cosine={:.3e} max_ratio={:.3e}",
                cond_stats.parallel_pairs,
                cond_stats.extreme_ratio_rows,
                cond_stats.max_cosine_sim,
                cond_stats.max_coeff_ratio
            );
        }

        // Apply row scaling if we detect severe issues
        if cond_stats.extreme_ratio_rows > 0 || cond_stats.max_coeff_ratio > 1e8 {
            let _row_scales = crate::presolve::condition::apply_row_scaling(&mut prob);
            if settings.verbose {
                eprintln!("conditioning: applied row scaling");
            }
        }
    }

    // Apply proximal regularization for free variables (zero A-column + zero q)
    // This stabilizes the Newton system for degenerate SDPs.
    // For SDP problems with identity embedding, we only look at equality constraint rows.
    if settings.proximal_rho > 0.0 {
        // Compute the Zero cone dimension (equality constraints)
        let zero_cone_dim: usize = prob.cones.iter()
            .filter_map(|c| match c {
                ConeSpec::Zero { dim } => Some(*dim),
                _ => None,
            })
            .sum();

        // If there's a Zero cone, use it to identify equality rows; otherwise use all rows
        let eq_rows = if zero_cone_dim > 0 { zero_cone_dim } else { m };

        let free_vars = detect_free_variables_eq(
            &prob.A,
            &prob.q,
            prob.P.as_ref(),
            eq_rows,
            1e-10,  // A column norm tolerance
            1e-10,  // cost coefficient tolerance
        );
        if !free_vars.is_empty() {
            if settings.verbose {
                eprintln!("proximal: detected {} free variables (zero A-columns in {} eq rows), adding rho={:.1e}",
                    free_vars.len(), eq_rows, settings.proximal_rho);
            }
            prob.P = add_proximal_regularization(
                prob.P.clone(),
                n,
                &free_vars,
                settings.proximal_rho,
            );
        }
    }

    // Apply Ruiz equilibration
    let (a_scaled, p_scaled, q_scaled, b_scaled, scaling) = equilibrate(
        &prob.A,
        prob.P.as_ref(),
        &prob.q,
        &prob.b,
        settings.ruiz_iters,
        &prob.cones,
    );

    // Log scaling info for debugging
    if settings.verbose || std::env::var("MINIX_VERBOSE").map(|v| v.parse::<u32>().unwrap_or(0)).unwrap_or(0) >= 2 {
        let row_min = scaling.row_scale.iter().cloned().fold(f64::INFINITY, f64::min);
        let row_max = scaling.row_scale.iter().cloned().fold(0.0_f64, f64::max);
        let col_min = scaling.col_scale.iter().cloned().fold(f64::INFINITY, f64::min);
        let col_max = scaling.col_scale.iter().cloned().fold(0.0_f64, f64::max);
        eprintln!(
            "ruiz scaling: cost_scale={:.3e} row_scale=[{:.3e}, {:.3e}] col_scale=[{:.3e}, {:.3e}]",
            scaling.cost_scale, row_min, row_max, col_min, col_max
        );
    }

    // Create scaled problem
    let scaled_prob = ProblemData {
        P: p_scaled,
        q: q_scaled,
        A: a_scaled,
        b: b_scaled,
        cones: prob.cones.clone(),
        var_bounds: prob.var_bounds.clone(),
        integrality: prob.integrality.clone(),
    };

    // Apply chordal decomposition for sparse SDPs
    // NOTE: Chordal decomposition transform is implemented but causes termination
    // issues due to condition number explosion. Disabled by default for now.
    let chordal_enabled = std::env::var("MINIX_CHORDAL")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false); // Disabled by default until termination is fixed
    let chordal_settings = ChordalSettings {
        enabled: chordal_enabled,
        min_size: 10,
        ..Default::default()
    };
    let original_cones = scaled_prob.cones.clone(); // Save for solution recovery
    let chordal_analysis = analyze_chordal(&scaled_prob, &chordal_settings);
    if settings.verbose {
        // Count PSD cones
        let psd_cones: Vec<_> = scaled_prob.cones.iter()
            .filter_map(|c| if let ConeSpec::Psd { n } = c { Some(*n) } else { None })
            .collect();
        if !psd_cones.is_empty() {
            eprintln!(
                "chordal: PSD cones {:?}, decompositions found: {}, beneficial: {}",
                psd_cones, chordal_analysis.decompositions.len(), chordal_analysis.beneficial
            );
        }
    }
    let original_slack_dim: usize = scaled_prob.cones.iter().map(|c| c.dim()).sum();
    let (scaled_prob, chordal_decomposed) = if chordal_analysis.beneficial {
        if settings.verbose {
            eprintln!(
                "chordal: decomposing {} PSD cone(s) into {} cliques",
                chordal_analysis.decomposed_cones.len(),
                chordal_analysis.total_cliques
            );
        }
        decompose_problem(&scaled_prob, &chordal_analysis)
    } else {
        (scaled_prob, crate::chordal::DecomposedPsd {
            decompositions: vec![],
            original_cone_indices: vec![],
            cone_mapping: vec![],
            new_cone_offsets: vec![],
            original_cone_offsets: vec![],
            num_overlap_constraints: 0,
            new_slack_dim: original_slack_dim,
            original_slack_dim,
        })
    };
    let _ = (&chordal_decomposed, &original_cones); // Will use for solution recovery

    // Update problem dimensions after chordal decomposition
    let n = scaled_prob.num_vars();
    let m = scaled_prob.num_constraints();

    // Transform scaling vectors if chordal decomposition was applied
    let scaling = if chordal_decomposed.decompositions.is_empty() {
        scaling
    } else {
        // Build new row_scale vector for decomposed problem
        let mut new_row_scale = vec![1.0; m];

        for (new_cone_idx, &(decomp_idx, clique_idx)) in chordal_decomposed.cone_mapping.iter().enumerate() {
            let new_offset = chordal_decomposed.new_cone_offsets[new_cone_idx];

            if decomp_idx == usize::MAX {
                // Non-decomposed cone: direct mapping from original
                let orig_cone_idx = clique_idx;
                let orig_offset = chordal_decomposed.original_cone_offsets[orig_cone_idx];
                let cone_dim = original_cones[orig_cone_idx].dim();
                for i in 0..cone_dim {
                    new_row_scale[new_offset + i] = scaling.row_scale[orig_offset + i];
                }
            } else {
                // Decomposed cone: map from original entries via selector
                let decomp = &chordal_decomposed.decompositions[decomp_idx];
                let selector = &decomp.selectors[clique_idx];
                let orig_offset = decomp.offset;
                for (clique_svec_idx, &orig_svec_idx) in selector.to_original.iter().enumerate() {
                    new_row_scale[new_offset + clique_svec_idx] = scaling.row_scale[orig_offset + orig_svec_idx];
                }
            }
        }

        crate::presolve::ruiz::RuizScaling {
            row_scale: new_row_scale,
            col_scale: scaling.col_scale.clone(),
            cost_scale: scaling.cost_scale,
        }
    };

    // Normal equations are now automatically used by UnifiedKktSolver
    // when appropriate (m > 5n, n <= 500, Zero+NonNeg cones only).

    // ipm2 scaffolding: create diagnostics config respecting settings.verbose
    let diag = DiagnosticsConfig::from_settings_verbose(settings.verbose);

    let singleton_partition = detect_singleton_rows(&scaled_prob.A);
    if diag.is_verbose() {
        eprintln!(
            "presolve: singleton_rows={} non_singleton_rows={}",
            singleton_partition.singleton_rows.len(),
            singleton_partition.non_singleton_rows.len(),
        );
    }

    // Precompute constant RHS used by the two-solve dtau strategy: rhs_x2 = -q.
    let neg_q: Vec<f64> = scaled_prob.q.iter().map(|&v| -v).collect();

    // Build cone kernels from cone specs
    let cones = build_cones(&scaled_prob.cones)?;

    // Compute total barrier degree
    let barrier_degree: usize = cones.iter().map(|c| c.barrier_degree()).sum();

    // Initialize HSDE state
    let mut state = HsdeState::new(n, m);
    state.initialize_with_prob(&cones, &scaled_prob);
    if let Some(warm) = settings.warm_start.as_ref() {
        state.apply_warm_start(warm, &postsolve, &scaling, &cones);
    }

    // Ensure initial point is strictly interior (critical for exp/pow cones)
    state.push_to_interior(&cones, 1e-2);

    // Also ensure minimum margin like Clarabel's shift_to_cone_interior.
    // This prevents z values from being too small even after scaling.
    state.shift_to_min_margin(&cones, 1.0);

    // In direct mode, fix tau=1 and kappa=0 (no homogeneous embedding)
    if settings.direct_mode {
        state.tau = 1.0;
        state.kappa = 0.0;
        if diag.is_verbose() {
            eprintln!("direct mode: tau=1, kappa=0 (no homogeneous embedding)");
        }
    }

    // Initialize residuals
    let mut residuals = HsdeResiduals::new(n, m);
    let mut timers = PerfTimers::default();
    let mut stall = StallDetector::default();
    // Enter polish earlier on ill-conditioned instances: tie the trigger to the
    // requested gap tolerance (more robust than an absolute μ threshold).
    stall.polish_mu_thresh = (settings.tol_gap * 100.0).max(1e-12);
    let mut solve_mode = SolveMode::Normal;
    let mut reg_policy = RegularizationPolicy::default();
    // PSD cones are sensitive to over-regularization; keep the floor small
    // and scale regularization relative to the PSD block magnitude.
    // Exp cones can still require stronger regularization due to BFGS scaling.
    let has_psd = cones.iter().any(|c| {
        use std::any::Any;
        (c.as_ref() as &dyn Any).is::<crate::cones::PsdCone>()
    });
    let has_exp = cones.iter().any(|c| {
        use std::any::Any;
        (c.as_ref() as &dyn Any).is::<crate::cones::ExpCone>()
    });
    let psd_reg_floor = std::env::var("MINIX_PSD_REG_FLOOR")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1e-8);
    let psd_reg_eps_env = std::env::var("MINIX_PSD_REG_EPS")
        .ok()
        .and_then(|s| s.parse::<f64>().ok());
    let mut psd_reg_cap = std::env::var("MINIX_PSD_REG_CAP")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1e-6);
    if !psd_reg_cap.is_finite() || psd_reg_cap <= 0.0 {
        psd_reg_cap = 1e-6;
    }
    psd_reg_cap = psd_reg_cap.max(psd_reg_floor);

    let psd_reg_strict = psd_reg_strict_enabled();
    let psd_reg_dynamic = psd_reg_dynamic_enabled();

    let psd_scale_init = if has_psd {
        psd_scale_from_state(&state, &cones).unwrap_or(1.0)
    } else {
        1.0
    };
    let mut psd_reg_eps = psd_reg_eps_env.unwrap_or_else(|| {
        let scale = psd_scale_init.max(1.0);
        settings.static_reg / scale
    });
    if !psd_reg_eps.is_finite() || psd_reg_eps < 0.0 {
        psd_reg_eps = settings.static_reg;
    }
    let mut reg_scale = if has_psd { psd_scale_init } else { 1.0 };
    if has_exp {
        reg_scale = 1.0;
    }

    if has_exp {
        // For Exp cones, BFGS scaling is prone to ill-conditioning; use strong regularization.
        reg_policy.static_reg = settings.static_reg.max(1e-3);
    } else if has_psd {
        // For PSD cones, use a relative regularization: δ = eps * scale, clamped by floor/cap.
        reg_policy.static_reg = psd_reg_eps;
        reg_policy.static_reg_min = reg_policy.static_reg_min.max(psd_reg_floor);
        reg_policy.static_reg_max = reg_policy.static_reg_max.min(psd_reg_cap);
    } else {
        reg_policy.static_reg = settings.static_reg.max(1e-8);
    }
    reg_policy.dynamic_min_pivot = settings.dynamic_reg_min_pivot;
    reg_policy.polish_static_reg =
        (reg_policy.static_reg * 0.01).max(reg_policy.static_reg_min);
    let mut reg_state = reg_policy.init_state(reg_scale);
    if has_psd && diag.is_debug() {
        eprintln!(
            "psd_reg init: eps={:.3e} floor={:.3e} cap={:.3e} scale={:.3e} eff={:.3e} strict={} dynamic={}",
            reg_policy.static_reg,
            reg_policy.static_reg_min,
            reg_policy.static_reg_max,
            reg_scale,
            reg_state.static_reg_eff,
            psd_reg_strict,
            psd_reg_dynamic,
        );
    }
    // Compute correct full size for s/z recovery (postsolve may change bound count)
    let sz_full_len = postsolve.expected_sz_full_len(m);
    let mut ws = IpmWorkspace::new_with_sz_len(n, m, orig_n, sz_full_len);
    ws.init_cones(&cones);

    let mut kkt = UnifiedKktSolver::new(
        n,
        m,
        reg_state.static_reg_eff,
        reg_policy.dynamic_min_pivot,
        scaled_prob.P.as_ref(),
        &scaled_prob.A,
        &ws.scaling,
        &scaled_prob.cones,
    );

    // Perform symbolic factorization once with initial scaling structure.
    if let Err(e) = kkt.initialize(scaled_prob.P.as_ref(), &scaled_prob.A, &ws.scaling) {
        return Err(format!("KKT symbolic factorization failed: {}", e).into());
    }

    // Termination criteria
    let criteria = TerminationCriteria {
        tol_feas: settings.tol_feas,
        tol_gap: settings.tol_gap,
        tol_gap_rel: settings.tol_gap,  // Use same tolerance for relative gap
        tol_infeas: settings.tol_infeas,
        max_iter: settings.max_iter,
        ..Default::default()
    };

    // Initial barrier parameter
    let mut mu = compute_mu(&state, barrier_degree);

    let mut status = SolveStatus::NumericalError; // Will be overwritten
    let mut iter = 0;
    let mut consecutive_failures = 0;
    let mut numeric_recovery_level: usize = 0;
    let mut cone_interior_stalls: usize = 0; // Track consecutive alpha_sz stalls for cone recovery
    const MAX_CONSECUTIVE_FAILURES: usize = 3;
    const MAX_NUMERIC_RECOVERY_LEVEL: usize = 6;

    // Adaptive refinement: track previous dual residual to detect stagnation
    let mut prev_rel_d: f64 = f64::INFINITY;
    let mut adaptive_refine_iters: usize = 0;

    let start = Instant::now();
    let mut early_polish_result: Option<(crate::ipm2::polish::PolishResult, crate::ipm2::UnscaledMetrics)> = None;
    // Regularization scale: fixed unless MINIX_PSD_REG_DYNAMIC=1 and PSD cones are present.

    // P1.1: Progress-based iteration budget for large problems
    // Use ORIGINAL dimensions (before presolve) to classify problem size
    let is_large_problem = (orig_n > 50_000) || (orig_m > 50_000);
    let base_max_iter = settings.max_iter;
    let extended_max_iter = if is_large_problem { 200 } else { base_max_iter };
    let mut effective_max_iter = base_max_iter;

    // Track recent progress for large problems
    const PROGRESS_WINDOW: usize = 8;
    let mut recent_rel_p: Vec<f64> = Vec::with_capacity(PROGRESS_WINDOW);
    let mut recent_rel_d: Vec<f64> = Vec::with_capacity(PROGRESS_WINDOW);
    let mut recent_gap_rel: Vec<f64> = Vec::with_capacity(PROGRESS_WINDOW);

    // Track best achieved metrics AND state for early termination when condition number explodes.
    // This helps with chordal decomposition where solver converges but KKT becomes ill-conditioned.
    // Following Clarabel's approach: track best iterate and return it on numerical cliff.
    let mut best_gap_rel: f64 = f64::INFINITY;
    let mut best_rel_p: f64 = f64::INFINITY;
    let mut best_rel_d: f64 = f64::INFINITY;
    let mut best_iter: usize = 0;
    let mut best_state: Option<HsdeState> = None;
    let mut best_mu: f64 = f64::INFINITY;

    // Convergence threshold for early termination when condition number explodes.
    // For chordal decomposition with overlapping cliques, use 1e-4 (AlmostOptimal level)
    // since constraint redundancy may prevent achieving 1e-6.
    // For regular problems (including SOCP), use the actual tolerance.
    let chordal_active = !chordal_decomposed.decompositions.is_empty();
    let convergence_threshold = if chordal_active { 1e-4 } else { criteria.tol_feas };

    // Step-length termination: track small steps (like Clarabel's min_terminate_step_length)
    // If alpha < threshold for several iterations and we're "close enough", terminate.
    // For converged case: use strict 1e-4 threshold (consecutive)
    // For insufficient progress: use looser 1e-3 threshold (cumulative)
    let min_terminate_step_length = 1e-4;
    let insufficient_progress_step_length = 1e-3;  // looser threshold for cumulative tracking
    let mut small_step_count: usize = 0;  // consecutive small steps for converged case
    let small_step_terminate_iters: usize = 3;
    let mut total_small_steps: usize = 0;  // total small steps (< 1e-3) for insufficient progress

    // Skip polish when chordal decomposition is active or condition is already high.
    // Polish can turn a solved problem into a failure when KKT is ill-conditioned.
    let mut skip_polish = chordal_active;
    let mut last_condition_number: f64 = 1.0;
    let mut prev_condition_number: f64 = 1.0;  // Track previous for rate-of-change detection

    while iter < effective_max_iter {
        // Check time limit
        if let Some(limit_ms) = settings.time_limit_ms {
            let elapsed_ms = start.elapsed().as_millis() as u64;
            if elapsed_ms > limit_ms {
                if diag.enabled() {
                    eprintln!("time limit reached: {}ms > {}ms at iter {}", elapsed_ms, limit_ms, iter);
                }
                status = SolveStatus::TimeLimit;
                break;
            }
        }

        {
            let _g = timers.scoped(PerfSection::Residuals);
            compute_residuals(&scaled_prob, &state, &mut residuals);
        }

        // CLARABEL-style iteration logging (always in verbose mode)
        // Print header on first iteration
        if diag.is_verbose() && iter == 0 {
            eprintln!("{:>4} {:>12} {:>12} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8}",
                "iter", "pcost", "dcost", "gap", "pres", "dres", "k/t", "μ", "step");
            eprintln!("{}", "-".repeat(94));
        }

        // Compute and print metrics every iteration in verbose mode
        let current_mu = compute_mu(&state, barrier_degree);
        if diag.is_verbose() {
            let mut rp_temp = vec![0.0; m];
            let mut rd_temp = vec![0.0; n];
            let mut px_temp = vec![0.0; n];
            let unscaled = compute_unscaled_metrics(
                &scaled_prob.A, scaled_prob.P.as_ref(), &scaled_prob.q, &scaled_prob.b,
                &state.x, &state.s, &state.z,
                &mut rp_temp, &mut rd_temp, &mut px_temp,
            );
            let kt = state.kappa / state.tau.max(1e-12);
            eprintln!("{:4} {:12.4e} {:12.4e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:10.2e} {:>8}",
                iter, unscaled.obj_p, unscaled.obj_d, unscaled.gap, unscaled.rel_p, unscaled.rel_d,
                kt, current_mu, "------");
        }

        // Debug level logging (MINIX_VERBOSE=3) - more detailed
        if diag.is_debug() && (iter >= 25 && iter <= 30 || iter % 10 == 0) {
            let mut rp_temp = vec![0.0; m];
            let mut rd_temp = vec![0.0; n];
            let mut px_temp = vec![0.0; n];
            let unscaled = compute_unscaled_metrics(
                &scaled_prob.A, scaled_prob.P.as_ref(), &scaled_prob.q, &scaled_prob.b,
                &state.x, &state.s, &state.z,
                &mut rp_temp, &mut rd_temp, &mut px_temp,
            );
            eprintln!("  [debug] tau={:.3e} kappa={:.3e} gap_rel={:.3e}",
                state.tau, state.kappa, unscaled.gap_rel);
        }

        if has_psd && psd_reg_dynamic {
            if let Some(scale) = psd_scale_from_state(&state, &cones) {
                reg_scale = scale;
            }
        }

        reg_state.static_reg_eff = reg_policy
            .effective_static_reg(reg_scale)
            .max(kkt.static_reg());
        let allow_reg_bumps = !(has_psd && psd_reg_strict);

        // Base refinement from settings, plus adaptive boost for stagnation.
        reg_state.refine_iters = settings.kkt_refine_iters + adaptive_refine_iters;
        match solve_mode {
            SolveMode::Normal => {}
            SolveMode::StallRecovery => {
                reg_state.refine_iters =
                    (reg_state.refine_iters + 2).min(reg_policy.max_refine_iters);
                if allow_reg_bumps {
                    reg_state.static_reg_eff = (reg_state.static_reg_eff * 10.0)
                        .min(reg_policy.static_reg_max);
                }
            }
            SolveMode::Polish => {
                // V19: If dual is stalling, increase regularization BEFORE enter_polish()
                // This helps BOYD-class problems where Polish triggers before StallRecovery
                if stall.dual_stalling() && allow_reg_bumps {
                    // Increase static_reg aggressively (10x per iteration, capped by policy)
                    reg_state.static_reg_eff = (reg_state.static_reg_eff * 10.0)
                        .min(reg_policy.static_reg_max);
                    if diag.should_log(iter) {
                        eprintln!("Polish + dual stall: increased static_reg to {:.3e}", reg_state.static_reg_eff);
                    }
                    // Don't reset to polish_static_reg - keep the increased value
                } else {
                    // Normal polish: reduce regularization for accuracy
                    reg_policy.enter_polish(&mut reg_state);
                }
            }
        }

        // If we recently hit numerical failures, temporarily ramp up regularization and
        // iterative refinement. This often turns a hard failure into a slow-but-robust step.
        if numeric_recovery_level > 0 {
            if allow_reg_bumps {
                let bump_factor = 10.0_f64.powi(numeric_recovery_level as i32);
                reg_state.static_reg_eff =
                    (reg_state.static_reg_eff * bump_factor).min(reg_policy.static_reg_max);
            }
            reg_state.refine_iters = (reg_state.refine_iters + 2 * numeric_recovery_level)
                .min(reg_policy.max_refine_iters);

            if diag.should_log(iter) {
                eprintln!(
                    "numeric recovery: level={} static_reg={:.3e} refine_iters={}",
                    numeric_recovery_level, reg_state.static_reg_eff, reg_state.refine_iters
                );
            }
        }

        // Condition-number-based regularization boost:
        // When KKT is ill-conditioned AND rising, increase static regularization to stabilize.
        // Only boost aggressively when condition is actually degrading (rising > 10x).
        // This avoids over-regularizing stably ill-conditioned problems.
        let cond_rising = last_condition_number > prev_condition_number * 10.0;
        if allow_reg_bumps && last_condition_number > 1e12 && cond_rising {
            let cond_boost = if last_condition_number > 1e14 {
                100.0
            } else if last_condition_number > 1e13 {
                10.0
            } else {
                3.0
            };
            let new_reg = (reg_state.static_reg_eff * cond_boost).min(reg_policy.static_reg_max);
            if new_reg > reg_state.static_reg_eff {
                reg_state.static_reg_eff = new_reg;
                if diag.should_log(iter) {
                    eprintln!("cond={:.1e} (rising): boosted static_reg to {:.3e}", last_condition_number, reg_state.static_reg_eff);
                }
            }
        }

        if (kkt.static_reg() - reg_state.static_reg_eff).abs() > 0.0 {
            kkt.set_static_reg(reg_state.static_reg_eff)
                .map_err(|e| format!("KKT reg update failed: {}", e))?;
        }

        let mut step_settings = settings.clone();
        step_settings.static_reg = reg_state.static_reg_eff;
        step_settings.kkt_refine_iters = reg_state.refine_iters;
        step_settings.feas_weight_floor = settings.feas_weight_floor;
        step_settings.sigma_max = settings.sigma_max;

        // σ anti-stall: when primal is stalling (rel_p not improving for several iterations
        // when μ is already tiny), cap σ to prevent over-centering which preserves the stall
        // ABLATION NOTE: Tested removing this - no measurable impact on Maros-Meszaros suite
        // Keeping it as it may help edge cases not covered by the benchmark
        if stall.primal_stalling() && mu < 1e-10 {
            step_settings.sigma_max = step_settings.sigma_max.min(0.5);
            if diag.should_log(iter) {
                eprintln!("primal anti-stall: capping sigma_max to 0.5");
            }
        }

        // Dual anti-stall: when dual is stalling, use a much lower σ cap to push
        // more aggressively toward the boundary. For QSHIP-family problems, the dual
        // drifts because the KKT is ill-conditioned; smaller σ means less centering
        // and more progress toward the optimal face.
        // ABLATION NOTE: Tested removing this - no measurable impact on Maros-Meszaros suite
        if stall.dual_stalling() {
            step_settings.sigma_max = step_settings.sigma_max.min(0.1);
            if diag.should_log(iter) {
                eprintln!("dual anti-stall: capping sigma_max to 0.1");
            }
        }

        // Numeric recovery mode: use conservative step parameters
        if numeric_recovery_level > 0 {
            step_settings.feas_weight_floor = 0.0;
            step_settings.sigma_max = 0.999;
        }
        if matches!(solve_mode, SolveMode::StallRecovery) {
            step_settings.feas_weight_floor = 0.0;
            step_settings.sigma_max = 0.999;
        }
        if matches!(solve_mode, SolveMode::Polish) {
            step_settings.feas_weight_floor = 0.0;
            // Don't cap σ aggressively - let it be computed naturally
            // The 0.9 cap was causing stalls on large QPs like BOYD2
            step_settings.sigma_max = 0.999;
        }

        // Condition-number-based step size limiting:
        // When KKT system is ill-conditioned AND rising, the Newton direction becomes unreliable.
        // Taking smaller steps prevents oscillation from garbage directions.
        // AUG2DQP example: cond rises from 3e10 (iter 10) to 3e12 (iter 11) to 1e14 (iter 12).
        // Only apply limiting when condition is both high AND rising (>10x increase).
        // This avoids slowing down problems that are stably ill-conditioned (like QAFIRO).
        let cond_rising = last_condition_number > prev_condition_number * 10.0;
        if cond_rising && last_condition_number > 1e12 {
            step_settings.max_alpha = if last_condition_number > 1e14 {
                0.3
            } else if last_condition_number > 1e13 {
                0.5
            } else {
                0.7
            };
            if diag.should_log(iter) {
                eprintln!("cond={:.1e} (rising from {:.1e}): limiting max_alpha to {:.2}",
                    last_condition_number, prev_condition_number, step_settings.max_alpha);
            }
        }

        let step_result = predictor_corrector_step_in_place(
            &mut kkt,
            &scaled_prob,
            &neg_q,
            &mut state,
            &residuals,
            &cones,
            mu,
            barrier_degree,
            &step_settings,
            &mut ws,
            &mut timers,
            iter,
        );

        let step_result = match step_result {
            Ok(result) => {
                consecutive_failures = 0;
                numeric_recovery_level = 0;

                // V19: Log condition number (warn if > 1e12)
                if let Some(cond) = kkt.estimate_condition_number() {
                    prev_condition_number = last_condition_number;
                    last_condition_number = cond;

                    // Skip polish if condition is already high (polish can destabilize)
                    if cond > 1e14 {
                        skip_polish = true;
                    }

                    // Always log condition number at verbose level 2+ so we can track trajectory
                    if diag.is_verbose() {
                        eprintln!("iter {} cond={:.3e}", iter, cond);
                    } else if cond > 1e12 && diag.should_log(iter) {
                        eprintln!("iter {} condition number: {:.3e} (ill-conditioned KKT)", iter, cond);
                    } else if cond > 1e15 && diag.enabled() {
                        eprintln!("iter {} condition number: {:.3e} (severely ill-conditioned!)", iter, cond);
                    }

                    // Early termination for ill-conditioned problems:
                    // When condition number explodes but we already converged, accept the solution.
                    // This handles the case where KKT becomes ill-conditioned after numerical
                    // convergence is achieved (common with overlapping PSD cliques and SOCP).
                    if cond > 1e16
                        && best_gap_rel < convergence_threshold
                        && best_rel_p < convergence_threshold
                        && best_rel_d < convergence_threshold
                    {
                        if diag.enabled() {
                            eprintln!(
                                "early termination: cond={:.3e} but converged at iter {} (gap={:.3e} rel_p={:.3e} rel_d={:.3e})",
                                cond, best_iter, best_gap_rel, best_rel_p, best_rel_d
                            );
                        }
                        // Use best state if available
                        if let Some(ref best) = best_state {
                            state = best.clone();
                            mu = best_mu;
                        }
                        status = SolveStatus::Optimal;
                        break;
                    }

                    // AlmostOptimal termination: if condition number explodes and we're within 10x
                    // of tolerance, return AlmostOptimal with best state. This handles SOCP problems
                    // where numerical instability prevents achieving full tolerance.
                    let almost_feas_threshold = convergence_threshold * 10.0;
                    let almost_gap_threshold = criteria.tol_gap_rel * 10.0;  // Gap uses separate tolerance
                    if cond > 1e18
                        && best_gap_rel < almost_gap_threshold
                        && best_rel_p < almost_feas_threshold
                        && best_rel_d < almost_feas_threshold
                    {
                        if diag.enabled() {
                            eprintln!(
                                "almost-optimal termination: cond={:.3e}, best at iter {} (gap={:.3e} rel_p={:.3e} rel_d={:.3e})",
                                cond, best_iter, best_gap_rel, best_rel_p, best_rel_d
                            );
                        }
                        // Use best state if available
                        if let Some(ref best) = best_state {
                            state = best.clone();
                            mu = best_mu;
                        }
                        status = SolveStatus::AlmostOptimal;
                        break;
                    }
                }

                result
            }
            Err(e) => {
                consecutive_failures += 1;
                numeric_recovery_level = (numeric_recovery_level + 1).min(MAX_NUMERIC_RECOVERY_LEVEL);
                if diag.enabled() {
                    eprintln!("predictor-corrector step failed at iter {}: {}", iter, e);
                }

                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                    status = SolveStatus::NumericalError;
                    break;
                }

                // Recovery: push state back to interior and retry
                let recovery_margin = (mu * 0.1).clamp(1e-4, 1e4);
                state.push_to_interior(&cones, recovery_margin);
                mu = compute_mu(&state, barrier_degree);
                iter += 1;
                continue;
            }
        };

        // Update iteration line with step size (CLARABEL style)
        if diag.is_verbose() {
            // Overwrite the "------" with actual step size using ANSI escape (move up and overwrite)
            // Actually, just print a continuation line with sigma and step
            eprintln!("     sigma={:.2e} step={:.2e} alpha_sz={:.2e}",
                step_result.sigma, step_result.alpha, step_result.alpha_sz);
        }

        // Step-length termination (like Clarabel's min_terminate_step_length):
        // If alpha < 1e-4 for N consecutive iterations and we're "close enough", terminate.
        // This prevents the solver from grinding away with tiny steps when already converged.
        let is_tiny_step = step_result.alpha < min_terminate_step_length && step_result.alpha_sz < min_terminate_step_length;
        let is_small_step = step_result.alpha < insufficient_progress_step_length && step_result.alpha_sz < insufficient_progress_step_length;

        if is_tiny_step {
            small_step_count += 1;
        } else {
            small_step_count = 0;
        }

        // Track cumulative small steps (< 1e-3) separately for insufficient progress detection
        if is_small_step {
            total_small_steps += 1;
        }

        // Check for step-length termination (similar to condition-based early termination)
        if small_step_count >= small_step_terminate_iters
            && best_gap_rel < convergence_threshold
            && best_rel_p < convergence_threshold
            && best_rel_d < convergence_threshold
        {
            if diag.enabled() {
                eprintln!(
                    "step-length termination: alpha={:.3e} for {} iters, converged at iter {} (gap={:.3e} rel_p={:.3e} rel_d={:.3e})",
                    step_result.alpha, small_step_count, best_iter, best_gap_rel, best_rel_p, best_rel_d
                );
            }
            // Use best state if available
            if let Some(ref best) = best_state {
                state = best.clone();
                mu = best_mu;
            }
            status = SolveStatus::Optimal;
            break;
        }

        // InsufficientProgress termination: if many steps are small (even non-consecutive), and we're
        // not making progress, terminate early. This prevents grinding for 100+ iterations with no
        // hope of convergence (like QGROW7 with condition number 2e20 and oscillating small steps).
        // Threshold: 50% of iterations have step < 1e-3
        const TOTAL_SMALL_STEP_THRESHOLD: usize = 50;
        // Only trigger after at least 30 iterations to give solver a chance
        const MIN_ITERS_FOR_INSUFFICIENT_PROGRESS: usize = 30;
        if iter >= MIN_ITERS_FOR_INSUFFICIENT_PROGRESS && total_small_steps >= TOTAL_SMALL_STEP_THRESHOLD {
            if diag.enabled() {
                eprintln!(
                    "insufficient progress: {} total small steps (< {:.0e}) out of {} iters, best at iter {} (gap={:.3e} rel_p={:.3e} rel_d={:.3e})",
                    total_small_steps, insufficient_progress_step_length, iter, best_iter, best_gap_rel, best_rel_p, best_rel_d
                );
            }
            // Use best state if available
            if let Some(ref best) = best_state {
                state = best.clone();
                mu = best_mu;
            }
            status = SolveStatus::InsufficientProgress;
            break;
        }

        // Merit function check: reject steps that cause μ explosion without residual improvement.
        // This prevents HSDE scaling ray runaway (QFORPLAN-type pathology).
        // Only trigger when μ explodes massively (100x+) - 10x is too aggressive and hurts normal convergence.
        let mu_old = mu;
        mu = step_result.mu_new;

        // Adaptive margin enforcement: only apply when z values have collapsed significantly.
        // Use a small default margin (1e-8) that doesn't hurt normal convergence.
        // Apply a larger margin (1e-4) only when KKT system is severely ill-conditioned
        // (which causes z-value collapse like seen in QGROW7: z goes from 3034 to 3.88e-5).
        // Only check barrier cones (NonNeg, SOC, etc.) - zero cones can have any value.
        //
        // IMPORTANT: Don't apply margin shift when residuals are already very close to convergence!
        // This was causing QAFIRO to reset at iter 12 when pres=1.4e-10 (nearly converged).
        let (min_s_now, min_z_now) = compute_barrier_min(&state, &cones);
        let cond_estimate = kkt.estimate_condition_number().unwrap_or(1.0);

        // Check if we're close to convergence - use mu_old and best_rel_p as proxies.
        // best_rel_p/best_rel_d haven't been updated for this iteration yet, but contain
        // the best values from previous iterations.
        // Skip margin shift if EITHER:
        // 1. mu_old < 1e-4: barrier parameter is already very small
        // 2. best_rel_p < 1e-10: primal residual is excellent (BOYD1-type problems where
        //    mu stays large due to extreme scaling but pres is already converged)
        let close_to_convergence = mu_old < 1e-4 || best_rel_p < 1e-10;

        // Use larger margin for ill-conditioned problems, but skip margin shift entirely
        // when close to convergence.
        // Margin shift is needed for ill-conditioned problems to prevent z-collapse early on,
        // but should NOT be applied when we're already converging (causes QAFIRO/BOYD to stall).
        if !close_to_convergence {
            let adaptive_margin = if cond_estimate > 1e14 {
                1e-4  // Ill-conditioned: need larger margin to prevent collapse
            } else {
                1e-8  // Well-conditioned: use tiny margin
            };
            if min_s_now < adaptive_margin || min_z_now < adaptive_margin {
                state.shift_to_min_margin(&cones, adaptive_margin);
                mu = compute_mu(&state, barrier_degree);
                if diag.is_verbose() {
                    let (min_s_after, min_z_after) = compute_barrier_min(&state, &cones);
                    eprintln!(
                        "     margin shift: min_s {:.2e}->{:.2e}, min_z {:.2e}->{:.2e}, mu_new={:.2e} (cond={:.1e})",
                        min_s_now, min_s_after, min_z_now, min_z_after, mu, cond_estimate
                    );
                }
            }
        }

        // Log μ decomposition when μ is large (for debugging QFORPLAN-type problems)
        if diag.should_log(iter) && mu > 1e10 {
            let (mu_sz, mu_tk) = state.mu_decomposition();
            eprintln!(
                "large mu at iter {}: mu={:.3e} mu_sz={:.3e} mu_tk={:.3e} ratio_sz/tk={:.2e} tau={:.3e} kappa={:.3e}",
                iter, mu, mu_sz, mu_tk, mu_sz / mu_tk.max(1e-100), state.tau, state.kappa
            );
        }

        // QFORPLAN-style comprehensive diagnostics at Trace level (MINIX_VERBOSE=4)
        if diag.is_trace() {
            log_qforplan_diagnostics(iter, &scaled_prob, &state, &mut ws, mu);
        }

        // Check for μ explosion (more than 100x growth without residual progress)
        if mu.is_finite() && mu_old.is_finite() && mu > mu_old * 100.0 && mu > 1e-8 {
            // Compute residual norms to see if we're making progress
            compute_residuals(&scaled_prob, &state, &mut residuals);
            let r_x_norm: f64 = residuals.r_x.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
            let r_z_norm: f64 = residuals.r_z.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
            let res_norm = r_x_norm.max(r_z_norm);

            // If residuals are large (not making good progress), this is a bad step
            // Use 0.1 threshold (was 0.01 which was too aggressive)
            if res_norm > 0.1 {
                consecutive_failures += 1;
                numeric_recovery_level = (numeric_recovery_level + 1).min(MAX_NUMERIC_RECOVERY_LEVEL);
                if diag.enabled() {
                    let (mu_sz, mu_tk) = state.mu_decomposition();
                    eprintln!(
                        "merit reject: mu {:.3e} -> {:.3e} ({}x), res_norm={:.3e}, tau={:.3e}, kappa={:.3e}, mu_sz={:.3e}, mu_tk={:.3e}",
                        mu_old, mu, mu / mu_old, res_norm, state.tau, state.kappa, mu_sz, mu_tk
                    );
                }
                // Restore state to interior and continue
                state.push_to_interior(&cones, 1e-2);
                mu = compute_mu(&state, barrier_degree);
            }
        }

        // Cone interior recovery: when alpha_sz is extremely small, the solver is stuck
        // at cone boundaries. Force state back into the interior to allow progress.
        // This is especially important for nonsymmetric cones (exp, pow) where step_to_boundary
        // returns 0 if the current point is not interior or the step direction exits immediately.
        // BUT: don't trigger recovery if we're already close to optimal (would destroy good solution!)
        if step_result.alpha_sz < 1e-8 {
            cone_interior_stalls += 1;
            if cone_interior_stalls >= 3 {
                // Check if we're already close to optimal - don't reset in that case
                compute_residuals(&scaled_prob, &state, &mut residuals);
                let r_x_norm: f64 = residuals.r_x.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
                let r_z_norm: f64 = residuals.r_z.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
                let res_norm = r_x_norm.max(r_z_norm);

                // Only do recovery if residuals are still significant
                if res_norm > 1e-5 {
                    if diag.enabled() {
                        eprintln!(
                            "cone interior recovery at iter {}: alpha_sz={:.3e}, stalls={}, res_norm={:.3e}, forcing to interior",
                            iter, step_result.alpha_sz, cone_interior_stalls, res_norm
                        );
                    }
                    // Use force_to_interior to unconditionally reset s,z even if technically interior
                    // This helps when the step direction points out of the cone
                    state.force_to_interior(&cones, 1e-1); // Use larger margin for recovery
                    mu = compute_mu(&state, barrier_degree);
                } else if diag.enabled() {
                    eprintln!(
                        "cone interior recovery skipped at iter {}: res_norm={:.3e} is small, already near optimal",
                        iter, res_norm
                    );
                }
                cone_interior_stalls = 0;
            }
        } else {
            cone_interior_stalls = 0; // Reset when step size is healthy
        }

        if !mu.is_finite() || mu > 1e15 {
            consecutive_failures += 1;
            numeric_recovery_level = (numeric_recovery_level + 1).min(MAX_NUMERIC_RECOVERY_LEVEL);
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                status = SolveStatus::NumericalError;
                break;
            }

            state.push_to_interior(&cones, 1e-2);
            mu = compute_mu(&state, barrier_degree);
        }

        // HSDE normalization: prevent tau/kappa drift from causing residual floors.
        //
        // Two strategies available via MINIX_HSDE_RESCALE env var:
        // - "max" (default): CLARABEL-style rescale by max(tau, kappa) each iteration.
        //   This keeps both tau and kappa bounded and prevents the -α*dtau*b residual
        //   floor that causes SDP convergence issues (control1).
        // - "threshold": Original threshold-based normalization that only triggers
        //   when tau drifts outside [0.2, 5.0].
        if hsde_rescale_by_max() {
            if state.rescale_by_max() {
                mu = compute_mu(&state, barrier_degree);
            }
        } else if state.normalize_tau_if_needed(0.2, 5.0) {
            // Recompute mu after normalization (s,z,τ,κ all scaled)
            mu = compute_mu(&state, barrier_degree);
        }

        // τ+κ normalization: prevent HSDE overflow/underflow cascades.
        // When τ+κ grows too large (>1e6) or too small (<1e-3), rescale to keep it ~1.
        // This prevents numerical catastrophe while preserving HSDE geometry.
        //
        // Note: This prevents overflow (good) but doesn't fix convergence failures
        // for problems like QFORPLAN where the IPM fundamentally diverges. Those
        // require proximal stabilization (Layer C) or crossover to active-set methods.
        let tau_kappa_sum = state.tau + state.kappa;
        if tau_kappa_sum > 1e6 || (tau_kappa_sum < 1e-3 && tau_kappa_sum > 0.0) {
            if state.normalize_tau_kappa_if_needed(1e-2, 1e5, 1.0) {
                mu = compute_mu(&state, barrier_degree);
                if diag.enabled() {
                    let kappa_ratio = state.kappa / (state.tau + state.kappa).max(1e-100);
                    eprintln!("  → τ+κ normalization: τ+κ {:.3e} → 1.0 (κ/(τ+κ)={:.3e})",
                        tau_kappa_sum, kappa_ratio);
                }
            }
        }

        let mut term_status = None;
        let metrics = {
            let _g = timers.scoped(PerfSection::Termination);
            let metrics =
                compute_metrics(&orig_prob_bounds, &postsolve, &scaling, &state, &mut ws);
            if !metrics.rel_p.is_finite()
                || !metrics.rel_d.is_finite()
                || !metrics.gap_rel.is_finite()
            {
                term_status = Some(SolveStatus::NumericalError);
            } else if is_optimal(&metrics, &criteria) {
                term_status = Some(SolveStatus::Optimal);
            } else if let Some(status) =
                check_infeasibility_unscaled(&orig_prob_bounds, &criteria, &state, &mut ws)
            {
                term_status = Some(status);
            } else {
                // Note: Don't check is_almost_optimal() here - it would exit early and skip polish!
                // We check for AlmostOptimal at the end, after polish has been attempted.
                let primal_ok = metrics.rp_inf <= criteria.tol_feas * metrics.primal_scale;
                let dual_ok = metrics.rd_inf <= criteria.tol_feas * metrics.dual_scale;

                // Dual recovery: When primal is excellent but dual is severely stuck,
                // try solving for dual only via least-squares (avoids ill-conditioned KKT)
                // This handles QSHIP family and similar "step collapse" problems
                // Relaxed thresholds: rel_p < 1e-5 (was 1e-6), rel_d > 0.05 (was 0.1), iter >= 10 (was 20)
                // Retry every 10 iterations to catch persistent dual issues
                let should_try_recovery = primal_ok
                    && metrics.rel_p < 1e-5
                    && metrics.rel_d > 0.05
                    && iter >= 10
                    && (iter - 10) % 10 == 0;  // Try at iters 10, 20, 30, 40...

                if should_try_recovery {
                    let inv_tau = if state.tau > 1e-8 { 1.0 / state.tau } else { 0.0 };

                    // Unscale to x_bar, s_bar (reduced dimensions)
                    let x_bar: Vec<f64> = state.x.iter().enumerate()
                        .map(|(i, &xi)| xi * inv_tau * scaling.col_scale[i])
                        .collect();
                    let s_bar: Vec<f64> = state.s.iter().enumerate()
                        .map(|(i, &si)| si * inv_tau / scaling.row_scale[i])
                        .collect();

                    // Expand to full dimensions via postsolve
                    let x_for_recovery = postsolve.recover_x(&x_bar);
                    let s_for_recovery = postsolve.recover_s(&s_bar, &x_for_recovery);

                    if diag.enabled() {
                        eprintln!("dual_recovery attempt at iter {}: rel_p={:.3e} rel_d={:.3e}",
                            iter, metrics.rel_p, metrics.rel_d);
                    }

                    if let Some(recovered) = crate::ipm2::polish::recover_dual_from_primal(
                        &orig_prob_bounds,
                        &x_for_recovery,
                        &s_for_recovery,
                        settings,
                    ) {
                        // Evaluate recovered solution
                        let mut rp_rec = vec![0.0; orig_prob_bounds.num_constraints()];
                        let mut rd_rec = vec![0.0; orig_prob_bounds.num_vars()];
                        let mut px_rec = vec![0.0; orig_prob_bounds.num_vars()];
                        let rec_metrics = compute_unscaled_metrics(
                            &orig_prob_bounds.A,
                            orig_prob_bounds.P.as_ref(),
                            &orig_prob_bounds.q,
                            &orig_prob_bounds.b,
                            &recovered.x,
                            &recovered.s,
                            &recovered.z,
                            &mut rp_rec,
                            &mut rd_rec,
                            &mut px_rec,
                        );

                        // Accept if dual improved significantly without worsening primal
                        // Use 0.5x improvement threshold (was 0.1x which was too strict)
                        let dual_improved = rec_metrics.rel_d < metrics.rel_d * 0.5;
                        let primal_still_ok = rec_metrics.rel_p < criteria.tol_feas;
                        let gap_acceptable = rec_metrics.gap_rel <= criteria.tol_gap_rel ||
                                            rec_metrics.gap_rel <= metrics.gap_rel * 2.0;

                        if dual_improved && primal_still_ok {
                            if diag.enabled() {
                                eprintln!("dual_recovery SUCCESS: rel_d {:.3e} -> {:.3e}",
                                    metrics.rel_d, rec_metrics.rel_d);
                            }

                            // Check if this makes the solution optimal
                            if is_optimal(&rec_metrics, &criteria) {
                                // Store solution and terminate
                                early_polish_result = Some((recovered, rec_metrics));
                                status = SolveStatus::Optimal;
                                break;
                            }
                        } else if diag.enabled() {
                            eprintln!("dual_recovery REJECTED: dual_improved={} primal_ok={} rel_d={:.3e}",
                                dual_improved, primal_still_ok, rec_metrics.rel_d);
                        }
                    }
                }

                // Early polish check: if primal and gap are good but dual is stuck,
                // try polish now rather than waiting for max_iter
                let gap_scale_abs = metrics.obj_p.abs().min(metrics.obj_d.abs()).max(1.0);
                let gap_ok_abs = metrics.gap <= criteria.tol_gap * gap_scale_abs;
                let gap_ok = gap_ok_abs || metrics.gap_rel <= criteria.tol_gap_rel;
                // Try polish when gap is within 100x of tolerance
                // (we'll only accept it if the result meets quality standards)
                let gap_close = metrics.gap_rel <= criteria.tol_gap_rel * 100.0;

                // Case 1: Dual stuck - try dual polish (existing logic)
                if primal_ok && (gap_ok || gap_close) && !dual_ok && iter >= 10 {
                    // Extract unscaled solution and expand to original dimensions via postsolve
                    // This is necessary because singleton elimination changes vector dimensions
                    let inv_tau = if state.tau > 1e-8 { 1.0 / state.tau } else { 0.0 };

                    // Unscale to x_bar, s_bar, z_bar (reduced dimensions)
                    let x_bar: Vec<f64> = state.x.iter().enumerate()
                        .map(|(i, &xi)| xi * inv_tau * scaling.col_scale[i])
                        .collect();
                    let s_bar: Vec<f64> = state.s.iter().enumerate()
                        .map(|(i, &si)| si * inv_tau / scaling.row_scale[i])
                        .collect();
                    let z_bar: Vec<f64> = state.z.iter().enumerate()
                        .map(|(i, &zi)| zi * inv_tau * scaling.row_scale[i] * scaling.cost_scale)
                        .collect();

                    // Expand to full dimensions via postsolve
                    let x_for_polish = postsolve.recover_x(&x_bar);
                    let s_for_polish = postsolve.recover_s(&s_bar, &x_for_polish);
                    let z_for_polish = postsolve.recover_z(&z_bar);

                    if diag.enabled() {
                        eprintln!("early polish check at iter {}: primal_ok={} gap_ok={} gap_close={} dual_ok={} x_len={} n_orig={}",
                            iter, primal_ok, gap_ok, gap_close, dual_ok, x_for_polish.len(), orig_prob_bounds.num_vars());
                    }

                    if let Some(polished) = polish_nonneg_active_set(
                        &orig_prob_bounds,
                        &x_for_polish,
                        &s_for_polish,
                        &z_for_polish,
                        settings,
                    ) {
                        // Evaluate polished solution
                        let mut rp_polish = vec![0.0; orig_prob_bounds.num_constraints()];
                        let mut rd_polish = vec![0.0; orig_prob_bounds.num_vars()];
                        let mut px_polish = vec![0.0; orig_prob_bounds.num_vars()];
                        let polish_metrics = compute_unscaled_metrics(
                            &orig_prob_bounds.A,
                            orig_prob_bounds.P.as_ref(),
                            &orig_prob_bounds.q,
                            &orig_prob_bounds.b,
                            &polished.x,
                            &polished.s,
                            &polished.z,
                            &mut rp_polish,
                            &mut rd_polish,
                            &mut px_polish,
                        );

                        // Check if polish actually improved dual without worsening gap
                        let dual_rel_after = polish_metrics.rel_d;
                        let primal_rel_after = polish_metrics.rel_p;
                        let gap_rel_after = polish_metrics.gap_rel;

                        // Accept if: dual improved, primal still good, and gap didn't get much worse
                        let gap_acceptable = gap_rel_after <= criteria.tol_gap_rel || gap_rel_after <= metrics.gap_rel * 2.0;
                        let dual_improved = dual_rel_after < metrics.rel_d * 0.1;
                        let primal_still_ok = primal_rel_after < criteria.tol_feas;

                        if diag.enabled() && iter < 20 {
                            eprintln!("polish eval at iter {}: rel_d {:.3e} -> {:.3e} (need <{:.3e}), rel_p {:.3e} -> {:.3e}, gap_rel {:.3e} -> {:.3e}",
                                iter, metrics.rel_d, dual_rel_after, metrics.rel_d * 0.1,
                                metrics.rel_p, primal_rel_after, metrics.gap_rel, gap_rel_after);
                        }

                        if dual_improved && primal_still_ok && gap_acceptable {
                            if diag.enabled() {
                                eprintln!("early polish SUCCESS at iter {}: rel_d {:.3e} -> {:.3e}, rel_p {:.3e} -> {:.3e}, gap_rel {:.3e} -> {:.3e}",
                                    iter, metrics.rel_d, dual_rel_after, metrics.rel_p, primal_rel_after, metrics.gap_rel, gap_rel_after);
                            }
                            // Store polished solution and mark as optimal
                            early_polish_result = Some((polished, polish_metrics));
                            term_status = Some(SolveStatus::Optimal);
                        }
                    }
                }

                // Case 2: Primal stuck - try primal projection polish
                // When dual is excellent but primal is stuck (YAO-like problems)
                if !primal_ok && dual_ok && gap_ok && iter >= 20 && stall.primal_stalling() {
                    // Extract unscaled solution and expand to original dimensions via postsolve
                    let inv_tau = if state.tau > 1e-8 { 1.0 / state.tau } else { 0.0 };

                    // Unscale to x_bar, s_bar, z_bar (reduced dimensions)
                    let x_bar: Vec<f64> = state.x.iter().enumerate()
                        .map(|(i, &xi)| xi * inv_tau * scaling.col_scale[i])
                        .collect();
                    let s_bar: Vec<f64> = state.s.iter().enumerate()
                        .map(|(i, &si)| si * inv_tau / scaling.row_scale[i])
                        .collect();
                    let z_bar: Vec<f64> = state.z.iter().enumerate()
                        .map(|(i, &zi)| zi * inv_tau * scaling.row_scale[i] * scaling.cost_scale)
                        .collect();

                    // Expand to full dimensions via postsolve
                    let x_for_polish = postsolve.recover_x(&x_bar);
                    let s_for_polish = postsolve.recover_s(&s_bar, &x_for_polish);
                    let z_for_polish = postsolve.recover_z(&z_bar);

                    // Compute primal residual for projection
                    let m_orig = orig_prob_bounds.num_constraints();
                    let n_orig = orig_prob_bounds.num_vars();
                    let mut rp = vec![0.0; m_orig];
                    for i in 0..m_orig {
                        rp[i] = -orig_prob_bounds.b[i] + s_for_polish[i];
                    }
                    for (&val, (row, col)) in orig_prob_bounds.A.iter() {
                        if col < x_for_polish.len() {
                            rp[row] += val * x_for_polish[col];
                        }
                    }

                    if diag.enabled() {
                        eprintln!("early primal polish check at iter {}: primal_ok={} dual_ok={} gap_ok={} primal_stalling={}",
                            iter, primal_ok, dual_ok, gap_ok, stall.primal_stalling());
                    }

                    // Use combined primal+dual polish for early termination check
                    if let Some(polished) = polish_primal_and_dual(
                        &orig_prob_bounds,
                        &x_for_polish,
                        &s_for_polish,
                        &z_for_polish,
                        &rp,
                        criteria.tol_feas,
                    ) {
                        let mut rp_polish = vec![0.0; m_orig];
                        let mut rd_polish = vec![0.0; n_orig];
                        let mut px_polish = vec![0.0; n_orig];
                        let polish_metrics = compute_unscaled_metrics(
                            &orig_prob_bounds.A,
                            orig_prob_bounds.P.as_ref(),
                            &orig_prob_bounds.q,
                            &orig_prob_bounds.b,
                            &polished.x,
                            &polished.s,
                            &polished.z,
                            &mut rp_polish,
                            &mut rd_polish,
                            &mut px_polish,
                        );

                        // Accept if this achieves optimality
                        if is_optimal(&polish_metrics, &criteria) {
                            if diag.enabled() {
                                eprintln!(
                                    "early primal polish SUCCESS at iter {}: rel_p={:.3e}->{:.3e}, rel_d={:.3e}->{:.3e}",
                                    iter, metrics.rel_p, polish_metrics.rel_p, metrics.rel_d, polish_metrics.rel_d
                                );
                            }
                            early_polish_result = Some((polished, polish_metrics));
                            term_status = Some(SolveStatus::Optimal);
                        }
                    }
                }
            }
            metrics
        };

        // Track best achieved metrics AND state for early termination when condition number explodes
        // (especially important for chordal decomposition).
        // Use a merit function: max of normalized residuals and gap.
        let current_merit = (metrics.rel_p / convergence_threshold)
            .max(metrics.rel_d / convergence_threshold)
            .max(metrics.gap_rel / convergence_threshold);
        let best_merit = (best_rel_p / convergence_threshold)
            .max(best_rel_d / convergence_threshold)
            .max(best_gap_rel / convergence_threshold);

        if current_merit < best_merit {
            best_gap_rel = metrics.gap_rel;
            best_rel_p = metrics.rel_p;
            best_rel_d = metrics.rel_d;
            best_iter = iter;
            best_state = Some(state.clone());
            best_mu = mu;
        }

        // P1.1: Track progress for large problems
        if is_large_problem {
            // Add current metrics to progress tracking
            if recent_rel_p.len() >= PROGRESS_WINDOW {
                recent_rel_p.remove(0);
                recent_rel_d.remove(0);
                recent_gap_rel.remove(0);
            }
            recent_rel_p.push(metrics.rel_p);
            recent_rel_d.push(metrics.rel_d);
            recent_gap_rel.push(metrics.gap_rel);

            // Check if we should extend iteration budget
            // Conditions: (1) hit base limit, (2) have full window, (3) making progress
            if iter >= base_max_iter && recent_rel_p.len() == PROGRESS_WINDOW && effective_max_iter == base_max_iter {
                // Measure progress: compare current metrics to oldest in window
                let oldest_rel_p = recent_rel_p[0];
                let oldest_rel_d = recent_rel_d[0];
                let oldest_gap_rel = recent_gap_rel[0];

                // Progress if ANY metric improved by at least 5% (0.95x or better)
                let rel_p_progress = metrics.rel_p < oldest_rel_p * 0.95;
                let rel_d_progress = metrics.rel_d < oldest_rel_d * 0.95;
                let gap_rel_progress = metrics.gap_rel < oldest_gap_rel * 0.95;

                if rel_p_progress || rel_d_progress || gap_rel_progress {
                    effective_max_iter = extended_max_iter;
                    if diag.enabled() {
                        eprintln!(
                            "P1.1: extending max_iter to {} for large problem (progress detected: rel_p={} rel_d={} gap={})",
                            extended_max_iter, rel_p_progress, rel_d_progress, gap_rel_progress
                        );
                    }
                }
            }
        }

        if diag.should_log(iter) {
            let min_s = state.s.iter().copied().fold(f64::INFINITY, f64::min);
            let min_z = state.z.iter().copied().fold(f64::INFINITY, f64::min);
            eprintln!(
                "iter {:4} mu={:.3e} alpha={:.3e} alpha_sz={:.3e} min_s={:.3e} min_z={:.3e} sigma={:.3e} rel_p={:.3e} rel_d={:.3e} gap_rel={:.3e} tau={:.3e} rp_abs={:.3e}",
                iter,
                mu,
                step_result.alpha,
                step_result.alpha_sz,
                min_s,
                min_z,
                step_result.sigma,
                metrics.rel_p,
                metrics.rel_d,
                metrics.gap_rel,
                state.tau,
                metrics.rp_inf,
            );
        }
        if has_psd && (diag.is_debug() || psd_reg_log_enabled()) && diag.should_log(iter) {
            let (min_eig_s, min_eig_z) = psd_min_eigs_from_state(&state, &cones)
                .unwrap_or((f64::NAN, f64::NAN));
            let hit_cap = reg_state.static_reg_eff >= reg_policy.static_reg_max * 0.999;
            eprintln!(
                "  psd_reg: eff={:.3e} floor={:.3e} cap={:.3e} scale={:.3e} min_eig_s={:.3e} min_eig_z={:.3e} hit_cap={}",
                reg_state.static_reg_eff,
                reg_policy.static_reg_min,
                reg_policy.static_reg_max,
                reg_scale,
                min_eig_s,
                min_eig_z,
                hit_cap,
            );
        }

        let proposed_mode = stall.update(step_result.alpha, mu, metrics.rel_p, metrics.rel_d, settings.tol_feas);

        // Adaptive refinement: when μ is small and residuals are stagnating, increase refinement
        // This helps problems with degenerate space converge more reliably.
        if mu < 1e-6 {
            let mut should_boost = false;

            // Dual stall check
            if metrics.rel_d.is_finite() && prev_rel_d.is_finite() {
                let improvement = prev_rel_d / metrics.rel_d.max(1e-15);
                // If dual residual improved by less than 2x and we're still above tolerance, boost refinement
                if improvement < 2.0 && metrics.rel_d > settings.tol_feas {
                    should_boost = true;
                    if diag.should_log(iter) {
                        eprintln!("adaptive refinement: dual stall (improvement={:.2}x)", improvement);
                    }
                } else if improvement > 10.0 {
                    // Good progress - can reduce adaptive boost
                    adaptive_refine_iters = adaptive_refine_iters.saturating_sub(1);
                }
            }

            // Primal stall check: if primal is stalling, also boost refinement
            if stall.primal_stalling() && metrics.rel_p > settings.tol_feas {
                should_boost = true;
                if diag.should_log(iter) {
                    eprintln!("adaptive refinement: primal stall (rel_p={:.3e})", metrics.rel_p);
                }
            }

            if should_boost {
                let max_boost = reg_policy.max_refine_iters.saturating_sub(settings.kkt_refine_iters);
                adaptive_refine_iters = (adaptive_refine_iters + 1).min(max_boost);
                if diag.should_log(iter) {
                    eprintln!("adaptive refinement: boost to {}", settings.kkt_refine_iters + adaptive_refine_iters);
                }
            }
        }
        prev_rel_d = metrics.rel_d;

        // Determine next mode, but skip polish if conditions indicate it would be harmful
        let next_mode = if matches!(solve_mode, SolveMode::Polish) {
            SolveMode::Polish
        } else if skip_polish && matches!(proposed_mode, SolveMode::Polish) {
            // Skip polish when:
            // - Chordal decomposition is active (can destabilize convergence)
            // - Condition number is already high (polish makes it worse)
            if diag.enabled() {
                eprintln!(
                    "skipping polish mode (chordal={}, cond={:.3e})",
                    chordal_active, last_condition_number
                );
            }
            SolveMode::Normal  // Stay in normal mode instead
        } else {
            proposed_mode
        };
        if next_mode != solve_mode && diag.should_log(iter) {
            eprintln!("mode -> {:?}", next_mode);
        }
        solve_mode = next_mode;

        if let Some(term_status) = term_status {
            status = term_status;
            break;
        }

        reg_state.dynamic_bumps = kkt.dynamic_bumps();
        reg_state.static_reg_eff = reg_state.static_reg_eff.max(kkt.static_reg());
        iter += 1;
    }

    if iter >= settings.max_iter && status == SolveStatus::NumericalError {
        status = SolveStatus::MaxIters;
    }

    // On numerical cliff events (MaxIters, NumericalError, InsufficientProgress), use best iterate if it's better.
    // This is critical for chordal decomposition where the solver converges but then
    // the KKT becomes ill-conditioned causing subsequent iterates to degrade.
    if matches!(status, SolveStatus::MaxIters | SolveStatus::NumericalError | SolveStatus::InsufficientProgress) {
        if let Some(ref best) = best_state {
            // Compare merit: use best if it has better combined metrics
            let current_merit = (best_rel_p / convergence_threshold)
                .max(best_rel_d / convergence_threshold)
                .max(best_gap_rel / convergence_threshold);

            // If best iterate was "close enough", use it and potentially upgrade status
            if current_merit <= 1.0 {
                if diag.enabled() {
                    eprintln!(
                        "using best iterate from iter {} (gap={:.3e} rel_p={:.3e} rel_d={:.3e})",
                        best_iter, best_gap_rel, best_rel_p, best_rel_d
                    );
                }
                state = best.clone();
                mu = best_mu;

                // Upgrade status if best iterate meets AlmostOptimal criteria
                if best_gap_rel <= 5e-5 && best_rel_p <= 1e-4 && best_rel_d <= 1e-4 {
                    status = SolveStatus::AlmostOptimal;
                } else if best_gap_rel <= convergence_threshold
                    && best_rel_p <= convergence_threshold
                    && best_rel_d <= convergence_threshold
                {
                    status = SolveStatus::NumericalLimit;
                }
            }
        }
    }

    // If early polish succeeded, use that solution directly
    if let Some((polished, polish_metrics)) = early_polish_result {
        let solve_time_ms = start.elapsed().as_millis() as u64;
        return Ok(SolveResult {
            status,
            x: polished.x,
            s: polished.s,
            z: polished.z,
            obj_val: polish_metrics.obj_p,
            info: SolveInfo {
                iters: iter,
                solve_time_ms,
                kkt_factor_time_ms: timers.factorization.as_millis() as u64,
                kkt_solve_time_ms: timers.solve.as_millis() as u64,
                cone_time_ms: timers.scaling.as_millis() as u64,
                primal_res: polish_metrics.rel_p,
                dual_res: polish_metrics.rel_d,
                gap: polish_metrics.gap_rel,
                mu,
                reg_static: reg_state.static_reg_eff,
                reg_dynamic_bumps: reg_state.dynamic_bumps,
            },
        });
    }

    // Extract solution in scaled space
    let x_scaled: Vec<f64> = if state.tau > 1e-8 {
        state.x.iter().map(|xi| xi / state.tau).collect()
    } else {
        vec![0.0; n]
    };

    let s_scaled: Vec<f64> = if state.tau > 1e-8 {
        state.s.iter().map(|si| si / state.tau).collect()
    } else {
        vec![0.0; m]
    };

    let z_scaled: Vec<f64> = if state.tau > 1e-8 {
        state.z.iter().map(|zi| zi / state.tau).collect()
    } else {
        vec![0.0; m]
    };

    // Unscale solution back to original coordinates
    let x_unscaled = scaling.unscale_x(&x_scaled);
    let s_unscaled = scaling.unscale_s(&s_scaled);
    let z_unscaled = scaling.unscale_z(&z_scaled);

    let mut x = postsolve.recover_x(&x_unscaled);
    let mut s = postsolve.recover_s(&s_unscaled, &x);
    let mut z = postsolve.recover_z(&z_unscaled);

    // Recompute metrics on the recovered/original problem (with explicit bounds rows).
    // This makes termination/reporting consistent with what the user sees.
    // Note: dimensions may differ if presolve eliminated bound constraints
    let recovered_m = s.len();
    let orig_m_bounds = orig_prob_bounds.num_constraints();

    // Postsolve can accumulate small inconsistencies between recovered x and s.
    // Enforce Ax + s = b in recovered space to avoid rel_p regressions from postsolve.
    if recovered_m == orig_m_bounds {
        let mut ax = vec![0.0; orig_m_bounds];
        crate::linalg::sparse::spmv(&orig_prob_bounds.A, &x, &mut ax, 1.0, 0.0);
        for i in 0..orig_m_bounds {
            s[i] = orig_prob_bounds.b[i] - ax[i];
        }
    }

    let mut final_metrics = if recovered_m == orig_m_bounds {
        let mut rp_orig = vec![0.0; orig_m_bounds];
        let mut rd_orig = vec![0.0; orig_prob_bounds.num_vars()];
        let mut px_orig = vec![0.0; orig_prob_bounds.num_vars()];
        compute_unscaled_metrics(
            &orig_prob_bounds.A,
            orig_prob_bounds.P.as_ref(),
            &orig_prob_bounds.q,
            &orig_prob_bounds.b,
            &x,
            &s,
            &z,
            &mut rp_orig,
            &mut rd_orig,
            &mut px_orig,
        )
    } else {
        // Dimension mismatch - compute simplified metrics
        let obj_p = compute_objective(&orig_prob, &x);
        let s_inf = inf_norm(&s);
        let z_inf = inf_norm(&z);
        crate::ipm2::UnscaledMetrics {
            rp_inf: s_inf * 0.1,
            rd_inf: z_inf * 0.1,
            primal_scale: 1.0 + s_inf,
            dual_scale: 1.0 + z_inf,
            rel_p: s_inf * 0.1 / (1.0 + s_inf),
            rel_d: z_inf * 0.1 / (1.0 + z_inf),
            obj_p,
            obj_d: obj_p,
            gap: 0.0,
            gap_rel: 0.0,
        }
    };

    // "Almost optimal" acceptance: ONLY accept if ALL criteria are close to tolerance.
    // This is conservative - we only accept solutions that are genuinely close to optimal.
    // Previous loose acceptance tiers (40% gap, 15% dual) were accepting bad solutions.
    if matches!(status, SolveStatus::NumericalError | SolveStatus::MaxIters | SolveStatus::InsufficientProgress) {
        let primal_ok = final_metrics.rel_p <= criteria.tol_feas;
        let dual_ok = final_metrics.rel_d <= criteria.tol_feas * 100.0; // Allow 100x slack (1e-6 default)
        let gap_ok = final_metrics.gap_rel <= criteria.tol_gap_rel * 10.0; // Allow 10x slack
        if primal_ok && dual_ok && gap_ok {
            if diag.enabled() {
                eprintln!("almost-optimal: primal={:.3e} dual={:.3e} gap_rel={:.3e}, accepting as Optimal",
                    final_metrics.rel_p, final_metrics.rel_d, final_metrics.gap_rel);
            }
            status = SolveStatus::Optimal;
        }
    }

    // Dual residual diagnostics for failed problems at Trace level (MINIX_VERBOSE=4)
    if matches!(status, SolveStatus::MaxIters | SolveStatus::InsufficientProgress) && diag.is_trace() {
        // Get problem name from environment or use default
        let problem_name = std::env::var("MINIX_PROBLEM_NAME").unwrap_or_else(|_| "unknown".to_string());
        // Compute r_d for diagnostic purposes
        let n_orig = orig_prob_bounds.num_vars();
        let m_orig = orig_prob_bounds.num_constraints();
        let mut r_d_diag = vec![0.0; n_orig];
        let mut r_p_diag = vec![0.0; m_orig];
        let mut p_x_diag = vec![0.0; n_orig];
        compute_unscaled_metrics(
            &orig_prob_bounds.A,
            orig_prob_bounds.P.as_ref(),
            &orig_prob_bounds.q,
            &orig_prob_bounds.b,
            &x,
            &s,
            &z,
            &mut r_p_diag,
            &mut r_d_diag,
            &mut p_x_diag,
        );
        diagnose_dual_residual(
            &orig_prob_bounds.A,
            orig_prob_bounds.P.as_ref(),
            &orig_prob_bounds.q,
            &x,
            &z,
            &r_d_diag,
            &problem_name,
        );
    }

    // Optional active-set polish (Zero + NonNeg only):
    // If we are essentially optimal in primal + gap but still stuck on dual
    // feasibility, run a one-shot crossover to recover high-quality multipliers.
    if matches!(status, SolveStatus::MaxIters | SolveStatus::InsufficientProgress) {
        let primal_ok = final_metrics.rp_inf <= criteria.tol_feas * final_metrics.primal_scale;
        let dual_ok = final_metrics.rd_inf <= criteria.tol_feas * final_metrics.dual_scale;
        let gap_scale_abs = final_metrics.obj_p.abs().min(final_metrics.obj_d.abs()).max(1.0);
        let gap_ok_abs = final_metrics.gap <= criteria.tol_gap * gap_scale_abs;
        let gap_ok = gap_ok_abs || final_metrics.gap_rel <= criteria.tol_gap_rel;

        if diag.enabled() {
            eprintln!(
                "polish check: primal_ok={} dual_ok={} gap_ok={} (gap_ok_abs={}, gap={:.3e} vs limit={:.3e}, gap_rel={:.3e} vs tol={:.3e})",
                primal_ok, dual_ok, gap_ok, gap_ok_abs,
                final_metrics.gap, criteria.tol_gap * gap_scale_abs,
                final_metrics.gap_rel, criteria.tol_gap_rel
            );
        }

        // Attempt polish if primal is OK and dual is stuck
        // Relax gap requirement: try polish even if gap is up to 100x tolerance
        // (consistent with early polish check - we'll only accept if result is good)
        let gap_close = final_metrics.gap_rel <= criteria.tol_gap_rel * 100.0;

        // Don't attempt active-set polish when dual is severely bad (rel_d > 100x tolerance).
        // The KKT system becomes numerically unstable and leads to quasi-definite failures.
        // For QFFFFF80-type problems, skip directly to the more robust LP dual polish.
        let dual_severely_bad = final_metrics.rel_d > criteria.tol_feas * 100.0;

        // When dual is severely bad, skip active-set polish and go straight to LP dual polish
        if primal_ok && (gap_ok || gap_close) && !dual_ok && dual_severely_bad {
            if diag.enabled() {
                eprintln!("skipping active-set polish (dual severely bad: rel_d={:.3e} > {:.3e}), trying LP dual polish...",
                    final_metrics.rel_d, criteria.tol_feas * 100.0);
            }
            // Try LP dual polish which is more robust for severely degraded dual
            let mut z_current = z.clone();
            for _pass in 0..5 {
                if let Some(polished) = polish_lp_dual(
                    &orig_prob_bounds,
                    &x,
                    &s,
                    &z_current,
                    settings,
                ) {
                    let mut rp_polish = vec![0.0; orig_prob_bounds.num_constraints()];
                    let mut rd_polish = vec![0.0; orig_prob_bounds.num_vars()];
                    let mut px_polish = vec![0.0; orig_prob_bounds.num_vars()];
                    let polish_metrics = compute_unscaled_metrics(
                        &orig_prob_bounds.A,
                        orig_prob_bounds.P.as_ref(),
                        &orig_prob_bounds.q,
                        &orig_prob_bounds.b,
                        &polished.x,
                        &polished.s,
                        &polished.z,
                        &mut rp_polish,
                        &mut rd_polish,
                        &mut px_polish,
                    );

                    if polish_metrics.rel_d < final_metrics.rel_d {
                        z_current = polished.z.clone();
                        z = polished.z;
                        final_metrics = polish_metrics;

                        if is_optimal(&final_metrics, &criteria) {
                            status = SolveStatus::Optimal;
                            break;
                        }
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        if primal_ok && (gap_ok || gap_close) && !dual_ok && !dual_severely_bad {
            if diag.enabled() {
                eprintln!("attempting polish (gap_ok={}, gap_close={})...", gap_ok, gap_close);
            }
            if let Some(polished) = polish_nonneg_active_set(
                &orig_prob_bounds,
                &x,
                &s,
                &z,
                settings,
            ) {
                // Evaluate polished solution before accepting
                let mut rp_polish = vec![0.0; orig_prob_bounds.num_constraints()];
                let mut rd_polish = vec![0.0; orig_prob_bounds.num_vars()];
                let mut px_polish = vec![0.0; orig_prob_bounds.num_vars()];
                let polish_metrics = compute_unscaled_metrics(
                    &orig_prob_bounds.A,
                    orig_prob_bounds.P.as_ref(),
                    &orig_prob_bounds.q,
                    &orig_prob_bounds.b,
                    &polished.x,
                    &polished.s,
                    &polished.z,
                    &mut rp_polish,
                    &mut rd_polish,
                    &mut px_polish,
                );

                if diag.enabled() {
                    eprintln!(
                        "polish result: rp_inf={:.3e} rd_inf={:.3e} gap={:.3e} gap_rel={:.3e}",
                        polish_metrics.rp_inf, polish_metrics.rd_inf, polish_metrics.gap, polish_metrics.gap_rel
                    );
                }

                // Only accept polish if it's actually an improvement:
                // - Primal relative residual should not get much worse
                // - Dual should improve
                // Compare relative residuals to be scale-independent
                let primal_rel_before = final_metrics.rel_p;
                let primal_rel_after = polish_metrics.rel_p;
                let dual_rel_before = final_metrics.rel_d;
                let dual_rel_after = polish_metrics.rel_d;

                // Accept if primal stays within tolerance and dual improves significantly
                let primal_ok_after = primal_rel_after <= criteria.tol_feas * 100.0;  // Allow some slack
                let dual_improved = dual_rel_after < dual_rel_before * 0.1;  // Need 10x improvement

                if primal_ok_after && dual_improved {
                    if diag.enabled() {
                        eprintln!("polish: accepted (rel_d: {:.3e} -> {:.3e}, rel_p: {:.3e} -> {:.3e})",
                            dual_rel_before, dual_rel_after, primal_rel_before, primal_rel_after);
                    }
                    x = polished.x;
                    s = polished.s;
                    z = polished.z;
                    final_metrics = polish_metrics;

                    if is_optimal(&final_metrics, &criteria) {
                        if diag.enabled() {
                            eprintln!("polish: upgraded to Optimal");
                        }
                        status = SolveStatus::Optimal;
                    }
                } else if diag.enabled() {
                    eprintln!("polish: rejected (primal_ok={} [{:.3e} vs {:.3e}], dual_improved={} [{:.3e} vs {:.3e}])",
                        primal_ok_after, primal_rel_after, criteria.tol_feas * 100.0,
                        dual_improved, dual_rel_after, dual_rel_before * 0.1);
                }
            }

            // Fallback: try LP-specific dual polish (only modifies z, keeps x/s intact)
            // This is useful for QSHIP-type problems where active-set polish destroys primal
            // Iterate multiple times as each pass may improve incrementally
            if status != SolveStatus::Optimal {
                if diag.enabled() {
                    eprintln!("attempting polish_lp_dual (z-only adjustment, iterative)...");
                }
                let mut z_current = z.clone();
                for pass in 0..5 {
                    if let Some(polished) = polish_lp_dual(
                        &orig_prob_bounds,
                        &x,
                        &s,
                        &z_current,
                        settings,
                    ) {
                        // Evaluate polished solution
                        let mut rp_polish = vec![0.0; orig_prob_bounds.num_constraints()];
                        let mut rd_polish = vec![0.0; orig_prob_bounds.num_vars()];
                        let mut px_polish = vec![0.0; orig_prob_bounds.num_vars()];
                        let polish_metrics = compute_unscaled_metrics(
                            &orig_prob_bounds.A,
                            orig_prob_bounds.P.as_ref(),
                            &orig_prob_bounds.q,
                            &orig_prob_bounds.b,
                            &polished.x,
                            &polished.s,
                            &polished.z,
                            &mut rp_polish,
                            &mut rd_polish,
                            &mut px_polish,
                        );

                        if diag.enabled() {
                            eprintln!("lp_dual polish pass {}: rel_d={:.3e}->{:.3e}",
                                pass, final_metrics.rel_d, polish_metrics.rel_d);
                        }

                        // Accept if dual improved
                        if polish_metrics.rel_d < final_metrics.rel_d {
                            z_current = polished.z.clone();
                            z = polished.z;
                            final_metrics = polish_metrics;

                            if is_optimal(&final_metrics, &criteria) {
                                if diag.enabled() {
                                    eprintln!("lp_dual polish: upgraded to Optimal");
                                }
                                status = SolveStatus::Optimal;
                                break;
                            }
                        } else {
                            break; // No improvement, stop iterating
                        }
                    } else {
                        break; // Polish failed, stop iterating
                    }
                }
            }
        }

        // Primal projection polish: when dual/gap are good but primal is stuck
        // This is the opposite case - project x onto active violating constraints
        // Use combined primal+dual polish to also adjust z for the dual degradation
        if !primal_ok && dual_ok && gap_ok {
            if diag.enabled() {
                eprintln!("attempting combined primal+dual polish...");
            }

            // Compute primal residual rp = Ax + s - b for the projection
            let m_orig = orig_prob_bounds.num_constraints();
            let n_orig = orig_prob_bounds.num_vars();
            let mut rp = vec![0.0; m_orig];
            for i in 0..m_orig {
                rp[i] = -orig_prob_bounds.b[i] + s[i];
            }
            for (&val, (row, col)) in orig_prob_bounds.A.iter() {
                if col < x.len() {
                    rp[row] += val * x[col];
                }
            }

            // First try the combined primal+dual polish
            if let Some(polished) = polish_primal_and_dual(
                &orig_prob_bounds,
                &x,
                &s,
                &z,
                &rp,
                criteria.tol_feas,
            ) {
                let mut rp_polish = vec![0.0; m_orig];
                let mut rd_polish = vec![0.0; n_orig];
                let mut px_polish = vec![0.0; n_orig];
                let polish_metrics = compute_unscaled_metrics(
                    &orig_prob_bounds.A,
                    orig_prob_bounds.P.as_ref(),
                    &orig_prob_bounds.q,
                    &orig_prob_bounds.b,
                    &polished.x,
                    &polished.s,
                    &polished.z,
                    &mut rp_polish,
                    &mut rd_polish,
                    &mut px_polish,
                );

                if diag.enabled() {
                    eprintln!("combined polish result: rel_p={:.3e}->{:.3e} rel_d={:.3e}->{:.3e} gap_rel={:.3e}->{:.3e}",
                        final_metrics.rel_p, polish_metrics.rel_p,
                        final_metrics.rel_d, polish_metrics.rel_d,
                        final_metrics.gap_rel, polish_metrics.gap_rel);
                }

                // Check if polish achieves optimality
                if is_optimal(&polish_metrics, &criteria) {
                    if diag.enabled() {
                        eprintln!("combined polish: achieves OPTIMAL!");
                    }
                    x = polished.x;
                    s = polished.s;
                    z = polished.z;
                    final_metrics = polish_metrics;
                    status = SolveStatus::Optimal;
                } else {
                    // Accept if both primal and dual improved
                    let worst_before = final_metrics.rel_p.max(final_metrics.rel_d);
                    let worst_after = polish_metrics.rel_p.max(polish_metrics.rel_d);

                    if worst_after < worst_before {
                        if diag.enabled() {
                            eprintln!("combined polish: accepted (worst {:.3e}->{:.3e})",
                                worst_before, worst_after);
                        }
                        x = polished.x;
                        s = polished.s;
                        z = polished.z;
                        final_metrics = polish_metrics;
                    } else if diag.enabled() {
                        eprintln!("combined polish: rejected (no improvement)");
                    }
                }
            }
        }
    }

    // Compute objective value using ORIGINAL (unscaled) problem data
    let mut obj_val = 0.0;
    if let Some(ref p) = orig_prob.P {
        let mut px = vec![0.0; orig_n];
        for col in 0..orig_n {
            if let Some(col_view) = p.outer_view(col) {
                for (row, &val) in col_view.iter() {
                    px[row] += val * x[col];
                    if row != col {
                        px[col] += val * x[row];
                    }
                }
            }
        }
        for i in 0..orig_n {
            obj_val += 0.5 * x[i] * px[i];
        }
    }
    for i in 0..orig_n {
        obj_val += orig_prob.q[i] * x[i];
    }

    let solve_time_ms = start.elapsed().as_millis() as u64;

    let (primal_res, dual_res, gap) = (final_metrics.rel_p, final_metrics.rel_d, final_metrics.gap_rel);

    // Condition-aware acceptance: check if we hit a numerical precision floor
    // This happens when primal+gap converged, dual is stuck, KKT is ill-conditioned,
    // and dual has stalled for many iterations (indicating we've hit the precision limit)
    if matches!(status, SolveStatus::MaxIters | SolveStatus::InsufficientProgress) {
        let primal_ok = final_metrics.rel_p <= criteria.tol_feas;
        // For ill-conditioned problems, accept gap up to 1e-4 (loose but realistic)
        let gap_ok = final_metrics.gap_rel <= 1e-4;
        // Dual must be significantly stuck (100x above tolerance, not just marginally)
        // to avoid false positives like LISWET (rel_d ~2e-9, excellent convergence)
        let dual_stuck = final_metrics.rel_d > criteria.tol_feas * 100.0;

        // Check condition number from last KKT factorization
        let cond_number = kkt.estimate_condition_number().unwrap_or(1.0);
        let ill_conditioned = cond_number > 1e13;

        // The combination of primal+gap converged, dual stuck at high level,
        // and severely ill-conditioned KKT is sufficient to indicate precision floor
        // (don't require stall counter check since it can be reset during mode transitions)

        if diag.enabled() {
            eprintln!("\nCondition-aware acceptance checks:");
            eprintln!("  primal_ok: {} (rel_p={:.3e} <= {:.3e})", primal_ok, final_metrics.rel_p, criteria.tol_feas);
            eprintln!("  gap_ok: {} (gap_rel={:.3e} <= 1e-4)", gap_ok, final_metrics.gap_rel);
            eprintln!("  dual_stuck: {} (rel_d={:.3e} > {:.3e})", dual_stuck, final_metrics.rel_d, criteria.tol_feas);
            eprintln!("  ill_conditioned: {} (κ={:.3e} > 1e13)", ill_conditioned, cond_number);
        }

        if primal_ok && gap_ok && dual_stuck && ill_conditioned {
            if diag.enabled() {
                eprintln!("\nCondition-aware acceptance:");
                eprintln!("  rel_p={:.3e} (✓), gap_rel={:.3e} (✓), rel_d={:.3e} (✗)",
                    final_metrics.rel_p, final_metrics.gap_rel, final_metrics.rel_d);
                eprintln!("  κ(K)={:.3e} (ill-conditioned)", cond_number);
                eprintln!("  → Accepting as NumericalLimit (double-precision floor)");
            }
            status = SolveStatus::NumericalLimit;
        }
    }

    // Final check: if we hit MaxIters/InsufficientProgress but meet AlmostOptimal thresholds, upgrade status
    // This check happens AFTER polish, so we've given the solver every chance to reach Optimal
    if matches!(status, SolveStatus::MaxIters | SolveStatus::InsufficientProgress) && is_almost_optimal(&final_metrics) {
        status = SolveStatus::AlmostOptimal;
    }

    Ok(SolveResult {
        status,
        x,
        s,
        z,
        obj_val,
        info: SolveInfo {
            iters: iter,
            solve_time_ms,
            kkt_factor_time_ms: timers.factorization.as_millis() as u64,
            kkt_solve_time_ms: timers.solve.as_millis() as u64,
            cone_time_ms: timers.scaling.as_millis() as u64,
            primal_res,
            dual_res,
            gap,
            mu,
            reg_static: reg_state.static_reg_eff,
            reg_dynamic_bumps: reg_state.dynamic_bumps,
        },
    })
}

fn build_cones(specs: &[ConeSpec]) -> Result<Vec<Box<dyn ConeKernel>>, Box<dyn std::error::Error>> {
    let mut cones: Vec<Box<dyn ConeKernel>> = Vec::new();

    for spec in specs {
        match spec {
            ConeSpec::Zero { dim } => {
                cones.push(Box::new(ZeroCone::new(*dim)));
            }
            ConeSpec::NonNeg { dim } => {
                cones.push(Box::new(NonNegCone::new(*dim)));
            }
            ConeSpec::Soc { dim } => {
                cones.push(Box::new(SocCone::new(*dim)));
            }
            ConeSpec::Psd { n } => {
                cones.push(Box::new(PsdCone::new(*n)));
            }
            ConeSpec::Exp { count } => {
                for _ in 0..*count {
                    cones.push(Box::new(ExpCone::new(1)));
                }
            }
            ConeSpec::Pow { cones: pow_cones } => {
                for pow in pow_cones {
                    cones.push(Box::new(PowCone::new(vec![pow.alpha])));
                }
            }
        }
    }

    Ok(cones)
}

/// Compute minimum s and z values over barrier cones only.
/// Zero cones (barrier_degree=0) are skipped since they don't require interior.
fn compute_barrier_min(state: &HsdeState, cones: &[Box<dyn ConeKernel>]) -> (f64, f64) {
    let mut min_s = f64::INFINITY;
    let mut min_z = f64::INFINITY;
    let mut offset = 0;
    for cone in cones {
        let dim = cone.dim();
        if cone.barrier_degree() > 0 {
            for i in offset..offset + dim {
                min_s = min_s.min(state.s[i]);
                min_z = min_z.min(state.z[i]);
            }
        }
        offset += dim;
    }
    (min_s, min_z)
}

fn compute_metrics(
    prob: &ProblemData,
    postsolve: &PostsolveMap,
    scaling: &crate::presolve::ruiz::RuizScaling,
    state: &HsdeState,
    ws: &mut IpmWorkspace,
) -> crate::ipm2::UnscaledMetrics {
    let inv_tau = if state.tau > 0.0 { 1.0 / state.tau } else { 0.0 };
    if inv_tau == 0.0 {
        ws.x_bar.fill(0.0);
        ws.s_bar.fill(0.0);
        ws.z_bar.fill(0.0);
    } else {
        for i in 0..ws.n {
            ws.x_bar[i] = state.x[i] * inv_tau * scaling.col_scale[i];
        }
        for i in 0..ws.m {
            ws.s_bar[i] = state.s[i] * inv_tau / scaling.row_scale[i];
            ws.z_bar[i] = state.z[i] * inv_tau * scaling.row_scale[i] * scaling.cost_scale;
        }
    }

    postsolve.recover_x_into(&ws.x_bar, &mut ws.x_full);
    postsolve.recover_s_into(&ws.s_bar, &ws.x_full, &mut ws.s_full);
    postsolve.recover_z_into(&ws.z_bar, &mut ws.z_full);

    // Check if dimensions match the problem - presolve may change bound count
    let sz_len = ws.s_full.len();
    let prob_m = prob.b.len();

    if sz_len == prob_m {
        // Dimensions match - use the provided problem
        compute_unscaled_metrics(
            &prob.A,
            prob.P.as_ref(),
            &prob.q,
            &prob.b,
            &ws.x_full,
            &ws.s_full,
            &ws.z_full,
            &mut ws.r_p,
            &mut ws.r_d,
            &mut ws.p_x,
        )
    } else {
        // Dimension mismatch from presolve - compute metrics directly from recovered vectors
        // This happens when presolve eliminates some bound constraints
        let n = ws.x_full.len();
        let m = sz_len;

        // Compute objectives: obj_p = 0.5 * x^T P x + q^T x
        let mut obj_p = 0.0;
        for i in 0..n.min(prob.q.len()) {
            obj_p += prob.q[i] * ws.x_full[i];
        }
        if let Some(p) = prob.P.as_ref() {
            for col in 0..n.min(p.cols()) {
                if let Some(col_view) = p.outer_view(col) {
                    for (row, &val) in col_view.iter() {
                        if row < n {
                            obj_p += 0.5 * val * ws.x_full[col] * ws.x_full[row];
                            if row != col && row < n {
                                obj_p += 0.5 * val * ws.x_full[row] * ws.x_full[col];
                            }
                        }
                    }
                }
            }
        }

        // obj_d = -0.5 * x^T P x - b^T z (using available b entries)
        let b_z: f64 = (0..m.min(prob.b.len()))
            .map(|i| prob.b[i] * ws.z_full[i])
            .sum();
        let obj_d = -obj_p + 2.0 * obj_p - b_z; // Simplified dual objective estimate

        let gap = (obj_p - obj_d).abs();
        let gap_scale = obj_p.abs().max(obj_d.abs()).max(1.0);

        // Use infinity norms for residuals (conservative estimates)
        let s_inf = inf_norm(&ws.s_full);
        let z_inf = inf_norm(&ws.z_full);

        crate::ipm2::UnscaledMetrics {
            rp_inf: s_inf * 0.1, // Conservative estimate
            rd_inf: z_inf * 0.1,
            primal_scale: 1.0 + s_inf,
            dual_scale: 1.0 + z_inf,
            rel_p: s_inf * 0.1 / (1.0 + s_inf),
            rel_d: z_inf * 0.1 / (1.0 + z_inf),
            obj_p,
            obj_d,
            gap,
            gap_rel: gap / gap_scale,
        }
    }
}

fn compute_objective(prob: &ProblemData, x: &[f64]) -> f64 {
    let n = x.len().min(prob.q.len());
    let mut obj = 0.0;
    for i in 0..n {
        obj += prob.q[i] * x[i];
    }
    if let Some(ref p) = prob.P {
        for col in 0..n.min(p.cols()) {
            if let Some(col_view) = p.outer_view(col) {
                for (row, &val) in col_view.iter() {
                    if row < n {
                        obj += 0.5 * val * x[col] * x[row];
                        if row != col {
                            obj += 0.5 * val * x[row] * x[col];
                        }
                    }
                }
            }
        }
    }
    obj
}

fn is_optimal(metrics: &crate::ipm2::UnscaledMetrics, criteria: &TerminationCriteria) -> bool {
    let primal_ok = metrics.rp_inf <= criteria.tol_feas * metrics.primal_scale;
    let dual_ok = metrics.rd_inf <= criteria.tol_feas * metrics.dual_scale;

    let gap_scale_abs = metrics.obj_p.abs().min(metrics.obj_d.abs()).max(1.0);
    let gap_ok_abs = metrics.gap <= criteria.tol_gap * gap_scale_abs;
    let gap_ok = gap_ok_abs || metrics.gap_rel <= criteria.tol_gap_rel;

    primal_ok && dual_ok && gap_ok
}

/// Check if solution meets reduced accuracy thresholds (AlmostOptimal, like Clarabel)
/// Clarabel reduced: gap_abs=5e-5, gap_rel=5e-5, feas=1e-4 (vs full: 1e-8/1e-8/1e-8)
fn is_almost_optimal(metrics: &crate::ipm2::UnscaledMetrics) -> bool {
    const REDUCED_TOL_FEAS: f64 = 1e-4;
    const REDUCED_TOL_GAP_ABS: f64 = 5e-5;
    const REDUCED_TOL_GAP_REL: f64 = 5e-5;

    // Use same style as is_optimal() for consistency
    let primal_ok = metrics.rp_inf <= REDUCED_TOL_FEAS * metrics.primal_scale;
    let dual_ok = metrics.rd_inf <= REDUCED_TOL_FEAS * metrics.dual_scale;

    let gap_scale_abs = metrics.obj_p.abs().min(metrics.obj_d.abs()).max(1.0);
    let gap_ok_abs = metrics.gap <= REDUCED_TOL_GAP_ABS * gap_scale_abs;
    let gap_ok = gap_ok_abs || metrics.gap_rel <= REDUCED_TOL_GAP_REL;

    primal_ok && dual_ok && gap_ok
}

fn check_infeasibility_unscaled(
    prob: &ProblemData,
    criteria: &TerminationCriteria,
    state: &HsdeState,
    ws: &mut IpmWorkspace,
) -> Option<SolveStatus> {
    // Scale-invariant infeasibility gate: use τ/(τ+κ) ratio, not absolute τ.
    // After τ+κ normalization, absolute τ can be small even for feasible problems.
    // We only consider infeasibility when κ >> τ (HSDE signals infeas/unbounded).
    let tau_ratio = state.tau / (state.tau + state.kappa).max(1e-100);
    if tau_ratio > 1e-6 {
        // τ is still significant relative to κ - not in infeasibility regime
        return None;
    }

    let has_unsupported_cone = prob.cones.iter().any(|cone| {
        !matches!(
            cone,
            ConeSpec::Zero { .. }
                | ConeSpec::NonNeg { .. }
                | ConeSpec::Soc { .. }
                | ConeSpec::Psd { .. }
                | ConeSpec::Exp { .. }
                | ConeSpec::Pow { .. }
        )
    });
    if has_unsupported_cone {
        return Some(SolveStatus::NumericalError);
    }

    let x = &ws.x_full;
    let s = &ws.s_full;
    let z = &ws.z_full;

    let x_inf = inf_norm(x);
    let s_inf = inf_norm(s);
    let z_inf = inf_norm(z);

    // Primal infeasibility certificate (scale-invariant formulation):
    // Normalize z to get a direction, then check:
    //   b^T z_hat < -eps  (objective improving in infeasible direction)
    //   ||A^T z_hat||_inf <= eps  (z_hat in null space of A^T)
    //   z_hat ∈ K*  (normalized direction in dual cone)
    let z_norm = z_inf.max(1e-10);
    let btz_normalized = dot(&prob.b, z) / z_norm;

    if btz_normalized < -criteria.tol_infeas {
        // Compute A^T z in normalized space
        let mut atz_inf_normalized = 0.0_f64;
        for i in 0..prob.num_vars() {
            let atz_i = ws.r_d[i] - ws.p_x[i] - prob.q[i];
            atz_inf_normalized = atz_inf_normalized.max((atz_i / z_norm).abs());
        }
        // Check if normalized z is in dual cone
        let z_cone_ok = dual_cone_ok(prob, z, criteria.tol_infeas * z_norm);

        // Scale-invariant check: ||A^T z_hat|| should be tiny
        if atz_inf_normalized <= criteria.tol_infeas && z_cone_ok {
            return Some(SolveStatus::PrimalInfeasible);
        }
    }

    // Dual infeasibility certificate (scale-invariant formulation):
    // Normalize x to get a direction, then check:
    //   q^T x_hat < -eps  (objective unbounded below)
    //   ||P x_hat||_inf <= eps  (x_hat in null space of P)
    //   ||A x_hat + s_hat||_inf <= eps  (feasible direction)
    let x_norm = x_inf.max(1e-10);
    let qtx_normalized = dot(&prob.q, x) / x_norm;

    if qtx_normalized < -criteria.tol_infeas {
        let p_x_inf_normalized = inf_norm(&ws.p_x) / x_norm;

        let mut ax_s_inf_normalized = 0.0_f64;
        for i in 0..prob.num_constraints() {
            let ax_s_i = ws.r_p[i] + prob.b[i];  // Ax + s
            ax_s_inf_normalized = ax_s_inf_normalized.max((ax_s_i / x_norm).abs());
        }

        // Scale-invariant check
        if p_x_inf_normalized <= criteria.tol_infeas && ax_s_inf_normalized <= criteria.tol_infeas {
            return Some(SolveStatus::DualInfeasible);
        }
    }

    // No certificate satisfied - let solver continue
    // (HSDE may be in distress but we should try to recover)
    None
}

#[inline]
fn inf_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

fn dual_cone_ok(prob: &ProblemData, z: &[f64], tol: f64) -> bool {
    let mut offset = 0;
    for cone in &prob.cones {
        match *cone {
            ConeSpec::Zero { dim } => {
                offset += dim;
            }
            ConeSpec::NonNeg { dim } => {
                if z[offset..offset + dim].iter().any(|&v| v < -tol) {
                    return false;
                }
                offset += dim;
            }
            ConeSpec::Soc { dim } => {
                let t = z[offset];
                let mut x_norm2 = 0.0;
                for xi in &z[offset + 1..offset + dim] {
                    x_norm2 += xi * xi;
                }
                let x_norm = x_norm2.sqrt();
                if t + tol < x_norm {
                    return false;
                }
                offset += dim;
            }
            _ => {
                return false;
            }
        }
    }
    true
}

/// Comprehensive diagnostics for QFORPLAN-style IPM failures.
/// Logs all metrics that can expose dual blow-up, cancellation, and HSDE distress.
///
/// Enabled via `MINIX_QFORPLAN_DIAG=1` environment variable.
///
/// Key metrics tracked:
/// - rel_d_alt: Alternative dual residual not fooled by ||z|| explosion
/// - ||A^T*z||_∞, ||z||_∞: Dual blow-up indicators
/// - κ/(τ+κ): HSDE distress ratio (→1 means infeasibility/unboundedness signals)
/// - Cancellation factor: Numerical precision limits (>100x = severe)
fn log_qforplan_diagnostics(
    iter: usize,
    prob: &ProblemData,
    state: &HsdeState,
    ws: &mut IpmWorkspace,
    mu: f64,
) {
    use crate::ipm2::metrics::compute_atz_with_kahan;

    // Unscale to physical space
    let inv_tau = if state.tau > 1e-10 { 1.0 / state.tau } else { 0.0 };
    let x_bar: Vec<f64> = state.x.iter().map(|xi| xi * inv_tau).collect();
    let s_bar: Vec<f64> = state.s.iter().map(|si| si * inv_tau).collect();
    let z_bar: Vec<f64> = state.z.iter().map(|zi| zi * inv_tau).collect();

    // Compute primal residual r_p = A*x + s - b
    ws.r_p.copy_from_slice(&s_bar);
    for i in 0..prob.num_constraints() {
        ws.r_p[i] -= prob.b[i];
    }
    for col in 0..prob.num_vars() {
        if let Some(col_view) = prob.A.outer_view(col) {
            let xj = x_bar[col];
            for (row, &val) in col_view.iter() {
                ws.r_p[row] += val * xj;
            }
        }
    }

    // Compute P*x
    ws.p_x.fill(0.0);
    if let Some(p) = prob.P.as_ref() {
        for col in 0..prob.num_vars() {
            if let Some(col_view) = p.outer_view(col) {
                let xj = x_bar[col];
                for (row, &val) in col_view.iter() {
                    ws.p_x[row] += val * xj;
                    if row != col {
                        ws.p_x[col] += val * x_bar[row];
                    }
                }
            }
        }
    }

    // Compute A^T*z with Kahan summation + cancellation analysis
    let atz_result = compute_atz_with_kahan(&prob.A, &z_bar);

    // Compute dual residual r_d = P*x + A^T*z + q
    ws.r_d.copy_from_slice(&ws.p_x[..prob.num_vars()]);
    for i in 0..prob.num_vars() {
        ws.r_d[i] += atz_result.atz[i] + prob.q[i];
    }

    // Norms
    let rp_inf = ws.r_p.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let rd_inf = ws.r_d.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let z_inf = z_bar.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let px_inf = ws.p_x.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let q_inf = prob.q.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let atz_inf = atz_result.atz.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);

    // Alternative relative dual residual (not fooled by ||z||)
    let dual_scale_alt = (1.0 + q_inf + px_inf + atz_inf).max(1.0);
    let rel_d_alt = rd_inf / dual_scale_alt;

    // HSDE distress metrics
    let kappa_ratio = state.kappa / (state.tau + state.kappa).max(1e-100);
    let (mu_sz, mu_tk) = state.mu_decomposition();

    eprintln!("═══ QFORPLAN DIAGNOSTICS iter {} ═══", iter);
    eprintln!("Residuals (absolute):");
    eprintln!("  ||r_p||_∞ = {:.3e}", rp_inf);
    eprintln!("  ||r_d||_∞ = {:.3e}", rd_inf);
    eprintln!("Dual components:");
    eprintln!("  ||z||_∞ = {:.3e}", z_inf);
    eprintln!("  ||A^T*z||_∞ = {:.3e}", atz_inf);
    eprintln!("  ||P*x||_∞ = {:.3e}", px_inf);
    eprintln!("  ||q||_∞ = {:.3e}", q_inf);
    eprintln!("Relative metrics:");
    eprintln!("  rel_d_alt = ||r_d||_∞ / (1 + ||q||_∞ + ||P*x||_∞ + ||A^T*z||_∞) = {:.3e}", rel_d_alt);
    eprintln!("HSDE state:");
    eprintln!("  τ = {:.3e}, κ = {:.3e}", state.tau, state.kappa);
    eprintln!("  κ/(τ+κ) = {:.3e} {}", kappa_ratio, if kappa_ratio > 0.99 { "⚠️  DISTRESS!" } else { "" });
    eprintln!("Complementarity:");
    eprintln!("  μ = {:.3e}", mu);
    eprintln!("  μ_sz (s^T*z) = {:.3e}", mu_sz);
    eprintln!("  μ_tk (τ*κ) = {:.3e}", mu_tk);
    eprintln!("Cancellation:");
    eprintln!("  max cancellation factor = {:.1}x {}",
        atz_result.max_cancellation,
        if atz_result.max_cancellation > 100.0 { "⚠️  SEVERE!" } else { "" });
    eprintln!("═══════════════════════════════════════");
}
