use sprs::CsMat;

#[derive(Debug, Copy, Clone)]
pub struct UnscaledMetrics {
    pub rp_inf: f64,
    pub rd_inf: f64,
    pub primal_scale: f64,
    pub dual_scale: f64,

    pub rel_p: f64,
    pub rel_d: f64,

    pub obj_p: f64,
    pub obj_d: f64,
    pub gap: f64,
    pub gap_rel: f64,
}

/// Result of A^T*z computation with cancellation analysis.
#[derive(Debug, Clone)]
pub struct AtzResult {
    /// Final A^T*z values (computed with Kahan summation)
    pub atz: Vec<f64>,
    /// Sum of |val * z| for each column (cancellation-free magnitude)
    pub atz_magnitude: Vec<f64>,
    /// Cancellation factor = atz_magnitude / |atz| for each column
    pub cancellation_factor: Vec<f64>,
    /// Maximum cancellation factor across all columns
    pub max_cancellation: f64,
}

#[inline]
fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |acc, &x| acc.max(x.abs()))
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// Compute unscaled metrics.
///
/// This function expects *already unscaled* `x_bar, s_bar, z_bar` (i.e., after:
/// 1) dividing by tau
/// 2) undoing Ruiz scaling).
///
/// It computes:
/// - r_p = A x_bar + s_bar - b
/// - r_d = P x_bar + A^T z_bar + q
/// - objectives + gap
///
/// The caller provides scratch buffers `r_p, r_d, p_x` to avoid allocations.
pub fn compute_unscaled_metrics(
    a: &CsMat<f64>,                  // m×n, CSC
    p_upper: Option<&CsMat<f64>>,    // n×n upper triangle (CSC) or full symmetric
    q: &[f64],
    b: &[f64],
    x_bar: &[f64],
    s_bar: &[f64],
    z_bar: &[f64],
    r_p: &mut [f64],
    r_d: &mut [f64],
    p_x: &mut [f64],
) -> UnscaledMetrics {
    let n = x_bar.len();
    let m = s_bar.len();

    debug_assert_eq!(a.rows(), m);
    debug_assert_eq!(a.cols(), n);
    debug_assert_eq!(b.len(), m);
    debug_assert_eq!(z_bar.len(), m);
    debug_assert_eq!(q.len(), n);
    debug_assert_eq!(r_p.len(), m);
    debug_assert_eq!(r_d.len(), n);
    debug_assert_eq!(p_x.len(), n);

    // r_p = A x + s - b
    r_p.copy_from_slice(s_bar);
    for i in 0..m {
        r_p[i] -= b[i];
    }
    for col in 0..n {
        if let Some(col_view) = a.outer_view(col) {
            let xj = x_bar[col];
            for (row, &val) in col_view.iter() {
                r_p[row] += val * xj;
            }
        }
    }

    // p_x = P x
    p_x.fill(0.0);
    if let Some(p) = p_upper {
        // Treat as symmetric: use stored entries and mirror off-diagonal.
        for col in 0..n {
            if let Some(col_view) = p.outer_view(col) {
                let xj = x_bar[col];
                for (row, &val) in col_view.iter() {
                    p_x[row] += val * xj;
                    if row != col {
                        p_x[col] += val * x_bar[row];
                    }
                }
            }
        }
    }

    // r_d = P x + A^T z + q
    r_d.copy_from_slice(&p_x[..n]);
    for i in 0..n {
        r_d[i] += q[i];
    }
    for col in 0..n {
        if let Some(col_view) = a.outer_view(col) {
            let mut acc = 0.0;
            for (row, &val) in col_view.iter() {
                acc += val * z_bar[row];
            }
            r_d[col] += acc;
        }
    }

    let rp_inf = inf_norm(r_p);
    let rd_inf = inf_norm(r_d);

    let b_inf = inf_norm(b);
    let q_inf = inf_norm(q);

    // Use data-based scaling only; avoid x/s/z magnitude so HSDE scaling
    // cannot hide large absolute residuals (especially on SDP instances).
    let primal_scale = b_inf.max(1.0);
    let dual_scale = q_inf.max(1.0);

    let rel_p = rp_inf / primal_scale;
    let rel_d = rd_inf / dual_scale;

    let xpx = dot(x_bar, p_x);
    let qtx = dot(q, x_bar);
    let btz = dot(b, z_bar);

    let obj_p = 0.5 * xpx + qtx;
    let obj_d = -0.5 * xpx - btz;

    let gap = (obj_p - obj_d).abs();
    let denom = obj_p.abs().max(obj_d.abs()).max(1.0);
    let gap_rel = gap / denom;

    UnscaledMetrics {
        rp_inf,
        rd_inf,
        primal_scale,
        dual_scale,
        rel_p,
        rel_d,
        obj_p,
        obj_d,
        gap,
        gap_rel,
    }
}

/// Compute A^T*z with Kahan (compensated) summation and cancellation analysis.
///
/// This function tracks both the final result and the magnitude of contributions
/// to detect catastrophic cancellation. When many large terms of opposite sign
/// cancel to produce a small result, the cancellation factor will be large (>100).
///
/// **Use case**: Diagnose whether a dual residual floor is due to:
/// - Algorithmic issues (low cancellation factor)
/// - Numerical precision limits (high cancellation factor, e.g., >100x)
///
/// The Kahan summation algorithm maintains a running compensation term to recover
/// low-order bits lost to rounding error, providing more accurate results than
/// naive summation when many terms are accumulated.
pub fn compute_atz_with_kahan(a: &CsMat<f64>, z: &[f64]) -> AtzResult {
    let n = a.cols();
    let m = a.rows();
    debug_assert_eq!(z.len(), m);

    let mut atz = vec![0.0; n];
    let mut atz_compensation = vec![0.0; n];  // Kahan compensation term
    let mut atz_magnitude = vec![0.0; n];     // Cancellation-free magnitude

    for col in 0..n {
        if let Some(col_view) = a.outer_view(col) {
            for (row, &val) in col_view.iter() {
                let contrib = val * z[row];

                // Kahan summation for atz
                let y = contrib - atz_compensation[col];
                let t = atz[col] + y;
                atz_compensation[col] = (t - atz[col]) - y;
                atz[col] = t;

                // Magnitude sum (no cancellation)
                atz_magnitude[col] += contrib.abs();
            }
        }
    }

    // Compute cancellation factor for each variable
    let mut cancellation_factor = vec![1.0; n];
    for i in 0..n {
        if atz[i].abs() > 1e-20 {
            cancellation_factor[i] = atz_magnitude[i] / atz[i].abs();
        } else if atz_magnitude[i] > 1e-20 {
            // Perfect cancellation: large magnitude but tiny result
            cancellation_factor[i] = 1e20;
        }
    }

    let max_cancellation = cancellation_factor.iter()
        .copied()
        .fold(0.0f64, f64::max);

    AtzResult {
        atz,
        atz_magnitude,
        cancellation_factor,
        max_cancellation,
    }
}

/// Decompose dual residual to diagnose which component is causing issues.
/// r_d = P*x + A^T*z + q
/// This helps identify if the problem is:
/// - Objective term (P*x + q)
/// - Dual variable blow-up (A^T*z)
/// - Numerical issues in recovery/scaling
pub fn diagnose_dual_residual(
    a: &CsMat<f64>,
    p_upper: Option<&CsMat<f64>>,
    q: &[f64],
    x_bar: &[f64],
    z_bar: &[f64],
    r_d: &[f64],
    problem_name: &str,
) {
    let n = x_bar.len();
    let m = z_bar.len();

    // Compute P*x (objective gradient term)
    let mut p_x = vec![0.0; n];
    if let Some(p) = p_upper {
        for col in 0..n {
            if let Some(col_view) = p.outer_view(col) {
                let xj = x_bar[col];
                for (row, &val) in col_view.iter() {
                    p_x[row] += val * xj;
                    if row != col {
                        p_x[col] += val * x_bar[row];
                    }
                }
            }
        }
    }

    // Compute g = P*x + q (objective gradient)
    let mut g = p_x.clone();
    for i in 0..n {
        g[i] += q[i];
    }

    // Compute A^T*z with Kahan summation + cancellation analysis
    let atz_result = compute_atz_with_kahan(a, z_bar);
    let atz = &atz_result.atz;

    // Find top 10 dual residual components by magnitude
    let mut indexed: Vec<(usize, f64)> = r_d.iter().enumerate().map(|(i, &v)| (i, v.abs())).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    eprintln!("\n{}", "=".repeat(80));
    eprintln!("DUAL RESIDUAL DECOMPOSITION: {}", problem_name);
    eprintln!("{}", "=".repeat(80));
    eprintln!("Top 10 dual residual components (r_d = P*x + A^T*z + q):");
    eprintln!("{:>5} {:>12} {:>12} {:>12} {:>12} {:>12}",
              "idx", "r_d", "g=Px+q", "A^T*z", "x", "z_max");
    eprintln!("{}", "-".repeat(80));

    for i in 0..10.min(indexed.len()) {
        let idx = indexed[i].0;
        let z_max = if m > 0 {
            z_bar.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()))
        } else {
            0.0
        };
        eprintln!("{:>5} {:>+12.3e} {:>+12.3e} {:>+12.3e} {:>+12.3e} {:>+12.3e}",
                  idx, r_d[idx], g[idx], atz[idx], x_bar[idx], z_max);
    }

    // Summary statistics
    let g_inf = g.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));
    let atz_inf = atz.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));
    let rd_inf = r_d.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));

    eprintln!("{}", "-".repeat(80));
    eprintln!("Summary:");
    eprintln!("  ||r_d||_inf = {:.3e} (total dual residual)", rd_inf);
    eprintln!("  ||g||_inf   = {:.3e} (objective gradient = P*x + q)", g_inf);
    eprintln!("  ||A^T*z||_inf = {:.3e} (dual variable contribution)", atz_inf);

    if atz_inf > g_inf * 10.0 {
        eprintln!("\n⚠️  DIAGNOSIS: Dual blow-up (A^T*z >> g)");
        eprintln!("     Likely causes: dual variables exploding, conditioning issues, or presolve recovery bug");
    } else if g_inf > atz_inf * 10.0 {
        eprintln!("\n⚠️  DIAGNOSIS: Objective gradient dominates (g >> A^T*z)");
        eprintln!("     Likely causes: scaling issues, data magnitude problems");
    } else {
        eprintln!("\n✓  Components are balanced (neither dominates)");
    }

    // Cancellation analysis
    eprintln!("\n{}", "-".repeat(80));
    eprintln!("CANCELLATION ANALYSIS (Kahan summation):");
    eprintln!("  Max cancellation factor: {:.1}x", atz_result.max_cancellation);

    if atz_result.max_cancellation > 100.0 {
        eprintln!("\n⚠️  SEVERE CANCELLATION DETECTED (factor > 100x)");
        eprintln!("     → Dual residual floor is dominated by numerical precision limits");
        eprintln!("     → This is a fundamental double-precision limitation, not a solver bug");
    } else if atz_result.max_cancellation > 10.0 {
        eprintln!("\n⚠️  Moderate cancellation (factor > 10x)");
        eprintln!("     → Some numerical precision loss in A^T*z computation");
    } else {
        eprintln!("\n✓  Low cancellation (factor < 10x)");
        eprintln!("     → Dual residual is not limited by catastrophic cancellation");
    }

    // Show top 5 variables with worst cancellation
    let mut cancel_indexed: Vec<(usize, f64)> = atz_result.cancellation_factor.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    cancel_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    if cancel_indexed[0].1 > 10.0 {
        eprintln!("\n  Top 5 variables with highest cancellation:");
        eprintln!("  {:>5} {:>12} {:>15} {:>15}",
                  "idx", "cancel_x", "A^T*z", "magnitude");
        for i in 0..5.min(cancel_indexed.len()) {
            let idx = cancel_indexed[i].0;
            let factor = cancel_indexed[i].1;
            if factor > 10.0 {
                eprintln!("  {:>5} {:>12.1} {:>+15.3e} {:>15.3e}",
                    idx, factor, atz_result.atz[idx], atz_result.atz_magnitude[idx]);
            }
        }
    }

    eprintln!("{}", "=".repeat(80));
}
