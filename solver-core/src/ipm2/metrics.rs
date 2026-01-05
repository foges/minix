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
    let x_inf = inf_norm(x_bar);
    let s_inf = inf_norm(s_bar);
    let z_inf = inf_norm(z_bar);

    let primal_scale = (b_inf + x_inf + s_inf).max(1.0);
    let dual_scale = (q_inf + x_inf + z_inf).max(1.0);

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

