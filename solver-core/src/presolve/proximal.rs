//! Proximal regularization for degenerate SDPs.
//!
//! Detects variables with no (or very weak) equality constraints and adds small
//! proximal regularization to stabilize the Newton system.

use sprs::CsMat;

/// Detect "free" columns in A that have very small norm.
/// Returns indices of columns where ||A[:, j]|| <= tol.
pub fn detect_zero_columns(a: &CsMat<f64>, tol: f64) -> Vec<usize> {
    let n = a.cols();
    let mut zero_cols = Vec::new();

    for col in 0..n {
        let col_norm_sq: f64 = a.outer_view(col)
            .map(|v| v.iter().map(|(_, &val)| val * val).sum())
            .unwrap_or(0.0);

        if col_norm_sq.sqrt() <= tol {
            zero_cols.push(col);
        }
    }

    zero_cols
}

/// Detect columns that are "free" in the sense that they have:
/// 1. Zero or very small norm in the equality constraint rows (first `eq_rows` rows of A)
/// 2. Zero or very small objective coefficient
///
/// For SDP problems with identity embedding, we only look at the equality
/// constraint rows, not the identity embedding rows.
///
/// These variables need proximal regularization to prevent wild drift.
pub fn detect_free_variables(
    a: &CsMat<f64>,
    q: &[f64],
    p: Option<&CsMat<f64>>,
    col_norm_tol: f64,
    cost_tol: f64,
) -> Vec<usize> {
    // Use all rows by default
    detect_free_variables_eq(a, q, p, a.rows(), col_norm_tol, cost_tol)
}

/// Detect free variables, only considering the first `eq_rows` rows of A.
/// This is useful for SDP problems where the A matrix has:
/// - First eq_rows: equality constraints (Zero cone)
/// - Remaining rows: identity embedding for PSD cone
pub fn detect_free_variables_eq(
    a: &CsMat<f64>,
    q: &[f64],
    p: Option<&CsMat<f64>>,
    eq_rows: usize,
    col_norm_tol: f64,
    cost_tol: f64,
) -> Vec<usize> {
    let n = a.cols();
    let mut free_vars = Vec::new();

    for col in 0..n {
        // Check A column norm only in equality constraint rows
        let col_norm_sq: f64 = a.outer_view(col)
            .map(|v| v.iter()
                .filter(|(row, _)| *row < eq_rows)  // Only equality rows
                .map(|(_, &val)| val * val)
                .sum())
            .unwrap_or(0.0);

        if col_norm_sq.sqrt() > col_norm_tol {
            continue; // Column has significant constraint contribution
        }

        // Check objective coefficient
        if q[col].abs() > cost_tol {
            continue; // Has significant objective contribution
        }

        // Check if P has diagonal entry
        if let Some(p_mat) = p {
            let p_diag: f64 = p_mat.outer_view(col)
                .and_then(|v| v.iter().find(|(row, _)| *row == col).map(|(_, &val)| val))
                .unwrap_or(0.0);
            if p_diag.abs() > cost_tol {
                continue; // Already has quadratic curvature
            }
        }

        free_vars.push(col);
    }

    free_vars
}

/// Detect columns with zero or very small P-diagonal (LP-like variables).
/// These variables lack quadratic regularization and may have slow dual convergence.
pub fn detect_lp_columns(
    p: Option<&CsMat<f64>>,
    n: usize,
    p_diag_tol: f64,
) -> Vec<usize> {
    let mut lp_cols = Vec::new();

    for col in 0..n {
        let p_diag: f64 = p
            .and_then(|p_mat| {
                p_mat.outer_view(col)
                    .and_then(|v| v.iter().find(|(row, _)| *row == col).map(|(_, &val)| val))
            })
            .unwrap_or(0.0);

        if p_diag.abs() <= p_diag_tol {
            lp_cols.push(col);
        }
    }

    lp_cols
}

/// Create a proximal regularization matrix P_reg = diag(rho) for free variables.
/// Returns a sparse diagonal matrix that can be added to P.
pub fn create_proximal_regularization(
    n: usize,
    free_vars: &[usize],
    rho: f64,
) -> CsMat<f64> {
    use sprs::TriMat;

    let mut triplets = TriMat::new((n, n));
    for &col in free_vars {
        triplets.add_triplet(col, col, rho);
    }

    triplets.to_csr()
}

/// Add proximal regularization to the problem's P matrix.
/// If P is None, creates a new sparse diagonal matrix.
/// Otherwise, adds the regularization to the existing P.
pub fn add_proximal_regularization(
    p: Option<CsMat<f64>>,
    n: usize,
    free_vars: &[usize],
    rho: f64,
) -> Option<CsMat<f64>> {
    if free_vars.is_empty() {
        return p;
    }

    let reg = create_proximal_regularization(n, free_vars, rho);

    match p {
        Some(p_mat) => {
            // Add regularization to existing P
            Some(&p_mat + &reg)
        }
        None => {
            // Create new P from just the regularization
            Some(reg)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::TriMat;

    #[test]
    fn test_detect_zero_columns() {
        // Create a 3x4 matrix with column 2 being zero
        let mut triplets = TriMat::new((3, 4));
        triplets.add_triplet(0, 0, 1.0);
        triplets.add_triplet(1, 1, 2.0);
        // Column 2 is empty (zero)
        triplets.add_triplet(2, 3, 3.0);
        let a: CsMat<f64> = triplets.to_csc();

        let zero_cols = detect_zero_columns(&a, 1e-10);
        assert_eq!(zero_cols, vec![2]);
    }
}
