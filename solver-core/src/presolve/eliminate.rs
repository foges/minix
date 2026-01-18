use sprs::TriMat;

use crate::postsolve::{PostsolveMap, RemovedRow, RemovedRowKind, RowMap};
use crate::presolve::bounds::PresolveResult;
use crate::presolve::singleton::detect_singleton_rows_cone_aware;
use crate::problem::{ConeSpec, ProblemData, VarBound};

pub fn eliminate_singleton_rows(prob: &ProblemData) -> PresolveResult {
    let n = prob.num_vars();
    let m = prob.num_constraints();

    // Check if problem has PSD cones - if so, don't eliminate zero cone singletons
    // because the resulting fixed variable elimination breaks SDP structure.
    // The variable x represents a symmetric matrix in svec form, and eliminating
    // one component leads to incorrect dual residual computation.
    let has_psd_cones = prob.cones.iter().any(|c| matches!(c, ConeSpec::Psd { .. }));

    // Check if problem has EXP cones - if so, don't eliminate zero cone singletons
    // because the resulting zero rows in A cause convergence issues.
    // When A has zero rows, s[i] = b[i] becomes a hard constraint that interacts
    // poorly with HSDE rescaling and the BFGS scaling for nonsymmetric cones.
    let has_exp_cones = prob.cones.iter().any(|c| matches!(c, ConeSpec::Exp { .. }));

    // Use cone-aware singleton detection to avoid eliminating rows from multi-dimensional cones
    let singletons = detect_singleton_rows_cone_aware(&prob.A, &prob.cones);
    if singletons.singleton_rows.is_empty() {
        return PresolveResult {
            problem: prob.clone(),
            postsolve: PostsolveMap::identity(n),
        };
    }

    let mut row_to_cone = vec![usize::MAX; m];
    let mut cone_starts = Vec::with_capacity(prob.cones.len());
    let mut offset = 0usize;
    for (cone_idx, cone) in prob.cones.iter().enumerate() {
        let dim = cone.dim();
        cone_starts.push((offset, offset + dim, cone));
        for row in offset..offset + dim {
            row_to_cone[row] = cone_idx;
        }
        offset += dim;
    }

    let mut lower: Vec<Option<f64>> = vec![None; n];
    let mut upper: Vec<Option<f64>> = vec![None; n];
    if let Some(bounds) = prob.var_bounds.as_ref() {
        for b in bounds {
            if let Some(l) = b.lower {
                lower[b.var] = Some(lower[b.var].map_or(l, |cur| cur.max(l)));
            }
            if let Some(u) = b.upper {
                upper[b.var] = Some(upper[b.var].map_or(u, |cur| cur.min(u)));
            }
        }
    }

    let mut remove_row = vec![false; m];
    let mut removed_rows = Vec::new();

    for row in &singletons.singleton_rows {
        let cone_idx = row_to_cone[row.row];
        if cone_idx == usize::MAX {
            continue;
        }
        match &prob.cones[cone_idx] {
            ConeSpec::Zero { .. } => {
                // Skip zero cone singleton elimination when PSD or EXP cones are present
                // to avoid breaking SDP structure or EXP cone convergence
                if has_psd_cones || has_exp_cones {
                    continue;
                }
                if row.val == 0.0 {
                    continue;
                }
                let rhs = prob.b[row.row];
                let fixed = rhs / row.val;
                lower[row.col] = Some(lower[row.col].map_or(fixed, |cur| cur.max(fixed)));
                upper[row.col] = Some(upper[row.col].map_or(fixed, |cur| cur.min(fixed)));
                remove_row[row.row] = true;
                removed_rows.push(RemovedRow {
                    row: row.row,
                    col: row.col,
                    val: row.val,
                    rhs,
                    kind: RemovedRowKind::Zero,
                    a_col_entries: Vec::new(), // Populated after kept_rows is known
                    q_col: prob.q[row.col],
                });
            }
            ConeSpec::NonNeg { .. } => {}
            _ => {}
        }
    }

    let mut kept_rows = Vec::with_capacity(m);
    let mut row_map = vec![None; m];
    let mut new_row = 0usize;
    for row in 0..m {
        if !remove_row[row] {
            row_map[row] = Some(new_row);
            kept_rows.push(row);
            new_row += 1;
        }
    }

    // Populate a_col_entries for removed Zero cone rows (for dual recovery)
    // z[row] = (-q_col - sum_{j in kept} A[j,col]*z[j]) / A[row,col]
    for removed in &mut removed_rows {
        if matches!(removed.kind, RemovedRowKind::Zero) {
            let col = removed.col;
            if let Some(col_view) = prob.A.outer_view(col) {
                for (orig_row, &val) in col_view.iter() {
                    if let Some(kept_idx) = row_map[orig_row] {
                        removed.a_col_entries.push((kept_idx, val));
                    }
                }
            }
        }
    }

    let mut a_tri = TriMat::new((kept_rows.len(), n));
    for col in 0..n {
        if let Some(col_view) = prob.A.outer_view(col) {
            for (row, &val) in col_view.iter() {
                if let Some(new_row_idx) = row_map[row] {
                    a_tri.add_triplet(new_row_idx, col, val);
                }
            }
        }
    }
    let a_new = a_tri.to_csc();

    let mut b_new = Vec::with_capacity(kept_rows.len());
    for &row in &kept_rows {
        b_new.push(prob.b[row]);
    }

    let mut cones_new = Vec::new();
    for (start, end, cone) in cone_starts {
        let mut removed_in_block = 0usize;
        for row in start..end {
            if remove_row[row] {
                removed_in_block += 1;
            }
        }
        let dim = end - start;
        if removed_in_block == 0 {
            cones_new.push(cone.clone());
            continue;
        }

        match cone {
            ConeSpec::Zero { .. } => {
                let new_dim = dim - removed_in_block;
                if new_dim > 0 {
                    cones_new.push(ConeSpec::Zero { dim: new_dim });
                }
            }
            ConeSpec::NonNeg { .. } => {
                let new_dim = dim - removed_in_block;
                if new_dim > 0 {
                    cones_new.push(ConeSpec::NonNeg { dim: new_dim });
                }
            }
            _ => {
                // Singleton elimination only applied to separable cones.
                cones_new.push(cone.clone());
            }
        }
    }

    let mut bounds_new = Vec::new();
    for var in 0..n {
        if lower[var].is_some() || upper[var].is_some() {
            bounds_new.push(VarBound {
                var,
                lower: lower[var],
                upper: upper[var],
            });
        }
    }

    let prob_new = ProblemData {
        P: prob.P.clone(),
        q: prob.q.clone(),
        A: a_new,
        b: b_new,
        cones: cones_new,
        var_bounds: if bounds_new.is_empty() {
            None
        } else {
            Some(bounds_new)
        },
        integrality: prob.integrality.clone(),
    };

    let postsolve = if removed_rows.is_empty() {
        PostsolveMap::identity(n)
    } else {
        let row_map = RowMap::new(m, kept_rows, removed_rows);
        PostsolveMap::identity(n).with_row_map(row_map)
    };

    PresolveResult {
        problem: prob_new,
        postsolve,
    }
}
