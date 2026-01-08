use sprs::TriMat;

use crate::postsolve::{PostsolveMap, RemovedRow, RemovedRowKind, RowMap};
use crate::presolve::bounds::PresolveResult;
use crate::presolve::singleton::detect_singleton_rows;
use crate::problem::{ConeSpec, ProblemData, VarBound};

pub fn eliminate_singleton_rows(prob: &ProblemData) -> PresolveResult {
    let n = prob.num_vars();
    let m = prob.num_constraints();

    let singletons = detect_singleton_rows(&prob.A);
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
                });
                eprintln!("presolve: removing Zero cone row {}: var={}, val={}, rhs={}", row.row, row.col, row.val, rhs);
            }
            ConeSpec::NonNeg { .. } => {
                eprintln!("presolve: skipping NonNeg cone row {}", row.row);
            }
            cone_spec => {
                eprintln!("presolve: skipping {:?} cone row {}", cone_spec, row.row);
            }
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
