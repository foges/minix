use sprs::TriMat;

use crate::postsolve::PostsolveMap;
use crate::problem::{ProblemData, VarBound, VarType};

#[derive(Debug, Clone)]
pub struct PresolveResult {
    pub problem: ProblemData,
    pub postsolve: PostsolveMap,
}

pub fn shift_bounds_and_eliminate_fixed(prob: &ProblemData) -> PresolveResult {
    shift_bounds_and_eliminate_fixed_with_postsolve(prob, PostsolveMap::identity(prob.num_vars()))
}

pub fn shift_bounds_and_eliminate_fixed_with_postsolve(
    prob: &ProblemData,
    postsolve: PostsolveMap,
) -> PresolveResult {
    let n = prob.num_vars();
    let m = prob.num_constraints();

    let Some(bounds) = prob.var_bounds.as_ref() else {
        let mut postsolve_out = PostsolveMap::identity(postsolve.orig_n());
        if let Some(row_map) = postsolve.into_row_map() {
            postsolve_out = postsolve_out.with_row_map(row_map);
        }
        return PresolveResult {
            problem: prob.clone(),
            postsolve: postsolve_out,
        };
    };

    let mut lower: Vec<Option<f64>> = vec![None; n];
    let mut upper: Vec<Option<f64>> = vec![None; n];
    for b in bounds {
        if let Some(l) = b.lower {
            lower[b.var] = Some(lower[b.var].map_or(l, |cur| cur.max(l)));
        }
        if let Some(u) = b.upper {
            upper[b.var] = Some(upper[b.var].map_or(u, |cur| cur.min(u)));
        }
    }

    let fixed_tol = 1e-12;
    let mut fixed = vec![false; n];
    for i in 0..n {
        if let (Some(l), Some(u)) = (lower[i], upper[i]) {
            if (u - l).abs() <= fixed_tol {
                fixed[i] = true;
            }
        }
    }

    let mut shift = vec![0.0; n];
    for i in 0..n {
        if let Some(l) = lower[i] {
            shift[i] = l;
        }
    }

    let mut b_shift = vec![0.0; m];
    for col in 0..n {
        let s = shift[col];
        if s == 0.0 {
            continue;
        }
        if let Some(col_view) = prob.A.outer_view(col) {
            for (row, &val) in col_view.iter() {
                b_shift[row] += val * s;
            }
        }
    }
    let mut b_new = prob.b.clone();
    for i in 0..m {
        b_new[i] -= b_shift[i];
    }

    let mut q_shift = vec![0.0; n];
    if let Some(p) = prob.P.as_ref() {
        for col in 0..n {
            if let Some(col_view) = p.outer_view(col) {
                let shift_col = shift[col];
                for (row, &val) in col_view.iter() {
                    q_shift[row] += val * shift_col;
                    if row != col {
                        q_shift[col] += val * shift[row];
                    }
                }
            }
        }
    }
    let mut q_new = prob.q.clone();
    for i in 0..n {
        q_new[i] += q_shift[i];
    }

    let mut kept_indices = Vec::new();
    let mut col_map = vec![None; n];
    for i in 0..n {
        if !fixed[i] {
            col_map[i] = Some(kept_indices.len());
            kept_indices.push(i);
        }
    }

    let n_keep = kept_indices.len();

    let mut a_tri = TriMat::new((m, n_keep));
    for col in 0..n {
        let Some(new_col) = col_map[col] else {
            continue;
        };
        if let Some(col_view) = prob.A.outer_view(col) {
            for (row, &val) in col_view.iter() {
                a_tri.add_triplet(row, new_col, val);
            }
        }
    }
    let a_new = a_tri.to_csc();

    let p_new = if let Some(p) = prob.P.as_ref() {
        let mut p_tri = TriMat::new((n_keep, n_keep));
        for col in 0..n {
            let Some(new_col) = col_map[col] else {
                continue;
            };
            if let Some(col_view) = p.outer_view(col) {
                for (row, &val) in col_view.iter() {
                    if let Some(new_row) = col_map[row] {
                        p_tri.add_triplet(new_row, new_col, val);
                    }
                }
            }
        }
        Some(p_tri.to_csc())
    } else {
        None
    };

    let mut q_reduced = Vec::with_capacity(n_keep);
    for &orig_idx in &kept_indices {
        q_reduced.push(q_new[orig_idx]);
    }

    let mut bounds_reduced = Vec::new();
    for &orig_idx in &kept_indices {
        let new_lower = lower[orig_idx].map(|_| 0.0);
        let new_upper = upper[orig_idx].map(|u| u - shift[orig_idx]);
        if new_lower.is_some() || new_upper.is_some() {
            let new_var = col_map[orig_idx].expect("kept index must be mapped");
            bounds_reduced.push(VarBound {
                var: new_var,
                lower: new_lower,
                upper: new_upper,
            });
        }
    }

    let integrality_reduced = prob.integrality.as_ref().map(|types| {
        kept_indices.iter().map(|&idx| types[idx]).collect::<Vec<VarType>>()
    });

    let prob_new = ProblemData {
        P: p_new,
        q: q_reduced,
        A: a_new,
        b: b_new,
        cones: prob.cones.clone(),
        var_bounds: if bounds_reduced.is_empty() {
            None
        } else {
            Some(bounds_reduced)
        },
        integrality: integrality_reduced,
    };

    let mut postsolve_out = PostsolveMap::new(postsolve.orig_n(), shift, kept_indices);
    if let Some(row_map) = postsolve.into_row_map() {
        postsolve_out = postsolve_out.with_row_map(row_map);
    }

    PresolveResult {
        problem: prob_new,
        postsolve: postsolve_out,
    }
}
