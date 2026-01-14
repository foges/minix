//! KKT system builder and solver.
//!
//! This module handles the construction and solution of KKT systems that arise
//! in interior point methods. The KKT matrix has the quasi-definite form:
//!
//! ```text
//! K = [ P + εI    A^T  ]
//!     [ A      -(H + εI)]
//! ```
//!
//! where:
//! - P is the cost Hessian (n×n, PSD)
//! - A is the constraint matrix (m×n)
//! - H is the cone scaling matrix (m×m, block diagonal, SPD)
//! - ε is static regularization
//!
//! The solver implements the two-solve strategy from §5.4.1 of the design doc
//! for efficient predictor-corrector steps.

use super::backend::{BackendError, KktBackend, QdldlBackend};
use super::kkt_trait::KktSolverTrait;
use super::sparse::{SparseCsc, SparseSymmetricCsc};
use crate::scaling::ScalingBlock;
use crate::scaling::nt::jordan_product_apply;
use crate::cones::psd::{mat_to_svec, svec_to_mat};
use nalgebra::DMatrix;
use sprs::TriMat;
use sprs_suitesparse_camd::try_camd;
use std::sync::OnceLock;

fn symm_matvec_upper(a: &SparseCsc, x: &[f64], y: &mut [f64]) {
    y.fill(0.0);
    for (val, (row, col)) in a.iter() {
        y[row] += val * x[col];
        if row != col {
            y[col] += val * x[row];
        }
    }
}

fn kkt_diagnostics_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        // Check new unified MINIX_VERBOSE first (level >= 3 means debug)
        if let Ok(v) = std::env::var("MINIX_VERBOSE") {
            if let Ok(n) = v.parse::<u8>() {
                return n >= 3;
            }
        }
        // Also check MINIX_VERBOSE_KKT for explicit KKT diagnostics
        if let Ok(v) = std::env::var("MINIX_VERBOSE_KKT") {
            return v != "0" && v.to_lowercase() != "false";
        }
        // Legacy: check MINIX_DIAGNOSTICS_KKT
        std::env::var("MINIX_DIAGNOSTICS_KKT")
            .ok()
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false)
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum KktRefineMode {
    Regularized,
    QdldlDebias,
    FullUnreg,
}

fn kkt_refine_mode() -> KktRefineMode {
    static MODE: OnceLock<KktRefineMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        if let Ok(v) = std::env::var("MINIX_KKT_REFINE_MODE") {
            match v.to_lowercase().as_str() {
                "off" | "0" | "false" | "regularized" => KktRefineMode::Regularized,
                "full" | "unreg" | "unregularized" => KktRefineMode::FullUnreg,
                "qdldl" | "debias" | "debiased" => KktRefineMode::QdldlDebias,
                _ => KktRefineMode::QdldlDebias,
            }
        } else if let Ok(v) = std::env::var("MINIX_KKT_UNREG_REFINE") {
            if v == "0" || v.to_lowercase() == "false" {
                KktRefineMode::Regularized
            } else {
                KktRefineMode::FullUnreg
            }
        } else {
            // Default to full unregularized refinement for better accuracy on ill-conditioned problems (SDPs)
            KktRefineMode::FullUnreg
        }
    })
}

fn psd_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        // Check new unified MINIX_VERBOSE first (level >= 4 means trace)
        if let Ok(v) = std::env::var("MINIX_VERBOSE") {
            if let Ok(n) = v.parse::<u8>() {
                return n >= 4;
            }
        }
        // Legacy: check MINIX_DEBUG_PSD_KKT
        std::env::var("MINIX_DEBUG_PSD_KKT").ok().as_deref() == Some("1")
    })
}

fn quad_rep_soc_in_place(
    w: &[f64],
    y: &[f64],
    out: &mut [f64],
    w_circ_y: &mut [f64],
    w_circ_w: &mut [f64],
    temp: &mut [f64],
    w2_circ_y: &mut [f64],
) {
    jordan_product_apply(w, y, w_circ_y);
    jordan_product_apply(w, w, w_circ_w);
    jordan_product_apply(w_circ_y, w, temp);
    for i in 0..w.len() {
        temp[i] *= 2.0;
    }
    jordan_product_apply(w_circ_w, y, w2_circ_y);
    for i in 0..w.len() {
        out[i] = temp[i] - w2_circ_y[i];
    }
}

struct SolveWorkspace {
    rhs_perm: Vec<f64>,
    rhs_perm2: Vec<f64>,
    sol_perm: Vec<f64>,
    kx: Vec<f64>,
    res: Vec<f64>,
    delta: Vec<f64>,
    rhs_x: Vec<f64>,
    rhs_z: Vec<f64>,
    sol_z: Vec<f64>,
}

impl SolveWorkspace {
    fn new(n: usize, m: usize) -> Self {
        let kkt_dim = n + m;
        Self {
            rhs_perm: vec![0.0; kkt_dim],
            rhs_perm2: vec![0.0; kkt_dim],
            sol_perm: vec![0.0; kkt_dim],
            kx: vec![0.0; kkt_dim],
            res: vec![0.0; kkt_dim],
            delta: vec![0.0; kkt_dim],
            rhs_x: vec![0.0; n],
            rhs_z: vec![0.0; m],
            sol_z: vec![0.0; m],
        }
    }
}

#[derive(Clone, Copy)]
enum RhsPermKind {
    Primary,
    Secondary,
}

fn fill_rhs_perm_with_perm(
    perm: Option<&[usize]>,
    n: usize,
    rhs_x: &[f64],
    rhs_z: &[f64],
    rhs_perm: &mut [f64],
) {
    let kkt_dim = n + rhs_z.len();
    if let Some(p) = perm {
        for i in 0..kkt_dim {
            let src = p[i];
            if src < n {
                rhs_perm[i] = rhs_x[src];
            } else {
                rhs_perm[i] = rhs_z[src - n];
            }
        }
    } else {
        rhs_perm[..n].copy_from_slice(rhs_x);
        rhs_perm[n..kkt_dim].copy_from_slice(rhs_z);
    }
}

fn fill_rhs_perm_two_with_perm(
    perm: Option<&[usize]>,
    n: usize,
    rhs_x1: &[f64],
    rhs_z1: &[f64],
    rhs_x2: &[f64],
    rhs_z2: &[f64],
    rhs_perm1: &mut [f64],
    rhs_perm2: &mut [f64],
) {
    let kkt_dim = n + rhs_z1.len();
    if let Some(perm) = perm {
        for (i, &pi) in perm.iter().enumerate().take(kkt_dim) {
            let src = pi;
            if src < n {
                rhs_perm1[i] = rhs_x1[src];
                rhs_perm2[i] = rhs_x2[src];
            } else {
                let src = src - n;
                rhs_perm1[i] = rhs_z1[src];
                rhs_perm2[i] = rhs_z2[src];
            }
        }
    } else {
        rhs_perm1[..n].copy_from_slice(rhs_x1);
        rhs_perm1[n..kkt_dim].copy_from_slice(rhs_z1);
        rhs_perm2[..n].copy_from_slice(rhs_x2);
        rhs_perm2[n..kkt_dim].copy_from_slice(rhs_z2);
    }
}

fn unpermute_solution_with_perm(
    perm_inv: Option<&[usize]>,
    n: usize,
    sol_perm: &[f64],
    sol_x: &mut [f64],
    sol_z: &mut [f64],
) {
    if let Some(p_inv) = perm_inv {
        for i in 0..n {
            sol_x[i] = sol_perm[p_inv[i]];
        }
        for i in 0..sol_z.len() {
            sol_z[i] = sol_perm[p_inv[n + i]];
        }
    } else {
        sol_x.copy_from_slice(&sol_perm[..n]);
        sol_z.copy_from_slice(&sol_perm[n..n + sol_z.len()]);
    }
}

fn prepare_rhs_singleton(
    singleton: &SingletonElim,
    rhs_x: &[f64],
    rhs_z: &[f64],
    ws: &mut SolveWorkspace,
) {
    ws.rhs_x.copy_from_slice(rhs_x);
    for (red_idx, &row) in singleton.kept_rows.iter().enumerate() {
        ws.rhs_z[red_idx] = rhs_z[row];
    }
    for (idx, row) in singleton.singletons.iter().enumerate() {
        let rhs_row = rhs_z[row.row];
        ws.rhs_x[row.col] += row.val * rhs_row * singleton.inv_h[idx];
    }
}

fn expand_solution_z_singleton(
    singleton: &SingletonElim,
    rhs_z: &[f64],
    sol_x: &[f64],
    sol_z: &mut [f64],
    sol_z_reduced: &[f64],
) {
    sol_z.fill(0.0);
    for (red_idx, &row) in singleton.kept_rows.iter().enumerate() {
        sol_z[row] = sol_z_reduced[red_idx];
    }
    for (idx, row) in singleton.singletons.iter().enumerate() {
        let rhs_row = rhs_z[row.row];
        sol_z[row.row] = (row.val * sol_x[row.col] - rhs_row) * singleton.inv_h[idx];
    }
}

fn solve_permuted_with_refinement<B: KktBackend>(
    backend: &B,
    static_reg: f64,
    kkt: Option<&SparseCsc>,
    ws: &mut SolveWorkspace,
    factor: &B::Factorization,
    rhs_kind: RhsPermKind,
    refine_iters: usize,
    reg_diag_sign: Option<&[f64]>,
    tag: Option<&'static str>,
) {
    let rhs_perm = match rhs_kind {
        RhsPermKind::Primary => &ws.rhs_perm,
        RhsPermKind::Secondary => &ws.rhs_perm2,
    };
    let kkt_dim = rhs_perm.len();
    let use_adjust = reg_diag_sign.is_some() && static_reg != 0.0;

    backend.solve(factor, rhs_perm, &mut ws.sol_perm);

    let mut refine_done = 0usize;
    if refine_iters > 0 {
        if let Some(kkt) = kkt {
            for _ in 0..refine_iters {
                symm_matvec_upper(kkt, &ws.sol_perm, &mut ws.kx);
                if static_reg != 0.0 {
                    for i in 0..kkt_dim {
                        ws.kx[i] += static_reg * ws.sol_perm[i];
                    }
                }
                for i in 0..kkt_dim {
                    ws.res[i] = rhs_perm[i] - ws.kx[i];
                }
                if use_adjust {
                    let reg_sign = reg_diag_sign.unwrap();
                    for i in 0..kkt_dim {
                        ws.res[i] += static_reg * reg_sign[i] * ws.sol_perm[i];
                    }
                }

                let res_norm = ws
                    .res
                    .iter()
                    .map(|v| v * v)
                    .sum::<f64>()
                    .sqrt();
                refine_done += 1;
                if !res_norm.is_finite() || res_norm < 1e-12 {
                    break;
                }

                backend.solve(factor, &ws.res, &mut ws.delta);
                for i in 0..kkt_dim {
                    ws.sol_perm[i] += ws.delta[i];
                }
            }
        }
    }

    if let Some(tag) = tag {
        if kkt_diagnostics_enabled() {
            if let Some(kkt) = kkt {
                symm_matvec_upper(kkt, &ws.sol_perm, &mut ws.kx);
                if static_reg != 0.0 {
                    for i in 0..kkt_dim {
                        ws.kx[i] += static_reg * ws.sol_perm[i];
                    }
                }
                for i in 0..kkt_dim {
                    ws.res[i] = rhs_perm[i] - ws.kx[i];
                }
                if use_adjust {
                    let reg_sign = reg_diag_sign.unwrap();
                    for i in 0..kkt_dim {
                        ws.res[i] += static_reg * reg_sign[i] * ws.sol_perm[i];
                    }
                }
                let res_inf = ws
                    .res
                    .iter()
                    .fold(0.0_f64, |acc, v| acc.max(v.abs()));
                eprintln!(
                    "kkt_resid[{tag}] inf={:.3e} refine={}/{} static_reg={:.1e} dyn_bumps={}",
                    res_inf,
                    refine_done,
                    refine_iters,
                    static_reg,
                    backend.dynamic_bumps(),
                );
            }
        }
    }
}

fn update_dense_block_in_place(
    static_reg: f64,
    h: &[f64; 9],
    positions: &[usize],
    data: &mut [f64],
) {
    let mut pos_idx = 0usize;
    for col in 0..3 {
        for row in 0..=col {
            let h_val = h[row * 3 + col];
            let mut val = -h_val;
            if row == col {
                val -= 2.0 * static_reg;
            }
            data[positions[pos_idx]] = val;
            pos_idx += 1;
        }
    }
}

/// Update SOC block in KKT matrix.
///
/// Computes Q_w column by column using Jordan products.
/// Optimized: precomputes w∘w once instead of per-column.
fn update_soc_block_in_place(
    static_reg: f64,
    scratch: &mut SocKktScratch,
    w: &[f64],
    positions: &[usize],
    data: &mut [f64],
) {
    let dim = w.len();
    scratch.ensure_dim(dim);
    let e = &mut scratch.e[..dim];
    let col = &mut scratch.col[..dim];
    let w_circ_y = &mut scratch.w_circ_y[..dim];
    let w_circ_w = &mut scratch.w_circ_w[..dim];
    let temp = &mut scratch.temp[..dim];
    let w2_circ_y = &mut scratch.w2_circ_y[..dim];

    // Precompute w∘w once (was being recomputed every column)
    jordan_product_apply(w, w, w_circ_w);

    let mut pos_idx = 0usize;
    for col_idx in 0..dim {
        e.fill(0.0);
        e[col_idx] = 1.0;

        // w ∘ e_i
        jordan_product_apply(w, e, w_circ_y);

        // 2 * (w ∘ e_i) ∘ w
        jordan_product_apply(w_circ_y, w, temp);
        for i in 0..dim {
            temp[i] *= 2.0;
        }

        // (w ∘ w) ∘ e_i
        jordan_product_apply(w_circ_w, e, w2_circ_y);

        // Q_w(e_i) = 2*(w ∘ e_i) ∘ w - (w ∘ w) ∘ e_i
        for i in 0..dim {
            col[i] = temp[i] - w2_circ_y[i];
        }

        for row_idx in 0..=col_idx {
            let mut val = -col[row_idx];
            if row_idx == col_idx {
                val -= 2.0 * static_reg;
            }
            data[positions[pos_idx]] = val;
            pos_idx += 1;
        }
    }
}

/// Update only the arrowhead entries of a SOC block in the KKT matrix.
/// This is O(dim) instead of O(dim²) for the dense case.
///
/// The arrowhead part of Q_w is:
/// - Q_arrowhead[0,0] = η = w₀² + ||w̄||²
/// - Q_arrowhead[0,j] = 2*w₀*w̄[j-1] for j > 0
/// - Q_arrowhead[i,i] = ρ = w₀² - ||w̄||² for i > 0
///
/// The full Q_w = Q_arrowhead + [0; sqrt(2)*w̄] * [0; sqrt(2)*w̄]ᵀ
/// The rank-1 correction is handled via Sherman-Morrison during solves.
fn update_soc_arrowhead_in_place(
    static_reg: f64,
    w: &[f64],
    first_row_positions: &[usize],
    diag_positions: &[usize],
    data: &mut [f64],
) {
    let dim = w.len();
    debug_assert_eq!(first_row_positions.len(), dim);
    debug_assert_eq!(diag_positions.len(), dim - 1);

    let w0 = w[0];
    let w_bar = &w[1..];
    let w_bar_sq: f64 = w_bar.iter().map(|x| x * x).sum();
    let eta = w0 * w0 + w_bar_sq;
    let rho = w0 * w0 - w_bar_sq;

    // First row: [0,0], [0,1], ..., [0,dim-1]
    // [0,0] entry: -(η) - 2ε
    data[first_row_positions[0]] = -eta - 2.0 * static_reg;

    // [0,j] entries for j > 0: -2*w₀*w̄[j-1]
    for j in 1..dim {
        data[first_row_positions[j]] = -2.0 * w0 * w_bar[j - 1];
    }

    // Diagonal [i,i] for i > 0: -(ρ) - 2ε
    // Note: the full Q_w[i,i] = ρ + 2*w̄[i-1]² but arrowhead stores only ρ
    for i in 0..dim - 1 {
        data[diag_positions[i]] = -rho - 2.0 * static_reg;
    }
}

fn apply_psd_scaling(
    w: &DMatrix<f64>,
    n: usize,
    v: &[f64],
    out: &mut [f64],
) {
    let v_mat = svec_to_mat(v, n);
    let out_mat = w * v_mat * w;
    mat_to_svec(&out_mat, out);
}

fn update_psd_block_in_place(
    static_reg: f64,
    n: usize,
    w_factor: &[f64],
    positions: &[usize],
    data: &mut [f64],
) {
    let dim = n * (n + 1) / 2;
    let w = DMatrix::<f64>::from_row_slice(n, n, w_factor);
    let mut e = vec![0.0; dim];
    let mut col = vec![0.0; dim];
    let mut pos_idx = 0usize;

    let debug = psd_trace_enabled();
    if debug {
        eprintln!("PSD KKT: n={}, dim={}, w_factor={:?}", n, dim, w_factor);
        eprintln!("PSD KKT: W = {:?}", w);
    }

    for col_idx in 0..dim {
        e.fill(0.0);
        e[col_idx] = 1.0;
        apply_psd_scaling(&w, n, &e, &mut col);
        if debug {
            eprintln!("PSD KKT: H * e_{} = {:?}", col_idx, col);
        }
        for row_idx in 0..=col_idx {
            let mut val = -col[row_idx];
            if row_idx == col_idx {
                val -= 2.0 * static_reg;
            }
            data[positions[pos_idx]] = val;
            pos_idx += 1;
        }
    }
}

fn update_h_blocks_in_place(
    static_reg: f64,
    m: usize,
    h_blocks: &[ScalingBlock],
    h_block_positions: &[HBlockPositions],
    kkt_mat: &mut SparseCsc,
    soc_scratch: &mut SocKktScratch,
) {
    let data = kkt_mat.data_mut();

    let mut offset = 0usize;
    for (block, block_pos) in h_blocks.iter().zip(h_block_positions.iter()) {
        let block_dim = match block {
            ScalingBlock::Zero { dim } => *dim,
            ScalingBlock::Diagonal { d } => d.len(),
            ScalingBlock::Dense3x3 { .. } => 3,
            ScalingBlock::SocStructured { w } => w.len(),
            ScalingBlock::PsdStructured { n, .. } => n * (n + 1) / 2,
        };

        match (block, block_pos) {
            (ScalingBlock::Zero { .. }, HBlockPositions::Diagonal { positions }) => {
                assert_eq!(positions.len(), block_dim);
                for i in 0..block_dim {
                    data[positions[i]] = -2.0 * static_reg;
                }
            }
            (ScalingBlock::Diagonal { d }, HBlockPositions::Diagonal { positions }) => {
                assert_eq!(positions.len(), block_dim);
                for i in 0..block_dim {
                    data[positions[i]] = -d[i] - 2.0 * static_reg;
                }
            }
            (ScalingBlock::Zero { .. }, HBlockPositions::UpperTriangle { dim, positions }) => {
                assert_eq!(*dim, block_dim);
                let mut pos_idx = 0usize;
                for col in 0..block_dim {
                    for row in 0..=col {
                        let val = if row == col { -2.0 * static_reg } else { 0.0 };
                        data[positions[pos_idx]] = val;
                        pos_idx += 1;
                    }
                }
            }
            (ScalingBlock::Diagonal { d }, HBlockPositions::UpperTriangle { dim, positions }) => {
                assert_eq!(*dim, block_dim);
                let mut pos_idx = 0usize;
                for col in 0..block_dim {
                    for row in 0..=col {
                        let val = if row == col { -d[row] - 2.0 * static_reg } else { 0.0 };
                        data[positions[pos_idx]] = val;
                        pos_idx += 1;
                    }
                }
            }
            (ScalingBlock::Dense3x3 { h }, HBlockPositions::UpperTriangle { dim, positions }) => {
                assert_eq!(*dim, block_dim);
                update_dense_block_in_place(static_reg, h, positions, data);
            }
            (ScalingBlock::SocStructured { w }, HBlockPositions::UpperTriangle { dim, positions }) => {
                assert_eq!(*dim, block_dim);
                update_soc_block_in_place(static_reg, soc_scratch, w, positions, data);
            }
            (ScalingBlock::SocStructured { w }, HBlockPositions::SocArrowhead { dim, first_row_positions, diag_positions, .. }) => {
                assert_eq!(*dim, block_dim);
                update_soc_arrowhead_in_place(static_reg, w, first_row_positions, diag_positions, data);
            }
            (ScalingBlock::PsdStructured { w_factor, n }, HBlockPositions::UpperTriangle { dim, positions }) => {
                assert_eq!(*dim, block_dim);
                update_psd_block_in_place(static_reg, *n, w_factor, positions, data);
            }
            _ => {
                panic!("H block positions mismatch");
            }
        }

        offset += block_dim;
    }

    assert_eq!(offset, m, "Scaling blocks must cover all {} slacks", m);
}

fn update_h_diagonal_in_place(
    static_reg: f64,
    m: usize,
    h_blocks: &[ScalingBlock],
    h_diag_positions: &[usize],
    kkt_mat: &mut SparseCsc,
) {
    let data = kkt_mat.data_mut();

    let mut offset = 0usize;
    for block in h_blocks {
        match block {
            ScalingBlock::Zero { dim } => {
                for i in 0..*dim {
                    let slack = offset + i;
                    data[h_diag_positions[slack]] = -2.0 * static_reg;
                }
                offset += *dim;
            }
            ScalingBlock::Diagonal { d } => {
                for (i, &di) in d.iter().enumerate() {
                    let slack = offset + i;
                    data[h_diag_positions[slack]] = -di - 2.0 * static_reg;
                }
                offset += d.len();
            }
            _ => panic!("update_h_diagonal_in_place called with non-diagonal ScalingBlock"),
        }
    }

    assert_eq!(offset, m, "Scaling blocks must cover all {} slacks", m);
}

fn update_schur_diagonal(
    singleton: Option<&SingletonElim>,
    p_diag_positions: Option<&[usize]>,
    p_diag_base: &[f64],
    p_diag_schur: &mut [f64],
    kkt_mat: &mut SparseCsc,
) {
    let Some(singleton) = singleton else {
        return;
    };
    let positions = p_diag_positions.expect("P diagonal positions not initialized");
    let data = kkt_mat.data_mut();

    for &col in &singleton.diag_update_cols {
        p_diag_schur[col] = 0.0;
    }
    for (idx, row) in singleton.singletons.iter().enumerate() {
        p_diag_schur[row.col] += row.val * row.val * singleton.inv_h[idx];
    }
    for &col in &singleton.diag_update_cols {
        data[positions[col]] = p_diag_base[col] + p_diag_schur[col];
    }
}

struct SocKktScratch {
    dim: usize,
    e: Vec<f64>,
    col: Vec<f64>,
    w_circ_y: Vec<f64>,
    w_circ_w: Vec<f64>,
    temp: Vec<f64>,
    w2_circ_y: Vec<f64>,
}

impl SocKktScratch {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            e: vec![0.0; dim],
            col: vec![0.0; dim],
            w_circ_y: vec![0.0; dim],
            w_circ_w: vec![0.0; dim],
            temp: vec![0.0; dim],
            w2_circ_y: vec![0.0; dim],
        }
    }

    fn ensure_dim(&mut self, dim: usize) {
        if dim <= self.dim {
            return;
        }
        self.dim = dim;
        self.e.resize(dim, 0.0);
        self.col.resize(dim, 0.0);
        self.w_circ_y.resize(dim, 0.0);
        self.w_circ_w.resize(dim, 0.0);
        self.temp.resize(dim, 0.0);
        self.w2_circ_y.resize(dim, 0.0);
    }
}

enum HBlockPositions {
    Diagonal { positions: Vec<usize> },
    UpperTriangle { dim: usize, positions: Vec<usize> },
    /// Arrowhead structure for SOC: first row + diagonal, with rank-1 correction via Sherman-Morrison
    /// Q_w = Q_arrowhead + [0; sqrt(2)*w_bar] * [0; sqrt(2)*w_bar]^T
    SocArrowhead {
        dim: usize,
        /// Positions of first row: [0,0], [0,1], ..., [0,dim-1]
        first_row_positions: Vec<usize>,
        /// Positions of diagonal excluding [0,0]: [1,1], [2,2], ..., [dim-1,dim-1]
        diag_positions: Vec<usize>,
        /// Global offset of this block in the slack variables
        offset: usize,
    },
}

/// Workspace for Sherman-Morrison/Woodbury rank-1 corrections in SOC arrowhead formulation.
/// For k SOC cones with arrowhead representation, we need to apply:
/// (K_arrowhead + Σᵢ uᵢuᵢᵀ)⁻¹b via Woodbury formula.
struct ShermanMorrisonWorkspace {
    /// Number of SOC cones with arrowhead representation
    num_soc_blocks: usize,
    /// Stored w_bar vectors for each SOC block (the spatial part of w, length dim-1)
    /// These define the rank-1 correction: u = [0; sqrt(2)*w_bar]
    w_bars: Vec<Vec<f64>>,
    /// Offsets of each SOC block in the slack variables
    soc_offsets: Vec<usize>,
    /// Dimensions of each SOC cone
    soc_dims: Vec<usize>,
    /// Workspace: solution vectors z_i = K_arrowhead^{-1} * u_i (stored columnwise)
    z_vecs: Vec<Vec<f64>>,
    /// Workspace: Gram matrix G = I + U^T K_arrowhead^{-1} U (k x k)
    gram_matrix: Vec<f64>,
    /// Workspace: factored Gram matrix (for k x k solve)
    gram_factor: Vec<f64>,
    /// Workspace: temporary vectors
    temp_rhs: Vec<f64>,
    temp_sol: Vec<f64>,
}

impl ShermanMorrisonWorkspace {
    fn new() -> Self {
        Self {
            num_soc_blocks: 0,
            w_bars: Vec::new(),
            soc_offsets: Vec::new(),
            soc_dims: Vec::new(),
            z_vecs: Vec::new(),
            gram_matrix: Vec::new(),
            gram_factor: Vec::new(),
            temp_rhs: Vec::new(),
            temp_sol: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.num_soc_blocks = 0;
        self.w_bars.clear();
        self.soc_offsets.clear();
        self.soc_dims.clear();
        self.z_vecs.clear();
    }

    fn reserve(&mut self, num_soc_blocks: usize, kkt_dim: usize) {
        self.w_bars.reserve(num_soc_blocks);
        self.soc_offsets.reserve(num_soc_blocks);
        self.soc_dims.reserve(num_soc_blocks);
        self.z_vecs.resize(num_soc_blocks, Vec::new());
        for z in &mut self.z_vecs {
            z.resize(kkt_dim, 0.0);
        }
        let k = num_soc_blocks;
        self.gram_matrix.resize(k * k, 0.0);
        self.gram_factor.resize(k * k, 0.0);
        self.temp_rhs.resize(kkt_dim, 0.0);
        self.temp_sol.resize(kkt_dim, 0.0);
    }

    /// Collect arrowhead SOC info from h_blocks and h_block_positions.
    /// Must be called after update_numeric updates the scaling.
    fn collect_arrowhead_info(
        &mut self,
        n: usize,
        h_blocks: &[ScalingBlock],
        h_block_positions: &[HBlockPositions],
    ) {
        self.clear();

        for (block, block_pos) in h_blocks.iter().zip(h_block_positions.iter()) {
            if let (
                ScalingBlock::SocStructured { w },
                HBlockPositions::SocArrowhead { dim, offset, .. },
            ) = (block, block_pos)
            {
                let w_bar: Vec<f64> = w[1..].to_vec();
                self.w_bars.push(w_bar);
                self.soc_offsets.push(*offset);
                self.soc_dims.push(*dim);
            }
        }

        self.num_soc_blocks = self.w_bars.len();

        // Resize workspace vectors
        let kkt_dim = n + h_blocks.iter().map(|b| match b {
            ScalingBlock::Zero { dim } => *dim,
            ScalingBlock::Diagonal { d } => d.len(),
            ScalingBlock::Dense3x3 { .. } => 3,
            ScalingBlock::SocStructured { w } => w.len(),
            ScalingBlock::PsdStructured { n, .. } => n * (n + 1) / 2,
        }).sum::<usize>();

        if self.num_soc_blocks > 0 {
            self.z_vecs.resize(self.num_soc_blocks, Vec::new());
            for z in &mut self.z_vecs {
                z.resize(kkt_dim, 0.0);
            }
            let k = self.num_soc_blocks;
            self.gram_matrix.resize(k * k, 0.0);
            self.gram_factor.resize(k * k, 0.0);
            self.temp_rhs.resize(kkt_dim, 0.0);
            self.temp_sol.resize(kkt_dim, 0.0);
        }
    }

    /// Check if Woodbury correction is needed.
    fn has_arrowhead_blocks(&self) -> bool {
        self.num_soc_blocks > 0
    }

    /// Apply Woodbury correction to the solution.
    ///
    /// The full KKT matrix is: K = K_arrowhead + Σᵢ uᵢuᵢᵀ
    /// We computed y = K_arrowhead⁻¹b, now we need: x = K⁻¹b = y - Z*G⁻¹*(U'y)
    /// where Z[i] = K_arrowhead⁻¹*uᵢ and G = I + U'Z
    ///
    /// For k=1 (single SOC), this simplifies to Sherman-Morrison:
    /// x = y - (u'y)/(1 + u'z) * z
    fn apply_woodbury_correction<B: KktBackend>(
        &mut self,
        backend: &B,
        factor: &B::Factorization,
        n: usize,
        perm: Option<&[usize]>,
        y: &mut [f64],  // Solution in permuted space, modified in place
    ) {
        let k = self.num_soc_blocks;
        if k == 0 {
            return;
        }

        let kkt_dim = y.len();
        let sqrt2 = std::f64::consts::SQRT_2;

        // Step 1: Solve K_arrowhead * z_i = u_i for each i
        // Form u_i in permuted coordinates and solve
        for i in 0..k {
            // Build u_i directly into temp_rhs
            self.temp_rhs.fill(0.0);
            let offset = self.soc_offsets[i];
            for (j, &wb) in self.w_bars[i].iter().enumerate() {
                self.temp_rhs[n + offset + 1 + j] = sqrt2 * wb;
            }

            // Apply permutation to u_i
            if let Some(p) = perm {
                self.temp_sol.copy_from_slice(&self.temp_rhs);
                for j in 0..kkt_dim {
                    self.temp_rhs[j] = self.temp_sol[p[j]];
                }
            }

            // Solve K_arrowhead * z_i = u_i
            backend.solve(factor, &self.temp_rhs, &mut self.z_vecs[i]);
        }

        // Step 2: Form Gram matrix G = I + U'Z
        // G[i,j] = δ_{ij} + u_i' * z_j
        for i in 0..k {
            // Build u_i in permuted coordinates directly
            self.temp_rhs.fill(0.0);
            let offset = self.soc_offsets[i];
            for (jj, &wb) in self.w_bars[i].iter().enumerate() {
                self.temp_rhs[n + offset + 1 + jj] = sqrt2 * wb;
            }
            if let Some(p) = perm {
                self.temp_sol.copy_from_slice(&self.temp_rhs);
                for j in 0..kkt_dim {
                    self.temp_rhs[j] = self.temp_sol[p[j]];
                }
            }

            for j in 0..k {
                let mut dot = 0.0;
                for idx in 0..kkt_dim {
                    dot += self.temp_rhs[idx] * self.z_vecs[j][idx];
                }
                // Note: minus sign because K = K_arrowhead - uu' (the rank-1 correction
                // has negative coefficient in the KKT matrix since we store -Q_w)
                self.gram_matrix[i * k + j] = if i == j { 1.0 - dot } else { -dot };
            }
        }

        // Step 3: Compute c = G⁻¹ * (U' * y)
        // First, compute U' * y (vector of length k)
        let mut uty = vec![0.0; k];
        for i in 0..k {
            // Build u_i in permuted coordinates directly
            self.temp_rhs.fill(0.0);
            let offset = self.soc_offsets[i];
            for (jj, &wb) in self.w_bars[i].iter().enumerate() {
                self.temp_rhs[n + offset + 1 + jj] = sqrt2 * wb;
            }
            if let Some(p) = perm {
                self.temp_sol.copy_from_slice(&self.temp_rhs);
                for j in 0..kkt_dim {
                    self.temp_rhs[j] = self.temp_sol[p[j]];
                }
            }

            let mut dot = 0.0;
            for idx in 0..kkt_dim {
                dot += self.temp_rhs[idx] * y[idx];
            }
            uty[i] = dot;
        }

        // Solve G * c = U'y where G = I - U'Z (for negative rank-1 correction)
        let c = if k == 1 {
            // Sherman-Morrison for (A - uu')⁻¹: c = (u'y) / (1 - u'z)
            vec![uty[0] / self.gram_matrix[0]]
        } else {
            // Small k×k solve via Gaussian elimination
            solve_small_system(&self.gram_matrix, &uty, k)
        };

        // Step 4: y = y + Z * c (plus sign for negative rank-1 correction)
        // (A - UU')⁻¹b = A⁻¹b + Z * G⁻¹ * U'y
        for i in 0..k {
            for idx in 0..kkt_dim {
                y[idx] += self.z_vecs[i][idx] * c[i];
            }
        }
    }
}

/// Solve a small dense k×k system Ax = b via Gaussian elimination with partial pivoting.
fn solve_small_system(a: &[f64], b: &[f64], k: usize) -> Vec<f64> {
    if k == 0 {
        return vec![];
    }
    if k == 1 {
        return vec![b[0] / a[0]];
    }

    // Copy A and b for in-place elimination
    let mut aug: Vec<f64> = vec![0.0; k * (k + 1)];
    for i in 0..k {
        for j in 0..k {
            aug[i * (k + 1) + j] = a[i * k + j];
        }
        aug[i * (k + 1) + k] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..k {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col * (k + 1) + col].abs();
        for row in col + 1..k {
            let val = aug[row * (k + 1) + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        // Swap rows
        if max_row != col {
            for j in 0..=k {
                let tmp = aug[col * (k + 1) + j];
                aug[col * (k + 1) + j] = aug[max_row * (k + 1) + j];
                aug[max_row * (k + 1) + j] = tmp;
            }
        }

        // Eliminate
        let pivot = aug[col * (k + 1) + col];
        if pivot.abs() < 1e-14 {
            // Nearly singular - return zero vector
            return vec![0.0; k];
        }
        for row in col + 1..k {
            let factor = aug[row * (k + 1) + col] / pivot;
            for j in col..=k {
                aug[row * (k + 1) + j] -= factor * aug[col * (k + 1) + j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; k];
    for col in (0..k).rev() {
        let mut sum = aug[col * (k + 1) + k];
        for j in col + 1..k {
            sum -= aug[col * (k + 1) + j] * x[j];
        }
        x[col] = sum / aug[col * (k + 1) + col];
    }

    x
}

struct SingletonRowInfo {
    row: usize,
    col: usize,
    val: f64,
    block_idx: usize,
    block_offset: usize,
}

enum BlockMap {
    Drop,
    KeepAll { reduced_idx: usize },
    KeepSubset { reduced_idx: usize, kept: Vec<usize> },
}

struct ReducedScaling {
    blocks: Vec<ScalingBlock>,
    block_maps: Vec<BlockMap>,
}

impl ReducedScaling {
    fn new(h_blocks: &[ScalingBlock], remove_row: &[bool]) -> Self {
        let mut blocks = Vec::new();
        let mut block_maps = Vec::with_capacity(h_blocks.len());

        let mut offset = 0usize;
        for block in h_blocks {
            let block_dim = match block {
                ScalingBlock::Zero { dim } => *dim,
                ScalingBlock::Diagonal { d } => d.len(),
                ScalingBlock::Dense3x3 { .. } => 3,
                ScalingBlock::SocStructured { w } => w.len(),
                ScalingBlock::PsdStructured { n, .. } => n * (n + 1) / 2,
            };

            let mut kept = Vec::new();
            for i in 0..block_dim {
                if !remove_row[offset + i] {
                    kept.push(i);
                }
            }

            let map = if kept.is_empty() {
                BlockMap::Drop
            } else if kept.len() == block_dim {
                let reduced_idx = blocks.len();
                blocks.push(match block {
                    ScalingBlock::Zero { dim } => ScalingBlock::Zero { dim: *dim },
                    ScalingBlock::Diagonal { d } => {
                        ScalingBlock::Diagonal { d: vec![0.0; d.len()] }
                    }
                    ScalingBlock::Dense3x3 { h } => ScalingBlock::Dense3x3 { h: *h },
                    ScalingBlock::SocStructured { w } => ScalingBlock::SocStructured {
                        w: vec![0.0; w.len()],
                    },
                    ScalingBlock::PsdStructured { w_factor, n } => ScalingBlock::PsdStructured {
                        w_factor: vec![0.0; w_factor.len()],
                        n: *n,
                    },
                });
                BlockMap::KeepAll { reduced_idx }
            } else {
                let reduced_idx = blocks.len();
                let reduced_block = match block {
                    ScalingBlock::Zero { .. } => ScalingBlock::Zero { dim: kept.len() },
                    ScalingBlock::Diagonal { .. } => ScalingBlock::Diagonal { d: vec![0.0; kept.len()] },
                    _ => panic!("Singleton elimination only supports diagonal cone blocks"),
                };
                blocks.push(reduced_block);
                BlockMap::KeepSubset { reduced_idx, kept }
            };

            block_maps.push(map);
            offset += block_dim;
        }

        Self { blocks, block_maps }
    }

    fn update_from_full(&mut self, full: &[ScalingBlock]) {
        for (full_idx, block) in full.iter().enumerate() {
            match &self.block_maps[full_idx] {
                BlockMap::Drop => {}
                BlockMap::KeepAll { reduced_idx } => {
                    let reduced = &mut self.blocks[*reduced_idx];
                    match (reduced, block) {
                        (ScalingBlock::Zero { .. }, ScalingBlock::Zero { .. }) => {}
                        (ScalingBlock::Diagonal { d: out }, ScalingBlock::Diagonal { d }) => {
                            out.copy_from_slice(d);
                        }
                        (ScalingBlock::Dense3x3 { h: out }, ScalingBlock::Dense3x3 { h }) => {
                            *out = *h;
                        }
                        (ScalingBlock::SocStructured { w: out }, ScalingBlock::SocStructured { w }) => {
                            out.copy_from_slice(w);
                        }
                        (
                            ScalingBlock::PsdStructured { w_factor: out, .. },
                            ScalingBlock::PsdStructured { w_factor, .. },
                        ) => {
                            out.copy_from_slice(w_factor);
                        }
                        _ => panic!("Reduced scaling block mismatch"),
                    }
                }
                BlockMap::KeepSubset { reduced_idx, kept } => {
                    let reduced = &mut self.blocks[*reduced_idx];
                    match (reduced, block) {
                        (ScalingBlock::Zero { .. }, ScalingBlock::Zero { .. }) => {}
                        (ScalingBlock::Diagonal { d: out }, ScalingBlock::Diagonal { d }) => {
                            for (out_idx, &full_idx) in kept.iter().enumerate() {
                                out[out_idx] = d[full_idx];
                            }
                        }
                        _ => panic!("Reduced scaling subset only supported for diagonal blocks"),
                    }
                }
            }
        }
    }
}

struct SingletonElim {
    kept_rows: Vec<usize>,
    singletons: Vec<SingletonRowInfo>,
    inv_h: Vec<f64>,
    diag_update_cols: Vec<usize>,
    reduced_a: SparseCsc,
    reduced_scaling: ReducedScaling,
}

impl SingletonElim {
    fn build(a: &SparseCsc, h_blocks: &[ScalingBlock]) -> Option<Self> {
        let m = a.rows();
        let n = a.cols();

        // Singleton elimination only works with diagonal/zero blocks.
        // For SOC/PSD/Dense3x3, the scaling block type can change during iteration
        // (e.g., SOC falls back to Diagonal when NT fails), which would cause a
        // mismatch with the ReducedScaling structure created here.
        let has_non_diagonal = h_blocks.iter().any(|b| {
            matches!(
                b,
                ScalingBlock::SocStructured { .. }
                    | ScalingBlock::PsdStructured { .. }
                    | ScalingBlock::Dense3x3 { .. }
            )
        });
        if has_non_diagonal {
            return None;
        }

        let partition = crate::presolve::singleton::detect_singleton_rows(a);
        if partition.singleton_rows.is_empty() {
            return None;
        }

        let mut row_block = vec![0usize; m];
        let mut row_offset = vec![0usize; m];
        let mut block_eliminable = Vec::with_capacity(h_blocks.len());

        let mut offset = 0usize;
        for (block_idx, block) in h_blocks.iter().enumerate() {
            let block_dim = match block {
                ScalingBlock::Zero { dim } => *dim,
                ScalingBlock::Diagonal { d } => d.len(),
                ScalingBlock::Dense3x3 { .. } => 3,
                ScalingBlock::SocStructured { w } => w.len(),
                ScalingBlock::PsdStructured { n, .. } => n * (n + 1) / 2,
            };

            let eliminable = matches!(block, ScalingBlock::Diagonal { .. });
            block_eliminable.push(eliminable);
            for i in 0..block_dim {
                row_block[offset + i] = block_idx;
                row_offset[offset + i] = i;
            }
            offset += block_dim;
        }
        assert_eq!(offset, m, "Scaling blocks must cover all {} slacks", m);

        let mut remove_row = vec![false; m];
        let mut singletons = Vec::new();

        for row in partition.singleton_rows {
            if row.val == 0.0 {
                continue;
            }
            let block_idx = row_block[row.row];
            if !block_eliminable[block_idx] {
                continue;
            }
            remove_row[row.row] = true;
            singletons.push(SingletonRowInfo {
                row: row.row,
                col: row.col,
                val: row.val,
                block_idx,
                block_offset: row_offset[row.row],
            });
        }

        if singletons.is_empty() {
            return None;
        }

        let mut kept_rows = Vec::with_capacity(m - singletons.len());
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
            if let Some(col_view) = a.outer_view(col) {
                for (row, &val) in col_view.iter() {
                    if let Some(new_row_idx) = row_map[row] {
                        a_tri.add_triplet(new_row_idx, col, val);
                    }
                }
            }
        }
        let reduced_a = a_tri.to_csc();

        let mut col_seen = vec![false; n];
        let mut diag_update_cols = Vec::new();
        for row in &singletons {
            if !col_seen[row.col] {
                col_seen[row.col] = true;
                diag_update_cols.push(row.col);
            }
        }

        let reduced_scaling = ReducedScaling::new(h_blocks, &remove_row);
        let singleton_len = singletons.len();

        Some(Self {
            kept_rows,
            singletons,
            inv_h: vec![0.0; singleton_len],
            diag_update_cols,
            reduced_a,
            reduced_scaling,
        })
    }

    fn update_scaling_from_full(&mut self, h_blocks: &[ScalingBlock]) {
        self.reduced_scaling.update_from_full(h_blocks);
    }

    fn update_inv_h(&mut self, h_blocks: &[ScalingBlock], static_reg: f64) {
        for (idx, row) in self.singletons.iter().enumerate() {
            let h = match &h_blocks[row.block_idx] {
                ScalingBlock::Diagonal { d } => d[row.block_offset],
                ScalingBlock::Zero { .. } => 0.0,
                _ => panic!("Singleton elimination encountered non-diagonal H block"),
            };
            let h_eff = h + static_reg;
            if !h_eff.is_finite() || h_eff <= 0.0 {
                panic!("Invalid H for singleton elimination: {}", h_eff);
            }
            self.inv_h[idx] = 1.0 / h_eff;
        }
    }
}

/// KKT system solver.
///
/// Manages the construction, factorization, and solution of KKT systems
/// arising in the IPM algorithm.
pub struct KktSolverImpl<B: KktBackend> {
    /// Problem dimensions
    n: usize, // Number of variables
    m: usize, // Number of constraints in the reduced KKT system
    m_full: usize, // Number of constraints in the original problem

    /// Sparse backend
    backend: B,

    /// Workspace for KKT matrix construction
    kkt_mat: Option<SparseCsc>,

    /// Static regularization
    static_reg: f64,

    /// Fill-reducing permutation (new index -> old index)
    perm: Option<Vec<usize>>,

    /// Inverse permutation (old index -> new index)
    perm_inv: Option<Vec<usize>>,

    /// Fast-path: positions of diagonal entries of the -(H + 2εI) block inside `kkt_mat`.
    /// Indexed by slack row `0..m` in the original (unpermuted) ordering.
    h_diag_positions: Option<Vec<usize>>,

    /// Workspace to make repeated solves allocation-free.
    solve_ws: SolveWorkspace,

    /// Regularization sign pattern (permuted) for unregularized refinement.
    reg_diag_sign: Vec<f64>,

    /// Cached KKT positions for H block updates (used for non-diagonal blocks).
    h_block_positions: Option<Vec<HBlockPositions>>,

    /// Scratch space for SOC block updates.
    soc_scratch: SocKktScratch,

    /// Optional singleton-row elimination data.
    singleton: Option<SingletonElim>,

    /// Cached P diagonal values (base) and positions for singleton Schur updates.
    p_diag_base: Vec<f64>,
    p_diag_positions: Option<Vec<usize>>,
    p_diag_schur: Vec<f64>,

    /// Sherman-Morrison workspace for SOC arrowhead rank-1 corrections.
    sm_workspace: ShermanMorrisonWorkspace,

    /// Whether to use arrowhead sparsity for SOC blocks (default: true for dim >= threshold)
    use_soc_arrowhead: bool,
}

impl<B: KktBackend> KktSolverImpl<B> {
    /// Create a new KKT solver.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of primal variables
    /// * `m` - Number of constraints (slack dimension)
    /// * `static_reg` - Static diagonal regularization
    /// * `dynamic_reg_min_pivot` - Dynamic regularization threshold
    pub fn new(n: usize, m: usize, static_reg: f64, dynamic_reg_min_pivot: f64) -> Self {
        Self::new_internal(
            n,
            m,
            m,
            static_reg,
            dynamic_reg_min_pivot,
            None,
        )
    }

    /// Create a new KKT solver with singleton-row Schur elimination enabled.
    pub fn new_with_singleton_elimination(
        n: usize,
        m: usize,
        static_reg: f64,
        dynamic_reg_min_pivot: f64,
        a: &SparseCsc,
        h_blocks: &[ScalingBlock],
    ) -> Self {
        let singleton = SingletonElim::build(a, h_blocks);
        if let Some(ref se) = singleton {
            if std::env::var("MINIX_DIAGNOSTICS").ok().as_deref() == Some("1") {
                eprintln!(
                    "kkt presolve: singleton elimination enabled: m_full={} m_reduced={} eliminated={} diag_update_cols={}",
                    m,
                    se.kept_rows.len(),
                    se.singletons.len(),
                    se.diag_update_cols.len()
                );
            }
        }
        if let Some(singleton) = singleton {
            let m_reduced = singleton.kept_rows.len();
            Self::new_internal(
                n,
                m,
                m_reduced,
                static_reg,
                dynamic_reg_min_pivot,
                Some(singleton),
            )
        } else {
            Self::new(n, m, static_reg, dynamic_reg_min_pivot)
        }
    }

    fn new_internal(
        n: usize,
        m_full: usize,
        m_reduced: usize,
        static_reg: f64,
        dynamic_reg_min_pivot: f64,
        singleton: Option<SingletonElim>,
    ) -> Self {
        let kkt_dim = n + m_reduced;
        let backend = B::new(kkt_dim, static_reg, dynamic_reg_min_pivot);

        Self {
            n,
            m: m_reduced,
            m_full,
            backend,
            kkt_mat: None,
            static_reg,
            perm: None,
            perm_inv: None,
            h_diag_positions: None,
            solve_ws: SolveWorkspace::new(n, m_reduced),
            reg_diag_sign: vec![0.0; kkt_dim],
            h_block_positions: None,
            soc_scratch: SocKktScratch::new(0),
            singleton,
            p_diag_base: vec![0.0; n],
            p_diag_positions: None,
            p_diag_schur: vec![0.0; n],
            sm_workspace: ShermanMorrisonWorkspace::new(),
            use_soc_arrowhead: true, // Enable by default
        }
    }

    /// Return the current static regularization value.
    pub fn static_reg(&self) -> f64 {
        self.static_reg
    }

    /// Update the static regularization value (used in KKT assembly + LDL).
    pub fn set_static_reg(&mut self, static_reg: f64) -> Result<(), BackendError> {
        self.static_reg = static_reg;
        self.backend.set_static_reg(static_reg)?;
        Ok(())
    }

    /// Increase static regularization to at least `min_static_reg`.
    pub fn bump_static_reg(&mut self, min_static_reg: f64) -> Result<bool, BackendError> {
        if min_static_reg > self.static_reg {
            self.set_static_reg(min_static_reg)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn compute_camd_perm(&self, kkt: &SparseCsc) -> Result<(Vec<usize>, Vec<usize>), BackendError> {
        let perm = try_camd(kkt.structure_view())
            .map_err(|e| BackendError::Message(format!("Ordering failed: {}", e)))?;
        Ok((perm.vec(), perm.inv_vec()))
    }

    /// Build the KKT matrix K = [[P + εI, A^T], [A, -(H + εI)]].
    ///
    /// This assembles the augmented system matrix from the problem data
    /// and current scaling matrix H.
    ///
    /// Note: QDLDL will add static_reg to all diagonal entries, so we assemble
    /// the (2,2) block as -(H + 2*ε) to get -(H + ε) after QDLDL's regularization.
    ///
    /// # Arguments
    ///
    /// * `p` - Cost Hessian P (n×n, upper triangle, optional)
    /// * `a` - Constraint matrix A (m×n)
    /// * `h_blocks` - Scaling matrix H as a list of diagonal blocks
    ///
    /// # Returns
    ///
    /// The KKT matrix in CSC format (upper triangle only).
    pub fn build_kkt_matrix(
        &self,
        p: Option<&SparseSymmetricCsc>,
        a: &SparseCsc,
        h_blocks: &[ScalingBlock],
    ) -> SparseCsc {
        self.build_kkt_matrix_with_perm(self.perm_inv.as_deref(), p, a, h_blocks)
    }

    fn build_kkt_matrix_with_perm(
        &self,
        perm: Option<&[usize]>,
        p: Option<&SparseSymmetricCsc>,
        a: &SparseCsc,
        h_blocks: &[ScalingBlock],
    ) -> SparseCsc {
        assert_eq!(a.rows(), self.m);
        assert_eq!(a.cols(), self.n);

        let kkt_dim = self.n + self.m;
        let mut tri = TriMat::new((kkt_dim, kkt_dim));
        let map_index = |idx: usize| perm.map_or(idx, |p| p[idx]);
        let add_triplet = |row: usize, col: usize, val: f64, tri: &mut TriMat<f64>| {
            let r = map_index(row);
            let c = map_index(col);
            if r <= c {
                tri.add_triplet(r, c, val);
            } else {
                tri.add_triplet(c, r, val);
            }
        };

        // ===================================================================
        // Top-left block: P (n×n, upper triangle) + regularization
        // ===================================================================
        if let Some(p_mat) = p {
            assert_eq!(p_mat.rows(), self.n);
            assert_eq!(p_mat.cols(), self.n);

            for (val, (row, col)) in p_mat.iter() {
                if row <= col {
                    // Only upper triangle
                    add_triplet(row, col, *val, &mut tri);
                }
            }
        }

        // Add regularization to all diagonal entries in the top-left block.
        // For LPs (P=None) or sparse QPs with missing diagonals, this ensures
        // the KKT matrix is well-conditioned. QDLDL may add additional dynamic
        // regularization during factorization if needed.
        for i in 0..self.n {
            add_triplet(i, i, self.static_reg, &mut tri);
        }

        // ===================================================================
        // Top-right block: A^T (stored as upper triangle of full matrix)
        // Since K is symmetric, we store A^T in the upper triangle.
        // Entry K[i, n+j] = A[j, i] for i < n, j < m
        // ===================================================================
        for (val, (row_a, col_a)) in a.iter() {
            // A[row_a, col_a] corresponds to K[col_a, n + row_a]
            // We want col >= row for upper triangle
            let kkt_row = col_a;
            let kkt_col = self.n + row_a;

            add_triplet(kkt_row, kkt_col, *val, &mut tri);
        }

        // ===================================================================
        // Bottom-right block: -H (m×m, block diagonal)
        // H is stored as a list of diagonal blocks. We assemble it here.
        // ===================================================================
        let mut offset = 0;
        for h_block in h_blocks {
            let block_dim = match h_block {
                ScalingBlock::Zero { dim } => *dim,
                ScalingBlock::Diagonal { d } => d.len(),
                ScalingBlock::Dense3x3 { .. } => 3,
                ScalingBlock::SocStructured { w } => w.len(),
                ScalingBlock::PsdStructured { n, .. } => n * (n + 1) / 2,
            };

            // Apply -(H + 2ε*I) to this block
            // QDLDL will add +ε later, giving us -(H + ε) as desired for quasi-definiteness
            match h_block {
                ScalingBlock::Zero { dim } => {
                    // For Zero cone (equality constraints), H = 0
                    // We want -(0 + ε) = -ε after QDLDL adds +ε
                    // So we assemble -2ε here
                    for i in 0..*dim {
                        let kkt_idx = self.n + offset + i;
                        add_triplet(kkt_idx, kkt_idx, -2.0 * self.static_reg, &mut tri);
                    }
                }
                ScalingBlock::Diagonal { d } => {
                    // -(H + 2ε) for diagonal scaling
                    for i in 0..d.len() {
                        let kkt_idx = self.n + offset + i;
                        add_triplet(kkt_idx, kkt_idx, -d[i] - 2.0 * self.static_reg, &mut tri);
                    }
                }
                ScalingBlock::Dense3x3 { h } => {
                    // -(H + 2ε*I) as a dense 3×3 block (upper triangle)
                    for i in 0..3 {
                        for j in i..3 {
                            let kkt_row = self.n + offset + i;
                            let kkt_col = self.n + offset + j;
                            let idx = i * 3 + j; // row-major storage
                            let mut val = -h[idx];
                            if i == j {
                                val -= 2.0 * self.static_reg;
                            }
                            add_triplet(kkt_row, kkt_col, val, &mut tri);
                        }
                    }
                }
                ScalingBlock::SocStructured { w } => {
                    // For SOC, the scaling matrix is H(w) = quadratic representation P(w)
                    // For large SOC, use arrowhead sparsity: Q_w = Q_arrowhead + 2*w̄*w̄ᵀ
                    // where Q_arrowhead is sparse (first row + diagonal) and rank-1 correction
                    // is handled via Sherman-Morrison during solves.
                    let dim = w.len();

                    // Use arrowhead for large SOC (dim >= 8), dense for small SOC
                    // Small cones don't benefit from arrowhead overhead
                    let use_arrowhead = self.use_soc_arrowhead && dim >= 8;

                    if use_arrowhead {
                        // Arrowhead pattern: first row + diagonal only
                        // Q_w[0,0] = η = w₀² + ||w̄||²
                        // Q_w[0,j] = 2*w₀*w̄[j-1] for j > 0
                        // Q_w[i,i] = ρ = w₀² - ||w̄||² for i > 0 (arrowhead part)
                        // Off-diagonal (i,j), i,j > 0 handled by Sherman-Morrison
                        let w0 = w[0];
                        let w_bar = &w[1..];
                        let w_bar_sq: f64 = w_bar.iter().map(|x| x * x).sum();
                        let eta = w0 * w0 + w_bar_sq;
                        let rho = w0 * w0 - w_bar_sq;

                        // First row: [0,0], [0,1], ..., [0,dim-1]
                        let kkt_row0 = self.n + offset;
                        // [0,0] entry
                        add_triplet(kkt_row0, kkt_row0, -eta - 2.0 * self.static_reg, &mut tri);
                        // [0,j] entries for j > 0
                        for j in 1..dim {
                            let kkt_col = self.n + offset + j;
                            let val = -2.0 * w0 * w_bar[j - 1];
                            add_triplet(kkt_row0, kkt_col, val, &mut tri);
                        }

                        // Diagonal [i,i] for i > 0
                        for i in 1..dim {
                            let kkt_idx = self.n + offset + i;
                            // Arrowhead diagonal: -ρ - 2ε
                            // Note: the full Q_w[i,i] = ρ + 2*w̄[i-1]² but we put only ρ here
                            // The 2*w̄[i-1]² part is handled by Sherman-Morrison
                            add_triplet(kkt_idx, kkt_idx, -rho - 2.0 * self.static_reg, &mut tri);
                        }
                    } else {
                        // Dense pattern for small SOC
                        let mut e_i = vec![0.0; dim];
                        let mut col_i = vec![0.0; dim];
                        for i in 0..dim {
                            e_i.fill(0.0);
                            e_i[i] = 1.0;
                            col_i.fill(0.0);
                            crate::scaling::nt::quad_rep_apply(w, &e_i, &mut col_i);

                            for j in 0..=i {
                                let kkt_row = self.n + offset + j;
                                let kkt_col = self.n + offset + i;
                                let mut val = -col_i[j];
                                if i == j {
                                    val -= 2.0 * self.static_reg;
                                }
                                add_triplet(kkt_row, kkt_col, val, &mut tri);
                            }
                        }
                    }
                }
                ScalingBlock::PsdStructured { .. } => {
                    let (n_psd, w_factor) = match h_block {
                        ScalingBlock::PsdStructured { n, w_factor } => (*n, w_factor),
                        _ => unreachable!(),
                    };
                    let dim = n_psd * (n_psd + 1) / 2;
                    let w = DMatrix::<f64>::from_row_slice(n_psd, n_psd, w_factor);
                    let mut e_i = vec![0.0; dim];
                    let mut col_i = vec![0.0; dim];
                    for i in 0..dim {
                        e_i.fill(0.0);
                        e_i[i] = 1.0;
                        apply_psd_scaling(&w, n_psd, &e_i, &mut col_i);
                        for j in 0..=i {
                            let kkt_row = self.n + offset + j;
                            let kkt_col = self.n + offset + i;
                            let mut val = -col_i[j];
                            if i == j {
                                val -= 2.0 * self.static_reg;
                            }
                            add_triplet(kkt_row, kkt_col, val, &mut tri);
                        }
                    }
                }
            }

            offset += block_dim;
        }

        assert_eq!(offset, self.m, "Scaling blocks must cover all {} slacks", self.m);

        tri.to_csc()
    }

    fn compute_h_diag_positions(&self, kkt: &SparseCsc) -> Vec<usize> {
        let kkt_dim = self.n + self.m;
        assert_eq!(kkt.rows(), kkt_dim);
        assert_eq!(kkt.cols(), kkt_dim);

        let indptr = kkt.indptr();
        let col_ptr = indptr.raw_storage();
        let row_idx = kkt.indices();

        let mut positions = vec![0usize; self.m];

        for slack in 0..self.m {
            let orig_idx = self.n + slack;
            let col = if let Some(p_inv) = &self.perm_inv {
                p_inv[orig_idx]
            } else {
                orig_idx
            };

            let start = col_ptr[col];
            let end = col_ptr[col + 1];

            let mut found = None;
            for idx in start..end {
                if row_idx[idx] == col {
                    found = Some(idx);
                    break;
                }
            }

            positions[slack] = found.unwrap_or_else(|| {
                panic!("KKT matrix missing diagonal entry at column {}", col);
            });
        }

        positions
    }

    fn compute_p_diag_positions(&self, kkt: &SparseCsc) -> Vec<usize> {
        let kkt_dim = self.n + self.m;
        assert_eq!(kkt.rows(), kkt_dim);
        assert_eq!(kkt.cols(), kkt_dim);

        let indptr = kkt.indptr();
        let col_ptr = indptr.raw_storage();
        let row_idx = kkt.indices();

        let mut positions = vec![0usize; self.n];

        for var in 0..self.n {
            let orig_idx = var;
            let col = if let Some(p_inv) = &self.perm_inv {
                p_inv[orig_idx]
            } else {
                orig_idx
            };

            let start = col_ptr[col];
            let end = col_ptr[col + 1];

            let mut found = None;
            for idx in start..end {
                if row_idx[idx] == col {
                    found = Some(idx);
                    break;
                }
            }

            positions[var] = found.unwrap_or_else(|| {
                panic!("KKT matrix missing diagonal entry at column {}", col);
            });
        }

        positions
    }

    fn fill_p_diag_base(&mut self, p: Option<&SparseSymmetricCsc>) {
        self.p_diag_base.fill(0.0);
        if let Some(p_mat) = p {
            assert_eq!(p_mat.rows(), self.n);
            assert_eq!(p_mat.cols(), self.n);
            for (val, (row, col)) in p_mat.iter() {
                if row == col {
                    self.p_diag_base[row] += *val;
                }
            }
        }
    }

    fn map_kkt_index(&self, idx: usize) -> usize {
        self.perm_inv.as_ref().map_or(idx, |p| p[idx])
    }

    fn find_kkt_position(&self, kkt: &SparseCsc, row: usize, col: usize) -> usize {
        let row_m = self.map_kkt_index(row);
        let col_m = self.map_kkt_index(col);
        let (r, c) = if row_m <= col_m {
            (row_m, col_m)
        } else {
            (col_m, row_m)
        };

        let indptr = kkt.indptr();
        let col_ptr = indptr.raw_storage();
        let row_idx = kkt.indices();

        let start = col_ptr[c];
        let end = col_ptr[c + 1];
        for idx in start..end {
            if row_idx[idx] == r {
                return idx;
            }
        }

        panic!("KKT matrix missing entry at ({}, {})", r, c);
    }

    fn compute_h_block_positions(
        &self,
        kkt: &SparseCsc,
        h_blocks: &[ScalingBlock],
    ) -> Vec<HBlockPositions> {
        let diag_positions = self.compute_h_diag_positions(kkt);
        let mut positions: Vec<HBlockPositions> = Vec::with_capacity(h_blocks.len());

        let mut offset = 0usize;
        for block in h_blocks {
            let block_dim = match block {
                ScalingBlock::Zero { dim } => *dim,
                ScalingBlock::Diagonal { d } => d.len(),
                ScalingBlock::Dense3x3 { .. } => 3,
                ScalingBlock::SocStructured { w } => w.len(),
                ScalingBlock::PsdStructured { n, .. } => n * (n + 1) / 2,
            };

            match block {
                ScalingBlock::Zero { .. } | ScalingBlock::Diagonal { .. } => {
                    positions.push(HBlockPositions::Diagonal {
                        positions: diag_positions[offset..offset + block_dim].to_vec(),
                    });
                }
                ScalingBlock::Dense3x3 { .. } => {
                    let mut block_positions = Vec::with_capacity(block_dim * (block_dim + 1) / 2);
                    for col in 0..block_dim {
                        let orig_col = self.n + offset + col;
                        for row in 0..=col {
                            let orig_row = self.n + offset + row;
                            block_positions.push(self.find_kkt_position(kkt, orig_row, orig_col));
                        }
                    }
                    positions.push(HBlockPositions::UpperTriangle {
                        dim: block_dim,
                        positions: block_positions,
                    });
                }
                ScalingBlock::SocStructured { w } => {
                    let dim = w.len();
                    let use_arrowhead = self.use_soc_arrowhead && dim >= 8;

                    if use_arrowhead {
                        // Arrowhead pattern: first row + diagonal (excluding [0,0] which is in first row)
                        // First row positions: [0,0], [0,1], ..., [0,dim-1]
                        let mut first_row_positions = Vec::with_capacity(dim);
                        let orig_row0 = self.n + offset;
                        for col in 0..dim {
                            let orig_col = self.n + offset + col;
                            first_row_positions.push(self.find_kkt_position(kkt, orig_row0, orig_col));
                        }

                        // Diagonal positions: [1,1], [2,2], ..., [dim-1,dim-1]
                        // Note: [0,0] is already in first_row_positions[0]
                        let diag_pos = diag_positions[offset + 1..offset + dim].to_vec();

                        positions.push(HBlockPositions::SocArrowhead {
                            dim,
                            first_row_positions,
                            diag_positions: diag_pos,
                            offset,
                        });
                    } else {
                        // Dense pattern for small SOC
                        let mut block_positions = Vec::with_capacity(block_dim * (block_dim + 1) / 2);
                        for col in 0..block_dim {
                            let orig_col = self.n + offset + col;
                            for row in 0..=col {
                                let orig_row = self.n + offset + row;
                                block_positions.push(self.find_kkt_position(kkt, orig_row, orig_col));
                            }
                        }
                        positions.push(HBlockPositions::UpperTriangle {
                            dim: block_dim,
                            positions: block_positions,
                        });
                    }
                }
                ScalingBlock::PsdStructured { .. } => {
                    let mut block_positions = Vec::with_capacity(block_dim * (block_dim + 1) / 2);
                    for col in 0..block_dim {
                        let orig_col = self.n + offset + col;
                        for row in 0..=col {
                            let orig_row = self.n + offset + row;
                            block_positions.push(self.find_kkt_position(kkt, orig_row, orig_col));
                        }
                    }
                    positions.push(HBlockPositions::UpperTriangle {
                        dim: block_dim,
                        positions: block_positions,
                    });
                }
            }

            offset += block_dim;
        }

        assert_eq!(offset, self.m, "Scaling blocks must cover all {} slacks", self.m);
        positions
    }


    /// Initialize the solver with the KKT matrix sparsity pattern.
    ///
    /// Performs symbolic factorization, which only needs to be done once
    /// if the sparsity pattern doesn't change.
    pub fn initialize(
        &mut self,
        p: Option<&SparseSymmetricCsc>,
        a: &SparseCsc,
        h_blocks: &[ScalingBlock],
    ) -> Result<(), BackendError> {
        if let Some(singleton) = self.singleton.as_mut() {
            singleton.update_scaling_from_full(h_blocks);
        }
        self.fill_p_diag_base(p);

        let (a_use, h_use) = if let Some(singleton) = self.singleton.as_ref() {
            (&singleton.reduced_a, singleton.reduced_scaling.blocks.as_slice())
        } else {
            (a, h_blocks)
        };

        // Step 1: Build unpermuted matrix for CAMD analysis
        let kkt_unpermuted = self.build_kkt_matrix_with_perm(None, p, a_use, h_use);

        // Step 2: Compute fill-reducing permutation
        let (perm, perm_inv) = self.compute_camd_perm(&kkt_unpermuted)?;

        // Step 3: Build correct matrix and set permutation
        let kkt = if perm.iter().enumerate().all(|(i, &pi)| i == pi) {
            // Identity permutation - reuse unpermuted matrix (fast path)
            self.perm = None;
            self.perm_inv = None;
            kkt_unpermuted
        } else {
            // Non-identity permutation - must rebuild with permutation applied
            // CRITICAL: Set perm_inv BEFORE calling build_kkt_matrix so it uses the permutation
            self.perm = Some(perm);
            self.perm_inv = Some(perm_inv);
            self.build_kkt_matrix(p, a_use, h_use)
        };

        // Step 4: Symbolic factorization on the (possibly permuted) matrix
        self.backend.symbolic_factorization(&kkt)?;
        self.kkt_mat = Some(kkt);
        self.h_diag_positions = None;
        self.h_block_positions = None;
        self.p_diag_positions = None;
        if self.singleton.is_some() {
            let kkt_ref = self.kkt_mat.as_ref().expect("KKT matrix not initialized");
            self.p_diag_positions = Some(self.compute_p_diag_positions(kkt_ref));
        }
        Ok(())
    }

    /// Factor the KKT system.
    ///
    /// Performs numeric factorization with the current values of P, A, and H.
    /// The sparsity pattern must match the one from initialize().
    pub fn factor(
        &mut self,
        p: Option<&SparseSymmetricCsc>,
        a: &SparseCsc,
        h_blocks: &[ScalingBlock],
    ) -> Result<B::Factorization, BackendError> {
        self.update_numeric(p, a, h_blocks)?;
        self.factorize()
    }

    /// Update the numeric values in the cached KKT matrix without factorization.
    pub fn update_numeric(
        &mut self,
        p: Option<&SparseSymmetricCsc>,
        a: &SparseCsc,
        h_blocks: &[ScalingBlock],
    ) -> Result<(), BackendError> {
        if let Some(singleton) = self.singleton.as_mut() {
            singleton.update_scaling_from_full(h_blocks);
            singleton.update_inv_h(h_blocks, self.static_reg);
        }

        let need_p_diag_positions = self.singleton.is_some() && self.p_diag_positions.is_none();
        if need_p_diag_positions {
            self.fill_p_diag_base(p);
        }

        let (a_use, h_use) = if let Some(singleton) = self.singleton.as_ref() {
            (&singleton.reduced_a, singleton.reduced_scaling.blocks.as_slice())
        } else {
            (a, h_blocks)
        };

        let diag_h = h_use
            .iter()
            .all(|b| matches!(b, ScalingBlock::Zero { .. } | ScalingBlock::Diagonal { .. }));

        if diag_h {
            if self.kkt_mat.is_none() {
                // Fallback: build once if initialize() was not called.
                self.kkt_mat = Some(self.build_kkt_matrix(p, a_use, h_use));
            }
            if self.h_diag_positions.is_none() {
                let kkt_ref = self.kkt_mat.as_ref().expect("KKT matrix not initialized");
                self.h_diag_positions = Some(self.compute_h_diag_positions(kkt_ref));
            }
            if need_p_diag_positions {
                let kkt_ref = self.kkt_mat.as_ref().expect("KKT matrix not initialized");
                self.p_diag_positions = Some(self.compute_p_diag_positions(kkt_ref));
            }

            {
                let kkt_mat = self.kkt_mat.as_mut().expect("KKT matrix not initialized");
                let h_diag_positions = self
                    .h_diag_positions
                    .as_ref()
                    .expect("H diagonal positions not initialized");
                update_h_diagonal_in_place(
                    self.static_reg,
                    self.m,
                    h_use,
                    h_diag_positions,
                    kkt_mat,
                );
                update_schur_diagonal(
                    self.singleton.as_ref(),
                    self.p_diag_positions.as_deref(),
                    &self.p_diag_base,
                    &mut self.p_diag_schur,
                    kkt_mat,
                );
            }

            return Ok(());
        }

        // General path: reuse KKT pattern and update cone blocks in place.
        if self.kkt_mat.is_none() {
            // Fallback: build once if initialize() was not called.
            self.kkt_mat = Some(self.build_kkt_matrix(p, a_use, h_use));
        }
        if self.h_block_positions.is_none() {
            let kkt_ref = self.kkt_mat.as_ref().expect("KKT matrix not initialized");
            self.h_block_positions = Some(self.compute_h_block_positions(kkt_ref, h_use));
        }
        if need_p_diag_positions {
            let kkt_ref = self.kkt_mat.as_ref().expect("KKT matrix not initialized");
            self.p_diag_positions = Some(self.compute_p_diag_positions(kkt_ref));
        }

        {
            let kkt_mat = self.kkt_mat.as_mut().expect("KKT matrix not initialized");
            let h_block_positions = self
                .h_block_positions
                .as_ref()
                .expect("H block positions not initialized");
            update_h_blocks_in_place(
                self.static_reg,
                self.m,
                h_use,
                h_block_positions,
                kkt_mat,
                &mut self.soc_scratch,
            );
            update_schur_diagonal(
                self.singleton.as_ref(),
                self.p_diag_positions.as_deref(),
                &self.p_diag_base,
                &mut self.p_diag_schur,
                kkt_mat,
            );
        }

        // Collect arrowhead SOC info for Sherman-Morrison correction during solves
        if let Some(h_block_positions) = self.h_block_positions.as_ref() {
            self.sm_workspace.collect_arrowhead_info(self.n, h_use, h_block_positions);
        }

        Ok(())
    }

    /// Factorize the cached KKT matrix after an update.
    pub fn factorize(&mut self) -> Result<B::Factorization, BackendError> {
        let kkt_ref = self
            .kkt_mat
            .as_ref()
            .ok_or_else(|| BackendError::Message("KKT matrix not initialized".to_string()))?;
        self.backend.numeric_factorization(kkt_ref)
    }

    /// Estimate condition number of the KKT matrix from factorization diagonal.
    /// Returns None if factorization not yet done or not supported by backend.
    pub fn estimate_condition_number(&self) -> Option<f64> {
        self.backend.estimate_condition_number()
    }

    /// Solve a single KKT system: K * [dx; dz] = [rhs_x; rhs_z].
    ///
    /// # Arguments
    ///
    /// * `factor` - Factorization from factor()
    /// * `rhs_x` - Right-hand side for x block (length n)
    /// * `rhs_z` - Right-hand side for z block (length m)
    /// * `sol_x` - Solution for x block (output, length n)
    /// * `sol_z` - Solution for z block (output, length m)
    pub fn solve(
        &mut self,
        factor: &B::Factorization,
        rhs_x: &[f64],
        rhs_z: &[f64],
        sol_x: &mut [f64],
        sol_z: &mut [f64],
    ) {
        self.solve_with_refinement(factor, rhs_x, rhs_z, sol_x, sol_z, 0, None);
    }

    /// Solve with optional iterative refinement.
    pub fn solve_refined(
        &mut self,
        factor: &B::Factorization,
        rhs_x: &[f64],
        rhs_z: &[f64],
        sol_x: &mut [f64],
        sol_z: &mut [f64],
        refine_iters: usize,
    ) {
        self.solve_with_refinement(factor, rhs_x, rhs_z, sol_x, sol_z, refine_iters, None);
    }

    /// Solve with optional iterative refinement and diagnostic tag.
    pub fn solve_refined_tagged(
        &mut self,
        factor: &B::Factorization,
        rhs_x: &[f64],
        rhs_z: &[f64],
        sol_x: &mut [f64],
        sol_z: &mut [f64],
        refine_iters: usize,
        tag: &'static str,
    ) {
        self.solve_with_refinement(
            factor,
            rhs_x,
            rhs_z,
            sol_x,
            sol_z,
            refine_iters,
            Some(tag),
        );
    }

    fn solve_with_refinement(
        &mut self,
        factor: &B::Factorization,
        rhs_x: &[f64],
        rhs_z: &[f64],
        sol_x: &mut [f64],
        sol_z: &mut [f64],
        refine_iters: usize,
        tag: Option<&'static str>,
    ) {
        assert_eq!(rhs_x.len(), self.n);
        assert_eq!(sol_x.len(), self.n);
        let static_reg = self.static_reg;
        let refine_mode = kkt_refine_mode();
        let reg_diag_sign = if refine_mode != KktRefineMode::Regularized && static_reg != 0.0 {
            self.fill_reg_diag_sign(refine_mode);
            Some(self.reg_diag_sign.as_slice())
        } else {
            None
        };
        let perm = self.perm.as_deref();
        let perm_inv = self.perm_inv.as_deref();
        let kkt = self.kkt_mat.as_ref();
        let n = self.n;

        // Check if we need Woodbury correction for arrowhead SOC blocks
        let need_woodbury = self.sm_workspace.has_arrowhead_blocks();

        if let Some(singleton) = self.singleton.as_ref() {
            assert_eq!(rhs_z.len(), self.m_full);
            assert_eq!(sol_z.len(), self.m_full);
            prepare_rhs_singleton(singleton, rhs_x, rhs_z, &mut self.solve_ws);
            fill_rhs_perm_with_perm(perm, self.n, &self.solve_ws.rhs_x, &self.solve_ws.rhs_z, &mut self.solve_ws.rhs_perm);
            solve_permuted_with_refinement(
                &self.backend,
                static_reg,
                kkt,
                &mut self.solve_ws,
                factor,
                RhsPermKind::Primary,
                refine_iters,
                reg_diag_sign,
                tag,
            );

            // Apply Woodbury correction for arrowhead SOC blocks
            if need_woodbury {
                self.sm_workspace.apply_woodbury_correction(
                    &self.backend,
                    factor,
                    n,
                    perm,
                    &mut self.solve_ws.sol_perm,
                );
            }

            unpermute_solution_with_perm(perm_inv, self.n, &self.solve_ws.sol_perm, sol_x, &mut self.solve_ws.sol_z);
            expand_solution_z_singleton(singleton, rhs_z, sol_x, sol_z, &self.solve_ws.sol_z);
        } else {
            assert_eq!(rhs_z.len(), self.m);
            assert_eq!(sol_z.len(), self.m);
            fill_rhs_perm_with_perm(perm, self.n, rhs_x, rhs_z, &mut self.solve_ws.rhs_perm);
            solve_permuted_with_refinement(
                &self.backend,
                static_reg,
                kkt,
                &mut self.solve_ws,
                factor,
                RhsPermKind::Primary,
                refine_iters,
                reg_diag_sign,
                tag,
            );

            // Apply Woodbury correction for arrowhead SOC blocks
            if need_woodbury {
                self.sm_workspace.apply_woodbury_correction(
                    &self.backend,
                    factor,
                    n,
                    perm,
                    &mut self.solve_ws.sol_perm,
                );
            }

            unpermute_solution_with_perm(perm_inv, self.n, &self.solve_ws.sol_perm, sol_x, sol_z);
        }
    }

    /// Two-solve strategy for predictor-corrector (§5.4.1 of design doc).
    ///
    /// Solves two systems with the same KKT matrix:
    /// K * [dx1; dz1] = [rhs_x1; rhs_z1]
    /// K * [dx2; dz2] = [rhs_x2; rhs_z2]
    ///
    /// This is more efficient than calling solve() twice because the
    /// factorization is reused and both RHS vectors are permuted together.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_two_rhs(
        &mut self,
        factor: &B::Factorization,
        rhs_x1: &[f64],
        rhs_z1: &[f64],
        rhs_x2: &[f64],
        rhs_z2: &[f64],
        sol_x1: &mut [f64],
        sol_z1: &mut [f64],
        sol_x2: &mut [f64],
        sol_z2: &mut [f64],
    ) {
        self.solve_two_rhs_with_refinement(
            factor,
            rhs_x1,
            rhs_z1,
            rhs_x2,
            rhs_z2,
            sol_x1,
            sol_z1,
            sol_x2,
            sol_z2,
            0,
            None,
            None,
        );
    }

    /// Two-solve strategy with iterative refinement.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_two_rhs_refined(
        &mut self,
        factor: &B::Factorization,
        rhs_x1: &[f64],
        rhs_z1: &[f64],
        rhs_x2: &[f64],
        rhs_z2: &[f64],
        sol_x1: &mut [f64],
        sol_z1: &mut [f64],
        sol_x2: &mut [f64],
        sol_z2: &mut [f64],
        refine_iters: usize,
    ) {
        self.solve_two_rhs_with_refinement(
            factor,
            rhs_x1,
            rhs_z1,
            rhs_x2,
            rhs_z2,
            sol_x1,
            sol_z1,
            sol_x2,
            sol_z2,
            refine_iters,
            None,
            None,
        );
    }

    /// Two-solve strategy with iterative refinement and diagnostic tags.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_two_rhs_refined_tagged(
        &mut self,
        factor: &B::Factorization,
        rhs_x1: &[f64],
        rhs_z1: &[f64],
        rhs_x2: &[f64],
        rhs_z2: &[f64],
        sol_x1: &mut [f64],
        sol_z1: &mut [f64],
        sol_x2: &mut [f64],
        sol_z2: &mut [f64],
        refine_iters: usize,
        tag1: &'static str,
        tag2: &'static str,
    ) {
        self.solve_two_rhs_with_refinement(
            factor,
            rhs_x1,
            rhs_z1,
            rhs_x2,
            rhs_z2,
            sol_x1,
            sol_z1,
            sol_x2,
            sol_z2,
            refine_iters,
            Some(tag1),
            Some(tag2),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_two_rhs_with_refinement(
        &mut self,
        factor: &B::Factorization,
        rhs_x1: &[f64],
        rhs_z1: &[f64],
        rhs_x2: &[f64],
        rhs_z2: &[f64],
        sol_x1: &mut [f64],
        sol_z1: &mut [f64],
        sol_x2: &mut [f64],
        sol_z2: &mut [f64],
        refine_iters: usize,
        tag1: Option<&'static str>,
        tag2: Option<&'static str>,
    ) {
        assert_eq!(rhs_x1.len(), self.n);
        assert_eq!(rhs_x2.len(), self.n);
        assert_eq!(sol_x1.len(), self.n);
        assert_eq!(sol_x2.len(), self.n);
        let static_reg = self.static_reg;
        let refine_mode = kkt_refine_mode();
        let reg_diag_sign = if refine_mode != KktRefineMode::Regularized && static_reg != 0.0 {
            self.fill_reg_diag_sign(refine_mode);
            Some(self.reg_diag_sign.as_slice())
        } else {
            None
        };
        let perm = self.perm.as_deref();
        let perm_inv = self.perm_inv.as_deref();
        let kkt = self.kkt_mat.as_ref();
        let n = self.n;

        // Check if we need Woodbury correction for arrowhead SOC blocks
        let need_woodbury = self.sm_workspace.has_arrowhead_blocks();

        if let Some(singleton) = self.singleton.as_ref() {
            assert_eq!(rhs_z1.len(), self.m_full);
            assert_eq!(rhs_z2.len(), self.m_full);
            assert_eq!(sol_z1.len(), self.m_full);
            assert_eq!(sol_z2.len(), self.m_full);

            prepare_rhs_singleton(singleton, rhs_x1, rhs_z1, &mut self.solve_ws);
            fill_rhs_perm_with_perm(perm, self.n, &self.solve_ws.rhs_x, &self.solve_ws.rhs_z, &mut self.solve_ws.rhs_perm);
            solve_permuted_with_refinement(
                &self.backend,
                static_reg,
                kkt,
                &mut self.solve_ws,
                factor,
                RhsPermKind::Primary,
                refine_iters,
                reg_diag_sign,
                tag1,
            );
            if need_woodbury {
                self.sm_workspace.apply_woodbury_correction(
                    &self.backend,
                    factor,
                    n,
                    perm,
                    &mut self.solve_ws.sol_perm,
                );
            }
            unpermute_solution_with_perm(perm_inv, self.n, &self.solve_ws.sol_perm, sol_x1, &mut self.solve_ws.sol_z);
            expand_solution_z_singleton(singleton, rhs_z1, sol_x1, sol_z1, &self.solve_ws.sol_z);

            prepare_rhs_singleton(singleton, rhs_x2, rhs_z2, &mut self.solve_ws);
            fill_rhs_perm_with_perm(perm, self.n, &self.solve_ws.rhs_x, &self.solve_ws.rhs_z, &mut self.solve_ws.rhs_perm2);
            solve_permuted_with_refinement(
                &self.backend,
                static_reg,
                kkt,
                &mut self.solve_ws,
                factor,
                RhsPermKind::Secondary,
                refine_iters,
                reg_diag_sign,
                tag2,
            );
            if need_woodbury {
                self.sm_workspace.apply_woodbury_correction(
                    &self.backend,
                    factor,
                    n,
                    perm,
                    &mut self.solve_ws.sol_perm,
                );
            }
            unpermute_solution_with_perm(perm_inv, self.n, &self.solve_ws.sol_perm, sol_x2, &mut self.solve_ws.sol_z);
            expand_solution_z_singleton(singleton, rhs_z2, sol_x2, sol_z2, &self.solve_ws.sol_z);
        } else {
            assert_eq!(rhs_z1.len(), self.m);
            assert_eq!(rhs_z2.len(), self.m);
            assert_eq!(sol_z1.len(), self.m);
            assert_eq!(sol_z2.len(), self.m);

            fill_rhs_perm_two_with_perm(
                perm,
                self.n,
                rhs_x1,
                rhs_z1,
                rhs_x2,
                rhs_z2,
                &mut self.solve_ws.rhs_perm,
                &mut self.solve_ws.rhs_perm2,
            );

            solve_permuted_with_refinement(
                &self.backend,
                static_reg,
                kkt,
                &mut self.solve_ws,
                factor,
                RhsPermKind::Primary,
                refine_iters,
                reg_diag_sign,
                tag1,
            );
            if need_woodbury {
                self.sm_workspace.apply_woodbury_correction(
                    &self.backend,
                    factor,
                    n,
                    perm,
                    &mut self.solve_ws.sol_perm,
                );
            }
            unpermute_solution_with_perm(perm_inv, self.n, &self.solve_ws.sol_perm, sol_x1, sol_z1);

            solve_permuted_with_refinement(
                &self.backend,
                static_reg,
                kkt,
                &mut self.solve_ws,
                factor,
                RhsPermKind::Secondary,
                refine_iters,
                reg_diag_sign,
                tag2,
            );
            if need_woodbury {
                self.sm_workspace.apply_woodbury_correction(
                    &self.backend,
                    factor,
                    n,
                    perm,
                    &mut self.solve_ws.sol_perm,
                );
            }
            unpermute_solution_with_perm(perm_inv, self.n, &self.solve_ws.sol_perm, sol_x2, sol_z2);
        }
    }

    fn fill_reg_diag_sign(&mut self, mode: KktRefineMode) {
        let kkt_dim = self.n + self.m;
        if self.reg_diag_sign.len() != kkt_dim {
            self.reg_diag_sign.resize(kkt_dim, 0.0);
        }
        let (primal_sign, dual_sign) = match mode {
            KktRefineMode::Regularized => (0.0, 0.0),
            KktRefineMode::QdldlDebias => (1.0, 0.0),
            KktRefineMode::FullUnreg => (2.0, -1.0),
        };
        if let Some(perm) = self.perm.as_ref() {
            for new_idx in 0..kkt_dim {
                let old_idx = perm[new_idx];
                self.reg_diag_sign[new_idx] = if old_idx < self.n {
                    primal_sign
                } else {
                    dual_sign
                };
            }
        } else {
            for i in 0..self.n {
                self.reg_diag_sign[i] = primal_sign;
            }
            for i in self.n..kkt_dim {
                self.reg_diag_sign[i] = dual_sign;
            }
        }
    }

    /// Get the number of dynamic regularization bumps from the last factorization.
    pub fn dynamic_bumps(&self) -> u64 {
        self.backend.dynamic_bumps()
    }
}

impl<B: KktBackend> KktSolverTrait for KktSolverImpl<B> {
    type Factor = B::Factorization;

    fn initialize(
        &mut self,
        p: Option<&SparseSymmetricCsc>,
        a: &SparseCsc,
        h_blocks: &[ScalingBlock],
    ) -> Result<(), BackendError> {
        KktSolverImpl::initialize(self, p, a, h_blocks)
    }

    fn update_numeric(
        &mut self,
        p: Option<&SparseSymmetricCsc>,
        a: &SparseCsc,
        h_blocks: &[ScalingBlock],
    ) -> Result<(), BackendError> {
        KktSolverImpl::update_numeric(self, p, a, h_blocks)
    }

    fn factorize(&mut self) -> Result<Self::Factor, BackendError> {
        KktSolverImpl::factorize(self)
    }

    fn solve_refined(
        &mut self,
        factor: &Self::Factor,
        rhs_x: &[f64],
        rhs_z: &[f64],
        sol_x: &mut [f64],
        sol_z: &mut [f64],
        refine_iters: usize,
    ) {
        KktSolverImpl::solve_refined(self, factor, rhs_x, rhs_z, sol_x, sol_z, refine_iters)
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_two_rhs_refined_tagged(
        &mut self,
        factor: &Self::Factor,
        rhs_x1: &[f64],
        rhs_z1: &[f64],
        rhs_x2: &[f64],
        rhs_z2: &[f64],
        sol_x1: &mut [f64],
        sol_z1: &mut [f64],
        sol_x2: &mut [f64],
        sol_z2: &mut [f64],
        refine_iters: usize,
        tag1: &'static str,
        tag2: &'static str,
    ) {
        KktSolverImpl::solve_two_rhs_refined_tagged(
            self, factor, rhs_x1, rhs_z1, rhs_x2, rhs_z2,
            sol_x1, sol_z1, sol_x2, sol_z2, refine_iters, tag1, tag2,
        )
    }

    fn static_reg(&self) -> f64 {
        KktSolverImpl::static_reg(self)
    }

    fn set_static_reg(&mut self, reg: f64) -> Result<(), BackendError> {
        KktSolverImpl::set_static_reg(self, reg)
    }

    fn bump_static_reg(&mut self, min_reg: f64) -> Result<bool, BackendError> {
        KktSolverImpl::bump_static_reg(self, min_reg)
    }

    fn dynamic_bumps(&self) -> u64 {
        KktSolverImpl::dynamic_bumps(self)
    }
}

#[cfg(feature = "suitesparse-ldl")]
use super::backends::SuiteSparseLdlBackend;

#[cfg(feature = "suitesparse-ldl")]
type DefaultBackend = SuiteSparseLdlBackend;

#[cfg(not(feature = "suitesparse-ldl"))]
type DefaultBackend = QdldlBackend;

pub type KktSolver = KktSolverImpl<DefaultBackend>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::sparse;

    #[test]
    fn test_kkt_simple_lp() {
        // Simple LP:
        //   min  x1 + x2
        //   s.t. x1 + x2 = 1   (equality)
        //        x1, x2 >= 0   (nonnegativity)
        //
        // Variables: x = [x1, x2]  (n=2)
        // Slacks: s = [s_eq, s1, s2]  (m=3)
        //   s_eq for equality (zero cone)
        //   s1, s2 for nonnegativity (nonneg cone)
        //
        // KKT system (4×4 with regularization omitted):
        //   [0  0 | 1  1  1 ] [dx1 ]   [r_x1 ]
        //   [0  0 | 1  1  1 ] [dx2 ]   [r_x2 ]
        //   [------+--------] [---- ] = [-----]
        //   [1  1 | 0  0  0 ] [dz_eq]   [r_zeq]
        //   [1  1 | 0 -h1 0 ] [dz1  ]   [r_z1 ]
        //   [1  1 | 0  0 -h2] [dz2  ]   [r_z2 ]
        //
        // For this test, we'll use h1 = h2 = 1.0

        let n = 2;
        let m = 3;

        // P = None (LP, no quadratic term)
        // A = [[1, 1], [1, 0], [0, 1]]  (m×n)
        let a_triplets = vec![
            (0, 0, 1.0), (0, 1, 1.0),  // Equality constraint
            (1, 0, 1.0),               // x1 >= 0
            (2, 1, 1.0),               // x2 >= 0
        ];
        let a = sparse::from_triplets(m, n, a_triplets);

        // H blocks: [Zero(1), Diagonal([1.0, 1.0])]
        let h_blocks = vec![
            ScalingBlock::Zero { dim: 1 },
            ScalingBlock::Diagonal { d: vec![1.0, 1.0] },
        ];

        let mut kkt_solver = KktSolver::new(n, m, 1e-8, 1e-7);

        // Initialize (symbolic factorization)
        kkt_solver.initialize(None, &a, &h_blocks).unwrap();

        // Factor (numeric)
        let factor = kkt_solver.factor(None, &a, &h_blocks).unwrap();

        // Solve a simple system: K * [dx; dz] = [1, 1, 0, 0, 0]
        let rhs_x = vec![1.0, 1.0];
        let rhs_z = vec![0.0, 0.0, 0.0];
        let mut sol_x = vec![0.0; 2];
        let mut sol_z = vec![0.0; 3];

        kkt_solver.solve(&factor, &rhs_x, &rhs_z, &mut sol_x, &mut sol_z);

        // Validate solution by checking residual against the regularized system.
        let kkt_unpermuted = kkt_solver.build_kkt_matrix_with_perm(None, None, &a, &h_blocks);
        let mut sol_full = vec![0.0; n + m];
        sol_full[..n].copy_from_slice(&sol_x);
        sol_full[n..].copy_from_slice(&sol_z);

        let mut kx = vec![0.0; n + m];
        symm_matvec_upper(&kkt_unpermuted, &sol_full, &mut kx);
        let static_reg = kkt_solver.static_reg();
        if static_reg != 0.0 {
            for i in 0..n + m {
                kx[i] += static_reg * sol_full[i];
            }
        }

        let mut res_norm = 0.0;
        for i in 0..n {
            let r = rhs_x[i] - kx[i];
            res_norm += r * r;
        }
        for i in 0..m {
            let r = rhs_z[i] - kx[n + i];
            res_norm += r * r;
        }
        res_norm = res_norm.sqrt();

        assert!(res_norm < 1e-6, "KKT residual too large: {}", res_norm);
    }

    #[test]
    fn test_kkt_with_p_matrix() {
        // QP with cost: 0.5 * (x1^2 + x2^2) + 0
        // Constraint: x1 + x2 >= 1
        //
        // P = [[1, 0], [0, 1]]
        // A = [[1, 1]]
        // H = [1.0] (nonneg cone)

        let n = 2;
        let m = 1;

        let p_triplets = vec![(0, 0, 1.0), (1, 1, 1.0)];
        let p = sparse::from_triplets_symmetric(n, p_triplets);

        let a_triplets = vec![(0, 0, 1.0), (0, 1, 1.0)];
        let a = sparse::from_triplets(m, n, a_triplets);

        let h_blocks = vec![ScalingBlock::Diagonal { d: vec![1.0] }];

        let mut kkt_solver = KktSolver::new(n, m, 1e-8, 1e-7);

        kkt_solver.initialize(Some(&p), &a, &h_blocks).unwrap();
        let factor = kkt_solver.factor(Some(&p), &a, &h_blocks).unwrap();

        // Solve trivial system
        let rhs_x = vec![1.0, 1.0];
        let rhs_z = vec![0.0];
        let mut sol_x = vec![0.0; 2];
        let mut sol_z = vec![0.0; 1];

        kkt_solver.solve(&factor, &rhs_x, &rhs_z, &mut sol_x, &mut sol_z);

        // Check that we got a solution
        assert!(sol_x[0].abs() + sol_x[1].abs() > 1e-6);
    }

    #[test]
    fn test_kkt_two_solve() {
        // Test the two-RHS solve strategy
        let n = 2;
        let m = 1;

        let p_triplets = vec![(0, 0, 1.0), (1, 1, 1.0)];
        let p = sparse::from_triplets_symmetric(n, p_triplets);

        let a_triplets = vec![(0, 0, 1.0), (0, 1, 1.0)];
        let a = sparse::from_triplets(m, n, a_triplets);

        let h_blocks = vec![ScalingBlock::Diagonal { d: vec![1.0] }];

        let mut kkt_solver = KktSolver::new(n, m, 1e-8, 1e-7);
        kkt_solver.initialize(Some(&p), &a, &h_blocks).unwrap();
        let factor = kkt_solver.factor(Some(&p), &a, &h_blocks).unwrap();

        // Two different RHS
        let rhs_x1 = vec![1.0, 0.0];
        let rhs_z1 = vec![0.0];
        let rhs_x2 = vec![0.0, 1.0];
        let rhs_z2 = vec![1.0];

        let mut sol_x1 = vec![0.0; 2];
        let mut sol_z1 = vec![0.0; 1];
        let mut sol_x2 = vec![0.0; 2];
        let mut sol_z2 = vec![0.0; 1];

        kkt_solver.solve_two_rhs(
            &factor,
            &rhs_x1, &rhs_z1,
            &rhs_x2, &rhs_z2,
            &mut sol_x1, &mut sol_z1,
            &mut sol_x2, &mut sol_z2,
        );

        // Check that both solutions are non-trivial
        assert!(sol_x1[0].abs() + sol_x1[1].abs() > 1e-6);
        assert!(sol_x2[0].abs() + sol_x2[1].abs() > 1e-6);

        // Solutions should be different
        assert!((sol_x1[0] - sol_x2[0]).abs() > 1e-6 || (sol_x1[1] - sol_x2[1]).abs() > 1e-6);
    }

    #[test]
    fn test_kkt_psd_cone() {
        // Test PSD cone KKT assembly
        // Simple 2x2 PSD cone:
        //   n = 3 (svec dimension for 2x2 symmetric matrix)
        //   m = 3 (single PSD cone block)
        //   A = -I (identity embedding: s = x)
        //   H = W ⊗ W where W = I (identity scaling)
        //
        // For W = I, H = I (identity on svec space).
        // KKT matrix:
        //   [ε*I    -I     ]
        //   [-I   -(I + 2ε)]
        //
        // This should be well-conditioned.

        let n = 3;  // svec dimension for 2x2
        let m = 3;

        // A = -I
        let a_triplets = vec![
            (0, 0, -1.0),
            (1, 1, -1.0),
            (2, 2, -1.0),
        ];
        let a = sparse::from_triplets(m, n, a_triplets);

        // H = PSD scaling with W = I (2x2 identity)
        // w_factor stores W row-major: [1, 0, 0, 1]
        let w_factor = vec![1.0, 0.0, 0.0, 1.0];
        let h_blocks = vec![
            ScalingBlock::PsdStructured { w_factor, n: 2 },
        ];

        // For this test, H diagonal is ~1, so regularization should be comparable.
        // Using very small regularization (1e-8) causes huge Schur complement pivots
        // during LDLT, leading to numerical errors. Use sqrt(eps) * scale instead.
        let static_reg = 1e-4;  // sqrt(1e-8) matched to H scale
        let dynamic_reg = 1e-7;

        let mut kkt_solver = KktSolver::new(n, m, static_reg, dynamic_reg);

        // Initialize (symbolic factorization)
        kkt_solver.initialize(None, &a, &h_blocks).unwrap();

        // Factor
        let factor = kkt_solver.factor(None, &a, &h_blocks).unwrap();

        // Debug: print the KKT matrix
        let kkt_mat = kkt_solver.build_kkt_matrix_with_perm(None, None, &a, &h_blocks);
        eprintln!("KKT matrix ({}x{}):", kkt_mat.rows(), kkt_mat.cols());
        for col in 0..kkt_mat.cols() {
            if let Some(col_view) = kkt_mat.outer_view(col) {
                for (row, &val) in col_view.iter() {
                    if val.abs() > 1e-14 {
                        eprintln!("  K[{},{}] = {:.6e}", row, col, val);
                    }
                }
            }
        }

        // Note: condition number estimate can be misleading for indefinite matrices
        // with small regularization. The key test is whether the solve is accurate.
        if let Some(cond) = kkt_solver.estimate_condition_number() {
            eprintln!("PSD KKT condition number: {:.3e}", cond);
        }

        // Solve a simple system
        let rhs_x = vec![1.0, 0.0, 0.0];
        let rhs_z = vec![0.0, 0.0, 0.0];
        let mut sol_x = vec![0.0; n];
        let mut sol_z = vec![0.0; m];

        kkt_solver.solve(&factor, &rhs_x, &rhs_z, &mut sol_x, &mut sol_z);

        eprintln!("PSD KKT solve: sol_x = {:?}", sol_x);
        eprintln!("PSD KKT solve: sol_z = {:?}", sol_z);

        // Verify solution accuracy
        // For the regularized system, the exact solution is:
        //   (ε + 1)*x ≈ rhs_x => x ≈ rhs_x / (ε + 1) ≈ rhs_x
        //   z = -x
        // The solution should be close to x = [1, 0, 0], z = [-1, 0, 0]
        eprintln!("Expected: x ≈ [1, 0, 0], z ≈ [-1, 0, 0]");

        // Check solution is roughly correct (allow O(ε) error due to regularization)
        let tol = 2.0 * static_reg;  // O(ε) tolerance
        assert!((sol_x[0] - 1.0).abs() < tol, "sol_x[0] should be ≈1, got {}", sol_x[0]);
        assert!(sol_x[1].abs() < tol, "sol_x[1] should be ≈0, got {}", sol_x[1]);
        assert!(sol_x[2].abs() < tol, "sol_x[2] should be ≈0, got {}", sol_x[2]);
        assert!((sol_z[0] + 1.0).abs() < tol, "sol_z[0] should be ≈-1, got {}", sol_z[0]);
        assert!(sol_z[1].abs() < tol, "sol_z[1] should be ≈0, got {}", sol_z[1]);
        assert!(sol_z[2].abs() < tol, "sol_z[2] should be ≈0, got {}", sol_z[2]);
    }
}
