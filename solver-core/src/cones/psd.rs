//! Positive semidefinite cone.
//!
//! Stored in svec format with sqrt(2) scaling on off-diagonals.

use super::traits::ConeKernel;
use nalgebra::DMatrix;
use nalgebra::linalg::SymmetricEigen;
use std::sync::OnceLock;

fn psd_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        // Check new unified MINIX_VERBOSE first (level >= 4 means trace)
        if let Ok(v) = std::env::var("MINIX_VERBOSE") {
            if let Ok(n) = v.parse::<u8>() {
                return n >= 4;
            }
        }
        // Legacy: check MINIX_DEBUG_PSD
        std::env::var("MINIX_DEBUG_PSD").ok().as_deref() == Some("1")
    })
}

/// PSD cone (placeholder)
#[derive(Debug, Clone)]
pub struct PsdCone {
    n: usize,
}

impl PsdCone {
    /// Create a new PSD cone for n×n matrices
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    /// Interior tolerance relative to ||X||.
    const INTERIOR_TOL: f64 = 1e-12;

    pub(crate) fn size(&self) -> usize {
        self.n
    }
}

impl ConeKernel for PsdCone {
    fn dim(&self) -> usize { self.n * (self.n + 1) / 2 }
    fn barrier_degree(&self) -> usize { self.n }
    fn is_interior_primal(&self, s: &[f64]) -> bool {
        assert_eq!(s.len(), self.dim());
        if s.iter().any(|&v| !v.is_finite()) {
            return false;
        }

        let x = svec_to_mat(s, self.n);
        let scale = x.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1.0);
        let tol = Self::INTERIOR_TOL * scale;

        let eig = SymmetricEigen::new(x);
        let min_eig = eig.eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
        min_eig.is_finite() && min_eig > tol
    }

    fn is_interior_dual(&self, z: &[f64]) -> bool {
        self.is_interior_primal(z)
    }

    fn step_to_boundary_primal(&self, s: &[f64], ds: &[f64]) -> f64 {
        assert_eq!(s.len(), self.dim());
        assert_eq!(ds.len(), self.dim());
        if ds.iter().all(|&v| v == 0.0) {
            return f64::INFINITY;
        }

        let x = svec_to_mat(s, self.n);
        let dx = svec_to_mat(ds, self.n);
        let eig_x = SymmetricEigen::new(x.clone());
        let min_eig_x = eig_x.eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);

        // Debug output for PSD step computation at trace level (MINIX_VERBOSE=4)
        let debug_psd = psd_trace_enabled();
        if debug_psd {
            eprintln!("PSD step_to_boundary: n={}, min_eig_x={:.3e}, s={:?}", self.n, min_eig_x, s);
            eprintln!("  ds={:?}", ds);
            eprintln!("  X eigenvalues: {:?}", eig_x.eigenvalues.as_slice());
        }

        if !min_eig_x.is_finite() || min_eig_x <= 0.0 {
            if debug_psd {
                eprintln!("  -> returning 0.0 (s not interior)");
            }
            return 0.0;
        }

        let inv_sqrt_vals = eig_x.eigenvalues.map(|v| 1.0 / v.sqrt());
        let x_inv_sqrt = &eig_x.eigenvectors
            * DMatrix::<f64>::from_diagonal(&inv_sqrt_vals)
            * eig_x.eigenvectors.transpose();

        let mut m = &x_inv_sqrt * dx * x_inv_sqrt.transpose();
        m = 0.5 * (&m + m.transpose());

        let eig_m = SymmetricEigen::new(m);
        let min_eig = eig_m.eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
        if !min_eig.is_finite() {
            return 0.0;
        }
        if min_eig >= 0.0 {
            f64::INFINITY
        } else {
            -1.0 / min_eig
        }
    }

    fn step_to_boundary_dual(&self, z: &[f64], dz: &[f64]) -> f64 {
        self.step_to_boundary_primal(z, dz)
    }

    fn barrier_value(&self, s: &[f64]) -> f64 {
        assert_eq!(s.len(), self.dim());
        let x = svec_to_mat(s, self.n);
        let eig = SymmetricEigen::new(x);
        let mut log_det = 0.0;
        for &lambda in eig.eigenvalues.iter() {
            if lambda <= 0.0 || !lambda.is_finite() {
                return f64::INFINITY;
            }
            log_det += lambda.ln();
        }
        -log_det
    }

    fn barrier_grad_primal(&self, s: &[f64], grad_out: &mut [f64]) {
        assert_eq!(s.len(), self.dim());
        assert_eq!(grad_out.len(), self.dim());
        let x = svec_to_mat(s, self.n);
        let eig = SymmetricEigen::new(x);
        let inv_vals = eig.eigenvalues.map(|v| 1.0 / v);
        let x_inv = &eig.eigenvectors
            * DMatrix::<f64>::from_diagonal(&inv_vals)
            * eig.eigenvectors.transpose();
        let grad_mat = -x_inv;
        mat_to_svec(&grad_mat, grad_out);
    }

    fn barrier_hess_apply_primal(&self, s: &[f64], v: &[f64], out: &mut [f64]) {
        assert_eq!(s.len(), self.dim());
        assert_eq!(v.len(), self.dim());
        assert_eq!(out.len(), self.dim());

        let x = svec_to_mat(s, self.n);
        let v_mat = svec_to_mat(v, self.n);

        let eig = SymmetricEigen::new(x);
        let inv_vals = eig.eigenvalues.map(|v| 1.0 / v);
        let x_inv = &eig.eigenvectors
            * DMatrix::<f64>::from_diagonal(&inv_vals)
            * eig.eigenvectors.transpose();

        let hess_v = &x_inv * v_mat * x_inv;
        mat_to_svec(&hess_v, out);
    }

    fn barrier_grad_dual(&self, z: &[f64], grad_out: &mut [f64]) {
        self.barrier_grad_primal(z, grad_out)
    }

    fn barrier_hess_apply_dual(&self, z: &[f64], v: &[f64], out: &mut [f64]) {
        self.barrier_hess_apply_primal(z, v, out)
    }

    fn dual_map(&self, _z: &[f64], _x_out: &mut [f64], _h_star: &mut [f64; 9]) {
        panic!("PSD cone is self-dual; dual_map not needed");
    }

    fn unit_initialization(&self, s_out: &mut [f64], z_out: &mut [f64]) {
        assert_eq!(s_out.len(), self.dim());
        assert_eq!(z_out.len(), self.dim());
        s_out.fill(0.0);
        z_out.fill(0.0);

        let mut idx = 0usize;
        for j in 0..self.n {
            for i in 0..=j {
                if i == j {
                    s_out[idx] = 1.0;
                    z_out[idx] = 1.0;
                }
                idx += 1;
            }
        }
    }
}

pub(crate) fn svec_to_mat(s: &[f64], n: usize) -> DMatrix<f64> {
    assert_eq!(s.len(), n * (n + 1) / 2);
    let mut out = DMatrix::<f64>::zeros(n, n);
    let mut idx = 0usize;
    let sqrt2 = std::f64::consts::SQRT_2;

    for j in 0..n {
        for i in 0..=j {
            let val = s[idx];
            if i == j {
                out[(i, j)] = val;
            } else {
                let scaled = val / sqrt2;
                out[(i, j)] = scaled;
                out[(j, i)] = scaled;
            }
            idx += 1;
        }
    }

    out
}

pub(crate) fn mat_to_svec(m: &DMatrix<f64>, out: &mut [f64]) {
    let n = m.nrows();
    assert_eq!(m.ncols(), n);
    assert_eq!(out.len(), n * (n + 1) / 2);
    let sqrt2 = std::f64::consts::SQRT_2;
    let mut idx = 0usize;
    for j in 0..n {
        for i in 0..=j {
            out[idx] = if i == j { m[(i, j)] } else { m[(i, j)] * sqrt2 };
            idx += 1;
        }
    }
}
