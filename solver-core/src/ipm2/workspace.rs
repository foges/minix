use crate::cones::{ConeKernel, SocCone, ExpCone, PowCone, PsdCone};
use crate::scaling::ScalingBlock;
use std::any::Any;

#[derive(Debug)]
pub struct IpmWorkspace {
    pub n: usize,
    pub m: usize,
    pub kkt_dim: usize,
    pub orig_n: usize,
    pub orig_m: usize,

    // Two RHS solves (two-solve strategy)
    pub rhs1: Vec<f64>,
    pub rhs2: Vec<f64>,
    pub sol1: Vec<f64>,
    pub sol2: Vec<f64>,

    // Predictor-corrector scratch (allocation-free hot loop)
    pub rhs_x: Vec<f64>,
    pub rhs_z: Vec<f64>,
    pub dx_aff: Vec<f64>,
    pub dz_aff: Vec<f64>,
    pub ds_aff: Vec<f64>,
    pub dx: Vec<f64>,
    pub dz: Vec<f64>,
    pub ds: Vec<f64>,
    pub dx2: Vec<f64>,
    pub dz2: Vec<f64>,
    pub d_s_comb: Vec<f64>,
    pub mul_p_xi: Vec<f64>,
    pub mul_p_xi_q: Vec<f64>,
    pub delta_w: Vec<f64>,
    pub mcc_delta: Vec<f64>,

    // Termination / metrics scratch
    pub r_p: Vec<f64>,
    pub r_d: Vec<f64>,
    pub p_x: Vec<f64>,

    // Recovered/unscaled vectors (optional)
    pub x_bar: Vec<f64>,
    pub s_bar: Vec<f64>,
    pub z_bar: Vec<f64>,
    pub x_full: Vec<f64>,
    pub s_full: Vec<f64>,
    pub z_full: Vec<f64>,

    // Scaling blocks (reused per iteration)
    pub scaling: Vec<ScalingBlock>,

    // SOC scratch buffers (sized to max SOC cone)
    pub soc_scratch: SocScratch,
}

impl IpmWorkspace {
    pub fn new(n: usize, m: usize, orig_n: usize, orig_m: usize) -> Self {
        Self::new_with_sz_len(n, m, orig_n, orig_m)
    }

    /// Create workspace with explicit s/z full vector length.
    ///
    /// Use this when postsolve may change the number of bound constraints,
    /// causing the recovered s/z vectors to have different sizes than orig_m.
    pub fn new_with_sz_len(n: usize, m: usize, orig_n: usize, sz_full_len: usize) -> Self {
        let kkt_dim = n + m;
        Self {
            n,
            m,
            kkt_dim,
            orig_n,
            orig_m: sz_full_len, // Store the actual full length for consistency
            rhs1: vec![0.0; kkt_dim],
            rhs2: vec![0.0; kkt_dim],
            sol1: vec![0.0; kkt_dim],
            sol2: vec![0.0; kkt_dim],
            rhs_x: vec![0.0; n],
            rhs_z: vec![0.0; m],
            dx_aff: vec![0.0; n],
            dz_aff: vec![0.0; m],
            ds_aff: vec![0.0; m],
            dx: vec![0.0; n],
            dz: vec![0.0; m],
            ds: vec![0.0; m],
            dx2: vec![0.0; n],
            dz2: vec![0.0; m],
            d_s_comb: vec![0.0; m],
            mul_p_xi: vec![0.0; n],
            mul_p_xi_q: vec![0.0; n],
            delta_w: vec![0.0; m],
            mcc_delta: vec![0.0; m],
            r_p: vec![0.0; sz_full_len],
            r_d: vec![0.0; orig_n],
            p_x: vec![0.0; orig_n],
            x_bar: vec![0.0; n],
            s_bar: vec![0.0; m],
            z_bar: vec![0.0; m],
            x_full: vec![0.0; orig_n],
            s_full: vec![0.0; sz_full_len],
            z_full: vec![0.0; sz_full_len],
            scaling: Vec::new(),
            soc_scratch: SocScratch::new(0),
        }
    }

    pub fn init_cones(&mut self, cones: &[Box<dyn ConeKernel>]) {
        self.scaling.clear();
        let mut max_soc_dim = 0usize;
        for cone in cones {
            let dim = cone.dim();
            if dim == 0 {
                self.scaling.push(ScalingBlock::Zero { dim });
                continue;
            }

            if cone.barrier_degree() == 0 {
                self.scaling.push(ScalingBlock::Zero { dim });
                continue;
            }

            let is_soc = (cone.as_ref() as &dyn Any).is::<SocCone>();
            if is_soc {
                // SOC identity element is e = (1, 0, ..., 0), so initial w = e
                let mut w = vec![0.0; dim];
                w[0] = 1.0;
                self.scaling.push(ScalingBlock::SocStructured { w });
                max_soc_dim = max_soc_dim.max(dim);
            } else if (cone.as_ref() as &dyn Any).is::<ExpCone>()
                || (cone.as_ref() as &dyn Any).is::<PowCone>()
            {
                self.scaling.push(ScalingBlock::Dense3x3 {
                    h: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                });
            } else if let Some(psd) = (cone.as_ref() as &dyn Any).downcast_ref::<PsdCone>() {
                let n = psd.size();
                let mut w_factor = vec![0.0; n * n];
                for i in 0..n {
                    w_factor[i * n + i] = 1.0;
                }
                self.scaling.push(ScalingBlock::PsdStructured { w_factor, n });
            } else {
                self.scaling.push(ScalingBlock::Diagonal { d: vec![1.0; dim] });
            }
        }

        self.soc_scratch.ensure_dim(max_soc_dim);
    }

    #[inline]
    pub fn clear_rhs(&mut self) {
        self.rhs1.fill(0.0);
        self.rhs2.fill(0.0);
    }

    #[inline]
    pub fn clear_solutions(&mut self) {
        self.sol1.fill(0.0);
        self.sol2.fill(0.0);
    }
}

#[derive(Debug)]
pub struct SocScratch {
    dim: usize,
    pub s_sqrt: Vec<f64>,
    pub u: Vec<f64>,
    pub u_inv: Vec<f64>,
    pub u_inv_sqrt: Vec<f64>,
    pub w_half: Vec<f64>,
    pub w_half_inv: Vec<f64>,
    pub lambda: Vec<f64>,
    pub w_inv_ds: Vec<f64>,
    pub w_dz: Vec<f64>,
    pub eta: Vec<f64>,
    pub lambda_sq: Vec<f64>,
    pub v: Vec<f64>,
    pub u_vec: Vec<f64>,
    pub d_s_block: Vec<f64>,
    pub h_dz: Vec<f64>,
    pub e1: Vec<f64>,
    pub e2: Vec<f64>,
    pub w_circ_y: Vec<f64>,
    pub w_circ_w: Vec<f64>,
    pub temp: Vec<f64>,
    pub w2_circ_y: Vec<f64>,
}

impl SocScratch {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            s_sqrt: vec![0.0; dim],
            u: vec![0.0; dim],
            u_inv: vec![0.0; dim],
            u_inv_sqrt: vec![0.0; dim],
            w_half: vec![0.0; dim],
            w_half_inv: vec![0.0; dim],
            lambda: vec![0.0; dim],
            w_inv_ds: vec![0.0; dim],
            w_dz: vec![0.0; dim],
            eta: vec![0.0; dim],
            lambda_sq: vec![0.0; dim],
            v: vec![0.0; dim],
            u_vec: vec![0.0; dim],
            d_s_block: vec![0.0; dim],
            h_dz: vec![0.0; dim],
            e1: vec![0.0; dim],
            e2: vec![0.0; dim],
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
        self.s_sqrt.resize(dim, 0.0);
        self.u.resize(dim, 0.0);
        self.u_inv.resize(dim, 0.0);
        self.u_inv_sqrt.resize(dim, 0.0);
        self.w_half.resize(dim, 0.0);
        self.w_half_inv.resize(dim, 0.0);
        self.lambda.resize(dim, 0.0);
        self.w_inv_ds.resize(dim, 0.0);
        self.w_dz.resize(dim, 0.0);
        self.eta.resize(dim, 0.0);
        self.lambda_sq.resize(dim, 0.0);
        self.v.resize(dim, 0.0);
        self.u_vec.resize(dim, 0.0);
        self.d_s_block.resize(dim, 0.0);
        self.h_dz.resize(dim, 0.0);
        self.e1.resize(dim, 0.0);
        self.e2.resize(dim, 0.0);
        self.w_circ_y.resize(dim, 0.0);
        self.w_circ_w.resize(dim, 0.0);
        self.temp.resize(dim, 0.0);
        self.w2_circ_y.resize(dim, 0.0);
    }
}
