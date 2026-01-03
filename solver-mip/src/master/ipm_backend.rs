//! Master backend using solver-core IPM.
//!
//! This backend solves the master LP/QP relaxation using the interior-point
//! solver from solver-core. Conic constraints (beyond Zero and NonNeg) are
//! relaxed and enforced via cuts.

use solver_core::{solve, ConeSpec, ProblemData, SolveStatus, SolverSettings};
use sprs::{CsMat, TriMat};

use super::{LinearCut, MasterBackend, MasterResult, MasterStatus};
use crate::error::{MipError, MipResult};
use crate::model::MipProblem;

/// Master backend using solver-core IPM.
pub struct IpmMasterBackend {
    /// Number of original variables.
    n: usize,

    /// Number of original constraints (from base problem).
    m_base: usize,

    /// Original problem data (with cones relaxed to NonNeg where needed).
    base_prob: Option<ProblemData>,

    /// Active cuts as (coefs, rhs) pairs.
    cuts: Vec<LinearCut>,

    /// Mapping from cut index to internal storage index.
    /// Some cuts may be removed, leaving gaps.
    cut_active: Vec<bool>,

    /// Current variable lower bounds.
    var_lb: Vec<f64>,

    /// Current variable upper bounds.
    var_ub: Vec<f64>,

    /// Solver settings.
    settings: SolverSettings,
}

impl IpmMasterBackend {
    /// Create a new IPM master backend.
    pub fn new(settings: SolverSettings) -> Self {
        Self {
            n: 0,
            m_base: 0,
            base_prob: None,
            cuts: Vec::new(),
            cut_active: Vec::new(),
            var_lb: Vec::new(),
            var_ub: Vec::new(),
            settings,
        }
    }

    /// Build the current master problem with all cuts and bounds.
    ///
    /// The master problem is:
    /// ```text
    /// min  0.5 x^T P x + q^T x
    /// s.t. A_base x + s_base = b_base,  s_base in K_base (Zero/NonNeg only)
    ///      a_i^T x + s_cut_i = rhs_i,   s_cut_i >= 0  (for each cut i)
    ///      x + s_ub = ub,               s_ub >= 0  (upper bounds)
    ///      -x + s_lb = -lb,             s_lb >= 0  (lower bounds)
    /// ```
    fn build_master_problem(&self) -> MipResult<ProblemData> {
        let base = self.base_prob.as_ref().ok_or_else(|| {
            MipError::InternalError("Master backend not initialized".to_string())
        })?;

        let n = self.n;

        // Count active cuts
        let active_cuts: Vec<&LinearCut> = self
            .cuts
            .iter()
            .zip(&self.cut_active)
            .filter(|(_, &active)| active)
            .map(|(cut, _)| cut)
            .collect();
        let num_cuts = active_cuts.len();

        // Count bound constraints (finite bounds only)
        let num_lb = self.var_lb.iter().filter(|&&lb| lb > f64::NEG_INFINITY).count();
        let num_ub = self.var_ub.iter().filter(|&&ub| ub < f64::INFINITY).count();

        // Total constraints
        let m_total = self.m_base + num_cuts + num_lb + num_ub;

        // Build combined A matrix
        let mut triplets: Vec<(usize, usize, f64)> = Vec::new();

        // Copy base A
        for (col_idx, col) in base.A.outer_iterator().enumerate() {
            for (row_idx, &val) in col.iter() {
                triplets.push((row_idx, col_idx, val));
            }
        }

        // Add cut rows
        let mut row = self.m_base;
        for cut in &active_cuts {
            for (j, &coef) in cut.coefs.iter().enumerate() {
                if coef.abs() > 1e-15 {
                    triplets.push((row, j, coef));
                }
            }
            row += 1;
        }

        // Add lower bound rows: -x <= -lb  =>  -x + s = -lb, s >= 0
        for (j, &lb) in self.var_lb.iter().enumerate() {
            if lb > f64::NEG_INFINITY {
                triplets.push((row, j, -1.0));
                row += 1;
            }
        }

        // Add upper bound rows: x <= ub  =>  x + s = ub, s >= 0
        for (j, &ub) in self.var_ub.iter().enumerate() {
            if ub < f64::INFINITY {
                triplets.push((row, j, 1.0));
                row += 1;
            }
        }

        // Build sparse matrix
        let a_combined = triplets_to_csc(m_total, n, &triplets);

        // Build combined b vector
        let mut b_combined = Vec::with_capacity(m_total);
        b_combined.extend_from_slice(&base.b);

        // Cut RHS
        for cut in &active_cuts {
            b_combined.push(cut.rhs);
        }

        // Lower bound RHS: -lb
        for &lb in &self.var_lb {
            if lb > f64::NEG_INFINITY {
                b_combined.push(-lb);
            }
        }

        // Upper bound RHS: ub
        for &ub in &self.var_ub {
            if ub < f64::INFINITY {
                b_combined.push(ub);
            }
        }

        // Build cone specification
        let mut cones = base.cones.clone();

        // All cuts and bounds use NonNeg cone
        let additional_nonneg = num_cuts + num_lb + num_ub;
        if additional_nonneg > 0 {
            cones.push(ConeSpec::NonNeg {
                dim: additional_nonneg,
            });
        }

        Ok(ProblemData {
            P: base.P.clone(),
            q: base.q.clone(),
            A: a_combined,
            b: b_combined,
            cones,
            var_bounds: None, // Bounds are encoded as constraints
            integrality: None, // Relaxation ignores integrality
        })
    }

    /// Convert the conic problem to a polyhedral master.
    ///
    /// This keeps Zero and NonNeg cones, and relaxes SOC/other cones
    /// (they will be enforced via cuts).
    fn create_polyhedral_relaxation(prob: &MipProblem) -> ProblemData {
        let conic = &prob.conic;

        // Keep only Zero and NonNeg cones
        // SOC and other cones are completely relaxed (no constraints added)
        // They will be enforced via K* cuts from the oracle

        let mut kept_rows = Vec::new();
        let mut kept_cones = Vec::new();
        let mut offset = 0;

        for cone in &conic.cones {
            let dim = cone.dim();
            match cone {
                ConeSpec::Zero { .. } | ConeSpec::NonNeg { .. } => {
                    // Keep these cone types
                    for i in 0..dim {
                        kept_rows.push(offset + i);
                    }
                    kept_cones.push(cone.clone());
                }
                _ => {
                    // Relax other cones (SOC, PSD, EXP, POW)
                    // No rows kept, no constraints in master
                }
            }
            offset += dim;
        }

        // If no rows kept, create a trivial problem
        if kept_rows.is_empty() {
            return ProblemData {
                P: conic.P.clone(),
                q: conic.q.clone(),
                A: CsMat::empty(sprs::CompressedStorage::CSC, 0),
                b: Vec::new(),
                cones: Vec::new(),
                var_bounds: None,
                integrality: None,
            };
        }

        // Extract kept rows from A and b
        let n = conic.num_vars();
        let m_new = kept_rows.len();

        let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
        let mut b_new = Vec::with_capacity(m_new);

        for (new_row, &old_row) in kept_rows.iter().enumerate() {
            // Extract row old_row from A (which is in CSC format)
            for (col_idx, col) in conic.A.outer_iterator().enumerate() {
                for (row_idx, &val) in col.iter() {
                    if row_idx == old_row {
                        triplets.push((new_row, col_idx, val));
                    }
                }
            }
            b_new.push(conic.b[old_row]);
        }

        let a_new = triplets_to_csc(m_new, n, &triplets);

        ProblemData {
            P: conic.P.clone(),
            q: conic.q.clone(),
            A: a_new,
            b: b_new,
            cones: kept_cones,
            var_bounds: None,
            integrality: None,
        }
    }
}

impl MasterBackend for IpmMasterBackend {
    fn initialize(&mut self, prob: &MipProblem) -> MipResult<()> {
        self.n = prob.num_vars();

        // Create polyhedral relaxation
        let base = Self::create_polyhedral_relaxation(prob);
        self.m_base = base.num_constraints();
        self.base_prob = Some(base);

        // Initialize bounds from problem
        self.var_lb = prob.var_lb.clone();
        self.var_ub = prob.var_ub.clone();

        // Clear cuts
        self.cuts.clear();
        self.cut_active.clear();

        Ok(())
    }

    fn add_cut(&mut self, cut: &LinearCut) -> usize {
        let idx = self.cuts.len();
        self.cuts.push(cut.clone());
        self.cut_active.push(true);
        idx
    }

    fn remove_cuts(&mut self, cut_ids: &[usize]) {
        for &id in cut_ids {
            if id < self.cut_active.len() {
                self.cut_active[id] = false;
            }
        }
    }

    fn set_var_bounds(&mut self, var: usize, lb: f64, ub: f64) {
        if var < self.n {
            self.var_lb[var] = lb;
            self.var_ub[var] = ub;
        }
    }

    fn solve(&mut self) -> MipResult<MasterResult> {
        let prob = self.build_master_problem()?;

        let result = solve(&prob, &self.settings).map_err(|e| {
            MipError::MasterSolveError(format!("solver-core error: {}", e))
        })?;

        let status = match result.status {
            SolveStatus::Optimal => MasterStatus::Optimal,
            SolveStatus::PrimalInfeasible => MasterStatus::Infeasible,
            SolveStatus::DualInfeasible | SolveStatus::Unbounded => MasterStatus::Unbounded,
            _ => MasterStatus::NumericalError,
        };

        if status == MasterStatus::Infeasible {
            return Ok(MasterResult::infeasible());
        }

        Ok(MasterResult {
            status,
            x: result.x,
            obj_val: result.obj_val,
            dual_obj: result.obj_val, // IPM gives primal=dual at optimality
            s: result.s,
            z: result.z,
        })
    }

    fn num_cuts(&self) -> usize {
        self.cut_active.iter().filter(|&&a| a).count()
    }

    fn num_vars(&self) -> usize {
        self.n
    }

    fn num_base_constraints(&self) -> usize {
        self.m_base
    }
}

/// Convert triplets to CSC sparse matrix.
fn triplets_to_csc(nrows: usize, ncols: usize, triplets: &[(usize, usize, f64)]) -> CsMat<f64> {
    if triplets.is_empty() {
        return CsMat::empty(sprs::CompressedStorage::CSC, ncols);
    }

    let mut tri = TriMat::new((nrows, ncols));
    for &(row, col, val) in triplets {
        tri.add_triplet(row, col, val);
    }
    tri.to_csc()
}

#[cfg(test)]
mod tests {
    use super::*;
    use solver_core::VarType;

    fn simple_lp() -> MipProblem {
        // min -x0 - x1
        // s.t. x0 + x1 <= 1.5  =>  x0 + x1 + s = 1.5, s >= 0
        // x0, x1 >= 0, <= 1
        // x0 binary
        let n = 2;
        let m = 1;
        let a = CsMat::new_csc((m, n), vec![0, 1, 2], vec![0, 0], vec![1.0, 1.0]);

        let prob = ProblemData {
            P: None,
            q: vec![-1.0, -1.0],
            A: a,
            b: vec![1.5],
            cones: vec![ConeSpec::NonNeg { dim: 1 }],
            var_bounds: Some(vec![
                solver_core::VarBound { var: 0, lower: Some(0.0), upper: Some(1.0) },
                solver_core::VarBound { var: 1, lower: Some(0.0), upper: Some(1.0) },
            ]),
            integrality: Some(vec![VarType::Binary, VarType::Continuous]),
        };

        MipProblem::new(prob).unwrap()
    }

    #[test]
    fn test_ipm_backend_basic() {
        let prob = simple_lp();
        let mut backend = IpmMasterBackend::new(SolverSettings::default());

        backend.initialize(&prob).unwrap();

        assert_eq!(backend.num_vars(), 2);
        assert_eq!(backend.num_cuts(), 0);

        let result = backend.solve().unwrap();
        assert_eq!(result.status, MasterStatus::Optimal);

        // Optimal for LP relaxation: x0 = x1 = 0.75 (constraint binding)
        // or x0 = x1 = 1.0 if bounds are tighter
        // With bounds [0,1] x [0,1] and x0+x1 <= 1.5, optimal is x0=x1=0.75, obj=-1.5
        // But actually with our setup, the constraint is x0+x1+s=1.5 with s>=0
        // So x0+x1 <= 1.5 is correct
    }
}
