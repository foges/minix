//! Conic subproblem oracle.
//!
//! The oracle validates integer candidate solutions by fixing integer variables
//! and solving the resulting continuous conic subproblem.

use solver_core::{solve, ConeSpec, ProblemData, SolveStatus, SolverSettings};
use sprs::{CsMat, TriMat};

use crate::error::{MipError, MipResult};
use crate::model::MipProblem;

/// Result from the conic oracle.
#[derive(Debug, Clone)]
pub struct OracleResult {
    /// Whether the fixed-integer subproblem is feasible.
    pub feasible: bool,

    /// If feasible, the continuous solution expanded to full space.
    pub x: Option<Vec<f64>>,

    /// If feasible, the slack variables.
    pub s: Option<Vec<f64>>,

    /// Dual variables z (for K* cut generation).
    pub z: Option<Vec<f64>>,

    /// Objective value (infinity if infeasible).
    pub obj_val: f64,
}

impl OracleResult {
    /// Create an infeasible result.
    pub fn infeasible(z: Vec<f64>) -> Self {
        Self {
            feasible: false,
            x: None,
            s: None,
            z: Some(z),
            obj_val: f64::INFINITY,
        }
    }

    /// Create a feasible result.
    pub fn feasible(x: Vec<f64>, s: Vec<f64>, z: Vec<f64>, obj_val: f64) -> Self {
        Self {
            feasible: true,
            x: Some(x),
            s: Some(s),
            z: Some(z),
            obj_val,
        }
    }
}

/// Conic subproblem oracle.
///
/// Given an integer candidate solution, the oracle fixes integer variables
/// and solves the continuous conic subproblem using solver-core.
pub struct ConicOracle {
    /// Original problem (with all cones).
    original: ProblemData,

    /// Integer variable indices.
    integer_vars: Vec<usize>,

    /// Solver settings.
    settings: SolverSettings,
}

impl ConicOracle {
    /// Create a new conic oracle.
    pub fn new(prob: &MipProblem, settings: SolverSettings) -> Self {
        Self {
            original: prob.conic.clone(),
            integer_vars: prob.integer_vars.clone(),
            settings,
        }
    }

    /// Validate an integer candidate solution.
    ///
    /// Fixes integer variables to their candidate values and solves the
    /// continuous conic subproblem.
    ///
    /// Returns:
    /// - If feasible: the polished continuous solution
    /// - If infeasible: dual variables for K* cut generation
    pub fn validate(&self, x_candidate: &[f64]) -> MipResult<OracleResult> {
        // Build subproblem with fixed integers
        let subproblem = self.build_fixed_problem(x_candidate);

        // Solve the subproblem
        let result = solve(&subproblem, &self.settings).map_err(|e| {
            MipError::OracleError(format!("solver-core error: {}", e))
        })?;

        match result.status {
            SolveStatus::Optimal => {
                // Feasible! Expand solution to full space
                let x_full = self.expand_solution(&result.x, x_candidate);
                Ok(OracleResult::feasible(x_full, result.s, result.z, result.obj_val))
            }
            SolveStatus::PrimalInfeasible => {
                // Infeasible - return dual certificate for K* cuts
                Ok(OracleResult::infeasible(result.z))
            }
            SolveStatus::DualInfeasible | SolveStatus::Unbounded => {
                // Unbounded shouldn't happen if master was bounded
                Err(MipError::OracleError("Subproblem unbounded".to_string()))
            }
            _ => {
                // Numerical issues - treat as infeasible conservatively
                Err(MipError::OracleError(format!(
                    "Subproblem solve failed: {:?}",
                    result.status
                )))
            }
        }
    }

    /// Build subproblem with integer variables fixed.
    ///
    /// We fix variables by adding equality constraints: x_i = v_i for each integer var.
    /// This is done by adding rows to A with Zero cone slacks.
    fn build_fixed_problem(&self, x_candidate: &[f64]) -> ProblemData {
        let n = self.original.num_vars();
        let m_orig = self.original.num_constraints();
        let num_fixed = self.integer_vars.len();

        if num_fixed == 0 {
            // No integers to fix, return original problem
            return self.original.clone();
        }

        // Build new A matrix: [A_orig; I_fixed]
        // where I_fixed has 1 in column i for each integer variable i
        let m_new = m_orig + num_fixed;

        let mut triplets: Vec<(usize, usize, f64)> = Vec::new();

        // Copy original A
        for (col_idx, col) in self.original.A.outer_iterator().enumerate() {
            for (row_idx, &val) in col.iter() {
                triplets.push((row_idx, col_idx, val));
            }
        }

        // Add fixing constraints: x_i = v_i  =>  x_i + s = v_i with s in Zero cone
        for (fix_idx, &var) in self.integer_vars.iter().enumerate() {
            triplets.push((m_orig + fix_idx, var, 1.0));
        }

        let a_new = triplets_to_csc(m_new, n, &triplets);

        // Build new b vector
        let mut b_new = self.original.b.clone();
        for &var in &self.integer_vars {
            b_new.push(x_candidate[var]);
        }

        // Build new cone specification
        let mut cones_new = self.original.cones.clone();
        cones_new.push(ConeSpec::Zero { dim: num_fixed });

        ProblemData {
            P: self.original.P.clone(),
            q: self.original.q.clone(),
            A: a_new,
            b: b_new,
            cones: cones_new,
            var_bounds: self.original.var_bounds.clone(),
            integrality: None, // No integrality in continuous subproblem
        }
    }

    /// Expand reduced solution to full variable space.
    ///
    /// The subproblem has all original variables, so this is mostly a copy,
    /// but we ensure integer variables are exactly at their fixed values.
    fn expand_solution(&self, x_sub: &[f64], x_candidate: &[f64]) -> Vec<f64> {
        let mut x_full = x_sub.to_vec();

        // Ensure integer vars are exactly at candidate values
        for &var in &self.integer_vars {
            x_full[var] = x_candidate[var].round();
        }

        x_full
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

    fn simple_mip() -> MipProblem {
        // min x0 + x1
        // s.t. x0 + x1 >= 1  (as -x0 - x1 + s = -1, s >= 0)
        // x0 binary [0,1], x1 continuous [0, inf)
        let n = 2;
        let m = 1;
        let a = CsMat::new_csc((m, n), vec![0, 1, 2], vec![0, 0], vec![-1.0, -1.0]);

        let prob = ProblemData {
            P: None,
            q: vec![1.0, 1.0],
            A: a,
            b: vec![-1.0],
            cones: vec![ConeSpec::NonNeg { dim: 1 }],
            var_bounds: Some(vec![
                solver_core::VarBound { var: 0, lower: Some(0.0), upper: Some(1.0) },
                solver_core::VarBound { var: 1, lower: Some(0.0), upper: None },
            ]),
            integrality: Some(vec![VarType::Binary, VarType::Continuous]),
        };

        MipProblem::new(prob).unwrap()
    }

    #[test]
    fn test_oracle_feasible() {
        let mip_prob = simple_mip();
        let oracle = ConicOracle::new(&mip_prob, SolverSettings::default());

        // x0 = 1, x1 = 0 should be feasible (1 + 0 = 1 >= 1)
        let result = oracle.validate(&[1.0, 0.0]).unwrap();
        assert!(result.feasible);
        assert!(result.x.is_some());
    }
}
