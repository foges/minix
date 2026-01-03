//! MIP problem representation.

use solver_core::{ProblemData, VarType};

use crate::error::{MipError, MipResult};

/// Mixed-integer problem wrapper.
///
/// Extracts and organizes integrality information from a `ProblemData`.
#[derive(Clone)]
pub struct MipProblem {
    /// Original conic problem data.
    pub conic: ProblemData,

    /// Indices of integer variables (includes binary).
    pub integer_vars: Vec<usize>,

    /// Indices of binary variables (subset of integer_vars).
    pub binary_vars: Vec<usize>,

    /// Current lower bounds for all variables.
    pub var_lb: Vec<f64>,

    /// Current upper bounds for all variables.
    pub var_ub: Vec<f64>,
}

impl MipProblem {
    /// Create a MipProblem from ProblemData.
    ///
    /// Extracts integer/binary variable indices and initializes bounds.
    pub fn new(prob: ProblemData) -> MipResult<Self> {
        let n = prob.num_vars();

        // Extract integrality information
        let mut integer_vars = Vec::new();
        let mut binary_vars = Vec::new();

        if let Some(ref integrality) = prob.integrality {
            for (i, var_type) in integrality.iter().enumerate() {
                match var_type {
                    VarType::Integer => {
                        integer_vars.push(i);
                    }
                    VarType::Binary => {
                        integer_vars.push(i);
                        binary_vars.push(i);
                    }
                    VarType::Continuous => {}
                }
            }
        }

        // Initialize bounds
        let mut var_lb = vec![f64::NEG_INFINITY; n];
        let mut var_ub = vec![f64::INFINITY; n];

        // Apply explicit bounds from problem
        if let Some(ref bounds) = prob.var_bounds {
            for bound in bounds {
                if bound.var >= n {
                    return Err(MipError::InvalidProblem(format!(
                        "Bound for variable {} but only {} variables",
                        bound.var, n
                    )));
                }
                if let Some(lb) = bound.lower {
                    var_lb[bound.var] = lb;
                }
                if let Some(ub) = bound.upper {
                    var_ub[bound.var] = ub;
                }
            }
        }

        // Binary variables have implicit [0, 1] bounds
        for &i in &binary_vars {
            var_lb[i] = var_lb[i].max(0.0);
            var_ub[i] = var_ub[i].min(1.0);
        }

        Ok(Self {
            conic: prob,
            integer_vars,
            binary_vars,
            var_lb,
            var_ub,
        })
    }

    /// Number of variables.
    pub fn num_vars(&self) -> usize {
        self.conic.num_vars()
    }

    /// Number of constraints.
    pub fn num_constraints(&self) -> usize {
        self.conic.num_constraints()
    }

    /// Number of integer variables (including binary).
    pub fn num_integers(&self) -> usize {
        self.integer_vars.len()
    }

    /// Check if a solution is integer-feasible within tolerance.
    pub fn is_integer_feasible(&self, x: &[f64], tol: f64) -> bool {
        for &i in &self.integer_vars {
            let val = x[i];
            let frac = (val - val.round()).abs();
            if frac > tol {
                return false;
            }
        }
        true
    }

    /// Get the fractionality of a variable (distance to nearest integer).
    pub fn fractionality(&self, val: f64) -> f64 {
        let frac = val.fract().abs();
        frac.min(1.0 - frac)
    }

    /// Round integer variables to nearest integer.
    pub fn round_integers(&self, x: &mut [f64]) {
        for &i in &self.integer_vars {
            x[i] = x[i].round();
        }
    }

    /// Get fractional integer variables and their values.
    ///
    /// Returns (var_index, current_value, fractionality) for each fractional variable.
    pub fn get_fractional_vars(&self, x: &[f64], tol: f64) -> Vec<(usize, f64, f64)> {
        let mut result = Vec::new();
        for &i in &self.integer_vars {
            let val = x[i];
            let frac = self.fractionality(val);
            if frac > tol {
                result.push((i, val, frac));
            }
        }
        result
    }

    /// Check if variable bounds are consistent (lb <= ub).
    pub fn bounds_feasible(&self) -> bool {
        for i in 0..self.num_vars() {
            if self.var_lb[i] > self.var_ub[i] + 1e-9 {
                return false;
            }
        }
        true
    }

    /// Check if a point satisfies variable bounds.
    pub fn satisfies_bounds(&self, x: &[f64], tol: f64) -> bool {
        for i in 0..self.num_vars() {
            if x[i] < self.var_lb[i] - tol || x[i] > self.var_ub[i] + tol {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solver_core::ConeSpec;
    use sprs::CsMat;

    fn simple_milp() -> ProblemData {
        // min x0 + x1
        // s.t. x0 + x1 >= 1  (as x0 + x1 + s = 1, s <= 0 ... actually s >= 0 for NonNeg)
        // x0 binary, x1 continuous
        let n = 2;
        let m = 1;
        let a = CsMat::new_csc(
            (m, n),
            vec![0, 1, 2],
            vec![0, 0],
            vec![1.0, 1.0],
        );

        ProblemData {
            P: None,
            q: vec![1.0, 1.0],
            A: a,
            b: vec![1.0],
            cones: vec![ConeSpec::NonNeg { dim: 1 }],
            var_bounds: None,
            integrality: Some(vec![VarType::Binary, VarType::Continuous]),
        }
    }

    #[test]
    fn test_mip_problem_creation() {
        let prob = simple_milp();
        let mip = MipProblem::new(prob).unwrap();

        assert_eq!(mip.num_vars(), 2);
        assert_eq!(mip.num_integers(), 1);
        assert_eq!(mip.integer_vars, vec![0]);
        assert_eq!(mip.binary_vars, vec![0]);

        // Binary var should have [0, 1] bounds
        assert_eq!(mip.var_lb[0], 0.0);
        assert_eq!(mip.var_ub[0], 1.0);
    }

    #[test]
    fn test_integer_feasibility() {
        let prob = simple_milp();
        let mip = MipProblem::new(prob).unwrap();

        // x0 = 1.0 is integer
        assert!(mip.is_integer_feasible(&[1.0, 0.5], 1e-6));

        // x0 = 0.5 is not integer
        assert!(!mip.is_integer_feasible(&[0.5, 0.5], 1e-6));

        // x0 = 0.9999999 is integer within tolerance
        assert!(mip.is_integer_feasible(&[0.9999999, 0.5], 1e-6));
    }

    #[test]
    fn test_fractionality() {
        let prob = simple_milp();
        let mip = MipProblem::new(prob).unwrap();

        assert!((mip.fractionality(0.5) - 0.5).abs() < 1e-10);
        assert!((mip.fractionality(0.3) - 0.3).abs() < 1e-10);
        assert!((mip.fractionality(0.7) - 0.3).abs() < 1e-10);
        assert!(mip.fractionality(1.0) < 1e-10);
        assert!(mip.fractionality(2.0) < 1e-10);
    }
}
