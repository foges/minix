//! MATLAB .mat file parser for benchmark data.
//!
//! Loads QP problem data in OSQP format from MATLAB v5 .mat files.
//! Used for Maros-Meszaros and NETLIB benchmarks from ClarabelBenchmarks.

use anyhow::{bail, Context, Result};
use matfile::{MatFile, NumericData};
use solver_core::linalg::sparse;
use solver_core::{ConeSpec, ProblemData};
use std::path::Path;

/// QP problem data in OSQP format.
///
/// Problem form:
///   minimize    (1/2) x'Px + q'x
///   subject to  l <= Ax <= u
#[derive(Debug, Clone)]
pub struct OsqpProblem {
    /// Problem name
    pub name: String,
    /// Number of variables
    pub n: usize,
    /// Number of constraints
    pub m: usize,
    /// Hessian matrix P (upper triangular, may be empty for LP)
    pub p_triplets: Vec<(usize, usize, f64)>,
    /// Constraint matrix A
    pub a_triplets: Vec<(usize, usize, f64)>,
    /// Linear cost vector q
    pub q: Vec<f64>,
    /// Constraint lower bounds l
    pub l: Vec<f64>,
    /// Constraint upper bounds u
    pub u: Vec<f64>,
    /// Constant term in objective (r)
    pub r: f64,
}

impl OsqpProblem {
    /// Convert to conic form ProblemData.
    ///
    /// Separates equality constraints (l == u) and inequality constraints.
    /// Two-sided inequalities l <= Ax <= u become:
    ///   Ax <= u  =>  Ax + s = u, s >= 0
    ///  -Ax <= -l => -Ax + s = -l, s >= 0
    pub fn to_problem_data(&self) -> Result<ProblemData> {
        const INF_THRESH: f64 = 1e19;

        // Separate equality and inequality constraints
        let mut eq_rows = Vec::new();
        let mut ineq_rows = Vec::new();

        for i in 0..self.m {
            let li = self.l[i];
            let ui = self.u[i];

            if (li - ui).abs() < 1e-12 {
                // Equality constraint
                eq_rows.push((i, li));
            } else {
                // Inequality constraint(s)
                ineq_rows.push((i, li, ui));
            }
        }

        // Build equality constraint matrix and RHS
        let n_eq = eq_rows.len();
        let mut eq_triplets = Vec::new();
        let mut b_eq = Vec::with_capacity(n_eq);

        for (new_row, (orig_row, rhs)) in eq_rows.iter().enumerate() {
            b_eq.push(*rhs);
            // Copy row from A
            for (row, col, val) in &self.a_triplets {
                if *row == *orig_row {
                    eq_triplets.push((new_row, *col, *val));
                }
            }
        }

        // Build inequality constraint matrix and RHS
        // For each two-sided inequality l <= Ax <= u:
        //   Ax <= u  (if u < inf)
        //  -Ax <= -l (if l > -inf)
        let mut ineq_triplets = Vec::new();
        let mut b_ineq = Vec::new();
        let mut ineq_row = 0;

        for (orig_row, li, ui) in &ineq_rows {
            // Upper bound: Ax <= u
            if *ui < INF_THRESH {
                b_ineq.push(*ui);
                for (row, col, val) in &self.a_triplets {
                    if *row == *orig_row {
                        ineq_triplets.push((ineq_row, *col, *val));
                    }
                }
                ineq_row += 1;
            }

            // Lower bound: -Ax <= -l
            if *li > -INF_THRESH {
                b_ineq.push(-*li);
                for (row, col, val) in &self.a_triplets {
                    if *row == *orig_row {
                        ineq_triplets.push((ineq_row, *col, -*val));
                    }
                }
                ineq_row += 1;
            }
        }

        let n_ineq = b_ineq.len();

        // Build combined constraint matrix: [A_eq; A_ineq] with slack columns
        // For inequalities: A_ineq * x + s = b_ineq, s >= 0
        // So we have: [A_eq 0; A_ineq -I] * [x; s] = [b_eq; b_ineq]

        let total_m = n_eq + n_ineq;
        let total_n = self.n + n_ineq; // Add slack variables

        let mut all_triplets = Vec::new();

        // Equality rows (no slack)
        for (row, col, val) in &eq_triplets {
            all_triplets.push((*row, *col, *val));
        }

        // Inequality rows with slack
        for (row, col, val) in &ineq_triplets {
            all_triplets.push((n_eq + *row, *col, *val));
        }

        // Slack variable columns: -I
        for i in 0..n_ineq {
            all_triplets.push((n_eq + i, self.n + i, -1.0));
        }

        let a = sparse::from_triplets(total_m, total_n, all_triplets);

        // Combined RHS
        let mut b = Vec::with_capacity(total_m);
        b.extend_from_slice(&b_eq);
        b.extend_from_slice(&b_ineq);

        // Objective: extend q with zeros for slack variables
        let mut q_extended = self.q.clone();
        q_extended.resize(total_n, 0.0);

        // Build P matrix (extend with zeros for slack variables)
        let p = if self.p_triplets.is_empty() {
            None
        } else {
            let p_csc = sparse::from_triplets(total_n, total_n, self.p_triplets.clone());
            Some(p_csc)
        };

        // Cones: Zero for equalities, NonNeg for slacks
        let cones = vec![
            ConeSpec::Zero { dim: n_eq },
            ConeSpec::NonNeg { dim: n_ineq },
        ];

        Ok(ProblemData {
            P: p,
            q: q_extended,
            A: a,
            b,
            cones,
            var_bounds: None,
            integrality: None,
        })
    }
}

/// Parse a MATLAB .mat file containing an OSQP-format QP problem.
pub fn parse_mat<P: AsRef<Path>>(path: P) -> Result<OsqpProblem> {
    let path = path.as_ref();
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let mat_file =
        MatFile::parse(std::fs::File::open(path)?).context("Failed to parse MAT file")?;

    // Extract arrays
    let n = get_scalar(&mat_file, "n")? as usize;
    let m = get_scalar(&mat_file, "m")? as usize;

    let q = get_vector(&mat_file, "q")?;
    let l = get_vector(&mat_file, "l")?;
    let u = get_vector(&mat_file, "u")?;

    let r = get_scalar(&mat_file, "r").unwrap_or(0.0);

    // Get sparse matrices P and A
    let p_triplets = get_sparse_matrix(&mat_file, "P").unwrap_or_default();
    let a_triplets = get_sparse_matrix(&mat_file, "A")?;

    Ok(OsqpProblem {
        name,
        n,
        m,
        p_triplets,
        a_triplets,
        q,
        l,
        u,
        r,
    })
}

/// Extract a scalar value from a MAT file.
fn get_scalar(mat: &MatFile, name: &str) -> Result<f64> {
    let array = mat
        .find_by_name(name)
        .ok_or_else(|| anyhow::anyhow!("Missing array: {}", name))?;

    match array.data() {
        NumericData::Double { real, .. } => {
            if real.is_empty() {
                bail!("Empty array: {}", name);
            }
            Ok(real[0])
        }
        NumericData::Single { real, .. } => {
            if real.is_empty() {
                bail!("Empty array: {}", name);
            }
            Ok(real[0] as f64)
        }
        NumericData::Int64 { real, .. } => {
            if real.is_empty() {
                bail!("Empty array: {}", name);
            }
            Ok(real[0] as f64)
        }
        NumericData::UInt64 { real, .. } => {
            if real.is_empty() {
                bail!("Empty array: {}", name);
            }
            Ok(real[0] as f64)
        }
        NumericData::Int32 { real, .. } => {
            if real.is_empty() {
                bail!("Empty array: {}", name);
            }
            Ok(real[0] as f64)
        }
        NumericData::UInt32 { real, .. } => {
            if real.is_empty() {
                bail!("Empty array: {}", name);
            }
            Ok(real[0] as f64)
        }
        _ => bail!("Unsupported numeric type for {}", name),
    }
}

/// Extract a vector from a MAT file.
fn get_vector(mat: &MatFile, name: &str) -> Result<Vec<f64>> {
    let array = mat
        .find_by_name(name)
        .ok_or_else(|| anyhow::anyhow!("Missing array: {}", name))?;

    match array.data() {
        NumericData::Double { real, .. } => Ok(real.clone()),
        NumericData::Single { real, .. } => Ok(real.iter().map(|&x| x as f64).collect()),
        _ => bail!("Unsupported numeric type for vector {}", name),
    }
}

/// Extract a matrix from a MAT file as triplets.
/// Note: matfile crate v0.5 ignores sparse arrays, so this only handles dense matrices.
/// Sparse arrays will not be found by find_by_name and will return an error.
fn get_sparse_matrix(mat: &MatFile, name: &str) -> Result<Vec<(usize, usize, f64)>> {
    let array = mat.find_by_name(name).ok_or_else(|| {
        anyhow::anyhow!(
            "Missing array: {} (note: sparse arrays not supported by matfile crate)",
            name
        )
    })?;

    // Get dimensions
    let shape = array.size();
    if shape.len() != 2 {
        bail!("Expected 2D array for {}, got {}D", name, shape.len());
    }
    let nrows = shape[0];
    let ncols = shape[1];

    // Dense matrix - convert to sparse triplets
    let values: Vec<f64> = match array.data() {
        NumericData::Double { real, .. } => real.clone(),
        NumericData::Single { real, .. } => real.iter().map(|&x| x as f64).collect(),
        _ => bail!("Unsupported numeric type for matrix {}", name),
    };

    let mut triplets = Vec::new();
    for col in 0..ncols {
        for row in 0..nrows {
            let val = values[row + col * nrows]; // Column-major
            if val.abs() > 1e-20 {
                triplets.push((row, col, val));
            }
        }
    }
    Ok(triplets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_osqp_to_conic() {
        // Simple 2-var, 2-constraint problem:
        // min x + 2y
        // s.t. x + y = 1 (equality)
        //      x >= 0 (inequality: 0 <= x <= inf)
        let prob = OsqpProblem {
            name: "test".to_string(),
            n: 2,
            m: 2,
            p_triplets: vec![],
            a_triplets: vec![
                (0, 0, 1.0),
                (0, 1, 1.0), // Row 0: x + y
                (1, 0, 1.0), // Row 1: x
            ],
            q: vec![1.0, 2.0],
            l: vec![1.0, 0.0],  // Equality at 1, lower bound 0
            u: vec![1.0, 1e20], // Equality at 1, upper bound inf
            r: 0.0,
        };

        let conic = prob.to_problem_data().unwrap();
        assert_eq!(conic.num_vars(), 3); // 2 original + 1 slack
        assert_eq!(conic.cones.len(), 2);
    }
}
