//! QPS file format parser for quadratic programming problems.
//!
//! QPS is an extension of MPS format that adds quadratic objective terms.
//! Format specification based on CPLEX and standard conventions.
//!
//! Sections:
//! - NAME: problem name
//! - ROWS: constraint definitions (N=objective, E/L/G=equality/less/greater)
//! - COLUMNS: A matrix coefficients
//! - RHS: right-hand side vector b
//! - RANGES: range constraints (optional)
//! - BOUNDS: variable bounds (optional)
//! - QUADOBJ/QMATRIX: quadratic objective terms (optional)
//! - ENDATA: end marker

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use solver_core::linalg::sparse;
use solver_core::{ConeSpec, ProblemData};

/// Parsed QPS problem data (before conversion to conic form).
#[derive(Debug, Clone)]
pub struct QpsProblem {
    /// Problem name
    pub name: String,
    /// Number of variables
    pub n: usize,
    /// Number of constraints (excluding objective)
    pub m: usize,
    /// Objective sense (1 = minimize, -1 = maximize)
    pub obj_sense: f64,
    /// Linear cost vector q (length n)
    pub q: Vec<f64>,
    /// Quadratic cost matrix P (n x n, upper triangle triplets)
    pub p_triplets: Vec<(usize, usize, f64)>,
    /// Constraint matrix A (m x n, triplets)
    pub a_triplets: Vec<(usize, usize, f64)>,
    /// Constraint lower bounds (length m)
    pub con_lower: Vec<f64>,
    /// Constraint upper bounds (length m)
    pub con_upper: Vec<f64>,
    /// Variable lower bounds (length n)
    pub var_lower: Vec<f64>,
    /// Variable upper bounds (length n)
    pub var_upper: Vec<f64>,
    /// Variable names
    pub var_names: Vec<String>,
    /// Constraint names
    pub con_names: Vec<String>,
}

impl QpsProblem {
    /// Convert to conic ProblemData format.
    ///
    /// Transforms the QP into standard conic form:
    /// - Equality constraints: A_eq x + s_eq = b_eq, s_eq in Zero cone
    /// - Inequality constraints: converted to slack form with NonNeg cone
    /// - Variable bounds: converted to inequality constraints
    pub fn to_problem_data(&self) -> Result<ProblemData> {
        // Count constraint types
        let mut n_eq = 0;
        let mut n_ineq = 0;

        for i in 0..self.m {
            let lb = self.con_lower[i];
            let ub = self.con_upper[i];

            if lb == ub && lb.is_finite() {
                n_eq += 1;
            } else {
                // Range or one-sided inequality
                if lb.is_finite() && lb > f64::NEG_INFINITY {
                    n_ineq += 1; // a'x >= lb
                }
                if ub.is_finite() && ub < f64::INFINITY {
                    n_ineq += 1; // a'x <= ub
                }
            }
        }

        // Count variable bound constraints
        let mut n_var_bounds = 0;
        for j in 0..self.n {
            if self.var_lower[j] > f64::NEG_INFINITY && self.var_lower[j] != 0.0 {
                n_var_bounds += 1;
            } else if self.var_lower[j] == 0.0 {
                n_var_bounds += 1; // x >= 0
            }
            if self.var_upper[j] < f64::INFINITY {
                n_var_bounds += 1;
            }
        }

        let total_constraints = n_eq + n_ineq + n_var_bounds;

        // Build constraint matrix and RHS
        let mut triplets = Vec::new();
        let mut b = Vec::with_capacity(total_constraints);
        let mut row = 0;

        // 1. Equality constraints (Zero cone)
        for i in 0..self.m {
            let lb = self.con_lower[i];
            let ub = self.con_upper[i];

            if lb == ub && lb.is_finite() {
                // Equality: Ax = b
                for &(r, c, v) in &self.a_triplets {
                    if r == i {
                        triplets.push((row, c, v));
                    }
                }
                b.push(lb);
                row += 1;
            }
        }
        let _eq_end = row;

        // 2. Inequality constraints (NonNeg cone)
        // Format: Ax + s = b, s >= 0
        // For a'x <= u: a'x + s = u, s >= 0
        // For a'x >= l: -a'x + s = -l, s >= 0
        for i in 0..self.m {
            let lb = self.con_lower[i];
            let ub = self.con_upper[i];

            if lb == ub && lb.is_finite() {
                continue; // Already handled as equality
            }

            // Upper bound: a'x <= ub
            if ub.is_finite() && ub < f64::INFINITY {
                for &(r, c, v) in &self.a_triplets {
                    if r == i {
                        triplets.push((row, c, v));
                    }
                }
                b.push(ub);
                row += 1;
            }

            // Lower bound: a'x >= lb => -a'x <= -lb
            if lb.is_finite() && lb > f64::NEG_INFINITY {
                for &(r, c, v) in &self.a_triplets {
                    if r == i {
                        triplets.push((row, c, -v));
                    }
                }
                b.push(-lb);
                row += 1;
            }
        }

        // 3. Variable bounds (NonNeg cone)
        // x_j >= l: -x_j + s = -l, s >= 0
        // x_j <= u: x_j + s = u, s >= 0
        for j in 0..self.n {
            let lb = self.var_lower[j];
            let ub = self.var_upper[j];

            // Lower bound
            if lb > f64::NEG_INFINITY {
                triplets.push((row, j, -1.0));
                b.push(-lb);
                row += 1;
            }

            // Upper bound
            if ub < f64::INFINITY {
                triplets.push((row, j, 1.0));
                b.push(ub);
                row += 1;
            }
        }

        assert_eq!(row, total_constraints);

        // Build sparse matrices
        let a = sparse::from_triplets(total_constraints, self.n, triplets);

        let p = if self.p_triplets.is_empty() {
            None
        } else {
            Some(sparse::from_triplets(self.n, self.n, self.p_triplets.clone()))
        };

        // Scale objective by sense
        let q: Vec<f64> = self.q.iter().map(|&v| v * self.obj_sense).collect();

        // Build cone specification
        let mut cones = Vec::new();
        if n_eq > 0 {
            cones.push(ConeSpec::Zero { dim: n_eq });
        }
        let ineq_total = n_ineq + n_var_bounds;
        if ineq_total > 0 {
            cones.push(ConeSpec::NonNeg { dim: ineq_total });
        }

        Ok(ProblemData {
            P: p,
            q,
            A: a,
            b,
            cones,
            var_bounds: None,
            integrality: None,
        })
    }
}

/// Parse a QPS file.
pub fn parse_qps<P: AsRef<Path>>(path: P) -> Result<QpsProblem> {
    let file = File::open(path.as_ref())
        .with_context(|| format!("Failed to open QPS file: {:?}", path.as_ref()))?;
    let reader = BufReader::new(file);

    let mut name = String::new();
    let mut obj_row: Option<String> = None;
    let mut row_types: HashMap<String, char> = HashMap::new();
    let mut row_order: Vec<String> = Vec::new();
    let mut var_map: HashMap<String, usize> = HashMap::new();
    let mut var_names: Vec<String> = Vec::new();
    let mut con_map: HashMap<String, usize> = HashMap::new();
    let mut con_names: Vec<String> = Vec::new();

    let mut a_triplets: Vec<(usize, usize, f64)> = Vec::new();
    let mut q_coeffs: HashMap<String, f64> = HashMap::new();
    let mut p_triplets: Vec<(usize, usize, f64)> = Vec::new();

    let mut rhs: HashMap<String, f64> = HashMap::new();
    let mut ranges: HashMap<String, f64> = HashMap::new();
    let mut var_lower: HashMap<String, f64> = HashMap::new();
    let mut var_upper: HashMap<String, f64> = HashMap::new();

    let mut section = String::new();

    for line_result in reader.lines() {
        let line = line_result?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('*') {
            continue;
        }

        // Check for section headers
        if line.starts_with("NAME") {
            name = line.split_whitespace().nth(1).unwrap_or("unknown").to_string();
            section = "NAME".to_string();
            continue;
        } else if line == "ROWS" {
            section = "ROWS".to_string();
            continue;
        } else if line == "COLUMNS" {
            section = "COLUMNS".to_string();
            continue;
        } else if line == "RHS" {
            section = "RHS".to_string();
            continue;
        } else if line == "RANGES" {
            section = "RANGES".to_string();
            continue;
        } else if line == "BOUNDS" {
            section = "BOUNDS".to_string();
            continue;
        } else if line == "QUADOBJ" || line == "QMATRIX" || line == "QSECTION" {
            section = "QUADOBJ".to_string();
            continue;
        } else if line == "ENDATA" {
            break;
        } else if line.starts_with("OBJSENSE") {
            section = "OBJSENSE".to_string();
            continue;
        }

        // Parse section content
        match section.as_str() {
            "OBJSENSE" => {
                // Handle OBJSENSE MAX or MIN
                if line.contains("MAX") {
                    // Will negate objective later
                }
            }
            "ROWS" => {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let rtype = parts[0].chars().next().unwrap_or('E');
                    let rname = parts[1].to_string();

                    if rtype == 'N' {
                        // Objective row
                        if obj_row.is_none() {
                            obj_row = Some(rname.clone());
                        }
                    } else {
                        // Constraint row
                        let idx = con_names.len();
                        con_map.insert(rname.clone(), idx);
                        con_names.push(rname.clone());
                    }
                    row_types.insert(rname.clone(), rtype);
                    row_order.push(rname);
                }
            }
            "COLUMNS" => {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    let var_name = parts[0].to_string();

                    // Get or create variable index
                    let var_idx = *var_map.entry(var_name.clone()).or_insert_with(|| {
                        let idx = var_names.len();
                        var_names.push(var_name.clone());
                        idx
                    });

                    // Parse pairs of (row_name, value)
                    let mut i = 1;
                    while i + 1 < parts.len() {
                        let row_name = parts[i];
                        let value: f64 = parts[i + 1].parse().unwrap_or(0.0);

                        if Some(row_name.to_string()) == obj_row {
                            // Objective coefficient
                            q_coeffs.insert(var_name.clone(), value);
                        } else if let Some(&con_idx) = con_map.get(row_name) {
                            // Constraint coefficient
                            a_triplets.push((con_idx, var_idx, value));
                        }

                        i += 2;
                    }
                }
            }
            "RHS" => {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    // Skip RHS name (first field), parse pairs
                    let mut i = 1;
                    while i + 1 < parts.len() {
                        let row_name = parts[i].to_string();
                        let value: f64 = parts[i + 1].parse().unwrap_or(0.0);
                        rhs.insert(row_name, value);
                        i += 2;
                    }
                }
            }
            "RANGES" => {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    let mut i = 1;
                    while i + 1 < parts.len() {
                        let row_name = parts[i].to_string();
                        let value: f64 = parts[i + 1].parse().unwrap_or(0.0);
                        ranges.insert(row_name, value.abs());
                        i += 2;
                    }
                }
            }
            "BOUNDS" => {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    let btype = parts[0];
                    let var_name = parts[2].to_string();
                    let value: f64 = if parts.len() > 3 {
                        parts[3].parse().unwrap_or(0.0)
                    } else {
                        0.0
                    };

                    match btype {
                        "LO" => {
                            var_lower.insert(var_name, value);
                        }
                        "UP" => {
                            var_upper.insert(var_name, value);
                        }
                        "FX" => {
                            var_lower.insert(var_name.clone(), value);
                            var_upper.insert(var_name, value);
                        }
                        "FR" => {
                            var_lower.insert(var_name.clone(), f64::NEG_INFINITY);
                            var_upper.insert(var_name, f64::INFINITY);
                        }
                        "MI" => {
                            var_lower.insert(var_name, f64::NEG_INFINITY);
                        }
                        "PL" => {
                            var_upper.insert(var_name, f64::INFINITY);
                        }
                        "BV" => {
                            // Binary variable
                            var_lower.insert(var_name.clone(), 0.0);
                            var_upper.insert(var_name, 1.0);
                        }
                        _ => {}
                    }
                }
            }
            "QUADOBJ" => {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    let var1 = parts[0].to_string();
                    let var2 = parts[1].to_string();
                    let value: f64 = parts[2].parse().unwrap_or(0.0);

                    if let (Some(&i), Some(&j)) = (var_map.get(&var1), var_map.get(&var2)) {
                        // Store upper triangle only
                        let (row, col) = if i <= j { (i, j) } else { (j, i) };
                        // QPS stores Q such that obj = 0.5 x'Qx, so we use value directly
                        p_triplets.push((row, col, value));
                    }
                }
            }
            _ => {}
        }
    }

    let n = var_names.len();
    let m = con_names.len();

    if n == 0 {
        return Err(anyhow!("No variables found in QPS file"));
    }

    // Build vectors
    let q: Vec<f64> = var_names
        .iter()
        .map(|name| *q_coeffs.get(name).unwrap_or(&0.0))
        .collect();

    // Build constraint bounds based on row types
    let mut con_lower = vec![f64::NEG_INFINITY; m];
    let mut con_upper = vec![f64::INFINITY; m];

    for (name, &idx) in &con_map {
        let rtype = row_types.get(name).copied().unwrap_or('E');
        let rhs_val = rhs.get(name).copied().unwrap_or(0.0);
        let range_val = ranges.get(name).copied().unwrap_or(0.0);

        match rtype {
            'E' => {
                // Equality
                con_lower[idx] = rhs_val;
                con_upper[idx] = rhs_val;
            }
            'L' => {
                // Less than or equal
                con_upper[idx] = rhs_val;
                if range_val > 0.0 {
                    con_lower[idx] = rhs_val - range_val;
                }
            }
            'G' => {
                // Greater than or equal
                con_lower[idx] = rhs_val;
                if range_val > 0.0 {
                    con_upper[idx] = rhs_val + range_val;
                }
            }
            _ => {}
        }
    }

    // Build variable bounds (default: x >= 0)
    let var_lower_vec: Vec<f64> = var_names
        .iter()
        .map(|name| var_lower.get(name).copied().unwrap_or(0.0))
        .collect();

    let var_upper_vec: Vec<f64> = var_names
        .iter()
        .map(|name| var_upper.get(name).copied().unwrap_or(f64::INFINITY))
        .collect();

    Ok(QpsProblem {
        name,
        n,
        m,
        obj_sense: 1.0, // Minimize by default
        q,
        p_triplets,
        a_triplets,
        con_lower,
        con_upper,
        var_lower: var_lower_vec,
        var_upper: var_upper_vec,
        var_names,
        con_names,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_qp_conversion() {
        // Create a simple QP manually
        let qps = QpsProblem {
            name: "test".to_string(),
            n: 2,
            m: 1,
            obj_sense: 1.0,
            q: vec![1.0, 1.0],
            p_triplets: vec![(0, 0, 2.0), (1, 1, 2.0)], // P = 2I
            a_triplets: vec![(0, 0, 1.0), (0, 1, 1.0)], // x1 + x2 = 1
            con_lower: vec![1.0],
            con_upper: vec![1.0],
            var_lower: vec![0.0, 0.0],
            var_upper: vec![f64::INFINITY, f64::INFINITY],
            var_names: vec!["x1".to_string(), "x2".to_string()],
            con_names: vec!["c1".to_string()],
        };

        let prob = qps.to_problem_data().unwrap();

        assert_eq!(prob.num_vars(), 2);
        assert!(prob.P.is_some());
        assert_eq!(prob.q.len(), 2);
    }
}
