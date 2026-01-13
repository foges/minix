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
        let mut row_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); self.m];
        for &(r, c, v) in &self.a_triplets {
            row_entries[r].push((c, v));
        }

        // 1. Equality constraints (Zero cone)
        for i in 0..self.m {
            let lb = self.con_lower[i];
            let ub = self.con_upper[i];

            if lb == ub && lb.is_finite() {
                // Equality: Ax = b
                for &(c, v) in &row_entries[i] {
                    triplets.push((row, c, v));
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
                for &(c, v) in &row_entries[i] {
                    triplets.push((row, c, v));
                }
                b.push(ub);
                row += 1;
            }

            // Lower bound: a'x >= lb => -a'x <= -lb
            if lb.is_finite() && lb > f64::NEG_INFINITY {
                for &(c, v) in &row_entries[i] {
                    triplets.push((row, c, -v));
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

        // Scale objective by sense.
        //
        // Note: For quadratic objectives, the QP form is (1/2) x'P x + q'x.
        // Converting MAX to MIN requires negating *both* q and P.
        let p = if self.p_triplets.is_empty() {
            None
        } else {
            let p_triplets: Vec<(usize, usize, f64)> = self
                .p_triplets
                .iter()
                .map(|&(i, j, v)| (i, j, v * self.obj_sense))
                .collect();
            Some(sparse::from_triplets(self.n, self.n, p_triplets))
        };

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

    /// Convert to SOCP form (no P matrix, quadratic term as rotated SOC constraint).
    ///
    /// Transforms QP: min (1/2) x'Px + q'x s.t. Ax + s = b, s ∈ K
    /// Into SOCP:     min t + q'x s.t. Ax + s = b, s ∈ K, (t, 1/2, Lx) ∈ RSOC
    ///
    /// where P = L'L (Cholesky factorization) and RSOC means 2*t*(1/2) >= ||Lx||²
    ///
    /// The rotated SOC (RSOC) is converted to standard SOC via:
    /// (u, v, w) ∈ RSOC iff ((u+v)/√2, (u-v)/√2, w) ∈ SOC
    ///
    /// This is the form CVXPY sends to conic solvers.
    pub fn to_socp_form(&self) -> Result<ProblemData> {
        // First build the standard QP form to get constraints
        let qp = self.to_problem_data()?;

        // If no quadratic term, this is already an LP - just return it without P
        if self.p_triplets.is_empty() {
            return Ok(ProblemData {
                P: None,
                q: qp.q,
                A: qp.A,
                b: qp.b,
                cones: qp.cones,
                var_bounds: None,
                integrality: None,
            });
        }

        let n = self.n;
        let m_orig = qp.A.rows();

        // Build P matrix and compute Cholesky factorization P = L'L
        // Use dense Cholesky for simplicity
        let p_triplets: Vec<(usize, usize, f64)> = self
            .p_triplets
            .iter()
            .map(|&(i, j, v)| (i, j, v * self.obj_sense))
            .collect();

        let mut p_dense = vec![0.0; n * n];
        for &(i, j, v) in &p_triplets {
            p_dense[i * n + j] += v;
            if i != j {
                p_dense[j * n + i] += v;
            }
        }

        // Compute Cholesky L such that P = L'L using simple implementation
        // For P that's only positive semidefinite (some zero eigenvalues), we handle
        // small/negative pivots by skipping those rows (they don't contribute to ||Lx||)
        let mut l = vec![0.0; n * n];
        let mut valid_rows = Vec::new();
        for i in 0..n {
            for j in 0..=i {
                let mut sum = p_dense[i * n + j];
                for k in 0..j {
                    sum -= l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    if sum > 1e-12 {
                        l[i * n + j] = sum.sqrt();
                        valid_rows.push(i);
                    } else {
                        // Zero or negative pivot - this row of L will be zero
                        l[i * n + j] = 0.0;
                    }
                } else if l[j * n + j].abs() > 1e-14 {
                    l[i * n + j] = sum / l[j * n + j];
                } else {
                    l[i * n + j] = 0.0;
                }
            }
        }

        // Count non-zero rows in L (for SOC dimension)
        let mut nonzero_rows = Vec::new();
        for i in 0..n {
            let row_norm_sq: f64 = (0..=i).map(|j| l[i * n + j] * l[i * n + j]).sum();
            if row_norm_sq > 1e-12 {
                nonzero_rows.push(i);
            }
        }

        if nonzero_rows.is_empty() {
            // P is zero matrix - just an LP
            return Ok(ProblemData {
                P: None,
                q: qp.q,
                A: qp.A,
                b: qp.b,
                cones: qp.cones,
                var_bounds: None,
                integrality: None,
            });
        }

        // For QP objective (1/2) x'Px + q'x, we use rotated SOC:
        // min t + q'x s.t. (t, v, Lx) ∈ RSOC, meaning 2*t*v >= ||Lx||²
        //
        // We want t >= ||Lx||²/2, so we need 2tv >= ||Lx||² with t >= ||Lx||²/(2v).
        // Setting v = 1 gives t >= ||Lx||²/2 as required.
        //
        // Convert RSOC to SOC: (u, v, w) ∈ RSOC iff ((u+v)/√2, (u-v)/√2, w) ∈ SOC
        // With u = t, v = 1, we get: ((t + 1)/√2, (t - 1)/√2, Lx) ∈ SOC
        //
        // We embed the constant 1 directly in the b vector (no auxiliary variable):
        // Row for soc[0]: -t/√2 + slack[0] = 1/√2  =>  slack[0] = (t+1)/√2
        // Row for soc[1]: -t/√2 + slack[1] = -1/√2 =>  slack[1] = (t-1)/√2
        // Rows for soc[2+k]: -L[k,:]*x + slack[2+k] = 0

        let sqrt2 = std::f64::consts::SQRT_2;
        let soc_dim = 2 + nonzero_rows.len(); // (t+1)/√2, (t-1)/√2, Lx...

        // New problem has n+1 variables: [x (n), t (1)]
        let new_n = n + 1;
        let new_m = m_orig + soc_dim; // original + SOC

        // Build new A matrix
        let mut new_triplets = Vec::new();

        // Copy original constraints (only affecting x, columns 0..n)
        for (&val, (row, col)) in qp.A.iter() {
            new_triplets.push((row, col, val));
        }

        // SOC constraint: ((t+1)/√2, (t-1)/√2, Lx) in SOC
        let soc_start = m_orig;

        // slack[0] = (t+1)/√2:  -t/√2 + slack[0] = 1/√2
        new_triplets.push((soc_start, n, -1.0 / sqrt2)); // -t/√2

        // slack[1] = (t-1)/√2:  -t/√2 + slack[1] = -1/√2
        new_triplets.push((soc_start + 1, n, -1.0 / sqrt2)); // -t/√2

        // Remaining rows: -L[i,:]*x + slack[2+k] = 0
        for (idx, &row_i) in nonzero_rows.iter().enumerate() {
            for j in 0..=row_i {
                let val = l[row_i * n + j];
                if val.abs() > 1e-14 {
                    new_triplets.push((soc_start + 2 + idx, j, -val));
                }
            }
        }

        let new_a = sparse::from_triplets(new_m, new_n, new_triplets);

        // Build new b vector
        let mut new_b = qp.b.clone();
        // SOC rows: b = [1/√2, -1/√2, 0, 0, ...]
        new_b.push(1.0 / sqrt2);  // for (t+1)/√2
        new_b.push(-1.0 / sqrt2); // for (t-1)/√2
        new_b.extend(vec![0.0; nonzero_rows.len()]); // for Lx

        // Build new q vector: [original q, 1.0 for t]
        let mut new_q = qp.q.clone();
        new_q.push(1.0); // coefficient for t in objective

        // Build cone specification
        let mut new_cones = qp.cones.clone();
        new_cones.push(ConeSpec::Soc { dim: soc_dim });

        Ok(ProblemData {
            P: None, // No quadratic term - absorbed into rotated SOC
            q: new_q,
            A: new_a,
            b: new_b,
            cones: new_cones,
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
    let mut obj_sense = 1.0; // 1 = minimize, -1 = maximize

    for line_result in reader.lines() {
        let line_raw = line_result?;
        let line = line_raw.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('*') {
            continue;
        }

        // Check for section headers (use trimmed line)
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
                    obj_sense = -1.0;
                }
            }
            "ROWS" => {
                // MPS format: type at columns 2-3 (1-indexed), name at columns 5-12
                // Some files put type at position 1, others at position 2 (0-indexed)
                let line_bytes = line_raw.as_bytes();
                let line_len = line_bytes.len();

                // Find row type - look for N/E/L/G at position 1 or 2
                let rtype = if line_len >= 3 {
                    let c1 = line_bytes[1] as char;
                    let c2 = line_bytes[2] as char;
                    if matches!(c1, 'N' | 'E' | 'L' | 'G') {
                        c1
                    } else if matches!(c2, 'N' | 'E' | 'L' | 'G') {
                        c2
                    } else {
                        continue;
                    }
                } else if line_len >= 2 {
                    line_bytes[1] as char
                } else {
                    continue;
                };

                // Extract row name (columns 5-12, 0-indexed 4-11)
                let rname = if line_len >= 12 {
                    String::from_utf8_lossy(&line_bytes[4..12]).trim().to_string()
                } else if line_len > 4 {
                    String::from_utf8_lossy(&line_bytes[4..]).trim().to_string()
                } else {
                    continue;
                };

                if rname.is_empty() {
                    continue;
                }

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
            "COLUMNS" => {
                // MPS fixed-width format: fields at columns 5-12, 15-22, 25-36, 40-47, 50-61
                // (0-indexed: 4-11, 14-21, 24-35, 39-46, 49-60)
                let line_bytes = line_raw.as_bytes();
                let line_len = line_bytes.len();

                // Extract variable name (columns 5-12, 0-indexed 4-11)
                let var_name = if line_len >= 12 {
                    String::from_utf8_lossy(&line_bytes[4..12]).trim().to_string()
                } else if line_len > 4 {
                    String::from_utf8_lossy(&line_bytes[4..]).trim().to_string()
                } else {
                    continue;
                };

                if var_name.is_empty() {
                    continue;
                }

                // Get or create variable index
                let var_idx = *var_map.entry(var_name.clone()).or_insert_with(|| {
                    let idx = var_names.len();
                    var_names.push(var_name.clone());
                    idx
                });

                // Parse first (row_name, value) pair at columns 15-22, 25-36
                if line_len >= 25 {
                    let row_name = String::from_utf8_lossy(
                        &line_bytes[14..22.min(line_len)]
                    ).trim().to_string();
                    let value_str = if line_len >= 36 {
                        String::from_utf8_lossy(&line_bytes[24..36]).trim().to_string()
                    } else if line_len > 24 {
                        String::from_utf8_lossy(&line_bytes[24..]).trim().to_string()
                    } else {
                        String::new()
                    };

                    if !row_name.is_empty() && !value_str.is_empty() {
                        let value: f64 = value_str.parse().unwrap_or(0.0);

                        if Some(row_name.clone()) == obj_row {
                            q_coeffs.insert(var_name.clone(), value);
                        } else if let Some(&con_idx) = con_map.get(&row_name) {
                            a_triplets.push((con_idx, var_idx, value));
                        }
                    }
                }

                // Parse second (row_name, value) pair at columns 40-47, 50-61
                if line_len >= 50 {
                    let row_name = String::from_utf8_lossy(
                        &line_bytes[39..47.min(line_len)]
                    ).trim().to_string();
                    let value_str = if line_len >= 61 {
                        String::from_utf8_lossy(&line_bytes[49..61]).trim().to_string()
                    } else if line_len > 49 {
                        String::from_utf8_lossy(&line_bytes[49..]).trim().to_string()
                    } else {
                        String::new()
                    };

                    if !row_name.is_empty() && !value_str.is_empty() {
                        let value: f64 = value_str.parse().unwrap_or(0.0);

                        if Some(row_name.clone()) == obj_row {
                            q_coeffs.insert(var_name.clone(), value);
                        } else if let Some(&con_idx) = con_map.get(&row_name) {
                            a_triplets.push((con_idx, var_idx, value));
                        }
                    }
                }
            }
            "RHS" => {
                // MPS fixed-width format: RHS name at columns 5-12, then pairs at 15-22/25-36 and 40-47/50-61
                let line_bytes = line_raw.as_bytes();
                let line_len = line_bytes.len();

                // First pair: row name at columns 15-22, value at 25-36
                if line_len >= 25 {
                    let row_name = String::from_utf8_lossy(
                        &line_bytes[14..22.min(line_len)]
                    ).trim().to_string();
                    let value_str = if line_len >= 36 {
                        String::from_utf8_lossy(&line_bytes[24..36]).trim().to_string()
                    } else if line_len > 24 {
                        String::from_utf8_lossy(&line_bytes[24..]).trim().to_string()
                    } else {
                        String::new()
                    };

                    if !row_name.is_empty() && !value_str.is_empty() {
                        let value: f64 = value_str.parse().unwrap_or(0.0);
                        rhs.insert(row_name, value);
                    }
                }

                // Second pair: row name at columns 40-47, value at 50-61
                if line_len >= 50 {
                    let row_name = String::from_utf8_lossy(
                        &line_bytes[39..47.min(line_len)]
                    ).trim().to_string();
                    let value_str = if line_len >= 61 {
                        String::from_utf8_lossy(&line_bytes[49..61]).trim().to_string()
                    } else if line_len > 49 {
                        String::from_utf8_lossy(&line_bytes[49..]).trim().to_string()
                    } else {
                        String::new()
                    };

                    if !row_name.is_empty() && !value_str.is_empty() {
                        let value: f64 = value_str.parse().unwrap_or(0.0);
                        rhs.insert(row_name, value);
                    }
                }
            }
            "RANGES" => {
                // MPS fixed-width format: same as RHS
                let line_bytes = line_raw.as_bytes();
                let line_len = line_bytes.len();

                // First pair: row name at columns 15-22, value at 25-36
                if line_len >= 25 {
                    let row_name = String::from_utf8_lossy(
                        &line_bytes[14..22.min(line_len)]
                    ).trim().to_string();
                    let value_str = if line_len >= 36 {
                        String::from_utf8_lossy(&line_bytes[24..36]).trim().to_string()
                    } else if line_len > 24 {
                        String::from_utf8_lossy(&line_bytes[24..]).trim().to_string()
                    } else {
                        String::new()
                    };

                    if !row_name.is_empty() && !value_str.is_empty() {
                        let value: f64 = value_str.parse().unwrap_or(0.0);
                        ranges.insert(row_name, value.abs());
                    }
                }

                // Second pair: row name at columns 40-47, value at 50-61
                if line_len >= 50 {
                    let row_name = String::from_utf8_lossy(
                        &line_bytes[39..47.min(line_len)]
                    ).trim().to_string();
                    let value_str = if line_len >= 61 {
                        String::from_utf8_lossy(&line_bytes[49..61]).trim().to_string()
                    } else if line_len > 49 {
                        String::from_utf8_lossy(&line_bytes[49..]).trim().to_string()
                    } else {
                        String::new()
                    };

                    if !row_name.is_empty() && !value_str.is_empty() {
                        let value: f64 = value_str.parse().unwrap_or(0.0);
                        ranges.insert(row_name, value.abs());
                    }
                }
            }
            "BOUNDS" => {
                // MPS fixed-width format: type at columns 2-3, bound name at 5-12, var name at 15-22, value at 25-36
                let line_bytes = line_raw.as_bytes();
                let line_len = line_bytes.len();

                // Extract bound type (columns 2-3, 0-indexed 1-2)
                let btype = if line_len >= 3 {
                    String::from_utf8_lossy(&line_bytes[1..3]).trim().to_string()
                } else {
                    continue;
                };

                // Extract variable name (columns 15-22, 0-indexed 14-21)
                let var_name = if line_len >= 22 {
                    String::from_utf8_lossy(&line_bytes[14..22]).trim().to_string()
                } else if line_len > 14 {
                    String::from_utf8_lossy(&line_bytes[14..]).trim().to_string()
                } else {
                    continue;
                };

                if var_name.is_empty() {
                    continue;
                }

                // Extract value (columns 25-36, 0-indexed 24-35)
                let value: f64 = if line_len >= 25 {
                    let value_str = if line_len >= 36 {
                        String::from_utf8_lossy(&line_bytes[24..36]).trim().to_string()
                    } else {
                        String::from_utf8_lossy(&line_bytes[24..]).trim().to_string()
                    };
                    value_str.parse().unwrap_or(0.0)
                } else {
                    0.0
                };

                match btype.as_str() {
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
            "QUADOBJ" => {
                // MPS fixed-width format for QUADOBJ: var1 at 5-12, var2 at 15-22, value at 25-36
                let line_bytes = line_raw.as_bytes();
                let line_len = line_bytes.len();

                // Extract var1 (columns 5-12, 0-indexed 4-11)
                let var1 = if line_len >= 12 {
                    String::from_utf8_lossy(&line_bytes[4..12]).trim().to_string()
                } else if line_len > 4 {
                    String::from_utf8_lossy(&line_bytes[4..]).trim().to_string()
                } else {
                    continue;
                };

                // Extract var2 (columns 15-22, 0-indexed 14-21)
                let var2 = if line_len >= 22 {
                    String::from_utf8_lossy(&line_bytes[14..22]).trim().to_string()
                } else if line_len > 14 {
                    String::from_utf8_lossy(&line_bytes[14..]).trim().to_string()
                } else {
                    continue;
                };

                // Extract value (columns 25-36, 0-indexed 24-35)
                let value: f64 = if line_len >= 25 {
                    let value_str = if line_len >= 36 {
                        String::from_utf8_lossy(&line_bytes[24..36]).trim().to_string()
                    } else {
                        String::from_utf8_lossy(&line_bytes[24..]).trim().to_string()
                    };
                    value_str.parse().unwrap_or(0.0)
                } else {
                    continue;
                };

                if let (Some(&i), Some(&j)) = (var_map.get(&var1), var_map.get(&var2)) {
                    // Store upper triangle only
                    let (row, col) = if i <= j { (i, j) } else { (j, i) };
                    // QPS stores Q such that obj = 0.5 x'Qx, so we use value directly
                    p_triplets.push((row, col, value));
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
        obj_sense, // Parsed from OBJSENSE section (1.0 = min, -1.0 = max)
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
    use std::io::Write;
    use tempfile::NamedTempFile;

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

    fn write_temp_qps(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        file.write_all(content.as_bytes()).expect("Failed to write temp file");
        file.flush().expect("Failed to flush temp file");
        file
    }

    #[test]
    fn test_fixed_width_parsing_with_spaces_in_names() {
        // Test that variable names with internal spaces are parsed correctly
        // using fixed-width column format (MPS standard).
        // This was the root cause of the QFORPLAN bug where "DEDO3 11" was
        // incorrectly split into two tokens.
        let qps_content = r#"NAME          TESTPROB
ROWS
 N  COST
 E  ROW1
COLUMNS
    VAR 1     COST                 1.0   ROW1                 1.0
    VAR 2     COST                 2.0   ROW1                 1.0
RHS
    RHS1      ROW1                 5.0
BOUNDS
 UP BOUND1    VAR 1                10.0
 UP BOUND1    VAR 2                10.0
ENDATA
"#;
        let temp_file = write_temp_qps(qps_content);
        let qps = parse_qps(temp_file.path()).expect("Should parse QPS with spaced names");

        // Should have 2 variables, not more (if split_whitespace was used, "VAR" and "1"
        // would be separate, causing wrong variable count)
        assert_eq!(qps.n, 2, "Expected 2 variables");
        assert_eq!(qps.var_names.len(), 2);
        assert!(qps.var_names.contains(&"VAR 1".to_string()),
            "Variable name 'VAR 1' should be preserved with space");
        assert!(qps.var_names.contains(&"VAR 2".to_string()),
            "Variable name 'VAR 2' should be preserved with space");
    }

    #[test]
    fn test_fixed_width_row_type_position() {
        // Test that row type can be at position 1 or 2 (different files use different positions)
        // Position 1: " N  COST" (type at index 1)
        // Position 2: "  N COST" (type at index 2)
        let qps_pos1 = r#"NAME          TEST1
ROWS
 N  OBJ
 E  CON1
COLUMNS
    X1        OBJ                  1.0   CON1                 1.0
RHS
    RHS1      CON1                 1.0
ENDATA
"#;
        let qps_pos2 = r#"NAME          TEST2
ROWS
  N OBJ
  E CON1
COLUMNS
    X1        OBJ                  1.0   CON1                 1.0
RHS
    RHS1      CON1                 1.0
ENDATA
"#;

        let temp1 = write_temp_qps(qps_pos1);
        let temp2 = write_temp_qps(qps_pos2);
        let result1 = parse_qps(temp1.path()).expect("Position 1 format");
        let result2 = parse_qps(temp2.path()).expect("Position 2 format");

        assert_eq!(result1.n, 1, "Position 1: expected 1 variable");
        assert_eq!(result1.m, 1, "Position 1: expected 1 constraint");
        assert_eq!(result2.n, 1, "Position 2: expected 1 variable");
        assert_eq!(result2.m, 1, "Position 2: expected 1 constraint");
    }
}
