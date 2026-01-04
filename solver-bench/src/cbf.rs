//! CBF (Conic Benchmark Format) parser.
//!
//! Parses CBF files as specified by MOSEK.
//! Reference: https://docs.mosek.com/latest/capi/cbf-format.html

use anyhow::{anyhow, bail, Context, Result};
use solver_core::linalg::sparse;
use solver_core::{ConeSpec, ProblemData};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Cone type in CBF format (before conversion to solver format).
#[derive(Debug, Clone)]
pub enum CbfCone {
    Free { dim: usize },       // No constraint
    NonNeg { dim: usize },     // x >= 0
    NonPos { dim: usize },     // x <= 0 (will be converted)
    Zero { dim: usize },       // x = 0
    Soc { dim: usize },        // Second-order cone
    RotatedSoc { dim: usize }, // Rotated second-order cone: 2uv >= ||w||², u,v >= 0
}

impl CbfCone {
    pub fn dim(&self) -> usize {
        match self {
            CbfCone::Free { dim } => *dim,
            CbfCone::NonNeg { dim } => *dim,
            CbfCone::NonPos { dim } => *dim,
            CbfCone::Zero { dim } => *dim,
            CbfCone::Soc { dim } => *dim,
            CbfCone::RotatedSoc { dim } => *dim,
        }
    }

    /// Convert to solver ConeSpec, returning None for Free cones.
    /// Note: RotatedSoc is converted to standard SOC (transformation applied to A, b).
    pub fn to_cone_spec(&self) -> Option<ConeSpec> {
        match self {
            CbfCone::Free { .. } => None,
            CbfCone::NonNeg { dim } => Some(ConeSpec::NonNeg { dim: *dim }),
            CbfCone::NonPos { dim } => Some(ConeSpec::NonNeg { dim: *dim }), // Handle sign in A
            CbfCone::Zero { dim } => Some(ConeSpec::Zero { dim: *dim }),
            CbfCone::Soc { dim } => Some(ConeSpec::Soc { dim: *dim }),
            CbfCone::RotatedSoc { dim } => Some(ConeSpec::Soc { dim: *dim }), // Becomes SOC after transformation
        }
    }

    /// Check if this cone is a rotated SOC (requires transformation).
    pub fn is_rotated(&self) -> bool {
        matches!(self, CbfCone::RotatedSoc { .. })
    }
}

/// Parsed CBF problem (intermediate representation).
#[derive(Debug, Clone)]
pub struct CbfProblem {
    pub name: String,
    pub version: u32,
    pub obj_sense: ObjSense,
    pub n: usize,                             // number of scalar variables
    pub m: usize,                             // number of scalar constraints
    pub var_cones: Vec<CbfCone>,              // cones for variables (with Free support)
    pub con_cones: Vec<CbfCone>,              // cones for constraints
    pub c: Vec<f64>,                          // objective coefficients (sparse, stored dense)
    pub a_triplets: Vec<(usize, usize, f64)>, // (row, col, val) for A
    pub b: Vec<f64>,                          // RHS (sparse, stored dense)
    pub int_vars: Vec<usize>,                 // integer variable indices
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ObjSense {
    Min,
    Max,
}

impl CbfProblem {
    fn new() -> Self {
        Self {
            name: String::new(),
            version: 0,
            obj_sense: ObjSense::Min,
            n: 0,
            m: 0,
            var_cones: Vec::new(),
            con_cones: Vec::new(),
            c: Vec::new(),
            a_triplets: Vec::new(),
            b: Vec::new(),
            int_vars: Vec::new(),
        }
    }
}

/// Parse a CBF file.
pub fn parse_cbf<P: AsRef<Path>>(path: P) -> Result<CbfProblem> {
    let file = File::open(path.as_ref())
        .with_context(|| format!("Failed to open CBF file: {:?}", path.as_ref()))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines().enumerate().peekable();

    let mut prob = CbfProblem::new();
    prob.name = path
        .as_ref()
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Helper to get next non-empty, non-comment line
    let mut next_line = || -> Result<Option<(usize, String)>> {
        loop {
            match lines.next() {
                Some((line_num, Ok(line))) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() || trimmed.starts_with('#') {
                        continue;
                    }
                    return Ok(Some((line_num + 1, trimmed.to_string())));
                }
                Some((line_num, Err(e))) => {
                    bail!("Error reading line {}: {}", line_num + 1, e);
                }
                None => return Ok(None),
            }
        }
    };

    // Parse keywords
    while let Some((line_num, keyword)) = next_line()? {
        match keyword.as_str() {
            "VER" => {
                let (_, version_line) =
                    next_line()?.ok_or_else(|| anyhow!("Expected version number after VER"))?;
                prob.version = version_line
                    .parse()
                    .with_context(|| format!("Invalid version at line {}", line_num))?;
            }

            "OBJSENSE" => {
                let (_, sense_line) =
                    next_line()?.ok_or_else(|| anyhow!("Expected MIN/MAX after OBJSENSE"))?;
                prob.obj_sense = match sense_line.to_uppercase().as_str() {
                    "MIN" => ObjSense::Min,
                    "MAX" => ObjSense::Max,
                    _ => bail!("Invalid OBJSENSE: {}", sense_line),
                };
            }

            "VAR" => {
                // VAR header: num_vars num_cone_specs
                let (_, header) = next_line()?.ok_or_else(|| anyhow!("Expected VAR header"))?;
                let parts: Vec<&str> = header.split_whitespace().collect();
                if parts.len() != 2 {
                    bail!("VAR header should have 2 numbers: {}", header);
                }
                prob.n = parts[0].parse()?;
                let num_cone_specs: usize = parts[1].parse()?;

                // Read cone specifications
                for _ in 0..num_cone_specs {
                    let (_, cone_line) =
                        next_line()?.ok_or_else(|| anyhow!("Expected cone specification"))?;
                    let cone = parse_cone_spec(&cone_line)?;
                    prob.var_cones.push(cone);
                }

                // Initialize c vector
                prob.c = vec![0.0; prob.n];
            }

            "CON" => {
                // CON header: num_cons num_cone_specs
                let (_, header) = next_line()?.ok_or_else(|| anyhow!("Expected CON header"))?;
                let parts: Vec<&str> = header.split_whitespace().collect();
                if parts.len() != 2 {
                    bail!("CON header should have 2 numbers: {}", header);
                }
                prob.m = parts[0].parse()?;
                let num_cone_specs: usize = parts[1].parse()?;

                // Read cone specifications
                for _ in 0..num_cone_specs {
                    let (_, cone_line) =
                        next_line()?.ok_or_else(|| anyhow!("Expected cone specification"))?;
                    let cone = parse_cone_spec(&cone_line)?;
                    prob.con_cones.push(cone);
                }

                // Initialize b vector
                prob.b = vec![0.0; prob.m];
            }

            "INT" => {
                // INT header: num_int_vars
                let (_, header) = next_line()?.ok_or_else(|| anyhow!("Expected INT header"))?;
                let num_int: usize = header.parse()?;

                for _ in 0..num_int {
                    let (_, idx_line) =
                        next_line()?.ok_or_else(|| anyhow!("Expected integer variable index"))?;
                    let idx: usize = idx_line.parse()?;
                    prob.int_vars.push(idx);
                }
            }

            "OBJACOORD" => {
                // Objective scalar coefficients
                let (_, header) =
                    next_line()?.ok_or_else(|| anyhow!("Expected OBJACOORD count"))?;
                let count: usize = header.parse()?;

                for _ in 0..count {
                    let (_, coord_line) =
                        next_line()?.ok_or_else(|| anyhow!("Expected objective coefficient"))?;
                    let parts: Vec<&str> = coord_line.split_whitespace().collect();
                    if parts.len() != 2 {
                        bail!("OBJACOORD entry should have 2 values: {}", coord_line);
                    }
                    let var_idx: usize = parts[0].parse()?;
                    let coef: f64 = parts[1].parse()?;
                    if var_idx < prob.n {
                        prob.c[var_idx] = coef;
                    }
                }
            }

            "ACOORD" => {
                // Constraint matrix coefficients (sparse triplets)
                let (_, header) = next_line()?.ok_or_else(|| anyhow!("Expected ACOORD count"))?;
                let count: usize = header.parse()?;

                for _ in 0..count {
                    let (_, coord_line) =
                        next_line()?.ok_or_else(|| anyhow!("Expected ACOORD triplet"))?;
                    let parts: Vec<&str> = coord_line.split_whitespace().collect();
                    if parts.len() != 3 {
                        bail!("ACOORD entry should have 3 values: {}", coord_line);
                    }
                    let row: usize = parts[0].parse()?;
                    let col: usize = parts[1].parse()?;
                    let val: f64 = parts[2].parse()?;
                    prob.a_triplets.push((row, col, val));
                }
            }

            "BCOORD" => {
                // RHS constants (sparse)
                let (_, header) = next_line()?.ok_or_else(|| anyhow!("Expected BCOORD count"))?;
                let count: usize = header.parse()?;

                for _ in 0..count {
                    let (_, coord_line) =
                        next_line()?.ok_or_else(|| anyhow!("Expected BCOORD pair"))?;
                    let parts: Vec<&str> = coord_line.split_whitespace().collect();
                    if parts.len() != 2 {
                        bail!("BCOORD entry should have 2 values: {}", coord_line);
                    }
                    let row: usize = parts[0].parse()?;
                    let val: f64 = parts[1].parse()?;
                    if row < prob.m {
                        prob.b[row] = val;
                    }
                }
            }

            // Skip unsupported keywords (PSD, etc.)
            "PSDVAR" | "PSDCON" | "OBJFCOORD" | "FCOORD" | "HCOORD" | "DCOORD" | "OBJBCOORD"
            | "POWCONES" | "POW*CONES" => {
                // Read and skip the data
                let (_, header) =
                    next_line()?.ok_or_else(|| anyhow!("Expected header for {}", keyword))?;

                // Try to parse as count, skip that many lines
                if let Ok(count) = header.parse::<usize>() {
                    for _ in 0..count {
                        next_line()?;
                    }
                }
            }

            _ => {
                // Unknown keyword - skip
                eprintln!("Warning: Skipping unknown keyword: {}", keyword);
            }
        }
    }

    Ok(prob)
}

/// Parse a cone specification like "Q 3" or "L= 2".
fn parse_cone_spec(line: &str) -> Result<CbfCone> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() != 2 {
        bail!("Cone spec should have 2 parts: {}", line);
    }

    let cone_type = parts[0];
    let dim: usize = parts[1]
        .parse()
        .with_context(|| format!("Invalid cone dimension: {}", parts[1]))?;

    match cone_type {
        "F" => Ok(CbfCone::Free { dim }),
        "L+" => Ok(CbfCone::NonNeg { dim }),
        "L-" => Ok(CbfCone::NonPos { dim }),
        "L=" => Ok(CbfCone::Zero { dim }),
        "Q" => Ok(CbfCone::Soc { dim }),
        "QR" => {
            // Rotated quadratic cone: 2*x1*x2 >= ||x3..xn||^2, x1,x2 >= 0
            if dim < 3 {
                bail!("Rotated SOC must have dimension >= 3, got {}", dim);
            }
            Ok(CbfCone::RotatedSoc { dim })
        }
        "EXP" | "EXP*" => {
            bail!("Exponential cones not yet supported");
        }
        _ if cone_type.starts_with("@") || cone_type == "POW" || cone_type == "POW*" => {
            bail!("Power cones not yet supported");
        }
        _ => bail!("Unknown cone type: {}", cone_type),
    }
}

/// Convert CBF problem to solver ProblemData.
///
/// CBF format (MOSEK convention):
///   min c'x
///   s.t. Ax + b ∈ K_con   (affine expression in cone)
///        x ∈ K_var        (variables in cone)
///
/// Our format:
///   min q'x
///   s.t. Ax + s = b, s ∈ K
///
/// Conversion:
///   - For constraint cones: s = Ax + b_cbf, so A_ours = -A_cbf, b_ours = b_cbf
///   - For variable cones: add rows -I x + s_var = 0, s_var ∈ K_var
///   - For rotated SOC: transform (u,v,w) → ((u+v)/√2, (u-v)/√2, w) to standard SOC
pub fn to_problem_data(cbf: &CbfProblem) -> Result<ProblemData> {
    // Check for unsupported features
    if !cbf.int_vars.is_empty() {
        bail!("Integer variables not yet supported");
    }

    let n = cbf.n;
    let m_con = cbf.m;

    // Count non-free variable cone dimensions
    let var_cone_dim: usize = cbf
        .var_cones
        .iter()
        .filter(|c| !matches!(c, CbfCone::Free { .. }))
        .map(|c| c.dim())
        .sum();

    // Check if all variable cones are Free (unconstrained)
    let all_vars_free = cbf
        .var_cones
        .iter()
        .all(|c| matches!(c, CbfCone::Free { .. }));

    // Total constraint rows = original constraints + variable cone constraints
    let m_total = m_con + var_cone_dim;

    // Build constraint cone row ranges with type info for transformations
    // Each entry: (start_row, end_row, is_rotated, is_nonpos)
    let mut con_cone_ranges: Vec<(usize, usize, bool, bool)> = Vec::new();
    let mut row_offset = 0;
    for cone in &cbf.con_cones {
        let dim = cone.dim();
        let is_rotated = cone.is_rotated();
        let is_nonpos = matches!(cone, CbfCone::NonPos { .. });
        con_cone_ranges.push((row_offset, row_offset + dim, is_rotated, is_nonpos));
        row_offset += dim;
    }

    // Build A matrix triplets with transformations:
    //
    // CBF format: Ax + b ∈ K
    // Our format: A_our x + s = b_our, where s ∈ K
    //
    // For NonNeg (L+): Ax + b >= 0 => s = Ax + b, so A_our = -A, b_our = b
    // For NonPos (L-): Ax + b <= 0 => s = -(Ax + b) >= 0, so A_our = A, b_our = -b
    // For SOC/Zero: same as NonNeg: A_our = -A, b_our = b
    // For RotatedSOC: apply rotation R, then A_our = -R*A, b_our = R*b
    //
    let sqrt2_inv = 1.0 / std::f64::consts::SQRT_2;

    // Helper to check row properties
    let row_info = |row: usize| -> (bool, bool) {
        for &(start, end, is_rot, is_nonpos) in &con_cone_ranges {
            if row >= start && row < end {
                return (is_rot, is_nonpos);
            }
        }
        (false, false)
    };

    // First, collect triplets by row for rotated cone transformation
    let mut row_triplets: std::collections::HashMap<usize, Vec<(usize, f64)>> =
        std::collections::HashMap::new();
    for &(row, col, val) in &cbf.a_triplets {
        row_triplets.entry(row).or_default().push((col, val));
    }

    let mut a_triplets: Vec<(usize, usize, f64)> = Vec::new();

    for (row, col, val) in &cbf.a_triplets {
        let (is_rotated, is_nonpos) = row_info(*row);

        if is_rotated {
            // Rotated cones handled separately below
            continue;
        }

        if is_nonpos {
            // NonPos: A_our = A (NOT negated)
            a_triplets.push((*row, *col, *val));
        } else {
            // Standard (NonNeg, Zero, SOC): A_our = -A (negated)
            a_triplets.push((*row, *col, -val));
        }
    }

    // Handle rotated cone A matrix entries
    for &(start, end, is_rotated, _is_nonpos) in &con_cone_ranges {
        if !is_rotated {
            continue;
        }

        // For a rotated cone, transform rows:
        // new_row[0] = (old_row[0] + old_row[1]) / √2
        // new_row[1] = (old_row[0] - old_row[1]) / √2
        // new_row[2..] = old_row[2..] (unchanged)
        let row0 = start;
        let row1 = start + 1;

        // Get coefficients for rows 0 and 1
        let coeffs0: Vec<(usize, f64)> = row_triplets.get(&row0).cloned().unwrap_or_default();
        let coeffs1: Vec<(usize, f64)> = row_triplets.get(&row1).cloned().unwrap_or_default();

        // Build combined coefficient map
        let mut combined: std::collections::HashMap<usize, (f64, f64)> =
            std::collections::HashMap::new();
        for (col, val) in coeffs0 {
            combined.entry(col).or_insert((0.0, 0.0)).0 = val;
        }
        for (col, val) in coeffs1 {
            combined.entry(col).or_insert((0.0, 0.0)).1 = val;
        }

        // Emit transformed rows (with negation for our format)
        for (col, (v0, v1)) in &combined {
            // new_row0 = (v0 + v1) / √2, then negate
            let new_val0 = -(v0 + v1) * sqrt2_inv;
            if new_val0.abs() > 1e-15 {
                a_triplets.push((row0, *col, new_val0));
            }

            // new_row1 = (v0 - v1) / √2, then negate
            let new_val1 = -(v0 - v1) * sqrt2_inv;
            if new_val1.abs() > 1e-15 {
                a_triplets.push((row1, *col, new_val1));
            }
        }

        // Remaining rows of this cone (row2 onwards) - just negate
        for row in (start + 2)..end {
            if let Some(coeffs) = row_triplets.get(&row) {
                for &(col, val) in coeffs {
                    a_triplets.push((row, col, -val));
                }
            }
        }
    }

    // RHS transformations:
    // NonNeg/Zero/SOC: b_our = b
    // NonPos: b_our = -b
    // RotatedSOC: b_our = R * b (rotation applied)
    let mut b: Vec<f64> = cbf.b.clone();

    // Apply transformations to b
    for &(start, end, is_rotated, is_nonpos) in &con_cone_ranges {
        if is_rotated {
            // Apply rotation to b
            let b0 = b[start];
            let b1 = b[start + 1];
            b[start] = (b0 + b1) * sqrt2_inv;
            b[start + 1] = (b0 - b1) * sqrt2_inv;
        } else if is_nonpos {
            // Negate b for NonPos cones
            for i in start..end {
                b[i] = -b[i];
            }
        }
        // Other cones: b unchanged
    }

    // Cones for constraints - convert CbfCone to ConeSpec
    let mut cones: Vec<ConeSpec> = Vec::new();
    for cone in &cbf.con_cones {
        match cone {
            CbfCone::Free { .. } => bail!("Free cone not allowed in constraints"),
            _ => cones.push(cone.to_cone_spec().unwrap()),
        }
    }

    // Handle variable cones: add -I x + s_var = 0, s_var ∈ K_var
    // This enforces x ∈ K_var by setting s_var = x and requiring s_var ∈ K_var
    // For rotated SOC: add -R x + s = 0 where R is the rotation matrix
    if !all_vars_free {
        let mut var_offset = 0; // Which variable column we're at
        let mut var_cone_row = m_con; // Which row we're adding (starts after constraint rows)
        for cone in &cbf.var_cones {
            let dim = cone.dim();
            match cone {
                CbfCone::Free { .. } => {
                    // No constraint for free variables
                    var_offset += dim;
                }
                CbfCone::NonNeg { .. } | CbfCone::Zero { .. } | CbfCone::Soc { .. } => {
                    // Add -I block for this cone: -x + s = 0 => s = x
                    for i in 0..dim {
                        a_triplets.push((var_cone_row + i, var_offset + i, -1.0));
                    }
                    b.extend(vec![0.0; dim]);
                    if let Some(cs) = cone.to_cone_spec() {
                        cones.push(cs);
                    }
                    var_offset += dim;
                    var_cone_row += dim;
                }
                CbfCone::RotatedSoc { dim: d } => {
                    // x ∈ QR means Rx ∈ SOC where R is the rotation matrix
                    // Add -Rx + s = 0 => s = Rx
                    // R transforms: (u, v, w) -> ((u+v)/√2, (u-v)/√2, w)

                    // First two rows get rotation
                    // row 0: -(x₀ + x₁)/√2 => coeffs (-1/√2, -1/√2)
                    a_triplets.push((var_cone_row, var_offset, -sqrt2_inv));
                    a_triplets.push((var_cone_row, var_offset + 1, -sqrt2_inv));

                    // row 1: -(x₀ - x₁)/√2 => coeffs (-1/√2, +1/√2)
                    a_triplets.push((var_cone_row + 1, var_offset, -sqrt2_inv));
                    a_triplets.push((var_cone_row + 1, var_offset + 1, sqrt2_inv));

                    // Remaining rows: -xᵢ
                    for i in 2..*d {
                        a_triplets.push((var_cone_row + i, var_offset + i, -1.0));
                    }

                    b.extend(vec![0.0; *d]);
                    cones.push(ConeSpec::Soc { dim: *d }); // Becomes standard SOC
                    var_offset += d;
                    var_cone_row += d;
                }
                CbfCone::NonPos { dim: d } => {
                    // x <= 0 means -x >= 0, so add +I block: +x + s = 0 => s = -x
                    for i in 0..*d {
                        a_triplets.push((var_cone_row + i, var_offset + i, 1.0));
                    }
                    b.extend(vec![0.0; *d]);
                    cones.push(ConeSpec::NonNeg { dim: *d });
                    var_offset += d;
                    var_cone_row += d;
                }
            }
        }
    }

    let a = sparse::from_triplets(m_total, n, a_triplets);

    // Objective
    let q = if cbf.obj_sense == ObjSense::Max {
        // Negate for maximization
        cbf.c.iter().map(|&v| -v).collect()
    } else {
        cbf.c.clone()
    };

    Ok(ProblemData {
        P: None,
        q,
        A: a,
        b,
        cones,
        var_bounds: None,
        integrality: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cone_spec() {
        assert!(matches!(
            parse_cone_spec("F 3").unwrap(),
            CbfCone::Free { dim: 3 }
        ));
        assert!(matches!(
            parse_cone_spec("L= 3").unwrap(),
            CbfCone::Zero { dim: 3 }
        ));
        assert!(matches!(
            parse_cone_spec("L+ 5").unwrap(),
            CbfCone::NonNeg { dim: 5 }
        ));
        assert!(matches!(
            parse_cone_spec("L- 2").unwrap(),
            CbfCone::NonPos { dim: 2 }
        ));
        assert!(matches!(
            parse_cone_spec("Q 4").unwrap(),
            CbfCone::Soc { dim: 4 }
        ));
        assert!(matches!(
            parse_cone_spec("QR 3").unwrap(),
            CbfCone::RotatedSoc { dim: 3 }
        ));
        assert!(matches!(
            parse_cone_spec("QR 5").unwrap(),
            CbfCone::RotatedSoc { dim: 5 }
        ));
        // Rotated SOC requires dim >= 3
        assert!(parse_cone_spec("QR 2").is_err());
    }

    #[test]
    fn test_cbf_cone_to_spec() {
        assert!(CbfCone::Free { dim: 5 }.to_cone_spec().is_none());
        assert!(matches!(
            CbfCone::NonNeg { dim: 3 }.to_cone_spec(),
            Some(ConeSpec::NonNeg { dim: 3 })
        ));
        assert!(matches!(
            CbfCone::Soc { dim: 4 }.to_cone_spec(),
            Some(ConeSpec::Soc { dim: 4 })
        ));
        // Rotated SOC becomes standard SOC after transformation
        assert!(matches!(
            CbfCone::RotatedSoc { dim: 5 }.to_cone_spec(),
            Some(ConeSpec::Soc { dim: 5 })
        ));
    }

    #[test]
    fn test_rotated_soc_is_rotated() {
        assert!(!CbfCone::Free { dim: 3 }.is_rotated());
        assert!(!CbfCone::NonNeg { dim: 3 }.is_rotated());
        assert!(!CbfCone::Soc { dim: 3 }.is_rotated());
        assert!(CbfCone::RotatedSoc { dim: 3 }.is_rotated());
    }
}
