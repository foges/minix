//! SDPLIB benchmark support - SDPA-sparse format parser and runner
//!
//! SDPA sparse format (.dat-s):
//! - Lines starting with " or * are comments
//! - mDim: number of constraint matrices (= primal constraints)
//! - nBlock: number of blocks
//! - blockStruct: sizes of each block (negative = diagonal block)
//! - c: objective vector (mDim values)
//! - Matrix entries: matNo blockNo i j value
//!
//! The SDP problem in SDPA format:
//! (P) max  tr(F0 * X)
//!     s.t. tr(Fi * X) = ci  for i = 1, ..., m
//!          X ⪰ 0
//!
//! (D) min  c'y
//!     s.t. F0 + sum_i yi*Fi ⪰ 0

use solver_core::{ProblemData, ConeSpec, SolverSettings, solve, SolveStatus};
use solver_core::linalg::sparse;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;

/// Parsed SDPA problem
#[derive(Debug, Clone)]
pub struct SdpaData {
    pub name: String,
    pub m_dim: usize,           // Number of constraints
    pub n_block: usize,         // Number of blocks
    pub block_struct: Vec<i32>, // Block sizes (negative = diagonal)
    pub c: Vec<f64>,            // Objective coefficients
    pub matrices: Vec<SdpaMatrix>, // F0, F1, ..., Fm
}

/// Sparse matrix in SDPA format
#[derive(Debug, Clone, Default)]
pub struct SdpaMatrix {
    pub entries: Vec<SdpaEntry>,
}

/// Single entry in SDPA matrix
#[derive(Debug, Clone)]
pub struct SdpaEntry {
    pub block: usize,  // 1-indexed block number
    pub i: usize,      // 1-indexed row
    pub j: usize,      // 1-indexed column
    pub value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdpaForm {
    Primal,
    Dual,
}

fn sdpa_form_from_env() -> SdpaForm {
    match env::var("MINIX_SDPA_FORM")
        .ok()
        .as_deref()
        .map(|v| v.to_ascii_lowercase())
        .as_deref()
    {
        Some("primal") | Some("p") | Some("0") => SdpaForm::Primal,
        Some("dual") | Some("d") | Some("1") => SdpaForm::Dual,
        _ => SdpaForm::Primal,
    }
}

/// Parse SDPA sparse format file
pub fn parse_sdpa_sparse(content: &str, name: &str) -> Result<SdpaData, String> {
    let mut lines = content.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('"') && !l.starts_with('*'));

    // Parse mDim
    let m_dim: usize = lines.next()
        .ok_or("Missing mDim")?
        .split_whitespace()
        .next()
        .ok_or("Empty mDim line")?
        .parse()
        .map_err(|e| format!("Invalid mDim: {}", e))?;

    // Parse nBlock
    let n_block: usize = lines.next()
        .ok_or("Missing nBlock")?
        .split_whitespace()
        .next()
        .ok_or("Empty nBlock line")?
        .parse()
        .map_err(|e| format!("Invalid nBlock: {}", e))?;

    // Parse blockStruct
    let block_line = lines.next().ok_or("Missing blockStruct")?;
    let block_struct: Vec<i32> = block_line
        .replace(['(', ')', '{', '}', ','], " ")
        .split_whitespace()
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<i32>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Invalid blockStruct: {}", e))?;

    if block_struct.len() != n_block {
        return Err(format!(
            "blockStruct length {} != nBlock {}",
            block_struct.len(), n_block
        ));
    }

    // Parse c vector
    let c_line = lines.next().ok_or("Missing c vector")?;
    let c: Vec<f64> = c_line
        .replace(['(', ')', '{', '}', ','], " ")
        .split_whitespace()
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Invalid c vector: {}", e))?;

    if c.len() != m_dim {
        return Err(format!("c length {} != mDim {}", c.len(), m_dim));
    }

    // Parse matrix entries
    let mut matrices: Vec<SdpaMatrix> = (0..=m_dim).map(|_| SdpaMatrix::default()).collect();

    for line in lines {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 5 {
            continue; // Skip malformed lines
        }

        let mat_no: usize = parts[0].parse()
            .map_err(|e| format!("Invalid matNo: {}", e))?;
        let block: usize = parts[1].parse()
            .map_err(|e| format!("Invalid blockNo: {}", e))?;
        let i: usize = parts[2].parse()
            .map_err(|e| format!("Invalid i: {}", e))?;
        let j: usize = parts[3].parse()
            .map_err(|e| format!("Invalid j: {}", e))?;
        let value: f64 = parts[4].parse()
            .map_err(|e| format!("Invalid value: {}", e))?;

        if mat_no > m_dim {
            return Err(format!("matNo {} > mDim {}", mat_no, m_dim));
        }

        matrices[mat_no].entries.push(SdpaEntry { block, i, j, value });
    }

    Ok(SdpaData {
        name: name.to_string(),
        m_dim,
        n_block,
        block_struct,
        c,
        matrices,
    })
}

fn sdpa_block_offsets(sdpa: &SdpaData) -> (Vec<usize>, usize) {
    let mut block_offsets = Vec::with_capacity(sdpa.n_block + 1);
    let mut total_dim = 0usize;
    block_offsets.push(0);

    for &size in &sdpa.block_struct {
        let n = size.unsigned_abs() as usize;
        let dim = if size < 0 {
            n
        } else {
            n * (n + 1) / 2
        };
        total_dim += dim;
        block_offsets.push(total_dim);
    }

    (block_offsets, total_dim)
}

/// Convert SDPA problem to our standard conic form
///
/// SDPA primal: max tr(F0 * X) s.t. tr(Fi * X) = ci, X >= 0
/// Our form:    min q'x s.t. Ax + s = b, s in K
///
/// Conversion:
/// - x = svec(X) (decision variables)
/// - q = -svec(F0) (negative because we minimize)
/// - A_i = svec(Fi) as row i
/// - b = c
/// - K = Zero(m) x PSD(n) where Zero handles equalities
pub fn sdpa_to_conic(sdpa: &SdpaData) -> Result<ProblemData, String> {
    let (block_offsets, total_dim) = sdpa_block_offsets(sdpa);

    // Build objective q = -svec(F0)
    let mut q = vec![0.0; total_dim];
    for entry in &sdpa.matrices[0].entries {
        let offset = block_offsets[entry.block - 1];
        let block_size = sdpa.block_struct[entry.block - 1];
        let idx = svec_index(entry.i, entry.j, block_size, offset);
        let scale = svec_scale(entry.i, entry.j, block_size);
        q[idx] -= entry.value * scale; // Negative for minimization
    }

    // Build constraint matrix A
    // Each row i corresponds to constraint tr(Fi * X) = ci
    // A[i, :] = svec(Fi)
    let mut triplets = Vec::new();

    for (mat_idx, matrix) in sdpa.matrices.iter().enumerate().skip(1) {
        let row = mat_idx - 1; // F1 -> row 0, F2 -> row 1, etc.
        for entry in &matrix.entries {
            let offset = block_offsets[entry.block - 1];
            let block_size = sdpa.block_struct[entry.block - 1];
            let col = svec_index(entry.i, entry.j, block_size, offset);
            let scale = svec_scale(entry.i, entry.j, block_size);
            triplets.push((row, col, entry.value * scale));
        }
    }

    // Add PSD cone constraint: -x + s_psd = 0
    // This embeds x into the PSD cone
    for i in 0..total_dim {
        triplets.push((sdpa.m_dim + i, i, -1.0));
    }

    let a = sparse::from_triplets(sdpa.m_dim + total_dim, total_dim, triplets);

    // RHS: b = [c; 0]
    let mut b = sdpa.c.clone();
    b.extend(vec![0.0; total_dim]);

    // Cones: Zero for equalities, then PSD/diagonal blocks
    let mut cones = vec![ConeSpec::Zero { dim: sdpa.m_dim }];

    // Add PSD/diagonal cones for each block
    for &size in &sdpa.block_struct {
        if size < 0 {
            // Diagonal block -> NonNeg cone
            let n = size.unsigned_abs() as usize;
            cones.push(ConeSpec::NonNeg { dim: n });
        } else {
            // Dense PSD block
            let n = size as usize;
            cones.push(ConeSpec::Psd { n });
        }
    }

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

/// Convert SDPA problem to the dual conic form.
///
/// SDPA dual: min c'y s.t. F0 - sum_i y_i Fi >= 0
/// Our form:  min q'y s.t. A y + s = b, s in K
///
/// Conversion:
/// - y = dual variables (length m)
/// - q = c
/// - b = svec(F0)
/// - A[:, i] = -svec(Fi) for i=1..m
/// - K = PSD blocks (no Zero cone)
pub fn sdpa_to_conic_dual(sdpa: &SdpaData) -> Result<ProblemData, String> {
    let (block_offsets, total_dim) = sdpa_block_offsets(sdpa);

    let q = sdpa.c.clone();

    let mut b = vec![0.0; total_dim];
    for entry in &sdpa.matrices[0].entries {
        let offset = block_offsets[entry.block - 1];
        let block_size = sdpa.block_struct[entry.block - 1];
        let idx = svec_index(entry.i, entry.j, block_size, offset);
        let scale = svec_scale(entry.i, entry.j, block_size);
        b[idx] += entry.value * scale;
    }

    let mut triplets = Vec::new();
    for (mat_idx, matrix) in sdpa.matrices.iter().enumerate().skip(1) {
        let col = mat_idx - 1;
        for entry in &matrix.entries {
            let offset = block_offsets[entry.block - 1];
            let block_size = sdpa.block_struct[entry.block - 1];
            let row = svec_index(entry.i, entry.j, block_size, offset);
            let scale = svec_scale(entry.i, entry.j, block_size);
            // Negate: A[:, i] = -svec(Fi) for the dual form
            triplets.push((row, col, -entry.value * scale));
        }
    }
    let a = sparse::from_triplets(total_dim, sdpa.m_dim, triplets);

    let mut cones = Vec::new();
    for &size in &sdpa.block_struct {
        if size < 0 {
            let n = size.unsigned_abs() as usize;
            cones.push(ConeSpec::NonNeg { dim: n });
        } else {
            let n = size as usize;
            cones.push(ConeSpec::Psd { n });
        }
    }

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

pub fn sdpa_to_conic_selected(sdpa: &SdpaData) -> Result<(ProblemData, SdpaForm), String> {
    let form = sdpa_form_from_env();
    let prob = match form {
        SdpaForm::Primal => sdpa_to_conic(sdpa)?,
        SdpaForm::Dual => sdpa_to_conic_dual(sdpa)?,
    };
    Ok((prob, form))
}

/// Compute svec index for (i, j) entry in a block
fn svec_index(i: usize, j: usize, block_size: i32, offset: usize) -> usize {
    let (i, j) = if i <= j { (i, j) } else { (j, i) };
    let i = i - 1; // Convert to 0-indexed
    let j = j - 1;

    if block_size < 0 {
        // Diagonal block: just use i (should have i == j)
        offset + i
    } else {
        // Dense block: upper triangular column-major
        // Index = j*(j+1)/2 + i for i <= j
        offset + j * (j + 1) / 2 + i
    }
}

/// Compute svec scaling factor for (i, j) entry
fn svec_scale(i: usize, j: usize, block_size: i32) -> f64 {
    if block_size < 0 {
        // Diagonal block: no scaling
        1.0
    } else if i == j {
        // Diagonal: no scaling
        1.0
    } else {
        // Off-diagonal: sqrt(2) scaling for svec
        std::f64::consts::SQRT_2
    }
}

/// Load SDPA problem from file
pub fn load_sdpa_file(path: &Path) -> Result<SdpaData, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read file: {}", e))?;

    let name = path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    parse_sdpa_sparse(&content, &name)
}

/// Solve an SDPA problem and return results
pub fn solve_sdpa(sdpa: &SdpaData, settings: &SolverSettings) -> Result<SdpaResult, String> {
    let (prob, form) = sdpa_to_conic_selected(sdpa)?;
    let result = solve(&prob, settings)
        .map_err(|e| format!("Solve failed: {}", e))?;

    let sdpa_obj = match form {
        SdpaForm::Primal => {
            // Our solver minimizes q'x. SDPA maximizes tr(F0 * X).
            // Since q = -svec(F0), our obj_val = -tr(F0 * X)
            -result.obj_val
        }
        SdpaForm::Dual => {
            // Dual form minimizes c^T y, which matches the primal optimum.
            result.obj_val
        }
    };

    Ok(SdpaResult {
        name: sdpa.name.clone(),
        status: result.status,
        primal_obj: sdpa_obj,
        dual_obj: sdpa_obj, // At optimality, primal = dual
        iterations: result.info.iters,
        solve_time_ms: result.info.solve_time_ms as f64,
    })
}

/// Result of solving an SDPA problem
#[derive(Debug, Clone)]
pub struct SdpaResult {
    pub name: String,
    pub status: SolveStatus,
    pub primal_obj: f64,
    pub dual_obj: f64,
    pub iterations: usize,
    pub solve_time_ms: f64,
}

/// Known SDPLIB problems with their expected optimal values
pub fn sdplib_reference_values() -> HashMap<&'static str, f64> {
    let mut map = HashMap::new();

    // Control theory problems
    map.insert("control1", 1.778463e+01);
    map.insert("control2", 8.300000e+00);
    map.insert("control3", 1.363327e+01);
    map.insert("control4", 1.979423e+01);
    map.insert("control5", 3.872609e+01);
    map.insert("control6", 3.693280e+01);
    map.insert("control7", 2.062507e+01);
    map.insert("control8", 2.028857e+01);
    map.insert("control9", 1.465544e+01);
    map.insert("control10", 3.812901e+01);
    map.insert("control11", 3.197364e+01);

    // Truss topology design
    map.insert("truss1", -8.999996e+00);
    map.insert("truss2", -1.233804e+02);
    map.insert("truss3", -9.109996e+00);
    map.insert("truss4", -9.009996e+00);
    map.insert("truss5", -1.323968e+02);
    map.insert("truss6", -9.009996e+00);
    map.insert("truss7", -9.009996e+00);
    map.insert("truss8", -1.331145e+02);

    // Graph theta problems
    map.insert("theta1", 2.300000e+01);
    map.insert("theta2", 3.287917e+01);
    map.insert("theta3", 4.216739e+01);
    map.insert("theta4", 5.032116e+01);
    map.insert("theta5", 5.723231e+01);
    map.insert("theta6", 6.348335e+01);

    // Max-cut problems
    map.insert("maxG11", -6.291546e+02);
    map.insert("maxG32", -1.567645e+03);
    map.insert("maxG51", -4.003809e+03);

    // Quadratic assignment
    map.insert("qap5", 4.360000e+02);
    map.insert("qap6", 3.810000e+02);
    map.insert("qap7", 4.240000e+02);
    map.insert("qap8", 7.560000e+02);
    map.insert("qap9", 1.410000e+03);
    map.insert("qap10", 1.094000e+03);

    // Minimum bisection
    map.insert("mcp100", 2.261574e+01);
    map.insert("mcp124-1", 1.419981e+01);
    map.insert("mcp124-2", 2.700017e+01);
    map.insert("mcp124-3", 4.677320e+01);
    map.insert("mcp124-4", 1.642999e+02);
    map.insert("mcp250-1", 3.172659e+01);
    map.insert("mcp250-2", 5.347332e+01);
    map.insert("mcp250-3", 9.681225e+01);
    map.insert("mcp250-4", 3.519997e+02);
    map.insert("mcp500-1", 6.214022e+01);
    map.insert("mcp500-2", 1.073149e+02);
    map.insert("mcp500-3", 1.962579e+02);
    map.insert("mcp500-4", 7.179996e+02);

    // Goemans-Williamson
    map.insert("gpp100", -4.494637e+01);
    map.insert("gpp124-1", -7.143998e+00);
    map.insert("gpp124-2", -4.068617e+01);
    map.insert("gpp124-3", -1.536847e+02);
    map.insert("gpp124-4", -4.199997e+02);
    map.insert("gpp250-1", -1.544997e+01);
    map.insert("gpp250-2", -8.183848e+01);
    map.insert("gpp250-3", -3.046476e+02);
    map.insert("gpp250-4", -8.509993e+02);
    map.insert("gpp500-1", -2.530000e+01);
    map.insert("gpp500-2", -1.562997e+02);
    map.insert("gpp500-3", -6.285432e+02);
    map.insert("gpp500-4", -1.737999e+03);

    // Equalizer design
    map.insert("equalG11", 6.291546e+02);
    map.insert("equalG32", 1.567645e+03);
    map.insert("equalG51", 4.003809e+03);

    // Random problems
    map.insert("hinf1", 2.032749e+00);
    map.insert("hinf2", 1.093083e+01);
    map.insert("hinf3", 5.695699e+01);
    map.insert("hinf4", 2.741482e+02);
    map.insert("hinf5", 3.627673e+02);
    map.insert("hinf6", 4.490119e+02);
    map.insert("hinf7", 3.905606e+02);
    map.insert("hinf8", 1.169094e+02);
    map.insert("hinf9", 2.364919e+02);
    map.insert("hinf10", 1.089917e+02);
    map.insert("hinf11", 6.590448e+01);
    map.insert("hinf12", 2.000000e-01);
    map.insert("hinf13", 4.565385e+01);
    map.insert("hinf14", 1.299996e+01);
    map.insert("hinf15", 2.440000e+01);

    // Sensor network localization
    map.insert("arch0", 5.66517e-01);
    map.insert("arch2", 6.71515e-01);
    map.insert("arch4", 9.32645e-01);
    map.insert("arch8", 7.05698e+00);

    // Toeplitz
    map.insert("taha1a", 9.95583e+06);
    map.insert("taha1b", 1.41211e+07);

    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_sdpa() {
        // Simple 2x2 SDP example
        let content = r#"
* Example problem
2
1
2
1.0 2.0
0 1 1 1 1.0
0 1 2 2 1.0
1 1 1 1 1.0
1 1 1 2 0.5
1 1 2 2 0.0
2 1 1 1 0.0
2 1 1 2 0.5
2 1 2 2 1.0
"#;
        let sdpa = parse_sdpa_sparse(content, "test").unwrap();
        assert_eq!(sdpa.m_dim, 2);
        assert_eq!(sdpa.n_block, 1);
        assert_eq!(sdpa.block_struct, vec![2]);
        assert_eq!(sdpa.c, vec![1.0, 2.0]);
        assert_eq!(sdpa.matrices[0].entries.len(), 2); // F0
        assert_eq!(sdpa.matrices[1].entries.len(), 3); // F1
        assert_eq!(sdpa.matrices[2].entries.len(), 3); // F2
    }

    #[test]
    fn test_sdpa_to_conic_simple() {
        // Simple trace minimization in SDPA form
        // max tr(I * X) s.t. tr(I * X) = 1, X >= 0
        // This is: max 1 s.t. X_11 + X_22 = 1
        let content = r#"
1
1
2
1.0
0 1 1 1 1.0
0 1 2 2 1.0
1 1 1 1 1.0
1 1 2 2 1.0
"#;
        let sdpa = parse_sdpa_sparse(content, "trace").unwrap();
        let prob = sdpa_to_conic(&sdpa).unwrap();

        // Should have 3 variables (2x2 svec: X11, X12, X22)
        assert_eq!(prob.q.len(), 3);

        // Solve it
        let mut settings = SolverSettings::default();
        settings.max_iter = 100;
        let result = solve(&prob, &settings).unwrap();

        println!("Status: {:?}", result.status);
        println!("Obj: {}", result.obj_val);

        // Optimal is X = 0.5*I, trace = 1, obj = -1 (we minimize -trace)
        assert!(matches!(result.status, SolveStatus::Optimal | SolveStatus::AlmostOptimal));
    }

    #[test]
    fn test_svec_index() {
        // Test svec indexing for 3x3 block
        // Upper triangular column-major: (1,1), (1,2), (2,2), (1,3), (2,3), (3,3)
        // Indices:                         0,     1,     2,     3,     4,     5
        assert_eq!(svec_index(1, 1, 3, 0), 0);
        assert_eq!(svec_index(1, 2, 3, 0), 1);
        assert_eq!(svec_index(2, 2, 3, 0), 2);
        assert_eq!(svec_index(1, 3, 3, 0), 3);
        assert_eq!(svec_index(2, 3, 3, 0), 4);
        assert_eq!(svec_index(3, 3, 3, 0), 5);

        // Test with offset
        assert_eq!(svec_index(1, 1, 3, 10), 10);
        assert_eq!(svec_index(2, 3, 3, 10), 14);
    }
}
