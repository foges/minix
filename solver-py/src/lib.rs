//! Python bindings for minix solver.
//!
//! This crate provides Python bindings via PyO3, exposing the minix conic
//! optimization solver to Python. It integrates with scipy sparse matrices
//! and numpy arrays.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use solver_core::{solve, ConeSpec, ProblemData, SolveResult, SolveStatus, SolverSettings};
use sprs::CsMat;

/// Convert scipy CSC arrays to sprs CsMat in CSC format.
///
/// scipy CSC format uses:
/// - indptr: column pointers (length ncols + 1)
/// - indices: row indices for each nonzero
/// - data: nonzero values
fn scipy_csc_to_sprs(
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
    shape: (usize, usize),
) -> CsMat<f64> {
    // Use new_csc for CSC format (scipy's default)
    CsMat::new_csc(shape, indptr, indices, data)
}

/// Parse cone specification from Python list of tuples.
///
/// Expected format: [("zero", 5), ("nonneg", 10), ("soc", 3), ...]
fn parse_cones(cones: Vec<(String, usize)>) -> PyResult<Vec<ConeSpec>> {
    let mut result = Vec::with_capacity(cones.len());

    for (cone_type, dim) in cones {
        let spec = match cone_type.to_lowercase().as_str() {
            "zero" | "z" | "eq" => ConeSpec::Zero { dim },
            "nonneg" | "nn" | "l" | "pos" => ConeSpec::NonNeg { dim },
            "soc" | "q" | "socp" => ConeSpec::Soc { dim },
            "psd" | "s" | "sdp" => {
                // For PSD, dim is the svec dimension = n(n+1)/2
                // We need to recover n
                // n(n+1)/2 = dim => n^2 + n - 2*dim = 0
                // n = (-1 + sqrt(1 + 8*dim)) / 2
                let discriminant = 1.0 + 8.0 * (dim as f64);
                let n = ((-1.0 + discriminant.sqrt()) / 2.0).round() as usize;
                ConeSpec::Psd { n }
            }
            "exp" | "ep" => ConeSpec::Exp { count: dim / 3 },
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown cone type: {}. Supported: zero, nonneg, soc, psd, exp",
                    cone_type
                )))
            }
        };
        result.push(spec);
    }

    Ok(result)
}

/// Result returned from the solve function.
#[pyclass]
#[derive(Clone)]
pub struct MinixResult {
    #[pyo3(get)]
    status: String,
    #[pyo3(get)]
    obj_val: f64,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    solve_time_ms: u64,
    #[pyo3(get)]
    primal_res: f64,
    #[pyo3(get)]
    dual_res: f64,
    #[pyo3(get)]
    gap: f64,

    // Store solution vectors internally
    x_vec: Vec<f64>,
    s_vec: Vec<f64>,
    z_vec: Vec<f64>,
}

#[pymethods]
impl MinixResult {
    /// Get primal solution vector x.
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice_bound(py, &self.x_vec)
    }

    /// Get slack variables s.
    fn s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice_bound(py, &self.s_vec)
    }

    /// Get dual variables z (y in some notations).
    fn z<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice_bound(py, &self.z_vec)
    }

    /// Alias for z (CVXPY uses y for dual variables).
    fn y<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.z(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "MinixResult(status='{}', obj_val={:.6e}, iters={}, time={:.1}ms)",
            self.status, self.obj_val, self.iterations, self.solve_time_ms
        )
    }
}

impl From<SolveResult> for MinixResult {
    fn from(result: SolveResult) -> Self {
        let status_str = match result.status {
            SolveStatus::Optimal => "optimal",
            SolveStatus::PrimalInfeasible => "primal_infeasible",
            SolveStatus::DualInfeasible => "dual_infeasible",
            SolveStatus::Unbounded => "unbounded",
            SolveStatus::MaxIters => "max_iterations",
            SolveStatus::TimeLimit => "time_limit",
            SolveStatus::NumericalError => "numerical_error",
        };

        MinixResult {
            status: status_str.to_string(),
            obj_val: result.obj_val,
            iterations: result.info.iters,
            solve_time_ms: result.info.solve_time_ms,
            primal_res: result.info.primal_res,
            dual_res: result.info.dual_res,
            gap: result.info.gap,
            x_vec: result.x,
            s_vec: result.s,
            z_vec: result.z,
        }
    }
}

/// Solve a conic optimization problem.
///
/// Problem form:
///     minimize    (1/2) x^T P x + q^T x
///     subject to  A x + s = b
///                 s ∈ K
///
/// where K is a Cartesian product of cones.
///
/// # Arguments
///
/// * `a_indptr` - CSC column pointers for constraint matrix A
/// * `a_indices` - CSC row indices for A
/// * `a_data` - CSC nonzero values for A
/// * `a_shape` - Shape of A as (rows, cols)
/// * `p_indptr` - CSC column pointers for quadratic cost P (optional)
/// * `p_indices` - CSC row indices for P (optional)
/// * `p_data` - CSC nonzero values for P (optional)
/// * `q` - Linear cost vector
/// * `b` - Constraint RHS vector
/// * `cones` - List of (cone_type, dimension) tuples
/// * `max_iter` - Maximum IPM iterations (default: 200)
/// * `verbose` - Print solver progress (default: false)
/// * `tol_feas` - Feasibility tolerance (default: 1e-8)
/// * `tol_gap` - Duality gap tolerance (default: 1e-8)
/// * `time_limit_ms` - Time limit in milliseconds (optional)
///
/// # Returns
///
/// MinixResult with solution status, primal/dual solutions, and diagnostics.
#[pyfunction]
#[pyo3(signature = (
    a_indptr,
    a_indices,
    a_data,
    a_shape,
    q,
    b,
    cones,
    p_indptr = None,
    p_indices = None,
    p_data = None,
    max_iter = None,
    verbose = None,
    tol_feas = None,
    tol_gap = None,
    time_limit_ms = None
))]
#[allow(clippy::too_many_arguments)]
fn solve_conic(
    _py: Python<'_>,
    // CSC matrix A (required)
    a_indptr: PyReadonlyArray1<i64>,
    a_indices: PyReadonlyArray1<i64>,
    a_data: PyReadonlyArray1<f64>,
    a_shape: (usize, usize),
    // Vectors
    q: PyReadonlyArray1<f64>,
    b: PyReadonlyArray1<f64>,
    // Cones
    cones: Vec<(String, usize)>,
    // CSC matrix P (optional, for QP)
    p_indptr: Option<PyReadonlyArray1<i64>>,
    p_indices: Option<PyReadonlyArray1<i64>>,
    p_data: Option<PyReadonlyArray1<f64>>,
    // Settings
    max_iter: Option<usize>,
    verbose: Option<bool>,
    tol_feas: Option<f64>,
    tol_gap: Option<f64>,
    time_limit_ms: Option<u64>,
) -> PyResult<MinixResult> {
    // Extract all data from Python arrays first (while we hold the GIL)
    let a_indptr_vec: Vec<usize> = a_indptr
        .as_slice()?
        .iter()
        .map(|&x| x as usize)
        .collect();
    let a_indices_vec: Vec<usize> = a_indices
        .as_slice()?
        .iter()
        .map(|&x| x as usize)
        .collect();
    let a_data_vec: Vec<f64> = a_data.as_slice()?.to_vec();
    let q_vec: Vec<f64> = q.as_slice()?.to_vec();
    let b_vec: Vec<f64> = b.as_slice()?.to_vec();

    // Extract P if provided
    let p_data_extracted = match (&p_indptr, &p_indices, &p_data) {
        (Some(indptr), Some(indices), Some(data)) => {
            let indptr_vec: Vec<usize> = indptr
                .as_slice()?
                .iter()
                .map(|&x| x as usize)
                .collect();
            let indices_vec: Vec<usize> = indices
                .as_slice()?
                .iter()
                .map(|&x| x as usize)
                .collect();
            let data_vec: Vec<f64> = data.as_slice()?.to_vec();
            Some((indptr_vec, indices_vec, data_vec))
        }
        _ => None,
    };

    // Parse cone specifications
    let cone_specs = parse_cones(cones)?;

    // Convert constraint matrix A
    let a_mat = scipy_csc_to_sprs(a_indptr_vec, a_indices_vec, a_data_vec, a_shape);

    // Convert quadratic cost P if provided
    let n = q_vec.len();
    let p_mat = p_data_extracted.map(|(indptr, indices, data)| {
        scipy_csc_to_sprs(indptr, indices, data, (n, n))
    });

    // Build problem
    let problem = ProblemData {
        P: p_mat,
        q: q_vec,
        A: a_mat,
        b: b_vec,
        cones: cone_specs,
        var_bounds: None,
        integrality: None,
    };

    // Build settings
    let mut settings = SolverSettings::default();
    if let Some(v) = max_iter {
        settings.max_iter = v;
    }
    if let Some(v) = verbose {
        settings.verbose = v;
    }
    if let Some(v) = tol_feas {
        settings.tol_feas = v;
    }
    if let Some(v) = tol_gap {
        settings.tol_gap = v;
    }
    if let Some(v) = time_limit_ms {
        settings.time_limit_ms = Some(v);
    }

    // Solve
    let result = solve(&problem, &settings).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Solver error: {}", e))
    })?;

    Ok(MinixResult::from(result))
}

/// Get version information.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Get default solver settings as a dict.
#[pyfunction]
fn default_settings(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let settings = SolverSettings::default();
    let dict = PyDict::new_bound(py);
    dict.set_item("max_iter", settings.max_iter)?;
    dict.set_item("verbose", settings.verbose)?;
    dict.set_item("tol_feas", settings.tol_feas)?;
    dict.set_item("tol_gap", settings.tol_gap)?;
    dict.set_item("tol_infeas", settings.tol_infeas)?;
    Ok(dict)
}

/// Python module definition.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_conic, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(default_settings, m)?)?;
    m.add_class::<MinixResult>()?;
    Ok(())
}
