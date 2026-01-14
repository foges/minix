//! Python bindings for minix solver.
//!
//! This crate provides Python bindings via PyO3, exposing the minix conic
//! optimization solver to Python. It integrates with scipy sparse matrices
//! and numpy arrays.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use solver_core::{
    ipm2, solve, ConeSpec, ProblemData, SolveResult, SolveStatus, SolverSettings, WarmStart,
};
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

fn build_problem(
    a_indptr: PyReadonlyArray1<i64>,
    a_indices: PyReadonlyArray1<i64>,
    a_data: PyReadonlyArray1<f64>,
    a_shape: (usize, usize),
    q: PyReadonlyArray1<f64>,
    b: PyReadonlyArray1<f64>,
    cones: Vec<(String, usize)>,
    p_indptr: Option<PyReadonlyArray1<i64>>,
    p_indices: Option<PyReadonlyArray1<i64>>,
    p_data: Option<PyReadonlyArray1<f64>>,
) -> PyResult<ProblemData> {
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

    Ok(ProblemData {
        P: p_mat,
        q: q_vec,
        A: a_mat,
        b: b_vec,
        cones: cone_specs,
        var_bounds: None,
        integrality: None,
    })
}

fn build_warm_start(
    warm_x: Option<PyReadonlyArray1<f64>>,
    warm_s: Option<PyReadonlyArray1<f64>>,
    warm_z: Option<PyReadonlyArray1<f64>>,
    warm_tau: Option<f64>,
    warm_kappa: Option<f64>,
) -> PyResult<Option<WarmStart>> {
    let warm_x_vec = match warm_x {
        Some(arr) => Some(arr.as_slice()?.to_vec()),
        None => None,
    };
    let warm_s_vec = match warm_s {
        Some(arr) => Some(arr.as_slice()?.to_vec()),
        None => None,
    };
    let warm_z_vec = match warm_z {
        Some(arr) => Some(arr.as_slice()?.to_vec()),
        None => None,
    };

    if warm_x_vec.is_some()
        || warm_s_vec.is_some()
        || warm_z_vec.is_some()
        || warm_tau.is_some()
        || warm_kappa.is_some()
    {
        Ok(Some(WarmStart {
            x: warm_x_vec,
            s: warm_s_vec,
            z: warm_z_vec,
            tau: warm_tau,
            kappa: warm_kappa,
        }))
    } else {
        Ok(None)
    }
}

#[allow(clippy::too_many_arguments)]
fn build_settings(
    max_iter: Option<usize>,
    verbose: Option<bool>,
    tol_feas: Option<f64>,
    tol_gap: Option<f64>,
    kkt_refine_iters: Option<usize>,
    mcc_iters: Option<usize>,
    centrality_beta: Option<f64>,
    centrality_gamma: Option<f64>,
    line_search_max_iters: Option<usize>,
    time_limit_ms: Option<u64>,
    warm_start: Option<WarmStart>,
) -> SolverSettings {
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
    if let Some(v) = kkt_refine_iters {
        settings.kkt_refine_iters = v;
    }
    if let Some(v) = mcc_iters {
        settings.mcc_iters = v;
    }
    if let Some(v) = centrality_beta {
        settings.centrality_beta = v;
    }
    if let Some(v) = centrality_gamma {
        settings.centrality_gamma = v;
    }
    if let Some(v) = line_search_max_iters {
        settings.line_search_max_iters = v;
    }
    if let Some(v) = time_limit_ms {
        settings.time_limit_ms = Some(v);
    }
    settings.warm_start = warm_start;
    settings
}

fn solve_with_backend(
    solver: Option<&str>,
    problem: &ProblemData,
    settings: &SolverSettings,
) -> PyResult<SolveResult> {
    let result = match solver {
        None => solve(problem, settings),
        Some(name) if name.eq_ignore_ascii_case("ipm") => solve(problem, settings),
        Some(name) if name.eq_ignore_ascii_case("ipm2") => ipm2::solve_ipm2(problem, settings),
        Some(name) => {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "Unknown solver '{}'. Expected 'ipm' or 'ipm2'.",
                name
            )));
        }
    };

    result.map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("Solver error: {}", e)))
}

fn update_vec_from_array(
    target: &mut Vec<f64>,
    source: &PyReadonlyArray1<f64>,
    name: &str,
) -> PyResult<()> {
    let slice = source.as_slice()?;
    if slice.len() != target.len() {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "{} has length {}, expected {}",
            name,
            slice.len(),
            target.len()
        )));
    }
    target.copy_from_slice(slice);
    Ok(())
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
            SolveStatus::AlmostOptimal => "almost_optimal",
            SolveStatus::PrimalInfeasible => "primal_infeasible",
            SolveStatus::DualInfeasible => "dual_infeasible",
            SolveStatus::Unbounded => "unbounded",
            SolveStatus::MaxIters => "max_iterations",
            SolveStatus::TimeLimit => "time_limit",
            SolveStatus::NumericalError => "numerical_error",
            SolveStatus::NumericalLimit => "numerical_limit",
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

/// Persistent solver instance for repeated solves with updated parameters.
#[pyclass]
pub struct MinixSolver {
    problem: ProblemData,
}

#[pymethods]
impl MinixSolver {
    #[new]
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
        p_data = None
    ))]
    fn new(
        a_indptr: PyReadonlyArray1<i64>,
        a_indices: PyReadonlyArray1<i64>,
        a_data: PyReadonlyArray1<f64>,
        a_shape: (usize, usize),
        q: PyReadonlyArray1<f64>,
        b: PyReadonlyArray1<f64>,
        cones: Vec<(String, usize)>,
        p_indptr: Option<PyReadonlyArray1<i64>>,
        p_indices: Option<PyReadonlyArray1<i64>>,
        p_data: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Self> {
        let problem = build_problem(
            a_indptr, a_indices, a_data, a_shape, q, b, cones, p_indptr, p_indices, p_data,
        )?;
        Ok(Self { problem })
    }

    #[pyo3(signature = (
        q = None,
        b = None,
        max_iter = None,
        verbose = None,
        tol_feas = None,
        tol_gap = None,
        kkt_refine_iters = None,
        mcc_iters = None,
        centrality_beta = None,
        centrality_gamma = None,
        line_search_max_iters = None,
        time_limit_ms = None,
        warm_x = None,
        warm_s = None,
        warm_z = None,
        warm_tau = None,
        warm_kappa = None,
        solver = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn solve(
        &mut self,
        q: Option<PyReadonlyArray1<f64>>,
        b: Option<PyReadonlyArray1<f64>>,
        max_iter: Option<usize>,
        verbose: Option<bool>,
        tol_feas: Option<f64>,
        tol_gap: Option<f64>,
        kkt_refine_iters: Option<usize>,
        mcc_iters: Option<usize>,
        centrality_beta: Option<f64>,
        centrality_gamma: Option<f64>,
        line_search_max_iters: Option<usize>,
        time_limit_ms: Option<u64>,
        warm_x: Option<PyReadonlyArray1<f64>>,
        warm_s: Option<PyReadonlyArray1<f64>>,
        warm_z: Option<PyReadonlyArray1<f64>>,
        warm_tau: Option<f64>,
        warm_kappa: Option<f64>,
        solver: Option<String>,
    ) -> PyResult<MinixResult> {
        if let Some(q_arr) = q {
            update_vec_from_array(&mut self.problem.q, &q_arr, "q")?;
        }
        if let Some(b_arr) = b {
            update_vec_from_array(&mut self.problem.b, &b_arr, "b")?;
        }

        let warm_start = build_warm_start(warm_x, warm_s, warm_z, warm_tau, warm_kappa)?;
        let settings = build_settings(
            max_iter,
            verbose,
            tol_feas,
            tol_gap,
            kkt_refine_iters,
            mcc_iters,
            centrality_beta,
            centrality_gamma,
            line_search_max_iters,
            time_limit_ms,
            warm_start,
        );

        let result = solve_with_backend(solver.as_deref(), &self.problem, &settings)?;
        Ok(MinixResult::from(result))
    }

    #[pyo3(signature = (q = None, b = None))]
    fn update(
        &mut self,
        q: Option<PyReadonlyArray1<f64>>,
        b: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<()> {
        if let Some(q_arr) = q {
            update_vec_from_array(&mut self.problem.q, &q_arr, "q")?;
        }
        if let Some(b_arr) = b {
            update_vec_from_array(&mut self.problem.b, &b_arr, "b")?;
        }
        Ok(())
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
/// * `warm_x` - Warm-start primal vector (optional)
/// * `warm_s` - Warm-start slack vector (optional)
/// * `warm_z` - Warm-start dual vector (optional)
/// * `warm_tau` - Warm-start tau value (optional)
/// * `warm_kappa` - Warm-start kappa value (optional)
/// * `solver` - Solver backend ("ipm" or "ipm2")
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
    kkt_refine_iters = None,
    mcc_iters = None,
    centrality_beta = None,
    centrality_gamma = None,
    line_search_max_iters = None,
    time_limit_ms = None,
    warm_x = None,
    warm_s = None,
    warm_z = None,
    warm_tau = None,
    warm_kappa = None,
    solver = None
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
    kkt_refine_iters: Option<usize>,
    mcc_iters: Option<usize>,
    centrality_beta: Option<f64>,
    centrality_gamma: Option<f64>,
    line_search_max_iters: Option<usize>,
    time_limit_ms: Option<u64>,
    warm_x: Option<PyReadonlyArray1<f64>>,
    warm_s: Option<PyReadonlyArray1<f64>>,
    warm_z: Option<PyReadonlyArray1<f64>>,
    warm_tau: Option<f64>,
    warm_kappa: Option<f64>,
    solver: Option<String>,
) -> PyResult<MinixResult> {
    let warm_start = build_warm_start(warm_x, warm_s, warm_z, warm_tau, warm_kappa)?;

    let problem = build_problem(
        a_indptr,
        a_indices,
        a_data,
        a_shape,
        q,
        b,
        cones,
        p_indptr,
        p_indices,
        p_data,
    )?;

    let settings = build_settings(
        max_iter,
        verbose,
        tol_feas,
        tol_gap,
        kkt_refine_iters,
        mcc_iters,
        centrality_beta,
        centrality_gamma,
        line_search_max_iters,
        time_limit_ms,
        warm_start,
    );

    let result = solve_with_backend(solver.as_deref(), &problem, &settings)?;

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
    dict.set_item("time_limit_ms", settings.time_limit_ms)?;
    dict.set_item("verbose", settings.verbose)?;
    dict.set_item("tol_feas", settings.tol_feas)?;
    dict.set_item("tol_gap", settings.tol_gap)?;
    dict.set_item("tol_infeas", settings.tol_infeas)?;
    dict.set_item("ruiz_iters", settings.ruiz_iters)?;
    dict.set_item("static_reg", settings.static_reg)?;
    dict.set_item("dynamic_reg_min_pivot", settings.dynamic_reg_min_pivot)?;
    dict.set_item("kkt_refine_iters", settings.kkt_refine_iters)?;
    dict.set_item("mcc_iters", settings.mcc_iters)?;
    dict.set_item("centrality_beta", settings.centrality_beta)?;
    dict.set_item("centrality_gamma", settings.centrality_gamma)?;
    dict.set_item("line_search_max_iters", settings.line_search_max_iters)?;
    Ok(dict)
}

/// Python module definition.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_conic, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(default_settings, m)?)?;
    m.add_class::<MinixResult>()?;
    m.add_class::<MinixSolver>()?;
    Ok(())
}
