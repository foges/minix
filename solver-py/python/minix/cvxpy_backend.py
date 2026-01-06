"""CVXPY solver interface for Minix.

This module provides a CVXPY-compatible solver backend for Minix, allowing
users to solve optimization problems using `problem.solve(solver="MINIX")`.

To use this backend, you need to register it with CVXPY:

    >>> from minix.cvxpy_backend import MINIX
    >>> import cvxpy as cp
    >>> # Register the solver
    >>> from cvxpy.reductions.solvers import defines as slv_def
    >>> slv_def.INSTALLED_SOLVERS["MINIX"] = MINIX
    >>>
    >>> # Now you can use it
    >>> x = cp.Variable(2)
    >>> problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), [x >= 0])
    >>> problem.solve(solver="MINIX")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse
import scipy.sparse as sp

import minix

if TYPE_CHECKING:
    from cvxpy.constraints.constraint import Constraint
    from cvxpy.problems.problem import Problem
    from cvxpy.reductions.solution import Solution

# CVXPY imports - these may fail if cvxpy is not installed
try:
    import cvxpy.settings as s
    from cvxpy.constraints import ExpCone, NonNeg, PSD, PowCone3D, SOC, Zero
    from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver

    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    ConicSolver = object  # type: ignore[misc,assignment]


def _get_status_map() -> dict[str, str]:
    """Map minix status strings to CVXPY status constants."""
    return {
        "optimal": s.OPTIMAL,
        "primal_infeasible": s.INFEASIBLE,
        "dual_infeasible": s.UNBOUNDED,
        "unbounded": s.UNBOUNDED,
        "max_iterations": s.OPTIMAL_INACCURATE,  # May have converged close enough
        "time_limit": s.OPTIMAL_INACCURATE,
        "numerical_error": s.SOLVER_ERROR,
    }


class MINIX(ConicSolver):
    """CVXPY interface for the Minix conic optimization solver.

    Minix is a high-performance interior point solver supporting:
    - Linear Programs (LP)
    - Quadratic Programs (QP)
    - Second-Order Cone Programs (SOCP)

    Example:
        >>> import cvxpy as cp
        >>> from minix.cvxpy_backend import MINIX
        >>>
        >>> x = cp.Variable(2)
        >>> constraints = [x >= 0, cp.sum(x) == 1]
        >>> problem = cp.Problem(cp.Minimize(x[0] + x[1]), constraints)
        >>> problem.solve(solver=MINIX())
    """

    # Solver capabilities
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = [Zero, NonNeg, SOC, PSD, ExpCone]  # PSD, EXP now supported

    def name(self) -> str:
        """Return the solver name."""
        return "MINIX"

    @staticmethod
    def cite() -> str:
        """Return a citation string for the solver."""
        return "Minix: A high-performance conic optimization solver"

    def import_solver(self) -> Any:
        """Import and return the minix module."""
        import minix

        return minix

    def supports_quad_obj(self) -> bool:
        """Return True since minix supports quadratic objectives."""
        return True

    @staticmethod
    def psd_format_mat(constr: Any) -> Any:
        """Return a linear operator to multiply by PSD constraint coefficients.

        Transforms full matrix form to svec (scaled lower triangular) form.
        Off-diagonal elements are scaled by sqrt(2) for proper inner product.
        """
        rows = cols = constr.expr.shape[0]
        entries = rows * (cols + 1) // 2

        row_arr = np.arange(0, entries)

        lower_diag_indices = np.tril_indices(rows)
        col_arr = np.sort(np.ravel_multi_index(lower_diag_indices,
                                               (rows, cols),
                                               order='F'))

        val_arr = np.zeros((rows, cols))
        val_arr[lower_diag_indices] = np.sqrt(2)
        np.fill_diagonal(val_arr, 1.0)
        val_arr = np.ravel(val_arr, order='F')
        val_arr = val_arr[np.nonzero(val_arr)]

        shape = (entries, rows * cols)
        scaled_lower_tri = sp.csc_array((val_arr, (row_arr, col_arr)), shape)

        idx = np.arange(rows * cols)
        val_symm = 0.5 * np.ones(2 * rows * cols)
        K = idx.reshape((rows, cols))
        row_symm = np.append(idx, np.ravel(K, order='F'))
        col_symm = np.append(idx, np.ravel(K.T, order='F'))
        symm_matrix = sp.csc_array((val_symm, (row_symm, col_symm)))

        return scaled_lower_tri @ symm_matrix

    def apply(
        self,
        problem: Problem,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Convert problem to solver-compatible form.

        This uses the parent class's standard conic form conversion.
        """
        return super().apply(problem, *args, **kwargs)

    def solve_via_data(
        self,
        data: dict[str, Any],
        warm_start: bool,
        verbose: bool,
        solver_opts: dict[str, Any],
        solver_cache: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Solve using the minix solver.

        Args:
            data: Problem data from CVXPY's conic form.
            warm_start: Whether to warm start (not yet supported by minix).
            verbose: Whether to print solver output.
            solver_opts: Additional solver options.
            solver_cache: Cache for warm starting (not used).

        Returns:
            Dictionary with solution data.
        """
        # Extract problem data from CVXPY's conic form
        # CVXPY provides: c (objective), A (constraint matrix), b (RHS)
        # and dims (cone dimensions)
        c = data[s.C]
        A = data[s.A]
        b = data[s.B]
        dims = data["dims"]

        # Get quadratic term if present (for QP)
        P = data.get(s.P, None)

        # Build cone specification from CVXPY dims
        cones: list[tuple[str, int]] = []

        # Zero cone (equality constraints)
        if dims.zero > 0:
            cones.append(("zero", dims.zero))

        # Nonnegative cone (inequality constraints)
        if dims.nonneg > 0:
            cones.append(("nonneg", dims.nonneg))

        # Second-order cones
        for soc_dim in dims.soc:
            cones.append(("soc", soc_dim))

        # PSD cones (if supported)
        for psd_dim in dims.psd:
            # psd_dim is the matrix size n, svec dimension is n(n+1)/2
            svec_dim = psd_dim * (psd_dim + 1) // 2
            cones.append(("psd", svec_dim))

        # Exponential cones
        if hasattr(dims, "exp") and dims.exp > 0:
            cones.append(("exp", dims.exp * 3))

        # Convert to sparse CSC if needed
        if not sparse.issparse(A):
            A = sparse.csc_matrix(A)
        else:
            A = sparse.csc_matrix(A)

        if P is not None:
            if not sparse.issparse(P):
                P = sparse.csc_matrix(P)
            else:
                P = sparse.csc_matrix(P)

        # Extract solver options
        max_iter = solver_opts.get("max_iter", None)
        tol_feas = solver_opts.get("eps_abs", solver_opts.get("tol_feas", None))
        tol_gap = solver_opts.get("eps_rel", solver_opts.get("tol_gap", None))
        time_limit = solver_opts.get("time_limit", None)
        time_limit_ms = int(time_limit * 1000) if time_limit is not None else None
        kkt_refine_iters = solver_opts.get("kkt_refine_iters", None)
        mcc_iters = solver_opts.get("mcc_iters", None)
        centrality_beta = solver_opts.get("centrality_beta", None)
        centrality_gamma = solver_opts.get("centrality_gamma", None)
        line_search_max_iters = solver_opts.get("line_search_max_iters", None)

        # Call minix solver
        result = minix.solve(
            A=A,
            b=b,
            q=c,  # CVXPY uses 'c' for linear objective, minix uses 'q'
            cones=cones,
            P=P,
            max_iter=max_iter,
            verbose=verbose,
            tol_feas=tol_feas,
            tol_gap=tol_gap,
            kkt_refine_iters=kkt_refine_iters,
            mcc_iters=mcc_iters,
            centrality_beta=centrality_beta,
            centrality_gamma=centrality_gamma,
            line_search_max_iters=line_search_max_iters,
            time_limit_ms=time_limit_ms,
        )

        # Map status
        status_map = _get_status_map()
        status = status_map.get(result.status, s.SOLVER_ERROR)

        # Build solution dictionary
        solution: dict[str, Any] = {
            s.STATUS: status,
            s.VALUE: result.obj_val,
            s.SOLVE_TIME: result.solve_time_ms / 1000.0,  # Convert to seconds
            s.NUM_ITERS: result.iterations,
        }

        if status in s.SOLUTION_PRESENT:
            solution[s.PRIMAL] = result.x()
            # ConicSolver.invert() expects eq_dual and ineq_dual
            # Split the dual vector based on cone structure
            y = result.y()
            n_eq = dims.zero if dims.zero > 0 else 0
            solution[s.EQ_DUAL] = y[:n_eq] if n_eq > 0 else np.array([])
            solution[s.INEQ_DUAL] = y[n_eq:]

        return solution

    def invert(
        self,
        solution: dict[str, Any],
        inverse_data: dict[str, Any],
    ) -> Solution:
        """Map solver output back to CVXPY solution.

        This uses the parent class's standard inversion.
        """
        return super().invert(solution, inverse_data)


def install() -> None:
    """Register MINIX solver with CVXPY.

    Call this function to make MINIX available as a solver in CVXPY:

        >>> from minix.cvxpy_backend import install
        >>> install()
        >>> problem.solve(solver="MINIX")
    """
    if not CVXPY_AVAILABLE:
        msg = "CVXPY is not installed. Install it with: pip install cvxpy"
        raise ImportError(msg)

    from cvxpy.reductions.solvers import defines as slv_def

    slv_def.INSTALLED_SOLVERS["MINIX"] = MINIX


def is_installed() -> bool:
    """Check if MINIX is registered with CVXPY."""
    if not CVXPY_AVAILABLE:
        return False

    try:
        from cvxpy.reductions.solvers import defines as slv_def

        return "MINIX" in slv_def.INSTALLED_SOLVERS
    except Exception:
        return False
