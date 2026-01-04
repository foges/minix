"""Minix: A high-performance conic optimization solver.

Minix solves convex conic optimization problems of the form:

    minimize    (1/2) x^T P x + q^T x
    subject to  A x + s = b
                s ∈ K

where K is a Cartesian product of cones (zero, nonnegative orthant,
second-order cone, positive semidefinite cone, exponential cone).

Example usage:
    >>> import numpy as np
    >>> from scipy import sparse
    >>> import minix
    >>>
    >>> # Simple LP: min x + y s.t. x + y = 1, x >= 0, y >= 0
    >>> A = sparse.csc_matrix([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    >>> b = np.array([1.0, 0.0, 0.0])
    >>> q = np.array([1.0, 1.0])
    >>> cones = [("zero", 1), ("nonneg", 2)]
    >>>
    >>> result = minix.solve(A, b, q, cones)
    >>> print(result.status)  # 'optimal'
    >>> print(result.x())     # Solution vector
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from minix._native import (
    MinixResult,
    MinixSolver as _NativeSolver,
    default_settings,
    solve_conic,
    version,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "solve",
    "Solver",
    "MinixResult",
    "version",
    "default_settings",
]

__version__ = version()


def _prepare_problem_inputs(
    A: sparse.spmatrix,
    b: NDArray[np.floating],
    q: NDArray[np.floating],
    cones: list[tuple[str, int]],
    P: sparse.spmatrix | None,
) -> tuple[sparse.csc_matrix, NDArray[np.floating], NDArray[np.floating], NDArray[np.int64] | None, NDArray[np.int64] | None, NDArray[np.floating] | None, int, int]:
    A_csc = sparse.csc_matrix(A)
    m, n = A_csc.shape

    b = np.asarray(b, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()

    if len(b) != m:
        msg = f"b has length {len(b)}, expected {m} (number of rows in A)"
        raise ValueError(msg)
    if len(q) != n:
        msg = f"q has length {len(q)}, expected {n} (number of columns in A)"
        raise ValueError(msg)

    total_cone_dim = sum(dim for _, dim in cones)
    if total_cone_dim != m:
        msg = f"Cone dimensions sum to {total_cone_dim}, expected {m} (number of rows in A)"
        raise ValueError(msg)

    p_indptr = None
    p_indices = None
    p_data = None

    if P is not None:
        P_csc = sparse.csc_matrix(P)
        if P_csc.shape != (n, n):
            msg = f"P has shape {P_csc.shape}, expected ({n}, {n})"
            raise ValueError(msg)
        p_indptr = P_csc.indptr.astype(np.int64)
        p_indices = P_csc.indices.astype(np.int64)
        p_data = P_csc.data.astype(np.float64)

    return A_csc, b, q, p_indptr, p_indices, p_data, m, n


def _as_vec(
    vec: NDArray[np.floating] | None,
    name: str,
    expected_len: int | None,
) -> NDArray[np.floating] | None:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float64).ravel()
    if expected_len is not None and len(arr) != expected_len:
        msg = f"{name} has length {len(arr)}, expected {expected_len}"
        raise ValueError(msg)
    return arr


def solve(
    A: sparse.spmatrix,
    b: NDArray[np.floating],
    q: NDArray[np.floating],
    cones: list[tuple[str, int]],
    P: sparse.spmatrix | None = None,
    *,
    max_iter: int | None = None,
    verbose: bool = False,
    tol_feas: float | None = None,
    tol_gap: float | None = None,
    kkt_refine_iters: int | None = None,
    mcc_iters: int | None = None,
    centrality_beta: float | None = None,
    centrality_gamma: float | None = None,
    line_search_max_iters: int | None = None,
    time_limit_ms: int | None = None,
    warm_start: MinixResult | None = None,
    warm_x: NDArray[np.floating] | None = None,
    warm_s: NDArray[np.floating] | None = None,
    warm_z: NDArray[np.floating] | None = None,
    warm_tau: float | None = None,
    warm_kappa: float | None = None,
    solver: str | None = None,
) -> MinixResult:
    """Solve a conic optimization problem.

    Problem form:
        minimize    (1/2) x^T P x + q^T x
        subject to  A x + s = b
                    s ∈ K

    Args:
        A: Constraint matrix (m x n), will be converted to CSC format.
        b: Constraint right-hand side vector (length m).
        q: Linear cost vector (length n).
        cones: List of (cone_type, dimension) tuples specifying the cone K.
            Supported types: "zero", "nonneg", "soc", "psd", "exp".
        P: Quadratic cost matrix (n x n), optional. Must be positive semidefinite.
            If None, problem is a linear program.
        max_iter: Maximum number of interior point iterations.
        verbose: If True, print solver progress.
        tol_feas: Primal/dual feasibility tolerance.
        tol_gap: Duality gap tolerance.
        kkt_refine_iters: Iterative refinement steps for each KKT solve.
        mcc_iters: Multiple centrality correction iterations.
        centrality_beta: Lower bound for s_i z_i relative to μ.
        centrality_gamma: Upper bound for s_i z_i relative to μ.
        line_search_max_iters: Max backtracking steps for centrality line search.
        time_limit_ms: Time limit in milliseconds.
        warm_start: MinixResult to reuse for warm start (x/s/z).
        warm_x: Warm start primal vector (length n).
        warm_s: Warm start slack vector (length m).
        warm_z: Warm start dual vector (length m).
        warm_tau: Warm start tau value.
        warm_kappa: Warm start kappa value.
        solver: Solver backend to use ("ipm" or "ipm2").

    Returns:
        MinixResult with solution status, primal/dual solutions, and diagnostics.

    Raises:
        ValueError: If inputs have incompatible dimensions.
        RuntimeError: If solver encounters an internal error.

    Example:
        >>> # Quadratic program: min 0.5*x^2 + x s.t. x >= 0
        >>> A = sparse.csc_matrix([[-1.0]])
        >>> b = np.array([0.0])
        >>> q = np.array([1.0])
        >>> P = sparse.csc_matrix([[1.0]])
        >>> cones = [("nonneg", 1)]
        >>> result = minix.solve(A, b, q, cones, P=P)
    """
    A_csc, b, q, p_indptr, p_indices, p_data, m, n = _prepare_problem_inputs(
        A, b, q, cones, P
    )

    if warm_start is not None:
        if warm_x is None:
            warm_x = warm_start.x()
        if warm_s is None:
            warm_s = warm_start.s()
        if warm_z is None:
            warm_z = warm_start.z()

    warm_x = _as_vec(warm_x, "warm_x", None)
    warm_s = _as_vec(warm_s, "warm_s", None)
    warm_z = _as_vec(warm_z, "warm_z", None)

    # Call native solver
    return solve_conic(
        a_indptr=A_csc.indptr.astype(np.int64),
        a_indices=A_csc.indices.astype(np.int64),
        a_data=A_csc.data.astype(np.float64),
        a_shape=(m, n),
        q=q,
        b=b,
        cones=cones,
        p_indptr=p_indptr,
        p_indices=p_indices,
        p_data=p_data,
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
        warm_x=warm_x,
        warm_s=warm_s,
        warm_z=warm_z,
        warm_tau=warm_tau,
        warm_kappa=warm_kappa,
        solver=solver,
    )


class Solver:
    """Persistent solver for repeated solves with updated parameters."""

    def __init__(
        self,
        A: sparse.spmatrix,
        b: NDArray[np.floating],
        q: NDArray[np.floating],
        cones: list[tuple[str, int]],
        P: sparse.spmatrix | None = None,
    ) -> None:
        A_csc, b_vec, q_vec, p_indptr, p_indices, p_data, m, n = _prepare_problem_inputs(
            A, b, q, cones, P
        )
        self._m = m
        self._n = n
        self._solver = _NativeSolver(
            a_indptr=A_csc.indptr.astype(np.int64),
            a_indices=A_csc.indices.astype(np.int64),
            a_data=A_csc.data.astype(np.float64),
            a_shape=(m, n),
            q=q_vec,
            b=b_vec,
            cones=cones,
            p_indptr=p_indptr,
            p_indices=p_indices,
            p_data=p_data,
        )

    def update(
        self,
        *,
        q: NDArray[np.floating] | None = None,
        b: NDArray[np.floating] | None = None,
    ) -> None:
        q_vec = _as_vec(q, "q", self._n)
        b_vec = _as_vec(b, "b", self._m)
        self._solver.update(q=q_vec, b=b_vec)

    def solve(
        self,
        *,
        q: NDArray[np.floating] | None = None,
        b: NDArray[np.floating] | None = None,
        max_iter: int | None = None,
        verbose: bool | None = None,
        tol_feas: float | None = None,
        tol_gap: float | None = None,
        kkt_refine_iters: int | None = None,
        mcc_iters: int | None = None,
        centrality_beta: float | None = None,
        centrality_gamma: float | None = None,
        line_search_max_iters: int | None = None,
        time_limit_ms: int | None = None,
        warm_start: MinixResult | None = None,
        warm_x: NDArray[np.floating] | None = None,
        warm_s: NDArray[np.floating] | None = None,
        warm_z: NDArray[np.floating] | None = None,
        warm_tau: float | None = None,
        warm_kappa: float | None = None,
        solver: str | None = None,
    ) -> MinixResult:
        q_vec = _as_vec(q, "q", self._n)
        b_vec = _as_vec(b, "b", self._m)

        if warm_start is not None:
            if warm_x is None:
                warm_x = warm_start.x()
            if warm_s is None:
                warm_s = warm_start.s()
            if warm_z is None:
                warm_z = warm_start.z()

        warm_x = _as_vec(warm_x, "warm_x", None)
        warm_s = _as_vec(warm_s, "warm_s", None)
        warm_z = _as_vec(warm_z, "warm_z", None)

        return self._solver.solve(
            q=q_vec,
            b=b_vec,
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
            warm_x=warm_x,
            warm_s=warm_s,
            warm_z=warm_z,
            warm_tau=warm_tau,
            warm_kappa=warm_kappa,
            solver=solver,
        )
