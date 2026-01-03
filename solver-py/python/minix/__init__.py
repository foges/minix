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
    default_settings,
    solve_conic,
    version,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "solve",
    "MinixResult",
    "version",
    "default_settings",
]

__version__ = version()


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
    # Convert A to CSC format
    A_csc = sparse.csc_matrix(A)
    m, n = A_csc.shape

    # Validate dimensions
    b = np.asarray(b, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()

    if len(b) != m:
        msg = f"b has length {len(b)}, expected {m} (number of rows in A)"
        raise ValueError(msg)
    if len(q) != n:
        msg = f"q has length {len(q)}, expected {n} (number of columns in A)"
        raise ValueError(msg)

    # Validate cone dimensions
    total_cone_dim = sum(dim for _, dim in cones)
    if total_cone_dim != m:
        msg = f"Cone dimensions sum to {total_cone_dim}, expected {m} (number of rows in A)"
        raise ValueError(msg)

    # Prepare P if provided
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
    )
