"""CVXPY solver interface for Minix."""

import numpy as np
from scipy import sparse as sp
import cvxpy.settings as s
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.psd import PSD
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver

# Status mapping from Minix to CVXPY (minix uses lowercase)
STATUS_MAP = {
    "optimal": s.OPTIMAL,
    "almost_optimal": s.OPTIMAL_INACCURATE,
    "primal_infeasible": s.INFEASIBLE,
    "dual_infeasible": s.UNBOUNDED,
    "max_iterations": s.USER_LIMIT,
    "time_limit": s.USER_LIMIT,
    "numerical_error": s.SOLVER_ERROR,
    "numerical_limit": s.SOLVER_ERROR,
    "unknown": s.SOLVER_ERROR,
}


def dims_to_minix_cones(cone_dims):
    """Convert CVXPY cone dimensions to Minix cone list."""
    cones = []

    # Zero cone (equality constraints)
    if cone_dims.zero > 0:
        cones.append(("zero", cone_dims.zero))

    # Nonnegative cone
    if cone_dims.nonneg > 0:
        cones.append(("nonneg", cone_dims.nonneg))

    # Second-order cones
    for dim in cone_dims.soc:
        cones.append(("soc", dim))

    # Exponential cones
    if cone_dims.exp > 0:
        for _ in range(cone_dims.exp):
            cones.append(("exp", 3))

    # PSD cones
    for dim in cone_dims.psd:
        # CVXPY gives matrix dimension n, Minix expects svec dimension n*(n+1)/2
        svec_dim = dim * (dim + 1) // 2
        cones.append(("psd", svec_dim))

    return cones


class MINIX(ConicSolver):
    """CVXPY interface for the Minix conic solver."""

    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [
        SOC,
        ExpCone,
        PSD,
    ]

    def name(self):
        return "MINIX"

    def import_solver(self):
        import minix
        return minix

    def supports_quad_obj(self) -> bool:
        """Minix supports quadratic objectives directly."""
        return True

    @staticmethod
    def cite():
        """Return citation string."""
        return "Minix: Conic optimization solver"

    def solve_via_data(
        self,
        data,
        warm_start: bool,
        verbose: bool,
        solver_opts,
        solver_cache=None,
    ):
        import minix

        # Extract problem data
        c = data[s.C]
        b = data[s.B]
        A = data[s.A]
        cone_dims = data[ConicSolver.DIMS]

        # Convert A to CSC if needed
        if not sp.isspmatrix_csc(A):
            A = sp.csc_matrix(A)

        # Extract P matrix if present (quadratic objective)
        P = None
        if s.P in data:
            P = data[s.P]
            if not sp.isspmatrix_csc(P):
                P = sp.csc_matrix(P)
            # CVXPY uses upper triangle
            P = sp.triu(P).tocsc()

        # Convert cones
        cones = dims_to_minix_cones(cone_dims)

        # Build solver options
        opts = {
            "verbose": verbose,
            "max_iter": solver_opts.get("max_iter", 200),
            "tol_feas": solver_opts.get("tol_feas", 1e-8),
            "tol_gap": solver_opts.get("tol_gap", 1e-8),
        }

        # Add optional settings
        for key in ["time_limit_ms", "kkt_refine_iters"]:
            if key in solver_opts:
                opts[key] = solver_opts[key]

        # Build arguments for solve_conic
        kwargs = {
            "a_indptr": A.indptr.astype(np.int64),
            "a_indices": A.indices.astype(np.int64),
            "a_data": A.data.astype(np.float64),
            "a_shape": A.shape,
            "q": np.asarray(c).flatten().astype(np.float64),
            "b": np.asarray(b).flatten().astype(np.float64),
            "cones": cones,
            **opts,
        }

        # Add P matrix if present
        if P is not None:
            kwargs["p_indptr"] = P.indptr.astype(np.int64)
            kwargs["p_indices"] = P.indices.astype(np.int64)
            kwargs["p_data"] = P.data.astype(np.float64)

        # Solve
        result = minix.solve_conic(**kwargs)

        # Extract solution (x, s, z are methods that return numpy arrays)
        sol = {
            "x": np.array(result.x()),
            "s": np.array(result.s()),
            "z": np.array(result.z()),
            "obj_val": result.obj_val,
            "status": result.status,
            "iterations": result.iterations,
            "solve_time_ms": getattr(result, "solve_time_ms", None),
        }

        return sol

    def invert(self, solution, inverse_data):
        """Map solver solution back to CVXPY problem."""
        status = STATUS_MAP.get(solution["status"], s.SOLVER_ERROR)

        if status in s.SOLUTION_PRESENT:
            primal_vars = {inverse_data[self.VAR_ID]: solution["x"]}
            dual_vars = utilities.get_dual_values(
                solution["z"],
                utilities.extract_dual_value,
                inverse_data[self.NEQ_CONSTR],
            )
            # Handle equality constraints
            eq_dual = utilities.get_dual_values(
                solution["z"],
                utilities.extract_dual_value,
                inverse_data[self.EQ_CONSTR],
            )
            dual_vars.update(eq_dual)

            return Solution(
                status,
                solution["obj_val"],
                primal_vars,
                dual_vars,
                {
                    "iterations": solution["iterations"],
                    "solve_time_ms": solution.get("solve_time_ms"),
                },
            )
        else:
            return failure_solution(status)


# Register the solver with CVXPY
def register_minix():
    """Register MINIX solver with CVXPY."""
    import cvxpy
    from cvxpy.reductions.solvers.defines import (
        INSTALLED_SOLVERS,
        CONIC_SOLVERS,
        SOLVER_MAP_CONIC,
    )

    solver_instance = MINIX()

    # Add to solver maps
    if "MINIX" not in INSTALLED_SOLVERS:
        INSTALLED_SOLVERS.append("MINIX")

    if "MINIX" not in CONIC_SOLVERS:
        CONIC_SOLVERS.append("MINIX")

    if "MINIX" not in SOLVER_MAP_CONIC:
        SOLVER_MAP_CONIC["MINIX"] = solver_instance

    # Add solver class to cvxpy module
    cvxpy.MINIX = solver_instance

    return solver_instance
