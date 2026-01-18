#!/usr/bin/env python3
"""Compare SOCP solutions between Minix and Clarabel."""

import numpy as np
import clarabel
from scipy import sparse
import os
# Problems to test - ones where we saw AlmostOptimal or failures
TEST_PROBLEMS = [
    "HS21", "HS35", "HS268",
    "DUAL1", "DUAL4",
    "DUALC1", "DUALC8",
    "PRIMAL1", "PRIMALC1", "PRIMALC2", "PRIMALC5", "PRIMALC8",
]

def load_qps(name: str):
    """Load a QPS file from the cache."""
    home = os.path.expanduser("~")
    cache_dir = os.path.join(home, ".cache", "minix-bench", "maros-meszaros")
    path = os.path.join(cache_dir, f"{name}.QPS")
    if not os.path.exists(path):
        return None

    # Simple QPS parser
    n = 0
    m = 0
    q = []
    b = []
    P_data = []  # (row, col, val)
    A_data = []  # (row, col, val)

    var_names = {}
    con_names = {}

    section = None
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*'):
                continue

            parts = line.split()
            if parts[0] in ['NAME', 'ROWS', 'COLUMNS', 'RHS', 'RANGES', 'BOUNDS', 'QUADOBJ', 'ENDATA']:
                section = parts[0]
                continue

            if section == 'ROWS':
                row_type = parts[0]
                row_name = parts[1]
                if row_type == 'N':
                    # Objective row - skip
                    pass
                else:
                    con_names[row_name] = len(con_names)

            elif section == 'COLUMNS':
                var_name = parts[0]
                if var_name not in var_names:
                    var_names[var_name] = len(var_names)
                col = var_names[var_name]

                i = 1
                while i < len(parts):
                    row_name = parts[i]
                    val = float(parts[i+1])
                    if row_name in con_names:
                        A_data.append((con_names[row_name], col, val))
                    i += 2

            elif section == 'RHS':
                i = 1
                while i < len(parts):
                    row_name = parts[i]
                    val = float(parts[i+1])
                    if row_name in con_names:
                        row = con_names[row_name]
                        while len(b) <= row:
                            b.append(0.0)
                        b[row] = val
                    i += 2

            elif section == 'QUADOBJ':
                col1_name = parts[0]
                col2_name = parts[1]
                val = float(parts[2])
                col1 = var_names[col1_name]
                col2 = var_names[col2_name]
                P_data.append((col1, col2, val))
                if col1 != col2:
                    P_data.append((col2, col1, val))

    n = len(var_names)
    m = len(con_names)

    # Build matrices
    if P_data:
        rows, cols, vals = zip(*P_data)
        P = sparse.csc_matrix((vals, (rows, cols)), shape=(n, n))
    else:
        P = sparse.csc_matrix((n, n))

    if A_data:
        rows, cols, vals = zip(*A_data)
        A = sparse.csc_matrix((vals, (rows, cols)), shape=(m, n))
    else:
        A = sparse.csc_matrix((m, n))

    q = np.zeros(n)  # Linear cost (extracted from objective row if needed)
    b = np.array(b + [0.0] * (m - len(b)))

    return {'P': P, 'A': A, 'q': q, 'b': b, 'n': n, 'm': m}


def qp_to_socp(qp):
    """Convert QP to SOCP form (like CVXPY does).

    QP: min 0.5 x'Px + q'x  s.t. Ax = b, x >= 0

    SOCP: min t  s.t. ||Lx|| <= t+1, Ax = b, x >= 0
    where P = L'L (Cholesky)
    """
    P = qp['P']
    A = qp['A']
    q = qp['q']
    b = qp['b']
    n = qp['n']
    m = qp['m']

    # Check if P is zero (LP, not QP)
    if P.nnz == 0:
        return None  # Skip LPs

    # Cholesky of P (with regularization for semidefinite cases)
    try:
        # Make sure P is symmetric and get dense version
        P_dense = P.toarray()
        P_sym = 0.5 * (P_dense + P_dense.T)
        # Add small regularization for numerical stability
        P_reg = P_sym + np.eye(n) * 1e-10
        L = np.linalg.cholesky(P_reg)
    except np.linalg.LinAlgError:
        # P not positive definite even with regularization
        return None

    # SOCP formulation:
    # Variables: [x (n), t (1)]
    # Constraints:
    #   Ax = b (equality)
    #   x >= 0 (nonneg)
    #   ||Lx||_2 <= t (SOC: [t; Lx] in SOC)

    return {
        'L': L,
        'A': A,
        'q': q,
        'b': b,
        'n': n,
        'm': m,
    }


def solve_with_clarabel(socp, tol=1e-8):
    """Solve SOCP using Clarabel directly."""
    L = socp['L']
    A_eq = socp['A']
    b_eq = socp['b']
    n = socp['n']
    m = socp['m']

    # Variables: [x (n), t (1)]
    n_vars = n + 1

    # Objective: min t (last variable)
    q_obj = np.zeros(n_vars)
    q_obj[-1] = 1.0  # minimize t

    # No quadratic objective in SOCP form
    P_obj = sparse.csc_matrix((n_vars, n_vars))

    # Constraints:
    # 1. Ax = b (equality, m rows)
    # 2. x >= 0 (nonneg, n rows)
    # 3. [t; Lx] in SOC (dim = 1 + n)

    soc_dim = 1 + n
    total_constraints = m + n + soc_dim

    # Build constraint matrix A_all
    A_rows = []
    A_cols = []
    A_vals = []

    # Equality constraints: A_eq * x = b_eq
    A_eq_coo = sparse.coo_matrix(A_eq)
    for i, j, v in zip(A_eq_coo.row, A_eq_coo.col, A_eq_coo.data):
        A_rows.append(i)
        A_cols.append(j)
        A_vals.append(v)

    # NonNeg constraints: -x <= 0 => -I * x in NonNeg
    for i in range(n):
        A_rows.append(m + i)
        A_cols.append(i)
        A_vals.append(-1.0)

    # SOC constraint: [t; Lx] in SOC
    # Row for t: coefficient is -1 on t variable
    A_rows.append(m + n)
    A_cols.append(n)  # t is last variable
    A_vals.append(-1.0)

    # Rows for Lx: L * x
    for i in range(n):
        for j in range(n):
            if abs(L[i, j]) > 1e-14:
                A_rows.append(m + n + 1 + i)
                A_cols.append(j)
                A_vals.append(-L[i, j])

    A_all = sparse.csc_matrix(
        (A_vals, (A_rows, A_cols)),
        shape=(total_constraints, n_vars)
    )

    # RHS
    b_all = np.zeros(total_constraints)
    b_all[:m] = b_eq
    # NonNeg and SOC have zero RHS

    # Cone specification
    cones = [
        clarabel.ZeroConeT(m),      # Equality
        clarabel.NonnegativeConeT(n),  # x >= 0
        clarabel.SecondOrderConeT(soc_dim),  # SOC
    ]

    # Solver settings
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_feas = tol
    settings.tol_gap_abs = tol
    settings.tol_gap_rel = tol

    # Solve
    solver = clarabel.DefaultSolver(P_obj, q_obj, A_all, b_all, cones, settings)
    solution = solver.solve()

    return {
        'status': str(solution.status),
        'obj': solution.obj_val,
        'x': solution.x[:n] if solution.x is not None else None,
        'iterations': solution.iterations,
        # Clarabel reports residuals
        'r_prim': solution.r_prim,
        'r_dual': solution.r_dual,
    }


def main():
    print("Comparing SOCP solutions: Minix vs Clarabel")
    print("=" * 70)
    print(f"{'Problem':<15} {'Clarabel Status':<20} {'Iters':>6} {'r_prim':>12} {'r_dual':>12}")
    print("-" * 70)

    for name in TEST_PROBLEMS:
        qp = load_qps(name)
        if qp is None:
            print(f"{name:<15} SKIP (no QPS file)")
            continue

        socp = qp_to_socp(qp)
        if socp is None:
            print(f"{name:<15} SKIP (not convertible to SOCP)")
            continue

        try:
            result = solve_with_clarabel(socp, tol=1e-8)
            status = result['status']
            iters = result['iterations']
            r_prim = result['r_prim'] if result['r_prim'] is not None else float('nan')
            r_dual = result['r_dual'] if result['r_dual'] is not None else float('nan')
            socp_name = f"{name}_SOCP"
            print(f"{socp_name:<15} {status:<20} {iters:>6} {r_prim:>12.2e} {r_dual:>12.2e}")
        except Exception as e:
            socp_name = f"{name}_SOCP"
            print(f"{socp_name:<15} ERROR: {e}")

    print("=" * 70)
    print("\nLegend:")
    print("  Solved = Optimal (met 1e-8 tolerance)")
    print("  AlmostSolved = AlmostOptimal (met reduced tolerance)")
    print("  MaxIterations = Failed to converge")


if __name__ == "__main__":
    main()
