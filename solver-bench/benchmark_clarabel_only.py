#!/usr/bin/env python3
"""Benchmark Clarabel SOCP timing only."""

import numpy as np
import clarabel
from scipy import sparse
import os
import time

# Problems to benchmark
BENCHMARK_PROBLEMS = [
    "HS21", "HS35",
    "DUALC1", "DUALC2", "DUALC5", "DUALC8",
    "PRIMALC5", "PRIMALC8",
    "QAFIRO",
]

NUM_RUNS = 5


def load_qps(name: str):
    """Load a QPS file from the cache."""
    home = os.path.expanduser("~")
    cache_dir = os.path.join(home, ".cache", "minix-bench", "maros-meszaros")
    path = os.path.join(cache_dir, f"{name}.QPS")
    if not os.path.exists(path):
        return None

    n = 0
    m = 0
    P_data = []
    A_data = []
    var_names = {}
    con_names = {}
    b = []

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
                if row_type != 'N':
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

    q = np.zeros(n)
    b = np.array(b + [0.0] * (m - len(b)))

    return {'P': P, 'A': A, 'q': q, 'b': b, 'n': n, 'm': m}


def qp_to_socp(qp):
    """Convert QP to SOCP form."""
    P = qp['P']
    n = qp['n']
    m = qp['m']

    if P.nnz == 0:
        return None

    try:
        P_dense = P.toarray()
        P_sym = 0.5 * (P_dense + P_dense.T)
        P_reg = P_sym + np.eye(n) * 1e-10
        L = np.linalg.cholesky(P_reg)
    except np.linalg.LinAlgError:
        return None

    return {
        'L': L,
        'A': qp['A'],
        'q': qp['q'],
        'b': qp['b'],
        'n': n,
        'm': m,
    }


def solve_clarabel(socp, tol=1e-8):
    """Solve SOCP using Clarabel."""
    L = socp['L']
    A_eq = socp['A']
    b_eq = socp['b']
    n = socp['n']
    m = socp['m']

    n_vars = n + 1
    q_obj = np.zeros(n_vars)
    q_obj[-1] = 1.0
    P_obj = sparse.csc_matrix((n_vars, n_vars))

    soc_dim = 1 + n
    total_constraints = m + n + soc_dim

    A_rows, A_cols, A_vals = [], [], []

    A_eq_coo = sparse.coo_matrix(A_eq)
    for i, j, v in zip(A_eq_coo.row, A_eq_coo.col, A_eq_coo.data):
        A_rows.append(i)
        A_cols.append(j)
        A_vals.append(v)

    for i in range(n):
        A_rows.append(m + i)
        A_cols.append(i)
        A_vals.append(-1.0)

    A_rows.append(m + n)
    A_cols.append(n)
    A_vals.append(-1.0)

    for i in range(n):
        for j in range(n):
            if abs(L[i, j]) > 1e-14:
                A_rows.append(m + n + 1 + i)
                A_cols.append(j)
                A_vals.append(-L[i, j])

    A_all = sparse.csc_matrix((A_vals, (A_rows, A_cols)), shape=(total_constraints, n_vars))
    b_all = np.zeros(total_constraints)
    b_all[:m] = b_eq

    cones = [
        clarabel.ZeroConeT(m),
        clarabel.NonnegativeConeT(n),
        clarabel.SecondOrderConeT(soc_dim),
    ]

    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_feas = tol
    settings.tol_gap_abs = tol
    settings.tol_gap_rel = tol

    start = time.perf_counter()
    solver = clarabel.DefaultSolver(P_obj, q_obj, A_all, b_all, cones, settings)
    solution = solver.solve()
    elapsed = time.perf_counter() - start

    return {
        'status': str(solution.status),
        'time_ms': elapsed * 1000,
        'iterations': solution.iterations,
    }


def main():
    print(f"Clarabel SOCP Benchmark ({NUM_RUNS} runs each)")
    print("=" * 60)
    print(f"{'Problem':<15} {'Avg Time(ms)':>12} {'Std':>8} {'Iters':>8} {'Status':<12}")
    print("-" * 60)

    for name in BENCHMARK_PROBLEMS:
        qp = load_qps(name)
        if qp is None:
            print(f"{name}_SOCP:<15 SKIP (no QPS file)")
            continue

        socp = qp_to_socp(qp)
        if socp is None:
            print(f"{name}_SOCP:<15 SKIP (not convertible)")
            continue

        # Warmup
        solve_clarabel(socp)

        times = []
        iters = []
        status = None
        for _ in range(NUM_RUNS):
            result = solve_clarabel(socp)
            times.append(result['time_ms'])
            iters.append(result['iterations'])
            status = result['status']

        avg_time = np.mean(times)
        std_time = np.std(times)
        avg_iters = np.mean(iters)
        status_short = 'OK' if 'Solved' in status else 'FAIL'

        print(f"{name}_SOCP:<15 {avg_time:>12.3f} {std_time:>8.3f} {avg_iters:>8.0f} {status_short:<12}")

    print("=" * 60)


if __name__ == "__main__":
    main()
