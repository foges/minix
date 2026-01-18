#!/usr/bin/env python3
"""Benchmark SOCP timing: Minix vs Clarabel.

Runs each problem multiple times to get average solve times.
"""

import numpy as np
import clarabel
from scipy import sparse
import os
import time
import sys

# Add minix to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solver-py'))

try:
    import minix
    HAS_MINIX = True
except ImportError:
    HAS_MINIX = False
    print("Warning: minix not available, will only benchmark Clarabel")

# Problems to benchmark
BENCHMARK_PROBLEMS = [
    "HS21", "HS35",
    "DUALC1", "DUALC2", "DUALC5", "DUALC8",
    "PRIMALC5", "PRIMALC8",
    "QAFIRO", "QBRANDY",
]

NUM_RUNS = 5  # Number of runs for averaging


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
    P_data = []
    A_data = []

    var_names = {}
    con_names = {}
    obj_row = None

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
                    obj_row = row_name
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
    A = qp['A']
    q = qp['q']
    b = qp['b']
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
        'A': A,
        'q': q,
        'b': b,
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
        'r_prim': solution.r_prim,
        'r_dual': solution.r_dual,
    }


def solve_minix(socp, tol=1e-8):
    """Solve SOCP using Minix."""
    if not HAS_MINIX:
        return None

    L = socp['L']
    A_eq = socp['A']
    b_eq = socp['b']
    n = socp['n']
    m = socp['m']

    # Build the same SOCP formulation for Minix
    n_vars = n + 1
    q_obj = np.zeros(n_vars)
    q_obj[-1] = 1.0

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

    # Convert to minix format
    cones = [
        ("zero", m),
        ("nonneg", n),
        ("soc", soc_dim),
    ]

    start = time.perf_counter()
    try:
        result = minix.solve(
            q=q_obj,
            A=A_all,
            b=b_all,
            cones=cones,
            tol_gap_rel=tol,
            tol_primal=tol,
            tol_dual=tol,
            verbose=False,
        )
        elapsed = time.perf_counter() - start
        return {
            'status': result.status,
            'time_ms': elapsed * 1000,
            'iterations': result.iterations,
            'r_prim': result.primal_residual,
            'r_dual': result.dual_residual,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            'status': f'error: {e}',
            'time_ms': elapsed * 1000,
            'iterations': 0,
            'r_prim': float('nan'),
            'r_dual': float('nan'),
        }


def benchmark_problem(name, socp, num_runs=NUM_RUNS):
    """Benchmark a single problem."""
    clarabel_times = []
    clarabel_iters = []
    clarabel_status = None

    minix_times = []
    minix_iters = []
    minix_status = None

    # Warmup run
    solve_clarabel(socp)
    if HAS_MINIX:
        solve_minix(socp)

    # Benchmark Clarabel
    for _ in range(num_runs):
        result = solve_clarabel(socp)
        clarabel_times.append(result['time_ms'])
        clarabel_iters.append(result['iterations'])
        clarabel_status = result['status']

    # Benchmark Minix
    if HAS_MINIX:
        for _ in range(num_runs):
            result = solve_minix(socp)
            if result:
                minix_times.append(result['time_ms'])
                minix_iters.append(result['iterations'])
                minix_status = result['status']

    return {
        'clarabel': {
            'avg_time_ms': np.mean(clarabel_times),
            'std_time_ms': np.std(clarabel_times),
            'avg_iters': np.mean(clarabel_iters),
            'status': clarabel_status,
        },
        'minix': {
            'avg_time_ms': np.mean(minix_times) if minix_times else float('nan'),
            'std_time_ms': np.std(minix_times) if minix_times else float('nan'),
            'avg_iters': np.mean(minix_iters) if minix_iters else float('nan'),
            'status': minix_status,
        } if HAS_MINIX else None,
    }


def main():
    print(f"SOCP Timing Benchmark: Minix vs Clarabel ({NUM_RUNS} runs each)")
    print("=" * 100)
    header = f"{'Problem':<15} {'Clarabel':<30} {'Minix':<30} {'Speedup':>10}"
    print(header)
    print(f"{'':15} {'time(ms)':>10} {'±':>5} {'iters':>6} {'status':<8} {'time(ms)':>10} {'±':>5} {'iters':>6} {'status':<8}")
    print("-" * 100)

    results = []
    for name in BENCHMARK_PROBLEMS:
        qp = load_qps(name)
        if qp is None:
            print(f"{name}_SOCP:<15 SKIP (no QPS file)")
            continue

        socp = qp_to_socp(qp)
        if socp is None:
            print(f"{name}_SOCP:<15 SKIP (not convertible)")
            continue

        try:
            result = benchmark_problem(name, socp)
            c = result['clarabel']
            m = result['minix']

            c_status = 'OK' if 'Solved' in c['status'] else 'FAIL'

            if m:
                m_status = 'OK' if m['status'] == 'Optimal' else 'FAIL'
                speedup = c['avg_time_ms'] / m['avg_time_ms'] if m['avg_time_ms'] > 0 else float('nan')
                print(f"{name}_SOCP:<15 {c['avg_time_ms']:>10.2f} {c['std_time_ms']:>5.2f} {c['avg_iters']:>6.0f} {c_status:<8} "
                      f"{m['avg_time_ms']:>10.2f} {m['std_time_ms']:>5.2f} {m['avg_iters']:>6.0f} {m_status:<8} {speedup:>10.2f}x")
            else:
                print(f"{name}_SOCP:<15 {c['avg_time_ms']:>10.2f} {c['std_time_ms']:>5.2f} {c['avg_iters']:>6.0f} {c_status:<8} "
                      f"{'N/A':>10} {'':>5} {'':>6} {'':8} {'':>10}")

            results.append({'name': name, **result})
        except Exception as e:
            print(f"{name}_SOCP:<15 ERROR: {e}")

    print("=" * 100)
    print("\nSpeedup > 1 means Minix is faster, < 1 means Clarabel is faster")


if __name__ == "__main__":
    main()
