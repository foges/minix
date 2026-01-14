#!/usr/bin/env python3
"""Benchmark Clarabel vs Minix via CVXPY."""

import time
import numpy as np
import cvxpy as cp
from scipy import sparse

# Register Minix with CVXPY
from minix_cvxpy import register_minix, MINIX
register_minix()

def benchmark_qp(n, name="QP"):
    """Standard QP: min (1/2) x'Px + q'x s.t. Ax <= b, x >= 0"""
    np.random.seed(42)

    # Generate random PSD P
    L = np.random.randn(n, n) * 0.1
    P = L @ L.T + np.eye(n) * 0.1
    q = np.random.randn(n)

    # Constraints: Ax <= b
    m = n // 2
    A = np.random.randn(m, n)
    b = np.abs(np.random.randn(m)) + 1.0

    x = cp.Variable(n)
    objective = 0.5 * cp.quad_form(x, P) + q @ x
    constraints = [A @ x <= b, x >= 0]
    prob = cp.Problem(cp.Minimize(objective), constraints)

    return prob, name

def benchmark_socp(n, name="SOCP"):
    """SOCP: min c'x s.t. ||Ax + b|| <= c'x + d"""
    np.random.seed(42)

    c = np.random.randn(n)
    A = np.random.randn(n, n) * 0.5
    b = np.random.randn(n)
    d_vec = np.random.randn(n)
    d = 1.0

    x = cp.Variable(n)
    objective = c @ x
    constraints = [cp.SOC(d_vec @ x + d, A @ x + b)]
    prob = cp.Problem(cp.Minimize(objective), constraints)

    return prob, name

def benchmark_qp_with_soc(n, name="QP+SOC"):
    """QP with SOC constraint: min (1/2)||x||^2 + q'x s.t. ||x|| <= r"""
    np.random.seed(42)

    q = -np.random.rand(n)  # Push toward boundary
    r = np.sqrt(n) * 0.5    # Norm bound

    x = cp.Variable(n)
    objective = 0.5 * cp.sum_squares(x) + q @ x
    constraints = [cp.norm(x) <= r]
    prob = cp.Problem(cp.Minimize(objective), constraints)

    return prob, name

def benchmark_portfolio(n, name="Portfolio"):
    """Portfolio optimization: min risk - lambda * return s.t. sum(x) = 1, x >= 0"""
    np.random.seed(42)

    # Generate covariance matrix
    F = np.random.randn(n, n // 2) * 0.1
    Sigma = F @ F.T + np.diag(np.random.rand(n) * 0.1)
    mu = np.random.randn(n) * 0.1

    risk_aversion = 1.0

    x = cp.Variable(n)
    risk = cp.quad_form(x, Sigma)
    ret = mu @ x
    objective = risk - risk_aversion * ret
    constraints = [cp.sum(x) == 1, x >= 0]
    prob = cp.Problem(cp.Minimize(objective), constraints)

    return prob, name

def benchmark_lasso(n, m, name="LASSO"):
    """LASSO regression: min ||Ax - b||^2 + lambda * ||x||_1"""
    np.random.seed(42)

    A = np.random.randn(m, n)
    x_true = np.zeros(n)
    x_true[:n//10] = np.random.randn(n//10)  # Sparse solution
    b = A @ x_true + np.random.randn(m) * 0.1

    lam = 0.1

    x = cp.Variable(n)
    objective = cp.sum_squares(A @ x - b) + lam * cp.norm1(x)
    prob = cp.Problem(cp.Minimize(objective), [])

    return prob, name

def run_benchmark(prob, name, solvers):
    """Run benchmark on a problem with multiple solvers."""
    results = {}

    for solver_name, solver in solvers.items():
        try:
            start = time.perf_counter()
            prob.solve(solver=solver, verbose=False)
            elapsed = (time.perf_counter() - start) * 1000

            results[solver_name] = {
                'time_ms': elapsed,
                'status': prob.status,
                'obj': prob.value if prob.value is not None else float('nan'),
                'error': None
            }
        except Exception as e:
            results[solver_name] = {
                'time_ms': float('nan'),
                'status': 'error',
                'obj': float('nan'),
                'error': str(e)
            }

    return results

def main():
    solvers = {
        'Clarabel': cp.CLARABEL,
        'Minix': cp.MINIX,
    }

    print("=" * 70)
    print("CVXPY Benchmark: Clarabel vs Minix")
    print("=" * 70)

    # Define test problems
    problems = [
        benchmark_qp(50, "QP-50"),
        benchmark_qp(100, "QP-100"),
        benchmark_qp(200, "QP-200"),
        benchmark_qp(500, "QP-500"),
        benchmark_socp(50, "SOCP-50"),
        benchmark_socp(100, "SOCP-100"),
        benchmark_socp(200, "SOCP-200"),
        benchmark_qp_with_soc(50, "QP+SOC-50"),
        benchmark_qp_with_soc(100, "QP+SOC-100"),
        benchmark_qp_with_soc(200, "QP+SOC-200"),
        benchmark_portfolio(50, "Portfolio-50"),
        benchmark_portfolio(100, "Portfolio-100"),
        benchmark_portfolio(200, "Portfolio-200"),
        benchmark_lasso(100, 50, "LASSO-100x50"),
        benchmark_lasso(200, 100, "LASSO-200x100"),
        benchmark_lasso(500, 250, "LASSO-500x250"),
    ]

    print(f"\n{'Problem':<20} {'Solver':<12} {'Time (ms)':<12} {'Status':<15} {'Objective':<15}")
    print("-" * 70)

    all_results = []

    for prob, name in problems:
        results = run_benchmark(prob, name, solvers)
        all_results.append((name, results))

        for solver_name, res in results.items():
            status_str = res['status'][:12] if res['status'] else 'None'
            obj_str = f"{res['obj']:.6e}" if not np.isnan(res['obj']) else 'N/A'
            time_str = f"{res['time_ms']:.2f}" if not np.isnan(res['time_ms']) else 'N/A'
            print(f"{name:<20} {solver_name:<12} {time_str:<12} {status_str:<15} {obj_str:<15}")
            if res['error']:
                print(f"  Error: {res['error']}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    clarabel_times = []
    minix_times = []

    for name, results in all_results:
        if 'Clarabel' in results and 'Minix' in results:
            ct = results['Clarabel']['time_ms']
            mt = results['Minix']['time_ms']
            if not np.isnan(ct) and not np.isnan(mt):
                clarabel_times.append(ct)
                minix_times.append(mt)
                print(f"{name}: Clarabel={ct:.1f}ms, Minix={mt:.1f}ms, ratio={mt/ct:.2f}x")

    if clarabel_times and minix_times:
        clarabel_total = sum(clarabel_times)
        minix_total = sum(minix_times)
        ratio = minix_total / clarabel_total if clarabel_total > 0 else float('nan')

        # Geometric mean
        clarabel_geom = np.exp(np.mean(np.log(clarabel_times)))
        minix_geom = np.exp(np.mean(np.log(minix_times)))

        print(f"\nTotals:")
        print(f"  Clarabel: {clarabel_total:.2f}ms (geom mean: {clarabel_geom:.2f}ms)")
        print(f"  Minix:    {minix_total:.2f}ms (geom mean: {minix_geom:.2f}ms)")
        print(f"  Ratio (Minix/Clarabel): {ratio:.2f}x")

if __name__ == "__main__":
    main()
