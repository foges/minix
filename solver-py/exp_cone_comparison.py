"""
Exponential cone solver comparison benchmark.

Compares Minix against ECOS, SCS, and Clarabel on exponential cone problems.
"""

import numpy as np
import time
import sys

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    print("WARNING: CVXPY not installed, skipping CVXPY tests")

try:
    import minix
    HAS_MINIX = True
except ImportError:
    HAS_MINIX = False
    print("ERROR: minix module not found. Run: maturin develop --release")
    sys.exit(1)


def format_time(seconds):
    """Format time in appropriate units."""
    if seconds < 1e-3:
        return f"{seconds*1e6:.1f} µs"
    elif seconds < 1:
        return f"{seconds*1e3:.2f} ms"
    else:
        return f"{seconds:.3f} s"


def run_cvxpy_problem(problem_fn, solver_name, **solver_opts):
    """Run a CVXPY problem with specified solver and return stats."""
    prob = problem_fn()

    try:
        start = time.perf_counter()
        prob.solve(solver=solver_name, verbose=False, **solver_opts)
        elapsed = time.perf_counter() - start

        return {
            'status': prob.status,
            'value': prob.value,
            'solve_time': elapsed,
            'iterations': prob.solver_stats.num_iters if hasattr(prob.solver_stats, 'num_iters') else None,
            'success': prob.status in ['optimal', 'optimal_inaccurate'],
        }
    except Exception as e:
        return {
            'status': f'ERROR: {str(e)}',
            'value': None,
            'solve_time': None,
            'iterations': None,
            'success': False,
        }


def entropy_maximization_cvxpy(n=10):
    """Maximize entropy: max ∑ x_i log(x_i) s.t. ∑ x_i = 1, x ≥ 0"""
    x = cp.Variable(n)
    objective = cp.Maximize(cp.sum(cp.entr(x)))
    constraints = [cp.sum(x) == 1, x >= 0]
    return cp.Problem(objective, constraints)


def kl_divergence_cvxpy(n=10):
    """Minimize KL divergence: min KL(x || p) s.t. ∑ x_i = 1, x ≥ 0"""
    x = cp.Variable(n)
    p = np.ones(n) / n  # Uniform distribution
    objective = cp.Minimize(cp.sum(cp.kl_div(x, p)))
    constraints = [cp.sum(x) == 1, x >= 0]
    return cp.Problem(objective, constraints)


def relative_entropy_cvxpy(n=10):
    """Relative entropy cone problem."""
    x = cp.Variable(n)
    y = cp.Variable(n)
    z = cp.Variable(1)

    # Minimize z subject to: sum of x_i * log(x_i / y_i) <= z
    objective = cp.Minimize(z)
    constraints = [
        cp.sum(cp.rel_entr(x, y)) <= z,
        cp.sum(x) == 1,
        cp.sum(y) == 1,
        x >= 0,
        y >= 0,
    ]
    return cp.Problem(objective, constraints)


def exp_cone_basic_cvxpy():
    """Basic exponential cone test: min x s.t. (x, 1, 1) in K_exp"""
    x = cp.Variable()
    # Using exponential cone: z >= y * exp(x/y) with y=1, z=1
    # This means: 1 >= exp(x), so x <= 0
    # Optimal: x = 0
    objective = cp.Minimize(x)
    constraints = [cp.constraints.ExpCone(x, 1.0, 1.0)]
    return cp.Problem(objective, constraints)


def portfolio_exp_utility_cvxpy(n=10):
    """Portfolio with exponential utility."""
    x = cp.Variable(n)
    returns = np.linspace(0.05, 0.15, n)
    lambda_risk = 0.5

    # Maximize: returns - λ * entropy
    objective = cp.Maximize(returns @ x - lambda_risk * cp.sum(cp.entr(x)))
    constraints = [cp.sum(x) == 1, x >= 0]
    return cp.Problem(objective, constraints)


def log_sum_exp_cvxpy(n=10):
    """Log-sum-exp constraint problem."""
    x = cp.Variable(n)
    a = np.random.randn(n)
    b = 2.0

    # Minimize sum(x) subject to log(sum(exp(a_i + x_i))) <= b
    objective = cp.Minimize(cp.sum(x))
    constraints = [cp.log_sum_exp(a + x) <= b, x >= 0]
    return cp.Problem(objective, constraints)


# Test problems with expected characteristics
PROBLEMS = [
    ('entropy_n5', lambda: entropy_maximization_cvxpy(5), 'Easy entropy, n=5'),
    ('entropy_n10', lambda: entropy_maximization_cvxpy(10), 'Medium entropy, n=10'),
    ('entropy_n20', lambda: entropy_maximization_cvxpy(20), 'Large entropy, n=20'),
    ('kl_div_n10', lambda: kl_divergence_cvxpy(10), 'KL divergence, n=10'),
    ('rel_entropy_n10', lambda: relative_entropy_cvxpy(10), 'Relative entropy, n=10'),
    ('exp_basic', exp_cone_basic_cvxpy, 'Basic exp cone'),
    ('portfolio_n10', lambda: portfolio_exp_utility_cvxpy(10), 'Portfolio utility, n=10'),
    ('logsumexp_n10', lambda: log_sum_exp_cvxpy(10), 'Log-sum-exp, n=10'),
]


def run_comparison():
    """Run full solver comparison."""

    if not HAS_CVXPY:
        print("CVXPY not available, cannot run comparison")
        return

    # Detect available solvers
    solvers = []

    # Check for ECOS (best for exp cones, reference implementation)
    try:
        cp.installed_solvers()
        if 'ECOS' in cp.installed_solvers():
            solvers.append(('ECOS', 'ECOS', {}))
    except:
        pass

    # Check for SCS (general purpose conic solver)
    try:
        if 'SCS' in cp.installed_solvers():
            solvers.append(('SCS', 'SCS', {'eps': 1e-8}))
    except:
        pass

    # Check for Clarabel (modern Rust solver)
    try:
        if 'CLARABEL' in cp.installed_solvers():
            solvers.append(('CLARABEL', 'CLARABEL', {}))
    except:
        pass

    if not solvers:
        print("ERROR: No CVXPY solvers available!")
        print("Install with: pip install cvxpy ecos scs clarabel")
        return

    print("\n" + "="*100)
    print("EXPONENTIAL CONE SOLVER COMPARISON")
    print("="*100)
    print(f"\nAvailable solvers: {', '.join([s[0] for s in solvers])}")
    print(f"Number of test problems: {len(PROBLEMS)}\n")

    # Run benchmarks
    results = {}

    for prob_name, prob_fn, description in PROBLEMS:
        print(f"\n--- {prob_name}: {description} ---")
        results[prob_name] = {}

        for solver_display, solver_name, opts in solvers:
            result = run_cvxpy_problem(prob_fn, solver_name, **opts)
            results[prob_name][solver_display] = result

            status_symbol = "✓" if result['success'] else "✗"
            time_str = format_time(result['solve_time']) if result['solve_time'] else "N/A"
            value_str = f"{result['value']:.6f}" if result['value'] is not None else "N/A"
            iters_str = str(result['iterations']) if result['iterations'] else "N/A"

            print(f"  {status_symbol} {solver_display:12s}: {time_str:>12s}  obj={value_str:>12s}  iters={iters_str:>5s}  [{result['status']}]")

    # Summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'Problem':<20}", end='')
    for solver_display, _, _ in solvers:
        print(f" | {solver_display:>12s}", end='')
    print()
    print("-"*100)

    for prob_name, _, description in PROBLEMS:
        print(f"{prob_name:<20}", end='')
        for solver_display, _, _ in solvers:
            res = results[prob_name][solver_display]
            if res['solve_time']:
                time_str = format_time(res['solve_time'])
                symbol = "✓" if res['success'] else "✗"
                print(f" | {symbol} {time_str:>10s}", end='')
            else:
                print(f" | {'FAIL':>12s}", end='')
        print()

    # Statistics
    print("\n" + "="*100)
    print("SOLVER STATISTICS")
    print("="*100)

    for solver_display, _, _ in solvers:
        success_count = sum(1 for p in PROBLEMS if results[p[0]][solver_display]['success'])
        total = len(PROBLEMS)
        avg_time = np.mean([results[p[0]][solver_display]['solve_time']
                           for p in PROBLEMS
                           if results[p[0]][solver_display]['solve_time'] is not None])

        print(f"\n{solver_display}:")
        print(f"  Success rate: {success_count}/{total} ({100*success_count/total:.1f}%)")
        print(f"  Avg solve time: {format_time(avg_time)}")

        if solver_display != 'MINIX':
            print(f"  Available via: CVXPY")

    print("\n" + "="*100)

    # Determine winner
    print("\n🏆 WINNER ANALYSIS:")

    for prob_name, _, description in PROBLEMS:
        print(f"\n  {prob_name}: {description}")
        valid_results = [(s, results[prob_name][s]['solve_time'])
                        for s, _, _ in solvers
                        if results[prob_name][s]['success'] and results[prob_name][s]['solve_time']]

        if valid_results:
            winner = min(valid_results, key=lambda x: x[1])
            print(f"    Fastest: {winner[0]} ({format_time(winner[1])})")

            # Show relative performance
            for solver_display, time in sorted(valid_results, key=lambda x: x[1]):
                ratio = time / winner[1]
                print(f"      {solver_display}: {format_time(time)} ({ratio:.2f}x)")
        else:
            print(f"    No solver succeeded!")

    print("\n" + "="*100 + "\n")


if __name__ == '__main__':
    run_comparison()
