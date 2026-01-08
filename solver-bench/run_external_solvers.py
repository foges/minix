#!/usr/bin/env python3
"""
Run external solvers on Maros-Meszaros QP problems and export results as JSON.

This script runs multiple QP solvers (Clarabel, OSQP, etc.) on the Maros-Meszaros
benchmark suite and exports results in the same JSON format as the Rust benchmarking tool,
enabling direct comparison via the `compare` command.

Usage:
    python run_external_solvers.py --solver clarabel --limit 10 --export results/clarabel.json
    python run_external_solvers.py --solver osqp --export results/osqp.json

    # Then compare:
    cargo run --release -p solver-bench -- compare results/minix.json results/clarabel.json results/osqp.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

# Try importing solvers (gracefully handle missing ones)
try:
    import clarabel
    HAS_CLARABEL = True
except ImportError:
    HAS_CLARABEL = False
    print("Warning: clarabel not installed (pip install clarabel)", file=sys.stderr)

try:
    import osqp
    HAS_OSQP = True
except ImportError:
    HAS_OSQP = False
    print("Warning: osqp not installed (pip install osqp)", file=sys.stderr)

try:
    import scipy.sparse as sp
    import numpy as np
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Error: scipy required (pip install scipy)", file=sys.stderr)
    sys.exit(1)

try:
    from qpsolvers_benchmark.maros_meszaros import get_problem
    HAS_QPSOLVERS = True
except ImportError:
    HAS_QPSOLVERS = False
    print("Warning: qpsolvers_benchmark not installed (pip install qpsolvers_benchmark)", file=sys.stderr)


# List of all Maros-Meszaros problems (138 total)
MM_PROBLEMS = [
    "AUG2D", "AUG2DC", "AUG2DCQP", "AUG2DQP", "AUG3D", "AUG3DC", "AUG3DCQP", "AUG3DQP",
    "BOYD1", "BOYD2", "CONT-050", "CONT-100", "CONT-101", "CONT-200", "CONT-201", "CONT-300",
    "CVXQP1_L", "CVXQP1_M", "CVXQP1_S", "CVXQP2_L", "CVXQP2_M", "CVXQP2_S", "CVXQP3_L",
    "CVXQP3_M", "CVXQP3_S", "DPKLO1", "DTOC3", "DUAL1", "DUAL2", "DUAL3", "DUAL4", "DUALC1",
    "DUALC2", "DUALC5", "DUALC8", "EXDATA", "GOULDQP2", "GOULDQP3", "HS118", "HS21", "HS268",
    "HS35", "HS35MOD", "HS51", "HS52", "HS53", "HS76", "HUES-MOD", "HUESTIS", "KSIP",
    "LASER", "LISWET1", "LISWET10", "LISWET11", "LISWET12", "LISWET2", "LISWET3", "LISWET4",
    "LISWET5", "LISWET6", "LISWET7", "LISWET8", "LISWET9", "LOTSCHD", "MOSARQP1", "MOSARQP2",
    "POWELL20", "PRIMAL1", "PRIMAL2", "PRIMAL3", "PRIMAL4", "PRIMALC1", "PRIMALC2", "PRIMALC5",
    "PRIMALC8", "Q25FV47", "QADLITTL", "QAFIRO", "QBANDM", "QBEACONF", "QBORE3D", "QBRANDY",
    "QCAPRI", "QE226", "QETAMACR", "QFFFFF80", "QFORPLAN", "QGFRDXPN", "QGROW15", "QGROW22",
    "QGROW7", "QISRAEL", "QPCBLEND", "QPCBOEI1", "QPCBOEI2", "QPCSTAIR", "QPILOTNO", "QRECIPE",
    "QSC205", "QSCAGR25", "QSCAGR7", "QSCFXM1", "QSCFXM2", "QSCFXM3", "QSCORPIO", "QSCRS8",
    "QSCSD1", "QSCSD6", "QSCSD8", "QSCTAP1", "QSCTAP2", "QSCTAP3", "QSEBA", "QSHARE1B",
    "QSHARE2B", "QSHELL", "QSHIP04L", "QSHIP04S", "QSHIP08L", "QSHIP08S", "QSHIP12L", "QSHIP12S",
    "QSIERRA", "QSTAIR", "QSTANDAT", "S268", "STADAT1", "STADAT2", "STADAT3", "STCQP1",
    "STCQP2", "TAME", "UBH1", "VALUES", "YAO", "ZECEVIC2",
]


def load_problem(name: str) -> Optional[Dict[str, Any]]:
    """Load a Maros-Meszaros problem using qpsolvers_benchmark."""
    if not HAS_QPSOLVERS:
        return None

    try:
        prob = get_problem(name)
        return {
            "P": prob.P,
            "q": prob.q,
            "A": prob.A if prob.A is not None else sp.csr_matrix((0, prob.P.shape[0])),
            "b": prob.b if prob.b is not None else np.array([]),
            "G": prob.G if prob.G is not None else sp.csr_matrix((0, prob.P.shape[0])),
            "h": prob.h if prob.h is not None else np.array([]),
            "lb": prob.lb if prob.lb is not None else None,
            "ub": prob.ub if prob.ub is not None else None,
            "n": prob.P.shape[0],
            "m": (prob.A.shape[0] if prob.A is not None else 0) + (prob.G.shape[0] if prob.G is not None else 0)
        }
    except Exception as e:
        print(f"Error loading {name}: {e}", file=sys.stderr)
        return None


def solve_with_clarabel(prob_data: Dict, max_iter: int = 50, tol: float = 1e-8) -> Dict:
    """Solve problem with Clarabel."""
    if not HAS_CLARABEL:
        return {
            "status": "NumericalError",
            "iterations": 0,
            "obj_val": float('nan'),
            "mu": float('nan'),
            "solve_time_ms": 0.0,
            "error": "Clarabel not installed"
        }

    try:
        P = prob_data["P"]
        q = prob_data["q"]
        A_eq = prob_data["A"]
        b_eq = prob_data["b"]
        G = prob_data["G"]
        h = prob_data["h"]
        lb = prob_data.get("lb")
        ub = prob_data.get("ub")

        # Convert to Clarabel's conic form
        # Build constraint matrix A and cones
        constraints = []
        n = P.shape[0]

        # Equality constraints: A_eq x = b_eq
        if A_eq.shape[0] > 0:
            constraints.append((A_eq, b_eq, clarabel.ZeroConeT(A_eq.shape[0])))

        # Inequality constraints: G x <= h  =>  Gx + s = h, s >= 0
        if G.shape[0] > 0:
            constraints.append((G, h, clarabel.NonnegativeConeT(G.shape[0])))

        # Box constraints on x
        if lb is not None:
            # x >= lb  =>  -x + s = -lb, s >= 0
            I = sp.eye(n, format='csr')
            constraints.append((-I, -lb, clarabel.NonnegativeConeT(n)))
        if ub is not None:
            # x <= ub  =>  x + s = ub, s >= 0
            I = sp.eye(n, format='csr')
            constraints.append((I, ub, clarabel.NonnegativeConeT(n)))

        # Stack all constraints
        if constraints:
            A_stack = sp.vstack([c[0] for c in constraints], format='csc')
            b_stack = np.concatenate([c[1] for c in constraints])
            cones = [c[2] for c in constraints]
        else:
            A_stack = sp.csc_matrix((0, n))
            b_stack = np.array([])
            cones = []

        # Solve
        settings = clarabel.DefaultSettings()
        settings.max_iter = max_iter
        settings.tol_feas = tol
        settings.tol_gap_abs = tol
        settings.tol_gap_rel = tol

        start = time.time()
        solver = clarabel.DefaultSolver(P, q, A_stack, b_stack, cones, settings)
        result = solver.solve()
        solve_time_ms = (time.time() - start) * 1000.0

        # Map Clarabel status to our status
        status_map = {
            clarabel.SolverStatus.Solved: "Optimal",
            clarabel.SolverStatus.PrimalInfeasible: "PrimalInfeasible",
            clarabel.SolverStatus.DualInfeasible: "DualInfeasible",
            clarabel.SolverStatus.AlmostSolved: "AlmostOptimal",
            clarabel.SolverStatus.MaxIterations: "MaxIters",
            clarabel.SolverStatus.MaxTime: "MaxIters",
        }
        status = status_map.get(result.status, "NumericalError")

        return {
            "status": status,
            "iterations": result.iterations,
            "obj_val": result.obj_val if result.x is not None else float('nan'),
            "mu": 0.0,  # Clarabel doesn't expose mu
            "solve_time_ms": solve_time_ms,
        }
    except Exception as e:
        return {
            "status": "NumericalError",
            "iterations": 0,
            "obj_val": float('nan'),
            "mu": float('nan'),
            "solve_time_ms": 0.0,
            "error": str(e)
        }


def solve_with_osqp(prob_data: Dict, max_iter: int = 50, tol: float = 1e-8) -> Dict:
    """Solve problem with OSQP."""
    if not HAS_OSQP:
        return {
            "status": "NumericalError",
            "iterations": 0,
            "obj_val": float('nan'),
            "mu": float('nan'),
            "solve_time_ms": 0.0,
            "error": "OSQP not installed"
        }

    try:
        P = prob_data["P"]
        q = prob_data["q"]
        A_eq = prob_data["A"]
        b_eq = prob_data["b"]
        G = prob_data["G"]
        h = prob_data["h"]
        lb = prob_data.get("lb")
        ub = prob_data.get("ub")

        # OSQP form: minimize (1/2)x'Px + q'x subject to l <= Ax <= u
        n = P.shape[0]

        # Stack all constraints
        constraints = []
        lower = []
        upper = []

        # Equality constraints: A_eq x = b_eq
        if A_eq.shape[0] > 0:
            constraints.append(A_eq)
            lower.append(b_eq)
            upper.append(b_eq)

        # Inequality constraints: G x <= h
        if G.shape[0] > 0:
            constraints.append(G)
            lower.append(np.full(G.shape[0], -np.inf))
            upper.append(h)

        # Box constraints
        if lb is not None or ub is not None:
            I = sp.eye(n, format='csr')
            constraints.append(I)
            lower.append(lb if lb is not None else np.full(n, -np.inf))
            upper.append(ub if ub is not None else np.full(n, np.inf))

        if constraints:
            A_osqp = sp.vstack(constraints, format='csc')
            l_osqp = np.concatenate(lower)
            u_osqp = np.concatenate(upper)
        else:
            A_osqp = sp.csc_matrix((0, n))
            l_osqp = np.array([])
            u_osqp = np.array([])

        # Solve
        solver = osqp.OSQP()
        solver.setup(P=P, q=q, A=A_osqp, l=l_osqp, u=u_osqp,
                     max_iter=max_iter, eps_abs=tol, eps_rel=tol,
                     verbose=False, polish=True)

        start = time.time()
        result = solver.solve()
        solve_time_ms = (time.time() - start) * 1000.0

        # Map OSQP status to our status
        status_map = {
            1: "Optimal",  # solved
            2: "PrimalInfeasible",  # primal infeasible
            3: "DualInfeasible",  # dual infeasible
            4: "MaxIters",  # max iterations reached
            -2: "MaxIters",  # max time reached
            -3: "PrimalInfeasible",  # primal infeasible (inaccurate)
            -4: "DualInfeasible",  # dual infeasible (inaccurate)
        }
        status = status_map.get(result.info.status_val, "NumericalError")

        return {
            "status": status,
            "iterations": result.info.iter,
            "obj_val": result.info.obj_val if result.x is not None else float('nan'),
            "mu": 0.0,  # OSQP doesn't expose mu
            "solve_time_ms": solve_time_ms,
        }
    except Exception as e:
        return {
            "status": "NumericalError",
            "iterations": 0,
            "obj_val": float('nan'),
            "mu": float('nan'),
            "solve_time_ms": 0.0,
            "error": str(e)
        }


def run_single_problem(name: str, solver: str, max_iter: int = 50, tol: float = 1e-8) -> Dict:
    """
    Run a single problem with the specified solver.

    Returns a result dict matching the Rust BenchmarkResult format.
    """
    # Load problem
    prob_data = load_problem(name)
    if prob_data is None:
        return {
            "name": name,
            "n": 0,
            "m": 0,
            "status": "NumericalError",
            "iterations": 0,
            "obj_val": float('nan'),
            "mu": float('nan'),
            "solve_time_ms": 0.0,
            "error": f"Failed to load problem {name}"
        }

    # Solve with chosen solver
    if solver == "clarabel":
        result = solve_with_clarabel(prob_data, max_iter, tol)
    elif solver == "osqp":
        result = solve_with_osqp(prob_data, max_iter, tol)
    else:
        result = {
            "status": "NumericalError",
            "iterations": 0,
            "obj_val": float('nan'),
            "mu": float('nan'),
            "solve_time_ms": 0.0,
            "error": f"Unknown solver: {solver}"
        }

    # Add problem metadata
    result["name"] = name
    result["n"] = prob_data.get("n", 0)
    result["m"] = prob_data.get("m", 0)

    return result


def compute_summary(results: List[Dict]) -> Dict:
    """Compute summary statistics matching Rust BenchmarkSummary format."""
    total = len(results)
    optimal = sum(1 for r in results if r["status"] == "Optimal")
    almost_optimal = sum(1 for r in results if r["status"] == "AlmostOptimal")
    max_iters = sum(1 for r in results if r["status"] == "MaxIters")
    numerical_errors = sum(1 for r in results if r.get("error") is not None)
    parse_errors = sum(1 for r in results if r.get("error", "").startswith("Parse"))

    total_time_s = sum(r["solve_time_ms"] / 1000.0 for r in results)

    # Geometric mean of iterations (for solved problems)
    iters = [r["iterations"] for r in results
             if r["status"] in ["Optimal", "AlmostOptimal"] and r["iterations"] > 0]
    geom_mean_iters = np.exp(np.mean(np.log(iters))) if iters else 0.0

    # Shifted geometric mean of solve times
    times = [r["solve_time_ms"] for r in results
             if r["status"] in ["Optimal", "AlmostOptimal"] and r["solve_time_ms"] > 0]
    geom_mean_time_ms = np.exp(np.mean(np.log([t + 1.0 for t in times]))) - 1.0 if times else 0.0

    return {
        "total": total,
        "optimal": optimal,
        "almost_optimal": almost_optimal,
        "max_iters": max_iters,
        "numerical_errors": numerical_errors,
        "parse_errors": parse_errors,
        "total_time_s": total_time_s,
        "geom_mean_iters": geom_mean_iters,
        "geom_mean_time_ms": geom_mean_time_ms
    }


def run_benchmark(solver: str, limit: Optional[int] = None, max_iter: int = 50) -> List[Dict]:
    """Run full benchmark suite with the specified solver."""
    problems = MM_PROBLEMS[:limit] if limit else MM_PROBLEMS
    results = []

    print(f"Running {solver} on {len(problems)} problems...")
    print("=" * 60)

    for i, name in enumerate(problems, 1):
        print(f"[{i}/{len(problems)}] {name} ... ", end='', flush=True)

        result = run_single_problem(name, solver, max_iter)

        status_str = {
            "Optimal": "✓",
            "AlmostOptimal": "~",
            "MaxIters": "M",
            "NumericalError": "N"
        }.get(result["status"], "?")

        if result.get("error"):
            print(f"ERROR: {result['error']}")
        else:
            print(f"{status_str} ({result['iterations']} iters, {result['solve_time_ms']:.1f}ms)")

        results.append(result)

    print("=" * 60)
    return results


def export_results(solver_name: str, results: List[Dict], path: str):
    """Export results to JSON file."""
    summary = compute_summary(results)

    output = {
        "solver_name": solver_name,
        "results": results,
        "summary": summary
    }

    with open(path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults exported to: {path}")
    print(f"Solved: {summary['optimal']} optimal + {summary['almost_optimal']} almost = "
          f"{summary['optimal'] + summary['almost_optimal']}/{summary['total']} "
          f"({100.0 * (summary['optimal'] + summary['almost_optimal']) / summary['total']:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Run external solvers on Maros-Meszaros QP benchmark"
    )
    parser.add_argument(
        "--solver",
        choices=["clarabel", "osqp"],
        required=True,
        help="Solver to use"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of problems to run (default: all 138)"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=50,
        help="Maximum iterations per problem (default: 50)"
    )
    parser.add_argument(
        "--export",
        required=True,
        help="Path to export JSON results"
    )
    parser.add_argument(
        "--solver-name",
        help="Custom solver name for export (default: based on solver choice)"
    )

    args = parser.parse_args()

    # Check dependencies
    if args.solver == "clarabel" and not HAS_CLARABEL:
        print("Error: clarabel not installed. Install with: pip install clarabel")
        sys.exit(1)

    if args.solver == "osqp" and not HAS_OSQP:
        print("Error: osqp not installed. Install with: pip install osqp")
        sys.exit(1)

    # Print warning about implementation status
    print("=" * 60)
    print("WARNING: This is a stub implementation!")
    print("=" * 60)
    print()
    print("To fully implement:")
    print("1. Add proper QPS parser (or use qpsolvers package)")
    print("2. Convert QP format to solver-specific formats")
    print("3. Map solver statuses to our standard statuses")
    print()
    print("For now, this script shows the framework and data format.")
    print("You can manually create JSON files for external solvers.")
    print()
    print("Suggested approach:")
    print("  pip install qpsolvers clarabel osqp")
    print("  Use qpsolvers.solve_qp() which handles format conversion")
    print("=" * 60)
    print()

    # Run benchmark
    solver_name = args.solver_name or args.solver.title()
    results = run_benchmark(args.solver, args.limit, args.max_iter)

    # Export results
    export_results(solver_name, results, args.export)


if __name__ == "__main__":
    main()
