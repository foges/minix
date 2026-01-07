#!/usr/bin/env python3
"""
Run Clarabel on all Maros-Meszaros problems and export results as JSON.

Usage:
    python benchmark_clarabel.py --export /tmp/clarabel.json --max-iter 50
"""

import argparse
import json
import time
import numpy as np
from scipy import sparse
import os
import sys

# Add solver-py to path for clarabel
sys.path.insert(0, '/Users/chris/code/minix/solver-py')

import clarabel

# List of all Maros-Meszaros problems (from compare_clarabel.py)
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

def parse_qps(path):
    """QPS parser from compare_clarabel.py"""
    with open(path) as f:
        lines = f.readlines()

    section = None
    rows = {}
    cols = {}
    A_data = []
    q_data = {}
    P_data = []
    rhs_data = {}
    bounds = {}
    row_idx = 0
    col_idx = 0
    obj_row = None

    for line in lines:
        line = line.rstrip()
        if not line or line.startswith('*'):
            continue

        if line.startswith('NAME'):
            continue
        if line.startswith('ROWS'):
            section = 'ROWS'
            continue
        if line.startswith('COLUMNS'):
            section = 'COLUMNS'
            continue
        if line.startswith('RHS'):
            section = 'RHS'
            continue
        if line.startswith('RANGES'):
            section = 'RANGES'
            continue
        if line.startswith('BOUNDS'):
            section = 'BOUNDS'
            continue
        if line.startswith('QUADOBJ'):
            section = 'QUADOBJ'
            continue
        if line.startswith('ENDATA'):
            break

        parts = line.split()

        if section == 'ROWS':
            rtype, rname = parts[0], parts[1]
            if rtype == 'N':
                obj_row = rname
            else:
                rows[rname] = (rtype, row_idx)
                row_idx += 1

        elif section == 'COLUMNS':
            cname = parts[0]
            if cname not in cols:
                cols[cname] = col_idx
                col_idx += 1
            ci = cols[cname]

            i = 1
            while i < len(parts):
                rname = parts[i]
                val = float(parts[i+1])
                if rname == obj_row:
                    q_data[ci] = val
                elif rname in rows:
                    ri = rows[rname][1]
                    A_data.append((ri, ci, val))
                i += 2

        elif section == 'RHS':
            i = 1
            while i < len(parts):
                rname = parts[i]
                val = float(parts[i+1])
                if rname in rows:
                    ri = rows[rname][1]
                    rhs_data[ri] = val
                i += 2

        elif section == 'BOUNDS':
            btype = parts[0]
            cname = parts[2]
            if cname not in cols:
                continue
            ci = cols[cname]
            if ci not in bounds:
                bounds[ci] = [0.0, np.inf]

            if btype == 'LO':
                bounds[ci][0] = float(parts[3])
            elif btype == 'UP':
                bounds[ci][1] = float(parts[3])
            elif btype == 'FX':
                bounds[ci] = [float(parts[3]), float(parts[3])]
            elif btype == 'FR':
                bounds[ci] = [-np.inf, np.inf]
            elif btype == 'MI':
                bounds[ci][0] = -np.inf
            elif btype == 'PL':
                bounds[ci][1] = np.inf

        elif section == 'QUADOBJ':
            c1, c2, val = parts[0], parts[1], float(parts[2])
            if c1 in cols and c2 in cols:
                i, j = cols[c1], cols[c2]
                P_data.append((i, j, val))
                if i != j:
                    P_data.append((j, i, val))

    n = col_idx
    m = row_idx

    q = np.zeros(n)
    for ci, val in q_data.items():
        q[ci] = val

    b = np.zeros(m)
    for ri, val in rhs_data.items():
        b[ri] = val

    row_types = ['E'] * m
    for rname, (rtype, ri) in rows.items():
        row_types[ri] = rtype

    if P_data:
        pi, pj, pv = zip(*P_data)
        P = sparse.csc_matrix((pv, (pi, pj)), shape=(n, n))
    else:
        P = sparse.csc_matrix((n, n))

    if A_data:
        ai, aj, av = zip(*A_data)
        A = sparse.csc_matrix((av, (ai, aj)), shape=(m, n))
    else:
        A = sparse.csc_matrix((m, n))

    return n, m, P, q, A, b, row_types, bounds


def solve_problem(name, max_iter=50, tol=1e-8):
    """Solve a single problem with Clarabel."""
    qps_path = os.path.expanduser(f"~/.cache/minix-bench/maros-meszaros/{name}.QPS")

    if not os.path.exists(qps_path):
        return {
            "name": name,
            "n": 0,
            "m": 0,
            "status": "NumericalError",
            "iterations": 0,
            "obj_val": float('nan'),
            "mu": float('nan'),
            "solve_time_ms": 0.0,
            "error": f"QPS file not found: {qps_path}"
        }

    try:
        # Parse QPS file
        n, m, P, q, A_raw, b_raw, row_types, var_bounds = parse_qps(qps_path)

        # Convert to Clarabel conic form
        eq_rows = [i for i, t in enumerate(row_types) if t == 'E']
        le_rows = [i for i, t in enumerate(row_types) if t == 'L']
        ge_rows = [i for i, t in enumerate(row_types) if t == 'G']

        A_blocks = []
        b_blocks = []
        cones = []

        if eq_rows:
            A_blocks.append(A_raw[eq_rows, :])
            b_blocks.append(b_raw[eq_rows])
            cones.append(clarabel.ZeroConeT(len(eq_rows)))

        if le_rows:
            A_blocks.append(-A_raw[le_rows, :])
            b_blocks.append(-b_raw[le_rows])
            cones.append(clarabel.NonnegativeConeT(len(le_rows)))

        if ge_rows:
            A_blocks.append(A_raw[ge_rows, :])
            b_blocks.append(b_raw[ge_rows])
            cones.append(clarabel.NonnegativeConeT(len(ge_rows)))

        # Add variable bounds
        for ci in range(n):
            lo, hi = var_bounds.get(ci, [0.0, np.inf])
            if np.isfinite(lo) and lo != 0:
                row = sparse.csc_matrix(([1.0], ([0], [ci])), shape=(1, n))
                A_blocks.append(row)
                b_blocks.append(np.array([lo]))
                cones.append(clarabel.NonnegativeConeT(1))
            elif lo == 0:
                row = sparse.csc_matrix(([-1.0], ([0], [ci])), shape=(1, n))
                A_blocks.append(row)
                b_blocks.append(np.array([0.0]))
                cones.append(clarabel.NonnegativeConeT(1))

            if np.isfinite(hi):
                row = sparse.csc_matrix(([-1.0], ([0], [ci])), shape=(1, n))
                A_blocks.append(row)
                b_blocks.append(np.array([-hi]))
                cones.append(clarabel.NonnegativeConeT(1))

        A = sparse.vstack(A_blocks).tocsc()
        b = np.concatenate(b_blocks)

        # Solve
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = max_iter
        settings.tol_feas = tol
        settings.tol_gap_abs = tol
        settings.tol_gap_rel = tol

        start = time.time()
        solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
        result = solver.solve()
        solve_time_ms = (time.time() - start) * 1000.0

        # Map status
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
            "name": name,
            "n": n,
            "m": m,
            "status": status,
            "iterations": result.iterations,
            "obj_val": result.obj_val if result.x is not None else float('nan'),
            "mu": 0.0,
            "solve_time_ms": solve_time_ms,
        }

    except Exception as e:
        return {
            "name": name,
            "n": 0,
            "m": 0,
            "status": "NumericalError",
            "iterations": 0,
            "obj_val": float('nan'),
            "mu": float('nan'),
            "solve_time_ms": 0.0,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Run Clarabel on Maros-Meszaros benchmark")
    parser.add_argument("--export", required=True, help="Path to export JSON results")
    parser.add_argument("--limit", type=int, help="Limit number of problems")
    parser.add_argument("--max-iter", type=int, default=50, help="Max iterations")
    parser.add_argument("--tol", type=float, default=1e-8, help="Tolerance")
    args = parser.parse_args()

    problems = MM_PROBLEMS[:args.limit] if args.limit else MM_PROBLEMS
    results = []

    print(f"Running Clarabel on {len(problems)} problems...")
    print("=" * 60)

    for i, name in enumerate(problems, 1):
        print(f"[{i}/{len(problems)}] {name:15s} ... ", end='', flush=True)
        result = solve_problem(name, args.max_iter, args.tol)

        status_symbols = {
            "Optimal": "✓",
            "AlmostOptimal": "~",
            "MaxIters": "M",
            "PrimalInfeasible": "P",
            "DualInfeasible": "D",
            "NumericalError": "N"
        }
        symbol = status_symbols.get(result["status"], "?")

        if result.get("error"):
            print(f"ERROR: {result['error']}")
        else:
            print(f"{symbol} ({result['iterations']:2d} iters, {result['solve_time_ms']:6.1f}ms)")

        results.append(result)

    print("=" * 60)

    # Compute summary
    total = len(results)
    optimal = sum(1 for r in results if r["status"] == "Optimal")
    almost = sum(1 for r in results if r["status"] == "AlmostOptimal")

    # Geometric mean for solved problems
    times = [r["solve_time_ms"] for r in results if r["status"] in ["Optimal", "AlmostOptimal"]]
    geom_mean_time = np.exp(np.mean(np.log([t + 1.0 for t in times]))) - 1.0 if times else 0.0

    summary = {
        "total": total,
        "optimal": optimal,
        "almost_optimal": almost,
        "geom_mean_time_ms": geom_mean_time
    }

    # Export
    output = {
        "solver_name": "Clarabel",
        "results": results,
        "summary": summary
    }

    with open(args.export, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults exported to: {args.export}")
    print(f"Pass rate: {optimal} + {almost} = {optimal + almost}/{total} ({100.0*(optimal+almost)/total:.1f}%)")
    print(f"Geometric mean time: {geom_mean_time:.2f}ms")


if __name__ == "__main__":
    main()
