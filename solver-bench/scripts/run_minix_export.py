#!/usr/bin/env python3
"""
Run Minix via subprocess and parse output to JSON format.
"""

import subprocess
import re
import json
import sys
import os

# List of all problems
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


def run_minix_problem(name, tol=1e-9, max_iter=50):
    """Run Minix on a single problem."""
    env = os.environ.copy()
    env['MINIX_TOL_FEAS'] = str(tol)
    env['MINIX_TOL_GAP'] = str(tol)

    result = subprocess.run(
        ['cargo', 'run', '--release', '-p', 'solver-bench', '--',
         'maros-meszaros', '--problem', name, '--max-iter', str(max_iter)],
        capture_output=True,
        text=True,
        env=env,
        cwd='/Users/chris/code/minix'
    )

    output = result.stdout + result.stderr

    # Parse output
    # Look for lines like: [1/136] AUG2D ... ✓ (9 iters, 0.2ms, obj=1.511e1)
    match = re.search(rf'{re.escape(name)}\s+\.\.\.\s+([✓~MN])\s+\((\d+)\s+iters?,\s+([\d.]+)ms', output)

    if match:
        status_symbol = match.group(1)
        iters = int(match.group(2))
        time_ms = float(match.group(3))

        status_map = {
            '✓': 'Optimal',
            '~': 'AlmostOptimal',
            'M': 'MaxIters',
            'N': 'NumericalError',
        }
        status = status_map.get(status_symbol, 'Unknown')

        return {
            "name": name,
            "status": status,
            "iterations": iters,
            "solve_time_ms": time_ms,
            "n": 0,  # Would need to parse from output
            "m": 0,
            "obj_val": 0.0,
            "mu": 0.0,
        }
    else:
        return {
            "name": name,
            "status": "NumericalError",
            "iterations": 0,
            "solve_time_ms": 0.0,
            "n": 0,
            "m": 0,
            "obj_val": float('nan'),
            "mu": float('nan'),
            "error": "Failed to parse output"
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", required=True, help="Output JSON path")
    parser.add_argument("--tol", type=float, default=1e-9, help="Tolerance")
    parser.add_argument("--max-iter", type=int, default=50, help="Max iterations")
    parser.add_argument("--limit", type=int, help="Limit number of problems")
    args = parser.parse_args()

    problems = MM_PROBLEMS[:args.limit] if args.limit else MM_PROBLEMS
    results = []

    print(f"Running Minix on {len(problems)} problems...")
    print("=" * 60)

    for i, name in enumerate(problems, 1):
        print(f"[{i}/{len(problems)}] {name:15s} ... ", end='', flush=True)
        result = run_minix_problem(name, args.tol, args.max_iter)

        status_symbols = {
            "Optimal": "✓",
            "AlmostOptimal": "~",
            "MaxIters": "M",
            "NumericalError": "N"
        }
        symbol = status_symbols.get(result["status"], "?")

        print(f"{symbol} ({result['iterations']:2d} iters, {result['solve_time_ms']:6.1f}ms)")
        results.append(result)

    print("=" * 60)

    # Compute summary
    import numpy as np
    optimal = sum(1 for r in results if r["status"] == "Optimal")
    almost = sum(1 for r in results if r["status"] == "AlmostOptimal")
    times = [r["solve_time_ms"] for r in results if r["status"] in ["Optimal", "AlmostOptimal"]]
    geom_mean_time = np.exp(np.mean(np.log([t + 1.0 for t in times]))) - 1.0 if times else 0.0

    summary = {
        "total": len(results),
        "optimal": optimal,
        "almost_optimal": almost,
        "geom_mean_time_ms": geom_mean_time
    }

    output = {
        "solver_name": "Minix",
        "results": results,
        "summary": summary
    }

    with open(args.export, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults exported to: {args.export}")
    print(f"Pass rate: {optimal} + {almost} = {optimal + almost}/{len(results)} ({100.0*(optimal+almost)/len(results):.1f}%)")
    print(f"Geometric mean time: {geom_mean_time:.2f}ms")


if __name__ == "__main__":
    main()
