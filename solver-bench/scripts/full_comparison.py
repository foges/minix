#!/usr/bin/env python3
"""
Full comparison of Minix vs Clarabel on Maros-Meszaros benchmark.

Outputs a table comparing iterations and wallclock time for each problem.
"""

import subprocess
import json
import os
import sys
import time
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
from scipy import sparse

# Small/medium problems first, large at end (for faster iteration)
# Skip: CONT-200/201/300, CVXQP*_L, BOYD1/2 (too slow)
MM_PROBLEMS_FAST = [
    # Tiny problems (<100ms typically)
    "HS21", "HS35", "HS35MOD", "HS51", "HS52", "HS53", "HS76", "HS118", "HS268",
    "TAME", "S268", "ZECEVIC2", "LOTSCHD", "QAFIRO",
    "CVXQP1_S", "CVXQP2_S", "CVXQP3_S",
    # Small problems (100ms-1s typically)
    "DPKLO1", "DTOC3",
    "DUAL1", "DUAL2", "DUAL3", "DUAL4", "DUALC1", "DUALC2", "DUALC5", "DUALC8",
    "PRIMAL1", "PRIMAL2", "PRIMAL3", "PRIMAL4", "PRIMALC1", "PRIMALC2", "PRIMALC5", "PRIMALC8",
    "AUG3D", "AUG3DC", "AUG3DCQP", "AUG3DQP",
    "EXDATA", "GOULDQP2", "GOULDQP3",
    "HUES-MOD", "HUESTIS", "KSIP", "LASER",
    "MOSARQP1", "MOSARQP2", "POWELL20",
    "STADAT1", "STADAT2", "STADAT3",
    "STCQP1", "STCQP2", "UBH1", "VALUES", "YAO",
    # Medium problems (1-10s typically)
    "CVXQP1_M", "CVXQP2_M", "CVXQP3_M",
    "AUG2D", "AUG2DC", "AUG2DCQP", "AUG2DQP",
    "CONT-050", "CONT-100", "CONT-101",
    "LISWET1", "LISWET2", "LISWET3", "LISWET4", "LISWET5", "LISWET6",
    "LISWET7", "LISWET8", "LISWET9", "LISWET10", "LISWET11", "LISWET12",
    # Q* problems
    "Q25FV47", "QADLITTL", "QBANDM", "QBEACONF", "QBORE3D", "QBRANDY",
    "QCAPRI", "QE226", "QETAMACR", "QFFFFF80", "QFORPLAN", "QGFRDXPN",
    "QGROW7", "QGROW15", "QGROW22",
    "QISRAEL", "QPCBLEND", "QPCBOEI1", "QPCBOEI2", "QPCSTAIR", "QPILOTNO", "QRECIPE",
    "QSC205", "QSCAGR25", "QSCAGR7", "QSCFXM1", "QSCFXM2", "QSCFXM3", "QSCORPIO", "QSCRS8",
    "QSCSD1", "QSCSD6", "QSCSD8", "QSCTAP1", "QSCTAP2", "QSCTAP3", "QSEBA", "QSHARE1B",
    "QSHARE2B", "QSHELL", "QSHIP04L", "QSHIP04S", "QSHIP08L", "QSHIP08S", "QSHIP12L", "QSHIP12S",
    "QSIERRA", "QSTAIR", "QSTANDAT",
]

# Full list including large problems
MM_PROBLEMS_ALL = MM_PROBLEMS_FAST + [
    "CVXQP1_L", "CVXQP2_L", "CVXQP3_L",
    "CONT-200", "CONT-201", "CONT-300",
    "BOYD1", "BOYD2",
]

MM_PROBLEMS = MM_PROBLEMS_FAST  # Default to fast set

@dataclass
class BenchResult:
    name: str
    status: str
    iterations: int
    time_ms: float
    objective: Optional[float] = None

def parse_qps(path):
    """Simple QPS parser - returns problem data for Clarabel."""
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

def run_minix(problem: str, timeout: int = 60) -> BenchResult:
    """Run Minix on a problem."""
    try:
        start = time.time()
        result = subprocess.run(
            ['cargo', 'run', '--release', '-p', 'solver-bench', '--',
             'benchmark', '--suite', 'mm', '--problem', problem],
            capture_output=True, text=True, timeout=timeout,
            cwd='/Users/chris/code/minix'
        )
        elapsed = (time.time() - start) * 1000

        output = result.stdout + result.stderr

        # Parse status
        status_match = re.search(r'Status:\s+(\w+)', output)
        status = status_match.group(1) if status_match else 'Unknown'

        # Parse iterations
        iter_match = re.search(r'Iterations:\s+(\d+)', output)
        iterations = int(iter_match.group(1)) if iter_match else 0

        # Parse time
        time_match = re.search(r'Time:\s+([\d.]+)\s*ms', output)
        time_ms = float(time_match.group(1)) if time_match else elapsed

        # Parse objective
        obj_match = re.search(r'Objective:\s+([-\d.e+]+)', output)
        objective = float(obj_match.group(1)) if obj_match else None

        return BenchResult(problem, status, iterations, time_ms, objective)
    except subprocess.TimeoutExpired:
        return BenchResult(problem, 'Timeout', 0, timeout * 1000)
    except Exception as e:
        return BenchResult(problem, f'Error: {e}', 0, 0)

def run_clarabel(problem: str, timeout: int = 300) -> BenchResult:
    """Run Clarabel on a problem."""
    try:
        import clarabel
    except ImportError:
        return BenchResult(problem, 'NotInstalled', 0, 0)

    qps_path = os.path.expanduser(f"~/.cache/minix-bench/maros-meszaros/{problem}.QPS")
    if not os.path.exists(qps_path):
        return BenchResult(problem, 'NoFile', 0, 0)

    try:
        n, m, P, q, A_raw, b_raw, row_types, var_bounds = parse_qps(qps_path)

        eq_rows = [i for i, t in enumerate(row_types) if t == 'E']
        le_rows = [i for i, t in enumerate(row_types) if t == 'L']
        ge_rows = [i for i, t in enumerate(row_types) if t == 'G']

        A_blocks = []
        b_blocks = []
        cones = []

        # Equality constraints: A*x = b -> Zero cone (s = 0)
        if eq_rows:
            A_blocks.append(A_raw[eq_rows, :])
            b_blocks.append(b_raw[eq_rows])
            cones.append(clarabel.ZeroConeT(len(eq_rows)))

        # Less-than constraints: a'x <= b -> a'x + s = b, s >= 0
        # s = b - a'x >= 0 means a'x <= b
        if le_rows:
            A_blocks.append(A_raw[le_rows, :])
            b_blocks.append(b_raw[le_rows])
            cones.append(clarabel.NonnegativeConeT(len(le_rows)))

        # Greater-than constraints: a'x >= b -> rewrite as -a'x <= -b
        # So: -(-a'x) + s = -(-b) -> a'x + s = b... no wait
        # a'x >= b means -a'x <= -b, so: -(-a')x + s = -(-b) -> a'x + s = b with s >= 0
        # This gives a'x = b - s... that's wrong! We need a'x >= b.
        # Correct: a'x >= b -> -a'x + s = -b with s >= 0 -> a'x = b + s >= b ✓
        if ge_rows:
            A_blocks.append(-A_raw[ge_rows, :])
            b_blocks.append(-b_raw[ge_rows])
            cones.append(clarabel.NonnegativeConeT(len(ge_rows)))

        # Variable bounds
        for ci in range(n):
            lo, hi = var_bounds.get(ci, [0.0, np.inf])

            # Lower bound: x_i >= lo -> -x_i + s = -lo, s >= 0 -> x_i = lo + s >= lo
            if np.isfinite(lo):
                row = sparse.csc_matrix(([-1.0], ([0], [ci])), shape=(1, n))
                A_blocks.append(row)
                b_blocks.append(np.array([-lo]))
                cones.append(clarabel.NonnegativeConeT(1))

            # Upper bound: x_i <= hi -> x_i + s = hi, s >= 0 -> x_i = hi - s <= hi
            if np.isfinite(hi):
                row = sparse.csc_matrix(([1.0], ([0], [ci])), shape=(1, n))
                A_blocks.append(row)
                b_blocks.append(np.array([hi]))
                cones.append(clarabel.NonnegativeConeT(1))

        if not A_blocks:
            # No constraints - create empty constraint matrix
            A = sparse.csc_matrix((0, n))
            b = np.array([])
            cones = []
        else:
            A = sparse.vstack(A_blocks).tocsc()
            b = np.concatenate(b_blocks)

        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = 200

        start = time.time()
        solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
        solution = solver.solve()
        elapsed_ms = (time.time() - start) * 1000

        status_map = {
            'Solved': 'Optimal',
            'SolvedInaccurate': 'AlmostSolved',
            'MaxIterations': 'MaxIters',
            'PrimalInfeasible': 'PrimalInfeasible',
            'DualInfeasible': 'DualInfeasible',
        }
        status = status_map.get(str(solution.status), str(solution.status))

        return BenchResult(problem, status, solution.iterations, elapsed_ms, solution.obj_val)
    except Exception as e:
        return BenchResult(problem, f'Error({type(e).__name__})', 0, 0)

def main():
    # Check for limit argument
    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            pass

    problems = MM_PROBLEMS[:limit] if limit else MM_PROBLEMS

    print(f"Running comparison on {len(problems)} problems...")
    print()

    results = []
    for i, problem in enumerate(problems):
        print(f"[{i+1}/{len(problems)}] {problem}...", end=' ', flush=True)

        minix = run_minix(problem)
        clarabel = run_clarabel(problem)

        results.append((minix, clarabel))

        # Quick summary
        if minix.status in ('Optimal', 'AlmostOptimal') and clarabel.status in ('Optimal', 'AlmostOptimal'):
            iter_ratio = minix.iterations / max(clarabel.iterations, 1)
            time_ratio = minix.time_ms / max(clarabel.time_ms, 0.1)
            emoji = "✓" if iter_ratio <= 1.5 else "⚠" if iter_ratio <= 2.0 else "✗"
            print(f"{emoji} M:{minix.iterations}/{minix.time_ms:.0f}ms C:{clarabel.iterations}/{clarabel.time_ms:.0f}ms")
        else:
            print(f"M:{minix.status}({minix.iterations}) C:{clarabel.status}({clarabel.iterations})")

    # Print summary table
    print()
    print("=" * 100)
    print(f"{'Problem':<15} {'Minix':^30} {'Clarabel':^30} {'Ratio':^20}")
    print(f"{'':<15} {'Status':<12} {'Iter':>5} {'Time':>10} {'Status':<12} {'Iter':>5} {'Time':>10} {'Iter':>8} {'Time':>8}")
    print("-" * 100)

    minix_wins_iter = 0
    clarabel_wins_iter = 0
    minix_wins_time = 0
    clarabel_wins_time = 0
    both_solved = 0

    worse_problems = []  # Problems where Minix is worse

    for minix, clarabel in results:
        if minix.status in ('Optimal', 'AlmostOptimal') and clarabel.status in ('Optimal', 'AlmostOptimal'):
            both_solved += 1
            iter_ratio = minix.iterations / max(clarabel.iterations, 1)
            time_ratio = minix.time_ms / max(clarabel.time_ms, 0.1)

            if iter_ratio < 0.9:
                minix_wins_iter += 1
            elif iter_ratio > 1.1:
                clarabel_wins_iter += 1

            if time_ratio < 0.9:
                minix_wins_time += 1
            elif time_ratio > 1.1:
                clarabel_wins_time += 1

            # Track problems where Minix is significantly worse
            if iter_ratio > 1.3 or time_ratio > 1.5:
                worse_problems.append((minix.name, iter_ratio, time_ratio, minix, clarabel))

            print(f"{minix.name:<15} {minix.status:<12} {minix.iterations:>5} {minix.time_ms:>9.1f}ms "
                  f"{clarabel.status:<12} {clarabel.iterations:>5} {clarabel.time_ms:>9.1f}ms "
                  f"{iter_ratio:>7.2f}x {time_ratio:>7.2f}x")
        else:
            print(f"{minix.name:<15} {minix.status:<12} {minix.iterations:>5} {minix.time_ms:>9.1f}ms "
                  f"{clarabel.status:<12} {clarabel.iterations:>5} {clarabel.time_ms:>9.1f}ms "
                  f"{'N/A':>8} {'N/A':>8}")

    print("=" * 100)
    print()
    print("SUMMARY")
    print(f"  Both solved: {both_solved}/{len(results)}")
    print(f"  Iterations: Minix wins {minix_wins_iter}, Clarabel wins {clarabel_wins_iter}")
    print(f"  Time: Minix wins {minix_wins_time}, Clarabel wins {clarabel_wins_time}")

    if worse_problems:
        print()
        print("=" * 100)
        print("PROBLEMS WHERE MINIX IS WORSE (iter_ratio > 1.3 or time_ratio > 1.5)")
        print("=" * 100)
        worse_problems.sort(key=lambda x: -x[1])  # Sort by iter ratio descending
        for name, iter_ratio, time_ratio, minix, clarabel in worse_problems:
            print(f"  {name}: {iter_ratio:.2f}x iters ({minix.iterations} vs {clarabel.iterations}), "
                  f"{time_ratio:.2f}x time ({minix.time_ms:.0f}ms vs {clarabel.time_ms:.0f}ms)")

    # Save results for later analysis
    with open('/tmp/claude/minix_vs_clarabel.json', 'w') as f:
        data = []
        for minix, clarabel in results:
            data.append({
                'problem': minix.name,
                'minix_status': minix.status,
                'minix_iter': minix.iterations,
                'minix_time': minix.time_ms,
                'minix_obj': minix.objective,
                'clarabel_status': clarabel.status,
                'clarabel_iter': clarabel.iterations,
                'clarabel_time': clarabel.time_ms,
                'clarabel_obj': clarabel.objective,
            })
        json.dump(data, f, indent=2)
    print()
    print(f"Results saved to /tmp/claude/minix_vs_clarabel.json")

if __name__ == '__main__':
    main()
