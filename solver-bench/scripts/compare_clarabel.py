#!/usr/bin/env python3
"""
Compare Minix and Clarabel iteration-by-iteration on a problem.

Usage:
    python compare_clarabel.py YAO
    python compare_clarabel.py STCQP1
"""

import sys
import os
import subprocess
import re
import numpy as np

def parse_minix_output(output):
    """Parse Minix diagnostic output."""
    iterations = []
    for line in output.split('\n'):
        # iter    0 mu=3.155e2 alpha=6.436e-1 alpha_sz=6.501e-1 min_s=0.000e0 min_z=-5.510e1 sigma=1.953e-1 rel_p=6.682e-1 rel_d=8.712e-1 gap_rel=1.329e0
        m = re.match(r'iter\s+(\d+)\s+mu=([^\s]+)\s+alpha=([^\s]+)\s+alpha_sz=([^\s]+)\s+.*sigma=([^\s]+)\s+rel_p=([^\s]+)\s+rel_d=([^\s]+)\s+gap_rel=([^\s]+)', line)
        if m:
            iterations.append({
                'iter': int(m.group(1)),
                'mu': float(m.group(2)),
                'alpha': float(m.group(3)),
                'alpha_sz': float(m.group(4)),
                'sigma': float(m.group(5)),
                'rel_p': float(m.group(6)),
                'rel_d': float(m.group(7)),
                'gap_rel': float(m.group(8)),
            })
    return iterations

def parse_clarabel_output(output):
    """Parse Clarabel verbose output."""
    iterations = []
    for line in output.split('\n'):
        # 0  -2.7278e+02  -2.7246e+02  1.19e-03  7.67e-01  7.87e-02  1.00e+00  1.72e+00   ------
        # 1  -2.7122e+02  -2.7698e+02  2.12e-02  3.47e-01  7.07e-03  1.21e+00  1.15e-01  9.34e-01
        m = re.match(r'\s*(\d+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)', line)
        if m:
            step = m.group(9)
            if step == '------':
                step = 0.0
            else:
                step = float(step)
            iterations.append({
                'iter': int(m.group(1)),
                'pcost': float(m.group(2)),
                'dcost': float(m.group(3)),
                'gap': float(m.group(4)),
                'pres': float(m.group(5)),
                'dres': float(m.group(6)),
                'kt': float(m.group(7)),
                'mu': float(m.group(8)),
                'step': step,
            })
    return iterations

def run_minix(problem):
    """Run Minix on a problem with diagnostics."""
    env = os.environ.copy()
    env['MINIX_DIAGNOSTICS'] = '1'
    result = subprocess.run(
        ['cargo', 'run', '--release', '-p', 'solver-bench', '--',
         'maros-meszaros', '--problem', problem],
        capture_output=True, text=True, env=env,
        cwd='/Users/chris/code/minix'
    )
    return result.stdout + result.stderr

def run_clarabel(problem):
    """Run Clarabel on a problem."""
    script = f'''
import clarabel
import numpy as np
from scipy import sparse
import os

def parse_qps(path):
    """Simple QPS parser"""
    with open(path) as f:
        lines = f.readlines()

    section = None
    rows = {{}}
    cols = {{}}
    A_data = []
    q_data = {{}}
    P_data = []
    rhs_data = {{}}
    bounds = {{}}
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

qps_path = os.path.expanduser("~/.cache/minix-bench/maros-meszaros/{problem}.QPS")
n, m, P, q, A_raw, b_raw, row_types, var_bounds = parse_qps(qps_path)

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

settings = clarabel.DefaultSettings()
settings.verbose = True
settings.max_iter = 200

solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
solution = solver.solve()

print(f"Status: {{solution.status}}")
print(f"Iterations: {{solution.iterations}}")
print(f"Objective: {{solution.obj_val:.6e}}")
'''

    result = subprocess.run(
        ['python3', '-c', script],
        capture_output=True, text=True,
        cwd='/Users/chris/code/minix/solver-py',
        env={**os.environ, 'VIRTUAL_ENV': '/Users/chris/code/minix/solver-py/.venv',
             'PATH': '/Users/chris/code/minix/solver-py/.venv/bin:' + os.environ['PATH']}
    )
    return result.stdout + result.stderr

def compare(problem):
    print(f"=== Comparing {problem} ===\n")

    # Run both
    print("Running Minix...")
    minix_out = run_minix(problem)
    minix_iters = parse_minix_output(minix_out)

    print("Running Clarabel...")
    clarabel_out = run_clarabel(problem)
    clarabel_iters = parse_clarabel_output(clarabel_out)

    print(f"\n{'='*80}")
    print(f"ITERATION COMPARISON: {problem}")
    print(f"{'='*80}")

    print(f"\n{'Iter':>4} | {'Minix mu':>10} {'alpha':>8} {'sigma':>8} {'rel_p':>10} | {'Clarabel mu':>12} {'step':>8} {'pres':>10}")
    print("-" * 90)

    max_iter = max(len(minix_iters), len(clarabel_iters))
    for i in range(min(max_iter, 30)):
        m = minix_iters[i] if i < len(minix_iters) else None
        c = clarabel_iters[i] if i < len(clarabel_iters) else None

        if m and c:
            print(f"{i:>4} | {m['mu']:>10.2e} {m['alpha']:>8.3f} {m['sigma']:>8.3f} {m['rel_p']:>10.2e} | {c['mu']:>12.2e} {c['step']:>8.3f} {c['pres']:>10.2e}")
        elif m:
            print(f"{i:>4} | {m['mu']:>10.2e} {m['alpha']:>8.3f} {m['sigma']:>8.3f} {m['rel_p']:>10.2e} | {'(done)':>12}")
        elif c:
            print(f"{i:>4} | {'(done)':>10} {'-':>8} {'-':>8} {'-':>10} | {c['mu']:>12.2e} {c['step']:>8.3f} {c['pres']:>10.2e}")

    if len(minix_iters) > 30:
        print(f"... {len(minix_iters) - 30} more Minix iterations")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Minix:    {len(minix_iters)} iterations")
    print(f"Clarabel: {len(clarabel_iters)} iterations")

    if minix_iters and clarabel_iters:
        # Analyze step sizes
        minix_alphas = [it['alpha'] for it in minix_iters]
        clarabel_steps = [it['step'] for it in clarabel_iters if it['step'] > 0]

        print(f"\nStep size analysis:")
        print(f"  Minix avg alpha:    {np.mean(minix_alphas):.3f} (min={np.min(minix_alphas):.3f}, max={np.max(minix_alphas):.3f})")
        if clarabel_steps:
            print(f"  Clarabel avg step:  {np.mean(clarabel_steps):.3f} (min={np.min(clarabel_steps):.3f}, max={np.max(clarabel_steps):.3f})")

        # Analyze sigma (centering)
        minix_sigmas = [it['sigma'] for it in minix_iters]
        print(f"\nCentering (sigma) analysis:")
        print(f"  Minix avg sigma:    {np.mean(minix_sigmas):.3f} (min={np.min(minix_sigmas):.3f}, max={np.max(minix_sigmas):.3f})")

        # mu reduction rate
        if len(minix_iters) >= 2:
            minix_mu_ratio = minix_iters[-1]['mu'] / minix_iters[0]['mu']
            minix_mu_per_iter = minix_mu_ratio ** (1/len(minix_iters))
            print(f"\nmu reduction per iteration:")
            print(f"  Minix:    {minix_mu_per_iter:.4f}x")
        if len(clarabel_iters) >= 2:
            clarabel_mu_ratio = clarabel_iters[-1]['mu'] / clarabel_iters[0]['mu']
            clarabel_mu_per_iter = clarabel_mu_ratio ** (1/len(clarabel_iters))
            print(f"  Clarabel: {clarabel_mu_per_iter:.4f}x")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python compare_clarabel.py PROBLEM_NAME")
        print("Example: python compare_clarabel.py YAO")
        sys.exit(1)

    problem = sys.argv[1]
    compare(problem)
