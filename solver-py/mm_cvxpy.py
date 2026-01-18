#!/usr/bin/env python3
"""MM benchmark via CVXPY."""
import os, time, sys
import numpy as np
import cvxpy as cp
from scipy import sparse
from pathlib import Path

from minix_cvxpy import register_minix
register_minix()

def parse_qps(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    section = None
    rows = {}
    col_set = set()
    row_list = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('*'):
            continue
        parts = line.split()
        if not parts:
            continue
        if parts[0] in ['NAME', 'ROWS', 'COLUMNS', 'RHS', 'RANGES', 'BOUNDS', 'QUADOBJ', 'ENDATA']:
            section = parts[0]
            continue
        if section == 'ROWS' and len(parts) >= 2:
            row_list.append((parts[1], parts[0]))
        elif section == 'COLUMNS' and len(parts) >= 3:
            col_set.add(parts[0])

    col_names = sorted(col_set)
    n, m = len(col_names), len(row_list)
    if n == 0:
        return None

    cols = {name: i for i, name in enumerate(col_names)}
    rows = {name: (typ, i) for i, (name, typ) in enumerate(row_list)}

    c = np.zeros(n)
    P_trips, A_trips = [], []
    b = np.zeros(m)
    lb, ub = np.zeros(n), np.full(n, np.inf)
    row_types = [typ for _, typ in row_list]

    section = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith('*'):
            continue
        parts = line.split()
        if not parts:
            continue
        if parts[0] in ['NAME', 'ROWS', 'COLUMNS', 'RHS', 'RANGES', 'BOUNDS', 'QUADOBJ', 'ENDATA']:
            section = parts[0]
            continue

        if section == 'COLUMNS' and len(parts) >= 3:
            col = cols.get(parts[0])
            if col is None:
                continue
            i = 1
            while i + 1 < len(parts):
                if parts[i] in rows:
                    typ, row = rows[parts[i]]
                    val = float(parts[i+1])
                    if typ == 'N':
                        c[col] = val
                    else:
                        A_trips.append((row, col, val))
                i += 2
        elif section == 'RHS' and len(parts) >= 2:
            i = 1
            while i + 1 < len(parts):
                if parts[i] in rows:
                    _, row = rows[parts[i]]
                    b[row] = float(parts[i+1])
                i += 2
        elif section == 'BOUNDS' and len(parts) >= 3:
            btype = parts[0]
            cname = parts[2] if len(parts) > 2 else parts[1]
            if cname in cols:
                col = cols[cname]
                val = float(parts[3]) if len(parts) > 3 else 0
                if btype == 'LO': lb[col] = val
                elif btype == 'UP': ub[col] = val
                elif btype == 'FX': lb[col] = ub[col] = val
                elif btype == 'FR': lb[col] = -np.inf
                elif btype == 'MI': lb[col] = -np.inf
        elif section == 'QUADOBJ' and len(parts) >= 3:
            c1 = cols.get(parts[0])
            if c1 is None:
                continue
            i = 1
            while i + 1 < len(parts):
                c2 = cols.get(parts[i])
                if c2 is not None:
                    val = float(parts[i+1])
                    P_trips.append((c1, c2, val))
                    if c1 != c2:
                        P_trips.append((c2, c1, val))
                i += 2

    if P_trips:
        pr, pc, pv = zip(*P_trips)
        P = sparse.csr_matrix((pv, (pr, pc)), shape=(n, n))
    else:
        P = sparse.csr_matrix((n, n))

    if A_trips:
        ar, ac, av = zip(*A_trips)
        A = sparse.csr_matrix((av, (ar, ac)), shape=(m, n))
    else:
        A = sparse.csr_matrix((m, n))

    return {'n': n, 'm': m, 'P': P, 'c': c, 'A': A, 'b': b, 'lb': lb, 'ub': ub, 'row_types': row_types}

mm_dir = Path(os.path.expanduser('~/.cache/minix-bench/maros-meszaros'))
files = sorted(mm_dir.glob('*.QPS'))

print(f'Testing {len(files)} MM problems')
print(f'{"Problem":15} {"n":>5} {"Clarabel":>10} {"Minix":>10} {"Ratio":>7}')
print('-' * 55)
sys.stdout.flush()

results = []
for f in files:
    name = f.stem
    try:
        data = parse_qps(f)
        if data is None or data['n'] > 2000:
            print(f'{name:15} SKIP')
            sys.stdout.flush()
            continue

        n = data['n']
        x = cp.Variable(n)
        P, c, A, b = data['P'], data['c'], data['A'], data['b']

        obj = (0.5 * cp.quad_form(x, P.toarray()) + c @ x) if P.nnz > 0 else c @ x
        cons = []
        for i, rt in enumerate(data['row_types']):
            if rt == 'N': continue
            row = A.getrow(i).toarray().flatten()
            if rt == 'E': cons.append(row @ x == b[i])
            elif rt == 'L': cons.append(row @ x <= b[i])
            elif rt == 'G': cons.append(row @ x >= b[i])

        lb, ub = data['lb'], data['ub']
        for i in range(n):
            if lb[i] > -1e10: cons.append(x[i] >= lb[i])
            if ub[i] < 1e10: cons.append(x[i] <= ub[i])

        prob = cp.Problem(cp.Minimize(obj), cons)

        start = time.perf_counter()
        prob.solve(solver=cp.CLARABEL, verbose=False, max_iter=100)
        ct = (time.perf_counter() - start) * 1000
        cs = prob.status[:8]

        start = time.perf_counter()
        prob.solve(solver=cp.MINIX, verbose=False, max_iter=100)
        mt = (time.perf_counter() - start) * 1000
        ms = prob.status[:8]

        ratio = mt/ct if ct > 0 else 0
        print(f'{name:15} {n:5} {ct:8.1f}ms {mt:8.1f}ms {ratio:6.2f}x')
        sys.stdout.flush()
        results.append((name, n, ct, mt, cs, ms))
    except Exception as e:
        print(f'{name:15} ERROR: {str(e)[:30]}')
        sys.stdout.flush()

print()
print('='*55)
if results:
    ct_sum = sum(r[2] for r in results)
    mt_sum = sum(r[3] for r in results)
    print(f'Total ({len(results)}): Clarabel={ct_sum:.0f}ms, Minix={mt_sum:.0f}ms, Ratio={mt_sum/ct_sum:.2f}x')

    both_opt = [r for r in results if 'optimal' in r[4] and 'optimal' in r[5]]
    if both_opt:
        ct_opt = sum(r[2] for r in both_opt)
        mt_opt = sum(r[3] for r in both_opt)
        print(f'Both optimal ({len(both_opt)}): Clarabel={ct_opt:.0f}ms, Minix={mt_opt:.0f}ms, Ratio={mt_opt/ct_opt:.2f}x')
