#!/usr/bin/env python3
"""Test Clarabel on QSCFXM3 with verbose output."""
import cvxpy as cp
import numpy as np
import time
from scipy import sparse
from pathlib import Path

def parse_qps(filepath):
    with open(filepath) as f:
        lines = f.read().split('\n')
    section = None
    rows, col_set, row_list = {}, set(), []
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
    if n == 0 or m == 0:
        return None
    cols = {name: i for i, name in enumerate(col_names)}
    rows = {name: (typ, i) for i, (name, typ) in enumerate(row_list)}
    c, P_trips, A_trips, b = np.zeros(n), [], [], np.zeros(m)
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
                if btype == 'LO':
                    lb[col] = float(parts[3]) if len(parts) > 3 else 0
                elif btype == 'UP':
                    ub[col] = float(parts[3]) if len(parts) > 3 else np.inf
                elif btype == 'FX':
                    lb[col] = ub[col] = float(parts[3]) if len(parts) > 3 else 0
                elif btype == 'FR':
                    lb[col] = -np.inf
                elif btype == 'MI':
                    lb[col] = -np.inf
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

data = parse_qps(Path.home() / '.cache/minix-bench/maros-meszaros/QSCFXM3.QPS')
n = data['n']
x = cp.Variable(n)
P, c, A, b = data['P'], data['c'], data['A'], data['b']
obj = 0.5 * cp.quad_form(x, P.toarray()) + c @ x if P.nnz > 0 else c @ x
cons = []
for i, rt in enumerate(data['row_types']):
    if rt == 'N':
        continue
    row = A.getrow(i).toarray().flatten()
    if rt == 'E':
        cons.append(row @ x == b[i])
    elif rt == 'L':
        cons.append(row @ x <= b[i])
    elif rt == 'G':
        cons.append(row @ x >= b[i])
lb, ub = data['lb'], data['ub']
for i in range(n):
    if np.isfinite(lb[i]) and lb[i] != 0:
        cons.append(x[i] >= lb[i])
    elif lb[i] == 0:
        cons.append(x[i] >= 0)
    if np.isfinite(ub[i]):
        cons.append(x[i] <= ub[i])
prob = cp.Problem(cp.Minimize(obj), cons)

print("Running Clarabel with verbose=True...")
start = time.perf_counter()
prob.solve(solver=cp.CLARABEL, verbose=True, max_iter=100)
elapsed = (time.perf_counter() - start) * 1000
print(f'\nTime: {elapsed:.1f}ms')
print(f'Status: {prob.status}')
print(f'Objective: {prob.value}')
