#!/usr/bin/env python3
"""Benchmark Clarabel vs Minix on Maros-Meszaros QP suite via CVXPY."""

import os
import time
import numpy as np
import cvxpy as cp
from scipy import sparse
from pathlib import Path

# Register Minix
from minix_cvxpy import register_minix
register_minix()

def parse_qps(filepath):
    """Parse a QPS file and return problem data."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    section = None
    name = ''
    rows = {}  # name -> (type, index)
    cols = {}  # name -> index

    # First pass: count rows and cols
    row_list = []
    col_set = set()

    for line in lines:
        line = line.rstrip()
        if not line or line.startswith('*'):
            continue

        if line.startswith('NAME'):
            name = line.split()[1] if len(line.split()) > 1 else ''
            continue

        if line.startswith('ROWS'):
            section = 'ROWS'
            continue
        elif line.startswith('COLUMNS'):
            section = 'COLUMNS'
            continue
        elif line.startswith('RHS'):
            section = 'RHS'
            continue
        elif line.startswith('RANGES'):
            section = 'RANGES'
            continue
        elif line.startswith('BOUNDS'):
            section = 'BOUNDS'
            continue
        elif line.startswith('QUADOBJ'):
            section = 'QUADOBJ'
            continue
        elif line.startswith('ENDATA'):
            break

        parts = line.split()

        if section == 'ROWS' and len(parts) >= 2:
            row_type = parts[0]
            row_name = parts[1]
            row_list.append((row_name, row_type))

        elif section == 'COLUMNS' and len(parts) >= 3:
            col_name = parts[0]
            col_set.add(col_name)

    # Build indices
    n_rows = len(row_list)
    rows = {name: (typ, i) for i, (name, typ) in enumerate(row_list)}

    col_names = sorted(col_set)
    n_cols = len(col_names)
    cols = {name: i for i, name in enumerate(col_names)}

    # Initialize data structures
    c = np.zeros(n_cols)
    A_data = []  # (row, col, val)
    b = np.zeros(n_rows)
    row_types = [rows[name][1] for name, _ in row_list]
    P_data = []  # (row, col, val)
    lb = np.full(n_cols, -np.inf)
    ub = np.full(n_cols, np.inf)

    # Second pass: read data
    section = None
    for line in lines:
        line = line.rstrip()
        if not line or line.startswith('*'):
            continue

        if line.startswith('NAME'):
            continue
        if line.startswith('ROWS'):
            section = 'ROWS'
            continue
        elif line.startswith('COLUMNS'):
            section = 'COLUMNS'
            continue
        elif line.startswith('RHS'):
            section = 'RHS'
            continue
        elif line.startswith('RANGES'):
            section = 'RANGES'
            continue
        elif line.startswith('BOUNDS'):
            section = 'BOUNDS'
            continue
        elif line.startswith('QUADOBJ'):
            section = 'QUADOBJ'
            continue
        elif line.startswith('ENDATA'):
            break

        parts = line.split()

        if section == 'COLUMNS' and len(parts) >= 3:
            col_name = parts[0]
            col_idx = cols[col_name]

            i = 1
            while i + 1 < len(parts):
                row_name = parts[i]
                val = float(parts[i+1])

                if row_name in rows:
                    row_type, row_idx = rows[row_name]
                    if row_type == 'N':  # Objective
                        c[col_idx] = val
                    else:
                        A_data.append((row_idx, col_idx, val))
                i += 2

        elif section == 'RHS' and len(parts) >= 3:
            i = 1
            while i + 1 < len(parts):
                row_name = parts[i]
                val = float(parts[i+1])
                if row_name in rows:
                    _, row_idx = rows[row_name]
                    b[row_idx] = val
                i += 2

        elif section == 'BOUNDS' and len(parts) >= 3:
            bound_type = parts[0]
            col_name = parts[2] if len(parts) > 2 else parts[1]

            if col_name in cols:
                col_idx = cols[col_name]

                if bound_type in ('LO', 'LI'):
                    lb[col_idx] = float(parts[3]) if len(parts) > 3 else 0
                elif bound_type in ('UP', 'UI'):
                    ub[col_idx] = float(parts[3]) if len(parts) > 3 else 0
                elif bound_type == 'FX':
                    val = float(parts[3]) if len(parts) > 3 else 0
                    lb[col_idx] = val
                    ub[col_idx] = val
                elif bound_type == 'FR':
                    lb[col_idx] = -np.inf
                    ub[col_idx] = np.inf
                elif bound_type == 'MI':
                    lb[col_idx] = -np.inf
                elif bound_type == 'PL':
                    ub[col_idx] = np.inf
                elif bound_type == 'BV':
                    lb[col_idx] = 0
                    ub[col_idx] = 1

        elif section == 'QUADOBJ' and len(parts) >= 3:
            col1 = parts[0]
            i = 1
            while i + 1 < len(parts):
                col2 = parts[i]
                val = float(parts[i+1])
                if col1 in cols and col2 in cols:
                    idx1 = cols[col1]
                    idx2 = cols[col2]
                    P_data.append((idx1, idx2, val))
                    if idx1 != idx2:
                        P_data.append((idx2, idx1, val))
                i += 2

    # Build sparse matrices
    if A_data:
        A_rows, A_cols, A_vals = zip(*A_data)
        A = sparse.csr_matrix((A_vals, (A_rows, A_cols)), shape=(n_rows, n_cols))
    else:
        A = sparse.csr_matrix((n_rows, n_cols))

    if P_data:
        P_rows, P_cols, P_vals = zip(*P_data)
        P = sparse.csr_matrix((P_vals, (P_rows, P_cols)), shape=(n_cols, n_cols))
    else:
        P = sparse.csr_matrix((n_cols, n_cols))

    return {
        'name': name,
        'n': n_cols,
        'm': n_rows,
        'P': P,
        'c': c,
        'A': A,
        'b': b,
        'row_types': row_types,
        'lb': lb,
        'ub': ub,
    }

def build_cvxpy_problem(data):
    """Build a CVXPY problem from parsed QPS data."""
    n = data['n']
    x = cp.Variable(n)

    # Objective: 0.5 x'Px + c'x
    P = data['P']
    c = data['c']

    if P.nnz > 0:
        objective = 0.5 * cp.quad_form(x, P.toarray()) + c @ x
    else:
        objective = c @ x

    constraints = []

    # Process row constraints
    A = data['A']
    b = data['b']
    row_types = data['row_types']

    for i, row_type in enumerate(row_types):
        if row_type == 'N':  # Objective row, skip
            continue

        row = A.getrow(i)
        if row.nnz == 0:
            continue

        # Convert to dense for constraint
        row_dense = row.toarray().flatten()
        rhs = b[i]

        if row_type == 'E':  # Equality
            constraints.append(row_dense @ x == rhs)
        elif row_type == 'L':  # Less than or equal
            constraints.append(row_dense @ x <= rhs)
        elif row_type == 'G':  # Greater than or equal
            constraints.append(row_dense @ x >= rhs)

    # Bounds
    lb = data['lb']
    ub = data['ub']

    lb_finite = np.isfinite(lb)
    ub_finite = np.isfinite(ub)

    if lb_finite.any():
        for i in np.where(lb_finite)[0]:
            constraints.append(x[i] >= lb[i])

    if ub_finite.any():
        for i in np.where(ub_finite)[0]:
            constraints.append(x[i] <= ub[i])

    return cp.Problem(cp.Minimize(objective), constraints), x


def main():
    # Main benchmark
    mm_dir = Path(os.path.expanduser('~/.cache/minix-bench/maros-meszaros'))
    qps_files = sorted(mm_dir.glob('*.QPS'))

    print(f'Found {len(qps_files)} QPS files')
    print()
    print(f'{"Problem":20} {"n":>6} {"m":>6} {"Clarabel":>12} {"Minix":>12} {"Ratio":>8} {"Status":>20}')
    print('-' * 90)

    results = []
    for qps_file in qps_files:
        name = qps_file.stem
        try:
            data = parse_qps(qps_file)
            prob, x = build_cvxpy_problem(data)

            # Clarabel
            try:
                start = time.perf_counter()
                prob.solve(solver=cp.CLARABEL, verbose=False)
                clarabel_time = (time.perf_counter() - start) * 1000
                clarabel_status = prob.status[:10]
                clarabel_obj = prob.value
            except Exception as e:
                clarabel_time = float('nan')
                clarabel_status = 'error'
                clarabel_obj = float('nan')

            # Minix
            try:
                start = time.perf_counter()
                prob.solve(solver=cp.MINIX, verbose=False)
                minix_time = (time.perf_counter() - start) * 1000
                minix_status = prob.status[:10]
                minix_obj = prob.value
            except Exception as e:
                minix_time = float('nan')
                minix_status = 'error'
                minix_obj = float('nan')

            ratio = minix_time / clarabel_time if clarabel_time > 0 else float('nan')
            status = f'{clarabel_status}/{minix_status}'

            print(f'{name:20} {data["n"]:6} {data["m"]:6} {clarabel_time:10.1f}ms {minix_time:10.1f}ms {ratio:7.2f}x {status:>20}')

            results.append({
                'name': name,
                'n': data['n'],
                'clarabel_time': clarabel_time,
                'minix_time': minix_time,
                'clarabel_status': clarabel_status,
                'minix_status': minix_status,
            })

        except Exception as e:
            print(f'{name:20} ERROR: {str(e)[:50]}')

    # Summary
    print()
    print('=' * 90)
    print('SUMMARY')
    print('=' * 90)

    valid = [r for r in results if not np.isnan(r['clarabel_time']) and not np.isnan(r['minix_time'])]

    both_optimal = [r for r in valid if 'optimal' in r['clarabel_status'] and 'optimal' in r['minix_status']]
    clarabel_only = [r for r in valid if 'optimal' in r['clarabel_status'] and 'optimal' not in r['minix_status']]
    minix_only = [r for r in valid if 'optimal' not in r['clarabel_status'] and 'optimal' in r['minix_status']]

    print(f'Total problems: {len(results)}')
    print(f'Valid comparisons: {len(valid)}')
    print(f'Both optimal: {len(both_optimal)}')
    print(f'Clarabel only optimal: {len(clarabel_only)}')
    print(f'Minix only optimal: {len(minix_only)}')

    if valid:
        ct = sum(r['clarabel_time'] for r in valid)
        mt = sum(r['minix_time'] for r in valid)

        # Geometric means
        ct_geom = np.exp(np.mean([np.log(r['clarabel_time']) for r in valid if r['clarabel_time'] > 0]))
        mt_geom = np.exp(np.mean([np.log(r['minix_time']) for r in valid if r['minix_time'] > 0]))

        print(f'\nAll valid ({len(valid)} problems):')
        print(f'  Clarabel: {ct:.1f}ms total, {ct_geom:.1f}ms geom mean')
        print(f'  Minix:    {mt:.1f}ms total, {mt_geom:.1f}ms geom mean')
        print(f'  Ratio (Minix/Clarabel): {mt/ct:.2f}x')

    if both_optimal:
        ct = sum(r['clarabel_time'] for r in both_optimal)
        mt = sum(r['minix_time'] for r in both_optimal)
        ct_geom = np.exp(np.mean([np.log(r['clarabel_time']) for r in both_optimal if r['clarabel_time'] > 0]))
        mt_geom = np.exp(np.mean([np.log(r['minix_time']) for r in both_optimal if r['minix_time'] > 0]))

        print(f'\nBoth optimal ({len(both_optimal)} problems):')
        print(f'  Clarabel: {ct:.1f}ms total, {ct_geom:.1f}ms geom mean')
        print(f'  Minix:    {mt:.1f}ms total, {mt_geom:.1f}ms geom mean')
        print(f'  Ratio (Minix/Clarabel): {mt/ct:.2f}x')

    # Faster/slower breakdown
    faster = [r for r in both_optimal if r['minix_time'] < r['clarabel_time'] * 0.9]
    slower = [r for r in both_optimal if r['minix_time'] > r['clarabel_time'] * 1.1]
    similar = [r for r in both_optimal if r not in faster and r not in slower]

    print(f'\nWhere both optimal:')
    print(f'  Minix faster (>10%): {len(faster)}')
    print(f'  Similar (within 10%): {len(similar)}')
    print(f'  Minix slower (>10%): {len(slower)}')

    if clarabel_only:
        print(f'\nProblems where only Clarabel found optimal:')
        for r in clarabel_only[:10]:
            print(f'  {r["name"]}: Clarabel={r["clarabel_status"]}, Minix={r["minix_status"]}')

    if minix_only:
        print(f'\nProblems where only Minix found optimal:')
        for r in minix_only[:10]:
            print(f'  {r["name"]}: Clarabel={r["clarabel_status"]}, Minix={r["minix_status"]}')


if __name__ == '__main__':
    main()
