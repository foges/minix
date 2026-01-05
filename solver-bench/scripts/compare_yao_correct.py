#!/usr/bin/env python3
"""Compare Minix and Clarabel on YAO with correct FX bounds handling."""

import clarabel
import numpy as np
from scipy import sparse
import os
import subprocess

def parse_qps(path):
    """Parse QPS file with proper FX bounds support."""
    with open(path) as f:
        lines = f.readlines()
    section = None
    rows, cols = {}, {}
    A_data, q_data, P_data, bounds = [], {}, [], {}
    row_idx, col_idx, obj_row = 0, 0, None

    for line in lines:
        line = line.rstrip()
        if not line or line.startswith("*"):
            continue
        for kw in ["NAME", "ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS", "QUADOBJ", "ENDATA"]:
            if line.startswith(kw):
                section = kw if kw not in ["NAME", "ENDATA"] else section
                break
        else:
            parts = line.split()
            if section == "ROWS" and len(parts) >= 2:
                if parts[0] == "N":
                    obj_row = parts[1]
                else:
                    rows[parts[1]] = (parts[0], row_idx)
                    row_idx += 1
            elif section == "COLUMNS" and len(parts) >= 3:
                if parts[0] not in cols:
                    cols[parts[0]] = len(cols)
                ci = cols[parts[0]]
                for i in range(1, len(parts)-1, 2):
                    rname, val = parts[i], float(parts[i+1])
                    if rname == obj_row:
                        q_data[ci] = val
                    elif rname in rows:
                        A_data.append((rows[rname][1], ci, val))
            elif section == "BOUNDS" and len(parts) >= 3:
                ci = cols.get(parts[2])
                if ci is not None:
                    if ci not in bounds:
                        bounds[ci] = [0.0, np.inf]
                    val = float(parts[3]) if len(parts) > 3 else 0.0
                    bt = parts[0]
                    if bt == "FR":
                        bounds[ci] = [-np.inf, np.inf]
                    elif bt == "LO":
                        bounds[ci][0] = val
                    elif bt == "UP":
                        bounds[ci][1] = val
                    elif bt == "MI":
                        bounds[ci][0] = -np.inf
                    elif bt == "FX":
                        bounds[ci] = [val, val]  # Fixed bounds
            elif section == "QUADOBJ" and len(parts) >= 3:
                i, j = cols.get(parts[0]), cols.get(parts[1])
                if i is not None and j is not None:
                    P_data.append((i, j, float(parts[2])))
                    if i != j:
                        P_data.append((j, i, float(parts[2])))

    n, m = len(cols), row_idx
    q = np.array([q_data.get(i, 0.0) for i in range(n)])

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

    row_types = ["E"] * m
    for rn, (rt, ri) in rows.items():
        row_types[ri] = rt

    return n, m, P, q, A, row_types, bounds


def run_clarabel(qps_path):
    """Run Clarabel with correct bounds handling."""
    n, m, P, q, A_raw, row_types, var_bounds = parse_qps(qps_path)

    A_blocks, b_blocks, cones = [], [], []

    # G constraints: Ax >= 0 => -Ax + s = 0, s >= 0
    ge_rows = [i for i, t in enumerate(row_types) if t == "G"]
    le_rows = [i for i, t in enumerate(row_types) if t == "L"]
    eq_rows = [i for i, t in enumerate(row_types) if t == "E"]

    if eq_rows:
        A_blocks.append(A_raw[eq_rows, :])
        b_blocks.append(np.zeros(len(eq_rows)))
        cones.append(clarabel.ZeroConeT(len(eq_rows)))

    if ge_rows:
        A_blocks.append(-A_raw[ge_rows, :])
        b_blocks.append(np.zeros(len(ge_rows)))
        cones.append(clarabel.NonnegativeConeT(len(ge_rows)))

    if le_rows:
        A_blocks.append(A_raw[le_rows, :])
        b_blocks.append(np.zeros(len(le_rows)))
        cones.append(clarabel.NonnegativeConeT(len(le_rows)))

    # Variable bounds
    n_var_bounds = 0
    for ci in range(n):
        lo, hi = var_bounds.get(ci, [0.0, np.inf])
        if lo > -np.inf:
            row = sparse.csc_matrix(([-1.0], ([0], [ci])), shape=(1, n))
            A_blocks.append(row)
            b_blocks.append(np.array([-lo]))
            cones.append(clarabel.NonnegativeConeT(1))
            n_var_bounds += 1
        if hi < np.inf:
            row = sparse.csc_matrix(([1.0], ([0], [ci])), shape=(1, n))
            A_blocks.append(row)
            b_blocks.append(np.array([hi]))
            cones.append(clarabel.NonnegativeConeT(1))
            n_var_bounds += 1

    A = sparse.vstack(A_blocks).tocsc()
    b = np.concatenate(b_blocks)

    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 200

    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solution = solver.solve()

    return {
        'status': str(solution.status),
        'iterations': solution.iterations,
        'obj_val': solution.obj_val,
        'n': n,
        'm_conic': A.shape[0],
    }


def run_minix(problem_name):
    """Run Minix on a problem."""
    result = subprocess.run(
        ['cargo', 'run', '--release', '-p', 'solver-bench', '--',
         'maros-meszaros', '--problem', problem_name],
        capture_output=True, text=True,
        cwd='/Users/chris/code/minix'
    )

    output = result.stdout + result.stderr

    # Parse output
    status, iters, obj = "Unknown", 0, float('nan')
    for line in output.split('\n'):
        if 'Status:' in line:
            status = line.split(':')[1].strip()
        elif 'Iterations:' in line:
            iters = int(line.split(':')[1].strip())
        elif 'Objective:' in line:
            obj = float(line.split(':')[1].strip())

    return {
        'status': status,
        'iterations': iters,
        'obj_val': obj,
    }


if __name__ == '__main__':
    import sys
    problem = sys.argv[1] if len(sys.argv) > 1 else 'YAO'
    qps_path = os.path.expanduser(f"~/.cache/minix-bench/maros-meszaros/{problem}.QPS")

    if not os.path.exists(qps_path):
        print(f"QPS file not found: {qps_path}")
        sys.exit(1)

    print(f"=== Comparing {problem} ===\n")

    print("Running Clarabel...")
    clarabel_result = run_clarabel(qps_path)

    print("Running Minix...")
    minix_result = run_minix(problem)

    print(f"\n{'Solver':<12} {'Status':<15} {'Iters':>6} {'Objective':>15}")
    print("-" * 50)
    print(f"{'Clarabel':<12} {clarabel_result['status']:<15} {clarabel_result['iterations']:>6} {clarabel_result['obj_val']:>15.6e}")
    print(f"{'Minix':<12} {minix_result['status']:<15} {minix_result['iterations']:>6} {minix_result['obj_val']:>15.6e}")

    # Compare
    obj_diff = abs(minix_result['obj_val'] - clarabel_result['obj_val'])
    obj_rel_diff = obj_diff / max(abs(clarabel_result['obj_val']), 1e-10)

    print(f"\nObjective difference: {obj_diff:.6e} (relative: {obj_rel_diff:.2%})")

    if minix_result['iterations'] < clarabel_result['iterations']:
        print(f"Minix is FASTER ({minix_result['iterations']} vs {clarabel_result['iterations']} iters)")
    elif minix_result['iterations'] > clarabel_result['iterations']:
        print(f"Clarabel is FASTER ({clarabel_result['iterations']} vs {minix_result['iterations']} iters)")
    else:
        print("Same iteration count")
