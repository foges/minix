#!/usr/bin/env python3
"""
Verify PIQP solution quality on problems we fail.
Check if PIQP is really solving to tight tolerances.
"""

import numpy as np
import piqp
from scipy import sparse
from pathlib import Path
import sys

def parse_qps_simple(filename):
    """Simple QPS parser for basic QP problems."""
    # This is a simplified parser - for a full implementation we'd need more
    # For now, let's just test with a simple problem we can construct
    pass

def compute_residuals(P, q, A, b, x, y):
    """
    Compute primal and dual residuals for QP:
        min  0.5 x'Px + q'x
        s.t. l <= Ax <= u

    KKT conditions:
        Primal: Ax - s = 0, l <= s <= u
        Dual: Px + q + A'y = 0
        Complementarity: y_i * (Ax_i - s_i) = 0
    """
    n = len(x)
    m = len(b)

    # Primal residual: ||Ax - b||
    if A is not None:
        Ax = A @ x
        r_prim = np.linalg.norm(Ax - b)
        r_prim_rel = r_prim / max(1.0, np.linalg.norm(b))
    else:
        r_prim = 0.0
        r_prim_rel = 0.0

    # Dual residual: ||Px + q + A'y||
    if P is not None:
        Px = P @ x
    else:
        Px = np.zeros(n)

    if A is not None and y is not None:
        ATy = A.T @ y
    else:
        ATy = np.zeros(n)

    r_dual = Px + q + ATy
    r_dual_norm = np.linalg.norm(r_dual)
    r_dual_rel = r_dual_norm / max(1.0, max(np.linalg.norm(q), np.linalg.norm(Px)))

    return {
        'r_prim': r_prim,
        'r_prim_rel': r_prim_rel,
        'r_dual': r_dual_norm,
        'r_dual_rel': r_dual_rel,
        'r_dual_vec': r_dual  # Full vector for inspection
    }

def test_simple_qp():
    """Test PIQP on a simple QP."""
    print("=" * 70)
    print("Test 1: Simple QP")
    print("=" * 70)

    # Simple QP: min x^2 + y^2 subject to x + y = 1
    n = 2
    m = 1

    P = sparse.csc_matrix([[2.0, 0.0],
                            [0.0, 2.0]])
    c = np.array([0.0, 0.0])
    A = sparse.csc_matrix([[1.0, 1.0]])
    b = np.array([1.0])

    solver = piqp.SparseSolver()
    solver.settings.verbose = True
    solver.setup(P, c, A, b)

    status = solver.solve()

    print(f"\nPIQP Status: {status}")
    print(f"PIQP x: {solver.result.x}")
    print(f"PIQP y (dual): {solver.result.y}")
    print(f"PIQP objective: {solver.result.info.primal_obj:.6e}")
    print(f"PIQP iterations: {solver.result.info.iter}")

    # Check PIQP's tolerances
    print(f"\nPIQP Settings:")
    print(f"  eps_abs: {solver.settings.eps_abs}")
    print(f"  eps_rel: {solver.settings.eps_rel}")
    print(f"  eps_duality_gap_abs: {solver.settings.eps_duality_gap_abs}")
    print(f"  eps_duality_gap_rel: {solver.settings.eps_duality_gap_rel}")

    # Compute residuals manually
    res = compute_residuals(P.toarray(), c, A.toarray(), b, solver.result.x, solver.result.y)

    print(f"\nManual Residual Check:")
    print(f"  Primal residual (abs): {res['r_prim']:.6e}")
    print(f"  Primal residual (rel): {res['r_prim_rel']:.6e}")
    print(f"  Dual residual (abs): {res['r_dual']:.6e}")
    print(f"  Dual residual (rel): {res['r_dual_rel']:.6e}")

    # What would Minix consider acceptable?
    minix_tol_abs = 1e-8
    minix_tol_rel = 1e-8

    print(f"\nMinix tolerances (tol_feas = tol_gap = 1e-8):")
    print(f"  Would accept primal? {res['r_prim_rel'] <= minix_tol_rel}")
    print(f"  Would accept dual? {res['r_dual_rel'] <= minix_tol_rel}")

def test_ill_conditioned_qp():
    """Test PIQP on an ill-conditioned QP."""
    print("\n" + "=" * 70)
    print("Test 2: Ill-conditioned QP (like QFORPLAN)")
    print("=" * 70)

    # Create a QP with rank-deficient P (like QFORPLAN)
    n = 5
    m = 3

    # P is rank-deficient (only 2 non-zero eigenvalues)
    P_dense = np.zeros((n, n))
    P_dense[0, 0] = 1.0
    P_dense[1, 1] = 0.1
    # P[2:] are all zero (rank-deficient!)

    c = np.random.randn(n) * 10

    # Random constraint matrix
    np.random.seed(42)
    A_dense = np.random.randn(m, n)
    b = np.random.randn(m)

    print(f"Problem size: n={n}, m={m}")
    print(f"P rank: {np.linalg.matrix_rank(P_dense)} / {n}")
    print(f"P condition number: {np.linalg.cond(P_dense + np.eye(n) * 1e-10):.2e}")

    # Convert to sparse
    P = sparse.csc_matrix(P_dense)
    A = sparse.csc_matrix(A_dense)

    # Solve with PIQP
    solver = piqp.SparseSolver()
    solver.settings.verbose = False
    solver.setup(P, c, A, b)

    status = solver.solve()

    print(f"\nPIQP Status: {status}")
    print(f"PIQP iterations: {solver.result.info.iter}")
    print(f"PIQP primal obj: {solver.result.info.primal_obj:.6e}")
    print(f"PIQP dual obj: {solver.result.info.dual_obj:.6e}")
    print(f"PIQP duality gap: {abs(solver.result.info.primal_obj - solver.result.info.dual_obj):.6e}")

    # Check residuals
    res = compute_residuals(P_dense, c, A_dense, b, solver.result.x, solver.result.y)

    print(f"\nResiduals:")
    print(f"  Primal residual (rel): {res['r_prim_rel']:.6e}")
    print(f"  Dual residual (rel): {res['r_dual_rel']:.6e}")
    print(f"  Dual residual (abs): {res['r_dual']:.6e}")

    # Compare with Minix tolerances
    print(f"\nMinix acceptability (1e-8 rel tolerance):")
    print(f"  Primal: {'✓' if res['r_prim_rel'] <= 1e-8 else '✗'} ({res['r_prim_rel']:.2e})")
    print(f"  Dual: {'✓' if res['r_dual_rel'] <= 1e-8 else '✗'} ({res['r_dual_rel']:.2e})")

    # Show top dual residual components
    print(f"\nTop 5 dual residual components:")
    dual_res = res['r_dual_vec']
    top_indices = np.argsort(np.abs(dual_res))[::-1][:5]
    for i in top_indices:
        print(f"  x[{i}]: {dual_res[i]:.6e}")

def test_lp():
    """Test PIQP on an LP (P=0)."""
    print("\n" + "=" * 70)
    print("Test 3: Linear Program (P=0, like many Maros-Meszaros problems)")
    print("=" * 70)

    n = 10
    m = 5

    # LP: P is zero or very small
    P_dense = np.eye(n) * 1e-12  # Tiny regularization
    c = np.random.randn(n)

    np.random.seed(123)
    A_dense = np.random.randn(m, n)
    b = np.random.randn(m)

    print(f"Problem: LP with n={n}, m={m}")
    print(f"P max element: {np.max(np.abs(P_dense)):.2e}")

    P = sparse.csc_matrix(P_dense)
    A = sparse.csc_matrix(A_dense)

    solver = piqp.SparseSolver()
    solver.settings.verbose = False
    solver.setup(P, c, A, b)

    status = solver.solve()

    print(f"\nPIQP Status: {status}")
    print(f"PIQP iterations: {solver.result.info.iter}")

    res = compute_residuals(P_dense, c, A_dense, b, solver.result.x, solver.result.y)

    print(f"\nResiduals:")
    print(f"  Primal residual (rel): {res['r_prim_rel']:.6e}")
    print(f"  Dual residual (rel): {res['r_dual_rel']:.6e}")

    print(f"\nMinix acceptability:")
    print(f"  Primal: {'✓' if res['r_prim_rel'] <= 1e-8 else '✗'}")
    print(f"  Dual: {'✓' if res['r_dual_rel'] <= 1e-8 else '✗'}")

if __name__ == "__main__":
    test_simple_qp()
    test_ill_conditioned_qp()
    test_lp()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("These tests check if PIQP truly achieves tight tolerances")
    print("or if it's declaring success with looser criteria than Minix.")
