# MOSEK-style improvements worth copying (concrete + implementable)

This note focuses on *implementation-level* techniques that MOSEK (and other top-tier commercial IPM solvers) publicly indicate they use, and that typically explain most of the *wall-clock* gap vs. “academic” conic IPMs built around QDLDL + AMD.

It also ships a **stand-alone Rust bundle** you can incorporate:
- dense/offending column detection (simple but tunable),
- Woodbury low-rank update solve `(S + U U^T) y = b`,
- CHOLMOD (SuiteSparse) SPD solver skeleton,
- METIS NodeND ordering skeleton,
- demo example.

Files:
- `mosek_standalone_linalg_bundle.txt`
- `split_mosek_standalone_linalg_bundle.py`

---

## 1) What MOSEK publicly signals matters most

### 1. Solve form selection + dualization heuristic
- MOSEK exposes a parameter controlling whether it solves the primal or dual (“solve form”), and their nonsymmetric conic talk says the implementation *dualizes if beneficial for linear algebra*.

**Action for MINIX**
- Add `SolveForm::{Auto,Primal,Dual}` and a heuristic for choosing (n vs m, sparsity, cone mix).
- Make this a *first-class cached decision* (you don’t want to rebuild structures every iteration).

### 2. Ordering: beyond AMD (graph partitioning, multiple seeds)
- MOSEK exposes multiple ordering methods including graph-partitioning based orderings, and a “#seeds” knob for GP.
- MOSEK’s dense column slides explicitly mention AMD or GP ordering and that pivot order is fixed over iterations.

**Action for MINIX**
- Add a graph-partition ordering backend (METIS NodeND / SCOTCH).
- Try a few seeds (small fixed set) and select best by symbolic stats (nnz(L), flop estimate).

### 3. Dense/offending column detection + special handling
- MOSEK has a knob `INTPNT_OFF_COL_TRH` controlling how aggressively it finds “offending columns” in the Jacobian/constraint matrix.
- Dense columns can dominate fill and factor time; isolate them and handle via low-rank updates / modified Schur complements.

**Action for MINIX**
- Detect dense columns (don’t rely only on a fixed nnz threshold).
- Split A = [A_sparse, A_dense].
- In normal-equations form, represent dense contribution as U U^T and solve via Woodbury/product-form instead of polluting the sparse factorization.

### 4. Efficient handling of bounds + fixed variables
- MOSEK explicitly calls out “simple bounds and fixed variables handled efficiently in linear algebra”.

**Action for MINIX**
- Do not add bounds as extra rows when possible.
- Eliminate fixed vars in presolve.
- Represent bounds as diagonal barrier contributions / diagonal blocks.

### 5. Multiple correctors, different primal/dual step sizes, path-following knob
- MOSEK exposes:
  - multiple corrector limit,
  - whether different primal/dual step sizes are allowed,
  - “central path” tolerance.

**Action for MINIX**
- Add multi-corrector support (keep off by default until linear algebra is fast).
- Allow separate alpha_p / alpha_d.
- Add a “path following” knob that biases sigma/centering.

---

## 2) Why these are directly relevant to your failures

BOYD1: gap and primal feasibility are tiny, but dual residual stalls ⇒ this is exactly where:
- better ordering + dense column handling + better linear solve accuracy
tends to unlock the last digits.

BOYD2: huge constraints and bounds dominate ⇒ bounds-as-rows explodes the KKT and worsens conditioning; solve-form selection + bound handling are key.

---

## 3) What’s included in the stand-alone bundle

1) `DenseColumnDetector`  
2) `build_sparse_trips_and_u()` that:
   - emits upper-triangle triplets for S_sparse = A_sparse D A_sparse^T
   - builds sparse vectors for U = sqrt(D_dense) * A_dense
3) `WoodburySolver` that solves `(S_sparse + U U^T) y = b` given any SPD base solver
4) `CholmodSpdSolver` skeleton (SuiteSparse CHOLMOD)
5) `MetisOrdering` skeleton (METIS NodeND)

The intent is: you can use this code to prototype MOSEK-like dense-column handling inside a normal-equations SPD path (dual or primal), and later tighten the detector and add symbolic ordering selection.

---

## 4) How to use

```bash
python split_mosek_standalone_linalg_bundle.py mosek_standalone_linalg_bundle.txt ./out
cd out/mosek_linalg_extras
cargo test
cargo run --example woodbury_demo
```

To enable CHOLMOD or METIS bindings:

```bash
cargo test --features suitesparse
cargo test --features metis
```

You will need SuiteSparse / METIS installed and visible to the linker.

---

## 5) Expectations

This bundle won’t magically match MOSEK by itself, but it is the “missing engineering” that typically creates a **large** part of the wall-clock delta:
- solve-form selection,
- GP ordering,
- dense column handling,
- supernodal Cholesky (CHOLMOD) for SPD paths.

