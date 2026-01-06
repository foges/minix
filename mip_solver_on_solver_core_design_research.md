# Mixed-Integer Solver Layer on top of `solver-core` (Rust)
Version: **0.2 (research-updated draft)**  
Date: **2026-01-03**

> **Goal:** implement a **production-grade mixed-integer solver** that supports **MILP / MIQP / MISOCP (v1)** and extends to **MIEXP / MIPOW / (eventually) MISDP**, by building a discrete layer (`solver-mip`) on top of the continuous conic-QP engine (`solver-core`).  
> The primary performance target is to be **competitive with modern commercial MISOCP / MIQP workflows**, by combining:
>
> - a fast, scalable **branch-and-bound / branch-and-cut** engine, and
> - a **conic-certificate outer-approximation** (OA) strategy using **\(\mathcal{K}^*\) cuts** (Pajarito-style) and **cone disaggregation / lifted relaxations**.

This document is intentionally **method-heavy** and contains concrete formulas and implementation contracts aimed at a seasoned developer who is not necessarily a convex/MIP specialist.

---

## 0. Executive summary

We implement `solver-mip` with **two complementary modes** that share the same data model and search machinery:

### Mode A — Conic Branch-and-Bound (baseline, correctness + small/medium instances)
- At each node, solve the **full continuous relaxation** (with all cones) using `solver-core`.
- Branch on fractional integer variables.
- Use warm starts for `(x,s,z,τ,κ)` to reduce interior-point iterations between parent/child nodes.
- **Pros:** simplest, minimal additional infrastructure.  
- **Cons:** too expensive for large MIP trees; unlikely to approach Gurobi-level node throughput.

### Mode B — Branch-and-Cut with Conic-Certificate Outer Approximation (main path to “Gurobi/MOSEK-class”)
- Maintain a **polyhedral (LP/QP) master** containing:
  - the linear constraints, variable bounds, integrality,
  - plus an evolving cut set that outer-approximates the nonpolyhedral cones.
- When the master finds an **integer candidate**, call `solver-core` only as a **conic oracle**:
  - if feasible: accept/polish, update incumbent,
  - if infeasible: extract a **\(\mathcal{K}^*\) certificate** and add a cut,
  - optionally: generate additional strengthening cuts.
- This matches the most successful open literature pattern for mixed-integer conic problems (Pajarito and follow-ups), and aligns with how commercial solvers implement many MISOCP features (outer approximation + disaggregation + selective barrier solves).

**Key research-grade ingredients that must be designed in from the start**
1. **\(\mathcal{K}^*\) (dual-cone) certificate cuts** from conic subproblems (global OA).  
2. **Disaggregated \(\mathcal{K}^*\) cuts** for product cones (tighter master relaxations).  
3. **Cone disaggregation / lifted formulations** for SOC/QC constraints to avoid an explosion of tangent cuts.  
4. (Roadmap) **Conic MIR (CMIR) cuts**, **perspective cuts**, and **disjunctive conic cuts** for stronger convexification when integrality is present.

---

## 1. Scope and target problem classes

### 1.1 Canonical mixed-integer conic QP form (internal)

We reuse `solver-core`’s internal canonical form, with integrality and bounds:
\[
\begin{aligned}
\min_{x,s}\;& \frac12 x^\top P x + q^\top x\\
\text{s.t.}\;& A x + s = b,\quad s\in \mathcal{K}\\
& \ell \le x \le u\\
& x_i \in \mathbb{Z}\ \text{for } i\in \mathcal{I},\quad x_i\in\{0,1\}\ \text{for } i\in \mathcal{B}.
\end{aligned}
\]

- \(P\succeq 0\) (possibly singular); \(A\) sparse.
- \(\mathcal{K}\) is a product of cone blocks: \( \mathcal{K} = \mathcal{K}_1 \times \cdots \times \mathcal{K}_B\).

### 1.2 v1 deliverables
- **MILP / MIQP / MISOCP** with robust statuses, deterministic defaults.
- OA-based MISOCP should be the primary “serious” performance path.
- Optional MIEXP/MIPOW supported in the OA oracle (integer feasibility checking), but do **not** expect commercial-level speed until master backend + cut generation matures.

### 1.3 Non-goals (v1)
- Full cut arsenal comparable to Gurobi/CPLEX for MILP (that’s multi-year).
- Global nonlinear branching/cutting for arbitrary MINLP outside conic representability.
- MISDP as anything but “experimental” (unless block structure is very favorable).

---

## 2. Architecture overview

### 2.1 Crates and module responsibilities
- `solver-core` (already designed): continuous conic-QP PDIPM + HSDE certificates.
- `solver-mip` (new):
  - `model`: mixed-integer problem wrapper, integrality, bounds, cone metadata
  - `search`: B&B tree, node queue, branching, incumbent handling
  - `master`: master relaxation model + backend interface
  - `oracle`: conic subproblem “validator” + certificate extraction using `solver-core`
  - `cuts`:
    - `kstar`: \(\mathcal{K}^*\) certificate cuts (global + disaggregated)
    - `soc`: SOC tangent/lifted cuts + cone disaggregation
    - `mip`: generic MILP cuts (later)
    - `convex`: objective cuts if we choose MILP master for MIQP (optional)
  - `propagation`: bound tightening, implied bounds, probing (later)
  - `heuristics`: rounding, diving, feasibility pump, RINS/local branching (staged)

### 2.2 Key abstraction: master backend interface

`solver-mip` must not hard-code a single LP/QP solver. The cut loop only works if we can add cuts and reopt cheaply. Define a narrow trait:

```rust
pub trait MasterBackend {
    type Model;
    type SolveInfo;

    fn new_model(&self, base: &MasterBaseModel) -> Self::Model;

    fn add_cut(&self, model: &mut Self::Model, cut: &LinearCut) -> CutId;

    fn add_lazy_cut(&self, model: &mut Self::Model, cut: &LinearCut) -> CutId;

    fn tighten_var_bound(&self, model: &mut Self::Model, var: VarId, lb: Option<f64>, ub: Option<f64>);

    fn solve_relaxation(&self, model: &mut Self::Model) -> Result<MasterSol, MasterStatus>;

    fn get_basis_warm_start(&self, model: &Self::Model) -> Option<BasisWarmStart>; // v0.3+
    fn set_basis_warm_start(&self, model: &mut Self::Model, ws: &BasisWarmStart); // v0.3+
}
```

Planned implementations:
- **`IpmMasterBackend` (v0.1–v0.2):** use `solver-core` with only `{Zero, NonNeg}` cones (LP/QP) for correctness.
- **`SimplexBackend` (v0.3+):** internal dual simplex for LP; critical for MILP speed.
- **`ExternalBackend` (optional feature flag):** bind to HiGHS (LP/MIP) for early performance baselining and to validate cut strategies.

---

## 3. Algorithm: Mode A (Conic B&B) baseline

### 3.1 Node relaxation solve
At a node, we tighten bounds for branched integer vars and solve:

\[
\min \tfrac12 x^\top P x + q^\top x\quad
\text{s.t.}\quad A x + s=b,\ s\in\mathcal{K},\ \ell^{(node)}\le x\le u^{(node)}
\]

with integrality relaxed.

### 3.2 Branching
- Pick variable with largest fractional part (v0.1), then move to:
  - pseudocost branching,
  - reliability branching,
  - strong branching on a small candidate list (later).

### 3.3 Why this mode matters
- It is a *reference* implementation to validate correctness of:
  - integrality handling,
  - warm starts,
  - termination + gap calculations,
  - and cone kernel correctness under repeated solves.

But it will not scale to large MIP trees.

---

## 4. Algorithm: Mode B (Conic-Certificate OA Branch-and-Cut)

This is the primary “top-of-the-line” MISOCP/MICP strategy.

### 4.1 Master problem definition

Let \(x\) be the original variables. The conic constraints are replaced by a growing set of linear cuts:
\[
\mathcal{P}_0 := \{x:\ \text{linear constraints + bounds}\}
\]
and at iteration \(k\):
\[
\mathcal{P}_k := \mathcal{P}_0 \cap \bigcap_{c\in \mathcal{C}_k} c(x)
\]
where each cut \(c(x)\) is a valid inequality implied by \(\mathcal{K}\).

The master is then:
- **MILP** if \(P=0\),
- **MIQP** if \(P\neq 0\),
- or optionally **MILP** with an epigraph variable and objective cuts (if you deliberately avoid MIQP in the master).

### 4.2 High-level OA loop (single-tree “lazy constraint” form)

We run one B&B tree on the master. When we reach an integer candidate \(x^I\), we validate with the conic oracle:

```text
Initialize master with linear constraints + (optional) initial SOC/QC relaxations

Run branch-and-bound on master:
  at each node: solve LP/QP relaxation

  when an integer-feasible master solution x^I is found:
      (feasible?, cert, x_cont, z_dual) = conic_oracle_check(x^I)
      if feasible:
          update incumbent using exact objective (and optional polishing)
          continue search (prune by bound)
      else:
          generate one or more K* cuts from cert (optionally disaggregated)
          add as lazy constraints
          re-solve (the master will exclude x^I and similar infeasible points)
```

Key: the master must remain an **outer approximation** (a relaxation) so its bound is valid.

### 4.3 Conic oracle contract (uses `solver-core`)

The oracle solves the continuous conic subproblem with integers fixed:

\[
\min \tfrac12 x^\top P x + q^\top x \quad \text{s.t. } A x + s = b,\ s\in\mathcal{K},\ x_i = x_i^I \ \forall i\in\mathcal{I}\cup\mathcal{B}.
\]

Outputs:
- `Feasible { x*, obj, dual z* }`, or
- `Infeasible { certificate y }`, where \(y\in \mathcal{K}^*\) can generate a separating cut.

**Important:** this requires `solver-core` to expose *usable* HSDE certificates (already part of the continuous design).

---

## 5. \(\mathcal{K}^*\) certificate cuts (the core research method)

### 5.1 The cut formula (general cone, all types)

From the canonical constraint:
\[
A x + s = b,\quad s\in\mathcal{K}
\]
we have \(s = b - A x\). For any \(y \in \mathcal{K}^*\) (dual cone), the definition of dual cone gives:
\[
y^\top s \;\ge\; 0\quad \forall s\in\mathcal{K}.
\]

Therefore any \(y\in\mathcal{K}^*\) yields a valid inequality:
\[
y^\top (b - A x) \ge 0.
\]

Rewrite as a standard linear cut:
\[
(A^\top y)^\top x \;\le\; b^\top y.
\]

So the implementation is trivial once you have \(y\):
- `a = A^T y`
- `rhs = b^T y`
- cut is `a^T x <= rhs`

This is the canonical **\(\mathcal{K}^*\) cut**.

### 5.2 Where to get \(y\) (practical hierarchy)

**Best (global) source — from conic subproblem certificates**
- If the fixed-integer conic subproblem is infeasible, `solver-core` should return a dual certificate \(y\in\mathcal{K}^*\) that separates \(x^I\). This is the most reliable way to cut off infeasible integer assignments.

**Also strong — from optimal dual solutions**
- If the subproblem is feasible/optimal, its dual variable \(z^*\in\mathcal{K}^*\) is often a good cut direction. You can add:
  \[
  (A^\top z^*)^\top x \le b^\top z^*
  \]
  if it is violated by the current master point (common at fractional nodes / root).

**Heuristic/cheap — projection-based separation on violated blocks**
- For SOC (and some PSD cases), you can compute a separating \(y\) cheaply from the violated slack \(s^*\) by projecting \(-s^*\) onto \(\mathcal{K}^*\) and normalizing.
- For EXP/POW, projection is still possible but typically needs a small root-finding loop; use it selectively (root only) unless profiling says it’s worth it.

### 5.3 Disaggregated \(\mathcal{K}^*\) cuts for product cones

If \(\mathcal{K}=\mathcal{K}_1\times\cdots\times\mathcal{K}_B\), then \(\mathcal{K}^*=\mathcal{K}_1^*\times\cdots\times\mathcal{K}_B^*\).  
So any certificate \(y\in\mathcal{K}^*\) decomposes as \(y=(y_1,\dots,y_B)\) with \(y_b\in\mathcal{K}_b^*\).

Let the constraint rows be partitioned accordingly:
- \(A = \begin{bmatrix} A_1 \\ \vdots \\ A_B \end{bmatrix}\),
- \(b = \begin{bmatrix} b_1 \\ \vdots \\ b_B \end{bmatrix}\).

Then each block gives its own valid cut:
\[
y_b^\top (b_b - A_b x) \ge 0
\quad\Longleftrightarrow\quad
(A_b^\top y_b)^\top x \le b_b^\top y_b.
\]

**Practical rule:** when you get a full certificate \(y\), generate *multiple disaggregated cuts* by selecting blocks with largest violation:
\[
\mathrm{viol}_b := -y_b^\top (b_b - A_b x^{cand}).
\]
Add the top-k violated disaggregated cuts (k=1..B, cap by budget).

This is one of the simplest “high leverage” upgrades to OA performance.

### 5.4 Cut normalization and numerics (mandatory)

Cuts from conic certificates can be numerically extreme. Always normalize.

Recommended normalization:
- If `a = A^T y` is not all zeros, scale so:
  \[
  \|a\|_\infty = 1.
  \]
  i.e. divide `(a, rhs)` by `max_i |a_i|`.

Additional safeguards:
- Drop cuts with tiny `||a||` (e.g. `||a||_∞ < 1e-12`) unless they cut off the point by a large margin.
- Compute violation with a feasibility tolerance `tol_cut` and only add if:
  \[
  a^\top x^{cand} > rhs + tol\_cut.
  \]
- Maintain a “duplicate cut” filter (hash rounded coefficients + rhs).

### 5.5 “Lifted” \(\mathcal{K}^*\) cuts for SOC/QC (important for speed)

A recurring empirical issue: high-dimensional SOC constraints can require *many* tangent cuts to approximate well (in the worst case, exponentially many).

Two complementary mitigations:

#### (A) Cone disaggregation (separable formulation)
For a Lorentz/SOC constraint:
\[
t \ge \|x\|_2,\quad t\ge 0
\]
introduce auxiliary variables \(y_i \ge 0\) and enforce:
\[
\sum_i y_i \le t,\qquad x_i^2 \le y_i\,t \quad \forall i.
\]
Each \(x_i^2 \le y_i t\) is a **3D rotated SOC** constraint. In an OA framework, this can be approximated much more efficiently because each small constraint yields strong cuts and “focuses” the approximation.

This is closely related to “cone disaggregation” strategies used in practice.

#### (B) Lifted polyhedral relaxations (Ben-Tal/Nemirovski; Vielma-style)
For SOC constraints, you can build a **lifted LP relaxation** with additional variables that provides a tunable approximation quality \(\varepsilon\), often with size scaling like \(O(n\log(1/\varepsilon))\) rather than exploding with dimension.

Implementation plan:
- Provide `SocLiftedRelaxationBuilder { eps, budget }` that expands a SOC block into:
  - extra variables,
  - a polyhedral outer approximation.
- This is used to seed the root master before any dynamic cutting.

This is optional in v1 but strongly recommended for competitive MISOCP.

---

## 6. SOC OA cuts: concrete formulas you can implement today

Even with certificate cuts, SOC is common enough that it deserves explicit “fast path” cuts.

### 6.1 Standard SOC tangent cut

Constraint: \((t,x)\in \mathcal{L}_d\) where \(\mathcal{L}_d = \{(t,x): t\ge \|x\|\}\).

Given a violating point \((t^*, x^*)\) with \(\|x^*\| > t^*\):
- compute \(u = x^*/\|x^*\|\)
- add cut:
\[
u^\top x \le t.
\]

Interpretation as a \(\mathcal{K}^*\) cut:
- dual cone is itself, \(\mathcal{L}_d^*=\mathcal{L}_d\),
- choose \(y=(1,\,-u)\in\mathcal{L}_d\),
- then \(y^\top (t^*,x^*) = t^* - u^\top x^* < 0\), so it separates.

### 6.2 Rotated SOC tangent cut

Rotated SOC: \((u,v,w)\in \mathcal{R}_d\) with \(2uv \ge \|w\|^2,\ u\ge 0,\ v\ge 0\).

In practice, prefer to transform RSOC → SOC via a linear map (as in `solver-core`) and reuse SOC separation.

---

## 7. EXP/POW OA: what “top-of-line” looks like

For EXP/POW, geometric tangent cuts exist, but the most reliable approach is still **certificate-based**: solve a conic subproblem (or a tiny separation problem per 3D block) and extract \(y\in \mathcal{K}^*\).

Implementation guidance:
- For **MIEXP/MIPOW**, do not attempt to handcraft a large set of tangents.
- Instead:
  1) validate integer candidate with conic oracle,
  2) use returned \(y\) to build \(\mathcal{K}^*\) cut(s),
  3) disaggregate per 3D cone block (often very effective).

---

## 8. Branching, bounds, and search policy

### 8.1 Node selection
- v0.1: best-bound.
- v0.2: best-bound with depth bias (helps find incumbents).
- v0.3+: hybrid best-bound + “plunge” for feasibility (industry standard).

### 8.2 Variable selection
- v0.1: most fractional.
- v0.2: pseudocost branching.
- v0.3: reliability branching:
  - strong branch on small candidate set at shallow depths,
  - fall back to pseudocost deeper.

### 8.3 Strong branching cost control
Strong branching requires solving 2 extra relaxations per tested variable. Keep it cheap by:
- limiting to root and early depths,
- early stopping relaxations (cap iterations / loose tolerances),
- caching relaxation warm starts.

---

## 9. Primal heuristics and incumbent polishing

Heuristics are mandatory for good performance.

### 9.1 Rounding + repair (v0.1)
- Round integer/binary vars.
- Fix them; call conic oracle (continuous solve) to repair feasibility and compute objective.
- If feasible: incumbent.

### 9.2 Diving (v0.1)
- Fix a few fractional vars and re-solve master relaxation repeatedly (depth-first).
- At each dive completion, call conic oracle.

### 9.3 Feasibility pump (v0.2)
- Alternate between:
  - rounding to satisfy integrality,
  - solving a relaxation to satisfy constraints.
- For conic problems, use master relaxation in the pump loop, and conic oracle check at the end.

### 9.4 RINS / local branching / polishing (v0.3)
- Once you have an incumbent, fix most vars to incumbent values with a neighborhood radius and search for improvements.
- Use conic oracle to polish final solution at the end.

---

## 10. Cut generation roadmap (beyond \(\mathcal{K}^*\) cuts)

### 10.1 Generic MILP cuts (requires LP basis)
Once `SimplexBackend` exists, add a standard cut loop:
- Gomory mixed-integer (GMI)
- mixed-integer rounding (MIR)
- cover cuts / knapsack cuts
- implied bound cuts

These are necessary for MILP competitiveness, but are orthogonal to conic OA.

### 10.2 Perspective reformulations and cuts (MIQP strength boost)
For MIQP models with indicator-like structure (binary variables turning quadratic pieces on/off), use **perspective reformulations**:
- detect patterns like \(z\in\{0,1\}\), \(x=0\) when \(z=0\), and a convex quadratic cost/constraint on \(x\) when \(z=1\),
- apply projected perspective inequalities / cuts to tighten the relaxation.

This is one of the most effective “research-to-practice” techniques in MIQP.

### 10.3 Conic mixed-integer rounding (CMIR) cuts for SOC (advanced)
CMIR cuts (Atamtürk–Narayanan) strengthen SOC MIPs by exploiting polyhedral substructures of SOC sets and applying rounding.
Design plan:
- implement CMIR as an optional cut generator for recognized SOC patterns,
- start with common application templates:
  - mean-variance / portfolio,
  - least squares with binary inputs,
- validate with gap improvement benchmarks before enabling by default.

### 10.4 Disjunctive conic cuts (very advanced, roadmap)
There is deep literature on disjunctive cuts for SOC (split cuts, two-term disjunctions, etc.). Implementing them well is nontrivial and should be postponed until:
- the master LP backend is mature,
- and core OA is stable.

---

## 11. Caching, warm starts, and reuse (critical for performance)

### 11.1 Master reoptimization
- With simplex: basis warm starts after adding cuts are the main speed lever.
- With IPM (temporary): reuse KKT symbolic + warm start primal/dual if possible, but expect limited benefit.

### 11.2 Conic oracle warm starts
When validating multiple nearby integer candidates:
- warm-start the conic solve with previous feasible `(x,s,z,τ,κ)` projected onto fixed-integer constraints,
- reuse scaling objects where safe.

### 11.3 Cut pools and aging
- Maintain a global cut pool.
- Track:
  - activity count,
  - last violation,
  - efficacy score (violation / norm).
- Periodically retire stale cuts from the active master model (but keep in pool for possible re-activation).

---

## 12. Testing and benchmarking

### 12.1 Correctness tests
- unit tests for:
  - cut generation (`a = A^T y`, `rhs = b^T y`),
  - disaggregation logic,
  - normalization and duplicate detection,
  - oracle feasibility decisions vs `solver-core` statuses.

### 12.2 Benchmark suites (minimum)
- MIPLIB (MILP/MIQP core performance)
- CBLIB / conic benchmark sets (MISOCP, mixed cones)
- Portfolio + robust regression canonical problems generated at scale
- CVXPY mixed-integer test suite (integration-level)

### 12.3 Performance reporting
- adopt **performance profiles** (Dolan–Moré) for:
  - time-to-solution,
  - nodes explored,
  - gap closed at fixed times (e.g., 10s, 60s).

---

## 13. Implementation milestones (order matters)

1. **Baseline conic B&B** (Mode A): correctness + warm starts.
2. **Certificate OA as lazy cuts** (Mode B) with `IpmMasterBackend`:
   - validate that K* cuts converge and solve MISOCP instances.
3. **Disaggregated K* cuts + SOC disaggregation**:
   - major practical improvement, low code complexity.
4. **External LP master backend (optional)** to baseline speed quickly.
5. **Internal simplex backend** (dual simplex LP) for long-term competitiveness.
6. Add **heuristics + propagation**.
7. Add **perspective / CMIR cuts** for MIQP/SOC strength.
8. GPU path: mostly impacts `solver-core`; MIP layer benefits indirectly.

---

## 14. References (starting point)

- Coey, Lubin, Vielma. *Outer Approximation With Conic Certificates For Mixed-Integer Convex Problems* (2018/2019).  
- Pajarito.jl (JuMP / Julia): open-source implementation of conic-certificate OA.  
- Vielma. *A Lifted Linear Programming Branch-and-Bound Algorithm for Mixed-Integer Conic Quadratic Programs* (2007/2008).  
- Dunning et al. *Extended Formulations in Mixed Integer Conic Quadratic Programming* (2016).  
- Atamtürk, Narayanan. *Conic Mixed-Integer Rounding Cuts* (Mathematical Programming, 2010).  
- Günlük, Linderoth and/or Frangioni–Gentile line of work on **perspective reformulations** for MIQP (2009–2017).  
- Literature on disjunctive/SOC cuts (Kılınç-Karzan, Yıldız; Belotti et al.) for roadmap items.

---

End of document.
