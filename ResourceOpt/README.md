# ResourceOpt: Continuous Production & Resource Allocation Under Constraints

ResourceOpt solves continuous resource allocation across production facilities using Lagrange multipliers, Karush-Kuhn-Tucker (KKT) optimality conditions, and Linear Independence Constraint Qualifications (LICQ).

---

## 1. Problem Formulation & Methodology

### Production Allocation Under Linear & Capacity Constraints
- **Objective:** Strictly convex quadratic production cost:
  $$\min_x f(x) = \frac{1}{2} x^T Q x + c^T x = \frac{1}{2} (4 x_1^2 + 6 x_2^2) - 20 x_1 - 30 x_2$$
- **Constraints:**
  - Production Target Equality: $x_1 + x_2 = b_{\text{target}}$
  - Machine Capacity Limit: $1.5 x_1 + x_2 \le 10.0$
  - Product 1 Storage Limit: $x_1 \le 6.0$
  - Non-Negativity: $x_1 \ge 0, x_2 \ge 0$
- **Theoretical Tools:**
  - Lagrange Multiplier Solver (direct block KKT linear system solver):
    $$\begin{bmatrix} Q & A^T \\ A & 0 \end{bmatrix} \begin{bmatrix} x \\ \lambda \end{bmatrix} = \begin{bmatrix} -c \\ b \end{bmatrix}$$
  - Karush-Kuhn-Tucker (KKT) Conditions Evaluator:
    1. Stationarity: $\|\nabla f(x) + A^T \lambda + G^T \mu\|_2 \le \text{tol}$
    2. Primal Feasibility: $\|A x - b\| \le \text{tol}$ and $G x - h \le \text{tol}$
    3. Dual Feasibility: $\mu_j \ge 0, \forall j$
    4. Complementary Slackness: $|\mu_j (G x - h)_j| \le \text{tol}, \forall j$
  - Linear Independence Constraint Qualification (LICQ) Checker:
    Evaluates rank and Gram determinant $\det(J_{\mathcal{A}} J_{\mathcal{A}}^T)$ of active constraint gradients at candidate solution $x$.

### The 5 Constrained Test Scenarios
1. **Case 1 (Feasible Interior Optimum):** Equality active, inequalities inactive with strictly positive slack. KKT satisfied, LICQ holds.
2. **Case 2 (Feasible Boundary Optimum):** Equality and machine capacity constraint simultaneously binding. Dual multiplier $\mu_1 = 48.0 > 0$, stationarity satisfied, LICQ holds.
3. **Case 3 (Infeasible Candidate Point):** Point $[6.5, 3.5]$ violates storage bound $x_1 \le 6.0$. Correctly rejected as infeasible.
4. **Case 4 (Linearly Independent Active Constraints):** Verified full row-rank active constraint Jacobian matrix ($\text{rank} = 2$, $\det(J_{\mathcal{A}} J_{\mathcal{A}}^T) > 0$).
5. **Case 5 (Linearly Dependent / Redundant Constraints):** Introduces duplicate constraint $2 x_1 + 2 x_2 = 18.0$. LICQ checker correctly identifies rank deficiency ($\text{rank} = 1 < 2$, $\text{Gram Det} = 0.0$) and rejects LICQ.

---

## 2. Directory Structure

```
ResourceOpt/
├── main.py                         # Interactive CLI & runner
├── README.md                       # Project documentation
├── problems/
│   └── production_allocation.py    # Constrained formulation and KKT/LICQ checks
├── experiments/
│   └── run_experiments.py         # End-to-end experiment pipeline
├── visualization/
│   └── plots.py                    # 2D feasible polytope & KKT diagnostic bar charts
├── results/
│   ├── plots/                      # Generated figures (.png)
│   ├── histories/                  # Run artifacts
│   └── reports/                    # Benchmark report (.txt)
└── tests/
    └── test_resource.py           # Automated unit tests
```

---

## 3. Running ResourceOpt

```bash
# Run experiments and generate plots/reports:
python main.py --all

# Run automated unit tests:
python main.py --test
```
