# SparseModel: Sparse Predictive Modeling & Subgradient Optimization

SparseModel investigates nonsmooth convex optimization, $L_1$ penalty subdifferentials, and feature selection via Lasso regularization.

---

## 1. Problem Formulation & Methodology

### Lasso Regression for High-Dimensional Resource Modeling
- **Objective:**
  $$\min_w J(w) = \frac{1}{2n} \|Xw - y\|_2^2 + \lambda \|w\|_1$$
- **Subgradient Formulation:**
  The subdifferential of the absolute value function is:
  $$\partial |w_j| = \begin{cases} \{1\} & \text{if } w_j > 0 \\ \{-1\} & \text{if } w_j < 0 \\ [-1, 1] & \text{if } w_j = 0 \end{cases}$$
  The subgradient of the composite Lasso objective is:
  $$g(w) = \frac{1}{n} X^T (Xw - y) + \lambda \cdot v, \quad v_j = \begin{cases} \text{sign}(w_j) & w_j \neq 0 \\ 0 & w_j = 0 \end{cases}$$
- **Optimization Algorithms:**
  - Subgradient Method for General Nonsmooth Optimization ($f(x) = \|x\|_2$) with diminishing step size $\alpha_k = \frac{a}{\sqrt{k}}$.
  - Subgradient Method for Lasso Loss.
- **Regularization Path Analysis:** Sweeps $\lambda \in [0.001, 0.5]$ to demonstrate coefficient shrinkage and feature selection compared to unregularized Ordinary Least Squares (OLS) baselines.

---

## 2. Directory Structure

```
SparseModel/
├── main.py                     # Interactive CLI & runner
├── README.md                   # Project documentation
├── data/
│   └── synthetic_sparse.py     # Multi-feature sparse dataset generator
├── problems/
│   └── lasso_formulation.py    # Lasso loss and subgradient routines
├── experiments/
│   └── run_experiments.py     # End-to-end experiment pipeline
├── visualization/
│   └── plots.py                # Regularization paths, stem charts, loss curves
├── results/
│   ├── plots/                  # Generated figures (.png)
│   ├── histories/              # Optimization epoch logs (.csv)
│   └── reports/                # Benchmark report (.txt)
└── tests/
    └── test_sparse.py          # Automated unit tests
```

---

## 3. Running SparseModel

```bash
# Run experiments and generate plots/reports:
python main.py --all

# Run automated unit tests:
python main.py --test
```
