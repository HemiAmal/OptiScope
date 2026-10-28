# EnergyOpt: Electricity Demand Prediction & Operational Resource Allocation

EnergyOpt is an optimization engineering system solving hourly electricity demand forecasting and continuous economic generation dispatch.

---

## 1. Problem Formulation & Methodology

### Hourly Electricity Demand Forecasting
- **Objective:** Minimize Mean Squared Error (MSE) on hourly load data:
  $$\min_{w, b} J(w, b) = \frac{1}{2n} \sum_{i=1}^n (x_i^T w + b - y_i)^2$$
- **Dataset Features:** Diurnal hour-of-day cycles, ambient temperature, quadratic cooling/heating responses, weekend commercial reductions, and autoregressive lag indicators.
- **Optimization Algorithms:**
  - Batch Gradient Descent
  - Stochastic Gradient Descent (SGD)
  - Mini-batch SGD ($m=32$)
  - AdaGrad
  - RMSProp
  - Adam

### Operational Resource Allocation (Economic Dispatch)
- **Objective:** Minimize continuous quadratic power generation production costs across 3 generation facilities:
  $$\min_{P} C(P) = \sum_{i=1}^3 \left(\frac{1}{2} a_i P_i^2 + b_i P_i\right) = \frac{1}{2} P^T Q P + c^T P$$
- **Constraints:**
  - Demand Balance (Equality): $\sum_{i=1}^3 P_i = D_{\text{pred}}$
  - Generator Capacity Limits (Inequalities): $P_i^{\min} \le P_i \le P_i^{\max}, \quad \forall i \in \{1, 2, 3\}$
- **Constrained Optimization & Verification:**
  - Lagrange Multiplier Solver (direct block KKT solution for equality systems)
  - Karush-Kuhn-Tucker (KKT) Conditions Evaluator
  - Linear Independence Constraint Qualification (LICQ) Checker

---

## 2. Project Directory Structure

```
EnergyOpt/
├── main.py                     # Interactive CLI and experiment runner
├── README.md                   # Project documentation
├── data/
│   └── synthetic_energy.py     # Hourly demand generator with zero-leakage normalization
├── problems/
│   ├── demand_regression.py    # Regression loss objectives & gradient computation
│   └── generation_dispatch.py  # Economic dispatch formulation & KKT verification
├── experiments/
│   └── run_experiments.py     # End-to-end experiment pipeline
├── visualization/
│   └── plots.py                # Visualizations for demand fits, loss curves & dispatch
├── results/
│   ├── plots/                  # Generated figures (.png)
│   ├── histories/              # Optimization epoch logs (.csv)
│   └── reports/                # Summary metrics (.txt)
└── tests/
    └── test_energy.py          # Automated unit test suite
```

---

## 3. Running EnergyOpt

```bash
# Run experiments and generate plots/reports:
python main.py --all

# Run automated unit tests:
python main.py --test
```
