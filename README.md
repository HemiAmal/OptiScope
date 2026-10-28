# Optimization-Laboratory: Real-World Optimization Experimental Suite

Welcome to the **Optimization-Laboratory**, an advanced numerical optimization platform for **Optimization Techniques for Data Science (IMA314)**.

This repository provides five standalone, domain-driven optimization engineering projects powered by a centralized, handwritten mathematical optimization library (`shared_optimization`).

---

## 1. Implementation Principles & Numerical Architecture

- **Zero Black-Box Solvers:**
  No calls to `scipy.optimize.minimize`, `scipy.optimize.line_search`, `sklearn.linear_model`, or `torch.optim`.
- **Supporting Libraries:**
  - **NumPy (`np.ndarray`):** Vector and matrix representations, slicing, array transformations, and supporting linear algebra utilities.
  - **Matplotlib:** Publication-quality filled 2D contour maps, overlaid optimization paths, decision boundaries, ROC/threshold curves, and semi-logarithmic loss histories.
- **100% Handwritten Algorithms:**
  All 23 optimization algorithms, line searches, momentum updates, adaptive state updates, subgradient evaluations, Newton step applications, BFGS rank-2 updates, Lagrange multiplier solvers, KKT verifications, and LICQ checkers are implemented from scratch in `shared_optimization/`.

---

## 2. Master Algorithm-to-Problem Traceability Matrix (All 23 Methods)

| No. | Algorithm / Method | Project | Mathematical Objective & Role | Variables | Constraints | Implementation File | Experiment Runner | Test Suite | Output Artifacts |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Gradient Descent (Fixed Step) | **NonlinearLab** | $\min f(x)$, $x_{k+1} = x_k - \alpha \nabla f(x_k)$ | $x \in \mathbb{R}^2$ | Unconstrained | `shared_optimization/algorithms/gradient_descent.py` | `NonlinearLab/experiments/run_experiments.py` | `test_nonlinear.py` | `quad_fixed_gd.csv`, `quadratic_trajectories.png` |
| **2** | GD with Exact Line Search in 2D | **NonlinearLab** | $\alpha^* = \frac{g_k^T g_k}{g_k^T Q g_k}$ on quadratic | $x \in \mathbb{R}^2$ | Unconstrained (Quadratic) | `shared_optimization/algorithms/line_search.py` | `NonlinearLab/experiments/run_experiments.py` | `test_line_search.py` | `quad_exact_line_search_gd.csv` |
| **3** | Backtracking Line Search | **NonlinearLab** | Step contraction $\alpha \leftarrow \rho \alpha$ | $\alpha > 0$ | Bound $\alpha \ge \alpha_{\min}$ | `shared_optimization/algorithms/line_search.py` | `NonlinearLab/experiments/run_experiments.py` | `test_line_search.py` | Numerical trajectory logs |
| **4** | Armijo Acceptance Rule | **NonlinearLab** | $f(x+\alpha p) \le f(x) + c_1 \alpha \nabla f^T p$ | Condition evaluator | None | `shared_optimization/algorithms/line_search.py` | `NonlinearLab/experiments/run_experiments.py` | `test_line_search.py` | Step verification logs |
| **5** | GD with Armijo Backtracking | **NonlinearLab** | Adaptive step descent on Rosenbrock | $x \in \mathbb{R}^2$ | Unconstrained | `shared_optimization/algorithms/gradient_descent.py` | `NonlinearLab/experiments/run_experiments.py` | `test_nonlinear.py` | `rosen_armijo_gd.csv`, `rosenbrock_trajectories.png` |
| **6** | Momentum-based GD | **NonlinearLab** | Polyak Heavy-Ball: $v_{k+1} = \beta v_k + \alpha \nabla f$ | $x \in \mathbb{R}^2$ | Unconstrained | `shared_optimization/algorithms/momentum.py` | `NonlinearLab/experiments/run_experiments.py` | `test_nonlinear.py` | `quad_polyak_momentum.csv` |
| **7** | Nesterov GD (NGD) | **NonlinearLab** | Accelerated look-ahead: $\nabla f(x_k - \beta v_k)$ | $x \in \mathbb{R}^2$ | Unconstrained | `shared_optimization/algorithms/nesterov.py` | `NonlinearLab/experiments/run_experiments.py` | `test_nonlinear.py` | `quad_nesterov_gd.csv` |
| **8** | Batch GD for Linear Regression | **EnergyOpt** | Demand prediction MSE: $\frac{1}{2n}\|Xw - y\|_2^2$ | $w \in \mathbb{R}^5, b \in \mathbb{R}$ | Unconstrained | `shared_optimization/algorithms/regression_optimizers.py` | `EnergyOpt/experiments/run_experiments.py` | `test_energy.py` | `energy_reg_batch_gd.csv`, `regression_loss_curves.png` |
| **9** | Stochastic GD for Linear Reg. | **EnergyOpt** | Streaming demand updates: single sample | $w \in \mathbb{R}^5, b \in \mathbb{R}$ | Unconstrained | `shared_optimization/algorithms/regression_optimizers.py` | `EnergyOpt/experiments/run_experiments.py` | `test_energy.py` | `energy_reg_sgd.csv` |
| **10** | Mini-batch SGD for Linear Reg. | **EnergyOpt** | Subset demand updates ($m=32$) | $w \in \mathbb{R}^5, b \in \mathbb{R}$ | Unconstrained | `shared_optimization/algorithms/regression_optimizers.py` | `EnergyOpt/experiments/run_experiments.py` | `test_energy.py` | `energy_reg_mini_batch_sgd.csv`, `demand_predictions.png` |
| **11** | Logistic Regression | **PredictMaint** | Machinery failure BCE log-loss | $w \in \mathbb{R}^5, b \in \mathbb{R}$ | Unconstrained | `shared_optimization/algorithms/regression_optimizers.py` | `PredictMaint/experiments/run_experiments.py` | `test_maintenance.py` | `maint_opt_logistic_reg_gd.csv`, `decision_boundary_vibration_temp.png` |
| **12** | AdaGrad | **EnergyOpt** & **PredictMaint** | Historical squared grad scaling: $G_k = G_{k-1} + g^2$ | $w \in \mathbb{R}^d$ | Unconstrained | `shared_optimization/algorithms/adaptive_optimizers.py` | Respective project runners | `test_energy.py`, `test_maintenance.py` | Convergence histories & comparison tables |
| **13** | RMSProp | **EnergyOpt** & **PredictMaint** | Moving average squared scaling: $v = \beta v + (1-\beta)g^2$ | $w \in \mathbb{R}^d$ | Unconstrained | `shared_optimization/algorithms/adaptive_optimizers.py` | Respective project runners | `test_energy.py`, `test_maintenance.py` | Loss logs & threshold trade-off curves |
| **14** | Adam | **EnergyOpt** & **PredictMaint** | 1st & 2nd moment bias correction $\hat{m}/(\sqrt{\hat{v}}+\epsilon)$ | $w \in \mathbb{R}^d$ | Unconstrained | `shared_optimization/algorithms/adaptive_optimizers.py` | Respective project runners | `test_energy.py`, `test_maintenance.py` | Confusion matrices & metric summaries |
| **15** | Subgradient Method (Functions) | **SparseModel** | Diminishing step size $\alpha_k = a/\sqrt{k}$ on $\|x\|_2$ | $x \in \mathbb{R}^5$ | Nonsmooth | `shared_optimization/algorithms/subgradient.py` | `SparseModel/experiments/run_experiments.py` | `test_sparse.py` | `general_subgradient_norm.csv`, `subgradient_convergence.png` |
| **16** | Subgradient for Lasso Loss | **SparseModel** | $\frac{1}{2n}\|Xw - y\|_2^2 + \lambda \|w\|_1$ with subdiff at 0 | $w \in \mathbb{R}^8$ | Nonsmooth ($L_1$) | `shared_optimization/algorithms/subgradient.py` | `SparseModel/experiments/run_experiments.py` | `test_sparse.py` | `lasso_regularization_path.png`, `coefficient_comparison.png` |
| **17** | Newton's Method | **NonlinearLab** | Curvature step $p = -H^{-1}\nabla f$ via Gaussian elim | $x \in \mathbb{R}^2$ | Unconstrained | `shared_optimization/algorithms/newton.py` | `NonlinearLab/experiments/run_experiments.py` | `test_nonlinear.py` | `rosen_newton_method.csv` |
| **18** | Damped Newton's Method | **NonlinearLab** | Newton direction + Armijo backtracking line search | $x \in \mathbb{R}^2$ | Unconstrained | `shared_optimization/algorithms/newton.py` | `NonlinearLab/experiments/run_experiments.py` | `test_nonlinear.py` | `rosen_damped_newton.csv` |
| **19** | Newton One-Step Demonstration | **NonlinearLab** | Proof & numerical verification on convex quadratics | $x \in \mathbb{R}^2$ | Unconstrained (Quadratic) | `shared_optimization/algorithms/newton.py` | `NonlinearLab/experiments/run_experiments.py` | `test_nonlinear.py` | Theorem verification log (`nonlinear_lab_summary.txt`) |
| **20** | BFGS Quasi-Newton | **NonlinearLab** | Inverse Hessian rank-2 update without inversion | $x \in \mathbb{R}^2$ | Unconstrained | `shared_optimization/algorithms/bfgs.py` | `NonlinearLab/experiments/run_experiments.py` | `test_nonlinear.py` | `rosen_bfgs.csv`, `rosenbrock_trajectories.png` |
| **21** | Lagrange Multiplier Solver | **EnergyOpt** & **ResourceOpt** | Direct block KKT solver for equality programs | $x \in \mathbb{R}^m, \lambda \in \mathbb{R}^k$ | Equalities $Ax = b$ | `shared_optimization/algorithms/constrained_optimization.py` | Respective project runners | `test_energy.py`, `test_resource.py` | `generation_dispatch.png`, `production_polytope.png` |
| **22** | KKT Conditions Evaluator | **EnergyOpt** & **ResourceOpt** | Verifies Stationarity, Primal/Dual Feas, Slackness | $x, \lambda, \mu$ | Equalities & Inequalities | `shared_optimization/algorithms/constrained_optimization.py` | Respective project runners | `test_energy.py`, `test_resource.py` | `kkt_residuals_summary.png`, diagnostic tables |
| **23** | LICQ Checker | **EnergyOpt** & **ResourceOpt** | Gram matrix determinant & rank of active constraints | Active Jacobian $J_{\mathcal{A}}$ | Active Constraints | `shared_optimization/algorithms/constrained_optimization.py` | Respective project runners | `test_energy.py`, `test_resource.py` | LICQ rank diagnostics across 5 scenarios |

---

## 3. The 5 Real-World Optimization Projects

### Project 1: EnergyOpt
- **Domain:** Electricity Demand Forecasting & Economic Generation Dispatch.
- **Demand Forecasting:** Linear regression benchmark (Batch GD, SGD, Mini-batch SGD, AdaGrad, RMSProp, Adam) predicting hourly electricity load from diurnal cycles, temperature, and lag indicators.
- **Economic Dispatch:** Continuous economic generation dispatch across 3 generation facilities minimizing quadratic costs subject to meeting predicted demand and generator capacity limits. Includes analysis of forecast error financial impact.

### Project 2: PredictMaint
- **Domain:** Industrial Equipment Breakdown Prevention from Telemetry.
- **Failure Classification:** Logistic regression failure classification using Vibration RMS, Operating Temperature, Service Hours, Differential Pressure, and Mechanical Load. Evaluates BCE loss, accuracy, precision, recall, F1, confusion matrices, 2D physical decision boundaries, and decision threshold sweeps.

### Project 3: NonlinearLab
- **Domain:** Energy System Calibration & Smooth Optimization Laboratory.
- **Smooth Benchmarks:** Unconstrained smooth optimization across an ill-conditioned quadratic benchmark ($\kappa = 10.0$) and a non-convex Rosenbrock banana valley.
- **Newton One-Step Demonstration:** Empirical demonstration and theoretical explanation of Newton's One-Step Convergence theorem on strictly convex quadratics ($x_1 = x^*$).
- **Visual Trajectories:** Overlaid 2D contour trajectory maps and semi-logarithmic convergence comparisons.

### Project 4: SparseModel
- **Domain:** Sparse Resource Usage Modeling & Parsimonious Feature Selection.
- **Lasso Subgradient Optimization:** Lasso regression optimizing the composite objective $J(w) = \frac{1}{2n}\|Xw - y\|_2^2 + \lambda \|w\|_1$ via subgradient descent. Generates full regularization paths ($\lambda \in [0.001, 0.5]$) and recovers true active feature drivers without artificial thresholding during optimization.

### Project 5: ResourceOpt
- **Domain:** Continuous Production & Resource Allocation Under Constraints.
- **Constrained Manufacturing Optimization:** Constrained quadratic manufacturing optimization balancing production targets against machine hours and storage capacity. Systematically tests 5 key operational scenarios:
  1. Feasible Interior Optimum (equality active only)
  2. Feasible Boundary Optimum (equality + machine hours binding)
  3. Infeasible Candidate Point (storage violated)
  4. Linearly Independent Active Constraints (LICQ holds, full rank)
  5. Linearly Dependent / Redundant Constraints (LICQ fails, rank deficient)

---

## 4. Running the Entire Suite

### Running Standalone Projects
Every project can be executed independently from its own directory:
```bash
# EnergyOpt
cd EnergyOpt
python main.py --all      # Run experiments
python main.py --test     # Run unit tests

# PredictMaint
cd PredictMaint
python main.py --all
python main.py --test

# NonlinearLab
cd NonlinearLab
python main.py --all
python main.py --test

# SparseModel
cd SparseModel
python main.py --all
python main.py --test

# ResourceOpt
cd ResourceOpt
python main.py --all
python main.py --test
```

### Running the Entire Laboratory from Root
From `Optimization-Laboratory/`:
```bash
# Run all 77 unit tests across shared_optimization and all 5 projects
python run_all.py --test

# Run all 5 end-to-end experiment pipelines and generate all figures & reports
python run_all.py --all

# Interactive menu
python run_all.py
```